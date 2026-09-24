"""WP-14 stream training segments: compaction, resume, wiring, buckets.

Buffer compaction bounds memory without changing the consumed sequence;
checkpoints resume by seek with bit-identical histories and fail closed on
tampered or legacy sidecars; runtime truth tables, feed telemetry, verbose
and MLflow mirrors, and capture-split segments wire through unchanged
training; stable bucket-grouped takes stay opt-in with legacy order as the
default. CPU lane, fixed seeds; history equality uses allclose, never
exact float equality.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.training._rc_resume import resolve_resume_plan
from hydra2.training._rc_root import load_run_config
from hydra2.training.stream_train import (
    _backward_pass_autocast_for,
    run_stream_training,
)
from tests.unit.test_stream_train import (
    _RATIOS,
    DATA_SEED,
    RUN_ID,
    _pick_stems,
    _totals,
    _write_game,
    _write_long_game,
    _write_run_yaml,
)

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.slow]


@pytest.mark.serial
class TestBufferCompaction:
    def _dataset(self, tmp_path: Path, monkeypatch: Any, threshold: int) -> Any:
        import hydra2.training.stream_train as driver
        from hydra2.data.stream_iter import GameStream
        from hydra2.data.stream_manifest import build_manifest

        monkeypatch.setattr(driver, "_BUFFER_COMPACT_ROWS", threshold)
        train_stems, _ = _pick_stems(need_train=4, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"compact-game-{index}")
        manifest = build_manifest(corpus.parent)
        return driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
        )

    def test_compacted_stream_matches_uncompacted_sequence(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Compaction bounds memory without changing the consumed sequence."""
        compacted = self._dataset(tmp_path / "a", monkeypatch, 6)
        taken_compacted = [
            r["decision_id"] for _ in range(6) for r in compacted._consume_microbatch(2)
        ]
        assert compacted.get_sampler_state()["offset"] == 12
        assert compacted.get_sampler_state()["dropped"] > 0
        assert len(compacted._rows) < 40
        control = self._dataset(tmp_path / "b", monkeypatch, 10**9)
        taken_control = [r["decision_id"] for _ in range(6) for r in control._consume_microbatch(2)]
        assert taken_compacted == taken_control
        with pytest.raises(ContractError, match="compacted prefix"):
            compacted.set_sampler_state({"offset": 0, "epoch": 0})


@pytest.mark.serial
class TestSeekResume:
    def _run_fresh(self, tmp_path: Path, *, max_updates: int, run_id: str = RUN_ID) -> Any:
        train_stems, _ = _pick_stems(need_train=3, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"fast-game-{index}")
        config_path = _write_run_yaml(
            tmp_path,
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            max_updates=max_updates,
            run_id=run_id,
        )
        config = load_run_config(config_path, environ={})
        return run_stream_training(config, None), config

    def test_seek_resume_bit_identical(self, tmp_path: Path) -> None:
        """Checkpoints resume by seek (verbatim buffer + RNG + prefix) with identical history."""
        fresh, config = self._run_fresh(tmp_path, max_updates=3)
        fresh_dir = Path(fresh["run_dir"])
        # Crash after update 1: drop later checkpoints, resume from ckpt-1.
        for name in ["ckpt-000002.pt", "ckpt-000002.json", "ckpt-000003.pt", "ckpt-000003.json"]:
            (fresh_dir / "checkpoints" / name).unlink(missing_ok=True)
        sidecar = json.loads(
            (fresh_dir / "checkpoints" / "ckpt-000001.json").read_text(encoding="utf-8")
        )
        assert isinstance(sidecar.get("dataset_buffer"), dict)
        assert len(sidecar["dataset_buffer"]["entries"]) > 0
        assert isinstance(sidecar.get("shuffle"), dict)
        assert isinstance(sidecar["shuffle"].get("prefix_hashes"), dict)
        prefix_file = fresh_dir / "checkpoints" / sidecar["shuffle"]["prefix_hashes"]["file"]
        assert prefix_file.is_file()
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        continued = run_stream_training(config, resume)
        torch.testing.assert_close(
            _totals(continued["loss_history"]), _totals(fresh["loss_history"])
        )
        assert continued["end_update"] == 3
        assert (fresh_dir / "cache" / "scan-cache.json").is_file()

    def test_legacy_sidecar_without_seek_state_fails_closed(self, tmp_path: Path) -> None:
        """Pre-simplification sidecars (no dataset_buffer) refuse resume (no drain remains)."""
        fresh, config = self._run_fresh(tmp_path, max_updates=2)
        fresh_dir = Path(fresh["run_dir"])
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        sidecar_path = fresh_dir / "checkpoints" / "ckpt-000001.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar.pop("dataset_buffer", None)
        sidecar["shuffle"] = {
            **sidecar["shuffle"],
            "buffer_keys": [],
            "buffer_rng_state": {},
            "buffer_entries": [],
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        with pytest.raises(ContractError, match="lacks seek state"):
            run_stream_training(config, resume)

    def test_tampered_buffer_key_fails_closed(self, tmp_path: Path) -> None:
        """Tampered game keys never resume silently (fail-closed raise)."""
        fresh, config = self._run_fresh(tmp_path, max_updates=2)
        fresh_dir = Path(fresh["run_dir"])
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        sidecar_path = fresh_dir / "checkpoints" / "ckpt-000001.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        entries = sidecar["dataset_buffer"]["entries"]
        assert len(entries) > 0
        entries[0] = {**entries[0], "key": "sha256:" + "0" * 64}
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        with pytest.raises(ContractError):
            run_stream_training(config, resume)

    def test_corrupt_scan_cache_falls_back_to_full_scan(self, tmp_path: Path) -> None:
        """Corrupt manifest cache never breaks training (falls back to full scan)."""
        fresh, config = self._run_fresh(tmp_path, max_updates=2)
        fresh_dir = Path(fresh["run_dir"])
        cache = fresh_dir / "cache" / "scan-cache.json"
        assert cache.is_file()
        cache.write_text("{corrupt", encoding="utf-8")
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        continued = run_stream_training(config, resume)
        torch.testing.assert_close(
            _totals(continued["loss_history"]), _totals(fresh["loss_history"])
        )
        assert cache.is_file()
        assert json.loads(cache.read_text(encoding="utf-8"))["version"] == 2

    def test_dataset_buffer_round_trip_verbatim(self, tmp_path: Path) -> None:
        """Dataset tail re-expansion is verbatim (rows + hash + counters)."""
        import hydra2.training.stream_train as driver
        from hydra2.data.stream_iter import GameStream
        from hydra2.data.stream_manifest import build_manifest

        train_stems, _ = _pick_stems(need_train=4, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"roundtrip-game-{index}")
        manifest = build_manifest(corpus.parent)

        def _factory(epoch: int = 0) -> Any:
            return GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            )

        ds = driver._StreamDataset(
            stream_factory=_factory,
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
        )
        for _ in range(4):
            ds._consume_microbatch(2)
        snap = ds.buffer_snapshot()
        assert len(snap["entries"]) > 0
        ds2 = driver._StreamDataset(
            stream_factory=_factory,
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
        )
        ds2.restore_buffer(snap)
        assert [r["decision_id"] for r in ds2._rows] == [r["decision_id"] for r in ds._rows]
        assert ds2.buffered_row_hash() == ds.buffered_row_hash()
        assert ds2.get_sampler_state()["offset"] == ds.get_sampler_state()["offset"]
        bad = dict(snap)
        bad_entries = [dict(e) for e in snap["entries"]]  # type: ignore[union-attr]
        bad_entries[0] = {**bad_entries[0], "key": "sha256:" + "f" * 64}
        bad["entries"] = bad_entries
        ds3 = driver._StreamDataset(
            stream_factory=_factory,
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
        )
        with pytest.raises(ContractError):
            ds3.restore_buffer(bad)


@pytest.mark.serial
class TestRuntimeWiring:
    def test_backward_pass_autocast_truth_table(self) -> None:
        """Only compiled non-fp32 derives 'off'; all other pairs keep None."""
        assert (
            _backward_pass_autocast_for(
                precision="bf16_mixed", compile_mode="max-autotune-no-cudagraphs"
            )
            == "off"
        )
        assert _backward_pass_autocast_for(precision="bf16_mixed", compile_mode="default") == "off"
        assert _backward_pass_autocast_for(precision="bf16_mixed", compile_mode="eager") is None
        assert (
            _backward_pass_autocast_for(precision="fp32", compile_mode="max-autotune-no-cudagraphs")
            is None
        )
        assert _backward_pass_autocast_for(precision="fp32", compile_mode="eager") is None

    def test_feed_telemetry_jsonl_written(self, tmp_path: Path) -> None:
        """A finished run leaves per-microbatch feed telemetry plus one summary."""
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"telemetry-game-{index}")
        config_path = _write_run_yaml(
            tmp_path,
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            max_updates=2,
            run_id="wp14-stream-telemetry",
        )
        config = load_run_config(config_path, environ={})

        summary = run_stream_training(config, None)

        assert summary["end_update"] == 2
        rows = [
            json.loads(line)
            for line in (Path(summary["run_dir"]) / "logs" / "feed-telemetry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() != ""
        ]
        micro_rows = [row for row in rows if row["kind"] == "microbatch"]
        summary_rows = [row for row in rows if row["kind"] == "summary"]
        assert len(micro_rows) == 2
        assert len(summary_rows) == 1
        for row in micro_rows:
            for key in (
                "queue_wait_ms",
                "fetch_decode_ms",
                "h2d_ms",
                "compute_ms",
                "forward_ms",
                "loss_ms",
                "backward_ms",
            ):
                assert math.isfinite(float(row[key]))


class TestTelemetryWiring:
    def test_verbose_sampler_and_profiler_markers(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verbose on + 1 CPU capture: rows land, skip marker logged, training intact."""
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"telemetry-game-{index}")
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
                run_id="wp14-stream-verbose",
                telemetry={
                    "mlflow_enabled": False,
                    "verbose_enabled": True,
                    "verbose_interval_ms": 50,
                    "profiler_captures": 1,
                },
            ),
            environ={},
        )
        summary = run_stream_training(config, None)
        assert summary["end_update"] == 2
        run_dir = Path(summary["run_dir"])
        rows = [
            json.loads(line)
            for line in (run_dir / "logs" / "verbose-telemetry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() != ""
        ]
        assert len(rows) >= 1
        assert all(row["v"] == 1 and row["run_id"] == "wp14-stream-verbose" for row in rows)
        train_log = (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
        assert "profiler:update=000002 skipped (cpu-device)" in train_log

    def test_mlflow_quiet_mirror_writes_store(self, tmp_path: Path, monkeypatch: Any) -> None:
        """MLflow on: REST fallback record + persisted run id appear; training intact."""
        monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"mlflow-game-{index}")
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
                run_id="wp14-stream-mlflow",
                telemetry={"mlflow_enabled": True, "verbose_enabled": False},
            ),
            environ={},
        )
        summary = run_stream_training(config, None)
        assert summary["end_update"] == 2
        run_dir = Path(summary["run_dir"])
        fallback = tmp_path / "artifacts" / "mirror" / "mlflow" / "mirror.jsonl"
        assert fallback.is_file()
        ops = [
            json.loads(line)
            for line in fallback.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert ops[0]["op"] == "start_run"
        assert any(op["op"] == "log_update" for op in ops)
        run_id_text = (run_dir / "mirror" / "mlflow_run_id").read_text(encoding="utf-8")
        assert run_id_text.strip() != ""
        assert run_id_text.strip() == ops[0]["run_id"]

    def test_mlflow_run_id_survives_resume(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Resume appends to the stored MLflow run instead of opening a new one."""
        monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"resume-mlflow-{index}")
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
                run_id="wp14-stream-resume",
                telemetry={"mlflow_enabled": True, "verbose_enabled": False},
            ),
            environ={},
        )
        fresh = run_stream_training(config, None)
        fresh_dir = Path(fresh["run_dir"])
        id_file = fresh_dir / "mirror" / "mlflow_run_id"
        first_id = id_file.read_text(encoding="utf-8").strip()
        assert first_id != ""
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        continued = run_stream_training(config, resume)
        assert continued["end_update"] == 2
        assert id_file.read_text(encoding="utf-8").strip() == first_id
        fallback = tmp_path / "artifacts" / "mirror" / "mlflow" / "mirror.jsonl"
        ops = [
            json.loads(line)
            for line in fallback.read_text(encoding="utf-8").splitlines()
            if line.strip() != ""
        ]
        # Fresh run logs updates 1,2; resume re-logs update 2 into the same run.
        assert sorted(
            op["step"]
            for op in ops
            if op["op"] == "log_update" and op["run_id"] == first_id and "total" in op["metrics"]
        ) == [1, 2, 2]


class TestTrainSegment:
    def test_capture_splits_segment_cpu(self, tmp_path: Path) -> None:
        """Capture update trains exactly once; CPU path logs the skip marker."""
        from hydra2.training.stream_train import _train_segment

        calls: list[int] = []

        class _FakeLoop:
            def train(self, max_updates: int) -> None:
                calls.append(max_updates)

        train_log = tmp_path / "train.log"
        done = _train_segment(
            loop=_FakeLoop(),  # type: ignore[arg-type]
            done=0,
            step=2,
            captures={2},
            profiler_dir=tmp_path / "profiler",
            train_log=train_log,
            cuda_ok=False,
        )
        assert done == 2
        assert calls == [1, 1]
        assert "profiler:update=000002 skipped (cpu-device)" in train_log.read_text(
            encoding="utf-8"
        )

    def test_no_captures_single_call(self, tmp_path: Path) -> None:
        from hydra2.training.stream_train import _train_segment

        calls: list[int] = []

        class _FakeLoop:
            def train(self, max_updates: int) -> None:
                calls.append(max_updates)

        done = _train_segment(
            loop=_FakeLoop(),  # type: ignore[arg-type]
            done=5,
            step=3,
            captures=set(),
            profiler_dir=tmp_path / "profiler",
            train_log=tmp_path / "train.log",
            cuda_ok=False,
        )
        assert done == 8
        assert calls == [3]


class TestHomogeneousBuckets:
    """Item 9: stable bucket-grouped takes (opt-in; default is legacy order).

    CPU-only: synthetic rows plus real corpora, fixed seeds, no CUDA
    context. ``homogeneous_buckets=True`` sorts the unconsumed window by
    ``(bucket, arrival index)`` before each contiguous-prefix take; the
    default (False) preserves byte-identical legacy order. Resume and
    residency tests use long mixed-bucket games (histories span 32/64/128)
    so grouping is genuinely active — never same-bucket-locked.
    """

    _SPECS: tuple[tuple[str, int], ...] = (
        ("r0", 10),
        ("r1", 200),
        ("r2", 40),
        ("r3", 5),
        ("r4", 100),
        ("r5", 33),
        ("r6", 120),
        ("r7", 250),
    )

    def _synthetic_rows(self) -> list[dict[str, Any]]:
        # Bucket counts are multiples of the take size (2): r0/r3 -> 32,
        # r2/r5 -> 64, r4/r6 -> 128, r1/r7 -> 256, so every grouped take is
        # exactly single-bucket.
        return [
            {
                "decision_id": name,
                "chosen_action_id": 0,
                "action_kind": "unknown",
                "actor_observation": {"visible_history": [None] * length},
            }
            for name, length in self._SPECS
        ]

    def _dataset(self, **kwargs: Any) -> Any:
        import hydra2.training.stream_train as driver

        def _never_pull(epoch: int = 0) -> Any:
            raise AssertionError("grouped-take test pre-buffers rows; no pulls expected")

        return driver._StreamDataset(
            stream_factory=_never_pull,
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            **kwargs,
        )

    def test_grouped_take_single_bucket_and_default_identical(self) -> None:
        """Flag on: every take single-bucket; flag off: legacy order kept."""
        from hydra2.training.stream_train import _history_bucket_of

        # Bucket-ceil unit pins (boundary + over-cap/malformed fallbacks).
        assert _history_bucket_of({"actor_observation": {"visible_history": [None] * 32}}) == 32
        assert _history_bucket_of({"actor_observation": {"visible_history": [None] * 33}}) == 64
        assert _history_bucket_of({"actor_observation": {"visible_history": [None] * 256}}) == 256
        assert _history_bucket_of({"actor_observation": {"visible_history": [None] * 257}}) == 256
        assert _history_bucket_of({}) == 256
        assert _history_bucket_of({"actor_observation": None}) == 256

        grouped = self._dataset(homogeneous_buckets=True)
        grouped._rows = self._synthetic_rows()
        takes = [grouped._consume_microbatch(2) for _ in range(4)]
        assert [r["decision_id"] for take in takes for r in take] == [
            "r0",
            "r3",
            "r2",
            "r5",
            "r4",
            "r6",
            "r1",
            "r7",
        ]
        for take in takes:
            assert len({_history_bucket_of(row) for row in take}) == 1

        legacy = self._dataset()
        legacy._rows = self._synthetic_rows()
        legacy_takes = [legacy._consume_microbatch(2) for _ in range(4)]
        assert [r["decision_id"] for take in legacy_takes for r in take] == [
            name for name, _ in self._SPECS
        ]
        # Grouping is a permutation: no row lost, none duplicated.
        assert sorted(r["decision_id"] for take in takes for r in take) == sorted(
            r["decision_id"] for take in legacy_takes for r in take
        )
        assert grouped.get_sampler_state()["offset"] == legacy.get_sampler_state()["offset"] == 8

    def _long_manifest(self, tmp_path: Path, *, games: int = 4) -> Any:
        from hydra2.data.stream_manifest import build_manifest

        train_stems, _ = _pick_stems(need_train=games, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_long_game(corpus / f"{stem}.mjai.json.zst", f"homog-long-{index}")
        return build_manifest(corpus.parent)

    def _long_dataset(self, manifest: Any, **kwargs: Any) -> Any:
        import hydra2.training.stream_train as driver
        from hydra2.data.stream_iter import GameStream

        # Python backend rows carry live ActorObservation histories, so the
        # visible_history key path is exercised end to end over histories
        # spanning buckets 32/64/128.
        return driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            replay_backend="rust",
            **kwargs,
        )

    def test_resume_roundtrip_mixed_buckets(self, tmp_path: Path) -> None:
        """Snapshot/restore past a game boundary replays takes exactly.

        Takes run past the first game boundary (odd row counts strand a
        tail, so the window mixes buckets and grouping genuinely reorders:
        the grouped run provably differs from the pull-order control).
        The snapshot carries the grouped decision_id permutation
        (``row_order``); restore re-expands pull order, reorders to it,
        then hash-verifies — revived takes match the uninterrupted run
        bit-for-bit. Tampered permutations fail closed.
        """
        manifest = self._long_manifest(tmp_path)
        live = self._long_dataset(manifest, homogeneous_buckets=True)
        plain = self._long_dataset(manifest)
        live_takes: list[list[str]] = []
        plain_takes: list[list[str]] = []
        # Cross the first game boundary (buffered entries >= 2), then two
        # more takes so the snapshot window is regrouped-mixed, not pull
        # order. Bounded loop: valid games always pull whole batches.
        for _ in range(200):
            live_takes.append([r["decision_id"] for r in live._consume_microbatch(2)])
            plain_takes.append([r["decision_id"] for r in plain._consume_microbatch(2)])
            if len(live._buffered_entries) >= 2 and len(live_takes) >= 26:
                break
        assert len(live._buffered_entries) >= 2
        assert live_takes != plain_takes
        snap = live.buffer_snapshot()
        assert isinstance(snap.get("row_order"), list)
        assert len(snap["row_order"]) == len(live._rows)

        revived = self._long_dataset(manifest, homogeneous_buckets=True)
        revived.restore_buffer(snap)
        assert [r["decision_id"] for r in revived._rows] == [r["decision_id"] for r in live._rows]
        assert revived.buffered_row_hash() == live.buffered_row_hash()
        assert revived.get_sampler_state() == live.get_sampler_state()
        # Window covers several more takes without new pulls (revived never
        # touches its stream), so subsequent takes match exactly.
        for _ in range(5):
            assert [r["decision_id"] for r in revived._consume_microbatch(2)] == [
                r["decision_id"] for r in live._consume_microbatch(2)
            ]
        assert revived._stream is None
        assert revived.replayed == live.replayed
        assert revived.sim_replayed == live.sim_replayed

        # Tampered permutations fail closed (unknown id, dropped row, bad shape).
        import copy

        from hydra2.contracts.common import ContractError

        bad_unknown = copy.deepcopy(snap)
        bad_unknown["row_order"] = ["no-such-decision", *bad_unknown["row_order"][1:]]
        bad_short = copy.deepcopy(snap)
        bad_short["row_order"] = bad_short["row_order"][:-1]
        bad_shape = copy.deepcopy(snap)
        bad_shape["row_order"] = "not-a-list"
        for bad in (bad_unknown, bad_short, bad_shape):
            with pytest.raises(ContractError):
                self._long_dataset(manifest, homogeneous_buckets=True).restore_buffer(bad)

    def test_grouped_residency_bounded(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Grouped takes strand at most 3x the resident games of plain takes.

        Small compact bound forces whole-game reclamation on both sides;
        both runs pull the same games on the same schedule (identical
        2-row takes keep live counts in lockstep), so the comparison is
        apples-to-apples over mixed-bucket windows.
        """
        import hydra2.training.stream_train as driver

        manifest = self._long_manifest(tmp_path)
        monkeypatch.setattr(driver, "_BUFFER_COMPACT_ROWS", 6)

        def _run(flag: bool) -> Any:
            ds = self._long_dataset(manifest, homogeneous_buckets=flag)
            for _ in range(200):
                ds._consume_microbatch(2)
                if ds.replayed + ds.sim_replayed >= 3 and ds.rows_consumed_in_epoch >= 80:
                    break
            return ds

        grouped = _run(True)
        plain = _run(False)
        assert grouped.replayed + grouped.sim_replayed >= 3
        assert plain.replayed + plain.sim_replayed >= 3
        assert grouped.rows_consumed_in_epoch == plain.rows_consumed_in_epoch
        assert len(plain._buffered_entries) > 0 and len(plain._rows) > 0
        assert len(grouped._buffered_entries) <= 3 * len(plain._buffered_entries)
        assert len(grouped._rows) <= 3 * len(plain._rows)

    def test_sidecar_parse_accepts_optional_row_order(self, tmp_path: Path) -> None:
        """Checkpoint sidecar validation passes row_order through, strictly.

        Pure validation (no training): absent key (old snapshots, flag-off
        runs) restores via the legacy path; present key must be a list of
        str or the checkpoint fails closed before any state is applied.
        """
        from hydra2.contracts.common import ContractError
        from hydra2.training.stream_train import _parse_dataset_buffer_sidecar

        manifest = self._long_manifest(tmp_path)
        live = self._long_dataset(manifest, homogeneous_buckets=True)
        for _ in range(3):
            live._consume_microbatch(2)
        snap = live.buffer_snapshot()
        ckpt = tmp_path / "ckpt-000001.pt"
        parsed = _parse_dataset_buffer_sidecar(dict(snap), ckpt=ckpt)
        assert parsed["row_order"] == snap["row_order"]

        legacy = dict(snap)
        del legacy["row_order"]
        assert "row_order" not in _parse_dataset_buffer_sidecar(legacy, ckpt=ckpt)

        bad = dict(snap)
        bad["row_order"] = [1, 2, 3]
        with pytest.raises(ContractError, match="row_order"):
            _parse_dataset_buffer_sidecar(bad, ckpt=ckpt)


@pytest.mark.serial
class TestEpochWrap:
    """Multi-epoch wrap: exhaustion rolls the epoch, resume crosses it, gates hold."""

    def _corpus(self, tmp_path: Path, *, games: int, tag: str) -> Any:
        from hydra2.data.stream_manifest import build_manifest

        train_stems, _ = _pick_stems(need_train=games, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"{tag}-{index}")
        return build_manifest(corpus.parent)

    def _dataset(self, manifest: Any) -> Any:
        import hydra2.training.stream_train as driver
        from hydra2.data.stream_iter import GameStream

        return driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
        )

    def test_roll_continues_past_exhaustion_with_same_multiset(self, tmp_path: Path) -> None:
        """Past-supply takes roll the epoch and replay the corpus (no stall, no tail dup)."""
        from hydra2.training.stream_checkpoint import _epoch_seed

        manifest = self._corpus(tmp_path, games=4, tag="wrap-game")
        probe = self._dataset(manifest)
        try:
            while probe._pull_game():
                pass
            epoch0_ids = [str(row["decision_id"]) for row in probe._rows]
        finally:
            probe.close()
        assert len(epoch0_ids) > 0
        # The shuffle seed is epoch-parameterized (rotation is structural, not hoped for).
        assert _epoch_seed(data_seed=DATA_SEED, epoch=0) != _epoch_seed(
            data_seed=DATA_SEED, epoch=1
        )
        # Single-row takes are epoch-pure: a take pops exactly one row, so the
        # live epoch observed after the take is that row's epoch (a take that
        # triggers the roll fills from the new epoch before popping).
        by_epoch: dict[int, list[str]] = {}
        live = self._dataset(manifest)
        try:
            for _ in range(10**6):
                rows = live._consume_microbatch(1)
                assert len(rows) == 1
                by_epoch.setdefault(live.epoch, []).append(str(rows[0]["decision_id"]))
                if live.epoch >= 2:
                    break
            else:
                raise AssertionError("epoch never rolled twice past exhaustion")
        finally:
            live.close()
        assert live.epoch == 2
        assert by_epoch[0] == epoch0_ids
        assert by_epoch[1] == epoch0_ids

    def test_resume_from_post_roll_checkpoint(self, tmp_path: Path) -> None:
        """A checkpoint written in epoch 1 resumes to a bit-identical finish."""
        fresh, config = TestSeekResume._run_fresh(self, tmp_path, max_updates=10)
        fresh_dir = Path(fresh["run_dir"])
        epochs = {
            name: int(
                json.loads((fresh_dir / "checkpoints" / name).read_text(encoding="utf-8"))[
                    "stream_epoch"
                ]
            )
            for name in sorted((fresh_dir / "checkpoints").glob("ckpt-*.json"))
        }
        assert max(epochs.values()) >= 1, f"no post-roll checkpoint: {epochs}"
        # Crash after update 9 (epoch-1 territory): resume must finish identically.
        (fresh_dir / "checkpoints" / "ckpt-000010.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000010.json").unlink()
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000009.pt")
        continued = run_stream_training(config, resume)
        torch.testing.assert_close(
            _totals(continued["loss_history"]), _totals(fresh["loss_history"])
        )
        assert continued["end_update"] == 10
        updates = [
            int(row["global_update"])
            for line in (fresh_dir / "logs" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ]
        assert sorted(updates) == list(range(1, 11))

    def test_prefoll_short_run_is_bit_identical(self, tmp_path: Path) -> None:
        """The wrap changes nothing pre-exhaustion: two short runs agree row-for-row."""
        first, _ = TestSeekResume._run_fresh(
            self, tmp_path / "a", max_updates=3, run_id=RUN_ID + "-parity-a"
        )
        second, _ = TestSeekResume._run_fresh(
            self, tmp_path / "b", max_updates=3, run_id=RUN_ID + "-parity-b"
        )
        torch.testing.assert_close(_totals(first["loss_history"]), _totals(second["loss_history"]))
        assert first["end_update"] == second["end_update"] == 3

    def test_tampered_stream_epoch_fails_closed(self, tmp_path: Path) -> None:
        """A forged stream epoch never resumes silently (stream/cursor mismatch)."""
        fresh, config = TestSeekResume._run_fresh(self, tmp_path, max_updates=2)
        fresh_dir = Path(fresh["run_dir"])
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        sidecar_path = fresh_dir / "checkpoints" / "ckpt-000001.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert int(sidecar["stream_epoch"]) == 0
        sidecar["stream_epoch"] = 5
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        with pytest.raises(ContractError, match="mismatch"):
            run_stream_training(config, resume)

    def test_forged_shuffle_seed_fails_closed(self, tmp_path: Path) -> None:
        """A forged shuffle epoch seed never resumes silently (seed tripwire)."""
        fresh, config = TestSeekResume._run_fresh(self, tmp_path, max_updates=2)
        fresh_dir = Path(fresh["run_dir"])
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        sidecar_path = fresh_dir / "checkpoints" / "ckpt-000001.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert isinstance(sidecar.get("shuffle"), dict)
        assert isinstance(sidecar["shuffle"].get("epoch_seed"), int)
        sidecar["shuffle"]["epoch_seed"] = int(sidecar["shuffle"]["epoch_seed"]) + 999
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        with pytest.raises(ContractError, match="epoch_seed"):
            run_stream_training(config, resume)
