"""WP-14 stream training driver: corpus → train → checkpoint → resume.

Tiny synthetic MJAI corpora (hand-built golden games on ``tmp_path``) drive
:func:`run_stream_training` end to end: N=2 finite updates land, ``ckpt-*``
files plus ``metrics.jsonl``/``train.log`` rows appear, resume from an
explicit checkpoint continues with bit-identical loss histories, a shared
wall across splits raises before training, and an undecodable game is
quarantined and counted instead of breaking the run. CPU lane, fixed seeds,
no wall-clock randomness; history equality uses allclose, never exact float
equality.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream import assign_split, group_key_for_path
from hydra2.training.run_config import load_run_config, resolve_resume_plan, run_dir_for
from hydra2.training.stream_train import (
    SPLIT_RATIOS,
    _backward_pass_autocast_for,
    _needs_privileged_labels,
    run_stream_training,
)

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.slow]

RUN_ID = "wp14-stream-train-e2e"
DATA_SEED = 7
TRAIN_SEED = 123
WALL = list(range(136))

_RATIOS = {"train": SPLIT_RATIOS["train"], "validation": SPLIT_RATIOS["validation"]}


def _golden_events(game_id: str, *, wall: list[int] | None = None) -> list[dict[str, object]]:
    start: dict[str, object] = {"type": "start_game", "game_id": game_id}
    if wall is not None:
        start["wall"] = list(wall)
    return [
        start,
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
                ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
                ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
                ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5p"},
        {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "5p"},
        {"type": "dahai", "actor": 3, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "dahai", "actor": 0, "pai": "6p", "tsumogiri": True},
        {"type": "end_game", "game_id": game_id, "scores": [30000, 25000, 20000, 15000]},
    ]


def _write_game(
    path: Path, game_id: str, *, wall: list[int] | None = None, no_wall: bool = False
) -> None:
    resolved = WALL if wall is None else wall
    events = _golden_events(game_id) if no_wall else _golden_events(game_id, wall=resolved)
    raw = "\n".join(json.dumps(event) for event in events) + "\n"
    path.write_bytes(zstd.ZstdCompressor().compress(raw.encode()))


def _write_long_game(path: Path, game_id: str, *, cycles: int = 47) -> None:
    """Long wall-less game: row histories span buckets 32/64/128 (mixed).

    Wall-less games replay through the sim path (draws in log order), so
    arbitrary tsumogiri cycles expand; the tile rotation uses 20 strings
    untouched by the golden tehais (tile conservation holds: every string
    drawn at most 3 times). Framing mirrors the existing wall-less
    fixture (ryukyoku/end_kyoku/bare end_game). Odd ``cycles`` keeps the
    per-game row count odd, so batch-2 takes strand a tail row at every
    game boundary and the post-boundary window mixes buckets — grouping
    genuinely reorders (never same-bucket-locked).
    """
    events = _golden_events(game_id)
    events[0].pop("wall", None)
    head = events[:2]
    tiles = (
        "E",
        "S",
        "W",
        "N",
        "P",
        "F",
        "C",
        "1s",
        "2s",
        "3s",
        "4s",
        "6s",
        "7s",
        "8s",
        "9s",
        "5p",
        "6p",
        "7p",
        "8p",
        "9p",
    )
    mid: list[dict[str, object]] = []
    for turn in range(cycles):
        actor = turn % 4
        pai = tiles[turn % len(tiles)]
        mid.append({"type": "tsumo", "actor": actor, "pai": pai})
        mid.append({"type": "dahai", "actor": actor, "pai": pai, "tsumogiri": True})
    tail: list[dict[str, object]] = [
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    raw = "\n".join(json.dumps(event) for event in [*head, *mid, *tail]) + "\n"
    path.write_bytes(zstd.ZstdCompressor().compress(raw.encode()))


def _write_invalid(path: Path) -> None:
    path.write_bytes(zstd.ZstdCompressor().compress(b'{"type": "nope"}\n'))


def _split_of(stem: str) -> str:
    probe = Path("tenhou") / f"{stem}.mjai.json.zst"
    return assign_split(group_key=group_key_for_path(probe), seed=DATA_SEED, ratios=_RATIOS)


def _pick_stems(*, need_train: int, need_val: int) -> tuple[list[str], list[str]]:
    """Deterministic filename stems with the required split coverage."""
    train: list[str] = []
    val: list[str] = []
    for day in range(1, 32):
        stem = f"202401{day:02d}00gm-00a9-0000-{day:07d}"
        split = _split_of(stem)
        if split == "train" and len(train) < need_train:
            train.append(stem)
        elif split == "validation" and len(val) < need_val:
            val.append(stem)
        if len(train) >= need_train and len(val) >= need_val:
            return train, val
    raise AssertionError(f"no split coverage: train={train} val={val}")


def _write_run_yaml(
    directory: Path,
    *,
    corpus_root: Path,
    artifact_root: Path,
    max_updates: int,
    run_id: str = RUN_ID,
    num_workers: int = 0,
    eval_frequency: int | None = None,
    eval_batches: int | None = None,
    telemetry: dict[str, Any] | None = None,
) -> Path:
    mapping: dict[str, Any] = {
        "run": {"id": run_id, "kind": "supervised", "description": "wp14 stream-train fixture"},
        "data": {"root": str(corpus_root), "num_workers": num_workers},
        "model": {"parameters": {"dropout": 0.0}},
        "optimizer": {"id": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0},
        "scheduler": {"id": "constant", "warmup_updates": 0},
        "runtime": {
            "adapter_id": "plain_pytorch",
            "device": "cpu",
            "precision": "fp32",
            "compile_mode": "eager",
        },
        "loop": {
            "microbatch_size": 2,
            "accumulation_steps": 1,
            "gradient_clip_norm": None,
            "max_updates": max_updates,
            "checkpoint_frequency_updates": 1,
            "precision": "fp32",
            "keep_last_checkpoints": None,
        },
        "seeds": {"data_seed": DATA_SEED, "train_seed": TRAIN_SEED, "selection_seed": 0},
        "output": {"artifact_root": str(artifact_root)},
    }
    if eval_frequency is not None or eval_batches is not None:
        mapping["eval"] = {
            key: value
            for key, value in (
                ("frequency_updates", eval_frequency),
                ("num_batches", eval_batches),
            )
            if value is not None
        }
    if telemetry is not None:
        mapping["telemetry"] = dict(telemetry)
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


def _metrics_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip() != "":
            rows.append(json.loads(line))
    return rows


def _totals(history: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([float(entry["total"]) for entry in history], dtype=torch.float64)


@pytest.mark.serial
class TestEndToEnd:
    def test_finite_updates_ckpt_and_quarantine_counted(self, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"e2e-game-{index}")
        _write_invalid(corpus / "2024010100xx-bad.mjai.json.zst")
        config_path = _write_run_yaml(
            tmp_path,
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            max_updates=2,
        )
        config = load_run_config(config_path, environ={})

        summary = run_stream_training(config, None)

        assert summary["start_update"] == 0
        run_dir = Path(summary["run_dir"])
        assert summary["checkpoints"] == ["ckpt-000001.pt", "ckpt-000002.pt"]
        assert summary["train_games"] == 2
        assert summary["quarantined"] == 1
        assert not (run_dir / "checkpoints" / "best-ckpt.pt").exists()
        for name in summary["checkpoints"]:
            assert (run_dir / "checkpoints" / name).is_file()
            assert (run_dir / "checkpoints" / name).with_suffix(".json").is_file()
        rows = _metrics_rows(run_dir)
        assert [row["global_update"] for row in rows] == [1, 2]
        assert all(math.isfinite(float(row["total"])) for row in rows)
        assert (run_dir / "logs" / "train.log").read_text(encoding="utf-8").strip() != ""
        history = summary["loss_history"]
        assert len(history) == 2
        torch.testing.assert_close(_totals(history), _totals(rows))

    def test_holdout_eval_artifacts_and_train_unaffected(self, tmp_path: Path) -> None:
        """Periodic held-out eval writes eval rows without touching training."""
        train_stems, val_stems = _pick_stems(need_train=2, need_val=2)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"eval-train-{index}")
        for index, stem in enumerate(val_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"eval-val-{index}", no_wall=True)
        config_path = _write_run_yaml(
            tmp_path,
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            max_updates=2,
            eval_frequency=1,
            eval_batches=1,
        )
        config = load_run_config(config_path, environ={})

        summary = run_stream_training(config, None)

        assert summary["end_update"] == 2
        assert len(summary["loss_history"]) == 2
        run_dir = Path(summary["run_dir"])
        eval_rows = [
            json.loads(line)
            for line in (run_dir / "eval" / "eval.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() != ""
        ]
        assert [row["update"] for row in eval_rows] == [1, 2]
        for row in eval_rows:
            assert math.isfinite(float(row["masked_nll"]))
            assert 0.0 <= float(row["top1"]) <= 1.0
            assert math.isfinite(float(row["calibration_ece"]))
            assert int(row["num_eval_batches"]) == 1
        log_text = (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
        assert log_text.count("eval:update=") == 2
        assert len(summary["evals"]) == 2

    def test_resume_continues_with_identical_history(self, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=2, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"resume-game-{index}")
        config_path = _write_run_yaml(
            tmp_path,
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            max_updates=2,
        )
        config = load_run_config(config_path, environ={})

        fresh = run_stream_training(config, None)
        fresh_dir = Path(fresh["run_dir"])

        # Simulate a crash after update 1: drop the update-2 checkpoint, then
        # resume explicitly from ckpt-1 under the byte-identical config.
        (fresh_dir / "checkpoints" / "ckpt-000002.pt").unlink()
        (fresh_dir / "checkpoints" / "ckpt-000002.json").unlink()
        resume = resolve_resume_plan(fresh_dir, which=fresh_dir / "checkpoints" / "ckpt-000001.pt")
        assert resume.global_update == 1
        continued = run_stream_training(config, resume)

        assert continued["start_update"] == 1
        assert continued["end_update"] == 2
        assert continued["updates_run"] == 1
        torch.testing.assert_close(
            _totals(continued["loss_history"]), _totals(fresh["loss_history"])
        )
        rows = _metrics_rows(fresh_dir)
        assert [row["global_update"] for row in rows] == [1, 2]
        assert (fresh_dir / "checkpoints" / "ckpt-000002.pt").is_file()
        assert not (fresh_dir / "checkpoints" / "best-ckpt.pt").exists()


class TestGuards:
    def test_wall_overlap_across_splits_raises(self, tmp_path: Path) -> None:
        train_stems, val_stems = _pick_stems(need_train=1, need_val=1)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        _write_game(corpus / f"{train_stems[0]}.mjai.json.zst", "wall-game-train", wall=WALL)
        _write_game(corpus / f"{val_stems[0]}.mjai.json.zst", "wall-game-val", wall=WALL)
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
            ),
            environ={},
        )
        with pytest.raises(ContractError, match="wall"):
            run_stream_training(config, None)
        run_dir = run_dir_for(config)
        assert list(run_dir.glob("checkpoints/ckpt-*.pt")) == []

    def test_prefetch_stream_trains(self, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=1, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        _write_game(corpus / f"{train_stems[0]}.mjai.json.zst", "prefetch-game-0")
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=1,
                num_workers=2,
            ),
            environ={},
        )
        summary = run_stream_training(config, None)
        assert summary["end_update"] == 1
        assert summary["checkpoints"] == ["ckpt-000001.pt"]
        assert len(summary["loss_history"]) == 1
        assert math.isfinite(float(summary["loss_history"][0]["total"]))

    def test_invalid_game_quarantined_and_counted(self, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=1, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        _write_game(corpus / f"{train_stems[0]}.mjai.json.zst", "quar-game-0")
        _write_invalid(corpus / "2024010100xx-bad.mjai.json.zst")
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=1,
            ),
            environ={},
        )
        summary = run_stream_training(config, None)
        assert summary["train_games"] == 1
        assert summary["quarantined"] == 1
        assert summary["end_update"] == 1


class TestPrivilegedGate:
    def test_bc_only_skips_privileged_labels(self, tmp_path: Path) -> None:
        import dataclasses

        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=1,
            ),
        )
        assert _needs_privileged_labels(config) is False
        boosted = dataclasses.replace(
            config, weights=dataclasses.replace(config.weights, w_placement=0.5)
        )
        assert _needs_privileged_labels(boosted) is True
        mapped = dataclasses.replace(
            config, weights=dataclasses.replace(config.weights, w_event={"head": 0.5})
        )
        assert _needs_privileged_labels(mapped) is True


@pytest.mark.serial
class TestWallLessNoScoresTrains:
    def test_bc_only_trains_without_privileged_labels(self, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=1, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        events = [
            {"type": "start_game"},
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "dora_marker": "E",
                "honba": 0,
                "kyoku": 1,
                "kyotaku": 0,
                "oya": 0,
                "scores": [25000, 25000, 25000, 25000],
                "tehais": [
                    ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
                    ["2m", "3m", "4m", "5s", "6s", "7s", "6m", "7m", "8m", "2p", "3p", "4p", "6s"],
                    ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "S", "S"],
                    ["E", "S", "W", "N", "P", "F", "C", "9m", "1p", "4p", "6p", "3s", "7s"],
                ],
            },
            {"type": "tsumo", "actor": 0, "pai": "4p"},
            {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "9s"},
            {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
            {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        raw = "\n".join(json.dumps(event) for event in events) + "\n"
        (corpus / f"{train_stems[0]}.mjai.json.zst").write_bytes(
            zstd.ZstdCompressor().compress(raw.encode())
        )
        config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=1,
            ),
            environ={},
        )
        summary = run_stream_training(config, None)
        assert summary["end_update"] == 1
        assert summary["expand_quarantined"] == 0
        assert summary["sim_replayed"] == 1
        assert summary["privileged_labels"] == "skipped-no-auxiliary-weights"
        assert math.isfinite(float(summary["loss_history"][0]["total"]))


@pytest.mark.serial
class TestBufferCompaction:
    def _dataset(self, tmp_path: Path, monkeypatch: Any, threshold: int) -> Any:
        import hydra2.training.stream_train as driver
        from hydra2.data.stream import GameStream, build_manifest

        monkeypatch.setattr(driver, "_BUFFER_COMPACT_ROWS", threshold)
        train_stems, _ = _pick_stems(need_train=4, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"compact-game-{index}")
        manifest = build_manifest(corpus)
        return driver._StreamDataset(
            stream_factory=lambda: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=0,
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
        assert json.loads(cache.read_text(encoding="utf-8"))["version"] == 1

    def test_dataset_buffer_round_trip_verbatim(self, tmp_path: Path) -> None:
        """Dataset tail re-expansion is verbatim (rows + hash + counters)."""
        import hydra2.training.stream_train as driver
        from hydra2.data.stream import GameStream, build_manifest

        train_stems, _ = _pick_stems(need_train=4, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_game(corpus / f"{stem}.mjai.json.zst", f"roundtrip-game-{index}")
        manifest = build_manifest(corpus)

        def _factory() -> Any:
            return GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=0,
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
        """MLflow on: SQLite store + persisted run id appear; training intact."""
        mlflow = pytest.importorskip("mlflow")
        _ = mlflow
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
        assert (tmp_path / "artifacts" / "mirror" / "mlflow" / "mlruns.db").is_file()
        run_id_text = (run_dir / "mirror" / "mlflow_run_id").read_text(encoding="utf-8")
        assert run_id_text.strip() != ""

    def test_mlflow_run_id_survives_resume(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Resume appends to the stored MLflow run instead of opening a new one."""
        mlflow = pytest.importorskip("mlflow")
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
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"sqlite:///{tmp_path}/artifacts/mirror/mlflow/mlruns.db"
        )
        # Fresh run logs updates 1,2; resume re-logs update 2 into the same run.
        assert sorted(p.step for p in client.get_metric_history(first_id, "total")) == [1, 2, 2]


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

        def _never_pull() -> Any:
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
        from hydra2.data.stream import build_manifest

        train_stems, _ = _pick_stems(need_train=games, need_val=0)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(train_stems):
            _write_long_game(corpus / f"{stem}.mjai.json.zst", f"homog-long-{index}")
        return build_manifest(corpus)

    def _long_dataset(self, manifest: Any, **kwargs: Any) -> Any:
        import hydra2.training.stream_train as driver
        from hydra2.data.stream import GameStream

        # Python backend rows carry live ActorObservation histories, so the
        # visible_history key path is exercised end to end over histories
        # spanning buckets 32/64/128.
        return driver._StreamDataset(
            stream_factory=lambda: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=0,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            replay_backend="python",
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
