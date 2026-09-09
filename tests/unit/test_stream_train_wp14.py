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
import os
import shutil
import subprocess
import sys
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
    _needs_privileged_labels,
    run_stream_training,
)

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.slow]

RUN_ID = "wp14-stream-train-e2e"
DATA_SEED = 7
TRAIN_SEED = 123
WALL = list(range(136))

_RATIOS = {"train": SPLIT_RATIOS["train"], "validation": SPLIT_RATIOS["validation"]}


@pytest.fixture(scope="session", autouse=True)
def _s8_rust_extension(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """S8 cutover: all replay flows through Rust; build the extension once."""
    import importlib
    import importlib.machinery

    root = Path(__file__).resolve().parents[2]
    crate = root / "tools" / "hydra2-replay-rs"
    env = {**os.environ, "PYO3_PYTHON": sys.executable}
    proc = subprocess.run(
        ["cargo", "build", "--offline", "-p", "hydra2-replay-rs"],
        cwd=crate,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cargo build failed:\n{proc.stderr[-4000:]}"
    built = crate / "target" / "debug" / "libhydra2_replay_rs.so"
    assert built.is_file(), f"expected cdylib at {built}"
    ext_dir = tmp_path_factory.mktemp("hydra2_replay_rs")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shutil.copy(built, ext_dir / f"hydra2_replay_rs{suffix}")
    sys.path.insert(0, str(ext_dir))
    try:
        yield importlib.import_module("hydra2_replay_rs")
    finally:
        sys.path.remove(str(ext_dir))


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
