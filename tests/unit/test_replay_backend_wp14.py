"""WP-14 replay backend flag: rust_json default, python shim fallback (Slice 8).

Covers the flag surface only (no replay logic): the ``data.replay_backend``
default (rust_json since the Slice-8 cutover) + strict parsing + run-digest
identity, the ``_expand_game_rows`` dispatch parity on the python path (thin
shim over the same Rust stream since Slice 8, byte-identical to the direct
``replay_game``/``expand_game`` calls it replaces), the rust_json
decision-sequence/quarantine parity on synthetic golden games, the
cross-backend resume refusal (run digest + dataset buffer), the
rust_json+workers fail-closed gate, and the row-cache backend tag. Rust
draining tests build the Slice-4 extension once per session (same recipe as
``test_rust_stream_wp14``) and run in the serial lane; the rest is
lane-default CPU with fixed seeds.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream import GameStream, assign_split, build_manifest, group_key_for_path
from hydra2.training import stream_train as driver
from hydra2.training.run_config import load_run_config, run_config_digest, run_config_to_dict

pytestmark = [pytest.mark.contract_package("WP-14")]

DATA_SEED = 7
WALL = list(range(136))

_TEHAIS = [
    ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
    ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
    ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
    ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
]


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
            "tehais": _TEHAIS,
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "end_game", "game_id": game_id, "scores": [30000, 25000, 20000, 15000]},
    ]


def _decode_game(game_id: str, *, wall: list[int] | None = None) -> Any:
    from hydra2.data.decode import decode_game_object

    raw = "\n".join(json.dumps(e) for e in _golden_events(game_id, wall=wall)) + "\n"
    return decode_game_object(
        object_id=f"stem-{game_id}",
        packaged_object_id=f"stem-{game_id}",
        decoded_bytes=raw.encode(),
    )


def _decode_bad_game(game_id: str) -> Any:
    """Decodable but replay-rejected: unknown mid event type."""
    from hydra2.data.decode import decode_game_object

    events = [
        *_golden_events(game_id)[:-1],
        {"type": "bogus_mid_event"},
        _golden_events(game_id)[-1],
    ]
    raw = "\n".join(json.dumps(e) for e in events) + "\n"
    return decode_game_object(
        object_id=f"stem-{game_id}",
        packaged_object_id=f"stem-{game_id}",
        decoded_bytes=raw.encode(),
    )


def _split_of(stem: str) -> str:
    probe = Path("tenhou") / f"{stem}.mjai.json.zst"
    ratios = {
        "train": driver.SPLIT_RATIOS["train"],
        "validation": driver.SPLIT_RATIOS["validation"],
    }
    return assign_split(group_key=group_key_for_path(probe), seed=DATA_SEED, ratios=ratios)


def _pick_stems(*, need_train: int, need_val: int = 0) -> tuple[list[str], list[str]]:
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


def _write_game(path: Path, game_id: str, *, wall: list[int] | None) -> None:
    raw = "\n".join(json.dumps(e) for e in _golden_events(game_id, wall=wall)) + "\n"
    path.write_bytes(zstd.ZstdCompressor().compress(raw.encode()))


def _write_run_yaml(
    directory: Path,
    *,
    corpus_root: Path,
    artifact_root: Path,
    max_updates: int,
    run_id: str,
    backend: str | None = None,
) -> Path:
    data: dict[str, Any] = {"root": str(corpus_root), "num_workers": 0}
    if backend is not None:
        data["replay_backend"] = backend
    mapping: dict[str, Any] = {
        "run": {"id": run_id, "kind": "supervised", "description": "replay-backend fixture"},
        "data": data,
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
        "seeds": {"data_seed": DATA_SEED, "train_seed": 123, "selection_seed": 0},
        "output": {"artifact_root": str(artifact_root)},
    }
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


def _minimal_mapping(tmp_path: Path) -> dict[str, Any]:
    return {
        "run": {"id": "backend-flag-001", "kind": "supervised", "description": "flag fixture"},
        "data": {"root": str(tmp_path / "corpus")},
    }


def _write_mapping(tmp_path: Path, name: str, mapping: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path

    def test_default_backend_is_rust_json(self, tmp_path: Path) -> None:
        config = load_run_config(
            _write_mapping(tmp_path, "run.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        assert config.data.replay_backend == "rust_json"
        assert run_config_to_dict(config)["data"]["replay_backend"] == "rust_json"

    def test_unknown_backend_rejected(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"]["replay_backend"] = "rust"
        with pytest.raises(ContractError, match="replay_backend"):
            load_run_config(_write_mapping(tmp_path, "bad.yaml", mapping), environ={})

    def test_digest_identity(self, tmp_path: Path) -> None:
        base = load_run_config(
            _write_mapping(tmp_path, "a.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        assert base.data.replay_backend == "rust_json"
        mapping = _minimal_mapping(tmp_path)
        mapping["data"]["replay_backend"] = "python"
        shim = load_run_config(_write_mapping(tmp_path, "b.yaml", mapping), environ={})
        assert shim.data.replay_backend == "python"
        # Backend is pinned in the run identity: old checkpoints never
        # silently resume cross-backend (digest mismatch refuses fail-closed).
        assert run_config_digest(shim) != run_config_digest(base)
        again = load_run_config(
            _write_mapping(tmp_path, "c.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        assert run_config_digest(again) == run_config_digest(base)

    def test_helper_rejects_unknown_backend(self) -> None:
        game = _decode_game("flag-unknown-be", wall=None)
        with pytest.raises(ContractError, match="replay_backend"):
            driver._expand_game_rows(game, "train", "bogus")
        walled = _decode_game("flag-unknown-be-w", wall=WALL)
        with pytest.raises(ContractError, match="replay_backend"):
            driver._expand_game_rows(walled, "train", "bogus")
    def test_python_backend_matches_direct_calls(self, rust_extension: Any) -> None:
        from hydra2.data.replay_expand import expand_game
        from hydra2.engines.riichienv.log_replay import replay_game

        wall_less = _decode_game("flag-py-direct")
        rows, sim_path = driver._expand_game_rows(wall_less, "train", "python")
        direct = replay_game(wall_less, split="train")
        assert sim_path is True
        assert [r.decision_id for r in rows] == [r.decision_id for r in direct]
        assert [r.chosen_action_id for r in rows] == [r.chosen_action_id for r in direct]

        walled = _decode_game("flag-py-direct-w", wall=WALL)
        rows_w, sim_w = driver._expand_game_rows(walled, "train", "python")
        direct_w = expand_game(walled, split="train")
        assert sim_w is False
        assert [r.decision_id for r in rows_w] == [r.decision_id for r in direct_w]

    def test_classifier_unifies_backend_quote_styles(self) -> None:
        py = ContractError(
            "sim replay desync game 'g1' kyoku 0 tsumo: live draw without a live tile"
        )
        ru = ContractError(
            'sim replay desync game "g1" kyoku 0 tsumo: live draw without a live tile'
        )
        assert driver._quarantine_class(ru) == driver._quarantine_class(py)
        assert "<id>" in driver._quarantine_class(ru)
        assert "g1" not in driver._quarantine_class(ru)

    def test_rust_json_workers_fail_closed(self, tmp_path: Path) -> None:
        """Slice-8 serial first: rust_json + expand_workers>0 refuses (pool lands later)."""
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(corpus)
        with pytest.raises(ContractError, match="fail-closed"):
            driver._StreamDataset(
                stream_factory=lambda: GameStream(
                    manifest,
                    seed=DATA_SEED,
                    ratios=dict(driver.SPLIT_RATIOS),
                    epoch=0,
                    split="train",
                    shuffle_buffer=0,
                ),
                num_actions=6792,
                feature_dim=64,
                seed=DATA_SEED,
                drop_last=True,
                replay_backend="rust_json",
                expand_workers=2,
            )


@pytest.fixture(scope="session")
def rust_extension(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build the Slice-4 extension once (same recipe as test_rust_stream_wp14)."""
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


@pytest.mark.serial
class TestRustBackend:
    def test_decision_sequence_matches_python(self, rust_extension: Any) -> None:
        from hydra2.engines.riichienv.log_replay import replay_game

        game = _decode_game("flag-rust-seq")
        rust_rows, sim_path = driver._expand_game_rows(game, "train", "rust_json")
        py_rows = replay_game(game, split="train")
        assert sim_path is True
        assert [r.decision_id for r in rust_rows] == [r.decision_id for r in py_rows]
        assert [r.chosen_action_id for r in rust_rows] == [r.chosen_action_id for r in py_rows]

    def test_rust_rows_bind_minted_digests(self, rust_extension: Any) -> None:
        """Rust-fed rows carry the minted Slice-5 trio (never the echo)."""
        from hydra2.config import repo_root
        from hydra2.contracts.action import ACTION_TABLE_RELPATH, load_action_table
        from hydra2.data.replay_expand import _adapter_hash, _load_rules
        from hydra2.engines.riichienv.adapter import _rules_identity
        from hydra2.engines.riichienv.log_replay import replay_game
        from hydra2.engines.riichienv.state import rules_identity_hash

        rules = _load_rules()
        expected_rules = _rules_identity(rules, str(rules_identity_hash(rules)))
        expected_adapter = _adapter_hash()
        expected_table = str(load_action_table(repo_root() / ACTION_TABLE_RELPATH).digest)
        game = _decode_game("flag-rust-digest")
        rust_rows, sim_path = driver._expand_game_rows(game, "train", "rust_json")
        assert sim_path is True
        assert rust_rows
        for row in rust_rows:
            assert row.rules_hash == expected_rules
            assert row.adapter_hash == expected_adapter
            assert row.action_table_hash == expected_table
            assert row.rules_hash != driver._RUST_JSON_SPEC_HASH
            assert row.adapter_hash != driver._RUST_JSON_SPEC_HASH
        # Provenance parity: the python backend binds the same trio.
        py_rows = replay_game(game, split="train")
        assert [r.rules_hash for r in rust_rows] == [r.rules_hash for r in py_rows]
        assert [r.adapter_hash for r in rust_rows] == [r.adapter_hash for r in py_rows]
        assert [r.action_table_hash for r in rust_rows] == [r.action_table_hash for r in py_rows]

    def test_wall_bound_never_touches_rust(self, rust_extension: Any) -> None:
        from hydra2.data.replay_expand import expand_game

        game = _decode_game("flag-rust-walled", wall=WALL)
        rust_rows, sim_path = driver._expand_game_rows(game, "train", "rust_json")
        direct = expand_game(game, split="train")
        assert sim_path is False
        assert [r.decision_id for r in rust_rows] == [r.decision_id for r in direct]

    def test_quarantine_class_identical(self, rust_extension: Any) -> None:
        from hydra2.engines.riichienv.log_replay import replay_game

        bad = _decode_bad_game("flag-rust-q")
        with pytest.raises(ContractError) as py_exc:
            replay_game(bad, split="train")
        with pytest.raises(ContractError) as ru_exc:
            driver._expand_game_rows(bad, "train", "rust_json")
        assert driver._quarantine_class(ru_exc.value) == driver._quarantine_class(py_exc.value)

    def test_expand_backend_parity(self, rust_extension: Any) -> None:
        game = _decode_game("flag-rust-pool")
        rows_py, sim_py = driver._expand_game_rows(game, "train", "python")
        rows_ru, sim_ru = driver._expand_game_rows(game, "train", "rust_json")
        assert sim_py is True and sim_ru is True
        assert [r.decision_id for r in rows_ru] == [r.decision_id for r in rows_py]

        bad = _decode_bad_game("flag-rust-pool-q")
        with pytest.raises(ContractError) as ru_exc:
            driver._expand_game_rows(bad, "train", "rust_json")
        with pytest.raises(ContractError) as py_exc:
            driver._expand_game_rows(bad, "train", "python")
        assert driver._quarantine_class(ru_exc.value) == driver._quarantine_class(py_exc.value)

    def test_missing_extension_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "hydra2_replay_rs", None)
        game = _decode_game("flag-rust-missing")
        with pytest.raises(RuntimeError, match="hydra2_replay_rs"):
            driver._expand_game_rows(game, "train", "rust_json")

    def test_dataset_sequence_and_restore(self, rust_extension: Any, tmp_path: Path) -> None:
        train_stems, _ = _pick_stems(need_train=2)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        _write_game(corpus / f"{train_stems[0]}.mjai.json.zst", "flag-ds-sim", wall=None)
        _write_game(corpus / f"{train_stems[1]}.mjai.json.zst", "flag-ds-wall", wall=WALL)
        manifest = build_manifest(tmp_path / "corpus")

        def _factory(split: str) -> GameStream:
            return GameStream(
                manifest,
                seed=DATA_SEED,
                ratios={
                    "train": driver.SPLIT_RATIOS["train"],
                    "validation": driver.SPLIT_RATIOS["validation"],
                },
                epoch=0,
                split=split,
                shuffle_buffer=0,
            )

        datasets: dict[str, Any] = {}
        for backend in ("python", "rust_json"):
            ds = driver._StreamDataset(
                stream_factory=lambda: _factory("train"),
                num_actions=6792,
                feature_dim=64,
                seed=DATA_SEED,
                drop_last=True,
                need_privileged=False,
                replay_backend=backend,
            )
            while ds._pull_game():
                pass
            datasets[backend] = ds

        py_rows = [(r["decision_id"], r["chosen_action_id"]) for r in datasets["python"]._rows]
        ru_rows = [(r["decision_id"], r["chosen_action_id"]) for r in datasets["rust_json"]._rows]
        assert len(py_rows) > 0 and ru_rows == py_rows
        for attr in ("replayed", "sim_replayed", "expand_quarantined"):
            assert getattr(datasets["rust_json"], attr) == getattr(datasets["python"], attr)
        assert datasets["python"].replayed == 1
        assert datasets["python"].sim_replayed == 1

        # Row-exact resume per backend: snapshot restores verbatim.
        for backend in ("python", "rust_json"):
            snap = datasets[backend].buffer_snapshot()
            fresh = driver._StreamDataset(
                stream_factory=lambda: _factory("train"),
                num_actions=6792,
                feature_dim=64,
                seed=DATA_SEED,
                drop_last=True,
                need_privileged=False,
                replay_backend=backend,
            )
            fresh.restore_buffer(snap)
            assert [r["decision_id"] for r in fresh._rows] == [
                r["decision_id"] for r in datasets[backend]._rows
            ]

        # Cross-backend buffer restore refuses fail-closed.
        cross = driver._StreamDataset(
            stream_factory=lambda: _factory("train"),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            replay_backend="rust_json",
        )
        with pytest.raises(ContractError, match="replay_backend"):
            cross.restore_buffer(datasets["python"].buffer_snapshot())


@pytest.mark.serial
@pytest.mark.slow
class TestCrossBackendResume:
    def test_resume_refuses_cross_backend(self, tmp_path: Path) -> None:
        from hydra2.training.run_config import load_run_config, resolve_resume_plan
        from hydra2.training.stream_train import run_stream_training

        train_stems, _ = _pick_stems(need_train=1)
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True)
        _write_game(corpus / f"{train_stems[0]}.mjai.json.zst", "flag-resume", wall=WALL)
        run_id = "flag-cross-backend"
        py_config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=1,
                run_id=run_id,
                backend="python",
            ),
            environ={},
        )
        summary = run_stream_training(py_config, None)
        assert summary["end_update"] == 1
        run_dir = Path(summary["run_dir"])

        rust_config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
                run_id=run_id,
                backend="rust_json",
            ),
            environ={},
        )
        assert run_config_digest(rust_config) != run_config_digest(py_config)
        resume = resolve_resume_plan(run_dir)
        with pytest.raises(ContractError, match="run_digest"):
            run_stream_training(rust_config, resume)
