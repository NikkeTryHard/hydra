"""WP-14 replay backend flag: python oracle backend + plane parity (K1 cutover).

Covers the flag surface only (no replay logic): the ``data.replay_backend``
default (``"rust"`` plane feed; ``"python"`` selects the oracle shim for parity;
the ``rust_json`` id refuses fail-closed) + strict parsing + run-digest identity,
the ``_expand_game_rows`` dispatch
parity on the python path (thin shim, byte-identical to the direct
``replay_game``/``expand_game`` calls), the plane-path chosen/quarantine
parity on synthetic golden games (``next_into_planes`` commits the oracle's
choices; string decision ids live cold-side only), the resume refusal on
digest change (run digest + dataset buffer tag), and the row-cache backend
tag. Plane-draining tests build the extension once per session (same recipe
as ``test_rust_stream_wp14``) and run in the serial lane; the rest is
lane-default CPU with fixed seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream import GameStream, assign_split, build_manifest, group_key_for_path
from hydra2.training import stream_train as driver
from hydra2.training._rc_digest import run_config_digest, run_config_to_dict
from hydra2.training._rc_root import load_run_config

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


class TestFlag:
    """Flag surface: default + strict parsing + digest identity + dispatch."""

    def test_default_backend_is_rust(self, tmp_path: Path) -> None:
        config = load_run_config(
            _write_mapping(tmp_path, "run.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        assert config.data.replay_backend == "rust"
        assert run_config_to_dict(config)["data"]["replay_backend"] == "rust"

    def test_unknown_backend_rejected(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"]["replay_backend"] = "bogus_backend"
        with pytest.raises(ContractError, match="replay_backend"):
            load_run_config(_write_mapping(tmp_path, "bad.yaml", mapping), environ={})

    def test_digest_identity(self, tmp_path: Path) -> None:
        base = load_run_config(
            _write_mapping(tmp_path, "a.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        assert base.data.replay_backend == "rust"
        # The deleted rust_json id refuses fail-closed (single backend).
        removed = _minimal_mapping(tmp_path)
        removed["data"]["replay_backend"] = "rust_json"
        with pytest.raises(ContractError, match="replay_backend"):
            load_run_config(_write_mapping(tmp_path, "b.yaml", removed), environ={})
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


@pytest.mark.serial
class TestPlaneBackend:
    def test_plane_chosen_sequence_matches_python(
        self, rust_extension: Any, tmp_path: Path
    ) -> None:
        """Plane fills commit the oracle's chosen ids in order.

        K1 cutover: DecisionRow parity moved to the thin plane bridge. The
        golden wall-less game is framed verbatim to a scratch dir and
        drained via ``next_into_planes``; committed ``chosen_action_id``
        must equal the python oracle's choices. String decision ids live
        cold-side only and have no plane equivalent by design.
        """
        from hydra2.engines.riichienv.log_replay import replay_game
        from hydra2.training import rust_stream

        game = _decode_game("flag-plane-seq")
        py_rows = replay_game(game, split="train")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "g.jsonl").write_text(
            "\n".join(json.dumps(e) for e in _golden_events("flag-plane-seq")) + "\n"
        )
        pins = {"source_hash": "s", "rules_hash": "r", "action_table_hash": "a"}
        chosen: list[int] = []
        with rust_stream.open_rust_plane_stream(
            [str(inputs)], 64, split="train", slot_rows=64, t_max=32, depth=2, device="cpu", **pins
        ) as stream:
            while True:
                batch, fill = stream.next_into_planes()
                if fill.rows:
                    chosen.extend(int(v) for v in batch["chosen_action_id"].tolist())
                if fill.rows == 0 and fill.games_consumed == 0:
                    break
        assert chosen == [r.chosen_action_id for r in py_rows]

    def test_wall_bound_matches_expand(self) -> None:
        from hydra2.data.replay_expand import expand_game

        game = _decode_game("flag-rust-walled", wall=WALL)
        rows, sim_path = driver._expand_game_rows(game, "train", "python")
        direct = expand_game(game, split="train")
        assert sim_path is False
        assert [r.decision_id for r in rows] == [r.decision_id for r in direct]

    def test_plane_quarantine_closed_vocab(self, rust_extension: Any, tmp_path: Path) -> None:
        """Bogus-mid games quarantine whole-game on the plane path.

        K1 cutover: the plane bridge quarantines with closed feed reason
        codes (never a new code) while the python oracle raises
        ContractError — both reject the same game.
        """
        from hydra2.engines.riichienv.log_replay import replay_game
        from hydra2.training import rust_stream

        bad = _decode_bad_game("flag-plane-q")
        with pytest.raises(ContractError):
            replay_game(bad, split="train")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "g.jsonl").write_text(
            "\n".join(json.dumps(e) for e in _golden_events("flag-plane-q")[:-1])
            + "\n"
            + json.dumps({"type": "bogus_mid_event"})
            + "\n"
            + json.dumps(_golden_events("flag-plane-q")[-1])
            + "\n"
        )
        pins = {"source_hash": "s", "rules_hash": "r", "action_table_hash": "a"}
        with rust_stream.open_rust_plane_stream(
            [str(inputs)], 64, split="train", slot_rows=64, t_max=32, depth=2, device="cpu", **pins
        ) as stream:
            while True:
                _, fill = stream.next_into_planes()
                if fill.rows == 0 and fill.games_consumed == 0:
                    break
            stats = stream.stats()
            quars = stream.quarantines()
        assert stats.rows_out == 0
        assert stats.games_quarantined == 1
        assert [q.game_id for q in quars] == ["g"]
        assert [q.reason_code for q in quars] == ["unknown-event"]

    def test_missing_extension_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from hydra2.training import rust_stream

        monkeypatch.setitem(sys.modules, "hydra2_replay_rs", None)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        with pytest.raises(RuntimeError, match="hydra2_replay_rs"):
            rust_stream.open_rust_plane_stream(
                [str(inputs)],
                2,
                split="train",
                source_hash="s",
                rules_hash="r",
                action_table_hash="a",
            )

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

        # K1 cutover: single python backend (the rust_json id is deleted).
        ds = driver._StreamDataset(
            stream_factory=lambda: _factory("train"),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            replay_backend="python",
        )
        while ds._pull_game():
            pass

        py_rows = [(r["decision_id"], r["chosen_action_id"]) for r in ds._rows]
        assert len(py_rows) > 0
        assert ds.replayed == 1
        assert ds.sim_replayed == 1

        # Row-exact resume: snapshot restores verbatim.
        snap = ds.buffer_snapshot()
        fresh = driver._StreamDataset(
            stream_factory=lambda: _factory("train"),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=False,
            replay_backend="python",
        )
        fresh.restore_buffer(snap)
        assert [r["decision_id"] for r in fresh._rows] == [r["decision_id"] for r in ds._rows]

        # Backend-tag refusal: a forged tag fails closed on replay_backend.
        forged = dict(snap)
        forged["replay_backend"] = "rust_json"
        with pytest.raises(ContractError, match="replay_backend"):
            fresh.restore_buffer(forged)


@pytest.mark.serial
@pytest.mark.slow
class TestResumeRefusal:
    def test_resume_refuses_digest_change(self, tmp_path: Path) -> None:
        from hydra2.training._rc_resume import resolve_resume_plan
        from hydra2.training._rc_root import load_run_config
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

        # K1 cutover: the rust_json backend id is gone, so the digest-change
        # axis here is max_updates (same run id, altered config).
        changed_config = load_run_config(
            _write_run_yaml(
                tmp_path,
                corpus_root=tmp_path / "corpus",
                artifact_root=tmp_path / "artifacts",
                max_updates=2,
                run_id=run_id,
                backend="python",
            ),
            environ={},
        )
        assert run_config_digest(changed_config) != run_config_digest(py_config)
        resume = resolve_resume_plan(run_dir)
        with pytest.raises(ContractError, match="run_digest"):
            run_stream_training(changed_config, resume)
