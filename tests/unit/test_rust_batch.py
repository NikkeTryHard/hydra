"""Tensor-native batch parity: Rust planes assemble identical training batches.

Proves :func:`assemble_training_batch` (whole-batch torch ops over Rust
plane slots, zero per-row Python) reproduces the real training input —
:func:`encode_observation_rows` over SIM-expanded dict rows —
element-for-element on shared rows, and that model forward logits are
bitwise identical downstream. Any divergence fails loud here — never as
silent training-label drift.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "tools" / "hydra2-replay-rs"

pytestmark = pytest.mark.serial


@pytest.fixture(scope="session")
def rust_extension() -> object:
    """Build the cdylib once and expose ``import hydra2_replay_rs``."""
    env = dict(os.environ)
    proc = subprocess.run(
        ["cargo", "build", "--offline", "-p", "hydra2-replay-rs"],
        cwd=CRATE,
        env=env,
        capture_output=True,
    )
    assert proc.returncode == 0, f"cargo build failed:\n{proc.stderr[-4000:]}"
    built = CRATE / "target" / "debug" / "libhydra2_replay_rs.so"
    assert built.is_file(), f"expected cdylib at {built}"
    ext_dir = Path(tempfile.mkdtemp(prefix="hydra2_replay_rs"))
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shutil.copy(built, ext_dir / f"hydra2_replay_rs{suffix}")
    sys.path.insert(0, str(ext_dir))
    try:
        yield importlib.import_module("hydra2_replay_rs")
    finally:
        sys.path.remove(str(ext_dir))


def _golden_tmpdir(*, walled: bool) -> Path:
    """Hermetic corpus dir with the golden 5-row game (aka-rich).

    Walled text embeds the identity wall so the Rust feed walks unfold
    (true takes), matching :func:`expand_game`; wall-less text walks fold,
    matching SIM replay.
    """
    from tests.unit.test_replay_expand_wp14 import _golden_events

    events = _golden_events()
    if walled:
        head = dict(events[0])
        head["wall"] = list(range(136))
        events = [head, *events[1:]]
    text = "\n".join(json.dumps(e) for e in events) + "\n"
    out = Path(tempfile.mkdtemp(prefix="hydra2-rust-batch"))
    (out / "golden.mjai.json.zst").write_bytes(zstd.ZstdCompressor(level=1).compress(text.encode()))
    return out


def _both_batches(*, walled: bool) -> tuple[dict, dict]:
    """Python training input vs Rust-assembled input on the golden rows."""
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.models.schema import BASELINE_ACTION_COUNT
    from hydra2.training import rust_stream
    from hydra2.training.dataset import encode_observation_rows
    from hydra2.training.rust_batch import assemble_training_batch
    from hydra2.training.stream_train import _row_to_dict

    corpus = _golden_tmpdir(walled=walled)
    manifest = build_manifest(corpus)
    game = next(iter(GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None))).game
    if walled:
        from hydra2.data.replay_expand import expand_game

        rows = expand_game(game, split="train")
    else:
        from hydra2.engines.riichienv.log_replay import replay_game

        rows = replay_game(game, split="train")
    assert len(rows) == 5
    py = encode_observation_rows([_row_to_dict(r) for r in rows], num_actions=BASELINE_ACTION_COUNT)
    with rust_stream.open_rust_plane_stream(
        [str(corpus)],
        64,
        split="train",
        source_hash="d",
        rules_hash="d",
        action_table_hash="d",
        slot_rows=64,
        t_max=32,
        depth=2,
        device="cpu",
    ) as st:
        planes, fill = st.next_into_planes()
        assert fill.rows == 5, f"expected one 5-row fill, got {fill}"
    got = assemble_training_batch(planes)
    return py, got


@pytest.mark.parametrize("walled", [False, True])
def test_assembler_matches_encoder_elementwise(rust_extension: object, walled: bool) -> None:
    _ = rust_extension
    py, got = _both_batches(walled=walled)
    assert torch.equal(got["chosen_action_id"], py["chosen_action_id"])
    assert torch.equal(got["legal_mask"], py["actor_batch"].legal_mask)
    py_feats = py["actor_batch"].features
    got_feats = got["actor_batch"].features
    assert set(got_feats.keys()) == set(py_feats.keys())
    for key in sorted(py_feats.keys()):
        want = py_feats[key]
        have = got_feats[key]
        assert have.dtype == want.dtype, f"{key}: {have.dtype} != {want.dtype}"
        if have.shape != want.shape:
            # History width only: fill t_len vs per-batch bucket ceil. Values
            # on the overlap must match; surplus on either side must be pad.
            assert key in ("history_event_kind", "history_mask"), (
                f"{key} shape {have.shape} != {want.shape}"
            )
            common = min(have.shape[1], want.shape[1])
            assert torch.equal(have[:, :common], want[:, :common]), f"{key} values drift"
            for tensor, width in ((have, have.shape[1]), (want, want.shape[1])):
                if width > common:
                    assert not bool(tensor[:, common:].any()), f"{key} surplus not pad"
        else:
            assert torch.equal(have, want), f"{key} values drift"


@pytest.mark.parametrize("walled", [False, True])
def test_assembled_batch_forward_loss_bitwise(rust_extension: object, walled: bool) -> None:
    _ = rust_extension
    py, got = _both_batches(walled=walled)
    from hydra2.models.model import Hydra2BaselineModel
    from hydra2.training.adapters import model_output_to_loss_dict
    from hydra2.training.objectives import compute_supervised_loss

    model = Hydra2BaselineModel().eval()
    weights = {"w_policy": 1.0}
    with torch.no_grad():
        want_out = model_output_to_loss_dict(model(py["actor_batch"]))
        have_out = model_output_to_loss_dict(model(got["actor_batch"]))
    assert torch.equal(have_out["policy_logits"], want_out["policy_logits"])
    want_loss = compute_supervised_loss(want_out, py, weights)
    have_loss = compute_supervised_loss(have_out, got, weights)
    assert torch.equal(have_loss["total"], want_loss["total"])
    assert torch.equal(have_loss["policy"], want_loss["policy"])


def _game_pull_vs_python(*, walled: bool) -> tuple[list[dict], list[dict]]:
    """Game-pull slim rows vs python oracle rows on the golden 5-row game."""
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.training.stream_train import _expand_game_planes, _expand_game_rows, _row_to_dict

    corpus = _golden_tmpdir(walled=walled)
    manifest = build_manifest(corpus)
    game = next(iter(GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None))).game
    actor_rows, _ = _expand_game_rows(game, "train", "python")
    py = [_row_to_dict(r) for r in actor_rows]
    rust, _ = _expand_game_planes(game, "train")
    assert len(rust) == len(py) == 5
    return rust, py


@pytest.mark.parametrize("walled", [False, True])
def test_game_pull_rows_match_python(rust_extension: object, walled: bool) -> None:
    _ = rust_extension
    rust, py = _game_pull_vs_python(walled=walled)
    for slim, full in zip(rust, py, strict=True):
        assert slim["decision_id"] == full["decision_id"]
        assert slim["chosen_action_id"] == full["chosen_action_id"]
        assert slim["action_kind"] == full["action_kind"]
        for key, value in full.items():
            if key in ("actor_observation", "observation_hash"):
                # Encoder-only inputs: the object the validating parse
                # fast-path keys on (dataset.py); slim rows assemble
                # straight from planes and never enter that path.
                continue
            assert slim[key] == value, f"{key} drift"
        from hydra2.training.rust_stream import _HOT_PLANES

        # Walk-output names: the pinned-slot layout with the packed legal
        # plane swapped for its unpacked wire form (legal_ids/legal_len).
        want_names = {name for name, _, _ in _HOT_PLANES} - {"legal_packed"}
        want_names |= {"legal_ids", "legal_len"}
        assert set(slim["_planes"].keys()) == want_names


@pytest.mark.parametrize("walled", [False, True])
def test_game_pull_batch_matches_python(rust_extension: object, walled: bool) -> None:
    _ = rust_extension
    from hydra2.models.schema import BASELINE_ACTION_COUNT
    from hydra2.training.dataset import encode_observation_rows
    from hydra2.training.rust_batch import assemble_slim_batch

    rust, py = _game_pull_vs_python(walled=walled)
    want = encode_observation_rows(py, num_actions=BASELINE_ACTION_COUNT)
    have = assemble_slim_batch(rust, action_count=BASELINE_ACTION_COUNT)
    assert torch.equal(have["chosen_action_id"], want["chosen_action_id"])
    assert torch.equal(have["legal_mask"], want["actor_batch"].legal_mask)
    assert have["_decision_ids"] == [str(r["decision_id"]) for r in py]
    assert have["_action_kinds"] == [str(r["action_kind"]) for r in py]
    for key in sorted(want["actor_batch"].features.keys()):
        assert torch.equal(have["actor_batch"].features[key], want["actor_batch"].features[key])


def test_dataset_game_pull_sequence_restore_parity(rust_extension: object) -> None:
    _ = rust_extension
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.models.schema import BASELINE_ACTION_COUNT
    from hydra2.training import stream_train as driver

    corpus = _golden_tmpdir(walled=False)
    manifest = build_manifest(corpus)

    def _factory() -> GameStream:
        return GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None)

    def _make(backend: str) -> driver._StreamDataset:
        return driver._StreamDataset(
            stream_factory=_factory,
            num_actions=BASELINE_ACTION_COUNT,
            feature_dim=64,
            seed=7,
            drop_last=True,
            need_privileged=False,
            replay_backend=backend,
        )

    rust_ds, py_ds = _make("rust"), _make("python")
    assert rust_ds._pull_game() and py_ds._pull_game()
    assert [(r["decision_id"], r["chosen_action_id"]) for r in rust_ds._rows] == [
        (r["decision_id"], r["chosen_action_id"]) for r in py_ds._rows
    ]
    snap = rust_ds.buffer_snapshot()
    fresh = _make("rust")
    fresh.restore_buffer(snap)
    have, want = rust_ds.next_batch(5), py_ds.next_batch(5)
    assert torch.equal(have["chosen_action_id"], want["chosen_action_id"])
    assert torch.equal(have["legal_mask"], want["legal_mask"])
    assert have["_decision_ids"] == want["_decision_ids"]
    assert have["_action_kinds"] == want["_action_kinds"]
    assert set(have["actor_batch"].features.keys()) == set(want["actor_batch"].features.keys())
    for key in sorted(want["actor_batch"].features.keys()):
        assert torch.equal(have["actor_batch"].features[key], want["actor_batch"].features[key])
    # Restored buffer replays the identical microbatch (re-expand + hash).
    revived = fresh.next_batch(5)
    assert torch.equal(revived["chosen_action_id"], have["chosen_action_id"])
    assert revived["_decision_ids"] == have["_decision_ids"]
    rust_ds.close()
    py_ds.close()
    fresh.close()


@pytest.mark.parametrize("walled", [False, True])
def test_expand_raw_matches_rebuilt(rust_extension: object, tmp_path: Path, walled: bool) -> None:
    """Raw-bytes expansion is identical to the legacy re-serialization path.

    The Rust walk must see parse-identical events whether Python re-dumps
    every event (legacy) or hands over the verbatim framed bytes with the
    wall spliced in Rust (hot path): chosen ids, kinds, decision ids, row
    offsets, history widths, and the shared plane blobs all match, on
    wall-less and walled games.
    """
    _ = rust_extension
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.training import stream_train as driver
    from tests.unit.test_parallel_expand_wp14 import _good_events, _write_events

    wall = list(range(136)) if walled else None
    _write_events(tmp_path / "game.mjai.json.zst", _good_events("raw-parity", wall=wall))
    manifest = build_manifest(tmp_path)
    streamed = next(iter(GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None)))
    assert len(streamed.raw) > 0 and streamed.raw.endswith(b"\n")
    have, have_sim = driver._expand_game_planes(streamed.game, streamed.split, streamed.raw)
    want, want_sim = driver._expand_game_planes(streamed.game, streamed.split, b"")
    assert have_sim == want_sim
    assert len(have) == len(want) > 0
    assert [r["decision_id"] for r in have] == [r["decision_id"] for r in want]
    assert [r["chosen_action_id"] for r in have] == [r["chosen_action_id"] for r in want]
    assert [r["action_kind"] for r in have] == [r["action_kind"] for r in want]
    for h, w in zip(have, want, strict=True):
        assert h["_row"] == w["_row"] and h["_t_len"] == w["_t_len"]
        assert h["_planes"] == w["_planes"]


@pytest.mark.parametrize("kind", ["bogus-mid", "illegal-discard"])
def test_expand_raw_quarantine_matches_rebuilt(
    rust_extension: object, tmp_path: Path, kind: str
) -> None:
    """Quarantine behavior is identical on raw and rebuilt bytes."""
    _ = rust_extension
    from hydra2.contracts.common import ContractError
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.training import stream_train as driver
    from tests.unit.test_parallel_expand_wp14 import _quarantine_events, _write_events

    _write_events(tmp_path / "bad.mjai.json.zst", _quarantine_events("raw-q", kind=kind))
    manifest = build_manifest(tmp_path)
    streamed = next(iter(GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None)))
    with pytest.raises(ContractError) as raw_exc:
        driver._expand_game_planes(streamed.game, streamed.split, streamed.raw)
    with pytest.raises(ContractError) as rebuilt_exc:
        driver._expand_game_planes(streamed.game, streamed.split, b"")
    assert str(raw_exc.value) == str(rebuilt_exc.value)


def test_batch_pull_matches_serial_pull(rust_extension: object, tmp_path: Path) -> None:
    """Batched rust pull buffers bit-identical state to the serial pull.

    One bridge call per 64-game batch must merge exactly like per-game
    expansion: row order/content, replayed/sim counters, quarantine counts
    and reason classes, and the whole-game buffer index — on walled,
    wall-less, and quarantine-bearing games.
    """
    _ = rust_extension
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.models.schema import BASELINE_ACTION_COUNT
    from hydra2.training import stream_train as driver
    from tests.unit.test_parallel_expand_wp14 import (
        _good_events,
        _quarantine_events,
        _write_events,
    )

    wall = list(range(136))
    _write_events(tmp_path / "walled.mjai.json.zst", _good_events("batch-w", wall=wall))
    _write_events(tmp_path / "sim.mjai.json.zst", _good_events("batch-s", wall=None))
    _write_events(
        tmp_path / "bogus.mjai.json.zst", _quarantine_events("batch-q1", kind="bogus-mid")
    )
    _write_events(
        tmp_path / "desync.mjai.json.zst", _quarantine_events("batch-q2", kind="illegal-discard")
    )
    manifest = build_manifest(tmp_path)

    def _factory() -> GameStream:
        return GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None)

    def _make() -> driver._StreamDataset:
        return driver._StreamDataset(
            stream_factory=_factory,
            num_actions=BASELINE_ACTION_COUNT,
            feature_dim=64,
            seed=7,
            drop_last=True,
            need_privileged=False,
            replay_backend="rust",
        )

    serial, batched = _make(), _make()
    try:
        while serial._pull_game():
            pass
        while batched._pull_game_batch(3):
            pass
        assert [(r["decision_id"], r["chosen_action_id"]) for r in serial._rows] == [
            (r["decision_id"], r["chosen_action_id"]) for r in batched._rows
        ]
        for s_row, b_row in zip(serial._rows, batched._rows, strict=True):
            assert s_row["action_kind"] == b_row["action_kind"]
            assert s_row["_row"] == b_row["_row"] and s_row["_t_len"] == b_row["_t_len"]
            assert s_row["_planes"] == b_row["_planes"]
        assert (serial.replayed, serial.sim_replayed) == (batched.replayed, batched.sim_replayed)
        assert serial.expand_quarantined == batched.expand_quarantined == 2
        assert serial.expand_quarantine_reasons == batched.expand_quarantine_reasons
        assert serial._buffered_entries == batched._buffered_entries
        assert serial.replayed > 0 and serial.sim_replayed > 0
    finally:
        serial.close()
        batched.close()


def test_ext_freshness_gate(tmp_path: Path) -> None:
    """Stale installed extensions fail closed; ad-hoc copies skip the check."""
    import json
    import os

    from hydra2.contracts.common import ContractError
    from hydra2.training import rust_batch

    src = tmp_path / "crates"
    src.mkdir()
    (src / "a.rs").write_text("x")

    def _so(name: str) -> str:
        path = tmp_path / name
        path.write_bytes(b"fake")
        return str(path)

    # No sidecar (tests/harness stage bare copies): allowed.
    rust_batch._check_fresh(_so("bare.so"))
    # Fresh sidecar: allowed.
    mt = max(p.stat().st_mtime for p in src.rglob("*") if p.is_file())
    (tmp_path / "hydra2_replay_rs.build.json").write_text(
        json.dumps({"source_root": str(src), "source_mtime": mt})
    )
    rust_batch._check_fresh(_so("fresh.so"))
    # Sources newer than the recorded build: fail closed.
    old = mt - 100.0
    (tmp_path / "hydra2_replay_rs.build.json").write_text(
        json.dumps({"source_root": str(src), "source_mtime": old})
    )
    os.utime(src / "a.rs", (old + 50.0, old + 50.0))
    with pytest.raises(ContractError, match="stale"):
        rust_batch._check_fresh(_so("stale.so"))
    # Missing source tree (source-less deploy): nothing to compare, allowed.
    (tmp_path / "hydra2_replay_rs.build.json").write_text(
        json.dumps({"source_root": str(tmp_path / "gone"), "source_mtime": old})
    )
    rust_batch._check_fresh(_so("noroot.so"))


def test_multi_group_gather_matches_python(rust_extension: object) -> None:
    """Multi-group assembly matches the python encoder element-for-element.

    Alternating rows from two games force one slice group per row (worst
    case for the group-concat path). Assembly output is freshly allocated
    per call (no cross-call buffers: shared mutable assembly state was
    removed after it corrupted multi-batch eval reads), so repeated
    assembly of the same rows is trivially stable; the load-bearing
    assertion here is byte equality with the python encoder path.
    """
    _ = rust_extension
    from hydra2.data.stream import GameStream, build_manifest
    from hydra2.models.schema import BASELINE_ACTION_COUNT
    from hydra2.training.dataset import encode_observation_rows
    from hydra2.training.rust_batch import assemble_slim_batch
    from hydra2.training.stream_train import _expand_game_planes, _expand_game_rows, _row_to_dict

    def _both(walled: bool) -> tuple[list[dict], list[dict]]:
        corpus = _golden_tmpdir(walled=walled)
        manifest = build_manifest(corpus)
        streamed = next(iter(GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None)))
        actor_rows, _ = _expand_game_rows(streamed.game, "train", "python")
        py = [_row_to_dict(r) for r in actor_rows]
        rust, _ = _expand_game_planes(streamed.game, "train", streamed.raw)
        assert len(rust) == len(py) == 5
        return rust, py

    rust_w, py_w = _both(walled=True)
    rust_s, py_s = _both(walled=False)
    rust_mixed = [row for pair in zip(rust_w, rust_s, strict=True) for row in pair]
    py_mixed = [row for pair in zip(py_w, py_s, strict=True) for row in pair]
    want = encode_observation_rows(py_mixed, num_actions=BASELINE_ACTION_COUNT)

    for _ in range(3):
        have = assemble_slim_batch(rust_mixed, action_count=BASELINE_ACTION_COUNT)
        for key in sorted(want["actor_batch"].features.keys()):
            assert torch.equal(have["actor_batch"].features[key], want["actor_batch"].features[key])
        assert torch.equal(have["chosen_action_id"], want["chosen_action_id"])
        assert torch.equal(have["legal_mask"], want["actor_batch"].legal_mask)
        assert have["_decision_ids"] == [str(r["decision_id"]) for r in py_mixed]
        assert have["_action_kinds"] == [str(r["action_kind"]) for r in py_mixed]
