"""WP-11 gate: native artifact seal/validate/order/export behavior."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest
import zstandard as zstd

pytestmark = pytest.mark.contract_package("WP-11")

ACTION_WIDTH = 6792

_BEHAVIOR = "sha256:" + "b" * 64
_TABLE = "sha256:" + "c" * 64


def _f32(value: float) -> float:
    """Exact f32 round-trip (the cross-language JCS lane carries these)."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _jcs_minimal(document: object) -> bytes:
    """Independent JCS for ASCII-only fixtures (no serde_jcs on this path).

    ``json.dumps`` with sorted keys and minimal separators is byte-identical
    to JCS when strings need no escaping and numbers are short-form — with
    two CPython spelling gaps closed below: padded exponents (``1e-08`` for
    JCS ``1e-8``) and integral floats (``10.0`` for JCS ``10``). Both passes
    are lexical and exact on these fixtures (no string holds an exponent or
    a trailing-``.0``-before-delimiter spelling). Negative zero is out of
    scope (no fixture produces ``-0.0``); it would fail loudly, never
    silently. This keeps the digest cross-checks independent of the native
    printer under test.
    """
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    raw = re.sub(r"e([+-])0+([0-9])", r"e\1\2", raw)
    return re.sub(r"\.0+(?=[,}\]:])", "", raw).encode("utf-8")


def _header(behavior: str = _BEHAVIOR) -> dict:
    return {
        "rules_hash": "sha256:" + "c" * 64,
        "utility_manifest_hash": "sha256:" + "c" * 64,
        "observation_schema_hash": "sha256:" + "c" * 64,
        "action_table_hash": _TABLE,
        "behavior_policy_checkpoint_hash": behavior,
        "behavior_model_spec_hash": "sha256:" + "c" * 64,
        "source_wall_ledger_hash": "sha256:" + "c" * 64,
        "row_derivation_hash": "sha256:" + "c" * 64,
    }


def _obs(game: str, dec: str, actor: int, legal: tuple[int, ...] = (3, 9)) -> dict:
    mask = [False] * ACTION_WIDTH
    for action in legal:
        mask[action] = True
    obs = {
        "game_id": game,
        "decision_id": dec,
        "actor": actor,
        "action_table_hash": _TABLE,
        "legal_mask": mask,
        "round_index": 0,
        "visible_history": [],
    }
    identity = dict(obs)
    obs["observation_hash"] = "sha256:" + hashlib.sha256(_jcs_minimal(identity)).hexdigest()
    return obs


def _row(
    wall: str,
    game: str,
    step: int,
    actor: int,
    dec: str,
    *,
    advantage: float,
    g4: tuple[float, float, float, float] = (10.0, 20.0, -15.0, -15.0),
    selected: int = 3,
    legal: tuple[int, ...] = (3, 9),
    behavior: str = _BEHAVIOR,
) -> dict:
    obs = _obs(game, dec, actor, legal)
    return {
        "wall_id": wall,
        "trajectory_step": step,
        "game_id": game,
        "round_id": "round-0",
        "decision_id": dec,
        "actor": actor,
        "observation_hash": obs["observation_hash"],
        "actor_observation": obs,
        "action_table_hash": _TABLE,
        "legal_mask": list(obs["legal_mask"]),
        "selected_action_id": selected,
        "behavior_policy_checkpoint_hash": behavior,
        "old_selected_log_probability": _f32(-0.5),
        "raw_advantage": advantage,
        "return_vector": list(g4),
        "bc_reference_checkpoint_hash": None,
        "bc_reference_logits": None,
        "transition_provenance_hash": "sha256:" + "d" * 64,
    }


def _entry(wall: str, game: str, step: int, actor: int, dec: str, v: float) -> dict:
    # Frozen recipe: A = f32(G[actor]) - v with G = (10, 20, -15, -15).
    g = (10.0, 20.0, -15.0, -15.0)
    advantage = float(_f32(g[actor]) - _f32(v))
    return {"doc": _row(wall, game, step, actor, dec, advantage=advantage), "v": _f32(v)}


def _two_game_tape() -> list[dict]:
    return [
        {
            "wall_id": "wall-1",
            "game_id": "game-a",
            "rows": [
                _entry("wall-1", "game-a", 0, 0, "dec-0", 8.0),
                _entry("wall-1", "game-a", 1, 1, "dec-1", 8.0),
            ],
        },
        {
            "wall_id": "wall-2",
            "game_id": "game-b",
            "rows": [_entry("wall-2", "game-b", 0, 1, "dec-2", 8.0)],
        },
    ]


def _seal_paths(tmp_path: Path, tag: str) -> tuple[str, str]:
    return str(tmp_path / f"{tag}.zst"), str(tmp_path / f"{tag}.metas.json")


def _bridge():
    from hydra2.training import rl_bridge

    return rl_bridge


def test_released_tape_cannot_serve_terminal_inputs(rust_extension: object, tmp_path: Path) -> None:
    native = rust_extension.rl_artifact
    bridge = _bridge()
    handle = bridge.stage_test_tape(_two_game_tape())
    artifact, metas = _seal_paths(tmp_path, "released")
    receipt = bridge.seal_sync(handle, _header(), ["wall-1", "wall-2"], [], False, artifact, metas)
    assert receipt.row_count == 3
    native.drop_tape(handle)
    with pytest.raises(ValueError, match="unknown"):
        native.terminal_inputs(handle)
    with pytest.raises(ValueError, match="unknown"):
        native.drop_tape(handle)


def test_seal_completion_independent_digest(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    art_a, metas_a = _seal_paths(tmp_path, "a")
    art_b, metas_b = _seal_paths(tmp_path, "b")
    handle = bridge.stage_test_tape(_two_game_tape())
    job = bridge.seal_begin(
        handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art_a, metas_a
    )
    assert isinstance(job, int) and job != 0
    # Poll at least once (status is running or already done; never an error).
    status = bridge.seal_poll(job)
    assert status["status"] in ("running", "done")
    receipt_async = bridge.seal_wait(job)
    receipt_sync = bridge.seal_sync(
        handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art_b, metas_b
    )
    assert receipt_async.artifact_digest == receipt_sync.artifact_digest
    assert receipt_async.row_count == 3 == receipt_sync.row_count


def test_decompressed_bytes_match_independent_jcs(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    art, metas = _seal_paths(tmp_path, "gate")
    handle = bridge.stage_test_tape(_two_game_tape())
    receipt = bridge.seal_sync(
        handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art, metas
    )
    # Streaming decode: the seal streams fragments without a content-size
    # flag (total unknown up front), so one-shot decompress cannot size it.
    raw = zstd.ZstdDecompressor().stream_reader(Path(art).read_bytes()).read()
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == receipt.artifact_digest
    root = json.loads(raw.decode("utf-8"))
    assert set(root) == set(_header()) | {"rows", "schema_version"}
    assert len(root["rows"]) == 3
    # Independent row-id recomputation for every row (no serde_jcs here).
    for row in root["rows"]:
        row_id = row["row_id"]
        body = {key: value for key, value in row.items() if key != "row_id"}
        assert "sha256:" + hashlib.sha256(_jcs_minimal(body)).hexdigest() == row_id
    # Canonical order: (wall, game, step) sorted regardless of completion.
    keys = [(row["wall_id"], row["game_id"], row["trajectory_step"]) for row in root["rows"]]
    assert keys == sorted(keys)


def test_seal_rejects_duplicate_eval_wall_behavior_illegal(
    rust_extension: object, tmp_path: Path
) -> None:
    _ = rust_extension
    bridge = _bridge()
    # Duplicate rows (same game/step/decision content twice) fail closed at
    # staging (seal-time duplicate rejection stays covered in Rust).
    dup = _entry("wall-1", "game-a", 0, 1, "dec-dup", 8.0)
    games = [{"wall_id": "wall-1", "game_id": "game-a", "rows": [dup, dict(dup)]}]
    with pytest.raises(ValueError):
        bridge.stage_test_tape(games)
    # Final-evaluation wall in the training source.
    games_eval = [
        {
            "wall_id": "eval-9",
            "game_id": "game-e",
            "rows": [_entry("eval-9", "game-e", 0, 1, "dec-e", 8.0)],
        }
    ]
    handle_eval = bridge.stage_test_tape(games_eval)
    art_e, metas_e = _seal_paths(tmp_path, "eval")
    with pytest.raises(ValueError):
        bridge.seal_sync(handle_eval, _header(), ["eval-9"], ["eval-9"], False, art_e, metas_e)
    # Wrong behavior checkpoint header.
    handle_ok = bridge.stage_test_tape(_two_game_tape())
    art_b, metas_b = _seal_paths(tmp_path, "beh")
    with pytest.raises(ValueError):
        bridge.seal_sync(
            handle_ok,
            _header("sha256:" + "f" * 64),
            ["wall-1", "wall-2"],
            [],
            False,
            art_b,
            metas_b,
        )
    # Selected action outside the legal mask.
    bad = _entry("wall-1", "game-a", 0, 1, "dec-bad", 8.0)
    bad["doc"]["selected_action_id"] = 4
    handle_bad = bridge.stage_test_tape([{"wall_id": "wall-1", "game_id": "game-a", "rows": [bad]}])
    art_c, metas_c = _seal_paths(tmp_path, "bad")
    with pytest.raises(ValueError):
        bridge.seal_sync(handle_bad, _header(), ["wall-1"], [], False, art_c, metas_c)


def test_validate_artifact_roundtrip(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    art, metas = _seal_paths(tmp_path, "roundtrip")
    handle = bridge.stage_test_tape(_two_game_tape())
    receipt = bridge.seal_sync(
        handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art, metas
    )
    again = bridge.validate_artifact(art, metas, _header(), ["wall-1", "wall-2"], ["eval-9"], False)
    assert again.artifact_digest == receipt.artifact_digest
    assert again.row_count == 3
    # Corrupt one compressed byte: revalidation must fail, never pass.
    raw = bytearray(Path(art).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    Path(art).write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        bridge.validate_artifact(art, metas, _header(), ["wall-1", "wall-2"], ["eval-9"], False)


def _shuffle_key(**overrides: object) -> dict:
    key = {
        "master_seed_hex": "00" * 32,
        "pipeline_spec_id": "pipe-1",
        "experiment_id": "exp-1",
        "generation": 3,
        "epoch": 0,
        "replicate_id": 0,
        "attempt_id": 0,
    }
    key.update(overrides)
    return key


def _metas_with_buckets(row_count: int, buckets: list[int]) -> dict:
    rows = []
    for i in range(row_count):
        rows.append(
            {
                "row_id": f"sha256:{i + 1:064x}",
                "t": buckets[i % len(buckets)],
                "old_lp": _f32(-0.5),
                "raw_a": 1.0,
                "g4": [1.0, 2.0, 3.0, 4.0],
                "v": 0.0,
                "wall_id": "wall-1",
                "game_id": "game-a",
                "step": i,
                "gi": 0,
                "ri": i,
            }
        )
    return {
        "schema": "hydra2.rl_metas_v1",
        "artifact_digest": "sha256:" + "a" * 64,
        "header": dict(_header()),
        "row_count": row_count,
        "rows": rows,
        "compressed_bytes": 8,
        "uncompressed_bytes": 80,
    }


def test_build_order_deterministic_grouped_exact_once(
    tmp_path: Path, rust_extension: object
) -> None:
    _ = rust_extension
    bridge = _bridge()
    metas_path = tmp_path / "order.metas.json"
    metas_path.write_bytes(json.dumps(_metas_with_buckets(24, [32, 64, 128, 256])).encode())
    first, index = bridge.build_order(str(metas_path), _shuffle_key())
    second, _ = bridge.build_order(str(metas_path), _shuffle_key())
    assert first["order_digest"] == second["order_digest"]
    assert [batch["t"] for batch in first["batches"]] == [32, 64, 128, 256]
    assert [batch["count"] for batch in first["batches"]] == [6, 6, 6, 6]
    assert sum(batch["count"] for batch in first["batches"]) == 24
    assert isinstance(index, bytes) and len(index) > 0
    # Epoch scope binds the order; tails survive cap splits exactly once.
    other, _ = bridge.build_order(str(metas_path), _shuffle_key(epoch=1))
    assert other["order_digest"] != first["order_digest"]
    tail_path = tmp_path / "tail.metas.json"
    tail_path.write_bytes(json.dumps(_metas_with_buckets(4100, [64])).encode())
    tail, _ = bridge.build_order(str(tail_path), _shuffle_key())
    assert [batch["count"] for batch in tail["batches"]] == [4096, 4]


def test_export_slab_facts_and_envelope(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    art, metas = _seal_paths(tmp_path, "slab")
    handle = bridge.stage_test_tape(_two_game_tape())
    receipt = bridge.seal_sync(
        handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art, metas
    )
    descriptor, index = bridge.build_order(metas, _shuffle_key())
    assert descriptor["row_count"] == 3
    spec, features, old_lp, raw_a, g4, selected, _legal, facts48, _envelope = bridge.export_slab(
        handle, index, metas, 0
    )
    assert spec.count == 3 and spec.bucket_t == 32
    assert spec.row_bytes == 7254 + 9 * 32 == 7542
    assert spec.feature_bytes == len(features)
    assert spec.fact_bytes == 3 * 48 == len(facts48)
    assert spec.artifact_digest == receipt.artifact_digest == descriptor["artifact_digest"]
    assert spec.order_digest == descriptor["order_digest"]
    assert struct.unpack("<3f", old_lp) == (_f32(-0.5),) * 3
    # Shuffled batch order: advantages assert as a multiset (A=2 once, 12 twice).
    assert sorted(struct.unpack("<3f", raw_a)) == [2.0, 12.0, 12.0]
    # G4 doubles are bit-exact canonical manifest values (4 seats per row,
    # identical across rows here, so every chunk matches in any order).
    assert len(g4) == 3 * 32
    for offset in (0, 32, 64):
        assert struct.unpack("<4d", g4[offset : offset + 32]) == (10.0, 20.0, -15.0, -15.0)
    # Stored selected ids ride the slab (GPU-return facts, never recomputed).
    assert sorted(struct.unpack("<3q", selected)) == [3, 3, 3]
    # Plane-major facts blocks decode at frozen block offsets (B=3 here).
    rows = 3
    assert struct.unpack("<3q", facts48[0 : rows * 8]) == (3,) * rows
    assert struct.unpack("<3f", facts48[rows * 8 : rows * 12]) == (_f32(-0.5),) * rows
    assert sorted(struct.unpack("<3f", facts48[rows * 12 : rows * 16])) == [2.0, 12.0, 12.0]
    for row in range(rows):
        chunk = facts48[rows * 16 + row * 32 : rows * 16 + (row + 1) * 32]
        assert struct.unpack("<4d", chunk) == (10.0, 20.0, -15.0, -15.0)
    # Registered batch envelope: independent digest + checksum verification.
    envelope = dict(spec.batch_envelope)
    assert envelope["rlo_digest"] == receipt.artifact_digest
    assert envelope["count"] == 3 and envelope["history_bucket_T"] == 32
    body = {key: value for key, value in envelope.items() if key != "digest"}
    assert "sha256:" + hashlib.sha256(_jcs_minimal(body)).hexdigest() == envelope["digest"]
    assert envelope["checksum"] == "sha256:" + hashlib.sha256(features).hexdigest()
    assert envelope["facts_digest"] == "sha256:" + hashlib.sha256(facts48).hexdigest()
    # Input layout names the plane-major planes the learner slices.
    names = [entry["name"] for entry in spec.input_layout]
    assert "legal_mask" in names and "history_mask" in names and "actor" in names


def test_fill_slab_matches_cold_and_geometry(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    feature_cap, fact_cap = bridge.slab_geometry(32, 4096)
    assert fact_cap == 4096 * 48
    assert feature_cap >= 4096 * 7542
    with pytest.raises(ValueError):
        bridge.slab_geometry(33, 4096)
    art, metas = _seal_paths(tmp_path, "fill")
    handle = bridge.stage_test_tape(_two_game_tape())
    bridge.seal_sync(handle, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art, metas)
    _descriptor, index = bridge.build_order(metas, _shuffle_key())
    # One native registration per generation: the hot fill takes handles
    # only, never index bytes or a metas path.
    replay = bridge.register_replay(index, metas)
    assert replay != 0
    try:
        features_slot = bytearray(feature_cap)
        facts_slot = bytearray(fact_cap)
        spec = bridge.fill_slab(handle, replay, 0, features_slot, facts_slot)
        assert spec.count == 3
        cold = bridge.export_slab(handle, index, metas, 0)
        assert bytes(features_slot[: spec.feature_bytes]) == cold[1]
        assert bytes(facts_slot[: spec.fact_bytes]) == cold[7]
        assert spec.artifact_digest == cold[0].artifact_digest
        # Undersized spans fail closed, never truncate.
        with pytest.raises(ValueError):
            bridge.fill_slab(handle, replay, 0, bytearray(8), bytearray(fact_cap))
        with pytest.raises(ValueError):
            bridge.fill_slab(handle, replay, 0, bytearray(feature_cap), bytearray(8))
    finally:
        bridge.drop_replay(replay)


def test_replay_handle_rejects_stale_and_wrong_source(
    rust_extension: object, tmp_path: Path
) -> None:
    _ = rust_extension
    bridge = _bridge()
    # Two sealed generations with disjoint wall ledgers (isolated sources).
    art_a, metas_a = _seal_paths(tmp_path, "replay-a")
    handle_a = bridge.stage_test_tape(_two_game_tape())
    bridge.seal_sync(handle_a, _header(), ["wall-1", "wall-2"], ["eval-9"], False, art_a, metas_a)
    _, index_a = bridge.build_order(metas_a, _shuffle_key())
    replay_a = bridge.register_replay(index_a, metas_a)
    art_b, metas_b = _seal_paths(tmp_path, "replay-b")
    games_b = _two_game_tape()
    for game in games_b:
        game["wall_id"] = "wall-3"
        for row in game["rows"]:
            row["doc"]["wall_id"] = "wall-3"
    handle_b = bridge.stage_test_tape(games_b)
    bridge.seal_sync(handle_b, _header(), ["wall-3"], ["eval-9"], False, art_b, metas_b)
    _, index_b = bridge.build_order(metas_b, _shuffle_key())
    replay_b = bridge.register_replay(index_b, metas_b)
    assert replay_a != replay_b
    try:
        feature_cap, fact_cap = bridge.slab_geometry(32, 4096)
        # Wrong-source fill (tape of B against frozen replay of A) rejects.
        with pytest.raises(ValueError):
            bridge.fill_slab(handle_b, replay_a, 0, bytearray(feature_cap), bytearray(fact_cap))
        # Unknown replay handle rejects (no implicit registration).
        with pytest.raises(ValueError):
            bridge.fill_slab(handle_a, 2**40, 0, bytearray(feature_cap), bytearray(fact_cap))
        # Unknown batch rejects against a valid registration.
        with pytest.raises(ValueError):
            bridge.fill_slab(handle_a, replay_a, 7, bytearray(feature_cap), bytearray(fact_cap))
        # Drop is terminal: double-drop and use-after-drop both reject.
        bridge.drop_replay(replay_b)
        with pytest.raises(ValueError):
            bridge.drop_replay(replay_b)
        with pytest.raises(ValueError):
            bridge.fill_slab(handle_b, replay_b, 0, bytearray(feature_cap), bytearray(fact_cap))
        # Mismatched index/metas companions fail at registration, never fill.
        with pytest.raises(ValueError):
            bridge.register_replay(index_a, metas_b)
    finally:
        bridge.drop_replay(replay_a)


def _unbound_entry(wall: str, game: str, step: int, actor: int, dec: str, v: float) -> dict:
    entry = _entry(wall, game, step, actor, dec, v)
    entry["doc"] = {
        key: value
        for key, value in entry["doc"].items()
        if key not in ("return_vector", "raw_advantage")
    }
    entry["bound"] = False
    return entry


def test_terminal_inputs_freeze_then_seal(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    games = [
        {
            "wall_id": "wall-1",
            "game_id": "game-a",
            "utilities": [10.0, 20.0, -15.0, -15.0],
            "rows": [
                _unbound_entry("wall-1", "game-a", 0, 0, "dec-t0", 8.0),
                _unbound_entry("wall-1", "game-a", 1, 1, "dec-t1", 8.0),
            ],
        }
    ]
    handle = bridge.stage_test_tape(games)
    descriptor, actors, game_index, values, utilities = bridge.terminal_inputs(handle)
    assert descriptor["row_count"] == 2 and descriptor["game_count"] == 1
    assert struct.unpack("<2q", actors) == (0, 1)
    assert struct.unpack("<2q", game_index) == (0, 0)
    assert struct.unpack("<8f", values) == (8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0)
    assert struct.unpack("<4d", utilities) == (10.0, 20.0, -15.0, -15.0)
    # Torch-boundary math in f32 (mirrors the frozen recipe exactly).
    rawa = struct.pack("<f", _f32(10.0) - _f32(8.0)) + struct.pack("<f", _f32(20.0) - _f32(8.0))
    frozen = bridge.freeze_terminal_advantages(handle, rawa)
    assert frozen["rows_frozen"] == 2
    # Exactly-once: a second freeze fails even with identical bytes.
    with pytest.raises(ValueError):
        bridge.freeze_terminal_advantages(handle, rawa)
    # Divergent values fail on a fresh handle (bit-identity gate).
    handle2 = bridge.stage_test_tape(games)
    bad = struct.pack("<f", 13.0) + struct.pack("<f", 12.0)
    with pytest.raises(ValueError):
        bridge.freeze_terminal_advantages(handle2, bad)
    with pytest.raises(ValueError):
        bridge.freeze_terminal_advantages(handle2, b"\x00" * 3)
    # Frozen tape seals and validates like a bound one.
    art, metas = _seal_paths(tmp_path, "terminal")
    receipt = bridge.seal_sync(handle, _header(), ["wall-1"], [], False, art, metas)
    assert receipt.row_count == 2
    again = bridge.validate_artifact(art, metas, _header(), ["wall-1"], [], False)

    assert again.artifact_digest == receipt.artifact_digest


def _derivation_doc() -> dict:
    return {
        "wall_recipe": "sha256:" + "aa" * 32,
        "sampling_recipe": "sha256:" + "bb" * 32,
        "return_recipe": "sha256:" + "cc" * 32,
        "shuffle_recipe": "sha256:" + "dd" * 32,
        "objective": {
            "clip_eps": 0.2,
            "eps_std": 1e-8,
            "w_value": 0.005,
            "w_bc": 0.0,
            "alpha": 0.01,
        },
        "numeric": {
            "logp": "fp32",
            "advantage": "fp32-frozen",
            "return_wire": "f64",
            "entropy": "legal-only",
        },
        "code": {
            "core_revision": "479c1faeb33d082965eef8198f63261a79c0fce3",
            "adapter_version": "2.0.0",
        },
        "model": {
            "model_spec": "sha256:" + "e" * 64,
            "action_table": "sha256:" + "c" * 64,
        },
    }


def test_verify_row_derivation_boundary(rust_extension: object) -> None:
    _ = rust_extension
    bridge = _bridge()
    doc = _derivation_doc()
    digest = "sha256:" + hashlib.sha256(_jcs_minimal(doc)).hexdigest()
    receipt = bridge.verify_row_derivation(digest, doc)
    assert receipt["row_derivation_hash"] == digest
    # Arbitrary digests never bind, even well-formed ones.
    with pytest.raises(ValueError):
        bridge.verify_row_derivation("sha256:" + "f" * 64, doc)
    # A renamed behavior lane fails against its own digest (gate, not hash).
    fork = json.loads(json.dumps(doc))
    fork["numeric"]["entropy"] = "all"
    fork_digest = "sha256:" + hashlib.sha256(_jcs_minimal(fork)).hexdigest()
    with pytest.raises(ValueError):
        bridge.verify_row_derivation(fork_digest, fork)


def _parity_tiles(seed: int) -> list[int]:
    import random as _random

    tiles = list(range(136))
    _random.Random(seed).shuffle(tiles)
    return [int(t) for t in tiles]


def _parity_profile(tiles: list[int], policy: str) -> dict:
    return {
        "pool_id": "parity-pool",
        "generation": 0,
        "policy_id": policy,
        "master_seed_hex": "ab" * 32,
        "experiment_id": "exp-test-rl-live-parity",
        "split_scope": "selection_training",
        "replicate_id": 0,
        "attempt_id": 0,
        "starting_points": 25000,
        "worker_count": 2,
        "games": [
            {
                "wall_id": "w0",
                "schedule_id": "parity-schedule",
                "schedule_digest": "sha256:" + "71" * 32,
                "schedule_tiles": tiles,
                "seat_perm": [0, 1, 2, 3],
            }
        ],
    }


def _parity_context(policy_bare: str) -> dict:
    digest = "sha256:" + policy_bare
    return {
        "rules_id": "tenhou_4p_hanchan",
        "rules_hash": digest,
        "action_table_hash": digest,
        "event_schema_hash": digest,
        "observation_schema_hash": digest,
        "packet_boundary_hash": digest,
        "adapter_version": "2.0.0",
        "rank_values": [1.0, 0.0, -0.5, -1.0],
        "behavior_ckpt": policy_bare,
    }


def _parity_descriptor(profile: dict, context: dict, pipeline: str) -> dict:
    games = [
        {"wall_id": game["wall_id"], "schedule_id": game["schedule_id"]}
        for game in profile["games"]
    ]
    return {
        "lifecycle_version": "1.0.0",
        "master_seed_hex": profile["master_seed_hex"],
        "experiment_id": profile["experiment_id"],
        "replicate_id": profile["replicate_id"],
        "attempt_id": profile["attempt_id"],
        "pipeline_spec_id": pipeline,
        "initial_checkpoint_id": profile["policy_id"],
        "adapter_version": context["adapter_version"],
        "wall_ledger_hash": "sha256:" + "cc" * 32,
        "row_derivation_hash": "sha256:" + "09" * 32,
        "cohort": {"generations": 8, "native_workers": 2},
        "rules": {
            "rules_id": context["rules_id"],
            "starting_points": profile["starting_points"],
            "rank_values": context["rank_values"],
        },
        "identities": {
            "rules_hash": context["rules_hash"],
            "action_table_hash": context["action_table_hash"],
            "event_schema_hash": context["event_schema_hash"],
            "observation_schema_hash": context["observation_schema_hash"],
            "packet_boundary_hash": context["packet_boundary_hash"],
            "utility_hash": "sha256:" + "ab" * 32,
            "model_spec_hash": "sha256:" + "ee" * 32,
        },
        "generations": [
            {"pool_id": profile["pool_id"], "games": games},
        ],
    }


def test_live_replay_matches_sealed_replay_bytewise(rust_extension: object, tmp_path: Path) -> None:
    """Direct typed registration must equal the sealed file path bytewise.

    Two equivalent pools play the same fixed cohort; the sealed flow
    (export/freeze/seal/order/register/fill) and the live flow
    (drain/freeze/direct-register/live-fill) must serve identical T/count
    geometry plus identical feature/fact bytes for every batch, matching
    the materialized cold oracle (hand-scoped, actor-visible buckets).
    Source confusion (wrong tape/policy/dropped replay, crossed fill
    APIs) rejects without partial writes.
    """
    import math as _math
    import struct as _struct

    import numpy as np

    from hydra2._native import rl as _rl
    from hydra2._native import rl_artifact as _artifact

    _ = rust_extension
    bridge = _bridge()
    policy_bare = "c0" * 32
    policy = "sha256:" + policy_bare
    pipeline = "sha256:" + "9d" * 32
    tiles = _parity_tiles(4242)
    profile = _parity_profile(tiles, policy)
    context = _parity_context(policy_bare)
    descriptor = _parity_descriptor(profile, context, pipeline)
    binding = {
        "pipeline_spec_id": pipeline,
        "generation": 0,
        "behavior_policy_id": policy,
    }
    binding_bytes = json.dumps(binding).encode()
    shuffle_key = {
        "master_seed_hex": profile["master_seed_hex"],
        "pipeline_spec_id": pipeline,
        "experiment_id": profile["experiment_id"],
        "generation": 0,
        "epoch": 0,
        "replicate_id": profile["replicate_id"],
        "attempt_id": profile["attempt_id"],
    }

    class _Done:
        def query(self) -> bool:
            return True

    def _play(pool: object) -> int:
        lanes: dict[int, tuple[object, object]] = {}
        for slot, bucket in enumerate((32, 64, 128, 256)):
            in_len, out_len = _rl.buffer_geometry(bucket, 8)
            host_in = np.zeros(in_len, dtype=np.uint8)
            host_out = np.zeros(out_len, dtype=np.uint8)
            pool.register_buffers(bucket, slot, host_in, host_out)
            lanes[slot] = (host_in, host_out)
        captured = 0
        for _ in range(20000):
            for slot in lanes:
                pool.request_ready_fill(slot)
            lease = pool.peek_ready()
            while lease is not None:
                pool.mark_h2d_issued(lease)
                pool.mark_inference_queued(lease)
                pool.mark_d2h_issued(lease)
                pool.mark_host_ready(lease, _Done(), _Done())
                rows = json.loads(pool.lease_descriptor(lease))["rows"]
                chosen: list[int] = []
                logps: list[float] = []
                for row in rows:
                    legal = sorted(row["legal"], key=lambda e: int(e["id"]))
                    assert legal
                    chosen.append(int(legal[0]["id"]))
                    logps.append(-_math.log(len(legal)))
                out = bytearray(len(chosen) * 28)
                for i, (cid, lp) in enumerate(zip(chosen, logps, strict=True)):
                    _struct.pack_into("<q", out, i * 8, cid)
                    _struct.pack_into("<f", out, 8 * len(chosen) + i * 4, lp)
                lanes[lease.slot_id][1][: len(out)] = np.frombuffer(out, dtype=np.uint8)
                captured += int(pool.commit_actions(lease))
                lease = pool.peek_ready()
            if all(entry["terminal"] for entry in json.loads(pool.completions())):
                return captured
        raise AssertionError("parity fixture did not finish the hanchan")

    def _freeze(tape: int, captured: int) -> None:
        desc, actor_raw, game_raw, value_raw, utility_raw = bridge.terminal_inputs(tape)
        assert desc["row_count"] == captured
        actors = np.frombuffer(actor_raw, dtype="<i8")
        games = np.frombuffer(game_raw, dtype="<i8")
        values = np.frombuffer(value_raw, dtype="<f4").reshape(-1, 4)
        utilities = np.frombuffer(utility_raw, dtype="<f8").reshape(-1, 4)
        advantages = (
            utilities.astype(np.float32)[games, actors] - values[np.arange(captured), actors]
        )
        assert (
            bridge.freeze_terminal_advantages(tape, advantages.tobytes())["rows_frozen"] == captured
        )

    # Sealed flow on an archive pool.
    pool_s = _rl.PyNativePool.open(json.dumps(profile), json.dumps(context), "sha256:" + "09" * 32)
    try:
        captured = _play(pool_s)
        pool_s.quiesce()
        tape_s = pool_s.export_tape()
    finally:
        pool_s.close()
    # Live flow on an equivalent live pool.
    pool_b = _rl.PyNativePool.open_live(
        json.dumps(profile), json.dumps(context), "sha256:" + "09" * 32, binding_bytes
    )
    try:
        assert _play(pool_b) == captured
        pool_b.quiesce()
        live_spec = json.loads(
            bytes(
                _rl.live_collector_spec(
                    run_descriptor_json=json.dumps(descriptor).encode(),
                    generation=0,
                    policy_id=policy,
                )
            )
        )
        tape_b = pool_b.drain_tape(binding_bytes)
    finally:
        pool_b.close()
    assert tape_s != tape_b
    _freeze(tape_s, captured)
    _freeze(tape_b, captured)

    art, metas = _seal_paths(tmp_path, "parity")
    header = {
        "rules_hash": context["rules_hash"],
        "utility_manifest_hash": "sha256:" + "ab" * 32,
        "observation_schema_hash": context["observation_schema_hash"],
        "action_table_hash": context["action_table_hash"],
        "behavior_policy_checkpoint_hash": policy,
        "behavior_model_spec_hash": "sha256:" + "ee" * 32,
        "source_wall_ledger_hash": "sha256:" + "cc" * 32,
        "row_derivation_hash": "sha256:" + "09" * 32,
    }
    bridge.seal_sync(tape_s, header, ["w0"], [], False, art, metas)
    sealed_desc, index = bridge.build_order(metas, shuffle_key)
    replay_s = bridge.register_replay(index, metas)
    live_handle, live_desc = bridge.register_live_replay(
        tape_b,
        json.dumps(live_spec).encode(),
        json.dumps(descriptor).encode(),
    )
    try:
        assert live_desc["schema"] == "hydra2.rl_live_replay/v1"
        assert "artifact_digest" not in live_desc and "order_digest" not in live_desc
        assert live_desc["row_count"] == sealed_desc["row_count"] == captured
        assert [(b["t"], b["count"]) for b in live_desc["batches"]] == [
            (b["t"], b["count"]) for b in sealed_desc["batches"]
        ]
        feature_cap, fact_cap = bridge.slab_geometry(256, 4096)
        features, facts = bytearray(feature_cap), bytearray(fact_cap)
        assert len(live_desc["batches"]) == len(sealed_desc["batches"]) > 0
        for idx, sealed_batch in enumerate(sealed_desc["batches"]):
            live_fill = bridge.fill_live_slab(tape_b, live_handle, idx, features, facts)
            live_features = bytes(features[: live_fill["feature_bytes"]])
            live_facts = bytes(facts[: live_fill["fact_bytes"]])
            sealed_spec = bridge.fill_slab(tape_s, replay_s, idx, features, facts)
            assert live_fill["t"] == sealed_spec.bucket_t == sealed_batch["t"]
            assert live_fill["count"] == sealed_spec.count == sealed_batch["count"]
            assert bytes(features[: sealed_spec.feature_bytes]) == live_features
            assert bytes(facts[: sealed_spec.fact_bytes]) == live_facts
            # Materialized cold oracle agrees (hand-scoped visible buckets).
            cold = bridge.export_slab(tape_s, index, metas, idx)
            assert cold[1] == live_features
            assert cold[7] == live_facts
        # Source confusion rejects without partial writes.
        with pytest.raises(ValueError):
            bridge.register_live_replay(
                tape_s, json.dumps(live_spec).encode(), json.dumps(descriptor).encode()
            )
        other_spec = dict(live_spec)
        other_spec["policy_id"] = "sha256:" + "ff" * 32
        with pytest.raises(ValueError):
            bridge.register_live_replay(
                tape_b, json.dumps(other_spec).encode(), json.dumps(descriptor).encode()
            )
        with pytest.raises(ValueError):
            bridge.fill_slab(tape_b, live_handle, 0, bytearray(8), bytearray(48))
        with pytest.raises(ValueError):
            bridge.fill_live_slab(tape_s, replay_s, 0, bytearray(8), bytearray(48))
        with pytest.raises(ValueError):
            bridge.fill_live_slab(tape_b, live_handle, len(live_desc["batches"]), features, facts)
    finally:
        bridge.drop_replay(replay_s)
        bridge.drop_replay(live_handle)
    _artifact.drop_tape(tape_s)
    _artifact.drop_tape(tape_b)
