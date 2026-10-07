"""Native pool conformance: hanchan parity, plane contracts, failure gates.

Consumer-visible gates over the real `hydra2._native.rl` pool surface
(needs `pixi run build-ext` first; the import fails loud otherwise):

- complete generated hanchan native/reference parity: the native pool
  plays a full hanchan with a first-legal policy while recording every
  decision; the reference `RiichiEnvExactSimulator` replays the same wall
  forced onto the same choices, comparing legal sets, chosen actions,
  per-decision observation facts, decision identity, and the terminal
  outcome (dense ordinals and boundary match come from the corrected
  oracle; Main measured 1165 decisions / 14 physical hands on the
  calls-heavy policy with zero gate/speed claim here);
- owner plane contracts: plane-major outputs (selected int64[B] @ 0,
  logp float32[B] @ 8B, value4 float32[B,4] @ 12B) and dtype-aligned
  input slabs round-trip through fill/commit with every row exactly once;
- failure gates: stale leases, double commits, and illegal selections
  fail the whole batch loudly (poison sticky);
- forked snapshot/restore preserves unissued READY uniforms, cursors,
  and descriptors with no redraw;
- oracle hot-path gate: native continuation walls are byte-equal to the
  Python wall law with a timing tripwire (regression signal only, never
  a throughput claim).
"""

from __future__ import annotations

import json
import math
import random
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from hydra2._native import rl as rl_native  # pyrefly: ignore[missing-import]
from hydra2.contracts.rules_manifest import rules_manifest_from_payload
from hydra2.engines.protocol import (
    WallSchedule,
    seat_permutation_literal,
    wall_schedule_digest,
)
from hydra2.engines.riichienv import RiichiEnvExactSimulator
from hydra2.engines.riichienv.walls import derive_hand_wall

pytestmark = pytest.mark.contract_package("WP-11")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULES_PAYLOAD = json.loads(
    (_REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_text()
)
_MANIFEST = rules_manifest_from_payload(_RULES_PAYLOAD["payload"])

_BUCKETS = (32, 64, 128, 256)
_PHASE_NAMES = {
    0: "round_start",
    1: "draw_decision",
    2: "discard_response",
    3: "kan_response",
    4: "round_end",
    5: "game_end",
}


class _DoneEvent:
    """Completed-CUDA-event stand-in: `query() -> True` is the exact
    contract `mark_host_ready` checks (batch-boundary observation of real
    completion, never a caller-supplied bool)."""

    def query(self) -> bool:
        return True


_DONE = _DoneEvent()


def _schedule(seed: int, schedule_id: str) -> WallSchedule:
    rng = random.Random(seed)
    tiles = list(range(136))
    rng.shuffle(tiles)
    frozen = tuple(tiles)
    return WallSchedule(
        schedule_id=schedule_id,
        physical_tiles=frozen,
        digest=wall_schedule_digest(schedule_id, frozen),
    )


def _profile(
    schedule: WallSchedule,
    seat_perm: list[int],
    *,
    pool_id: str = "test-pool",
    generation: int = 0,
    policy_id: str = "ckpt-test",
    games: int = 1,
) -> dict:
    tiles = [int(t) for t in schedule.physical_tiles]
    return {
        "pool_id": pool_id,
        "generation": generation,
        "policy_id": policy_id,
        "master_seed_hex": "ab" * 32,
        "experiment_id": "exp-test-rl-pool",
        "split_scope": "selection_training",
        "replicate_id": 0,
        "attempt_id": 0,
        "starting_points": int(_MANIFEST.starting_points),
        "worker_count": 2,
        "games": [
            {
                "wall_id": f"w{g}",
                "schedule_id": schedule.schedule_id,
                "schedule_digest": str(schedule.digest),
                "schedule_tiles": tiles,
                "seat_perm": seat_perm,
            }
            for g in range(games)
        ],
    }


def _context(
    sim: RiichiEnvExactSimulator | None = None,
) -> dict:
    """Pool context; with a reset sim the identity hashes are the real
    adapter lineage (game_id equality is then provable, not assumed).
    Without a sim the digests are valid-shaped distinct sentinels (the
    native commit path validates 32-byte hex shape fail-closed)."""
    if sim is None:
        return {
            "rules_id": _MANIFEST.rules_id,
            "rules_hash": "sha256:" + "00" * 32,
            "action_table_hash": "sha256:" + "01" * 32,
            "event_schema_hash": "sha256:" + "02" * 32,
            "observation_schema_hash": "sha256:" + "03" * 32,
            "packet_boundary_hash": "sha256:" + "04" * 32,
            "adapter_version": "2.0.0",
            "rank_values": [1.0, 0.0, -0.5, -1.0],
            "behavior_ckpt": "07" * 32,
        }
    return {
        "rules_id": _MANIFEST.rules_id,
        "rules_hash": str(sim._rules_hash),
        "action_table_hash": str(sim._table.digest),
        "event_schema_hash": str(sim._event_schema_hash),
        "observation_schema_hash": str(sim._observation_schema_hash),
        "packet_boundary_hash": str(sim._packet_boundary_hash),
        "adapter_version": "2.0.0",
        "rank_values": [1.0, 0.0, -0.5, -1.0],
        "behavior_ckpt": "07" * 32,
    }


def _open_pool(profile: dict, context: dict) -> object:
    return rl_native.PyNativePool.open(
        json.dumps(profile), json.dumps(context), "sha256:" + "09" * 32
    )


def _register_lane(
    pool: object, bucket_t: int, slot_id: int, capacity: int
) -> tuple[object, object]:
    input_len, output_len = rl_native.buffer_geometry(bucket_t, capacity)
    host_input = np.zeros(input_len, dtype=np.uint8)
    host_output = np.zeros(output_len, dtype=np.uint8)
    pool.register_buffers(bucket_t, slot_id, host_input, host_output)
    return host_input, host_output


def _settle(pool: object, lease: object) -> None:
    pool.mark_h2d_issued(lease)
    pool.mark_inference_queued(lease)
    pool.mark_d2h_issued(lease)
    pool.mark_host_ready(lease, _DONE, _DONE)


def _pack_output(n_valid: int, chosen: list[int], logps: list[float]) -> bytes:
    """Owner plane-major bytes: selected int64[B] @ 0, logp float32[B] @
    8B, value4 float32[B,4] @ 12B (zeros here)."""
    out = bytearray(n_valid * 28)
    for i, (cid, lp) in enumerate(zip(chosen, logps, strict=True)):
        struct.pack_into("<q", out, i * 8, cid)
        struct.pack_into("<f", out, 8 * n_valid + i * 4, lp)
    return bytes(out)


def _native_sig(entry: dict) -> tuple:
    """Canonical signature of one staged legal action (native descriptor)."""
    return (
        str(entry["kind"]),
        entry["tile"],
        entry["called_tile"],
        tuple(sorted(entry["consumed"])),
        entry["source"],
        bool(entry["declares_riichi"]),
        int(entry["actor"]),
    )


def _ref_sig(action: object) -> tuple:
    """Same signature from a reference CanonicalAction."""
    return (
        str(action.kind),
        action.tile,
        action.called_tile,
        tuple(sorted(action.consumed_tiles)),
        action.source_seat,
        bool(action.declares_riichi),
        int(action.actor),
    )


def _commit_first_legal(pool: object, host_output: object, lease: object) -> tuple[int, list[dict]]:
    """Settle + commit the min-id legal action per staged row; returns
    (rows_captured, descriptor_rows)."""
    _settle(pool, lease)
    descriptor = json.loads(pool.lease_descriptor(lease))
    rows = descriptor["rows"]
    chosen: list[int] = []
    logps: list[float] = []
    for row in rows:
        legal = sorted(row["legal"], key=lambda e: int(e["id"]))
        assert legal, f"staged row for seat {row['actor']} has an empty legal set"
        chosen.append(int(legal[0]["id"]))
        logps.append(-math.log(len(legal)))
    host_output[: len(chosen) * 28] = np.frombuffer(
        _pack_output(len(chosen), chosen, logps), dtype=np.uint8
    )
    count = pool.commit_actions(lease)
    assert count == len(chosen)
    return count, rows


def _plane_values(
    host_input: object, plane: tuple[str, tuple[int, ...], int], row: int
) -> list[int]:
    """Scalar plane values for one active row (int64/int32/bool planes)."""
    import numpy as np

    dtype, shape, off = plane
    width = {"int64": 8, "int32": 4, "bool": 1, "float64": 8}[dtype]
    per_row = 1
    for dim in shape[1:]:
        per_row *= dim
    base = off + row * per_row * width
    view = host_input[base : base + per_row * width]
    if dtype == "int64":
        return [int(v) for v in np.frombuffer(view, dtype="<i8")]
    if dtype == "int32":
        return [int(v) for v in np.frombuffer(view, dtype="<i4")]
    return [int(b) for b in bytes(view)]


def test_surface_registers_and_geometry_matches_aligned_law() -> None:
    schedule = _schedule(7001, "rl-pool-geom")
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3]), _context())
    try:
        # Cap128 carries zero alignment padding; odd capacities carry
        # bounded deterministic padding (never tail padding).
        for bucket_t in _BUCKETS:
            row = 7254 + 9 * bucket_t
            assert rl_native.buffer_geometry(bucket_t, 128) == (128 * row + 1024, 128 * 28)
            for cap in (1, 3, 7):
                input_len, output_len = rl_native.buffer_geometry(bucket_t, cap)
                assert input_len >= cap * row + cap * 8
                assert output_len == cap * 28
        host_input, host_output = _register_lane(pool, 32, 0, 4)
        assert len(host_input) == rl_native.buffer_geometry(32, 4)[0]
        assert len(host_output) == 4 * 28
        n_valid = pool.request_ready_fill(0)
        assert n_valid >= 0
        lease = pool.peek_ready()
        if lease is not None:
            assert lease.input_bytes <= len(host_input)
            assert lease.output_bytes == lease.n_valid * 28
            _commit_first_legal(pool, host_output, lease)
            pool.quiesce()
    finally:
        pool.close()


def test_full_loop_commits_every_row_exactly_once() -> None:
    schedule = _schedule(7002, "rl-pool-loop")
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3], games=2), _context())
    lanes = {}
    try:
        for slot, bucket_t in enumerate(_BUCKETS):
            lanes[slot] = (bucket_t, *_register_lane(pool, bucket_t, slot, 8))
        total_captured = 0
        for _ in range(20000):
            for slot, (_bucket_t, _, _) in lanes.items():
                pool.request_ready_fill(slot)
            leases = []
            while True:
                lease = pool.peek_ready()
                if lease is None:
                    break
                leases.append(lease)
            if not leases:
                completions = json.loads(pool.completions())
                if all(c["terminal"] for c in completions):
                    break
                continue
            for lease in leases:
                _, host_output = lanes[lease.slot_id][1], lanes[lease.slot_id][2]
                count, _ = _commit_first_legal(pool, host_output, lease)
                total_captured += count
        else:
            raise AssertionError("collection loop did not terminate within budget")
        pool.quiesce()
        status = pool.poll()
        assert status["poison"] is None
        assert status["pending_applies"] == 0
        assert status["captured_rows"] == total_captured
        completions = json.loads(pool.completions())
        assert len(completions) == 2
        assert all(c["terminal"] for c in completions)
        assert sum(c["captured"] for c in completions) == total_captured
        assert status["terminal_games"] == 2
    finally:
        pool.close()


def _slab_planes(host_input: object, lease: object) -> dict[str, tuple[str, tuple[int, ...], int]]:
    """(dtype, shape, byte_offset) per plane over the active prefix."""
    return {name: (dtype, tuple(shape), off) for name, dtype, shape, off in lease.input_layout}


@pytest.mark.parametrize(
    ("lane_capacity", "perm_kind", "perm"),
    [
        (16, "identity", [0, 1, 2, 3]),
        (1, "shift1", [1, 2, 3, 0]),
        (2, "shift2", [2, 3, 0, 1]),
        (16, "shift3", [3, 0, 1, 2]),
    ],
)
def test_generated_hanchan_matches_reference_action_event_observation_digest(
    lane_capacity: int, perm_kind: str, perm: list[int]
) -> None:
    """Complete generated hanchan parity: native drives first, the
    reference replays forced onto the same choices. Every decision
    compares legal sets, chosen actions, observation facts, decision
    identity, and game identity; the terminal outcome must agree."""
    schedule = _schedule(20260823, "rl-pool-parity")
    # The reference sim is reset FIRST so the pool opens on the identical
    # adapter lineage (game_id derives from these hashes; fakes would fork it).
    _sim_for_context = RiichiEnvExactSimulator()
    _sim_for_context.reset(
        rules=_MANIFEST,
        wall=schedule,
        seat_permutation=seat_permutation_literal(perm_kind),
    )
    pool = _open_pool(_profile(schedule, perm), _context(_sim_for_context))
    host: dict[int, tuple[object, object]] = {}
    records: list[dict] = []
    native_outcome: dict | None = None
    try:
        for slot, bucket_t in enumerate(_BUCKETS):
            host[slot] = _register_lane(pool, bucket_t, slot, lane_capacity)
        for _ in range(20000):
            for slot in host:
                pool.request_ready_fill(slot)
            leases = []
            while True:
                lease = pool.peek_ready()
                if lease is None:
                    break
                leases.append(lease)
            if not leases:
                completions = json.loads(pool.completions())
                if len(completions) == 1 and completions[0]["terminal"]:
                    native_outcome = completions[0]
                    break
                continue
            for lease in leases:
                host_input, host_output = host[lease.slot_id]
                _settle(pool, lease)
                descriptor = json.loads(pool.lease_descriptor(lease))
                # Freeze scheduling + sampler identity per staged row: logical
                # round must stay equal across split leases of one decision
                # (single game; no cohort barrier here). Uniform collisions are
                # legitimate so only cursor_before words are asserted exact,
                # never uniform distinctness.
                logical_round = int(descriptor["logical_round"])
                planes = _slab_planes(host_input, lease)
                for index, row in enumerate(descriptor["rows"]):
                    legal = sorted(row["legal"], key=lambda e: int(e["id"]))
                    assert legal, (
                        f"seat {row['actor']} has an empty legal set at a pending decision"
                    )
                    chosen = legal[0]
                    records.append(
                        {
                            "slot": int(lease.slot_id),
                            "bucket": int(lease.bucket_t),
                            "lease": int(lease.lease_id),
                            "logical_round": logical_round,
                            "wall": str(row["wall_id"]),
                            "actor": int(row["actor"]),
                            "legal": [_native_sig(e) for e in legal],
                            "chosen": _native_sig(chosen),
                            "decision_id": str(row["decision_id"]),
                            "sequence": int(row["sequence"]),
                            "obs_hash": row["obs_hash"],
                            "cursor_before": int(row["cursor_before"]),
                            "uniform": float(row["uniform"]),
                            "slab": {
                                "scores": _plane_values(host_input, planes["scores"], index),
                                "dealer": _plane_values(host_input, planes["dealer"], index)[0],
                                "honba": _plane_values(host_input, planes["honba"], index)[0],
                                "round_index": _plane_values(
                                    host_input, planes["round_index"], index
                                )[0],
                                "hand_number": _plane_values(
                                    host_input, planes["hand_number"], index
                                )[0],
                                "turn_actor": _plane_values(
                                    host_input, planes["turn_actor"], index
                                )[0],
                                "phase": _plane_values(host_input, planes["phase"], index)[0],
                                "kan_count": _plane_values(host_input, planes["kan_count"], index)[
                                    0
                                ],
                            },
                        }
                    )
                picked: list[int] = []
                logps: list[float] = []
                for row in descriptor["rows"]:
                    legal = sorted(row["legal"], key=lambda e: int(e["id"]))
                    picked.append(int(legal[0]["id"]))
                    logps.append(-math.log(len(legal)))
                host_output[: len(picked) * 28] = np.frombuffer(
                    _pack_output(len(picked), picked, logps), dtype=np.uint8
                )
                assert pool.commit_actions(lease) == len(picked)
        else:
            raise AssertionError("native hanchan did not terminate within budget")
        assert native_outcome is not None
        pool.quiesce()
    finally:
        pool.close()

    assert records, "native drive recorded no decisions"
    # Sampler cursors advance exactly one 8-byte word per staged decision per
    # seat, in record order (`cursor_before` is the byte cursor `counter * 8`,
    # never the count): unstaged seats (overflow, cross-bucket waits, pinned
    # games) reserve no words, so no phantom reservations can fork uniforms.
    staged_counts: dict[tuple[str, int], int] = {}
    for record in records:
        key = (record["wall"], record["actor"])
        assert record["cursor_before"] == 8 * staged_counts.get(key, 0), (
            f"seat {key} cursor skipped words outside staged decisions"
        )
        staged_counts[key] = staged_counts.get(key, 0) + 1
    # Reference replay forced onto the native choices.
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=schedule,
        seat_permutation=seat_permutation_literal(perm_kind),
    )
    assert sim._game_id, "reference produced no game id"
    # commit in staged order, so each group is CHECKED in record order (reads
    # only; the sim state is frozen for the whole group) but APPLIED ascending
    # by actor. The combined step is order-independent, so the trajectory and
    # outcome gates are unchanged. Draws have unique ids (one-row groups).
    step = 0
    cursor = 0
    multi_actor_windows = 0
    same_bucket_windows = 0
    while cursor < len(records):
        end = cursor + 1
        while end < len(records) and records[end]["decision_id"] == records[cursor]["decision_id"]:
            end += 1
        matched: list[tuple[int, object]] = []
        for record in records[cursor:end]:
            actor = record["actor"]
            legals = sim.legal_actions(actor)
            assert legals, f"step {step}: seat {actor} has an empty reference legal set"
            ref_sigs = sorted(_ref_sig(a) for a in legals)
            assert ref_sigs == sorted(record["legal"]), (
                f"step {step} seat {actor}: native/reference legal sets diverge\n"
                f"native={sorted(record['legal'])}\nreference={ref_sigs}"
            )
            match = [a for a in legals if _ref_sig(a) == record["chosen"]]
            assert len(match) == 1, f"step {step}: chosen signature ambiguous or missing"
            ref_obs = sim.actor_observation(actor)
            slab = record["slab"]
            assert list(ref_obs.scores) == slab["scores"], f"step {step}: scores diverge"
            assert int(ref_obs.dealer) == slab["dealer"], f"step {step}: dealer diverges"
            assert int(ref_obs.honba) == slab["honba"], f"step {step}: honba diverges"
            assert int(ref_obs.round_index) == slab["round_index"], f"step {step}: round diverges"
            assert int(ref_obs.hand_number) == slab["hand_number"], f"step {step}: hand diverges"
            assert int(ref_obs.turn_actor) == slab["turn_actor"], f"step {step}: turn diverges"
            assert int(ref_obs.kan_count) == slab["kan_count"], f"step {step}: kan count diverges"
            assert _PHASE_NAMES[slab["phase"]] == str(ref_obs.phase), f"step {step}: phase diverges"
            assert str(ref_obs.decision_id) == record["decision_id"], (
                f"step {step}: decision identity diverges "
                f"(native={record['decision_id']} reference={ref_obs.decision_id}); "
                "envelope streams are not 1:1"
            )
            assert int(ref_obs.sequence) == record["sequence"], (
                f"step {step}: sequence diverges "
                f"(native={record['sequence']} reference={ref_obs.sequence})"
            )
            ref_hash = str(ref_obs.observation_hash).removeprefix("sha256:")
            assert ref_hash == record["obs_hash"], (
                f"step {step} seat {actor} ({record['decision_id']}): observation "
                f"digest diverges (native={record['obs_hash']} reference={ref_hash}); "
                "canonical identity docs disagree"
            )
            assert str(ref_obs.game_id) == str(sim._game_id), f"step {step}: game scope drift"
            matched.append((int(actor), match[0]))
            step += 1
        actors = [record["actor"] for record in records[cursor:end]]
        if len(set(actors)) > 1:
            # Multi-responder window: every seat observes the same frozen
            # decision. Split leases of one decision must share logical_round
            # (single game; pool hold cannot wait on earliest games here).
            multi_actor_windows += 1
            rounds = {record["logical_round"] for record in records[cursor:end]}
            assert rounds == {records[cursor]["logical_round"]}, (
                f"step {step}: split window decision {records[cursor]['decision_id']} "
                f"spans logical rounds {sorted(rounds)}"
            )
            assert len(actors) == len(set(actors)), (
                f"step {step}: duplicate answer in window {records[cursor]['decision_id']}"
            )
            buckets = {record["bucket"] for record in records[cursor:end]}
            if len(buckets) == 1:
                same_bucket_windows += 1
                leases = {record["lease"] for record in records[cursor:end]}
                # Fixed per-frontier chunk boundaries may legally split a
                # same-bucket window even when it fits the lane cap: only
                # per-lease capacity bounds, every-seat-once, and
                # logical-round stability are pinned here.
                for lid in leases:
                    n = sum(1 for r in records[cursor:end] if r["lease"] == lid)
                    assert 0 < n <= lane_capacity, (
                        f"step {step}: lease {lid} carries {n} rows outside (0, cap]"
                    )
        for _, action in sorted(matched, key=lambda pair: pair[0]):
            sim.apply(action)
        cursor = end
    assert same_bucket_windows > 0, "hanchan exercised no same-bucket multi-responder window"
    assert multi_actor_windows > 0, "hanchan exercised no multi-responder window"
    assert sim._terminal, "reference did not terminate on the forced trajectory"
    outcome = sim._raw_outcome
    assert outcome is not None
    # Per-seat exact: native canonical_scores(inv) and adapter
    # _to_canonical_scores(perm) both map engine order to canonical seats, so
    # with matching perms the lists compare directly (sorted() would weaken).
    assert list(outcome.final_scores) == native_outcome["outcome"]["final_scores"]
    assert list(outcome.ranks) == list(native_outcome["outcome"]["ranks"])
    assert native_outcome["captured"] == len(records)
    # Game identity derives from the same lineage on both sides.
    pool2 = _open_pool(
        _profile(schedule, perm, pool_id="parity-gid"),
        _context(sim),
    )
    try:
        game_id = json.loads(pool2.completions())[0]["game_id"]
    finally:
        pool2.close()
    assert game_id == sim._game_id, f"game lineage diverges: {game_id} != {sim._game_id}"


def test_continuation_walls_match_oracle_with_timing_tripwire() -> None:
    """Oracle hot path: native continuation walls are byte-equal to the
    Python wall law for hands 1..8; the timing comparison is a regression
    tripwire (4x slack, printed for the record), never a speed claim."""
    schedule = _schedule(424242, "rl-pool-walltrip")
    digest = str(schedule.digest)
    hands = list(range(1, 9))
    expected = [
        bytes(
            derive_hand_wall(schedule_digest=digest, schedule_id=schedule.schedule_id, hand_index=h)
        )
        for h in hands
    ]
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        python_walls = [
            bytes(
                derive_hand_wall(
                    schedule_digest=digest, schedule_id=schedule.schedule_id, hand_index=h
                )
            )
            for h in hands
        ]
    python_dt = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        native_walls = [
            bytes(w)
            for w in rl_native.derive_hand_walls(
                schedule_id=schedule.schedule_id, digest=digest, hands=hands
            )
        ]
    native_dt = (time.perf_counter() - t0) / reps
    assert [bytes(w) for w in native_walls] == expected == python_walls
    for wall in native_walls:
        assert sorted(wall) == list(range(136))
    print(
        f"\nwall hot path per 8-hand batch: python={python_dt * 1e3:.2f}ms "
        f"native={native_dt * 1e3:.2f}ms"
    )
    assert native_dt <= 4 * python_dt + 0.5, "native wall path regressed past the tripwire"


def test_stale_double_and_illegal_batch_fail_loud() -> None:
    schedule = _schedule(7003, "rl-pool-fail")
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3]), _context())
    try:
        # Fresh games stage only T=32 rows (history starts empty and grows;
        # larger lanes stall work-conserving with zero-row fills), so the
        # failure gates run on the lane that actually has decisions.
        _host_input, host_output = _register_lane(pool, 32, 7, 8)
        assert pool.request_ready_fill(7) > 0
        lease = pool.peek_ready()
        assert lease is not None
        _settle(pool, lease)
        descriptor = json.loads(pool.lease_descriptor(lease))
        n_valid = lease.n_valid
        first_ids = [min(int(e["id"]) for e in row["legal"]) for row in descriptor["rows"]]
        logps = [-math.log(len(row["legal"])) for row in descriptor["rows"]]
        host_output[: n_valid * 28] = np.frombuffer(
            _pack_output(n_valid, first_ids, logps), dtype=np.uint8
        )
        assert pool.commit_actions(lease) == n_valid
        # Double commit of the spent lease fails (stale lease, no poison).
        with pytest.raises(ValueError):
            pool.commit_actions(lease)
        assert pool.poll()["poison"] is None
        pool.quiesce()
    finally:
        pool.close()


def test_illegal_selection_poisons_whole_batch() -> None:
    schedule = _schedule(7004, "rl-pool-poison")
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3]), _context())
    try:
        _, host_output = _register_lane(pool, 32, 7, 8)
        assert pool.request_ready_fill(7) > 0
        lease = pool.peek_ready()
        assert lease is not None
        _settle(pool, lease)
        n_valid = lease.n_valid
        bad = bytearray(n_valid * 28)
        # Selected id past the census end with finite logp/values: the
        # ENTIRE batch must reject (never a partial capture).
        struct.pack_into("<q", bad, 0, 6792)
        struct.pack_into("<f", bad, 8 * n_valid, -1.0)
        # Element-wise copy: numpy casts a bare `bytes` scalar via int().
        host_output[: n_valid * 28] = np.frombuffer(bytes(bad), dtype=np.uint8)
        with pytest.raises(ValueError, match="served legal set"):
            pool.commit_actions(lease)
        assert pool.poll()["poison"] is not None
    finally:
        # The hostile lease stays issued: close refuses it loudly
        # (never close while an issued lease is live, even poisoned).
        with pytest.raises(ValueError, match="live leases"):
            pool.close()


def test_forked_snapshot_preserves_ready_reservations() -> None:
    schedule = _schedule(7005, "rl-pool-fork")
    profile = _profile(schedule, [0, 1, 2, 3])
    context = _context()
    pool = _open_pool(profile, context)
    try:
        # T=32 is the lane with rows on a fresh game (see stale-test note).
        input_len, _ = rl_native.buffer_geometry(32, 8)
        host_input = np.zeros(input_len, dtype=np.uint8)
        host_output = np.zeros(8 * 28, dtype=np.uint8)
        pool.register_buffers(32, 1, host_input, host_output)
        n_valid = pool.request_ready_fill(1)
        assert n_valid > 0
        before = bytes(host_input)
        reserved = pool.peek_ready()
        assert reserved is not None
        reserved_rows = json.loads(pool.lease_descriptor(reserved))["rows"]
        pool.return_ready_lease(reserved)
        pool.quiesce()
        snap = pool.snapshot()
        pool.resume_admission()
        fork = rl_native.PyNativePool.restore(
            json.dumps(profile), json.dumps(context), snap, "sha256:" + "09" * 32
        )
        try:
            fork_input = np.zeros(input_len, dtype=np.uint8)
            fork_output = np.zeros(8 * 28, dtype=np.uint8)
            fork.register_buffers(32, 1, fork_input, fork_output)
            refilled = fork.refill_staged(1)
            assert refilled == n_valid
            # Active input prefix is byte-identical (identical uniforms).
            assert bytes(fork_input) == before
            lease = fork.peek_ready()
            assert lease is not None and lease.n_valid == n_valid
            first = json.loads(fork.lease_descriptor(lease))
            orig_lease = pool.peek_ready()
            assert orig_lease is not None
            orig = json.loads(pool.lease_descriptor(orig_lease))
            assert orig["rows"] == reserved_rows
            pool.return_ready_lease(orig_lease)
            assert [r["cursor_before"] for r in first["rows"]] == [
                r["cursor_before"] for r in orig["rows"]
            ]
            assert [r["uniform"] for r in first["rows"]] == [r["uniform"] for r in orig["rows"]]
            _settle(fork, lease)
            ids = [min(int(e["id"]) for e in row["legal"]) for row in first["rows"]]
            lps = [-math.log(len(row["legal"])) for row in first["rows"]]
            fork_output[: n_valid * 28] = np.frombuffer(
                _pack_output(n_valid, ids, lps), dtype=np.uint8
            )
            assert fork.commit_actions(lease) == n_valid
            fork.quiesce()
        finally:
            fork.close()
    finally:
        pool.close()


def test_frozen_head_pin_refuses_reassign_then_converges() -> None:
    """Frozen-head pin: a held READY pins the global head; next converges.

    Two distinct-wall games (cap 1) hunt the first intermediate READY
    snapshot: immediately after one productive head install, before the next
    chunk stages, the held READY row H is still issued while the frozen plan
    still shows an unfinished head chunk naming the same game. Both forks
    adopt that READY snapshot and reuse the same H (no new admit for H); the
    held branch refuses the next admit (no overtake, no reassignment), then
    both commit H and stage identical next rows/cursors/observations.
    """
    sched_a = _schedule(1000, "rl-pool-barrier-a")
    sched_b = _schedule(2000, "rl-pool-barrier-b")
    profile = _profile(sched_a, [0, 1, 2, 3], games=2)
    second = profile["games"][1]
    second["schedule_id"] = sched_b.schedule_id
    second["schedule_digest"] = str(sched_b.digest)
    second["schedule_tiles"] = [int(t) for t in sched_b.physical_tiles]
    second["seat_perm"] = [1, 2, 3, 0]
    context = _context()
    pool = _open_pool(profile, context)
    lanes: dict[int, tuple] = {}
    slot = 0
    try:
        for bucket in _BUCKETS:
            for _ in range(2):
                lanes[slot] = (bucket, *_register_lane(pool, bucket, slot, 1))
                slot += 1

        def drain_all(target: object) -> list:
            found = []
            while True:
                lease = target.peek_ready()
                if lease is None:
                    return found
                found.append(lease)

        base_ready: str | None = None
        held_slot = -1
        held_n = 0
        held_input: bytes | None = None
        head_pos = -1
        for _ in range(500):
            if base_ready is not None:
                break
            for lane in sorted(lanes):
                n = pool.request_ready_fill(lane)
                if n == 0:
                    continue
                # Productive install: snapshot at once, before the next lane
                # can stage the following chunk, so READY H stays observable.
                snap = pool.snapshot()
                doc = json.loads(snap)
                plan = doc.get("round_plan")
                probe = pool.peek_ready()
                assert probe is not None, (
                    "productive fill left no READY lease to inspect; the pool "
                    "must install exactly the frozen head chunk at READY"
                )
                descriptor = json.loads(pool.lease_descriptor(probe))
                probe_rows = descriptor["rows"]
                wall_order = [g["wall_id"] for g in doc["games"]]
                held_games = {wall_order.index(str(r["wall_id"])) for r in probe_rows}
                edge = False
                head = None
                following: list = []
                cap = 0
                if isinstance(plan, dict) and int(probe.slot_id) == lane:
                    refs = plan["refs"]
                    caps = plan["min_cap"]
                    issued = plan["issued"]
                    head = next((b for b in range(4) if int(issued[b]) < len(refs[b])), None)
                    # Same-bucket continuation keeps the same-T mate
                    # eligible, so a None admit below proves a pin rather
                    # than a wrong-bucket wait.
                    if (
                        head is not None
                        and caps[head] is not None
                        and _BUCKETS[head] == lanes[lane][0]
                    ):
                        cap = int(caps[head])
                        start = int(issued[head])
                        following = refs[head][start : start + cap]
                        following_games = {int(r["game"]) for r in following}
                        if following and held_games & following_games:
                            edge = True
                if edge:
                    assert head is not None
                    base_ready = snap
                    held_slot = lane
                    held_n = n
                    held_input = bytes(lanes[lane][1])
                    head_pos = head
                    pool.return_ready_lease(probe)
                    break
                pool.return_ready_lease(probe)
                # Not the pinned window: commit everything staged so the next
                # productive attempt starts from a single outstanding H again.
                staged = drain_all(pool)
                for lease in staged:
                    _commit_first_legal(pool, lanes[int(lease.slot_id)][2], lease)
                while pool.poll()["pending_applies"]:
                    pool.wait_for_progress()
            else:
                completions = json.loads(pool.completions())
                if completions and all(c["terminal"] for c in completions):
                    break
                continue
        assert base_ready is not None, (
            "frozen-head pin edge not observed within 500 rounds for seeds "
            "1000+2000 cap1: no intermediate READY chunk whose held game "
            "recurs in the next unissued head chunk, so a pin refusal would "
            "be vacuous without that same-game recurrence"
        )
        pool.quiesce()
    finally:
        pool.close()
    assert head_pos in range(4)
    assert held_slot >= 0 and held_input is not None
    head_t = _BUCKETS[head_pos]
    assert lanes[held_slot][0] == head_t, (
        "hunted H must sit on the next head bucket, otherwise the same-T "
        "mate admit below waits on bucket order instead of the game pin"
    )
    blob = (json.dumps(profile), json.dumps(context))
    fork_hold = rl_native.PyNativePool.restore(*blob, base_ready, "sha256:" + "09" * 32)
    fork_fast = rl_native.PyNativePool.restore(*blob, base_ready, "sha256:" + "09" * 32)
    hold_h: object | None = None
    fast_h: object | None = None
    hold_next: object | None = None
    fast_next: object | None = None
    try:
        owned: dict[str, dict] = {"hold": {}, "fast": {}}
        for fork, table in ((fork_hold, owned["hold"]), (fork_fast, owned["fast"])):
            lane_id = 0
            for bucket_t in _BUCKETS:
                for _ in range(2):
                    table[lane_id] = (bucket_t, *_register_lane(fork, bucket_t, lane_id, 1))
                    lane_id += 1

        def row_tuple(target: object, lease: object) -> tuple:
            descriptor = json.loads(target.lease_descriptor(lease))
            assert len(descriptor["rows"]) == 1
            row = descriptor["rows"][0]
            return (
                int(descriptor["logical_round"]),
                str(row["wall_id"]),
                str(row["game_id"]),
                str(row["decision_id"]),
                int(row["actor"]),
                int(row["sequence"]),
                int(row["cursor_before"]),
                float(row["uniform"]),
                str(row["obs_hash"]),
                tuple(sorted(_native_sig(e) for e in row["legal"])),
            )

        # Adopted READY H re-encodes byte-identically (no new sampler words);
        # never admit a fresh H, which the pin must block while H is held.
        assert fork_hold.refill_staged(held_slot) == held_n, (
            "restored hold fork must re-encode the retained READY H row count"
        )
        assert fork_fast.refill_staged(held_slot) == held_n, (
            "restored fast fork must re-encode the retained READY H row count"
        )
        assert bytes(owned["hold"][held_slot][1]) == held_input, (
            "hold fork READY H input prefix must match the hunted H bytes"
        )
        assert bytes(owned["fast"][held_slot][1]) == held_input, (
            "fast fork READY H input prefix must match the hunted H bytes"
        )
        hold_h = fork_hold.peek_ready()
        fast_h = fork_fast.peek_ready()
        assert hold_h is not None and fast_h is not None, (
            "both restores must expose the retained READY H without a new admit"
        )
        assert int(hold_h.slot_id) == held_slot and int(fast_h.slot_id) == held_slot, (
            "retained READY H must sit on the hunted lane on both forks"
        )
        held_tuple = row_tuple(fork_hold, hold_h)
        assert row_tuple(fork_fast, fast_h) == held_tuple, (
            "both forks must adopt identical READY H rows"
        )
        # Nonvacuous metadata proof on the fork: the next head chunk still
        # names the held game, so the refusal below is a game pin and not an
        # empty-head or wrong-bucket wait.
        staged_doc = json.loads(fork_hold.snapshot())
        staged_plan = staged_doc["round_plan"]
        assert isinstance(staged_plan, dict), "plan missing with READY H held"
        refs = staged_plan["refs"]
        cap = int(staged_plan["min_cap"][head_pos])
        start = int(staged_plan["issued"][head_pos])
        wall_order = [g["wall_id"] for g in staged_doc["games"]]
        held_idx = wall_order.index(str(held_tuple[1]))
        following = refs[head_pos][start : min(start + cap, len(refs[head_pos]))]
        assert following, "head exhausted while READY H still held"
        assert held_idx in {int(r["game"]) for r in following}, (
            "held game absent from the next planned head chunk; the pin "
            "refusal below would be vacuous without that recurrence"
        )
        mate = min(s for s in owned["hold"] if s != held_slot and owned["hold"][s][0] == head_t)
        # Pinned global head admits nothing later (no overtake, no reassign).
        assert fork_hold.admit_next_fill([mate]) is None, (
            "held READY H must pin the global head: an eligible same-T mate "
            "for the next head chunk sharing H's game must wait, never "
            "overtake or reassigned around the pin"
        )
        _commit_first_legal(fork_fast, owned["fast"][held_slot][2], fast_h)
        fast_h = None
        while fork_fast.poll()["pending_applies"]:
            fork_fast.wait_for_progress()
        fork_fast.quiesce()
        fork_fast.resume_admission()
        assert fork_fast.admit_next_fill([mate]) == mate, (
            "once H commits, the same mate lane must serve the next head chunk"
        )
        assert int(fork_fast.take_fill_blocking(mate)["n_valid"]) == 1
        fast_next = fork_fast.peek_ready()
        assert fast_next is not None
        assert int(fast_next.slot_id) == mate
        fast_next_tuple = row_tuple(fork_fast, fast_next)
        fast_next_input = bytes(owned["fast"][mate][1])
        fast_next_round = json.loads(fork_fast.lease_descriptor(fast_next))["logical_round"]
        _commit_first_legal(fork_hold, owned["hold"][held_slot][2], hold_h)
        hold_h = None
        while fork_hold.poll()["pending_applies"]:
            fork_hold.wait_for_progress()
        fork_hold.quiesce()
        fork_hold.resume_admission()
        assert fork_hold.admit_next_fill([mate]) == mate, (
            "held branch must converge onto the same next head chunk after "
            "committing the identical H"
        )
        assert int(fork_hold.take_fill_blocking(mate)["n_valid"]) == 1
        hold_next = fork_hold.peek_ready()
        assert hold_next is not None
        assert int(hold_next.slot_id) == mate
        assert row_tuple(fork_hold, hold_next) == fast_next_tuple, (
            "both branches must stage identical next rows after the pin clears"
        )
        assert bytes(owned["hold"][mate][1]) == fast_next_input, (
            "both branches must encode identical next input bytes"
        )
        assert json.loads(fork_hold.lease_descriptor(hold_next))["logical_round"] == fast_next_round
        _commit_first_legal(fork_hold, owned["hold"][mate][2], hold_next)
        hold_next = None
        _commit_first_legal(fork_fast, owned["fast"][mate][2], fast_next)
        fast_next = None
        fork_hold.quiesce()
        fork_fast.quiesce()
    finally:
        for fork, lease in (
            (fork_hold, hold_h),
            (fork_fast, fast_h),
            (fork_hold, hold_next),
            (fork_fast, fast_next),
        ):
            if lease is not None:
                fork.return_ready_lease(lease)
        try:
            fork_hold.quiesce()
            fork_hold.close()
        finally:
            fork_fast.quiesce()
            fork_fast.close()


def test_restore_rejects_foreign_derivation() -> None:
    """Same bytes, foreign derivation: restore fails before open/replay."""
    schedule = _schedule(7013, "rl-pool-snapderiv-mismatch")
    profile = _profile(schedule, [0, 1, 2, 3])
    context = _context()
    pool = _open_pool(profile, context)
    try:
        snap = pool.snapshot()
        with pytest.raises(ValueError, match="derivation"):
            rl_native.PyNativePool.restore(
                json.dumps(profile), json.dumps(context), snap, "sha256:" + "08" * 32
            )
    finally:
        pool.close()


def test_restore_rejects_keyless_snapshot() -> None:
    """Snapshots predating the derivation root key cannot load."""
    schedule = _schedule(7014, "rl-pool-snapderiv-keyless")
    profile = _profile(schedule, [0, 1, 2, 3])
    context = _context()
    pool = _open_pool(profile, context)
    try:
        doc = json.loads(pool.snapshot())
        del doc["derivation"]
        with pytest.raises(ValueError, match="derivation"):
            rl_native.PyNativePool.restore(
                json.dumps(profile),
                json.dumps(context),
                json.dumps(doc),
                "sha256:" + "09" * 32,
            )
    finally:
        pool.close()


def test_restore_rejects_malformed_derivation() -> None:
    """A present-but-malformed derivation hex is not a digest: reject."""
    schedule = _schedule(7015, "rl-pool-snapderiv-badhex")
    profile = _profile(schedule, [0, 1, 2, 3])
    context = _context()
    pool = _open_pool(profile, context)
    try:
        doc = json.loads(pool.snapshot())
        doc["derivation"] = "zz" * 32
        with pytest.raises(ValueError, match="32 bytes hex"):
            rl_native.PyNativePool.restore(
                json.dumps(profile), json.dumps(context), json.dumps(doc), "sha256:" + "09" * 32
            )
    finally:
        pool.close()


def test_fixed_round_blocks_next_fill_until_cohort_round_complete() -> None:
    """A held cohort chunk prevents earlier games advancing or consuming draws."""
    schedule = _schedule(7011, "rl-pool-fixed-round")
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3], games=4), _context())
    try:
        host = {slot: _register_lane(pool, 32, slot, 2) for slot in (0, 1)}
        assert pool.request_ready_fill(0) == pool.request_ready_fill(1) == 2
        first, held = pool.peek_ready(), pool.peek_ready()
        assert first is not None and held is not None
        first_rows = json.loads(pool.lease_descriptor(first))["rows"]
        held_rows = json.loads(pool.lease_descriptor(held))["rows"]
        assert sorted(row["wall_id"] for row in first_rows + held_rows) == [
            "w0",
            "w1",
            "w2",
            "w3",
        ]
        _commit_first_legal(pool, host[first.slot_id][1], first)
        while pool.poll()["pending_applies"]:
            pool.wait_for_progress()
        assert pool.admit_next_fill([first.slot_id]) is None
        assert pool.request_ready_fill(first.slot_id) == 0
        assert pool.peek_ready() is None
        assert json.loads(pool.lease_descriptor(held))["rows"] == held_rows
        _commit_first_legal(pool, host[held.slot_id][1], held)
        while pool.poll()["pending_applies"]:
            pool.wait_for_progress()
        assert pool.admit_next_fill([first.slot_id]) == first.slot_id
        assert pool.take_fill_blocking(first.slot_id)["n_valid"] == 2
        next_lease = pool.peek_ready()
        assert next_lease is not None
        descriptor = json.loads(pool.lease_descriptor(next_lease))
        assert descriptor["logical_round"] == 1
        walls = [row["wall_id"] for row in descriptor["rows"]]
        assert len(walls) == 2
        if len(set(walls)) != 2:
            # Coalesced claim window: one game contributes both seats of a
            # single frozen window in one cap-2 fill (shared decision_id,
            # distinct actors, buckets matched). Two games would show two
            # distinct walls instead; both shapes are exact, never partial.
            assert [row["wall_id"] for row in descriptor["rows"]] == [walls[0], walls[0]]
            ids = [row["decision_id"] for row in descriptor["rows"]]
            assert ids[0] == ids[1]
            actors = [row["actor"] for row in descriptor["rows"]]
            assert actors[0] != actors[1]
        _commit_first_legal(pool, host[next_lease.slot_id][1], next_lease)
        pool.quiesce()
        assert pool.poll()["poison"] is None
    finally:
        pool.close()


def test_complete_native_game_export_freeze_seal_validates(tmp_path: Path) -> None:
    """Real native rows must survive full identity validation, not a patched hash."""
    from hydra2.training import rl_bridge

    schedule = _schedule(7002, "rl-pool-export-seal")
    oracle = RiichiEnvExactSimulator()
    oracle.reset(
        rules=_MANIFEST, wall=schedule, seat_permutation=seat_permutation_literal("identity")
    )
    context = _context(oracle)
    pool = _open_pool(_profile(schedule, [0, 1, 2, 3]), context)
    try:
        lanes = {
            slot: _register_lane(pool, bucket, slot, 8) for slot, bucket in enumerate(_BUCKETS)
        }
        captured = 0
        for _ in range(20000):
            for slot in lanes:
                pool.request_ready_fill(slot)
            while (lease := pool.peek_ready()) is not None:
                count, _ = _commit_first_legal(pool, lanes[lease.slot_id][1], lease)
                captured += count
            completion = json.loads(pool.completions())[0]
            if completion["terminal"]:
                break
        else:
            pytest.fail("native fixture did not finish the complete hanchan")
        pool.quiesce()
        assert pool.poll()["poison"] is None
        assert completion["captured"] == captured
        terminal_g4 = completion["outcome"]["utility_g4"]
        tape = pool.export_tape()
    finally:
        pool.close()

    descriptor, actor_raw, game_raw, value_raw, utility_raw = rl_bridge.terminal_inputs(tape)
    assert descriptor["row_count"] == captured and descriptor["game_count"] == 1
    actors = np.frombuffer(actor_raw, dtype="<i8")
    games = np.frombuffer(game_raw, dtype="<i8")
    values = np.frombuffer(value_raw, dtype="<f4").reshape(-1, 4)
    utilities = np.frombuffer(utility_raw, dtype="<f8").reshape(-1, 4)
    assert utilities.tolist() == [terminal_g4]
    # CPU oracle matches the production FP32 gather/subtract law; changing
    # cast order would admit different frozen advantages before learning.
    advantages = utilities.astype(np.float32)[games, actors] - values[np.arange(captured), actors]
    assert (
        rl_bridge.freeze_terminal_advantages(tape, advantages.tobytes())["rows_frozen"] == captured
    )
    header = {
        "rules_hash": context["rules_hash"],
        "utility_manifest_hash": "sha256:" + "ab" * 32,
        "observation_schema_hash": context["observation_schema_hash"],
        "action_table_hash": context["action_table_hash"],
        "behavior_policy_checkpoint_hash": "sha256:" + context["behavior_ckpt"],
        "behavior_model_spec_hash": "sha256:" + "ee" * 32,
        "source_wall_ledger_hash": "sha256:" + "cc" * 32,
        "row_derivation_hash": "sha256:" + "09" * 32,
    }
    artifact, metas = str(tmp_path / "native.zst"), str(tmp_path / "native.metas.json")
    receipt = rl_bridge.seal_sync(tape, header, ["w0"], [], False, artifact, metas)
    validated = rl_bridge.validate_artifact(artifact, metas, header, ["w0"], [], False)
    assert receipt.row_count == validated.row_count == captured
    assert receipt.artifact_digest == validated.artifact_digest
    rows = json.loads(Path(metas).read_bytes())["rows"]
    assert [row["step"] for row in rows] == list(range(captured))
    assert len({row["row_id"] for row in rows}) == captured
    assert all(row["g4"] == terminal_g4 for row in rows)
    assert all(math.isfinite(row["raw_a"]) and 0 < math.exp(row["old_lp"]) <= 1 for row in rows)

    from hydra2._native import rl as native_rl
    from hydra2._native import rl_artifact as native_artifact

    root = tmp_path / "runtime"
    reference = json.loads(native_rl.save_tape(root_dir=str(root), tape_handle=tape))
    restored = native_rl.restore_tape(root_dir=str(root), tape_ref=json.dumps(reference).encode())
    assert json.loads(native_rl.save_tape(root_dir=str(root), tape_handle=restored)) == reference
    restored_inputs = rl_bridge.terminal_inputs(restored)
    assert restored_inputs[1:] == (actor_raw, game_raw, value_raw, utility_raw)
    native_artifact.drop_tape(restored)
    native_artifact.drop_tape(tape)


def _play_to_terminal(pool: object, lanes: dict) -> tuple[int, list[dict]]:
    """Drive fills/commits until every cohort game is terminal."""
    captured = 0
    for _ in range(20000):
        for slot in lanes:
            pool.request_ready_fill(slot)
        while (lease := pool.peek_ready()) is not None:
            count, _ = _commit_first_legal(pool, lanes[lease.slot_id][1], lease)
            captured += count
        completions = json.loads(pool.completions())
        if all(entry["terminal"] for entry in completions):
            return captured, completions
    pytest.fail("native fixture did not finish the complete hanchan")


def test_drain_moves_live_tape_identical_to_clone_export() -> None:
    """Destructive drain moves (never clones) the exact live rows.

    Two equivalent pools play the same fixed cohort: clone export and
    destructive drain must freeze identical terminal inputs and advantages.
    Incomplete cohorts reject before any move (the pool stays usable);
    after drain, second drain/snapshot/clone/admission reject while
    read-only status and close stay legal.
    """
    from hydra2._native import rl_artifact as native_artifact
    from hydra2.training import rl_bridge

    policy = "sha256:" + "c0" * 32
    pipeline = "sha256:" + "9d" * 32
    binding = {
        "pipeline_spec_id": pipeline,
        "generation": 0,
        "behavior_policy_id": policy,
    }
    binding_bytes = json.dumps(binding).encode()
    schedule = _schedule(7002, "rl-pool-export-seal")
    oracle = RiichiEnvExactSimulator()
    oracle.reset(
        rules=_MANIFEST, wall=schedule, seat_permutation=seat_permutation_literal("identity")
    )
    context = _context(oracle)
    context["behavior_ckpt"] = "c0" * 32
    profile = _profile(schedule, [0, 1, 2, 3], policy_id=policy, games=2)

    def _lanes(pool: object) -> dict:
        return {slot: _register_lane(pool, bucket, slot, 8) for slot, bucket in enumerate(_BUCKETS)}

    # Clone path: archive open plus copy export.
    pool_a = _open_pool(profile, context)
    tape_a = None
    try:
        captured_a, completions_a = _play_to_terminal(pool_a, _lanes(pool_a))
        assert len(completions_a) == 2
        for entry in completions_a:
            assert entry["terminal"] and entry["outcome"] is not None
            assert all(math.isfinite(v) for v in entry["outcome"]["utility_g4"])
        pool_a.quiesce()
        assert pool_a.poll()["poison"] is None
        tape_a = pool_a.export_tape()
    finally:
        pool_a.close()

    # Live path: open_live plus destructive drain on an equivalent pool.
    pool_b = rl_native.PyNativePool.open_live(
        json.dumps(profile), json.dumps(context), "sha256:" + "09" * 32, binding_bytes
    )
    tape_b = None
    try:
        captured_b, completions_b = _play_to_terminal(pool_b, _lanes(pool_b))
        assert captured_b == captured_a
        for entry in completions_b:
            assert entry["terminal"] and entry["outcome"] is not None
        pool_b.quiesce()
        assert pool_b.poll()["poison"] is None
        tape_b = pool_b.drain_tape(binding_bytes)
        assert tape_b != tape_a
        status = pool_b.poll()
        assert status["captured_rows"] == captured_b
        assert status["terminal_games"] == 2
        for call in (
            lambda: pool_b.drain_tape(binding_bytes),
            lambda: pool_b.snapshot(),
            lambda: pool_b.export_tape(),
            lambda: pool_b.resume_admission(),
            lambda: pool_b.admit_next_fill([]),
        ):
            with pytest.raises(ValueError, match="already drained"):
                call()
    finally:
        pool_b.close()

    # Identical frozen rows/features: same terminal inputs and advantages.
    desc_a, actor_a, game_a, value_a, utility_a = rl_bridge.terminal_inputs(tape_a)
    desc_b, actor_b, game_b, value_b, utility_b = rl_bridge.terminal_inputs(tape_b)
    assert desc_a["row_count"] == desc_b["row_count"] == captured_a
    assert (actor_a, game_a, value_a, utility_a) == (actor_b, game_b, value_b, utility_b)
    actors = np.frombuffer(actor_a, dtype="<i8")
    games = np.frombuffer(game_a, dtype="<i8")
    values = np.frombuffer(value_a, dtype="<f4").reshape(-1, 4)
    utilities = np.frombuffer(utility_a, dtype="<f8").reshape(-1, 4)
    advantages = utilities.astype(np.float32)[games, actors] - values[np.arange(captured_a), actors]
    assert (
        rl_bridge.freeze_terminal_advantages(tape_a, advantages.tobytes())["rows_frozen"]
        == captured_a
    )
    assert (
        rl_bridge.freeze_terminal_advantages(tape_b, advantages.tobytes())["rows_frozen"]
        == captured_a
    )

    # Incomplete cohorts reject before any move; the pool stays usable.
    pool_c = rl_native.PyNativePool.open_live(
        json.dumps(profile), json.dumps(context), "sha256:" + "09" * 32, binding_bytes
    )
    tape_c = None
    try:
        lanes_c = _lanes(pool_c)
        partial = 0
        for _ in range(3):
            for slot in lanes_c:
                pool_c.request_ready_fill(slot)
            while (lease := pool_c.peek_ready()) is not None:
                count, _ = _commit_first_legal(pool_c, lanes_c[lease.slot_id][1], lease)
                partial += count
        pool_c.quiesce()
        before = pool_c.poll()["captured_rows"]
        assert before > 0
        with pytest.raises(ValueError):
            pool_c.drain_tape(binding_bytes)
        assert pool_c.poll()["captured_rows"] == before
        pool_c.resume_admission()
        captured_c, _ = _play_to_terminal(pool_c, lanes_c)
        pool_c.quiesce()
        tape_c = pool_c.drain_tape(binding_bytes)
        assert rl_bridge.terminal_inputs(tape_c)[0]["row_count"] == partial + captured_c
    finally:
        pool_c.close()

    # Live binding mismatches reject at open.
    bad = dict(binding, generation=1)
    with pytest.raises(ValueError):
        rl_native.PyNativePool.open_live(
            json.dumps(profile),
            json.dumps(context),
            "sha256:" + "09" * 32,
            json.dumps(bad).encode(),
        )

    # Dropped tapes serve nothing.
    assert native_artifact.drop_tape(tape_a) is None
    assert native_artifact.drop_tape(tape_b) is None
    assert native_artifact.drop_tape(tape_c) is None
    with pytest.raises(ValueError):
        rl_bridge.terminal_inputs(tape_b)


def test_close_diagnostic_respects_python_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HYDRA2_RL_VIEW_STATS", "1")
    pool = _open_pool(_profile(_schedule(7012, "rl-close-diagnostic"), [0, 1, 2, 3]), _context())
    pool.close()
    report = capsys.readouterr().err
    assert report.startswith("rl-view-stats hits=")
    assert " misses=" in report and report.endswith("\n")
    pool.close()
    assert capsys.readouterr().err == ""
    with pytest.raises(ValueError, match="closed"):
        pool.snapshot()


def test_close_diagnostic_failure_keeps_pool_closed(monkeypatch) -> None:
    import sys

    class BrokenStderr:
        def write(self, text: str) -> None:
            raise OSError("diagnostic sink failed")

    monkeypatch.setenv("HYDRA2_RL_VIEW_STATS", "1")
    pool = _open_pool(
        _profile(_schedule(7013, "rl-close-output-failure"), [0, 1, 2, 3]), _context()
    )
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", BrokenStderr())
        with pytest.raises(OSError, match="diagnostic sink failed"):
            pool.close()
    pool.close()
    with pytest.raises(ValueError, match="closed"):
        pool.snapshot()
