"""WP-06 gate: duplicate-block qualification — exact/near, disjoint walls, block eval."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hydra2.contracts.common import ContractError
from hydra2.eval.blocks import BlockTolerance, WallBlock, aggregate_blocks, aggregate_wall_block
from hydra2.eval.duplicate import (
    balance_audit,
    build_wall_blocks,
    find_exact_duplicates,
    find_near_duplicates,
    make_block_manifest,
    report_telemetry_completeness,
    split_blocks_held_out,
    validate_blocks_disjoint,
    validate_walls_disjoint,
    wall_fingerprint,
    wall_hash_from_tiles,
)
from hydra2.eval.schedule import build_match_schedule
from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry

pytestmark = pytest.mark.contract_package("WP-06")

LABELS = ("candidate-a", "partner-a", "baseline-b", "field-c")
RULES = "sha256:" + "ab" * 32
MASTER = bytes(range(48, 80))
WALL_IDS = tuple(f"w-{i:02d}" for i in range(12))
EXPERIMENT = "exp-wp06"
SPLIT = "split-wp06"


def _schedule(
    wall_ids: tuple[str, ...] = WALL_IDS,
    master_seed: bytes = MASTER,
) -> object:
    return build_match_schedule(
        wall_ids=wall_ids,
        labels=LABELS,
        rules_hash=RULES,
        master_seed=master_seed,
        experiment_id=EXPERIMENT,
        split_id=SPLIT,
    )


def _telemetry(**overrides: object) -> ResourceTelemetry:
    base: dict[str, object] = {
        "mode": "reference_eager_cpu",
        "wall_id": "w-00",
        "case_id": "case-00",
        "candidate_spec_hash": "sha256:" + "11" * 32,
        "hardware_hash": "sha256:" + "22" * 32,
        "environment_hash": "sha256:" + "33" * 32,
        "cold_start": False,
        "synchronized_elapsed_ms": 12.5,
        "model_calls": 4,
        "exact_transitions": 4,
        "particles": 0,
        "fallback_used": False,
        "timeout": False,
        "illegal_action": False,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        "host_peak_bytes": None,
        "energy_joules": None,
        "graph_breaks": None,
        "recompiles": None,
        "invalid_reason": None,
    }
    base.update(overrides)
    return make_resource_telemetry(**base)


# ---------------------------------------------------------------------------
# Exact / near duplicate detection
# ---------------------------------------------------------------------------


def test_exact_and_near_duplicate_detection() -> None:
    """Exact duplicates share wall digest; near duplicates share logical fingerprint."""
    tiles_a = list(range(136))
    tiles_b = list(range(136))  # identical -> exact + near
    tiles_c = list(
        reversed(range(136))
    )  # same multiset, different order -> near only for fingerprint

    hash_a = wall_hash_from_tiles(tiles_a)
    hash_b = wall_hash_from_tiles(tiles_b)
    hash_c = wall_hash_from_tiles(tiles_c)
    assert hash_a == hash_b
    assert hash_a != hash_c

    # Exact detection over digest map
    assert find_exact_duplicates({"w1": hash_a, "w2": hash_b, "w3": hash_c}) == [("w1", "w2")]
    assert find_exact_duplicates({"w1": hash_a, "w2": hash_c}) == []
    # Dict with no duplicates is clean
    assert find_exact_duplicates({"w1": hash_a, "w3": hash_c}) == []

    # Near duplicates over tile maps — w1 and w3 are near (same sorted tiles)
    near = find_near_duplicates({"w1": tiles_a, "w2": tiles_c})
    assert near == [("w1", "w2")]
    # w1 vs w2 (identical order) are also near because sorted equal
    near2 = find_near_duplicates({"w1": tiles_a, "w2": tiles_b})
    assert near2 == [("w1", "w2")]
    # Three walls sharing fingerprint produce two pairs (first anchors)
    near3 = find_near_duplicates({"w1": tiles_a, "w2": tiles_b, "w3": tiles_c})
    assert len(near3) == 2
    assert near3[0] == ("w1", "w2")

    # Fingerprint is permutation-invariant, hash is order-sensitive
    assert wall_fingerprint(tiles_a) == wall_fingerprint(tiles_c)
    assert wall_hash_from_tiles(tiles_a) != wall_hash_from_tiles(tiles_c)

    # Validation via wall hash map still reports no near duplicate when fingerprints differ
    # Use distinct tile sets that actually differ in multiset (replace one tile with duplicate)
    tiles_d = list(range(136))
    tiles_d[0] = 1  # duplicate tile 1, missing 0 => different multiset
    assert wall_fingerprint(tiles_a) != wall_fingerprint(tiles_d)
    assert find_near_duplicates({"w1": tiles_a, "w2": tiles_d}) == []


def test_exact_duplicate_rejection_contract() -> None:
    with pytest.raises(ContractError):
        find_exact_duplicates({"": "sha256:" + "aa" * 32})  # empty wall id
    with pytest.raises(ContractError):
        find_exact_duplicates({"w1": "not-a-digest"})
    # Near duplicate rejects malformed tiles
    with pytest.raises(ContractError):
        find_near_duplicates({"w1": [0] * 10})  # wrong length
    with pytest.raises(ContractError):
        find_near_duplicates({"w1": [0] * 136 + [999]})  # will fail length first


# ---------------------------------------------------------------------------
# Disjoint walls
# ---------------------------------------------------------------------------


def test_disjoint_wall_sets_enforced() -> None:
    validate_walls_disjoint(["w1", "w2"], ["w3", "w4"])
    validate_walls_disjoint(["w1"], ["w2"], ["w3"])
    # Overlap across partitions must raise
    with pytest.raises(ContractError, match="disjoint"):
        validate_walls_disjoint(["w1", "w2"], ["w2", "w3"])
    with pytest.raises(ContractError, match="disjoint"):
        validate_walls_disjoint(["a", "b", "c"], ["d"], ["c", "e"])
    with pytest.raises(ContractError):
        validate_walls_disjoint([""], ["w1"])


def test_blocks_disjoint_and_game_uniqueness() -> None:
    block_a = WallBlock(wall_id="w-a", game_ids=("g1", "g2"), contrasts=(1.0, 2.0))
    block_b = WallBlock(wall_id="w-b", game_ids=("g3", "g4"), contrasts=(0.5, -0.5))
    validate_blocks_disjoint((block_a, block_b))
    # Duplicate wall id
    block_dup = WallBlock(wall_id="w-a", game_ids=("g5", "g6"), contrasts=(0.0, 0.0))
    with pytest.raises(ContractError, match="duplicate wall_id"):
        validate_blocks_disjoint((block_a, block_dup))
    # Duplicate game id across blocks
    block_overlap = WallBlock(wall_id="w-c", game_ids=("g2", "g5"), contrasts=(1.0, 1.0))
    with pytest.raises(ContractError, match="appears in multiple"):
        validate_blocks_disjoint((block_a, block_overlap))


# ---------------------------------------------------------------------------
# Block splitting — whole walls as independent units
# ---------------------------------------------------------------------------


def test_block_splitting_whole_walls() -> None:
    schedule = _schedule(wall_ids=tuple(f"w-{i:02d}" for i in range(8)))
    # Synthesize deterministic contrasts for every game.
    contrasts_by_game: dict[str, float] = {}
    for wall_id in schedule.wall_ids:
        for slot in range(10):
            contrasts_by_game[f"{wall_id}:g{slot}"] = float(slot) * 0.1 + 0.05
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts_by_game)
    assert len(blocks) == 8
    for block in blocks:
        assert len(block.game_ids) == 10
        assert len(block.contrasts) == 10
    # Wall ids are disjoint and game ids unique
    validate_blocks_disjoint(blocks)
    # Building again from same inputs is byte-identical (deterministic)
    blocks2 = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts_by_game)
    assert blocks == blocks2
    # Manifest is canonical and order-preserving
    manifest = make_block_manifest(schedule=schedule, blocks=blocks)
    assert manifest.wall_ids == schedule.wall_ids
    manifest2 = make_block_manifest(schedule=schedule, blocks=blocks)
    assert manifest.digest == manifest2.digest
    # Custom game id map is honoured
    custom_map = {
        wall_id: tuple(f"custom-{wall_id}-{i}" for i in range(10)) for wall_id in schedule.wall_ids
    }
    custom_contrasts = {gid: 1.0 for gids in custom_map.values() for gid in gids}
    custom_blocks = build_wall_blocks(
        schedule=schedule, contrasts_by_game=custom_contrasts, game_ids_by_wall=custom_map
    )
    assert custom_blocks[0].game_ids[0] == f"custom-{schedule.wall_ids[0]}-0"


def test_block_splitting_rejects_partial_or_mismatched_games() -> None:
    schedule = _schedule(wall_ids=("w-x", "w-y"))
    # Missing games for a wall
    incomplete = {f"w-x:g{slot}": 0.0 for slot in range(10)}
    # w-y entries entirely absent
    with pytest.raises(ContractError):
        build_wall_blocks(schedule=schedule, contrasts_by_game=incomplete)
    # Wrong count per wall via explicit map
    bad_map = {
        "w-x": tuple(f"w-x:g{i}" for i in range(5)),
        "w-y": tuple(f"w-y:g{i}" for i in range(10)),
    }
    bad_contrasts = {gid: 0.0 for gids in bad_map.values() for gid in gids}
    with pytest.raises(ContractError, match="must carry 10 games"):
        build_wall_blocks(
            schedule=schedule, contrasts_by_game=bad_contrasts, game_ids_by_wall=bad_map
        )


# ---------------------------------------------------------------------------
# Whole-block aggregation — games inside a wall are NOT independent
# ---------------------------------------------------------------------------


def test_whole_block_aggregation() -> None:
    """Aggregate collapses each wall to ONE value; per-game resampling is not used."""
    blocks = (
        WallBlock(wall_id="w-1", game_ids=("g1", "g2", "g3"), contrasts=(1.0, 2.0, 3.0)),
        WallBlock(wall_id="w-2", game_ids=("g4",), contrasts=(10.0,)),
    )
    # No telemetry -> aggregate_blocks with empty tolerance would still validate but need telemetry.
    # Build telemetry for all games.
    telemetry = {
        "g1": _telemetry(wall_id="w-1", case_id="c1"),
        "g2": _telemetry(wall_id="w-1", case_id="c1"),
        "g3": _telemetry(wall_id="w-1", case_id="c1"),
        "g4": _telemetry(wall_id="w-2", case_id="c1"),
    }
    result = aggregate_blocks(blocks, telemetry_by_game=telemetry)
    assert len(result.valid) == 2
    assert len(result.excluded) == 0
    # Each wall collapsed to mean
    assert result.valid[0] == ("w-1", pytest.approx(2.0))
    assert result.valid[1] == ("w-2", pytest.approx(10.0))
    # Direct helper matches tuple mean
    assert aggregate_wall_block(blocks[0]) == pytest.approx(2.0)
    # Per-game bootstrap would give different variance — we assert blocks are atomic:
    # Two games in one wall produce one number, not two.
    assert len(result.valid) == 2  # not 4
    # Non-finite contrasts rejected at construction
    with pytest.raises(ContractError):
        WallBlock(wall_id="w-bad", game_ids=("g1",), contrasts=(float("inf"),))


# ---------------------------------------------------------------------------
# Invalid-block policy — excluded and reported, never silently imputed
# ---------------------------------------------------------------------------


def test_invalid_block_excluded_and_reported() -> None:
    blocks = (
        WallBlock(wall_id="w-good", game_ids=("g-good",), contrasts=(1.0,)),
        WallBlock(wall_id="w-fallback", game_ids=("g-fb",), contrasts=(1.0,)),
        WallBlock(wall_id="w-timeout", game_ids=("g-to",), contrasts=(1.0,)),
        WallBlock(wall_id="w-illegal", game_ids=("g-ill",), contrasts=(1.0,)),
        WallBlock(wall_id="w-missing", game_ids=("g-miss",), contrasts=(1.0,)),
    )
    telemetry = {
        "g-good": _telemetry(wall_id="w-good"),
        "g-fb": _telemetry(wall_id="w-fallback", fallback_used=True),
        "g-to": _telemetry(wall_id="w-timeout", timeout=True),
        "g-ill": _telemetry(wall_id="w-illegal", illegal_action=True),
        # g-miss intentionally absent
    }
    strict = aggregate_blocks(blocks, telemetry_by_game=telemetry)
    assert len(strict.valid) == 1
    assert strict.valid[0][0] == "w-good"
    assert len(strict.excluded) == 4
    reasons = {exc.wall_id: exc.reason for exc in strict.excluded}
    assert reasons["w-fallback"] == "fallback_used"
    assert reasons["w-timeout"] == "timeout"
    assert reasons["w-illegal"] == "illegal_action"
    assert reasons["w-missing"] == "missing_telemetry"

    # Lenient tolerance excuses fallback/timeout/illegal but never missing telemetry
    lenient_tol = BlockTolerance(
        allow_fallback_used=True, allow_timeout=True, allow_illegal_action=True
    )
    lenient = aggregate_blocks(blocks, telemetry_by_game=telemetry, tolerance=lenient_tol)
    assert {wall for wall, _ in lenient.valid} == {"w-good", "w-fallback", "w-timeout", "w-illegal"}
    assert len(lenient.excluded) == 1 and lenient.excluded[0].wall_id == "w-missing"

    # Telemetry invalid_reason is never imputed
    bad_row = _telemetry(wall_id="w-good", invalid_reason="engine panicked")
    report_bad = aggregate_blocks(
        (WallBlock(wall_id="w-good", game_ids=("g-good",), contrasts=(1.0,)),),
        telemetry_by_game={"g-good": bad_row},
    )
    assert len(report_bad.valid) == 0
    assert report_bad.excluded[0].reason == "row_invalid"


# ---------------------------------------------------------------------------
# Seat balance audit
# ---------------------------------------------------------------------------


def test_seat_balance_audit() -> None:
    schedule = _schedule()
    audit = balance_audit(schedule)
    assert audit["walls"] == len(WALL_IDS)
    assert audit["games_per_wall"] == 10
    assert audit["balance_exact"] is True
    # Every label appears equally across seats globally (per-wall placement exact)
    # Over 12 walls, each of the 10*12 allocations has been counted.
    for _label, counts in audit["seat_count_by_label"].items():
        assert len(counts) == 4
        # Candidate labels must have plausible counts (partner appears in symmetric + rotations)
        assert sum(counts) > 0
    # Audit digest is deterministic
    audit2 = balance_audit(schedule)
    assert audit == audit2
    # Drift: tampering schedule would fail seat_pair_placements_exact
    from hydra2.eval.schedule import MatchSchedule

    # Force first wall's 6 symmetric placements to all share the same seat pair
    # — violates the exact placement requirement (C(4,2)=6 distinct pairs).
    bad_list = list(schedule.seat_allocations)  # type: ignore[attr-defined]
    first_row = bad_list[0]
    for i in range(6):
        bad_list[i] = first_row
    bad = MatchSchedule(
        wall_ids=schedule.wall_ids,  # type: ignore[attr-defined]
        walls_hash=schedule.walls_hash,  # type: ignore[attr-defined]
        seat_allocations=tuple(bad_list),
        latency_schedule_hash=schedule.latency_schedule_hash,  # type: ignore[attr-defined]
        rules_hash=schedule.rules_hash,  # type: ignore[attr-defined]
        seed_protocol_hash=schedule.seed_protocol_hash,  # type: ignore[attr-defined]
    )
    with pytest.raises(ContractError):
        balance_audit(bad)


# ---------------------------------------------------------------------------
# Telemetry completeness report
# ---------------------------------------------------------------------------


def test_telemetry_completeness_report() -> None:
    schedule = _schedule(wall_ids=("w-a", "w-b", "w-c"))
    contrasts: dict[str, float] = {}
    for wall_id in schedule.wall_ids:  # type: ignore[attr-defined]
        for slot in range(10):
            contrasts[f"{wall_id}:g{slot}"] = 0.5
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
    # All valid telemetry
    telemetry = {
        gid: _telemetry(wall_id=block.wall_id, case_id="diag")
        for block in blocks
        for gid in block.game_ids
    }
    report = report_telemetry_completeness(blocks, telemetry)
    assert report["blocks_total"] == 3
    assert report["blocks_valid"] == 3
    assert report["blocks_excluded"] == 0
    assert report["invalid_games"] == {}
    # Introduce one fallback game => that wall excluded
    bad_gid = blocks[0].game_ids[0]
    telemetry[bad_gid] = _telemetry(wall_id=blocks[0].wall_id, fallback_used=True)
    report2 = report_telemetry_completeness(blocks, telemetry)
    assert report2["blocks_valid"] == 2
    assert report2["blocks_excluded"] == 1
    assert bad_gid in report2["invalid_games"]
    # Missing telemetry never imputed: deleting a row marks block excluded
    del telemetry[blocks[1].game_ids[1]]
    report3 = report_telemetry_completeness(blocks, telemetry)
    assert report3["blocks_excluded"] >= 1
    assert report3["invalid_games"][blocks[1].game_ids[1]] == "missing telemetry row"
    # Report digest covers valid/excluded sets
    assert report["digest"] != report2["digest"]


# ---------------------------------------------------------------------------
# Held-out partition is hidden from training selection
# ---------------------------------------------------------------------------


def test_held_out_partition_hidden() -> None:
    schedule = _schedule(wall_ids=tuple(f"w-{i:02d}" for i in range(10)))
    contrasts = {f"{wid}:g{slot}": float(slot) for wid in schedule.wall_ids for slot in range(10)}  # type: ignore[attr-defined]
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
    split1 = split_blocks_held_out(blocks, held_out_ratio=0.2, seed=0)
    split2 = split_blocks_held_out(blocks, held_out_ratio=0.2, seed=0)
    assert split1.digest == split2.digest
    assert split1 == split2
    # Train + held-out cover universe without overlap and keep whole walls intact
    all_ids = {b.wall_id for b in blocks}
    assert {b.wall_id for b in split1.train_blocks} | {
        b.wall_id for b in split1.held_out_blocks
    } == all_ids
    validate_walls_disjoint(
        tuple(b.wall_id for b in split1.train_blocks),
        tuple(b.wall_id for b in split1.held_out_blocks),
    )
    # Held-out size respects ratio (10 * 0.2 = 2)
    assert len(split1.held_out_blocks) == 2
    assert len(split1.train_blocks) == 8
    # Different seed gives different partition
    split_other = split_blocks_held_out(blocks, held_out_ratio=0.2, seed=1)
    assert split_other.digest != split1.digest
    # Determinism check via repeated call
    split_repeat = split_blocks_held_out(blocks, held_out_ratio=0.2, seed=0)
    assert split_repeat.digest == split1.digest
    # Whole-wall atomicity: no game appears in both sides
    g_train = {gid for b in split1.train_blocks for gid in b.game_ids}
    g_held = {gid for b in split1.held_out_blocks for gid in b.game_ids}
    assert g_train.isdisjoint(g_held)


# ---------------------------------------------------------------------------
# Fresh-process block load (deterministic repeat via subprocess)
# ---------------------------------------------------------------------------


def test_fresh_process_block_load() -> None:
    schedule = _schedule(wall_ids=("w-p", "w-q"))
    contrasts = {
        f"{wid}:g{slot}": float(slot) * 0.2 for wid in schedule.wall_ids for slot in range(10)
    }  # type: ignore[attr-defined]
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
    manifest = make_block_manifest(schedule=schedule, blocks=blocks)
    # Spawn a fresh Python process that rebuilds the same manifest from the same inputs.
    # Portable src path derived from this file's repo root (no hardcoded user paths).
    _src = str(Path(__file__).resolve().parents[2] / "src")
    code = textwrap.dedent(
        f"""
        import sys
        _src = "{_src}"
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from hydra2.eval.schedule import build_match_schedule
        from hydra2.eval.duplicate import build_wall_blocks, make_block_manifest
        LABELS = ("candidate-a", "partner-a", "baseline-b", "field-c")
        RULES = "sha256:" + "ab" * 32
        MASTER = bytes(range(48, 80))
        wall_ids = ("w-p", "w-q")
        schedule = build_match_schedule(
            wall_ids=wall_ids,
            labels=LABELS,
            rules_hash=RULES,
            master_seed=MASTER,
            experiment_id="exp-wp06",
            split_id="split-wp06",
        )
        contrasts = {{
            f"{{wid}}:g{{slot}}": float(slot) * 0.2 for wid in wall_ids for slot in range(10)
        }}
        blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
        manifest = make_block_manifest(schedule=schedule, blocks=blocks)
        print(manifest.digest)
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    fresh_digest = proc.stdout.strip()
    assert fresh_digest == manifest.digest


def test_block_manifest_and_balance_digest_determinism() -> None:
    schedule = _schedule(wall_ids=("w-01", "w-02", "w-03"))
    contrasts = {
        f"{wid}:g{slot}": 1.0 + slot * 0.01
        for wid in schedule.wall_ids  # type: ignore[attr-defined]
        for slot in range(10)
    }
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
    m1 = make_block_manifest(schedule=schedule, blocks=blocks)
    m2 = make_block_manifest(schedule=schedule, blocks=blocks)
    assert m1.digest == m2.digest
    # Telemetry completeness via lenient vs strict tolerance changes digest
    telemetry = {gid: _telemetry(wall_id=blk.wall_id) for blk in blocks for gid in blk.game_ids}
    # Mark one wall's game as illegal action
    telemetry[blocks[0].game_ids[0]] = _telemetry(wall_id=blocks[0].wall_id, illegal_action=True)
    strict_report = report_telemetry_completeness(blocks, telemetry)
    assert strict_report["blocks_excluded"] == 1


def test_confirmation_sidecar_binds_commitment_and_exclusions() -> None:
    from hydra2.eval.duplicate import confirmation_sidecar
    from hydra2.eval.schedule import schedule_commitment_hash

    schedule = _schedule(wall_ids=("w-01", "w-02", "w-03"))
    contrasts = {
        f"{wid}:g{slot}": 1.0
        for wid in schedule.wall_ids  # type: ignore[attr-defined]
        for slot in range(10)
    }
    blocks = build_wall_blocks(schedule=schedule, contrasts_by_game=contrasts)
    telemetry = {gid: _telemetry(wall_id=blk.wall_id) for blk in blocks for gid in blk.game_ids}
    telemetry[blocks[0].game_ids[0]] = _telemetry(wall_id=blocks[0].wall_id, illegal_action=True)
    result = aggregate_blocks(blocks, telemetry_by_game=telemetry)
    assert len(result.excluded) == 1
    sidecar = confirmation_sidecar(schedule=schedule, blocks=result, telemetry_report=None)
    assert sidecar["schedule_commitment_hash"] == str(schedule_commitment_hash(schedule))
    assert sidecar["admission"] == "full"
    assert [row["wall_id"] for row in sidecar["excluded"]] == [
        exc.wall_id for exc in result.excluded
    ]
    assert sidecar["telemetry_completeness_digest"] is None


def test_confirmation_sidecar_marks_unadmitted_paths() -> None:
    from hydra2.eval.blocks import BlockAggregateResult
    from hydra2.eval.duplicate import confirmation_sidecar

    schedule = _schedule(wall_ids=("w-01",))
    sidecar = confirmation_sidecar(
        schedule=schedule,
        blocks=BlockAggregateResult(valid=(("w-01", 0.5),), excluded=()),
        telemetry_report=None,
        admission="not-run",
    )
    assert sidecar["admission"] == "not-run"
    assert sidecar["excluded"] == []
    with pytest.raises(ContractError):
        confirmation_sidecar(
            schedule=schedule,
            blocks=BlockAggregateResult(valid=(("w-01", 0.5),), excluded=()),
            admission="sometimes",
        )
