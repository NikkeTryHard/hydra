"""WP-03B gate: wall-block aggregation, invalid-block policy, telemetry schema."""

from __future__ import annotations

import pytest

from hydra2.contracts.common import ContractError
from hydra2.eval.blocks import (
    BlockTolerance,
    WallBlock,
    aggregate_blocks,
    aggregate_wall_block,
)
from hydra2.eval.telemetry import (
    REQUIRED_CORE_FIELDS,
    TELEMETRY_FIELDS,
    ResourceTelemetry,
    TelemetryTolerance,
    block_missing_telemetry_report,
    make_resource_telemetry,
    telemetry_invalid_reason,
)

pytestmark = pytest.mark.contract_package("WP-03B")


def _telemetry(**overrides: object) -> ResourceTelemetry:
    base: dict[str, object] = {
        "mode": "cuda_eager",
        "wall_id": "w-1",
        "case_id": None,
        "candidate_spec_hash": "sha256:" + "11" * 32,
        "hardware_hash": "sha256:" + "22" * 32,
        "environment_hash": "sha256:" + "33" * 32,
        "cold_start": False,
        "synchronized_elapsed_ms": 12.5,
        "model_calls": 3,
        "exact_transitions": 40,
        "particles": 0,
        "fallback_used": False,
        "timeout": False,
        "illegal_action": False,
        "cuda_peak_allocated_bytes": 1024,
        "cuda_peak_reserved_bytes": 2048,
        "host_peak_bytes": None,
        "energy_joules": None,
        "graph_breaks": None,
        "recompiles": None,
        "invalid_reason": None,
    }
    base.update(overrides)
    return make_resource_telemetry(**base)


def test_telemetry_field_order_matches_spec() -> None:
    """SPEC 18.2 field identity: exact names in exact order."""
    assert TELEMETRY_FIELDS == (
        "mode",
        "wall_id",
        "case_id",
        "candidate_spec_hash",
        "hardware_hash",
        "environment_hash",
        "cold_start",
        "synchronized_elapsed_ms",
        "model_calls",
        "exact_transitions",
        "particles",
        "fallback_used",
        "timeout",
        "illegal_action",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "host_peak_bytes",
        "energy_joules",
        "graph_breaks",
        "recompiles",
        "invalid_reason",
    )


def test_telemetry_validation_rejects_bad_rows() -> None:
    with pytest.raises(TypeError, match="mode"):
        _telemetry(mode="")
    with pytest.raises(Exception, match="must match"):
        _telemetry(candidate_spec_hash="md5:nope")
    with pytest.raises(TypeError, match="model_calls"):
        _telemetry(model_calls=-1)
    with pytest.raises(TypeError, match="synchronized_elapsed_ms"):
        _telemetry(synchronized_elapsed_ms=float("nan"))
    with pytest.raises(TypeError, match="fallback_used"):
        _telemetry(fallback_used="no")
    with pytest.raises(TypeError, match="unknown"):
        _telemetry(nonsense=1)


def test_missing_required_telemetry_invalidates_and_is_never_imputed() -> None:
    tolerance = TelemetryTolerance()
    row = _telemetry(cuda_peak_allocated_bytes=None)
    reason = telemetry_invalid_reason(row, tolerance)
    assert reason is not None and "cuda_peak_allocated_bytes" in reason
    # Tolerance may never excuse core fields at all.
    with pytest.raises(ValueError, match="required"):
        TelemetryTolerance(allow_missing=frozenset({"model_calls"}))
    # Mode-required fields cannot be excused either (checked per mode).
    lenient = TelemetryTolerance(allow_missing=frozenset({"cuda_peak_allocated_bytes"}))
    with pytest.raises(ValueError, match="mode-required"):
        lenient.required_for("cuda_eager")


def test_mode_extras_become_required_only_for_that_mode() -> None:
    eager = _telemetry(mode="reference_eager_cpu", cuda_peak_allocated_bytes=None)
    assert telemetry_invalid_reason(eager, TelemetryTolerance()) is None


def test_row_marked_invalid_surfaces_verbatim() -> None:
    row = _telemetry(invalid_reason="engine panicked")
    reason = telemetry_invalid_reason(row, TelemetryTolerance())
    assert reason == "row marked invalid: engine panicked"


def test_block_aggregation_collapses_games_to_one_value() -> None:
    """Games inside a wall are NOT independent units: one number per block."""
    block = WallBlock(wall_id="w-1", game_ids=("g1", "g2", "g3"), contrasts=(1.0, 2.0, 6.0))
    assert aggregate_wall_block(block) == pytest.approx(3.0)
    with pytest.raises(ContractError):
        WallBlock(wall_id="w-2", game_ids=("g1",), contrasts=())
    with pytest.raises(ContractError):
        WallBlock(wall_id="w-3", game_ids=("g1", "g2"), contrasts=(1.0,))
    with pytest.raises(ContractError):
        WallBlock(wall_id="w-4", game_ids=("g1",), contrasts=(float("inf"),))


def test_invalid_block_policy_excludes_and_reports() -> None:
    blocks = (
        WallBlock(wall_id="w-good", game_ids=("ga",), contrasts=(0.5,)),
        WallBlock(
            wall_id="w-fallback",
            game_ids=("gb",),
            contrasts=(9.9,),
        ),
        WallBlock(wall_id="w-empty", game_ids=(), contrasts=()),
    )
    telemetry = {
        "ga": _telemetry(),
        "gb": _telemetry(fallback_used=True),
    }
    result = aggregate_blocks(blocks, telemetry_by_game=telemetry, tolerance=BlockTolerance())
    assert [wall for wall, _ in result.valid] == ["w-good"]
    assert result.valid[0][1] == pytest.approx(0.5)
    assert [(item.wall_id, item.reason) for item in result.excluded] == [
        ("w-empty", "empty_block"),
        ("w-fallback", "fallback_used"),
    ]
    assert "used fallback" in result.excluded[1].detail

    lenient = BlockTolerance(allow_fallback_used=True)
    lenient_result = aggregate_blocks(blocks, telemetry_by_game=telemetry, tolerance=lenient)
    assert [wall for wall, _ in lenient_result.valid] == ["w-fallback", "w-good"]


def test_missing_telemetry_row_excludes_block_with_report() -> None:
    blocks = (WallBlock(wall_id="w-x", game_ids=("gx",), contrasts=(1.0,)),)
    report = block_missing_telemetry_report({}, TelemetryTolerance())
    assert report == {}
    result = aggregate_blocks(blocks, telemetry_by_game={}, tolerance=BlockTolerance())
    assert result.valid == ()
    assert result.excluded[0].reason == "missing_telemetry"
    per_game = block_missing_telemetry_report(
        {"gy": _telemetry(invalid_reason="bad")}, TelemetryTolerance()
    )
    assert per_game == {"gy": "row marked invalid: bad"}
    assert set(REQUIRED_CORE_FIELDS) <= set(TELEMETRY_FIELDS)
