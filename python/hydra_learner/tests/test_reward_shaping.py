from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from hydra_learner.reward_shaping import (
    GRP_PERM_TABLE,
    PHI_DEFINITION_GRP_EXPECTED_U_A_V1,
    REQUIRED_GATE_RESULT_CATEGORIES,
    STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1,
    THRESHOLD_CONTRACT_VERSION_V1,
    VALIDATION_THRESHOLDS_ABSENT_REASON,
    RewardShapingConfig,
    RewardShapingValidationReport,
    apply_pbrs_reward,
    compute_grp_metrics,
    grp_expected_phi,
    normalize_reward_shaping_metadata,
    telescoping_terminal_residual,
    truncation_shaping_residual,
)
from hydra_learner.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA, PLACEMENT_UTILITY_DEFAULT


def test_grp_phi_one_hot_mixed_logits_probs_all_actors() -> None:
    probs = torch.zeros(4, 24, dtype=torch.float32)
    for actor in range(4):
        class_index = next(index for index, perm in enumerate(GRP_PERM_TABLE) if perm[2] == actor)
        probs[actor, class_index] = 1.0
    actor = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    phi = grp_expected_phi(grp_probs=probs, actor=actor)

    torch.testing.assert_close(phi, torch.full((4,), PLACEMENT_UTILITY_DEFAULT[2]))

    mixed = torch.zeros(1, 24, dtype=torch.float32)
    class_first = next(index for index, perm in enumerate(GRP_PERM_TABLE) if perm[0] == 0)
    class_last = next(index for index, perm in enumerate(GRP_PERM_TABLE) if perm[3] == 0)
    mixed[0, class_first] = 0.25
    mixed[0, class_last] = 0.75
    expected = 0.25 * PLACEMENT_UTILITY_DEFAULT[0] + 0.75 * PLACEMENT_UTILITY_DEFAULT[3]
    torch.testing.assert_close(grp_expected_phi(grp_probs=mixed, actor=torch.tensor([0])), torch.tensor([expected]))

    logits = torch.full((1, 24), -80.0, dtype=torch.float32)
    logits[0, class_first] = 80.0
    torch.testing.assert_close(grp_expected_phi(grp_logits=logits, actor=torch.tensor([0])), torch.tensor([1.0]))


def test_terminal_phi_forces_zero_and_validation_rejects_bad_inputs() -> None:
    logits = torch.randn(2, 24, dtype=torch.float32)
    actor = torch.tensor([0, 1], dtype=torch.int64)
    terminal = torch.tensor([True, False])

    phi = grp_expected_phi(grp_logits=logits, actor=actor, terminal=terminal)

    assert float(phi[0]) == 0.0
    with pytest.raises(ValueError, match="shape"):
        grp_expected_phi(grp_logits=torch.zeros(1, 23), actor=torch.tensor([0]))
    with pytest.raises(ValueError, match="actor"):
        grp_expected_phi(grp_logits=torch.zeros(1, 24), actor=torch.tensor([4]))
    bad_probs = torch.full((1, 24), 1.0 / 23.0)
    with pytest.raises(AssertionError, match="sum to one"):
        grp_expected_phi(grp_probs=bad_probs, actor=torch.tensor([0]))


def test_validation_report_strict_json_thresholds_absent_and_rejects_bad_payload(tmp_path: Path) -> None:
    report = RewardShapingValidationReport.thresholds_absent(
        predictor_checkpoint_id="candidate", source_identity={"source": "synthetic"}
    )
    path = tmp_path / "report.json"

    report.write(path)
    raw = path.read_text(encoding="utf-8")
    loaded = RewardShapingValidationReport.load(path)

    assert json.loads(raw)["reason"] == VALIDATION_THRESHOLDS_ABSENT_REASON
    assert loaded.validated is False
    assert loaded.beta_activation_allowed is False
    assert "NaN" not in raw

    payload = report.to_dict()
    del payload["metrics"]
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        RewardShapingValidationReport.load(bad)

    payload = report.to_dict()
    payload["metrics"] = {"nll": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        RewardShapingValidationReport(**_ctor_payload(payload)).to_dict()


def test_validation_report_rejects_hidden_private_oracle_marker() -> None:
    payload = RewardShapingValidationReport.thresholds_absent().to_dict()
    payload["metrics"] = {"hidden_private_phi": True}
    with pytest.raises(ValueError, match="forbidden"):
        RewardShapingValidationReport(**_ctor_payload(payload)).to_dict()


def test_activation_gates_fail_closed_and_synthetic_validated_match_accepts(tmp_path: Path) -> None:
    RewardShapingConfig().validate()
    with pytest.raises(ValueError, match="enabled=false"):
        RewardShapingConfig(enabled=False, pbrs_beta=0.1).validate()
    with pytest.raises(ValueError, match="validation_artifact_path"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1).validate()

    unvalidated = RewardShapingValidationReport.thresholds_absent()
    unvalidated_path = tmp_path / "unvalidated.json"
    unvalidated.write(unvalidated_path)
    with pytest.raises(ValueError, match="does not authorize"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=unvalidated_path).validate()

    valid = _synthetic_validated_report()
    unauthorized = _synthetic_validated_report(authorized=False)
    unauthorized_path = tmp_path / "unauthorized.json"
    unauthorized.write(unauthorized_path)
    with pytest.raises(ValueError, match="threshold_contract"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=unauthorized_path).validate()

    valid_path = tmp_path / "valid.json"
    valid.write(valid_path)
    assert RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=valid_path).validate() == valid

    exception_only = _synthetic_validated_report(
        gate_override={
            category: {"passed": False, "approved_exception": True} for category in REQUIRED_GATE_RESULT_CATEGORIES
        }
    )
    exception_only_path = tmp_path / "exception-only.json"
    exception_only.write(exception_only_path)
    with pytest.raises(ValueError, match="did not pass"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=exception_only_path).validate()

    failed_gate = _synthetic_validated_report(gate_override={REQUIRED_GATE_RESULT_CATEGORIES[0]: {"passed": False}})
    failed_gate_path = tmp_path / "failed-gate.json"
    failed_gate.write(failed_gate_path)
    with pytest.raises(ValueError, match="did not pass"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=failed_gate_path).validate()

    mismatched_gamma_path = tmp_path / "mismatch-gamma.json"
    _synthetic_validated_report(gamma=0.99).write(mismatched_gamma_path)
    with pytest.raises(ValueError, match="gamma"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=mismatched_gamma_path).validate()

    mismatched_source_path = tmp_path / "mismatch-source.json"
    _synthetic_validated_report(source_identity={"source": "synthetic"}).write(mismatched_source_path)
    with pytest.raises(ValueError, match="source_identity"):
        RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=mismatched_source_path).validate(
            source_identity={"source": "other"}
        )


def test_enabled_reward_shaping_metadata_requires_artifact_and_threshold_authority(tmp_path: Path) -> None:
    report = _synthetic_validated_report()
    report_path = tmp_path / "valid.json"
    report.write(report_path)
    metadata = RewardShapingConfig(enabled=True, pbrs_beta=0.1, validation_artifact_path=report_path).metadata()
    assert metadata["threshold_contract_version"] == THRESHOLD_CONTRACT_VERSION_V1
    assert metadata["threshold_contract_id"] == "synthetic-threshold-contract"
    assert metadata["threshold_contract_hash"] == "sha256:synthetic-threshold-contract"

    for key in ("validation_artifact_id", "validation_artifact_hash", "validation_artifact_path"):
        bad = dict(metadata)
        bad[key] = ""
        with pytest.raises(ValueError, match=key):
            normalize_reward_shaping_metadata(bad)

    bad_threshold = dict(metadata)
    bad_threshold["threshold_contract_version"] = "wrong"
    with pytest.raises(ValueError, match="threshold_contract_version"):
        normalize_reward_shaping_metadata(bad_threshold)


def test_pbrs_formula_beta_zero_terminal_and_telescoping() -> None:
    base = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    phi_t = torch.tensor([0.7, 0.2, -0.4], dtype=torch.float32)
    phi_next = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float32)
    terminal_next = torch.tensor([False, False, True])

    torch.testing.assert_close(apply_pbrs_reward(base, phi_t, phi_next, pbrs_beta=0.0, gamma=0.9), base)
    shaped = apply_pbrs_reward(base, phi_t, phi_next, pbrs_beta=0.5, gamma=0.9, terminal_next=terminal_next)
    expected = base + 0.5 * (0.9 * torch.tensor([0.2, -0.4, 0.0]) - phi_t)
    torch.testing.assert_close(shaped, expected)

    stream_phi = torch.tensor([0.7, 0.2, -0.4, 0.0], dtype=torch.float32)
    assert telescoping_terminal_residual(stream_phi, gamma=0.9) == pytest.approx(0.0, abs=1.0e-6)
    assert truncation_shaping_residual(
        torch.tensor([0.7, 0.2], dtype=torch.float32), gamma=0.9, bootstrap_phi=-0.4
    ) == pytest.approx(0.0, abs=1.0e-6)


def test_metrics_and_safety_fixture_do_not_use_raw_score_shortcuts() -> None:
    probs = torch.zeros(2, 24, dtype=torch.float32)
    probs[0, 0] = 1.0
    probs[1, 23] = 1.0
    actor = torch.tensor([0, 0], dtype=torch.int64)
    phi = grp_expected_phi(grp_probs=probs, actor=actor)

    assert phi.tolist() == [1.0, -1.0]
    assert phi.tolist() != [-1.0, 1.0]

    metrics = compute_grp_metrics(probs, torch.tensor([0, 23]), torch.tensor([1.0, -1.0]), phi)
    assert metrics.sample_count == 2
    assert metrics.top1_accuracy == pytest.approx(1.0)

    payload = RewardShapingValidationReport.thresholds_absent().to_dict()
    payload["metrics"] = {"raw_score_phi": True}
    with pytest.raises(ValueError, match="forbidden"):
        RewardShapingValidationReport(**_ctor_payload(payload)).to_dict()


def _synthetic_validated_report(
    *,
    gamma: float = DEFAULT_GAE_GAMMA,
    source_identity: dict[str, object] | None = None,
    authorized: bool = True,
    gate_override: dict[str, dict[str, object]] | None = None,
) -> RewardShapingValidationReport:
    return RewardShapingValidationReport(
        schema_version=1,
        contract_version="reward_shaping_validation_v1",
        predictor_checkpoint_hash="sha256:synthetic",
        predictor_checkpoint_id="synthetic-validated",
        predictor_checkpoint_path="/synthetic/path.pt",
        model_config={"profile": "synthetic"},
        source_identity=source_identity or {},
        encoder_shape=(192, 34),
        action_space=46,
        grp_classes=24,
        rank_utility_id="U_A",
        rank_utility=(1.0, 0.3, -0.3, -1.0),
        phi_definition=PHI_DEFINITION_GRP_EXPECTED_U_A_V1,
        state_boundary=STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1,
        gamma=gamma,
        gae_lambda=DEFAULT_GAE_LAMBDA,
        public_only=True,
        validated=True,
        beta_activation_allowed=True,
        metrics={"nll": 0.1},
        bucket_metrics={"actor_0": {"sample_count": 4}},
        baselines={"score_only": {"missing": True}},
        threshold_contract_version=THRESHOLD_CONTRACT_VERSION_V1 if authorized else None,
        threshold_contract_id="synthetic-threshold-contract" if authorized else None,
        threshold_contract_hash="sha256:synthetic-threshold-contract" if authorized else None,
        gate_results=cast(
            "dict[str, object]",
            gate_override
            if gate_override is not None
            else ({category: {"passed": True} for category in REQUIRED_GATE_RESULT_CATEGORIES} if authorized else {}),
        ),
    )


def _ctor_payload(payload: dict[str, object]) -> dict[str, Any]:
    data = dict(payload)
    data["encoder_shape"] = tuple(cast("list[int]", data["encoder_shape"]))
    data["rank_utility"] = tuple(cast("list[float]", data["rank_utility"]))
    return data
