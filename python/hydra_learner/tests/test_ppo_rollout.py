from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from hydra_learner.checkpoint import (
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    load_checkpoint_init_only,
)
from hydra_learner.losses import LossWeights
from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.ppo_rollout import (
    PpoRolloutMetadata,
    append_ppo_metrics_jsonl,
    artifact_to_ppo_batch,
    load_ppo_rollout_artifact,
    save_ppo_rollout_artifact,
    save_ppo_training_checkpoint,
    train_step_from_rollout_artifact,
)
from hydra_learner.ppo_smoke import (
    RustDecisionRow,
    RustGameRollout,
    load_rust_game_rollout_json,
    write_ppo_smoke_rollout_artifact,
)
from hydra_learner.ppo_step import PpoBatch, PpoTrainStepConfig
from hydra_learner.rl import EntropyController, masked_log_prob


def test_ppo_rollout_artifact_roundtrip_and_batch_conversion(tmp_path: Path) -> None:
    batch = _valid_batch()
    path = tmp_path / "rollout.pt"
    metadata = PpoRolloutMetadata(rank_utility_used="U_A", gae_gamma=0.995, gae_lambda=0.95)

    save_ppo_rollout_artifact(path, batch, metadata)
    artifact = load_ppo_rollout_artifact(path)
    loaded_batch = artifact_to_ppo_batch(artifact)

    assert artifact.metadata == metadata
    for name in (
        "obs",
        "actions",
        "legal_mask",
        "old_logprob",
        "value_old",
        "raw_advantages",
        "returns",
        "bc_logits",
        "legal_count",
    ):
        torch.testing.assert_close(getattr(loaded_batch, name), getattr(batch, name), rtol=0.0, atol=0.0)
    assert loaded_batch.rank_utility_used == "U_A"
    loaded_batch.validate()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 999, "schema_version"),
        ("obs", torch.zeros(2, 191, 34, dtype=torch.float32), "obs"),
        ("legal_mask", torch.ones(2, ACTION_SPACE, dtype=torch.float32), "legal_mask"),
        ("legal_count", torch.tensor([1, 2], dtype=torch.int64), "legal_count"),
        ("old_logprob", torch.tensor([0.0, torch.nan], dtype=torch.float32), "old_logprob"),
    ],
)
def test_ppo_rollout_artifact_validation_hard_errors(tmp_path: Path, field: str, value: object, match: str) -> None:
    path = tmp_path / "bad.pt"
    payload = _artifact_payload(_valid_batch())
    payload[field] = value
    torch.save(payload, path)

    with pytest.raises((TypeError, ValueError), match=match):
        load_ppo_rollout_artifact(path)


def test_ppo_rollout_artifact_rejects_all_illegal_selected_illegal_and_bad_bc_logits(tmp_path: Path) -> None:
    for mutate, match in (
        (_mutate_all_illegal, "all-illegal"),
        (_mutate_selected_illegal, "legal"),
        (_mutate_bad_bc_logits, "bc_logits"),
    ):
        path = tmp_path / f"bad-{match}.pt"
        payload = _artifact_payload(_valid_batch())
        mutate(payload)
        torch.save(payload, path)
        with pytest.raises((TypeError, ValueError), match=match):
            load_ppo_rollout_artifact(path)


def test_train_step_from_rollout_artifact_real_model_json_metrics_and_update(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    artifact_path = tmp_path / "rollout.pt"
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))
    save_ppo_rollout_artifact(artifact_path, batch, PpoRolloutMetadata(rank_utility_used="U_A"))
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    result = train_step_from_rollout_artifact(
        artifact_path=artifact_path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05),
        config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.01, grad_clip_norm=0.5),
    )

    json.dumps(result.metrics, allow_nan=False)
    assert _any_parameter_changed(model.state_dict(), before)
    assert 0.0 <= result.entropy_controller.alpha <= 0.05
    assert result.artifact_metadata["rank_utility_used"] == "U_A"
    assert result.metrics["rollout_contract_version"] == "ppo_rollout_v1"
    json.dumps(result.metrics, allow_nan=False)


def test_noop_artifact_has_near_zero_bc_kl_and_no_policy_drift(tmp_path: Path) -> None:
    torch.manual_seed(37)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    artifact_path = tmp_path / "noop.pt"
    batch = _batch_from_model(model, raw_advantages=torch.zeros(2, dtype=torch.float32), returns_from_model=True)
    save_ppo_rollout_artifact(artifact_path, batch, PpoRolloutMetadata(rank_utility_used="U_A"))
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    result = train_step_from_rollout_artifact(
        artifact_path=artifact_path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=PpoTrainStepConfig(value_coef=0.0, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )

    assert result.metrics["bc_kl_reverse"] == pytest.approx(0.0, abs=1.0e-7)
    assert result.metrics["loss_policy"] == pytest.approx(0.0, abs=1.0e-7)
    assert not _any_parameter_changed(model.state_dict(), before)
    json.dumps(result.metrics, allow_nan=False)


def test_append_ppo_metrics_jsonl_strict_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "ppo.jsonl"
    append_ppo_metrics_jsonl(path, {"b": 2.0, "a": {"x": 1}})
    append_ppo_metrics_jsonl(path, {"b": 3.0, "a": {"x": 2}})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows == [{"a": {"x": 1}, "b": 2.0}, {"a": {"x": 2}, "b": 3.0}]
    with pytest.raises(ValueError, match="Out of range float"):
        append_ppo_metrics_jsonl(path, {"bad": float("nan")})


def test_artifact_train_step_extreme_old_logprob_hard_errors(tmp_path: Path) -> None:
    torch.manual_seed(39)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    artifact_path = tmp_path / "extreme.pt"
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))
    batch = PpoBatch(
        obs=batch.obs,
        actions=batch.actions,
        legal_mask=batch.legal_mask,
        old_logprob=torch.full((2,), -1.0e38, dtype=torch.float32),
        value_old=batch.value_old,
        raw_advantages=batch.raw_advantages,
        returns=batch.returns,
        bc_logits=batch.bc_logits,
        legal_count=batch.legal_count,
        player_id=batch.player_id,
        seat_id=batch.seat_id,
        game_id=batch.game_id,
        turn=batch.turn,
        rank_utility_used=batch.rank_utility_used,
    )
    save_ppo_rollout_artifact(artifact_path, batch, PpoRolloutMetadata(rank_utility_used="U_A"))

    with pytest.raises(ValueError, match="ratio"):
        train_step_from_rollout_artifact(
            artifact_path=artifact_path,
            model=model,
            optimizer=optimizer,
            entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
            config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
        )


def test_checkpoint_smoke_after_artifact_train_step_init_only_reload(tmp_path: Path) -> None:
    torch.manual_seed(41)
    random.seed(42)
    np.random.seed(43)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    artifact_path = tmp_path / "rollout.pt"
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))
    save_ppo_rollout_artifact(artifact_path, batch, PpoRolloutMetadata(rank_utility_used="U_A"))

    train_step_from_rollout_artifact(
        artifact_path=artifact_path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.0, grad_clip_norm=0.5),
    )
    ckpt = tmp_path / "ckpt.pt"
    save_ppo_training_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=None,
        global_step=9,
        samples_seen=18,
    )
    fresh = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1.0e-3)
    before_optimizer = fresh_optimizer.state_dict()
    random.seed(50)
    np.random.seed(51)
    torch.manual_seed(52)
    expected_random = (random.random(), np.random.rand(2), torch.randn(2))
    random.seed(50)
    np.random.seed(51)
    torch.manual_seed(52)

    state = load_checkpoint_init_only(ckpt, model=fresh, expected_model_config=_model_config())
    actual_random = (random.random(), np.random.rand(2), torch.randn(2))

    assert state.global_step == 9
    assert state.samples_seen == 18
    assert state.weight_source == "raw"
    assert fresh_optimizer.state_dict() == before_optimizer
    assert actual_random[0] == expected_random[0]
    np.testing.assert_array_equal(actual_random[1], expected_random[1])
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0.0, atol=0.0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_phase2_smoke_artifact_update_metrics_checkpoint_and_init_reload(tmp_path: Path) -> None:
    torch.manual_seed(61)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    rollout = _real_rust_rollout_fixture(tmp_path)
    artifact_path = tmp_path / "phase2-rollout.pt"

    artifact_result = write_ppo_smoke_rollout_artifact(artifact_path, rollout, model=model, torch_seed=1234)
    artifact = load_ppo_rollout_artifact(artifact_path)
    batch = artifact_to_ppo_batch(artifact)
    batch.validate()
    assert artifact_result.metrics["artifact_rows"] == len(rollout.rows)
    assert artifact_result.metrics["num_games"] == 1
    assert artifact_result.metrics["illegal_action_count"] == 0
    assert artifact_result.metrics["all_illegal_count"] == 0
    assert artifact_result.metrics["placement_histogram"] == [1, 1, 1, 1]
    assert artifact_result.metrics["seed"] == rollout.seed

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    train_result = train_step_from_rollout_artifact(
        artifact_path=artifact_path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05),
        config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.01, grad_clip_norm=0.5),
    )
    metrics = dict(artifact_result.metrics)
    metrics.update(train_result.metrics)
    metrics["checkpoint_path"] = str(tmp_path / "ppo-smoke.pt")
    metrics["artifact_path"] = str(artifact_path)
    for key in (
        "artifact_rows",
        "num_games",
        "num_decisions",
        "illegal_action_count",
        "all_illegal_count",
        "loss_total",
        "loss_policy",
        "loss_value",
        "entropy",
        "entropy_alpha_before",
        "entropy_alpha_after",
        "bc_kl_reverse",
        "approx_kl_old",
        "clip_fraction",
        "ratio_mean",
        "grad_norm",
        "mean_U_A",
        "placement_histogram",
        "checkpoint_path",
        "artifact_path",
        "seed",
    ):
        assert key in metrics
    json.dumps(metrics, allow_nan=False)
    metrics_path = tmp_path / "metrics" / "ppo-smoke.jsonl"
    append_ppo_metrics_jsonl(metrics_path, metrics)
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 1
    assert _any_parameter_changed(model.state_dict(), before)

    ckpt = tmp_path / "ppo-smoke.pt"
    save_ppo_training_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        model_config=_model_config(),
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=None,
        global_step=1,
        samples_seen=batch.obs.shape[0],
    )
    fresh = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    state = load_checkpoint_init_only(ckpt, model=fresh, expected_model_config=_model_config())
    assert state.global_step == 1
    assert state.samples_seen == len(rollout.rows)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_phase2_smoke_artifact_is_deterministic_for_same_seed_and_model(tmp_path: Path) -> None:
    torch.manual_seed(67)
    first_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    second_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    second_model.load_state_dict(first_model.state_dict())
    rollout = _real_rust_rollout_fixture(tmp_path)

    first = write_ppo_smoke_rollout_artifact(tmp_path / "first.pt", rollout, model=first_model, torch_seed=99).batch
    second = write_ppo_smoke_rollout_artifact(tmp_path / "second.pt", rollout, model=second_model, torch_seed=99).batch

    for name in ("obs", "actions", "legal_mask", "old_logprob", "value_old", "raw_advantages", "returns", "bc_logits"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rollout: _replace_rollout_row(rollout, 0, action=45), "row action must be legal"),
        (lambda rollout: _replace_rollout_row(rollout, 0, legal_count=1), "legal_count"),
        (lambda rollout: _replace_rollout_row(rollout, 0, obs=torch.zeros(191, 34, dtype=torch.float32)), "obs"),
        (lambda rollout: _replace_rollout_row(rollout, 0, obs=_bad_finite_obs()), "obs must be finite"),
        (lambda rollout: _bad_placements_rollout(rollout), "placements"),
    ],
)
def test_phase2_smoke_rollout_negative_boundary_errors(
    mutate: Callable[[RustGameRollout], RustGameRollout], match: str, tmp_path: Path
) -> None:
    torch.manual_seed(71)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    bad = mutate(_real_rust_rollout_fixture(tmp_path))
    with pytest.raises((TypeError, ValueError), match=match):
        write_ppo_smoke_rollout_artifact(tmp_path / "bad.pt", bad, model=model, torch_seed=5)


def _valid_batch() -> PpoBatch:
    obs = torch.zeros(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([1, 3], dtype=torch.int64)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=torch.zeros(2, dtype=torch.float32),
        value_old=torch.zeros(2, dtype=torch.float32),
        raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=torch.zeros(2, dtype=torch.float32),
        bc_logits=torch.zeros(2, ACTION_SPACE, dtype=torch.float32),
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([123, 123], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _batch_from_model(
    model: HydraPolicyNet, *, raw_advantages: torch.Tensor, returns_from_model: bool = False
) -> PpoBatch:
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    with torch.inference_mode():
        out = model(obs)
        old_logprob = masked_log_prob(out.policy_logits, legal_mask, actions).to(dtype=torch.float32)
        value_old = out.value.squeeze(1).to(dtype=torch.float32)
        returns = value_old.clone() if returns_from_model else value_old + torch.tensor([0.25, -0.25])
        bc_logits = out.policy_logits.detach().clone().to(dtype=torch.float32)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=bc_logits,
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([1, 1], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _artifact_payload(batch: PpoBatch) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_version": "ppo_rollout_v1",
        "obs": batch.obs,
        "actions": batch.actions,
        "legal_mask": batch.legal_mask,
        "old_logprob": batch.old_logprob,
        "value_old": batch.value_old,
        "raw_advantages": batch.raw_advantages,
        "returns": batch.returns,
        "bc_logits": batch.bc_logits,
        "legal_count": batch.legal_count,
        "metadata": {"rank_utility_used": "U_A", "gae_gamma": 0.995, "gae_lambda": 0.95},
        "player_id": batch.player_id,
        "seat_id": batch.seat_id,
        "game_id": batch.game_id,
        "turn": batch.turn,
    }


def _mutate_all_illegal(payload: dict[str, object]) -> None:
    legal_mask = cast("torch.Tensor", payload["legal_mask"]).clone()
    legal_mask[0].zero_()
    payload["legal_mask"] = legal_mask
    payload["legal_count"] = legal_mask.sum(dim=1).to(dtype=torch.int64)


def _mutate_selected_illegal(payload: dict[str, object]) -> None:
    actions = cast("torch.Tensor", payload["actions"]).clone()
    actions[0] = 45
    payload["actions"] = actions


def _mutate_bad_bc_logits(payload: dict[str, object]) -> None:
    bc_logits = cast("torch.Tensor", payload["bc_logits"]).clone()
    bc_logits[0, 0] = torch.nan
    payload["bc_logits"] = bc_logits


def _real_rust_rollout_fixture(tmp_path: Path, seed: int = 20260524) -> RustGameRollout:
    output = tmp_path / f"rust-rollout-{seed}.json"
    subprocess.run(
        [
            "pixi",
            "run",
            "cargo",
            "run",
            "--quiet",
            "--package",
            "hydra-core",
            "--example",
            "ppo_smoke_fixture",
            "--no-default-features",
            "--",
            str(output),
            str(seed),
        ],
        check=True,
    )
    rollout = load_rust_game_rollout_json(output)
    assert rollout.rows
    return rollout


def _replace_rollout_row(rollout: RustGameRollout, row: int, **updates: object) -> RustGameRollout:
    rows = list(rollout.rows)
    current = rows[row]
    values: dict[str, object] = {
        "obs": current.obs,
        "legal_mask": current.legal_mask,
        "player_id": current.player_id,
        "seat_id": current.seat_id,
        "game_id": current.game_id,
        "turn": current.turn,
        "action": current.action,
        "legal_count": current.legal_count,
    }
    values.update(updates)
    rows[row] = RustDecisionRow(
        obs=cast("torch.Tensor", values["obs"]),
        legal_mask=cast("torch.Tensor", values["legal_mask"]),
        player_id=cast("int", values["player_id"]),
        seat_id=cast("int", values["seat_id"]),
        game_id=cast("int", values["game_id"]),
        turn=cast("int", values["turn"]),
        action=cast("int | None", values["action"]),
        legal_count=cast("int | None", values.get("legal_count")),
    )
    return RustGameRollout(tuple(rows), rollout.final_scores, rollout.placements, rollout.seed)


def _bad_finite_obs() -> torch.Tensor:
    obs = torch.zeros(192, 34, dtype=torch.float32)
    obs[0, 0] = torch.nan
    return obs


def _bad_placements_rollout(rollout: RustGameRollout) -> RustGameRollout:
    return RustGameRollout(
        rollout.rows,
        rollout.final_scores,
        cast("tuple[int, int, int, int]", (0, 1, 2)),
        rollout.seed,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(hidden=8, blocks=1, bottleneck=4)


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        variant="eager_bf16", loss_mode="full_base", precision_mode="bf16_autocast", compile_fullgraph_check=False
    )


def _any_parameter_changed(after: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> bool:
    return any(tensor.is_floating_point() and not torch.equal(tensor, before[name]) for name, tensor in after.items())
