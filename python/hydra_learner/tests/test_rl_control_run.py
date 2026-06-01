from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import torch

from hydra_learner import arena_eval
from hydra_learner.ach_step import AchTrainStepConfig
from hydra_learner.checkpoint import ModelConfig, OptimizerConfig, RuntimeConfig, load_checkpoint_init_only
from hydra_learner.checkpoint_eval import PairedCheckpointEvalThresholds, build_paired_checkpoint_eval_summary
from hydra_learner.losses import LossWeights
from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.ppo_rollout import PpoRolloutMetadata, save_ppo_rollout_artifact
from hydra_learner.ppo_step import PpoBatch, PpoTrainStepConfig
from hydra_learner.reward_shaping import default_reward_shaping_metadata
from hydra_learner.rl import EntropyController, masked_log_prob
from hydra_learner.rl_control_run import (
    RlCheckpointConfig,
    RlControlRunConfig,
    RlEvalConfig,
    RlObjectiveConfig,
    make_native_arena_eval_pair,
    run_rl_control_run,
    run_rl_native_eval_pair,
)


def test_n1_control_run_saves_isolated_ppo_and_ach_checkpoints_and_summary(tmp_path: Path) -> None:
    torch.manual_seed(401)
    initial_model = _model_factory()
    initial_state = _clone_state(initial_model)
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path,
        _batch_from_model(initial_model),
        PpoRolloutMetadata(rank_utility_used="U_A", gae_gamma=0.995, gae_lambda=0.95),
    )

    result = run_rl_control_run(
        config=_control_config(tmp_path, (artifact_path,), update_steps=1),
        model_factory=_model_factory,
        initial_state_dict=initial_state,
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
    )

    assert result.ppo_checkpoint_path.exists()
    assert result.ach_checkpoint_path.exists()
    payload = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert payload == result.summary
    json.dumps(payload, allow_nan=False)
    assert payload["run_id"] == "rl-control-test"
    assert payload["update_step_count"] == 1
    assert payload["checkpoint_paths"] == {
        "ppo": str(result.ppo_checkpoint_path),
        "direct_sampled_ach": str(result.ach_checkpoint_path),
    }
    assert payload["objective_configs"] == {
        "ppo": {
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "bc_kl_reverse_coef": 0.01,
            "grad_clip_norm": None,
            "epochs": 1,
            "target_kl": None,
            "advantage_epsilon": 1.0e-8,
        },
        "direct_sampled_ach": {
            "eta": 1.0,
            "eps": 0.5,
            "l_th": 8.0,
            "pi_old_min": 1.0e-8,
            "advantage_epsilon": 1.0e-8,
            "value_coef": 0.5,
            "bc_kl_reverse_coef": 0.01,
            "grad_clip_norm": None,
        },
    }
    assert payload["artifact_metadata"][0]["path"] == str(artifact_path)
    assert payload["artifact_metadata"][0]["rank_utility_used"] == "U_A"
    default_shaping = default_reward_shaping_metadata(gamma=0.995, gae_lambda=0.95)
    assert payload["artifact_metadata"][0]["reward_shaping"] == default_shaping
    assert payload["reward_contract"] == {
        "name": "U_A",
        "base_reward": "terminal_U_A",
        "rank_utility_used": "U_A",
        "gae_gamma": 0.995,
        "gae_lambda": 0.95,
        "state_boundary": "player_local_decision_stream_v1",
        "reward_shaping": default_shaping,
    }
    assert "loss_total" in payload["final_train_metrics"]["ppo"]
    assert "loss_total" in payload["final_train_metrics"]["direct_sampled_ach"]

    ppo_payload = torch.load(result.ppo_checkpoint_path, map_location="cpu", weights_only=True)
    ach_payload = torch.load(result.ach_checkpoint_path, map_location="cpu", weights_only=True)
    assert ppo_payload["training_objective"]["objective"] == "ppo"
    assert ach_payload["training_objective"]["objective"] == "direct_sampled_ach"
    assert ppo_payload["training_objective"]["source_init_id"] == "init-fixture"
    assert ach_payload["training_objective"]["reward_contract"]["name"] == "U_A"
    assert ppo_payload["training_objective"]["reward_contract"]["reward_shaping"] == default_shaping
    assert ppo_payload["global_step"] == 1
    assert ach_payload["global_step"] == 1
    assert ppo_payload["samples_seen"] == 2
    assert ach_payload["samples_seen"] == 2
    assert _any_parameter_changed(ppo_payload["model_state"], initial_state)
    assert _any_parameter_changed(ach_payload["model_state"], initial_state)
    assert _states_differ(ppo_payload["model_state"], ach_payload["model_state"])
    assert ppo_payload["optimizer_state"] is not ach_payload["optimizer_state"]

    reloaded = _model_factory()
    state = load_checkpoint_init_only(result.ppo_checkpoint_path, model=reloaded, expected_model_config=_model_config())
    assert state.global_step == 1


def test_artifact_sequence_ordering_is_deterministic(tmp_path: Path) -> None:
    torch.manual_seed(403)
    initial_model = _model_factory()
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    save_ppo_rollout_artifact(first, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A"))
    save_ppo_rollout_artifact(second, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A"))

    result = run_rl_control_run(
        config=_control_config(tmp_path, (second, first), update_steps=3),
        model_factory=_model_factory,
        initial_state_dict=_clone_state(initial_model),
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
    )

    metadata = cast("list[dict[str, object]]", result.summary["artifact_metadata"])
    paths = [entry["path"] for entry in metadata]
    assert paths == [str(second), str(first)]
    assert result.summary["update_step_count"] == 3
    ppo_payload = torch.load(result.ppo_checkpoint_path, map_location="cpu", weights_only=True)
    assert ppo_payload["training_objective"]["artifacts"][0]["path"] == str(second)
    assert ppo_payload["training_objective"]["artifacts"][1]["path"] == str(first)


def test_control_run_rejects_mixed_reward_shaping_metadata(tmp_path: Path) -> None:
    torch.manual_seed(409)
    initial_model = _model_factory()
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    save_ppo_rollout_artifact(first, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A"))
    save_ppo_rollout_artifact(
        second,
        _batch_from_model(initial_model),
        PpoRolloutMetadata(
            rank_utility_used="U_A",
            reward_shaping={**default_reward_shaping_metadata(), "state_boundary": "other_boundary_v1"},
        ),
    )

    with pytest.raises(ValueError, match="mixed rollout reward metadata"):
        run_rl_control_run(
            config=_control_config(tmp_path, (first, second), update_steps=1),
            model_factory=_model_factory,
            initial_state_dict=_clone_state(initial_model),
            optimizer_factory=_optimizer_factory,
            entropy_controller=_entropy_controller(),
        )


def test_eval_surface_accepts_synthetic_metrics_without_arena(tmp_path: Path) -> None:
    torch.manual_seed(405)
    initial_model = _model_factory()
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A")
    )
    calls: list[tuple[Path, Path, int]] = []

    def fake_eval(baseline: Path, candidate: Path, seed: int) -> Mapping[str, object]:
        calls.append((baseline, candidate, seed))
        return {
            "games": 8,
            "candidate_top2_rate": 0.55,
            "baseline_top2_rate": 0.50,
            "candidate_fourth_rate": 0.20,
            "baseline_fourth_rate": 0.22,
            "candidate_mean_placement": 2.30,
            "baseline_mean_placement": 2.40,
        }

    result = run_rl_control_run(
        config=_control_config(
            tmp_path,
            (artifact_path,),
            update_steps=1,
            eval_config=RlEvalConfig(
                baseline_objective="ppo",
                candidate_objective="direct_sampled_ach",
                seed=123,
                thresholds=PairedCheckpointEvalThresholds(min_games=4),
            ),
        ),
        model_factory=_model_factory,
        initial_state_dict=_clone_state(initial_model),
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        eval_pair=fake_eval,
    )

    assert calls == [(result.ppo_checkpoint_path, result.ach_checkpoint_path, 123)]
    paired = result.summary["paired_eval"]
    assert isinstance(paired, dict)
    assert paired["seed"] == 123
    assert paired["decision"]["decision"] == "promote"


def test_native_arena_eval_pair_calls_arena_eval_and_returns_json_safe_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    calls: list[arena_eval.ArenaEvalConfig] = []

    def fake_run_arena_eval(config: arena_eval.ArenaEvalConfig) -> dict[str, object]:
        calls.append(config)
        return {
            "config": {},
            "baseline": {"path": str(config.baseline)},
            "candidates": [
                {
                    "candidate": "candidate",
                    "candidate_path": str(config.candidates[0]),
                    "result": {
                        "games": config.games,
                        "candidate_top2_rate": 0.56,
                        "baseline_top2_rate": 0.50,
                        "candidate_fourth_rate": 0.18,
                        "baseline_fourth_rate": 0.20,
                        "candidate_mean_placement": 2.30,
                        "baseline_mean_placement": 2.40,
                    },
                }
            ],
        }

    monkeypatch.setattr(arena_eval, "run_arena_eval", fake_run_arena_eval)
    seedless_output = tmp_path / "arena.json"
    eval_pair = make_native_arena_eval_pair(
        games=16,
        output_path=seedless_output,
        temperature=0.75,
        device="cpu",
        extension="fake_ext",
        arena_batch_decisions=32,
        arena_threads=2,
    )

    metrics = eval_pair(baseline, candidate, 909)

    assert metrics["games"] == 16
    assert calls == [
        arena_eval.ArenaEvalConfig(
            baseline=baseline,
            candidates=(candidate,),
            games=16,
            seed=909,
            temperature=0.75,
            output_path=seedless_output,
            per_game_path=None,
            tensorboard_dir=None,
            weight_source="raw",
            device="cpu",
            extension="fake_ext",
            extension_path=None,
            arena_batch_decisions=32,
            rust_native=True,
            arena_threads=2,
            hidden=arena_eval.DEFAULT_HIDDEN,
            blocks=arena_eval.DEFAULT_BLOCKS,
            bottleneck=arena_eval.DEFAULT_SE_BOTTLENECK,
            residual_profile=arena_eval.RESIDUAL_PROFILE_DEFAULT,
            backbone_profile=arena_eval.BACKBONE_PROFILE_DEFAULT,
            conv_memory_format=arena_eval.CONV_MEMORY_FORMAT_DEFAULT,
        )
    ]
    json.dumps(metrics, allow_nan=False, sort_keys=True, separators=(",", ":"))


def test_native_arena_eval_pair_feeds_control_run_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(409)
    initial_model = _model_factory()
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A")
    )
    seen: list[tuple[Path, Path, int]] = []

    def fake_run_arena_eval(config: arena_eval.ArenaEvalConfig) -> dict[str, object]:
        seen.append((config.baseline, config.candidates[0], config.seed))
        return {
            "candidates": [
                {
                    "result": {
                        "games": config.games,
                        "candidate_top2_rate": 0.58,
                        "baseline_top2_rate": 0.50,
                        "candidate_fourth_rate": 0.18,
                        "baseline_fourth_rate": 0.20,
                        "candidate_mean_placement": 2.30,
                        "baseline_mean_placement": 2.40,
                    }
                }
            ]
        }

    monkeypatch.setattr(arena_eval, "run_arena_eval", fake_run_arena_eval)
    result = run_rl_control_run(
        config=_control_config(
            tmp_path,
            (artifact_path,),
            update_steps=1,
            eval_config=RlEvalConfig(
                baseline_objective="ppo",
                candidate_objective="direct_sampled_ach",
                seed=707,
                thresholds=PairedCheckpointEvalThresholds(min_games=4, min_top2_delta=0.05),
            ),
        ),
        model_factory=_model_factory,
        initial_state_dict=_clone_state(initial_model),
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        eval_pair=make_native_arena_eval_pair(
            games=12,
            output_path=tmp_path / "arena.json",
            device="cpu",
            extension="fake_ext",
        ),
    )

    assert seen == [(result.ppo_checkpoint_path, result.ach_checkpoint_path, 707)]
    paired = result.summary["paired_eval"]
    assert isinstance(paired, dict)
    assert paired["baseline"] == str(result.ppo_checkpoint_path)
    assert paired["candidate"] == str(result.ach_checkpoint_path)
    assert paired["seed"] == 707
    assert paired["games"] == 12
    assert paired["decision"]["decision"] == "promote"
    json.dumps(result.summary, allow_nan=False, sort_keys=True, separators=(",", ":"))


def test_native_arena_eval_pair_uses_arena_export_semantics_for_pt_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    calls: list[tuple[Path, Path]] = []

    def fake_run_arena_eval(config: arena_eval.ArenaEvalConfig) -> dict[str, object]:
        baseline_export = arena_eval.resolve_native_arena_path(config.baseline, config)
        candidate_export = arena_eval.resolve_native_arena_path(config.candidates[0], config)
        calls.append((baseline_export, candidate_export))
        return {"candidates": [{"result": {"games": config.games}}]}

    def fake_resolve(path: Path, config: arena_eval.ArenaEvalConfig) -> Path:
        assert config.rust_native
        assert path.suffix == ".pt"
        return path.with_suffix("") / "export"

    monkeypatch.setattr(arena_eval, "run_arena_eval", fake_run_arena_eval)
    monkeypatch.setattr(arena_eval, "resolve_native_arena_path", fake_resolve)

    metrics = run_rl_native_eval_pair(
        baseline=baseline,
        candidate=candidate,
        seed=11,
        games=3,
        output_path=tmp_path / "arena.json",
        device="cpu",
    )

    assert metrics == {"games": 3}
    assert calls == [(baseline.with_suffix("") / "export", candidate.with_suffix("") / "export")]


def test_native_arena_eval_pair_missing_configured_metric_rejects() -> None:
    metrics = {"games": 8, "candidate_top2_rate": 0.6, "baseline_top2_rate": 0.5}

    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        arena_metrics=metrics,
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=None, min_mean_u_a_delta=0.0),
        seed=5,
    )

    assert summary.decision.decision == "reject"
    assert summary.decision.reasons == ("missing_mean_u_a_delta",)


def test_summary_json_rejects_non_finite_metrics_before_write(tmp_path: Path) -> None:
    torch.manual_seed(407)
    initial_model = _model_factory()
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A")
    )

    def bad_eval(_baseline: Path, _candidate: Path, _seed: int) -> Mapping[str, object]:
        return {"games": 1, "candidate_top2_rate": float("nan")}

    with pytest.raises(ValueError, match="must be finite"):
        run_rl_control_run(
            config=_control_config(
                tmp_path,
                (artifact_path,),
                update_steps=1,
                eval_config=RlEvalConfig(baseline_objective="ppo", candidate_objective="direct_sampled_ach", seed=1),
            ),
            model_factory=_model_factory,
            initial_state_dict=_clone_state(initial_model),
            optimizer_factory=_optimizer_factory,
            entropy_controller=_entropy_controller(),
            eval_pair=bad_eval,
        )


def test_control_run_scope_has_no_neurd_residual_tau_rebase_grp_search_or_burn_fields() -> None:
    field_names = {field.name for field in fields(RlObjectiveConfig)} | {
        field.name for field in fields(RlControlRunConfig)
    }
    nested_names = {field.name for field in fields(PpoTrainStepConfig)} | {
        field.name for field in fields(AchTrainStepConfig)
    }
    forbidden = {"neurd", "tau", "rebase", "grp", "search", "burn"}
    assert not (field_names | nested_names) & forbidden


def _control_config(
    tmp_path: Path,
    artifact_paths: tuple[Path, ...],
    *,
    update_steps: int,
    eval_config: RlEvalConfig | None = None,
) -> RlControlRunConfig:
    return RlControlRunConfig(
        run_id="rl-control-test",
        artifact_paths=artifact_paths,
        update_steps=update_steps,
        source_init_id="init-fixture",
        objectives=RlObjectiveConfig(
            ppo=PpoTrainStepConfig(bc_kl_reverse_coef=0.01, grad_clip_norm=None),
            ach=AchTrainStepConfig(bc_kl_reverse_coef=0.01, grad_clip_norm=None),
        ),
        checkpoint=RlCheckpointConfig(
            model=_model_config(),
            optimizer=_optimizer_config(),
            runtime=_runtime_config(),
            loss_weights=LossWeights(),
        ),
        output_dir=tmp_path / "out",
        eval=eval_config,
    )


def _model_factory() -> HydraPolicyNet:
    return HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)


def _optimizer_factory(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=1.0e-3)


def _entropy_controller() -> EntropyController:
    return EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05)


def _model_config() -> ModelConfig:
    return ModelConfig(hidden=8, blocks=1, bottleneck=4)


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(variant="eager", loss_mode="rl_control", precision_mode="fp32", compile_fullgraph_check=False)


def _batch_from_model(model: HydraPolicyNet) -> PpoBatch:
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    with torch.inference_mode():
        out = model(obs)
        old_logprob = masked_log_prob(out.policy_logits, legal_mask, actions).to(dtype=torch.float32)
        value_old = out.value.squeeze(1).to(dtype=torch.float32)
        bc_logits = out.policy_logits.detach().clone().to(dtype=torch.float32)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=value_old + torch.tensor([0.25, -0.25], dtype=torch.float32),
        bc_logits=bc_logits,
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([1, 1], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _any_parameter_changed(after: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor]) -> bool:
    return any(not torch.equal(tensor.detach().cpu(), before[key].detach().cpu()) for key, tensor in after.items())


def _states_differ(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return any(not torch.equal(tensor.detach().cpu(), right[key].detach().cpu()) for key, tensor in left.items())
