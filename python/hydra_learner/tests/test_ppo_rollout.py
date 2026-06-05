from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol, cast, override

import numpy as np
import pytest
import torch

import hydra_learner.mahjax.ppo_rollout as mahjax_rollout
from hydra_learner import ppo_control
from hydra_learner.checkpointing.core import (
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    load_checkpoint_init_only,
)
from hydra_learner.mahjax import observation as mahjax_observation_adapter
from hydra_learner.mahjax.ppo_rollout import (
    _completion_sync_interval,
    _configure_jax_compilation_cache,
    _final_score_metrics,
    _jax_compilation_cache_dir,
    _parts_to_batch,
    _RowPart,
    _use_jax_aot,
)
from hydra_learner.mahjax.ppo_rollout import _gae_for_slots as _mahjax_gae_for_slots
from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.model.losses import LossWeights
from hydra_learner.ppo.config import (
    PpoControlConfig,
    _compatible_resume_config_digests,
    _config_digest,
    _json_config,
    parse_args,
    validate_args,
)
from hydra_learner.ppo.control_rollout import (
    _batch_from_native_payload_fast,
    _batch_to_device,
    _make_ppo_inference_callback,
    _terminal_gae_from_cpu_values,
)
from hydra_learner.ppo.rl import (
    EntropyController,
    PlayerDecisionStep,
    compute_player_local_gae,
    masked_log_prob,
    masked_log_softmax,
)
from hydra_learner.ppo.rollout import (
    PpoRolloutMetadata,
    PpoSnapshotMetadata,
    append_ppo_metrics_jsonl,
    artifact_to_ppo_batch,
    build_ppo_snapshot_metadata,
    load_ppo_policy_snapshot_artifact,
    load_ppo_rollout_artifact,
    save_ppo_policy_snapshot_artifact,
    save_ppo_rollout_artifact,
    save_ppo_training_checkpoint,
    train_step_from_rollout_artifact,
)
from hydra_learner.ppo.smoke import (
    RustDecisionRow,
    RustGameRollout,
    build_ppo_batch_from_rust_rollout,
    write_ppo_smoke_rollout_artifact,
)
from hydra_learner.ppo.step import PpoBatch, PpoTrainStepConfig, ppo_train_step
from hydra_learner.rl_experiments.reward_shaping import default_reward_shaping_metadata
from tests.fixtures import (
    TINY_CHECKPOINT_CONFIG_DIGEST,
    tiny_checkpoint_ppo_control_config,
    tiny_ppo_rollout,
    tiny_run_local_paths,
)


class _DebugPpoInferenceCallback(Protocol):
    _timings: MutableMapping[str, float]

    def __call__(self, obs_f32_le: bytearray, *args: object) -> object: ...

    def _packed_buffer_ptr(self) -> int: ...

    def _packed_capacity(self) -> int: ...


def _debug_ppo_callback(callback: Callable[..., object]) -> _DebugPpoInferenceCallback:
    return cast("_DebugPpoInferenceCallback", callback)


def _minimal_valid_ppo_batch(rows: int = 2) -> PpoBatch:
    legal_mask = torch.ones(rows, ACTION_SPACE, dtype=torch.bool)
    return PpoBatch(
        obs=torch.zeros(rows, 192, 34),
        actions=torch.zeros(rows, dtype=torch.int64),
        legal_mask=legal_mask,
        old_logprob=torch.zeros(rows),
        value_old=torch.zeros(rows),
        raw_advantages=torch.zeros(rows),
        returns=torch.zeros(rows),
        bc_logits=torch.zeros(rows, ACTION_SPACE),
        legal_count=legal_mask.sum(dim=1).to(torch.int64),
    )


def test_ppo_rollout_inference_accepts_opt_in_mahjax_gpu(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "tiny",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
            "--rollout-inference",
            "mahjax-gpu",
        ]
    )

    config = validate_args(args)

    assert config.rollout_inference == "mahjax-gpu"


def test_ppo_rollout_inference_defaults_to_mahjax_gpu_for_serial_ppo(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "tiny",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )

    config = validate_args(args)

    assert config.rollout_inference == "mahjax-gpu"


def test_ppo_rollout_inference_defaults_to_mahjax_gpu_for_pipeline_ppo(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "tiny",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
            "--ppo-pipeline-depth",
            "1",
        ]
    )
    config = validate_args(args)

    assert config.rollout_inference == "mahjax-gpu"


def test_mahjax_gpu_default_keeps_observation_contract_unblocked(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "tiny",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )
    config = validate_args(args)

    assert mahjax_observation_adapter.MAHJAX_DEFAULT_BLOCKED_CHANNELS == ()
    assert config.rollout_inference == "mahjax-gpu"


def test_ppo_rollout_inference_accepts_mahjax_gpu_pipeline(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "tiny",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
            "--rollout-inference",
            "mahjax-gpu",
            "--ppo-pipeline-depth",
            "1",
        ]
    )

    config = validate_args(args)

    assert config.rollout_inference == "mahjax-gpu"
    assert config.ppo_pipeline_depth == 1


def test_ppo_collect_rollout_dispatches_mahjax_gpu_without_onnx_or_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = replace(
        _ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=0), rollout_inference="mahjax-gpu"
    )
    batch = _minimal_valid_ppo_batch()
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        ppo_control,
        "export_loaded_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("onnx export called")),
    )
    monkeypatch.setattr(
        ppo_control,
        "_collect_native_rollout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native arena called")),
    )
    monkeypatch.setattr(
        ppo_control,
        "_collect_callback_rollout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("callback arena called")),
    )

    def fake_mahjax_collect(**kwargs: object) -> object:
        seen.update(kwargs)
        return SimpleNamespace(
            batch=batch,
            timing={"mahjax_total_ms": 1.0},
            row_count=batch.obs.shape[0],
            metrics={"episode_reward_mean": 0.25},
        )

    monkeypatch.setattr(ppo_control, "collect_mahjax_ppo_rollout", fake_mahjax_collect)

    result = ppo_control._collect_rollout_batch(
        config=config,
        config_digest="digest",
        model_config=_model_config(),
        extension=None,
        export_dir=tmp_path / "exports",
        model=HydraPolicyNet(hidden=8, blocks=1, bottleneck=4),
        global_step=3,
        samples_seen=11,
        completed_games=5,
    )

    assert seen["seed"] == 12
    assert result.snapshot.inference_backend == "mahjax-gpu"
    assert result.payload["snapshot_metadata"] == result.snapshot.to_payload()
    assert result.batch is batch
    assert result.outcome_metrics == {"episode_reward_mean": 0.25}


def test_mahjax_final_score_metrics_report_seat_outcomes() -> None:
    metrics = _final_score_metrics(torch.tensor([[35000, 25000, 15000, 5000], [5000, 15000, 25000, 35000]]))

    assert metrics["episode_score_mean"] == 20000.0
    assert metrics["episode_reward_mean"] == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["seat0_first_rate"] == 0.5
    assert metrics["seat0_last_rate"] == 0.5
    assert metrics["seat3_first_rate"] == 0.5
    assert metrics["seat3_last_rate"] == 0.5


def test_mahjax_final_score_metrics_split_tied_placement_credit() -> None:
    metrics = _final_score_metrics(torch.full((2, 4), 25000))

    assert metrics["episode_reward_mean"] == pytest.approx(0.0, abs=1.0e-7)
    for seat in range(4):
        assert metrics[f"seat{seat}_reward_mean"] == pytest.approx(0.0, abs=1.0e-7)
        assert metrics[f"seat{seat}_placement_mean"] == 1.5
        assert metrics[f"seat{seat}_first_rate"] == 0.25
        assert metrics[f"seat{seat}_last_rate"] == 0.25


def test_mahjax_rollout_batch_does_not_require_native_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = replace(
        _ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=0), rollout_inference="mahjax-gpu"
    )
    batch = _minimal_valid_ppo_batch()

    monkeypatch.setattr(
        ppo_control,
        "_load_extension",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native extension loaded")),
    )
    monkeypatch.setattr(
        ppo_control,
        "default_arena_pyo3_library_path",
        lambda: (_ for _ in ()).throw(AssertionError("arena path resolved")),
    )
    monkeypatch.setattr(
        ppo_control,
        "collect_mahjax_ppo_rollout",
        lambda **_kwargs: SimpleNamespace(
            batch=batch, timing={"mahjax_total_ms": 1.0}, row_count=batch.obs.shape[0], metrics={}
        ),
    )

    result = ppo_control._collect_rollout_batch(
        config=config,
        config_digest="digest",
        model_config=_model_config(),
        extension=None,
        export_dir=tmp_path / "exports",
        model=HydraPolicyNet(hidden=8, blocks=1, bottleneck=4),
        global_step=3,
        samples_seen=11,
        completed_games=5,
    )

    assert result.batch is batch
    assert result.snapshot.inference_backend == "mahjax-gpu"
    assert result.payload["snapshot_metadata"] == result.snapshot.to_payload()


def test_mahjax_compilation_cache_uses_env_or_local_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYDRA_MAHJAX_JAX_CACHE_DIR", raising=False)
    assert _jax_compilation_cache_dir() == "local/jax_cache/mahjax_ppo"

    cache_dir = tmp_path / "jax-cache"
    monkeypatch.setenv("HYDRA_MAHJAX_JAX_CACHE_DIR", str(cache_dir))
    assert _jax_compilation_cache_dir() == str(cache_dir)


def test_mahjax_aot_toggle_defaults_on_and_accepts_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYDRA_MAHJAX_AOT", raising=False)
    assert _use_jax_aot()

    monkeypatch.setenv("HYDRA_MAHJAX_AOT", "0")
    assert not _use_jax_aot()

    monkeypatch.setenv("HYDRA_MAHJAX_AOT", "false")
    assert not _use_jax_aot()

    monkeypatch.setenv("HYDRA_MAHJAX_AOT", "on")
    assert _use_jax_aot()


def test_mahjax_aot_toggle_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRA_MAHJAX_AOT", "maybe")

    with pytest.raises(ValueError, match="HYDRA_MAHJAX_AOT"):
        _use_jax_aot()


def test_mahjax_compilation_cache_configures_once_and_ignores_optional_xla_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "jax-cache"
    updates: list[tuple[str, object]] = []

    class FakeConfig:
        def update(self, name: str, value: object) -> None:
            updates.append((name, value))
            if name == "jax_persistent_cache_enable_xla_caches":
                raise ValueError("unsupported option")

    fake_jax = SimpleNamespace(config=FakeConfig())
    monkeypatch.setenv("HYDRA_MAHJAX_JAX_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(mahjax_rollout, "_JAX_COMPILATION_CACHE_CONFIGURED_DIR", [])

    _configure_jax_compilation_cache(fake_jax)
    _configure_jax_compilation_cache(fake_jax)

    assert cache_dir.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache_dir)),
        ("jax_persistent_cache_min_compile_time_secs", 0),
        ("jax_persistent_cache_min_entry_size_bytes", -1),
        ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"),
    ]


def test_mahjax_compilation_cache_rejects_runtime_cache_dir_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeConfig:
        def update(self, name: str, value: object) -> None:
            pass

    fake_jax = SimpleNamespace(config=FakeConfig())
    first_cache_dir = tmp_path / "first-jax-cache"
    monkeypatch.setattr(mahjax_rollout, "_JAX_COMPILATION_CACHE_CONFIGURED_DIR", [])
    monkeypatch.setenv("HYDRA_MAHJAX_JAX_CACHE_DIR", str(first_cache_dir))
    _configure_jax_compilation_cache(fake_jax)

    monkeypatch.setenv("HYDRA_MAHJAX_JAX_CACHE_DIR", str(tmp_path / "second-jax-cache"))
    with pytest.raises(ValueError, match="cannot change"):
        _configure_jax_compilation_cache(fake_jax)


def test_mahjax_slot_gae_matches_reference() -> None:
    player_ids = torch.tensor([0, 1, 0, 2, 3, 1, 2, 3, 0, 1], dtype=torch.int64)
    game_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=torch.int64)
    values = torch.tensor([0.2, -0.1, 0.4, 0.05, -0.2, 0.3, 0.1, -0.4, 0.0, 0.25], dtype=torch.float32)
    final_scores = torch.tensor([[35000, 28000, 22000, 15000], [24000, 42000, 18000, 16000]], dtype=torch.int32)

    actual_adv, actual_returns = _mahjax_gae_for_slots(
        player_id=player_ids,
        value_old=values,
        game_id=game_ids,
        final_scores=final_scores,
        device=torch.device("cpu"),
    )
    expected_adv = torch.empty_like(values)
    expected_returns = torch.empty_like(values)
    for slot in (0, 1):
        indices = [index for index, game in enumerate(game_ids.tolist()) if game == slot]
        scores = final_scores[slot].tolist()
        ordered = sorted(range(4), key=lambda player: (-scores[player], player))
        placements = [0, 0, 0, 0]
        for rank, player in enumerate(ordered):
            placements[player] = rank
        gae = compute_player_local_gae(
            [PlayerDecisionStep(player_id=int(player_ids[index]), value_old=float(values[index])) for index in indices],
            final_placements=placements,
        )
        expected_adv[indices] = gae.raw_advantages
        expected_returns[indices] = gae.returns

    torch.testing.assert_close(actual_adv, expected_adv)
    torch.testing.assert_close(actual_returns, expected_returns)


def test_mahjax_slot_gae_splits_tied_final_score_rewards() -> None:
    player_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    game_ids = torch.zeros(4, dtype=torch.int64)
    values = torch.zeros(4, dtype=torch.float32)
    final_scores = torch.full((1, 4), 25000, dtype=torch.int32)

    actual_adv, actual_returns = _mahjax_gae_for_slots(
        player_id=player_ids,
        value_old=values,
        game_id=game_ids,
        final_scores=final_scores,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(actual_adv, torch.zeros_like(values))
    torch.testing.assert_close(actual_returns, torch.zeros_like(values))


def test_mahjax_parts_to_batch_uses_detached_slot_game_ids_for_gae() -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    snapshot = _snapshot_metadata()
    legal_mask = torch.ones(4, ACTION_SPACE, dtype=torch.bool)
    values = torch.tensor([0.2, -0.1, 0.4, 0.05], dtype=torch.float32)
    final_scores = torch.tensor([[35000, 28000, 22000, 15000], [24000, 42000, 18000, 16000]], dtype=torch.int32)
    game_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    player_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    part = _RowPart(
        obs=torch.zeros(4, 192, 34, dtype=torch.float32),
        legal_mask=legal_mask,
        action=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        old_logprob=torch.zeros(4, dtype=torch.float32),
        value_old=values,
        logits=torch.zeros(4, ACTION_SPACE, dtype=torch.float32),
        player_id=player_ids,
        game_id=game_ids,
        turn=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
    )

    batch = _parts_to_batch([part], final_scores=final_scores, model=model, snapshot_metadata=snapshot)
    expected_adv, expected_returns = _mahjax_gae_for_slots(
        player_id=player_ids,
        value_old=values,
        game_id=game_ids,
        final_scores=final_scores,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(batch.game_id, game_ids)
    torch.testing.assert_close(batch.seat_id, player_ids)
    torch.testing.assert_close(batch.raw_advantages, expected_adv)
    torch.testing.assert_close(batch.returns, expected_returns)


def test_mahjax_parts_to_batch_releases_row_parts_after_finalize() -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    snapshot = _snapshot_metadata()
    legal_mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
    obs = torch.zeros(2, 192, 34, dtype=torch.float32)
    logits = torch.zeros(2, ACTION_SPACE, dtype=torch.float32)
    parts = [
        _RowPart(
            obs=obs,
            legal_mask=legal_mask,
            action=torch.tensor([0, 1], dtype=torch.int64),
            old_logprob=torch.zeros(2, dtype=torch.float32),
            value_old=torch.tensor([0.2, -0.1], dtype=torch.float32),
            logits=logits,
            player_id=torch.tensor([0, 1], dtype=torch.int64),
            game_id=torch.tensor([0, 0], dtype=torch.int64),
            turn=torch.tensor([0, 1], dtype=torch.int64),
        )
    ]

    batch = _parts_to_batch(
        parts,
        final_scores=torch.tensor([[35000, 28000, 22000, 15000]], dtype=torch.int32),
        model=model,
        snapshot_metadata=snapshot,
    )

    assert parts == []
    assert batch.obs.data_ptr() != obs.data_ptr()
    assert batch.bc_logits.data_ptr() != logits.data_ptr()


def test_terminal_gae_fast_path_matches_reference() -> None:
    player_ids = [0, 1, 0, 2, 3, 1, 2, 3, 0, 1]
    values = torch.tensor([0.2, -0.1, 0.4, 0.05, -0.2, 0.3, 0.1, -0.4, 0.0, 0.25], dtype=torch.float32)
    spans = [(0, 6, (0, 1, 2, 3)), (6, 10, (2, 0, 3, 1))]
    actual_adv, actual_returns = _terminal_gae_from_cpu_values(
        player_ids_cpu=player_ids,
        value_old_cpu=values,
        game_spans=spans,
        device=torch.device("cpu"),
    )
    expected_adv = torch.empty_like(values)
    expected_returns = torch.empty_like(values)
    for start, end, placements in spans:
        gae = compute_player_local_gae(
            [
                PlayerDecisionStep(player_id=player_ids[index], value_old=float(values[index]))
                for index in range(start, end)
            ],
            final_placements=placements,
        )
        expected_adv[start:end] = gae.raw_advantages
        expected_returns[start:end] = gae.returns
    torch.testing.assert_close(actual_adv, expected_adv)
    torch.testing.assert_close(actual_returns, expected_returns)


def test_ppo_control_uses_run_local_artifact_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "campaign" / "stages" / "T1_ppo_control" / "runs" / "run-1"
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    seen: dict[str, object] = {}

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeExtension:
        pass

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    def fake_save(path: Path, *_args: object, **_kwargs: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")

    def fake_export(config: Any, **_kwargs: object) -> None:
        seen["policy_dir"] = config.output_dir
        config.output_dir.mkdir(parents=True, exist_ok=True)
        seen["export_mode"] = config.export_mode

    def fake_collect(
        _extension: object,
        _config: PpoControlConfig,
        policy_dir: Path,
        _seed: int,
        snapshot: PpoSnapshotMetadata,
    ) -> dict[str, object]:
        seen["collect_policy_dir"] = policy_dir
        seen["snapshot_id"] = snapshot.snapshot_id
        return {"snapshot_metadata": snapshot.to_payload()}

    batch = PpoBatch(
        obs=torch.zeros(2, 192, 34),
        actions=torch.zeros(2, dtype=torch.int64),
        legal_mask=torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        old_logprob=torch.zeros(2),
        value_old=torch.zeros(2),
        raw_advantages=torch.zeros(2),
        returns=torch.zeros(2),
        bc_logits=torch.zeros(2, ACTION_SPACE),
        legal_count=torch.full((2,), ACTION_SPACE, dtype=torch.int64),
    )

    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(ppo_control, "_save_t1_checkpoint", fake_save)
    monkeypatch.setattr(ppo_control, "export_loaded_policy", fake_export)
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: _FakeExtension())
    monkeypatch.setattr(ppo_control, "_collect_native_rollout", fake_collect)
    monkeypatch.setattr(ppo_control, "_batch_from_native_payload_fast", lambda _payload, _model: batch)
    monkeypatch.setattr(ppo_control, "_batch_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(ppo_control, "ppo_train_step", lambda **_kwargs: _FakeResult())

    config = PpoControlConfig(
        init_checkpoint=init_checkpoint,
        output_dir=run_dir,
        steps=1,
        games_per_update=1,
        seed=7,
        device="cpu",
        temperature=1.0,
        arena_batch_decisions=1,
        arena_threads=0,
        extension_path=tmp_path / "libfake.so",
        hidden=8,
        blocks=1,
        bottleneck=4,
        residual_profile="tiny",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        lr=1.0e-3,
        min_lr=0.0,
        lr_warmup_samples=0,
        lr_decay_samples=None,
        grad_clip_norm=None,
        microbatch_size=1,
        epochs=1,
        target_kl=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        adamw_fused="off",
        adamw_foreach="off",
        bc_kl_reverse_coef=0.0,
        entropy_alpha=1.0e-3,
        entropy_beta=1.0e-2,
        entropy_alpha_max=0.05,
        log_every_steps=1,
        checkpoint_every_steps=1,
        keep_step_checkpoints=True,
        resume=None,
        tensorboard_dir=None,
        quiet=True,
        rollout_inference="rust-ort",
        ppo_pipeline_depth=0,
    )

    summary = ppo_control.run_ppo_control(config)

    assert (run_dir / "logs" / "events.jsonl").is_file()
    assert (run_dir / "logs" / "train_steps.jsonl").is_file()
    assert (run_dir / "logs" / "tensorboard").is_dir()
    assert (run_dir / "checkpoints" / "latest.pt").is_file()
    assert (run_dir / "checkpoints" / "step_1.pt").is_file()
    assert (run_dir / "exports" / "onnx_step_00000000").is_dir()
    assert (run_dir / "rollouts").is_dir()
    assert (run_dir / "eval").is_dir()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "launch_metadata.json").is_file()
    assert seen["policy_dir"] == run_dir / "exports" / "onnx_step_00000000"
    assert seen["collect_policy_dir"] == run_dir / "exports" / "onnx_step_00000000"
    assert seen["export_mode"] == "ppo_policy_value"
    assert summary["paths"] == {
        "run_dir": str(run_dir),
        "logs": str(run_dir / "logs"),
        "checkpoints": str(run_dir / "checkpoints"),
        "exports": str(run_dir / "exports"),
        "rollouts": str(run_dir / "rollouts"),
        "eval": str(run_dir / "eval"),
        "tensorboard": str(run_dir / "logs" / "tensorboard"),
    }
    with (run_dir / "summary.json").open(encoding="utf-8") as file:
        assert json.load(file) == summary


def test_native_binary_payload_uses_cached_policy_scalars_without_model_recompute() -> None:
    row_count = 2
    obs = torch.arange(row_count * 192 * 34, dtype=torch.float32).reshape(row_count, 192, 34) / 1000.0
    legal_mask = torch.zeros(row_count, ACTION_SPACE, dtype=torch.uint8)
    legal_mask[0, :3] = 1
    legal_mask[1, 2:6] = 1
    actions = bytes([1, 3])
    legal_counts = bytes([3, 4])
    player_ids = bytes([0, 1])
    seat_ids = bytes([0, 1])
    game_ids = torch.tensor([0, 0], dtype=torch.uint64).numpy().tobytes()
    turns = torch.tensor([1, 2], dtype=torch.uint32).numpy().tobytes()
    starts = torch.tensor([0], dtype=torch.uint64).numpy().tobytes()
    ends = torch.tensor([2], dtype=torch.uint64).numpy().tobytes()
    placements = bytes([0, 1, 2, 3])
    old_logits = torch.full((row_count, ACTION_SPACE), -9.0, dtype=torch.float32)
    old_logits[0, 1] = 4.0
    old_logits[1, 3] = 5.0
    value_old = torch.tensor([0.25, -0.5], dtype=torch.float32)
    old_logprob = torch.tensor([-0.125, -0.25], dtype=torch.float32)
    raw_advantages = torch.tensor([1.25, -1.5], dtype=torch.float32)
    returns = torch.tensor([1.5, -2.0], dtype=torch.float32)

    class ExplodingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def policy_value(self, _obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("native exact payload must not recompute policy_value")

    payload: dict[str, object] = {
        "obs_f32_le": bytearray(obs.numpy().tobytes()),
        "legal_mask_u8": bytearray(legal_mask.numpy().tobytes()),
        "actions": bytearray(actions),
        "legal_counts": bytearray(legal_counts),
        "player_ids": bytearray(player_ids),
        "seat_ids": bytearray(seat_ids),
        "game_ids_u64_le": bytearray(game_ids),
        "turns_u32_le": bytearray(turns),
        "game_row_starts_u64_le": bytearray(starts),
        "game_row_ends_u64_le": bytearray(ends),
        "placements_u8": bytearray(placements),
        "old_logits_f32_le": bytearray(old_logits.numpy().tobytes()),
        "value_old_f32_le": bytearray(value_old.numpy().tobytes()),
        "old_logprob_f32_le": bytearray(old_logprob.numpy().tobytes()),
        "raw_advantages_f32_le": bytearray(raw_advantages.numpy().tobytes()),
        "returns_f32_le": bytearray(returns.numpy().tobytes()),
        "row_count": row_count,
        "games": [{"game_id": 0, "placements": [0, 1, 2, 3]}],
    }

    batch = _batch_from_native_payload_fast(payload, cast("HydraPolicyNet", ExplodingModel()))

    torch.testing.assert_close(batch.bc_logits, old_logits)
    torch.testing.assert_close(batch.value_old, value_old)
    torch.testing.assert_close(batch.old_logprob, old_logprob)
    torch.testing.assert_close(batch.raw_advantages, raw_advantages)
    torch.testing.assert_close(batch.returns, returns)


def test_native_binary_legal_only_payload_reconstructs_masked_logits_and_validates() -> None:
    row_count = 2
    obs = torch.arange(row_count * 192 * 34, dtype=torch.float32).reshape(row_count, 192, 34) / 1000.0
    legal_mask = torch.zeros(row_count, ACTION_SPACE, dtype=torch.uint8)
    legal_mask[0, [0, 2, 4]] = 1
    legal_mask[1, [1, 3, 5, 7]] = 1
    full_logits = torch.full((row_count, ACTION_SPACE), 99.0, dtype=torch.float32)
    full_logits[0, [0, 2, 4]] = torch.tensor([0.25, -0.5, 1.0])
    full_logits[1, [1, 3, 5, 7]] = torch.tensor([-1.25, 0.0, 0.75, 1.25])
    old_legal_logits = full_logits[legal_mask.to(dtype=torch.bool)]
    value_old = torch.tensor([0.25, -0.5], dtype=torch.float32)
    old_logprob = masked_log_prob(full_logits, legal_mask.to(dtype=torch.bool), torch.tensor([4, 7]))
    raw_advantages = torch.tensor([1.25, -1.5], dtype=torch.float32)
    returns = torch.tensor([1.5, -2.0], dtype=torch.float32)

    class ExplodingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def policy_value(self, _obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("legal-only exact payload must not recompute policy_value")

    payload: dict[str, object] = {
        "obs_f32_le": bytearray(obs.numpy().tobytes()),
        "legal_mask_u8": bytearray(legal_mask.numpy().tobytes()),
        "actions": bytearray([4, 7]),
        "legal_counts": bytearray([3, 4]),
        "player_ids": bytearray([0, 1]),
        "seat_ids": bytearray([0, 1]),
        "game_ids_u64_le": bytearray(torch.tensor([0, 0], dtype=torch.uint64).numpy().tobytes()),
        "turns_u32_le": bytearray(torch.tensor([1, 2], dtype=torch.uint32).numpy().tobytes()),
        "game_row_starts_u64_le": bytearray(torch.tensor([0], dtype=torch.uint64).numpy().tobytes()),
        "game_row_ends_u64_le": bytearray(torch.tensor([2], dtype=torch.uint64).numpy().tobytes()),
        "placements_u8": bytearray([0, 1, 2, 3]),
        "old_legal_logits_f32_le": bytearray(old_legal_logits.numpy().tobytes()),
        "value_old_f32_le": bytearray(value_old.numpy().tobytes()),
        "old_logprob_f32_le": bytearray(old_logprob.numpy().tobytes()),
        "raw_advantages_f32_le": bytearray(raw_advantages.numpy().tobytes()),
        "returns_f32_le": bytearray(returns.numpy().tobytes()),
        "row_count": row_count,
        "games": [{"game_id": 0, "placements": [0, 1, 2, 3]}],
    }

    batch = _batch_from_native_payload_fast(payload, cast("HydraPolicyNet", ExplodingModel()))

    batch.validate()
    torch.testing.assert_close(
        masked_log_softmax(batch.bc_logits, batch.legal_mask), masked_log_softmax(full_logits, batch.legal_mask)
    )
    assert torch.count_nonzero(batch.bc_logits.masked_select(~batch.legal_mask)) == 0
    torch.testing.assert_close(batch.value_old, value_old)
    torch.testing.assert_close(batch.old_logprob, old_logprob)


def test_mahjax_completion_sync_interval_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL", raising=False)
    assert _completion_sync_interval() == 32

    monkeypatch.setenv("HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL", "64")
    assert _completion_sync_interval() == 64

    monkeypatch.setenv("HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _completion_sync_interval()


def test_native_binary_legal_only_payload_rejects_bad_length() -> None:
    row_count = 1
    obs = torch.zeros(row_count, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(row_count, ACTION_SPACE, dtype=torch.uint8)
    legal_mask[0, :3] = 1
    payload: dict[str, object] = {
        "obs_f32_le": bytearray(obs.numpy().tobytes()),
        "legal_mask_u8": bytearray(legal_mask.numpy().tobytes()),
        "actions": bytearray([0]),
        "legal_counts": bytearray([3]),
        "player_ids": bytearray([0]),
        "seat_ids": bytearray([0]),
        "game_ids_u64_le": bytearray(torch.tensor([0], dtype=torch.uint64).numpy().tobytes()),
        "turns_u32_le": bytearray(torch.tensor([1], dtype=torch.uint32).numpy().tobytes()),
        "game_row_starts_u64_le": bytearray(torch.tensor([0], dtype=torch.uint64).numpy().tobytes()),
        "game_row_ends_u64_le": bytearray(torch.tensor([1], dtype=torch.uint64).numpy().tobytes()),
        "placements_u8": bytearray([0, 1, 2, 3]),
        "old_legal_logits_f32_le": bytearray(torch.zeros(2, dtype=torch.float32).numpy().tobytes()),
        "value_old_f32_le": bytearray(torch.zeros(1, dtype=torch.float32).numpy().tobytes()),
        "old_logprob_f32_le": bytearray(torch.zeros(1, dtype=torch.float32).numpy().tobytes()),
        "raw_advantages_f32_le": bytearray(torch.ones(1, dtype=torch.float32).numpy().tobytes()),
        "returns_f32_le": bytearray(torch.ones(1, dtype=torch.float32).numpy().tobytes()),
        "row_count": row_count,
        "games": [{"game_id": 0, "placements": [0, 1, 2, 3]}],
    }
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)

    with pytest.raises(ValueError, match="old_legal_logits_f32_le length"):
        _batch_from_native_payload_fast(payload, model)


def test_ppo_control_logs_stage_and_resource_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "campaign" / "stages" / "T1_ppo_control" / "runs" / "run-telemetry"
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    batch = PpoBatch(
        obs=torch.zeros(2, 192, 34),
        actions=torch.zeros(2, dtype=torch.int64),
        legal_mask=torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        old_logprob=torch.zeros(2),
        value_old=torch.zeros(2),
        raw_advantages=torch.zeros(2),
        returns=torch.zeros(2),
        bc_logits=torch.zeros(2, ACTION_SPACE),
        legal_count=torch.full((2,), ACTION_SPACE, dtype=torch.int64),
    )

    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(
        ppo_control, "_save_t1_checkpoint", lambda path, *_args, **_kwargs: path.write_bytes(b"checkpoint")
    )
    monkeypatch.setattr(
        ppo_control,
        "export_loaded_policy",
        lambda config, **_kwargs: config.output_dir.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: object())
    monkeypatch.setattr(
        ppo_control,
        "_collect_native_rollout",
        lambda *_args, **_kwargs: {
            "timing": {
                "mean_requests_per_batch": 1.0,
                "batch_bucket_le_64": 1,
                "active_games_mean": 1.0,
                "shard_request_min_mean": 1.0,
                "shard_request_max_mean": 1.0,
                "shard_unused_quota": 0,
                "shard_request_deficit": 0,
                "collection_passes": 1,
            }
        },
    )
    monkeypatch.setattr(ppo_control, "_batch_from_native_payload_fast", lambda _payload, _model: batch)
    monkeypatch.setattr(ppo_control, "ppo_train_step", lambda **_kwargs: _FakeResult())

    config = PpoControlConfig(
        init_checkpoint=init_checkpoint,
        output_dir=run_dir,
        steps=1,
        games_per_update=1,
        seed=7,
        device="cpu",
        temperature=1.0,
        arena_batch_decisions=1,
        arena_threads=0,
        extension_path=tmp_path / "libfake.so",
        hidden=8,
        blocks=1,
        bottleneck=4,
        residual_profile="tiny",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        lr=1.0e-3,
        min_lr=0.0,
        lr_warmup_samples=0,
        lr_decay_samples=None,
        grad_clip_norm=None,
        microbatch_size=1,
        epochs=1,
        target_kl=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        adamw_fused="off",
        adamw_foreach="off",
        bc_kl_reverse_coef=0.0,
        entropy_alpha=1.0e-3,
        entropy_beta=1.0e-2,
        entropy_alpha_max=0.05,
        log_every_steps=1,
        checkpoint_every_steps=1,
        keep_step_checkpoints=False,
        resume=None,
        tensorboard_dir=None,
        quiet=True,
        rollout_inference="rust-ort",
        ppo_pipeline_depth=0,
    )

    summary = ppo_control.run_ppo_control(config)
    with (run_dir / "logs" / "events.jsonl").open(encoding="utf-8") as file:
        run_start = json.loads(file.readline())
    with (run_dir / "logs" / "train_steps.jsonl").open(encoding="utf-8") as file:
        train_step_event = json.loads(file.readline())
    train_step = {key: value for key, value in train_step_event.items() if key not in {"event", "ts"}}
    summary_payload = cast("dict[str, Any]", summary["summary"])

    assert "resources/start/gpu_util_percent" in run_start
    for key in (
        "checkpoint_save_ms",
        "onnx_export_ms",
        "native_rollout_ms",
        "batch_build_ms",
        "h2d_ms",
        "train_step_ms",
        "resources/update/cpu_percent",
        "resources/update/disk_read_mb_s",
        "resources/update/disk_write_mb_s",
        "resources/update/gpu_util_percent",
        "native_timing/mean_requests_per_batch",
        "native_timing/batch_bucket_le_64",
        "native_timing/active_games_mean",
        "native_timing/shard_request_min_mean",
        "native_timing/shard_request_max_mean",
        "native_timing/shard_unused_quota",
        "native_timing/shard_request_deficit",
        "native_timing/collection_passes",
        "snapshot_id",
        "snapshot_global_step",
        "snapshot_samples_seen",
        "snapshot_completed_games",
        "pipeline_depth",
        "pipeline_enabled",
        "rollout_wait_ms",
        "train_overlap_ms",
        "overlap_efficiency",
        "future_rollout_ms",
        "in_flight_discarded",
    ):
        assert key in train_step_event
    assert train_step_event["snapshot_global_step"] == 0
    assert train_step_event["snapshot_samples_seen"] == 0
    assert train_step_event["snapshot_completed_games"] == 0
    assert isinstance(train_step_event["snapshot_id"], str)
    assert train_step_event["pipeline_depth"] == 0
    assert train_step_event["pipeline_enabled"] is False
    assert summary_payload["last_train_metrics"] == train_step


def test_batch_to_device_returns_same_batch_when_already_on_target() -> None:
    batch = _valid_batch()

    moved = _batch_to_device(batch, torch.device("cpu"))

    assert moved is batch


def test_native_payload_path_preserves_snapshot_metadata() -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    snapshot = _snapshot_metadata()
    payload: dict[str, object] = {
        "snapshot_metadata": snapshot.to_payload(),
        "row_count": 1,
        "obs_f32_le": bytearray(torch.zeros(1, 192, 34, dtype=torch.float32).numpy().tobytes()),
        "legal_mask_u8": bytearray([1] * ACTION_SPACE),
        "actions": bytearray([0]),
        "legal_counts": bytearray([ACTION_SPACE]),
        "player_ids": bytearray([0]),
        "seat_ids": bytearray([0]),
        "game_ids_u64_le": bytearray((0).to_bytes(8, "little")),
        "turns_u32_le": bytearray((0).to_bytes(4, "little")),
        "game_row_starts_u64_le": bytearray((0).to_bytes(8, "little")),
        "game_row_ends_u64_le": bytearray((1).to_bytes(8, "little")),
        "placements_u8": bytearray([0, 1, 2, 3]),
        "old_logits_f32_le": bytearray(torch.zeros(1, ACTION_SPACE, dtype=torch.float32).numpy().tobytes()),
        "value_old_f32_le": bytearray(torch.zeros(1, dtype=torch.float32).numpy().tobytes()),
        "old_logprob_f32_le": bytearray(torch.zeros(1, dtype=torch.float32).numpy().tobytes()),
        "raw_advantages_f32_le": bytearray(torch.ones(1, dtype=torch.float32).numpy().tobytes()),
        "returns_f32_le": bytearray(torch.ones(1, dtype=torch.float32).numpy().tobytes()),
        "games": [{"game_id": 0, "seed": 7, "placements": [0, 1, 2, 3]}],
    }

    batch = _batch_from_native_payload_fast(payload, model)

    assert batch.snapshot_metadata == snapshot.to_payload()
    torch.testing.assert_close(batch.old_logprob, torch.zeros(1), rtol=0.0, atol=0.0)


def test_ppo_inference_callback_returns_packed_reusable_memoryview() -> None:
    torch.manual_seed(83)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    callback = _debug_ppo_callback(_make_ppo_inference_callback(model, torch.device("cpu"), initial_capacity=3))
    obs = torch.randn(3, 192, 34, dtype=torch.float32)

    packed = callback(bytearray(obs.numpy().tobytes()), obs.shape[0])

    assert isinstance(packed, memoryview)
    packed_array = np.frombuffer(packed, dtype=np.float32).reshape(3, ACTION_SPACE + 1)
    assert packed_array.shape == (3, ACTION_SPACE + 1)
    assert packed_array.dtype == np.float32
    assert packed_array.flags.c_contiguous
    timings = callback._timings
    assert timings["callback_obs_h2d_ms"] >= 0.0
    assert timings["callback_forward_ms"] >= 0.0
    assert timings["callback_d2h_pack_ms"] >= 0.0
    assert timings["callback_pack_copy_ms"] >= 0.0
    assert timings["callback_return_view_ms"] >= 0.0
    with torch.inference_mode():
        expected_logits, expected_values = model.policy_value(obs)
    np.testing.assert_allclose(packed_array[:, :ACTION_SPACE], expected_logits.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(packed_array[:, ACTION_SPACE], expected_values.reshape(3).numpy(), rtol=0.0, atol=0.0)

    initial_ptr = callback._packed_buffer_ptr()
    initial_capacity = callback._packed_capacity()
    callback(bytearray(obs[:2].numpy().tobytes()), 2)
    assert callback._packed_buffer_ptr() == initial_ptr
    assert callback._packed_capacity() == initial_capacity

    larger_obs = torch.randn(5, 192, 34, dtype=torch.float32)
    callback(bytearray(larger_obs.numpy().tobytes()), 5)
    assert callback._packed_capacity() == 5
    assert callback._packed_buffer_ptr() != initial_ptr


def test_ppo_inference_callback_returns_legal_only_packed_memoryview() -> None:
    torch.manual_seed(84)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    callback = _debug_ppo_callback(_make_ppo_inference_callback(model, torch.device("cpu"), initial_capacity=1))
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.uint8)
    legal_mask[0, [0, 3, 5]] = 1
    legal_mask[1, [2, 4]] = 1

    packed = callback(bytearray(obs.numpy().tobytes()), bytearray(legal_mask.numpy().tobytes()), obs.shape[0])
    assert isinstance(packed, memoryview)

    packed_array = np.frombuffer(packed, dtype=np.float32)
    assert packed_array.shape == (7,)
    timings = callback._timings
    assert timings["callback_legal_gather_ms"] >= 0.0
    assert timings["callback_legal_d2h_pack_ms"] >= 0.0
    assert timings["callback_d2h_pack_ms"] >= timings["callback_legal_d2h_pack_ms"]
    assert timings["callback_legal_transport_ratio"] == pytest.approx(7 / (2 * (ACTION_SPACE + 1)))
    with torch.inference_mode():
        expected_logits, expected_values = model.policy_value(obs)
    expected = torch.cat(
        [expected_logits[legal_mask.to(dtype=torch.bool)], expected_values.reshape(2)],
        dim=0,
    )
    np.testing.assert_allclose(packed_array, expected.numpy(), rtol=0.0, atol=0.0)


def test_ppo_inference_callback_rejects_negative_rows() -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    callback = _make_ppo_inference_callback(model, torch.device("cpu"))

    with pytest.raises(ValueError, match="rows must be non-negative"):
        callback(bytearray(), -1)


def _snapshot_metadata() -> PpoSnapshotMetadata:
    return build_ppo_snapshot_metadata(
        config_digest_sha256="a" * 64,
        global_step=3,
        samples_seen=20,
        completed_games=5,
        rollout_seed=12,
        temperature=0.8,
        inference_backend="torch-callback",
        hidden=8,
        blocks=1,
        bottleneck=4,
        residual_profile="tiny",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        encoder_shape=(192, 34),
        action_space=ACTION_SPACE,
    )


def test_ppo_rollout_artifact_roundtrip_and_batch_conversion(tmp_path: Path) -> None:
    batch = _valid_batch()
    path = tmp_path / "rollout.pt"
    snapshot = _snapshot_metadata()
    metadata = PpoRolloutMetadata(
        rank_utility_used="U_A",
        gae_gamma=0.995,
        gae_lambda=0.95,
        reward_shaping=default_reward_shaping_metadata(gamma=0.995, gae_lambda=0.95),
        snapshot=snapshot,
    )
    save_ppo_rollout_artifact(path, batch, metadata)
    artifact = load_ppo_rollout_artifact(path)
    loaded_batch = artifact_to_ppo_batch(artifact)

    assert artifact.metadata == metadata
    assert loaded_batch.snapshot_metadata == snapshot.to_payload()
    assert artifact.metadata.snapshot is not None
    assert artifact.metadata.snapshot.snapshot_id == snapshot.snapshot_id
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


def test_ppo_rollout_artifact_rejects_malformed_snapshot_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad-snapshot.pt"
    payload = _artifact_payload(_valid_batch())
    metadata = cast("dict[str, object]", payload["metadata"])
    snapshot = _snapshot_metadata().to_payload()
    snapshot.pop("snapshot_id")
    metadata["snapshot"] = snapshot
    torch.save(payload, path)

    with pytest.raises(ValueError, match="snapshot_id"):
        load_ppo_rollout_artifact(path)


def test_ppo_rollout_artifact_rejects_new_schema_missing_snapshot_metadata(tmp_path: Path) -> None:
    path = tmp_path / "missing-snapshot.pt"
    payload = _artifact_payload(_valid_batch())
    metadata = cast("dict[str, object]", payload["metadata"])
    metadata["snapshot_required"] = True
    torch.save(payload, path)

    with pytest.raises(ValueError, match="snapshot"):
        load_ppo_rollout_artifact(path)


def test_ppo_rollout_default_reward_shaping_uses_artifact_gamma_lambda(tmp_path: Path) -> None:
    path = tmp_path / "custom-gamma.pt"
    save_ppo_rollout_artifact(
        path,
        _valid_batch(),
        PpoRolloutMetadata(rank_utility_used="U_A", gae_gamma=0.9, gae_lambda=0.8, reward_shaping=None),
    )

    artifact = load_ppo_rollout_artifact(path)
    reward_shaping = cast("dict[str, object]", artifact.metadata.reward_shaping)

    assert artifact.metadata.gae_gamma == pytest.approx(0.9)
    assert artifact.metadata.gae_lambda == pytest.approx(0.8)
    assert reward_shaping["enabled"] is False
    assert reward_shaping["gae_gamma"] == pytest.approx(0.9)
    assert reward_shaping["gae_lambda"] == pytest.approx(0.8)


def test_legacy_ppo_rollout_missing_reward_shaping_uses_artifact_gamma_lambda(tmp_path: Path) -> None:
    path = tmp_path / "legacy-custom-gamma.pt"
    payload = _artifact_payload(_valid_batch())
    metadata = cast("dict[str, object]", payload["metadata"])
    metadata["gae_gamma"] = 0.9
    metadata["gae_lambda"] = 0.8
    metadata.pop("reward_shaping", None)
    torch.save(payload, path)

    artifact = load_ppo_rollout_artifact(path)
    reward_shaping = cast("dict[str, object]", artifact.metadata.reward_shaping)

    assert artifact.metadata.gae_gamma == pytest.approx(0.9)
    assert artifact.metadata.gae_lambda == pytest.approx(0.8)
    assert reward_shaping["enabled"] is False
    assert reward_shaping["gae_gamma"] == pytest.approx(0.9)
    assert reward_shaping["gae_lambda"] == pytest.approx(0.8)


def test_ppo_rollout_artifact_rejects_incomplete_enabled_reward_shaping(tmp_path: Path) -> None:
    path = tmp_path / "bad-shaping.pt"
    payload = _artifact_payload(_valid_batch())
    metadata = cast("dict[str, object]", payload["metadata"])
    metadata["reward_shaping"] = {"enabled": True, "kind": "pbrs"}
    torch.save(payload, path)

    with pytest.raises(ValueError, match="reward_shaping"):
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
    assert cast("dict[str, object]", result.artifact_metadata["reward_shaping"])["enabled"] is False
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


def test_ppo_zero_bc_kl_skips_bad_reference_logits() -> None:
    torch.manual_seed(371)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    batch = _batch_from_model(model, raw_advantages=torch.tensor([0.5, -0.25], dtype=torch.float32))
    batch = replace(batch, bc_logits=torch.full_like(batch.bc_logits, torch.nan))

    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=PpoTrainStepConfig(value_coef=0.0, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )

    assert result.metrics["bc_kl_reverse"] == pytest.approx(0.0)
    json.dumps(result.metrics, allow_nan=False)


def test_ppo_positive_bc_kl_still_validates_reference_logits() -> None:
    torch.manual_seed(372)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    batch = _batch_from_model(model, raw_advantages=torch.tensor([0.5, -0.25], dtype=torch.float32))
    batch = replace(batch, bc_logits=torch.full_like(batch.bc_logits, torch.nan))

    with pytest.raises(ValueError, match="bc_logits"):
        ppo_train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
            config=PpoTrainStepConfig(value_coef=0.0, bc_kl_reverse_coef=0.01, grad_clip_norm=None),
        )


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


def test_ppo_rollout_smoke_artifact_update_metrics_checkpoint_and_init_reload(tmp_path: Path) -> None:
    torch.manual_seed(61)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    rollout = _tiny_rust_rollout_fixture()
    artifact_path = tmp_path / "ppo-rollout.pt"

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
    artifact = load_ppo_rollout_artifact(artifact_result.artifact_path)
    assert artifact.metadata.reward_shaping is not None
    assert artifact.metadata.reward_shaping["enabled"] is False
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


def test_ppo_rollout_smoke_artifact_is_deterministic_for_same_seed_and_model(tmp_path: Path) -> None:
    torch.manual_seed(67)
    first_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    second_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    second_model.load_state_dict(first_model.state_dict())
    rollout = _tiny_rust_rollout_fixture()

    first = write_ppo_smoke_rollout_artifact(tmp_path / "first.pt", rollout, model=first_model, torch_seed=99).batch
    second = write_ppo_smoke_rollout_artifact(tmp_path / "second.pt", rollout, model=second_model, torch_seed=99).batch

    for name in ("obs", "actions", "legal_mask", "old_logprob", "value_old", "raw_advantages", "returns", "bc_logits"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0.0, atol=0.0)


def test_ppo_rollout_batch_uses_model_device_without_moving_model() -> None:
    torch.manual_seed(69)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    rollout = _tiny_rust_rollout_fixture()
    before_device = next(model.parameters()).device

    batch = build_ppo_batch_from_rust_rollout(rollout, model=model, torch_seed=99, output_device=before_device)

    assert next(model.parameters()).device == before_device
    assert batch.obs.device == before_device
    assert batch.old_logprob.device == before_device
    batch.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU rollout batch path")
def test_ppo_rollout_batch_accepts_cuda_model_without_cpu_demote() -> None:
    torch.manual_seed(70)
    device = torch.device("cuda:0")
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4).to(device)
    rollout = _tiny_rust_rollout_fixture()
    before_device = next(model.parameters()).device

    batch = build_ppo_batch_from_rust_rollout(rollout, model=model, torch_seed=99, output_device=device)
    torch.cuda.synchronize(device)

    assert next(model.parameters()).device == before_device
    assert batch.obs.device == device
    assert batch.old_logprob.device == device
    batch.validate()


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
def test_ppo_rollout_negative_boundary_errors(
    mutate: Callable[[RustGameRollout], RustGameRollout], match: str, tmp_path: Path
) -> None:
    torch.manual_seed(71)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    bad = mutate(_tiny_rust_rollout_fixture())
    with pytest.raises((TypeError, ValueError), match=match):
        write_ppo_smoke_rollout_artifact(tmp_path / "bad.pt", bad, model=model, torch_seed=5)


def _valid_batch(snapshot_metadata: dict[str, object] | None = None) -> PpoBatch:
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
        snapshot_metadata=snapshot_metadata,
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


def _tiny_rust_rollout_fixture() -> RustGameRollout:
    rollout = tiny_ppo_rollout()
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


def test_ppo_policy_snapshot_artifact_roundtrip_strict_loads(tmp_path: Path) -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4, residual_profile="mish_se")
    model_config = ModelConfig(hidden=8, blocks=1, bottleneck=4)
    snapshot = _snapshot_metadata()

    path, size = save_ppo_policy_snapshot_artifact(tmp_path, model=model, model_config=model_config, snapshot=snapshot)
    artifact = load_ppo_policy_snapshot_artifact(path, expected_snapshot=snapshot, expected_model_config=model_config)

    assert path.name == f"snapshot_{snapshot.global_step:08d}_{snapshot.snapshot_id}.pt"
    assert size > 0
    assert not path.with_suffix(".pt.tmp").exists()
    assert artifact.snapshot_metadata.snapshot_id == snapshot.snapshot_id
    assert artifact.model_config == model_config
    assert set(artifact.model_state) == set(model.state_dict())


def test_ppo_policy_snapshot_artifact_rejects_wrong_snapshot_id(tmp_path: Path) -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4, residual_profile="mish_se")
    model_config = ModelConfig(hidden=8, blocks=1, bottleneck=4)
    snapshot = _snapshot_metadata()
    other = build_ppo_snapshot_metadata(
        config_digest_sha256=snapshot.config_digest_sha256,
        global_step=snapshot.global_step + 1,
        samples_seen=snapshot.samples_seen,
        completed_games=snapshot.completed_games,
        rollout_seed=snapshot.rollout_seed,
        temperature=snapshot.temperature,
        inference_backend=snapshot.inference_backend,
        hidden=snapshot.hidden,
        blocks=snapshot.blocks,
        bottleneck=snapshot.bottleneck,
        residual_profile=snapshot.residual_profile,
        backbone_profile=snapshot.backbone_profile,
        conv_memory_format=snapshot.conv_memory_format,
        encoder_shape=snapshot.encoder_shape,
        action_space=snapshot.action_space,
    )
    path, _size = save_ppo_policy_snapshot_artifact(tmp_path, model=model, model_config=model_config, snapshot=snapshot)

    with pytest.raises(ValueError, match="metadata does not match"):
        load_ppo_policy_snapshot_artifact(path, expected_snapshot=other, expected_model_config=model_config)


def test_ppo_pipeline_rejects_mixed_snapshot_batch() -> None:
    snapshot = _snapshot_metadata()
    batch = _valid_batch(snapshot_metadata={**snapshot.to_payload(), "snapshot_id": "other"})

    with pytest.raises(ValueError, match="different policies"):
        ppo_control._validate_batch_snapshot(batch, snapshot)


def test_ppo_frozen_snapshot_does_not_share_parameter_storage(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = PpoControlConfig(
        init_checkpoint=init_checkpoint,
        output_dir=tmp_path / "run",
        steps=1,
        games_per_update=1,
        seed=7,
        device="cpu",
        temperature=1.0,
        arena_batch_decisions=1,
        arena_threads=0,
        extension_path=tmp_path / "libfake.so",
        hidden=8,
        blocks=1,
        bottleneck=4,
        residual_profile="mish_se",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        lr=1.0e-3,
        min_lr=0.0,
        lr_warmup_samples=0,
        lr_decay_samples=None,
        grad_clip_norm=None,
        microbatch_size=1,
        epochs=1,
        target_kl=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        adamw_fused="off",
        adamw_foreach="off",
        bc_kl_reverse_coef=0.0,
        entropy_alpha=1.0e-3,
        entropy_beta=1.0e-2,
        entropy_alpha_max=0.05,
        log_every_steps=1,
        checkpoint_every_steps=1,
        keep_step_checkpoints=False,
        resume=None,
        tensorboard_dir=None,
        quiet=True,
        rollout_inference="rust-ort",
        ppo_pipeline_depth=1,
    )
    live = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)

    frozen = ppo_control._capture_frozen_model_snapshot(config, live)
    live_param = next(live.parameters())
    frozen_param = next(frozen.parameters())
    before = frozen_param.detach().clone()
    with torch.no_grad():
        live_param.add_(1.0)

    assert live_param.data_ptr() != frozen_param.data_ptr()
    torch.testing.assert_close(frozen_param, before)
    assert frozen.training is False


def test_ppo_pipeline_depth_one_snapshot_order_and_batch_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    order: list[str] = []

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    def collect(**kwargs: object) -> ppo_control._RolloutResult:
        step = cast("int", kwargs["global_step"])
        samples = cast("int", kwargs["samples_seen"])
        completed = cast("int", kwargs["completed_games"])
        cfg = cast("PpoControlConfig", kwargs["config"])
        snapshot = ppo_control._snapshot_metadata(
            cfg, cast("str", kwargs["config_digest"]), step, samples, completed, cfg.seed + completed
        )
        order.append(f"snapshot {step}")
        order.append(f"rollout {step}")
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload()},
            batch=_valid_batch(snapshot_metadata=snapshot.to_payload()),
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=0.0,
        )

    def rollout_from_snapshot(**kwargs: object) -> ppo_control._RolloutResult:
        snapshot = cast("PpoSnapshotMetadata", kwargs["expected_snapshot"])
        order.append(f"snapshot {snapshot.global_step}")
        order.append(f"rollout {snapshot.global_step}")
        batch = _valid_batch(snapshot_metadata=snapshot.to_payload())
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload()},
            batch=batch,
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=25.0,
            child_start_load_ms=3.0,
        )

    class _FakeFuture:
        def __init__(self, result: ppo_control._RolloutResult) -> None:
            self._result = result

        def result(self) -> ppo_control._RolloutResult:
            return self._result

        def cancel(self) -> bool:
            return True

    class _FakeExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(self, fn: Callable[..., ppo_control._RolloutResult], **kwargs: object) -> _FakeFuture:
            return _FakeFuture(fn(**kwargs))

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            assert cancel_futures is True

    def train(**_kwargs: object) -> _FakeResult:
        order.append(f"train {len([item for item in order if item.startswith('train')])}")
        return _FakeResult()

    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: object())
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch", collect)
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch_from_snapshot_artifact", rollout_from_snapshot)
    monkeypatch.setattr(ppo_control, "ppo_train_step", train)
    monkeypatch.setattr(ppo_control, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        ppo_control,
        "save_ppo_policy_snapshot_artifact",
        lambda _run_dir, **_kwargs: (tmp_path / "snapshot.pt", 7),
    )
    monkeypatch.setattr(
        ppo_control,
        "_save_t1_checkpoint",
        lambda path, *_args, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_bytes(b"checkpoint"),
    )
    config = _ppo_control_config(tmp_path, init_checkpoint, steps=2, pipeline_depth=1)

    summary = ppo_control.run_ppo_control(config)

    assert order == ["snapshot 0", "rollout 0", "train 0", "snapshot 1", "rollout 1", "train 1"]
    metrics = cast("dict[str, Any]", cast("dict[str, Any]", summary["summary"])["last_train_metrics"])
    assert metrics["pipeline_enabled"] is True
    assert metrics["pipeline_mode"] == "process_short_lived"
    assert metrics["snapshot_global_step"] == 1
    assert metrics["train_overlap_ms"] > 0.0
    assert metrics["overlap_efficiency"] > 0.0
    assert metrics["snapshot_save_ms"] >= 0.0
    assert metrics["snapshot_artifact_bytes"] == 7
    assert metrics["child_start_load_ms"] == 3.0


def test_ppo_mahjax_pipeline_depth_uses_serial_gpu_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    submitted: bool | None = False

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    class _FakeExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(self, *_args: object, **_kwargs: object) -> object:
            nonlocal submitted
            submitted = True
            raise AssertionError("mahjax-gpu must not spawn the process pipeline")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            del wait, cancel_futures

    def collect(**kwargs: object) -> ppo_control._RolloutResult:
        cfg = cast("PpoControlConfig", kwargs["config"])
        snapshot = ppo_control._snapshot_metadata(
            cfg,
            cast("str", kwargs["config_digest"]),
            cast("int", kwargs["global_step"]),
            cast("int", kwargs["samples_seen"]),
            cast("int", kwargs["completed_games"]),
            cfg.seed + cast("int", kwargs["completed_games"]),
        )
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload(), "timing": {}},
            batch=_valid_batch(snapshot_metadata=snapshot.to_payload()),
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=0.0,
        )

    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(
        ppo_control, "_load_extension", lambda _path: (_ for _ in ()).throw(AssertionError("extension loaded"))
    )
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch", collect)
    monkeypatch.setattr(ppo_control, "ppo_train_step", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr(ppo_control, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        ppo_control,
        "_save_t1_checkpoint",
        lambda path, *_args, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True)
        or path.write_bytes(b"checkpoint"),
    )
    config = replace(
        _ppo_control_config(tmp_path, init_checkpoint, steps=2, pipeline_depth=1), rollout_inference="mahjax-gpu"
    )

    summary = ppo_control.run_ppo_control(config)

    assert not submitted
    metrics = cast("dict[str, Any]", cast("dict[str, Any]", summary["summary"])["last_train_metrics"])
    assert metrics["pipeline_depth"] == 1
    assert metrics["pipeline_enabled"] is False
    assert metrics["pipeline_mode"] == "mahjax_serial_gpu"


def test_ppo_pipeline_failed_future_aborts_before_next_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    train_calls = 0

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    def collect(**kwargs: object) -> ppo_control._RolloutResult:
        step = cast("int", kwargs["global_step"])
        if step == 1:
            raise RuntimeError("future failed")
        cfg = cast("PpoControlConfig", kwargs["config"])
        snapshot = ppo_control._snapshot_metadata(cfg, cast("str", kwargs["config_digest"]), 0, 0, 0, cfg.seed)
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload()},
            batch=_valid_batch(snapshot_metadata=snapshot.to_payload()),
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=0.0,
        )

    def rollout_from_snapshot(**kwargs: object) -> ppo_control._RolloutResult:
        snapshot = cast("PpoSnapshotMetadata", kwargs["expected_snapshot"])
        if snapshot.global_step == 1:
            raise RuntimeError("future failed")
        return collect(
            config=kwargs["config"],
            config_digest=kwargs["config_digest"],
            global_step=snapshot.global_step,
            samples_seen=snapshot.samples_seen,
            completed_games=snapshot.completed_games,
        )

    class _FakeFuture:
        def __init__(self, result: ppo_control._RolloutResult | BaseException) -> None:
            self._result = result

        def result(self) -> ppo_control._RolloutResult:
            if isinstance(self._result, BaseException):
                raise self._result
            return self._result

        def cancel(self) -> bool:
            return True

    class _FakeExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(self, fn: Callable[..., ppo_control._RolloutResult], **kwargs: object) -> _FakeFuture:
            try:
                return _FakeFuture(fn(**kwargs))
            except BaseException as exc:
                return _FakeFuture(exc)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            assert cancel_futures is True

    def train(**_kwargs: object) -> _FakeResult:
        nonlocal train_calls
        train_calls += 1
        return _FakeResult()

    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: object())
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch", collect)
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch_from_snapshot_artifact", rollout_from_snapshot)
    monkeypatch.setattr(ppo_control, "ppo_train_step", train)
    monkeypatch.setattr(ppo_control, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        ppo_control,
        "save_ppo_policy_snapshot_artifact",
        lambda _run_dir, **_kwargs: (tmp_path / "snapshot.pt", 7),
    )
    monkeypatch.setattr(
        ppo_control, "_save_t1_checkpoint", lambda path, *_args, **_kwargs: path.write_bytes(b"checkpoint")
    )

    with pytest.raises(RuntimeError, match="future failed"):
        ppo_control.run_ppo_control(_ppo_control_config(tmp_path, init_checkpoint, steps=2, pipeline_depth=1))

    assert train_calls == 1


def test_ppo_pipeline_in_flight_rollout_discarded_on_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    cancelled = False

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0}]

    class _FakeInit:
        global_step = 0
        samples_seen = 0

    class _FakeResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"loss_total": 0.0}
            self.entropy_controller: None = None

    class _FakeFuture:
        def result(self) -> ppo_control._RolloutResult:
            raise AssertionError("discarded future must not be consumed")

        def cancel(self) -> bool:
            nonlocal cancelled
            cancelled = True
            return True

    class _FakeExecutor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(self, *_args: object, **_kwargs: object) -> _FakeFuture:
            return _FakeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is False
            assert cancel_futures is True

    def collect(**kwargs: object) -> ppo_control._RolloutResult:
        cfg = cast("PpoControlConfig", kwargs["config"])
        snapshot = ppo_control._snapshot_metadata(cfg, cast("str", kwargs["config_digest"]), 0, 0, 0, cfg.seed)
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload()},
            batch=_valid_batch(snapshot_metadata=snapshot.to_payload()),
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=0.0,
        )

    def save_checkpoint(_path: Path, *_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(ppo_control, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(ppo_control, "_model", lambda _config: _FakeModel())
    monkeypatch.setattr(ppo_control, "build_optimizer", lambda _model, _optimizer_config: _FakeOptimizer())
    monkeypatch.setattr(ppo_control, "load_checkpoint_init_only", lambda *_args, **_kwargs: _FakeInit())
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: object())
    monkeypatch.setattr(ppo_control, "_collect_rollout_batch", collect)
    monkeypatch.setattr(
        ppo_control,
        "_collect_rollout_batch_from_snapshot_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(ppo_control, "ppo_train_step", lambda **_kwargs: _FakeResult())
    monkeypatch.setattr(
        ppo_control,
        "save_ppo_policy_snapshot_artifact",
        lambda _run_dir, **_kwargs: (tmp_path / "snapshot.pt", 7),
    )
    monkeypatch.setattr(ppo_control, "_save_t1_checkpoint", save_checkpoint)

    with pytest.raises(KeyboardInterrupt):
        ppo_control.run_ppo_control(_ppo_control_config(tmp_path, init_checkpoint, steps=2, pipeline_depth=1))

    assert cancelled


def test_ppo_control_serial_default_depth_zero(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = ppo_control.parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "mish_se",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )

    config = ppo_control.validate_args(args)

    assert config.ppo_pipeline_depth == 0


def test_ppo_control_rollout_device_defaults_to_train_device(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = ppo_control.parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "mish_se",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )

    config = ppo_control.validate_args(args)

    assert config.rollout_device is None
    assert ppo_control._effective_rollout_device(config) == "cpu"


def test_ppo_control_accepts_explicit_cpu_rollout_device(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = ppo_control.parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--ppo-rollout-device",
            "cpu",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "mish_se",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )

    config = ppo_control.validate_args(args)

    assert config.rollout_device == "cpu"


def test_ppo_control_invalid_rollout_device_rejected(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    args = ppo_control.parse_args(
        [
            "--init-checkpoint",
            str(init_checkpoint),
            "--out",
            str(tmp_path / "run"),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--ppo-rollout-device",
            "mps",
            "--hidden",
            "8",
            "--blocks",
            "1",
            "--bottleneck",
            "4",
            "--residual-profile",
            "mish_se",
            "--backbone-profile",
            "conv2d_local3",
            "--conv-memory-format",
            "contiguous",
        ]
    )

    with pytest.raises(ValueError, match="--ppo-rollout-device"):
        ppo_control.validate_args(args)


def test_ppo_pipeline_child_receives_requested_rollout_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = _ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=1)
    config = replace(config, rollout_device="cpu")
    seen: dict[str, str] = {}

    class _FakeArtifact:
        def __init__(self) -> None:
            self.model_config = _model_config()
            self.model_state: dict[str, torch.Tensor] = {}

    class _FakeModel(torch.nn.Module):
        @override
        def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ) -> torch.nn.modules.module._IncompatibleKeys:
            del state_dict, assign
            assert strict is True
            return torch.nn.modules.module._IncompatibleKeys([], [])

        @override
        def to(self, *args: Any, **kwargs: Any) -> _FakeModel:
            device = args[0] if args else kwargs.get("device")
            seen["model_device"] = str(device)
            return self

        @override
        def eval(self) -> _FakeModel:
            return self

    monkeypatch.setattr(ppo_control, "load_ppo_policy_snapshot_artifact", lambda *_args, **_kwargs: _FakeArtifact())
    monkeypatch.setattr(ppo_control, "HydraPolicyNet", lambda **_kwargs: _FakeModel())
    monkeypatch.setattr(ppo_control, "_load_extension", lambda _path: object())

    def collect(**kwargs: object) -> ppo_control._RolloutResult:
        cfg = cast("PpoControlConfig", kwargs["config"])
        seen["rollout_device"] = ppo_control._effective_rollout_device(cfg)
        snapshot = ppo_control._snapshot_metadata(
            cfg,
            cast("str", kwargs["config_digest"]),
            cast("int", kwargs["global_step"]),
            cast("int", kwargs["samples_seen"]),
            cast("int", kwargs["completed_games"]),
            cfg.seed + cast("int", kwargs["completed_games"]),
        )
        return ppo_control._RolloutResult(
            payload={"snapshot_metadata": snapshot.to_payload()},
            batch=_valid_batch(snapshot_metadata=snapshot.to_payload()),
            snapshot=snapshot,
            rollout_seed=snapshot.rollout_seed,
            onnx_export_ms=0.0,
            native_rollout_ms=0.0,
            batch_build_ms=0.0,
            outcome_metrics={},
            rollout_started=ppo_control.time.perf_counter(),
            future_rollout_ms=0.0,
        )

    monkeypatch.setattr(ppo_control, "_collect_rollout_batch", collect)

    snapshot = ppo_control._snapshot_metadata(config, "0" * 64, 1, 2, 3, 10)
    result = ppo_control._collect_rollout_batch_from_snapshot_artifact(
        config=config,
        config_digest="0" * 64,
        model_config=_model_config(),
        snapshot_path=tmp_path / "snapshot.pt",
        expected_snapshot=snapshot,
        export_dir=tmp_path / "exports",
    )

    assert result.snapshot == snapshot
    assert seen == {"model_device": "cpu", "rollout_device": "cpu"}


def test_ppo_control_invalid_pipeline_depth_rejected(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(SystemExit):
        ppo_control.parse_args(
            [
                "--init-checkpoint",
                str(init_checkpoint),
                "--out",
                str(tmp_path / "run"),
                "--steps",
                "1",
                "--hidden",
                "8",
                "--blocks",
                "1",
                "--bottleneck",
                "4",
                "--residual-profile",
                "mish_se",
                "--backbone-profile",
                "conv2d_local3",
                "--conv-memory-format",
                "contiguous",
                "--ppo-pipeline-depth",
                "2",
            ]
        )


def test_ppo_resume_compatible_digests_include_current_and_lr_decay_legacy(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = replace(_ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=1), lr_decay_samples=1000)
    legacy = _json_config(config)
    legacy["resume"] = None
    legacy["lr_decay_samples"] = None

    digests = _compatible_resume_config_digests(config)

    assert _config_digest(config) in digests
    assert ppo_control_config_digest_for_payload(legacy) in digests


def test_ppo_resume_compatible_digests_include_legacy_rollout_field_omission(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = _ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=0)
    legacy = _json_config(config)
    legacy["resume"] = None
    del legacy["ppo_pipeline_depth"]
    del legacy["rollout_device"]

    assert ppo_control_config_digest_for_payload(legacy) in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_do_not_omit_non_default_pipeline_depth(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = _ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=1)
    legacy = _json_config(config)
    legacy["resume"] = None
    del legacy["ppo_pipeline_depth"]
    del legacy["rollout_device"]

    assert ppo_control_config_digest_for_payload(legacy) not in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_do_not_omit_explicit_rollout_device(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = replace(_ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=0), rollout_device="cpu")
    legacy = _json_config(config)
    legacy["resume"] = None
    del legacy["ppo_pipeline_depth"]
    del legacy["rollout_device"]

    assert ppo_control_config_digest_for_payload(legacy) not in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_include_combined_legacy_omissions(tmp_path: Path) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    config = replace(_ppo_control_config(tmp_path, init_checkpoint, steps=1, pipeline_depth=0), lr_decay_samples=1000)
    legacy = _json_config(config)
    legacy["resume"] = None
    legacy["lr_decay_samples"] = None
    del legacy["ppo_pipeline_depth"]
    del legacy["rollout_device"]

    assert ppo_control_config_digest_for_payload(legacy) in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_include_checkpoint_fixture() -> None:
    config = _checkpoint_fixture_ppo_control_config()

    assert TINY_CHECKPOINT_CONFIG_DIGEST in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_allow_torch_callback_to_mahjax_migration() -> None:
    config = replace(_checkpoint_fixture_ppo_control_config(), rollout_inference="mahjax-gpu", ppo_pipeline_depth=1)

    assert TINY_CHECKPOINT_CONFIG_DIGEST in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_do_not_allow_mahjax_backend_migration_with_changed_games() -> None:
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        rollout_inference="mahjax-gpu",
        ppo_pipeline_depth=1,
        games_per_update=512,
    )

    assert TINY_CHECKPOINT_CONFIG_DIGEST not in _compatible_resume_config_digests(config)


TINY_RETENTION_COMPAT_CONFIG_DIGEST = TINY_CHECKPOINT_CONFIG_DIGEST


def test_ppo_resume_compatible_digests_allow_step_checkpoint_retention_change() -> None:
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        keep_step_checkpoints=True,
        rollout_inference="mahjax-gpu",
        ppo_pipeline_depth=1,
    )

    assert TINY_RETENTION_COMPAT_CONFIG_DIGEST in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_do_not_allow_checkpoint_cadence_change() -> None:
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        checkpoint_every_steps=1,
        rollout_inference="mahjax-gpu",
        ppo_pipeline_depth=1,
    )

    assert TINY_CHECKPOINT_CONFIG_DIGEST not in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_ignore_run_local_fields_for_fixture() -> None:
    output_dir, resume, tensorboard_dir = tiny_run_local_paths()
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        output_dir=output_dir,
        steps=1,
        resume=resume,
        tensorboard_dir=tensorboard_dir,
    )

    assert TINY_CHECKPOINT_CONFIG_DIGEST in _compatible_resume_config_digests(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("microbatch_size", 769),
        ("epochs", 4),
        ("hidden", 385),
        ("lr", 0.0002),
        ("games_per_update", 1025),
    ],
)
def test_ppo_resume_compatible_digests_keep_safety_fields_for_fixture(
    field: Literal["microbatch_size", "epochs", "hidden", "lr", "games_per_update"], value: float
) -> None:
    output_dir, resume, tensorboard_dir = tiny_run_local_paths()
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "steps": 1,
        "resume": resume,
        "tensorboard_dir": tensorboard_dir,
        field: value,
    }
    config = replace(_checkpoint_fixture_ppo_control_config(), **kwargs)

    assert TINY_CHECKPOINT_CONFIG_DIGEST not in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_run_local_omission_does_not_omit_non_default_pipeline_depth() -> None:
    output_dir, resume, tensorboard_dir = tiny_run_local_paths()
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        output_dir=output_dir,
        steps=1,
        resume=resume,
        tensorboard_dir=tensorboard_dir,
        ppo_pipeline_depth=1,
    )

    assert TINY_CHECKPOINT_CONFIG_DIGEST not in _compatible_resume_config_digests(config)


def test_ppo_resume_compatible_digests_run_local_omission_does_not_omit_explicit_rollout_device() -> None:
    output_dir, resume, tensorboard_dir = tiny_run_local_paths()
    config = replace(
        _checkpoint_fixture_ppo_control_config(),
        output_dir=output_dir,
        steps=1,
        resume=resume,
        tensorboard_dir=tensorboard_dir,
        rollout_device="cuda:0",
    )

    assert TINY_CHECKPOINT_CONFIG_DIGEST not in _compatible_resume_config_digests(config)


def _checkpoint_fixture_ppo_control_config() -> PpoControlConfig:
    return tiny_checkpoint_ppo_control_config()


def ppo_control_config_digest_for_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ppo_control_config(tmp_path: Path, init_checkpoint: Path, *, steps: int, pipeline_depth: int) -> PpoControlConfig:
    return PpoControlConfig(
        init_checkpoint=init_checkpoint,
        output_dir=tmp_path / "run",
        steps=steps,
        games_per_update=1,
        seed=7,
        device="cpu",
        temperature=1.0,
        arena_batch_decisions=1,
        arena_threads=0,
        extension_path=tmp_path / "libfake.so",
        hidden=8,
        blocks=1,
        bottleneck=4,
        residual_profile="tiny",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        lr=1.0e-3,
        min_lr=0.0,
        lr_warmup_samples=0,
        lr_decay_samples=None,
        grad_clip_norm=None,
        microbatch_size=1,
        epochs=1,
        target_kl=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        adamw_fused="off",
        adamw_foreach="off",
        bc_kl_reverse_coef=0.0,
        entropy_alpha=1.0e-3,
        entropy_beta=1.0e-2,
        entropy_alpha_max=0.05,
        log_every_steps=1,
        checkpoint_every_steps=1,
        keep_step_checkpoints=False,
        resume=None,
        tensorboard_dir=None,
        quiet=True,
        rollout_inference="rust-ort",
        ppo_pipeline_depth=pipeline_depth,
    )


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
