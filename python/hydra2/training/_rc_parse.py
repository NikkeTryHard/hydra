"""Run-config core section parsers: run/data/model/weights/optimizer/scheduler/runtime/loop/rl.

Owns the strict per-section constructors for the eight training-plane
sections plus the closed RL profile. The six observer-side sections live in
:mod:`hydra2.training._rc_parse_aux`; root assembly lives in
:mod:`hydra2.training._rc_root`.
"""

from __future__ import annotations

from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_sections import DataConfig as DataConfig
from hydra2.training._rc_sections import LoopConfig as LoopConfig
from hydra2.training._rc_sections import ModelConfig as ModelConfig
from hydra2.training._rc_sections import OptimizerConfig as OptimizerConfig
from hydra2.training._rc_sections import RlConfig as RlConfig
from hydra2.training._rc_sections import RunMeta as RunMeta
from hydra2.training._rc_sections import RuntimeConfig as RuntimeConfig
from hydra2.training._rc_sections import SchedulerConfig as SchedulerConfig
from hydra2.training._rc_sections import WeightsConfig as WeightsConfig

try:
    from hydra2._native import contracts as _rc_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _rc_bridge = None  # type: ignore[assignment]


def _require_rc_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _rc_bridge is None or not hasattr(_rc_bridge, "rc_parse_run"):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _rc_bridge


__all__ = [
    "_parse_data",
    "_parse_loop",
    "_parse_model",
    "_parse_optimizer",
    "_parse_rl",
    "_parse_run",
    "_parse_runtime",
    "_parse_scheduler",
    "_parse_weights",
]

# ---------------------------------------------------------------------------
# Section parsers (each rejects unknown keys; cross-checks at the end)
# ---------------------------------------------------------------------------


def _parse_run(raw: Any) -> RunMeta:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_run`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_run(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    run_id: str = fields["run_id"]
    kind: str = fields["kind"]
    description: str = fields["description"]
    return RunMeta(run_id=run_id, kind=kind, description=description)


def _parse_data(raw: Any) -> DataConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_data`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_data(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    # `rc_parse_data` owns the roots shape (a list of [id, path] string
    # pairs); translate the trusted structure without revalidating it here.
    roots_raw: Any = fields["roots"]
    roots: tuple[tuple[str, str], ...] = tuple((pair[0], pair[1]) for pair in roots_raw)
    scope: str = fields["scope"]
    train_split: str = fields["train_split"]
    val_split: str = fields["val_split"]
    dataset_manifest_hash: str | None = fields["dataset_manifest_hash"]
    shuffle_buffer_size: int = fields["shuffle_buffer_size"]
    drop_last: bool = fields["drop_last"]
    num_workers: int = fields["num_workers"]
    world_size: int = fields["world_size"]
    replay_backend: str = fields["replay_backend"]
    decode_prefetch: int = fields["decode_prefetch"]
    expand_batch_games: int = fields["expand_batch_games"]
    homogeneous_buckets: bool = fields["homogeneous_buckets"]
    return DataConfig(
        roots=roots,
        scope=scope,
        train_split=train_split,
        val_split=val_split,
        wall_disjoint=True,
        leakage_check=True,
        dora_width=5,
        actor_privileged_split=True,
        dataset_manifest_hash=dataset_manifest_hash,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_last=drop_last,
        num_workers=num_workers,
        world_size=world_size,
        replay_backend=replay_backend,
        decode_prefetch=decode_prefetch,
        expand_batch_games=expand_batch_games,
        homogeneous_buckets=homogeneous_buckets,
    )


def _parse_model(raw: Any) -> ModelConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_model`` decides.

    The architecture registry and baseline-width equality stay Python
    (torch-owned ``models/schema.py`` patch-points).
    """
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_model(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    architecture_id: str = fields["architecture_id"]
    try:
        from hydra2.models.schema import KNOWN_ARCHITECTURES

        if architecture_id not in KNOWN_ARCHITECTURES:
            raise ContractError(f"model.architecture_id unknown: {architecture_id!r}")
    except ImportError:
        pass
    action_count: int = fields["action_count"]
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if action_count != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"model.action_count {action_count} != baseline {BASELINE_ACTION_COUNT}"
        )
    parameters: dict[str, Any] = fields["parameters"]
    return ModelConfig(
        architecture_id=architecture_id,
        action_count=action_count,
        parameters=dict(parameters),
    )


def _parse_weights(raw: Any) -> WeightsConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_weights`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_weights(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    w_policy: float = fields["w_policy"]
    w_placement: float = fields["w_placement"]
    w_value: float = fields["w_value"]
    w_event: dict[str, float] | None = fields["w_event"]
    w_belief: dict[str, float] | None = fields["w_belief"]
    privileged_source_hash: str | None = fields["privileged_source_hash"]
    label_smoothing: float = fields["label_smoothing"]
    return WeightsConfig(
        w_policy=w_policy,
        w_placement=w_placement,
        w_value=w_value,
        w_event=w_event,
        w_belief=w_belief,
        privileged_source_hash=privileged_source_hash,
        label_smoothing=label_smoothing,
    )


def _parse_optimizer(raw: Any) -> OptimizerConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_optimizer`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_optimizer(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    optimizer_id: str = fields["id"]
    lr: float = fields["lr"]
    betas_raw: list[float] = fields["betas"]
    betas: tuple[float, float] = (betas_raw[0], betas_raw[1])
    weight_decay: float = fields["weight_decay"]
    head_lr_mult: float = fields["head_lr_mult"]
    return OptimizerConfig(
        name=optimizer_id,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        head_lr_mult=head_lr_mult,
    )


def _parse_scheduler(raw: Any) -> SchedulerConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_scheduler`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_scheduler(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    scheduler_id: str = fields["id"]
    warmup_updates: int = fields["warmup_updates"]
    scheduler_params: dict[str, Any] = fields["parameters"]
    final_factor: float = fields["final_factor"]
    warmup_start_factor: float = fields["warmup_start_factor"]
    return SchedulerConfig(
        name=scheduler_id,
        warmup_updates=warmup_updates,
        parameters=dict(scheduler_params),
        final_factor=final_factor,
        warmup_start_factor=warmup_start_factor,
    )


def _parse_runtime(raw: Any) -> RuntimeConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_runtime`` decides.

    The ``validate_runtime_spec`` protocol check stays Python
    (import-guarded ``runtime/protocol.py`` patch-point).
    """
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_runtime(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    adapter_id: str = fields["adapter_id"]
    device: str = fields["device"]
    precision: str = fields["precision"]
    compile_mode: str = fields["compile_mode"]
    try:
        from hydra2.runtime.protocol import RuntimeSpec, validate_runtime_spec

        validate_runtime_spec(
            RuntimeSpec(
                adapter_id=adapter_id,  # type: ignore[arg-type]
                device=device,
                precision=precision,  # type: ignore[arg-type]
                compile_mode=compile_mode,  # type: ignore[arg-type]
            )
        )
    except ImportError:
        pass
    return RuntimeConfig(
        adapter_id=adapter_id,
        device=device,
        precision=precision,
        compile_mode=compile_mode,
    )


def _parse_loop(raw: Any) -> LoopConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_loop`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_loop(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    microbatch_size: int = fields["microbatch_size"]
    accumulation_steps: int = fields["accumulation_steps"]
    gradient_clip_norm: float | None = fields["gradient_clip_norm"]
    max_updates: int = fields["max_updates"]
    checkpoint_frequency_updates: int = fields["checkpoint_frequency_updates"]
    loop_precision: str = fields["precision"]
    keep_last_checkpoints: int | None = fields["keep_last_checkpoints"]
    stratified_sampling: bool = fields["stratified_sampling"]
    sampling_ratios: dict[str, float] | None = fields["sampling_ratios"]
    log_per_type_metrics: bool = fields["log_per_type_metrics"]
    fit_temperature: bool = fields["fit_temperature"]
    fetch_prefetch_depth: int = fields["fetch_prefetch_depth"]
    pack_histories: bool = fields["pack_histories"]
    return LoopConfig(
        microbatch_size=microbatch_size,
        accumulation_steps=accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        max_updates=max_updates,
        checkpoint_frequency_updates=checkpoint_frequency_updates,
        precision=loop_precision,
        keep_last_checkpoints=keep_last_checkpoints,
        stratified_sampling=stratified_sampling,
        sampling_ratios=sampling_ratios,
        log_per_type_metrics=log_per_type_metrics,
        fit_temperature=fit_temperature,
        fetch_prefetch_depth=fetch_prefetch_depth,
        pack_histories=pack_histories,
    )


def _parse_rl(raw: Any) -> RlConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_rl`` decides."""
    try:
        fields: dict[str, Any] = _require_rc_bridge().rc_parse_rl(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return RlConfig(
        schema_version=fields["schema_version"],
        initial_checkpoint_id=fields["initial_checkpoint_id"],
        initial_model_spec_hash=fields["initial_model_spec_hash"],
        master_seed_hex=fields["master_seed_hex"],
        experiment_id=fields["experiment_id"],
        split_scope=fields["split_scope"],
        replicate_id=fields["replicate_id"],
        attempt_id=fields["attempt_id"],
        generations=fields["generations"],
        games_per_generation=fields["games_per_generation"],
        native_workers=fields["native_workers"],
        env_slots=fields["env_slots"],
        max_lookahead=fields["max_lookahead"],
        inference_cap=fields["inference_cap"],
        microbatch_cap=fields["microbatch_cap"],
        accumulation_cap=fields["accumulation_cap"],
        optimizer_minibatch_cap=fields["optimizer_minibatch_cap"],
        epochs=fields["epochs"],
        row_cap=fields["row_cap"],
        decision_cap_per_game=fields["decision_cap_per_game"],
        max_updates_per_generation=fields["max_updates_per_generation"],
        compressed_byte_cap=fields["compressed_byte_cap"],
        min_free_disk_bytes=fields["min_free_disk_bytes"],
        host_byte_budget=fields["host_byte_budget"],
        gpu_cache_byte_budget=fields["gpu_cache_byte_budget"],
        clip_eps=fields["clip_eps"],
        eps_std=fields["eps_std"],
        w_value=fields["w_value"],
        w_bc=fields["w_bc"],
        ppo_alpha=fields["ppo_alpha"],
        gamma=fields["gamma"],
        lam=fields["lam"],
        optimizer_lr=fields["optimizer_lr"],
        optimizer_beta1=fields["optimizer_beta1"],
        optimizer_beta2=fields["optimizer_beta2"],
        optimizer_weight_decay=fields["optimizer_weight_decay"],
        head_lr_mult=fields["head_lr_mult"],
        gradient_clip_norm=fields["gradient_clip_norm"],
        warmup_updates=fields["warmup_updates"],
        rules_hash=fields["rules_hash"],
        utility_hash=fields["utility_hash"],
        action_table_hash=fields["action_table_hash"],
        event_schema_hash=fields["event_schema_hash"],
        observation_schema_hash=fields["observation_schema_hash"],
        packet_boundary_hash=fields["packet_boundary_hash"],
        input_schema_hash=fields["input_schema_hash"],
        wall_ledger_hash=fields["wall_ledger_hash"],
        opponent_catalog_hash=fields["opponent_catalog_hash"],
        wall_recipe_hash=fields["wall_recipe_hash"],
        sampling_recipe_hash=fields["sampling_recipe_hash"],
        return_recipe_hash=fields["return_recipe_hash"],
        shuffle_recipe_hash=fields["shuffle_recipe_hash"],
        core_revision=fields["core_revision"],
    )
