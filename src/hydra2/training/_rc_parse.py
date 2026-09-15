"""Run-config core section parsers: run/data/model/weights/optimizer/scheduler/runtime/loop.

Owns the strict per-section constructors for the eight training-plane
sections. The six observer-side sections live in
:mod:`hydra2.training._rc_parse_aux`; root assembly lives in
:mod:`hydra2.training._rc_root`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_require import _digest_pin_or_none as _digest_pin_or_none
from hydra2.training._rc_require import _positive_float_map as _positive_float_map
from hydra2.training._rc_require import _reject_unknown as _reject_unknown
from hydra2.training._rc_require import _require_bounded_float as _require_bounded_float
from hydra2.training._rc_require import _require_bounded_int as _require_bounded_int
from hydra2.training._rc_require import _require_loop_bool as _require_loop_bool
from hydra2.training._rc_require import _require_nonempty_str as _require_nonempty_str
from hydra2.training._rc_require import _require_nonnegative_float as _require_nonnegative_float
from hydra2.training._rc_require import _require_nonnegative_int as _require_nonnegative_int
from hydra2.training._rc_require import _require_positive_int as _require_positive_int
from hydra2.training._rc_require import _weight_map as _weight_map
from hydra2.training._rc_sections import _ADAPTER_IDS as _ADAPTER_IDS
from hydra2.training._rc_sections import _COMPILE_MODES as _COMPILE_MODES
from hydra2.training._rc_sections import _CUDA_DEVICE_RE as _CUDA_DEVICE_RE
from hydra2.training._rc_sections import _OPTIMIZER_IDS as _OPTIMIZER_IDS
from hydra2.training._rc_sections import _RUN_ID_RE as _RUN_ID_RE
from hydra2.training._rc_sections import _RUNTIME_PRECISIONS as _RUNTIME_PRECISIONS
from hydra2.training._rc_sections import _SCHEDULER_IDS as _SCHEDULER_IDS
from hydra2.training._rc_sections import _TENHOU_SCOPE as _TENHOU_SCOPE
from hydra2.training._rc_sections import RUN_KINDS as RUN_KINDS
from hydra2.training._rc_sections import DataConfig as DataConfig
from hydra2.training._rc_sections import LoopConfig as LoopConfig
from hydra2.training._rc_sections import ModelConfig as ModelConfig
from hydra2.training._rc_sections import OptimizerConfig as OptimizerConfig
from hydra2.training._rc_sections import RunMeta as RunMeta
from hydra2.training._rc_sections import RuntimeConfig as RuntimeConfig
from hydra2.training._rc_sections import SchedulerConfig as SchedulerConfig
from hydra2.training._rc_sections import WeightsConfig as WeightsConfig

__all__ = [
    "_parse_data",
    "_parse_loop",
    "_parse_model",
    "_parse_optimizer",
    "_parse_run",
    "_parse_runtime",
    "_parse_scheduler",
    "_parse_weights",
]

# ---------------------------------------------------------------------------
# Section parsers (each rejects unknown keys; cross-checks at the end)
# ---------------------------------------------------------------------------


def _parse_run(raw: Any) -> RunMeta:
    fields = _reject_unknown(raw, ("id", "kind", "description"), where="run")
    run_id = fields.get("id", "")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError(f"run.id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {run_id!r}")
    kind = fields.get("kind", "supervised")
    if kind not in RUN_KINDS:
        raise ContractError(f"run.kind must be one of {list(RUN_KINDS)}, got {kind!r}")
    description = fields.get("description", "")
    if not isinstance(description, str):
        raise ContractError("run.description must be a string")
    return RunMeta(run_id=run_id, kind=kind, description=description)


def _parse_data(raw: Any) -> DataConfig:
    fields = _reject_unknown(
        raw,
        (
            "root",
            "scope",
            "train_split",
            "val_split",
            "wall_disjoint",
            "leakage_check",
            "dora_width",
            "actor_privileged_split",
            "dataset_manifest_hash",
            "shuffle_buffer_size",
            "drop_last",
            "num_workers",
            "world_size",
            "replay_backend",
            "decode_prefetch",
            "expand_batch_games",
            "homogeneous_buckets",
        ),
        where="data",
    )
    root = _require_nonempty_str(fields, "root", where="data")
    if not Path(root).is_absolute():
        raise ContractError(f"data.root must be an absolute path after interpolation, got {root!r}")
    scope = fields.get("scope", _TENHOU_SCOPE)
    if scope != _TENHOU_SCOPE:
        raise ContractError(
            f"data.scope is Tenhou-only in v1 (must be {_TENHOU_SCOPE!r}), got {scope!r}"
        )
    train_split = fields.get("train_split", "train")
    val_split = fields.get("val_split", "validation")
    if not isinstance(train_split, str) or train_split == "":
        raise ContractError("data.train_split must be a non-empty string")
    if not isinstance(val_split, str) or val_split == "":
        raise ContractError("data.val_split must be a non-empty string")
    for flag in ("wall_disjoint", "leakage_check", "actor_privileged_split"):
        if fields.get(flag, True) is not True:
            raise ContractError(
                f"data.{flag} must stay true (ephemeral rows carry the full "
                "validate/quarantine/split/leakage contract); refusing to weaken it"
            )
    dora_width = fields.get("dora_width", 5)
    if dora_width != 5:
        raise ContractError(f"data.dora_width must be 5 (frozen (5,) dora), got {dora_width!r}")
    drop_last = fields.get("drop_last", True)
    if not isinstance(drop_last, bool):
        raise ContractError(f"data.drop_last must be a bool, got {drop_last!r}")
    homogeneous_buckets = fields.get("homogeneous_buckets", False)
    if not isinstance(homogeneous_buckets, bool):
        raise ContractError(f"data.homogeneous_buckets must be a bool, got {homogeneous_buckets!r}")
    replay_backend = fields.get("replay_backend", "rust")
    if replay_backend not in ("python", "rust"):
        raise ContractError(
            f"data.replay_backend must be 'python' or 'rust', got {replay_backend!r}"
        )
    return DataConfig(
        root=root,
        scope=str(scope),
        train_split=train_split,
        val_split=val_split,
        wall_disjoint=True,
        leakage_check=True,
        dora_width=5,
        actor_privileged_split=True,
        dataset_manifest_hash=_digest_pin_or_none(fields, "dataset_manifest_hash", where="data"),
        shuffle_buffer_size=_require_nonnegative_int(fields, "shuffle_buffer_size", where="data")
        if "shuffle_buffer_size" in fields
        else 10000,
        drop_last=drop_last,
        num_workers=_require_nonnegative_int(fields, "num_workers", where="data")
        if "num_workers" in fields
        else 0,
        world_size=_require_positive_int(fields, "world_size", where="data")
        if "world_size" in fields
        else 1,
        replay_backend=str(replay_backend),
        decode_prefetch=_require_bounded_int(fields, "decode_prefetch", where="data", lo=1, hi=1024)
        if "decode_prefetch" in fields
        else 64,
        expand_batch_games=_require_bounded_int(
            fields, "expand_batch_games", where="data", lo=1, hi=1024
        )
        if "expand_batch_games" in fields
        else 64,
        homogeneous_buckets=homogeneous_buckets,
    )


def _parse_model(raw: Any) -> ModelConfig:
    fields = _reject_unknown(raw, ("architecture_id", "action_count", "parameters"), where="model")
    architecture_id = fields.get("architecture_id", "hydra2_baseline_transformer_v1")
    if not isinstance(architecture_id, str) or architecture_id.strip() == "":
        raise ContractError("model.architecture_id must be a non-empty string")
    try:
        from hydra2.models.schema import KNOWN_ARCHITECTURES

        if architecture_id not in KNOWN_ARCHITECTURES:
            raise ContractError(f"model.architecture_id unknown: {architecture_id!r}")
    except ImportError:
        pass
    action_count = fields.get("action_count", 6792)
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count <= 0:
        raise ContractError(f"model.action_count must be a positive int, got {action_count!r}")
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if action_count != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"model.action_count {action_count} != baseline {BASELINE_ACTION_COUNT}"
        )
    parameters = fields.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ContractError("model.parameters must be a mapping")
    return ModelConfig(
        architecture_id=architecture_id,
        action_count=action_count,
        parameters=dict(parameters),
    )


def _parse_weights(raw: Any) -> WeightsConfig:
    fields = _reject_unknown(
        raw,
        (
            "w_policy",
            "w_placement",
            "w_value",
            "w_event",
            "w_belief",
            "privileged_source_hash",
            "label_smoothing",
        ),
        where="weights",
    )
    return WeightsConfig(
        w_policy=_require_nonnegative_float(fields, "w_policy", where="weights")
        if "w_policy" in fields
        else 1.0,
        w_placement=_require_nonnegative_float(fields, "w_placement", where="weights")
        if "w_placement" in fields
        else 0.0,
        w_value=_require_nonnegative_float(fields, "w_value", where="weights")
        if "w_value" in fields
        else 0.0,
        w_event=_weight_map(fields, "w_event", where="weights") if "w_event" in fields else None,
        w_belief=_weight_map(fields, "w_belief", where="weights") if "w_belief" in fields else None,
        privileged_source_hash=_digest_pin_or_none(
            fields, "privileged_source_hash", where="weights"
        ),
        label_smoothing=_require_bounded_float(
            fields, "label_smoothing", where="weights", lo=0.0, hi=1.0, hi_open=True
        )
        if "label_smoothing" in fields and fields.get("label_smoothing") is not None
        else 0.03,
    )


def _parse_optimizer(raw: Any) -> OptimizerConfig:
    fields = _reject_unknown(
        raw, ("id", "lr", "betas", "weight_decay", "head_lr_mult"), where="optimizer"
    )
    name = fields.get("id", "adamw")
    if name not in _OPTIMIZER_IDS:
        raise ContractError(f"optimizer.id must be one of {list(_OPTIMIZER_IDS)}, got {name!r}")
    lr = fields.get("lr", 3e-4)
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not (float(lr) == float(lr))
        or float(lr) in (float("inf"), float("-inf"))
        or float(lr) <= 0.0
    ):
        raise ContractError(f"optimizer.lr must be positive and finite, got {lr!r}")
    betas_raw = fields.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ContractError(f"optimizer.betas must be a 2-element list, got {betas_raw!r}")
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    for beta in betas:
        if not 0.0 <= beta < 1.0 or beta != beta:
            raise ContractError(f"optimizer.betas entries must lie in [0, 1), got {betas!r}")
    decay = fields.get("weight_decay", 0.01)
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not (float(decay) == float(decay))
        or float(decay) in (float("inf"), float("-inf"))
        or float(decay) < 0.0
    ):
        raise ContractError(f"optimizer.weight_decay must be finite non-negative, got {decay!r}")
    head_mult = fields.get("head_lr_mult", 3.0)
    if (
        isinstance(head_mult, bool)
        or not isinstance(head_mult, (int, float))
        or not (float(head_mult) == float(head_mult))
        or float(head_mult) in (float("inf"), float("-inf"))
        or float(head_mult) <= 0.0
    ):
        raise ContractError(
            f"optimizer.head_lr_mult must be positive and finite, got {head_mult!r}"
        )
    return OptimizerConfig(
        name=name,
        lr=float(lr),
        betas=betas,
        weight_decay=float(decay),
        head_lr_mult=float(head_mult),
    )


def _parse_scheduler(raw: Any) -> SchedulerConfig:
    fields = _reject_unknown(
        raw,
        ("id", "warmup_updates", "parameters", "final_factor", "warmup_start_factor"),
        where="scheduler",
    )
    name = fields.get("id", "cosine")
    if name not in _SCHEDULER_IDS:
        raise ContractError(f"scheduler.id must be one of {list(_SCHEDULER_IDS)}, got {name!r}")
    warmup = fields.get("warmup_updates", 100)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ContractError(f"scheduler.warmup_updates must be a non-negative int, got {warmup!r}")
    parameters = fields.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ContractError("scheduler.parameters must be a mapping")
    return SchedulerConfig(
        name=name,
        warmup_updates=warmup,
        parameters=dict(parameters),
        final_factor=_require_bounded_float(
            fields, "final_factor", where="scheduler", lo=0.0, hi=1.0
        )
        if "final_factor" in fields and fields.get("final_factor") is not None
        else 0.0,
        warmup_start_factor=_require_bounded_float(
            fields, "warmup_start_factor", where="scheduler", lo=0.0, hi=1.0, lo_open=True
        )
        if "warmup_start_factor" in fields and fields.get("warmup_start_factor") is not None
        else 0.01,
    )


def _parse_runtime(raw: Any) -> RuntimeConfig:
    fields = _reject_unknown(
        raw, ("adapter_id", "device", "precision", "compile_mode"), where="runtime"
    )
    adapter_id = fields.get("adapter_id", "plain_pytorch")
    if adapter_id not in _ADAPTER_IDS:
        raise ContractError(
            f"runtime.adapter_id must be one of {list(_ADAPTER_IDS)}, got {adapter_id!r}"
        )
    device = fields.get("device", "cuda")
    if not isinstance(device, str) or (
        device != "cpu" and _CUDA_DEVICE_RE.fullmatch(device) is None
    ):
        raise ContractError(f"runtime.device must be 'cpu' or cuda[:N], got {device!r}")
    precision = fields.get("precision", "fp32")
    if precision not in _RUNTIME_PRECISIONS:
        raise ContractError(
            f"runtime.precision must be one of {list(_RUNTIME_PRECISIONS)} "
            f"(fp16 excluded by design), got {precision!r}"
        )
    compile_mode = fields.get("compile_mode", "eager")
    if compile_mode not in _COMPILE_MODES:
        raise ContractError(
            f"runtime.compile_mode must be one of {list(_COMPILE_MODES)}, got {compile_mode!r}"
        )
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
    fields = _reject_unknown(
        raw,
        (
            "microbatch_size",
            "accumulation_steps",
            "gradient_clip_norm",
            "max_updates",
            "checkpoint_frequency_updates",
            "precision",
            "keep_last_checkpoints",
            "stratified_sampling",
            "sampling_ratios",
            "log_per_type_metrics",
            "fit_temperature",
            "fetch_prefetch_depth",
        ),
        where="loop",
    )
    clip = fields.get("gradient_clip_norm", 1.0)
    if clip is not None and (
        isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not (float(clip) == float(clip))
        or float(clip) in (float("inf"), float("-inf"))
        or float(clip) <= 0.0
    ):
        raise ContractError(
            f"loop.gradient_clip_norm must be null or positive finite, got {clip!r}"
        )
    precision = fields.get("precision", "fp32")
    if precision not in ("fp32", "bf16_mixed"):
        raise ContractError(f"loop.precision must be 'fp32' or 'bf16_mixed', got {precision!r}")
    return LoopConfig(
        microbatch_size=_require_positive_int(fields, "microbatch_size", where="loop")
        if "microbatch_size" in fields
        else 4,
        accumulation_steps=_require_positive_int(fields, "accumulation_steps", where="loop")
        if "accumulation_steps" in fields
        else 8,
        gradient_clip_norm=None if clip is None else float(clip),
        max_updates=_require_positive_int(fields, "max_updates", where="loop")
        if "max_updates" in fields
        else 10000,
        checkpoint_frequency_updates=_require_positive_int(
            fields, "checkpoint_frequency_updates", where="loop"
        )
        if "checkpoint_frequency_updates" in fields
        else 500,
        precision=str(precision),
        keep_last_checkpoints=None
        if fields.get("keep_last_checkpoints") is None
        else _require_positive_int(fields, "keep_last_checkpoints", where="loop"),
        stratified_sampling=_require_loop_bool(fields, "stratified_sampling", default=False),
        sampling_ratios=_positive_float_map(fields, "sampling_ratios", where="loop"),
        log_per_type_metrics=_require_loop_bool(fields, "log_per_type_metrics", default=True),
        fit_temperature=_require_loop_bool(fields, "fit_temperature", default=True),
        fetch_prefetch_depth=_require_bounded_int(
            fields, "fetch_prefetch_depth", where="loop", lo=1, hi=16
        )
        if "fetch_prefetch_depth" in fields
        else 3,
    )
