"""Run construction: model, optimizer, scheduler, digests, RNG anchors.

Owns binding the config to the live baseline model, the three-group
optimizer split, warmup/decay schedulers, the loop manifest digests
derived from authoritative sources (never constants), the RNG anchors
checkpoints carry, and the sidecar cursor projection plus prefix-hash
helpers the checkpoint writer and resume gates share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hydra2.artifacts.digest import (
    of_canonical as of_canonical,
)
from hydra2.artifacts.digest import (
    sha256_digest as sha256_digest,
)
from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.stream_manifest import manifest_digest as manifest_digest
from hydra2.runtime.checkpoint import capture_rng_state as capture_rng_state
from hydra2.training._rc_digest import run_config_digest as run_config_digest
from hydra2.training.stream_expand import _MODEL_PARAMETERS as _MODEL_PARAMETERS

try:
    from hydra2._native import contracts as _build_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _build_bridge = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from hydra2.data.stream_manifest import StreamManifest as StreamManifest
    from hydra2.data.stream_read import StreamCursor as DataStreamCursor
    from hydra2.training._rc_sections import RunConfig as RunConfig

__all__ = [
    "_NO_DECAY_NAME_TAGS",
    "_POLICY_HEAD_PREFIX",
    "_build_model",
    "_build_optimizer",
    "_build_scheduler",
    "_load_prefix_hashes",
    "_manifest_hashes_for_loop",
    "_optimizer_fused_kwargs",
    "_optimizer_param_groups",
    "_prefix_hashes_name",
    "_rng_anchors",
    "_scheduler_lr_factor",
    "_sidecar_cursor",
    "_verify_rng_anchors",
]


def _build_model(config: RunConfig) -> Any:
    """Real baseline model bound to the config (config-driven, fail closed)."""
    from hydra2.models.model import Hydra2BaselineModel
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if config.model.architecture_id != "hydra2_baseline_transformer_v1":
        raise ContractError(
            f"stream training supports architecture "
            f"'hydra2_baseline_transformer_v1', got {config.model.architecture_id!r}"
        )
    if config.model.action_count != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"model.action_count {config.model.action_count} != baseline {BASELINE_ACTION_COUNT}"
        )
    params = dict(config.model.parameters)
    unknown = sorted(k for k in params if k not in _MODEL_PARAMETERS)
    if len(unknown) > 0:
        raise ContractError(f"model.parameters unknown keys {unknown}")
    for key in ("d_model", "n_layers", "n_heads", "d_ff"):
        if key in params and (isinstance(params[key], bool) or not isinstance(params[key], int)):
            raise ContractError(f"model.parameters[{key!r}] must be an int")
    if "dropout" in params and (
        isinstance(params["dropout"], bool)
        or not isinstance(params["dropout"], (int, float))
        or not 0.0 <= float(params["dropout"]) < 1.0
    ):
        raise ContractError("model.parameters['dropout'] must lie in [0, 1)")
    return Hydra2BaselineModel(action_count=config.model.action_count, **params)


_POLICY_HEAD_PREFIX = "policy_head."

#: Parameter-name tags that force the no-decay group (case as stored;
#: model names are lowercase). ``"embed"`` covers ``*embedding`` tables;
#: the small ``*_emb`` tables (actor/phase/wind/furiten) carry no tag and
#: decay with the trunk — negligible at their size, kept literal per spec.
_NO_DECAY_NAME_TAGS: tuple[str, ...] = ("norm", "bias", "embed")


def _optimizer_param_groups(config: RunConfig, model: Any) -> list[dict[str, Any]] | None:
    """Three-group AdamW split: trunk-decay, trunk-no-decay, policy-head.

    * Trunk decay: all non-head parameters except the no-decay set below,
      ``lr=base`` and ``weight_decay=config.weight_decay``.
    * Trunk no-decay: non-head parameters with ``param.dim() < 2`` (params
      without ``.dim`` fall back to the decay group) or whose name contains
      ``norm``/``bias``/``embed``; ``lr=base`` and ``weight_decay=0.0``
      (biases, norms, and embeddings are not decayed).
    * Policy head: ``policy_head.`` parameters, ``lr=base*head_lr_mult``
      and ``weight_decay=0.0`` (dominant-head precedent).

    Policy-only head LR by design: the placement/value/event/belief heads
    stay in the trunk groups at the base LR, training alongside the trunk
    they read out from, while the dominant policy head takes the larger
    step.  Group order is decay, no-decay, head so the scheduler peak
    (first group) stays the base LR.  ``None`` only when the model exposes
    no named parameters (single-group uniform fallback in the builder).
    Empty buckets are omitted (stub models without a policy head train the
    trunk groups only).
    """
    base_lr = config.optimizer.lr
    weight_decay = config.optimizer.weight_decay
    head_mult = config.optimizer.head_lr_mult
    try:
        named: list[tuple[str, Any]] = list(model.named_parameters())  # pyrefly: ignore[unknown-argument-type] # model is Any; named_parameters owns the pairs
    except Exception:
        return None
    decay: list[Any] = []
    no_decay: list[Any] = []
    head: list[Any] = []
    for name, param in named:
        if name.startswith(_POLICY_HEAD_PREFIX):
            head.append(param)
            continue
        try:
            dims: int = param.dim()
            low_dim: bool = dims < 2
        except AttributeError:
            low_dim = False  # foreign stub params without .dim decay with the trunk
        if low_dim or any(tag in name for tag in _NO_DECAY_NAME_TAGS):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = []
    if len(decay) > 0:
        groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay})
    if len(no_decay) > 0:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})
    if len(head) > 0:
        groups.append({"params": head, "lr": base_lr * head_mult, "weight_decay": 0.0})
    if len(groups) == 0:
        return None
    return groups


def _optimizer_fused_kwargs() -> dict[str, Any]:
    """Foreach/fused selection (fused > foreach > for-loop ordering).

    CUDA uses the stable fused kernel; CPU uses foreach (fused CPU is beta).
    Both preserve numerics (parity-gated); the for-loop fallback is gone.
    """
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            return {"fused": True}
    except Exception:
        pass
    return {"foreach": True}


def _build_optimizer(config: RunConfig, model: Any) -> Any:
    """Registered optimizer id plus the config battery (no silent defaults)."""
    if config.optimizer.name == "adamw":
        groups = _optimizer_param_groups(config, model)
        kwargs = _optimizer_fused_kwargs()
        if groups is None:
            return torch.optim.AdamW(
                model.parameters(),  # pyrefly: ignore[unknown-argument-type] # model is Any; torch owns the params
                lr=config.optimizer.lr,
                betas=config.optimizer.betas,
                weight_decay=config.optimizer.weight_decay,
                **kwargs,
            )
        return torch.optim.AdamW(
            groups,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            **kwargs,
        )
    if config.optimizer.name == "adam":
        groups = _optimizer_param_groups(config, model)
        kwargs = _optimizer_fused_kwargs()
        if groups is None:
            return torch.optim.Adam(
                model.parameters(),  # pyrefly: ignore[unknown-argument-type] # model is Any; torch owns the params
                lr=config.optimizer.lr,
                betas=config.optimizer.betas,
                weight_decay=config.optimizer.weight_decay,
                **kwargs,
            )
        return torch.optim.Adam(
            groups,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            **kwargs,
        )
    if config.optimizer.name == "sgd":
        # v1 config carries no momentum knob: plain SGD (momentum 0).
        return torch.optim.SGD(
            model.parameters(),  # pyrefly: ignore[unknown-argument-type] # model is Any; torch owns the params
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ContractError("optimizer.id must be one of ['adamw', 'adam', 'sgd']")


def _build_scheduler(config: RunConfig, optimizer: Any) -> Any:
    """Warmup + registered decay, stepped once per global update by the loop."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

    name = config.scheduler.name
    warmup = config.scheduler.warmup_updates
    horizon = config.loop.max_updates
    parameters = dict(config.scheduler.parameters)
    final_factor = config.scheduler.final_factor
    warmup_start = config.scheduler.warmup_start_factor
    if name == "cosine":
        unknown = sorted(k for k in parameters if k != "T_max")
        if len(unknown) > 0:
            raise ContractError(f"scheduler.parameters unknown keys {unknown}")
        main_span = max(1, horizon - warmup) if warmup < horizon else horizon
        # Peak/final mapping: eta_min is final_factor fraction of peak.
        # Single-group exact; multi-group uses the first group's peak (all
        # r1 groups share the same final_factor ratio; absolute minima then
        # scale with their peaks, preserving the ratio per group only when
        # peaks are uniform — documented approximation, exact at 0.0).
        try:
            peak_group: dict[str, Any] = optimizer.param_groups[0]
            _peak: float = peak_group["lr"]
        except Exception:
            _peak = config.optimizer.lr
        main = CosineAnnealingLR(
            optimizer, T_max=int(parameters.get("T_max", main_span)), eta_min=_peak * final_factor
        )
    elif name == "constant":
        if len(parameters) > 0:
            raise ContractError(f"scheduler.parameters unknown keys {sorted(parameters)}")
        main = LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    elif name == "linear":
        unknown = sorted(k for k in parameters if k != "end_factor")
        if len(unknown) > 0:
            raise ContractError(f"scheduler.parameters unknown keys {unknown}")
        # Canonical final_factor wins; ``end_factor`` preserved when explicitly
        # set (back-compat for pre-factor-config runs).
        end_factor = float(parameters["end_factor"]) if "end_factor" in parameters else final_factor
        if not 0.0 <= end_factor <= 1.0:
            raise ContractError(f"scheduler end_factor must lie in [0, 1], got {end_factor}")
        span = max(1, horizon - warmup) if warmup < horizon else horizon
        main = LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=span)
    else:
        raise ContractError("scheduler.id must be one of ['cosine', 'constant', 'linear']")
    if warmup <= 0 or warmup >= horizon:
        if warmup <= 0:
            return main
        # Warmup covers the whole horizon: single warmup ramp, no decay tail.
        return LinearLR(optimizer, start_factor=warmup_start, total_iters=horizon)
    warm = LinearLR(optimizer, start_factor=warmup_start, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])


#: Bridge leaf for the scheduler closed form (single source:
#: ``hydra2._native.contracts`` ``scheduler_lr_factor``; the stale-.so
#: fallback is the byte-identical closed form below).
_scheduler_lr_factor_bridge = getattr(_build_bridge, "scheduler_lr_factor", None)


def _scheduler_lr_factor(
    *,
    scheduler: str,
    step: int,
    warmup_updates: int,
    max_updates: int,
    final_factor: float,
    warmup_start_factor: float,
    param_keys: Sequence[str],
    t_max: int | None = None,
    end_factor: float | None = None,
) -> float:
    """Closed-form LR multiplier for one step int (pure; no torch objects).

    Same warmup ramp + ``cosine`` / ``constant`` / ``linear`` decay as
    :func:`_build_scheduler`, projected to a float so planners can query the
    curve without constructing an optimizer. Bridge-first; the fallback is
    the identical formula (tolerance ``1e-12`` vs the bridge ``cos``).
    """
    if _scheduler_lr_factor_bridge is not None:
        try:
            factored: float = _scheduler_lr_factor_bridge(
                scheduler,
                step,
                warmup_updates,
                max_updates,
                final_factor,
                warmup_start_factor,
                list(param_keys),
                t_max,
                end_factor,
            )
            return factored
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    import math

    if scheduler == "cosine":
        unknown = sorted(k for k in param_keys if k != "T_max")
    elif scheduler == "linear":
        unknown = sorted(k for k in param_keys if k != "end_factor")
    elif scheduler == "constant":
        unknown = sorted(param_keys)
    else:
        raise ContractError("scheduler.id must be one of ['cosine', 'constant', 'linear']")
    if len(unknown) > 0:
        raise ContractError(f"scheduler.parameters unknown keys {unknown}")
    end: float = end_factor if end_factor is not None else final_factor
    if scheduler == "linear" and not 0.0 <= end <= 1.0:
        raise ContractError(f"scheduler end_factor must lie in [0, 1], got {end}")
    span = max(1, max_updates - warmup_updates) if warmup_updates < max_updates else max_updates
    if warmup_updates > 0 and warmup_updates >= max_updates:
        frac = min(max(step, 0), max_updates) / max_updates
        return warmup_start_factor + (1.0 - warmup_start_factor) * frac
    if warmup_updates > 0 and step < warmup_updates:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_updates)
    after = step - warmup_updates if warmup_updates > 0 else step
    if scheduler == "cosine":
        resolved = t_max if t_max is not None else span
        clamped = min(max(after, 0), resolved)
        if clamped >= resolved:
            return final_factor
        return final_factor + (1.0 - final_factor) * 0.5 * (
            1.0 + math.cos(math.pi * clamped / resolved)
        )
    if scheduler == "linear":
        return 1.0 + (end - 1.0) * (min(max(after, 0), span) / span)
    return 1.0


def _manifest_hashes_for_loop(
    config: RunConfig, *, model: Any, manifest: StreamManifest
) -> dict[str, str]:
    """Derive all ten loop manifest digests from authoritative sources.

    Never constant hashes: run/manifest digests come from the resolved config
    and stream manifest, model/utility digests from the live model, rules and
    schema digests from the pinned repo artifacts, the environment digest from
    the engine identity, and optimizer/scheduler digests from the pinned
    config sections. Derivation failure raises loudly.
    """
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import load_action_table
    from hydra2.contracts.observation_schema import observation_schema_digest
    from hydra2.engines.riichienv.identity import ENGINE_IDENTITY

    repo = repo_root()
    try:
        rules_bytes = (repo / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot derive rules_hash for training: {exc}") from exc
    try:
        action_table_path = repo / "configs" / "contracts" / "action_table_v1.json"
        action_digest = str(load_action_table(action_table_path).digest)
    except (ContractError, OSError, ValueError) as exc:
        raise ContractError(f"cannot derive action_schema_hash for training: {exc}") from exc
    try:
        observation_digest = str(observation_schema_digest())
    except (ContractError, ValueError) as exc:
        raise ContractError(f"cannot derive observation_schema_hash for training: {exc}") from exc
    optimizer_doc = {
        "head_lr_mult": config.optimizer.head_lr_mult,
        "id": config.optimizer.name,
        "lr": config.optimizer.lr,
        "betas": list(config.optimizer.betas),
        "weight_decay": config.optimizer.weight_decay,
    }
    scheduler_doc = {
        "id": config.scheduler.name,
        "warmup_updates": config.scheduler.warmup_updates,
        "parameters": dict(config.scheduler.parameters),
    }
    stream_digest = manifest_digest(manifest)
    optimizer_hex: str = of_canonical(optimizer_doc)
    scheduler_hex: str = of_canonical(scheduler_doc)
    rules_hex: str = sha256_digest(rules_bytes)
    return {
        "run_spec_hash": run_config_digest(config),
        "model_spec_hash": model.model_identity,  # pyrefly: ignore[unknown-argument-type] # live model owns the identity digest
        "optimizer_spec_hash": optimizer_hex,
        "scheduler_spec_hash": scheduler_hex,
        "environment_hash": str(ENGINE_IDENTITY.environment_hash),
        "rules_hash": rules_hex,
        "utility_manifest_hash": model.utility_manifest_hash,  # pyrefly: ignore[unknown-argument-type] # live model owns the manifest digest
        "action_schema_hash": action_digest,
        "observation_schema_hash": observation_digest,
        "dataset_manifest_hash": stream_digest,
    }


def _rng_anchors() -> dict[str, Any]:
    """JSON-safe anchors over the live torch RNG states (sidecar bundle)."""
    state: dict[str, Any] = capture_rng_state()
    cpu_state: bytes = bytes(state["cpu"])
    anchors: dict[str, Any] = {"torch_cpu_sha256": sha256_digest(cpu_state)}
    if "cuda" in state and state["cuda"] is not None:
        cuda_states: Any = state["cuda"]
        if isinstance(cuda_states, list):
            anchors["torch_cuda_sha256"] = [
                sha256_digest(device_bytes)
                for s in cuda_states
                for device_bytes in [bytes(s)]  # pyrefly: ignore[unknown-argument-type] # torch CUDA state dynamic; sha256 owns the bytes gate
            ]
        else:
            anchors["torch_cuda_sha256"] = sha256_digest(bytes(cuda_states))
    else:
        anchors["torch_cuda_sha256"] = None
    return anchors


def _verify_rng_anchors(payload_rng: Any, anchors: Any, *, ckpt: Path) -> None:
    """Recompute RNG anchors from the loaded payload; raise on any mismatch."""
    if not isinstance(payload_rng, dict) or "cpu" not in payload_rng:
        raise ContractError(f"checkpoint payload rng_state malformed: {ckpt}")
    if not isinstance(anchors, dict):
        raise ContractError(f"checkpoint sidecar rng_state must be a mapping: {ckpt}")
    expected_cpu = anchors.get("torch_cpu_sha256")
    raw_cpu: Any = payload_rng["cpu"]
    payload_cpu: bytes = bytes(raw_cpu)  # pyrefly: ignore[unknown-argument-type] # payload RNG dynamic; sha256 owns the bytes gate
    actual_cpu: str = sha256_digest(payload_cpu)
    if expected_cpu != actual_cpu:
        raise ContractError(f"checkpoint RNG state mismatch (torch cpu): {ckpt}")
    payload_cuda = payload_rng.get("cuda")
    expected_cuda = anchors.get("torch_cuda_sha256")
    if payload_cuda is None and expected_cuda is None:
        return
    if payload_cuda is None or expected_cuda is None:
        raise ContractError(f"checkpoint RNG state mismatch (torch cuda): {ckpt}")
    cuda_states: list[bytes] = payload_cuda
    actual_list: list[str] | str = (
        [sha256_digest(s) for s in cuda_states]  # pyrefly: ignore[unknown-argument-type] # payload CUDA states dynamic; sha256 owns the bytes gate
        if isinstance(payload_cuda, list)
        else sha256_digest(payload_cuda)  # pyrefly: ignore[unknown-argument-type] # payload CUDA state dynamic; sha256 owns the bytes gate
    )
    if actual_list != expected_cuda:
        raise ContractError(f"checkpoint RNG state mismatch (torch cuda): {ckpt}")


def _sidecar_cursor(data_cursor: DataStreamCursor) -> dict[str, int]:
    """Project the stream cursor onto the 5-field resume envelope."""
    return {
        "file_index": data_cursor.file_index,
        "byte_offset": data_cursor.byte_offset,
        "games_seen": data_cursor.games_seen,
        "seed": data_cursor.seed,
        "epoch": data_cursor.epoch,
    }


#: Bridge leaf for the prefix-hash filename (single source:
#: ``hydra2._native.contracts`` ``prefix_hashes_name``).
_prefix_hashes_name_bridge = getattr(_build_bridge, "prefix_hashes_name", None)


def _prefix_hashes_name(update: int) -> str:
    """Sidecar-adjacent prefix-hash filename for ``ckpt-<update>``."""
    if _prefix_hashes_name_bridge is not None:
        try:
            bound: str = _prefix_hashes_name_bridge(update)
            return bound
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    return f"prefix-hashes-{update:06d}.txt"


def _load_prefix_hashes(
    checkpoint_dir: Path, record: Mapping[str, Any], *, ckpt: Path
) -> list[str]:
    """Load + verify the sidecar-adjacent prefix-hash file (tamper → raise)."""
    from pathlib import Path

    name = record.get("file")
    count = record.get("count")
    digest = record.get("sha256")
    if not isinstance(name, str) or name == "" or "/" in name or name.startswith("."):
        raise ContractError(f"checkpoint prefix_hashes file invalid: {ckpt}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ContractError(f"checkpoint prefix_hashes count invalid: {ckpt}")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ContractError(f"checkpoint prefix_hashes digest invalid: {ckpt}")
    try:
        blob = (Path(checkpoint_dir) / name).read_bytes()
    except OSError as exc:
        raise ContractError(f"checkpoint prefix_hashes unreadable: {ckpt} ({exc})") from exc
    if str(sha256_digest(blob)) != digest:
        raise ContractError(f"checkpoint prefix_hashes digest mismatch: {ckpt}")
    lines = blob.decode("utf-8").splitlines()
    if len(lines) != count:
        raise ContractError(f"checkpoint prefix_hashes count mismatch: {ckpt}")
    for sha in lines:
        if not sha.startswith("sha256:") or sha == "sha256:":
            raise ContractError(f"checkpoint prefix_hashes entry invalid: {ckpt}")
    if lines != sorted(lines):
        raise ContractError(f"checkpoint prefix_hashes order invalid: {ckpt}")
    return lines
