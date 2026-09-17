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
from hydra2.data.stream import manifest_digest as manifest_digest
from hydra2.runtime.checkpoint import capture_rng_state as capture_rng_state
from hydra2.training.run_config import run_config_digest as run_config_digest
from hydra2.training.stream_expand import _MODEL_PARAMETERS as _MODEL_PARAMETERS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from hydra2.data.stream import StreamCursor as DataStreamCursor
    from hydra2.data.stream import StreamManifest as StreamManifest
    from hydra2.training.run_config import RunConfig as RunConfig

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
        named = list(model.named_parameters())
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
            low_dim = param.dim() < 2
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
                model.parameters(),
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
                model.parameters(),
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
            model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
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
            _peak = float(optimizer.param_groups[0]["lr"])
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
    return {
        "run_spec_hash": run_config_digest(config),
        "model_spec_hash": str(model.model_identity),
        "optimizer_spec_hash": str(of_canonical(optimizer_doc)),
        "scheduler_spec_hash": str(of_canonical(scheduler_doc)),
        "environment_hash": str(ENGINE_IDENTITY.environment_hash),
        "rules_hash": str(sha256_digest(rules_bytes)),
        "utility_manifest_hash": str(model.utility_manifest_hash),
        "action_schema_hash": action_digest,
        "observation_schema_hash": observation_digest,
        "dataset_manifest_hash": stream_digest,
    }


def _rng_anchors() -> dict[str, Any]:
    """JSON-safe anchors over the live torch RNG states (sidecar bundle)."""
    state = capture_rng_state()
    anchors: dict[str, Any] = {"torch_cpu_sha256": str(sha256_digest(bytes(state["cpu"])))}
    if "cuda" in state and state["cuda"] is not None:
        cuda = state["cuda"]
        if isinstance(cuda, list):
            anchors["torch_cuda_sha256"] = [
                str(sha256_digest(bytes(device_state))) for device_state in cuda
            ]
        else:
            anchors["torch_cuda_sha256"] = str(sha256_digest(bytes(cuda)))
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
    actual_cpu = str(sha256_digest(bytes(payload_rng["cpu"])))
    if expected_cpu != actual_cpu:
        raise ContractError(f"checkpoint RNG state mismatch (torch cpu): {ckpt}")
    payload_cuda = payload_rng.get("cuda")
    expected_cuda = anchors.get("torch_cuda_sha256")
    if payload_cuda is None and expected_cuda is None:
        return
    if payload_cuda is None or expected_cuda is None:
        raise ContractError(f"checkpoint RNG state mismatch (torch cuda): {ckpt}")
    actual_list = (
        [str(sha256_digest(bytes(s))) for s in payload_cuda]
        if isinstance(payload_cuda, list)
        else str(sha256_digest(bytes(payload_cuda)))
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


def _prefix_hashes_name(update: int) -> str:
    """Sidecar-adjacent prefix-hash filename for ``ckpt-<update>``."""
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
