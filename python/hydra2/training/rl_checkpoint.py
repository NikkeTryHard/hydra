"""Torch-forced cold boundary for continuous RL.

Python owns ONLY required Torch glue: deterministic construction seeding,
two module shells built by the shared ``stream_build`` constructor, one
verified C0 weight load, a fresh optimizer/scheduler, and snapshots
through the existing six-section ordinary writer. Native code owns every
profile/hash/wall/order/RunSpec decision; this module never re-verifies
what native already enforces.

Single-call setup: :func:`prepare_torch_state` takes the validated
``RunConfig`` plus the explicit CLI source locator and both roots, seeds
Torch from the native cold-init stream BEFORE any shell exists, builds the
learner/actor pair from the live model, verifies the C0 container through
the existing writer, asks native for the run descriptor from the verified
manifest, copies the transport bytes into the run root, and returns every
live handle Main needs. No external prerequisite ordering, no default
archive path, no sidecar files.

RNG discipline: construction draws are seeded from the native cold-init
stream BEFORE any shell exists. Unseeded construction perturbs the
captured ``rng_state``, which forks ordinary manifest ids across fresh
processes running identical inputs. Restoration stays strictly last,
after all construction, weight loads, and optimizer/scheduler builds.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from dataclasses import replace as _replace_config
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.training._rc_sections import RunConfig

try:
    from hydra2._native import contracts as _contracts_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _contracts_bridge = None


@dataclass(frozen=True, slots=True)
class RlIdentities:
    """Torch-derivable identities for ``build_rl_manifest`` (native cannot
    compute these).

    Only the model/utility bindings live here. Every other manifest arg
    (run_spec/optimizer/scheduler/environment/rules/action/observation
    hashes) comes from the native descriptor's verified
    ``run_spec_hash``/``spec_hashes``/``environment_hash``/``identities``
    sections, never from Python re-derivation.
    """

    model_spec_hash: str
    utility_manifest_hash: str
    utility_id: str


@dataclass(slots=True)
class TorchColdState:
    """Live cold handles plus the native descriptor and C0 receipt.

    ``c0_receipt`` is root-relative (``checkpoints/c0.pt`` locator plus
    the native-verified manifest id); no absolute source path or sidecar
    reference leaves this boundary.
    """

    learner: Any
    actor: Any
    optimizer: Any
    scheduler: Any
    manifest_c0: Any
    identities: RlIdentities
    native_run_descriptor: dict[str, Any]
    c0_receipt: dict[str, Any]


def _require_rl(config: Any) -> Any:
    """Return the validated ``rl`` section or fail closed (structural only)."""
    rl = getattr(config, "rl", None)
    if rl is None:
        raise ContractError("rl_checkpoint requires run.kind 'rl' (rl section absent)")
    return rl


def verify_model_binding(*, model: Any, rl: Any) -> tuple[str, str, str]:
    """Bind the live shell's model/utility identities to the ``rl`` section.

    Native code cannot derive these (architecture and utility live in
    Torch), so this gate stays Python-side. Everything else in ``rl`` is
    native-enforced by the ``prepare_run`` call inside
    :func:`prepare_torch_state`.
    """
    model_spec_hash = str(model.model_identity)
    if str(rl.initial_model_spec_hash) != model_spec_hash:
        raise ContractError("rl_checkpoint rl.initial_model_spec_hash != live model identity")
    utility_hash = str(model.utility_manifest_hash)
    if str(rl.utility_hash) != utility_hash:
        raise ContractError("rl_checkpoint rl.utility_hash != live utility manifest digest")
    return (model_spec_hash, utility_hash, str(model.utility_id))


def apply_cold_seed(*, seed: bytes) -> int:
    """Seed Torch CPU/CUDA construction RNG from native cold-init bytes.

    Uses the first 8 bytes big-endian as the u64 (single conversion site;
    the full 32 bytes remain the stream seed for any future consumer).
    Must run before any shell construction; returns the applied u64.
    """
    import torch

    if len(seed) != 32:
        raise ContractError(f"rl_checkpoint cold seed must be 32 bytes, got {len(seed)}")
    value = int.from_bytes(seed[:8], "big")
    _ = torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value


def _live_utility_doc(learner: Any) -> dict[str, Any]:
    """Utility framing from real owners: id from the live Torch shell.

    Rank values come from the frozen contracts bridge golden (the same
    values the native asset gate cross-checks against the rules-file uma);
    this module never restates them. Native verifies both fields.
    """
    if _contracts_bridge is None or not hasattr(_contracts_bridge, "ORACLE_RANK_VALUES"):
        raise ImportError(
            "hydra2._native contracts with ORACLE_RANK_VALUES not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return {
        "utility_id": str(learner.utility_id),
        "rank_values": [float(v) for v in _contracts_bridge.ORACLE_RANK_VALUES],
    }


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_c0_copy(*, source: Path, run_root: Path) -> Path:
    """Copy verified C0 transport bytes into the run root (no sidecar).

    Streams bytes through a temp file plus atomic replace with fsync;
    adjacent sidecar/confidential files are never touched. Call only
    after :func:`load_initial_weights` has verified the source.
    """
    source = Path(source)
    dest = Path(run_root) / "checkpoints" / "c0.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_tmp = dest.with_name(dest.name + ".tmp")
    with open(source, "rb") as fh_in, open(staging_tmp, "wb") as fh_out:
        shutil.copyfileobj(fh_in, fh_out)
        fh_out.flush()
        os.fsync(fh_out.fileno())
    os.replace(staging_tmp, dest)
    _fsync_dir(dest.parent)
    return dest


def _freeze_actor(*, actor: Any, learner: Any) -> None:
    """Freeze a C0-loaded actor: eval, no grad, distinct finite storage."""
    import torch

    actor.load_state_dict(learner.state_dict(), strict=True)
    actor.eval()
    actor.requires_grad_(False)
    for (_, lparam), (aname, aparam) in zip(
        learner.named_parameters(), actor.named_parameters(), strict=True
    ):
        if lparam.data_ptr() == aparam.data_ptr():
            raise ContractError(f"rl_checkpoint actor shares storage with learner at {aname!r}")
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if not bool(torch.isfinite(param).all()):
                raise ContractError(f"rl_checkpoint actor weights non-finite at {name!r}")


def prepare_torch_state(
    *,
    config: RunConfig,
    source: Path,
    device: Any = None,
    repo_root: Path,
    run_root: Path,
) -> TorchColdState:
    """Single-call cold assembly: seed, shells, C0 weights, descriptor.

    Order: derive and apply the native cold-init seed BEFORE any shell ->
    construct the learner shell via the shared ``stream_build`` constructor
    -> bind its model/utility identities (reused, no third shell) ->
    construct the distinct actor shell -> load verified IL weights into the
    learner (strict, eval) -> ask native for the run descriptor from the
    verified manifest -> fresh fused AdamW + constant scheduler -> freeze
    the actor onto C0 weights (distinct storage, eval, no grad) -> copy the
    verified transport bytes into the run root. Torch shells are ephemeral
    until return: any native/weight gate failure raises before Main
    receives handles. RNG restoration is Main's explicit next step (see
    :func:`restore_rng_last`) and always happens after construction.
    ``source`` is the explicit CLI locator; nothing here defaults or
    guesses an archive path.
    """
    from hydra2.engines.riichienv.identity import ADAPTER_VERSION, ENGINE_IDENTITY
    from hydra2.runtime.checkpoint import manifest_to_json
    from hydra2.training._rc_digest import run_config_to_dict
    from hydra2.training.rl_checkpoint_native import (
        build_run_config_json,
        native_cold_init_seed,
        native_prepare_run,
    )
    from hydra2.training.stream_build import _build_model

    rl = _require_rl(config)
    seed = native_cold_init_seed(
        master_seed_hex=str(rl.master_seed_hex),
        experiment_id=str(rl.experiment_id),
        replicate_id=int(rl.replicate_id),
        attempt_id=int(rl.attempt_id),
        initial_checkpoint_id=str(rl.initial_checkpoint_id),
    )
    _ = apply_cold_seed(seed=seed)
    learner = _build_model(config)
    if device is not None:
        learner.to(device)
    learner.eval()
    learner.requires_grad_(True)
    model_spec_hash, utility_hash, utility_id = verify_model_binding(model=learner, rl=rl)
    actor = _build_model(config)
    if device is not None:
        actor.to(device)
    expected = {
        "run_spec_hash": "",
        "source_hash": "",
        "model_spec_hash": model_spec_hash,
        "rules_hash": str(rl.rules_hash),
        "utility_manifest_hash": utility_hash,
        "action_schema_hash": str(rl.action_table_hash),
        "observation_schema_hash": str(rl.observation_schema_hash),
    }
    manifest_doc = _peek_manifest_ids(source=Path(source))
    expected["run_spec_hash"] = manifest_doc["run_spec_hash"]
    expected["source_hash"] = manifest_doc["source_hash"]
    manifest_c0 = load_initial_weights(source=Path(source), model=learner, expected=expected)
    run_config_json = build_run_config_json(
        resolved=run_config_to_dict(config),
        adapter_version=str(ADAPTER_VERSION),
        utility=_live_utility_doc(learner),
        opponent_catalog=[],
        environment_hash=str(ENGINE_IDENTITY.environment_hash),
    )
    descriptor = native_prepare_run(
        run_config_json=run_config_json,
        initial_manifest_json=json.dumps(manifest_to_json(manifest_c0)),
        root_dir=str(run_root),
        repo_dir=str(repo_root),
    )
    optimizer = build_fresh_adamw(config=config, model=learner)
    scheduler = build_constant_scheduler(optimizer=optimizer)
    _freeze_actor(actor=actor, learner=learner)
    _ = _publish_c0_copy(source=Path(source), run_root=Path(run_root))
    try:
        c0_manifest_id = str(descriptor["initial_checkpoint_id"])
    except KeyError as exc:
        raise ContractError("rl_checkpoint descriptor lacks initial_checkpoint_id") from exc
    if c0_manifest_id != str(rl.initial_checkpoint_id):
        raise ContractError("rl_checkpoint descriptor C0 id drift after verify")
    return TorchColdState(
        learner=learner,
        actor=actor,
        optimizer=optimizer,
        scheduler=scheduler,
        manifest_c0=manifest_c0,
        identities=RlIdentities(
            model_spec_hash=model_spec_hash,
            utility_manifest_hash=utility_hash,
            utility_id=utility_id,
        ),
        native_run_descriptor=descriptor,
        c0_receipt={"locator": "checkpoints/c0.pt", "manifest_id": c0_manifest_id},
    )


def _peek_manifest_ids(*, source: Path) -> dict[str, str]:
    """Read the source container's self-claimed run/source ids (one load).

    These two ids only select the writer's expectations below; every
    content binding (model/rules/utility/action/observation plus all
    section integrity digests) is verified by
    :func:`load_initial_weights` before any weight applies.
    """
    import torch

    from hydra2.contracts.common import ContractError as _ContractError

    source = Path(source)
    if not source.is_file():
        raise _ContractError(f"rl_checkpoint initial source does not exist: {source}")
    try:
        container = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise _ContractError(f"rl_checkpoint source container unreadable: {source}") from exc
    if not isinstance(container, dict) or "checkpoint_manifest" not in container:
        raise _ContractError(f"rl_checkpoint source container malformed: {source}")
    manifest_doc = container["checkpoint_manifest"]
    if not isinstance(manifest_doc, dict):
        raise _ContractError(f"rl_checkpoint source manifest malformed: {source}")
    return {
        "run_spec_hash": str(manifest_doc.get("run_spec_hash", "")),
        "source_hash": str(manifest_doc.get("dataset_manifest_hash", "")),
    }


def load_initial_weights(*, source: Path, model: Any, expected: dict[str, str]) -> Any:
    """Verify the IL container fully, then load ONLY its model weights.

    ``expected`` carries ``run_spec_hash``, ``source_hash`` (the
    container's own dataset identity, read by :func:`_peek_manifest_ids`),
    ``model_spec_hash``, ``rules_hash``, ``utility_manifest_hash``,
    ``action_schema_hash``, and ``observation_schema_hash``. Optimizer,
    scheduler, and RNG sections are integrity-checked but never applied: a
    fresh trajectory must not inherit supervised moments. Returns the
    verified manifest (C0) with the model left in eval mode.
    """
    from hydra2.runtime.checkpoint import load_model_weights_only

    manifest, weights = load_model_weights_only(
        source=Path(source),
        expected_run_spec_hash=expected["run_spec_hash"],
        expected_source_hash=expected["source_hash"],
        expected_model_spec_hash=expected["model_spec_hash"],
        expected_rules_hash=expected["rules_hash"],
        expected_utility_manifest_hash=expected["utility_manifest_hash"],
        expected_action_schema_hash=expected["action_schema_hash"],
        expected_observation_schema_hash=expected["observation_schema_hash"],
    )
    model.load_state_dict(weights, strict=True)
    model.eval()
    return manifest


def build_fresh_adamw(*, config: RunConfig, model: Any) -> Any:
    """Fresh fused AdamW over the existing three-group split (no moments).

    Bucket membership and the fused/Foreach kernel choice come from the
    shared ``stream_build`` owners; the numerics (lr/betas/decay/head
    multiplier) come from the native-verified ``rl`` section, framed as a
    real ``RunConfig`` view so the shared grouping function runs
    unchanged. Never inherits supervised moments.
    """
    import torch

    from hydra2.training._rc_sections import OptimizerConfig
    from hydra2.training.stream_build import _optimizer_fused_kwargs, _optimizer_param_groups

    rl = _require_rl(config)
    opt_view = _replace_config(
        config,
        optimizer=OptimizerConfig(
            name=config.optimizer.name,
            lr=float(rl.optimizer_lr),
            betas=(float(rl.optimizer_beta1), float(rl.optimizer_beta2)),
            weight_decay=float(rl.optimizer_weight_decay),
            head_lr_mult=float(rl.head_lr_mult),
        ),
    )
    groups = _optimizer_param_groups(opt_view, model)
    kwargs = _optimizer_fused_kwargs()
    if groups is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(rl.optimizer_lr),
            betas=(float(rl.optimizer_beta1), float(rl.optimizer_beta2)),
            weight_decay=float(rl.optimizer_weight_decay),
            **kwargs,
        )
    return torch.optim.AdamW(
        groups,
        lr=float(rl.optimizer_lr),
        betas=(float(rl.optimizer_beta1), float(rl.optimizer_beta2)),
        weight_decay=float(rl.optimizer_weight_decay),
        **kwargs,
    )


def build_constant_scheduler(*, optimizer: Any) -> Any:
    """Constant LR scheduler (warmup 0: the LambdaLR identity arm)."""
    import torch

    # Mirrors the ``constant`` arm of the supervised scheduler builder with
    # warmup<=0 (returns the main scheduler directly): LambdaLR at 1.0.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)


def ordinary_snapshot(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    sampler_state: Any,
    training_state: Any,
) -> dict[str, Any]:
    """Six-section ordinary payload (existing writer contract).

    ``sampler_state``/``training_state`` MUST come from the native
    skeletons (counters/order only, never progress). ``state_dict`` views
    alias live tensors: the caller quiesces first (completed optimizer
    boundary, accumulation zero); the existing writer persists the bytes.
    """
    from hydra2.runtime.checkpoint import capture_rng_state

    scheduler_state: Any = scheduler.state_dict() if hasattr(scheduler, "state_dict") else {}
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "training_state": dict(training_state),
        "sampler_state": dict(sampler_state),
        "rng_state": capture_rng_state(),
    }


def build_rl_manifest(
    *,
    payload: dict[str, Any],
    run_spec_hash: str,
    model_spec_hash: str,
    optimizer_spec_hash: str,
    scheduler_spec_hash: str,
    environment_hash: str,
    rules_hash: str,
    utility_manifest_hash: str,
    action_schema_hash: str,
    observation_schema_hash: str,
    rollout_artifact_hash: str,
    parent_checkpoint_hash: str,
) -> Any:
    """RL ordinary manifest (dataset null, rollout set, fixed parent).

    Optimizer/scheduler spec hashes arrive from the native descriptor
    ``spec_hashes`` (formed over the supervised doc shapes); this function
    only frames the existing manifest builder, never recomputes structure.
    """
    from hydra2.runtime.checkpoint import build_manifest

    return build_manifest(
        run_spec_hash=run_spec_hash,
        model_spec_hash=model_spec_hash,
        optimizer_spec_hash=optimizer_spec_hash,
        scheduler_spec_hash=scheduler_spec_hash,
        environment_hash=environment_hash,
        rules_hash=rules_hash,
        utility_manifest_hash=utility_manifest_hash,
        action_schema_hash=action_schema_hash,
        observation_schema_hash=observation_schema_hash,
        dataset_manifest_hash=None,
        rollout_artifact_hash=rollout_artifact_hash,
        parent_checkpoint_hash=parent_checkpoint_hash,
        payload=payload,
    )


def save_ordinary(*, destination: Path, manifest: Any, payload: dict[str, Any]) -> Path:
    """Atomically publish one ordinary checkpoint (existing writer)."""
    from hydra2.runtime.checkpoint import save_checkpoint

    return save_checkpoint(destination=Path(destination), manifest=manifest, payload=payload)


def reload_and_verify(
    *, source: Path, run_spec_hash: str, source_hash: str
) -> tuple[Any, dict[str, Any]]:
    """Verified reload: identity gates pass before any object mutates."""
    from hydra2.runtime.checkpoint import load_checkpoint

    return load_checkpoint(
        source=Path(source),
        expected_run_spec_hash=run_spec_hash,
        expected_source_hash=source_hash,
    )


def restore_actor_weights(*, model: Any, weights: Any) -> None:
    """Strict frozen-actor restore (distinct storage, eval, no grad, finite)."""
    import torch

    model.load_state_dict(weights, strict=True)
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not bool(torch.isfinite(param).all()):
                raise ContractError(f"rl_checkpoint actor weights non-finite at {name!r}")


def restore_rng_last(*, rng_state: Any) -> None:
    """Restore Torch CPU/CUDA RNG state. Call strictly last, after all
    construction, weight loads, and optimizer/scheduler builds."""
    from hydra2.runtime.checkpoint import _restore_rng_state

    _restore_rng_state(rng_state)


def apply_verified_state(
    *,
    payload: dict[str, Any],
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
) -> None:
    """Apply an already-verified payload: model, optimizer, scheduler,
    then RNG last (enforced order; the distinct actor uses
    :func:`restore_actor_weights` instead of this path)."""
    from hydra2.runtime.checkpoint import apply_checkpoint

    apply_checkpoint(
        payload, model=model, optimizer=optimizer, scheduler=scheduler, restore_rng=False
    )
    restore_rng_last(rng_state=payload["rng_state"])


__all__ = [
    "RlIdentities",
    "TorchColdState",
    "apply_cold_seed",
    "apply_verified_state",
    "build_constant_scheduler",
    "build_fresh_adamw",
    "build_rl_manifest",
    "load_initial_weights",
    "ordinary_snapshot",
    "prepare_torch_state",
    "reload_and_verify",
    "restore_actor_weights",
    "restore_rng_last",
    "save_ordinary",
    "verify_model_binding",
]
