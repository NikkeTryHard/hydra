"""Project-owned checkpoint save/load (SPEC 10).

The adapter never defines the schema; this module does. Serialization is
ordinary single-device ``torch.save``; manifest identity is independent of
container metadata. ``run_spec_hash`` and the selected source identity hash
(dataset_manifest_hash for supervised/distill, rollout_artifact_hash for RL)
are verified before any runtime object is mutated.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from hydra2._canon import (
    NonFiniteNumberError,
    _mkstemp_o_excl,
    require_digest_match,
    sha256_digest_of_json,
)
from hydra2.contracts.common import (
    ContractError,
    CorruptArtifactError,
    DigestText,
    SchemaVersion,
    make_digest_text,
    make_schema_version,
)

CHECKPOINT_SCHEMA_VERSION = SchemaVersion("1.0.0")
SUPPORTED_CHECKPOINT_VERSIONS = ("1.0.0",)

PAYLOAD_SECTION_KEYS = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "training_state",
    "sampler_state",
    "rng_state",
)


class StateTreeError(CorruptArtifactError):
    """Training state contains a leaf type that cannot be hashed deterministically."""


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    checkpoint_version: SchemaVersion
    run_spec_hash: DigestText
    model_spec_hash: DigestText
    model_state_hash: DigestText
    optimizer_spec_hash: DigestText
    optimizer_state_hash: DigestText
    scheduler_spec_hash: DigestText
    scheduler_state_hash: DigestText
    training_state_hash: DigestText
    sampler_state_hash: DigestText
    rng_state_hash: DigestText
    environment_hash: DigestText
    rules_hash: DigestText
    utility_manifest_hash: DigestText
    action_schema_hash: DigestText
    observation_schema_hash: DigestText
    dataset_manifest_hash: DigestText | None
    rollout_artifact_hash: DigestText | None
    parent_checkpoint_hash: DigestText | None


# ---------------------------------------------------------------------------
# Deterministic state-tree hashing
# ---------------------------------------------------------------------------


def state_tree(value: Any) -> Any:
    """Canonical structural form of arbitrary training state."""
    import torch

    if isinstance(value, torch.Tensor):
        cpu = value.detach().to("cpu", copy=True).contiguous()
        raw = cpu.flatten().view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "bytes_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, dict):
        dict_value: dict[Any, Any] = cast("dict[Any, Any]", value)
        items: dict[str, Any] = {}
        for key, item in dict_value.items():
            if isinstance(key, str):
                encoded = key
            elif isinstance(key, bool):
                raise ContractError("state map key must not be bool")
            elif isinstance(key, int):
                # Injective, deterministic encoding for optimizer param indices.
                encoded = f"#int:{key}"
            else:
                raise ContractError(f"state map key must be str or int, got {type(key).__name__}")
            items[encoded] = state_tree(cast("Any", item))
        return {"kind": "map", "items": items}
    if isinstance(value, (list, tuple)):
        seq_value: list[Any] = cast("list[Any]", list(value))
        return {"kind": "array", "items": [state_tree(cast("Any", item)) for item in seq_value]}
    if isinstance(value, bytes):
        return {"kind": "bytes_hex", "value": value.hex()}
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise NonFiniteNumberError(f"non-finite float in state: {value!r}")
        return {"kind": "scalar", "value": value}
    raise StateTreeError(
        f"unsupported state leaf type {type(value).__name__}; convert to tensor/scalar first"
    )


def hash_state_tree(value: Any) -> DigestText:
    """Bitwise-identity digest of arbitrary training state."""
    return sha256_digest_of_json(state_tree(value))


# ---------------------------------------------------------------------------
# Manifest construction and validation
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    run_spec_hash: str,
    model_spec_hash: str,
    optimizer_spec_hash: str,
    scheduler_spec_hash: str,
    environment_hash: str,
    rules_hash: str,
    utility_manifest_hash: str,
    action_schema_hash: str,
    observation_schema_hash: str,
    dataset_manifest_hash: str | None,
    rollout_artifact_hash: str | None,
    parent_checkpoint_hash: str | None = None,
    payload: Mapping[str, Any],
) -> CheckpointManifest:
    """Build a manifest from identity digests plus live payload sections.

    Source identity mirrors RunSpec: supervised/distill checkpoints carry
    ``dataset_manifest_hash`` and null rollout; RL checkpoints carry
    ``rollout_artifact_hash`` and null dataset — exactly one MUST be set.
    """
    _require_payload_sections(payload)
    if (dataset_manifest_hash is None) == (rollout_artifact_hash is None):
        raise ContractError(
            "checkpoint source identity must set exactly one of "
            "dataset_manifest_hash (supervised/distill) or "
            "rollout_artifact_hash (RL)"
        )
    manifest = CheckpointManifest(
        checkpoint_version=make_schema_version(CHECKPOINT_SCHEMA_VERSION),
        run_spec_hash=make_digest_text(run_spec_hash),
        model_spec_hash=make_digest_text(model_spec_hash),
        model_state_hash=hash_state_tree(payload["model_state"]),
        optimizer_spec_hash=make_digest_text(optimizer_spec_hash),
        optimizer_state_hash=hash_state_tree(payload["optimizer_state"]),
        scheduler_spec_hash=make_digest_text(scheduler_spec_hash),
        scheduler_state_hash=hash_state_tree(payload["scheduler_state"]),
        training_state_hash=hash_state_tree(payload["training_state"]),
        sampler_state_hash=hash_state_tree(payload["sampler_state"]),
        rng_state_hash=hash_state_tree(payload["rng_state"]),
        environment_hash=make_digest_text(environment_hash),
        rules_hash=make_digest_text(rules_hash),
        utility_manifest_hash=make_digest_text(utility_manifest_hash),
        action_schema_hash=make_digest_text(action_schema_hash),
        observation_schema_hash=make_digest_text(observation_schema_hash),
        dataset_manifest_hash=_optional_digest("dataset_manifest_hash", dataset_manifest_hash),
        rollout_artifact_hash=_optional_digest("rollout_artifact_hash", rollout_artifact_hash),
        parent_checkpoint_hash=_optional_digest("parent_checkpoint_hash", parent_checkpoint_hash),
    )
    validate_checkpoint_manifest(manifest)
    return manifest


def _optional_digest(name: str, value: str | None) -> DigestText | None:
    return None if value is None else make_digest_text(value)


def validate_checkpoint_manifest(manifest: CheckpointManifest) -> None:
    expected_fields = tuple(f.name for f in fields(CheckpointManifest))
    actual_fields = tuple(f.name for f in fields(type(manifest)))
    if expected_fields != actual_fields:  # pragma: no cover - schema drift guard
        raise ContractError("checkpoint manifest schema drift detected")
    if manifest.checkpoint_version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ContractError(
            f"unsupported checkpoint_version {manifest.checkpoint_version!r}; "
            f"supported: {list(SUPPORTED_CHECKPOINT_VERSIONS)}"
        )
    if (manifest.dataset_manifest_hash is None) == (manifest.rollout_artifact_hash is None):
        raise ContractError(
            "manifest must declare exactly one source identity: "
            "dataset_manifest_hash XOR rollout_artifact_hash"
        )


def manifest_to_json(manifest: CheckpointManifest) -> dict[str, Any]:
    return asdict(manifest)


def manifest_from_json(raw: Mapping[str, Any]) -> CheckpointManifest:
    expected_keys = {f.name for f in fields(CheckpointManifest)}
    missing: list[str] = sorted(expected_keys - set(raw))
    unknown: list[str] = sorted(set(raw) - expected_keys)
    if len(missing) != 0 or len(unknown) != 0:
        raise ContractError(
            f"checkpoint manifest envelope mismatch; missing={missing} unknown={unknown}"
        )
    try:
        manifest = CheckpointManifest(
            checkpoint_version=make_schema_version(raw["checkpoint_version"]),
            run_spec_hash=make_digest_text(raw["run_spec_hash"]),
            model_spec_hash=make_digest_text(raw["model_spec_hash"]),
            model_state_hash=make_digest_text(raw["model_state_hash"]),
            optimizer_spec_hash=make_digest_text(raw["optimizer_spec_hash"]),
            optimizer_state_hash=make_digest_text(raw["optimizer_state_hash"]),
            scheduler_spec_hash=make_digest_text(raw["scheduler_spec_hash"]),
            scheduler_state_hash=make_digest_text(raw["scheduler_state_hash"]),
            training_state_hash=make_digest_text(raw["training_state_hash"]),
            sampler_state_hash=make_digest_text(raw["sampler_state_hash"]),
            rng_state_hash=make_digest_text(raw["rng_state_hash"]),
            environment_hash=make_digest_text(raw["environment_hash"]),
            rules_hash=make_digest_text(raw["rules_hash"]),
            utility_manifest_hash=make_digest_text(raw["utility_manifest_hash"]),
            action_schema_hash=make_digest_text(raw["action_schema_hash"]),
            observation_schema_hash=make_digest_text(raw["observation_schema_hash"]),
            dataset_manifest_hash=(
                None
                if raw["dataset_manifest_hash"] is None
                else make_digest_text(raw["dataset_manifest_hash"])
            ),
            rollout_artifact_hash=(
                None
                if raw["rollout_artifact_hash"] is None
                else make_digest_text(raw["rollout_artifact_hash"])
            ),
            parent_checkpoint_hash=(
                None
                if raw["parent_checkpoint_hash"] is None
                else make_digest_text(raw["parent_checkpoint_hash"])
            ),
        )
    except KeyError as exc:
        raise ContractError(f"checkpoint manifest field unreadable: {exc}") from exc
    validate_checkpoint_manifest(manifest)
    return manifest


def _require_payload_sections(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ContractError("checkpoint payload must be a mapping")
    missing: list[str] = [key for key in PAYLOAD_SECTION_KEYS if key not in payload]
    if len(missing) != 0:
        raise ContractError(f"checkpoint payload missing sections: {missing}")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_checkpoint(
    *,
    destination: Path,
    manifest: CheckpointManifest,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically publish ``{manifest, payload}`` via single-device torch.save."""
    import torch

    _require_payload_sections(payload)
    destination = Path(destination)
    container = {
        "checkpoint_manifest": manifest_to_json(manifest),
        "payload": dict(payload),
    }
    fd, temp_name = _mkstemp_o_excl(destination.parent, destination.name)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(container, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(destination.parent)
    return destination


def load_checkpoint(
    *,
    source: Path,
    expected_run_spec_hash: str,
    expected_source_hash: str,
) -> tuple[CheckpointManifest, dict[str, Any]]:
    """Read a checkpoint and fully verify identity BEFORE any object mutation.

    Order of gates: container integrity, manifest schema, run_spec_hash,
    selected source identity hash, then per-section integrity digests. Any
    failure raises before a mutable runtime object is touched.
    """
    import torch

    source = Path(source)
    if not source.is_file():
        raise CorruptArtifactError(f"checkpoint file does not exist: {source}")
    try:
        container = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CorruptArtifactError(f"checkpoint container unreadable: {source}") from exc
    if (
        not isinstance(container, dict)
        or "checkpoint_manifest" not in container
        or "payload" not in container
    ):
        raise CorruptArtifactError(f"checkpoint container malformed: {source}")

    container_dict: dict[str, Any] = cast("dict[str, Any]", container)
    manifest = manifest_from_json(
        cast("Mapping[str, Any]", cast("dict[str, Any]", container_dict["checkpoint_manifest"]))
    )
    payload: dict[str, Any] = cast("dict[str, Any]", container_dict["payload"])
    _require_payload_sections(cast("Mapping[str, Any]", payload))

    require_digest_match(
        recorded=expected_run_spec_hash,
        recomputed=manifest.run_spec_hash,
        subject="run_spec_hash",
    )
    selected: DigestText | None = (
        manifest.dataset_manifest_hash
        if manifest.dataset_manifest_hash is not None
        else manifest.rollout_artifact_hash
    )
    assert selected is not None  # guaranteed by validate_checkpoint_manifest
    require_digest_match(
        recorded=expected_source_hash,
        recomputed=selected,
        subject="selected source identity hash",
    )

    section_hashes = (
        ("model_state", manifest.model_state_hash),
        ("optimizer_state", manifest.optimizer_state_hash),
        ("scheduler_state", manifest.scheduler_state_hash),
        ("training_state", manifest.training_state_hash),
        ("sampler_state", manifest.sampler_state_hash),
        ("rng_state", manifest.rng_state_hash),
    )
    for section, recorded in section_hashes:
        recomputed = hash_state_tree(cast("Any", payload[section]))
        if recomputed != recorded:
            raise CorruptArtifactError(
                f"checkpoint section {section!r} corrupt: "
                f"recorded {recorded} != recomputed {recomputed}"
            )
    return manifest, payload


def apply_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> None:
    """Mutate runtime objects from an already-verified payload."""

    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])


def resume_checkpoint(
    *,
    source: Path,
    run_spec_hash: str,
    source_hash: str,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
) -> CheckpointManifest:
    """Verified load followed by application, in that enforced order."""
    manifest, payload = load_checkpoint(
        source=source,
        expected_run_spec_hash=run_spec_hash,
        expected_source_hash=source_hash,
    )
    apply_checkpoint(payload, model=model, optimizer=optimizer, scheduler=scheduler)
    return manifest


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    import torch

    if "cpu" in rng_state:
        torch.set_rng_state(_as_byte_tensor(cast("Any", rng_state["cpu"])))
    if "cuda" in rng_state and torch.cuda.is_available():
        cuda_states: Any = cast("Any", rng_state["cuda"])
        if isinstance(cuda_states, list):
            cuda_list: list[Any] = cast("list[Any]", cuda_states)
            torch.cuda.set_rng_state_all([_as_byte_tensor(cast("Any", s)) for s in cuda_list])
        else:
            torch.cuda.set_rng_state(_as_byte_tensor(cast("Any", cuda_states)))


def _as_byte_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.uint8, device="cpu")
    return torch.tensor(list(value), dtype=torch.uint8)


def _fsync_dir(directory: Path) -> None:
    # Portable directory fsync: Windows NT cannot open a directory with
    # os.open(O_RDONLY) (PermissionError/OSError); directory fsync is a
    # no-op on NTFS where file-handle fsync already guarantees durability.
    # Evidence: https://docs.python.org/3/library/os.html#os.fsync
    # Evidence: https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-flushfilebuffers
    # Evidence: https://github.com/tox-dev/platformdirs
    # Evidence: https://github.com/fsspec/universal_pathlib (XDG/portable)
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:  # noqa: SIM105 — explicit try/except mirrors atomic.py portable pattern (os.name nt guard)
            os.close(fd)
        except OSError:
            pass
