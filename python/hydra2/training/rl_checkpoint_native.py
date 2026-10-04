"""Thin native wrappers for the continuous-RL cold boundary.

Typing/firewall only: JSON framing plus fail-closed bridge resolution.
All structure/identity lives in ``hydra2._native.rl`` (Rust); Torch ops
live in :mod:`hydra2.training.rl_checkpoint`. This split keeps the Torch
module under its size budget with no behavioral fork.
"""

from __future__ import annotations

import json
from typing import Any

from hydra2.contracts.common import ContractError

try:
    from hydra2._native import rl as _native_rl  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _native_rl = None  # type: ignore[assignment]


def _require_native_rl() -> Any:
    """Resolve the native cold surface, fail closed when unbuilt."""
    if _native_rl is None or not hasattr(_native_rl, "prepare_run"):
        raise ImportError(
            "hydra2._native extension with rl lifecycle not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _native_rl


def _encode(document: Any) -> bytes:
    try:
        return json.dumps(document).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"rl_checkpoint document not JSON-encodable: {exc}") from exc


def _decode(blob: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(bytes(blob).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"rl_checkpoint {name} not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"rl_checkpoint {name} must be an object")
    return value


def build_run_config_json(
    *,
    resolved: dict[str, Any],
    adapter_version: str,
    utility: dict[str, Any],
    opponent_catalog: list[dict[str, Any]],
    environment_hash: str,
) -> str:
    """Frame the native ``prepare_run`` input (single framing site)."""
    return json.dumps(
        {
            "resolved": dict(resolved),
            "adapter_version": str(adapter_version),
            "utility": dict(utility),
            "opponent_catalog": list(opponent_catalog),
            "environment_hash": str(environment_hash),
        }
    )


def native_recipe_digests() -> dict[str, Any]:
    """Frozen recipe composition digests (pin into the RL config first)."""
    bridge = _require_native_rl()
    try:
        blob = bytes(bridge.recipe_digests())
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _decode(blob, name="recipe digests")


def native_cold_init_seed(
    *,
    master_seed_hex: str,
    experiment_id: str,
    replicate_id: int,
    attempt_id: int,
    initial_checkpoint_id: str,
) -> bytes:
    """32 cold-init seed bytes (Torch applies the first 8 BE as its u64)."""
    bridge = _require_native_rl()
    try:
        blob = bytes(
            bridge.cold_init_seed(
                master_seed_hex=str(master_seed_hex),
                experiment_id=str(experiment_id),
                replicate_id=int(replicate_id),
                attempt_id=int(attempt_id),
                initial_checkpoint_id=str(initial_checkpoint_id),
            )
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    if len(blob) != 32:
        raise ContractError(f"rl_checkpoint cold seed must be 32 bytes, got {len(blob)}")
    return blob


def native_prepare_run(
    *, run_config_json: str, initial_manifest_json: str, root_dir: str, repo_dir: str
) -> dict[str, Any]:
    """Cold descriptor (native validates assets and forms identities)."""
    bridge = _require_native_rl()
    try:
        blob = bytes(
            bridge.prepare_run(
                run_config_json=str(run_config_json),
                initial_manifest_json=str(initial_manifest_json),
                root_dir=str(root_dir),
                repo_dir=str(repo_dir),
            )
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _decode(blob, name="run descriptor")


def live_collector_spec(
    *, run_descriptor: dict[str, Any], generation: int, policy_id: str
) -> dict[str, Any]:
    """Archive-free collector binding for one generation (native-bound)."""
    bridge = _require_native_rl()
    try:
        blob = bytes(
            bridge.live_collector_spec(
                run_descriptor_json=_encode(run_descriptor),
                generation=int(generation),
                policy_id=str(policy_id),
            )
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _decode(blob, name="live collector spec")


def policy_identity(
    *, run_descriptor: dict[str, Any], generation: int, parent_policy_id: str
) -> str:
    """Deterministic lineage id for the frozen behavior collecting a generation."""
    bridge = _require_native_rl()
    try:
        return str(
            bridge.policy_identity(
                run_descriptor_json=_encode(run_descriptor),
                generation=int(generation),
                parent_policy_id=str(parent_policy_id),
            )
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def native_choose_work(
    *, preferred_inference: bool, inference_eligible: bool, learner_eligible: bool
) -> tuple[str, bool]:
    """Pure scheduling verdict (kind, next preference)."""
    bridge = _require_native_rl()
    try:
        kind, nxt = bridge.choose_work(
            preferred_inference=bool(preferred_inference),
            inference_eligible=bool(inference_eligible),
            learner_eligible=bool(learner_eligible),
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return str(kind), bool(nxt)


def native_ppo_report(*, count: int, stats: list[float]) -> dict[str, Any]:
    """One accumulated 23-scalar buffer at the true count into report means."""
    bridge = _require_native_rl()
    if len(stats) != 23:
        raise ContractError(f"rl_checkpoint ppo_report needs 23 scalars, got {len(stats)}")
    try:
        blob = bytes(bridge.ppo_report(count=int(count), stats=[float(s) for s in stats]))
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _decode(blob, name="ppo report")


__all__ = [
    "build_run_config_json",
    "live_collector_spec",
    "native_choose_work",
    "native_cold_init_seed",
    "native_ppo_report",
    "native_prepare_run",
    "native_recipe_digests",
    "policy_identity",
]
