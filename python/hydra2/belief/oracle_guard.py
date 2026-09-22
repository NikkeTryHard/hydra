"""WP-07B oracle guard — privileged firewall and split/wall leakage gates.

Owns the firewall vocabulary (privileged keys, actor-forbidden keys, the
authorized train split) and the fail-closed validators shared by the store
and join paths: actor batch/row leakage checks, wall/split disjointness,
and privileged/actor shard-path helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError, CorruptArtifactError

#: Frozen firewall vocabulary (single source: ``hydra2._native.contracts``;
#: this module keeps the translators + loader-coupled roots, ``__all__`` unchanged).
PRIVILEGED_KEYS: frozenset[str] = _bridge_contracts.GUARD_PRIVILEGED_KEYS
FORBIDDEN_IN_ACTOR_KEYS: frozenset[str] = _bridge_contracts.GUARD_FORBIDDEN_IN_ACTOR_KEYS

# Only train split is authorized for oracle distillation
AUTHORIZED_TRAIN_SPLIT: str = _bridge_contracts.GUARD_AUTHORIZED_TRAIN_SPLIT
AUTHORIZED_SPLITS_FOR_INFERENCE: frozenset[str] = (
    _bridge_contracts.GUARD_AUTHORIZED_SPLITS_FOR_INFERENCE
)
# Oracle loader rejects any split != train


def _require_train_split(split: str) -> None:
    try:
        _bridge_contracts.guard_require_train_split(split)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def validate_actor_batch_no_privileged(batch: dict[str, Any]) -> None:
    """Hard failure if actor-visible batch contains any privileged field.

    Inference encoders MUST call this before encoding. It checks both top-level
    keys and nested ``actor_observation`` dict keys for leakage.
    """
    # Nested observation dict
    _obs_a: Any | None = batch.get("actor_observation")
    _obs_b: Any | None = batch.get("observation")
    _obs_c: Any | None = batch.get("obs")
    obs: Any | None = _obs_a if _obs_a is not None else (_obs_b if _obs_b is not None else _obs_c)
    try:
        _bridge_contracts.guard_validate_actor_batch(
            list(batch.keys()), list(obs.keys()) if isinstance(obs, dict) else []
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def assert_no_privileged_leakage_in_actor_row(row: dict[str, Any]) -> None:
    """Check a raw parquet row dict for privileged leakage (actor shard)."""
    for bad in FORBIDDEN_IN_ACTOR_KEYS:
        if bad in row:
            raise ContractError(f"privileged field {bad!r} in actor row {row.get('decision_id')!r}")
    obs_raw = row.get("actor_observation")
    if isinstance(obs_raw, str):
        try:
            obs = json.loads(obs_raw)
        except Exception as exc:
            raise CorruptArtifactError(
                f"actor_observation not JSON for {row.get('decision_id')!r}"
            ) from exc
        if isinstance(obs, dict):
            for k_any in obs:
                k2: str = k_any  # type: ignore[assignment]  # reason: dict key statically Any; str validated by membership check
                if k2 in FORBIDDEN_IN_ACTOR_KEYS:
                    raise ContractError(
                        f"privileged field {k2!r} inside actor_observation for {row.get('decision_id')!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
                    )
    elif isinstance(obs_raw, dict):
        for k_any in obs_raw:
            k3: str = k_any  # type: ignore[assignment]  # reason: dict key statically Any; str validated by membership check
            if k3 in FORBIDDEN_IN_ACTOR_KEYS:
                raise ContractError(
                    f"privileged field {k3!r} inside actor_observation dict for {row.get('decision_id')!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
                )


def check_wall_leakage(
    train_wall_ids: list[str] | set[str], held_out_wall_ids: list[str] | set[str]
) -> None:
    """Hard failure if any wall_id appears in both train and held-out.

    Walls are the partitioning unit (whole games before decisions, SPEC 12.4).
    Leakage corrupts held-out proper scores.
    """
    try:
        _bridge_contracts.guard_check_wall_leakage(list(train_wall_ids), list(held_out_wall_ids))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def check_split_disjoint(
    train_ids: list[str] | set[str], held_out_ids: list[str] | set[str]
) -> None:
    """Hard failure if any decision_id crosses train/held_out boundary."""
    try:
        _bridge_contracts.guard_check_split_disjoint(list(train_ids), list(held_out_ids))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _privileged_shard_paths(parquet_dir: Path) -> list[Path]:
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.is_dir():
        raise ContractError(f"privileged parquet_dir not found: {parquet_dir}")
    shards = sorted(parquet_dir.glob("privileged-*.parquet"))
    # Fallback: also accept "oracle-*.parquet"
    if len(shards) == 0:
        shards = sorted(parquet_dir.glob("oracle-*.parquet"))
    return shards


def _actor_shard_paths(parquet_dir: Path) -> list[Path]:
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.is_dir():
        raise ContractError(f"parquet_dir not found: {parquet_dir}")
    return sorted(parquet_dir.glob("actor-*.parquet"))


__all__ = [
    "AUTHORIZED_SPLITS_FOR_INFERENCE",
    "AUTHORIZED_TRAIN_SPLIT",
    "FORBIDDEN_IN_ACTOR_KEYS",
    "PRIVILEGED_KEYS",
    "_actor_shard_paths",
    "_privileged_shard_paths",
    "_require_train_split",
    "assert_no_privileged_leakage_in_actor_row",
    "check_split_disjoint",
    "check_wall_leakage",
    "validate_actor_batch_no_privileged",
]
