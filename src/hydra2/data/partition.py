"""Partition whole games before expansion — checklist item 6.

Requirements:
  - Assign complete games before decisions (never split a game)
  - Enforce source/player/time grouping when metadata permits
  - Reject exact and near duplicates across partitions
  - Keep rollout/evaluation walls disjoint
  - Split manifest stores algorithm/version/seed/input hashes
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.data.decode import GameRecord

Partition = Literal["train", "validation", "test", "decision_eval", "block_eval"]

__all__ = [
    "GameIdentity",
    "SplitManifest",
    "SplitSpec",
    "assign_partitions",
    "detect_exact_duplicates",
    "detect_near_duplicates",
    "write_split_manifest",
]


@dataclass(frozen=True, slots=True)
class GameIdentity:
    game_id: str
    object_id: str
    source_id: str
    player_ids: tuple[str, ...]
    timestamp: str | None
    wall_hash: str | None
    decoded_hash: str


@dataclass(frozen=True, slots=True)
class SplitSpec:
    algorithm: str
    version: str
    seed: int
    ratios: dict[Partition, float]
    grouping_keys: tuple[str, ...]
    wall_disjoint: bool


@dataclass(frozen=True, slots=True)
class SplitManifest:
    spec: SplitSpec
    assignments: dict[str, Partition]
    input_hashes: dict[str, str]
    digest: str


def _game_identity_for(record: GameRecord, acquisition_metadata: dict[str, object]) -> GameIdentity:
    source = str(acquisition_metadata.get("source", "unknown"))
    _pids_raw: object = acquisition_metadata.get("player_ids", [])
    if isinstance(_pids_raw, (list, tuple)):
        player_ids = tuple(str(x) for x in cast("list[object]", list(_pids_raw)))
    else:
        player_ids = ()
    if len(player_ids) == 0:
        player_ids = ("unknown",)
    ts = acquisition_metadata.get("timestamp")
    timestamp = ts if isinstance(ts, str) else None
    wall_hash = None
    if record.wall_tiles is not None:
        wall_hash = "sha256:" + hashlib.sha256(canonical_bytes(list(record.wall_tiles))).hexdigest()
    decoded_hash = record.raw_bytes_sha256
    return GameIdentity(
        game_id=record.game_id,
        object_id=record.object_id,
        source_id=source,
        player_ids=player_ids,
        timestamp=timestamp,
        wall_hash=wall_hash,
        decoded_hash=decoded_hash,
    )


def detect_exact_duplicates(games: list[GameIdentity]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    dups: list[tuple[str, str]] = []
    for g in games:
        h = g.decoded_hash
        if h in seen:
            dups.append((seen[h], g.game_id))
        else:
            seen[h] = g.game_id
    return dups


def detect_near_duplicates(games: list[GameIdentity]) -> list[tuple[str, str]]:
    # Near duplicate: same wall hash (identical wall)
    wall_map: dict[str, str] = {}
    dups: list[tuple[str, str]] = []
    for g in games:
        if g.wall_hash is None:
            continue
        if g.wall_hash in wall_map:
            dups.append((wall_map[g.wall_hash], g.game_id))
        else:
            wall_map[g.wall_hash] = g.game_id
    return dups


def assign_partitions(
    *,
    game_records: list[GameRecord],
    acquisition_by_object: dict[str, dict[str, object]],
    spec: SplitSpec,
) -> SplitManifest:
    """Assign whole games to partitions; enforces grouping and duplicate checks."""
    if len(game_records) == 0:
        raise ContractError("no games to partition")
    identities: list[GameIdentity] = []
    for rec in game_records:
        meta = acquisition_by_object.get(rec.object_id, {})
        identities.append(_game_identity_for(rec, meta))
    # Detect duplicates: they cannot cross partition
    exact_dups = detect_exact_duplicates(identities)
    if len(exact_dups) > 0:
        raise ContractError(f"exact duplicates detected: {exact_dups[:3]}")
    near_dups = detect_near_duplicates(identities)
    if len(near_dups) > 0:
        raise ContractError(f"near duplicates detected: {near_dups[:3]}")

    # Grouping when metadata permits, group by source/player/time
    groups: dict[str, list[GameIdentity]] = {}
    for ident in identities:
        key_parts: list[str] = []
        for gk in spec.grouping_keys:
            if gk == "source":
                key_parts.append(ident.source_id)
            elif gk == "player":
                key_parts.append("|".join(sorted(ident.player_ids)))
            elif gk == "time":
                ts_val: str = ident.timestamp if ident.timestamp is not None else "unknown"
                key_parts.append(ts_val[:10])
            else:
                key_parts.append("unknown")
        group_key = "|".join(key_parts) if len(key_parts) > 0 else ident.game_id
        groups.setdefault(group_key, []).append(ident)

    total = sum(spec.ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ContractError(f"ratios must sum to 1.0, got {total}")
    partition_order: list[Partition] = [
        "train",
        "validation",
        "test",
        "decision_eval",
        "block_eval",
    ]
    active_parts = [p for p in partition_order if p in spec.ratios and spec.ratios[p] > 0]
    if len(active_parts) == 0:
        raise ContractError("no active partitions in ratios")

    assignments: dict[str, Partition] = {}

    cumulative: list[tuple[Partition, float]] = []
    cum = 0.0
    for p in active_parts:
        cum += spec.ratios[p]
        cumulative.append((p, cum))
    for group_key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        h_bytes = hashlib.sha256(f"{spec.seed}|{group_key}".encode()).digest()
        h_val = int.from_bytes(h_bytes[:8], "big") / (2**64)
        chosen: Partition = active_parts[-1]
        for part, thresh in cumulative:
            if h_val < thresh:
                chosen = part
                break
        for ident in members:
            assignments[ident.game_id] = chosen

    if len(assignments) != len(game_records):
        raise ContractError("partition count mismatch: game split detected")
    if spec.wall_disjoint:
        wall_to_part: dict[str, Partition] = {}
        for ident in identities:
            if ident.wall_hash is None:
                continue
            part = assignments[ident.game_id]
            if ident.wall_hash in wall_to_part and wall_to_part[ident.wall_hash] != part:
                raise ContractError(f"wall {ident.wall_hash[:12]} appears in multiple partitions")
            wall_to_part[ident.wall_hash] = part
        eval_walls: set[str] = {
            wh
            for ident in identities
            if assignments[ident.game_id] in ("block_eval", "decision_eval")
            and (wh := ident.wall_hash) is not None
        }
        train_walls: set[str] = {
            wh
            for ident in identities
            if assignments[ident.game_id] == "train" and (wh := ident.wall_hash) is not None
        }
        if len(eval_walls & train_walls) > 0:
            raise ContractError("rollout/evaluation walls disjoint violation")

    input_hashes = {
        "spec": "sha256:"
        + hashlib.sha256(
            canonical_bytes(
                {
                    "algorithm": spec.algorithm,
                    "version": spec.version,
                    "seed": spec.seed,
                    "ratios": spec.ratios,
                    "grouping_keys": list(spec.grouping_keys),
                    "wall_disjoint": spec.wall_disjoint,
                }
            )
        ).hexdigest(),
        "games": "sha256:"
        + hashlib.sha256(canonical_bytes(sorted([g.decoded_hash for g in identities]))).hexdigest(),
    }
    manifest_payload = {
        "spec": {
            "algorithm": spec.algorithm,
            "version": spec.version,
            "seed": spec.seed,
            "ratios": spec.ratios,
            "grouping_keys": list(spec.grouping_keys),
            "wall_disjoint": spec.wall_disjoint,
        },
        "assignments": assignments,
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(manifest_payload)).hexdigest()
    return SplitManifest(
        spec=spec, assignments=assignments, input_hashes=input_hashes, digest=digest
    )


def write_split_manifest(destination: Path, manifest: SplitManifest) -> str:
    payload = {
        "schema_version": "1.0.0",
        "spec": {
            "algorithm": manifest.spec.algorithm,
            "version": manifest.spec.version,
            "seed": manifest.spec.seed,
            "ratios": manifest.spec.ratios,
            "grouping_keys": list(manifest.spec.grouping_keys),
            "wall_disjoint": manifest.spec.wall_disjoint,
        },
        "assignments": manifest.assignments,
        "input_hashes": manifest.input_hashes,
        "digest": manifest.digest,
    }
    atomic_replace_bytes(destination, canonical_bytes(payload))
    return manifest.digest
