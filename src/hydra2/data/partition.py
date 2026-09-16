"""Partition whole games before expansion — checklist item 6.

Hard dependency (shrink end-state): :func:`assign_partitions` is a thin
delegate over the Rust bridge (``hydra2_replay_rs.packet_decode``
``assign_partitions``; ``hydra-feed`` partition owns grouping, the
``u64BE / 2**64`` split draw on ``sha256(f"{seed}|{group}")``, duplicate
rejection, wall-disjoint checks, and the canon-bound digest).
``ImportError`` (extension not built) raises with a ``build-ext`` hint —
NO oracle fallback, never silent. Evidence: packet 283/283 + seals
migrated (B3). B2 held: the split draw stays the feed-owned sha draw
(held-out splits stay on the ``torch.randperm`` oracle behind KAT; no
``feed::shuffle`` edge here).

Requirements:
  - Assign complete games before decisions (never split a game)
  - Enforce source/player/time grouping when metadata permits
  - Reject exact and near duplicates across partitions
  - Keep rollout/evaluation walls disjoint
  - Split manifest stores algorithm/version/seed/input hashes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical
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


def _require_packet_decode() -> Any:
    """Import the built ``packet_decode`` bridge surface (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with packet_decode not importable; "
            "build the bridge with `pixi run build-ext` before partitioning games"
        ) from exc
    try:
        return _ext.packet_decode
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.packet_decode submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _game_identity_for(record: GameRecord, acquisition_metadata: dict[str, object]) -> GameIdentity:
    """Thin framer: metadata passthrough + hard-Rust wall hash.

    Grouping/draw math lives in the bridge; the wall hash is minted via
    the hard-Rust digest owner (same ``sha256`` over canon wall bytes the
    bridge ``wall_hash`` computes; ``None`` walls stay ``None``).
    """
    source = str(acquisition_metadata.get("source", "unknown"))
    _pids_raw: object = acquisition_metadata.get("player_ids", [])
    if isinstance(_pids_raw, (list, tuple)):
        player_ids = tuple(str(x) for x in list(_pids_raw))
    else:
        player_ids = ()
    if len(player_ids) == 0:
        player_ids = ("unknown",)
    ts = acquisition_metadata.get("timestamp")
    timestamp = ts if isinstance(ts, str) else None
    wall_hash = None
    if record.wall_tiles is not None:
        wall_hash = str(of_canonical(list(record.wall_tiles)))
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
    """First-seen-wins scan on decoded hashes (kept: no bridge pyfn)."""
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
    """Wall-hash scan, wall-less games skipped (kept: no bridge pyfn)."""
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
    """Assign whole games to partitions via the Rust bridge (fail closed).

    Identity framing stays here; grouping, draws, duplicate/wall-disjoint
    enforcement, and the digest live in ``hydra-feed`` partition. Bridge
    data-shape rejects surface as :class:`ContractError`.
    """
    if len(game_records) == 0:
        raise ContractError("no games to partition")
    packet_decode = _require_packet_decode()
    games: list[dict[str, object]] = []
    for rec in game_records:
        meta = acquisition_by_object.get(rec.object_id, {})
        ident = _game_identity_for(rec, meta)
        games.append(
            {
                "game_id": ident.game_id,
                "object_id": ident.object_id,
                "source_id": ident.source_id,
                "player_ids": list(ident.player_ids),
                "timestamp": ident.timestamp,
                "wall_hash": ident.wall_hash,
                "decoded_hash": ident.decoded_hash,
            }
        )
    spec_doc: dict[str, object] = {
        "algorithm": spec.algorithm,
        "version": spec.version,
        "seed": spec.seed,
        "ratios": dict(spec.ratios),
        "grouping_keys": list(spec.grouping_keys),
        "wall_disjoint": spec.wall_disjoint,
    }
    try:
        out = packet_decode.assign_partitions(games, spec_doc)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    assignments = cast(
        "dict[str, Partition]",
        {str(k): str(v) for k, v in dict(out["assignments"]).items()},
    )
    input_hashes = {str(k): str(v) for k, v in dict(out["input_hashes"]).items()}
    return SplitManifest(
        spec=spec, assignments=assignments, input_hashes=input_hashes, digest=str(out["digest"])
    )


def write_split_manifest(destination: Path, manifest: SplitManifest) -> str:
    """Frame the bridge-minted manifest via the canon authority (kept)."""
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
