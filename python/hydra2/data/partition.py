"""Partition whole games before expansion — checklist item 6.

Hard dependency (shrink end-state): :func:`assign_partitions` is a thin
delegate over the Rust bridge (``hydra2._native.packet_decode``
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
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from collections.abc import Callable
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


def _require_packet() -> Any:
    """Import the built ``packet`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with packet not importable; "
            "build the bridge with `pixi run build-ext` before partitioning games"
        ) from exc
    try:
        return _ext.packet
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.packet submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _require_packet_decode() -> Any:
    """Import the built ``packet_decode`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with packet_decode not importable; "
            "build the bridge with `pixi run build-ext` before partitioning games"
        ) from exc
    try:
        return _ext.packet_decode
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.packet_decode submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _require_columnar() -> Any:
    """Import the built ``columnar`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with columnar not importable; "
            "build the bridge with `pixi run build-ext` before scanning partition duplicates"
        ) from exc
    try:
        return _ext.columnar
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.columnar submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _game_identity_for(record: GameRecord, acquisition_metadata: dict[str, object]) -> GameIdentity:
    """Thin framer: metadata passthrough + hard-Rust wall hash.

    Grouping/draw math lives in the bridge; the wall hash is minted via
    the ``packet`` bridge (``packet.wall_hash`` over the 136-entry wall
    list; byte-identical to the retired ``of_canonical`` oracle, 4.42x
    faster; ``None`` walls stay ``None``). Non-136 walls fail closed via
    :class:`ContractError` (bridge ``ValueError`` mapped, same contract
    as :func:`assign_partitions`).
    """
    source = str(acquisition_metadata.get("source", "unknown"))
    _pids_raw: object = acquisition_metadata.get("player_ids", [])
    if isinstance(_pids_raw, (list, tuple)):
        raw_ids: list[object] = list(_pids_raw)
        player_ids = tuple(str(x) for x in raw_ids)
    else:
        player_ids = ()
    if len(player_ids) == 0:
        player_ids = ("unknown",)
    ts = acquisition_metadata.get("timestamp")
    timestamp = ts if isinstance(ts, str) else None
    wall_hash: str | None = None
    if record.wall_tiles is not None:
        packet = _require_packet()
        try:
            wall_text: str = packet.wall_hash(record.wall_tiles)
            wall_hash = wall_text
        except ValueError as exc:
            raise ContractError(f"wall hash rejected for {record.game_id}: {exc}") from exc
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
    """First-seen-wins scan on decoded hashes.

    Thin bridge delegate: ``hydra2._native.columnar.partition_detect_exact_duplicates``
    groups detached over plain ``(game_id, decoded_hash)`` pairs through the
    feed-owned ``hydra_feed::partition::detect_exact_duplicates``; this function keeps
    the ``GameIdentity`` staging and the ``__all__`` name.
    """
    columnar = _require_columnar()
    try:
        exact_bridge: Callable[[list[tuple[str, str]]], list[tuple[str, str]]] = (
            columnar.partition_detect_exact_duplicates
        )
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.columnar.partition_detect_exact_duplicates missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    pairs: list[tuple[str, str]] = [(g.game_id, g.decoded_hash) for g in games]
    out: list[tuple[str, str]] = exact_bridge(pairs)
    return [(first, second) for first, second in out]


def detect_near_duplicates(games: list[GameIdentity]) -> list[tuple[str, str]]:
    """Wall-hash scan, wall-less games skipped.

    Thin bridge delegate: ``hydra2._native.columnar.partition_detect_near_duplicates``
    scans detached over plain ``(game_id, wall_hash_or_None)`` pairs through the
    feed-owned ``hydra_feed::partition::detect_near_duplicates``; this function keeps
    the ``GameIdentity`` staging and the ``__all__`` name.
    """
    columnar = _require_columnar()
    try:
        near_bridge: Callable[[list[tuple[str, str | None]]], list[tuple[str, str]]] = (
            columnar.partition_detect_near_duplicates
        )
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.columnar.partition_detect_near_duplicates missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    near_pairs: list[tuple[str, str | None]] = [(g.game_id, g.wall_hash) for g in games]
    near_out: list[tuple[str, str]] = near_bridge(near_pairs)
    return [(first, second) for first, second in near_out]


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
        out: dict[str, Any] = packet_decode.assign_partitions(games, spec_doc)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    raw_assignments: object = out["assignments"]
    assignments_src: dict[object, object] = cast("dict[object, object]", raw_assignments)
    assignments = cast(
        "dict[str, Partition]",
        {str(k): str(v) for k, v in assignments_src.items()},
    )
    raw_hashes: object = out["input_hashes"]
    hashes_src: dict[object, object] = cast("dict[object, object]", raw_hashes)
    input_hashes = {str(k): str(v) for k, v in hashes_src.items()}
    raw_digest: object = out["digest"]
    return SplitManifest(
        spec=spec, assignments=assignments, input_hashes=input_hashes, digest=str(raw_digest)
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
