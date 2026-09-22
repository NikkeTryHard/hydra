"""Duplicate wall identity helpers (exact and near duplicates).

Single home for `WALL_TILES`, wall-id and digest gates, the single bridge
wall digest, `wall_hash_from_tiles`, `wall_fingerprint`,
`find_exact_duplicates`, and `find_near_duplicates`. Bridge digests decide;
Python keeps the 136-int gates. Failure mode is fail-closed `ContractError`
on bad walls, bare `ValueError` on malformed digests, never silent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import eval as _bridge_eval  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import validate_digest as validate_digest
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import DigestText as DigestText

#: Physical wall length, tiles 0..135 (bridge-owned; fallback keeps pre-wire imports green).
WALL_TILES: int = (
    int(_bridge_eval.WALL_TILES)  # type: ignore[attr-defined]  # reason: eval_leaves2 leaf lands with MAIN wiring; hasattr fallback covers stale .so
    if hasattr(_bridge_eval, "WALL_TILES")
    else 136
)


# ---------------------------------------------------------------------------
# Wall identity helpers
# ---------------------------------------------------------------------------


def _require_wall_id(wall_id: object) -> str:
    if not isinstance(wall_id, str) or wall_id == "":
        raise ContractError(f"wall_id must be nonempty str, got {wall_id!r}")
    return wall_id


def _require_digest_value(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{field_name} must be nonempty digest str")
    return str(validate_digest(value))


def _bridge_wall_digest(wall: list[int], *, subject: str) -> DigestText:
    """Single bridge wall digest (Python validation stays, fail-closed).

    ``hydra2._native.packet.wall_hash`` decides: ``sha256:`` over the canon
    bytes of the 136-entry wall list. Evidence: byte-identical to the retired
    double-compute (``of_canonical`` + feed recompute + compare) on identity,
    shuffled, and sorted walls, 5.6x faster. The bridge validates shape only,
    so the 136-int/bool-excluded ContractError gate above stays Python. A
    missing bridge raises ImportError with a ``build-ext`` hint; any other
    bridge failure raises ContractError — never silent, never a default.
    """
    try:
        import importlib as _importlib

        _packet = _importlib.import_module("hydra2._native").packet
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "hydra2._native extension with packet not importable; "
            "rebuild the bridge with `pixi run build-ext` before hashing "
            f"{subject}"
        ) from exc
    try:
        wall_hash: str | None = _packet.wall_hash(wall)
        return _bridge_contracts.make_digest_text(str(wall_hash))
    except Exception as exc:
        raise ContractError(f"{subject}: bridge wall digest failed: {exc}") from exc


def wall_hash_from_tiles(wall_tiles: Sequence[int]) -> DigestText:
    """Digest of a 136-length wall (physical tile ids 0..135).

    The digest is over the canonical bytes of the tile list; any reordering
    changes the hash. Use :func:`wall_fingerprint` for a permutation-insensitive
    near-duplicate check.
    Bridge-first via :func:`_bridge_wall_digest` (``packet.wall_hash`` decides;
    the 136-int ContractError gate above stays Python because the bridge only
    checks shape). B2 held:
    no split permutation lives here — splits stay the torch.randperm oracle.
    """
    if len(wall_tiles) != WALL_TILES:
        raise ContractError(f"wall must have {WALL_TILES} tiles, got {len(wall_tiles)}")
    for tile in wall_tiles:
        if not isinstance(tile, int) or isinstance(tile, bool) or not 0 <= tile < WALL_TILES:
            raise ContractError(f"tile ids must be int in [0,{WALL_TILES}), got {tile!r}")
    tiles = list(wall_tiles)
    return _bridge_wall_digest(tiles, subject="wall_hash_from_tiles")


def wall_fingerprint(wall_tiles: Sequence[int]) -> DigestText:
    """Logical fingerprint: digest over the sorted tile multiset.

    Two walls with identical tile multisets in different dealing orders share
    the same fingerprint (near duplicate). Exact duplicates require
    byte-identical wall order and are caught by :func:`wall_hash_from_tiles`.
    Bridge-first via :func:`_bridge_wall_digest` (``packet.wall_hash`` decides
    over the sorted multiset; the 136-int ContractError gate above stays
    Python because the bridge only checks shape). B2 held:
    no split permutation lives here — splits stay the torch.randperm oracle.
    """
    if len(wall_tiles) != WALL_TILES:
        raise ContractError(f"wall must have {WALL_TILES} tiles, got {len(wall_tiles)}")
    for tile in wall_tiles:
        if not isinstance(tile, int) or isinstance(tile, bool) or not 0 <= tile < WALL_TILES:
            raise ContractError(f"tile ids must be int in [0,{WALL_TILES}), got {tile!r}")
    sorted_tiles = sorted(wall_tiles)
    return _bridge_wall_digest(sorted_tiles, subject="wall_fingerprint")


def find_exact_duplicates(wall_hashes: Mapping[str, str]) -> list[tuple[str, str]]:
    """Detect exact duplicates: two distinct wall ids sharing identical digest.

    ``wall_hashes`` maps wall id -> wall digest (``sha256:…``). Returns a list
    of duplicate pairs ``(first_id, second_id)`` in discovery order.

    Thin bridge delegate: ``hydra2._native.contracts.eval_find_exact_duplicates``
    stages the normalised dict and groups detached; this function keeps the
    ``Mapping`` gate, the ``ContractError`` shaping, and the ``__all__`` name.
    Malformed digests re-raise the bridge ``ValueError`` bare (oracle
    ``validate_digest`` passthrough, never ``ContractError``).
    """
    if not isinstance(wall_hashes, Mapping):
        raise ContractError("wall_hashes must be a mapping wall_id -> digest")
    payload = wall_hashes if type(wall_hashes) is dict else dict(wall_hashes)
    try:
        out: list[tuple[str, str]] = _bridge_contracts.eval_find_exact_duplicates(payload)
    except (ValueError, TypeError) as exc:
        if str(exc) == "contracts digest_text must match sha256:<64 lowercase hex>":
            raise
        raise ContractError(str(exc)) from exc
    return [(first, second) for first, second in out]


def find_near_duplicates(
    wall_tiles_by_id: Mapping[str, Sequence[int]],
) -> list[tuple[str, str]]:
    """Detect near duplicates: walls sharing the logical fingerprint.

    Walls with the same sorted tile multiset are near duplicates even when
    their dealing order differs. Input maps wall id -> 136 tile ids. Returns
    duplicate pairs in discovery order.
    """
    if not isinstance(wall_tiles_by_id, Mapping):
        raise ContractError("wall_tiles_by_id must be mapping wall_id -> tiles")
    fingerprint_to_first: dict[str, str] = {}
    dups: list[tuple[str, str]] = []
    for wall_id, tiles in wall_tiles_by_id.items():
        _ = _require_wall_id(wall_id)
        if not isinstance(tiles, Sequence):
            raise ContractError(f"tiles for {wall_id!r} must be a sequence")
        fingerprint = str(wall_fingerprint(tiles))
        if fingerprint in fingerprint_to_first:
            dups.append((fingerprint_to_first[fingerprint], wall_id))
        else:
            fingerprint_to_first[fingerprint] = wall_id
    return dups
