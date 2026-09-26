"""Stream scan cache: envelope, matchers, load/save (stdlib only).

Split from ``stream_manifest`` (LOC gate): the scan-cache envelope plus
its exact-match helpers live here so the manifest module keeps the
vocabulary, records, build, and digest. ``stream_manifest`` re-exports
every public name, so all existing import sites keep working unchanged.
Cache stays advisory: any miss or corruption fails closed to a full scan.

Owns the scan-plane vocabulary shared by the reader and the training
driver: the cache envelope, the exact-match helpers, and the load/save
pair. Anything that must stay byte-identical across scan caches lives
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "SCAN_CACHE_VERSION",
    "_holdout_match",
    "_ratios_match",
    "_root_games_match",
    "_roots_match",
    "load_scan_cache",
    "save_scan_cache",
    "scan_cache_path",
]


#: Scan-cache envelope version (bump on schema change; mismatch → miss).
#: v2 adds the `roots` + `holdout` + `root_games` keys; v1 envelopes miss,
#: never coerce (a v1 cache predates namespaced digests, so its pin could
#: never match a v2 digest anyway).
SCAN_CACHE_VERSION = 2


def _roots_match(cached: object, expected: Sequence[tuple[str, str]]) -> bool:
    """Exact root-list comparison (order matters; fail-closed to miss).

    JSON round-trips pairs as 2-lists; tuples coerce element-wise. A changed
    root list changes the manifest digest already, so this is belt-and-braces
    provenance (the envelope names its corpus even when the digest is read
    out of band).
    """
    if not isinstance(cached, list):
        return False
    want = [[rid, base] for rid, base in expected]
    if len(cached) != len(want):
        return False
    for have, need in zip(cached, want, strict=True):
        if not isinstance(have, list) or have != need:
            return False
    return True


def _holdout_match(cached: object, expected: Mapping[str, object]) -> bool:
    """Exact holdout-spec comparison (fail-closed to miss)."""
    if not isinstance(cached, dict):
        return False
    return dict(cached) == dict(expected)


def _root_games_match(cached: object) -> list[list[object]] | None:
    """Validate the cached per-root game tallies, else ``None`` (miss).

    Shape is ``[[root-id, train-games, val-games], ...]`` with true ints;
    bool excluded (bool is an int subclass). Counts re-derive from the scan
    on a miss, never defaulted.
    """
    if not isinstance(cached, list):
        return None
    out: list[list[object]] = []
    for row in cached:
        if not isinstance(row, list) or len(row) != 3:
            return None
        rid, train_games, val_games = row
        if not isinstance(rid, str) or rid == "":
            return None
        for value in (train_games, val_games):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
        out.append([rid, train_games, val_games])
    return out


def scan_cache_path(run_dir: Path | str) -> Path:
    """Cache file for the manifest/split pre-scan under ``run_dir``."""
    return Path(run_dir) / "cache" / "scan-cache.json"


def _ratios_match(cached: object, expected: Mapping[str, float]) -> bool:
    """Exact-ish ratio comparison (JSON round-trip safe, fail-closed to miss).

    1e-9 tolerates JSON round-trip without admitting a different split;
    bool is excluded because bool is an int subclass (True == 1 would pass
    a count check).
    """
    if not isinstance(cached, dict):
        return False
    if set(cached) != set(expected):
        return False
    for key, value in expected.items():
        raw = cached.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return False
        try:
            if abs(float(raw) - value) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def load_scan_cache(
    path: Path | str,
    *,
    manifest_digest: str,
    seed: int,
    ratios: Mapping[str, float],
    train_split: str,
    val_split: str,
    roots: Sequence[tuple[str, str]],
    holdout: Mapping[str, object],
) -> dict[str, object] | None:
    """Load a cached scan report on exact key match, else ``None`` (miss).
    Corrupt/stale entries fail closed to miss (full scan), never raise.
    """
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        raw: object = json.loads(raw_text)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        version: object = raw.get("version", -1)
        if version != SCAN_CACHE_VERSION:
            return None
        if raw.get("manifest_digest") != manifest_digest:
            return None
        if raw.get("data_seed") != seed:
            return None
        if raw.get("train_split") != train_split or raw.get("val_split") != val_split:
            return None
        if not _ratios_match(raw.get("ratios"), ratios):
            return None
        if not _roots_match(raw.get("roots"), roots):
            return None
        if not _holdout_match(raw.get("holdout"), holdout):
            return None
        scan = raw.get("scan")
        if not isinstance(scan, dict):
            return None
        train_walls = scan.get("train_walls")
        val_walls = scan.get("val_walls")
        if not isinstance(train_walls, list) or not isinstance(val_walls, list):
            return None
        if any(not isinstance(w, str) for w in train_walls):
            return None
        if any(not isinstance(w, str) for w in val_walls):
            return None
        # Re-checks disjointness on load: a cache written before a wall fix
        # must not resurrect cross-split walls — fail closed to miss.
        if len(set(train_walls) & set(val_walls)) > 0:
            return None
        counts: dict[str, object] = {}
        for key in (
            "train_games",
            "val_games",
            "train_sim_games",
            "val_sim_games",
            "framed",
            "emitted",
            "quarantined",
            "duplicates",
        ):
            value = scan.get(key)
            # bool excluded (isinstance(True, int)): counts are true ints.
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            counts[key] = value
        root_games = _root_games_match(scan.get("root_games"))
        if root_games is None:
            return None
        return {
            "train_walls": sorted(train_walls),
            "val_walls": sorted(val_walls),
            "root_games": root_games,
            **counts,
        }
    except (TypeError, ValueError, AttributeError):
        return None


def save_scan_cache(
    path: Path | str,
    *,
    manifest_digest: str,
    seed: int,
    ratios: Mapping[str, float],
    train_split: str,
    val_split: str,
    roots: Sequence[tuple[str, str]],
    holdout: Mapping[str, object],
    scan: Mapping[str, object],
) -> None:
    """Best-effort atomic cache write (never fails training on I/O error).

    tmp+replace publishes atomically: a crash leaves old or new, never
    torn. Sorted walls make the payload deterministic; every failure
    returns silently because cache is advisory.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCAN_CACHE_VERSION,
            "manifest_digest": manifest_digest,
            "data_seed": seed,
            "ratios": dict(ratios),
            "train_split": train_split,
            "val_split": val_split,
            "roots": [[rid, base] for rid, base in roots],
            "holdout": dict(holdout),
            "scan": {
                # reason: type arg-type on scan payload object; sorted/int
                # re-validated by isinstance guards on load, fail-closed miss.
                "train_walls": sorted(scan["train_walls"]),  # type: ignore[arg-type]
                "val_walls": sorted(scan["val_walls"]),  # type: ignore[arg-type]
                "root_games": [list(row) for row in scan["root_games"]],  # type: ignore[arg-type]
                "train_games": int(scan["train_games"]),  # type: ignore[arg-type]
                "val_games": int(scan["val_games"]),  # type: ignore[arg-type]
                "train_sim_games": int(scan["train_sim_games"]),  # type: ignore[arg-type]
                "val_sim_games": int(scan["val_sim_games"]),  # type: ignore[arg-type]
                "framed": int(scan["framed"]),  # type: ignore[arg-type]
                "emitted": int(scan["emitted"]),  # type: ignore[arg-type]
                "quarantined": int(scan["quarantined"]),  # type: ignore[arg-type]
                "duplicates": int(scan["duplicates"]),  # type: ignore[arg-type]
            },
        }
        tmp = target.with_suffix(".tmp")
        # Atomic cache write/publish is the effect; counts/paths discarded.
        _ = tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _ = tmp.replace(target)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return
