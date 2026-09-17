"""Pre-train corpus scan: wall-disjointness plus the eval ledger.

Owns the ordered serial scan, its bit-identical process-parallel twin, and
the content-hash scan cache (run-dir copy authoritative; shared copy keyed
on the same manifest digest/seed/ratios plus per-file fingerprints). Both
paths report the identical wall sets and quarantine counters, and any wall
shared across the two splits fails closed before training state exists.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context as _mp_get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.artifacts.digest import sha256_digest as sha256_digest
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import CorruptArtifactError as CorruptArtifactError
from hydra2.data.decode import decode_game_object as decode_game_object
from hydra2.data.stream import GameStream as GameStream
from hydra2.data.stream import ZstdLineStream as ZstdLineStream
from hydra2.data.stream import assign_split as assign_split
from hydra2.data.stream import compute_wall_hash as compute_wall_hash
from hydra2.data.stream import group_key_for_path as group_key_for_path
from hydra2.data.stream import load_scan_cache as load_scan_cache
from hydra2.data.stream import save_scan_cache as save_scan_cache
from hydra2.data.stream import scan_cache_path as scan_cache_path
from hydra2.data.stream import stem_of as stem_of
from hydra2.data.validate import validate_game as validate_game
from hydra2.training.stream_expand import _pool_worker_init as _pool_worker_init

if TYPE_CHECKING:
    from hydra2.data.stream import StreamManifest as StreamManifest
    from hydra2.training._rc_sections import RunConfig as RunConfig

__all__ = [
    "_ScanReport",
    "_scan_corpus",
    "_scan_corpus_cached",
    "_scan_corpus_parallel",
    "_scan_file_games",
    "_scan_report",
    "_shared_scan_cache_path",
]


@dataclass(slots=True)
class _ScanReport:
    """Pre-train corpus scan: split walls, eval ledger, quarantine counts."""

    train_walls: frozenset[str]
    val_walls: frozenset[str]
    train_games: int
    val_games: int
    train_sim_games: int
    val_sim_games: int
    framed: int
    emitted: int
    quarantined: int
    duplicates: int


def _scan_file_games(path_str: str) -> list[tuple[str, str | None, bool] | None]:
    """Decode+validate every game in one file (scan-worker pure function).

    Returns per game, in file order: ``None`` when undecodable/invalid,
    else ``(raw_bytes_sha256, wall_hash_or_None, wall_tiles_is_None)``.
    Mirrors :meth:`GameStream._decode_inline` failure classes exactly;
    file-level I/O errors propagate like the serial scan (fail closed).
    """
    from pathlib import Path

    out: list[tuple[str, str | None, bool] | None] = []
    for _, game_bytes in ZstdLineStream(path_str).iter_games():
        try:
            game = decode_game_object(
                object_id=stem_of(Path(path_str)),
                packaged_object_id=stem_of(Path(path_str)),
                decoded_bytes=game_bytes,
            )
        except (ContractError, CorruptArtifactError, ValueError):
            out.append(None)
            continue
        try:
            _ = validate_game(game)  # intentionally discarded: raises on invalid, outcome unneeded
        except (ContractError, CorruptArtifactError, ValueError):
            out.append(None)
            continue
        out.append((game.raw_bytes_sha256, compute_wall_hash(game), game.wall_tiles is None))
    return out


def _scan_corpus(
    manifest: StreamManifest, *, config: RunConfig, ratios: dict[str, float]
) -> _ScanReport:
    """One ordered pass: wall sets per split, eval ledger, quarantine counts.

    Uses ``split=None`` so every valid game is emitted with its assignment
    attached, then partitions by the config split names. Raises before any
    training state exists when one wall lands in both splits.
    """
    if config.data.num_workers > 0:
        return _scan_corpus_parallel(manifest, config=config, ratios=ratios)
    common: dict[str, Any] = {
        "seed": config.seeds.data_seed,
        "ratios": ratios,
        "epoch": 0,
        "split": None,
        "shuffle_buffer": 0,
    }
    stream = GameStream(manifest, **common)
    train_walls: set[str] = set()
    val_walls: set[str] = set()
    train_games = 0
    val_games = 0
    train_sim_games = 0
    val_sim_games = 0
    for game in stream:
        if game.split == config.data.train_split:
            train_games += 1
            if game.wall_hash is not None:
                train_walls.add(game.wall_hash)
            if game.game.wall_tiles is None:
                train_sim_games += 1
        elif game.split == config.data.val_split:
            val_games += 1
            if game.wall_hash is not None:
                val_walls.add(game.wall_hash)
            if game.game.wall_tiles is None:
                val_sim_games += 1
    stats = stream.stats
    return _scan_report(
        config,
        train_walls,
        val_walls,
        train_games,
        val_games,
        train_sim_games,
        val_sim_games,
        stats.framed,
        stats.emitted,
        stats.quarantined,
        stats.duplicates,
    )


def _scan_report(
    config: RunConfig,
    train_walls: set[str],
    val_walls: set[str],
    train_games: int,
    val_games: int,
    train_sim_games: int,
    val_sim_games: int,
    framed: int,
    emitted: int,
    quarantined: int,
    duplicates: int,
) -> _ScanReport:
    """Shared scan tail: wall-disjoint gate plus the report record."""
    overlap = sorted(train_walls & val_walls)
    if len(overlap) > 0:
        raise ContractError(
            f"wall {overlap[0][:16]} in splits "
            f"{config.data.train_split} and {config.data.val_split}"
        )
    return _ScanReport(
        train_walls=frozenset(train_walls),
        val_walls=frozenset(val_walls),
        train_games=train_games,
        val_games=val_games,
        train_sim_games=train_sim_games,
        val_sim_games=val_sim_games,
        framed=framed,
        emitted=emitted,
        quarantined=quarantined,
        duplicates=duplicates,
    )


def _scan_corpus_parallel(
    manifest: StreamManifest, *, config: RunConfig, ratios: dict[str, float]
) -> _ScanReport:
    """Process-parallel scan; bit-identical to the serial pass by construction.

    Decode+validate fan out over files (pure per-file work); the merge below
    replays :meth:`GameStream._finish_decode` accounting line-for-line in
    manifest order (``pool.map`` preserves order, so exact-hash dedup keeps
    first-seen-wins). Split assignment is pure per game
    (:func:`assign_split`), so parallel decode cannot change partitions.
    """
    paths = [str(entry.path) for entry in manifest.files]
    keys = [group_key_for_path(entry.path) for entry in manifest.files]
    train_walls: set[str] = set()
    val_walls: set[str] = set()
    train_games = 0
    val_games = 0
    train_sim_games = 0
    val_sim_games = 0
    framed = 0
    emitted = 0
    quarantined = 0
    duplicates = 0
    hashes: set[str] = set()
    # Spawn (never fork): the parent has torch OMP threads alive and
    # fork-with-threads deadlocks racily; spawn re-imports clean workers.
    # Built once per run (the cached scan runs a single pass before any
    # training state exists — never per epoch); workers start single-threaded
    # under _pool_worker_init (see the joint budget at
    # _PARALLEL_EXPAND_MAX_WORKERS).
    ctx = _mp_get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.data.num_workers, mp_context=ctx, initializer=_pool_worker_init
    ) as pool:
        for file_index, games in enumerate(pool.map(_scan_file_games, paths, chunksize=32)):
            group_key = keys[file_index]
            for item in games:
                framed += 1
                if item is None:
                    quarantined += 1
                    continue
                raw_sha, wall_hash, is_sim = item
                if raw_sha in hashes:
                    quarantined += 1
                    duplicates += 1
                    continue
                hashes.add(raw_sha)
                assigned = assign_split(
                    group_key=group_key, seed=config.seeds.data_seed, ratios=ratios
                )
                emitted += 1
                if assigned == config.data.train_split:
                    train_games += 1
                    if wall_hash is not None:
                        train_walls.add(wall_hash)
                    if is_sim:
                        train_sim_games += 1
                elif assigned == config.data.val_split:
                    val_games += 1
                    if wall_hash is not None:
                        val_walls.add(wall_hash)
                    if is_sim:
                        val_sim_games += 1
    return _scan_report(
        config,
        train_walls,
        val_walls,
        train_games,
        val_games,
        train_sim_games,
        val_sim_games,
        framed,
        emitted,
        quarantined,
        duplicates,
    )


def _shared_scan_cache_path(
    *,
    manifest: StreamManifest,
    stream_digest: str,
    seed: int,
    ratios: dict[str, float],
    train_split: str,
    val_split: str,
) -> Path:
    """Machine-local scan-cache path shared across run dirs (same key family).

    The run-dir copy stays authoritative for audit/resume; this path only
    avoids re-scanning an immutable corpus on every fresh run_dir. Keyed by
    the identical fields ``load_scan_cache`` re-validates plus a per-file
    ``(path, size, mtime_ns)`` fingerprint, so any add/remove/resize/touch
    misses to a full scan (same staleness semantics as ``make``). Override
    the directory with ``HYDRA2_SCAN_CACHE_DIR`` (never inside a data root).
    """
    from hydra2.data.stream import SCAN_CACHE_VERSION

    base_raw = os.environ.get("HYDRA2_SCAN_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "hydra2", "scan"
    )
    stamps: list[bytes] = []
    for entry in manifest.files:
        try:
            fingerprint_stat = entry.path.stat()
            stamp = f"{fingerprint_stat.st_size}:{fingerprint_stat.st_mtime_ns}"
        except OSError:
            stamp = "missing"
        stamps.append(f"{entry.path.as_posix()}:{stamp}\n".encode())
    # Same bytes the retired incremental ``hashlib.sha256`` covered; the
    # digest is minted by the hard-Rust digest owner (fail closed with a
    # ``build-ext`` hint when the extension is not built).
    files_hex = str(sha256_digest(b"".join(stamps))).removeprefix("sha256:")
    fingerprint = repr(
        (SCAN_CACHE_VERSION, stream_digest, seed, sorted(ratios.items()), train_split, val_split)
    )
    key = str(sha256_digest((fingerprint + files_hex).encode())).removeprefix("sha256:")[:32]
    return Path(base_raw) / f"scan-{key}.json"


def _scan_corpus_cached(
    manifest: StreamManifest,
    *,
    config: RunConfig,
    ratios: dict[str, float],
    stream_digest: str,
    run_dir: Path,
) -> _ScanReport:
    """Pre-train scan with content-hash cache (miss → full scan + save).
    Cache key is ``(manifest digest, seed, ratios, split names)``; reload
    re-verifies the digest and wall-disjointness. Stale/corrupt entries fail
    closed to a full scan (never raise, never partial).
    """
    cache_file = scan_cache_path(run_dir)
    shared = _shared_scan_cache_path(
        manifest=manifest,
        stream_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )
    cached = load_scan_cache(
        cache_file,
        manifest_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
    )
    if cached is None:
        cached = load_scan_cache(
            shared,
            manifest_digest=stream_digest,
            seed=config.seeds.data_seed,
            ratios=ratios,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
        )
        if cached is not None:
            print("phase: scan-cache shared hit", flush=True)
            save_scan_cache(
                cache_file,
                manifest_digest=stream_digest,
                seed=config.seeds.data_seed,
                ratios=ratios,
                train_split=config.data.train_split,
                val_split=config.data.val_split,
                scan=dict(cached),
            )
    if cached is not None:
        try:
            return _scan_report(
                config,
                set(cached["train_walls"]),  # type: ignore[arg-type]
                set(cached["val_walls"]),  # type: ignore[arg-type]
                int(cached["train_games"]),  # type: ignore[arg-type]
                int(cached["val_games"]),  # type: ignore[arg-type]
                int(cached["train_sim_games"]),  # type: ignore[arg-type]
                int(cached["val_sim_games"]),  # type: ignore[arg-type]
                int(cached["framed"]),  # type: ignore[arg-type]
                int(cached["emitted"]),  # type: ignore[arg-type]
                int(cached["quarantined"]),  # type: ignore[arg-type]
                int(cached["duplicates"]),  # type: ignore[arg-type]
            )
        except (ContractError, ValueError, TypeError, KeyError, AttributeError):
            pass
    report = _scan_corpus(manifest, config=config, ratios=ratios)
    save_scan_cache(
        cache_file,
        manifest_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        scan={
            "train_walls": sorted(report.train_walls),
            "val_walls": sorted(report.val_walls),
            "train_games": report.train_games,
            "val_games": report.val_games,
            "train_sim_games": report.train_sim_games,
            "val_sim_games": report.val_sim_games,
            "framed": report.framed,
            "emitted": report.emitted,
            "quarantined": report.quarantined,
            "duplicates": report.duplicates,
        },
    )
    save_scan_cache(
        shared,
        manifest_digest=stream_digest,
        seed=config.seeds.data_seed,
        ratios=ratios,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        scan={
            "train_walls": sorted(report.train_walls),
            "val_walls": sorted(report.val_walls),
            "train_games": report.train_games,
            "val_games": report.val_games,
            "train_sim_games": report.train_sim_games,
            "val_sim_games": report.val_sim_games,
            "framed": report.framed,
            "emitted": report.emitted,
            "quarantined": report.quarantined,
            "duplicates": report.duplicates,
        },
    )
    return report
