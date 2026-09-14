"""Streaming-first game reader: train straight from ``.mjai.json.zst``.

Re-export facade over the split modules: :mod:`hydra2.data.stream_manifest`
(partition vocabulary, manifest, digest, reservoir blob, scan cache, RNG
codec), :mod:`hydra2.data.stream_read` (identity math, framer, cursor/game
records, single-game fetch), :mod:`hydra2.data.stream_iter` (the
``GameStream`` pass), and :mod:`hydra2.data.stream_decode` (spawn-decode
worker, ``PrefetchGameStream``, microbatch/disjoint/firewall helpers).
Import from this path; it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.data.stream_decode import (
    PrefetchGameStream as PrefetchGameStream,
)
from hydra2.data.stream_decode import (
    actor_payload as actor_payload,
)
from hydra2.data.stream_decode import (
    check_wall_disjoint as check_wall_disjoint,
)
from hydra2.data.stream_decode import (
    count_decisions as count_decisions,
)
from hydra2.data.stream_decode import (
    slice_microbatches as slice_microbatches,
)
from hydra2.data.stream_decode import (
    verify_no_privileged_leakage as verify_no_privileged_leakage,
)
from hydra2.data.stream_iter import (
    GameStream as GameStream,
)
from hydra2.data.stream_manifest import (
    DEFAULT_GROUPING_KEYS as DEFAULT_GROUPING_KEYS,
)
from hydra2.data.stream_manifest import (
    PARTITION_ORDER as PARTITION_ORDER,
)
from hydra2.data.stream_manifest import (
    RESERVOIR_BLOB_VERSION as RESERVOIR_BLOB_VERSION,
)
from hydra2.data.stream_manifest import (
    SCAN_CACHE_VERSION as SCAN_CACHE_VERSION,
)
from hydra2.data.stream_manifest import (
    FileEntry as FileEntry,
)
from hydra2.data.stream_manifest import (
    SplitName as SplitName,
)
from hydra2.data.stream_manifest import (
    StreamManifest as StreamManifest,
)
from hydra2.data.stream_manifest import (
    build_manifest as build_manifest,
)
from hydra2.data.stream_manifest import (
    load_scan_cache as load_scan_cache,
)
from hydra2.data.stream_manifest import (
    manifest_digest as manifest_digest,
)
from hydra2.data.stream_manifest import (
    parse_shuffle_rng as parse_shuffle_rng,
)
from hydra2.data.stream_manifest import (
    read_reservoir_blob as read_reservoir_blob,
)
from hydra2.data.stream_manifest import (
    save_scan_cache as save_scan_cache,
)
from hydra2.data.stream_manifest import (
    scan_cache_path as scan_cache_path,
)
from hydra2.data.stream_manifest import (
    serialize_shuffle_rng as serialize_shuffle_rng,
)
from hydra2.data.stream_manifest import (
    write_reservoir_blob as write_reservoir_blob,
)
from hydra2.data.stream_read import (
    StreamCursor as StreamCursor,
)
from hydra2.data.stream_read import (
    StreamGame as StreamGame,
)
from hydra2.data.stream_read import (
    StreamStats as StreamStats,
)
from hydra2.data.stream_read import (
    ZstdLineStream as ZstdLineStream,
)
from hydra2.data.stream_read import (
    assign_split as assign_split,
)
from hydra2.data.stream_read import (
    compute_wall_hash as compute_wall_hash,
)
from hydra2.data.stream_read import (
    fetch_game_at as fetch_game_at,
)
from hydra2.data.stream_read import (
    group_key_for as group_key_for,
)
from hydra2.data.stream_read import (
    group_key_for_path as group_key_for_path,
)
from hydra2.data.stream_read import (
    stem_of as stem_of,
)

__all__ = [
    "DEFAULT_GROUPING_KEYS",
    "PARTITION_ORDER",
    "RESERVOIR_BLOB_VERSION",
    "FileEntry",
    "GameStream",
    "PrefetchGameStream",
    "StreamCursor",
    "StreamGame",
    "StreamManifest",
    "StreamStats",
    "ZstdLineStream",
    "actor_payload",
    "assign_split",
    "build_manifest",
    "check_wall_disjoint",
    "compute_wall_hash",
    "count_decisions",
    "fetch_game_at",
    "group_key_for",
    "group_key_for_path",
    "load_scan_cache",
    "manifest_digest",
    "parse_shuffle_rng",
    "read_reservoir_blob",
    "save_scan_cache",
    "scan_cache_path",
    "serialize_shuffle_rng",
    "stem_of",
    "verify_no_privileged_leakage",
    "write_reservoir_blob",
]

# Names importable from this path before the split that live in the
# submodules now (kept so engine helpers and type-checking imports resolve).
from hydra2.data.stream_decode import _DECODE_BATCH_GAMES as _DECODE_BATCH_GAMES
from hydra2.data.stream_decode import (
    _DECODE_PROC_MAX_WORKERS as _DECODE_PROC_MAX_WORKERS,
)
from hydra2.data.stream_decode import (
    _DECODE_WORKER_COLLECT_EVERY as _DECODE_WORKER_COLLECT_EVERY,
)
from hydra2.data.stream_decode import _DECODE_WORKER_TASKS as _DECODE_WORKER_TASKS
from hydra2.data.stream_decode import _decode_frames_worker as _decode_frames_worker
from hydra2.data.stream_decode import _decode_worker_init as _decode_worker_init
from hydra2.data.stream_decode import _WorkerBatchResult as _WorkerBatchResult
from hydra2.data.stream_manifest import _RESERVOIR_MAGIC as _RESERVOIR_MAGIC
from hydra2.data.stream_manifest import _ratios_match as _ratios_match
from hydra2.data.stream_read import _CHUNK_SIZE as _CHUNK_SIZE
from hydra2.data.stream_read import _DIGIT_RUN as _DIGIT_RUN
from hydra2.data.stream_read import _END_TYPES as _END_TYPES
from hydra2.data.stream_read import _START_TYPES as _START_TYPES
from hydra2.data.stream_read import _check_int as _check_int
from hydra2.data.stream_read import _Run as _Run
from hydra2.data.stream_read import _shuffle_key as _shuffle_key
from hydra2.data.stream_read import _ShuffleState as _ShuffleState
