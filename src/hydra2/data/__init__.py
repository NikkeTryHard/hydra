"""WP-04B Authoritative Data Lineage — package entry.

Exports the public data pipeline API. Inference loaders cannot import privileged
row constructors (enforced by __all__ and separate modules).
"""

from __future__ import annotations

from hydra2.data.attestation import SYNTHETIC_ATTESTATION, Attestation
from hydra2.data.cache import CacheKey, build_cache, cache_key_digest, load_cache
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.ingest import IngestedObject, ingest_packaged_objects
from hydra2.data.loader import load_batch_in_fresh_process, verify_and_load_batch
from hydra2.data.parquet import (
    DecisionRow,
    PrivilegedRow,
    write_actor_shards,
    write_privileged_shards,
)
from hydra2.data.partition import GameIdentity, SplitManifest, SplitSpec, assign_partitions
from hydra2.data.quarantine import QuarantinedRecord, quarantine_invalid
from hydra2.data.rows import PackagedObjectRow, RawObjectRow, make_raw_object_row
from hydra2.data.validate import ValidationOutcome, validate_game

__all__ = [
    "SYNTHETIC_ATTESTATION",
    "Attestation",
    "CacheKey",
    "DecisionRow",
    "GameIdentity",
    "GameRecord",
    "IngestedObject",
    "PackagedObjectRow",
    "PrivilegedRow",
    "QuarantinedRecord",
    "RawObjectRow",
    "SplitManifest",
    "SplitSpec",
    "ValidationOutcome",
    "assign_partitions",
    "build_cache",
    "cache_key_digest",
    "decode_game_object",
    "ingest_packaged_objects",
    "load_batch_in_fresh_process",
    "load_cache",
    "make_raw_object_row",
    "quarantine_invalid",
    "validate_game",
    "verify_and_load_batch",
    "write_actor_shards",
    "write_privileged_shards",
]
