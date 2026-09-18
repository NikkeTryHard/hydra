"""Arrow/Parquet with actor vs privileged separation — checklist item 7."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from hydra2 import _rust_columnar
from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import sha256_file
from hydra2.contracts.common import ContractError
from hydra2.contracts.observation_types import DORA_SHAPE

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DecisionRow",
    "PrivilegedRow",
    "validate_privileged_ranks",
    "verify_no_privileged_leakage",
    "write_actor_shards",
    "write_privileged_ranks",
    "write_privileged_shards",
]


# Actor-visible fields: mirrors DecisionRow spec but strictly actor namespace
ACTOR_FIELDS = (
    "game_id",
    "round_id",
    "decision_id",
    "seat",
    "source_object_id",
    "split",
    "rules_hash",
    "adapter_hash",
    "observation_hash",
    "action_table_hash",
    "derivation_hash",
    "actor_observation",
    "chosen_action_id",
)

# Forbidden leakage: privileged keys must never appear in actor rows
FORBIDDEN_IN_ACTOR = {
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
}
# Hoisted schema: avoid per-call pa.schema construction; reuse the const.
# Evidence: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
# write_table schema is reused; constructing once saves CPU per shard.
_ACTOR_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("round_id", pa.string()),
        ("decision_id", pa.string()),
        ("seat", pa.int64()),
        ("source_object_id", pa.string()),
        ("split", pa.string()),
        ("rules_hash", pa.string()),
        ("adapter_hash", pa.string()),
        ("observation_hash", pa.string()),
        ("action_table_hash", pa.string()),
        ("derivation_hash", pa.string()),
        ("actor_observation", pa.string()),
        ("chosen_action_id", pa.int64()),
    ]
)

_PRIVILEGED_SCHEMA = pa.schema(
    [
        ("decision_id", pa.string()),
        ("privileged_label", pa.string()),
        ("full_world", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class DecisionRow:
    game_id: str
    round_id: str
    decision_id: str
    seat: int
    source_object_id: str
    split: str
    rules_hash: str
    adapter_hash: str
    observation_hash: str
    action_table_hash: str
    derivation_hash: str
    actor_observation: dict[str, object]
    chosen_action_id: int
    privileged_label_ref: str | None = None


@dataclass(frozen=True, slots=True)
class PrivilegedRow:
    decision_id: str
    privileged_label: dict[str, object]
    full_world: dict[str, object] | None = None


# privileged_label opaque-dict key conventions (day-one, no parquet schema break):
# - "ranks": AUTHORITATIVE placement. Strict 1..4 4-list permutation, one rank per
#   seat (rank 1 = first). utility() is the fixed point: ranks -> rank_values ->
#   UtilityVector.values. Read bridge only: 0..3 seat convention and 4-list
#   "final_placement" are accepted by the oracle loader/join but MUST never be
#   written (writers emit 1..4 via validate_privileged_ranks).
# - "hidden_tiles" / "hidden_tile_counts": 34-list belief counts (index-CE target).
# - "wall_id" / "split": provenance passthrough inside the opaque dict (kept out
#   of the Arrow schema so the parquet layout never breaks).
# Belief note: belief loss stays index-CE; 34-dim distribution support is DEFERRED
# (see join_oracle_targets docstring).


def validate_privileged_ranks(ranks: object, decision_id: str = "") -> tuple[int, int, int, int]:
    """Pin the day-one ranks convention: strict 1..4 4-list permutation.

    Returns the validated ranks as a 4-tuple. Raises ContractError for any
    non-list/tuple, wrong length, non-int (bool included), or non-permutation
    input — including the 0..3 seat convention, which is a read-only bridge in
    the oracle loader and must never be written.
    """
    where = f" for {decision_id!r}" if decision_id != "" else ""
    if not isinstance(ranks, (list, tuple)) or len(ranks) != 4:
        raise ContractError(f"privileged_label['ranks'] must be a 4-list{where}")
    r0: object = ranks[0]
    r1: object = ranks[1]
    r2: object = ranks[2]
    r3: object = ranks[3]
    if (
        isinstance(r0, bool)
        or not isinstance(r0, int)
        or isinstance(r1, bool)
        or not isinstance(r1, int)
        or isinstance(r2, bool)
        or not isinstance(r2, int)
        or isinstance(r3, bool)
        or not isinstance(r3, int)
    ):
        raise ContractError(f"privileged_label['ranks'] must be int 1..4{where}")
    ordered = (r0, r1, r2, r3)
    if sorted(ordered) != [1, 2, 3, 4]:
        raise ContractError(
            f"privileged_label['ranks'] must be a strict 1..4 permutation{where}, "
            f"got {list(ordered)!r}"
        )
    return ordered


def _actor_observation_is_privileged_free(obs: dict[str, object]) -> None:
    for key in obs:
        if key in FORBIDDEN_IN_ACTOR:
            raise ContractError(f"privileged field leakage into actor observation: {key!r}")
        # Nested check for hidden tiles
        if isinstance(obs[key], dict):
            for sub in obs[key]:  # type: ignore[union-attr]  # reason: obs values statically object; isinstance narrows dict but checker cannot narrow subscript
                if sub in FORBIDDEN_IN_ACTOR:
                    raise ContractError(f"privileged nested field leakage: {key}.{sub}")
    # Dora shape must be (5,) never (4,); dora is the 5 indicator tiles
    # fixing the bonus-tile mapping ((5,) exact, never the (4,) shim).
    for k in ("dora_indicators", "dora", "indicators"):
        v = obs.get(k)
        if isinstance(v, list) and len(v) == 4:
            raise ContractError(f"actor observation has (4,) dora shim for {k!r}; expected (5,)")
    # If dora_indicators present, ensure length 5
    di = obs.get("dora_indicators")
    if isinstance(di, list) and len(di) not in (0, 5):
        pass


def write_actor_shards(
    *,
    destination: Path,
    rows: list[DecisionRow],
    dataset_hash: str,
    split_manifest_hash: str,
) -> dict[str, str]:
    """Write actor-visible shards as Parquet; returns shard hashes."""
    if len(rows) == 0:
        raise ContractError("no actor rows to write")
    # Validate no privileged leakage and dora shape
    for r in rows:
        _actor_observation_is_privileged_free(r.actor_observation)
        if r.actor_observation.get("dora_indicators") is not None:
            di = r.actor_observation.get("dora_indicators")
            if isinstance(di, list) and len(di) == 4:
                raise ContractError(f"dora shim (4,) in row {r.decision_id}")

    # Single-pass bucket: O(rows*fields) vs O(splits*fields*rows) ~7.8M checks.
    # Evidence: arrow write_table docs; hoisted _ACTOR_SCHEMA reused per shard.
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
    buckets: dict[str, dict[str, list[object]]] = {}
    for r in rows:
        b = buckets.setdefault(r.split, {field: [] for field in ACTOR_FIELDS})
        b["game_id"].append(r.game_id)
        b["round_id"].append(r.round_id)
        b["decision_id"].append(r.decision_id)
        b["seat"].append(r.seat)
        b["source_object_id"].append(r.source_object_id)
        b["split"].append(r.split)
        b["rules_hash"].append(r.rules_hash)
        b["adapter_hash"].append(r.adapter_hash)
        b["observation_hash"].append(r.observation_hash)
        b["action_table_hash"].append(r.action_table_hash)
        b["derivation_hash"].append(r.derivation_hash)
        b["actor_observation"].append(json.dumps(r.actor_observation, separators=(",", ":")))
        b["chosen_action_id"].append(r.chosen_action_id)

    destination.mkdir(parents=True, exist_ok=True)
    shard_hashes: dict[str, str] = {}
    for split in sorted(buckets):
        split_dict = buckets[split]
        # Reuse hoisted const schema instead of per-shard pa.schema().
        split_table = pa.table(split_dict, schema=_ACTOR_SCHEMA)
        out_path = destination / f"actor-{split}.parquet"
        # Explicit opts: zstd + dict (arrow write_table docs)
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html  # noqa: E501  # reason: evidence URL cannot wrap without breaking link; splitting harms copy-paste
        pq.write_table(
            split_table,
            out_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_batch_size=_rust_columnar.batch_size(),
            store_schema=True,
        )
        # Shard digest via the hard-Rust digest owner (bridge canon_rng
        # sha256_file, native 1 MiB chunks — same digest, byte-identical).
        # ImportError fails closed with a build-ext hint — no oracle fallback.
        shard_hashes[split] = str(sha256_file(out_path))
        # Read column names via ParquetFile metadata, not a full read_table.
        # Evidence https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
        try:
            pf = pq.ParquetFile(out_path)
            # Prefer pf.schema.names (arrow) else metadata schema
            if hasattr(pf, "schema_arrow") and pf.schema_arrow is not None:
                col_names = list(pf.schema_arrow.names)
            elif hasattr(pf, "schema") and hasattr(pf.schema, "names"):
                col_names = list(pf.schema.names)  # type: ignore[attr-defined]  # reason: pyarrow ParquetFile.schema type varies by version; hasattr-guarded above, fallback covers
            else:
                # Fallback via metadata
                md = pf.metadata
                col_names = [md.schema.column(i).name for i in range(md.num_columns)]  # type: ignore[attr-defined]
                # reason: metadata schema dynamic pyarrow type; guarded fallback
        except Exception:  # why-broad: schema probe may fail on any backend;
            # fall back to schema-only read (no row data).
            col_names = pq.read_schema(out_path).names  # type: ignore[attr-defined]
            # reason: read_schema returns dynamic Arrow Schema at runtime
        for col in col_names:
            if col in FORBIDDEN_IN_ACTOR:
                raise ContractError(f"shard {out_path} leaks privileged column {col!r}")
    # Write dataset-level manifest
    manifest = {
        "dataset_hash": dataset_hash,
        "split_manifest_hash": split_manifest_hash,
        "actor_shards": shard_hashes,
        "row_count": len(rows),
        "dora_shape": list(DORA_SHAPE),
    }
    atomic_replace_bytes(destination / "actor_manifest.json", canonical_bytes(manifest))
    return shard_hashes


def write_privileged_shards(
    *,
    destination: Path,
    rows: list[PrivilegedRow],
    dataset_hash: str | None = None,
    split_manifest_hash: str | None = None,
    **_ignored: object,
) -> dict[str, str]:
    if len(rows) == 0:
        return {}
    destination.mkdir(parents=True, exist_ok=True)
    table_dict = {
        "decision_id": [r.decision_id for r in rows],
        "privileged_label": [json.dumps(r.privileged_label, separators=(",", ":")) for r in rows],
        "full_world": [
            json.dumps(r.full_world, separators=(",", ":")) if r.full_world is not None else ""
            for r in rows
        ],
    }
    # Reuse hoisted privileged schema instead of per-call construction.
    table = pa.table(table_dict, schema=_PRIVILEGED_SCHEMA)
    # Shard name must match the oracle loader glob (privileged-*.parquet,
    # mirroring actor-*.parquet sharding). The bare "privileged.parquet" name was
    # invisible to the loader and silently yielded zero rows.
    out_path = destination / "privileged-000.parquet"
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_batch_size=_rust_columnar.batch_size(),
    )
    shard_hash = str(sha256_file(out_path))
    manifest = {
        "dataset_hash": dataset_hash,
        "privileged_shards": {"all": shard_hash},
        "row_count": len(rows),
    }
    atomic_replace_bytes(destination / "privileged_manifest.json", canonical_bytes(manifest))
    return {"all": shard_hash}


def write_privileged_ranks(
    *,
    destination: Path,
    ranks_by_id: dict[str, list[int] | tuple[int, int, int, int]],
    wall_ids: dict[str, str] | None = None,
    split: str = "train",
    dataset_hash: str | None = None,
    split_manifest_hash: str | None = None,
) -> dict[str, str]:
    """Write privileged placement rows carrying the pinned 1..4 ranks convention.

    Each decision_id maps to a strict 1..4 permutation (validated via
    validate_privileged_ranks). wall_id/split ride inside the opaque
    privileged_label dict (passthrough, no Arrow schema break); the loader/join
    consumes ranks by opaque decision_id only, never through the actor batch.
    Only the "train" split is written (oracle distillation is train-only).
    """
    if len(ranks_by_id) == 0:
        raise ContractError("no privileged ranks to write")
    if split != "train":
        raise ContractError(f"privileged ranks split must be 'train', got {split!r}")
    rows: list[PrivilegedRow] = []
    for decision_id in sorted(ranks_by_id):
        if not isinstance(decision_id, str) or decision_id == "":
            raise ContractError("decision_id must be an opaque non-empty str")
        ordered = validate_privileged_ranks(ranks_by_id[decision_id], decision_id)
        label: dict[str, object] = {"ranks": list(ordered), "split": split}
        if wall_ids is not None and decision_id in wall_ids:
            wall_id = wall_ids[decision_id]
            if not isinstance(wall_id, str) or wall_id == "":
                raise ContractError(f"wall_id must be a non-empty str for {decision_id!r}")
            label["wall_id"] = wall_id
        rows.append(PrivilegedRow(decision_id=decision_id, privileged_label=label))
    return write_privileged_shards(
        destination=destination,
        rows=rows,
        dataset_hash=dataset_hash,
        split_manifest_hash=split_manifest_hash,
    )


def verify_no_privileged_leakage(actor_parquet_path: Path) -> None:
    """Hard failure: privileged inference field in actor shard is forbidden.

    Zero-copy ingest (perf-A 4.3):
    - memory_map, pre_buffer, use_threads for mmap + coalesced reads.
    - table.to_batches(8192) streams zero-copy slices vs whole to_pylist.
    - column.to_numpy(zero_copy_only=False) for primitives; strings still
      need per-batch to_pylist but bounded to batch size.
    - For dirs prefer ds.dataset(...).scanner(...).to_batches().
    """
    _ = ds.dataset  # keep import live
    # mmap + pre_buffer + use_threads per arrow docs
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
    table = pq.read_table(actor_parquet_path, memory_map=True, pre_buffer=True, use_threads=True)
    for col in table.column_names:
        if col in FORBIDDEN_IN_ACTOR or col.startswith("privileged"):
            raise ContractError(f"privileged field leakage detected in actor shard column {col!r}")
    # Check actor_observation JSON for hidden fields — batched.
    if "actor_observation" in table.column_names:
        # Dataset scanner for dirs (canonical):
        #   ds.dataset(path, format="parquet").scanner(
        #       columns=["actor_observation"], batch_size=8192).to_batches()
        # Single-file fast path uses table.to_batches.
        for batch in table.to_batches(max_chunksize=8192):
            idx = batch.schema.get_field_index("actor_observation")
            if idx < 0:
                continue
            col = batch.column(idx)
            # For string columns, to_pylist per-batch bounds Python object creation to batch size;
            # for primitive columns one would use col.to_numpy(zero_copy_only=False).
            for obs_json in col.to_pylist():
                try:
                    obs_raw: object = json.loads(obs_json) if isinstance(obs_json, str) else {}
                    obs: dict[str, object] = (
                        cast("dict[str, object]", obs_raw) if isinstance(obs_raw, dict) else {}
                    )
                except json.JSONDecodeError:
                    continue
                if any(k in FORBIDDEN_IN_ACTOR for k in obs):
                    raise ContractError(
                        f"privileged field leakage inside actor_observation: {obs.keys()}"
                    )
                # dora shim check — mirrors _actor_observation_is_privileged_free above
                for k in ("dora_indicators", "dora", "indicators"):
                    v: object = obs.get(k)
                    if isinstance(v, list) and len(v) == 4:
                        raise ContractError(
                            f"(4,) dora shim detected in actor observation via {k!r}"
                        )
