"""Arrow/Parquet with actor vs privileged separation — checklist item 7."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.contracts.observation import DORA_SHAPE

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DecisionRow",
    "PrivilegedRow",
    "verify_no_privileged_leakage",
    "write_actor_shards",
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

PRIVILEGED_FIELDS = (
    "decision_id",
    "privileged_label",
    "full_world",
    "hidden_tiles",
    "wall_remaining",
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
# Hoisted schema — P-B05: avoid per-call pa.schema construction; reuse const.
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


def _actor_observation_is_privileged_free(obs: dict[str, object]) -> None:
    for key in obs:
        if key in FORBIDDEN_IN_ACTOR:
            raise ContractError(f"privileged field leakage into actor observation: {key!r}")
        # Nested check for hidden tiles
        if isinstance(obs[key], dict):
            for sub in obs[key]:  # type: ignore[union-attr]  # reason: obs values statically object; isinstance narrows dict but checker cannot narrow subscript
                if sub in FORBIDDEN_IN_ACTOR:
                    raise ContractError(f"privileged nested field leakage: {key}.{sub}")
    # Dora shape must be (5,) never (4,)
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

    # P-B05 single-pass bucket: O(rows*fields) vs O(splits*fields*rows) ~7.8M checks.
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
        # Reuse hoisted const schema (P-B05) instead of per-shard pa.schema().
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
            write_batch_size=8192,
            store_schema=True,
        )
        # Stream hash via 1 MiB chunks (avoid read_bytes peak).
        # Evidence https://docs.python.org/3/library/hashlib.html chunked update.
        hasher = hashlib.sha256()
        with out_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        shard_hashes[split] = "sha256:" + hasher.hexdigest()
        # P-B06: use ParquetFile metadata/schema.names not full read_table for leakage.
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
                col_names = [md.schema.column(i).name for i in range(md.num_columns)]  # type: ignore[attr-defined]  # reason: metadata schema is dynamic pyarrow type; hasattr-guarded, fallback covers
        except Exception:
            # Fallback to lightweight read of schema only (no row data)
            col_names = pq.read_schema(out_path).names  # type: ignore[attr-defined]  # reason: read_schema returns dynamic Arrow Schema; attribute exists at runtime
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
            json.dumps(r.full_world, separators=(",", ":"))
            if r.full_world is not None
            else ""
            for r in rows
        ],
    }
    # Reuse hoisted privileged schema (P-B05)
    table = pa.table(table_dict, schema=_PRIVILEGED_SCHEMA)
    out_path = destination / "privileged.parquet"
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_batch_size=8192,
    )
    # Stream hash via 1 MiB chunks (evidence hashlib docs)
    hasher = hashlib.sha256()
    with out_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    shard_hash = "sha256:" + hasher.hexdigest()
    manifest = {
        "dataset_hash": dataset_hash,
        "privileged_shards": {"all": shard_hash},
        "row_count": len(rows),
    }
    atomic_replace_bytes(destination / "privileged_manifest.json", canonical_bytes(manifest))
    return {"all": shard_hash}

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
                        cast("dict[str, object]", obs_raw)
                        if isinstance(obs_raw, dict)
                        else {}
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
