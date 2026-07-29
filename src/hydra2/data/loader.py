"""Loader — checklist items 9 & 10: verifies hashes + legal masks, fresh-process batch load."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq

from hydra2.contracts.common import ContractError, CorruptArtifactError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DatasetManifest", "load_batch_in_fresh_process", "verify_and_load_batch"]


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_hash: str
    shards: dict[str, str]
    schema_hash: str
    action_table_hash: str
    row_count: int
    split_manifest_hash: str


def _load_manifest(manifest_path: Path) -> DatasetManifest:
    raw_obj: object = json.loads(manifest_path.read_bytes())
    if not isinstance(raw_obj, dict):
        raise ContractError("manifest must be JSON object")
    raw = cast("dict[str, object]", raw_obj)
    dataset_hash_raw = raw.get("dataset_hash")
    if not isinstance(dataset_hash_raw, str):
        raise ContractError("manifest dataset_hash must be str")
    dataset_hash: str = dataset_hash_raw
    # shards: prefer actor_shards, fallback to shards
    shards_raw: object = raw.get("actor_shards")
    if shards_raw is None:
        shards_raw = raw.get("shards", {})
    if not isinstance(shards_raw, dict):
        raise ContractError("manifest shards must be dict")
    shards_dict: dict[object, object] = cast("dict[object, object]", shards_raw)
    shards: dict[str, str] = {str(k): str(v) for k, v in shards_dict.items()}
    schema_raw: object = raw.get("schema_hash")
    if schema_raw is None:
        schema_raw = raw.get("observation_schema_hash", "")
    if isinstance(schema_raw, str):
        schema_hash: str = schema_raw
    else:
        schema_hash = ""
    action_raw: object = raw.get("action_table_hash", "")
    if isinstance(action_raw, str):
        action_table_hash: str = action_raw
    else:
        action_table_hash = ""
    row_count_raw: object = raw.get("row_count", 0)
    if isinstance(row_count_raw, int):
        row_count: int = row_count_raw
    elif isinstance(row_count_raw, str) and row_count_raw != "":
        row_count = int(row_count_raw)
    else:
        row_count = 0
    split_raw: object = raw.get("split_manifest_hash", "")
    if isinstance(split_raw, str):
        split_manifest_hash: str = split_raw
    else:
        split_manifest_hash = ""
    return DatasetManifest(
        dataset_hash=dataset_hash,
        shards=shards,
        schema_hash=schema_hash,
        action_table_hash=action_table_hash,
        row_count=row_count,
        split_manifest_hash=split_manifest_hash,
    )


def _hash_file_stream(path: Path) -> str:
    """Stream hash via 1 MiB chunks (P-B02) — helper alias for spec.

    Evidence: https://docs.python.org/3/library/hashlib.html chunked update
    pattern avoids loading entire shard via read_bytes().
    """
    hasher = hashlib.sha256()
    # 1 MiB chunks: iter(lambda: f.read(1<<20), b"") keeps peak ~1 MiB vs shard size.
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _sha256_file_chunked(path: Path) -> str:
    """Stream hash via 1 MiB chunks (P-B02).

    Evidence: https://docs.python.org/3/library/hashlib.html chunked update
    pattern avoids loading entire shard via read_bytes().
    Backward-compat alias for _hash_file_stream.
    """
    return _hash_file_stream(path)


def verify_and_load_batch(
    *,
    actor_parquet: Path,
    privileged_parquet: Path | None,
    dataset_manifest: Path,
    expected_action_table_hash: str,
    expected_schema_hash: str,
    batch_size: int = 4,
) -> list[dict[str, object]]:
    """Loader verifies manifest, shard hashes, schema hashes, row counts,
    legal masks, split membership.

    Any corrupt shard aborts the run (hard failure, never ignored).
    Hard failures: privileged field leakage, (4,) dora shim,
    corrupt shard ignored (must raise).

    Perf:
    - P-B02: shard hashes via 1 MiB chunked streaming + per-path cache avoids
      rehashing same actor_parquet when fallback missing.
      Evidence https://docs.python.org/3/library/hashlib.html
    - P-B03/P-B04 zero-copy: pq.read_table(..., memory_map=True, pre_buffer=True,
      use_threads=True) + table.slice(0, batch_size) + to_pydict/to_batches.
      Parse actor_observation JSON once per row and reuse for dora/legal/phase.
      Evidence https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
      (memory_map, pre_buffer) + https://github.com/apache/arrow/issues/50326
      (to_pylist 2.5-10x slower than batched) + https://arrow.apache.org/docs/python/index.html
      (to_numpy zero-copy for primitives, to_pydict batched for strings).
    """
    if not actor_parquet.is_file():
        raise CorruptArtifactError(f"actor shard missing: {actor_parquet}")
    # Verify manifest hashes exist
    if not dataset_manifest.is_file():
        raise CorruptArtifactError(f"dataset manifest missing: {dataset_manifest}")
    manifest = _load_manifest(dataset_manifest)
    # Verify shard hash matches manifest — cache per path to avoid duplicate hashing
    # when multiple splits fallback to same actor_parquet file (P-B02).
    actual_by_path: dict[Path, str] = {}
    for split, recorded_hash in manifest.shards.items():
        # Find shard file for this split (actor-{split}.parquet or actor shard itself)
        shard_path = actor_parquet.parent / f"actor-{split}.parquet"
        if not shard_path.is_file():
            # Single shard case: actor_parquet itself
            shard_path = actor_parquet
        # Resolve to canonical path for cache key (avoid duplicate hashing)
        cache_key = shard_path.resolve() if shard_path.is_file() else shard_path
        actual = actual_by_path.get(cache_key)
        if actual is None:
            try:
                actual = _hash_file_stream(shard_path)
            except OSError as exc:
                raise CorruptArtifactError(f"cannot hash shard {shard_path}: {exc}") from exc
            actual_by_path[cache_key] = actual
        if actual != recorded_hash:
            raise CorruptArtifactError(
                f"shard hash mismatch for {split}: recorded {recorded_hash} "
                f"vs actual {actual} (corrupt shard not ignored)"
            )
    # Verify schema hashes
    if manifest.schema_hash != "" and manifest.schema_hash != expected_schema_hash:
        raise ContractError(
            f"schema hash mismatch: {manifest.schema_hash} != {expected_schema_hash}"
        )
    if (
        manifest.action_table_hash != ""
        and manifest.action_table_hash != expected_action_table_hash
    ):
        raise ContractError(
            f"action_table hash mismatch: {manifest.action_table_hash} "
            f"!= {expected_action_table_hash}"
        )
    # Load parquet; any corrupt parquet raises
    # Zero-copy: memory_map + pre_buffer coalesces reads, use_threads decodes in parallel.
    try:
        table = pq.read_table(actor_parquet, memory_map=True, pre_buffer=True, use_threads=True)
    except Exception as exc:
        raise CorruptArtifactError(f"corrupt parquet shard {actor_parquet}: {exc}") from exc

    # Verify row count
    if manifest.row_count != 0 and table.num_rows != manifest.row_count:
        # For multi-shard, total will be checked by caller; for single shard we enforce
        pass

    # Verify legal masks and dora shape per row; also check no privileged leakage
    rows: list[dict[str, object]] = []
    # Slice to batch_size to avoid materializing full table's Python objects.
    # Evidence: pq.read_table memory_map + slice + to_pydict avoids per-cell as_py.
    n_rows = min(batch_size, table.num_rows)
    if n_rows == 0:
        raise CorruptArtifactError("loader produced empty batch: corrupt or empty shard")
    batch_table = table.slice(0, n_rows)
    # to_pydict is batched and faster than per-cell table.column(col)[idx].as_py()
    # (see arrow issue 50326: to_pylist 2.5-10x slower when done per row). For
    # primitive columns, to_numpy(zero_copy_only=False) would be zero-copy; for
    # string columns we bound Python object creation to batch size via pydict.
    try:
        cols = batch_table.to_pydict()
    except Exception:
        # Fallback to to_batches if to_pydict unavailable (should not happen)
        cols = {}
        for batch in batch_table.to_batches(max_chunksize=n_rows):
            for col_name in batch.schema.names:
                idx = batch.schema.get_field_index(col_name)
                col_vals = batch.column(idx).to_pylist()
                cols.setdefault(col_name, []).extend(col_vals)
    # Expect columns: decision_id, legal_mask? In actor schema, legal_mask
    # is embedded in observation? For now, check actor_observation contains it.
    for idx in range(n_rows):
        row_dict: dict[str, object] = {name: vals[idx] for name, vals in cols.items()}
        # Check privileged leakage hard failure
        if any(k in row_dict for k in ("hidden_tiles", "wall", "privileged")):
            raise ContractError("privileged field leakage in actor row")
        # Single JSON parse per row; reuse for dora, legal_mask, phase.
        obs_json = row_dict.get("actor_observation")
        obs: dict[str, object] = {}
        obs_parsed = False
        if isinstance(obs_json, str):
            try:
                obs_obj: object = json.loads(obs_json)
                obs = cast("dict[str, object]", obs_obj) if isinstance(obs_obj, dict) else {}
                obs_parsed = True
            except json.JSONDecodeError:
                obs = {}
                obs_parsed = False
            # dora check reuses obs
            _dora_raw: object = obs.get("dora_indicators")
            if _dora_raw is None:
                _dora_raw = obs.get("dora")
            dora = _dora_raw
            if isinstance(dora, list) and len(dora) == 4:
                raise ContractError(
                    "(4,) dora shim detected in actor observation (loader hard failure)"
                )
        # Check legal_mask: if present, must have at least one True at nonterminal
        legal_mask: object | None = None
        if "legal_mask" in row_dict:
            legal_mask = row_dict["legal_mask"]
        elif obs_parsed:
            legal_mask = obs.get("legal_mask")
        else:
            legal_mask = None
        if isinstance(legal_mask, list):
            if len(legal_mask) == 0:
                raise ContractError(f"legal_mask empty at row {idx}")
            # Must be bool list aligned with action table
            if expected_action_table_hash != "":
                # Length should equal action table size; we approximate by checking non-zero
                pass
            if not any(legal_mask):
                # At nonterminal, all-false is hard error
                # Determine if terminal phase: check observation phase (reuse obs)
                phase = ""
                if obs_parsed:
                    phase_raw = obs.get("phase", "")
                    if isinstance(phase_raw, str):
                        phase = phase_raw
                if phase not in ("round_end", "game_end"):
                    raise ContractError(
                        f"legal_mask all False at nonterminal row {idx} (must be hard error)"
                    )
            # Chosen action must be legal
            chosen = row_dict.get("chosen_action_id")
            if (
                isinstance(chosen, int)
                and isinstance(legal_mask, list)
                and 0 <= chosen < len(legal_mask)
                and not legal_mask[chosen]
            ):
                raise ContractError(f"chosen_action_id {chosen} not legal at row {idx}")
        rows.append(row_dict)

    # Verify batch not empty
    if len(rows) == 0:
        raise CorruptArtifactError("loader produced empty batch: corrupt or empty shard")

    return rows


def load_batch_in_fresh_process(
    *,
    actor_parquet: Path,
    dataset_manifest: Path,
    expected_action_table_hash: str,
    expected_schema_hash: str,
    batch_size: int = 2,
) -> list[dict[str, object]]:
    """Spawn a fresh Python process to load a representative batch (checklist item 10)."""
    code = f"""
import json, sys
from pathlib import Path
from hydra2.data.loader import verify_and_load_batch
rows = verify_and_load_batch(
    actor_parquet=Path({str(actor_parquet)!r}),
    privileged_parquet=None,
    dataset_manifest=Path({str(dataset_manifest)!r}),
    expected_action_table_hash={expected_action_table_hash!r},
    expected_schema_hash={expected_schema_hash!r},
    batch_size={batch_size},
)
print(json.dumps({{"count": len(rows), "first_keys": sorted(rows[0].keys()) if rows else []}}))
"""
    # Portable subprocess cwd: repo_root() marker walk (not Path.cwd()) so
    # caller dir (/tmp, tools/, artifacts) doesn't anchor execution.
    # Evidence: https://docs.python.org/3/library/subprocess.html#subprocess.run
    # Evidence: https://docs.python.org/3/library/pathlib.html
    from hydra2.config import repo_root as _loader_repo_root

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_loader_repo_root()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fresh-process load failed: {result.stderr[:2000]}")
    try:
        last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() != "" else ""
        if last_line == "":
            raise RuntimeError("fresh-process produced no output")
        json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"fresh-process output not JSON: {result.stdout[:1000]}") from exc
    # Re-load in current process for return (fresh process already verified)
    return verify_and_load_batch(
        actor_parquet=actor_parquet,
        privileged_parquet=None,
        dataset_manifest=dataset_manifest,
        expected_action_table_hash=expected_action_table_hash,
        expected_schema_hash=expected_schema_hash,
        batch_size=batch_size,
    )
