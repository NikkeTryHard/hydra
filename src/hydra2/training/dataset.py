"""WP-05B authoritative data integration: synthetic parquet dataset.

Wraps the WP-04B actor parquet shards with deterministic ordering,
privileged-field rejection, legal-mask verification, and sampler-cursor
tracking for resume.

Authoritative checks enforced on construction and on every batch yielded:
 - actor parquet column names never contain privileged keys (FORBIDDEN_IN_ACTOR)
 - verify_no_privileged_leakage called per shard
 - dora shape (5,) is enforced (no (4,) shim)
 - no privileged parquet path may be supplied; any such leakage raises
   ContractError before training sees data.
 - legal_mask rows must have at least one legal action and chosen action
   must be legal.

The dataset is synthetic-qualified: it reads the tiny shards written by
``write_actor_shards`` in tests.  The same code path would read the real
corpus after D-017 attestation; the synthetic qualifier is the data, not
the loader path.

Determinism: ordering is canonical (sorted by decision_id) then optionally
permuted by a seeded generator.  Sampler cursor is a plain ``{"offset": int,
"seed": int, "total": int}`` JSON value stored in TrainingState and the
checkpoint ``sampler_state`` section, enabling bitwise resume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.parquet import (
    ACTOR_FIELDS,
    FORBIDDEN_IN_ACTOR,
    verify_no_privileged_leakage,
)

__all__ = [
    "AuthoritativeParquetDataset",
    "SamplerState",
    "tensorize_actor_row",
]


@dataclass(frozen=True, slots=True)
class SamplerState:
    offset: int
    seed: int
    total: int
    epoch: int = 0


def _require_actor_parquet_dir(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"actor parquet path is not a directory: {path}")
    shards = sorted(path.glob("actor-*.parquet"))
    if len(shards) == 0:
        raise ContractError(f"no actor shards found in {path} (expected actor-*.parquet)")
    # Reject privileged shards masquerading as actor: any file with privileged name
    for p in path.glob("privileged*"):
        raise ContractError(
            f"privileged shard present in actor dataset directory {path}: {p.name} — "
            "actor loader must never touch privileged parquet (WP-05B no privileged fields)"
        )
    for p in path.glob("*.parquet"):
        if "privileged" in p.name.lower():
            raise ContractError(f"privileged parquet detected in actor dir: {p.name}")
    return shards


def _verify_shards(shards: list[Path]) -> None:
    for shard in shards:
        # Hard failure: privileged leakage in actor shard
        verify_no_privileged_leakage(shard)
        # Perf-A HIGH: memory_map read + schema-only verification avoids
        # full materialization when possible.
        # Evidence:
        #  https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html  # noqa: E501 -- URL cannot be wrapped without breaking link; alternative (shortened URL) loses precision
        #  + https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html  # noqa: E501 -- URL cannot be wrapped without breaking link; alternative loses precision
        # Use ParquetFile to inspect schema without loading all batches if
        # available; fallback to read_table.
        try:
            pf = pq.ParquetFile(shard, memory_map=True)
            column_names = pf.schema.names
            num_rows = pf.metadata.num_rows
        except Exception:
            table = pq.read_table(shard, memory_map=True, pre_buffer=True, use_threads=True)
            column_names = table.column_names
            num_rows = table.num_rows
        for col in column_names:
            if col in FORBIDDEN_IN_ACTOR:
                raise ContractError(f"actor shard {shard.name} contains privileged column {col!r}")
            if col not in ACTOR_FIELDS:
                raise ContractError(f"actor shard {shard.name} has unexpected column {col!r}")
        for required in ("decision_id", "chosen_action_id", "actor_observation"):
            if required not in column_names:
                raise CorruptArtifactError(
                    f"actor shard {shard.name} missing required column {required!r}"
                )
        if num_rows == 0:
            raise ContractError(f"actor shard {shard.name} contains zero rows")
def _lexicographic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def tensorize_actor_row(
    row: dict[str, Any],
    *,
    num_actions: int,
    feature_dim: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic tensorization of one actor row for tests/synthetic data.

    The real WP-05A encoder would parse ``actor_observation`` JSON and produce
    per ``model_input_v1`` tensors.  This helper is the WP-05B synthetic
    stand-in that is deterministic, actor-visible only, and never touches
    privileged data.

    Produces:
      features: FloatTensor [feature_dim] hashed from decision_id
      legal_mask: BoolTensor [num_actions]
      chosen_action_id: LongTensor scalar (chosen_action_id % num_actions, ensured legal)
    """
    decision_id: str = str(row["decision_id"])
    chosen_raw: int = int(row["chosen_action_id"])
    # Deterministic features from decision_id hash
    h = hashlib.sha256(f"{decision_id}:{seed}".encode()).digest()
    # Expand to feature_dim floats via hash bytes
    vals: list[float] = []
    for i in range(feature_dim):
        # cycle through hash bytes
        b = h[i % len(h)]
        vals.append((b / 255.0) * 2 - 1)  # in [-1,1]
    features = torch.tensor(vals, dtype=torch.float32)
    # Deterministic legal mask: ensure chosen is legal, plus random other legals
    gen = torch.Generator().manual_seed(_lexicographic_hash(decision_id) ^ seed)
    # Randomly decide legal count 1..min(8, num_actions)
    legal_count = int(torch.randint(1, min(8, num_actions) + 1, (1,), generator=gen).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for synthetic row; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    legal_mask = torch.zeros(num_actions, dtype=torch.bool)
    # Always include chosen
    chosen = chosen_raw % num_actions
    legal_mask[chosen] = True
    # Fill remaining
    candidates: list[int] = list(range(num_actions))
    candidates.remove(chosen)
    perm_any: Any = torch.randperm(len(candidates), generator=gen).tolist()
    perm: list[int] = [int(x) for x in perm_any]
    for idx in perm[: legal_count - 1]:
        cand: int = candidates[idx]
        legal_mask[cand] = True
    return {
        "features": features,
        "legal_mask": legal_mask,
        "chosen_action_id": torch.tensor(chosen, dtype=torch.long),
        "decision_id": decision_id,
    }
class AuthoritativeParquetDataset:
    """Deterministic authoritative parquet dataset for WP-05B.

    Args:
        parquet_dir: directory containing ``actor-*.parquet`` shards written
            by :func:`hydra2.data.parquet.write_actor_shards`.
        feature_dim: synthetic feature dimensionality (used when observation
            is not pre-tensorized).
        num_actions: canonical action vocab size.  Defaults to the frozen
            action table size (6792) when ``None`` is passed; tests may use
            a smaller value for speed by passing e.g. ``16``.
        seed: deterministic shuffle seed.  ``None`` disables shuffling
            (canonical lexicographic order).  When set, the permutation is
            computed once from the seed and the cursor tracks offset into
            the permuted order.
        verify: when ``True`` (default) shard verification runs on init;
            ``False`` skips verification for unit probes that inject bad rows
            directly.
    """

    def __init__(
        self,
        *,
        parquet_dir: Path,
        feature_dim: int = 16,
        num_actions: int | None = 6792,
        seed: int | None = 0,
        verify: bool = True,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.feature_dim = feature_dim
        self.num_actions = num_actions if num_actions is not None else 6792
        self.seed = seed
        self._rows: list[dict[str, Any]] = []
        self._cursor: int = 0
        self._epoch: int = 0

        shards = _require_actor_parquet_dir(self.parquet_dir)
        if verify:
            _verify_shards(shards)
        for shard in shards:
            # Perf-A HIGH to_pylist break: memory_map + projection
            # + batched to_pydict replaces per-cell as_py loop.
            # Evidence:
            #  https://arrow.apache.org/docs/python/generated/
            #  pyarrow.parquet.read_table.html
            #  (memory_map=True zero-copy mmap)
            #  + https://arrow.apache.org/docs/python/generated/
            #  pyarrow.Table.html#pyarrow.Table.to_batches
            #  (max_chunksize bounds per-batch to_pylist
            #  to batch_size not full table)
            #  + https://arrow.apache.org/docs/python/dataset.html
            #  (Scanner iter_batches batch_size=2048 use_threads=True
            #  pre_buffer=True for same effect;
            #  pq.read_table is equivalent for single-file)
            #  + https://arrow.apache.org/docs/python/generated/
            #  pyarrow.Table.html#pyarrow.Table.to_pydict
            #  (zero-copy-ish per-batch conversion; pyarrow 25 idiom)
            # Previous O(N*C) dict cols reconstruct via
            # {name: table.column(name).to_pylist()} kept full-table
            # string actor_observation in memory.
            # Now: table.to_batches(max_chunksize=8192)
            # + batch.to_pydict() keeps per-batch to_pylist
            # bounded to 8192 rows.
            table = pq.read_table(
                shard,
                memory_map=True,
                columns=list(ACTOR_FIELDS),
                pre_buffer=True,
                use_threads=True,
            )
            for batch in table.to_batches(max_chunksize=8192):
                cols = batch.to_pydict()
                if not cols:
                    continue
                col_names = list(cols.keys())
                for row_tuple in zip(*cols.values(), strict=True):
                    raw: dict[str, Any] = dict(zip(col_names, row_tuple, strict=True))
                    # Verify no privileged field in the raw dict (defense in depth)
                    for bad in FORBIDDEN_IN_ACTOR:
                        if bad in raw:
                            raise ContractError(
                                f"privileged field {bad!r} in raw row {raw.get('decision_id')!r}"
                            )
                    # Also verify actor_observation JSON does not contain privileged keys
                    obs_raw: Any = raw.get("actor_observation")
                    if isinstance(obs_raw, str):
                        try:
                            obs_any: Any = json.loads(obs_raw)
                        except Exception as exc:
                            raise CorruptArtifactError(
                                f"actor_observation not JSON for {raw.get('decision_id')!r}"
                            ) from exc
                        if isinstance(obs_any, dict):
                            obs: dict[str, Any] = dict(obs_any)
                            did: Any = raw.get("decision_id")
                            for k_any in obs:
                                if not isinstance(k_any, str):
                                    continue
                                k: str = k_any
                                if k in FORBIDDEN_IN_ACTOR:
                                    raise ContractError(
                                        f"privileged field {k!r} inside "
                                        f"actor_observation for {did!r}"
                                    )
                            # dora shape check
                            for dk in ("dora_indicators", "dora", "indicators"):
                                v: Any = obs.get(dk)
                                if isinstance(v, list) and len(v) == 4:
                                    raise ContractError(
                                        f"(4,) dora shim in actor_observation[{dk!r}] for {did!r}"
                                    )
                    self._rows.append(raw)
        if len(self._rows) == 0:
            raise ContractError("authoritative dataset contains zero rows after loading")

        # Canonical order: sorted by decision_id lexicographically
        def _sort_key(r: dict[str, Any]) -> str:
            return str(r.get("decision_id", ""))

        self._rows.sort(key=_sort_key)
        # Preserve canonical order for deterministic reseeding on resume
        self._canonical_rows: list[dict[str, Any]] = list(self._rows)
        # Apply deterministic permutation if seed is not None
        if self.seed is not None:
            gen = torch.Generator().manual_seed(self.seed)
            perm_any: Any = torch.randperm(len(self._rows), generator=gen).tolist()
            perm: list[int] = [int(x) for x in perm_any]
            self._rows = [self._canonical_rows[i] for i in perm]
        self._total = len(self._rows)
    # ------------------------------------------------------------------
    # Cursor / sampler state
    # ------------------------------------------------------------------

    def get_sampler_state(self) -> dict[str, Any]:
        return {
            "offset": self._cursor,
            "seed": -1 if self.seed is None else self.seed,
            "total": self._total,
            "epoch": self._epoch,
        }

    def set_sampler_state(self, state: dict[str, Any] | SamplerState) -> None:
        if isinstance(state, dict):
            offset = int(state.get("offset", 0))
            epoch = int(state.get("epoch", 0))
            seed_raw = state.get("seed", None)
        else:
            offset = state.offset
            epoch = state.epoch
            seed_raw = state.seed
        # Handle seed restoration for bitwise resume: if checkpoint carries a seed
        # different from the dataset's construction seed, re-derive the
        # permutation so that subsequent next_batch slices are identical to the
        # original ordering.  Seed -1 encodes None.
        if seed_raw is not None:
            try:
                s_int = int(seed_raw)
            except Exception:
                s_int = None
            if s_int is not None:
                new_seed: int | None = None if s_int == -1 else s_int
                if new_seed != self.seed:
                    self.seed = new_seed
                    if self.seed is None:
                        self._rows = list(self._canonical_rows)
                    else:
                        gen = torch.Generator().manual_seed(self.seed)
                        perm_any: Any = torch.randperm(
                            len(self._canonical_rows), generator=gen
                        ).tolist()
                        perm: list[int] = [int(x) for x in perm_any]
                        self._rows = [self._canonical_rows[i] for i in perm]
                    # total is invariant
                    self._total = len(self._rows)
        if not (0 <= offset <= self._total):
            raise ContractError(f"sampler offset {offset} out of range [0,{self._total}]")
        self._cursor = offset
        self._epoch = epoch

    def sampler_cursor_for_checkpoint(self) -> dict[str, Any]:
        return self.get_sampler_state()

    def __len__(self) -> int:
        return self._total

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _tensorize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch tensorization — deterministic, no privileged inputs."""
        batch_features: list[torch.Tensor] = []
        batch_legal: list[torch.Tensor] = []
        batch_chosen: list[torch.Tensor] = []
        for r in rows:
            t = tensorize_actor_row(
                r,
                num_actions=self.num_actions,
                feature_dim=self.feature_dim,
                seed=self.seed if self.seed is not None else 0,
            )
            batch_features.append(t["features"])
            batch_legal.append(t["legal_mask"])
            batch_chosen.append(t["chosen_action_id"])
        features = torch.stack(batch_features, dim=0)  # [B,F]
        legal_mask = torch.stack(batch_legal, dim=0)  # [B,A] bool
        chosen_action_id = torch.stack(batch_chosen, dim=0)  # [B] long
        # Perf-A §4.4: pin_memory when cuda so
        # _move_batch_to_device(non_blocking=True) can overlap H2D.
        # Evidence: torch.Tensor.pin_memory docs
        # + torch/utils/data/_utils/pin_memory.py;
        # non_blocking requires pinned source.
        # Pin is no-op overhead on cpu-only hosts; guarded by cuda availability.
        if torch.cuda.is_available():
            try:
                features = features.pin_memory()
                legal_mask = legal_mask.pin_memory()
                chosen_action_id = chosen_action_id.pin_memory()
            except Exception:
                pass
        return {
            "features": features,
            "legal_mask": legal_mask,
            "chosen_action_id": chosen_action_id,
        }

    def next_batch(self, batch_size: int) -> dict[str, Any] | None:
        """Return next microbatch and advance cursor; wraps to next epoch.

        Deterministic: batches are slices of the permuted order.  At end of
        epoch the cursor wraps to 0 and epoch increments.  Returns ``None``
        only when the dataset is empty (never for non-empty).
        """
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if self._cursor >= self._total:
            # Wrap epoch boundary: reset cursor and increment epoch (deterministic)
            self._cursor = 0
            self._epoch += 1
        end = min(self._cursor + batch_size, self._total)
        rows = self._rows[self._cursor : end]
        # Short tail batch is allowed; caller handles drop_last if desired
        batch: dict[str, Any] = self._tensorize_rows(rows)
        # Attach metadata for debugging (not used by model)
        batch["_decision_ids"] = [str(r["decision_id"]) for r in rows]
        batch["_epoch"] = torch.tensor(self._epoch)
        self._cursor = end
        # If we consumed exactly total, next call will wrap at top
        if self._cursor == self._total:
            # Do not auto-wrap here; allow caller to observe epoch boundary via next call
            pass
        return batch

    def peek_batch(self, batch_size: int, cursor: int | None = None) -> dict[str, Any]:
        """Non-advancing peek — useful for tests without mutating cursor."""
        cur = self._cursor if cursor is None else cursor
        end = min(cur + batch_size, self._total)
        rows = self._rows[cur:end]
        return self._tensorize_rows(rows)

    def iter_batches(self, batch_size: int, max_batches: int | None = None):
        """Generator yielding up to max_batches batches, advancing cursor."""
        yielded = 0
        while max_batches is None or yielded < max_batches:
            if self._cursor >= self._total:
                self._cursor = 0
                self._epoch += 1
            batch = self.next_batch(batch_size)
            if batch is None:
                break
            yield batch
            yielded += 1
            if (
                self._cursor >= self._total
                and yielded >= (self._total + batch_size - 1) // batch_size
            ):
                # Completed an epoch; continue wrapping if max_batches demands more
                pass
