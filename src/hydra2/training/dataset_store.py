"""Dataset store: authoritative parquet rows, sampler order, and batching.

Owns the :class:`AuthoritativeParquetDataset` lifecycle shared by the
supervised loop and the actor-learner replay: shard discovery and
privileged-leakage verification, deterministic row loading in canonical
order, the seeded permutation, the stratified rare-action sampler order,
the cursor/epoch resume contract, and microbatch tensorization (synthetic
stand-in by default, the shared real encode path on request). Row parsing
lives in :mod:`hydra2.training.dataset_parse`; tensor math lives in
:mod:`hydra2.training.dataset_encode`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.parquet import ACTOR_FIELDS, FORBIDDEN_IN_ACTOR
from hydra2.training.dataset_encode import (
    _require_action_width,
    encode_observation_rows,
    tensorize_actor_row,
)
from hydra2.training.dataset_parse import (
    _require_actor_parquet_dir,
    _verify_shards,
)

__all__ = [
    "DEFAULT_STRATIFIED_RATIOS",
    "RARE_ACTION_KINDS",
    "AuthoritativeParquetDataset",
    "SamplerState",
    "build_stratified_order",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplerState:
    offset: int
    seed: int
    total: int
    epoch: int = 0


#: Rare-action families oversampled by the stratified sampler (SOTA R5).
#: winning-call kinds.  Ratios are caller-supplied; the r1 recipe uses 3x.
RARE_ACTION_KINDS: tuple[str, ...] = ("daiminkan", "ankan", "kakan", "ron", "tsumo")

#: Starter oversample ratios (SOTA R5 quotes ``2``-``4``x for rare kan/ron;
#: mechanism default when the caller enables stratification without ratios).
DEFAULT_STRATIFIED_RATIOS: dict[str, float] = {
    "daiminkan": 3.0,
    "ankan": 3.0,
    "kakan": 3.0,
    "ron": 3.0,
}


def _row_action_kind(row: dict[str, Any]) -> str:
    """Best-effort action kind for one raw row (sampler bucketing only).

    Explicit ``action_kind``/``decision_kind`` string fields win (the stream
    layer threads these in a follow-up); otherwise the bucket is
    ``"unknown"``.  Never parses observations and never touches the
    ingress/encode path — bucketing labels only.
    """
    for key in ("action_kind", "decision_kind", "kind"):
        value = row.get(key)
        if isinstance(value, str) and value != "":
            return value
    return "unknown"


def _validate_sampling_ratios(ratios: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Strict ``{kind: positive-finite-mult}`` validation (null stays null)."""
    if ratios is None:
        return None
    if not isinstance(ratios, Mapping):
        raise ContractError(
            f"sampling_ratios must be a mapping or null, got {type(ratios).__name__}"
        )
    out: dict[str, float] = {}
    for kind, mult in ratios.items():
        if not isinstance(kind, str) or kind == "":
            raise ContractError("sampling_ratios keys must be non-empty strings")
        if (
            isinstance(mult, bool)
            or not isinstance(mult, (int, float))
            or not (float(mult) == float(mult))
            or float(mult) in (float("inf"), float("-inf"))
            or float(mult) <= 0.0
        ):
            raise ContractError(f"sampling_ratios[{kind!r}] must be positive and finite")
        out[kind] = float(mult)
    return out


def _validate_kind_by_id(kind_by_id: Mapping[str, str] | None) -> dict[str, str] | None:
    """Strict ``{decision_id: kind}`` validation (null stays null)."""
    if kind_by_id is None:
        return None
    if not isinstance(kind_by_id, Mapping):
        raise ContractError(
            f"kind_by_id must be a mapping or null, got {type(kind_by_id).__name__}"
        )
    out: dict[str, str] = {}
    for did, kind in kind_by_id.items():
        if not isinstance(did, str) or did == "":
            raise ContractError("kind_by_id keys must be non-empty decision ids")
        if not isinstance(kind, str) or kind == "":
            raise ContractError(f"kind_by_id[{did!r}] must be a non-empty kind string")
        out[did] = kind
    return out


def build_stratified_order(
    kinds: Sequence[str],
    ratios: Mapping[str, float] | None,
    *,
    seed: int = 0,
) -> list[int]:
    """Deterministic stratified/oversampled row order over ``range(len(kinds))``.

    Each kind ``k`` with count ``c`` and ratio ``r`` (default ``1.0``)
    contributes ``max(1, round(c * r))`` positions (cycled deterministically
    through its rows when ``r > 1``); kinds interleave round-robin in sorted
    kind order so rare kinds spread evenly through the epoch instead of
    clustering.  ``seed`` is accepted for API symmetry with the cursor
    sampler (the construction is fully determined by ``(kinds, ratios)``;
    the caller's base permutation already carries the seed entropy).
    """
    kind_list = list(kinds)
    for kind in kind_list:
        if not isinstance(kind, str) or kind == "":
            raise ContractError(f"stratified kinds must be non-empty strings, got {kind!r}")
    resolved = _validate_sampling_ratios(ratios if ratios is not None else {})
    by_kind: dict[str, list[int]] = {}
    for pos, kind in enumerate(kind_list):
        by_kind.setdefault(kind, []).append(pos)
    expanded: dict[str, list[int]] = {}
    for kind in sorted(by_kind):
        members = by_kind[kind]
        ratio = (resolved if resolved is not None else {}).get(kind, 1.0)
        target = max(1, round(len(members) * ratio))
        expanded[kind] = [members[i % len(members)] for i in range(target)]
    order: list[int] = []
    pending = True
    cursor = 0
    ordered_kinds = sorted(expanded)
    while pending:
        pending = False
        for kind in ordered_kinds:
            members = expanded[kind]
            if cursor < len(members):
                order.append(members[cursor])
                pending = True
        cursor += 1
    _ = seed  # documented no-op: determinism comes from (kinds, ratios)
    return order


class AuthoritativeParquetDataset:
    """Deterministic authoritative parquet dataset for WP-05B.

    Args:
        parquet_dir: directory containing ``actor-*.parquet`` shards written
            by :func:`hydra2.data.parquet.write_actor_shards`.
        feature_dim: synthetic feature dimensionality (used when observation
            is not pre-tensorized).  In ``'real'`` mode it sets the folded
            real-feature width.
        num_actions: canonical action vocab size.  Defaults to the frozen
            action table size (6792) when ``None`` is passed; any other value
            is test-only and requires ``allow_narrow=True``.  In ``'real'``
            mode a narrow value slices the observation's own legal mask.
        allow_narrow: test-only flag permitting ``num_actions != 6792``
            (with modulo-remapped labels).  Production defaults to ``False``.
        seed: deterministic shuffle seed.  ``None`` disables shuffling
            (canonical lexicographic order).  When set, the permutation is
            computed once from the seed and the cursor tracks offset into
            the permuted order.
        verify: when ``True`` (default) shard verification runs on init;
            ``False`` skips verification for unit probes that inject bad rows
            directly.
        tensorize: ``'synthetic'`` (default) keeps the deterministic
            decision_id-hash stand-in; ``'real'`` parses ``actor_observation``
            JSON through :func:`encode_observations` and raises
            :class:`ContractError` on unparseable rows (never hash-falls-back).
        stratified: when ``True`` the sampler oversamples rare action kinds
            per ``sampling_ratios`` (deterministic interleaved order, same
            cursor/epoch resume contract over the expanded order).
            ``False`` (default) preserves the canonical cursor order
            bit-identically.
        sampling_ratios: ``{kind: multiplier}`` oversample map (values MUST
            be positive and finite); ``None`` with ``stratified=True`` uses
            :data:`DEFAULT_STRATIFIED_RATIOS` (kan/ron families ``3x``).
        kind_by_id: optional ``{decision_id: kind}`` bucket labels (explicit
            wins; rows missing from the map fall back to the row's own
            ``action_kind``/``decision_kind`` field, else ``"unknown"``).
            Batches carry the resolved labels under ``"_action_kinds"`` for
            per-type scorecards.
    """

    def __init__(
        self,
        *,
        parquet_dir: Path,
        feature_dim: int = 16,
        num_actions: int | None = 6792,
        seed: int | None = 0,
        verify: bool = True,
        tensorize: Literal["synthetic", "real"] = "synthetic",
        stratified: bool = False,
        sampling_ratios: Mapping[str, float] | None = None,
        kind_by_id: Mapping[str, str] | None = None,
        allow_narrow: bool = False,
    ) -> None:
        if tensorize not in ("synthetic", "real"):
            raise ContractError(f"tensorize must be 'synthetic' or 'real', got {tensorize!r}")
        self.tensorize: Literal["synthetic", "real"] = tensorize
        self.parquet_dir = Path(parquet_dir)
        self.feature_dim = feature_dim
        self.num_actions = num_actions if num_actions is not None else 6792
        _require_action_width(
            self.num_actions, allow_narrow=allow_narrow, where="AuthoritativeParquetDataset"
        )
        self.allow_narrow = allow_narrow
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
        if not isinstance(stratified, bool):
            raise ContractError(f"stratified must be a bool, got {stratified!r}")
        self._stratified: bool = stratified
        self._sampling_ratios: dict[str, float] | None = _validate_sampling_ratios(sampling_ratios)
        self._kind_by_id: dict[str, str] | None = _validate_kind_by_id(kind_by_id)
        self._kinds: list[str] | None = None
        self._order: list[int] = []
        self._total: int = 0
        self._rebuild_sampler_order()

    def _rebuild_sampler_order(self) -> None:
        """(Re)build kinds + order after (re)permutation; total follows order.

        Default (non-stratified, no labels): identity order, kinds ``None``,
        total ``len(rows)`` — bit-identical to the pre-stratification path.
        Stratified: kinds resolve per row (``kind_by_id`` wins, else the
        row's own kind field, else ``"unknown"``) and the order expands per
        ``sampling_ratios`` (or :data:`DEFAULT_STRATIFIED_RATIOS`).
        """
        if self._kind_by_id is not None or self._stratified:
            by_id = self._kind_by_id
            self._kinds = [
                by_id.get(str(r.get("decision_id", "")), _row_action_kind(r))
                if by_id is not None
                else _row_action_kind(r)
                for r in self._rows
            ]
        else:
            self._kinds = None
        if self._stratified:
            assert self._kinds is not None
            if self._sampling_ratios is not None and len(self._sampling_ratios) > 0:
                ratios = self._sampling_ratios
            else:
                ratios = dict(DEFAULT_STRATIFIED_RATIOS)
            self._order = build_stratified_order(
                self._kinds, ratios, seed=self.seed if self.seed is not None else 0
            )
        else:
            self._order = list(range(len(self._rows)))
        self._total = len(self._order)

    # ------------------------------------------------------------------
    # Cursor / sampler state
    # ------------------------------------------------------------------

    def get_sampler_state(self) -> dict[str, Any]:
        return {
            "offset": self._cursor,
            "seed": -1 if self.seed is None else self.seed,
            "total": self._total,
            "epoch": self._epoch,
            "stratified": self._stratified,
            "sampling_ratios": None
            if self._sampling_ratios is None
            else dict(self._sampling_ratios),
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
        reseeded = False
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
                    reseeded = True
        if isinstance(state, dict):
            # Checkpoints written with stratification carry it; older states
            # (or SamplerState) keep this object's construction settings.
            if "stratified" in state and state["stratified"] is not None:
                s_new = state["stratified"]
                if not isinstance(s_new, bool):
                    raise ContractError(f"sampler stratified must be a bool, got {s_new!r}")
                if s_new != self._stratified:
                    self._stratified = s_new
                    reseeded = True
            if "sampling_ratios" in state:
                r_new = _validate_sampling_ratios(state["sampling_ratios"])
                if r_new != self._sampling_ratios:
                    self._sampling_ratios = r_new
                    reseeded = True
        if reseeded:
            # Order (and total) follows rows + stratified settings
            self._rebuild_sampler_order()
        if not (0 <= offset <= self._total):
            raise ContractError(f"sampler offset {offset} out of range [0,{self._total}]")
        self._cursor = offset
        self._epoch = epoch

    def __len__(self) -> int:
        return self._total

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _tensorize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch tensorization — deterministic, no privileged inputs.

        ``'synthetic'`` (default) uses the decision_id-hash stand-in and never
        carries ``actor_batch``; ``'real'`` parses ``actor_observation`` JSON
        through the real encoder (fail-closed :class:`ContractError`, never
        hash-falls-back) and also carries the encoded
        :class:`ActorTensorBatch` under ``actor_batch`` (flat keys
        byte-identical for compat).
        """
        if self.tensorize == "real":
            return encode_observation_rows(
                rows,
                num_actions=self.num_actions,
                feature_dim=self.feature_dim,
                allow_narrow=self.allow_narrow,
            )
        batch_features: list[torch.Tensor] = []
        batch_legal: list[torch.Tensor] = []
        batch_chosen: list[torch.Tensor] = []
        for r in rows:
            t = tensorize_actor_row(
                r,
                num_actions=self.num_actions,
                feature_dim=self.feature_dim,
                seed=self.seed if self.seed is not None else 0,
                allow_narrow=self.allow_narrow,
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
            except Exception as exc:
                logger.warning("dataset pin_memory failed, using pageable fallback: %s", exc)
        return {
            "features": features,
            "legal_mask": legal_mask,
            "chosen_action_id": chosen_action_id,
        }

    def next_batch(self, batch_size: int) -> dict[str, Any] | None:
        """Return next microbatch and advance cursor; wraps to next epoch.

        Deterministic: batches are slices of the sampler order (the seeded
        permutation, or the stratified interleaved order when enabled).  At
        end of epoch the cursor wraps to 0 and epoch increments.
        Returns ``None`` only when the dataset is empty (never for non-empty).
        """
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if self._cursor >= self._total:
            # Wrap epoch boundary: reset cursor and increment epoch (deterministic)
            self._cursor = 0
            self._epoch += 1
        end = min(self._cursor + batch_size, self._total)
        order_slice = self._order[self._cursor : end]
        rows = [self._rows[i] for i in order_slice]
        # Short tail batch is allowed; caller handles drop_last if desired
        batch: dict[str, Any] = self._tensorize_rows(rows)
        # Attach metadata for debugging (not used by model)
        batch["_decision_ids"] = [str(r["decision_id"]) for r in rows]
        batch["_epoch"] = torch.tensor(self._epoch)
        if self._kinds is not None:
            # Per-type scorecard labels: "_" prefix keeps the loop's H2D
            # mover passing them through untouched (never tensorized).
            batch["_action_kinds"] = [self._kinds[i] for i in order_slice]
        self._cursor = end
        # If we consumed exactly total, next call will wrap at top
        if self._cursor == self._total:
            # Do not auto-wrap here; allow caller to observe epoch boundary via next call
            pass
        return batch

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
