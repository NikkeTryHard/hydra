"""Dataset buffered window: takes, snapshots, restore, sidecar validation.

Owns the :class:`_StreamDatasetBufferMixin` buffered-window half of the
streaming dataset (bucket-grouped takes, microbatch encode, sampler
state, whole-game snapshot/restore with verify-before-mutate) plus the
strict sidecar/payload validators the resume gates run. Tamper or drift
raises fail-closed; live state mutates only after verification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.replay_expand import expand_privileged_rows as expand_privileged_rows
from hydra2.training.stream_dataset import _StreamDatasetCore as _StreamDatasetCore
from hydra2.training.stream_expand import _expand_game_planes as _expand_game_planes
from hydra2.training.stream_expand import _history_bucket_of as _history_bucket_of
from hydra2.training.stream_expand import _sidecar_window_hash as _sidecar_window_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "_StreamDataset",
    "_StreamDatasetBufferMixin",
    "_parse_dataset_buffer_sidecar",
    "_verify_fast_snapshots",
]


class _StreamDatasetBufferMixin(_StreamDatasetCore):
    """Buffered-window half of the streaming dataset (takes/snapshots/restore)."""

    _rows: list[dict[str, Any]]
    _offset: int
    _dropped: int
    _microbatches_in_epoch: int
    _buffered_entries: list[dict[str, Any]]
    _seed: int
    _replay_backend: str
    _need_privileged: bool
    _homogeneous_buckets: bool
    _pack_histories: bool
    _num_actions: int
    _feature_dim: int
    _epoch: int
    privileged: dict[str, dict[str, Any]]
    replayed: int
    sim_replayed: int
    expand_quarantined: int
    expand_quarantine_reasons: dict[str, int]
    pin_memory: bool

    def _group_unconsumed_by_bucket(self, start: int) -> None:
        """Stably partition the unconsumed window by history bucket (in place).

        Sort key is ``(bucket, global arrival index)``: equal buckets keep
        pull order, so default-off runs are untouched and reruns reproduce
        bit-for-bit. Consumed rows always tile ``_rows[0:start]`` (grouping
        never moves them; takes extend the consumed prefix contiguously),
        so take slicing plus :meth:`_compact` counts stay exact and only
        whole consumed games are ever reclaimed. Resume needs no snapshot
        *format* change: the flag itself rides the run digest (flipping it
        fails closed on drift before any state is applied), and the live
        layout rides the optional ``row_order`` snapshot key (absent when
        the flag is off, so old snapshots restore via the pre-``row_order``
        snapshot path untouched) — takes stay contiguous-prefix advances
        (``_rows``, ``_offset``), so sampler offset/hash/counter semantics
        are unchanged. Whole-game entry alignment degrades gracefully (a
        grouped take may strand partial games; bounded by pull size).
        """
        window = self._rows[start:]
        if len(window) < 2:
            return
        keys = [_history_bucket_of(row) for row in window]
        if all(key == keys[0] for key in keys):
            return
        base = self._dropped + start

        def _order_key(i: int) -> tuple[int, int]:
            return (keys[i], base + i)

        order = sorted(range(len(window)), key=_order_key)
        self._rows[start:] = [window[i] for i in order]

    def _consume_microbatch(self, batch_size: int) -> list[dict[str, Any]]:
        """Advance exactly one microbatch through the fill machine."""
        self._fill(batch_size)
        start = self._offset - self._dropped
        if self._homogeneous_buckets:
            # Stable bucket-grouped take: the contiguous-prefix take below
            # then carries a single bucket (batch pads to one bucket ceil
            # instead of the batch max). Default False preserves
            # byte-identical pre-``row_order`` snapshot order.
            self._group_unconsumed_by_bucket(start)
        taken = self._rows[start : start + batch_size]
        self._offset += len(taken)
        self._microbatches_in_epoch += 1
        self._compact()
        return taken

    def next_batch(self, batch_size: int) -> dict[str, Any]:
        """Assemble and return the next microbatch, advancing the cursor."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ContractError(f"batch_size must be a positive int, got {batch_size!r}")
        taken = self._consume_microbatch(batch_size)
        if len(taken) == 0:
            raise ContractError("stream dataset produced an empty microbatch")
        return self._next_batch_planes(taken)

    def _next_batch_planes(self, taken: list[dict[str, Any]]) -> dict[str, Any]:
        """Assemble one microbatch from Rust plane blobs (no encoder).

        Groups consecutive rows sharing one game blob (buffer order is game
        order), slices zero-copy plane views, concats across games, and
        finishes through :func:`assemble_slim_batch` (shared with eval).
        Decision ids, kinds, and epoch ride the slim rows, so scorecards
        and joins behave identically to the encoder path.
        """
        from hydra2.training.rust_batch import assemble_slim_batch

        batch = assemble_slim_batch(
            taken,
            action_count=self._num_actions,
            pack_histories=bool(getattr(self, "_pack_histories", False)),
        )
        batch["_epoch"] = torch.tensor(self._epoch)
        return batch

    def get_sampler_state(self) -> dict[str, Any]:
        return {
            "offset": self._offset,
            "seed": self._seed,
            "total": len(self._rows),
            "epoch": self._epoch,
            "dropped": self._dropped,
        }

    def set_sampler_state(self, state: Any) -> None:
        if isinstance(state, dict):
            mapping: dict[str, Any] = state
            offset: int = mapping.get("offset", 0)
            epoch: int = mapping.get("epoch", 0)
        else:
            offset = getattr(state, "offset", 0)
            epoch = getattr(state, "epoch", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ContractError(f"sampler offset invalid: {offset!r}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ContractError(f"sampler epoch must be a non-negative int, got {epoch!r}")
        if offset < self._dropped:
            raise ContractError(
                f"sampler offset {offset} precedes compacted prefix {self._dropped}"
            )
        if offset > len(self._rows) + self._dropped:
            raise ContractError(f"sampler offset {offset} beyond buffered rows {len(self._rows)}")
        self._offset = offset
        self._epoch = epoch

    def buffered_row_hash(self) -> str:
        """Game-keyed sidecar hash over the live window (K4: no per-row id strings)."""
        return _sidecar_window_hash(self._buffered_entries, self._rows)

    def buffer_snapshot(self) -> dict[str, Any]:
        """Fast-resume snapshot: whole-game entries + counters + row hash.

        Bridge-compat shape (checked by ``hydra-bridge`` ``resume``):
        ``entries`` carry exactly ``{key, path, offset, split, rows}``
        (``check_buffer_entries``); the RNG triple ``{key, block, pos}``
        (``feed::rng::StreamSnapshot``) rides the shuffle snapshot
        (``buffer_rng_state``), not these entries. Snapshot envelope
        versioning is bridge-owned (writes v2, dual-reads v1 drain-only
        for one release).
        """
        total = sum(int(entry["rows"]) for entry in self._buffered_entries)
        if total != len(self._rows):
            raise ContractError(
                f"buffer index drift: {total} indexed rows != {len(self._rows)} buffered"
            )
        snap: dict[str, Any] = {
            "entries": [dict(entry) for entry in self._buffered_entries],
            "offset": self._offset,
            "dropped": self._dropped,
            "epoch": self._epoch,
            "microbatches_in_epoch": self._microbatches_in_epoch,
            "replayed": self.replayed,
            "sim_replayed": self.sim_replayed,
            "expand_quarantined": self.expand_quarantined,
            "expand_quarantine_reasons": dict(self.expand_quarantine_reasons),
            "row_hash": self.buffered_row_hash(),
            "total_rows": len(self._rows),
            "replay_backend": self._replay_backend,
        }
        if self._homogeneous_buckets:
            # Grouped-order history: takes consume grouped (non-pull-order)
            # prefixes, so (offset, entries) alone underdetermines the live
            # layout — snapshot the decision_id permutation to restore it
            # exactly. Optional key (absent when the flag is off): old
            # snapshots restore via the pre-``row_order`` snapshot path untouched.
            snap["row_order"] = [str(row["decision_id"]) for row in self._rows]
        return snap

    def restore_buffer(self, snapshot: Mapping[str, Any]) -> None:
        """Rebuild ``_rows`` verbatim from snapshot entries (fail-closed).
        Re-expands each buffered game through the identical serial entry points
        as live pulls, reorders to the optional ``row_order`` permutation when
        present (grouped layouts consume non-pull-order prefixes, so offset
        alone underdetermines them; absent key keeps the legacy pull-order
        path), then verifies length + row hash before restoring logical
        counters. Only the buffered tail is re-expanded (``O(buffer)``), never
        the epoch.
        """
        from hydra2.data.stream_read import fetch_game_at as _fetch

        if not isinstance(snapshot, dict):
            raise ContractError("dataset buffer snapshot must be a mapping")
        entries = snapshot.get("entries")
        if not isinstance(entries, list):
            raise ContractError("dataset buffer entries must be a list")
        recorded_backend = snapshot.get("replay_backend", "rust")
        if recorded_backend != self._replay_backend:
            raise ContractError(
                f"dataset buffer replay_backend {recorded_backend!r} != "
                f"live backend {self._replay_backend!r} (cross-backend restore refused)"
            )
        for raw_entry in entries:
            entry: dict[str, Any] = raw_entry
            if not isinstance(entry, dict):
                raise ContractError("dataset buffer entry must be a mapping")
            unknown = sorted(
                k for k in entry if k not in ("key", "path", "offset", "split", "rows")
            )
            if len(unknown) > 0:
                raise ContractError(f"dataset buffer entry unknown keys {unknown}")
            field_key, field_path, field_offset, field_split, field_rows = (
                entry.get("key"),
                entry.get("path"),
                entry.get("offset"),
                entry.get("split"),
                entry.get("rows"),
            )
            if not isinstance(field_key, str) or field_key == "":
                raise ContractError("dataset buffer entry key must be a non-empty str")
            if not isinstance(field_path, str) or field_path == "":
                raise ContractError("dataset buffer entry path must be a non-empty str")
            if (
                isinstance(field_offset, bool)
                or not isinstance(field_offset, int)
                or field_offset < 0
            ):
                raise ContractError("dataset buffer entry offset must be non-negative int")
            if not isinstance(field_split, str) or field_split == "":
                raise ContractError("dataset buffer entry split must be a non-empty str")
            if isinstance(field_rows, bool) or not isinstance(field_rows, int) or field_rows <= 0:
                raise ContractError("dataset buffer entry rows must be a positive int")
        offset: Any = snapshot.get("offset")
        dropped: Any = snapshot.get("dropped")
        epoch: Any = snapshot.get("epoch")
        micro: Any = snapshot.get("microbatches_in_epoch")
        for name, value in (
            ("offset", offset),
            ("dropped", dropped),
            ("epoch", epoch),
            ("microbatches_in_epoch", micro),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"dataset buffer {name} must be a non-negative int")
        expected_hash = snapshot.get("row_hash")
        if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256:"):
            raise ContractError("dataset buffer row_hash must be a sha256 string")
        rebuilt_rows: list[dict[str, Any]] = []
        rebuilt_priv: dict[str, dict[str, Any]] = {}
        replayed = 0
        sim_replayed = 0
        for raw_entry in entries:
            entry: dict[str, Any] = raw_entry
            recorded_split: str = entry["split"]
            fetch_path: str = entry["path"]
            fetch_offset: int = entry["offset"]
            fetch_key: str = entry["key"]
            fetched = _fetch(
                fetch_path,
                fetch_offset,
                seed=self._seed,
                ratios={recorded_split: 1.0},
                expected_sha=fetch_key,
            )
            try:
                row_dicts, sim_path = _expand_game_planes(fetched.game, recorded_split, fetched.raw)
                priv_rows = (
                    expand_privileged_rows(fetched.game, split=recorded_split)
                    if self._need_privileged
                    else []
                )
            except ContractError as exc:
                raise ContractError(f"buffered game failed to re-expand: {exc}") from exc
            if len(row_dicts) != int(entry["rows"]):
                raise ContractError("buffered game row count mismatch on restore")
            rebuilt_rows.extend(row_dicts)
            for priv in priv_rows:
                label = dict(priv.privileged_label)
                # intentionally discarded: existing label wins
                _ = rebuilt_priv.setdefault(str(priv.decision_id), label)
            if sim_path:
                sim_replayed += 1
            else:
                replayed += 1
        if (order := snapshot.get("row_order")) is not None:
            # Grouped-order restore: re-expansion yields pull order, but the
            # live layout may be grouped (takes consume non-pull-order
            # prefixes). Reorder the rebuilt rows to the snapshotted
            # decision_id permutation before the hash check, so restore is
            # bit-exact and subsequent takes match the uninterrupted run.
            # Absent key (old snapshots, flag-off runs): pre-``row_order``
            # snapshot path. Any id/count mismatch fails closed; content tamper
            # still fails on the positional row hash below.
            if not isinstance(order, list) or any(not isinstance(v, str) for v in order):
                raise ContractError("dataset buffer row_order must be a list of str")
            if len(order) != len(rebuilt_rows):
                raise ContractError("dataset buffer row_order length mismatch on restore")
            slots: dict[str, list[int]] = {}
            for pos, row in enumerate(rebuilt_rows):
                did = row.get("decision_id")
                if not isinstance(did, str):
                    raise ContractError("dataset buffer row lacks a decision_id on restore")
                slots.setdefault(did, []).append(pos)
            perm: list[int] = []
            for did in order:
                queue: list[int] | None = slots.get(did)  # pyrefly: ignore[unknown-argument-type] # row_order ids dynamic; slots owns the mapping
                if queue is None or len(queue) == 0:
                    raise ContractError(
                        "dataset buffer row_order references an unknown decision id on restore"
                    )
                perm.append(queue.pop(0))
            if any(queue for queue in slots.values()):
                raise ContractError("dataset buffer row_order omits buffered rows on restore")
            rebuilt_rows = [rebuilt_rows[pos] for pos in perm]
        # Verify verbatim before mutating live state (game-keyed sidecar hash).
        typed_entries: list[dict[str, Any]] = [dict(entry) for entry in entries]  # pyrefly: ignore[unknown-argument-type] # snapshot entries dynamic; dict owns the copy
        actual_hash = _sidecar_window_hash(typed_entries, rebuilt_rows)
        if actual_hash != expected_hash:
            raise ContractError("dataset buffer row hash mismatch on restore")
        if len(rebuilt_rows) != int(snapshot.get("total_rows", len(rebuilt_rows))):
            raise ContractError("dataset buffer total_rows mismatch on restore")
        if not (dropped <= offset <= dropped + len(rebuilt_rows)):  # type: ignore[operator]
            raise ContractError("dataset buffer offset outside live window on restore")
        # Commit (verify-then-mutate): rows + privileged + index + counters.
        self._rows = rebuilt_rows
        self.privileged = rebuilt_priv
        self._buffered_entries = [dict(entry) for entry in typed_entries]
        self._offset = int(offset)  # type: ignore[arg-type]
        self._dropped = int(dropped)  # type: ignore[arg-type]
        self._microbatches_in_epoch = int(micro)  # type: ignore[arg-type]
        self._epoch = int(epoch)  # type: ignore[arg-type]
        # Prefix totals (fail-closed when absent/malformed; never recomputed).
        for name in ("replayed", "sim_replayed", "expand_quarantined"):
            value = snapshot.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"dataset buffer {name} must be a non-negative int")
        reasons = snapshot.get("expand_quarantine_reasons")
        if not isinstance(reasons, dict):
            raise ContractError("dataset buffer quarantine reasons must be a mapping")
        self.replayed = int(snapshot["replayed"])  # type: ignore[arg-type]
        self.sim_replayed = int(snapshot["sim_replayed"])  # type: ignore[arg-type]
        self.expand_quarantined = int(snapshot["expand_quarantined"])  # type: ignore[arg-type]
        self.expand_quarantine_reasons = {str(k): int(v) for k, v in reasons.items()}  # pyrefly: ignore[unknown-argument-type] # snapshot reasons dynamic; str/int own the gates

    @property
    def cursor(self) -> int:
        return self._offset

    def __len__(self) -> int:
        return len(self._rows)


def _verify_fast_snapshots(*, sidecar: dict[str, Any], payload: Any, ckpt: Path) -> None:
    """Cross-check sidecar vs payload fast snapshots (tamper → raise)."""
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload must be a mapping: {ckpt}")
    side_dataset = sidecar.get("dataset_buffer")
    payload_dataset = payload.get("dataset_buffer")
    if not isinstance(side_dataset, dict) or not isinstance(payload_dataset, dict):
        raise ContractError(f"checkpoint fast snapshots missing: {ckpt}")
    if side_dataset != payload_dataset:
        raise ContractError(f"checkpoint dataset buffer mismatch (tampered?): {ckpt}")
    side_shuffle = sidecar.get("shuffle")
    payload_shuffle = payload.get("shuffle_buffer")
    if not isinstance(side_shuffle, dict) or not isinstance(payload_shuffle, dict):
        raise ContractError(f"checkpoint shuffle snapshots missing: {ckpt}")
    for key in ("buffer_keys", "buffer_rng_state", "epoch_seed", "buffer_size", "prefix_hashes"):
        if side_shuffle.get(key) != payload_shuffle.get(key):
            raise ContractError(f"checkpoint shuffle {key} mismatch (tampered?): {ckpt}")
    side_entries: list[Any] = side_shuffle.get("buffer_entries", [])
    payload_entries: list[Any] = payload_shuffle.get("buffer_entries", [])
    if side_entries != payload_entries:
        raise ContractError(f"checkpoint shuffle entries mismatch (tampered?): {ckpt}")


def _parse_dataset_buffer_sidecar(raw: Any, *, ckpt: Path) -> dict[str, Any]:
    """Strict validation of the sidecar ``dataset_buffer`` (tamper → raise)."""
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint dataset_buffer must be a mapping: {ckpt}")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ContractError(f"checkpoint dataset_buffer entries must be a list: {ckpt}")
    for raw_entry in entries:
        entry: dict[str, Any] = raw_entry
        if not isinstance(entry, dict):
            raise ContractError(f"checkpoint dataset_buffer entry must be a mapping: {ckpt}")
        unknown = sorted(k for k in entry if k not in ("key", "path", "offset", "split", "rows"))
        if len(unknown) > 0:
            raise ContractError(f"checkpoint dataset_buffer entry unknown keys {unknown}: {ckpt}")
    for name in ("offset", "dropped", "epoch", "microbatches_in_epoch", "total_rows"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"checkpoint dataset_buffer {name} invalid: {ckpt}")
    for name in ("replayed", "sim_replayed", "expand_quarantined"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"checkpoint dataset_buffer {name} invalid: {ckpt}")
    reasons = raw.get("expand_quarantine_reasons")
    if not isinstance(reasons, dict):
        raise ContractError(f"checkpoint dataset_buffer quarantine reasons invalid: {ckpt}")
    row_hash = raw.get("row_hash")
    if not isinstance(row_hash, str) or not row_hash.startswith("sha256:"):
        raise ContractError(f"checkpoint dataset_buffer row_hash invalid: {ckpt}")
    # Optional grouped-order permutation (homogeneous takes only; absent in
    # old snapshots and flag-off runs, which restore via the pre-``row_order`` path).
    if (row_order := raw.get("row_order")) is not None and (
        not isinstance(row_order, list) or any(not isinstance(v, str) for v in row_order)
    ):
        raise ContractError(f"checkpoint dataset_buffer row_order invalid: {ckpt}")
    exhausted: bool = raw.get("stream_exhausted", False)
    if not isinstance(exhausted, bool):
        raise ContractError(f"checkpoint dataset_buffer stream_exhausted invalid: {ckpt}")
    return dict(raw)


class _StreamDataset(_StreamDatasetBufferMixin):
    """Loop-facing microbatch source over a lazily-pulled game stream.

    Exposes the :class:`SupervisedLoop` dataset surface (``next_batch``,
    ``get_sampler_state``/``set_sampler_state``, ``__len__``, ``cursor``).
    Games are pulled on demand, expanded to slim plane rows by the Rust walk,
    and assembled per microbatch; privileged rows ride along into
    :attr:`privileged` (train-split rows only).
    Multi-epoch wrap: the live epoch starts at 0 and rolls forward when the
    corpus is exhausted while rows are still demanded (only an empty split
    fails closed); resume seeks to the recorded frontier at the recorded
    epoch with verbatim buffer + RNG + dedup-prefix restore.
    """
