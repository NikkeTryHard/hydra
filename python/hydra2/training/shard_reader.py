"""Phase-4 offline shard reader: mmap feed over builder tensor shards.

Consumes the builder manifest (owned by ``data/shard_build.py``) READ-ONLY.
Manifest contract (final, builder-owned)::

    {version: 1,
     planes: [{name, dtype, storage_dtype, shape, file, sha256, chunk}],
     rows, schema_digest, dataset_hash, split,
     order: {kind: canonical | perm, seed},
     attestation, build: {...}}

Storage reality (builder pin):

- One manifest entry per ``(plane, chunk)``; chunk rows concatenate in
  ``chunk`` order to ``rows`` total. Single-chunk planes carry ``chunk: 0``.
- Each plane file is one ``pa.ipc.write_tensor`` payload, read via
  ``pa.ipc.read_tensor(pa.memory_map(path))`` — the mmap stays file-backed,
  only taken batch rows copy out. IPC-mmap is FIRST (the planes above ARE
  the read path): ``.npy``/``.bin`` memmap is the fallback alternate, never
  first (bridge release: canon_rng/columnar/search/ring/resume built;
  feed 172 + shard 311 + search 119 green).
- Bool planes are stored ``uint8`` (``storage_dtype``) and viewed
  ``.view(bool)`` zero-copy; ``dtype`` is the logical dtype.
- ``decision_ids.json`` / ``observation_hashes.json`` sidecars list one
  entry per row in final row order.
- ``dataset_hash == "sha256:" + sha256("\\n".join(sorted_ids))`` (single LF,
  no trailing newline, utf-8); re-verified on open when the sidecar exists.

Raw ``.npy`` / ``.bin`` planes (same envelope) are tolerated as alternates;
fixtures use them for the small-vocab shape, production uses tensor IPC.

Feed model (MosaicML deterministic-resumption style):

- Canonical order is the on-disk row order when ``order.kind`` is
  ``canonical``. A constructor ``seed`` applies one fixed ``torch.randperm``
  (same idiom as :class:`AuthoritativeParquetDataset`), fixed across epochs.
- When ``order.kind`` is ``perm`` the on-disk order already is the seeded
  perm: a matching/``None`` seed reads it as-is; a *different* seed recovers
  canonical via the ``decision_ids`` sidecar (argsort) and re-permutes, else
  raises (never double-permutes silently).
- The resume cursor is ``{epoch, sample_in_epoch}``: ``set_state`` seeks the
  single sequential producer to that position — never replays prior samples.
- Bucket-homogeneous batches: rows bucket by history length (``history_len``
  plane, else ``history_mask`` row sums) into ``HISTORY_BUCKET_LENGTHS``;
  each batch holds one bucket and history-width planes are column-sliced to
  the bucket width.
- One background producer thread feeds a ``queue.Queue`` of depth >= 2.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.models.schema import BASELINE_ACTION_COUNT, HISTORY_BUCKET_LENGTHS

try:
    from hydra2._native import contracts as _shard_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _shard_bridge = None  # type: ignore[assignment]

__all__ = [
    "MANIFEST_VERSION",
    "MIN_PREFETCH_DEPTH",
    "ShardReader",
]

#: Manifest version this reader accepts (builder contract, final).
MANIFEST_VERSION = 1

#: Minimum producer queue depth (ticket acceptance: depth>=2 queue).
MIN_PREFETCH_DEPTH = 2

_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "version",
        "planes",
        "rows",
        "schema_digest",
        "dataset_hash",
        "split",
        "order",
        "attestation",
    }
)

#: Plane entries always carry these; ``dtype``/``storage_dtype`` need only
#: one present (logical defaults to storage and vice versa); ``chunk``
#: defaults to 0. Extra keys are ignored.
_REQUIRED_PLANE_KEYS = frozenset({"name", "shape", "file", "sha256"})

_IPC_SUFFIXES = frozenset({".ipc", ".arrow", ".arrow.ipc", ".tensor"})
_NPY_SUFFIXES = frozenset({".npy"})
_RAW_SUFFIXES = frozenset({".bin", ".raw", ".dat"})


#: Bridge leaf for the plane-suffix gate (single source:
#: ``hydra2._native.contracts`` ``plane_suffix``; the stale-.so fallback is
#: the byte-identical oracle below).
_plane_suffix_bridge = getattr(_shard_bridge, "plane_suffix", None)


def _plane_suffix(filename: str) -> str:
    if _plane_suffix_bridge is not None:
        try:
            suffix: str = _plane_suffix_bridge(filename)
            return suffix
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    lowered = filename.lower()
    for suffix in (".arrow.ipc", ".ipc", ".arrow", ".tensor", ".npy", ".bin", ".raw", ".dat"):
        if lowered.endswith(suffix):
            return suffix
    raise ContractError(f"plane file {filename!r} has unknown storage suffix")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_hex(digest: str) -> str:
    text = digest.strip().lower()
    return text.removeprefix("sha256:")


def _bucket_ceil(length: int, buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]


def _canonical_perm(seed: int, rows: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    perm: list[int] = torch.randperm(rows, generator=gen).tolist()
    return perm


def dataset_hash_of(decision_ids: list[str]) -> str:
    """Builder-pinned dataset hash: ``sha256:`` + hex over LF-joined sorted ids."""
    joined = "\n".join(sorted(decision_ids)).encode("utf-8")
    return "sha256:" + hashlib.sha256(joined).hexdigest()


class _NpyPlane:
    """Read-only numpy plane (``.npy`` mmap or raw ``.bin`` memmap)."""

    def __init__(self, path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> None:
        suffix = _plane_suffix(path.name)
        if suffix in _NPY_SUFFIXES:
            arr = np.load(path, mmap_mode="r")
            if arr.dtype != dtype or arr.shape != shape:
                raise CorruptArtifactError(
                    f"plane {path.name} on-disk {(arr.dtype, arr.shape)} != "
                    f"manifest {(dtype, shape)}"
                )
            self._arr: np.ndarray[Any, Any] = arr
        elif suffix in _RAW_SUFFIXES:
            self._arr = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        else:
            raise ContractError(f"plane {path.name} is not a numpy-backed plane")
        self._arr.flags.writeable = False

    def close(self) -> None:
        del self._arr

    def take_rows(self, idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.ascontiguousarray(self._arr[idx])


class _TensorIpcPlane:
    """Read-only tensor-IPC plane, possibly multi-chunk.

    Each chunk file is one ``pa.ipc.write_tensor`` payload kept
    memory-mapped; chunk arrays are zero-copy numpy views over the mmap
    (bool stays a ``uint8`` view until served). ``take_rows`` copies only the
    requested global rows, preserving index order across chunk edges.
    """

    def __init__(
        self,
        files: list[Path],
        *,
        name: str,
        storage_dtype: np.dtype[Any],
        logical_dtype: np.dtype[Any],
        chunk_rows: list[int],
        tail_shape: tuple[int, ...],
    ) -> None:
        self._name = name
        self._logical_dtype = logical_dtype
        self._is_bool_view = logical_dtype == np.dtype(np.bool_) and storage_dtype != logical_dtype
        if self._is_bool_view and storage_dtype != np.dtype(np.uint8):
            raise ContractError(
                f"plane {name!r} bool view needs uint8 storage, got {storage_dtype}"
            )
        self._mmaps: list[Any] = []
        self._chunks: list[tuple[int, np.ndarray[Any, Any]]] = []
        offset = 0
        for path, nrows in zip(files, chunk_rows, strict=True):
            mmap = pa.memory_map(str(path), "r")
            self._mmaps.append(mmap)
            tensor = pa_ipc.read_tensor(mmap)
            raw: np.ndarray[Any, Any] = np.ascontiguousarray(tensor.to_numpy())
            if tuple(raw.shape) != (nrows, *tail_shape):
                raise CorruptArtifactError(
                    f"plane {name!r} payload {tuple(raw.shape)} != manifest {(nrows, *tail_shape)}"
                )
            if raw.dtype != storage_dtype:
                raw = raw.astype(storage_dtype, copy=False)
            if self._is_bool_view:
                raw = raw.view(np.bool_)
            raw.flags.writeable = False
            self._chunks.append((offset, raw))
            offset += nrows

    def close(self) -> None:
        self._chunks = []
        for mmap in self._mmaps:
            mmap.close()
        self._mmaps = []

    def take_rows(self, idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        (first_offset, first) = self._chunks[0]
        if len(self._chunks) == 1 and first_offset == 0:
            return np.ascontiguousarray(first[idx])
        out = np.empty((len(idx), *first.shape[1:]), dtype=first.dtype)
        for start, arr in self._chunks:
            mask: np.ndarray[Any, Any] = (idx >= start) & (idx < start + arr.shape[0])
            sel: np.ndarray[Any, Any] = np.nonzero(mask)[0]
            if len(sel) > 0:
                gathered: np.ndarray[Any, Any] = arr[idx[sel] - start]  # pyrefly: ignore[unknown-argument-type] # fancy-index rows dynamic; take_rows owns the gather
                out[sel] = gathered
        return np.ascontiguousarray(out)


def _load_manifest(shard_dir: Path, manifest_name: str) -> dict[str, Any]:
    path = shard_dir / manifest_name
    if not path.is_file():
        raise ContractError(f"shard manifest not found: {path}")
    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorruptArtifactError(f"shard manifest {path} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise CorruptArtifactError(f"shard manifest {path} must be a JSON object")
    missing = _REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    if len(missing) > 0:
        raise ContractError(f"shard manifest {path} missing keys {sorted(missing)}")
    return manifest


def _load_sidecar_list(shard_dir: Path, filename: str, rows: int, *, what: str) -> list[str] | None:
    path = shard_dir / filename
    if not path.is_file():
        return None
    try:
        values: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorruptArtifactError(f"sidecar {path} is not valid JSON") from exc
    if not isinstance(values, list) or len(values) != rows:
        raise CorruptArtifactError(f"sidecar {path} must be a list of {rows} {what}")
    typed: list[Any] = list(values)
    return [str(v) for v in typed]


class ShardReader:
    """Sequential mmap feed over one builder shard directory.

    Args:
        shard_dir: directory holding the manifest + per-plane files.
        batch_size: rows per batch (bucket-edge tails may be short; batches
            are never mixed-bucket).
        seed: shuffle seed (``None`` = on-disk order). Must match the
            manifest perm seed when ``order.kind`` is ``perm`` (or be
            ``None``); a different seed needs the ``decision_ids`` sidecar.
        prefetch_depth: producer queue depth; must be >= 2.
        bucket_batches: when ``True`` (default) batches are
            bucket-homogeneous with history-width planes sliced to the bucket
            width; when ``False`` batches are pure epoch-order slices.
        verify: verify per-plane sha256 + dataset_hash on open.
        expect_schema_digest: when given, the manifest ``schema_digest`` must
            equal it (production passes ``model_input_schema_digest()``).
        manifest_name: manifest file basename inside ``shard_dir``.
        allow_narrow: test-only flag permitting a manifest ``action_width``
            (or ``legal_mask`` plane) narrower than the frozen baseline.
    """

    def __init__(
        self,
        shard_dir: Path,
        *,
        batch_size: int,
        seed: int | None = None,
        prefetch_depth: int = MIN_PREFETCH_DEPTH,
        bucket_batches: bool = True,
        verify: bool = True,
        expect_schema_digest: str | None = None,
        manifest_name: str = "manifest.json",
        allow_narrow: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if prefetch_depth < MIN_PREFETCH_DEPTH:
            raise ContractError(f"prefetch_depth {prefetch_depth} < minimum {MIN_PREFETCH_DEPTH}")
        self._dir = Path(shard_dir)
        if not self._dir.is_dir():
            raise ContractError(f"shard dir is not a directory: {self._dir}")
        self.batch_size = batch_size
        self.bucket_batches = bucket_batches

        manifest = _load_manifest(self._dir, manifest_name)
        self._manifest = manifest
        if int(manifest["version"]) != MANIFEST_VERSION:
            raise ContractError(f"manifest version {manifest['version']!r} != {MANIFEST_VERSION}")
        rows = int(manifest["rows"])
        if rows <= 0:
            raise ContractError(f"manifest rows must be positive, got {rows!r}")
        self._rows = rows

        order: dict[str, Any] = manifest["order"]
        if not isinstance(order, dict) or order.get("kind") not in ("canonical", "perm"):
            raise ContractError(f"manifest order must be {{kind: canonical|perm}}, got {order!r}")
        self._order_kind: str = str(order["kind"])
        self._manifest_seed: int | None = int(order["seed"]) if self._order_kind == "perm" else None
        if self._order_kind == "perm" and "seed" not in order:
            raise ContractError("manifest order.kind 'perm' requires a seed")

        expected = str(manifest["schema_digest"])
        if expect_schema_digest is not None and expected != expect_schema_digest:
            raise ContractError(
                f"schema_digest {manifest['schema_digest']!r} != expected {expect_schema_digest!r}"
            )

        self._planes, self._plane_meta = self._open_planes(manifest, verify=verify)
        self._allow_narrow = allow_narrow
        self._action_width = self._resolve_action_width(manifest)
        self._decision_ids = _load_sidecar_list(self._dir, "decision_ids.json", rows, what="ids")
        self._observation_hashes = _load_sidecar_list(
            self._dir, "observation_hashes.json", rows, what="hashes"
        )
        if verify and self._decision_ids is not None:
            actual = dataset_hash_of(self._decision_ids)
            if _normalize_hex(str(manifest["dataset_hash"])) != _normalize_hex(actual):
                raise CorruptArtifactError("manifest dataset_hash != decision_ids.json content")

        self._seed = self._resolve_seed(seed)
        base = self._base_order()
        self._buckets = self._row_buckets() if bucket_batches else np.zeros(rows, dtype=np.int64)
        self._epoch_seq = self._epoch_order(base)
        self._spans = self._batch_spans()
        self._t_width = self._history_width()

        self._epoch = 0
        self._sample_in_epoch = 0
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=prefetch_depth)
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._start_producer(self._epoch, self._sample_in_epoch)

    # ------------------------------------------------------------------
    # Open / verify
    # ------------------------------------------------------------------

    def _open_planes(
        self, manifest: dict[str, Any], *, verify: bool
    ) -> tuple[dict[str, _NpyPlane | _TensorIpcPlane], dict[str, dict[str, Any]]]:
        raw_planes = manifest["planes"]
        if not isinstance(raw_planes, list) or len(raw_planes) == 0:
            raise ContractError("manifest 'planes' must be a non-empty list")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in raw_planes:
            if not isinstance(entry, dict) or not set(entry.keys()) >= _REQUIRED_PLANE_KEYS:
                raise ContractError(f"manifest plane entry malformed: {entry!r}")
            plane_name: str = entry["name"]
            grouped.setdefault(plane_name, []).append(entry)
        planes: dict[str, _NpyPlane | _TensorIpcPlane] = {}
        meta: dict[str, dict[str, Any]] = {}
        for name, entries in grouped.items():

            def _chunk_key(entry: dict[str, Any]) -> int:
                return int(entry.get("chunk", 0))

            entries = sorted(entries, key=_chunk_key)
            if [int(e.get("chunk", 0)) for e in entries] != list(range(len(entries))):
                raise ContractError(f"plane {name!r} chunks must be 0..N-1 contiguous")
            first = entries[0]
            dtype_raw = first.get("dtype", first.get("storage_dtype"))
            storage_raw = first.get("storage_dtype", first.get("dtype"))
            if dtype_raw is None or storage_raw is None:
                raise ContractError(f"plane {name!r} needs dtype/storage_dtype")
            try:
                logical_dtype: np.dtype[Any] = np.dtype(str(dtype_raw))
                storage_dtype: np.dtype[Any] = np.dtype(str(storage_raw))
            except TypeError as exc:
                raise ContractError(f"plane {name!r} dtype unknown") from exc
            if logical_dtype.hasobject or storage_dtype.hasobject:
                raise ContractError(f"plane {name!r} object dtype is not mappable")
            tail: tuple[int, ...] | None = None
            chunk_rows: list[int] = []
            files: list[Path] = []
            kinds: set[str] = set()
            for entry in entries:
                shape_raw = entry["shape"]
                if (
                    not isinstance(shape_raw, list)
                    or len(shape_raw) == 0
                    or not all(isinstance(d, int) and d > 0 for d in shape_raw)
                ):
                    raise ContractError(f"plane {name!r} shape {shape_raw!r} invalid")
                shape: tuple[int, ...] = tuple(shape_raw)
                entry_dtype = np.dtype(str(entry.get("dtype", entry.get("storage_dtype"))))
                entry_storage = np.dtype(str(entry.get("storage_dtype", entry.get("dtype"))))
                if entry_dtype != logical_dtype or entry_storage != storage_dtype:
                    raise ContractError(f"plane {name!r} chunks disagree on dtype")
                if tail is None:
                    tail = shape[1:]
                elif tail != shape[1:]:
                    raise ContractError(f"plane {name!r} chunks disagree on tail shape")
                chunk_rows.append(shape[0])
                path = self._dir / str(entry["file"])
                if not path.is_file():
                    raise ContractError(f"plane {name!r} file missing: {path}")
                if verify and _normalize_hex(str(entry["sha256"])) != _sha256_file(path):
                    raise CorruptArtifactError(f"plane {name!r} sha256 mismatch: {path}")
                kinds.add(_plane_suffix(path.name))
                files.append(path)
            assert tail is not None
            if sum(chunk_rows) != self._rows:
                raise ContractError(
                    f"plane {name!r} chunk rows {sum(chunk_rows)} != manifest rows {self._rows}"
                )
            if len(kinds) != 1:
                raise ContractError(f"plane {name!r} mixes storage kinds {sorted(kinds)}")
            kind = next(iter(kinds))
            # IPC-mmap-first: tensor-IPC planes are the primary production
            # path (columnar evidence: Probe-A/B live); ``.npy``/``.bin``
            # memmap below is the fallback alternate, never first.
            if kind in _IPC_SUFFIXES:
                planes[name] = _TensorIpcPlane(
                    files,
                    name=name,
                    storage_dtype=storage_dtype,
                    logical_dtype=logical_dtype,
                    chunk_rows=chunk_rows,
                    tail_shape=tail,
                )
            elif kind in _NPY_SUFFIXES | _RAW_SUFFIXES:
                if len(files) != 1:
                    raise ContractError(f"plane {name!r} numpy planes must be single-chunk")
                planes[name] = _NpyPlane(files[0], dtype=storage_dtype, shape=(self._rows, *tail))
            else:  # pragma: no cover — _plane_suffix already rejects
                raise ContractError(f"plane {name!r} has unknown storage suffix")
            meta[name] = {"dtype": logical_dtype, "shape": (self._rows, *tail)}
        return planes, meta

    def _resolve_action_width(self, manifest: dict[str, Any]) -> int | None:
        """Fail closed on undeclared or aliased action widths.

        The builder records the vocab width as ``action_width``; it must agree
        with the ``legal_mask`` plane when both are present.  A width other
        than the frozen baseline is test-only and requires ``allow_narrow``.
        Returns the resolved width, or ``None`` when neither the manifest nor
        the planes declare one (legacy envelope without a legal plane).
        """
        declared = manifest.get("action_width")
        if declared is not None and (
            not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0
        ):
            raise ContractError(f"manifest action_width invalid: {declared!r}")
        actual: int | None = None
        legal_meta = self._plane_meta.get("legal_mask")
        if legal_meta is not None:
            shape: tuple[int, ...] = legal_meta["shape"]
            if isinstance(shape, tuple) and len(shape) == 2:
                actual = shape[1]
        if declared is not None and actual is not None and declared != actual:
            raise ContractError(
                f"manifest action_width {declared} != legal_mask plane width {actual}"
            )
        width = declared if declared is not None else actual
        if width is not None and width != BASELINE_ACTION_COUNT and not self._allow_narrow:
            raise ContractError(
                f"shard action width {width} != baseline {BASELINE_ACTION_COUNT} "
                "requires allow_narrow=True (test-only narrow vocab)"
            )
        return width

    # ------------------------------------------------------------------
    # Order / buckets / spans
    # ------------------------------------------------------------------

    def _resolve_seed(self, seed: int | None) -> int | None:
        if self._order_kind == "canonical":
            return seed
        # On-disk order already is perm(manifest_seed).
        if seed is None or seed == self._manifest_seed:
            return seed if seed is not None else self._manifest_seed
        if self._decision_ids is None:
            raise ContractError(
                f"manifest order is perm seed {self._manifest_seed}, cannot re-derive "
                f"canonical for seed {seed} without decision_ids.json"
            )
        return seed

    def _base_order(self) -> np.ndarray[Any, Any]:
        if self._order_kind == "canonical":
            if self._seed is None:
                return np.arange(self._rows)
            return np.asarray(_canonical_perm(self._seed, self._rows), dtype=np.int64)
        assert self._manifest_seed is not None
        if self._seed == self._manifest_seed:
            return np.arange(self._rows)  # on-disk order already is the perm
        # Different seed: recover canonical via sidecar order, then permute.
        assert self._decision_ids is not None
        ids: list[str] = self._decision_ids

        def _canon_key(i: int) -> str:
            return ids[i]

        canon = np.asarray(sorted(range(self._rows), key=_canon_key), dtype=np.int64)
        assert self._seed is not None
        return canon[np.asarray(_canonical_perm(self._seed, self._rows), dtype=np.int64)]

    def _row_buckets(self) -> np.ndarray[Any, Any]:
        if "history_len" in self._planes:
            raw = self._planes["history_len"].take_rows(np.arange(self._rows))
            lengths = np.asarray(raw).reshape(self._rows).astype(np.int64)
        elif "history_mask" in self._planes:
            mask = self._planes["history_mask"].take_rows(np.arange(self._rows))
            lengths = np.asarray(mask).reshape(self._rows, -1).sum(axis=1).astype(np.int64)
        else:
            return np.zeros(self._rows, dtype=np.int64)
        return np.asarray([_bucket_ceil(int(v)) for v in lengths.tolist()], dtype=np.int64)

    def _epoch_order(self, base: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if not self.bucket_batches or len(np.unique(self._buckets)) <= 1:
            return np.asarray(base, dtype=np.int64)
        # Stable bucket sort: bucket ascending, base order kept within.
        keys = self._buckets[np.asarray(base, dtype=np.int64)]
        return np.asarray(base, dtype=np.int64)[np.argsort(keys, kind="stable")]

    def _batch_spans(self) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pos = 0
        while pos < self._rows:
            end = min(pos + self.batch_size, self._rows)
            if self.bucket_batches:
                # Epoch order is bucket-sorted, so spans are naturally aligned;
                # shrink the tail back to the bucket edge (short, homogeneous).
                bucket = self._buckets[self._epoch_seq[pos]]
                while end > pos + 1 and self._buckets[self._epoch_seq[end - 1]] != bucket:
                    end -= 1
            spans.append((pos, end))
            pos = end
        return spans

    def _history_width(self) -> int | None:
        for key in ("history_mask", "history_event_kind"):
            plane_meta = self._plane_meta.get(key)
            if plane_meta is not None and len(plane_meta["shape"]) == 2:
                width: int = plane_meta["shape"][1]
                return width
        return None

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _materialize(self, start: int, end: int, epoch: int) -> dict[str, Any]:
        rows_idx = self._epoch_seq[start:end]
        batch_bucket = int(self._buckets[self._epoch_seq[start]])
        width = self._t_width if self.bucket_batches else None
        batch: dict[str, Any] = {}
        for name, plane in self._planes.items():
            values = plane.take_rows(rows_idx)
            if (
                width is not None
                and values.ndim == 2
                and values.shape[1] == width
                and batch_bucket < width
            ):
                values = np.ascontiguousarray(values[:, :batch_bucket])
            if values.dtype.kind in ("S", "U", "O"):  # pyrefly: ignore[unknown-argument-type] # numpy dtype kind dynamic; branch owns the string decode
                batch[name] = [str(v) for v in values.tolist()]
            else:
                batch[name] = torch.from_numpy(np.ascontiguousarray(values))
        legal = batch.get("legal_mask")
        if isinstance(legal, torch.Tensor) and legal.dim() == 2:
            expected = (
                self._action_width if self._action_width is not None else BASELINE_ACTION_COUNT
            )
            if legal.shape[1] != expected:
                raise ContractError(
                    f"legal_mask width {legal.shape[1]} != expected {expected} "
                    "(aliased widths are never trained)"
                )
        if self._decision_ids is not None:
            order = self._epoch_seq[start:end].tolist()
            batch["_decision_ids"] = [self._decision_ids[int(i)] for i in order]
        batch["_epoch"] = epoch
        batch["_sample_start"] = start
        batch["_sample_end"] = end
        batch["_bucket"] = batch_bucket if self.bucket_batches else -1
        return batch

    def _run_producer(self, epoch: int, pos: int) -> None:
        span_idx = 0
        while span_idx < len(self._spans) and self._spans[span_idx][1] <= pos:
            span_idx += 1
        resume_at: int | None = pos
        while not self._stop.is_set():
            if span_idx >= len(self._spans):
                epoch += 1
                span_idx = 0
            start, end = self._spans[span_idx]
            if resume_at is not None:
                start = max(start, resume_at)
                resume_at = None
            if start >= end:
                span_idx += 1
                continue
            try:
                self._queue.put(self._materialize(start, end, epoch), timeout=0.1)
            except queue.Full:
                continue
            span_idx += 1

    def _start_producer(self, epoch: int, pos: int) -> None:
        self._stop.clear()
        while True:
            try:
                _ = self._queue.get_nowait()  # intentionally discarded: draining stale batches
            except queue.Empty:
                break
        self._producer = threading.Thread(
            target=self._run_producer,
            args=(epoch, pos),
            name="shard-reader-producer",
            daemon=True,
        )
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop.set()
        producer, self._producer = self._producer, None
        if producer is not None and producer.is_alive():
            producer.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._rows

    @property
    def batches_per_epoch(self) -> int:
        return len(self._spans)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    @property
    def dataset_hash(self) -> str:
        return str(self._manifest["dataset_hash"])

    @property
    def schema_digest(self) -> str:
        return str(self._manifest["schema_digest"])

    @property
    def split(self) -> str:
        return str(self._manifest["split"])

    @property
    def attestation(self) -> Any:
        return self._manifest["attestation"]

    @property
    def queue_depth(self) -> int:
        return self._queue.maxsize

    def get_state(self) -> dict[str, Any]:
        """Resume cursor ``{epoch, sample_in_epoch}`` (+ seed/rows guards)."""
        return {
            "epoch": self._epoch,
            "sample_in_epoch": self._sample_in_epoch,
            "seed": -1 if self._seed is None else self._seed,
            "rows": self._rows,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Seek to a cursor from :meth:`get_state`; never replays samples."""
        epoch = int(state["epoch"])
        pos = int(state["sample_in_epoch"])
        seed_raw = state.get("seed", -1 if self._seed is None else self._seed)
        seed = None if int(seed_raw) == -1 else int(seed_raw)
        if seed != self._seed:
            raise ContractError(f"cursor seed {seed!r} != reader seed {self._seed!r}")
        if int(state.get("rows", self._rows)) != self._rows:
            raise ContractError("cursor rows != reader rows")
        if epoch < 0 or not (0 <= pos <= self._rows):
            raise ContractError(f"cursor out of range: epoch={epoch} sample_in_epoch={pos}")
        self._stop_producer()
        self._epoch = epoch
        self._sample_in_epoch = pos
        self._start_producer(epoch, pos)

    def next_batch(self) -> dict[str, Any]:
        """Return the next batch, advancing the cursor (wraps epochs)."""
        batch = self._queue.get()
        self._epoch = int(batch["_epoch"])
        self._sample_in_epoch = int(batch["_sample_end"])
        return batch

    def iter_epoch(self) -> Iterator[dict[str, Any]]:
        """Yield the rest of the current epoch (stops at the epoch edge)."""
        while True:
            batch = self.next_batch()
            yield batch
            if int(batch["_sample_end"]) >= self._rows:
                break

    def close(self) -> None:
        self._stop_producer()
        for plane in self._planes.values():
            plane.close()

    def __enter__(self) -> ShardReader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
