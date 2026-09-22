"""Content-addressed tensor caches — checklist item 8.

Primary format is `.safetensors` (safetensors 0.8 layout: u64-LE header length +
JSON header `{name: {dtype, shape, data_offsets}}` + raw little-endian bytes),
read through `mmap` with header-only parse + offset validation and tensor bytes
borrowed zero-copy — no pickle opcode executes on the tensor path. The legacy
`.pt` pickle path is retained ONLY behind explicit opt-in flags, for
compat-read of pre-existing caches and the small-manifest exception, never as
the default tensor path (M4). Non-tensor blobs use the Arrow IPC-file path,
never pickle; small deterministic manifests stay canonical-JSON sidecars.

Hardening (fail-closed): safetensors is FIRST — the legacy pickle surface
runs ONLY behind explicit opt-in flags (`allow_legacy_pickle_write` /
`allow_legacy_pickle_read`); digest/dtype/layout pins raise ContractError
(never reshape, never a silent miss); unreadable/corrupt manifests raise
ContractError (never a raw passthrough). No Rust math here — header parse +
offset validation are pure Python over mmap; torch owns the tensor views.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["CacheKey", "build_cache", "cache_key_digest", "load_cache"]


@dataclass(frozen=True, slots=True)
class CacheKey:
    dataset_manifest_hash: str
    split: str
    schema_hash: str
    preprocess_id: str
    layout: str
    dtype: str
    library_id: str  # e.g., torch version
    library_version: str


def cache_key_digest(key: CacheKey) -> str:
    payload = {
        "dataset_manifest_hash": key.dataset_manifest_hash,
        "split": key.split,
        "schema_hash": key.schema_hash,
        "preprocess_id": key.preprocess_id,
        "layout": key.layout,
        "dtype": key.dtype,
        "library_id": key.library_id,
        "library_version": key.library_version,
    }
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _cache_path(cache_root: Path, digest: str) -> Path:
    # Two-level fan-out keeps any one directory small at corpus scale;
    # digest-addressed so identical keys share bytes. PRIMARY path is the
    # safetensors file (mmap-readable, header-validated, no pickle exec).
    hexpart = digest.removeprefix("sha256:")
    return cache_root / hexpart[:2] / hexpart[2:4] / f"{hexpart}.safetensors"


def _legacy_cache_path(cache_root: Path, digest: str) -> Path:
    # Compat-read address of pre-safetensors `.pt` pickle caches. Reached ONLY
    # behind the legacy opt-in flags — never probed by default.
    hexpart = digest.removeprefix("sha256:")
    return cache_root / hexpart[:2] / hexpart[2:4] / f"{hexpart}.pt"


# torch dtype <-> safetensors dtype-name map (safetensors 0.8 "DTypes" vocabulary).
# No other dtypes are representable: cache tensors outside this map are a
# ContractError, not a silent downcast.
_TORCH_TO_SAFETENSORS: dict[torch.dtype, str] = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_SAFETENSORS_TO_TORCH: dict[str, torch.dtype] = {
    name: dtype for dtype, name in _TORCH_TO_SAFETENSORS.items()
}


class _MmapBackedDict(dict):  # type: ignore[type-arg]
    """Plain dict that keeps mmap/file keepalives alive.

    `torch.frombuffer` borrows its storage: the mmap (and its file) must
    outlive every tensor view. `isinstance(..., dict)` stays True, so the
    `load_cache` return shape is unchanged; only this private subclass name
    differs, which no caller observes.
    """

    _keepalive: tuple[object, ...] = ()


def _write_safetensors(
    path: Path, manifest: dict[str, object], tensors: dict[str, torch.Tensor]
) -> None:
    """Write one safetensors-layout file atomically (tmp + rename).

    Header `__metadata__` carries only flat strings (spec-compatible): the
    digest/dtype/layout pins plus the full manifest as one canonical-JSON
    string, so a single-file read revalidates without the sidecar. Tensor
    bytes are little-endian contiguous CPU dumps in header order.
    """
    names = list(tensors.keys())
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in names:
        ten = tensors[name]
        if not isinstance(ten, torch.Tensor):
            raise ContractError(
                f"cache tensor {name!r} is {type(ten).__name__}, not torch.Tensor; "
                "non-tensor blobs use the Arrow IPC-file path, never pickle"
            )
        try:
            dtype_name = _TORCH_TO_SAFETENSORS[ten.dtype]
        except KeyError:
            raise ContractError(
                f"cache tensor {name!r} dtype {ten.dtype} has no safetensors mapping"
            ) from None
        # One-time save cost, documented: detach + CPU move + contiguous copy.
        # uint8 view first so raw bytes work for every dtype (numpy has no BF16).
        cpu = ten.detach().to("cpu").contiguous()
        raw = cpu.view(torch.uint8).flatten().numpy().tobytes()
        span = [offset, offset + len(raw)]
        header[name] = {"dtype": dtype_name, "shape": list(cpu.shape), "data_offsets": span}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {
        "digest": manifest.get("digest", ""),
        "dtype": manifest.get("dtype", ""),
        "layout": manifest.get("layout", ""),
        "manifest": canonical_bytes(manifest).decode("utf-8"),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        _ = fh.write(struct.pack("<Q", len(header_bytes)))
        _ = fh.write(header_bytes)
        fh.writelines(blobs)
    _ = tmp.rename(path)


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    """Header-only parse: returns (header dict, data-section start offset).

    Reads ONLY the 8-byte length + header JSON (one small copy); tensor bytes
    are never touched here, so cache-hit verification costs no tensor I/O.
    Every structural violation is a ContractError, never a pass-through.
    """
    try:
        with open(path, "rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise ContractError(f"cache file truncated (no header length): {path}")
            (header_len,) = struct.unpack("<Q", prefix)
            header_bytes = fh.read(header_len)
            if len(header_bytes) != header_len:
                raise ContractError(f"cache file truncated (header {header_len}B): {path}")
            header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"cache header unreadable: {path} ({type(exc).__name__})") from exc
    if not isinstance(header, dict):
        raise ContractError(f"cache header must be a JSON object: {path}")
    return header, 8 + header_len


def _check_manifest_meta(
    meta: object, *, key: CacheKey, digest: str, path: Path
) -> dict[str, object]:
    """Validate header `__metadata__` pins against the requested key."""
    if not isinstance(meta, dict):
        raise ContractError(f"cache manifest metadata must be an object: {path}")
    if meta.get("digest") != digest:
        raise ContractError(f"cache digest mismatch or missing metadata: {path}")
    if meta.get("dtype") != key.dtype or meta.get("layout") != key.layout:
        msg = "cache incompatible with requested key (would reshape)"
        raise ContractError(f"{msg}: {path}")
    try:
        raw_manifest: object = meta.get("manifest", "")
        manifest: dict[str, object] = json.loads(str(raw_manifest))
    except ValueError as exc:
        raise ContractError(f"cache manifest payload corrupt: {path}") from exc
    if not isinstance(manifest, dict):
        raise ContractError(f"cache manifest payload must be an object: {path}")
    return manifest


def _load_safetensors(*, path: Path, key: CacheKey, digest: str) -> dict[str, object]:
    """FIRST read path: mmap + header parse + offset-validate + borrowed slices.

    Offsets are validated BEFORE any tensor materializes (integer begin/end,
    within-file, byte-length == numel x itemsize); each tensor is a
    `torch.frombuffer` view over the mmap (ACCESS_COPY: writable mapping,
    copy-on-write, file never mutated), so tensor bytes are borrowed, never
    copied and never executed.
    """
    header, data_off = _read_safetensors_header(path)
    manifest = _check_manifest_meta(header.get("__metadata__"), key=key, digest=digest, path=path)
    with open(path, "rb") as fh:
        try:
            file_len = fh.seek(0, 2)
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
        except (OSError, ValueError) as exc:
            raise ContractError(f"cache mmap failed: {path} ({type(exc).__name__})") from exc
    out: dict[str, object] = _MmapBackedDict()
    try:
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(spec, dict):
                raise ContractError(f"cache tensor spec must be an object: {name}")
            dtype_name = spec.get("dtype")
            shape = spec.get("shape")
            offsets = spec.get("data_offsets")
            if not isinstance(dtype_name, str) or dtype_name not in _SAFETENSORS_TO_TORCH:
                raise ContractError(f"cache tensor {name!r} has unknown dtype {dtype_name!r}")
            torch_dtype = _SAFETENSORS_TO_TORCH[dtype_name]
            if not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape):
                raise ContractError(f"cache tensor {name!r} has invalid shape {shape!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(o, int) or o < 0 for o in offsets)
            ):
                raise ContractError(f"cache tensor {name!r} has invalid data_offsets {offsets!r}")
            begin, end = offsets
            if end < begin or data_off + end > file_len:
                raise ContractError(f"cache tensor {name!r} offsets out of file bounds")
            numel = 1
            for d in shape:
                numel *= d
            itemsize = torch.tensor([], dtype=torch_dtype).element_size()
            if end - begin != numel * itemsize:
                raise ContractError(
                    f"cache tensor {name!r} offset span {end - begin}B != "
                    f"shape {shape} x {itemsize}B ({dtype_name})"
                )
            view = memoryview(mm)[data_off + begin : data_off + end]
            out[name] = torch.frombuffer(view, dtype=torch_dtype).view(shape)
        out["__metadata__"] = manifest
    except BaseException:
        mm.close()
        fh.close()
        raise
    assert isinstance(out, _MmapBackedDict)
    out._keepalive = (fh, mm)
    return out


def build_cache(
    *,
    cache_root: Path,
    key: CacheKey,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
    allow_legacy_pickle_write: bool = False,
) -> Path:
    """Build content-addressed cache; returns path.

    - Cache miss rebuilds.
    - Incompatible cache never reshapes: if existing cache has different
      dtype/layout, it is not overwritten.
    - DEFAULT writes the `.safetensors` path (header-validated, mmap-readable,
      no pickle exec). The `.pt` pickle write is ONLY for compat migration and
      requires `allow_legacy_pickle_write=True`.
    """
    digest = cache_key_digest(key)
    dest = _cache_path(cache_root, digest)
    if dest.is_file():
        # Header-only compat read: parse the header, revalidate pins, never
        # touch tensor bytes; incompatibility raises, never reshapes.
        header, _ = _read_safetensors_header(dest)
        _ = _check_manifest_meta(header.get("__metadata__"), key=key, digest=digest, path=dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "digest": digest,
        "key": {
            "dataset_manifest_hash": key.dataset_manifest_hash,
            "split": key.split,
            "schema_hash": key.schema_hash,
            "preprocess_id": key.preprocess_id,
            "layout": key.layout,
            "dtype": key.dtype,
            "library_id": key.library_id,
            "library_version": key.library_version,
        },
        "dtype": key.dtype,
        "layout": key.layout,
        "tensor_names": list(tensors.keys()),
        "extra": metadata if metadata is not None else {},
    }
    if allow_legacy_pickle_write:
        # Compat-migration write ONLY (explicit opt-in, never the default):
        # keeps the addressed `.pt` pickle next to the safetensors file for
        # pre-migration readers. pickle is an arbitrary-code-execution surface
        # (`torch.load` unpickles); new readers MUST NOT take this path.
        legacy = _legacy_cache_path(cache_root, digest)
        legacy_payload: dict[str, object] = dict(tensors)
        legacy_payload["__metadata__"] = manifest
        legacy_tmp = legacy.with_suffix(".tmp")
        torch.save(legacy_payload, legacy_tmp)
        _ = legacy_tmp.rename(legacy)
    _write_safetensors(dest, manifest, tensors)
    sidecar = dest.with_name(dest.stem + ".manifest.json")
    atomic_replace_bytes(sidecar, canonical_bytes(manifest))
    return dest


def load_cache(
    *, cache_root: Path, key: CacheKey, allow_legacy_pickle_read: bool = False
) -> dict[str, object]:
    """Load a content-addressed cache (safetensors FIRST).

    Tries, in order: (1) the `.safetensors` path via mmap + header-only parse
    + offset validation; (2) the `.pt` pickle path ONLY when
    `allow_legacy_pickle_read=True` (compat-read of pre-migration caches —
    `torch.load` executes pickle opcodes, so this flag marks the one site
    allowed to run them); (3) a stable `.manifest.json` sidecar (the
    small-manifest exception: header-only copy, tensor values set to None)
    when the caller only needs pins/metadata, never tensors.
    """
    digest = cache_key_digest(key)
    path = _cache_path(cache_root, digest)
    if path.is_file():
        return _load_safetensors(path=path, key=key, digest=digest)
    legacy = _legacy_cache_path(cache_root, digest)
    if allow_legacy_pickle_read and legacy.is_file():
        # Compat-read ONLY (flag-gated): pre-migration `.pt` cache.
        # `weights_only=False` executes pickle opcodes — this flag-gated call
        # is the remaining exec surface in the area (M4); default is miss.
        data = torch.load(legacy, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ContractError("cache payload must be dict")
        meta: object = data.get("__metadata__", {})
        if not isinstance(meta, dict) or meta.get("digest") != digest:
            raise ContractError("cache digest mismatch or missing metadata")
        if meta.get("dtype") != key.dtype or meta.get("layout") != key.layout:
            raise ContractError("cache incompatible with requested key (would reshape)")
        return data
    manifest_path = path.with_name(path.stem + ".manifest.json")
    if manifest_path.is_file():
        # Small-manifest exception: header-only copy, no tensors resolved.
        # `canonical_bytes` revalidates the manifest; digest/dtype/layout pins
        # are enforced, tensor values intentionally left None.
        try:
            manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ContractError(
                f"cache manifest unreadable: {manifest_path} ({type(exc).__name__})"
            ) from exc
        if not isinstance(manifest, dict):
            raise ContractError("cache manifest must be an object")
        meta = manifest
        if meta.get("digest") != digest:
            raise ContractError("cache digest mismatch or missing metadata")
        if meta.get("dtype") != key.dtype or meta.get("layout") != key.layout:
            raise ContractError("cache incompatible with requested key (would reshape)")
        names = list(_manifest_tensor_names(manifest))
        return {"__metadata__": manifest, **dict.fromkeys(names)}
    raise FileNotFoundError(f"cache miss for {digest} at {path}")


def _manifest_tensor_names(manifest: dict[str, object]) -> list[str]:
    """Tensor names recorded in a manifest payload (writer-stamped list)."""
    names = manifest.get("tensor_names")
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        return list(names)
    extra = manifest.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("tensors"), list):
        legacy = extra["tensors"]
        if all(isinstance(n, str) for n in legacy):
            return list(legacy)
    return []
