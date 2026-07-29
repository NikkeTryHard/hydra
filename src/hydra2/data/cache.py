"""Content-addressed tensor caches — checklist item 8."""

from __future__ import annotations

import hashlib
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
    # Content-addressed: cache_root/<hex>/tensor.pt
    hexpart = digest.removeprefix("sha256:")
    return cache_root / hexpart[:2] / hexpart[2:4] / f"{hexpart}.pt"


def build_cache(
    *,
    cache_root: Path,
    key: CacheKey,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
) -> Path:
    """Build content-addressed cache; returns path.

    - Cache miss rebuilds.
    - Incompatible cache never reshapes: if existing cache has different
      dtype/layout, it is not overwritten.
    """
    digest = cache_key_digest(key)
    dest = _cache_path(cache_root, digest)
    if dest.is_file():
        # Verify existing cache matches key's dtype/shape; if incompatible, do not reshape
        try:
            existing = torch.load(dest, map_location="cpu", weights_only=False)
        except Exception:
            # Corrupt cache: treat as miss and rebuild
            pass
        else:
            meta = existing.get("__metadata__", {}) if isinstance(existing, dict) else {}
            if meta.get("dtype") != key.dtype or meta.get("layout") != key.layout:
                raise ContractError(
                    "cache incompatible: existing "
                    f"{meta.get('dtype')}/{meta.get('layout')} vs "
                    f"requested {key.dtype}/{key.layout}; refusing to reshape"
                )
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = dict(tensors)
    payload["__metadata__"] = {
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
        "extra": metadata if metadata is not None else {},
    }
    tmp = dest.with_suffix(".tmp")
    torch.save(payload, tmp)
    _ = tmp.rename(dest)
    sidecar = dest.with_suffix(".json")
    atomic_replace_bytes(sidecar, canonical_bytes(payload["__metadata__"]))
    return dest


def load_cache(*, cache_root: Path, key: CacheKey) -> dict[str, object]:
    digest = cache_key_digest(key)
    path = _cache_path(cache_root, digest)
    if not path.is_file():
        raise FileNotFoundError(f"cache miss for {digest} at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ContractError("cache payload must be dict")
    meta: object = data.get("__metadata__", {})
    if not isinstance(meta, dict) or meta.get("digest") != digest:
        raise ContractError("cache digest mismatch or missing metadata")
    if meta.get("dtype") != key.dtype or meta.get("layout") != key.layout:
        raise ContractError("cache incompatible with requested key (would reshape)")
    return data
