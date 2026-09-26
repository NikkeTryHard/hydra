"""Stream manifest: partition vocabulary, file list, digests, and caches.

Owns the scan-plane vocabulary shared by the reader and the training
driver: partition order/grouping constants, the ``FileEntry``/``StreamManifest``
records with the sha256-hex ordering builder, the byte-identical manifest
digest, the reservoir-blob snapshot format, the scan-cache envelope, the
shuffle-RNG codec, and the partition/identity math. Anything that must stay
byte-identical across checkpoints and scan caches lives here.

Split note: the scan-cache envelope lives in
``python/hydra2/data/stream_scan_cache.py`` (LOC gate); this module
re-exports its public names so existing import sites keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream_reservoir import RESERVOIR_BLOB_VERSION as RESERVOIR_BLOB_VERSION
from hydra2.data.stream_reservoir import read_reservoir_blob as read_reservoir_blob
from hydra2.data.stream_reservoir import write_reservoir_blob as write_reservoir_blob
from hydra2.data.stream_scan_cache import SCAN_CACHE_VERSION as SCAN_CACHE_VERSION
from hydra2.data.stream_scan_cache import _holdout_match as _holdout_match
from hydra2.data.stream_scan_cache import _ratios_match as _ratios_match
from hydra2.data.stream_scan_cache import _root_games_match as _root_games_match
from hydra2.data.stream_scan_cache import _roots_match as _roots_match
from hydra2.data.stream_scan_cache import load_scan_cache as load_scan_cache
from hydra2.data.stream_scan_cache import save_scan_cache as save_scan_cache
from hydra2.data.stream_scan_cache import scan_cache_path as scan_cache_path


def _contracts_fn(name: str) -> Any | None:
    """Bridge leaf on ``hydra2._native.contracts`` or ``None`` when stale."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError:
        return None
    sub = getattr(_ext, "contracts", None)
    return getattr(sub, name, None) if sub is not None else None


def _contracts_const(name: str, fallback: object) -> Any:
    """Bridge const or the fallback (Any: callers coerce to int/str/tuple)."""
    try:
        from hydra2 import _native as _ext_const  # pyrefly: ignore[missing-import]
    except ImportError:
        return fallback
    sub = getattr(_ext_const, "contracts", None)
    return getattr(sub, name, fallback) if sub is not None else fallback


if TYPE_CHECKING:
    import random
    from collections.abc import Sequence


__all__ = [
    "DEFAULT_GROUPING_KEYS",
    "HOLDOUT_SPEC",
    "MANIFEST_ARTIFACT_VERSION",
    "PARTITION_ORDER",
    "RESERVOIR_BLOB_VERSION",
    "SCAN_CACHE_VERSION",
    "FileEntry",
    "SplitName",
    "StreamManifest",
    "build_manifest",
    "load_manifest_artifact",
    "load_scan_cache",
    "manifest_digest",
    "parse_shuffle_rng",
    "read_reservoir_blob",
    "resolve_root_id",
    "save_manifest_artifact",
    "save_scan_cache",
    "scan_cache_path",
    "serialize_shuffle_rng",
    "write_reservoir_blob",
]


def _require_canon_rng() -> Any:
    """Import the built ``canon_rng`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with canon_rng not importable; "
            "build the bridge with `pixi run build-ext` before ordering a stream manifest"
        ) from exc
    try:
        return _ext.canon_rng
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.canon_rng submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _require_resume() -> Any:
    """Import the built ``resume`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with resume not importable; "
            "build the bridge with `pixi run build-ext` before reading reservoir blobs"
        ) from exc
    try:
        return _ext.resume
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.resume submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _require_columnar() -> Any:
    """Import the built ``columnar`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with columnar not importable; "
            "build the bridge with `pixi run build-ext` before digesting a stream manifest"
        ) from exc
    try:
        sub = _ext.columnar
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.columnar submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    if getattr(sub, "stream_manifest_digest", None) is None:
        raise ImportError(
            "hydra2._native.columnar.stream_manifest_digest missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return sub


#: Partition names in :mod:`partition.py` cumulative-threshold order.
PARTITION_ORDER: tuple[str, ...] = tuple(
    _contracts_const(
        "RECORD_PARTITION_ORDER",
        ("train", "validation", "test", "decision_eval", "block_eval"),
    )
)

#: Default grouping; mirrors ``grouping_keys=("source", "time")``.
DEFAULT_GROUPING_KEYS: tuple[str, ...] = ("source", "time")
SplitName = Literal["train", "validation", "test", "decision_eval", "block_eval"]


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One corpus file: absolute IO path plus its digest identity.

    ``path`` is the absolute filesystem path (IO/stat only, never hashed);
    the digest identity is ``(root_id, relpath, bytes)`` so rerouting a root
    to another mount keeps the digest stable while a changed root list (or a
    renamed root id) fails closed exactly once via the digest mismatch.
    """

    path: Path
    bytes: int
    root_id: str = ""
    relpath: str = ""
    game_count: int | None = None
    wall_hashes: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class StreamManifest:
    """Deterministic file list; order is sha256-hex of root-id + NUL + relpath."""

    files: tuple[FileEntry, ...]
    roots: tuple[tuple[str, str], ...] = ()
    #: Artifact-load attestation only (constructor-rejected via ``init=False``,
    #: attached by ``object.__setattr__`` after the bridge re-verify in
    #: ``load_manifest_artifact`` — no caller can forge it through normal
    #: construction): the digest pinned in the manifest-artifact header,
    #: verified byte-equal to a fresh bridge digest at load time. Lets
    #: ``manifest_digest`` skip re-hashing on warm launches. Excluded from
    #: equality (``compare=False``) so stored and fresh manifests compare by
    #: file list; a non-``None`` value that is not a ``sha256:`` string fails
    #: closed in ``manifest_digest`` instead of returning.
    _stored_digest: str | None = field(init=False, default=None, compare=False, repr=False)

    def __len__(self) -> int:
        return len(self.files)

    def paths(self) -> tuple[Path, ...]:
        """File paths in stream order."""
        return tuple(entry.path for entry in self.files)


#: Whole-root holdout spec bound into digests and scan-cache envelopes.
#: Exclusion is the mechanism (quarantined/future directories never enter a
#: train manifest); per-root 80/20 group-hash validation covers
#: in-distribution eval. Empty `val_roots` today: era-holdouts are separate
#: eval-only runs, never a training split.
HOLDOUT_SPEC: dict[str, object] = {"mechanism": "per-root-stratified", "val_roots": []}


def _normalize_roots(
    roots: Sequence[tuple[str, Path | str]] | Path | str,
) -> list[tuple[str, Path]]:
    """Normalize the manifest source list to ``[(root-id, absolute dir)]``.

    A bare path stays valid (single root keyed by its leaf-dir basename, so
    probes and unit corpora need no pairs); otherwise every item must be an
    ``(id, path)`` pair with a non-empty basename-style id (no `/`, no NUL —
    ids join the order-key preimage) and an existing directory. Duplicate ids
    fail closed: two roots sharing an id would order and split identically.
    """
    if isinstance(roots, (str, Path)):
        pairs: list[tuple[str, Path | str]] = [(Path(roots).name, roots)]
    else:
        pairs = list(roots)
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ContractError(f"manifest roots must be (root-id, path) pairs, got {item!r}")
        rid, raw_base = item
        if not isinstance(rid, str) or rid == "" or "/" in rid or "\0" in rid:
            raise ContractError(f"manifest root id must be a non-empty basename, got {rid!r}")
        if rid in seen:
            raise ContractError(f"manifest duplicate root id {rid!r}")
        seen.add(rid)
        base = raw_base if isinstance(raw_base, Path) else Path(str(raw_base))
        if not base.is_dir():
            raise ContractError(f"stream root not a directory: {base}")
        normalized.append((rid, base))
    return normalized


def build_manifest(
    roots: Sequence[tuple[str, Path | str]] | Path | str, pattern: str = "*.mjai.json.zst"
) -> StreamManifest:
    """Collect files across roots in sha256-hex ``root-id + NUL + relpath`` order.

    The hash orders (never splits): split assignment uses
    :func:`assign_split` exclusively. Order keys are minted in one
    ``canon_rng.batch_sha256`` call over the namespaced preimage bytes
    (``ImportError`` with a ``build-ext`` hint when the extension is not
    built — NO oracle fallback); duplicate relpaths under different roots
    order and split independently, never colliding.

    A stored manifest artifact short-circuits the walk/hash/sort below on
    repeat launches with unchanged roots (same digest, zero hashing); any
    fingerprint drift or digest mismatch falls through to this exact cold
    path, so a hit can never serve a stale order.
    """
    normalized = _normalize_roots(roots)
    root_pairs = [(rid, str(base)) for rid, base in normalized]
    artifact_path: Path | None = None
    fingerprints: list[list[int]] | None = None
    try:
        marks: list[list[int]] = []
        for _, base in normalized:
            stamp = base.stat()
            marks.append([stamp.st_mtime_ns, stamp.st_size])
        fingerprints = marks
        artifact_path = _manifest_artifact_path(roots=root_pairs, pattern=pattern)
        hit = load_manifest_artifact(
            artifact_path, roots=root_pairs, pattern=pattern, fingerprints=marks
        )
        if hit is not None:
            return hit
    except OSError:
        artifact_path = None
        fingerprints = None
    found: list[tuple[str, Path, str]] = []
    for rid, base in normalized:
        for path in base.rglob(pattern):
            if path.is_file():
                found.append((rid, path, path.relative_to(base).as_posix()))
    canon_rng = _require_canon_rng()
    try:
        digests: list[str] = canon_rng.batch_sha256(
            [(rid + "\0" + rel).encode() for rid, _, rel in found]
        )
        keys = [text.removeprefix("sha256:") for text in digests]
    except ValueError as exc:
        raise ContractError(f"manifest ordering failed: {exc}") from exc
    keyed = sorted(zip(keys, found, strict=True), key=lambda kv: kv[0])
    entries = tuple(
        FileEntry(path=path, bytes=path.stat().st_size, root_id=rid, relpath=rel)
        for _, (rid, path, rel) in keyed
    )
    manifest = StreamManifest(files=entries, roots=tuple(root_pairs))
    if artifact_path is not None and fingerprints is not None:
        # Digest failure stays loud (a broken cold path must raise); the
        # artifact write itself is best-effort inside `save_manifest_artifact`.
        digest = manifest_digest(manifest)
        save_manifest_artifact(
            artifact_path,
            manifest=manifest,
            digest=digest,
            roots=root_pairs,
            pattern=pattern,
            fingerprints=fingerprints,
        )
    return manifest


def manifest_digest(manifest: StreamManifest) -> str:
    """Bind the file list for RunSpec provenance.

    Thin bridge delegate: ``hydra2._native.columnar.stream_manifest_digest``
    binds ``[(root_id, relpath, bytes)]`` verbatim in manifest order through
    the feed-owned ``hydra_feed::manifest::manifest_digest`` (ASCII fast-path
    with fail-closed canon cross-check — ``ImportError`` with a
    ``build-ext`` hint when the extension is not built, NO oracle
    fallback), so digests are byte-identical to
    ``sha256(canonical_bytes([{bytes, path, root_id}, ...]))`` with
    ``path`` = relpath. Any divergence breaks scan-cache keys loudly
    (raise here, miss downstream), never silently. Absolute IO paths stay
    outside the digest: rerouting a root to another mount keeps the digest
    stable, while a changed root list fails the pin exactly once.

    Stays off ``packet_decode.manifest_digest`` (sha-sort fork): order here
    is the stored manifest order, never re-sorted.

    A manifest carrying a verified ``_stored_digest`` (artifact-load only,
    digest re-verified at load) returns it without calling the bridge; any
    other non-``None`` shape raises instead of returning.
    """
    stored = manifest._stored_digest
    if stored is not None:
        if not isinstance(stored, str) or not stored.startswith("sha256:"):
            raise ContractError(f"manifest stored digest is not a sha256 binding: {stored!r}")
        return stored
    columnar = _require_columnar()
    try:
        digest_text: str = columnar.stream_manifest_digest(
            [(entry.root_id, entry.relpath, entry.bytes) for entry in manifest.files]
        )
        return digest_text
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


#: Manifest-artifact layout version (bump on format change; mismatch → miss,
#: never coerce: the stored order must stay byte-identical to a fresh build
#: or scan-cache keys and resume cursors silently reinterpret).
MANIFEST_ARTIFACT_VERSION = 1


def _manifest_artifact_path(*, roots: Sequence[tuple[str, str]], pattern: str) -> Path:
    """Artifact file for one root-list + pattern pair (read path never mkdirs).

    One file per root-list hash, so concurrent runs never share a target:
    worst case is a redundant rebuild, never corruption (writers publish
    via tmp+replace). Override the directory with
    ``HYDRA2_MANIFEST_ARTIFACT_DIR`` for tests (never inside a data root);
    default mirrors the scan-cache family under ``XDG_CACHE_HOME``.
    """
    cache_dir: str | None = os.environ.get("HYDRA2_MANIFEST_ARTIFACT_DIR")
    base_raw: str = (
        cache_dir
        if cache_dir is not None
        else os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "hydra2",
            "manifest",
        )
    )
    key = hashlib.sha256(repr((list(roots), pattern)).encode()).hexdigest()[:16]
    return Path(base_raw) / f"manifest-{key}.jsonl.zst"


def save_manifest_artifact(
    path: Path | str,
    *,
    manifest: StreamManifest,
    digest: str,
    roots: Sequence[tuple[str, str]],
    pattern: str,
    fingerprints: Sequence[Sequence[int]],
) -> None:
    """Best-effort artifact write (never fails training on I/O error).

    Payload: one JSON header line (version, digest, roots, pattern,
    per-root ``[mtime_ns, size]`` fingerprints, row count) then the
    zstd-compressed JSONL body, one ``[root_id, relpath, bytes]`` array per
    line in stored order. tmp+replace publishes atomically: a crash leaves
    old or new, never torn. Deletion is always safe (absence is a miss).
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "version": MANIFEST_ARTIFACT_VERSION,
            "digest": digest,
            "roots": [[rid, base] for rid, base in roots],
            "pattern": pattern,
            "fingerprints": [list(fp) for fp in fingerprints],
            "count": len(manifest.files),
        }
        body = "\n".join(
            json.dumps([entry.root_id, entry.relpath, entry.bytes]) for entry in manifest.files
        ).encode("utf-8")
        compressed = zstd.ZstdCompressor(level=1).compress(body)
        tmp = target.with_suffix(".tmp")
        # Atomic artifact publish is the effect; byte count discarded.
        _ = tmp.write_bytes(json.dumps(header, sort_keys=True).encode("utf-8") + b"\n" + compressed)
        _ = tmp.replace(target)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, zstd.ZstdError):
        return


def load_manifest_artifact(
    path: Path | str,
    *,
    roots: Sequence[tuple[str, str]],
    pattern: str,
    fingerprints: Sequence[Sequence[int]],
) -> StreamManifest | None:
    """Rebuild the stored file order on exact header match, else ``None`` (miss).

    Two-tier validation: cheap per-root ``[mtime_ns, size]`` fingerprints
    reject drifted trees before reading the body, then the rebuilt order is
    re-hashed through the bridge and must equal the pinned digest (sound
    hit even for nested roots: content drift fails the digest, not the
    fingerprint). The returned manifest carries the verified digest, so
    ``manifest_digest`` skips re-hashing. Any error — short read, header
    mismatch, zstd/JSON failure, row-shape mismatch, unknown root id, row
    count drift, digest inequality — is a miss, never partial, never raise
    (a missing bridge propagates ``ImportError`` like the cold path, since
    no manifest of any kind is buildable without it).
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    try:
        header_raw, body = data.split(b"\n", 1)
        header = json.loads(header_raw.decode("utf-8"))
        if not isinstance(header, dict):
            return None
        if header.get("version") != MANIFEST_ARTIFACT_VERSION:
            return None
        digest = header.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            return None
        if header.get("pattern") != pattern:
            return None
        if header.get("roots") != [[rid, base] for rid, base in roots]:
            return None
        if header.get("fingerprints") != [list(fp) for fp in fingerprints]:
            return None
        count = header.get("count")
        raw_body = zstd.ZstdDecompressor().decompress(body)
        rows: list[tuple[str, str, int]] = []
        for line in raw_body.decode("utf-8").split("\n"):
            if not line:
                continue
            row = json.loads(line)
            rid, rel, size = row
            if (
                not isinstance(rid, str)
                or not isinstance(rel, str)
                or not isinstance(size, int)
                or isinstance(size, bool)
            ):
                return None
            rows.append((rid, rel, size))
        if not isinstance(count, int) or isinstance(count, bool) or len(rows) != count:
            return None
        bases = {rid: Path(base) for rid, base in roots}
        stored_roots = tuple((rid, str(bases[rid])) for rid, _ in roots)
        entries = tuple(
            FileEntry(path=bases[rid] / rel, bytes=size, root_id=rid, relpath=rel)
            for rid, rel, size in rows
        )
        manifest = StreamManifest(files=entries, roots=stored_roots)
        if manifest_digest(manifest) != digest:
            return None
        # Declared slot, so `object.__setattr__` lands despite frozenness;
        # unreachable before the re-verify above (any mismatch returned).
        object.__setattr__(manifest, "_stored_digest", digest)
        return manifest
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        zstd.ZstdError,
        ContractError,
    ):
        return None


def resolve_root_id(roots: Sequence[tuple[str, str]], path: Path) -> str:
    """Resolve the manifest root id owning an absolute corpus path.

    The longest matching root prefix wins (roots are disjoint shelf
    directories). No match fails closed — the corpus moved out from under
    the manifest, and the digest pin would refuse it too. Callers on hot
    paths read ``entry.root_id`` directly and never call this; resume
    restore entries (path/offset only) and the serial scan's file-boundary
    tracker are the only users.
    """
    text = path.as_posix()
    best: str | None = None
    best_len = -1
    for rid, base in roots:
        prefix = base if base.endswith("/") else base + "/"
        if (text == base or text.startswith(prefix)) and len(prefix) > best_len:
            best, best_len = rid, len(prefix)
    if best is None:
        raise ContractError(f"path outside every manifest root: {path}")
    return best


def serialize_shuffle_rng(rng: random.Random) -> dict[str, object]:
    """JSON-safe snapshot of a shuffle ``random.Random`` (fail-closed on misuse).

    Stores the version+internal+gauss_next getstate triple; misuse fails
    closed via ContractError so a corrupt sidecar resumes from origin.
    """
    state = rng.getstate()
    version, internal, gauss_next = state[0], state[1], state[2]
    bridge = _contracts_fn("record_serialize_shuffle_rng")
    if bridge is not None:
        try:
            framed: dict[str, object] = bridge(version, internal, gauss_next)
            return framed
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        except (ImportError, AttributeError):
            pass
    if not isinstance(version, int) or not isinstance(internal, tuple):
        raise ContractError("shuffle RNG state malformed")
    if gauss_next is not None and not isinstance(gauss_next, float):
        raise ContractError("shuffle RNG gauss state malformed")
    internal_seq: tuple[int, ...] = internal
    return {"version": version, "state": list(internal_seq), "gauss_next": gauss_next}


def parse_shuffle_rng(raw: object) -> tuple[int, tuple[int, ...], float | None]:
    """Inverse of :func:`serialize_shuffle_rng`; raises on any mismatch."""
    bridge = _contracts_fn("record_parse_shuffle_rng")
    if bridge is not None:
        try:
            parsed: tuple[int, tuple[int, ...], float | None] = bridge(raw)
            version_b, state_b, gauss_b = parsed
            return (version_b, state_b, gauss_b)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        except (ImportError, AttributeError):
            pass
    if not isinstance(raw, dict):
        raise ContractError("shuffle buffer_rng_state must be a mapping")
    version = raw.get("version")
    internal = raw.get("state")
    gauss_next = raw.get("gauss_next")
    raw_map: dict[str, object] = raw
    unknown = sorted(k for k in raw_map if k not in ("version", "state", "gauss_next"))
    if len(unknown) > 0:
        raise ContractError(f"shuffle buffer_rng_state unknown keys {unknown}")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ContractError("shuffle buffer_rng_state.version must be an int")
    if not isinstance(internal, list) or len(internal) == 0:
        raise ContractError("shuffle buffer_rng_state.state must be a non-empty int list")
    for value in internal:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("shuffle buffer_rng_state.state must hold non-negative ints")
    if gauss_next is not None and (
        isinstance(gauss_next, bool) or not isinstance(gauss_next, (int, float))
    ):
        raise ContractError("shuffle buffer_rng_state.gauss_next must be a float or null")
    gauss: float | None = None if gauss_next is None else float(gauss_next)
    internal_list: list[int] = internal
    return (version, tuple(internal_list), gauss)
