"""Atomic artifact publication — SPEC 2.3 verbatim.

``publish_atomic`` guarantees: bytes match the declared digest, the parent
directory exists and is not a symlink, a unique same-directory O_CREAT|O_EXCL
temp file carries the payload (write/flush/fsync), an existing destination is
verified byte-identical before reuse, first publication lands through a
no-clobber link so a racing writer can never be overwritten, and the parent
directory is fsynced before returning. On any error the owned temporary path
is removed where possible; an existing destination is NEVER removed.
"""

from __future__ import annotations

import contextlib
import errno
import os
from pathlib import Path

from hydra2.artifacts.digest import sha256_digest, sha256_file
from hydra2.contracts.common import ContractError, DigestMismatchError, DigestText

__all__ = ["atomic_replace_bytes", "publish_atomic"]


def _require_real_directory(parent: Path) -> None:
    if not parent.is_dir():
        raise ContractError(f"destination parent does not exist or is not a directory: {parent}")
    if parent.is_symlink():
        raise ContractError(f"destination parent must not be a symlink: {parent}")


def _mkstemp_o_excl(directory: Path, basename: str) -> tuple[int, str]:
    prefix = f".{basename}.tmp-"
    for _ in range(64):
        candidate = directory / (prefix + os.urandom(8).hex())
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                raise ValueError(f"artifact parent is not a directory: {directory}") from exc
            raise
        return fd, str(candidate)
    raise OSError("could not allocate a unique temporary artifact path")


def _fsync_dir(directory: Path) -> None:
    # Portable directory fsync: Windows (NT) cannot open a directory with
    # os.open(O_RDONLY) — raises OSError/PermissionError. On NTFS the file
    # handle fsync already guarantees durability, so directory fsync is a
    # no-op. On POSIX, failure to fsync the directory (e.g., tmpfs) is
    # non-fatal for the atomicity contract and is ignored.
    # Evidence: https://docs.python.org/3/library/os.html#os.fsync
    #  — \"On Windows, FlushFileBuffers analog\"; CPython Lib/pathlib uses
    #  try/except around os.open for directory fd on NT.
    #  https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-flushfilebuffers
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def publish_atomic(*, destination: Path, data: bytes, expected: DigestText) -> None:
    """Publish ``data`` at ``destination`` exactly as SPEC 2.3 prescribes.

    Idempotent for byte-identical republish (existing-destination verification
    shortcut). Raises :class:`DigestMismatchError` when ``data`` or an already
    existing destination does not hash to ``expected`` — an overwrite attempt
    therefore fails instead of replacing immutable bytes.
    """
    computed = sha256_digest(data)
    if computed != expected:
        raise DigestMismatchError(f"{destination}: data digests to {computed}, expected {expected}")
    destination = Path(destination)
    parent = destination.parent
    _require_real_directory(parent)
    fd, temp_name = _mkstemp_o_excl(parent, destination.name)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # No-clobber publication: link() fails atomically when a destination
        # appeared concurrently, so another writer's bytes are never replaced.
        # NOTE (EXDEV contract): cross-mount os.link raises EXDEV(18) and propagates
        # deliberately — same-directory temp is load-bearing for crash atomicity.
        # Evidence: https://man7.org/linux/man-pages/man2/link.2.html
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            if sha256_file(destination) != expected:
                raise DigestMismatchError(
                    f"{destination}: existing content does not match {expected}; "
                    "overwrite of an immutable artifact is rejected"
                ) from None
            return  # identical republish: keep original bytes, drop temp
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
    _fsync_dir(parent)


def atomic_replace_bytes(destination: Path, data: bytes) -> None:
    """Atomically replace a MUTABLE control file (e.g. registry index).

    Unique same-directory temp (O_CREAT|O_EXCL), write/flush/fsync, rename
    over destination, fsync parent. Unlike :func:`publish_atomic` this may
    overwrite; it exists only for mutable bookkeeping files whose identity is
    carried by their own content hash, never for immutable artifacts.
    """
    destination = Path(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = _mkstemp_o_excl(parent, destination.name)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(parent)
