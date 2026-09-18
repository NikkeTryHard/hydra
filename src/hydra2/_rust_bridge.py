"""Thin digest judge over the ``hydra2_replay_rs.canon_rng`` boundary.

Rust computes (``sha2`` over canon bytes, detached batch); Python compares
and fail-closes. No canon/hash/RNG math lives here — this module only moves
bytes across the boundary and asserts equality.

Resolution notes:
- Legacy entry point stays ``import hydra2_replay_rs`` (crate/lib names
  untouched); the ``hydra_bridge._native`` rename is a Phase-6 maturin
  ``module-name`` cutover and MUST NOT be attempted here.
- ``open_stream`` is for NEW streams only. Held-out splits stay on the
  ``torch.randperm`` oracle and sha-Gumbels stay verbatim; Philox never
  silently unifies them.
- Pixi staleness-gate sketch (Phase 6, DO NOT edit ``pyproject.toml`` yet)::

    [tasks]
    build-ext = { depends-on = ["_build-bridge"] }
    _build-bridge = {
      cmd = "maturin develop --manifest-path tools/hydra2-replay-rs/crates/hydra-bridge/Cargo.toml",
      inputs = ["Cargo.lock", "pyproject.toml",
                "src/hydra2/_rust_bridge.py",
                "tools/hydra2-replay-rs/crates/hydra-bridge/src/**/*.rs",
                "tools/hydra2-replay-rs/crates/hydra-feed/src/**/*.rs"],
      outputs = [".../hydra2_replay_rs*.so"],
    }
    # DELETE the hand-rolled mtime gate (``rust_batch._check_fresh`` sidecar)
    # in Phase 6; pixi fingerprints replace it.

"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from hydra2.artifacts.digest import require_digest_match
from hydra2.contracts.common import DigestText

__all__ = [
    "DigestJudge",
    "StreamHandle",
    "default_chunk_bytes",
    "judge_stats",
    "of_canonical",
    "open_stream",
]

#: Document bytes the judge accepts (bytes-like only; never ``str``).
_CANON_DOC_TYPES = (bytes, bytearray, memoryview)

#: Injectable native backend (tests monkeypatch this; production leaves None
#: so `_canon_rng()` imports the compiled extension).
_NATIVE_OVERRIDE: Any = None


def _native() -> Any:
    """Import the compiled bridge (fail closed, no fallback hash)."""
    try:
        return importlib.import_module("hydra2_replay_rs")
    except ImportError as exc:
        raise RuntimeError(
            "hydra2_replay_rs extension with canon_rng not importable; "
            "build the bridge before using the digest judge"
        ) from exc


def _canon_rng() -> Any:
    if _NATIVE_OVERRIDE is not None:
        return _NATIVE_OVERRIDE
    ext = _native()
    try:
        return ext.canon_rng
    except AttributeError as exc:
        raise RuntimeError(
            "hydra2_replay_rs.canon_rng submodule missing; rebuild the bridge"
        ) from exc


def of_canonical(doc: bytes | bytearray | memoryview) -> DigestText:
    """Digest over a JSON document's canonical bytes (detached batch of one).

    ``doc`` is the raw JSON document bytes (already-framed input), NOT a
    Python object: Rust parses (I-JSON boundary), canonicalizes (JCS), and
    hashes; Python only receives the ``sha256:<hex>`` text.
    """
    if not isinstance(doc, _CANON_DOC_TYPES):
        raise TypeError(f"judge doc must be bytes-like, got {type(doc).__name__}")
    return DigestText(str(_canon_rng().of_canonical_json(bytes(doc))))


@dataclass(frozen=True, slots=True)
class DigestJudge:
    """Fail-close comparator: assert recomputed == recorded, else raise."""

    subject: str = "judge"

    def verify(self, *, recorded: str, doc: bytes | bytearray | memoryview) -> DigestText:
        """Recompute via Rust and raise ``DigestMismatchError`` on mismatch."""
        recomputed: DigestText = of_canonical(doc)
        require_digest_match(recorded=recorded, recomputed=recomputed, subject=self.subject)
        return recomputed

    def verify_batch(
        self, *, recorded: list[str], docs: list[bytes | bytearray | memoryview]
    ) -> list[DigestText]:
        """Detached batch verify (one GIL release for the whole fan-out)."""
        if len(recorded) != len(docs):
            raise ValueError(
                f"judge batch length mismatch: {len(recorded)} recorded vs {len(docs)} docs"
            )
        payload: list[bytearray | bytes | memoryview] = [bytes(d) for d in docs]
        for doc in docs:
            if not isinstance(doc, _CANON_DOC_TYPES):
                raise TypeError(f"judge doc must be bytes-like, got {type(doc).__name__}")
        native_out = _canon_rng().batch_of_canonical(payload)
        recomputed: list[DigestText] = [DigestText(str(t)) for t in native_out]
        for want, got in zip(recorded, recomputed, strict=True):
            require_digest_match(recorded=want, recomputed=got, subject=self.subject)
        return recomputed


@dataclass(frozen=True, slots=True)
class StreamHandle:
    """Opened NEW-stream spec (validated frozen value, no draws yet)."""

    domain: bytes
    seed: bytes

    def block(self, index: int) -> tuple[int, int, int, int]:
        """One Philox block (pure function of key + counter; block, not word)."""
        if index < 0 or index >= 2**64:
            raise ValueError(f"stream block must be a u64, got {index!r}")
        spec = _canon_rng().open_stream(bytes(self.domain), bytes(self.seed))
        return tuple(int(w) for w in _canon_rng().philox_block(spec, index))  # type: ignore[return-value]

    def below(self, word: int, n: int) -> int:
        """ONE sanctioned ``[0, n)`` draw (Lemire, never ``% n``)."""
        if n < 1:
            raise ValueError(f"stream bound must be >= 1, got {n!r}")
        return int(_canon_rng().bounded(int(word) & 0xFFFF_FFFF, n))


def open_stream(*, domain: bytes, seed: bytes) -> StreamHandle:
    """Open a NEW stream (validates 1..=14-byte domain + 32-byte seed).

    NEW streams only — never for held-out splits or sha-Gumbels.
    """
    if not isinstance(domain, (bytes, bytearray)):
        raise TypeError(f"stream domain must be bytes, got {type(domain).__name__}")
    if not isinstance(seed, (bytes, bytearray)):
        raise TypeError(f"stream seed must be bytes, got {type(seed).__name__}")
    _canon_rng().open_stream(bytes(domain), bytes(seed))  # validates now, fails closed
    return StreamHandle(domain=bytes(domain), seed=bytes(seed))


def default_chunk_bytes() -> int:
    """Interpreter-safe default chunk size (Rust ``PyOnceLock`` init)."""
    return int(_canon_rng().default_chunk_bytes())


def judge_stats() -> tuple[int, int]:
    """Judge observability: ``(digests, batches)`` (never identity)."""
    digests, batches = _canon_rng().judge_stats()
    return (int(digests), int(batches))
