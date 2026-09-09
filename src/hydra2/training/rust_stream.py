"""PyO3 JSON-rows handoff (S8 cutover: primary replay path, no tensors/buffers).

Thin Python side of ``tools/hydra2-replay-rs/src/py_stream.rs``: open a
serial ``PyHydra2ReplayStream`` over framed per-game inputs, pull
newline-delimited DecisionRow-JSON batches through a caller-owned bytearray,
and validate every row before it reaches training code.

Enforced per batch (defense in depth, Rust enforces the same gates before
any byte crosses the FFI boundary):

- row keys are EXACTLY the 13 ``ACTOR_FIELDS`` (extras = privileged leak,
  missing = schema fork, both fail closed);
- ``actor_observation`` parses and carries no ``FORBIDDEN_REPLAY_KEYS``;
- ``split``/``rules_hash``/``adapter_hash``/``action_table_hash`` echo the
  open binding: ``split`` is the dataset split, the three digests are the
  pinned Slice-5 authorities (published rules bytes, engine-identity digest,
  verified table CONTENT digest). The legacy ``spec_hash`` stays a required
  transport pin for FFI stability but is never bound into rows.

Scratch discipline mirrors the Rust ``parity_84`` gate: when
``HYDRA2_ARTIFACT_ROOT`` is set it must sit outside the raw corpus roots
(``HYDRA2_DATA_ROOT``, ``HYDRA2_TENHOU_MOUNT``); the corpus stays read-only.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hydra2.data.parquet import ACTOR_FIELDS
from hydra2.training.replay import FORBIDDEN_REPLAY_KEYS

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ACTOR_FIELD_SET",
    "RustJsonStream",
    "RustStreamQuarantine",
    "RustStreamStats",
    "assert_artifact_root_outside_raw_roots",
    "open_rust_stream",
]

#: Exact row envelope: the 13 actor-namespace fields, nothing else.
ACTOR_FIELD_SET = frozenset(ACTOR_FIELDS)

_RAW_ROOT_KEYS = ("HYDRA2_DATA_ROOT", "HYDRA2_TENHOU_MOUNT")


def assert_artifact_root_outside_raw_roots() -> None:
    """Fail closed when artifacts would land inside a raw corpus mount."""
    artifact = os.environ.get("HYDRA2_ARTIFACT_ROOT", "")
    if not artifact:
        return
    for key in _RAW_ROOT_KEYS:
        raw = os.environ.get(key, "")
        if not raw:
            continue
        if artifact.startswith(raw) or raw.startswith(artifact):
            raise ValueError(
                f"{key} raw root must sit outside HYDRA2_ARTIFACT_ROOT: "
                f"artifact={artifact} raw={raw}"
            )


def _load_extension() -> Any:
    """Import the compiled ``hydra2_replay_rs`` module (fail closed)."""
    try:
        return importlib.import_module("hydra2_replay_rs")
    except ImportError as exc:
        raise RuntimeError(
            "hydra2_replay_rs extension not importable; build it with "
            "`cargo build -p hydra2-replay-rs` and put the resulting "
            "`hydra2_replay_rs` shared object on sys.path "
            "(see tests/unit/test_rust_stream_wp14.py for the exact recipe)"
        ) from exc


def _minted_provenance() -> tuple[str, str, str]:
    """Resolve the pinned Slice-5 digest authorities from the Python oracle.

    Returns ``(rules_hash, adapter_hash, action_table_hash)`` byte-identical
    to the Rust ``PINNED_*`` constants: sha256 over the published rules-file
    bytes (published wins, mirroring ``_rules_identity``), ``of_canonical``
    over the machine-stable engine identity (mirroring ``_adapter_hash``),
    and the verified table CONTENT digest (``table.digest``, never file
    bytes). Imports stay lazy so this module never drags the engine graph
    into collectors that only need the envelope constants.
    """
    from hydra2.config import repo_root
    from hydra2.contracts.action import ACTION_TABLE_RELPATH, load_action_table
    from hydra2.data.replay_expand import _adapter_hash, _load_rules
    from hydra2.engines.riichienv.adapter import _rules_identity
    from hydra2.engines.riichienv.state import rules_identity_hash

    rules = _load_rules()
    resolved = (
        _rules_identity(rules, str(rules_identity_hash(rules))),
        _adapter_hash(),
        str(load_action_table(repo_root() / ACTION_TABLE_RELPATH).digest),
    )
    for name, value in zip(
        ("rules_hash", "adapter_hash", "action_table_hash"), resolved, strict=True
    ):
        if not value:
            raise RuntimeError(f"rust stream oracle left {name} empty; refusing to open")
    return resolved


@dataclass(frozen=True, slots=True)
class RustStreamStats:
    open_count: int
    games_ok: int
    games_quarantined: int
    rows_out: int


@dataclass(frozen=True, slots=True)
class RustStreamQuarantine:
    game_id: str
    reason_code: str
    detail: str


class RustJsonStream:
    """Serial DecisionRow-JSON stream over framed per-game inputs.

    ``next()`` returns a validated ``list[dict]`` (empty at exhaustion);
    ``close()`` is idempotent; use as a context manager for scoped runs.

    The open digest binding is the Slice-5 minted trio
    (``rules_hash``/``adapter_hash``/``action_table_hash``); omitted digests
    resolve from the Python oracle (byte-identical to the Rust ``PINNED_*``
    authorities), so every row must carry the minted values verbatim.
    ``spec_hash`` stays a required transport pin for FFI stability and is
    never read back out of rows.
    """

    def __init__(
        self,
        data_dirs: list[str],
        batch: int,
        workers: int = 0,
        queue: int = 4,
        split: str = "train",
        spec_hash: str = "",
        capacity_bytes: int = 0,
        rules_hash: str | None = None,
        adapter_hash: str | None = None,
        action_table_hash: str | None = None,
    ) -> None:
        assert_artifact_root_outside_raw_roots()
        if not data_dirs:
            raise ValueError("rust stream data_dirs must not be empty")
        if batch < 1:
            raise ValueError(f"rust stream batch must be >= 1, got {batch}")
        if workers != 0:
            raise ValueError("rust stream workers>0 is deferred to Slice 7; pass workers=0")
        if split not in ("train", "validation"):
            raise ValueError(f"rust stream split must be train|validation, got {split!r}")
        if not spec_hash:
            raise ValueError("rust stream spec_hash must not be empty")
        if rules_hash is None or adapter_hash is None or action_table_hash is None:
            minted = _minted_provenance()
            if rules_hash is None:
                rules_hash = minted[0]
            if adapter_hash is None:
                adapter_hash = minted[1]
            if action_table_hash is None:
                action_table_hash = minted[2]
        for name, value in (
            ("rules_hash", rules_hash),
            ("adapter_hash", adapter_hash),
            ("action_table_hash", action_table_hash),
        ):
            if not value:
                raise ValueError(f"rust stream {name} must not be empty")
        ext = _load_extension()
        self._split = split
        self._rules_hash = rules_hash
        self._adapter_hash = adapter_hash
        self._action_table_hash = action_table_hash
        capacity = capacity_bytes or max(65536, batch * 16384)
        self._buf = bytearray(capacity)
        self._closed = False
        self._handle: Any = ext.PyHydra2ReplayStream.open(
            [str(d) for d in data_dirs], batch, workers, queue, split, spec_hash
        )

    def _validated_rows(self, payload: bytes) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in payload.split(b"\n"):
            if not line:
                continue
            row: Any = json.loads(line)
            if not isinstance(row, dict) or set(row) != ACTOR_FIELD_SET:
                raise ValueError(
                    "rust stream row envelope drift: keys must be exactly ACTOR_FIELDS, "
                    f"got {sorted(row) if isinstance(row, dict) else type(row).__name__}"
                )
            forbidden = set(row) & set(FORBIDDEN_REPLAY_KEYS)
            if forbidden:
                raise ValueError(f"rust stream row carries forbidden keys {sorted(forbidden)}")
            if row["split"] != self._split:
                raise ValueError(
                    f"rust stream row split {row['split']!r} != open split {self._split!r}"
                )
            for key, expected in (
                ("rules_hash", self._rules_hash),
                ("adapter_hash", self._adapter_hash),
                ("action_table_hash", self._action_table_hash),
            ):
                value = row[key]
                if not isinstance(value, str) or value != expected:
                    raise ValueError(f"rust stream row {key} does not bind the open {key}")
            actor_text = row["actor_observation"]
            if not isinstance(actor_text, str):
                raise ValueError("rust stream actor_observation must ride as a JSON string")
            actor_doc: Any = json.loads(actor_text)
            if not isinstance(actor_doc, dict):
                raise ValueError("rust stream actor_observation must decode to an object")
            forbidden_doc = set(actor_doc) & set(FORBIDDEN_REPLAY_KEYS)
            if forbidden_doc:
                raise ValueError(
                    f"rust stream actor_observation carries forbidden keys {sorted(forbidden_doc)}"
                )
            chosen = row["chosen_action_id"]
            if chosen is not None and not isinstance(chosen, int):
                raise ValueError("rust stream chosen_action_id must be int or null")
            rows.append(row)
        return rows

    def next(self) -> list[dict[str, Any]]:
        """Drain up to ``batch`` validated rows ([] at exhaustion)."""
        if self._closed:
            raise ValueError("hydra2 replay stream is closed")
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        nxt: Any = self._handle.next_into_json(addr, len(self._buf))
        payload = bytes(self._buf[: int(nxt.bytes_written)])
        return self._validated_rows(payload)

    def stats(self) -> RustStreamStats:
        """Cumulative counters (``open_count == 1`` on a live stream)."""
        raw: Any = self._handle.stats()
        return RustStreamStats(
            open_count=int(raw.open_count),
            games_ok=int(raw.games_ok),
            games_quarantined=int(raw.games_quarantined),
            rows_out=int(raw.rows_out),
        )

    def quarantines(self) -> list[RustStreamQuarantine]:
        """Whole-game quarantines in load order (Slice-3 reason codes)."""
        return [
            RustStreamQuarantine(
                game_id=str(item.game_id),
                reason_code=str(item.reason_code),
                detail=str(item.detail),
            )
            for item in self._handle.quarantines()
        ]

    def close(self) -> None:
        """Idempotent close (second call is a no-op success)."""
        if self._closed:
            return
        self._closed = True
        self._handle.close()

    def __enter__(self) -> RustJsonStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        batch = self.next()
        while batch:
            yield from batch
            batch = self.next()


def open_rust_stream(
    data_dirs: list[str],
    batch: int,
    workers: int = 0,
    queue: int = 4,
    split: str = "train",
    spec_hash: str = "",
    capacity_bytes: int = 0,
    rules_hash: str | None = None,
    adapter_hash: str | None = None,
    action_table_hash: str | None = None,
) -> RustJsonStream:
    """Open a serial DecisionRow-JSON stream (see :class:`RustJsonStream`).

    Omitted digests resolve from the Python oracle (byte-identical to the
    Rust ``PINNED_*`` authorities); explicit digests bind strictly.
    """
    return RustJsonStream(
        data_dirs=data_dirs,
        batch=batch,
        workers=workers,
        queue=queue,
        split=split,
        spec_hash=spec_hash,
        capacity_bytes=capacity_bytes,
        rules_hash=rules_hash,
        adapter_hash=adapter_hash,
        action_table_hash=action_table_hash,
    )
