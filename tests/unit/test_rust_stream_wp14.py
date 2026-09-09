"""Rust JSON-rows handoff over the PyO3 boundary (Slice 5, serial lane).

Drives ``hydra2.training.rust_stream`` (thin stub) over the compiled
``hydra2_replay_rs`` extension: framed per-game inputs in, validated
DecisionRow-JSON batches out. Proves the Slice-5 acceptance set:

- serial drain green with the exact 13-field envelope on every row;
- rows bind the minted Slice-5 digests (``rules_hash``/``adapter_hash``/
  ``action_table_hash`` from the pinned authorities, byte-identical to the
  Python oracle; the legacy ``spec_hash`` echo is gone);
- determinism: two full passes byte-identical at the FFI boundary;
- stats ``open_count == 1`` with ok/quarantine/rows counters exact;
- quarantine classes identical to Slice 3 (``double-ron`` only here; the
  JSON path adds zero new reason codes);
- fail-closed edges: tiny buffer, ``workers > 0``, bad open args, empty or
  mismatched digest bindings, and ``HYDRA2_ARTIFACT_ROOT`` inside a raw root.

The ``rust_extension`` fixture builds the extension once per session
(``cargo build`` is incremental after the first compile) and maps the
cdylib onto ``sys.path`` under its import name; every test takes the
fixture so the import path is live while the stub opens streams.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import hydra2.training.rust_stream as rust_stream
from hydra2.data.parquet import ACTOR_FIELDS
from hydra2.training.replay import FORBIDDEN_REPLAY_KEYS

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.serial]

SPEC_HASH = "slice4-serial-probe-v1"


def _expected_digests() -> tuple[str, str, str]:
    """Pinned Slice-5 authorities from the Python oracle (== Rust ``PINNED_*``)."""
    from hydra2.config import repo_root
    from hydra2.contracts.action import ACTION_TABLE_RELPATH, load_action_table
    from hydra2.data.replay_expand import _adapter_hash, _load_rules
    from hydra2.engines.riichienv.adapter import _rules_identity
    from hydra2.engines.riichienv.state import rules_identity_hash

    rules = _load_rules()
    return (
        _rules_identity(rules, str(rules_identity_hash(rules))),
        _adapter_hash(),
        str(load_action_table(repo_root() / ACTION_TABLE_RELPATH).digest),
    )


#: Slice-3 quarantine taxonomy (tools/hydra2-replay-rs/README.md): the JSON
#: path must introduce zero new reason codes.
SLICE3_QUARANTINE_CODES = frozenset(
    {
        "framing",
        "wall-bearing",
        "bare-dora",
        "double-ron",
        "unknown-event",
        "tile-conservation",
        "turn-order",
        "claim-no-offer",
        "draw-past-wall",
        "kyushu-ambiguous",
        "unmapped-ryukyoku-reason",
        "action-id-unresolved",
    }
)

_TILES = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
]


def _tile_at(slot: int) -> str:
    return _TILES[slot % len(_TILES)]


def _tehais() -> list[list[str]]:
    return [[_tile_at(seat * 13 + i) for i in range(13)] for seat in range(4)]


def _winning_tehais(seat: int) -> list[list[str]]:
    """Tehais where `seat` is tenpai (1112223334445m) for an engine-honest win.

    All four hands share one conservation budget (<=4 copies per tile):
    the winner's manzu are exclusive to it; other seats hold honors/sou
    only. Intermediate tsumogiri draws (1s/2s/3s/4s) stay within budget.
    """
    others = [
        ["E", "E", "E", "S", "S", "S", "W", "W", "W", "N", "N", "N", "P"],
        ["F", "F", "F", "C", "C", "C", "1s", "1s", "1s", "2s", "2s", "2s", "3s"],
        ["4s", "4s", "4s", "5s", "5s", "5s", "6s", "6s", "6s", "7s", "7s", "7s", "8s"],
    ]
    hands: list[list[str]] = []
    pool = list(others)
    for index in range(4):
        if index == seat:
            hands.append(
                [
                    "1m",
                    "1m",
                    "1m",
                    "2m",
                    "2m",
                    "2m",
                    "3m",
                    "3m",
                    "3m",
                    "4m",
                    "4m",
                    "4m",
                    "5m",
                ]
            )
        else:
            hands.append(pool.pop(0))
    return hands


def _framed_game(
    draws_before: int,
    tail: list[str],
    *,
    tehais: list[list[str]] | None = None,
) -> str:
    lines = [
        '{"type":"start_game"}',
        json.dumps(
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "dora_marker": "3m",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "scores": [25000, 25000, 25000, 25000],
                "tehais": tehais if tehais is not None else _tehais(),
            }
        ),
    ]
    for turn in range(draws_before):
        seat = turn % 4
        pai = _tile_at(52 + turn)
        lines.append(json.dumps({"type": "tsumo", "actor": seat, "pai": pai}))
        lines.append(json.dumps({"type": "dahai", "actor": seat, "pai": pai, "tsumogiri": True}))
    lines.extend(tail)
    lines.append('{"type":"end_game"}')
    return "\n".join(lines) + "\n"


def _tsumo_win_game(draws_before: int) -> str:
    seat = draws_before % 4
    # Engine-honest win: tenpai 1112223334445m + tsumo 5m (tanyao+tsumo).
    return _framed_game(
        draws_before,
        [
            json.dumps({"type": "tsumo", "actor": seat, "pai": "5m"}),
            json.dumps(
                {
                    "type": "hora",
                    "actor": seat,
                    "target": seat,
                    "tsumo": True,
                    "deltas": [8000, -2000, -2000, -4000],
                }
            ),
            '{"type":"end_kyoku"}',
        ],
        tehais=_winning_tehais(seat),
    )


def _double_ron_game() -> str:
    return _framed_game(
        0,
        [
            json.dumps({"type": "tsumo", "actor": 0, "pai": "E"}),
            json.dumps({"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True}),
            json.dumps(
                {
                    "type": "hora",
                    "actor": 1,
                    "target": 0,
                    "deltas": [8000, 8000, -8000, -8000],
                }
            ),
            json.dumps(
                {
                    "type": "hora",
                    "actor": 2,
                    "target": 0,
                    "deltas": [8000, -8000, 8000, -8000],
                }
            ),
        ],
    )


def _write_inputs(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "gA-good.jsonl").write_text(_tsumo_win_game(3))
    (inputs / "gB-good.jsonl").write_text(_tsumo_win_game(4))
    (inputs / "gC-double-ron.jsonl").write_text(_double_ron_game())
    return inputs


@pytest.fixture(scope="session")
def rust_extension(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build the extension once and expose ``import hydra2_replay_rs``."""
    root = Path(__file__).resolve().parents[2]
    crate = root / "tools" / "hydra2-replay-rs"
    env = {**os.environ, "PYO3_PYTHON": sys.executable}
    proc = subprocess.run(
        ["cargo", "build", "--offline", "-p", "hydra2-replay-rs"],
        cwd=crate,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cargo build failed:\n{proc.stderr[-4000:]}"
    built = crate / "target" / "debug" / "libhydra2_replay_rs.so"
    assert built.is_file(), f"expected cdylib at {built}"
    ext_dir = tmp_path_factory.mktemp("hydra2_replay_rs")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    shutil.copy(built, ext_dir / f"hydra2_replay_rs{suffix}")
    sys.path.insert(0, str(ext_dir))
    try:
        yield importlib.import_module("hydra2_replay_rs")
    finally:
        sys.path.remove(str(ext_dir))


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Artifact root on scratch, raw roots unset (pass path of the gate)."""
    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.delenv("HYDRA2_DATA_ROOT", raising=False)
    monkeypatch.delenv("HYDRA2_TENHOU_MOUNT", raising=False)
    return tmp_path


def _drain_raw(ext: Any, data_dir: Path, batch: int) -> bytes:
    """Drain one full pass at the FFI boundary (raw payload bytes)."""
    handle = ext.PyHydra2ReplayStream.open([str(data_dir)], batch, 0, 4, "train", SPEC_HASH)
    try:
        out = bytearray()
        buf = bytearray(1 << 20)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        while True:
            nxt = handle.next_into_json(addr, len(buf))
            out += buf[: int(nxt.bytes_written)]
            if int(nxt.rows) == 0 and int(nxt.games_consumed) == 0:
                break
        return bytes(out)
    finally:
        handle.close()


def test_walk_stats_quarantine_envelope(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    rules_hash, adapter_hash, action_table_hash = _expected_digests()
    stream = rust_stream.RustJsonStream(
        [str(inputs)],
        batch=2,
        split="train",
        spec_hash=SPEC_HASH,
        rules_hash=rules_hash,
        adapter_hash=adapter_hash,
        action_table_hash=action_table_hash,
    )
    rows = list(stream)
    # gA: 3 dahai + tsumo-hora; gB: 4 dahai + tsumo-hora; gC quarantined.
    assert len(rows) == 9
    assert len({row["decision_id"] for row in rows}) == 9
    for row in rows:
        assert set(row) == set(ACTOR_FIELDS)
        assert not (set(row) & set(FORBIDDEN_REPLAY_KEYS))
        assert row["split"] == "train"
        # Minted authorities bind every row; the Slice-4 spec_hash echo is gone.
        assert row["rules_hash"] == rules_hash
        assert row["adapter_hash"] == adapter_hash
        assert row["action_table_hash"] == action_table_hash
        assert row["rules_hash"] != SPEC_HASH
        assert row["adapter_hash"] != SPEC_HASH
        actor_doc = json.loads(row["actor_observation"])
        assert not (set(actor_doc) & set(FORBIDDEN_REPLAY_KEYS))
        assert row["chosen_action_id"] is None or isinstance(row["chosen_action_id"], int)
    stats = stream.stats()
    assert stats.open_count == 1
    assert stats.games_ok == 2
    assert stats.games_quarantined == 1
    assert stats.rows_out == 9
    quarantines = stream.quarantines()
    assert [item.reason_code for item in quarantines] == ["double-ron"]
    assert {item.reason_code for item in quarantines} <= SLICE3_QUARANTINE_CODES
    stream.close()
    stream.close()
    with pytest.raises(ValueError, match="closed"):
        stream.stats()
    with pytest.raises(ValueError, match="closed"):
        stream.next()


def test_two_passes_byte_identical(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    first = _drain_raw(rust_extension, inputs, batch=2)
    second = _drain_raw(rust_extension, inputs, batch=2)
    assert first
    assert first == second


def test_default_binding_mints_oracle_digests(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Omitted digests resolve from the oracle (the driver open shape)."""
    inputs = _write_inputs(tmp_path)
    rules_hash, adapter_hash, action_table_hash = _expected_digests()
    with rust_stream.RustJsonStream(
        [str(inputs)], batch=2, split="train", spec_hash=SPEC_HASH
    ) as stream:
        rows = list(stream)
    assert len(rows) == 9
    for row in rows:
        assert row["rules_hash"] == rules_hash
        assert row["adapter_hash"] == adapter_hash
        assert row["action_table_hash"] == action_table_hash


def test_open_digest_args_fail_closed(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    """Empty digest bindings refuse at open (fail closed on missing)."""
    inputs = _write_inputs(tmp_path)
    _rules_hash, adapter_hash, action_table_hash = _expected_digests()
    with pytest.raises(ValueError, match="rules_hash"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, spec_hash=SPEC_HASH, rules_hash="")
    with pytest.raises(ValueError, match="adapter_hash"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, spec_hash=SPEC_HASH, adapter_hash="")
    with pytest.raises(ValueError, match="action_table_hash"):
        rust_stream.RustJsonStream(
            [str(inputs)], batch=2, spec_hash=SPEC_HASH, action_table_hash=""
        )
    # A wrong pin binds nothing: the drain fails closed on first batch.
    with (
        rust_stream.RustJsonStream(
            [str(inputs)],
            batch=2,
            split="train",
            spec_hash=SPEC_HASH,
            rules_hash="sha256:" + "0" * 64,
            adapter_hash=adapter_hash,
            action_table_hash=action_table_hash,
        ) as stream,
        pytest.raises(ValueError, match="rules_hash"),
    ):
        list(stream)


def test_row_digest_mismatch_fail_closed(
    rust_extension: Any, clean_env: Path, tmp_path: Path
) -> None:
    """Forged or dropped digests fail closed at the validation gate."""
    inputs = _write_inputs(tmp_path)
    rules_hash, adapter_hash, action_table_hash = _expected_digests()
    raw = _drain_raw(rust_extension, inputs, batch=2)
    first, *rest = [line for line in raw.split(b"\n") if line]
    doc = json.loads(first)
    with rust_stream.RustJsonStream(
        [str(inputs)],
        batch=2,
        split="train",
        spec_hash=SPEC_HASH,
        rules_hash=rules_hash,
        adapter_hash=adapter_hash,
        action_table_hash=action_table_hash,
    ) as stream:
        # Slice-4 echo shape no longer binds: spec_hash in a digest field dies.
        echo = dict(doc)
        echo["rules_hash"] = SPEC_HASH
        with pytest.raises(ValueError, match="rules_hash"):
            stream._validated_rows(b"\n".join([json.dumps(echo).encode(), *rest]) + b"\n")
        # Wrong adapter pin dies on the adapter gate.
        forged = dict(doc)
        forged["adapter_hash"] = "sha256:" + "f" * 64
        with pytest.raises(ValueError, match="adapter_hash"):
            stream._validated_rows(json.dumps(forged).encode() + b"\n")
        # Dropped content digest dies on the envelope gate (missing = fork).
        dropped = dict(doc)
        del dropped["action_table_hash"]
        with pytest.raises(ValueError, match="exactly ACTOR_FIELDS"):
            stream._validated_rows(json.dumps(dropped).encode() + b"\n")
        # Untouched payload still validates (the gate targets digests only).
        assert len(stream._validated_rows(raw)) == 9


def test_buffer_too_small_fail_closed(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    handle = rust_extension.PyHydra2ReplayStream.open([str(inputs)], 64, 0, 4, "train", SPEC_HASH)
    try:
        tiny = bytearray(8)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(tiny))
        with pytest.raises(BufferError):
            handle.next_into_json(addr, len(tiny))
    finally:
        handle.close()
    # Nothing drained by the failed call: a stub pass still yields all rows.
    with rust_stream.RustJsonStream(
        [str(inputs)], batch=2, split="train", spec_hash=SPEC_HASH
    ) as stream:
        assert len(list(stream)) == 9


def test_open_args_fail_closed(rust_extension: Any, clean_env: Path, tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="workers"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, workers=1, spec_hash=SPEC_HASH)
    with pytest.raises(ValueError, match="workers"):
        rust_extension.PyHydra2ReplayStream.open([str(inputs)], 2, 1, 4, "train", SPEC_HASH)
    with pytest.raises(ValueError, match="split"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, split="test", spec_hash=SPEC_HASH)
    with pytest.raises(ValueError, match="spec_hash"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, spec_hash="")
    with pytest.raises(ValueError, match="data_dirs"):
        rust_stream.RustJsonStream([], batch=2, spec_hash=SPEC_HASH)


def test_artifact_root_inside_raw_root_fail_closed(
    rust_extension: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _write_inputs(tmp_path)
    artifact = tmp_path / "artifacts"
    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(artifact))
    monkeypatch.setenv("HYDRA2_DATA_ROOT", str(artifact))
    with pytest.raises(ValueError, match="outside HYDRA2_ARTIFACT_ROOT"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, spec_hash=SPEC_HASH)
    monkeypatch.setenv("HYDRA2_DATA_ROOT", str(artifact / "nested"))
    with pytest.raises(ValueError, match="outside HYDRA2_ARTIFACT_ROOT"):
        rust_stream.RustJsonStream([str(inputs)], batch=2, spec_hash=SPEC_HASH)
