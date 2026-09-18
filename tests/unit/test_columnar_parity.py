"""Probe-C: same-batches-to-both-writers parity (Rust ArrowWriter vs pq.write_table).

Contract under test: the Rust 5-exact writer
(``tools/hydra2-replay-rs/crates/hydra-shard/src/writer.rs``:
``write_parity_probe`` over ``writer_props_5exact``) writes the IDENTICAL
logical batch as the Python oracle call (``parquet.py:238-246`` 5-exact:
zstd/level-3/dict/batch-8192/store_schema). Comparison is by file-content
class — schema names + logical types, row counts, per-column logical values
in file order, ``dataset_hash`` over sorted ids, footer close-counts +
``store_schema`` KV + default ``created_by`` — NEVER byte-identical (writer
internals differ: ``created_by`` strings, page offsets, stats truncation).

Gates (every mismatch raises, never warns-and-continues):
- ``dataset_hash`` equal on both read-back files (sorted ids);
- per-column logical types + values equal, file order included;
- footer asserts green on both (rows, row groups, ``ARROW:schema`` KV,
  non-empty default ``created_by``);
- 6th-knob-creep grep: ``writer.rs`` code contains ONLY the 5-exact setters.

Cutover contract (NOT yet performed — documented next step below):
- The Python writer stays primary until Probe-C is green.
- A future caller cutover may prefer the Rust entry, but the ONLY acceptable
  fallback trigger is ``ImportError`` (extension not built); any
  ``dataset_hash`` / logical-value / footer mismatch MUST raise fail-closed.
- Next step (separate ticket, bridge-owner surface): expose
  ``write_parity_probe`` through the ``hydra-bridge`` columnar submodule,
  then flip one caller behind the ImportError-only fallback with this test
  as the gate. This file performs no caller migration and no fallback logic.

Parity table: printed by the main test (per-column compression / dict /
stats / row-group / page observations on both files) and returned in the
ticket yield; pytest captures it (visible with ``-s`` or on failure).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hydra2 import _rust_columnar
from hydra2.data.parquet import _ACTOR_SCHEMA, ACTOR_FIELDS

pytestmark = pytest.mark.slow

_N = 24
_STEP = 7
_RUST_PROBE_FILTER = "probe_c_parity_batch_round_trip"
_RUST_PROBE_FILE = "probe-c-rust.parquet"

#: The ONLY writer-property setters allowed in ``writer.rs`` code (comments
#: stripped): the 5-exact surface. Any 6th knob fails the creep gate below.
_FIVE_EXACT_SETTERS = frozenset(
    {
        "set_compression",
        "set_dictionary_enabled",
        "set_write_batch_size",
        "set_key_value_metadata",
    }
)


def _probe_ids() -> list[str]:
    """Canonical decision ids in FILE order (permuted, NOT sorted).

    Mirror of ``writer.rs::parity_probe_ids`` (``(i * 7) % 24`` is a full
    permutation: 7 is coprime to 24). A one-sided value edit raises in the
    file-id asserts below (mismatch=raise).
    """
    return [f"probe-c:d{(i * _STEP) % _N:04d}" for i in range(_N)]


def _probe_table() -> pa.Table:
    """Canonical 24-row batch with the pinned logical values both writers use.

    Value mirror of ``writer.rs::parity_probe_batch``; schema is the real
    ``_ACTOR_SCHEMA`` (13 ``ACTOR_FIELDS``, ``chosen_action_id`` sole
    nullable), exactly like ``write_actor_shards`` builds it.
    """
    return pa.table(
        {
            "game_id": ["probe-c-g0000"] * _N,
            "round_id": [f"probe-c-g0000:r{i % 4:02d}" for i in range(_N)],
            "decision_id": _probe_ids(),
            "seat": [i % 4 for i in range(_N)],
            "source_object_id": ["probe-c-src"] * _N,
            "split": ["train"] * _N,
            "rules_hash": ["sha256:" + "a" * 64] * _N,
            "adapter_hash": ["sha256:" + "b" * 64] * _N,
            "observation_hash": ["sha256:" + "c" * 64] * _N,
            "action_table_hash": ["sha256:" + "d" * 64] * _N,
            "derivation_hash": ["sha256:" + "e" * 64] * _N,
            "actor_observation": [
                f'{{"dora_indicators":[1,2,3,4,5],"seat":{i % 4}}}' for i in range(_N)
            ],
            "chosen_action_id": [i % 9 for i in range(_N)],
        },
        schema=_ACTOR_SCHEMA,
    )


def _rust_probe_parquet(workdir: Path) -> Path:
    """Run the Rust Probe-C judge and return its emitted parquet path.

    The judge (``writer.rs`` probe test named by ``_RUST_PROBE_FILTER``)
    writes ``probe-c-rust.parquet`` under ``HYDRA2_PROBEC_OUT`` (scratch-only
    discipline enforced Rust-side). Fail closed: nonzero exit or a missing
    file raises — never a silent skip.
    """
    root = Path(__file__).resolve().parents[2]
    crate = root / "tools" / "hydra2-replay-rs"
    emit = workdir / "rust-out"
    emit.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYO3_PYTHON": sys.executable, "HYDRA2_PROBEC_OUT": str(emit)}
    proc = subprocess.run(
        [
            "cargo",
            "nextest",
            "run",
            "--offline",
            "-p",
            "hydra-shard",
            "--lib",
            _RUST_PROBE_FILTER,
        ],
        cwd=crate,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"rust Probe-C judge failed:\n{proc.stderr[-4000:]}"
    out = emit / _RUST_PROBE_FILE
    assert out.is_file(), "rust Probe-C judge left no probe-c-rust.parquet (mismatch=raise)"
    return out


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _footer_facts(path: Path) -> dict:
    """Footer + column-chunk observations for one closed parquet file."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    kv = dict(md.metadata or {})
    rg = md.row_group(0)
    cols = []
    for i in range(rg.num_columns):
        col = rg.column(i)
        stats = col.statistics
        cols.append(
            {
                "name": col.path_in_schema,
                "compression": f"{col.compression}",
                "encodings": "+".join(f"{e}" for e in col.encodings),
                "dict_page": col.dictionary_page_offset is not None,
                "has_stats": stats is not None and stats.has_min_max,
                "stats_min": None if stats is None else stats.min,
                "stats_max": None if stats is None else stats.max,
            }
        )
    return {
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "created_by": md.created_by,
        "kv": kv,
        "cols": cols,
    }


def _print_parity_table(py_path: Path, rust_path: Path, py_facts: dict, rust_facts: dict) -> None:
    py_kv = py_facts["kv"].get(b"ARROW:schema", b"")
    rust_kv = rust_facts["kv"].get(b"ARROW:schema", b"")
    print("Probe-C parity table: same 24-row batch -> pq.write_table vs Rust ArrowWriter")
    print("file  rows  groups  created_by                       ARROW:schema  sha256[:16]")
    print(
        f"py    {py_facts['num_rows']:<4}  {py_facts['num_row_groups']:<6}  "
        f"{py_facts['created_by']:<32}  {len(py_kv):<12}  {_file_sha256(py_path)[7:23]}"
    )
    print(
        f"rust  {rust_facts['num_rows']:<4}  {rust_facts['num_row_groups']:<6}  "
        f"{rust_facts['created_by']:<32}  {len(rust_kv):<12}  {_file_sha256(rust_path)[7:23]}"
    )
    print("column             compression  dict   stats  encodings")
    trunc_notes = []
    for py_c, rust_c in zip(py_facts["cols"], rust_facts["cols"], strict=True):
        print(
            f"{py_c['name']:<18} {py_c['compression']}/{rust_c['compression']:<11} "
            f"{'yes' if py_c['dict_page'] else 'no'}/"
            f"{'yes' if rust_c['dict_page'] else 'no':<6} "
            f"{'yes' if py_c['has_stats'] else 'no'}/"
            f"{'yes' if rust_c['has_stats'] else 'no':<6} "
            f"{py_c['encodings']}"
        )
        if (
            py_c["has_stats"]
            and rust_c["has_stats"]
            and (
                py_c["stats_min"] != rust_c["stats_min"] or py_c["stats_max"] != rust_c["stats_max"]
            )
        ):
            trunc_notes.append(py_c["name"])
    if trunc_notes:
        print(
            "stats note: exact min/max bytes differ on "
            + ", ".join(trunc_notes)
            + " (parquet-rs truncates long-string stats per spec, parquet-cpp keeps "
            + "full values; both spec-legal, value-neutral — logical reads are identical)"
        )


def test_probe_c_same_batches_both_writers_parity(tmp_path: Path) -> None:
    """Same batch through both 5-exact writers: hash + values + footer agree."""
    table = _probe_table()
    assert table.schema.equals(_ACTOR_SCHEMA)
    py_path = tmp_path / "probe-c-py.parquet"
    # Python oracle call, 5-exact (mirrors parquet.py:238-246; stays primary).
    pq.write_table(
        table,
        py_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_batch_size=8192,
        store_schema=True,
    )
    rust_path = _rust_probe_parquet(tmp_path)

    py_back = pq.read_table(py_path)
    rust_back = pq.read_table(rust_path)

    # Mirror pin: both files carry the predicted permuted ids in file order.
    assert py_back.column("decision_id").to_pylist() == _probe_ids()
    assert rust_back.column("decision_id").to_pylist() == _probe_ids()

    # Gate 1: dataset_hash equal (sorted ids) on both read-back files.
    py_hash = _rust_columnar.table_dataset_hash(py_back)
    rust_hash = _rust_columnar.table_dataset_hash(rust_back)
    assert py_hash == rust_hash
    assert py_hash == _rust_columnar.dataset_hash_of_ids(_probe_ids())

    # Gate 2: file-content class — names, logical types, row counts, per-column
    # logical values in file order (ChunkedArray.equals is order-sensitive).
    # NOT bytes: created_by/page internals/stats truncation differ by writer.
    assert py_back.schema.names == list(ACTOR_FIELDS)
    assert rust_back.schema.names == list(ACTOR_FIELDS)
    assert py_back.num_rows == rust_back.num_rows == _N
    for name in py_back.schema.names:
        py_col = py_back.column(name)
        rust_col = rust_back.column(name)
        assert py_col.type == rust_col.type, name
        assert py_col.equals(rust_col), name

    # Gate 3: footer asserts green on both files.
    py_facts = _footer_facts(py_path)
    rust_facts = _footer_facts(rust_path)
    for facts in (py_facts, rust_facts):
        assert facts["num_rows"] == _N
        assert facts["num_row_groups"] == 1
        assert facts["created_by"], "created_by missing (default provenance expected)"
        assert facts["kv"].get(b"ARROW:schema"), "ARROW:schema KV missing (store_schema)"
    assert "parquet-cpp" in py_facts["created_by"]
    assert "parquet-rs" in rust_facts["created_by"]
    # Distinct default provenances prove the NOT-byte-identical class.
    assert py_facts["created_by"] != rust_facts["created_by"]
    assert _file_sha256(py_path) != _file_sha256(rust_path)
    for py_c, rust_c in zip(py_facts["cols"], rust_facts["cols"], strict=True):
        assert py_c["compression"] == rust_c["compression"] == "ZSTD", py_c["name"]
        assert py_c["dict_page"] and rust_c["dict_page"], py_c["name"]
        assert "RLE_DICTIONARY" in py_c["encodings"], py_c["name"]
        assert "RLE_DICTIONARY" in rust_c["encodings"], rust_c["name"]
        assert py_c["has_stats"] and rust_c["has_stats"], py_c["name"]

    _print_parity_table(py_path, rust_path, py_facts, rust_facts)


def test_probe_c_rust_writer_stays_five_exact() -> None:
    """6th-knob-creep gate: ``writer.rs`` code sets ONLY the 5-exact props.

    Strips ``//`` comments (the MUST-NOT-ADD doc list names forbidden setters
    on purpose) and asserts the remaining ``set_*`` tokens are exactly the
    allowlist. Any new row_group/page/dict/stats/encoding/bloom/sort setter
    fails here before it can fork ``dataset_hash`` goldens.
    """
    root = Path(__file__).resolve().parents[2]
    src = (
        root / "tools" / "hydra2-replay-rs" / "crates" / "hydra-shard" / "src" / "writer.rs"
    ).read_text(encoding="utf-8")
    code_lines = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        head = line.split("//")[0]
        assert head.count('"') % 2 == 0, (
            f"writer.rs:{lineno} has // inside a string literal; refine comment stripping"
        )
        code_lines.append(head)
    # Method-call setters only (leading dot): a bare `set_` pattern also matches
    # inside identifiers like `dataset_hash_of_sorted_ids`, which is not a knob.
    found = {"set_" + m for m in re.findall(r"\.set_(\w+)", "\n".join(code_lines))}
    assert found == set(_FIVE_EXACT_SETTERS), f"6th-knob creep in writer.rs: {sorted(found)}"
