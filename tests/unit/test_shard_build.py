"""Phase-4 offline tensor shards: build 1k rows from fixtures, manifest validates.

Builds 200 wall-bound fixture games (5 rows each, golden shape) through the
real expand -> firewall -> encode-once -> per-plane IPC path and verifies the
manifest contract end to end: sha256s, row counts, fixed [N, 256] history +
lengths, canonical order, schema digest, dataset hash, attestation, and the
games/rows/MiB-per-second build report. Serial lane (spawn pool inside).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

if TYPE_CHECKING:
    import numpy as np

from pathlib import Path

from hydra2.contracts.common import ContractError
from hydra2.data.attestation import SYNTHETIC_ATTESTATION
from hydra2.data.decode import GameRecord
from hydra2.data.replay_expand import expand_game
from hydra2.data.shard_build import (
    DECISION_IDS_FILENAME,
    FIXED_HISTORY_T,
    HISTORY_LEN_PLANE,
    _firewall_check_row,
    build_shards,
    validate_manifest,
)
from hydra2.models.encoder import input_schema_hash

pytestmark = [pytest.mark.serial, pytest.mark.slow]


@pytest.fixture(scope="session", autouse=True)
def _s8_rust_extension(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """S8 cutover: all replay flows through Rust; build the extension once."""
    import importlib
    import importlib.machinery

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


_TEHAIS = [
    ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
    ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
    ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
    ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
]
_WALL = tuple(range(136))
_NGAMES = 200
_ROWS_PER_GAME = 5


def _events(game_id: str, *, bad: bool = False) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {"type": "start_game", "game_id": game_id},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": _TEHAIS,
        },
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5p"},
        {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "5p"},
        {"type": "dahai", "actor": 3, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "dahai", "actor": 0, "pai": "6p", "tsumogiri": True},
        {"type": "end_game", "game_id": game_id, "scores": [30000, 25000, 20000, 15000]},
    ]
    if bad:
        events.insert(-1, {"type": "bogus_mid_event"})
    return events


def _make_game(index: int, *, bad: bool = False) -> GameRecord:
    game_id = f"shard-game-{index:04d}"
    return GameRecord(
        game_id=game_id,
        object_id=f"shard-object-{index:04d}",
        packaged_object_id="shard-packaged-01",
        events=tuple(_events(game_id, bad=bad)),
        raw_bytes_sha256="sha256:" + "0" * 64,
        wall_tiles=_WALL,
        source={"type": "start_game"},
    )


def _read_plane(out_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    with pa.memory_map(str(out_dir / str(entry["file"]))) as src:
        tensor = pa_ipc.read_tensor(src)
    return tensor.to_numpy()


def _plane_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [e for e in manifest["planes"] if e["name"] == name]
    assert len(matches) == 1
    return matches[0]


def test_build_1k_shards_manifest_validates(tmp_path: Path) -> None:
    """1k rows expand/encode/write; manifest validates; content bit-exact."""
    games = [_make_game(i) for i in range(_NGAMES)]
    out_dir = tmp_path / "shards"
    manifest = build_shards(games, out_dir=out_dir, expand_workers=2)

    assert manifest["rows"] == _NGAMES * _ROWS_PER_GAME == 1000
    assert manifest["chunks"] == 1
    assert manifest["order"] == {"kind": "canonical", "seed": 0}
    assert manifest["schema_digest"] == str(input_schema_hash())
    assert manifest["attestation"]["attestation_id"] == SYNTHETIC_ATTESTATION.attestation_id
    assert manifest["attestation"]["kind"] == "synthetic"
    build = manifest["build"]
    assert build["games"] == _NGAMES
    assert build["games_replayed"] == _NGAMES
    assert build["games_sim_replayed"] == 0
    assert build["games_quarantined"] == 0
    assert build["rows"] == 1000
    assert build["rows_per_s"] > 0 and build["games_per_s"] > 0 and build["mib_per_s"] > 0
    assert build["uncompressed_bytes"] > 0
    # 26 encoder features + chosen_action_id + history_len.
    assert len(manifest["planes"]) == 28
    assert {e["name"] for e in manifest["planes"]} >= {"legal_mask", "chosen_action_id"}

    # The manifest on disk is the returned manifest; it validates clean.
    on_disk = json.loads((out_dir / "manifest.json").read_bytes().decode())
    assert on_disk == manifest
    validate_manifest(manifest, out_dir=out_dir)

    # Content cross-check against the live path: per-game chosen pattern from
    # a direct serial expansion tiles across the whole shard.
    pattern = [row.chosen_action_id for row in expand_game(games[0], split="train")]
    assert len(pattern) == _ROWS_PER_GAME
    chosen = _read_plane(out_dir, _plane_by_name(manifest, "chosen_action_id"))
    assert chosen.tolist() == pattern * _NGAMES

    # Fixed [N, 256] history + explicit lengths agree with the mask sums.
    kind_entry = _plane_by_name(manifest, "history_event_kind")
    mask_entry = _plane_by_name(manifest, "history_mask")
    len_entry = _plane_by_name(manifest, HISTORY_LEN_PLANE)
    assert kind_entry["shape"] == [1000, FIXED_HISTORY_T] == mask_entry["shape"]
    assert kind_entry["dtype"] == "int64" and mask_entry["dtype"] == "bool"
    assert mask_entry["storage_dtype"] == "uint8"
    assert len_entry["shape"] == [1000]
    mask = _read_plane(out_dir, mask_entry)
    lengths = _read_plane(out_dir, len_entry)
    assert (mask.sum(axis=1) == lengths).all()
    assert bool((lengths >= 1).all() and (lengths <= FIXED_HISTORY_T).all())
    kind = _read_plane(out_dir, kind_entry)
    for row in range(0, 1000, 137):
        width = int(lengths[row])
        assert bool((kind[row, width:] == 0).all())

    # Row identity: sorted, unique, and binding the dataset hash.
    decision_ids = json.loads((out_dir / DECISION_IDS_FILENAME).read_bytes().decode())
    assert isinstance(decision_ids, list)
    assert len(decision_ids) == 1000 and len(set(decision_ids)) == 1000
    assert decision_ids == sorted(decision_ids)
    expected_hash = "sha256:" + hashlib.sha256("\n".join(decision_ids).encode()).hexdigest()
    assert manifest["dataset_hash"] == expected_hash

    # Legal mask plane keeps full baseline width and the logical bool dtype.
    legal = _plane_by_name(manifest, "legal_mask")
    assert legal["shape"] == [1000, 6792]
    assert legal["dtype"] == "bool" and legal["storage_dtype"] == "uint8"


def test_serial_parallel_bit_identity(tmp_path: Path) -> None:
    """Ordered pool merge is bit-identical to the serial reference."""
    games = [_make_game(i) for i in range(12)]
    serial = build_shards(games, out_dir=tmp_path / "serial", expand_workers=0)
    parallel = build_shards(games, out_dir=tmp_path / "parallel", expand_workers=2)
    assert serial["rows"] == parallel["rows"] == 12 * _ROWS_PER_GAME
    assert serial["dataset_hash"] == parallel["dataset_hash"]
    assert [str(e["sha256"]) for e in serial["planes"]] == [
        str(e["sha256"]) for e in parallel["planes"]
    ]
    validate_manifest(serial, out_dir=tmp_path / "serial")
    validate_manifest(parallel, out_dir=tmp_path / "parallel")


def test_quarantine_counted_not_failed(tmp_path: Path) -> None:
    """Expansion-rejected games count under reason classes; rows still build."""
    games = [_make_game(i) for i in range(10)] + [_make_game(100 + i, bad=True) for i in range(2)]
    manifest = build_shards(games, out_dir=tmp_path / "shards", expand_workers=2)
    assert manifest["rows"] == 10 * _ROWS_PER_GAME
    assert manifest["build"]["games"] == 12
    assert manifest["build"]["games_quarantined"] == 2
    assert len(manifest["build"]["quarantine_reasons"]) >= 1
    validate_manifest(manifest, out_dir=tmp_path / "shards")


def test_firewall_and_guards_fail_closed(tmp_path: Path) -> None:
    """Privileged leakage, bad knobs, and tampered manifests fail closed."""
    good_obs = {"dora_indicators": [1, 2, 3, 4, 5], "legal_mask": [True]}
    _firewall_check_row({"decision_id": "d0", "actor_observation": dict(good_obs)})
    with pytest.raises(ContractError, match="privileged field leakage"):
        _firewall_check_row(
            {"decision_id": "d0", "actor_observation": {**good_obs, "hidden_tiles": []}}
        )
    with pytest.raises(ContractError, match="privileged nested field leakage"):
        _firewall_check_row(
            {"decision_id": "d0", "actor_observation": {**good_obs, "nest": {"wall": []}}}
        )
    with pytest.raises(ContractError, match=r"\(5,\)"):
        _firewall_check_row(
            {"decision_id": "d0", "actor_observation": {**good_obs, "dora_indicators": [1, 2]}}
        )

    games = [_make_game(i) for i in range(2)]
    with pytest.raises(ContractError, match="at least one game"):
        build_shards([], out_dir=tmp_path / "empty")
    with pytest.raises(ContractError, match="quarantined all"):
        build_shards([_make_game(999, bad=True)], out_dir=tmp_path / "allbad")
    with pytest.raises(ContractError, match="order must be"):
        build_shards(games, out_dir=tmp_path / "badorder", order="shuffle")
    with pytest.raises(ContractError, match="chunk_rows"):
        build_shards(games, out_dir=tmp_path / "badchunk", chunk_rows=0)
    with pytest.raises(ContractError, match="expand_workers invalid"):
        build_shards(games, out_dir=tmp_path / "badworkers", expand_workers=-1)

    out_dir = tmp_path / "shards"
    manifest = build_shards(games, out_dir=out_dir, expand_workers=0)
    tampered = json.loads(json.dumps(manifest))
    tampered["planes"][0]["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="sha256 mismatch"):
        validate_manifest(tampered, out_dir=out_dir)
    tampered = json.loads(json.dumps(manifest))
    tampered["dataset_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ContractError, match="dataset_hash mismatch"):
        validate_manifest(tampered, out_dir=out_dir)
    missing = json.loads(json.dumps(manifest))
    (out_dir / str(missing["planes"][0]["file"])).unlink()
    with pytest.raises(ContractError, match="plane file missing"):
        validate_manifest(missing, out_dir=out_dir)
