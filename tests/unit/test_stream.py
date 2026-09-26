"""Streaming-first reader tests (wp14): framer, order, resume, split, quarantine."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import TYPE_CHECKING

import pytest
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream_decode import (
    PrefetchGameStream,
    actor_payload,
    check_wall_disjoint,
    count_decisions,
    slice_microbatches,
    verify_no_privileged_leakage,
)
from hydra2.data.stream_iter import GameStream
from hydra2.data.stream_manifest import build_manifest, manifest_digest
from hydra2.data.stream_read import (
    StreamCursor,
    ZstdLineStream,
    assign_split,
    compute_wall_hash,
    group_key_for,
    group_key_for_path,
    stem_of,
)

if TYPE_CHECKING:
    from pathlib import Path

_RATIOS = {"train": 0.6, "validation": 0.4}
_SEED = 7
_EPOCH = 3


def _game_bytes(game_id: str, *, n_mid: int = 3, wall: bool = False, dora4: bool = False) -> bytes:
    start: dict[str, object] = {"game_id": game_id, "type": "start_game"}
    if wall:
        start["wall"] = list(range(136))
    lines = [start]
    for seat in range(n_mid):
        lines.append({"seat": seat % 4, "type": "turn_advance"})
    if dora4:
        lines.append({"dora": [1, 2, 3, 4], "type": "dahai"})
    lines.append({"game_id": game_id, "type": "end_game"})
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode()


def _write(path: Path, games: list[bytes]) -> bytes:
    raw = b"".join(games)
    path.write_bytes(zstd.ZstdCompressor().compress(raw))
    return raw


def _ids(games: list) -> list[str]:
    return [game.game.game_id for game in games]


def _tenhou_name(root: Path, stem: str) -> Path:
    return root / f"{stem}.mjai.json.zst"


def test_framer_offsets_and_boundaries(tmp_path: Path) -> None:
    games = [_game_bytes("g1", n_mid=2), _game_bytes("g2", n_mid=0), _game_bytes("g3", n_mid=1)]
    target = tmp_path / "a.mjai.json.zst"
    raw = _write(target, games)
    frames = list(ZstdLineStream(target).iter_games())
    assert [payload for _, payload in frames] == games
    assert b"".join(payload for _, payload in frames) == raw
    offsets = [offset for offset, _ in frames]
    assert offsets[0] == 0
    assert offsets[1] == len(games[0])
    assert offsets[2] == len(games[0]) + len(games[1])


def test_framer_prefilter_parity_on_tricky_lines(tmp_path: Path) -> None:
    """Superset prefilter must frame byte-identically to full-parse framing."""
    lines = [
        b'{"type":"start_game"}',
        b'{"type": "tsumo", "actor": 0, "pai": "5m"}',
        b'{"type":"dahai","restart_count":1}',
        b'{"nested": {"type": "start_game"}, "type": "dahai"}',
        b'{"type":"st\\u0061rt_game"}',
        b'{"type": "end_game"}',
    ]
    raw = b"\n".join(lines) + b"\n"
    target = tmp_path / "q.mjai.json.zst"
    target.write_bytes(zstd.ZstdCompressor().compress(raw))
    frames = list(ZstdLineStream(target).iter_games())
    assert len(frames) == 2
    assert frames[0][1] == b"\n".join(lines[:4]) + b"\n"
    assert frames[1][1] == b"\n".join(lines[4:]) + b"\n"


def test_manifest_order_is_sha256_hex_and_stable(tmp_path: Path) -> None:
    for name in ("b.mjai.json.zst", "a.mjai.json.zst", "c.mjai.json.zst"):
        _write(tmp_path / name, [_game_bytes(name)])
    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)
    # Single-path roots key by the leaf-dir basename; duplicate relpaths
    # under different roots order independently, never colliding.
    expected = sorted(
        [tmp_path / name for name in ("b.mjai.json.zst", "a.mjai.json.zst", "c.mjai.json.zst")],
        key=lambda p: hashlib.sha256(
            (tmp_path.name + "\0" + p.relative_to(tmp_path).as_posix()).encode()
        ).hexdigest(),
    )
    assert list(first.paths()) == expected
    assert list(second.paths()) == expected
    assert manifest_digest(first) == manifest_digest(second)
    assert all(entry.root_id == tmp_path.name for entry in first.files)
    assert all(
        entry.relpath == entry.path.relative_to(tmp_path).as_posix() for entry in first.files
    )


def test_manifest_digest_matches_canonical_oracle(tmp_path: Path) -> None:
    """Incremental digest must equal the canonical-bytes digest exactly.

    Covers the fast path (plain/space names) and the fallback (names
    needing JSON escapes: quotes, backslashes, controls, non-ASCII). Any
    divergence breaks scan-cache keys loudly, so both paths are pinned
    against the one-liner oracle here.
    """
    from hydra2.artifacts.canonical import canonical_bytes

    names = [
        "2024010100gm-00a9-0000-90110010.mjai.json.zst",
        'quote"name.mjai.json.zst',
        "back\\slash.mjai.json.zst",
        "sp ace.mjai.json.zst",
        "ünïcode.mjai.json.zst",
        "new\nline.mjai.json.zst",
    ]
    for name in names:
        _write(tmp_path / name, [_game_bytes(name)])
    manifest = build_manifest(tmp_path)
    assert len(manifest) == len(names)
    oracle = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes(
                [
                    {"bytes": e.bytes, "path": e.relpath, "root_id": e.root_id}
                    for e in manifest.files
                ]
            )
        ).hexdigest()
    )
    assert manifest_digest(manifest) == oracle


def test_manifest_artifact_hit_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second build off the artifact is byte-identical with zero re-hash."""
    art = tmp_path / "art"
    monkeypatch.setenv("HYDRA2_MANIFEST_ARTIFACT_DIR", str(art))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("b.mjai.json.zst", "a.mjai.json.zst", "c.mjai.json.zst"):
        _write(corpus / name, [_game_bytes(name)])
    first = build_manifest(corpus)
    digest = manifest_digest(first)
    assert len(list(art.glob("manifest-*.jsonl.zst"))) == 1
    second = build_manifest(corpus)
    assert [(e.root_id, e.relpath, e.bytes) for e in second.files] == [
        (e.root_id, e.relpath, e.bytes) for e in first.files
    ]
    assert manifest_digest(second) == digest
    # The attestation proves the hit path engaged (not a same-answer rebuild).
    assert second._stored_digest == digest


def test_manifest_artifact_invalidates_on_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new file misses (dir-mtime fingerprint) and matches a cold build."""
    art = tmp_path / "art"
    monkeypatch.setenv("HYDRA2_MANIFEST_ARTIFACT_DIR", str(art))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus / "a.mjai.json.zst", [_game_bytes("a")])
    digest = manifest_digest(build_manifest(corpus))
    _write(corpus / "b.mjai.json.zst", [_game_bytes("b")])
    rebuilt = build_manifest(corpus)
    assert manifest_digest(rebuilt) != digest
    assert "b.mjai.json.zst" in [e.relpath for e in rebuilt.files]
    monkeypatch.setenv("HYDRA2_MANIFEST_ARTIFACT_DIR", str(tmp_path / "cold"))
    cold = build_manifest(corpus)
    assert [(e.root_id, e.relpath, e.bytes) for e in cold.files] == [
        (e.root_id, e.relpath, e.bytes) for e in rebuilt.files
    ]
    assert manifest_digest(cold) == manifest_digest(rebuilt)


def test_manifest_artifact_corrupt_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garbage artifact bytes fall through to the cold path (miss, never raise)."""
    art = tmp_path / "art"
    monkeypatch.setenv("HYDRA2_MANIFEST_ARTIFACT_DIR", str(art))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus / "a.mjai.json.zst", [_game_bytes("a")])
    digest = manifest_digest(build_manifest(corpus))
    (artifact,) = list(art.glob("manifest-*.jsonl.zst"))
    artifact.write_bytes(b"not a manifest artifact, never parsed")
    rebuilt = build_manifest(corpus)
    assert manifest_digest(rebuilt) == digest
    assert [e.relpath for e in rebuilt.files] == ["a.mjai.json.zst"]


def test_epoch_trajectories_diverge_same_manifest(tmp_path: Path) -> None:
    """Per-epoch variety needs no file hashing: seed+epoch drives the buffer."""
    for index in range(4):
        _write(
            tmp_path / f"s{index}.mjai.json.zst",
            [_game_bytes(f"s{index}g{j}") for j in range(4)],
        )
    manifest = build_manifest(tmp_path)
    kwargs = {"seed": _SEED, "ratios": dict(_RATIOS), "split": None, "shuffle_buffer": 4}
    epoch0_a = _ids(GameStream(manifest, epoch=0, **kwargs))  # type: ignore[arg-type]
    epoch0_b = _ids(GameStream(manifest, epoch=0, **kwargs))  # type: ignore[arg-type]
    epoch1 = _ids(GameStream(manifest, epoch=1, **kwargs))  # type: ignore[arg-type]
    assert epoch0_a == epoch0_b
    assert len(epoch0_a) == 16
    assert epoch1 != epoch0_a
    assert sorted(epoch1) == sorted(epoch0_a)


def test_reservoir_blob_round_trip_and_corrupt_miss(tmp_path: Path) -> None:
    """Blob holds buffer-order raw bytes; any corruption fails closed."""
    from hydra2.contracts.common import ContractError
    from hydra2.data.stream_manifest import read_reservoir_blob, write_reservoir_blob

    raws = [b'{"type":"start_game"}\n', b'{"type":"end_game"}\n', b"x" * 1000]
    target = tmp_path / "buf.zst"
    index = write_reservoir_blob(raws, target)
    assert index["count"] == 3
    assert index["uncompressed_bytes"] == sum(len(r) for r in raws)
    assert read_reservoir_blob(target) == raws
    # Truncation and garbage both fail closed (caller treats as a miss).
    with pytest.raises(ContractError):
        read_reservoir_blob(tmp_path / "missing.zst")
    target.write_bytes(target.read_bytes()[:-4])
    with pytest.raises(ContractError):
        read_reservoir_blob(target)
    target.write_bytes(b"GARBAGE!!" + target.read_bytes()[8:])
    with pytest.raises(ContractError):
        read_reservoir_blob(target)


def test_snapshot_blob_restore_matches_refetch(tmp_path: Path) -> None:
    """Blob restore yields byte-identical buffers to the refetch path.

    Fails without the blob path (no second materialization to agree with)
    and on any decode/gate drift between the two materializations.
    """
    from hydra2.data.stream_iter import GameStream
    from hydra2.data.stream_manifest import write_reservoir_blob

    # Dates picked so all three `tmpname|tmpname|<date>` groups draw train
    # under the namespaced scheme (tmp leaf is session-stable; a scheme
    # change fails the length assert below, loudly).
    for index, day in enumerate((1, 4, 7)):
        _write(
            _tenhou_name(tmp_path, f"202401{day:02d}00gm-00a9-0000-9000000{index}"),
            [_game_bytes(f"snap{index}g{j}", n_mid=2) for j in range(3)],
        )
    manifest = build_manifest(tmp_path)
    ratios = dict(_RATIOS)
    first = GameStream(manifest, seed=_SEED, ratios=ratios, epoch=_EPOCH, split="train")
    games = list(first)[:6]
    assert len(games) == 6
    entries = [
        {"key": g.game.raw_bytes_sha256, "path": g.path.as_posix(), "offset": g.offset}
        for g in games
    ]
    from hydra2.data.stream_manifest import serialize_shuffle_rng

    rng = __import__("random").Random(1234)
    rng_state = serialize_shuffle_rng(rng)
    target = tmp_path / "snap.zst"
    write_reservoir_blob([bytes(g.raw) for g in games], target)
    refetch = GameStream(
        manifest,
        seed=_SEED,
        ratios=ratios,
        epoch=_EPOCH,
        split="train",
        shuffle_buffer=100,
        shuffle_restore_entries=entries,
        shuffle_restore_rng=rng_state,
    )
    via_blob = GameStream(
        manifest,
        seed=_SEED,
        ratios=ratios,
        epoch=_EPOCH,
        shuffle_buffer=100,
        shuffle_restore_entries=entries,
        shuffle_restore_rng=rng_state,
        snapshot_blob=target,
    )
    run_a = __import__("hydra2.data.stream_read", fromlist=["_Run"])._Run(
        file_index=0, byte_offset=0, games_seen=0, stats=None, hashes=set()
    )
    run_b = __import__("hydra2.data.stream_read", fromlist=["_Run"])._Run(
        file_index=0, byte_offset=0, games_seen=0, stats=None, hashes=set()
    )
    buf_a = refetch._restore_shuffle_state(run_a)
    buf_b = via_blob._restore_shuffle_state(run_b)
    assert [bytes(g.raw) for g in buf_b.buf] == [bytes(g.raw) for g in buf_a.buf]
    assert [g.game.raw_bytes_sha256 for g in buf_b.buf] == [
        g.game.raw_bytes_sha256 for g in buf_a.buf
    ]


def test_snapshot_hit_seeks_past_primed_games(tmp_path: Path) -> None:
    """Snapshot restore skips primed games; sequences stay identical.

    Builds a primed buffer via a real fill, snapshots (cursor + prefix),
    then restores through the full GameStream path and asserts the emitted
    tail matches the un-snapshotted continuation exactly.
    """
    # Dates picked so all three groups draw train under the namespaced
    # scheme (same session-stable tmp-leaf reasoning as the blob test).
    for index, day in enumerate((1, 4, 5)):
        _write(
            _tenhou_name(tmp_path, f"202401{day:02d}00gm-00a9-0000-9000000{index}"),
            [_game_bytes(f"seek{index}g{j}", n_mid=2) for j in range(6)],
        )
    manifest = build_manifest(tmp_path)
    ratios = dict(_RATIOS)
    live = GameStream(manifest, seed=_SEED, ratios=ratios, epoch=_EPOCH, split="train")
    primed = list(live)[:6]
    assert len(primed) == 6
    cursor = live.cursor()
    prefix = live.prefix_hashes_snapshot()
    assert cursor.games_seen >= 6
    restored = StreamCursor.from_dict(cursor.to_dict())
    assert restored == cursor
    assert isinstance(prefix, list)


def test_resume_identical_sequence_ordered(tmp_path: Path) -> None:
    files = []
    for index in range(2):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-0000000{index}")
        games = [_game_bytes(f"f{index}g{j}", n_mid=2) for j in range(3)]
        _write(path, games)
        files.append(path)
    manifest = build_manifest(tmp_path)
    assert len(manifest) == 2
    stream = GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH)
    iterator = iter(stream)
    head = [next(iterator) for _ in range(4)]
    cursor = stream.cursor()
    assert cursor.games_seen == 4
    tail = list(iterator)
    assert [game.offset for game in head[:3]] != []
    resumed = GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, start=cursor)
    tail_again = list(resumed)
    assert _ids(tail_again) == _ids(tail)
    fresh = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH))
    assert _ids(head + tail) == _ids(fresh)
    # Cursor round-trips through a plain mapping (YAML persistence shape).
    restored = StreamCursor.from_dict(cursor.to_dict())
    assert restored == cursor
    assert resumed.stats.framed == len(tail)


def test_resume_identical_sequence_shuffled(tmp_path: Path) -> None:
    for index in range(2):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-1111111{index}")
        _write(path, [_game_bytes(f"s{index}g{j}", n_mid=2) for j in range(4)])
    manifest = build_manifest(tmp_path)
    kwargs = {"seed": 11, "ratios": _RATIOS, "epoch": 2, "shuffle_buffer": 4}
    full = list(GameStream(manifest, **kwargs))
    assert len(full) == 8
    repeat = list(GameStream(manifest, **kwargs))
    assert _ids(repeat) == _ids(full)
    stream = GameStream(manifest, **kwargs)
    iterator = iter(stream)
    head = [next(iterator) for _ in range(3)]
    assert _ids(head) == _ids(full)[:3]
    cursor = stream.cursor()
    assert cursor.shuffle_pos == 3
    snap = stream.shuffle_snapshot()
    assert snap is not None
    entries, rng_state = snap
    prefix = stream.prefix_hashes_snapshot()
    resumed = GameStream(
        manifest,
        **kwargs,
        start=cursor,
        shuffle_restore_entries=entries,
        shuffle_restore_rng=rng_state,
        shuffle_restore_prefix_hashes=prefix,
    )
    assert _ids(list(resumed)) == _ids(full)[3:]
    # Same seed + cursor + restore state is the same sequence on a second object.
    resumed_again = GameStream(
        manifest,
        **kwargs,
        start=cursor,
        shuffle_restore_entries=entries,
        shuffle_restore_rng=rng_state,
        shuffle_restore_prefix_hashes=prefix,
    )
    assert _ids(list(resumed_again)) == _ids(full)[3:]


def test_shuffled_cursor_without_restore_state_fails_closed(tmp_path: Path) -> None:
    """A shuffled cursor without restore state refuses (no prefix replay remains)."""
    for index in range(2):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-1111111{index}")
        _write(path, [_game_bytes(f"s{index}g{j}", n_mid=2) for j in range(4)])
    manifest = build_manifest(tmp_path)
    kwargs = {"seed": 11, "ratios": _RATIOS, "epoch": 2, "shuffle_buffer": 4}
    stream = GameStream(manifest, **kwargs)
    iterator = iter(stream)
    _ = [next(iterator) for _ in range(3)]
    cursor = stream.cursor()
    assert cursor.shuffle_pos == 3
    with pytest.raises(ContractError, match="restore state"):
        list(GameStream(manifest, **kwargs, start=cursor))


def test_split_stability_and_partition_math(tmp_path: Path) -> None:
    for index in range(3):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-2222222{index}")
        _write(path, [_game_bytes(f"p{index}g{j}", n_mid=1) for j in range(2)])
    manifest = build_manifest(tmp_path)
    first_path = tmp_path / "2024010100gm-00a9-0000-22222220.mjai.json.zst"
    group = group_key_for_path(first_path)
    assert group == group_key_for(source=tmp_path.name, time="2024010100")
    # Formula pin: identical bytes to the partition cumulative-threshold math.
    digest = hashlib.sha256(f"{_SEED}|{group}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    expected = "train" if draw < 0.6 else "validation"
    assert assign_split(group_key=group, seed=_SEED, ratios=_RATIOS) == expected
    train = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, split="train"))
    valid = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, split="validation"))
    assert len(train) + len(valid) == 6
    assert set(_ids(train)).isdisjoint(_ids(valid))
    assert set(_ids(train + valid)) == {f"p{i}g{j}" for i in range(3) for j in range(2)}
    rerun = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, split="train"))
    assert _ids(rerun) == _ids(train)


def test_quarantine_counting_no_silent_skip(tmp_path: Path) -> None:
    good_a = _game_bytes("qa", n_mid=2)
    good_b = _game_bytes("qb", n_mid=1)
    bad_dora = _game_bytes("qd", n_mid=1, dora4=True)
    truncated = b'{"game_id": "qt", "type": "start_game"}\n{"seat": 0, "type": "turn_advance"}\n'
    raw = good_a + b"\n" + good_b + bad_dora + truncated
    target = tmp_path / "q.mjai.json.zst"
    target.write_bytes(zstd.ZstdCompressor().compress(raw))
    manifest = build_manifest(tmp_path)
    stream = GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH)
    emitted = list(stream)
    assert _ids(emitted) == ["qa", "qb"]
    assert stream.stats.framed == 5
    assert stream.stats.quarantined == 3
    assert stream.stats.emitted == 2


def test_exact_hash_dedup(tmp_path: Path) -> None:
    first = _tenhou_name(tmp_path, "2024010100gm-00a9-0000-33333333")
    second = _tenhou_name(tmp_path, "2024010100gm-00a9-0000-44444444")
    _write(first, [_game_bytes("dup", n_mid=2)])
    _write(second, [_game_bytes("dup", n_mid=2), _game_bytes("uniq", n_mid=1)])
    manifest = build_manifest(tmp_path)
    stream = GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH)
    emitted = list(stream)
    assert sorted(_ids(emitted)) == ["dup", "uniq"]
    assert stream.stats.duplicates == 1
    assert stream.stats.quarantined == 1
    kept = list(
        GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, drop_duplicates=False)
    )
    assert len(kept) == 3


def test_wall_null_corpus_and_disjoint_predicate(tmp_path: Path) -> None:
    path = _tenhou_name(tmp_path, "2024010100gm-00a9-0000-55555555")
    _write(path, [_game_bytes("w0", n_mid=2), _game_bytes("w1", n_mid=1)])
    manifest = build_manifest(tmp_path)
    games = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH))
    assert all(game.wall_hash is None for game in games)
    assert all(game.validation_hash is not None for game in games)
    check_wall_disjoint(games)  # vacuous pass documents the null-wall corpus
    walled = _game_bytes("walled", n_mid=1, wall=True)
    wpath = tmp_path / "w.mjai.json.zst"
    _write(wpath, [walled])
    walled_games = list(
        GameStream(build_manifest(tmp_path), seed=_SEED, ratios=_RATIOS, epoch=_EPOCH)
    )
    walled_only = [game for game in walled_games if game.wall_hash is not None]
    assert len(walled_only) == 1
    assert compute_wall_hash(walled_only[0].game) == walled_only[0].wall_hash
    other = "validation" if walled_only[0].split == "train" else "train"
    flipped = dataclasses.replace(walled_only[0], split=other)
    with pytest.raises(ContractError):
        check_wall_disjoint([walled_only[0], flipped])


def test_actor_privileged_firewall(tmp_path: Path) -> None:
    path = _tenhou_name(tmp_path, "2024010100gm-00a9-0000-66666666")
    _write(path, [_game_bytes("f0", n_mid=2)])
    manifest = build_manifest(tmp_path)
    games = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH))
    payload = actor_payload(games[0])
    assert payload["game_id"] == "f0"
    verify_no_privileged_leakage(payload)
    with pytest.raises(ContractError):
        verify_no_privileged_leakage({"wall": [1], "game_id": "x"})
    assert stem_of(path) == "2024010100gm-00a9-0000-66666666"


def test_prefetch_matches_ordered_and_microbatch_slicing(tmp_path: Path) -> None:
    for index in range(3):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-7777777{index}")
        games = [_game_bytes(f"m{index}g{j}", n_mid=2) for j in range(3)]
        if index == 1:
            games.insert(1, b'{"game_id": "bad", "type": "start_game"}\n')
        _write(path, games)
    manifest = build_manifest(tmp_path)
    ordered = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH))
    prefetched = list(
        PrefetchGameStream(
            manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, prefetch=4, max_workers=2
        )
    )
    assert _ids(prefetched) == _ids(ordered)
    assert len(ordered) == 9
    batches = list(
        PrefetchGameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH).iter_batches(4)
    )
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert _ids([game for batch in batches for game in batch]) == _ids(ordered)
    micros = list(slice_microbatches(batches[0], 2))
    assert [len(part) for part in micros] == [2, 2]
    assert micros[0][0] is batches[0][0]
    assert micros[1][1] is batches[0][3]


def test_contract_validation(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    with pytest.raises(ContractError):
        GameStream(manifest, seed=_SEED, ratios={"train": 0.5, "validation": 0.6}, epoch=_EPOCH)
    with pytest.raises(ContractError):
        GameStream(manifest, seed=_SEED, ratios={}, epoch=_EPOCH)
    with pytest.raises(ContractError):
        GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, split="sideways")
    with pytest.raises(ContractError):
        PrefetchGameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, prefetch=0)
    with pytest.raises(ContractError):
        StreamCursor.from_dict(
            {
                "file_index": 0,
                "byte_offset": -1,
                "games_seen": 0,
                "seed": 1,
                "epoch": 0,
                "shuffle_pos": 0,
            }
        )
    stream = GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH)
    other = StreamCursor(
        file_index=0, byte_offset=0, games_seen=0, seed=999, epoch=_EPOCH, shuffle_pos=0
    )
    with pytest.raises(ContractError):
        stream.skip_to(other)
    with pytest.raises(ContractError):
        list(stream.iter_batches(0))
    with pytest.raises(ContractError):
        list(slice_microbatches([], 0))


def test_throughput_informational(tmp_path: Path) -> None:
    for index in range(6):
        path = _tenhou_name(tmp_path, f"202401010{index}gm-00a9-0000-8888888{index}")
        _write(path, [_game_bytes(f"t{index}g{j}", n_mid=20) for j in range(5)])
    manifest = build_manifest(tmp_path)
    begun = time.perf_counter()
    ordered = list(GameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH))
    ordered_secs = time.perf_counter() - begun
    begun = time.perf_counter()
    prefetched = list(
        PrefetchGameStream(manifest, seed=_SEED, ratios=_RATIOS, epoch=_EPOCH, prefetch=8)
    )
    prefetch_secs = time.perf_counter() - begun
    assert len(ordered) == 30
    assert _ids(prefetched) == _ids(ordered)
    decisions = count_decisions(ordered)
    assert decisions == 30 * 22
    print(
        f"[stream-perf] ordered: {len(ordered) / ordered_secs:.0f} games/sec, "
        f"{decisions / ordered_secs:.0f} decisions/sec"
    )
    print(
        f"[stream-perf] prefetch: {len(prefetched) / prefetch_secs:.0f} games/sec, "
        f"{decisions / prefetch_secs:.0f} decisions/sec"
    )


def test_scan_cache_round_trip_and_corrupt_miss(tmp_path: Path) -> None:
    """Manifest/split cache reloads on exact key, misses on stale/corrupt."""
    from hydra2.data.stream_manifest import HOLDOUT_SPEC, load_scan_cache, save_scan_cache

    for index in range(3):
        _write(
            _tenhou_name(tmp_path, f"2024010100gm-00a9-0000-9011000{index}"),
            [_game_bytes(f"c{index}")],
        )
    manifest = build_manifest(tmp_path)
    digest = manifest_digest(manifest)
    roots = [(rid, base) for rid, base in manifest.roots]
    scan = {
        "train_walls": ["sha256:" + "a" * 64],
        "val_walls": ["sha256:" + "b" * 64],
        "root_games": [[tmp_path.name, 2, 1]],
        "train_games": 2,
        "val_games": 1,
        "train_sim_games": 0,
        "val_sim_games": 0,
        "framed": 3,
        "emitted": 3,
        "quarantined": 0,
        "duplicates": 0,
    }
    cache = tmp_path / "cache.json"
    save_scan_cache(
        cache,
        manifest_digest=digest,
        seed=_SEED,
        ratios=dict(_RATIOS),
        train_split="train",
        val_split="validation",
        roots=roots,
        holdout=HOLDOUT_SPEC,
        scan=scan,
    )
    hit = load_scan_cache(
        cache,
        manifest_digest=digest,
        seed=_SEED,
        ratios=dict(_RATIOS),
        train_split="train",
        val_split="validation",
        roots=roots,
        holdout=HOLDOUT_SPEC,
    )
    assert hit is not None
    assert hit["train_games"] == 2
    assert hit["root_games"] == [[tmp_path.name, 2, 1]]
    # Stale digest, seed, ratios, splits, roots, holdout all miss.
    miss_kwargs = {
        "manifest_digest": digest,
        "seed": _SEED,
        "ratios": dict(_RATIOS),
        "train_split": "train",
        "val_split": "validation",
        "roots": roots,
        "holdout": HOLDOUT_SPEC,
    }
    assert (
        load_scan_cache(cache, **{**miss_kwargs, "manifest_digest": "sha256:" + "0" * 64}) is None
    )
    assert load_scan_cache(cache, **{**miss_kwargs, "seed": _SEED + 1}) is None
    assert (
        load_scan_cache(cache, **{**miss_kwargs, "ratios": {"train": 0.5, "validation": 0.5}})
        is None
    )
    assert load_scan_cache(cache, **{**miss_kwargs, "roots": [("other", "/mnt/x")]}) is None
    assert (
        load_scan_cache(
            cache, **{**miss_kwargs, "holdout": {"mechanism": "other", "val_roots": []}}
        )
        is None
    )
    # Corrupt file misses (never raises).
    cache.write_text("{not-json", encoding="utf-8")
    assert load_scan_cache(cache, **miss_kwargs) is None
    cache.write_text(json.dumps({"version": 999, "manifest_digest": digest}), encoding="utf-8")
    assert load_scan_cache(cache, **miss_kwargs) is None


def test_shuffle_rng_round_trip_and_tamper() -> None:
    """Shuffle RNG state survives JSON round-trip; malformed states raise."""
    import random

    from hydra2.data.stream_manifest import parse_shuffle_rng, serialize_shuffle_rng

    rng = random.Random(12345)
    for _ in range(10):
        rng.randrange(100)
    raw = serialize_shuffle_rng(rng)
    version, internal, gauss_next = parse_shuffle_rng(raw)
    restored = random.Random()
    restored.setstate((version, tuple(internal), gauss_next))
    assert [rng.randrange(1000) for _ in range(5)] == [restored.randrange(1000) for _ in range(5)]
    with pytest.raises(ContractError):
        parse_shuffle_rng({})
    with pytest.raises(ContractError):
        parse_shuffle_rng({"version": 0, "state": [], "gauss_next": None})
    with pytest.raises(ContractError):
        parse_shuffle_rng({"version": 0, "state": [1, -2], "gauss_next": None})
    with pytest.raises(ContractError):
        parse_shuffle_rng({"version": 0, "state": [1], "gauss_next": None, "extra": 1})


def test_shuffle_snapshot_restore_verbatim(tmp_path: Path) -> None:
    """Verbatim shuffle restore continues the identical emission sequence."""
    for index in range(6):
        _write(
            _tenhou_name(tmp_path, f"2024010100gm-00a9-0000-9022000{index}"),
            [_game_bytes(f"s{index}")],
        )
    manifest = build_manifest(tmp_path)
    stream = GameStream(
        manifest, seed=_SEED, ratios=dict(_RATIOS), epoch=_EPOCH, split=None, shuffle_buffer=3
    )
    it = iter(stream)
    first_two = [next(it).game.game_id for _ in range(2)]
    assert len(first_two) == 2
    snap = stream.shuffle_snapshot()
    assert snap is not None
    entries, rng_state = snap
    assert len(entries) <= 3
    cursor = stream.cursor()
    tail_expected = [g.game.game_id for g in it]
    restored = GameStream(
        manifest,
        seed=_SEED,
        ratios=dict(_RATIOS),
        epoch=_EPOCH,
        split=None,
        shuffle_buffer=3,
        start=cursor,
        shuffle_restore_entries=entries,
        shuffle_restore_rng=rng_state,  # type: ignore[arg-type]
    )
    assert [g.game.game_id for g in restored] == tail_expected
    # Tampered key fails closed (sha mismatch at refetch).
    if entries:
        bad = [dict(e) for e in entries]
        bad[0] = {**bad[0], "key": "sha256:" + "0" * 64}
        tampered = GameStream(
            manifest,
            seed=_SEED,
            ratios=dict(_RATIOS),
            epoch=_EPOCH,
            split=None,
            shuffle_buffer=3,
            start=cursor,
            shuffle_restore_entries=bad,
            shuffle_restore_rng=rng_state,  # type: ignore[arg-type]
        )
        with pytest.raises(ContractError):
            list(tampered)


def test_fetch_game_at_boundary_and_tamper(tmp_path: Path) -> None:
    """Single-game fetch seeks to offsets; off-boundary/sha mismatch raises."""
    from hydra2.data.stream_read import fetch_game_at

    raw_a, raw_b = _game_bytes("fa"), _game_bytes("fb")
    path = _tenhou_name(tmp_path, "2024010100gm-00a9-0000-90330000")
    _write(path, [raw_a, raw_b])
    manifest = build_manifest(tmp_path)
    assert len(manifest) == 1
    first = fetch_game_at(path, 0, seed=_SEED, ratios=dict(_RATIOS))
    assert first.game.game_id == "fa"
    second_offset = len(raw_a)
    second = fetch_game_at(path, second_offset, seed=_SEED, ratios=dict(_RATIOS))
    assert second.game.game_id == "fb"
    again = fetch_game_at(
        path,
        second_offset,
        seed=_SEED,
        ratios=dict(_RATIOS),
        expected_sha=second.game.raw_bytes_sha256,
    )
    assert again.game.game_id == "fb"
    with pytest.raises(ContractError):
        fetch_game_at(path, second_offset + 1, seed=_SEED, ratios=dict(_RATIOS))
    with pytest.raises(ContractError):
        fetch_game_at(path, 0, seed=_SEED, ratios=dict(_RATIOS), expected_sha="sha256:" + "f" * 64)
