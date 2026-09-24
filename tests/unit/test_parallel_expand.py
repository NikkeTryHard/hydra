"""WP-14 parallel game expansion: cache-free ordered pool, serial bit-parity.

Covers the cache-free parallel expansion path in
:mod:`hydra2.training.stream_train` only (no cache, no replay/encode
changes): serial-vs-parallel parity (row bytes + quarantine classes + order)
on a mixed fixture set carrying quarantine-bearing games, fail-closed worker
accounting, pool bounds, worker thread clamping, once-per-run pool identity
(epoch-2+ respawn cost zero), scan bit-identity, and the serial terminal
messages verbatim under parallel fill. CPU lane, fixed seeds; spawn pools are
closed explicitly via try/finally (never left to interpreter exit).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import zstandard as zstd

from hydra2.contracts.common import ContractError
from hydra2.data.stream_decode import PrefetchGameStream
from hydra2.data.stream_iter import GameStream
from hydra2.data.stream_manifest import build_manifest
from hydra2.data.stream_read import assign_split, group_key_for_entry
from hydra2.training import stream_train as driver
from hydra2.training.stream_train import SPLIT_RATIOS

pytestmark = [pytest.mark.contract_package("WP-14"), pytest.mark.slow]

DATA_SEED = 7
WALL = list(range(136))

_RATIOS = {"train": SPLIT_RATIOS["train"], "validation": SPLIT_RATIOS["validation"]}


_TEHAIS = [
    ["1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p"],
    ["2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p"],
    ["3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p"],
    ["4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p"],
]


def _good_events(game_id: str, *, wall: list[int] | None) -> list[dict[str, object]]:
    start: dict[str, object] = {"type": "start_game", "game_id": game_id}
    if wall is not None:
        start["wall"] = list(wall)
    return [
        start,
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
        {"type": "end_game", "game_id": game_id, "scores": [30000, 25000, 20000, 15000]},
    ]


def _quarantine_events(game_id: str, *, kind: str) -> list[dict[str, object]]:
    """Decodable + stream-valid but expansion-rejected (wall-less sim path)."""
    events = _good_events(game_id, wall=None)
    if kind == "bogus-mid":
        events.insert(-1, {"type": "bogus_mid_event"})
    elif kind == "illegal-discard":
        # Seat 1 discards a tile it never held nor drew: replay desync.
        events[5] = {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": False}
    else:  # pragma: no cover - fixture bug, not runtime behavior
        raise AssertionError(f"unknown quarantine kind {kind!r}")
    return events


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    raw = "\n".join(json.dumps(event) for event in events) + "\n"
    path.write_bytes(zstd.ZstdCompressor().compress(raw.encode()))


def _split_of(stem: str) -> str:
    probe = Path("corpus") / "tenhou" / f"{stem}.mjai.json.zst"
    return assign_split(
        group_key=group_key_for_entry(root_id="tenhou", path=probe),
        seed=DATA_SEED,
        ratios=_RATIOS,
    )


def _pick_train_stems(need_train: int) -> list[str]:
    """Deterministic filename stems assigned to the train split."""
    train: list[str] = []
    for day in range(1, 32):
        stem = f"202402{day:02d}12gm-00a9-0000-{day:07d}"
        if _split_of(stem) == "train" and len(train) < need_train:
            train.append(stem)
        if len(train) >= need_train:
            return train
    raise AssertionError(f"no train split coverage: {train}")


def _write_parity_corpus(corpus: Path) -> None:
    """Mixed 12-game corpus: sim + walled successes, quarantine-bearing games."""
    corpus.mkdir(parents=True, exist_ok=True)
    stems = _pick_train_stems(12)
    kinds = ["bogus-mid", "illegal-discard"]
    bad = 0
    for index, stem in enumerate(stems):
        game_id = f"parity-game-{index}"
        if index % 4 == 3:
            kind = kinds[bad % len(kinds)]
            bad += 1
            _write_events(corpus / f"{stem}.mjai.json.zst", _quarantine_events(game_id, kind=kind))
        elif index % 4 == 1:
            _write_events(corpus / f"{stem}.mjai.json.zst", _good_events(game_id, wall=WALL))
        else:
            _write_events(corpus / f"{stem}.mjai.json.zst", _good_events(game_id, wall=None))


def _drain(dataset: Any, *, batch_size: int = 2) -> None:
    """Consume to corpus exhaustion; the epoch must roll (post-roll flow is asserted elsewhere)."""
    start_epoch = dataset.epoch
    for _ in range(10**6):
        dataset._consume_microbatch(batch_size)
        if dataset.epoch != start_epoch:
            break
    else:
        raise AssertionError("epoch never rolled past exhaustion")


def _snapshot_state(dataset: Any) -> dict[str, Any]:
    return {
        "rows": [repr(row) for row in dataset._rows],
        "privileged": sorted((key, repr(value)) for key, value in dataset.privileged.items()),
        "replayed": dataset.replayed,
        "sim_replayed": dataset.sim_replayed,
        "expand_quarantined": dataset.expand_quarantined,
        "reasons": list(dataset.expand_quarantine_reasons.items()),
        "entries": [dict(entry) for entry in dataset._buffered_entries],
        "row_hash": dataset.buffered_row_hash(),
    }


@pytest.mark.serial
class TestSerialParallelParity:
    def test_rows_quarantine_and_order_identical(self, tmp_path: Path) -> None:
        """Serial vs parallel agree on rows bytes, quarantine classes, and order."""
        corpus = tmp_path / "corpus" / "tenhou"
        _write_parity_corpus(corpus)
        manifest = build_manifest(corpus)

        serial = driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=True,
            replay_backend="rust",
        )
        # Production shape: prefetch decode plus the bounded expansion pool.
        parallel = driver._StreamDataset(
            stream_factory=lambda epoch=0: PrefetchGameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
                prefetch=8,
                max_workers=2,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=True,
            replay_backend="rust",
            expand_workers=4,
        )
        try:
            _drain(serial)
            _drain(parallel)
            # The parallel arm really expanded in the pool; serial never built one.
            assert parallel._expand_pool is not None
            assert parallel._expand_workers == 4
            assert serial._expand_pool is None
            serial_state = _snapshot_state(serial)
            parallel_state = _snapshot_state(parallel)
        finally:
            serial.close()
            parallel.close()
        # Quarantine-bearing fixtures must actually fire (else the test is vacuous).
        # Lower bound, not exact: the flipping take already expands epoch-1 games.
        assert serial_state["expand_quarantined"] >= 3
        assert len(serial_state["reasons"]) == 2
        assert serial_state["replayed"] > 0 and serial_state["sim_replayed"] > 0
        assert len(serial_state["rows"]) > 0
        # Bit-identity: rows bytes, privileged joins, counters, reason classes
        # in first-seen order, buffer index, and the row hash all match.
        assert parallel_state == serial_state

    def test_rows_quarantine_and_order_identical_rust(self, tmp_path: Path) -> None:
        """Serial vs pool expansion agree on the rust backend (spawn-safe)."""
        corpus = tmp_path / "corpus" / "tenhou"
        _write_parity_corpus(corpus)
        manifest = build_manifest(corpus)

        serial = driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=True,
            replay_backend="rust",
        )
        parallel = driver._StreamDataset(
            stream_factory=lambda epoch=0: PrefetchGameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
                prefetch=8,
                max_workers=2,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=True,
            replay_backend="rust",
            expand_workers=4,
        )
        try:
            _drain(serial)
            _drain(parallel)
            assert parallel._expand_pool is not None
            assert parallel._expand_workers == 4
            assert serial._expand_pool is None
            serial_state = _snapshot_state(serial)
            parallel_state = _snapshot_state(parallel)
        finally:
            serial.close()
            parallel.close()
        assert serial_state["expand_quarantined"] >= 3
        assert len(serial_state["reasons"]) == 2
        assert serial_state["replayed"] > 0 and serial_state["sim_replayed"] > 0
        assert len(serial_state["rows"]) > 0
        assert parallel_state == serial_state


@pytest.mark.serial
class TestParallelGuards:
    def test_expand_workers_validated(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(corpus)
        for bad in (True, -1, "4", 2.0, None):
            with pytest.raises(ContractError, match="expand_workers"):
                driver._StreamDataset(
                    stream_factory=lambda epoch=0: GameStream(
                        manifest,
                        seed=DATA_SEED,
                        ratios=dict(_RATIOS),
                        epoch=epoch,
                        split="train",
                        shuffle_buffer=0,
                    ),
                    num_actions=6792,
                    feature_dim=64,
                    seed=DATA_SEED,
                    drop_last=True,
                    replay_backend="rust",
                    expand_workers=bad,  # type: ignore[arg-type]
                )

    def test_expand_workers_capped_at_sixteen(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(corpus)
        dataset = driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            replay_backend="rust",
            expand_workers=99,
        )
        try:
            assert dataset._expand_workers == driver._PARALLEL_EXPAND_MAX_WORKERS == 16
        finally:
            dataset.close()

    def test_parallel_empty_split_message_matches_serial(self, tmp_path: Path) -> None:
        """Parallel fill on an empty split raises the serial terminal verbatim."""
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(corpus)
        dataset = driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            replay_backend="rust",
            expand_workers=2,
        )
        try:
            with pytest.raises(ContractError, match="stream yielded zero train rows"):
                dataset._consume_microbatch(2)
        finally:
            dataset.close()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus" / "tenhou"
        corpus.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(corpus)
        dataset = driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            replay_backend="rust",
            expand_workers=2,
        )
        dataset.close()
        dataset.close()


@pytest.mark.serial
class TestPoolHygiene:
    def _parallel_dataset(self, manifest: Any) -> Any:
        return driver._StreamDataset(
            stream_factory=lambda epoch=0: GameStream(
                manifest,
                seed=DATA_SEED,
                ratios=dict(_RATIOS),
                epoch=epoch,
                split="train",
                shuffle_buffer=0,
            ),
            num_actions=6792,
            feature_dim=64,
            seed=DATA_SEED,
            drop_last=True,
            need_privileged=True,
            replay_backend="rust",
            expand_workers=2,
        )

    def test_expand_workers_run_single_threaded(self, tmp_path: Path) -> None:
        """Pool initializer clamps workers to one torch/OMP thread (no NxC thrash)."""
        corpus = tmp_path / "corpus" / "tenhou"
        _write_parity_corpus(corpus)
        manifest = build_manifest(corpus)
        dataset = self._parallel_dataset(manifest)
        try:
            dataset._consume_microbatch(2)
            pool = dataset._expand_pool
            assert pool is not None
            thread_counts = {
                future.result() for future in [pool.submit(torch.get_num_threads) for _ in range(4)]
            }
            omp_values = {
                future.result()
                for future in [pool.submit(os.getenv, "OMP_NUM_THREADS") for _ in range(2)]
            }
            assert thread_counts == {1}
            assert omp_values == {"1"}
        finally:
            dataset.close()

    def test_expand_pool_built_once_no_respawn(self, tmp_path: Path) -> None:
        """Epoch-2+ spawn cost is zero: one pool object across all fills.

        The pool-identity half (``dataset._expand_pool is pool``) is the
        defect signal: a respawn allocates a new pool. Worker PIDs are NOT
        asserted — spawn races recycle PIDs between the two samplings
        (~1/10 flake), which proves nothing about the pool itself.
        """
        corpus = tmp_path / "corpus" / "tenhou"
        _write_parity_corpus(corpus)
        manifest = build_manifest(corpus)
        dataset = self._parallel_dataset(manifest)
        try:
            dataset._consume_microbatch(2)
            pool = dataset._expand_pool
            assert pool is not None
            _drain(dataset)
            assert dataset._expand_pool is pool
        finally:
            dataset.close()

    def test_scan_parallel_matches_serial(self, tmp_path: Path) -> None:
        """Scan pool (initializer wired) stays bit-identical to the serial scan."""
        corpus = tmp_path / "corpus" / "tenhou"
        _write_parity_corpus(corpus)
        manifest = build_manifest(corpus)
        serial_cfg = SimpleNamespace(
            seeds=SimpleNamespace(data_seed=DATA_SEED),
            data=SimpleNamespace(num_workers=0, train_split="train", val_split="validation"),
        )
        parallel_cfg = SimpleNamespace(
            seeds=SimpleNamespace(data_seed=DATA_SEED),
            data=SimpleNamespace(num_workers=2, train_split="train", val_split="validation"),
        )
        serial = driver._scan_corpus(manifest, config=serial_cfg, ratios=dict(_RATIOS))
        parallel = driver._scan_corpus(manifest, config=parallel_cfg, ratios=dict(_RATIOS))
        assert parallel == serial
