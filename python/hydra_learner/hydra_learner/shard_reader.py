from __future__ import annotations

import mmap
import time
from pathlib import Path
from typing import BinaryIO, Self

import numpy as np

from hydra_learner.shard_contracts import (
    ACTION_SPACE,
    FLAG_DELTA_Q,
    FLAG_EXIT,
    FLAG_SAFETY_RESIDUAL,
    GRP_CLASS_COUNT,
    NUM_CHANNELS,
    OPPONENT_COUNT,
    PLAYER_COUNT,
    SCORE_BINS,
    SPATIAL_TARGET_SIZE,
    TILE_WIDTH,
    PolicyBatch,
    _ShardMeta,
)
from hydra_learner.shard_decode import _decode_rows
from hydra_learner.shard_manifest import _load_split_shards, _verify_mapped_shard


class _MappedShard:
    def __init__(self, path: Path, meta: _ShardMeta) -> None:
        self.path = path
        self.meta = meta
        self._file: BinaryIO = path.open("rb")
        self.mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        _verify_mapped_shard(self.mmap, meta, path)

    def close(self) -> None:
        self.mmap.close()
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


class BcShardReader:
    """Mmap-backed compact shard reader for policy BC fields."""

    def __init__(self, manifest_path: Path, split: str = "train") -> None:
        summary, metas = _load_split_shards(manifest_path, split, check_files=False)
        if not metas:
            raise ValueError(f"BC shard manifest has no {split!r} shards")
        self.manifest_path = manifest_path
        self.summary = summary
        self.split = split
        base_dir = manifest_path.parent
        self._shards = [_MappedShard(base_dir / meta.file_name, meta) for meta in metas]
        self._starts = np.array([shard.meta.first_sample_index for shard in self._shards], dtype=np.int64)

    @property
    def sample_count(self) -> int:
        return sum(shard.meta.sample_count for shard in self._shards)

    @property
    def feature_flags(self) -> int:
        return self._shards[0].meta.feature_flags

    @property
    def record_size(self) -> int:
        return self._shards[0].meta.record_size

    def close(self) -> None:
        for shard in self._shards:
            shard.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def batch_range(self, start: int, batch_size: int) -> PolicyBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        end = start + batch_size
        if start < 0 or end > self.sample_count:
            raise ValueError(f"BC shard batch range {start}..{end} exceeds sample count {self.sample_count}")
        obs = np.zeros((batch_size, NUM_CHANNELS, TILE_WIDTH), dtype=np.float32)
        actions = np.empty((batch_size,), dtype=np.int64)
        legal = np.empty((batch_size, ACTION_SPACE), dtype=np.bool_)
        value_target = np.empty((batch_size,), dtype=np.float32)
        grp_target = np.zeros((batch_size, GRP_CLASS_COUNT), dtype=np.float32)
        oracle_target = np.empty((batch_size, PLAYER_COUNT), dtype=np.float32)
        oracle_target_mask = np.empty((batch_size,), dtype=np.float32)
        tenpai = np.empty((batch_size, OPPONENT_COUNT), dtype=np.float32)
        opp_next = np.zeros((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        danger = np.empty((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        danger_mask = np.empty((batch_size, SPATIAL_TARGET_SIZE), dtype=np.float32)
        score_pdf = np.zeros((batch_size, SCORE_BINS), dtype=np.float32)
        score_cdf = np.zeros((batch_size, SCORE_BINS), dtype=np.float32)
        safety_target = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32)
            if self.feature_flags & FLAG_SAFETY_RESIDUAL
            else None
        )
        safety_mask = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32)
            if self.feature_flags & FLAG_SAFETY_RESIDUAL
            else None
        )
        exit_target = np.zeros((batch_size, ACTION_SPACE), dtype=np.float32) if self.feature_flags & FLAG_EXIT else None
        exit_mask = np.zeros((batch_size, ACTION_SPACE), dtype=np.float32) if self.feature_flags & FLAG_EXIT else None
        deltaq_target = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32) if self.feature_flags & FLAG_DELTA_Q else None
        )
        deltaq_mask = (
            np.zeros((batch_size, ACTION_SPACE), dtype=np.float32) if self.feature_flags & FLAG_DELTA_Q else None
        )
        row = 0
        remaining = batch_size
        sample = start
        while remaining > 0:
            shard_index = int(np.searchsorted(self._starts, sample, side="right") - 1)
            if shard_index < 0:
                raise ValueError(f"BC shard sample index {sample} out of bounds")
            shard = self._shards[shard_index]
            local = sample - shard.meta.first_sample_index
            take = min(remaining, shard.meta.sample_count - local)
            _decode_rows(
                shard.mmap,
                shard.meta.record_size,
                shard.meta.feature_flags,
                local,
                take,
                obs,
                actions,
                legal,
                value_target,
                grp_target,
                oracle_target,
                oracle_target_mask,
                tenpai,
                opp_next,
                danger,
                danger_mask,
                score_pdf,
                score_cdf,
                safety_target,
                safety_mask,
                exit_target,
                exit_mask,
                deltaq_target,
                deltaq_mask,
                row,
            )
            row += take
            sample += take
            remaining -= take
        return PolicyBatch(
            obs=obs,
            actions=actions,
            legal_mask=legal,
            value_target=value_target,
            grp_target=grp_target,
            oracle_target=oracle_target,
            oracle_target_mask=oracle_target_mask,
            tenpai=tenpai,
            opp_next=opp_next,
            danger=danger,
            danger_mask=danger_mask,
            score_pdf=score_pdf,
            score_cdf=score_cdf,
            safety_target=safety_target,
            safety_mask=safety_mask,
            exit_target=exit_target,
            exit_mask=exit_mask,
            deltaq_target=deltaq_target,
            deltaq_mask=deltaq_mask,
        )


class BcShardDataset:
    """Sequential real-shard batch source for policy-only BC."""

    def __init__(self, manifest_path: Path, batch_size: int, split: str = "train") -> None:
        self.reader = BcShardReader(manifest_path, split=split)
        self.batch_size = batch_size
        self._cursor = 0
        self.last_fetch_decode_ms = 0.0

    @property
    def sample_count(self) -> int:
        return self.reader.sample_count

    def next_batch(self) -> PolicyBatch:
        if self.sample_count < self.batch_size:
            raise ValueError(f"BC shard split has {self.sample_count} samples, needs batch {self.batch_size}")
        if self._cursor + self.batch_size > self.sample_count:
            self._cursor = 0
        started = time.perf_counter()
        batch = self.reader.batch_range(self._cursor, self.batch_size)
        self.last_fetch_decode_ms = (time.perf_counter() - started) * 1000.0
        self._cursor += self.batch_size
        return batch

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()
