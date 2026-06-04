from __future__ import annotations

import json
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from queue import Queue
from typing import Self

from hydra_learner.data.raw_mjai.codec import (
    FRAME_BATCH,
    FRAME_END,
    FRAME_HEADER,
    FRAME_PROGRESS,
    _decode_header,
    _read_exact,
    decode_batch,
)
from hydra_learner.data.raw_mjai.pinned import BuildProgress
from hydra_learner.data.shard_contracts import ManifestSummary, PolicyBatch


class RawMjaiDirectStream:
    def __init__(
        self,
        *,
        data_dirs: Sequence[Path],
        batch_size: int,
        prefetch_batches: int,
        queue_bound: int,
        worker_threads: int,
        max_games: int | None,
        max_samples: int | None,
        train_fraction: float,
        augment: bool,
        split: str,
        skip_games: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("raw MJAI batch size must be > 0")
        if prefetch_batches <= 0:
            raise ValueError("raw MJAI prefetch batches must be > 0")
        if queue_bound <= 0:
            raise ValueError("raw MJAI queue bound must be > 0")
        if worker_threads <= 0:
            raise ValueError("raw MJAI worker threads must be > 0")
        self.data_dirs = tuple(data_dirs)
        if not self.data_dirs:
            raise ValueError("raw MJAI direct stream requires at least one data dir")
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.queue_bound = queue_bound
        self.worker_threads = worker_threads
        self.max_games = max_games
        self.max_samples = max_samples
        self.skip_games = skip_games
        self.train_fraction = train_fraction
        self.augment = augment
        self.split = split
        self._queue: Queue[tuple[PolicyBatch, float] | BaseException | None] = Queue(maxsize=prefetch_batches)
        self._stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._started = 0.0
        self._finished = 0.0
        self._progress = BuildProgress(manifest_path=None, complete=False, build_seconds=0.0)
        self._progress_lock = threading.Lock()

    @property
    def manifest_path(self) -> Path:
        return self.data_dirs[0]

    @property
    def manifest_summary(self) -> ManifestSummary | None:
        with self._progress_lock:
            samples = self._progress.samples
        if samples == 0:
            return None
        return ManifestSummary(
            path=self.data_dirs[0],
            train_samples=samples,
            validation_samples=0,
            shard_count=0,
            record_size=0,
            feature_flags=0,
        )

    def start(self) -> None:
        cmd = build_raw_mjai_stream_command(
            data_dirs=self.data_dirs,
            batch_size=self.batch_size,
            max_games=self.max_games,
            max_samples=self.max_samples,
            skip_games=self.skip_games,
            queue_bound=self.queue_bound,
            worker_threads=self.worker_threads,
            train_fraction=self.train_fraction,
            augment=self.augment,
            split=self.split,
        )
        print(
            "raw_mjai_direct_start "
            f"dirs={len(self.data_dirs)} batch={self.batch_size} prefetch={self.prefetch_batches} "
            f"queue_bound={self.queue_bound} workers={self.worker_threads} max_games={self.max_games} "
            f"max_samples={self.max_samples} skip_games={self.skip_games} split={self.split}",
            file=sys.stderr,
            flush=True,
        )
        self._started = time.perf_counter()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader_thread = threading.Thread(
            target=self._run_reader, name="hydra-raw-mjai-direct-reader", daemon=True
        )
        self._reader_thread.start()
        print(
            f"raw_mjai_direct_started pid={self._process.pid if self._process else None}", file=sys.stderr, flush=True
        )

    def next_batch(self) -> tuple[PolicyBatch, float]:
        started = time.perf_counter()
        item = self._queue.get()
        if item is None:
            raise StopIteration("raw MJAI stream exhausted")
        if isinstance(item, BaseException):
            raise item
        wait_ms = (time.perf_counter() - started) * 1000.0
        print(
            f"raw_mjai_direct_next wait_ms={wait_ms:.3f} rows={item[0].actions.shape[0]} fetch_ms={item[1]:.3f}",
            file=sys.stderr,
            flush=True,
        )
        return item

    def progress(self) -> BuildProgress:
        with self._progress_lock:
            progress = self._progress
        finished = self._finished if self._finished != 0.0 else time.perf_counter()
        return BuildProgress(
            manifest_path=None,
            complete=progress.complete,
            build_seconds=finished - self._started,
            loaded_games=progress.loaded_games,
            skipped_games=progress.skipped_games,
            samples=progress.samples,
            batches=progress.batches,
            max_games_reached=progress.max_games_reached,
            max_samples_reached=progress.max_samples_reached,
        )

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def _run_reader(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._queue.put(RuntimeError("raw MJAI stream process was not started"))
            self._queue.put(None)
            return
        try:
            self._read_frames(process)
        except BaseException as exc:
            self._queue.put(exc)
        finally:
            self._finished = time.perf_counter()
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
            status = process.wait()
            print(
                f"raw_mjai_direct_exit status={status} stderr_bytes={len(stderr)}",
                file=sys.stderr,
                flush=True,
            )
            if status != 0:
                msg = stderr.decode("utf-8", errors="replace").strip()
                self._queue.put(RuntimeError(f"raw MJAI stream failed with status {status}: {msg}"))
            self._queue.put(None)

    def _read_frames(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        saw_header = False
        while not self._stop.is_set():
            prefix = process.stdout.read(9)
            if prefix == b"":
                break
            if len(prefix) != 9:
                raise ValueError("truncated raw MJAI frame header")
            kind = prefix[0]
            payload_len = struct.unpack_from("<Q", prefix, 1)[0]
            payload = _read_exact(process.stdout, payload_len)
            if kind == FRAME_HEADER:
                _decode_header(payload, self.batch_size)
                saw_header = True
            elif kind == FRAME_BATCH:
                if not saw_header:
                    raise ValueError("raw MJAI batch arrived before stream header")
                started = time.perf_counter()
                batch = decode_batch(payload)
                self._queue.put((batch, (time.perf_counter() - started) * 1000.0))
            elif kind == FRAME_PROGRESS:
                self._set_progress(payload, complete=False)
            elif kind == FRAME_END:
                self._set_progress(payload, complete=True)
                break
            else:
                raise ValueError(f"unknown raw MJAI frame kind {kind}")

    def _set_progress(self, payload: bytes | bytearray, *, complete: bool) -> None:
        data = json.loads(payload.decode("utf-8"))
        with self._progress_lock:
            self._progress = BuildProgress(
                manifest_path=None,
                complete=complete,
                build_seconds=0.0,
                loaded_games=int(data.get("loaded_games", 0)),
                skipped_games=int(data.get("skipped_games", 0)),
                samples=int(data.get("samples", 0)),
                batches=int(data.get("batches", 0)),
                max_games_reached=bool(data.get("max_games_reached", False)),
                max_samples_reached=bool(data.get("max_samples_reached", False)),
            )


def build_raw_mjai_stream_command(
    *,
    data_dirs: Sequence[Path],
    batch_size: int,
    max_games: int | None,
    max_samples: int | None,
    queue_bound: int,
    worker_threads: int,
    train_fraction: float,
    augment: bool,
    split: str = "train",
    skip_games: int = 0,
) -> list[str]:
    cmd = [
        "pixi",
        "run",
        "-e",
        "default",
        "cargo",
        "run",
        "--quiet",
        "--package",
        "hydra-train",
        "--features",
        "training",
        "--bin",
        "raw_mjai_stream",
        "--",
    ]
    for data_dir in data_dirs:
        cmd.extend(["--input", str(data_dir)])
    cmd.extend(
        [
            "--batch-size",
            str(batch_size),
            "--queue-bound",
            str(queue_bound),
            "--num-threads",
            str(worker_threads),
            "--train-fraction",
            str(train_fraction),
            "--split",
            split,
        ]
    )
    if max_games is not None:
        cmd.extend(["--max-games", str(max_games)])
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if skip_games:
        cmd.extend(["--skip-games", str(skip_games)])
    if augment:
        cmd.append("--augment")
    return cmd
