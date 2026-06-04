from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra_learner.data.raw_mjai.direct import RawMjaiDirectStream, build_raw_mjai_stream_command
from hydra_learner.data.raw_mjai.pinned import (
    BuildProgress,
    PinnedPolicyBatch,
    RawMjaiBridgeStats,
    RawMjaiPinnedQueueStats,
    RawMjaiPinnedStream,
    build_progress_json,
    default_raw_mjai_pyo3_library_path,
)

RAW_MJAI_TRANSPORT_PINNED_PYO3 = "pinned_pyo3"
RAW_MJAI_TRANSPORT_STDOUT = "stdout"
RAW_MJAI_TRANSPORTS = (RAW_MJAI_TRANSPORT_PINNED_PYO3, RAW_MJAI_TRANSPORT_STDOUT)


def raw_mjai_cursor_resume_supported() -> bool:
    return True


def raw_mjai_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    data_dirs = getattr(args, "raw_mjai_data_dirs", None)
    if not data_dirs:
        return None
    return {
        "data_dirs": [str(path) for path in data_dirs],
        "prefetch_batches": args.raw_mjai_prefetch_batches,
        "queue_bound": args.raw_mjai_queue_bound,
        "worker_threads": args.raw_mjai_worker_threads,
        "max_games": args.raw_mjai_max_games,
        "max_samples": args.raw_mjai_max_samples,
        "skip_games": args.raw_mjai_skip_games,
        "train_fraction": args.raw_mjai_train_fraction,
        "augment": args.raw_mjai_augment,
        "validation_augment": args.raw_mjai_validation_augment,
        "split": args.raw_mjai_split,
        "transport": args.raw_mjai_transport,
        "pyo3_lib": None if args.raw_mjai_pyo3_lib is None else str(args.raw_mjai_pyo3_lib),
    }


def add_raw_mjai_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-mjai-data-dir", type=Path, action="append", dest="raw_mjai_data_dirs")
    parser.add_argument("--raw-mjai-prefetch-batches", type=int, default=2)
    parser.add_argument("--raw-mjai-queue-bound", type=int, default=8)
    parser.add_argument("--raw-mjai-worker-threads", type=int, default=20)
    parser.add_argument("--raw-mjai-max-games", type=int)
    parser.add_argument("--raw-mjai-max-samples", type=int)
    parser.add_argument("--raw-mjai-skip-games", type=int, default=0)
    parser.add_argument("--raw-mjai-train-fraction", type=float, default=0.9)
    parser.add_argument("--raw-mjai-augment", action="store_true")
    parser.add_argument("--raw-mjai-validation-augment", action="store_true")
    parser.add_argument("--raw-mjai-split", choices=("train", "validation"), default="train")
    parser.add_argument("--raw-mjai-transport", choices=RAW_MJAI_TRANSPORTS, default=RAW_MJAI_TRANSPORT_PINNED_PYO3)
    parser.add_argument("--raw-mjai-pyo3-lib", type=Path, help="override libhydra_raw_mjai_pyo3.so path")


def validate_raw_mjai_source_args(args: argparse.Namespace) -> None:
    has_manifest = args.manifest is not None
    has_raw = bool(args.raw_mjai_data_dirs)
    if has_manifest == has_raw:
        raise ValueError("provide exactly one of --manifest or --raw-mjai-data-dir")
    if has_raw:
        if args.raw_mjai_skip_games < 0:
            raise ValueError("--raw-mjai-skip-games must be >= 0")
        for data_dir in args.raw_mjai_data_dirs:
            if not data_dir.exists():
                raise ValueError(f"raw MJAI data dir does not exist: {data_dir}")
    if has_raw and args.raw_mjai_train_fraction <= 0.0:
        raise ValueError("--raw-mjai-train-fraction must be > 0")

    if has_raw and args.raw_mjai_transport == RAW_MJAI_TRANSPORT_PINNED_PYO3:
        library_path = args.raw_mjai_pyo3_lib or default_raw_mjai_pyo3_library_path()
        if not library_path.exists():
            raise ValueError(
                "raw MJAI pinned PyO3 transport selected but extension is missing at "
                f"{library_path}; build `pixi run cargo build -p hydra-raw-mjai-pyo3 --release --quiet`, "
                "set HYDRA_RAW_MJAI_PYO3_LIB, pass --raw-mjai-pyo3-lib, or select --raw-mjai-transport stdout"
            )
    if has_raw and (getattr(args, "w_exit", 0.0) > 0.0 or getattr(args, "w_deltaq", 0.0) > 0.0):
        raise ValueError("positive ExIt/DeltaQ weights require compact shard labels")


__all__: Sequence[str] = (
    "RAW_MJAI_TRANSPORTS",
    "RAW_MJAI_TRANSPORT_PINNED_PYO3",
    "RAW_MJAI_TRANSPORT_STDOUT",
    "BuildProgress",
    "PinnedPolicyBatch",
    "RawMjaiBridgeStats",
    "RawMjaiDirectStream",
    "RawMjaiPinnedQueueStats",
    "RawMjaiPinnedStream",
    "add_raw_mjai_args",
    "build_progress_json",
    "build_raw_mjai_stream_command",
    "default_raw_mjai_pyo3_library_path",
    "raw_mjai_config_from_args",
    "raw_mjai_cursor_resume_supported",
    "validate_raw_mjai_source_args",
)
