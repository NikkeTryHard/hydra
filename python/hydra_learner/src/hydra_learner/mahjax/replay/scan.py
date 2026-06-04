from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydra_learner.mahjax.replay.parity import (
    DATASET_ROOT,
    compare_replay_prefix_to_hydra_authority,
    hydra_authority_rows_for_paths,
)


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": _git_output(["git", "status", "--short"]),
        "config": {
            "dataset_root": str(args.dataset_root),
            "row_limit": args.row_limit,
            "event_limit": args.event_limit,
            "full": args.full,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "limit": args.limit,
        },
        "env": {
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
    }


def _git_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _scan_one(args: tuple[str, Path, int | None, int | None, list[dict[str, object]] | None]) -> dict[str, object]:
    replay, path, row_limit, event_limit, authority_rows = args
    result = compare_replay_prefix_to_hydra_authority(
        path, row_limit=row_limit, event_limit=event_limit, authority_rows=authority_rows
    )
    return {
        "replay": replay,
        "path": str(path),
        "matched_rows": result.matched_rows,
        "authority_rows": result.authority_rows,
        "stopped_reason": result.stopped_reason,
        "passed": result.passed,
        "first_failure": result.first_failure,
    }


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        print(json.dumps(report, separators=(",", ":")))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _read_replay_list(path: Path) -> list[str]:
    replays: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            replays.append(stripped)
    return replays


def _discover_dataset_replays(root: Path) -> list[str]:
    replays: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        replays.extend(
            str((base / filename).relative_to(root)) for filename in filenames if filename.endswith(".mjai.json.zst")
        )
    replays.sort()
    return replays


def _scan_jobs(
    jobs: list[tuple[str, Path, int | None, int | None, list[dict[str, object]] | None]], workers: int
) -> list[dict[str, object]]:
    if workers == 1:
        return [_scan_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_scan_one, jobs))


def _metric_int(result: dict[str, object], key: str) -> int:
    value = result[key]
    if not isinstance(value, int):
        raise TypeError(f"scan result {key} must be int")
    return value


def _timing_report(
    started: float, authority_trace_s: float, mahjax_scan_s: float, results: list[dict[str, object]]
) -> dict[str, object]:
    elapsed_s = time.perf_counter() - started
    rows = sum(_metric_int(result, "matched_rows") for result in results)
    replays = len(results)
    return {
        "elapsed_s": elapsed_s,
        "authority_trace_s": authority_trace_s,
        "mahjax_scan_s": mahjax_scan_s,
        "replays_per_s": 0.0 if elapsed_s <= 0.0 else replays / elapsed_s,
        "rows_per_s": 0.0 if elapsed_s <= 0.0 else rows / elapsed_s,
    }


def _build_report(
    results: list[dict[str, object]],
    metadata: dict[str, object],
    total_replays: int | None = None,
    timing: dict[str, object] | None = None,
) -> dict[str, object]:
    total = len(results) if total_replays is None else total_replays
    histogram = Counter(str(result["stopped_reason"]) for result in results)
    total_authority_rows = sum(_metric_int(result, "authority_rows") for result in results)
    matched_rows = sum(_metric_int(result, "matched_rows") for result in results)
    failures = [result for result in results if not bool(result["passed"])]
    unsupported_count = sum(1 for result in failures if "unsupported" in str(result["stopped_reason"]))
    mismatch_count = len(failures) - unsupported_count
    return {
        "passed": not failures and len(results) == total,
        "total_replays": total,
        "compared_replays": len(results),
        "pending_replays": total - len(results),
        "total_authority_rows": total_authority_rows,
        "matched_rows": matched_rows,
        "mismatch_count": mismatch_count,
        "unsupported_count": unsupported_count,
        "stopped_reason_histogram": dict(sorted(histogram.items())),
        "first_failure": failures[0] if failures else None,
        "results": results,
        "timing": timing or {},
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan Tenhou replays against Hydra authority with MahJAX state parity."
    )
    parser.add_argument("replays", nargs="*", help="Replay filenames under --dataset-root, or absolute paths")
    parser.add_argument("--replay-list", type=Path, help="Newline-separated replay filenames or absolute paths")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--row-limit", type=int, default=32)
    parser.add_argument("--event-limit", type=int, default=128)
    parser.add_argument("--full", action="store_true", help="Compare all authority rows and all replay events")
    parser.add_argument("--report", type=Path, help="Write strict JSON report to this path; stdout when omitted")
    parser.add_argument("--resume-from-report", type=Path, help="Reuse passed replay results from an earlier report")
    parser.add_argument(
        "--all-dataset", action="store_true", help="Scan every .mjai.json.zst replay under --dataset-root"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Authority trace batch size for large scans")
    parser.add_argument("--limit", type=int, help="Scan only the first N selected replays after sorting/list expansion")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel replay processes. Use one worker for GPU scans; "
            "CPU parity sweeps are faster with one worker per replay."
        ),
    )
    args = parser.parse_args()
    row_limit = None if args.full else args.row_limit
    event_limit = None if args.full else args.event_limit

    replays = list(args.replays)
    if args.replay_list is not None:
        replays.extend(_read_replay_list(args.replay_list))
    if args.all_dataset:
        replays.extend(_discover_dataset_replays(args.dataset_root))
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        replays = replays[: args.limit]
    if not replays:
        raise SystemExit("at least one replay, --replay-list entry, or --all-dataset is required")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    prior = _load_report(args.resume_from_report)
    by_replay: dict[str, dict[str, object]] = {}
    if prior is not None:
        for result in prior.get("results", []):
            if isinstance(result, dict) and result.get("passed") is True:
                replay = result.get("replay")
                if isinstance(replay, str):
                    by_replay[replay] = result

    started = time.perf_counter()
    authority_trace_s = 0.0
    mahjax_scan_s = 0.0
    metadata = _metadata(args)
    pending = [replay for replay in replays if replay not in by_replay]
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start : start + args.batch_size]
        paths = [Path(replay) if Path(replay).is_absolute() else args.dataset_root / replay for replay in chunk]
        authority_started = time.perf_counter()
        authority_by_path = hydra_authority_rows_for_paths(paths, row_limit)
        authority_trace_s += time.perf_counter() - authority_started
        jobs: list[tuple[str, Path, int | None, int | None, list[dict[str, object]] | None]] = []
        for replay, path in zip(chunk, paths, strict=True):
            jobs.append((replay, path, row_limit, event_limit, authority_by_path[path]))
        scan_started = time.perf_counter()
        scanned = _scan_jobs(jobs, args.workers)
        mahjax_scan_s += time.perf_counter() - scan_started
        for result in scanned:
            replay = result["replay"]
            if not isinstance(replay, str):
                raise TypeError("scan result replay must be str")
            by_replay[replay] = result
        partial_results = [by_replay[replay] for replay in replays if replay in by_replay]
        timing = _timing_report(started, authority_trace_s, mahjax_scan_s, partial_results)
        _write_report(args.report, _build_report(partial_results, metadata, len(replays), timing))

    results = [by_replay[replay] for replay in replays]
    timing = _timing_report(started, authority_trace_s, mahjax_scan_s, results)
    report = _build_report(results, metadata, len(replays), timing)
    _write_report(args.report, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
