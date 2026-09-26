"""Sidecar ClearML reporter — per-update curves without touching training.

Reads local authoritative JSONL (logs/metrics.jsonl, eval/eval.jsonl,
logs/feed-telemetry.jsonl) and reports every row via the ClearML SDK.
Training files never import clearml; this script is the sole SDK owner.
Mirrors fire only at checkpoint cadence, so this sidecar is the
only source of per-update loss/accuracy curves.

Modes: one-shot (default, hermetic offline) and live (--online --follow
for server runs). Offset tracking in logs/.sidecar-offset makes every
invocation report only unseen bytes, so a final sweep after --follow
is a cheap no-op for old rows. --follow exits when the wrapper drops
logs/.sidecar-done (train process ended) and a full pass finds no new
rows, or at --follow-timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


def _install_termination_flush(logger: Any, task: Any, *, argv0: str = "sidecar") -> None:
    """Flush + close the task on SIGTERM/SIGINT instead of dying buffered.

    The SDK batches scalar reports; a killed follower used to take up to a
    full pass of buffered rows to the grave while the offset file already
    called them consumed (offsets advance per pass, uploads lag behind).
    The handler flushes, closes (final server push), then re-raises the
    signal's default disposition so the exit code still signals termination.
    Install once in follow mode; one-shot exits via ``task.close()`` below.
    """

    def _handle(signum: int, _frame: Any) -> None:
        with contextlib.suppress(Exception):
            logger.flush()
        with contextlib.suppress(Exception):
            task.close()
        print(f"{argv0}: terminated by signal {signum} after flush", flush=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


OFFSET_NAME = ".sidecar-offset"
DONE_NAME = ".sidecar-done"
LOCK_NAME = ".sidecar-lock"

METRICS_FILE = Path("logs") / "metrics.jsonl"
EVAL_FILE = Path("eval") / "eval.jsonl"
FEED_FILE = Path("logs") / "feed-telemetry.jsonl"
VERBOSE_FILE = Path("logs") / "verbose-telemetry.jsonl"
SPANS_FILE = Path("logs") / "stage-spans.jsonl"
RUN_YAML = Path("run.yaml")

# Effective training rows per optimizer update (microbatch 2048 x accum 4).
# Used ONLY for verbose-derived throughput (global_update progress over real
# sample wall time). Metrics/feed t_wall_s are checkpoint-flush stamps
# (microseconds apart for 1000-row batches) and MUST NOT feed throughput.
ROWS_PER_UPDATE = 8192.0


def _iter_jsonl(path: Path) -> Any:
    """Yield parsed objects, skipping blank and malformed lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        if line.strip() == "":
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out
    return None


def _read_new_rows(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Return (parsed rows, new offset) for bytes after ``offset``.

    Truncated files (size < offset) rewind to zero: a resumed run that
    rewrites history must not strand the reader past EOF.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:
        offset = 0
    try:
        with open(path, encoding="utf-8") as f:
            f.seek(offset)
            text = f.read()
    except OSError:
        return [], offset
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip() == "":
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, size


def _report_metrics_row(logger: Any, row: dict[str, Any]) -> None:
    try:
        iteration = int(row.get("global_update"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    for series in ("total", "policy", "placement", "value", "event", "belief"):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(title="loss", series=series, value=value, iteration=iteration)
    for series in ("top1", "top3", "top5"):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(title="accuracy", series=series, value=value, iteration=iteration)
    for series in ("masked_nll", "legal_uniform_nll", "legal_uniform_gap"):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(title="nll", series=series, value=value, iteration=iteration)
    for key, value_raw in row.items():
        if not isinstance(key, str):
            continue
        if key.startswith("event_") or key.startswith("belief_"):
            value = _as_float(value_raw)
            if value is not None:
                logger.report_scalar(title="heads", series=key, value=value, iteration=iteration)
    # Optimizer health (same x): skip counters exist in every entry; LR and
    # grad-norm appear only on successor runs (Step 4) — absent keys no-op.
    for series in (
        "skipped_updates",
        "skipped_this_update",
        "grad_norm_pre",
        "grad_norm_post",
        "lr_now",
    ):
        value = _as_float(row.get(series))
        if value is not None and value == value and abs(value) != float("inf"):
            logger.report_scalar(title="opt", series=series, value=value, iteration=iteration)
    # Overfitting U-turn overlay: same train values under a shared title so a
    # future eval NLL series shares one plot without moving existing loss/*.
    for series in ("total", "masked_nll"):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(
                title="loss-vs-eval", series=f"train_{series}", value=value, iteration=iteration
            )


def _report_eval_row(logger: Any, row: dict[str, Any]) -> None:
    try:
        iteration = int(row.get("update", -1))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if iteration < 0:
        return
    for key, value_raw in row.items():
        if key == "update":
            continue
        value = _as_float(value_raw)
        if value is not None:
            logger.report_scalar(title="eval", series=str(key), value=value, iteration=iteration)
    # U-turn overlay: eval NLL family shares loss-vs-eval with train.
    for series in (
        "masked_nll",
        "discard_nll",
        "top1",
        "top3",
        "top5",
        "calibration_ece",
        "temperature",
        "calibrated_nll",
        "calibrated_ece",
    ):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(
                title="loss-vs-eval", series=f"eval_{series}", value=value, iteration=iteration
            )
    # Support shape under its own title (keeps eval/* small).
    for series in (
        "support_min",
        "support_max",
        "strata",
        "num_eval_batches",
        "num_eval_rows",
        "masked_nll_se",
        "top1_se",
    ):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(
                title="eval-support", series=series, value=value, iteration=iteration
            )
    # CONSOLE marker (never SCALARS): one line per eval row.
    with contextlib.suppress(TypeError, ValueError):
        logger.report_text(
            f"eval update={iteration} "
            f"masked_nll={float(row.get('masked_nll', float('nan'))):.4f} "
            f"discard_nll={float(row.get('discard_nll', float('nan'))):.4f} "
            f"top1={float(row.get('top1', float('nan'))):.4f} "
            f"ece={float(row.get('calibration_ece', float('nan'))):.4f}"
        )


def _report_feed_row(logger: Any, row: dict[str, Any], fallback: int) -> None:
    kind = row.get("kind")
    if kind == "summary":
        return
    # Optimizer-update summary rows (UpdateTelemetry.to_dict) never hit JSONL
    # mid-run today; forward under opt when present, ignore absence.
    if kind == "update":
        try:
            iteration = int(row.get("global_update", fallback))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            iteration = fallback
        for series in ("optimizer_ms", "logging_ms"):
            value = _as_float(row.get(series))
            if value is not None:
                logger.report_scalar(title="opt", series=series, value=value, iteration=iteration)
        return
    try:
        iteration = int(row.get("global_update", fallback))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        iteration = fallback
    for series in (
        "fetch_decode_ms",
        "compute_ms",
        "queue_wait_ms",
        "h2d_ms",
        "forward_ms",
        "loss_ms",
        "backward_ms",
        "producer_wait_s",
    ):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(title="feed", series=series, value=value, iteration=iteration)
    # GC attribution (monotone counters; deltas computed offline in UI).
    for series in ("gc_collections_gen0", "gc_collections_gen1", "gc_collections_gen2"):
        value = _as_float(row.get(series))
        if value is not None:
            logger.report_scalar(title="feed-gc", series=series, value=value, iteration=iteration)


def _report_verbose_row(
    logger: Any, row: dict[str, Any], fallback: int, prev: dict[str, float]
) -> None:
    """GPU/torch/host health + true throughput from real sample clocks.

    Verbose rows carry real 50ms sample wall time (t_wall_s) plus the
    global_update join key, unlike metrics/feed t_wall_s which are
    checkpoint-flush stamps. Throughput = d(global_update)/d(t_wall) *
    ROWS_PER_UPDATE; skipped on non-positive dt/du or first row.
    """
    try:
        iteration = int(row.get("global_update", fallback))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        iteration = fallback
    gpu = row.get("gpu") if isinstance(row.get("gpu"), dict) else {}
    torch_sec = row.get("torch") if isinstance(row.get("torch"), dict) else {}
    cpu = row.get("cpu") if isinstance(row.get("cpu"), dict) else {}
    for series in ("util_gpu_pct", "mem_used_mb", "power_w", "temp_c", "clock_sm_mhz"):
        value = _as_float(gpu.get(series))
        if value is not None:
            logger.report_scalar(title="gpu", series=series, value=value, iteration=iteration)
    # Latest GPU state for the follow-mode heartbeat (observer-only copy).
    util = _as_float(gpu.get("util_gpu_pct"))
    mem = _as_float(gpu.get("mem_used_mb"))
    if util is not None:
        prev["gpu_util"] = util
    if mem is not None:
        prev["gpu_mem_mb"] = mem
    for series in ("allocated_mb", "reserved_mb", "ooms", "alloc_retries"):
        value = _as_float(torch_sec.get(series))
        if value is not None:
            logger.report_scalar(title="torch-mem", series=series, value=value, iteration=iteration)
    rss = _as_float(cpu.get("rss_mb"))
    if rss is not None:
        logger.report_scalar(title="host", series="rss_mb", value=rss, iteration=iteration)
    t_wall = _as_float(row.get("t_wall_s"))
    prev_update = prev.get("update")
    prev_wall = prev.get("wall")
    if t_wall is None:
        return
    if prev_update is None or prev_wall is None:
        prev["update"] = float(iteration)
        prev["wall"] = float(t_wall)
        return
    # True update interval: only advance the clock when global_update changes.
    # Samples tick every 50ms but updates take ~300ms; using every sample's
    # dt would divide by the 50ms sample gap and overestimate 3-6x.
    if float(iteration) == float(prev_update):
        return
    dt = t_wall - prev_wall
    du = float(iteration) - float(prev_update)
    if dt > 0 and du > 0:
        rows_per_s = du / dt * ROWS_PER_UPDATE
        if rows_per_s < 1_000_000:
            logger.report_scalar(
                title="throughput", series="rows_per_s", value=rows_per_s, iteration=iteration
            )
    prev["update"] = float(iteration)
    prev["wall"] = float(t_wall)


def _report_span_row(logger: Any, row: dict[str, Any], fallback: int) -> None:
    """Sparse save/eval stage durations (nil cost, high stall signal)."""
    try:
        iteration = int(row.get("update", fallback))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        iteration = fallback
    stage = row.get("stage")
    dur = _as_float(row.get("dur_ms"))
    if not isinstance(stage, str) or dur is None or stage == "":
        return
    logger.report_scalar(title="spans", series=stage, value=dur, iteration=iteration)


def _report_eval_tables(logger: Any, row: dict[str, Any]) -> None:
    """Per-type + calibration tables under PLOTS (no-op when keys absent)."""
    try:
        update = int(row.get("update", -1))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if update < 0:
        return
    header = ["kind", "n", "nll", "top1", "top3", "ece", "recall", "low_support"]
    table: list[list[Any]] = [header]
    by_kind: dict[str, dict[str, Any]] = {}
    for key, value_raw in row.items():
        if not isinstance(key, str) or not key.startswith("per_type/"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            continue
        _, kind, metric = parts
        by_kind.setdefault(kind, {})[metric] = value_raw
    if by_kind:
        for kind in sorted(by_kind):
            m = by_kind[kind]
            table.append(
                [
                    kind,
                    m.get("n"),
                    m.get("nll"),
                    m.get("top1"),
                    m.get("top3"),
                    m.get("ece"),
                    m.get("recall"),
                    m.get("low_support"),
                ]
            )
        with contextlib.suppress(Exception):
            logger.report_table(title="eval-per-type", series=f"update-{update}", table_plot=table)
    cal_keys = (
        "calibration_ece",
        "temperature",
        "calibrated_nll",
        "calibrated_ece",
        "legal_uniform_nll",
        "legal_uniform_gap",
    )
    if any(k in row for k in cal_keys):
        cal_table: list[list[Any]] = [["metric", "value"]]
        cal_table.extend([k, row[k]] for k in cal_keys if k in row)
        with contextlib.suppress(Exception):
            logger.report_table(
                title="eval-calibration", series=f"update-{update}", table_plot=cal_table
            )


SCAN_CACHE_FILE = Path("cache") / "scan-cache.json"


def _report_source_mix_once(task: Any, logger: Any, run_dir: Path) -> None:
    """Per-root train/val game tallies from the scan report (once, observer-only).

    Reads ``cache/scan-cache.json`` (written before training starts; the
    ``root_games`` tallies come from Step-2's scan, never the hot path).
    Reports one lean ``task.connect()`` dict plus one CONSOLE text line per
    root — never SCALARS (per-root series would explode the scalar index).
    File counts are deliberately absent: no run_dir artifact records them,
    and re-walking the corpus from the sidecar would break the hermetic
    offline contract; game rows are the load-bearing mix signal. Missing or
    malformed cache skips silently (the scan phase owns fail-closed).
    """
    try:
        raw = json.loads((run_dir / SCAN_CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    scan = raw.get("scan")
    if not isinstance(scan, dict):
        return
    cached = scan.get("root_games")
    if not isinstance(cached, list) or len(cached) == 0:
        return
    tallies: list[tuple[str, int, int]] = []
    for row in cached:
        if not isinstance(row, list) or len(row) != 3:
            return
        rid, train_games, val_games = row
        if not isinstance(rid, str) or rid == "":
            return
        if (
            isinstance(train_games, bool)
            or not isinstance(train_games, int)
            or train_games < 0
            or isinstance(val_games, bool)
            or not isinstance(val_games, int)
            or val_games < 0
        ):
            return
        tallies.append((rid, train_games, val_games))
    lean: dict[str, int] = {"source_roots": len(tallies)}
    for rid, train_games, val_games in tallies:
        lean[f"source_{rid}_train_games"] = train_games
        lean[f"source_{rid}_val_games"] = val_games
    with contextlib.suppress(Exception):
        task.connect(lean)
    for rid, train_games, val_games in tallies:
        with contextlib.suppress(Exception):
            logger.report_text(
                f"root {rid} train_games={train_games} val_games={val_games} "
                f"rows={train_games + val_games}"
            )


def _report_hyperparams_once(task: Any, run_dir: Path) -> None:
    """Full run.yaml (HOCON) + lean connect() dict for +HYPERPARAM columns."""
    run_yaml = run_dir / RUN_YAML
    if not run_yaml.is_file():
        return
    with contextlib.suppress(Exception):
        task.connect_configuration(str(run_yaml))
    try:
        import yaml  # type: ignore[import-untyped]

        cfg = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(cfg, dict):
        return
    loop = cfg.get("loop") if isinstance(cfg.get("loop"), dict) else {}
    opt = cfg.get("optimizer") if isinstance(cfg.get("optimizer"), dict) else {}
    sched = cfg.get("scheduler") if isinstance(cfg.get("scheduler"), dict) else {}
    ev = cfg.get("eval") if isinstance(cfg.get("eval"), dict) else {}
    lean = {
        "microbatch": loop.get("microbatch_size"),
        "accum": loop.get("accumulation_steps"),
        "lr": opt.get("lr"),
        "warmup": sched.get("warmup_updates"),
        "ckpt_every": loop.get("checkpoint_frequency_updates"),
        "eval_every": ev.get("frequency_updates"),
    }
    lean = {k: v for k, v in lean.items() if isinstance(v, (int, float))}
    if lean:
        with contextlib.suppress(Exception):
            task.connect(lean)


def _report_checkpoint_text(logger: Any, run_dir: Path, offsets: dict[str, Any]) -> int:
    """One CONSOLE line per new ckpt file + epoch-roll marker. Returns count."""
    try:
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt-*.pt"))
    except OSError:
        return 0
    done = offsets.get("ckpt_text_done")
    done_set = set(done) if isinstance(done, list) else set()
    last_epoch = offsets.get("last_epoch")
    count = 0
    for ckpt in ckpts:
        name = ckpt.name
        if name in done_set:
            continue
        try:
            size = ckpt.stat().st_size
        except OSError:
            size = -1
        with contextlib.suppress(Exception):
            logger.report_text(f"checkpoint {name} bytes={size}")
        try:
            sidecar = ckpt.with_suffix(".json")
            if sidecar.is_file():
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                epoch = data.get("stream_epoch")
                if isinstance(epoch, int) and last_epoch is not None and epoch != last_epoch:
                    with contextlib.suppress(Exception):
                        logger.report_text(f"epoch roll -> {epoch} at {name}")
                if isinstance(epoch, int):
                    last_epoch = epoch
        except (OSError, ValueError):
            pass
        done_set.add(name)
        count += 1
    offsets["ckpt_text_done"] = sorted(done_set)
    if last_epoch is not None:
        offsets["last_epoch"] = last_epoch
    return count


def _offset_path(run_dir: Path) -> Path:
    return run_dir / "logs" / OFFSET_NAME


def _load_offsets(run_dir: Path) -> dict[str, Any]:
    try:
        raw = _offset_path(run_dir).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_offsets(run_dir: Path, offsets: dict[str, Any]) -> None:
    with contextlib.suppress(OSError):
        _offset_path(run_dir).write_text(json.dumps(offsets), encoding="utf-8")


def _report_meta_once(task: Any, logger: Any, run_dir: Path) -> None:
    """Hyperparams (once) + legacy Summary single values (best-effort).

    Replaces the old train_summary.json no-op: run.yaml is the live source.
    Best-so-far NLL single values are updated per-pass in _pass, not here.
    """
    _report_hyperparams_once(task, run_dir)
    summary_path = run_dir / "train_summary.json"
    if summary_path.is_file():
        for row in _iter_jsonl(summary_path):
            if isinstance(row, dict):
                for key in ("run_digest", "stream_manifest_hash"):
                    value = _as_float(row.get(key))
                    if value is not None:
                        logger.report_single_value(name=key, value=value)


def _upload_artifacts(task: Any, run_dir: Path) -> list[str]:
    """Upload last checkpoint + sidecar JSON + eval report (fileserver only).

    Checkpoints are ~25MB, over the 15MB events-API cap, so they must go
    through artifact upload. Missing files are skipped, never fatal.
    """
    uploaded: list[str] = []
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt-*.pt"))
    if ckpts:
        with contextlib.suppress(Exception):
            task.upload_artifact("checkpoint-last", str(ckpts[-1]), wait_on_upload=True)
            uploaded.append(ckpts[-1].name)
        # Full shuffle/cursor/RNG state (3-4MB JSON) alongside the weights.
        sidecar = ckpts[-1].with_suffix(".json")
        if sidecar.is_file():
            with contextlib.suppress(Exception):
                task.upload_artifact("checkpoint-last-json", str(sidecar), wait_on_upload=True)
                uploaded.append(sidecar.name)
    eval_path = run_dir / "eval" / "eval.jsonl"
    if eval_path.is_file():
        with contextlib.suppress(Exception):
            task.upload_artifact("eval-jsonl", str(eval_path), wait_on_upload=True)
            uploaded.append("eval.jsonl")
    return uploaded


def _pass(
    logger: Any,
    run_dir: Path,
    offsets: dict[str, Any],
    verbose_prev: dict[str, float] | None = None,
) -> tuple[int, int, int, int, int]:
    """Report every unseen row; persist offsets; return (metrics, eval, feed, verbose, spans)."""
    counts = [0, 0, 0, 0, 0]
    # Rollback point for the flush gate at the end: offsets mutate in place
    # throughout the pass, so a failed flush must restore (not just skip the
    # save) or the rows are skipped anyway. Shallow copy suffices: every
    # mutation below rebinds keys, never mutates a shared value in place.
    _offsets_snapshot = dict(offsets)
    best_train = offsets.get("best_train_nll")
    best_eval = offsets.get("best_eval_nll")
    if not isinstance(best_train, (int, float)):
        best_train = None
    if not isinstance(best_eval, (int, float)):
        best_eval = None
    rows, offsets["metrics"] = _read_new_rows(
        run_dir / METRICS_FILE, int(offsets.get("metrics", 0))
    )
    for row in rows:
        _report_metrics_row(logger, row)
        counts[0] += 1
        nll = _as_float(row.get("masked_nll"))
        if nll is not None and (best_train is None or nll < best_train):
            best_train = nll
            with contextlib.suppress(Exception):
                logger.report_single_value(name="best_train_nll", value=nll)
    rows, offsets["eval"] = _read_new_rows(run_dir / EVAL_FILE, int(offsets.get("eval", 0)))
    for row in rows:
        _report_eval_row(logger, row)
        _report_eval_tables(logger, row)
        counts[1] += 1
        nll = _as_float(row.get("masked_nll"))
        if nll is not None and (best_eval is None or nll < best_eval):
            best_eval = nll
            with contextlib.suppress(Exception):
                logger.report_single_value(name="best_eval_nll", value=nll)
    feed_count = int(offsets.get("feed_count", 0))
    rows, offsets["feed"] = _read_new_rows(run_dir / FEED_FILE, int(offsets.get("feed", 0)))
    # Flood guard: 4 microbatches/update hit ~400k rows over 100k updates.
    # Report every 4th (one per update); GC counters stay monotone so offline
    # deltas remain exact. Update-kind rows (optimizer_ms/logging_ms) are
    # rare and always reported.
    for row in rows:
        if row.get("kind") == "update" or (feed_count % 4 == 0):
            _report_feed_row(logger, row, feed_count)
            counts[2] += 1
        feed_count += 1
    offsets["feed_count"] = feed_count
    # Verbose GPU/torch/host at real 50ms sample clocks (throughput derived here).
    if verbose_prev is None:
        verbose_prev = {}
        prev_update_raw = offsets.get("verbose_prev_update")
        prev_wall_raw = offsets.get("verbose_prev_wall")
        if isinstance(prev_update_raw, (int, float)) and isinstance(prev_wall_raw, (int, float)):
            verbose_prev["update"] = float(prev_update_raw)
            verbose_prev["wall"] = float(prev_wall_raw)
    verbose_fallback = int(offsets.get("verbose_fallback", 0))
    rows, offsets["verbose"] = _read_new_rows(
        run_dir / VERBOSE_FILE, int(offsets.get("verbose", 0))
    )
    # Flood guard: 50ms rows hit ~720k rows / ~7M events over 100k updates
    # (~1GB JSONL). Report every 10th (500ms, ~72k rows); throughput stays
    # correct (update-change interval, not sample gap) since prev advances
    # only on reported rows.
    for idx, row in enumerate(rows):
        if idx % 10 == 0:
            _report_verbose_row(logger, row, verbose_fallback, verbose_prev)
            counts[3] += 1
        verbose_fallback += 1
    if "update" in verbose_prev and "wall" in verbose_prev:
        offsets["verbose_prev_update"] = verbose_prev["update"]
        offsets["verbose_prev_wall"] = verbose_prev["wall"]
    # Sparse stage spans (nil cost).
    spans_fallback = int(offsets.get("spans_fallback", 0))
    rows, offsets["spans"] = _read_new_rows(run_dir / SPANS_FILE, int(offsets.get("spans", 0)))
    for row in rows:
        _report_span_row(logger, row, spans_fallback)
        spans_fallback += 1
        counts[4] += 1
    offsets["spans_fallback"] = spans_fallback
    # CONSOLE checkpoint + epoch markers (never SCALARS).
    _report_checkpoint_text(logger, run_dir, offsets)
    if best_train is not None:
        offsets["best_train_nll"] = best_train
    if best_eval is not None:
        offsets["best_eval_nll"] = best_eval
    try:
        logger.flush()
    except Exception as exc:
        # Durability ordering: offsets advance only past server-confirmed
        # rows. A failed flush rolls the in-memory offsets back so the next
        # pass re-reports the same rows (harmless duplicates) instead of
        # losing them while claiming them consumed. A kill between report
        # and flush loses at most one pass; the SIGTERM handler closes that
        # window.
        print(f"sidecar: logger.flush failed, offsets held: {exc}", flush=True)
        offsets.clear()
        offsets.update(_offsets_snapshot)
        return counts[0], counts[1], counts[2], counts[3], counts[4]
    _save_offsets(run_dir, offsets)
    return counts[0], counts[1], counts[2], counts[3], counts[4]


def _task_init_kwargs(run_id: str, args: Any) -> dict[str, Any]:
    """Task.init kwargs: explicit --task-id continues with history intact.

    Without --task-id the SDK reuses by name and WIPES previous outputs
    (overwrite, not continue) — so resumes that must not fragment history
    pass the exact id. ``False`` preserves the legacy default exactly.
    """
    return {
        "project_name": args.project,
        "task_name": run_id,
        "tags": args.tags,
        "continue_last_task": args.task_id or False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report run JSONL to ClearML.")
    parser.add_argument("--run-dir", required=True, help="runs/<id>/ directory")
    parser.add_argument("--project", default="hydra2-tenhou-4p")
    parser.add_argument("--online", action="store_true", help="live server upload")
    parser.add_argument("--tags", nargs="*", default=[], help="task tags")
    parser.add_argument("--follow", action="store_true", help="tail until done sentinel")
    parser.add_argument("--follow-interval", type=float, default=30.0)
    parser.add_argument("--follow-timeout", type=float, default=57600.0)
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument(
        "--task-id",
        default=None,
        help="continue this exact ClearML task (history intact). Without it "
        "Task.init reuses by name and WIPES previous outputs.",
    )
    args = parser.parse_args()

    # ClearML import lives here only; training modules never import the SDK.
    from clearml import Logger, Task

    run_dir = Path(args.run_dir)
    run_id = run_dir.name

    # Single-reporter guard: exclusive non-blocking flock on logs/.sidecar-lock.
    # ClearML's BackgroundMonitor fork (mp.py os.fork) inherits the same FD so
    # parent+child share the lock; a second sidecar (new FD) exits 42. Never
    # delete logs/.sidecar-offset (old rows would re-report).
    lock_path = run_dir / "logs" / LOCK_NAME
    lock_fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid() if hasattr(os, "getpid") else "sidecar"))
        lock_fd.flush()
    except OSError as exc:
        print("sidecar: already running", file=sys.stderr)
        raise SystemExit(42) from exc
    if not args.online:
        Task.set_offline(True)
    task = Task.init(**_task_init_kwargs(run_id, args))
    logger = Logger.current_logger()
    if args.task_id is not None and task.id != args.task_id:
        print(f"sidecar: task mismatch {task.id} != {args.task_id}", file=sys.stderr)
        raise SystemExit(3)

    offsets = _load_offsets(run_dir)
    total_m = total_e = total_f = total_v = total_s = 0
    verbose_prev: dict[str, float] = {}
    if isinstance(offsets.get("verbose_prev_update"), (int, float)) and isinstance(
        offsets.get("verbose_prev_wall"), (int, float)
    ):
        verbose_prev["update"] = float(offsets["verbose_prev_update"])  # type: ignore[arg-type]
        verbose_prev["wall"] = float(offsets["verbose_prev_wall"])  # type: ignore[arg-type]
    if not offsets.get("meta_done"):
        _report_meta_once(task, logger, run_dir)
        offsets["meta_done"] = True
        _save_offsets(run_dir, offsets)
    # Upgrade path: old runs set meta_done before hyperparams existed.
    # Report connect_configuration/connect() once, then mark versioned.
    if not offsets.get("hyperparams_done"):
        _report_hyperparams_once(task, run_dir)
        offsets["hyperparams_done"] = True
        _save_offsets(run_dir, offsets)
    # Source mix: per-root game tallies from the scan report, once.
    if not offsets.get("source_mix_done"):
        _report_source_mix_once(task, logger, run_dir)
        offsets["source_mix_done"] = True
        _save_offsets(run_dir, offsets)

    mode = "follow" if args.follow else "oneshot"
    if args.follow:
        _install_termination_flush(logger, task)
        deadline = time.monotonic() + args.follow_timeout
        while True:
            m, e, f, v, s = _pass(logger, run_dir, offsets, verbose_prev)
            total_m += m
            total_e += e
            total_f += f
            total_v += v
            total_s += s
            try:
                logger.flush()
            except Exception as exc:
                # Flush failures used to vanish inside suppress while offsets
                # kept advancing: rows marked consumed that never reached the
                # server. Loud now; the rows stay reported-or-retried.
                print(f"sidecar: logger.flush failed: {exc}", flush=True)
            # Heartbeat: wall-clock proof of forward motion every pass (the
            # training log only moves per segment; a silent follower used to
            # look dead). Observer-only; never touches training state.
            print(
                f"watch: update={verbose_prev.get('update')} "
                f"t_wall_s={verbose_prev.get('wall')} "
                f"gpu_util={verbose_prev.get('gpu_util')} "
                f"gpu_mem_mb={verbose_prev.get('gpu_mem_mb')} "
                f"passed(m/e/f/v/s)={m}/{e}/{f}/{v}/{s}",
                flush=True,
            )
            done_sentinel = (run_dir / "logs" / DONE_NAME).exists()
            if done_sentinel and (m + e + f + v + s) == 0:
                mode = "follow-complete"
                break
            if time.monotonic() >= deadline:
                mode = "follow-timeout"
                break
            time.sleep(args.follow_interval)
    else:
        m, e, f, v, s = _pass(logger, run_dir, offsets, verbose_prev)
        total_m, total_e, total_f, total_v, total_s = m, e, f, v, s

    uploaded: list[str] = []
    if args.upload_artifacts:
        uploaded = _upload_artifacts(task, run_dir)

    print(
        f"sidecar: run={run_id} mode={mode} metric_rows={total_m} "
        f"eval_rows={total_e} feed_rows={total_f} verbose_rows={total_v} "
        f"span_rows={total_s} uploaded={uploaded}"
    )
    task.close()


if __name__ == "__main__":
    main()
