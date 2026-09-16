#!/usr/bin/env python3
"""Feed-rate bench harness — Wave-A P0-A baseline on the CURRENT tree.

Measures the Python data-path feed rate (stream framing + replay expansion)
in decisions/sec. No Rust changes, no torch-math changes, zero new Python
deps: imports are stdlib + torch + in-repo ``hydra2`` modules only.

Subcommands (all via ``pixi run python bench/feed_rate.py ...``):
  build-corpus  generate F1/F2/F8 corpus files + emit bench/corpus_entries.json
                (entry records for P0-B to embed in bench_corpus_manifest.json;
                this script NEVER writes bench_corpus_manifest.json).
  smoke         F1 expand count==5 + F8 660-emission leg + F9 geometry check.
  run           cold (3x fresh-process) + warm (2 discard + 5 timed) passes,
                reported SEPARATELY (never averaged), canonical artifact to
                $HYDRA2_ARTIFACT_ROOT/reports/feed-rate/<manifest>/<config>.json
                with blank (null) thresholds.
  run-once      internal single timed pass (used by cold fresh-process spawns).

Contracts:
  - dora (5,) -1 sentinel NEVER padded (asserted on every expanded row).
  - cold+warm reported separately; unpinned runs are marked
    threads="unpinned", gate="invalid".
  - This script NEVER calls bench/perf_wrap.sh (manual perf wrapper only).
  - Percentiles reuse the pinned_ring._percentile semantics (sorted,
    idx=min(int(q*n), n-1)); median via stdlib statistics.median.
  - B3 (synth-gen downgraded gate F5): any Rust synth helper for the F2/F8
    RNG loops is gated sha256-of-raw-bytes + quarantine-0, NEVER MT19937
    (`random.Random`) draw equality vs Rust by construction; F10/F11 template
    stamping stays Python (template shas + `expand_game` probe are
    Python-side contracts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# CUBLAS pre-CUDA precedent (tests/conftest.py): MUST precede torch/CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra2.artifacts.atomic import atomic_replace_bytes  # noqa: E402
from hydra2.artifacts.canonical import canonical_bytes  # noqa: E402
from hydra2.artifacts.digest import sha256_digest, sha256_file  # noqa: E402
from hydra2.data.decode import GameRecord  # noqa: E402
from hydra2.data.replay_expand import (  # noqa: E402
    _END_TYPES,
    _START_TYPES,
    expand_game,
    iter_microbatches,
)
from hydra2.data.stream import (  # noqa: E402
    PrefetchGameStream,
    build_manifest,
    slice_microbatches,
)
from hydra2.training.pinned_ring import (  # noqa: E402
    PinnedRing,
    ring_nbytes,
    slot_dtypes,
    slot_layout,
    slot_nbytes,
)

# F1 builders reused from the golden test module (never duplicated here).
try:
    from tests.unit.test_replay_expand_wp14 import (  # noqa: E402
        OBJECT_ID as _GOLD_OBJECT_ID,
    )
    from tests.unit.test_replay_expand_wp14 import (
        _golden_events as _test_golden_events,
    )
    from tests.unit.test_replay_expand_wp14 import (
        _golden_game as _test_golden_game,
    )
except ImportError:  # pragma: no cover - fallback loads the same module by path
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "test_replay_expand_wp14",
        REPO_ROOT / "tests" / "unit" / "test_replay_expand_wp14.py",
    )
    assert _spec is not None and _spec.loader is not None
    _mod = _ilu.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    _test_golden_events = _mod._golden_events
    _test_golden_game = _mod._golden_game
    _GOLD_OBJECT_ID = _mod.OBJECT_ID

# ---------------------------------------------------------------------------
# Constants: fixture ids, seeds, ratios (mirrors tests/unit/test_stream_wp14).
# ---------------------------------------------------------------------------

_RATIOS = {"train": 0.6, "validation": 0.4}
_SEED = 7
_F2_SEED = 20260909
_F2_TARGET_LINES = 1500
_F8_FILES = 30
_F8_GAMES_PER_FILE = 22
_F8_MID = 3
_F1_TOTAL_LINES = 50
_F1_DECISIONS = 5
_F10_FILES = 8
_F10_GAMES_PER_FILE = 12
_F10_BUILDER = "f10-v1"
_F11_FILES = 64
_F11_GAMES_PER_FILE = 12
_F11_BUILDER = "f11-v1"
_F10_TEMPLATES = (
    "tools/hydra2-replay-rs/tests/fixtures/s7/walled-synth.jsonl",
    "tools/hydra2-replay-rs/tests/fixtures/s7/walled-real.jsonl",
)

_CORPUS_FILES = {
    "F1": "f1-smoke-50x5.mjai.json",
    "F2": "f2-primary-1500.mjai.json.zst",
    "F8_DIR": "f8-stream-30x22",
    "F10_DIR": "f10-decisions",
    "F11_DIR": "f11-saturated",
}

# F3-F7 reference existing repo fixtures (never copied, only pinned).
_REF_FIXTURES: dict[str, list[str]] = {
    "F3": [
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-synth.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-real.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-ankan.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-kakan.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-truncated.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s7/walled-post-terminal.jsonl",
    ],
    "F4": [
        "tools/hydra2-replay-rs/tests/fixtures/s4/good-a.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/good-b.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-framing.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-truncated.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-unknown-event.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-bare-dora.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-double-ron.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-turn-order.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-tile-conservation.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s4/q-wall-bearing.jsonl",
    ],
    "F5": [
        "tools/hydra2-replay-rs/tests/fixtures/s5/priv-a.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s5/priv-tie.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s5/priv-missing.jsonl",
        "tools/hydra2-replay-rs/tests/fixtures/s5/priv-alias.jsonl",
    ],
    "F6": [
        "tools/mjai-dataset-packager/tests/fixtures/golden/a.mjai.json.zst",
        "tools/mjai-dataset-packager/tests/fixtures/golden/b.mjai.json.zst",
    ],
    "F7": [
        "tools/hydra2-replay-rs/tests/fixtures/frozen-row-hashes.json",
        "scripts/freeze_row_hashes.py",
    ],
}

_GATE_IDS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


class BenchError(RuntimeError):
    """Fail-closed bench error (never silent, never a fallback number)."""


def _percentile(xs: list[float], q: float) -> float | None:
    """pNN with pinned_ring._percentile semantics (sorted, idx=min(int(q*n), n-1))."""
    if not xs:
        return None
    ordered = sorted(xs)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


# ---------------------------------------------------------------------------
# Runtime configuration: thread pinning, determinism (conftest precedent).
# ---------------------------------------------------------------------------


def _default_threads() -> int:
    try:
        return max(1, len(os.sched_affinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


_INTEROP_SET = False


def _clamp_interop(n: int, record: dict[str, object]) -> None:
    """set_num_interop_threads is one-shot per process; later calls raise."""
    global _INTEROP_SET
    if _INTEROP_SET:
        return
    try:
        torch.set_num_interop_threads(min(2, n))
        _INTEROP_SET = True
    except RuntimeError as exc:
        record["interop_note"] = f"already bound: {exc}"


def _configure_runtime(*, threads: int | None, pinned: bool) -> dict[str, object]:
    """Pin threads + deterministic flags. Returns the observed thread record."""
    record: dict[str, object] = {"pinned": pinned}
    if pinned:
        n = threads or _default_threads()
        try:
            avail = sorted(os.sched_affinity(0))
            os.sched_affinity(0, avail[: max(1, min(n, len(avail)))])
            record["affinity"] = sorted(os.sched_affinity(0))
        except (AttributeError, OSError) as exc:
            record["affinity"] = f"unavailable:{exc}"
        # BLAS pools bind at first use; exports must precede process start to
        # bind external pools — record effective values either way.
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
        torch.set_num_threads(n)
        _clamp_interop(n, record)
        record["threads"] = n
        record["omp"] = os.environ.get("OMP_NUM_THREADS")
        record["mkl"] = os.environ.get("MKL_NUM_THREADS")
    else:
        record["threads"] = "unpinned"
        try:
            record["affinity"] = sorted(os.sched_affinity(0))
        except (AttributeError, OSError) as exc:
            record["affinity"] = f"unavailable:{exc}"
        record["observed_torch_threads"] = torch.get_num_threads()
    if torch.cuda.is_available():
        # Deterministic precedent; CUBLAS config exported at module top.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    record["cublas"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    record["cuda_available"] = torch.cuda.is_available()
    return record


def _fingerprint() -> dict[str, object]:
    lock = REPO_ROOT / "pixi.lock"
    try:
        pixi = str(sha256_file(lock))
    except OSError:
        pixi = "missing"
    try:
        affinity: object = sorted(os.sched_affinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        gpu: object = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (RuntimeError, AssertionError):
        gpu = "query-failed"
    return {
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "cuda_available": torch.cuda.is_available(),
        "kernel": platform.release(),
        "nproc": os.cpu_count(),
        "affinity": affinity,
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "pixi_lock": pixi,
    }


# ---------------------------------------------------------------------------
# Corpus generators (stdlib json + zstd CLI via subprocess; no newpy deps).
# ---------------------------------------------------------------------------


def _zstd_bin() -> str | None:
    return shutil.which("zstd")


def _zstd_version() -> str | None:
    exe = _zstd_bin()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout else None


def _compress_zst(raw: bytes, *, level: int) -> bytes:
    exe = _zstd_bin()
    if exe is None:
        raise BenchError("zstd CLI not found on PATH; cannot write .zst corpus")
    try:
        proc = subprocess.run(
            [exe, f"-{level}", "-c"], input=raw, capture_output=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise BenchError(f"zstd compression failed: {exc.stderr.decode()[:200]}") from exc
    return proc.stdout


def _zst_line_count(path: Path) -> int | None:
    """Decompressed newline count via zstd CLI (stdlib subprocess only)."""
    exe = _zstd_bin()
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, "-dc", str(path)], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    data = proc.stdout
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def _f1_events() -> list[dict[str, object]]:
    """Golden 5-decision events padded with skip-type dora no-ops to 50 lines.

    ``dora`` is in _SKIP_TYPES: the expander skips it verbatim (never a row,
    never simulator state), so the decision count stays exactly 5.
    """
    base = [dict(e) for e in _test_golden_events()]
    if len(base) >= _F1_TOTAL_LINES:
        raise BenchError(f"golden base already {len(base)} lines; cannot pad to 50")
    pads = [{"type": "dora"} for _ in range(_F1_TOTAL_LINES - len(base))]
    events = base[:2] + pads + base[2:]
    assert len(events) == _F1_TOTAL_LINES
    for line in events:
        if line.get("type") in _START_TYPES or line.get("type") in _END_TYPES:
            line.setdefault("game_id", "bench-f1-smoke")
    return events


def _stream_game_lines(
    game_id: str, n_mid: int, *, wall: bool = False, dora4: bool = False
) -> list[dict[str, object]]:
    """One stream-test-shaped game (mirrors tests/unit/test_stream_wp14)."""
    start: dict[str, object] = {"game_id": game_id, "type": "start_game"}
    if wall:
        start["wall"] = list(range(136))
    lines = [start]
    for seat in range(n_mid):
        lines.append({"seat": seat % 4, "type": "turn_advance"})
    if dora4:
        lines.append({"dora": [1, 2, 3, 4], "type": "dahai"})
    lines.append({"game_id": game_id, "type": "end_game"})
    return lines


def _jsonl_bytes(games: list[list[dict[str, object]]]) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    n_lines = 0
    for game in games:
        for line in game:
            chunks.append(json.dumps(line).encode() + b"\n")
            n_lines += 1
    return b"".join(chunks), n_lines


def _build_f1(corpus_dir: Path) -> dict[str, object]:
    raw, n_lines = _jsonl_bytes([_f1_events()])
    path = corpus_dir / _CORPUS_FILES["F1"]
    path.write_bytes(raw)
    digest = str(sha256_digest(raw))
    return {
        "id": "F1",
        "relpath": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": digest,
        "n_lines": n_lines,
        "n_games": 1,
        "n_decisions": _F1_DECISIONS,
        "kind": "synth-expand-smoke",
        "note": "golden builders + skip-type dora pads; expand must yield 5 rows",
    }


def _build_f2(corpus_dir: Path, *, level: int) -> dict[str, object]:
    rng = random.Random(_F2_SEED)
    games: list[list[dict[str, object]]] = []
    total = 0
    gid = 0
    while True:
        n_mid = rng.randint(0, 8)
        size = n_mid + 2
        if total + size > _F2_TARGET_LINES:
            n_mid = _F2_TARGET_LINES - total - 2
            if n_mid < 0:
                break
            size = n_mid + 2
        games.append(_stream_game_lines(f"bench-f2-g{gid:05d}", n_mid))
        total += size
        gid += 1
        if total >= _F2_TARGET_LINES:
            break
    assert total == _F2_TARGET_LINES, f"F2 line budget missed: {total}"
    raw, n_lines = _jsonl_bytes(games)
    assert n_lines == _F2_TARGET_LINES
    zst = _compress_zst(raw, level=level)
    path = corpus_dir / _CORPUS_FILES["F2"]
    path.write_bytes(zst)
    return {
        "id": "F2",
        "relpath": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": str(sha256_digest(zst)),
        "sha256_raw": str(sha256_digest(raw)),
        "n_lines": n_lines,
        "n_games": len(games),
        "kind": "synth-stream-primary",
        "zstd_level": level,
        "zstd_version": _zstd_version(),
        "seed": _F2_SEED,
        "note": "scales test_stream_wp14 _game_bytes; stream leg only (turn_advance is not expand vocab)",
    }


def _build_f8(corpus_dir: Path, *, level: int) -> list[dict[str, object]]:
    f8_dir = corpus_dir / _CORPUS_FILES["F8_DIR"]
    f8_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for fi in range(_F8_FILES):
        games = [
            _stream_game_lines(f"bench-f8-{fi:02d}-{gi:02d}", _F8_MID)
            for gi in range(_F8_GAMES_PER_FILE)
        ]
        raw, n_lines = _jsonl_bytes(games)
        zst = _compress_zst(raw, level=level)
        # Tenhou-style stem required by stream tenhou-name rules.
        path = f8_dir / f"2024010100gm-00a9-0000-{fi:08d}.mjai.json.zst"
        path.write_bytes(zst)
        entries.append(
            {
                "id": "F8",
                "part": path.name,
                "relpath": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": str(sha256_digest(zst)),
                "n_lines": n_lines,
                "n_games": len(games),
                "n_decisions": len(games),
                "kind": "stream-leg-shard",
                "zstd_level": level,
                "note": "30x22 emission leg; 660 games total, counted via PrefetchGameStream",
            }
        )

    return entries


def _f10_templates() -> list[list[dict[str, object]]]:
    """Load the s7 MJAI templates (synth 5-row, real 4-row incl. hora)."""
    out: list[list[dict[str, object]]] = []
    for rel in _F10_TEMPLATES:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise BenchError(f"F10 template missing: {rel}")
        out.append(
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        )
    return out


def _f10_stamp(
    template: list[dict[str, object]], game_id: str
) -> list[dict[str, object]]:
    """Clone one template game with boundary game_id stamped (wall shared)."""
    events: list[dict[str, object]] = []
    for line in template:
        e = dict(line)
        if e.get("type") in _START_TYPES or e.get("type") in _END_TYPES:
            e["game_id"] = game_id
        events.append(e)
    return events
def _f10_wall(template: list[dict[str, object]]) -> tuple[int, ...]:
    """Template's own 136-wall (mirrors decode.py extraction, not identity)."""
    for ev in template:
        for key in ("wall", "wall_tiles", "tiles"):
            w = ev.get(key)
            if isinstance(w, list) and len(w) == 136 and all(isinstance(x, int) for x in w):
                return tuple(int(x) for x in w)
    return tuple(range(136))
def _f10_template_rows(templates: list[list[dict[str, object]]]) -> list[int]:
    """Exact per-template decisions via expand_game (fail-closed on zero)."""
    counts: list[int] = []
    for ti, tpl in enumerate(templates):
        game = GameRecord(
            game_id=f"bench-f10-probe-{ti}",
            object_id=f"bench-f10-probe-obj-{ti}",
            packaged_object_id=f"bench-f10-probe-pkg-{ti}",
            events=tuple(_f10_stamp(tpl, f"bench-f10-probe-{ti}")),
            raw_bytes_sha256="sha256:" + "0" * 64,
            wall_tiles=_f10_wall(tpl),
            source={"type": "start_game"},
        )
        rows = expand_game(game)
        if not rows:
            raise BenchError(f"F10 template {ti} expands to zero rows (vacuous)")
        counts.append(len(rows))
    return counts

def _build_f10(corpus_dir: Path, *, level: int) -> list[dict[str, object]]:
    """Decision-bearing primary: F10 files of real dahai/hora MJAI games.

    Deterministic alternation over the s7 templates (no RNG); per-template
    row counts measured once via expand_game so n_decisions is exact, never
    estimated. Builder version + template shas pin the bytes (builder hash).
    """
    templates = _f10_templates()
    template_shas = [
        str(sha256_file(REPO_ROOT / rel)) for rel in _F10_TEMPLATES
    ]
    # Exact per-template decisions (fail-closed: template must expand).
    template_rows = _f10_template_rows(templates)
    builder_hash = str(
        sha256_digest((_F10_BUILDER + "\n" + "\n".join(template_shas)).encode())
    )
    f10_dir = corpus_dir / _CORPUS_FILES["F10_DIR"]
    f10_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for fi in range(_F10_FILES):
        games = [
            _f10_stamp(templates[(fi + gi) % len(templates)], f"bench-f10-{fi:02d}-{gi:02d}")
            for gi in range(_F10_GAMES_PER_FILE)
        ]
        raw, n_lines = _jsonl_bytes(games)
        zst = _compress_zst(raw, level=level)
        path = f10_dir / f"2024010100gm-00a9-0000-91{fi:06d}.mjai.json.zst"
        path.write_bytes(zst)
        n_decisions = sum(
            template_rows[(fi + gi) % len(templates)] for gi in range(_F10_GAMES_PER_FILE)
        )
        entries.append(
            {
                "id": "F10",
                "part": path.name,
                "relpath": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": str(sha256_digest(zst)),
                "n_lines": n_lines,
                "n_games": len(games),
                "n_decisions": n_decisions,
                "kind": "decision-primary",
                "builder": _F10_BUILDER,
                "builder_hash": builder_hash,
                "template_sha256": template_shas,
                "zstd_level": level,
                "note": "real dahai/hora rows via replay_expand; stream leg expands every game",
            }
        )
    return entries


def _build_f11(corpus_dir: Path, *, level: int) -> list[dict[str, object]]:
    """Saturated decision-primary: F11 files of real dahai/hora MJAI games.

    f10-v1-style deterministic cloning over the same s7 templates (fixed
    round-robin, no RNG); every game carries a file-unique game_id stamp
    (bench-f11-<file>-<game>) so exact decoded-hash dedup keeps all 768
    games (BenchHarnessBuilder dedup caveat: no two games share bytes).
    Same zstd level as the build; builder version + template shas pin the
    bytes (builder hash). Per-template row counts measured once via
    expand_game so n_decisions is exact, never estimated.
    """
    templates = _f10_templates()
    template_shas = [
        str(sha256_file(REPO_ROOT / rel)) for rel in _F10_TEMPLATES
    ]
    # Exact per-template decisions (fail-closed: template must expand).
    template_rows = _f10_template_rows(templates)
    builder_hash = str(
        sha256_digest((_F11_BUILDER + "\n" + "\n".join(template_shas)).encode())
    )
    f11_dir = corpus_dir / _CORPUS_FILES["F11_DIR"]
    f11_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for fi in range(_F11_FILES):
        games = [
            _f10_stamp(templates[(fi + gi) % len(templates)], f"bench-f11-{fi:02d}-{gi:02d}")
            for gi in range(_F11_GAMES_PER_FILE)
        ]
        raw, n_lines = _jsonl_bytes(games)
        zst = _compress_zst(raw, level=level)
        path = f11_dir / f"2024010100gm-00a9-0000-92{fi:06d}.mjai.json.zst"
        path.write_bytes(zst)
        n_decisions = sum(
            template_rows[(fi + gi) % len(templates)] for gi in range(_F11_GAMES_PER_FILE)
        )
        entries.append(
            {
                "id": "F11",
                "part": path.name,
                "relpath": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": str(sha256_digest(zst)),
                "n_lines": n_lines,
                "n_games": len(games),
                "n_decisions": n_decisions,
                "kind": "decision-primary",
                "builder": _F11_BUILDER,
                "builder_hash": builder_hash,
                "template_sha256": template_shas,
                "zstd_level": level,
                "note": "saturated 64x12 decision-primary; file-unique game_id stamps keep exact-hash dedup at 768 games",
            }
        )
    return entries


def _count_text_lines(path: Path) -> int:
    data = path.read_bytes()
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _count_start_games(path: Path) -> int | None:
    try:
        n = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") in _START_TYPES:
                n += 1
        return n
    except (OSError, UnicodeDecodeError):
        return None


def _ref_entry(fid: str, rel: str) -> dict[str, object]:
    path = REPO_ROOT / rel
    if not path.is_file():
        raise BenchError(f"reference fixture missing: {rel}")
    entry: dict[str, object] = {
        "id": fid,
        "part": Path(rel).name,
        "relpath": rel,
        "sha256": str(sha256_file(path)),
        "kind": "repo-fixture",
    }
    if path.suffix == ".zst":
        entry["n_lines"] = _zst_line_count(path)
        entry["n_games"] = None
    else:
        entry["n_lines"] = _count_text_lines(path)
        entry["n_games"] = (
            _count_start_games(path) if path.suffix in (".jsonl", ".json") else None
        )
    return entry


def _geom_entry() -> dict[str, object]:
    """F9: 26-plane geometry + B=1024/T=256 buffer-math spot check (~9.8MB/slot)."""
    layout = slot_layout(1024, 256)
    if len(layout) != 26:
        raise BenchError(f"F9 geometry changed: {len(layout)} fields, expected 26")
    slot = slot_nbytes(layout)
    ring = ring_nbytes(layout, 2)
    if not 9.5e6 <= slot <= 10.0e6:
        raise BenchError(f"F9 slot budget violated: {slot} bytes not in [9.5MB, 10MB]")
    if ring != 2 * slot:
        raise BenchError("F9 ring geometry violated: ring != 2 * slot")
    return {
        "id": "F9",
        "relpath": None,
        "sha256": None,
        "n_lines": None,
        "n_games": None,
        "kind": "computed-geometry",
        "details": {
            "n_fields": len(layout),
            "B": 1024,
            "T": 256,
            "depth": 2,
            "slot_nbytes": slot,
            "ring_nbytes": ring,
            "note": "current-tree 26-plane baseline; Rust hot-13 ceiling is DERIVED "
            "from the ROW_BYTES sum (~0.89KB@T64/~2.6KB@T256, plan delta #EF78), "
            "never a hand constant",
        },
    }


def build_corpus(corpus_dir: Path, *, level: int) -> list[dict[str, object]]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = [_build_f1(corpus_dir), _build_f2(corpus_dir, level=level)]
    entries.extend(_build_f8(corpus_dir, level=level))
    entries.extend(_build_f10(corpus_dir, level=level))
    entries.extend(_build_f11(corpus_dir, level=level))
    for fid, rels in _REF_FIXTURES.items():
        for rel in rels:
            entries.append(_ref_entry(fid, rel))
    entries.append(_geom_entry())
    return entries


# ---------------------------------------------------------------------------
# Timed legs.
# ---------------------------------------------------------------------------


def _frame_jsonl_games(path: Path) -> list[list[dict[str, object]]]:
    games: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    in_game = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        kind = obj.get("type") if isinstance(obj, dict) else None
        if kind in _START_TYPES:
            current = [obj]
            in_game = True
        elif in_game:
            current.append(obj)
            if kind in _END_TYPES:
                games.append(current)
                current = []
                in_game = False
    if in_game:
        raise BenchError(f"unterminated game frame in {path}")
    return games


def _expand_pass(
    corpus_file: Path, *, chunk: int, ring: bool, B: int, T: int, depth: int
) -> dict[str, object]:
    t0 = time.perf_counter()
    games = _frame_jsonl_games(corpus_file)
    t_scan = time.perf_counter()
    rows: list[object] = []
    for idx, events in enumerate(games):
        game = GameRecord(
            game_id=f"bench-expand-{idx:04d}",
            object_id=f"bench-expand-obj-{idx:04d}",
            packaged_object_id=f"bench-expand-pkg-{idx:04d}",
            events=tuple(events),
            raw_bytes_sha256="sha256:" + "0" * 64,
            wall_tiles=tuple(range(136)),
            source={"type": "start_game"},
        )
        grown = expand_game(game)
        for row in grown:
            dora = row.actor_observation["dora_indicators"]
            if not isinstance(dora, list) or len(dora) != 5:
                raise BenchError("dora (5,) sentinel violated in bench expansion")
        rows.extend(grown)
    t_expand = time.perf_counter()
    batches = list(iter_microbatches(rows, chunk))
    t_micro = time.perf_counter()
    ring_info: dict[str, object] | None = None
    if ring:
        ring_info = _ring_probe(B=B, T=T, depth=depth)
    t_end = time.perf_counter()
    elapsed = t_end - t0
    n = len(rows)
    return {
        "unit": "expanded_rows",
        "n_decisions": n,
        "n_events": sum(len(g) for g in games),
        "n_batches": len(batches),
        "decisions_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "events_per_sec": sum(len(g) for g in games) / elapsed if elapsed > 0 else 0.0,
        "batches_per_sec": len(batches) / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
        "phases": {
            "scan_s": t_scan - t0,
            "expand_s": t_expand - t_scan,
            "microbatch_s": t_micro - t_expand,
            "ring_s": t_end - t_micro,
            "total_s": elapsed,
        },
        "ring": ring_info,
        "feed": "ring" if ring_info else "sync",
    }


def _expand_streamed(games: list) -> tuple[int, int]:
    """Expand every streamed game fail-closed; returns (n_rows, n_events).

    F10-ONLY: every game must bear decisions. Expansion failure aborts the
    pass (never a silent skip); dora (5,) is asserted per row. Framing-only
    corpora (F2/F8) never reach this helper (turn_advance is unmapped in
    replay_expand and wall-less — expansion would raise there).
    """
    n_rows = 0
    n_events = 0
    for game in games:
        n_events += len(game.game.events)
        try:
            rows = expand_game(game.game)
        except Exception as exc:
            raise BenchError(
                f"stream-leg expansion failed for {game.game.game_id}: {exc}"
            ) from exc
        for row in rows:
            dora = row.actor_observation["dora_indicators"]
            if not isinstance(dora, list) or len(dora) != 5:
                raise BenchError("dora (5,) sentinel violated in stream-leg expansion")
        n_rows += len(rows)
    return n_rows, n_events


def _stream_pass(
    corpus_root: Path, *, chunk: int, seed: int, ring: bool, B: int, T: int, depth: int,
    expand: bool = False,
) -> dict[str, object]:
    """Stream leg. ``expand=True`` (F10 only) expands every game fail-closed and
    reports decisions as expanded rows; framing-only corpora (F2/F8) keep the
    legacy emitted-games path (turn_advance is unmapped in replay_expand and
    wall-less, so expansion would raise — never attempted there).
    """
    t0 = time.perf_counter()
    manifest = build_manifest(corpus_root)
    t_scan = time.perf_counter()
    stream = PrefetchGameStream(manifest, seed=seed, ratios=dict(_RATIOS), split=None)
    games = list(stream)
    t_frame = time.perf_counter()
    if expand:
        n_decisions, n_events = _expand_streamed(games)
        unit = "expanded_rows"
    else:
        n_decisions, n_events = len(games), None
        unit = "emitted_games"
    t_expand = time.perf_counter()
    micros = list(slice_microbatches(games, chunk))
    t_micro = time.perf_counter()
    ring_info: dict[str, object] | None = None
    if ring:
        ring_info = _ring_probe(B=B, T=T, depth=depth)
    t_end = time.perf_counter()
    elapsed = t_end - t0
    n = len(games)
    stats = stream.stats
    counters: dict[str, object] = {}
    for key in ("framed", "emitted", "quarantined", "skipped", "waits"):
        if hasattr(stats, key):
            counters[key] = getattr(stats, key)
    return {
        "unit": unit,
        "n_games": n,
        "n_decisions": n_decisions,
        "n_events": n_events,
        "n_batches": len(micros),
        "decisions_per_sec": n_decisions / elapsed if elapsed > 0 else 0.0,
        "events_per_sec": (n_events / elapsed if elapsed > 0 else 0.0)
        if n_events is not None
        else None,
        "batches_per_sec": len(micros) / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
        "phases": {
            "scan_s": t_scan - t0,
            "framing_s": t_frame - t_scan,
            "expand_s": t_expand - t_frame,
            "microbatch_s": t_micro - t_expand,
            "ring_s": t_end - t_micro,
            "total_s": elapsed,
        },
        "ring": ring_info,
        "feed": "ring" if ring_info else "sync",
        "stream_counters": counters,
    }


def _ring_probe(*, B: int, T: int, depth: int) -> dict[str, object]:
    """Live ring collector: N=depth+3 refills, stats() carries h2d/sync windows."""
    layout = slot_layout(B, T)
    dtypes = slot_dtypes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ring = PinnedRing.open(layout, depth=depth, device=device)
    try:
        shapes = {name: shape for name, (shape, _dt) in layout.items()}
        for _ in range(depth + 3):
            batch = {
                name: torch.empty(shapes[name], dtype=dtypes[name]) for name in shapes
            }
            ring.next(batch)
        stats = ring.stats()
    finally:
        ring.close()
    return {
        "device": stats["device"],
        "cuda_available": stats["cuda_available"],
        "depth": stats["depth"],
        "acquires": stats["acquires"],
        "transfers": stats["transfers"],
        "slot_nbytes": stats["slot_nbytes"],
        "ring_nbytes": stats["ring_nbytes"],
        "h2d_ms_last": stats["h2d_ms_last"],
        "h2d_ms_p50": stats["h2d_ms_p50"],
        "h2d_ms_p99": stats["h2d_ms_p99"],
        "sync_wait_ms_last": stats["sync_wait_ms_last"],
        "sync_wait_ms_p50": stats["sync_wait_ms_p50"],
        "sync_wait_ms_p99": stats["sync_wait_ms_p99"],
    }


# ---------------------------------------------------------------------------
# Smoke (F1 + F8 + F9 acceptance legs, green on current tree).
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus_dir)
    # F1: golden builders + 50-line pads expand to exactly 5 rows.
    rows = expand_game(_test_golden_game(events=_f1_events()))
    assert len(rows) == _F1_DECISIONS, f"F1 smoke count==5 violated: {len(rows)}"
    f1_path = corpus_dir / _CORPUS_FILES["F1"]
    if f1_path.is_file():
        framed = _frame_jsonl_games(f1_path)
        assert len(framed) == 1, f"F1 corpus must hold 1 game, got {len(framed)}"
        file_rows = expand_game(_test_golden_game(events=[dict(e) for e in framed[0]]))
        assert len(file_rows) == _F1_DECISIONS, f"F1 file yields {len(file_rows)}, want 5"
    print(f"F1 smoke: count=={len(rows)} OK ({_F1_TOTAL_LINES}-line synth, 5 decisions)")
    # F8: 30x22 emission leg through the threaded stream.
    f8_dir = corpus_dir / _CORPUS_FILES["F8_DIR"]
    manifest = build_manifest(f8_dir)
    stream = PrefetchGameStream(manifest, seed=_SEED, ratios=dict(_RATIOS), split=None)
    games = list(stream)
    want = _F8_FILES * _F8_GAMES_PER_FILE
    assert len(games) == want, f"F8 leg: {len(games)} != {want}"
    micros = list(slice_microbatches(games, 32))
    covered = sum(len(m) for m in micros)
    assert covered == want, f"F8 microbatch cover {covered} != {want}"
    print(f"F8 leg: {len(games)} emissions (660-decision) OK in {len(micros)} slices")
    # F10: decision-bearing primary — every streamed game expands to rows.
    f10_dir = corpus_dir / _CORPUS_FILES["F10_DIR"]
    manifest10 = build_manifest(f10_dir)
    stream10 = PrefetchGameStream(manifest10, seed=_SEED, ratios=dict(_RATIOS), split=None)
    games10 = list(stream10)
    want10 = _F10_FILES * _F10_GAMES_PER_FILE
    assert len(games10) == want10, f"F10 leg: {len(games10)} != {want10}"
    rows10, _ = _expand_streamed(games10)
    assert rows10 > 0, "F10 leg vacuous: zero expanded rows"
    template_rows = _f10_template_rows(_f10_templates())
    want_rows = sum(
        template_rows[(fi + gi) % len(template_rows)]
        for fi in range(_F10_FILES)
        for gi in range(_F10_GAMES_PER_FILE)
    )
    assert rows10 == want_rows, f"F10 rows {rows10} != template total {want_rows}"
    print(f"F10 leg: {len(games10)} games -> {rows10} decisions OK (non-vacuous)")
    # F9: geometry spot check.
    geom = _geom_entry()
    det = geom["details"]
    assert isinstance(det, dict)
    print(f"F9 geometry: 26-plane slot={det['slot_nbytes']}B ring={det['ring_nbytes']}B OK")
    print("smoke: GREEN")
    return 0


# ---------------------------------------------------------------------------
# run-once (single timed pass; cold spawns this fresh) + run (cold+warm).
# ---------------------------------------------------------------------------


def _timed_pass(args: argparse.Namespace) -> dict[str, object]:
    _configure_runtime(threads=args.threads, pinned=args.pin)
    corpus_dir = Path(args.corpus_dir)
    if args.leg == "expand":
        return _expand_pass(
            corpus_dir / _CORPUS_FILES["F1"],
            chunk=args.chunk,
            ring=args.ring,
            B=args.B,
            T=args.T,
            depth=args.depth,
        )
    if args.leg == "stream":
        if args.corpus == "f2":
            # F2 is a single-file corpus: stream its parent with a narrow pattern
            # is unsupported, so the F2 timed leg frames the file directly.
            return _stream_file_pass(
                corpus_dir / _CORPUS_FILES["F2"],
                chunk=args.chunk,
                seed=args.seed,
                ring=args.ring,
                B=args.B,
                T=args.T,
                depth=args.depth,
            )
        subdir = _CORPUS_FILES["F10_DIR"] if args.corpus == "f10" else _CORPUS_FILES["F8_DIR"]
        return _stream_pass(
            corpus_dir / subdir, chunk=args.chunk, seed=args.seed, ring=args.ring,
            B=args.B, T=args.T, depth=args.depth, expand=(args.corpus == "f10"),
        )
    raise BenchError(f"unknown leg {args.leg!r}")


def _stream_file_pass(
    path: Path, *, chunk: int, seed: int, ring: bool, B: int, T: int, depth: int
) -> dict[str, object]:
    """Framing-only single-file leg (F2): legacy emitted-games path, untouched."""
    t0 = time.perf_counter()
    manifest = build_manifest(path.parent, pattern=path.name)
    t_scan = time.perf_counter()
    stream = PrefetchGameStream(manifest, seed=seed, ratios=dict(_RATIOS), split=None)
    games = list(stream)
    t_frame = time.perf_counter()
    micros = list(slice_microbatches(games, chunk))
    t_micro = time.perf_counter()
    ring_info = _ring_probe(B=B, T=T, depth=depth) if ring else None
    t_end = time.perf_counter()
    elapsed = t_end - t0
    n = len(games)
    return {
        "unit": "emitted_games",
        "n_games": n,
        "n_decisions": n,
        "n_events": _F2_TARGET_LINES if path.name == _CORPUS_FILES["F2"] else None,
        "n_batches": len(micros),
        "decisions_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "events_per_sec": None,
        "batches_per_sec": len(micros) / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
        "phases": {
            "scan_s": t_scan - t0,
            "framing_s": t_frame - t_scan,
            "microbatch_s": t_micro - t_frame,
            "ring_s": t_end - t_micro,
            "total_s": elapsed,
        },
        "ring": ring_info,
        "feed": "ring" if ring_info else "sync",
    }


def cmd_run_once(args: argparse.Namespace) -> int:
    result = _timed_pass(args)
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    return 0


def _summarize(runs: list[float]) -> dict[str, object]:
    return {
        "runs": runs,
        "median": statistics.median(runs),
        "p50": _percentile(runs, 0.50),
        "p99": _percentile(runs, 0.99),
    }


def cmd_run(args: argparse.Namespace) -> int:
    threads_record = _configure_runtime(threads=args.threads, pinned=args.pin)
    corpus_dir = Path(args.corpus_dir)
    entries_path = Path(args.entries)
    entries_digest = hashlib.sha256(
        canonical_bytes(json.loads(entries_path.read_text(encoding="utf-8")))
        if entries_path.is_file()
        else b"no-entries"
    ).hexdigest()
    if args.manifest_digest:
        manifest_tag = args.manifest_digest.removeprefix("sha256:")[:16]
    else:
        manifest_tag = entries_digest[:16]

    # Cold: 3 fresh processes (import + CUDA init inside the wall, by design).
    cold_passes: list[dict[str, object]] = []
    for _ in range(3):
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "run-once",
            "--corpus-dir", str(corpus_dir), "--leg", args.leg, "--corpus", args.corpus,
            "--chunk", str(args.chunk), "--seed", str(args.seed),
            "--B", str(args.B), "--T", str(args.T), "--depth", str(args.depth),
            "--threads", str(args.threads or 0),
        ]
        cmd.append("--pin" if args.pin else "--no-pin")
        cmd.append("--ring" if args.ring else "--no-ring")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise BenchError(f"cold pass failed: {proc.stderr[-2000:]}")
        cold_passes.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    # Warm: 2 discard + 5 timed, in-process.
    for _ in range(2):
        _timed_pass(args)
    warm_passes = [_timed_pass(args) for _ in range(5)]

    def rates(key: str) -> list[float]:
        return [float(p[key]) for p in warm_passes]

    def cold_rates(key: str) -> list[float]:
        return [float(p[key]) for p in cold_passes]

    def pick(key: str) -> float | None:
        vals = [p[key] for p in warm_passes if p[key] is not None]
        return float(vals[-1]) if vals else None

    def cold_pick(key: str) -> float | None:
        vals = [p[key] for p in cold_passes if p[key] is not None]
        return float(vals[0]) if vals else None

    cold_block = {
        "decisions_per_sec_runs": cold_rates("decisions_per_sec"),
        **{k: v for k, v in _summarize(cold_rates("decisions_per_sec")).items() if k != "runs"},
        "events_per_sec": cold_pick("events_per_sec"),
        "batches_per_sec": cold_pick("batches_per_sec"),
        "unit": cold_passes[0]["unit"],
    }
    warm_block = {
        "decisions_per_sec_runs": rates("decisions_per_sec"),
        **{k: v for k, v in _summarize(rates("decisions_per_sec")).items() if k != "runs"},
        "events_per_sec": pick("events_per_sec"),
        "batches_per_sec": pick("batches_per_sec"),
        "unit": warm_passes[-1]["unit"],
    }
    # Top-level aliases matching bench_corpus_manifest.json required_fields
    # (warm leg is the headline series; cold/warm blocks stay authoritative).
    def opt_rates(key: str) -> list[float]:
        return [float(p[key]) for p in warm_passes if p[key] is not None]

    def opt_median(vals: list[float]) -> float | None:
        return float(statistics.median(vals)) if vals else None

    warm_decisions = rates("decisions_per_sec")
    warm_events = opt_rates("events_per_sec")
    warm_batches = rates("batches_per_sec")

    if args.gate_verdicts:
        gate_verdicts: object = json.loads(Path(args.gate_verdicts).read_text(encoding="utf-8"))
    else:
        gate_verdicts = {gid: {"status": "pending", "owner": "P0-B"} for gid in _GATE_IDS}

    pinned = bool(args.pin)
    config = {
        "leg": args.leg,
        "corpus": args.corpus,
        "chunk": args.chunk,
        "zstd": args.zstd,
        "B": args.B,
        "T": args.T,
        "depth": args.depth,
        "threads": threads_record["threads"],
        "pinned": pinned,
        "seed": args.seed,
        "ring_probe": bool(args.ring),
    }
    slug = (
        f"{args.leg}-{args.corpus}-chunk{args.chunk}-z{args.zstd}"
        f"-B{args.B}-T{args.T}-d{args.depth}"
        f"-t{threads_record['threads'] if pinned else 'unpinned'}"
    )
    artifact = {
        "artifact": "feed-rate-baseline",
        "leg": args.leg,
        "manifest": manifest_tag,
        "manifest_digest": args.manifest_digest,
        "config": config,
        "fingerprint": _fingerprint(),
        "threads": threads_record,
        "cold": cold_block,
        "warm": warm_block,
        "decisions_per_sec_runs": warm_decisions,
        "decisions_per_sec_median": opt_median(warm_decisions),
        "events_per_sec_runs": warm_events,
        "events_per_sec_median": opt_median(warm_events),
        "batches_per_sec_runs": warm_batches,
        "batches_per_sec_median": opt_median(warm_batches),
        "gate_verdicts": gate_verdicts,
        "thresholds": None,
        "phases": {
            "warm_last": warm_passes[-1]["phases"],
            "cold_first": cold_passes[0]["phases"],
        },
        "ring": warm_passes[-1]["ring"],
        "geometry": {
            "slot_nbytes": slot_nbytes(slot_layout(args.B, args.T)),
            "ring_nbytes": ring_nbytes(slot_layout(args.B, args.T), args.depth),
        },
        "gate": "baseline-blank" if pinned else "invalid",
    }
    root = Path(os.environ.get("HYDRA2_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts")))
    dest = root / "reports" / "feed-rate" / manifest_tag / f"{slug}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(artifact)
    atomic_replace_bytes(dest, payload)
    digest = sha256_file(dest)
    print(f"cold decisions/sec runs: {cold_block['decisions_per_sec_runs']}")
    print(f"cold median: {cold_block['median']:.3f} {cold_block['unit']}/s")
    print(f"warm decisions/sec runs: {warm_block['decisions_per_sec_runs']}")
    print(f"warm median: {warm_block['median']:.3f} {warm_block['unit']}/s "
          f"(p50={warm_block['p50']:.3f} p99={warm_block['p99']:.3f})")
    print(f"gate: {artifact['gate']} (thresholds blank baseline, no pass/fail)")
    print(f"artifact: {dest} [{digest}]")
    return 0


def cmd_build_corpus(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus_dir)
    entries = build_corpus(corpus_dir, level=args.zstd)
    out = Path(args.entries_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries_version": "1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zstd_level": args.zstd,
        "zstd_version": _zstd_version(),
        "entries": entries,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"corpus: {corpus_dir} ({len(entries)} entries -> {out})")
    for entry in entries:
        print(f"  {entry['id']} {entry.get('part', '')} {entry.get('relpath')} "
              f"lines={entry.get('n_lines')} games={entry.get('n_games')}")
    return 0


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--corpus-dir", default=str(REPO_ROOT / "bench" / "corpus"))
    sp.add_argument("--threads", type=int, default=0,
                    help="worker threads when pinned (0 = auto affinity count)")
    pin = sp.add_mutually_exclusive_group()
    pin.add_argument("--pin", dest="pin", action="store_true", default=True)
    pin.add_argument("--no-pin", dest="pin", action="store_false",
                     help="unpinned run: threads=unpinned, gate=invalid")
    ring = sp.add_mutually_exclusive_group()
    ring.add_argument("--ring", dest="ring", action="store_true", default=True)
    ring.add_argument("--no-ring", dest="ring", action="store_false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hydra2 feed-rate bench (P0-A baseline)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-corpus")
    b.add_argument("--corpus-dir", default=str(REPO_ROOT / "bench" / "corpus"))
    b.add_argument("--entries-out", default=str(REPO_ROOT / "bench" / "corpus_entries.json"))
    b.add_argument("--zstd", type=int, default=1, choices=[1, 2, 3])
    b.set_defaults(fn=cmd_build_corpus)

    s = sub.add_parser("smoke")
    s.add_argument("--corpus-dir", default=str(REPO_ROOT / "bench" / "corpus"))
    s.set_defaults(fn=cmd_smoke)

    for name in ("run", "run-once"):
        p = sub.add_parser(name)
        _add_common(p)
        p.add_argument("--leg", choices=["expand", "stream"], default="stream")
        p.add_argument("--corpus", choices=["f1", "f2", "f8", "f10"], default="f10")
        p.add_argument("--chunk", type=int, default=32)
        p.add_argument("--seed", type=int, default=_SEED)
        p.add_argument("--B", type=int, default=32)
        p.add_argument("--T", type=int, default=64)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--zstd", type=int, default=1)
        p.add_argument("--entries", default=str(REPO_ROOT / "bench" / "corpus_entries.json"))
        p.add_argument("--manifest-digest", default=None)
        p.add_argument("--gate-verdicts", default=None)
        p.set_defaults(fn=cmd_run if name == "run" else cmd_run_once)
    args = parser.parse_args(argv)
    if getattr(args, "threads", 0) == 0:
        args.threads = _default_threads() if getattr(args, "pin", True) else 0
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
