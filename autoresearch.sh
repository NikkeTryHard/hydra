#!/usr/bin/env bash
# Autoresearch harness: sustained-GPU training (production streaming path).
# Workload: configs/training/probe-recompiles.yaml, 150 supervised updates,
# fixed seed, Tenhou houou 2024 slice, full production path (prefetch workers,
# expand pool, pinned-ring H2D, compiled model, GC freeze). Deterministic work:
# same seed + same corpus order every run; update 0 absorbs inductor compile.
# Primary: mean SM utilization % inside the exact run window (higher = fewer gaps).
# Anti-gaming (ALL must pass or NO metric is emitted):
#  1. CUDA required (nvidia-smi present, dmon yields windowed samples; CPU lane fails).
#  2. HYDRA2_DATA_ROOT must hold tenhou-houou-mjai-2024 (fail closed, no tiny-corpus shortcut).
#  3. exactly 150 microbatch rows, global_update 0..149, from this run's dir (fail closed, no fallback).
#  4. loss finite on all 150 metrics.jsonl rows; stdout must read `train: updates 0->150`.
#  5. dual-clock: windowed 1Hz SM sample count vs external elapsed agree within 8s.
#  6. work pins emitted (config digest, corpus file-list hash, train/val game counts).
set -u
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="configs/training/probe-recompiles.yaml"
EXP_UPDATES=150
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
fail() { echo "HARNESS FAIL: $1" >&2; exit 1; }

cd "${ROOT}" || exit 1
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi missing (GPU harness)"
command -v pixi >/dev/null 2>&1 || fail "pixi missing"
[ -f "${CFG}" ] || fail "probe config missing: ${CFG}"
[ -n "${HYDRA2_DATA_ROOT:-}" ] || fail "HYDRA2_DATA_ROOT unset (mount the corpus)"
[ -d "${HYDRA2_DATA_ROOT}/tenhou-houou-mjai-2024" ] \
    || fail "tenhou-houou-mjai-2024 missing under HYDRA2_DATA_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HYDRA2_ARTIFACT_ROOT="${SCRATCH}/artifacts"
[ "$(ls -U "${HYDRA2_DATA_ROOT}/tenhou-houou-mjai-2024" | grep -c '\.zst$')" -gt 1000 ] \
    || fail "corpus slice looks truncated (need the full 2024 slice)"

# --- timed production run + timestamped SM sampler ---
LOAD_BEFORE="$(cat /proc/loadavg)"
nvidia-smi dmon -s u -o DT -d 1 -f "${SCRATCH}/sm.log" >/dev/null 2>&1 &
SM_PID=$!
cleanup_sm() { kill "${SM_PID}" 2>/dev/null || true; wait "${SM_PID}" 2>/dev/null || true; }
trap 'cleanup_sm; rm -rf "${SCRATCH}"' EXIT
sleep 2  # let sampler settle before training starts
EXT_START=$(date +%s%N)
WIN_START="$(date +"%Y%m%d %H:%M:%S")"
pixi run hydra2 train "${CFG}" >"${SCRATCH}/train.log" 2>&1 \
    || { tail -20 "${SCRATCH}/train.log" >&2; fail "train failed"; }
EXT_END=$(date +%s%N)
WIN_END="$(date +"%Y%m%d %H:%M:%S")"
LOAD_AFTER="$(cat /proc/loadavg)"
cleanup_sm
trap 'rm -rf "${SCRATCH}"' EXIT

RUN_DIR="$(grep -oP 'run_dir\S* \K\S+' "${SCRATCH}/train.log" | head -1)"
[ -n "${RUN_DIR}" ] || fail "run_dir missing from train output (no stale-run fallback)"
[ -d "${RUN_DIR}" ] || fail "run dir not found"
FEED="${RUN_DIR}/logs/feed-telemetry.jsonl"
METRICS_JSONL="${RUN_DIR}/logs/metrics.jsonl"
[ -f "${FEED}" ] || fail "feed-telemetry missing: ${FEED}"
[ -f "${METRICS_JSONL}" ] || fail "metrics.jsonl missing: ${METRICS_JSONL}"
grep -q "train: updates 0->150" "${SCRATCH}/train.log" || fail "updates != 0->150"
CONFIG_DIGEST="$(grep -oP 'digest:\s*\K\S+' "${SCRATCH}/train.log" | head -1)"
[ -n "${CONFIG_DIGEST}" ] || fail "config digest missing from train output"

# --- metric extraction (python stdlib only) ---
export HARNESS_FEED="${FEED}" HARNESS_MJSONL="${METRICS_JSONL}" HARNESS_SM="${SCRATCH}/sm.log"
export HARNESS_OUT="${SCRATCH}/metrics.txt" HARNESS_CORPUS="${HYDRA2_DATA_ROOT}/tenhou-houou-mjai-2024"
export HARNESS_EXT_S="$(( (EXT_END - EXT_START) / 1000000000 ))"
export HARNESS_WIN_START="${WIN_START}" HARNESS_WIN_END="${WIN_END}"
pixi run python - >"${SCRATCH}/extract.log" 2>&1 <<'PYEOF' || { tail -20 "${SCRATCH}/extract.log" >&2; fail "extract failed"; }
import hashlib, json, math, os, re, statistics
feed, mj, smpath, outp = (os.environ["HARNESS_FEED"], os.environ["HARNESS_MJSONL"],
                          os.environ["HARNESS_SM"], os.environ["HARNESS_OUT"])
rows = []
with open(feed) as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
mb = [r for r in rows if r.get("kind") == "microbatch"]
assert len(mb) == 150, f"microbatch rows {len(mb)} != 150"
ups = sorted(r["global_update"] for r in mb)
assert ups == list(range(150)), "global_update not exactly 0..149"
loss_rows = []
with open(mj) as fh:
    for line in fh:
        line = line.strip()
        if line:
            loss_rows.append(json.loads(line))
assert len(loss_rows) == 150, f"metrics rows {len(loss_rows)} != 150"
for r in loss_rows:
    assert math.isfinite(float(r["total"])), f"non-finite loss at update {r.get('global_update')}"
comp = sorted(r["compute_ms"] for r in mb)
fetch = sorted(r["fetch_decode_ms"] for r in mb)
queue = sorted(r["queue_wait_ms"] for r in mb)
w0, w1 = os.environ["HARNESS_WIN_START"], os.environ["HARNESS_WIN_END"]
sm_col = None
sm = []
with open(smpath) as fh:
    for line in fh:
        toks = line.replace("#", " ").split()
        if len(toks) < 4:
            continue
        if toks[0].isalpha():
            if "sm" in [t.lower() for t in toks]:
                sm_col = [t.lower() for t in toks].index("sm")
            continue
        if sm_col is None or not re.fullmatch(r"\d{8}", toks[0]):
            continue
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", toks[1]):
            continue
        stamp = toks[0] + " " + toks[1]
        if not (w0 <= stamp <= w1):
            continue
        try:
            sm.append(float(toks[sm_col]))
        except (IndexError, ValueError):
            continue
assert sm_col is not None, "dmon header with sm column not found"
assert len(sm) > 60, f"too few windowed SM samples: {len(sm)}"
ext_s = int(os.environ["HARNESS_EXT_S"])
assert abs(len(sm) - ext_s) <= 8, f"sampler clock drift: {len(sm)} samples vs {ext_s}s"
names = sorted(os.listdir(os.environ["HARNESS_CORPUS"]))
names = [n for n in names if n.endswith(".zst")]
assert len(names) > 1000, "corpus file list truncated"
h = hashlib.sha256()
for n in names:
    h.update(f"{n}:{os.path.getsize(os.path.join(os.environ['HARNESS_CORPUS'], n))}\n".encode())
with open(outp, "w") as fh:
    fh.write(f"sm_mean={statistics.fmean(sm):.2f}\n")
    fh.write(f"sm_p50={statistics.median(sm):.2f}\n")
    fh.write(f"sm_n={len(sm)}\n")
    fh.write(f"compute_worst={comp[-1]:.1f}\n")
    fh.write(f"compute_p99={comp[int(0.99 * (len(comp) - 1))]:.1f}\n")
    fh.write(f"fetch_worst={fetch[-1]:.1f}\n")
    fh.write(f"queue_worst={queue[-1]:.1f}\n")
    fh.write(f"corpus_hash={h.hexdigest()[:16]}\n")
    fh.write(f"corpus_files={len(names)}\n")
PYEOF
# shellcheck disable=SC1090
source "${SCRATCH}/metrics.txt"
[ -n "${sm_mean:-}" ] && [ -n "${corpus_hash:-}" ] || fail "metric parse failed"

# --- persist audit bundle outside scratch ---
BUNDLE="${HOME}/tmp/harness-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${BUNDLE}"
cp "${SCRATCH}/train.log" "${SCRATCH}/sm.log" "${SCRATCH}/metrics.txt" "${BUNDLE}/"
cp "${FEED}" "${METRICS_JSONL}" "${BUNDLE}/"
{
    echo "load_before: ${LOAD_BEFORE}"; echo "load_after: ${LOAD_AFTER}"
    echo "config_digest: ${CONFIG_DIGEST}"
} >"${BUNDLE}/provenance.txt"

EXT_MS="$(( (EXT_END - EXT_START) / 1000000 ))"
UPS="$(awk -v e="$EXT_MS" 'BEGIN {printf "%.3f", 150/(e/1000)}')"
echo "METRIC gpu_sm_util_pct=${sm_mean}"
echo "METRIC sm_p50_pct=${sm_p50}"
echo "METRIC updates_per_sec=${UPS}"
echo "METRIC compute_worst_ms=${compute_worst}"
echo "METRIC compute_p99_ms=${compute_p99}"
echo "METRIC fetch_worst_ms=${fetch_worst}"
echo "METRIC queue_worst_ms=${queue_worst}"
echo "METRIC wall_seconds=$(awk -v e="$EXT_MS" 'BEGIN {printf "%.3f", e/1000}')"
echo "METRIC corpus_hash=${corpus_hash}"
echo "METRIC corpus_files=${corpus_files}"
echo "METRIC config_digest=${CONFIG_DIGEST}"
echo "BUNDLE ${BUNDLE}"
