#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUTPUT_DIR="${HYDRA_COVERAGE_DIR:-$ROOT_DIR/target/coverage}"
HTML_DIR="$OUTPUT_DIR/html"
LCOV_PATH="$OUTPUT_DIR/lcov.info"
SUMMARY_PATH="$OUTPUT_DIR/summary.txt"
SUMMARY_JSON_PATH="$OUTPUT_DIR/summary.json"
RUN_LOG_PATH="$OUTPUT_DIR/run.log"
CACHE_STATS_BEFORE_PATH="$OUTPUT_DIR/cache-before.txt"
CACHE_STATS_AFTER_PATH="$OUTPUT_DIR/cache-after.txt"
TIMINGS_PATH="$OUTPUT_DIR/timings.txt"
QUIET_MODE="${HYDRA_COVERAGE_QUIET:-1}"
CLEAN_MODE="${HYDRA_COVERAGE_CLEAN:-0}"
BUILD_JOBS="${HYDRA_BUILD_JOBS:-16}"
TEST_THREADS="${HYDRA_TEST_THREADS:-$BUILD_JOBS}"
GENERATE_HTML="${HYDRA_COVERAGE_HTML:-0}"
GENERATE_LCOV="${HYDRA_COVERAGE_LCOV:-0}"
NEXTTEST_FILTERS="${HYDRA_COVERAGE_NEXTTEST_FILTERS:-}"
PROFILE_NAME="${HYDRA_COVERAGE_PROFILE:-coverage}"
FAST_MODE="${HYDRA_COVERAGE_FAST:-0}"
START_TS="$(date +%s)"

declare -a STAGE_NAMES=()
declare -a STAGE_DURATIONS=()

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  printf 'error: cargo-llvm-cov is required. Install system package first (`pacman -S cargo-llvm-cov` on CachyOS/Arch), fallback = `cargo install cargo-llvm-cov --locked`.\n' >&2
  exit 1
fi

if ! command -v llvm-cov >/dev/null 2>&1 || ! command -v llvm-profdata >/dev/null 2>&1; then
  printf 'error: LLVM coverage tools are required. Install system `llvm` package first; rustup llvm-tools-preview only needed for explicit rustup toolchains.\n' >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$GENERATE_HTML" == "1" ]]; then
  mkdir -p "$HTML_DIR"
fi
: > "$RUN_LOG_PATH"

cd "$ROOT_DIR"

printf 'coverage run: out=%s profile=%s scope=%s jobs=%s tests=%s clean=%s fast=%s html=%s lcov=%s quiet=%s\n' \
  "$OUTPUT_DIR" \
  "$PROFILE_NAME" \
  "${NEXTTEST_FILTERS:-workspace}" \
  "$BUILD_JOBS" \
  "$TEST_THREADS" \
  "$CLEAN_MODE" \
  "$FAST_MODE" \
  "$GENERATE_HTML" \
  "$GENERATE_LCOV" \
  "$QUIET_MODE"

export CARGO_BUILD_JOBS="$BUILD_JOBS"

format_elapsed() {
  local elapsed="$1"
  local minutes=$((elapsed / 60))
  local seconds=$((elapsed % 60))
  printf '%02dm%02ds' "$minutes" "$seconds"
}

format_clock() {
  date '+%H:%M:%S'
}

record_stage() {
  local description="$1"
  local elapsed="$2"

  STAGE_NAMES+=("$description")
  STAGE_DURATIONS+=("$elapsed")
}

print_stage_start() {
  local description="$1"

  printf -- '-> %s [%s]\n' "$description" "running"
}

print_stage_done() {
  local description="$1"
  local elapsed="$2"

  printf '   done in %s\n' "$(format_elapsed "$elapsed")"
}

print_stage_skipped() {
  local description="$1"

  printf -- '-> %s [%s]\n' "$description" "skipped"
}

write_sccache_stats() {
  local output_path="$1"

  if ! command -v sccache >/dev/null 2>&1; then
    : > "$output_path"
    return
  fi

  if ! sccache --show-stats >"$output_path" 2>/dev/null; then
    printf 'unavailable\n' >"$output_path"
  fi
}

extract_sccache_metric() {
  local stats_path="$1"
  local label="$2"

  python3 - "$stats_path" "$label" <<'PY'
from pathlib import Path
import re
import sys

stats_path = Path(sys.argv[1])
label = sys.argv[2]
if not stats_path.exists():
    print("n/a")
    raise SystemExit(0)

pattern = re.compile(rf"^{re.escape(label)}\s+(.+?)\s*$")
for line in stats_path.read_text().splitlines():
    match = pattern.match(line)
    if match:
        print(match.group(1).strip())
        raise SystemExit(0)

print("n/a")
PY
}

print_sccache_summary() {
  local label="$1"
  local stats_path="$2"

  if ! command -v sccache >/dev/null 2>&1; then
    return
  fi

  local hits
  local misses
  local rate
  local requests
  hits="$(extract_sccache_metric "$stats_path" 'Cache hits')"
  misses="$(extract_sccache_metric "$stats_path" 'Cache misses')"
  rate="$(extract_sccache_metric "$stats_path" 'Cache hits rate')"
  requests="$(extract_sccache_metric "$stats_path" 'Compile requests')"

  printf 'cache %s: hits=%s misses=%s hit_rate=%s requests=%s\n' \
    "$label" "$hits" "$misses" "$rate" "$requests"
}

write_timings_artifact() {
  local count="${#STAGE_NAMES[@]}"
  local i

  : > "$TIMINGS_PATH"
  for ((i = 0; i < count; i++)); do
    printf '%s\t%s\n' "${STAGE_NAMES[i]}" "$(format_elapsed "${STAGE_DURATIONS[i]}")" >>"$TIMINGS_PATH"
  done
}

print_summary_snapshot() {
  if [[ ! -f "$SUMMARY_JSON_PATH" ]]; then
    return
  fi

  python3 - "$SUMMARY_JSON_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

totals = data.get("data", [{}])[0].get("totals", {})
files = data.get("data", [{}])[0].get("files", [])

def pct(block):
    count = block.get("count", 0) or 0
    covered = block.get("covered", 0) or 0
    if count == 0:
        return "-"
    return f"{covered / count * 100:.2f}%"

print("coverage totals:")
print(f"  regions={pct(totals.get('regions', {}))} lines={pct(totals.get('lines', {}))} functions={pct(totals.get('functions', {}))}")

ranked = []
for entry in files:
    summary = entry.get("summary", {})
    regions = summary.get("regions", {})
    count = regions.get("count", 0) or 0
    covered = regions.get("covered", 0) or 0
    if count <= 0:
        continue
    pct_value = covered / count * 100.0
    ranked.append((pct_value, entry.get("filename", "<unknown>")))

ranked.sort(key=lambda item: item[0])
if ranked:
    print("lowest region coverage:")
    for pct_value, filename in ranked[:5]:
        print(f"  {pct_value:6.2f}%  {filename}")
PY
}

run_logged_command() {
  if [[ "$QUIET_MODE" == "1" ]]; then
    "$@" >>"$RUN_LOG_PATH" 2>&1
  else
    "$@"
  fi
}

run_step() {
  local description="$1"
  shift
  local step_start
  local step_end
  local elapsed

  print_stage_start "$description"
  step_start="$(date +%s)"

  if ! run_logged_command "$@"; then
    printf '\nerror: coverage step failed while %s\n' "$description" >&2
    printf 'full log: %s\n' "$RUN_LOG_PATH" >&2
    printf '\nlast log lines:\n' >&2
    tail -n 200 "$RUN_LOG_PATH" >&2 || true
    exit 1
  fi

  step_end="$(date +%s)"
  elapsed=$((step_end - step_start))
  record_stage "$description" "$elapsed"
  print_stage_done "$description" "$elapsed"
}

run_summary_step() {
  local step_start
  local step_end
  local elapsed
  local description='writing text summary'

  print_stage_start "$description"
  step_start="$(date +%s)"

  if [[ "$QUIET_MODE" == "1" ]]; then
    if ! cargo llvm-cov report --profile "$PROFILE_NAME" --json --summary-only --output-path "$SUMMARY_JSON_PATH" >>"$RUN_LOG_PATH" 2>&1; then
      printf '\nerror: coverage step failed while %s\n' "$description" >&2
      printf 'full log: %s\n' "$RUN_LOG_PATH" >&2
      printf '\nlast log lines:\n' >&2
      tail -n 200 "$RUN_LOG_PATH" >&2 || true
      exit 1
    fi
  else
    cargo llvm-cov report --profile "$PROFILE_NAME" --json --summary-only --output-path "$SUMMARY_JSON_PATH"
  fi

  python3 - "$SUMMARY_JSON_PATH" "$SUMMARY_PATH" <<'PY'
import json
import sys

src, dest = sys.argv[1], sys.argv[2]
with open(src, 'r', encoding='utf-8') as f:
    data = json.load(f)

totals = data.get("data", [{}])[0].get("totals", {})
files = data.get("data", [{}])[0].get("files", [])

def pct(block):
    count = block.get("count", 0) or 0
    covered = block.get("covered", 0) or 0
    missed = count - covered
    percent = covered / count * 100 if count else 0.0
    return count, missed, percent

lines = []
lines.append("Coverage summary\n")
lines.append("================\n")

for label in ("regions", "lines", "functions"):
    count, missed, percent = pct(totals.get(label, {}))
    lines.append(f"total {label}: {percent:.2f}% ({count - missed}/{count}, missed {missed})\n")

ranked = []
for entry in files:
    regions = entry.get("summary", {}).get("regions", {})
    count = regions.get("count", 0) or 0
    covered = regions.get("covered", 0) or 0
    if not count:
        continue
    percent = covered / count * 100.0
    ranked.append((percent, entry.get("filename", "<unknown>"), covered, count))

ranked.sort(key=lambda item: item[0])
lines.append("\nLowest region coverage files\n")
lines.append("----------------------------\n")
for percent, filename, covered, count in ranked[:20]:
    lines.append(f"{percent:6.2f}%  {filename} ({covered}/{count})\n")

with open(dest, 'w', encoding='utf-8') as f:
    f.writelines(lines)
PY

  step_end="$(date +%s)"
  elapsed=$((step_end - step_start))
  record_stage "$description" "$elapsed"
  print_stage_done "$description" "$elapsed"
}

print_timing_summary() {
  local count="${#STAGE_NAMES[@]}"
  local i
  local j
  local tmp_name
  local tmp_duration
  local width=0
  local label

  if [[ "$count" -eq 0 ]]; then
    return
  fi

  for label in "${STAGE_NAMES[@]}"; do
    if (( ${#label} > width )); then
      width=${#label}
    fi
  done

  for ((i = 0; i < count; i++)); do
    for ((j = i + 1; j < count; j++)); do
      if (( STAGE_DURATIONS[j] > STAGE_DURATIONS[i] )); then
        tmp_duration="${STAGE_DURATIONS[i]}"
        STAGE_DURATIONS[i]="${STAGE_DURATIONS[j]}"
        STAGE_DURATIONS[j]="$tmp_duration"

        tmp_name="${STAGE_NAMES[i]}"
        STAGE_NAMES[i]="${STAGE_NAMES[j]}"
        STAGE_NAMES[j]="$tmp_name"
      fi
    done
  done

  write_timings_artifact

  printf '\nTiming recap:\n'
  local limit=$count
  if (( limit > 3 )); then
    limit=3
  fi
  for ((i = 0; i < limit; i++)); do
    printf "  %-*s %s\n" "$width" "${STAGE_NAMES[i]}" "$(format_elapsed "${STAGE_DURATIONS[i]}")"
  done

  if (( count > 0 )); then
    printf '\nBottleneck hint:\n'
    printf '  Slowest stage was "%s" at %s.\n' "${STAGE_NAMES[0]}" "$(format_elapsed "${STAGE_DURATIONS[0]}")"
  fi
}

if [[ "$CLEAN_MODE" == "1" ]]; then
  run_step 'cleaning previous coverage artifacts' cargo llvm-cov clean --workspace
else
  print_stage_skipped 'cleaning previous coverage artifacts'
fi

write_sccache_stats "$CACHE_STATS_BEFORE_PATH"
print_sccache_summary 'before' "$CACHE_STATS_BEFORE_PATH"

if [[ -n "$NEXTTEST_FILTERS" ]]; then
  nextest_filter_args=( $NEXTTEST_FILTERS )
  run_step 'running scoped coverage tests' cargo llvm-cov nextest --cargo-profile "$PROFILE_NAME" --build-jobs "$BUILD_JOBS" --test-threads "$TEST_THREADS" --no-report "${nextest_filter_args[@]}"
else
  run_step 'running workspace coverage tests' cargo llvm-cov nextest --cargo-profile "$PROFILE_NAME" --build-jobs "$BUILD_JOBS" --test-threads "$TEST_THREADS" --workspace --no-report
fi

if [[ "$FAST_MODE" == "1" ]]; then
  print_stage_skipped 'generating HTML report'
elif [[ "$GENERATE_HTML" == "1" ]]; then
  run_step 'generating HTML report' cargo llvm-cov report --profile "$PROFILE_NAME" --html --output-dir "$HTML_DIR"
else
  print_stage_skipped 'generating HTML report'
fi

if [[ "$FAST_MODE" == "1" ]]; then
  print_stage_skipped 'writing text summary'
else
  run_summary_step
fi

if [[ "$FAST_MODE" == "1" ]]; then
  print_stage_skipped 'writing LCOV report'
elif [[ "$GENERATE_LCOV" == "1" ]]; then
  run_step 'writing LCOV report' cargo llvm-cov report --profile "$PROFILE_NAME" --lcov --output-path "$LCOV_PATH"
else
  print_stage_skipped 'writing LCOV report'
fi

write_sccache_stats "$CACHE_STATS_AFTER_PATH"
print_sccache_summary 'after' "$CACHE_STATS_AFTER_PATH"

END_TS="$(date +%s)"

if [[ "$FAST_MODE" == "1" ]]; then
  printf '\nCoverage summary: skipped in fast mode\n'
else
  printf '\n'
  print_summary_snapshot
fi

printf '\nCoverage timings:\n'
printf '  Total:   %s\n' "$(format_elapsed "$((END_TS - START_TS))")"
print_timing_summary

slow_lines="$(grep 'SLOW \[>' "$RUN_LOG_PATH" | tail -n 5 || true)"
if [[ -n "$slow_lines" ]]; then
  printf '\nSlow test hints (latest 5):\n'
  printf '%s\n' "$slow_lines"
  printf '  more: %s\n' "$RUN_LOG_PATH"
fi

printf '\nCoverage artifacts:\n'
if [[ "$GENERATE_HTML" == "1" ]]; then
  printf '  HTML:    %s/index.html\n' "$HTML_DIR"
elif [[ "$FAST_MODE" == "1" ]]; then
  printf '  HTML:    skipped in fast mode\n'
else
  printf '  HTML:    skipped\n'
fi
if [[ "$GENERATE_LCOV" == "1" ]]; then
  printf '  LCOV:    %s\n' "$LCOV_PATH"
elif [[ "$FAST_MODE" == "1" ]]; then
  printf '  LCOV:    skipped in fast mode\n'
else
  printf '  LCOV:    skipped\n'
fi
if [[ "$FAST_MODE" == "1" ]]; then
  printf '  Summary: skipped in fast mode\n'
else
  printf '  Summary: %s\n' "$SUMMARY_PATH"
  printf '  Summary JSON: %s\n' "$SUMMARY_JSON_PATH"
fi
printf '  Timings: %s\n' "$TIMINGS_PATH"
if command -v sccache >/dev/null 2>&1; then
  printf '  Cache stats: %s , %s\n' "$CACHE_STATS_BEFORE_PATH" "$CACHE_STATS_AFTER_PATH"
fi
printf '  Run log: %s\n' "$RUN_LOG_PATH"
