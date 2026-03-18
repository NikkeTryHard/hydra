#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUTPUT_DIR="${HYDRA_COVERAGE_DIR:-$ROOT_DIR/target/coverage}"
HTML_DIR="$OUTPUT_DIR/html"
LCOV_PATH="$OUTPUT_DIR/lcov.info"
SUMMARY_PATH="$OUTPUT_DIR/summary.txt"
RUN_LOG_PATH="$OUTPUT_DIR/run.log"
QUIET_MODE="${HYDRA_COVERAGE_QUIET:-1}"
CLEAN_MODE="${HYDRA_COVERAGE_CLEAN:-0}"

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  printf 'error: cargo-llvm-cov is required. Install it with `cargo install cargo-llvm-cov --locked`.\n' >&2
  exit 1
fi

if ! rustup component list --installed 2>/dev/null | grep -Eq '^llvm-tools($|-)|^llvm-tools-preview($|-)'; then
  printf 'error: LLVM tools are required. Install them with `rustup component add llvm-tools-preview`.\n' >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$HTML_DIR"
: > "$RUN_LOG_PATH"

cd "$ROOT_DIR"

printf 'Generating workspace coverage report in %s\n' "$OUTPUT_DIR"
printf '  quiet mode: %s (set HYDRA_COVERAGE_QUIET=0 for live cargo output)\n' "$QUIET_MODE"
printf '  clean mode: %s (set HYDRA_COVERAGE_CLEAN=1 for a full clean rebuild)\n' "$CLEAN_MODE"

run_step() {
  local description="$1"
  shift

  printf -- '-> %s\n' "$description"

  if [[ "$QUIET_MODE" == "1" ]]; then
    if ! "$@" >>"$RUN_LOG_PATH" 2>&1; then
      printf '\nerror: coverage step failed while %s\n' "$description" >&2
      printf 'full log: %s\n' "$RUN_LOG_PATH" >&2
      printf '\nlast log lines:\n' >&2
      tail -n 200 "$RUN_LOG_PATH" >&2 || true
      exit 1
    fi
  else
    "$@"
  fi
}

if [[ "$CLEAN_MODE" == "1" ]]; then
  run_step 'cleaning previous coverage artifacts' cargo llvm-cov clean --workspace
else
  printf -- '-> reusing existing build artifacts when possible\n'
fi

run_step 'running workspace coverage tests' cargo llvm-cov nextest --workspace --no-report
run_step 'generating HTML report' cargo llvm-cov report --html --output-dir "$HTML_DIR"

printf -- '-> writing text summary\n'
if [[ "$QUIET_MODE" == "1" ]]; then
  if ! cargo llvm-cov report --summary-only >"$SUMMARY_PATH" 2>>"$RUN_LOG_PATH"; then
    printf '\nerror: coverage step failed while writing text summary\n' >&2
    printf 'full log: %s\n' "$RUN_LOG_PATH" >&2
    printf '\nlast log lines:\n' >&2
    tail -n 200 "$RUN_LOG_PATH" >&2 || true
    exit 1
  fi
else
  cargo llvm-cov report --summary-only >"$SUMMARY_PATH"
fi

run_step 'writing LCOV report' cargo llvm-cov report --lcov --output-path "$LCOV_PATH"

printf '\nCoverage summary:\n'
cat "$SUMMARY_PATH"

printf '\nCoverage artifacts:\n'
printf '  HTML:    %s/index.html\n' "$HTML_DIR"
printf '  LCOV:    %s\n' "$LCOV_PATH"
printf '  Summary: %s\n' "$SUMMARY_PATH"
printf '  Run log: %s\n' "$RUN_LOG_PATH"
