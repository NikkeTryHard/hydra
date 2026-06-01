#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: scripts/quiet-gate.sh [--fast|--full]

Runs Hydra's quiet quality gate. Passing commands print nothing. Failing commands print
only their captured diagnostics so agents can fix the root cause.

  --fast  fmt-check + lint + default no-heavy Rust/Python tests (default)
  --full  fast gate + all-feature Rust test gate
USAGE
}

mode=fast
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) mode=fast ;;
    --full) mode=full ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

run_quiet() {
  local label="$1"
  shift
  local output_file="target/quiet-gate/${label//[^A-Za-z0-9_.-]/_}.log"
  mkdir -p "${output_file%/*}"

  set +e
  "$@" >"$output_file" 2>&1
  local status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    printf '[hydra gate] %s failed with exit code %s\n' "$label" "$status" >&2
    printf '%s\n' '----- captured output -----' >&2
    python - "$output_file" >&2 <<'PY'
from pathlib import Path
import sys
sys.stdout.write(Path(sys.argv[1]).read_text())
PY
    exit "$status"
  fi
}

run_quiet fmt-rust cargo fmt --all -- --check
run_quiet fmt-python ruff format --check python scripts/hydra_pytorch_oracle.py
run_quiet lint scripts/lint-check.sh --fast
run_quiet rust-tests scripts/nextest-quiet.sh run --workspace --all-targets --no-default-features --cargo-profile dev --cargo-quiet --no-fail-fast
run_quiet python-tests scripts/pytest-quiet.sh python/hydra_learner

if [[ "$mode" == "full" ]]; then
  run_quiet rust-tests-all-features scripts/nextest-quiet.sh run --workspace --all-targets --all-features --cargo-profile dev --cargo-quiet --no-fail-fast
fi
