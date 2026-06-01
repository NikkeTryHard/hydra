#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

output_file="target/pyrefly-quiet-output.log"
mkdir -p "${output_file%/*}"

set +e
pyrefly check "$@" >"$output_file" 2>&1
status=$?
set -e

output="$(python - "$output_file" <<'PY'
from pathlib import Path
import sys
sys.stdout.write(Path(sys.argv[1]).read_text())
PY
)"

if [[ "$status" -ne 0 ]]; then
  printf 'pyrefly failed with exit code %s\n' "$status" >&2
  printf '%s\n' '----- captured output -----' >&2
  printf '%s' "$output" >&2
  exit "$status"
fi

if [[ "$output" == *" WARN "* || "$output" == *" warning"* || "$output" == *"Warning"* ]]; then
  printf 'pyrefly emitted warnings\n' >&2
  printf '%s\n' '----- captured output -----' >&2
  printf '%s' "$output" >&2
  exit 1
fi
