#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

output_file="target/ruff-quiet-output.log"
mkdir -p "${output_file%/*}"

set +e
ruff "$@" >"$output_file" 2>&1
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  printf 'ruff failed with exit code %s\n' "$status" >&2
  printf '%s\n' '----- captured output -----' >&2
  python - "$output_file" >&2 <<'PY'
from pathlib import Path
import sys
sys.stdout.write(Path(sys.argv[1]).read_text())
PY
fi

exit "$status"
