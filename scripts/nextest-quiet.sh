#!/usr/bin/env bash
set -u

output_file="target/nextest-quiet-output.log"
mkdir -p "${output_file%/*}"

set +e
cargo nextest "$@" >"$output_file" 2>&1
status=$?
set -e

if [ "$status" -ne 0 ]; then
    printf 'cargo nextest failed with exit code %s\n' "$status"
    printf '%s\n' '----- captured output -----'
    python - "$output_file" <<'PY'
from pathlib import Path
import sys
sys.stdout.write(Path(sys.argv[1]).read_text())
PY
fi

exit "$status"
