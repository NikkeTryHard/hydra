#!/usr/bin/env bash
set -euo pipefail

status=0
pytest -q "$@" || status=$?
if [[ "$status" == "5" ]]; then
  printf 'no Python tests collected yet\n'
  exit 0
fi
exit "$status"
