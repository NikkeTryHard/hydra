#!/usr/bin/env bash
set -euo pipefail

if command -v sscache >/dev/null 2>&1; then
  exec sscache "$@"
fi

exec rustc "$@"
