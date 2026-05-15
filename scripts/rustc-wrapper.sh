#!/usr/bin/env bash
set -euo pipefail

# Cargo calls RUSTC_WRAPPER as: rustc-wrapper.sh <rustc> <rustc-args...>.
# Direct use is also supported: rustc-wrapper.sh --version behaves like rustc --version.

default_compiler="${RUSTC:-rustc}"

if (($# == 0)) || [[ "$1" == -* ]]; then
  compiler="$default_compiler"
else
  compiler="$1"
  shift
fi

if command -v sccache >/dev/null 2>&1; then
  exec sccache "$compiler" "$@"
fi

exec "$compiler" "$@"
