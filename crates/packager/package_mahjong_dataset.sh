#!/usr/bin/env bash
set -euo pipefail

# Portable PROJECT_ROOT via BASH_SOURCE (BashFAQ 028) with 12-factor env override.
# Evidence: BashFAQ 028 https://mywiki.wooledge.org/BashFAQ/028
#           12-factor config https://12factor.net/config
PROJECT_ROOT="${HYDRA2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly PROJECT_ROOT
readonly PACKAGER_DIR="${PROJECT_ROOT}/tools/mjai-dataset-packager"
readonly SOURCE_DIR="${1:?usage: package_mahjong_dataset.sh SOURCE_DIR [DEST_DIR]}"
# DEST_DIR configurable via 2nd arg or HYDRA2_DATASET_DEST env (12-factor); default repo-local cache.
# https://12factor.net/config
readonly DEST_DIR="${2:-${HYDRA2_DATASET_DEST:-${PROJECT_ROOT}/.cache/mahjong_dataset}}"
readonly BINARY="${PACKAGER_DIR}/target/release/mjai-dataset-packager"
readonly LOG_DIR="${DEST_DIR}/.packager-logs"
readonly BUILD_STAMP="${PACKAGER_DIR}/target/release/.mjai-packager-native.stamp"

[[ -d "${SOURCE_DIR}" ]] || { echo "source directory missing: ${SOURCE_DIR}" >&2; exit 1; }
# Portable mountpoint guard: do not hard-exit; warn and continue.
# Guarded with `command -v mountpoint` for portability (macOS/BusyBox lack mountpoint).
if command -v mountpoint >/dev/null 2>&1; then
    if ! mountpoint -q "$(dirname "${DEST_DIR}")" 2>/dev/null; then
        echo "warning: destination parent $(dirname "${DEST_DIR}") is not a mountpoint, continuing" >&2
    fi
else
    if [[ ! -d "$(dirname "${DEST_DIR}")" ]]; then
        echo "warning: destination parent $(dirname "${DEST_DIR}") does not exist, will attempt to create" >&2
    fi
fi
mkdir -p "${DEST_DIR}" "${LOG_DIR}"
[[ -w "${DEST_DIR}" ]] || { echo "destination is not writable: ${DEST_DIR}" >&2; exit 1; }

cd "${PACKAGER_DIR}"
if [[ ! -x "${BINARY}" || ! -f "${BUILD_STAMP}" ]] || \
   [[ Cargo.toml -nt "${BUILD_STAMP}" || Cargo.lock -nt "${BUILD_STAMP}" ]] || \
   [[ -n "$(find src -type f -newer "${BUILD_STAMP}" -print -quit 2>/dev/null)" ]]; then
    # mold: use if available for faster linking; otherwise fall back to default linker.
    if command -v mold >/dev/null 2>&1; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-fuse-ld=mold"
    fi
    if [[ "${NATIVE:-0}" == "1" ]]; then
        echo "Building native optimized packager (NATIVE=1, target-cpu=native)..."
        # rustc target-cpu https://doc.rust-lang.org/rustc/codegen-options/index.html#target-cpu
        # NATIVE=1 opts into -C target-cpu=native; default is portable generic.
        cargo rustc --release --locked -- -C target-cpu=native
    else
        echo "Building packager (portable generic target)..."
        cargo build --release --locked
    fi
    touch "${BUILD_STAMP}"
else
    echo "Using current release binary: ${BINARY}"
fi

readonly RUN_LOG="${LOG_DIR}/packager-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Preflighting ${SOURCE_DIR} -> ${DEST_DIR}" | tee -a "${RUN_LOG}"
"${BINARY}" preflight "${SOURCE_DIR}" "${DEST_DIR}" 2>&1 | tee -a "${RUN_LOG}"

available_bytes=$(df --output=avail -B1 "${DEST_DIR}" | awk 'NR==2 {print $1}')
[[ "${available_bytes}" =~ ^[0-9]+$ ]] || { echo "could not determine destination free capacity" >&2; exit 1; }
echo "Destination free bytes after preflight: ${available_bytes}" | tee -a "${RUN_LOG}"
echo "Starting resumable conversion with 16 workers and 4 GiB payload cap" | tee -a "${RUN_LOG}"
"${BINARY}" convert "${SOURCE_DIR}" "${DEST_DIR}" \
    --threads 16 --memory-limit-bytes 4294967296 2>&1 | tee -a "${RUN_LOG}"
