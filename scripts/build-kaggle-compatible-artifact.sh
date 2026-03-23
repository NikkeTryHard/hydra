#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
ARTIFACT_DIR="${ROOT_DIR}/dist/kaggle-compat"

rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}"

docker buildx build \
  --file "${ROOT_DIR}/docker/train/Dockerfile.kaggle-compat" \
  --target artifact \
  --output "type=local,dest=${ARTIFACT_DIR}" \
  "${ROOT_DIR}"

if [[ ! -f "${ARTIFACT_DIR}/lib-summary.txt" || ! -f "${ARTIFACT_DIR}/lib-manifest.tsv" || ! -f "${ARTIFACT_DIR}/runtime-manifest.json" || ! -f "${ARTIFACT_DIR}/ldd-train-summary.txt" ]]; then
  printf 'Compat artifact export is missing expected manifest outputs in %s\n' "${ARTIFACT_DIR}" >&2
  exit 1
fi

cat <<EOF
Kaggle compatibility artifact exported to:
  ${ARTIFACT_DIR}

Contents:
  bin/train
  bin/mjai_audit
  bin/recompress
  lib/
  lib-summary.txt
  lib-manifest.tsv
  runtime-manifest.json
  ldd-train.txt
  ldd-train-summary.txt
  abi-symbols.txt

Runtime validation artifacts:
  ${ARTIFACT_DIR}/lib-summary.txt
  ${ARTIFACT_DIR}/lib-manifest.tsv
  ${ARTIFACT_DIR}/runtime-manifest.json
  ${ARTIFACT_DIR}/ldd-train.txt
  ${ARTIFACT_DIR}/ldd-train-summary.txt
  ${ARTIFACT_DIR}/abi-symbols.txt

Check runtime-manifest.json for the producer-owned runtime contract,
lib-manifest.tsv for the exact exported closure, and ldd-train.txt plus
ldd-train-summary.txt to confirm train resolves every required runtime
dependency from the artifact export.
EOF
