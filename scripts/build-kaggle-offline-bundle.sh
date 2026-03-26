#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
TMP_BASE="/home/nikketryhard/tmp"
mkdir -p "${TMP_BASE}"
STAGE_ROOT="${TMP_BASE}/hydra-tmp-kaggle-bundle"
STAGE_DIR="${STAGE_ROOT}/hydra"
OUT_DIR="${ROOT_DIR}/notebooks/offline-bundle"
BIN_PATH="${OUT_DIR}/hydra-kaggle-offline-bundle.bin"
LEGACY_ZIP_PATH="${OUT_DIR}/hydra-kaggle-offline-bundle.zip"
COMPAT_DIR="${ROOT_DIR}/dist/kaggle-compat"
NOTEBOOK_PATH="${ROOT_DIR}/notebooks/hydra_baseline_training.ipynb"
README_PATH="${STAGE_DIR}/KAGGLE_OFFLINE_README.md"
required_paths=(
  "${NOTEBOOK_PATH}"
  "${COMPAT_DIR}/bin/train"
  "${COMPAT_DIR}/lib"
  "${COMPAT_DIR}/lib-summary.txt"
  "${COMPAT_DIR}/lib-manifest.tsv"
  "${COMPAT_DIR}/runtime-manifest.json"
  "${COMPAT_DIR}/ldd-train.txt"
  "${COMPAT_DIR}/ldd-train-summary.txt"
  "${COMPAT_DIR}/abi-symbols.txt"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'Refusing to build offline bundle: missing required path %s\n' "${required_path}" >&2
    exit 1
  fi
done

if grep -q 'not found' "${COMPAT_DIR}/ldd-train.txt"; then
  printf 'Refusing to build offline bundle: unresolved runtime dependencies remain in %s\n' "${COMPAT_DIR}/ldd-train.txt" >&2
  exit 1
fi

rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_DIR}/notebooks" "${STAGE_DIR}/dist/kaggle-compat/bin"

cp "${NOTEBOOK_PATH}" "${STAGE_DIR}/notebooks/"
cp "${COMPAT_DIR}/bin/train" "${STAGE_DIR}/dist/kaggle-compat/bin/"
cp -a "${COMPAT_DIR}/lib" "${STAGE_DIR}/dist/kaggle-compat/lib"
cp \
  "${COMPAT_DIR}/runtime-manifest.json" \
  "${COMPAT_DIR}/ldd-train.txt" \
  "${COMPAT_DIR}/ldd-train-summary.txt" \
  "${COMPAT_DIR}/lib-summary.txt" \
  "${COMPAT_DIR}/lib-manifest.tsv" \
  "${COMPAT_DIR}/abi-symbols.txt" \
  "${STAGE_DIR}/dist/kaggle-compat/"

cat > "${README_PATH}" <<'EOF'
Minimal Hydra Kaggle offline bundle

Bundle shape:
- notebooks/hydra_baseline_training.ipynb
- dist/kaggle-compat/bin/train
- dist/kaggle-compat/lib/
- dist/kaggle-compat/runtime-manifest.json
- dist/kaggle-compat/ldd-train.txt
- dist/kaggle-compat/ldd-train-summary.txt
- dist/kaggle-compat/lib-summary.txt
- dist/kaggle-compat/lib-manifest.tsv
- dist/kaggle-compat/abi-symbols.txt
- bundle-manifest.json

Runtime strategy:
- prefer dist/kaggle-compat/bin/train in offline bundle mode
- prepend dist/kaggle-compat/lib to LD_LIBRARY_PATH before launching train
- validate dist/kaggle-compat/runtime-manifest.json before reusing a persisted Kaggle checkout
- validate bundle-manifest.json before reusing a persisted extracted bundle payload
- if manifest mismatch or required files are missing, wipe the stale extracted/runtime checkout and refresh from the current payload
- use ldd-train.txt and ldd-train-summary.txt to confirm the shipped lib/ directory matches the train runtime closure
EOF

ROOT_DIR_ENV="${ROOT_DIR}" python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['ROOT_DIR_ENV'])
stage_dir = Path('/home/nikketryhard/tmp/hydra-tmp-kaggle-bundle/hydra')
manifest_path = stage_dir / 'bundle-manifest.json'

included_files = [
    stage_dir / 'notebooks' / 'hydra_baseline_training.ipynb',
    stage_dir / 'KAGGLE_OFFLINE_README.md',
    stage_dir / 'dist' / 'kaggle-compat' / 'bin' / 'train',
    stage_dir / 'dist' / 'kaggle-compat' / 'runtime-manifest.json',
    stage_dir / 'dist' / 'kaggle-compat' / 'ldd-train.txt',
    stage_dir / 'dist' / 'kaggle-compat' / 'ldd-train-summary.txt',
    stage_dir / 'dist' / 'kaggle-compat' / 'lib-summary.txt',
    stage_dir / 'dist' / 'kaggle-compat' / 'lib-manifest.tsv',
    stage_dir / 'dist' / 'kaggle-compat' / 'abi-symbols.txt',
]
included_files.extend(
    sorted(
        path
        for path in (stage_dir / 'dist' / 'kaggle-compat' / 'lib').iterdir()
        if path.is_file()
    )
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for path in included_files:
    if not path.exists() or not path.is_file():
        raise SystemExit(f'Missing staged bundle file during manifest generation: {path}')
    files.append(
        {
            'path': path.relative_to(stage_dir).as_posix(),
            'size_bytes': path.stat().st_size,
            'sha256': sha256_file(path),
        }
    )

runtime_manifest = json.loads((stage_dir / 'dist' / 'kaggle-compat' / 'runtime-manifest.json').read_text(encoding='utf-8'))
notebook_entry = next(entry for entry in files if entry['path'] == 'notebooks/hydra_baseline_training.ipynb')

bundle_manifest = {
    'schema_version': 1,
    'bundle_kind': 'hydra-kaggle-offline-runtime-bundle',
    'root_dir': 'hydra',
    'required_files': [entry['path'] for entry in files],
    'files': files,
    'freshness_contract': {
        'notebook': notebook_entry,
        'runtime_manifest': {
            'path': 'dist/kaggle-compat/runtime-manifest.json',
            'sha256': next(entry['sha256'] for entry in files if entry['path'] == 'dist/kaggle-compat/runtime-manifest.json'),
            'size_bytes': next(entry['size_bytes'] for entry in files if entry['path'] == 'dist/kaggle-compat/runtime-manifest.json'),
            'binary_sha256': runtime_manifest['binary']['sha256'],
            'runtime_lib_count': len(runtime_manifest['required_runtime_libraries']),
        },
    },
}

manifest_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

mkdir -p "${OUT_DIR}"
rm -f "${BIN_PATH}" "${LEGACY_ZIP_PATH}"

ROOT_DIR_ENV="${ROOT_DIR}" python3 - <<'PY'
import os
from pathlib import Path
import zipfile

root_dir = Path(os.environ['ROOT_DIR_ENV'])
root = Path('/home/nikketryhard/tmp/hydra-tmp-kaggle-bundle')
out = root_dir / 'notebooks' / 'offline-bundle' / 'hydra-kaggle-offline-bundle.bin'

with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            arcname = rel.rstrip('/') + '/'
            info = zipfile.ZipInfo(arcname)
            info.date_time = (2026, 3, 21, 12, 0, 0)
            info.external_attr = (0o755 << 16) | 0x10
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, b'')
        else:
            info = zipfile.ZipInfo(rel)
            info.date_time = (2026, 3, 21, 12, 0, 0)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
PY

cat <<EOF
Kaggle offline bundle ready:
  ${BIN_PATH}

Bundle contents are runtime-only:
  notebooks/hydra_baseline_training.ipynb
  dist/kaggle-compat/bin/train
  dist/kaggle-compat/lib/
  dist/kaggle-compat/runtime-manifest.json
  dist/kaggle-compat/ldd-train.txt
  dist/kaggle-compat/ldd-train-summary.txt
  dist/kaggle-compat/lib-summary.txt
  dist/kaggle-compat/lib-manifest.tsv
  dist/kaggle-compat/abi-symbols.txt
  bundle-manifest.json
  KAGGLE_OFFLINE_README.md
EOF
