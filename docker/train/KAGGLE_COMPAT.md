# Kaggle-compatible Hydra train artifact

Path exists for one problem: Kaggle may fail local Hydra compile and may not run host-built `train` due newer `glibc` / `libstdc++` ABI reqs.

## Strategy

Build `train` in older Ubuntu 22.04 / glibc 2.35 userspace while still sourcing libtorch from Python PyTorch via `LIBTORCH_USE_PYTORCH=1`.

This Kaggle-only path targets Kaggle older ABI floor, but must stay compatible with Hydra current `tch` / `torch-sys` stack. Therefore builder keeps PyTorch `2.9.0+cu128`, because Hydra current `tch 0.22.0` build expects that version family.

This keeps Hydra train/runtime contract same, but lowers produced binary ABI floor so Kaggle may run it.

## Build

From repo root:

```bash
bash scripts/build-kaggle-compatible-artifact.sh
```

This builds `docker/train/Dockerfile.kaggle-compat` and exports runtime artifact under:

```text
dist/kaggle-compat/
```

## What gets exported

- `bin/train`
- `bin/mjai_audit`
- `bin/recompress`
- `lib/` with only exact shared-library closure `bin/train` resolves from `ldd`
- `runtime-manifest.json` as producer-owned runtime contract for exported train runtime
- `lib-manifest.tsv` with per-library size and sha256 for shipped `lib/` closure
- `lib-summary.txt` with total runtime payload size and builder search roots used to resolve it
- `ldd-train.txt`
- `ldd-train-summary.txt`
- `abi-symbols.txt`

Key point: `dist/kaggle-compat/lib/` no longer broad libtorch seed dump. Exported dir must match actual `ldd` closure of shipped `bin/train` binary. If `ldd` says library unused by `train`, it should not ship in final compat artifact unless producer documents and encodes exception.

Builder installs PyTorch with `uv`, not plain `pip`, matching repo Python package-manager preference for new workflows.

It also installs `protobuf-compiler`, because current Hydra train deps include `tboard` build steps needing `protoc` during compilation.

## Validation before Kaggle upload

Inspect:

- `dist/kaggle-compat/runtime-manifest.json`
- `dist/kaggle-compat/lib-manifest.tsv`
- `dist/kaggle-compat/ldd-train.txt`
- `dist/kaggle-compat/ldd-train-summary.txt`
- `dist/kaggle-compat/abi-symbols.txt`

Main goal: resulting `train` binary must **not** require newer glibc or libstdc++ ABI than Kaggle exposes.

`runtime-manifest.json` = source of truth notebook/bundle flow should validate before reusing persisted Kaggle working dirs. This lets producer decide exact valid runtime payload instead of relying on shallow sentinel files.

If ABI floor still too new, move builder image/toolchain older.

## Important note

This is artifact-production path only. It does **not** auto-update Kaggle notebook bundle. If artifact validates, wire `bin/train`, matching exact `lib/` closure, and manifest metadata into Kaggle bundle/launcher path.