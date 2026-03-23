# Kaggle-compatible Hydra train artifact

This path exists for one specific problem: when Kaggle cannot compile Hydra locally and
cannot run a host-built `train` binary because of newer `glibc` / `libstdc++` ABI
requirements.

## Strategy

Build `train` in an older Ubuntu 22.04 / glibc 2.35 userspace while still sourcing
libtorch from Python PyTorch via `LIBTORCH_USE_PYTORCH=1`.

This Kaggle-specific path intentionally targets Kaggle's older ABI floor, but it still
has to stay compatible with Hydra's current `tch` / `torch-sys` stack. That means the
builder keeps PyTorch `2.9.0+cu128`, because Hydra's current `tch 0.22.0` build expects
that version family.

This keeps the Hydra train/runtime contract the same, but lowers the ABI floor of the
produced binary so it has a chance to run on Kaggle.

## Build

From the repo root:

```bash
bash scripts/build-kaggle-compatible-artifact.sh
```

This builds `docker/train/Dockerfile.kaggle-compat` and exports a runtime artifact under:

```text
dist/kaggle-compat/
```

## What gets exported

- `bin/train`
- `bin/mjai_audit`
- `bin/recompress`
- `lib/` with only the exact shared-library closure that `bin/train` resolves from `ldd`
- `runtime-manifest.json` as the producer-owned runtime contract for the exported train runtime
- `lib-manifest.tsv` with per-library size and sha256 for the shipped `lib/` closure
- `lib-summary.txt` with total runtime payload size and the builder search roots used to resolve it
- `ldd-train.txt`
- `ldd-train-summary.txt`
- `abi-symbols.txt`

The important bit is that `dist/kaggle-compat/lib/` is no longer a broad libtorch seed
dump. The exported directory is required to match the actual `ldd` closure of the shipped
`bin/train` binary. If `ldd` says a library is not used by `train`, it should not ride along
in the final compat artifact unless that exception is documented and encoded by the producer.

The builder installs PyTorch with `uv` instead of plain `pip`, matching the repo's tool
preference for Python package management on new workflows.

It also installs `protobuf-compiler`, because current Hydra train dependencies include
`tboard` build steps that require `protoc` during compilation.

## Validation before Kaggle upload

You should inspect:

- `dist/kaggle-compat/runtime-manifest.json`
- `dist/kaggle-compat/lib-manifest.tsv`
- `dist/kaggle-compat/ldd-train.txt`
- `dist/kaggle-compat/ldd-train-summary.txt`
- `dist/kaggle-compat/abi-symbols.txt`

The key goal is that the resulting `train` binary does **not** require a newer glibc or
libstdc++ ABI than Kaggle exposes.

`runtime-manifest.json` is the source of truth the notebook/bundle flow should validate
against before reusing persisted Kaggle working directories. That lets the producer decide
exactly which runtime payload is valid instead of relying on shallow sentinel files.

If the ABI floor is still too new, the builder image/toolchain must be moved even older.

## Important note

This is the artifact-production path only. It does **not** automatically update the
Kaggle notebook bundle. If the artifact validates, wire `bin/train`, the matching exact
`lib/` closure, and the manifest metadata into the Kaggle bundle/launcher path.
