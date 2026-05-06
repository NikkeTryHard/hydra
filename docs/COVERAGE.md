# Coverage Reporting

Hydra use `cargo-llvm-cov` for workspace-wide Rust coverage reporting. Report strengthen regression review, not replace correctness testing. Day-to-day regression checks: keep `cargo nextest run --release` as default fast path.

## Why Hydra Tracks Coverage

Hydra has failure modes can silently poison training, not crash loud: encoder drift, replay mismatches, state-transition bugs, training-pipeline shape mismatches. Coverage answer simple question after change: did tests hit risky code paths we think they did?

Matter most for:

- `crates/hydra-core` encoder, simulator, seeding, replay surfaces
- `crates/hydra-engine` legal-action, scoring, state-machine logic
- `crates/hydra-train` batch shaping, model smoke paths, supervision gates

## Prerequisites

Install LLVM coverage helper once:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
```

## Generate a Local Coverage Report

From repo root:

```bash
./scripts/coverage.sh
```

By default script now uses workspace `coverage` Cargo profile through
nextest's `--cargo-profile` passthrough, which keeps coverage runs cheaper than
Hydra's shipping `release` profile while staying compatible with `cargo llvm-cov nextest`.

For fast inner-loop coverage while touching only few modules, keep same script
but scope nextest run and skip heavy artifacts:

```bash
HYDRA_COVERAGE_FAST=1 \
HYDRA_COVERAGE_HTML=0 \
HYDRA_COVERAGE_LCOV=0 \
HYDRA_COVERAGE_NEXTTEST_FILTERS='-p hydra-core arena robust_opponent bridge' \
./scripts/coverage.sh
```

Script now prints per-step timings and total runtime so you can see whether
time goes into test execution, HTML generation, or LCOV export.

By default script also pins both Cargo build jobs and nextest runtime test
threads to 16. Override either with `HYDRA_BUILD_JOBS` or `HYDRA_TEST_THREADS`
if you need different build-vs-test scheduling balance.

Fast mode skips summary, HTML, and LCOV generation entirely, useful when
you only care about collecting fresh coverage data and timing hot test stage.
For fresh text summary after fast run, rerun with `HYDRA_COVERAGE_FAST=0`.

To override coverage build profile:

```bash
HYDRA_COVERAGE_PROFILE=release ./scripts/coverage.sh
```

Artifacts write under `target/coverage/` by default:

- `target/coverage/html/index.html` — browsable HTML report
- `target/coverage/lcov.info` — LCOV export for tooling
- `target/coverage/summary.txt` — text summary for quick review

To write reports somewhere else:

```bash
HYDRA_COVERAGE_DIR=/absolute/path ./scripts/coverage.sh
```

## Coverage Artifacts

Local coverage run writes three outputs under `target/coverage/`:

- HTML report directory
- LCOV file
- text summary

## How to Use Coverage Well

Coverage useful only when treated like engineering safety signal, not vanity number.

- High total coverage does not prove engine correctness.
- Low coverage in critical paths = real regression risk even if total percentage looks fine.
- Review per-crate and per-module gaps first, especially around encoder channels, replay roundtrip behavior, scoring, legal action generation, and training-label gating.
- Treat coverage changes as suspicious when they coincide with changes to `hydra-core` encoder/runtime logic or `hydra-engine` state transitions.

Hydra safest when all of these are green together:

```bash
cargo clippy --all-targets -- -D warnings
cargo nextest run --release
./scripts/coverage.sh
```