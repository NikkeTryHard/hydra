# Coverage Reporting

Hydra uses `cargo-llvm-cov` for workspace-wide Rust coverage reporting. The report is meant to strengthen regression review, not replace correctness testing. For day-to-day regression checks, keep using `cargo nextest run --release` as the default fast path.

## Why Hydra Tracks Coverage

Hydra has several failure modes that can silently poison training rather than crash loudly: encoder drift, replay mismatches, state-transition bugs, and training-pipeline shape mismatches. Coverage helps answer a simple question after a change: did the tests actually execute the risky code paths we think they did?

That matters most for:

- `crates/hydra-core` encoder, simulator, seeding, and replay surfaces
- `crates/hydra-engine` legal-action, scoring, and state-machine logic
- `crates/hydra-train` batch shaping, model smoke paths, and supervision gates

## Prerequisites

Install the LLVM coverage helper once:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked
```

## Generate a Local Coverage Report

From the repo root:

```bash
./scripts/coverage.sh
```

By default the script now uses the workspace `coverage` Cargo profile through
nextest's `--cargo-profile` passthrough, which keeps coverage runs cheaper than
Hydra's shipping `release` profile while staying compatible with `cargo llvm-cov nextest`.

For fast inner-loop coverage while you are only touching a few modules, keep the
same script but scope the nextest run and skip heavy artifacts:

```bash
HYDRA_COVERAGE_FAST=1 \
HYDRA_COVERAGE_HTML=0 \
HYDRA_COVERAGE_LCOV=0 \
HYDRA_COVERAGE_NEXTTEST_FILTERS='-p hydra-core arena robust_opponent bridge' \
./scripts/coverage.sh
```

The script now prints per-step timings and total runtime so you can see whether
time is going into test execution, HTML generation, or LCOV export.

By default the script also pins both Cargo build jobs and nextest runtime test
threads to 16. Override either with `HYDRA_BUILD_JOBS` or `HYDRA_TEST_THREADS`
if you need a different build-vs-test scheduling balance.

Fast mode skips summary, HTML, and LCOV generation entirely, which is useful when
you only care about collecting fresh coverage data and timing the hot test stage.
For a fresh text summary after a fast run, rerun with `HYDRA_COVERAGE_FAST=0`.

To override the coverage build profile:

```bash
HYDRA_COVERAGE_PROFILE=release ./scripts/coverage.sh
```

Artifacts are written under `target/coverage/` by default:

- `target/coverage/html/index.html` — browsable HTML report
- `target/coverage/lcov.info` — LCOV export for tooling
- `target/coverage/summary.txt` — text summary for quick review

To write reports somewhere else:

```bash
HYDRA_COVERAGE_DIR=/absolute/path ./scripts/coverage.sh
```

## Coverage Artifacts

The local coverage run writes three outputs under `target/coverage/`:

- the HTML report directory
- the LCOV file
- the text summary

## How to Use Coverage Well

Coverage is only useful when it is interpreted like an engineering safety signal instead of a vanity number.

- High total coverage does not prove the engine is correct.
- Low coverage in critical paths is a real regression risk even if the total percentage looks fine.
- Review per-crate and per-module gaps first, especially around encoder channels, replay roundtrip behavior, scoring, legal action generation, and training-label gating.
- Treat coverage changes as suspicious when they coincide with changes to `hydra-core` encoder/runtime logic or `hydra-engine` state transitions.

Hydra is safest when all of these are green together:

```bash
cargo clippy --all-targets -- -D warnings
cargo nextest run --release
./scripts/coverage.sh
```
