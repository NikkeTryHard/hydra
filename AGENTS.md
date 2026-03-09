# AGENTS.md -- Hydra Codebase Guide

## Project Overview

Hydra is an open-source Riichi Mahjong AI targeting LuckyJ-level play (10+ dan).
100% Rust stack (game engine + training via Burn framework). hydra-core is BSL-1.1, hydra-engine is Apache-2.0.
Game engine (hydra-core) is built, training pipeline (hydra-train) is next.

Reference codebase `Mortal-Policy/` is AGPL -- **never derive code from it**, reference only.

### Repository Layout

```
research/           # Research docs organized into design/, infrastructure/, intel/, evidence/, comparisons/
docs/               # Documentation for implemented components
crates/             # Workspace crates
  hydra-core/       # Rust crate: game engine, encoder, simulator, safety, seeding
  hydra-engine/     # Vendored riichienv-core crate (Apache-2.0)
  hydra-train/      # Rust crate: Burn model, PPO training loop, self-play arena (planned)
scripts/            # Utility scripts (export, evaluate) (planned)
Mortal-Policy/      # Reference only (AGPL, git-ignored) -- DO NOT copy code from here
```

**Ignored directories** (not part of the project):
- `.opencode/` -- editor config
- `RiichiEnv/` -- upstream reference checkout (git-ignored)
- `Mortal-Policy/` -- AGPL reference (git-ignored)
- `.worktrees/` -- git worktrees for development

### Key Specs (read these before making changes)

| Doc | Governs |
|-----|---------|
| `research/design/HYDRA_FINAL.md` | Architecture (SSOT), 9 techniques, oracle pondering ExIt, 2000 GPU hours |
| `research/design/TRAINING.md` | 3-phase training pipeline, loss functions, hyperparameters |
| `research/design/TESTING.md` | Testing strategy, golden tests, property-based tests |
| `research/design/SEEDING.md` | RNG hierarchy, reproducibility |
| `research/infrastructure/INFRASTRUCTURE.md` | Rust stack, data pipeline, CI, hardware |
| `docs/GAME_ENGINE.md` | hydra-core architecture, modules, APIs |


---

## Build, Lint, and Test Commands

### Rust (hydra-core)

```bash
# Build
cargo build --release

# Lint -- treat warnings as errors
cargo clippy --all-targets -- -D warnings

# Run all tests
cargo test --release

# Run a single test by name
cargo test --release test_name_here

# Run a test module
cargo test --release module_name::

# Encoder golden regression tests (critical -- catches silent encoding drift)
cargo test --release encoder_golden_tests

# Benchmarks
cargo bench
```


---

## Architecture Quick Reference

- **Model**: SE-ResNet, 40 blocks, 256 channels, ~16.5M params
- **Input**: 85x34 tensor (62 base + 23 safety channels)
- **Output heads**: Policy (46 actions), Value (scalar), GRP (24-way), Tenpai (3), Danger (3x34), Opp-next (3x34), Score-pdf (64), Score-cdf (64) + oracle critic (training only)
- **Normalization**: GroupNorm(32) -- NOT BatchNorm
- **Activation**: Mish
- **Precision**: bf16 (no GradScaler needed)
- **Tile encoding**: 0-33 index (9m + 9p + 9s + 7 honors)
- **Action space**: 46 actions (Mortal-compatible). Riichi/kan use two-phase selection.
- **Inference**: Burn direct inference or CModule via burn-tch

---

## Code Style and Conventions

### Licensing -- CRITICAL

- **hydra-core** is BSL-1.1 licensed. **hydra-engine** is Apache-2.0 (vendored upstream).
- Dependencies must be MIT, Apache-2.0, or BSD compatible.
- **NEVER** copy, adapt, or derive from `Mortal-Policy/` (AGPL). Reference techniques only.
- **NEVER** add AGPL, GPL, or LGPL dependencies.

### Rust (hydra-core)

- **Edition**: 2024 (or latest stable)
- **Formatting**: `rustfmt` defaults. Run `cargo fmt` before committing.
- **Linting**: Zero warnings policy. `cargo clippy -- -D warnings` must pass.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for types/traits, `SCREAMING_SNAKE` for constants.
- **Error handling**: Use `anyhow::Result` for application-level errors. `thiserror` for library error enums.
- **No `unwrap()`** in library code. Use `?` or explicit error handling. `unwrap()` is acceptable in tests.
- **Module layout**: Flat under `src/` (see INFRASTRUCTURE.md module table). No deeply nested module trees.
- **Memory**: Pre-allocate buffers and reuse. Avoid per-turn allocations in hot paths (encoder, simulator).
- **Parallelism**: Use `rayon` for batch simulation.
- **Documentation**: Use `///` for public items and `//!` for module-level docs. Follow [RFC 1574](https://rust-lang.github.io/rfcs/1574-more-api-documentation-conventions.html) conventions:
  - First line: single-sentence summary (imperative mood: "Returns...", "Computes...").
  - Then a blank `///` line, then extended description if needed.
  - Use `# Examples`, `# Panics`, `# Errors`, `# Safety` sections where applicable.
  - All `pub` items (functions, structs, enums, traits, constants) must have `///` docs.
  - Module files must start with `//!` describing the module's purpose.
