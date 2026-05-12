# AGENTS.md -- Hydra Root Guide

## Purpose

Hydra = open-source Riichi Mahjong AI. Target LuckyJ-level. Optimize strength first, not simplicity. If research shows stronger viable path, ship strength.

## Goal and active build doctrine

- `research/design/HYDRA_FINAL.md` defines long-term ceiling (`Max Hydra`).
- Hydra v1 = active first checkpoint.
- For impl work, choose next lane from `research/design/HYDRA_RECONCILIATION.md`, confirm shipped/staged state in `docs/CURRENT_STATUS.md`, confirm exact runtime contracts in `docs/GAME_ENGINE.md` plus current code.
- Do not treat `HYDRA_FINAL.md` alone as authorization to build every destination-facing feature now.

Crate ownership summary:
- `crates/hydra-engine/` — vendored upstream rules engine, Apache-2.0
- `crates/hydra-runtime-types/` — shared runtime type rails
- `crates/hydra-safety/` — runtime safety rails
- `crates/hydra-belief-search/` — belief-state search pieces
- `crates/hydra-encoder/` — observation encoder pieces
- `crates/hydra-core/` — first-party runtime, simulator, seeding, integration API
- `crates/hydra-data-core/` — pure training sample DTOs and scoring helpers
- `crates/hydra-replay-sidecar/` — pure replay sidecar JSONL contracts
- `crates/hydra-sample-cache/` — parsed sample cache storage
- `crates/hydra-replay-loader/` — replay ingest/loading pipeline
- `crates/hydra-bc-shards/` — behavior-cloning shard format/tools
- `crates/hydra-train-types/` — training scalar/coordination types
- `crates/hydra-model/` — Burn neural model components
- `crates/hydra-train-algo/` — pure training algorithms
- `crates/hydra-train-runtime/` — training CLI/config/preflight/probe contracts
- `crates/hydra-train-exec/` — training execution composition layer
- `crates/hydra-search-labels/` — search label generation
- `crates/hydra-selfplay/` — self-play coordination primitives
- `crates/hydra-train/` — first-party Burn training binaries and user-facing stack

Hard source boundary:
- `Mortal-Policy/` is AGPL. Never copy, adapt, derive code from it.

Ignored non-project dirs: `.opencode/`, `RiichiEnv/`, `Mortal-Policy/`, `.worktrees/`.

## AGENTS scope

This root `AGENTS.md` = only tracked AGENTS guide in repo tree.

## Authority and routing

Do not treat all docs equal. For doctrine/planning, follow order below and stop when enough truth found:

1. `README.md` for repo entry routing
2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` as epistemic root for research conclusions
3. Derived archive views only as helpers; if drift, JSONL wins
4. Promoted doctrine: `research/design/HYDRA_FINAL.md` and `research/design/HYDRA_RECONCILIATION.md`
5. Promoted current-state/runtime surfaces: `docs/CURRENT_STATUS.md`, `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`
6. Current code when docs drift

Archive doctrine rules:
- `ARCHIVE_CANONICAL_CLAIMS.jsonl` = source ledger for research conclusions.
- `ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` and `ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` = derived views.
- Raw archive corpus like `research/agent_handoffs/combined_all_variants/` is not promoted doctrine.
- Promote archive conclusions into `HYDRA_FINAL.md` / `HYDRA_RECONCILIATION.md`; derive `docs/CURRENT_STATUS.md` from promoted doctrine plus code/runtime validation, not raw archive intake.

## Build, test, and tooling defaults

Preferred validation defaults:
- `cargo nextest run --release`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo build --release`
- use `cargo test --release` only for narrow single-test or module cases
- strict lint gate: `scripts/lint-check.sh` = Markdown compression hook + anti-game scan + rustfmt + all-features clippy; all-features path includes `cuda-graph`, so CUDA libtorch/CUDA toolkit must be available. `scripts/lint-check.sh --cuda-graph` adds focused `hydra-train` cuda-graph clippy.

Tooling rules:
- `.cargo/config.toml` is local-only and gitignored; do not commit
- `Cargo.lock` must stay committed
- Recommended Linux setup on `x86_64-unknown-linux-gnu`: `clang` + `mold` + `scripts/rustc-wrapper.sh`
- `scripts/rustc-wrapper.sh` uses `sccache` if available, else falls back to `rustc`
- Prefer `uv` / `uv tool` for Python CLIs and one-off tooling; avoid repo-local virtualenvs unless task explicitly needs one
- Keep generated/runtime-heavy artifacts out of normal commits unless intentionally refreshing them (`target/`, `output/`, local notebooks payloads)

Codebase-memory MCP:
- Use for broad code discovery before search/read loops: `search_graph`, `trace_path`, `get_code_snippet`, `detect_changes`.
- Use LSP for definition/reference/rename; use `read` before editing exact lines.
- Local index should stay cache-only. Do not commit `.codebase-memory/` artifacts.
- `.cbmignore` intentionally excludes research/provenance blobs; keep code/config/docs indexed.

## Licensing and code constraints

Licensing:
- `hydra-core` and `hydra-train` use repo-root BSL license
- `hydra-engine` is Apache-2.0 vendored upstream
- Allowed dependency licenses: MIT, Apache-2.0, BSD-compatible
- Never add AGPL, GPL, LGPL dependencies
- Never copy, adapt, derive from `Mortal-Policy/`

Rust conventions:
- Edition: 2024 or current stable if updated repo-wide
- Format with `rustfmt`
- Lint with zero warnings; `cargo clippy -- -D warnings` must pass
- Naming: `snake_case` for functions and vars, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants
- Use `anyhow::Result` for app-level errors and `thiserror` for library error enums
- No `unwrap()` in library code; tests may use it where fit
- Keep module layout flat under `src/` unless subtree guide says otherwise
- Avoid per-turn allocations in hot paths
- Use `rayon` for batch simulation when parallelism needed

Test layout:
- No inline `#[cfg(test)] mod tests { ... }` blocks in production source bodies.
- Use hybrid Rust test structure:
  - Public contract tests: `crates/<crate>/tests/*.rs`; only public crate API.
  - Subsystem white-box tests: `src/<subsystem>/tests/*.rs`; use public/`pub(crate)`/`pub(super)` subsystem API.
  - Leaf private tests: `src/<leaf>/tests.rs`; only when tests need private funcs/types/fields/constants in that exact file.
- Do not make private items `pub` only for cleaner test placement.
- `pub(super)` is OK only when item is real subsystem-internal API, not test-only visibility churn.
- If moving test upward causes privacy errors, keep it leaf-adjacent.
- Prefer clarity over uniformity; mirror Tokio-style subsystem grouping where privacy allows.

docs:
- Use `///` for public items and `//!` for module docs
- Follow RFC 1574 style
- All public items need docs
- Module files should start with `//!` summary

Markdown compression:
|- All repo `*.md` files should stay caveman-compressed: terse, exact, low-fluff, preserve technical substance.
|- Do not keep `*.original.md` backup files in repo. Rust `caveman-rs` compressor overwrites atomically without original backups.
|- Pre-commit hook runs `scripts/caveman-compress-hook.sh` on staged Markdown and refuses staged `*.original.md`.
|- If compression fails validation, leave source unchanged and report exact failure. No bluff.
