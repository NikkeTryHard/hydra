# AGENTS.md -- Hydra Root Guide

Nextest enforced when applicable. No exceptions. Save time when possible.

## Purpose

Hydra = open-source Riichi Mahjong AI targeting LuckyJ-level play. Optimize for strength first, not simplicity. If research shows stronger viable path, ship strength over simplicity.

## Goal and active build doctrine

- `research/design/HYDRA_FINAL.md` defines long-term ceiling (`Max Hydra`).
- Hydra v1 = active checkpoint to build first.
- For implementation work, choose next lane from `research/design/HYDRA_RECONCILIATION.md`, confirm shipped/staged status in `docs/CURRENT_STATUS.md`, confirm exact runtime contracts in `docs/GAME_ENGINE.md` plus current code.
- Do not treat `HYDRA_FINAL.md` alone as authorization to build every destination-facing feature now.

Core ownership summary:
- `crates/hydra-core/` — first-party runtime, encoder, simulator, safety, seeding
- `crates/hydra-train/` — first-party Burn training and model stack
- `crates/hydra-engine/` — vendored upstream rules engine, Apache-2.0

Hard source boundary:
- `Mortal-Policy/` is AGPL. Never copy, adapt, or derive code from it.

Ignored non-project directories include `.opencode/`, `RiichiEnv/`, `Mortal-Policy/`, and `.worktrees/`.

## AGENTS scope

This root `AGENTS.md` = only tracked AGENTS guide in repo tree.

## Authority and routing

Do not treat all docs equal. For doctrine and planning, follow this order and stop when enough truth found:

1. `README.md` for repo entry routing
2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` as epistemic root for research conclusions
3. Derived archive views only as helpers; if they drift, JSONL wins
4. Promoted doctrine: `research/design/HYDRA_FINAL.md` and `research/design/HYDRA_RECONCILIATION.md`
5. Promoted current-state/runtime surfaces: `docs/CURRENT_STATUS.md`, `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`
6. Current code when docs drift

Archive doctrine rules:
- `ARCHIVE_CANONICAL_CLAIMS.jsonl` = source ledger for research conclusions.
- `ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` and `ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` = derived views.
- Raw archive corpus like `research/agent_handoffs/combined_all_variants/` is not promoted doctrine.
- Promote archive conclusions into `HYDRA_FINAL.md` / `HYDRA_RECONCILIATION.md`; derive `docs/CURRENT_STATUS.md` from promoted doctrine plus code/runtime validation, not directly from raw archive intake.

## Build, test, and tooling defaults

Preferred validation defaults:
- `cargo nextest run --release`
- `cargo clippy --all-targets -- -D warnings`
- `cargo build --release`
- use `cargo test --release` only for narrow single-test or module cases

Tooling rules:
- `.cargo/config.toml` is local-only and gitignored; do not commit it
- `Cargo.lock` must stay committed
- Recommended Linux setup on `x86_64-unknown-linux-gnu`: `clang` + `mold` + `scripts/rustc-wrapper.sh`
- `scripts/rustc-wrapper.sh` uses `sccache` when available, else falls back to `rustc`
- Prefer `uv` / `uv tool` for Python CLIs and one-off tooling; avoid repo-local virtualenvs unless the task explicitly needs one
- Keep generated/runtime-heavy artifacts out of normal commits unless intentionally refreshing them (`target/`, `output/`, local notebooks payloads, transient graphify scratch files)

## Critical runtime and execution invariants

Keep these globally visible; easy to violate, expensive to rediscover:

- Encoder/model contract: `192x34`
- Old `85x34` view = baseline-prefix only (`channels 0..84`)
- Action space: 46 actions, Mortal-compatible
- Riichi and kan use two-phase handling
- Tile encoding uses normalized tile indices `0..33`
- Aka tiles stay distinct in 136-format mapping

Docker/runtime contract summary:
- Prefer Docker for real train/preflight execution, including RL/self-play and GPU runs
- Container entrypoint is `train` and expects YAML config
- Keep mounted paths aligned with config expectations: `/data` and `/output`
- BC training supports either flat MJAI directory of `.json` / `.json.gz` files or direct `.tar.zst` archive path
- User-edited training and preflight settings should live in YAML; do not rely on ad-hoc `HYDRA_PREFLIGHT_*` env vars for normal workflows

## Licensing and code constraints

Licensing:
- `hydra-core` and `hydra-train` use repo-root BSL license
- `hydra-engine` is Apache-2.0 vendored upstream
- Allowed dependency licenses: MIT, Apache-2.0, BSD-compatible
- Never add AGPL, GPL, or LGPL dependencies
- Never copy, adapt, or derive from `Mortal-Policy/`

Rust conventions:
- Edition: 2024 or current stable if updated repo-wide
- Format with `rustfmt`
- Lint with zero warnings; `cargo clippy -- -D warnings` must pass
- Naming: `snake_case` for functions and variables, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants
- Use `anyhow::Result` for app-level errors and `thiserror` for library error enums
- No `unwrap()` in library code; tests may use it when appropriate
- Keep module layout flat under `src/` unless subtree guide says otherwise
- Avoid per-turn allocations in hot paths
- Use `rayon` for batch simulation when parallelism needed

Documentation:
- Use `///` for public items and `//!` for module docs
- Follow RFC 1574 style
- All public items need docs
- Module files should start with `//!` summary

## graphify

Graphify = primary navigation layer for codebase structure and repo discovery.

Current repo graph:
- Built with `graphifyy 0.7.7` from the uv tool install at `~/.local/share/uv/tools/graphifyy/bin/python`.
- Latest observed graph report: `graphify-out/GRAPH_REPORT.md` from 2026-05-05, 5,738 nodes, 11,737 edges, 269 communities.
- Agent-crawlable wiki exists at `graphify-out/wiki/index.md` (194 markdown files generated with `graphify.wiki.to_wiki`).
- `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/manifest.json`, and `graphify-out/wiki/` are the durable graph artifacts. `.graphify_*` root scratch files are legacy/transient.

Rules:
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw file trees.
- Prefer graphify and targeted docs over keeping large static repo maps in this file.
- Use the recorded interpreter when present: `PYTHON=$(python3 -c "from pathlib import Path; print(Path('graphify-out/.graphify_python').read_text().strip())")`.
- If interpreter missing/stale, restore via uv: `uv tool install graphifyy` or `uv tool upgrade graphifyy`, then write `graphify-out/.graphify_python` to `~/.local/share/uv/tools/graphifyy/bin/python`.
- After modifying code files, rebuild code graph with: `"$PYTHON" -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"`.
- For full graph refreshes, prefer latest graphify skill instructions and uv-managed graphify over system `pip`; keep OMP-specific task/subagent rules from `~/.omp/agent/skills/graphify/SKILL.md`.