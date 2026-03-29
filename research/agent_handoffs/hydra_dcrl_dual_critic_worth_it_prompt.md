# Hydra prompt — DCRL / dual-critic worth-it evaluation

<role>
You are evaluating whether one nearby prior-art family deserves explicit Hydra attention.
This is a narrow research-decision prompt, not a generic RL literature survey.
You are allowed to conclude that the right action is tiny, docs-only, or no-op if that is what the evidence supports.
</role>

<task>
Answer the narrow question: is it worth explicitly integrating DCRL / dual-critic / asymmetric-critic prior-art coverage into Hydra's promoted docs, and is any deeper method attention warranted beyond docs positioning?

Treat this as two separate decisions that may diverge:
- docs-worth-it: should Hydra explicitly cite/discuss DCRL or adjacent dual/asymmetric critic work in README, promoted doctrine, or reference surfaces?
- method-worth-it: should Hydra change architecture priorities, add a concrete reserve lane, or otherwise materially react at the method level?

We want a detailed answer that makes clear:
- what the repo artifacts already prove about Hydra's current oracle critic / CTDE / RVR / search-feature / training surfaces
- what exact overlap exists between Hydra and DCRL-like prior work versus what is only superficial naming overlap
- what external primary sources materially change the answer
- what is directly supported by artifacts, what is externally supported, and what is only inference
- what confidence each major conclusion deserves
- the smallest docs patch set if explicit coverage is worth it
- the smallest method-level follow-up if deeper attention is worth it
- the exact no-op rationale if the right answer is only 'mention lightly' or 'do nothing'
- what evidence would falsify the current recommendation

Do not stop at 'yes it is related' or 'no it is different.' We need a decision-ready verdict with the narrowest justified next action. Use the artifacts below to derive your conclusions.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections you may customize when the prompt needs it
- distinguish direct artifact support from your own inference
- use search/browse aggressively when it can strengthen the answer: find the original paper, adjacent papers, official docs, repos, and other primary sources; use abstracts or summaries mainly for discovery, not as the final evidence base
- use the bash tool to run Python for lightweight research support work when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, and validation
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- if you claim a path works, survives, or is implementation-ready, show why that confidence is justified and how the claim can be validated or falsified later
- inspect your own draft before finishing: if a confident claim is not objectively justified by visible evidence, downgrade it to inference, proposal, or blocked
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated, falsified, or truly blocked, and do not stop just because the first pass produced a plausible answer
- keep docs-worth-it and method-worth-it separate all the way through the answer
- do not treat adjacency as equivalence; distinguish shared ingredients, same training trust object, same deployment trust object, and same optimization target
- prefer full papers, official docs, and original repos for DCRL, UAAC, asymmetric critic, oracle guiding, and related primary comparisons when available
- if Hydra already covers a stronger or more directly relevant adjacent prior-art family than DCRL, say that explicitly instead of inflating DCRL importance
- if the right recommendation is only a citation or two plus one paragraph of positioning, say that instead of inventing a bigger lane
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when you sound confident, show the justification for that confidence level
- for every important claim, make the validation path visible enough that a reviewer can test it later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate, reproduce, or falsify it ourselves (pdfs, sources, links, similar projects, concrete checks)
- lead with a crisp verdict before the deeper justification
- include a small decision matrix or comparison table if it sharpens the answer
- if you recommend docs edits, make them concrete enough that a maintainer could implement them without another research pass
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>

## Artifact 01 — Prompt-packing reminder
Artifact id: `prompt-packing-reminder`
Source label: META
Type: `literal`
Why it matters: Short task-specific reminder aligned with Hydra's prompt style guide: answer the narrow hard question, but pack enough local evidence that the research agent does not waste its first pass rediscovering repo reality.

```text
This is a serious but narrow Hydra research-decision prompt.
The goal is not a generic literature survey; the goal is a decision-ready verdict on whether DCRL / dual-critic / asymmetric-critic prior art is worth explicit Hydra attention.
Bias toward the smallest justified next action.
If the repo already contains the stronger adjacent prior-art framing, say so.
If DCRL matters mainly as a documentation-positioning citation rather than a method pivot, say that clearly.
```

## Artifact 02 — Local narrowing summary
Artifact id: `task-context`
Source label: META
Type: `literal`
Why it matters: Compact local context from repo triage and external note reading so the research agent starts from the real narrowed question instead of redoing first-mile speculation.

```text
User context: people mocked Hydra because Hydra docs did not explicitly mention a nearby dual-critic paper / adjacent prior-art family.
Important distinction already narrowed locally: this is not automatically a claim that Hydra's method is fake or copied; the narrower question is whether missing that prior-art positioning makes the docs materially worse and whether the method should react at all.
Repo-grounded starting point from local search: promoted Hydra docs and research notes already mention CTDE, oracle critics, oracle guiding, RVR, centralized value function language, and Suphx / LuckyJ / AlphaStar-style adjacent ideas; exact DCRL / UAAC / 'dual critic' wording was not found in README or promoted docs during local triage.
OpenReview note context from local external triage: the linked DCRL note supports novelty / literature-positioning skepticism more than a strong accusation that a specific prior paper was literally unknown.
Your job is to decide whether explicit DCRL-or-adjacent coverage is actually worth adding, and whether that answer changes only docs positioning or also Hydra's method / roadmap.
```

## Artifact 03 — Top-level repo routing and research surface
Artifact id: `repo-readme`
Source label: DOC
Type: `file_full`
Source: `README.md`
Why it matters: Shows the public-facing project summary, current documentation routing, and what the repo chooses to expose at the README layer. This is the main evidence for whether DCRL-adjacent prior-art omission is visible at the most public surface.

```markdown
[DOC L0001] # Hydra
[DOC L0002] 
[DOC L0003] Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.
[DOC L0004] 
[DOC L0005] > ## Compute support
[DOC L0006] > This research used the Delta advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.
[DOC L0007] 
[DOC L0008] ## Goal
[DOC L0009] 
[DOC L0010] Train a mahjong AI that:
[DOC L0011] - Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
[DOC L0012] - Releases weights under a permissive license
[DOC L0013] - Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs
[DOC L0014] 
[DOC L0015] ## Architecture
[DOC L0016] 
[DOC L0017] Hydra uses a layered authority flow built from the archive handoff canon upward:
[DOC L0018] 
[DOC L0019] 1. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — epistemic root / canonical archive SSOT for upstream research conclusions
[DOC L0020] 2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) and [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) — derived archive views over that canonical source ledger
[DOC L0021] 3. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine built from archive canon plus repo validation
[DOC L0022] 4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine and roadmap to Hydra v1 built from archive canon plus repo validation
[DOC L0023] 5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current-status snapshot for already-built repo surfaces
[DOC L0024] 6. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — runtime semantics and compatibility surfaces; current code wins when docs drift
[DOC L0025] 
[DOC L0026] Hydra's documentation split is simple:
[DOC L0027] 
[DOC L0028] - `HYDRA_FINAL.md` describes the max-ceiling destination
[DOC L0029] - `HYDRA_RECONCILIATION.md` is the roadmap to Hydra v1
[DOC L0030] - `docs/CURRENT_STATUS.md` says what is already shipped or still staged today
[DOC L0031] 
[DOC L0032] Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` remain raw archive corpus, not promoted doctrine.
[DOC L0033] 
[DOC L0034] ## Fresh-agent routing
[DOC L0035] 
[DOC L0036] If you are entering Hydra with zero prior memory, use this order and stop when you have enough truth for the task:
[DOC L0037] 
[DOC L0038] 1. `README.md` for repo routing
[DOC L0039] 2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` for canonical archive intake
[DOC L0040] 3. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` for derived archive triage
[DOC L0041] 4. `research/design/HYDRA_RECONCILIATION.md` for the roadmap to Hydra v1
[DOC L0042] 5. `research/design/HYDRA_FINAL.md` for the long-term ceiling
[DOC L0043] 6. `docs/CURRENT_STATUS.md` for shipped/staged truth
[DOC L0044] 7. `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` for runtime truth
[DOC L0045] 
[DOC L0046] `combined_all_variants/` remains raw archive corpus for provenance only.
[DOC L0047] 
[DOC L0048] ## Status vocabulary
[DOC L0049] 
[DOC L0050] For implementation work, choose the next lane from
[DOC L0051] `research/design/HYDRA_RECONCILIATION.md`, confirm whether it already exists in
[DOC L0052] `docs/CURRENT_STATUS.md`, and confirm exact runtime contracts in
[DOC L0053] `docs/GAME_ENGINE.md` plus current code.
[DOC L0054] 
[DOC L0055] | Term | Meaning |
[DOC L0056] |---|---|
[DOC L0057] | `active path` | what Hydra should optimize/build now |
[DOC L0058] | `shipped baseline` | implemented and part of the current live baseline |
[DOC L0059] | `implemented but not default-on` | implemented in code, intentionally not the default path |
[DOC L0060] | `implemented but staged` | implemented enough to exist, but activation/promotion is intentionally deferred |
[DOC L0061] | `reserve shelf` | preserved later-work direction, not current mainline |
[DOC L0062] | `blocked` | not ready because a real dependency or semantic gap remains |
[DOC L0063] | `rejected` | not part of the current plan |
[DOC L0064] | `historical` | preserved context only; not governing truth |
[DOC L0065] 
[DOC L0066] ## Crate ownership
[DOC L0067] 
[DOC L0068] | Crate | Owns | Does not own |
[DOC L0069] |---|---|---|
[DOC L0070] | `crates/hydra-engine` | vendored rules engine behavior | Hydra-specific runtime/training orchestration |
[DOC L0071] | `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing | Burn training logic or vendored rules ownership |
[DOC L0072] | `crates/hydra-train` | model, targets, losses, BC/RL/self-play orchestration, train binary | low-level rules engine behavior |
[DOC L0073] 
[DOC L0074] If you are deciding what to build next, follow the Fresh-agent routing order above.
[DOC L0075] `research/design/HYDRA_SPEC.md` remains historical context only.
[DOC L0076] 
[DOC L0077] ## Research
[DOC L0078] 
[DOC L0079] | File | What's In It |
[DOC L0080] |------|-------------|
[DOC L0081] | [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
[DOC L0082] | [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
[DOC L0083] | [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of the canonical archive ledger |
[DOC L0084] | [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
[DOC L0085] | [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted operational doctrine summary and roadmap to Hydra v1 |
[DOC L0086] | [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Reserve-only design/archive planning surfaces |
[DOC L0087] | [HYDRA_SPEC.md](research/design/HYDRA_SPEC.md) | Historical architecture spec only |
[DOC L0088] | [MORTAL_ANALYSIS.md](research/intel/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
[DOC L0089] | [OPPONENT_MODELING.md](research/design/OPPONENT_MODELING.md) | Opponent-modeling rationale; includes both active ideas and reserve/future extensions |
[DOC L0090] | [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
[DOC L0091] | [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
[DOC L0092] | [CHECKPOINTING.md](research/infrastructure/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
[DOC L0093] | [ECOSYSTEM.md](research/intel/ECOSYSTEM.md) | Useful repos, tooling, and framework references |
[DOC L0094] | [REWARD_DESIGN.md](research/design/REWARD_DESIGN.md) | Reward design and RVR notes |
[DOC L0095] | [COMMUNITY_INSIGHTS.md](research/intel/COMMUNITY_INSIGHTS.md) | Community observations and external signals |
[DOC L0096] | [REFERENCES.md](research/intel/REFERENCES.md) | Citation index |
[DOC L0097] | [TESTING.md](research/design/TESTING.md) | Testing strategy, correctness verification, property-based tests |
[DOC L0098] | [RUST_STACK.md](research/infrastructure/RUST_STACK.md) | 100% Rust decision and framework notes |
[DOC L0099] 
[DOC L0100] ## Status
[DOC L0101] 
[DOC L0102] Hydra is in active implementation. For the current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).
[DOC L0103] 
[DOC L0104] ## Testing and Coverage
[DOC L0105] 
[DOC L0106] Hydra uses `cargo nextest run --release` as the default workspace test path and `cargo-llvm-cov` for workspace-wide coverage reporting. For local coverage generation details, read [`docs/COVERAGE.md`](docs/COVERAGE.md).
[DOC L0107] 
[DOC L0108] ## License
[DOC L0109] 
[DOC L0110] - **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
[DOC L0111] - **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
```

## Artifact 04 — Current shipped / staged status snapshot
Artifact id: `current-status`
Source label: DOC
Type: `file_full`
Source: `docs/CURRENT_STATUS.md`
Why it matters: Live status snapshot helps the research agent separate current implementation reality from long-term doctrine when judging whether DCRL-adjacent work is only a citation issue or a live method issue.

```markdown
[DOC L0001] # Hydra Current Status
[DOC L0002] 
[DOC L0003] Current shipped/staged status for Hydra's already-built surfaces.
[DOC L0004] 
[DOC L0005] This file is Hydra's promoted current-status snapshot for things that already exist in code or are partially implemented in code. Use it to answer questions like "what is shipped today?", "what is implemented but still staged?", and "what is implemented but not default-on yet?"
[DOC L0006] 
[DOC L0007] This file reports shipped/staged status only.
[DOC L0008] 
[DOC L0009] - For the roadmap to Hydra v1, read `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0010] - For runtime semantics and compatibility truth, read `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.
[DOC L0011] 
[DOC L0012] When this file and current code disagree, current code wins. When this file and `HYDRA_RECONCILIATION.md` disagree on active vs reserve vs staged priority, refresh reconciliation and then refresh this file. When reconciliation or current status drift from the archive root, refresh the promoted docs rather than demoting the canonical archive source ledger.
[DOC L0013] 
[DOC L0014] ## Status vocabulary
[DOC L0015] 
[DOC L0016] This file uses the status vocabulary defined in `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0017] 
[DOC L0018] | Term | Meaning |
[DOC L0019] |---|---|
[DOC L0020] | `shipped baseline` | implemented and part of the current live baseline |
[DOC L0021] | `implemented but not default-on` | implemented and validated enough to exist in-code, but intentionally not the default runtime/training path |
[DOC L0022] | `implemented but staged` | core code path exists, but promotion/activation is still intentionally deferred |
[DOC L0023] | `reserve shelf` | documented later-work direction, not current mainline priority |
[DOC L0024] | `historical` | preserved context only; not current governing truth |
[DOC L0025] 
[DOC L0026] ## Runtime and training snapshot
[DOC L0027] 
[DOC L0028] ### Shipped baseline
[DOC L0029] 
[DOC L0030] - `hydra-core` is a real first-party runtime/encoder/simulator crate.
[DOC L0031] - The live encoder/model contract is `192x34`; the old `85x34` view is baseline-prefix only.
[DOC L0032] - The fixed runtime action space is 46 actions with two-phase riichi and kan handling.
[DOC L0033] - BC training now supports **epoch-boundary-only** reuse of matching preflight-selected runtime for the selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, derived `accum_steps`), while fresh runs remain config-derived, partial-epoch resumes still require identical runtime, and loader-runtime stays config-derived.
[DOC L0034] - The stronger public-teacher belief-semantics tranche is shipped as part of the current training baseline.
[DOC L0035] - The current Hand-EV realism upgrade is shipped as part of the live baseline surface.
[DOC L0036] - Replay-derived `safety_residual` is shipped as a narrow supervised lane.
[DOC L0037] - ExIt now has an end-to-end carrier across the live self-play lane and the replay/sample sidecar-first lane.
[DOC L0038] 
[DOC L0039] ### Implemented but not default-on
[DOC L0040] 
[DOC L0041] - The narrow DeltaQ supervision lane is implemented in code and promotion-gated through an arena-confirmation path.
[DOC L0042] - DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but the lane is still **not** default-on.
[DOC L0043] 
[DOC L0044] ### Implemented but staged
[DOC L0045] 
[DOC L0046] - `mixture_weight` promotion remains staged.
[DOC L0047] - Richer opponent-target closure remains staged.
[DOC L0048] - Representative-world / per-particle CT-SMC Hand-EV remains staged.
[DOC L0049] - Selective AFBS / endgame deepening remains staged.
[DOC L0050] 
[DOC L0051] ### Reserve shelf
[DOC L0052] 
[DOC L0053] - Broader public-belief search as project identity remains reserve-shelf, not active-path.
[DOC L0054] - Deeper robust-opponent search backups remain reserve-shelf.
[DOC L0055] - Larger latent-opponent / richer auxiliary-head expansion remains reserve-shelf until existing target closure improves.
[DOC L0056] 
[DOC L0057] ## Area-by-area summary
[DOC L0058] 
[DOC L0059] | Area | Current status | Notes |
[DOC L0060] |---|---|---|
[DOC L0061] | Runtime encoder / action semantics | shipped baseline | See `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` |
[DOC L0062] | Hand-EV baseline surface | shipped baseline | Stronger local evaluator is live; representative-world CT-SMC Hand-EV remains staged |
[DOC L0063] | Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche is in the live baseline |
[DOC L0064] | BC runtime authority | shipped baseline | Fresh runs are config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
[DOC L0065] | `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
[DOC L0066] | ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
[DOC L0067] | DeltaQ lane | implemented but not default-on | Arena-confirmation path implemented; promotion artifact now records pre-arena recommendation plus final `arena_decision`/`arena_report` |
[DOC L0068] | `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
[DOC L0069] | `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
[DOC L0070] | AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |
[DOC L0071] 
[DOC L0072] ## Where to read next
[DOC L0073] 
[DOC L0074] - Need the current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
[DOC L0075] - Need the roadmap to Hydra v1 or the active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0076] - Need the north-star architecture rather than current shipped status? Read `research/design/HYDRA_FINAL.md`.
```

## Artifact 05 — Runtime semantics and engine truth
Artifact id: `game-engine`
Source label: DOC
Type: `file_full`
Source: `docs/GAME_ENGINE.md`
Why it matters: Grounds the current runtime surface so the research agent can avoid inventing architecture deltas that do not match live engine semantics.

```markdown
[DOC L0001] # Hydra Game Engine (hydra-core)
[DOC L0002] 
[DOC L0003] Reference documentation for the `hydra-core` Rust crate, the game engine powering the Hydra Riichi Mahjong AI.
[DOC L0004] 
[DOC L0005] ## Overview
[DOC L0006] 
[DOC L0007] `hydra-core` is a Rust library that provides everything the Hydra training pipeline and runtime need from the game side: a complete Riichi Mahjong simulator, observation encoding, safety analysis, search/belief feature bridging, and batch execution. It wraps `riichienv-core` as the underlying game engine and layers Hydra-specific encoding, seeding, and orchestration on top.
[DOC L0008] 
[DOC L0009] Core responsibilities:
[DOC L0010] 
[DOC L0011] - Tile representation and suit permutation for data augmentation
[DOC L0012] - A 46-action space with bidirectional conversion to/from `riichienv` actions
[DOC L0013] - A currently implemented **192-channel x 34-tile fixed-superset observation encoder**, whose first 85 channels preserve the original public+safety baseline while Groups C/D add live search/belief and Hand-EV planes
[DOC L0014] - Tile safety analysis (genbutsu, suji, kabe, one-chance)
[DOC L0015] - Deterministic seeding via SHA-256 KDF + ChaCha8Rng
[DOC L0016] - Parallel batch simulation with `rayon`
[DOC L0017] - A game loop abstraction with pluggable action selection
[DOC L0018] 
[DOC L0019] Hydra uses a 100% Rust stack (see `research/infrastructure/RUST_STACK.md`). The training pipeline (hydra-train, using Burn framework) consumes hydra-core directly -- same process, same memory, zero IPC.
[DOC L0020] 
[DOC L0021] ## Foundation: RiichiEnv
[DOC L0022] 
[DOC L0023] The game engine is built on top of [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` crate, Apache-2.0 license).
[DOC L0024] 
[DOC L0025] RiichiEnv provides:
[DOC L0026] 
[DOC L0027] - Full 4-player and 3-player Riichi Mahjong rules
[DOC L0028] - Red dora (aka-dora) support for all three suits
[DOC L0029] - All kan types: ankan (closed), daiminkan (open), shouminkan (added)
[DOC L0030] - Native MJAI protocol compatibility for game state representation
[DOC L0031] - Correctness verified by running MortalAgent (AGPL, used as a black-box MJAI player -- no code shared) over 1M+ hanchan without errors ([source: RiichiEnv README](https://github.com/smly/RiichiEnv#-features))
[DOC L0032] 
[DOC L0033] Hydra treats `riichienv-core` as a black-box game engine. All game state progression, legality checks, and rule enforcement happen inside RiichiEnv. Hydra's own code handles encoding, analysis, and orchestration only.
[DOC L0034] 
[DOC L0035] Because riichienv-core's correctness is already verified upstream -- smly ran Mortal as a black-box MJAI player (separate process, no linking) over 1M+ hanchan on RiichiEnv with zero errors ([source](https://github.com/smly/RiichiEnv)) -- Hydra does not need its own cross-engine validation. The correctness guarantee is inherited through the dependency. No Mortal code exists in RiichiEnv or Hydra.
[DOC L0036] 
[DOC L0037] ## Module Reference
[DOC L0038] 
[DOC L0039] | Module | File | Description |
[DOC L0040] |--------|------|-------------|
[DOC L0041] | `tile` | `tile.rs` | Tile types (0-33), 136-format representation, aka-dora handling, suit permutation |
[DOC L0042] | `action` | `action.rs` | 46-action space, `HydraAction` enum, bidirectional riichienv conversion, legal mask builder |
[DOC L0043] | `encoder` | `encoder.rs` | 192x34 fixed-superset observation tensor, `ObservationEncoder`, incremental encoding with `DirtyFlags` |
[DOC L0044] | `safety` | `safety.rs` | `SafetyInfo` per-opponent tile safety: genbutsu, suji, kabe, one-chance |
[DOC L0045] | `simulator` | `simulator.rs` | `BatchSimulator` with rayon thread pool, `BatchConfig`, `GameResult` collection |
[DOC L0046] | `seeding` | `seeding.rs` | SHA-256 KDF, `SessionRng`, deterministic wall generation, Fisher-Yates shuffle |
[DOC L0047] | `bridge` | `bridge.rs` | Converts riichienv `Observation` into encoder-ready data via `extract_*` functions |
[DOC L0048] | `game_loop` | `game_loop.rs` | `GameRunner`, `ActionSelector` trait, step-by-step or run-to-completion execution |
[DOC L0049] | `batch_encoder` | `batch_encoder.rs` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
[DOC L0050] | `shanten_batch` | `shanten_batch.rs` | Batch shanten with hierarchical hash caching (base + all 34 discards in one pass) |
[DOC L0051] 
[DOC L0052] 
[DOC L0053] ## Tile System (`tile.rs`)
[DOC L0054] 
[DOC L0055] ### TileType
[DOC L0056] 
[DOC L0057] All tiles use a `TileType(u8)` newtype representing the 34 distinct Mahjong tile kinds:
[DOC L0058] 
[DOC L0059] | Range | Tiles | Count |
[DOC L0060] |-------|-------|-------|
[DOC L0061] | 0-8 | 1m through 9m (manzu/characters) | 9 |
[DOC L0062] | 9-17 | 1p through 9p (pinzu/circles) | 9 |
[DOC L0063] | 18-26 | 1s through 9s (souzu/bamboo) | 9 |
[DOC L0064] | 27-33 | East, South, West, North, Haku, Hatsu, Chun | 7 |
[DOC L0065] 
[DOC L0066] The physical game uses 136 tiles (4 copies of each type). The 136-format index identifies a specific physical tile, while `TileType` identifies its kind. Converting between them is a simple `tile136 / 4` truncation.
[DOC L0067] 
[DOC L0068] ### Aka-Dora (Red Fives)
[DOC L0069] 
[DOC L0070] Three tiles in the 136-format set are designated red dora (aka-dora):
[DOC L0071] 
[DOC L0072] - Red 5m (manzu)
[DOC L0073] - Red 5p (pinzu)
[DOC L0074] - Red 5s (souzu)
[DOC L0075] 
[DOC L0076] These are the 0th copy (index 0 within each group of 4) of the respective 5-tiles: 136-format indices 16 (5m), 52 (5p), 88 (5s). Extended tile type indices 34-36 represent aka variants in the action space. The encoder and action space both handle aka-dora as distinct from regular fives where needed.
[DOC L0077] 
[DOC L0078] ### Suit Permutation
[DOC L0079] 
[DOC L0080] For data augmentation during training, `tile.rs` provides suit permutation functions. There are 6 permutations of the three numbered suits (manzu, pinzu, souzu), leaving honor tiles untouched. Given a permutation index (0-5), the module remaps all tile types in a hand/observation to the permuted suit assignment. This 6x data augmentation helps the model learn suit-invariant patterns.
[DOC L0081] 
[DOC L0082] ## Action Space (`action.rs`)
[DOC L0083] 
[DOC L0084] ### 46-Action Space
[DOC L0085] 
[DOC L0086] Hydra uses a fixed 46-action output space. Every decision point in the game maps to one of these action indices:
[DOC L0087] 
[DOC L0088] | Index | Action | Notes |
[DOC L0089] |-------|--------|-------|
[DOC L0090] | 0-33 | Discard tile type 0-33 | Standard discard (non-red) |
[DOC L0091] | 34-36 | Discard aka 5m, 5p, 5s | Discard a specific red five |
[DOC L0092] | 37 | Declare riichi | Announces riichi; tile selection follows |
[DOC L0093] | 38-40 | Chi (3 variants) | Left/middle/right chi calls |
[DOC L0094] | 41 | Pon | Open pon call |
[DOC L0095] | 42 | Kan | Any kan type (ankan, daiminkan, shouminkan) |
[DOC L0096] | 43 | Agari | Win declaration (tsumo or ron) |
[DOC L0097] | 44 | Ryuukyoku | Abortive draw declaration (kyuushu kyuuhai, etc.) |
[DOC L0098] | 45 | Pass | Decline a call opportunity |
[DOC L0099] 
[DOC L0100] ### Two-Phase Actions
[DOC L0101] 
[DOC L0102] Riichi and kan use a two-phase selection process. The model first outputs a phase-1 action (index 37 for riichi, 42 for kan). Then the game engine presents the legal tile choices and the model picks which specific tile to discard (riichi) or which specific kan to declare. This keeps the action space compact at 46 while supporting the full combinatorial range.
[DOC L0103] 
[DOC L0104] ### HydraAction
[DOC L0105] 
[DOC L0106] `HydraAction` is a validated newtype wrapper around `u8`:
[DOC L0107] 
[DOC L0108] ```rust
[DOC L0109] pub struct HydraAction(u8);
[DOC L0110] ```
[DOC L0111] 
[DOC L0112] It validates the index is in range 0-45 on construction via `HydraAction::new(id) -> Option<Self>`. Methods like `is_discard()`, `is_aka_discard()`, and `discard_tile_type()` provide type-safe access. Bidirectional conversion functions `hydra_to_riichienv()` and `riichienv_to_hydra()` translate between Hydra's compact action space and riichienv-core's `Action` struct, using a `GameContext` to resolve context-dependent actions (chi consume tiles, tsumo vs ron, kan type).
[DOC L0113] 
[DOC L0114] ### Legal Action Mask
[DOC L0115] 
[DOC L0116] The `build_legal_action_mask` function takes the current riichienv game state and returns a `[bool; 46]` array. Each slot is `true` if that action is legal in the current state. The training pipeline uses this mask to zero out illegal actions before softmax, guaranteeing the model never selects an impossible move.
[DOC L0117] 
[DOC L0118] ## Observation Encoder (`encoder.rs`)
[DOC L0119] 
[DOC L0120] ### Tensor Shape
[DOC L0121] 
[DOC L0122] **Routing note:** this file records current runtime reality for the live encoder/runtime, and current code wins if docs drift. For repo entry routing, trust/status vocabulary, and crate ownership, read `README.md`. For active-path / staged-vs-reserve decisions, read `research/design/HYDRA_RECONCILIATION.md`. For the compact compatibility contract, read `docs/COMPATIBILITY_SURFACE.md`. The original `85 x 34` tensor now describes the **baseline prefix** of the live encoder, not the full live encoder. The current implementation is already a **fixed-shape 192 x 34 superset** with Groups C/D plus presence-mask channels.
[DOC L0123] 
[DOC L0124] Each observation is a `192 x 34` float tensor (6,528 values). The first 85 channels retain the baseline public+safety encoding; the remaining channels provide fixed-shape search/belief and Hand-EV context with zero-fill plus explicit presence masks when dynamic features are unavailable. This full shape feeds directly into the current SE-ResNet model input.
[DOC L0125] 
[DOC L0126] ### Baseline Prefix Channel Layout (channels 0-84)
[DOC L0127] 
[DOC L0128] The 85 channels break down into these groups:
[DOC L0129] 
[DOC L0130] | Channels | Name | Encoding |
[DOC L0131] |----------|------|----------|
[DOC L0132] | 0-3 | Closed hand | Thresholded: ch N is 1.0 if tile count >= N+1 |
[DOC L0133] | 4-7 | Open meld hand | Same thresholding for tiles exposed in open melds |
[DOC L0134] | 8 | Drawn tile | One-hot: 1.0 at the tile type just drawn (tsumo only) |
[DOC L0135] | 9-10 | Shanten masks | Ch 9: keep-shanten (tiles whose discard does not increase shanten). Ch 10: next-shanten (tiles whose discard decreases shanten) |
[DOC L0136] | 11-13 | Player 0 discards | Presence (1.0 if discarded), tedashi flag (1.0 if from hand, not tsumogiri), temporal weight (exp(-0.2 * age)) |
[DOC L0137] | 14-16 | Player 1 discards | Same three channels, relative to seat |
[DOC L0138] | 17-19 | Player 2 discards | Same three channels, relative to seat |
[DOC L0139] | 20-22 | Player 3 discards | Same three channels, relative to seat |
[DOC L0140] | 23-25 | Player 0 melds | Chi (1.0 for tiles in chi melds), pon (tiles in pon), kan (tiles in kan) |
[DOC L0141] | 26-28 | Player 1 melds | Same three channels |
[DOC L0142] | 29-31 | Player 2 melds | Same three channels |
[DOC L0143] | 32-34 | Player 3 melds | Same three channels |
[DOC L0144] | 35-39 | Dora indicators | Thermometer encoding: ch N is 1.0 if N+1 or more dora indicators revealed |
[DOC L0145] | 40-42 | Aka dora flags | Per-suit plane: ch 40 = manzu red five, ch 41 = pinzu, ch 42 = souzu. 1.0 at the 5-tile column if that red five is visible |
[DOC L0146] | 43-46 | Riichi flags | One channel per player. Entire plane is 1.0 if that player has declared riichi |
[DOC L0147] | 47-50 | Scores | One channel per player. Entire plane filled with score / 100,000 |
[DOC L0148] | 51-54 | Relative score gaps | One channel per player. Filled with (player_score - my_score) / 30,000 |
[DOC L0149] | 55-58 | Shanten one-hot | Ch 55 = tenpai (shanten 0), ch 56 = iishanten (1), ch 57 = ryanshanten (2), ch 58 = 3+ shanten. Entire plane is 1.0 for the matching shanten count |
[DOC L0150] | 59 | Round number | Entire plane filled with kyoku / 8.0 (normalized round index) |
[DOC L0151] | 60 | Honba count | Entire plane filled with honba / 10.0 |
[DOC L0152] | 61 | Kyotaku (riichi sticks) | Entire plane filled with kyotaku / 10.0 |
[DOC L0153] | 62-84 | Safety channels | 23 channels of per-opponent tile safety data (see Safety System section) |
[DOC L0154] 
[DOC L0155] **Safety channel breakdown (channels 62-84):**
[DOC L0156] 
[DOC L0157] | Channels | Name |
[DOC L0158] |----------|------|
[DOC L0159] | 62-64 | Genbutsu (all): 1.0 for tiles each opponent discarded (one ch per opponent) |
[DOC L0160] | 65-67 | Genbutsu (tedashi): restricted to tiles discarded from hand (not tsumogiri) |
[DOC L0161] | 68-70 | Genbutsu (riichi-era): restricted to tiles discarded after opponent's riichi |
[DOC L0162] | 71-73 | Suji: float 0.0-1.0 for suji-inferred safety against each opponent |
[DOC L0163] | 74-76 | Half-suji indicator | 1.0 when the tile is half-suji-safe against that opponent |
[DOC L0164] | 77-79 | Matagi-suji danger | float danger signal for matagi-suji patterns against that opponent |
[DOC L0165] | 80 | Kabe: 1.0 for tiles with all 4 copies visible (global, not per-opponent) |
[DOC L0166] | 81 | One-chance: 1.0 for tiles where exactly 3 of 4 copies are visible |
[DOC L0167] | 82-84 | Tenpai hints | Opponent tenpai hints (implemented baseline use: riichi or cached tenpai prediction threshold) |
[DOC L0168] 
[DOC L0169] ### ObservationEncoder
[DOC L0170] 
[DOC L0171] `ObservationEncoder` is the main struct for building observation tensors. In the current implementation it holds a pre-allocated `[f32; 192 * 34]` buffer marked `#[repr(C)]` for predictable memory layout. The baseline public+safety channels remain intact in the first 85 planes; Groups C/D are already present as fixed-shape extensions.
[DOC L0172] 
[DOC L0173] ```rust
[DOC L0174] #[repr(C)]
[DOC L0175]     pub struct ObservationEncoder {
[DOC L0176]     buffer: [f32; 6528],  // 192 channels x 34 tiles, row-major
[DOC L0177] }
[DOC L0178] ```
[DOC L0179] 
[DOC L0180] ### Incremental Encoding with DirtyFlags
[DOC L0181] 
[DOC L0182] `DirtyFlags` is a bitflags struct where each bit corresponds to a channel group (hand, discards, melds, dora, scores, safety, etc.). When the game state changes, only the relevant flags are set. On the next `encode()` call, only flagged channel groups are recomputed. Unchanged channels keep their previous values in the buffer.
[DOC L0183] 
[DOC L0184] This matters for performance: a single discard only dirties the discard and safety channels, skipping the more expensive hand/meld/dora re-encoding. During batch simulation of thousands of games, these savings compound.
[DOC L0185] 
[DOC L0186] ## Safety System (`safety.rs`)
[DOC L0187] 
[DOC L0188] The safety module computes per-opponent, per-tile safety information used to populate encoder channels 62-84 and to inform defensive play decisions.
[DOC L0189] 
[DOC L0190] ### SafetyInfo
[DOC L0191] 
[DOC L0192] `SafetyInfo` holds safety data from one player's perspective against all 3 opponents:
[DOC L0193] 
[DOC L0194] ```rust
[DOC L0195] #[repr(C)]
[DOC L0196] pub struct SafetyInfo {
[DOC L0197]     pub genbutsu_all: [[bool; 34]; 3],       // per-opponent
[DOC L0198]     pub genbutsu_tedashi: [[bool; 34]; 3],   // per-opponent
[DOC L0199]     pub genbutsu_riichi_era: [[bool; 34]; 3], // per-opponent
[DOC L0200]     pub suji: [[f32; 34]; 3],                // per-opponent, float 0.0-1.0
[DOC L0201]     pub kabe: [bool; 34],                     // global
[DOC L0202]     pub one_chance: [bool; 34],               // global
[DOC L0203]     pub visible_counts: [u8; 34],             // global tile visibility
[DOC L0204]     pub opponent_riichi: [bool; 3],           // per-opponent riichi status
[DOC L0205] }
[DOC L0206] ```
[DOC L0207] 
[DOC L0208] **Genbutsu** (safe tiles) tracks tiles that a specific opponent cannot ron:
[DOC L0209] 
[DOC L0210] - `genbutsu_all`: any tile the opponent discarded (always safe against that player)
[DOC L0211] - `genbutsu_tedashi`: only tiles discarded from the opponent's hand (not tsumogiri), indicating intentional discards
[DOC L0212] - `genbutsu_riichi_era`: only tiles discarded after the opponent declared riichi, relevant for reading post-riichi waits
[DOC L0213] 
[DOC L0214] **Suji** inference identifies tiles protected by the 1-4-7 / 2-5-8 / 3-6-9 suji relationship. If an opponent discarded a 4m, then 1m and 7m get suji safety (float 1.0) against that opponent. Suji only applies to suited tiles (indices 0-26); honors have no suji. Values update incrementally as new discards appear.
[DOC L0215] 
[DOC L0216] **Kabe** (wall block) marks tiles where all 4 copies are accounted for in visible information (discards, melds, own hand). A tile with all copies visible can't be part of any opponent's winning hand.
[DOC L0217] 
[DOC L0218] **One-chance** marks tiles where exactly 3 of 4 copies are visible, meaning only one unknown copy remains. These tiles carry reduced but nonzero risk.
[DOC L0219] 
[DOC L0220] All safety arrays update incrementally. When a new discard or meld occurs, only the affected opponent's `SafetyInfo` is recomputed.
[DOC L0221] 
[DOC L0222] ## Batch Simulator (`simulator.rs`)
[DOC L0223] 
[DOC L0224] ### BatchSimulator
[DOC L0225] 
[DOC L0226] `BatchSimulator` runs many games in parallel using a `rayon::ThreadPool`. Each game runs on its own thread with no shared mutable state between games.
[DOC L0227] 
[DOC L0228] ```rust
[DOC L0229] pub struct BatchSimulator {
[DOC L0230]     pool: rayon::ThreadPool,
[DOC L0231] }
[DOC L0232] ```
[DOC L0233] 
[DOC L0234] ### BatchConfig
[DOC L0235] 
[DOC L0236] ```rust
[DOC L0237] pub struct BatchConfig {
[DOC L0238]     pub num_games: usize,
[DOC L0239]     pub base_seed: Option<u64>,
[DOC L0240]     pub num_threads: Option<usize>,  // None = rayon default (num CPUs)
[DOC L0241]     pub game_mode: u8,               // 0 = hanchan, 1 = east only
[DOC L0242] }
[DOC L0243] ```
[DOC L0244] 
[DOC L0245] Each game derives its seed as `base_seed + game_index`. Two runs with the same `BatchConfig` produce identical results regardless of thread scheduling.
[DOC L0246] 
[DOC L0247] ### GameResult
[DOC L0248] 
[DOC L0249] `GameResult` collects the outcome of a single game: final scores for all four players, rounds played, total actions taken, and the seed used. The batch simulator returns a `Vec<GameResult>`.
[DOC L0250] 
[DOC L0251] ### Convenience Function
[DOC L0252] 
[DOC L0253] `run_batch_simple` is a free function that uses rayon's global thread pool instead of a dedicated one. It's the easiest entry point for scripts and benchmarks that don't need custom thread pool configuration.
[DOC L0254] 
[DOC L0255] ### Planned: Pre-Allocated Game Pools
[DOC L0256] 
[DOC L0257] Currently each game in a batch allocates a fresh `GameState`. A future optimization is to maintain a pool of pre-allocated game states that get recycled between batches, eliminating per-game allocation overhead during high-throughput self-play.
[DOC L0258] 
[DOC L0259] ## Seeding (`seeding.rs`)
[DOC L0260] 
[DOC L0261] Deterministic seeding is critical for reproducible training and evaluation. The seeding module provides a hierarchical RNG system.
[DOC L0262] 
[DOC L0263] ### Key Derivation
[DOC L0264] 
[DOC L0265] The session seed is a `[u8; 32]` byte array. `SessionRng` derives per-game seeds via `SHA-256(session_seed || game_index_le_bytes)`. The `derive_kyoku_seed` function further derives per-round seeds: `SHA-256(session_seed || nonce || kyoku || honba)`.
[DOC L0266] 
[DOC L0267] ```
[DOC L0268] game_seed = SHA-256(session_seed || game_index)[0..32]
[DOC L0269] ```
[DOC L0270] 
[DOC L0271] This ensures every game in a batch gets a unique, deterministic seed derived from the single session seed. Changing the session seed changes all games. Changing the game index changes only that game.
[DOC L0272] 
[DOC L0273] ### SessionRng
[DOC L0274] 
[DOC L0275] `SessionRng` holds a 32-byte seed and an auto-incrementing game index counter. Each call to `next_game_seed()` derives a new 32-byte seed and advances the counter. This gives 2^64 independent game seeds from a single session seed.
[DOC L0276] 
[DOC L0277] ### Wall Generation
[DOC L0278] 
[DOC L0279] `generate_wall` takes a session seed, nonce, kyoku number, and honba count. It derives a kyoku-specific seed, seeds a fresh `ChaCha8Rng`, initializes a sorted `[0..135]` wall, and applies a vendored Fisher-Yates shuffle. The vendored implementation avoids dependence on `rand::seq::SliceRandom` internals that might change between rand versions.
[DOC L0280] 
[DOC L0281] ### Determinism Guarantees
[DOC L0282] 
[DOC L0283] Given the same session seed and batch config, `hydra-core` produces bit-identical results across:
[DOC L0284] 
[DOC L0285] - Different runs on the same machine
[DOC L0286] - Different thread counts (rayon scheduling is deterministic per-game)
[DOC L0287] - Different platforms (x86_64, aarch64) thanks to the vendored shuffle
[DOC L0288] 
[DOC L0289] The only requirement is the same Rust toolchain version, since floating-point encoder output depends on compiler codegen.
[DOC L0290] 
[DOC L0291] ## Game Loop (`game_loop.rs`)
[DOC L0292] 
[DOC L0293] ### GameRunner
[DOC L0294] 
[DOC L0295] `GameRunner` orchestrates a single game from start to finish. It holds the riichienv `GameState`, a `[SafetyInfo; 4]` array (one per player perspective), and action/round counters.
[DOC L0296] 
[DOC L0297] The runner exposes two execution modes:
[DOC L0298] 
[DOC L0299] - `step_once(selector)`: advance the game by one step using the provided `ActionSelector`. Handles round transitions (auto-resets safety), WaitAct vs WaitResponse phases. Returns `false` when the game is over.
[DOC L0300] - `run_to_completion(selector)`: play an entire game by calling `step_once` in a loop. Provides accessor methods for `scores()`, `total_actions()`, `rounds_played()`, and `safety(player)` after completion.
[DOC L0301] 
[DOC L0302] ### ActionSelector Trait
[DOC L0303] 
[DOC L0304] ```rust
[DOC L0305] pub trait ActionSelector {
[DOC L0306]     fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action;
[DOC L0307] }
[DOC L0308] ```
[DOC L0309] 
[DOC L0310] Any type implementing `ActionSelector` can drive the game loop. `FirstActionSelector` is a simple built-in that picks the first legal action (useful for testing and benchmarks). The training pipeline provides its own selectors that call the neural network.
[DOC L0311] 
[DOC L0312] ### Safety Tracking
[DOC L0313] 
[DOC L0314] During play, `GameRunner` maintains a `[SafetyInfo; 4]` array (one per player perspective). After every discard, call, and riichi event, the runner's `track_action` method incrementally updates the relevant safety data across all perspectives. These feed directly into the encoder's safety channels (62-84) on the next observation request.
[DOC L0315] 
[DOC L0316] ## Bridge (`bridge.rs`)
[DOC L0317] 
[DOC L0318] The bridge module converts riichienv's `Observation` struct into the data the encoder needs. It acts as a translation layer so the encoder doesn't depend on riichienv types directly.
[DOC L0319] 
[DOC L0320] ### Extract Functions
[DOC L0321] 
[DOC L0322] Each `extract_*` function pulls one category of data from the riichienv observation:
[DOC L0323] 
[DOC L0324] - `extract_hand()`: closed hand tile counts and open meld tile counts
[DOC L0325] - `extract_discards()`: per-player discard sequences with tedashi and temporal info
[DOC L0326] - `extract_melds()`: per-player meld data (chi/pon/kan tile lists)
[DOC L0327] - `extract_dora()`: dora indicator tiles and aka-dora visibility
[DOC L0328] - `extract_metadata()`: scores, round number, honba, kyotaku, riichi states, shanten
[DOC L0329] 
[DOC L0330] ### Entry Point
[DOC L0331] 
[DOC L0332] `encode_observation` is the main bridge function. It takes a riichienv `Observation`, calls all `extract_*` functions, and feeds the results into the `ObservationEncoder`. Returns the filled 192x34 fixed-superset float buffer ready for the model.
[DOC L0333] 
[DOC L0334] Current runtime-status note: the bridge now carries two live Hand-EV paths on the same fixed surface. By default it computes Hand-EV from public remaining counts. When search context supplies a CT-SMC posterior, it upgrades that path to use wall-weighted remaining counts from the posterior while keeping the same encoder/runtime interface. The same bridge surface also populates fixed-shape Group C search/belief planes from live mixture/search/risk context when those signals are present and falls back to zero-filled planes plus presence masks otherwise. This file records that runtime reality only; promoted sequencing/doctrine still lives in `research/design/HYDRA_FINAL.md` and `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0335] 
[DOC L0336] ## Testing
[DOC L0337] 
[DOC L0338] Every module in `hydra-core` has inline unit tests (`#[cfg(test)]` modules). Beyond unit tests, the `tests/` directory contains integration tests:
[DOC L0339] 
[DOC L0340] | Test File | What It Covers |
[DOC L0341] |-----------|---------------|
[DOC L0342] | `golden_encoder.rs` | Regression tests for the encoder. Compares encoder output against saved golden snapshots. Catches silent encoding drift when any channel logic changes. |
[DOC L0343] | `mjai_replay.rs` | Replays recorded MJAI game logs through the engine and verifies that game state, actions, and observations match the expected sequence. The current regression surface explicitly covers replay round-reset correctness and kan-action legality matching so MJAI replay stays aligned with runtime legality checks. |
[DOC L0344] | `proptest_invariants.rs` | Property-based tests using `proptest`. Generates random game states and verifies invariants: legal mask consistency, encoder channel bounds, tile count conservation, action round-trip fidelity. |
[DOC L0345] | `game_loop_integration.rs` | End-to-end game loop tests. Runs complete games with `FirstActionSelector` and verifies termination, score consistency, and result collection. |
[DOC L0346] 
[DOC L0347] Current replay-status note: after fixing MJAI replay round-start reset semantics and kan replay matching in the vendored engine layer, the Hydra MJAI loader was re-audited against the Tenhou Houou 2025 corpus (`178,897` files) with `0` skips.
[DOC L0348] 
[DOC L0349] ### Benchmarks
[DOC L0350] 
[DOC L0351] The `benches/` directory uses `criterion` for performance benchmarks:
[DOC L0352] 
[DOC L0353] - `single_game`: time to run one complete game from start to finish
[DOC L0354] - `batch_100`: time to run 100 games in parallel with `BatchSimulator`
[DOC L0355] - `encode_observation_1000x`: time to encode 1,000 observations (measuring encoder throughput)
[DOC L0356] 
[DOC L0357] Run benchmarks with `cargo bench`.
[DOC L0358] 
[DOC L0359] ## Dependencies
[DOC L0360] 
[DOC L0361] ### Runtime
[DOC L0362] 
[DOC L0363] | Crate | Purpose |
[DOC L0364] |-------|---------|
[DOC L0365] | `riichienv-core` | Game engine (rules, state, legality) |
[DOC L0366] | `rayon` | Work-stealing thread pool for parallel batch simulation |
[DOC L0367] | `serde` | Serialization for configs, game results, replay data |
[DOC L0368] | `ndarray` | N-dimensional array operations for observation tensors |
[DOC L0369] | `serde_json` | JSON serialization for MJAI protocol data |
[DOC L0370] | `chacha20` | ChaCha20 cipher (pinned version for determinism) |
[DOC L0371] | `rand` | RNG traits and distributions |
[DOC L0372] | `rand_chacha` | ChaCha8Rng for deterministic seeding |
[DOC L0373] | `sha2` | SHA-256 hashing for seed key derivation |
[DOC L0374] | `anyhow` | Application-level error handling |
[DOC L0375] | `thiserror` | Derive macro for library error enums |
[DOC L0376] 
[DOC L0377] ### Dev / Test
[DOC L0378] 
[DOC L0379] | Crate | Purpose |
[DOC L0380] |-------|---------|
[DOC L0381] | `proptest` | Property-based testing framework |
[DOC L0382] | `criterion` | Benchmarking framework |
[DOC L0383] 
[DOC L0384] ## License
[DOC L0385] 
[DOC L0386] hydra-core is BSL-1.1 (see [hydra-core/LICENSE](../crates/hydra-core/LICENSE)). hydra-engine is Apache-2.0 (vendored upstream). All dependencies use MIT, Apache-2.0, or BSD-compatible licenses.
```

## Artifact 06 — Compatibility-sensitive runtime surface
Artifact id: `compatibility-surface`
Source label: DOC
Type: `file_full`
Source: `docs/COMPATIBILITY_SURFACE.md`
Why it matters: Short but important live-surface reminder showing what compatibility contracts already constrain Hydra's real build path.

```markdown
[DOC L0001] # Hydra Compatibility Surface
[DOC L0002] 
[DOC L0003] Compact compatibility contract for agents and developers touching runtime, training, or model-shape-sensitive code.
[DOC L0004] 
[DOC L0005] If you change any row in this file, you should assume matching docs, tests, and consumers need review.
[DOC L0006] 
[DOC L0007] Primary runtime owner: `docs/GAME_ENGINE.md`
[DOC L0008] 
[DOC L0009] ## Compatibility table
[DOC L0010] 
[DOC L0011] | Surface | Current contract | Owner / source of truth | Notes |
[DOC L0012] |---|---|---|---|
[DOC L0013] | Encoder/model input shape | `192x34` | `docs/GAME_ENGINE.md`, `hydra-core/src/encoder.rs` | Live full contract |
[DOC L0014] | Baseline prefix | `85x34` (`channels 0..84`) | `docs/GAME_ENGINE.md` | Historical baseline-prefix only; not the full live encoder |
[DOC L0015] | Action space | `46` actions | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Mortal-compatible action indexing |
[DOC L0016] | Riichi handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare riichi, then choose discard |
[DOC L0017] | Kan handling | two-phase | `hydra-core/src/action.rs`, `docs/GAME_ENGINE.md` | Declare kan, then choose specific kan when needed |
[DOC L0018] | Tile kind indices | `0..33` normalized tile indices | `hydra-core/src/tile.rs`, `docs/GAME_ENGINE.md` | 34 tile kinds |
[DOC L0019] | Aka tile behavior | aka tiles stay distinct in 136-format/action handling where needed | `hydra-core/src/tile.rs`, `hydra-core/src/action.rs` | Red 5m/5p/5s remain special cases |
[DOC L0020] | Legal action mask shape | `[bool; 46]` | `hydra-core/src/action.rs` | Training/inference must agree on mask semantics |
[DOC L0021] | Runtime/train entrypoint | `crates/hydra-train/src/bin/train.rs` | root `AGENTS.md`, crate docs | Main train binary entry surface |
[DOC L0022] | BC selected-runtime authority | fresh run = config-derived; epoch-boundary resume may reuse matching preflight-selected runtime; partial-epoch resume requires identical runtime | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/resume.rs` | Applies only to selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, `accum_steps`) |
[DOC L0023] | BC loader-runtime authority | config-derived | `crates/hydra-train/src/bin/train/bootstrap.rs`, `crates/hydra-train/src/bin/train/config_runtime.rs` | Matching BC preflight cache does not make loader-runtime authoritative |
[DOC L0024] | Runtime truth on drift | current code wins | `docs/GAME_ENGINE.md`, root `AGENTS.md` | Docs are compatibility aids, not stronger than code |
[DOC L0025] 
[DOC L0026] ## Crate ownership quick reference
[DOC L0027] 
[DOC L0028] | Crate | Owns |
[DOC L0029] |---|---|
[DOC L0030] | `crates/hydra-engine` | vendored rules engine behavior |
[DOC L0031] | `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing |
[DOC L0032] | `crates/hydra-train` | model, targets, losses, training/inference orchestration |
[DOC L0033] 
[DOC L0034] ## Read next
[DOC L0035] 
[DOC L0036] - Need the full runtime explanation? Read `docs/GAME_ENGINE.md`.
[DOC L0037] - Need the repo routing / trust map? Read `README.md`.
[DOC L0038] - Need active-vs-staged status? Read `research/design/HYDRA_RECONCILIATION.md` and `docs/CURRENT_STATUS.md`.
```

## Artifact 07 — Promoted Hydra architecture doctrine
Artifact id: `hydra-final`
Source label: DESIGN
Type: `file_full`
Source: `research/design/HYDRA_FINAL.md`
Why it matters: Primary promoted architecture doctrine. High-value evidence for Hydra's current framing around ExIt, search-as-feature, CTDE, oracle critics, belief/search features, and training priorities.

```markdown
[DESIGN L0001] # HYDRA: A Maximum-Ceiling 4-Player Riichi Mahjong AI
[DESIGN L0002] 
[DESIGN L0003] **Promoted architecture doctrine summary.** This document is Hydra's architecture north star after filtering the canonical archive SSOT through current repo/code validation. It supersedes the two prior internal variants: the throughput-first "compute-constrained elegance" plan and the "information-geometric / all-out" plan. Hydra keeps their best ideas, removes their ceilings, and adds a rigorously grounded robustness layer.
[DESIGN L0004] 
[DESIGN L0005] This file owns the target architecture, not the full live repo status board. For current shipped/staged status, read `docs/CURRENT_STATUS.md`. For active-path / staged-vs-reserve execution decisions, read `research/design/HYDRA_RECONCILIATION.md`. For runtime compatibility/runtime reality, read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
[DESIGN L0006] 
[DESIGN L0007] ---
[DESIGN L0008] 
[DESIGN L0009] ## 0. Abstract
[DESIGN L0010] 
[DESIGN L0011] 4-player Riichi Mahjong is a large, general-sum, imperfect-information game with a **finite shared hidden pool** (multivariate hypergeometric), **hard conservation constraints**, and **decision-critical correlations** that strengthen late game.
[DESIGN L0012] 
[DESIGN L0013] Hydra is built around one central engine:
[DESIGN L0014] 
[DESIGN L0015] > **ExIt + Pondering + Search-as-Feature (SaF)**
[DESIGN L0016] > Deep anytime belief-search generates training targets continuously during self-play, amplified by opponent-turn idle time; those targets are amortized back into the policy/value networks so inference remains fast.
[DESIGN L0017] 
[DESIGN L0018] The system couples this engine with:
[DESIGN L0019] 
[DESIGN L0020] 1. **Belief correctness with constraints**: SIB / Mixture-SIB (Sinkhorn KL projection) + **CT-SMC exact contingency-table sampler** exploiting Mahjong's small row counts ($r \le 4$) for correlation-faithful beliefs via a 3,375-state DP (~4M ops, <1ms in Rust).
[DESIGN L0021] 2. **Anytime Factored-Belief Search (AFBS)**: top-k pruning, heavy caching, incremental reuse, predictive pondering, and **endgame exactification** (exact chance enumeration when wall $\le 10$).
[DESIGN L0022] 3. **Robust opponent modeling inside search**: opponent nodes solved as distributionally robust soft-min within a KL uncertainty set around the learned opponent policy.
[DESIGN L0023] 4. **Conservative safety math that is tight enough to matter**: Negative dependence / Strongly Rayleigh + Hunter/Kounias union tightening + bounded-error Monte Carlo intersections.
[DESIGN L0024] 5. **Hand-EV oracle features**: CPU-precomputed per-discard tenpai probability, win probability, expected score, and ukeire -- proven by Suphx as their biggest practical win.
[DESIGN L0025] 6. **ACH training** (Actor-Critic Hedge, LuckyJ's algorithm): +0.4 fan over PPO via Hedge-derived conservative clipping. Global $\eta$, per-(s,a) gating, standard GAE, one epoch per batch. Compatible with oracle guiding via CTDE.
[DESIGN L0026] 7. **Two-tier network** (12-block actor / 24-block learner): 40-block teacher data-starved at 7 spp on hard states only. 24-block learner (245 spp) handles both training and deep AFBS. Continuous distillation learner -> actor.
[DESIGN L0027] 
[DESIGN L0028] Goal: **maximize expected Tenhou stable rank**; LuckyJ's 10.68 stable dan is the current public benchmark.
[DESIGN L0029] 
[DESIGN L0030] ---
[DESIGN L0031] 
[DESIGN L0032] ## 1. Design principles
[DESIGN L0033] 
[DESIGN L0034] ### P1. Ceiling first, then amortize
[DESIGN L0035] If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, deep search, offline solvers, or distillation targets -- not in the critical inference loop.
[DESIGN L0036] 
[DESIGN L0037] ### P2. Search targets must optimize the information state, not the hidden state
[DESIGN L0038] Any training target used to update the deployable policy must be a function of the public/information state, not privileged knowledge. We allow perfect-information networks for variance reduction and diagnostics, but the improvement operator must respect the information constraints.
[DESIGN L0039] 
[DESIGN L0040] ### P3. Every "guarantee-like" claim must be either a theorem (with conditions), a bound (with explicit constants), or an empirical gate with a measurable pass/fail threshold.
[DESIGN L0041] 
[DESIGN L0042] ### P4. Robustness is not optional in 4-player general-sum
[DESIGN L0043] Instead of equilibrium-style guarantees (which do not cleanly extend to 4p), we use distributional robustness: robust to belief error, robust to opponent policy misspecification, robust to population shifts.
[DESIGN L0044] 
[DESIGN L0045] ---
[DESIGN L0046] 
[DESIGN L0047] ## 2. Game model and notation
[DESIGN L0048] 
[DESIGN L0049] - Tile types: $k \in \{1,\dots,34\}$, multiplicity 4, total 136 tiles.
[DESIGN L0050] - Hidden locations: $z \in \{1,2,3,W\}$: three opponent concealed hands + wall remainder.
[DESIGN L0051] - Public information state at time $t$: $I_t$ (our hand, discards/melds, riichi, dora, scores, round meta).
[DESIGN L0052] - Remaining tile counts: $r_t(k) = 4 - \mathrm{visible}_t(k)$.
[DESIGN L0053] - Hidden location sizes: $s_t(z) \in \mathbb{Z}_{\ge 0}$, $\sum_z s_t(z) = \sum_k r_t(k)$.
[DESIGN L0054] - Hidden allocation matrix: $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$, $\sum_z X_t(k,z) = r_t(k)$, $\sum_k X_t(k,z)=s_t(z)$.
[DESIGN L0055] 
[DESIGN L0056] Under purely random dealing, $X_t$ is multivariate hypergeometric; under strategic play, $p(X_t\mid I_t)$ is shaped by action likelihoods.
[DESIGN L0057] 
[DESIGN L0058] ---
[DESIGN L0059] 
[DESIGN L0060] ## 3. System overview -- four interacting loops
[DESIGN L0061] 
[DESIGN L0062] **Loop A: Belief loop** -- Mixture-SIB for fast marginal updates under constraints, particle SMC for joint correlation capture.
[DESIGN L0063] 
[DESIGN L0064] **Loop B: Search loop** -- AFBS on $I_t$ with belief $q_t$: on-turn (shallow, feature-producing), off-turn/pondering (deep, cached, predictive).
[DESIGN L0065] 
[DESIGN L0066] **Loop C: Distillation loop** -- Train policy/value to predict $\pi^{\text{ExIt}}$, $V^{\text{ExIt}}$, and calibrated safety features.
[DESIGN L0067] 
[DESIGN L0068] **Loop D: Population loop** -- League with self-play variants, human-style anchors, adversarial exploiters.
[DESIGN L0069] 
[DESIGN L0070] ---
[DESIGN L0071] 
[DESIGN L0072] ## 4. Neural architecture
[DESIGN L0073] 
[DESIGN L0074] ### 4.1 Input tensor
[DESIGN L0075] 
[DESIGN L0076] **Group A -- Public encoding (~80-120 planes):** Hand, ordered discards (recency), open melds, riichi state, dora, round/scoring context, shanten/uke-ire summaries.
[DESIGN L0077] 
[DESIGN L0078] **Group B -- Safety planes (~23 planes):** Tenpai hints, furiten, genbutsu/suji/kabe safe-tile masks.
[DESIGN L0079] 
[DESIGN L0080] **Group C -- Search and belief features (dynamic, ~60-200 planes):** Belief marginals $B_t(k,z)$, mixture weights/entropy/ESS, AFBS action deltas $\Delta Q(a)$, risk estimates, robust opponent stress indicators. Zeroed with presence mask when unavailable.
[DESIGN L0081] 
[DESIGN L0082] **Group D -- Hand-EV oracle features (~34-68 planes, CPU-precomputed):** For each discard candidate $a$ (34 tile types), pre-compute look-ahead analysis on the existing 42-plane interface:
[DESIGN L0083] - $P_{\text{tenpai}}^{(d)}(a)$: probability of reaching tenpai within $d \in \{1,2,3\}$ self-draws.
[DESIGN L0084] - $P_{\text{win}}^{(d)}(a)$: probability of winning within $d$ draws (tsumo + simplified ron model).
[DESIGN L0085] - $\mathbb{E}[\text{score} \mid \text{win}, a]$: expected hand value (han/fu/score) if we win after discarding $a$.
[DESIGN L0086] - Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.
[DESIGN L0087] 
[DESIGN L0088] These features are computed by the CPU-side hand analyzer (`shanten_batch.rs` + scoring engine) using belief-weighted remaining tile counts from CT-SMC. Zero GPU cost -- CPU pre-computes during game step processing. Suphx reported these look-ahead features as their single biggest practical improvement (Li et al. 2020).
[DESIGN L0089] 
[DESIGN L0090] Runtime reality note: the live repo already carries the same 42-plane Hand-EV surface through `HandEvFeatures`, bridge code, and encoder channels. Runtime bridge code uses public remaining counts by default and CT-SMC wall-weighted remaining counts when search context is present. For shipped/staged status of that surface, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md`.
[DESIGN L0091] 
[DESIGN L0092] ### 4.2 Two-tier architecture
[DESIGN L0093] 
[DESIGN L0094] **Why not monolithic 40-block?** At 2000 GPU hours, self-play generates ~2.45B decisions (35M games). Samples-per-parameter ratio:
[DESIGN L0095] 
[DESIGN L0096] | Config | Params | Samples/param | vs Mortal (514) | Verdict |
[DESIGN L0097] |--------|-------:|-------------:|----------------:|---------|
[DESIGN L0098] | 40-block mono | 16.5M | 148 | 0.29x | Undertrained AND too slow for rollouts |
[DESIGN L0099] | 24-block | 10M | 245 | 0.48x | Viable with ExIt quality boost |
[DESIGN L0100] | 12-block | 5M | 490 | 0.95x | Well-trained, fast inference |
[DESIGN L0101] 
[DESIGN L0102] (Based on ~35M games * 70 decisions = 2.45B total samples.)
[DESIGN L0103] 
[DESIGN L0104] A 40-block teacher trained only on hard states (1-5%) gets just ~7 spp -- catastrophic data starvation. **Two-tier architecture avoids this paradox:**
[DESIGN L0105] 
[DESIGN L0106] | Network | Blocks | Params | Role | Runtime placement |
[DESIGN L0107] |---------|-------:|-------:|------|-------------------|
[DESIGN L0108] | **LearnerNet** | 24 | ~10M | Training (ACH/ExIt) + deep AFBS on hard positions | Main Delta A100 training resources |
[DESIGN L0109] | **ActorNet** | 12 | ~5M | Self-play data generation + shallow SaF features | Fast rollout / self-play generation resources |
[DESIGN L0110] 
[DESIGN L0111] All use SE-ResNet with GroupNorm(32) and Mish. Target deployment precision is bf16-capable, but the current repo remains fp32-first unless backend autocast is wired explicitly. **Continuous distillation**: Learner -> Actor (every 1-2 minutes, IMPALA-style). ActorNet inference: ~0.2ms. LearnerNet inference: ~0.35ms. LearnerNet runs deeper AFBS only on hard-position ExIt labels when throughput budget allows.
[DESIGN L0112] 
[DESIGN L0113] ### 4.3 Heads (multi-task)
[DESIGN L0114] 
[DESIGN L0115] **Core decision heads:** (1) Policy $\pi_\theta(a\mid I_t)$, 46 actions. (2) Value $V_\theta(I_t)$, scalar. (3) Score distribution: pdf + cdf (64 bins, KataGo-style).
[DESIGN L0116] 
[DESIGN L0117] **Opponent and safety heads:** (4) Opponent tenpai (3 sigmoids). (5) Opponent next discard (3x34). (6) Danger: per-tile deal-in probability (3x34).
[DESIGN L0118] 
[DESIGN L0119] **Belief heads:** (7) Mixture-SIB external fields $F_\theta^{(\ell)}(k,z)$ and mixture weight logits. (8) Opponent hand-type latent predictor.
[DESIGN L0120] 
[DESIGN L0121] **Search distillation heads:** (9) $\Delta Q$ regression (predict search advantage over baseline). (10) Safety bound residual (predict conservatism gap).
[DESIGN L0122] 
[DESIGN L0123] Runtime reality note: the live model already exposes these advanced output families structurally in one output contract (`belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`). For which of those surfaces are shipped baseline vs implemented-but-staged vs implemented-but-not-default-on, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md` rather than treating this architecture file as the status owner.
[DESIGN L0124] 
[DESIGN L0125] ---
[DESIGN L0126] 
[DESIGN L0127] ## 5. Belief inference: SIB, Mixture-SIB, and particle posterior
[DESIGN L0128] 
[DESIGN L0129] ### 5.1 SIB as KL projection
[DESIGN L0130] 
[DESIGN L0131] Let
[DESIGN L0132] 
[DESIGN L0133] $$K_\theta(k,z)=\exp(F_\theta(k,z))>0$$
[DESIGN L0134] 
[DESIGN L0135] The transportation polytope is
[DESIGN L0136] 
[DESIGN L0137] $$\mathcal{U}(r_t,s_t)=\{B\ge 0: B\mathbf{1}=r_t, B^\top \mathbf{1}=s_t\}$$
[DESIGN L0138] 
[DESIGN L0139] **SIB operator:**
[DESIGN L0140] 
[DESIGN L0141] $$\mathrm{SIB}(K_\theta;r_t,s_t) := \arg\min_{B\in\mathcal{U}} D_{\mathrm{KL}}(B\|K_\theta)$$
[DESIGN L0142] 
[DESIGN L0143] Sinkhorn-Knopp gives the solution
[DESIGN L0144] 
[DESIGN L0145] $$B^*=\mathrm{diag}(u)\cdot K_\theta\cdot\mathrm{diag}(v)$$
[DESIGN L0146] 
[DESIGN L0147] ### 5.2 Mixture-SIB for multimodality
[DESIGN L0148] 
[DESIGN L0149] With $L$ components, the mixture posterior is
[DESIGN L0150] 
[DESIGN L0151] $$q_t(X)=\sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X)$$
[DESIGN L0152] 
[DESIGN L0153] Each component marginal is
[DESIGN L0154] 
[DESIGN L0155] $$B_t^{(\ell)}=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t)$$
[DESIGN L0156] 
[DESIGN L0157] Weight update (Bayes):
[DESIGN L0158] 
[DESIGN L0159] $$w_{t+1}^{(\ell)}\propto w_t^{(\ell)} \cdot p_\phi(e_t\mid I_t, B_t^{(\ell)}, \ell)$$
[DESIGN L0160] 
[DESIGN L0161] Here, $e_t$ is the observed public event (opponent discard, call, riichi, or pass). Anti-collapse via entropy regularizer, split-merge on low ESS, and a diversity penalty between components.
[DESIGN L0162] 
[DESIGN L0163] ### 5.3 Particle posterior (SMC) for joint structure
[DESIGN L0164] 
[DESIGN L0165] Particles $\{X_t^{(p)},\alpha_t^{(p)}\}_{p=1}^P$ targeting $p(X_t\mid I_t)$. Proposal via constrained sequential fill guided by mixture component. Resample when $\mathrm{ESS}<0.4P$. Rejuvenation via Metropolis-Hastings swap moves preserving row/col sums.
[DESIGN L0166] 
[DESIGN L0167] ### 5.4 Correlation scale diagnostic
[DESIGN L0168] 
[DESIGN L0169] The correlation scale is
[DESIGN L0170] 
[DESIGN L0171] $$|\rho_{ij}|=\sqrt{K_i K_j} / \sqrt{(H-K_i)(H-K_j)}$$
[DESIGN L0172] 
[DESIGN L0173] At $H=50$ and $K=4$, this gives $|\rho|=4/46=0.087$. At $H=25$, it gives $|\rho|=0.190$. Late-game correlations motivate Mixture-SIB plus particles over first-moment alone.
[DESIGN L0174] 
[DESIGN L0175] ### 5.5 CT-SMC: Exact contingency-table sampling (replaces generic particle proposals)
[DESIGN L0176] 
[DESIGN L0177] The hidden allocation $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$ is a **fixed-margin contingency table**. Key Mahjong insight: each row sum $r_t(k) \le 4$, so per-row compositions are tiny ($\binom{r+3}{3} \le 35$).
[DESIGN L0178] 
[DESIGN L0179] **Exact DP partition function.** Order tile types $k=1,\dots,34$. Let residual capacities be $\mathbf{c}=(c_1,c_2,c_3,c_W)$. Define:
[DESIGN L0180] 
[DESIGN L0181] $$Z_k(\mathbf{c}) = \sum_{x \in \mathcal{X}_k(\mathbf{c})} \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x), \quad Z_{35}(\mathbf{0})=1$$
[DESIGN L0182] 
[DESIGN L0183] The learned field weight for each row is
[DESIGN L0184] 
[DESIGN L0185] $$\phi_k(x)=\prod_j \omega_{kj}^{x_j}$$
[DESIGN L0186] 
[DESIGN L0187] The wall residual is derived from the other capacities:
[DESIGN L0188] 
[DESIGN L0189] $$c_W = R_k - (c_1+c_2+c_3)$$
[DESIGN L0190] 
[DESIGN L0191] Here, $R_k = \sum_{t \ge k} r_t$ is the remaining hidden tile count at DP step $k$. So the DP state is 3D: $(c_1,c_2,c_3)$. State count: $\le (15)^3 = 3{,}375$ (max 14 tiles after draw, before discard). Each transition enumerates $\le 35$ compositions. Total: $\sim 34 \times 3375 \times 35 \approx 4.0M$ ops -- **trivially sub-millisecond in Rust**. Use log-space DP for numerical stability.
[DESIGN L0192] 
[DESIGN L0193] **Exact backward sampling:**
[DESIGN L0194] 
[DESIGN L0195] $$p(x_k = x \mid \mathbf{c}) = \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x) / Z_k(\mathbf{c})$$
[DESIGN L0196] 
[DESIGN L0197] This gives **exact samples with correct correlations** from the conservation-constrained distribution -- not mean-field approximations.
[DESIGN L0198] 
[DESIGN L0199] **SMC integration.** The full posterior is
[DESIGN L0200] 
[DESIGN L0201] $$p(X \mid \mathcal{O}_{1:t}) \propto p_0(X) \cdot L(X)$$
[DESIGN L0202] 
[DESIGN L0203] Here, $L(X)$ is the opponent action likelihood. Sample $X^{(n)} \sim p_0$ via CT-DP (fast, correlation-correct), then assign weights with
[DESIGN L0204] 
[DESIGN L0205] $$w^{(n)} \leftarrow L(X^{(n)})$$
[DESIGN L0206] 
[DESIGN L0207] Normalize and resample. The proposal already respects the hardest constraint (tile conservation) exactly, so ESS stays high.
[DESIGN L0208] 
[DESIGN L0209] **What CT-SMC replaces:** The generic particle proposal from Section 5.3. Mixture-SIB is KEPT as the fast amortized belief head for network input; CT-SMC is the search-grade belief for AFBS and safety queries.
[DESIGN L0210] 
[DESIGN L0211] **Validation gates:**
[DESIGN L0212] - **Gate A (posterior log-likelihood):** At end of hand, evaluate $\log p(X^* \mid \mathcal{O}_{1:t})$ under CT-SMC vs generic CMPS. CT-SMC must win.
[DESIGN L0213] - **Gate B (pairwise MI calibration):** Compare the estimated mutual information between whether tile $A$ is in hidden hand $z$ and whether tile $B$ is in hidden hand $z$ against empirical values. It must capture correlations generic CMPS misses.
[DESIGN L0214] 
[DESIGN L0215] ---
[DESIGN L0216] 
[DESIGN L0217] ## 6. Conservative safety estimates without over-folding
[DESIGN L0218] 
[DESIGN L0219] ### 6.1 Strongly Rayleigh / negative dependence foundations
[DESIGN L0220] 
[DESIGN L0221] The remaining-tile distribution under "draw without replacement" is Strongly Rayleigh (BBL 2009), implying strong negative dependence. Used only for bounding monotone danger events.
[DESIGN L0222] 
[DESIGN L0223] ### 6.2 Hunter bound (spanning tree correction)
[DESIGN L0224] 
[DESIGN L0225] For threat events $A_1, \ldots, A_J$ and any spanning tree $T$:
[DESIGN L0226] 
[DESIGN L0227] $$P\left(\bigcup_{j=1}^{J} A_j\right) \le \sum_{j=1}^{J} P(A_j) - \sum_{(u,v)\in T} P(A_u \cap A_v)$$
[DESIGN L0228] 
[DESIGN L0229] Maximum-weight spanning tree gives the tightest bound. Kounias (1968) bound is a member; we take the minimum computable bound.
[DESIGN L0230] 
[DESIGN L0231] ### 6.3 Computing intersections reliably
[DESIGN L0232] 
[DESIGN L0233] Analytic formulas for simple events; particle estimates with Hoeffding CIs otherwise. Never use an intersection estimate unless CI half-width $<\delta_\cap$ (e.g., 0.01). Fall back to conservative Boole if CI not met.
[DESIGN L0234] 
[DESIGN L0235] ---
[DESIGN L0236] 
[DESIGN L0237] ## 7. Anytime Factored Belief Search (AFBS)
[DESIGN L0238] 
[DESIGN L0239] ### 7.1 Tree structure
[DESIGN L0240] 
[DESIGN L0241] Node state: $(I, \mathcal{B}, \mathcal{P})$ -- info state, Mixture-SIB summary, particle set handle.
[DESIGN L0242] 
[DESIGN L0243] ### 7.2 Beam parameters
[DESIGN L0244] 
[DESIGN L0245] | Mode | Beam W | Depth D | Particles P | Mixture L |
[DESIGN L0246] |------|-------:|--------:|------------:|----------:|
[DESIGN L0247] | On-turn | 64-128 | 4-6 | 128-256 | 4-8 |
[DESIGN L0248] | Ponder | 256-1024 | 10-14 | 1024-4096 | 8-32 |
[DESIGN L0249] 
[DESIGN L0250] ### 7.3 Caches
[DESIGN L0251] 
[DESIGN L0252] Transposition table (public hash + belief signature), neural eval cache (batched GPU, LRU), Sinkhorn warm-start cache (u,v scalings), predictive ponder cache (subtrees for top-M predicted opponent actions).
[DESIGN L0253] 
[DESIGN L0254] ### 7.4 Incremental reuse across turns
[DESIGN L0255] 
[DESIGN L0256] On event: lookup predicted child key; if match, shift root and keep statistics; else reuse TT/NN cache and rebuild shallow frontier.
[DESIGN L0257] 
[DESIGN L0258] ### 7.5 Endgame exactification (wall-small solver)
[DESIGN L0259] 
[DESIGN L0260] Runtime reality note: the live repo currently implements a selective particle-weighted PIMC shell for this area rather than a full exact multiplayer endgame solver. Keep exactification as the target direction; defer current shipped/staged status to `docs/CURRENT_STATUS.md` and runtime semantics to `docs/GAME_ENGINE.md`.
[DESIGN L0261] 
[DESIGN L0262] **Trigger:** Activate when the remaining wall is 10 tiles or fewer and at least one threatening signal is present (riichi, open tenpai, high-tempo opponent).
[DESIGN L0263] 
[DESIGN L0264] **PIMC with top-k draw pruning.** Full Expectimax over wall=10 is too slow (~661K paths per particle at 0.1ms each = 66s). Instead, use **Pure PIMC**: for each CT-SMC particle, sample ONE draw sequence (weighted by hypergeometric probabilities) and ONE opponent action sequence (from ActorNet policy). Average over P particles. This reduces to P forward passes per endgame evaluation. With top-mass particle reduction (keep particles covering 95% weight, typically P=50-100): **5-10ms per decision**, well within budget. Top-k draw pruning (branch only on the 2-3 most likely draws at our nodes) provides a middle ground between PIMC and full Expectimax when more precision is needed.
[DESIGN L0265] 
[DESIGN L0266] $$Q(a) \approx \frac{1}{P}\sum_{p=1}^{P} PIMC(a \mid X^{(p)})$$
[DESIGN L0267] 
[DESIGN L0268] The inner value is exact over wall draws; opponent actions remain modeled by the robust policy (KL ball). This removes chance uncertainty variance at the most sensitive game phase (oorasu placement swings).
[DESIGN L0269] 
[DESIGN L0270] **Caching.** Late-game states repeat structurally across particles. Cache by: our hand canonicalization + remaining wall multiset signature (34-count vector) + riichi state + turn index. DP results reused heavily.
[DESIGN L0271] 
[DESIGN L0272] **Why this matters:** Late-game decisions are disproportionately high-EV. A single wrong fold or push in oorasu can flip placement from 1st to 4th (~90,000 point swing in uma). Exact computation eliminates the approximation error precisely where it's most costly.
[DESIGN L0273] 
[DESIGN L0274] **Validation gate:** Collect 50K endgame positions (last 10 draws). Compare deal-in rate, win conversion rate, and placement swings between standard AFBS vs endgame-exact mode. Endgame mode must improve all three.
[DESIGN L0275] 
[DESIGN L0276] ---
[DESIGN L0277] 
[DESIGN L0278] ## 8. Robust opponent modeling inside search
[DESIGN L0279] 
[DESIGN L0280] ### 8.1 Opponent uncertainty set
[DESIGN L0281] 
[DESIGN L0282] Learned opponent policy is $p(a)$. The true policy $q(a)$ lies in the KL ball
[DESIGN L0283] 
[DESIGN L0284] $$\mathcal{Q}_\varepsilon(p)=\{q: D_{\mathrm{KL}}(q\|p)\le \varepsilon\}$$
[DESIGN L0285] 
[DESIGN L0286] $\varepsilon$ is calibrated from data as the empirical upper quantile of observed KL, bucketed by context.
[DESIGN L0287] 
[DESIGN L0288] ### 8.2 Robust value at opponent nodes
[DESIGN L0289] 
[DESIGN L0290] Robust value at opponent nodes is
[DESIGN L0291] 
[DESIGN L0292] $$V_{\text{rob}}=\min_{q\in \mathcal{Q}_\varepsilon(p)} \sum_a q(a) Q(a)$$
[DESIGN L0293] 
[DESIGN L0294] The solution has the form
[DESIGN L0295] 
[DESIGN L0296] $$q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)$$
[DESIGN L0297] 
[DESIGN L0298] Choose $\tau$ so that $D_{\mathrm{KL}}(q_\tau\|p)=\varepsilon$.
[DESIGN L0299] 
[DESIGN L0300] **Contract.** For any opponent policy $q$ in the KL ball, AFBS's robust backup gives a lower bound on expected value against $q$.
[DESIGN L0301] 
[DESIGN L0302] ### 8.3 OLSS-style opponent strategy set
[DESIGN L0303] 
[DESIGN L0304] In addition to continuous KL robustness, maintain $N$ discrete opponent archetypes $\{\sigma_1,\dots,\sigma_N\}$ (e.g., aggressive/defensive/speed/value, $N=4$). At opponent nodes, evaluate:
[DESIGN L0305] $$Q(a) = -\tau_{\text{arch}} \log \sum_{i=1}^N w_i \exp(-Q^{\sigma_i}(a)/\tau_{\text{arch}})$$
[DESIGN L0306] 
[DESIGN L0307] where $w_i$ are archetype weights (uniform $1/N$ initially, updated by posterior over opponent type) and $\tau_{\text{arch}}$ is the archetype soft-min temperature (distinct from Section 8.2's $\tau$ found by binary search).
[DESIGN L0308] 
[DESIGN L0309] This soft-min over archetypes directly mirrors LuckyJ's OLSS-II approach (Liu et al., ICML 2023) and hardens against "wrong opponent model" -- a dominant failure mode in multiplayer search. Archetypes are trained as lightweight shared-backbone adapters during population training.
[DESIGN L0310] 
[DESIGN L0311] ---
[DESIGN L0312] 
[DESIGN L0313] ## 9. Search-as-Feature (SaF)
[DESIGN L0314] 
[DESIGN L0315] For each legal action $a$, AFBS returns: $\Delta Q(a)$, deal-in risk estimates (Boole/Hunter/robust), epistemic terms (entropy drop), robust stress ($\tau$), uncertainty (variance, ESS).
[DESIGN L0316] 
[DESIGN L0317] **Logit-residual policy:**
[DESIGN L0318] 
[DESIGN L0319] $$\ell_{\text{final}}(a)=\ell_\theta(a) + \alpha_{\text{SaF}}\cdot g_\psi(f(a))\cdot m(a)$$
[DESIGN L0320] 
[DESIGN L0321] Here, $m(a)\in\{0,1\}$ indicates whether features are present. $g_\psi$ is a tiny shared MLP (hidden dim 32-64).
[DESIGN L0322] 
[DESIGN L0323] **SaF-dropout:** during training, randomly zero $m$ even when features are available ($p_{\text{drop}}=0.3$) to prevent over-reliance. Train $g_\psi$ first via supervised regression on $\delta(a)=\log\pi_{\text{search}}(a)-\log\pi_{\text{base}}(a)$, then switch to joint end-to-end.
[DESIGN L0324] 
[DESIGN L0325] ---
[DESIGN L0326] 
[DESIGN L0327] ## 10. ExIt + Pondering as the central training engine
[DESIGN L0328] 
[DESIGN L0329] ### 10.1 ExIt targets
[DESIGN L0330] 
[DESIGN L0331] Current Hydra doctrine and implementation direction use a masked, visit-based root-child distribution as the ExIt teacher object. `root_exit_policy()` / q-softmax is not the teacher object for the live AFBS-generated ExIt lane.
[DESIGN L0332] 
[DESIGN L0333] ### 10.2 Pondering = label amplification
[DESIGN L0334] 
[DESIGN L0335] 75% idle time used for: deepening current root search + precomputing searches for predicted near-future states. Every completed search yields additional labeled training examples.
[DESIGN L0336] 
[DESIGN L0337] ### 10.3 Playout cap randomization
[DESIGN L0338] 
[DESIGN L0339] More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.
[DESIGN L0340] 
[DESIGN L0341] ---
[DESIGN L0342] 
[DESIGN L0343] ## 11. Training pipeline
[DESIGN L0344] 
[DESIGN L0345] ### Compute budget (about 2000 GPU-hours on Delta GPU `gpuA100x4` with 1 shared A100)
[DESIGN L0346] 
[DESIGN L0347] | Phase | GPU-hrs | Nets trained | Games | Key output |
[DESIGN L0348] |-------|--------:|-------------|------:|-----------|
[DESIGN L0349] | Phase -1: Benchmarks | 150 | All nets | N/A | Latency/throughput/distill gates |
[DESIGN L0350] | Phase 0: BC | 50 | LearnerNet (24-block) | N/A (5-6M expert) | Initialize from human data |
[DESIGN L0351] | Phase 1: Oracle guiding | 200 | LearnerNet + oracle critic | ~5M | Oracle-calibrated beliefs/danger |
[DESIGN L0352] | Phase 2: DRDA-wrapped ACH | 800 | LearnerNet via ACH+DRDA | ~18M | Game-theoretic base + early ExIt |
[DESIGN L0353] | Phase 3: ExIt + Pondering | 800 | LearnerNet (deep AFBS on hard positions) | ~12M | Deep search ExIt + endgame |
[DESIGN L0354] | **Total** | **2000** | | **~35M** | |
[DESIGN L0355] 
[DESIGN L0356] Logical role split: training, self-play generation, and pondering/search amplification should be partitioned across the available Delta A100 budget as throughput permits. Treat these as workload roles, not a claim that Hydra will have exclusive use of a full node. Distillation: Learner -> Actor continuously (IMPALA-style).
[DESIGN L0357] 
[DESIGN L0358] ### Phase -1: Hard reality benchmarks (150 GPU hours reserve)
[DESIGN L0359] Unlocked BEFORE committing the full budget. Must pass:
[DESIGN L0360] - **Latency gate**: AFBS on-turn < 150ms, CT-SMC DP < 1ms, endgame solver < 100ms
[DESIGN L0361] - **Throughput gate**: ActorNet self-play > 20 games/sec sustained
[DESIGN L0362] - **Distillation gate**: Learner->Actor KL drift < threshold over 100 updates
[DESIGN L0363] - **Hyperparameter sweep**: ACH eta, DRDA tau_drda, beam W, depth D, particles P
[DESIGN L0364] If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.
[DESIGN L0365] 
[DESIGN L0366] ### Phase 0: BC warm start (50 GPU hours)
[DESIGN L0367] Train LearnerNet (24-block) on 5-6M expert games (Tenhou Houou + Majsoul). 24x augmentation (6 suit perms x 4 seat rotations). All heads supervised. Distill to ActorNet (12-block) at end.
[DESIGN L0368] 
[DESIGN L0369] ### Phase 1: Oracle-visible supervision (200 GPU hours)
[DESIGN L0370] Self-play with full hidden state access. Train the oracle critic under the zero-sum constraint
[DESIGN L0371] 
[DESIGN L0372] $$\sum_i V_i = 0$$
[DESIGN L0373] 
[DESIGN L0374] and train the belief likelihood model alongside it.
[DESIGN L0375] 
[DESIGN L0376] Use the Suphx-style Bernoulli dropout schedule
[DESIGN L0377] 
[DESIGN L0378] $$\gamma_t: 1 \to 0$$
[DESIGN L0379] 
[DESIGN L0380] Post-oracle stability uses LR decay by $\times 0.1$ plus importance weight rejection when $\gamma_t$ reaches 0.
[DESIGN L0381] 
[DESIGN L0382] ### Phase 2: DRDA-wrapped ACH self-play (800 GPU hours)
[DESIGN L0383] 
[DESIGN L0384] **DRDA-wrapped ACH**: ACH is LuckyJ's inner optimizer (+0.4 fan over PPO) but its theory covers only 2-player zero-sum. For 4-player stability, wrap it in DRDA's multi-round structure (ICLR 2025).
[DESIGN L0385] 
[DESIGN L0386] The policy is
[DESIGN L0387] 
[DESIGN L0388] $$\pi_\theta(a|x) = \mathrm{softmax}(\ell_{\text{base}}(x,a) + y_\theta(x,a)/\tau_{\text{drda}})$$
[DESIGN L0389] 
[DESIGN L0390] Here, $\ell_{\text{base}}$ is a frozen checkpoint, $y_\theta$ is a trainable residual, and $\tau_{\text{drda}} \in \{2, 4, 8\}$ (tune via Phase -1; target median KL to base in $[0.05, 0.20]$).
[DESIGN L0391] 
[DESIGN L0392] **Rebase rule (CRITICAL):** Every 25-50 GPU hours, fold residual into base with
[DESIGN L0393] 
[DESIGN L0394] $$\ell_{\text{base}} \leftarrow \ell_{\text{base}} + y_\theta/\tau_{\text{drda}}$$
[DESIGN L0395] 
[DESIGN L0396] Then zero $y_\theta$ and reset optimizer moments. This preserves $\pi$ exactly across boundaries and prevents double-counting accumulated regret.
[DESIGN L0397] 
[DESIGN L0398] ACH update (per-(s,a) sample):
[DESIGN L0399] $$L_\pi(s,a) = -c(s,a) \cdot \eta \cdot \frac{y(a|s;\theta)}{\pi_{\text{old}}(a|s)} \cdot A(s,a)$$
[DESIGN L0400] 
[DESIGN L0401] - $\eta$: global scalar hyperparameter (try $\eta \in \{1,2,3\}$), NOT state-dependent in practice
[DESIGN L0402] - $c(s,a) \in \{0,1\}$: per-sample gate zeroing update when ratio exceeds $1\pm\epsilon$ OR centered logit exceeds $\pm l_{\text{th}}$
[DESIGN L0403] - Uses **logits** $y(a)$ (not log-probs), centered by $\bar{y}(s)$ and clamped to $[-l_{\text{th}}, l_{\text{th}}]$
[DESIGN L0404] - Standard GAE for advantages (per-player $V_i$, $\lambda=0.95$, $\gamma=0.995$)
[DESIGN L0405] - **One update epoch per batch** (not PPO's 3-10 epochs)
[DESIGN L0406] - Recommended: $\epsilon=0.5$, $l_{\text{th}}=8$, $\beta_{\text{ent}}=5\times10^{-4}$, LR $2.5\times10^{-4}$
[DESIGN L0407] 
[DESIGN L0408] Oracle critic provides advantages via CTDE: actor conditions on public info only. Normalize advantages per-minibatch for scale stability.
[DESIGN L0409] 
[DESIGN L0410] **Start cheap ExIt mid-Phase 2**: From ~400 GPU hours, run shallow AFBS (depth 3-4, P=64) on 20% of states. Don't wait for Phase 3 to begin amortizing search into the learner.
[DESIGN L0411] 
[DESIGN L0412] **Fallback:** If DRDA-wrapped ACH proves unstable, fall back to PPO with entropy 0.05-0.1.
[DESIGN L0413] 
[DESIGN L0414] ### Phase 2 (continuous): Distill rollout net
[DESIGN L0415] 
[DESIGN L0416] **RolloutNet** (ActorNet-sized, 12 blocks): LuckyJ's "environmental model" concept. Policy + value for fast AFBS rollouts. Distilled from LearnerNet **continuously** (not every 50h -- confirmed too stale). Same input encoding. Run distillation worker on spare GPU cycles.
[DESIGN L0417] 
[DESIGN L0418] ### Phase 3: ExIt + AFBS + Pondering (800 GPU hours)
[DESIGN L0419] 
[DESIGN L0420] LearnerNet runs deep AFBS for **hard positions only** (top-2 policy gap < 10%, high-risk defense, low particle ESS) when the available Delta GPU A100 throughput budget allows. ExIt targets distilled into LearnerNet's own training loss (ACH + ExIt + SaF auxiliary regression). ActorNet updated from LearnerNet continuously.
[DESIGN L0421] 
[DESIGN L0422] ### Population training
[DESIGN L0423] League: latest ActorNet, trailing checkpoints, human-style anchors (BC-heavy), adversarial exploiters.
[DESIGN L0424] 
[DESIGN L0425] ---
[DESIGN L0426] 
[DESIGN L0427] ## 12. Risk, information, and placement
[DESIGN L0428] 
[DESIGN L0429] ### 12.1 Distributional value and CVaR
[DESIGN L0430] Score pdf/cdf heads. CVaR for "avoid 4th" objectives.
[DESIGN L0431] 
[DESIGN L0432] ### 12.2 Information-Value Decomposition (IVD)
[DESIGN L0433] The full decomposition is
[DESIGN L0434] 
[DESIGN L0435] $$Q^{\text{total}}(I,a)=Q^{\text{inst}}(I,a)+\beta_{\text{epi}} Q^{\text{epi}}(I,a)+\xi Q^{\text{str}}(I,a)$$
[DESIGN L0436] 
[DESIGN L0437] Here, instrumental means score utility, epistemic means posterior entropy decrease, and strategic means concealment or leakage penalty. Note that $\beta_{\text{epi}}$ is the epistemic weight, distinct from ACH's $\eta$.
[DESIGN L0438] 
[DESIGN L0439] ### 12.3 Primal-dual risk constraints
[DESIGN L0440] Constraints keep deal-in risk below $\kappa_{\text{deal}}$ and information leakage below $\kappa_{\text{leak}}$.
[DESIGN L0441] 
[DESIGN L0442] Dual updates use
[DESIGN L0443] 
[DESIGN L0444] $$\lambda \leftarrow [\lambda+\alpha(\hat{C}-\kappa)]_+$$
[DESIGN L0445] 
[DESIGN L0446] ### DeltaQ lane runtime note
[DESIGN L0447] 
[DESIGN L0448] The target architecture continues to include a DeltaQ supervision family. For current repo maturity and promotion state, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md` instead of treating this architecture summary as the live status owner.
[DESIGN L0449] 
[DESIGN L0450] ---
[DESIGN L0451] 
[DESIGN L0452] ## 13. Validation gates
[DESIGN L0453] 
[DESIGN L0454] **G0:** Does Mixture-SIB + particles + AFBS produce positive decision improvement? 200K stratified states, mean $\Delta>0$, <40% negative.
[DESIGN L0455] 
[DESIGN L0456] **G1:** Robustness calibration. KL deviations between opponent model and held-out opponents at 95th percentile.
[DESIGN L0457] 
[DESIGN L0458] **G2:** Safety bound usefulness. Hunter reduces over-folding without underestimating risk beyond CI.
[DESIGN L0459] 
[DESIGN L0460] **G3:** SaF amortization. Shallow search + SaF must dominate shallow search alone.
[DESIGN L0461] 
[DESIGN L0462] ---
[DESIGN L0463] 
[DESIGN L0464] ## 14. Deployment profile
[DESIGN L0465] 
[DESIGN L0466] **Fast path:** Network forward + SaF adaptor. **Slow path:** Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time. Agari guard always active.
[DESIGN L0467] 
[DESIGN L0468] ---
[DESIGN L0469] 
[DESIGN L0470] ## 15. Heritage from prior Hydra variants
[DESIGN L0471] 
[DESIGN L0472] **From the throughput-first plan:** Asynchronous pondering as "free" label compute, distributional value heads, oracle guiding/critic, PPO hyperparameters (entropy coeff 0.05+), double-buffered weight sync, ExIt safety valves.
[DESIGN L0473] 
[DESIGN L0474] **From the all-out plan:** Mixture-SIB, anytime FBS, SaF, Hunter/Kounias tightening, ExIt+Pondering centrality, SR concentration.
[DESIGN L0475] 
[DESIGN L0476] **OMEGA additions:** CT-SMC exact contingency-table belief sampler, robust opponent nodes (KL-uncertainty soft-min + OLSS-style archetype set), hand-EV oracle features, endgame exactification, DRDA-wrapped ACH training with explicit rebase rule, 2-tier network (12/24), early ExIt from mid-Phase 2, explicit calibration gates.
[DESIGN L0477] 
[DESIGN L0478] **Verified ablation data (Suphx Figure 8):** SL baseline ~7.65 dan, +RL basic +0.41, +GRP +0.18, +oracle guiding +0.12. Oracle guiding alone is modest; the stack is what matters.
[DESIGN L0479] 
[DESIGN L0480] ---
[DESIGN L0481] 
[DESIGN L0482] ## 16. Limitations
[DESIGN L0483] 
[DESIGN L0484] 1. **4-player general-sum has no clean exploitability target.** We use robustness + population training instead.
[DESIGN L0485] 2. **Belief model misspecification** remains the core risk; G0 detects it early.
[DESIGN L0486] 3. **Compute allocation**: deep AFBS is expensive; depends on caching, pondering hit rate, distillation efficiency.
[DESIGN L0487] 4. **Strategy fusion / determinization pitfalls**: particles + robust opponent nodes mitigate but do not eliminate all pathologies.
[DESIGN L0488] 
[DESIGN L0489] ---
[DESIGN L0490] 
[DESIGN L0491] ## 17. References
[DESIGN L0492] 
[DESIGN L0493] 1. Sinkhorn, Knopp. "Doubly Stochastic Matrices." *Pacific J. Math*, 1967.
[DESIGN L0494] 2. Hunter. "Upper Bound for Union." *J. Applied Probability*, 1976.
[DESIGN L0495] 3. Kounias. "Bounds for Union." *Annals Math Stat*, 1968.
[DESIGN L0496] 4. Borcea, Branden, Liggett. "SR and Geometry of Polynomials." *JAMS*, 2009.
[DESIGN L0497] 5. Bardenet, Maillard. "Concentration for Sampling Without Replacement." *Bernoulli*, 2015.
[DESIGN L0498] 6. Anthony, Tian, Barber. "Expert Iteration." *NeurIPS*, 2017.
[DESIGN L0499] 7. Silver et al. "Mastering Go Without Human Knowledge." *Nature* 550, 2017.
[DESIGN L0500] 8. Wu. "Accelerating Self-Play Learning in Go (KataGo)." *arXiv 1902.10565*, 2020.
[DESIGN L0501] 9. Li et al. "Suphx: Mastering Mahjong with Deep RL." *arXiv 2003.13590*, 2020.
[DESIGN L0502] 10. Li et al. "Speedup Training via Reward Variance Reduction." *IEEE CoG*, 2022.
[DESIGN L0503] 11. Farina et al. "DRDA for Multiplayer POSGs." *ICLR*, 2025.
[DESIGN L0504] 12. Rudolph et al. "Reevaluating PG Methods in IIGs." *arXiv 2502.08938*, 2025.
[DESIGN L0505] 13. Kalogiannis, Farina. "PG Converge in IIEFGs." *NeurIPS*, 2024.
[DESIGN L0506] 14. Schulman et al. "Proximal Policy Optimization." *arXiv 1707.06347*, 2017.
[DESIGN L0507] 15. Perolat et al. "Mastering Stratego (DeepNash)." *Science*, 2022.
[DESIGN L0508] 16. Boney et al. "Learning to Play IIGs by Imitating an Oracle Planner." *IEEE Trans. Games*, 2021.
[DESIGN L0509] 17. Abbasi-Yadkori et al. "POLITEX." *ICML*, 2019.
[DESIGN L0510] 18. Cuturi. "Sinkhorn Distances." *NeurIPS*, 2013.
[DESIGN L0511] 19. Chen, Diaconis, Holmes, Liu. "Sequential Monte Carlo Methods for Statistical Analysis of Tables." *JASA*, 2005.
[DESIGN L0512] 20. Patefield. "Algorithm AS 159: An Efficient Method of Generating R x C Tables with Given Row and Column Totals." *Applied Statistics*, 1981.
[DESIGN L0513] 21. Fu et al. "Actor-Critic Hedge for Imperfect-Information Games (ACH)." *ICLR*, 2022.
[DESIGN L0514] 22. Liu et al. "OLSS: Opponent-Limited Online Search for Imperfect-Information Games." *ICML*, 2023.
```

## Artifact 08 — Operational doctrine and roadmap to Hydra v1
Artifact id: `hydra-reconciliation`
Source label: DESIGN
Type: `file_full`
Source: `research/design/HYDRA_RECONCILIATION.md`
Why it matters: Operational doctrine and near-term sequencing authority. Useful for judging whether DCRL-adjacent prior art should trigger any real roadmap change rather than just docs positioning.

```markdown
[DESIGN L0001] # Hydra Reconciliation
[DESIGN L0002] 
[DESIGN L0003] > **Promoted operational doctrine and roadmap to Hydra v1.**
[DESIGN L0004] >
[DESIGN L0005] > This file owns Hydra's active-path sequencing, roadmap to Hydra v1, and
[DESIGN L0006] > active-vs-staged-vs-reserve decisions after reconciling the canonical archive
[DESIGN L0007] > SSOT with current repository state.
[DESIGN L0008] >
[DESIGN L0009] > If a downstream implementation or reference doc conflicts with this file on
[DESIGN L0010] > sequencing, promotion order, or active-vs-staged-vs-reserve status, this file
[DESIGN L0011] > wins.
[DESIGN L0012] >
[DESIGN L0013] > If this file drifts from
[DESIGN L0014] > `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` or current
[DESIGN L0015] > code/runtime, refresh this file instead of treating the drift as a demotion of
[DESIGN L0016] > the upstream source or of runtime truth.
[DESIGN L0017] 
[DESIGN L0018] This file is Hydra's promoted operational doctrine.
[DESIGN L0019] 
[DESIGN L0020] It has one job:
[DESIGN L0021] 
[DESIGN L0022] - keep **Max Hydra** as the long-term destination from `HYDRA_FINAL.md`
[DESIGN L0023] - define **Hydra v1** as the active path to ship and train first
[DESIGN L0024] - make Hydra v1 the most efficient path to start training soon without
[DESIGN L0025]   collapsing every advanced idea into the first training promise
[DESIGN L0026] 
[DESIGN L0027] In plain English:
[DESIGN L0028] 
[DESIGN L0029] - Hydra should not restart from zero
[DESIGN L0030] - Hydra should not wait for every north-star mechanism before training starts
[DESIGN L0031] - Hydra should train first on the strongest credible baseline already supported
[DESIGN L0032]   by the repo
[DESIGN L0033] - Hydra should promote harder lanes only when they clear real evidence gates
[DESIGN L0034] 
[DESIGN L0035] Relationship to adjacent surfaces:
[DESIGN L0036] 
[DESIGN L0037] - `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` is the epistemic
[DESIGN L0038]   root / canonical archive source ledger that powers downstream promoted
[DESIGN L0039]   doctrine.
[DESIGN L0040] - `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` is the derived
[DESIGN L0041]   archive prioritization view over that same root.
[DESIGN L0042] - `research/agent_handoffs/` preserves archive evidence, provenance, and claim
[DESIGN L0043]   trust; it does **not** replace this file as the owner of current active-path
[DESIGN L0044]   status.
[DESIGN L0045] - `research/design/HYDRA_FINAL.md` owns Hydra's architecture north star and max
[DESIGN L0046]   ceiling.
[DESIGN L0047] - `docs/CURRENT_STATUS.md` owns the promoted already-built shipped/staged repo
[DESIGN L0048]   snapshot derived from this file plus code/runtime validation.
[DESIGN L0049] - `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` own runtime
[DESIGN L0050]   semantics and compatibility-sensitive invariants.
[DESIGN L0051] 
[DESIGN L0052] Scope:
[DESIGN L0053] 
[DESIGN L0054] - Canonical archive SSOT:
[DESIGN L0055]   `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
[DESIGN L0056] - Target architecture summary: `research/design/HYDRA_FINAL.md`
[DESIGN L0057] - Current shipped/staged status snapshot: `docs/CURRENT_STATUS.md`
[DESIGN L0058] - Verified runtime reality: current code plus runtime docs
[DESIGN L0059] - Operational question answered here: what Hydra should train and promote next,
[DESIGN L0060]   in what order, and what explicitly remains later
[DESIGN L0061] 
[DESIGN L0062] ## 1. Roadmap thesis
[DESIGN L0063] 
[DESIGN L0064] Hydra has two valid horizons, and they should not be confused.
[DESIGN L0065] 
[DESIGN L0066] 1. **Max Hydra** is the destination.
[DESIGN L0067]    - Hydra's north star remains the maximum-ceiling system described in
[DESIGN L0068]      `HYDRA_FINAL.md`: ExIt-centered training, richer belief/search machinery,
[DESIGN L0069]      selective search amplification, stronger opponent modeling, and later
[DESIGN L0070]      endgame precision.
[DESIGN L0071] 2. **Hydra v1** is the immediate target.
[DESIGN L0072]    - Hydra v1 is the strongest version Hydra can train soon with credible
[DESIGN L0073]      labels, closed enough loops, and controlled complexity.
[DESIGN L0074] 
[DESIGN L0075] This roadmap chooses Hydra v1 as the active path because it is the most
[DESIGN L0076] efficient route to long-run strength.
[DESIGN L0077] 
[DESIGN L0078] That means:
[DESIGN L0079] 
[DESIGN L0080] - **trainable beats theoretically fuller** when the fuller system still depends
[DESIGN L0081]   on weak labels, open promotion questions, or broad compute-heavy integration
[DESIGN L0082] - **close loops before expanding architecture** because the repo already has a
[DESIGN L0083]   lot of advanced surface area
[DESIGN L0084] - **promote by evidence, not by excitement** because code existence alone is not
[DESIGN L0085]   proof that a lane should become default-on
[DESIGN L0086] 
[DESIGN L0087] Hydra v1 is not a retreat from Max Hydra. It is the shortest honest path
[DESIGN L0088] from today's repo state to sustained training and later ceiling-raising
[DESIGN L0089] promotion.
[DESIGN L0090] 
[DESIGN L0091] ## 2. Status and roadmap vocabulary
[DESIGN L0092] 
[DESIGN L0093] ### 2.1 Status vocabulary
[DESIGN L0094] 
[DESIGN L0095] These terms are shared with `docs/CURRENT_STATUS.md`.
[DESIGN L0096] 
[DESIGN L0097] | Term | Meaning |
[DESIGN L0098] |---|---|
[DESIGN L0099] | `active path` | current mainline direction to optimize/build now |
[DESIGN L0100] | `shipped baseline` | implemented and part of the current live Hydra baseline |
[DESIGN L0101] | `implemented but not default-on` | implemented and intentionally not the default runtime/training path |
[DESIGN L0102] | `implemented but staged` | implemented in core form, but activation/promotion remains intentionally deferred |
[DESIGN L0103] | `reserve shelf` | preserved later-work direction; not current mainline |
[DESIGN L0104] | `blocked` | not ready because a real dependency, semantic gap, or promotion requirement remains |
[DESIGN L0105] | `rejected` | not part of the current plan |
[DESIGN L0106] | `historical` | preserved context only; not current governing truth |
[DESIGN L0107] 
[DESIGN L0108] ### 2.2 Roadmap vocabulary
[DESIGN L0109] 
[DESIGN L0110] | Term | Meaning |
[DESIGN L0111] |---|---|
[DESIGN L0112] | `Hydra v1` | the immediate, training-first active path |
[DESIGN L0113] | `Max Hydra` | the long-term destination owned by `HYDRA_FINAL.md` |
[DESIGN L0114] | `baseline first` | start training from the shipped baseline before broadening the active surface |
[DESIGN L0115] | `staged lane` | a capability kept off the default path until evidence supports promotion |
[DESIGN L0116] | `promotion gate` | an explicit pass/fail condition required before a staged lane moves upward |
[DESIGN L0117] | `training start condition` | the minimum baseline health required before the next training cycle should begin |
[DESIGN L0118] | `anti-chaos principle` | a rule that prevents Hydra from broadening too many uncertain fronts at once |
[DESIGN L0119] 
[DESIGN L0120] ## 3. Starting baseline for the next version
[DESIGN L0121] 
[DESIGN L0122] Hydra v1 does not start from a blank slate. It starts from the strongest
[DESIGN L0123] repo surface already promoted as baseline or near-baseline truth.
[DESIGN L0124] 
[DESIGN L0125] ### 3.1 Shipped baseline
[DESIGN L0126] 
[DESIGN L0127] The current shipped baseline includes:
[DESIGN L0128] 
[DESIGN L0129] - `hydra-core` as a real first-party runtime/encoder/simulator crate
[DESIGN L0130] - the live encoder/model contract at `192x34`, with the old `85x34` view treated
[DESIGN L0131]   as baseline-prefix only
[DESIGN L0132] - the fixed 46-action runtime with two-phase riichi and kan handling
[DESIGN L0133] - the stronger public-teacher belief-semantics tranche as part of the current
[DESIGN L0134]   training baseline
[DESIGN L0135] - the current Hand-EV realism upgrade as part of the live baseline surface
[DESIGN L0136] - replay-derived `safety_residual` as a narrow supervised lane
[DESIGN L0137] - an end-to-end ExIt carrier across the live self-play lane and the
[DESIGN L0138]   replay/sample sidecar-first lane
[DESIGN L0139] 
[DESIGN L0140] ### 3.2 Implemented but not default-on
[DESIGN L0141] 
[DESIGN L0142] The current challenger lane is:
[DESIGN L0143] 
[DESIGN L0144] - the narrow DeltaQ supervision lane, which is implemented in code and remains
[DESIGN L0145]   promotion-gated through an arena-confirmation path
[DESIGN L0146] 
[DESIGN L0147] ### 3.3 Implemented but staged
[DESIGN L0148] 
[DESIGN L0149] The current staged lanes are:
[DESIGN L0150] 
[DESIGN L0151] - `mixture_weight` promotion
[DESIGN L0152] - richer opponent-target closure
[DESIGN L0153] - representative-world / per-particle CT-SMC Hand-EV
[DESIGN L0154] - selective AFBS / endgame deepening
[DESIGN L0155] 
[DESIGN L0156] ### 3.4 Reserve shelf
[DESIGN L0157] 
[DESIGN L0158] The current reserve shelf includes:
[DESIGN L0159] 
[DESIGN L0160] - broader public-belief search as project identity
[DESIGN L0161] - deeper robust-opponent search backups
[DESIGN L0162] - larger latent-opponent / richer auxiliary-head expansion until existing target
[DESIGN L0163]   closure improves
[DESIGN L0164] 
[DESIGN L0165] This baseline is already strong enough to justify a Hydra v1 training plan.
[DESIGN L0166] The roadmap should not keep talking about shipped baseline work as if it were
[DESIGN L0167] still hypothetical future work.
[DESIGN L0168] 
[DESIGN L0169] ## 4. Hydra v1, the active path
[DESIGN L0170] 
[DESIGN L0171] ### 4.1 What Hydra v1 is
[DESIGN L0172] 
[DESIGN L0173] Hydra v1 is the strongest trainable Hydra that can be pursued now without
[DESIGN L0174] turning the first version into a chaos pile of half-closed lanes.
[DESIGN L0175] 
[DESIGN L0176] Hydra v1 is:
[DESIGN L0177] 
[DESIGN L0178] - a **strong learned policy/value baseline first**
[DESIGN L0179] - a **supervision-first, search-second** training path
[DESIGN L0180] - an **ExIt-aware** training path that keeps the live carrier in scope
[DESIGN L0181] - a baseline that already includes the shipped belief-semantics tranche and the
[DESIGN L0182]   shipped Hand-EV realism tranche
[DESIGN L0183] - a path that keeps selective search where it clearly pays instead of making it
[DESIGN L0184]   the project identity too early
[DESIGN L0185] - a promotion-based roadmap where staged lanes move only when they clear proof
[DESIGN L0186] 
[DESIGN L0187] ### 4.2 What Hydra v1 is not
[DESIGN L0188] 
[DESIGN L0189] Hydra v1 is not:
[DESIGN L0190] 
[DESIGN L0191] - a broad “search everywhere” AFBS project
[DESIGN L0192] - a freeze-until-Max-Hydra project
[DESIGN L0193] - a fork away from the architecture in `HYDRA_FINAL.md`
[DESIGN L0194] - a head-count expansion phase
[DESIGN L0195] - a promise that every advanced surface in the repo becomes default-on in the
[DESIGN L0196]   first training cycle
[DESIGN L0197] 
[DESIGN L0198] ### 4.3 Why this is the active path now
[DESIGN L0199] 
[DESIGN L0200] Hydra v1 is the active path now because:
[DESIGN L0201] 
[DESIGN L0202] - the repo already contains a partially built advanced baseline
[DESIGN L0203] - the strongest near-term leverage is better training loop closure, not broader
[DESIGN L0204]   search identity
[DESIGN L0205] - the shipped belief and Hand-EV tranches already raise the baseline without
[DESIGN L0206]   demanding that every harder lane be promoted first
[DESIGN L0207] - broad search-first Hydra remains more compute-heavy, more integration-heavy,
[DESIGN L0208]   and less likely to accelerate the first honest training campaign
[DESIGN L0209] 
[DESIGN L0210] ## 5. Staged lanes and promotion order
[DESIGN L0211] 
[DESIGN L0212] Hydra v1 should grow through a narrow promotion order.
[DESIGN L0213] 
[DESIGN L0214] ### Lane A. Baseline training launch
[DESIGN L0215] 
[DESIGN L0216] This is the immediate roadmap target.
[DESIGN L0217] 
[DESIGN L0218] What is in:
[DESIGN L0219] 
[DESIGN L0220] - the shipped baseline surface from Section 3.1
[DESIGN L0221] - training on the live `192x34` / 46-action contract
[DESIGN L0222] - the shipped belief baseline
[DESIGN L0223] - the shipped Hand-EV realism baseline
[DESIGN L0224] - the replay-derived `safety_residual` lane
[DESIGN L0225] - the live ExIt carrier as part of the training story
[DESIGN L0226] 
[DESIGN L0227] What stays out of the first baseline promise:
[DESIGN L0228] 
[DESIGN L0229] - default-on DeltaQ
[DESIGN L0230] - promoted `mixture_weight`
[DESIGN L0231] - richer opponent-target closure
[DESIGN L0232] - representative-world CT-SMC Hand-EV
[DESIGN L0233] - selective AFBS / endgame deepening as a required launch condition
[DESIGN L0234] 
[DESIGN L0235] Immediate objective:
[DESIGN L0236] 
[DESIGN L0237] - start the next honest Hydra training cycle from the strongest already-promoted
[DESIGN L0238]   baseline instead of delaying for full Max-Hydra closure
[DESIGN L0239] 
[DESIGN L0240] ### Lane B. Controlled promotion lanes
[DESIGN L0241] 
[DESIGN L0242] These lanes are the first candidates for measured promotion after baseline
[DESIGN L0243] training is healthy.
[DESIGN L0244] 
[DESIGN L0245] Current priority order:
[DESIGN L0246] 
[DESIGN L0247] 1. **DeltaQ as a challenger lane**
[DESIGN L0248]    - implemented, measurable, and explicitly promotion-gated
[DESIGN L0249]    - remains non-default until its promotion evidence clears
[DESIGN L0250] 2. **Belief-adjacent staged semantics**
[DESIGN L0251]    - preserve `mixture_weight` as staged until the teacher object is stronger
[DESIGN L0252]      than the current staged reading
[DESIGN L0253] 3. **Richer opponent-target closure**
[DESIGN L0254]    - keep staged until labels and ontology are more credible
[DESIGN L0255] 
[DESIGN L0256] Principle:
[DESIGN L0257] 
[DESIGN L0258] - promotion lanes should be narrow, measurable, and one-fight-at-a-time
[DESIGN L0259] 
[DESIGN L0260] ### Lane C. Selective search-strength lanes
[DESIGN L0261] 
[DESIGN L0262] These are real strength multipliers, but they are not the first training start
[DESIGN L0263] condition.
[DESIGN L0264] 
[DESIGN L0265] They include:
[DESIGN L0266] 
[DESIGN L0267] - representative-world / per-particle CT-SMC Hand-EV
[DESIGN L0268] - selective AFBS / endgame deepening
[DESIGN L0269] - later search-grade integration improvements that build on a healthier training
[DESIGN L0270]   loop
[DESIGN L0271] 
[DESIGN L0272] Principle:
[DESIGN L0273] 
[DESIGN L0274] - search should stay selective and specialist until the baseline training path
[DESIGN L0275]   is alive and promotion evidence says the broader cost is worth paying
[DESIGN L0276] 
[DESIGN L0277] ### Lane D. Destination-facing Max Hydra lanes
[DESIGN L0278] 
[DESIGN L0279] These remain aligned with `HYDRA_FINAL.md`, but they are not Hydra v1
[DESIGN L0280] blockers.
[DESIGN L0281] 
[DESIGN L0282] They include:
[DESIGN L0283] 
[DESIGN L0284] - deeper robust-opponent search backups / safe exploitation layers
[DESIGN L0285] - broader public-belief-search identity
[DESIGN L0286] - richer latent-opponent / more unified opponent modeling
[DESIGN L0287] - deeper endgame exactification and later hard-state expansion policies
[DESIGN L0288] - optimizer/game-theory escalations that depend on a healthier training loop
[DESIGN L0289] 
[DESIGN L0290] Principle:
[DESIGN L0291] 
[DESIGN L0292] - preserve these lanes, but do not let them outrank a working Hydra v1
[DESIGN L0293]   training loop
[DESIGN L0294] 
[DESIGN L0295] ## 6. Training start conditions
[DESIGN L0296] 
[DESIGN L0297] Hydra v1 is ready to begin the next training cycle when the following are
[DESIGN L0298] true.
[DESIGN L0299] 
[DESIGN L0300] ### 6.1 Required to start
[DESIGN L0301] 
[DESIGN L0302] - the shipped baseline is the declared default training surface
[DESIGN L0303] - this roadmap and `docs/CURRENT_STATUS.md` agree on what is baseline versus
[DESIGN L0304]   staged versus reserve
[DESIGN L0305] - shipped belief semantics and shipped Hand-EV realism are treated as current
[DESIGN L0306]   baseline truth, not as future work
[DESIGN L0307] - ExIt remains part of the baseline training story through its live carrier
[DESIGN L0308] - staged lanes that are not part of the baseline remain explicitly off by
[DESIGN L0309]   default
[DESIGN L0310] 
[DESIGN L0311] ### 6.2 Not required to start
[DESIGN L0312] 
[DESIGN L0313] The next training cycle does **not** require:
[DESIGN L0314] 
[DESIGN L0315] - broad public-belief search as the main runtime identity
[DESIGN L0316] - default-on AFBS everywhere
[DESIGN L0317] - default-on DeltaQ
[DESIGN L0318] - promoted `mixture_weight`
[DESIGN L0319] - representative-world / per-particle CT-SMC Hand-EV
[DESIGN L0320] - selective AFBS / endgame deepening
[DESIGN L0321] - deeper robust-opponent search backups
[DESIGN L0322] - richer opponent-target closure
[DESIGN L0323] - full Max-Hydra search stack closure
[DESIGN L0324] 
[DESIGN L0325] This section exists to stop Hydra from delaying training in the name of features
[DESIGN L0326] that are explicitly later.
[DESIGN L0327] 
[DESIGN L0328] ## 7. Promotion gates
[DESIGN L0329] 
[DESIGN L0330] Implemented code is not enough to earn default-on status. Promotion follows
[DESIGN L0331] gates.
[DESIGN L0332] 
[DESIGN L0333] ### 7.1 Baseline gate
[DESIGN L0334] 
[DESIGN L0335] Baseline work is ready when:
[DESIGN L0336] 
[DESIGN L0337] - the capability is already promoted as shipped baseline truth
[DESIGN L0338] - its semantics are honest in docs and in runtime/training surfaces
[DESIGN L0339] - it does not depend on still-staged lanes to justify training start
[DESIGN L0340] 
[DESIGN L0341] ### 7.2 Challenger lane gates
[DESIGN L0342] 
[DESIGN L0343] An implemented-but-not-default-on or implemented-but-staged lane moves upward
[DESIGN L0344] only when:
[DESIGN L0345] 
[DESIGN L0346] - its labels or targets are semantically credible
[DESIGN L0347] - its activation behavior is explicit rather than accidental
[DESIGN L0348] - its contribution is measurable in training/eval rather than inferred from
[DESIGN L0349]   theory alone
[DESIGN L0350] - promoting it does not blur the distinction between baseline and experiment
[DESIGN L0351] 
[DESIGN L0352] Explicit example:
[DESIGN L0353] 
[DESIGN L0354] - DeltaQ remains implemented but not default-on because its promotion is tied to
[DESIGN L0355]   an arena-confirmation path rather than mere structural existence
[DESIGN L0356] 
[DESIGN L0357] ### 7.3 Search-strength gates
[DESIGN L0358] 
[DESIGN L0359] A search-strength lane moves upward only when:
[DESIGN L0360] 
[DESIGN L0361] - the baseline training loop is already alive
[DESIGN L0362] - the lane has a clear insertion point and a narrow scope
[DESIGN L0363] - the lane improves real strength-per-complexity instead of reopening project
[DESIGN L0364]   identity debates
[DESIGN L0365] 
[DESIGN L0366] ### 7.4 Max-Hydra-only gates
[DESIGN L0367] 
[DESIGN L0368] Destination-facing lanes should only become active-path work when:
[DESIGN L0369] 
[DESIGN L0370] - Hydra v1 has already proved too weak or too capped
[DESIGN L0371] - the simpler promotion lanes have been fairly tested first
[DESIGN L0372] - the extra complexity is justified by evidence instead of north-star gravity
[DESIGN L0373] 
[DESIGN L0374] ## 8. Anti-chaos principles
[DESIGN L0375] 
[DESIGN L0376] These principles are mandatory for Hydra v1.
[DESIGN L0377] 
[DESIGN L0378] 1. **Baseline before breadth**
[DESIGN L0379]    - do not broaden multiple uncertain lanes before the baseline training path is
[DESIGN L0380]      live
[DESIGN L0381] 2. **One promotion fight at a time**
[DESIGN L0382]    - do not try to promote several staged lanes at once
[DESIGN L0383] 3. **No architecture identity flip midstream**
[DESIGN L0384]    - the next version is training-first, not search-first by surprise later
[DESIGN L0385] 4. **Shipped means baseline, staged means staged**
[DESIGN L0386]    - do not keep talking about shipped baseline work as if it were still future
[DESIGN L0387]      work
[DESIGN L0388] 5. **North star is destination, not checklist**
[DESIGN L0389]    - `HYDRA_FINAL.md` remains the target architecture, but it does not force all
[DESIGN L0390]      destination-facing machinery into the first training promise
[DESIGN L0391] 6. **Preserve reserve ideas without letting them steer**
[DESIGN L0392]    - reserve shelf exists to keep good ideas alive, not to dominate current
[DESIGN L0393]      sequencing
[DESIGN L0394] 
[DESIGN L0395] ## 9. Destination-facing reserve shelf
[DESIGN L0396] 
[DESIGN L0397] These lanes remain consistent with Max Hydra and should stay documented.
[DESIGN L0398] 
[DESIGN L0399] ### 9.1 Preserve for later
[DESIGN L0400] 
[DESIGN L0401] - deeper robust-opponent search backups / safe exploitation layers
[DESIGN L0402] - broader public-belief-search identity
[DESIGN L0403] - richer latent-opponent / more unified opponent modeling
[DESIGN L0404] - deeper AFBS semantics and hard-state expansion policies
[DESIGN L0405] - selective exactification and stronger endgame resolvers
[DESIGN L0406] - deeper belief-network experiments
[DESIGN L0407] - optimizer/game-theory escalations that depend on a healthier training loop
[DESIGN L0408] 
[DESIGN L0409] ### 9.2 Not active for the next version
[DESIGN L0410] 
[DESIGN L0411] These are not rejected forever. They are simply not allowed to steer Hydra v1.
[DESIGN L0412] 
[DESIGN L0413] - broad “search everywhere” AFBS rollout
[DESIGN L0414] - full public-belief search as immediate project identity
[DESIGN L0415] - adding more heads before existing advanced surfaces are properly promoted
[DESIGN L0416] - large optimizer-theory detours ahead of the first honest training campaign
[DESIGN L0417] - speculative novelty that lacks a strong repo insertion point
[DESIGN L0418] 
[DESIGN L0419] ## 10. Hydra v1 roadmap summary
[DESIGN L0420] 
[DESIGN L0421] Hydra's roadmap to v1 is straightforward.
[DESIGN L0422] 
[DESIGN L0423] ### Immediate objective
[DESIGN L0424] 
[DESIGN L0425] Start training soon on Hydra v1: the strongest credible baseline already
[DESIGN L0426] supported by promoted doctrine and current shipped surfaces.
[DESIGN L0427] 
[DESIGN L0428] ### First version scope
[DESIGN L0429] 
[DESIGN L0430] Hydra v1 means:
[DESIGN L0431] 
[DESIGN L0432] - fixed live runtime/encoder compatibility surface
[DESIGN L0433] - shipped belief baseline
[DESIGN L0434] - shipped Hand-EV realism baseline
[DESIGN L0435] - narrow replay-derived `safety_residual`
[DESIGN L0436] - live ExIt carrier
[DESIGN L0437] - staged lanes kept staged unless they clear promotion gates
[DESIGN L0438] 
[DESIGN L0439] ### First promotions after launch
[DESIGN L0440] 
[DESIGN L0441] After baseline training is healthy, Hydra should evaluate narrow challenger lanes
[DESIGN L0442] in order, starting with DeltaQ and only then considering later staged belief,
[DESIGN L0443] opponent-target, and search-strength promotions.
[DESIGN L0444] 
[DESIGN L0445] ### Long-term destination
[DESIGN L0446] 
[DESIGN L0447] Max Hydra from `HYDRA_FINAL.md` remains the long-term destination. Hydra v1
[DESIGN L0448] exists to reach that destination efficiently, not to replace it.
[DESIGN L0449] 
[DESIGN L0450] ### Final doctrine sentence
[DESIGN L0451] 
[DESIGN L0452] Hydra should begin with the strongest trainable baseline it can honestly defend,
[DESIGN L0453] then grow toward its full ceiling through narrow, evidence-gated promotion. That
[DESIGN L0454] is the active path most likely to produce a strong Hydra over time.
```

## Artifact 09 — Reward variance reduction and oracle value design
Artifact id: `reward-design`
Source label: DESIGN
Type: `file_full`
Source: `research/design/REWARD_DESIGN.md`
Why it matters: Crucial adjacent prior-art surface: explains Hydra's take on RVR, oracle baselines, expected reward network, centralized value function language, and what Hydra already treats as directly relevant work.

```markdown
[DESIGN L0001] # Hydra Reward Design
[DESIGN L0002] 
[DESIGN L0003] > **Status note:** this is a mixed design/reference document. Keep the reward-analysis evidence and reserve ideas here. For active-path doctrine, use `research/design/HYDRA_RECONCILIATION.md`. For runtime truth, use `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.
[DESIGN L0004] >
[DESIGN L0005] > Do not treat older `TRAINING.md` references as current governing doctrine.
[DESIGN L0006] 
[DESIGN L0007] Hydra's reward function design, informed by cross-domain analysis of reward systems in Pluribus, ReBeL, AlphaStar, OpenAI Five, and RVR Mahjong.
[DESIGN L0008] 
[DESIGN L0009] > **Background reading:** The full literature survey of reward functions across landmark AI systems is not currently preserved as a standalone archive file in this repo; treat the references and analysis below as the surviving summary surface.
[DESIGN L0010] 
[DESIGN L0011] ---
[DESIGN L0012] 
[DESIGN L0013] ## Table of Contents
[DESIGN L0014] 
[DESIGN L0015] 1. [Reward Variance Reduction for Mahjong (IEEE CoG 2022)](#1-reward-variance-reduction-for-mahjong-ieee-cog-2022)
[DESIGN L0016] 2. [Hydra's Reward Function — Final Decision](#2-hydras-reward-function--final-decision)
[DESIGN L0017] 3. [References](#references)
[DESIGN L0018] 
[DESIGN L0019] ---
[DESIGN L0020] 
[DESIGN L0021] ## 1. Reward Variance Reduction for Mahjong (IEEE CoG 2022)
[DESIGN L0022] 
[DESIGN L0023] **Paper:** "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" — Li, Wu, Fu, Fu, Zhao, Xing (Tencent AI Lab + CAS + Tsinghua), IEEE CoG 2022
[DESIGN L0024] **Game:** 4-player Mahjong (Chinese rules)
[DESIGN L0025] 
[DESIGN L0026] ### The Core Problem
[DESIGN L0027] 
[DESIGN L0028] Mahjong reward has **extremely high variance** from two sources:
[DESIGN L0029] 1. **Invisibility:** 3/4 of tiles are hidden (vs. ~50% in poker), making value estimation noisy
[DESIGN L0030] 2. **Stochasticity:** The last tile drawn determines win/loss outcome, and *how* you win (tsumo vs. ron, specific tile) dramatically changes the point value
[DESIGN L0031] 
[DESIGN L0032] ### RVR Technique
[DESIGN L0033] 
[DESIGN L0034] Two neural networks work together:
[DESIGN L0035] 
[DESIGN L0036] #### Component 1: Relative Value Network
[DESIGN L0037] 
[DESIGN L0038] - **Purpose:** Reduce variance from hidden information (invisibility)
[DESIGN L0039] - **Input:** Oracle view (all 4 players' hands — privileged information)
[DESIGN L0040] - **Output:** Simultaneous value estimates for all 4 players: V_θ = (V₁, V₂, V₃, V₄)
[DESIGN L0041] - **Zero-sum constraint:** Loss function enforces Σ V_i = 0
[DESIGN L0042] 
[DESIGN L0043] This is exactly **Suphx's oracle guiding** / AlphaStar's centralized value function applied to Mahjong. By seeing all hands during training, the value estimate has much lower variance than one estimated from the acting player's partial observation alone.
[DESIGN L0044] 
[DESIGN L0045] #### Component 2: Expected Reward Network
[DESIGN L0046] 
[DESIGN L0047] - **Purpose:** Reduce variance from end-of-hand stochasticity (luck)
[DESIGN L0048] - **Input:** Game state at round T−1 (the penultimate state before the game ends)
[DESIGN L0049] - **Output:** Predicted expected reward f_θ(g^{T-1})
[DESIGN L0050] - **Key insight:** The *last* tile draw introduces massive variance. A hand might be worth 0 or 12000 points depending on the final draw. By predicting the *expected* reward from the state just before the final draw, this filters out last-tile luck.
[DESIGN L0051] 
[DESIGN L0052] #### Combined Training
[DESIGN L0053] 
[DESIGN L0054] - During training, the raw game reward r_i is replaced with f_θ(g^{T-1}) for the RL update
[DESIGN L0055] - The Relative Value Network provides the baseline V(s) for advantage computation
[DESIGN L0056] - Together, they reduce both sources of variance simultaneously
[DESIGN L0057] 
[DESIGN L0058] ### Exact Reward Formula
[DESIGN L0059] 
[DESIGN L0060] ```
[DESIGN L0061] RL reward = f_θ(g^{T-1})    [Expected Reward Network output]
[DESIGN L0062] Advantage = f_θ(g^{T-1}) − V_oracle(s)   [relative to oracle value baseline]
[DESIGN L0063] ```
[DESIGN L0064] 
[DESIGN L0065] ### Per-Step vs Per-Episode
[DESIGN L0066] 
[DESIGN L0067] **Per-episode** (per-hand). The reward is the final placement/point change from one hand, but filtered through the Expected Reward Network.
[DESIGN L0068] 
[DESIGN L0069] ### Baseline Subtraction
[DESIGN L0070] 
[DESIGN L0071] - **Relative Value Network** serves as the value baseline
[DESIGN L0072] - Zero-sum constraint ensures the four players' advantages sum to zero
[DESIGN L0073] - Oracle information (all tiles visible) dramatically tightens the baseline
[DESIGN L0074] 
[DESIGN L0075] ### Reward Normalization
[DESIGN L0076] 
[DESIGN L0077] Not explicitly described. The zero-sum constraint naturally bounds the rewards.
[DESIGN L0078] 
[DESIGN L0079] ### Results
[DESIGN L0080] 
[DESIGN L0081] - Reported faster training convergence compared to vanilla PPO (the paper describes "speedup" qualitatively but does not state a specific speedup multiplier)
[DESIGN L0082] - Achieves the same final policy quality with significantly less compute
[DESIGN L0083] 
[DESIGN L0084] ### Key Takeaway for Hydra
[DESIGN L0085] 
[DESIGN L0086] This is **the most directly relevant work.** For Hydra:
[DESIGN L0087] 1. **Oracle value baseline** (Relative Value Network) = already planned via oracle distillation
[DESIGN L0088] 2. **Expected Reward Network** at T−1 is novel and high-value: it directly addresses Mahjong's biggest variance source (last-tile luck)
[DESIGN L0089] 3. **Zero-sum constraint** on value estimates is cheap to implement and provably correct
[DESIGN L0090] 4. The convergence speedup matters enormously for Hydra's single-GPU training constraint
[DESIGN L0091] 
[DESIGN L0092] ---
[DESIGN L0093] 
[DESIGN L0094] ## 2. Hydra's Reward Function — Final Decision
[DESIGN L0095] 
[DESIGN L0096] Based on the earlier cross-domain survey work (no longer preserved here as a standalone `archive/REWARD_SURVEY.md` file), Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:
[DESIGN L0097] 
[DESIGN L0098] ### The Formula
[DESIGN L0099] 
[DESIGN L0100] The exact reward formula and implementation priority should be treated as active only when promoted by the reconciled doctrine. Keep the analysis below as reference/evidence rather than a hidden source of authority.
[DESIGN L0101] 
[DESIGN L0102] ### Why This Design
[DESIGN L0103] 
[DESIGN L0104] | Decision | Choice | Evidence |
[DESIGN L0105] |----------|--------|----------|
[DESIGN L0106] | **Episode boundary** | Per-kyoku | Both Mortal and Suphx use this. ~100× lower variance than per-game. |
[DESIGN L0107] | **Reward signal** | GRP ΔE[pts] | Mortal's proven approach. Equivalent to potential-based reward shaping (Ng 1999) — policy-invariant. |
[DESIGN L0108] | **Placement points** | [3, 1, -1, -3] | Mortal's training default. Symmetric, zero-sum. Each rank step = 2 pts. Platform-specific via config swap. |
[DESIGN L0109] | **GRP design** | 24-class permutation softmax | Captures inter-player rank correlations. 4-class loses this. Mortal proved it works. |
[DESIGN L0110] | **Discount γ** | 1.0 | Mortal uses γ=1. Kyoku is short enough (~15 steps). No need for temporal discounting. |
[DESIGN L0111] | **Variance reduction** | Oracle critic + ERN | RVR paper: significant speedup. Attacks both variance sources (hidden info + last-tile luck). |
[DESIGN L0112] | **GRP lifecycle** | Pretrained, frozen during RL | Stable reward signal. Mortal does this. Avoids moving-target problem. |
[DESIGN L0113] | **Reward normalization** | Running std (Welford) | Mortal-Policy's exact approach. Essential for PPO in high-variance games. |
[DESIGN L0114] | **No reward shaping** | Skip (GRP delta IS PBRS already) | Double-shaping adds risk. Shanten-based shaping creates offensive bias — worst possible for Mahjong. |
[DESIGN L0115] | **No intrinsic motivation** | Skip | SL warm-start solves exploration. RND/ICM would add noise from tile draw stochasticity. |
[DESIGN L0116] | **Same reward all phases** | Mandatory | Changing reward invalidates value function. Cal-QL (NeurIPS 2023) showed this causes "unlearning." |
[DESIGN L0117] 
[DESIGN L0118] ### Confirmed Anti-Patterns (From Mortal Community)
[DESIGN L0119] 
[DESIGN L0120] The anti-pattern list below is retained as reference guidance; do not treat dead `TRAINING.md` links as live authority.
[DESIGN L0121] 
[DESIGN L0122] ### Platform-Specific Fine-Tuning (Via pts_vector Swap)
[DESIGN L0123] 
[DESIGN L0124] | Target Platform | pts_vector | Strategy Bias |
[DESIGN L0125] |----------------|------------|---------------|
[DESIGN L0126] | General training | [3, 1, -1, -3] | Balanced (default) |
[DESIGN L0127] | Tenhou Houou | [3, 1.5, 0, -4.5] | Avoid 4th (normalized Tenhou net pts) |
[DESIGN L0128] | Mahjong Soul Throne | [3, 1, -1, -3] | Balanced (Majsoul uma is already nearly symmetric) |
[DESIGN L0129] | WRC / EMA tournament | [3, 1, -1, -3] | Balanced (identical incentive structure) |
[DESIGN L0130] | M-League style | [5, 1, -1, -3] | Push for 1st |
[DESIGN L0131] 
[DESIGN L0132] ---
[DESIGN L0133] 
[DESIGN L0134] ## References
[DESIGN L0135] 
[DESIGN L0136] | Ref | Paper | Year | Venue |
[DESIGN L0137] |-----|-------|------|-------|
[DESIGN L0138] | [6] | Li et al., "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" | 2022 | IEEE CoG |
[DESIGN L0139] | [9] | Ng et al., "Policy invariance under reward transformations" | 1999 | ICML |
[DESIGN L0140] | [10] | Harutyunyan et al., "Hindsight Credit Assignment" | 2019 | NeurIPS |
[DESIGN L0141] | [11] | Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" | 2020 | arXiv |
[DESIGN L0142] | [12] | Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning" | 2023 | NeurIPS |
[DESIGN L0143] | [14] | Engstrom et al., "Implementation Matters in Deep Policy Gradients" | 2020 | ICLR |
[DESIGN L0144] | [15] | Huang, "The 37 Implementation Details of Proximal Policy Optimization" | 2022 | Blog/ICLR |
[DESIGN L0145] 
[DESIGN L0146] > References [1]-[5] and [7]-[8] come from the earlier cross-domain reward survey work, but that standalone archive file is not currently present in this repo.
```

## Artifact 10 — Training paradigm comparison survey
Artifact id: `training-paradigms`
Source label: COMPARE
Type: `file_full`
Source: `research/comparisons/TRAINING_PARADIGMS.md`
Why it matters: Major local survey artifact for asymmetric/oracle training, ExIt, world-model lanes, and other adjacent paradigms. Helps the research agent see what Hydra already knows about asymmetric oracle training versus what DCRL might add.

```markdown
[COMPARE L0001] # Alternative Training Paradigms: Beyond Standard Self-Play
[COMPARE L0002] 
[COMPARE L0003] **Date**: 2026-03-03
[COMPARE L0004] **Purpose**: Survey alternatives to standard self-play RL (PPO/ACH/R-NaD) for stronger policies with equal or less compute.
[COMPARE L0005] **Relevance**: Hydra's Phase 2 (oracle distillation) and Phase 3 (league self-play) could benefit from these approaches.
[COMPARE L0006] 
[COMPARE L0007] ---
[COMPARE L0008] 
[COMPARE L0009] ## Executive Summary
[COMPARE L0010] 
[COMPARE L0011] | Paradigm | Beats Self-Play? | Measured Gains | Compute Cost | Hydra Relevance |
[COMPARE L0012] |---|---|---|---|---|
[COMPARE L0013] | Offline RL (CQL/IQL/DT) | No (ceiling = dataset) | Bootstrapping only | Lower | Phase 1 warm-start |
[COMPARE L0014] | Expert Iteration (ExIt) | Yes | Defeats pure RL baselines | Higher (search) | Phase 3 upgrade |
[COMPARE L0015] | Counterfactual (CFR) | Yes (IIGs) | Foundation of poker AI | Variable | Complementary to RL |
[COMPARE L0016] | Imagination (LAMIR) | Yes | Up to 80% WR vs R-NaD | Higher | Promising but immature |
[COMPARE L0017] | Inverse RL | Uncertain | No game AI evidence | High | Low priority |
[COMPARE L0018] | Multi-task/Auxiliary | Improves sample efficiency | 2-5x faster convergence | Neutral | Already in Hydra spec |
[COMPARE L0019] | Asymmetric (Oracle) | Yes | Suphx: top 0.01% Tenhou | Moderate | Phase 2 is this |
[COMPARE L0020] | Student of Games | Yes | Beats SOTA in poker/Scotland Yard | Higher | Future consideration |
[COMPARE L0021] 
[COMPARE L0022] **Bottom line**: ExIt (search-guided training) and asymmetric oracle training are the two most
[COMPARE L0023] actionable paradigms for Hydra. Imagination-augmented (LAMIR) is the most exciting recent
[COMPARE L0024] development but needs maturation for 4-player mahjong scale.
[COMPARE L0025] 
[COMPARE L0026] ---
[COMPARE L0027] 
[COMPARE L0028] ## 1. Offline RL on Expert Data (CQL, IQL, Decision Transformer)
[COMPARE L0029] 
[COMPARE L0030] ### What It Is
[COMPARE L0031] 
[COMPARE L0032] Train a policy entirely from a static dataset of expert games (no environment interaction).
[COMPARE L0033] Three main approaches:
[COMPARE L0034] 
[COMPARE L0035] - **CQL** (Conservative Q-Learning): Learns Q-values but penalizes Q-values for out-of-distribution
[COMPARE L0036]   actions via `logsumexp(Q) - mean(Q)` regularization. Prevents overestimation.
[COMPARE L0037] - **IQL** (Implicit Q-Learning): Avoids querying Q-values on unseen actions entirely. Uses
[COMPARE L0038]   expectile regression on the dataset's Q-distribution.
[COMPARE L0039] - **Decision Transformer (DT)**: Reformulates RL as sequence modeling. Conditions on desired
[COMPARE L0040]   return and autoregressively predicts actions. No Q-values at all.
[COMPARE L0041] 
[COMPARE L0042] ### Measured Comparisons (Caunhye & Jeewa 2025, arXiv:2511.16475)
[COMPARE L0043] 
[COMPARE L0044] D4RL Ant continuous-control benchmarks (normalized score, 4 seeds):
[COMPARE L0045] 
[COMPARE L0046] | Dataset | Reward | CQL | IQL | DT |
[COMPARE L0047] |---|---|---|---|---|
[COMPARE L0048] | medium | sparse | **91.55** | 84.49 | 87.9 |
[COMPARE L0049] | medium-replay | sparse | **71.99** | 42.14 | 66.3 |
[COMPARE L0050] | medium-expert | sparse | 103.38 | 85.95 | **120.6** |
[COMPARE L0051] | medium | dense | **99.49** | 95.5 | 88.0 |
[COMPARE L0052] | medium-replay | dense | 92.99 | **97.5** | 88.07 |
[COMPARE L0053] | medium-expert | dense | 107.0 | **124.2** | 90.24 |
[COMPARE L0054] 
[COMPARE L0055] **Compute**: DT 7.5h, CQL 5.0h, IQL 2.0h (100k steps, 4 seeds).
[COMPARE L0056] 
[COMPARE L0057] **Takeaway**: No universal winner. CQL excels on lower-quality sparse data. IQL excels on
[COMPARE L0058] dense-reward high-quality data. DT is most stable/low-variance across settings.
[COMPARE L0059] 
[COMPARE L0060] ### Mortal's Use of CQL
[COMPARE L0061] 
[COMPARE L0062] Mortal uses CQL specifically in its **offline training mode** (DeepWiki: Mortal Training Pipeline):
[COMPARE L0063] - Combined loss = DQN loss (MSE to MC Q-targets) + CQL loss * `min_q_weight` + next-rank loss
[COMPARE L0064] - CQL is active during offline training from historical Tenhou logs
[COMPARE L0065] - CQL is **disabled** during online self-play (where `min_q_weight = 0`)
[COMPARE L0066] 
[COMPARE L0067] ### CQL Limitations (Critical for Hydra)
[COMPARE L0068] 
[COMPARE L0069] 1. **Dataset ceiling**: CQL can never exceed the quality of the expert data it trains on.
[COMPARE L0070]    The conservative penalty actively prevents exploration beyond the dataset distribution.
[COMPARE L0071] 2. **Conservative bias**: By design, CQL underestimates Q-values. This makes it safe but
[COMPARE L0072]    suboptimal -- the policy becomes overly cautious.
[COMPARE L0073] 3. **No self-improvement**: Unlike online RL, CQL cannot discover novel strategies. It can
[COMPARE L0074]    only compress and generalize existing expert behavior.
[COMPARE L0075] 4. **Distribution mismatch**: If the dataset has systematic biases (e.g., all players from
[COMPARE L0076]    one rank tier), CQL will inherit those biases.
[COMPARE L0077] 5. **Hyperparameter sensitivity**: The `min_q_weight` balance between DQN loss and CQL
[COMPARE L0078]    regularization requires careful tuning. Too high = too conservative, too low = overestimation.
[COMPARE L0079] 
[COMPARE L0080] **Verdict for Hydra**: CQL is useful **only for Phase 1 warm-start** from expert logs.
[COMPARE L0081] It cannot replace online self-play for Phase 3. Mortal's architecture confirms this --
[COMPARE L0082] they use CQL offline then switch to pure online RL.
[COMPARE L0083] 
[COMPARE L0084] **Sources**: [CQL Paper (NeurIPS 2020)](https://arxiv.org/abs/2006.04779) |
[COMPARE L0085] [CQL vs IQL vs DT Comparison](https://arxiv.org/abs/2511.16475) |
[COMPARE L0086] [Mortal Training Pipeline](https://deepwiki.com/Equim-chan/Mortal/3.3-training-pipeline)
[COMPARE L0087] 
[COMPARE L0088] ---
[COMPARE L0089] 
[COMPARE L0090] ## 2. Expert Iteration (ExIt)
[COMPARE L0091] 
[COMPARE L0092] ### What It Is
[COMPARE L0093] 
[COMPARE L0094] ExIt ("Thinking Fast and Slow with Deep Learning and Tree Search", Anthony et al. 2017)
[COMPARE L0095] decomposes learning into two interacting systems:
[COMPARE L0096] 
[COMPARE L0097] 1. **Expert (slow)**: Tree search (MCTS or CFR) that produces strong but expensive policies
[COMPARE L0098] 2. **Apprentice (fast)**: Neural network that learns to imitate the search output
[COMPARE L0099] 
[COMPARE L0100] The loop:
[COMPARE L0101] ```
[COMPARE L0102] Repeat:
[COMPARE L0103]   1. Expert uses search (guided by current apprentice) to produce improved action targets
[COMPARE L0104]   2. Apprentice trains on these search-generated targets via supervised learning
[COMPARE L0105]   3. Apprentice's improved policy guides the expert's search in the next iteration
[COMPARE L0106] ```
[COMPARE L0107] 
[COMPARE L0108] This is essentially what AlphaGo/AlphaZero does: MCTS generates training targets, and the
[COMPARE L0109] policy network learns to predict those targets. AlphaZero IS Expert Iteration.
[COMPARE L0110] 
[COMPARE L0111] ### Why ExIt Beats Pure RL
[COMPARE L0112] 
[COMPARE L0113] The key insight: **search produces higher-quality training signal than raw RL returns**.
[COMPARE L0114] 
[COMPARE L0115] In pure RL (e.g., PPO), the policy gradient uses noisy game outcomes as the training signal.
[COMPARE L0116] In ExIt, the search process looks ahead many moves and produces a more informed action
[COMPARE L0117] distribution. The neural network then learns from this better signal.
[COMPARE L0118] 
[COMPARE L0119] ### Measured Results
[COMPARE L0120] 
[COMPARE L0121] - **Hex**: ExIt outperforms REINFORCE for training neural Hex players. The final ExIt agent
[COMPARE L0122]   (trained tabula rasa) defeats **MoHex 1.0** (strongest publicly available Olympiad champion
[COMPARE L0123]   at time of publication).
[COMPARE L0124] - **Go (AlphaZero)**: AlphaZero (which is ExIt with MCTS) defeats Stockfish, Elmo, and the
[COMPARE L0125]   original AlphaGo without any human data.
[COMPARE L0126] - **Quality delta**: The search "expert" consistently provides better training targets than
[COMPARE L0127]   the network alone, and this gap persists even as the network improves (because search
[COMPARE L0128]   depth keeps amplifying the network's improvements).
[COMPARE L0129] 
[COMPARE L0130] ### Applicability to Mahjong / Hydra
[COMPARE L0131] 
[COMPARE L0132] **Challenge**: ExIt requires a search procedure. For imperfect-info games like mahjong,
[COMPARE L0133] standard MCTS doesn't work. You need:
[COMPARE L0134] - CFR-based search (like Student of Games uses), or
[COMPARE L0135] - Information-set MCTS (IS-MCTS), or
[COMPARE L0136] - Learned-model search (like LAMIR)
[COMPARE L0137] 
[COMPARE L0138] **Opportunity**: If Hydra implements inference-time search (which is already planned per the
[COMPARE L0139] spec), ExIt is the natural training paradigm. Instead of pure PPO self-play, use search at
[COMPARE L0140] training time to generate stronger training targets, then distill into the policy network.
[COMPARE L0141] 
[COMPARE L0142] **Estimated compute cost**: Higher than pure RL per sample (search is expensive), but
[COMPARE L0143] potentially much better sample efficiency -- fewer total environment steps needed.
[COMPARE L0144] 
[COMPARE L0145] **Sources**: [ExIt Paper (NeurIPS 2017)](https://arxiv.org/abs/1705.08439) |
[COMPARE L0146] [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
[COMPARE L0147] 
[COMPARE L0148] ---
[COMPARE L0149] 
[COMPARE L0150] ## 3. Hindsight Learning / Counterfactual Training
[COMPARE L0151] 
[COMPARE L0152] ### What It Is
[COMPARE L0153] 
[COMPARE L0154] Two distinct concepts here:
[COMPARE L0155] 
[COMPARE L0156] **A. Hindsight Experience Replay (HER)** -- Andrychowicz et al. 2017
[COMPARE L0157] - Originally for goal-conditioned robotics with sparse rewards
[COMPARE L0158] - After a failed trajectory, relabel the goal to be what was actually achieved
[COMPARE L0159] - Turns every failure into a successful training example for some goal
[COMPARE L0160] - **Not directly applicable to competitive games** (no goal relabeling analog)
[COMPARE L0161] 
[COMPARE L0162] **B. Counterfactual Regret Minimization (CFR)** -- Zinkevich et al. 2007
[COMPARE L0163] - THE method for imperfect-information games (poker, etc.)
[COMPARE L0164] - Asks: "What regret would I have for not playing action X, across all possible hidden states?"
[COMPARE L0165] - Iteratively minimizes total counterfactual regret, converging to Nash equilibrium
[COMPARE L0166] - **Pluribus** (superhuman 6-player poker) and **Libratus** both use CFR variants
[COMPARE L0167] 
[COMPARE L0168] ### Game AI Applications
[COMPARE L0169] 
[COMPARE L0170] **CFR for Mahjong (CFR-p, arXiv:2307.12087)**:
[COMPARE L0171] - Applies CFR to two-player mahjong with hierarchical abstraction
[COMPARE L0172] - Game-theoretic analysis + winning-policy-based abstraction
[COMPARE L0173] - Demonstrates CFR feasibility for mahjong variants, though 4-player Riichi is much larger
[COMPARE L0174] 
[COMPARE L0175] **ReBeL (Brown et al. 2020, Facebook AI)**:
[COMPARE L0176] - Combines CFR with learned value networks
[COMPARE L0177] - Self-play generates data, CFR resolves subgames at test time
[COMPARE L0178] - Achieves strong performance in poker and Liar's Dice
[COMPARE L0179] - Key innovation: treats belief states as "public states" and learns values over them
[COMPARE L0180] 
[COMPARE L0181] **Counterfactual value networks (DeepStack)**:
[COMPARE L0182] - Learns a "what-if" value function: given this hidden state, what would each action be worth?
[COMPARE L0183] - This is inherently counterfactual -- evaluating unchosen actions across unobserved states
[COMPARE L0184] 
[COMPARE L0185] ### Applicability to Hydra
[COMPARE L0186] 
[COMPARE L0187] The counterfactual perspective is already embedded in CFR-based approaches. For Hydra:
[COMPARE L0188] - **Phase 3 could incorporate CFR-style reasoning** instead of/alongside PPO
[COMPARE L0189] - The danger-head and tenpai-head in Hydra's architecture are already a form of counterfactual
[COMPARE L0190]   reasoning ("what would happen if opponent is tenpai?")
[COMPARE L0191] - Full CFR is likely too expensive for 4-player Riichi's game tree, but **depth-limited CFR
[COMPARE L0192]   with learned values** (as in ReBeL/Student of Games) is feasible
[COMPARE L0193] 
[COMPARE L0194] **Sources**: [HER Paper](https://arxiv.org/abs/1707.01495) |
[COMPARE L0195] [CFR-p for Mahjong](https://arxiv.org/abs/2307.12087) |
[COMPARE L0196] [ReBeL Paper](https://arxiv.org/abs/2007.13544)
[COMPARE L0197] 
[COMPARE L0198] ---
[COMPARE L0199] 
[COMPARE L0200] ## 4. Imagination-Augmented Training (Learned World Models)
[COMPARE L0201] 
[COMPARE L0202] ### What It Is
[COMPARE L0203] 
[COMPARE L0204] MuZero (Schrittwieser et al. 2020) learns a world model in latent space:
[COMPARE L0205] - **Representation**: encodes observations into latent states
[COMPARE L0206] - **Dynamics**: predicts next latent state given action
[COMPARE L0207] - **Prediction**: outputs policy, value, and reward from latent state
[COMPARE L0208] 
[COMPARE L0209] Training generates "imagined" trajectories in latent space, providing extra training data
[COMPARE L0210] beyond real experience. This is like dreaming -- the model practices in its imagination.
[COMPARE L0211] 
[COMPARE L0212] ### LAMIR: Extending to Imperfect-Information Games (Oct 2024, arXiv:2510.05048)
[COMPARE L0213] 
[COMPARE L0214] **LAMIR** (Learned Abstract Model for Imperfect-information Reasoning) is the most relevant
[COMPARE L0215] recent work. Key innovations:
[COMPARE L0216] 
[COMPARE L0217] 1. **Information-set representations**: Learns latent representations of players' belief states
[COMPARE L0218]    (not just world states), capturing what each player knows
[COMPARE L0219] 2. **Abstract subgame construction**: Learns a domain-independent abstraction of information
[COMPARE L0220]    sets, capped at size L, making subgames tractable
[COMPARE L0221] 3. **CFR+ resolving at test time**: Instead of MCTS (unsound for IIGs), uses CFR+ with
[COMPARE L0222]    continual resolving over the learned model
[COMPARE L0223] 4. **Depth-limited search**: Learned value functions at the horizon boundary
[COMPARE L0224] 
[COMPARE L0225] ### Measured Results (Beating R-NaD)
[COMPARE L0226] 
[COMPARE L0227] Head-to-head win rates vs RNaD (3M training episodes):
[COMPARE L0228] 
[COMPARE L0229] | Game | LAMIR Win Rate |
[COMPARE L0230] |---|---|
[COMPARE L0231] | II Goofspiel 10 | **54.5% +/- 0.25** |
[COMPARE L0232] | II Goofspiel 13 | **60.7% +/- 0.34** |
[COMPARE L0233] | II Goofspiel 15 | **80.5% +/- 0.26** |
[COMPARE L0234] 
[COMPARE L0235] These are massive wins. The advantage grows with game complexity, suggesting learned models
[COMPARE L0236] become more valuable as games get larger/harder.
[COMPARE L0237] 
[COMPARE L0238] ### Limitations (from the paper)
[COMPARE L0239] 
[COMPARE L0240] - Does **not explicitly model chance nodes** (relevant for mahjong's tile draws)
[COMPARE L0241] - CFR guarantees may weaken with imperfect-recall abstractions
[COMPARE L0242] - Action-space size is not abstracted (mahjong has 46 actions, manageable)
[COMPARE L0243] - Only tested on Goofspiel variants so far, not on games at mahjong's scale
[COMPARE L0244] 
[COMPARE L0245] ### Applicability to Hydra
[COMPARE L0246] 
[COMPARE L0247] **High potential but high risk.** LAMIR's approach is exactly what Hydra would need for
[COMPARE L0248] search-augmented training in a IIG. However:
[COMPARE L0249] - 4-player Riichi Mahjong is vastly larger than Goofspiel
[COMPARE L0250] - The chance-node limitation is a real problem (tile draws are central to mahjong)
[COMPARE L0251] - Implementation complexity is significant (learned model + CFR resolving + value networks)
[COMPARE L0252] 
[COMPARE L0253] **Recommendation**: Monitor LAMIR closely. If the approach scales to larger games in future
[COMPARE L0254] work, it could be Hydra's Phase 4 upgrade. Not ready for Phase 3 today.
[COMPARE L0255] 
[COMPARE L0256] **Sources**: [MuZero Paper](https://arxiv.org/abs/1911.08265) |
[COMPARE L0257] [LAMIR Paper (2024)](https://arxiv.org/abs/2510.05048) |
[COMPARE L0258] [Demystifying MuZero](https://arxiv.org/abs/2411.04580)
[COMPARE L0259] 
[COMPARE L0260] ---
[COMPARE L0261] 
[COMPARE L0262] ## 5. Inverse RL from Expert Play
[COMPARE L0263] 
[COMPARE L0264] ### What It Is
[COMPARE L0265] 
[COMPARE L0266] Instead of defining a reward function and optimizing it, IRL:
[COMPARE L0267] 1. Observes expert behavior (human pro mahjong games)
[COMPARE L0268] 2. Infers what reward function the expert must be optimizing
[COMPARE L0269] 3. Uses that learned reward to train an RL agent
[COMPARE L0270] 
[COMPARE L0271] The idea: human experts might be optimizing for subtle objectives that hand-crafted reward
[COMPARE L0272] functions miss (e.g., "this discard is safe AND develops hand flexibility AND signals to
[COMPARE L0273] opponents I'm not dangerous").
[COMPARE L0274] 
[COMPARE L0275] ### State of the Art (2024-2025)
[COMPARE L0276] 
[COMPARE L0277] Recent survey (Springer 2025): IRL is advancing but primarily in robotics and autonomous
[COMPARE L0278] driving, not competitive games.
[COMPARE L0279] 
[COMPARE L0280] - **AIRL + reward shaping** (arXiv:2410.03847): Model-based reward shaping for adversarial
[COMPARE L0281]   IRL. Improves performance in stochastic environments. No game applications.
[COMPARE L0282] - **Potential-based reward shaping for IRL** (ICLR 2025): Reduces computational burden of
[COMPARE L0283]   IRL sub-problems. Theoretical contribution, not game-specific.
[COMPARE L0284] - **Gamer behavior decoding** (Yale 2024): Uses IRL to understand player motivations in
[COMPARE L0285]   gaming. Analytical, not for training stronger agents.
[COMPARE L0286] 
[COMPARE L0287] ### Could This Capture Nuances That Placement Score Misses?
[COMPARE L0288] 
[COMPARE L0289] **In theory, yes.** If you had a large dataset of 10-dan games and ran IRL on them, you
[COMPARE L0290] might discover reward shaping terms that placement-based rewards miss. For example:
[COMPARE L0291] - Implicit risk preferences (not just expected value but variance aversion)
[COMPARE L0292] - Tempo/pace-of-play preferences
[COMPARE L0293] - Meta-game signaling rewards
[COMPARE L0294] 
[COMPARE L0295] **In practice, doubtful.** Problems:
[COMPARE L0296] 1. IRL is computationally expensive (requires solving many forward RL problems)
[COMPARE L0297] 2. The recovered reward is often degenerate (multiple rewards explain the same behavior)
[COMPARE L0298] 3. No demonstrated improvement over hand-crafted rewards in competitive game AI
[COMPARE L0299] 4. Mahjong's stochasticity makes reward inference very noisy
[COMPARE L0300] 
[COMPARE L0301] **Verdict for Hydra**: Low priority. The reward design in REWARD_DESIGN.md (placement-based
[COMPARE L0302] with RVR variance reduction) is likely sufficient. If anything, the multi-head architecture
[COMPARE L0303] (value + GRP + tenpai + danger) already captures the nuances that IRL would discover.
[COMPARE L0304] 
[COMPARE L0305] **Sources**: [IRL Survey (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11100-0) |
[COMPARE L0306] [Model-Based Reward Shaping for AIRL](https://arxiv.org/abs/2410.03847)
[COMPARE L0307] 
[COMPARE L0308] ---
[COMPARE L0309] 
[COMPARE L0310] ## 6. Multi-Task Learning / Auxiliary Objectives
[COMPARE L0311] 
[COMPARE L0312] ### What It Is
[COMPARE L0313] 
[COMPARE L0314] Train the model on multiple related tasks simultaneously. The shared representation learns
[COMPARE L0315] features useful across all tasks, improving generalization and sample efficiency.
[COMPARE L0316] 
[COMPARE L0317] ### Evidence Base
[COMPARE L0318] 
[COMPARE L0319] **UNREAL (Jaderberg et al. 2017, DeepMind)**:
[COMPARE L0320] - Added auxiliary tasks (reward prediction, pixel control, feature control) to A3C
[COMPARE L0321] - **10x median improvement** across 57 Atari games
[COMPARE L0322] - Auxiliary tasks act as "free" additional gradient signal
[COMPARE L0323] 
[COMPARE L0324] **Comparing Auxiliary Tasks for RL (arXiv:2310.04241, ICLR venue)**:
[COMPARE L0325] Most helpful auxiliary tasks ranked:
[COMPARE L0326] 1. **Forward state prediction (fsp)**: predict next observation given current obs + action
[COMPARE L0327] 2. **Forward state-difference prediction (fsdp)**: predict delta between observations
[COMPARE L0328] 3. **Reward prediction (rwp)**: least helpful of the three
[COMPARE L0329] 
[COMPARE L0330] Key finding: **auxiliary tasks help more as task complexity increases.** Simple environments
[COMPARE L0331] see minimal benefit; complex environments (like mahjong!) see large gains.
[COMPARE L0332] 
[COMPARE L0333] ### What Hydra Already Has
[COMPARE L0334] 
[COMPARE L0335] Hydra's spec already includes multi-task heads:
[COMPARE L0336] - **Value head**: scalar expected placement score
[COMPARE L0337] - **GRP head (24-way)**: global reward prediction (placement distribution)
[COMPARE L0338] - **Tenpai head (3-way)**: opponent tenpai probability
[COMPARE L0339] - **Danger head (3x34)**: per-tile danger probabilities per opponent
[COMPARE L0340] 
[COMPARE L0341] Mortal uses: **next-rank prediction** as its auxiliary task.
[COMPARE L0342] 
[COMPARE L0343] ### What Could Be Added
[COMPARE L0344] 
[COMPARE L0345] Additional auxiliary objectives that could help:
[COMPARE L0346] 1. **Opponent action prediction**: predict what each opponent will discard next
[COMPARE L0347] 2. **Tile draw prediction**: predict distribution over next drawn tile (given visible info)
[COMPARE L0348] 3. **Hand reconstruction**: predict opponents' hidden hands from visible information
[COMPARE L0349] 4. **Shanten prediction**: predict own/opponents' shanten count
[COMPARE L0350] 5. **Forward state prediction**: predict next game state features after your action
[COMPARE L0351] 
[COMPARE L0352] ### Measured Improvement Expectations
[COMPARE L0353] 
[COMPARE L0354] Based on the auxiliary task literature:
[COMPARE L0355] - Sample efficiency improvement: **2-5x** for complex tasks (UNREAL benchmarks)
[COMPARE L0356] - Maximum performance improvement: **moderate** (helps learn faster, eventual ceiling similar)
[COMPARE L0357] - Most benefit during **early/mid training**, diminishing returns at convergence
[COMPARE L0358] - The tenpai and danger heads in Hydra already capture the most important auxiliary signals
[COMPARE L0359] 
[COMPARE L0360] **Verdict for Hydra**: Already well-served by current design. Adding opponent-action prediction
[COMPARE L0361] as a 6th head would be the highest-value addition. Low-hanging fruit since the encoder
[COMPARE L0362] already processes all visible game state.
[COMPARE L0363] 
[COMPARE L0364] **Sources**: [UNREAL Paper](https://arxiv.org/abs/1611.05397) |
[COMPARE L0365] [Auxiliary Task Comparison](https://arxiv.org/abs/2310.04241) |
[COMPARE L0366] [Hydra Final](../design/HYDRA_FINAL.md)
[COMPARE L0367] 
[COMPARE L0368] ---
[COMPARE L0369] 
[COMPARE L0370] ## 7. Asymmetric Self-Play (Oracle-Student Training)
[COMPARE L0371] 
[COMPARE L0372] ### What It Is
[COMPARE L0373] 
[COMPARE L0374] During training, one agent (the "oracle") sees hidden information that the other agent
[COMPARE L0375] (the "student") doesn't. The oracle's superior play provides a stronger training signal.
[COMPARE L0376] 
[COMPARE L0377] Two main approaches:
[COMPARE L0378] 1. **Oracle as opponent**: Oracle plays against student, student learns from harder games
[COMPARE L0379] 2. **Oracle as teacher**: Oracle's value estimates guide the student's learning (distillation)
[COMPARE L0380] 
[COMPARE L0381] ### Suphx's Oracle Guiding (Li et al. 2020, Microsoft Research)
[COMPARE L0382] 
[COMPARE L0383] Suphx pioneered this for mahjong:
[COMPARE L0384] 
[COMPARE L0385] 1. **Train an oracle agent** that sees all players' tiles (perfect information)
[COMPARE L0386] 2. **Oracle produces value estimates** for each game state
[COMPARE L0387] 3. **Student agent learns from oracle's value function** via distillation, but plays with
[COMPARE L0388]    only its own visible information at test time
[COMPARE L0389] 4. **Global reward prediction** provides the reward signal
[COMPARE L0390] 
[COMPARE L0391] **Results**: Suphx reached the top **0.01%** of all officially ranked human players on Tenhou,
[COMPARE L0392] achieving a stable rating above **10-dan** level. This was the first AI to outperform most
[COMPARE L0393] top human players in Mahjong.
[COMPARE L0394] 
[COMPARE L0395] ### Why It Works
[COMPARE L0396] 
[COMPARE L0397] The oracle sees ground truth (all tiles), so its value estimates are much more accurate
[COMPARE L0398] than values learned from partial information. When the student distills from these estimates:
[COMPARE L0399] - It learns better representations of hidden state
[COMPARE L0400] - It gets a lower-variance training signal
[COMPARE L0401] - It converges faster because the teacher already "knows the answer"
[COMPARE L0402] 
[COMPARE L0403] Think of it like having the answer key while studying -- you learn more efficiently even
[COMPARE L0404] though you won't have the answer key during the test.
[COMPARE L0405] 
[COMPARE L0406] ### Latest Research on Asymmetric Training
[COMPARE L0407] 
[COMPARE L0408] **Student of Games (SoG, Schmid et al. 2023, Science Advances)**:
[COMPARE L0409] - Unifies search + self-play + game-theoretic reasoning
[COMPARE L0410] - Uses **growing-tree CFR (GT-CFR)** for sound search in both perfect and imperfect info games
[COMPARE L0411] - Beats strongest openly available agent in heads-up no-limit Texas hold'em
[COMPARE L0412] - Defeats SOTA agent in Scotland Yard
[COMPARE L0413] - Achieves strong performance in chess and Go
[COMPARE L0414] 
[COMPARE L0415] SoG's "sound self-play" ensures the search-generated training data doesn't introduce
[COMPARE L0416] exploitable biases, which is a known risk of naive asymmetric training.
[COMPARE L0417] 
[COMPARE L0418] **DeepNash (Perolat et al. 2022, Science)**:
[COMPARE L0419] - R-NaD (Regularized Nash Dynamics) for Stratego
[COMPARE L0420] - Model-free, search-free, pure self-play
[COMPARE L0421] - Achieves human-expert level, top-3 all-time on Gravon platform
[COMPARE L0422] - Key insight: R-NaD converges TO Nash equilibrium instead of cycling around it
[COMPARE L0423] - Not asymmetric, but relevant as the baseline that LAMIR beats
[COMPARE L0424] 
[COMPARE L0425] ### Applicability to Hydra
[COMPARE L0426] 
[COMPARE L0427] **This IS Hydra's Phase 2.** The training pipeline already specifies oracle distillation:
[COMPARE L0428] - Phase 1: Supervised warm-start from expert logs
[COMPARE L0429] - Phase 2: Oracle distillation (oracle sees all tiles, student learns from oracle values)
[COMPARE L0430] - Phase 3: League self-play (student plays against itself and past versions)
[COMPARE L0431] 
[COMPARE L0432] The Suphx evidence strongly supports this pipeline. The question is whether to enhance
[COMPARE L0433] Phase 2 with search (making it ExIt-style oracle distillation) or keep it pure value
[COMPARE L0434] distillation.
[COMPARE L0435] 
[COMPARE L0436] **Recommendation**: Phase 2 as designed is well-supported by evidence. Consider adding
[COMPARE L0437] search-guided training in Phase 3 (ExIt-style) for additional improvement.
[COMPARE L0438] 
[COMPARE L0439] **Sources**: [Suphx Paper](https://arxiv.org/abs/2003.13590) |
[COMPARE L0440] [Student of Games (Science Advances 2023)](https://www.science.org/doi/10.1126/sciadv.adg3256) |
[COMPARE L0441] [DeepNash / R-NaD (Science 2022)](https://www.science.org/doi/10.1126/science.add4679)
[COMPARE L0442] 
[COMPARE L0443] ---
[COMPARE L0444] 
[COMPARE L0445] ## 8. Recent Papers Beating Standard Self-Play (2024-2025)
[COMPARE L0446] 
[COMPARE L0447] ### LAMIR (Oct 2024) -- Learned World Model + CFR for IIGs
[COMPARE L0448] 
[COMPARE L0449] Already covered in Section 4. Up to **80% win rate** vs R-NaD in Goofspiel variants.
[COMPARE L0450] The most impressive recent result for alternatives to standard self-play in IIGs.
[COMPARE L0451] 
[COMPARE L0452] ### Student of Games (2023, published Science Advances)
[COMPARE L0453] 
[COMPARE L0454] Already covered in Section 7. First algorithm to achieve strong performance across both
[COMPARE L0455] perfect AND imperfect information games with a single unified approach.
[COMPARE L0456] 
[COMPARE L0457] ### SPIRAL (2025) -- Self-Play for LLM Reasoning
[COMPARE L0458] 
[COMPARE L0459] - Uses self-play on zero-sum games to improve LLM reasoning
[COMPARE L0460] - Not directly applicable to game AI, but shows self-play principles extending to new domains
[COMPARE L0461] - Source: [github.com/spiral-rl/spiral](https://github.com/spiral-rl/spiral)
[COMPARE L0462] 
[COMPARE L0463] ### Dynamic Discounted CFR (DDCFR, 2024-2025)
[COMPARE L0464] 
[COMPARE L0465] - Automatically adjusts discounting weights in CFR variants
[COMPARE L0466] - Improves convergence rate over vanilla CFR, CFR+, DCFR
[COMPARE L0467] - Relevant for any approach using CFR-based search/training
[COMPARE L0468] 
[COMPARE L0469] ### Auto-designing CFR Algorithms (AIJ 2024)
[COMPARE L0470] 
[COMPARE L0471] - Sciencedirect paper on automatically designing CFR algorithms for IIGs
[COMPARE L0472] - Meta-learning approach: learn which CFR variant works best for a given game
[COMPARE L0473] - Future direction for automating the search component
[COMPARE L0474] 
[COMPARE L0475] ### Self-Play Survey (Aug 2024, arXiv:2408.01072)
[COMPARE L0476] 
[COMPARE L0477] Comprehensive survey classifying all self-play methods in RL:
[COMPARE L0478] - Categorizes by: opponent selection, learning dynamics, convergence properties
[COMPARE L0479] - Identifies open challenges: non-stationarity, catastrophic forgetting, scalability
[COMPARE L0480] - Covers: fictitious play, PSRO, R-NaD, population-based training, league training
[COMPARE L0481] 
[COMPARE L0482] ---
[COMPARE L0483] 
[COMPARE L0484] ## Hydra-Specific Recommendations
[COMPARE L0485] 
[COMPARE L0486] ### Tier 1: Already in Pipeline (High Confidence)
[COMPARE L0487] 
[COMPARE L0488] 1. **Offline RL warm-start (Phase 1)**: CQL or simple behavioral cloning on expert logs.
[COMPARE L0489]    Use this to get a reasonable starting policy before expensive self-play.
[COMPARE L0490] 2. **Oracle distillation (Phase 2)**: Suphx-style asymmetric training. Strong evidence
[COMPARE L0491]    from Suphx's 10-dan results. Already in Hydra's training spec.
[COMPARE L0492] 3. **Multi-task auxiliary heads**: Tenpai, danger, GRP heads already specified. These
[COMPARE L0493]    provide free gradient signal during training.
[COMPARE L0494] 
[COMPARE L0495] ### Tier 2: Strong Evidence, Worth Implementing (Medium Effort)
[COMPARE L0496] 
[COMPARE L0497] 4. **Expert Iteration for Phase 3**: Instead of pure PPO self-play, use search at training
[COMPARE L0498]    time to generate stronger training targets. This is what makes AlphaZero work. Requires
[COMPARE L0499]    implementing a search procedure for 4-player mahjong (significant effort).
[COMPARE L0500] 5. **Opponent action prediction head**: Add as 6th auxiliary head. Predicts opponent discards.
[COMPARE L0501]    Low implementation cost, moderate training signal benefit.
[COMPARE L0502] 
[COMPARE L0503] ### Tier 3: Promising but Premature (Watch List)
[COMPARE L0504] 
[COMPARE L0505] 6. **LAMIR-style learned world model**: Most exciting recent development. 80% WR vs R-NaD.
[COMPARE L0506]    But not tested at mahjong scale, and chance-node modeling is unsolved. Monitor closely.
[COMPARE L0507] 7. **CFR-based training (ReBeL-style)**: Counterfactual reasoning during training could
[COMPARE L0508]    produce more robust policies. Requires significant infrastructure. Consider for Phase 4.
[COMPARE L0509] 8. **Student of Games**: Most general approach. If Hydra later wants to support multiple
[COMPARE L0510]    game types or integrate search+game-theory, SoG is the template.
[COMPARE L0511] 
[COMPARE L0512] ### Not Recommended
[COMPARE L0513] 
[COMPARE L0514] 9. **Inverse RL**: No evidence it improves on hand-crafted rewards for games. High compute
[COMPARE L0515]    cost. The multi-head architecture already captures what IRL would discover.
[COMPARE L0516] 10. **Decision Transformer**: Poor fit for competitive games. Requires conditioning on
[COMPARE L0517]     desired return at inference time, which is awkward for multi-player competitive settings.
[COMPARE L0518] 
[COMPARE L0519] ---
[COMPARE L0520] 
[COMPARE L0521] ## Key Insight for Hydra
[COMPARE L0522] 
[COMPARE L0523] The biggest delta between "standard self-play" and "state-of-the-art training" is
[COMPARE L0524] **search-guided training signal quality**. Every major advance (AlphaZero, ExIt, Student
[COMPARE L0525] of Games, LAMIR) achieves its gains by using search to generate better training targets
[COMPARE L0526] than raw RL returns provide.
[COMPARE L0527] 
[COMPARE L0528] For Hydra, this means: **the planned inference-time search (from the spec) isn't just an
[COMPARE L0529] inference-time upgrade -- it's a training paradigm upgrade.** Once search is working, it
[COMPARE L0530] should be integrated into the training loop (ExIt-style) for Phase 3, not just used at
[COMPARE L0531] test time.
[COMPARE L0532] 
[COMPARE L0533] The compute tradeoff: search at training time is expensive per sample, but the sample
[COMPARE L0534] efficiency gains typically more than compensate. AlphaZero uses ~100x fewer environment
[COMPARE L0535] interactions than pure PPO to reach the same strength, because each interaction produces
[COMPARE L0536] a much higher-quality training signal.
```

## Artifact 11 — Hydra citation index
Artifact id: `references`
Source label: REF
Type: `file_full`
Source: `research/intel/REFERENCES.md`
Why it matters: Single citation index for the repo. Key evidence for whether adjacent work is already cited and what canonical external papers Hydra currently foregrounds.

```markdown
[REF L0001] # Hydra References
[REF L0002] 
[REF L0003] Single source of truth for all citations in the Hydra project.
[REF L0004] 
[REF L0005] ---
[REF L0006] 
[REF L0007] ## Academic Papers
[REF L0008] 
[REF L0009] ### Mahjong AI
[REF L0010] 
[REF L0011] | Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
[REF L0012] |-------|---------|------|-------------|------------------|---------------------|
[REF L0013] | Suphx: Mastering Mahjong with Deep Reinforcement Learning | Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon | 2020 | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) | Oracle guiding, Global Reward Prediction (GRP), run-time policy adaptation, 10-dan achievement on Tenhou. Architecture: 50 residual blocks, 256 filters, separate models per action type with 838 input channels (discard/riichi) and 958 input channels (chow/pong/kong) (Table 2, Figures 4-5). | Core inspiration for oracle distillation and GRP head design |
[REF L0014] | Tjong: A Transformer-based Mahjong AI via Hierarchical Decision-Making and Fan Backward | Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu | 2024 | [CAAI Trans. Intel. Tech.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298) DOI: 10.1049/cit2.12298 | Hierarchical decision-making (action type → tile selection), transformer architecture for game sequences, fan backward reward shaping | Alternative architecture reference; fan backward considered for yaku awareness |
[REF L0015] | Information Set Monte Carlo Tree Search | P. I. Cowling, E. J. Powley, D. Whitehouse | 2012 | [IEEE TCIAIG](https://ieeexplore.ieee.org/document/6203567) | Foundation for handling imperfect information via determinization and information-set sampling | Theoretical basis for imperfect-info game approaches |
[REF L0016] | Real-time Mahjong AI based on Monte Carlo Tree Search (Bakuuchi) | Mizukami et al. | 2014 | IEEE | Pre-deep-learning SOTA using ISMCTS + rule-based heuristics | Historical baseline for MCTS approaches |
[REF L0017] | An Open-Source Interpretable and Reproducible Mahjong Agent (Phoenix) | — | 2021 | [USC CSCI 527 Course Project](https://csci527-phoenix.github.io/documents/Paper.pdf) | Transparent baseline with interpretable decision-making | Open-source baseline reference |
[REF L0018] | Building a Computer Mahjong Player via Deep Convolutional Neural Networks | — | 2018 | IEEE | CNN for Mahjong, baseline methods | Early CNN approach for mahjong |
[REF L0019] | Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction | Li, Wu, Fu, Fu, Zhao, Xing | 2022 | [IEEE CoG](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) | RVR technique for reducing gradient noise from luck variance, oracle critic + expected reward network | Enables training on limited hardware; hand-luck baseline subtraction |
[REF L0020] | Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game | Fu, Liu, Wu, Wang, Yang, Li, Xing, Li, Ma, Fu, Yang | 2022 | [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) | ACH (Actor-Critic Hedge): merges deep RL with Weighted CFR for Nash Equilibrium convergence in imperfect-info games. Core offline training algorithm for Tencent's LuckyJ. | Game-theoretic RL alternative to PPO/DQN; LuckyJ's ACH + OLSS reached 10.68 stable dan on Tenhou |
[REF L0021] | Opponent-Limited Online Search for Imperfect Information Games | Liu, Fu, Fu, Yang | 2023 | [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) | OLSS: imperfect-info subgame solving with opponent-limited tree pruning, orders of magnitude faster than common-knowledge methods. Tested on 2-player mahjong. | Core search component for LuckyJ; search-as-feature integration enables real-time strategy adjustment |
[REF L0022] | Look-ahead Reasoning with a Learned Model in Imperfect Information Games (LAMIR) | Kubicek, Lisy | 2026 | [ICLR 2026](https://openreview.net/forum?id=NnBbr4hI8a) | Learns abstract game models from agent-environment interaction, enables CFR-based depth-limited look-ahead search in imperfect-info games. Tested on 2-player games. [arXiv:2510.05048](https://arxiv.org/abs/2510.05048), [Code](https://github.com/aicenter/lamir) | Inspiration for Hydra's inference-time search direction (historical `SEARCH_PGOI.md` planning surface; not present as a standalone doc in the current repo). Referenced in TACC allocation proposal as "LAS" framing. |
[REF L0023] | Hierarchical CFR with Policy Abstraction in Mahjong | (CFR-p authors) | 2023 | [arXiv:2307.12087](https://arxiv.org/abs/2307.12087) | Applied vanilla CFR to a simplified 2-player 68-tile Mahjong variant with hierarchical policy abstraction. Even this heavily reduced game had ~10^43 leaf nodes before abstraction. Only known CFR application to any Mahjong variant. | Confirms 4-player Mahjong remains intractable for tabular CFR. Supports Hydra's RL-based approach over game-theoretic solving. |
[REF L0024] 
[REF L0025] ### General Game AI
[REF L0026] 
[REF L0027] | Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
[REF L0028] |-------|---------|------|-------------|------------------|---------------------|
[REF L0029] | Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) | Silver et al. | 2017 | [arXiv](https://arxiv.org/abs/1712.01815) | MCTS + neural network self-play, general game learning | Baseline game AI paradigm |
[REF L0030] | Superhuman AI for Multiplayer Poker (Pluribus) | Brown, Sandholm | 2019 | Science | Imperfect-information game solving at scale | Opponent modeling in imperfect-info games |
[REF L0031] | OpenAI Five | OpenAI | 2019 | [OpenAI](https://openai.com/five/) | Large-scale PPO for complex games | Training stability and PPO scaling |
[REF L0032] | AlphaStar: Mastering the Real-Time Strategy Game StarCraft II | Vinyals et al. | 2019 | Nature | League training for multi-agent robustness | League training methodology for Phase 3 |
[REF L0033] | Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning (DeepNash) | Perolat et al. | 2022 | Science | R-NaD for Nash equilibrium approximation | Considered and rejected; Nash approach less suitable for 4-player ranking |
[REF L0034] 
[REF L0035] ### Architecture Components
[REF L0036] 
[REF L0037] | Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
[REF L0038] |-------|---------|------|-------------|------------------|---------------------|
[REF L0039] | Squeeze-and-Excitation Networks | Hu et al. | 2018 | CVPR | SE attention blocks for channel recalibration | Backbone design: dual-pool SE attention in every ResBlock |
[REF L0040] | CBAM: Convolutional Block Attention Module | Woo et al. | 2018 | ECCV | Channel + spatial attention via dual-pool (avg+max) shared MLP | Hydra's SE module uses CBAM's channel attention component (dual-pool shared MLP) |
[REF L0041] | Group Normalization | Wu & He | 2018 | ECCV | Batch-independent normalization | Training stability: GroupNorm(32) replaces BatchNorm |
[REF L0042] | Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) | PPO clipped surrogate objective | Core RL algorithm for Phases 2-3 |
[REF L0043] | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | Transformer architecture | Considered for backbone; used by Kanachan and Tjong |
[REF L0044] | Learning Confidence for Out-of-Distribution Detection | DeVries, Taylor | 2018 | [arXiv:1802.04865](https://arxiv.org/abs/1802.04865) | Confidence estimation as training regularization | Used by NAGA for calibrated action distributions |
[REF L0045] 
[REF L0046] ---
[REF L0047] 
[REF L0048] ## Open Source Projects
[REF L0049] 
[REF L0050] ### Mahjong AI
[REF L0051] 
[REF L0052] | Project | URL | Language | Stars | License | Notes |
[REF L0053] |---------|-----|----------|-------|---------|-------|
[REF L0054] | Mortal | https://github.com/Equim-chan/Mortal | Rust/Python | 1.3K+ | AGPL-3.0-or-later | Primary competitor. ResNet(40 blocks, 192ch) + Channel Attention → DQN(Dueling) + CQL. Reference only — AGPL, cannot derive code. Study: obs encoding (1012×34), action masking (46 actions), GRP head, 1v3 duplicate evaluation. Weights have additional distribution restrictions beyond AGPL. |
[REF L0055] | Kanachan | https://github.com/Cryolite/kanachan | C++/Python | 300+ | Unlicensed | **Transformer encoder (BERT-style)** — two configs: base (~90M params, 12L/768d) and large (~310M params, 24L/1024d). Trained on 65M+ Majsoul rounds (Gold+), zero hand-crafted features. 184 tokens: 33 sparse + 6 numeric + 113 progression + 32 candidates. Pipeline: BC → curriculum fine-tuning → offline RL (IQL/ILQL/CQL). **No published benchmarks** despite multi-year development (public repo created 2021-08-05). Parameter count makes online RL infeasible. ⚠️ No LICENSE file in repo — do not depend on code. |
[REF L0056] | Akochan | https://github.com/critter-mj/akochan | C++ | ~280 | Custom (restrictive, Japanese) | EV-based heuristic engine with explicit suji/kabe/genbutsu analysis. Not ML-based. Matters: its hand-crafted defense logic is a useful sanity check — if Hydra's neural network disagrees with Akochan's defense in obvious spots, something is wrong. Also used as the backend for the original mjai-reviewer. |
[REF L0057] | MahjongAI | https://github.com/erreurt/MahjongAI | Python | ~450 | — | Extensible agent framework with pluggable strategies. Matters less for architecture, more for its Tenhou client implementation — shows how to connect an AI to Tenhou's protocol if we ever need that. |
[REF L0058] | AlphaJong | https://github.com/Jimboom7/AlphaJong | JavaScript | — | — | Browser-based heuristic engine (NOT AlphaZero despite the name). Tunable offense/defense balance via sliders. Matters only as a weak baseline — useful for sanity-checking that Hydra beats simple heuristics by a wide margin. |
[REF L0059] | mjai-manue | https://github.com/gimite/mjai-manue | Ruby | 37 | — | Original MJAI protocol client. Matters as protocol reference — defines the canonical MJAI message format that Hydra must be compatible with. |
[REF L0060] | NAGA | https://dmv.nico/en/articles/mahjong_ai_naga/ | — | — | Commercial | **Pure supervised learning** — 4 independent CNNs (discard, call, riichi, kan) trained on Tenhou Houou game logs via imitation learning. No self-play, no RL. Uses confidence estimation (DeVries & Taylor 2018) as training regularization and Guided Backpropagation (Springenberg et al. 2014) for interpretability. 5 playstyle variants (Omega, Gamma, Nishiki, Hibakari, Kagashi) differentiated by training on different players' game records, not architecture changes. CNN architecture details (layers, filters, input shape) never publicly disclosed — the [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) is the sole official technical document. Achieved 10-dan on Tenhou (26,598 games — source unverified; number does not appear in the DMV article or any locatable public source), current models estimated ~9-dan stable. Not open-source. NAGA's "match%" metric is a common (but imperfect) benchmark. |
[REF L0061] | LuckyJ | https://haobofu.github.io/ | — | — | Commercial | Tencent's mahjong AI (绝艺/JueYi brand). 10-dan on Tenhou in 1,321 games, 10.68 stable dan — strongest known AI. ACH + OLSS architecture, pure self-play. See [COMMUNITY_INSIGHTS § LuckyJ](COMMUNITY_INSIGHTS.md#4-luckyj-tencent) for detailed architecture analysis. |
[REF L0062] 
[REF L0063] ### Analysis & Review Tools
[REF L0064] 
[REF L0065] | Project | URL | Stars | Description |
[REF L0066] |---------|-----|-------|-------------|
[REF L0067] | mjai-reviewer | https://github.com/Equim-chan/mjai-reviewer | 1.1K+ | CLI that generates HTML review reports showing Q-value differences per discard. Primary tool for evaluating Hydra's play quality. Apache-2.0 — can use directly. |
[REF L0068] | mjai-reviewer3p | https://github.com/hidacow/mjai-reviewer3p | — | 3-player (sanma) fork of mjai-reviewer. Matters only if Hydra targets sanma. |
[REF L0069] | killer_mortal_gui | https://github.com/killerducky/killer_mortal_gui | — | Enhanced Mortal review with deal-in heuristic multipliers (ryanmen 3.5×, kanchan suji-trap 2.6×, honor tanki/shanpon 1.7×, etc). Matters: these empirically-tuned danger multipliers are the best public reference for tile danger calibration — useful for validating Hydra's learned defense signals. |
[REF L0070] | crx-mortal | https://github.com/announce/crx-mortal | — | Chrome extension for in-browser Mortal analysis. Low relevance for training. |
[REF L0071] | mjai-batch-review | https://github.com/Xerxes-2/mjai-batch-review | 9 | Batch analyze multiple game logs at once. Matters for large-scale evaluation — when testing Hydra across thousands of games, batch review is faster than one-by-one. |
[REF L0072] 
[REF L0073] ### Mortal Forks
[REF L0074] 
[REF L0075] | Fork | URL | Key Difference |
[REF L0076] |------|-----|----------------|
[REF L0077] | Mortal-Policy | https://github.com/Nitasurin/Mortal-Policy | PPO instead of DQN, GroupNorm instead of BatchNorm, entropy weight tuning. AGPL-3.0, reference only. Matters: closest public reference to Hydra's own architecture choice (PPO + GroupNorm). Study their AWR→PPO transition code path and how they handle the policy gradient with mahjong's 46-action space. |
[REF L0078] 
[REF L0079] ### Components
[REF L0080] 
[REF L0081] | Project | URL | Language | License | Purpose |
[REF L0082] |---------|-----|----------|---------|---------|
[REF L0083] | xiangting | https://github.com/Apricot-S/xiangting | Rust | MIT | Primary shanten library. Compile-time embedded tables (~200KB), `no_std` compatible, 3-player support, returns both shanten number and necessary/unnecessary tile sets. 34× faster than brute-force for replacement tile calculation. Hydra uses this for obs encoding channels (shanten features) and action masking. |
[REF L0084] | xiangting-py | — | Python | MIT | Python bindings for xiangting via PyO3. Useful for training-side shanten calculation if needed. |
[REF L0085] | tomohxx/shanten-number | — | C++ | LGPL-3.0 | Original table-based shanten algorithm that xiangting is derived from. Algorithm reference only — LGPL prevents static linking. Tables: suhai (1.9M entries, ~19.4MB), jihai (78K entries, ~0.78MB). Base-5 encoding for tile state indexing. |
[REF L0086] | PyO3 | https://pyo3.rs/ | Rust | Apache-2.0 | Rust↔Python FFI for exposing game engine bindings to the training loop. |
[REF L0087] | rayon | https://docs.rs/rayon/ | Rust | Apache-2.0 | Work-stealing data parallelism for batch game simulation. |
[REF L0088] | serde / serde_json | https://serde.rs/ | Rust | Apache-2.0 | JSON serialization/deserialization for MJAI protocol parsing. |
[REF L0089] | ndarray | https://docs.rs/ndarray/ | Rust | Apache-2.0 | N-dimensional array operations for constructing observation tensors. |
[REF L0090] | ort | https://docs.rs/ort/ | Rust | Apache-2.0 | ONNX Runtime Rust bindings. Primary inference engine for self-play: loads exported PyTorch model as ONNX, runs forward passes with CUDA EP, CUDA graphs, and I/O binding for <5ms latency. This is the hot path during self-play — inference speed directly limits training throughput. |
[REF L0091] | tract | https://docs.rs/tract/ | Rust | MIT OR Apache-2.0 | Pure Rust ML inference engine (no C++ deps). CPU-only fallback for environments without CUDA. Useful for CI testing and CPU-only deployment. |
[REF L0092] | candle | https://github.com/huggingface/candle | Rust | Apache-2.0 | HuggingFace's Rust ML framework with CUDA and Metal support. Alternative to ONNX path — write inference directly in Rust, avoiding the PyTorch→ONNX export step. Worth evaluating if ONNX export causes accuracy loss or operator compatibility issues. |
[REF L0093] | Burn | https://github.com/tracel-ai/burn | Rust | MIT OR Apache-2.0 | Native Rust training + inference framework with WGPU, CUDA, and LibTorch backends. Long-term option for moving the entire training loop to Rust (eliminating Python entirely). Growing ONNX import support. |
[REF L0094] | tch-rs | — | Rust | MIT OR Apache-2.0 | Rust bindings for LibTorch. Alternative to PyO3 approach — call LibTorch directly from Rust instead of going through Python. Trades Python flexibility for lower FFI overhead. |
[REF L0095] | mahjong (Python) | https://github.com/MahjongRepository/mahjong | Python | MIT | Hand scoring oracle — yaku detection, han/fu/score calculation, validated against 11M+ Tenhou hands. Pin to v1.4.0. Dev dependency for Rust engine verification and test case extraction. |
[REF L0096] | agari | https://github.com/rysb-dev/agari | Rust | MIT (no LICENSE file) | Complete scoring engine (35 yaku, fu, payment, hand decomposition, ~100 unit tests). Most architecturally clean Rust mahjong scorer — study its `HandDecomposition` trait and `Fu` calculation for Hydra's own scoring module. `Cargo.toml` declares MIT but repo lacks a LICENSE file — safe to use as reference. |
[REF L0097] | mahc | https://github.com/DrCheeseFace/mahc | Rust | BSD-3 | Scoring library with explicit `Fu` enum (each fu source is a named variant, not magic numbers). 38 yaku, 30K crates.io downloads. Study the `Fu` enum pattern — makes fu calculation self-documenting and testable vs Mortal's opaque approach. |
[REF L0098] | mahjax | https://github.com/nissymori/mahjax | Python/JAX | Apache-2.0 | JAX-vectorized riichi environment reaching ~1.6M steps/sec on 8×A100 via JIT compilation. Matters for self-play: JAX vectorization can run thousands of games simultaneously on GPU, potentially 10-100x faster than sequential Rust simulator for generating training data. Study their state representation and vectorized game logic. |
[REF L0099] | RiichiEnv | https://github.com/smly/RiichiEnv | Rust/Python | Apache-2.0 | Gym-style RL environment with Rust core + Python bindings, Mortal-compatible MJAI output. Verified correct over 1M+ games. Matters because it provides a ready-made OpenAI Gym interface — if Hydra's training loop uses standard Gym APIs (reset/step/reward), this slots in directly. Also useful as correctness oracle for our own Rust game engine. |
[REF L0100] | Meowjong | https://github.com/VictorZXY/Meowjong | Python | MIT | Only open-source 3-player (sanma) mahjong AI. IEEE CoG 2022. Includes 5 CNN model variants and a Tenhou sanma log downloader. Matters because sanma is a stretch goal — if Hydra ever targets 3-player, this is the only reference implementation with published results. Also validates that CNN architectures work for reduced-player mahjong. |
[REF L0101] | CleanRL | https://github.com/vwxyzjn/cleanrl | Python | MIT | Single-file PPO implementation (~250 lines) with wandb integration. Accompanied by the "37 Implementation Details of PPO" blog post that documents every hyperparameter and trick that matters. Hydra's PPO should be validated against CleanRL's implementation — same clipping, advantage normalization, value loss clipping, entropy coefficient schedule. The blog post is required reading before writing our PPO. |
[REF L0102] | OpenSpiel | https://github.com/google-deepmind/open_spiel | C++/Python | Apache-2.0 | DeepMind's game RL framework with 70+ games, including AlphaZero, MCTS, CFR, and self-play training loops. Matters for Hydra's Phase 3 (league training): study their self-play loop architecture — how they manage opponent pools, ELO tracking, and policy selection. Also has imperfect-info game solvers that inform belief-state approaches. |
[REF L0103] | Microsoft Olive | https://github.com/microsoft/Olive | Python | MIT | End-to-end model optimization: PyTorch → ONNX with quantization, pruning, operator fusion, shape inference via YAML config. Matters for inference speed during self-play: training generates millions of forward passes, so even 2x speedup from INT8 quantization directly halves self-play wall time. Use after model architecture stabilizes. |
[REF L0104] | rlcard | https://github.com/datamllab/rlcard | Python | MIT | RL toolkit with a mahjong environment and pre-built DQN/NFSP agents. Lower fidelity than mahjax/RiichiEnv (simplified rules), but useful for rapid prototyping of reward shaping and training loop mechanics before running on the full environment. |
[REF L0105] | mjai.app | https://github.com/smly/mjai.app | — | AGPL-3.0 | RiichiLab competition platform using MJAI protocol with Docker-based evaluation. Matters because this is a target venue — Hydra must produce MJAI-compatible output to enter competitions and benchmark against other AIs. Study their Docker submission format and evaluation harness. |
[REF L0106] 
[REF L0107] ### Protocol & Infrastructure
[REF L0108] 
[REF L0109] | Project | URL | Description |
[REF L0110] |---------|-----|-------------|
[REF L0111] | mjai | https://github.com/gimite/mjai | Original MJAI protocol server |
[REF L0112] | mjai-gateway | https://github.com/tomohxx/mjai-gateway | MJAI ↔ Tenhou translator |
[REF L0113] 
[REF L0114] ---
[REF L0115] 
[REF L0116] ## Community Resources
[REF L0117] 
[REF L0118] ### Documentation
[REF L0119] 
[REF L0120] | Resource | URL | Content |
[REF L0121] |----------|-----|---------|
[REF L0122] | Mortal Documentation | https://mortal.ekyu.moe | Architecture insights, performance data, playstyle statistics |
[REF L0123] | MJAI Protocol Wiki | https://gimite.net/pukiwiki/index.php?MJAI | Standard protocol specification (⚠️ may require login) |
[REF L0124] | MJAI Web Reviewer | https://mjai.ekyu.moe/ | Web interface for instant game reviews |
[REF L0125] | Tenhou Documentation | https://tenhou.net/man/ | Tenhou log format specification (old `/doc/` path returns 404) |
[REF L0126] | Majsoul API | Various GitHub repos | Log extraction methods via WebSocket capture |
[REF L0127] | NAGA Documentation | https://dmv.nico/en/articles/mahjong_ai_naga/ | Commercial AI architecture overview |
[REF L0128] | Riichi Wiki — NAGA | https://riichi.wiki/Mahjong_AI_%E3%80%8CNAGA%E3%80%8D | Community wiki page on NAGA |
[REF L0129] | Phoenix Paper | https://csci527-phoenix.github.io/documents/Paper.pdf | Open-source reproducible mahjong agent |
[REF L0130] | ONNX Runtime | https://onnxruntime.ai/ | Production inference runtime |
[REF L0131] 
[REF L0132] ### Discussion Sources
[REF L0133] 
[REF L0134] | Source | Topics |
[REF L0135] |--------|--------|
[REF L0136] | Mortal GitHub Issues & Discussions | Known weaknesses, training problems, oracle guiding removal |
[REF L0137] | r/Mahjong (Reddit) | Player perspective on AI behavior, known weaknesses |
[REF L0138] | Discord (Riichi Mahjong) | Community testing, strategy discussion |
[REF L0139] | Tenhou forums | High-level play analysis |
[REF L0140] | Note.com mahjong blogs (Japanese) | 場況 (bakyou) struggles, efficiency vs situational tactics |
[REF L0141] 
[REF L0142] ---
[REF L0143] 
[REF L0144] ## Training Data Sources
[REF L0145] 
[REF L0146] > See [ECOSYSTEM.md § Data Sources & Datasets](ECOSYSTEM.md#3-data-sources--datasets) for the current training data summary. A separate `archive/DATA_SOURCES.md` file is not present in the current repo.
[REF L0147] 
[REF L0148] ---
[REF L0149] 
[REF L0150] ## Algorithm References
[REF L0151] 
[REF L0152] ### Shanten Calculation
[REF L0153] 
[REF L0154] | Resource | Description |
[REF L0155] |----------|-------------|
[REF L0156] | tomohxx Algorithm | Set-based recurrence, O(n) complexity; table-based lookup |
[REF L0157] | tomohxx Tables | Suhai table: 1,940,777 entries × 10 bytes (~19.4 MB); Jihai table: 78,032 entries × 10 bytes (~0.78 MB) |
[REF L0158] | tomohxx Indexing | Base-5 encoding: `tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)` |
[REF L0159] | tomohxx Compressed | shanten_suhai.bin.gz (191 KB), shanten_jihai.bin.gz (5.6 KB) |
[REF L0160] | xiangting Implementation | Rust port with 3-player support |
[REF L0161] | Kanachan xiangting | LOUDS-based TRIE shanten calculator |
[REF L0162] | Mahjong Algorithm Book | Japanese reference, theoretical background |
[REF L0163] | Cryolite (2023) | "A Fast and Space-Efficient Algorithm for Calculating Deficient Numbers" |
[REF L0164] 
[REF L0165] ### Suji / Kabe / Genbutsu
[REF L0166] 
[REF L0167] | Resource | Description |
[REF L0168] |----------|-------------|
[REF L0169] | Japanese Mahjong Strategy Books | Traditional defense theory |
[REF L0170] | Daina Chiba's Defense | Quantitative suji analysis |
[REF L0171] | Tenhou Player Guides | Statistical safety percentages |
[REF L0172] | Suji Safety Note | Suji is approximately 60-70% safe (not 100%); protects only against ryanmen waits |
[REF L0173] | Genbutsu Definition | 100% safe — tiles discarded by or after opponent's riichi |
[REF L0174] | Kabe Definition | All 4 copies visible → no-chance wait; 3 copies = one-chance |
[REF L0175] | Half-suji / Full-suji | One side visible vs both sides visible |
[REF L0176] | killer_mortal_gui Heuristics | Ryanmen 3.5×, Kanchan 0.21×, Kanchan suji-trap 2.6×, Penchan 1.0×, Honor tanki/shanpon 1.7×; modifiers: Dora 1.2×, Ura-suji 1.3×, Matagi early 0.6×, Matagi riichi 1.2×, Red 5 discard 0.14× |
[REF L0177] 
[REF L0178] ### Scoring
[REF L0179] 
[REF L0180] | Resource | Description |
[REF L0181] |----------|-------------|
[REF L0182] | Tenhou Scoring Tables | Standard yaku/fu calculation |
[REF L0183] | World Riichi Championship Rules | International standard |
[REF L0184] | EMA Rules | European standard |
[REF L0185] 
[REF L0186] ---
[REF L0187] 
[REF L0188] ## Benchmark References
[REF L0189] 
[REF L0190] ### Tenhou Ranking
[REF L0191] 
[REF L0192] | Rank | Dan | Approx. Strength |
[REF L0193] |------|-----|-------------------|
[REF L0194] | R2000+ | 7-dan+ | Expert |
[REF L0195] | R1800-2000 | 5-6 dan | Strong |
[REF L0196] | R1600-1800 | 3-4 dan | Intermediate |
[REF L0197] 
[REF L0198] ### AI Achievements
[REF L0199] 
[REF L0200] | AI | Platform | Achievement | Year | Notes |
[REF L0201] |----|----------|-------------|------|-------|
[REF L0202] | NAGA | Tenhou | 10-dan (26,598 games — unverified) | 2018+ | Pure imitation learning; current models ~9-dan stable |
[REF L0203] | Suphx | Tenhou | 10-dan (5,373 games), 8.74 stable | 2020 | SL + RL + oracle guiding; paper states 100+ humans have achieved 10-dan |
[REF L0204] | LuckyJ | Tenhou | **10-dan (1,321 games), 10.68 stable** | 2023 | ACH + OLSS; statistically stronger than both NAGA and Suphx |
[REF L0205] | Mortal | — | **No ranked play** | — | Tenhou rejected Mortal's AI account request ([FAQ](https://github.com/Equim-chan/mjai-reviewer/blob/master/faq.md): "Tenhou rejected my AI account request for Mortal because Mortal was developed by an individual rather than a company"). Community-estimated ~7-dan play strength from mjai-reviewer analysis. |
[REF L0206] | NAGA | Majsoul | Celestial | 2022 | — |
[REF L0207] 
[REF L0208] ---
[REF L0209] 
[REF L0210] ## License Compatibility
[REF L0211] 
[REF L0212] > License policy: See [../infrastructure/INFRASTRUCTURE.md#license-compatibility](../infrastructure/INFRASTRUCTURE.md#license-compatibility)
[REF L0213] 
[REF L0214] ---
[REF L0215] 
[REF L0216] ## GitHub Discussions
[REF L0217] 
[REF L0218] Mortal repository discussions relevant to Hydra design decisions:
[REF L0219] 
[REF L0220] | Discussion # | Topic | Key Insight |
[REF L0221] |-------------|-------|-------------|
[REF L0222] | (source code) | MC returns vs TD | Mortal uses MC returns (not TD) for Q-targets — confirmed from source code (`train.py` Q-target computation). `q_target = gamma^steps_to_done * kyoku_reward` with no bootstrap from next-state Q-values. Hydra follows the same approach. |
[REF L0223] | #27 | Batch size recommendations | Practical guidance on training batch sizes for mahjong RL. |
[REF L0224] | #43 | torch.compile speedup | torch.compile gives 15-20% training speedup on Mortal. Hydra should enable this from day one. |
[REF L0225] | #52 | NextRankPredictor rationale | Auxiliary task that predicts next placement — stabilizes feature learning by giving the backbone a secondary objective beyond Q-values. |
[REF L0226] | #64 | Catastrophic forgetting in online RL | When transitioning from offline (behavioral cloning) to online (self-play), the model forgets offline knowledge. Equim-chan confirms this is a real problem. Hydra must plan for gradual transition with replay buffer mixing. |
[REF L0227] | #70 | DeepCFR for GRP replacement | Community explored using DeepCFR instead of GRP. Conclusion: not practical for 4-player mahjong due to game tree size. |
[REF L0228] | #91 | Mortal-Policy (PPO fork) | Nitasurin's PPO fork open-sourced. Confirms PPO works for mahjong, validates Hydra's algorithm choice. |
[REF L0229] | #102 | Oracle guiding removed | Equim-chan: "didn't bring improvements in practice." Critical for Hydra — Suphx's oracle guiding (our Phase 1 inspiration) was tried and abandoned by Mortal's author. Hydra's oracle approach must differ from Suphx's naive implementation. |
[REF L0230] | #108 | Maximum player score in observations | Discussion about score capping at 30K in observation encoding. Relevant to Hydra's uncapped score encoding decision. |
[REF L0231] 
[REF L0232] ---
[REF L0233] 
[REF L0234] ## GitHub Issues
[REF L0235] 
[REF L0236] Mortal repository issues relevant to Hydra improvements:
[REF L0237] 
[REF L0238] | Issue # | Description |
[REF L0239] |---------|-------------|
[REF L0240] | #111 | Overtake score miscalculation — Mortal miscalculates hand-building near placement thresholds; motivates Hydra's uncapped score encoding |
[REF L0241] | #113 | Rating system closure discussion — community debate on whether to shut down Mortal's rating feature |
[REF L0242] 
[REF L0243] ---
[REF L0244] 
[REF L0245] ## Citation Format
[REF L0246] 
[REF L0247] For academic reference to Hydra:
[REF L0248] 
[REF L0249] ```
[REF L0250] Hydra: A Practical Mahjong AI Architecture
[REF L0251] Combining Oracle Distillation with Explicit Opponent Modeling
[REF L0252] 2026
[REF L0253] ```
[REF L0254] 
[REF L0255] Key techniques to cite:
[REF L0256] - Oracle Distillation: Li et al. (2020) "Suphx"
[REF L0257] - SE-ResNet Backbone: Hu et al. (2018) "Squeeze-and-Excitation Networks"
[REF L0258] - PPO Training: Schulman et al. (2017) "Proximal Policy Optimization"
[REF L0259] - GroupNorm: Wu & He (2018) "Group Normalization"
[REF L0260] - League Training: Vinyals et al. (2019) "AlphaStar"
```

## Artifact 12 — Community and LuckyJ prior-art reconstruction
Artifact id: `community-insights`
Source label: INTEL
Type: `file_full`
Source: `research/intel/COMMUNITY_INSIGHTS.md`
Why it matters: High-signal background on LuckyJ, Suphx, search-as-feature, RVR, and how Hydra reconstructs adjacent production systems from public evidence. Important for deciding whether DCRL is a core omission or only one adjacent citation among many.

```markdown
[INTEL L0001] # Community Insights: Mahjong AI Discussions
[INTEL L0002] 
[INTEL L0003] Research compilation from Reddit, Japanese blogs, RL communities, and public AI analysis discussions. Focused on insights directly relevant to Hydra's development.
[INTEL L0004] 
[INTEL L0005] > **Source volatility note:** Several references link to personal blogs (note.com, hatenablog, Ghost, nicovideo blomaga, modern-jan.com) that may go offline. All critical data points (statistics, architecture details, p-values) are reproduced inline so this document remains self-contained even if external links rot. Last verified: 2026-02-11.
[INTEL L0006] > **Maintenance cadence:** Re-verify external links and source-backed numeric claims quarterly (or before major documentation releases), and update this timestamp when verification is completed.
[INTEL L0007] 
[INTEL L0008] ---
[INTEL L0009] 
[INTEL L0010] ## 1. Mortal Strengths & Weaknesses (r/Mahjong, r/mahjongsoul)
[INTEL L0011] 
[INTEL L0012] ### Confirmed Strengths
[INTEL L0013] 
[INTEL L0014] | Strength | Evidence | Source |
[INTEL L0015] |----------|----------|--------|
[INTEL L0016] | **~7 dan level play** | Better than vast majority of Tokujou players on Tenhou | [r/Mahjong](https://www.reddit.com/r/Mahjong/comments/14ex61l/) |
[INTEL L0017] | **Error detection** | Consistent at identifying clearly bad discards — large eval differences = real mistakes | Same thread |
[INTEL L0018] | **Free & accessible** | Supports Tenhou, Mahjong Soul, and Riichi City log analysis | Multiple sources |
[INTEL L0019] | **4th-place avoidance** | Trained on uma distribution 90/45/0/−135 (similar to MJS ranked) | Mortal documentation |
[INTEL L0020] 
[INTEL L0021] ### Confirmed Weaknesses
[INTEL L0022] 
[INTEL L0023] | Weakness | Details | Hydra Relevance |
[INTEL L0024] |----------|---------|-----------------|
[INTEL L0025] | **Cannot explain reasoning** | No interpretable output — users must infer "why" from raw Q-values | Hydra should consider explainability hooks |
[INTEL L0026] | **Poor future planning** | Struggles with "reading the wall" and multi-turn planning; no lookahead search | Opportunity for search-augmented approach |
[INTEL L0027] | **Sub-optimal multi-threat defense** | When multiple opponents push, may pick tiles safe for now but dangerous if second riichi appears | Multi-player defense modeling gap |
[INTEL L0028] | **Conservative bias** | Recommends folding more often than NAGA or Akochan in equivalent spots | Different training objectives lead to different playstyles |
[INTEL L0029] | **Rule-based agari guard required** | Neural network occasionally fails at basic winning decisions; heuristic override is needed | Raw NN may miss trivial game logic |
[INTEL L0030] | **Not a "source of truth"** | Unlike Stockfish in chess, many decisions are preference-based; high-level players frequently disagree | Mahjong has inherently multiple "correct" plays |
[INTEL L0031] | **Fixed uma optimization** | Trained for one specific point spread; doesn't adapt to different tournament rules | Hydra should parameterize scoring context |
[INTEL L0032] | **No opponent modeling** | Treats all opponents identically; cannot exploit tendencies or detect damaten | Core gap Hydra aims to fill |
[INTEL L0033] 
[INTEL L0034] ### Key Quote
[INTEL L0035] > "In Mahjong, there are many different perfectly playable options. Mortal may have preferences that match with certain high-level players' decisions and not with others." — r/Mahjong community
[INTEL L0036] 
[INTEL L0037] ---
[INTEL L0038] 
[INTEL L0039] ## 2. NAGA vs Mortal Comparison
[INTEL L0040] 
[INTEL L0041] ### NAGA Architecture (Confirmed)
[INTEL L0042] 
[INTEL L0043] NAGA is a **pure supervised learning system** — no self-play, no reinforcement learning. It uses **4 independent CNNs** (discard, call, riichi, kan), each trained on Tenhou Houou table game logs via imitation learning. The CNN architecture details (layers, filters, input shape) have never been publicly disclosed. The [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) is the sole official technical document; there are no academic papers, patents, or conference presentations.
[INTEL L0044] 
[INTEL L0045] **Key technical features:**
[INTEL L0046] - **Confidence estimation** (DeVries & Taylor 2018) — during training, low-confidence predictions incur a penalty and are corrected toward ground truth, improving calibration
[INTEL L0047] - **Guided Backpropagation** (Springenberg 2014) — used for interpretability, visualizing which input features drove each decision
[INTEL L0048] - **Heuristics** — only for final-round winning judgment (avoiding wins that result in last place); everything else is purely CNN output
[INTEL L0049] 
[INTEL L0050] **5 playstyle variants**, each trained on different players' game records:
[INTEL L0051] 
[INTEL L0052] | Model | Style | Training Source |
[INTEL L0053] |-------|-------|----------------|
[INTEL L0054] | **Omega (オメガ)** | Aggressive calling | Watanabe Futoshi (M-League pro) — 100% |
[INTEL L0055] | **Gamma (ガンマ)** | Defensive | One undisclosed private player |
[INTEL L0056] | **Nishiki (ニシキ)** | Balanced | Multiple players (~1/3 Watanabe Futoshi) |
[INTEL L0057] | **Hibakari (ヒバカリ)** | Closed-hand focused | One undisclosed private player |
[INTEL L0058] | **Kagashi (カガシ)** | Extremely aggressive calling | One undisclosed private player (furo rate >40%) |
[INTEL L0059] 
[INTEL L0060] **Performance:** Current models estimated ~9-dan stable on Tenhou. The original NAGA25 reached 10-dan in 26,598 games (source unverified — this number does not appear in the DMV article or any locatable public source). All 5 current models reportedly outperform the original NAGA25. An action with NAGA recommendation rate <5% is flagged as a "bad move" (悪手) — this is a stylistic judgment, not a mathematical optimality claim.
[INTEL L0061] 
[INTEL L0062] **Critical implication for Hydra:** Because NAGA is pure imitation learning, it **cannot exceed its training data**. Its output is a probability distribution reflecting what top humans would likely choose, not an optimized strategy. Long-term strategy (folding, round-aware play) is learned implicitly from behavioral patterns. This fundamental ceiling is why RL-based approaches (Suphx, LuckyJ, and Hydra) have higher potential despite NAGA's commercial polish.
[INTEL L0063] 
[INTEL L0064] **Sources:** [DMV official article](https://dmv.nico/en/articles/mahjong_ai_naga/), [note.com analysis](https://note.com/bold_myrtle4902/n/n8015e4508fe3), [witchverse.hatenablog.com](https://witchverse.hatenablog.com/entry/2025/06/02/124431), [KADOKAWA book](https://www.kadokawa.co.jp/product/322311000197) (co-authored by developer Odagiri Yuuri and pro player Watanabe Futoshi)
[INTEL L0065] 
[INTEL L0066] ### Head-to-Head Differences
[INTEL L0067] 
[INTEL L0068] | Dimension | Mortal | NAGA |
[INTEL L0069] |-----------|--------|------|
[INTEL L0070] | **Playstyle** | More conservative/defensive | More aggressive push tendencies |
[INTEL L0071] | **Riichi decisions** | Hesitant in marginal spots | Strongly favors riichi when +EV |
[INTEL L0072] | **Kan decisions** | Mortal and NAGA frequently disagree on kan timing | NAGA tends more toward aggressive kan |
[INTEL L0073] | **Accessibility** | Free, open-source | Paid, proprietary |
[INTEL L0074] | **Explanation** | None (raw values only) | Human-readable analysis per discard |
[INTEL L0075] | **Calibration** | 7 dan equivalent | 10 dan, with NAGA Rating metrics |
[INTEL L0076] | **Push/fold** | Conservative — values position safety | Calibrated to 4th-avoidance at 7-dan rates |
[INTEL L0077] 
[INTEL L0078] ### NAGA Rating System Limitations
[INTEL L0079] - NAGA's "match%" and "bad move rate" metrics are imperfect proxies for actual strength
[INTEL L0080] - Suphx (9–10 dan) only scored match% of 74.4 and average NAGA Rating of 86.3 — stats comparable to average 7-dan in 2020
[INTEL L0081] - Tencent's LuckyJ hit 10 dan with bad move rates >10% in many games (riichinotes quote: "...LuckyJ hit 10 Dan with bad move rates of >10% in many games.")
[INTEL L0082] - **Takeaway**: Agreement with a specific AI is a poor metric for absolute strength
[INTEL L0083] 
[INTEL L0084] Source: [riichinotes.blogspot.com](https://riichinotes.blogspot.com/2023/06/reviewing-my-first-50-houou-games-with.html)
[INTEL L0085] 
[INTEL L0086] ---
[INTEL L0087] 
[INTEL L0088] ## 3. Push/Fold Mathematics (r/Mahjong)
[INTEL L0089] 
[INTEL L0090] ### Poker Pot Odds Framework for Riichi
[INTEL L0091] A community member adapted poker pot odds into a riichi mahjong EV estimation framework:
[INTEL L0092] 
[INTEL L0093] - **Round EV** = expected point outcome per hand (not per game)
[INTEL L0094] - **Decision**: Push if Round EV > 0 in flat positions (East 1–3)
[INTEL L0095] - **Deal-in rate thresholds**: Based on tile danger level (suji, kabe, genbutsu)
[INTEL L0096] - **Good shape**: Tenpai with 5+ tiles acceptance → more pushable
[INTEL L0097] - **Bad shape**: Tenpai with ≤4 tiles → requires higher reward to justify
[INTEL L0098] 
[INTEL L0099] ### Factors Beyond Round EV
[INTEL L0100] NAGA accounts for 4th-avoidance but base math starts with Round EV. Human exceptions:
[INTEL L0101] 1. **Exploitative folding** — opponent tendency reads
[INTEL L0102] 2. **Lateral movement** — how points flow between other players
[INTEL L0103] 3. **Negative rates** — specific statistical disadvantages in the current position
[INTEL L0104] 
[INTEL L0105] **Hydra Relevance**: Score-aware and placement-aware adjustments on top of base tile EV is exactly what makes an AI jump from "good" to "great." This is a confirmed gap in Mortal.
[INTEL L0106] 
[INTEL L0107] Source: [r/Mahjong Push/Fold thread](https://www.reddit.com/r/Mahjong/comments/17rgvq3/)
[INTEL L0108] 
[INTEL L0109] ---
[INTEL L0110] 
[INTEL L0111] ## 4. LuckyJ (Tencent)
[INTEL L0112] 
[INTEL L0113] ### Identity
[INTEL L0114] 
[INTEL L0115] LuckyJ (ⓃLuckyJ on Tenhou, 绝艺/JueYi brand) is developed by **Tencent** (AI Platform Department). Key researcher: **Haobo Fu** (Principal Research Scientist, Tencent). The 绝艺 brand is shared with Tencent's Go AI that competed in international Go competitions. LuckyJ achieved **10-dan on Tenhou on May 30, 2023** in only **1,321 games** — the most efficient path to 10-dan by any AI.
[INTEL L0116] 
[INTEL L0117] ### Performance
[INTEL L0118] 
[INTEL L0119] | Metric | Value | Source |
[INTEL L0120] |--------|-------|--------|
[INTEL L0121] | Peak Tenhou rank | 10-dan | All sources |
[INTEL L0122] | Stable dan | **10.68** | [Tencent official](https://sports.sina.com.cn/go/2023-07-12/doc-imzamafw0364307.shtml) |
[INTEL L0123] | Games to 10-dan | **1,321** | [haobofu.github.io](https://haobofu.github.io/) |
[INTEL L0124] | vs Suphx | Statistically significantly stronger (p=0.02883) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |
[INTEL L0125] | vs NAGA | Statistically significantly stronger (p=0.00003) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |
[INTEL L0126] 
[INTEL L0127] Early stats (370 games, from pro player Kihara): Average rank 2.259, stable dan 11.25, 1st place 31.3%, last place 15.9%. Source: [ch.nicovideo.jp/kihara/blomaga/ar2149306](https://ch.nicovideo.jp/kihara/blomaga/ar2149306)
[INTEL L0128] 
[INTEL L0129] ### Architecture (Reconstructed from Published Papers)
[INTEL L0130] 
[INTEL L0131] There is **no single "LuckyJ" paper**, but the architecture is reconstructable from Haobo Fu's publication trail:
[INTEL L0132] 
[INTEL L0133] **Component 1 — Offline Training: ACH (Actor-Critic Hedge)**
[INTEL L0134] - Paper: [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) — "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"
[INTEL L0135] - Merges deep RL with Weighted CFR for Nash Equilibrium convergence
[INTEL L0136] - **Pure self-play, zero human data** — trains entirely from scratch
[INTEL L0137] - Lower variance than previous sampled regret methods
[INTEL L0138] 
[INTEL L0139] **Component 2 — Online Search: OLSS (Opponent-Limited Subgame Solving)**
[INTEL L0140] - Paper: [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) — "Opponent-Limited Online Search for Imperfect Information Games"
[INTEL L0141] - Imperfect-info subgame solving with opponent-limited tree pruning
[INTEL L0142] - Orders of magnitude faster than common-knowledge subgame solving
[INTEL L0143] - Explicitly tested on 2-player mahjong
[INTEL L0144] 
[INTEL L0145] **Component 3 — Search-as-Feature Integration (Unpublished)**
[INTEL L0146] - Search results are input as **features** into the policy neural network — they don't directly override the policy (unlike AlphaGo-style MCTS)
[INTEL L0147] - Enables learned integration of search information with trained policy for real-time strategy adjustment
[INTEL L0148] - Source: [Tencent official article](https://modern-jan.com/blog/luckyj_article_ja/)
[INTEL L0149] 
[INTEL L0150] **Component 4 — Training Acceleration: RVR**
[INTEL L0151] - Paper: [IEEE CoG 2022](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) — "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction"
[INTEL L0152] - Same team (Li, Wu, Fu, Fu, Zhao, Xing)
[INTEL L0153] 
[INTEL L0154] ### Observed Playstyle
[INTEL L0155] 
[INTEL L0156] From [note.com analysis](https://note.com/comtefurapote/n/ne7c3668b6e09) and [doramahjong.org](https://doramahjong.org/?p=11393):
[INTEL L0157] - **High meld rate (~35.9%)** — aggressive calling for yakuhai, honitsu, toitoi
[INTEL L0158] - **Defensive priority** — keeps 2 safe tiles at 2-shanten, 1 at 1-shanten; practices early folding on poor hands
[INTEL L0159] - **Shanten backtracking** — reduces efficiency to pursue expensive hands (honitsu, sanshoku, ittsuu)
[INTEL L0160] - **Dama over riichi** on double-mushuji 4-5-6 waits
[INTEL L0161] - **Situational play** shifts dramatically based on rank/score from South 2 onwards
[INTEL L0162] 
[INTEL L0163] ### What Remains Unknown
[INTEL L0164] 
[INTEL L0165] 1. Exact neural network architecture (layers, embedding dims, input encoding)
[INTEL L0166] 2. How ACH and OLSS were adapted from 2-player to 4-player mahjong (the papers demonstrate on 2-player)
[INTEL L0167] 3. Search-as-feature integration details
[INTEL L0168] 4. Compute requirements and inference latency
[INTEL L0169] 5. Whether it uses separate models (like NAGA's 4 CNNs) or a unified architecture
[INTEL L0170] 
[INTEL L0171] ### Comparison Table
[INTEL L0172] 
[INTEL L0173] | Aspect | NAGA | Suphx | LuckyJ |
[INTEL L0174] |--------|------|-------|--------|
[INTEL L0175] | **Training data** | Human expert logs | Human logs + self-play RL | **Pure self-play, zero human data** |
[INTEL L0176] | **Method** | Imitation learning | Imitation → RL | Game-theoretic RL (ACH) |
[INTEL L0177] | **Search** | None | Monte Carlo Policy Adaptation | **OLSS (subgame solving)** |
[INTEL L0178] 
[INTEL L0179] > **Deprecated (2026-03-03):** pMCPA (Monte Carlo Policy Adaptation) removed from inference plans. Requires ~100K trajectories per round, infeasible in real-time even with 90s idle. See RESEARCH_LOG.md entry 4.
[INTEL L0180] | **Theory** | None (pattern matching) | Partial (oracle guiding) | **Nash Equilibrium convergence** |
[INTEL L0181] | **Games to 10-dan** | 26,598 | 5,373 | **1,321** |
[INTEL L0182] | **Stable dan** | ~9.0 (current v2) | 8.74 | **10.68** |
[INTEL L0183] 
[INTEL L0184] Source: [modern-jan.com](https://modern-jan.com/2023/09/06/luckyj_vs_naga_and_suphx/)
[INTEL L0185] 
[INTEL L0186] ### Hydra Relevance
[INTEL L0187] 
[INTEL L0188] LuckyJ proves that combining game-theoretic RL with imperfect-information online search yields dramatically better sample efficiency and higher stable performance than pure RL (Suphx) or pure imitation (NAGA). The search-as-feature integration — where search outputs become neural network inputs rather than direct policy overrides — is the most novel and least documented piece. If Hydra ever adds search, OLSS is the starting point.
[INTEL L0189] 
[INTEL L0190] ---
[INTEL L0191] 
[INTEL L0192] ## 5. AI Analysis Best Practices (Community Guide)
[INTEL L0193] 
[INTEL L0194] ### How to Properly Use AI Review
[INTEL L0195] Key insights from the [Riichi City analysis guide](https://gamesoftrobo.ghost.io/untitled-6/):
[INTEL L0196] 
[INTEL L0197] 1. **Focus on process, not results**: AI makes "correct" moves that sometimes deal into hands — that's not a mistake
[INTEL L0198] 2. **Don't aim for 100% accuracy**: Mortal's authors warn against using accuracy % as skill metric; 100% match = cheating red flag
[INTEL L0199] 3. **Supplement with human reasoning**: AI can't explain "why" — use community and theory to fill gaps
[INTEL L0200] 4. **Efficiency vs Value trade-off**: Mortal often picks most efficient wait, but humans may correctly choose less efficient wait for higher point value (dora targeting)
[INTEL L0201] 5. **Hindsight bias is the enemy**: Evaluate decisions with info available at decision time
[INTEL L0202] 
[INTEL L0203] ### Mortal Analysis Modes
[INTEL L0204] - **"Last Avoidance Type" (ラス回避)**: Optimized for Mahjong Soul ranked play
[INTEL L0205] - **Multiple model versions**: v1 through v4 with evolving architecture
[INTEL L0206] - **Integration**: Built into Riichi City as official AI analysis tool (v4)
[INTEL L0207] 
[INTEL L0208] ---
[INTEL L0209] 
[INTEL L0210] ## 6. Imperfect Information Game RL (r/reinforcementlearning)
[INTEL L0211] 
[INTEL L0212] ### Approaches Discussed
[INTEL L0213] 
[INTEL L0214] | Approach | Description | Applicability to Mahjong |
[INTEL L0215] |----------|-------------|-------------------------|
[INTEL L0216] | **CFR (Counterfactual Regret Minimization)** | Standard for poker; computes Nash equilibria | Game tree too large for direct CFR in mahjong |
[INTEL L0217] | **Standard RL (DQN, PPO, A2C)** | Train against static/self environment | What Mortal uses (DQN) |
[INTEL L0218] | **MARL (Multi-Agent RL)** | Full multi-agent training | Expensive but theoretically ideal |
[INTEL L0219] | **Opponent modeling** | Train against hardcoded/top-tier/human policies | Avoids full MARL complexity |
[INTEL L0220] 
[INTEL L0221] ### ReBeL (Meta AI)
[INTEL L0222] - **Paper**: [arxiv.org/abs/2007.13544](https://arxiv.org/abs/2007.13544)
[INTEL L0223] - **Key innovation**: Combines deep RL + search for imperfect information games
[INTEL L0224] - **Concept**: Expands "state" to probabilistic beliefs about actual state based on common knowledge
[INTEL L0225] - **Limitation**: Proven convergent only for 2-player zero-sum; mahjong is 4-player
[INTEL L0226] - **Hydra Relevance**: Belief-state approach for opponent hand estimation aligns with Hydra's opponent modeling goals
[INTEL L0227] 
[INTEL L0228] ---
[INTEL L0229] 
[INTEL L0230] ## 7. PPO Self-Play Challenges (r/reinforcementlearning)
[INTEL L0231] 
[INTEL L0232] ### The "Fearful Agent" Problem
[INTEL L0233] When using PPO with self-play, a critical failure mode occurs:
[INTEL L0234] 
[INTEL L0235] **Symptoms**:
[INTEL L0236] - Agent becomes overly conservative after experiencing losses
[INTEL L0237] - Focuses entirely on loss avoidance rather than winning
[INTEL L0238] - In mahjong terms: folds everything, never pushes for wins
[INTEL L0239] 
[INTEL L0240] **Root Causes**:
[INTEL L0241] 1. **Large reward disparity** — heavy penalties for losing overwhelm heuristic rewards
[INTEL L0242] 2. **Catastrophic forgetting** — agent forgets winning tactics as it adapts to specific opponents
[INTEL L0243] 3. **Sparse rewards** — long games (1000+ actions) need heuristics but these can break zero-sum balance
[INTEL L0244] 
[INTEL L0245] **Community Solutions**:
[INTEL L0246] 
[INTEL L0247] | Solution | Description |
[INTEL L0248] |----------|-------------|
[INTEL L0249] | **Opponent pool** | Sample from past N network states, not just latest | 
[INTEL L0250] | **Random opponents** | Periodically play vs random to maintain basic competency |
[INTEL L0251] | **Reward normalization** | Balance gradual heuristics with win/loss bonuses |
[INTEL L0252] | **Asymmetric bonuses** | Bonus only to winner; no penalty to loser |
[INTEL L0253] | **Weight freezing** | Freeze opponent weights during training passes |
[INTEL L0254] | **Increased exploration** | Higher entropy to discover new winning strategies |
[INTEL L0255] 
[INTEL L0256] **Hydra Relevance**: Mortal already has catastrophic forgetting issues documented in [GitHub Discussion #64](https://github.com/Equim-chan/Mortal/discussions/64). The opponent pool and reward normalization techniques are directly applicable.
[INTEL L0257] 
[INTEL L0258] Source: [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/comments/1c2ym5s/)
[INTEL L0259] 
[INTEL L0260] ---
[INTEL L0261] 
[INTEL L0262] ## 8. Self-Play Training Best Practices (HuggingFace Deep RL Course)
[INTEL L0263] 
[INTEL L0264] ### Key Hyperparameters for Opponent Pool
[INTEL L0265] 
[INTEL L0266] | Parameter | Effect |
[INTEL L0267] |-----------|--------|
[INTEL L0268] | `window` | Number of saved opponent policies. Larger = more diverse training |
[INTEL L0269] | `save_steps` | Steps between saves. Higher = wider skill range in pool |
[INTEL L0270] | `play_against_latest_ratio` | Probability of facing current vs historical policy |
[INTEL L0271] | `swap_steps` | How often opponents rotate |
[INTEL L0272] 
[INTEL L0273] ### ELO as Training Metric
[INTEL L0274] - **Why ELO > cumulative reward**: In adversarial games, reward depends on opponent skill. ELO measures relative skill in zero-sum context
[INTEL L0275] - **K-factor**: Maximum adjustment per game; controls rating volatility
[INTEL L0276] - **Self-correcting**: Better opponents yield more points on victory
[INTEL L0277] 
[INTEL L0278] ### Core Trade-off
[INTEL L0279] > Balance final policy's **skill level** and **generality** against **training stability**.
[INTEL L0280] 
[INTEL L0281] Training against slowly-changing adversaries = more stable but risk of overfitting to specific behaviors.
[INTEL L0282] 
[INTEL L0283] Source: [HuggingFace Deep RL Course Unit 7](https://huggingface.co/learn/deep-rl-course/unit7/self-play)
[INTEL L0284] 
[INTEL L0285] ---
[INTEL L0286] 
[INTEL L0287] ## 9. Japanese Community Sources
[INTEL L0288] 
[INTEL L0289] ### Shanten Algorithm (Qiita — tomohxx)
[INTEL L0290] 
[INTEL L0291] The standard shanten algorithm used by Mortal and most other mahjong AIs:
[INTEL L0292] 
[INTEL L0293] **Mathematical Foundation**:
[INTEL L0294] - Shanten S(h) = T(h) − 1, where T = minimum tile exchanges to tenpai
[INTEL L0295] - Distance function: d(h, g) = ½ Σ(|h_i − g_i| + h_i − g_i) over 34 tile types
[INTEL L0296] - Special-case formulas for Chiitoitsu (7 pairs) and Kokushi (13 orphans)
[INTEL L0297] 
[INTEL L0298] **DP Algorithm for Regular Hands**:
[INTEL L0299] 1. Break hand into 4 groups (man, pin, sou, honors)
[INTEL L0300] 2. Precompute partial replacement numbers for all possible suit combinations (~5^9 states)
[INTEL L0301] 3. Merge groups via DP: t^(n+1)_m = min over splits of meld counts
[INTEL L0302] 4. Result: t^(3)_4 = shanten for 4 melds + 1 pair
[INTEL L0303] 
[INTEL L0304] **Performance**: O(1) after precomputation; independent of hand size or shanten value.
[INTEL L0305] 
[INTEL L0306] Source: [Qiita (tomohxx)](https://qiita.com/tomohxx/items/75b5f771285e1334c0a5), [GitHub](https://github.com/tomohxx/shanten-number)
[INTEL L0307] 
[INTEL L0308] ### Japanese Mahjong AI Development Blog (TadaoYamaoka)
[INTEL L0309] 
[INTEL L0310] An independent developer documenting their attempt to build a mahjong AI from scratch using PPO:
[INTEL L0311] 
[INTEL L0312] **Key Technical Points**:
[INTEL L0313] - Uses **PPO** (vs Mortal's DQN) as the baseline algorithm
[INTEL L0314] - **Reward variance reduction**: Value model uses "global information" (including opponent private tiles) to reduce noise from random initial hands
[INTEL L0315] - **Zero-sum property**: Loss function designed so sum of 4 players' predicted values = 0
[INTEL L0316] - Referenced **LuckyJ** (Tencent's unpublished AI) which uses search-based techniques for higher performance
[INTEL L0317] - **Search excluded from baseline** due to implementation complexity
[INTEL L0318] 
[INTEL L0319] **Hydra Relevance**: Confirms PPO as viable alternative to DQN for mahjong; validates reward variance reduction with global info.
[INTEL L0320] 
[INTEL L0321] Source: [TadaoYamaoka's blog](https://tadaoyamaoka.hatenablog.com/entry/2023/10/03/233925)
[INTEL L0322] 
[INTEL L0323] ### Mortal User Reviews (note.com, ai-bo.jp)
[INTEL L0324] 
[INTEL L0325] Japanese community consensus:
[INTEL L0326] - Mortal rated as "excellent" (優秀) by regular NAGA users
[INTEL L0327] - Primary value: Free + supports Mahjong Soul log import
[INTEL L0328] - Primary frustration: No explanation of reasoning (users must infer intent)
[INTEL L0329] - Comparison verdict: NAGA has higher analysis power but costs money
[INTEL L0330] 
[INTEL L0331] ---
[INTEL L0332] 
[INTEL L0333] ## 10. Mortal Architecture Deep Dive
[INTEL L0334] 
[INTEL L0335] > See [MORTAL_ANALYSIS.md](MORTAL_ANALYSIS.md) for the full architecture analysis including DQN head evolution (v1–v4), training loss components, distributed training, and 1v3 duplicate evaluation protocol.
[INTEL L0336] 
[INTEL L0337] ---
[INTEL L0338] 
[INTEL L0339] ## 11. Defense & Betaori Analysis
[INTEL L0340] 
[INTEL L0341] ### Standard Defense Framework (riichi.wiki, community)
[INTEL L0342] 
[INTEL L0343] **Tile Safety Hierarchy** (from safest to least safe):
[INTEL L0344] 1. **Genbutsu**: 100% safe (already discarded by riichi declarer)
[INTEL L0345] 2. **Suji**: ~94% safe against riichi
[INTEL L0346] 3. **Kabe (wall)**: Safe when all 4 copies of connecting tiles are visible
[INTEL L0347] 4. **Honor tiles**: Variable safety based on game state
[INTEL L0348] 5. **Middle tiles (4-5-6)**: Most dangerous
[INTEL L0349] 
[INTEL L0350] ### AI Defense Limitations
[INTEL L0351] - **No damaten detection**: AIs can't reliably detect hidden tenpai (opponent waiting without riichi)
[INTEL L0352] - **Multi-player defense**: Folding against one opponent may push dangerous tiles toward another
[INTEL L0353] - **Score context**: When to push depends heavily on current scores/placement — Mortal uses fixed uma
[INTEL L0354] 
[INTEL L0355] ### Push/Fold Decision Framework
[INTEL L0356] Community consensus ("2 of 3" rule):
[INTEL L0357] 1. Am I in **tenpai**?
[INTEL L0358] 2. Do I have a **good wait** (5+ tiles)?
[INTEL L0359] 3. Is my hand **high value**?
[INTEL L0360] 
[INTEL L0361] If 2 of 3 → push. Otherwise → fold. Additional factors: round number, current scores, danger level of tiles to push.
[INTEL L0362] 
[INTEL L0363] ---
[INTEL L0364] 
[INTEL L0365] ## 12. Mahjong AI Landscape Summary
[INTEL L0366] 
[INTEL L0367] | AI | Level | Architecture | Open Source | Analysis | Key Trait |
[INTEL L0368] |----|-------|-------------|-------------|----------|-----------|
[INTEL L0369] | **Mortal** | ~7 dan | SE-ResNet + Dueling DQN | ✅ Yes | Free log review | Best open-source option |
[INTEL L0370] | **NAGA** | ~9 dan (stable) | 4 CNNs, pure imitation learning | ❌ No | Paid, detailed | 5 playstyle variants trained on different players |
[INTEL L0371] | **Suphx** | 8.74 dan (stable) | ResNet + Oracle guiding | ❌ No | Replay viewing only | First to reach 10 dan; GRP + oracle pioneering |
[INTEL L0372] | **LuckyJ** | **10.68 dan (stable)** | ACH (RL+CFR) + OLSS (search) | ❌ No | None | Strongest known; game-theoretic RL + online search |
[INTEL L0373] | **Kanachan** | Unknown (no benchmarks) | Transformer (BERT, ~90-310M params) | ✅ Yes (⚠️ no LICENSE file) | None | Zero hand-crafted features; impractical for online RL |
[INTEL L0374] | **Akochan** | ~8 dan | EV-based heuristic (not ML) | ✅ Yes | Reviewer tool | Explicit suji/kabe/genbutsu defense logic |
[INTEL L0375] | **Bakuuchi** | 9 dan | ISMCTS | ❌ No | None | Legacy, outperformed |
[INTEL L0376] 
[INTEL L0377] ---
[INTEL L0378] 
[INTEL L0379] ## 13. Key Takeaways for Hydra
[INTEL L0380] 
[INTEL L0381] > **Ownership note:** This section captures community-observed signals and hypotheses. Canonical Mortal limitation statements live in `MORTAL_ANALYSIS.md`; current Hydra architecture-level deltas live across `README.md`, `HYDRA_FINAL.md`, `HYDRA_RECONCILIATION.md`, and focused design docs like `OPPONENT_MODELING.md`.
[INTEL L0382] 
[INTEL L0383] ### Confirmed Gaps in Existing AIs (Opportunities for Hydra)
[INTEL L0384] 
[INTEL L0385] 1. **Opponent Modeling**: No existing AI models opponent tendencies or detects damaten
[INTEL L0386] 2. **Score/Placement Awareness**: Mortal uses fixed uma; dynamic adjustment is an open problem  
[INTEL L0387] 3. **Multi-Turn Planning**: LuckyJ uses online search (OLSS, ICML 2023) and is the strongest AI — but the 4-player adaptation and search-as-feature integration are unpublished. No open-source AI uses search.
[INTEL L0388] 4. **Explainability**: All AIs are black boxes; interpretable decision factors would be novel
[INTEL L0389] 5. **Multi-Player Defense**: Simultaneous defense against 2+ threats is poorly handled
[INTEL L0390] 6. **Adaptive Playstyle**: NAGA offers multiple styles but doesn't adapt dynamically per-game
[INTEL L0391] 
[INTEL L0392] ### Training Methodology Recommendations
[INTEL L0393] 
[INTEL L0394] 1. **PPO over DQN**: TadaoYamaoka's work and community discussion suggest PPO is viable and may be preferable for policy-based mahjong AI
[INTEL L0395] 2. **Reward Variance Reduction**: Use global info in value model to distinguish skill from luck
[INTEL L0396] 3. **Opponent Pool**: Essential for preventing catastrophic forgetting and the "fearful agent" problem
[INTEL L0397] 4. **CQL for Offline**: Mortal's CQL integration prevents Q-value overestimation on unseen actions
[INTEL L0398] 5. **ELO Tracking**: Better progress metric than cumulative reward during self-play training
[INTEL L0399] 6. **1v3 Duplicate**: Gold standard evaluation method; eliminates variance
[INTEL L0400] 
[INTEL L0401] ### Community Red Flags
[INTEL L0402] 
[INTEL L0403] - **100% AI accuracy = cheating indicator**: Mortal is used for real-time assistance (Akagi tool); this is a known anti-cheat concern
[INTEL L0404] - **Playstyle subjectivity**: No single "correct" play in many mahjong situations; AI agreement is a weak proxy for quality
[INTEL L0405] - **AI metrics are imperfect**: NAGA Rating, match%, and bad move rate don't reliably predict actual playing strength
```

## Artifact 13 — Belief-state and oracle-guiding survey
Artifact id: `belief-state-survey`
Source label: EVIDENCE
Type: `file_full`
Source: `research/evidence/BELIEF_STATE_SURVEY.md`
Why it matters: Relevant evidence for partial observability, oracle guiding, belief-state training, and teacher / student framing under imperfect information.

```markdown
[EVIDENCE L0001] # Belief State Tracking & Opponent Hand Inference in Tile/Card Games
[EVIDENCE L0002] ## Literature Survey for Hydra Mahjong AI
[EVIDENCE L0003] 
[EVIDENCE L0004] **Date**: 2026-03-02
[EVIDENCE L0005] **Purpose**: Identify best approach for sampling plausible opponent hands (100+ samples/sec) from Sinkhorn head marginal probabilities while maintaining tile-count constraints.
[EVIDENCE L0006] 
[EVIDENCE L0007] ---
[EVIDENCE L0008] 
[EVIDENCE L0009] ## Table of Contents
[EVIDENCE L0010] 
[EVIDENCE L0011] 1. [Determinization: GIB (Bridge)](#1-determinization-gib-bridge)
[EVIDENCE L0012] 2. [CFR-Based: Pluribus / Libratus / DeepStack (Poker)](#2-cfr-based-pluribus--libratus--deepstack-poker)
[EVIDENCE L0013] 3. [ISMCTS: Information Set Monte Carlo Tree Search](#3-ismcts-information-set-monte-carlo-tree-search)
[EVIDENCE L0014] 4. [Bayesian Hand Inference in Mahjong: Suphx & Mortal](#4-bayesian-hand-inference-in-mahjong-suphx--mortal)
[EVIDENCE L0015] 5. [Constraint-Based Belief States with Belief Propagation](#5-constraint-based-belief-states-with-belief-propagation)
[EVIDENCE L0016] 6. [MCMC / Gibbs Sampling for History Generation](#6-mcmc--gibbs-sampling-for-history-generation)
[EVIDENCE L0017] 7. [Sinkhorn Operator for Constrained Sampling](#7-sinkhorn-operator-for-constrained-sampling)
[EVIDENCE L0018] 8. [Synthesis: Recommended Approach for Hydra](#8-synthesis-recommended-approach-for-hydra)
[EVIDENCE L0019] 
[EVIDENCE L0020] ---
[EVIDENCE L0021] 
[EVIDENCE L0022] ## 1. Determinization: GIB (Bridge)
[EVIDENCE L0023] 
[EVIDENCE L0024] **Paper**: Ginsberg, "GIB: Imperfect Information in a Computationally Challenging Game" (JAIR 2001)
[EVIDENCE L0025] **Link**: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume14/ginsberg01a.pdf
[EVIDENCE L0026] **arXiv**: https://arxiv.org/abs/1106.0669
[EVIDENCE L0027] 
[EVIDENCE L0028] ### Core Idea
[EVIDENCE L0029] GIB handles bridge (a 52-card imperfect-information game) via **Monte Carlo determinization**: sample N possible card deals consistent with observations, solve each as a perfect-information game, aggregate results.
[EVIDENCE L0030] 
[EVIDENCE L0031] ### Algorithm (Section 3, Algorithm 3.0.1)
[EVIDENCE L0032] 1. Construct set D of deals consistent with bidding + play history
[EVIDENCE L0033] 2. For each move m and each deal d, evaluate double-dummy result s(m, d)
[EVIDENCE L0034] 3. Return move m maximizing SUM_d s(m, d)
[EVIDENCE L0035] 
[EVIDENCE L0036] ### Sample Counts
[EVIDENCE L0037] | Context | Samples | Time |
[EVIDENCE L0038] |---------|---------|------|
[EVIDENCE L0039] | Production play | **50 deals** | ~1-2 sec total |
[EVIDENCE L0040] | Extended resource | **100 deals** | N/A |
[EVIDENCE L0041] | World Championship (1998) | **500 deals** | N/A |
[EVIDENCE L0042] 
[EVIDENCE L0043] ### Bayesian Weighting (Section 3, p.323)
[EVIDENCE L0044] Deals aren't equally weighted. GIB uses Bayesian inference on play history:
[EVIDENCE L0045] - If a player fails to play a King, GIB adjusts P(player holds King) via Bayes' rule
[EVIDENCE L0046] - Weighted evaluation: SUM_d w_d * s(m, d) where w_d is deal weight
[EVIDENCE L0047] 
[EVIDENCE L0048] ### The Strategy Fusion Problem (Section 3, p.319)
[EVIDENCE L0049] The fundamental flaw of determinization: the solver assumes different decisions can be made for different sampled worlds, even though those worlds are indistinguishable to the player. Example: if the Queen might be with West or East, determinization says "play line A when West has it, line B when East has it" -- but you can't know which world you're in!
[EVIDENCE L0050] 
[EVIDENCE L0051] ### Fix: Achievable Sets (Section 7.1, Definition 7.1.1)
[EVIDENCE L0052] Instead of maximizing average tricks, find a **single plan** that wins for a maximal subset of worlds. Forces commitment to one line of play.
[EVIDENCE L0053] 
[EVIDENCE L0054] ### Relevance to Hydra
[EVIDENCE L0055] - **Directly applicable**: Mahjong is structurally similar to bridge (hidden hands, known total tile counts)
[EVIDENCE L0056] - **50 samples was sufficient** for competitive bridge -- Mahjong may need more due to 3 opponents vs 1
[EVIDENCE L0057] - **Bayesian weighting from play history** maps to weighting by discard patterns
[EVIDENCE L0058] - **Strategy fusion is a real risk** if we use determinized search
[EVIDENCE L0059] 
[EVIDENCE L0060] ---
[EVIDENCE L0061] 
[EVIDENCE L0062] ## 2. CFR-Based: Pluribus / Libratus / DeepStack (Poker)
[EVIDENCE L0063] 
[EVIDENCE L0064] ### 2a. Pluribus (6-player No-Limit Hold'em)
[EVIDENCE L0065] 
[EVIDENCE L0066] **Paper**: Brown & Sandholm, "Superhuman AI for Multiplayer Poker" (Science, 2019)
[EVIDENCE L0067] **Link**: https://www.science.org/doi/10.1126/science.aay2400
[EVIDENCE L0068] 
[EVIDENCE L0069] #### Blueprint Strategy
[EVIDENCE L0070] - Computed **offline via self-play** using Linear CFR (a variant of MCCFR)
[EVIDENCE L0071] - 8 days, 12,400 CPU core-hours, ~$144 compute cost
[EVIDENCE L0072] - Uses card abstraction (bucketing similar hands) and action abstraction (limited bet sizes)
[EVIDENCE L0073] 
[EVIDENCE L0074] #### Real-Time Search
[EVIDENCE L0075] Pluribus does NOT sample opponent hands naively. Instead:
[EVIDENCE L0076] 1. After round 1, performs **depth-limited search** using CFR on the current subgame
[EVIDENCE L0077] 2. Tracks **reach probabilities** over all possible opponent hands (a distribution, not samples)
[EVIDENCE L0078] 3. At the depth limit, each remaining player chooses among **k=4 continuation strategies**:
[EVIDENCE L0079]    - Blueprint strategy
[EVIDENCE L0080]    - Fold-biased blueprint
[EVIDENCE L0081]    - Call-biased blueprint  
[EVIDENCE L0082]    - Raise-biased blueprint
[EVIDENCE L0083] 4. Leaf values computed by **rolling out** under chosen continuation strategies
[EVIDENCE L0084] 
[EVIDENCE L0085] #### Key Insight
[EVIDENCE L0086] Pluribus doesn't "sample hands" -- it works with the full distribution over information sets. This avoids strategy fusion entirely but requires CFR infrastructure.
[EVIDENCE L0087] 
[EVIDENCE L0088] ### 2b. Libratus (Heads-Up No-Limit Hold'em)
[EVIDENCE L0089] 
[EVIDENCE L0090] **Paper**: Brown & Sandholm, "Superhuman AI for Heads-Up No-Limit Poker" (Science, 2018)
[EVIDENCE L0091] **Link**: https://www.science.org/doi/10.1126/science.aao1733
[EVIDENCE L0092] 
[EVIDENCE L0093] Three modules:
[EVIDENCE L0094] 1. **Blueprint**: Coarse strategy via CFR on abstracted game
[EVIDENCE L0095] 2. **Nested Subgame Solving**: Repeatedly refines strategy in real-time with safety guarantees
[EVIDENCE L0096] 3. **Self-Improver**: Fills in missing branches overnight
[EVIDENCE L0097] 
[EVIDENCE L0098] #### Private Information Handling
[EVIDENCE L0099] Libratus maintains a **range** (probability distribution over opponent's possible hands) and updates it based on observed actions. Nested subgame solving ensures the refined strategy is **safe** (never worse than blueprint).
[EVIDENCE L0100] 
[EVIDENCE L0101] ### 2c. DeepStack (Heads-Up No-Limit Hold'em)
[EVIDENCE L0102] 
[EVIDENCE L0103] **Paper**: Moravcik et al., "DeepStack: Expert-Level AI in Heads-Up No-Limit Poker" (Science, 2017)
[EVIDENCE L0104] 
[EVIDENCE L0105] Key innovations:
[EVIDENCE L0106] - **Continual Resolving**: Re-solves the game from scratch at every decision point, maintaining consistency via counterfactual values
[EVIDENCE L0107] - **Neural Network Leaf Evaluation**: Instead of playing to completion, uses trained neural nets to estimate values at depth limit
[EVIDENCE L0108] - **No pre-computed blueprint** needed (unlike Libratus/Pluribus)
[EVIDENCE L0109] 
[EVIDENCE L0110] ### Relevance to Hydra
[EVIDENCE L0111] - **CFR approach is theoretically superior** but requires building full CFR infrastructure
[EVIDENCE L0112] - **4-player Mahjong is harder** than 2-player poker for CFR (exponentially larger info sets)
[EVIDENCE L0113] - **Pluribus's k=4 continuation strategies** is clever -- could adapt for Mahjong
[EVIDENCE L0114] - **DeepStack's neural leaf evaluation** is closest to what Hydra already does (value head)
[EVIDENCE L0115] - **Key takeaway**: If you can afford it, work with distributions not samples. If you must sample (for speed), weight carefully.
[EVIDENCE L0116] 
[EVIDENCE L0117] ---
[EVIDENCE L0118] 
[EVIDENCE L0119] ## 3. ISMCTS: Information Set Monte Carlo Tree Search
[EVIDENCE L0120] 
[EVIDENCE L0121] **Paper**: Cowling, Powley, Whitehouse, "Information Set Monte Carlo Tree Search" (IEEE TCIAIG, 2012)
[EVIDENCE L0122] **Link**: https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf
[EVIDENCE L0123] 
[EVIDENCE L0124] ### How It Works
[EVIDENCE L0125] Instead of searching multiple determinized game trees, ISMCTS builds a **single tree where nodes are information sets** (not states).
[EVIDENCE L0126] 
[EVIDENCE L0127] 1. At each iteration, sample a determinization d from the root information set
[EVIDENCE L0128] 2. Descend the information-set tree, only visiting nodes compatible with d
[EVIDENCE L0129] 3. Expand, simulate (rollout), and backpropagate as normal MCTS
[EVIDENCE L0130] 4. Statistics accumulate across determinizations at each info-set node
[EVIDENCE L0131] 
[EVIDENCE L0132] ### Avoiding Strategy Fusion
[EVIDENCE L0133] Because statistics are collected at **information set nodes** (not state nodes), the algorithm finds moves that are good across many possible hidden states, not just one specific determinization.
[EVIDENCE L0134] 
[EVIDENCE L0135] ### Three Variants
[EVIDENCE L0136] | Variant | Description | Use Case |
[EVIDENCE L0137] |---------|-------------|----------|
[EVIDENCE L0138] | **SO-ISMCTS** | Single observer, root player's info sets | Simplest, good for card games |
[EVIDENCE L0139] | **SO-ISMCTS+POM** | Handles partially observable opponent moves | Games where you can't see opponent actions |
[EVIDENCE L0140] | **MO-ISMCTS** | Separate tree per player | Most accurate opponent modeling |
[EVIDENCE L0141] 
[EVIDENCE L0142] ### Performance Numbers
[EVIDENCE L0143] - **10,000 iterations per decision** (~1 second on 2010 hardware)
[EVIDENCE L0144] - Tested on Lord of the Rings: The Confrontation, Phantom games, and Dou Di Zhu (Chinese card game)
[EVIDENCE L0145] - Dou Di Zhu has **avg 88 legal moves per state** -- comparable to Mahjong complexity
[EVIDENCE L0146] 
[EVIDENCE L0147] ### Relevance to Hydra
[EVIDENCE L0148] - **ISMCTS is the natural fit** for determinization-based search in Mahjong
[EVIDENCE L0149] - Solves strategy fusion without needing full CFR
[EVIDENCE L0150] - **10K iterations in 1 second (2010 hardware)** -- on modern hardware, easily 100K+
[EVIDENCE L0151] - **MO-ISMCTS with per-player trees** would allow opponent modeling
[EVIDENCE L0152] - Can be **combined with neural network rollout policy** (replace random rollouts with Hydra's policy head)
[EVIDENCE L0153] 
[EVIDENCE L0154] ---
[EVIDENCE L0155] 
[EVIDENCE L0156] ## 4. Bayesian Hand Inference in Mahjong: Suphx & Mortal
[EVIDENCE L0157] 
[EVIDENCE L0158] ### 4a. Suphx (Microsoft, 2020)
[EVIDENCE L0159] 
[EVIDENCE L0160] **Paper**: Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" (arXiv:2003.13590)
[EVIDENCE L0161] **Link**: https://arxiv.org/abs/2003.13590
[EVIDENCE L0162] 
[EVIDENCE L0163] #### Imperfect Information Handling
[EVIDENCE L0164] Suphx does NOT explicitly predict opponent hands. Instead it uses:
[EVIDENCE L0165] 
[EVIDENCE L0166] 1. **Oracle Guiding** (Section 3.3): Train an "oracle agent" with perfect information (sees all tiles), then gradually transition to a normal agent via **perfect feature dropout** (probability gamma_t decays from 1 to 0). The normal agent implicitly learns to infer hidden state.
[EVIDENCE L0167] 
[EVIDENCE L0168] 2. **Run-time Policy Adaptation (pMCPA)** (Section 3.4): At the start of each round:
[EVIDENCE L0169]    - Randomly sample opponent tiles and wall tiles from remaining pool
[EVIDENCE L0170]    - Run K rollouts using offline policy
[EVIDENCE L0171]    - Fine-tune policy using these K trajectories
[EVIDENCE L0172]    - Play with adapted policy
[EVIDENCE L0173] 
[EVIDENCE L0174] #### Feature Encoding
[EVIDENCE L0175] - Input: 34 x 838 (discard model) or 34 x 958 (other models)
[EVIDENCE L0176] - Includes: discard sequences of all players, open melds, accumulated scores, dealer info, riichi bets
[EVIDENCE L0177] - Over 10^48 hidden states per information set
[EVIDENCE L0178] 
[EVIDENCE L0179] #### Key Insight
[EVIDENCE L0180] Suphx's pMCPA is essentially **determinization + policy adaptation**. It samples random opponent hands and adapts, rather than trying to infer the exact distribution.
[EVIDENCE L0181] 
[EVIDENCE L0182] > **Deprecated (2026-03-03):** pMCPA removed from inference plans. Requires ~100K trajectories per round, infeasible in real-time even with 90s idle. See RESEARCH_LOG.md entry 4.
[EVIDENCE L0183] 
[EVIDENCE L0184] ### 4b. Mortal (Equim-chan, Open Source)
[EVIDENCE L0185] 
[EVIDENCE L0186] **Repo**: https://github.com/Equim-chan/Mortal
[EVIDENCE L0187] **Docs**: https://mortal.ekyu.moe/
[EVIDENCE L0188] 
[EVIDENCE L0189] Mortal uses a similar approach to Suphx but open-source:
[EVIDENCE L0190] - SE-ResNet architecture (the basis for Hydra's architecture)
[EVIDENCE L0191] - No explicit opponent hand inference
[EVIDENCE L0192] - Implicit learning through self-play RL
[EVIDENCE L0193] - 40K hanchans/hour simulation speed
[EVIDENCE L0194] 
[EVIDENCE L0195] ### Key Mahjong-Specific Insight
[EVIDENCE L0196] Neither Suphx nor Mortal do explicit Bayesian hand inference. Both rely on:
[EVIDENCE L0197] 1. Neural networks learning implicit belief representations
[EVIDENCE L0198] 2. Safety features (which tiles are "safe" based on discards)
[EVIDENCE L0199] 3. Oracle training to transfer perfect-info reasoning to imperfect-info play
[EVIDENCE L0200] 
[EVIDENCE L0201] **This is the gap Hydra's Sinkhorn head fills** -- explicit probabilistic tile allocation.
[EVIDENCE L0202] 
[EVIDENCE L0203] ---
[EVIDENCE L0204] 
[EVIDENCE L0205] ## 5. Constraint-Based Belief States with Belief Propagation
[EVIDENCE L0206] 
[EVIDENCE L0207] **Paper**: "Modeling Uncertainty: Constraint-Based Belief States in Imperfect Information Games" (2025)
[EVIDENCE L0208] **Link**: https://arxiv.org/abs/2507.19263
[EVIDENCE L0209] 
[EVIDENCE L0210] ### THIS IS THE MOST DIRECTLY RELEVANT PAPER FOR HYDRA.
[EVIDENCE L0211] 
[EVIDENCE L0212] ### Core Framework
[EVIDENCE L0213] Model hidden tile/piece identities as a **Constraint Satisfaction Problem (CSP)**:
[EVIDENCE L0214] - **Variables**: Each unknown piece/tile
[EVIDENCE L0215] - **Domains**: Set of possible identities for each variable
[EVIDENCE L0216] - **Global Cardinality Constraints (GCC)**: "Number of occurrences of each identity must remain within allowed limits" -- i.e., exactly 4 copies of each tile type in Mahjong
[EVIDENCE L0217] 
[EVIDENCE L0218] ### Belief Propagation for Marginals
[EVIDENCE L0219] 1. Reinterpret CSP as a **factor graph** (variables = variable nodes, constraints = factor nodes)
[EVIDENCE L0220] 2. Decompose each GCC into simpler count constraints (one per identity)
[EVIDENCE L0221] 3. Run iterative BP message passing to estimate **marginal probabilities**
[EVIDENCE L0222] 4. Result: P(tile_i has identity_j) for every unknown tile
[EVIDENCE L0223] 
[EVIDENCE L0224] ### Sampling from Marginals
[EVIDENCE L0225] Two approaches tested:
[EVIDENCE L0226] - **Constraint-based**: Sample uniformly at random, with constraint propagation ensuring consistency
[EVIDENCE L0227] - **Probabilistic**: Sample guided by BP marginal distributions, select variables in order of confidence
[EVIDENCE L0228] 
[EVIDENCE L0229] ### Performance
[EVIDENCE L0230] - 10 determinizations per move, 1000 simulations per action
[EVIDENCE L0231] - Tested on Mini-Stratego (5x5, hidden identities) and Goofspiel (13-card bidding)
[EVIDENCE L0232] - Key finding: "Added cost of probabilistic inference may be unjustified when constraint filtering already approximates the state well"
[EVIDENCE L0233] 
[EVIDENCE L0234] ### Direct Mapping to Hydra
[EVIDENCE L0235] | Paper Concept | Hydra Equivalent |
[EVIDENCE L0236] |--------------|-----------------|
[EVIDENCE L0237] | Variables (unknown pieces) | Unknown tiles (opponent hands + wall) |
[EVIDENCE L0238] | GCC (piece count limits) | Exactly 4 of each tile type, minus known |
[EVIDENCE L0239] | BP marginal estimates | **Sinkhorn head output** |
[EVIDENCE L0240] | Constraint-based sampling | Sequential tile allocation with GCC pruning |
[EVIDENCE L0241] | Factor graph | Tile-to-player allocation graph |
[EVIDENCE L0242] 
[EVIDENCE L0243] ---
[EVIDENCE L0244] 
[EVIDENCE L0245] ## 6. MCMC / Gibbs Sampling for History Generation
[EVIDENCE L0246] 
[EVIDENCE L0247] **Paper**: "History Filtering in Imperfect Information Games: Algorithms and Complexity" (2023)
[EVIDENCE L0248] **Link**: https://arxiv.org/abs/2311.14651
[EVIDENCE L0249] 
[EVIDENCE L0250] ### Core Algorithm: Gibbs Sampler with RingSwap
[EVIDENCE L0251] For trick-taking card games (bridge, hearts, Oh Hell), the paper presents an MCMC sampler:
[EVIDENCE L0252] 
[EVIDENCE L0253] 1. Start with any valid deal (found via max-flow in polynomial time)
[EVIDENCE L0254] 2. **RingSwap**: Swap cards between suits/players while maintaining correct row/column sums
[EVIDENCE L0255] 3. Accept/reject via Metropolis-Hastings with reach probability ratio
[EVIDENCE L0256] 4. Repeat to generate diverse samples from the correct distribution
[EVIDENCE L0257] 
[EVIDENCE L0258] ### Theoretical Guarantees
[EVIDENCE L0259] - **Aperiodic and irreducible** (Theorem 3): Can reach any valid history
[EVIDENCE L0260] - **Correct stationary distribution** (Theorem 4): Converges to P^pi (policy-weighted)
[EVIDENCE L0261] - **Polynomial time per transition** with respect to history length
[EVIDENCE L0262] 
[EVIDENCE L0263] ### Performance on Oh Hell
[EVIDENCE L0264] | PBS Size | Histories | Samples Needed | State Transitions |
[EVIDENCE L0265] |----------|-----------|----------------|-------------------|
[EVIDENCE L0266] | Small | 192 | ~100 | ~2,000 |
[EVIDENCE L0267] | Medium | 12,960 | ~200 | ~4,000 |
[EVIDENCE L0268] | Large | 544,320 | **~400** | **~8,000** |
[EVIDENCE L0269] 
[EVIDENCE L0270] - **Burn-in of 20 steps** was sufficient for accurate value estimates
[EVIDENCE L0271] - Significantly outperformed importance sampling for large state spaces
[EVIDENCE L0272] - Memory-efficient: doesn't enumerate all histories, just transitions locally
[EVIDENCE L0273] 
[EVIDENCE L0274] ### Adaptation to Mahjong
[EVIDENCE L0275] The RingSwap concept maps to Mahjong:
[EVIDENCE L0276] - **"Suits" = tile types** (man, pin, sou, honors)
[EVIDENCE L0277] - **"Players" = 3 opponents + wall**
[EVIDENCE L0278] - **Constraints**: Void information (player discarded all of type X => can't hold more)
[EVIDENCE L0279] - **Initial valid deal**: Construct via max-flow on tile-to-player bipartite graph
[EVIDENCE L0280] - **Swap**: Move tile from opponent A to opponent B (or wall), maintaining hand sizes
[EVIDENCE L0281] 
[EVIDENCE L0282] **This is likely the most efficient sampling algorithm for Hydra's use case.**
[EVIDENCE L0283] 
[EVIDENCE L0284] ---
[EVIDENCE L0285] 
[EVIDENCE L0286] ## 7. Sinkhorn Operator for Constrained Sampling
[EVIDENCE L0287] 
[EVIDENCE L0288] ### 7a. Gumbel-Sinkhorn Networks (Mena et al., 2018)
[EVIDENCE L0289] 
[EVIDENCE L0290] **Tutorial**: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html
[EVIDENCE L0291] 
[EVIDENCE L0292] The Sinkhorn operator iteratively normalizes rows and columns of a matrix to produce a **doubly-stochastic matrix** (all rows and columns sum to 1). This is the continuous relaxation of a permutation matrix.
[EVIDENCE L0293] 
[EVIDENCE L0294] ### 7b. Sinkhorn Policy Gradient (Emami & Ranka, 2018)
[EVIDENCE L0295] 
[EVIDENCE L0296] **Paper**: https://arxiv.org/abs/1805.07010
[EVIDENCE L0297] 
[EVIDENCE L0298] Key details:
[EVIDENCE L0299] - Sinkhorn operator S^L(X/tau): L iterations of alternating row/column normalization
[EVIDENCE L0300] - Temperature tau controls sharpness (tau->0 gives hard permutation)
[EVIDENCE L0301] - **10 Sinkhorn iterations** is optimal tradeoff (Section D, p.15)
[EVIDENCE L0302] - Rounding via **Hungarian algorithm O(n^3)** to get discrete permutation
[EVIDENCE L0303] - Exploration via k=2 row exchanges (epsilon-greedy)
[EVIDENCE L0304] 
[EVIDENCE L0305] ### Relevance to Hydra's Sinkhorn Head
[EVIDENCE L0306] Hydra's Sinkhorn tile allocation head outputs a matrix where:
[EVIDENCE L0307] - Rows = tile types (34 types)
[EVIDENCE L0308] - Columns = locations (3 opponents + wall)
[EVIDENCE L0309] - Entries = P(tile_type_i is at location_j)
[EVIDENCE L0310] - **Row sums** = number of copies of each tile type remaining (known exactly)
[EVIDENCE L0311] - **Column sums** = number of tiles each opponent/wall holds (known exactly)
[EVIDENCE L0312] 
[EVIDENCE L0313] The Sinkhorn operator **directly enforces these constraints** during training. The question is how to **sample** from this output.
[EVIDENCE L0314] 
[EVIDENCE L0315] ### Sampling Approaches
[EVIDENCE L0316] 
[EVIDENCE L0317] **Approach A: Hungarian Rounding**
[EVIDENCE L0318] - Round doubly-stochastic matrix to nearest integer allocation via Hungarian algorithm
[EVIDENCE L0319] - Gives one "most likely" allocation, not diverse samples
[EVIDENCE L0320] - O(n^3) per sample
[EVIDENCE L0321] 
[EVIDENCE L0322] **Approach B: Sequential Allocation with Gumbel Noise**
[EVIDENCE L0323] - Add Gumbel noise to log-probabilities, then apply Sinkhorn
[EVIDENCE L0324] - Each noise sample gives a different valid allocation
[EVIDENCE L0325] - Differentiable if needed for training
[EVIDENCE L0326] - Fast: just Sinkhorn iterations + argmax
[EVIDENCE L0327] 
[EVIDENCE L0328] **Approach C: Categorical Sampling with Constraint Repair**
[EVIDENCE L0329] - Sample each tile independently from marginals
[EVIDENCE L0330] - "Repair" violations (too many tiles assigned to one player) via redistribution
[EVIDENCE L0331] - Fast but may distort distribution
[EVIDENCE L0332] 
[EVIDENCE L0333] **Approach D: Gibbs Sampling from Sinkhorn Marginals** (RECOMMENDED)
[EVIDENCE L0334] - Start from any valid allocation
[EVIDENCE L0335] - Use Sinkhorn marginals as proposal distribution
[EVIDENCE L0336] - RingSwap-style moves (swap tile between two locations)
[EVIDENCE L0337] - Accept/reject based on marginal probability ratio
[EVIDENCE L0338] - Guaranteed to converge to correct distribution
[EVIDENCE L0339] - Very fast per-step, good mixing due to informed proposals
[EVIDENCE L0340] 
[EVIDENCE L0341] ---
[EVIDENCE L0342] 
[EVIDENCE L0343] ## 8. Synthesis: Recommended Approach for Hydra
[EVIDENCE L0344] 
[EVIDENCE L0345] ### The Problem Statement
[EVIDENCE L0346] Given:
[EVIDENCE L0347] - Sinkhorn head outputs: M[tile_type][location] = P(tile at location) for all unknown tiles
[EVIDENCE L0348] - Row constraints: sum_j M[i][j] = remaining_count[tile_type_i] (known exactly)
[EVIDENCE L0349] - Column constraints: sum_i M[i][j] = hand_size[player_j] (known exactly)
[EVIDENCE L0350] - Need: 100+ valid samples/second of complete opponent hands
[EVIDENCE L0351] 
[EVIDENCE L0352] ### Recommended: Gibbs Sampler with Sinkhorn-Informed Proposals
[EVIDENCE L0353] 
[EVIDENCE L0354] ```
[EVIDENCE L0355] Algorithm: Sinkhorn-Gibbs Tile Sampler
[EVIDENCE L0356] ========================================
[EVIDENCE L0357] Input: M (Sinkhorn marginals), constraints (tile counts, hand sizes)
[EVIDENCE L0358] Output: Stream of valid tile allocations
[EVIDENCE L0359] 
[EVIDENCE L0360] 1. INITIALIZE: Construct initial valid allocation via greedy sequential
[EVIDENCE L0361]    assignment (assign tiles to locations in order of decreasing marginal
[EVIDENCE L0362]    probability, respecting constraints)
[EVIDENCE L0363] 
[EVIDENCE L0364] 2. SAMPLE LOOP (for each desired sample):
[EVIDENCE L0365]    a. Pick random tile t from unknown tiles
[EVIDENCE L0366]    b. Pick random swap partner: another tile t' at a different location
[EVIDENCE L0367]    c. Propose swap: move t to t's location, t' to t's location
[EVIDENCE L0368]    d. If swap maintains constraints (hand sizes, tile counts):
[EVIDENCE L0369]       - Accept with probability min(1, M[t'][new_loc] * M[t][new_loc] / 
[EVIDENCE L0370]                                         M[t][old_loc] * M[t'][old_loc])
[EVIDENCE L0371]    e. Repeat steps a-d for B burn-in steps (B ~ 20-50)
[EVIDENCE L0372]    f. Record current allocation as a sample
[EVIDENCE L0373] 
[EVIDENCE L0374] 3. Return collected samples
[EVIDENCE L0375] ```
[EVIDENCE L0376] 
[EVIDENCE L0377] ### Why This Approach
[EVIDENCE L0378] 
[EVIDENCE L0379] | Criterion | Score | Reason |
[EVIDENCE L0380] |-----------|-------|--------|
[EVIDENCE L0381] | Speed | Excellent | O(1) per swap step, no matrix operations |
[EVIDENCE L0382] | Correctness | Proven | Gibbs sampler converges to target distribution |
[EVIDENCE L0383] | Constraint satisfaction | Guaranteed | Only valid swaps are proposed |
[EVIDENCE L0384] | Uses Sinkhorn output | Yes | Marginals guide acceptance probability |
[EVIDENCE L0385] | Diversity | Good | MCMC explores space, not just MAP |
[EVIDENCE L0386] | Burn-in | Fast | ~20 steps sufficient (per history filtering paper) |
[EVIDENCE L0387] | Implementation complexity | Low | Just swaps + probability ratios |
[EVIDENCE L0388] 
[EVIDENCE L0389] ### Performance Estimate
[EVIDENCE L0390] - ~50 unknown tiles (typical mid-game Mahjong)
[EVIDENCE L0391] - ~20 swap steps per sample (burn-in)  
[EVIDENCE L0392] - Each swap: 2 random indices + 1 probability ratio + 1 comparison
[EVIDENCE L0393] - **Estimate: 10,000+ samples/second on single CPU core**
[EVIDENCE L0394] - With 100 samples for search: **100+ search evaluations per second**
[EVIDENCE L0395] 
[EVIDENCE L0396] ### Alternative: Vectorized Gumbel-Sinkhorn Sampling
[EVIDENCE L0397] For GPU acceleration during training:
[EVIDENCE L0398] 1. Generate batch of B Gumbel noise matrices
[EVIDENCE L0399] 2. Add to log(M) element-wise
[EVIDENCE L0400] 3. Apply Sinkhorn operator (10 iterations each)
[EVIDENCE L0401] 4. Round to integer allocation via argmax per tile
[EVIDENCE L0402] 5. Entire batch computed in parallel on GPU
[EVIDENCE L0403] 
[EVIDENCE L0404] This gives B valid samples in one forward pass. Good for training-time sampling.
[EVIDENCE L0405] 
[EVIDENCE L0406] ### Comparison to Existing Approaches
[EVIDENCE L0407] 
[EVIDENCE L0408] | Approach | Speed | Accuracy | Complexity | Used By |
[EVIDENCE L0409] |----------|-------|----------|------------|---------|
[EVIDENCE L0410] | Uniform sampling + rejection | Poor | Poor | Low | Naive baseline |
[EVIDENCE L0411] | Bayesian weighting (GIB) | Good | Good | Medium | GIB (50-500 samples) |
[EVIDENCE L0412] | Full CFR on info sets | N/A | Best | Very High | Pluribus/Libratus |
[EVIDENCE L0413] | pMCPA random sampling (Suphx) | Good | Moderate | Low | Suphx |
[EVIDENCE L0414] | CSP + BP (constraint paper) | Good | Good | Medium | Mini-Stratego |
[EVIDENCE L0415] | **Gibbs + Sinkhorn marginals** | **Best** | **Good** | **Low** | **Proposed for Hydra** |
[EVIDENCE L0416] | ISMCTS (per-iteration sampling) | Good | Good | Medium | Card games |
[EVIDENCE L0417] 
[EVIDENCE L0418] ### Implementation Priorities for Hydra
[EVIDENCE L0419] 
[EVIDENCE L0420] 1. **Phase 1 (Now)**: Implement Sinkhorn head to get marginals (already planned)
[EVIDENCE L0421] 2. **Phase 2**: Implement Gibbs sampler for CPU-side sampling during search
[EVIDENCE L0422] 3. **Phase 3**: Implement Gumbel-Sinkhorn batch sampling for GPU training
[EVIDENCE L0423] 4. **Phase 4**: Integrate with ISMCTS-style search (or Pluribus-style depth-limited search)
[EVIDENCE L0424] 
[EVIDENCE L0425] ---
[EVIDENCE L0426] 
[EVIDENCE L0427] ## Paper Reference Table
[EVIDENCE L0428] 
[EVIDENCE L0429] | # | Paper | Year | Topic | Key Contribution | Link |
[EVIDENCE L0430] |---|-------|------|-------|-----------------|------|
[EVIDENCE L0431] | 1 | Ginsberg, "GIB" | 2001 | Bridge AI | Monte Carlo determinization, achievable sets, Bayesian weighting. 50-500 samples. | [JAIR](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume14/ginsberg01a.pdf) |
[EVIDENCE L0432] | 2 | Brown & Sandholm, "Pluribus" | 2019 | 6-player Poker | Blueprint + depth-limited search, k=4 continuation strategies, reach probabilities | [Science](https://www.science.org/doi/10.1126/science.aay2400) |
[EVIDENCE L0433] | 3 | Brown & Sandholm, "Libratus" | 2018 | Heads-up Poker | Nested safe subgame solving, blueprint + self-improver | [Science](https://www.science.org/doi/10.1126/science.aao1733) |
[EVIDENCE L0434] | 4 | Moravcik et al., "DeepStack" | 2017 | Heads-up Poker | Continual resolving, neural leaf evaluation, no pre-computed blueprint | [Science](https://doi.org/10.1126/science.aam6960) |
[EVIDENCE L0435] | 5 | Li et al., "Suphx" | 2020 | Riichi Mahjong | Oracle guiding, pMCPA run-time adaptation, random tile sampling | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) |
[EVIDENCE L0436] | 6 | Cowling et al., "ISMCTS" | 2012 | Card Games | Info set MCTS, avoids strategy fusion, 3 variants | [IEEE](https://ieeexplore.ieee.org/document/6203567) |
[EVIDENCE L0437] | 7 | (Anonymous), "Constraint-Based Belief States" | 2025 | Stratego/Goofspiel | CSP + GCC + Belief Propagation for marginals, constrained sampling | [arXiv:2507.19263](https://arxiv.org/abs/2507.19263) |
[EVIDENCE L0438] | 8 | (Authors), "History Filtering" | 2023 | Trick-taking Games | MCMC Gibbs sampler, RingSwap, 400 samples for 544K histories, 20-step burn-in | [arXiv:2311.14651](https://arxiv.org/abs/2311.14651) |
[EVIDENCE L0439] | 9 | Emami & Ranka, "Sinkhorn Policy Gradient" | 2018 | Combinatorial Opt | Sinkhorn layer, Hungarian rounding, epsilon-greedy exploration | [arXiv:1805.07010](https://arxiv.org/abs/1805.07010) |
[EVIDENCE L0440] | 10 | Billings et al., "Selective Sampling in Poker" | 1999 | Poker (Loki) | Weighted sampling from opponent hand distributions, 500 trials | [UAlberta](https://poker.cs.ualberta.ca/publications/AAAISS99.pdf) |
[EVIDENCE L0441] 
[EVIDENCE L0442] ---
[EVIDENCE L0443] 
[EVIDENCE L0444] ## Glossary
[EVIDENCE L0445] 
[EVIDENCE L0446] - **Determinization**: Sampling a specific hidden state and solving as perfect-information
[EVIDENCE L0447] - **Strategy Fusion**: Bug where determinization assumes different decisions per world
[EVIDENCE L0448] - **Information Set**: Set of game states indistinguishable to a player
[EVIDENCE L0449] - **CFR**: Counterfactual Regret Minimization -- iterative algorithm converging to Nash equilibrium
[EVIDENCE L0450] - **GCC**: Global Cardinality Constraint -- ensures count of each value stays within limits
[EVIDENCE L0451] - **BP**: Belief Propagation -- message passing on factor graphs to estimate marginals
[EVIDENCE L0452] - **Sinkhorn Operator**: Iterative row/column normalization producing doubly-stochastic matrix
[EVIDENCE L0453] - **RingSwap**: Moving cards/tiles between players while maintaining count constraints
[EVIDENCE L0454] - **pMCPA**: Parametric Monte-Carlo Policy Adaptation (Suphx's run-time search)
[EVIDENCE L0455] - **ISMCTS**: Information Set Monte Carlo Tree Search
```

## Artifact 14 — Incremental belief and POMDP survey
Artifact id: `incremental-belief-survey`
Source label: EVIDENCE
Type: `file_full`
Source: `research/evidence/incremental-belief-survey.md`
Why it matters: Large evidence surface covering DreamerV3, POMDP belief methods, and partial-observability handling. Useful as adjacent context when deciding whether DCRL deserves more attention than the other nearby POMDP / oracle / belief lines Hydra already tracks.

```markdown
[EVIDENCE L0001] # Prior Art Survey: Incremental Belief Networks for Imperfect-Information Games
[EVIDENCE L0002] 
[EVIDENCE L0003] **Date**: 2026-03-03
[EVIDENCE L0004] **Context**: Hydra Mahjong AI -- exploring NNUE-inspired incremental belief tracking
[EVIDENCE L0005] 
[EVIDENCE L0006] ---
[EVIDENCE L0007] 
[EVIDENCE L0008] ## Executive Summary
[EVIDENCE L0009] 
[EVIDENCE L0010] **Your idea of NNUE-style incrementally-updatable belief networks for imperfect-information games appears genuinely novel.** No prior work directly combines NNUE's accumulator-based incremental feature updates with belief state tracking for hidden-information games. The closest related work falls into 7 distinct research threads documented below.
[EVIDENCE L0011] 
[EVIDENCE L0012] ---
[EVIDENCE L0013] 
[EVIDENCE L0014] ## 1. NNUE: The Incremental Update Foundation (Perfect Information Only)
[EVIDENCE L0015] 
[EVIDENCE L0016] **Key insight**: NNUE achieves 10-15x speedup by incrementally updating only changed features.
[EVIDENCE L0017] 
[EVIDENCE L0018] ### Architecture
[EVIDENCE L0019] - Overparameterized input layer (e.g., 40,960 inputs for HalfKP in chess)
[EVIDENCE L0020] - Only ~30 features active at any position
[EVIDENCE L0021] - Accumulator stores first-layer output as persistent state
[EVIDENCE L0022] 
[EVIDENCE L0023] ### Core Equations
[EVIDENCE L0024] 
[EVIDENCE L0025] **Full refresh** (from scratch):
[EVIDENCE L0026] ```
[EVIDENCE L0027] accumulator = bias + SUM(weight_column[i] for i in active_features)
[EVIDENCE L0028] ```
[EVIDENCE L0029] 
[EVIDENCE L0030] **Incremental update** (on move, removing set R, adding set S):
[EVIDENCE L0031] ```
[EVIDENCE L0032] accumulator_new = accumulator_old - SUM(W[:, r] for r in R) + SUM(W[:, a] for a in S)
[EVIDENCE L0033] ```
[EVIDENCE L0034] 
[EVIDENCE L0035] Between consecutive positions, typically only 2-4 features change, so this is O(changed_features * hidden_dim) instead of O(total_features * hidden_dim).
[EVIDENCE L0036] 
[EVIDENCE L0037] **Sources**:
[EVIDENCE L0038] - [Stockfish NNUE docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
[EVIDENCE L0039] - [NNUE Architecture Reference (DeepWiki)](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference)
[EVIDENCE L0040] - [HalfKP/NNUE GitHub](https://github.com/HalfKP/NNUE)
[EVIDENCE L0041] - [NNUE Wikipedia](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
[EVIDENCE L0042] 
[EVIDENCE L0043] **Gap**: NNUE has NEVER been applied to imperfect-information games. It operates on fully-observable board states only (chess, shogi). Your proposed extension to belief states is novel.
[EVIDENCE L0044] 
[EVIDENCE L0045] ---
[EVIDENCE L0046] 
[EVIDENCE L0047] ## 2. ReBeL: Recursive Belief-Based Learning (Meta AI, 2020)
[EVIDENCE L0048] 
[EVIDENCE L0049] **The foundational paper for neural belief-state game AI.**
[EVIDENCE L0050] 
[EVIDENCE L0051] ### Paper
[EVIDENCE L0052] Brown, Noam et al. "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games." NeurIPS 2020.
[EVIDENCE L0053] - [arXiv](https://arxiv.org/abs/2007.13544)
[EVIDENCE L0054] - [Meta AI Blog](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/)
[EVIDENCE L0055] - [GitHub (Liar's Dice)](https://github.com/facebookresearch/rebel)
[EVIDENCE L0056] 
[EVIDENCE L0057] ### Key Concept: Public Belief State (PBS)
[EVIDENCE L0058] 
[EVIDENCE L0059] A PBS is a probability distribution over hidden information sets, conditioned on public history:
[EVIDENCE L0060] 
[EVIDENCE L0061] ```
[EVIDENCE L0062] PBS_t = b_t = Pr(hidden_infosets | public_history, common_knowledge_policy)
[EVIDENCE L0063] ```
[EVIDENCE L0064] 
[EVIDENCE L0065] This transforms an imperfect-information game into a continuous-state perfect-information game over beliefs.
[EVIDENCE L0066] 
[EVIDENCE L0067] ### Belief Update (Bayes Rule)
[EVIDENCE L0068] 
[EVIDENCE L0069] Given weights w(s) over hidden states s, and observed public action a:
[EVIDENCE L0070] 
[EVIDENCE L0071] ```
[EVIDENCE L0072] P(a) = SUM_s P(a|s) * w(s) / SUM_s w(s)
[EVIDENCE L0073] 
[EVIDENCE L0074] w'(s) = w(s) * P(a|s) / SUM_s' w(s') * P(a|s')
[EVIDENCE L0075] ```
[EVIDENCE L0076] 
[EVIDENCE L0077] This is standard Bayesian filtering, but the key innovation is that P(a|s) comes from the CURRENT policy (which changes during training), making this a moving target.
[EVIDENCE L0078] 
[EVIDENCE L0079] ### Search: CFR in Belief Space
[EVIDENCE L0080] 
[EVIDENCE L0081] - AlphaZero-style MCTS is intractable in belief space (actions become continuous distributions)
[EVIDENCE L0082] - ReBeL uses Counterfactual Regret Minimization (CFR) as search in depth-limited subgames
[EVIDENCE L0083] - At each decision: build subgame at current PBS, run K iterations of CFR, sample action, update PBS
[EVIDENCE L0084] 
[EVIDENCE L0085] ### Value Network
[EVIDENCE L0086] 
[EVIDENCE L0087] - Trained on terminal game values of self-play trajectories
[EVIDENCE L0088] - Input: PBS (belief distribution)
[EVIDENCE L0089] - Output: expected value for each player
[EVIDENCE L0090] 
[EVIDENCE L0091] ### Convergence
[EVIDENCE L0092] 
[EVIDENCE L0093] Provably converges to epsilon-Nash equilibrium in two-player zero-sum games.
[EVIDENCE L0094] 
[EVIDENCE L0095] **Relevance to your idea**: ReBeL's Bayesian belief update is the "what" to update, but it's computed externally (not learned). An NNUE-style network could LEARN the belief update as an incremental weight adjustment.
[EVIDENCE L0096] 
[EVIDENCE L0097] ---
[EVIDENCE L0098] 
[EVIDENCE L0099] ## 3. Student of Games (DeepMind/Amii, 2023)
[EVIDENCE L0100] 
[EVIDENCE L0101] **Unifies perfect and imperfect-information game AI.**
[EVIDENCE L0102] 
[EVIDENCE L0103] ### Paper
[EVIDENCE L0104] Schmid et al. "Student of Games: A unified learning algorithm for both perfect and imperfect information games." Science Advances, 2023.
[EVIDENCE L0105] - [Science](https://www.science.org/doi/10.1126/sciadv.adg3256)
[EVIDENCE L0106] - [arXiv](https://arxiv.org/abs/2112.03178)
[EVIDENCE L0107] 
[EVIDENCE L0108] ### Key Innovation: GT-CFR + CVPN
[EVIDENCE L0109] 
[EVIDENCE L0110] - **Growing-Tree CFR (GT-CFR)**: Incrementally grows the search tree (instead of fixed-depth subgames like ReBeL)
[EVIDENCE L0111] - **Counterfactual Value-and-Policy Network (CVPN)**: Neural network that takes a PBS as input and outputs both counterfactual values AND action policies for each information state
[EVIDENCE L0112] - Self-play generates two types of training data:
[EVIDENCE L0113]   1. Search queries (PBS nodes queried during GT-CFR regret updates)
[EVIDENCE L0114]   2. Full-game trajectories
[EVIDENCE L0115] 
[EVIDENCE L0116] ### Architecture
[EVIDENCE L0117] - CVPN: belief -> (values_per_infostate, policy_per_infostate)
[EVIDENCE L0118] - Sound for both perfect-info (chess, Go) and imperfect-info (poker, Scotland Yard)
[EVIDENCE L0119] 
[EVIDENCE L0120] **Relevance**: SoG's CVPN is the closest existing architecture to what you're proposing -- it processes belief states through a neural network. But it does NOT use incremental updates; it recomputes from scratch each time.
[EVIDENCE L0121] 
[EVIDENCE L0122] ---
[EVIDENCE L0123] 
[EVIDENCE L0124] ## 4. DreamerV3 RSSM: World Models for Partial Observability
[EVIDENCE L0125] 
[EVIDENCE L0126] **The state-of-the-art for learned latent dynamics under partial observability.**
[EVIDENCE L0127] 
[EVIDENCE L0128] ### Paper
[EVIDENCE L0129] Hafner et al. "Mastering Diverse Domains through World Models." Nature, 2025.
[EVIDENCE L0130] - [Nature](https://www.nature.com/articles/s41586-025-08744-2)
[EVIDENCE L0131] - [DreamerV3 RSSM (DeepWiki)](https://deepwiki.com/danijar/dreamerv3/4.1-world-model-(rssm))
[EVIDENCE L0132] 
[EVIDENCE L0133] ### Recurrent State-Space Model (RSSM)
[EVIDENCE L0134] 
[EVIDENCE L0135] The latent state is hybrid: s_t = (d_t, z_t) where:
[EVIDENCE L0136] - d_t in R^8192: deterministic recurrent component (GRU-like)
[EVIDENCE L0137] - z_t in {0,1}^(32x64): stochastic discrete component (categorical)
[EVIDENCE L0138] 
[EVIDENCE L0139] **Observe mode** (with real observations):
[EVIDENCE L0140] ```
[EVIDENCE L0141] x_t = Encoder(o_t)
[EVIDENCE L0142] (d_t, z_t), carry_t = RSSM.observe(carry_{t-1}, x_t, a_{t-1}, reset_t)
[EVIDENCE L0143] ```
[EVIDENCE L0144] 
[EVIDENCE L0145] **Imagine mode** (no observations, planning):
[EVIDENCE L0146] ```
[EVIDENCE L0147] a_t ~ pi(concat(d_t, flatten(z_t)))
[EVIDENCE L0148] (d_{t+1}, z_{t+1}), carry_{t+1} = RSSM.imagine((d_t, z_t), a_t)
[EVIDENCE L0149] ```
[EVIDENCE L0150] 
[EVIDENCE L0151] **Feature vector for all heads**:
[EVIDENCE L0152] ```
[EVIDENCE L0153] f_t = concat(d_t, flatten(z_t)) in R^10240
[EVIDENCE L0154] ```
[EVIDENCE L0155] 
[EVIDENCE L0156] ### How It Handles Partial Observability
[EVIDENCE L0157] 
[EVIDENCE L0158] - The deterministic recurrent state d_t summarizes ALL past observations/actions
[EVIDENCE L0159] - This IS a learned belief state -- it's a sufficient statistic of history
[EVIDENCE L0160] - The stochastic z_t captures remaining uncertainty
[EVIDENCE L0161] - Training uses KL divergence between prior (dynamics-predicted) and posterior (observation-informed) z_t
[EVIDENCE L0162] 
[EVIDENCE L0163] **Relevance**: The RSSM is probably the closest existing architecture to an "incremental belief update network." Each step updates d_t incrementally through a GRU. However:
[EVIDENCE L0164] 1. It's designed for single-agent POMDPs, not multi-agent games
[EVIDENCE L0165] 2. The update is implicit (learned GRU weights), not sparse/efficient like NNUE
[EVIDENCE L0166] 3. It doesn't exploit the structure of card games (sparse, discrete changes)
[EVIDENCE L0167] 
[EVIDENCE L0168] ---
[EVIDENCE L0169] 
[EVIDENCE L0170] ## 5. Deep CFR and DREAM: Neural CFR for Large Games
[EVIDENCE L0171] 
[EVIDENCE L0172] ### Deep Counterfactual Regret Minimization
[EVIDENCE L0173] Brown et al. "Deep Counterfactual Regret Minimization." ICML 2019.
[EVIDENCE L0174] - [arXiv](https://arxiv.org/abs/1811.00164)
[EVIDENCE L0175] - [Meta AI](https://ai.meta.com/research/publications/deep-counterfactual-regret-minimization/)
[EVIDENCE L0176] 
[EVIDENCE L0177] Replaces tabular CFR with neural networks that approximate cumulative regrets.
[EVIDENCE L0178] - At each CFR iteration, a neural network V_theta predicts counterfactual values
[EVIDENCE L0179] - Advantage network stores regrets as training targets in a reservoir buffer
[EVIDENCE L0180] - Strategy network averages over time
[EVIDENCE L0181] 
[EVIDENCE L0182] ### DREAM (Deep Regret minimization with Advantage baselines and Model-free learning)
[EVIDENCE L0183] - [GitHub](https://github.com/EricSteinberger/DREAM)
[EVIDENCE L0184] - Scalable implementation of Deep CFR variants
[EVIDENCE L0185] - Includes SD-CFR, Deep CFR, and NFSP implementations in the PokerRL framework
[EVIDENCE L0186] 
[EVIDENCE L0187] **Relevance**: Deep CFR's advantage network could potentially benefit from NNUE-style incremental updates -- regrets change incrementally between CFR iterations.
[EVIDENCE L0188] 
[EVIDENCE L0189] ---
[EVIDENCE L0190] 
[EVIDENCE L0191] ## 6. Efficient Incremental Belief Updates (Non-Game, Bayesian)
[EVIDENCE L0192] 
[EVIDENCE L0193] ### Paper
[EVIDENCE L0194] "Efficient Incremental Belief Updates" (arXiv 2402.06940, 2024)
[EVIDENCE L0195] - [arXiv](https://arxiv.org/html/2402.06940)
[EVIDENCE L0196] 
[EVIDENCE L0197] ### Core Idea: Weighted Virtual Observations
[EVIDENCE L0198] 
[EVIDENCE L0199] Compress past posterior into a small set of weighted "virtual observations" that approximately preserve the posterior:
[EVIDENCE L0200] 
[EVIDENCE L0201] ```
[EVIDENCE L0202] min_w KL(p(x|y*) || p(x|y_hat, w))
[EVIDENCE L0203] ```
[EVIDENCE L0204] 
[EVIDENCE L0205] Where y* is original data, y_hat are virtual observations, w are learned weights.
[EVIDENCE L0206] 
[EVIDENCE L0207] **Weighted virtual observation likelihood**:
[EVIDENCE L0208] ```
[EVIDENCE L0209] p(w|x) = exp(log h(w) + SUM_i w_i * log p(y_hat_i | x))
[EVIDENCE L0210] ```
[EVIDENCE L0211] 
[EVIDENCE L0212] **Incremental update**: Instead of re-running inference on ALL historical observations, condition on:
[EVIDENCE L0213] ```
[EVIDENCE L0214] new_posterior = inference(compressed_past_belief + new_observations)
[EVIDENCE L0215] ```
[EVIDENCE L0216] 
[EVIDENCE L0217] This is the most mathematically rigorous incremental belief update framework found, but it's for general probabilistic programming, not game AI.
[EVIDENCE L0218] 
[EVIDENCE L0219] **Relevance**: Direct mathematical foundation for your approach. Could be combined with NNUE-style accumulator: the "accumulator" stores a compressed belief state, and new observations trigger incremental weight adjustments.
[EVIDENCE L0220] 
[EVIDENCE L0221] ---
[EVIDENCE L0222] 
[EVIDENCE L0223] ## 7. BetaZero: Belief-State Planning for POMDPs
[EVIDENCE L0224] 
[EVIDENCE L0225] ### Paper
[EVIDENCE L0226] Moss et al. "BetaZero: Belief-State Planning for Long-Horizon POMDPs using Learned Approximations." RLC 2024.
[EVIDENCE L0227] - [arXiv](https://arxiv.org/abs/2306.00249)
[EVIDENCE L0228] - [GitHub](https://github.com/sisl/BetaZero.jl)
[EVIDENCE L0229] 
[EVIDENCE L0230] ### Architecture
[EVIDENCE L0231] - AlphaZero-style self-play + MCTS, but over BELIEF states instead of world states
[EVIDENCE L0232] - Neural network approximates value and policy given a BELIEF REPRESENTATION
[EVIDENCE L0233] - User defines `input_representation(belief)` -- e.g., mean and std of particle filter
[EVIDENCE L0234] - Uses PUCT search in belief space
[EVIDENCE L0235] 
[EVIDENCE L0236] ### Key Design
[EVIDENCE L0237] ```julia
[EVIDENCE L0238] function BetaZero.input_representation(b::ParticleCollection)
[EVIDENCE L0239]     mu, sigma = mean_and_std(s.y for s in particles(b))
[EVIDENCE L0240]     return Float32[mu, sigma]
[EVIDENCE L0241] end
[EVIDENCE L0242] ```
[EVIDENCE L0243] 
[EVIDENCE L0244] The belief is compressed into sufficient statistics (mean, variance) before being fed to the neural network.
[EVIDENCE L0245] 
[EVIDENCE L0246] **Relevance**: BetaZero shows that MCTS can work over belief states if you have good sufficient statistics. Your NNUE-style network could LEARN these sufficient statistics incrementally.
[EVIDENCE L0247] 
[EVIDENCE L0248] ---
[EVIDENCE L0249] 
[EVIDENCE L0250] ## 8. Deep Belief Markov Models (DBMM) for POMDP Inference
[EVIDENCE L0251] 
[EVIDENCE L0252] ### Paper
[EVIDENCE L0253] "Deep Belief Markov Models for POMDP Inference" (arXiv 2503.13438, 2025)
[EVIDENCE L0254] - [arXiv](https://arxiv.org/abs/2503.13438)
[EVIDENCE L0255] - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025012675)
[EVIDENCE L0256] 
[EVIDENCE L0257] ### Key Claims
[EVIDENCE L0258] - Learns network belief representations that converge to ground-truth belief representations in discrete POMDPs
[EVIDENCE L0259] - Provides better information than just observations + EnKF in continuous POMDPs
[EVIDENCE L0260] - Model-formulation agnostic (works without knowing the POMDP dynamics)
[EVIDENCE L0261] 
[EVIDENCE L0262] **Relevance**: If DBMM can learn belief representations that converge to true beliefs, it validates the idea that neural networks CAN learn compact belief states. Your NNUE extension would add the incremental update efficiency on top.
[EVIDENCE L0263] 
[EVIDENCE L0264] ---
[EVIDENCE L0265] 
[EVIDENCE L0266] ## 9. Mahjong-Specific: Mortal and Suphx
[EVIDENCE L0267] 
[EVIDENCE L0268] ### Mortal (Equim-chan, open-source)
[EVIDENCE L0269] - [GitHub](https://github.com/Equim-chan/Mortal)
[EVIDENCE L0270] - ResNet-based feature extractor ("Brain") + DQN for action selection
[EVIDENCE L0271] - 46 discrete actions (Hydra-compatible action space)
[EVIDENCE L0272] - Brain outputs: latent distribution (mu, log_sigma) in v1, direct features (phi) in v2-v4
[EVIDENCE L0273] - Has `is_oracle` mode for training with vs. without hidden information
[EVIDENCE L0274] - **Does NOT use incremental updates** -- full forward pass every decision
[EVIDENCE L0275] 
[EVIDENCE L0276] ### Suphx (Microsoft, 2020)
[EVIDENCE L0277] - [arXiv](https://arxiv.org/abs/2003.13590)
[EVIDENCE L0278] - Global reward prediction for long-horizon credit assignment
[EVIDENCE L0279] - Oracle guiding: train with perfect info, then transfer to imperfect info
[EVIDENCE L0280] - Run-time policy adaptation (parametric Monte Carlo)
[EVIDENCE L0281] - **Does NOT use belief tracking** in the NNUE sense -- uses observation history directly
[EVIDENCE L0282] 
[EVIDENCE L0283] ---
[EVIDENCE L0284] 
[EVIDENCE L0285] ## 10. Cicero/Diplomacy: Multiplayer Imperfect-Info with Language
[EVIDENCE L0286] 
[EVIDENCE L0287] ### Paper
[EVIDENCE L0288] Meta FAIR. "Human-level play in the game of Diplomacy." Science, 2022.
[EVIDENCE L0289] - [Science](https://www.science.org/doi/10.1126/science.ade9097)
[EVIDENCE L0290] - [GitHub](https://github.com/facebookresearch/diplomacy_cicero)
[EVIDENCE L0291] 
[EVIDENCE L0292] ### Belief Tracking
[EVIDENCE L0293] - Uses bilateral search (bqre1p_agent.py) for opponent modeling
[EVIDENCE L0294] - piKL (policy-anchored KL): regularizes search policy toward a learned "anchor" policy
[EVIDENCE L0295] - Searches over opponent action distributions, not explicit belief states
[EVIDENCE L0296] - Combines language model predictions with strategic planning
[EVIDENCE L0297] 
[EVIDENCE L0298] **Relevance**: Cicero shows that in multiplayer games, you need opponent modeling beyond just Bayes updates. Mahjong (4-player) faces similar challenges.
[EVIDENCE L0299] 
[EVIDENCE L0300] ---
[EVIDENCE L0301] 
[EVIDENCE L0302] ## 11. EfficientZero V2 and Model-Based RL
[EVIDENCE L0303] 
[EVIDENCE L0304] ### EfficientZero V2
[EVIDENCE L0305] - [arXiv](https://arxiv.org/abs/2403.00564)
[EVIDENCE L0306] - General framework for sample-efficient model-based RL
[EVIDENCE L0307] - Handles discrete/continuous actions, visual/low-dimensional inputs
[EVIDENCE L0308] - Still fundamentally MDP-focused (not designed for hidden information games)
[EVIDENCE L0309] 
[EVIDENCE L0310] ### Model-Based RL for Imperfect Info (Earlier Work)
[EVIDENCE L0311] - [IEEE 2014](https://ieeexplore.ieee.org/document/6797023): Model-based RL for Hearts (card game)
[EVIDENCE L0312] - POMDP formulation with learned transition model
[EVIDENCE L0313] - Limited to single-agent approximation of multi-agent game
[EVIDENCE L0314] 
[EVIDENCE L0315] ---
[EVIDENCE L0316] 
[EVIDENCE L0317] ## Synthesis: The Novelty Gap
[EVIDENCE L0318] 
[EVIDENCE L0319] | Approach | Incremental Update | Belief Tracking | Game AI | Neural |
[EVIDENCE L0320] |---|---|---|---|---|
[EVIDENCE L0321] | NNUE (Stockfish) | YES | NO | YES (perfect info) | YES |
[EVIDENCE L0322] | ReBeL | NO | YES (Bayesian) | YES (imperfect info) | YES (value net) |
[EVIDENCE L0323] | Student of Games | NO | YES (CVPN) | YES (both) | YES |
[EVIDENCE L0324] | DreamerV3 RSSM | PARTIAL (GRU) | YES (implicit) | NO (single agent) | YES |
[EVIDENCE L0325] | BetaZero | NO | YES (particle) | YES (POMDP) | YES |
[EVIDENCE L0326] | Incremental Belief Updates | YES | YES | NO | NO |
[EVIDENCE L0327] | Deep CFR | NO | PARTIAL | YES | YES |
[EVIDENCE L0328] | **Your Proposal** | **YES** | **YES** | **YES** | **YES** |
[EVIDENCE L0329] 
[EVIDENCE L0330] **No existing system combines all four properties.** Your NNUE-inspired incremental belief network for mahjong would be the first to:
[EVIDENCE L0331] 1. Use sparse, incrementally-updatable feature representations (like NNUE)
[EVIDENCE L0332] 2. Track belief states over hidden information (like ReBeL/BetaZero)
[EVIDENCE L0333] 3. Target imperfect-information game AI (like ReBeL/SoG/Deep CFR)
[EVIDENCE L0334] 4. Learn the belief update function neurally (like DreamerV3's RSSM)
[EVIDENCE L0335] 
[EVIDENCE L0336] ---
[EVIDENCE L0337] 
[EVIDENCE L0338] ## Proposed Architecture Sketch (Based on Prior Art)
[EVIDENCE L0339] 
[EVIDENCE L0340] Drawing from the above, a hypothetical "Incremental Belief NNUE" for mahjong could work like:
[EVIDENCE L0341] 
[EVIDENCE L0342] ### Accumulator = Belief State
[EVIDENCE L0343] 
[EVIDENCE L0344] ```
[EVIDENCE L0345] belief_accumulator in R^d    (analogous to NNUE's accumulator)
[EVIDENCE L0346] ```
[EVIDENCE L0347] 
[EVIDENCE L0348] ### On each game event (discard, draw, call, reveal):
[EVIDENCE L0349] 
[EVIDENCE L0350] ```
[EVIDENCE L0351] # Identify changed features (like NNUE)
[EVIDENCE L0352] removed_features = features_invalidated_by_event
[EVIDENCE L0353] added_features = features_activated_by_event
[EVIDENCE L0354] 
[EVIDENCE L0355] # Incremental belief update (NNUE-style)
[EVIDENCE L0356] belief_accumulator -= SUM(W_belief[:, r] for r in removed_features)
[EVIDENCE L0357] belief_accumulator += SUM(W_belief[:, a] for a in added_features)
[EVIDENCE L0358] 
[EVIDENCE L0359] # Bayesian correction (ReBeL-style, but learned)
[EVIDENCE L0360] belief_accumulator = belief_accumulator + delta_bayes(event, belief_accumulator)
[EVIDENCE L0361] ```
[EVIDENCE L0362] 
[EVIDENCE L0363] ### Why this could work for Mahjong specifically:
[EVIDENCE L0364] 
[EVIDENCE L0365] 1. **Sparse changes**: Each event changes very few tiles (1 discard = 1 tile revealed out of ~136)
[EVIDENCE L0366] 2. **Structured hidden info**: Unknown tiles form a finite, trackable set
[EVIDENCE L0367] 3. **Incremental Bayes**: When player X discards tile Y, your belief about their hand updates sparsely
[EVIDENCE L0368] 4. **Speed**: Mahjong AI needs fast inference for real-time play; incremental updates avoid redundant computation
[EVIDENCE L0369] 
[EVIDENCE L0370] ### Open Questions:
[EVIDENCE L0371] 
[EVIDENCE L0372] 1. Can the Bayesian correction term delta_bayes be learned end-to-end?
[EVIDENCE L0373] 2. How to handle "king moves" (equivalent: large state changes like a kan declaration)?
[EVIDENCE L0374] 3. Training: self-play with incremental updates vs. full-recompute baseline?
[EVIDENCE L0375] 4. Does the accumulator maintain enough information for multi-step lookahead?
[EVIDENCE L0376] 
[EVIDENCE L0377] ---
[EVIDENCE L0378] 
[EVIDENCE L0379] ## Key References (Ranked by Relevance)
[EVIDENCE L0380] 
[EVIDENCE L0381] 1. **ReBeL** - Brown et al., NeurIPS 2020 - [arXiv:2007.13544](https://arxiv.org/abs/2007.13544)
[EVIDENCE L0382] 2. **NNUE** - Stockfish - [Docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
[EVIDENCE L0383] 3. **Student of Games** - Schmid et al., Science Advances 2023 - [DOI](https://www.science.org/doi/10.1126/sciadv.adg3256)
[EVIDENCE L0384] 4. **DreamerV3** - Hafner et al., Nature 2025 - [DOI](https://www.nature.com/articles/s41586-025-08744-2)
[EVIDENCE L0385] 5. **BetaZero** - Moss et al., RLC 2024 - [arXiv:2306.00249](https://arxiv.org/abs/2306.00249)
[EVIDENCE L0386] 6. **Efficient Incremental Belief Updates** - arXiv 2024 - [arXiv:2402.06940](https://arxiv.org/abs/2402.06940)
[EVIDENCE L0387] 7. **Deep CFR** - Brown et al., ICML 2019 - [arXiv:1811.00164](https://arxiv.org/abs/1811.00164)
[EVIDENCE L0388] 8. **DBMM** - arXiv 2025 - [arXiv:2503.13438](https://arxiv.org/abs/2503.13438)
[EVIDENCE L0389] 9. **Mortal** - Equim-chan - [GitHub](https://github.com/Equim-chan/Mortal)
[EVIDENCE L0390] 10. **Suphx** - Li et al., 2020 - [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
[EVIDENCE L0391] 11. **Cicero** - Meta FAIR, Science 2022 - [DOI](https://www.science.org/doi/10.1126/science.ade9097)
[EVIDENCE L0392] 12. **DREAM** - Steinberger et al. - [GitHub](https://github.com/EricSteinberger/DREAM)
```

## Artifact 15 — Broad game-AI paradigm scan
Artifact id: `paradigm-breaking-game-ai`
Source label: EVIDENCE
Type: `file_full`
Source: `research/evidence/paradigm_breaking_game_ai_research.md`
Why it matters: Dense adjacent-field artifact covering world models, broader game-AI methods, and cross-field possibility space. Gives the research agent enough surrounding context to judge whether DCRL is actually a standout omission or just one neighboring ingredient family.

```markdown
[EVIDENCE L0001] # Paradigm-Breaking Game AI Approaches: Research for Hydra Mahjong AI
[EVIDENCE L0002] 
[EVIDENCE L0003] > Research compiled from primary sources, papers, and official documentation.
[EVIDENCE L0004] > Focus: Technical details that could inspire novel Mahjong AI design.
[EVIDENCE L0005] 
[EVIDENCE L0006] ---
[EVIDENCE L0007] 
[EVIDENCE L0008] ## Table of Contents
[EVIDENCE L0009] 
[EVIDENCE L0010] 1. [NNUE: Efficiently Updatable Neural Networks](#1-nnue-efficiently-updatable-neural-networks)
[EVIDENCE L0011] 2. [Novel NN + Classical Algorithm Hybrids](#2-novel-nn-classical-algorithm-hybrids)
[EVIDENCE L0012] 3. [Poker AI: Libratus & Pluribus](#3-poker-ai-beyond-standard-cfr)
[EVIDENCE L0013] 4. [Recent Imperfect Information Game AI (2020-2025)](#4-recent-imperfect-information-game-ai-2020-2025)
[EVIDENCE L0014] 5. [Cross-Domain Approaches](#5-cross-domain-approaches)
[EVIDENCE L0015] 6. [Synthesis: Ideas for Mahjong AI](#6-synthesis-paradigm-breaking-ideas-for-hydra-mahjong-ai)
[EVIDENCE L0016] 
[EVIDENCE L0017] ---
[EVIDENCE L0018] 
[EVIDENCE L0019] ## 1. NNUE: Efficiently Updatable Neural Networks
[EVIDENCE L0020] 
[EVIDENCE L0021] **Source**: [Chessprogramming Wiki](https://www.chessprogramming.org/NNUE) | [Stockfish NNUE PyTorch](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference)
[EVIDENCE L0022] 
[EVIDENCE L0023] ### 1.1 The Paradigm Break
[EVIDENCE L0024] 
[EVIDENCE L0025] Before NNUE (2018-2020), chess engines had two camps:
[EVIDENCE L0026] - **Handcrafted eval** (Stockfish classic): fast, ran millions of nodes/sec, but limited by human knowledge
[EVIDENCE L0027] - **Large NN eval** (Leela/AlphaZero): superhuman knowledge, but ~1000x slower inference, needs GPU
[EVIDENCE L0028] 
[EVIDENCE L0029] **NNUE's insight**: You can have NN-quality evaluation at handcrafted-eval speeds by exploiting the *structure of how game states change incrementally*.
[EVIDENCE L0030] 
[EVIDENCE L0031] ### 1.2 Architecture
[EVIDENCE L0032] 
[EVIDENCE L0033] ```
[EVIDENCE L0034] Input Layer (HalfKP): 40,960 binary features (per perspective)
[EVIDENCE L0035]     |
[EVIDENCE L0036] Feature Transformer: 40960 -> 256 (with ACCUMULATOR)
[EVIDENCE L0037]     |
[EVIDENCE L0038] [White Accumulator (256)] || [Black Accumulator (256)] = 512
[EVIDENCE L0039]     |
[EVIDENCE L0040] Hidden Layer 1: 512 -> 32
[EVIDENCE L0041]     |
[EVIDENCE L0042] Hidden Layer 2: 32 -> 32
[EVIDENCE L0043]     |
[EVIDENCE L0044] Output: 32 -> 1 (centipawn score)
[EVIDENCE L0045] ```
[EVIDENCE L0046] 
[EVIDENCE L0047] Modern Stockfish uses **8 LayerStack buckets** (material-count-indexed mixture-of-experts).
[EVIDENCE L0048] 
[EVIDENCE L0049] ### 1.3 HalfKP Feature Encoding
[EVIDENCE L0050] 
[EVIDENCE L0051] "Half-King-Piece" -- encodes piece-king spatial relationships as binary features:
[EVIDENCE L0052] 
[EVIDENCE L0053] ```
[EVIDENCE L0054] index = piece_square + (piece_type * 2 + piece_color + king_square * 10) * 64
[EVIDENCE L0055] ```
[EVIDENCE L0056] 
[EVIDENCE L0057] - 64 king squares x 10 piece types (Q,R,B,N,P for each side) x 64 piece squares = **40,960 features**
[EVIDENCE L0058] - Both perspectives maintained = 81,920 total binary inputs
[EVIDENCE L0059] - Extreme sparsity: only ~30 active features per position (~0.07%)
[EVIDENCE L0060] 
[EVIDENCE L0061] ### 1.4 The Key Innovation: Incremental Accumulator Updates
[EVIDENCE L0062] 
[EVIDENCE L0063] This is the million-dollar insight. The first layer output (accumulator) is:
[EVIDENCE L0064] 
[EVIDENCE L0065] **Full computation:**
[EVIDENCE L0066] ```
[EVIDENCE L0067] accumulator = bias + SUM(W[:, i] for i in active_features)
[EVIDENCE L0068] ```
[EVIDENCE L0069] 
[EVIDENCE L0070] **Incremental update (after a move):**
[EVIDENCE L0071] ```
[EVIDENCE L0072] acc_new = acc_old - SUM(W[:, r] for r in removed_features)
[EVIDENCE L0073]                   + SUM(W[:, a] for a in added_features)
[EVIDENCE L0074] ```
[EVIDENCE L0075] 
[EVIDENCE L0076] A typical non-king move changes only **2-4 features** out of 40,960. So instead of
[EVIDENCE L0077] recomputing 40,960 x 256 multiplications, you do 2-4 vector additions/subtractions
[EVIDENCE L0078] of size 256. That's a **~10-15x speedup** over full recomputation.
[EVIDENCE L0079] 
[EVIDENCE L0080] **Why this works**: Because the input is *binary* (0/1), the "multiplication" is just
[EVIDENCE L0081] conditional addition. And because moves change very few pieces, the delta is tiny.
[EVIDENCE L0082] 
[EVIDENCE L0083] **King moves are special**: They invalidate ALL features for that perspective (every
[EVIDENCE L0084] feature encodes king position). Full refresh required. Stockfish uses "Finny Tables"
[EVIDENCE L0085] (cached accumulator per king bucket) to amortize this cost.
[EVIDENCE L0086] 
[EVIDENCE L0087] ### 1.5 Quantization for CPU SIMD
[EVIDENCE L0088] 
[EVIDENCE L0089] NNUE runs entirely in integer arithmetic on CPU SIMD:
[EVIDENCE L0090] 
[EVIDENCE L0091] | Component | Type | Scale | Range |
[EVIDENCE L0092] |-----------|------|-------|-------|
[EVIDENCE L0093] | FT weights/biases | int16 | 127 | [-127, 127] |
[EVIDENCE L0094] | FT activations | int16 (ClippedReLU) | - | [0, 127] |
[EVIDENCE L0095] | Hidden weights | int8 | 64 | [-127, 127] |
[EVIDENCE L0096] | Hidden activations | int8 (ClippedReLU) | - | [0, 127] |
[EVIDENCE L0097] | Output accumulation | int32 | - | full range |
[EVIDENCE L0098] 
[EVIDENCE L0099] **Feature Transformer quantization:**
[EVIDENCE L0100] ```
[EVIDENCE L0101] W_ij = round(127 * w_ij)     // float -> int16
[EVIDENCE L0102] B_j  = round(127 * b_j)
[EVIDENCE L0103] Y_j  = SUM(x_i * W_ij) + B_j  // x_i is 0 or 1, so this is conditional add
[EVIDENCE L0104] output_j = clamp(Y_j, 0, 127)  // ClippedReLU
[EVIDENCE L0105] ```
[EVIDENCE L0106] 
[EVIDENCE L0107] **Hidden layer quantization:**
[EVIDENCE L0108] ```
[EVIDENCE L0109] W_jk = round(64 * w_jk)      // float -> int8
[EVIDENCE L0110] B_k  = round(b_k * 127 * 64) // int32
[EVIDENCE L0111] Y_k  = (SUM(X_j * W_jk) + B_k) / 64
[EVIDENCE L0112] output_k = clamp(Y_k, 0, 127)
[EVIDENCE L0113] ```
[EVIDENCE L0114] 
[EVIDENCE L0115] ### 1.6 Advanced Techniques
[EVIDENCE L0116] 
[EVIDENCE L0117] - **SCReLU** (Squared ClippedReLU): `clamp(x,0,1)^2` -- stronger than CReLU, harder to vectorize
[EVIDENCE L0118] - **Pairwise Multiplication**: Split accumulator, multiply pairs to reduce width
[EVIDENCE L0119] - **King Input Buckets**: Multiple weight sets per king region (mixture-of-experts)
[EVIDENCE L0120] - **LayerStacks**: Switch post-accumulator parameters by material count
[EVIDENCE L0121] - **Lizard SCReLU trick**: Compute `(v*w)*v` instead of `(v*v)*w` to stay in int16 range
[EVIDENCE L0122] 
[EVIDENCE L0123] ---
[EVIDENCE L0124] 
[EVIDENCE L0125] ## 2. Novel NN + Classical Algorithm Hybrids
[EVIDENCE L0126] 
[EVIDENCE L0127] ### 2.1 Student of Games (DeepMind, 2023) -- GT-CFR
[EVIDENCE L0128] 
[EVIDENCE L0129] **Source**: [Science Advances 2023](https://www.science.org/doi/pdf/10.1126/sciadv.adg3256) | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10651118/)
[EVIDENCE L0130] 
[EVIDENCE L0131] **The paradigm break**: First algorithm that works for BOTH perfect AND imperfect
[EVIDENCE L0132] information games with a single unified framework. AlphaZero only works for perfect
[EVIDENCE L0133] info; CFR-based approaches only work for imperfect info. SoG does both.
[EVIDENCE L0134] 
[EVIDENCE L0135] **How it works -- Growing-Tree CFR (GT-CFR)**:
[EVIDENCE L0136] 1. Like MCTS, it grows a search tree non-uniformly toward promising states
[EVIDENCE L0137] 2. Like CFR, it uses regret minimization for game-theoretic soundness
[EVIDENCE L0138] 3. Uses a **Counterfactual Value-and-Policy Network (CVPN)** at frontier leaves
[EVIDENCE L0139] 
[EVIDENCE L0140] **CVPN (the neural network)**:
[EVIDENCE L0141] - Input: Public Belief State beta = (s_pub, r) where r = range distributions
[EVIDENCE L0142] - Output: Counterfactual values + policy targets
[EVIDENCE L0143] - Training losses: Huber loss (values) + cross-entropy (policy)
[EVIDENCE L0144] - Targets from both game outcomes AND bootstrapped GT-CFR solves
[EVIDENCE L0145] 
[EVIDENCE L0146] **Key insight -- Sound Self-Play**:
[EVIDENCE L0147] Searches at different public states must be *globally consistent* with each other.
[EVIDENCE L0148] This is trivial in perfect-info (each subtree is independent) but critical in
[EVIDENCE L0149] imperfect-info where beliefs propagate across the tree.
[EVIDENCE L0150] 
[EVIDENCE L0151] **Complexity**: GT-CFR re-solving with T iterations, k expanded children has
[EVIDENCE L0152] O(kT^2) public state visits. In perfect-info, simplifies to O(T) network calls.
[EVIDENCE L0153] 
[EVIDENCE L0154] ### 2.2 ReBeL (Meta/CMU, 2020) -- RL+Search for Imperfect Info
[EVIDENCE L0155] 
[EVIDENCE L0156] **Source**: [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf) | [Meta AI Blog](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/)
[EVIDENCE L0157] 
[EVIDENCE L0158] **The paradigm break**: Generalizes AlphaZero's "RL + Search" paradigm to imperfect
[EVIDENCE L0159] information games. Before ReBeL, RL+Search was believed fundamentally incompatible
[EVIDENCE L0160] with imperfect information.
[EVIDENCE L0161] 
[EVIDENCE L0162] **The core concept -- Public Belief States (PBS)**:
[EVIDENCE L0163] - In imperfect-info games, a "state" is a probability distribution over possible
[EVIDENCE L0164]   hidden states, not a single state
[EVIDENCE L0165] - PBS = (action_history, belief_distributions_per_player)
[EVIDENCE L0166] - This converts the imperfect-info game into a continuous-state perfect-info game
[EVIDENCE L0167] - Then you can apply AlphaZero-style RL+Search on this converted game
[EVIDENCE L0168] 
[EVIDENCE L0169] **Why naive conversion fails**: The PBS space has extremely high dimensionality.
[EVIDENCE L0170] In a toy poker game, the action space alone is 156-dimensional.
[EVIDENCE L0171] 
[EVIDENCE L0172] **How ReBeL handles it**: Uses CFR as a "gradient descent for games" -- an efficient
[EVIDENCE L0173] search procedure that exploits the convex optimization structure of two-player
[EVIDENCE L0174] zero-sum games. Proven to converge to Nash equilibrium.
[EVIDENCE L0175] 
[EVIDENCE L0176] ### 2.3 Deep Synoptic Monte Carlo Planning (DSMCP, 2021)
[EVIDENCE L0177] 
[EVIDENCE L0178] **Source**: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf)
[EVIDENCE L0179] 
[EVIDENCE L0180] **The paradigm break**: Uses particle filters + stochastic abstractions for
[EVIDENCE L0181] imperfect-info planning.
[EVIDENCE L0182] 
[EVIDENCE L0183] **How it works**:
[EVIDENCE L0184] 1. Maintains belief state via **unweighted particle filter** (set of possible worlds)
[EVIDENCE L0185] 2. Plans by sampling from belief state and doing playouts
[EVIDENCE L0186] 3. Instead of reasoning about exact information states, uses **"synopses"** --
[EVIDENCE L0187]    novel stochastic abstractions that summarize information states
[EVIDENCE L0188] 4. Neural networks evaluate synopsis-conditioned states
[EVIDENCE L0189] 
[EVIDENCE L0190] **Key insight**: You don't need to enumerate all possible hidden states. Sample
[EVIDENCE L0191] them, abstract them into "synopses" (statistical summaries), and plan over those.
[EVIDENCE L0192] 
[EVIDENCE L0193] ---
[EVIDENCE L0194] 
[EVIDENCE L0195] ## 3. Poker AI: Beyond Standard CFR
[EVIDENCE L0196] 
[EVIDENCE L0197] ### 3.1 Libratus (CMU, 2017) -- Three-Module Architecture
[EVIDENCE L0198] 
[EVIDENCE L0199] **Source**: [IJCAI 2017](https://noambrown.github.io/papers/17-IJCAI-Libratus.pdf) | [NSF](https://par.nsf.gov/servlets/purl/10077470)
[EVIDENCE L0200] 
[EVIDENCE L0201] First AI to defeat top human professionals in heads-up no-limit Texas Hold'em.
[EVIDENCE L0202] 
[EVIDENCE L0203] **Module 1: Blueprint Strategy (offline)**
[EVIDENCE L0204] - Game has ~10^161 decision points; abstracted to ~10^12
[EVIDENCE L0205] - Solved with MCCFR + **Regret-Based Pruning (RBP)**
[EVIDENCE L0206]   - RBP: skip branches with strongly negative cumulative regret
[EVIDENCE L0207]   - Speeds convergence AND mitigates imperfect-recall abstraction errors
[EVIDENCE L0208] - No card abstraction on preflop/flop; coarser on later rounds:
[EVIDENCE L0209]   - Round 3: 55M hands -> 2.5M buckets
[EVIDENCE L0210]   - Round 4: 2.4B possibilities -> 1.25M buckets
[EVIDENCE L0211] 
[EVIDENCE L0212] **Module 2: Nested Safe Subgame Solving (online)**
[EVIDENCE L0213] This is the key novelty:
[EVIDENCE L0214] - When opponent makes any bet, construct and solve a NEW subgame in real-time
[EVIDENCE L0215] - Each subgame uses: NO card abstraction + DENSE action abstraction
[EVIDENCE L0216] - **Safety guarantee**: New strategy is provably no worse than blueprint
[EVIDENCE L0217] - Uses opponent's cumulative mistakes to EXPAND the safe optimization polytope
[EVIDENCE L0218] - **Dynamic action abstractions**: Each subgame uses different bet sizes,
[EVIDENCE L0219]   forcing opponents to constantly adapt
[EVIDENCE L0220] 
[EVIDENCE L0221] Prior approaches used "action translation" (round off-tree bets to nearest
[EVIDENCE L0222] in-tree action). Libratus eliminates this entirely in later rounds.
[EVIDENCE L0223] 
[EVIDENCE L0224] **Module 3: Self-Improver (background)**
[EVIDENCE L0225] - Monitors which off-tree actions opponent plays most frequently
[EVIDENCE L0226] - Adds those actions to the abstraction
[EVIDENCE L0227] - Selection criterion: frequency * distance_from_nearest_abstract_action
[EVIDENCE L0228] - Computes new strategy for each added action via subgame solving
[EVIDENCE L0229] 
[EVIDENCE L0230] ### 3.2 Pluribus (Meta/CMU, 2019) -- Multiplayer Depth-Limited Search
[EVIDENCE L0231] 
[EVIDENCE L0232] **Source**: [Science 2019](https://www.science.org/doi/10.1126/science.aay2400)
[EVIDENCE L0233] 
[EVIDENCE L0234] First superhuman AI for 6-player no-limit Texas Hold'em.
[EVIDENCE L0235] 
[EVIDENCE L0236] **Key novelty -- Depth-Limited Imperfect-Info Search**:
[EVIDENCE L0237] 
[EVIDENCE L0238] Libratus could only search when close enough to solve to endgame. With 6 players,
[EVIDENCE L0239] the game tree explodes exponentially. Pluribus solved this with **depth-limited
[EVIDENCE L0240] search + continuation policies**.
[EVIDENCE L0241] 
[EVIDENCE L0242] **Blueprint**: Computed with Monte Carlo Linear CFR
[EVIDENCE L0243] - **Linear CFR**: Iteration T weighted by T (not 1), so early random iterations
[EVIDENCE L0244]   decay as 1/SUM(t=1..T, t) instead of 1/T. Much faster convergence.
[EVIDENCE L0245] 
[EVIDENCE L0246] **Real-time search with continuation policies**:
[EVIDENCE L0247] At the depth limit (where you can't solve to endgame), each player independently
[EVIDENCE L0248] chooses from k=4 **continuation strategies**:
[EVIDENCE L0249] 1. Blueprint strategy (balanced play)
[EVIDENCE L0250] 2. Blueprint biased toward folding
[EVIDENCE L0251] 3. Blueprint biased toward calling
[EVIDENCE L0252] 4. Blueprint biased toward raising
[EVIDENCE L0253] 
[EVIDENCE L0254] Terminal values estimated by **rolling out** the remainder of the game with the
[EVIDENCE L0255] selected continuation profile. This avoids the fundamental problem of
[EVIDENCE L0256] imperfect-info games: leaf values depend on the strategy chosen in the subgame.
[EVIDENCE L0257] 
[EVIDENCE L0258] **During search**: Pluribus tracks its **range** (probability distribution over
[EVIDENCE L0259] private hands) and computes a strategy that's balanced across ALL possible hands,
[EVIDENCE L0260] then samples the action for the actual hand held.
[EVIDENCE L0261] 
[EVIDENCE L0262] ### 3.3 What Made Poker AI Paradigm-Breaking (Summary)
[EVIDENCE L0263] 
[EVIDENCE L0264] | Innovation | Why It Matters |
[EVIDENCE L0265] |-----------|---------------|
[EVIDENCE L0266] | Safe subgame solving | Proves new strategy >= old, enables online refinement |
[EVIDENCE L0267] | Depth-limited search + continuation policies | Makes online search tractable for multiplayer |
[EVIDENCE L0268] | Linear CFR weighting | Faster convergence by discounting early noise |
[EVIDENCE L0269] | Blueprint + online refinement | Two-phase architecture: coarse offline, fine online |
[EVIDENCE L0270] | Regret-based pruning | Skip losing branches, focus compute on promising ones |
[EVIDENCE L0271] | Range-balanced play | Strategy coherent across all possible private states |
[EVIDENCE L0272] 
[EVIDENCE L0273] ---
[EVIDENCE L0274] 
[EVIDENCE L0275] ## 4. Recent Imperfect Information Game AI (2020-2025)
[EVIDENCE L0276] 
[EVIDENCE L0277] ### 4.1 DeepNash / R-NaD (DeepMind, 2022) -- Model-Free Nash Convergence
[EVIDENCE L0278] 
[EVIDENCE L0279] **Source**: [Science 2022](https://www.science.org/doi/10.1126/science.add4679) | [arXiv:2206.15378](https://arxiv.org/abs/2206.15378)
[EVIDENCE L0280] 
[EVIDENCE L0281] Mastered Stratego (10^535 game tree, 10^175x larger than Go) with NO search at all.
[EVIDENCE L0282] 
[EVIDENCE L0283] **The paradigm break**: Model-free RL that provably converges to Nash equilibrium
[EVIDENCE L0284] in imperfect-info games. No CFR, no search, no explicit belief modeling.
[EVIDENCE L0285] 
[EVIDENCE L0286] **R-NaD Algorithm (Regularized Nash Dynamics)**:
[EVIDENCE L0287] 
[EVIDENCE L0288] The core problem: standard policy gradient in multi-agent games CYCLES around Nash
[EVIDENCE L0289] equilibrium (like Rock-Paper-Scissors -- you keep adjusting and overshooting).
[EVIDENCE L0290] 
[EVIDENCE L0291] R-NaD fixes this with a **reward transformation**:
[EVIDENCE L0292] 
[EVIDENCE L0293] ```
[EVIDENCE L0294] r_transformed(pi_i, pi_-i, a_i, a_-i) =
[EVIDENCE L0295]     r(a_i, a_-i)
[EVIDENCE L0296]     - eta * log(pi_i(a_i) / pi_reg_i(a_i))     // penalize deviation from reg
[EVIDENCE L0297]     + eta * log(pi_-i(a_-i) / pi_reg_-i(a_-i))  // reward opponent staying close
[EVIDENCE L0298] ```
[EVIDENCE L0299] 
[EVIDENCE L0300] Where:
[EVIDENCE L0301] - eta > 0 is the regularization strength
[EVIDENCE L0302] - pi_reg is the "regularization policy" (the anchor point)
[EVIDENCE L0303] - The log-ratio terms are gradients of KL divergence
[EVIDENCE L0304] 
[EVIDENCE L0305] **Why this prevents cycling -- Lyapunov function**:
[EVIDENCE L0306] 
[EVIDENCE L0307] The transformed game has a UNIQUE fixed point pi_fix. The distance to this fixed
[EVIDENCE L0308] point decreases exponentially:
[EVIDENCE L0309] 
[EVIDENCE L0310] ```
[EVIDENCE L0311] d/dt H(pi_fix, pi_t) <= -eta * H(pi_fix, pi_t)
[EVIDENCE L0312] ```
[EVIDENCE L0313] 
[EVIDENCE L0314] where H is the KL divergence from pi_fix to current policy pi_t.
[EVIDENCE L0315] 
[EVIDENCE L0316] **The nested loop structure**:
[EVIDENCE L0317] 1. OUTER LOOP: Set regularization policy pi_reg
[EVIDENCE L0318] 2. INNER LOOP: Run replicator dynamics (policy gradient) on transformed game
[EVIDENCE L0319]    until convergence to fixed point pi_fix
[EVIDENCE L0320] 3. UPDATE: Set pi_reg = pi_fix for next outer iteration
[EVIDENCE L0321] 4. REPEAT: Sequence of fixed points converges to Nash equilibrium of ORIGINAL game
[EVIDENCE L0322] 
[EVIDENCE L0323] **NeuRD (Neural Replicator Dynamics) for deep learning**:
[EVIDENCE L0324] 
[EVIDENCE L0325] In practice, R-NaD uses NeuRD -- a neural network parameterization of the
[EVIDENCE L0326] replicator dynamics:
[EVIDENCE L0327] 
[EVIDENCE L0328] - Fast parameters (theta_n): Updated every step via Adam on NeuRD loss
[EVIDENCE L0329] - Slow target parameters: theta_{n+1,target} = gamma * theta_{n+1} + (1-gamma) * theta_{n,target}
[EVIDENCE L0330] - After Delta_m steps: extract pi_fix from slow params, set as new pi_reg
[EVIDENCE L0331] 
[EVIDENCE L0332] The NeuRD update operates on LOGITS (not probabilities), with clipping to prevent
[EVIDENCE L0333] logit explosion:
[EVIDENCE L0334] 
[EVIDENCE L0335] ```
[EVIDENCE L0336] Lambda_n = -[lr * grad(L_critic) + (1/T) * SUM_t SUM_a grad(logit(a) * Clip(Q(a), c))]
[EVIDENCE L0337] ```
[EVIDENCE L0338] 
[EVIDENCE L0339] **For Mahjong relevance**: R-NaD shows you can get Nash equilibrium convergence
[EVIDENCE L0340] WITHOUT search, WITHOUT explicit belief tracking, purely through model-free RL
[EVIDENCE L0341] with the right reward transformation. This is MASSIVE for 4-player mahjong where
[EVIDENCE L0342] search is computationally intractable.
[EVIDENCE L0343] 
[EVIDENCE L0344] ### 4.2 Suphx (Microsoft Research, 2020) -- State of Mahjong AI
[EVIDENCE L0345] 
[EVIDENCE L0346] **Source**: [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
[EVIDENCE L0347] 
[EVIDENCE L0348] The current strongest Mahjong AI (10-dan on Tenhou, top 0.01% of humans).
[EVIDENCE L0349] 
[EVIDENCE L0350] **Three key techniques**:
[EVIDENCE L0351] 
[EVIDENCE L0352] 1. **Global Reward Prediction**: Instead of per-hand reward, predict tournament-level
[EVIDENCE L0353]    outcomes. Aligns training signal with actual competitive objective.
[EVIDENCE L0354] 
[EVIDENCE L0355] 2. **Oracle Guiding**: Train with oracle (perfect information) as teacher, then
[EVIDENCE L0356]    distill to imperfect-info student. The oracle sees all tiles; the student learns
[EVIDENCE L0357]    to approximate oracle decisions from partial information.
[EVIDENCE L0358] 
[EVIDENCE L0359] 3. **Run-time Policy Adaptation**: Adjust policy during play based on observed
[EVIDENCE L0360]    opponent patterns. Not fixed strategy -- adapts in real-time.
[EVIDENCE L0361] 
[EVIDENCE L0362] **What Suphx does NOT do**: No search, no CFR, no explicit belief modeling over
[EVIDENCE L0363] opponent hands. It's a pure policy network with clever training tricks.
[EVIDENCE L0364] 
[EVIDENCE L0365] ### 4.3 Bayesian Opponent Modeling with Belief Updates (2024-2025)
[EVIDENCE L0366] 
[EVIDENCE L0367] **Source**: [arXiv:2405.14122](https://arxiv.org/abs/2405.14122) | [HORSE-CFR (2024)](https://www.sciencedirect.com/science/article/pii/S0957417424025648)
[EVIDENCE L0368] 
[EVIDENCE L0369] Recent work combines Bayesian belief tracking with game-theoretic solving:
[EVIDENCE L0370] 
[EVIDENCE L0371] **Key concepts**:
[EVIDENCE L0372] - Maintain posterior distribution over opponent types/strategies
[EVIDENCE L0373] - Update beliefs using Bayes' theorem after each observed action
[EVIDENCE L0374] - Use updated beliefs to select exploitation strategy
[EVIDENCE L0375] - Balance between Nash (safe) play and exploitative (Bayesian) play
[EVIDENCE L0376] 
[EVIDENCE L0377] **HORSE-CFR**: Hierarchical Opponent Reasoning for Safe Exploitation
[EVIDENCE L0378] - Neural network infers missing information to improve Bayesian posterior accuracy
[EVIDENCE L0379] - Accounts for UNCERTAINTY in the belief update (not just point estimates)
[EVIDENCE L0380] - Hierarchical: reasons about opponent's model of YOUR strategy
[EVIDENCE L0381] 
[EVIDENCE L0382] ### 4.4 Preference-CFR: Beyond Nash Equilibrium (2024)
[EVIDENCE L0383] 
[EVIDENCE L0384] **Source**: [Semantic Scholar](https://www.semanticscholar.org/paper/Preference-CFR%3A-Beyond-Nash-Equilibrium-for-Better-Ju-Tellier/548481122339a162bf1bba36f878536380003061)
[EVIDENCE L0385] 
[EVIDENCE L0386] **Key insight**: Nash equilibrium is DEFENSIVE -- it's the unexploitable strategy.
[EVIDENCE L0387] But against weak opponents, you want to EXPLOIT their mistakes, not just be safe.
[EVIDENCE L0388] Preference-CFR computes strategies that go beyond Nash by incorporating preferences
[EVIDENCE L0389] about opponent tendencies.
[EVIDENCE L0390] 
[EVIDENCE L0391] ---
[EVIDENCE L0392] 
[EVIDENCE L0393] ## 5. Cross-Domain Approaches
[EVIDENCE L0394] 
[EVIDENCE L0395] ### 5.1 Decision Transformers (2021) -- RL as Sequence Modeling
[EVIDENCE L0396] 
[EVIDENCE L0397] **Source**: [NeurIPS 2021](https://openreview.net/forum?id=gaCGNwsWITG) | [Berkeley](https://sites.google.com/berkeley.edu/decision-transformer)
[EVIDENCE L0398] 
[EVIDENCE L0399] **The paradigm break**: Recast reinforcement learning as a SEQUENCE MODELING problem.
[EVIDENCE L0400] No value functions, no policy gradients, no Bellman equations. Just a transformer
[EVIDENCE L0401] that predicts the next action given the history.
[EVIDENCE L0402] 
[EVIDENCE L0403] **Architecture**:
[EVIDENCE L0404] - Input sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_t, s_t)
[EVIDENCE L0405] - R_t = return-to-go (desired future cumulative reward)
[EVIDENCE L0406] - Output: predicted action a_t
[EVIDENCE L0407] - Trained on offline trajectories via standard cross-entropy/MSE loss
[EVIDENCE L0408] - At inference: set R_1 = desired_return, autoregressive generation
[EVIDENCE L0409] 
[EVIDENCE L0410] **Why it matters**: No need for temporal difference learning, no bootstrapping,
[EVIDENCE L0411] no exploration-exploitation tradeoff. The transformer learns the MAPPING from
[EVIDENCE L0412] (desired outcome + history) -> action. Want better play? Set higher return-to-go.
[EVIDENCE L0413] 
[EVIDENCE L0414] **For Mahjong**: A mahjong game is naturally a sequence of observations and actions.
[EVIDENCE L0415] A Decision Transformer could learn from expert replays without any reward shaping.
[EVIDENCE L0416] 
[EVIDENCE L0417] ### 5.2 DreamerV3 (2023) -- World Models
[EVIDENCE L0418] 
[EVIDENCE L0419] **Source**: [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2) | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
[EVIDENCE L0420] 
[EVIDENCE L0421] **The paradigm break**: Learn a world model in LATENT space, then train the policy
[EVIDENCE L0422] entirely inside "dreams" (imagined trajectories). No real environment interaction
[EVIDENCE L0423] needed during policy optimization.
[EVIDENCE L0424] 
[EVIDENCE L0425] **Architecture**:
[EVIDENCE L0426] 1. **Encoder**: observation -> latent state z_t
[EVIDENCE L0427] 2. **Dynamics model**: (z_t, a_t) -> z_{t+1} (predicts next latent state)
[EVIDENCE L0428] 3. **Reward predictor**: z_t -> r_t
[EVIDENCE L0429] 4. **Decoder**: z_t -> reconstructed observation (for training signal)
[EVIDENCE L0430] 5. **Actor-Critic**: trained on imagined trajectories in latent space
[EVIDENCE L0431] 
[EVIDENCE L0432] **Key innovation -- Symlog predictions**: Handles rewards across many orders of
[EVIDENCE L0433] magnitude without manual normalization. Uses symlog(x) = sign(x) * ln(|x| + 1).
[EVIDENCE L0434] 
[EVIDENCE L0435] **For Mahjong**: A world model could learn the "physics" of mahjong tile dynamics
[EVIDENCE L0436] -- what draws are likely given visible information, how opponent strategies evolve.
[EVIDENCE L0437] Train policy in imagined games rather than expensive self-play.
[EVIDENCE L0438] 
[EVIDENCE L0439] ### 5.3 MPC + RL Unification (Bertsekas, 2024) -- Newton's Method Bridge
[EVIDENCE L0440] 
[EVIDENCE L0441] **Source**: [arXiv:2406.00592](https://arxiv.org/abs/2406.00592) | [MIT](https://web.mit.edu/dimitrib/www/IFAC_Overview_Paper_2024.pdf)
[EVIDENCE L0442] 
[EVIDENCE L0443] **The paradigm break**: Shows that Model Predictive Control (MPC) and RL are
[EVIDENCE L0444] actually the SAME algorithm viewed through different lenses, connected by
[EVIDENCE L0445] Newton's method for solving Bellman equations.
[EVIDENCE L0446] 
[EVIDENCE L0447] **Two-phase architecture**:
[EVIDENCE L0448] 1. **Offline training**: Learn approximate value function via RL/self-play
[EVIDENCE L0449] 2. **Online planning**: Use value function as terminal cost in MPC-style
[EVIDENCE L0450]    lookahead planning (Newton step refinement)
[EVIDENCE L0451] 
[EVIDENCE L0452] The offline phase provides the "landscape"; the online phase does local
[EVIDENCE L0453] optimization on that landscape. Each improves the other.
[EVIDENCE L0454] 
[EVIDENCE L0455] **For Mahjong**: This is essentially what Libratus does (blueprint + online
[EVIDENCE L0456] subgame solving), but formalized mathematically. Could inspire a principled
[EVIDENCE L0457] offline/online architecture for Hydra.
[EVIDENCE L0458] 
[EVIDENCE L0459] ### 5.4 Particle Filter Belief Tracking for Games (DSMCP, 2021)
[EVIDENCE L0460] 
[EVIDENCE L0461] Covered in Section 2.3, but from a cross-domain perspective: particle filters
[EVIDENCE L0462] are standard in robotics (SLAM, object tracking) but novel in game AI.
[EVIDENCE L0463] 
[EVIDENCE L0464] **Key cross-pollination idea**: Instead of maintaining exact beliefs over opponent
[EVIDENCE L0465] hands (combinatorially explosive), maintain a PARTICLE SET -- a collection of
[EVIDENCE L0466] plausible opponent hand configurations -- and update them via sequential Monte
[EVIDENCE L0467] Carlo methods as new evidence (discards, calls, etc.) arrives.
[EVIDENCE L0468] 
[EVIDENCE L0469] ---
[EVIDENCE L0470] 
[EVIDENCE L0471] ## 6. Synthesis: Paradigm-Breaking Ideas for Hydra Mahjong AI
[EVIDENCE L0472] 
[EVIDENCE L0473] ### 6.1 The Mahjong Problem Space
[EVIDENCE L0474] 
[EVIDENCE L0475] Mahjong uniquely combines challenges from multiple domains:
[EVIDENCE L0476] - **Imperfect info** (like poker): 3 opponents' hands + wall are hidden
[EVIDENCE L0477] - **Sequential decisions** (like chess): ~70 decision points per hand
[EVIDENCE L0478] - **4 players** (like Pluribus): No Nash equilibrium guaranteed in general
[EVIDENCE L0479] - **Stochastic** (like backgammon): Tile draws are random
[EVIDENCE L0480] - **Rich action space**: Discard (34), Chi/Pon/Kan, Riichi, Tsumo, Ron
[EVIDENCE L0481] 
[EVIDENCE L0482] ### 6.2 Idea 1: NNUE-Style Incremental Encoding for Mahjong
[EVIDENCE L0483] 
[EVIDENCE L0484] **Inspiration**: NNUE's accumulator update
[EVIDENCE L0485] 
[EVIDENCE L0486] **The insight**: In mahjong, each action (draw, discard, call) changes very few
[EVIDENCE L0487] features in the game state. The current Hydra 85x34 encoding could benefit from
[EVIDENCE L0488] NNUE-style incremental updates:
[EVIDENCE L0489] 
[EVIDENCE L0490] - A tile draw: 1 tile moves from wall to hand (+1 feature change)
[EVIDENCE L0491] - A discard: 1 tile moves from hand to discard pond (+2 feature changes)
[EVIDENCE L0492] - A call (chi/pon): 2-3 tiles move from hand to melds (+3-4 feature changes)
[EVIDENCE L0493] 
[EVIDENCE L0494] **Proposed architecture**:
[EVIDENCE L0495] ```
[EVIDENCE L0496] Feature Transformer: sparse_features -> 256 (with accumulator)
[EVIDENCE L0497]     |-- Incremental update on each action
[EVIDENCE L0498]     |
[EVIDENCE L0499] Residual Blocks: 256-channel SE-ResNet (current Hydra architecture)
[EVIDENCE L0500]     |
[EVIDENCE L0501] Output Heads: Policy(46) + Value(1) + GRP(24) + Tenpai(3) + Danger(3x34)
[EVIDENCE L0502] ```
[EVIDENCE L0503] 
[EVIDENCE L0504] The Feature Transformer maintains an accumulator that's incrementally updated.
[EVIDENCE L0505] The residual blocks still run fully each time (they're the "thinking" part),
[EVIDENCE L0506] but the expensive input encoding is amortized.
[EVIDENCE L0507] 
[EVIDENCE L0508] **Potential speedup**: During search/simulation, if Hydra ever does lookahead,
[EVIDENCE L0509] the accumulator updates would make position evaluation much cheaper.
[EVIDENCE L0510] 
[EVIDENCE L0511] **Quantization angle**: NNUE's int8/int16 quantization could let Hydra run
[EVIDENCE L0512] inference on CPU without GPU dependency -- critical for deployment.
[EVIDENCE L0513] 
[EVIDENCE L0514] ### 6.3 Idea 2: R-NaD-Style Training for 4-Player Convergence
[EVIDENCE L0515] 
[EVIDENCE L0516] **Inspiration**: DeepNash's R-NaD
[EVIDENCE L0517] 
[EVIDENCE L0518] **The problem**: Standard self-play RL in 4-player games doesn't converge to Nash.
[EVIDENCE L0519] The policies cycle. This is why most mahjong AIs (Suphx, Mortal) use supervised
[EVIDENCE L0520] learning from human data as a major component.
[EVIDENCE L0521] 
[EVIDENCE L0522] **Proposed approach**: Apply R-NaD's reward transformation to Hydra's PPO training:
[EVIDENCE L0523] 
[EVIDENCE L0524] ```
[EVIDENCE L0525] r_transformed_i = r_original_i
[EVIDENCE L0526]     - eta * KL(pi_i || pi_reg_i)      // stay near anchor
[EVIDENCE L0527]     + eta * SUM_j!=i KL(pi_j || pi_reg_j)  // adapted for 4-player
[EVIDENCE L0528] ```
[EVIDENCE L0529] 
[EVIDENCE L0530] The nested loop:
[EVIDENCE L0531] 1. Train with current pi_reg as anchor (inner loop, many PPO steps)
[EVIDENCE L0532] 2. Extract converged policy, set as new pi_reg (outer loop)
[EVIDENCE L0533] 3. Repeat until equilibrium
[EVIDENCE L0534] 
[EVIDENCE L0535] **Challenge**: R-NaD is proven for 2-player zero-sum. Mahjong is 4-player,
[EVIDENCE L0536] not strictly zero-sum (one player wins, three lose, but with varying scores).
[EVIDENCE L0537] The Lyapunov convergence proof may not hold. However, empirically R-NaD-style
[EVIDENCE L0538] regularization could still stabilize training.
[EVIDENCE L0539] 
[EVIDENCE L0540] ### 6.4 Idea 3: Pluribus-Style Depth-Limited Search with Continuation Policies
[EVIDENCE L0541] 
[EVIDENCE L0542] **Inspiration**: Pluribus
[EVIDENCE L0543] 
[EVIDENCE L0544] **The approach**: Hydra trains a policy network (like current plan). At inference
[EVIDENCE L0545] time, before making critical decisions, run a lightweight depth-limited search:
[EVIDENCE L0546] 
[EVIDENCE L0547] 1. For the current game state, enumerate K plausible opponent hand configurations
[EVIDENCE L0548]    (using the danger/tenpai heads as a belief model)
[EVIDENCE L0549] 2. For each configuration, simulate D turns ahead using the policy network
[EVIDENCE L0550] 3. At depth limit, evaluate using k=4 continuation variants:
[EVIDENCE L0551]    - Balanced (base policy)
[EVIDENCE L0552]    - Defensive (bias toward safe discards)
[EVIDENCE L0553]    - Aggressive (bias toward riichi/calls)
[EVIDENCE L0554]    - Opportunistic (bias toward value hands)
[EVIDENCE L0555] 4. Average values across opponent configurations, weighted by belief probability
[EVIDENCE L0556] 5. Choose action with highest expected value
[EVIDENCE L0557] 
[EVIDENCE L0558] **Key advantage**: The policy network provides fast evaluation. Search only happens
[EVIDENCE L0559] at critical decision points (riichi decisions, dangerous discards, calling choices).
[EVIDENCE L0560] Most turns use the policy network directly.
[EVIDENCE L0561] 
[EVIDENCE L0562] ### 6.5 Idea 4: Particle Filter Belief Tracking
[EVIDENCE L0563] 
[EVIDENCE L0564] **Inspiration**: DSMCP + robotics SLAM
[EVIDENCE L0565] 
[EVIDENCE L0566] **The approach**: Instead of encoding beliefs about opponent hands as fixed features,
[EVIDENCE L0567] maintain a set of N=1000 "particles" -- each particle is a complete assignment of
[EVIDENCE L0568] hidden tiles to opponents + wall.
[EVIDENCE L0569] 
[EVIDENCE L0570] After each observed action (discard, call, skip):
[EVIDENCE L0571] 1. Weight each particle by P(action | particle's hidden state, opponent_policy)
[EVIDENCE L0572] 2. Resample particles proportional to weights
[EVIDENCE L0573] 3. Add noise (jitter) to prevent particle depletion
[EVIDENCE L0574] 
[EVIDENCE L0575] This gives a continuously updated Bayesian belief over the hidden game state,
[EVIDENCE L0576] which can be:
[EVIDENCE L0577] - Summarized as features for the policy network
[EVIDENCE L0578] - Used directly for search (sample particles, plan in each world)
[EVIDENCE L0579] - Used to compute danger estimates (what fraction of particles have opponent tenpai?)
[EVIDENCE L0580] 
[EVIDENCE L0581] ### 6.6 Idea 5: Decision Transformer for Mahjong
[EVIDENCE L0582] 
[EVIDENCE L0583] **Inspiration**: Decision Transformers + Suphx's oracle guiding
[EVIDENCE L0584] 
[EVIDENCE L0585] **The approach**: Frame mahjong as sequence modeling:
[EVIDENCE L0586] 
[EVIDENCE L0587] Input tokens:
[EVIDENCE L0588] ```
[EVIDENCE L0589] [R_target, obs_1, action_1, obs_2, action_2, ..., obs_t]
[EVIDENCE L0590] -> predict action_t
[EVIDENCE L0591] ```
[EVIDENCE L0592] 
[EVIDENCE L0593] Where R_target is the desired placement (1st/2nd/3rd/4th) or score.
[EVIDENCE L0594] 
[EVIDENCE L0595] **Training**: On human expert replays from Tenhou. No reward shaping needed.
[EVIDENCE L0596] The transformer learns the mapping: (desired outcome + game history) -> action.
[EVIDENCE L0597] 
[EVIDENCE L0598] **Key advantage**: At inference, set R_target = "1st place" and the model
[EVIDENCE L0599] generates actions conditioned on achieving that goal. Want safer play?
[EVIDENCE L0600] Set R_target = "2nd place". This gives natural risk-reward control.
[EVIDENCE L0601] 
[EVIDENCE L0602] **Combined with oracle guiding**: Train two models:
[EVIDENCE L0603] 1. Oracle DT: sees all hands, conditioned on outcome
[EVIDENCE L0604] 2. Student DT: sees only own hand, trained to match oracle's actions
[EVIDENCE L0605] 
[EVIDENCE L0606] This is essentially Suphx's approach but using transformer architecture
[EVIDENCE L0607] instead of CNN, potentially capturing longer-range dependencies.
[EVIDENCE L0608] 
[EVIDENCE L0609] ### 6.7 Idea 6: World Model for Mahjong (DreamerV3-inspired)
[EVIDENCE L0610] 
[EVIDENCE L0611] **Inspiration**: DreamerV3
[EVIDENCE L0612] 
[EVIDENCE L0613] **The approach**: Learn a latent dynamics model of mahjong:
[EVIDENCE L0614] 
[EVIDENCE L0615] 1. **Encoder**: game_observation -> z_t (latent state, ~256 dims)
[EVIDENCE L0616] 2. **Dynamics**: (z_t, action_t) -> z_{t+1}
[EVIDENCE L0617]    But also: (z_t, opponent_action_t) -> z_{t+1}
[EVIDENCE L0618]    And: (z_t) -> draw_distribution (what tile am I likely to draw?)
[EVIDENCE L0619] 3. **Reward**: z_t -> expected_final_score
[EVIDENCE L0620] 4. **Policy**: trained in "dreams" -- imagined game trajectories in latent space
[EVIDENCE L0621] 
[EVIDENCE L0622] **Key advantage for mahjong**: The dynamics model implicitly learns:
[EVIDENCE L0623] - What tiles are likely still in the wall
[EVIDENCE L0624] - How opponents' discards correlate with their hands
[EVIDENCE L0625] - The "flow" of a mahjong game (early game exploration -> mid game direction -> endgame defense)
[EVIDENCE L0626] 
[EVIDENCE L0627] **Critical challenge**: Mahjong has very high stochasticity. Each draw is random
[EVIDENCE L0628] from a depleting wall. The world model needs to capture this uncertainty well.
[EVIDENCE L0629] DreamerV3's symlog predictions could help with the varying reward scales in mahjong.
[EVIDENCE L0630] 
[EVIDENCE L0631] ### 6.8 Idea 7: Two-Phase Architecture (Most Promising Synthesis)
[EVIDENCE L0632] 
[EVIDENCE L0633] **Inspiration**: Libratus/Pluribus + NNUE + R-NaD
[EVIDENCE L0634] 
[EVIDENCE L0635] The most promising approach combines multiple paradigms:
[EVIDENCE L0636] 
[EVIDENCE L0637] **Phase 1: Offline (Training)**
[EVIDENCE L0638] - Train a strong policy network using R-NaD-style regularized self-play
[EVIDENCE L0639] - Use NNUE-style architecture: incremental feature transformer + residual blocks
[EVIDENCE L0640] - Multi-head output: policy + value + danger + tenpai + game result prediction
[EVIDENCE L0641] - Quantize to int8/int16 for CPU inference
[EVIDENCE L0642] 
[EVIDENCE L0643] **Phase 2: Online (Inference)**
[EVIDENCE L0644] - Most turns: use policy network directly (fast, ~1ms per decision)
[EVIDENCE L0645] - Critical decisions (riichi, dangerous discards, late game):
[EVIDENCE L0646]   - Run particle filter to estimate opponent hands
[EVIDENCE L0647]   - Do depth-limited search (4-8 turns ahead) with policy network as evaluator
[EVIDENCE L0648]   - Use Pluribus-style continuation policies at search leaves
[EVIDENCE L0649]   - Choose action balancing expected value across belief particles
[EVIDENCE L0650] 
[EVIDENCE L0651] **Why this could be paradigm-breaking for mahjong**:
[EVIDENCE L0652] - Current SOTA (Mortal, Suphx) uses pure policy networks with NO online search
[EVIDENCE L0653] - Adding even lightweight search at critical moments could be a significant jump
[EVIDENCE L0654] - NNUE-style incremental updates make search feasible on CPU
[EVIDENCE L0655] - R-NaD-style training gives better convergence than standard self-play
[EVIDENCE L0656] - Particle filter beliefs give principled opponent modeling
[EVIDENCE L0657] 
[EVIDENCE L0658] ---
[EVIDENCE L0659] 
[EVIDENCE L0660] ## Key References
[EVIDENCE L0661] 
[EVIDENCE L0662] 1. **NNUE**: Yu Nasu (2018). Chessprogramming Wiki. Stockfish NNUE-PyTorch.
[EVIDENCE L0663] 2. **Student of Games**: Schmid et al. (2023). Science Advances.
[EVIDENCE L0664] 3. **ReBeL**: Brown et al. (2020). NeurIPS 2020.
[EVIDENCE L0665] 4. **Libratus**: Brown & Sandholm (2017). IJCAI 2017.
[EVIDENCE L0666] 5. **Pluribus**: Brown & Sandholm (2019). Science.
[EVIDENCE L0667] 6. **DeepNash / R-NaD**: Perolat et al. (2022). Science.
[EVIDENCE L0668] 7. **DSMCP**: Markowitz et al. (2021). NeurIPS 2021.
[EVIDENCE L0669] 8. **Decision Transformer**: Chen et al. (2021). NeurIPS 2021.
[EVIDENCE L0670] 9. **DreamerV3**: Hafner et al. (2023). Nature 2025.
[EVIDENCE L0671] 10. **Suphx**: Li et al. (2020). arXiv:2003.13590.
[EVIDENCE L0672] 11. **MPC+RL**: Bertsekas (2024). arXiv:2406.00592.
[EVIDENCE L0673] 12. **Preference-CFR**: Ju & Tellier (2024).
[EVIDENCE L0674] 13. **HORSE-CFR**: (2024). Expert Systems with Applications.
```

## Artifact 16 — R-NaD and DRDA evidence report
Artifact id: `rnad-drda-report`
Source label: EVIDENCE
Type: `file_full`
Source: `research/evidence/rnad_drda_report.md`
Why it matters: Useful nearby reinforcement-learning evidence. Also helps the agent avoid acronym confusion between DRDA and DCRL while comparing Hydra's existing research intake against the claimed omission.

```markdown
[EVIDENCE L0001] # R-NaD vs DRDA: Neural-Scale Evidence Report
[EVIDENCE L0002] 
[EVIDENCE L0003] ## TL;DR
[EVIDENCE L0004] 
[EVIDENCE L0005] **The judges are right about DRDA -- it has ZERO neural-scale experiments.** But you can pivot this into a strength: R-NaD (the algorithm family DRDA extends) IS proven at neural scale via DeepNash. The gap is specifically in DRDA's paper, not in the regularized dynamics approach itself.
[EVIDENCE L0006] 
[EVIDENCE L0007] | Algorithm | Neural-Scale Proof? | Largest Game | Compute | Open-Source? |
[EVIDENCE L0008] |-----------|-------------------|--------------|---------|--------------|
[EVIDENCE L0009] | **R-NaD** (DeepNash) | YES -- 1024 TPUs, Stratego (10^535 states) | Stratego (full 10x10) | 768 learner + 256 actor TPU nodes | NO (DeepMind internal) |
[EVIDENCE L0010] | **DRDA** (ICLR 2025) | NO -- purely tabular | 4-player Kuhn poker (6 ranks) | Dynamic programming | NO |
[EVIDENCE L0011] | **NFSP** | YES -- Texas Hold'em | Limit Texas Hold'em | Single GPU | YES (OpenSpiel) |
[EVIDENCE L0012] 
[EVIDENCE L0013] ---
[EVIDENCE L0014] 
[EVIDENCE L0015] ## 1. R-NaD at Neural Scale: DeepNash (Perolat et al., Science 2022)
[EVIDENCE L0016] 
[EVIDENCE L0017] **Paper**: [arXiv:2206.15378](https://arxiv.org/abs/2206.15378) / [Science](https://www.science.org/doi/10.1126/science.add4679)
[EVIDENCE L0018] 
[EVIDENCE L0019] ### Architecture
[EVIDENCE L0020] 
[EVIDENCE L0021] DeepNash uses a **U-Net / Pyramid convolutional network** with residual blocks and skip connections:
[EVIDENCE L0022] 
[EVIDENCE L0023] - **Input**: 10x10x82 tensor (82 stacked frames encoding private info, public info, move history)
[EVIDENCE L0024] - **Torso**: Pyramid Module with N=2 outer blocks, M=2 inner blocks
[EVIDENCE L0025]   - Outer channels: **256**
[EVIDENCE L0026]   - Inner channels: **320**
[EVIDENCE L0027]   - Uses Conv ResBlocks (3x3 kernels, bottleneck at C//2) and Deconv ResBlocks with symmetric skip connections
[EVIDENCE L0028] - **4 Output Heads**:
[EVIDENCE L0029]   1. **Value head** (Pyramid N=0, M=0) -> scalar
[EVIDENCE L0030]   2. **Deployment policy** (Pyramid N=1, M=0) -> 10x10 distribution
[EVIDENCE L0031]   3. **Piece-selection policy** (Pyramid N=1, M=0) -> 10x10 distribution
[EVIDENCE L0032]   4. **Piece-displacement policy** (Pyramid N=1, M=0) -> 10x10 distribution
[EVIDENCE L0033] - **Parameter count**: Not explicitly stated, but based on the architecture (256/320 channel U-Net with ~14 ResBlocks across torso + 4 heads), estimated in the **low millions** range
[EVIDENCE L0034] 
[EVIDENCE L0035] ### Training Infrastructure
[EVIDENCE L0036] 
[EVIDENCE L0037] - **Hardware**: 768 TPU nodes (learners) + 256 TPU nodes (actors) = **1,024 TPU nodes total**
[EVIDENCE L0038] - **Pipeline**: Sebulba/Podracer architecture
[EVIDENCE L0039]   - Actors: C++ environment loop with OpenSpiel interfaces
[EVIDENCE L0040]   - Replay buffer: Full-game replay, variable-length trajectories
[EVIDENCE L0041]   - Learner: JAX distributed synchronous training via `pmap`
[EVIDENCE L0042] - **Total compute**: Not explicitly quantified in TPU-hours, but 1,024 TPU nodes is ~DeepMind scale
[EVIDENCE L0043] 
[EVIDENCE L0044] ### Training Hyperparameters (Table 2 of paper)
[EVIDENCE L0045] 
[EVIDENCE L0046] | Parameter | Value |
[EVIDENCE L0047] |-----------|-------|
[EVIDENCE L0048] | Optimizer | Adam (b1=0.0, b2=0.999, eps=1e-8) |
[EVIDENCE L0049] | Learning rate | 5e-5 |
[EVIDENCE L0050] | Batch size | 768 trajectories/step |
[EVIDENCE L0051] | Max training steps | **7.21 million** |
[EVIDENCE L0052] | Trajectory length | 3600 |
[EVIDENCE L0053] | eta (regularization) | **0.2** |
[EVIDENCE L0054] | Gamma averaging | 0.001 |
[EVIDENCE L0055] | Logit threshold (beta) | 2 |
[EVIDENCE L0056] | NeuRD clip | 1000 |
[EVIDENCE L0057] | Gradient clip | 1000 |
[EVIDENCE L0058] 
[EVIDENCE L0059] ### R-NaD Iteration Schedule
[EVIDENCE L0060] 
[EVIDENCE L0061] The R-NaD outer loop went through **165+ iterations** with this schedule:
[EVIDENCE L0062] - m <= 100: delta_m = 10,000 steps per iteration
[EVIDENCE L0063] - 100 < m <= 165: delta_m = 100,000 steps per iteration
[EVIDENCE L0064] - m > 165: delta_m = 35,000 steps per iteration
[EVIDENCE L0065] 
[EVIDENCE L0066] The regularization target network is updated at each iteration boundary, with a smooth alpha interpolation between old and new regularization targets.
[EVIDENCE L0067] 
[EVIDENCE L0068] ### Training Stability
[EVIDENCE L0069] 
[EVIDENCE L0070] Key stability mechanisms:
[EVIDENCE L0071] 1. **Lyapunov function**: R-NaD defines dynamics with a provably decreasing Lyapunov function
[EVIDENCE L0072] 2. **Exponential target averaging**: gamma=0.001 for soft target network updates
[EVIDENCE L0073] 3. **V-trace**: Adapted for two-player imperfect-info (off-policy correction)
[EVIDENCE L0074] 4. **NeuRD update**: Neural Replicator Dynamics for policy gradient
[EVIDENCE L0075] 5. **Post-processing**: Thresholding, discretization (n=32), repetition-avoidance heuristics at test time
[EVIDENCE L0076] 
[EVIDENCE L0077] ---
[EVIDENCE L0078] 
[EVIDENCE L0079] ## 2. DRDA (ICLR 2025) -- PURELY TABULAR
[EVIDENCE L0080] 
[EVIDENCE L0081] **Paper**: [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html)
[EVIDENCE L0082] **Authors**: Runyu Lu, Yuanheng Zhu, Dongbin Zhao
[EVIDENCE L0083] 
[EVIDENCE L0084] ### Critical Finding: NO Neural Experiments
[EVIDENCE L0085] 
[EVIDENCE L0086] DRDA is **purely tabular**. The paper explicitly uses dynamic programming for value computation:
[EVIDENCE L0087] 
[EVIDENCE L0088] > "Since the per-iteration time complexity of discrete-time DRDA (SDRDA; see Algorithm 1) is a standard O(|H|) when we use **dynamic programming** to compute the advantage value..." (Section 5, page 8)
[EVIDENCE L0089] 
[EVIDENCE L0090] > "Since the evaluation of value functions requires repeated dynamic programming in each iteration, we only run a total of 100 iterations..." (Section 5.2, page 10)
[EVIDENCE L0091] 
[EVIDENCE L0092] Algorithm 1 (page 29), step 3 explicitly says: "Compute all Pr(h|pi_m) and A^i_{pi_m}(h, a^i) for all i in N (**using dynamic programming**)."
[EVIDENCE L0093] 
[EVIDENCE L0094] ### Tested Environments (ALL small/tabular)
[EVIDENCE L0095] 
[EVIDENCE L0096] | Game | Type | Size |
[EVIDENCE L0097] |------|------|------|
[EVIDENCE L0098] | 2-action matrix game | NFG | 2x2 |
[EVIDENCE L0099] | 3-action bimatrix game | NFG | 3x3 |
[EVIDENCE L0100] | 3-action 3-player game | NFG | 3x3x3 (27 joint actions) |
[EVIDENCE L0101] | 3-player Kuhn poker (5 ranks) | EFG | Small |
[EVIDENCE L0102] | 4-player Kuhn poker (6 ranks) | EFG | Small-medium |
[EVIDENCE L0103] | Leduc poker variants | EFG | Small |
[EVIDENCE L0104] | Soccer grid-world | MG | ~5x4 grid |
[EVIDENCE L0105] | Adversarial/Competitive Tiger | POSG | H=2,3,4 |
[EVIDENCE L0106] 
[EVIDENCE L0107] ### R-NaD IS a baseline in the paper
[EVIDENCE L0108] 
[EVIDENCE L0109] DRDA compares against R-NaD in EFG experiments. The paper notes:
[EVIDENCE L0110] > "R-NaD has a multi-round learning pattern close to DRDA, but the process is much slower and suffers from an oscillation in the 4-player scenario." (page 10)
[EVIDENCE L0111] 
[EVIDENCE L0112] ### Neural Scaling Mentioned Only as Future Work
[EVIDENCE L0113] 
[EVIDENCE L0114] The paper's motivation acknowledges neural scaling:
[EVIDENCE L0115] > "Last-iterate convergence... is an **ideal property for further extension to deep reinforcement learning (DRL)**, as it is intractable to time-average function approximators like neural networks." (page 2)
[EVIDENCE L0116] 
[EVIDENCE L0117] But **no neural experiments were conducted**.
[EVIDENCE L0118] 
[EVIDENCE L0119] ---
[EVIDENCE L0120] 
[EVIDENCE L0121] ## 3. Open-Source Implementations
[EVIDENCE L0122] 
[EVIDENCE L0123] ### R-NaD Implementations
[EVIDENCE L0124] 
[EVIDENCE L0125] **a) baskuit/R-NaD** (50 stars, PyTorch)
[EVIDENCE L0126] - **URL**: https://github.com/baskuit/R-NaD
[EVIDENCE L0127] - **Permalink (rnad.py)**: https://github.com/baskuit/R-NaD/blob/0d163921bc597405040c33c89e151b18da68fa6e/learn/rnad.py
[EVIDENCE L0128] - **Permalink (net.py)**: https://github.com/baskuit/R-NaD/blob/0d163921bc597405040c33c89e151b18da68fa6e/nn/net.py
[EVIDENCE L0129] - **What it is**: R-NaD on abstract stochastic matrix trees (GPU-accelerated)
[EVIDENCE L0130] - **Neural nets**: MLP (2-layer) and ConvNet (ResBlock tower) implementations
[EVIDENCE L0131] - **Uses**: Full R-NaD loop with v-trace, NeuRD, entropy scheduling, identical hyperparams to DeepNash paper
[EVIDENCE L0132] - **Scale**: Consumer hardware -- designed for accessible experimentation, NOT game-scale
[EVIDENCE L0133] 
[EVIDENCE L0134] **b) AbhinavPeri/DeepNash** (7 stars, PyTorch)
[EVIDENCE L0135] - **URL**: https://github.com/AbhinavPeri/DeepNash
[EVIDENCE L0136] - **Permalink (network.py)**: https://github.com/AbhinavPeri/DeepNash/blob/94925a04ae547a282d91e83eb48091dac65825e9/deep_nash/network.py
[EVIDENCE L0137] - **Permalink (rnad.py)**: https://github.com/AbhinavPeri/DeepNash/blob/94925a04ae547a282d91e83eb48091dac65825e9/deep_nash/rnad.py
[EVIDENCE L0138] - **What it is**: Full DeepNash architecture recreation for Stratego (4x4 variant)
[EVIDENCE L0139] - **Neural nets**: PyramidModule U-Net with ConvResBlock/DeconvResBlock, 4 heads (deployment, selection, movement, value)
[EVIDENCE L0140] - **Scale**: 4x4 Stratego variant (reduced from 10x10)
[EVIDENCE L0141] 
[EVIDENCE L0142] **c) Other community implementations**:
[EVIDENCE L0143] - **spktrm/pokesim**: R-NaD applied to Pokemon battles (1 star)
[EVIDENCE L0144] - **JimZhouZZY/RNaD-JunQi**: R-NaD for JunQi (Chinese military chess)
[EVIDENCE L0145] - **valvarl/deepnash-torchrl**: DeepNash using TorchRL
[EVIDENCE L0146] - **nathanlct/IIG-RL-Benchmark**: R-NaD benchmark for imperfect info games
[EVIDENCE L0147] 
[EVIDENCE L0148] ### OpenSpiel Status
[EVIDENCE L0149] 
[EVIDENCE L0150] **OpenSpiel does NOT have an official R-NaD implementation.**
[EVIDENCE L0151] 
[EVIDENCE L0152] Searched exhaustively:
[EVIDENCE L0153] - `open_spiel/python/algorithms/` -- no rnad directory or file
[EVIDENCE L0154] - `open_spiel/python/jax/` -- has NFSP but no R-NaD
[EVIDENCE L0155] - GitHub code search across all google-deepmind repos -- zero R-NaD results
[EVIDENCE L0156] 
[EVIDENCE L0157] OpenSpiel has NFSP (`open_spiel/python/jax/nfsp.py`) and various CFR variants, but R-NaD was kept internal to DeepMind's DeepNash project.
[EVIDENCE L0158] 
[EVIDENCE L0159] ### DRDA Implementations
[EVIDENCE L0160] 
[EVIDENCE L0161] **No open-source implementation of DRDA exists.** GitHub search returned zero results for DRDA in the game-theory context.
[EVIDENCE L0162] 
[EVIDENCE L0163] ---
[EVIDENCE L0164] 
[EVIDENCE L0165] ## 4. R-NaD vs NFSP at Neural Scale
[EVIDENCE L0166] 
[EVIDENCE L0167] No direct published comparison exists between R-NaD and NFSP at neural scale. However:
[EVIDENCE L0168] 
[EVIDENCE L0169] | Dimension | NFSP (Heinrich & Silver, 2016) | R-NaD (DeepNash, 2022) |
[EVIDENCE L0170] |-----------|-------------------------------|------------------------|
[EVIDENCE L0171] | Largest game | Limit Texas Hold'em (~10^18 states) | Stratego (~10^535 states) |
[EVIDENCE L0172] | Architecture | DQN + supervised policy net | U-Net Pyramid + 4 heads |
[EVIDENCE L0173] | Convergence target | Average-iterate NE | Last-iterate NE |
[EVIDENCE L0174] | Compute | Single GPU | 1,024 TPU nodes |
[EVIDENCE L0175] | Open-source | YES (OpenSpiel) | NO |
[EVIDENCE L0176] | Key advantage | Simple, well-understood | No cycling, last-iterate convergence |
[EVIDENCE L0177] 
[EVIDENCE L0178] The R-NaD/DRDA family's key theoretical advantage is **last-iterate convergence** -- you don't need to average policies across training, which is impractical with neural nets. NFSP works around this with a separate supervised network tracking the average policy, but this is an approximation.
[EVIDENCE L0179] 
[EVIDENCE L0180] ---
[EVIDENCE L0181] 
[EVIDENCE L0182] ## 5. Follow-Up Work After DeepNash
[EVIDENCE L0183] 
[EVIDENCE L0184] No published work has applied R-NaD to another game at DeepNash-scale. The community implementations (baskuit, AbhinavPeri, etc.) are all smaller-scale experiments. DeepMind themselves noted R-NaD "can be directly applied to other two-player zero-sum games" but haven't published such follow-ups.
[EVIDENCE L0185] 
[EVIDENCE L0186] DRDA (ICLR 2025) is the primary theoretical follow-up, extending R-NaD's regularized dynamics to multiplayer POSGs, but stays purely tabular.
[EVIDENCE L0187] 
[EVIDENCE L0188] ---
[EVIDENCE L0189] 
[EVIDENCE L0190] ## 6. Strategic Implications for Your Proposal
[EVIDENCE L0191] 
[EVIDENCE L0192] ### The judges' critique is valid but addressable:
[EVIDENCE L0193] 
[EVIDENCE L0194] 1. **DRDA itself is unproven at neural scale** -- this is factually correct. The DRDA paper has zero neural experiments.
[EVIDENCE L0195] 
[EVIDENCE L0196] 2. **BUT R-NaD (the parent algorithm) IS proven at neural scale** -- DeepNash is the proof. 1,024 TPUs, Stratego (10^535 states), published in Science.
[EVIDENCE L0197] 
[EVIDENCE L0198] 3. **DRDA's key contribution is theoretical** -- it provides last-iterate convergence guarantees and extends to multiplayer, which R-NaD alone doesn't formally provide for the general POSG case.
[EVIDENCE L0199] 
[EVIDENCE L0200] ### Suggested response to reviewers:
[EVIDENCE L0201] 
[EVIDENCE L0202] > "While DRDA itself has only been validated in tabular settings (Lu et al., ICLR 2025), the closely related R-NaD algorithm -- which DRDA directly extends -- has been validated at unprecedented neural scale in DeepNash (Perolat et al., Science 2022), using a U-Net pyramid architecture trained on 1,024 TPU nodes for 7.21M steps on Stratego (10^535 states). DRDA's theoretical extensions (last-iterate convergence, multiplayer POSG support) are algorithmically lightweight modifications to R-NaD's reward transformation scheme, and the neural-scale infrastructure (v-trace, NeuRD updates, entropy scheduling) transfers directly. Our approach specifically leverages these proven neural-scale components while incorporating DRDA's improved convergence properties."
[EVIDENCE L0203] 
[EVIDENCE L0204] ### What you could also do:
[EVIDENCE L0205] - Reference baskuit/R-NaD as evidence the algorithm is implementable outside DeepMind
[EVIDENCE L0206] - Note that OpenSpiel lacks R-NaD, making community implementations even more valuable
[EVIDENCE L0207] - Emphasize that YOUR contribution would be one of the first neural-scale DRDA implementations -- this is a FEATURE, not a bug
```

## Artifact 17 — Training heads including oracle critic
Artifact id: `heads-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-train/src/heads.rs`
Why it matters: Code-level ground truth for how Hydra currently defines its heads, including the explicit oracle critic surface. Important for method-overlap judgments.

```rust
[CODE L0001] //! Output heads: 8 inference heads + 1 oracle critic.
[CODE L0002] 
[CODE L0003] use burn::nn::{
[CODE L0004]     Linear, LinearConfig,
[CODE L0005]     conv::{Conv1d, Conv1dConfig},
[CODE L0006] };
[CODE L0007] use burn::prelude::*;
[CODE L0008] 
[CODE L0009] #[derive(Module, Debug)]
[CODE L0010] pub struct PolicyHead<B: Backend> {
[CODE L0011]     linear: Linear<B>,
[CODE L0012] }
[CODE L0013] 
[CODE L0014] impl<B: Backend> PolicyHead<B> {
[CODE L0015]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0016]         self.linear.forward(pooled)
[CODE L0017]     }
[CODE L0018] }
[CODE L0019] 
[CODE L0020] #[derive(Module, Debug)]
[CODE L0021] pub struct ValueHead<B: Backend> {
[CODE L0022]     linear: Linear<B>,
[CODE L0023] }
[CODE L0024] 
[CODE L0025] impl<B: Backend> ValueHead<B> {
[CODE L0026]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0027]         self.linear.forward(pooled).tanh()
[CODE L0028]     }
[CODE L0029] }
[CODE L0030] 
[CODE L0031] #[derive(Module, Debug)]
[CODE L0032] pub struct ScorePdfHead<B: Backend> {
[CODE L0033]     linear: Linear<B>,
[CODE L0034] }
[CODE L0035] 
[CODE L0036] impl<B: Backend> ScorePdfHead<B> {
[CODE L0037]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0038]         self.linear.forward(pooled)
[CODE L0039]     }
[CODE L0040] }
[CODE L0041] 
[CODE L0042] #[derive(Module, Debug)]
[CODE L0043] pub struct ScoreCdfHead<B: Backend> {
[CODE L0044]     linear: Linear<B>,
[CODE L0045] }
[CODE L0046] 
[CODE L0047] impl<B: Backend> ScoreCdfHead<B> {
[CODE L0048]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0049]         self.linear.forward(pooled)
[CODE L0050]     }
[CODE L0051] }
[CODE L0052] 
[CODE L0053] #[derive(Module, Debug)]
[CODE L0054] pub struct OppTenpaiHead<B: Backend> {
[CODE L0055]     linear: Linear<B>,
[CODE L0056] }
[CODE L0057] 
[CODE L0058] impl<B: Backend> OppTenpaiHead<B> {
[CODE L0059]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0060]         self.linear.forward(pooled)
[CODE L0061]     }
[CODE L0062] }
[CODE L0063] 
[CODE L0064] #[derive(Module, Debug)]
[CODE L0065] pub struct GrpHead<B: Backend> {
[CODE L0066]     linear: Linear<B>,
[CODE L0067] }
[CODE L0068] 
[CODE L0069] impl<B: Backend> GrpHead<B> {
[CODE L0070]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0071]         self.linear.forward(pooled)
[CODE L0072]     }
[CODE L0073] }
[CODE L0074] 
[CODE L0075] #[derive(Module, Debug)]
[CODE L0076] pub struct OppNextDiscardHead<B: Backend> {
[CODE L0077]     conv: Conv1d<B>,
[CODE L0078] }
[CODE L0079] 
[CODE L0080] impl<B: Backend> OppNextDiscardHead<B> {
[CODE L0081]     pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
[CODE L0082]         self.conv.forward(spatial)
[CODE L0083]     }
[CODE L0084] }
[CODE L0085] 
[CODE L0086] #[derive(Module, Debug)]
[CODE L0087] pub struct DangerHead<B: Backend> {
[CODE L0088]     conv: Conv1d<B>,
[CODE L0089] }
[CODE L0090] 
[CODE L0091] impl<B: Backend> DangerHead<B> {
[CODE L0092]     pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
[CODE L0093]         self.conv.forward(spatial)
[CODE L0094]     }
[CODE L0095] }
[CODE L0096] 
[CODE L0097] #[derive(Module, Debug)]
[CODE L0098] pub struct OracleCriticHead<B: Backend> {
[CODE L0099]     linear: Linear<B>,
[CODE L0100] }
[CODE L0101] 
[CODE L0102] impl<B: Backend> OracleCriticHead<B> {
[CODE L0103]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0104]         self.linear.forward(pooled)
[CODE L0105]     }
[CODE L0106] }
[CODE L0107] 
[CODE L0108] #[derive(Module, Debug)]
[CODE L0109] pub struct BeliefFieldHead<B: Backend> {
[CODE L0110]     conv: Conv1d<B>,
[CODE L0111] }
[CODE L0112] 
[CODE L0113] impl<B: Backend> BeliefFieldHead<B> {
[CODE L0114]     pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
[CODE L0115]         self.conv.forward(spatial)
[CODE L0116]     }
[CODE L0117] }
[CODE L0118] 
[CODE L0119] #[derive(Module, Debug)]
[CODE L0120] pub struct MixtureWeightHead<B: Backend> {
[CODE L0121]     linear: Linear<B>,
[CODE L0122] }
[CODE L0123] 
[CODE L0124] impl<B: Backend> MixtureWeightHead<B> {
[CODE L0125]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0126]         self.linear.forward(pooled)
[CODE L0127]     }
[CODE L0128] }
[CODE L0129] 
[CODE L0130] #[derive(Module, Debug)]
[CODE L0131] pub struct OpponentHandTypeHead<B: Backend> {
[CODE L0132]     linear: Linear<B>,
[CODE L0133] }
[CODE L0134] 
[CODE L0135] impl<B: Backend> OpponentHandTypeHead<B> {
[CODE L0136]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0137]         self.linear.forward(pooled)
[CODE L0138]     }
[CODE L0139] }
[CODE L0140] 
[CODE L0141] #[derive(Module, Debug)]
[CODE L0142] pub struct DeltaQHead<B: Backend> {
[CODE L0143]     linear: Linear<B>,
[CODE L0144] }
[CODE L0145] 
[CODE L0146] impl<B: Backend> DeltaQHead<B> {
[CODE L0147]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0148]         self.linear.forward(pooled)
[CODE L0149]     }
[CODE L0150] }
[CODE L0151] 
[CODE L0152] #[derive(Module, Debug)]
[CODE L0153] pub struct SafetyResidualHead<B: Backend> {
[CODE L0154]     linear: Linear<B>,
[CODE L0155] }
[CODE L0156] 
[CODE L0157] impl<B: Backend> SafetyResidualHead<B> {
[CODE L0158]     pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0159]         self.linear.forward(pooled)
[CODE L0160]     }
[CODE L0161] }
[CODE L0162] 
[CODE L0163] impl HeadsConfig {
[CODE L0164]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0165]         if self.hidden_channels == 0 {
[CODE L0166]             return Err("hidden_channels must be > 0");
[CODE L0167]         }
[CODE L0168]         if self.action_space == 0 {
[CODE L0169]             return Err("action_space must be > 0");
[CODE L0170]         }
[CODE L0171]         if self.num_belief_components == 0 {
[CODE L0172]             return Err("num_belief_components must be > 0");
[CODE L0173]         }
[CODE L0174]         if self.opponent_hand_type_classes == 0 {
[CODE L0175]             return Err("opponent_hand_type_classes must be > 0");
[CODE L0176]         }
[CODE L0177]         Ok(())
[CODE L0178]     }
[CODE L0179] }
[CODE L0180] 
[CODE L0181] #[derive(Config, Debug)]
[CODE L0182] pub struct HeadsConfig {
[CODE L0183]     #[config(default = "256")]
[CODE L0184]     pub hidden_channels: usize,
[CODE L0185]     #[config(default = "46")]
[CODE L0186]     pub action_space: usize,
[CODE L0187]     #[config(default = "64")]
[CODE L0188]     pub score_bins: usize,
[CODE L0189]     #[config(default = "3")]
[CODE L0190]     pub num_opponents: usize,
[CODE L0191]     #[config(default = "24")]
[CODE L0192]     pub grp_classes: usize,
[CODE L0193]     #[config(default = "4")]
[CODE L0194]     pub num_belief_components: usize,
[CODE L0195]     #[config(default = "8")]
[CODE L0196]     pub opponent_hand_type_classes: usize,
[CODE L0197] }
[CODE L0198] 
[CODE L0199] impl HeadsConfig {
[CODE L0200]     pub fn init_policy<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
[CODE L0201]         PolicyHead {
[CODE L0202]             linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
[CODE L0203]         }
[CODE L0204]     }
[CODE L0205] 
[CODE L0206]     pub fn init_value<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
[CODE L0207]         ValueHead {
[CODE L0208]             linear: LinearConfig::new(self.hidden_channels, 1).init(device),
[CODE L0209]         }
[CODE L0210]     }
[CODE L0211] 
[CODE L0212]     pub fn init_score_pdf<B: Backend>(&self, device: &B::Device) -> ScorePdfHead<B> {
[CODE L0213]         ScorePdfHead {
[CODE L0214]             linear: LinearConfig::new(self.hidden_channels, self.score_bins).init(device),
[CODE L0215]         }
[CODE L0216]     }
[CODE L0217] 
[CODE L0218]     pub fn init_score_cdf<B: Backend>(&self, device: &B::Device) -> ScoreCdfHead<B> {
[CODE L0219]         ScoreCdfHead {
[CODE L0220]             linear: LinearConfig::new(self.hidden_channels, self.score_bins).init(device),
[CODE L0221]         }
[CODE L0222]     }
[CODE L0223] 
[CODE L0224]     pub fn init_opp_tenpai<B: Backend>(&self, device: &B::Device) -> OppTenpaiHead<B> {
[CODE L0225]         OppTenpaiHead {
[CODE L0226]             linear: LinearConfig::new(self.hidden_channels, self.num_opponents).init(device),
[CODE L0227]         }
[CODE L0228]     }
[CODE L0229] 
[CODE L0230]     pub fn init_grp<B: Backend>(&self, device: &B::Device) -> GrpHead<B> {
[CODE L0231]         GrpHead {
[CODE L0232]             linear: LinearConfig::new(self.hidden_channels, self.grp_classes).init(device),
[CODE L0233]         }
[CODE L0234]     }
[CODE L0235] 
[CODE L0236]     pub fn init_opp_next_discard<B: Backend>(&self, device: &B::Device) -> OppNextDiscardHead<B> {
[CODE L0237]         OppNextDiscardHead {
[CODE L0238]             conv: Conv1dConfig::new(self.hidden_channels, self.num_opponents, 1).init(device),
[CODE L0239]         }
[CODE L0240]     }
[CODE L0241] 
[CODE L0242]     pub fn init_danger<B: Backend>(&self, device: &B::Device) -> DangerHead<B> {
[CODE L0243]         DangerHead {
[CODE L0244]             conv: Conv1dConfig::new(self.hidden_channels, self.num_opponents, 1).init(device),
[CODE L0245]         }
[CODE L0246]     }
[CODE L0247] 
[CODE L0248]     pub fn init_oracle_critic<B: Backend>(&self, device: &B::Device) -> OracleCriticHead<B> {
[CODE L0249]         OracleCriticHead {
[CODE L0250]             linear: LinearConfig::new(self.hidden_channels, 4).init(device),
[CODE L0251]         }
[CODE L0252]     }
[CODE L0253] 
[CODE L0254]     pub fn init_belief_field<B: Backend>(&self, device: &B::Device) -> BeliefFieldHead<B> {
[CODE L0255]         BeliefFieldHead {
[CODE L0256]             conv: Conv1dConfig::new(self.hidden_channels, self.num_belief_components * 4, 1)
[CODE L0257]                 .init(device),
[CODE L0258]         }
[CODE L0259]     }
[CODE L0260] 
[CODE L0261]     pub fn init_mixture_weight<B: Backend>(&self, device: &B::Device) -> MixtureWeightHead<B> {
[CODE L0262]         MixtureWeightHead {
[CODE L0263]             linear: LinearConfig::new(self.hidden_channels, self.num_belief_components)
[CODE L0264]                 .init(device),
[CODE L0265]         }
[CODE L0266]     }
[CODE L0267] 
[CODE L0268]     pub fn init_opponent_hand_type<B: Backend>(
[CODE L0269]         &self,
[CODE L0270]         device: &B::Device,
[CODE L0271]     ) -> OpponentHandTypeHead<B> {
[CODE L0272]         OpponentHandTypeHead {
[CODE L0273]             linear: LinearConfig::new(
[CODE L0274]                 self.hidden_channels,
[CODE L0275]                 self.num_opponents * self.opponent_hand_type_classes,
[CODE L0276]             )
[CODE L0277]             .init(device),
[CODE L0278]         }
[CODE L0279]     }
[CODE L0280] 
[CODE L0281]     pub fn init_delta_q<B: Backend>(&self, device: &B::Device) -> DeltaQHead<B> {
[CODE L0282]         DeltaQHead {
[CODE L0283]             linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
[CODE L0284]         }
[CODE L0285]     }
[CODE L0286] 
[CODE L0287]     pub fn init_safety_residual<B: Backend>(&self, device: &B::Device) -> SafetyResidualHead<B> {
[CODE L0288]         SafetyResidualHead {
[CODE L0289]             linear: LinearConfig::new(self.hidden_channels, self.action_space).init(device),
[CODE L0290]         }
[CODE L0291]     }
[CODE L0292] }
[CODE L0293] 
[CODE L0294] #[cfg(test)]
[CODE L0295] mod tests {
[CODE L0296]     use super::*;
[CODE L0297]     use burn::backend::NdArray;
[CODE L0298] 
[CODE L0299]     type B = NdArray<f32>;
[CODE L0300] 
[CODE L0301]     fn cfg() -> HeadsConfig {
[CODE L0302]         HeadsConfig::new()
[CODE L0303]     }
[CODE L0304] 
[CODE L0305]     #[test]
[CODE L0306]     fn policy_head_shape() {
[CODE L0307]         let device = Default::default();
[CODE L0308]         let head = cfg().init_policy::<B>(&device);
[CODE L0309]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0310]         assert_eq!(head.forward(x).dims(), [4, 46]);
[CODE L0311]     }
[CODE L0312] 
[CODE L0313]     #[test]
[CODE L0314]     fn value_head_shape_and_range() {
[CODE L0315]         let device = Default::default();
[CODE L0316]         let head = cfg().init_value::<B>(&device);
[CODE L0317]         let x = Tensor::<B, 2>::random(
[CODE L0318]             [4, 256],
[CODE L0319]             burn::tensor::Distribution::Normal(0.0, 1.0),
[CODE L0320]             &device,
[CODE L0321]         );
[CODE L0322]         let out = head.forward(x);
[CODE L0323]         assert_eq!(out.dims(), [4, 1]);
[CODE L0324]         let data = out.to_data();
[CODE L0325]         for &v in data.as_slice::<f32>().expect("f32 slice") {
[CODE L0326]             assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
[CODE L0327]         }
[CODE L0328]     }
[CODE L0329] 
[CODE L0330]     #[test]
[CODE L0331]     fn score_pdf_head_shape() {
[CODE L0332]         let device = Default::default();
[CODE L0333]         let head = cfg().init_score_pdf::<B>(&device);
[CODE L0334]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0335]         assert_eq!(head.forward(x).dims(), [4, 64]);
[CODE L0336]     }
[CODE L0337] 
[CODE L0338]     #[test]
[CODE L0339]     fn score_cdf_head_shape() {
[CODE L0340]         let device = Default::default();
[CODE L0341]         let head = cfg().init_score_cdf::<B>(&device);
[CODE L0342]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0343]         assert_eq!(head.forward(x).dims(), [4, 64]);
[CODE L0344]     }
[CODE L0345] 
[CODE L0346]     #[test]
[CODE L0347]     fn opp_tenpai_head_shape() {
[CODE L0348]         let device = Default::default();
[CODE L0349]         let head = cfg().init_opp_tenpai::<B>(&device);
[CODE L0350]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0351]         assert_eq!(head.forward(x).dims(), [4, 3]);
[CODE L0352]     }
[CODE L0353] 
[CODE L0354]     #[test]
[CODE L0355]     fn grp_head_shape() {
[CODE L0356]         let device = Default::default();
[CODE L0357]         let head = cfg().init_grp::<B>(&device);
[CODE L0358]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0359]         assert_eq!(head.forward(x).dims(), [4, 24]);
[CODE L0360]     }
[CODE L0361] 
[CODE L0362]     #[test]
[CODE L0363]     fn opp_next_discard_head_shape() {
[CODE L0364]         let device = Default::default();
[CODE L0365]         let head = cfg().init_opp_next_discard::<B>(&device);
[CODE L0366]         let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
[CODE L0367]         assert_eq!(head.forward(x).dims(), [4, 3, 34]);
[CODE L0368]     }
[CODE L0369] 
[CODE L0370]     #[test]
[CODE L0371]     fn danger_head_shape() {
[CODE L0372]         let device = Default::default();
[CODE L0373]         let head = cfg().init_danger::<B>(&device);
[CODE L0374]         let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
[CODE L0375]         assert_eq!(head.forward(x).dims(), [4, 3, 34]);
[CODE L0376]     }
[CODE L0377] 
[CODE L0378]     #[test]
[CODE L0379]     fn oracle_critic_head_shape() {
[CODE L0380]         let device = Default::default();
[CODE L0381]         let head = cfg().init_oracle_critic::<B>(&device);
[CODE L0382]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0383]         assert_eq!(head.forward(x).dims(), [4, 4]);
[CODE L0384]     }
[CODE L0385] 
[CODE L0386]     #[test]
[CODE L0387]     fn belief_field_head_shape() {
[CODE L0388]         let device = Default::default();
[CODE L0389]         let head = cfg().init_belief_field::<B>(&device);
[CODE L0390]         let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
[CODE L0391]         assert_eq!(head.forward(x).dims(), [4, 16, 34]);
[CODE L0392]     }
[CODE L0393] 
[CODE L0394]     #[test]
[CODE L0395]     fn mixture_weight_head_shape() {
[CODE L0396]         let device = Default::default();
[CODE L0397]         let head = cfg().init_mixture_weight::<B>(&device);
[CODE L0398]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0399]         assert_eq!(head.forward(x).dims(), [4, 4]);
[CODE L0400]     }
[CODE L0401] 
[CODE L0402]     #[test]
[CODE L0403]     fn opponent_hand_type_head_shape() {
[CODE L0404]         let device = Default::default();
[CODE L0405]         let head = cfg().init_opponent_hand_type::<B>(&device);
[CODE L0406]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0407]         assert_eq!(head.forward(x).dims(), [4, 24]);
[CODE L0408]     }
[CODE L0409] 
[CODE L0410]     #[test]
[CODE L0411]     fn delta_q_head_shape() {
[CODE L0412]         let device = Default::default();
[CODE L0413]         let head = cfg().init_delta_q::<B>(&device);
[CODE L0414]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0415]         assert_eq!(head.forward(x).dims(), [4, 46]);
[CODE L0416]     }
[CODE L0417] 
[CODE L0418]     #[test]
[CODE L0419]     fn safety_residual_head_shape() {
[CODE L0420]         let device = Default::default();
[CODE L0421]         let head = cfg().init_safety_residual::<B>(&device);
[CODE L0422]         let x = Tensor::<B, 2>::zeros([4, 256], &device);
[CODE L0423]         assert_eq!(head.forward(x).dims(), [4, 46]);
[CODE L0424]     }
[CODE L0425] }
```

## Artifact 18 — Model wiring and head outputs
Artifact id: `model-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-train/src/model.rs`
Why it matters: Core code evidence for how the model wires value, policy, delta_q, and oracle-related outputs together. The research agent can use this to judge whether DCRL is only adjacent framing or structurally close to current Hydra model reality.

```rust
[CODE L0001] //! Full HydraModel combining backbone and all output heads.
[CODE L0002] 
[CODE L0003] use burn::prelude::*;
[CODE L0004] use hydra_core::action::HYDRA_ACTION_SPACE;
[CODE L0005] use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};
[CODE L0006] 
[CODE L0007] use crate::backbone::{SEResNet, SEResNetConfig};
[CODE L0008] use crate::config::INPUT_CHANNELS;
[CODE L0009] use crate::heads::*;
[CODE L0010] 
[CODE L0011] pub struct HydraOutput<B: Backend> {
[CODE L0012]     pub policy_logits: Tensor<B, 2>,
[CODE L0013]     pub value: Tensor<B, 2>,
[CODE L0014]     pub score_pdf: Tensor<B, 2>,
[CODE L0015]     pub score_cdf: Tensor<B, 2>,
[CODE L0016]     pub opp_tenpai: Tensor<B, 2>,
[CODE L0017]     pub grp: Tensor<B, 2>,
[CODE L0018]     pub opp_next_discard: Tensor<B, 3>,
[CODE L0019]     pub danger: Tensor<B, 3>,
[CODE L0020]     pub oracle_critic: Tensor<B, 2>,
[CODE L0021]     pub belief_fields: Tensor<B, 3>,
[CODE L0022]     pub mixture_weight_logits: Tensor<B, 2>,
[CODE L0023]     pub opponent_hand_type: Tensor<B, 2>,
[CODE L0024]     pub delta_q: Tensor<B, 2>,
[CODE L0025]     pub safety_residual: Tensor<B, 2>,
[CODE L0026] }
[CODE L0027] 
[CODE L0028] pub type ActorNet<B> = HydraModel<B>;
[CODE L0029] pub type LearnerNet<B> = HydraModel<B>;
[CODE L0030] 
[CODE L0031] impl<B: Backend> HydraOutput<B> {
[CODE L0032]     pub fn masked_policy(&self, legal_mask: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0033]         let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
[CODE L0034]         self.policy_logits.clone() + neg_inf
[CODE L0035]     }
[CODE L0036] 
[CODE L0037]     pub fn policy_logits_cpu(&self) -> Option<Vec<f32>> {
[CODE L0038]         self.policy_logits
[CODE L0039]             .to_data()
[CODE L0040]             .convert::<f32>()
[CODE L0041]             .as_slice::<f32>()
[CODE L0042]             .ok()
[CODE L0043]             .map(|s| s.to_vec())
[CODE L0044]     }
[CODE L0045] 
[CODE L0046]     pub fn value_scalar(&self) -> Option<f32> {
[CODE L0047]         self.value
[CODE L0048]             .to_data()
[CODE L0049]             .convert::<f32>()
[CODE L0050]             .as_slice::<f32>()
[CODE L0051]             .ok()
[CODE L0052]             .and_then(|s| s.first().copied())
[CODE L0053]     }
[CODE L0054] 
[CODE L0055]     pub fn is_finite(&self) -> bool {
[CODE L0056]         let check2 = |t: &Tensor<B, 2>| -> bool {
[CODE L0057]             if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
[CODE L0058]                 s.iter().all(|v| v.is_finite())
[CODE L0059]             } else {
[CODE L0060]                 false
[CODE L0061]             }
[CODE L0062]         };
[CODE L0063]         let check3 = |t: &Tensor<B, 3>| -> bool {
[CODE L0064]             if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
[CODE L0065]                 s.iter().all(|v| v.is_finite())
[CODE L0066]             } else {
[CODE L0067]                 false
[CODE L0068]             }
[CODE L0069]         };
[CODE L0070]         check2(&self.policy_logits)
[CODE L0071]             && check2(&self.value)
[CODE L0072]             && check2(&self.score_pdf)
[CODE L0073]             && check2(&self.score_cdf)
[CODE L0074]             && check2(&self.opp_tenpai)
[CODE L0075]             && check2(&self.grp)
[CODE L0076]             && check2(&self.oracle_critic)
[CODE L0077]             && check3(&self.opp_next_discard)
[CODE L0078]             && check3(&self.danger)
[CODE L0079]             && check3(&self.belief_fields)
[CODE L0080]             && check2(&self.mixture_weight_logits)
[CODE L0081]             && check2(&self.opponent_hand_type)
[CODE L0082]             && check2(&self.delta_q)
[CODE L0083]             && check2(&self.safety_residual)
[CODE L0084]     }
[CODE L0085] }
[CODE L0086] 
[CODE L0087] fn zero_linear_head<B: Backend>(batch: usize, width: usize, device: &B::Device) -> Tensor<B, 2> {
[CODE L0088]     Tensor::<B, 2>::zeros([batch, width], device)
[CODE L0089] }
[CODE L0090] 
[CODE L0091] fn zero_spatial_head<B: Backend>(
[CODE L0092]     batch: usize,
[CODE L0093]     channels: usize,
[CODE L0094]     width: usize,
[CODE L0095]     device: &B::Device,
[CODE L0096] ) -> Tensor<B, 3> {
[CODE L0097]     Tensor::<B, 3>::zeros([batch, channels, width], device)
[CODE L0098] }
[CODE L0099] 
[CODE L0100] #[derive(Module, Debug)]
[CODE L0101] pub struct HydraModel<B: Backend> {
[CODE L0102]     backbone: SEResNet<B>,
[CODE L0103]     policy: PolicyHead<B>,
[CODE L0104]     value: ValueHead<B>,
[CODE L0105]     score_pdf: ScorePdfHead<B>,
[CODE L0106]     score_cdf: ScoreCdfHead<B>,
[CODE L0107]     opp_tenpai: OppTenpaiHead<B>,
[CODE L0108]     grp: GrpHead<B>,
[CODE L0109]     opp_next_discard: OppNextDiscardHead<B>,
[CODE L0110]     danger: DangerHead<B>,
[CODE L0111]     oracle_critic: OracleCriticHead<B>,
[CODE L0112]     belief_field: BeliefFieldHead<B>,
[CODE L0113]     mixture_weight: MixtureWeightHead<B>,
[CODE L0114]     opponent_hand_type: OpponentHandTypeHead<B>,
[CODE L0115]     delta_q: DeltaQHead<B>,
[CODE L0116]     safety_residual: SafetyResidualHead<B>,
[CODE L0117] }
[CODE L0118] 
[CODE L0119] #[derive(Config, Debug)]
[CODE L0120] pub struct HydraModelConfig {
[CODE L0121]     pub num_blocks: usize,
[CODE L0122]     #[config(default = "192")]
[CODE L0123]     pub input_channels: usize,
[CODE L0124]     #[config(default = "256")]
[CODE L0125]     pub hidden_channels: usize,
[CODE L0126]     #[config(default = "32")]
[CODE L0127]     pub num_groups: usize,
[CODE L0128]     #[config(default = "64")]
[CODE L0129]     pub se_bottleneck: usize,
[CODE L0130]     #[config(default = "46")]
[CODE L0131]     pub action_space: usize,
[CODE L0132]     #[config(default = "64")]
[CODE L0133]     pub score_bins: usize,
[CODE L0134]     #[config(default = "3")]
[CODE L0135]     pub num_opponents: usize,
[CODE L0136]     #[config(default = "24")]
[CODE L0137]     pub grp_classes: usize,
[CODE L0138]     #[config(default = "4")]
[CODE L0139]     pub num_belief_components: usize,
[CODE L0140]     #[config(default = "8")]
[CODE L0141]     pub opponent_hand_type_classes: usize,
[CODE L0142] }
[CODE L0143] 
[CODE L0144] impl HydraModelConfig {
[CODE L0145]     pub fn summary(&self) -> String {
[CODE L0146]         let kind = if self.num_blocks <= 12 {
[CODE L0147]             "actor"
[CODE L0148]         } else {
[CODE L0149]             "learner"
[CODE L0150]         };
[CODE L0151]         format!(
[CODE L0152]             "{}(blocks={}, ch={})",
[CODE L0153]             kind, self.num_blocks, self.hidden_channels
[CODE L0154]         )
[CODE L0155]     }
[CODE L0156] 
[CODE L0157]     pub fn is_actor(&self) -> bool {
[CODE L0158]         self.num_blocks == 12
[CODE L0159]     }
[CODE L0160]     pub fn is_learner(&self) -> bool {
[CODE L0161]         self.num_blocks == 24
[CODE L0162]     }
[CODE L0163] 
[CODE L0164]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0165]         if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
[CODE L0166]             return Err("hidden_channels must be divisible by num_groups");
[CODE L0167]         }
[CODE L0168]         if self.num_blocks == 0 {
[CODE L0169]             return Err("num_blocks must be > 0");
[CODE L0170]         }
[CODE L0171]         if self.se_bottleneck == 0 {
[CODE L0172]             return Err("se_bottleneck must be > 0");
[CODE L0173]         }
[CODE L0174]         if self.num_belief_components == 0 {
[CODE L0175]             return Err("num_belief_components must be > 0");
[CODE L0176]         }
[CODE L0177]         if self.opponent_hand_type_classes == 0 {
[CODE L0178]             return Err("opponent_hand_type_classes must be > 0");
[CODE L0179]         }
[CODE L0180]         Ok(())
[CODE L0181]     }
[CODE L0182] 
[CODE L0183]     pub fn actor() -> Self {
[CODE L0184]         Self::new(12).with_input_channels(INPUT_CHANNELS)
[CODE L0185]     }
[CODE L0186] 
[CODE L0187]     pub fn estimated_params(&self) -> usize {
[CODE L0188]         let h = self.hidden_channels;
[CODE L0189]         let se_b = self.se_bottleneck;
[CODE L0190]         let input_conv = self.input_channels * h * 3 + h;
[CODE L0191]         let gn = h * 2;
[CODE L0192]         let block = (h * h * 3 + h) * 2 + gn * 2 + (h * se_b + se_b) + (se_b * h + h);
[CODE L0193]         let backbone = input_conv + gn + block * self.num_blocks + gn;
[CODE L0194]         let policy = h * self.action_space + self.action_space;
[CODE L0195]         let value = h + 1;
[CODE L0196]         let score = (h * self.score_bins + self.score_bins) * 2;
[CODE L0197]         let tenpai = h * self.num_opponents + self.num_opponents;
[CODE L0198]         let grp = h * self.grp_classes + self.grp_classes;
[CODE L0199]         let opp_next = h * self.num_opponents + self.num_opponents;
[CODE L0200]         let danger = h * self.num_opponents + self.num_opponents;
[CODE L0201]         let oracle = h * 4 + 4;
[CODE L0202]         let belief_field = h * (self.num_belief_components * 4) + (self.num_belief_components * 4);
[CODE L0203]         let mixture_weight = h * self.num_belief_components + self.num_belief_components;
[CODE L0204]         let opponent_hand_type = h * (self.num_opponents * self.opponent_hand_type_classes)
[CODE L0205]             + (self.num_opponents * self.opponent_hand_type_classes);
[CODE L0206]         let delta_q = h * self.action_space + self.action_space;
[CODE L0207]         let safety_residual = h * self.action_space + self.action_space;
[CODE L0208]         backbone
[CODE L0209]             + policy
[CODE L0210]             + value
[CODE L0211]             + score
[CODE L0212]             + tenpai
[CODE L0213]             + grp
[CODE L0214]             + opp_next
[CODE L0215]             + danger
[CODE L0216]             + oracle
[CODE L0217]             + belief_field
[CODE L0218]             + mixture_weight
[CODE L0219]             + opponent_hand_type
[CODE L0220]             + delta_q
[CODE L0221]             + safety_residual
[CODE L0222]     }
[CODE L0223] 
[CODE L0224]     pub fn learner() -> Self {
[CODE L0225]         Self::new(24).with_input_channels(INPUT_CHANNELS)
[CODE L0226]     }
[CODE L0227] 
[CODE L0228]     pub fn init<B: Backend>(&self, device: &B::Device) -> HydraModel<B> {
[CODE L0229]         let backbone_cfg = SEResNetConfig::new(
[CODE L0230]             self.num_blocks,
[CODE L0231]             self.input_channels,
[CODE L0232]             self.hidden_channels,
[CODE L0233]             self.num_groups,
[CODE L0234]             self.se_bottleneck,
[CODE L0235]         );
[CODE L0236]         let heads_cfg = HeadsConfig::new()
[CODE L0237]             .with_hidden_channels(self.hidden_channels)
[CODE L0238]             .with_action_space(self.action_space)
[CODE L0239]             .with_score_bins(self.score_bins)
[CODE L0240]             .with_num_opponents(self.num_opponents)
[CODE L0241]             .with_grp_classes(self.grp_classes)
[CODE L0242]             .with_num_belief_components(self.num_belief_components)
[CODE L0243]             .with_opponent_hand_type_classes(self.opponent_hand_type_classes);
[CODE L0244]         HydraModel {
[CODE L0245]             backbone: backbone_cfg.init(device),
[CODE L0246]             policy: heads_cfg.init_policy(device),
[CODE L0247]             value: heads_cfg.init_value(device),
[CODE L0248]             score_pdf: heads_cfg.init_score_pdf(device),
[CODE L0249]             score_cdf: heads_cfg.init_score_cdf(device),
[CODE L0250]             opp_tenpai: heads_cfg.init_opp_tenpai(device),
[CODE L0251]             grp: heads_cfg.init_grp(device),
[CODE L0252]             opp_next_discard: heads_cfg.init_opp_next_discard(device),
[CODE L0253]             danger: heads_cfg.init_danger(device),
[CODE L0254]             oracle_critic: heads_cfg.init_oracle_critic(device),
[CODE L0255]             belief_field: heads_cfg.init_belief_field(device),
[CODE L0256]             mixture_weight: heads_cfg.init_mixture_weight(device),
[CODE L0257]             opponent_hand_type: heads_cfg.init_opponent_hand_type(device),
[CODE L0258]             delta_q: heads_cfg.init_delta_q(device),
[CODE L0259]             safety_residual: heads_cfg.init_safety_residual(device),
[CODE L0260]         }
[CODE L0261]     }
[CODE L0262] }
[CODE L0263] 
[CODE L0264] impl<B: Backend> HydraModel<B> {
[CODE L0265]     pub fn policy_logits_for(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
[CODE L0266]         let (_, pooled) = self.backbone.forward(x);
[CODE L0267]         self.policy.forward(pooled)
[CODE L0268]     }
[CODE L0269] 
[CODE L0270]     /// Runs a single observation through the full model and returns policy
[CODE L0271]     /// logits and value scalar on the CPU.
[CODE L0272]     ///
[CODE L0273]     /// This is the adapter used by the live ExIt producer during self-play.
[CODE L0274]     /// It performs a single-sample forward pass, extracts the policy logits
[CODE L0275]     /// as a fixed-size array and the value head output as a scalar.
[CODE L0276]     ///
[CODE L0277]     /// # Panics
[CODE L0278]     ///
[CODE L0279]     /// Panics if the forward pass produces non-extractable tensor data.
[CODE L0280]     pub fn policy_value_cpu(
[CODE L0281]         &self,
[CODE L0282]         obs: &[f32; OBS_SIZE],
[CODE L0283]         device: &B::Device,
[CODE L0284]     ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
[CODE L0285]         let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
[CODE L0286]             1,
[CODE L0287]             NUM_CHANNELS,
[CODE L0288]             NUM_TILES,
[CODE L0289]         ]);
[CODE L0290]         let (policy_logits, value) = self.forward_policy_value(input);
[CODE L0291]         let logits_vec = policy_logits
[CODE L0292]             .to_data()
[CODE L0293]             .convert::<f32>()
[CODE L0294]             .as_slice::<f32>()
[CODE L0295]             .expect("policy logits extraction failed")
[CODE L0296]             .to_vec();
[CODE L0297]         let logits: [f32; HYDRA_ACTION_SPACE] = logits_vec
[CODE L0298]             .try_into()
[CODE L0299]             .expect("policy logits length mismatch");
[CODE L0300]         let value_scalar = value
[CODE L0301]             .to_data()
[CODE L0302]             .convert::<f32>()
[CODE L0303]             .as_slice::<f32>()
[CODE L0304]             .expect("value extraction failed")[0];
[CODE L0305]         (logits, value_scalar)
[CODE L0306]     }
[CODE L0307] 
[CODE L0308]     pub fn policy_cpu(
[CODE L0309]         &self,
[CODE L0310]         obs: &[f32; OBS_SIZE],
[CODE L0311]         device: &B::Device,
[CODE L0312]     ) -> [f32; HYDRA_ACTION_SPACE] {
[CODE L0313]         let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
[CODE L0314]             1,
[CODE L0315]             NUM_CHANNELS,
[CODE L0316]             NUM_TILES,
[CODE L0317]         ]);
[CODE L0318]         let (policy_logits, _) = self.forward_policy_value(input);
[CODE L0319]         let logits_data = policy_logits.to_data().convert::<f32>();
[CODE L0320]         let logits_slice = logits_data
[CODE L0321]             .as_slice::<f32>()
[CODE L0322]             .expect("policy logits extraction failed");
[CODE L0323]         let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0324]         logits.copy_from_slice(&logits_slice[..HYDRA_ACTION_SPACE]);
[CODE L0325]         logits
[CODE L0326]     }
[CODE L0327] 
[CODE L0328]     pub fn value_cpu(&self, obs: &[f32; OBS_SIZE], device: &B::Device) -> f32 {
[CODE L0329]         let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
[CODE L0330]             1,
[CODE L0331]             NUM_CHANNELS,
[CODE L0332]             NUM_TILES,
[CODE L0333]         ]);
[CODE L0334]         let value = self.forward_value(input);
[CODE L0335]         value
[CODE L0336]             .to_data()
[CODE L0337]             .convert::<f32>()
[CODE L0338]             .as_slice::<f32>()
[CODE L0339]             .expect("value extraction failed")[0]
[CODE L0340]     }
[CODE L0341] 
[CODE L0342]     pub fn policy_and_value_cpu(
[CODE L0343]         &self,
[CODE L0344]         obs: &[f32; OBS_SIZE],
[CODE L0345]         device: &B::Device,
[CODE L0346]     ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
[CODE L0347]         (self.policy_cpu(obs, device), self.value_cpu(obs, device))
[CODE L0348]     }
[CODE L0349] 
[CODE L0350]     /// Batch inference using a caller-provided flat buffer to avoid
[CODE L0351]     /// per-call allocation. The buffer is cleared and reused each call.
[CODE L0352]     pub fn fill_batch_policy_value_cpu(
[CODE L0353]         &self,
[CODE L0354]         observations: &[[f32; OBS_SIZE]],
[CODE L0355]         device: &B::Device,
[CODE L0356]         flat_buf: &mut Vec<f32>,
[CODE L0357]         outputs: &mut Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
[CODE L0358]     ) {
[CODE L0359]         if observations.is_empty() {
[CODE L0360]             outputs.clear();
[CODE L0361]             return;
[CODE L0362]         }
[CODE L0363]         let n = observations.len();
[CODE L0364]         flat_buf.clear();
[CODE L0365]         flat_buf.reserve(n * OBS_SIZE);
[CODE L0366]         for obs in observations {
[CODE L0367]             flat_buf.extend_from_slice(obs);
[CODE L0368]         }
[CODE L0369]         let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
[CODE L0370]             n as i32,
[CODE L0371]             NUM_CHANNELS as i32,
[CODE L0372]             NUM_TILES as i32,
[CODE L0373]         ]);
[CODE L0374]         let (policy_logits, value) = self.forward_policy_value(input);
[CODE L0375]         let logits_data = policy_logits.to_data().convert::<f32>();
[CODE L0376]         let logits_flat = logits_data
[CODE L0377]             .as_slice::<f32>()
[CODE L0378]             .expect("batch policy logits extraction failed");
[CODE L0379]         let values_data = value.to_data().convert::<f32>();
[CODE L0380]         let values_flat = values_data
[CODE L0381]             .as_slice::<f32>()
[CODE L0382]             .expect("batch value extraction failed");
[CODE L0383] 
[CODE L0384]         outputs.clear();
[CODE L0385]         outputs.reserve(n);
[CODE L0386]         for (i, &value) in values_flat.iter().enumerate().take(n) {
[CODE L0387]             let logits_start = i * HYDRA_ACTION_SPACE;
[CODE L0388]             let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
[CODE L0389]                 [logits_start..logits_start + HYDRA_ACTION_SPACE]
[CODE L0390]                 .try_into()
[CODE L0391]                 .expect("logits slice length mismatch");
[CODE L0392]             outputs.push((logits, value));
[CODE L0393]         }
[CODE L0394]     }
[CODE L0395] 
[CODE L0396]     pub fn batch_policy_value_cpu_reuse(
[CODE L0397]         &self,
[CODE L0398]         observations: &[[f32; OBS_SIZE]],
[CODE L0399]         device: &B::Device,
[CODE L0400]         flat_buf: &mut Vec<f32>,
[CODE L0401]         outputs: &mut Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
[CODE L0402]     ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
[CODE L0403]         self.fill_batch_policy_value_cpu(observations, device, flat_buf, outputs);
[CODE L0404]         outputs.clone()
[CODE L0405]     }
[CODE L0406] 
[CODE L0407]     pub fn fill_batch_value_cpu(
[CODE L0408]         &self,
[CODE L0409]         observations: &[[f32; OBS_SIZE]],
[CODE L0410]         device: &B::Device,
[CODE L0411]         flat_buf: &mut Vec<f32>,
[CODE L0412]         values_out: &mut Vec<f32>,
[CODE L0413]     ) {
[CODE L0414]         if observations.is_empty() {
[CODE L0415]             values_out.clear();
[CODE L0416]             return;
[CODE L0417]         }
[CODE L0418]         let n = observations.len();
[CODE L0419]         flat_buf.clear();
[CODE L0420]         flat_buf.reserve(n * OBS_SIZE);
[CODE L0421]         for obs in observations {
[CODE L0422]             flat_buf.extend_from_slice(obs);
[CODE L0423]         }
[CODE L0424]         let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
[CODE L0425]             n as i32,
[CODE L0426]             NUM_CHANNELS as i32,
[CODE L0427]             NUM_TILES as i32,
[CODE L0428]         ]);
[CODE L0429]         let value = self.forward_value(input);
[CODE L0430]         let values_data = value.to_data().convert::<f32>();
[CODE L0431]         let values = values_data
[CODE L0432]             .as_slice::<f32>()
[CODE L0433]             .expect("batch value extraction failed");
[CODE L0434]         values_out.clear();
[CODE L0435]         values_out.extend_from_slice(values);
[CODE L0436]     }
[CODE L0437] 
[CODE L0438]     pub fn batch_value_cpu_reuse(
[CODE L0439]         &self,
[CODE L0440]         observations: &[[f32; OBS_SIZE]],
[CODE L0441]         device: &B::Device,
[CODE L0442]         flat_buf: &mut Vec<f32>,
[CODE L0443]         values_out: &mut Vec<f32>,
[CODE L0444]     ) -> Vec<f32> {
[CODE L0445]         self.fill_batch_value_cpu(observations, device, flat_buf, values_out);
[CODE L0446]         values_out.clone()
[CODE L0447]     }
[CODE L0448] 
[CODE L0449]     /// Runs a batch of observations through the full model and returns
[CODE L0450]     /// per-sample policy logits and value scalars on the CPU.
[CODE L0451]     ///
[CODE L0452]     /// This amortizes GPU kernel launch overhead across N samples. The
[CODE L0453]     /// input observations are concatenated into a single `[N, C, T]` tensor
[CODE L0454]     /// for one forward pass, then results are sliced per sample.
[CODE L0455]     pub fn batch_policy_value_cpu(
[CODE L0456]         &self,
[CODE L0457]         observations: &[[f32; OBS_SIZE]],
[CODE L0458]         device: &B::Device,
[CODE L0459]     ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
[CODE L0460]         if observations.is_empty() {
[CODE L0461]             return Vec::new();
[CODE L0462]         }
[CODE L0463]         let n = observations.len();
[CODE L0464]         let mut flat = Vec::with_capacity(n * OBS_SIZE);
[CODE L0465]         for obs in observations {
[CODE L0466]             flat.extend_from_slice(obs);
[CODE L0467]         }
[CODE L0468]         let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([
[CODE L0469]             n as i32,
[CODE L0470]             NUM_CHANNELS as i32,
[CODE L0471]             NUM_TILES as i32,
[CODE L0472]         ]);
[CODE L0473]         let (policy_logits, value) = self.forward_policy_value(input);
[CODE L0474]         let logits_flat = policy_logits
[CODE L0475]             .to_data()
[CODE L0476]             .convert::<f32>()
[CODE L0477]             .as_slice::<f32>()
[CODE L0478]             .expect("batch policy logits extraction failed")
[CODE L0479]             .to_vec();
[CODE L0480]         let values_flat = value
[CODE L0481]             .to_data()
[CODE L0482]             .convert::<f32>()
[CODE L0483]             .as_slice::<f32>()
[CODE L0484]             .expect("batch value extraction failed")
[CODE L0485]             .to_vec();
[CODE L0486] 
[CODE L0487]         (0..n)
[CODE L0488]             .map(|i| {
[CODE L0489]                 let logits_start = i * HYDRA_ACTION_SPACE;
[CODE L0490]                 let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
[CODE L0491]                     [logits_start..logits_start + HYDRA_ACTION_SPACE]
[CODE L0492]                     .try_into()
[CODE L0493]                     .expect("logits slice length mismatch");
[CODE L0494]                 let value = values_flat[i];
[CODE L0495]                 (logits, value)
[CODE L0496]             })
[CODE L0497]             .collect()
[CODE L0498]     }
[CODE L0499] 
[CODE L0500]     /// Runs only backbone + policy + value heads.
[CODE L0501]     ///
[CODE L0502]     /// Self-play inference only needs logits and value. Skipping the
[CODE L0503]     /// other 12 heads avoids ~12 unnecessary matmuls and their VRAM
[CODE L0504]     /// allocations per forward pass.
[CODE L0505]     pub fn forward_value(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
[CODE L0506]         let (_, pooled) = self.backbone.forward(x);
[CODE L0507]         self.value.forward(pooled)
[CODE L0508]     }
[CODE L0509] 
[CODE L0510]     pub fn forward_policy_value(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
[CODE L0511]         let (_, pooled) = self.backbone.forward(x);
[CODE L0512]         let policy_logits = self.policy.forward(pooled.clone());
[CODE L0513]         let value = self.value.forward(pooled);
[CODE L0514]         (policy_logits, value)
[CODE L0515]     }
[CODE L0516] 
[CODE L0517]     /// Forward pass that detaches outputs of zero-weight heads.
[CODE L0518]     ///
[CODE L0519]     /// All heads still run their forward pass (shapes must match), but
[CODE L0520]     /// heads with zero loss weight have their outputs detached from the
[CODE L0521]     /// autograd graph. This prevents gradient computation and reduces
[CODE L0522]     /// VRAM usage for activations that won't contribute to the loss.
[CODE L0523]     pub fn forward_active(
[CODE L0524]         &self,
[CODE L0525]         x: Tensor<B, 3>,
[CODE L0526]         loss_cfg: &crate::training::losses::HydraLossConfig,
[CODE L0527]     ) -> HydraOutput<B> {
[CODE L0528]         self.forward_with_warmup(x, loss_cfg, &[])
[CODE L0529]     }
[CODE L0530] 
[CODE L0531]     pub fn forward_with_warmup(
[CODE L0532]         &self,
[CODE L0533]         x: Tensor<B, 3>,
[CODE L0534]         loss_cfg: &crate::training::losses::HydraLossConfig,
[CODE L0535]         warmup_heads: &[crate::training::head_gates::AdvancedHead],
[CODE L0536]     ) -> HydraOutput<B> {
[CODE L0537]         let (spatial, pooled) = self.backbone.forward(x);
[CODE L0538]         let oracle_input = pooled.clone().detach();
[CODE L0539]         let is_warmup =
[CODE L0540]             |head: crate::training::head_gates::AdvancedHead| warmup_heads.contains(&head);
[CODE L0541]         let batch = pooled.dims()[0];
[CODE L0542]         let device = pooled.device();
[CODE L0543] 
[CODE L0544]         let policy_logits = self.policy.forward(pooled.clone());
[CODE L0545]         let value = self.value.forward(pooled.clone());
[CODE L0546]         let score_pdf = self.score_pdf.forward(pooled.clone());
[CODE L0547]         let score_cdf = self.score_cdf.forward(pooled.clone());
[CODE L0548]         let opp_tenpai = self.opp_tenpai.forward(pooled.clone());
[CODE L0549]         let grp = self.grp.forward(pooled.clone());
[CODE L0550]         let opp_next_discard = self.opp_next_discard.forward(spatial.clone());
[CODE L0551]         let danger = self.danger.forward(spatial.clone());
[CODE L0552]         let oracle_critic = if loss_cfg.w_oracle_critic > 0.0
[CODE L0553]             && !is_warmup(crate::training::head_gates::AdvancedHead::OracleCritic)
[CODE L0554]         {
[CODE L0555]             self.oracle_critic.forward(oracle_input)
[CODE L0556]         } else {
[CODE L0557]             zero_linear_head(batch, 4, &device)
[CODE L0558]         };
[CODE L0559]         let belief_fields = if loss_cfg.w_belief_fields > 0.0
[CODE L0560]             && !is_warmup(crate::training::head_gates::AdvancedHead::BeliefFields)
[CODE L0561]         {
[CODE L0562]             self.belief_field.forward(spatial.clone())
[CODE L0563]         } else {
[CODE L0564]             zero_spatial_head(batch, 16, 34, &device)
[CODE L0565]         };
[CODE L0566]         let mixture_weight_logits = if loss_cfg.w_mixture_weight > 0.0
[CODE L0567]             && !is_warmup(crate::training::head_gates::AdvancedHead::MixtureWeight)
[CODE L0568]         {
[CODE L0569]             self.mixture_weight.forward(pooled.clone())
[CODE L0570]         } else {
[CODE L0571]             zero_linear_head(batch, 4, &device)
[CODE L0572]         };
[CODE L0573]         let opponent_hand_type = if loss_cfg.w_opponent_hand_type > 0.0
[CODE L0574]             && !is_warmup(crate::training::head_gates::AdvancedHead::OpponentHandType)
[CODE L0575]         {
[CODE L0576]             self.opponent_hand_type.forward(pooled.clone())
[CODE L0577]         } else {
[CODE L0578]             zero_linear_head(batch, 24, &device)
[CODE L0579]         };
[CODE L0580]         let delta_q = if loss_cfg.w_delta_q > 0.0
[CODE L0581]             && !is_warmup(crate::training::head_gates::AdvancedHead::DeltaQ)
[CODE L0582]         {
[CODE L0583]             self.delta_q.forward(pooled.clone())
[CODE L0584]         } else {
[CODE L0585]             zero_linear_head(batch, HYDRA_ACTION_SPACE, &device)
[CODE L0586]         };
[CODE L0587]         let safety_residual = if loss_cfg.w_safety_residual > 0.0
[CODE L0588]             && !is_warmup(crate::training::head_gates::AdvancedHead::SafetyResidual)
[CODE L0589]         {
[CODE L0590]             self.safety_residual.forward(pooled)
[CODE L0591]         } else {
[CODE L0592]             zero_linear_head(batch, HYDRA_ACTION_SPACE, &device)
[CODE L0593]         };
[CODE L0594] 
[CODE L0595]         HydraOutput {
[CODE L0596]             policy_logits,
[CODE L0597]             value,
[CODE L0598]             score_pdf: if loss_cfg.w_score > 0.0 {
[CODE L0599]                 score_pdf
[CODE L0600]             } else {
[CODE L0601]                 score_pdf.detach()
[CODE L0602]             },
[CODE L0603]             score_cdf: if loss_cfg.w_score > 0.0 {
[CODE L0604]                 score_cdf
[CODE L0605]             } else {
[CODE L0606]                 score_cdf.detach()
[CODE L0607]             },
[CODE L0608]             opp_tenpai: if loss_cfg.w_tenpai > 0.0 {
[CODE L0609]                 opp_tenpai
[CODE L0610]             } else {
[CODE L0611]                 opp_tenpai.detach()
[CODE L0612]             },
[CODE L0613]             grp: if loss_cfg.w_grp > 0.0 {
[CODE L0614]                 grp
[CODE L0615]             } else {
[CODE L0616]                 grp.detach()
[CODE L0617]             },
[CODE L0618]             opp_next_discard: if loss_cfg.w_opp > 0.0 {
[CODE L0619]                 opp_next_discard
[CODE L0620]             } else {
[CODE L0621]                 opp_next_discard.detach()
[CODE L0622]             },
[CODE L0623]             danger: if loss_cfg.w_danger > 0.0 {
[CODE L0624]                 danger
[CODE L0625]             } else {
[CODE L0626]                 danger.detach()
[CODE L0627]             },
[CODE L0628]             oracle_critic,
[CODE L0629]             belief_fields,
[CODE L0630]             mixture_weight_logits,
[CODE L0631]             opponent_hand_type,
[CODE L0632]             delta_q,
[CODE L0633]             safety_residual,
[CODE L0634]         }
[CODE L0635]     }
[CODE L0636] 
[CODE L0637]     pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
[CODE L0638]         let (spatial, pooled) = self.backbone.forward(x);
[CODE L0639]         let oracle_input = pooled.clone().detach();
[CODE L0640]         HydraOutput {
[CODE L0641]             policy_logits: self.policy.forward(pooled.clone()),
[CODE L0642]             value: self.value.forward(pooled.clone()),
[CODE L0643]             score_pdf: self.score_pdf.forward(pooled.clone()),
[CODE L0644]             score_cdf: self.score_cdf.forward(pooled.clone()),
[CODE L0645]             opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
[CODE L0646]             grp: self.grp.forward(pooled.clone()),
[CODE L0647]             opp_next_discard: self.opp_next_discard.forward(spatial.clone()),
[CODE L0648]             danger: self.danger.forward(spatial.clone()),
[CODE L0649]             oracle_critic: self.oracle_critic.forward(oracle_input),
[CODE L0650]             belief_fields: self.belief_field.forward(spatial),
[CODE L0651]             mixture_weight_logits: self.mixture_weight.forward(pooled.clone()),
[CODE L0652]             opponent_hand_type: self.opponent_hand_type.forward(pooled.clone()),
[CODE L0653]             delta_q: self.delta_q.forward(pooled.clone()),
[CODE L0654]             safety_residual: self.safety_residual.forward(pooled),
[CODE L0655]         }
[CODE L0656]     }
[CODE L0657] }
[CODE L0658] 
[CODE L0659] #[cfg(test)]
[CODE L0660] mod tests {
[CODE L0661]     use super::*;
[CODE L0662]     use crate::training::losses::{tests::make_dummy_targets, HydraLoss, HydraLossConfig};
[CODE L0663]     use burn::backend::Autodiff;
[CODE L0664]     use burn::backend::LibTorch;
[CODE L0665]     use burn::backend::NdArray;
[CODE L0666]     use burn::optim::AdamConfig;
[CODE L0667]     use burn::optim::Optimizer;
[CODE L0668]     use burn::tensor::bf16;
[CODE L0669] 
[CODE L0670]     type B = NdArray<f32>;
[CODE L0671]     type AB = Autodiff<NdArray<f32>>;
[CODE L0672] 
[CODE L0673]     fn assert_output_shapes(out: &HydraOutput<B>, batch: usize) {
[CODE L0674]         assert_eq!(out.policy_logits.dims(), [batch, 46]);
[CODE L0675]         assert_eq!(out.value.dims(), [batch, 1]);
[CODE L0676]         assert_eq!(out.score_pdf.dims(), [batch, 64]);
[CODE L0677]         assert_eq!(out.score_cdf.dims(), [batch, 64]);
[CODE L0678]         assert_eq!(out.opp_tenpai.dims(), [batch, 3]);
[CODE L0679]         assert_eq!(out.grp.dims(), [batch, 24]);
[CODE L0680]         assert_eq!(out.opp_next_discard.dims(), [batch, 3, 34]);
[CODE L0681]         assert_eq!(out.danger.dims(), [batch, 3, 34]);
[CODE L0682]         assert_eq!(out.oracle_critic.dims(), [batch, 4]);
[CODE L0683]         assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
[CODE L0684]         assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
[CODE L0685]         assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
[CODE L0686]         assert_eq!(out.delta_q.dims(), [batch, 46]);
[CODE L0687]         assert_eq!(out.safety_residual.dims(), [batch, 46]);
[CODE L0688]     }
[CODE L0689] 
[CODE L0690]     #[test]
[CODE L0691]     fn actor_net_all_output_shapes() {
[CODE L0692]         let device = Default::default();
[CODE L0693]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0694]         let x = Tensor::<B, 3>::zeros([4, INPUT_CHANNELS, 34], &device);
[CODE L0695]         let out = model.forward(x);
[CODE L0696]         assert_output_shapes(&out, 4);
[CODE L0697]     }
[CODE L0698] 
[CODE L0699]     #[test]
[CODE L0700]     fn learner_net_all_output_shapes() {
[CODE L0701]         let device = Default::default();
[CODE L0702]         let model = HydraModelConfig::learner().init::<B>(&device);
[CODE L0703]         let x = Tensor::<B, 3>::zeros([2, INPUT_CHANNELS, 34], &device);
[CODE L0704]         let out = model.forward(x);
[CODE L0705]         assert_output_shapes(&out, 2);
[CODE L0706]     }
[CODE L0707] 
[CODE L0708]     #[test]
[CODE L0709]     fn value_head_bounded() {
[CODE L0710]         let device = Default::default();
[CODE L0711]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0712]         let x = Tensor::<B, 3>::random(
[CODE L0713]             [4, INPUT_CHANNELS, 34],
[CODE L0714]             burn::tensor::Distribution::Normal(0.0, 1.0),
[CODE L0715]             &device,
[CODE L0716]         );
[CODE L0717]         let out = model.forward(x);
[CODE L0718]         let data = out.value.to_data();
[CODE L0719]         for &v in data.as_slice::<f32>().expect("f32") {
[CODE L0720]             assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
[CODE L0721]         }
[CODE L0722]     }
[CODE L0723] 
[CODE L0724]     #[test]
[CODE L0725]     fn policy_value_cpu_returns_correct_shapes() {
[CODE L0726]         let device = Default::default();
[CODE L0727]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0728]         let obs = [0.0f32; OBS_SIZE];
[CODE L0729]         let (logits, value) = model.policy_value_cpu(&obs, &device);
[CODE L0730]         assert_eq!(logits.len(), HYDRA_ACTION_SPACE);
[CODE L0731]         assert!(value.is_finite());
[CODE L0732]         assert!(logits.iter().all(|v| v.is_finite()));
[CODE L0733]     }
[CODE L0734] 
[CODE L0735]     #[test]
[CODE L0736]     fn batch_policy_value_cpu_matches_single_sample_path() {
[CODE L0737]         let device = Default::default();
[CODE L0738]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0739]         let obs_a = [0.0f32; OBS_SIZE];
[CODE L0740]         let obs_b = [0.25f32; OBS_SIZE];
[CODE L0741]         let observations = [obs_a, obs_b];
[CODE L0742] 
[CODE L0743]         let single_outputs: Vec<_> = observations
[CODE L0744]             .iter()
[CODE L0745]             .map(|obs| model.policy_value_cpu(obs, &device))
[CODE L0746]             .collect();
[CODE L0747]         let batch_outputs = model.batch_policy_value_cpu(&observations, &device);
[CODE L0748] 
[CODE L0749]         assert_eq!(batch_outputs.len(), single_outputs.len());
[CODE L0750]         for ((batch_logits, batch_value), (single_logits, single_value)) in
[CODE L0751]             batch_outputs.iter().zip(single_outputs.iter())
[CODE L0752]         {
[CODE L0753]             for (batch, single) in batch_logits.iter().zip(single_logits.iter()) {
[CODE L0754]                 assert!((batch - single).abs() < 1e-6);
[CODE L0755]             }
[CODE L0756]             assert!((batch_value - single_value).abs() < 1e-6);
[CODE L0757]         }
[CODE L0758]     }
[CODE L0759] 
[CODE L0760]     #[test]
[CODE L0761]     fn batch_policy_value_cpu_reuse_matches_non_reuse_path() {
[CODE L0762]         let device = Default::default();
[CODE L0763]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0764]         let obs_a = [0.1f32; OBS_SIZE];
[CODE L0765]         let obs_b = [0.2f32; OBS_SIZE];
[CODE L0766]         let obs_c = [0.3f32; OBS_SIZE];
[CODE L0767]         let observations = [obs_a, obs_b, obs_c];
[CODE L0768] 
[CODE L0769]         let expected = model.batch_policy_value_cpu(&observations, &device);
[CODE L0770]         let mut flat_buf = vec![42.0f32; 17];
[CODE L0771]         let mut outputs_buf = Vec::new();
[CODE L0772]         let reused = model.batch_policy_value_cpu_reuse(
[CODE L0773]             &observations,
[CODE L0774]             &device,
[CODE L0775]             &mut flat_buf,
[CODE L0776]             &mut outputs_buf,
[CODE L0777]         );
[CODE L0778] 
[CODE L0779]         assert_eq!(reused.len(), expected.len());
[CODE L0780]         for ((reuse_logits, reuse_value), (expected_logits, expected_value)) in
[CODE L0781]             reused.iter().zip(expected.iter())
[CODE L0782]         {
[CODE L0783]             for (reuse, expected) in reuse_logits.iter().zip(expected_logits.iter()) {
[CODE L0784]                 assert!((reuse - expected).abs() < 1e-6);
[CODE L0785]             }
[CODE L0786]             assert!((reuse_value - expected_value).abs() < 1e-6);
[CODE L0787]         }
[CODE L0788]     }
[CODE L0789] 
[CODE L0790]     #[test]
[CODE L0791]     fn batch_value_cpu_reuse_matches_policy_value_values_on_dirty_buffer() {
[CODE L0792]         let device = Default::default();
[CODE L0793]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0794]         let observations = [
[CODE L0795]             [0.05f32; OBS_SIZE],
[CODE L0796]             [0.15f32; OBS_SIZE],
[CODE L0797]             [0.25f32; OBS_SIZE],
[CODE L0798]         ];
[CODE L0799] 
[CODE L0800]         let expected = model.batch_policy_value_cpu(&observations, &device);
[CODE L0801]         let mut flat_buf = vec![13.0f32; 29];
[CODE L0802]         let mut values_buf = Vec::new();
[CODE L0803]         let values =
[CODE L0804]             model.batch_value_cpu_reuse(&observations, &device, &mut flat_buf, &mut values_buf);
[CODE L0805] 
[CODE L0806]         assert_eq!(values.len(), expected.len());
[CODE L0807]         for (value, (_, expected_value)) in values.iter().zip(expected.iter()) {
[CODE L0808]             assert!((value - expected_value).abs() < 1e-6);
[CODE L0809]         }
[CODE L0810]     }
[CODE L0811] 
[CODE L0812]     #[test]
[CODE L0813]     fn batch_value_cpu_reuse_supports_libtorch_bf16_backend() {
[CODE L0814]         type Bf16Backend = LibTorch<bf16>;
[CODE L0815] 
[CODE L0816]         let tiny_model_config = HydraModelConfig::new(2)
[CODE L0817]             .with_input_channels(INPUT_CHANNELS)
[CODE L0818]             .with_hidden_channels(16)
[CODE L0819]             .with_num_groups(4)
[CODE L0820]             .with_se_bottleneck(4);
[CODE L0821] 
[CODE L0822]         let device = burn::backend::libtorch::LibTorchDevice::Cpu;
[CODE L0823]         let model = tiny_model_config.init::<Bf16Backend>(&device);
[CODE L0824]         let observations = [[0.05f32; OBS_SIZE]];
[CODE L0825]         let mut flat_buf = vec![7.0f32; 11];
[CODE L0826]         let mut values_buf = Vec::new();
[CODE L0827] 
[CODE L0828]         let values =
[CODE L0829]             model.batch_value_cpu_reuse(&observations, &device, &mut flat_buf, &mut values_buf);
[CODE L0830]         assert_eq!(values.len(), observations.len());
[CODE L0831]         assert!(values.iter().all(|value| value.is_finite()));
[CODE L0832] 
[CODE L0833]         let outputs = model.batch_policy_value_cpu(&observations, &device);
[CODE L0834]         for (value, (_, expected_value)) in values.iter().zip(outputs.iter()) {
[CODE L0835]             assert!((value - expected_value).abs() < 1e-4);
[CODE L0836]         }
[CODE L0837]     }
[CODE L0838] 
[CODE L0839]     #[test]
[CODE L0840]     fn hydra_loss_runs_on_libtorch_bf16_backend() {
[CODE L0841]         type Bf16Backend = LibTorch<bf16>;
[CODE L0842] 
[CODE L0843]         let tiny_model_config = HydraModelConfig::new(2)
[CODE L0844]             .with_input_channels(INPUT_CHANNELS)
[CODE L0845]             .with_hidden_channels(16)
[CODE L0846]             .with_num_groups(4)
[CODE L0847]             .with_se_bottleneck(4);
[CODE L0848] 
[CODE L0849]         let device = burn::backend::libtorch::LibTorchDevice::Cpu;
[CODE L0850]         let model = tiny_model_config.init::<Bf16Backend>(&device);
[CODE L0851]         let x = Tensor::<Bf16Backend, 3>::zeros([1, INPUT_CHANNELS, 34], &device);
[CODE L0852]         let out = model.forward(x);
[CODE L0853]         let targets = make_dummy_targets::<Bf16Backend>(&device, 1);
[CODE L0854]         let hydra_loss = HydraLoss::<Bf16Backend>::new(HydraLossConfig::new());
[CODE L0855]         let breakdown = hydra_loss.total_loss(&out, &targets);
[CODE L0856]         let total = breakdown
[CODE L0857]             .total
[CODE L0858]             .into_data()
[CODE L0859]             .convert::<f32>()
[CODE L0860]             .as_slice::<f32>()
[CODE L0861]             .expect("bf16 total loss should be readable as f32")[0];
[CODE L0862] 
[CODE L0863]         assert!(total.is_finite());
[CODE L0864]         assert!(total >= 0.0);
[CODE L0865]     }
[CODE L0866] 
[CODE L0867]     #[test]
[CODE L0868]     fn hydra_training_step_runs_on_libtorch_bf16_backend() {
[CODE L0869]         type Bf16Backend = Autodiff<LibTorch<bf16>>;
[CODE L0870] 
[CODE L0871]         let tiny_model_config = HydraModelConfig::new(2)
[CODE L0872]             .with_input_channels(INPUT_CHANNELS)
[CODE L0873]             .with_hidden_channels(16)
[CODE L0874]             .with_num_groups(4)
[CODE L0875]             .with_se_bottleneck(4);
[CODE L0876] 
[CODE L0877]         let device = burn::backend::libtorch::LibTorchDevice::Cpu;
[CODE L0878]         let model = tiny_model_config.init::<Bf16Backend>(&device);
[CODE L0879]         let x = Tensor::<Bf16Backend, 3>::zeros([1, INPUT_CHANNELS, 34], &device);
[CODE L0880]         let out = model.forward(x);
[CODE L0881]         let targets = make_dummy_targets::<Bf16Backend>(&device, 1);
[CODE L0882]         let hydra_loss = HydraLoss::<Bf16Backend>::new(HydraLossConfig::new());
[CODE L0883]         let breakdown = hydra_loss.total_loss(&out, &targets);
[CODE L0884]         let loss = breakdown.total;
[CODE L0885]         let grads = loss.backward();
[CODE L0886]         let grads = burn::optim::GradientsParams::from_grads(grads, &model);
[CODE L0887]         let mut optim = AdamConfig::new().init();
[CODE L0888]         let model = optim.step(1e-4, model, grads);
[CODE L0889] 
[CODE L0890]         let outputs = model.batch_policy_value_cpu(&[[0.0f32; OBS_SIZE]], &device);
[CODE L0891]         assert_eq!(outputs.len(), 1);
[CODE L0892]         assert!(outputs[0].1.is_finite());
[CODE L0893]     }
[CODE L0894] 
[CODE L0895]     #[test]
[CODE L0896]     fn actor_and_learner_param_counts_differ() {
[CODE L0897]         let device = Default::default();
[CODE L0898]         let actor = HydraModelConfig::actor().init::<B>(&device);
[CODE L0899]         let learner = HydraModelConfig::learner().init::<B>(&device);
[CODE L0900]         let a_params = actor.num_params();
[CODE L0901]         let l_params = learner.num_params();
[CODE L0902]         assert!(
[CODE L0903]             l_params > a_params,
[CODE L0904]             "learner ({l_params}) should have more params than actor ({a_params})"
[CODE L0905]         );
[CODE L0906]         assert!(
[CODE L0907]             a_params > 1_000_000,
[CODE L0908]             "actor should have >1M params, got {a_params}"
[CODE L0909]         );
[CODE L0910]         assert!(
[CODE L0911]             l_params > 5_000_000,
[CODE L0912]             "learner should have >5M params, got {l_params}"
[CODE L0913]         );
[CODE L0914]     }
[CODE L0915] 
[CODE L0916]     #[test]
[CODE L0917]     fn all_outputs_finite_for_random_input() {
[CODE L0918]         let device = Default::default();
[CODE L0919]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0920]         let x = Tensor::<B, 3>::random(
[CODE L0921]             [8, INPUT_CHANNELS, 34],
[CODE L0922]             burn::tensor::Distribution::Normal(0.0, 1.0),
[CODE L0923]             &device,
[CODE L0924]         );
[CODE L0925]         let out = model.forward(x);
[CODE L0926]         let check = |t: &Tensor<B, 2>, name: &str| {
[CODE L0927]             let d = t.to_data();
[CODE L0928]             for &v in d.as_slice::<f32>().expect("f32") {
[CODE L0929]                 assert!(v.is_finite(), "{name} has non-finite: {v}");
[CODE L0930]             }
[CODE L0931]         };
[CODE L0932]         let check_spatial = |t: &Tensor<B, 3>, name: &str| {
[CODE L0933]             let d = t.to_data();
[CODE L0934]             for &v in d.as_slice::<f32>().expect("f32") {
[CODE L0935]                 assert!(v.is_finite(), "{name} has non-finite: {v}");
[CODE L0936]             }
[CODE L0937]         };
[CODE L0938]         check(&out.policy_logits, "policy");
[CODE L0939]         check(&out.value, "value");
[CODE L0940]         check(&out.score_pdf, "score_pdf");
[CODE L0941]         check(&out.score_cdf, "score_cdf");
[CODE L0942]         check(&out.opp_tenpai, "opp_tenpai");
[CODE L0943]         check(&out.grp, "grp");
[CODE L0944]         check(&out.oracle_critic, "oracle_critic");
[CODE L0945]         check_spatial(&out.opp_next_discard, "opp_next_discard");
[CODE L0946]         check_spatial(&out.danger, "danger");
[CODE L0947]         check_spatial(&out.belief_fields, "belief_fields");
[CODE L0948]         check(&out.mixture_weight_logits, "mixture_weight_logits");
[CODE L0949]         check(&out.opponent_hand_type, "opponent_hand_type");
[CODE L0950]         check(&out.delta_q, "delta_q");
[CODE L0951]         check(&out.safety_residual, "safety_residual");
[CODE L0952]     }
[CODE L0953] 
[CODE L0954]     #[test]
[CODE L0955]     fn oracle_head_does_not_backprop_to_backbone_input() {
[CODE L0956]         let device = Default::default();
[CODE L0957]         let model = HydraModelConfig::actor().init::<AB>(&device);
[CODE L0958]         let x = Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device).require_grad();
[CODE L0959]         let out = model.forward(x.clone());
[CODE L0960]         let target = Tensor::<AB, 2>::ones([2, 4], &device);
[CODE L0961]         let diff = out.oracle_critic - target;
[CODE L0962]         let loss = (diff.clone() * diff).mean();
[CODE L0963]         let grads = loss.backward();
[CODE L0964] 
[CODE L0965]         assert!(
[CODE L0966]             x.grad(&grads).is_none(),
[CODE L0967]             "oracle-only loss must not backpropagate through the shared backbone"
[CODE L0968]         );
[CODE L0969]     }
[CODE L0970] 
[CODE L0971]     #[test]
[CODE L0972]     fn delta_q_warmup_detaches_backbone_input() {
[CODE L0973]         let device = Default::default();
[CODE L0974]         let model = HydraModelConfig::actor().init::<AB>(&device);
[CODE L0975]         let x = Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device).require_grad();
[CODE L0976]         let loss_cfg = crate::training::losses::HydraLossConfig::new().with_w_delta_q(1.0);
[CODE L0977]         let out = model.forward_with_warmup(
[CODE L0978]             x.clone(),
[CODE L0979]             &loss_cfg,
[CODE L0980]             &[crate::training::head_gates::AdvancedHead::DeltaQ],
[CODE L0981]         );
[CODE L0982]         let target = Tensor::<AB, 2>::ones([2, HYDRA_ACTION_SPACE], &device);
[CODE L0983]         let diff = out.delta_q - target;
[CODE L0984]         let loss = (diff.clone() * diff).mean();
[CODE L0985]         let grads = loss.backward();
[CODE L0986] 
[CODE L0987]         assert!(
[CODE L0988]             x.grad(&grads).is_none(),
[CODE L0989]             "delta_q warmup loss must not backpropagate through the shared backbone"
[CODE L0990]         );
[CODE L0991]     }
[CODE L0992] 
[CODE L0993]     #[test]
[CODE L0994]     fn active_delta_q_backprops_to_backbone_input() {
[CODE L0995]         let device = Default::default();
[CODE L0996]         let model = HydraModelConfig::actor().init::<AB>(&device);
[CODE L0997]         let x = Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device).require_grad();
[CODE L0998]         let loss_cfg = crate::training::losses::HydraLossConfig::new().with_w_delta_q(1.0);
[CODE L0999]         let out = model.forward_active(x.clone(), &loss_cfg);
[CODE L1000]         let target = Tensor::<AB, 2>::ones([2, HYDRA_ACTION_SPACE], &device);
[CODE L1001]         let diff = out.delta_q - target;
[CODE L1002]         let loss = (diff.clone() * diff).mean();
[CODE L1003]         let grads = loss.backward();
[CODE L1004] 
[CODE L1005]         assert!(
[CODE L1006]             x.grad(&grads).is_some(),
[CODE L1007]             "active delta_q loss should backpropagate through the shared backbone"
[CODE L1008]         );
[CODE L1009]     }
[CODE L1010] 
[CODE L1011]     #[test]
[CODE L1012]     fn inactive_advanced_heads_return_zero_tensors() {
[CODE L1013]         let device = Default::default();
[CODE L1014]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1015]         let x = Tensor::<B, 3>::zeros([2, INPUT_CHANNELS, 34], &device);
[CODE L1016]         let loss_cfg = crate::training::losses::HydraLossConfig::new();
[CODE L1017]         let out = model.forward_active(x, &loss_cfg);
[CODE L1018] 
[CODE L1019]         for &value in out.oracle_critic.to_data().as_slice::<f32>().expect("f32") {
[CODE L1020]             assert_eq!(value, 0.0);
[CODE L1021]         }
[CODE L1022]         for &value in out.belief_fields.to_data().as_slice::<f32>().expect("f32") {
[CODE L1023]             assert_eq!(value, 0.0);
[CODE L1024]         }
[CODE L1025]         for &value in out
[CODE L1026]             .mixture_weight_logits
[CODE L1027]             .to_data()
[CODE L1028]             .as_slice::<f32>()
[CODE L1029]             .expect("f32")
[CODE L1030]         {
[CODE L1031]             assert_eq!(value, 0.0);
[CODE L1032]         }
[CODE L1033]         for &value in out
[CODE L1034]             .opponent_hand_type
[CODE L1035]             .to_data()
[CODE L1036]             .as_slice::<f32>()
[CODE L1037]             .expect("f32")
[CODE L1038]         {
[CODE L1039]             assert_eq!(value, 0.0);
[CODE L1040]         }
[CODE L1041]         for &value in out.delta_q.to_data().as_slice::<f32>().expect("f32") {
[CODE L1042]             assert_eq!(value, 0.0);
[CODE L1043]         }
[CODE L1044]         for &value in out
[CODE L1045]             .safety_residual
[CODE L1046]             .to_data()
[CODE L1047]             .as_slice::<f32>()
[CODE L1048]             .expect("f32")
[CODE L1049]         {
[CODE L1050]             assert_eq!(value, 0.0);
[CODE L1051]         }
[CODE L1052]     }
[CODE L1053] 
[CODE L1054]     #[test]
[CODE L1055]     fn model_config_actor_learner_defaults() {
[CODE L1056]         let actor = HydraModelConfig::actor();
[CODE L1057]         assert_eq!(actor.num_blocks, 12);
[CODE L1058]         assert_eq!(actor.hidden_channels, 256);
[CODE L1059]         assert_eq!(actor.num_groups, 32);
[CODE L1060]         let learner = HydraModelConfig::learner();
[CODE L1061]         assert_eq!(learner.num_blocks, 24);
[CODE L1062]         assert_eq!(learner.hidden_channels, 256);
[CODE L1063]     }
[CODE L1064] 
[CODE L1065]     #[test]
[CODE L1066]     fn validate_passes_for_standard_configs() {
[CODE L1067]         assert!(HydraModelConfig::actor().validate().is_ok());
[CODE L1068]         assert!(HydraModelConfig::learner().validate().is_ok());
[CODE L1069]     }
[CODE L1070] }
```

## Artifact 19 — Training losses and weighting logic
Artifact id: `losses-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-train/src/training/losses.rs`
Why it matters: Large but high-value code artifact showing how Hydra currently composes losses, target weights, and head-gating behavior. Useful for deciding whether any DCRL-like weighting insight is absent or already subsumed.

```rust
[CODE L0001] use burn::prelude::*;
[CODE L0002] use burn::tensor::activation;
[CODE L0003] use std::marker::PhantomData;
[CODE L0004] 
[CODE L0005] use crate::model::HydraOutput;
[CODE L0006] use crate::training::head_gates::TargetPresence;
[CODE L0007] 
[CODE L0008] #[derive(Clone)]
[CODE L0009] pub struct HydraTargets<B: Backend> {
[CODE L0010]     pub policy_target: Tensor<B, 2>,
[CODE L0011]     pub legal_mask: Tensor<B, 2>,
[CODE L0012]     pub value_target: Tensor<B, 1>,
[CODE L0013]     pub grp_target: Tensor<B, 2>,
[CODE L0014]     pub tenpai_target: Tensor<B, 2>,
[CODE L0015]     pub danger_target: Tensor<B, 3>,
[CODE L0016]     pub danger_mask: Tensor<B, 3>,
[CODE L0017]     pub opp_next_target: Tensor<B, 3>,
[CODE L0018]     pub score_pdf_target: Tensor<B, 2>,
[CODE L0019]     pub score_cdf_target: Tensor<B, 2>,
[CODE L0020]     pub oracle_target: Option<Tensor<B, 2>>,
[CODE L0021]     pub belief_fields_target: Option<Tensor<B, 3>>,
[CODE L0022]     pub belief_fields_mask: Option<Tensor<B, 1>>,
[CODE L0023]     pub mixture_weight_target: Option<Tensor<B, 2>>,
[CODE L0024]     pub mixture_weight_mask: Option<Tensor<B, 1>>,
[CODE L0025]     pub opponent_hand_type_target: Option<Tensor<B, 2>>,
[CODE L0026]     pub delta_q_target: Option<Tensor<B, 2>>,
[CODE L0027]     pub delta_q_mask: Option<Tensor<B, 2>>,
[CODE L0028]     pub safety_residual_target: Option<Tensor<B, 2>>,
[CODE L0029]     pub safety_residual_mask: Option<Tensor<B, 2>>,
[CODE L0030]     pub oracle_guidance_mask: Option<Tensor<B, 1>>,
[CODE L0031]     pub target_presence: Option<TargetPresence>,
[CODE L0032] }
[CODE L0033] 
[CODE L0034] impl<B: Backend> HydraTargets<B> {
[CODE L0035]     /// Slice all target tensors along the batch dimension (dim 0).
[CODE L0036]     ///
[CODE L0037]     /// Produces a sub-batch covering `[start..end)`. Used by microbatch
[CODE L0038]     /// accumulation to split a full RL batch into VRAM-friendly chunks.
[CODE L0039]     #[allow(clippy::single_range_in_vec_init)]
[CODE L0040]     pub fn slice_batch(&self, start: usize, end: usize) -> Self {
[CODE L0041]         let r1 = [start..end];
[CODE L0042]         let r2 = [start..end];
[CODE L0043]         let r3 = [start..end];
[CODE L0044]         Self {
[CODE L0045]             policy_target: self.policy_target.clone().slice(r1.clone()),
[CODE L0046]             legal_mask: self.legal_mask.clone().slice(r1.clone()),
[CODE L0047]             value_target: self.value_target.clone().slice(r2.clone()),
[CODE L0048]             grp_target: self.grp_target.clone().slice(r1.clone()),
[CODE L0049]             tenpai_target: self.tenpai_target.clone().slice(r1.clone()),
[CODE L0050]             danger_target: self.danger_target.clone().slice(r3.clone()),
[CODE L0051]             danger_mask: self.danger_mask.clone().slice(r3.clone()),
[CODE L0052]             opp_next_target: self.opp_next_target.clone().slice(r3.clone()),
[CODE L0053]             score_pdf_target: self.score_pdf_target.clone().slice(r1.clone()),
[CODE L0054]             score_cdf_target: self.score_cdf_target.clone().slice(r1.clone()),
[CODE L0055]             oracle_target: self
[CODE L0056]                 .oracle_target
[CODE L0057]                 .as_ref()
[CODE L0058]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0059]             belief_fields_target: self
[CODE L0060]                 .belief_fields_target
[CODE L0061]                 .as_ref()
[CODE L0062]                 .map(|t| t.clone().slice(r3.clone())),
[CODE L0063]             belief_fields_mask: self
[CODE L0064]                 .belief_fields_mask
[CODE L0065]                 .as_ref()
[CODE L0066]                 .map(|t| t.clone().slice(r2.clone())),
[CODE L0067]             mixture_weight_target: self
[CODE L0068]                 .mixture_weight_target
[CODE L0069]                 .as_ref()
[CODE L0070]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0071]             mixture_weight_mask: self
[CODE L0072]                 .mixture_weight_mask
[CODE L0073]                 .as_ref()
[CODE L0074]                 .map(|t| t.clone().slice(r2.clone())),
[CODE L0075]             opponent_hand_type_target: self
[CODE L0076]                 .opponent_hand_type_target
[CODE L0077]                 .as_ref()
[CODE L0078]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0079]             delta_q_target: self
[CODE L0080]                 .delta_q_target
[CODE L0081]                 .as_ref()
[CODE L0082]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0083]             delta_q_mask: self
[CODE L0084]                 .delta_q_mask
[CODE L0085]                 .as_ref()
[CODE L0086]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0087]             safety_residual_target: self
[CODE L0088]                 .safety_residual_target
[CODE L0089]                 .as_ref()
[CODE L0090]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0091]             safety_residual_mask: self
[CODE L0092]                 .safety_residual_mask
[CODE L0093]                 .as_ref()
[CODE L0094]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0095]             oracle_guidance_mask: self
[CODE L0096]                 .oracle_guidance_mask
[CODE L0097]                 .as_ref()
[CODE L0098]                 .map(|t| t.clone().slice(r2)),
[CODE L0099]             target_presence: None,
[CODE L0100]         }
[CODE L0101]     }
[CODE L0102] }
[CODE L0103] 
[CODE L0104] #[derive(Config, Debug)]
[CODE L0105] pub struct HydraLossConfig {
[CODE L0106]     #[config(default = "1.0")]
[CODE L0107]     pub w_pi: f32,
[CODE L0108]     #[config(default = "0.5")]
[CODE L0109]     pub w_v: f32,
[CODE L0110]     #[config(default = "0.2")]
[CODE L0111]     pub w_grp: f32,
[CODE L0112]     #[config(default = "0.1")]
[CODE L0113]     pub w_tenpai: f32,
[CODE L0114]     #[config(default = "0.1")]
[CODE L0115]     pub w_danger: f32,
[CODE L0116]     #[config(default = "0.1")]
[CODE L0117]     pub w_opp: f32,
[CODE L0118]     #[config(default = "0.025")]
[CODE L0119]     pub w_score: f32,
[CODE L0120]     #[config(default = "0.0")]
[CODE L0121]     pub w_oracle_critic: f32,
[CODE L0122]     #[config(default = "0.0")]
[CODE L0123]     pub w_belief_fields: f32,
[CODE L0124]     #[config(default = "0.0")]
[CODE L0125]     pub w_mixture_weight: f32,
[CODE L0126]     #[config(default = "0.0")]
[CODE L0127]     pub w_opponent_hand_type: f32,
[CODE L0128]     #[config(default = "0.0")]
[CODE L0129]     pub w_delta_q: f32,
[CODE L0130]     #[config(default = "0.0")]
[CODE L0131]     pub w_safety_residual: f32,
[CODE L0132] }
[CODE L0133] 
[CODE L0134] impl HydraLossConfig {
[CODE L0135]     pub fn total_weight(&self) -> f32 {
[CODE L0136]         self.w_pi
[CODE L0137]             + self.w_v
[CODE L0138]             + self.w_grp
[CODE L0139]             + self.w_tenpai
[CODE L0140]             + self.w_danger
[CODE L0141]             + self.w_opp
[CODE L0142]             + self.w_score * 2.0
[CODE L0143]             + self.w_oracle_critic
[CODE L0144]             + self.w_belief_fields
[CODE L0145]             + self.w_mixture_weight
[CODE L0146]             + self.w_opponent_hand_type
[CODE L0147]             + self.w_delta_q
[CODE L0148]             + self.w_safety_residual
[CODE L0149]     }
[CODE L0150] 
[CODE L0151]     pub fn scale_all(&self, factor: f32) -> Self {
[CODE L0152]         Self::new()
[CODE L0153]             .with_w_pi(self.w_pi * factor)
[CODE L0154]             .with_w_v(self.w_v * factor)
[CODE L0155]             .with_w_grp(self.w_grp * factor)
[CODE L0156]             .with_w_tenpai(self.w_tenpai * factor)
[CODE L0157]             .with_w_danger(self.w_danger * factor)
[CODE L0158]             .with_w_opp(self.w_opp * factor)
[CODE L0159]             .with_w_score(self.w_score * factor)
[CODE L0160]             .with_w_oracle_critic(self.w_oracle_critic * factor)
[CODE L0161]             .with_w_belief_fields(self.w_belief_fields * factor)
[CODE L0162]             .with_w_mixture_weight(self.w_mixture_weight * factor)
[CODE L0163]             .with_w_opponent_hand_type(self.w_opponent_hand_type * factor)
[CODE L0164]             .with_w_delta_q(self.w_delta_q * factor)
[CODE L0165]             .with_w_safety_residual(self.w_safety_residual * factor)
[CODE L0166]     }
[CODE L0167] 
[CODE L0168]     pub fn summary(&self) -> String {
[CODE L0169]         format!(
[CODE L0170]             "loss(pi={:.1}, v={:.1}, grp={:.1})",
[CODE L0171]             self.w_pi, self.w_v, self.w_grp
[CODE L0172]         )
[CODE L0173]     }
[CODE L0174] 
[CODE L0175]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0176]         if self.w_pi < 0.0
[CODE L0177]             || self.w_v < 0.0
[CODE L0178]             || self.w_grp < 0.0
[CODE L0179]             || self.w_tenpai < 0.0
[CODE L0180]             || self.w_danger < 0.0
[CODE L0181]             || self.w_opp < 0.0
[CODE L0182]             || self.w_score < 0.0
[CODE L0183]             || self.w_oracle_critic < 0.0
[CODE L0184]             || self.w_belief_fields < 0.0
[CODE L0185]             || self.w_mixture_weight < 0.0
[CODE L0186]             || self.w_opponent_hand_type < 0.0
[CODE L0187]             || self.w_delta_q < 0.0
[CODE L0188]             || self.w_safety_residual < 0.0
[CODE L0189]         {
[CODE L0190]             return Err("loss weights must be non-negative");
[CODE L0191]         }
[CODE L0192]         Ok(())
[CODE L0193]     }
[CODE L0194] }
[CODE L0195] 
[CODE L0196] pub struct HydraLoss<B: Backend> {
[CODE L0197]     pub config: HydraLossConfig,
[CODE L0198]     _backend: PhantomData<B>,
[CODE L0199] }
[CODE L0200] 
[CODE L0201] impl<B: Backend> HydraLoss<B> {
[CODE L0202]     pub fn new(config: HydraLossConfig) -> Self {
[CODE L0203]         Self {
[CODE L0204]             config,
[CODE L0205]             _backend: PhantomData,
[CODE L0206]         }
[CODE L0207]     }
[CODE L0208] }
[CODE L0209] 
[CODE L0210] const NEG_INF: f32 = -1e9;
[CODE L0211] 
[CODE L0212] pub fn policy_ce<B: Backend>(
[CODE L0213]     logits: Tensor<B, 2>,
[CODE L0214]     target: Tensor<B, 2>,
[CODE L0215]     mask: Tensor<B, 2>,
[CODE L0216] ) -> Tensor<B, 1> {
[CODE L0217]     let masked = logits + (mask.ones_like() - mask) * NEG_INF;
[CODE L0218]     let log_probs = activation::log_softmax(masked, 1);
[CODE L0219]     (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
[CODE L0220] }
[CODE L0221] 
[CODE L0222] pub fn value_mse<B: Backend>(pred: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1> {
[CODE L0223]     let diff = pred - target;
[CODE L0224]     diff.clone() * diff * 0.5
[CODE L0225] }
[CODE L0226] 
[CODE L0227] pub fn grp_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0228]     let log_probs = activation::log_softmax(logits, 1);
[CODE L0229]     (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
[CODE L0230] }
[CODE L0231] 
[CODE L0232] pub fn tenpai_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0233]     let loss = bce_with_logits(logits, target);
[CODE L0234]     loss.mean_dim(1).squeeze_dim::<1>(1)
[CODE L0235] }
[CODE L0236] 
[CODE L0237] pub fn danger_focal_bce<B: Backend>(
[CODE L0238]     logits: Tensor<B, 3>,
[CODE L0239]     target: Tensor<B, 3>,
[CODE L0240]     mask: Tensor<B, 3>,
[CODE L0241] ) -> Tensor<B, 1> {
[CODE L0242]     let alpha = 0.25f32;
[CODE L0243]     let gamma = 2.0f32;
[CODE L0244]     let p = activation::sigmoid(logits.clone());
[CODE L0245]     let bce = bce_with_logits_3d(logits, target.clone());
[CODE L0246]     let p_t = target.clone() * p.clone() + (target.ones_like() - target) * (p.ones_like() - p);
[CODE L0247]     let focal_weight = (p_t.ones_like() - p_t).powf_scalar(gamma) * alpha;
[CODE L0248]     let focal = focal_weight * bce * mask;
[CODE L0249]     let sum_per_sample = focal.sum_dim(2).sum_dim(1);
[CODE L0250]     sum_per_sample.squeeze_dim::<2>(2).squeeze_dim::<1>(1)
[CODE L0251] }
[CODE L0252] 
[CODE L0253] pub fn opp_next_ce<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
[CODE L0254]     let [batch, opps, tiles] = logits.dims();
[CODE L0255]     let logits_flat = logits.reshape([batch * opps, tiles]);
[CODE L0256]     let target_flat = target.reshape([batch * opps, tiles]);
[CODE L0257]     let log_probs = activation::log_softmax(logits_flat, 1);
[CODE L0258]     let per_sample = (target_flat * log_probs)
[CODE L0259]         .sum_dim(1)
[CODE L0260]         .neg()
[CODE L0261]         .squeeze_dim::<1>(1);
[CODE L0262]     per_sample
[CODE L0263]         .reshape([batch, opps])
[CODE L0264]         .mean_dim(1)
[CODE L0265]         .squeeze_dim::<1>(1)
[CODE L0266] }
[CODE L0267] 
[CODE L0268] pub fn score_pdf_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0269]     let log_probs = activation::log_softmax(logits, 1);
[CODE L0270]     (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
[CODE L0271] }
[CODE L0272] 
[CODE L0273] pub fn score_cdf_bce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0274]     let loss = bce_with_logits(logits, target);
[CODE L0275]     loss.mean_dim(1).squeeze_dim::<1>(1)
[CODE L0276] }
[CODE L0277] 
[CODE L0278] pub fn belief_fields_bce<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
[CODE L0279]     belief_fields_bce_per_sample(logits, target).mean()
[CODE L0280] }
[CODE L0281] 
[CODE L0282] pub fn belief_fields_bce_per_sample<B: Backend>(
[CODE L0283]     logits: Tensor<B, 3>,
[CODE L0284]     target: Tensor<B, 3>,
[CODE L0285] ) -> Tensor<B, 1> {
[CODE L0286]     let [batch, channels, tiles] = logits.dims();
[CODE L0287]     bce_with_logits_3d(logits, target)
[CODE L0288]         .reshape([batch, channels * tiles])
[CODE L0289]         .mean_dim(1)
[CODE L0290]         .squeeze_dim::<1>(1)
[CODE L0291] }
[CODE L0292] 
[CODE L0293] pub fn mixture_weight_ce<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0294]     mixture_weight_ce_per_sample(logits, target).mean()
[CODE L0295] }
[CODE L0296] 
[CODE L0297] pub fn mixture_weight_ce_per_sample<B: Backend>(
[CODE L0298]     logits: Tensor<B, 2>,
[CODE L0299]     target: Tensor<B, 2>,
[CODE L0300] ) -> Tensor<B, 1> {
[CODE L0301]     cross_entropy_soft(activation::log_softmax(logits, 1), target)
[CODE L0302] }
[CODE L0303] 
[CODE L0304] pub fn opponent_hand_type_ce<B: Backend>(
[CODE L0305]     logits: Tensor<B, 2>,
[CODE L0306]     target: Tensor<B, 2>,
[CODE L0307] ) -> Tensor<B, 1> {
[CODE L0308]     opponent_hand_type_ce_per_sample(logits, target).mean()
[CODE L0309] }
[CODE L0310] 
[CODE L0311] pub fn opponent_hand_type_ce_per_sample<B: Backend>(
[CODE L0312]     logits: Tensor<B, 2>,
[CODE L0313]     target: Tensor<B, 2>,
[CODE L0314] ) -> Tensor<B, 1> {
[CODE L0315]     cross_entropy_soft(activation::log_softmax(logits, 1), target)
[CODE L0316] }
[CODE L0317] 
[CODE L0318] pub fn dense_regression_mse<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0319]     let diff = pred - target;
[CODE L0320]     (diff.clone() * diff).mean() * 0.5
[CODE L0321] }
[CODE L0322] 
[CODE L0323] pub fn masked_action_mse<B: Backend>(
[CODE L0324]     pred: Tensor<B, 2>,
[CODE L0325]     target: Tensor<B, 2>,
[CODE L0326]     mask: Tensor<B, 2>,
[CODE L0327] ) -> Tensor<B, 1> {
[CODE L0328]     let diff = pred - target;
[CODE L0329]     let sq = diff.clone() * diff * 0.5;
[CODE L0330]     let masked = sq * mask.clone();
[CODE L0331]     let denom = mask.sum().clamp_min(1.0);
[CODE L0332]     masked.sum() / denom
[CODE L0333] }
[CODE L0334] 
[CODE L0335] pub fn compute_cvar(pdf: &[f32], alpha: f32) -> f32 {
[CODE L0336]     let n = pdf.len();
[CODE L0337]     if n == 0 || alpha <= 0.0 {
[CODE L0338]         return 0.0;
[CODE L0339]     }
[CODE L0340]     let mut cumsum = 0.0f32;
[CODE L0341]     let mut weighted_sum = 0.0f32;
[CODE L0342]     let bin_width = 1.0 / n as f32;
[CODE L0343]     for (i, &p) in pdf.iter().enumerate() {
[CODE L0344]         let next_cum = cumsum + p;
[CODE L0345]         if cumsum < alpha {
[CODE L0346]             let contrib = p.min(alpha - cumsum);
[CODE L0347]             let bin_center = (i as f32 + 0.5) * bin_width;
[CODE L0348]             weighted_sum += contrib * bin_center;
[CODE L0349]         }
[CODE L0350]         cumsum = next_cum;
[CODE L0351]     }
[CODE L0352]     if alpha > 0.0 {
[CODE L0353]         weighted_sum / alpha
[CODE L0354]     } else {
[CODE L0355]         0.0
[CODE L0356]     }
[CODE L0357] }
[CODE L0358] 
[CODE L0359] pub fn value_target_from_gae(gae_return: f32, value_baseline: f32, lambda_weight: f32) -> f32 {
[CODE L0360]     (lambda_weight * gae_return + (1.0 - lambda_weight) * value_baseline).clamp(-1.0, 1.0)
[CODE L0361] }
[CODE L0362] 
[CODE L0363] pub fn soft_target_from_exit<B: Backend>(
[CODE L0364]     model_logits: Tensor<B, 2>,
[CODE L0365]     exit_target: Tensor<B, 2>,
[CODE L0366]     mask: Tensor<B, 2>,
[CODE L0367]     mix: f32,
[CODE L0368] ) -> Tensor<B, 2> {
[CODE L0369]     let model_probs = burn::tensor::activation::softmax(
[CODE L0370]         model_logits + (mask.ones_like() - mask.clone()) * (-1e9f32),
[CODE L0371]         1,
[CODE L0372]     );
[CODE L0373]     model_probs * (1.0 - mix) + exit_target * mix
[CODE L0374] }
[CODE L0375] 
[CODE L0376] pub fn label_smoothing<B: Backend>(target: Tensor<B, 2>, alpha: f32) -> Tensor<B, 2> {
[CODE L0377]     let n = target.dims()[1] as f32;
[CODE L0378]     target * (1.0 - alpha) + (alpha / n)
[CODE L0379] }
[CODE L0380] 
[CODE L0381] pub fn policy_ce_with_temperature<B: Backend>(
[CODE L0382]     logits: Tensor<B, 2>,
[CODE L0383]     target: Tensor<B, 2>,
[CODE L0384]     mask: Tensor<B, 2>,
[CODE L0385]     temperature: f32,
[CODE L0386] ) -> Tensor<B, 1> {
[CODE L0387]     policy_ce(logits / temperature, target, mask)
[CODE L0388] }
[CODE L0389] 
[CODE L0390] pub fn loss_abs<B: Backend>(loss: &Tensor<B, 1>) -> f32 {
[CODE L0391]     loss.clone()
[CODE L0392]         .abs()
[CODE L0393]         .into_data()
[CODE L0394]         .convert::<f32>()
[CODE L0395]         .as_slice::<f32>()
[CODE L0396]         .expect("loss scalar should be readable as f32")[0]
[CODE L0397] }
[CODE L0398] 
[CODE L0399] pub fn loss_is_finite<B: Backend>(loss: &Tensor<B, 1>) -> bool {
[CODE L0400]     let v = loss
[CODE L0401]         .clone()
[CODE L0402]         .into_data()
[CODE L0403]         .convert::<f32>()
[CODE L0404]         .as_slice::<f32>()
[CODE L0405]         .expect("loss scalar should be readable as f32")[0];
[CODE L0406]     v.is_finite()
[CODE L0407] }
[CODE L0408] 
[CODE L0409] pub fn total_loss_scalar<B: Backend>(breakdown: &LossBreakdown<B>) -> f32 {
[CODE L0410]     breakdown
[CODE L0411]         .total
[CODE L0412]         .clone()
[CODE L0413]         .into_data()
[CODE L0414]         .convert::<f32>()
[CODE L0415]         .as_slice::<f32>()
[CODE L0416]         .expect("total loss scalar should be readable as f32")[0]
[CODE L0417] }
[CODE L0418] 
[CODE L0419] pub fn batch_kl_from_target<B: Backend>(
[CODE L0420]     logits: Tensor<B, 2>,
[CODE L0421]     mask: Tensor<B, 2>,
[CODE L0422]     target: Tensor<B, 2>,
[CODE L0423] ) -> Tensor<B, 1> {
[CODE L0424]     let log_probs = masked_log_softmax(logits, mask);
[CODE L0425]     let probs = log_probs.clone().exp();
[CODE L0426]     kl_divergence(probs, target)
[CODE L0427] }
[CODE L0428] 
[CODE L0429] pub fn grad_norm_approx<B: Backend>(loss: Tensor<B, 1>) -> f32 {
[CODE L0430]     loss.abs()
[CODE L0431]         .into_data()
[CODE L0432]         .convert::<f32>()
[CODE L0433]         .as_slice::<f32>()
[CODE L0434]         .expect("grad norm scalar should be readable as f32")[0]
[CODE L0435] }
[CODE L0436] 
[CODE L0437] pub fn batch_value_variance<B: Backend>(values: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0438]     let mean = values.clone().mean_dim(0);
[CODE L0439]     let diff = values - mean;
[CODE L0440]     (diff.clone() * diff).mean_dim(0).squeeze_dim::<1>(0)
[CODE L0441] }
[CODE L0442] 
[CODE L0443] pub fn batch_policy_entropy<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0444]     let log_probs = masked_log_softmax(logits, mask.clone());
[CODE L0445]     let probs = log_probs.clone().exp();
[CODE L0446]     (probs * log_probs * mask).sum_dim(1).neg().mean()
[CODE L0447] }
[CODE L0448] 
[CODE L0449] pub fn mean_entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0450]     entropy(probs).mean()
[CODE L0451] }
[CODE L0452] 
[CODE L0453] pub fn masked_log_softmax<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0454]     let neg_inf = (mask.ones_like() - mask) * (-1e9f32);
[CODE L0455]     burn::tensor::activation::log_softmax(logits + neg_inf, 1)
[CODE L0456] }
[CODE L0457] 
[CODE L0458] pub fn cross_entropy_soft<B: Backend>(
[CODE L0459]     log_probs: Tensor<B, 2>,
[CODE L0460]     target: Tensor<B, 2>,
[CODE L0461] ) -> Tensor<B, 1> {
[CODE L0462]     (target * log_probs).sum_dim(1).neg().squeeze_dim::<1>(1)
[CODE L0463] }
[CODE L0464] 
[CODE L0465] pub fn entropy<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0466]     let eps = 1e-8f32;
[CODE L0467]     let safe = probs.clone().clamp(eps, 1.0);
[CODE L0468]     (probs * safe.log()).sum_dim(1).neg().squeeze_dim::<1>(1)
[CODE L0469] }
[CODE L0470] 
[CODE L0471] pub fn kl_divergence<B: Backend>(p: Tensor<B, 2>, q: Tensor<B, 2>) -> Tensor<B, 1> {
[CODE L0472]     let eps = 1e-8f32;
[CODE L0473]     let p_safe = p.clone().clamp(eps, 1.0);
[CODE L0474]     let q_safe = q.clamp(eps, 1.0);
[CODE L0475]     (p * (p_safe.log() - q_safe.log()))
[CODE L0476]         .sum_dim(1)
[CODE L0477]         .squeeze_dim::<1>(1)
[CODE L0478] }
[CODE L0479] 
[CODE L0480] pub fn oracle_target_from_scores(final_scores: [i32; 4]) -> [f32; 4] {
[CODE L0481]     let mean = final_scores.iter().sum::<i32>() as f32 / 4.0;
[CODE L0482]     let mut target = [0.0f32; 4];
[CODE L0483]     for (i, &s) in final_scores.iter().enumerate() {
[CODE L0484]         target[i] = (s as f32 - mean) / 100_000.0;
[CODE L0485]     }
[CODE L0486]     target
[CODE L0487] }
[CODE L0488] 
[CODE L0489] pub fn oracle_critic_loss<B: Backend>(
[CODE L0490]     v_oracle: Tensor<B, 2>,
[CODE L0491]     target: Tensor<B, 2>,
[CODE L0492] ) -> Tensor<B, 1> {
[CODE L0493]     oracle_critic_loss_per_sample(v_oracle, target).mean()
[CODE L0494] }
[CODE L0495] 
[CODE L0496] pub fn oracle_critic_loss_per_sample<B: Backend>(
[CODE L0497]     v_oracle: Tensor<B, 2>,
[CODE L0498]     target: Tensor<B, 2>,
[CODE L0499] ) -> Tensor<B, 1> {
[CODE L0500]     let v_norm = v_oracle.clone() - v_oracle.clone().mean_dim(1);
[CODE L0501]     let diff = v_norm - target;
[CODE L0502]     let mse = (diff.clone() * diff).mean_dim(1).squeeze_dim::<1>(1) * 0.5;
[CODE L0503]     let zero_sum_penalty = v_oracle.sum_dim(1).squeeze_dim::<1>(1);
[CODE L0504]     let zero_sum_penalty = zero_sum_penalty.clone() * zero_sum_penalty * 10.0;
[CODE L0505]     mse + zero_sum_penalty
[CODE L0506] }
[CODE L0507] 
[CODE L0508] fn bce_with_logits<B: Backend>(logits: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0509]     let max_val = logits.clone().clamp_min(0.0);
[CODE L0510]     let neg_abs = logits.clone().abs().neg();
[CODE L0511]     max_val - logits * target + neg_abs.exp().add_scalar(1.0).log()
[CODE L0512] }
[CODE L0513] 
[CODE L0514] fn bce_with_logits_3d<B: Backend>(logits: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 3> {
[CODE L0515]     let max_val = logits.clone().clamp_min(0.0);
[CODE L0516]     let neg_abs = logits.clone().abs().neg();
[CODE L0517]     max_val - logits * target + neg_abs.exp().add_scalar(1.0).log()
[CODE L0518] }
[CODE L0519] 
[CODE L0520] fn masked_mean<B: Backend>(per_sample: Tensor<B, 1>, mask: Option<Tensor<B, 1>>) -> Tensor<B, 1> {
[CODE L0521]     match mask {
[CODE L0522]         Some(mask) => {
[CODE L0523]             let denom = mask.clone().sum().clamp_min(1.0);
[CODE L0524]             (per_sample * mask).sum() / denom
[CODE L0525]         }
[CODE L0526]         None => per_sample.mean(),
[CODE L0527]     }
[CODE L0528] }
[CODE L0529] 
[CODE L0530] fn combine_sample_masks<B: Backend>(
[CODE L0531]     primary: Option<Tensor<B, 1>>,
[CODE L0532]     secondary: Option<Tensor<B, 1>>,
[CODE L0533] ) -> Option<Tensor<B, 1>> {
[CODE L0534]     match (primary, secondary) {
[CODE L0535]         (Some(primary), Some(secondary)) => Some(primary * secondary),
[CODE L0536]         (Some(primary), None) => Some(primary),
[CODE L0537]         (None, Some(secondary)) => Some(secondary),
[CODE L0538]         (None, None) => None,
[CODE L0539]     }
[CODE L0540] }
[CODE L0541] 
[CODE L0542] pub struct LossBreakdown<B: Backend> {
[CODE L0543]     pub policy: Tensor<B, 1>,
[CODE L0544]     pub value: Tensor<B, 1>,
[CODE L0545]     pub grp: Tensor<B, 1>,
[CODE L0546]     pub tenpai: Tensor<B, 1>,
[CODE L0547]     pub danger: Tensor<B, 1>,
[CODE L0548]     pub opp_next: Tensor<B, 1>,
[CODE L0549]     pub score_pdf: Tensor<B, 1>,
[CODE L0550]     pub score_cdf: Tensor<B, 1>,
[CODE L0551]     pub oracle_critic: Tensor<B, 1>,
[CODE L0552]     pub belief_fields: Tensor<B, 1>,
[CODE L0553]     pub mixture_weight: Tensor<B, 1>,
[CODE L0554]     pub opponent_hand_type: Tensor<B, 1>,
[CODE L0555]     pub delta_q: Tensor<B, 1>,
[CODE L0556]     pub safety_residual: Tensor<B, 1>,
[CODE L0557]     pub total: Tensor<B, 1>,
[CODE L0558] }
[CODE L0559] 
[CODE L0560] impl<B: Backend> LossBreakdown<B> {
[CODE L0561]     pub fn all_finite(&self) -> bool {
[CODE L0562]         let metrics = Tensor::cat(
[CODE L0563]             vec![
[CODE L0564]                 self.policy.clone(),
[CODE L0565]                 self.value.clone(),
[CODE L0566]                 self.grp.clone(),
[CODE L0567]                 self.tenpai.clone(),
[CODE L0568]                 self.danger.clone(),
[CODE L0569]                 self.opp_next.clone(),
[CODE L0570]                 self.score_pdf.clone(),
[CODE L0571]                 self.score_cdf.clone(),
[CODE L0572]                 self.oracle_critic.clone(),
[CODE L0573]                 self.belief_fields.clone(),
[CODE L0574]                 self.mixture_weight.clone(),
[CODE L0575]                 self.opponent_hand_type.clone(),
[CODE L0576]                 self.delta_q.clone(),
[CODE L0577]                 self.safety_residual.clone(),
[CODE L0578]                 self.total.clone(),
[CODE L0579]             ],
[CODE L0580]             0,
[CODE L0581]         )
[CODE L0582]         .into_data()
[CODE L0583]         .convert::<f32>();
[CODE L0584]         metrics
[CODE L0585]             .as_slice::<f32>()
[CODE L0586]             .expect("loss breakdown scalars should be readable as f32")
[CODE L0587]             .iter()
[CODE L0588]             .all(|v| v.is_finite())
[CODE L0589]     }
[CODE L0590] }
[CODE L0591] 
[CODE L0592] impl<B: Backend> HydraLoss<B> {
[CODE L0593]     pub fn total_loss(
[CODE L0594]         &self,
[CODE L0595]         outputs: &HydraOutput<B>,
[CODE L0596]         targets: &HydraTargets<B>,
[CODE L0597]     ) -> LossBreakdown<B> {
[CODE L0598]         let oracle_mask = targets.oracle_guidance_mask.clone();
[CODE L0599]         let l_pi = policy_ce(
[CODE L0600]             outputs.policy_logits.clone(),
[CODE L0601]             targets.policy_target.clone(),
[CODE L0602]             targets.legal_mask.clone(),
[CODE L0603]         )
[CODE L0604]         .mean();
[CODE L0605]         let l_v = value_mse(
[CODE L0606]             outputs.value.clone().squeeze_dim::<1>(1),
[CODE L0607]             targets.value_target.clone(),
[CODE L0608]         )
[CODE L0609]         .mean();
[CODE L0610]         let l_grp = grp_ce(outputs.grp.clone(), targets.grp_target.clone()).mean();
[CODE L0611]         let l_tenpai = tenpai_bce(outputs.opp_tenpai.clone(), targets.tenpai_target.clone()).mean();
[CODE L0612]         let l_danger = danger_focal_bce(
[CODE L0613]             outputs.danger.clone(),
[CODE L0614]             targets.danger_target.clone(),
[CODE L0615]             targets.danger_mask.clone(),
[CODE L0616]         )
[CODE L0617]         .mean();
[CODE L0618]         let l_opp = opp_next_ce(
[CODE L0619]             outputs.opp_next_discard.clone(),
[CODE L0620]             targets.opp_next_target.clone(),
[CODE L0621]         )
[CODE L0622]         .mean();
[CODE L0623]         let l_pdf =
[CODE L0624]             score_pdf_ce(outputs.score_pdf.clone(), targets.score_pdf_target.clone()).mean();
[CODE L0625]         let l_cdf =
[CODE L0626]             score_cdf_bce(outputs.score_cdf.clone(), targets.score_cdf_target.clone()).mean();
[CODE L0627]         let zero = outputs.value.clone().sum() * 0.0;
[CODE L0628]         let l_oracle = match &targets.oracle_target {
[CODE L0629]             Some(target) => masked_mean(
[CODE L0630]                 oracle_critic_loss_per_sample(outputs.oracle_critic.clone(), target.clone()),
[CODE L0631]                 oracle_mask.clone(),
[CODE L0632]             ),
[CODE L0633]             None => zero.clone(),
[CODE L0634]         };
[CODE L0635]         let l_belief = match (&targets.belief_fields_target, &targets.belief_fields_mask) {
[CODE L0636]             (Some(target), Some(mask)) => masked_mean(
[CODE L0637]                 belief_fields_bce_per_sample(outputs.belief_fields.clone(), target.clone()),
[CODE L0638]                 combine_sample_masks(Some(mask.clone()), oracle_mask.clone()),
[CODE L0639]             ),
[CODE L0640]             _ => zero.clone(),
[CODE L0641]         };
[CODE L0642]         let l_mix = match (&targets.mixture_weight_target, &targets.mixture_weight_mask) {
[CODE L0643]             (Some(target), Some(mask)) => masked_mean(
[CODE L0644]                 mixture_weight_ce_per_sample(outputs.mixture_weight_logits.clone(), target.clone()),
[CODE L0645]                 combine_sample_masks(Some(mask.clone()), oracle_mask.clone()),
[CODE L0646]             ),
[CODE L0647]             _ => zero.clone(),
[CODE L0648]         };
[CODE L0649]         let l_hand_type = match &targets.opponent_hand_type_target {
[CODE L0650]             Some(target) => masked_mean(
[CODE L0651]                 opponent_hand_type_ce_per_sample(
[CODE L0652]                     outputs.opponent_hand_type.clone(),
[CODE L0653]                     target.clone(),
[CODE L0654]                 ),
[CODE L0655]                 oracle_mask.clone(),
[CODE L0656]             ),
[CODE L0657]             None => zero.clone(),
[CODE L0658]         };
[CODE L0659]         let l_delta_q = match (&targets.delta_q_target, &targets.delta_q_mask) {
[CODE L0660]             (Some(target), Some(mask)) => {
[CODE L0661]                 masked_action_mse(outputs.delta_q.clone(), target.clone(), mask.clone())
[CODE L0662]             }
[CODE L0663]             _ => zero.clone(),
[CODE L0664]         };
[CODE L0665]         let l_safety_residual = match (
[CODE L0666]             &targets.safety_residual_target,
[CODE L0667]             &targets.safety_residual_mask,
[CODE L0668]         ) {
[CODE L0669]             (Some(target), Some(mask)) => masked_action_mse(
[CODE L0670]                 outputs.safety_residual.clone(),
[CODE L0671]                 target.clone(),
[CODE L0672]                 mask.clone(),
[CODE L0673]             ),
[CODE L0674]             _ => zero.clone(),
[CODE L0675]         };
[CODE L0676]         let c = &self.config;
[CODE L0677]         let total = l_pi.clone() * c.w_pi
[CODE L0678]             + l_v.clone() * c.w_v
[CODE L0679]             + l_grp.clone() * c.w_grp
[CODE L0680]             + l_tenpai.clone() * c.w_tenpai
[CODE L0681]             + l_danger.clone() * c.w_danger
[CODE L0682]             + l_opp.clone() * c.w_opp
[CODE L0683]             + l_pdf.clone() * c.w_score
[CODE L0684]             + l_cdf.clone() * c.w_score
[CODE L0685]             + l_oracle.clone() * c.w_oracle_critic
[CODE L0686]             + l_belief.clone() * c.w_belief_fields
[CODE L0687]             + l_mix.clone() * c.w_mixture_weight
[CODE L0688]             + l_hand_type.clone() * c.w_opponent_hand_type
[CODE L0689]             + l_delta_q.clone() * c.w_delta_q
[CODE L0690]             + l_safety_residual.clone() * c.w_safety_residual;
[CODE L0691]         LossBreakdown {
[CODE L0692]             policy: l_pi,
[CODE L0693]             value: l_v,
[CODE L0694]             grp: l_grp,
[CODE L0695]             tenpai: l_tenpai,
[CODE L0696]             danger: l_danger,
[CODE L0697]             opp_next: l_opp,
[CODE L0698]             score_pdf: l_pdf,
[CODE L0699]             score_cdf: l_cdf,
[CODE L0700]             oracle_critic: l_oracle,
[CODE L0701]             belief_fields: l_belief,
[CODE L0702]             mixture_weight: l_mix,
[CODE L0703]             opponent_hand_type: l_hand_type,
[CODE L0704]             delta_q: l_delta_q,
[CODE L0705]             safety_residual: l_safety_residual,
[CODE L0706]             total,
[CODE L0707]         }
[CODE L0708]     }
[CODE L0709] }
[CODE L0710] 
[CODE L0711] #[cfg(test)]
[CODE L0712] pub mod tests {
[CODE L0713]     use super::*;
[CODE L0714]     use crate::model::HydraModelConfig;
[CODE L0715]     use burn::backend::NdArray;
[CODE L0716] 
[CODE L0717]     type B = NdArray<f32>;
[CODE L0718] 
[CODE L0719]     #[test]
[CODE L0720]     fn test_policy_ce_with_mask() {
[CODE L0721]         let device = Default::default();
[CODE L0722]         let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0, -1.0]], &device);
[CODE L0723]         let mut mask_data = [1.0f32; 4];
[CODE L0724]         mask_data[3] = 0.0;
[CODE L0725]         let mask = Tensor::<B, 2>::from_floats([mask_data], &device);
[CODE L0726]         let target = Tensor::<B, 2>::from_floats([[0.0, 0.0, 1.0, 0.0]], &device);
[CODE L0727]         let loss = policy_ce(logits, target, mask);
[CODE L0728]         let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L0729]         assert!(val > 0.0, "policy CE should be positive, got {val}");
[CODE L0730]         assert!(val < 5.0, "policy CE too large: {val}");
[CODE L0731]     }
[CODE L0732] 
[CODE L0733]     #[test]
[CODE L0734]     fn test_policy_ce_illegal_action_zero_gradient() {
[CODE L0735]         let device = Default::default();
[CODE L0736]         let logits = Tensor::<B, 2>::from_floats([[10.0, -10.0, 0.0]], &device);
[CODE L0737]         let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
[CODE L0738]         let target = Tensor::<B, 2>::from_floats([[0.5, 0.0, 0.5]], &device);
[CODE L0739]         let loss = policy_ce(logits.clone(), target, mask);
[CODE L0740]         let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L0741]         assert!(val.is_finite(), "masked loss should be finite: {val}");
[CODE L0742]     }
[CODE L0743] 
[CODE L0744]     #[test]
[CODE L0745]     fn test_soft_target_differs_from_hard() {
[CODE L0746]         let device = Default::default();
[CODE L0747]         let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 0.5]], &device);
[CODE L0748]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0749]         let hard = Tensor::<B, 2>::from_floats([[0.0, 1.0, 0.0]], &device);
[CODE L0750]         let soft = Tensor::<B, 2>::from_floats([[0.3, 0.7, 0.0]], &device);
[CODE L0751]         let l_hard = policy_ce(logits.clone(), hard, mask.clone());
[CODE L0752]         let l_soft = policy_ce(logits, soft, mask);
[CODE L0753]         let h = l_hard.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L0754]         let s = l_soft.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L0755]         assert!(
[CODE L0756]             (h - s).abs() > 0.01,
[CODE L0757]             "soft vs hard should differ: {h} vs {s}"
[CODE L0758]         );
[CODE L0759]     }
[CODE L0760] 
[CODE L0761]     #[test]
[CODE L0762]     fn test_oracle_critic_zero_sum() {
[CODE L0763]         let device = Default::default();
[CODE L0764]         let v = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
[CODE L0765]         let target = Tensor::<B, 2>::from_floats([[1.0, -1.0, 2.0, -2.0]], &device);
[CODE L0766]         let loss = oracle_critic_loss(v, target);
[CODE L0767]         let val = loss.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L0768]         assert!(
[CODE L0769]             val.abs() < 1e-4,
[CODE L0770]             "zero-sum input should give near-zero loss, got {val}"
[CODE L0771]         );
[CODE L0772]     }
[CODE L0773] 
[CODE L0774]     #[test]
[CODE L0775]     fn test_oracle_target_zero_sum() {
[CODE L0776]         let target = oracle_target_from_scores([30000, 25000, 25000, 20000]);
[CODE L0777]         let sum: f32 = target.iter().sum();
[CODE L0778]         assert!(sum.abs() < 1e-5, "oracle target should be zero-sum: {sum}");
[CODE L0779]         assert!(target[0] > 0.0, "1st place should be positive");
[CODE L0780]         assert!(target[3] < 0.0, "4th place should be negative");
[CODE L0781]     }
[CODE L0782] 
[CODE L0783]     #[test]
[CODE L0784]     fn test_oracle_target_populates_breakdown_only() {
[CODE L0785]         let device = Default::default();
[CODE L0786]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0787]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0788]         let outputs = model.forward(x);
[CODE L0789]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L0790]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L0791]         targets.oracle_target = Some(Tensor::<B, 2>::zeros([2, 4], &device));
[CODE L0792]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L0793]         let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
[CODE L0794]         let total_loss: f32 = breakdown.total.into_scalar().elem();
[CODE L0795]         assert!(oracle_loss.is_finite() && oracle_loss >= 0.0);
[CODE L0796]         assert!(total_loss.is_finite() && total_loss >= 0.0);
[CODE L0797]     }
[CODE L0798] 
[CODE L0799]     #[test]
[CODE L0800]     fn test_oracle_absent_with_mask_keeps_oracle_loss_zero() {
[CODE L0801]         let device = Default::default();
[CODE L0802]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0803]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0804]         let outputs = model.forward(x);
[CODE L0805]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_oracle_critic(1.0));
[CODE L0806]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L0807]         targets.oracle_target = None;
[CODE L0808]         targets.oracle_guidance_mask = Some(Tensor::<B, 1>::zeros([2], &device));
[CODE L0809]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L0810]         let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
[CODE L0811]         let total_loss: f32 = breakdown.total.into_scalar().elem();
[CODE L0812]         assert!(
[CODE L0813]             oracle_loss.abs() < 1e-8,
[CODE L0814]             "oracle loss should be zero when target absent"
[CODE L0815]         );
[CODE L0816]         assert!(total_loss.is_finite() && total_loss >= 0.0);
[CODE L0817]     }
[CODE L0818] 
[CODE L0819]     #[test]
[CODE L0820]     fn test_oracle_target_contributes_to_total_when_weight_enabled() {
[CODE L0821]         let device = Default::default();
[CODE L0822]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0823]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0824]         let outputs = model.forward(x);
[CODE L0825]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L0826]         targets.oracle_target = Some(Tensor::<B, 2>::ones([2, 4], &device));
[CODE L0827] 
[CODE L0828]         let base = HydraLoss::<B>::new(HydraLossConfig::new()).total_loss(&outputs, &targets);
[CODE L0829]         let with_oracle = HydraLoss::<B>::new(HydraLossConfig::new().with_w_oracle_critic(1.0))
[CODE L0830]             .total_loss(&outputs, &targets);
[CODE L0831] 
[CODE L0832]         let total_base: f32 = base.total.into_scalar().elem();
[CODE L0833]         let total_oracle: f32 = with_oracle.total.into_scalar().elem();
[CODE L0834]         let oracle_loss: f32 = with_oracle.oracle_critic.into_scalar().elem();
[CODE L0835]         assert!(oracle_loss > 0.0, "oracle loss should be active");
[CODE L0836]         assert!(
[CODE L0837]             total_oracle > total_base,
[CODE L0838]             "oracle weighting should raise total loss"
[CODE L0839]         );
[CODE L0840]     }
[CODE L0841] 
[CODE L0842]     #[test]
[CODE L0843]     fn test_oracle_guidance_mask_disables_masked_optional_losses() {
[CODE L0844]         let device = Default::default();
[CODE L0845]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0846]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0847]         let outputs = model.forward(x);
[CODE L0848]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L0849]         targets.oracle_target = Some(Tensor::<B, 2>::ones([2, 4], &device));
[CODE L0850]         targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
[CODE L0851]         targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
[CODE L0852]             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
[CODE L0853]             &device,
[CODE L0854]         ));
[CODE L0855]         targets.opponent_hand_type_target = Some(Tensor::<B, 2>::from_floats(
[CODE L0856]             [
[CODE L0857]                 [
[CODE L0858]                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L0859]                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L0860]                 ],
[CODE L0861]                 [
[CODE L0862]                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L0863]                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L0864]                 ],
[CODE L0865]             ],
[CODE L0866]             &device,
[CODE L0867]         ));
[CODE L0868]         targets.oracle_guidance_mask = Some(Tensor::<B, 1>::zeros([2], &device));
[CODE L0869] 
[CODE L0870]         let loss_fn = HydraLoss::<B>::new(
[CODE L0871]             HydraLossConfig::new()
[CODE L0872]                 .with_w_oracle_critic(1.0)
[CODE L0873]                 .with_w_belief_fields(1.0)
[CODE L0874]                 .with_w_mixture_weight(1.0)
[CODE L0875]                 .with_w_opponent_hand_type(1.0),
[CODE L0876]         );
[CODE L0877]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L0878]         assert!(breakdown.oracle_critic.into_scalar().elem::<f32>().abs() < 1e-8);
[CODE L0879]         assert!(breakdown.belief_fields.into_scalar().elem::<f32>().abs() < 1e-8);
[CODE L0880]         assert!(breakdown.mixture_weight.into_scalar().elem::<f32>().abs() < 1e-8);
[CODE L0881]         assert!(
[CODE L0882]             breakdown
[CODE L0883]                 .opponent_hand_type
[CODE L0884]                 .into_scalar()
[CODE L0885]                 .elem::<f32>()
[CODE L0886]                 .abs()
[CODE L0887]                 < 1e-8
[CODE L0888]         );
[CODE L0889]     }
[CODE L0890] 
[CODE L0891]     #[test]
[CODE L0892]     fn test_oracle_guidance_mask_intersects_belief_and_mixture_masks() {
[CODE L0893]         let device = Default::default();
[CODE L0894]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0895]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0896]         let outputs = model.forward(x);
[CODE L0897]         let mut first_only = make_dummy_targets::<B>(&device, 1);
[CODE L0898]         first_only.belief_fields_target = Some(Tensor::<B, 3>::ones([1, 16, 34], &device));
[CODE L0899]         first_only.belief_fields_mask = Some(Tensor::<B, 1>::ones([1], &device));
[CODE L0900]         first_only.mixture_weight_target =
[CODE L0901]             Some(Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 0.0]], &device));
[CODE L0902]         first_only.mixture_weight_mask = Some(Tensor::<B, 1>::ones([1], &device));
[CODE L0903] 
[CODE L0904]         let mut masked_targets = make_dummy_targets::<B>(&device, 2);
[CODE L0905]         masked_targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
[CODE L0906]         masked_targets.belief_fields_mask = Some(Tensor::<B, 1>::ones([2], &device));
[CODE L0907]         masked_targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
[CODE L0908]             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
[CODE L0909]             &device,
[CODE L0910]         ));
[CODE L0911]         masked_targets.mixture_weight_mask = Some(Tensor::<B, 1>::ones([2], &device));
[CODE L0912]         masked_targets.oracle_guidance_mask =
[CODE L0913]             Some(Tensor::<B, 1>::from_floats([1.0, 0.0], &device));
[CODE L0914] 
[CODE L0915]         let loss_fn = HydraLoss::<B>::new(
[CODE L0916]             HydraLossConfig::new()
[CODE L0917]                 .with_w_belief_fields(1.0)
[CODE L0918]                 .with_w_mixture_weight(1.0),
[CODE L0919]         );
[CODE L0920] 
[CODE L0921]         #[allow(clippy::single_range_in_vec_init)]
[CODE L0922]         let first_outputs = HydraOutput {
[CODE L0923]             policy_logits: outputs.policy_logits.clone().slice([0..1]),
[CODE L0924]             value: outputs.value.clone().slice([0..1]),
[CODE L0925]             grp: outputs.grp.clone().slice([0..1]),
[CODE L0926]             opp_tenpai: outputs.opp_tenpai.clone().slice([0..1]),
[CODE L0927]             danger: outputs.danger.clone().slice([0..1]),
[CODE L0928]             opp_next_discard: outputs.opp_next_discard.clone().slice([0..1]),
[CODE L0929]             score_pdf: outputs.score_pdf.clone().slice([0..1]),
[CODE L0930]             score_cdf: outputs.score_cdf.clone().slice([0..1]),
[CODE L0931]             oracle_critic: outputs.oracle_critic.clone().slice([0..1]),
[CODE L0932]             belief_fields: outputs.belief_fields.clone().slice([0..1]),
[CODE L0933]             mixture_weight_logits: outputs.mixture_weight_logits.clone().slice([0..1]),
[CODE L0934]             opponent_hand_type: outputs.opponent_hand_type.clone().slice([0..1]),
[CODE L0935]             delta_q: outputs.delta_q.clone().slice([0..1]),
[CODE L0936]             safety_residual: outputs.safety_residual.clone().slice([0..1]),
[CODE L0937]         };
[CODE L0938] 
[CODE L0939]         let first_breakdown = loss_fn.total_loss(&first_outputs, &first_only);
[CODE L0940]         let masked_breakdown = loss_fn.total_loss(&outputs, &masked_targets);
[CODE L0941]         let belief_first: f32 = first_breakdown.belief_fields.into_scalar().elem();
[CODE L0942]         let mixture_first: f32 = first_breakdown.mixture_weight.into_scalar().elem();
[CODE L0943]         let belief_with: f32 = masked_breakdown.belief_fields.into_scalar().elem();
[CODE L0944]         let mixture_with: f32 = masked_breakdown.mixture_weight.into_scalar().elem();
[CODE L0945] 
[CODE L0946]         assert!(belief_first.is_finite() && belief_first > 0.0);
[CODE L0947]         assert!(mixture_first.is_finite() && mixture_first > 0.0);
[CODE L0948]         assert!(belief_with.is_finite() && belief_with > 0.0);
[CODE L0949]         assert!(mixture_with.is_finite() && mixture_with > 0.0);
[CODE L0950]         assert!((belief_with - belief_first).abs() < 1e-6);
[CODE L0951]         assert!((mixture_with - mixture_first).abs() < 1e-6);
[CODE L0952]     }
[CODE L0953] 
[CODE L0954]     #[test]
[CODE L0955]     fn test_optional_belief_losses_require_presence_masks() {
[CODE L0956]         let device = Default::default();
[CODE L0957]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0958]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0959]         let outputs = model.forward(x);
[CODE L0960]         let loss_fn = HydraLoss::<B>::new(
[CODE L0961]             HydraLossConfig::new()
[CODE L0962]                 .with_w_belief_fields(1.0)
[CODE L0963]                 .with_w_mixture_weight(1.0),
[CODE L0964]         );
[CODE L0965]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L0966]         targets.belief_fields_target = Some(Tensor::<B, 3>::ones([2, 16, 34], &device));
[CODE L0967]         targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
[CODE L0968]             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
[CODE L0969]             &device,
[CODE L0970]         ));
[CODE L0971] 
[CODE L0972]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L0973]         assert!(breakdown.belief_fields.into_scalar().elem::<f32>().abs() < 1e-8);
[CODE L0974]         assert!(breakdown.mixture_weight.into_scalar().elem::<f32>().abs() < 1e-8);
[CODE L0975]     }
[CODE L0976] 
[CODE L0977]     #[test]
[CODE L0978]     fn test_optional_belief_losses_default_to_zero() {
[CODE L0979]         let device = Default::default();
[CODE L0980]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L0981]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L0982]         let outputs = model.forward(x);
[CODE L0983]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L0984]         let targets = make_dummy_targets::<B>(&device, 2);
[CODE L0985]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L0986]         let belief: f32 = breakdown.belief_fields.into_scalar().elem();
[CODE L0987]         let mixture: f32 = breakdown.mixture_weight.into_scalar().elem();
[CODE L0988]         let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
[CODE L0989]         let hand_type: f32 = breakdown.opponent_hand_type.into_scalar().elem();
[CODE L0990]         let delta_q: f32 = breakdown.delta_q.into_scalar().elem();
[CODE L0991]         let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
[CODE L0992]         assert!(
[CODE L0993]             oracle_loss.abs() < 1e-8,
[CODE L0994]             "missing oracle target should contribute zero oracle loss"
[CODE L0995]         );
[CODE L0996]         assert!(
[CODE L0997]             belief.abs() < 1e-8,
[CODE L0998]             "missing belief target should contribute zero loss"
[CODE L0999]         );
[CODE L1000]         assert!(
[CODE L1001]             mixture.abs() < 1e-8,
[CODE L1002]             "missing mixture target should contribute zero loss"
[CODE L1003]         );
[CODE L1004]         assert!(
[CODE L1005]             hand_type.abs() < 1e-8,
[CODE L1006]             "missing hand-type target should contribute zero loss"
[CODE L1007]         );
[CODE L1008]         assert!(
[CODE L1009]             delta_q.abs() < 1e-8,
[CODE L1010]             "missing delta-q target should contribute zero loss"
[CODE L1011]         );
[CODE L1012]         assert!(
[CODE L1013]             safety_residual.abs() < 1e-8,
[CODE L1014]             "missing safety-residual target should contribute zero loss"
[CODE L1015]         );
[CODE L1016]     }
[CODE L1017] 
[CODE L1018]     #[test]
[CODE L1019]     fn test_optional_belief_losses_activate_when_targets_present() {
[CODE L1020]         let device = Default::default();
[CODE L1021]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1022]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1023]         let outputs = model.forward(x);
[CODE L1024]         let loss_fn = HydraLoss::<B>::new(
[CODE L1025]             HydraLossConfig::new()
[CODE L1026]                 .with_w_belief_fields(0.1)
[CODE L1027]                 .with_w_mixture_weight(0.1)
[CODE L1028]                 .with_w_opponent_hand_type(0.1)
[CODE L1029]                 .with_w_delta_q(0.1)
[CODE L1030]                 .with_w_safety_residual(0.1),
[CODE L1031]         );
[CODE L1032]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L1033]         targets.belief_fields_target = Some(Tensor::<B, 3>::zeros([2, 16, 34], &device));
[CODE L1034]         targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
[CODE L1035]             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
[CODE L1036]             &device,
[CODE L1037]         ));
[CODE L1038]         targets.opponent_hand_type_target = Some(Tensor::<B, 2>::from_floats(
[CODE L1039]             [
[CODE L1040]                 [
[CODE L1041]                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L1042]                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L1043]                 ],
[CODE L1044]                 [
[CODE L1045]                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L1046]                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
[CODE L1047]                 ],
[CODE L1048]             ],
[CODE L1049]             &device,
[CODE L1050]         ));
[CODE L1051]         targets.delta_q_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
[CODE L1052]         targets.safety_residual_target = Some(Tensor::<B, 2>::from_floats(
[CODE L1053]             [[0.5f32; 46], [-0.5f32; 46]],
[CODE L1054]             &device,
[CODE L1055]         ));
[CODE L1056]         targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
[CODE L1057]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L1058]         let belief: f32 = breakdown.belief_fields.into_scalar().elem();
[CODE L1059]         let mixture: f32 = breakdown.mixture_weight.into_scalar().elem();
[CODE L1060]         let hand_type: f32 = breakdown.opponent_hand_type.into_scalar().elem();
[CODE L1061]         let delta_q: f32 = breakdown.delta_q.into_scalar().elem();
[CODE L1062]         let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
[CODE L1063]         let total: f32 = breakdown.total.into_scalar().elem();
[CODE L1064]         assert!(belief.is_finite() && belief >= 0.0);
[CODE L1065]         assert!(mixture.is_finite() && mixture >= 0.0);
[CODE L1066]         assert!(hand_type.is_finite() && hand_type >= 0.0);
[CODE L1067]         assert!(delta_q.is_finite() && delta_q >= 0.0);
[CODE L1068]         assert!(safety_residual.is_finite() && safety_residual >= 0.0);
[CODE L1069]         assert!(total.is_finite() && total > 0.0);
[CODE L1070]     }
[CODE L1071] 
[CODE L1072]     #[test]
[CODE L1073]     fn test_safety_residual_aux_loss_is_nonzero_when_enabled_and_present() {
[CODE L1074]         let device = Default::default();
[CODE L1075]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1076]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1077]         let outputs = model.forward(x);
[CODE L1078]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
[CODE L1079]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L1080]         targets.safety_residual_target = Some(Tensor::<B, 2>::from_floats(
[CODE L1081]             [[0.25f32; 46], [-0.75f32; 46]],
[CODE L1082]             &device,
[CODE L1083]         ));
[CODE L1084]         targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
[CODE L1085]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L1086]         let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
[CODE L1087]         assert!(
[CODE L1088]             safety_residual.is_finite() && safety_residual > 0.0,
[CODE L1089]             "signed safety residual targets with a mask should contribute nonzero aux loss"
[CODE L1090]         );
[CODE L1091]     }
[CODE L1092] 
[CODE L1093]     #[test]
[CODE L1094]     fn test_safety_residual_requires_mask() {
[CODE L1095]         let device = Default::default();
[CODE L1096]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1097]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1098]         let outputs = model.forward(x);
[CODE L1099]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
[CODE L1100]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L1101]         targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
[CODE L1102]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L1103]         let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
[CODE L1104]         assert!(
[CODE L1105]             safety_residual.abs() < 1e-8,
[CODE L1106]             "missing mask should disable safety residual loss"
[CODE L1107]         );
[CODE L1108]     }
[CODE L1109] 
[CODE L1110]     #[test]
[CODE L1111]     fn test_safety_residual_all_zero_mask_zeroes_loss() {
[CODE L1112]         let device = Default::default();
[CODE L1113]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1114]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1115]         let outputs = model.forward(x);
[CODE L1116]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(1.0));
[CODE L1117]         let mut targets = make_dummy_targets::<B>(&device, 2);
[CODE L1118]         targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
[CODE L1119]         targets.safety_residual_mask = Some(Tensor::<B, 2>::zeros([2, 46], &device));
[CODE L1120]         let breakdown = loss_fn.total_loss(&outputs, &targets);
[CODE L1121]         let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
[CODE L1122]         assert!(
[CODE L1123]             safety_residual.abs() < 1e-8,
[CODE L1124]             "zero mask should disable safety residual loss"
[CODE L1125]         );
[CODE L1126]     }
[CODE L1127] 
[CODE L1128]     #[test]
[CODE L1129]     fn test_total_loss_positive() {
[CODE L1130]         let device = Default::default();
[CODE L1131]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1132]         let x = Tensor::<B, 3>::random(
[CODE L1133]             [4, crate::config::INPUT_CHANNELS, 34],
[CODE L1134]             burn::tensor::Distribution::Normal(0.0, 0.1),
[CODE L1135]             &device,
[CODE L1136]         );
[CODE L1137]         let out = model.forward(x);
[CODE L1138]         let targets = make_dummy_targets::<B>(&device, 4);
[CODE L1139]         let hydra_loss = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L1140]         let breakdown = hydra_loss.total_loss(&out, &targets);
[CODE L1141]         let total = breakdown.total.to_data().as_slice::<f32>().expect("f32")[0];
[CODE L1142]         assert!(total > 0.0, "total loss should be positive, got {total}");
[CODE L1143]         assert!(total.is_finite(), "total loss should be finite");
[CODE L1144]     }
[CODE L1145] 
[CODE L1146]     #[test]
[CODE L1147]     fn test_loss_weights_configurable() {
[CODE L1148]         let device = Default::default();
[CODE L1149]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1150]         let x = Tensor::<B, 3>::random(
[CODE L1151]             [4, crate::config::INPUT_CHANNELS, 34],
[CODE L1152]             burn::tensor::Distribution::Normal(0.0, 0.1),
[CODE L1153]             &device,
[CODE L1154]         );
[CODE L1155]         let out = model.forward(x);
[CODE L1156]         let targets = make_dummy_targets::<B>(&device, 4);
[CODE L1157]         let loss1 = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L1158]         let loss2 = HydraLoss::<B>::new(HydraLossConfig::new().with_w_pi(2.0));
[CODE L1159]         let t1 = loss1
[CODE L1160]             .total_loss(&out, &targets)
[CODE L1161]             .total
[CODE L1162]             .into_scalar()
[CODE L1163]             .elem::<f32>();
[CODE L1164]         let t2 = loss2
[CODE L1165]             .total_loss(&out, &targets)
[CODE L1166]             .total
[CODE L1167]             .into_scalar()
[CODE L1168]             .elem::<f32>();
[CODE L1169]         assert!((t1 - t2).abs() > 0.001, "different weights should differ");
[CODE L1170]     }
[CODE L1171] 
[CODE L1172]     #[test]
[CODE L1173]     fn test_focal_bce_vs_standard_bce() {
[CODE L1174]         let device = Default::default();
[CODE L1175]         let logits = Tensor::<B, 3>::from_floats([[[3.0; 34]; 3]], &device);
[CODE L1176]         let target = Tensor::<B, 3>::ones([1, 3, 34], &device);
[CODE L1177]         let mask = Tensor::<B, 3>::ones([1, 3, 34], &device);
[CODE L1178]         let focal = danger_focal_bce(logits.clone(), target.clone(), mask.clone());
[CODE L1179]         let standard = bce_with_logits_3d(logits, target);
[CODE L1180]         let standard_sum = (standard * mask)
[CODE L1181]             .sum_dim(2)
[CODE L1182]             .sum_dim(1)
[CODE L1183]             .squeeze_dim::<2>(2)
[CODE L1184]             .squeeze_dim::<1>(1);
[CODE L1185]         let f = focal.into_scalar().elem::<f32>();
[CODE L1186]         let s = standard_sum.into_scalar().elem::<f32>();
[CODE L1187]         assert!(
[CODE L1188]             f < s,
[CODE L1189]             "focal ({f}) should be < standard ({s}) for high-confidence correct"
[CODE L1190]         );
[CODE L1191]     }
[CODE L1192] 
[CODE L1193]     fn onehot2d<B: Backend>(
[CODE L1194]         device: &B::Device,
[CODE L1195]         batch: usize,
[CODE L1196]         classes: usize,
[CODE L1197]         idx: usize,
[CODE L1198]     ) -> Tensor<B, 2> {
[CODE L1199]         let mut d = vec![0.0f32; batch * classes];
[CODE L1200]         for i in 0..batch {
[CODE L1201]             d[i * classes + idx] = 1.0;
[CODE L1202]         }
[CODE L1203]         Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, classes])
[CODE L1204]     }
[CODE L1205] 
[CODE L1206]     fn onehot3d<B: Backend>(
[CODE L1207]         device: &B::Device,
[CODE L1208]         batch: usize,
[CODE L1209]         c1: usize,
[CODE L1210]         c2: usize,
[CODE L1211]     ) -> Tensor<B, 3> {
[CODE L1212]         let mut d = vec![0.0f32; batch * c1 * c2];
[CODE L1213]         for i in 0..(batch * c1) {
[CODE L1214]             d[i * c2] = 1.0;
[CODE L1215]         }
[CODE L1216]         Tensor::<B, 1>::from_floats(d.as_slice(), device).reshape([batch, c1, c2])
[CODE L1217]     }
[CODE L1218] 
[CODE L1219]     #[test]
[CODE L1220]     #[ignore = "slow backward integration test"]
[CODE L1221]     fn test_total_loss_backward() {
[CODE L1222]         use burn::backend::Autodiff;
[CODE L1223]         use burn::optim::GradientsParams;
[CODE L1224]         type AB = Autodiff<NdArray<f32>>;
[CODE L1225] 
[CODE L1226]         let device = Default::default();
[CODE L1227]         let model = HydraModelConfig::new(1)
[CODE L1228]             .with_hidden_channels(32)
[CODE L1229]             .with_num_groups(8)
[CODE L1230]             .with_se_bottleneck(8)
[CODE L1231]             .init::<AB>(&device);
[CODE L1232]         let x = Tensor::<AB, 3>::zeros([1, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1233]         let out = model.forward(x);
[CODE L1234]         let targets = make_dummy_targets::<AB>(&device, 1);
[CODE L1235]         let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
[CODE L1236]         let bd = loss_fn.total_loss(&out, &targets);
[CODE L1237]         let total_val: f32 = bd.total.clone().into_scalar().elem();
[CODE L1238]         assert!(total_val > 0.0, "total should be > 0");
[CODE L1239]         let grads = bd.total.backward();
[CODE L1240]         let grads = GradientsParams::from_grads(grads, &model);
[CODE L1241]         let num_grads = grads.len();
[CODE L1242]         assert!(num_grads > 0, "backward should produce gradients");
[CODE L1243]     }
[CODE L1244] 
[CODE L1245]     #[test]
[CODE L1246]     fn test_all_head_losses_positive() {
[CODE L1247]         let device = Default::default();
[CODE L1248]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1249]         let x = Tensor::<B, 3>::random(
[CODE L1250]             [4, crate::config::INPUT_CHANNELS, 34],
[CODE L1251]             burn::tensor::Distribution::Normal(0.0, 0.1),
[CODE L1252]             &device,
[CODE L1253]         );
[CODE L1254]         let out = model.forward(x);
[CODE L1255]         let targets = make_dummy_targets::<B>(&device, 4);
[CODE L1256]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L1257]         let bd = loss_fn.total_loss(&out, &targets);
[CODE L1258]         let check = |name: &str, t: &Tensor<B, 1>| {
[CODE L1259]             let v: f32 = t.clone().into_scalar().elem();
[CODE L1260]             assert!(v > 0.0 && v.is_finite(), "{name} loss = {v}");
[CODE L1261]         };
[CODE L1262]         check("policy", &bd.policy);
[CODE L1263]         check("value", &bd.value);
[CODE L1264]         check("grp", &bd.grp);
[CODE L1265]         check("opp_next", &bd.opp_next);
[CODE L1266]         check("score_pdf", &bd.score_pdf);
[CODE L1267]         check("score_cdf", &bd.score_cdf);
[CODE L1268]     }
[CODE L1269] 
[CODE L1270]     #[test]
[CODE L1271]     fn test_zero_weight_advanced_heads_keep_baseline_losses_unchanged() {
[CODE L1272]         let device = Default::default();
[CODE L1273]         let model = HydraModelConfig::actor().init::<B>(&device);
[CODE L1274]         let x = Tensor::<B, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
[CODE L1275]         let outputs = model.forward(x.clone());
[CODE L1276]         let optimized_outputs = model.forward_active(x, &HydraLossConfig::new());
[CODE L1277]         let targets = make_dummy_targets::<B>(&device, 2);
[CODE L1278]         let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new());
[CODE L1279]         let baseline = loss_fn.total_loss(&outputs, &targets);
[CODE L1280]         let optimized = loss_fn.total_loss(&optimized_outputs, &targets);
[CODE L1281] 
[CODE L1282]         let scalar = |t: Tensor<B, 1>| t.into_scalar().elem::<f32>();
[CODE L1283]         assert!((scalar(baseline.total) - scalar(optimized.total)).abs() < 1e-6);
[CODE L1284]         assert!((scalar(baseline.policy) - scalar(optimized.policy)).abs() < 1e-6);
[CODE L1285]         assert!((scalar(baseline.value) - scalar(optimized.value)).abs() < 1e-6);
[CODE L1286]         assert!((scalar(baseline.grp) - scalar(optimized.grp)).abs() < 1e-6);
[CODE L1287]         assert!((scalar(baseline.tenpai) - scalar(optimized.tenpai)).abs() < 1e-6);
[CODE L1288]         assert!((scalar(baseline.danger) - scalar(optimized.danger)).abs() < 1e-6);
[CODE L1289]         assert!((scalar(baseline.opp_next) - scalar(optimized.opp_next)).abs() < 1e-6);
[CODE L1290]         assert!((scalar(baseline.score_pdf) - scalar(optimized.score_pdf)).abs() < 1e-6);
[CODE L1291]         assert!((scalar(baseline.score_cdf) - scalar(optimized.score_cdf)).abs() < 1e-6);
[CODE L1292]     }
[CODE L1293] 
[CODE L1294]     #[test]
[CODE L1295]     fn test_default_weights_match_roadmap() {
[CODE L1296]         let cfg = HydraLossConfig::new();
[CODE L1297]         assert!((cfg.w_pi - 1.0).abs() < 1e-6);
[CODE L1298]         assert!((cfg.w_v - 0.5).abs() < 1e-6);
[CODE L1299]         assert!((cfg.w_grp - 0.2).abs() < 1e-6);
[CODE L1300]         assert!((cfg.w_tenpai - 0.1).abs() < 1e-6);
[CODE L1301]         assert!((cfg.w_danger - 0.1).abs() < 1e-6);
[CODE L1302]         assert!((cfg.w_opp - 0.1).abs() < 1e-6);
[CODE L1303]         assert!((cfg.w_score - 0.025).abs() < 1e-6);
[CODE L1304]         assert!((cfg.w_oracle_critic - 0.0).abs() < 1e-6);
[CODE L1305]         assert!((cfg.w_belief_fields - 0.0).abs() < 1e-6);
[CODE L1306]         assert!((cfg.w_mixture_weight - 0.0).abs() < 1e-6);
[CODE L1307]         assert!((cfg.w_opponent_hand_type - 0.0).abs() < 1e-6);
[CODE L1308]         assert!((cfg.w_delta_q - 0.0).abs() < 1e-6);
[CODE L1309]         assert!((cfg.w_safety_residual - 0.0).abs() < 1e-6);
[CODE L1310]         let total_weight = cfg.w_pi
[CODE L1311]             + cfg.w_v
[CODE L1312]             + cfg.w_grp
[CODE L1313]             + cfg.w_tenpai
[CODE L1314]             + cfg.w_danger
[CODE L1315]             + cfg.w_opp
[CODE L1316]             + cfg.w_score * 2.0
[CODE L1317]             + cfg.w_oracle_critic
[CODE L1318]             + cfg.w_belief_fields
[CODE L1319]             + cfg.w_mixture_weight
[CODE L1320]             + cfg.w_opponent_hand_type
[CODE L1321]             + cfg.w_delta_q
[CODE L1322]             + cfg.w_safety_residual;
[CODE L1323]         assert!(
[CODE L1324]             (total_weight - 2.05).abs() < 1e-4,
[CODE L1325]             "total weight = {total_weight}"
[CODE L1326]         );
[CODE L1327]     }
[CODE L1328] 
[CODE L1329]     #[test]
[CODE L1330]     fn test_validate_rejects_negative_primary_weights() {
[CODE L1331]         assert!(
[CODE L1332]             HydraLossConfig::new()
[CODE L1333]                 .with_w_tenpai(-0.1)
[CODE L1334]                 .validate()
[CODE L1335]                 .is_err()
[CODE L1336]         );
[CODE L1337]         assert!(
[CODE L1338]             HydraLossConfig::new()
[CODE L1339]                 .with_w_danger(-0.1)
[CODE L1340]                 .validate()
[CODE L1341]                 .is_err()
[CODE L1342]         );
[CODE L1343]         assert!(HydraLossConfig::new().with_w_opp(-0.1).validate().is_err());
[CODE L1344]         assert!(
[CODE L1345]             HydraLossConfig::new()
[CODE L1346]                 .with_w_score(-0.1)
[CODE L1347]                 .validate()
[CODE L1348]                 .is_err()
[CODE L1349]         );
[CODE L1350]     }
[CODE L1351] 
[CODE L1352]     #[test]
[CODE L1353]     fn test_compute_cvar() {
[CODE L1354]         let pdf = [0.1f32, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1];
[CODE L1355]         let cvar = compute_cvar(&pdf, 0.3);
[CODE L1356]         assert!(cvar >= 0.0 && cvar.is_finite(), "CVaR: {cvar}");
[CODE L1357]         let cvar_full = compute_cvar(&pdf, 1.0);
[CODE L1358]         assert!(cvar <= cvar_full, "CVaR(0.3) <= CVaR(1.0)");
[CODE L1359]     }
[CODE L1360] 
[CODE L1361]     #[test]
[CODE L1362]     fn test_bce_extreme_logits() {
[CODE L1363]         let device = Default::default();
[CODE L1364]         let logits = Tensor::<B, 2>::from_floats([[100.0, -100.0]], &device);
[CODE L1365]         let target = Tensor::<B, 2>::from_floats([[1.0, 0.0]], &device);
[CODE L1366]         let loss = bce_with_logits(logits, target);
[CODE L1367]         let data = loss.to_data();
[CODE L1368]         for &v in data.as_slice::<f32>().expect("f32") {
[CODE L1369]             assert!(v.is_finite(), "extreme logits should give finite BCE: {v}");
[CODE L1370]         }
[CODE L1371]     }
[CODE L1372] 
[CODE L1373]     #[test]
[CODE L1374]     fn test_policy_ce_single_legal_action() {
[CODE L1375]         let device = Default::default();
[CODE L1376]         let mut mask_data = [0.0f32; 46];
[CODE L1377]         mask_data[5] = 1.0;
[CODE L1378]         let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device).reshape([1, 46]);
[CODE L1379]         let target = mask.clone();
[CODE L1380]         let logits = Tensor::<B, 2>::zeros([1, 46], &device);
[CODE L1381]         let loss = policy_ce(logits, target, mask);
[CODE L1382]         let v: f32 = loss.into_scalar().elem();
[CODE L1383]         assert!(v < 0.01, "single legal action loss should be ~0, got {v}");
[CODE L1384]     }
[CODE L1385] 
[CODE L1386]     #[test]
[CODE L1387]     fn test_value_mse_extreme_values() {
[CODE L1388]         let device = Default::default();
[CODE L1389]         let pred = Tensor::<B, 1>::from_floats([0.99, -0.99], &device);
[CODE L1390]         let target = Tensor::<B, 1>::from_floats([1.0, -1.0], &device);
[CODE L1391]         let loss = value_mse(pred, target);
[CODE L1392]         let data = loss.to_data();
[CODE L1393]         for &v in data.as_slice::<f32>().expect("f32") {
[CODE L1394]             assert!(v.is_finite(), "extreme value MSE should be finite, got {v}");
[CODE L1395]             assert!(v < 0.01, "near-boundary MSE should be small, got {v}");
[CODE L1396]         }
[CODE L1397]     }
[CODE L1398] 
[CODE L1399]     #[test]
[CODE L1400]     fn test_oracle_target_from_scores_zero_sum() {
[CODE L1401]         let target = oracle_target_from_scores([25000, 25000, 25000, 25000]);
[CODE L1402]         for (i, &v) in target.iter().enumerate() {
[CODE L1403]             assert!(
[CODE L1404]                 v.abs() < 1e-6,
[CODE L1405]                 "equal scores should give zero delta, player {i} got {v}"
[CODE L1406]             );
[CODE L1407]         }
[CODE L1408]     }
[CODE L1409] 
[CODE L1410]     #[test]
[CODE L1411]     fn test_kl_divergence_identical_distributions() {
[CODE L1412]         let device = Default::default();
[CODE L1413]         let p = Tensor::<B, 2>::from_floats([[0.3, 0.5, 0.2]], &device);
[CODE L1414]         let kl = kl_divergence(p.clone(), p);
[CODE L1415]         let v: f32 = kl.into_scalar().elem();
[CODE L1416]         assert!(v.abs() < 1e-6, "KL(p, p) should be ~0, got {v}");
[CODE L1417]     }
[CODE L1418] 
[CODE L1419]     pub fn make_dummy_targets<B: Backend>(device: &B::Device, batch: usize) -> HydraTargets<B> {
[CODE L1420]         HydraTargets {
[CODE L1421]             policy_target: onehot2d(device, batch, 46, 0),
[CODE L1422]             legal_mask: Tensor::ones([batch, 46], device),
[CODE L1423]             value_target: Tensor::zeros([batch], device),
[CODE L1424]             grp_target: onehot2d(device, batch, 24, 0),
[CODE L1425]             tenpai_target: Tensor::zeros([batch, 3], device),
[CODE L1426]             danger_target: Tensor::zeros([batch, 3, 34], device),
[CODE L1427]             danger_mask: Tensor::ones([batch, 3, 34], device),
[CODE L1428]             opp_next_target: onehot3d(device, batch, 3, 34),
[CODE L1429]             score_pdf_target: onehot2d(device, batch, 64, 32),
[CODE L1430]             score_cdf_target: Tensor::zeros([batch, 64], device),
[CODE L1431]             oracle_target: None,
[CODE L1432]             belief_fields_target: None,
[CODE L1433]             belief_fields_mask: None,
[CODE L1434]             mixture_weight_target: None,
[CODE L1435]             mixture_weight_mask: None,
[CODE L1436]             opponent_hand_type_target: None,
[CODE L1437]             delta_q_target: None,
[CODE L1438]             delta_q_mask: None,
[CODE L1439]             safety_residual_target: None,
[CODE L1440]             safety_residual_mask: None,
[CODE L1441]             oracle_guidance_mask: None,
[CODE L1442]             target_presence: None,
[CODE L1443]         }
[CODE L1444]     }
[CODE L1445] 
[CODE L1446]     #[test]
[CODE L1447]     fn slice_batch_clears_cached_target_presence() {
[CODE L1448]         let device = Default::default();
[CODE L1449]         let mut targets = make_dummy_targets::<B>(&device, 4);
[CODE L1450]         targets.target_presence = Some(crate::training::head_gates::TargetPresence {
[CODE L1451]             counts: [1, 2, 3, 4, 5, 6],
[CODE L1452]             delta_q_actions_present: 7,
[CODE L1453]             batch_size: 4,
[CODE L1454]         });
[CODE L1455] 
[CODE L1456]         let sliced = targets.slice_batch(1, 3);
[CODE L1457]         assert!(
[CODE L1458]             sliced.target_presence.is_none(),
[CODE L1459]             "sliced targets must drop cached full-batch presence metadata"
[CODE L1460]         );
[CODE L1461]     }
[CODE L1462] }
```

## Artifact 20 — DRDA training module
Artifact id: `drda-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-train/src/training/drda.rs`
Why it matters: Nearby training logic artifact included so the research agent can compare current advanced training surfaces against external adjacent methods without conflating acronyms or mechanisms.

```rust
[CODE L0001] //! DRDA wrapper: Dilated Regularized Dual Averaging (Farina et al., ICLR 2025).
[CODE L0002] 
[CODE L0003] use burn::prelude::*;
[CODE L0004] use burn::tensor::activation;
[CODE L0005] 
[CODE L0006] #[derive(Config, Debug)]
[CODE L0007] pub struct DrdaConfig {
[CODE L0008]     #[config(default = "4.0")]
[CODE L0009]     pub tau_drda: f32,
[CODE L0010] }
[CODE L0011] 
[CODE L0012] pub const MIN_TAU_DRDA: f32 = 2.0;
[CODE L0013] pub const MIN_REBASE_INTERVAL_HOURS: f32 = 25.0;
[CODE L0014] 
[CODE L0015] impl DrdaConfig {
[CODE L0016]     pub fn summary(&self) -> String {
[CODE L0017]         format!("drda(tau={:.1})", self.tau_drda)
[CODE L0018]     }
[CODE L0019] 
[CODE L0020]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0021]         if self.tau_drda < MIN_TAU_DRDA {
[CODE L0022]             return Err("tau_drda below minimum 2.0");
[CODE L0023]         }
[CODE L0024]         Ok(())
[CODE L0025]     }
[CODE L0026] }
[CODE L0027] 
[CODE L0028] pub struct RebaseTracker {
[CODE L0029]     pub gpu_hours_since_rebase: f32,
[CODE L0030]     pub rebase_interval_hours: f32,
[CODE L0031]     pub total_rebases: u32,
[CODE L0032] }
[CODE L0033] 
[CODE L0034] impl RebaseTracker {
[CODE L0035]     pub fn default_phase2() -> Self {
[CODE L0036]         Self::new(37.5)
[CODE L0037]     }
[CODE L0038] 
[CODE L0039]     pub fn new(interval_hours: f32) -> Self {
[CODE L0040]         Self {
[CODE L0041]             gpu_hours_since_rebase: 0.0,
[CODE L0042]             rebase_interval_hours: interval_hours.max(MIN_REBASE_INTERVAL_HOURS),
[CODE L0043]             total_rebases: 0,
[CODE L0044]         }
[CODE L0045]     }
[CODE L0046] 
[CODE L0047]     pub fn progress(&self) -> f32 {
[CODE L0048]         if self.rebase_interval_hours <= 0.0 {
[CODE L0049]             return 0.0;
[CODE L0050]         }
[CODE L0051]         (self.gpu_hours_since_rebase / self.rebase_interval_hours).min(1.0)
[CODE L0052]     }
[CODE L0053] 
[CODE L0054]     pub fn hours_until_next(&self) -> f32 {
[CODE L0055]         (self.rebase_interval_hours - self.gpu_hours_since_rebase).max(0.0)
[CODE L0056]     }
[CODE L0057] 
[CODE L0058]     pub fn is_overdue(&self, factor: f32) -> bool {
[CODE L0059]         self.gpu_hours_since_rebase >= self.rebase_interval_hours * factor
[CODE L0060]     }
[CODE L0061] 
[CODE L0062]     pub fn should_rebase(&self) -> bool {
[CODE L0063]         self.gpu_hours_since_rebase >= self.rebase_interval_hours
[CODE L0064]     }
[CODE L0065] 
[CODE L0066]     pub fn record_rebase(&mut self) {
[CODE L0067]         self.total_rebases += 1;
[CODE L0068]         self.gpu_hours_since_rebase = 0.0;
[CODE L0069]     }
[CODE L0070] 
[CODE L0071]     pub fn summary(&self) -> String {
[CODE L0072]         format!(
[CODE L0073]             "rebases={} hours_since={:.1}",
[CODE L0074]             self.total_rebases, self.gpu_hours_since_rebase
[CODE L0075]         )
[CODE L0076]     }
[CODE L0077] 
[CODE L0078]     pub fn tick(&mut self, hours: f32) {
[CODE L0079]         self.gpu_hours_since_rebase += hours;
[CODE L0080]     }
[CODE L0081] }
[CODE L0082] 
[CODE L0083] type BaseLogitsFn<B> = Box<dyn Fn(Tensor<B, 3>) -> Tensor<B, 2>>;
[CODE L0084] 
[CODE L0085] pub struct DrdaWrapper<B: Backend> {
[CODE L0086]     pub base_logits_fn: Option<BaseLogitsFn<B>>,
[CODE L0087]     pub tau_drda: f32,
[CODE L0088] }
[CODE L0089] 
[CODE L0090] impl<B: Backend> DrdaWrapper<B> {
[CODE L0091]     pub fn new(tau_drda: f32) -> Self {
[CODE L0092]         Self {
[CODE L0093]             base_logits_fn: None,
[CODE L0094]             tau_drda: tau_drda.max(MIN_TAU_DRDA),
[CODE L0095]         }
[CODE L0096]     }
[CODE L0097] 
[CODE L0098]     pub fn combined_logits(
[CODE L0099]         &self,
[CODE L0100]         base_logits: Tensor<B, 2>,
[CODE L0101]         residual_logits: Tensor<B, 2>,
[CODE L0102]     ) -> Tensor<B, 2> {
[CODE L0103]         base_logits + residual_logits / self.tau_drda
[CODE L0104]     }
[CODE L0105] }
[CODE L0106] 
[CODE L0107] pub fn combined_logits<B: Backend>(
[CODE L0108]     base_logits: Tensor<B, 2>,
[CODE L0109]     residual_logits: Tensor<B, 2>,
[CODE L0110]     tau_drda: f32,
[CODE L0111] ) -> Tensor<B, 2> {
[CODE L0112]     base_logits + residual_logits / tau_drda
[CODE L0113] }
[CODE L0114] 
[CODE L0115] pub fn verify_rebase_preserves_pi<B: Backend>(
[CODE L0116]     pi_before: Tensor<B, 2>,
[CODE L0117]     pi_after: Tensor<B, 2>,
[CODE L0118] ) -> f32 {
[CODE L0119]     let eps = 1e-8f32;
[CODE L0120]     let p = pi_before.clamp(eps, 1.0);
[CODE L0121]     let q = pi_after.clamp(eps, 1.0);
[CODE L0122]     let log_ratio = (p.clone() / q).log();
[CODE L0123]     let kl = (p * log_ratio).sum_dim(1).mean();
[CODE L0124]     kl.into_data()
[CODE L0125]         .convert::<f32>()
[CODE L0126]         .as_slice::<f32>()
[CODE L0127]         .expect("kl scalar should be readable as f32")[0]
[CODE L0128] }
[CODE L0129] 
[CODE L0130] pub fn compute_rebase_kl<B: Backend>(
[CODE L0131]     base_logits: Tensor<B, 2>,
[CODE L0132]     residual_logits: Tensor<B, 2>,
[CODE L0133]     tau_drda: f32,
[CODE L0134]     legal_mask: Tensor<B, 2>,
[CODE L0135] ) -> f32 {
[CODE L0136]     let combined = combined_logits(base_logits.clone(), residual_logits, tau_drda);
[CODE L0137]     let neg_inf = (legal_mask.clone().ones_like() - legal_mask) * (-1e9f32);
[CODE L0138]     let pi_before = activation::softmax(combined + neg_inf.clone(), 1);
[CODE L0139]     let pi_after = activation::softmax(base_logits + neg_inf, 1);
[CODE L0140]     verify_rebase_preserves_pi(pi_before, pi_after)
[CODE L0141] }
[CODE L0142] 
[CODE L0143] pub fn compute_new_base_logits<B: Backend>(
[CODE L0144]     base_logits: Tensor<B, 2>,
[CODE L0145]     residual_logits: Tensor<B, 2>,
[CODE L0146]     tau_drda: f32,
[CODE L0147] ) -> Tensor<B, 2> {
[CODE L0148]     base_logits + residual_logits / tau_drda
[CODE L0149] }
[CODE L0150] 
[CODE L0151] pub fn policy_head_is_zeroed<B: Backend>(logits: Tensor<B, 2>) -> bool {
[CODE L0152]     let max_abs = logits
[CODE L0153]         .abs()
[CODE L0154]         .max()
[CODE L0155]         .into_data()
[CODE L0156]         .convert::<f32>()
[CODE L0157]         .as_slice::<f32>()
[CODE L0158]         .expect("max-abs scalar should be readable as f32")[0];
[CODE L0159]     max_abs < 1e-6
[CODE L0160] }
[CODE L0161] 
[CODE L0162] #[cfg(test)]
[CODE L0163] mod tests {
[CODE L0164]     use super::*;
[CODE L0165]     use burn::backend::NdArray;
[CODE L0166] 
[CODE L0167]     type B = NdArray<f32>;
[CODE L0168] 
[CODE L0169]     #[test]
[CODE L0170]     fn test_drda_defaults_match_roadmap() {
[CODE L0171]         let cfg = DrdaConfig::new();
[CODE L0172]         assert!((cfg.tau_drda - 4.0).abs() < 1e-6);
[CODE L0173]     }
[CODE L0174] 
[CODE L0175]     #[test]
[CODE L0176]     fn test_drda_combined_logits() {
[CODE L0177]         let device = Default::default();
[CODE L0178]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
[CODE L0179]         let residual = Tensor::<B, 2>::from_floats([[4.0, 8.0, 12.0]], &device);
[CODE L0180]         let out = combined_logits(base, residual, 4.0);
[CODE L0181]         let data = out.to_data();
[CODE L0182]         let vals = data.as_slice::<f32>().expect("f32");
[CODE L0183]         assert!((vals[0] - 2.0).abs() < 1e-5);
[CODE L0184]         assert!((vals[1] - 4.0).abs() < 1e-5);
[CODE L0185]         assert!((vals[2] - 6.0).abs() < 1e-5);
[CODE L0186]     }
[CODE L0187] 
[CODE L0188]     #[test]
[CODE L0189]     fn test_drda_rebase_preserves_pi() {
[CODE L0190]         let device = Default::default();
[CODE L0191]         let pi = Tensor::<B, 2>::from_floats([[0.2, 0.3, 0.5]], &device);
[CODE L0192]         let kl = verify_rebase_preserves_pi(pi.clone(), pi);
[CODE L0193]         assert!(kl.abs() < 1e-6, "KL should be ~0, got {kl}");
[CODE L0194]     }
[CODE L0195] 
[CODE L0196]     #[test]
[CODE L0197]     fn test_drda_zero_residual_equals_base() {
[CODE L0198]         let device = Default::default();
[CODE L0199]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
[CODE L0200]         let zero = Tensor::<B, 2>::zeros([1, 3], &device);
[CODE L0201]         let out = combined_logits(base.clone(), zero, 4.0);
[CODE L0202]         let b_data = base.to_data();
[CODE L0203]         let o_data = out.to_data();
[CODE L0204]         let b = b_data.as_slice::<f32>().expect("f32");
[CODE L0205]         let o = o_data.as_slice::<f32>().expect("f32");
[CODE L0206]         for i in 0..3 {
[CODE L0207]             assert!(
[CODE L0208]                 (b[i] - o[i]).abs() < 1e-6,
[CODE L0209]                 "zero residual should equal base at {i}"
[CODE L0210]             );
[CODE L0211]         }
[CODE L0212]     }
[CODE L0213] 
[CODE L0214]     #[test]
[CODE L0215]     fn test_drda_wrapper_method() {
[CODE L0216]         let device = Default::default();
[CODE L0217]         let wrapper = DrdaWrapper::<B>::new(4.0);
[CODE L0218]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &device);
[CODE L0219]         let res = Tensor::<B, 2>::from_floats([[8.0, 4.0]], &device);
[CODE L0220]         let out = wrapper.combined_logits(base, res);
[CODE L0221]         let data = out.to_data();
[CODE L0222]         let vals = data.as_slice::<f32>().expect("f32");
[CODE L0223]         assert!((vals[0] - 3.0).abs() < 1e-5);
[CODE L0224]         assert!((vals[1] - 3.0).abs() < 1e-5);
[CODE L0225]     }
[CODE L0226] 
[CODE L0227]     #[test]
[CODE L0228]     fn test_compute_rebase_kl_zero_residual() {
[CODE L0229]         let device = Default::default();
[CODE L0230]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
[CODE L0231]         let zero_res = Tensor::<B, 2>::zeros([1, 3], &device);
[CODE L0232]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0233]         let kl = compute_rebase_kl(base, zero_res, 4.0, mask);
[CODE L0234]         assert!(kl.abs() < 1e-5, "zero residual should give KL~0: {kl}");
[CODE L0235]     }
[CODE L0236] 
[CODE L0237]     #[test]
[CODE L0238]     fn test_compute_rebase_kl_nonzero_residual() {
[CODE L0239]         let device = Default::default();
[CODE L0240]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
[CODE L0241]         let res = Tensor::<B, 2>::from_floats([[5.0, -5.0, 0.0]], &device);
[CODE L0242]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0243]         let kl = compute_rebase_kl(base, res, 4.0, mask);
[CODE L0244]         assert!(kl > 0.0, "non-zero residual should give positive KL: {kl}");
[CODE L0245]     }
[CODE L0246] 
[CODE L0247]     #[test]
[CODE L0248]     fn test_drda_tau_below_minimum() {
[CODE L0249]         let cfg = DrdaConfig { tau_drda: 1.5 };
[CODE L0250]         let result = cfg.validate();
[CODE L0251]         assert!(result.is_err(), "tau_drda=1.5 should fail validation");
[CODE L0252]         assert_eq!(result.unwrap_err(), "tau_drda below minimum 2.0");
[CODE L0253]     }
[CODE L0254] 
[CODE L0255]     #[test]
[CODE L0256]     fn test_drda_rebase_tracker_timing() {
[CODE L0257]         let mut tracker = RebaseTracker::new(37.5);
[CODE L0258]         assert!(!tracker.should_rebase(), "fresh tracker should not rebase");
[CODE L0259] 
[CODE L0260]         tracker.tick(38.0);
[CODE L0261]         assert!(
[CODE L0262]             tracker.should_rebase(),
[CODE L0263]             "after 38h with 37.5h interval, should_rebase must be true"
[CODE L0264]         );
[CODE L0265] 
[CODE L0266]         tracker.record_rebase();
[CODE L0267]         assert!(
[CODE L0268]             !tracker.should_rebase(),
[CODE L0269]             "after record_rebase, should_rebase must be false"
[CODE L0270]         );
[CODE L0271]         assert_eq!(tracker.total_rebases, 1);
[CODE L0272]     }
[CODE L0273] 
[CODE L0274]     #[test]
[CODE L0275]     fn test_compute_new_base_logits() {
[CODE L0276]         let device = Default::default();
[CODE L0277]         let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
[CODE L0278]         let residual = Tensor::<B, 2>::from_floats([[4.0, 0.0, -4.0]], &device);
[CODE L0279]         let new_base = compute_new_base_logits(base, residual, 4.0);
[CODE L0280]         let data: Vec<f32> = new_base.to_data().as_slice::<f32>().unwrap().to_vec();
[CODE L0281]         assert!((data[0] - 2.0).abs() < 1e-5, "1.0 + 4.0/4.0 = 2.0");
[CODE L0282]         assert!((data[1] - 2.0).abs() < 1e-5, "2.0 + 0.0/4.0 = 2.0");
[CODE L0283]         assert!((data[2] - 2.0).abs() < 1e-5, "3.0 + -4.0/4.0 = 2.0");
[CODE L0284]     }
[CODE L0285] }
```

## Artifact 21 — ACH training module
Artifact id: `ach-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-train/src/training/ach.rs`
Why it matters: Code-level evidence for one of Hydra's major chosen training paradigms. Helps the research agent compare DCRL-like prior art against Hydra's actual active optimization path, not just doc prose.

```rust
[CODE L0001] //! Actor-Critic Hedge loss (LuckyJ's algorithm, ICLR 2022).
[CODE L0002] 
[CODE L0003] use burn::prelude::*;
[CODE L0004] use burn::tensor::activation;
[CODE L0005] 
[CODE L0006] #[derive(Config, Debug)]
[CODE L0007] pub struct AchConfig {
[CODE L0008]     #[config(default = "1.0")]
[CODE L0009]     pub eta: f32,
[CODE L0010]     #[config(default = "0.5")]
[CODE L0011]     pub eps: f32,
[CODE L0012]     #[config(default = "8.0")]
[CODE L0013]     pub l_th: f32,
[CODE L0014]     #[config(default = "5e-4")]
[CODE L0015]     pub beta_ent: f32,
[CODE L0016] }
[CODE L0017] 
[CODE L0018] impl AchConfig {
[CODE L0019]     pub fn summary(&self) -> String {
[CODE L0020]         format!(
[CODE L0021]             "ach(eta={:.1}, eps={:.1}, l_th={:.0}, ent={:.1e})",
[CODE L0022]             self.eta, self.eps, self.l_th, self.beta_ent
[CODE L0023]         )
[CODE L0024]     }
[CODE L0025] 
[CODE L0026]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0027]         if self.eta <= 0.0 {
[CODE L0028]             return Err("eta must be positive");
[CODE L0029]         }
[CODE L0030]         if self.eps <= 0.0 || self.eps >= 1.0 {
[CODE L0031]             return Err("eps must be in (0,1)");
[CODE L0032]         }
[CODE L0033]         if self.l_th <= 0.0 {
[CODE L0034]             return Err("l_th must be positive");
[CODE L0035]         }
[CODE L0036]         Ok(())
[CODE L0037]     }
[CODE L0038] }
[CODE L0039] 
[CODE L0040] pub fn ach_policy_loss<B: Backend>(
[CODE L0041]     logits: Tensor<B, 2>,
[CODE L0042]     legal_mask: Tensor<B, 2>,
[CODE L0043]     actions: Tensor<B, 1, Int>,
[CODE L0044]     pi_old: Tensor<B, 1>,
[CODE L0045]     advantages: Tensor<B, 1>,
[CODE L0046]     cfg: &AchConfig,
[CODE L0047] ) -> Tensor<B, 1> {
[CODE L0048]     let neg_inf_mask = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
[CODE L0049]     let masked_logits = logits + neg_inf_mask;
[CODE L0050] 
[CODE L0051]     let legal_sum = legal_mask.clone().sum_dim(1).clamp_min(1.0);
[CODE L0052]     let legal_mean = (masked_logits.clone() * legal_mask.clone()).sum_dim(1) / legal_sum;
[CODE L0053]     let centered = masked_logits - legal_mean;
[CODE L0054]     let clamped = centered.clamp(-cfg.l_th, cfg.l_th);
[CODE L0055] 
[CODE L0056]     let neg_inf_mask2 = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
[CODE L0057]     let for_softmax = clamped.clone() + neg_inf_mask2;
[CODE L0058]     let pi = activation::softmax(for_softmax, 1);
[CODE L0059] 
[CODE L0060]     let actions_2d = actions.unsqueeze_dim::<2>(1);
[CODE L0061]     let y_a = clamped.gather(1, actions_2d.clone()).squeeze_dim::<1>(1);
[CODE L0062]     let pi_a = pi.clone().gather(1, actions_2d).squeeze_dim::<1>(1);
[CODE L0063] 
[CODE L0064]     let pi_old_safe = pi_old.clone().clamp_min(1e-8);
[CODE L0065]     let ratio = pi_a.clone() / pi_old_safe.clone();
[CODE L0066] 
[CODE L0067]     let adv_pos = advantages.clone().clamp_min(0.0);
[CODE L0068]     let adv_neg = advantages.clone().clamp_max(0.0);
[CODE L0069]     let has_pos = adv_pos.clone().sign();
[CODE L0070]     let has_neg = adv_neg.clone().sign().neg();
[CODE L0071] 
[CODE L0072]     let gate_pos_ratio = ratio.clone().lower_elem(1.0 + cfg.eps).float();
[CODE L0073]     let gate_pos_logit = y_a.clone().lower_elem(cfg.l_th).float();
[CODE L0074]     let gate_pos = has_pos * gate_pos_ratio * gate_pos_logit;
[CODE L0075] 
[CODE L0076]     let gate_neg_ratio = ratio.clone().greater_elem(1.0 - cfg.eps).float();
[CODE L0077]     let gate_neg_logit = y_a.clone().greater_elem(-cfg.l_th).float();
[CODE L0078]     let gate_neg = has_neg * gate_neg_ratio * gate_neg_logit;
[CODE L0079] 
[CODE L0080]     let gate = gate_pos + gate_neg;
[CODE L0081] 
[CODE L0082]     let policy_loss = (gate * y_a / pi_old_safe * advantages).neg().mean();
[CODE L0083] 
[CODE L0084]     let log_pi = pi.clone().clamp(1e-8, 1.0).log();
[CODE L0085]     let entropy = (pi * log_pi * legal_mask).sum_dim(1).neg().mean();
[CODE L0086]     let ent_bonus = entropy * cfg.beta_ent;
[CODE L0087] 
[CODE L0088]     policy_loss * cfg.eta - ent_bonus
[CODE L0089] }
[CODE L0090] 
[CODE L0091] #[cfg(test)]
[CODE L0092] mod tests {
[CODE L0093]     use super::*;
[CODE L0094]     use burn::backend::{Autodiff, NdArray};
[CODE L0095] 
[CODE L0096]     type B = NdArray<f32>;
[CODE L0097]     type AB = Autodiff<NdArray<f32>>;
[CODE L0098]     type AchInputs = (
[CODE L0099]         Tensor<B, 2>,
[CODE L0100]         Tensor<B, 2>,
[CODE L0101]         Tensor<B, 1, Int>,
[CODE L0102]         Tensor<B, 1>,
[CODE L0103]         Tensor<B, 1>,
[CODE L0104]     );
[CODE L0105] 
[CODE L0106]     fn make_ach_inputs(device: &<B as Backend>::Device) -> AchInputs {
[CODE L0107]         let logits = Tensor::<B, 2>::from_floats([[0.0, 1.0, -1.0]], device);
[CODE L0108]         let mask = Tensor::<B, 2>::ones([1, 3], device);
[CODE L0109]         let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], device);
[CODE L0110]         let pi_old = Tensor::<B, 1>::from_floats([0.5], device);
[CODE L0111]         let advantages = Tensor::<B, 1>::from_floats([1.0], device);
[CODE L0112]         (logits, mask, actions, pi_old, advantages)
[CODE L0113]     }
[CODE L0114] 
[CODE L0115]     #[test]
[CODE L0116]     fn test_ach_defaults_match_roadmap() {
[CODE L0117]         let cfg = AchConfig::new();
[CODE L0118]         assert!((cfg.eta - 1.0).abs() < 1e-6);
[CODE L0119]         assert!((cfg.eps - 0.5).abs() < 1e-6);
[CODE L0120]         assert!((cfg.l_th - 8.0).abs() < 1e-6);
[CODE L0121]         assert!((cfg.beta_ent - 5e-4).abs() < 1e-8);
[CODE L0122]     }
[CODE L0123] 
[CODE L0124]     #[test]
[CODE L0125]     fn test_ach_gate_positive_adv() {
[CODE L0126]         let device = Default::default();
[CODE L0127]         let (logits, mask, actions, pi_old, advantages) = make_ach_inputs(&device);
[CODE L0128]         let cfg = AchConfig::new();
[CODE L0129]         let loss = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
[CODE L0130]         let val = loss.into_scalar().elem::<f32>();
[CODE L0131]         assert!(val.is_finite(), "ACH loss should be finite: {val}");
[CODE L0132]     }
[CODE L0133] 
[CODE L0134]     #[test]
[CODE L0135]     fn test_ach_gate_clips_ratio() {
[CODE L0136]         let device = Default::default();
[CODE L0137]         let logits = Tensor::<B, 2>::from_floats([[0.0, 5.0, -5.0]], &device);
[CODE L0138]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0139]         let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
[CODE L0140]         let pi_old = Tensor::<B, 1>::from_floats([0.01], &device);
[CODE L0141]         let adv = Tensor::<B, 1>::from_floats([1.0], &device);
[CODE L0142]         let cfg = AchConfig::new();
[CODE L0143]         let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
[CODE L0144]         let val = loss.into_scalar().elem::<f32>();
[CODE L0145]         assert!(val.is_finite());
[CODE L0146]     }
[CODE L0147] 
[CODE L0148]     #[test]
[CODE L0149]     fn test_ach_negative_adv() {
[CODE L0150]         let device = Default::default();
[CODE L0151]         let (logits, mask, actions, pi_old, _) = make_ach_inputs(&device);
[CODE L0152]         let neg_adv = Tensor::<B, 1>::from_floats([-1.0], &device);
[CODE L0153]         let cfg = AchConfig::new();
[CODE L0154]         let loss = ach_policy_loss(logits, mask, actions, pi_old, neg_adv, &cfg);
[CODE L0155]         let val = loss.into_scalar().elem::<f32>();
[CODE L0156]         assert!(val.is_finite());
[CODE L0157]     }
[CODE L0158] 
[CODE L0159]     #[test]
[CODE L0160]     fn test_ach_gate_clips_logit() {
[CODE L0161]         let device = Default::default();
[CODE L0162]         let logits = Tensor::<B, 2>::from_floats([[0.0, 20.0, -20.0]], &device);
[CODE L0163]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0164]         let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
[CODE L0165]         let pi_old = Tensor::<B, 1>::from_floats([0.5], &device);
[CODE L0166]         let adv = Tensor::<B, 1>::from_floats([1.0], &device);
[CODE L0167]         let cfg = AchConfig::new();
[CODE L0168]         let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
[CODE L0169]         let val = loss.into_scalar().elem::<f32>();
[CODE L0170]         assert!(val.is_finite(), "clipped logit should produce finite loss");
[CODE L0171]     }
[CODE L0172] 
[CODE L0173]     #[test]
[CODE L0174]     fn ach_loss_zero_pi_old_no_nan() {
[CODE L0175]         let device = Default::default();
[CODE L0176]         let logits = Tensor::<B, 2>::from_floats([[0.0, 1.0, -1.0]], &device);
[CODE L0177]         let mask = Tensor::<B, 2>::ones([1, 3], &device);
[CODE L0178]         let actions = Tensor::<B, 1, Int>::from_ints(&[1i32][..], &device);
[CODE L0179]         let pi_old = Tensor::<B, 1>::from_floats([0.0], &device);
[CODE L0180]         let advantages = Tensor::<B, 1>::from_floats([1.0], &device);
[CODE L0181]         let cfg = AchConfig::new();
[CODE L0182]         let loss = ach_policy_loss(logits, mask, actions, pi_old, advantages, &cfg);
[CODE L0183]         let val = loss.into_scalar().elem::<f32>();
[CODE L0184]         assert!(
[CODE L0185]             val.is_finite(),
[CODE L0186]             "pi_old=0 should not produce NaN/Inf: {val}"
[CODE L0187]         );
[CODE L0188]     }
[CODE L0189] 
[CODE L0190]     #[test]
[CODE L0191]     fn test_ach_batch_of_8() {
[CODE L0192]         let device = Default::default();
[CODE L0193]         let logits = Tensor::<B, 2>::random(
[CODE L0194]             [8, 46],
[CODE L0195]             burn::tensor::Distribution::Normal(0.0, 1.0),
[CODE L0196]             &device,
[CODE L0197]         );
[CODE L0198]         let mask = Tensor::<B, 2>::ones([8, 46], &device);
[CODE L0199]         let actions = Tensor::<B, 1, Int>::from_ints(&[0i32, 1, 2, 3, 4, 5, 6, 7][..], &device);
[CODE L0200]         let pi_old = Tensor::<B, 1>::from_floats([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2], &device);
[CODE L0201]         let adv = Tensor::<B, 1>::from_floats([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.1], &device);
[CODE L0202]         let cfg = AchConfig::new();
[CODE L0203]         let loss = ach_policy_loss(logits, mask, actions, pi_old, adv, &cfg);
[CODE L0204]         let val = loss.into_scalar().elem::<f32>();
[CODE L0205]         assert!(val.is_finite(), "batch ACH should be finite: {val}");
[CODE L0206]     }
[CODE L0207] 
[CODE L0208]     #[test]
[CODE L0209]     fn test_ach_one_epoch_changes_weights() {
[CODE L0210]         use crate::model::HydraModelConfig;
[CODE L0211]         use crate::training::losses::{HydraLoss, HydraLossConfig, tests::make_dummy_targets};
[CODE L0212]         use crate::training::rl::{RlBatch, RlConfig, rl_step};
[CODE L0213]         use burn::optim::AdamConfig;
[CODE L0214] 
[CODE L0215]         let device = Default::default();
[CODE L0216]         let model = HydraModelConfig::new(2)
[CODE L0217]             .with_hidden_channels(32)
[CODE L0218]             .with_se_bottleneck(8)
[CODE L0219]             .with_num_groups(4)
[CODE L0220]             .init::<AB>(&device);
[CODE L0221] 
[CODE L0222]         let obs = Tensor::<AB, 3>::random(
[CODE L0223]             [2, crate::config::INPUT_CHANNELS, 34],
[CODE L0224]             burn::tensor::Distribution::Normal(0.0, 0.1),
[CODE L0225]             &device,
[CODE L0226]         );
[CODE L0227] 
[CODE L0228]         let out_before = model.forward(obs.clone());
[CODE L0229]         let val_before: f32 = out_before.value.clone().mean().into_scalar().elem();
[CODE L0230] 
[CODE L0231]         let batch = RlBatch {
[CODE L0232]             obs: obs.clone(),
[CODE L0233]             actions: Tensor::<AB, 1, Int>::from_ints(&[0i32, 1][..], &device),
[CODE L0234]             pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.3], &device),
[CODE L0235]             advantages: Tensor::<AB, 1>::from_floats([1.0, -0.5], &device),
[CODE L0236]             base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
[CODE L0237]             targets: make_dummy_targets::<AB>(&device, 2),
[CODE L0238]             exit_target: None,
[CODE L0239]             exit_mask: None,
[CODE L0240]         };
[CODE L0241]         let cfg = RlConfig::default_phase2().with_lr(1e-3);
[CODE L0242]         let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
[CODE L0243]         let mut opt = AdamConfig::new().init();
[CODE L0244] 
[CODE L0245]         let (model_after, _) = rl_step(model, &batch, &cfg, &loss_fn, &mut opt);
[CODE L0246] 
[CODE L0247]         let out_after = model_after.forward(obs);
[CODE L0248]         let val_after: f32 = out_after.value.clone().mean().into_scalar().elem();
[CODE L0249] 
[CODE L0250]         assert!(
[CODE L0251]             (val_before - val_after).abs() > 1e-8,
[CODE L0252]             "one ACH epoch must change weights: before={val_before}, after={val_after}"
[CODE L0253]         );
[CODE L0254]     }
[CODE L0255] }
```

## Artifact 22 — Search-feature bridge and delta_q plumbing
Artifact id: `bridge-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-core/src/bridge.rs`
Why it matters: Important code surface for search-as-feature wiring and search-derived labels. Gives the research agent the real bridge between doctrine and code when deciding if DCRL-level method attention is warranted.

```rust
[CODE L0001] //! Bridge between riichienv-core game state and Hydra's observation encoder.
[CODE L0002] //!
[CODE L0003] //! Converts riichienv [`Observation`] data into the encoder's input types,
[CODE L0004] //! then runs the full fixed-superset encoding pipeline. This is the critical glue
[CODE L0005] //! between the game engine and the neural network.
[CODE L0006] 
[CODE L0007] use riichienv_core::observation::Observation;
[CODE L0008] use riichienv_core::observation_ref::ObservationRef;
[CODE L0009] use riichienv_core::shanten::calc_shanten_from_counts;
[CODE L0010] use riichienv_core::types::MeldType as RiichiMeldType;
[CODE L0011] 
[CODE L0012] use crate::afbs::{AfbsTree, NodeIdx};
[CODE L0013] use crate::ct_smc::CtSmc;
[CODE L0014] use crate::encoder::{
[CODE L0015]     DiscardEntry, DoraInfo, GameMetadata, MeldInfo, MeldType, OBS_SIZE, ObservationEncoder,
[CODE L0016]     PlayerDiscards, PlayerMelds, SearchFeaturePlanes,
[CODE L0017] };
[CODE L0018] use crate::hand_ev::{HandEvFeatures, compute_hand_ev};
[CODE L0019] use crate::safety::SafetyInfo;
[CODE L0020] use crate::sinkhorn::MixtureSib;
[CODE L0021] use crate::tile::NUM_TILE_TYPES;
[CODE L0022] 
[CODE L0023] const NUM_OPPONENTS: usize = 3;
[CODE L0024] const NUM_MIXTURE_COMPONENTS: usize = 4;
[CODE L0025] const NUM_BELIEF_ZONES: usize = 4;
[CODE L0026] 
[CODE L0027] /// Optional runtime context used to populate Group C search/belief channels.
[CODE L0028] ///
[CODE L0029] /// This allows the fixed-superset encoder to consume real belief/search features
[CODE L0030] /// when they are available, while preserving a backward-safe path when they are not.
[CODE L0031] #[derive(Clone, Copy, Default)]
[CODE L0032] pub struct SearchContext<'a> {
[CODE L0033]     /// Optional Mixture-SIB belief state.
[CODE L0034]     pub mixture: Option<&'a MixtureSib>,
[CODE L0035]     /// Optional CT-SMC posterior used for belief-weighted Hand-EV counts.
[CODE L0036]     pub ct_smc: Option<&'a CtSmc>,
[CODE L0037]     /// Optional AFBS search tree.
[CODE L0038]     pub afbs_tree: Option<&'a AfbsTree>,
[CODE L0039]     /// Optional AFBS root node corresponding to `afbs_tree`.
[CODE L0040]     pub afbs_root: Option<NodeIdx>,
[CODE L0041]     /// Optional externally produced per-opponent tile risk planes.
[CODE L0042]     pub opponent_risk: Option<&'a [[f32; NUM_TILE_TYPES]; NUM_OPPONENTS]>,
[CODE L0043]     /// Optional externally produced per-opponent scalar stress values.
[CODE L0044]     pub opponent_stress: Option<&'a [f32; NUM_OPPONENTS]>,
[CODE L0045] }
[CODE L0046] 
[CODE L0047] /// Convert a 136-format tile ID (u32) to its 34-format tile type (u8).
[CODE L0048] #[inline]
[CODE L0049] fn tile136_to_type(tile136: u32) -> u8 {
[CODE L0050]     (tile136 / 4) as u8
[CODE L0051] }
[CODE L0052] 
[CODE L0053] #[inline]
[CODE L0054] fn aka_flags_from_tiles<I>(tiles: I) -> [bool; 3]
[CODE L0055] where
[CODE L0056]     I: IntoIterator<Item = u8>,
[CODE L0057] {
[CODE L0058]     let mut aka_flags = [false; 3];
[CODE L0059]     for tile in tiles {
[CODE L0060]         match tile {
[CODE L0061]             16 => aka_flags[0] = true,
[CODE L0062]             52 => aka_flags[1] = true,
[CODE L0063]             88 => aka_flags[2] = true,
[CODE L0064]             _ => {}
[CODE L0065]         }
[CODE L0066]     }
[CODE L0067]     aka_flags
[CODE L0068] }
[CODE L0069] 
[CODE L0070] #[inline]
[CODE L0071] fn dora_info_from_parts<I, J>(indicator_tiles: I, observer_tiles: J) -> DoraInfo
[CODE L0072] where
[CODE L0073]     I: IntoIterator<Item = u8>,
[CODE L0074]     J: IntoIterator<Item = u8>,
[CODE L0075] {
[CODE L0076]     let mut indicators = [0u8; 5];
[CODE L0077]     let mut indicator_count = 0u8;
[CODE L0078]     for (idx, tile) in indicator_tiles.into_iter().take(5).enumerate() {
[CODE L0079]         indicators[idx] = tile;
[CODE L0080]         indicator_count += 1;
[CODE L0081]     }
[CODE L0082] 
[CODE L0083]     DoraInfo {
[CODE L0084]         indicators,
[CODE L0085]         indicator_count,
[CODE L0086]         aka_flags: aka_flags_from_tiles(observer_tiles),
[CODE L0087]     }
[CODE L0088] }
[CODE L0089] 
[CODE L0090] #[inline]
[CODE L0091] fn metadata_from_parts(
[CODE L0092]     observer: usize,
[CODE L0093]     riichi_declared: &[bool; 4],
[CODE L0094]     scores: &[i32; 4],
[CODE L0095]     kyoku_index: u8,
[CODE L0096]     honba: u8,
[CODE L0097]     kyotaku: u8,
[CODE L0098]     hand_counts: &[u8; NUM_TILE_TYPES],
[CODE L0099] ) -> GameMetadata {
[CODE L0100]     let hand_total: u8 = hand_counts.iter().sum();
[CODE L0101]     let len_div3 = hand_total / 3;
[CODE L0102]     let shanten = calc_shanten_from_counts(hand_counts, len_div3);
[CODE L0103] 
[CODE L0104]     GameMetadata {
[CODE L0105]         riichi: std::array::from_fn(|i| riichi_declared[(observer + i) % 4]),
[CODE L0106]         scores: std::array::from_fn(|i| scores[(observer + i) % 4]),
[CODE L0107]         shanten,
[CODE L0108]         kyoku_index,
[CODE L0109]         honba,
[CODE L0110]         kyotaku,
[CODE L0111]     }
[CODE L0112] }
[CODE L0113] 
[CODE L0114] #[inline]
[CODE L0115] #[allow(clippy::too_many_arguments)]
[CODE L0116] fn encode_extracted_observation(
[CODE L0117]     encoder: &mut ObservationEncoder,
[CODE L0118]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0119]     drawn_tile: Option<u8>,
[CODE L0120]     open_meld_counts: &[u8; NUM_TILE_TYPES],
[CODE L0121]     discards: &[PlayerDiscards; 4],
[CODE L0122]     melds: &[PlayerMelds; 4],
[CODE L0123]     dora: &DoraInfo,
[CODE L0124]     meta: &GameMetadata,
[CODE L0125]     safety: &SafetyInfo,
[CODE L0126]     search_context: &SearchContext<'_>,
[CODE L0127] ) -> [f32; OBS_SIZE] {
[CODE L0128]     let hand_ev = compute_hand_ev_from_context(hand, discards, melds, dora, search_context);
[CODE L0129]     let search_features = build_search_features(safety, search_context);
[CODE L0130]     let slice = encoder.encode_with_context(
[CODE L0131]         hand,
[CODE L0132]         drawn_tile,
[CODE L0133]         open_meld_counts,
[CODE L0134]         discards,
[CODE L0135]         melds,
[CODE L0136]         dora,
[CODE L0137]         meta,
[CODE L0138]         safety,
[CODE L0139]         Some(&search_features),
[CODE L0140]         Some(&hand_ev),
[CODE L0141]     );
[CODE L0142]     *slice
[CODE L0143] }
[CODE L0144] 
[CODE L0145] /// Extract hand tile counts from an Observation.
[CODE L0146] ///
[CODE L0147] /// Only the observer's own hand is meaningful (opponents' hands are hidden).
[CODE L0148] /// Converts from 136-format `Vec<u32>` to 34-bin histogram `[u8; 34]`.
[CODE L0149] #[inline]
[CODE L0150] pub fn extract_hand(obs: &Observation) -> [u8; NUM_TILE_TYPES] {
[CODE L0151]     let observer = obs.player_id as usize;
[CODE L0152]     let mut counts = [0u8; NUM_TILE_TYPES];
[CODE L0153]     for &tile136 in &obs.hands[observer] {
[CODE L0154]         let t = tile136_to_type(tile136) as usize;
[CODE L0155]         if t < NUM_TILE_TYPES {
[CODE L0156]             counts[t] = counts[t].saturating_add(1);
[CODE L0157]         }
[CODE L0158]     }
[CODE L0159]     counts
[CODE L0160] }
[CODE L0161] 
[CODE L0162] /// Extract discard info for all 4 players from an Observation.
[CODE L0163] ///
[CODE L0164] /// Player indices are RELATIVE to the observer (index 0 = observer).
[CODE L0165] /// Uses `tsumogiri_flags` to determine tedashi (tedashi = !tsumogiri).
[CODE L0166] #[inline]
[CODE L0167] pub fn extract_discards(obs: &Observation) -> [PlayerDiscards; 4] {
[CODE L0168]     let observer = obs.player_id as usize;
[CODE L0169]     std::array::from_fn(|relative_idx| {
[CODE L0170]         let abs = (observer + relative_idx) % 4;
[CODE L0171]         let disc = &obs.discards[abs];
[CODE L0172]         let tsumogiri = &obs.tsumogiri_flags[abs];
[CODE L0173]         let mut pd = PlayerDiscards::new();
[CODE L0174]         for (turn, &tile136) in disc.iter().enumerate() {
[CODE L0175]             let is_tsumogiri = tsumogiri.get(turn).copied().unwrap_or(false);
[CODE L0176]             pd.push(DiscardEntry {
[CODE L0177]                 tile: tile136_to_type(tile136),
[CODE L0178]                 is_tedashi: !is_tsumogiri,
[CODE L0179]                 turn: turn as u16,
[CODE L0180]             });
[CODE L0181]         }
[CODE L0182]         pd
[CODE L0183]     })
[CODE L0184] }
[CODE L0185] 
[CODE L0186] /// Extract meld info for all 4 players from an Observation.
[CODE L0187] ///
[CODE L0188] /// Maps riichienv `MeldType` variants to the encoder's three-category system:
[CODE L0189] /// - Chi -> `MeldType::Chi`
[CODE L0190] /// - Pon -> `MeldType::Pon`
[CODE L0191] /// - Daiminkan/Ankan/Kakan -> `MeldType::Kan` (all kan variants merged)
[CODE L0192] ///
[CODE L0193] /// Meld tile IDs are converted from 136-format (u8) to 34-format tile types.
[CODE L0194] #[inline]
[CODE L0195] pub fn extract_melds(obs: &Observation) -> [PlayerMelds; 4] {
[CODE L0196]     let observer = obs.player_id as usize;
[CODE L0197]     std::array::from_fn(|relative_idx| {
[CODE L0198]         let abs = (observer + relative_idx) % 4;
[CODE L0199]         let mut pm = PlayerMelds::new();
[CODE L0200]         for meld in &obs.melds[abs] {
[CODE L0201]             let mut tiles = [0u8; 4];
[CODE L0202]             let tile_count = meld.tile_count;
[CODE L0203]             for (i, &t) in meld.tiles_slice().iter().enumerate() {
[CODE L0204]                 tiles[i] = t / 4;
[CODE L0205]             }
[CODE L0206]             let meld_type = match meld.meld_type {
[CODE L0207]                 RiichiMeldType::Chi => MeldType::Chi,
[CODE L0208]                 RiichiMeldType::Pon => MeldType::Pon,
[CODE L0209]                 RiichiMeldType::Daiminkan | RiichiMeldType::Ankan | RiichiMeldType::Kakan => {
[CODE L0210]                     MeldType::Kan
[CODE L0211]                 }
[CODE L0212]             };
[CODE L0213]             pm.push(MeldInfo {
[CODE L0214]                 tiles,
[CODE L0215]                 tile_count,
[CODE L0216]                 meld_type,
[CODE L0217]             });
[CODE L0218]         }
[CODE L0219]         pm
[CODE L0220]     })
[CODE L0221] }
[CODE L0222] 
[CODE L0223] /// Count tile types across the observer's melds for channel 4-7 encoding.
[CODE L0224] ///
[CODE L0225] /// Returns a 34-element histogram where each entry is the number of tiles
[CODE L0226] /// of that type present in the observer's open/called melds.
[CODE L0227] #[inline]
[CODE L0228] pub fn extract_observer_meld_counts(obs: &Observation) -> [u8; NUM_TILE_TYPES] {
[CODE L0229]     let observer = obs.player_id as usize;
[CODE L0230]     let mut counts = [0u8; NUM_TILE_TYPES];
[CODE L0231]     for meld in &obs.melds[observer] {
[CODE L0232]         for &tile in meld.tiles_slice() {
[CODE L0233]             let t = (tile / 4) as usize;
[CODE L0234]             if t < NUM_TILE_TYPES {
[CODE L0235]                 counts[t] = counts[t].saturating_add(1);
[CODE L0236]             }
[CODE L0237]         }
[CODE L0238]     }
[CODE L0239]     counts
[CODE L0240] }
[CODE L0241] 
[CODE L0242] /// Extract dora information from an Observation.
[CODE L0243] ///
[CODE L0244] /// Converts dora indicator tile IDs from 136-format to 34-format tile types.
[CODE L0245] /// Scans the observer's hand for aka dora (red fives) at 136-format
[CODE L0246] /// indices 16 (5m), 52 (5p), 88 (5s).
[CODE L0247] #[inline]
[CODE L0248] pub fn extract_dora(obs: &Observation) -> DoraInfo {
[CODE L0249]     let observer = obs.player_id as usize;
[CODE L0250]     dora_info_from_parts(
[CODE L0251]         obs.dora_indicators.iter().copied().map(tile136_to_type),
[CODE L0252]         obs.hands[observer].iter().copied().map(|tile| tile as u8),
[CODE L0253]     )
[CODE L0254] }
[CODE L0255] 
[CODE L0256] /// Extract game metadata from an Observation.
[CODE L0257] ///
[CODE L0258] /// Computes shanten from the observer's hand counts. All player-relative
[CODE L0259] /// fields (riichi, scores) are rotated so index 0 = observer,
[CODE L0260] /// index 1 = shimocha, etc.
[CODE L0261] #[inline]
[CODE L0262] pub fn extract_metadata(obs: &Observation, hand_counts: &[u8; NUM_TILE_TYPES]) -> GameMetadata {
[CODE L0263]     let observer = obs.player_id as usize;
[CODE L0264]     metadata_from_parts(
[CODE L0265]         observer,
[CODE L0266]         &obs.riichi_declared,
[CODE L0267]         &obs.scores,
[CODE L0268]         obs.kyoku_index,
[CODE L0269]         obs.honba,
[CODE L0270]         obs.riichi_sticks.min(255) as u8,
[CODE L0271]         hand_counts,
[CODE L0272]     )
[CODE L0273] }
[CODE L0274] 
[CODE L0275] /// Compute public-state remaining tile counts for the observer.
[CODE L0276] ///
[CODE L0277] /// This subtracts all tiles visible to the observer: their concealed hand,
[CODE L0278] /// all open melds, all discards, and visible dora indicators. This is a safe
[CODE L0279] /// bridge-side approximation for Hand-EV features until belief-weighted
[CODE L0280] /// remaining counts from CT-SMC are threaded into the encoder path.
[CODE L0281] #[inline]
[CODE L0282] pub fn extract_public_remaining_counts(
[CODE L0283]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0284]     discards: &[PlayerDiscards; 4],
[CODE L0285]     melds: &[PlayerMelds; 4],
[CODE L0286]     dora: &DoraInfo,
[CODE L0287] ) -> [f32; NUM_TILE_TYPES] {
[CODE L0288]     let mut remaining = [4.0f32; NUM_TILE_TYPES];
[CODE L0289] 
[CODE L0290]     for (tile, &count) in hand.iter().enumerate() {
[CODE L0291]         remaining[tile] -= count as f32;
[CODE L0292]     }
[CODE L0293]     for player_discards in discards {
[CODE L0294]         for entry in player_discards
[CODE L0295]             .discards
[CODE L0296]             .iter()
[CODE L0297]             .take(player_discards.len as usize)
[CODE L0298]         {
[CODE L0299]             remaining[entry.tile as usize] -= 1.0;
[CODE L0300]         }
[CODE L0301]     }
[CODE L0302]     for player_melds in melds {
[CODE L0303]         for meld in player_melds.melds.iter().take(player_melds.len as usize) {
[CODE L0304]             for &tile in meld.tiles.iter().take(meld.tile_count as usize) {
[CODE L0305]                 remaining[tile as usize] -= 1.0;
[CODE L0306]             }
[CODE L0307]         }
[CODE L0308]     }
[CODE L0309]     for &indicator in dora.indicators.iter().take(dora.indicator_count as usize) {
[CODE L0310]         remaining[indicator as usize] -= 1.0;
[CODE L0311]     }
[CODE L0312] 
[CODE L0313]     for value in &mut remaining {
[CODE L0314]         *value = value.max(0.0);
[CODE L0315]     }
[CODE L0316]     remaining
[CODE L0317] }
[CODE L0318] 
[CODE L0319] /// Compute bridge-side Hand-EV features from public-state remaining counts.
[CODE L0320] #[inline]
[CODE L0321] pub fn compute_public_hand_ev(
[CODE L0322]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0323]     discards: &[PlayerDiscards; 4],
[CODE L0324]     melds: &[PlayerMelds; 4],
[CODE L0325]     dora: &DoraInfo,
[CODE L0326] ) -> HandEvFeatures {
[CODE L0327]     let remaining = extract_public_remaining_counts(hand, discards, melds, dora);
[CODE L0328]     compute_hand_ev(hand, &remaining)
[CODE L0329] }
[CODE L0330] 
[CODE L0331] /// Compute wall-weighted remaining tile counts from a CT-SMC posterior.
[CODE L0332] #[inline]
[CODE L0333] pub fn extract_ct_smc_remaining_counts(ct_smc: &CtSmc) -> [f32; NUM_TILE_TYPES] {
[CODE L0334]     let mut remaining = [0.0f32; NUM_TILE_TYPES];
[CODE L0335]     if ct_smc.is_empty() {
[CODE L0336]         return remaining;
[CODE L0337]     }
[CODE L0338]     for (tile, slot) in remaining.iter_mut().enumerate() {
[CODE L0339]         *slot = ct_smc.weighted_mean_tile_count(tile as u8, 3);
[CODE L0340]     }
[CODE L0341]     remaining
[CODE L0342] }
[CODE L0343] 
[CODE L0344] /// Compute bridge-side Hand-EV features from CT-SMC belief-weighted counts.
[CODE L0345] #[inline]
[CODE L0346] pub fn compute_ct_smc_hand_ev(hand: &[u8; NUM_TILE_TYPES], ct_smc: &CtSmc) -> HandEvFeatures {
[CODE L0347]     let remaining = extract_ct_smc_remaining_counts(ct_smc);
[CODE L0348]     compute_hand_ev(hand, &remaining)
[CODE L0349] }
[CODE L0350] 
[CODE L0351] #[inline]
[CODE L0352] fn compute_hand_ev_from_context(
[CODE L0353]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0354]     discards: &[PlayerDiscards; 4],
[CODE L0355]     melds: &[PlayerMelds; 4],
[CODE L0356]     dora: &DoraInfo,
[CODE L0357]     search_context: &SearchContext<'_>,
[CODE L0358] ) -> HandEvFeatures {
[CODE L0359]     if let Some(ct_smc) = search_context.ct_smc
[CODE L0360]         && !ct_smc.is_empty()
[CODE L0361]     {
[CODE L0362]         return compute_ct_smc_hand_ev(hand, ct_smc);
[CODE L0363]     }
[CODE L0364]     compute_public_hand_ev(hand, discards, melds, dora)
[CODE L0365] }
[CODE L0366] 
[CODE L0367] /// Build fixed-shape Group C search/belief planes from available runtime context.
[CODE L0368] ///
[CODE L0369] /// Current sources:
[CODE L0370] /// - Mixture-SIB -> belief fields, weights, entropy, ESS
[CODE L0371] /// - AFBS root -> discard-level delta-Q summary for expanded discard actions
[CODE L0372] /// - safety/opponent model cache -> per-opponent stress and matagi danger fallback
[CODE L0373] /// - explicit robust risk/stress overrides when provided
[CODE L0374] #[inline]
[CODE L0375] pub fn build_search_features(
[CODE L0376]     safety: &SafetyInfo,
[CODE L0377]     context: &SearchContext<'_>,
[CODE L0378] ) -> SearchFeaturePlanes {
[CODE L0379]     let mut features = SearchFeaturePlanes::default();
[CODE L0380] 
[CODE L0381]     if let Some(mixture) = context.mixture {
[CODE L0382]         let weights = mixture.weights();
[CODE L0383]         let mut ranked: [usize; NUM_MIXTURE_COMPONENTS] = std::array::from_fn(|idx| idx);
[CODE L0384]         ranked.sort_by(|&a, &b| {
[CODE L0385]             weights[b]
[CODE L0386]                 .partial_cmp(&weights[a])
[CODE L0387]                 .unwrap_or(std::cmp::Ordering::Equal)
[CODE L0388]         });
[CODE L0389] 
[CODE L0390]         for (rank, component_idx) in ranked.iter().copied().enumerate() {
[CODE L0391]             let weight = weights[component_idx];
[CODE L0392]             features.mixture_weights[rank] = weight as f32;
[CODE L0393]             for zone in 0..NUM_BELIEF_ZONES {
[CODE L0394]                 let channel = rank * NUM_BELIEF_ZONES + zone;
[CODE L0395]                 for tile in 0..NUM_TILE_TYPES {
[CODE L0396]                     features.belief_fields[channel][tile] = mixture.components[component_idx].belief
[CODE L0397]                         [tile * NUM_BELIEF_ZONES + zone]
[CODE L0398]                         as f32;
[CODE L0399]                 }
[CODE L0400]             }
[CODE L0401]         }
[CODE L0402] 
[CODE L0403]         features.mixture_entropy = mixture.weight_entropy() as f32;
[CODE L0404]         features.mixture_ess = mixture.ess() as f32;
[CODE L0405]         features.belief_features_present = true;
[CODE L0406]         features.context_features_present = true;
[CODE L0407]     }
[CODE L0408] 
[CODE L0409]     if let (Some(tree), Some(root)) = (context.afbs_tree, context.afbs_root) {
[CODE L0410]         let root_q = tree.node_q_value(root);
[CODE L0411]         let mut any_delta_q = false;
[CODE L0412]         for action in 0..NUM_TILE_TYPES as u8 {
[CODE L0413]             if let Some(child) = tree.find_child_by_action(root, action) {
[CODE L0414]                 features.delta_q[action as usize] = tree.node_q_value(child) - root_q;
[CODE L0415]                 any_delta_q = true;
[CODE L0416]             }
[CODE L0417]         }
[CODE L0418]         if any_delta_q {
[CODE L0419]             features.search_features_present = true;
[CODE L0420]             features.context_features_present = true;
[CODE L0421]         }
[CODE L0422]     }
[CODE L0423] 
[CODE L0424]     for opp in 0..NUM_OPPONENTS {
[CODE L0425]         features.opponent_risk[opp] = safety.matagi[opp];
[CODE L0426]         features.opponent_stress[opp] = if safety.opponent_riichi[opp] {
[CODE L0427]             1.0
[CODE L0428]         } else {
[CODE L0429]             safety.cached_tenpai_prob[opp]
[CODE L0430]         };
[CODE L0431]     }
[CODE L0432] 
[CODE L0433]     if let Some(risk) = context.opponent_risk {
[CODE L0434]         features.opponent_risk = *risk;
[CODE L0435]     }
[CODE L0436]     if let Some(stress) = context.opponent_stress {
[CODE L0437]         features.opponent_stress = *stress;
[CODE L0438]     }
[CODE L0439] 
[CODE L0440]     let robust_signal_present = features
[CODE L0441]         .opponent_risk
[CODE L0442]         .iter()
[CODE L0443]         .flat_map(|plane| plane.iter())
[CODE L0444]         .any(|&v| v != 0.0)
[CODE L0445]         || features.opponent_stress.iter().any(|&v| v != 0.0);
[CODE L0446]     if robust_signal_present {
[CODE L0447]         features.robust_features_present = true;
[CODE L0448]         features.context_features_present = true;
[CODE L0449]     }
[CODE L0450] 
[CODE L0451]     features
[CODE L0452] }
[CODE L0453] 
[CODE L0454] /// Encode a full observation into the fixed-superset tensor with optional Group C runtime context.
[CODE L0455] #[inline]
[CODE L0456] pub fn encode_observation_with_search_context(
[CODE L0457]     encoder: &mut ObservationEncoder,
[CODE L0458]     obs: &Observation,
[CODE L0459]     safety: &SafetyInfo,
[CODE L0460]     drawn_tile: Option<u8>,
[CODE L0461]     search_context: &SearchContext<'_>,
[CODE L0462] ) -> [f32; OBS_SIZE] {
[CODE L0463]     let hand = extract_hand(obs);
[CODE L0464]     let discards = extract_discards(obs);
[CODE L0465]     let melds = extract_melds(obs);
[CODE L0466]     let open_meld_counts = extract_observer_meld_counts(obs);
[CODE L0467]     let dora = extract_dora(obs);
[CODE L0468]     let meta = extract_metadata(obs, &hand);
[CODE L0469]     encode_extracted_observation(
[CODE L0470]         encoder,
[CODE L0471]         &hand,
[CODE L0472]         drawn_tile,
[CODE L0473]         &open_meld_counts,
[CODE L0474]         &discards,
[CODE L0475]         &melds,
[CODE L0476]         &dora,
[CODE L0477]         &meta,
[CODE L0478]         safety,
[CODE L0479]         search_context,
[CODE L0480]     )
[CODE L0481] }
[CODE L0482] 
[CODE L0483] /// Encode a full observation into the fixed-superset tensor.
[CODE L0484] ///
[CODE L0485] /// This is the main bridge entry point. Extracts all components from
[CODE L0486] /// a riichienv [`Observation`], feeds them through the encoder pipeline,
[CODE L0487] /// and returns a reference to the filled fixed-superset observation buffer
[CODE L0488] /// (`[f32; OBS_SIZE]`, currently `192 x 34`).
[CODE L0489] ///
[CODE L0490] /// # Drawn tile limitation
[CODE L0491] ///
[CODE L0492] /// The drawn tile cannot be reliably determined from `Observation` alone.
[CODE L0493] /// Encode a full observation into the fixed-superset tensor.
[CODE L0494] ///
[CODE L0495] /// `drawn_tile` should be `Some(tile_type)` when the observer just drew a
[CODE L0496] /// tile (obtain from `GameState.drawn_tile` mapped to tile type via `/ 4`).
[CODE L0497] /// Pass `None` when no draw occurred or the information is unavailable.
[CODE L0498] #[inline]
[CODE L0499] pub fn encode_observation(
[CODE L0500]     encoder: &mut ObservationEncoder,
[CODE L0501]     obs: &Observation,
[CODE L0502]     safety: &SafetyInfo,
[CODE L0503]     drawn_tile: Option<u8>,
[CODE L0504] ) -> [f32; OBS_SIZE] {
[CODE L0505]     let search_context = SearchContext::default();
[CODE L0506]     encode_observation_with_search_context(encoder, obs, safety, drawn_tile, &search_context)
[CODE L0507] }
[CODE L0508] 
[CODE L0509] // ---------------------------------------------------------------------------
[CODE L0510] // ObservationRef extractors (zero-copy path)
[CODE L0511] // ---------------------------------------------------------------------------
[CODE L0512] 
[CODE L0513] /// Extract hand tile counts from an ObservationRef.
[CODE L0514] ///
[CODE L0515] /// Converts from 136-format `&[u8]` to 34-bin histogram.
[CODE L0516] #[inline]
[CODE L0517] pub fn extract_hand_ref(obs: &ObservationRef<'_>) -> [u8; NUM_TILE_TYPES] {
[CODE L0518]     let mut counts = [0u8; NUM_TILE_TYPES];
[CODE L0519]     for &tile136 in obs.observer_hand {
[CODE L0520]         let t = (tile136 / 4) as usize;
[CODE L0521]         if t < NUM_TILE_TYPES {
[CODE L0522]             counts[t] = counts[t].saturating_add(1);
[CODE L0523]         }
[CODE L0524]     }
[CODE L0525]     counts
[CODE L0526] }
[CODE L0527] 
[CODE L0528] /// Extract discard info for all 4 players from an ObservationRef.
[CODE L0529] ///
[CODE L0530] /// Player indices are RELATIVE to the observer (index 0 = observer).
[CODE L0531] #[inline]
[CODE L0532] pub fn extract_discards_ref(obs: &ObservationRef<'_>) -> [PlayerDiscards; 4] {
[CODE L0533]     let observer = obs.player_id as usize;
[CODE L0534]     std::array::from_fn(|relative_idx| {
[CODE L0535]         let abs = (observer + relative_idx) % 4;
[CODE L0536]         let disc = obs.discards[abs];
[CODE L0537]         let tedashi = obs.tsumogiri_flags[abs];
[CODE L0538]         let mut pd = PlayerDiscards::new();
[CODE L0539]         for (turn, &tile136) in disc.iter().enumerate() {
[CODE L0540]             pd.push(DiscardEntry {
[CODE L0541]                 tile: (tile136 / 4),
[CODE L0542]                 is_tedashi: tedashi.get(turn).copied().unwrap_or(false),
[CODE L0543]                 turn: turn as u16,
[CODE L0544]             });
[CODE L0545]         }
[CODE L0546]         pd
[CODE L0547]     })
[CODE L0548] }
[CODE L0549] 
[CODE L0550] /// Extract meld info for all 4 players from an ObservationRef.
[CODE L0551] #[inline]
[CODE L0552] pub fn extract_melds_ref(obs: &ObservationRef<'_>) -> [PlayerMelds; 4] {
[CODE L0553]     let observer = obs.player_id as usize;
[CODE L0554]     std::array::from_fn(|relative_idx| {
[CODE L0555]         let abs = (observer + relative_idx) % 4;
[CODE L0556]         let mut pm = PlayerMelds::new();
[CODE L0557]         for meld in obs.melds[abs] {
[CODE L0558]             let mut tiles = [0u8; 4];
[CODE L0559]             let tile_count = meld.tile_count;
[CODE L0560]             for (i, &t) in meld.tiles_slice().iter().enumerate() {
[CODE L0561]                 tiles[i] = t / 4;
[CODE L0562]             }
[CODE L0563]             let meld_type = match meld.meld_type {
[CODE L0564]                 RiichiMeldType::Chi => MeldType::Chi,
[CODE L0565]                 RiichiMeldType::Pon => MeldType::Pon,
[CODE L0566]                 RiichiMeldType::Daiminkan | RiichiMeldType::Ankan | RiichiMeldType::Kakan => {
[CODE L0567]                     MeldType::Kan
[CODE L0568]                 }
[CODE L0569]             };
[CODE L0570]             pm.push(MeldInfo {
[CODE L0571]                 tiles,
[CODE L0572]                 tile_count,
[CODE L0573]                 meld_type,
[CODE L0574]             });
[CODE L0575]         }
[CODE L0576]         pm
[CODE L0577]     })
[CODE L0578] }
[CODE L0579] 
[CODE L0580] /// Count tile types across the observer's melds from an ObservationRef.
[CODE L0581] #[inline]
[CODE L0582] pub fn extract_observer_meld_counts_ref(obs: &ObservationRef<'_>) -> [u8; NUM_TILE_TYPES] {
[CODE L0583]     let observer = obs.player_id as usize;
[CODE L0584]     let mut counts = [0u8; NUM_TILE_TYPES];
[CODE L0585]     for meld in obs.melds[observer] {
[CODE L0586]         for &tile in meld.tiles_slice() {
[CODE L0587]             let t = (tile / 4) as usize;
[CODE L0588]             if t < NUM_TILE_TYPES {
[CODE L0589]                 counts[t] = counts[t].saturating_add(1);
[CODE L0590]             }
[CODE L0591]         }
[CODE L0592]     }
[CODE L0593]     counts
[CODE L0594] }
[CODE L0595] 
[CODE L0596] /// Extract dora information from an ObservationRef.
[CODE L0597] #[inline]
[CODE L0598] pub fn extract_dora_ref(obs: &ObservationRef<'_>) -> DoraInfo {
[CODE L0599]     dora_info_from_parts(
[CODE L0600]         obs.dora_indicators.iter().copied().map(|tile| tile / 4),
[CODE L0601]         obs.observer_hand.iter().copied(),
[CODE L0602]     )
[CODE L0603] }
[CODE L0604] 
[CODE L0605] /// Extract game metadata from an ObservationRef.
[CODE L0606] #[inline]
[CODE L0607] pub fn extract_metadata_ref(
[CODE L0608]     obs: &ObservationRef<'_>,
[CODE L0609]     hand_counts: &[u8; NUM_TILE_TYPES],
[CODE L0610] ) -> GameMetadata {
[CODE L0611]     metadata_from_parts(
[CODE L0612]         obs.player_id as usize,
[CODE L0613]         &obs.riichi_declared,
[CODE L0614]         &obs.scores,
[CODE L0615]         obs.kyoku_index,
[CODE L0616]         obs.honba,
[CODE L0617]         obs.riichi_sticks.min(255) as u8,
[CODE L0618]         hand_counts,
[CODE L0619]     )
[CODE L0620] }
[CODE L0621] 
[CODE L0622] /// Compute public-state remaining tile counts from a zero-copy observation.
[CODE L0623] #[inline]
[CODE L0624] pub fn extract_public_remaining_counts_ref(
[CODE L0625]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0626]     discards: &[PlayerDiscards; 4],
[CODE L0627]     melds: &[PlayerMelds; 4],
[CODE L0628]     dora: &DoraInfo,
[CODE L0629] ) -> [f32; NUM_TILE_TYPES] {
[CODE L0630]     extract_public_remaining_counts(hand, discards, melds, dora)
[CODE L0631] }
[CODE L0632] 
[CODE L0633] /// Compute bridge-side Hand-EV features from a zero-copy observation path.
[CODE L0634] #[inline]
[CODE L0635] pub fn compute_public_hand_ev_ref(
[CODE L0636]     hand: &[u8; NUM_TILE_TYPES],
[CODE L0637]     discards: &[PlayerDiscards; 4],
[CODE L0638]     melds: &[PlayerMelds; 4],
[CODE L0639]     dora: &DoraInfo,
[CODE L0640] ) -> HandEvFeatures {
[CODE L0641]     compute_public_hand_ev(hand, discards, melds, dora)
[CODE L0642] }
[CODE L0643] 
[CODE L0644] /// Encode a zero-copy observation into the fixed-superset tensor with optional Group C runtime context.
[CODE L0645] #[inline]
[CODE L0646] pub fn encode_observation_ref_with_search_context(
[CODE L0647]     encoder: &mut ObservationEncoder,
[CODE L0648]     obs: &ObservationRef<'_>,
[CODE L0649]     safety: &SafetyInfo,
[CODE L0650]     search_context: &SearchContext<'_>,
[CODE L0651] ) -> [f32; OBS_SIZE] {
[CODE L0652]     let hand = extract_hand_ref(obs);
[CODE L0653]     let discards = extract_discards_ref(obs);
[CODE L0654]     let melds = extract_melds_ref(obs);
[CODE L0655]     let open_meld_counts = extract_observer_meld_counts_ref(obs);
[CODE L0656]     let dora = extract_dora_ref(obs);
[CODE L0657]     let meta = extract_metadata_ref(obs, &hand);
[CODE L0658]     let drawn_tile = obs.drawn_tile.map(|t| t / 4);
[CODE L0659]     encode_extracted_observation(
[CODE L0660]         encoder,
[CODE L0661]         &hand,
[CODE L0662]         drawn_tile,
[CODE L0663]         &open_meld_counts,
[CODE L0664]         &discards,
[CODE L0665]         &melds,
[CODE L0666]         &dora,
[CODE L0667]         &meta,
[CODE L0668]         safety,
[CODE L0669]         search_context,
[CODE L0670]     )
[CODE L0671] }
[CODE L0672] 
[CODE L0673] /// Encode directly from a zero-copy observation reference.
[CODE L0674] ///
[CODE L0675] /// This bypasses `get_observation()` and its ~15 Vec allocations.
[CODE L0676] /// The `drawn_tile` from `ObservationRef` is automatically converted
[CODE L0677] /// from 136-format to tile type (/ 4).
[CODE L0678] #[inline]
[CODE L0679] pub fn encode_observation_ref(
[CODE L0680]     encoder: &mut ObservationEncoder,
[CODE L0681]     obs: &ObservationRef<'_>,
[CODE L0682]     safety: &SafetyInfo,
[CODE L0683] ) -> [f32; OBS_SIZE] {
[CODE L0684]     let search_context = SearchContext::default();
[CODE L0685]     encode_observation_ref_with_search_context(encoder, obs, safety, &search_context)
[CODE L0686] }
[CODE L0687] 
[CODE L0688] #[cfg(test)]
[CODE L0689] mod tests {
[CODE L0690]     use super::*;
[CODE L0691]     use riichienv_core::action::{Action, ActionType};
[CODE L0692]     use riichienv_core::rule::GameRule;
[CODE L0693]     use riichienv_core::state::GameState;
[CODE L0694] 
[CODE L0695]     /// Create a fresh observation from a newly dealt game.
[CODE L0696]     fn fresh_obs() -> Observation {
[CODE L0697]         let rule = GameRule::default_tenhou();
[CODE L0698]         let mut state = GameState::new(0, true, Some(42), 0, rule);
[CODE L0699]         state.get_observation(0)
[CODE L0700]     }
[CODE L0701] 
[CODE L0702]     #[test]
[CODE L0703]     fn extract_hand_has_13_or_14_tiles() {
[CODE L0704]         let obs = fresh_obs();
[CODE L0705]         let hand = extract_hand(&obs);
[CODE L0706]         let total: u8 = hand.iter().sum();
[CODE L0707]         assert!(
[CODE L0708]             (13..=14).contains(&total),
[CODE L0709]             "hand has {total} tiles, expected 13 or 14",
[CODE L0710]         );
[CODE L0711]     }
[CODE L0712] 
[CODE L0713]     #[test]
[CODE L0714]     fn extract_discards_initially_empty() {
[CODE L0715]         let obs = fresh_obs();
[CODE L0716]         let discards = extract_discards(&obs);
[CODE L0717]         for pd in &discards {
[CODE L0718]             assert_eq!(pd.len, 0);
[CODE L0719]         }
[CODE L0720]     }
[CODE L0721] 
[CODE L0722]     #[test]
[CODE L0723]     fn extract_discards_ref_matches_owned_observation_tedashi_flags() {
[CODE L0724]         let rule = GameRule::default_tenhou();
[CODE L0725]         let mut state = GameState::new(0, true, Some(42), 0, rule);
[CODE L0726]         let pid = state.current_player;
[CODE L0727]         if let Some(tile136) = state.players[pid as usize].hand_slice().first().copied() {
[CODE L0728]             let mut actions = [None; 4];
[CODE L0729]             actions[pid as usize] =
[CODE L0730]                 Some(Action::new(ActionType::Discard, Some(tile136), &[], None));
[CODE L0731]             state.step_unchecked(&actions);
[CODE L0732]         }
[CODE L0733] 
[CODE L0734]         let owned = extract_discards(&state.get_observation(state.current_player));
[CODE L0735]         let observed = state.observe(state.current_player);
[CODE L0736]         let borrowed = extract_discards_ref(&observed);
[CODE L0737] 
[CODE L0738]         for rel in 0..4 {
[CODE L0739]             assert_eq!(owned[rel].len, borrowed[rel].len);
[CODE L0740]             for idx in 0..owned[rel].len as usize {
[CODE L0741]                 assert_eq!(
[CODE L0742]                     owned[rel].as_slice()[idx].tile,
[CODE L0743]                     borrowed[rel].as_slice()[idx].tile
[CODE L0744]                 );
[CODE L0745]                 assert_eq!(
[CODE L0746]                     owned[rel].as_slice()[idx].is_tedashi,
[CODE L0747]                     borrowed[rel].as_slice()[idx].is_tedashi
[CODE L0748]                 );
[CODE L0749]             }
[CODE L0750]         }
[CODE L0751]     }
[CODE L0752] 
[CODE L0753]     #[test]
[CODE L0754]     fn extract_melds_initially_empty() {
[CODE L0755]         let obs = fresh_obs();
[CODE L0756]         let melds = extract_melds(&obs);
[CODE L0757]         for player_melds in &melds {
[CODE L0758]             assert_eq!(player_melds.len, 0);
[CODE L0759]         }
[CODE L0760]     }
[CODE L0761] 
[CODE L0762]     #[test]
[CODE L0763]     fn extract_dora_has_one_indicator() {
[CODE L0764]         let obs = fresh_obs();
[CODE L0765]         let dora = extract_dora(&obs);
[CODE L0766]         assert_eq!(dora.indicator_count, 1, "initial game has 1 dora indicator");
[CODE L0767]         assert!(dora.indicators[0] < 34, "tile type must be 0-33");
[CODE L0768]     }
[CODE L0769] 
[CODE L0770]     #[test]
[CODE L0771]     fn extract_metadata_sane_values() {
[CODE L0772]         let obs = fresh_obs();
[CODE L0773]         let hand = extract_hand(&obs);
[CODE L0774]         let meta = extract_metadata(&obs, &hand);
[CODE L0775]         assert_eq!(meta.kyoku_index, obs.kyoku_index);
[CODE L0776]         assert_eq!(meta.honba, 0);
[CODE L0777]         assert_eq!(meta.kyotaku, 0);
[CODE L0778]         // Shanten for a dealt hand should be reasonable (-1 to 8)
[CODE L0779]         assert!(
[CODE L0780]             (-1..=8).contains(&meta.shanten),
[CODE L0781]             "shanten {} out of range",
[CODE L0782]             meta.shanten,
[CODE L0783]         );
[CODE L0784]     }
[CODE L0785] 
[CODE L0786]     #[test]
[CODE L0787]     fn extract_observer_meld_counts_initially_zero() {
[CODE L0788]         let obs = fresh_obs();
[CODE L0789]         let counts = extract_observer_meld_counts(&obs);
[CODE L0790]         assert_eq!(counts.iter().sum::<u8>(), 0, "no melds at game start");
[CODE L0791]     }
[CODE L0792] 
[CODE L0793]     #[test]
[CODE L0794]     fn encode_observation_produces_nonzero() {
[CODE L0795]         let obs = fresh_obs();
[CODE L0796]         let safety = SafetyInfo::new();
[CODE L0797]         let mut encoder = ObservationEncoder::new();
[CODE L0798]         let result = encode_observation(&mut encoder, &obs, &safety, None);
[CODE L0799]         let nonzero = result.iter().filter(|&&v| v != 0.0).count();
[CODE L0800]         assert!(
[CODE L0801]             nonzero > 0,
[CODE L0802]             "encoded observation should have nonzero values"
[CODE L0803]         );
[CODE L0804]     }
[CODE L0805] 
[CODE L0806]     #[test]
[CODE L0807]     fn public_remaining_counts_subtract_visible_tiles() {
[CODE L0808]         let mut hand = [0u8; NUM_TILE_TYPES];
[CODE L0809]         hand[0] = 2;
[CODE L0810]         hand[1] = 1;
[CODE L0811] 
[CODE L0812]         let mut discards = std::array::from_fn(|_| PlayerDiscards::new());
[CODE L0813]         discards[0].push(DiscardEntry {
[CODE L0814]             tile: 0,
[CODE L0815]             is_tedashi: true,
[CODE L0816]             turn: 0,
[CODE L0817]         });
[CODE L0818] 
[CODE L0819]         let mut melds = std::array::from_fn(|_| PlayerMelds::new());
[CODE L0820]         melds[1].push(MeldInfo {
[CODE L0821]             tiles: [1, 1, 1, 0],
[CODE L0822]             tile_count: 3,
[CODE L0823]             meld_type: MeldType::Pon,
[CODE L0824]         });
[CODE L0825] 
[CODE L0826]         let dora = DoraInfo {
[CODE L0827]             indicators: [0, 0, 0, 0, 0],
[CODE L0828]             indicator_count: 1,
[CODE L0829]             aka_flags: [false; 3],
[CODE L0830]         };
[CODE L0831] 
[CODE L0832]         let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
[CODE L0833]         assert_eq!(
[CODE L0834]             remaining[0], 0.0,
[CODE L0835]             "2 in hand + 1 discard + 1 dora indicator exhaust tile 0"
[CODE L0836]         );
[CODE L0837]         assert_eq!(remaining[1], 0.0, "1 in hand + pon exhaust tile 1");
[CODE L0838]         assert_eq!(
[CODE L0839]             remaining[2], 4.0,
[CODE L0840]             "unseen tile should keep full remaining count"
[CODE L0841]         );
[CODE L0842]     }
[CODE L0843] 
[CODE L0844]     #[test]
[CODE L0845]     fn compute_public_hand_ev_on_real_observation_has_signal() {
[CODE L0846]         let obs = fresh_obs();
[CODE L0847]         let hand = extract_hand(&obs);
[CODE L0848]         let discards = extract_discards(&obs);
[CODE L0849]         let melds = extract_melds(&obs);
[CODE L0850]         let dora = extract_dora(&obs);
[CODE L0851]         let hand_ev = compute_public_hand_ev(&hand, &discards, &melds, &dora);
[CODE L0852] 
[CODE L0853]         let any_tenpai = hand_ev
[CODE L0854]             .tenpai_prob
[CODE L0855]             .iter()
[CODE L0856]             .flat_map(|p| p.iter())
[CODE L0857]             .any(|&v| v > 0.0);
[CODE L0858]         let any_ukeire = hand_ev
[CODE L0859]             .ukeire
[CODE L0860]             .iter()
[CODE L0861]             .flat_map(|u| u.iter())
[CODE L0862]             .any(|&v| v > 0.0);
[CODE L0863] 
[CODE L0864]         assert!(
[CODE L0865]             any_tenpai || any_ukeire,
[CODE L0866]             "public Hand-EV should expose some nonzero signal"
[CODE L0867]         );
[CODE L0868]     }
[CODE L0869] 
[CODE L0870]     #[test]
[CODE L0871]     fn extract_ct_smc_remaining_counts_uses_wall_column_only() {
[CODE L0872]         let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
[CODE L0873]         smc.particles = vec![
[CODE L0874]             crate::ct_smc::Particle {
[CODE L0875]                 allocation: {
[CODE L0876]                     let mut allocation = [[0u8; 4]; 34];
[CODE L0877]                     allocation[3] = [1, 1, 0, 0];
[CODE L0878]                     allocation[7] = [0, 0, 1, 0];
[CODE L0879]                     allocation
[CODE L0880]                 },
[CODE L0881]                 log_weight: 0.0,
[CODE L0882]             },
[CODE L0883]             crate::ct_smc::Particle {
[CODE L0884]                 allocation: {
[CODE L0885]                     let mut allocation = [[0u8; 4]; 34];
[CODE L0886]                     allocation[3] = [0, 1, 1, 1];
[CODE L0887]                     allocation
[CODE L0888]                 },
[CODE L0889]                 log_weight: 0.0,
[CODE L0890]             },
[CODE L0891]         ];
[CODE L0892] 
[CODE L0893]         let remaining = extract_ct_smc_remaining_counts(&smc);
[CODE L0894]         assert!((remaining[3] - 0.5).abs() < 1e-6);
[CODE L0895]         assert_eq!(remaining[7], 0.0);
[CODE L0896]         assert_eq!(remaining[2], 0.0);
[CODE L0897]     }
[CODE L0898] 
[CODE L0899]     #[test]
[CODE L0900]     fn compute_ct_smc_hand_ev_uses_weighted_remaining_counts() {
[CODE L0901]         let mut hand = [0u8; NUM_TILE_TYPES];
[CODE L0902]         hand[0] = 1;
[CODE L0903]         hand[1] = 1;
[CODE L0904] 
[CODE L0905]         let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
[CODE L0906]         smc.particles = vec![
[CODE L0907]             crate::ct_smc::Particle {
[CODE L0908]                 allocation: {
[CODE L0909]                     let mut allocation = [[0u8; 4]; 34];
[CODE L0910]                     allocation[0] = [1, 0, 0, 0];
[CODE L0911]                     allocation
[CODE L0912]                 },
[CODE L0913]                 log_weight: 0.0,
[CODE L0914]             },
[CODE L0915]             crate::ct_smc::Particle {
[CODE L0916]                 allocation: {
[CODE L0917]                     let mut allocation = [[0u8; 4]; 34];
[CODE L0918]                     allocation[0] = [0, 0, 0, 1];
[CODE L0919]                     allocation
[CODE L0920]                 },
[CODE L0921]                 log_weight: 0.0,
[CODE L0922]             },
[CODE L0923]         ];
[CODE L0924] 
[CODE L0925]         let features = compute_ct_smc_hand_ev(&hand, &smc);
[CODE L0926]         assert!(features.ukeire[1][0] > 0.0);
[CODE L0927]         assert!(features.expected_score[1] > 0.0);
[CODE L0928]     }
[CODE L0929] 
[CODE L0930]     #[test]
[CODE L0931]     fn build_search_features_from_mixture_populates_belief_and_weights() {
[CODE L0932]         let kernel = [1.0f64; 136];
[CODE L0933]         let row_sums = [4.0f64; 34];
[CODE L0934]         let col_sums = [34.0f64; 4];
[CODE L0935]         let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
[CODE L0936]         mixture.bayesian_update(&[1.5, 0.5, -0.5, -1.5]);
[CODE L0937] 
[CODE L0938]         let mut safety = SafetyInfo::new();
[CODE L0939]         safety.set_tenpai_prediction(0, 0.6);
[CODE L0940]         safety.on_discard(5, 1, true);
[CODE L0941] 
[CODE L0942]         let context = SearchContext {
[CODE L0943]             mixture: Some(&mixture),
[CODE L0944]             ..SearchContext::default()
[CODE L0945]         };
[CODE L0946]         let features = build_search_features(&safety, &context);
[CODE L0947] 
[CODE L0948]         assert!(features.belief_features_present);
[CODE L0949]         assert!(features.context_features_present);
[CODE L0950]         assert!(features.mixture_weights.iter().any(|&v| v > 0.0));
[CODE L0951]         assert!(features.mixture_entropy > 0.0);
[CODE L0952]         assert!(features.mixture_ess > 0.0);
[CODE L0953]         assert!(features.belief_fields.iter().flatten().any(|&v| v > 0.0));
[CODE L0954]         assert!(features.opponent_risk[1][4] > 0.0 || features.opponent_risk[1][6] > 0.0);
[CODE L0955]         assert!((features.opponent_stress[0] - 0.6).abs() < 1e-6);
[CODE L0956]     }
[CODE L0957] 
[CODE L0958]     #[test]
[CODE L0959]     fn build_search_features_from_afbs_populates_delta_q() {
[CODE L0960]         let mut tree = AfbsTree::new();
[CODE L0961]         let root = tree.add_node(100, 1.0, false);
[CODE L0962]         tree.nodes[root as usize].visit_count = 10;
[CODE L0963]         tree.nodes[root as usize].total_value = 4.0; // q = 0.4
[CODE L0964] 
[CODE L0965]         let child_a = tree.add_node(101, 0.6, false);
[CODE L0966]         tree.nodes[child_a as usize].visit_count = 4;
[CODE L0967]         tree.nodes[child_a as usize].total_value = 3.2; // q = 0.8
[CODE L0968] 
[CODE L0969]         let child_b = tree.add_node(105, 0.4, false);
[CODE L0970]         tree.nodes[child_b as usize].visit_count = 4;
[CODE L0971]         tree.nodes[child_b as usize].total_value = 0.4; // q = 0.1
[CODE L0972] 
[CODE L0973]         tree.nodes[root as usize].children = vec![(0, child_a), (5, child_b)].into();
[CODE L0974] 
[CODE L0975]         let context = SearchContext {
[CODE L0976]             afbs_tree: Some(&tree),
[CODE L0977]             afbs_root: Some(root),
[CODE L0978]             ..SearchContext::default()
[CODE L0979]         };
[CODE L0980]         let features = build_search_features(&SafetyInfo::new(), &context);
[CODE L0981] 
[CODE L0982]         assert!(features.search_features_present);
[CODE L0983]         assert!(features.context_features_present);
[CODE L0984]         assert!((features.delta_q[0] - 0.4).abs() < 1e-6);
[CODE L0985]         assert!((features.delta_q[5] + 0.3).abs() < 1e-6);
[CODE L0986]     }
[CODE L0987] 
[CODE L0988]     #[test]
[CODE L0989]     fn encode_observation_populates_hand_ev_planes() {
[CODE L0990]         let obs = fresh_obs();
[CODE L0991]         let safety = SafetyInfo::new();
[CODE L0992]         let mut encoder = ObservationEncoder::new();
[CODE L0993]         let result = encode_observation(&mut encoder, &obs, &safety, None);
[CODE L0994] 
[CODE L0995]         let mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
[CODE L0996]         assert_eq!(
[CODE L0997]             result[mask_offset], 1.0,
[CODE L0998]             "Hand-EV presence mask should be enabled"
[CODE L0999]         );
[CODE L1000] 
[CODE L1001]         let hand_ev_payload =
[CODE L1002]             &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..mask_offset];
[CODE L1003]         let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
[CODE L1004]         assert!(
[CODE L1005]             nonzero > 0,
[CODE L1006]             "encoded observation should contain nonzero Hand-EV payload"
[CODE L1007]         );
[CODE L1008]     }
[CODE L1009] 
[CODE L1010]     #[test]
[CODE L1011]     fn encode_observation_with_search_context_populates_group_c_planes() {
[CODE L1012]         let obs = fresh_obs();
[CODE L1013]         let mut safety = SafetyInfo::new();
[CODE L1014]         safety.set_tenpai_prediction(0, 0.7);
[CODE L1015]         safety.on_discard(5, 1, true);
[CODE L1016] 
[CODE L1017]         let kernel = [1.0f64; 136];
[CODE L1018]         let row_sums = [4.0f64; 34];
[CODE L1019]         let col_sums = [34.0f64; 4];
[CODE L1020]         let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
[CODE L1021]         mixture.bayesian_update(&[1.0, 0.0, -0.5, -1.0]);
[CODE L1022] 
[CODE L1023]         let mut tree = AfbsTree::new();
[CODE L1024]         let root = tree.add_node(7, 1.0, false);
[CODE L1025]         tree.nodes[root as usize].visit_count = 10;
[CODE L1026]         tree.nodes[root as usize].total_value = 2.0;
[CODE L1027]         let child = tree.add_node(11, 1.0, false);
[CODE L1028]         tree.nodes[child as usize].visit_count = 5;
[CODE L1029]         tree.nodes[child as usize].total_value = 3.0;
[CODE L1030]         tree.nodes[root as usize].children = vec![(0, child)].into();
[CODE L1031] 
[CODE L1032]         let context = SearchContext {
[CODE L1033]             mixture: Some(&mixture),
[CODE L1034]             afbs_tree: Some(&tree),
[CODE L1035]             afbs_root: Some(root),
[CODE L1036]             ..SearchContext::default()
[CODE L1037]         };
[CODE L1038] 
[CODE L1039]         let mut encoder = ObservationEncoder::new();
[CODE L1040]         let result =
[CODE L1041]             encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);
[CODE L1042] 
[CODE L1043]         let belief_mask = crate::encoder::SEARCH_MASK_CHANNEL_START * NUM_TILE_TYPES;
[CODE L1044]         let search_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 1) * NUM_TILE_TYPES;
[CODE L1045]         let robust_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 2) * NUM_TILE_TYPES;
[CODE L1046]         assert_eq!(result[belief_mask], 1.0);
[CODE L1047]         assert_eq!(result[search_mask], 1.0);
[CODE L1048]         assert_eq!(result[robust_mask], 1.0);
[CODE L1049] 
[CODE L1050]         let belief_payload = result[crate::encoder::SEARCH_BELIEF_CHANNEL_START * NUM_TILE_TYPES
[CODE L1051]             ..crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES]
[CODE L1052]             .iter()
[CODE L1053]             .filter(|&&v| v != 0.0)
[CODE L1054]             .count();
[CODE L1055]         let delta_q_payload = result[crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES];
[CODE L1056]         assert!(
[CODE L1057]             belief_payload > 0,
[CODE L1058]             "belief/search payload should be nonzero"
[CODE L1059]         );
[CODE L1060]         assert!(
[CODE L1061]             delta_q_payload > 0.0,
[CODE L1062]             "delta-q channel should reflect AFBS context"
[CODE L1063]         );
[CODE L1064]     }
[CODE L1065] 
[CODE L1066]     #[test]
[CODE L1067]     fn encode_observation_with_ct_smc_context_uses_belief_weighted_hand_ev() {
[CODE L1068]         let obs = fresh_obs();
[CODE L1069]         let safety = SafetyInfo::new();
[CODE L1070]         let hand = extract_hand(&obs);
[CODE L1071] 
[CODE L1072]         let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
[CODE L1073]         smc.particles = vec![
[CODE L1074]             crate::ct_smc::Particle {
[CODE L1075]                 allocation: {
[CODE L1076]                     let mut allocation = [[0u8; 4]; 34];
[CODE L1077]                     for tile in 0..NUM_TILE_TYPES {
[CODE L1078]                         if hand[tile] == 0 {
[CODE L1079]                             allocation[tile][3] = 1;
[CODE L1080]                         }
[CODE L1081]                     }
[CODE L1082]                     allocation
[CODE L1083]                 },
[CODE L1084]                 log_weight: 0.0,
[CODE L1085]             },
[CODE L1086]             crate::ct_smc::Particle {
[CODE L1087]                 allocation: {
[CODE L1088]                     let mut allocation = [[0u8; 4]; 34];
[CODE L1089]                     for tile in 0..NUM_TILE_TYPES {
[CODE L1090]                         if hand[tile] == 0 && tile % 2 == 0 {
[CODE L1091]                             allocation[tile][2] = 1;
[CODE L1092]                         }
[CODE L1093]                     }
[CODE L1094]                     allocation
[CODE L1095]                 },
[CODE L1096]                 log_weight: 0.0,
[CODE L1097]             },
[CODE L1098]         ];
[CODE L1099] 
[CODE L1100]         let context = SearchContext {
[CODE L1101]             ct_smc: Some(&smc),
[CODE L1102]             ..SearchContext::default()
[CODE L1103]         };
[CODE L1104] 
[CODE L1105]         let mut encoder = ObservationEncoder::new();
[CODE L1106]         let result =
[CODE L1107]             encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);
[CODE L1108] 
[CODE L1109]         let hand_ev_payload = &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES
[CODE L1110]             ..crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES];
[CODE L1111]         let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
[CODE L1112]         assert!(
[CODE L1113]             nonzero > 0,
[CODE L1114]             "CT-SMC context should produce nonzero Hand-EV payload"
[CODE L1115]         );
[CODE L1116]     }
[CODE L1117] 
[CODE L1118]     #[test]
[CODE L1119]     fn tile136_to_type_basics() {
[CODE L1120]         assert_eq!(tile136_to_type(0), 0); // 1m copy 0
[CODE L1121]         assert_eq!(tile136_to_type(3), 0); // 1m copy 3
[CODE L1122]         assert_eq!(tile136_to_type(4), 1); // 2m copy 0
[CODE L1123]         assert_eq!(tile136_to_type(135), 33); // chun copy 3
[CODE L1124]     }
[CODE L1125] 
[CODE L1126]     #[test]
[CODE L1127]     fn aka_flags_and_dora_parts_detect_red_fives_and_cap_indicators() {
[CODE L1128]         assert_eq!(aka_flags_from_tiles([16, 52, 88]), [true, true, true]);
[CODE L1129]         assert_eq!(aka_flags_from_tiles([0, 4, 8]), [false, false, false]);
[CODE L1130] 
[CODE L1131]         let dora = dora_info_from_parts([1, 2, 3, 4, 5, 6], [16, 0, 52]);
[CODE L1132]         assert_eq!(dora.indicators, [1, 2, 3, 4, 5]);
[CODE L1133]         assert_eq!(dora.indicator_count, 5);
[CODE L1134]         assert_eq!(dora.aka_flags, [true, true, false]);
[CODE L1135]     }
[CODE L1136] 
[CODE L1137]     #[test]
[CODE L1138]     fn metadata_parts_rotate_relative_state_and_compute_shanten() {
[CODE L1139]         let mut hand_counts = [0u8; NUM_TILE_TYPES];
[CODE L1140]         hand_counts[0] = 3;
[CODE L1141]         hand_counts[1] = 3;
[CODE L1142]         hand_counts[2] = 3;
[CODE L1143]         hand_counts[27] = 2;
[CODE L1144]         hand_counts[28] = 2;
[CODE L1145] 
[CODE L1146]         let meta = metadata_from_parts(
[CODE L1147]             2,
[CODE L1148]             &[true, false, true, false],
[CODE L1149]             &[25000, 26000, 27000, 28000],
[CODE L1150]             3,
[CODE L1151]             1,
[CODE L1152]             2,
[CODE L1153]             &hand_counts,
[CODE L1154]         );
[CODE L1155] 
[CODE L1156]         assert_eq!(meta.riichi, [true, false, true, false]);
[CODE L1157]         assert_eq!(meta.scores, [27000, 28000, 25000, 26000]);
[CODE L1158]         assert_eq!(meta.kyoku_index, 3);
[CODE L1159]         assert_eq!(meta.honba, 1);
[CODE L1160]         assert_eq!(meta.kyotaku, 2);
[CODE L1161]         assert!((-1..=8).contains(&meta.shanten));
[CODE L1162]     }
[CODE L1163] 
[CODE L1164]     #[test]
[CODE L1165]     fn ct_smc_empty_and_context_fallbacks_use_safe_defaults() {
[CODE L1166]         let hand = [0u8; NUM_TILE_TYPES];
[CODE L1167]         let discards = std::array::from_fn(|_| PlayerDiscards::new());
[CODE L1168]         let melds = std::array::from_fn(|_| PlayerMelds::new());
[CODE L1169]         let dora = DoraInfo {
[CODE L1170]             indicators: [0; 5],
[CODE L1171]             indicator_count: 0,
[CODE L1172]             aka_flags: [false; 3],
[CODE L1173]         };
[CODE L1174]         let empty_smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(1));
[CODE L1175] 
[CODE L1176]         assert_eq!(
[CODE L1177]             extract_ct_smc_remaining_counts(&empty_smc),
[CODE L1178]             [0.0; NUM_TILE_TYPES]
[CODE L1179]         );
[CODE L1180] 
[CODE L1181]         let from_empty_context = compute_hand_ev_from_context(
[CODE L1182]             &hand,
[CODE L1183]             &discards,
[CODE L1184]             &melds,
[CODE L1185]             &dora,
[CODE L1186]             &SearchContext::default(),
[CODE L1187]         );
[CODE L1188]         let from_empty_smc = compute_hand_ev_from_context(
[CODE L1189]             &hand,
[CODE L1190]             &discards,
[CODE L1191]             &melds,
[CODE L1192]             &dora,
[CODE L1193]             &SearchContext {
[CODE L1194]                 ct_smc: Some(&empty_smc),
[CODE L1195]                 ..SearchContext::default()
[CODE L1196]             },
[CODE L1197]         );
[CODE L1198] 
[CODE L1199]         assert_eq!(from_empty_context.tenpai_prob, from_empty_smc.tenpai_prob);
[CODE L1200]         assert_eq!(from_empty_context.ukeire, from_empty_smc.ukeire);
[CODE L1201]         assert_eq!(
[CODE L1202]             from_empty_context.expected_score,
[CODE L1203]             from_empty_smc.expected_score
[CODE L1204]         );
[CODE L1205]     }
[CODE L1206] }
```

## Artifact 23 — Arena trajectory labels and validation surfaces
Artifact id: `arena-rs`
Source label: CODE
Type: `file_full`
Source: `crates/hydra-core/src/arena.rs`
Why it matters: Ground truth for how search labels and trajectory artifacts are validated at the core layer. Included so the research agent can judge implementation implications rather than staying at doc-level analogies.

```rust
[CODE L0001] //! Self-play arena: batch game simulation with trajectory collection.
[CODE L0002] 
[CODE L0003] use crate::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
[CODE L0004] use crate::encoder::OBS_SIZE;
[CODE L0005] 
[CODE L0006] #[derive(Clone, Copy, Debug, PartialEq)]
[CODE L0007] pub struct TrajectoryExitLabel {
[CODE L0008]     pub target: [f32; HYDRA_ACTION_SPACE],
[CODE L0009]     pub mask: [f32; HYDRA_ACTION_SPACE],
[CODE L0010] }
[CODE L0011] 
[CODE L0012] #[derive(Clone, Copy, Debug, PartialEq)]
[CODE L0013] pub struct TrajectoryDeltaQLabel {
[CODE L0014]     pub target: [f32; HYDRA_ACTION_SPACE],
[CODE L0015]     pub mask: [f32; HYDRA_ACTION_SPACE],
[CODE L0016] }
[CODE L0017] 
[CODE L0018] fn label_from_slices(
[CODE L0019]     target: &[f32],
[CODE L0020]     mask: &[f32],
[CODE L0021] ) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])> {
[CODE L0022]     if target.len() != HYDRA_ACTION_SPACE || mask.len() != HYDRA_ACTION_SPACE {
[CODE L0023]         return None;
[CODE L0024]     }
[CODE L0025]     let mut target_arr = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0026]     let mut mask_arr = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0027]     target_arr.copy_from_slice(target);
[CODE L0028]     mask_arr.copy_from_slice(mask);
[CODE L0029]     Some((target_arr, mask_arr))
[CODE L0030] }
[CODE L0031] 
[CODE L0032] fn label_to_vec_pair(
[CODE L0033]     target: [f32; HYDRA_ACTION_SPACE],
[CODE L0034]     mask: [f32; HYDRA_ACTION_SPACE],
[CODE L0035] ) -> (Vec<f32>, Vec<f32>) {
[CODE L0036]     (target.to_vec(), mask.to_vec())
[CODE L0037] }
[CODE L0038] 
[CODE L0039] fn masked_softmax_probs(
[CODE L0040]     logits: &[f32; HYDRA_ACTION_SPACE],
[CODE L0041]     legal_mask: &[bool; HYDRA_ACTION_SPACE],
[CODE L0042]     temperature: f32,
[CODE L0043] ) -> [f32; HYDRA_ACTION_SPACE] {
[CODE L0044]     let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
[CODE L0045]     let mut max_val = f32::NEG_INFINITY;
[CODE L0046]     for i in 0..HYDRA_ACTION_SPACE {
[CODE L0047]         if legal_mask[i] {
[CODE L0048]             adjusted[i] = logits[i] / temperature;
[CODE L0049]             if adjusted[i] > max_val {
[CODE L0050]                 max_val = adjusted[i];
[CODE L0051]             }
[CODE L0052]         }
[CODE L0053]     }
[CODE L0054] 
[CODE L0055]     let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0056]     let mut total = 0.0f32;
[CODE L0057]     for i in 0..HYDRA_ACTION_SPACE {
[CODE L0058]         if legal_mask[i] {
[CODE L0059]             probs[i] = (adjusted[i] - max_val).exp();
[CODE L0060]             total += probs[i];
[CODE L0061]         }
[CODE L0062]     }
[CODE L0063]     if total > 0.0 {
[CODE L0064]         for p in &mut probs {
[CODE L0065]             *p /= total;
[CODE L0066]         }
[CODE L0067]     }
[CODE L0068]     probs
[CODE L0069] }
[CODE L0070] 
[CODE L0071] impl TrajectoryDeltaQLabel {
[CODE L0072]     pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
[CODE L0073]         let (target, mask) = label_from_slices(target, mask)?;
[CODE L0074]         Some(Self { target, mask })
[CODE L0075]     }
[CODE L0076] 
[CODE L0077]     pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
[CODE L0078]         (self.target, self.mask)
[CODE L0079]     }
[CODE L0080] 
[CODE L0081]     pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
[CODE L0082]         label_to_vec_pair(self.target, self.mask)
[CODE L0083]     }
[CODE L0084] }
[CODE L0085] 
[CODE L0086] impl TrajectoryExitLabel {
[CODE L0087]     pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
[CODE L0088]         let (target, mask) = label_from_slices(target, mask)?;
[CODE L0089]         Some(Self { target, mask })
[CODE L0090]     }
[CODE L0091] 
[CODE L0092]     pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
[CODE L0093]         (self.target, self.mask)
[CODE L0094]     }
[CODE L0095] 
[CODE L0096]     pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
[CODE L0097]         label_to_vec_pair(self.target, self.mask)
[CODE L0098]     }
[CODE L0099] }
[CODE L0100] 
[CODE L0101] pub struct ArenaConfig {
[CODE L0102]     pub num_parallel_games: usize,
[CODE L0103]     pub game_mode: u8,
[CODE L0104]     pub temperature_range: (f32, f32),
[CODE L0105]     pub exit_fraction: f32,
[CODE L0106]     pub max_trajectory_buffer: usize,
[CODE L0107] }
[CODE L0108] 
[CODE L0109] impl ArenaConfig {
[CODE L0110]     pub fn summary(&self) -> String {
[CODE L0111]         format!(
[CODE L0112]             "arena(games={}, temp={:.1}-{:.1}, buf={})",
[CODE L0113]             self.num_parallel_games,
[CODE L0114]             self.temperature_range.0,
[CODE L0115]             self.temperature_range.1,
[CODE L0116]             self.max_trajectory_buffer
[CODE L0117]         )
[CODE L0118]     }
[CODE L0119] 
[CODE L0120]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0121]         if self.num_parallel_games == 0 {
[CODE L0122]             return Err("num_parallel_games > 0");
[CODE L0123]         }
[CODE L0124]         if self.max_trajectory_buffer == 0 {
[CODE L0125]             return Err("max_trajectory_buffer > 0");
[CODE L0126]         }
[CODE L0127]         if self.temperature_range.0 <= 0.0 {
[CODE L0128]             return Err("temperature range start > 0");
[CODE L0129]         }
[CODE L0130]         if self.temperature_range.1 < self.temperature_range.0 {
[CODE L0131]             return Err("temperature range end >= start");
[CODE L0132]         }
[CODE L0133]         Ok(())
[CODE L0134]     }
[CODE L0135] }
[CODE L0136] 
[CODE L0137] impl Default for ArenaConfig {
[CODE L0138]     fn default() -> Self {
[CODE L0139]         Self {
[CODE L0140]             num_parallel_games: 500,
[CODE L0141]             game_mode: 0,
[CODE L0142]             temperature_range: (0.5, 1.5),
[CODE L0143]             exit_fraction: 0.2,
[CODE L0144]             max_trajectory_buffer: 100_000,
[CODE L0145]         }
[CODE L0146]     }
[CODE L0147] }
[CODE L0148] 
[CODE L0149] pub struct SelfPlayConfig {
[CODE L0150]     pub arena: ArenaConfig,
[CODE L0151]     pub gae_gamma: f32,
[CODE L0152]     pub gae_lambda: f32,
[CODE L0153]     pub rebase_interval_hours: f32,
[CODE L0154] }
[CODE L0155] 
[CODE L0156] impl SelfPlayConfig {
[CODE L0157]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0158]         self.arena.validate()?;
[CODE L0159]         if self.gae_gamma <= 0.0 || self.gae_gamma >= 1.0 {
[CODE L0160]             return Err("gae_gamma in (0,1)");
[CODE L0161]         }
[CODE L0162]         if self.gae_lambda <= 0.0 || self.gae_lambda >= 1.0 {
[CODE L0163]             return Err("gae_lambda in (0,1)");
[CODE L0164]         }
[CODE L0165]         Ok(())
[CODE L0166]     }
[CODE L0167] }
[CODE L0168] 
[CODE L0169] impl SelfPlayConfig {
[CODE L0170]     pub fn with_games(mut self, n: usize) -> Self {
[CODE L0171]         self.arena.num_parallel_games = n;
[CODE L0172]         self
[CODE L0173]     }
[CODE L0174] 
[CODE L0175]     pub fn summary(&self) -> String {
[CODE L0176]         format!(
[CODE L0177]             "selfplay(games={}, gamma={:.3}, rebase={:.0}h)",
[CODE L0178]             self.arena.num_parallel_games, self.gae_gamma, self.rebase_interval_hours
[CODE L0179]         )
[CODE L0180]     }
[CODE L0181] }
[CODE L0182] 
[CODE L0183] impl Default for SelfPlayConfig {
[CODE L0184]     fn default() -> Self {
[CODE L0185]         Self {
[CODE L0186]             arena: ArenaConfig::default(),
[CODE L0187]             gae_gamma: 0.995,
[CODE L0188]             gae_lambda: 0.95,
[CODE L0189]             rebase_interval_hours: 37.5,
[CODE L0190]         }
[CODE L0191]     }
[CODE L0192] }
[CODE L0193] 
[CODE L0194] #[repr(C)]
[CODE L0195] pub struct TrajectoryStep {
[CODE L0196]     pub obs: [f32; OBS_SIZE],
[CODE L0197]     pub action: u8,
[CODE L0198]     pub pi_old: [f32; HYDRA_ACTION_SPACE],
[CODE L0199]     pub legal_mask: [bool; HYDRA_ACTION_SPACE],
[CODE L0200]     pub exit_label: Option<TrajectoryExitLabel>,
[CODE L0201]     pub delta_q_label: Option<TrajectoryDeltaQLabel>,
[CODE L0202]     pub reward: f32,
[CODE L0203]     pub done: bool,
[CODE L0204]     pub player_id: u8,
[CODE L0205]     pub game_id: u32,
[CODE L0206]     pub turn: u16,
[CODE L0207]     pub temperature: f32,
[CODE L0208] }
[CODE L0209] 
[CODE L0210] pub struct Trajectory {
[CODE L0211]     pub steps: Vec<TrajectoryStep>,
[CODE L0212]     pub final_scores: [i32; 4],
[CODE L0213]     pub game_id: u32,
[CODE L0214]     pub seed: u64,
[CODE L0215] }
[CODE L0216] 
[CODE L0217] pub struct Arena {
[CODE L0218]     pub config: ArenaConfig,
[CODE L0219]     pub trajectory_buffer: Vec<Trajectory>,
[CODE L0220]     pub games_completed: u64,
[CODE L0221] }
[CODE L0222] 
[CODE L0223] impl Arena {
[CODE L0224]     pub fn new(config: ArenaConfig) -> Self {
[CODE L0225]         Self {
[CODE L0226]             config,
[CODE L0227]             trajectory_buffer: Vec::new(),
[CODE L0228]             games_completed: 0,
[CODE L0229]         }
[CODE L0230]     }
[CODE L0231] 
[CODE L0232]     pub fn add_trajectory(&mut self, traj: Trajectory) {
[CODE L0233]         if self.trajectory_buffer.len() < self.config.max_trajectory_buffer {
[CODE L0234]             self.trajectory_buffer.push(traj);
[CODE L0235]         }
[CODE L0236]         self.games_completed += 1;
[CODE L0237]     }
[CODE L0238] 
[CODE L0239]     pub fn max_capacity(&self) -> usize {
[CODE L0240]         self.config.max_trajectory_buffer
[CODE L0241]     }
[CODE L0242] 
[CODE L0243]     pub fn is_full(&self) -> bool {
[CODE L0244]         self.trajectory_buffer.len() >= self.config.max_trajectory_buffer
[CODE L0245]     }
[CODE L0246] 
[CODE L0247]     pub fn completed_trajectories(&self) -> usize {
[CODE L0248]         self.trajectory_buffer
[CODE L0249]             .iter()
[CODE L0250]             .filter(|t| t.is_complete())
[CODE L0251]             .count()
[CODE L0252]     }
[CODE L0253] 
[CODE L0254]     pub fn total_steps(&self) -> usize {
[CODE L0255]         self.trajectory_buffer.iter().map(|t| t.steps.len()).sum()
[CODE L0256]     }
[CODE L0257] 
[CODE L0258]     pub fn num_buffered(&self) -> usize {
[CODE L0259]         self.trajectory_buffer.len()
[CODE L0260]     }
[CODE L0261] 
[CODE L0262]     pub fn validate_all(&self) -> Result<(), String> {
[CODE L0263]         for (i, traj) in self.trajectory_buffer.iter().enumerate() {
[CODE L0264]             traj.validate()
[CODE L0265]                 .map_err(|e| format!("trajectory {i}: {e}"))?;
[CODE L0266]         }
[CODE L0267]         Ok(())
[CODE L0268]     }
[CODE L0269] 
[CODE L0270]     pub fn drain_trajectories(&mut self) -> Vec<Trajectory> {
[CODE L0271]         std::mem::take(&mut self.trajectory_buffer)
[CODE L0272]     }
[CODE L0273] 
[CODE L0274]     pub fn mean_scores(&self) -> [f32; 4] {
[CODE L0275]         if self.trajectory_buffer.is_empty() {
[CODE L0276]             return [0.0; 4];
[CODE L0277]         }
[CODE L0278]         let n = self.trajectory_buffer.len() as f32;
[CODE L0279]         let mut sums = [0.0f32; 4];
[CODE L0280]         for t in &self.trajectory_buffer {
[CODE L0281]             for (i, &s) in t.final_scores.iter().enumerate() {
[CODE L0282]                 sums[i] += s as f32;
[CODE L0283]             }
[CODE L0284]         }
[CODE L0285]         for s in &mut sums {
[CODE L0286]             *s /= n;
[CODE L0287]         }
[CODE L0288]         sums
[CODE L0289]     }
[CODE L0290] 
[CODE L0291]     pub fn placement_distribution(&self, player_id: u8) -> [f32; 4] {
[CODE L0292]         if self.trajectory_buffer.is_empty() {
[CODE L0293]             return [0.25; 4];
[CODE L0294]         }
[CODE L0295]         let mut counts = [0u32; 4];
[CODE L0296]         let n = self.trajectory_buffer.len();
[CODE L0297]         for t in &self.trajectory_buffer {
[CODE L0298]             let mut scores_indexed: Vec<(i32, u8)> = t
[CODE L0299]                 .final_scores
[CODE L0300]                 .iter()
[CODE L0301]                 .enumerate()
[CODE L0302]                 .map(|(i, &s)| (s, i as u8))
[CODE L0303]                 .collect();
[CODE L0304]             scores_indexed.sort_by(|a, b| b.0.cmp(&a.0));
[CODE L0305]             for (rank, (_, idx)) in scores_indexed.iter().enumerate() {
[CODE L0306]                 if *idx == player_id && rank < 4 {
[CODE L0307]                     counts[rank] += 1;
[CODE L0308]                 }
[CODE L0309]             }
[CODE L0310]         }
[CODE L0311]         let mut dist = [0.0f32; 4];
[CODE L0312]         for (i, &c) in counts.iter().enumerate() {
[CODE L0313]             dist[i] = c as f32 / n as f32;
[CODE L0314]         }
[CODE L0315]         dist
[CODE L0316]     }
[CODE L0317] 
[CODE L0318]     pub fn compute_rewards(&self, player_id: u8) -> Vec<Vec<f32>> {
[CODE L0319]         self.trajectory_buffer
[CODE L0320]             .iter()
[CODE L0321]             .map(|t| {
[CODE L0322]                 t.steps
[CODE L0323]                     .iter()
[CODE L0324]                     .filter(|s| s.player_id == player_id)
[CODE L0325]                     .map(|s| s.reward)
[CODE L0326]                     .collect()
[CODE L0327]             })
[CODE L0328]             .collect()
[CODE L0329]     }
[CODE L0330] 
[CODE L0331]     pub fn reset(&mut self) {
[CODE L0332]         self.trajectory_buffer.clear();
[CODE L0333]         self.games_completed = 0;
[CODE L0334]     }
[CODE L0335] 
[CODE L0336]     pub fn mean_score_for(&self, player: u8) -> f32 {
[CODE L0337]         if self.trajectory_buffer.is_empty() {
[CODE L0338]             return 0.0;
[CODE L0339]         }
[CODE L0340]         let sum: f32 = self
[CODE L0341]             .trajectory_buffer
[CODE L0342]             .iter()
[CODE L0343]             .map(|t| t.score_for(player) as f32)
[CODE L0344]             .sum();
[CODE L0345]         sum / self.trajectory_buffer.len() as f32
[CODE L0346]     }
[CODE L0347] 
[CODE L0348]     pub fn score_variance(&self) -> f32 {
[CODE L0349]         if self.trajectory_buffer.is_empty() {
[CODE L0350]             return 0.0;
[CODE L0351]         }
[CODE L0352]         let means = self.mean_scores();
[CODE L0353]         let n = self.trajectory_buffer.len() as f32;
[CODE L0354]         let mut var = 0.0f32;
[CODE L0355]         for t in &self.trajectory_buffer {
[CODE L0356]             for (i, &s) in t.final_scores.iter().enumerate() {
[CODE L0357]                 var += (s as f32 - means[i]).powi(2);
[CODE L0358]             }
[CODE L0359]         }
[CODE L0360]         var / (n * 4.0)
[CODE L0361]     }
[CODE L0362] 
[CODE L0363]     pub fn mean_game_length(&self) -> f32 {
[CODE L0364]         if self.trajectory_buffer.is_empty() {
[CODE L0365]             return 0.0;
[CODE L0366]         }
[CODE L0367]         let total_turns: u32 = self
[CODE L0368]             .trajectory_buffer
[CODE L0369]             .iter()
[CODE L0370]             .map(|t| t.max_turn() as u32)
[CODE L0371]             .sum();
[CODE L0372]         total_turns as f32 / self.trajectory_buffer.len() as f32
[CODE L0373]     }
[CODE L0374] 
[CODE L0375]     pub fn latest_game_id(&self) -> Option<u32> {
[CODE L0376]         self.trajectory_buffer.last().map(|t| t.game_id)
[CODE L0377]     }
[CODE L0378] 
[CODE L0379]     pub fn mean_placement_for(&self, player_id: u8) -> f32 {
[CODE L0380]         if self.trajectory_buffer.is_empty() {
[CODE L0381]             return 2.5;
[CODE L0382]         }
[CODE L0383]         let sum: f32 = self
[CODE L0384]             .trajectory_buffer
[CODE L0385]             .iter()
[CODE L0386]             .map(|t| t.placement_for(player_id) as f32 + 1.0)
[CODE L0387]             .sum();
[CODE L0388]         sum / self.trajectory_buffer.len() as f32
[CODE L0389]     }
[CODE L0390] 
[CODE L0391]     pub fn fourth_place_count(&self, player_id: u8) -> usize {
[CODE L0392]         self.trajectory_buffer
[CODE L0393]             .iter()
[CODE L0394]             .filter(|t| t.placement_for(player_id) == 3)
[CODE L0395]             .count()
[CODE L0396]     }
[CODE L0397] 
[CODE L0398]     pub fn win_rate_for(&self, player_id: u8) -> f32 {
[CODE L0399]         if self.trajectory_buffer.is_empty() {
[CODE L0400]             return 0.0;
[CODE L0401]         }
[CODE L0402]         self.win_count(player_id) as f32 / self.trajectory_buffer.len() as f32
[CODE L0403]     }
[CODE L0404] 
[CODE L0405]     pub fn win_count(&self, player_id: u8) -> usize {
[CODE L0406]         self.trajectory_buffer
[CODE L0407]             .iter()
[CODE L0408]             .filter(|t| t.winner() == player_id)
[CODE L0409]             .count()
[CODE L0410]     }
[CODE L0411] 
[CODE L0412]     pub fn oldest_game_id(&self) -> Option<u32> {
[CODE L0413]         self.trajectory_buffer.first().map(|t| t.game_id)
[CODE L0414]     }
[CODE L0415] 
[CODE L0416]     pub fn utilization(&self) -> String {
[CODE L0417]         format!(
[CODE L0418]             "{}/{} ({:.0}%)",
[CODE L0419]             self.num_buffered(),
[CODE L0420]             self.max_capacity(),
[CODE L0421]             self.fill_ratio() * 100.0
[CODE L0422]         )
[CODE L0423]     }
[CODE L0424] 
[CODE L0425]     pub fn fill_ratio(&self) -> f32 {
[CODE L0426]         if self.config.max_trajectory_buffer == 0 {
[CODE L0427]             return 0.0;
[CODE L0428]         }
[CODE L0429]         self.trajectory_buffer.len() as f32 / self.config.max_trajectory_buffer as f32
[CODE L0430]     }
[CODE L0431] 
[CODE L0432]     pub fn avg_trajectory_length(&self) -> f32 {
[CODE L0433]         if self.trajectory_buffer.is_empty() {
[CODE L0434]             return 0.0;
[CODE L0435]         }
[CODE L0436]         self.total_steps() as f32 / self.trajectory_buffer.len() as f32
[CODE L0437]     }
[CODE L0438] 
[CODE L0439]     pub fn stats_summary(&self) -> String {
[CODE L0440]         format!(
[CODE L0441]             "games={} steps={} buffered={} complete={}",
[CODE L0442]             self.games_completed,
[CODE L0443]             self.total_steps(),
[CODE L0444]             self.num_buffered(),
[CODE L0445]             self.completed_trajectories()
[CODE L0446]         )
[CODE L0447]     }
[CODE L0448] 
[CODE L0449]     pub fn collect_player_steps(&self, player_id: u8) -> Vec<&TrajectoryStep> {
[CODE L0450]         self.trajectory_buffer
[CODE L0451]             .iter()
[CODE L0452]             .flat_map(|t| t.steps.iter())
[CODE L0453]             .filter(|s| s.player_id == player_id)
[CODE L0454]             .collect()
[CODE L0455]     }
[CODE L0456] }
[CODE L0457] 
[CODE L0458] impl Trajectory {
[CODE L0459]     pub fn num_steps(&self) -> usize {
[CODE L0460]         self.steps.len()
[CODE L0461]     }
[CODE L0462] 
[CODE L0463]     pub fn active_players(&self) -> Vec<u8> {
[CODE L0464]         let mut seen = [false; 4];
[CODE L0465]         for s in &self.steps {
[CODE L0466]             if (s.player_id as usize) < 4 {
[CODE L0467]                 seen[s.player_id as usize] = true;
[CODE L0468]             }
[CODE L0469]         }
[CODE L0470]         (0..4).filter(|&i| seen[i as usize]).collect()
[CODE L0471]     }
[CODE L0472] 
[CODE L0473]     pub fn score_delta(&self, player: u8) -> i32 {
[CODE L0474]         let mean = self.final_scores.iter().sum::<i32>() / 4;
[CODE L0475]         self.score_for(player) - mean
[CODE L0476]     }
[CODE L0477] 
[CODE L0478]     pub fn score_for(&self, player: u8) -> i32 {
[CODE L0479]         self.final_scores.get(player as usize).copied().unwrap_or(0)
[CODE L0480]     }
[CODE L0481] 
[CODE L0482]     pub fn placement_for(&self, player: u8) -> u8 {
[CODE L0483]         compute_placements(self.final_scores)[player as usize]
[CODE L0484]     }
[CODE L0485] 
[CODE L0486]     pub fn winner(&self) -> u8 {
[CODE L0487]         compute_placements(self.final_scores)
[CODE L0488]             .iter()
[CODE L0489]             .position(|&p| p == 0)
[CODE L0490]             .unwrap_or(0) as u8
[CODE L0491]     }
[CODE L0492] 
[CODE L0493]     pub fn max_turn(&self) -> u16 {
[CODE L0494]         self.steps.last().map_or(0, |s| s.turn)
[CODE L0495]     }
[CODE L0496] 
[CODE L0497]     pub fn player_reward_sum(&self, player_id: u8) -> f32 {
[CODE L0498]         self.steps
[CODE L0499]             .iter()
[CODE L0500]             .filter(|s| s.player_id == player_id)
[CODE L0501]             .map(|s| s.reward)
[CODE L0502]             .sum()
[CODE L0503]     }
[CODE L0504] 
[CODE L0505]     pub fn total_reward(&self) -> f32 {
[CODE L0506]         self.steps.iter().map(|s| s.reward).sum()
[CODE L0507]     }
[CODE L0508] 
[CODE L0509]     pub fn is_complete(&self) -> bool {
[CODE L0510]         self.steps.last().is_some_and(|s| s.done)
[CODE L0511]     }
[CODE L0512] 
[CODE L0513]     pub fn steps_for_player(&self, player_id: u8) -> Vec<&TrajectoryStep> {
[CODE L0514]         self.steps
[CODE L0515]             .iter()
[CODE L0516]             .filter(|s| s.player_id == player_id)
[CODE L0517]             .collect()
[CODE L0518]     }
[CODE L0519] 
[CODE L0520]     pub fn new(game_id: u32, seed: u64) -> Self {
[CODE L0521]         Self {
[CODE L0522]             steps: Vec::new(),
[CODE L0523]             final_scores: [0; 4],
[CODE L0524]             game_id,
[CODE L0525]             seed,
[CODE L0526]         }
[CODE L0527]     }
[CODE L0528] 
[CODE L0529]     pub fn validate(&self) -> Result<(), String> {
[CODE L0530]         if self.steps.is_empty() {
[CODE L0531]             return Err("trajectory has no steps".into());
[CODE L0532]         }
[CODE L0533]         for (i, step) in self.steps.iter().enumerate() {
[CODE L0534]             if step.player_id >= 4 {
[CODE L0535]                 return Err(format!("step {i}: invalid player_id {}", step.player_id));
[CODE L0536]             }
[CODE L0537]             if step.action as usize >= HYDRA_ACTION_SPACE {
[CODE L0538]                 return Err(format!("step {i}: invalid action {}", step.action));
[CODE L0539]             }
[CODE L0540]             if !step.legal_mask.iter().any(|&is_legal| is_legal) {
[CODE L0541]                 return Err(format!("step {i}: legal_mask has no legal actions"));
[CODE L0542]             }
[CODE L0543]             if !step.legal_mask[step.action as usize] {
[CODE L0544]                 return Err(format!(
[CODE L0545]                     "step {i}: selected action {} is not marked legal",
[CODE L0546]                     step.action
[CODE L0547]                 ));
[CODE L0548]             }
[CODE L0549]             let pi_sum: f32 = step.pi_old.iter().sum();
[CODE L0550]             if pi_sum > 0.0 && (pi_sum - 1.0).abs() > 0.05 {
[CODE L0551]                 return Err(format!("step {i}: pi_old sums to {pi_sum}"));
[CODE L0552]             }
[CODE L0553]             if let Some(exit_label) = step.exit_label {
[CODE L0554]                 let mut masked_mass = 0.0f32;
[CODE L0555]                 let mut saw_masked_action = false;
[CODE L0556]                 for action_idx in 0..HYDRA_ACTION_SPACE {
[CODE L0557]                     let mask_value = exit_label.mask[action_idx];
[CODE L0558]                     if mask_value < -1e-6 || (mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6 {
[CODE L0559]                         return Err(format!(
[CODE L0560]                             "step {i}: exit mask at action {action_idx} is not approximately binary ({mask_value})"
[CODE L0561]                         ));
[CODE L0562]                     }
[CODE L0563]                     let target_value = exit_label.target[action_idx];
[CODE L0564]                     if target_value < -1e-6 {
[CODE L0565]                         return Err(format!(
[CODE L0566]                             "step {i}: exit target at action {action_idx} is negative ({target_value})"
[CODE L0567]                         ));
[CODE L0568]                     }
[CODE L0569]                     if mask_value > 0.5 {
[CODE L0570]                         saw_masked_action = true;
[CODE L0571]                         if !step.legal_mask[action_idx] {
[CODE L0572]                             return Err(format!(
[CODE L0573]                                 "step {i}: exit label masks illegal action {action_idx}"
[CODE L0574]                             ));
[CODE L0575]                         }
[CODE L0576]                         if action_idx > DISCARD_END as usize {
[CODE L0577]                             return Err(format!(
[CODE L0578]                                 "step {i}: exit label masks non-discard action {action_idx}"
[CODE L0579]                             ));
[CODE L0580]                         }
[CODE L0581]                         if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
[CODE L0582]                             return Err(format!(
[CODE L0583]                                 "step {i}: exit label includes aka discard action {action_idx}"
[CODE L0584]                             ));
[CODE L0585]                         }
[CODE L0586]                         masked_mass += target_value;
[CODE L0587]                     } else if target_value.abs() > 1e-5 {
[CODE L0588]                         return Err(format!(
[CODE L0589]                             "step {i}: exit target has non-zero mass outside mask at action {action_idx}"
[CODE L0590]                         ));
[CODE L0591]                     }
[CODE L0592]                 }
[CODE L0593]                 if !saw_masked_action {
[CODE L0594]                     return Err(format!("step {i}: exit label mask is empty"));
[CODE L0595]                 }
[CODE L0596]                 if (masked_mass - 1.0).abs() > 1e-3 {
[CODE L0597]                     return Err(format!(
[CODE L0598]                         "step {i}: exit target mass over masked actions is {masked_mass}"
[CODE L0599]                     ));
[CODE L0600]                 }
[CODE L0601]             }
[CODE L0602]             if let Some(delta_q_label) = step.delta_q_label {
[CODE L0603]                 let mut saw_masked_action = false;
[CODE L0604]                 for action_idx in 0..HYDRA_ACTION_SPACE {
[CODE L0605]                     let mask_value = delta_q_label.mask[action_idx];
[CODE L0606]                     if mask_value < -1e-6 || (mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6 {
[CODE L0607]                         return Err(format!(
[CODE L0608]                             "step {i}: delta_q mask at action {action_idx} is not approximately binary ({mask_value})"
[CODE L0609]                         ));
[CODE L0610]                     }
[CODE L0611]                     let target_value = delta_q_label.target[action_idx];
[CODE L0612]                     if !target_value.is_finite() {
[CODE L0613]                         return Err(format!(
[CODE L0614]                             "step {i}: delta_q target at action {action_idx} is not finite ({target_value})"
[CODE L0615]                         ));
[CODE L0616]                     }
[CODE L0617]                     if mask_value > 0.5 {
[CODE L0618]                         saw_masked_action = true;
[CODE L0619]                         if !step.legal_mask[action_idx] {
[CODE L0620]                             return Err(format!(
[CODE L0621]                                 "step {i}: delta_q label masks illegal action {action_idx}"
[CODE L0622]                             ));
[CODE L0623]                         }
[CODE L0624]                         if action_idx > DISCARD_END as usize {
[CODE L0625]                             return Err(format!(
[CODE L0626]                                 "step {i}: delta_q label masks non-discard action {action_idx}"
[CODE L0627]                             ));
[CODE L0628]                         }
[CODE L0629]                         if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
[CODE L0630]                             return Err(format!(
[CODE L0631]                                 "step {i}: delta_q label includes aka discard action {action_idx}"
[CODE L0632]                             ));
[CODE L0633]                         }
[CODE L0634]                     } else if target_value.abs() > 1e-5 {
[CODE L0635]                         return Err(format!(
[CODE L0636]                             "step {i}: delta_q target has non-zero value outside mask at action {action_idx}"
[CODE L0637]                         ));
[CODE L0638]                     }
[CODE L0639]                 }
[CODE L0640]                 if !saw_masked_action {
[CODE L0641]                     return Err(format!("step {i}: delta_q label mask is empty"));
[CODE L0642]                 }
[CODE L0643]             }
[CODE L0644]         }
[CODE L0645]         Ok(())
[CODE L0646]     }
[CODE L0647] }
[CODE L0648] 
[CODE L0649] pub fn softmax_temperature(
[CODE L0650]     logits: &[f32; HYDRA_ACTION_SPACE],
[CODE L0651]     legal_mask: &[bool; HYDRA_ACTION_SPACE],
[CODE L0652]     temperature: f32,
[CODE L0653] ) -> [f32; HYDRA_ACTION_SPACE] {
[CODE L0654]     masked_softmax_probs(logits, legal_mask, temperature)
[CODE L0655] }
[CODE L0656] 
[CODE L0657] pub fn games_played(scores: &[[i32; 4]]) -> usize {
[CODE L0658]     scores.len()
[CODE L0659] }
[CODE L0660] 
[CODE L0661] pub fn total_score_sum(scores: &[[i32; 4]]) -> i64 {
[CODE L0662]     scores
[CODE L0663]         .iter()
[CODE L0664]         .flat_map(|s| s.iter())
[CODE L0665]         .map(|&s| s as i64)
[CODE L0666]         .sum()
[CODE L0667] }
[CODE L0668] 
[CODE L0669] pub fn score_std(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0670]     let mean = avg_score(scores, player);
[CODE L0671]     if scores.is_empty() {
[CODE L0672]         return 0.0;
[CODE L0673]     }
[CODE L0674]     let var: f32 = scores
[CODE L0675]         .iter()
[CODE L0676]         .map(|s| (s[player as usize] as f32 - mean).powi(2))
[CODE L0677]         .sum::<f32>()
[CODE L0678]         / scores.len() as f32;
[CODE L0679]     var.sqrt()
[CODE L0680] }
[CODE L0681] 
[CODE L0682] pub fn avg_score(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0683]     if scores.is_empty() {
[CODE L0684]         return 0.0;
[CODE L0685]     }
[CODE L0686]     scores
[CODE L0687]         .iter()
[CODE L0688]         .map(|s| s[player as usize] as f32)
[CODE L0689]         .sum::<f32>()
[CODE L0690]         / scores.len() as f32
[CODE L0691] }
[CODE L0692] 
[CODE L0693] pub fn top_two_rate(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0694]     if scores.is_empty() {
[CODE L0695]         return 0.0;
[CODE L0696]     }
[CODE L0697]     let top2 = scores
[CODE L0698]         .iter()
[CODE L0699]         .filter(|s| compute_placements(**s)[player as usize] <= 1)
[CODE L0700]         .count();
[CODE L0701]     top2 as f32 / scores.len() as f32
[CODE L0702] }
[CODE L0703] 
[CODE L0704] pub fn fourth_place_rate(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0705]     if scores.is_empty() {
[CODE L0706]         return 0.0;
[CODE L0707]     }
[CODE L0708]     let fourths = scores
[CODE L0709]         .iter()
[CODE L0710]         .filter(|s| compute_placements(**s)[player as usize] == 3)
[CODE L0711]         .count();
[CODE L0712]     fourths as f32 / scores.len() as f32
[CODE L0713] }
[CODE L0714] 
[CODE L0715] pub fn win_rate_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0716]     if scores.is_empty() {
[CODE L0717]         return 0.0;
[CODE L0718]     }
[CODE L0719]     let wins = scores
[CODE L0720]         .iter()
[CODE L0721]         .filter(|s| compute_placements(**s)[player as usize] == 0)
[CODE L0722]         .count();
[CODE L0723]     wins as f32 / scores.len() as f32
[CODE L0724] }
[CODE L0725] 
[CODE L0726] pub fn mean_placement_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
[CODE L0727]     if scores.is_empty() {
[CODE L0728]         return 2.5;
[CODE L0729]     }
[CODE L0730]     let sum: f32 = scores
[CODE L0731]         .iter()
[CODE L0732]         .map(|s| compute_placements(*s)[player as usize] as f32 + 1.0)
[CODE L0733]         .sum();
[CODE L0734]     sum / scores.len() as f32
[CODE L0735] }
[CODE L0736] 
[CODE L0737] pub fn compute_placements(scores: [i32; 4]) -> [u8; 4] {
[CODE L0738]     let mut indexed: [(i32, u8); 4] = [
[CODE L0739]         (scores[0], 0),
[CODE L0740]         (scores[1], 1),
[CODE L0741]         (scores[2], 2),
[CODE L0742]         (scores[3], 3),
[CODE L0743]     ];
[CODE L0744]     indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
[CODE L0745]     let mut placements = [0u8; 4];
[CODE L0746]     for (rank, &(_, player)) in indexed.iter().enumerate() {
[CODE L0747]         placements[player as usize] = rank as u8;
[CODE L0748]     }
[CODE L0749]     placements
[CODE L0750] }
[CODE L0751] 
[CODE L0752] pub fn greedy_action(
[CODE L0753]     logits: &[f32; HYDRA_ACTION_SPACE],
[CODE L0754]     legal_mask: &[bool; HYDRA_ACTION_SPACE],
[CODE L0755] ) -> u8 {
[CODE L0756]     let mut best = 0u8;
[CODE L0757]     let mut best_val = f32::NEG_INFINITY;
[CODE L0758]     for (i, (&l, &m)) in logits.iter().zip(legal_mask.iter()).enumerate() {
[CODE L0759]         if m && l > best_val {
[CODE L0760]             best_val = l;
[CODE L0761]             best = i as u8;
[CODE L0762]         }
[CODE L0763]     }
[CODE L0764]     best
[CODE L0765] }
[CODE L0766] 
[CODE L0767] pub fn sample_action_with_temperature(
[CODE L0768]     logits: &[f32; HYDRA_ACTION_SPACE],
[CODE L0769]     legal_mask: &[bool; HYDRA_ACTION_SPACE],
[CODE L0770]     temperature: f32,
[CODE L0771]     rng_val: f32,
[CODE L0772] ) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
[CODE L0773]     let probs = masked_softmax_probs(logits, legal_mask, temperature);
[CODE L0774]     let mut cumsum = 0.0f32;
[CODE L0775]     let mut chosen = 0u8;
[CODE L0776]     for (i, &p) in probs.iter().enumerate() {
[CODE L0777]         cumsum += p;
[CODE L0778]         if rng_val <= cumsum {
[CODE L0779]             chosen = i as u8;
[CODE L0780]             break;
[CODE L0781]         }
[CODE L0782]     }
[CODE L0783]     if !legal_mask[chosen as usize] {
[CODE L0784]         for (i, &m) in legal_mask.iter().enumerate() {
[CODE L0785]             if m {
[CODE L0786]                 chosen = i as u8;
[CODE L0787]                 break;
[CODE L0788]             }
[CODE L0789]         }
[CODE L0790]     }
[CODE L0791]     (chosen, probs)
[CODE L0792] }
[CODE L0793] 
[CODE L0794] #[cfg(test)]
[CODE L0795] mod tests {
[CODE L0796]     use super::*;
[CODE L0797] 
[CODE L0798]     fn legal_step(action: u8, player_id: u8, reward: f32, done: bool, turn: u16) -> TrajectoryStep {
[CODE L0799]         let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0800]         pi_old[action as usize] = 1.0;
[CODE L0801]         let mut legal_mask = [false; HYDRA_ACTION_SPACE];
[CODE L0802]         legal_mask[action as usize] = true;
[CODE L0803]         TrajectoryStep {
[CODE L0804]             obs: [0.0; OBS_SIZE],
[CODE L0805]             action,
[CODE L0806]             pi_old,
[CODE L0807]             legal_mask,
[CODE L0808]             exit_label: None,
[CODE L0809]             delta_q_label: None,
[CODE L0810]             reward,
[CODE L0811]             done,
[CODE L0812]             player_id,
[CODE L0813]             game_id: 0,
[CODE L0814]             turn,
[CODE L0815]             temperature: 1.0,
[CODE L0816]         }
[CODE L0817]     }
[CODE L0818] 
[CODE L0819]     #[test]
[CODE L0820]     fn labels_roundtrip_and_reject_wrong_lengths() {
[CODE L0821]         let target = vec![0.25f32; HYDRA_ACTION_SPACE];
[CODE L0822]         let mask = vec![1.0f32; HYDRA_ACTION_SPACE];
[CODE L0823] 
[CODE L0824]         let exit = TrajectoryExitLabel::from_slices(&target, &mask).expect("valid exit label");
[CODE L0825]         let delta =
[CODE L0826]             TrajectoryDeltaQLabel::from_slices(&target, &mask).expect("valid delta q label");
[CODE L0827] 
[CODE L0828]         let (exit_target, exit_mask) = exit.to_vec_pair();
[CODE L0829]         let (delta_target, delta_mask) = delta.to_vec_pair();
[CODE L0830]         assert_eq!(exit_target, target);
[CODE L0831]         assert_eq!(exit_mask, mask);
[CODE L0832]         assert_eq!(delta_target, target);
[CODE L0833]         assert_eq!(delta_mask, mask);
[CODE L0834] 
[CODE L0835]         assert!(TrajectoryExitLabel::from_slices(&target[..10], &mask).is_none());
[CODE L0836]         assert!(TrajectoryDeltaQLabel::from_slices(&target, &mask[..10]).is_none());
[CODE L0837]     }
[CODE L0838] 
[CODE L0839]     #[test]
[CODE L0840]     fn arena_and_selfplay_configs_validate_expected_bounds() {
[CODE L0841]         let mut arena_cfg = ArenaConfig::default();
[CODE L0842]         assert!(arena_cfg.validate().is_ok());
[CODE L0843]         assert!(arena_cfg.summary().contains("arena(games=500"));
[CODE L0844] 
[CODE L0845]         arena_cfg.num_parallel_games = 0;
[CODE L0846]         assert_eq!(arena_cfg.validate(), Err("num_parallel_games > 0"));
[CODE L0847] 
[CODE L0848]         let arena_cfg = ArenaConfig {
[CODE L0849]             max_trajectory_buffer: 0,
[CODE L0850]             ..ArenaConfig::default()
[CODE L0851]         };
[CODE L0852]         assert_eq!(arena_cfg.validate(), Err("max_trajectory_buffer > 0"));
[CODE L0853] 
[CODE L0854]         let mut selfplay = SelfPlayConfig::default().with_games(128);
[CODE L0855]         assert_eq!(selfplay.arena.num_parallel_games, 128);
[CODE L0856]         assert!(selfplay.summary().contains("selfplay(games=128"));
[CODE L0857]         assert!(selfplay.validate().is_ok());
[CODE L0858] 
[CODE L0859]         selfplay.gae_gamma = 1.0;
[CODE L0860]         assert_eq!(selfplay.validate(), Err("gae_gamma in (0,1)"));
[CODE L0861]     }
[CODE L0862] 
[CODE L0863]     #[test]
[CODE L0864]     fn score_summary_helpers_handle_empty_and_ranked_games() {
[CODE L0865]         let scores = [
[CODE L0866]             [30_000, 25_000, 20_000, 15_000],
[CODE L0867]             [15_000, 30_000, 25_000, 20_000],
[CODE L0868]         ];
[CODE L0869]         assert_eq!(games_played(&scores), 2);
[CODE L0870]         assert_eq!(total_score_sum(&scores), 180_000);
[CODE L0871]         assert_eq!(avg_score(&scores, 0), 22_500.0);
[CODE L0872]         assert!(score_std(&scores, 0) > 0.0);
[CODE L0873]         assert_eq!(top_two_rate(&scores, 1), 1.0);
[CODE L0874]         assert_eq!(fourth_place_rate(&scores, 0), 0.5);
[CODE L0875]         assert_eq!(win_rate_from_scores(&scores, 0), 0.5);
[CODE L0876]         assert_eq!(mean_placement_from_scores(&scores, 0), 2.5);
[CODE L0877] 
[CODE L0878]         assert_eq!(avg_score(&[], 0), 0.0);
[CODE L0879]         assert_eq!(score_std(&[], 0), 0.0);
[CODE L0880]         assert_eq!(top_two_rate(&[], 0), 0.0);
[CODE L0881]         assert_eq!(fourth_place_rate(&[], 0), 0.0);
[CODE L0882]         assert_eq!(win_rate_from_scores(&[], 0), 0.0);
[CODE L0883]         assert_eq!(mean_placement_from_scores(&[], 0), 2.5);
[CODE L0884]     }
[CODE L0885] 
[CODE L0886]     #[test]
[CODE L0887]     fn trajectory_and_arena_summary_helpers_compute_expected_values() {
[CODE L0888]         let mut t1 = Trajectory::new(7, 111);
[CODE L0889]         t1.final_scores = [30_000, 20_000, 25_000, 15_000];
[CODE L0890]         t1.steps.push(legal_step(0, 0, 1.5, false, 0));
[CODE L0891]         t1.steps.push(legal_step(1, 1, -0.5, true, 3));
[CODE L0892] 
[CODE L0893]         let mut t2 = Trajectory::new(8, 222);
[CODE L0894]         t2.final_scores = [15_000, 35_000, 25_000, 25_000];
[CODE L0895]         t2.steps.push(legal_step(2, 1, 2.0, true, 5));
[CODE L0896] 
[CODE L0897]         assert_eq!(t1.num_steps(), 2);
[CODE L0898]         assert_eq!(t1.active_players(), vec![0, 1]);
[CODE L0899]         assert_eq!(t1.score_for(2), 25_000);
[CODE L0900]         assert_eq!(t1.score_delta(0), 7_500);
[CODE L0901]         assert_eq!(t1.placement_for(0), 0);
[CODE L0902]         assert_eq!(t1.winner(), 0);
[CODE L0903]         assert_eq!(t1.max_turn(), 3);
[CODE L0904]         assert_eq!(t1.player_reward_sum(0), 1.5);
[CODE L0905]         assert_eq!(t1.total_reward(), 1.0);
[CODE L0906]         assert!(t1.is_complete());
[CODE L0907]         assert_eq!(t1.steps_for_player(1).len(), 1);
[CODE L0908] 
[CODE L0909]         let mut arena = Arena::new(ArenaConfig {
[CODE L0910]             max_trajectory_buffer: 4,
[CODE L0911]             ..Default::default()
[CODE L0912]         });
[CODE L0913]         arena.add_trajectory(t1);
[CODE L0914]         arena.add_trajectory(t2);
[CODE L0915] 
[CODE L0916]         assert_eq!(arena.max_capacity(), 4);
[CODE L0917]         assert!(!arena.is_full());
[CODE L0918]         assert_eq!(arena.completed_trajectories(), 2);
[CODE L0919]         assert_eq!(arena.total_steps(), 3);
[CODE L0920]         assert_eq!(arena.num_buffered(), 2);
[CODE L0921]         assert_eq!(arena.oldest_game_id(), Some(7));
[CODE L0922]         assert_eq!(arena.latest_game_id(), Some(8));
[CODE L0923]         assert_eq!(
[CODE L0924]             arena.mean_scores(),
[CODE L0925]             [22_500.0, 27_500.0, 25_000.0, 20_000.0]
[CODE L0926]         );
[CODE L0927]         assert_eq!(arena.mean_score_for(1), 27_500.0);
[CODE L0928]         assert!(arena.score_variance() > 0.0);
[CODE L0929]         assert_eq!(arena.mean_game_length(), 4.0);
[CODE L0930]         assert_eq!(arena.mean_placement_for(1), 2.0);
[CODE L0931]         assert_eq!(arena.fourth_place_count(0), 1);
[CODE L0932]         assert_eq!(arena.win_count(1), 1);
[CODE L0933]         assert_eq!(arena.win_rate_for(1), 0.5);
[CODE L0934]         assert_eq!(arena.fill_ratio(), 0.5);
[CODE L0935]         assert_eq!(arena.utilization(), "2/4 (50%)");
[CODE L0936]         assert_eq!(arena.avg_trajectory_length(), 1.5);
[CODE L0937]         assert!(
[CODE L0938]             arena
[CODE L0939]                 .stats_summary()
[CODE L0940]                 .contains("games=2 steps=3 buffered=2 complete=2")
[CODE L0941]         );
[CODE L0942]         assert_eq!(arena.collect_player_steps(1).len(), 2);
[CODE L0943]         assert_eq!(arena.compute_rewards(1), vec![vec![-0.5], vec![2.0]]);
[CODE L0944]         assert_eq!(arena.placement_distribution(1), [0.5, 0.0, 0.5, 0.0]);
[CODE L0945]         assert!(arena.validate_all().is_ok());
[CODE L0946] 
[CODE L0947]         arena.reset();
[CODE L0948]         assert_eq!(arena.games_completed, 0);
[CODE L0949]         assert!(arena.trajectory_buffer.is_empty());
[CODE L0950]     }
[CODE L0951] 
[CODE L0952]     #[test]
[CODE L0953]     fn masked_softmax_and_sampling_fallback_handle_degenerate_inputs() {
[CODE L0954]         let logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0955]         let legal_mask = [false; HYDRA_ACTION_SPACE];
[CODE L0956]         let probs = softmax_temperature(&logits, &legal_mask, 1.0);
[CODE L0957]         assert!(probs.iter().all(|&p| p == 0.0));
[CODE L0958] 
[CODE L0959]         let mut single_legal = [false; HYDRA_ACTION_SPACE];
[CODE L0960]         single_legal[9] = true;
[CODE L0961]         let (action, probs) = sample_action_with_temperature(&logits, &single_legal, 1.0, 1.5);
[CODE L0962]         assert_eq!(action, 9);
[CODE L0963]         assert_eq!(probs[9], 1.0);
[CODE L0964]     }
[CODE L0965] 
[CODE L0966]     #[test]
[CODE L0967]     fn trajectory_validate_rejects_bad_policy_and_label_shapes() {
[CODE L0968]         let mut traj = Trajectory::new(1, 2);
[CODE L0969]         let mut bad_step = legal_step(0, 0, 0.0, true, 0);
[CODE L0970]         bad_step.pi_old[0] = 0.7;
[CODE L0971]         bad_step.pi_old[1] = 0.7;
[CODE L0972]         traj.steps.push(bad_step);
[CODE L0973]         assert!(traj.validate().unwrap_err().contains("pi_old sums to"));
[CODE L0974] 
[CODE L0975]         let mut traj = Trajectory::new(1, 2);
[CODE L0976]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L0977]         let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0978]         exit_mask[0] = 1.0;
[CODE L0979]         exit_mask[(DISCARD_END as usize) + 1] = 1.0;
[CODE L0980]         let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0981]         exit_target[0] = 0.5;
[CODE L0982]         exit_target[(DISCARD_END as usize) + 1] = 0.5;
[CODE L0983]         step.exit_label = Some(TrajectoryExitLabel {
[CODE L0984]             target: exit_target,
[CODE L0985]             mask: exit_mask,
[CODE L0986]         });
[CODE L0987]         traj.steps.push(step);
[CODE L0988]         assert!(
[CODE L0989]             traj.validate()
[CODE L0990]                 .unwrap_err()
[CODE L0991]                 .contains("exit label masks illegal action")
[CODE L0992]         );
[CODE L0993] 
[CODE L0994]         let mut traj = Trajectory::new(1, 2);
[CODE L0995]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L0996]         let mut delta_mask = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0997]         delta_mask[0] = 1.0;
[CODE L0998]         let mut delta_target = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L0999]         delta_target[0] = f32::NAN;
[CODE L1000]         step.delta_q_label = Some(TrajectoryDeltaQLabel {
[CODE L1001]             target: delta_target,
[CODE L1002]             mask: delta_mask,
[CODE L1003]         });
[CODE L1004]         traj.steps.push(step);
[CODE L1005]         assert!(
[CODE L1006]             traj.validate()
[CODE L1007]                 .unwrap_err()
[CODE L1008]                 .contains("delta_q target at action 0 is not finite")
[CODE L1009]         );
[CODE L1010]     }
[CODE L1011] 
[CODE L1012]     #[test]
[CODE L1013]     fn config_validation_catches_temperature_and_lambda_bounds() {
[CODE L1014]         let arena_cfg = ArenaConfig {
[CODE L1015]             temperature_range: (0.0, 1.0),
[CODE L1016]             ..ArenaConfig::default()
[CODE L1017]         };
[CODE L1018]         assert_eq!(arena_cfg.validate(), Err("temperature range start > 0"));
[CODE L1019] 
[CODE L1020]         let arena_cfg = ArenaConfig {
[CODE L1021]             temperature_range: (1.2, 1.1),
[CODE L1022]             ..ArenaConfig::default()
[CODE L1023]         };
[CODE L1024]         assert_eq!(arena_cfg.validate(), Err("temperature range end >= start"));
[CODE L1025] 
[CODE L1026]         let selfplay = SelfPlayConfig {
[CODE L1027]             gae_lambda: 1.0,
[CODE L1028]             ..SelfPlayConfig::default()
[CODE L1029]         };
[CODE L1030]         assert_eq!(selfplay.validate(), Err("gae_lambda in (0,1)"));
[CODE L1031]     }
[CODE L1032] 
[CODE L1033]     #[test]
[CODE L1034]     fn arena_helpers_cover_empty_defaults_and_drain_behavior() {
[CODE L1035]         let mut arena = Arena::new(ArenaConfig {
[CODE L1036]             max_trajectory_buffer: 2,
[CODE L1037]             ..Default::default()
[CODE L1038]         });
[CODE L1039] 
[CODE L1040]         assert_eq!(arena.mean_scores(), [0.0; 4]);
[CODE L1041]         assert_eq!(arena.placement_distribution(0), [0.25; 4]);
[CODE L1042]         assert!(arena.compute_rewards(0).is_empty());
[CODE L1043]         assert_eq!(arena.mean_score_for(0), 0.0);
[CODE L1044]         assert_eq!(arena.score_variance(), 0.0);
[CODE L1045]         assert_eq!(arena.mean_game_length(), 0.0);
[CODE L1046]         assert_eq!(arena.mean_placement_for(0), 2.5);
[CODE L1047]         assert_eq!(arena.win_rate_for(0), 0.0);
[CODE L1048]         assert_eq!(arena.win_count(0), 0);
[CODE L1049]         assert_eq!(arena.oldest_game_id(), None);
[CODE L1050]         assert_eq!(arena.latest_game_id(), None);
[CODE L1051]         assert_eq!(arena.fill_ratio(), 0.0);
[CODE L1052]         assert_eq!(arena.avg_trajectory_length(), 0.0);
[CODE L1053]         assert!(arena.collect_player_steps(0).is_empty());
[CODE L1054] 
[CODE L1055]         let mut t = Trajectory::new(1, 9);
[CODE L1056]         t.steps.push(legal_step(0, 0, 0.5, true, 0));
[CODE L1057]         arena.add_trajectory(t);
[CODE L1058]         let drained = arena.drain_trajectories();
[CODE L1059]         assert_eq!(drained.len(), 1);
[CODE L1060]         assert!(arena.trajectory_buffer.is_empty());
[CODE L1061]     }
[CODE L1062] 
[CODE L1063]     #[test]
[CODE L1064]     fn trajectory_validate_rejects_no_legal_actions_illegal_choice_and_bad_exit_mass() {
[CODE L1065]         let mut traj = Trajectory::new(1, 2);
[CODE L1066]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L1067]         step.legal_mask = [false; HYDRA_ACTION_SPACE];
[CODE L1068]         traj.steps.push(step);
[CODE L1069]         assert!(
[CODE L1070]             traj.validate()
[CODE L1071]                 .unwrap_err()
[CODE L1072]                 .contains("legal_mask has no legal actions")
[CODE L1073]         );
[CODE L1074] 
[CODE L1075]         let mut traj = Trajectory::new(1, 2);
[CODE L1076]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L1077]         step.legal_mask[0] = false;
[CODE L1078]         step.legal_mask[1] = true;
[CODE L1079]         traj.steps.push(step);
[CODE L1080]         assert!(
[CODE L1081]             traj.validate()
[CODE L1082]                 .unwrap_err()
[CODE L1083]                 .contains("selected action 0 is not marked legal")
[CODE L1084]         );
[CODE L1085] 
[CODE L1086]         let mut traj = Trajectory::new(1, 2);
[CODE L1087]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L1088]         let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1089]         exit_mask[0] = 1.0;
[CODE L1090]         let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1091]         exit_target[0] = 0.7;
[CODE L1092]         step.exit_label = Some(TrajectoryExitLabel {
[CODE L1093]             target: exit_target,
[CODE L1094]             mask: exit_mask,
[CODE L1095]         });
[CODE L1096]         traj.steps.push(step);
[CODE L1097]         assert!(
[CODE L1098]             traj.validate()
[CODE L1099]                 .unwrap_err()
[CODE L1100]                 .contains("exit target mass over masked actions is 0.7")
[CODE L1101]         );
[CODE L1102]     }
[CODE L1103] 
[CODE L1104]     #[test]
[CODE L1105]     fn trajectory_validate_rejects_bad_delta_masks_and_invalid_action_index() {
[CODE L1106]         let mut traj = Trajectory::new(1, 2);
[CODE L1107]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L1108]         let mut delta_mask = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1109]         delta_mask[0] = 0.3;
[CODE L1110]         step.delta_q_label = Some(TrajectoryDeltaQLabel {
[CODE L1111]             target: [0.0; HYDRA_ACTION_SPACE],
[CODE L1112]             mask: delta_mask,
[CODE L1113]         });
[CODE L1114]         traj.steps.push(step);
[CODE L1115]         assert!(
[CODE L1116]             traj.validate()
[CODE L1117]                 .unwrap_err()
[CODE L1118]                 .contains("delta_q mask at action 0 is not approximately binary")
[CODE L1119]         );
[CODE L1120] 
[CODE L1121]         let mut traj = Trajectory::new(1, 2);
[CODE L1122]         let mut step = legal_step(0, 0, 0.0, true, 0);
[CODE L1123]         step.action = HYDRA_ACTION_SPACE as u8;
[CODE L1124]         traj.steps.push(step);
[CODE L1125]         assert!(traj.validate().unwrap_err().contains("invalid action"));
[CODE L1126]     }
[CODE L1127] 
[CODE L1128]     #[test]
[CODE L1129]     fn temperature_sampling_legal_only() {
[CODE L1130]         let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1131]         logits[0] = 10.0;
[CODE L1132]         logits[1] = -10.0;
[CODE L1133]         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1134]         mask[1] = true;
[CODE L1135]         mask[2] = true;
[CODE L1136]         for rng in [0.0, 0.5, 0.99] {
[CODE L1137]             let (action, _) = sample_action_with_temperature(&logits, &mask, 1.0, rng);
[CODE L1138]             assert!(mask[action as usize], "selected illegal action {action}");
[CODE L1139]         }
[CODE L1140]     }
[CODE L1141] 
[CODE L1142]     #[test]
[CODE L1143]     fn trajectory_non_empty() {
[CODE L1144]         let mut traj = Trajectory::new(0, 42);
[CODE L1145]         traj.steps.push(TrajectoryStep {
[CODE L1146]             obs: [0.0; OBS_SIZE],
[CODE L1147]             action: 0,
[CODE L1148]             pi_old: [0.0; HYDRA_ACTION_SPACE],
[CODE L1149]             legal_mask: {
[CODE L1150]                 let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1151]                 mask[0] = true;
[CODE L1152]                 mask
[CODE L1153]             },
[CODE L1154]             exit_label: None,
[CODE L1155]             delta_q_label: None,
[CODE L1156]             reward: 0.0,
[CODE L1157]             done: false,
[CODE L1158]             player_id: 0,
[CODE L1159]             game_id: 0,
[CODE L1160]             turn: 0,
[CODE L1161]             temperature: 1.0,
[CODE L1162]         });
[CODE L1163]         assert!(!traj.steps.is_empty());
[CODE L1164]     }
[CODE L1165] 
[CODE L1166]     #[test]
[CODE L1167]     fn trajectory_roundtrip() {
[CODE L1168]         let mut traj = Trajectory::new(42, 12345);
[CODE L1169]         traj.final_scores = [25000, 30000, 20000, 25000];
[CODE L1170]         traj.steps.push(TrajectoryStep {
[CODE L1171]             obs: [0.5; OBS_SIZE],
[CODE L1172]             action: 7,
[CODE L1173]             pi_old: {
[CODE L1174]                 let mut p = [0.0; HYDRA_ACTION_SPACE];
[CODE L1175]                 p[7] = 0.8;
[CODE L1176]                 p[45] = 0.2;
[CODE L1177]                 p
[CODE L1178]             },
[CODE L1179]             legal_mask: {
[CODE L1180]                 let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1181]                 mask[7] = true;
[CODE L1182]                 mask[45] = true;
[CODE L1183]                 mask
[CODE L1184]             },
[CODE L1185]             exit_label: None,
[CODE L1186]             delta_q_label: None,
[CODE L1187]             reward: 1.5,
[CODE L1188]             done: false,
[CODE L1189]             player_id: 2,
[CODE L1190]             game_id: 42,
[CODE L1191]             turn: 10,
[CODE L1192]             temperature: 0.8,
[CODE L1193]         });
[CODE L1194]         let step = &traj.steps[0];
[CODE L1195]         assert_eq!(step.action, 7);
[CODE L1196]         assert_eq!(step.player_id, 2);
[CODE L1197]         assert_eq!(step.turn, 10);
[CODE L1198]         assert!((step.reward - 1.5).abs() < 1e-5);
[CODE L1199]         assert!((step.temperature - 0.8).abs() < 1e-5);
[CODE L1200]         assert_eq!(traj.game_id, 42);
[CODE L1201]         assert_eq!(traj.seed, 12345);
[CODE L1202]         assert_eq!(traj.final_scores, [25000, 30000, 20000, 25000]);
[CODE L1203]         assert!((step.obs[0] - 0.5).abs() < 1e-5);
[CODE L1204]         assert!((step.pi_old[7] - 0.8).abs() < 1e-5);
[CODE L1205]     }
[CODE L1206] 
[CODE L1207]     #[test]
[CODE L1208]     fn arena_trajectory_management() {
[CODE L1209]         let config = ArenaConfig {
[CODE L1210]             max_trajectory_buffer: 3,
[CODE L1211]             ..Default::default()
[CODE L1212]         };
[CODE L1213]         let mut arena = Arena::new(config);
[CODE L1214]         assert_eq!(arena.total_steps(), 0);
[CODE L1215]         for i in 0..5u32 {
[CODE L1216]             let mut t = Trajectory::new(i, i as u64);
[CODE L1217]             t.steps.push(TrajectoryStep {
[CODE L1218]                 obs: [0.0; OBS_SIZE],
[CODE L1219]                 action: 0,
[CODE L1220]                 pi_old: [0.0; HYDRA_ACTION_SPACE],
[CODE L1221]                 legal_mask: {
[CODE L1222]                     let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1223]                     mask[0] = true;
[CODE L1224]                     mask
[CODE L1225]                 },
[CODE L1226]                 exit_label: None,
[CODE L1227]                 delta_q_label: None,
[CODE L1228]                 reward: 0.0,
[CODE L1229]                 done: true,
[CODE L1230]                 player_id: 0,
[CODE L1231]                 game_id: i,
[CODE L1232]                 turn: 0,
[CODE L1233]                 temperature: 1.0,
[CODE L1234]             });
[CODE L1235]             arena.add_trajectory(t);
[CODE L1236]         }
[CODE L1237]         assert_eq!(arena.games_completed, 5);
[CODE L1238]         assert_eq!(arena.trajectory_buffer.len(), 3);
[CODE L1239]         let drained = arena.drain_trajectories();
[CODE L1240]         assert_eq!(drained.len(), 3);
[CODE L1241]         assert!(arena.trajectory_buffer.is_empty());
[CODE L1242]     }
[CODE L1243] 
[CODE L1244]     #[test]
[CODE L1245]     fn temperature_affects_distribution() {
[CODE L1246]         let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1247]         logits[0] = 3.0;
[CODE L1248]         logits[1] = 1.0;
[CODE L1249]         logits[2] = 0.0;
[CODE L1250]         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1251]         mask[0] = true;
[CODE L1252]         mask[1] = true;
[CODE L1253]         mask[2] = true;
[CODE L1254]         let (_, probs_low) = sample_action_with_temperature(&logits, &mask, 0.1, 0.0);
[CODE L1255]         let (_, probs_high) = sample_action_with_temperature(&logits, &mask, 10.0, 0.0);
[CODE L1256]         assert!(
[CODE L1257]             probs_low[0] > probs_high[0],
[CODE L1258]             "low temp should concentrate: {:.3} vs {:.3}",
[CODE L1259]             probs_low[0],
[CODE L1260]             probs_high[0]
[CODE L1261]         );
[CODE L1262]     }
[CODE L1263] 
[CODE L1264]     #[test]
[CODE L1265]     fn single_legal_action_always_selected() {
[CODE L1266]         let logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1267]         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1268]         mask[33] = true;
[CODE L1269]         for rng in [0.0, 0.5, 0.99] {
[CODE L1270]             let (action, probs) = sample_action_with_temperature(&logits, &mask, 1.0, rng);
[CODE L1271]             assert_eq!(action, 33);
[CODE L1272]             assert!((probs[33] - 1.0).abs() < 1e-5);
[CODE L1273]         }
[CODE L1274]     }
[CODE L1275] 
[CODE L1276]     #[test]
[CODE L1277]     fn compute_placements_correct() {
[CODE L1278]         let p = compute_placements([40000, 30000, 20000, 10000]);
[CODE L1279]         assert_eq!(p, [0, 1, 2, 3]);
[CODE L1280]         let p2 = compute_placements([10000, 40000, 20000, 30000]);
[CODE L1281]         assert_eq!(p2, [3, 0, 2, 1]);
[CODE L1282]     }
[CODE L1283] 
[CODE L1284]     #[test]
[CODE L1285]     fn greedy_picks_highest_legal() {
[CODE L1286]         let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1287]         logits[10] = 100.0;
[CODE L1288]         logits[20] = 50.0;
[CODE L1289]         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1290]         mask[20] = true;
[CODE L1291]         mask[30] = true;
[CODE L1292]         let action = greedy_action(&logits, &mask);
[CODE L1293]         assert_eq!(action, 20, "should pick highest LEGAL action");
[CODE L1294]     }
[CODE L1295] 
[CODE L1296]     #[test]
[CODE L1297]     fn trajectory_validate_catches_bad_player() {
[CODE L1298]         let mut traj = Trajectory::new(0, 0);
[CODE L1299]         traj.steps.push(TrajectoryStep {
[CODE L1300]             obs: [0.0; OBS_SIZE],
[CODE L1301]             action: 0,
[CODE L1302]             pi_old: {
[CODE L1303]                 let mut p = [0.0; HYDRA_ACTION_SPACE];
[CODE L1304]                 p[0] = 1.0;
[CODE L1305]                 p
[CODE L1306]             },
[CODE L1307]             legal_mask: {
[CODE L1308]                 let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1309]                 mask[0] = true;
[CODE L1310]                 mask
[CODE L1311]             },
[CODE L1312]             exit_label: None,
[CODE L1313]             delta_q_label: None,
[CODE L1314]             reward: 0.0,
[CODE L1315]             done: true,
[CODE L1316]             player_id: 5,
[CODE L1317]             game_id: 0,
[CODE L1318]             turn: 0,
[CODE L1319]             temperature: 1.0,
[CODE L1320]         });
[CODE L1321]         assert!(traj.validate().is_err());
[CODE L1322]     }
[CODE L1323] 
[CODE L1324]     #[test]
[CODE L1325]     fn trajectory_validate_passes_good_data() {
[CODE L1326]         let mut traj = Trajectory::new(0, 0);
[CODE L1327]         traj.steps.push(TrajectoryStep {
[CODE L1328]             obs: [0.0; OBS_SIZE],
[CODE L1329]             action: 3,
[CODE L1330]             pi_old: {
[CODE L1331]                 let mut p = [0.0; HYDRA_ACTION_SPACE];
[CODE L1332]                 p[3] = 1.0;
[CODE L1333]                 p
[CODE L1334]             },
[CODE L1335]             legal_mask: {
[CODE L1336]                 let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1337]                 mask[3] = true;
[CODE L1338]                 mask
[CODE L1339]             },
[CODE L1340]             exit_label: None,
[CODE L1341]             delta_q_label: None,
[CODE L1342]             reward: 0.0,
[CODE L1343]             done: true,
[CODE L1344]             player_id: 0,
[CODE L1345]             game_id: 0,
[CODE L1346]             turn: 0,
[CODE L1347]             temperature: 1.0,
[CODE L1348]         });
[CODE L1349]         assert!(traj.validate().is_ok());
[CODE L1350]     }
[CODE L1351] 
[CODE L1352]     #[test]
[CODE L1353]     fn arena_500_games_completes() {
[CODE L1354]         let config = ArenaConfig {
[CODE L1355]             num_parallel_games: 500,
[CODE L1356]             max_trajectory_buffer: 600,
[CODE L1357]             ..Default::default()
[CODE L1358]         };
[CODE L1359]         let mut arena = Arena::new(config);
[CODE L1360]         for g in 0..500u32 {
[CODE L1361]             let mut traj = Trajectory::new(g, g as u64);
[CODE L1362]             for turn in 0..10u16 {
[CODE L1363]                 traj.steps.push(TrajectoryStep {
[CODE L1364]                     obs: [0.0; OBS_SIZE],
[CODE L1365]                     action: (turn % 34) as u8,
[CODE L1366]                     pi_old: {
[CODE L1367]                         let mut p = [0.0; HYDRA_ACTION_SPACE];
[CODE L1368]                         p[(turn % 34) as usize] = 1.0;
[CODE L1369]                         p
[CODE L1370]                     },
[CODE L1371]                     legal_mask: {
[CODE L1372]                         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1373]                         mask[(turn % 34) as usize] = true;
[CODE L1374]                         mask
[CODE L1375]                     },
[CODE L1376]                     exit_label: None,
[CODE L1377]                     delta_q_label: None,
[CODE L1378]                     reward: 0.0,
[CODE L1379]                     done: turn == 9,
[CODE L1380]                     player_id: (turn % 4) as u8,
[CODE L1381]                     game_id: g,
[CODE L1382]                     turn,
[CODE L1383]                     temperature: 1.0,
[CODE L1384]                 });
[CODE L1385]             }
[CODE L1386]             traj.final_scores = [25000; 4];
[CODE L1387]             arena.add_trajectory(traj);
[CODE L1388]         }
[CODE L1389]         assert_eq!(arena.games_completed, 500);
[CODE L1390]         assert!(arena.total_steps() >= 5000);
[CODE L1391]         assert!(arena.validate_all().is_ok());
[CODE L1392]     }
[CODE L1393] 
[CODE L1394]     #[test]
[CODE L1395]     fn softmax_temperature_sums_to_one() {
[CODE L1396]         let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
[CODE L1397]         logits[0] = 3.0;
[CODE L1398]         logits[5] = 1.0;
[CODE L1399]         let mut mask = [false; HYDRA_ACTION_SPACE];
[CODE L1400]         mask[0] = true;
[CODE L1401]         mask[5] = true;
[CODE L1402]         mask[10] = true;
[CODE L1403]         let probs = softmax_temperature(&logits, &mask, 1.0);
[CODE L1404]         let sum: f32 = probs.iter().sum();
[CODE L1405]         assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
[CODE L1406]     }
[CODE L1407] 
[CODE L1408]     #[test]
[CODE L1409]     fn test_trajectory_empty_fails_validation() {
[CODE L1410]         let traj = Trajectory::new(0, 42);
[CODE L1411]         assert!(traj.steps.is_empty());
[CODE L1412]         let result = traj.validate();
[CODE L1413]         assert!(result.is_err(), "empty trajectory should fail validation");
[CODE L1414]     }
[CODE L1415] }
```

</artifacts>
