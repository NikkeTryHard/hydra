# Hydra prompt — hidden-world pass two generated packet

<role>
Example role placeholder.
Replace this with the role that fits your actual prompt.
Keep it short and task-specific.
You are solving hidden-world lane pass two: convert a narrowed diagnosis into the strongest actual strength-producing design stack.
You may overturn current Hydra support, current staging, and current interfaces if the stronger design requires it.
</role>

<task>
Example task placeholder.
Replace this with the actual job the agent should do.

An example task block might ask for:
- what the artifacts directly support
- what is only inference
- what confidence level each major conclusion deserves
- what simpler or stronger local alternatives exist inside the same lane
- what should be kept, narrowed, or removed
- why the confident parts of the answer are actually justified
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
Work toward the strongest exact blueprint for the hidden-world lane after the target-object narrowing pass has already happened.
We want a detailed answer that makes clear:
- the best full design, even if it exceeds current Hydra support
- the shortest honest tranche that preserves the winning design's semantics
- the ranked algorithm-family decision table
- the teacher hierarchy and information fences
- the concrete training recipe
- the measurable evaluation gates and kill criteria
- exactly where Hydra is wrong if current assumptions block the stronger design
Use the artifacts below to derive your conclusions.
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
Do not rerun pass-one diagnosis or spend the answer rediscovering the hidden-world object.
Current Hydra support is build-cost evidence, not a veto on stronger designs.
Keep deployable student objects public-information legal unless you explicitly fence them as teacher-only or diagnostics-only.
If a stronger design requires contract migration, say so explicitly instead of weakening the design to fit the current carrier.
Do not end in a survey. End in a ranked verdict with measurable gates and falsification criteria.
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
Bias toward exact design decisions: name objects, losses, transition order, validation surfaces, and kill conditions.
Use tables when they clarify family comparisons or promotion rules.
Make it possible for another strong agent to implement or falsify the recommendation later without guessing.
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>

## Artifact 01 — Pass-two objective
Artifact id: `pass-two-objective`
Type: `literal`
Why it matters: Compact reminder that this packet is for design closure, not rediscovery.

```text
Pass two is not a rediscovery pass.
Assume pass one already narrowed the hidden-world lane.
Now choose the strongest design stack, define the shortest honest tranche, and state where Hydra must be willing to be wrong to get stronger.
```

## Artifact 02 — Repo routing and project goal
Artifact id: `repo-readme`
Source label: README
Type: `file_full`
Source: `README.md`
Why it matters: Top-level routing artifact showing Hydra’s goal and repo routing logic.

```md
[README L0001] # Hydra
[README L0002]
[README L0003] Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent, 10.68 stable dan on Tenhou) with open weights.
[README L0004]
[README L0005] ## Goal
[README L0006]
[README L0007] Train a mahjong AI that:
[README L0008] - Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
[README L0009] - Releases weights under a permissive license
[README L0010] - Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs
[README L0011]
[README L0012] ## Architecture
[README L0013]
[README L0014] Hydra uses a layered authority flow built from the archive handoff canon upward:
[README L0015]
[README L0016] 1. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) — epistemic root / canonical archive SSOT for upstream research conclusions
[README L0017] 2. [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) and [`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) — derived archive views over that canonical source ledger
[README L0018] 3. [`research/design/HYDRA_FINAL.md`](research/design/HYDRA_FINAL.md) — promoted architecture doctrine built from archive canon plus repo validation
[README L0019] 4. [`research/design/HYDRA_RECONCILIATION.md`](research/design/HYDRA_RECONCILIATION.md) — promoted operational doctrine and roadmap to Hydra v1 built from archive canon plus repo validation
[README L0020] 5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — promoted current-status snapshot for already-built repo surfaces
[README L0021] 6. [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md) — runtime semantics and compatibility surfaces; current code wins when docs drift
[README L0022]
[README L0023] Hydra's documentation split is simple:
[README L0024]
[README L0025] - `HYDRA_FINAL.md` describes the max-ceiling destination
[README L0026] - `HYDRA_RECONCILIATION.md` is the roadmap to Hydra v1
[README L0027] - `docs/CURRENT_STATUS.md` says what is already shipped or still staged today
[README L0028]
[README L0029] Raw `answer_*_combined.md` files in `research/agent_handoffs/combined_all_variants/` remain raw archive corpus, not promoted doctrine.
[README L0030]
[README L0031] ## Fresh-agent routing
[README L0032]
[README L0033] If you are entering Hydra with zero prior memory, use this order and stop when you have enough truth for the task:
[README L0034]
[README L0035] 1. `README.md` for repo routing
[README L0036] 2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` for canonical archive intake
[README L0037] 3. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` for derived archive triage
[README L0038] 4. `research/design/HYDRA_RECONCILIATION.md` for the roadmap to Hydra v1
[README L0039] 5. `research/design/HYDRA_FINAL.md` for the long-term ceiling
[README L0040] 6. `docs/CURRENT_STATUS.md` for shipped/staged truth
[README L0041] 7. `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` for runtime truth
[README L0042]
[README L0043] `combined_all_variants/` remains raw archive corpus for provenance only.
[README L0044]
[README L0045] ## Status vocabulary
[README L0046]
[README L0047] For implementation work, choose the next lane from
[README L0048] `research/design/HYDRA_RECONCILIATION.md`, confirm whether it already exists in
[README L0049] `docs/CURRENT_STATUS.md`, and confirm exact runtime contracts in
[README L0050] `docs/GAME_ENGINE.md` plus current code.
[README L0051]
[README L0052] | Term | Meaning |
[README L0053] |---|---|
[README L0054] | `active path` | what Hydra should optimize/build now |
[README L0055] | `shipped baseline` | implemented and part of the current live baseline |
[README L0056] | `implemented but not default-on` | implemented in code, intentionally not the default path |
[README L0057] | `implemented but staged` | implemented enough to exist, but activation/promotion is intentionally deferred |
[README L0058] | `reserve shelf` | preserved later-work direction, not current mainline |
[README L0059] | `blocked` | not ready because a real dependency or semantic gap remains |
[README L0060] | `rejected` | not part of the current plan |
[README L0061] | `historical` | preserved context only; not governing truth |
[README L0062]
[README L0063] ## Crate ownership
[README L0064]
[README L0065] | Crate | Owns | Does not own |
[README L0066] |---|---|---|
[README L0067] | `crates/hydra-engine` | vendored rules engine behavior | Hydra-specific runtime/training orchestration |
[README L0068] | `crates/hydra-core` | runtime bridge, encoder, simulator, seeding, search/runtime feature plumbing | Burn training logic or vendored rules ownership |
[README L0069] | `crates/hydra-train` | model, targets, losses, BC/RL/self-play orchestration, train binary | low-level rules engine behavior |
[README L0070]
[README L0071] If you are deciding what to build next, follow the Fresh-agent routing order above.
[README L0072] `research/design/HYDRA_SPEC.md` remains historical context only.
[README L0073]
[README L0074] ## Research
[README L0075]
[README L0076] | File | What's In It |
[README L0077] |------|-------------|
[README L0078] | [ARCHIVE_CANONICAL_CLAIMS.jsonl](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl) | Epistemic root / canonical archive SSOT for upstream research intake |
[README L0079] | [ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md) | Derived archive prioritization view over canonical archive claims |
[README L0080] | [ARCHIVE_CANONICAL_CLAIMS_RENDERED.md](research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md) | Generated human-readable mirror of the canonical archive ledger |
[README L0081] | [HYDRA_FINAL.md](research/design/HYDRA_FINAL.md) | Promoted architecture doctrine summary |
[README L0082] | [HYDRA_RECONCILIATION.md](research/design/HYDRA_RECONCILIATION.md) | Promoted operational doctrine summary and roadmap to Hydra v1 |
[README L0083] | [HYDRA_ARCHIVE.md](research/design/HYDRA_ARCHIVE.md) | Reserve-only design/archive planning surfaces |
[README L0084] | [HYDRA_SPEC.md](research/design/HYDRA_SPEC.md) | Historical architecture spec only |
[README L0085] | [MORTAL_ANALYSIS.md](research/intel/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
[README L0086] | [OPPONENT_MODELING.md](research/design/OPPONENT_MODELING.md) | Opponent-modeling rationale; includes both active ideas and reserve/future extensions |
[README L0087] | [INFRASTRUCTURE.md](research/infrastructure/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
[README L0088] | [SEEDING.md](research/design/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
[README L0089] | [CHECKPOINTING.md](research/infrastructure/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
[README L0090] | [ECOSYSTEM.md](research/intel/ECOSYSTEM.md) | Useful repos, tooling, and framework references |
[README L0091] | [REWARD_DESIGN.md](research/design/REWARD_DESIGN.md) | Reward design and RVR notes |
[README L0092] | [COMMUNITY_INSIGHTS.md](research/intel/COMMUNITY_INSIGHTS.md) | Community observations and external signals |
[README L0093] | [REFERENCES.md](research/intel/REFERENCES.md) | Citation index |
[README L0094] | [TESTING.md](research/design/TESTING.md) | Testing strategy, correctness verification, property-based tests |
[README L0095] | [RUST_STACK.md](research/infrastructure/RUST_STACK.md) | 100% Rust decision and framework notes |
[README L0096]
[README L0097] ## Status
[README L0098]
[README L0099] Hydra is in active implementation. For the current shipped/staged repo snapshot, read [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). For runtime semantics and compatibility-sensitive invariants, read [`docs/GAME_ENGINE.md`](docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](docs/COMPATIBILITY_SURFACE.md).
[README L0100]
[README L0101] ## Testing and Coverage
[README L0102]
[README L0103] Hydra uses `cargo nextest run --release` as the default workspace test path and `cargo-llvm-cov` for workspace-wide coverage reporting. For local coverage generation details, read [`docs/COVERAGE.md`](docs/COVERAGE.md).
[README L0104]
[README L0105] ## License
[README L0106]
[README L0107] - **hydra-core** (encoder, training pipeline): [BSL 1.1](crates/hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
[README L0108] - **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
```

## Artifact 03 — Current staged-vs-later hidden-world and DeltaQ sequencing
Artifact id: `hydra-reconciliation-slices`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:133-160`
Why it matters: Active-path slice showing belief baseline, DeltaQ lane, staged mixture/opponent closure, and later CT-SMC / AFBS promotions.

```md
[RECON L0133] - the stronger public-teacher belief-semantics tranche as part of the current
[RECON L0134]   training baseline
[RECON L0135] - the current Hand-EV realism upgrade as part of the live baseline surface
[RECON L0136] - replay-derived `safety_residual` as a narrow supervised lane
[RECON L0137] - an end-to-end ExIt carrier across the live self-play lane and the
[RECON L0138]   replay/sample sidecar-first lane
[RECON L0139]
[RECON L0140] ### 3.2 Implemented but not default-on
[RECON L0141]
[RECON L0142] The current challenger lane is:
[RECON L0143]
[RECON L0144] - the narrow DeltaQ supervision lane, which is implemented in code and remains
[RECON L0145]   promotion-gated through an arena-confirmation path
[RECON L0146]
[RECON L0147] ### 3.3 Implemented but staged
[RECON L0148]
[RECON L0149] The current staged lanes are:
[RECON L0150]
[RECON L0151] - `mixture_weight` promotion
[RECON L0152] - richer opponent-target closure
[RECON L0153] - representative-world / per-particle CT-SMC Hand-EV
[RECON L0154] - selective AFBS / endgame deepening
[RECON L0155]
[RECON L0156] ### 3.4 Reserve shelf
[RECON L0157]
[RECON L0158] The current reserve shelf includes:
[RECON L0159]
[RECON L0160] - broader public-belief search as project identity
```

## Artifact 04 — Current shipped vs staged status for belief, DeltaQ, and CT-SMC lanes
Artifact id: `current-status-slices`
Source label: STATUS
Type: `file_range`
Source: `docs/CURRENT_STATUS.md:34-75`
Why it matters: Promoted current-status slice showing what is shipped, not default-on, staged, and reserve-shelf.

```md
[STATUS L0034] - The stronger public-teacher belief-semantics tranche is shipped as part of the current training baseline.
[STATUS L0035] - The current Hand-EV realism upgrade is shipped as part of the live baseline surface.
[STATUS L0036] - Replay-derived `safety_residual` is shipped as a narrow supervised lane.
[STATUS L0037] - ExIt now has an end-to-end carrier across the live self-play lane and the replay/sample sidecar-first lane.
[STATUS L0038]
[STATUS L0039] ### Implemented but not default-on
[STATUS L0040]
[STATUS L0041] - The narrow DeltaQ supervision lane is implemented in code and promotion-gated through an arena-confirmation path.
[STATUS L0042] - DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but the lane is still **not** default-on.
[STATUS L0043]
[STATUS L0044] ### Implemented but staged
[STATUS L0045]
[STATUS L0046] - `mixture_weight` promotion remains staged.
[STATUS L0047] - Richer opponent-target closure remains staged.
[STATUS L0048] - Representative-world / per-particle CT-SMC Hand-EV remains staged.
[STATUS L0049] - Selective AFBS / endgame deepening remains staged.
[STATUS L0050]
[STATUS L0051] ### Reserve shelf
[STATUS L0052]
[STATUS L0053] - Broader public-belief search as project identity remains reserve-shelf, not active-path.
[STATUS L0054] - Deeper robust-opponent search backups remain reserve-shelf.
[STATUS L0055] - Larger latent-opponent / richer auxiliary-head expansion remains reserve-shelf until existing target closure improves.
[STATUS L0056]
[STATUS L0057] ## Area-by-area summary
[STATUS L0058]
[STATUS L0059] | Area | Current status | Notes |
[STATUS L0060] |---|---|---|
[STATUS L0061] | Runtime encoder / action semantics | shipped baseline | See `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` |
[STATUS L0062] | Hand-EV baseline surface | shipped baseline | Stronger local evaluator is live; representative-world CT-SMC Hand-EV remains staged |
[STATUS L0063] | Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche is in the live baseline |
[STATUS L0064] | BC runtime authority | shipped baseline | Fresh runs are config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
[STATUS L0065] | `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
[STATUS L0066] | ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
[STATUS L0067] | DeltaQ lane | implemented but not default-on | Arena-confirmation path implemented; promotion artifact now records pre-arena recommendation plus final `arena_decision`/`arena_report` |
[STATUS L0068] | `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
[STATUS L0069] | `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
[STATUS L0070] | AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |
[STATUS L0071]
[STATUS L0072] ## Where to read next
[STATUS L0073]
[STATUS L0074] - Need the current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
[STATUS L0075] - Need the roadmap to Hydra v1 or the active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
```

## Artifact 05 — CT-SMC implementation roadmap slice
Artifact id: `impl-roadmap-ctsmc-build-plan`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:622-676`
Why it matters: Concrete CT-SMC roadmap slice for buildability and migration planning.

```md
[IMPL L0622] ## Step 9: CT-SMC Exact DP Sampler (Pure Rust, No Burn)
[IMPL L0623]
[IMPL L0624] ### File: `hydra-core/src/ct_smc.rs`
[IMPL L0625]
[IMPL L0626] ### 9.1 State Space
[IMPL L0627]
[IMPL L0628] Hidden allocation matrix X: shape 34x4 (tile types x locations: opp1, opp2, opp3, wall).
[IMPL L0629] Row sums r(k) = 4 - visible(k), column sums s(z) = concealed hand sizes + wall size.
[IMPL L0630]
[IMPL L0631] **Key insight**: c_W = R_k - (c1 + c2 + c3) is derived, so DP state is 3D: (c1, c2, c3).
[IMPL L0632] - Max concealed hand = 14 tiles. State space: each c_i in [0, 14], <= 15^3 = 3,375 states per DP layer.
[IMPL L0633] - 34 tile-type layers, <= 35 compositions per layer. Total: ~4.0M ops.
[IMPL L0634]
[IMPL L0635] ### 9.2 Data Structures
[IMPL L0636]
[IMPL L0637] **CtSmcConfig**: `num_particles: usize` (128-4096), `ess_threshold: f32` (0.4), `rng_seed: u64`
[IMPL L0638]
[IMPL L0639] **Particle**: `allocation: [[u8; 4]; 34]` (X[k][z], z=0,1,2 are opponents, z=3 is wall), `log_weight: f64`
[IMPL L0640]
[IMPL L0641] **Precomputed compositions**: `COMPOSITIONS: LazyLock<[Vec<[u8; 4]>; 5]>` -- for r in 0..=4, all (x0,x1,x2,x3) with sum=r. Sizes: r=0->1, r=1->4, r=2->10, r=3->20, r=4->35. Total: 70.
[IMPL L0642]
[IMPL L0643] ### 9.3 Exact DP Recurrence (Log-Space)
[IMPL L0644]
[IMPL L0645] - Signature: `forward_dp(row_sums: &[u8; 34], col_sums: &[usize; 4], log_omega: &[[f64; 4]; 34]) -> Vec<FxHashMap<(u8,u8,u8), f64>>`
[IMPL L0646] - Base: dp[34][(0,0,0)] = 0.0 (log 1), all others -inf.
[IMPL L0647] - Recurrence: dp[k][(c1,c2,c3)] = logsumexp over compositions x of r(k) where `log_phi_k(x) = sum_j x[j] * log(omega[k][j])`, checking capacity constraints.
[IMPL L0648] - Helper: `logsumexp2(a, b)` -- numerically stable log(exp(a) + exp(b)).
[IMPL L0649]
[IMPL L0650] ### 9.4 Exact Backward Sampling
[IMPL L0651]
[IMPL L0652] - Signature: `backward_sample(dp, row_sums, col_sums, log_omega, rng) -> [[u8; 4]; 34]`
[IMPL L0653] - For k=0..34: sample composition x_k ~ p(x_k | remaining capacity) where `p(x_k = x | c) = exp(log_phi_k(x) + dp[k+1][c-x] - dp[k][c])`.
[IMPL L0654] - Exact: no rejection, no MCMC. One pass through 34 tile types.
[IMPL L0655]
[IMPL L0656] ### 9.5 SMC Reweighting + ESS + Resampling
[IMPL L0657]
[IMPL L0658] **CtSmc**: `config: CtSmcConfig`, `particles: Vec<Particle>`, `dp_cache: Option<...>`
[IMPL L0659] - `update(&mut self, row_sums, col_sums, log_omega, likelihood_fn, rng)`: Forward DP -> sample P particles -> weight by likelihood -> normalize -> ESS check -> systematic resample if ESS < threshold * P.
[IMPL L0660] - `ess(&self) -> f32`: Standard ESS = 1 / sum(w_i^2).
[IMPL L0661] - `systematic_resample(&mut self, rng)`: O(N), low variance.
[IMPL L0662]
[IMPL L0663] ### 9.6 Public API
[IMPL L0664]
[IMPL L0665] In `hydra-core/src/lib.rs`: `pub mod ct_smc;`
[IMPL L0666] Exports: `CtSmc`, `CtSmcConfig`, `Particle`
[IMPL L0667]
[IMPL L0668] ### 9.7 Tests + Benchmark
[IMPL L0669]
[IMPL L0670] | Test | Assertion |
[IMPL L0671] |------|-----------|
[IMPL L0672] | `uniform_omega_matches_hypergeometric` | For r=[1,1,0,...] with uniform omega, marginals match analytic hypergeometric |
[IMPL L0673] | `particles_satisfy_constraints` | Every particle satisfies row and column sum constraints |
[IMPL L0674] | `uniform_likelihood_high_ess` | Uniform likelihood -> ESS close to P |
[IMPL L0675]
[IMPL L0676] Benchmark: `bench_ct_smc_full_pipeline` -- median < 1ms over 1000 runs (forward DP + 128 backward samples).
```

## Artifact 06 — Current Stage-A belief teacher implementation
Artifact id: `belief-teacher-code`
Source label: BELIEF
Type: `file_range`
Source: `crates/hydra-train/src/teacher/belief.rs:90-166`
Why it matters: Grounds what the current teacher path actually emits and gates.

```rust
[BELIEF L0090]
[BELIEF L0091] pub fn project_hidden_counts_to_col_sums(
[BELIEF L0092]     hidden_counts: &[usize; BELIEF_ZONES],
[BELIEF L0093] ) -> [f64; BELIEF_ZONES] {
[BELIEF L0094]     let mut col_sums = [0.0f64; BELIEF_ZONES];
[BELIEF L0095]     for (dst, &value) in col_sums.iter_mut().zip(hidden_counts.iter()) {
[BELIEF L0096]         *dst = value as f64;
[BELIEF L0097]     }
[BELIEF L0098]     col_sums
[BELIEF L0099] }
[BELIEF L0100]
[BELIEF L0101] pub fn build_stage_a_teacher(
[BELIEF L0102]     remaining: &[f32; BELIEF_TILES],
[BELIEF L0103]     hidden_counts: &[usize; BELIEF_ZONES],
[BELIEF L0104]     config: StageABeliefConfig,
[BELIEF L0105] ) -> Option<StageABeliefTarget> {
[BELIEF L0106]     if config.num_components as usize != BELIEF_COMPONENTS {
[BELIEF L0107]         return None;
[BELIEF L0108]     }
[BELIEF L0109]     let hidden_tiles: usize = hidden_counts.iter().sum();
[BELIEF L0110]     if hidden_tiles == 0 {
[BELIEF L0111]         return None;
[BELIEF L0112]     }
[BELIEF L0113]
[BELIEF L0114]     let row_sums = project_public_remaining_to_row_sums(remaining);
[BELIEF L0115]     let total_row: f64 = row_sums.iter().sum();
[BELIEF L0116]     if total_row <= 0.0 {
[BELIEF L0117]         return None;
[BELIEF L0118]     }
[BELIEF L0119]
[BELIEF L0120]     let col_sums = project_hidden_counts_to_col_sums(hidden_counts);
[BELIEF L0121]     let total_col: f64 = col_sums.iter().sum();
[BELIEF L0122]     if (total_row - total_col).abs() > 0.5 {
[BELIEF L0123]         return None;
[BELIEF L0124]     }
[BELIEF L0125]     let kernel = build_uniform_kernel();
[BELIEF L0126]     let mixture = MixtureSib::new(config.num_components, &kernel, &row_sums, &col_sums);
[BELIEF L0127]     let weights = mixture.weights();
[BELIEF L0128]     let entropy = mixture.weight_entropy() as f32;
[BELIEF L0129]     let ess = mixture.ess() as f32;
[BELIEF L0130]     let trust = ((ess / config.num_components as f32).clamp(0.0, 1.0) * 0.7
[BELIEF L0131]         + (1.0 - (entropy / 1.3863).clamp(0.0, 1.0)) * 0.3)
[BELIEF L0132]         .clamp(0.0, 1.0);
[BELIEF L0133]
[BELIEF L0134]     if trust < config.trust_threshold {
[BELIEF L0135]         return None;
[BELIEF L0136]     }
[BELIEF L0137]
[BELIEF L0138]     let mut belief_fields = [0.0f32; BELIEF_FIELDS_SIZE];
[BELIEF L0139]     for component in 0..BELIEF_COMPONENTS {
[BELIEF L0140]         for zone in 0..BELIEF_ZONES {
[BELIEF L0141]             let channel = component * BELIEF_ZONES + zone;
[BELIEF L0142]             for tile in 0..BELIEF_TILES {
[BELIEF L0143]                 belief_fields[channel * BELIEF_TILES + tile] =
[BELIEF L0144]                     mixture.components[component].belief[tile * BELIEF_ZONES + zone] as f32;
[BELIEF L0145]             }
[BELIEF L0146]         }
[BELIEF L0147]     }
[BELIEF L0148]
[BELIEF L0149]     let mixture_weights =
[BELIEF L0150]         if config.mixture_entropy_threshold >= 0.0 && entropy <= config.mixture_entropy_threshold {
[BELIEF L0151]             let mut out = [0.0f32; BELIEF_COMPONENTS];
[BELIEF L0152]             for (dst, src) in out.iter_mut().zip(weights.iter().copied()) {
[BELIEF L0153]                 *dst = src as f32;
[BELIEF L0154]             }
[BELIEF L0155]             Some(out)
[BELIEF L0156]         } else {
[BELIEF L0157]             None
[BELIEF L0158]         };
[BELIEF L0159]
[BELIEF L0160]     Some(StageABeliefTarget {
[BELIEF L0161]         belief_fields,
[BELIEF L0162]         mixture_weights,
[BELIEF L0163]         trust,
[BELIEF L0164]         ess,
[BELIEF L0165]         entropy,
[BELIEF L0166]     })
```

## Artifact 07 — Mixture-SIB construction and posterior update machinery
Artifact id: `sinkhorn-mixture-core`
Source label: SINK
Type: `file_range`
Source: `crates/hydra-core/src/sinkhorn.rs:196-360`
Why it matters: Shows the current hidden-state carrier mechanics and where posterior updates actually live.

```rust
[SINK L0196]     pub fn new(
[SINK L0197]         num_components: u8,
[SINK L0198]         kernel: &[f64; 136],
[SINK L0199]         row_sums: &[f64; 34],
[SINK L0200]         col_sums: &[f64; 4],
[SINK L0201]     ) -> Self {
[SINK L0202]         let belief = sinkhorn_project(kernel, row_sums, col_sums, 50, 1e-8);
[SINK L0203]         let components = (0..num_components)
[SINK L0204]             .map(|_| SibComponent {
[SINK L0205]                 belief,
[SINK L0206]                 log_weight: -(num_components as f64).ln(),
[SINK L0207]             })
[SINK L0208]             .collect();
[SINK L0209]         Self { components }
[SINK L0210]     }
[SINK L0211]
[SINK L0212]     pub fn bayesian_update(&mut self, event_log_likelihoods: &[f64]) {
[SINK L0213]         assert_eq!(event_log_likelihoods.len(), self.components.len());
[SINK L0214]         for (comp, &ll) in self.components.iter_mut().zip(event_log_likelihoods) {
[SINK L0215]             comp.log_weight += ll;
[SINK L0216]         }
[SINK L0217]         self.renormalize_log_weights();
[SINK L0218]     }
[SINK L0219]
[SINK L0220]     pub fn apply_entropy_regularizer(&mut self, mix: f64) {
[SINK L0221]         if self.components.is_empty() {
[SINK L0222]             return;
[SINK L0223]         }
[SINK L0224]         let gamma = mix.clamp(0.0, 1.0);
[SINK L0225]         if gamma <= 0.0 {
[SINK L0226]             return;
[SINK L0227]         }
[SINK L0228]
[SINK L0229]         let uniform = 1.0 / self.components.len() as f64;
[SINK L0230]         let blended: Vec<f64> = self
[SINK L0231]             .weights()
[SINK L0232]             .into_iter()
[SINK L0233]             .map(|w| (1.0 - gamma) * w + gamma * uniform)
[SINK L0234]             .collect();
[SINK L0235]         self.set_weights_from_probs(&blended);
[SINK L0236]     }
[SINK L0237]
[SINK L0238]     pub fn apply_diversity_penalty(&mut self, penalty: f64) {
[SINK L0239]         if self.components.len() < 2 || penalty <= 0.0 {
[SINK L0240]             return;
[SINK L0241]         }
[SINK L0242]
[SINK L0243]         let mut adjusted = self.weights();
[SINK L0244]         for (i, weight) in adjusted.iter_mut().enumerate() {
[SINK L0245]             let mut overlap_penalty = 0.0;
[SINK L0246]             for j in 0..self.components.len() {
[SINK L0247]                 if i == j {
[SINK L0248]                     continue;
[SINK L0249]                 }
[SINK L0250]                 let dist = Self::belief_l1_distance(
[SINK L0251]                     &self.components[i].belief,
[SINK L0252]                     &self.components[j].belief,
[SINK L0253]                 );
[SINK L0254]                 overlap_penalty += 1.0 / (1.0 + dist);
[SINK L0255]             }
[SINK L0256]             *weight *= (-penalty * overlap_penalty).exp();
[SINK L0257]         }
[SINK L0258]         self.set_weights_from_probs(&adjusted);
[SINK L0259]     }
[SINK L0260]
[SINK L0261]     pub fn split_dominant_component_if_low_ess(&mut self, min_ess_ratio: f64, jitter: f64) -> bool {
[SINK L0262]         if self.components.is_empty() {
[SINK L0263]             return false;
[SINK L0264]         }
[SINK L0265]         let threshold = min_ess_ratio * self.components.len() as f64;
[SINK L0266]         if self.ess() >= threshold {
[SINK L0267]             return false;
[SINK L0268]         }
[SINK L0269]
[SINK L0270]         let dominant = self.dominant_component();
[SINK L0271]         let mut clone = SibComponent {
[SINK L0272]             belief: self.components[dominant].belief,
[SINK L0273]             log_weight: self.components[dominant].log_weight,
[SINK L0274]         };
[SINK L0275]         let total_mass: f64 = clone.belief.iter().sum();
[SINK L0276]         let delta = jitter.abs();
[SINK L0277]         for (idx, value) in clone.belief.iter_mut().enumerate() {
[SINK L0278]             let factor = if idx % 2 == 0 {
[SINK L0279]                 1.0 + delta
[SINK L0280]             } else {
[SINK L0281]                 1.0 - delta
[SINK L0282]             };
[SINK L0283]             *value = (*value * factor).max(0.0);
[SINK L0284]         }
[SINK L0285]         let new_mass: f64 = clone.belief.iter().sum();
[SINK L0286]         if total_mass > 0.0 && new_mass > 0.0 {
[SINK L0287]             let scale = total_mass / new_mass;
[SINK L0288]             for value in &mut clone.belief {
[SINK L0289]                 *value *= scale;
[SINK L0290]             }
[SINK L0291]         }
[SINK L0292]
[SINK L0293]         let split_weight = self.components[dominant].log_weight - (2.0f64).ln();
[SINK L0294]         self.components[dominant].log_weight = split_weight;
[SINK L0295]         clone.log_weight = split_weight;
[SINK L0296]         self.components.push(clone);
[SINK L0297]         self.renormalize_log_weights();
[SINK L0298]         true
[SINK L0299]     }
[SINK L0300]
[SINK L0301]     pub fn merge_closest_components(&mut self, distance_threshold: f64) -> bool {
[SINK L0302]         if self.components.len() < 2 {
[SINK L0303]             return false;
[SINK L0304]         }
[SINK L0305]
[SINK L0306]         let mut best_pair = None;
[SINK L0307]         let mut best_distance = f64::INFINITY;
[SINK L0308]         for i in 0..self.components.len() {
[SINK L0309]             for j in (i + 1)..self.components.len() {
[SINK L0310]                 let distance = Self::belief_l1_distance(
[SINK L0311]                     &self.components[i].belief,
[SINK L0312]                     &self.components[j].belief,
[SINK L0313]                 );
[SINK L0314]                 if distance < best_distance {
[SINK L0315]                     best_distance = distance;
[SINK L0316]                     best_pair = Some((i, j));
[SINK L0317]                 }
[SINK L0318]             }
[SINK L0319]         }
[SINK L0320]
[SINK L0321]         let Some((left, right)) = best_pair else {
[SINK L0322]             return false;
[SINK L0323]         };
[SINK L0324]         if best_distance > distance_threshold {
[SINK L0325]             return false;
[SINK L0326]         }
[SINK L0327]
[SINK L0328]         let log_pair = log_sum_exp(&[
[SINK L0329]             self.components[left].log_weight,
[SINK L0330]             self.components[right].log_weight,
[SINK L0331]         ]);
[SINK L0332]         let left_w = (self.components[left].log_weight - log_pair).exp();
[SINK L0333]         let right_w = (self.components[right].log_weight - log_pair).exp();
[SINK L0334]         let mut merged = [0.0f64; BELIEF_SIZE];
[SINK L0335]         for (idx, value) in merged.iter_mut().enumerate() {
[SINK L0336]             *value = left_w * self.components[left].belief[idx]
[SINK L0337]                 + right_w * self.components[right].belief[idx];
[SINK L0338]         }
[SINK L0339]
[SINK L0340]         self.components[left].belief = merged;
[SINK L0341]         self.components[left].log_weight = log_pair;
[SINK L0342]         self.components.remove(right);
[SINK L0343]         self.renormalize_log_weights();
[SINK L0344]         true
[SINK L0345]     }
[SINK L0346]
[SINK L0347]     pub fn posterior_step(
[SINK L0348]         &mut self,
[SINK L0349]         event_log_likelihoods: &[f64],
[SINK L0350]         entropy_mix: f64,
[SINK L0351]         min_ess_ratio: f64,
[SINK L0352]         split_jitter: f64,
[SINK L0353]         merge_distance_threshold: f64,
[SINK L0354]         diversity_penalty: f64,
[SINK L0355]     ) {
[SINK L0356]         self.bayesian_update(event_log_likelihoods);
[SINK L0357]         self.apply_diversity_penalty(diversity_penalty);
[SINK L0358]         self.apply_entropy_regularizer(entropy_mix);
[SINK L0359]         self.split_dominant_component_if_low_ess(min_ess_ratio, split_jitter);
[SINK L0360]         self.merge_closest_components(merge_distance_threshold);
```

## Artifact 08 — CT-SMC config and defaults
Artifact id: `ct-smc-core`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:1-47`
Why it matters: Defines the concrete particle/ESS budget and validation surface for the current CT-SMC implementation.

```rust
[CTSMC L0001] //! CT-SMC: Exact contingency-table sampler via log-space DP.
[CTSMC L0002] //!
[CTSMC L0003] //! Samples hidden tile allocations X[34][4] respecting row/column sum
[CTSMC L0004] //! constraints. Uses a 3D DP (c1,c2,c3) with c_W derived. Sub-millisecond
[CTSMC L0005] //! for 128 backward samples in optimized Rust.
[CTSMC L0006]
[CTSMC L0007] use rand::Rng;
[CTSMC L0008] use std::collections::HashMap;
[CTSMC L0009]
[CTSMC L0010] pub struct CtSmcConfig {
[CTSMC L0011]     pub num_particles: usize,
[CTSMC L0012]     pub ess_threshold: f32,
[CTSMC L0013]     pub rng_seed: u64,
[CTSMC L0014] }
[CTSMC L0015]
[CTSMC L0016] impl CtSmcConfig {
[CTSMC L0017]     pub fn with_particles(mut self, n: usize) -> Self {
[CTSMC L0018]         self.num_particles = n;
[CTSMC L0019]         self
[CTSMC L0020]     }
[CTSMC L0021]
[CTSMC L0022]     pub fn summary(&self) -> String {
[CTSMC L0023]         format!(
[CTSMC L0024]             "ct_smc(P={}, ess_th={:.1})",
[CTSMC L0025]             self.num_particles, self.ess_threshold
[CTSMC L0026]         )
[CTSMC L0027]     }
[CTSMC L0028]
[CTSMC L0029]     pub fn validate(&self) -> Result<(), &'static str> {
[CTSMC L0030]         if self.num_particles == 0 {
[CTSMC L0031]             return Err("num_particles must be > 0");
[CTSMC L0032]         }
[CTSMC L0033]         if self.ess_threshold <= 0.0 || self.ess_threshold >= 1.0 {
[CTSMC L0034]             return Err("ess_threshold in (0,1)");
[CTSMC L0035]         }
[CTSMC L0036]         Ok(())
[CTSMC L0037]     }
[CTSMC L0038] }
[CTSMC L0039]
[CTSMC L0040] impl Default for CtSmcConfig {
[CTSMC L0041]     fn default() -> Self {
[CTSMC L0042]         Self {
[CTSMC L0043]             num_particles: 128,
[CTSMC L0044]             ess_threshold: 0.4,
[CTSMC L0045]             rng_seed: 42,
[CTSMC L0046]         }
[CTSMC L0047]     }
```

## Artifact 09 — CT-SMC update, ESS, weighted means, and resampling
Artifact id: `ct-smc-update-and-means`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:280-440`
Why it matters: High-value pass-two artifact for deciding whether weighted world posteriors are already strong enough to anchor the teacher path.

```rust
[CTSMC L0280]     pub fn update<R: Rng>(
[CTSMC L0281]         &mut self,
[CTSMC L0282]         row_sums: &[u8; 34],
[CTSMC L0283]         col_sums: &[usize; 4],
[CTSMC L0284]         log_omega: &[[f64; 4]; 34],
[CTSMC L0285]         likelihood_fn: &dyn Fn(&Particle) -> f64,
[CTSMC L0286]         rng: &mut R,
[CTSMC L0287]     ) {
[CTSMC L0288]         self.sample_particles(row_sums, col_sums, log_omega, rng);
[CTSMC L0289]         for p in &mut self.particles {
[CTSMC L0290]             p.log_weight = likelihood_fn(p);
[CTSMC L0291]         }
[CTSMC L0292]         let ess = self.ess();
[CTSMC L0293]         if ess < self.config.ess_threshold * self.config.num_particles as f32 {
[CTSMC L0294]             self.systematic_resample(rng);
[CTSMC L0295]         }
[CTSMC L0296]     }
[CTSMC L0297]
[CTSMC L0298]     pub fn clear(&mut self) {
[CTSMC L0299]         self.particles.clear();
[CTSMC L0300]         self.dp_cache = None;
[CTSMC L0301]     }
[CTSMC L0302]
[CTSMC L0303]     pub fn weighted_mean_tile_count(&self, tile: u8, col: u8) -> f32 {
[CTSMC L0304]         if self.particles.is_empty() || tile >= 34 || col >= 4 {
[CTSMC L0305]             return 0.0;
[CTSMC L0306]         }
[CTSMC L0307]         let max_w = self
[CTSMC L0308]             .particles
[CTSMC L0309]             .iter()
[CTSMC L0310]             .map(|p| p.log_weight)
[CTSMC L0311]             .fold(f64::NEG_INFINITY, f64::max);
[CTSMC L0312]         let mut sum = 0.0f64;
[CTSMC L0313]         let mut w_sum = 0.0f64;
[CTSMC L0314]         for p in &self.particles {
[CTSMC L0315]             let w = (p.log_weight - max_w).exp();
[CTSMC L0316]             sum += w * p.allocation[tile as usize][col as usize] as f64;
[CTSMC L0317]             w_sum += w;
[CTSMC L0318]         }
[CTSMC L0319]         if w_sum > 0.0 {
[CTSMC L0320]             (sum / w_sum) as f32
[CTSMC L0321]         } else {
[CTSMC L0322]             0.0
[CTSMC L0323]         }
[CTSMC L0324]     }
[CTSMC L0325]
[CTSMC L0326]     pub fn is_empty(&self) -> bool {
[CTSMC L0327]         self.particles.is_empty()
[CTSMC L0328]     }
[CTSMC L0329]
[CTSMC L0330]     pub fn needs_resample(&self) -> bool {
[CTSMC L0331]         self.ess() < self.config.ess_threshold * self.particles.len() as f32
[CTSMC L0332]     }
[CTSMC L0333]
[CTSMC L0334]     pub fn ess_ratio(&self) -> f32 {
[CTSMC L0335]         if self.particles.is_empty() {
[CTSMC L0336]             return 0.0;
[CTSMC L0337]         }
[CTSMC L0338]         self.ess() / self.particles.len() as f32
[CTSMC L0339]     }
[CTSMC L0340]
[CTSMC L0341]     pub fn mean_allocation(&self) -> [[f32; 4]; 34] {
[CTSMC L0342]         let mut result = [[0.0f32; 4]; 34];
[CTSMC L0343]         if self.particles.is_empty() {
[CTSMC L0344]             return result;
[CTSMC L0345]         }
[CTSMC L0346]         let n = self.particles.len() as f32;
[CTSMC L0347]         for p in &self.particles {
[CTSMC L0348]             for (res_row, alloc_row) in result.iter_mut().zip(p.allocation.iter()) {
[CTSMC L0349]                 for (v, &a) in res_row.iter_mut().zip(alloc_row.iter()) {
[CTSMC L0350]                     *v += a as f32;
[CTSMC L0351]                 }
[CTSMC L0352]             }
[CTSMC L0353]         }
[CTSMC L0354]         for row in &mut result {
[CTSMC L0355]             for v in row {
[CTSMC L0356]                 *v /= n;
[CTSMC L0357]             }
[CTSMC L0358]         }
[CTSMC L0359]         result
[CTSMC L0360]     }
[CTSMC L0361]
[CTSMC L0362]     pub fn summary(&self) -> String {
[CTSMC L0363]         format!("smc(P={}, ess={:.1})", self.num_particles(), self.ess())
[CTSMC L0364]     }
[CTSMC L0365]
[CTSMC L0366]     pub fn max_log_weight(&self) -> f64 {
[CTSMC L0367]         self.particles
[CTSMC L0368]             .iter()
[CTSMC L0369]             .map(|p| p.log_weight)
[CTSMC L0370]             .fold(f64::NEG_INFINITY, f64::max)
[CTSMC L0371]     }
[CTSMC L0372]
[CTSMC L0373]     pub fn has_dp_cache(&self) -> bool {
[CTSMC L0374]         self.dp_cache.is_some()
[CTSMC L0375]     }
[CTSMC L0376]
[CTSMC L0377]     pub fn num_particles(&self) -> usize {
[CTSMC L0378]         self.particles.len()
[CTSMC L0379]     }
[CTSMC L0380]
[CTSMC L0381]     pub fn ess(&self) -> f32 {
[CTSMC L0382]         if self.particles.is_empty() {
[CTSMC L0383]             return 0.0;
[CTSMC L0384]         }
[CTSMC L0385]         let max_w = self
[CTSMC L0386]             .particles
[CTSMC L0387]             .iter()
[CTSMC L0388]             .map(|p| p.log_weight)
[CTSMC L0389]             .fold(f64::NEG_INFINITY, f64::max);
[CTSMC L0390]         let weights: Vec<f64> = self
[CTSMC L0391]             .particles
[CTSMC L0392]             .iter()
[CTSMC L0393]             .map(|p| (p.log_weight - max_w).exp())
[CTSMC L0394]             .collect();
[CTSMC L0395]         let sum: f64 = weights.iter().sum();
[CTSMC L0396]         let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
[CTSMC L0397]         if sum_sq == 0.0 {
[CTSMC L0398]             return 0.0;
[CTSMC L0399]         }
[CTSMC L0400]         ((sum * sum) / sum_sq) as f32
[CTSMC L0401]     }
[CTSMC L0402]
[CTSMC L0403]     pub fn systematic_resample<R: Rng>(&mut self, rng: &mut R) {
[CTSMC L0404]         let n = self.particles.len();
[CTSMC L0405]         if n == 0 {
[CTSMC L0406]             return;
[CTSMC L0407]         }
[CTSMC L0408]         let max_w = self
[CTSMC L0409]             .particles
[CTSMC L0410]             .iter()
[CTSMC L0411]             .map(|p| p.log_weight)
[CTSMC L0412]             .fold(f64::NEG_INFINITY, f64::max);
[CTSMC L0413]         let weights: Vec<f64> = self
[CTSMC L0414]             .particles
[CTSMC L0415]             .iter()
[CTSMC L0416]             .map(|p| (p.log_weight - max_w).exp())
[CTSMC L0417]             .collect();
[CTSMC L0418]         let total: f64 = weights.iter().sum();
[CTSMC L0419]         let step = total / n as f64;
[CTSMC L0420]         let mut u: f64 = rng.random::<f64>() * step;
[CTSMC L0421]         let mut cumsum = 0.0;
[CTSMC L0422]         let mut indices = Vec::with_capacity(n);
[CTSMC L0423]         let mut j = 0;
[CTSMC L0424]         for _ in 0..n {
[CTSMC L0425]             while cumsum + weights[j] < u && j + 1 < n {
[CTSMC L0426]                 cumsum += weights[j];
[CTSMC L0427]                 j += 1;
[CTSMC L0428]             }
[CTSMC L0429]             indices.push(j);
[CTSMC L0430]             u += step;
[CTSMC L0431]         }
[CTSMC L0432]         let old = std::mem::take(&mut self.particles);
[CTSMC L0433]         self.particles = indices
[CTSMC L0434]             .into_iter()
[CTSMC L0435]             .map(|i| Particle {
[CTSMC L0436]                 allocation: old[i].allocation,
[CTSMC L0437]                 log_weight: 0.0,
[CTSMC L0438]             })
[CTSMC L0439]             .collect();
[CTSMC L0440]     }
```

## Artifact 10 — Current train-bin loss blocking rules
Artifact id: `loss-policy-blocks`
Source label: LOSSPOL
Type: `file_full`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs`
Why it matters: Shows exactly which advanced losses are blocked in the train entrypoints and therefore where contract migration pressure exists.

```rust
[LOSSPOL L0001] use hydra_train::training::bc::BcExitConfig;
[LOSSPOL L0002] use hydra_train::training::losses::HydraLossConfig;
[LOSSPOL L0003]
[LOSSPOL L0004] use super::config::AdvancedLossConfig;
[LOSSPOL L0005]
[LOSSPOL L0006] fn reject_blocked_advanced_loss_presence(field: &str, weight: Option<f32>) -> Result<(), String> {
[LOSSPOL L0007]     match weight {
[LOSSPOL L0008]         Some(_) => Err(format!(
[LOSSPOL L0009]             "advanced_loss.{field} is not supported in train.rs because this BC data path does not safely support it yet"
[LOSSPOL L0010]         )),
[LOSSPOL L0011]         None => Ok(()),
[LOSSPOL L0012]     }
[LOSSPOL L0013] }
[LOSSPOL L0014]
[LOSSPOL L0015] pub(super) fn build_loss_config(
[LOSSPOL L0016]     advanced_loss: Option<&AdvancedLossConfig>,
[LOSSPOL L0017] ) -> Result<HydraLossConfig, String> {
[LOSSPOL L0018]     if let Some(cfg) = advanced_loss {
[LOSSPOL L0019]         reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
[LOSSPOL L0020]         reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
[LOSSPOL L0021]         reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
[LOSSPOL L0022]     }
[LOSSPOL L0023]
[LOSSPOL L0024]     let safety_residual = advanced_loss
[LOSSPOL L0025]         .and_then(|cfg| cfg.safety_residual)
[LOSSPOL L0026]         .unwrap_or(0.0);
[LOSSPOL L0027]     let delta_q = advanced_loss.and_then(|cfg| cfg.delta_q).unwrap_or(0.0);
[LOSSPOL L0028]
[LOSSPOL L0029]     let loss_config = HydraLossConfig::new()
[LOSSPOL L0030]         .with_w_safety_residual(safety_residual)
[LOSSPOL L0031]         .with_w_delta_q(delta_q);
[LOSSPOL L0032]     loss_config
[LOSSPOL L0033]         .validate()
[LOSSPOL L0034]         .map_err(|err| format!("invalid loss config: {err}"))?;
[LOSSPOL L0035]     Ok(loss_config)
[LOSSPOL L0036] }
[LOSSPOL L0037]
[LOSSPOL L0038] pub(super) fn build_bc_exit_config(advanced_loss: Option<&AdvancedLossConfig>) -> BcExitConfig {
[LOSSPOL L0039]     let exit_weight = advanced_loss.and_then(|cfg| cfg.exit).unwrap_or(0.0);
[LOSSPOL L0040]     BcExitConfig { exit_weight }
[LOSSPOL L0041] }
[LOSSPOL L0042]
[LOSSPOL L0043] pub(super) fn build_rl_loss_config(
[LOSSPOL L0044]     advanced_loss: Option<&AdvancedLossConfig>,
[LOSSPOL L0045] ) -> Result<HydraLossConfig, String> {
[LOSSPOL L0046]     if let Some(cfg) = advanced_loss {
[LOSSPOL L0047]         reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
[LOSSPOL L0048]         reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
[LOSSPOL L0049]         reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
[LOSSPOL L0050]     }
[LOSSPOL L0051]
[LOSSPOL L0052]     let mut loss = HydraLossConfig::new();
[LOSSPOL L0053]     if let Some(cfg) = advanced_loss {
[LOSSPOL L0054]         loss = loss
[LOSSPOL L0055]             .with_w_safety_residual(cfg.safety_residual.unwrap_or(0.0))
[LOSSPOL L0056]             .with_w_delta_q(cfg.delta_q.unwrap_or(0.0));
[LOSSPOL L0057]     }
[LOSSPOL L0058]     loss.validate()
[LOSSPOL L0059]         .map_err(|err| format!("invalid RL loss config: {err}"))?;
[LOSSPOL L0060]     Ok(loss)
[LOSSPOL L0061] }
[LOSSPOL L0062]
[LOSSPOL L0063] #[cfg(test)]
[LOSSPOL L0064] mod tests {
[LOSSPOL L0065]     use super::*;
[LOSSPOL L0066]
[LOSSPOL L0067]     #[test]
[LOSSPOL L0068]     fn build_loss_config_defaults_to_zero_optional_weights() {
[LOSSPOL L0069]         let cfg = build_loss_config(None).expect("default loss config should be valid");
[LOSSPOL L0070]         assert_eq!(cfg.w_safety_residual, 0.0);
[LOSSPOL L0071]         assert_eq!(cfg.w_delta_q, 0.0);
[LOSSPOL L0072]
[LOSSPOL L0073]         let exit_cfg = build_bc_exit_config(None);
[LOSSPOL L0074]         assert_eq!(exit_cfg.exit_weight, 0.0);
[LOSSPOL L0075]     }
[LOSSPOL L0076]
[LOSSPOL L0077]     #[test]
[LOSSPOL L0078]     fn build_loss_config_rejects_blocked_advanced_loss_fields() {
[LOSSPOL L0079]         let advanced = AdvancedLossConfig {
[LOSSPOL L0080]             belief_fields: Some(0.1),
[LOSSPOL L0081]             ..AdvancedLossConfig::default()
[LOSSPOL L0082]         };
[LOSSPOL L0083]         let err = build_loss_config(Some(&advanced)).expect_err("belief fields should be blocked");
[LOSSPOL L0084]         assert!(err.contains("advanced_loss.belief_fields is not supported"));
[LOSSPOL L0085]     }
[LOSSPOL L0086]
[LOSSPOL L0087]     #[test]
[LOSSPOL L0088]     fn build_configs_propagate_supported_weights() {
[LOSSPOL L0089]         let advanced = AdvancedLossConfig {
[LOSSPOL L0090]             exit: Some(0.4),
[LOSSPOL L0091]             safety_residual: Some(0.2),
[LOSSPOL L0092]             delta_q: Some(0.3),
[LOSSPOL L0093]             ..AdvancedLossConfig::default()
[LOSSPOL L0094]         };
[LOSSPOL L0095]
[LOSSPOL L0096]         let bc_loss = build_loss_config(Some(&advanced)).expect("bc loss config should build");
[LOSSPOL L0097]         assert_eq!(bc_loss.w_safety_residual, 0.2);
[LOSSPOL L0098]         assert_eq!(bc_loss.w_delta_q, 0.3);
[LOSSPOL L0099]
[LOSSPOL L0100]         let bc_exit = build_bc_exit_config(Some(&advanced));
[LOSSPOL L0101]         assert_eq!(bc_exit.exit_weight, 0.4);
[LOSSPOL L0102]
[LOSSPOL L0103]         let rl_loss = build_rl_loss_config(Some(&advanced)).expect("rl loss config should build");
[LOSSPOL L0104]         assert_eq!(rl_loss.w_safety_residual, 0.2);
[LOSSPOL L0105]         assert_eq!(rl_loss.w_delta_q, 0.3);
[LOSSPOL L0106]     }
[LOSSPOL L0107] }
```

## Artifact 11 — Head activation discipline and gating surfaces
Artifact id: `head-gates-core`
Source label: HGATE
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1-260`
Why it matters: Strong pass-two artifact for promotion-gate reuse or replacement: density, conflict, warmup, and target-presence mechanics.

```rust
[HGATE L0001] //! Head activation discipline: density, interference, and warmup gates.
[HGATE L0002] //!
[HGATE L0003] //! Prevents sparse or noisy advanced heads from dragging the shared SE-ResNet
[HGATE L0004] //! trunk backward via negative transfer. Implements the archive's gate pack
[HGATE L0005] //! from `answer_13_combined.md` sections 3.3, 5, and 6:
[HGATE L0006] //!
[HGATE L0007] //! - **Density gate**: Per-head label density `rho_h` for dense heads
[HGATE L0008] //!   (threshold: `rho >= 0.8`) and samples-per-param `spp_h` for sparse
[HGATE L0009] //!   search-derived heads (threshold: `spp >= 5.0`).
[HGATE L0010] //!
[HGATE L0011] //! - **Gradient conflict gate**: Shared-trunk gradient cosine between each
[HGATE L0012] //!   auxiliary head loss and the policy+value loss. Heads are kept off if
[HGATE L0013] //!   cosine is negative on >30% of checks after warmup.
[HGATE L0014] //!
[HGATE L0015] //! - **Warmup protocol**: When activating a head, train head-only (trunk
[HGATE L0016] //!   frozen) for a configurable number of steps before unfreezing. Transition
[HGATE L0017] //!   to full activation only if gradient conflict gate passes.
[HGATE L0018] //!
[HGATE L0019] //! # Gate sequence
[HGATE L0020] //!
[HGATE L0021] //! 1. Target correctness audit (manual prerequisite, not automated here).
[HGATE L0022] //! 2. Density gate: `rho_h >= min_dense_rho` or `spp_h >= min_sparse_spp`.
[HGATE L0023] //! 3. Head-only warmup with trunk frozen for `warmup_steps` updates.
[HGATE L0024] //! 4. Gradient conflict gate: negative cosine fraction < `max_negative_frac`.
[HGATE L0025] //! 5. Feature-ablation gate (requires evaluation infrastructure, documented
[HGATE L0026] //!    but not automated here).
[HGATE L0027] //!
[HGATE L0028] //! The controller manages per-head state transitions:
[HGATE L0029] //! `Off` -> (density passes) -> `Warmup` -> (conflict passes) -> `Active`.
[HGATE L0030] //!
[HGATE L0031] //! # Integration
[HGATE L0032] //!
[HGATE L0033] //! The caller (orchestrator) is responsible for:
[HGATE L0034] //! - Calling [`extract_target_presence`] and
[HGATE L0035] //!   [`HeadActivationController::record_batch`] each training step.
[HGATE L0036] //! - Periodically computing shared-trunk gradient cosine (see
[HGATE L0037] //!   [`grad_cosine_from_flat`]) and calling
[HGATE L0038] //!   [`HeadActivationController::record_grad_cosine`].
[HGATE L0039] //! - Using [`HeadActivationController::approved_loss_config`] to get effective
[HGATE L0040] //!   loss weights (unapproved heads are zeroed out).
[HGATE L0041] //! - Checking [`HeadActivationController::warmup_heads`] and detaching trunk
[HGATE L0042] //!   outputs for heads in warmup state so they train head-only.
[HGATE L0043] //!
[HGATE L0044] //! # Important: do not use `grad_norm_approx` for the conflict gate
[HGATE L0045] //!
[HGATE L0046] //! The existing `grad_norm_approx` in `losses.rs` is a loss-magnitude proxy,
[HGATE L0047] //! not a true parameter-gradient norm. Use [`grad_cosine_from_flat`] with
[HGATE L0048] //! real flattened shared-trunk gradients instead.
[HGATE L0049]
[HGATE L0050] use crate::training::losses::{HydraLossConfig, HydraTargets};
[HGATE L0051] use burn::prelude::*;
[HGATE L0052]
[HGATE L0053] // ---------------------------------------------------------------------------
[HGATE L0054] // Constants (archive-recommended defaults from answer_13_combined.md)
[HGATE L0055] // ---------------------------------------------------------------------------
[HGATE L0056]
[HGATE L0057] /// Number of gated advanced heads.
[HGATE L0058] pub const NUM_ADVANCED_HEADS: usize = 6;
[HGATE L0059]
[HGATE L0060] /// Dense heads require at least 80% of samples to carry the target.
[HGATE L0061] pub const DEFAULT_MIN_DENSE_RHO: f32 = 0.8;
[HGATE L0062]
[HGATE L0063] /// Sparse search-derived heads require at least 5 labeled samples per
[HGATE L0064] /// learner parameter.
[HGATE L0065] pub const DEFAULT_MIN_SPARSE_SPP: f32 = 5.0;
[HGATE L0066]
[HGATE L0067] /// A head is considered conflicting if shared-trunk gradient cosine with
[HGATE L0068] /// policy+value is negative on more than 30% of checks.
[HGATE L0069] pub const DEFAULT_MAX_NEGATIVE_FRAC: f32 = 0.3;
[HGATE L0070]
[HGATE L0071] /// Head-only warmup duration (trunk frozen) before unfreeze decision.
[HGATE L0072] pub const DEFAULT_WARMUP_STEPS: usize = 10_000;
[HGATE L0073]
[HGATE L0074] /// Minimum accumulated samples before density evaluation is meaningful.
[HGATE L0075] pub const DEFAULT_MIN_EVAL_SAMPLES: u64 = 1000;
[HGATE L0076]
[HGATE L0077] /// Minimum gradient cosine checks before conflict gate is evaluated.
[HGATE L0078] pub const DEFAULT_MIN_CONFLICT_CHECKS: u64 = 10;
[HGATE L0079]
[HGATE L0080] // ---------------------------------------------------------------------------
[HGATE L0081] // AdvancedHead -- the six gated output heads
[HGATE L0082] // ---------------------------------------------------------------------------
[HGATE L0083]
[HGATE L0084] /// Advanced output heads subject to activation gating.
[HGATE L0085] ///
[HGATE L0086] /// These are the heads whose loss weights default to zero and require
[HGATE L0087] /// density/interference clearance before activation.
[HGATE L0088] #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
[HGATE L0089] pub enum AdvancedHead {
[HGATE L0090]     OracleCritic,
[HGATE L0091]     BeliefFields,
[HGATE L0092]     MixtureWeight,
[HGATE L0093]     OpponentHandType,
[HGATE L0094]     DeltaQ,
[HGATE L0095]     SafetyResidual,
[HGATE L0096] }
[HGATE L0097]
[HGATE L0098] impl AdvancedHead {
[HGATE L0099]     /// All advanced heads in index order.
[HGATE L0100]     pub const ALL: [AdvancedHead; NUM_ADVANCED_HEADS] = [
[HGATE L0101]         Self::OracleCritic,
[HGATE L0102]         Self::BeliefFields,
[HGATE L0103]         Self::MixtureWeight,
[HGATE L0104]         Self::OpponentHandType,
[HGATE L0105]         Self::DeltaQ,
[HGATE L0106]         Self::SafetyResidual,
[HGATE L0107]     ];
[HGATE L0108]
[HGATE L0109]     /// Returns the array index for this head.
[HGATE L0110]     pub fn index(self) -> usize {
[HGATE L0111]         match self {
[HGATE L0112]             Self::OracleCritic => 0,
[HGATE L0113]             Self::BeliefFields => 1,
[HGATE L0114]             Self::MixtureWeight => 2,
[HGATE L0115]             Self::OpponentHandType => 3,
[HGATE L0116]             Self::DeltaQ => 4,
[HGATE L0117]             Self::SafetyResidual => 5,
[HGATE L0118]         }
[HGATE L0119]     }
[HGATE L0120]
[HGATE L0121]     /// Returns whether this head uses dense or sparse-search density rules.
[HGATE L0122]     pub fn kind(self) -> HeadKind {
[HGATE L0123]         match self {
[HGATE L0124]             Self::DeltaQ => HeadKind::SparseSearch,
[HGATE L0125]             _ => HeadKind::Dense,
[HGATE L0126]         }
[HGATE L0127]     }
[HGATE L0128]
[HGATE L0129]     /// Returns the snake_case name matching `HydraLossConfig` field names.
[HGATE L0130]     pub fn name(self) -> &'static str {
[HGATE L0131]         match self {
[HGATE L0132]             Self::OracleCritic => "oracle_critic",
[HGATE L0133]             Self::BeliefFields => "belief_fields",
[HGATE L0134]             Self::MixtureWeight => "mixture_weight",
[HGATE L0135]             Self::OpponentHandType => "opponent_hand_type",
[HGATE L0136]             Self::DeltaQ => "delta_q",
[HGATE L0137]             Self::SafetyResidual => "safety_residual",
[HGATE L0138]         }
[HGATE L0139]     }
[HGATE L0140] }
[HGATE L0141]
[HGATE L0142] // ---------------------------------------------------------------------------
[HGATE L0143] // HeadKind -- density threshold selection
[HGATE L0144] // ---------------------------------------------------------------------------
[HGATE L0145]
[HGATE L0146] /// Classification that determines which density threshold applies.
[HGATE L0147] #[derive(Clone, Copy, Debug, PartialEq, Eq)]
[HGATE L0148] pub enum HeadKind {
[HGATE L0149]     /// Replay-derived head. Gate: `rho_h >= min_dense_rho`.
[HGATE L0150]     Dense,
[HGATE L0151]     /// Search-derived head with sparse labels. Gate: `spp_h >= min_sparse_spp`.
[HGATE L0152]     SparseSearch,
[HGATE L0153] }
[HGATE L0154]
[HGATE L0155] // ---------------------------------------------------------------------------
[HGATE L0156] // HeadState -- per-head activation state machine
[HGATE L0157] // ---------------------------------------------------------------------------
[HGATE L0158]
[HGATE L0159] /// Per-head activation state.
[HGATE L0160] #[derive(Clone, Copy, Debug, PartialEq, Eq)]
[HGATE L0161] pub enum HeadState {
[HGATE L0162]     /// Head is off: loss weight is forced to zero.
[HGATE L0163]     Off,
[HGATE L0164]     /// Head is warming up: loss weight is nonzero but the caller should
[HGATE L0165]     /// freeze (detach) trunk outputs for this head's loss so only the head
[HGATE L0166]     /// parameters train.
[HGATE L0167]     Warmup,
[HGATE L0168]     /// Head is fully active: loss weight is nonzero and trunk gradient flow
[HGATE L0169]     /// is unrestricted.
[HGATE L0170]     Active,
[HGATE L0171] }
[HGATE L0172]
[HGATE L0173] // ---------------------------------------------------------------------------
[HGATE L0174] // TargetPresence -- per-batch target availability snapshot
[HGATE L0175] // ---------------------------------------------------------------------------
[HGATE L0176]
[HGATE L0177] /// Per-head count of samples with valid targets in a single batch.
[HGATE L0178] #[derive(Clone, Debug)]
[HGATE L0179] pub struct TargetPresence {
[HGATE L0180]     /// Per-head count of samples carrying a valid target in this batch.
[HGATE L0181]     pub counts: [usize; NUM_ADVANCED_HEADS],
[HGATE L0182]     /// Total samples in this batch.
[HGATE L0183]     pub batch_size: usize,
[HGATE L0184] }
[HGATE L0185]
[HGATE L0186] impl Default for TargetPresence {
[HGATE L0187]     fn default() -> Self {
[HGATE L0188]         Self {
[HGATE L0189]             counts: [0; NUM_ADVANCED_HEADS],
[HGATE L0190]             batch_size: 0,
[HGATE L0191]         }
[HGATE L0192]     }
[HGATE L0193] }
[HGATE L0194]
[HGATE L0195] impl TargetPresence {
[HGATE L0196]     /// Creates a presence snapshot with the given batch size and all counts zero.
[HGATE L0197]     pub fn with_batch_size(batch_size: usize) -> Self {
[HGATE L0198]         Self {
[HGATE L0199]             counts: [0; NUM_ADVANCED_HEADS],
[HGATE L0200]             batch_size,
[HGATE L0201]         }
[HGATE L0202]     }
[HGATE L0203]
[HGATE L0204]     /// Returns the number of samples with a valid target for `head`.
[HGATE L0205]     pub fn count(&self, head: AdvancedHead) -> usize {
[HGATE L0206]         self.counts[head.index()]
[HGATE L0207]     }
[HGATE L0208] }
[HGATE L0209]
[HGATE L0210] /// Extracts per-head target presence from a batch of [`HydraTargets`].
[HGATE L0211] ///
[HGATE L0212] /// For targets with per-sample masks (`belief_fields`, `mixture_weight`),
[HGATE L0213] /// counts the number of samples where the mask is nonzero. For targets
[HGATE L0214] /// without per-sample masks, counts `batch_size` when the target is present.
[HGATE L0215] pub fn extract_target_presence<B: Backend>(targets: &HydraTargets<B>) -> TargetPresence {
[HGATE L0216]     let batch_size = targets.policy_target.dims()[0];
[HGATE L0217]     let mut counts = [0usize; NUM_ADVANCED_HEADS];
[HGATE L0218]
[HGATE L0219]     // Oracle critic: uses oracle_guidance_mask for per-sample gating.
[HGATE L0220]     if targets.oracle_target.is_some() {
[HGATE L0221]         counts[AdvancedHead::OracleCritic.index()] = match &targets.oracle_guidance_mask {
[HGATE L0222]             Some(mask) => count_nonzero_1d(mask),
[HGATE L0223]             None => batch_size,
[HGATE L0224]         };
[HGATE L0225]     }
[HGATE L0226]
[HGATE L0227]     // Belief fields: per-sample mask.
[HGATE L0228]     if targets.belief_fields_target.is_some() {
[HGATE L0229]         counts[AdvancedHead::BeliefFields.index()] = match &targets.belief_fields_mask {
[HGATE L0230]             Some(mask) => {
[HGATE L0231]                 count_nonzero_1d_with_optional_gate(mask, targets.oracle_guidance_mask.as_ref())
[HGATE L0232]             }
[HGATE L0233]             None => batch_size,
[HGATE L0234]         };
[HGATE L0235]     }
[HGATE L0236]
[HGATE L0237]     // Mixture weight: per-sample mask.
[HGATE L0238]     if targets.mixture_weight_target.is_some() {
[HGATE L0239]         counts[AdvancedHead::MixtureWeight.index()] = match &targets.mixture_weight_mask {
[HGATE L0240]             Some(mask) => {
[HGATE L0241]                 count_nonzero_1d_with_optional_gate(mask, targets.oracle_guidance_mask.as_ref())
[HGATE L0242]             }
[HGATE L0243]             None => batch_size,
[HGATE L0244]         };
[HGATE L0245]     }
[HGATE L0246]
[HGATE L0247]     // Opponent hand type: shares oracle_guidance_mask.
[HGATE L0248]     if targets.opponent_hand_type_target.is_some() {
[HGATE L0249]         counts[AdvancedHead::OpponentHandType.index()] = match &targets.oracle_guidance_mask {
[HGATE L0250]             Some(mask) => count_nonzero_1d(mask),
[HGATE L0251]             None => batch_size,
[HGATE L0252]         };
[HGATE L0253]     }
[HGATE L0254]
[HGATE L0255]     counts[AdvancedHead::DeltaQ.index()] = match (&targets.delta_q_target, &targets.delta_q_mask) {
[HGATE L0256]         (Some(_), Some(mask)) => count_nonzero_rows_2d(mask),
[HGATE L0257]         _ => 0,
[HGATE L0258]     };
[HGATE L0259]
[HGATE L0260]     counts[AdvancedHead::SafetyResidual.index()] = match (
```

## Artifact 12 — DeltaQ promotion reports and thresholds
Artifact id: `delta-q-promotion-core`
Source label: DQPROM
Type: `file_range`
Source: `crates/hydra-train/src/training/delta_q_promotion.rs:30-220`
Why it matters: Concrete in-repo template for designing belief-lane promotion and rejection logic.

```rust
[DQPROM L0030] #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
[DQPROM L0031] pub struct DeltaQPromotionReport {
[DQPROM L0032]     pub eligible_states: u64,
[DQPROM L0033]     pub compared_states: u64,
[DQPROM L0034]     pub masked_entries: u64,
[DQPROM L0035]     pub supported_actions_sum: u64,
[DQPROM L0036]     pub candidate_top1_agreement_count: u64,
[DQPROM L0037]     pub baseline_top1_agreement_count: u64,
[DQPROM L0038]     pub candidate_high_gap_top1_count: u64,
[DQPROM L0039]     pub baseline_high_gap_top1_count: u64,
[DQPROM L0040]     pub high_gap_states: u64,
[DQPROM L0041]     pub candidate_regret_sum: f64,
[DQPROM L0042]     pub baseline_regret_sum: f64,
[DQPROM L0043]     pub decision_lift_sum: f64,
[DQPROM L0044]     pub negative_lift_count: u64,
[DQPROM L0045]     pub candidate_regret_beats_baseline_count: u64,
[DQPROM L0046]     pub candidate_top1_beats_baseline_count: u64,
[DQPROM L0047] }
[DQPROM L0048]
[DQPROM L0049] #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
[DQPROM L0050] pub struct DeltaQPolicyTransferReport {
[DQPROM L0051]     pub compared_states: u64,
[DQPROM L0052]     pub candidate_policy_top1_to_teacher_count: u64,
[DQPROM L0053]     pub baseline_policy_top1_to_teacher_count: u64,
[DQPROM L0054]     pub candidate_policy_regret_sum: f64,
[DQPROM L0055]     pub baseline_policy_regret_sum: f64,
[DQPROM L0056]     pub candidate_beats_baseline_count: u64,
[DQPROM L0057]     pub negative_transfer_count: u64,
[DQPROM L0058] }
[DQPROM L0059]
[DQPROM L0060] #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
[DQPROM L0061] pub struct DeltaQArenaReport {
[DQPROM L0062]     pub compared_games: usize,
[DQPROM L0063]     pub baseline_mean_placement: f64,
[DQPROM L0064]     pub candidate_mean_placement: f64,
[DQPROM L0065]     pub delta_mean_placement: f64,
[DQPROM L0066]     pub baseline_stable_dan: f64,
[DQPROM L0067]     pub candidate_stable_dan: f64,
[DQPROM L0068]     pub delta_stable_dan: f64,
[DQPROM L0069]     pub lower_confidence_bound_mean_placement: f64,
[DQPROM L0070]     pub upper_confidence_bound_mean_placement: f64,
[DQPROM L0071] }
[DQPROM L0072]
[DQPROM L0073] impl DeltaQArenaReport {
[DQPROM L0074]     pub fn from_paired_eval(
[DQPROM L0075]         result: &crate::eval::PairedArenaEvalResult,
[DQPROM L0076]         lower_confidence_bound_mean_placement: f32,
[DQPROM L0077]     ) -> Self {
[DQPROM L0078]         Self {
[DQPROM L0079]             compared_games: result.compared_games,
[DQPROM L0080]             baseline_mean_placement: result.baseline_mean_placement as f64,
[DQPROM L0081]             candidate_mean_placement: result.candidate_mean_placement as f64,
[DQPROM L0082]             delta_mean_placement: result.delta_mean_placement as f64,
[DQPROM L0083]             baseline_stable_dan: result.baseline_stable_dan as f64,
[DQPROM L0084]             candidate_stable_dan: result.candidate_stable_dan as f64,
[DQPROM L0085]             delta_stable_dan: result.delta_stable_dan as f64,
[DQPROM L0086]             lower_confidence_bound_mean_placement: lower_confidence_bound_mean_placement as f64,
[DQPROM L0087]             upper_confidence_bound_mean_placement: result.upper_confidence_bound_mean_placement
[DQPROM L0088]                 as f64,
[DQPROM L0089]         }
[DQPROM L0090]     }
[DQPROM L0091] }
[DQPROM L0092]
[DQPROM L0093] #[derive(Debug, Clone)]
[DQPROM L0094] pub struct DeltaQPolicyTransferThresholds {
[DQPROM L0095]     pub min_compared_states: u64,
[DQPROM L0096]     pub max_candidate_policy_mean_teacher_regret_ratio: f64,
[DQPROM L0097]     pub max_negative_transfer_fraction: f64,
[DQPROM L0098]     pub min_candidate_beats_baseline_rate: f64,
[DQPROM L0099] }
[DQPROM L0100]
[DQPROM L0101] impl Default for DeltaQPolicyTransferThresholds {
[DQPROM L0102]     fn default() -> Self {
[DQPROM L0103]         Self {
[DQPROM L0104]             min_compared_states: 1_000,
[DQPROM L0105]             max_candidate_policy_mean_teacher_regret_ratio: 0.95,
[DQPROM L0106]             max_negative_transfer_fraction: 0.45,
[DQPROM L0107]             min_candidate_beats_baseline_rate: 0.55,
[DQPROM L0108]         }
[DQPROM L0109]     }
[DQPROM L0110] }
[DQPROM L0111]
[DQPROM L0112] #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
[DQPROM L0113] pub struct DeltaQPolicyTransferResult {
[DQPROM L0114]     pub passed: bool,
[DQPROM L0115]     pub criteria: Vec<DeltaQPromotionCriterionResult>,
[DQPROM L0116] }
[DQPROM L0117]
[DQPROM L0118] impl DeltaQPolicyTransferReport {
[DQPROM L0119]     pub fn new() -> Self {
[DQPROM L0120]         Self {
[DQPROM L0121]             compared_states: 0,
[DQPROM L0122]             candidate_policy_top1_to_teacher_count: 0,
[DQPROM L0123]             baseline_policy_top1_to_teacher_count: 0,
[DQPROM L0124]             candidate_policy_regret_sum: 0.0,
[DQPROM L0125]             baseline_policy_regret_sum: 0.0,
[DQPROM L0126]             candidate_beats_baseline_count: 0,
[DQPROM L0127]             negative_transfer_count: 0,
[DQPROM L0128]         }
[DQPROM L0129]     }
[DQPROM L0130]
[DQPROM L0131]     pub fn merge(&mut self, other: &Self) {
[DQPROM L0132]         self.compared_states += other.compared_states;
[DQPROM L0133]         self.candidate_policy_top1_to_teacher_count += other.candidate_policy_top1_to_teacher_count;
[DQPROM L0134]         self.baseline_policy_top1_to_teacher_count += other.baseline_policy_top1_to_teacher_count;
[DQPROM L0135]         self.candidate_policy_regret_sum += other.candidate_policy_regret_sum;
[DQPROM L0136]         self.baseline_policy_regret_sum += other.baseline_policy_regret_sum;
[DQPROM L0137]         self.candidate_beats_baseline_count += other.candidate_beats_baseline_count;
[DQPROM L0138]         self.negative_transfer_count += other.negative_transfer_count;
[DQPROM L0139]     }
[DQPROM L0140]
[DQPROM L0141]     pub fn candidate_policy_top1_to_teacher(&self) -> f64 {
[DQPROM L0142]         ratio_u64(
[DQPROM L0143]             self.candidate_policy_top1_to_teacher_count,
[DQPROM L0144]             self.compared_states,
[DQPROM L0145]         )
[DQPROM L0146]     }
[DQPROM L0147]
[DQPROM L0148]     pub fn baseline_policy_top1_to_teacher(&self) -> f64 {
[DQPROM L0149]         ratio_u64(
[DQPROM L0150]             self.baseline_policy_top1_to_teacher_count,
[DQPROM L0151]             self.compared_states,
[DQPROM L0152]         )
[DQPROM L0153]     }
[DQPROM L0154]
[DQPROM L0155]     pub fn candidate_policy_mean_teacher_regret(&self) -> f64 {
[DQPROM L0156]         ratio_f64(self.candidate_policy_regret_sum, self.compared_states)
[DQPROM L0157]     }
[DQPROM L0158]
[DQPROM L0159]     pub fn baseline_policy_mean_teacher_regret(&self) -> f64 {
[DQPROM L0160]         ratio_f64(self.baseline_policy_regret_sum, self.compared_states)
[DQPROM L0161]     }
[DQPROM L0162]
[DQPROM L0163]     pub fn mean_regret_improvement(&self) -> f64 {
[DQPROM L0164]         self.baseline_policy_mean_teacher_regret() - self.candidate_policy_mean_teacher_regret()
[DQPROM L0165]     }
[DQPROM L0166]
[DQPROM L0167]     pub fn candidate_beats_baseline_rate(&self) -> f64 {
[DQPROM L0168]         ratio_u64(self.candidate_beats_baseline_count, self.compared_states)
[DQPROM L0169]     }
[DQPROM L0170]
[DQPROM L0171]     pub fn negative_transfer_fraction(&self) -> f64 {
[DQPROM L0172]         ratio_u64(self.negative_transfer_count, self.compared_states)
[DQPROM L0173]     }
[DQPROM L0174] }
[DQPROM L0175]
[DQPROM L0176] impl Default for DeltaQPolicyTransferReport {
[DQPROM L0177]     fn default() -> Self {
[DQPROM L0178]         Self::new()
[DQPROM L0179]     }
[DQPROM L0180] }
[DQPROM L0181]
[DQPROM L0182] impl DeltaQPolicyTransferResult {
[DQPROM L0183]     pub fn recommendation(&self) -> DeltaQPromotionRecommendation {
[DQPROM L0184]         if self.passed {
[DQPROM L0185]             DeltaQPromotionRecommendation::RequiresArenaConfirmation
[DQPROM L0186]         } else {
[DQPROM L0187]             DeltaQPromotionRecommendation::RejectAtOfflineGate
[DQPROM L0188]         }
[DQPROM L0189]     }
[DQPROM L0190] }
[DQPROM L0191]
[DQPROM L0192] impl DeltaQPromotionReport {
[DQPROM L0193]     pub fn new() -> Self {
[DQPROM L0194]         Self {
[DQPROM L0195]             eligible_states: 0,
[DQPROM L0196]             compared_states: 0,
[DQPROM L0197]             masked_entries: 0,
[DQPROM L0198]             supported_actions_sum: 0,
[DQPROM L0199]             candidate_top1_agreement_count: 0,
[DQPROM L0200]             baseline_top1_agreement_count: 0,
[DQPROM L0201]             candidate_high_gap_top1_count: 0,
[DQPROM L0202]             baseline_high_gap_top1_count: 0,
[DQPROM L0203]             high_gap_states: 0,
[DQPROM L0204]             candidate_regret_sum: 0.0,
[DQPROM L0205]             baseline_regret_sum: 0.0,
[DQPROM L0206]             decision_lift_sum: 0.0,
[DQPROM L0207]             negative_lift_count: 0,
[DQPROM L0208]             candidate_regret_beats_baseline_count: 0,
[DQPROM L0209]             candidate_top1_beats_baseline_count: 0,
[DQPROM L0210]         }
[DQPROM L0211]     }
[DQPROM L0212]
[DQPROM L0213]     pub fn merge(&mut self, other: &Self) {
[DQPROM L0214]         self.eligible_states += other.eligible_states;
[DQPROM L0215]         self.compared_states += other.compared_states;
[DQPROM L0216]         self.masked_entries += other.masked_entries;
[DQPROM L0217]         self.supported_actions_sum += other.supported_actions_sum;
[DQPROM L0218]         self.candidate_top1_agreement_count += other.candidate_top1_agreement_count;
[DQPROM L0219]         self.baseline_top1_agreement_count += other.baseline_top1_agreement_count;
[DQPROM L0220]         self.candidate_high_gap_top1_count += other.candidate_high_gap_top1_count;
```

## Artifact 13 — Validation loop and promotion snapshot collection
Artifact id: `validation-loop-surface`
Source label: VALID
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/validation.rs:40-220`
Why it matters: Shows how validation summaries and promotion metrics are actually collected in train-bin flows.

```rust
[VALID L0040] #[derive(Clone, Copy, Debug, serde::Serialize)]
[VALID L0041] pub(super) struct DeltaQPromotionSnapshot {
[VALID L0042]     pub(super) compared_states: u64,
[VALID L0043]     pub(super) candidate_top1_agreement: f64,
[VALID L0044]     pub(super) candidate_mean_regret: f64,
[VALID L0045]     pub(super) baseline_mean_regret: f64,
[VALID L0046]     pub(super) mean_decision_lift: f64,
[VALID L0047]     pub(super) negative_lift_fraction: f64,
[VALID L0048]     pub(super) regret_beats_baseline_rate: f64,
[VALID L0049]     pub(super) top1_beats_baseline_rate: f64,
[VALID L0050]     pub(super) passed: bool,
[VALID L0051] }
[VALID L0052]
[VALID L0053] impl DeltaQPromotionSnapshot {
[VALID L0054]     fn from_report(report: &DeltaQPromotionReport, result: &DeltaQPromotionResult) -> Self {
[VALID L0055]         Self {
[VALID L0056]             compared_states: report.compared_states,
[VALID L0057]             candidate_top1_agreement: report.candidate_top1_agreement(),
[VALID L0058]             candidate_mean_regret: report.candidate_mean_regret(),
[VALID L0059]             baseline_mean_regret: report.baseline_mean_regret(),
[VALID L0060]             mean_decision_lift: report.mean_decision_lift(),
[VALID L0061]             negative_lift_fraction: report.negative_lift_fraction(),
[VALID L0062]             regret_beats_baseline_rate: report.candidate_regret_beats_baseline_rate(),
[VALID L0063]             top1_beats_baseline_rate: report.candidate_top1_beats_baseline_rate(),
[VALID L0064]             passed: result.passed,
[VALID L0065]         }
[VALID L0066]     }
[VALID L0067] }
[VALID L0068]
[VALID L0069] #[derive(Clone, Copy, Debug, serde::Serialize)]
[VALID L0070] pub(super) struct DeltaQPolicyTransferSnapshot {
[VALID L0071]     pub(super) compared_states: u64,
[VALID L0072]     pub(super) candidate_policy_top1_to_teacher: f64,
[VALID L0073]     pub(super) baseline_policy_top1_to_teacher: f64,
[VALID L0074]     pub(super) candidate_policy_mean_teacher_regret: f64,
[VALID L0075]     pub(super) baseline_policy_mean_teacher_regret: f64,
[VALID L0076]     pub(super) candidate_beats_baseline_rate: f64,
[VALID L0077]     pub(super) negative_transfer_fraction: f64,
[VALID L0078] }
[VALID L0079]
[VALID L0080] impl DeltaQPolicyTransferSnapshot {
[VALID L0081]     fn from_report(report: &DeltaQPolicyTransferReport) -> Self {
[VALID L0082]         Self {
[VALID L0083]             compared_states: report.compared_states,
[VALID L0084]             candidate_policy_top1_to_teacher: report.candidate_policy_top1_to_teacher(),
[VALID L0085]             baseline_policy_top1_to_teacher: report.baseline_policy_top1_to_teacher(),
[VALID L0086]             candidate_policy_mean_teacher_regret: report.candidate_policy_mean_teacher_regret(),
[VALID L0087]             baseline_policy_mean_teacher_regret: report.baseline_policy_mean_teacher_regret(),
[VALID L0088]             candidate_beats_baseline_rate: report.candidate_beats_baseline_rate(),
[VALID L0089]             negative_transfer_fraction: report.negative_transfer_fraction(),
[VALID L0090]         }
[VALID L0091]     }
[VALID L0092] }
[VALID L0093]
[VALID L0094] #[derive(Clone)]
[VALID L0095] pub(super) struct ValidationSummary {
[VALID L0096]     pub(super) total_loss: f64,
[VALID L0097]     pub(super) policy_loss: f64,
[VALID L0098]     pub(super) agreement: f64,
[VALID L0099]     pub(super) samples: usize,
[VALID L0100]     pub(super) delta_q_promotion: Option<DeltaQPromotionReport>,
[VALID L0101]     pub(super) delta_q_promotion_result: Option<DeltaQPromotionResult>,
[VALID L0102]     pub(super) delta_q_promotion_snapshot: Option<DeltaQPromotionSnapshot>,
[VALID L0103]     pub(super) delta_q_policy_transfer: Option<DeltaQPolicyTransferReport>,
[VALID L0104]     pub(super) delta_q_policy_transfer_result: Option<DeltaQPolicyTransferResult>,
[VALID L0105]     pub(super) delta_q_policy_transfer_snapshot: Option<DeltaQPolicyTransferSnapshot>,
[VALID L0106] }
[VALID L0107]
[VALID L0108] pub(super) fn validation_batch_stats<B: Backend>(
[VALID L0109]     sample_count: usize,
[VALID L0110]     output: &HydraOutput<B>,
[VALID L0111]     batch: &hydra_train::data::sample::MjaiBatch<B>,
[VALID L0112]     targets: &HydraTargets<B>,
[VALID L0113]     breakdown: &hydra_train::training::losses::LossBreakdown<B>,
[VALID L0114]     total_loss: &Tensor<B, 1>,
[VALID L0115]     exit_cfg: &BcExitConfig,
[VALID L0116] ) -> BatchStats {
[VALID L0117]     let agreement = policy_agreement(
[VALID L0118]         output.policy_logits.clone(),
[VALID L0119]         targets.legal_mask.clone(),
[VALID L0120]         batch.actions.clone(),
[VALID L0121]     );
[VALID L0122]     let mut stats = batch_stats_from_breakdown(sample_count, agreement, &breakdown);
[VALID L0123]     if batch.exit_target.is_some() && batch.exit_mask.is_some() && exit_cfg.exit_weight > 0.0 {
[VALID L0124]         stats.total_loss = total_loss
[VALID L0125]             .clone()
[VALID L0126]             .into_scalar()
[VALID L0127]             .elem::<f64>();
[VALID L0128]     }
[VALID L0129]     stats
[VALID L0130] }
[VALID L0131]
[VALID L0132] pub(super) fn is_better_validation(
[VALID L0133]     summary: &ValidationSummary,
[VALID L0134]     best: Option<BestValidation>,
[VALID L0135] ) -> bool {
[VALID L0136]     match best {
[VALID L0137]         None => true,
[VALID L0138]         Some(best) => {
[VALID L0139]             summary.policy_loss < best.policy_loss
[VALID L0140]                 || ((summary.policy_loss - best.policy_loss).abs() <= f64::EPSILON
[VALID L0141]                     && summary.agreement > best.agreement)
[VALID L0142]         }
[VALID L0143]     }
[VALID L0144] }
[VALID L0145]
[VALID L0146] pub(super) fn run_validation(
[VALID L0147]     model: &HydraModel<TrainBackend>,
[VALID L0148]     context: ValidationContext<'_>,
[VALID L0149]     runtime: ValidationRuntime<'_>,
[VALID L0150] ) -> Result<ValidationSummary, String> {
[VALID L0151]     run_validation_with_policy_baseline(model, model, context, runtime)
[VALID L0152] }
[VALID L0153]
[VALID L0154] pub(super) fn run_validation_with_policy_baseline(
[VALID L0155]     model: &HydraModel<TrainBackend>,
[VALID L0156]     baseline_model: &HydraModel<TrainBackend>,
[VALID L0157]     context: ValidationContext<'_>,
[VALID L0158]     runtime: ValidationRuntime<'_>,
[VALID L0159] ) -> Result<ValidationSummary, String> {
[VALID L0160]     let ValidationContext {
[VALID L0161]         config,
[VALID L0162]         loader_config,
[VALID L0163]         manifest,
[VALID L0164]         cached_samples,
[VALID L0165]         device,
[VALID L0166]         loss_fn,
[VALID L0167]         exit_cfg,
[VALID L0168]     } = context;
[VALID L0169]     let ValidationRuntime {
[VALID L0170]         head_controller,
[VALID L0171]         progress,
[VALID L0172]     } = runtime;
[VALID L0173]     let model_valid = model.valid();
[VALID L0174]     let baseline_valid = baseline_model.valid();
[VALID L0175]     let validation_batch_size = validation_microbatch_size(config);
[VALID L0176]     let validation_sample_limit = validation_sample_limit(config);
[VALID L0177]     let mut stats = ScalarAverages::default();
[VALID L0178]     let mut total_samples = 0usize;
[VALID L0179]     let mut head_controller = head_controller;
[VALID L0180]     let mut delta_q_promotion = DeltaQPromotionReport::new();
[VALID L0181]     let mut delta_q_policy_transfer = DeltaQPolicyTransferReport::new();
[VALID L0182]     let mut saw_delta_q_targets = false;
[VALID L0183]
[VALID L0184]     let run_chunk = |capped_chunk: &[MjaiSample],
[VALID L0185]                      stats: &mut ScalarAverages,
[VALID L0186]                      total_samples: &mut usize,
[VALID L0187]                      head_controller: &mut Option<&mut HeadActivationController>,
[VALID L0188]                      delta_q_promotion: &mut DeltaQPromotionReport,
[VALID L0189]                      delta_q_policy_transfer: &mut DeltaQPolicyTransferReport,
[VALID L0190]                      saw_delta_q_targets: &mut bool|
[VALID L0191]      -> Result<(), String> {
[VALID L0192]         let Some((obs, batch)) =
[VALID L0193]             collate_batch_samples::<ValidBackend>(capped_chunk, false, device)
[VALID L0194]                 .map_err(|err| format!("validation collation failed: {err}"))?
[VALID L0195]         else {
[VALID L0196]             return Ok(());
[VALID L0197]         };
[VALID L0198]         let targets = batch.to_hydra_targets();
[VALID L0199]         let (active_loss_fn, warmup_heads) =
[VALID L0200]             gated_bc_context(head_controller.as_deref_mut(), loss_fn, &targets);
[VALID L0201]         let output = model_valid.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
[VALID L0202]         let breakdown = active_loss_fn.total_loss(&output, &targets);
[VALID L0203]         let total = bc_total_with_exit(&output, &batch, &targets, &active_loss_fn, exit_cfg);
[VALID L0204]         let batch_stats = validation_batch_stats(
[VALID L0205]             capped_chunk.len(),
[VALID L0206]             &output,
[VALID L0207]             &batch,
[VALID L0208]             &targets,
[VALID L0209]             &breakdown,
[VALID L0210]             &total,
[VALID L0211]             exit_cfg,
[VALID L0212]         );
[VALID L0213]         if targets.delta_q_target.is_some() && targets.delta_q_mask.is_some() {
[VALID L0214]             let baseline_output = baseline_valid.forward(obs);
[VALID L0215]             delta_q_promotion.merge(&collect_promotion_metrics_from_outputs(
[VALID L0216]                 &output, &targets, 0.75,
[VALID L0217]             ));
[VALID L0218]             delta_q_policy_transfer.merge(&collect_policy_transfer_metrics_from_policy_outputs(
[VALID L0219]                 output.policy_logits.clone(),
[VALID L0220]                 baseline_output.policy_logits.clone(),
```

## Artifact 14 — Eval results, benchmark gates, and paired arena decisions
Artifact id: `eval-and-arena-gates`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/eval.rs:34-240`
Why it matters: High-value artifact for turning pass-two recommendations into measurable promotion rules.

```rust
[EVAL L0034] #[derive(Debug, Clone)]
[EVAL L0035] pub struct EvalResult {
[EVAL L0036]     pub mean_placement: f32,
[EVAL L0037]     pub stable_dan: f32,
[EVAL L0038]     pub win_rate: f32,
[EVAL L0039]     pub deal_in_rate: f32,
[EVAL L0040]     pub tsumo_rate: f32,
[EVAL L0041] }
[EVAL L0042]
[EVAL L0043] impl EvalResult {
[EVAL L0044]     pub fn meets_target(&self, target_dan: f32) -> bool {
[EVAL L0045]         self.stable_dan >= target_dan
[EVAL L0046]     }
[EVAL L0047]
[EVAL L0048]     pub fn is_mortal_level(&self) -> bool {
[EVAL L0049]         self.stable_dan >= 8.0
[EVAL L0050]     }
[EVAL L0051]
[EVAL L0052]     pub fn is_tendan_plus(&self) -> bool {
[EVAL L0053]         self.stable_dan >= 10.0
[EVAL L0054]     }
[EVAL L0055]
[EVAL L0056]     pub fn summary(&self) -> String {
[EVAL L0057]         format!(
[EVAL L0058]             "placement={:.2} dan={:.1} win={:.1}% deal_in={:.1}%",
[EVAL L0059]             self.mean_placement,
[EVAL L0060]             self.stable_dan,
[EVAL L0061]             self.win_rate * 100.0,
[EVAL L0062]             self.deal_in_rate * 100.0
[EVAL L0063]         )
[EVAL L0064]     }
[EVAL L0065] }
[EVAL L0066]
[EVAL L0067] impl EvalResult {
[EVAL L0068]     pub fn from_mean_placement(mean_placement: f32) -> Self {
[EVAL L0069]         Self {
[EVAL L0070]             mean_placement,
[EVAL L0071]             stable_dan: compute_stable_dan(mean_placement),
[EVAL L0072]             ..Default::default()
[EVAL L0073]         }
[EVAL L0074]     }
[EVAL L0075] }
[EVAL L0076]
[EVAL L0077] impl Default for EvalResult {
[EVAL L0078]     fn default() -> Self {
[EVAL L0079]         Self {
[EVAL L0080]             mean_placement: 2.5,
[EVAL L0081]             stable_dan: 0.0,
[EVAL L0082]             win_rate: 0.0,
[EVAL L0083]             deal_in_rate: 0.0,
[EVAL L0084]             tsumo_rate: 0.0,
[EVAL L0085]         }
[EVAL L0086]     }
[EVAL L0087] }
[EVAL L0088]
[EVAL L0089] pub struct TrainingMetrics {
[EVAL L0090]     pub epoch: u32,
[EVAL L0091]     pub total_loss: f64,
[EVAL L0092]     pub policy_agreement: f64,
[EVAL L0093]     pub value_mse: f64,
[EVAL L0094]     pub games_completed: u64,
[EVAL L0095]     pub arena_mean_score: f32,
[EVAL L0096]     pub distill_kl: f32,
[EVAL L0097]     pub elo: f32,
[EVAL L0098] }
[EVAL L0099]
[EVAL L0100] impl Default for TrainingMetrics {
[EVAL L0101]     fn default() -> Self {
[EVAL L0102]         Self {
[EVAL L0103]             epoch: 0,
[EVAL L0104]             total_loss: 0.0,
[EVAL L0105]             policy_agreement: 0.0,
[EVAL L0106]             value_mse: 0.0,
[EVAL L0107]             games_completed: 0,
[EVAL L0108]             arena_mean_score: 0.0,
[EVAL L0109]             distill_kl: 0.0,
[EVAL L0110]             elo: 1500.0,
[EVAL L0111]         }
[EVAL L0112]     }
[EVAL L0113] }
[EVAL L0114]
[EVAL L0115] pub struct BenchmarkGates {
[EVAL L0116]     pub afbs_on_turn_ms: f32,
[EVAL L0117]     pub ct_smc_dp_ms: f32,
[EVAL L0118]     pub endgame_ms: f32,
[EVAL L0119]     pub self_play_games_per_sec: f32,
[EVAL L0120]     pub distill_kl_drift: f32,
[EVAL L0121] }
[EVAL L0122]
[EVAL L0123] impl BenchmarkGates {
[EVAL L0124]     pub fn summary(&self) -> String {
[EVAL L0125]         format!(
[EVAL L0126]             "afbs={:.0}ms smc={:.2}ms endgame={:.0}ms play={:.0}g/s kl={:.3}",
[EVAL L0127]             self.afbs_on_turn_ms,
[EVAL L0128]             self.ct_smc_dp_ms,
[EVAL L0129]             self.endgame_ms,
[EVAL L0130]             self.self_play_games_per_sec,
[EVAL L0131]             self.distill_kl_drift
[EVAL L0132]         )
[EVAL L0133]     }
[EVAL L0134]     pub fn passes(&self) -> bool {
[EVAL L0135]         self.afbs_on_turn_ms < 150.0
[EVAL L0136]             && self.ct_smc_dp_ms < 1.0
[EVAL L0137]             && self.endgame_ms < 100.0
[EVAL L0138]             && self.self_play_games_per_sec > 20.0
[EVAL L0139]             && self.distill_kl_drift < 0.1
[EVAL L0140]     }
[EVAL L0141] }
[EVAL L0142]
[EVAL L0143] impl TrainingMetrics {
[EVAL L0144]     pub fn is_improving(&self, prev_loss: f64) -> bool {
[EVAL L0145]         self.total_loss < prev_loss
[EVAL L0146]     }
[EVAL L0147]
[EVAL L0148]     pub fn summary(&self) -> String {
[EVAL L0149]         format!(
[EVAL L0150]             "epoch={} loss={:.4} agree={:.2}% games={} elo={:.0}",
[EVAL L0151]             self.epoch,
[EVAL L0152]             self.total_loss,
[EVAL L0153]             self.policy_agreement * 100.0,
[EVAL L0154]             self.games_completed,
[EVAL L0155]             self.elo
[EVAL L0156]         )
[EVAL L0157]     }
[EVAL L0158] }
[EVAL L0159]
[EVAL L0160] #[derive(Config, Debug)]
[EVAL L0161] pub struct PairedArenaEvalConfig {
[EVAL L0162]     #[config(default = "10000")]
[EVAL L0163]     pub min_games: usize,
[EVAL L0164]     #[config(default = "42")]
[EVAL L0165]     pub seed: u64,
[EVAL L0166]     #[config(default = "0.025")]
[EVAL L0167]     pub max_mean_placement_regression: f32,
[EVAL L0168]     #[config(default = "0.0")]
[EVAL L0169]     pub strong_promotion_mean_placement_target: f32,
[EVAL L0170]     #[config(default = "true")]
[EVAL L0171]     pub same_seeds: bool,
[EVAL L0172]     #[config(default = "true")]
[EVAL L0173]     pub same_seat_rotation_schedule: bool,
[EVAL L0174]     #[config(default = "true")]
[EVAL L0175]     pub same_search_budget: bool,
[EVAL L0176]     #[config(default = "true")]
[EVAL L0177]     pub same_temperature: bool,
[EVAL L0178]     #[config(default = "true")]
[EVAL L0179]     pub same_frozen_opponent_pool: bool,
[EVAL L0180] }
[EVAL L0181]
[EVAL L0182] impl PairedArenaEvalConfig {
[EVAL L0183]     pub fn validate(&self) -> Result<(), &'static str> {
[EVAL L0184]         if self.min_games == 0 {
[EVAL L0185]             return Err("min_games must be > 0");
[EVAL L0186]         }
[EVAL L0187]         if self.max_mean_placement_regression < 0.0 {
[EVAL L0188]             return Err("max_mean_placement_regression must be >= 0");
[EVAL L0189]         }
[EVAL L0190]         Ok(())
[EVAL L0191]     }
[EVAL L0192]
[EVAL L0193]     pub fn summary(&self) -> String {
[EVAL L0194]         format!(
[EVAL L0195]             "paired_arena(min_games={}, seed={}, max_reg={:.3}, strong_target={:.3}, same_seeds={}, same_rotation={}, same_budget={}, same_temp={}, frozen_pool={})",
[EVAL L0196]             self.min_games,
[EVAL L0197]             self.seed,
[EVAL L0198]             self.max_mean_placement_regression,
[EVAL L0199]             self.strong_promotion_mean_placement_target,
[EVAL L0200]             self.same_seeds,
[EVAL L0201]             self.same_seat_rotation_schedule,
[EVAL L0202]             self.same_search_budget,
[EVAL L0203]             self.same_temperature,
[EVAL L0204]             self.same_frozen_opponent_pool,
[EVAL L0205]         )
[EVAL L0206]     }
[EVAL L0207] }
[EVAL L0208]
[EVAL L0209] #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
[EVAL L0210] pub enum ArenaPromotionDecision {
[EVAL L0211]     Reject,
[EVAL L0212]     NonRegressionOnly,
[EVAL L0213]     StrongPromotion,
[EVAL L0214] }
[EVAL L0215]
[EVAL L0216] impl ArenaPromotionDecision {
[EVAL L0217]     pub fn summary(self) -> &'static str {
[EVAL L0218]         match self {
[EVAL L0219]             Self::Reject => "reject",
[EVAL L0220]             Self::NonRegressionOnly => "non_regression_only",
[EVAL L0221]             Self::StrongPromotion => "strong_promotion",
[EVAL L0222]         }
[EVAL L0223]     }
[EVAL L0224] }
[EVAL L0225]
[EVAL L0226] #[derive(Debug, Clone)]
[EVAL L0227] pub struct PairedArenaEvalResult {
[EVAL L0228]     pub candidate_mean_placement: f32,
[EVAL L0229]     pub baseline_mean_placement: f32,
[EVAL L0230]     pub delta_mean_placement: f32,
[EVAL L0231]     pub candidate_stable_dan: f32,
[EVAL L0232]     pub baseline_stable_dan: f32,
[EVAL L0233]     pub delta_stable_dan: f32,
[EVAL L0234]     pub upper_confidence_bound_mean_placement: f32,
[EVAL L0235]     pub compared_games: usize,
[EVAL L0236] }
[EVAL L0237]
[EVAL L0238] impl PairedArenaEvalResult {
[EVAL L0239]     pub fn passes_non_regression(&self, config: &PairedArenaEvalConfig) -> bool {
[EVAL L0240]         self.upper_confidence_bound_mean_placement <= config.max_mean_placement_regression
```

## Artifact 15 — Existing label-producer validation harness template
Artifact id: `exit-validation-template`
Source label: EXITVAL
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:19-240`
Why it matters: A strong local template for what a hidden-world teacher validation harness could look like: emission, coverage, KL, and thresholded pass/fail logic.

```rust
[EXITVAL L0019] /// Aggregated metrics from a shadow ExIt validation run.
[EXITVAL L0020] ///
[EXITVAL L0021] /// Each field corresponds to one criterion from the Agent 22/9/16
[EXITVAL L0022] /// blueprint. The harness collects these by running the live producer
[EXITVAL L0023] /// on self-play states without using the labels for training.
[EXITVAL L0024] #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
[EXITVAL L0025] pub struct ExitValidationReport {
[EXITVAL L0026]     /// Total decision states examined.
[EXITVAL L0027]     pub total_states: u64,
[EXITVAL L0028]     /// States that passed the compatible-discard-only gate.
[EXITVAL L0029]     pub compatible_discard_states: u64,
[EXITVAL L0030]     /// States that passed the hard-state gate (top2_policy_gap < 0.10).
[EXITVAL L0031]     pub hard_states: u64,
[EXITVAL L0032]     /// States where the producer emitted a real label (not None).
[EXITVAL L0033]     pub labels_emitted: u64,
[EXITVAL L0034]     /// States where the producer returned None (any gate failed).
[EXITVAL L0035]     pub labels_rejected: u64,
[EXITVAL L0036]     /// Rejected because state was not a compatible discard state.
[EXITVAL L0037]     pub rejected_incompatible_state: u64,
[EXITVAL L0038]     /// Rejected because fewer than 2 legal discards.
[EXITVAL L0039]     pub rejected_too_few_discards: u64,
[EXITVAL L0040]     /// Rejected because state was not hard (top-2 gap >= threshold).
[EXITVAL L0041]     pub rejected_not_hard_state: u64,
[EXITVAL L0042]     /// Rejected by child observation failure.
[EXITVAL L0043]     pub rejected_child_obs_failure: u64,
[EXITVAL L0044]     /// Rejected by coverage gate (< 0.60).
[EXITVAL L0045]     pub rejected_low_coverage: u64,
[EXITVAL L0046]     /// Rejected by KL safety valve.
[EXITVAL L0047]     pub rejected_kl_safety: u64,
[EXITVAL L0048]     /// Rejected by other or currently un-attributed gates.
[EXITVAL L0049]     pub rejected_other: u64,
[EXITVAL L0050]     /// Sum of coverage values across emitted labels.
[EXITVAL L0051]     pub coverage_sum: f64,
[EXITVAL L0052]     /// Sum of supported action counts across emitted labels.
[EXITVAL L0053]     pub supported_actions_sum: u64,
[EXITVAL L0054]     /// Sum of root visit counts across emitted labels.
[EXITVAL L0055]     pub root_visits_sum: u64,
[EXITVAL L0056]     /// Count of emitted labels where top-1 action matches base policy top-1.
[EXITVAL L0057]     pub top1_agreement_count: u64,
[EXITVAL L0058]     /// Sum of KL(base || exit) across emitted labels.
[EXITVAL L0059]     pub kl_sum: f64,
[EXITVAL L0060] }
[EXITVAL L0061]
[EXITVAL L0062] impl ExitValidationReport {
[EXITVAL L0063]     /// Creates an empty report with all counters at zero.
[EXITVAL L0064]     pub fn new() -> Self {
[EXITVAL L0065]         Self {
[EXITVAL L0066]             total_states: 0,
[EXITVAL L0067]             compatible_discard_states: 0,
[EXITVAL L0068]             hard_states: 0,
[EXITVAL L0069]             labels_emitted: 0,
[EXITVAL L0070]             labels_rejected: 0,
[EXITVAL L0071]             rejected_incompatible_state: 0,
[EXITVAL L0072]             rejected_too_few_discards: 0,
[EXITVAL L0073]             rejected_not_hard_state: 0,
[EXITVAL L0074]             rejected_child_obs_failure: 0,
[EXITVAL L0075]             rejected_low_coverage: 0,
[EXITVAL L0076]             rejected_kl_safety: 0,
[EXITVAL L0077]             rejected_other: 0,
[EXITVAL L0078]             coverage_sum: 0.0,
[EXITVAL L0079]             supported_actions_sum: 0,
[EXITVAL L0080]             root_visits_sum: 0,
[EXITVAL L0081]             top1_agreement_count: 0,
[EXITVAL L0082]             kl_sum: 0.0,
[EXITVAL L0083]         }
[EXITVAL L0084]     }
[EXITVAL L0085]
[EXITVAL L0086]     /// Merges another report into this one.
[EXITVAL L0087]     pub fn merge(&mut self, other: &ExitValidationReport) {
[EXITVAL L0088]         self.total_states += other.total_states;
[EXITVAL L0089]         self.compatible_discard_states += other.compatible_discard_states;
[EXITVAL L0090]         self.hard_states += other.hard_states;
[EXITVAL L0091]         self.labels_emitted += other.labels_emitted;
[EXITVAL L0092]         self.labels_rejected += other.labels_rejected;
[EXITVAL L0093]         self.rejected_incompatible_state += other.rejected_incompatible_state;
[EXITVAL L0094]         self.rejected_too_few_discards += other.rejected_too_few_discards;
[EXITVAL L0095]         self.rejected_not_hard_state += other.rejected_not_hard_state;
[EXITVAL L0096]         self.rejected_child_obs_failure += other.rejected_child_obs_failure;
[EXITVAL L0097]         self.rejected_low_coverage += other.rejected_low_coverage;
[EXITVAL L0098]         self.rejected_kl_safety += other.rejected_kl_safety;
[EXITVAL L0099]         self.rejected_other += other.rejected_other;
[EXITVAL L0100]         self.coverage_sum += other.coverage_sum;
[EXITVAL L0101]         self.supported_actions_sum += other.supported_actions_sum;
[EXITVAL L0102]         self.root_visits_sum += other.root_visits_sum;
[EXITVAL L0103]         self.top1_agreement_count += other.top1_agreement_count;
[EXITVAL L0104]         self.kl_sum += other.kl_sum;
[EXITVAL L0105]     }
[EXITVAL L0106]
[EXITVAL L0107]     /// Returns the label emission rate.
[EXITVAL L0108]     pub fn emission_rate(&self) -> f64 {
[EXITVAL L0109]         ratio_u64(self.labels_emitted, self.total_states)
[EXITVAL L0110]     }
[EXITVAL L0111]
[EXITVAL L0112]     /// Returns the hard-state rate.
[EXITVAL L0113]     pub fn hard_state_rate(&self) -> f64 {
[EXITVAL L0114]         ratio_u64(self.hard_states, self.total_states)
[EXITVAL L0115]     }
[EXITVAL L0116]
[EXITVAL L0117]     /// Returns the mean coverage across emitted labels.
[EXITVAL L0118]     pub fn mean_coverage(&self) -> f64 {
[EXITVAL L0119]         ratio_f64(self.coverage_sum, self.labels_emitted)
[EXITVAL L0120]     }
[EXITVAL L0121]
[EXITVAL L0122]     /// Returns the mean supported actions across emitted labels.
[EXITVAL L0123]     pub fn mean_supported_actions(&self) -> f64 {
[EXITVAL L0124]         ratio_u64(self.supported_actions_sum, self.labels_emitted)
[EXITVAL L0125]     }
[EXITVAL L0126]
[EXITVAL L0127]     /// Returns the mean root visits across emitted labels.
[EXITVAL L0128]     pub fn mean_root_visits(&self) -> f64 {
[EXITVAL L0129]         ratio_u64(self.root_visits_sum, self.labels_emitted)
[EXITVAL L0130]     }
[EXITVAL L0131]
[EXITVAL L0132]     /// Returns the top-1 action agreement rate.
[EXITVAL L0133]     pub fn top1_agreement_rate(&self) -> f64 {
[EXITVAL L0134]         ratio_u64(self.top1_agreement_count, self.labels_emitted)
[EXITVAL L0135]     }
[EXITVAL L0136]
[EXITVAL L0137]     /// Returns the mean KL divergence between base policy and ExIt labels.
[EXITVAL L0138]     pub fn mean_kl(&self) -> f64 {
[EXITVAL L0139]         ratio_f64(self.kl_sum, self.labels_emitted)
[EXITVAL L0140]     }
[EXITVAL L0141] }
[EXITVAL L0142]
[EXITVAL L0143] impl Default for ExitValidationReport {
[EXITVAL L0144]     fn default() -> Self {
[EXITVAL L0145]         Self::new()
[EXITVAL L0146]     }
[EXITVAL L0147] }
[EXITVAL L0148]
[EXITVAL L0149] impl fmt::Display for ExitValidationReport {
[EXITVAL L0150]     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
[EXITVAL L0151]         writeln!(f, "=== ExIt Validation Report ===")?;
[EXITVAL L0152]         writeln!(f, "States examined:       {}", self.total_states)?;
[EXITVAL L0153]         writeln!(
[EXITVAL L0154]             f,
[EXITVAL L0155]             "Compatible discard:    {} ({:.1}%)",
[EXITVAL L0156]             self.compatible_discard_states,
[EXITVAL L0157]             ratio_u64(self.compatible_discard_states, self.total_states) * 100.0
[EXITVAL L0158]         )?;
[EXITVAL L0159]         writeln!(
[EXITVAL L0160]             f,
[EXITVAL L0161]             "Hard states:           {} ({:.1}%)",
[EXITVAL L0162]             self.hard_states,
[EXITVAL L0163]             self.hard_state_rate() * 100.0
[EXITVAL L0164]         )?;
[EXITVAL L0165]         writeln!(
[EXITVAL L0166]             f,
[EXITVAL L0167]             "Labels emitted:        {} ({:.2}%)",
[EXITVAL L0168]             self.labels_emitted,
[EXITVAL L0169]             self.emission_rate() * 100.0
[EXITVAL L0170]         )?;
[EXITVAL L0171]         writeln!(f, "Labels rejected:       {}", self.labels_rejected)?;
[EXITVAL L0172]         writeln!(f, "--- Rejection breakdown ---")?;
[EXITVAL L0173]         writeln!(
[EXITVAL L0174]             f,
[EXITVAL L0175]             "  Incompatible state:  {}",
[EXITVAL L0176]             self.rejected_incompatible_state
[EXITVAL L0177]         )?;
[EXITVAL L0178]         writeln!(
[EXITVAL L0179]             f,
[EXITVAL L0180]             "  Too few discards:    {}",
[EXITVAL L0181]             self.rejected_too_few_discards
[EXITVAL L0182]         )?;
[EXITVAL L0183]         writeln!(f, "  Not hard state:      {}", self.rejected_not_hard_state)?;
[EXITVAL L0184]         writeln!(
[EXITVAL L0185]             f,
[EXITVAL L0186]             "  Child obs failure:   {}",
[EXITVAL L0187]             self.rejected_child_obs_failure
[EXITVAL L0188]         )?;
[EXITVAL L0189]         writeln!(f, "  Low coverage:        {}", self.rejected_low_coverage)?;
[EXITVAL L0190]         writeln!(f, "  KL safety valve:     {}", self.rejected_kl_safety)?;
[EXITVAL L0191]         writeln!(f, "  Other:               {}", self.rejected_other)?;
[EXITVAL L0192]         writeln!(f, "--- Label quality ---")?;
[EXITVAL L0193]         writeln!(f, "  Mean coverage:       {:.3}", self.mean_coverage())?;
[EXITVAL L0194]         writeln!(
[EXITVAL L0195]             f,
[EXITVAL L0196]             "  Mean supported acts: {:.1}",
[EXITVAL L0197]             self.mean_supported_actions()
[EXITVAL L0198]         )?;
[EXITVAL L0199]         writeln!(f, "  Mean root visits:    {:.0}", self.mean_root_visits())?;
[EXITVAL L0200]         writeln!(
[EXITVAL L0201]             f,
[EXITVAL L0202]             "  Top-1 agreement:     {:.1}%",
[EXITVAL L0203]             self.top1_agreement_rate() * 100.0
[EXITVAL L0204]         )?;
[EXITVAL L0205]         writeln!(f, "  Mean KL:             {:.4}", self.mean_kl())?;
[EXITVAL L0206]         Ok(())
[EXITVAL L0207]     }
[EXITVAL L0208] }
[EXITVAL L0209]
[EXITVAL L0210] /// Thresholds for the ExIt validation pass/fail decision.
[EXITVAL L0211] ///
[EXITVAL L0212] /// Derived from Agent 22 + Agent 9 + Agent 16 blueprints. These are the
[EXITVAL L0213] /// minimum requirements before the producer can be enabled.
[EXITVAL L0214] #[derive(Debug, Clone)]
[EXITVAL L0215] pub struct ExitValidationThresholds {
[EXITVAL L0216]     /// Minimum fraction of total states that must emit labels.
[EXITVAL L0217]     pub min_emission_rate: f64,
[EXITVAL L0218]     /// Minimum mean coverage across emitted labels.
[EXITVAL L0219]     pub min_mean_coverage: f64,
[EXITVAL L0220]     /// Minimum mean supported actions per emitted label.
[EXITVAL L0221]     pub min_mean_supported_actions: f64,
[EXITVAL L0222]     /// Maximum mean KL divergence between base policy and ExIt labels.
[EXITVAL L0223]     pub max_mean_kl: f64,
[EXITVAL L0224]     /// Minimum top-1 agreement rate.
[EXITVAL L0225]     pub min_top1_agreement: f64,
[EXITVAL L0226]     /// Minimum total states examined for the report to be meaningful.
[EXITVAL L0227]     pub min_sample_size: u64,
[EXITVAL L0228] }
[EXITVAL L0229]
[EXITVAL L0230] impl Default for ExitValidationThresholds {
[EXITVAL L0231]     fn default() -> Self {
[EXITVAL L0232]         Self {
[EXITVAL L0233]             min_emission_rate: 0.01,
[EXITVAL L0234]             min_mean_coverage: 0.70,
[EXITVAL L0235]             min_mean_supported_actions: 3.0,
[EXITVAL L0236]             max_mean_kl: 0.05,
[EXITVAL L0237]             min_top1_agreement: 0.95,
[EXITVAL L0238]             min_sample_size: 1000,
[EXITVAL L0239]         }
[EXITVAL L0240]     }
```

## Artifact 16 — Golden tests, cross-validation, and training smoke tests
Artifact id: `testing-slices`
Source label: TEST
Type: `file_range`
Source: `research/design/TESTING.md:98-260`
Why it matters: Pass-two falsification surface: encoder goldens, roundtrip/cross-validation, and training-stack smoke tests.

```md
[TEST L0098] ## Observation Encoding Correctness
[TEST L0099]
[TEST L0100] ### Baseline-Prefix Verification (Channels 0-84)
[TEST L0101]
[TEST L0102] Each of the first 85 channels must encode the baseline public+safety prefix exactly as described in `docs/GAME_ENGINE.md`. Build a test harness that constructs known game states and verifies the baseline prefix element by element, while keeping the full live tensor shape at `192x34`.
[TEST L0103]
[TEST L0104] **Channel-by-channel tests:**
[TEST L0105]
[TEST L0106] | Channel Range | Verification |
[TEST L0107] |---------------|-------------|
[TEST L0108] | 0-3 (hand thermometer) | Set hand to [1m, 1m, 1m, 2m], verify ch0-2 at index 0 are 1.0, ch3 is 0.0 |
[TEST L0109] | 8 (drawn tile) | Draw 5p, verify only index 12 is 1.0, all others 0.0 |
[TEST L0110] | 9-10 (shanten masks) | Construct a tenpai hand, verify keep-shanten and next-shanten masks match `xiangting` output |
[TEST L0111] | 11-22 (discards) | Discard 3 tiles with known tedashi/tsumogiri flags, verify encoding |
[TEST L0112] | 35-42 (dora/aka) | Set 2 dora indicators, verify thermometer encoding; check aka planes for red 5s |
[TEST L0113] | 42-45 (riichi status) | Declare riichi for player 2, verify only ch43 is all-1.0 |
[TEST L0114] | 46-49 (scores) | Set scores to [25000, 30000, 20000, 25000], verify normalization by 100000 |
[TEST L0115] | 62-70 (genbutsu) | Opponent declares riichi then player discards 7s → 7s is genbutsu for that opponent |
[TEST L0116] | 71-79 (suji) | Opponent discards 4m → verify 1m and 7m have suji safety > 0 |
[TEST L0117] | 80-81 (kabe/one-chance) | All 4 copies of 3p visible → verify kabe flag at index 11 |
[TEST L0118]
[TEST L0119] ### Known-State Golden Tests
[TEST L0120]
[TEST L0121] Maintain a set of 20+ hand-crafted game states with pre-computed expected tensors, serialized as `.npz` files. These serve as regression tests — any encoder change that alters golden outputs must be reviewed and the golden files explicitly regenerated.
[TEST L0122]
[TEST L0123] ### Roundtrip Tests
[TEST L0124]
[TEST L0125] Construct a game state programmatically → encode to the live `192x34` tensor → verify expected values. The encoder is one-way (state → tensor), so "roundtrip" means verifying that the tensor faithfully represents the state, not that the state can be recovered from the tensor.
[TEST L0126]
[TEST L0127] ---
[TEST L0128]
[TEST L0129] ## MJAI Parsing
[TEST L0130]
[TEST L0131] ### Log Reconstruction
[TEST L0132]
[TEST L0133] Parse real Tenhou and Majsoul game logs in MJAI format, replay the events through the game engine, and verify that the reconstructed game state matches the log's recorded outcomes (final scores, winner, winning hand, yaku).
[TEST L0134]
[TEST L0135] Current status note: the live replay path now has explicit regression coverage for replay round-reset semantics and kan replay legality matching, and a full Tenhou Houou 2025 audit (`178,897` MJAI files) completed with `0` skips after those fixes. Remaining replay failures should be treated as true file/data faults unless a new regression reproducer says otherwise.
[TEST L0136]
[TEST L0137] **Minimum test corpus:**
[TEST L0138]
[TEST L0139] - 100 randomly sampled Tenhou Houou games
[TEST L0140] - 100 randomly sampled Majsoul Throne games
[TEST L0141] - 50 games containing special events (see edge cases below)
[TEST L0142]
[TEST L0143] ### Edge Cases
[TEST L0144]
[TEST L0145] | Scenario | What to Verify |
[TEST L0146] |----------|---------------|
[TEST L0147] | Multiple ron (double/triple) | Both/all winners detected, correct payment split |
[TEST L0148] | Chankan | Ron on an added kan, correct yaku assignment |
[TEST L0149] | Rinshan tsumo | Win from dead wall draw after kan, rinshan kaihou yaku applied |
[TEST L0150] | Double riichi | Riichi declared on first turn (no prior calls), double riichi yaku applied |
[TEST L0151] | Ippatsu with intervening call | Opponent calls between riichi and next draw, ippatsu denied |
[TEST L0152] | Haitei/Houtei | Win on last draw/discard, correct yaku applied |
[TEST L0153]
[TEST L0154] ### Event Roundtrip
[TEST L0155]
[TEST L0156] Generate a game programmatically → serialize to MJAI events → parse events back through the engine → verify final state matches. This catches serialization/deserialization asymmetries.
[TEST L0157]
[TEST L0158] ---
[TEST L0159]
[TEST L0160] ## Suit Permutation Augmentation
[TEST L0161]
[TEST L0162] ### Validity
[TEST L0163]
[TEST L0164] All 6 permutations of `[manzu, pinzu, souzu]` must produce valid game states. For each permutation:
[TEST L0165]
[TEST L0166] 1. Apply permutation to a game's MJAI event stream
[TEST L0167] 2. Replay permuted events through the engine
[TEST L0168] 3. Verify: no illegal states, no assertion failures, game reaches the same terminal condition
[TEST L0169]
[TEST L0170] ### Aka-Dora Roundtrip
[TEST L0171]
[TEST L0172] The `deaka → permute → re_akaize` chain must preserve aka-dora identity:
[TEST L0173]
[TEST L0174] - Red 5m permuted to pinzu → becomes red 5p (not normal 5p)
[TEST L0175] - Red 5p permuted to souzu → becomes red 5s
[TEST L0176] - Identity permutation [m→m, p→p, s→s] produces bit-identical output
[TEST L0177]
[TEST L0178] ### Score Invariance
[TEST L0179]
[TEST L0180] The same game played under all 6 permutations must produce identical final scores. Suits are strategically interchangeable — no yaku depends on suit identity (unlike honor tiles).
[TEST L0181]
[TEST L0182] ### Identity Permutation
[TEST L0183]
[TEST L0184] Permutation [0, 1, 2] (identity) must produce output identical to no permutation. Byte-for-byte comparison of encoded observations.
[TEST L0185]
[TEST L0186] ---
[TEST L0187]
[TEST L0188] ## Property-Based Testing
[TEST L0189]
[TEST L0190] Use the `proptest` crate for Rust engine invariants. Property-based tests generate thousands of random inputs and check that invariants hold for all of them.
[TEST L0191]
[TEST L0192] ### Core Invariants
[TEST L0193]
[TEST L0194] | Property | Invariant |
[TEST L0195] |----------|-----------|
[TEST L0196] | Legal action mask | At least 1 legal action when game is not terminal |
[TEST L0197] | Score conservation | Sum of all 4 player scores equals 100,000 at all times (before riichi deposit adjustments, accounting for kyotaku) |
[TEST L0198] | Shanten bounds | Shanten is non-negative and at most 6 for any valid hand |
[TEST L0199] | Tile count bounds | No tile type appears more than 4 times across all visible locations |
[TEST L0200] | Total tile count | Exactly 136 tiles exist across wall, hands, discards, melds, and dead wall |
[TEST L0201] | State machine validity | No legal action sequence from a valid state produces an invalid state |
[TEST L0202] | Terminal detection | A terminal state has an empty legal action set |
[TEST L0203]
[TEST L0204] ### Strategy
[TEST L0205]
[TEST L0206] 1. Generate a random valid initial game state (deal 13 tiles to each player from a shuffled 136-tile wall)
[TEST L0207] 2. At each step, choose a random legal action from the legal action mask
[TEST L0208] 3. Apply the action, check all invariants
[TEST L0209] 4. Repeat until terminal or 500 actions (capped to prevent infinite loops in degenerate cases)
[TEST L0210] 5. Run 10,000+ such random games per CI run
[TEST L0211]
[TEST L0212] ---
[TEST L0213]
[TEST L0214] ## Cross-Validation
[TEST L0215]
[TEST L0216] ### Shanten
[TEST L0217]
[TEST L0218] Compare the Rust `xiangting` crate's shanten calculation against an independent implementation on N=100,000 randomly generated hands.
[TEST L0219]
[TEST L0220] **Methodology:**
[TEST L0221]
[TEST L0222] 1. Generate 100K random 13-tile hands (sampling without replacement from 136 tiles)
[TEST L0223] 2. Compute shanten using `xiangting` (Rust)
[TEST L0224] 3. Compute shanten using an independent algorithm (e.g., lookup table or brute-force)
[TEST L0225] 4. Any disagreement is a bug — log the hand tiles and both results
[TEST L0226] 5. Include edge cases: complete hands (shanten = -1), kokushi tenpai, chiitoitsu tenpai
[TEST L0227]
[TEST L0228] ### Scoring
[TEST L0229]
[TEST L0230] Cross-validate Rust scoring against the `mahjong` Python library on 100K randomly constructed winning hands.
[TEST L0231]
[TEST L0232] **Methodology:**
[TEST L0233]
[TEST L0234] 1. Generate random winning hands (tenpai hands + a completing tile)
[TEST L0235] 2. Assign random game context (round wind, seat wind, dora, riichi, tsumo/ron)
[TEST L0236] 3. Compute yaku/han/fu/score in both Rust and Python
[TEST L0237] 4. Diff results — any mismatch is logged with full context for debugging
[TEST L0238] 5. Special attention to fu calculation edge cases (open pinfu, closed tsumo, etc.)
[TEST L0239]
[TEST L0240] ---
[TEST L0241]
[TEST L0242] ## Burn Training Stack
[TEST L0243]
[TEST L0244] ### Model Smoke Tests
[TEST L0245]
[TEST L0246] - Forward pass with random input `[1, 192, 34]` produces the output shapes asserted by `hydra-train/src/model.rs` for the current `ActorNet` / `LearnerNet`
[TEST L0247] - Legal action masking: masked logits are negative infinity, softmax produces zero probability for illegal actions
[TEST L0248] - Inference: run forward pass through burn-tch backend, verify output matches expected within tolerance (atol=1e-5)
[TEST L0249]
[TEST L0250] ### Loss Function Tests
[TEST L0251]
[TEST L0252] - Policy CE loss with known logits and labels — verify against hand-computed value
[TEST L0253] - GRP 24-way CE loss sums to correct value for a known permutation distribution
[TEST L0254] - Focal BCE loss with γ=2.0 produces lower loss for high-confidence correct predictions than standard BCE
[TEST L0255] - Composite loss with known component values → verify weighted sum matches expected total
[TEST L0256]
[TEST L0257] ### Data Pipeline Tests
[TEST L0258]
[TEST L0259] - Burn DataLoaderBuilder yields batches of correct shape `[2048, 192, 34]`
[TEST L0260] - 3-level shuffle produces different orderings across epochs (statistical test: correlation < 0.1)
```

## Artifact 17 — Belief activation gates and required future closure
Artifact id: `answer15-belief-gates`
Source label: A15
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md:452-502`
Why it matters: Compact archive slice on the minimum conditions for honest belief activation, useful for pass-two gate design.

```md
[A15 L0452] 3. **Belief conservation test:** the teacher must represent `\bar B_t \in \mathcal U(r_t,s_t)` using true public row sums and true public zone sizes. Equal four-way splitting of total hidden mass fails this. This kills current Stage A belief supervision. ([GitHub][4])
[A15 L0453]
[A15 L0454] 4. **Public-history likelihood test:** a belief teacher must condition on the public action history, not just on static remaining counts. Uniform-kernel projection with no event-likelihood update fails this. This kills current Stage A as a doctrinal public teacher. ([GitHub][4])
[A15 L0455]
[A15 L0456] 5. **Identifiability test:** if the supervised object is not unique up to representation gauge or component permutation, do not supervise it directly. This kills raw external-field targets and non-canonical mixture weights. ([GitHub][7])
[A15 L0457]
[A15 L0458] 6. **Component-canonicalization test:** if runtime and teacher can order mixture components differently, direct component-wise supervision is fake. The bridge already rank-sorts components by weight for runtime feature planes, so future component supervision would need explicit canonicalization. Until then, keep it absent. ([GitHub][12])
[A15 L0459]
[A15 L0460] 7. **Mask completeness test:** sparse or partial labels require explicit masks at the same granularity as support. This kills current `delta_q_target` activation because there is no `[B,46]` mask and current bridge support is discard-only. It also blocks opponent-hand-type partial labeling because current loss uses only sample-level oracle masking. ([GitHub][3])
[A15 L0461]
[A15 L0462] 8. **Loss-object fit test:** the loss must match the target semantics. BCE on projected transportation-mass cells is not obviously the right loss; CE on arbitrary mixture IDs is not valid without canonical decomposition. This kills current belief/mix activation. ([GitHub][3])
[A15 L0463]
[A15 L0464] 9. **Runtime reconstructibility test:** would the deployed policy ever be able to infer the supervised object from public information, perhaps with compute? If not, reject it. This kills realized hidden allocation and exact yaku potential as student targets. ([GitHub][4])
[A15 L0465]
[A15 L0466] 10. **Sequencing / compute test:** if activation requires turning full public-belief search into the immediate mainline, reject it for now. The reconciliation memo explicitly says not to do that yet. This kills any “just make CT-SMC belief search the whole next tranche” answer. ([GitHub][1])
[A15 L0467]
[A15 L0468] Ideas that do **not** survive this pass:
[A15 L0469]
[A15 L0470] * **Current Stage A belief-field supervision:** fails checks 3, 4, 5, 8.
[A15 L0471] * **Current Stage A mixture-weight supervision:** fails checks 5, 6, 8.
[A15 L0472] * **Using current bridge `ΔQ` planes as dense `[B,46]` labels:** fails check 7.
[A15 L0473] * **Eventual winning-yaku / exact-yaku hand-type labels:** fail checks 1, 2, 9, and are especially weak in a 4-player general-sum setting because they collapse latent opponent state into a winner-conditioned future outcome. ([GitHub][6])
[A15 L0474]
[A15 L0475] Ideas that **do** survive:
[A15 L0476]
[A15 L0477] * **`safety_residual_target` as narrow replay-derived masked auxiliary**
[A15 L0478] * **later AFBS-derived `ΔQ` and ExIt, but only after trust-gated search-label plumbing exists**
[A15 L0479] * **public-posterior belief supervision as a semantic target object, not as a currently activatable code path through Stage A**
[A15 L0480]
[A15 L0481] ## 8. Final recommendation: what Hydra should activate now, later, or never
[A15 L0482]
[A15 L0483] ### Activate now
[A15 L0484]
[A15 L0485] **Activate only `safety_residual_target`, and activate it narrowly.** It is the only advanced target that is already end-to-end concrete in current master: replay builder, explicit `[B,46]` tensor, explicit `[B,46]` action mask, existing head, and masked loss. Its target semantics are simple and local:
[A15 L0486]
[A15 L0487] [
[A15 L0488] s_t^*(a)
[A15 L0489] ========
[A15 L0490]
[A15 L0491] \operatorname{clip}(u_H(a)-d^*(a),,0,,1),
[A15 L0492] ]
[A15 L0493]
[A15 L0494] where `u_H(a)` is Hydra’s public safety score for a discard action and `d^\*(a)` is the exact replay-hidden immediate ron indicator for that tile type. Only legal discard actions are masked in; aka discards are mapped back to base tile type for scoring. ([GitHub][5])
[A15 L0495]
[A15 L0496] This activation should carry an explicit provenance label: **replay-derived, privileged, discard-only auxiliary**. It is allowed now because it is narrow and already concretely wired, not because it satisfies the public-teacher doctrine. Do not use it as precedent for belief supervision. Its minimum falsifiable gate is simple: held-out masked regression quality on valid discard actions must improve over a zero baseline or naive public-score baseline, and primary policy metrics must not regress. ([GitHub][4])
[A15 L0497]
[A15 L0498] ### Activate later
[A15 L0499]
[A15 L0500] **Activate ExIt and `delta_q_target` later, not now.** They are the right next target families, but only after Hydra has a trust-gated AFBS label builder that produces either full `[B,46]` support or an explicit action mask. The current bridge `ΔQ` plane is a discard-only runtime feature summary, not a trainable `[B,46]` target. ExIt can reuse the policy head, but current master does not yet have a clean search-policy carrier/provenance split distinct from ordinary replay one-hot `policy_target`. Minimum gate: hard-state AFBS only, search trust threshold, explicit support mask, and coverage logging. ([GitHub][4])
[A15 L0501]
[A15 L0502] **Activate belief supervision only after the teacher object is fixed, and even then not as the current Stage A path.** The semantic target must be the public posterior marginal `\bar B_t(k,z)` or a gauge-fixed transform of it, built from a public teacher. In current repo reality, that means Stage A must stay off. A future activation would need, at minimum: true public column sizes, public-history likelihood conditioning, weighted CT-SMC posterior expectations or an equivalent valid public teacher, and a canonical mapping into the existing head surface. Minimum gate: row/column conservation residuals near zero, teacher log-likelihood/calibration better than Stage A, and no component-supervision unless decomposition canonicalization is demonstrated. ([GitHub][4])
```

## Artifact 18 — Reward design and variance-reduction analysis
Artifact id: `reward-design-full`
Source label: REWARD
Type: `file_full`
Source: `research/design/REWARD_DESIGN.md`
Why it matters: Pass-two design artifact for how the hidden-world lane should interact with reward-side variance reduction rather than competing with it.

```md
[REWARD L0001] # Hydra Reward Design
[REWARD L0002]
[REWARD L0003] > **Status note:** this is a mixed design/reference document. Keep the reward-analysis evidence and reserve ideas here. For active-path doctrine, use `research/design/HYDRA_RECONCILIATION.md`. For runtime truth, use `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.
[REWARD L0004] >
[REWARD L0005] > Do not treat older `TRAINING.md` references as current governing doctrine.
[REWARD L0006]
[REWARD L0007] Hydra's reward function design, informed by cross-domain analysis of reward systems in Pluribus, ReBeL, AlphaStar, OpenAI Five, and RVR Mahjong.
[REWARD L0008]
[REWARD L0009] > **Background reading:** The full literature survey of reward functions across landmark AI systems is not currently preserved as a standalone archive file in this repo; treat the references and analysis below as the surviving summary surface.
[REWARD L0010]
[REWARD L0011] ---
[REWARD L0012]
[REWARD L0013] ## Table of Contents
[REWARD L0014]
[REWARD L0015] 1. [Reward Variance Reduction for Mahjong (IEEE CoG 2022)](#1-reward-variance-reduction-for-mahjong-ieee-cog-2022)
[REWARD L0016] 2. [Hydra's Reward Function — Final Decision](#2-hydras-reward-function--final-decision)
[REWARD L0017] 3. [References](#references)
[REWARD L0018]
[REWARD L0019] ---
[REWARD L0020]
[REWARD L0021] ## 1. Reward Variance Reduction for Mahjong (IEEE CoG 2022)
[REWARD L0022]
[REWARD L0023] **Paper:** "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" — Li, Wu, Fu, Fu, Zhao, Xing (Tencent AI Lab + CAS + Tsinghua), IEEE CoG 2022
[REWARD L0024] **Game:** 4-player Mahjong (Chinese rules)
[REWARD L0025]
[REWARD L0026] ### The Core Problem
[REWARD L0027]
[REWARD L0028] Mahjong reward has **extremely high variance** from two sources:
[REWARD L0029] 1. **Invisibility:** 3/4 of tiles are hidden (vs. ~50% in poker), making value estimation noisy
[REWARD L0030] 2. **Stochasticity:** The last tile drawn determines win/loss outcome, and *how* you win (tsumo vs. ron, specific tile) dramatically changes the point value
[REWARD L0031]
[REWARD L0032] ### RVR Technique
[REWARD L0033]
[REWARD L0034] Two neural networks work together:
[REWARD L0035]
[REWARD L0036] #### Component 1: Relative Value Network
[REWARD L0037]
[REWARD L0038] - **Purpose:** Reduce variance from hidden information (invisibility)
[REWARD L0039] - **Input:** Oracle view (all 4 players' hands — privileged information)
[REWARD L0040] - **Output:** Simultaneous value estimates for all 4 players: V_θ = (V₁, V₂, V₃, V₄)
[REWARD L0041] - **Zero-sum constraint:** Loss function enforces Σ V_i = 0
[REWARD L0042]
[REWARD L0043] This is exactly **Suphx's oracle guiding** / AlphaStar's centralized value function applied to Mahjong. By seeing all hands during training, the value estimate has much lower variance than one estimated from the acting player's partial observation alone.
[REWARD L0044]
[REWARD L0045] #### Component 2: Expected Reward Network
[REWARD L0046]
[REWARD L0047] - **Purpose:** Reduce variance from end-of-hand stochasticity (luck)
[REWARD L0048] - **Input:** Game state at round T−1 (the penultimate state before the game ends)
[REWARD L0049] - **Output:** Predicted expected reward f_θ(g^{T-1})
[REWARD L0050] - **Key insight:** The *last* tile draw introduces massive variance. A hand might be worth 0 or 12000 points depending on the final draw. By predicting the *expected* reward from the state just before the final draw, this filters out last-tile luck.
[REWARD L0051]
[REWARD L0052] #### Combined Training
[REWARD L0053]
[REWARD L0054] - During training, the raw game reward r_i is replaced with f_θ(g^{T-1}) for the RL update
[REWARD L0055] - The Relative Value Network provides the baseline V(s) for advantage computation
[REWARD L0056] - Together, they reduce both sources of variance simultaneously
[REWARD L0057]
[REWARD L0058] ### Exact Reward Formula
[REWARD L0059]
[REWARD L0060] ```
[REWARD L0061] RL reward = f_θ(g^{T-1})    [Expected Reward Network output]
[REWARD L0062] Advantage = f_θ(g^{T-1}) − V_oracle(s)   [relative to oracle value baseline]
[REWARD L0063] ```
[REWARD L0064]
[REWARD L0065] ### Per-Step vs Per-Episode
[REWARD L0066]
[REWARD L0067] **Per-episode** (per-hand). The reward is the final placement/point change from one hand, but filtered through the Expected Reward Network.
[REWARD L0068]
[REWARD L0069] ### Baseline Subtraction
[REWARD L0070]
[REWARD L0071] - **Relative Value Network** serves as the value baseline
[REWARD L0072] - Zero-sum constraint ensures the four players' advantages sum to zero
[REWARD L0073] - Oracle information (all tiles visible) dramatically tightens the baseline
[REWARD L0074]
[REWARD L0075] ### Reward Normalization
[REWARD L0076]
[REWARD L0077] Not explicitly described. The zero-sum constraint naturally bounds the rewards.
[REWARD L0078]
[REWARD L0079] ### Results
[REWARD L0080]
[REWARD L0081] - Reported faster training convergence compared to vanilla PPO (the paper describes "speedup" qualitatively but does not state a specific speedup multiplier)
[REWARD L0082] - Achieves the same final policy quality with significantly less compute
[REWARD L0083]
[REWARD L0084] ### Key Takeaway for Hydra
[REWARD L0085]
[REWARD L0086] This is **the most directly relevant work.** For Hydra:
[REWARD L0087] 1. **Oracle value baseline** (Relative Value Network) = already planned via oracle distillation
[REWARD L0088] 2. **Expected Reward Network** at T−1 is novel and high-value: it directly addresses Mahjong's biggest variance source (last-tile luck)
[REWARD L0089] 3. **Zero-sum constraint** on value estimates is cheap to implement and provably correct
[REWARD L0090] 4. The convergence speedup matters enormously for Hydra's single-GPU training constraint
[REWARD L0091]
[REWARD L0092] ---
[REWARD L0093]
[REWARD L0094] ## 2. Hydra's Reward Function — Final Decision
[REWARD L0095]
[REWARD L0096] Based on the earlier cross-domain survey work (no longer preserved here as a standalone `archive/REWARD_SURVEY.md` file), Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:
[REWARD L0097]
[REWARD L0098] ### The Formula
[REWARD L0099]
[REWARD L0100] The exact reward formula and implementation priority should be treated as active only when promoted by the reconciled doctrine. Keep the analysis below as reference/evidence rather than a hidden source of authority.
[REWARD L0101]
[REWARD L0102] ### Why This Design
[REWARD L0103]
[REWARD L0104] | Decision | Choice | Evidence |
[REWARD L0105] |----------|--------|----------|
[REWARD L0106] | **Episode boundary** | Per-kyoku | Both Mortal and Suphx use this. ~100× lower variance than per-game. |
[REWARD L0107] | **Reward signal** | GRP ΔE[pts] | Mortal's proven approach. Equivalent to potential-based reward shaping (Ng 1999) — policy-invariant. |
[REWARD L0108] | **Placement points** | [3, 1, -1, -3] | Mortal's training default. Symmetric, zero-sum. Each rank step = 2 pts. Platform-specific via config swap. |
[REWARD L0109] | **GRP design** | 24-class permutation softmax | Captures inter-player rank correlations. 4-class loses this. Mortal proved it works. |
[REWARD L0110] | **Discount γ** | 1.0 | Mortal uses γ=1. Kyoku is short enough (~15 steps). No need for temporal discounting. |
[REWARD L0111] | **Variance reduction** | Oracle critic + ERN | RVR paper: significant speedup. Attacks both variance sources (hidden info + last-tile luck). |
[REWARD L0112] | **GRP lifecycle** | Pretrained, frozen during RL | Stable reward signal. Mortal does this. Avoids moving-target problem. |
[REWARD L0113] | **Reward normalization** | Running std (Welford) | Mortal-Policy's exact approach. Essential for PPO in high-variance games. |
[REWARD L0114] | **No reward shaping** | Skip (GRP delta IS PBRS already) | Double-shaping adds risk. Shanten-based shaping creates offensive bias — worst possible for Mahjong. |
[REWARD L0115] | **No intrinsic motivation** | Skip | SL warm-start solves exploration. RND/ICM would add noise from tile draw stochasticity. |
[REWARD L0116] | **Same reward all phases** | Mandatory | Changing reward invalidates value function. Cal-QL (NeurIPS 2023) showed this causes "unlearning." |
[REWARD L0117]
[REWARD L0118] ### Confirmed Anti-Patterns (From Mortal Community)
[REWARD L0119]
[REWARD L0120] The anti-pattern list below is retained as reference guidance; do not treat dead `TRAINING.md` links as live authority.
[REWARD L0121]
[REWARD L0122] ### Platform-Specific Fine-Tuning (Via pts_vector Swap)
[REWARD L0123]
[REWARD L0124] | Target Platform | pts_vector | Strategy Bias |
[REWARD L0125] |----------------|------------|---------------|
[REWARD L0126] | General training | [3, 1, -1, -3] | Balanced (default) |
[REWARD L0127] | Tenhou Houou | [3, 1.5, 0, -4.5] | Avoid 4th (normalized Tenhou net pts) |
[REWARD L0128] | Mahjong Soul Throne | [3, 1, -1, -3] | Balanced (Majsoul uma is already nearly symmetric) |
[REWARD L0129] | WRC / EMA tournament | [3, 1, -1, -3] | Balanced (identical incentive structure) |
[REWARD L0130] | M-League style | [5, 1, -1, -3] | Push for 1st |
[REWARD L0131]
[REWARD L0132] ---
[REWARD L0133]
[REWARD L0134] ## References
[REWARD L0135]
[REWARD L0136] | Ref | Paper | Year | Venue |
[REWARD L0137] |-----|-------|------|-------|
[REWARD L0138] | [6] | Li et al., "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" | 2022 | IEEE CoG |
[REWARD L0139] | [9] | Ng et al., "Policy invariance under reward transformations" | 1999 | ICML |
[REWARD L0140] | [10] | Harutyunyan et al., "Hindsight Credit Assignment" | 2019 | NeurIPS |
[REWARD L0141] | [11] | Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" | 2020 | arXiv |
[REWARD L0142] | [12] | Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning" | 2023 | NeurIPS |
[REWARD L0143] | [14] | Engstrom et al., "Implementation Matters in Deep Policy Gradients" | 2020 | ICLR |
[REWARD L0144] | [15] | Huang, "The 37 Implementation Details of Proximal Policy Optimization" | 2022 | Blog/ICLR |
[REWARD L0145]
[REWARD L0146] > References [1]-[5] and [7]-[8] come from the earlier cross-domain reward survey work, but that standalone archive file is not currently present in this repo.
```

## Artifact 19 — Repo-grounded belief-before-mixture closure blueprint
Artifact id: `answer27-closure-blueprint`
Source label: A27
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_27_combined.md:337-554`
Why it matters: Pass-two blueprint slice that explains how belief can move before mixture using current carrier surfaces while mixture stays off.

```md
[A27 L0337]   <![CDATA[Here is the smallest repo-grounded path I would take.
[A27 L0338]
[A27 L0339] `belief_fields` can move before `mixture_weight`, but only if you stop treating the 16×34 surface as a true per-component mixture teacher and instead use it as a carrier for a **single public-posterior belief object** replicated across the four existing component blocks. `mixture_weight` should stay fully off in v1, both at the producer and at `train.rs` policy-gate layers.
[A27 L0340]
[A27 L0341] That recommendation is grounded by four direct repo facts.
[A27 L0342]
[A27 L0343] First, the current loader seam already supports belief present while mixture absent. `build_stage_a_belief_targets` returns belief and mixture independently as `(Option<[f32;16*34]>, Option<[f32;4]>, bool, bool)`, so you already have the carrier needed for “belief on, mixture off” with no batch-schema change (Artifact 10 LOADER L0308-L0335).
[A27 L0344]
[A27 L0345] Second, the batch and model contracts are already the right size. `belief_fields_target` is `[B,16,34]`, `mixture_weight_target` is `[B,4]`, with sample-level masks `[B]`; the model outputs the same shapes; no new head is required (Artifacts 11, 25, 33, 35, 36).
[A27 L0346]
[A27 L0347] Third, the current Stage-A mixture is weaker than it looks. `build_stage_a_teacher` calls `MixtureSib::new(...)` and then never calls `bayesian_update`, `posterior_step`, `reproject`, split, or merge before reading back beliefs and weights (Artifact 09 BELIEF L0096-L0153; Artifact 30 SINK L0196-L0210, L0347-L0361). `MixtureSib::new` initializes **all components with the same projected belief** and equal log-weights (Artifact 30 SINK L0202-L0207). So Stage-A is already emitting identical component beliefs with uniform weights. I also checked the current trust formula numerically: with 4 uniform weights, entropy is `ln 4 ≈ 1.386294` and trust is about `0.700001`, above the default `0.55` threshold, while mixture emission is blocked by the `1.15` entropy threshold. So the current code can emit “belief” broadly while the “mixture” is uninformative and suppressed. That makes the existing trust/entropy heuristics the wrong thing to preserve.
[A27 L0348]
[A27 L0349] Fourth, `mixture_weight` has an explicit identity problem in current runtime code. `build_search_features` sorts components by current weight before writing them into the fixed planes (Artifact 27 BRIDGE L0316-L0335). That means component slot 0 is “top-ranked this sample,” not a canonical cross-sample identity. So even before doctrine, the repo already treats component order as local/ranked rather than stable.
[A27 L0350]
[A27 L0351] That is why I would split the work into a substrate phase and an activation phase.
[A27 L0352]
[A27 L0353] ## 1. Substrate phase: land the stronger lane with train gates unchanged
[A27 L0354]
[A27 L0355] This is the part I would do first, before touching `train.rs` policy gates.
[A27 L0356]
[A27 L0357] ### 1.1 `crates/hydra-core/src/ct_smc.rs`
[A27 L0358]
[A27 L0359] **Direct support:** current API exposes `weighted_mean_tile_count(tile,col)` and `mean_allocation()`; the latter is unweighted, and the archive explicitly warns that a future public belief teacher should use the weighted path, not simple particle averaging (Artifact 17 CTSMC L0303-L0323; Artifact 32 CTSMC L0341-L0359; Artifact 23 A15 L0331-L0334).
[A27 L0360]
[A27 L0361] **Change first:**
[A27 L0362] Add a narrow helper:
[A27 L0363]
[A27 L0364] ```rust
[A27 L0365] impl CtSmc {
[A27 L0366]     pub fn weighted_mean_allocation(&self) -> [[f32; 4]; 34] {
[A27 L0367]         let mut out = [[0.0f32; 4]; 34];
[A27 L0368]         if self.particles.is_empty() {
[A27 L0369]             return out;
[A27 L0370]         }
[A27 L0371]
[A27 L0372]         let max_w = self.max_log_weight();
[A27 L0373]         let mut w_sum = 0.0f64;
[A27 L0374]
[A27 L0375]         for p in &self.particles {
[A27 L0376]             let w = (p.log_weight - max_w).exp();
[A27 L0377]             w_sum += w;
[A27 L0378]             for tile in 0..34 {
[A27 L0379]                 for zone in 0..4 {
[A27 L0380]                     out[tile][zone] += (w * p.allocation[tile][zone] as f64) as f32;
[A27 L0381]                 }
[A27 L0382]             }
[A27 L0383]         }
[A27 L0384]
[A27 L0385]         if w_sum > 0.0 {
[A27 L0386]             let inv = (1.0 / w_sum) as f32;
[A27 L0387]             for row in &mut out {
[A27 L0388]                 for v in row {
[A27 L0389]                     *v *= inv;
[A27 L0390]                 }
[A27 L0391]             }
[A27 L0392]         }
[A27 L0393]         out
[A27 L0394]     }
[A27 L0395] }
[A27 L0396] ```
[A27 L0397]
[A27 L0398] **Why this is justified:** it is the smallest change that closes the weighted-posterior seam without touching heads, loaders, or runtime encoder layout.
[A27 L0399]
[A27 L0400] **Tests to add before anything else:**
[A27 L0401]
[A27 L0402] * `weighted_mean_allocation_respects_particle_weights`
[A27 L0403] * `weighted_mean_allocation_differs_from_mean_allocation_when_weights_skewed`
[A27 L0404]
[A27 L0405] Validation is simple: create two particles with different allocations and log weights `0.0` and `-10.0`; `weighted_mean_allocation()` should sit near the first particle, while `mean_allocation()` sits near the arithmetic average.
[A27 L0406]
[A27 L0407] ### 1.2 `crates/hydra-train/src/teacher/belief.rs`
[A27 L0408]
[A27 L0409] This is the main file to change first.
[A27 L0410]
[A27 L0411] **What stays exact:**
[A27 L0412]
[A27 L0413] * `BELIEF_COMPONENTS = 4`
[A27 L0414] * `BELIEF_ZONES = 4`
[A27 L0415] * `BELIEF_TILES = 34`
[A27 L0416] * flattened carrier size `16 * 34`
[A27 L0417]
[A27 L0418] **What should be removed from the active path:**
[A27 L0419]
[A27 L0420] * `build_uniform_kernel()`
[A27 L0421] * `project_hidden_count_to_col_sums(hidden_tiles)`
[A27 L0422] * `build_stage_a_teacher(...)` as the loader’s default path
[A27 L0423]
[A27 L0424] Keep them only as legacy helpers if you want archived regression tests, but get them out of the active loader path.
[A27 L0425]
[A27 L0426] **What to add:**
[A27 L0427] A narrow, semantics-agnostic encoder from a stronger public teacher object into the existing 16×34 carrier. The teacher object coming from Agent A can still vary; the encoding path does not need to.
[A27 L0428]
[A27 L0429] I would standardize the train-time object as rowwise public posteriors:
[A27 L0430]
[A27 L0431] [
[A27 L0432] P_t(z \mid k) = \frac{\bar B_t(k,z)}{\sum_{z'} \bar B_t(k,z')}
[A27 L0433] ]
[A27 L0434]
[A27 L0435] for rows with positive remaining mass, and zero rows otherwise (Artifact 23 A15 L0264-L0272).
[A27 L0436]
[A27 L0437] Then encode that into the existing 16×34 slot by **repeating the same 4-zone row distribution across all four component blocks**:
[A27 L0438]
[A27 L0439] ```rust
[A27 L0440] pub fn encode_repeated_public_belief(
[A27 L0441]     row_post: &[[f32; BELIEF_ZONES]; BELIEF_TILES],
[A27 L0442] ) -> [f32; BELIEF_FIELDS_SIZE] {
[A27 L0443]     let mut out = [0.0f32; BELIEF_FIELDS_SIZE];
[A27 L0444]     for component in 0..BELIEF_COMPONENTS {
[A27 L0445]         for zone in 0..BELIEF_ZONES {
[A27 L0446]             let ch = component * BELIEF_ZONES + zone;
[A27 L0447]             for tile in 0..BELIEF_TILES {
[A27 L0448]                 out[ch * BELIEF_TILES + tile] = row_post[tile][zone];
[A27 L0449]             }
[A27 L0450]         }
[A27 L0451]     }
[A27 L0452]     out
[A27 L0453] }
[A27 L0454] ```
[A27 L0455]
[A27 L0456] and a normalizer:
[A27 L0457]
[A27 L0458] ```rust
[A27 L0459] pub fn posterior_counts_to_row_post(
[A27 L0460]     counts: &[[f32; BELIEF_ZONES]; BELIEF_TILES],
[A27 L0461] ) -> [[f32; BELIEF_ZONES]; BELIEF_TILES] {
[A27 L0462]     let mut out = [[0.0f32; BELIEF_ZONES]; BELIEF_TILES];
[A27 L0463]     for tile in 0..BELIEF_TILES {
[A27 L0464]         let row_sum: f32 = counts[tile].iter().sum();
[A27 L0465]         if row_sum > 0.0 {
[A27 L0466]             for zone in 0..BELIEF_ZONES {
[A27 L0467]                 out[tile][zone] = counts[tile][zone] / row_sum;
[A27 L0468]             }
[A27 L0469]         }
[A27 L0470]     }
[A27 L0471]     out
[A27 L0472] }
[A27 L0473] ```
[A27 L0474]
[A27 L0475] **Why I recommend duplication across component blocks:**
[A27 L0476] This is the least invasive way to reuse `[16,34]` without inventing a new head or pretending component identity exists. It is also not a semantic regression relative to the current Stage-A path, because Stage-A already initializes identical component beliefs and never differentiates them before emission.
[A27 L0477]
[A27 L0478] **What to emit in v1:**
[A27 L0479]
[A27 L0480] * `belief_fields = Some(encode_repeated_public_belief(...))`
[A27 L0481] * `mixture_weights = None`
[A27 L0482]
[A27 L0483] Do not try to fit or emit component weights in this file in v1.
[A27 L0484]
[A27 L0485] **Deterministic validity checks instead of trust heuristics:**
[A27 L0486] Replace the current `trust_threshold` / `mixture_entropy_threshold` gating with hard validity checks:
[A27 L0487]
[A27 L0488] * finite
[A27 L0489] * nonnegative
[A27 L0490] * if counts are available before normalization: row sums match public remaining counts within tolerance
[A27 L0491] * if counts are available before normalization: column sums match public zone sizes within tolerance
[A27 L0492] * after normalization: each valid row sums to `1 ± eps`; invalid rows are exactly zero
[A27 L0493]
[A27 L0494] That is a better gate than the current entropy/trust pair, which is not tied to teacher correctness.
[A27 L0495]
[A27 L0496] **Tests to replace/add here:**
[A27 L0497]
[A27 L0498] * replace `stage_a_teacher_can_emit_mixture_weights` with `public_belief_target_omits_mixture_weights_in_v1`
[A27 L0499] * replace the current “finite and nonnegative” belief test with:
[A27 L0500]
[A27 L0501]   * `public_belief_target_rows_normalize_or_zero`
[A27 L0502]   * `public_belief_target_repeats_across_component_blocks`
[A27 L0503]   * `public_belief_target_rejects_nonfinite_or_negative_inputs`
[A27 L0504]   * if counts path is available: `public_belief_target_preserves_row_and_col_marginals_before_normalization`
[A27 L0505]
[A27 L0506] ## 2. Loader phase: swap the teacher, not the carriers
[A27 L0507]
[A27 L0508] ### 2.1 `crates/hydra-train/src/data/mjai_loader.rs`
[A27 L0509]
[A27 L0510] The exact seam is `build_stage_a_belief_targets(...)` (Artifact 10 LOADER L0308-L0335).
[A27 L0511]
[A27 L0512] **Change:**
[A27 L0513] Rename it to something like `build_belief_targets(...)` and stop computing a single `hidden_tiles` total as the active teacher input.
[A27 L0514]
[A27 L0515] The loader already has the public row marginals:
[A27 L0516]
[A27 L0517] * `remaining = extract_public_remaining_counts(...)`
[A27 L0518]
[A27 L0519] It also already has the public zone-size pieces:
[A27 L0520]
[A27 L0521] * each opponent concealed size from `state.players[*].hand_len`
[A27 L0522] * wall remainder from `state.wall.remaining()`
[A27 L0523]
[A27 L0524] So the loader should compute exact zone sizes in canonical zone order and pass those to the stronger teacher provider instead of collapsing them to one total.
[A27 L0525]
[A27 L0526] Do **not** restate the zone order from memory. Pin it by test. The artifact packet does not show the full table, and this is exactly the sort of thing that should be frozen by an asymmetric unit test rather than verbal recall.
[A27 L0527]
[A27 L0528] Suggested helper:
[A27 L0529]
[A27 L0530] ```rust
[A27 L0531] fn hidden_zone_sizes(state: &GameState, actor: usize) -> [usize; 4] {
[A27 L0532]     let opp = canonical_belief_zone_order(actor); // test this explicitly
[A27 L0533]     [
[A27 L0534]         state.players[opp[0]].hand_len as usize,
[A27 L0535]         state.players[opp[1]].hand_len as usize,
[A27 L0536]         state.players[opp[2]].hand_len as usize,
[A27 L0537]         state.wall.remaining(),
[A27 L0538]     ]
[A27 L0539] }
[A27 L0540] ```
[A27 L0541]
[A27 L0542] **Critical v1 rule:** if the stronger teacher is not available, return `(None, None, false, false)`.
[A27 L0543] Do not silently fall back to Stage-A. That matches the doctrine to leave unavailable targets absent rather than fabricating weak labels (Artifact 19 RECON L0445-L0455).
[A27 L0544]
[A27 L0545] **This is the exact v1 return you want when belief is available but mixture stays off:**
[A27 L0546]
[A27 L0547] ```rust
[A27 L0548] (
[A27 L0549]     Some(belief_fields),
[A27 L0550]     None,
[A27 L0551]     true,
[A27 L0552]     false,
[A27 L0553] )
[A27 L0554] ```
```

## Artifact 20 — Semantic-object belief closure and Stage-A diagnosis
Artifact id: `answer28-semantic-object-closure`
Source label: A28
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_28_combined.md:275-509`
Why it matters: Pass-two semantic anchor for what current Stage-A still gets wrong and what the stronger object replaces.

```md
[A28 L0275]   <![CDATA[The replacement object should be **a single public posterior over hidden allocation**, not a fake 4-component mixture.
[A28 L0276]
[A28 L0277] ## 1. What the current Stage-A teacher is actually doing
[A28 L0278]
[A28 L0279] **Direct artifact support.** `belief.rs` takes `remaining[34]`, clips it to nonnegative row sums, turns one scalar `hidden_tiles` into four equal column sums, uses an all-ones kernel, calls `MixtureSib::new`, and then writes the returned component beliefs into the `[4 components × 4 zones × 34 tiles]` tensor. `MixtureSib::new` itself computes **one** Sinkhorn projection and clones that same table into every component with equal log-weights; Stage A never calls `bayesian_update`. (Artifact 08, BELIEF L0077-L0152; Artifact 22, SINK L0196-L0209.)
[A28 L0280]
[A28 L0281] From that code, the Stage-A object has a closed form.
[A28 L0282]
[A28 L0283] Let
[A28 L0284]
[A28 L0285] [
[A28 L0286] r_k := \max(\texttt{remaining}[k], 0), \qquad R := \sum_k r_k,
[A28 L0287] ]
[A28 L0288] [
[A28 L0289] c_z := H/4 \quad \text{for } z\in{1,2,3,w}, \qquad H:=\texttt{hidden_tiles},
[A28 L0290] ]
[A28 L0291] [
[A28 L0292] K_{kz} := 1.
[A28 L0293] ]
[A28 L0294]
[A28 L0295] Because the kernel is constant, Sinkhorn returns the rank-1 table
[A28 L0296]
[A28 L0297] [
[A28 L0298] B_{kz}=\frac{r_k c_z}{R}=\frac{r_k H}{4R}.
[A28 L0299] ]
[A28 L0300]
[A28 L0301] And because `MixtureSib::new` clones that same `B` into every component,
[A28 L0302]
[A28 L0303] [
[A28 L0304] B^{(1)}=B^{(2)}=B^{(3)}=B^{(4)}=B,
[A28 L0305] \qquad
[A28 L0306] w_\ell = \frac14.
[A28 L0307] ]
[A28 L0308]
[A28 L0309] So the emitted `belief_fields` tensor is just **four identical copies** of the same 4-zone table. This is not a multimodal teacher. It is one outer-product table duplicated four times. (Artifact 08, BELIEF L0111-L0144; Artifact 22, SINK L0196-L0209.)
[A28 L0310]
[A28 L0311] The row-conditional object makes the weakness even clearer:
[A28 L0312]
[A28 L0313] [
[A28 L0314] P(z\mid k)=\frac{B_{kz}}{\sum_{z'} B_{kz'}} = \frac14
[A28 L0315] ]
[A28 L0316]
[A28 L0317] for every tile row with positive mass. So Stage A’s hidden-zone belief is **uniform across the 4 zones for every tile type**. The only nontrivial quantity it carries is the public row magnitude (r_k), which is already determined by visible tiles. That is why the archive calls it a projection artifact rather than a posterior. Confidence: **high**. (Artifact 06/15; Artifact 16 L0300-L0314.)
[A28 L0318]
[A28 L0319] There is a second, narrower bug-like seam in the current tests. The test case uses `remaining = [1;34]`, so (R=34), but passes `hidden_tiles=40`, so (\sum_z c_z = 40). Under the formula above, each row becomes (40/34 \approx 1.17647), not 1. So the current tests do **not** validate conservation against the supplied row marginal; they only check finiteness/nonnegativity. That is a direct consequence of the artifact code and the literal test inputs, not a guess about production behavior. Confidence: **high**. (Artifact 08, BELIEF L0165-L0187.)
[A28 L0320]
[A28 L0321] The trust gate is also much weaker than it looks. Since all component weights are always uniform at construction time, Stage A always has
[A28 L0322]
[A28 L0323] [
[A28 L0324] \text{ESS}=L,\qquad H_w=\log L.
[A28 L0325] ]
[A28 L0326]
[A28 L0327] For the default (L=4),
[A28 L0328]
[A28 L0329] [
[A28 L0330] \texttt{trust}
[A28 L0331] = 0.7\cdot \frac{\text{ESS}}{L}
[A28 L0332]
[A28 L0333] * 0.3\cdot \left(1-\frac{H_w}{1.3863}\right)
[A28 L0334]   \approx 0.700001.
[A28 L0335]   ]
[A28 L0336]
[A28 L0337] That is above the default threshold (0.55), so for any sample with `hidden_tiles > 0` and positive row mass, belief is basically emitted by default. But `mixture_weights` are suppressed by default because (\log 4 \approx 1.38629 > 1.15). So the default behavior is effectively **belief on / mixture off**, regardless of any real posterior evidence. Confidence: **high**. (Artifact 08, BELIEF L0067-L0075, L0113-L0144; Artifact 22, SINK L0367-L0441.)
[A28 L0338]
[A28 L0339] ## 2. What the stronger public teacher object should be
[A28 L0340]
[A28 L0341] The semantically correct object is the **public posterior expected hidden allocation**.
[A28 L0342]
[A28 L0343] Let (I_t) be the public information state. Let hidden allocation be a fixed-margin contingency table
[A28 L0344]
[A28 L0345] [
[A28 L0346] X_t \in \mathbb{Z}_{\ge 0}^{34\times 4},
[A28 L0347] ]
[A28 L0348]
[A28 L0349] with row sums equal to public remaining counts (r_t(k)) and column sums equal to public hidden-zone sizes (s_t(z)) (three opponent concealed hands plus wall). Hydra’s own doctrine already points to exactly this transport/polytope view, and in partially observed control more generally the correct belief object is a probability distribution over hidden states, not a single hidden realization. Sinkhorn’s role in the fast path is to compute a KL / entropic projection onto those fixed marginals; the projected plan is the semantically meaningful object. ([MIT CSAIL][1])
[A28 L0350]
[A28 L0351] So the canonical teacher should be
[A28 L0352]
[A28 L0353] [
[A28 L0354] q_t(X) := p_{\text{teacher}}(X\mid I_t),
[A28 L0355] \qquad
[A28 L0356] \bar B_t(k,z) := \mathbb{E}_{q_t}[X_t(k,z)].
[A28 L0357] ]
[A28 L0358]
[A28 L0359] That is the object that should replace the current Stage-A pseudo-posterior. It is:
[A28 L0360]
[A28 L0361] * public-side,
[A28 L0362] * identifiable,
[A28 L0363] * permutation-invariant,
[A28 L0364] * and still lives in the same 34×4 hidden-zone space.
[A28 L0365]
[A28 L0366] This is exactly the object the archive excerpt reconstructs as the “projected public posterior expected allocation,” and it is the object that stays meaningful even if the underlying posterior engine is CT-SMC or Mixture-SIB. Confidence: **high**. (Artifact 07, FINAL L0147-L0161; Artifact 13, OPPMODEL L0663-L0672; Artifact 16, A15 L0234-L0286.)
[A28 L0367]
[A28 L0368] ## 3. What `belief_fields` should mean
[A28 L0369]
[A28 L0370] For the **canonical** teacher, keep (\bar B_t) as the SSOT object.
[A28 L0371]
[A28 L0372] For the **`belief_fields` carrier**, the cleanest deterministic representation is the row-conditional version:
[A28 L0373]
[A28 L0374] [
[A28 L0375] P_t(z\mid k)=
[A28 L0376] \begin{cases}
[A28 L0377] \bar B_t(k,z) / r_t(k), & r_t(k)>0 \
[A28 L0378] \text{masked}, & r_t(k)=0.
[A28 L0379] \end{cases}
[A28 L0380] ]
[A28 L0381]
[A28 L0382] Why this is the right `belief_fields` object:
[A28 L0383]
[A28 L0384] 1. It preserves exactly the same posterior semantics as (\bar B_t).
[A28 L0385] 2. It removes the trivial public row-marginal factor, which the observation already determines.
[A28 L0386] 3. It turns each tile row into a genuine 4-way uncertainty target.
[A28 L0387] 4. It fits the doctrinal rule “projected/public-teacher belief objects or gauge-fixed marginals, not raw fields.” (Artifact 12, RECON L0407-L0409, L0467-L0469; Artifact 16, A15 L0257-L0286.)
[A28 L0388]
[A28 L0389] I would **not** supervise raw Sinkhorn external fields (F_\theta). Those are gauge-dependent: adding row and column potentials can leave the projected plan unchanged after scaling. More broadly, matrix-scaling theory identifies the scaled plan, not raw pre-scaling fields; even the diagonal scalings are unique only up to scalar factors. That is why the projected table or a gauge-fixed transform is the identifiable object, not the raw field tensor. ([ScienceDirect][2])
[A28 L0390]
[A28 L0391] If Hydra later wants a logit-like representation instead of probabilities, the acceptable transform is the archive’s gauge-fixed row logit:
[A28 L0392]
[A28 L0393] [
[A28 L0394] g_t(k,z)=
[A28 L0395] \log(\bar B_t(k,z)+\varepsilon)
[A28 L0396] -\frac14\sum_{z'}\log(\bar B_t(k,z')+\varepsilon).
[A28 L0397] ]
[A28 L0398]
[A28 L0399] But that is a **derived encoding** of the same (\bar B_t), not a different teacher.
[A28 L0400]
[A28 L0401] ## 4. What the teacher source should be
[A28 L0402]
[A28 L0403] The strongest exact teacher source is:
[A28 L0404]
[A28 L0405] [
[A28 L0406] \bar B_t(k,z)=\texttt{ct_smc.weighted_mean_tile_count}(k,z).
[A28 L0407] ]
[A28 L0408]
[A28 L0409] That is already available cellwise from current CT-SMC APIs, and the repo artifacts explicitly distinguish that weighted path from the unweighted `mean_allocation()` helper. The weighted path is the correct one whenever particle weights still encode posterior likelihood. Confidence: **high**. (Artifact 10, CTSMC L0303-L0323; Artifact 24, CTSMC L0303-L0359; Artifact 16, A15 L0331-L0334.)
[A28 L0410]
[A28 L0411] The **narrowest semantically honest v1** is therefore:
[A28 L0412]
[A28 L0413] * **use CT-SMC weighted posterior mean when CT-SMC is present;**
[A28 L0414] * **otherwise emit no belief label.**
[A28 L0415]
[A28 L0416] I would **not** let Stage A survive as a fallback. Artifact doctrine is explicit that unavailable targets should stay absent rather than be replaced by weak fabricated labels. Confidence: **high**. (Artifact 12, RECON L0444-L0455.)
[A28 L0417]
[A28 L0418] There is one acceptable **later** fallback inside the same lane: if Hydra has a real posterior-updated Mixture-SIB object with correct public marginals and nontrivial kernels, then the permutation-invariant aggregate
[A28 L0419]
[A28 L0420] [
[A28 L0421] \bar B_t = \sum_{\ell} w_\ell B_t^{(\ell)}
[A28 L0422] ]
[A28 L0423]
[A28 L0424] is a semantically valid approximate teacher. But that is a coverage-expansion fallback, not the narrowest honest v1. Confidence: **medium**. (Artifact 07, FINAL L0133-L0141; Artifact 22, SINK L0416-L0425.)
[A28 L0425]
[A28 L0426] ## 5. `mixture_weight` should remain off
[A28 L0427]
[A28 L0428] This is the clearest answer in the packet: **keep `mixture_weight` off for now.**
[A28 L0429]
[A28 L0430] Reason 1: even with a correct aggregate posterior (\bar B_t), the decomposition
[A28 L0431]
[A28 L0432] [
[A28 L0433] \bar B_t = \sum_{\ell=1}^4 w_\ell B_t^{(\ell)}
[A28 L0434] ]
[A28 L0435]
[A28 L0436] is non-unique. Aggregate posterior does **not** identify a unique 4-component fit. Confidence: **high**. (Artifact 16, A15 L0318-L0335.)
[A28 L0437]
[A28 L0438] Reason 2: the runtime side currently sorts components by descending weight before encoding them into search features. That is a within-sample ranking convention, not a stable cross-sample component-identity contract. Confidence: **high**. (Artifact 20, BRIDGE L0315-L0335.)
[A28 L0439]
[A28 L0440] Reason 3: mixture models have a standard label-switching problem. Stan’s guide states that mixture components are exchangeable and that only label-switching-invariant inferences are sound; Stephens shows componentwise posterior means and marginal summaries can become nonsensical under label switching. So until Hydra defines a **canonical public-teacher mixture fit** and a **canonical component ordering**, `mixture_weight` supervision is not semantically closed. ([Stan][3])
[A28 L0441]
[A28 L0442] So the right policy is:
[A28 L0443]
[A28 L0444] * `belief_fields`: can be repaired around the aggregate public posterior.
[A28 L0445] * `mixture_weight`: **stay off** until canonical mixture identity exists.
[A28 L0446]
[A28 L0447] Confidence: **high**.
[A28 L0448]
[A28 L0449] ## 6. Invariants the replacement teacher must satisfy
[A28 L0450]
[A28 L0451] 1. **Public-only semantics.**
[A28 L0452]    The emitted target must be a function of (I_t) and a teacher-side posterior built from public information, never the realized hidden allocation.
[A28 L0453]    Validation: hold public history fixed, vary hidden realization among states consistent with it; the teacher must not change.
[A28 L0454]    Confidence: **high**. (Artifact 16, A15 L0288-L0298.) ([MIT CSAIL][1])
[A28 L0455]
[A28 L0456] 2. **Exact margin conservation.**
[A28 L0457]    [
[A28 L0458]    \sum_z \bar B_t(k,z)=r_t(k), \qquad
[A28 L0459]    \sum_k \bar B_t(k,z)=s_t(z).
[A28 L0460]    ]
[A28 L0461]    Validation: row/column sums checked to tolerance on every emitted sample.
[A28 L0462]    Confidence: **high**. (Artifact 13, OPPMODEL L0663-L0668.)
[A28 L0463]
[A28 L0464] 3. **Weighted posterior expectation, not unweighted particle average.**
[A28 L0465]    Validation: construct a nonuniform-weight particle set and verify teacher differs from `mean_allocation()`.
[A28 L0466]    Confidence: **high**. (Artifact 24, CTSMC L0303-L0359.)
[A28 L0467]
[A28 L0468] 4. **Permutation-invariant v1 semantics.**
[A28 L0469]    The v1 teacher must not depend on arbitrary component IDs.
[A28 L0470]    Validation: permuting mixture components leaves the aggregate teacher unchanged.
[A28 L0471]    Confidence: **high**. ([Stan][3])
[A28 L0472]
[A28 L0473] 5. **Zero-row masking.**
[A28 L0474]    Rows with (r_t(k)=0) are masked, not supervised.
[A28 L0475]    Validation: row mask equals `1[r_t(k)>0]`.
[A28 L0476]    Confidence: **high**. (Artifact 16, A15 L0264-L0272.)
[A28 L0477]
[A28 L0478] 6. **Audit metrics are not semantic gates.**
[A28 L0479]    ESS / entropy can be logged, but a concentrated posterior is not an invalid posterior.
[A28 L0480]    Validation: a single-particle / collapsed posterior should still emit a label if numerically sane and margin-consistent.
[A28 L0481]    Confidence: **medium-high**. This is a proposal, but it follows directly from how posterior concentration works and from the current Stage-A gate’s perversity.
[A28 L0482]
[A28 L0483] ## 7. What to keep, narrow, defer, reject
[A28 L0484]
[A28 L0485] **Keep**
[A28 L0486]
[A28 L0487] * the transport-polytope framing;
[A28 L0488] * the 4-zone public semantics including wall;
[A28 L0489] * the doctrine “projected/public belief object, not hidden realization”;
[A28 L0490] * CT-SMC weighted means as the search-grade source.
[A28 L0491]   Confidence: **high**.
[A28 L0492]
[A28 L0493] **Narrow**
[A28 L0494]
[A28 L0495] * `belief_fields` to the **aggregate public posterior** only;
[A28 L0496] * the `belief_fields` carrier to row-conditional (P(z\mid k)) or a gauge-fixed transform of that same object;
[A28 L0497] * audit stats to diagnostics only.
[A28 L0498]   Confidence: **high**.
[A28 L0499]
[A28 L0500] **Defer**
[A28 L0501]
[A28 L0502] * any per-component belief supervision;
[A28 L0503] * `mixture_weight` activation;
[A28 L0504] * canonical mixture fitting / relabeling / ordering contracts.
[A28 L0505]   Confidence: **high**.
[A28 L0506]
[A28 L0507] **Reject**
[A28 L0508]
[A28 L0509] * equalized hidden-zone totals;
```

## Artifact 21 — Pass-two external design-closure artifact bank
Artifact id: `followup-external-bank`
Source label: EXT2
Type: `file_full`
Source: `research/agent_handoffs/genie_hidden_world_followup_external_artifacts.md`
Why it matters: A small, focused external bank for pass-two decisions: calibration evaluation, offline-to-online RL, and privileged-information distillation.

```md
[EXT2 L0001] # Hydra hidden-world pass-two — external design-closure artifact bank
[EXT2 L0002]
[EXT2 L0003] This file is intentionally small and pass-two-specific. The first hidden-world packet already carried a broad cross-field discovery bank. This follow-up bank is narrower. It exists to help the genie choose the actual winning design stack, training recipe, evaluation gates, and kill criteria rather than re-running general discovery.
[EXT2 L0004]
[EXT2 L0005] The selection rule is strict:
[EXT2 L0006]
[EXT2 L0007] - include only outside artifacts that sharpen a concrete pass-two decision,
[EXT2 L0008] - especially algorithm choice, teacher/student distillation, calibration-gate design, and offline-to-online policy improvement,
[EXT2 L0009] - avoid broad survey artifacts that merely repeat pass-one context.
[EXT2 L0010]
[EXT2 L0011] ## Artifact F01 — Estimating Expected Calibration Errors
[EXT2 L0012]
[EXT2 L0013] - URL: https://arxiv.org/abs/2109.03480
[EXT2 L0014] - Domain: calibration evaluation
[EXT2 L0015] - Type: primary paper
[EXT2 L0016] - Suggested label: `ext_pass2_ece_estimation`
[EXT2 L0017]
[EXT2 L0018] Why it matters in pass two:
[EXT2 L0019]
[EXT2 L0020] Pass one already established that calibration matters. Pass two needs something stricter: calibration **evaluation** itself can be misleading if the metric is chosen or estimated poorly. This paper directly sharpens the pass-two job of defining promotion gates and kill criteria for belief-adjacent confidence outputs, danger outputs, tenpai probability, trust scores, or search-deferral triggers.
[EXT2 L0021]
[EXT2 L0022] Exact pass-two use:
[EXT2 L0023]
[EXT2 L0024] - force the genie to distinguish “measure calibration” from “measure calibration well”
[EXT2 L0025] - justify richer gate design than a single naive ECE scalar
[EXT2 L0026] - encourage phase-conditioned, bucket-sensitive, and estimator-aware calibration checks
[EXT2 L0027]
[EXT2 L0028] What it should influence:
[EXT2 L0029]
[EXT2 L0030] - `Evaluation Gates`
[EXT2 L0031] - `Kill Criteria`
[EXT2 L0032] - any design that uses belief confidence to gate runtime search, abstention, or action trust
[EXT2 L0033]
[EXT2 L0034] ## Artifact F02 — Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization
[EXT2 L0035]
[EXT2 L0036] - URL: https://www.ijcai.org/proceedings/2024/0477
[EXT2 L0037] - Domain: offline-to-online RL transition
[EXT2 L0038] - Type: primary conference paper
[EXT2 L0039] - Suggested label: `ext_pass2_offline_to_online_cpr`
[EXT2 L0040]
[EXT2 L0041] Why it matters in pass two:
[EXT2 L0042]
[EXT2 L0043] The hidden-world lane probably will not go straight from a perfect teacher object to a strong live policy. There is a transition problem: how to turn offline or replay-side supervision into online improvement without policy lock-in, instability, or brittle imitation. CPR is useful because it is not another generic RL paper; it explicitly focuses on stable fine-tuning after an offline-pretrained policy and on keeping improvement stable when the starting policy is already overtrained or brittle.
[EXT2 L0044]
[EXT2 L0045] Exact pass-two use:
[EXT2 L0046]
[EXT2 L0047] - sharpen the `Training Recipe` section for a hidden-world lane that starts from offline teacher signals and then moves into online or self-play refinement
[EXT2 L0048] - help the genie specify whether and how Hydra should separate feature reuse from policy revitalization
[EXT2 L0049] - provide a useful external precedent for why “good offline teacher” is not automatically “good online improvement path”
[EXT2 L0050]
[EXT2 L0051] What it should influence:
[EXT2 L0052]
[EXT2 L0053] - `Training Recipe`
[EXT2 L0054] - `Shortest Honest Tranche`
[EXT2 L0055] - `Kill Criteria` for offline-to-online instability
[EXT2 L0056]
[EXT2 L0057] ## Artifact F03 — Privileged Information Distillation for Language Models
[EXT2 L0058]
[EXT2 L0059] - URL: https://arxiv.org/abs/2602.04942
[EXT2 L0060] - Domain: privileged-information teacher/student distillation
[EXT2 L0061] - Type: primary paper
[EXT2 L0062] - Suggested label: `ext_pass2_privileged_distillation`
[EXT2 L0063]
[EXT2 L0064] Why it matters in pass two:
[EXT2 L0065]
[EXT2 L0066] Pass one already made room for privileged or oracle objects as teacher-only tools. Pass two must decide how that actually turns into a legal student. This paper is useful because it is not just “teacher distillation is good”; it gives a clearer pass-two precedent for a privileged teacher, an unconditioned student, and a training process that keeps the teacher/student split explicit instead of hoping the transfer happens by magic.
[EXT2 L0067]
[EXT2 L0068] Exact pass-two use:
[EXT2 L0069]
[EXT2 L0070] - sharpen the `Teacher Hierarchy`
[EXT2 L0071] - strengthen the `Training Recipe` where oracle or privileged hidden-state teachers are allowed during training but must disappear at deployment
[EXT2 L0072] - encourage an explicit answer on whether Hydra should use joint teacher-student objectives, staged masking, or on-policy self-distillation once the hidden-world teacher exists
[EXT2 L0073]
[EXT2 L0074] What it should influence:
[EXT2 L0075]
[EXT2 L0076] - `Teacher Hierarchy`
[EXT2 L0077] - `Training Recipe`
[EXT2 L0078] - `Where Hydra Is Wrong` if current Hydra underuses privileged-teacher distillation patterns
[EXT2 L0079]
[EXT2 L0080] ## Artifact F04 — Optional calibration-gating appendix slot
[EXT2 L0081]
[EXT2 L0082] - Status: intentionally left as an appendix slot rather than a locked artifact
[EXT2 L0083]
[EXT2 L0084] Rationale:
[EXT2 L0085]
[EXT2 L0086] Pass two may want a selective-classification / abstention paper if the genie strongly recommends confidence-gated search deferral or abstain-to-search behavior. But do not force this into the generated packet unless the exact paper is chosen and actually sharpens the design. The first three artifacts above already add value without dragging the packet back into broad-search mode.
[EXT2 L0087]
[EXT2 L0088] ## Minimal inclusion rule
[EXT2 L0089]
[EXT2 L0090] If the generated pass-two packet needs to stay focused, include only:
[EXT2 L0091]
[EXT2 L0092] 1. `ext_pass2_ece_estimation`
[EXT2 L0093] 2. `ext_pass2_offline_to_online_cpr`
[EXT2 L0094] 3. `ext_pass2_privileged_distillation`
[EXT2 L0095]
[EXT2 L0096] That is enough external pressure for pass two. Everything else should only be added if the generated packet still lacks the evidence needed to decide the winning design, training loop, or gate structure.
```

## Artifact 22 — Pass-two answer contract
Artifact id: `pass-two-output-contract`
Type: `literal`
Why it matters: Keeps the generated packet aligned with the follow-up prompt contract without bloating the shell.

```text
Required top-level sections: Executive Verdict; Best Full Design; Shortest Honest Tranche; Algorithm Family Decision Table; Teacher Hierarchy; Training Recipe; Evaluation Gates; Kill Criteria; Where Hydra Is Wrong; What Stays Open; Minimal Experiment Matrix.
Do not rerun pass-one diagnosis. Produce a ranked design verdict.
Current Hydra support is build-cost evidence, not a veto on stronger designs.
```

</artifacts>
