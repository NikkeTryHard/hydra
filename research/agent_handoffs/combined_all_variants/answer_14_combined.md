<combined_run_record run_id="answer_14" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 14 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_14_VALIDATE_HAND_EV_AND_ENDGAME_EXACTIFICATION.md">
  <![CDATA[# Hydra prompt — validate Hand-EV realism and endgame exactification as long-run separator paths

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a dedicated audit for one of Hydra's biggest live bottlenecks: whether better offensive local evaluation and later-game exactification are among the strongest remaining long-run investments.

Read the core docs holistically first. Do not jump straight from generic endgame or local-evaluator papers to Hydra recommendations.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/TESTING.md`
5. `research/design/SEEDING.md`
6. code-grounding files
7. outside retrieval

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs

Relevant prior answers and prompt references:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md

You are validating whether Hand-EV realism and endgame exactification are genuine long-run separator candidates for Hydra, or just important but bounded second-wave cleanup.

Focus on:
- whether current Hand-EV is too heuristic to support stronger search/distillation
- whether improving Hand-EV realism is one of the strongest next investments
- whether late-game exactification deserves more mainline attention
- how these interact with CT-SMC, world compression, `delta_q`, and AFBS sequencing

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the existing broad findings and use new retrieval only to validate, falsify, or sharpen this Hand-EV / endgame lane.

Assume the prior combined handoffs already established Hand-EV realism as important and endgame exactification as a plausible later path. This prompt should test whether that conclusion is overstated, missequenced, or blocked by repo reality.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- It is acceptable to conclude that the path is important but not a separator.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, evaluator definitions, tensor/interface details, thresholds, or benchmark details when they matter.
- When in doubt, include more mathematical and mechanism detail rather than less.
</verbosity_controls>

<calculation_validation_rules>
- Use Python in bash for evaluator-cost accounting, error-budget arithmetic, latency comparisons, and any claim about exactification frontier or compression benefits.
- Do not leave numerical feasibility claims uncomputed when they can be checked quickly.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart broad Hydra future search.
- New retrieval should only validate, falsify, or sharpen Hand-EV realism and endgame exactification.
</tool_persistence_rules>

<dependency_checks>
- Verify what Hand-EV computes today, what is public-count based versus CT-SMC-weighted, and what endgame exactification already exists.
- Verify whether current runtime and bridge plumbing make Hand-EV realism upgrades and endgame upgrades actually insertable.
- Verify whether any proposed teacher/export path depends on labels or evaluators Hydra does not yet have.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in the provided docs/code.
- Mark any unevidenced runtime hook, label path, or evaluator quantity as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Is Hand-EV realism just “good hygiene,” not a separator?
  - Does improved Hand-EV only help because AFBS is still underpowered, making it a temporary crutch?
  - Does endgame exactification help only in a narrow late-game slice, limiting total upside?
  - Are posterior-quality issues upstream of both Hand-EV realism and exactification?
  - Does this path beat the strongest simpler alternative, or is it just the obvious next cleanup?
</self_red_team_rules>

<minimum_falsification_rules>
- Define the minimum offline and runtime benchmarks that would prove Hand-EV realism or endgame exactification is worth serious mainline attention.
- Reject any proposed path that cannot show a narrow benchmark advantage before broader integration.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a broad offense/search survey.
- Stay inside Hand-EV realism, endgame exactification, and their direct sequencing consequences.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current doctrine for Hand-EV, endgame exactification, AFBS, and CT-SMC sequencing.
2. Validate the current Hand-EV and endgame runtime surfaces.
3. Evaluate whether better Hand-EV realism is one of the strongest remaining long-run investments.
4. Evaluate whether stronger endgame exactification is a separator, second-wave investment, or just localized cleanup.
5. Write down the strongest surviving upgrade paths with exact math, interfaces, and falsification plans.

## Deliverables
1. Hydra posture reconstruction for Hand-EV / endgame / CT-SMC / AFBS sequencing
2. Current repo surfaces and missing pieces
3. Best surviving Hand-EV realism upgrade paths
4. Best surviving endgame exactification upgrade paths
5. Exact math / evaluator definitions / tensor-interface notes
6. Dependency closure table
7. Minimum falsifiable benchmark plan
8. Failure modes and kill criteria
9. Final recommendation:
   - separator path, second-wave path, or cleanup only
   - what to try first
   - what to defer

## Hard constraints
- no broad architecture reset
- no pretending Hand-EV or endgame exactification automatically solves posterior quality
- no vague “improve local evaluator” answers without concrete operator changes and benchmarks
- no recommendation that cannot be inserted into current Hydra surfaces or clearly marked `[blocked]`
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_14.md">
  <![CDATA[## 1. Hydra posture reconstruction for Hand-EV / endgame / CT-SMC / AFBS sequencing

Hydra’s current doctrine is narrower than the original ambition in `HYDRA_FINAL.md`. The settled sequence in `HYDRA_RECONCILIATION.md` is: keep one unified belief stack (Mixture-SIB for amortized belief, CT-SMC for search-grade posterior), close advanced target/supervision loops first, then make Hand-EV real, then treat deeper AFBS and stronger endgame work as selective specialist extensions rather than immediate mainline expansion. In other words, the repo’s governing memo does **not** say “search first”; it says “supervision closure first, Hand-EV realism next, deeper AFBS later.” (research/design/HYDRA_RECONCILIATION.md:52-55,117-160,187-205,266-298,535-539)

Within that doctrine, Hand-EV is supposed to be a serious offensive local evaluator, not a decorative heuristic. `HYDRA_FINAL.md` defines Group D as per-discard features with three tenpai planes, three win-probability planes, one expected-score plane, and 34 ukeire planes, computed from a CPU-side hand analyzer using belief-weighted counts, and it explicitly cites Suphx-style look-ahead features as a major practical lever. At the design level, Hand-EV is therefore intended to be a **real local oracle** on the fixed 192×34 observation surface, not a placeholder until AFBS matures. (research/design/HYDRA_FINAL.md:78-87)

CT-SMC’s intended role is also clear. Hydra does not want a separate third belief system. Mixture-SIB is the amortized public-policy belief object; CT-SMC is the exact contingency-table posterior used when search-grade or calibration-grade hidden-state reasoning is needed. The reconciliation memo and `HYDRA_FINAL.md` both treat CT-SMC as the posterior object that should back stronger local evaluation, harder-state search, and selective late-game reasoning, subject to posterior validation gates. (research/design/HYDRA_FINAL.md:121-162; research/design/HYDRA_RECONCILIATION.md:206-208,266-274)

AFBS is sequenced as a **specialist** tool, not the thing that should swallow the roadmap. The docs still reserve a place for `delta_q`, exit-policy distillation, and pondering, but the reconciled plan is explicit that Hydra should not reopen a broad public-belief-search push before the repo’s existing loops are closed and Hand-EV is made less fake. Prior handoffs are consistent on this point: the immediate tranche is supervision closure, then Hand-EV realism, then selective AFBS/endgame work. (research/design/HYDRA_RECONCILIATION.md:120-160,187-205,285-298,535-539)

Endgame exactification survives in doctrine, but as a bounded late-game specialist. `HYDRA_FINAL.md` keeps wall-short exactification as a ceiling-raising addition, but it also explicitly warns that full expectimax is too slow and recommends PIMC/top-k draw pruning instead. So even at the design level, “exactification” was never “solve the whole multiplayer game tree”; it was “be much more exact, much later, and only where the wall is short enough that correlations and placement pressure dominate.” (research/design/HYDRA_FINAL.md:202-217)

One more sequencing fact matters from the inference-server file you pasted: the live consumer is a prebuilt 192×34 observation tensor or a cached `PonderResult`. That makes Hand-EV a direct encode-time lever, but it makes endgame a search/ponder-side lever unless some upstream caller explicitly invokes it. That asymmetry is important later.

---

## 2. Current repo surfaces and missing pieces

The current Hand-EV implementation is much more heuristic than the doctrine implies. The live struct is exactly `tenpai_prob: [[f32;3];34]`, `win_prob: [[f32;3];34]`, `expected_score: [f32;34]`, and `ukeire: [[f32;34];34]`. `ukeire` is structurally reasonable: for each discard it counts shanten-improving draws against a count vector. But win and score are not computed by a real local offensive DP. The code uses acceptance ratio, a `continuation_boost` heuristic, a fallback `0.35 * acceptance_ratio` win floor, and a hand-value formula built from pair/triplet/flush/honor/diversity bonuses. Worse, the docs say the score plane should be `E[score | win, a]`, but the implementation stores `win_prob[discard][2] * score_estimate`, which is an unconditional probability-weighted heuristic, not the documented conditional value. That semantic mismatch alone is enough to say the current Hand-EV is too weak to serve as a serious teacher or search leaf. (hydra-core/src/hand_ev.rs:6-10,24-43,151-222,253-309)

The bridge wiring is real, but what CT-SMC contributes today is only a first-moment collapse. `SearchContext` already has optional `mixture`, `ct_smc`, `afbs_tree`, `afbs_root`, and optional risk/stress overrides. `encode_observation_with_search_context()` really does compute Hand-EV and Group C features before handing the result to the encoder. But `compute_ct_smc_hand_ev()` does **not** evaluate per-particle worlds. It sums `ct_smc.weighted_mean_tile_count(tile, col)` across columns into a single `[f32;34]` remaining-count vector and then calls the same heuristic `compute_hand_ev()` as the public-count path. So current “CT-SMC Hand-EV” means “heuristic Hand-EV on posterior first moments,” not “particle-weighted offensive evaluation.” (hydra-core/src/bridge.rs:27-45,251-299)

The tensor surface is fixed and narrower than the architecture prose can make it sound. Hydra’s live observation is 192×34 = 6,528 floats. Group D is exactly 42 channels: 3 tenpai planes, 3 win planes, 1 score plane, 34 ukeire planes, 1 mask plane. Group C is 65 channels, including a **single** discard-level `delta_q` channel whose 34 cells correspond to tile actions, not the full 46-action space. So Hand-EV is inherently discard-centric on the current surface, and search distillation reaching the encoder is also discard-centric at runtime even though the global action space is 46. There is no spare channel budget here for a more elaborate local-evaluator interface without repurposing existing semantics. (docs/GAME_ENGINE.md:122-170)

The training path is one of the biggest reality checks. `mjai_loader.rs` builds observations with `encode_observation(...)`, not the search-context version, so training examples currently include **public-count Hand-EV**, not CT-SMC-enriched Hand-EV. The same loader does create Stage-A belief targets from public remaining counts and hidden-tile counts, and it does generate replay-safe safety residuals, but that is not the same thing as a closed search/teacher loop. In `sample.rs`, the batch path clones `belief_fields_target` and `mixture_weight_target`, but sets `opponent_hand_type_target: None` and `delta_q_target: None`. In losses, the advanced targets exist structurally, yet the default weights for belief, mixture, delta-Q, and safety residual are zero. So the repo is still in the state the reconciliation memo warned about: advanced surfaces exist, but the mainline training loop does not yet fully consume or supervise them. (hydra-train/src/data/mjai_loader.rs:303-321,360-410; hydra-train/src/data/sample.rs:156-219; hydra-train/src/training/losses.rs:33-60)

That training/runtime split matters directly for Hand-EV. A better **public-count** Hand-EV can be dropped into the current loader and runtime encoder immediately, because both already use `encode_observation()`. A better **CT-SMC world-aware** Hand-EV cannot be treated as an immediate drop-in model-input upgrade, because the current training loader does not generate search-context observations. Without a training-time CT-SMC re-encode path, shipping world-aware Hand-EV only at inference would introduce a train/infer feature shift. That does not kill the long-run idea, but it absolutely kills the claim that it is a frictionless near-term mainline insertion.

The endgame surface is real but thinner than “exactification” suggests. `EndgameSolver` has `max_wall=10`, `mass_threshold=0.95`, `should_activate(wall_remaining, has_threat)`, and `solve_with_particles(...)`, but the solver itself just selects top-mass particles and computes a weighted average of `eval_fn(particle, action)` over legal actions. There is no recursion, no opponent branch model, and no exact transition logic in `endgame.rs`. So current endgame is a particle aggregator around an externally supplied leaf evaluator, not a true exact late-game solver. (hydra-core/src/endgame.rs:6-18,23-87,90-184)

There is also a live-wiring asymmetry. Hand-EV is definitely on the encode path. Endgame is not definitely on the live inference path from the evidence in scope. In the inference-server file you provided, the server either uses a cached `PonderResult.exit_policy` or runs the actor+SaF fast path on a prebuilt observation tensor. That gives endgame a plausible insertion point through pondering/cached exit policy, but a direct live caller from mainline inference to `endgame.rs` is not evidenced in the inspected materials. I would therefore mark “endgame exactifier is already in the live mainline action path” as **[blocked by evidence]**.

---

## 3. Best surviving Hand-EV realism upgrade paths

### H1a. Exact one-step Hand-EV semantic repair on the current surface

This is the strongest **immediate** Hand-EV upgrade that survives strict validation.

What survives is not “rewrite Hand-EV into a giant local solver.” What survives is a bounded, concrete replacement of the fake parts of Group D while keeping the exact same `HandEvFeatures` interface. Concretely:

* keep the current 42-channel layout;
* keep current exact-ish ukeire computation structure;
* replace heuristic `win_prob[:,0]` with exact one-draw tsumo probability from the current count model;
* replace heuristic `tenpai_prob[:,0]` with exact one-draw-to-tenpai probability after optimal continuation discard;
* repair the score plane to actually mean a conditional win value, not `P(win)*heuristic_score`;
* only derive horizons 2 and 3 from that exact one-step base in the first stage.

This survives because it is insertable **today**. It requires only `hand_ev.rs` and bridge-side use of the same output struct. It also reaches current training immediately on the public-count path because the loader already uses `encode_observation()`. It does **not** require new heads, new channels, or a closed `delta_q` teacher path. (hydra-core/src/hand_ev.rs:6-10,253-309; hydra-train/src/data/mjai_loader.rs:360-410; hydra-train/src/data/sample.rs:156-219)

What failed inside this lane: plain coefficient retuning does not survive. It leaves the semantic mismatch untouched, still collapses posterior worlds to one heuristic count vector, and still makes the score plane mean the wrong thing. That is hygiene, not a serious investment.

What also failed here: an immediate ron model inside Hand-EV. Hydra does not currently show a concrete runtime hook for “probability an opponent discards our wait before horizon d” that is both local and posterior-consistent. The repo has safety features, tenpai targets, opp-next targets, and stress/risk planes, but not a clean offensive ron-hazard API in the inspected path. I would mark immediate ron modeling inside Hand-EV as **defer**, not because it is unimportant, but because it is not concrete enough under the current surfaces.

### H1b. CT-SMC world-aware Hand-EV with representative integer worlds

This is the only Hand-EV path that still looks even **potentially** separator-like after the strict pass.

The reason is mathematical, not rhetorical. For one-step probabilities, a first-moment CT-SMC count vector is usable: a fractional count vector still defines a valid categorical draw distribution for “one more draw.” But for multi-draw exactification, first moments are the wrong object. Without-replacement recursion is naturally defined on integer remaining multisets. CT-SMC particles already give those integer worlds. So genuine multi-draw Hand-EV realism wants selected representative particles, not `weighted_mean_tile_count` collapsed to one vector. That is the point where Hand-EV stops being “cleaner heuristics” and starts being a real posterior-aware local evaluator.

This path survives because the insertion surface is real: `SearchContext.ct_smc` exists, bridge dispatch already chooses a CT-SMC path when the posterior is present, and Group D is already wired into the model input. It also survives because Suphx-style local look-ahead really can matter. But it only survives **conditionally**. The repo reality forces four gates:

1. CT-SMC posterior quality must already pass its own validation gates.
2. A world selector/compressor must beat first-moment and naive top-mass baselines on a regret-vs-calls frontier.
3. Hydra must resolve the train/infer feature-parity issue if the world-aware evaluator is going to be a model input.
4. The first promoted version should remain discard-centric and local; it should not pretend to solve opponent dynamics. (research/design/HYDRA_FINAL.md:78-87,121-162,202-217)

What failed inside this lane: immediate runtime value-directed compression as the **first** integration. The attractive version is decision-focused medoid compression in evaluator/regret space, but that requires a prepass over many worlds using some local evaluator, and that prepass can erase the savings unless the surrogate is extremely cheap. So value-directed compression survives as an **offline benchmark and second-stage runtime candidate**, not as the first thing to ship.

What failed here too: “use top-mass 95% particles and call it done.” Hydra’s own docs say that usually means roughly 50–100 particles. That is only about a 1.28× to 2.56× reduction versus 128 particles, which is not enough to transform a genuinely expensive evaluator, and it provides no decision-quality certificate. It is a baseline selector, not a separator.

---

## 4. Best surviving endgame exactification upgrade paths

### E1. Endgame leaf exactification inside the current particle shell

The best surviving endgame path is **not** “build a new exact late-game solver.” It is: keep the existing `EndgameSolver` shell, and make its `eval_fn(&Particle, action)` much more exact in the late game.

That survives because the shell already exists, takes the full 46-action legal mask, and aggregates over posterior particles. If Hydra strengthens Hand-EV into a real local tsumo evaluator, the same evaluator can become the late-game leaf under each selected world. This is the cleanest reuse point in the codebase. It is also where endgame exactification most honestly fits Hydra’s architecture: not as a separate planner stack, but as a better late-game leaf within a selective particle-based wrapper. (hydra-core/src/endgame.rs:6-18,76-87,136-184)

The other thing that survives here is placement-aware utility. Endgame in four-player riichi is not about raw round EV only. The loader already computes placement labels from final scores, so the repo clearly recognizes placement semantics. A late-game leaf that only optimizes expected point gain is misaligned with the use-case. The natural late-game utility is therefore some monotone function of placement and score delta, captured by the caller and passed through the `eval_fn` closure. The closure interface allows that. The direct core implementation is still **[inference]**, because the inspected `endgame.rs` file does not show a scorer/score-context hook, but the surface is compatible. (hydra-train/src/data/mjai_loader.rs:165-202,334-399)

What failed: full multiplayer exact endgame solving over wall≤10. That fails on three grounds at once. First, Hydra’s own docs already say full expectimax is too slow and recommend PIMC/top-k pruning. Second, four-player general-sum partial observability makes “exact” far more than a draw-tree problem. Third, the current module does not even contain the transition semantics needed for that claim. So “true exactification” does not survive the strict pass; only “leaf exactification inside the existing shell” does. (research/design/HYDRA_FINAL.md:202-217)

### E2. Ponder/AFBS deployment, not fast-path mainline deployment

The inference-server file you supplied makes the most plausible deployment path clear: expensive late-game work should land through pondering/cached `PonderResult.exit_policy`, not by bloating the fast network path. The server already has a slow-path reuse mechanism keyed by info-state hash. That matches the doctrine: AFBS and late-game exactification are specialist tools. So stronger endgame survives as a **ponder/search-side specialist**, not as an every-turn mainline feature rewrite.

What failed here: “make endgame exactification a mainline separator before Hand-EV and supervision closure.” That does not survive repo reality. Its live caller is not evidenced, its teacher/export path is not closed, and its slice of game states is inherently narrow. Even if the late-game upside is real, it is still later and narrower than Hand-EV realism on current surfaces.

---

## 5. Exact math / evaluator definitions / tensor-interface notes

### 5.1 Current Hand-EV math in the repo

Let `h` be the 34-bin hand-count vector, `a` a discard tile type, `h_a = h - e_a`, and `r ∈ R_+^34` the remaining-count vector used by the evaluator.

The current code computes:

[
u_a(t) = r_t \cdot \mathbf{1}{sh(h_a + e_t) < sh(h_a)}
]

[
A_a = \sum_t u_a(t), \qquad R = \sum_t r_t, \qquad \rho_a = \mathrm{clip}(A_a / R, 0, 1)
]

For horizon-1 tenpai:

[
P^{(1)}_{\text{tenpai}}(a) =
\begin{cases}
1 & \text{if } sh(h_a) \le 0 \
\rho_a & \text{otherwise}
\end{cases}
]

For horizon-1 win:

[
P^{(1)}_{\text{win}}(a) =
\begin{cases}
1 & \text{if } sh(h_a) < 0 \
\frac{\sum_t r_t \mathbf{1}{sh(h_a + e_t) < 0}}{R} & \text{if } sh(h_a)=0 \
0 & \text{otherwise}
\end{cases}
]

Then it defines

[
\text{base_win}(a) = \max!\big(P^{(1)}_{\text{win}}(a),\ 0.35,\rho_a\big)
]

and uses a heuristic continuation term

[
c(d,s,\rho) = \mathrm{clip}\big(\rho \cdot \text{horizon_scale}(d)\cdot \text{shanten_scale}(s), 0, 1\big)
]

to build horizons 2 and 3 as:

[
P^{(d)}*{\text{tenpai}}(a) =
1 - \big(1 - P^{(1)}*{\text{tenpai}}(a)\big)^d \cdot \big(1 - c(d-1, sh(h_a), \rho_a)\big)
]

[
P^{(d)}_{\text{win}}(a) =
1 - \big(1 - \text{base_win}(a)\big)^d \cdot \big(1 - c(d-1, sh(h_a)-1, \rho_a)\big)
]

Finally it builds a heuristic hand-value estimate from suit mix, honor count, pair/triplet bonuses, flush bias, tile concentration, diversity penalty, and an honor-discard bonus, and stores

[
\text{expected_score}[a] = P^{(3)}_{\text{win}}(a)\cdot \text{score_estimate}(h,a).
]

That last line is the semantic mismatch with the doc’s intended `E[score \mid win, a]`. (hydra-core/src/hand_ev.rs:303-309)

### 5.2 Proposed H1a exact one-step replacement

For the first promoted upgrade, do not change the tensor interface. Change the evaluator semantics.

Let the evaluator consume either one count vector `r` or a selected weighted set of worlds
[
\mathcal{W} = {(r^{(k)}, w_k)}_{k=1}^K, \qquad \sum_k w_k = 1.
]

For one-step tsumo look-ahead, define:

[
U_a^{(k)}(t) = r_t^{(k)} \cdot \mathbf{1}{sh(h_a + e_t) < sh(h_a)}
]

[
P^{(1,k)}_{\text{win}}(a)
= \sum_t \frac{r_t^{(k)}}{R_k},\mathbf{1}{\mathrm{agari}(h_a + e_t)}
]

[
P^{(1,k)}*{\text{tenpai}}(a)
= P^{(1,k)}*{\text{win}}(a)

* \sum_t \frac{r_t^{(k)}}{R_k},\mathbf{1}{\neg \mathrm{agari}(h_a + e_t)}
  \max_{b \in \mathcal{D}(h_a + e_t)} \mathbf{1}{sh(h_a + e_t - e_b)=0}
  ]

where (\mathcal{D}(h_a+e_t)) is the set of legal continuation discards after the draw.

Define score mass

[
S^{(1,k)}(a)
= \sum_t \frac{r_t^{(k)}}{R_k},\mathrm{score}(h_a+e_t,\text{ctx}),
\mathbf{1}{\mathrm{agari}(h_a+e_t)}
]

and aggregate

[
P^{(1)}*{\text{win}}(a)=\sum_k w_k P^{(1,k)}*{\text{win}}(a), \qquad
P^{(1)}*{\text{tenpai}}(a)=\sum_k w_k P^{(1,k)}*{\text{tenpai}}(a)
]

[
\mathrm{EscoreCond}(a)=
\frac{\sum_k w_k S^{(1,k)}(a)}
{\max(P^{(1)}_{\text{win}}(a), \varepsilon)}
]

[
\mathrm{ukeire}[a,t] = \sum_k w_k U_a^{(k)}(t).
]

This exactly fixes the semantic center of Group D for horizon 1. For H1a, horizons 2 and 3 can be filled from the exact one-step base with a simple cumulative continuation rule,
[
P^{(d)} = 1-(1-P^{(1)})^d,
]
which is still approximate, but far less fake than the current continuation-boost machinery. H1a is therefore “exact one-step, approximate multi-step,” which is a fair and concrete first promotion. The scoring hook inside `hand_ev.rs` is still **[inference]** in the inspected files, though the design docs clearly assume a scoring engine exists. (research/design/HYDRA_FINAL.md:80-87)

### 5.3 Proposed H1b world-aware multi-draw extension

H1b is the real long-run candidate.

Use selected integer CT-SMC worlds (X^{(k)}), with remaining counts
[
r_t^{(k)} = \sum_{\text{hidden cols}} X^{(k)}_{t,\text{col}},
]
and recurse on those integer multisets rather than on posterior first moments.

For horizon (d>1), define a continuation policy (\pi_d) over 14-tile post-draw states and a top-(M) draw set (T_M) for branch pruning. Then the pruned recursion is

[
W_d(h_a, r^{(k)}) =
\sum_{t\in T_M} \frac{r_t^{(k)}}{R_k},\beta_t

* (1-Z_M),\widehat{W}_{d,\text{fallback}},
  ]

where
[
Z_M = \sum_{t\in T_M} \frac{r_t^{(k)}}{R_k},
]
and
[
\beta_t =
\begin{cases}
1 & \text{if } \mathrm{agari}(h_a+e_t) \
W_{d-1}(h', r^{(k)}-e_t) & \text{if } h'=\pi_d(h_a+e_t, r^{(k)}-e_t)
\end{cases}
]

with analogous recurrences for tenpai and score mass.

The important point is not the exact pruning scheme; it is that the state object must now be an integer world, not a weighted-mean count vector, if the recursion is supposed to mean anything exact-ish.

### 5.4 Compute sanity checks

From the current `hand_ev.rs` structure, one per-world evaluation costs about **36–70 shanten calls per distinct discard**: one base shanten call plus up to 34 inside `compute_ukeire`, one more for `shanten_after`, and another up to 34 if `immediate_win_probability` runs. That means:

* at 14 distinct discards and 128 worlds, naive full-particle Hand-EV is about **64,512–125,440 shanten calls**;
* at 14 discards and 8 worlds, it is about **4,032–7,840 shanten calls**.

For branch counts, exact one-step world-aware evaluation is manageable. With (K=8) worlds, (D=14) discard types, and (U=20) live draw types, the one-step state count is only
[
KDU = 2240.
]
But naive three-step branching is not:
[
KD(U + U^2 + U^3)=943{,}040
]
for the same (K,D,U). Top-(M) pruning changes that drastically: with (M=3), the branch count drops to **4,368**; with (M=5), to **17,360**. So H1b only survives as a pruned or sampled continuation, not as naive full branching.

For endgame, the current shell itself is cheap:
[
\text{leaf calls} = A \times W
]
for (A) legal actions and (W) selected worlds. With (A=14), (W=50) means 700 leaf calls; (W=100) means 1400. The problem is not the shell. The problem is the leaf cost. At 10 ms total budget, a 50 µs leaf only fits about 14 worlds for 14 actions; a 100 µs leaf fits only about 7. So any real endgame leaf exactification still needs either aggressive world reduction, a very cheap leaf, a ponder/off-turn path, or all three.

### 5.5 Tensor/interface notes

The hard interface facts are:

| Surface                   | Shape / meaning                                                               |
| ------------------------- | ----------------------------------------------------------------------------- |
| Observation tensor        | `192 × 34 = 6528` floats                                                      |
| Group D Hand-EV           | 42 channels = 3 tenpai + 3 win + 1 score + 34 ukeire + 1 mask                 |
| `HandEvFeatures`          | `[[f32;3];34]`, `[[f32;3];34]`, `[f32;34]`, `[[f32;34];34]`                   |
| Group C `delta_q` feature | one **34-cell discard plane**, not full 46-action tensor                      |
| Full action space         | 46 actions                                                                    |
| `SearchContext`           | optional `mixture`, `ct_smc`, `afbs_tree`, `afbs_root`, risk/stress overrides |
| Endgame shell             | `solve_with_particles(&[Particle], &[bool;46], &dyn Fn(&Particle,u8)->f32)`   |

The fixed-shape consequence is simple: Hand-EV upgrades must reuse the existing 42-plane semantics, and any claim that this path directly upgrades full-action search semantics is overstated. Hand-EV is primarily a discard-axis local evaluator on current Hydra surfaces. ([GitHub][5])

---

## 6. Dependency closure table

| Item                                               | Already present                                                                 | Missing closure                                                                                              | Status                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| H1a exact one-step Hand-EV on public counts        | `hand_ev.rs`, `bridge.rs`, 42-plane encoder, loader uses `encode_observation()` | score hook inside `hand_ev.rs` is **[inference]** from docs, not shown directly                              | **Viable now**                       |
| H1a exact one-step Hand-EV on CT-SMC first moments | `SearchContext.ct_smc`, bridge CT-SMC path                                      | none for horizon-1 probabilities; multi-draw exactness still absent                                          | **Viable now**                       |
| H1b world-aware multi-draw Hand-EV                 | CT-SMC particles, bridge dispatch hook, fixed Group D surface                   | representative-world selection, recursion/caching, train/infer parity for CT-SMC search-context observations | **Second-stage only**                |
| Training parity for world-aware Hand-EV            | loader and encoder exist                                                        | loader does not build search-context observations; no CT-SMC re-encode path in scope                         | **[blocked] for immediate mainline** |
| Direct Hand-EV / endgame distillation targets      | losses support optional advanced targets                                        | `delta_q_target` dropped in batch path; advanced loss weights default zero; no Hand-EV teacher path          | **[blocked]**                        |
| Endgame leaf exactification                        | `endgame.rs` shell and particle interface exist                                 | better leaf evaluator; placement-aware utility hook; direct live caller not evidenced                        | **Specialist viable**                |
| Live endgame deployment                            | ponder cache / cached exit policy path exists in provided inference file        | explicit caller wiring from late-game logic to `endgame.rs` or ponder path not shown                         | **[inference]**                      |
| Value-directed runtime world compression           | CT-SMC particles exist                                                          | cheap enough surrogate metric and benchmark win over simpler baselines                                       | **Benchmark first, not ship first**  |

The table above is the narrow closure picture implied by the repo and the provided inference file. The main takeaways are: public/one-step Hand-EV is directly insertable; world-aware Hand-EV is real but not frictionless; endgame is specialist and not yet evidently on the live path. ([GitHub][6])

---

## 7. Minimum falsifiable benchmark plan

### Phase A: posterior prerequisite for any world-aware claim

Before promoting H1b or any endgame exactification claim that relies on CT-SMC worlds, require Hydra’s own posterior gates from `HYDRA_FINAL.md` / reconciliation to pass on the evaluation stack being used. If Gate A / Gate B posterior validation fails, stop. Otherwise any improvement from world-aware Hand-EV or endgame is just “more faithfully wrong posterior usage.” ([GitHub][2])

### Phase B: H1a exact-one-step Hand-EV benchmark

Build a discard-only replay suite with exact hidden world available. For each state and legal discard, compute a hidden-world oracle for:

* (P_{\text{win}}^{(1)}(a))
* (P_{\text{tenpai}}^{(1)}(a))
* (\mathrm{E}[score \mid win, a])
* `ukeire[a,t]`

using the true remaining multiset from replay.

Compare current Hand-EV vs H1a on:

[
\text{MSE}*{\text{win}},\ \text{MSE}*{\text{tenpai}},\ \text{MSE}*{\text{score}},\ \text{MSE}*{\text{ukeire}}
]

and on ranking metrics:

[
\text{Top1Match},\qquad
\text{Regret}(s)=Q^*(s,a^*)-Q^*(s,\hat a).
]

Promotion gate for serious mainline attention:

* mean regret reduction **≥ 10%** vs current Hand-EV;
* top-1 discard match **≥ 3 percentage points** better than current Hand-EV;
* non-regression on a riichi-threat slice;
* encode-time wall clock **≤ 1.25×** current public Hand-EV encode cost.

If H1a does not clear that, kill it. It is then not even a good second-wave upgrade.

### Phase C: H1b world-aware / compression frontier

Use posterior-sensitive states only: low wall, high entropy, low ESS, and states where first-moment counts and full particles disagree.

Take a strong but expensive reference:
[
Q_{\text{full}}(a)=\sum_{i=1}^{P} w_i q_i(a)
]
under the improved local evaluator.

Compare:

* first-moment baseline;
* top-mass baseline;
* any representative-world selector with (K \in {4,8,16}).

Use

[
\text{Regret}*C(s)=
\max_a Q*{\text{full}}(s,a) -
Q_{\text{full}}\big(s,\arg\max_a Q_C(s,a)\big)
]

and evaluator-call ratio

[
\text{CallRatio}_C = \frac{\text{measured calls or time}*C}{\text{calls or time}*{\text{full}}}.
]

Promotion gate:

* some (K \le 8) or (16) must achieve
  [
  \text{BenefitRatio}_C =
  1 - \frac{\mathbb{E}[\text{Regret}*C]}{\mathbb{E}[\text{Regret}*{\text{first-moment}}]}
  \ge 0.9
  ]
  while
  [
  \text{CallRatio}_C \le 0.25;
  ]
* it must beat plain first-moment counts on the posterior-sensitive slice, not just tie them.

If no such point exists, kill H1b as a mainline path and demote Hand-EV realism to H1a-level cleanup plus semantics repair.

### Phase D: endgame specialist benchmark

Restrict to `wall_remaining <= 10 && has_threat`, with a dedicated orasu / close-placement slice.

Benchmark the current endgame shell with current leaf vs the same shell with improved leaf. Require:

* positive paired improvement in placement-aware utility on the full late-game suite;
* positive paired improvement on the orasu / close-gap slice specifically;
* non-worse deal-in rate;
* standalone specialist p95 runtime within the late-game budget actually granted by the caller. Since the live caller is not evidenced in scope, measure standalone p95 for now and do not promote past offline status until the caller exists.

If improved late-game leaf wins only on raw round EV but not placement-aware utility, kill it. That is exactly the kind of false positive four-player general-sum endgames produce.

---

## 8. Failure modes and kill criteria

1. **Posterior-quality upstream failure.**
   If CT-SMC calibration is weak, world-aware Hand-EV and endgame exactification should both be deferred. They amplify posterior mistakes; they do not fix them. ([GitHub][2])

2. **Train/infer feature-shift failure for CT-SMC Hand-EV.**
   If Hydra cannot generate training-time search-context observations, do not ship world-aware Group D as a model-input change. Keep it offline or search-side only.

3. **H1a fails the exact-one-step oracle benchmark.**
   Then Hand-EV realism is not ready for mainline attention. If it cannot beat the current heuristic on the local quantity it is supposed to predict, stop.

4. **H1b fails the regret-vs-calls frontier.**
   If representative worlds do not beat first-moment counts by a clear margin at materially lower cost, kill the world-aware separator claim.

5. **Action-sufficient runtime compression prepass erases its own savings.**
   Then it stays offline only. Do not ship a “smart compressor” whose prepass dominates the total cost.

6. **Endgame leaf wins locally but not in placement-aware late-game utility.**
   Then it is not a useful riichi endgame improvement; defer it.

7. **No live caller for endgame.**
   Until endgame is wired either into pondering or another explicit caller, it remains a reserve-shelf module, not a mainline investment.

8. **AFBS/delta-q closure later absorbs the gain.**
   If after search-target closure the marginal value of improved Hand-EV collapses, then the path was an interim crutch, not a separator. The repo today is not yet in a position to know that for sure, which is another reason not to overclaim.

Failed ideas that do **not** survive the strict pass:

* coefficient retuning of current Hand-EV;
* immediate ron modeling inside Hand-EV;
* exact multi-draw DP on CT-SMC weighted-mean counts;
* full multiplayer endgame expectimax;
* direct Hand-EV/endgame distillation now;
* runtime value-directed compression as the first shipped step.

---

## 9. Final recommendation

**Overall classification: second-wave path.**

More precisely: the broad claim “Hand-EV realism plus endgame exactification is one of Hydra’s biggest long-run separator paths” is **too strong as stated**. After a stricter repo pass, the lane survives, but only in a narrower form.

The part that survives strongly is **Hand-EV realism**, and even there the immediate concrete move is **not** the full long-run dream. The first thing to try is **H1a**: repair Group D to be an exact one-step local evaluator on the current surface, fix the score-plane semantics, and benchmark it against exact hidden-world one-step oracles. That is concrete, directly insertable, reaches current training immediately on the public-count path, and can be falsified cleanly. After that, and only if posterior validation and compression benchmarks clear, Hydra should test **H1b**: representative-world CT-SMC Hand-EV. That is the only subpath here that still has real separator-like upside, because it is the only one that turns Hand-EV from “better heuristics on one count vector” into “posterior-aware local offense.” ([GitHub][1])

**Endgame exactification is later and narrower.** It survives as a **specialist path**, not as a current mainline separator. The practical form that survives is late-game leaf exactification inside the existing particle shell, likely deployed through pondering/cached exit policies rather than the fast path. It should be deferred until H1a is validated and until Hydra either closes more of the search-target loop or at least wires an explicit caller path. Full “exact endgame solving” does not survive the repo and compute realities in scope. ([GitHub][2])

So the narrowed action order is:

1. **Try first:** H1a exact one-step Hand-EV semantic repair on the existing 42-plane surface, with exact-one-step oracle benchmarking.
2. **Try second, only if H1a wins and CT-SMC gates pass:** H1b representative-world CT-SMC Hand-EV, benchmarked on a regret-vs-calls frontier before any mainline promotion.
3. **Defer:** ron modeling inside Hand-EV, direct Hand-EV/endgame teacher export, value-directed runtime compression as the first shipped step, and stronger endgame exactification beyond a leaf/specialist implementation.

That is the narrowest answer I think the evidence supports: **not cleanup only, not a current separator path, but a strong second-wave lane whose only genuine separator candidate is world-aware Hand-EV, not endgame.**

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/hand_ev.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/hand_ev.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[6]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs"
[7]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
