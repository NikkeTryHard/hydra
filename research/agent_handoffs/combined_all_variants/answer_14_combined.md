<combined_run_record run_id="answer_14" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 14 + returned agent answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_14_VALIDATE_HAND_EV_AND_ENDGAME_EXACTIFICATION.md">
<![CDATA[# Hydra prompt — validate Hand-EV realism and endgame exactification as long-run separator paths

Primary source material in raw GitHub links below.

## Critical directive

Dedicated audit for one of Hydra's biggest live bottlenecks: whether better offensive local evaluation + later-game exactification rank among strongest remaining long-run investments.

Read core docs holistically first. Do not jump straight from generic endgame or local-evaluator papers to Hydra recommendations.

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

Relevant prior answers + prompt references:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md

Validate whether Hand-EV realism + endgame exactification are genuine long-run separator candidates for Hydra, or important but bounded second-wave cleanup.

Focus on:
- whether current Hand-EV too heuristic to support stronger search/distillation
- whether improving Hand-EV realism is one of strongest next investments
- whether late-game exactification deserves more mainline attention
- how these interact with CT-SMC, world compression, `delta_q`, and AFBS sequencing

Broad exploration already done in `research/agent_handoffs/combined_all_variants/`. Do not redo broad work. Start from existing broad findings; use new retrieval only to validate, falsify, or sharpen this Hand-EV / endgame lane.

Assume prior combined handoffs already established Hand-EV realism as important and endgame exactification as plausible later path. This prompt should test whether that conclusion is overstated, missequenced, or blocked by repo reality.

<output_contract>
- Return exactly requested sections, in requested order.
- Be as detailed + explicit as needed; do not optimize for brevity.
- Return full technical treatment, not compressed memo.
- OK to conclude path important but not separator.
- Short answer usually failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, evaluator definitions, tensor/interface details, thresholds, or benchmark details when they matter.
- When in doubt, include more math + mechanism detail, not less.
</verbosity_controls>

<calculation_validation_rules>
- Use Python in bash for evaluator-cost accounting, error-budget arithmetic, latency comparisons, and any claim about exactification frontier or compression benefits.
- Do not leave numerical feasibility claims uncomputed when quick to check.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart broad Hydra future search.
- New retrieval should only validate, falsify, or sharpen Hand-EV realism + endgame exactification.
</tool_persistence_rules>

<dependency_checks>
- Verify what Hand-EV computes today, what is public-count based vs CT-SMC-weighted, and what endgame exactification already exists.
- Verify whether current runtime + bridge plumbing make Hand-EV realism upgrades + endgame upgrades insertable.
- Verify whether any proposed teacher/export path depends on labels or evaluators Hydra does not yet have.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in provided docs/code.
- Mark any unevidenced runtime hook, label path, or evaluator quantity as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Is Hand-EV realism only “good hygiene,” not separator?
  - Does improved Hand-EV help only because AFBS still underpowered, making it temporary crutch?
  - Does endgame exactification help only in narrow late-game slice, limiting total upside?
  - Are posterior-quality issues upstream of both Hand-EV realism + exactification?
  - Does this path beat strongest simpler alternative, or is it only obvious next cleanup?
</self_red_team_rules>

<minimum_falsification_rules>
- Define minimum offline + runtime benchmarks that would prove Hand-EV realism or endgame exactification worth serious mainline attention.
- Reject any proposed path that cannot show narrow benchmark advantage before broader integration.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into broad offense/search survey.
- Stay inside Hand-EV realism, endgame exactification, and direct sequencing consequences.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current doctrine for Hand-EV, endgame exactification, AFBS, and CT-SMC sequencing.
2. Validate current Hand-EV + endgame runtime surfaces.
3. Evaluate whether better Hand-EV realism is one of strongest remaining long-run investments.
4. Evaluate whether stronger endgame exactification is separator, second-wave investment, or localized cleanup.
5. Write strongest surviving upgrade paths with exact math, interfaces, and falsification plans.

## Deliverables
1. Hydra posture reconstruction for Hand-EV / endgame / CT-SMC / AFBS sequencing
2. Current repo surfaces and missing pieces
3. Best surviving Hand-EV realism upgrade paths
4. Best surviving endgame exactification upgrade paths
5. Exact math / evaluator definitions / tensor-interface notes
6. Dependency closure table
7. Minimum falsifiable benchmark plan
8. Failure modes and kill criteria
9. Final rec:
   - separator path, second-wave path, or cleanup only
   - what to try first
   - what to defer

## Hard constraints
- no broad architecture reset
- no pretending Hand-EV or endgame exactification automatically solves posterior quality
- no vague “improve local evaluator” answers without concrete operator changes + benchmarks
- no rec that cannot be inserted into current Hydra surfaces or clearly marked `[blocked]`
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_14.md">
<![CDATA[## 1. Hydra posture reconstruction for Hand-EV / endgame / CT-SMC / AFBS sequencing

Hydra current doctrine narrower than original ambition in `HYDRA_FINAL.md`. Settled sequence in `HYDRA_RECONCILIATION.md`: keep one unified belief stack (Mixture-SIB for amortized belief, CT-SMC for search-grade posterior), close advanced target/supervision loops first, then make Hand-EV real, then treat deeper AFBS + stronger endgame work as selective specialist extensions, not immediate mainline expansion. Repo governing memo does **not** say “search first”; it says “supervision closure first, Hand-EV realism next, deeper AFBS later.” (research/design/HYDRA_RECONCILIATION.md:52-55,117-160,187-205,266-298,535-539)

Within that doctrine, Hand-EV supposed to be serious offensive local evaluator, not decorative heuristic. `HYDRA_FINAL.md` defines Group D as per-discard features with three tenpai planes, three win-probability planes, one expected-score plane, and 34 ukeire planes, computed from CPU-side hand analyzer using belief-weighted counts, and explicitly cites Suphx-style look-ahead features as major practical lever. Design-level meaning: real local oracle on fixed 192×34 observation surface, not placeholder until AFBS matures. (research/design/HYDRA_FINAL.md:78-87)

CT-SMC intended role also clear. Hydra does not want separate third belief system. Mixture-SIB = amortized public-policy belief object; CT-SMC = exact contingency-table posterior used when search-grade or calibration-grade hidden-state reasoning needed. Reconciliation memo + `HYDRA_FINAL.md` both treat CT-SMC as posterior object that should back stronger local evaluation, harder-state search, and selective late-game reasoning, subject to posterior validation gates. (research/design/HYDRA_FINAL.md:121-162; research/design/HYDRA_RECONCILIATION.md:206-208,266-274)

AFBS sequenced as specialist tool, not thing that should swallow roadmap. Docs still reserve place for `delta_q`, exit-policy distillation, and pondering, but reconciled plan explicit: Hydra should not reopen broad public-belief-search push before existing loops close and Hand-EV becomes less fake. Prior handoffs align: immediate tranche = supervision closure, then Hand-EV realism, then selective AFBS/endgame work. (research/design/HYDRA_RECONCILIATION.md:120-160,187-205,285-298,535-539)

Endgame exactification survives in doctrine, but as bounded late-game specialist. `HYDRA_FINAL.md` keeps wall-short exactification as ceiling-raising addition, but explicitly warns full expectimax too slow and recommends PIMC/top-k draw pruning instead. So even design-level “exactification” never meant “solve whole multiplayer game tree”; it meant “be much more exact, much later, only where wall short enough that correlations + placement pressure dominate.” (research/design/HYDRA_FINAL.md:202-217)

One more sequencing fact matters from inference-server file you pasted: live consumer is prebuilt 192×34 observation tensor or cached `PonderResult`. That makes Hand-EV direct encode-time lever, but makes endgame search/ponder-side lever unless some upstream caller explicitly invokes it. That asymmetry matters later.

---

## 2. Current repo surfaces and missing pieces

Current Hand-EV impl much more heuristic than doctrine implies. Live struct exactly `tenpai_prob: [[f32;3];34]`, `win_prob: [[f32;3];34]`, `expected_score: [f32;34]`, and `ukeire: [[f32;34];34]`. `ukeire` structurally reasonable: for each discard it counts shanten-improving draws against count vector. But win + score not computed by real local offensive DP. Code uses acceptance ratio, `continuation_boost` heuristic, fallback `0.35 * acceptance_ratio` win floor, and hand-value formula built from pair/triplet/flush/honor/diversity bonuses. Worse, docs say score plane should be `E[score | win, a]`, but impl stores `win_prob[discard][2] * score_estimate`, unconditional probability-weighted heuristic, not documented conditional value. That semantic mismatch alone enough to say current Hand-EV too weak to serve as serious teacher or search leaf. (hydra-core/src/hand_ev.rs:6-10,24-43,151-222,253-309)

Bridge wiring real, but what CT-SMC contributes today only first-moment collapse. `SearchContext` already has optional `mixture`, `ct_smc`, `afbs_tree`, `afbs_root`, and optional risk/stress overrides. `encode_observation_with_search_context()` computes Hand-EV + Group C features before handing result to encoder. But `compute_ct_smc_hand_ev()` does **not** evaluate per-particle worlds. It sums `ct_smc.weighted_mean_tile_count(tile, col)` across columns into single `[f32;34]` remaining-count vector and then calls same heuristic `compute_hand_ev()` as public-count path. So current “CT-SMC Hand-EV” means “heuristic Hand-EV on posterior first moments,” not “particle-weighted offensive evaluation.” (hydra-core/src/bridge.rs:27-45,251-299)

Tensor surface fixed and narrower than architecture prose can suggest. Hydra live observation = 192×34 = 6,528 floats. Group D exactly 42 channels: 3 tenpai planes, 3 win planes, 1 score plane, 34 ukeire planes, 1 mask plane. Group C = 65 channels, including single discard-level `delta_q` channel whose 34 cells correspond to tile actions, not full 46-action space. So Hand-EV inherently discard-centric on current surface, and search distillation reaching encoder also discard-centric at runtime even though global action space is 46. No spare channel budget here for more elaborate local-evaluator interface without repurposing existing semantics. (docs/GAME_ENGINE.md:122-170)

Training path one of biggest reality checks. `mjai_loader.rs` builds observations with `encode_observation(...)`, not search-context version, so training examples include **public-count Hand-EV**, not CT-SMC-enriched Hand-EV. Same loader does create Stage-A belief targets from public remaining counts + hidden-tile counts, and does generate replay-safe safety residuals, but that not same as closed search/teacher loop. In `sample.rs`, batch path clones `belief_fields_target` and `mixture_weight_target`, but sets `opponent_hand_type_target: None` and `delta_q_target: None`. In losses, advanced targets exist structurally, yet default weights for belief, mixture, delta-Q, and safety residual are zero. Repo still in state reconciliation memo warned about: advanced surfaces exist, but mainline training loop does not yet fully consume or supervise them. (hydra-train/src/data/mjai_loader.rs:303-321,360-410; hydra-train/src/data/sample.rs:156-219; hydra-train/src/training/losses.rs:33-60)

That training/runtime split matters directly for Hand-EV. Better **public-count** Hand-EV can drop into current loader + runtime encoder immediately, because both already use `encode_observation()`. Better **CT-SMC world-aware** Hand-EV cannot be treated as immediate drop-in model-input upgrade, because current training loader does not generate search-context observations. Without training-time CT-SMC re-encode path, shipping world-aware Hand-EV only at inference would introduce train/infer feature shift. That does not kill long-run idea, but absolutely kills claim that it is frictionless near-term mainline insertion.

Endgame surface real but thinner than “exactification” suggests. `EndgameSolver` has `max_wall=10`, `mass_threshold=0.95`, `should_activate(wall_remaining, has_threat)`, and `solve_with_particles(...)`, but solver itself selects top-mass particles and computes weighted average of `eval_fn(particle, action)` over legal actions. No recursion, no opponent branch model, no exact transition logic in `endgame.rs`. So current endgame is particle aggregator around externally supplied leaf evaluator, not true exact late-game solver. (hydra-core/src/endgame.rs:6-18,23-87,90-184)

There is also live-wiring asymmetry. Hand-EV definitely on encode path. Endgame not definitely on live inference path from evidence in scope. In inference-server file you provided, server either uses cached `PonderResult.exit_policy` or runs actor+SaF fast path on prebuilt observation tensor. That gives endgame plausible insertion point through pondering/cached exit policy, but direct live caller from mainline inference to `endgame.rs` not evidenced in inspected materials. So “endgame exactifier already in live mainline action path” = **[blocked by evidence]**.

---

## 3. Best surviving Hand-EV realism upgrade paths

### H1a. Exact one-step Hand-EV semantic repair on the current surface

Strongest **immediate** Hand-EV upgrade that survives strict validation.

What survives is not “rewrite Hand-EV into giant local solver.” What survives is bounded, concrete replacement of fake parts of Group D while keeping exact same `HandEvFeatures` interface. Concretely:

* keep current 42-channel layout;
* keep current exact-ish ukeire computation structure;
* replace heuristic `win_prob[:,0]` with exact one-draw tsumo probability from current count model;
* replace heuristic `tenpai_prob[:,0]` with exact one-draw-to-tenpai probability after optimal continuation discard;
* repair score plane to mean conditional win value, not `P(win)*heuristic_score`;
* only derive horizons 2 and 3 from that exact one-step base in first stage.

This survives because insertable **today**. Requires only `hand_ev.rs` and bridge-side use of same output struct. Also reaches current training immediately on public-count path because loader already uses `encode_observation()`. It does **not** require new heads, new channels, or closed `delta_q` teacher path. (hydra-core/src/hand_ev.rs:6-10,253-309; hydra-train/src/data/mjai_loader.rs:360-410; hydra-train/src/data/sample.rs:156-219)

What failed inside this lane: plain coefficient retuning. It leaves semantic mismatch untouched, still collapses posterior worlds to one heuristic count vector, and still makes score plane mean wrong thing. That is hygiene, not serious investment.

What also failed here: immediate ron model inside Hand-EV. Hydra does not show concrete runtime hook for “probability opponent discards our wait before horizon d” that is both local and posterior-consistent. Repo has safety features, tenpai targets, opp-next targets, and stress/risk planes, but not clean offensive ron-hazard API in inspected path. Mark immediate ron modeling inside Hand-EV as **defer**, not because unimportant, but because not concrete enough under current surfaces.

### H1b. CT-SMC world-aware Hand-EV with representative integer worlds

Only Hand-EV path that still looks even **potentially** separator-like after strict pass.

Reason mathematical, not rhetorical. For one-step probabilities, first-moment CT-SMC count vector usable: fractional count vector still defines valid categorical draw distribution for “one more draw.” But for multi-draw exactification, first moments are wrong object. Without-replacement recursion naturally defined on integer remaining multisets. CT-SMC particles already give those integer worlds. So genuine multi-draw Hand-EV realism wants selected representative particles, not `weighted_mean_tile_count` collapsed to one vector. That is point where Hand-EV stops being “cleaner heuristics” and starts being real posterior-aware local evaluator.

This path survives because insertion surface real: `SearchContext.ct_smc` exists, bridge dispatch already chooses CT-SMC path when posterior present, and Group D already wired into model input. It also survives because Suphx-style local look-ahead can matter. But it survives **conditionally**. Repo reality forces four gates:

1. CT-SMC posterior quality must already pass its own validation gates.
2. World selector/compressor must beat first-moment + naive top-mass baselines on regret-vs-calls frontier.
3. Hydra must resolve train/infer feature-parity issue if world-aware evaluator will become model input.
4. First promoted version should remain discard-centric + local; it should not pretend to solve opponent dynamics. (research/design/HYDRA_FINAL.md:78-87,121-162,202-217)

What failed inside this lane: immediate runtime value-directed compression as **first** integration. Attractive version = decision-focused medoid compression in evaluator/regret space, but that requires prepass over many worlds using some local evaluator, and that prepass can erase savings unless surrogate extremely cheap. So value-directed compression survives as **offline benchmark + second-stage runtime candidate**, not first thing to ship.

What failed here too: “use top-mass 95% particles and call it done.” Hydra docs say that usually means roughly 50–100 particles. That only about 1.28× to 2.56× reduction versus 128 particles, not enough to transform genuinely expensive evaluator, and provides no decision-quality certificate. Baseline selector, not separator.

---

## 4. Best surviving endgame exactification upgrade paths

### E1. Endgame leaf exactification inside the current particle shell

Best surviving endgame path is **not** “build new exact late-game solver.” It is: keep existing `EndgameSolver` shell, and make its `eval_fn(&Particle, action)` much more exact in late game.

That survives because shell already exists, takes full 46-action legal mask, and aggregates over posterior particles. If Hydra strengthens Hand-EV into real local tsumo evaluator, same evaluator can become late-game leaf under each selected world. Cleanest reuse point in codebase. Also where endgame exactification most honestly fits Hydra architecture: not separate planner stack, but better late-game leaf within selective particle-based wrapper. (hydra-core/src/endgame.rs:6-18,76-87,136-184)

Other thing that survives here: placement-aware utility. Endgame in four-player riichi not about raw round EV only. Loader already computes placement labels from final scores, so repo clearly recognizes placement semantics. Late-game leaf that only optimizes expected point gain is misaligned with use-case. Natural late-game utility therefore some monotone function of placement + score delta, captured by caller and passed through `eval_fn` closure. Closure interface allows that. Direct core impl still **[inference]**, because inspected `endgame.rs` file does not show scorer/score-context hook, but surface compatible. (hydra-train/src/data/mjai_loader.rs:165-202,334-399)

What failed: full multiplayer exact endgame solving over wall≤10. That fails on three grounds at once. First, Hydra docs already say full expectimax too slow and recommend PIMC/top-k pruning. Second, four-player general-sum partial observability makes “exact” far more than draw-tree problem. Third, current module does not even contain transition semantics needed for that claim. So “true exactification” does not survive strict pass; only “leaf exactification inside existing shell” does. (research/design/HYDRA_FINAL.md:202-217)

### E2. Ponder/AFBS deployment, not fast-path mainline deployment

Inference-server file you supplied makes most plausible deployment path clear: expensive late-game work should land through pondering/cached `PonderResult.exit_policy`, not by bloating fast network path. Server already has slow-path reuse mechanism keyed by info-state hash. That matches doctrine: AFBS + late-game exactification are specialist tools. So stronger endgame survives as **ponder/search-side specialist**, not as every-turn mainline feature rewrite.

What failed here: “make endgame exactification mainline separator before Hand-EV + supervision closure.” That does not survive repo reality. Live caller not evidenced, teacher/export path not closed, and state slice inherently narrow. Even if late-game upside real, it is still later + narrower than Hand-EV realism on current surfaces.

---

## 5. Exact math / evaluator definitions / tensor-interface notes

### 5.1 Current Hand-EV math in the repo

Let `h` be 34-bin hand-count vector, `a` discard tile type, `h_a = h - e_a`, and `r ∈ R_+^34` remaining-count vector used by evaluator.

Current code computes:

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

and uses heuristic continuation term

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

Finally it builds heuristic hand-value estimate from suit mix, honor count, pair/triplet bonuses, flush bias, tile concentration, diversity penalty, and honor-discard bonus, and stores

[
\text{expected_score}[a] = P^{(3)}_{\text{win}}(a)\cdot \text{score_estimate}(h,a).
]

That last line = semantic mismatch with doc’s intended `E[score \mid win, a]`. (hydra-core/src/hand_ev.rs:303-309)

### 5.2 Proposed H1a exact one-step replacement

For first promoted upgrade, do not change tensor interface. Change evaluator semantics.

Let evaluator consume either one count vector `r` or selected weighted set of worlds
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

where (\mathcal{D}(h_a+e_t)) = set of legal continuation discards after draw.

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

This exactly fixes semantic center of Group D for horizon 1. For H1a, horizons 2 and 3 can be filled from exact one-step base with simple cumulative continuation rule,
[
P^{(d)} = 1-(1-P^{(1)})^d,
]
which still approximate, but far less fake than current continuation-boost machinery. So H1a = “exact one-step, approximate multi-step,” fair + concrete first promotion. Scoring hook inside `hand_ev.rs` still **[inference]** in inspected files, though design docs clearly assume scoring engine exists. (research/design/HYDRA_FINAL.md:80-87)

### 5.3 Proposed H1b world-aware multi-draw extension

H1b = real long-run candidate.

Use selected integer CT-SMC worlds (X^{(k)}), with remaining counts
[
r_t^{(k)} = \sum_{\text{hidden cols}} X^{(k)}_{t,\text{col}},
]
and recurse on those integer multisets rather than posterior first moments.

For horizon (d>1), define continuation policy (\pi_d) over 14-tile post-draw states and top-(M) draw set (T_M) for branch pruning. Then pruned recursion is

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

with analogous recurrences for tenpai + score mass.

Important point not exact pruning scheme; it is that state object must now be integer world, not weighted-mean count vector, if recursion supposed to mean anything exact-ish.

### 5.4 Compute sanity checks

From current `hand_ev.rs` structure, one per-world evaluation costs about **36–70 shanten calls per distinct discard**: one base shanten call plus up to 34 inside `compute_ukeire`, one more for `shanten_after`, and another up to 34 if `immediate_win_probability` runs. That means:

* at 14 distinct discards and 128 worlds, naive full-particle Hand-EV about **64,512–125,440 shanten calls**;
* at 14 discards and 8 worlds, **4,032–7,840 shanten calls**.

For branch counts, exact one-step world-aware evaluation manageable. With (K=8) worlds, (D=14) discard types, and (U=20) live draw types, one-step state count only
[
KDU = 2240.
]
But naive three-step branching not:
[
KD(U + U^2 + U^3)=943{,}040
]
for same (K,D,U). Top-(M) pruning changes that drastically: with (M=3), branch count drops to **4,368**; with (M=5), **17,360**. So H1b survives only as pruned or sampled continuation, not naive full branching.

For endgame, current shell itself cheap:
[
\text{leaf calls} = \times W
]
for legal actions and (W) selected worlds. With (A=14), (W=50) means 700 leaf calls; (W=100) means 1400. Problem not shell. Problem = leaf cost. At 10 ms total budget, 50 µs leaf only fits about 14 worlds for 14 actions; 100 µs leaf fits only about 7. So any real endgame leaf exactification still needs either aggressive world reduction, cheap leaf, ponder/off-turn path, or all three.

### 5.5 Tensor/interface notes

Hard interface facts:

| Surface                   | Shape / meaning                                                               |
| ------------------------- | ----------------------------------------------------------------------------- |
| Observation tensor | `192 × 34 = 6528` floats |
| Group D Hand-EV           | 42 channels = 3 tenpai + 3 win + 1 score + 34 ukeire + 1 mask                 |
| `HandEvFeatures` | `[[f32;3];34]`, `[[f32;3];34]`, `[f32;34]`, `[[f32;34];34]` |
| Group C `delta_q` feature | one **34-cell discard plane**, not full 46-action tensor |
| Full action space         | 46 actions                                                                    |
| `SearchContext` | optional `mixture`, `ct_smc`, `afbs_tree`, `afbs_root`, risk/stress overrides |
| Endgame shell | `solve_with_particles(&[Particle], &[bool;46], &dyn Fn(&Particle,u8)->f32)` |

Fixed-shape consequence simple: Hand-EV upgrades must reuse existing 42-plane semantics, and any claim that this path directly upgrades full-action search semantics is overstated. Hand-EV primarily discard-axis local evaluator on current Hydra surfaces. ([GitHub][5])

---

## 6. Dependency closure table

| Item                                               | Already present                                                                 | Missing closure                                                                                              | Status                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| H1a exact one-step Hand-EV on public counts | `hand_ev.rs`, `bridge.rs`, 42-plane encoder, loader uses `encode_observation()` | score hook inside `hand_ev.rs` is **[inference]** from docs, not shown directly | **Viable now** |
| H1a exact one-step Hand-EV on CT-SMC first moments | `SearchContext.ct_smc`, bridge CT-SMC path | none for horizon-1 probabilities; multi-draw exactness still absent | **Viable now** |
| H1b world-aware multi-draw Hand-EV                 | CT-SMC particles, bridge dispatch hook, fixed Group D surface                   | representative-world selection, recursion/caching, train/infer parity for CT-SMC search-context observations | **Second-stage only**                |
| Training parity for world-aware Hand-EV            | loader and encoder exist                                                        | loader does not build search-context observations; no CT-SMC re-encode path in scope                         | **[blocked] for immediate mainline** |
| Direct Hand-EV / endgame distillation targets | losses support optional advanced targets | `delta_q_target` dropped in batch path; advanced loss weights default zero; no Hand-EV teacher path | **[blocked]** |
| Endgame leaf exactification | `endgame.rs` shell and particle interface exist | better leaf evaluator; placement-aware utility hook; direct live caller not evidenced | **Specialist viable** |
| Live endgame deployment | ponder cache / cached exit policy path exists in provided inference file | explicit caller wiring from late-game logic to `endgame.rs` or ponder path not shown | **[inference]** |
| Value-directed runtime world compression           | CT-SMC particles exist                                                          | cheap enough surrogate metric and benchmark win over simpler baselines                                       | **Benchmark first, not ship first**  |

Table above = narrow closure picture implied by repo + provided inference file. Main takeaways: public/one-step Hand-EV directly insertable; world-aware Hand-EV real but not frictionless; endgame specialist and not yet evidently on live path. ([GitHub][6])

---

## 7. Minimum falsifiable benchmark plan

### Phase A: posterior prerequisite for any world-aware claim

Before promoting H1b or any endgame exactification claim that relies on CT-SMC worlds, require Hydra’s own posterior gates from `HYDRA_FINAL.md` / reconciliation to pass on evaluation stack being used. If Gate / Gate B posterior validation fails, stop. Otherwise any improvement from world-aware Hand-EV or endgame = “more faithfully wrong posterior usage.” ([GitHub][2])

### Phase B: H1a exact-one-step Hand-EV benchmark

Build discard-only replay suite with exact hidden world available. For each state + legal discard, compute hidden-world oracle for:

* (P_{\text{win}}^{(1)}(a))
* (P_{\text{tenpai}}^{(1)}(a))
* (\mathrm{E}[score \mid win,
* `ukeire[a,t]`

using true remaining multiset from replay.

Compare current Hand-EV vs H1a on:

[
\text{MSE}*{\text{win}},\ \text{MSE}*{\text{tenpai}},\ \text{MSE}*{\text{score}},\ \text{MSE}*{\text{ukeire}}
]

and on ranking metrics:

[
\text{Top1Match},\qquad
\text{Regret}(s)=Q^*(s,a^*)-Q^*(s,\hat
]

Promotion gate for serious mainline attention:

* mean regret reduction **≥ 10%** vs current Hand-EV;
* top-1 discard match **≥ 3 percentage points** better than current Hand-EV;
* non-regression on riichi-threat slice;
* encode-time wall clock **≤ 1.25×** current public Hand-EV encode cost.

If H1a does not clear that, kill it. Then not even good second-wave upgrade.

### Phase C: H1b world-aware / compression frontier

Use posterior-sensitive states only: low wall, high entropy, low ESS, and states where first-moment counts and full particles disagree.

Take strong but expensive reference:
[
Q_{\text{full}}(a)=\sum_{i=1}^{P} w_i q_i(a)
]
under improved local evaluator.

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
* it must beat plain first-moment counts on posterior-sensitive slice, not tie them.

If no such point exists, kill H1b as mainline path and demote Hand-EV realism to H1a-level cleanup + semantics repair.

### Phase D: endgame specialist benchmark

Restrict to `wall_remaining <= 10 && has_threat`, with dedicated orasu / close-placement slice.

Benchmark current endgame shell with current leaf vs same shell with improved leaf. Require:

* positive paired improvement in placement-aware utility on full late-game suite;
* positive paired improvement on orasu / close-gap slice specifically;
* non-worse deal-in rate;
* standalone specialist p95 runtime within late-game budget granted by caller. Since live caller not evidenced in scope, measure standalone p95 for now and do not promote past offline status until caller exists.

If improved late-game leaf wins only on raw round EV but not placement-aware utility, kill it. That is exactly kind of false positive four-player general-sum endgames produce.

---

## 8. Failure modes and kill criteria

1. **Posterior-quality upstream failure.**
If CT-SMC calibration weak, world-aware Hand-EV + endgame exactification should both be deferred. They amplify posterior mistakes; they do not fix them. ([GitHub][2])

2. **Train/infer feature-shift failure for CT-SMC Hand-EV.**
If Hydra cannot generate training-time search-context observations, do not ship world-aware Group D as model-input change. Keep it offline or search-side only.

3. **H1a fails exact-one-step oracle benchmark.**
Then Hand-EV realism not ready for mainline attention. If it cannot beat current heuristic on local quantity it is supposed to predict, stop.

4. **H1b fails regret-vs-calls frontier.**
If representative worlds do not beat first-moment counts by clear margin at materially lower cost, kill world-aware separator claim.

5. **Action-sufficient runtime compression prepass erases its own savings.**
Then it stays offline only. Do not ship “smart compressor” whose prepass dominates total cost.

6. **Endgame leaf wins locally but not in placement-aware late-game utility.**
Then not useful riichi endgame improvement; defer it.

7. **No live caller for endgame.**
Until endgame wired either into pondering or another explicit caller, it remains reserve-shelf module, not mainline investment.

8. **AFBS/delta-q closure later absorbs gain.**
If after search-target closure marginal value of improved Hand-EV collapses, then path was interim crutch, not separator. Repo today not yet in position to know that for sure, another reason not to overclaim.

Failed ideas that do **not** survive strict pass:

* coefficient retuning of current Hand-EV;
* immediate ron modeling inside Hand-EV;
* exact multi-draw DP on CT-SMC weighted-mean counts;
* full multiplayer endgame expectimax;
* direct Hand-EV/endgame distillation now;
* runtime value-directed compression as first shipped step.

---

## 9. Final recommendation

**Overall classification: second-wave path.**

More precisely: broad claim “Hand-EV realism plus endgame exactification is one of Hydra’s biggest long-run separator paths” is **too strong as stated**. After stricter repo pass, lane survives, but only in narrower form.

Part that survives strongly = **Hand-EV realism**, and even there immediate concrete move is **not** full long-run dream. First thing to try = **H1a**: repair Group D to be exact one-step local evaluator on current surface, fix score-plane semantics, and benchmark against exact hidden-world one-step oracles. That is concrete, directly insertable, reaches current training immediately on public-count path, and can be falsified cleanly. After that, and only if posterior validation + compression benchmarks clear, Hydra should test **H1b**: representative-world CT-SMC Hand-EV. That is only subpath here that still has real separator-like upside, because it is only one that turns Hand-EV from “better heuristics on one count vector” into “posterior-aware local offense.” ([GitHub][1])

**Endgame exactification later + narrower.** It survives as **specialist path**, not current mainline separator. Practical form that survives = late-game leaf exactification inside existing particle shell, likely deployed through pondering/cached exit policies rather than fast path. It should be deferred until H1a validated and until Hydra either closes more of search-target loop or at least wires explicit caller path. Full “exact endgame solving” does not survive repo + compute realities in scope. ([GitHub][2])

So narrowed action order:

1. **Try first:** H1a exact one-step Hand-EV semantic repair on existing 42-plane surface, with exact-one-step oracle benchmarking.
2. **Try second, only if H1a wins and CT-SMC gates pass:** H1b representative-world CT-SMC Hand-EV, benchmarked on regret-vs-calls frontier before any mainline promotion.
3. **Defer:** ron modeling inside Hand-EV, direct Hand-EV/endgame teacher export, value-directed runtime compression as first shipped step, and stronger endgame exactification beyond leaf/specialist impl.

That is narrowest answer evidence supports: **not cleanup only, not current separator path, but strong second-wave lane whose only genuine separator candidate is world-aware Hand-EV, not endgame.**

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