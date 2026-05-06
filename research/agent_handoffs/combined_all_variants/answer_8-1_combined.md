Read rest. Then compress only natural language, keep structure exact.<combined_run_record run_id="008_invent_new_techniques" variant_id="agent_8_revised_variant_a" schema_version="1">
<metadata>
<notes>Revised answer variant for prompt 8.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md">
<![CDATA[# Hydra fresh-context prompt — invent genuinely new techniques with math and red-team pressure

Primary source material in raw GitHub links below.

  ## Critical directive — how to read the core Hydra docs

Avoid known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for this task:
  - keyword search first
  - isolated 20-100 line reads around keywords
  - treating docs like logs or grep DB
  - inventing techniques before holistic Hydra understanding

For this task, disqualifying.

Required reading workflow:
  1. Use browse/fetch on raw GitHub links for core docs below.
  2. Read core docs holistically, sequentially, before narrow search.
  3. Build high-level model of Hydra active path, reserve shelf, runtime structure, training surfaces, already-partial loops.
  4. Only then use narrower search for exact detail and outside inspiration.

Do not use grep-style keyword hunting as primary reading strategy for these core docs.

<holistic_ingestion_rules>
  - Read core docs as whole docs before narrowing.
  - Do not start with keyword search on core docs.
  - Do not rely on fragmented line-window retrieval for architecture understanding.
  - After holistic reading, targeted search for exact detail allowed.
</holistic_ingestion_rules>

  ## Reading order

  1. `research/design/HYDRA_RECONCILIATION.md`
  2. `research/design/HYDRA_FINAL.md`
  3. `docs/GAME_ENGINE.md`
  4. `research/design/OPPONENT_MODELING.md`
  5. `research/design/TESTING.md`
  6. `research/design/SEEDING.md`
  7. `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  8. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
  9. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
  10. code-grounding files
  11. outside retrieval

  ## Raw GitHub links

Core docs:
  - `README.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md
  - `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
  - `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
  - `research/design/HYDRA_ARCHIVE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_ARCHIVE.md
  - `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
  - `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
  - `research/design/TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
  - `research/design/SEEDING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/SEEDING.md
  - `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
  - `research/infrastructure/INFRASTRUCTURE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/infrastructure/INFRASTRUCTURE.md

Code-grounding files:
  - `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
  - `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
  - `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
  - `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
  - `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
  - `hydra-core/src/robust_opponent.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/robust_opponent.rs
  - `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
  - `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
  - `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
  - `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs

Prior answer archive:
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

You act as long-think breakthrough engineer for Hydra.

Job: discover genuinely new Hydra techniques, not renamed known tricks or shiny repackaging. Combining ingredients allowed only when resulting mechanism is mathematically explicit, architecture-respecting, and survives adversarial self-review.

Target not mere novelty. Target separator-level breakthrough: technique that could matter for Hydra like LuckyJ signature ACH/search-era breakthroughs mattered for LuckyJ. Do not mechanically imitate ACH. Find comparable strategic breakthrough for Hydra actual architecture and bottlenecks.

If idea not technically crisp, kill it.

<output_contract>
  - Return exactly requested sections, in requested order.
  - Be as detailed and explicit as needed; do not optimize for brevity.
  - Return full technical treatment, not compressed memo.
  - Return only 1-3 serious techniques.
  - Short answer usually = failure mode.
</output_contract>

<verbosity_controls>
  - Prefer full technical exposition over summary.
  - Use multi-paragraph explanation when needed.
  - Do not omit equations, derivations, tensor/interface details, pseudocode, assumptions, thresholds, edge cases, or impl caveats when they matter.
  - When unsure, include more math, derivation, and mechanism detail.
</verbosity_controls>

<research_mode>
  - Work in 3 passes:
    1. Ingest: read Hydra docs holistically; reconstruct real current mainline, reserve shelf, missing closures.
    2. Retrieve: search broadly for ingredient families, neighboring mechanisms, counterexamples.
    3. Synthesize: keep only techniques genuinely novel for Hydra and technically viable under Hydra constraints.
  - Stop only when more search unlikely to change final ranking.
</research_mode>

<tool_persistence_rules>
  - Prefer full-document browse/fetch for core docs over fragmented terminal-style chunk reading.
  - Search beyond already-surfaced papers when that could materially improve novelty or falsification.
  - Do not stop at first plausible invention.
</tool_persistence_rules>

<calculation_validation_rules>
  - If rec depends on quantitative reasoning, derive explicitly.
  - Use executable arithmetic or small scripts when needed to sanity-check formulas, tensor shapes, threshold logic, or algorithm invariants.
  - Do not fake arithmetic that could have been checked.
</calculation_validation_rules>

<dependency_checks>
  - Before proposing impl, verify Hydra already has or could cheaply expose needed signals, labels, or runtime hooks.
  - Before proposing new objective or target, check whether needed trajectories, teacher outputs, or hidden-state labels exist or can be derived safely.
</dependency_checks>

<posture_reconstruction_rules>
  - Before proposing any technique, include short "Hydra posture reconstruction" with 5-10 bullets.
  - Bullets must distinguish current mainline doctrine, reserve-shelf ideas, partially closed loops, and 2-3 non-goals or deprioritized paths.
  - Do not propose breakthrough candidates until posture reconstruction complete.
</posture_reconstruction_rules>

<citation_rules>
  - Cite only sources retrieved in this workflow or included in raw links above.
  - Never fabricate references.
  - Attach citations to exact claims they support.
  - Include full reference detail and direct links when possible.
</citation_rules>

<grounding_rules>
  - Ground Hydra-specific claims in raw links above.
  - Ground outside-technique claims in retrieved sources.
  - Label inference as inference.
  - If sources conflict, state conflict explicitly.
  - Any repo touchpoint, label source, tensor, or runtime hook not explicitly evidenced from provided materials must be marked `inference` or `[blocked]`.
</grounding_rules>

<novelty_viability_rules>
  - Do not invent shallow, buzzword-heavy acronyms.
  - If you propose novel technique, you must prove technical viability.
  - In thinking space, explicitly write:
    - mathematical formulation
    - tensor shapes in and out of network
    - exact algorithm pseudocode
  - If you cannot mathematically define technique within Hydra architecture constraints, discard it.
  - Do not confuse renamed known trick with genuinely new mechanism.
  - Be explicit: borrowed unchanged, adapted, newly proposed.
</novelty_viability_rules>

<self_red_team_rules>
  - Before finalizing rec, Red-Team own ideas.
  - For every proposed technique, actively search provided documents for failure reasons.
  - Ask explicitly:
    - How does this break in 4-player general-sum game?
    - Does this violate partial observability constraints?
    - Does this require labels, targets, or privileged signals Hydra does not have?
    - Is claimed novelty fake because method collapses to known technique under Hydra constraints?
    - Does simpler existing Hydra path already dominate this?
  - - Does supposed breakthrough collapse into incremental tuning trick once written mathematically?
  - Present only techniques that survive adversarial self-review.
</self_red_team_rules>

<anti_survey_rules>
  - Do not return literature survey, field map, or long adjacent-idea list without convergence.
  - Every cited outside paper, repo, or mechanism must earn place by changing final candidate set or red-team analysis.
  - If paragraph does not help define, falsify, compare, or prototype surviving candidate, cut it.
</anti_survey_rules>

<novelty_honesty_rules>
  - For every surviving technique, include "closest known baseline" subsection.
  - State nearest known method/family, exact overlap, irreducible difference.
  - If method reduces to known technique under realistic Hydra constraints, downgrade or reject it.
  - Label each surviving candidate:
    - `A`: genuinely new mechanism
    - `B`: known mechanism with Hydra-specific adaptation that plausibly changes capability
    - `C`: renamed or lightly modified known trick
  - Reject all `C` candidates.
</novelty_honesty_rules>

<minimum_falsification_rules>
  - For every surviving technique, define minimum falsifiable prototype that tests claimed breakthrough mechanism in isolation.
  - If core claim cannot be tested without large coupled rollout or major stack build-out, reject as too diffuse.
  - First benchmark should distinguish idea from stronger tuning, more search, more data, or easier teacher signals.
</minimum_falsification_rules>

<completeness_contract>
  - Treat task incomplete until every surviving technique includes exact mechanism, mathematical formulation, tensor shapes, pseudocode, repo insertion points, cheapest prototype path, benchmark plan, and kill criteria.
  - Mark any underspecified item [blocked] rather than pretending ready.
</completeness_contract>

<verification_loop>
  - Before finalizing, verify you read core Hydra docs holistically before narrowing.
  - Verify each surviving technique is not merely renamed known trick.
  - Verify each surviving technique is mathematically defined strongly enough that coding agent could start prototyping.
  - Verify novelty claim survives own red-team pass.
</verification_loop>

<dig_deeper_nudge>
  - Do not stop at first cool invention.
  - Prefer capability-changing mechanisms over cosmetic complexity.
  - Search especially hard around belief compression, decision-focused uncertainty, teacher trust gating, opponent-mixture optimization, selective compute, multiplayer credit assignment, privileged-to-public distillation.
</dig_deeper_nudge>

  ## What to do

  1. Reconstruct Hydra real current posture before inventing.
  2. Search for outside ingredients, neighboring ideas, counterexamples.
  3. Invent smallest number of serious techniques that look genuinely new for Hydra, technically workable, and plausibly separator-level rather than minor gain.
  4. For each candidate, write objective/update rule, tensor interfaces, algorithm pseudocode, repo insertion points, cheapest falsification path.
  5. Compare each surviving candidate against strongest simpler mainline alternative; reject if mostly incremental.
  6. Try hard to kill each idea before keeping it.

  ## Deliverables

Return only 1-3 serious techniques.

If no candidate survives novelty, grounding, and prototypeability filters, return `0 surviving candidates` and explain why.

For each technique give, in order:
  1. name
  2. problem solved
  3. outside ingredients and exact sources
  4. what is borrowed unchanged
  5. what is adapted for Hydra
  6. what is genuinely novel synthesis
  7. why it fits Hydra specifically
  8. exact mathematical formulation
  9. tensor shapes and affected network interfaces
  10. exact algorithm pseudocode
  11. exact Hydra surfaces it would touch
  12. prototype path
  13. benchmark plan
  14. what success would look like
  15. what would kill idea quickly
  16. red-team failure analysis
  17. why this is more likely to matter than strongest simpler mainline alternative
  18. closest known baseline and why this does not reduce to it
  19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker
  20. minimum falsifiable prototype

Then end with:
  - single best technique to try first
  - single best cheap benchmark to run first
  - single biggest hidden impl risk
  - 2-5 most tempting rejected directions and exactly why rejected

  ## Hard constraints

  - no generic bigger-model or bigger-search proposals
  - no AGPL-derived code advice
  - no broad architecture resets
  - no pretty but untestable inventions
  - no keeping weak options alive because prompt implicitly wanted answer
  - no shallow acronym invention
  - no rec that cannot be written as executable objective/update rule plus pseudocode inside Hydra constraints
  - no answer mainly summarizing literature instead of converging to prototype-worthy separator candidate
  - no candidate surviving only because prompt implicitly demanded at least one answer

Success = final answer feels like serious long-think invention pass and leaves at least one genuinely new Hydra technique that is mathematically explicit, red-teamed, prototypeable, and plausibly capable of becoming LuckyJ-caliber separator for Hydra rather than another incremental trick.
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_8new.md">
<![CDATA[**Hydra posture reconstruction**

  * **Current mainline doctrine:** repo active path = not “add more modules,” but close already-declared advanced-target loops. Reconciled next move: ExIt/exit, `ΔQ`, safety-residual target generation first; belief fields, mixture weights, opponent-hand-type stay inactive until Hydra has credible **public-teacher** path for them. Reconciliation memo also says no new heads, no architecture expansion in that tranche. ([GitHub][1])

  * **Runtime structure richer than training closure:** live contract already fixed **192×34** encoder with preserved 85-channel baseline prefix, 46-action space, Group C dynamic search/belief features, Group D Hand-EV features in architectural SSOT. Project already points at two-tier 12-block actor / 24-block learner, not broad monolith. ([GitHub][2])

  * **Belief doctrine settled:** intended belief stack unified, not duplicated—Mixture-SIB for amortized belief, CT-SMC for search-grade posterior—so proposals creating second belief pipeline already off-posture. ([GitHub][3])

  * **Search posture settled too:** AFBS should be **selective / specialist / hard-state gated**, not broad default runtime search, and prior internal analysis already identified “trust-gated selective AFBS” as surviving search-side bet. So next separator more likely from **better teacher provenance and compute targeting** than from “deeper search everywhere.” ([GitHub][4])

  * **Hand-EV important but not open separator here:** Hydra final architecture strongly elevates Hand-EV, and prior internal analysis says Hand-EV realism comes before deeper AFBS. But that already known mainline closure item, not fresh breakthrough by itself. ([GitHub][5])

  * **Opponent modeling strategically central, but heavy extensions reserve-shelf unless promoted:** opponent-modeling doc says Hydra primary differentiator = opponent modeling, while warning heavier extensions should stay reserve/future unless reconciled doctrine promotes them. So any surviving opponent-side invention must unify with mainline, not reopen speculative complexity. ([GitHub][6])

  * **Code-grounded gap real and narrow:** in provided raw files, `HydraOutput` already exposes `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`; `HydraTargets` already has optional slots for those targets; `bridge.rs` already emits mixture weights, belief fields, entropy/ESS, AFBS `delta_q`; and `CtSmc`, `AfbsTree`, `robust_opponent` already contain usable runtime primitives. But current sample/loader path still leaves `delta_q_target` and `opponent_hand_type_target` absent at collation, while loader builds replay-safe safety-residual and stage-A belief targets instead. Pattern = “advanced surfaces exist, credible labels/inference loops do not.”

  * **Current non-goals / deprioritized paths:** broad “search everywhere” AFBS, duplicated belief stacks, new heads before old heads get real labels, optimizer-theory detours, speculative deception extras all already demoted or rejected by reconciled posture. So weak candidates that mostly repackage those directions should die. ([GitHub][4])

---

  ## Technique 1

**1. name**

Posterior-consensus ExIt/ΔQ distillation with action-straddle search gating.

**2. problem solved**

Hydra next tranche wants ExIt and `ΔQ`, but dangerous failure mode = distilling from **single privileged hidden world** and pretending its action ranking is public-realizable. In Mahjong, hardest states often exactly those where different plausible hidden worlds flip best discard. Single-world search labels then leak unattainable info, while current AFBS gate still mostly heuristic. This technique fixes both: distill only **posterior-stable** teacher signal, and turn posterior instability into explicit label for **where search worth compute**. ([GitHub][1])

**3. outside ingredients and exact sources**

Outside ingredients that changed final candidate set: GATES, core idea = privileged/self-distillation should be gated by **tutor consensus** instead of assuming privileged supervision correct; HIB, core idea = privileged-to-public distillation should reduce oracle/student value gap; value-directed belief approximation, core point = belief approximation should preserve **decision quality**, not marginal fidelity; calibrated value-aware model learning, showing naive value-aware objectives can be uncalibrated and stochastic uncertainty still matters even when deterministic values look good. Student of Games and 2024 IJCAI look-ahead paper mattered as search-side baselines showing guided search summaries can be distilled into policy/value system without turning whole stack into full-search training. ([arXiv][7])

**4. what is borrowed unchanged**

Borrowed unchanged: consensus-gated distillation from GATES; privileged-to-public value-gap framing from HIB; value-directed warning that right compression target is **decision-relevant**, not merely distribution-close; and general search-as-teacher pattern from Student of Games / look-ahead search distillation. ([arXiv][7])

**5. what is adapted for Hydra**

Adaptation: “multiple tutor traces” become **multiple hidden-world teacher evaluations for same public state**. Tutor not text model with extra context; tutor = shallow Hydra teacher evaluated on (K) posterior worlds sampled from CT-SMC or, in cheapest prototype, from style-agnostic CT-SMC prior. Distilled target not final answer but public-state action object: posterior-mean `ΔQ`, consensus-shaped exit policy, posterior-mean safety residual. Disagreement signal then reused as **search-worth** label instead of discarded. This fits Hydra architecture far better than off-the-shelf uncertainty-weighted imitation. ([GitHub][8])

**6. what is genuinely novel synthesis**

Irreducible new part = split teacher signal into two outputs from same posterior-world ensemble:

  1. **stable component** Hydra should learn directly:
[
\bar{\Delta}(a),;\pi^*(a),;\bar r(a)
]

  2. **unstable component** Hydra should not force into student policy, but should instead use to predict **search value-of-compute**:
[
c(a),;s_\epsilon,;v(a),;G(I)
]

This not merely “confidence-weighted distillation.” It is one mechanism turning partial observability into **teacher-abstention rule** and **selective-compute label**.

**7. why it fits Hydra specifically**

Hydra already has nearly all required surfaces. Docs explicitly say belief/search targets must come from credible public teachers, not realized hidden-state labels; active tranche already wants `exit_target`, `delta_q_target`, safety residual; `bridge.rs` already knows how to emit mixture weights, belief fields, entropy/ESS, AFBS `delta_q`; and AFBS shell already has root exit policy and node-Q summaries. So this not architecture reset. It gives Hydra’s already-existing advanced surfaces **right semantics** under partial observability. ([GitHub][1])

**8. exact mathematical formulation**

Let (I) be public Hydra information state, with legal mask (m \in {0,1}^{46}). Let (x_k \sim b(\cdot \mid I)), (k=1,\dots,K), be posterior worlds with normalized weights (\alpha_k), (\sum_k \alpha_k = 1).

For each sampled world (x_k), let teacher return:

  * (\Delta_k \in \mathbb{R}^{46}): world-conditioned action deltas, using Hydra existing AFBS semantics (Q(a)-Q(\text{root})) when available.
  * (\pi_k \in \Delta^{45}): teacher exit/root policy.
  * (r_k \in [0,1]^{46}): teacher safety residual or world-conditioned deal-in residual target.

Define posterior means:
[
\bar{\Delta}*a = \sum*{k=1}^{K} \alpha_k \Delta_{k,a},
\qquad
\bar r_a = \sum_{k=1}^{K} \alpha_k r_{k,a}.
]

For compatibility with current narrow-tranche `ΔQ` scaling from prior Hydra work, define clipped/scaled target
[
\tilde{\Delta}_a
================

\mathrm{clip}!\left(\frac{\bar{\Delta}*a}{s*\Delta}, -1, 1\right),
\qquad s_\Delta = 0.15.
]
I kept (s_\Delta=0.15) because it matches current narrow-tranche `ΔQ` normalization proposal rather than inventing new scale. ([GitHub][1])

Define posterior top-action mass:
[
c_a
===

\sum_{k=1}^{K}
\alpha_k,
\mathbf 1!\left[
= \arg\max_{b:m_b=1}\Delta_{k,b}
\right].
]

Define posterior variance:
[
v_a
===

\sum_{k=1}^{K}
\alpha_k,
(\Delta_{k,a} - \bar{\Delta}_a)^2.
]

Let
[
a_1 = \arg\max_{a:m_a=1}\bar{\Delta}*a,
\qquad
a_2 = \arg\max*{a\neq a_1,;m_a=1}\bar{\Delta}*a.
]
Define action-straddle mass:
[
s*\epsilon
==========

\sum_{k=1}^{K}
\alpha_k,
\mathbf 1!\left[
\Delta_{k,a_2} \ge \Delta_{k,a_1} - \epsilon_{\mathrm{flip}}
\right].
]

For prototype, I sanity-checked (\epsilon_{\mathrm{flip}}=0.05), (\tau_c=0.70), (\tau_s=0.35), and (\tau_v=0.05) on synthetic stable / top-two-flip / high-variance cases; they cleanly separated those toy cases, but only seed thresholds, not validated production constants.

Define teacher reliability for **distillation**:
[
\lambda_{\mathrm{teach}}
========================

\mathrm{clip}!\left(
\frac{\max_a c_a - \tau_c}{1-\tau_c},,0,,1
\right)
\cdot
\exp!\left(
-\frac{\max_a v_a}{\tau_v}
\right)
\cdot
\exp!\left(
-\frac{s_\epsilon}{\tau_s}
\right)
\cdot
\chi_{\mathrm{ESS}},
]
where
[
\chi_{\mathrm{ESS}}
===================

\mathrm{clip}!\left(
\frac{\mathrm{ESS}/P - \tau_{\mathrm{ess}}}{1-\tau_{\mathrm{ess}}},
0,1
\right)
]
if CT-SMC ESS available, and (\chi_{\mathrm{ESS}}=1) in cheapest prior-only prototype.

Define consensus-shaped exit target:
[
\pi^*_a
=======

\frac{
m_a,
\exp(\bar{\Delta}*a/\tau*\pi),
(c_a+\epsilon_c)^\gamma
}{
\sum_b
m_b,
\exp(\bar{\Delta}*b/\tau*\pi),
(c_b+\epsilon_c)^\gamma
}.
]

For **search-worth**, do not threshold instability directly. Instead compute posterior expected regret reduction from switching current base action to ensemble-best action:
[
a_{\mathrm{base}}
=================

\arg\max_{a:m_a=1}
\Big(
\ell_\theta(a\mid I)
+
\beta_\Delta \hat{\Delta}_\theta(a\mid I)
\Big),
]
[
= a_1,
]
[
G(I)
====

\sum_{k=1}^{K}
\alpha_k
\left[
\Delta_{k,a^*}
--------------

\Delta_{k,a_{\mathrm{base}}}
\right].
]

Then define search-gate label as
[
y_{\mathrm{search}}
===================

\mathbf 1!\left[
G(I) > \delta_g
\right].
]
For prototype, (\delta_g) should be chosen as held-out quantile of empirical (G(I)) distribution—e.g. top quartile—rather than unvalidated absolute constant.

Losses:

Policy / ExIt:
[
L_{\pi}
=======

\lambda_{\mathrm{teach}}
\cdot
  \mathrm{KL}\big(\pi^*,|, \hat{\pi}_\theta\big).
]

Delta-Q:
[
L_{\Delta}
==========

\lambda_{\mathrm{teach}}
\cdot
\frac{
\sum_a m_a,\mathrm{Huber}!\left(
\hat{\Delta}_\theta(a)-\tilde{\Delta}_a
\right)
}{
\max(1,\sum_a m_a)
}.
]

Safety residual:
[
L_{\mathrm{safety}}
===================

\lambda_{\mathrm{teach}}
\cdot
\frac{
\sum_a m_a,\mathrm{Huber}!\left(
\hat r_\theta(a)-\bar r_a
\right)
}{
\max(1,\sum_a m_a)
}.
]

Gate:
[
L_{\mathrm{gate}}
=================

\mathrm{BCE}!\left(g_\psi(z(I)),, y_{\mathrm{search}}\right),
]
with (z(I)) public/search feature vector defined below.

**9. tensor shapes and affected network interfaces**

Main model outputs unchanged in cheapest prototype:

  * `policy_logits`: ([B,46])
  * `delta_q`: ([B,46])
  * `safety_residual`: ([B,46])

These shapes already exist in Hydra provided model and loss surfaces.

Teacher ensemble tensors per sample:

  * world deltas: ([K,46])
  * world safety residuals: ([K,46])
  * world weights: ([K])
  * posterior-optimal-action masses (c): ([46])
  * posterior variances (v): ([46])
  * search-worth scalar (G(I)): ([1])

New dataset-side tensors:

  * `delta_q_target`: ([B,46]) — already supported train-side, unfilled in provided sample path.
  * `exit_target`: ([B,46]) — upstream production explicitly called out as missing-but-desired in reconciliation path. ([GitHub][8])
  * `safety_residual_target`: ([B,46]) — already live replay-side in current loader.
  * `search_need_label`: ([B,1]) — new auxiliary dataset field for gate, not new Hydra head.

Gate feature vector:
[
z(I)\in\mathbb{R}^9
]
with prototype features:

  1. base-policy top-2 gap,
  2. predicted `delta_q` top-2 gap,
  3. mixture entropy,
  4. mixture ESS,
  5. max opponent tenpai probability,
  6. danger of current top action,
  7. wall fraction remaining,
  8. orasu flag,
  9. score-gap-to-next-rank.

Everything in that vector already present or cheap to expose from current bridge/model/runtime state. ([GitHub][5])

**10. exact algorithm pseudocode**

  ```text
  OFFLINE_LABEL_BUILD(I, model_snapshot):
      worlds = SAMPLE_POSTERIOR_WORLDS(I, K)     # CT-SMC if available; prior-only CT-SMC for cheapest proto
      for each (x_k, alpha_k) in worlds:
          delta_k, exit_k, safety_k = WORLD_TEACHER(I, x_k)
          store delta_k, exit_k, safety_k

      bar_delta[a] = sum_k alpha_k * delta_k[a]
      bar_safety[a] = sum_k alpha_k * safety_k[a]

      c[a] = sum_k alpha_k * 1[a == argmax_legal(delta_k)]
      v[a] = sum_k alpha_k * (delta_k[a] - bar_delta[a])^2

      a1 = argmax_legal(bar_delta)
      a2 = second_best_legal(bar_delta)
      s_eps = sum_k alpha_k * 1[delta_k[a2] >= delta_k[a1] - eps_flip]

      lambda_teach = distill_reliability(c, v, s_eps, ess_ratio(worlds))
      pi_star = consensus_shaped_exit(bar_delta, c, legal_mask(I))

      a_base = argmax_legal(policy_logits(model_snapshot, I) + beta_delta * pred_delta(model_snapshot, I))
      G = sum_k alpha_k * (delta_k[a1] - delta_k[a_base])
      y_search = 1[G > delta_g_quantile]

      emit:
          delta_q_target = clip(bar_delta / s_delta, -1, 1)
          exit_target = pi_star
          safety_residual_target = bar_safety
          target_weight = lambda_teach
          search_need_label = y_search
          gate_features = z(I)
  ```

  ```text
  TRAIN_STEP(batch):
      out = hydra_model.forward(batch.obs)

      L = core_policy_value_losses(out, batch)

      if batch.delta_q_target present:
          L += w_delta_q * weighted_masked_huber(out.delta_q, batch.delta_q_target, batch.legal_mask, batch.target_weight)

      if batch.exit_target present:
          L += w_exit * weighted_kl(log_softmax(out.policy_logits), batch.exit_target, batch.target_weight)

      if batch.safety_residual_target present:
          L += w_safety * weighted_masked_huber(out.safety_residual, batch.safety_residual_target, batch.legal_mask, batch.target_weight)

      gate_logit = gate_mlp(batch.gate_features)
      L += w_gate * BCEWithLogits(gate_logit, batch.search_need_label)

      update(L)
  ```

  ```text
  RUNTIME_DECISION(I):
      out = hydra_model.forward(I.obs)
      z = build_gate_features(I, out, mixture, ct_smc, score_ctx)

      if gate_mlp(z) > tau_gate:
          run_selective_afbs(I)
      else:
          skip_search()
  ```

**11. exact Hydra surfaces it would touch**

  * `hydra-core/src/ct_smc.rs`: use existing world sampler / ESS outputs for posterior worlds.
  * `hydra-core/src/afbs.rs`: expose shallow world teacher object used in offline label generation; optionally replace or augment `compute_ponder_priority` with learned gate output.
  * `hydra-core/src/bridge.rs`: export gate features (z), since it already aggregates mixture weights, belief fields, entropy/ESS, AFBS `delta_q`.
  * `hydra-train/src/data/sample.rs`: add `delta_q_target`, `exit_target`, `search_need_label`, `gate_features`, and per-sample `target_weight`. `delta_q_target` already supported train-side but not collated in provided sample path.
  * `hydra-train/src/data/mjai_loader.rs`: add offline hard-state replay walker that reconstructs public states and builds posterior-world teacher labels. ([GitHub][8])
  * `hydra-train/src/training/losses.rs`: reuse existing masked action losses and optional target slots; add weighted exit KL if needed.
  * `hydra-train/src/training/bc.rs` / `rl.rs`: wire upstream `exit_target` production, which reconciliation memo explicitly says missing and belongs in tranche. ([GitHub][8])

**12. prototype path**

Use cheapest path that still tests mechanism itself:

  1. restrict to **discard decisions on hard-state slices**,
  2. use (K=8) posterior worlds sampled from existing CT-SMC prior / public-count-consistent sampler,
  3. use **shallow world teacher** instead of full AFBS—e.g. current Hand-EV + exact deal-in risk + score-context scalarization,
  4. train only:

     * `delta_q`,
     * policy KL to `exit_target`,
     * gate MLP.

This keeps first test independent of fully mature AFBS teacher and independent of any new opponent-style model. Mechanism falsified if it fails even with that cheap setup.

I also checked cost arithmetic. If current searched hard-state budget roughly “1 world × 128 visit-equivalents,” then “8 worlds × 16 visit-equivalents” costs about **1.21× to 1.78× per searched hard state** when world-init overhead is 4–16 visit-equivalents. If only 5% of states are searched, that is only about **1.01× to 1.04× overall**. So ensemble teacher not automatically too expensive if it redistributes existing hard-state compute instead of blindly multiplying it.

**13. benchmark plan**

Offline first.

Primary metric:
[
\mathrm{PER}(\hat
====================

\sum_k \alpha_k
\left[
\max_a \Delta_k(a) - \Delta_k(\hat
\right],
]
posterior expected regret of chosen action.

Compare three labelers at fixed teacher compute:

  * **Baseline single exact hidden-world teacher,
  * **Baseline B:** posterior-mean teacher without consensus shaping,
  * **Candidate:** posterior-consensus teacher + learned search gate.

Report:

  * posterior expected regret,
  * policy agreement with posterior-mean best action,
  * gate AUROC / PR-AUC for (y_{\mathrm{search}}),
  * fraction of AFBS calls spent on genuinely high-(G(I)) states,
  * duplicate online delta at fixed search budget.

Key control = fixed total teacher compute. If candidate wins only because it uses more search, it fails separator claim.

**14. what success would look like**

Success =

  * lower posterior expected regret than both baselines at same teacher budget,
  * gate clearly beating current heuristic gate on predicting actual search gain,
  * online improvement at same AFBS budget because search concentrates on states where hidden-world disagreement matters,
  * and, critically, better stability of `ΔQ` / ExIt supervision rather than merely more labels.

**15. what would kill idea quickly**

Kill if any happen:

  * posterior-consensus targets collapse to almost same thing as ordinary posterior-mean targets,
  * learned gate adds no predictive lift over current heuristic AFBS gating,
  * gains disappear when teacher compute matched,
  * cheap posterior-world sampler too noisy, making consensus weights meaningless,
  * or online win comes only from gate spending more search, not from better label semantics.

**16. red-team failure analysis**

How it breaks in 4-player general-sum game: posterior expectation of one-step action deltas is not multiplayer equilibrium object. True. Response: measure success by **posterior expected regret** and duplicate match results, not by pretending target is solved game-theoretic value.

How it could violate partial observability: if you distill single hidden worlds, you absolutely leak privileged info. This technique survives only because it distills **posterior-stable** component and turns unstable component into abstention / search labels. If someone simplifies it back to “average worlds and train harder,” novelty mostly vanishes.

How it could depend on unavailable labels: full version needs offline posterior-world label builder. That builder not fully wired today. But provided code already has CT-SMC, AFBS root summaries, replay-safe loader machinery, so missing part = integration, not impossible dependency.

How novelty could be fake: if gate logistic on existing heuristics and world-ensemble distillation dropped, this reduces to “better AFBS gating,” incremental. If gate kept but target remains single-world privileged, this reduces to ordinary trust-gated search supervision. Candidate survives only as **joint** mechanism.

How simpler Hydra path might dominate it: strongest simpler path = existing narrow-tranche `ExIt + ΔQ + safety_residual` closure with heuristic hard-state gating. If that already captures almost all recoverable search gain, this candidate not separator-level and should die. ([GitHub][1])

**17. why this is more likely to matter than strongest simpler mainline alternative**

Strongest simpler alternative = current Hydra mainline: wire `exit_target`, `delta_q_target`, safety residual, then use heuristic hard-state gate. That closes loops, but still trains on labels that can be **wrong for public state** when hidden-world action rankings flip. This candidate attacks that exact error source. More likely to matter if Hydra real ceiling blocker is “search labels semantically misaligned under partial observability,” not “we need more labels.” ([GitHub][1])

**18. closest known baseline and why this does not reduce to it**

Closest known baseline: consensus-gated privileged distillation / uncertainty-weighted distillation.

Exact overlap: use disagreement to reduce trust in privileged supervision. ([arXiv][7])

Irreducible difference: here disagreement = **posterior hidden-world action disagreement for one public state**, not model disagreement or trace disagreement, and same posterior-world ensemble is used both to shape action target and to create **search-gain** label. That extra coupling is why I classify it as **B**: known mechanism family with Hydra-specific adaptation that plausibly changes capability.

**19. dependency closure table**

                                                                                                                                                                                                                                                                | required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker                                                                                                                                                              |
  | -------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 192×34 public observation tensor | already exists | `TESTING.md` status note says live contract is 192×34. ([GitHub][2]) |
                                                                                                                                                                                                                                                                | 46-action legal mask                                     | already exists                             | Game-engine action contract fixed at 46 actions. ([GitHub][9])                                                                                                                   |
| CT-SMC particle sampler + ESS | already exists | `CtSmcConfig`, ESS thresholding, weighted tile counts already exist. |
| AFBS root policy / node-Q summaries | already exists | `root_exit_policy`, `node_q_value`, ponder shell already exist. |
| `delta_q_target` train-side slot | already exists | `HydraTargets` already includes it. |
| `safety_residual_target` train-side slot | already exists | `HydraTargets` already includes it; loader already builds replay-side safety residual. |
| upstream `exit_target` production | cheap to expose / partially missing | reconciliation says upstream production belongs in tranche; current gap = integration. ([GitHub][8]) |
| `delta_q_target` collation in `MjaiBatch` | missing | provided sample path leaves `delta_q_target` absent. |
| gate feature vector (z) | cheap to expose | bridge already emits mixture weights, belief fields, entropy/ESS, AFBS `delta_q`, and risk/stress hooks. |
                                                                                                                                                                                                                                                                | posterior-world offline label builder                    | missing                                    | no explicit offline generation path wired today; reconciliation warns not to silently couple replay loading to runtime-only search context without such path. ([GitHub][8])     |

**20. minimum falsifiable prototype**

Take hard-state replay slice of discard decisions. For each public state:

  * sample (K=8) CT-SMC-consistent worlds,
  * compute shallow world-conditioned discard score (Hand-EV + exact deal-in risk + score-context term),
  * build posterior-consensus `ΔQ` / exit targets,
  * label search-worth from posterior expected regret reduction (G(I)),
  * train `delta_q` and tiny gate MLP only.

Benchmark against:

  * single-world teacher,
  * posterior-mean teacher without consensus shaping,
  * current heuristic gate.

If posterior expected regret and gate AUROC do not improve at fixed teacher compute, kill immediately.

---

  ## Technique 2

**1. name**

Archetype-coupled CT-SMC via Rao-Blackwellized opponent regime posteriors.

**2. problem solved**

Hydra has three partially separate objects:

  * hidden-tile posterior (Mixture-SIB / CT-SMC),
  * opponent-hand-type / archetype surfaces,
  * robust-opponent backup math.

Open problem: public opponent actions should update **all three together**. discard or call is evidence about concealed tiles, but meaning of that evidence depends on opponent style / plan. Keeping style and hidden tiles separate throws away exact signal Hydra claims as primary differentiator. This technique makes opponent regime posterior first-class part of CT-SMC state so style evidence sharpens tile beliefs and tile beliefs sharpen style evidence. ([GitHub][6])

**3. outside ingredients and exact sources**

Two outside sources materially mattered. Differentiable Interacting Multiple Model Particle Filtering contributed concrete idea of maintaining regime probabilities alongside particles and letting regime probability guide computation. Multiplayer opponent-modeling paper mattered because it is direct counterexample to lazy equilibrium-only posture: in multiplayer imperfect-information games, using observations of opponents can outperform equilibrium-style play. ([arXiv][10])

**4. what is borrowed unchanged**

Borrowed unchanged: IMM idea that sequential inference can maintain small regime posterior jointly with particle state, and generic idea that regime probability can be updated by action likelihoods and used to allocate downstream compute. ([arXiv][10])

**5. what is adapted for Hydra**

Adaptation: particle state not dynamical-system latent but Mahjong hidden-tile contingency allocation. Regimes not motion models but **opponent strategic archetypes / public-teacher hand-type modes**. Emission model not sensor likelihood but likelihood of opponent **public Mahjong action** given particle concealed hand allocation and regime. Resulting posterior then fed directly into Hydra existing robust-opponent code and dormant `opponent_hand_type` supervision surface. ([GitHub][6])

**6. what is genuinely novel synthesis**

Novel synthesis not merely “add opponent clusters.” It is closed Bayesian loop:

  1. public prior over regimes comes from Hydra public encoder,
  2. each CT-SMC particle carries per-opponent regime posterior,
  3. public actions update both particle weight and regime posterior,
  4. aggregated regime posterior becomes:

     * soft teacher for `opponent_hand_type`,
     * conditioning object for robust search backup,
     * and better hidden-world posterior for any belief/search target downstream.

This collapses three disconnected Hydra shelves into one object.

**7. why it fits Hydra specifically**

Hydra unusually ready for this:

  * CT-SMC already present,
  * `SearchContext` already has optional `CtSmc`, `MixtureSib`, and opponent-risk/stress hooks,
  * `robust_opponent.rs` already contains `robust_backup` and `archetype_softmin`,
  * and model/loss surfaces already include `opponent_hand_type`, though current code path does not yet produce credible target for it. ([GitHub][1])

That why this worth keeping despite being more ambitious than Technique 1: it aligns with Hydra intended final identity rather than inventing side project.

**8. exact mathematical formulation**

Let (x) denote hidden tile allocation and (r_j \in {1,\dots,R}) latent regime for opponent (j \in {1,2,3}).

Maintain (P) particles:
[
{x^{(m)}, w^{(m)}, \rho^{(m)}*{1}, \rho^{(m)}*{2}, \rho^{(m)}*{3}}*{m=1}^{P},
]
where each
[
\rho^{(m)}_{j} \in \Delta^{R-1}
]
is categorical posterior over regimes for opponent (j) inside particle (m).

Public prior from Hydra public model or stand-alone prior net:
[
q_\eta(r_j \mid H^{pub}*{j,t}).
]
Initialize:
[
\rho^{(m)}*{j,0} = q_\eta(\cdot \mid H^{pub}_{j,t})
\quad \forall m.
]

Let (o_t^j) be observed public action from opponent (j) at time (t). Let emission model be
[
\ell_\phi(o_t^j \mid I_t, x^{(m)}, r)
=====================================

p_\phi(o_t^j \mid \phi(I_t, x^{(m)}_j), r),
]
where (\phi(I_t, x^{(m)}_j)) is feature extractor over public state plus concealed-hand summary for opponent (j) implied by particle (m).

Rao-Blackwellized update for one acting opponent (j):

[
L_j^{(m)}
=========

\sum_{r=1}^{R}
\rho_{j,r}^{(m)}
\ell_\phi(o_t^j \mid I_t, x^{(m)}, r),
]

[
w^{(m)}
\leftarrow
\frac{
w^{(m)} L_j^{(m)}
}{
\sum_{m'}
w^{(m')} L_j^{(m')}
},
]

[
\rho_{j,r}^{(m)}
\leftarrow
\frac{
\rho_{j,r}^{(m)}
\ell_\phi(o_t^j \mid I_t, x^{(m)}, r)
}{
L_j^{(m)}
}.
]

If multiple opponents act before next filter step, multiply corresponding (L_j^{(m)}) factors.

Optional regime drift:
[
\rho_j^{(m)} \leftarrow T^\top \rho_j^{(m)}
]
before update, with (T=I) in first prototype.

Aggregate regime posterior:
[
\bar{\rho}_{j,r}
================

\sum_{m=1}^{P}
w^{(m)} \rho_{j,r}^{(m)}.
]

This becomes soft teacher target:
[
y^{type}*{j,r} = \bar{\rho}*{j,r}.
]

Training objective for public prior and emission model on replay events with exact hidden states:
[
L_{\mathrm{emit}}
=================

-\sum_{e}
\log
\sum_{r=1}^{R}
q_\eta(r \mid H^{pub}*e),
\ell*\phi(o_e \mid I_e, x_e, r).
]

To avoid trivial regime collapse, add mild balance regularizer:
[
L_{\mathrm{bal}}
================

\lambda_{\mathrm{bal}}
D_{KL}
\left(
\frac{1}{N}\sum_e q_\eta(\cdot \mid H^{pub}_e)
  ;\middle|;
\mathrm{Unif}(R)
\right).
]

Total training loss:
[
L = L_{\mathrm{emit}} + L_{\mathrm{bal}}.
]

Search integration: if (Q_r(a)) is action value under regime (r), use Hydra existing robust-opponent machinery:
[
\tilde{Q}(a)
============

  \min_{q \in \Delta^R,; D_{KL}(q | \bar{\rho}*j)\le \epsilon_j}
\sum*{r=1}^{R} q_r Q_r(a).
]

That exactly matches KL-ball robust backup Hydra already has math for.

Optional compute allocation per regime, adapted from IMM:
[
N_r
===

\max(N_{\min},
\left\lfloor
P \cdot
\frac{\bar{\rho}*r^\alpha (U_r+\varepsilon)^\beta}
{\sum*{r'} \bar{\rho}*{r'}^\alpha (U*{r'}+\varepsilon)^\beta}
\right\rfloor),
]
where (U_r) can be within-regime entropy or ESS deficit. I would **not** put this in minimum prototype.

**9. tensor shapes and affected network interfaces**

Runtime sidecar state:

  * existing particle allocation: ([P,34,4])
  * new regime sidecar: ([P,3,R])

At (P=128):

  * (R=4) adds (128\times3\times4 = 1536) float32s (\approx 6) KiB,
  * (R=8) adds (3072) float32s (\approx 12) KiB.

So memory overhead negligible.

Network side:

  * reuse `opponent_hand_type` as public prior head:
[
[B, 3R].
]
Current code already makes class count configurable via `opponent_hand_type_classes`. If checkpoint compatibility matters, keep (R=8); if prototype simplicity matters, set (R=4). Parameter delta from (R=8\to4) at hidden size 256 is only **3084 parameters**, so this not architecture crisis.

Train-side targets:

  * `opponent_hand_type_target`: ([B,3R])
  * optional `opponent_hand_type_mask`: ([B,3R])

Emission model:

  * discard-only prototype input:
[
\phi \in \mathbb{R}^{F_{\mathrm{opp}}},
\quad F_{\mathrm{opp}} \approx 64
]
[inference: concealed hand histogram + shanten / wait / yaku flags + public score/wall context],
  * output logits:
[
[B, R, 34]
]
for discard-tile likelihoods in discard-only prototype.

Per observed opponent action, number of regime-likelihood evaluations only (P \times R) for one acting opponent: 512 at (P=128,R=4), or 1024 at (P=128,R=8). Even three simultaneous acting-opponent updates remain small.

**10. exact algorithm pseudocode**

  ```text
  # Phase A: fit public prior + regime-conditioned emission model on replay
  for event e in replay_opponent_events:
      q = prior_model(pub_history_e)                       # [R]
      for r in 1..R:
          loglik[r] = emission_model(pub_ctx_e, exact_hidden_hand_e, r, observed_action_e)

      loss = -logsumexp(log(q) + loglik) + lambda_bal * KL(mean_batch(q), uniform_R)
      update(prior_model, emission_model)
  ```

  ```text
  # Phase B: runtime / offline coupled filter update
  for particle m in 1..P:
      for acting opponent j:
          for r in 1..R:
              l[r] = emission_model(pub_ctx_t, hidden_hand_from_particle(m, j), r, observed_action_t_j)

          L = dot(rho[m][j], l)
          rho[m][j] = rho[m][j] * l / L
          w[m] = w[m] * L

  normalize(w)

  if ESS(w) < ess_threshold:
      resample particles
      copy rho sidecars along ancestry

  for opponent j:
      bar_rho[j] = sum_m w[m] * rho[m][j]
  ```

  ```text
  # Phase C: train soft opponent-hand-type target
  for public state I in training batch:
      out = hydra_model.forward(I.obs)
      target = concat_j(bar_rho[j])          # [3R]
      loss += w_type * KL(target || softmax(out.opponent_hand_type))
  ```

  ```text
  # Phase D: search-side robust backup
  for opponent node j:
      q_per_arch = archetype_conditioned_action_values(node, j)   # [R][A]
      q_robust = archetype_softmin(q_per_arch, bar_rho[j], tau_arch or KL-ball epsilon)
      use q_robust in backup
  ```

**11. exact Hydra surfaces it would touch**

  * `hydra-core/src/ct_smc.rs`: extend particle state or add parallel sidecar for (\rho[P,3,R]); reuse existing update / ESS / resampling path.
  * `hydra-core/src/bridge.rs`: export (\bar\rho), regime entropy, and optionally regime-conditioned stress/risk summaries. `SearchContext` already has right neighborhood for this.
  * `hydra-core/src/robust_opponent.rs`: consume (\bar\rho) directly via `robust_backup` / `archetype_softmin`.
  * `hydra-core/src/afbs.rs`: optional root-only robust opponent backup first; full opponent-node semantics later.
  * `hydra-train/src/model.rs`: reuse existing configurable `opponent_hand_type` head as public regime prior.
  * `hydra-train/src/training/losses.rs`: activate `opponent_hand_type_target` loss once credible labels exist.
  * `hydra-train/src/data/sample.rs`: collate `opponent_hand_type_target`, absent in provided sample path.
  * new module `[cheap new file]` such as `hydra-core/src/opponent_emission.rs`: only genuinely missing runtime surface.

**12. prototype path**

Do not start with full joint monster. Start with discard-only falsifier:

  1. choose (R=4),
  2. train public prior + regime-conditioned **discard** likelihood model on replay events with exact hidden hands,
  3. add (\rho[P,3,R]) sidecars to offline CT-SMC only,
  4. update particle weights from observed discards only,
  5. evaluate whether hidden-world posterior calibration improves,
  6. only then turn (\bar\rho) into `opponent_hand_type_target`,
  7. only after that test root-only robust backup.

This keeps first test focused on actual claim: style-coupled filtering sharpens posterior belief.

**13. benchmark plan**

First benchmark: filtering quality, not Elo.

Compare:

  * style-agnostic CT-SMC,
  * CT-SMC + public prior only,
  * candidate coupled filter.

Hold particles fixed.

Report:

  * held-out opponent discard NLL,
  * wait-set calibration / exact-wait recall,
  * posterior log-likelihood of true concealed tile membership,
  * `opponent_hand_type` ECE / KL once that head is trained.

Second benchmark: fixed-state search slices.

  * Use same hard states with same particle count and AFBS budget.
  * Compare action quality / posterior expected regret with and without coupled filter.

Third benchmark: duplicate online play.

  * Same AFBS visits, same particle count, same evaluation seed bank.

**14. what success would look like**

Success = clean offline win before any large online claim:

  * better held-out discard likelihood,
  * better wait / concealed-tile posterior calibration,
  * nontrivial, well-calibrated (\bar\rho) rather than near-uniform mush,
  * and then root-level action improvement at fixed compute when robust backup allowed to use (\bar\rho).

If those do not happen, no reason to believe full search-side version will matter.

**15. what would kill idea quickly**

Kill if:

  * regime model collapses to one mode,
  * coupled filter does not improve posterior calibration over style-agnostic CT-SMC,
  * (\bar\rho) remains too diffuse to change any robust backup,
  * or gains disappear once particle count and AFBS visits are matched.

**16. red-team failure analysis**

How it breaks in 4-player general-sum game: exploitive style inference can backfire badly if regime posterior overconfident or opponents nonstationary. Right antidote = exactly Hydra own robust-opponent math: treat (\bar\rho) as center of KL ball, not perfect point estimate. ([arXiv][11])

How it could violate partial observability: using particle hidden hands in emission likelihood is fine; ordinary Bayesian filtering. Failure mode not “illegitimate hidden info,” but **overconfident likelihoods** collapsing particle weights. That why emission model needs calibration and first prototype should be offline only.

How it depends on missing surfaces: major missing piece = opponent-action emission model. Real gap. Everything else either already in provided code or one integration level away.

How novelty could be fake: if (\rho) depends only on public history and does **not** feed back into particle weights, this collapses to plain public-only opponent classifier. If (\rho) updates particle weights but never used in robust search or supervision, it becomes merely fancy filter variant. Full claim survives only when same posterior drives **belief**, **training target**, and **search backup**.

How simpler Hydra path might dominate it: simpler path = “train `opponent_hand_type` from public-only classifier and use fixed archetype weights in robust backup.” If that works equally well, this candidate too fancy. That why offline posterior-calibration benchmark is kill switch.

**17. why this is more likely to matter than strongest simpler mainline alternative**

Strongest simpler alternative = public-only opponent-hand-type head plus later robust residual. That leaves major Bayesian signal unused: action meaning depends on concealed hand, and concealed-hand posterior should change when style hypothesis changes. Candidate more likely to matter if Hydra edge is “read opponents better than everyone else,” because this is first mechanism here making opponent reading update hidden-world posterior directly rather than living as separate side prediction. ([GitHub][6])

**18. closest known baseline and why this does not reduce to it**

Closest known baseline: interacting-multiple-model particle filtering / mixture-of-experts sequential inference.

Exact overlap: multiple regimes, sequential regime posterior update, regime-guided compute. ([arXiv][10])

Irreducible difference: regime latent attached to Mahjong hidden-tile allocation particle filter, and resulting posterior reused for **public-teacher soft labels** and **robust search backup**. That makes it **B** candidate, not core family known, but Hydra-specific coupling is capability-relevant and does not collapse into ordinary opponent classifier.

**19. dependency closure table**

                                                                                                                                                                                                                                                                | required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker                                                                                                                               |
  | -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| CT-SMC particle state + ESS / resampling | already exists | `CtSmc` already has particles, update, ESS, weighted counts. |
| public search context hooks for risk / stress | already exists | `SearchContext` already includes opponent-risk / stress hooks. |
| robust KL-ball / archetype softmin math | already exists | `robust_backup`, `archetype_softmin`, calibration helpers already exist. |
| public prior head over opponent modes | already exists / configurable | `opponent_hand_type` head and configurable class count already exist. |
| loss slot for opponent-hand-type soft target | already exists | `HydraTargets` and `HydraLossConfig` already include it. |
| sample / batch collation for `opponent_hand_type_target` | missing | provided sample path leaves it as `None`. |
                                                                                                                                                                                                                                                                | replay-hidden opponent hand reconstruction               | cheap to expose / mostly already there     | loader already reconstructs exact waits / hidden-state-derived danger signals; extending that to opponent-action events is plausible [inference]. |
                                                                                                                                                                                                                                                                | opponent-action emission model (p(o\mid I,x,r))          | missing                                    | main new module; no direct impl surface exists yet                                                                                      |
                                                                                                                                                                                                                                                                | search-side consumption of (\bar\rho)                    | cheap to expose later                      | robust-opponent utilities and search context hooks already exist.                                                                                 |

**20. minimum falsifiable prototype**

Train discard-only regime-conditioned emission model on replay events with exact hidden hands. Then run offline CT-SMC on held-out states with and without (\rho[P,3,R]) sidecar and compare posterior calibration at same particle count.

If discard NLL, wait recall, and concealed-tile posterior quality do not improve, kill before touching AFBS or main Hydra model.

---

  * single best technique to try first:** Technique 1 — posterior-consensus ExIt/ΔQ distillation with action-straddle search gating.

  * single best cheap benchmark to run first:** offline hard-state discard suite where score = **posterior expected regret** at fixed teacher compute, comparing single-world labels, posterior-mean labels, and posterior-consensus labels, plus gate AUROC on search-worth (G(I)).

  * single biggest hidden impl risk:** posterior quality itself. Both surviving candidates depend on posterior representing **right** public uncertainty object; if Hydra current style-agnostic posterior too blurry or miscalibrated, Technique 1 can abstain on wrong states and Technique 2 can confidently sharpen wrong worlds.

  * 2-5 most tempting rejected directions and exactly why they were rejected:**

    * **Gauge-fixed / projected belief supervision alone:** important, already identified internally, but by itself not separator-level here. Written mathematically, still mostly “better belief auxiliary targets,” not capability-changing compute/teacher mechanism. ([GitHub][4])
    * **Hand-EV realism rewrite:** likely strong, already high-priority, but mainline closure rather than fresh breakthrough candidate. Improves existing Hydra bet; does not solve partial-observability teacher-misalignment problem. ([GitHub][5])
    * **Stronger endgame exactification:** promising and probably worth doing, but too slice-specific for this invention pass. High leverage in late-game states, not best first separator for Hydra current mainline bottleneck. ([GitHub][4])
    * **Confidence-gated safe exploitation residual:** attractive because cheap, but once written out mostly bounded root residual on top of archetype posterior. Without better posterior quality, incremental and brittle. ([GitHub][4])
    * **Full public-history subgame decomposition / explicit history filtering:** rejected because it fights Hydra selective-search posture and because history filtering / subgame-root construction can become intractable outside favorable domains. Too diffuse and too likely to become broad-search identity creep. ([arXiv][12])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md"
[3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md"
[4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md"
[7]: https://arxiv.org/abs/2602.20574 "https://arxiv.org/abs/2602.20574"
[8]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[9]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[10]: https://arxiv.org/abs/2410.00620 "https://arxiv.org/abs/2410.00620"
[11]: https://arxiv.org/abs/2212.06027 "https://arxiv.org/abs/2212.06027"
[12]: https://arxiv.org/abs/2311.14651 "https://arxiv.org/abs/2311.14651"
]]>
</answer_text>
</answer_section>
</combined_run_record>