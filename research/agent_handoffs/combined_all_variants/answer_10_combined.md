<combined_run_record run_id="answer_10" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 10 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_10_VALIDATE_COMPUTE_ROUTING_AND_PONDERING.md">
  <![CDATA[# Hydra prompt — validate compute routing, hard-state gating, and pondering control

Primary source material lives in the raw GitHub links below.

## Critical directive — how to read the core Hydra docs

You must avoid fragmented keyword-peeking over large architecture docs.

Required reading workflow:
1. Read the core Hydra docs holistically first.
2. Reconstruct Hydra's fast path, slow path, pondering posture, and AFBS specialist doctrine.
3. Only then inspect the exact code surfaces and outside evidence.

<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- After holistic reading, use targeted search for exact hooks and signals.
</holistic_ingestion_rules>

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/TESTING.md`
5. code-grounding files
6. outside retrieval

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `research/design/TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs

Relevant prior variant writeups and prompt references:
- `research/agent_handoffs/combined_all_variants/005_followup_compute_router_and_robustness.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/005_followup_compute_router_and_robustness.md
- `research/agent_handoffs/combined_all_variants/007_primary_agent_7.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_primary_agent_7.md
- `research/agent_handoffs/combined_all_variants/007_variant_agent_7new.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_variant_agent_7new.md
- `research/agent_handoffs/combined_all_variants/007_variant_agent_7new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_variant_agent_7new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_pack_005.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_pack_005.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_005_cross_field_transfer_hunter.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_005_cross_field_transfer_hunter.md

You are validating whether learned compute routing and pondering control is one of the best long-run Hydra investments.

The family under review is:
- value-of-computation routing
- hard-state gate replacement or upgrade
- ponder queue prioritization
- budget-aware selection between fast path, Hand-EV enrichment, AFBS variants, and endgame exactification

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior compute-router / robustness materials and only search enough to validate, falsify, and sharpen this compute-allocation lane.

Do not re-litigate Hydra's broad selective-compute doctrine. Focus only on whether a learned router or ponder allocator materially beats the current heuristic gate at equal budget.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, objective definitions, latency accounting, thresholds, or benchmark details when they matter.
</verbosity_controls>

<calculation_validation_rules>
- Use Python in bash for budget arithmetic, break-even checks, latency frontier calculations, and cheap router-threshold sanity checks.
- Do not leave cost-vs-benefit claims uncomputed when they can be checked.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart a broad cross-field idea hunt.
- New retrieval should only validate, falsify, or improve the current compute-routing family.
</tool_persistence_rules>

<dependency_checks>
- Verify which runtime signals and hooks already exist: top-2 gap, risk score, ESS, ponder cache, AFBS root stats, endgame trigger, Hand-EV summaries, score pressure, wall depth.
- Verify whether a learned router can be added without new heads.
</dependency_checks>

<self_red_team_rules>
- Ask explicitly:
  - Does this collapse to a small heuristic improvement instead of a real capability change?
  - Is the current heuristic already close to optimal for the available compute arms?
  - Are there enough decision-flip states for a learned router to matter?
  - Does the router just learn to mimic “search hard when uncertain” with extra complexity?
</self_red_team_rules>

<anti_survey_rules>
- Do not re-open broad “what should Hydra do?” exploration.
- Stay inside compute routing, hard-state gating, and pondering allocation only.
</anti_survey_rules>

<minimum_falsification_rules>
- Define the minimum offline benchmark that proves a learned router beats current heuristics at equal latency budget.
- Reject any router variant that cannot beat current heuristics offline before runtime integration.
</minimum_falsification_rules>

## What to do
1. Reconstruct Hydra's current selective-compute doctrine.
2. Validate the exact compute arms that exist now.
3. Evaluate whether a learned router is a real separator path or just minor optimization.
4. Write down the exact routing objective, feature set, pseudocode, and benchmark plan.
5. Compare router variants for on-turn compute, ponder scheduling, or both.

## Deliverables
Return, in order:
1. Hydra posture reconstruction for selective compute
2. Existing compute arms and signals
3. Surviving router formulations
4. Exact math/objective
5. Feature vector and tensor/interface shapes
6. Exact pseudocode
7. Dependency closure table
8. Minimum falsifiable offline benchmark
9. Runtime integration risks
10. Final recommendation on whether this is worth serious Hydra investment

## Hard constraints
- no broad search-everywhere proposals
- no new main model heads
- no vague “just do metareasoning” answers
- no recommendation that cannot be tested offline first at equal compute budget
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_10.md">
  <![CDATA[## 1. Methodology and revalidation scope

This is a second-pass audit of my own provisional draft for the original compute-routing / pondering prompt. I re-read the original prompt, re-checked the provisional claims I had assembled in this conversation, and then re-read the Hydra materials that were in scope for that prompt: `research/design/HYDRA_RECONCILIATION.md`, `research/design/HYDRA_FINAL.md`, `docs/GAME_ENGINE.md`, `research/design/TESTING.md`, `hydra-core/src/afbs.rs`, `hydra-core/src/bridge.rs`, `hydra-core/src/ct_smc.rs`, `hydra-core/src/endgame.rs`, `hydra-train/src/inference.rs`, and `hydra-train/src/model.rs`. I also re-read the prompt-authorized prior handoff materials `005_followup_compute_router_and_robustness.md`, `007_primary_agent_7.md`, `007_variant_agent_7new.md`, `007_variant_agent_7new1.md`, `reference_prompt_pack_005.md`, and `reference_prompt_template_005_cross_field_transfer_hunter.md`, but treated them as hypothesis input rather than repo-grounded evidence. (research/design/HYDRA_RECONCILIATION.md:20-60,117-160,187-205,285-298; research/design/HYDRA_FINAL.md:11-25,181-217,251-280; docs/GAME_ENGINE.md:7-17,122-170; research/design/TESTING.md:1-10,230-250; hydra-core/src/afbs.rs:472-507; hydra-core/src/bridge.rs:27-45,251-312; hydra-core/src/ct_smc.rs:1-18,228-258; hydra-core/src/endgame.rs:1-18,48-87; hydra-train/src/inference.rs:151-220; hydra-train/src/model.rs:9-27,83-125,247-271)

The major claims I explicitly re-checked were: whether Hydra currently has a real multi-arm runtime router; whether Hand-EV is a distinct optional arm or already baseline; whether the current heuristic gate is thin enough that a learned replacement might matter; whether the needed signals already exist without adding new model heads; and whether an equal-budget offline benchmark can be specified concretely before runtime integration. I also red-teamed the possibility that this collapses to “search when uncertain” with extra complexity, and the possibility that the current heuristic is already near-optimal for the runtime arms that actually exist today. (research/design/HYDRA_RECONCILIATION.md:20-60,117-160,187-205,285-298; hydra-train/src/inference.rs:151-220; hydra-core/src/bridge.rs:251-312)

I used bash/python sanity checks for arithmetic only: comparing tiny-router overhead against Hydra’s documented 150 ms on-turn and 50 ms call-reaction budgets; checking simple break-even `λΔt` examples for extra compute; and rough-envelope comparisons of `HYDRA_FINAL.md`’s on-turn AFBS ranges versus ponder AFBS ranges. Those calculations did not create new factual claims; they only sanity-checked the implications of the documented budgets and beam ranges. The underlying budgets and search envelopes come from the repo docs. (hydra-train/src/inference.rs:15-28,151-220; research/design/HYDRA_FINAL.md:189-217,282-288)

I classified claims with the following standard. `grounded` means directly supported by current Hydra docs/code through an exact function, field, constant, or explicit design statement. `supported inference` means a disciplined proposal that fits the grounded surfaces but is not yet implemented or directly specified. `[blocked]` means the claim depends on a missing runtime arm, missing logging/label surface, or missing integration path. `removed` means the claim failed re-checking or collapsed into a weaker already-covered conclusion. Relative to my provisional draft, I kept the “tiny learned scorer” idea, narrowed it from “general learned router” to “ponder-priority / hard-state-score replacement,” demoted live on-turn routing to future-only, and deleted the claim that Hand-EV is a distinct present-tense routing arm. (research/design/HYDRA_RECONCILIATION.md:20-60,117-160,187-205,285-298; hydra-train/src/inference.rs:151-220; hydra-core/src/bridge.rs:251-312)

## 2. What changed from the previous draft

The answer became materially smaller and narrower after revalidation.

The strongest claim got stronger: Hydra really does have the ingredients for a small learned compute-allocation scorer without adding new main model heads, because the repo already exposes a thin heuristic ponder priority, cached ponder reuse, policy-gap helpers, CT-SMC ESS, Mixture-SIB entropy/ESS, AFBS root summaries, robust-risk/stress features, and endgame activation logic. (hydra-core/src/afbs.rs:472-507,634-635; hydra-train/src/inference.rs:98-109,151-220,288-305,342-350; hydra-core/src/bridge.rs:301-355; hydra-core/src/ct_smc.rs:115-129,228-258; hydra-core/src/endgame.rs:48-87)

Two broader claims got weaker and were narrowed. First, “full multi-arm learned router now” weakened because `hydra-train/src/inference.rs` currently does not call live AFBS or endgame solving on-turn; it is effectively `ponder cache hit -> else fast network + SaF fast path`. Second, “learned router over fast / Hand-EV / AFBS / endgame” weakened because Hand-EV is already computed in the default bridge path rather than being a separate optional arm. (hydra-train/src/inference.rs:151-220; hydra-core/src/bridge.rs:251-299)

One claim was deleted outright: a distinct Hand-EV routing arm does not survive repo reality. The bridge computes public Hand-EV by default and only swaps to CT-SMC-weighted Hand-EV when CT-SMC context is present, so the present-tense arm distinction is not “with vs without Hand-EV”; it is at most “baseline Hand-EV vs enriched Hand-EV,” and that enriched path is not wired as an on-demand arm in `inference.rs`. (hydra-core/src/bridge.rs:251-299; hydra-train/src/inference.rs:151-220)

The final recommendation is therefore narrower: this is worth a serious Hydra investment only as a staged, equal-budget-validated replacement for the current ponder-priority / hard-state heuristic. It is not yet justified as a broad multi-arm routing program. That narrowing is also consistent with `HYDRA_RECONCILIATION.md`, which keeps `HYDRA_FINAL.md` as the north star but explicitly says the current repo is the baseline and recommends keeping AFBS specialist and hard-state-gated rather than immediately making deeper search the mainline. (research/design/HYDRA_RECONCILIATION.md:52-55,149-160,187-205,285-298)

## 3. Claim audit ledger

| Claim                                                                                                                                         | Status              | Evidence                                                                                                                                                                                                                                                                                                                                                                               | Why status changed or stayed                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Hydra’s intended posture is selective compute: cheap fast path by default, deeper compute on hard states, heavy use of pondering/cache reuse. | grounded            | `HYDRA_FINAL.md` describes “ExIt + Pondering + Search-as-Feature,” shallow on-turn AFBS, deep off-turn pondering, fast/slow deployment split, and extra compute when gap is small / risk high / ESS low. `HYDRA_RECONCILIATION.md` says keep AFBS specialist and hard-state-gated. ([GitHub][5])                                                                                       | Stayed. This is directly stated in docs.                                                                      |
| Current integrated runtime already exposes a real multi-arm router across fast, Hand-EV, AFBS variants, and endgame.                          | removed             | `inference.rs` only shows `lookup_ponder(hash)` and otherwise fast path `actor.policy_logits_for(...)` + `apply_saf_fast_path(...)`; no live AFBS or endgame call is present. ([GitHub][2])                                                                                                                                                                                            | Narrowed after code re-check. The design docs envision more arms than the live inference path currently uses. |
| Hand-EV is a separate present-tense routing arm.                                                                                              | removed             | `bridge.rs` computes public Hand-EV by default, swaps to CT-SMC-weighted Hand-EV only when CT-SMC is present, and tests show default `encode_observation(...)` enables the Hand-EV mask and nonzero payload. ([GitHub][4])                                                                                                                                                             | Failed revalidation. Hand-EV is baseline encoding, not a distinct arm.                                        |
| The current heuristic gate/queue score is simple enough that a learned replacement could plausibly matter.                                    | grounded            | `afbs.rs` defines `GameStateSnapshot { info_state_hash, top2_policy_gap, risk_score, particle_ess }` and `compute_ponder_priority(top2_gap, risk_score, particle_ess)`. `PonderManager::enqueue_snapshot` uses that score. ([GitHub][3])                                                                                                                                               | Stayed and became more central.                                                                               |
| Hydra already has most of the cheap signals needed for a no-new-head learned scorer.                                                          | grounded            | Existing surfaces include policy gap/entropy helpers in `inference.rs`, Mixture-SIB entropy/ESS and opponent risk/stress in `bridge.rs`, CT-SMC ESS/weighted counts in `ct_smc.rs`, AFBS root summaries in `afbs.rs`, endgame urgency/activation in `endgame.rs`, and advanced model outputs in `model.rs`. `HYDRA_RECONCILIATION.md` also says not to invent new heads. ([GitHub][2]) | Stayed.                                                                                                       |
| The best immediate learned-routing target is ponder queue prioritization / hard-state score replacement, not full on-turn arm selection.      | supported inference | Repo-grounded pieces: `compute_ponder_priority`, `PonderManager`, `PonderCache`, `lookup_ponder`, and docs’ “75% idle time” / pondering emphasis. Outside evidence supports cost-aware compute allocation in settings with expensive deliberation. ([GitHub][3])                                                                                                                       | Strengthened by revalidation because the present runtime arm set is thin.                                     |
| A future binary live-AFBS gate could be worthwhile once live AFBS is callable from `infer_with_budget`.                                       | supported inference | Docs specify hard-state criteria and latency budgets for on-turn AFBS; current `inference.rs` lacks the call site. ([GitHub][5])                                                                                                                                                                                                                                                       | Narrowed to future-only because the runtime arm is not integrated now.                                        |
| A full multi-arm router over AFBS small / AFBS large / endgame is currently actionable.                                                       | [blocked]           | `HYDRA_FINAL.md` defines those conceptual arms and budgets, but the current inference server does not invoke them. ([GitHub][5])                                                                                                                                                                                                                                                       | Demoted because the arm-selection surface is missing in the live path.                                        |
| A learned router can be added without new main model heads.                                                                                   | grounded            | `HYDRA_RECONCILIATION.md` says not to invent new main heads, while `model.rs` and `bridge.rs` already expose rich auxiliary signals. ([GitHub][1])                                                                                                                                                                                                                                     | Stayed.                                                                                                       |
| The minimum acceptable proof is an equal-budget offline benchmark; if a simple heuristic or logistic matches, kill the router.                | supported inference | This matches the repo’s general validation ethos in `TESTING.md` and the prompt-authorized compute-router handoff proposals, but is not itself a current repo benchmark. ([GitHub][6])                                                                                                                                                                                                 | Stayed, but clearly labeled proposal rather than repo-grounded fact.                                          |

## 4. Revalidated core answer

### Grounded findings

Hydra’s selective-compute doctrine is real, but it exists at two different levels: the design-doc north star and the current runtime baseline. The design docs describe a system centered on `ExIt + Pondering + Search-as-Feature`, with shallow on-turn AFBS, deeper off-turn pondering, predictive subtree reuse, and extra compute specifically on close, risky, or low-ESS states. `HYDRA_RECONCILIATION.md` then narrows that posture for the present repo, explicitly recommending AFBS remain a specialist module that is hard-state-gated rather than an always-on mainline search. ([GitHub][5])

The current integrated inference runtime is much thinner than the full design space. In `hydra-train/src/inference.rs`, the operative choice is: reuse a cached ponder result if the info-state hash hits, otherwise run the actor on the encoded observation and apply the SaF fast path. That file exposes helpers like `policy_top2_gap`, `policy_top1_confidence`, `compute_entropy_from_logits`, and `needs_search`, but it does not show live on-turn invocation of AFBS variants or endgame exactification. ([GitHub][2])

The current heuristic compute-allocation logic is correspondingly small. `hydra-core/src/afbs.rs` defines a `GameStateSnapshot` with only `info_state_hash`, `top2_policy_gap`, `risk_score`, and `particle_ess`, and `compute_ponder_priority` is a simple additive formula over those three scalar diagnostics. `PonderManager::enqueue_snapshot` uses that score to rank pondering work, and `PonderCache` stores `PonderResult` objects carrying `exit_policy`, value, depth, and visit count. ([GitHub][3])

The signal surface is richer than the current heuristic uses. `bridge.rs` can already build Group C search/belief features from Mixture-SIB weights, entropy, and ESS; AFBS root delta-Q summaries; opponent risk planes; and opponent stress values. `ct_smc.rs` already exposes ESS ratios and belief-weighted remaining counts. `endgame.rs` exposes `urgency(wall,danger)` and `should_activate(wall_remaining, has_threat)`. `model.rs` already exposes auxiliary outputs such as `belief_fields`, `mixture_weight_logits`, `delta_q`, and `safety_residual`. None of that requires adding a new main network head just to train a tiny router. ([GitHub][4])

A key correction from revalidation is that Hand-EV is not a separate current routing arm. The bridge computes public-state Hand-EV by default and only switches to CT-SMC-weighted Hand-EV when CT-SMC is present. The default encoding tests explicitly check that the Hand-EV mask is enabled and that Hand-EV payload is nonzero even without any search context. ([GitHub][4])

### Supported inference

Given those grounded facts, the strongest surviving compute-routing investment is not a grand multi-arm metareasoner. It is a small value-of-computation scorer that replaces the current hand-written ponder-priority / hard-state heuristic with something trained offline on equal-budget labels. This is the closest thing to a real capability change that still fits current repo reality. It uses existing signals, touches an already-live decision point, and does not require new heads or a redesign of the inference stack. Similar cost-aware decision-time planning work in the literature supports the general idea that fixed heuristics for allocating expensive deliberation can leave performance on the table, but that only matters here because Hydra already has a live ponder queue to optimize. ([GitHub][3])

A future binary live-AFBS gate can also survive as a disciplined proposal, but only as a second stage. The design docs specify what a hard state is supposed to look like and what the latency envelope is supposed to be, yet the current inference file does not actually invoke on-turn AFBS. So there is no honest present-tense claim that a live learned on-turn router can be integrated today without first exposing the missing arm. ([GitHub][5])

### Blocked / missing evidence

A full learned router over `{fast, Hand-EV enrichment, AFBS variants, endgame exactification}` is blocked by current runtime surfaces. Hand-EV is baseline rather than a separate arm; live AFBS and endgame are specified in design docs but not wired through `infer_with_budget`; and the repo does not already provide the per-state utility labels needed to train a cost-sensitive arm selector. The concept is not incoherent, but it is not currently repo-grounded enough to recommend as the immediate investment. ([GitHub][2])

## 5. Surviving mechanisms / recommendations / policies

### 5.1 Learned ponder-priority / hard-state score replacement

#### Grounded findings

Hydra already has an explicit place where cheap routing happens: `compute_ponder_priority` and `PonderManager::enqueue_snapshot`. The current score depends on only three diagnostics plus the hash. The docs also make pondering operationally important by treating idle time as a major compute reservoir and by making subtree reuse part of the deployment profile. ([GitHub][3])

#### Supported inference

This should be the first and main surviving recommendation. Replace the scalar heuristic with a tiny cost-sensitive scorer trained to predict the value of spending ponder compute on a state, under a fixed budget. Start with a linear or very small MLP model over a compact feature vector assembled from signals Hydra already computes. Do not start with a large router, bandit framework, or new network head.

#### Blocked / missing evidence

The missing piece is not runtime inference cost; a tiny scorer is negligible against Hydra’s documented 150 ms / 50 ms budgets. The missing piece is offline labels: the repo does not yet hand you `VOC(state)` targets for free. Those have to be built by an offline ladder that compares no-extra-compute vs fixed ponder compute under the same evaluation protocol. ([GitHub][2])

### 5.2 Future binary live-AFBS gate

#### Grounded findings

The docs define hard states using policy-gap, risk, and particle-ESS signals, and they set a latency envelope for on-turn AFBS. `afbs.rs` and `bridge.rs` also expose the root statistics and delta-Q summaries you would want once search is actually running. ([GitHub][5])

#### Supported inference

Once `infer_with_budget` can actually call a shallow AFBS arm, reuse the same learned score as a binary gate: fast path if predicted net value is non-positive, shallow AFBS otherwise. This is the smallest credible on-turn learned-routing extension.

#### Blocked / missing evidence

Do not build this first. It is blocked by the absence of a live AFBS call path in the current inference server. ([GitHub][2])

### 5.3 Do not prioritize a full multi-arm router yet

#### Grounded findings

The full arm menu is stronger in `HYDRA_FINAL.md` than in the current runtime. The design doc has shallow/deep AFBS regimes and endgame exactification, but the live inference file does not select among them on-turn. ([GitHub][5])

#### Supported inference

A per-instance algorithm portfolio can be powerful when the arms are truly heterogeneous; SATzilla is a classic example of that general phenomenon. But that argument only applies once Hydra actually exposes heterogeneous callable arms at inference time. ([s.aaai.org][7])

#### Blocked / missing evidence

Right now this would mostly be a paper design, not a runtime improvement plan.

## 6. Rejected or demoted ideas

### Grounded findings

The distinct “Hand-EV routing arm” was rejected because current `bridge.rs` behavior contradicts it. Public Hand-EV is already in the baseline encoding path, and CT-SMC only enriches it when available. That makes the earlier arm decomposition too broad for the current repo. ([GitHub][4])

The immediate “full multi-arm router” was demoted because `inference.rs` does not currently expose live AFBS small, AFBS large, or endgame as on-turn callable arms. Without those arms, a multi-arm selector is not a real present-tense mechanism. ([GitHub][2])

### Supported inference

The idea “router learns something deeper than ‘search when uncertain’” survives only conditionally. If a simple threshold or logistic model over the current core signals—top-2 gap, risk, ESS, and maybe wall/urgency—matches the learned scorer at equal budget, then the supposed routing capability collapses to heuristic tuning and should be killed. This demotion criterion came out stronger after revalidation, not weaker. ([GitHub][3])

The idea of a sophisticated joint on-turn plus ponder scheduler was also demoted. It may be long-run sensible, but with current repo surfaces it would mostly be speculative coordination logic sitting on top of one live decision point and several unintegrated future arms.

### Blocked / missing evidence

Anything that assumes new main model heads is rejected by scope and by `HYDRA_RECONCILIATION.md`. Anything that assumes current per-mode utility labels already exist in mainline is blocked by missing label/provenance surfaces. (research/design/HYDRA_RECONCILIATION.md:120-160,187-205,244-263)

## 7. Exact math, tensor/interface shapes, and pseudocode for surviving mechanisms

### 7.1 Surviving mechanism A: learned ponder-priority / hard-state score

#### Grounded findings

Current routing score:
[
h_{\text{heur}}(s)
==================

10 \cdot \max(0, 0.1 - g(s))
+
\max(0, r(s))
+
\max(0, 1 - e(s)),
]
where `g(s)=top2_policy_gap`, `r(s)=risk_score`, and `e(s)=particle_ess`, exactly as encoded by `compute_ponder_priority(top2_gap, risk_score, particle_ess)` in `afbs.rs`. The current queue input surface is therefore a hash plus three scalar diagnostics. (hydra-core/src/afbs.rs:472-507)

#### Supported inference

Use the smallest cost-sensitive replacement that is still a real learned scorer:

[
\mathrm{VOC}_{\text{ponder}}(s)
===============================

## u_{\text{ponder}}(s)

## u_{\text{fast}}(s)

\lambda \big(c_{\text{ponder}}(s)-c_{\text{fast}}(s)\big).
]

Definitions:

* (u_{\text{fast}}(s)): utility of the fast-path decision on state (s).
* (u_{\text{ponder}}(s)): utility after allocating the fixed ponder budget to (s).
* (c_{\text{fast}}(s)), (c_{\text{ponder}}(s)): realized milliseconds or normalized compute tokens.
* (\lambda): cost-per-ms tradeoff chosen so that the learned policy is compared at equal average budget to the heuristic baseline.

Train a scalar predictor
[
\hat v_\phi(s)=f_\phi(x(s),m(s))
]
to regress (\mathrm{VOC}_{\text{ponder}}(s)), where (x(s)) is a compact feature vector and (m(s)) is a presence mask.

A strict v1 feature interface that fits current surfaces is:

```rust
pub struct RoutingFeaturesV1 {
    pub info_state_hash: u64,
    pub scalar: [f32; 10],
    pub present: [u8; 10],
}
```

Recommended `scalar` order:

1. `top2_policy_gap`
2. `policy_entropy`
3. `risk_score`
4. `particle_ess`
5. `mixture_entropy`
6. `mixture_ess`
7. `ct_smc_ess_ratio`
8. `max_opponent_stress`
9. `max_opponent_risk`
10. `endgame_urgency`

The first four are already close to the current runtime gate surface; 5–10 are cheap summaries available from `bridge.rs`, `ct_smc.rs`, and `endgame.rs`, with presence bits allowing zero-filled fallback when they are unavailable. The batch tensor shape for the scorer is then:

* scalar tensor: ([B,10])
* mask tensor: ([B,10])
* concatenated scorer input: ([B,20])
* scorer output: ([B,1])

A deliberately narrow v1 scorer is linear:
[
\hat v_\phi(s) = w^\top [x(s);;m(s)] + b.
]

Loss:
[
\mathcal L(\phi)=\frac1N\sum_{i=1}^N \mathrm{Huber}\big(\hat v_\phi(s_i)-\mathrm{VOC}_{\text{ponder}}(s_i)\big).
]

Runtime policy:
[
\text{priority}(s)=\hat v_\phi(s).
]

If (\hat v_\phi) is not better than a linear/logistic baseline over the current three-feature heuristic surface, stop there and do not upgrade to an MLP. This keeps the mechanism honest. The general value-of-computation formulation is also consistent with the prompt-authorized compute-router handoff, but that prior material remains proposal-level rather than repo-grounded. (hydra-core/src/bridge.rs:301-355)

Exact runtime pseudocode:

```rust
fn build_routing_features_v1(state: &StateCtx) -> RoutingFeaturesV1 {
    let mut x = [0.0f32; 10];
    let mut m = [0u8; 10];

    x[0] = state.top2_policy_gap;          m[0] = 1;
    x[1] = state.policy_entropy;           m[1] = 1;
    x[2] = state.risk_score;               m[2] = 1;
    x[3] = state.particle_ess;             m[3] = 1;

    if let Some(v) = state.mixture_entropy { x[4] = v; m[4] = 1; }
    if let Some(v) = state.mixture_ess { x[5] = v; m[5] = 1; }
    if let Some(v) = state.ct_smc_ess_ratio { x[6] = v; m[6] = 1; }
    if let Some(v) = state.max_opponent_stress { x[7] = v; m[7] = 1; }
    if let Some(v) = state.max_opponent_risk { x[8] = v; m[8] = 1; }
    if let Some(v) = state.endgame_urgency { x[9] = v; m[9] = 1; }

    RoutingFeaturesV1 {
        info_state_hash: state.info_state_hash,
        scalar: x,
        present: m,
    }
}

fn enqueue_snapshot_router(pm: &mut PonderManager, feat: RoutingFeaturesV1) {
    let score = router_predict_linear(&feat.scalar, &feat.present);
    pm.enqueue_with_score(feat.info_state_hash, score);
}
```

#### Blocked / missing evidence

The blocked part is label generation, not inference cost. The repo does not already define `u_fast`, `u_ponder`, or a route-regret dataset. Those must be created offline.

### 7.2 Surviving mechanism B: future binary live-AFBS gate

#### Grounded findings

The docs define hard-state compute triggers and an on-turn latency envelope for AFBS, but `infer_with_budget` does not currently call AFBS. (research/design/HYDRA_FINAL.md:189-217,263-264,282-288; hydra-train/src/inference.rs:151-180)

#### Supported inference

Once a shallow AFBS arm exists in the live inference path, reuse the same feature interface and define:

[
\mathrm{VOC}_{\text{AFBS}}(s)
=============================

## u_{\text{afbs}}(s)

## u_{\text{fast}}(s)

\lambda \big(c_{\text{afbs}}(s)-c_{\text{fast}}(s)\big).
]

Binary gate:
[
g_\phi(s)=\mathbf 1[\hat v_\phi(s) > 0].
]

Exact output shape:

* input: ([B,20])
* output: either scalar ([B,1]) thresholded at zero, or logits ([B,2]) for `FAST` vs `AFBS_SHALLOW`.

Exact pseudocode:

```rust
fn infer_with_budget_router(state: &StateCtx, budget_ms: u32) -> Action {
    if let Some(pondered) = lookup_ponder(state.info_state_hash) {
        return sample_from_exit_policy(pondered.exit_policy);
    }

    let feat = build_routing_features_v1(state);
    let score = router_predict_linear(&feat.scalar, &feat.present);

    if score > 0.0 && budget_ms >= SHALLOW_AFBS_BUDGET_MS {
        return infer_via_shallow_afbs(state, SHALLOW_AFBS_BUDGET_MS);
    } else {
        return infer_via_fast_path(state);
    }
}
```

#### Blocked / missing evidence

This remains future-only until `infer_via_shallow_afbs(...)` actually exists in `hydra-train/src/inference.rs`.

## 8. Minimum falsifiable benchmark or validation plan

### Grounded findings

Hydra’s docs already insist on strong offline validation and explicit latency envelopes. `HYDRA_FINAL.md` gives concrete latency targets for AFBS, CT-SMC DP, and endgame exactification, and it defines hard-state triggers around close policy gap, high risk, and low ESS. It also proposes a 200K stratified-state gate for deep-AFBS validation and a 50K endgame-position gate for exactification. `TESTING.md` strongly emphasizes deterministic verification against independent ground truth for anything that affects training data. ([GitHub][5])

### Supported inference

The minimum falsifier for the surviving queue scorer is:

1. Build a fixed offline state bank stratified into at least the slices Hydra already cares about: close top-2 gap, high defensive risk, low ESS, and late-wall / endgame-pressure states.
2. For each state (s), compute the current heuristic queue score and the candidate learned score.
3. Under a fixed total ponder budget, simulate queue selection with the current heuristic and with the learned scorer.
4. Label each state with (\mathrm{VOC}_{\text{ponder}}(s)) from a reference ladder: no-extra-compute baseline vs fixed ponder budget, using the same evaluation utility and the same compute accounting for both policies.
5. Promote only if the learned scorer beats the heuristic on mean net utility at equal average total compute, with confidence intervals that do not overlap materially.
6. Kill immediately if a simple 3-feature or 4-feature threshold/logistic baseline matches within confidence intervals, or if gains disappear after matching mean and p95 compute.

That is the smallest honest benchmark because it tests the thing we actually want: better allocation of the same compute, not simply more compute. The arm ladder math is consistent with the prompt-authorized compute-router handoff, but the benchmark itself is a proposal, not a current repo artifact. ([GitHub][8])

For the future live-AFBS gate, the minimum falsifier is even simpler: on a fixed hard-state bank, compare the current doc-style heuristic gate (`gap < 0.1` or high risk or low ESS) to the learned binary gate under matched average milliseconds. If the learned gate does not reduce regret at equal budget, do not integrate it. ([GitHub][5])

### Blocked / missing evidence

The repo does not yet provide the route-label dataset, the reference ladder implementation, or the equal-budget evaluation harness. Those are the missing validation surfaces.

## 9. Dependency closure table

| Dependency                                                | Status                       | Evidence                                                                                                              | Notes                                                                   |
| --------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `top2_policy_gap` helper                                  | already present              | `policy_top2_gap`, `needs_search` in `inference.rs`; `GameStateSnapshot.top2_policy_gap` in `afbs.rs`. ([GitHub][2])  | Core v1 feature.                                                        |
| `risk_score` and `particle_ess` snapshot fields           | already present              | `GameStateSnapshot` and `compute_ponder_priority`. ([GitHub][3])                                                      | Core v1 features.                                                       |
| Ponder queue and cache                                    | already present              | `PonderManager`, `PonderCache`, `lookup`, `enqueue_snapshot`, `PonderResult`. ([GitHub][3])                           | Immediate insertion point.                                              |
| Mixture entropy / ESS                                     | already present              | `build_search_features` fills `mixture_entropy`, `mixture_ess`. ([GitHub][4])                                         | Easy v1 feature.                                                        |
| Opponent risk / stress summaries                          | already present              | `build_search_features` populates `opponent_risk` and `opponent_stress`. ([GitHub][4])                                | Easy v1 feature after scalar reduction.                                 |
| CT-SMC ESS ratio                                          | already present              | `ess_ratio()`, `needs_resample()`. ([GitHub][9])                                                                      | Easy v1 feature.                                                        |
| AFBS root stats and delta-Q                               | already present              | `node_q_value`, `find_child_by_action`, `root_visit_count`, `max_depth`, `summary`, `root_exit_policy`. ([GitHub][3]) | Useful for v2 or post-search logging.                                   |
| Endgame urgency / trigger                                 | already present              | `urgency(wall,danger)` and `should_activate(wall_remaining, has_threat)`. ([GitHub][10])                              | Easy extension point once wall count is threaded into routing features. |
| Score-pressure features                                   | easy extension point         | `GameMetadata.scores`, `kyoku_index`, `honba`, `kyotaku` exist in bridge metadata. ([GitHub][11])                     | Not needed for v1.                                                      |
| Tiny external router model                                | easy extension point         | No new main heads needed; existing auxiliary surfaces already available. ([GitHub][1])                                | Keep tiny and external.                                                 |
| Live AFBS call from `infer_with_budget`                   | [blocked by missing surface] | `inference.rs` does not show on-turn AFBS invocation. ([GitHub][2])                                                   | Blocks binary on-turn gate.                                             |
| Live endgame exactification call from `infer_with_budget` | [blocked by missing surface] | Endgame solver exists, but no call path is shown in the live inference server. ([GitHub][10])                         | Blocks full multi-arm router.                                           |
| Offline `VOC(state)` labels / route-regret dataset        | [blocked by missing surface] | No current repo surface provides these labels directly.                                                               | Needed to train any learned scorer honestly.                            |

## 10. Final recommendation / final narrowed conclusion

After strict revalidation, the answer is narrower than my provisional draft.

A broad learned compute router is **not** currently one of the best serious Hydra investments in its full original form. The repo does not yet expose the necessary live arm menu: `inference.rs` is basically `ponder-hit reuse -> else fast path`, Hand-EV is already baseline rather than a separate arm, and live AFBS/endgame arm selection is not wired through the current inference server. So the “full router over fast / Hand-EV / AFBS variants / endgame” story does **not** survive as a present-tense recommendation. ([GitHub][2])

What **does** survive, and is worth a serious but tightly scoped investment, is a **tiny learned value-of-computation scorer for ponder prioritization / hard-state scoring**. That recommendation is grounded because Hydra already has the decision point (`compute_ponder_priority` / `enqueue_snapshot`), the cache and reuse path (`PonderCache`, `lookup_ponder`), and the cheap signals needed to improve over the current three-term heuristic without adding new model heads. It is also the compute-allocation surface most aligned with Hydra’s actual current runtime, because pondering and cache reuse are already central in both the design docs and the code. ([GitHub][3])

So the narrowed recommendation is:

1. **Do now:** offline-train and offline-validate a tiny learned replacement for the current ponder-priority / hard-state heuristic, using existing signals only.
2. **Promote only if:** it wins on an equal-budget offline benchmark against the current heuristic and against a very small logistic/threshold baseline.
3. **Do next, only if step 1 wins and live AFBS is integrated:** reuse that scorer as a binary on-turn AFBS gate.
4. **Do not do yet:** a full multi-arm router over AFBS sizes and endgame exactification.

That is the tightest answer that remains true after revalidation.

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/inference.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/inference.rs"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md"
[7]: https://s.aaai.org/Library/JAIR/Vol32/jair32-014.php "https://s.aaai.org/Library/JAIR/Vol32/jair32-014.php"
[8]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/005_followup_compute_router_and_robustness.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/005_followup_compute_router_and_robustness.md"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs"
[10]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs"
[11]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
