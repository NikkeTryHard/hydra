<combined_run_record run_id="answer_16" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 16 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_16_VALIDATE_ROLLOUT_DISTILLATION_GATES_AND_FALLBACKS.md">
  <![CDATA[# Hydra prompt — validate rollout-distillation quality gates, trust boundaries, and fallback protocol

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a narrow operational prompt for one of Hydra's biggest real risks: the rollout net making search worse instead of cheaper.

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior combined answers and current doctrine, and use new retrieval only to validate, falsify, or sharpen rollout-distillation gates, trust boundaries, and fallback behavior.

Do not treat this as a broad architecture or breakthrough prompt.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/TESTING.md`
5. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
6. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
7. code-grounding files
8. outside retrieval only if needed to validate distillation / search-quality gating

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-train/src/eval.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/eval.rs
- `hydra-train/src/config.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/config.rs

Relevant prior answers and references:
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md

You are validating Hydra's rollout-distillation operational doctrine, specifically:
- when the rollout net is allowed to participate in AFBS
- when LearnerNet must remain the search-quality anchor
- what quantitative gates decide whether rollout distillation is acceptable
- what exact fallback protocol Hydra should use if rollout drift is too high

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit trust gates, fallback conditions, latency/quality tradeoffs, or benchmark thresholds when they matter.
</verbosity_controls>

<tool_persistence_rules>
- Do not restart broad Hydra future-planning.
- New retrieval should only validate, falsify, or sharpen rollout-distillation trust boundaries and fallback policy.
- Use Python in bash for latency/quality arithmetic, drift thresholds, and break-even checks when helpful.
</tool_persistence_rules>

<dependency_checks>
- Verify the intended roles of ActorNet, LearnerNet, and RolloutNet from the current docs.
- Verify which search modes depend on rollout quality and which should stay learner-anchored.
- Verify where current repo/runtime surfaces could carry rollout-quality metrics or fallback decisions.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in the provided docs/code.
- Mark any unevidenced runtime switch, gate metric, or fallback protocol as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Does the rollout net silently become the real quality anchor for decisive search states?
  - Is the distillation gate too weak to catch quality collapse?
  - Would falling back to LearnerNet-only hard-state search be safer than a degraded rollout path?
  - Are throughput gains large enough to justify the extra quality risk?
</self_red_team_rules>

<minimum_falsification_rules>
- Define the exact minimum benchmark that would prove rollout distillation is acceptable.
- If drift or search-quality loss cannot be bounded by a concrete gate, recommend disabling or narrowing rollout-net usage.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a generic distillation paper survey.
- Stay inside Hydra's rollout-net trust policy, gates, and fallback behavior.
</anti_survey_rules>

## What to do
1. Reconstruct the intended ActorNet / LearnerNet / RolloutNet split.
2. Define the exact trust boundary for rollout-net usage.
3. Define the exact fallback protocol when rollout quality is not good enough.
4. Give the gating metrics, thresholds, and benchmark plan that would make this safe.
5. Decide whether rollout-net usage should stay optional, narrow, or central.

## Deliverables
1. Hydra posture reconstruction for actor / learner / rollout roles
2. Search-quality trust boundary
3. Exact fallback protocol
4. Gate metrics and thresholds
5. Minimum falsifiable benchmark plan
6. Dependency closure table
7. Final recommendation: central, narrow, optional, or disable-until-better

## Hard constraints
- no generic “distillation is usually fine” answers
- no broad architecture resets
- no pretending rollout quality can be inferred without explicit gates
- no recommendation that lets the rollout net quietly become the decisive search-quality anchor
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_16.md">
  <![CDATA[## 1. Hydra posture reconstruction for actor / learner / rollout roles

`HYDRA_FINAL.md` defines Hydra’s current intended split as a two-tier system. **LearnerNet** is the 24-block, ~10M model used for training and for **deep AFBS on hard positions**. **ActorNet** is the 12-block, ~5M model used for self-play data generation and shallow SaF features, with continuous Learner→Actor distillation every 1–2 minutes. The current training code matches that two-model shape: `HydraModelConfig::learner()` is 24 blocks, `HydraModelConfig::actor()` is 12 blocks, and `PipelineState` versions the learner and actor separately. ([GitHub][1])

`HYDRA_FINAL.md` then adds a **Phase-2 RolloutNet**: ActorNet-sized, continuously distilled from LearnerNet, same input encoding, intended for **fast AFBS rollouts**. That is the doctrine. But in the inspected code-grounding files, the surfaced runtime objects are still centered on learner/actor, not learner/actor/rollout: the model and config surfaces expose actor and learner constructors and versioning, and the benchmark surface tracks global distillation drift, not a rollout-specific runtime identity or provenance path. So today RolloutNet is present as a **design intent**, not as a first-class operationally closed surface in the inspected repo. ([GitHub][1])

`HYDRA_RECONCILIATION.md` narrows the doctrine further. Broad “search everywhere” AFBS rollout is explicitly removed from the active path; AFBS is supposed to be **specialist and hard-state gated**, not the broad runtime identity of Hydra. `HYDRA_FINAL.md` is consistent with that narrowing: the **fast path** is network forward + SaF adaptor, the **slow path** is reuse of a pondered AFBS subtree, and the deep AFBS budget belongs on **hard positions only**. That makes the intended hierarchy clear: **LearnerNet is the search-quality anchor**, **ActorNet is the cheap production/self-play workhorse**, and **RolloutNet, if introduced, is only an acceleration layer subordinate to LearnerNet rather than a co-equal decision authority**. The last clause is an inference, but it is the only reading that is consistent with both the reconciliation doctrine and the current repo surfaces. ([GitHub][2])

A crucial operational point follows from that reconstruction: **ActorNet and RolloutNet should not be conflated just because both are 12-block-sized**. ActorNet’s documented role is self-play generation and shallow runtime features. RolloutNet’s documented role is fast AFBS support. The inspected code currently gives Hydra a learner/actor split, not a learner/actor/rollout split. So using “the cheap 12-block net” as an un-gated AFBS authority would not be a neutral implementation detail; it would collapse two different trust roles into one path without a dedicated gate or provenance boundary. That is exactly the kind of silent trust drift the user is warning about. ([GitHub][1])

## 2. Search-quality trust boundary

In Hydra’s AFBS code, the network that supplies priors and values is not a harmless helper; it directly shapes the search decision. `puct_select` ranks children using **prior** and **q_value**, expansion keeps a top-k slice of network policy mass and renormalizes it, backpropagation accumulates value into visit statistics, `root_exit_policy` is computed from child q-values, and `best_action` is chosen by visit count. So if a rollout net is allowed to supply decisive priors or values in important states, it does not merely make search cheaper; it becomes the **real quality anchor** for that search outcome. That answers one of the red-team questions directly: **yes, rollout silently becomes the real anchor if it is allowed to drive decisive AFBS states**. ([GitHub][3])

The trust problem is broader than root move selection. `bridge.rs` converts AFBS root Q differences into `delta_q` search features, and `model.rs` already exposes `delta_q` and `safety_residual` heads. That means a weak rollout path can damage Hydra in two ways at once: it can distort the **current** decision through AFBS, and it can distort **future supervision/features** if search-derived signals are exported from that path. `TESTING.md` is explicit about the failure mode: bad labels silently train a model that becomes “confidently wrong.” So rollout quality is not just a runtime issue; it is a training-data trust issue. ([GitHub][4])

Hydra’s own doctrine and runtime already identify the right boundary variables. `HYDRA_FINAL.md` says hard positions are characterized by a **small top-2 policy gap**, **high-risk defense**, or **low particle ESS**. The runtime surfaces exactly those signals in `GameStateSnapshot`: `top2_policy_gap`, `risk_score`, and `particle_ess`. The code’s own `compute_ponder_priority` already prioritizes states when gap is low, risk is high, and ESS is weak. So Hydra already encodes the geometry of “states where cheap approximation is most dangerous.” Those are precisely the states where **LearnerNet must remain the anchor**. ([GitHub][1])

That gives the exact trust boundary:

1. **LearnerNet-only zone (must remain learner-anchored):**
   all hard states; all on-turn decisive AFBS roots; all search-derived label export (`exit_policy`, `delta_q`); and any cached subtree whose provenance is unknown. This is grounded in the doctrine that AFBS is specialist/hard-state gated and in the AFBS code path that turns priors/q-values into root action. ([GitHub][2])

2. **Rollout-forbidden zone today:**
   anything that can directly decide the root action, root exit policy, root visit allocation, or exported search label on those states. This is an inference from the AFBS mechanics, but it is a necessary inference, not a speculative one. If rollout is allowed there, LearnerNet is no longer the practical anchor. ([GitHub][3])

3. **Rollout-permissible zone later, if ever [inference]:**
   non-hard, non-decisive acceleration only—such as off-turn pondering or inner-loop rollout assistance—provided the **final on-turn root is revalidated by LearnerNet before action emission or label export**. This is the narrowest interpretation that preserves Hydra’s stated doctrine. ([GitHub][2])

The boundary is tighter in the current repo than it would be in a fully instrumented rollout system, because the cache/provenance layer is incomplete. `SearchContext` is optional and the encoder has a default empty context, so Hydra already has a clean **learner-only / no-search** fallback surface. But `PonderResult` records exit policy, value, depth, visits, and timestamp—not the source network. So if rollout-generated subtrees existed, the current surfaced structure would not let Hydra distinguish them from learner-generated subtrees at reuse time. That means cached AFBS results touched by rollout are **operationally untrusted** until provenance is added. ([GitHub][4])

The prior combined answers add plausible refinements—`wall <= 10`, `orasu`, and tighter endgame/context gates—but those refinements are not surfaced in the inspected `GameStateSnapshot`. So those context enrichments are reasonable, but they are **`[blocked]` for automatic runtime gating** until the corresponding signals are explicitly plumbed into the same decision surface that currently holds gap/risk/ESS. ([GitHub][5])

Which search modes actually depend on rollout quality is therefore straightforward. **AFBS root/leaf evaluation, pondered subtree reuse, and any exported `delta_q`/exit labels** are rollout-sensitive if rollout is in the loop. **Fast path without SearchContext, CT-SMC DP, and endgame solver gates** are not rollout-sensitive in the inspected surfaces. So the correct trust rule is not “rollout is fine if average KL is small”; it is “rollout is only admissible where its errors cannot quietly become decisive action or supervision errors.” ([GitHub][4])

## 3. Exact fallback protocol

The safest exact fallback protocol is **not** “keep using rollout but with reduced trust.” It is: **prefer LearnerNet-only hard-state search; otherwise fall back to the existing fast path.** That matches both the doctrine and the runtime surfaces. ([GitHub][2])

**Step 1 — Default live posture now:** set rollout participation in live AFBS to **disabled**. A rollout path may exist only in shadow evaluation **[inference]**, because the inspected runtime surfaces do not expose a dedicated rollout version/provenance path, and the current cache surface cannot distinguish rollout-generated subtrees from learner-generated ones. ([GitHub][1])

**Step 2 — Per-state hard-state classifier:** before any rollout participation, compute a hard-state bit from the currently surfaced runtime signals:

```text
hard =
    (top2_policy_gap < 0.10)
 or (risk_score > 0.08)         [inference]
 or (particle_ess < 0.55)       [inference]
```

`top2_policy_gap < 0.10` is directly aligned with `HYDRA_FINAL.md`. The `0.08` risk and `0.55` ESS cutoffs are **inference**, sharpened conservatively from the prior combined answers and from the code’s own ponder-priority geometry. They are chosen to **narrow** rollout admission, not broaden it. `wall` and `orasu` remain `[blocked]` until those signals are surfaced on the same runtime path. ([GitHub][1])

**Step 3 — If `hard == true`: rollout is completely bypassed.** Use **LearnerNet-only AFBS** for root priors/values and for any leaf evaluation that can affect root visits/Q. Search-derived labels remain allowed only under the existing safety valves: `min_visits >= 64`, `KL(pi_exit || pi_base) <= 2.0`, and `delta_q` exported only on actions with `visits(a) >= 4`. If those label guards fail, Hydra may still act using learner-anchored search, but it must export **no search-derived label** from that state. ([GitHub][6])

**Step 4 — If `hard == false` and rollout is globally admitted [future-only, inference]:** rollout may assist only below the decisive boundary. Before the final on-turn action is emitted, Hydra must recompute or verify the root with LearnerNet if **any** of the following holds:
(a) rollout top-1 root action differs from learner top-1 root action;
(b) root visit count is below 64;
(c) the label-trust score `lambda_exit` is `<= 0.5` (defined in section 4);
(d) the hard-state bit flips true after updated beliefs/search context;
(e) subtree provenance is unknown.
Any trigger forces a **LearnerNet-only root recomputation** and suppresses rollout-derived label export for that state. This whole step is inference, but it is the minimum policy that preserves the stated learner-anchor doctrine. ([GitHub][6])

**Step 5 — Cache fallback rule:** because `PonderResult` lacks source-network provenance and `PipelineState` exposes `learner_version` and `actor_version` but not a rollout-version surface, disabling rollout or changing rollout distillation state must trigger a **full ponder-cache flush**. Selective cache eviction is **`[blocked]`** until the cache stores at least `(source_net, source_version)` for each subtree/result. ([GitHub][3])

**Step 6 — Global disable triggers:** immediately disable rollout and revert to learner-only hard-state search + fast path elsewhere if **any** of the following trips:
`afbs_on_turn_ms >= 150`, `ct_smc_dp_ms >= 1`, `endgame_ms >= 100`, `self_play_games_per_sec <= 20`, `distill_kl_drift >= 0.1`, any hard-state fidelity gate in section 4 fails, duplicate online noninferiority fails, target coverage drops below 90%, or the aux/core gradient ratio exceeds 0.35 for sustained windows. The first five are grounded current gates; the latter rollout-specific safety triggers come from the prior combined answers and are the minimum needed to stop silent supervision drift. ([GitHub][7])

**Step 7 — Recovery protocol:** recovery must happen through **shadow requalification**, not live gradual trust. Re-enable rollout only after the full benchmark plan in section 5 passes for **three consecutive validation windows** **[inference]**. Until then, LearnerNet remains the only search-quality anchor in hard states. ([GitHub][2])

This fallback answers another red-team question directly: **yes, falling back to LearnerNet-only hard-state search is safer than a degraded rollout path**. It is safer doctrinally, safer mechanically, and safer with respect to supervision contamination. ([GitHub][2])

## 4. Gate metrics and thresholds

The current repo/doctrine already gives Hydra several **necessary** gates. They are not sufficient for rollout admission, but they are the baseline that must remain intact:

* `afbs_on_turn_ms < 150`
* `ct_smc_dp_ms < 1`
* `endgame_ms < 100`
* `self_play_games_per_sec > 20`
* `distill_kl_drift < 0.1`
* `min_visits >= 64` for ExIt-style root export
* `safety_valve_max_kl = 2.0` for the exit-policy safety valve
* `delta_q` exported only on actions with meaningful search support; the prior combined answer recommends `visits(a) >= 4`
* tranche-health guards: target coverage `>= 90%` and aux/core gradient ratio `<= 0.35` sustained. ([GitHub][7])

Those gates are **necessary but not sufficient** for rollout distillation. In particular, the current `distill_kl_drift < 0.1` gate is too weak to certify rollout safety. It is an average drift gate; it can pass while rollout still flips a small number of rare, decisive hard states. That answers the red-team question directly: **yes, the current distillation gate is too weak to catch search-quality collapse by itself.** ([GitHub][7])

Hydra therefore needs an additional rollout-admission gate stack. The minimum safe version is:

### 4.1 State-admission gate

```text
g_rollout = 1 - g_hard

g_hard =
    1[top2_policy_gap < 0.10
      or risk_score > 0.08
      or particle_ess < 0.55]
```

`top2_policy_gap < 0.10` is grounded in `HYDRA_FINAL.md`. The risk and ESS thresholds are **inference**, chosen conservatively to keep rollout narrow. `wall` and `orasu` are `[blocked]` additions until surfaced. ([GitHub][1])

### 4.2 Hard-state fidelity gate versus LearnerNet-only AFBS

For states with `g_hard = 1`, rollout is acceptable only if all of the following hold in shadow comparison against **LearnerNet-only AFBS**:

* **Root top-1 action agreement:** `>= 99.0%` **[inference]**
* **Root legal-action policy KL:** mean `<= 0.02`, p95 `<= 0.05` **[inference]**
* **`delta_q` sign agreement** on actions with `visits(a) >= 4`: `>= 97%` **[inference]**
* **No label export** if any of the above fails on the state. ([GitHub][3])

These thresholds are intentionally much tighter than the global `distill_kl_drift < 0.1` benchmark because hard-state AFBS errors are sparse and high-cost.

### 4.3 Non-hard-state fidelity gate

Even outside the learner-only zone, rollout should not be admitted unless it remains very close to the learner reference:

* **Root top-1 action agreement:** `>= 98.0%` **[inference]**
* **Root legal-action policy KL:** mean `<= 0.03`, p95 `<= 0.08` **[inference]**
* **`delta_q` sign agreement** on supported actions: `>= 95%` **[inference]**. ([GitHub][3])

### 4.4 Search-label trust gate

The prior combined answer supplies the right shape for a trust-weighted search-label export rule. The safest version is:

```text
lambda_exit =
    clip((N_root - 64) / (256 - 64), 0, 1)
  * clip((m_expanded - 0.85) / 0.10, 0, 1)
  * exp(-sigma_Q / 0.15)
  * clip(particle_ess / 0.60, 0, 1)
```

Export `exit_target` or `delta_q_target` only if `lambda_exit > 0.5`. This formula is **inference**, but it is a disciplined inference from the prior combined answer and the existing repo safety valves. It prevents noisy search from masquerading as ground truth. The grounded parts that must remain are `N_root >= 64`, the exit-policy KL safety valve, and per-action support masking for `delta_q`. ([GitHub][5])

### 4.5 Safety-head trust rule

`safety_residual` should **not** become a rollout-derived target. The prior combined answer recommends keeping it replay-derived and privileged rather than inventing a search teacher. That is the safer choice here as well, because rollout trust is exactly what is under dispute. ([GitHub][8])

### 4.6 Latency / throughput justification gate

Rollout is not justified unless it buys **material** wall-clock benefit. `HYDRA_FINAL.md` gives ActorNet inference at ~0.2 ms and LearnerNet at ~0.35 ms. That is a **1.75x** per-forward speedup, or about **42.9%** less forward time. In the best possible inference-bound case, a 150 ms turn would compress to about **85.7 ms**, saving **64.3 ms**. But to realize even a **25%** wall-clock reduction from that per-forward gain, pure network inference would need to consume about **58.3%** of the total turn time. So rollout should be admitted only if it either (a) is what makes the system pass the `<150 ms` AFBS gate at all, or (b) produces at least **25% admitted-state AFBS wall-clock reduction**. Otherwise the quality risk is not justified by the plausible speedup ceiling. ([GitHub][1])

This answers the last red-team question directly: **no, the throughput upside is not automatically large enough to justify extra quality risk**. The upside is real, but bounded; the safety bar must therefore be high. ([GitHub][1])

## 5. Minimum falsifiable benchmark plan

The benchmark that would prove rollout distillation acceptable must compare **rollout-assisted AFBS against LearnerNet-only AFBS**, not against no-search and not against ActorNet alone. The risk under discussion is “making search worse instead of cheaper,” so the reference must be the learner-anchored search path. Anything weaker fails the falsification requirement. ([GitHub][1])

### Stage A — Offline shadow fidelity on a fixed stratified corpus

Reuse the `HYDRA_FINAL.md` convention of a **200K stratified-state** validation set, but redefine the comparison: run **LearnerNet-only AFBS** and **rollout-assisted AFBS** with matched legal masks, top-k behavior, and search budgets on the same states. Also require fixed slice reporting on the suites emphasized in the prior combined answer: **hard defensive states, hand-building/offensive states, last-10-draw endgame states, and South-4 close-placement states**. This stage passes only if all section-4 fidelity gates hold, especially on the hard-state slice. ([GitHub][1])

A concrete minimum hard-state requirement is warranted: the corpus should contain **at least 20K hard states** under the section-4 detector **[inference]**; if the stratified 200K set yields fewer, augment it until that floor is met. Without a substantial hard-state slice, rollout acceptability is not falsifiable on the states that matter most. ([GitHub][1])

### Stage B — Fallback / provenance fault testing

Run explicit fault tests for the scenarios that would otherwise create silent trust drift:

* rollout disabled mid-run;
* rollout distillation version changed;
* cached ponder result reused after a disable;
* state transitions from non-hard to hard after updated beliefs/context.

Under the current surfaced structures, the only safe expected behavior is **global cache flush** plus learner-only recomputation. Selective cache preservation is **`[blocked]`** until provenance fields exist. If any unknown-provenance subtree survives a disable event and is reused on-turn, rollout is not safe. ([GitHub][3])

### Stage C — Label-safety tranche validation

Search-derived label export must be validated as its own tranche, not assumed from runtime agreement. Require:

* target coverage `>= 90%`,
* nonzero auxiliary contribution at the expected rate,
* no NaN / Inf / silent-all-None paths,
* aux/core gradient ratio `<= 0.35`,
* no promotion if offline search agreement improves but duplicate online play does not. ([GitHub][5])

This stage matters because `bridge.rs` feeds AFBS-derived quantities back into model features/targets; a rollout path can poison future learning even if its immediate move error rate looks small. ([GitHub][4])

### Stage D — Duplicate online noninferiority against LearnerNet-only control

Run duplicate, paired, seat-rotated evaluation of the rollout-assisted candidate against the **LearnerNet-only mainline** using the online metrics already surfaced in `eval.rs`: mean placement, top-2 rate, 4th rate, win rate, and deal-in rate. Use enough games to achieve the stated confidence bounds, with a minimum floor of **20K paired hanchan** **[inference]** if a fixed floor is needed. ([GitHub][7])

A conservative minimum noninferiority rule is:

* **mean placement delta** (rollout-assisted minus learner-only): 95% CI upper bound `<= +0.01` **[inference]**
* **4th-rate delta:** 95% CI upper bound `<= +0.25 pp` **[inference]**
* **top-2-rate delta:** 95% CI lower bound `>= -0.25 pp` **[inference]**
* **deal-in-rate delta:** 95% CI upper bound `<= +0.25 pp` **[inference]**
* **no catastrophic regression** on the hard-state and endgame slice suites. ([GitHub][7])

### Stage E — Throughput proof

Promotion additionally requires one of two things:

1. rollout is the difference between **failing** and **passing** the repo’s AFBS-on-turn `<150 ms` gate; or
2. rollout yields at least **25% admitted-state AFBS wall-clock reduction** while preserving the noninferiority results above. ([GitHub][7])

### Exact minimum falsifier

Rollout distillation is **not acceptable** if **any** of the following happens:

* hard-state root top-1 agreement with LearnerNet-only AFBS falls below **99.0%**;
* hard-state root legal-action KL exceeds the section-4 limits;
* duplicate online noninferiority fails on placement / 4th / top-2 / deal-in;
* search-label tranche health fails (`coverage < 90%`, `aux/core > 0.35`, silent label-path breakage);
* rollout does not deliver material latency benefit. ([GitHub][5])

That is the exact minimum benchmark bar that would prove rollout distillation acceptable. Anything weaker does **not** bound drift well enough.

## 6. Dependency closure table

| Dependency / question      | Grounded current surface                                                                                                                                                                                | Missing / blocked                                                                                          | Operational consequence                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Role split                 | LearnerNet = 24-block training + deep AFBS on hard states; ActorNet = 12-block self-play/shallow SaF; RolloutNet appears in doctrine as actor-sized fast AFBS support. ([GitHub][1])                    | No first-class rollout constructor/version surfaced in inspected runtime state. ([GitHub][6])              | Rollout cannot be treated as a co-equal runtime authority today.                                   |
| Hard-state detector inputs | `GameStateSnapshot` exposes `top2_policy_gap`, `risk_score`, `particle_ess`; ponder priority already upweights low-gap/high-risk/low-ESS states. ([GitHub][3])                                          | `wall`, `orasu`, and richer score-context triggers are not surfaced on that same path. ([GitHub][5])       | Current automatic rollout gate can use gap/risk/ESS now; endgame/context refiners are `[blocked]`. |
| Decisive AFBS path         | PUCT uses prior + q; expansion renormalizes network priors; backprop updates visit/value; root exit policy and best action are derived from q/visits. ([GitHub][3])                                     | None conceptually; this part is already decisive.                                                          | Any rollout net providing decisive priors/values becomes the true search-quality anchor.           |
| Search-to-model bridge     | `bridge.rs` injects AFBS-derived `delta_q` search features; model exposes `delta_q` and `safety_residual` heads. ([GitHub][4])                                                                          | No per-feature provenance/trust mask shown in the inspected bridge/model surfaces.                         | Rollout mistakes can contaminate both current decisions and future supervision.                    |
| Global benchmark surface   | `BenchmarkGates` tracks AFBS latency, CT-SMC latency, endgame latency, self-play throughput, and distill KL drift. `TrainingMetrics` tracks policy agreement, value MSE, distill KL, Elo. ([GitHub][7]) | No hard-state-specific root-fidelity gate in the surfaced benchmark structs.                               | Current gates are necessary but insufficient for rollout admission.                                |
| ExIt / label safety valves | `ExitConfig` defaults include `min_visits = 64` and `safety_valve_max_kl = 2.0`; prior combined answers add `visits(a) >= 4`, coverage ≥90%, aux/core ≤0.35. ([GitHub][6])                              | No rollout-specific label trust weight is codified in inspected code.                                      | A trust-weighted label export gate must be added before rollout-derived labels are safe.           |
| Cache / provenance         | `PonderResult` stores exit policy, value, depth, visits, timestamp. ([GitHub][3])                                                                                                                       | No `source_net` / `source_version`; `PipelineState` versions learner and actor, not rollout. ([GitHub][6]) | Selective rollback/eviction is `[blocked]`; safe disable requires full cache flush.                |
| Fast-path fallback         | `SearchContext` is optional and the encoder has a default empty search context; `HYDRA_FINAL.md` already defines fast-path vs slow-path behavior. ([GitHub][4])                                         | None for basic fallback.                                                                                   | Hydra already has a clean learner-only / no-search fallback route.                                 |
| Online evaluation outputs  | `eval.rs` surfaces mean placement, top-2 rate, 4th rate, win rate, deal-in rate. ([GitHub][7])                                                                                                          | No explicit rollout-vs-learner duplicate benchmark harness shown in inspected files.                       | Acceptability must be proven by an added paired noninferiority protocol, not assumed.              |

## 7. Final recommendation: central, narrow, optional, or disable-until-better

**Final recommendation: disable-until-better.** ([GitHub][2])

It should **not** be central. That would directly violate the reconciliation doctrine that broad AFBS rollout is not the active path and that deep AFBS belongs on hard positions only. It should **not** be merely optional, because “optional” without provenance and hard-state fidelity gates still allows rollout to quietly become the decisive anchor through AFBS roots, cached ponder subtrees, and search-derived labels. And it is **not even safely narrow today**, because the inspected runtime surfaces do not yet close the provenance/version/fidelity loop needed for a trustworthy narrow rollout path. ([GitHub][2])

The red-team answers are therefore:

* **Does the rollout net silently become the real quality anchor for decisive search states?** Yes, if it supplies priors/values that determine AFBS root q/visits or exported search labels. ([GitHub][3])
* **Is the current distillation gate too weak to catch quality collapse?** Yes. `distill_kl_drift < 0.1` is necessary but insufficient; it does not certify hard-state root fidelity. ([GitHub][7])
* **Would falling back to LearnerNet-only hard-state search be safer than a degraded rollout path?** Yes, and it is already aligned with both doctrine and runtime surfaces. ([GitHub][2])
* **Are throughput gains large enough to justify the extra quality risk?** Not by default. The theoretical per-forward gain is bounded, and rollout only earns trust if it materially improves wall-clock performance **without** measurable hard-state search loss. ([GitHub][1])

The correct future target is therefore: **disable-until-better now; if provenance, learner-anchor enforcement, and the benchmark plan all pass, promote only to narrow, never central.** In that future narrow mode, rollout may assist only in non-hard, non-decisive acceleration, while LearnerNet remains the sole authority for decisive hard-state search and for any exported search supervision. Until that closure exists, keeping rollout out of live decisive AFBS is the safer and more Hydra-consistent choice. ([GitHub][2])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md"
[6]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/config.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/config.rs"
[7]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/eval.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/eval.rs"
[8]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
