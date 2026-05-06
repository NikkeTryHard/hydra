<combined_run_record run_id="answer_16" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 16 + returned agent answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_16_VALIDATE_ROLLOUT_DISTILLATION_GATES_AND_FALLBACKS.md">
<![CDATA[# Hydra prompt — validate rollout-distillation quality gates, trust boundaries, and fallback protocol

Primary source material in raw GitHub links below.

## Critical directive

Narrow operational prompt for one of Hydra's biggest risks: rollout net makes search worse, not cheaper.

Broad exploration already done in `research/agent_handoffs/combined_all_variants/`. Do not redo broad work. Start from prior combined answers + current doctrine. Use new retrieval only to validate, falsify, or sharpen rollout-distillation gates, trust boundaries, + fallback behavior.

Do not treat this as broad architecture or breakthrough prompt.

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

Relevant prior answers + refs:
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md

Validate Hydra rollout-distillation operational doctrine, specifically:
- when rollout net may participate in AFBS
- when LearnerNet must remain search-quality anchor
- what quantitative gates decide rollout distillation acceptability
- what exact fallback protocol Hydra should use if rollout drift too high

<output_contract>
- Return exactly requested sections, in requested order.
- Be as detailed + explicit as needed; do not optimize for brevity.
- Return full technical treatment, not compressed memo.
- Short answer usually = failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit trust gates, fallback conditions, latency/quality tradeoffs, or benchmark thresholds when they matter.
</verbosity_controls>

<tool_persistence_rules>
- Do not restart broad Hydra future-planning.
- New retrieval should only validate, falsify, or sharpen rollout-distillation trust boundaries + fallback policy.
- Use Python in bash for latency/quality arithmetic, drift thresholds, + break-even checks when helpful.
</tool_persistence_rules>

<dependency_checks>
- Verify intended roles of ActorNet, LearnerNet, + RolloutNet from current docs.
- Verify which search modes depend on rollout quality + which should stay learner-anchored.
- Verify where current repo/runtime surfaces could carry rollout-quality metrics or fallback decisions.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in provided docs/code.
- Mark any unevidenced runtime switch, gate metric, or fallback protocol as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Does rollout net silently become real quality anchor for decisive search states?
  - Is distillation gate too weak to catch quality collapse?
  - Would fallback to LearnerNet-only hard-state search be safer than degraded rollout path?
  - Are throughput gains large enough to justify extra quality risk?
</self_red_team_rules>

<minimum_falsification_rules>
- Define exact minimum benchmark that would prove rollout distillation acceptable.
- If drift or search-quality loss cannot be bounded by concrete gate, recommend disabling or narrowing rollout-net usage.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into generic distillation paper survey.
- Stay inside Hydra rollout-net trust policy, gates, + fallback behavior.
</anti_survey_rules>

## What to do
1. Reconstruct intended ActorNet / LearnerNet / RolloutNet split.
2. Define exact trust boundary for rollout-net usage.
3. Define exact fallback protocol when rollout quality not good enough.
4. Give gating metrics, thresholds, + benchmark plan that make this safe.
5. Decide whether rollout-net usage should stay optional, narrow, or central.

## Deliverables
1. Hydra posture reconstruction for actor / learner / rollout roles
2. Search-quality trust boundary
3. Exact fallback protocol
4. Gate metrics and thresholds
5. Minimum falsifiable benchmark plan
6. Dependency closure table
7. Final rec: central, narrow, optional, or disable-until-better

## Hard constraints
- no generic “distillation is usually fine” answers
- no broad architecture resets
- no pretending rollout quality can be inferred without explicit gates
- no rec that lets rollout net quietly become decisive search-quality anchor
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_16.md">
<![CDATA[## 1. Hydra posture reconstruction for actor / learner / rollout roles

`HYDRA_FINAL.md` defines current intended split as two-tier system. **LearnerNet** = 24-block, ~10M model for training + **deep AFBS on hard positions**. **ActorNet** = 12-block, ~5M model for self-play generation + shallow SaF features, with continuous Learner→Actor distillation every 1–2 minutes. Current training code matches that two-model shape: `HydraModelConfig::learner()` = 24 blocks, `HydraModelConfig::actor()` = 12 blocks, `PipelineState` versions learner + actor separately. (research/design/HYDRA_FINAL.md:24-25,95-107,324-324; hydra-train/src/config.rs:261-324)

`HYDRA_FINAL.md` then adds **Phase-2 RolloutNet**: ActorNet-sized, continuously distilled from LearnerNet, same input encoding, intended for **fast AFBS rollouts**. That is doctrine. But in inspected code-grounding files, surfaced runtime objects still center on learner/actor, not learner/actor/rollout: model + config surfaces expose actor + learner constructors/versioning, and benchmark surface tracks global distillation drift, not rollout-specific runtime identity or provenance path. So today RolloutNet exists as **design intent**, not first-class operationally closed surface in inspected repo. (research/design/HYDRA_FINAL.md:320-324; hydra-train/src/config.rs:261-324; hydra-train/src/eval.rs:85-137)

`HYDRA_RECONCILIATION.md` narrows doctrine further. Broad “search everywhere” AFBS rollout explicitly removed from active path; AFBS should be **specialist + hard-state gated**, not broad runtime identity of Hydra. `HYDRA_FINAL.md` matches that narrowing: **fast path** = network forward + SaF adaptor, **slow path** = reuse of pondered AFBS subtree, deep AFBS budget belongs on **hard positions only**. That makes intended hierarchy clear: **LearnerNet = search-quality anchor**, **ActorNet = cheap production/self-play workhorse**, **RolloutNet, if introduced, = acceleration layer subordinate to LearnerNet, not co-equal decision authority**. Last clause is inference, but only reading consistent with reconciliation doctrine + current repo surfaces. (research/design/HYDRA_RECONCILIATION.md:39-55,117-160,187-205,285-298; research/design/HYDRA_FINAL.md:13-25,251-264,320-324)

Crucial operational point follows: **ActorNet + RolloutNet must not be conflated because both are 12-block-sized**. ActorNet documented role = self-play generation + shallow runtime features. RolloutNet documented role = fast AFBS support. Inspected code gives Hydra learner/actor split, not learner/actor/rollout split. So using “cheap 12-block net” as un-gated AFBS authority would not be neutral impl detail; it would collapse two different trust roles into one path without dedicated gate or provenance boundary. That is exact silent trust drift user warns about. (research/design/HYDRA_FINAL.md:24-25,102-107,320-324; hydra-train/src/config.rs:261-324)

## 2. Search-quality trust boundary

In Hydra AFBS code, network supplying priors + values is not harmless helper; it directly shapes search decision. `puct_select` ranks children using **prior** + **q_value**, expansion keeps top-k slice of network policy mass + renormalizes it, backprop accumulates value into visit stats, `root_exit_policy` is computed from child q-values, and `best_action` is chosen by visit count. So if rollout net may supply decisive priors or values in important states, it does not merely make search cheaper; it becomes **real quality anchor** for that search outcome. That answers one red-team question directly: **yes, rollout silently becomes real anchor if allowed to drive decisive AFBS states**. (hydra-core/src/afbs.rs:167-186,188-219,238-263,265-314)

Trust problem broader than root move selection. `bridge.rs` converts AFBS root Q diffs into `delta_q` search features, and `model.rs` already exposes `delta_q` + `safety_residual` heads. That means weak rollout path can damage Hydra two ways at once: distort **current** decision through AFBS, and distort **future supervision/features** if search-derived signals export from that path. `TESTING.md` is explicit about failure mode: bad labels silently train model that becomes “confidently wrong.” So rollout quality is not only runtime issue; it is training-data trust issue. (hydra-core/src/bridge.rs:301-355; hydra-train/src/model.rs:18-23,94-99,263-270; research/design/TESTING.md:7-11)

Hydra doctrine + runtime already identify right boundary vars. `HYDRA_FINAL.md` says hard positions are characterized by **small top-2 policy gap**, **high-risk defense**, or **low particle ESS**. Runtime surfaces exactly those signals in `GameStateSnapshot`: `top2_policy_gap`, `risk_score`, `particle_ess`. Code's own `compute_ponder_priority` already prioritizes states when gap low, risk high, ESS weak. So Hydra already encodes geometry of “states where cheap approximation is most dangerous.” Those are exactly states where **LearnerNet must remain anchor**. (research/design/HYDRA_FINAL.md:263-264,324-324; hydra-core/src/afbs.rs:472-507,634-635)

That gives exact trust boundary:

1. **LearnerNet-only zone (must remain learner-anchored):**
all hard states; all on-turn decisive AFBS roots; all search-derived label export (`exit_policy`, `delta_q`); and any cached subtree whose provenance unknown. Grounded in doctrine that AFBS is specialist/hard-state gated and in AFBS code path that turns priors/q-values into root action. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298; hydra-core/src/afbs.rs:167-186,265-314)

2. **Rollout-forbidden zone today:**
anything that can directly decide root action, root exit policy, root visit allocation, or exported search label on those states. This is inference from AFBS mechanics, but necessary inference, not speculative one. If rollout allowed there, LearnerNet no longer practical anchor. (hydra-core/src/afbs.rs:167-186,188-219,265-314)

3. **Rollout-permissible zone later, if ever [inference]:**
non-hard, non-decisive acceleration only—such as off-turn pondering or inner-loop rollout assistance—provided **final on-turn root is revalidated by LearnerNet before action emission or label export**. This is narrowest interpretation that preserves Hydra stated doctrine. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

Boundary is tighter in current repo than it would be in fully instrumented rollout system, because cache/provenance layer incomplete. `SearchContext` is optional and encoder has default empty context, so Hydra already has clean **learner-only / no-search** fallback surface. But `PonderResult` records exit policy, value, depth, visits, + timestamp—not source network. So if rollout-generated subtrees existed, current surfaced structure would not let Hydra distinguish them from learner-generated subtrees at reuse time. That means cached AFBS results touched by rollout are **operationally untrusted** until provenance is added. (hydra-core/src/bridge.rs:27-45; hydra-core/src/afbs.rs:398-459)

Prior combined answers add plausible refinements—`wall <= 10`, `orasu`, + tighter endgame/context gates—but those refinements are not surfaced in inspected `GameStateSnapshot`. So those context enrichments are reasonable, but **`[blocked]` for automatic runtime gating** until corresponding signals are explicitly plumbed into same decision surface that holds gap/risk/ESS. (research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:344-354,493-503,534-541,696-697)

Which search modes depend on rollout quality is therefore straightforward. **AFBS root/leaf evaluation, pondered subtree reuse, + any exported `delta_q`/exit labels** are rollout-sensitive if rollout is in loop. **Fast path without SearchContext, CT-SMC DP, + endgame solver gates** are not rollout-sensitive in inspected surfaces. So correct trust rule is not “rollout is fine if average KL is small”; it is “rollout is only admissible where its errors cannot quietly become decisive action or supervision errors.” (hydra-core/src/bridge.rs:27-45,301-355; hydra-core/src/ct_smc.rs:1-5,10-18,228-258; hydra-core/src/endgame.rs:1-18,72-87)

## 3. Exact fallback protocol

Safest exact fallback protocol is **not** “keep using rollout but with reduced trust.” It is: **prefer LearnerNet-only hard-state search; otherwise fall back to existing fast path.** That matches both doctrine + runtime surfaces. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298; hydra-train/src/inference.rs:151-180)

**Step 1 — Default live posture now:** set rollout participation in live AFBS to **disabled**. Rollout path may exist only in shadow evaluation **[inference]**, because inspected runtime surfaces do not expose dedicated rollout version/provenance path, and current cache surface cannot distinguish rollout-generated subtrees from learner-generated ones. (research/design/HYDRA_FINAL.md:320-324; hydra-core/src/afbs.rs:398-459; hydra-train/src/config.rs:261-324)

**Step 2 — Per-state hard-state classifier:** before any rollout participation, compute hard-state bit from surfaced runtime signals:

```text
hard =
    (top2_policy_gap < 0.10)
 or (risk_score > 0.08)         [inference]
 or (particle_ess < 0.55)       [inference]
```

`top2_policy_gap < 0.10` directly aligns with `HYDRA_FINAL.md`. `0.08` risk + `0.55` ESS cutoffs are **inference**, sharpened conservatively from prior combined answers + code's own ponder-priority geometry. Chosen to **narrow** rollout admission, not broaden it. `wall` + `orasu` remain `[blocked]` until those signals are surfaced on same runtime path. (research/design/HYDRA_FINAL.md:263-264,324-324; hydra-core/src/afbs.rs:472-507; research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:344-354,493-503,534-541)

**Step 3 — If `hard == true`: rollout completely bypassed.** Use **LearnerNet-only AFBS** for root priors/values and for any leaf evaluation that can affect root visits/Q. Search-derived labels remain allowed only under existing safety valves: `min_visits >= 64`, `KL(pi_exit || pi_base) <= 2.0`, + `delta_q` exported only on actions with `visits(a) >= 4`. If those label guards fail, Hydra may still act using learner-anchored search, but it must export **no search-derived label** from that state. (hydra-train/src/config.rs:583-602; research/agent_handoffs/combined_all_variants/answer_2-1_combined.md:209-243)

**Step 4 — If `hard == false` and rollout globally admitted [future-only, inference]:** rollout may assist only below decisive boundary. Before final on-turn action is emitted, Hydra must recompute or verify root with LearnerNet if **any** of following holds:
rollout top-1 root action differs from learner top-1 root action;
(b) root visit count below 64;
(c) label-trust score `lambda_exit` is `<= 0.5` (defined in section 4);
(d) hard-state bit flips true after updated beliefs/search context;
(e) subtree provenance unknown.
Any trigger forces **LearnerNet-only root recomputation** and suppresses rollout-derived label export for that state. Whole step is inference, but it is minimum policy that preserves stated learner-anchor doctrine. (hydra-train/src/config.rs:583-602; research/agent_handoffs/combined_all_variants/answer_2-1_combined.md:209-243)

**Step 5 — Cache fallback rule:** because `PonderResult` lacks source-network provenance and `PipelineState` exposes `learner_version` + `actor_version` but not rollout-version surface, disabling rollout or changing rollout distillation state must trigger **full ponder-cache flush**. Selective cache eviction is **`[blocked]`** until cache stores at least `(source_net, source_version)` for each subtree/result. (hydra-core/src/afbs.rs:398-459; hydra-train/src/config.rs:261-324)

**Step 6 — Global disable triggers:** immediately disable rollout + revert to learner-only hard-state search + fast path elsewhere if **any** of following trips:
`afbs_on_turn_ms >= 150`, `ct_smc_dp_ms >= 1`, `endgame_ms >= 100`, `self_play_games_per_sec <= 20`, `distill_kl_drift >= 0.1`, any hard-state fidelity gate in section 4 fails, duplicate online noninferiority fails, target coverage drops below 90%, or aux/core gradient ratio exceeds 0.35 for sustained windows. First five are grounded current gates; latter rollout-specific safety triggers come from prior combined answers + are minimum needed to stop silent supervision drift. (hydra-train/src/eval.rs:31-37,111-137,178-204; research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:656-681,696-697)

**Step 7 — Recovery protocol:** recovery must happen through **shadow requalification**, not live gradual trust. Re-enable rollout only after full benchmark plan in section 5 passes for **three consecutive validation windows** **[inference]**. Until then, LearnerNet remains only search-quality anchor in hard states. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

This fallback answers another red-team question directly: **yes, fallback to LearnerNet-only hard-state search is safer than degraded rollout path**. Safer doctrinally, mechanically, + for supervision contamination. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

## 4. Gate metrics and thresholds

Current repo/doctrine already gives Hydra several **necessary** gates. Not sufficient for rollout admission, but baseline that must remain intact:

* `afbs_on_turn_ms < 150`
* `ct_smc_dp_ms < 1`
* `endgame_ms < 100`
* `self_play_games_per_sec > 20`
* `distill_kl_drift < 0.1`
* `min_visits >= 64` for ExIt-style root export
* `safety_valve_max_kl = 2.0` for exit-policy safety valve
* `delta_q` exported only on actions with meaningful search support; prior combined answer recommends `visits(a) >= 4`
* tranche-health guards: target coverage `>= 90%` and aux/core gradient ratio `<= 0.35` sustained. (hydra-train/src/eval.rs:111-137; research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:656-681,696-697)

Those gates are **necessary but not sufficient** for rollout distillation. In particular, current `distill_kl_drift < 0.1` gate is too weak to certify rollout safety. It is average drift gate; it can pass while rollout still flips small number of rare, decisive hard states. That answers red-team question directly: **yes, current distillation gate is too weak to catch search-quality collapse by itself.** (hydra-train/src/eval.rs:111-137)

Hydra therefore needs additional rollout-admission gate stack. Minimum safe version:

### 4.1 State-admission gate

```text
g_rollout = 1 - g_hard

g_hard =
    1[top2_policy_gap < 0.10
      or risk_score > 0.08
      or particle_ess < 0.55]
```

`top2_policy_gap < 0.10` grounded in `HYDRA_FINAL.md`. Risk + ESS thresholds are **inference**, chosen conservatively to keep rollout narrow. `wall` + `orasu` are `[blocked]` additions until surfaced. (research/design/HYDRA_FINAL.md:263-264,324-324; hydra-core/src/afbs.rs:472-507; research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:344-354,493-503,534-541)

### 4.2 Hard-state fidelity gate versus LearnerNet-only AFBS

For states with `g_hard = 1`, rollout acceptable only if all of following hold in shadow comparison against **LearnerNet-only AFBS**:

* **Root top-1 action agreement:** `>= 99.0%` **[inference]**
* **Root legal-action policy KL:** mean `<= 0.02`, p95 `<= 0.05` **[inference]**
* **`delta_q` sign agreement** on actions with `visits(a) >= 4`: `>= 97%` **[inference]**
* **No label export** if any of above fails on state. (hydra-core/src/afbs.rs:265-314)

These thresholds are intentionally much tighter than global `distill_kl_drift < 0.1` benchmark because hard-state AFBS errors are sparse + high-cost.

### 4.3 Non-hard-state fidelity gate

Even outside learner-only zone, rollout should not be admitted unless it remains close to learner reference:

* **Root top-1 action agreement:** `>= 98.0%` **[inference]**
* **Root legal-action policy KL:** mean `<= 0.03`, p95 `<= 0.08` **[inference]**
* **`delta_q` sign agreement** on supported actions: `>= 95%` **[inference]**. (hydra-core/src/afbs.rs:265-314)

### 4.4 Search-label trust gate

Prior combined answer supplies right shape for trust-weighted search-label export rule. Safest version:

```text
lambda_exit =
    clip((N_root - 64) / (256 - 64), 0, 1)
  * clip((m_expanded - 0.85) / 0.10, 0, 1)
  * exp(-sigma_Q / 0.15)
  * clip(particle_ess / 0.60, 0, 1)
```

Export `exit_target` or `delta_q_target` only if `lambda_exit > 0.5`. This formula is **inference**, but disciplined inference from prior combined answer + existing repo safety valves. It prevents noisy search from masquerading as ground truth. Grounded parts that must remain are `N_root >= 64`, exit-policy KL safety valve, + per-action support masking for `delta_q`. (research/agent_handoffs/combined_all_variants/answer_2-1_combined.md:209-243)

### 4.5 Safety-head trust rule

`safety_residual` should **not** become rollout-derived target. Prior combined answer recommends keeping it replay-derived + privileged rather than inventing search teacher. That is safer choice here too, because rollout trust is exact disputed point. (research/agent_handoffs/combined_all_variants/answer_2-1_combined.md:229-237)

### 4.6 Latency / throughput justification gate

Rollout not justified unless it buys **material** wall-clock benefit. `HYDRA_FINAL.md` gives ActorNet inference at ~0.2 ms and LearnerNet at ~0.35 ms. That is **1.75x** per-forward speedup, or about **42.9%** less forward time. In best possible inference-bound case, 150 ms turn would compress to about **85.7 ms**, saving **64.3 ms**. But to realize even **25%** wall-clock reduction from that per-forward gain, pure network inference would need to consume about **58.3%** of total turn time. So rollout should be admitted only if it either is what makes system pass `<150 ms` AFBS gate at all, or (b) produces at least **25% admitted-state AFBS wall-clock reduction**. Otherwise quality risk not justified by plausible speedup ceiling. (research/design/HYDRA_FINAL.md:102-107; hydra-train/src/eval.rs:111-137)

This answers last red-team question directly: **no, throughput upside is not automatically large enough to justify extra quality risk**. Upside is real, but bounded; safety bar must therefore be high. (research/design/HYDRA_FINAL.md:102-107; hydra-train/src/eval.rs:111-137)

## 5. Minimum falsifiable benchmark plan

Benchmark that would prove rollout distillation acceptable must compare **rollout-assisted AFBS against LearnerNet-only AFBS**, not against no-search and not against ActorNet alone. Risk under discussion is “making search worse instead of cheaper,” so reference must be learner-anchored search path. Anything weaker fails falsification requirement. (research/design/HYDRA_FINAL.md:24-25,102-107,324-324)

### Stage A — Offline shadow fidelity on a fixed stratified corpus

Reuse `HYDRA_FINAL.md` convention of **200K stratified-state** validation set, but redefine comparison: run **LearnerNet-only AFBS** + **rollout-assisted AFBS** with matched legal masks, top-k behavior, + search budgets on same states. Also require fixed slice reporting on suites emphasized in prior combined answer: **hard defensive states, hand-building/offensive states, last-10-draw endgame states, and South-4 close-placement states**. This stage passes only if all section-4 fidelity gates hold, especially on hard-state slice. (research/design/HYDRA_FINAL.md:216-216,346-346)

Concrete minimum hard-state requirement warranted: corpus should contain **at least 20K hard states** under section-4 detector **[inference]**; if stratified 200K set yields fewer, augment it until that floor is met. Without substantial hard-state slice, rollout acceptability is not falsifiable on states that matter most. (research/design/HYDRA_FINAL.md:346-346)

### Stage B — Fallback / provenance fault testing

Run explicit fault tests for scenarios that would otherwise create silent trust drift:

* rollout disabled mid-run;
* rollout distillation version changed;
* cached ponder result reused after disable;
* state transitions from non-hard to hard after updated beliefs/context.

Under current surfaced structures, only safe expected behavior is **global cache flush** + learner-only recomputation. Selective cache preservation is **`[blocked]`** until provenance fields exist. If any unknown-provenance subtree survives disable event + is reused on-turn, rollout is not safe. (hydra-core/src/afbs.rs:398-459,510-527)

### Stage C — Label-safety tranche validation

Search-derived label export must be validated as its own tranche, not assumed from runtime agreement. Require:

* target coverage `>= 90%`,
* nonzero auxiliary contribution at expected rate,
* no NaN / Inf / silent-all-None paths,
* aux/core gradient ratio `<= 0.35`,
* no promotion if offline search agreement improves but duplicate online play does not. (research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:656-681,696-697)

This stage matters because `bridge.rs` feeds AFBS-derived quantities back into model features/targets; rollout path can poison future learning even if immediate move error rate looks small. (hydra-core/src/bridge.rs:301-355)

### Stage D — Duplicate online noninferiority against LearnerNet-only control

Run duplicate, paired, seat-rotated evaluation of rollout-assisted candidate against **LearnerNet-only mainline** using online metrics already surfaced in `eval.rs`: mean placement, top-2 rate, 4th rate, win rate, + deal-in rate. Use enough games to achieve stated confidence bounds, with minimum floor of **20K paired hanchan** **[inference]** if fixed floor needed. (hydra-train/src/eval.rs:31-37,164-204)

Conservative minimum noninferiority rule:

* **mean placement delta** (rollout-assisted minus learner-only): 95% CI upper bound `<= +0.01` **[inference]**
* **4th-rate delta:** 95% CI upper bound `<= +0.25 pp` **[inference]**
* **top-2-rate delta:** 95% CI lower bound `>= -0.25 pp` **[inference]**
* **deal-in-rate delta:** 95% CI upper bound `<= +0.25 pp` **[inference]**
* **no catastrophic regression** on hard-state + endgame slice suites. (hydra-train/src/eval.rs:164-204)

### Stage E — Throughput proof

Promotion additionally requires one of two things:

1. rollout is difference between **failing** and **passing** repo AFBS-on-turn `<150 ms` gate; or
2. rollout yields at least **25% admitted-state AFBS wall-clock reduction** while preserving noninferiority results above. (hydra-train/src/eval.rs:111-137)

### Exact minimum falsifier

Rollout distillation is **not acceptable** if **any** of following happens:

* hard-state root top-1 agreement with LearnerNet-only AFBS falls below **99.0%**;
* hard-state root legal-action KL exceeds section-4 limits;
* duplicate online noninferiority fails on placement / 4th / top-2 / deal-in;
* search-label tranche health fails (`coverage < 90%`, `aux/core > 0.35`, silent label-path breakage);
* rollout does not deliver material latency benefit. (research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:344-354,656-681)

That is exact minimum benchmark bar that would prove rollout distillation acceptable. Anything weaker does **not** bound drift well enough.

## 6. Dependency closure table

| Dependency / question      | Grounded current surface                                                                                                                                                                                | Missing / blocked                                                                                          | Operational consequence                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Role split                 | LearnerNet = 24-block training + deep AFBS on hard states; ActorNet = 12-block self-play/shallow SaF; RolloutNet appears in doctrine as actor-sized fast AFBS support. (research/design/HYDRA_FINAL.md:24-25,102-107,320-324)                    | No first-class rollout constructor/version surfaced in inspected runtime state. (hydra-train/src/config.rs:261-324)              | Rollout cannot be treated as co-equal runtime authority today.                                   |
| Hard-state detector inputs | `GameStateSnapshot` exposes `top2_policy_gap`, `risk_score`, `particle_ess`; ponder priority already upweights low-gap/high-risk/low-ESS states. (hydra-core/src/afbs.rs:472-507) | `wall`, `orasu`, + richer score-context triggers are not surfaced on same path. (research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:344-354,493-503,534-541) | Current automatic rollout gate can use gap/risk/ESS now; endgame/context refiners are `[blocked]`. |
| Decisive AFBS path         | PUCT uses prior + q; expansion renormalizes network priors; backprop updates visit/value; root exit policy + best action derive from q/visits. (hydra-core/src/afbs.rs:167-186,188-219,238-314)                                     | None conceptually; this part already decisive.                                                          | Any rollout net providing decisive priors/values becomes true search-quality anchor.           |
| Search-to-model bridge | `bridge.rs` injects AFBS-derived `delta_q` search features; model exposes `delta_q` + `safety_residual` heads. (hydra-core/src/bridge.rs:301-355; hydra-train/src/model.rs:18-23,94-99,263-270) | No per-feature provenance/trust mask shown in inspected bridge/model surfaces. | Rollout mistakes can contaminate both current decisions + future supervision. |
| Global benchmark surface | `BenchmarkGates` tracks AFBS latency, CT-SMC latency, endgame latency, self-play throughput, + distill KL drift. `TrainingMetrics` tracks policy agreement, value MSE, distill KL, Elo. (hydra-train/src/eval.rs:85-137) | No hard-state-specific root-fidelity gate in surfaced benchmark structs. | Current gates are necessary but insufficient for rollout admission. |
| ExIt / label safety valves | `ExitConfig` defaults include `min_visits = 64` + `safety_valve_max_kl = 2.0`; prior combined answers add `visits(a) >= 4`, coverage ≥90%, aux/core ≤0.35. (hydra-train/src/config.rs:583-602; research/agent_handoffs/combined_all_variants/answer_2-1_combined.md:221-243; research/agent_handoffs/combined_all_variants/answer_3-1_combined.md:656-681,696-697) | No rollout-specific label trust weight codified in inspected code. | Trust-weighted label export gate must be added before rollout-derived labels are safe. |
| Cache / provenance | `PonderResult` stores exit policy, value, depth, visits, timestamp. (hydra-core/src/afbs.rs:398-459) | No `source_net` / `source_version`; `PipelineState` versions learner + actor, not rollout. (hydra-train/src/config.rs:261-324) | Selective rollback/eviction is `[blocked]`; safe disable requires full cache flush. |
| Fast-path fallback | `SearchContext` is optional and encoder has default empty search context; `HYDRA_FINAL.md` already defines fast-path vs slow-path behavior. (hydra-core/src/bridge.rs:27-45; research/design/HYDRA_FINAL.md:13-15,251-264) | None for basic fallback. | Hydra already has clean learner-only / no-search fallback route. |
| Online evaluation outputs | `eval.rs` surfaces mean placement, top-2 rate, 4th rate, win rate, deal-in rate. (hydra-train/src/eval.rs:31-37,164-204) | No explicit rollout-vs-learner duplicate benchmark harness shown in inspected files. | Acceptability must be proven by added paired noninferiority protocol, not assumed. |

## 7. Final recommendation: central, narrow, optional, or disable-until-better

**Final rec: disable-until-better.** (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

It should **not** be central. That would directly violate reconciliation doctrine that broad AFBS rollout is not active path and that deep AFBS belongs on hard positions only. It should **not** be merely optional, because “optional” without provenance + hard-state fidelity gates still allows rollout to quietly become decisive anchor through AFBS roots, cached ponder subtrees, + search-derived labels. And it is **not even safely narrow today**, because inspected runtime surfaces do not yet close provenance/version/fidelity loop needed for trustworthy narrow rollout path. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

Red-team answers therefore:

* **Does rollout net silently become real quality anchor for decisive search states?** Yes, if it supplies priors/values that determine AFBS root q/visits or exported search labels. (hydra-core/src/afbs.rs:167-186,188-219,265-314)
* **Is current distillation gate too weak to catch quality collapse?** Yes. `distill_kl_drift < 0.1` is necessary but insufficient; it does not certify hard-state root fidelity. (hydra-train/src/eval.rs:111-137)
* **Would falling back to LearnerNet-only hard-state search be safer than degraded rollout path?** Yes, and it already aligns with both doctrine + runtime surfaces. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298; hydra-train/src/inference.rs:151-180)
* **Are throughput gains large enough to justify extra quality risk?** Not by default. Theoretical per-forward gain is bounded, and rollout only earns trust if it materially improves wall-clock performance **without** measurable hard-state search loss. (research/design/HYDRA_FINAL.md:102-107; hydra-train/src/eval.rs:111-137)

Correct future target therefore: **disable-until-better now; if provenance, learner-anchor enforcement, + benchmark plan all pass, promote only to narrow, never central.** In that future narrow mode, rollout may assist only in non-hard, non-decisive acceleration, while LearnerNet remains sole authority for decisive hard-state search + any exported search supervision. Until that closure exists, keeping rollout out of live decisive AFBS is safer + more Hydra-consistent choice. (research/design/HYDRA_RECONCILIATION.md:43-55,149-160,197-205,285-298)

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