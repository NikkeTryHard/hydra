<combined_run_record run_id="answer_16-1" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 16-1 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_16_FOLLOWUP_STRICT_REVALIDATION.md">
  <![CDATA[# Hydra follow-up prompt — strict revalidation pass for Prompt 16 rollout-distillation gates

Use this as a **follow-up to your existing Prompt 16 answer**, not as a fresh broad research task.

Your previous answer is **provisional, not final**. You must now do a stricter second-pass validation that stays fully inside the original Prompt 16 scope:
- rollout-distillation quality gates
- trust boundaries for rollout participation in AFBS
- fallback protocol if rollout quality is not good enough
- minimum falsifiable benchmark needed to justify rollout usage

Do **not** broaden into general Hydra ideation, broad architecture redesign, or unrelated breakthrough proposals.

## Inputs to read first
1. `PROMPT_16_VALIDATE_ROLLOUT_DISTILLATION_GATES_AND_FALLBACKS.md`
2. your current draft in `agent_16.md`
3. `research/design/HYDRA_RECONCILIATION.md`
4. `research/design/HYDRA_FINAL.md`
5. `docs/GAME_ENGINE.md`
6. `research/design/TESTING.md`
7. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
8. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
9. code-grounding files named in the original Prompt 16

## Core instruction
Do not finalize prematurely. Do a much stricter validation pass, but stay fully inside the scope of the original prompt.

In this pass:
- treat your current draft as provisional, not final
- re-check the strongest claims against the cited Hydra docs/code and prior handoff material
- use bash/python if helpful to verify repo surfaces, tensor shapes, thresholds, budget math, overlap, or simple sanity checks
- aggressively red-team each surviving idea against Hydra’s actual constraints, especially:
  - 4-player general-sum dynamics
  - partial observability
  - missing labels / targets / hooks / provenance surfaces
  - compute limits
  - sequencing doctrine in `HYDRA_RECONCILIATION.md`
- kill anything that sounds good but cannot be made concrete within the original prompt’s hard constraints

For ideas that do **not** survive, mention them briefly and state **exactly why** they failed.
For ideas that **do** survive, be more verbose: give the exact math, tensor/interface shapes, pseudocode, dependency closure, and the minimum falsifiable benchmark or gate.

Do not stop at the first plausible answer. End only after you have:
1. re-grounded the answer in repo reality,
2. removed weak or duplicate ideas,
3. hardened the survivors with concrete technical detail,
4. and made the final recommendation narrower if the evidence demands it.

Optimize for truth, verification, and implementation usefulness, not breadth or elegance.

## Why this follow-up exists
Your previous answer had a strong core thesis, but some of the most operationally important thresholds and fallback triggers appeared more precise than the current repo/docs can fully justify.

This follow-up is meant to separate:
- what is explicitly grounded in Hydra docs/code,
- what is a supported inference,
- what is still blocked by missing runtime/config/provenance surfaces,
- and what should be deleted entirely.

## Non-negotiable methodology requirements

You must include an explicit methodology section near the top.

That methodology section must state:
1. which files you actually re-read,
2. which claims you re-checked,
3. whether you used any bash/python sanity checks,
4. how you decided a claim was `grounded`, `supported inference`, or `[blocked]`,
5. and which parts of your earlier draft were demoted, removed, or kept.

You must also include a **claim audit ledger** before the final recommendation.

## Claim audit ledger requirements
For each major claim from your original draft, include a compact table with these columns:

| Claim | Status | Evidence | Why status changed or stayed |
|---|---|---|---|

Where `Status` must be exactly one of:
- `grounded`
- `supported inference`
- `[blocked]`
- `removed`

Rules:
- Every major threshold, gate, fallback trigger, and benchmark requirement must appear in this ledger.
- If a threshold is not directly supported by the repo/docs, it cannot be called `grounded`.
- If a threshold survives only as a reasonable proposal, label it `supported inference` and explain why.
- If a claim depends on missing rollout-specific provenance/versioning/runtime hooks, mark it `[blocked]` unless you are explicitly proposing it as future-only policy.
- If you cannot defend a claim after re-checking, mark it `removed`.

## Evidence discipline requirements
- Do not rely on citation bundles like “([GitHub][7])” alone for important claims.
- For each important claim, include the exact file path and a short quoted phrase or specific code surface.
- If the same conclusion depends on both repo code and prior handoff material, separate those two evidence tiers.
- Do not let prior handoff material silently upgrade a repo-missing claim into a repo-grounded claim.

## Required output structure
Return exactly these sections, in this order:

1. **Methodology and revalidation scope**
2. **What changed from the previous draft**
3. **Claim audit ledger**
4. **Hydra posture reconstruction for actor / learner / rollout roles**
5. **Strict trust boundary after revalidation**
6. **Fallback protocol after deleting weak assumptions**
7. **Surviving gates and thresholds**
8. **Rejected or demoted gates / ideas**
9. **Exact math, tensor/interface shapes, and pseudocode for surviving mechanisms**
10. **Minimum falsifiable benchmark plan**
11. **Dependency closure table**
12. **Final recommendation: central, narrow, optional, or disable-until-better**

## Section-specific requirements

### 1. Methodology and revalidation scope
- Say explicitly that this is a second-pass audit of your own draft.
- List files re-read.
- List any bash/python validation you used.
- State your standard for `grounded`, `supported inference`, `[blocked]`, and `removed`.

### 2. What changed from the previous draft
- Briefly list which claims became weaker, stronger, narrower, or were removed.
- If almost nothing changed, explain why that is justified.

### 3. Claim audit ledger
- Must cover all major claims from the prior draft, especially:
  - rollout default posture
  - hard-state detector fields and thresholds
  - learner-only zone
  - cache/provenance trust rule
  - rollout disable triggers
  - recovery rule
  - hard-state fidelity gates
  - non-hard-state fidelity gates
  - trust-weighted label export rule
  - throughput justification gate
  - benchmark slice sizes / paired-hanchan floors / confidence rules

### 4-8. Revalidated policy sections
- Rewrite these sections so that removed claims are gone.
- If the evidence is weaker than before, the answer must become narrower and more conservative.
- If exact thresholds do not survive, replace them with the narrowest honest statement rather than pretending precision.

### 9. Exact math, tensor/interface shapes, and pseudocode for surviving mechanisms
For each surviving mechanism, provide:
- exact formula
- all symbols defined
- expected tensor or array shapes
- interface/data structure shape if it is runtime-facing
- pseudocode

Examples of acceptable specificity:
- `exit_policy: [46]`
- `delta_q: [46]`
- `legal_mask: [46]`
- `GameStateSnapshot { top2_policy_gap: f32, risk_score: f32, particle_ess: f32, ... }`
- `PonderResult { exit_policy, value, visit_count, ... }`

If you cannot give exact math or interface shapes for a surviving proposal, it should probably not survive.

### 10. Minimum falsifiable benchmark plan
- The benchmark must still test “rollout-assisted AFBS vs LearnerNet-only AFBS.”
- Separate:
  - repo-grounded benchmark gates already present,
  - added validation needed but not yet implemented,
  - and any inferred thresholds that remain merely proposed.
- Include the exact minimum falsifier.

### 11. Dependency closure table
- Make missing rollout-specific runtime surfaces explicit.
- Separate “already present,” “easy extension point,” and “[blocked by missing surface].”

### 12. Final recommendation
- You must choose exactly one:
  - `central`
  - `narrow`
  - `optional`
  - `disable-until-better`
- The recommendation must be justified by the audited ledger, not by rhetorical preference.

## Hard constraints for this follow-up
- Do not broaden the scope beyond Prompt 16.
- Do not restart general Hydra future-planning.
- Do not hide unsupported thresholds inside polished prose.
- Do not let prior combined answers count as repo proof.
- Do not keep duplicate gates that collapse to the same operational rule.
- Do not preserve a threshold merely because it sounds conservative.
- Do not present a future implementation hook as if it already exists.
- Do not omit failures. If a proposed gate or protocol dies under revalidation, say so clearly.

## Desired behavior
If strict revalidation forces a smaller answer, give the smaller answer.
If the honest result is “Hydra only supports a much narrower conclusion than my first draft,” then say that plainly.

The goal of this follow-up is not to sound smarter than the first answer.
The goal is to leave behind a **truer**, **tighter**, and **more implementation-usable** answer.
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_16-1.md">
  <![CDATA[## 1. Methodology and revalidation scope

This is a **second-pass audit of my own previous draft**, not a fresh Hydra design pass. I kept the scope strictly inside Prompt 16: rollout-distillation quality gates, rollout participation trust boundaries in AFBS, fallback when rollout quality is not good enough, and the minimum falsifiable benchmark needed to justify rollout usage.

I re-read these inputs in this pass:

* the Prompt 16 text in this conversation as the effective contents of `PROMPT_16_VALIDATE_ROLLOUT_DISTILLATION_GATES_AND_FALLBACKS.md`
* my previous answer in the prior assistant turn as the effective contents of `agent_16.md`
* `research/design/HYDRA_RECONCILIATION.md`
* `research/design/HYDRA_FINAL.md`
* `docs/GAME_ENGINE.md`
* `research/design/TESTING.md`
* `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
* `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
* `hydra-core/src/afbs.rs`
* `hydra-core/src/bridge.rs`
* `hydra-train/src/model.rs`
* `hydra-train/src/eval.rs`
* `hydra-train/src/config.rs`

I re-checked these major claims from the previous draft:

* rollout default posture
* hard-state detector fields and thresholds
* learner-only zone
* cache/provenance trust rule
* rollout disable triggers
* recovery rule
* hard-state and non-hard-state fidelity gates
* trust-weighted label export
* throughput justification gate
* benchmark floor / paired-hanchan / confidence claims
* whether repo/runtime surfaces actually close the loop for rollout-specific trust enforcement

I used bash/python sanity checks in addition to rereading:

* `grep`, `sed`, and line-number inspection over downloaded raw repo files to verify exact structs, function signatures, constants, and tensor shapes
* a Python arithmetic check on the documented inference times (`0.35 ms` learner vs `0.20 ms` actor/rollout-sized) to compute the maximum per-forward speedup and the fraction of turn time that would have to be network-bound for rollout to buy meaningful wall-clock savings

My status standard in this pass was:

* **grounded**: directly supported by current Hydra docs/code that I re-read here
* **supported inference**: not explicitly stated, but a constrained consequence of grounded repo facts
* **[blocked]**: would require rollout-specific runtime hooks, provenance, label builders, or benchmark harnesses that the re-read repo surfaces do not currently provide
* **removed**: I used it in the previous draft, but after re-checking I cannot defend it, or it duplicates a stricter surviving rule

What I kept, narrowed, demoted, or deleted:

* kept: the core thesis that LearnerNet must remain the search-quality anchor and rollout must not quietly become decisive
* narrowed: the hard-state gate; only `top2_policy_gap < 0.10` survives as an exact threshold
* demoted: `visits(a) >= 4` for `delta_q` support mask; it remains a disciplined proposal, not repo-grounded policy
* blocked: the earlier `lambda_exit` trust-weight formula; required surfaces are not present
* deleted: the prior numeric risk/ESS thresholds, hard/non-hard fidelity percentages, CI floors, paired-hanchan floor, multi-window recovery rule, and the exact throughput promotion bar because the repo/docs do not justify that precision ([GitHub][1])

## 2. What changed from the previous draft

The answer became **narrower and more conservative**.

The strongest surviving point stayed the same: AFBS mechanics make the participating network decisive, so an un-gated rollout path can quietly become the real quality anchor. That claim got stronger in this pass because I re-grounded it directly in `afbs.rs` selection / expansion / exit-policy / best-action logic rather than leaving it at the doctrine level. ([GitHub][2])

The hard-state detector got weaker. In the previous draft I carried forward numeric thresholds like `risk_score > 0.08` and `particle_ess < 0.55`. After revalidation, those numbers do **not** survive. The repo/docs expose `risk_score` and `particle_ess`, and `HYDRA_FINAL.md` does say “high-risk defense” and “low particle ESS,” but it does **not** define numeric cutoffs for them. Only `top2_policy_gap < 0.10` survives as an exact threshold. ([GitHub][1])

Several rollout-specific gates from the previous draft were deleted entirely: the 99% / 98% root-agreement gates, KL p95 thresholds, the `25%` throughput promotion bar, the `20K paired hanchan` floor, CI deltas, and the “three consecutive validation windows” recovery rule. They may be plausible as future policy choices, but they are not supported by the current repo/docs. ([GitHub][3])

One technically important point became more precise: I now distinguish the **runtime AFBS-derived `delta_q` feature plane** from the **model’s `delta_q` output head**. In `bridge.rs`, `delta_q` is populated only over `0..NUM_TILE_TYPES`—that is tile-aligned `[34]` search context—not the full `[46]` action space. In `model.rs`, the supervised `delta_q` head is `[batch, 46]`. My previous draft blurred those two surfaces too much. ([GitHub][4])

The fallback story also tightened. I still conclude that the safest rollback is a **full cache flush** plus learner-only / no-search fallback, but I now label that properly as a **supported inference**, not a repo-grounded runtime rule, because the repo exposes `PonderCache::clear()` but does not yet tag cache entries with rollout provenance. ([GitHub][2])

## 3. Claim audit ledger

| Claim                                                                                                            | Status              | Evidence                                                                                                                                                                                                               | Why status changed or stayed                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rollout default posture should be “disabled in live decisive AFBS right now”                                     | supported inference | `HYDRA_RECONCILIATION.md`: broad “search everywhere” AFBS rollout dropped; “keep the rollout narrow.” `config.rs`: only learner/actor versions. `afbs.rs`: no rollout provenance in `PonderResult`. ([GitHub][5])      | Core thesis stayed, but I downgraded it because the docs do not literally say “disable now”; that remains the safest inference from current doctrine plus missing surfaces. |
| `top2_policy_gap < 0.10` is a valid exact hard-state threshold                                                   | grounded            | `HYDRA_FINAL.md`: hard positions include “top-2 policy gap < 10%.” `afbs.rs`: `GameStateSnapshot.top2_policy_gap`; `compute_ponder_priority` uses `(0.1 - top2_gap).max(0.0)`. ([GitHub][1])                           | Stayed. This is the only exact hard-state threshold that survives.                                                                                                          |
| `risk_score > 0.08` and `particle_ess < 0.55` are valid exact hard-state thresholds                              | removed             | Current repo/docs expose `risk_score` and `particle_ess`, but do not define those cutoffs. Those numbers came from prior handoff material, not the repo. ([GitHub][2])                                                 | Removed because the precision is not defensible from current repo/docs.                                                                                                     |
| Add `wall <= 10` and `orasu` to rollout admission                                                                | removed             | `HYDRA_FINAL.md` uses `wall <= 10` for endgame exactification, not rollout admission; `orasu` appears only in prior handoff proposals. ([GitHub][6])                                                                   | Removed as mis-scoped. It belongs to endgame exactification, not rollout trust gating.                                                                                      |
| Learner-only zone should include decisive AFBS root states and search-label export states                        | supported inference | `afbs.rs`: PUCT selection, top-k expansion, `root_exit_policy`, and `best_action` all depend directly on priors/Q/visits; `HYDRA_FINAL.md`: LearnerNet does deep AFBS on hard positions. ([GitHub][2])                 | Stayed, but now stated as inference from the mechanics rather than as a repo-literal rule.                                                                                  |
| Unknown-provenance cached search results should force full cache flush on rollout disable/change                 | supported inference | `afbs.rs`: `PonderResult` holds `exit_policy`, `value`, `search_depth`, `visit_count`, `timestamp`; no source net/version. `PonderCache` exposes `clear()`. `PipelineState` versions only learner/actor. ([GitHub][2]) | Stayed, but now explicitly marked inference because provenance tagging is missing.                                                                                          |
| Recovery rule requires three consecutive validation windows                                                      | removed             | No such rule appears in current docs/code.                                                                                                                                                                             | Removed because the precision was invented in my prior draft.                                                                                                               |
| Current repo has exact global gates `afbs<150`, `ct_smc<1`, `endgame<100`, `selfplay>20`, `distill_kl_drift<0.1` | grounded            | `HYDRA_FINAL.md` Phase -1 benchmark gates and `eval.rs::BenchmarkGates::passes()`. ([GitHub][1])                                                                                                                       | Stayed. These gates are real.                                                                                                                                               |
| Those global gates should be used as rollout disable triggers                                                    | supported inference | The gates exist in current code/design; using them as rollout fallback triggers is a policy consequence, not a repo-literal rollout rule. ([GitHub][1])                                                                | Narrowed from “grounded trigger list” to “supported inference.”                                                                                                             |
| `min_visits = 64` and `safety_valve_max_kl = 2.0` are current rollout-label emission gates                       | [blocked]           | `config.rs` tests assert those constants, but `HYDRA_RECONCILIATION.md` says `mjai_loader.rs` still has “no production path” for `exit_target` / `delta_q_target`. ([GitHub][7])                                       | Constants are grounded, but calling them active rollout gates is blocked because the production path is missing.                                                            |
| Hard-state fidelity gates like “99% root agreement” / KL p95 thresholds survive                                  | removed             | No current repo/docs benchmark harness defines those numbers. They were my prior proposal only.                                                                                                                        | Removed.                                                                                                                                                                    |
| Non-hard-state fidelity gates like “98% root agreement” / KL mean/p95 survive                                    | removed             | Same problem: no current repo/docs support.                                                                                                                                                                            | Removed.                                                                                                                                                                    |
| Trust-weighted label export via `lambda_exit` survives                                                           | [blocked]           | Formula appears only in prior handoff material; inspected repo surfaces do not expose `m_expanded`, `sigma_Q`, or a wired label-builder that could apply it. ([GitHub][6])                                             | Blocked by missing instrumentation and missing target-production path.                                                                                                      |
| `visits(a) >= 4` survives as the minimal `delta_q` support mask                                                  | supported inference | Prior handoff explicitly says this is “a recommendation, not a branch quote”; current repo exposes AFBS child visits and `delta_q` path, but not this cutoff. ([GitHub][8])                                            | Demoted from implied rule to disciplined proposal.                                                                                                                          |
| Throughput admission rule “require at least 25% AFBS wall-clock win” survives                                    | removed             | Current docs give only the inference-time inputs (`0.35 ms` learner, `0.20 ms` actor-sized). The `25%` threshold was my invention. ([GitHub][1])                                                                       | Removed. The math survives; the threshold does not.                                                                                                                         |
| Minimum benchmark must compare rollout-assisted AFBS vs LearnerNet-only AFBS                                     | supported inference | `HYDRA_FINAL.md` makes LearnerNet the deep AFBS engine on hard positions; prompt scope is “search worse instead of cheaper,” so the only valid control is LearnerNet-only AFBS. ([GitHub][1])                          | Stayed. This is logically required even if not spelled out in the repo.                                                                                                     |
| Paired-hanchan floor / CI thresholds from the prior draft survive                                                | removed             | `eval.rs` exposes metrics, but no current repo/doc specifies those sample floors or CI bars. ([GitHub][3])                                                                                                             | Removed.                                                                                                                                                                    |
| Current inspected repo lacks rollout-specific version/provenance/runtime hooks                                   | grounded            | `PipelineState` has `learner_version` and `actor_version` only; `PonderResult` has no source net/version fields. ([GitHub][7])                                                                                         | Stayed, and became more central.                                                                                                                                            |
| Runtime `delta_q` search feature and model `delta_q` target head are not the same shape                          | grounded            | `bridge.rs` populates `features.delta_q[action]` over `0..NUM_TILE_TYPES`; `GAME_ENGINE.md`/`TESTING.md` fix the live encoder at `192x34`; `model.rs` asserts `delta_q` output is `[batch, 46]`. ([GitHub][4])         | New explicit correction versus my earlier draft.                                                                                                                            |
| `safety_residual` should stay replay-derived rather than rollout-derived                                         | supported inference | Prior handoff says keep it replay-derived/privileged in this tranche; `HYDRA_RECONCILIATION.md` prioritizes provenance discipline and replay-credible targets first. ([GitHub][8])                                     | Stayed, but clearly labeled as inference rather than repo-literal rule.                                                                                                     |

## 4. Hydra posture reconstruction for actor / learner / rollout roles

### Grounded findings

`HYDRA_FINAL.md` still defines Hydra as a two-tier deployed/training system plus a later rollout-distillation idea. The grounded role split is: **LearnerNet** = 24 blocks, ~10M parameters, training plus deep AFBS on hard positions; **ActorNet** = 12 blocks, ~5M parameters, self-play data generation plus shallow SaF features; **RolloutNet** = a later, ActorNet-sized 12-block policy+value model for “fast AFBS rollouts,” distilled continuously from LearnerNet. The same design doc also gives the documented forward times: ActorNet about `0.20 ms`, LearnerNet about `0.35 ms`. ([GitHub][1])

The current inspected code surfaces still close cleanly around **actor** and **learner**, not around a first-class rollout runtime identity. In `model.rs`, `HydraModelConfig::actor()` creates a 12-block model and `HydraModelConfig::learner()` creates a 24-block model. In `config.rs`, `PipelineState` tracks `learner_version` and `actor_version`, but there is no rollout version field in the inspected runtime/config surface. ([GitHub][9])

The current model surface also already exposes the advanced outputs that matter for Prompt 16: `delta_q` and `safety_residual` exist as heads in `HydraOutput`, alongside the rest of the advanced surfaces. `HYDRA_RECONCILIATION.md` explicitly says the model already exposes `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`, and the immediate tranche should use the existing surface “exactly as-is,” not add new heads. ([GitHub][9])

### Supported inference

After revalidation, the safest reading is: **RolloutNet is design doctrine, not an operationally closed runtime trust role in the inspected repo surfaces.** That does not mean the idea is forbidden; it means the current code-grounded surfaces do not yet give it distinct versioning, provenance, or cache identity. So ActorNet and RolloutNet should **not** be silently collapsed into “the cheap 12-block net” for decisive search authority. That conclusion is not repo-literal, but it is the cleanest consequence of the current surface area. ([GitHub][1])

### Blocked / missing evidence

What is missing for a true actor / learner / rollout operational split is exactly what Prompt 16 cares about: a rollout-specific version surface, rollout-specific cache provenance, and a rollout-vs-learner search-fidelity benchmark harness. None of those are visible in the re-read runtime/config/code files. ([GitHub][7])

## 5. Strict trust boundary after revalidation

### Grounded findings

Hydra’s top-level doctrine makes this a high-stakes trust problem, not a generic compression problem. `HYDRA_FINAL.md` frames Riichi Mahjong as a **4-player general-sum, imperfect-information** game, says search targets must optimize the **information state** rather than hidden state, and says any guarantee-like claim must resolve to a theorem, bound, or empirical pass/fail gate. That means a rollout-distillation policy cannot be accepted on vague “average drift seems okay” grounds. ([GitHub][1])

In `afbs.rs`, the participating network is directly inside the decision loop. `puct_select()` uses `q + C_PUCT * prior * sqrt(parent_visits)/(1 + child_visits)`, `expand_node()` truncates to `TOP_K = 5` and renormalizes the selected policy mass, `root_exit_policy()` converts child q-values into a `[46]` policy, and `best_action()` chooses the child with the highest visit count. Those are decisive search surfaces, not advisory side channels. ([GitHub][2])

`HYDRA_RECONCILIATION.md` is equally clear about current sequencing doctrine: AFBS should be “selective and specialist,” not the default path everywhere; broad “search everywhere” rollout is dropped from the active path; and search-only targets must not be mixed into baseline batches unless provenance is explicit. That is the active doctrine I have to respect in this pass. ([GitHub][5])

Hydra also already has a grounded safe path that does **not** depend on rollout. `SearchContext` is entirely optional, `encode_observation()` builds `SearchContext::default()`, and `GAME_ENGINE.md` / `TESTING.md` say the live encoder/model contract is `192x34` with dynamic search/belief context zero-filled behind presence masks when unavailable. So the fallback/no-search path is already part of the current runtime story. ([GitHub][4])

### Supported inference

The strict trust boundary that survives revalidation is therefore narrower than in my first draft:

1. **Grounded hard-state minimum:** `top2_policy_gap < 0.10` is enough by itself to force the search-quality anchor back to LearnerNet.
2. **Decisive-root boundary:** any path that can set decisive AFBS priors/q-values, root exit policy, or root visit allocation cannot be treated as “just a cheap helper.”
3. **Search-label boundary:** any path that exports `exit_target` / `delta_q_target` must be provenance-explicit and learner-anchored before it is trusted.

That is the operational consequence of the current AFBS mechanics plus Hydra’s information-state doctrine. ([GitHub][1])

I am **not** keeping the earlier exact `risk_score` and `particle_ess` cutoffs. What survives is narrower: `GameStateSnapshot` does expose `risk_score` and `particle_ess`, and `HYDRA_FINAL.md` does speak qualitatively about “high-risk defense” and “low particle ESS,” but the current repo/docs do not specify numeric cutoffs. Those signals still matter, but they do not survive this audit as exact rollout admission thresholds. ([GitHub][2])

A subtle but important trust-boundary correction also survives: the runtime AFBS-derived `delta_q` **feature plane** is not the same surface as the supervised `delta_q` **head**. `bridge.rs` only fills `delta_q` over `0..NUM_TILE_TYPES`—tile-type actions, effectively `[34]`—while `model.rs` asserts the head shape is `[batch, 46]`. So any rollout-distillation doctrine that talks about “`delta_q`” must say which of those two surfaces it means; otherwise it is not implementation-usable. ([GitHub][4])

### Blocked / missing evidence

The current inspected surfaces do **not** let me specify a finer live trust boundary for rollout participation in non-hard states, because three critical pieces are missing:

* no rollout-specific versioning surface in `PipelineState`
* no `source_net` / `source_version` on `PonderResult`
* no rollout-vs-learner hard-state fidelity harness in the current benchmark surface ([GitHub][7])

## 6. Fallback protocol after deleting weak assumptions

### Grounded findings

Two grounded fallback primitives already exist in the repo surfaces I re-read:

* `encode_observation()` and `encode_observation_ref()` both create `SearchContext::default()`, which is the existing no-search / no-runtime-context bridge path
* `PonderCache` already exposes `clear()` for whole-cache invalidation ([GitHub][4])

### Supported inference

After deleting the unsupported precision from my first draft, the fallback protocol compresses to this:

**Step 1 — default posture now.**
Do **not** let RolloutNet participate in live decisive AFBS root computation or live search-label export. That is not repo-literal wording, but it is the narrowest safe policy implied by the active doctrine plus the missing provenance surfaces. ([GitHub][5])

**Step 2 — grounded hard-state fallback.**
If `top2_policy_gap < 0.10`, treat the state as learner-anchored. In this pass I am not preserving any extra hard-state numeric thresholds beyond that one. If search is run on that state, the only safe authoritative source is LearnerNet. ([GitHub][1])

**Step 3 — cache fallback.**
On rollout disable, rollout-weight change, or any loss of rollout validation, clear the entire ponder cache. The repo gives `PonderCache::clear()` but not per-entry rollout provenance, so selective reuse is not defendable yet. ([GitHub][2])

**Step 4 — runtime fallback target.**
If Hydra is not using learner-only search on a state, fall back to the already-existing default bridge path: `SearchContext::default()` plus the `192x34` zero-filled/presence-mask encoder contract. That is the current safe “no rollout, no extra runtime context” path. ([GitHub][4])

### Blocked / missing evidence

I am no longer claiming a precise multi-trigger disable matrix, a three-window recovery rule, or a selective cache invalidation policy. Those all require rollout-specific validation/provenance surfaces that the inspected repo does not currently expose. ([GitHub][7])

## 7. Surviving gates and thresholds

### Grounded findings

These exact thresholds survive revalidation:

* **Hard-state minimum threshold:** `top2_policy_gap < 0.10` ([GitHub][1])
* **Current benchmark gates in code:**
  `afbs_on_turn_ms < 150.0`
  `ct_smc_dp_ms < 1.0`
  `endgame_ms < 100.0`
  `self_play_games_per_sec > 20.0`
  `distill_kl_drift < 0.1` ([GitHub][1])
* **Current ExIt config constants present in config:**
  `min_visits = 64`
  `safety_valve_max_kl = 2.0` ([GitHub][7])

### Supported inference

Only one important qualitative interpretation survives: **`distill_kl_drift < 0.1` is a global benchmark gate, not a sufficient rollout-search-quality gate.** It tells Hydra something about learner→actor drift, but Prompt 16 is about whether rollout makes search worse instead of cheaper. The AFBS trust problem is sparse and state-dependent; average distillation drift is not enough. ([GitHub][1])

### Blocked / missing evidence

These do **not** survive as exact gates:

* numeric `risk_score` cutoff
* numeric `particle_ess` cutoff
* any hard/non-hard rollout fidelity percentage or KL threshold
* any exact throughput promotion threshold
* any exact CI or sample-size promotion rule
* any claim that `min_visits = 64` / `KL <= 2.0` are already wired into a live rollout-label emission path, because `HYDRA_RECONCILIATION.md` still says the current loader path has “no production path” for those advanced targets ([GitHub][5])

## 8. Rejected or demoted gates / ideas

**Rejected: `risk_score > 0.08` and `particle_ess < 0.55` as exact rollout gates.**
Why they failed: the current docs/code expose the fields, but not those numeric cutoffs. Those numbers came from prior handoff material, not from current repo-grounded policy. ([GitHub][2])

**Rejected: `wall <= 10` and `orasu` as rollout trust-boundary gates.**
Why they failed: `HYDRA_FINAL.md` uses `wall <= 10` for endgame exactification. That is a different control problem. Carrying it into rollout admission blurred two different doctrines. ([GitHub][6])

**Rejected: 99% / 98% hard/non-hard root-agreement gates and KL p95 bars.**
Why they failed: no current repo benchmark surface defines those thresholds; I proposed them in the previous draft, but I cannot honestly call them repo-grounded or implementation-ready from current surfaces. ([GitHub][3])

**Rejected: `>= 25%` AFBS wall-clock win as a promotion threshold.**
Why it failed: the inference-time math is real, but the promotion bar was my own earlier policy choice, not a repo-grounded threshold. ([GitHub][1])

**Rejected: `20K` paired-hanchan floor, CI deltas, and “three consecutive validation windows” recovery.**
Why they failed: these were plausible evaluation ideas, but no current repo/doc supports those exact numbers. `eval.rs` gives metric surfaces, not those experimental thresholds. ([GitHub][3])

**Demoted to supported inference: `visits(a) >= 4` for `delta_q` support masking.**
Why it only partly survived: the prior handoff explicitly labels it as a recommendation rather than a branch quote. It is still a reasonable guard, but not repo-grounded policy. ([GitHub][8])

**Blocked: `lambda_exit` trust-weighted label export.**
Why it failed operationally: the formula depends on surfaces like expanded branch mass and root-Q variance that are not present in the inspected runtime/config surfaces, and the current repo still lacks the upstream production path for `exit_target` / `delta_q_target`. ([GitHub][6])

**Rejected: rollout-derived `safety_residual` teacher in this tranche.**
Why it failed: it collides with Prompt 16’s trust problem and with the provenance discipline in `HYDRA_RECONCILIATION.md`. The narrower replay-derived interpretation is still the safer one. ([GitHub][8])

## 9. Exact math, tensor/interface shapes, and pseudocode for surviving mechanisms

### Mechanism A — AFBS decisive-root computation (**grounded**)

**Exact math**

For a child node (a) under a parent with visit count (N_p):

[
\mathrm{UCB}(a) = Q(a) + C_{\text{PUCT}} \cdot P(a) \cdot \frac{\sqrt{N_p}}{1 + N(a)}
]

where:

* (Q(a)) is the child q-value (`child.q_value()`)
* (P(a)) is the child prior (`child.prior`)
* (N(a)) is child visit count
* (C_{\text{PUCT}} = 2.5)

Expansion keeps only the top `TOP_K = 5` legal priors and renormalizes the retained mass.

For root exit policy with temperature (\tau > 0):

[
\pi_{\text{exit}}(a) = \frac{\exp((Q(a)-Q_{\max})/\tau)}{\sum_b \exp((Q(b)-Q_{\max})/\tau)}
]

If (\tau \le 0), the implementation returns a one-hot argmax over child q-values.

Root action choice is:

[
a^* = \arg\max_a N(a)
]

**Tensor / interface shapes**

* `policy_logits: [46]`
* `legal_mask: [46]`
* `exit_policy: [46]`
* `NodeIdx = u32`
* `PonderResult { exit_policy: [46], value: f32, search_depth: u8, visit_count: u32, timestamp: Instant }`

**Pseudocode**

```rust
priors = masked_softmax(policy_logits[46], legal_mask[46])
children = top_k(priors, 5)
children = renormalize(children)

repeat search_iteration:
    a = argmax_child( Q[a] + 2.5 * P[a] * sqrt(N_parent) / (1 + N[a]) )
    leaf_value = eval_fn(leaf)
    backpropagate(leaf_value)

exit_policy[46] = softmax(root_child_q / tau)
best_action = argmax(root_child_visit_count)
```

This mechanism is why rollout trust is a real boundary: the net providing priors and leaf values directly changes the decisive search object. ([GitHub][2])

### Mechanism B — minimal exact hard-state gate and ponder-priority surface (**grounded**)

**Exact math**

The only exact hard-state gate that survives revalidation is:

[
g_{\text{hard-min}}(s) = \mathbf{1}[\texttt{top2_policy_gap}(s) < 0.10]
]

The current runtime also exposes a **priority score**, not a trust gate:

[
\mathrm{ponder_priority}(s) = \max(0, 0.1 - \texttt{top2_gap}) \cdot 10 + \max(0, \texttt{risk_score}) + \max(0, 1 - \texttt{particle_ess})
]

**Interface shape**

```rust
GameStateSnapshot {
    info_state_hash: u64,
    top2_policy_gap: f32,
    risk_score: f32,
    particle_ess: f32,
}
```

**Pseudocode**

```rust
hard_min = snapshot.top2_policy_gap < 0.10

ponder_priority =
    max(0.0, 0.1 - snapshot.top2_policy_gap) * 10.0
  + max(0.0, snapshot.risk_score)
  + max(0.0, 1.0 - snapshot.particle_ess)
```

Important constraint: in this pass I am **not** upgrading `risk_score` or `particle_ess` into exact gates, because the current repo/docs do not define those cutoffs. ([GitHub][2])

### Mechanism C — current benchmark gate formula (**grounded**)

**Exact math**

[
g_{\text{bench}} =
\mathbf{1}[
\texttt{afbs_on_turn_ms} < 150
\land
\texttt{ct_smc_dp_ms} < 1
\land
\texttt{endgame_ms} < 100
\land
\texttt{self_play_games_per_sec} > 20
\land
\texttt{distill_kl_drift} < 0.1
]
]

**Interface shape**

```rust
BenchmarkGates {
    afbs_on_turn_ms: f32,
    ct_smc_dp_ms: f32,
    endgame_ms: f32,
    self_play_games_per_sec: f32,
    distill_kl_drift: f32,
}
```

**Pseudocode**

```rust
fn passes(g: BenchmarkGates) -> bool {
    g.afbs_on_turn_ms < 150.0
        && g.ct_smc_dp_ms < 1.0
        && g.endgame_ms < 100.0
        && g.self_play_games_per_sec > 20.0
        && g.distill_kl_drift < 0.1
}
```

This survives as a real gate, but it is not enough by itself to certify rollout search quality. ([GitHub][1])

### Mechanism D — current safe no-search fallback path (**grounded**)

**Authoritative shape note**

Use the live encoder/model contract as **`[192, 34]`**, not the stale bridge doc comment that still mentions `[f32; 2890]`. `TESTING.md` and `GAME_ENGINE.md` are explicit that the live contract is `192x34`; `bridge.rs` still contains an outdated comment string. ([GitHub][10])

**Interface shape**

```rust
SearchContext<'a> {
    mixture: Option<&'a MixtureSib>,
    ct_smc: Option<&'a CtSmc>,
    afbs_tree: Option<&'a AfbsTree>,
    afbs_root: Option<NodeIdx>,
    opponent_risk: Option<&'a [[f32; 34]; 3]>,
    opponent_stress: Option<&'a [f32; 3]>,
}

Observation tensor: [192, 34]
Legal action mask: [46]
```

**Pseudocode**

```rust
let search_context = SearchContext::default();
let obs_tensor: [f32; 192 * 34] =
    encode_observation_with_search_context(
        encoder,
        obs,
        safety,
        drawn_tile,
        &search_context,
    );
```

This is the runtime-safe fallback that already exists when search/belief context is absent or intentionally disabled. ([GitHub][4])

### Mechanism E — cache flush on rollout-state change (**supported inference using existing API**)

**Exact interface shape**

```rust
PonderResult {
    exit_policy: [f32; 46],
    value: f32,
    search_depth: u8,
    visit_count: u32,
    timestamp: Instant,
}

PonderCache {
    entries: DashMap<u64, PonderResult>
}
```

Notably absent from `PonderResult`: any `source_net`, `source_version`, or rollout provenance field.

**Pseudocode**

```rust
// supported-inference policy using current API
on_rollout_disable_or_rollout_weight_change:
    ponder_cache.clear()
```

This survives because it is the narrowest safe action available from current surfaces. Selective invalidation is blocked until cache entries actually carry provenance. ([GitHub][2])

### Mechanism F — `delta_q` feature/target surface separation (**grounded**)

**Exact shapes**

Runtime search feature in `bridge.rs`:

* `delta_q_feature_plane: [34]`
  populated over `0..NUM_TILE_TYPES`

Model output head in `model.rs`:

* `delta_q_head: [batch, 46]`
* `safety_residual_head: [batch, 46]`

Related action-space objects:

* `exit_policy: [46]`
* `legal_mask: [46]`

**Pseudocode**

```rust
// runtime search feature
for action in 0..34 {
    if let Some(child) = tree.find_child_by_action(root, action) {
        features.delta_q[action] = q(child) - q(root);
    }
}

// supervised output surface
model.forward(x).delta_q  // shape [batch, 46]
```

This is implementation-critical: rollout doctrine cannot treat “`delta_q`” as one undifferentiated object. The runtime feature surface and the supervised target/output surface are not the same. ([GitHub][4])

### Mechanism G — throughput sanity check math (**supported inference**)

Using the documented forward times:

* learner forward (t_L = 0.35) ms
* actor/rollout-sized forward (t_R = 0.20) ms

Per-forward speedup:

[
\text{speedup} = \frac{t_L}{t_R} = \frac{0.35}{0.20} = 1.75
]

Per-forward time reduction:

[
1 - \frac{t_R}{t_L} = 1 - \frac{0.20}{0.35} \approx 0.4286
]

If fraction (\phi) of total AFBS turn time is actually replaced network-eval time, then:

[
T_{\text{new}} = T_{\text{old}}\left((1-\phi) + \phi \frac{0.20}{0.35}\right)
]

So to get a target wall-clock reduction (r):

[
\phi_{\text{required}}(r) = \frac{r}{1 - 0.20/0.35}
]

Examples:

* (r = 0.10 \Rightarrow \phi \approx 0.2333)
* (r = 0.25 \Rightarrow \phi \approx 0.5833)

This is why the throughput upside is real but bounded. The math survives; my earlier exact promotion bar does not. ([GitHub][1])

## 10. Minimum falsifiable benchmark plan

### Repo-grounded validation already present

Hydra already has a real **global benchmark gate surface** in `HYDRA_FINAL.md` and `eval.rs`: AFBS latency, CT-SMC latency, endgame latency, self-play throughput, and distillation drift. Those must still pass for any rollout candidate. ([GitHub][1])

Hydra also already has the **shape correctness** surfaces needed to avoid a bogus rollout benchmark: the live encoder/model contract is `192x34`, the legal mask is `[bool; 46]`, and the current model output shapes are asserted in `model.rs`. That matters because a search-quality benchmark is meaningless if the compared systems are not even aligned on the same action/tensor contract. ([GitHub][10])

Finally, Hydra already exposes at least part of the **online evaluation vocabulary** that a rollout candidate would need to report: `EvalResult` includes mean placement, win rate, deal-in rate, and `eval.rs` provides `compute_top2_rate()` and `compute_4th_rate()`. ([GitHub][3])

### Added validation needed but not yet implemented

A rollout candidate is **not** justified by the current repo surfaces alone. Hydra still needs an explicit benchmark harness that compares **rollout-assisted AFBS vs LearnerNet-only AFBS on the same states**. The minimum structure of that harness is:

1. **Matched offline AFBS comparison**
   Same states, same legal masks, same AFBS budget, same root. The control must be LearnerNet-only AFBS, not no-search and not ActorNet-only. This follows from the design role of LearnerNet as the hard-position deep-AFBS engine. ([GitHub][1])

2. **Grounded hard-state slice at minimum**
   At minimum, the benchmark must stratify a slice where `top2_policy_gap < 0.10`, because that is the only exact hard-state threshold that survives this audit. Additional risk/ESS slices may be added later, but they are not exact gates yet. ([GitHub][1])

3. **Authoritative-root audit**
   The benchmark must explicitly compare rollout-assisted vs learner-only root decisions and root exit policies on that hard-state slice, because AFBS mechanics make those decisive objects. ([GitHub][2])

4. **Cache fault-injection test**
   The benchmark must test rollout disable / rollout weight change / validation loss and verify that cached search outputs are not silently reused across that boundary without an explicit clear. Current surfaces only support whole-cache clear safely. ([GitHub][2])

5. **Duplicate online comparison against learner-only control**
   Online evaluation must compare against the LearnerNet-only mainline and report at least mean placement, top-2 rate, 4th rate, win rate, and deal-in rate. Exact sample floors and CI bars do not survive as grounded numbers in this pass, so I am not pretending otherwise. ([GitHub][3])

### Inferred thresholds that remain only proposals

None of the rollout-specific numeric promotion thresholds from my previous draft survive as operational thresholds. I am explicitly deleting:

* 99% / 98% agreement bars
* KL p95 bars
* 25% wall-clock promotion bar
* paired-hanchan floor and CI deltas
* three-window recovery rule

Those may be future policy choices, but the current repo/docs do not justify presenting them as the answer. ([GitHub][6])

### Exact minimum falsifier

Given the narrower evidence base, the **minimum falsifier** is also narrower and more honest:

A rollout candidate is **not acceptable** if **any** of the following is true:

1. it is **not** benchmarked against **LearnerNet-only AFBS** on the same states;
2. it fails the current grounded `BenchmarkGates::passes()` gate;
3. it is allowed to become the authoritative root-decision path on the grounded hard-state slice (`top2_policy_gap < 0.10`) without LearnerNet anchoring;
4. it exports search labels from a path whose provenance is not explicit;
5. it reuses cached search results across rollout disable / rollout weight change without a full clear;
6. its justification rests only on global distillation drift (for example `distill_kl_drift < 0.1`) rather than on rollout-vs-learner search comparison.

That is the tightest falsifiable bar the current evidence supports. Anything more precise would be me inventing a finished policy where the repo still has open trust surfaces. ([GitHub][1])

## 11. Dependency closure table

| Dependency                          | Already present                                                                                               | Easy extension point                                                     | [blocked by missing surface]                                                                                                | Consequence                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Hard-state signal surface           | `GameStateSnapshot { top2_policy_gap, risk_score, particle_ess }` exists. ([GitHub][2])                       | Add calibrated hard-state labels / logged slices keyed off these fields. | Exact numeric `risk_score` / `particle_ess` cutoffs are absent.                                                             | Only `top2_policy_gap < 0.10` survives as an exact gate.                                     |
| Decisive AFBS root surfaces         | `puct_select`, top-k expansion, `root_exit_policy`, `best_action` all exist. ([GitHub][2])                    | Add rollout-vs-learner root-diff logging.                                | No rollout-specific fidelity harness in current benchmark surface.                                                          | Cannot safely let rollout be decisive without new validation.                                |
| Search-free / no-context fallback   | `SearchContext::default()` and `encode_observation()` already exist; live contract is `192x34`. ([GitHub][4]) | None needed for basic fallback.                                          | Bridge comment still contains stale `[f32; 2890]` text. ([GitHub][4])                                                       | Use `TESTING.md` / `GAME_ENGINE.md` as shape authority.                                      |
| Global benchmark gate surface       | `BenchmarkGates::passes()` exists in `eval.rs`; Phase -1 gates exist in `HYDRA_FINAL.md`. ([GitHub][1])       | Add rollout-specific gate fields beside existing latency/drift gates.    | Current gates are global, not rollout-search-fidelity gates.                                                                | Necessary but not sufficient.                                                                |
| Online comparison metrics           | `EvalResult` plus `compute_top2_rate()` and `compute_4th_rate()` exist. ([GitHub][3])                         | Add paired duplicate evaluation against learner-only control.            | No current sample-floor / CI policy.                                                                                        | Online reporting is possible; promotion rules are not closed.                                |
| Search-label config constants       | `min_visits = 64`, `safety_valve_max_kl = 2.0` exist in config tests. ([GitHub][7])                           | Wire those constants into actual target builders.                        | `HYDRA_RECONCILIATION.md` says loader path has “no production path” yet for `exit_target` / `delta_q_target`. ([GitHub][5]) | Present as constants, not as active rollout-label gates.                                     |
| Cache invalidation primitive        | `PonderCache::clear()` exists. ([GitHub][2])                                                                  | Add targeted invalidation once cache entries carry provenance.           | `PonderResult` has no source net/version field. ([GitHub][2])                                                               | Full flush is the safest available policy.                                                   |
| Model version tracking              | `PipelineState` tracks learner and actor versions. ([GitHub][7])                                              | Add rollout version if RolloutNet becomes a real runtime role.           | No rollout version field today.                                                                                             | Rollout state cannot be managed as a first-class deployment identity yet.                    |
| `delta_q` feature vs target mapping | Runtime bridge `delta_q` is tile-aligned; model head `delta_q` is `[batch, 46]`. ([GitHub][4])                | Explicit target-builder mapping / mask semantics.                        | No current repo-grounded rollout-label builder closes this mapping.                                                         | Any rollout label policy must distinguish feature-plane and action-head surfaces explicitly. |

## 12. Final recommendation: central, narrow, optional, or disable-until-better

**disable-until-better**

After strict revalidation, that recommendation is **more** justified than in my previous draft, not less. The reason is not that rollout is a bad idea in principle. The reason is that the current repo/docs do **not** yet close the exact trust surfaces Prompt 16 needs: rollout-specific versioning, rollout-specific cache provenance, a wired search-label production path, and a rollout-vs-LearnerNet search-fidelity benchmark. ([GitHub][7])

The evidence that actually survives is narrower than my earlier draft. Only one exact hard-state threshold survives (`top2_policy_gap < 0.10`). The global benchmark gates survive. The ExIt config constants survive as constants. But the numeric rollout-fidelity thresholds, throughput promotion bar, paired-hanchan floor, CI rules, and recovery windows do **not** survive. Once those are deleted, the honest conclusion is that Hydra currently supports a **negative operational doctrine** more clearly than a positive rollout-admission doctrine: do not let rollout quietly become the decisive search-quality anchor. ([GitHub][1])

So the final tightened answer is:

* **not central** — contradicted by `HYDRA_RECONCILIATION.md`, which explicitly drops broad rollout/search-everywhere behavior from the active path
* **not optional** — too weak, because “optional” without provenance and fidelity closure still lets rollout quietly become decisive
* **not even live-narrow yet** — because the current repo surfaces are missing the operational hooks needed to enforce a narrow live policy safely
* therefore **disable-until-better** is the truest and most implementation-usable answer now ([GitHub][5])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[2]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/eval.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/eval.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md"
[7]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/config.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/config.rs"
[8]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
[9]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs"
[10]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
