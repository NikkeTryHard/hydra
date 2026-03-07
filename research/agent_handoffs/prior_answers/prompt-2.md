# Hydra breakthrough prompt 2 — repo-aware next tranche

Attach this zip to the model:
- `hydra_breakthrough_thin_source_pack.zip`

Zip structure the model should expect:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`
- thin source validation slice containing exactly these repo-reality anchors:
  - `hydra-core/src/encoder.rs`
  - `hydra-core/src/ct_smc.rs`
  - `hydra-core/src/afbs.rs`
  - `hydra-core/src/hand_ev.rs`
  - `hydra-core/src/endgame.rs`
  - `hydra-core/src/robust_opponent.rs`
  - `hydra-train/src/data/sample.rs`
  - `hydra-train/src/training/losses.rs`
  - `hydra-train/src/model.rs`

The zip should be sufficient by itself. Use raw links only if the attachment is inaccessible or corrupted.

You are a research advisor helping Hydra become stronger than LuckyJ.

Your job is to propose the **smallest high-leverage next implementation tranche** that most increases Hydra's chance of reaching that ceiling.

<memo_mode>
- Write in a polished implementation memo style.
- Prefer exact boundaries and sequencing over broad theory.
- Synthesize repo reality and design intent into one concrete tranche recommendation.
</memo_mode>

<output_contract>
- Return exactly the sections requested, in the requested order.
- Keep the answer compact, precise, and implementation-facing.
- No nested bullets.
</output_contract>

<verbosity_controls>
- Prefer concise, information-dense writing.
- Avoid repeating the repo context unless it changes the tranche decision.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the few tranche-boundary questions that matter.
  2. Retrieve: use the provided package to resolve what is build-now versus later.
  3. Synthesize: produce one tranche, one boundary, and one clear file/interface plan.
- Stop only when more searching is unlikely to change the tranche choice.
</research_mode>

<citation_rules>
- Only cite sources in the provided package or explicitly supplied links.
- Never fabricate references.
- Attach citations to tranche-defining claims.
</citation_rules>

<grounding_rules>
- Base claims on the supplied docs and any included source slice.
- If a target or interface is only inferred, label it as an inference.
- If repo reality and docs conflict, say so and explain which should dominate.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. chosen one next tranche,
  2. classified key targets into now/later/out,
  3. defined the exact boundary,
  4. given a minimal implementation plan,
  5. named the main scope-creep risks.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Did you choose one tranche rather than several equal options?
  - Did you keep search-dependent items separate from replay-credible items?
  - Is the tranche small enough to implement without architecture sprawl?
</verification_loop>

<dig_deeper_nudge>
- Do not mistake dormant scaffolding for immediate execution readiness.
- Look for hidden dependencies that would make the tranche larger than it first appears.
</dig_deeper_nudge>

Primary sources:
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md`
- `research/design/HYDRA_FINAL.md`
- `docs/GAME_ENGINE.md`
- `research/agent_handoffs/prior_answers/ANSWER_2-1.md`
- `research/agent_handoffs/prior_answers/ANSWER_3-1.md`

Raw GitHub fallback references:
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `BUILD_AGENT_PROMPT.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/BUILD_AGENT_PROMPT.md
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `ANSWER_2-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_2-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

If the thin-source validation pack is not available and you need repo-reality anchors, use these raw source links selectively:
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs

If source files are included in the package, use them only to ground repo reality, not to redesign everything.

Assume these are fixed:
- the active path is supervision-first
- Hand-EV is next after supervision closure
- AFBS stays specialist
- broad search expansion is not the next move

Your job:
1. Pick the one next tranche Hydra should build.
2. Separate everything into:
   - build now
   - later after teacher/search infra exists
   - not part of this tranche
3. Define the exact boundary:
   - what data enters
   - what targets are produced
   - what files/interfaces must change
   - what is explicitly out of scope
4. Explain why this tranche is the best move if the goal is eventually beating LuckyJ.

Constraints:
- Do not widen the tranche into a full architecture program.
- Do not activate every dormant head just because it exists.
- Do not assume search-dependent labels already exist.
- Be ruthless about scope control.

Output format:
1. Best single next tranche
2. Build-now vs later table
3. Exact tranche boundary
4. Minimal file/interface plan
5. Main scope-creep risks
6. Why this tranche matters for LuckyJ-level trajectory

Success means a coding agent could implement the tranche without drifting into a giant side quest.


## 1. Best single next tranche

**Replay-safe safety-residual closure**: wire the already-present `safety_residual` head end-to-end with explicit action-mask/presence plumbing, and leave every search-derived label off for now. That is the smallest tranche that closes a real dormant supervision loop without widening into search, belief-teacher, or oracle-path work. The design memo still says “supervision closure first,” but the source slice sharpens what that means now: the normal batch path already feeds policy, value, GRP, tenpai, danger, opp-next, and score targets, while `belief_fields`, `mixture_weight`, `opponent_hand_type`, `delta_q`, and `safety_residual` are still absent and zero-weighted. `[research/design/HYDRA_RECONCILIATION.md:105-118, 186-190; hydra-train/src/data/sample.rs:145-170, 174-199, 220-275; hydra-train/src/training/losses.rs:30-57]`

I would **not** pull ExIt or `delta_q` into this tranche, even though an earlier handoff leaned that way. The thin-source slice exposes hidden dependencies that make those search labels larger than they first look: ExIt is still documented as a later arena/distill pipeline, `PonderResult` exposes only exit policy/value/visit count, and the encoder-side `delta_q` surface is tile-level `[34]` while the train head is action-space `[46]`. On this question, repo reality should dominate older roadmap simplifications and any temptation to “activate every dormant head.” `[research/agent_handoffs/prior_answers/ANSWER_2-1.md:155-169; research/design/IMPLEMENTATION_ROADMAP.md:67-69, 170-186, 245-261, 710-741; hydra-core/src/afbs.rs:349-364; hydra-core/src/encoder.rs:122-136; hydra-train/src/model.rs:9-24, 83-100, 297-298; research/design/HYDRA_RECONCILIATION.md:234-239, 293-297]`

## 2. Build-now vs later table

| Target / surface                                                                    | Status                                      | Why                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `safety_residual_target` plus explicit action mask / presence                       | **Build now**                               | The head already exists as `[B,46]`, the target slot already exists, and its loss weight is currently `0.0`. This is the only dormant advanced surface that can be credibly activated from replay without requiring a search teacher. `[hydra-train/src/model.rs:22-23, 297-298; hydra-train/src/training/losses.rs:25-27, 55-57; hydra-train/src/data/sample.rs:41-53]` |
| `exit_policy_target`                                                                | **Later after teacher/search infra exists** | The validated slice shows search scaffolding, not a ready search-labeled batch path. ExIt is still staged in the later arena/distill pipeline. `[research/design/IMPLEMENTATION_ROADMAP.md:710-741; hydra-core/src/afbs.rs:349-364]`                                                                                                                                     |
| `delta_q_target`                                                                    | **Later after teacher/search infra exists** | Hidden dependency: encoder-side `delta_q` is `[34]`, while the model head is `[46]`; `PonderResult` also does not export per-action Q/support. That is not a “small supervision” change. `[hydra-core/src/encoder.rs:124-130; hydra-core/src/afbs.rs:349-364; hydra-train/src/model.rs:22, 297]`                                                                         |
| `belief_fields_target`, `mixture_weight_target`, `opponent_hand_type_target`        | **Later after teacher/search infra exists** | The batch path currently sets them to `None`, and the stronger prior guidance is to avoid supervising raw belief surfaces from realized hidden state. `[hydra-train/src/data/sample.rs:165-169, 194-198; hydra-train/src/training/losses.rs:21-26, 510-533; research/agent_handoffs/prior_answers/ANSWER_3-1.md:407-429, 609-613]`                                       |
| Existing baseline aux suite: `grp`, `tenpai`, `danger`, `opp_next`, `score_pdf/cdf` | **Not part of this tranche**                | These are already collated and already have nonzero default weights; they are not the remaining closure gap. `[hydra-train/src/data/sample.rs:153-163, 182-192, 220-275; hydra-train/src/training/losses.rs:32-45]`                                                                                                                                                      |
| `oracle_critic`                                                                     | **Not part of this tranche**                | The model still detaches the oracle path, and the reconciliation memo explicitly says not to widen public-target closure into privileged-path expansion yet. `[hydra-train/src/model.rs:253-266; research/design/HYDRA_RECONCILIATION.md:284-291]`                                                                                                                       |
| Hand-EV realism, AFBS runtime expansion, endgame, robust-opponent integration       | **Not part of this tranche**                | Hand-EV is next **after** supervision closure; deeper search and robust-opponent work are later specialist layers. `[research/design/HYDRA_RECONCILIATION.md:121-146, 149-159, 186-190]`                                                                                                                                                                                 |

## 3. Exact tranche boundary

**Data in.** Use replay samples plus the same hidden-state reconstruction path already implied by the current future-conditioned labels (`tenpai`, `opp_next`, `danger`) to derive an exact immediate deal-in event for discard actions. Pair that with the existing public conservative danger bound from the current safety stack; if that bound is only available behind a loader/bridge accessor, exposing it is an acceptable narrow inference. No AFBS teacher output enters this tranche. `[hydra-train/src/data/sample.rs:41-53, 220-245; docs/GAME_ENGINE.md:185-217]`

**Targets out.** Produce exactly `safety_residual_target: [46]` and `safety_residual_mask: [46]`. Mask should be `1` only for discard-like actions where risk is well-defined (`0..36`, including aka discards) and `0` for calls, pass, agari, riichi-phase stubs, or any row lacking exact replay risk. The target should be the clipped conservatism gap, `clip(u_public(a) - d_exact(a), 0, 1)`, where `d_exact(a)` is “would this discard immediately ron into any opponent under replay-hidden state?” This formula is partly inherited from the prior handoff and adapted to the current `[46]` head shape. `[hydra-train/src/model.rs:22-23, 297-298; research/agent_handoffs/prior_answers/ANSWER_2-1.md:218-226]`

**Files / interfaces.** Extend only the sample-batch-target path and the loss path: one optional action-vector target, one same-shape mask, one masked regression loss. Keep `policy_target` as replay-action supervision in this tranche; do not introduce soft ExIt policy mixing yet. Keep the model, encoder, AFBS, CT-SMC, Hand-EV, endgame, and robust-opponent files behaviorally unchanged. `[hydra-train/src/data/sample.rs:145-170, 174-199, 305-327; hydra-train/src/training/losses.rs:503-556; hydra-train/src/model.rs:253-270]`

**Explicitly out of scope.** No ExIt label generation, no `delta_q` label generation, no belief supervision redesign, no oracle alignment pass, no Hand-EV rewrite, no AFBS semantics rewrite, and no new heads. `[research/design/HYDRA_RECONCILIATION.md:234-239, 284-297; research/design/IMPLEMENTATION_ROADMAP.md:170-186]`

## 4. Minimal file/interface plan

`hydra-train/src/training/losses.rs`: add `safety_residual_mask: Option<Tensor<B,2>>` to `HydraTargets`, add `masked_action_mse(pred, target, mask)`, and use that for `l_safety_residual` instead of the current unmasked dense regression. Do **not** reuse `oracle_guidance_mask`; the current optional-loss path already over-relies on that mask. Keep `w_delta_q = 0.0`; turn on only `w_safety_residual` in the tranche training config, at a small value such as `0.02`. `[hydra-train/src/training/losses.rs:10-27, 46-57, 503-556; research/agent_handoffs/prior_answers/ANSWER_3-1.md:357-395]`

`hydra-train/src/data/sample.rs`: add `safety_residual: Option<[f32; 46]>` and `safety_residual_mask: Option<[f32; 46]>` to `MjaiSample`, thread them through `MjaiBatch`, and populate `HydraTargets.safety_residual_target` plus the new mask. Also add a reusable 46-action suit-permutation helper and apply it in `augment_samples_6x`, because the current augmentation path only knows how to permute single actions and legal masks. `[hydra-train/src/data/sample.rs:41-53, 128-142, 145-170, 204-276, 305-327]`

`hydra-train/src/data/mjai_loader.rs` or equivalent replay-label builder **(inference; file not in slice)**: compute `d_exact(a)` from replay-hidden state, pull the current public danger upper bound with the same aggregation the runtime already uses, and emit only the masked discard-action residual target. If that public bound is not visible from the loader, add a read-only accessor at the safety/bridge boundary; do not invent new semantics. This inference is grounded by the fact that the current samples already carry future-conditioned defensive targets. `[hydra-train/src/data/sample.rs:41-53; docs/GAME_ENGINE.md:185-217; research/agent_handoffs/prior_answers/ANSWER_2-1.md:345-346]`

Everything else stays unchanged: `hydra-train/src/model.rs`, `hydra-core/src/encoder.rs`, `hydra-core/src/afbs.rs`, `hydra-core/src/ct_smc.rs`, `hydra-core/src/hand_ev.rs`, `hydra-core/src/endgame.rs`, and `hydra-core/src/robust_opponent.rs`. `[hydra-train/src/model.rs:253-270; hydra-core/src/encoder.rs:795-822, 832-861; hydra-core/src/afbs.rs:349-364]`

## 5. Main scope-creep risks

The biggest risk is letting this become a stealth ExIt/`delta_q` tranche because the search surfaces already exist. The slice shows that would drag in teacher provenance, root Q export, hard-state gating, and shape reconciliation, which is exactly the architecture sprawl this prompt wants to avoid. `[hydra-core/src/afbs.rs:349-364; hydra-core/src/encoder.rs:124-130; hydra-train/src/model.rs:297-298]`

The second risk is treating a `[46]` action head like a `[34]` tile plane. If the new target and mask are not permuted through suit augmentation exactly the same way as action-space labels, the tranche will silently poison supervision. `[hydra-train/src/data/sample.rs:305-327; hydra-train/src/model.rs:297-298]`

The third risk is “generalizing while we are here”: belief, mixture, hand-type, or oracle work because those fields already exist. The reconciliation memo is explicit that current planning should feed existing surfaces with credible targets, not widen the head program or privileged path. `[research/design/HYDRA_RECONCILIATION.md:234-239, 284-297; hydra-train/src/training/losses.rs:21-27, 503-540]`

## 6. Why this tranche matters for LuckyJ-level trajectory

Hydra does not need another architecture pass before it can climb; it needs one more **credible** supervision surface brought online cleanly. A replay-safe safety residual does that: it teaches the network where the current public danger stack is too conservative, with dense exact defensive labels and no new runtime burden. Against LuckyJ-class opposition, avoidable discard mistakes are expensive, and this is the cheapest path to tightening that part of the learned base. `[research/design/HYDRA_RECONCILIATION.md:177-200; research/design/HYDRA_FINAL.md:113-118]`

Just as importantly, this tranche keeps the sequencing intact. It finishes the smallest remaining supervision closure that is actually build-now, leaves search-derived labels in the “later when teacher/search infra is real” bucket, and preserves the very next slot for Hand-EV realism—the doc-ranked next multiplier after supervision closure. That is the right trajectory toward a LuckyJ ceiling: stronger learned base first, richer offensive oracle second, selective search third. `[research/design/HYDRA_RECONCILIATION.md:121-146, 186-190; research/agent_handoffs/prior_answers/ANSWER_3-1.md:607-619, 707-709]
