<combined_run_record run_id="answer_12" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 12 + returned agent answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_12_VALIDATE_POSTERIOR_CONSENSUS_AND_REGIME_MODELS.md">
<![CDATA[# Hydra prompt — validate posterior-consensus distillation and regime-coupled opponent modeling

Primary source material in raw GitHub links below.

## Critical directive

Prompt targets most promising-but-untrustworthy belief/opponent ideas. Reject aggressively.

Read core docs holistically before search. Do not treat ideas as viable merely because they sound separator-level.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `research/design/OPPONENT_MODELING.md`
4. `docs/GAME_ENGINE.md`
5. `research/design/TESTING.md`
6. code-grounding files
7. outside retrieval

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
- `hydra-core/src/robust_opponent.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/robust_opponent.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs

Relevant prior variant writeups + prompt refs:
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/008_diagnostic_agent_8.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_diagnostic_agent_8.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_variant_007_prompt_upgrade_ach_like.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_variant_007_prompt_upgrade_ach_like.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_006_new_technique_inventor.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_006_new_technique_inventor.md

Validate two high-upside, low-trust families:
1. posterior-consensus ExIt / delta-q distillation from multi-world teacher agreement
2. regime-coupled CT-SMC / opponent-mode posteriors tied into search or supervision

Broad exploration already done in `research/agent_handoffs/combined_all_variants/`. Do not redo broad work. Start from prior 8-series materials; use new retrieval only to validate, falsify, or reject these two narrow high-risk families.

Default posture: reserve-shelf or kill. Keep survivor only if expressible with existing or clearly projectable public-teacher objects already identified in prior handoffs.

<output_contract>
- Return exactly requested sections, requested order.
- Be as detailed + explicit as needed; do not optimize for brevity.
- Return full technical treatment, not compressed memo.
- `0 surviving candidates` fully acceptable.
- Short answer usually failure mode for this prompt.
</output_contract>

<novelty_viability_rules>
- Do not preserve candidate because it sounds like separator.
- If method cannot be implemented without major missing label builders, posterior machinery, or runtime coupling, reject.
- Explicitly write exact loss, tensor shapes, + pseudocode for any surviving candidate.
</novelty_viability_rules>

<tool_persistence_rules>
- Do not reopen broad belief/opponent idea search.
- New retrieval should only validate, falsify, or sharpen these two candidate families.
</tool_persistence_rules>

<self_red_team_rules>
- Ask explicitly:
  - Does this require labels or posterior objects Hydra does not have?
  - Does this violate reconciliation by pushing broad belief/search expansion too early?
  - Is this merely fancy restatement of “improve posterior quality” without real operator?
  - Does it collapse into public-only opponent classifier once hidden-state coupling removed?
  - In 4-player general-sum game, does opponent-style overconfidence make whole mechanism brittle?
</self_red_team_rules>

<minimum_falsification_rules>
- For every surviving idea, define minimum offline benchmark that could kill it before mainline integration.
- If that minimum benchmark still needs large coupled build-out, reject.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not produce broad future-of-belief-modeling memo.
- Stay inside posterior-consensus distillation + regime-coupled opponent/posterior mechanisms only.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current belief/opponent doctrine + active non-goals.
2. Validate whether either family is real future separator or still speculative reserve-shelf research.
3. Write exact mathematical formulation, tensor shapes, + pseudocode for any surviving candidate.
4. Produce dependency closure table brutally honest about missing teacher/label/runtime pieces.
5. Prefer rejection over wishful thinking.

## Deliverables
1. Hydra posture reconstruction for belief/opponent modeling
2. Candidate family 1 verdict
3. Candidate family 2 verdict
4. Exact math / tensor shapes / pseudocode for survivors only
5. Dependency closure table
6. Minimum falsifiable prototype
7. Closest known baseline and novelty honesty check
8. Final rec: reserve-shelf, second-wave, or kill

## Hard constraints
- no broad public-belief-search proposal unless it survives ruthless falsification
- no vague “posterior quality is important” answer without real operator
- no rec dependent on missing labels without clearly marking `[blocked]`
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_12.md">
<![CDATA[## Hydra posture reconstruction for belief/opponent modeling

* Controlling doctrine still **supervision-first closure**, not “belief/search expansion first.” Reconciliation memo says immediate tranche = populate advanced targets, activate existing advanced losses in staged way, keep deeper AFBS integration for later. ([GitHub][1])

* Hydra’s **belief story already fixed**: Mixture-SIB = amortized belief object; CT-SMC = search-grade posterior. Repo explicitly should not grow competing belief stack now. ([GitHub][2])

* Doctrine also explicit on what is **not** active now: broad AFBS rollout, new heads, duplicated belief stacks, speculative novelty with weak insertion points all dropped from current path; richer latent opponent posteriors + robust-opponent backups = reserve-shelf, not first-tranche work. ([GitHub][2])

* Hydra already has **surface area**. `model.rs` exposes `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, + `safety_residual`, and reconciliation says use existing output surface as-is rather than expand architecture. ([GitHub][1])

* Core problem = **closure, not missing modules**. Advanced losses exist but default zero, normal batch path still mostly emits baseline targets, and reconciliation explicitly records `exit_target`, `delta_q_target`, belief, mixture, and opponent-hand-type production paths still missing or incomplete. ([GitHub][1])

* What is live today is narrower than architecture north star. Loader does populate `oracle_target`, `safety_residual`, + Stage-A belief teacher, but that Stage-A teacher is only **public projection** from remaining counts + total hidden-tile count, built with equalized hidden-zone column sums + uniform kernel. It is not event-conditioned posterior over hidden worlds. ([GitHub][3])

* CT-SMC present, but only as **generic exact contingency-table sampler with external likelihood hook**. `bridge.rs` already exposes mixture weights, entropy/ESS, AFBS `delta_q` summaries, + risk/stress features, but no repo-evidenced unified latent-opponent posterior object flows through training + search today. ([GitHub][4])

## Candidate family 1 verdict

**Verdict: kill as scoped. There are 0 surviving candidates inside family 1 after stricter pass.**

Family attractive for right reason: single hidden-world teachers are semantically wrong for public-state training. But full posterior-consensus ExIt / `delta_q` proposal needs exactly what Hydra does **not** yet have: credible **public-teacher action object** over hidden worlds. Reconciliation says tranche should first wire real `exit_target` / `delta_q_target` / safety targets using existing heads + no new architecture, and explicitly records those production paths still missing. ([GitHub][1])

Current code-grounded belief teacher = Stage-A public projection, not world-conditioned action teacher. This matters. Stage-A can justify projected belief supervision; it cannot justify distinctive family-1 operator: “multi-world teacher agreement over action deltas and exit targets.” Without grounded per-world public action teacher, “posterior consensus” collapses into either posterior-mean label averaging or generic confidence-weighted ExIt. ([GitHub][3])

Your pasted prior materials make case stricter, not softer. `8new` requires new dataset fields (`exit_target`, `search_need_label`, `gate_features`, `target_weight`) + new offline posterior-world label builder. Then `8new1` already moves away from that line and explicitly rejects “privileged multi-teacher KD now” because, once written carefully, it mostly collapses into standard KD without sharper public target object. I agree with later self-critique.

There is also surface mismatch. Verified `delta_q` head is 46-wide, but cheapest family-1 prototype in prior handoff is discard-only. That can be useful research slice, but it is not honest closure of Hydra’s full `delta_q [B,46]` semantics. Gate half of proposal does not fix that; it only adds another missing label path on top of already-missing teacher object. ([GitHub][5])

Four-player general-sum red team also unfavorable. Expert Iteration is real baseline, and decision-focused uncertainty is real baseline, but posterior consensus over one-step action deltas still is not multiplayer equilibrium object. In four-player mahjong, hidden-world disagreement is only one error source; score dynamics, opponent-response structure, + non-discard branches matter too. That would be acceptable only if minimum offline benchmark were cheap + decisive. It is not, because even minimum benchmark still needs grounded world-teacher builder not evidenced in current prompt’s repo surfaces. So family fails both **minimum-falsification** rule + **novelty-honesty** rule. Under current Hydra constraints it is closer to “uncertainty-weighted ExIt once real teacher exists” than real separator. ([NeurIPS Papers][6])

Only preserve note, not survivor: if Hydra later gets real public world-teacher action object, rerun fixed-compute offline comparison of single-world vs posterior-mean vs consensus labels. That is reserve docs, not approved candidate.

## Candidate family 2 verdict

**Verdict: kill.**

Family collides with Hydra’s sequencing doctrine almost perfectly. Reconciliation says richer latent opponent posterior machinery + robust-opponent backup logic are later reserve-shelf multipliers, while immediate task = feed existing heads with better targets + improve Hand-EV realism first. Family 2 asks for new latent coupling, new posterior machinery, new supervision, + new search usage all at once. That is exactly early complexity Hydra is trying not to add. (research/design/HYDRA_RECONCILIATION.md:162-185,221-243,309-316)

Grounded code surfaces insufficient. `model.rs` does give Hydra `opponent_hand_type` head, and loss surface has corresponding optional target slot, but normal batch path still leaves that target `None`. `ct_smc.rs` only gives particle sampling + external likelihood hook, and `robust_opponent.rs` is helper math rather than deeply wired runtime. Missing centerpiece = calibrated regime-conditioned opponent-action emission model `p(o \mid I, x, r)`. Without that, no coupled filter exists, only head, loss slot, + later-search math. (hydra-train/src/model.rs:19-22,97-99,240-269,294-297; hydra-train/src/data/sample.rs:178-181,210-213; hydra-core/src/ct_smc.rs:228-258; hydra-core/src/robust_opponent.rs:1-16,81-143,235-305)

Prior family-2 writeup already contains its own failure condition: if regime posterior does not feed back into particle weights, it collapses to public-only opponent classifier; if it does feed back into weights but is never used downstream, it becomes merely fancy filter variant. Under current Hydra constraints, that collapse is exactly what would happen first, because emission model, `rho[P,3,R]` sidecar, + search-time `bar_rho` consumer are all still missing. ([GitHub][7])

Four-player general-sum red team worse here than in family 1. Interacting-multiple-model particle filtering is known family, and safe robust opponent-exploitation search is known family, but both depend on calibration. In multiplayer mahjong, overconfident style inference can easily sharpen wrong hidden-world posterior and then harden wrong robust backup. Fact Hydra already has KL-ball/archetype math does not solve that; it only gives later insertion point for posterior that would first need calibration proof. ([arXiv][8])

Supposed “cheap” falsifier is not cheap enough. Even discard-only offline version already needs new emission model, replay event extraction, sidecar filter state + resampling semantics, and posterior-quality evaluation. That is large coupled build-out for mechanism already off-sequence and only B-level novel in best case. So family is killed, not merely delayed.

## Exact math / tensor shapes / pseudocode for survivors only

There are no survivors after stricter pass.

I intentionally am not promoting adjacent ideas from `8new1` here. They may be more concrete, but they are outside two families you asked me to validate. On scope you set, correct answer = `0 surviving candidates`.

## Dependency closure table

| Family | Required object                                                                   |      Status | Evidence / blocker                                                                                                                                 | Consequence                  |
| ------ | --------------------------------------------------------------------------------- | ----------: | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 1      | CT-SMC world sampler / ESS                                                        |      exists | CT-SMC already exists as search-grade sampler and can reweight particles through external likelihood function. (hydra-core/src/ct_smc.rs:10-18,115-129,228-258)                | not blocker                  |
| 1 | `delta_q` / `safety_residual` model surfaces | exists | Existing output heads already present in `model.rs`. (hydra-train/src/model.rs:21-23,97-99,241-270,297-298) | not blocker |
| 1 | `delta_q_target` train path | partial | Train-side slot exists, but normal sample path still leaves `delta_q_target` absent. (hydra-train/src/data/sample.rs:180-181,212-213) | major gap |
| 1 | `exit_target` production | missing | Reconciliation says upstream `exit_target` production must become part of tranche; it is not there yet. (research/design/HYDRA_RECONCILIATION.md:91-99,120-133,320-346,495-495) | major gap |
| 1 | Multi-world **public action teacher** | `[blocked]` | Repo-evidenced teacher object = Stage-A belief projection, not grounded world-conditioned action teacher. (hydra-train/src/teacher/belief.rs:11-17,60-75,96-153) | novelty collapses without it |
| 1 | Reliability / gate labels (`search_need_label`, `target_weight`, `gate_features`) | missing | Required by prior family-1 formulation, but not present in repo data plumbing. ([GitHub][7]) | fails minimum falsifier |
| 1 | Full 46-action semantics for proposal | `[blocked]` | Verified model head is 46-wide; prior cheap prototype is discard-only, so full-head semantics not honestly closed yet. (hydra-train/src/model.rs:103-124,219-226,285-299) | head/teacher mismatch |
| 2 | `opponent_hand_type` head | exists | Existing model surface present. (hydra-train/src/model.rs:21,97,240,268,296) | not blocker |
| 2 | `opponent_hand_type_target` path | partial | Train-side slot exists, but normal batch path still leaves it `None`. (hydra-train/src/data/sample.rs:180,212) | major gap |
| 2 | Regime-conditioned emission model `p(o \mid I,x,r)` | missing | No grounded module or label builder of this kind evidenced in current repo surfaces. | hard blocker |
| 2 | `rho[P,3,R]` sidecar plus resampling semantics | missing | Not present in CT-SMC; would require new particle-side state + update logic. ([GitHub][11]) | hard blocker |
| 2 | `bar_rho` consumption in bridge / search | missing | Current grounded bridge/search context carries mixture, CT-SMC, and risk/stress hooks, not regime posterior plumbing. (hydra-core/src/bridge.rs:27-45,301-355) | hard blocker |
| 2 | Robust backup math | exists | Helper math exists in `robust_opponent.rs`. (hydra-core/src/robust_opponent.rs:1-16,81-143,235-305) | not enough alone |
| 2      | Sequencing permission                                                             |        fail | Reconciliation explicitly keeps richer latent opponent posteriors + robust-opponent search backups on reserve shelf for later. (research/design/HYDRA_RECONCILIATION.md:162-185,221-243,309-316) | doctrine violation           |

## Minimum falsifiable prototype

For **family 1**, honest minimum falsifier still = prior hard-state offline suite: fixed-compute comparison of single-world, posterior-mean, and posterior-consensus labels on discard decisions, scored by posterior expected regret + gate AUROC. Benchmark conceptually fine. Problem: it already needs missing piece that makes family real at all, namely grounded world-conditioned public action teacher. Without that, benchmark tests substitute mechanism, not family you asked about. So family 1 fails minimum-falsification rule before mainline integration.

One impl note from stricter pass: earlier back-of-envelope compute math was slightly generous. If teacher modeled as `8 worlds × (16 visit-equivalents + 4–16 world-init overhead)`, that is about `1.25×–2.0×` cost of `1 world × 128 visit-equivalents` per searched hard state, not `1.21×–1.78×`. If only about 5% of states are searched, overall multiplier still only about `1.01×–1.05×`. So raw compute is not real blocker. Teacher provenance is.

For **family 2**, honest minimum falsifier = discard-only offline coupled-filter test from prior writeup: train public prior + regime-conditioned discard-likelihood model on replay opponent events, run offline CT-SMC with and without `rho[P,3,R]`, and compare held-out discard NLL, wait recall, and concealed-tile posterior quality at fixed particle count. Sensible kill gate. But it already needs new emission model, replay event extraction, sidecar filter state, + posterior-quality harness. Because minimum benchmark itself is already sizable coupled build-out, family 2 also fails minimum-falsification rule and should be rejected before mainline planning.

## Closest known baseline and novelty honesty check

**Family 1.** Closest known baseline = **Expert Iteration** plus uncertainty-aware / decision-focused teacher weighting. Expert Iteration already treats search as teacher and neural network as student, and decision-focused uncertainty already asks model uncertainty to respect downstream decision loss. Irreducible Hydra twist would have been “posterior hidden-world consensus over one public state.” Under current Hydra constraints that twist does not survive, because public world-teacher object is missing. Once family stripped down to what is grounded, it reduces to ordinary ExIt with confidence weighting or heuristic gating. That is **C** today, not **B**. ([NeurIPS Papers][6])

**Family 2.** Closest known baseline = **interacting-multiple-model / regime-switching particle filtering** plus **safe robust opponent-exploitation search**. Hydra-specific twist would have been to use same latent regime posterior for three jobs at once: hidden-tile filtering, soft `opponent_hand_type` supervision, and robust search backup. Under current Hydra constraints that twist also does not survive. If missing weight feedback removed, it becomes public-only opponent classifier. If filter kept but search use removed, it is merely known IMM-style filter variant. That is also **C** today, not **B**. ([arXiv][8])

There is also internal honesty check from 8-series itself. Follow-on `8new1` material gets more concrete precisely by **moving away** from these two families and toward narrower objects like action-sufficient world compression + projected CT-SMC belief teachers. I am not promoting those here because they are out of scope, but their existence is evidence current two families are too diffuse in present form.

## Final recommendation: reserve-shelf, second-wave, or kill

**Kill.**

After stricter pass, there are **0 surviving candidates** inside two families as scoped. Family 1 not ready because differentiating public-teacher object is missing and proposal collapses into ordinary uncertainty-weighted ExIt when simplified. Family 2 farther away: it violates sequencing, needs new emission model + coupled filter, and collapses into known IMM/opponent-classifier variants if partially implemented. Hydra’s actual doctrine remains narrower: close existing target loops first, then reassess richer belief/opponent machinery from healthier base. (research/design/HYDRA_RECONCILIATION.md:120-160,162-185,320-346)

Only thing I would preserve is note, not candidate: once Hydra has credible public world-teacher object and existing `exit_target` / `delta_q_target` plumbing is real, rerun family-1 matched-compute offline test. That is reserve docs. It is not second-wave rec today.

[1]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[5]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs"
[6]: https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf "https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf"
[7]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md"
[8]: https://arxiv.org/html/2410.00620v2 "https://arxiv.org/html/2410.00620v2"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[10]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/teacher/belief.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/teacher/belief.rs"
[11]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs"
]]>
</answer_text>
</answer_section>
</combined_run_record>