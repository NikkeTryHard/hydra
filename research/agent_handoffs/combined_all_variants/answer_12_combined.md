<combined_run_record run_id="answer_12" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 12 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_12_VALIDATE_POSTERIOR_CONSENSUS_AND_REGIME_MODELS.md">
  <![CDATA[# Hydra prompt — validate posterior-consensus distillation and regime-coupled opponent modeling

Primary source material lives in the raw GitHub links below.

## Critical directive

This prompt is for the most promising-but-untrustworthy belief/opponent ideas. Be aggressive about rejection.

Read the core docs holistically before searching. Do not treat these ideas as viable just because they sound separator-level.

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

Relevant prior variant writeups and prompt references:
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/008_diagnostic_agent_8.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_diagnostic_agent_8.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_variant_007_prompt_upgrade_ach_like.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_variant_007_prompt_upgrade_ach_like.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_006_new_technique_inventor.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_006_new_technique_inventor.md

You are validating two high-upside but low-trust families:
1. posterior-consensus ExIt / delta-q distillation from multi-world teacher agreement
2. regime-coupled CT-SMC / opponent-mode posteriors tied into search or supervision

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior 8-series materials and use new retrieval only to validate, falsify, or reject these two narrow high-risk families.

Default posture: reserve-shelf or kill. Only keep a survivor if it can be expressed using existing or clearly projectable public-teacher objects already identified in the prior handoffs.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- It is fully acceptable to conclude `0 surviving candidates`.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<novelty_viability_rules>
- Do not preserve a candidate because it sounds like a separator.
- If the method cannot be implemented without major missing label builders, posterior machinery, or runtime coupling, reject it.
- Explicitly write the exact loss, tensor shapes, and pseudocode for any surviving candidate.
</novelty_viability_rules>

<tool_persistence_rules>
- Do not reopen broad belief/opponent idea search.
- New retrieval should only validate, falsify, or sharpen these two candidate families.
</tool_persistence_rules>

<self_red_team_rules>
- Ask explicitly:
  - Does this require labels or posterior objects Hydra does not actually have?
  - Does this violate reconciliation by pushing broad belief/search expansion too early?
  - Is this just a fancy restatement of “improve posterior quality” without a real operator?
  - Does it collapse into a public-only opponent classifier once hidden-state coupling is removed?
  - In a 4-player general-sum game, does opponent-style overconfidence make the whole mechanism brittle?
</self_red_team_rules>

<minimum_falsification_rules>
- For every surviving idea, define the minimum offline benchmark that could kill it before mainline integration.
- If that minimum benchmark still needs a large coupled build-out, reject it.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not produce a broad future-of-belief-modeling memo.
- Stay inside posterior-consensus distillation and regime-coupled opponent/posterior mechanisms only.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current belief/opponent doctrine and active non-goals.
2. Validate whether either family is a real future separator or still speculative reserve-shelf research.
3. Write down exact mathematical formulation, tensor shapes, and pseudocode for any surviving candidate.
4. Produce a dependency closure table that is brutally honest about missing teacher/label/runtime pieces.
5. Prefer rejection over wishful thinking.

## Deliverables
1. Hydra posture reconstruction for belief/opponent modeling
2. Candidate family 1 verdict
3. Candidate family 2 verdict
4. Exact math / tensor shapes / pseudocode for survivors only
5. Dependency closure table
6. Minimum falsifiable prototype
7. Closest known baseline and novelty honesty check
8. Final recommendation: reserve-shelf, second-wave, or kill

## Hard constraints
- no broad public-belief-search proposal unless it survives ruthless falsification
- no vague “posterior quality is important” answers without a real operator
- no recommendation that depends on missing labels without clearly marking `[blocked]`
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_12.md">
  <![CDATA[## Hydra posture reconstruction for belief/opponent modeling

* The controlling doctrine is still **supervision-first closure**, not “belief/search expansion first.” The reconciliation memo says the immediate tranche is to populate advanced targets, activate existing advanced losses in a staged way, and keep deeper AFBS integration for later. ([GitHub][1])

* Hydra’s **belief story is already fixed**: Mixture-SIB is the amortized belief object and CT-SMC is the search-grade posterior. The repo is explicitly not supposed to grow a competing belief stack right now. ([GitHub][2])

* The doctrine is also explicit about what is **not** active now: broad AFBS rollout, new heads, duplicated belief stacks, and speculative novelty with weak insertion points are dropped from the current path; richer latent opponent posteriors and robust-opponent backups are reserve-shelf, not first-tranche work. ([GitHub][2])

* Hydra already has the **surface area**. `model.rs` exposes `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`, and reconciliation says to use the existing output surface as-is rather than expanding architecture. ([GitHub][1])

* The core problem is **closure, not missing modules**. Advanced losses exist but default to zero, the normal batch path still mostly emits baseline targets, and reconciliation explicitly records that `exit_target`, `delta_q_target`, belief, mixture, and opponent-hand-type production paths are still missing or incomplete. ([GitHub][1])

* What is actually live today is narrower than the architecture north star. The loader does populate `oracle_target`, `safety_residual`, and a Stage-A belief teacher, but that Stage-A teacher is only a **public projection** from remaining counts plus total hidden-tile count, built with equalized hidden-zone column sums and a uniform kernel. It is not an event-conditioned posterior over hidden worlds. ([GitHub][3])

* CT-SMC is present, but only as a **generic exact contingency-table sampler with an external likelihood hook**. `bridge.rs` already exposes mixture weights, entropy/ESS, AFBS `delta_q` summaries, and risk/stress features, but there is no repo-evidenced unified latent-opponent posterior object flowing through training and search today. ([GitHub][4])

## Candidate family 1 verdict

**Verdict: kill as currently scoped. There are 0 surviving candidates inside family 1 after the stricter pass.**

This family is attractive for the right reason: single hidden-world teachers are semantically wrong for public-state training. But the full posterior-consensus ExIt / `delta_q` proposal needs exactly the thing Hydra does **not** yet have: a credible **public-teacher action object** over hidden worlds. Reconciliation says the tranche should first wire real `exit_target` / `delta_q_target` / safety targets using the existing heads and no new architecture, and it explicitly records that those production paths are still missing. ([GitHub][1])

The current code-grounded belief teacher is Stage-A public projection, not a world-conditioned action teacher. That matters. Stage-A can justify projected belief supervision; it cannot justify the distinctive family-1 operator, which is “multi-world teacher agreement over action deltas and exit targets.” Without a grounded per-world public action teacher, “posterior consensus” collapses into either posterior-mean label averaging or generic confidence-weighted ExIt. ([GitHub][3])

Your pasted prior materials make the case stricter, not softer. `8new` requires new dataset fields (`exit_target`, `search_need_label`, `gate_features`, `target_weight`) and a new offline posterior-world label builder. Then `8new1` already starts moving away from that line and explicitly rejects “privileged multi-teacher KD now” because, once written out carefully, it mostly collapses into standard KD without a sharper public target object. I agree with that later self-critique.

There is also a surface mismatch. The verified `delta_q` head is 46-wide, but the cheapest family-1 prototype in the prior handoff is discard-only. That can be a useful research slice, but it is not an honest closure of Hydra’s full `delta_q [B,46]` semantics. The gate half of the proposal does not fix that; it only adds another missing label path on top of an already-missing teacher object. ([GitHub][5])

The four-player general-sum red team is unfavorable too. Expert Iteration is a real baseline, and decision-focused uncertainty is a real baseline, but posterior consensus over one-step action deltas is still not a multiplayer equilibrium object. In four-player mahjong, hidden-world disagreement is only one source of error; score dynamics, opponent-response structure, and non-discard branches matter too. That would be acceptable only if the minimum offline benchmark were cheap and decisive. It is not, because even the minimum benchmark still needs a grounded world-teacher builder that is not evidenced in the current prompt’s repo surfaces. So the family fails both the **minimum-falsification** rule and the **novelty-honesty** rule. Under current Hydra constraints it is closer to “uncertainty-weighted ExIt once a real teacher exists” than to a real separator. ([NeurIPS Papers][6])

The only thing worth preserving is a note, not a survivor: if Hydra later gets a real public world-teacher action object, rerun the fixed-compute offline comparison of single-world vs posterior-mean vs consensus labels. That is reserve documentation, not an approved candidate.

## Candidate family 2 verdict

**Verdict: kill.**

This family collides with Hydra’s sequencing doctrine almost perfectly. Reconciliation says richer latent opponent posterior machinery and robust-opponent backup logic are later reserve-shelf multipliers, while the immediate task is to feed existing heads with better targets and improve Hand-EV realism first. Family 2 asks for new latent coupling, new posterior machinery, new supervision, and new search usage all at once. That is exactly the kind of early complexity Hydra is trying not to add. ([GitHub][2])

The grounded code surfaces are insufficient. `model.rs` does give Hydra an `opponent_hand_type` head, and the loss surface has the corresponding optional target slot, but the normal batch path still leaves that target `None`. `ct_smc.rs` only gives particle sampling plus an external likelihood hook, and `robust_opponent.rs` is helper math rather than a deeply wired runtime. The missing centerpiece is a calibrated regime-conditioned opponent-action emission model `p(o \mid I, x, r)`. Without that, there is no coupled filter—only a head, a loss slot, and some later-search math. ([GitHub][1])

The prior family-2 writeup already contains its own failure condition: if the regime posterior does not feed back into particle weights, it collapses to a public-only opponent classifier; if it does feed back into weights but is never used downstream, it becomes just a fancy filter variant. Under current Hydra constraints, that collapse is exactly what would happen first, because the emission model, the `rho[P,3,R]` sidecar, and the search-time `bar_rho` consumer are all still missing. ([GitHub][7])

The four-player general-sum red team is worse here than in family 1. Interacting-multiple-model particle filtering is a known family, and safe robust opponent-exploitation search is a known family, but both depend on calibration. In multiplayer mahjong, overconfident style inference can easily sharpen the wrong hidden-world posterior and then harden the wrong robust backup. The fact that Hydra already has KL-ball/archetype math does not solve that; it only gives a later insertion point for a posterior that would first have to be proven calibrated. ([arXiv][8])

The supposed “cheap” falsifier is not actually cheap enough. Even the discard-only offline version already needs a new emission model, replay event extraction, sidecar filter state and resampling semantics, and posterior-quality evaluation. That is a large coupled build-out for a mechanism that is already off-sequence and only B-level novel even in its best case. So this family is killed, not merely delayed.

## Exact math / tensor shapes / pseudocode for survivors only

There are no survivors after the stricter pass.

I am intentionally not promoting adjacent ideas from `8new1` here. They may be more concrete, but they are outside the two families you asked me to validate. On the scope you set, the correct answer is `0 surviving candidates`.

## Dependency closure table

| Family | Required object                                                                   |      Status | Evidence / blocker                                                                                                                                 | Consequence                  |
| ------ | --------------------------------------------------------------------------------- | ----------: | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 1      | CT-SMC world sampler / ESS                                                        |      exists | CT-SMC already exists as the search-grade sampler and can reweight particles through an external likelihood function. ([GitHub][2])                | not the blocker              |
| 1      | `delta_q` / `safety_residual` model surfaces                                      |      exists | Existing output heads are already present in `model.rs`. ([GitHub][1])                                                                             | not the blocker              |
| 1      | `delta_q_target` train path                                                       |     partial | Train-side slot exists, but the normal sample path still leaves `delta_q_target` absent. ([GitHub][9])                                             | major gap                    |
| 1      | `exit_target` production                                                          |     missing | Reconciliation says upstream `exit_target` production must become part of the tranche; it is not there yet. ([GitHub][1])                          | major gap                    |
| 1      | Multi-world **public action teacher**                                             | `[blocked]` | The repo-evidenced teacher object is Stage-A belief projection, not a grounded world-conditioned action teacher. ([GitHub][10])                    | novelty collapses without it |
| 1      | Reliability / gate labels (`search_need_label`, `target_weight`, `gate_features`) |     missing | Required by the prior family-1 formulation, but not present in repo data plumbing. ([GitHub][7])                                                   | fails minimum falsifier      |
| 1      | Full 46-action semantics for the proposal                                         | `[blocked]` | Verified model head is 46-wide; the prior cheap prototype is discard-only, so full-head semantics are not honestly closed yet. ([GitHub][5])       | head/teacher mismatch        |
| 2      | `opponent_hand_type` head                                                         |      exists | Existing model surface present. ([GitHub][5])                                                                                                      | not the blocker              |
| 2      | `opponent_hand_type_target` path                                                  |     partial | Train-side slot exists, but the normal batch path still leaves it `None`. ([GitHub][9])                                                            | major gap                    |
| 2      | Regime-conditioned emission model `p(o \mid I,x,r)`                               |     missing | No such grounded module or label builder is evidenced in the current repo surfaces.                                                                | hard blocker                 |
| 2      | `rho[P,3,R]` sidecar plus resampling semantics                                    |     missing | Not present in CT-SMC; would require new particle-side state and update logic. ([GitHub][11])                                                      | hard blocker                 |
| 2      | `bar_rho` consumption in bridge / search                                          |     missing | Current grounded bridge/search context carries mixture, CT-SMC, and risk/stress hooks, not regime posterior plumbing. ([GitHub][4])                | hard blocker                 |
| 2      | Robust backup math                                                                |      exists | Helper math exists in `robust_opponent.rs`. ([GitHub][2])                                                                                          | not enough on its own        |
| 2      | Sequencing permission                                                             |        fail | Reconciliation explicitly keeps richer latent opponent posteriors and robust-opponent search backups on the reserve shelf for later. ([GitHub][2]) | doctrine violation           |

## Minimum falsifiable prototype

For **family 1**, the honest minimum falsifier is still the prior hard-state offline suite: fixed-compute comparison of single-world, posterior-mean, and posterior-consensus labels on discard decisions, scored by posterior expected regret and gate AUROC. That benchmark is conceptually fine. The problem is that it already needs the missing piece that makes the family real at all: a grounded world-conditioned public action teacher. Without that, the benchmark is testing a substitute mechanism, not the family you asked about. So family 1 fails the minimum-falsification rule before mainline integration.

One implementation note from my stricter pass: the earlier back-of-envelope compute math was slightly generous. If you model the teacher as `8 worlds × (16 visit-equivalents + 4–16 world-init overhead)`, that is about `1.25×–2.0×` the cost of `1 world × 128 visit-equivalents` per searched hard state, not `1.21×–1.78×`. If only about 5% of states are searched, the overall multiplier is still only about `1.01×–1.05×`. So raw compute is not the real blocker. Teacher provenance is.

For **family 2**, the honest minimum falsifier is the discard-only offline coupled-filter test from the prior writeup: train a public prior plus regime-conditioned discard-likelihood model on replay opponent events, run offline CT-SMC with and without `rho[P,3,R]`, and compare held-out discard NLL, wait recall, and concealed-tile posterior quality at fixed particle count. That is a sensible kill gate. But it already needs a new emission model, replay event extraction, sidecar filter state, and posterior-quality harness. Because the minimum benchmark itself is already a sizable coupled build-out, family 2 also fails the minimum-falsification rule and should be rejected before mainline planning.

## Closest known baseline and novelty honesty check

**Family 1.** The closest known baseline is **Expert Iteration** plus uncertainty-aware / decision-focused teacher weighting. Expert Iteration already treats search as the teacher and a neural network as the student, and decision-focused uncertainty already asks model uncertainty to respect downstream decision loss. The irreducible Hydra twist would have been “posterior hidden-world consensus over one public state.” Under current Hydra constraints that twist does not survive, because the public world-teacher object is missing. Once you strip the family down to what is actually grounded, it reduces to ordinary ExIt with confidence weighting or heuristic gating. That is a **C** today, not a **B**. ([NeurIPS Papers][6])

**Family 2.** The closest known baseline is **interacting-multiple-model / regime-switching particle filtering** plus **safe robust opponent-exploitation search**. The Hydra-specific twist would have been to use the same latent regime posterior for three jobs at once: hidden-tile filtering, soft `opponent_hand_type` supervision, and robust search backup. Under current Hydra constraints that twist also does not survive. If you remove the missing weight feedback, it becomes a public-only opponent classifier. If you keep the filter but not the search use, it is just a known IMM-style filter variant. That is also a **C** today, not a **B**. ([arXiv][8])

There is also an internal honesty check from the 8-series itself. The follow-on `8new1` material you pasted gets more concrete precisely by **moving away** from these two families and toward narrower objects like action-sufficient world compression and projected CT-SMC belief teachers. I am not promoting those here because they are out of scope, but their existence is evidence that the current two families are too diffuse in their present form.

## Final recommendation: reserve-shelf, second-wave, or kill

**Kill.**

After the stricter pass, there are **0 surviving candidates** inside the two families as currently scoped. Family 1 is not ready because its differentiating public-teacher object is missing and the proposal collapses into ordinary uncertainty-weighted ExIt when simplified. Family 2 is further away: it violates sequencing, needs a new emission model and coupled filter, and collapses into known IMM/opponent-classifier variants if partially implemented. Hydra’s actual doctrine remains narrower: close existing target loops first, then reassess richer belief/opponent machinery from a healthier base. ([GitHub][1])

The only thing I would preserve is a note, not a candidate: once Hydra has a credible public world-teacher object and the existing `exit_target` / `delta_q_target` plumbing is real, rerun the family-1 matched-compute offline test. That is reserve documentation. It is not a second-wave recommendation today.

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
