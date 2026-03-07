# Hydra prompt — validate action-sufficient CT-SMC world compression

Primary source material lives in the raw GitHub links below.

## Critical directive

Read the core Hydra docs holistically before proposing any compression scheme. Do not jump straight from generic POMDP compression papers to Hydra recommendations.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md`
5. `research/design/TESTING.md`
6. code-grounding files
7. outside retrieval

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
- `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs

Relevant prior variant writeups and prompt references:
- `research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md

You are validating whether action-sufficient world compression is a real long-run Hydra path.

Focus on:
- compressing CT-SMC worlds by decision relevance, not probability mass
- using current Hand-EV / endgame evaluators as local regret geometry
- whether this is a cheap and real seam before deeper AFBS expansion

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior DEBC-AR and 8-series materials and only search enough to validate, falsify, or tighten this specific compression lane.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<calculation_validation_rules>
- Use Python in bash for compression-ratio arithmetic, evaluator-call accounting, and toy regret-clustering sanity checks.
- Do not leave claims like “8 worlds replaces 50-100 worlds” uncomputed.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart a broad belief-compression survey.
- New retrieval should only validate, falsify, or sharpen action-sufficient compression for Hydra's actual runtime seam.
</tool_persistence_rules>

<dependency_checks>
- Verify what CT-SMC exposes today, what Hand-EV/endgame evaluators exist, and whether current runtime already supports the relevant insertion points.
- Verify whether later `delta_q` export is real or still future-only.
</dependency_checks>

<self_red_team_rules>
- Ask explicitly:
  - Does this only preserve noisy evaluator mistakes more efficiently?
  - Is posterior quality the actual bottleneck, not compression quality?
  - Does this fail if Hand-EV realism is not improved first?
  - Does the result reduce to top-mass particle pruning with fancier vocabulary?
</self_red_team_rules>

<minimum_falsification_rules>
- Define the minimum offline benchmark that compares top-mass particle pruning against action-sufficient compression on decision regret at equal evaluator budget.
- Reject the method if it cannot beat simple top-mass pruning offline.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not drift into generic POMDP compression literature review.
- Stay inside CT-SMC world compression for Hydra's current Hand-EV / endgame seam.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current CT-SMC / Hand-EV / endgame posture.
2. Validate whether action-sufficient compression is timely under reconciliation.
3. Write down the exact objective, compression rule, tensor/interfaces, and pseudocode.
4. Separate MVP runtime-only compression from later training export ideas.
5. Decide if this is a serious second-wave investment or still too early.

## Deliverables
1. Hydra posture reconstruction for CT-SMC / Hand-EV / endgame
2. Existing repo surfaces and blockers
3. Exact mathematical formulation of the compression criterion
4. Tensor shapes / runtime payloads
5. Exact pseudocode
6. Dependency closure table
7. Offline falsification benchmark
8. Failure modes and kill criteria
9. Final recommendation: worth it now, later, or not worth it

## Hard constraints
- no broad AFBS rewrite
- no new belief stack
- no fake novelty by renaming belief compression
- no training-first recommendation if runtime-only falsification has not been proven
