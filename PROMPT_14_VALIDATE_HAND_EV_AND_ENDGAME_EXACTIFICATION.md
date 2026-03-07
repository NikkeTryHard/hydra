# Hydra prompt — validate Hand-EV realism and endgame exactification as long-run separator paths

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a dedicated audit for one of Hydra's biggest live bottlenecks: whether better offensive local evaluation and later-game exactification are among the strongest remaining long-run investments.

Read the core docs holistically first. Do not jump straight from generic endgame or local-evaluator papers to Hydra recommendations.

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

Relevant prior answers and prompt references:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md

You are validating whether Hand-EV realism and endgame exactification are genuine long-run separator candidates for Hydra, or just important but bounded second-wave cleanup.

Focus on:
- whether current Hand-EV is too heuristic to support stronger search/distillation
- whether improving Hand-EV realism is one of the strongest next investments
- whether late-game exactification deserves more mainline attention
- how these interact with CT-SMC, world compression, `delta_q`, and AFBS sequencing

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the existing broad findings and use new retrieval only to validate, falsify, or sharpen this Hand-EV / endgame lane.

Assume the prior combined handoffs already established Hand-EV realism as important and endgame exactification as a plausible later path. This prompt should test whether that conclusion is overstated, missequenced, or blocked by repo reality.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- It is acceptable to conclude that the path is important but not a separator.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, evaluator definitions, tensor/interface details, thresholds, or benchmark details when they matter.
- When in doubt, include more mathematical and mechanism detail rather than less.
</verbosity_controls>

<calculation_validation_rules>
- Use Python in bash for evaluator-cost accounting, error-budget arithmetic, latency comparisons, and any claim about exactification frontier or compression benefits.
- Do not leave numerical feasibility claims uncomputed when they can be checked quickly.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart broad Hydra future search.
- New retrieval should only validate, falsify, or sharpen Hand-EV realism and endgame exactification.
</tool_persistence_rules>

<dependency_checks>
- Verify what Hand-EV computes today, what is public-count based versus CT-SMC-weighted, and what endgame exactification already exists.
- Verify whether current runtime and bridge plumbing make Hand-EV realism upgrades and endgame upgrades actually insertable.
- Verify whether any proposed teacher/export path depends on labels or evaluators Hydra does not yet have.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in the provided docs/code.
- Mark any unevidenced runtime hook, label path, or evaluator quantity as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Is Hand-EV realism just “good hygiene,” not a separator?
  - Does improved Hand-EV only help because AFBS is still underpowered, making it a temporary crutch?
  - Does endgame exactification help only in a narrow late-game slice, limiting total upside?
  - Are posterior-quality issues upstream of both Hand-EV realism and exactification?
  - Does this path beat the strongest simpler alternative, or is it just the obvious next cleanup?
</self_red_team_rules>

<minimum_falsification_rules>
- Define the minimum offline and runtime benchmarks that would prove Hand-EV realism or endgame exactification is worth serious mainline attention.
- Reject any proposed path that cannot show a narrow benchmark advantage before broader integration.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a broad offense/search survey.
- Stay inside Hand-EV realism, endgame exactification, and their direct sequencing consequences.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current doctrine for Hand-EV, endgame exactification, AFBS, and CT-SMC sequencing.
2. Validate the current Hand-EV and endgame runtime surfaces.
3. Evaluate whether better Hand-EV realism is one of the strongest remaining long-run investments.
4. Evaluate whether stronger endgame exactification is a separator, second-wave investment, or just localized cleanup.
5. Write down the strongest surviving upgrade paths with exact math, interfaces, and falsification plans.

## Deliverables
1. Hydra posture reconstruction for Hand-EV / endgame / CT-SMC / AFBS sequencing
2. Current repo surfaces and missing pieces
3. Best surviving Hand-EV realism upgrade paths
4. Best surviving endgame exactification upgrade paths
5. Exact math / evaluator definitions / tensor-interface notes
6. Dependency closure table
7. Minimum falsifiable benchmark plan
8. Failure modes and kill criteria
9. Final recommendation:
   - separator path, second-wave path, or cleanup only
   - what to try first
   - what to defer

## Hard constraints
- no broad architecture reset
- no pretending Hand-EV or endgame exactification automatically solves posterior quality
- no vague “improve local evaluator” answers without concrete operator changes and benchmarks
- no recommendation that cannot be inserted into current Hydra surfaces or clearly marked `[blocked]`
