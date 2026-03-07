# Hydra prompt — validate compute routing, hard-state gating, and pondering control

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
