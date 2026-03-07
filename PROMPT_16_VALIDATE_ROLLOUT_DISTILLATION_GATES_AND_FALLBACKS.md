# Hydra prompt — validate rollout-distillation quality gates, trust boundaries, and fallback protocol

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
