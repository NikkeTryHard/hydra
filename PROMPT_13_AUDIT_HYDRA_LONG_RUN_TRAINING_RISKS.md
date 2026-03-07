# Hydra prompt — audit long-run training risks, rollout distillation, capacity, and sequencing

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a fear-audit prompt. Your job is not to defend Hydra. Your job is to find where Hydra could be overstuffed, undertrained, poorly sequenced, or relying on weak teacher semantics.

Read the core docs holistically first.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/TESTING.md`
5. `research/design/SEEDING.md`
6. code-grounding files
7. outside retrieval if needed

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/eval.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/eval.rs
- `hydra-train/src/config.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/config.rs

Relevant prior answers and prompt references:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_001_technical_breakthrough.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_001_technical_breakthrough.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_003_strategic_cutter.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_variant_008_prompt_upgrade_shorter.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_variant_008_prompt_upgrade_shorter.md

You are auditing the long-run Hydra training plan for these fears:
- rollout-net distillation reducing search quality
- learner/actor split being too lossy
- too many advanced heads / targets for the available parameter budget
- too much hidden-state/search information baked into one shared representation
- 2000 GPU-hours being enough only for the conservative Hydra, not the maximal Hydra

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the existing broad findings and use new retrieval only to validate, falsify, or sharpen the fear audit.

This is a residual risk audit after reconciliation and the prior combined handoffs, not a fresh architecture-prioritization exercise.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<calculation_validation_rules>
- Use Python in bash for sample-per-parameter arithmetic, storage/throughput sanity checks, target-density calculations, and any parameter-budget comparisons that matter.
- Do not leave numerical feasibility claims uncomputed when they can be checked.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart broad Hydra future-planning.
- New retrieval should only validate, falsify, or sharpen the training-risk and capacity audit.
</tool_persistence_rules>

<self_red_team_rules>
- Ask explicitly:
  - Which parts of Hydra are robust even if rollout distillation underperforms?
  - Which losses/heads are likely to learn early versus just absorb noise?
  - What is most likely to be undertrained or teacher-limited under the current budget?
  - Does the two-tier network actually reduce risk, or just hide it?
  - If one component had to be deferred, which one should it be and why?
</self_red_team_rules>

<minimum_falsification_rules>
- For every major fear, define the smallest benchmark or gate that would validate or kill it.
- Prefer narrow measurable gates over broad architecture debates.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a generic architecture review.
- Stay inside rollout distillation, model capacity, target pressure, sequencing, and compute sufficiency.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's actual active training posture, not the maximal fantasy version.
2. Separate current mainline from reserve-shelf ambitions.
3. Audit the learner/actor/rollout split.
4. Audit model-capacity and target-pressure risks.
5. Audit compute sufficiency against the staged plan.
6. Recommend the narrowest safe long-run sequencing if the goal is still beating LuckyJ.
7. Do not reopen dormant-head activation strategy except where it creates a concrete risk under the current staged plan.

## Deliverables
1. Hydra posture reconstruction for training/inference/search
2. Learner vs actor vs rollout roles
3. Parameter/capacity risk analysis
4. Compute-budget risk analysis
5. Which heads/targets should learn first vs later
6. Which fears are real vs overstated
7. Concrete gates and benchmarks to de-risk the plan
8. Final recommendation: what to narrow, what to keep, what to defer

## Hard constraints
- no generic “just scale compute” answers
- no broad architecture resets
- no pretending every advanced target should be trained equally early
- no ignoring reconciliation’s sequencing authority
