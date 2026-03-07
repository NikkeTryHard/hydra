# Hydra prompt — validate posterior-consensus distillation and regime-coupled opponent modeling

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
