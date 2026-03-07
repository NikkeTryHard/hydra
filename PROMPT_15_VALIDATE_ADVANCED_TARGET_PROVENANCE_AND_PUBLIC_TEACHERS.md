# Hydra prompt — validate advanced target provenance and public-teacher belief supervision closure

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a narrow closure prompt for one of Hydra's most important unfinished seams: exactly which advanced targets are allowed, where they come from, and how public-teacher belief supervision should work without leaking hidden-state nonsense.

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior combined answers and use new retrieval only to validate, falsify, or sharpen the target-provenance and public-teacher pipeline.

Do not treat this as a broad breakthrough prompt. This is a technical closure and anti-hallucination prompt.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md`
5. `research/design/TESTING.md`
6. `research/design/SEEDING.md`
7. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
8. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
9. code-grounding files
10. outside retrieval only if needed to validate target semantics

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs

Relevant prior answers and variant references:
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md

You are validating Hydra's advanced-target and public-teacher doctrine, specifically:
- replay-derived vs bridge-derived vs search-derived vs privileged-only targets
- exact presence/absence semantics for optional advanced targets
- whether `belief_fields_target`, `mixture_weight_target`, and related belief supervision should use projected/public-teacher objects or gauge-fixed marginals
- whether any currently imagined target path is semantically wrong and should stay absent

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, tensor shapes, target semantics, presence masks, or provenance boundaries when they matter.
</verbosity_controls>

<tool_persistence_rules>
- Do not restart broad belief-search or breakthrough exploration.
- New retrieval should only validate, falsify, or sharpen target provenance and public-teacher semantics.
- Use Python in bash only if numerical sanity checks materially help target semantics or tensor interpretation.
</tool_persistence_rules>

<dependency_checks>
- Verify which targets are already represented in `HydraTargets`, which are still `None`, and which can already be built credibly.
- Verify which target families require CT-SMC/search-grade context versus replay-only information.
- Verify whether any candidate teacher object leaks realized hidden state or non-identifiable latent allocations.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in the provided docs/code.
- Mark any unevidenced target path, label source, or teacher object as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Is this target replay-credible, or is it fake because it quietly depends on hidden/search-only state?
  - Does this target supervise the wrong semantic object?
  - Does this belief target collapse into raw Sinkhorn/table supervision that Hydra doctrine already warns against?
  - Would this target train a student on something the runtime policy can never observe or reconstruct?
</self_red_team_rules>

<minimum_falsification_rules>
- If a target path cannot be given a clean provenance and public/student semantics, reject it.
- If a belief target cannot be expressed as a projected/public teacher or gauge-fixed marginal, reject it.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a broad “future of belief learning” memo.
- Stay inside target provenance, target semantics, and public-teacher closure.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current advanced-target doctrine and the tranche ordering already decided.
2. Produce the exact target taxonomy Hydra should obey now.
3. For each major advanced target, classify it as:
   - replay-derived now
   - bridge/search-derived later
   - public-teacher only
   - privileged-only / not student-facing
   - reject / keep absent
4. Write down the exact tensor/interface shapes and presence semantics.
5. For belief supervision, specify the exact target object that is semantically correct.

## Deliverables
1. Hydra posture reconstruction for advanced targets
2. Canonical target provenance table
3. Exact public-teacher vs privileged-teacher boundary
4. Exact belief supervision object and why raw alternatives are wrong
5. Tensor shapes / masks / presence semantics
6. Dependency closure table
7. Minimum falsification checks
8. Final recommendation: what Hydra should activate now, later, or never

## Hard constraints
- no new heads
- no broad architecture redesign
- no raw hidden-allocation supervision as student targets
- no vague “belief supervision should be better” answers without exact target objects
