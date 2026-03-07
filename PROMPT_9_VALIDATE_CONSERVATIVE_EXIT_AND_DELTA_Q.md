# Hydra prompt — validate conservative ExIt and delta-Q teacher shaping

Primary source material lives in the raw GitHub links below.

## Critical directive — how to read the core Hydra docs

You must avoid a known bad behavior: fragmented keyword-peeking over large architecture docs.

Bad behavior for this task:
- searching for keywords first
- reading isolated 20-100 line chunks around those keywords
- treating the docs like logs or a grep database
- proposing teacher-shaping math before understanding Hydra's current active doctrine

For this task, that behavior is disqualifying.

Required reading workflow:
1. Use your browse/fetch tool on the raw GitHub links for the core docs listed below.
2. Read those core docs holistically and sequentially before doing narrower searching.
3. Build a high-level model of Hydra's mainline doctrine, especially target-generation closure, AFBS specialist gating, and Hand-EV sequencing.
4. Only after that may you use narrower searching for exact details and outside evidence.

<holistic_ingestion_rules>
- Read the core docs as whole documents before narrowing.
- Do not start with keyword search on the core docs.
- Do not rely on fragmented line-window retrieval for architecture understanding.
- After holistic reading, you may use targeted search for exact details.
</holistic_ingestion_rules>

## Reading order

1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/TESTING.md`
5. `research/design/SEEDING.md`
6. `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
7. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
8. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
9. code-grounding files
10. outside retrieval

## Raw GitHub links

Core Hydra docs:
- `README.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/HYDRA_ARCHIVE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_ARCHIVE.md
- `research/design/IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `research/design/TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `research/design/SEEDING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/SEEDING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md

Code-grounding files:
- `hydra-core/src/afbs.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/afbs.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/training/rl.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/rl.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs

Prior answer archive:
- `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md

Relevant prior variant writeups and prompt references:
- `research/agent_handoffs/combined_all_variants/007_variant_agent_7new.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_variant_agent_7new.md
- `research/agent_handoffs/combined_all_variants/007_variant_agent_7new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_variant_agent_7new1.md
- `research/agent_handoffs/combined_all_variants/007_primary_agent_7.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/007_primary_agent_7.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_pack_005.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_pack_005.md

You are validating whether conservative search-teacher shaping is genuinely one of the best long-run Hydra moves.

This is not an invention prompt. This is a validation-and-design prompt for the family:
- conservative ExIt
- lower-confidence teacher shaping
- support-aware `delta_q` supervision
- narrow target-generation closure around AFBS root stats

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior combined answers and variant writeups as your current frontier, then use new retrieval only to validate, falsify, sharpen, or reject the narrow family under review.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- Return only the few strongest surviving variants.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Use multi-paragraph explanations when a short paragraph would hide important logic.
- Do not omit equations, tensor/interface details, pseudocode, thresholds, assumptions, edge cases, or implementation caveats when they matter.
- When in doubt, include more mathematical detail and mechanism detail rather than less.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Ingest: reconstruct Hydra's current doctrine and exact repo surfaces for ExIt, AFBS root stats, `delta_q`, and safety residuals.
  2. Retrieve: validate only the unresolved evidence around safe policy improvement, trust-gated imitation, support-aware distillation, and search-target reliability.
  3. Synthesize: keep only the conservative teacher operators that survive Hydra-specific grounding and adversarial self-review.
- Stop only when more searching is unlikely to change the final recommendation.
</research_mode>

<tool_persistence_rules>
- Prefer full-document browse/fetch for core docs over fragmented line reading.
- Use Python in bash when math, thresholds, storage arithmetic, or trust-region behavior depends on quantitative checks.
- Do not stop at the first plausible conservative target rule.
- Do not restart a broad idea hunt. Search only where it materially validates, falsifies, or sharpens the already-identified operator family.
</tool_persistence_rules>

<calculation_validation_rules>
- If the proposal depends on uncertainty penalties, KL budgets, support thresholds, storage cost, or target density, compute or sanity-check them explicitly.
- Use short Python scripts for monotonicity checks, threshold sweeps, sparse-vs-dense storage arithmetic, and toy projection sanity checks.
- Do not leave quantitative claims uncomputed if they can be checked quickly.
</calculation_validation_rules>

<dependency_checks>
- Before recommending any target path, verify whether `exit_target`, `delta_q_target`, AFBS root stats, and safety-residual targets already exist, are missing, or are only partially implemented.
- Before proposing a teacher source, verify whether it is replay-credible, bridge-derived, search-derived, or still missing.
</dependency_checks>

<citation_rules>
- Cite only sources actually retrieved in the workflow or included above.
- Never fabricate references.
- Attach citations to exact claims.
</citation_rules>

<grounding_rules>
- Ground Hydra-specific claims in the raw links above.
- Mark any missing target path, runtime hook, or teacher payload as `inference` or `[blocked]` if not directly evidenced.
</grounding_rules>

<self_red_team_rules>
- Before finalizing, try hard to kill each teacher-shaping variant.
- Ask explicitly:
  - Does this amplify AFBS bias faster than it adds useful signal?
  - Does this assume teacher quality that current Hand-EV or current AFBS does not yet deserve?
  - Does this secretly reduce to a simpler existing `exit_target` mixture rule?
  - Does this become too sparse to matter once support and trust thresholds are enforced?
</self_red_team_rules>

<anti_survey_rules>
- Do not return a literature survey.
- Every outside method must change the final operator design, benchmark plan, or rejection decision.
- Do not re-rank broad Hydra futures; stay inside this single target-family lane.
</anti_survey_rules>

<minimum_falsification_rules>
- For every surviving variant, define the minimum falsifiable prototype using only narrow file changes and existing model heads.
- If the variant requires a broad search rewrite or a new belief stack, reject it.
</minimum_falsification_rules>

## What to do

1. Reconstruct Hydra's exact active doctrine for ExIt, AFBS, `delta_q`, and safety residuals.
2. Validate whether conservative teacher shaping is genuinely one of the best long-run Hydra bets.
3. Compare only the narrow operator variants still unresolved after the prior combined handoffs, such as:
   - naïve raw softmax ExIt
   - lower-confidence advantage projection
   - KL-capped teacher distillation
   - sparse support-only `delta_q` supervision
4. Write down the exact math, tensor shapes, and pseudocode for the strongest surviving operators.
5. Kill any variant that depends on Hand-EV realism being solved first unless it explicitly stages Hand-EV out of v0.
6. If the combined handoffs already settled the best operator and your audit finds no material improvement, say `no material change from prior conclusion`.

## Deliverables

Return, in order:
1. Hydra posture reconstruction
2. Current repo fit and missing surfaces
3. Surviving operator variants (max 3)
4. Exact mathematical formulation for each variant
5. Tensor shapes and data payloads
6. Exact pseudocode
7. Dependency closure table
8. Minimum falsifiable benchmark plan
9. Failure modes and kill criteria
10. Final recommendation:
   - best variant to try first
   - what to reject
   - whether this is merely good engineering or a real separator path

## Hard constraints

- no new heads
- no broad AFBS rewrite
- no broad public-belief search
- no vague “just trust search less” answers
- no recommendation that cannot be implemented with current Hydra surfaces plus narrow target plumbing
