# Hydra breakthrough handoff package guide

This directory contains the tracked handoff artifacts for external deep agents.

## Prompt files

- `prompts/PROMPT_1_TECHNICAL_BREAKTHROUGH.md`
- `prompts/PROMPT_2_REPO_AWARE_NEXT_TRANCHE.md`
- `prompts/PROMPT_3_STRATEGIC_CUTTER.md`
- `prompts/PROMPT_4_OUTSIDE_THE_BOX_BUT_GROUNDED.md`

## Prior answer archive

- `prior_answers/ANSWER_1.md`
- `prior_answers/ANSWER_2.md`
- `prior_answers/ANSWER_3.md`
- `prior_answers/ANSWER_1-1.md`
- `prior_answers/ANSWER_2-1.md`
- `prior_answers/ANSWER_3-1.md`

## Recommended package split

### `hydra_breakthrough_docs_pack.zip` — docs-heavy breakthrough pack
Include:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md`
- `research/design/OPPONENT_MODELING.md`
- `research/design/TESTING.md`
- `research/design/SEEDING.md`
- `docs/GAME_ENGINE.md`
- `research/infrastructure/INFRASTRUCTURE.md`
- `research/design/AGENT_FOLLOWUP_1.md`
- `research/design/AGENT_FOLLOWUP_2.md`
- `research/design/AGENT_FOLLOWUP_3.md`
- all files under `research/agent_handoffs/prior_answers/`
- all files under `research/agent_handoffs/prompts/`

### `hydra_breakthrough_thin_source_pack.zip` — thin-source validation pack
This contains everything in `hydra_breakthrough_docs_pack.zip`, plus:
- `hydra-core/src/encoder.rs`
- `hydra-core/src/ct_smc.rs`
- `hydra-core/src/afbs.rs`
- `hydra-core/src/hand_ev.rs`
- `hydra-core/src/endgame.rs`
- `hydra-core/src/robust_opponent.rs`
- `hydra-train/src/data/sample.rs`
- `hydra-train/src/training/losses.rs`
- `hydra-train/src/model.rs`

## Notes

- `agent_answers/` is intentionally local-only and untracked.
- `.cargo/` is intentionally local-only and untracked.
- Use raw GitHub links for tracked files when an external agent cannot access the zip attachment.
- The 4 prompt files are intentionally structured for GPT-5.4-style deep work using explicit output contracts, research-mode passes, grounding/citation rules, and completion criteria.

## Prompt-to-package routing

- `PROMPT_1_TECHNICAL_BREAKTHROUGH.md` -> attach `hydra_breakthrough_docs_pack.zip`
- `PROMPT_2_REPO_AWARE_NEXT_TRANCHE.md` -> attach `hydra_breakthrough_thin_source_pack.zip`
- `PROMPT_3_STRATEGIC_CUTTER.md` -> attach `hydra_breakthrough_docs_pack.zip`
- `PROMPT_4_OUTSIDE_THE_BOX_BUT_GROUNDED.md` -> attach `hydra_breakthrough_docs_pack.zip` and `deep_agent_20_pdfs.zip`

Self-sufficiency rule:
- `hydra_breakthrough_docs_pack.zip` should be sufficient for prompts 1, 3, and 4.
- `hydra_breakthrough_thin_source_pack.zip` should be sufficient for prompt 2.
- Raw GitHub links are fallback-only when the attachment is inaccessible or corrupted.
