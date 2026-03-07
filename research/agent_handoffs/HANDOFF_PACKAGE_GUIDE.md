# Hydra breakthrough handoff package guide

This directory contains the tracked handoff artifacts for external deep agents.

`research/agent_handoffs/combined_all_variants/` is now the SSOT for prompt/answer context artifacts.

See `research/agent_handoffs/RUN_INDEX.md` for the current mapping.

## SSOT archive

- `combined_all_variants/`

This folder now contains:

- direct prompt/answer combined records
- revised answer variants
- diagnostic/context records when needed
- mixed transcript extracts from `agent_answers/`
- prompt-pack and prompt-template preservation records

## Older folders now considered redundant once cleanup completes

- `runs/`
- `combined_runs/`
- `legacy/`
- `prior_answers/`
- `prompts/`
- `archival_runs_without_preserved_answers/`
- `prompt_packs/`

## Future archival rule

For every new external-agent artifact:

1. create one self-contained markdown file under `research/agent_handoffs/combined_all_variants/`
2. place prompt/context at the top
3. place answer/context note at the bottom
4. mark provenance clearly
5. do not create a second archive folder that duplicates the same information

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
- all files under `research/agent_handoffs/combined_all_variants/`

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

- `combined_all_variants/` is now the canonical tracked home for prompt/answer context artifacts.
- older sibling folders are redundant staging/compatibility copies once cleanup completes.
- `.cargo/` is intentionally local-only and untracked.
- Use raw GitHub links for tracked files when an external agent cannot access the zip attachment.
- The tracked prompt templates are intentionally structured for GPT-5.4-style deep work using explicit output contracts, research-mode passes, grounding/citation rules, and completion criteria.

## Prompt-to-package routing

- `PROMPT_1_TECHNICAL_BREAKTHROUGH.md` -> attach `hydra_breakthrough_docs_pack.zip`
- `PROMPT_2_REPO_AWARE_NEXT_TRANCHE.md` -> attach `hydra_breakthrough_thin_source_pack.zip`
- `PROMPT_3_STRATEGIC_CUTTER.md` -> attach `hydra_breakthrough_docs_pack.zip`
- `PROMPT_4_OUTSIDE_THE_BOX_BUT_GROUNDED.md` -> attach `hydra_breakthrough_docs_pack.zip` and `deep_agent_20_pdfs.zip`
- `PROMPT_5_CROSS_FIELD_TRANSFER_HUNTER.md` -> attach `hydra_breakthrough_docs_pack.zip` and `deep_agent_20_pdfs.zip`
- `PROMPT_6_NEW_TECHNIQUE_INVENTOR.md` -> attach `hydra_breakthrough_docs_pack.zip` and `deep_agent_20_pdfs.zip`

Self-sufficiency rule:
- `hydra_breakthrough_docs_pack.zip` should be sufficient for prompts 1, 3, and 4.
- `hydra_breakthrough_thin_source_pack.zip` should be sufficient for prompt 2.
- `hydra_breakthrough_docs_pack.zip` plus `deep_agent_20_pdfs.zip` should be sufficient for prompts 5 and 6.
- Raw GitHub links are fallback-only when the attachment is inaccessible or corrupted.
