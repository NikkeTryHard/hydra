<run_record run_id="004_outside_the_box_but_grounded" schema_version="1">
  <metadata>
    <notes>Prompt is preserved. No real answer body survives in the repository, so the answer section below is reconstructed by hand from the surviving prompt intent and neighboring archive context.</notes>
    <layout>single_file_prompt_plus_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="research/agent_handoffs/archival_runs_without_preserved_answers/004_outside_the_box_but_grounded/prompt.md">
  <![CDATA[# Hydra breakthrough prompt 4 — unconventional but grounded
  
  Attach these files to the model:
  - `hydra_breakthrough_docs_pack.zip`
  - `deep_agent_20_pdfs.zip`
  
  Zip structure the model should expect in `hydra_breakthrough_docs_pack.zip`:
  - core design docs under `research/design/`
  - runtime summary under `docs/`
  - infra summary under `research/infrastructure/`
  - prior answer archive under `research/agent_handoffs/prior_answers/`
  - prompt files under `research/agent_handoffs/prompts/`
  
  Zip structure the model should expect in `deep_agent_20_pdfs.zip`:
  - paper PDFs under `deep_agent_20_pdfs/`
  
  The attached zips should be sufficient by themselves. Use raw links only if the attachments are inaccessible or corrupted.
  For this task, do not go looking for the thin-source validation pack unless the docs pack is missing and you have a specific reason to believe source validation is necessary.
  
  You are a research advisor trying to find one unconventional but credible idea that could give Hydra an edge over LuckyJ.
  
  This is the only prompt where you are allowed to push beyond the current obvious active path — but you must stay grounded.
  
  <memo_mode>
  - Write in a compact research memo style.
  - Be explicit about what is evidence-backed versus speculative.
  - Prefer one strong unconventional bet over a list of cool possibilities.
  </memo_mode>
  
  <output_contract>
  - Return exactly the requested sections, in order.
  - Keep the answer compact and high-signal.
  - Choose one candidate unconventional edge or reject the category honestly.
  </output_contract>
  
  <verbosity_controls>
  - Prefer concise, information-dense writing.
  - Avoid generic novelty language.
  </verbosity_controls>
  
  <research_mode>
  - Work in 3 passes:
    1. Plan: identify candidate unconventional edges worth checking.
    2. Retrieve: test them against the supplied evidence and constraints.
    3. Synthesize: keep one or reject all.
  - Stop only when more searching is unlikely to change the final candidate.
  </research_mode>
  
  <citation_rules>
  - Only cite sources in the provided package or explicitly supplied links.
  - Never fabricate references.
  - Attach citations to the claims supporting the unconventional idea.
  </citation_rules>
  
  <grounding_rules>
  - Base claims on provided evidence or clearly labeled inference.
  - If the unconventional idea lacks enough support, say so and reject it.
  - Do not smuggle in broad architecture rewrites as "unconventional bets."
  </grounding_rules>
  
  <completeness_contract>
  - The task is incomplete until you have:
    1. selected one unconventional candidate or rejected all,
    2. explained why it beats more obvious ideas,
    3. identified Hydra insertion point,
    4. proposed the cheapest test,
    5. stated kill criteria.
  </completeness_contract>
  
  <verification_loop>
  - Before finalizing, check:
    - Is the idea truly non-obvious relative to the active path?
    - Is it still technically grounded?
    - Is the test cheap enough to be realistic?
  </verification_loop>
  
  <dig_deeper_nudge>
  - Do not keep an unconventional idea alive just because it is novel.
  - Search for the strongest non-obvious bet, not the strangest one.
  </dig_deeper_nudge>
  
  Primary sources:
  - `research/design/HYDRA_FINAL.md`
  - `research/design/HYDRA_RECONCILIATION.md`
  - `research/evidence/` docs relevant to belief/search/training
  - `research/intel/` docs relevant to Mahjong techniques and references
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
  
  Raw GitHub fallback references:
  - `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
  - `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
  - `ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md
  - `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
  
  Suggested evidence/intel anchors if the zip is unavailable:
  - `research/evidence/multiplayer_search_research.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/multiplayer_search_research.md
  - `research/evidence/BELIEF_STATE_SURVEY.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/BELIEF_STATE_SURVEY.md
  - `research/evidence/TRAINING_TECHNIQUES_SURVEY.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/TRAINING_TECHNIQUES_SURVEY.md
  - `research/intel/CROSS_FIELD_TECHNIQUES.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/intel/CROSS_FIELD_TECHNIQUES.md
  - `research/intel/MAHJONG_TECHNIQUES.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/intel/MAHJONG_TECHNIQUES.md
  
  Your task:
  1. Find the single best unconventional idea that is still plausible for Hydra.
  2. The idea must satisfy all of:
     - not already the obvious active-path item
     - plausible strength upside against LuckyJ-level systems
     - technically testable without rewriting the whole project
     - grounded in at least adjacent evidence, not pure fantasy
  3. Explain exactly how it would fit into Hydra.
  4. Give the cheapest experiment to test it.
  5. Say clearly why it is still worth trying despite the active-path pruning.
  
  Constraints:
  - No broad architecture reboot.
  - No AGPL/code-copy ideas.
  - No giant compute fantasies without a realistic test path.
  - If the idea is too speculative, say so and reject it yourself.
  
  Output format:
  1. Candidate unconventional edge
  2. Why it might beat more obvious ideas
  3. Evidence level
  4. Hydra insertion point
  5. Cheapest test
  6. Kill criteria
  
  Success means you surface one unusually strong non-obvious bet, or conclude honestly that no such bet currently beats the narrowed active path.
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="reconstructed" source_path="hand_written_reconstruction" reconstructed_by="assistant" basis="preserved_prompt_plus_repo_context">
  <![CDATA[## 1. Candidate unconventional edge
  
  Decision-focused belief compression with cheap root action signatures is the strongest unconventional but still grounded bet for this lane. The core idea is to stop compressing hidden-world uncertainty by raw tile similarity alone and instead compress it by whether two hidden worlds would actually change Hydra's root decision.
  
  ## 2. Why it might beat more obvious ideas
  
  This beats more obvious ideas because it targets a bottleneck that can quietly waste compute even when the model, belief, and search modules all improve independently. Hydra already has a serious belief-and-search stack on paper, but if search spends effort separating hidden worlds that are decision-equivalent, then a lot of added search depth just burns latency without changing actions. A decision-focused compression layer is more unconventional than simply making AFBS deeper, but still grounded because it is a narrow interface idea rather than a full architecture reboot.
  
  ## 3. Evidence level
  
  Evidence level: medium. The closest support comes from problem-driven scenario reduction, belief compression, abstraction-refinement, and hidden-information search pathologies. The exact Hydra-specific form would still need validation, but it is technically plausible and cheap enough to test without rewriting the whole project.
  
  ## 4. Hydra insertion point
  
  Primary insertion point would be between CT-SMC particle generation and any expensive root-only AFBS or endgame escalation path. Concretely, the likely touch points are `hydra-core/src/ct_smc.rs`, `hydra-core/src/afbs.rs`, and `hydra-core/src/bridge.rs`, because those are the natural places to derive per-particle root summaries and then decide whether multiple particles can safely share one expensive root evaluation.
  
  ## 5. Cheapest test
  
  The cheapest test is a fixed-state offline benchmark over hard discard states. For each state, compare three variants: baseline fast policy only, current per-particle expensive evaluation, and compressed representative evaluation keyed by root-action signature. Measure root-action agreement with the expensive reference, latency reduction, and whether the compressed path misses catastrophic-defense cases.
  
  ## 6. Kill criteria
  
  Kill the idea quickly if either of these happens: (1) root-action agreement does not stay high enough on the hard-state slice to justify the added complexity, or (2) the compression only helps on easy states where Hydra would not need expensive search anyway.
  
  ## Final note
  
  This reconstruction is hand-written from the preserved prompt and neighboring Hydra archive context. It is not presented as the original missing run output.
  ]]>
  </answer_text>
  </answer_section>
</run_record>
