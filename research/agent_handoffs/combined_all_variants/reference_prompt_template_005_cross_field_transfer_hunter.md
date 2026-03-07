<combined_run_record run_id="prompt_5_cross_field_transfer_hunter" variant_id="prompt_template" schema_version="1">
  <metadata>
    <notes>Historical prompt template preserved inside combined_all_variants for SSOT coverage.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="research/agent_handoffs/prompts/PROMPT_5_CROSS_FIELD_TRANSFER_HUNTER.md" kind="prompt_template">
  <![CDATA[# Hydra breakthrough prompt 5 — cross-field transfer hunter

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

  The attached zips should be sufficient for core Hydra context, but for this task you are explicitly allowed to search and browse additional papers online from other fields. Use raw links only if the attachments are inaccessible or corrupted.

  You are a research advisor trying to help Hydra beat LuckyJ by importing the **best transferable ideas from other fields**, not by staying inside Mahjong-only literature.

  <memo_mode>
  - Write in a compact research memo style.
  - Separate proven transfer, plausible transfer, and speculative transfer clearly.
  - Prefer a few high-quality transfers over a long list of weak ones.
  </memo_mode>

  <output_contract>
  - Return exactly the requested sections, in order.
  - Keep the answer compact, high-signal, and evidence-backed.
  - Do not produce a generic survey.
  </output_contract>

  <verbosity_controls>
  - Prefer concise, information-dense writing.
  - Do not waste space explaining obvious background.
  </verbosity_controls>

  <research_mode>
  - Work in 3 passes:
    1. Plan: identify 4-6 outside fields or problem classes most likely to transfer into Hydra.
    2. Retrieve: search papers/examples from those fields and follow 1-2 strong second-order leads.
    3. Synthesize: keep only the transfers that survive Hydra-specific scrutiny.
  - Stop only when more searching is unlikely to change the ranking of the top transfer candidates.
  </research_mode>

  <citation_rules>
  - Cite the provided Hydra docs and any externally retrieved papers/examples used in your argument.
  - Never fabricate references.
  - Attach citations to the exact transfer claims they support.
  </citation_rules>

  <grounding_rules>
  - Ground every transfer idea in both:
    1. outside evidence, and
    2. a concrete Hydra insertion point.
  - If the transfer is analogy-heavy rather than directly evidenced, say so explicitly.
  </grounding_rules>

  <completeness_contract>
  - The task is incomplete until you have:
    1. identified the best outside fields,
    2. ranked the top transfer candidates,
    3. explained why each transfers to Mahjong specifically,
    4. named the Hydra insertion point,
    5. proposed a cheap test for each serious candidate.
  </completeness_contract>

  <verification_loop>
  - Before finalizing, check:
    - Did you search outside Mahjong rather than just adjacent Mahjong papers?
    - Did you reject weak transfers instead of preserving them?
    - Is each surviving transfer linked to a real Hydra bottleneck or leverage point?
  </verification_loop>

  <dig_deeper_nudge>
  - Don’t stop at the first cool adjacent paper.
  - Look for fields with structural similarity to Mahjong: partial observability, finite hidden pools, bounded-horizon planning, robustness to misspecification, deferred expensive computation, and calibration under uncertainty.
  </dig_deeper_nudge>

  Primary Hydra sources:
  - `research/design/HYDRA_FINAL.md`
  - `research/design/HYDRA_RECONCILIATION.md`
  - `research/design/IMPLEMENTATION_ROADMAP.md`
  - `research/design/OPPONENT_MODELING.md`
  - `research/design/TESTING.md`
  - `research/agent_handoffs/combined_all_variants/answer_1-1_combined.md`
  - `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`

  Use these outside-search targets aggressively:
  - probabilistic inference / Bayesian filtering
  - planning under uncertainty / imperfect-information game AI
  - structured prediction / matching / transport
  - robust optimization / distribution shift
  - retrieval-augmented planning / memory reuse
  - selective compute / cascaded inference / test-time compute allocation
  - decision-time calibration / abstention / epistemic uncertainty
  - cross-disciplinary systems that solve “expensive-but-not-always-needed” reasoning problems well

  Your job:
  1. Search broadly across other fields.
  2. Find the **best transferable techniques** that Hydra is not already using clearly.
  3. Rank the top 3–5 transfers by expected upside for beating LuckyJ.
  4. For each, explain:
     - why it transfers,
     - what exact Hydra problem it solves,
     - what it would look like inside Hydra,
     - why it might fail,
     - and the cheapest experiment to test it.

  Constraints:
  - Do not propose generic “use transformers / use bigger models / use more search.”
  - Do not keep a transfer idea alive if the mapping to Mahjong is weak.
  - Do not recommend ideas whose cost obviously overwhelms likely benefit unless you can argue for a cheap test path.
  - Do not rely on AGPL code or implementation borrowing.

  Output format:
  1. Best outside fields to mine
  2. Top transferable techniques (ranked)
  3. Hydra insertion points
  4. Cheap tests
  5. Kill criteria
  6. Final recommendation

  Success means you surface a small number of outside-field ideas that could plausibly give Hydra a real edge over LuckyJ rather than just making it more complicated.
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This artifact preserves a historical prompt template. It is included in the SSOT folder so the prompt text survives archive cleanup, but it does not represent a single answer-bearing run by itself.
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
