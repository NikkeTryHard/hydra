<combined_run_record run_id="reference_example_002" variant_id="broad_novel_fuse_loop" schema_version="1">
  <metadata>
    <notes>Ref prompt exemplar: broad novelty search, artifact grounding, fusion loops.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
  <![CDATA[# Reference example — broad novel fuse-loop blueprint

<role>
Example role placeholder.
Replace with role fitting actual prompt.
Keep short, task-specific.
</role>

<task>
Example task placeholder.
Replace with actual agent job.

Example task block may ask:
- what current artifact set supports
- what current framing gets right/wrong
- confidence level for each major conclusion
- whether stronger adjacent-field formulation exists
- whether useful fragments from multiple fields can fuse into better formulation
- what survives after fused ideas validated against artifacts
- why confident parts actually justified
- what narrowest buildable surviving path is

Use artifacts below to derive conclusions.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections; customize when prompt needs
- separate direct artifact support from own inference
- search/browse hard: find original paper, adjacent papers, official docs, repos, other primary sources that may change candidate set; use abstracts/summaries mainly for discovery, not final evidence base
- after grounding in artifacts, explore many adjacent fields for competing formulations of same problem, keep searching for useful fragments worth fusing, continue explore -> think hard -> validate loop until strongest fused formulation either survives or dies under artifact constraints
- use bash tool to run Python for light research support when useful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, validation
- do not dump logic; every important mechanism, threshold, recommendation should be inferable from evidence or explicit in blueprint so it can be validated/reproduced
- if claiming path works, survives, or is implementation-ready, show why confidence justified and how claim can be validated/falsified later
- inspect own draft before finish: if confident claim lacks objective visible evidence, downgrade to inference, proposal, or blocked
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturated, falsified, or truly blocked; do not stop because first pass gave plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when sounding confident, show confidence justification
- for every important claim, make validation path visible enough for later reviewer testing
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail for validation, reproduction, or falsification (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now. Not guaranteed correct. Treat as evidence to inspect/critique, not inherited truth. High chance some are incomplete, misleading, stale, or semantically wrong, so validate all.
</artifact_note>

<artifacts>
[Insert dense task-specific code/doc/test/formula artifacts and external-source anchors here.]
</artifacts>]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This file = ref prompt example only. Preserved as format exemplar for broader cross-field novelty prompts. No paired answer included.]]>
  </answer_text>
  </answer_section>
</combined_run_record>