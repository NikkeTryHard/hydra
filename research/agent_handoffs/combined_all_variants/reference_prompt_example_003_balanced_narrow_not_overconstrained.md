<combined_run_record run_id="reference_example_003" variant_id="balanced_narrow_not_overconstrained" schema_version="1">
  <metadata>
    <notes>Ref prompt ex for narrow tasks needing some reasoning freedom. Avoid template prison.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
  <![CDATA[# Reference example — balanced narrow prompt

<role>
Ex role placeholder.
Replace with role fitting actual prompt.
Keep short, task-specific.
</role>

<task>
Ex task placeholder.
Replace with actual agent job.

Ex task block may ask:
- what artifacts directly support
- what is inference only
- what confidence each major conclusion deserves
- what simpler or stronger local alts exist in same lane
- what to keep, narrow, remove
- why confident answer parts are justified
- how to implement or validate surviving path with minimal guesswork

Use artifacts below to derive conclusions.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections; customize when prompt needs
- distinguish direct artifact support from own inference
- use search/browse aggressively when it strengthens answer: find original paper, adjacent papers, official docs, repos, other primary sources; use abstracts or summaries mainly for discovery, not final evidence base
- use bash tool to run Python for lightweight research support work when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, validation
- do not dump logic; every important mechanism, threshold, recommendation should be inferable from evidence or explicit in blueprint so it can be validated, reproduced
- if claiming path works, survives, or is implementation-ready, show why confidence justified and how claim can be validated or falsified later
- inspect own draft before finishing: if confident claim lacks objective visible evidence, downgrade to inference, proposal, blocked
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturated, falsified, or truly blocked; do not stop because first pass gave plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when sounding confident, show justification for confidence level
- for every important claim, make validation path visible enough that reviewer can test later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate, reproduce, falsify ourselves (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now. Not guaranteed fully correct. Treat as evidence to inspect, critique; not truth to inherit. High chance some incomplete, misleading, stale, semantically wrong, so validate all.
</artifact_note>

<artifacts>
[Insert dense task-specific code/doc/test/formula artifacts here.]
</artifacts>]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This file = reference prompt ex only. Preserved as format exemplar for balanced narrow prompts. No paired answer included.]]>
  </answer_text>
  </answer_section>
</combined_run_record>