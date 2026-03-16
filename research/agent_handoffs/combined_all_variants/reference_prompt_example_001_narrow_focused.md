<combined_run_record run_id="reference_example_001" variant_id="narrow_focused" schema_version="1">
  <metadata>
    <notes>Reference example prompt for narrow artifact-first blueprint tasks.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
  <![CDATA[# Reference example — narrow focused artifact-first blueprint

<role>
Example role placeholder.
Replace this with the role that fits your actual prompt.
Keep it short and task-specific.
</role>

<task>
Example task placeholder.
Replace this with the actual job the agent should do.

An example task block might ask for:
- what the current quantities or mechanisms really mean
- what is semantically broken or misleading
- what confidence level each major conclusion deserves
- what the clean repaired meanings should be
- what should stay exact, what should stay approximate, and what should be dropped or demoted
- why the confident parts of the answer are actually justified
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections you may customize when the prompt needs it
- distinguish direct artifact support from your own inference
- use search/browse aggressively when it can strengthen the answer: find the original paper, adjacent papers, official docs, repos, and other primary sources; use abstracts or summaries mainly for discovery, not as the final evidence base
- use the bash tool to run Python for lightweight research support work when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, and validation
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- if you claim a path works, survives, or is implementation-ready, show why that confidence is justified and how the claim can be validated or falsified later
- inspect your own draft before finishing: if a confident claim is not objectively justified by visible evidence, downgrade it to inference, proposal, or blocked
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated, falsified, or truly blocked, and do not stop just because the first pass produced a plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when you sound confident, show the justification for that confidence level
- for every important claim, make the validation path visible enough that a reviewer can test it later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate, reproduce, or falsify it ourselves (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
[Insert dense task-specific code/doc/test/formula artifacts here.]
</artifacts>]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This file is a reference prompt example only. It is preserved as a format exemplar for narrow artifact-first blueprint prompts and does not include a paired answer.]]>
  </answer_text>
  </answer_section>
</combined_run_record>
