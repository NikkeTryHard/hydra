<combined_run_record run_id="reference_example_001" variant_id="narrow_focused" schema_version="1">
<metadata>
<notes>Ref prompt exemplar for narrow artifact-first blueprint tasks.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
<![CDATA[# Reference example — narrow focused artifact-first blueprint

<role>
Example role placeholder.
Replace with role fitting actual prompt.
Keep short, task-specific.
</role>

<task>
Example task placeholder.
Replace with actual agent job.

Example task block may ask:
- what current quantities or mechanisms mean
- what semantically broken or misleading
- what confidence each major conclusion deserves
- what clean repaired meanings should be
- what stays exact, approximate, or dropped/demoted
- why confident parts justified
- how implement or validate surviving path with minimal guesswork

Use artifacts below to derive conclusions.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections you may customize when prompt needs it
- distinguish direct artifact support from your own inference
- use search/browse aggressively when it can strengthen answer: find original paper, adjacent papers, official docs, repos, other primary sources; use abstracts or summaries mainly for discovery, not final evidence base
- use bash tool to run Python for lightweight research support work when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, validation
- do not dump logic; every important mechanism, threshold, or rec should be inferable from evidence or explicit in blueprint so it can be validated and reproduced
- if you claim path works, survives, or is impl-ready, show why confidence justified and how claim can be validated or falsified later
- inspect your own draft before finishing: if confident claim not objectively justified by visible evidence, downgrade to inference, proposal, or blocked
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturated, falsified, or truly blocked, and do not stop because first pass produced plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when you sound confident, show confidence justification
- for every important claim, make validation path visible enough that reviewer can test later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate, reproduce, or falsify ourselves (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs appear to say now. Not guaranteed fully correct. Treat as evidence to inspect and critique, not truth to inherit. High chance some incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
[Insert dense task-specific code/doc/test/formula artifacts here.]
</artifacts>]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="context_note" source_path="assistant_generated_context_note">
<![CDATA[File = reference prompt example only. Preserved as format exemplar for narrow artifact-first blueprint prompts; no paired answer included.]]>
</answer_text>
</answer_section>
</combined_run_record>