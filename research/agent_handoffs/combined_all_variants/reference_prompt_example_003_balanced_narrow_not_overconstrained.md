<combined_run_record run_id="reference_example_003" variant_id="balanced_narrow_not_overconstrained" schema_version="1">
  <metadata>
    <notes>Reference example prompt for narrow tasks that still need some reasoning freedom and should avoid template prison.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
  <![CDATA[# Reference example — balanced narrow prompt

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for a narrow task, but do not treat the current framing as automatically complete or correct.

We want a detailed answer that makes clear:
- what the artifacts directly support
- what is only inference
- what simpler or stronger local alternatives exist inside the same lane
- what should be kept, narrowed, or removed
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
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
  <![CDATA[This file is a reference prompt example only. It is preserved as a format exemplar for balanced narrow prompts and does not include a paired answer.]]>
  </answer_text>
  </answer_section>
</combined_run_record>
