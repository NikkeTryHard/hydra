<combined_run_record run_id="reference_example_002" variant_id="broad_novel_fuse_loop" schema_version="1">
  <metadata>
    <notes>Reference example prompt for broad novelty search with artifact grounding and fusion loops.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="assistant_generated_reference_example" kind="reference_prompt_example">
  <![CDATA[# Reference example — broad novel fuse-loop blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for a hard problem where adjacent-field or cross-field formulations might beat the current framing.

We want a detailed answer that makes clear:
- what the current artifact set supports
- what the current framing gets right and wrong
- whether a stronger adjacent-field formulation exists
- whether useful fragments from multiple fields can be fused into a better formulation
- what survives after those fused ideas are validated against the artifacts
- what the narrowest buildable surviving path is

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
- after grounding in the artifacts, explore many adjacent fields for competing formulations of the same problem, keep searching for interesting fragments worth fusing together, and continue the explore -> think hard -> validate loop until the strongest fused formulation either survives or is killed by the artifact constraints
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
[Insert dense task-specific code/doc/test/formula artifacts and external-source anchors here.]
</artifacts>]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="context_note" source_path="assistant_generated_context_note">
  <![CDATA[This file is a reference prompt example only. It is preserved as a format exemplar for broader cross-field novelty prompts and does not include a paired answer.]]>
  </answer_text>
  </answer_section>
</combined_run_record>
