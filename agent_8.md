# Why PROMPT_7 and PROMPT_8 started producing bold labels with short paragraphs

Yes — we pretty clearly told it to do something in that direction.

## Short answer

The formatting is mostly coming from the prompt design, not from random model drift.

Both `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md` and `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md` explicitly push the model toward:

- exact section ordering
- compact output
- information-dense writing
- prototype-oriented answers instead of long narrative prose

That combo strongly nudges GPT-5.4 Pro into a structured markdown brief style, which often shows up as:

- a heading per major technique
- bold field labels like `**Problem solved.**`
- one short paragraph per field

So yeah: we did not literally say "use bold labels," but we absolutely created the conditions that make the model choose that format.

## The strongest evidence from the prompt files

From `PROMPT_7_FRESH_CONTEXT_CROSS_FIELD_BREAKTHROUGH.md`:

- `Return exactly the requested sections, in the requested order.`
- `Keep the answer compact, evidence-backed, and prototype-oriented.`
- `Prefer concise, information-dense writing.`
- `Return only 1-3 serious candidates.`
- `For each candidate give:` followed by a field list

From `PROMPT_8_FRESH_CONTEXT_INVENT_NEW_TECHNIQUES.md`:

- `Return exactly the requested sections, in the requested order.`
- `Keep the final answer compact, evidence-backed, and prototype-aware.`
- `Prefer concise, information-dense writing.`
- `Return only 1-3 serious techniques.`
- `For each one, provide:` followed by a field list

That is basically a template for "structured research brief" output instead of "verbose essay" output.

## Why the model chose bold labels specifically

This part is probably a mix of prompt pressure plus GPT-5.4 default behavior.

Our local prompt notes in `prompting.md` say:

- `GPT-5.4 often defaults to more structured formatting and may overuse bullet lists.`

And the Hydra prompt style guide says:

- `explicit output contracts improve reliability`
- `concise, information-dense output works better than vague verbosity`

So once we told the model:

1. be compact
2. be exact about section order
3. cover a fixed field list for each candidate
4. avoid fluff

the easiest stable markdown shape became:

- section heading
- bold field label
- short explanatory paragraph

That is a normal model response to this kind of contract-heavy prompt.

## Why it feels less verbose than older runs

Because we deliberately optimized these later prompts for density and control.

Compared with earlier prompt generations, these fresh-context prompts added stronger pressure around:

- compactness
- exact deliverable shape
- rejection of scene-setting
- only keeping 1-3 serious ideas
- answering like something a coding agent could prototype

That naturally compresses the prose.

In plain English: we traded some "essay voice" for tighter engineering-brief voice.

## Did we explicitly request bold formatting?

No, not explicitly.

I do not see a line in `PROMPT_7` or `PROMPT_8` that says to use bold labels.

But we did explicitly request the conditions that usually produce it:

- rigid section structure
- concise output
- dense information packing
- markdown-friendly deliverables

So the answer is:

- **Did we directly command bold labels?** No.
- **Did we strongly steer the model into that style?** Yes.

## If you want the old more verbose style back

Then the prompts should stop over-constraining brevity.

The biggest lines causing compression are the ones like:

- `Keep the answer compact...`
- `Prefer concise, information-dense writing.`
- `Return exactly the requested sections...`

If we want more expansive outputs next time, the prompt should instead say something more like:

- explain each candidate in full prose, not label-style fragments
- use multi-paragraph reasoning per candidate
- include a longer comparative discussion before the final ranking
- do not optimize for compactness if detail would help implementation

## Bottom line

The weird part is not that GPT-5.4 got randomly shorter.

The real reason is that our prompt stack for `PROMPT_7` and `PROMPT_8` was tuned to produce disciplined, compact, section-locked engineering briefs. GPT-5.4 then rendered that in a common markdown pattern: bold labels followed by short paragraphs.

So yes — this output style is mostly downstream of our own instructions.
