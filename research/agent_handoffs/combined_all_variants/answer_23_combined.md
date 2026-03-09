<combined_run_record run_id="answer_23" variant_id="prompt_config_answer_compact" schema_version="1">
  <metadata>
    <notes>Compact self-contained combined record for Agent 23 delta-q closure research. It preserves the prompt shell, the full generator config, a compact artifact manifest with real file/range references, and the literal answer inline without embedding the full multi-thousand-line rendered prompt packet.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_config_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — delta_q closure blueprint packet

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for honest `delta_q_target` closure in Hydra.

We want a detailed answer that makes clear:
- what the canonical target object should be in current doctrine
- whether Hydra can honestly close `delta_q_target` now or whether it must stay off
- if a narrow surviving path exists, the exact producer, mask, provenance, and validation contract
- how to resolve the discard-level `[34]` runtime object against the dense `[46]` train head
- what must stay deferred, rejected, or explicitly blocked

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (rust or python)
- include worked examples when helpful
- distinguish direct artifact support from your own inference
- prefer the narrowest honest blueprint over a broader shaky one
- if the correct answer is blocked, say `keep-off-blocked` plainly and list the exact unmet prerequisites
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now.
They are not guaranteed to be fully correct.
Treat them as evidence to inspect and critique, not truth to inherit.
Some are likely stale, partial, inconsistent, or semantically wrong.
</artifact_note>

<required_output>
Your answer must contain these sections:
1. Verdict — one of `implement-now`, `implement-narrowly`, `keep-off-blocked`
2. Canonical target object
3. Provenance contract
4. Shape + mask contract
5. Producer blueprint
6. Loss blueprint
7. Validation blueprint
8. Do not do
</required_output>

<artifacts_manifest>

## Artifact 01 — Hydra prompt doctrine
Artifact id: `style-guide-shell`
Source label: STYLE
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:20-160`
Why it matters: Generator should follow Hydra's artifact-first prompt doctrine instead of inventing a loose memo prompt.

## Artifact 02 — Reference narrow artifact-first prompt
Artifact id: `narrow-reference`
Source label: REFNARROW
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/reference_prompt_example_001_narrow_focused.md`
Why it matters: Concrete reference for how a narrow blueprint prompt should feel.

## Artifact 03 — Current reconciliation doctrine for advanced target closure
Artifact id: `reconciliation-delta-q`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:424-474`
Why it matters: This is the active-path doctrine for what should be closed next and what should stay absent.

## Artifact 04 — Archive roadmap dependency table for delta_q
Artifact id: `roadmap-delta-q`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:155-235`
Why it matters: This spells out the archive's current blocker language and execution order around delta_q.

## Artifact 05 — Canonical claims JSONL
Artifact id: `claims-jsonl-full`
Source label: CLAIMSJSONL
Type: `file_full`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
Why it matters: Raw canonical-claims artifact. Use it to cross-check the markdown condensation and preserve exact rows.

## Artifact 06 — Canonical claims markdown table
Artifact id: `claims-md-full`
Source label: CLAIMSMD
Type: `file_full`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md`
Why it matters: Human-readable version of the validated archive claims. Includes the typed-hole and keep-off rows for delta_q.

## Artifact 07 — Combined answer 18 full artifact
Artifact id: `answer-18-full`
Source label: A18
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_18_combined.md`
Why it matters: Contains the strongest keep-off / typed-hole argument for delta_q in its current dense form.

## Artifact 08 — Combined answer 11 full artifact
Artifact id: `answer-11-full`
Source label: A11
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_11_combined.md`
Why it matters: Contains the 34-vs-46 action-space realism critique and the missing-mask argument.

## Artifact 09 — Combined answer 15 full artifact
Artifact id: `answer-15-full`
Source label: A15
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md`
Why it matters: Contains the provenance table and the strict keep-absent verdicts across target families.

## Artifact 10 — Combined answer 2 full artifact
Artifact id: `answer-2-full`
Source label: A2
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_2_combined.md`
Why it matters: Contains one candidate delta_q formula and adjacent search-distillation proposals that may or may not survive stricter doctrine.

## Artifact 11 — Combined answer 2-1 full artifact
Artifact id: `answer-2-1-full`
Source label: A21
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
Why it matters: Contains narrowed later-target language, support gating hints, and related ExIt/delta_q validation ideas.

## Artifact 12 — Combined answer 7 full artifact
Artifact id: `answer-7-full`
Source label: A7
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_7_combined.md`
Why it matters: Earlier predecessor support for safe search-distillation and delta_q target closure ideas.

## Artifact 13 — Combined answer 7-1 full artifact
Artifact id: `answer-7-1-full`
Source label: A71
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_7-1_combined.md`
Why it matters: Contains model-surface and target-closure observations around advanced heads including delta_q.

## Artifact 14 — Combined answer 16 full artifact
Artifact id: `answer-16-full`
Source label: A16
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_16_combined.md`
Why it matters: Contains trust/provenance concerns for search-derived exports including delta_q.

## Artifact 15 — Combined answer 16-1 full artifact
Artifact id: `answer-16-1-full`
Source label: A161
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_16-1_combined.md`
Why it matters: Contains stricter revalidation on search-derived trust, hard-state gating, and export discipline.

## Artifact 16 — Combined answer 14 full artifact
Artifact id: `answer-14-full`
Source label: A14
Type: `file_full`
Source: `research/agent_handoffs/combined_all_variants/answer_14_combined.md`
Why it matters: Contains discard-centric runtime-surface arguments relevant to whether delta_q can be a full 46-action object today.

## Artifact 17 — Bridge-side delta_q export
Artifact id: `bridge-delta-q`
Source label: BRIDGE
Type: `file_range`
Source: `hydra-core/src/bridge.rs:330-369`
Why it matters: This is the live runtime object currently called delta_q on the bridge side.

## Artifact 18 — Loss helpers relevant to delta_q
Artifact id: `losses-delta-q`
Source label: LOSSES
Type: `file_range`
Source: `hydra-train/src/training/losses.rs:240-262`
Why it matters: Shows the current dense MSE helper and the existing masked-action pattern used elsewhere.

## Artifact 19 — Sample-to-target delta_q absence
Artifact id: `sample-delta-q`
Source label: SAMPLE
Type: `file_range`
Source: `hydra-train/src/data/sample.rs:165-219`
Why it matters: Shows that the current batch path still hardcodes delta_q_target to None.

## Artifact 20 — Model output surface including delta_q
Artifact id: `model-delta-q`
Source label: MODEL1
Type: `file_range`
Source: `hydra-train/src/model.rs:18-30`
Why it matters: Confirms the dense [B,46] model output surface.

## Artifact 21 — Encoder search features struct including delta_q plane
Artifact id: `encoder-delta-q`
Source label: ENC
Type: `file_range`
Source: `hydra-core/src/encoder.rs:120-150`
Why it matters: Confirms the runtime encoder carries a 34-cell discard plane for delta_q, not a 46-action tensor.

## Artifact 22 — AFBS q-value semantics
Artifact id: `afbs-q-values`
Source label: AFBS
Type: `file_range`
Source: `hydra-core/src/afbs.rs:120-185`
Why it matters: Grounds what node q-value means in the actual search tree code.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <config_section>
  <config_text status="preserved" source_path="embedded_generator_config">
  <![CDATA[{
  "version": 1,
  "repo_root": ".",
  "defaults": {
    "title": "Hydra prompt — delta_q closure blueprint packet",
    "artifact_container_tag": "artifacts",
    "artifact_ids": [
      "style-guide-shell",
      "narrow-reference",
      "reconciliation-delta-q",
      "roadmap-delta-q",
      "claims-jsonl-full",
      "claims-md-full",
      "answer-18-full",
      "answer-11-full",
      "answer-15-full",
      "answer-2-full",
      "answer-2-1-full",
      "answer-7-full",
      "answer-7-1-full",
      "answer-16-full",
      "answer-16-1-full",
      "answer-14-full",
      "bridge-delta-q",
      "losses-delta-q",
      "sample-delta-q",
      "model-delta-q",
      "encoder-delta-q",
      "afbs-q-values"
    ],
    "shell_sections": [
      {
        "tag": "role",
        "lines": [
          "Produce an implementation-ready blueprint.",
          "Do not give a memo.",
          "Your answer itself must be the blueprint."
        ]
      },
      {
        "tag": "direction",
        "lines": [
          "Work toward the strongest exact blueprint for honest `delta_q_target` closure in Hydra.",
          "",
          "We want a detailed answer that makes clear:",
          "- what the canonical target object should be in current doctrine",
          "- whether Hydra can honestly close `delta_q_target` now or whether it must stay off",
          "- if a narrow surviving path exists, the exact producer, mask, provenance, and validation contract",
          "- how to resolve the discard-level `[34]` runtime object against the dense `[46]` train head",
          "- what must stay deferred, rejected, or explicitly blocked",
          "",
          "Use the artifacts below to derive your conclusions."
        ]
      },
      {
        "tag": "style",
        "lines": [
          "- no high-level survey",
          "- no vague answer",
          "- include reasoning",
          "- include formulas when needed",
          "- include code-like detail when helpful (rust or python)",
          "- include worked examples when helpful",
          "- distinguish direct artifact support from your own inference",
          "- prefer the narrowest honest blueprint over a broader shaky one",
          "- if the correct answer is blocked, say `keep-off-blocked` plainly and list the exact unmet prerequisites",
          "- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked"
        ]
      },
      {
        "tag": "artifact_note",
        "lines": [
          "The artifacts below reflect what the current codebase/docs appear to say right now.",
          "They are not guaranteed to be fully correct.",
          "Treat them as evidence to inspect and critique, not truth to inherit.",
          "Some are likely stale, partial, inconsistent, or semantically wrong."
        ]
      },
      {
        "tag": "required_output",
        "lines": [
          "Your answer must contain these sections:",
          "1. Verdict — one of `implement-now`, `implement-narrowly`, `keep-off-blocked`",
          "2. Canonical target object",
          "3. Provenance contract",
          "4. Shape + mask contract",
          "5. Producer blueprint",
          "6. Loss blueprint",
          "7. Validation blueprint",
          "8. Do not do"
        ]
      }
    ]
  },
  "artifacts": [
    {
      "id": "style-guide-shell",
      "type": "file_range",
      "path": "research/agent_handoffs/PROMPT_STYLE_GUIDE.md",
      "start_line": 20,
      "end_line": 160,
      "label": "Hydra prompt doctrine",
      "explanation": "Generator should follow Hydra's artifact-first prompt doctrine instead of inventing a loose memo prompt.",
      "source_label": "STYLE",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "narrow-reference",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/reference_prompt_example_001_narrow_focused.md",
      "label": "Reference narrow artifact-first prompt",
      "explanation": "Concrete reference for how a narrow blueprint prompt should feel.",
      "source_label": "REFNARROW",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "reconciliation-delta-q",
      "type": "file_range",
      "path": "research/design/HYDRA_RECONCILIATION.md",
      "start_line": 424,
      "end_line": 474,
      "label": "Current reconciliation doctrine for advanced target closure",
      "explanation": "This is the active-path doctrine for what should be closed next and what should stay absent.",
      "source_label": "RECON",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "roadmap-delta-q",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md",
      "start_line": 155,
      "end_line": 235,
      "label": "Archive roadmap dependency table for delta_q",
      "explanation": "This spells out the archive's current blocker language and execution order around delta_q.",
      "source_label": "ROADMAP",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "claims-jsonl-full",
      "type": "file_full",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl",
      "label": "Canonical claims JSONL",
      "explanation": "Raw canonical-claims artifact. Use it to cross-check the markdown condensation and preserve exact rows.",
      "source_label": "CLAIMSJSONL",
      "fence_language": "json",
      "show_line_numbers": true
    },
    {
      "id": "claims-md-full",
      "type": "file_full",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md",
      "label": "Canonical claims markdown table",
      "explanation": "Human-readable version of the validated archive claims. Includes the typed-hole and keep-off rows for delta_q.",
      "source_label": "CLAIMSMD",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-18-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_18_combined.md",
      "label": "Combined answer 18 full artifact",
      "explanation": "Contains the strongest keep-off / typed-hole argument for delta_q in its current dense form.",
      "source_label": "A18",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-11-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_11_combined.md",
      "label": "Combined answer 11 full artifact",
      "explanation": "Contains the 34-vs-46 action-space realism critique and the missing-mask argument.",
      "source_label": "A11",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-15-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_15_combined.md",
      "label": "Combined answer 15 full artifact",
      "explanation": "Contains the provenance table and the strict keep-absent verdicts across target families.",
      "source_label": "A15",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-2-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_2_combined.md",
      "label": "Combined answer 2 full artifact",
      "explanation": "Contains one candidate delta_q formula and adjacent search-distillation proposals that may or may not survive stricter doctrine.",
      "source_label": "A2",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-2-1-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_2-1_combined.md",
      "label": "Combined answer 2-1 full artifact",
      "explanation": "Contains narrowed later-target language, support gating hints, and related ExIt/delta_q validation ideas.",
      "source_label": "A21",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-7-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_7_combined.md",
      "label": "Combined answer 7 full artifact",
      "explanation": "Earlier predecessor support for safe search-distillation and delta_q target closure ideas.",
      "source_label": "A7",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-7-1-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_7-1_combined.md",
      "label": "Combined answer 7-1 full artifact",
      "explanation": "Contains model-surface and target-closure observations around advanced heads including delta_q.",
      "source_label": "A71",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-16-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_16_combined.md",
      "label": "Combined answer 16 full artifact",
      "explanation": "Contains trust/provenance concerns for search-derived exports including delta_q.",
      "source_label": "A16",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-16-1-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_16-1_combined.md",
      "label": "Combined answer 16-1 full artifact",
      "explanation": "Contains stricter revalidation on search-derived trust, hard-state gating, and export discipline.",
      "source_label": "A161",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-14-full",
      "type": "file_full",
      "path": "research/agent_handoffs/combined_all_variants/answer_14_combined.md",
      "label": "Combined answer 14 full artifact",
      "explanation": "Contains discard-centric runtime-surface arguments relevant to whether delta_q can be a full 46-action object today.",
      "source_label": "A14",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "bridge-delta-q",
      "type": "file_range",
      "path": "hydra-core/src/bridge.rs",
      "start_line": 330,
      "end_line": 369,
      "label": "Bridge-side delta_q export",
      "explanation": "This is the live runtime object currently called delta_q on the bridge side.",
      "source_label": "BRIDGE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "losses-delta-q",
      "type": "file_range",
      "path": "hydra-train/src/training/losses.rs",
      "start_line": 240,
      "end_line": 262,
      "label": "Loss helpers relevant to delta_q",
      "explanation": "Shows the current dense MSE helper and the existing masked-action pattern used elsewhere.",
      "source_label": "LOSSES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "sample-delta-q",
      "type": "file_range",
      "path": "hydra-train/src/data/sample.rs",
      "start_line": 165,
      "end_line": 219,
      "label": "Sample-to-target delta_q absence",
      "explanation": "Shows that the current batch path still hardcodes delta_q_target to None.",
      "source_label": "SAMPLE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "model-delta-q",
      "type": "file_range",
      "path": "hydra-train/src/model.rs",
      "start_line": 18,
      "end_line": 30,
      "label": "Model output surface including delta_q",
      "explanation": "Confirms the dense [B,46] model output surface.",
      "source_label": "MODEL1",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "encoder-delta-q",
      "type": "file_range",
      "path": "hydra-core/src/encoder.rs",
      "start_line": 120,
      "end_line": 150,
      "label": "Encoder search features struct including delta_q plane",
      "explanation": "Confirms the runtime encoder carries a 34-cell discard plane for delta_q, not a 46-action tensor.",
      "source_label": "ENC",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "afbs-q-values",
      "type": "file_range",
      "path": "hydra-core/src/afbs.rs",
      "start_line": 120,
      "end_line": 185,
      "label": "AFBS q-value semantics",
      "explanation": "Grounds what node q-value means in the actual search tree code.",
      "source_label": "AFBS",
      "fence_language": "rust",
      "show_line_numbers": true
    }
  ],
  "variants": [
    {
      "name": "delta-q-closure",
      "title": "Hydra prompt — delta_q closure blueprint packet",
      "output_file": "delta_q_research_prompt.md",
      "artifact_ids": [],
      "shell_sections": []
    }
  ]
}
]]>
  </config_text>
  </config_section>

  <answer_section>
  <answer_text status="preserved" source_path="embedded_agent_answer">
  <![CDATA[## 1. Verdict

`keep-off-blocked`

**Artifact-supported**

The current repo/doctrine does **not** close `delta_q_target` today.

* The model head is dense `[B,46]` (`HydraOutput.delta_q`) while the live runtime/search object currently called `delta_q` is a discard/tile plane `[34]` in `SearchFeaturePlanes`, populated only for `0..NUM_TILE_TYPES` as `tree.node_q_value(child) - root_q` (`MODEL1 L0018-L0023`, `ENC L0124-L0130`, `BRIDGE L0343-L0356`).
* The normal batch path still hardcodes `delta_q_target: None` (`SAMPLE L0180-L0181`, `SAMPLE L0212-L0213`).
* The current loss helper for `delta_q` is dense unmasked MSE over the whole tensor; there is no `delta_q_mask` path yet (`LOSSES L0247-L0262`).
* Current doctrine/roadmap explicitly says this lane is only **partial** and that the blocker is exactly the mismatch between dense `[46]` training surface and discard-only `[34]` runtime surface; earliest honest promotion is “after object + mask + builder closure” (`ROADMAP L0177`, `CLAIMSMD L0033`, `RECON L0428-L0447`).

**Blueprint decision**

Hydra must keep `delta_q_target` **off** in current mainline training.

A **narrow future lane does survive**, but it is not honest to call it closed now. The surviving lane is:

* search-derived
* learner-only
* root-only
* discard-compatible only
* `[46]` head-space with an explicit `[46]` mask
* default-off until the producer, carrier, and validation matrix are actually closed

**Exact unmet prerequisites**

1. A real upstream producer in the same family as the current live ExIt lane, including the still-missing concrete root-search adapter / child-observation reconstruction seam (`CLAIMSJSONL L0041-L0043`).
2. End-to-end `delta_q_mask` plumbing.
3. A masked `delta_q` loss path; the current dense unmasked helper is the wrong contract.
4. A validation matrix tying the produced labels to learner-only root search quality before enablement.
5. Head-gate approval for a sparse search-derived head.

---

## 2. Canonical target object

**Artifact-supported**

There is **no canonically live `delta_q_target` today**. The only current live `delta_q` family semantics in repo code are the bridge/runtime semantics:

```text
delta_q_runtime[tile] = Q(child(tile)) - Q(root)
```

for `tile in 0..NUM_TILE_TYPES`, where `Q(node) = total_value / visit_count` from AFBS (`BRIDGE L0343-L0356`, `AFBS L0125-L0137`).

Earlier broader formulas such as policy-centered or clipped `ΔQ` appear in older archive proposals (`A2`, `A21`), but the stricter canonical ledger does **not** preserve those as current doctrine. What survives canonically is only that the current dense `[46]` path is **not closed**.

**Blueprint decision**

The canonical future object for current doctrine should be:

```text
delta_q_target_v1_discard_masked_root_qdelta
```

with semantics:

```text
For state s, root r, supported action a:

Q_s(n) = total_value(n) / visit_count(n),   visit_count(n) > 0

delta_q*_v1(s, a) = Q_s(child_a) - Q_s(root)     if a ∈ M_v1(s)
delta_q*_v1(s, a) = undefined                    if a ∉ M_v1(s)
```

where `M_v1(s)` is an explicit action mask defined in section 4.

That is the right v1 object because it does the minimum necessary to unify the named blocker:

* it matches the **actual live runtime family semantics** already called `delta_q`
* it preserves the existing `[46]` model head
* it does **not** invent dense labels for unsupported actions
* it does **not** silently redefine `delta_q` into a different centered/softmaxed object

**What this means operationally**

* **Live canonical object today:** absent.
* **Future canonical closure object:** masked root-child q-delta over the existing `[46]` action head.

**Why not use a centered / clipped / softmax-like object in v1**

That would be a **new** target family, not a closure of the existing one. If Hydra later wants:

```text
delta_q_v2_centered(s,a) = Q(child_a) - E_{pi_base}[Q(child)]
```

or a clipped/normalized version, that should be a **version bump**, not a silent reinterpretation of `delta_q`.

**Reserve fallback only**

If even discard-masked v1 cannot be produced honestly, the only fallback that survives is a separately named object such as:

```text
delta_q_taken_only
```

and it must **not** be presented as closing `delta_q_target`.

---

## 3. Provenance contract

**Artifact-supported**

Current doctrine requires advanced targets to carry explicit provenance such as replay-derived, bridge-derived, or search-derived, and to stay absent when the producer is not valid (`RECON L0425-L0430`, `RECON L0441-L0447`).

The current narrow live ExIt lane already tightened the search-label doctrine to:

* learner-only
* root-only
* default-off until validation
* emit `None` on failed gates (`CLAIMSMD L0061`, `CLAIMSJSONL L0041-L0043`)

The runtime provenance hardening tranche is also already recorded as completed:

* `source_net_hash`
* `source_version`
* `trust_level`
* `cache_namespace`
* `generation` (`CLAIMSJSONL L0005`, `CLAIMSMD L0024`)

**Blueprint decision**

`delta_q_target_v1` must use this provenance contract:

```rust
pub enum DeltaQTargetVersion {
    RootChildQDeltaDiscardMaskedV1 = 1,
}

pub struct SearchLabelMeta {
    pub head: &'static str,          // "delta_q"
    pub version: u16,                // 1
    pub provenance: ProvenanceKind,  // SearchDerived
    pub source_net_hash: u64,
    pub source_version: u64,
    pub trust_level: TrustLevel,     // LearnerOnly
    pub cache_namespace: CacheNamespace, // LearnerTarget if cached
    pub generation: u64,
}
```

**Required provenance rules**

1. **Provenance class** = `SearchDerived`.
2. **Search anchor** = learner-only. No rollout-anchored labels.
3. **Scope** = root-only. No node-level / broad-tree export.
4. **Action family** = discard-compatible only in v1.
5. **Failure policy** = emit `None`, never weak or fabricated labels.
6. **Cache rule**:

   * acceptable only if `trust_level == LearnerOnly`
   * `source_net_hash`, `source_version`, and `generation` match current learner state
   * if cached for label reuse, use `CacheNamespace::LearnerTarget`
7. **Runtime bridge plane rule**:

   * the `[34]` encoder plane is a **bridge projection of search**
   * it is **not** itself the training-label provenance source

**Rejected provenance shortcuts**

* replay-derived fake `delta_q`
* bridge-plane-as-label
* rollout-generated label without learner-only anchoring
* unknown-provenance cached search result

---

## 4. Shape + mask contract

**Artifact-supported**

* Model output: `delta_q: Tensor<B,2>` with shape `[B,46]` (`MODEL1 L0018-L0023`).
* Runtime encoder search feature: `delta_q: [f32; NUM_TILES]` i.e. `[34]` (`ENC L0124-L0130`).
* Current loss path has no `delta_q_mask` but already has the correct masked-action pattern for `safety_residual` (`LOSSES L0252-L0262`).

**Blueprint decision**

Keep the head shape `[B,46]`. Add an explicit action mask of the same shape.

```rust
pub struct HydraTargets<B: Backend> {
    // existing fields...
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
}
```

At the sample / trajectory label boundary, use a structured masked object so target and mask cannot drift apart:

```rust
pub struct MaskedActionTarget<const A: usize> {
    pub values: [f32; A],
    pub mask: [f32; A],
    pub meta: SearchLabelMeta,
}
```

### Exact v1 support rule

The only grounded action-ID support rule from the current artifacts is the bridge rule:

```rust
pub fn delta_q_v1_slot(raw_action: u8) -> Option<usize> {
    if raw_action < NUM_TILE_TYPES as u8 {
        Some(raw_action as usize)
    } else {
        None
    }
}
```

So v1 may light up only those canonical discard/tile slots directly represented by current AFBS/bridge semantics.

### Per-sample mask definition

For sample `i`, action `a`:

```text
delta_q_mask[i,a] = 1  iff
    a is a v1-supported canonical discard action
    and legal_mask[i,a] > 0
    and child(a) exists under the produced AFBS root
    and child_visit_count(a) >= min_child_visits
    and Q(root), Q(child(a)) are finite

delta_q_mask[i,a] = 0 otherwise
```

### Presence invariant

```text
delta_q_target.is_some() == delta_q_mask.is_some()
```

and per sample:

```text
sample i is present iff sum_a delta_q_mask[i,a] > 0
```

### How this resolves `[34]` vs `[46]`

This is the exact resolution:

* `[34]` stays the **runtime discard/tile projection**
* `[46]` stays the **training/model head space**
* closure is obtained by making the training object a **masked `[46]` tensor** whose support is the canonical discard subset represented by the current runtime/search family

So the relation is:

```text
runtime [34] delta_q  =  projection of the v1 masked [46] family onto canonical discard tile slots
training [46] delta_q =  masked action-space target, not dense all-action supervision
```

**Do not reverse this direction.** Never derive the `[46]` target by naively expanding the `[34]` plane.

### Consequence for non-discard / extra action IDs

In v1:

* all non-discard action IDs stay mask `0`
* any extra action IDs not directly represented by the current AFBS/bridge family stay mask `0`

That includes any action aliases not explicitly closed by current search semantics.

---

## 5. Producer blueprint

**Artifact-supported**

The current narrow live ExIt lane already defines the right envelope:

* root-only AFBS
* learner-only value-head evaluator
* all-legal discard seeding
* default-off until validation
* `None` on failed gates (`CLAIMSMD L0025`, `CLAIMSMD L0061`, `CLAIMSJSONL L0043`)

The current `delta_q` bridge family already gives the label algebra to extract:

```text
Q(child) - Q(root)
```

(`BRIDGE L0343-L0356`).

The current replay loader/sample path is not the right source; it still hardcodes `delta_q_target: None` (`SAMPLE L0180-L0181`, `SAMPLE L0212-L0213`).

**Blueprint decision**

Do **not** create a replay-loader builder for `delta_q`.
Do **not** run a second duplicate search.

Instead, extend the same root-search label envelope used for live ExIt into a shared root-label producer.

### File-level plan

1. Extend `hydra-train/src/training/live_exit.rs` into a shared root-label producer, or add sibling `live_delta_q.rs` with shared search orchestration.
2. Extend `hydra-train/src/selfplay.rs` so one root search can emit both `exit_target` and `delta_q_target`.
3. Add collation in `hydra-train/src/training/rl.rs` for `delta_q_target` + `delta_q_mask`.
4. Add carriers in `hydra-train/src/data/sample.rs` / batch structs only where search-derived targets actually flow.
5. Leave `mjai_loader.rs` replay path at `None` for `delta_q` in this tranche.

### Exact producer types

```rust
pub struct LiveDeltaQConfig {
    pub enabled: bool,            // default false
    pub min_root_visits: u32,     // wire to existing search-label min_visits gate when enabled
    pub min_child_visits: u32,    // default 1 in v1
}

pub struct RootSearchLabels {
    pub exit: Option<MaskedActionTarget<46>>,   // policy-style target can reuse different wrapper later
    pub delta_q: Option<MaskedActionTarget<46>>,
}
```

### Exact producer algorithm

```rust
pub fn build_delta_q_from_afbs_tree(
    tree: &AfbsTree,
    root: NodeIdx,
    legal_mask: &[bool; 46],
    cfg: &LiveDeltaQConfig,
    meta: SearchLabelMeta,
) -> Option<MaskedActionTarget<46>> {
    let root_visits = tree.nodes[root as usize].visit_count;
    if root_visits < cfg.min_root_visits {
        return None;
    }

    let root_q = tree.node_q_value(root);
    if !root_q.is_finite() {
        return None;
    }

    let mut values = [0.0f32; 46];
    let mut mask = [0.0f32; 46];

    for raw_action in 0..NUM_TILE_TYPES as u8 {
        let idx = raw_action as usize;
        if !legal_mask[idx] {
            continue;
        }

        let Some(child) = tree.find_child_by_action(root, raw_action) else {
            continue;
        };

        let child_visits = tree.nodes[child as usize].visit_count;
        if child_visits < cfg.min_child_visits {
            continue;
        }

        let child_q = tree.node_q_value(child);
        if !child_q.is_finite() {
            continue;
        }

        values[idx] = child_q - root_q;
        mask[idx] = 1.0;
    }

    if mask.iter().all(|&m| m == 0.0) {
        return None;
    }

    Some(MaskedActionTarget { values, mask, meta })
}
```

### Exact shared search envelope

`delta_q_target_v1` must use the same root-search envelope as narrow ExIt:

1. compatible discard-only state
2. root-only learner-only AFBS
3. all-legal discard seeding at the root
4. finite root/child q values
5. explicit support mask
6. emit `None` on any gate failure

### Why all-legal discard seeding is required

Current AFBS expansion keeps `TOP_K = 5` children in ordinary expansion (`A161 L0462-L0463`).
That is fine for ordinary search, but **not** fine for label support if Hydra is pretending to supervise a broader discard surface.

So the producer must reuse the ExIt fix:

* bypass root `TOP_K` truncation
* seed **all legal discard children**
* then apply the explicit support mask using visit counts / finite q

Without that, the support mask is partly a hidden artifact of top-k truncation rather than a real target contract.

### Carrier path

Use one shared root-search label carrier, not two independent ones.

```rust
pub struct TrajectorySearchLabel {
    pub exit: Option<MaskedActionTarget<46>>,
    pub delta_q: Option<MaskedActionTarget<46>>,
}
```

Then collate into `HydraTargets` in the RL/self-play batch path. Mixed batches must become:

* target/mask tensors for present rows
* zero-masked rows for absent rows

### Important narrowness rule

`delta_q_target_v1` must **not** move ahead of `exit_target` carrier closure.

It should piggyback on the same:

* adapter
* root-search call
* provenance
* validation envelope

If Hydra cannot close that shared root-label seam, `delta_q_target` stays absent.

---

## 6. Loss blueprint

**Artifact-supported**

* Current `delta_q` loss path is dense unmasked MSE (`LOSSES L0247-L0250`).
* Current masked action loss pattern already exists (`LOSSES L0252-L0262`).
* Doctrine says target presence should control whether an advanced loss exists at all (`RECON L0441-L0447`).

**Blueprint decision**

Replace dense unmasked `delta_q` regression with action-masked regression.

### Exact loss

For prediction `Δ̂q[i,a]`, target `Δq*[i,a]`, and mask `m[i,a]`:

```text
L_delta_q =
    1 / max(1, sum_{i,a} m[i,a])
    * sum_{i,a} m[i,a] * 1/2 * (Δ̂q[i,a] - Δq*[i,a])^2
```

This is exactly the existing masked-action MSE pattern already used elsewhere.

### Exact code change

```rust
let l_delta_q = match (&targets.delta_q_target, &targets.delta_q_mask) {
    (Some(target), Some(mask)) => masked_action_mse(
        outputs.delta_q.clone(),
        target.clone(),
        mask.clone(),
    ),
    _ => zero.clone(),
};
```

### Required target struct change

```rust
pub struct HydraTargets<B: Backend> {
    // ...
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
}
```

### Missing-target behavior

* `delta_q_target = None` and `delta_q_mask = None` -> zero delta-q loss
* target without mask -> batch validation failure; do not silently train
* mask all zero -> zero delta-q loss, counted as absent support

### Activation rule

* Keep default `w_delta_q = 0.0`
* do not activate by weight alone
* only activate after:

  1. producer exists
  2. validation matrix passes
  3. sparse-head gate controller approves the head

### Breakdown logging

Make missing-target behavior obvious. Add at least:

```text
delta_q_examples_present
delta_q_actions_present
delta_q_examples_absent
delta_q_loss
```

---

## 7. Validation blueprint

**Artifact-supported**

* BC and RL should have explicit mixed-case tests for advanced targets and obvious shape failures (`RECON L0454-L0468`).
* Advanced head activation discipline is already implemented:

  * sparse search heads require sparse-density gate
  * gradient conflict gate
  * warmup protocol (`CLAIMSJSONL L0004`, `CLAIMSMD L0023`)
* The only exact hard-state threshold that survives stricter validation is `top2_policy_gap < 0.10` (`A161 L0389-L0415`, `A161 L0508-L0542`).
* Current global benchmark gates already exist:
  `afbs_on_turn_ms < 150`, `ct_smc_dp_ms < 1`, `endgame_ms < 100`, `self_play_games_per_sec > 20`, `distill_kl_drift < 0.1` (`A161 L0544-L0587`).

**Blueprint decision**

Validation has three layers: object correctness, carrier correctness, and activation correctness.

### A. Unit tests: object and mask correctness

Add these tests.

1. `delta_q_v1_matches_afbs_root_child_delta`

   * build a tiny AFBS tree
   * set root/child `total_value` and `visit_count`
   * assert `target[a] == q(child_a) - q(root)`

2. `delta_q_v1_masks_only_supported_canonical_discard_slots`

   * assert only `0..NUM_TILE_TYPES` may be masked in v1
   * assert all remaining action IDs are mask `0`

3. `delta_q_v1_omits_unvisited_children`

   * child exists but `visit_count == 0`
   * assert mask `0`

4. `delta_q_v1_returns_none_when_no_valid_support`

   * no supported child survives
   * assert `None`

5. `delta_q_v1_rejects_nonfinite_q`

   * nonfinite root or child q -> `None`

### B. Batch / loss tests

6. `delta_q_roundtrips_through_collation`

   * present row stays present
   * absent row becomes zero-masked row

7. `delta_q_requires_mask`

   * target without mask must fail batch validation

8. `delta_q_masked_loss_ignores_unmasked_positions`

   * perturb masked-out positions
   * loss unchanged

9. `delta_q_all_zero_mask_zeroes_loss`

   * loss is zero when mask sum is zero

### C. Integration tests

10. `rl_batch_mixed_cases_handle_delta_q`

    * baseline only
    * baseline + delta_q
    * baseline + exit + delta_q

11. `bc_batch_mixed_cases_handle_delta_q`

    * same mixed cases for BC path

12. `shared_root_search_produces_exit_and_delta_q_from_same_tree`

    * same search run
    * same `source_net_hash` / `source_version` / `generation`
    * same legality basis

### D. Audit contract before enablement

For a bounded audit window, log at minimum:

```text
examples_present
actions_present
mean_abs
min
max
frac_zero
frac_positive
frac_negative
```

**Artifact-supported exact gates that survive**

* use the current global benchmark gates unchanged
* stratify at least the slice with `top2_policy_gap < 0.10`

**Recommended but not canonized exact numeric gate**

Older handoffs proposed `visits(a) >= 4` as a support threshold; stricter revalidation demoted that from canon. So:

* `min_child_visits = 1` is the narrow v1 object contract
* `min_child_visits = 4` may be tested as a stricter validation-tuned configuration, but is **not** the canonical object definition

### E. Sparse-head activation gate

Register `delta_q` with the existing advanced head controller as a sparse search head.

Do not activate until the controller approves it under the already-implemented sparse-head protocol:

* sparse-density gate passes (`spp_h >= 5.0` in the current implemented controller)
* gradient conflict gate passes
* warmup completes (`CLAIMSJSONL L0004`, `CLAIMSMD L0023`)

### F. Minimum label-fidelity benchmark

Because the current doctrine does **not** settle exact rollout-style promotion thresholds, use this minimum benchmark before enabling:

* compare the produced `delta_q_target_v1` against a **deeper learner-only AFBS reference** on the same compatible discard states
* at minimum, report:

  * masked-action sign agreement
  * best-discard agreement under masked `delta_q`
  * MAE on masked entries
  * coverage (`examples_present`, `actions_present`)

If that benchmark is not run, or if it fails, producer stays off.

### G. Enablement rule

`LiveDeltaQConfig.enabled` must default to `false`.

Flip it only after all of the following are true:

1. the shared exit/delta-q adapter exists
2. the unit + integration tests pass
3. the audit window shows nontrivial support
4. the sparse-head controller approves activation
5. the deeper learner-only comparison has been run on the `top2_policy_gap < 0.10` slice and reviewed

Until then, `delta_q_target` remains absent.

---

## 8. Do not do

1. **Do not** treat the current bridge/runtime `[34]` plane as if it already closed the training target.
2. **Do not** zero-extend or broadcast `[34] -> [46]` and call that supervision.
3. **Do not** keep the current dense unmasked `delta_q` regression path.
4. **Do not** emit `delta_q_target` without `delta_q_mask`.
5. **Do not** fabricate labels for non-discard / call / kan / riichi action slots in v1.
6. **Do not** let `delta_q_target` outrun `exit_target` carrier closure. Use one shared root-search producer.
7. **Do not** use replay-derived hidden-state hacks to fake `delta_q`.
8. **Do not** accept rollout-generated, unknown-provenance, or non-learner-anchored search outputs as v1 label sources.
9. **Do not** bless rejected exact thresholds such as `risk_score > 0.08` or `particle_ess < 0.55` as current canon; stricter revalidation explicitly demoted them (`A161 L0410-L0441`).
10. **Do not** smuggle in the chosen-action-only fallback under the same name. If Hydra ever uses that weaker fallback, rename it explicitly (`delta_q_taken_only`) and keep it separate from `delta_q_target_v1`.
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
