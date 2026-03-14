<combined_run_record run_id="answer_24" variant_id="prompt_config_answer_compact" schema_version="1">
  <metadata>
    <notes>Compact self-contained combined record for Agent 24 replay/sample ExIt producer-path research. It preserves the prompt shell reconstructed from the authoritative generator config, the full generator config inline, and the preserved Agent 24 answer.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_config_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — replay/sample ExIt producer-path research

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for the replay/sample-side ExIt producer-path question.

We want a detailed answer that makes clear:
- whether current authoritative docs plus current code already imply a buildable replay/sample-side ExIt producer path
- whether the strongest archive artifacts (`answer_15`, `answer_9`, `answer_22`, `answer_23`, canonical roadmap/jsonl rows) compose into a real actionable blueprint or still stop short
- whether the ordinary replay/sample loader should remain absent for `exit_target` / `exit_mask` in the current tranche
- whether the only honest surviving closure lane is a separate shared root-search producer / offline producer family rather than a replay-loader builder
- what exact provenance class the surviving path belongs to: replay-derived, bridge-derived, search-derived, or blocked
- what exact file-level plan survives after validating doctrine, archive evidence, and current code against each other
- what must stay narrow / deferred / rejected / absent
- if the path is still blocked, what the smallest decisive unblocker is

Use the artifacts below to derive your conclusions.

Your answer must end in one of two forms:
1. an implementation-ready blueprint for a doctrine-compatible replay/sample-side or shared-producer closure path, with exact file-level changes, invariants, provenance rules, and tests; or,
2. a blocked-verdict blueprint that proves why the normal replay/sample path must stay absent for now, names the smallest honest alternative producer family that survives, and states the smallest decisive research/implementation step needed to unblock it.
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
Authority order still wins: README.md -> research/design/HYDRA_FINAL.md -> research/design/HYDRA_RECONCILIATION.md -> docs/GAME_ENGINE.md. Archive and handoff artifacts are evidence only unless validated against that chain and current code.
</artifact_note>

<artifacts_manifest>

## Artifact 01 — Exact research question
Artifact id: `task-brief`
Type: `literal`
Why it matters: Pins the task to the real blocked seam so the answer does not drift into a broad ExIt survey.

## Artifact 02 — Scope and anti-drift rules
Artifact id: `scope-rules`
Type: `literal`
Why it matters: Keeps the prompt narrow and blocks the common failure modes that already bit this lane.

## Artifact 03 — Repo status summary
Artifact id: `readme-status`
Source label: README
Type: `file_range`
Source: `README.md:60-63`
Why it matters: Top-level authority says replay/sample exit_target production is still an immediate need, so the prompt has to explain whether that need is actually already solved somewhere else or still open.

## Artifact 04 — Hydra final doctrine for ExIt targets
Artifact id: `hydra-final-exit`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:257-280`
Why it matters: SSOT-level statement of the ExIt teacher object and the wider training-engine framing; it defines semantics, not necessarily the missing producer path.

## Artifact 05 — Current execution doctrine for target closure
Artifact id: `reconciliation-tranche`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:432-558`
Why it matters: This is the main active-path doctrine: it names the missing producer seam, warns against weak labels, and defines the minimal tranche acceptance checklist.

## Artifact 06 — Roadmap head-summary status note
Artifact id: `roadmap-head-status`
Source label: ROADMAP
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:178-199`
Why it matters: Subordinate reference showing which advanced targets already have normal supervised carriers and which do not.

## Artifact 07 — Roadmap ExIt pipeline status block
Artifact id: `roadmap-exit-status`
Source label: ROADMAP
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:757-783`
Why it matters: Useful because it truth-aligns the live self-play producer as implemented while still saying the normal replay/sample path does not emit exit_target.

## Artifact 08 — Archive triage map for ExIt and adjacent lanes
Artifact id: `archive-roadmap`
Source label: AR-ROAD
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:93-167`
Why it matters: This is the strongest archive summary of what got closed in the live lane versus what still remains open in the normal path. Evidence only, not doctrine.

## Artifact 09 — Canonical archive rows for live ExIt status and remaining gaps
Artifact id: `archive-jsonl-status`
Source label: AR-JSONL
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl:43-49`
Why it matters: Source-ledger version of the archive claims. Includes the live ExIt blueprint/status rows and the later truth-alignment note that replay/sample still does not emit exit_target.

## Artifact 10 — Answer 15 provenance taxonomy and public-teacher boundary
Artifact id: `answer-15-provenance`
Source label: A15
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md:150-225`
Why it matters: This is the best prior artifact for why ExIt/search-derived labels are later, why replay reconstructibility is not enough, and why provenance class matters.

## Artifact 11 — Answer 9 missing-closure audit
Artifact id: `answer-9-missing-closure`
Source label: A9
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_9_combined.md:1141-1206`
Why it matters: Earlier narrow audit of what was structurally missing or semantically unclosed; especially useful for the search-compatible state and the warning not to inherit root_exit_policy blindly.

## Artifact 12 — Answer 9 exact target objects and sample-carrier proposal
Artifact id: `answer-9-target-objects`
Source label: A9
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_9_combined.md:1240-1419`
Why it matters: This is the strongest old artifact for exact exit_target/delta_q target definitions and a proposed SearchSupervision carrier. It may be stale or superseded, so it must be critiqued against current doctrine and code.

## Artifact 13 — Answer 22 live AFBS ExIt evaluator blueprint
Artifact id: `answer-22-live-exit-blueprint`
Source label: A22
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_22_combined.md:227-339`
Why it matters: High-signal artifact for the live self-play lane: visit-based labels, value-head evaluator, all-legal seeding, and why root_exit_policy is not the teacher.

## Artifact 14 — Answer 23 shared root-search producer blueprint
Artifact id: `answer-23-shared-producer`
Source label: A23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:871-1029`
Why it matters: Closest archived file-level producer blueprint. Mostly written for delta_q piggybacking on ExIt, but directly relevant because it argues for one shared root-search label envelope rather than a replay-loader builder.

## Artifact 15 — Answer 7-1 conservative ExIt proposal
Artifact id: `answer-7-1-conservative-exit`
Source label: A7-1
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_7-1_combined.md:296-415`
Why it matters: Important predecessor artifact for a narrow, trust-gated conservative ExIt path. Useful because it argues from current AFBS shell and target closure without requiring broad new architecture.

## Artifact 16 — Answer 8-1 offline posterior-world builder proposal
Artifact id: `answer-8-1-offline-posterior-builder`
Source label: A8-1
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_8-1_combined.md:665-799`
Why it matters: This is the strongest explicit archive proposal for an offline replay label builder with added dataset fields. It is not doctrine, but it directly addresses the kind of path we might otherwise have to re-invent.

## Artifact 17 — Answer 12 stricter kill-pass on multi-world ExIt family
Artifact id: `answer-12-kill-family`
Source label: A12
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_12_combined.md:131-214`
Why it matters: High-signal counterweight artifact: it explicitly argues that offline posterior-consensus ExIt is not ready under current Hydra constraints because the public world-teacher action object is missing.

## Artifact 18 — Answer 5-1 offline label-generation hint
Artifact id: `answer-5-1-offline-label-hint`
Source label: A5-1
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_5-1_combined.md:285-326`
Why it matters: Not an ExIt producer blueprint directly, but a useful artifact for the general pattern of offline ladder generation and maybe_emit_exit_target_if_safety_valve_passes semantics.

## Artifact 19 — Current replay sample and batch structs
Artifact id: `sample-structs`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:46-159`
Why it matters: Shows what the ordinary supervised data path actually carries today. Important for checking whether a real replay/sample ExIt slot already exists or would need a new carrier.

## Artifact 20 — Current sample-to-HydraTargets conversion
Artifact id: `sample-to-targets`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:387-449`
Why it matters: Critical reality check: this is where optional advanced targets become HydraTargets in the ordinary supervised path, and it currently leaves delta_q absent and has no ExIt lane here.

## Artifact 21 — Current replay loader sample builder
Artifact id: `mjai-loader-replay-builder`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:346-425`
Why it matters: Shows exactly what the normal MJAI replay path constructs today: replay-derived safety residual and Stage-A belief targets, but no exit_target producer.

## Artifact 22 — Current Stage-A belief teacher implementation
Artifact id: `belief-teacher-stage-a`
Source label: BELIEF
Type: `file_range`
Source: `crates/hydra-train/src/teacher/belief.rs:1-201`
Why it matters: Concrete example of a target path that exists in code but is semantically weaker than doctrine wants. Useful contrast for replay/sample ExIt discussions about 'just because we can build a target does not mean we should'.

## Artifact 23 — Current ExIt teacher and collation code
Artifact id: `exit-teacher-code`
Source label: EXIT
Type: `file_range`
Source: `crates/hydra-train/src/training/exit.rs:1-320`
Why it matters: Core implementation of the ExIt teacher semantics, gates, canonical AFBS bridge, and batch collation logic. This is the most important current-code artifact for what the producer is supposed to emit.

## Artifact 24 — Current live self-play ExIt producer
Artifact id: `live-exit-producer`
Source label: LIVE
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:1-320`
Why it matters: Important because it shows one real upstream producer that already works, including the adapter seam and all-legal root-child seeding. The research question is whether this can or should generalize to replay/sample, not whether it exists.

## Artifact 25 — HydraTargets, optional advanced targets, and total-loss behavior
Artifact id: `loss-surfaces-and-target-presence`
Source label: LOSSES
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:1-699`
Why it matters: Important because reconciliation insists target presence must control whether an advanced loss exists at all. This artifact shows the actual target slots, masked-vs-dense loss patterns, and the current total_loss behavior for optional advanced targets.

## Artifact 26 — RL consumption of exit_target and exit_mask
Artifact id: `rl-exit-consumer`
Source label: RL
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:130-219`
Why it matters: Shows that RL can already consume exit_target when an upstream producer exists, so the blocked seam is upstream production rather than the consumer math.

## Artifact 27 — Maintenance-plan gating for live ExIt
Artifact id: `orchestrator-live-exit-gates`
Source label: ORCH
Type: `file_range`
Source: `crates/hydra-train/src/training/orchestrator.rs:158-205`
Why it matters: Shows how the current training system thinks about live ExIt enablement and keeps the producer phase-aware rather than universally on in all training contexts.

## Artifact 28 — Current model output surfaces and policy_value_cpu adapter
Artifact id: `model-output-surface`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:1-340`
Why it matters: Grounds what the learner actually exposes today: the 46-wide action surfaces, value head, and the single-observation adapter used by live ExIt. Useful for deciding whether a proposed replay/offline producer is actually compatible with current surfaces.

## Artifact 29 — Self-play carrier path from trajectory labels to RL batch
Artifact id: `selfplay-exit-carrier`
Source label: SELFPLAY
Type: `file_range`
Source: `crates/hydra-train/src/selfplay.rs:270-529`
Why it matters: This is the real live lane closure path: labels get attached at decision time, forwarded through trajectories, and collated into RL batches.

## Artifact 30 — TrajectoryExitLabel carrier object
Artifact id: `trajectory-exit-label`
Source label: ARENA
Type: `file_range`
Source: `crates/hydra-core/src/arena.rs:7-29`
Why it matters: Tiny but important object that proves the live lane already has a dedicated carrier shape independent of the ordinary replay/sample structs.

## Artifact 31 — AFBS expansion, search loop, and root_exit_policy
Artifact id: `afbs-search-shape`
Source label: AFBS
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:188-305`
Why it matters: Needed to understand why TOP_K expansion is fine for ordinary search but not sufficient for broad ExIt label support, and why root_exit_policy is a different object from child-visit labels.

## Artifact 32 — Ponder result stores q-softmax exit_policy
Artifact id: `afbs-ponder-policy`
Source label: AFBS
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:412-438`
Why it matters: Useful contrast artifact: runtime pondering stores root_exit_policy/q-softmax for cache/reporting, which is not automatically the ExIt teacher object.

## Artifact 33 — Bridge search-context surface
Artifact id: `bridge-search-context`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:27-45`
Why it matters: Shows that runtime/search context can carry AFBS and belief objects, but does not itself prove a replay/offline export path. Relevant to the doctrine warning against coupling replay loading to runtime-only search context.

## Artifact 34 — Exit validation harness and step-level attribution
Artifact id: `exit-validation-harness`
Source label: XVAL
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:1-679`
Why it matters: Adds the actual validation-matrix machinery, not just the status prose. Useful for reasoning about what an honest future producer would need to prove before activation.

## Artifact 35 — Integration tests that prove current ExIt plumbing
Artifact id: `integration-exit-tests`
Source label: ITEST
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:113-245`
Why it matters: End-to-end evidence for what the repo currently proves: trajectory label carrier, RL collation, and AFBS q-softmax helper behavior.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <config_section>
  <config_text status="preserved" source_path="replay_exit_producer_prompt_config.json">
  <![CDATA[
{
  "version": 1,
  "repo_root": "../..",
  "defaults": {
    "title": "Hydra prompt — replay/sample ExIt producer-path research",
    "artifact_container_tag": "artifacts",
    "artifact_ids": [
      "task-brief",
      "scope-rules",
      "readme-status",
      "hydra-final-exit",
      "reconciliation-tranche",
      "roadmap-head-status",
      "roadmap-exit-status",
      "archive-roadmap",
      "archive-jsonl-status",
      "answer-15-provenance",
      "answer-9-missing-closure",
      "answer-9-target-objects",
      "answer-22-live-exit-blueprint",
      "answer-23-shared-producer",
      "answer-7-1-conservative-exit",
      "answer-8-1-offline-posterior-builder",
      "answer-12-kill-family",
      "answer-5-1-offline-label-hint",
      "sample-structs",
      "sample-to-targets",
      "mjai-loader-replay-builder",
      "belief-teacher-stage-a",
      "exit-teacher-code",
      "live-exit-producer",
      "loss-surfaces-and-target-presence",
      "rl-exit-consumer",
      "orchestrator-live-exit-gates",
      "model-output-surface",
      "selfplay-exit-carrier",
      "trajectory-exit-label",
      "afbs-search-shape",
      "afbs-ponder-policy",
      "bridge-search-context",
      "exit-validation-harness",
      "integration-exit-tests"
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
        "tag": "style",
        "lines": [
          "- no high-level survey",
          "- no vague answer",
          "- include reasoning",
          "- include formulas when needed",
          "- include code-like detail when helpful (python or rust)",
          "- include worked examples when helpful",
          "- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)",
          "- distinguish direct artifact support from your own inference",
          "- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base",
          "- use the bash tool to run Python for calculations, math checks, and validation when rigor matters",
          "- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced",
          "- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)"
        ]
      },
      {
        "tag": "artifact_note",
        "lines": [
          "The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.",
          "Authority order still wins: README.md -> research/design/HYDRA_FINAL.md -> research/design/HYDRA_RECONCILIATION.md -> docs/GAME_ENGINE.md. Archive and handoff artifacts are evidence only unless validated against that chain and current code."
        ]
      }
    ]
  },
  "artifacts": [
    {
      "id": "task-brief",
      "type": "literal",
      "label": "Exact research question",
      "explanation": "Pins the task to the real blocked seam so the answer does not drift into a broad ExIt survey.",
      "fence_language": "text",
      "content_lines": [
        "We already know the live self-play ExIt lane is implemented.",
        "The blocked question is narrower: does Hydra already have enough doctrine + archive + code evidence to define an honest replay/sample-side ExIt producer path, or is that path still genuinely under-specified?",
        "Your job is to decide whether the surviving evidence implies a buildable path now, or whether the correct blueprint is to keep the normal replay/sample path absent until a different upstream producer exists.",
        "If a robust path exists, give the narrowest file-level blueprint that stays doctrine-compatible.",
        "If no robust path exists, give a blocked verdict blueprint that explains exactly why and names the smallest decisive research or implementation step that would unblock it."
      ]
    },
    {
      "id": "scope-rules",
      "type": "literal",
      "label": "Scope and anti-drift rules",
      "explanation": "Keeps the prompt narrow and blocks the common failure modes that already bit this lane.",
      "fence_language": "text",
      "content_lines": [
        "Do not redesign Hydra architecture.",
        "Do not propose broad AFBS expansion.",
        "Do not invent hidden-state or replay-only labels and call them public-teacher supervision.",
        "Do not assume replay reconstructibility implies public legitimacy.",
        "Do not treat archive artifacts as doctrine by default.",
        "Do not answer only the live self-play ExIt question; that lane is already mostly settled.",
        "Focus on whether the normal replay/sample path can honestly emit exit_target/exit_mask, and if so from what upstream source and with what provenance contract.",
        "If the honest answer is 'keep replay/sample absent', say that directly and explain the smallest alternate producer family that survives."
      ]
    },
    {
      "id": "readme-status",
      "type": "file_range",
      "path": "README.md",
      "start_line": 60,
      "end_line": 63,
      "label": "Repo status summary",
      "explanation": "Top-level authority says replay/sample exit_target production is still an immediate need, so the prompt has to explain whether that need is actually already solved somewhere else or still open.",
      "source_label": "README",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "hydra-final-exit",
      "type": "file_range",
      "path": "research/design/HYDRA_FINAL.md",
      "start_line": 257,
      "end_line": 280,
      "label": "Hydra final doctrine for ExIt targets",
      "explanation": "SSOT-level statement of the ExIt teacher object and the wider training-engine framing; it defines semantics, not necessarily the missing producer path.",
      "source_label": "FINAL",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "reconciliation-tranche",
      "type": "file_range",
      "path": "research/design/HYDRA_RECONCILIATION.md",
      "start_line": 432,
      "end_line": 558,
      "label": "Current execution doctrine for target closure",
      "explanation": "This is the main active-path doctrine: it names the missing producer seam, warns against weak labels, and defines the minimal tranche acceptance checklist.",
      "source_label": "RECON",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "roadmap-head-status",
      "type": "file_range",
      "path": "research/design/IMPLEMENTATION_ROADMAP.md",
      "start_line": 178,
      "end_line": 199,
      "label": "Roadmap head-summary status note",
      "explanation": "Subordinate reference showing which advanced targets already have normal supervised carriers and which do not.",
      "source_label": "ROADMAP",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "roadmap-exit-status",
      "type": "file_range",
      "path": "research/design/IMPLEMENTATION_ROADMAP.md",
      "start_line": 757,
      "end_line": 783,
      "label": "Roadmap ExIt pipeline status block",
      "explanation": "Useful because it truth-aligns the live self-play producer as implemented while still saying the normal replay/sample path does not emit exit_target.",
      "source_label": "ROADMAP",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "archive-roadmap",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md",
      "start_line": 93,
      "end_line": 167,
      "label": "Archive triage map for ExIt and adjacent lanes",
      "explanation": "This is the strongest archive summary of what got closed in the live lane versus what still remains open in the normal path. Evidence only, not doctrine.",
      "source_label": "AR-ROAD",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "archive-jsonl-status",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl",
      "start_line": 43,
      "end_line": 49,
      "label": "Canonical archive rows for live ExIt status and remaining gaps",
      "explanation": "Source-ledger version of the archive claims. Includes the live ExIt blueprint/status rows and the later truth-alignment note that replay/sample still does not emit exit_target.",
      "source_label": "AR-JSONL",
      "fence_language": "json",
      "show_line_numbers": true
    },
    {
      "id": "answer-15-provenance",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_15_combined.md",
      "start_line": 150,
      "end_line": 225,
      "label": "Answer 15 provenance taxonomy and public-teacher boundary",
      "explanation": "This is the best prior artifact for why ExIt/search-derived labels are later, why replay reconstructibility is not enough, and why provenance class matters.",
      "source_label": "A15",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-9-missing-closure",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_9_combined.md",
      "start_line": 1141,
      "end_line": 1206,
      "label": "Answer 9 missing-closure audit",
      "explanation": "Earlier narrow audit of what was structurally missing or semantically unclosed; especially useful for the search-compatible state and the warning not to inherit root_exit_policy blindly.",
      "source_label": "A9",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-9-target-objects",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_9_combined.md",
      "start_line": 1240,
      "end_line": 1419,
      "label": "Answer 9 exact target objects and sample-carrier proposal",
      "explanation": "This is the strongest old artifact for exact exit_target/delta_q target definitions and a proposed SearchSupervision carrier. It may be stale or superseded, so it must be critiqued against current doctrine and code.",
      "source_label": "A9",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-22-live-exit-blueprint",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_22_combined.md",
      "start_line": 227,
      "end_line": 339,
      "label": "Answer 22 live AFBS ExIt evaluator blueprint",
      "explanation": "High-signal artifact for the live self-play lane: visit-based labels, value-head evaluator, all-legal seeding, and why root_exit_policy is not the teacher.",
      "source_label": "A22",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-23-shared-producer",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_23_combined.md",
      "start_line": 871,
      "end_line": 1029,
      "label": "Answer 23 shared root-search producer blueprint",
      "explanation": "Closest archived file-level producer blueprint. Mostly written for delta_q piggybacking on ExIt, but directly relevant because it argues for one shared root-search label envelope rather than a replay-loader builder.",
      "source_label": "A23",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-7-1-conservative-exit",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_7-1_combined.md",
      "start_line": 296,
      "end_line": 415,
      "label": "Answer 7-1 conservative ExIt proposal",
      "explanation": "Important predecessor artifact for a narrow, trust-gated conservative ExIt path. Useful because it argues from current AFBS shell and target closure without requiring broad new architecture.",
      "source_label": "A7-1",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-8-1-offline-posterior-builder",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_8-1_combined.md",
      "start_line": 665,
      "end_line": 799,
      "label": "Answer 8-1 offline posterior-world builder proposal",
      "explanation": "This is the strongest explicit archive proposal for an offline replay label builder with added dataset fields. It is not doctrine, but it directly addresses the kind of path we might otherwise have to re-invent.",
      "source_label": "A8-1",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-12-kill-family",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_12_combined.md",
      "start_line": 131,
      "end_line": 214,
      "label": "Answer 12 stricter kill-pass on multi-world ExIt family",
      "explanation": "High-signal counterweight artifact: it explicitly argues that offline posterior-consensus ExIt is not ready under current Hydra constraints because the public world-teacher action object is missing.",
      "source_label": "A12",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer-5-1-offline-label-hint",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_5-1_combined.md",
      "start_line": 285,
      "end_line": 326,
      "label": "Answer 5-1 offline label-generation hint",
      "explanation": "Not an ExIt producer blueprint directly, but a useful artifact for the general pattern of offline ladder generation and maybe_emit_exit_target_if_safety_valve_passes semantics.",
      "source_label": "A5-1",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "sample-structs",
      "type": "file_range",
      "path": "crates/hydra-train/src/data/sample.rs",
      "start_line": 46,
      "end_line": 159,
      "label": "Current replay sample and batch structs",
      "explanation": "Shows what the ordinary supervised data path actually carries today. Important for checking whether a real replay/sample ExIt slot already exists or would need a new carrier.",
      "source_label": "SAMPLE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "sample-to-targets",
      "type": "file_range",
      "path": "crates/hydra-train/src/data/sample.rs",
      "start_line": 387,
      "end_line": 449,
      "label": "Current sample-to-HydraTargets conversion",
      "explanation": "Critical reality check: this is where optional advanced targets become HydraTargets in the ordinary supervised path, and it currently leaves delta_q absent and has no ExIt lane here.",
      "source_label": "SAMPLE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "mjai-loader-replay-builder",
      "type": "file_range",
      "path": "crates/hydra-train/src/data/mjai_loader.rs",
      "start_line": 346,
      "end_line": 425,
      "label": "Current replay loader sample builder",
      "explanation": "Shows exactly what the normal MJAI replay path constructs today: replay-derived safety residual and Stage-A belief targets, but no exit_target producer.",
      "source_label": "LOADER",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "belief-teacher-stage-a",
      "type": "file_range",
      "path": "crates/hydra-train/src/teacher/belief.rs",
      "start_line": 1,
      "end_line": 201,
      "label": "Current Stage-A belief teacher implementation",
      "explanation": "Concrete example of a target path that exists in code but is semantically weaker than doctrine wants. Useful contrast for replay/sample ExIt discussions about 'just because we can build a target does not mean we should'.",
      "source_label": "BELIEF",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "exit-teacher-code",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit.rs",
      "start_line": 1,
      "end_line": 320,
      "label": "Current ExIt teacher and collation code",
      "explanation": "Core implementation of the ExIt teacher semantics, gates, canonical AFBS bridge, and batch collation logic. This is the most important current-code artifact for what the producer is supposed to emit.",
      "source_label": "EXIT",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "live-exit-producer",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/live_exit.rs",
      "start_line": 1,
      "end_line": 320,
      "label": "Current live self-play ExIt producer",
      "explanation": "Important because it shows one real upstream producer that already works, including the adapter seam and all-legal root-child seeding. The research question is whether this can or should generalize to replay/sample, not whether it exists.",
      "source_label": "LIVE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "loss-surfaces-and-target-presence",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/losses.rs",
      "start_line": 1,
      "end_line": 699,
      "label": "HydraTargets, optional advanced targets, and total-loss behavior",
      "explanation": "Important because reconciliation insists target presence must control whether an advanced loss exists at all. This artifact shows the actual target slots, masked-vs-dense loss patterns, and the current total_loss behavior for optional advanced targets.",
      "source_label": "LOSSES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "rl-exit-consumer",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/rl.rs",
      "start_line": 130,
      "end_line": 219,
      "label": "RL consumption of exit_target and exit_mask",
      "explanation": "Shows that RL can already consume exit_target when an upstream producer exists, so the blocked seam is upstream production rather than the consumer math.",
      "source_label": "RL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "orchestrator-live-exit-gates",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/orchestrator.rs",
      "start_line": 158,
      "end_line": 205,
      "label": "Maintenance-plan gating for live ExIt",
      "explanation": "Shows how the current training system thinks about live ExIt enablement and keeps the producer phase-aware rather than universally on in all training contexts.",
      "source_label": "ORCH",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "model-output-surface",
      "type": "file_range",
      "path": "crates/hydra-train/src/model.rs",
      "start_line": 1,
      "end_line": 340,
      "label": "Current model output surfaces and policy_value_cpu adapter",
      "explanation": "Grounds what the learner actually exposes today: the 46-wide action surfaces, value head, and the single-observation adapter used by live ExIt. Useful for deciding whether a proposed replay/offline producer is actually compatible with current surfaces.",
      "source_label": "MODEL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "selfplay-exit-carrier",
      "type": "file_range",
      "path": "crates/hydra-train/src/selfplay.rs",
      "start_line": 270,
      "end_line": 529,
      "label": "Self-play carrier path from trajectory labels to RL batch",
      "explanation": "This is the real live lane closure path: labels get attached at decision time, forwarded through trajectories, and collated into RL batches.",
      "source_label": "SELFPLAY",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "trajectory-exit-label",
      "type": "file_range",
      "path": "crates/hydra-core/src/arena.rs",
      "start_line": 7,
      "end_line": 29,
      "label": "TrajectoryExitLabel carrier object",
      "explanation": "Tiny but important object that proves the live lane already has a dedicated carrier shape independent of the ordinary replay/sample structs.",
      "source_label": "ARENA",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "afbs-search-shape",
      "type": "file_range",
      "path": "crates/hydra-core/src/afbs.rs",
      "start_line": 188,
      "end_line": 305,
      "label": "AFBS expansion, search loop, and root_exit_policy",
      "explanation": "Needed to understand why TOP_K expansion is fine for ordinary search but not sufficient for broad ExIt label support, and why root_exit_policy is a different object from child-visit labels.",
      "source_label": "AFBS",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "afbs-ponder-policy",
      "type": "file_range",
      "path": "crates/hydra-core/src/afbs.rs",
      "start_line": 412,
      "end_line": 438,
      "label": "Ponder result stores q-softmax exit_policy",
      "explanation": "Useful contrast artifact: runtime pondering stores root_exit_policy/q-softmax for cache/reporting, which is not automatically the ExIt teacher object.",
      "source_label": "AFBS",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "bridge-search-context",
      "type": "file_range",
      "path": "crates/hydra-core/src/bridge.rs",
      "start_line": 27,
      "end_line": 45,
      "label": "Bridge search-context surface",
      "explanation": "Shows that runtime/search context can carry AFBS and belief objects, but does not itself prove a replay/offline export path. Relevant to the doctrine warning against coupling replay loading to runtime-only search context.",
      "source_label": "BRIDGE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "exit-validation-harness",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit_validation.rs",
      "start_line": 1,
      "end_line": 679,
      "label": "Exit validation harness and step-level attribution",
      "explanation": "Adds the actual validation-matrix machinery, not just the status prose. Useful for reasoning about what an honest future producer would need to prove before activation.",
      "source_label": "XVAL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "integration-exit-tests",
      "type": "file_range",
      "path": "crates/hydra-train/tests/integration_pipeline.rs",
      "start_line": 113,
      "end_line": 245,
      "label": "Integration tests that prove current ExIt plumbing",
      "explanation": "End-to-end evidence for what the repo currently proves: trajectory label carrier, RL collation, and AFBS q-softmax helper behavior.",
      "source_label": "ITEST",
      "fence_language": "rust",
      "show_line_numbers": true
    }
  ],
  "variants": [
    {
      "name": "replay-exit-producer-research",
      "title": "Hydra prompt — replay/sample ExIt producer-path research",
      "output_file": "replay_exit_producer_research_prompt.md",
      "artifact_ids": [],
      "shell_sections": [
        {
          "tag": "direction",
          "lines": [
            "Work toward the strongest exact blueprint for the replay/sample-side ExIt producer-path question.",
            "",
            "We want a detailed answer that makes clear:",
            "- whether current authoritative docs plus current code already imply a buildable replay/sample-side ExIt producer path",
            "- whether the strongest archive artifacts (`answer_15`, `answer_9`, `answer_22`, `answer_23`, canonical roadmap/jsonl rows) compose into a real actionable blueprint or still stop short",
            "- whether the ordinary replay/sample loader should remain absent for `exit_target` / `exit_mask` in the current tranche",
            "- whether the only honest surviving closure lane is a separate shared root-search producer / offline producer family rather than a replay-loader builder",
            "- what exact provenance class the surviving path belongs to: replay-derived, bridge-derived, search-derived, or blocked",
            "- what exact file-level plan survives after validating doctrine, archive evidence, and current code against each other",
            "- what must stay narrow / deferred / rejected / absent",
            "- if the path is still blocked, what the smallest decisive unblocker is",
            "",
            "Use the artifacts below to derive your conclusions.",
            "",
            "Your answer must end in one of two forms:",
            "1. an implementation-ready blueprint for a doctrine-compatible replay/sample-side or shared-producer closure path, with exact file-level changes, invariants, provenance rules, and tests; or",
            "2. a blocked-verdict blueprint that proves why the normal replay/sample path must stay absent for now, names the smallest honest alternative producer family that survives, and states the smallest decisive research/implementation step needed to unblock it."
          ]
        }
      ],
      "artifacts": []
    }
  ]
}
]]>
  </config_text>
  </config_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_24.md">
  <![CDATA[# Implementation-ready blueprint: keep the ordinary replay/sample loader absent, and close replay-side ExIt through a shared **search-derived** offline producer

## 1. Decision surface

| Question                                                                                                                                        | Verdict                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Do current authoritative docs + current code already define a buildable **ordinary** replay/sample loader path for `exit_target` / `exit_mask`? | **No.**                                                                                            |
| Do they define a buildable **shared producer** path that can serve replay-indexed data honestly?                                                | **Yes.**                                                                                           |
| Should the normal `mjai_loader` / `MjaiSample` path stay absent for `exit_target` / `exit_mask` in the current tranche?                         | **Yes.**                                                                                           |
| What provenance class survives?                                                                                                                 | **Search-derived** (replay-indexed, but not replay-derived).                                       |
| What is the smallest decisive unblocker?                                                                                                        | Add an explicit **offline root-search sidecar producer** that reuses the current live ExIt kernel. |

**Direct artifact support**

* README still says replay/sample `exit_target` production is an immediate need, so it is not already closed in the ordinary path (`README L0062`).
* Reconciliation explicitly says `mjai_loader.rs` still has **no normal production path** for `exit_target`, calls for a **real upstream producer path**, says unavailable targets must stay absent, and warns against coupling replay loading to runtime-only search context unless there is an **explicit offline generation path** (`RECON L0434-L0443`, `RECON L0496-L0504`, `RECON L0528-L0537`).
* `MjaiSample` / `MjaiBatch` have no ordinary ExIt carrier, and `into_hydra_targets()` / `to_hydra_targets()` still do not produce any ExIt target (`SAMPLE L0046-L0064`, `SAMPLE L0387-L0449`).
* `load_game_from_events()` builds replay-derived `safety_residual` and Stage-A belief targets, but not ExIt (`LOADER L0399-L0425`).
* The live self-play lane is already real, and the roadmap truth-aligns that the **normal replay/sample path still does not emit `exit_target`** (`ROADMAP L0765-L0783`, `AR-JSONL L0049`).

**Inference**

The ordinary replay loader is not merely “missing a few fields.” It is the wrong place to synthesize ExIt, because ExIt is a **model-scoped search artifact**, not a static replay-derived truth.

---

## 2. Canonical object to preserve

The only target object that survives doctrine is the masked root-child **visit** distribution:

[
t_{\text{exit}}(a \mid I_t;\theta,c)=
\begin{cases}
\dfrac{N_{\theta,c}(I_t,a)}{\sum_{b \in C(I_t)} N_{\theta,c}(I_t,b)} & a \in C(I_t) \
0 & \text{otherwise}
\end{cases}
]

with mask

[
m_{\text{exit}}(a \mid I_t)=\mathbf 1[a \in C(I_t)].
]

Here:

* (I_t) is the public information state.
* (N_{\theta,c}(I_t,a)) is the root child visit count from the current AFBS search, under model checkpoint (\theta) and search config (c).
* (C(I_t)) is the supported discard-compatible child set after gating.

Current Hydra code already encodes the exact gates:

* discard-compatible only
* no aka-discard ambiguity
* at least 2 legal discards
* `child_visits >= 2`
* `avg_root_visits_per_legal_discard >= 8.0`
* `coverage >= 0.60`
* KL safety valve against the base policy (`EXIT L0141-L0226`).

**Direct artifact support**

* `HYDRA_FINAL` says the ExIt teacher object is the masked, visit-based root-child distribution, and that `root_exit_policy()` / q-softmax is **not** the teacher object (`FINAL L0259-L0262`).
* `exit.rs` implements that exact object and those exact gates (`EXIT L0157-L0261`).
* `AfbsTree::root_exit_policy()` is a different q-softmax object (`AFBS L0265-L0305`), and `PonderResult::from_tree()` stores that runtime/reporting object, not the training teacher (`AFBS L0419-L0438`).
* `answer_22` survives on the same distinction (`A22 L0247-L0275`).

**External source cross-check**

ExIt’s original paper explicitly defines the stronger imitation target as the **root visit distribution** (n(s,a)/n(s)), not the chosen action, and motivates it because it is cost-sensitive and better for future search guidance. The same paper also says the apprentice can improve search by guiding it or by quickly estimating values of encountered states. AlphaZero likewise trains the policy against **search probabilities** returned from root visit counts. Grill et al. later show that the empirical visit distribution (\hat\pi) only approximates a different regularized-policy solution (\bar\pi), and that low simulation budgets make (\hat\pi) discretized and under-supportive of unsampled actions; that is exactly why Hydra must not silently replace the visit teacher with q-softmax. ([NeurIPS Papers][1])

---

## 3. Proof that the normal replay/sample loader must stay absent

### 3.1 The loader does not have the required object

A replay log gives:

[
(\text{public state } I_t,\ \text{realized action } a_t^{\text{replay}},\ \text{future events})
]

but ExIt needs:

[
{N_{\theta,c}(I_t,a)}_{a \in \mathcal A}
]

the whole root child visit vector under a current search run.

Those counts are **not** present in the ordinary replay sample path today (`SAMPLE`, `LOADER`), and they are not recoverable from the single realized action without rerunning search.

### 3.2 Even if you replay the state, the label is still not replay-derived

If you reconstruct (I_t) from the replay and then run AFBS with the current model to get (N_{\theta,c}(I_t,a)), the output is:

[
g_{\text{AFBS}}(I_t;\theta,c)
]

That is **search-derived**.

The replay is only the **state enumerator**.

So the honest provenance is:

* **replay-derived**: labels computed directly from the log without search
* **search-derived**: labels computed by running search on the reconstructed public state

This is exactly why `answer_15` matters: replay reconstructibility is not the same thing as public-teacher legitimacy, and ExIt belongs to the bridge/search-derived-later family, not the replay-derived-now family (`A15 L0160-L0164`, `A15 L0178-L0184`, `A15 L0188-L0204`).

### 3.3 The target is checkpoint-scoped, so it should not be hidden inside the ordinary loader

For a fixed replay state (I_t), checkpoint (\theta_1) may emit a label, while checkpoint (\theta_2):

* fails the hard-state gate,
* changes the supported children,
* changes the KL gate,
* changes the normalized visit distribution.

So `exit_target` is not immutable replay truth. It is a **teacher-product of a specific checkpoint + search config**.

That is loader-incompatible unless you make the loader explicitly model-dependent and provenance-aware - which is no longer an ordinary replay loader. It is a separate offline producer.

### 3.4 Current code architecture already points that way

Live ExIt already follows this split:

* `exit.rs` defines teacher semantics.
* `live_exit.rs` produces labels upstream.
* `selfplay.rs` carries them into `RlBatch`.
* `rl.rs` consumes them separately from `HydraTargets`.

That architecture says: ExIt is a **producer-side policy teacher**, not an ordinary `HydraTargets` field (`LIVE L0178-L0271`, `SELFPLAY L0377-L0476`, `RL L0148-L0156`).

---

## 4. What from the archive still survives, and what does not

| Archive artifact | Survives                                                                                                                                                                                                | Does **not** survive                                                                                                                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `answer_15`      | The provenance taxonomy. ExIt is not replay-derived-now; replay reconstructibility is not enough (`A15 L0160-L0184`, `A15 L0188-L0204`).                                                                | Any temptation to call replay-only labels “public-teacher ExIt.”                                                                                                                                                                                             |
| `answer_9`       | `compatible_discard_state`, discard-only narrowness, child-visit teacher object, coverage/support gates, warning not to inherit `root_exit_policy()` (`A9 L1242-L1325`, `A9 L1191-L1200`).              | The old carrier proposal that predates the current live RL split. Current code uses separate `exit_target/exit_mask` in `RlBatch`, not an ExIt slot in `HydraTargets` (`A9 L1381-L1419` is directionally useful, but not the exact surviving carrier shape). |
| `answer_22`      | The live producer algorithm: learner-only, root-only AFBS, current public value head as leaf scorer, all-legal discard seeding, visit teacher, reject q-softmax (`A22 L0229-L0241`, `A22 L0247-L0319`). | The older default-off status language; current truth-aligned code is default-on for the live lane (`ROADMAP L0765-L0783`, `AR-JSONL L0048-L0049`).                                                                                                           |
| `answer_23`      | The shared-producer verdict: do **not** build a replay-loader builder; reuse one root-search envelope and keep replay path absent until the producer exists (`A23 L0871-L0885`, `A23 L0956-L1007`).     | Any pressure to pull `delta_q` into the same first patch. `delta_q` remains blocked until mask + root-perspective Q closure exist.                                                                                                                           |
| `answer_8-1`     | Only the generic pattern “explicit offline builder if a real teacher exists.”                                                                                                                           | The posterior-consensus ExIt family itself. `answer_12` kills it as current scope because the public world-conditioned action teacher is missing (`A12 L0137-L0151`, `A12 L0179-L0183`, `A12 L0208-L0213`).                                                  |

**Bottom line**

The archive set **does compose** into a real actionable blueprint, but only after narrowing it to:

> **shared root-search producer family**
>
> * **offline replay-indexed sidecar generator**
> * **ordinary loader still absent until the sidecar exists**

It does **not** compose into an honest “put search inside `mjai_loader.rs`” plan.

---

## 5. Surviving architecture

```text
MJAI replay events
    |
    |  (public-state reconstruction only)
    v
shared replay walk ----------------------------+
    |                                          |
    |                                          |
    v                                          v
ordinary replay loader                    offline ExIt producer
(replay-derived targets only)             (search-derived)
    |                                          |
    |                                          +--> sidecar.jsonl.zst
    |                                                   key + target + mask + provenance
    |                                                            |
    +-------------------------- optional later join -------------+
                                                               |
                                                               v
                                                     supervised batch with
                                                     exit_target/exit_mask
```

### Provenance classification

* `safety_residual_target`: **replay-derived**
* `build_search_features()` / bridge summaries: **bridge-derived** summaries only
* live self-play ExIt: **search-derived**
* offline replay-indexed ExIt sidecar: **search-derived**
* replay loader with no search: **blocked**
* q-softmax from `root_exit_policy()` / ponder cache: **rejected for teacher use**

---

## 6. The implementation plan that survives

## Patch A (current tranche): **build the producer, keep the ordinary loader absent**

This is the decisive unblocker.

### A1. `crates/hydra-train/src/training/live_exit.rs`

**Goal:** decouple the current search kernel from `StepRecord` enough to reuse it offline.

### Change

Introduce a small request struct and make the current live helper wrap it.

```rust
#[derive(Clone)]
pub struct RootDecisionContext {
    pub obs_encoded: [f32; OBS_SIZE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub policy_logits: [f32; HYDRA_ACTION_SPACE],
    pub player_id: u8,
}
```

Add:

```rust
pub fn try_exit_label_from_context<M, A>(
    state: &GameState,
    obs: &Observation,
    ctx: &RootDecisionContext,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
```

Implementation: move the body of current `try_live_exit_label(...)` here, replacing:

* `step.player_id -> ctx.player_id`
* `step.legal_mask -> ctx.legal_mask`
* `step.policy_logits -> ctx.policy_logits`

Keep current `try_live_exit_label(...)` as a thin wrapper:

```rust
pub fn try_live_exit_label<M, A>(...) -> Option<TrajectoryExitLabel> {
    let ctx = RootDecisionContext::from_step(step);
    try_exit_label_from_context(state, obs, &ctx, safety, cfg, model_pv, adapter)
}
```

Make `obs_hash` `pub(crate)` so the offline producer can reuse the exact same root key logic.

### Why this is the right minimal change

**Direct support:** `live_exit.rs` already has the generic `ExitSearchAdapter` trait and generic producer over adapter/model closure (`LIVE L0029-L0052`, `LIVE L0194-L0205`).
**Inference:** the only real selfplay coupling left is `StepRecord`. Remove that, and the live kernel becomes a reusable offline kernel with no doctrine change.

---

### A2. New file: `crates/hydra-train/src/training/replay_exit.rs`

**Goal:** add the replay-indexed search-derived producer.

### Types

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDecisionKey {
    pub source_hash: u64,
    pub event_index: u32,
    pub actor: u8,
    pub obs_hash: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayExitRecordV1 {
    pub version: u32, // = 1
    pub semantics: String, // "exit_root_child_visits_v1"
    pub provenance: String, // "search-derived"
    pub key: ReplayDecisionKey,

    pub source_net_hash: u64,
    pub source_version: u32,

    pub root_visit_count: u32,
    pub legal_discard_count: u8,
    pub supported_actions: u8,
    pub coverage: f32,
    pub kl_to_base: f32,

    pub target: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],

    // recommended for audit only:
    pub child_visits: Option<[u32; HYDRA_ACTION_SPACE]>,
}
```

### Adapter

Add `ReplayExitAdapter` implementing `ExitSearchAdapter`. It should mirror `SelfPlayExitAdapter`:

* clone `GameState`
* apply discard using a real tile136 from the player hand
* call `child_state.get_observation(player)`
* re-encode with `encode_observation(..., safety, None)`

Because `compatible_discard_state()` rejects aka-legal states, choosing the first matching 136 tile for a non-aka discard is semantically safe: copy identity among non-red duplicates does not change the public child observation in this narrow tranche.

### Producer entry point

```rust
pub fn generate_replay_exit_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayExitRecordV1>, ExitValidationReport)>
```

For each sampled replay decision:

1. reconstruct the same public state the loader would reconstruct
2. build the exact same encoded observation and legal mask the loader uses
3. run `model.policy_value_cpu(&obs_encoded, device)` to get root logits
4. build `RootDecisionContext`
5. call `try_exit_label_from_context(...)`
6. if it emits:

   * compute `supported_actions`, `coverage`, `kl_to_base`
   * write `ReplayExitRecordV1`
7. if it does not emit:

   * do **not** fabricate anything
   * update the audit report only

### Required invariants

* same `should_sample_replay_event(event)` selection as the ordinary loader
* same phase handling (`Normal` vs `RiichiSelect`) as the ordinary loader
* same observation encoding as the ordinary loader
* same `budget_from_legal_count()` as the live producer
* same `seed_root_children_all_legal()` as the live producer
* same canonical teacher builder: `build_exit_from_afbs_tree()`
* never call `exit_policy_from_q()` or `make_exit_target()` here

---

### A3. New file: `crates/hydra-train/src/bin/build_replay_exit_sidecar.rs`

**Goal:** make the producer explicit, offline, inspectable, and provenance-bearing.

### CLI surface

```text
build_replay_exit_sidecar \
  --input path/to/replays_or_manifest \
  --checkpoint path/to/model \
  --output path/to/exit_sidecar.jsonl.zst \
  --source-version 123 \
  --min-visits 64 \
  --hard-state-threshold 0.10 \
  --max-kl 2.0
```

### Behavior

* load checkpoint
* compute / require `source_net_hash`
* iterate replay files
* call `generate_replay_exit_records(...)`
* write records as JSONL (optionally zstd-compressed)
* write `exit_report.json` with `ExitValidationReport`

### Why JSONL for v1

* easy to inspect
* easy to diff
* easy to reject stale semantics/version
* compact enough for `[46]` vectors in v1

If volume later becomes painful, switch the on-disk encoding, but keep the same logical schema.

---

### A4. Optional but preferred helper extraction: `crates/hydra-train/src/data/replay_walk.rs`

**Goal:** keep loader-side and producer-side replay traversal aligned.

Extract the current `mjai_loader.rs` replay walk into one shared enumerator over sampled decision states.

Preferred shape:

```rust
pub trait ReplayDecisionVisitor {
    fn visit(
        &mut self,
        source_hash: u64,
        event_index: usize,
        actor: u8,
        state: &GameState,
        obs: &Observation,
        obs_encoded: [f32; OBS_SIZE],
        legal_mask_f32: [f32; HYDRA_ACTION_SPACE],
        hydra_action: u8,
        safety: &SafetyInfo,
    ) -> io::Result<()>;
}
```

Then:

* `mjai_loader.rs` uses it to build `MjaiSample`
* `replay_exit.rs` uses it to build sidecar records

If this extraction feels too invasive for Patch A, duplicate the current loop once, but **share the key-generation function** and keep the traversal byte-for-byte aligned with loader semantics.

---

### A5. Documentation hardening

#### `crates/hydra-train/src/training/exit.rs`

Add module docs:

* canonical Hydra ExIt teacher = `make_exit_target_from_child_visits()` / `build_exit_from_afbs_tree()`
* `exit_policy_from_q()` and `make_exit_target()` are not the canonical teacher object

#### `crates/hydra-core/src/afbs.rs`

Add docs on `root_exit_policy()` / `PonderResult::from_tree()`:

* runtime/reporting/cache object only
* not the training teacher

This is a low-cost footgun removal.

---

## Patch B (follow-on, **not** current tranche): join the sidecar into supervised replay/sample batches

This patch only starts **after Patch A exists and emits real labels**.

### B1. `crates/hydra-train/src/data/sample.rs`

Add optional ExIt carrier fields to `MjaiSample` and `MjaiBatch`:

```rust
pub struct MjaiSample {
    ...
    pub exit_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub exit_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
}

pub struct MjaiBatch<B: Backend> {
    ...
    pub exit_target: Option<Tensor<B, 2>>,
    pub exit_mask: Option<Tensor<B, 2>>,
}
```

Important rule:

* **do not** add ExIt to `HydraTargets`
* ExIt is a separate policy distillation term, exactly as in `RlBatch`

Use the existing `collate_exit_targets(...)` logic to build these tensors. Mixed batches follow the current RL convention:

* if every row is `None` -> `(None, None)`
* else absent rows become all-zero target / all-zero mask rows (`EXIT L0263-L0290`)

### B2. `crates/hydra-train/src/data/mjai_loader.rs`

Add an **optional sidecar lookup input**, not search.

Sketch:

```rust
pub fn load_game_from_events_with_sidecar(
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
) -> io::Result<MjaiGame>
```

At sample creation time:

* compute `ReplayDecisionKey`
* lookup sidecar
* if present and provenance/version matches:

  * set `sample.exit_target = Some(record.target)`
  * set `sample.exit_mask = Some(record.mask)`
* else:

  * leave both `None`

Absolute rule: `mjai_loader.rs` still does **zero search**.

### B3. `crates/hydra-train/src/training/bc.rs`

Mirror the RL pattern.

Do **not** replace the replay one-hot `policy_target`.
Do **not** silently blend ExIt into `policy_target`.
Do **not** route it through `HydraLoss.total_loss()`.

Instead:

```rust
let mut total = aux.total;

if let (Some(exit_target), Some(exit_mask)) = (&batch.exit_target, &batch.exit_mask) {
    total = total + crate::training::exit::exit_loss(
        outputs.policy_logits.clone(),
        exit_target.clone(),
        exit_mask.clone(),
        cfg.exit_weight,
    );
}
```

Default:

* `cfg.exit_weight = 0.0`
* no accidental activation
* no loss when targets are absent

This preserves Reconciliation’s “target presence controls whether the loss exists” rule (`RECON L0455-L0461`).

---

## 7. Exact invariants that must hold

These are non-negotiable.

### 7.1 Teacher invariants

1. ExIt teacher is **visit-based root child distribution**, not q-softmax.
2. Target is normalized only over the masked subset.
3. Mask support = discard-compatible, legal, `child_visits >= 2`.
4. No non-discard actions in support.
5. No aka-legal states in support.

### 7.2 Search invariants

6. Root-only, learner-only AFBS.
7. Current public model value head is the leaf scorer.
8. Root seeding must be **all legal discards**, not `expand_node()`.

Worked example:

* legal discards (L = 10)
* current `TOP_K = 5`
* max possible coverage with `expand_node()` = (5/10 = 0.50)
* required coverage = `0.60`

So `expand_node()` can never pass the gate on that state.
That is why `seed_root_children_all_legal()` is mandatory, not optional (`A22 L0305-L0319`).

### 7.3 Provenance invariants

9. Replay-indexed offline ExIt records are tagged `search-derived`.
10. Sidecar records carry `source_net_hash`, `source_version`, and semantics version.
11. Loader joins only matching semantics/version; mismatches are ignored or hard-failed.
12. `PonderResult.exit_policy` is never used as a teacher.

### 7.4 Training invariants

13. `HydraTargets` stays unchanged for ExIt.
14. Replay hard action target stays present; ExIt is an additional policy distillation term.
15. Missing ExIt rows produce no ExIt loss.
16. No zero-filled fake labels at sample creation time; only batch collation may zero-fill absent rows when at least one row is present.

### 7.5 Scope invariants

17. No new heads.
18. No broad AFBS rewrite.
19. No bridge-only label promotion.
20. No delta-q activation in the same patch.
21. No belief/mixture/opponent-hand-type activation.

---

## 8. Validation and test plan

## Patch A tests

### `training/live_exit.rs`

Add:

* `root_decision_context_from_step_matches_old_behavior`
* `try_exit_label_from_context_matches_try_live_exit_label_on_selfplay_fixture`

### `training/replay_exit.rs`

Add:

* `replay_exit_adapter_builds_child_obs_for_non_aka_discard`
* `replay_exit_rejects_non_discard_legal_state`
* `replay_exit_rejects_aka_legal_state`
* `replay_exit_uses_budget_from_legal_count`
* `replay_exit_all_legal_seeding_allows_coverage_on_10_discard_state`
* `replay_exit_record_serializes_roundtrip`
* `replay_exit_records_are_tagged_search_derived`
* `replay_exit_producer_never_calls_q_softmax_teacher`
  (assert by construction: the producer path goes through `build_exit_from_afbs_tree`, not `make_exit_target`)

### `bin/build_replay_exit_sidecar.rs`

Integration:

* build sidecar from a tiny replay corpus and tiny checkpoint
* assert file writes
* assert report exists
* assert at least one emitted label on a controlled fixture

## Patch B tests

### `data/sample.rs`

* `collate_exit_targets_all_none_returns_none`
* `collate_exit_targets_mixed_rows_zero_masks_absent_rows`

### `data/mjai_loader.rs`

* `loader_without_sidecar_keeps_exit_absent`
* `loader_with_sidecar_populates_exit_fields`
* `loader_rejects_mismatched_sidecar_semantics`

### `training/bc.rs`

* `bc_step_baseline_only`
* `bc_step_baseline_plus_exit`
* `bc_step_baseline_plus_exit_plus_safety_residual`
* assert finite total loss
* assert nonzero ExIt contribution when labels exist
* assert zero ExIt contribution when labels absent

## Structural validation gate for Patch A

Reuse `ExitValidationReport` semantics. For an offline replay slice, require:

* sample size `>= 1000`
* emission rate `>= 1%`
* mean coverage `>= 0.70`
* mean supported actions `>= 3`
* mean KL `<= 0.05`

Treat top-1 agreement as **report-only** until the teacher checkpoint is at least BC-warm-start quality, matching the live-lane truth alignment where random-weight top-1 agreement was not a meaningful enablement gate (`ROADMAP L0777-L0783`).

---

## 9. What must stay narrow / deferred / rejected / absent

### Stay absent now

* ordinary replay loader production of `exit_target` / `exit_mask`
* any `exit_target` creation from replay action alone
* any use of exact waits / future outcomes / hidden allocations in ExIt labels

### Rejected

* `root_exit_policy()` / q-softmax as ExIt teacher
* ponder/cache `exit_policy` as a training label
* bridge-only summaries as ExIt labels
* posterior-consensus / multi-world offline ExIt family from `8-1`; that family remains killed under the current doctrine (`A12 L0137-L0151`, `A12 L0208-L0213`)

### Deferred

* `delta_q` piggyback on the shared producer
* generic `TrajectorySearchLabel` / `RootSearchLabels` carrier
* Stage-A belief / mixture activation
* `opponent_hand_type_target`
* any 46-action non-discard ExIt broadening

For `delta_q`, keep the A23 rule: no replay-loader builder, no duplicate search, no activation until explicit mask + root-perspective Q closure + validation exist (`A23 L0871-L1007`).

---

## 10. The exact closure statement

**The ordinary replay/sample loader must stay absent for `exit_target` / `exit_mask` in the current tranche.**
That absence is not a bug in the doctrine; it is the honest consequence of the target object.

**The surviving implementation path is:**

1. keep `mjai_loader.rs` replay-pure
2. reuse the existing live ExIt kernel by decoupling it from `StepRecord`
3. add a replay-indexed, search-derived offline sidecar producer
4. only after that producer emits real labels, optionally join the sidecar into supervised batches

That path is fully doctrine-compatible, matches the strongest surviving archive evidence, and uses the current code’s real seam rather than inventing a replay-loader fiction. ExIt dataset generation is naturally parallel and compact, which is another reason to implement it as a separate producer family instead of embedding it inside the core loader. ([NeurIPS Papers][1])

[1]: https://papers.nips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf "https://papers.nips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
