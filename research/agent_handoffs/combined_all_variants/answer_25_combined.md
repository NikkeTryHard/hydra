<combined_run_record run_id="answer_25" variant_id="prompt_config_answer_compact" schema_version="1">
  <metadata>
    <notes>Compact self-contained combined record for Agent 25 RL-only delta_q validation-and-enable research. It preserves the rendered prompt shell reconstructed from the authoritative generator config, the full generator config inline, and the preserved Agent 25 answer.</notes>
    <notes>Historical notice: Agent 25 explicitly treats the inspected artifact tranche as newer than the public GitHub snapshot. Preserve that as historical answer context only; archive evidence does not outrank current Hydra doctrine.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_config_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — RL-only delta_q validation-and-enable contract

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (Rust preferred)
- include enough detail that we can validate it ourselves
- distinguish direct artifact support from your own inference
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked
- confidence must be justified, not asserted
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now.
They are not guaranteed to be fully correct.
Treat them as evidence to inspect and critique, not truth to inherit.
Archive and handoff artifacts are evidence only and must not be promoted over current Hydra doctrine.
</artifact_note>

<direction>
Work toward the strongest exact blueprint for Hydra's RL-only `delta_q` validation-and-enable tranche.
We want a detailed answer that makes clear:
- the narrowest doctrine-consistent RL-only `delta_q` validation contract
- exact metrics and thresholds that should govern whether RL-side `delta_q` is trusted enough to enable
- fail behavior and keep-off behavior
- the exact controller/orchestrator hookup path
- what must stay narrow / deferred / rejected
- how to implement and validate the surviving path with minimal guesswork
- what is directly supported vs inferred vs still blocked
Use the artifacts below to derive your conclusions.
Do not broaden scope into replay/offline `delta_q` or BC/train-bin activation unless an artifact forces it.
</direction>

<scope_note>
This is not a broad `delta_q` design prompt.
The semantics of the current narrow lane are already mostly settled:
- learner-only
- root-only
- search-derived
- discard-compatible
- masked `[46]`
- `Q(child) - Q(root)`
- RL/self-play lane only
- replay/offline absent
- BC/train-bin activation blocked
The question is narrower:
What is the strongest repo-grounded RL-only validation-and-enable contract for this lane?
</scope_note>

<deliverable>
Give a blueprint, not a memo.

Your answer must include:
1. direct support / inference / proposal / blocked buckets
2. surviving RL-only `delta_q` validation contract
3. minimal implementation plan
4. hookup design
5. validation plan
6. confidence justification
7. final decision: either 'repo-backed enough to implement now' or 'still blocked, and here is the smallest unresolved fact'
</deliverable>

<artifacts_manifest>

## Artifact 01 — Current repo status summary for delta_q
Artifact id: `readme-status`
Source label: README
Type: `file_range`
Source: `README.md:60-62`
Why it matters: Grounds the current staged status: narrow live self-play RL lane exists, while replay/offline production and train-bin activation remain blocked.

## Artifact 02 — Hydra north-star note on delta_q family
Artifact id: `final-current-note`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:119-121`
Why it matters: Shows that delta_q remains part of the search-distillation family but is not equally closed today.

## Artifact 03 — Reconciliation tranche rules for delta_q, RL, and BC
Artifact id: `recon-deltaq-tranche`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:448-489`
Why it matters: This is the strongest current doctrine for delta_q staging, masked loss semantics, RL presence, and continued BC/train-bin blocking.

## Artifact 04 — Current ExIt validation harness and refreshed roadmap note
Artifact id: `roadmap-exit-validation`
Source label: ROADMAP
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:775-783`
Why it matters: Provides the reusable in-repo validation pattern and current note that delta_q now exists only as a narrow RL lane while broader activation stays deferred.

## Artifact 05 — delta_q target builder and collation code
Artifact id: `exit-deltaq-builder`
Source label: EXIT
Type: `file_range`
Source: `crates/hydra-train/src/training/exit.rs:263-357`
Why it matters: Defines the current narrow object contract: masked discard-compatible root-child q-delta and batch collation into the [46] head space.

## Artifact 06 — Shared live search producer for ExIt and delta_q
Artifact id: `live-shared-search-labels`
Source label: LIVE
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:240-363`
Why it matters: Shows that delta_q is emitted by the same root-only live search envelope as ExIt, including upstream compatibility checks and shared provenance.

## Artifact 07 — delta_q live-producer test
Artifact id: `live-deltaq-tests`
Source label: LIVE
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:758-778`
Why it matters: Proves that good compatible input produces a real delta_q label with expected monotonic ordering over supported discard actions.

## Artifact 08 — Trajectory delta_q carrier validation
Artifact id: `arena-deltaq-validation`
Source label: ARENA
Type: `file_range`
Source: `crates/hydra-core/src/arena.rs:559-599`
Why it matters: Shows strict structural invariants already enforced for carried delta_q labels: legal-only, discard-only, binary mask, zero outside support, and non-empty support.

## Artifact 09 — Self-play to RL batch delta_q carrier path
Artifact id: `selfplay-rl-deltaq-carrier`
Source label: SELFPLAY
Type: `file_range`
Source: `crates/hydra-train/src/selfplay.rs:379-480`
Why it matters: Shows the real live lane from trajectory labels into RlBatch.targets.delta_q_target and delta_q_mask.

## Artifact 10 — Masked delta_q loss semantics
Artifact id: `losses-deltaq-masked`
Source label: LOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:550-595`
Why it matters: Shows that delta_q only contributes loss when both target and mask exist, and that unsupported actions do not silently train through dense zeros.

## Artifact 11 — Loss tests for missing targets and activated optional losses
Artifact id: `losses-optional-and-activation-tests`
Source label: LOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:782-899`
Why it matters: Shows current repo expectations around zero loss when optional targets are absent and finite aux loss when optional targets are present, including delta_q in the mixed optional-target tests.

## Artifact 12 — HeadActivationController config and sparse-search thresholds
Artifact id: `head-gates-core`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1-90`
Why it matters: Contains the generic sparse-head activation thresholds that would apply to DeltaQ if the controller were used live.

## Artifact 13 — DeltaQ classification, target presence, coverage, and conflict tracking
Artifact id: `head-gates-presence-and-coverage`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:90-509`
Why it matters: Shows how DeltaQ is classified as SparseSearch, how target presence is counted, and how coverage/conflict statistics are accumulated before any head can be approved.

## Artifact 14 — HeadActivationController core evaluation and approved loss config logic
Artifact id: `head-gates-controller-core`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:510-786`
Why it matters: Contains the actual controller state machine, per-head reports, activation attempts, and weight gating logic that a future delta_q enablement path would have to call.

## Artifact 15 — DeltaQ sparse-head gate tests and approved weight behavior
Artifact id: `head-gates-deltaq-tests`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1053-1208`
Why it matters: Shows concrete tested controller behavior for DeltaQ: sparse SPP rejection and approved_loss_config zeroing when the head remains off.

## Artifact 16 — Maintenance plan and live ExIt enablement
Artifact id: `orchestrator-live-exit-plan`
Source label: ORCH
Type: `file_range`
Source: `crates/hydra-train/src/training/orchestrator.rs:161-208`
Why it matters: Shows the existing validation-to-enable style for live ExIt producer phase gating, which may or may not transfer to delta_q without new doctrine.

## Artifact 17 — Existing ExIt validation harness
Artifact id: `exit-validation-harness`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:1-220`
Why it matters: This is the strongest live validation blueprint in repo; use it to judge what a delta_q-specific validation harness would need and what is currently missing.

## Artifact 18 — ExIt thresholds, report evaluation, and observational runner
Artifact id: `exit-validation-thresholds-and-report`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:220-475`
Why it matters: Provides the concrete in-repo metric pack, pass/fail evaluator, and shadow validation runner that a delta_q-specific validator would most naturally mirror.

## Artifact 19 — ExIt validation harness tests
Artifact id: `exit-validation-tests`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:560-811`
Why it matters: Shows the exact kinds of report-default, pass/fail, merge, and observational-run tests the repo already considers sufficient for a live validation harness.

## Artifact 20 — RL tests with advanced aux targets including delta_q
Artifact id: `rl-deltaq-tests`
Source label: RL
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:260-389`
Why it matters: Shows that RL-side delta_q consumption already exists in library code and remains numerically stable in combined auxiliary-loss scenarios.

## Artifact 21 — Integration proof that trajectory delta_q labels survive into RL batches
Artifact id: `integration-deltaq-batch-proof`
Source label: INTEG
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:149-250`
Why it matters: End-to-end evidence that the live lane is real: trajectory labels persist into RlBatch.targets with the expected values and masks.

## Artifact 22 — Replay loader keeps delta_q absent
Artifact id: `replay-absence-guard`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:437-459`
Why it matters: Shows replay/offline delta_q remains intentionally absent in the normal loader path.

## Artifact 23 — Replay absence regression test
Artifact id: `replay-absence-test`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:784-792`
Why it matters: Proves replay samples must not quietly grow delta_q targets or masks.

## Artifact 24 — Train-bin blocked advanced-loss policy
Artifact id: `train-loss-policy-block`
Source label: TRAIN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs:1-33`
Why it matters: Shows BC/train-bin activation for delta_q remains intentionally rejected by policy.

## Artifact 25 — delta_q rejection tests in train binary
Artifact id: `train-deltaq-reject-tests`
Source label: TRAIN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train.rs:1042-1060`
Why it matters: Confirms train-bin rejection is intentional even when delta_q is present at zero weight.

## Artifact 26 — Canonical archive status update for advanced head activation discipline
Artifact id: `archive-canonical-status-update-head-gates`
Source label: CLAIMS
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl:4-4`
Why it matters: Evidence-only row showing the archive's preserved understanding of the head-gate controller, density thresholds, gradient conflict tracking, and warmup protocol now implemented in repo.

## Artifact 27 — Canonical archive rows for open delta_q lane and surviving honest closure
Artifact id: `archive-canonical-deltaq-open-lane`
Source label: CLAIMS
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl:14-15`
Why it matters: Evidence-only rows that sharply separate what the current repo already closes from what still remains blocked for delta_q activation.

## Artifact 28 — Canonical archive roadmap narrowing for delta_q
Artifact id: `archive-roadmap-deltaq`
Source label: ARCHIVE
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:108-113`
Why it matters: Evidence-only summary of the surviving masked root-child q-delta lane and the requirement for validation-backed activation before broader use.

## Artifact 29 — Archive evidence for activation rule and validation blueprint
Artifact id: `answer23-activation-blueprint`
Source label: A23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:1064-1147`
Why it matters: Evidence-only packet with the strongest prior blueprint for missing-target behavior, activation rule, and validation structure.

## Artifact 30 — Archive evidence for audit window and enablement discipline
Artifact id: `answer23-enable-audit`
Source label: A23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:1148-1235`
Why it matters: Evidence-only support for the narrowest audit and enablement sequence, including support metrics and explicit keep-off rules.
</artifacts_manifest>]]>
  </prompt_text>
  </prompt_section>

  <config_section>
  <config_text status="preserved" source_path="embedded_generator_config">
  <![CDATA[{
  "version": 1,
  "repo_root": "../..",
  "defaults": {
    "title": "Hydra prompt — RL-only delta_q validation contract",
    "artifact_container_tag": "artifacts",
    "artifact_ids": [],
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
          "- include code-like detail when helpful (Rust preferred)",
          "- include enough detail that we can validate it ourselves",
          "- distinguish direct artifact support from your own inference",
          "- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked",
          "- confidence must be justified, not asserted"
        ]
      },
      {
        "tag": "artifact_note",
        "lines": [
          "The artifacts below reflect what the current codebase/docs appear to say right now.",
          "They are not guaranteed to be fully correct.",
          "Treat them as evidence to inspect and critique, not truth to inherit.",
          "Archive and handoff artifacts are evidence only and must not be promoted over current Hydra doctrine."
        ]
      }
    ]
  },
  "artifacts": [
    {
      "id": "readme-status",
      "type": "file_range",
      "path": "README.md",
      "start_line": 60,
      "end_line": 62,
      "label": "Current repo status summary for delta_q",
      "explanation": "Grounds the current staged status: narrow live self-play RL lane exists, while replay/offline production and train-bin activation remain blocked.",
      "source_label": "README",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "final-current-note",
      "type": "file_range",
      "path": "research/design/HYDRA_FINAL.md",
      "start_line": 119,
      "end_line": 121,
      "label": "Hydra north-star note on delta_q family",
      "explanation": "Shows that delta_q remains part of the search-distillation family but is not equally closed today.",
      "source_label": "FINAL",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "recon-deltaq-tranche",
      "type": "file_range",
      "path": "research/design/HYDRA_RECONCILIATION.md",
      "start_line": 448,
      "end_line": 489,
      "label": "Reconciliation tranche rules for delta_q, RL, and BC",
      "explanation": "This is the strongest current doctrine for delta_q staging, masked loss semantics, RL presence, and continued BC/train-bin blocking.",
      "source_label": "RECON",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "roadmap-exit-validation",
      "type": "file_range",
      "path": "research/design/IMPLEMENTATION_ROADMAP.md",
      "start_line": 775,
      "end_line": 783,
      "label": "Current ExIt validation harness and refreshed roadmap note",
      "explanation": "Provides the reusable in-repo validation pattern and current note that delta_q now exists only as a narrow RL lane while broader activation stays deferred.",
      "source_label": "ROADMAP",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "exit-deltaq-builder",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit.rs",
      "start_line": 263,
      "end_line": 357,
      "label": "delta_q target builder and collation code",
      "explanation": "Defines the current narrow object contract: masked discard-compatible root-child q-delta and batch collation into the [46] head space.",
      "source_label": "EXIT",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "live-shared-search-labels",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/live_exit.rs",
      "start_line": 240,
      "end_line": 363,
      "label": "Shared live search producer for ExIt and delta_q",
      "explanation": "Shows that delta_q is emitted by the same root-only live search envelope as ExIt, including upstream compatibility checks and shared provenance.",
      "source_label": "LIVE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "live-deltaq-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/live_exit.rs",
      "start_line": 758,
      "end_line": 778,
      "label": "delta_q live-producer test",
      "explanation": "Proves that good compatible input produces a real delta_q label with expected monotonic ordering over supported discard actions.",
      "source_label": "LIVE",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "arena-deltaq-validation",
      "type": "file_range",
      "path": "crates/hydra-core/src/arena.rs",
      "start_line": 559,
      "end_line": 599,
      "label": "Trajectory delta_q carrier validation",
      "explanation": "Shows strict structural invariants already enforced for carried delta_q labels: legal-only, discard-only, binary mask, zero outside support, and non-empty support.",
      "source_label": "ARENA",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "selfplay-rl-deltaq-carrier",
      "type": "file_range",
      "path": "crates/hydra-train/src/selfplay.rs",
      "start_line": 379,
      "end_line": 480,
      "label": "Self-play to RL batch delta_q carrier path",
      "explanation": "Shows the real live lane from trajectory labels into RlBatch.targets.delta_q_target and delta_q_mask.",
      "source_label": "SELFPLAY",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "losses-deltaq-masked",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/losses.rs",
      "start_line": 550,
      "end_line": 595,
      "label": "Masked delta_q loss semantics",
      "explanation": "Shows that delta_q only contributes loss when both target and mask exist, and that unsupported actions do not silently train through dense zeros.",
      "source_label": "LOSS",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "head-gates-core",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/head_gates.rs",
      "start_line": 1,
      "end_line": 90,
      "label": "HeadActivationController config and sparse-search thresholds",
      "explanation": "Contains the generic sparse-head activation thresholds that would apply to DeltaQ if the controller were used live.",
      "source_label": "GATES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "head-gates-deltaq-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/head_gates.rs",
      "start_line": 1053,
      "end_line": 1208,
      "label": "DeltaQ sparse-head gate tests and approved weight behavior",
      "explanation": "Shows concrete tested controller behavior for DeltaQ: sparse SPP rejection and approved_loss_config zeroing when the head remains off.",
      "source_label": "GATES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "orchestrator-live-exit-plan",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/orchestrator.rs",
      "start_line": 161,
      "end_line": 208,
      "label": "Maintenance plan and live ExIt enablement",
      "explanation": "Shows the existing validation-to-enable style for live ExIt producer phase gating, which may or may not transfer to delta_q without new doctrine.",
      "source_label": "ORCH",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "exit-validation-harness",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit_validation.rs",
      "start_line": 1,
      "end_line": 220,
      "label": "Existing ExIt validation harness",
      "explanation": "This is the strongest live validation blueprint in repo; use it to judge what a delta_q-specific validation harness would need and what is currently missing.",
      "source_label": "EVAL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "exit-validation-thresholds-and-report",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit_validation.rs",
      "start_line": 220,
      "end_line": 475,
      "label": "ExIt thresholds, report evaluation, and observational runner",
      "explanation": "Provides the concrete in-repo metric pack, pass/fail evaluator, and shadow validation runner that a delta_q-specific validator would most naturally mirror.",
      "source_label": "EVAL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "exit-validation-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/exit_validation.rs",
      "start_line": 560,
      "end_line": 811,
      "label": "ExIt validation harness tests",
      "explanation": "Shows the exact kinds of report-default, pass/fail, merge, and observational-run tests the repo already considers sufficient for a live validation harness.",
      "source_label": "EVAL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "replay-absence-guard",
      "type": "file_range",
      "path": "crates/hydra-train/src/data/mjai_loader.rs",
      "start_line": 437,
      "end_line": 459,
      "label": "Replay loader keeps delta_q absent",
      "explanation": "Shows replay/offline delta_q remains intentionally absent in the normal loader path.",
      "source_label": "LOADER",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "replay-absence-test",
      "type": "file_range",
      "path": "crates/hydra-train/src/data/mjai_loader.rs",
      "start_line": 784,
      "end_line": 792,
      "label": "Replay absence regression test",
      "explanation": "Proves replay samples must not quietly grow delta_q targets or masks.",
      "source_label": "LOADER",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "train-loss-policy-block",
      "type": "file_range",
      "path": "crates/hydra-train/src/bin/train/loss_policy.rs",
      "start_line": 1,
      "end_line": 33,
      "label": "Train-bin blocked advanced-loss policy",
      "explanation": "Shows BC/train-bin activation for delta_q remains intentionally rejected by policy.",
      "source_label": "TRAIN",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "train-deltaq-reject-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/bin/train.rs",
      "start_line": 1042,
      "end_line": 1060,
      "label": "delta_q rejection tests in train binary",
      "explanation": "Confirms train-bin rejection is intentional even when delta_q is present at zero weight.",
      "source_label": "TRAIN",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "head-gates-presence-and-coverage",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/head_gates.rs",
      "start_line": 90,
      "end_line": 509,
      "label": "DeltaQ classification, target presence, coverage, and conflict tracking",
      "explanation": "Shows how DeltaQ is classified as SparseSearch, how target presence is counted, and how coverage/conflict statistics are accumulated before any head can be approved.",
      "source_label": "GATES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "head-gates-controller-core",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/head_gates.rs",
      "start_line": 510,
      "end_line": 786,
      "label": "HeadActivationController core evaluation and approved loss config logic",
      "explanation": "Contains the actual controller state machine, per-head reports, activation attempts, and weight gating logic that a future delta_q enablement path would have to call.",
      "source_label": "GATES",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "archive-roadmap-deltaq",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md",
      "start_line": 108,
      "end_line": 113,
      "label": "Canonical archive roadmap narrowing for delta_q",
      "explanation": "Evidence-only summary of the surviving masked root-child q-delta lane and the requirement for validation-backed activation before broader use.",
      "source_label": "ARCHIVE",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer23-activation-blueprint",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_23_combined.md",
      "start_line": 1064,
      "end_line": 1147,
      "label": "Archive evidence for activation rule and validation blueprint",
      "explanation": "Evidence-only packet with the strongest prior blueprint for missing-target behavior, activation rule, and validation structure.",
      "source_label": "A23",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "answer23-enable-audit",
      "type": "file_range",
      "path": "research/agent_handoffs/combined_all_variants/answer_23_combined.md",
      "start_line": 1148,
      "end_line": 1235,
      "label": "Archive evidence for audit window and enablement discipline",
      "explanation": "Evidence-only support for the narrowest audit and enablement sequence, including support metrics and explicit keep-off rules.",
      "source_label": "A23",
      "fence_language": "text",
      "show_line_numbers": true
    },
    {
      "id": "archive-canonical-status-update-head-gates",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl",
      "start_line": 4,
      "end_line": 4,
      "label": "Canonical archive status update for advanced head activation discipline",
      "explanation": "Evidence-only row showing the archive's preserved understanding of the head-gate controller, density thresholds, gradient conflict tracking, and warmup protocol now implemented in repo.",
      "source_label": "CLAIMS",
      "fence_language": "json",
      "show_line_numbers": true
    },
    {
      "id": "archive-canonical-deltaq-open-lane",
      "type": "file_range",
      "path": "research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl",
      "start_line": 14,
      "end_line": 15,
      "label": "Canonical archive rows for open delta_q lane and surviving honest closure",
      "explanation": "Evidence-only rows that sharply separate what the current repo already closes from what still remains blocked for delta_q activation.",
      "source_label": "CLAIMS",
      "fence_language": "json",
      "show_line_numbers": true
    },
    {
      "id": "rl-deltaq-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/rl.rs",
      "start_line": 260,
      "end_line": 389,
      "label": "RL tests with advanced aux targets including delta_q",
      "explanation": "Shows that RL-side delta_q consumption already exists in library code and remains numerically stable in combined auxiliary-loss scenarios.",
      "source_label": "RL",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "integration-deltaq-batch-proof",
      "type": "file_range",
      "path": "crates/hydra-train/tests/integration_pipeline.rs",
      "start_line": 149,
      "end_line": 250,
      "label": "Integration proof that trajectory delta_q labels survive into RL batches",
      "explanation": "End-to-end evidence that the live lane is real: trajectory labels persist into RlBatch.targets with the expected values and masks.",
      "source_label": "INTEG",
      "fence_language": "rust",
      "show_line_numbers": true
    },
    {
      "id": "losses-optional-and-activation-tests",
      "type": "file_range",
      "path": "crates/hydra-train/src/training/losses.rs",
      "start_line": 782,
      "end_line": 899,
      "label": "Loss tests for missing targets and activated optional losses",
      "explanation": "Shows current repo expectations around zero loss when optional targets are absent and finite aux loss when optional targets are present, including delta_q in the mixed optional-target tests.",
      "source_label": "LOSS",
      "fence_language": "rust",
      "show_line_numbers": true
    }
  ],
  "variants": [
    {
      "name": "rl-delta-q-validation-contract",
      "title": "Hydra prompt — RL-only delta_q validation-and-enable contract",
      "output_file": "delta_q_rl_validation_contract.md",
      "artifact_ids": [
        "readme-status",
        "final-current-note",
        "recon-deltaq-tranche",
        "roadmap-exit-validation",
        "exit-deltaq-builder",
        "live-shared-search-labels",
        "live-deltaq-tests",
        "arena-deltaq-validation",
        "selfplay-rl-deltaq-carrier",
        "losses-deltaq-masked",
        "losses-optional-and-activation-tests",
        "head-gates-core",
        "head-gates-presence-and-coverage",
        "head-gates-controller-core",
        "head-gates-deltaq-tests",
        "orchestrator-live-exit-plan",
        "exit-validation-harness",
        "exit-validation-thresholds-and-report",
        "exit-validation-tests",
        "rl-deltaq-tests",
        "integration-deltaq-batch-proof",
        "replay-absence-guard",
        "replay-absence-test",
        "train-loss-policy-block",
        "train-deltaq-reject-tests",
        "archive-canonical-status-update-head-gates",
        "archive-canonical-deltaq-open-lane",
        "archive-roadmap-deltaq",
        "answer23-activation-blueprint",
        "answer23-enable-audit"
      ],
      "shell_sections": [
        {
          "tag": "direction",
          "lines": [
            "Work toward the strongest exact blueprint for Hydra's RL-only `delta_q` validation-and-enable tranche.",
            "We want a detailed answer that makes clear:",
            "- the narrowest doctrine-consistent RL-only `delta_q` validation contract",
            "- exact metrics and thresholds that should govern whether RL-side `delta_q` is trusted enough to enable",
            "- fail behavior and keep-off behavior",
            "- the exact controller/orchestrator hookup path",
            "- what must stay narrow / deferred / rejected",
            "- how to implement and validate the surviving path with minimal guesswork",
            "- what is directly supported vs inferred vs still blocked",
            "Use the artifacts below to derive your conclusions.",
            "Do not broaden scope into replay/offline `delta_q` or BC/train-bin activation unless an artifact forces it."
          ]
        },
        {
          "tag": "scope_note",
          "lines": [
            "This is not a broad `delta_q` design prompt.",
            "The semantics of the current narrow lane are already mostly settled:",
            "- learner-only",
            "- root-only",
            "- search-derived",
            "- discard-compatible",
            "- masked `[46]`",
            "- `Q(child) - Q(root)`",
            "- RL/self-play lane only",
            "- replay/offline absent",
            "- BC/train-bin activation blocked",
            "The question is narrower:",
            "What is the strongest repo-grounded RL-only validation-and-enable contract for this lane?"
          ]
        },
        {
          "tag": "deliverable",
          "lines": [
            "Give a blueprint, not a memo.",
            "",
            "Your answer must include:",
            "1. direct support / inference / proposal / blocked buckets",
            "2. surviving RL-only `delta_q` validation contract",
            "3. minimal implementation plan",
            "4. hookup design",
            "5. validation plan",
            "6. confidence justification",
            "7. final decision: either 'repo-backed enough to implement now' or 'still blocked, and here is the smallest unresolved fact'"
          ]
        }
      ]
    }
  ]
}]]>
  </config_text>
  </config_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_25.md">
  <![CDATA[Use the artifact tranche as the branch authority for this work. The public GitHub snapshot still looks pre-tranche: its README still says `delta_q` closure is an immediate need, public `live_exit.rs` is ExIt-only, public `selfplay.rs` still writes `delta_q_target: None`, and public `losses.rs` still shows dense unmasked `delta_q` regression, while the sparse-head controller and the train-bin `advanced_loss.delta_q` rejection are already present there. Treat the artifacts as a newer working tree or branch under review, not as what web indexing currently sees. ([GitHub][1])

## 1. Buckets

### 1.1 Direct support

* `delta_q` survives only in the narrow live self-play RL lane. Replay/offline production stays absent, and BC / `train.rs` activation stays blocked. That is explicit in Artifacts 1, 2, 3, 22, 23, 24, 25.
* The surviving object is exactly a masked `[46]` target over the training head space, but only canonical discard-compatible actions may be supported. For each supported child:
  [
  \delta q(a) = Q(\text{child}_a) - Q(\text{root})
  ]
  and unsupported actions are zero-target / zero-mask. Root and supported children must have finite `node_q_value`, and children must have `visit_count > 0`. If no supported child survives, the whole label is `None`. That is direct from Artifacts 5 and 8.
* The producer is root-only, learner-only, and shared with ExIt. The same root-search envelope gates both labels: compatible discard state, at least 2 legal discards, raw-logit base policy, hard-state gate, root-only AFBS, cached child values, then shared label construction. That is direct from Artifact 6.
* The carrier path is real end-to-end in the artifact tranche:
  `TrajectoryStep.delta_q_label`
  → `collate_delta_q_targets`
  → `RlBatch.targets.{delta_q_target, delta_q_mask}`
  → masked delta-q loss. That is direct from Artifacts 8, 9, 10, 21.
* The activation controller already exists and already classifies `DeltaQ` as a sparse search head with exact thresholds:

  * `min_sparse_spp = 5.0`
  * `max_negative_frac = 0.3`
  * `warmup_steps = 10_000`
  * `min_eval_samples = 1000`
  * `min_conflict_checks = 10`
    and `approved_loss_config()` already zeroes `w_delta_q` while the head is `Off`. That is direct from Artifacts 12, 13, 14, 15.
* The current orchestrator only phase-gates the shared live search producer through the ExIt maintenance plan; controller integration and trunk detachment are still not wired. That is direct from Artifacts 16 and 26.

### 1.2 Inference

* **Critical fix before trusting any controller statistic:** `extract_target_presence()` as shown in Artifact 13 is wrong for mixed `delta_q` batches if it counts `batch_size` whenever `delta_q_target.is_some()`. Artifact 5’s collation returns `Some(target_tensor), Some(mask_tensor)` for the whole batch whenever *any* row is present, while absent rows are encoded as all-zero masks. Therefore the correct per-batch count is
  [
  examples_present = \sum_{i=1}^{B} \mathbf{1}!\left[\sum_{a=0}^{45} m_{i,a} > 0\right],
  ]
  not `batch_size`. Without this fix, `spp_h` is inflated and `DeltaQ` can be approved too early.
* The valid pair states for RL training must be:

  * `(None, None)` → absent, no delta-q loss.
  * `(Some(target), Some(mask))` → present, masked delta-q loss.
    The invalid pair states
  * `(Some(_), None)`
  * `(None, Some(_))`
    must be treated as plumbing failure, not silently as zero loss. That follows from Artifact 3’s rule that target presence must control whether an advanced loss exists at all, plus Artifact 10’s intended masked semantics.
* The support audit should reuse ExIt’s structural support thresholds, because the shared producer envelope is the same and the same coverage notion exists here, but **must not** reuse ExIt’s KL or base-policy top-1 agreement gates. `delta_q` is not a policy teacher; its fidelity has to be checked against a deeper `delta_q` reference, not against the base policy. This is the strongest doctrine-consistent reuse of Artifacts 4, 17, 18, 29, 30.
* In the current branch state, “enable `delta_q`” should mean **allow nonzero RL loss through the controller**, not “invent a second producer.” The producer/carrier lane already exists in the artifact tranche.

### 1.3 Proposal

* Add a dedicated `delta_q_validation.rs` shadow harness instead of overloading `exit_validation.rs`.
* Keep the producer shadow-on in search-enabled RL phases even while the head is `Off`, so validation data keeps accumulating.
* Add a small `OptionalTargetStats`/`DeltaQBatchStats` logging helper rather than mutating `LossBreakdown` into a giant metrics bag.
* Add a model forward option to detach trunk features for `DeltaQ` during controller warmup.

### 1.4 Blocked

* No repo-backed exact numeric promotion threshold exists yet for the deeper-reference `delta_q` metrics:

  * sign agreement
  * best-discard agreement
  * masked-entry MAE
    Current doctrine requires the benchmark to be run and reviewed, but does not canonize auto-pass numbers. That is the smallest unresolved fact.
* No canonized RL `w_delta_q` magnitude is provided by the artifacts. Default stays `0.0`. Candidate nonzero weight remains a config choice gated by the controller.
* Do **not** broaden this tranche into replay/offline `delta_q`, BC consumption, or `train.rs` acceptance.
* Do **not** change the canonical object from `visit_count > 0` support to `visit_count >= 4`. `>= 4` is validation-tuned experimentation only, not the object contract.

---

## 2. Surviving RL-only `delta_q` validation contract

### 2.1 Object contract

For one decision state (s), let:

* (A = 46) be the training head action space.
* (D(s) = {a \mid a \le \texttt{DISCARD_END},\ \texttt{legal}[a]=1}) be legal canonical discard slots.
* (m_a \in {0,1}) be the support mask.
* (q_r = Q(\text{root})).
* (q_a = Q(\text{child}_a)) for a root child action (a).

Emit a `delta_q` label iff all of these hold:

1. `compatible_discard_state(legal_mask)` is true.
2. `|D(s)| >= 2`.
3. `top2_policy_gap(base_pi[D(s)]) < 0.10`.
4. `root.visit_count > 0`.
5. `q_r` is finite.
6. At least one root child satisfies:

   * `a ∈ D(s)`
   * `a` is canonical discard only
   * `a` is not aka
   * `child.visit_count > 0`
   * `q_a` is finite.

Then define, for each action (a \in {0,\dots,45}),
[
m_a =
\begin{cases}
1 & \text{if supported child for } a \text{ survives}\
0 & \text{otherwise}
\end{cases}
]
and
[
\delta q_a =
\begin{cases}
q_a - q_r & \text{if } m_a = 1\
0 & \text{if } m_a = 0
\end{cases}
]

If no action survives, emit `None`.

Keep this exact. Do not normalize, do not temperature-softmax it, do not densify unsupported actions, and do not map the runtime `[34]` plane into training supervision.

### 2.2 Carrier contract

Keep the carrier narrow:

* producer source: shared root-search producer only
* provenance basis: same in-memory search run, same legality basis, same step
* lane: self-play RL only
* replay/offline: absent
* BC/train-bin: blocked
* non-discard actions: masked out
* aka discards: rejected
* unsupported rows in collation: zero-target / zero-mask row

### 2.3 Loss contract

Use only masked regression:
[
L_{\Delta Q}
============

\frac{1}{\max!\left(1,\sum_{i=1}^{B}\sum_{a=0}^{45} m_{i,a}\right)}
\sum_{i=1}^{B}\sum_{a=0}^{45}
m_{i,a}\left(\hat{\delta q}*{i,a} - \delta q*{i,a}\right)^2
]

Rules:

* `(None, None)` → loss term does not exist.
* `(Some(target), Some(mask))` → masked MSE.
* `(Some, None)` or `(None, Some)` → invalid batch state:

  * log error
  * `force_off(DeltaQ)`
  * effective `w_delta_q = 0` for that step
  * continue baseline RL training
* all-zero mask → delta-q loss must evaluate to zero

### 2.4 Activation contract

Use controller state exactly as follows.

#### `HeadState::Off`

* delta-q labels may still be present in RL batches
* `approved_loss_config()` forces `w_delta_q = 0`
* support statistics still accumulate
* this is the default state

#### `HeadState::Warmup`

Enter only when all external prereqs are true:

1. producer/carrier exists in RL self-play (already true in this tranche)
2. object/collation/loss/integration tests are green
3. support audit passes
4. deeper-reference benchmark has been run and reviewed as acceptable
5. unchanged global benchmark gates pass
6. controller density gate passes:
   [
   total_samples \ge 1000,\qquad
   spp_{\Delta Q} = \frac{labeled_samples_{\Delta Q}}{learner_params} \ge 5.0
   ]

During warmup:

* delta-q loss weight may be nonzero after controller gating
* trunk features for the `DeltaQ` head must be detached
* warmup length is exactly `10_000` updates
* collect real shared-trunk gradient cosine checks for delta-q vs policy+value

#### `HeadState::Active`

Promote from `Warmup` only when:

* warmup countdown reached zero
* at least `10` cosine checks have been recorded
* negative cosine fraction is at most `0.30`:
  [
  neg_frac_{\Delta Q}
  ===================

  \frac{1}{K}
  \sum_{k=1}^{K}
  \mathbf{1}!\left[\cos(g_k^{\Delta Q}, g_k^{PV}) < 0\right]
  \le 0.30
  ]

If the conflict gate fails, revert to `Off`.

### 2.5 Keep-off behavior

Keep `DeltaQ` loss **off** when any of these holds:

* current phase does not have the shared live search producer enabled
* support audit fails
* deeper-reference benchmark not run
* deeper-reference benchmark run but not yet manually reviewed pass
* controller density gate fails
* warmup conflict gate fails
* target/mask pair invalid
* replay/offline path tries to populate `delta_q`
* BC/train-bin path tries to activate `delta_q`
* unchanged global benchmark gates fail

Keep-off means:

* `HeadState::Off`
* `approved_loss_config().w_delta_q = 0.0`
* shadow carrier may remain on in RL search-enabled phases
* baseline RL training continues

---

## 3. Minimal implementation plan

### 3.1 `crates/hydra-train/src/training/head_gates.rs`

**Mandatory first fix**

Replace delta-q target presence extraction with mask-row counting.

```rust
fn count_nonzero_rows_2d<B: Backend>(tensor: &Tensor<B, 2>) -> usize {
    let [_rows, cols] = tensor.dims();
    match tensor.to_data().as_slice::<f32>() {
        Ok(data) => data.chunks(cols).filter(|row| row.iter().any(|&v| v > 0.0)).count(),
        Err(_) => 0,
    }
}

pub fn extract_target_presence<B: Backend>(targets: &HydraTargets<B>) -> TargetPresence {
    let batch_size = targets.policy_target.dims()[0];
    let mut counts = [0usize; NUM_ADVANCED_HEADS];

    // ... existing heads ...

    counts[AdvancedHead::DeltaQ.index()] = match (&targets.delta_q_target, &targets.delta_q_mask) {
        (Some(_target), Some(mask)) => count_nonzero_rows_2d(mask),
        (None, None) => 0,
        _ => 0, // invalid pair; surfaced by explicit pair validation upstream
    };

    // ... safety residual unchanged ...
    TargetPresence { counts, batch_size }
}
```

Add tests:

* `extract_target_presence_counts_only_nonzero_delta_q_rows`
* `extract_target_presence_delta_q_invalid_pair_counts_zero`
* keep existing sparse SPP tests, but make them mask-row based

Reason: the current collation semantics can create mixed batches with some absent rows even when the batch tensors are `Some(...)`.

### 3.2 `crates/hydra-train/src/training/losses.rs`

Add strict optional-pair validation and debug stats.

```rust
#[derive(Debug, Clone, Default)]
pub struct DeltaQBatchStats {
    pub examples_present: usize,
    pub actions_present: usize,
    pub examples_absent: usize,
}

impl DeltaQBatchStats {
    pub fn from_targets<B: Backend>(targets: &HydraTargets<B>) -> Result<Self, String> {
        match (&targets.delta_q_target, &targets.delta_q_mask) {
            (None, None) => Ok(Self {
                examples_present: 0,
                actions_present: 0,
                examples_absent: targets.policy_target.dims()[0],
            }),
            (Some(_), Some(mask)) => {
                let [batch, cols] = mask.dims();
                let data = mask
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|_| "delta_q mask unreadable".to_string())?;
                let mut examples_present = 0usize;
                let mut actions_present = 0usize;
                for row in data.chunks(cols).take(batch) {
                    let row_actions = row.iter().filter(|&&v| v > 0.0).count();
                    if row_actions > 0 {
                        examples_present += 1;
                        actions_present += row_actions;
                    }
                }
                Ok(Self {
                    examples_present,
                    actions_present,
                    examples_absent: batch - examples_present,
                })
            }
            _ => Err("delta_q target/mask mismatch".to_string()),
        }
    }
}
```

Do **not** rely on `_ => zero.clone()` to hide broken delta-q plumbing. Validate upstream and log:

* `delta_q_examples_present`
* `delta_q_actions_present`
* `delta_q_examples_absent`
* `delta_q_loss`

Add tests:

* `delta_q_requires_mask`
* `delta_q_masked_loss_ignores_unmasked_positions`
* `delta_q_all_zero_mask_zeroes_loss`

Also add one explicit regression test for the artifact tension:

* `delta_q_target_without_mask_is_invalid_even_if_other_aux_targets_are_present`

### 3.3 `crates/hydra-train/src/training/delta_q_validation.rs` (new)

Add a dedicated RL-only shadow validator.

Core report:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQValidationReport {
    pub total_states: u64,
    pub compatible_discard_states: u64,
    pub hard_states: u64,
    pub labels_emitted: u64,
    pub labels_rejected: u64,
    pub rejected_incompatible_state: u64,
    pub rejected_too_few_discards: u64,
    pub rejected_not_hard_state: u64,
    pub rejected_child_obs_failure: u64,
    pub rejected_other: u64,

    pub coverage_sum: f64,
    pub supported_actions_sum: u64,
    pub root_visits_sum: u64,

    pub masked_abs_sum: f64,
    pub masked_zero_count: u64,
    pub masked_positive_count: u64,
    pub masked_negative_count: u64,
    pub masked_entry_count: u64,
    pub masked_min: Option<f32>,
    pub masked_max: Option<f32>,

    pub ref_compared_states: u64,
    pub ref_intersection_actions: u64,
    pub ref_sign_agreement_count: u64,
    pub ref_best_discard_agreement_count: u64,
    pub ref_mae_sum: f64,
}
```

Derived metrics:

[
emission_rate = \frac{labels_emitted}{total_states}
]
[
mean_coverage = \frac{1}{labels_emitted}\sum_i \frac{#{a: m_{i,a}=1}}{|D(s_i)|}
]
[
mean_supported_actions = \frac{supported_actions_sum}{labels_emitted}
]
[
mean_abs = \frac{masked_abs_sum}{masked_entry_count}
]
[
frac_zero = \frac{masked_zero_count}{masked_entry_count}
]
[
frac_positive = \frac{masked_positive_count}{masked_entry_count}
]
[
frac_negative = \frac{masked_negative_count}{masked_entry_count}
]
[
ref_sign_agreement =
\frac{ref_sign_agreement_count}{ref_intersection_actions}
]
[
ref_best_discard_agreement =
\frac{ref_best_discard_agreement_count}{ref_compared_states}
]
[
ref_mae = \frac{ref_mae_sum}{ref_intersection_actions}
]

Thresholds:

```rust
#[derive(Debug, Clone)]
pub struct DeltaQValidationThresholds {
    pub min_sample_size: u64,              // 1000
    pub min_emission_rate: f64,            // 0.01
    pub min_mean_coverage: f64,            // 0.70
    pub min_mean_supported_actions: f64,   // 3.0
    pub require_reference_benchmark: bool, // true
}
```

Use the exact defaults above. The first four are the strongest narrow reuse of the in-repo ExIt harness because the producer envelope and support geometry are shared; the fifth is direct doctrine from Artifact 30.

**Important:** do not create KL or base-policy top-1 thresholds for delta-q.

### 3.4 Deeper-reference comparison

Implement reference comparison only in the validation runner, not in the live producer.

Keep the live label untouched. For validation only, compute a second label on the same decision with a strictly deeper learner-only AFBS run. Compare on the mask intersection:

[
I_i = {a \mid m^{live}*{i,a}=1 \land m^{ref}*{i,a}=1}
]

Then:

[
sign_agree(i,a) =
\mathbf{1}!\left[\operatorname{sgn}(\delta q^{live}*{i,a}) =
\operatorname{sgn}(\delta q^{ref}*{i,a})\right]
]

[
best(i) =
\arg\max_{a \in I_i} \delta q_{i,a}
]

[
mae(i,a) = |\delta q^{live}*{i,a} - \delta q^{ref}*{i,a}|
]

Do **not** auto-pass/fail on these numbers yet. Store them, print them, and require explicit review.

### 3.5 `crates/hydra-train/src/training/rl.rs`

Wire controller use into RL only.

Per RL step:

1. validate optional target pairs
2. record corrected target presence
3. compute effective loss config through `approved_loss_config`
4. detach trunk for warmup heads
5. compute and log `DeltaQBatchStats`
6. if `DeltaQ` is in warmup, collect real gradient cosine checks and `tick_warmup()`

Skeleton:

```rust
pub fn rl_phase_train_step<B: AutodiffBackend>(
    ctx: &mut RlTrainCtx<B>,
    batch: &RlBatch<B>,
) -> Result<f32, String> {
    if let Err(err) = batch.targets.validate_optional_pairs() {
        ctx.head_ctrl.force_off(AdvancedHead::DeltaQ);
        ctx.last_delta_q_pair_error = Some(err.clone());
    }

    let presence = extract_target_presence(&batch.targets);
    ctx.head_ctrl.record_batch(&presence);

    let delta_q_stats = DeltaQBatchStats::from_targets(&batch.targets)?;
    ctx.logger.log_delta_q_stats(&delta_q_stats);

    let effective_loss_cfg = ctx.head_ctrl.approved_loss_config(&ctx.base_loss_cfg);
    let warmup_heads = ctx.head_ctrl.warmup_heads();
    let outputs = ctx.model.forward_with_options(
        batch.obs.clone(),
        ForwardOptions::detach_for(&warmup_heads),
    );

    let loss_fn = HydraLoss::<B>::new(effective_loss_cfg);
    let breakdown = loss_fn.total_loss(&outputs, &batch.targets);

    if ctx.head_ctrl.head_state(AdvancedHead::DeltaQ) == HeadState::Warmup {
        let cos = measure_delta_q_grad_cosine(ctx, batch)?;
        ctx.head_ctrl.record_grad_cosine(AdvancedHead::DeltaQ, cos);
        ctx.head_ctrl.tick_warmup();
    }

    Ok(breakdown.total.into_scalar().elem())
}
```

Add tests:

* `rl_batch_mixed_cases_handle_delta_q`

  * baseline only
  * baseline + delta_q
  * baseline + exit + delta_q
* `rl_delta_q_invalid_pair_forces_off`
* `rl_delta_q_warmup_uses_detached_trunk`

Do **not** add BC delta-q consumption tests.

### 3.6 `crates/hydra-train/src/training/orchestrator.rs`

Do not invent a separate delta-q search phase. Piggyback on the existing shared search-enabled phases.

Use the existing plan:

* `DrdaAchSelfPlay` after `phase_progress > 0.5`
* `ExitPondering` always

as the only phases where delta-q shadow carriage is allowed, because those are already the phases where the shared live search producer is enabled.

Add a narrow validation/approval state:

```rust
pub struct DeltaQValidationState {
    pub support_report_passed: bool,
    pub reference_review_passed: bool,
    pub global_benchmarks_passed: bool,
}
```

On maintenance tick:

```rust
fn maybe_try_activate_delta_q(
    head_ctrl: &mut HeadActivationController,
    v: &DeltaQValidationState,
) {
    let ready =
        v.support_report_passed &&
        v.reference_review_passed &&
        v.global_benchmarks_passed;

    if ready {
        let _ = head_ctrl.try_activate(AdvancedHead::DeltaQ);
    } else {
        head_ctrl.force_off(AdvancedHead::DeltaQ);
    }
}
```

This is the exact orchestrator rule. No BC path. No replay path. No train-bin path.

### 3.7 `crates/hydra-train/src/model.rs` or equivalent forward path

Add only the minimal warmup detachment hook.

```rust
pub struct ForwardOptions {
    pub detach_for: std::collections::HashSet<AdvancedHead>,
}

pub fn forward_with_options<B: AutodiffBackend>(
    &self,
    x: Tensor<B, 3>,
    opts: &ForwardOptions,
) -> HydraOutputs<B> {
    let trunk = self.backbone.forward(x);
    let trunk_detached = trunk.clone().detach();

    let trunk_for = |head: AdvancedHead| {
        if opts.detach_for.contains(&head) {
            trunk_detached.clone()
        } else {
            trunk.clone()
        }
    };

    HydraOutputs {
        policy_logits: self.policy_head.forward(trunk.clone()),
        value: self.value_head.forward(trunk.clone()),
        // ...
        delta_q: self.delta_q_head.forward(trunk_for(AdvancedHead::DeltaQ)),
        safety_residual: self.safety_residual_head.forward(trunk_for(AdvancedHead::SafetyResidual)),
        // ...
    }
}
```

Do not detach policy/value. Do not detach unrelated heads.

---

## 4. Hookup design

### 4.1 Data path

Keep exactly this path:

```text
maintenance_plan
  -> live search-enabled phases only
  -> shared root-search producer
  -> TrajectorySearchLabels { exit, delta_q }
  -> TrajectoryStep.delta_q_label
  -> collate_delta_q_targets
  -> RlBatch.targets.{delta_q_target, delta_q_mask}
  -> extract_target_presence (fixed)
  -> HeadActivationController.record_batch
  -> approved_loss_config(base_loss_cfg)
  -> HydraLoss(masked delta_q MSE)
```

### 4.2 Approval path

Keep exactly this path:

```text
shadow validation run
  + deeper learner-only AFBS comparison
  + unchanged global benchmark gates
  -> DeltaQValidationState
  -> maybe_try_activate_delta_q()
  -> HeadState::Warmup
  -> real gradient cosine checks during warmup
  -> HeadState::Active or back to Off
```

### 4.3 Do not broaden the producer

Do not add:

* a replay/search sidecar builder
* a delta-q-only search call
* a second legality basis
* a runtime `[34]` → training `[46]` densifier
* BC/train-bin acceptance

The shared search producer is the provenance contract for this tranche.

---

## 5. Validation plan

### 5.1 Object tests

Add or confirm:

1. `delta_q_v1_matches_afbs_root_child_delta`
2. `delta_q_v1_masks_only_supported_canonical_discard_slots`
3. `delta_q_v1_omits_unvisited_children`
4. `delta_q_v1_returns_none_when_no_valid_support`
5. `delta_q_v1_rejects_nonfinite_q`

### 5.2 Batch/loss tests

Add:

6. `delta_q_roundtrips_through_collation`
7. `delta_q_requires_mask`
8. `delta_q_masked_loss_ignores_unmasked_positions`
9. `delta_q_all_zero_mask_zeroes_loss`
10. `extract_target_presence_counts_only_nonzero_delta_q_rows`

### 5.3 RL integration tests

Add:

11. `shared_root_search_produces_exit_and_delta_q_from_same_tree`
12. `rl_batch_mixed_cases_handle_delta_q`
13. `rl_delta_q_force_off_on_invalid_pair`
14. `rl_delta_q_warmup_detaches_trunk`

Keep existing regressions:

15. replay loader keeps delta-q absent
16. `train.rs` still rejects `advanced_loss.delta_q`, even at zero

### 5.4 Shadow validation thresholds

Use these exact gates for the support audit:

* `sample_size >= 1000`
* `emission_rate >= 0.01`
* `mean_coverage >= 0.70`
* `mean_supported_actions >= 3.0`

Use these exact structural conditions:

* hard-state slice is `top2_policy_gap < 0.10`
* producer remains root-only, learner-only, shared with ExIt
* support contract remains `visit_count > 0`, not `>= 4`

Use these exact controller gates:

* `spp_delta_q >= 5.0`
* `warmup_steps = 10_000`
* `min_conflict_checks = 10`
* `negative_frac <= 0.30`

Use the unchanged global benchmark gates:

* `afbs_on_turn_ms < 150`
* `ct_smc_dp_ms < 1`
* `endgame_ms < 100`
* `self_play_games_per_sec > 20`
* `distill_kl_drift < 0.1`

### 5.5 Reference benchmark

Require it, report it, review it.

At minimum report:

* `ref_sign_agreement`
* `ref_best_discard_agreement`
* `ref_mae`
* `examples_present`
* `actions_present`
* `mean_abs`
* `min`
* `max`
* `frac_zero`
* `frac_positive`
* `frac_negative`

Do **not** auto-enable on numeric thresholds for these yet. Current doctrine does not close that fact.

---

## 6. Confidence justification

High confidence on the narrow object, carrier, and keep-off boundaries:

* multiple artifacts agree on RL-only scope, shared search provenance, masked `[46]` semantics, replay absence, and train-bin rejection
* the controller thresholds are exact and already implemented
* the key file surfaces are localized

Medium confidence on the support-audit thresholds:

* the `1% / 0.70 / 3 / 1000` pack is the strongest narrow reuse of the in-repo ExIt validator, but it is still reuse, not a delta-q-specific SSOT line

High confidence that the `extract_target_presence` fix is necessary:

* it follows mechanically from the artifact collation semantics and the controller’s current counting rule

Lower confidence on fully automatic final promotion:

* the artifacts preserve the *metrics* for deeper-reference comparison, but not exact numeric pass thresholds
* a few artifact snippets showing mixed aux tests do not visibly show `delta_q_mask`, so explicit pair-validation tests should be treated as mandatory closure work, not optional cleanup

---

## 7. Final decision

**repo-backed enough to implement now**

Implement the RL-only validation tranche now, with this exact boundary:

* carrier/shadow lane stays narrow and RL-only
* replay/BC/train-bin stay blocked
* controller math is fixed first
* support audit + unchanged benchmark gates are exact pass/fail
* deeper-reference comparison is mandatory but remains manual-review-gated until numeric promotion thresholds are canonized

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/README.md"]]>
  </answer_text>
  </answer_section>
</combined_run_record>
