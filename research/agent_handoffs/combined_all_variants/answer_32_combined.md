<combined_run_record run_id="answer_32" variant_id="safetyresidual-promotion-standard" schema_version="1">
  <metadata>
    <notes>Self-contained combined record for Agent 32. It preserves the compact prompt shell and artifact manifest generated from the authoritative prompt config, plus the preserved answer text.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — elevate SafetyResidual to DeltaQ-grade promotion discipline

<role>
Example role placeholder.
Replace this with the role that fits your actual prompt.
Keep it short and task-specific.
You are designing a narrow promotion/validation blueprint for an already-admitted auxiliary lane.
</role>

<task>
Example task placeholder.
Replace this with the actual job the agent should do.

An example task block might ask for:
- what the artifacts directly support
- what is only inference
- what confidence level each major conclusion deserves
- what simpler or stronger local alternatives exist inside the same lane
- what should be kept, narrowed, or removed
- why the confident parts of the answer are actually justified
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
Determine the strongest exact blueprint for bringing SafetyResidual up to the same measured promotion discipline as DeltaQ.

We want a detailed answer that makes clear:
- what SafetyResidual already has today (labels, loss wiring, admission, routing)
- what it still lacks compared with DeltaQ (validation harness, runner logging, promotion criteria, or semantics)
- whether the right next move is to clone DeltaQ-style structure, reuse ExIt-style behavioral criteria, or define a narrower SafetyResidual-specific gate
- what exact files and tests should change first
- what should remain deliberately narrower than DeltaQ, if anything

Do not widen into belief or opponent-target work. Use the artifacts below to derive a validation-ready blueprint.
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

<artifacts_manifest>

## Artifact 01 — Prompt-packing reminder
Artifact id: `prompt-packing-reminder`
Type: `literal`
Why it matters: Task-specific reminder consistent with the prompt style guide.

## Artifact 02 — Repo status snapshot
Artifact id: `repo-status`
Source label: README
Type: `file_range`
Source: `README.md:70-72`
Why it matters: Current repo truth on which lanes are already shipped versus still selective or staged.

## Artifact 03 — Promoted doctrine: file-by-file first tranche coding spec
Artifact id: `recon-first-tranche-spec`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:420-544`
Why it matters: Concrete promoted coding spec showing the live DeltaQ and SafetyResidual closure context, acceptance checklist, and no-new-heads rule.

## Artifact 04 — Promoted doctrine: SafetyResidual semantics and staging
Artifact id: `safetyresidual-doctrine`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:434-453`
Why it matters: Promoted doctrine for signed replay-derived SafetyResidual semantics and the rule against drifting it into search-derived semantics.

## Artifact 05 — Implementation roadmap live snapshot note
Artifact id: `impl-roadmap-live-snapshot`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:12-24`
Why it matters: Reference-only live snapshot clarifying current crate surfaces and which advanced lanes are already shipped or staged.

## Artifact 06 — Implementation roadmap advanced-head activation note
Artifact id: `impl-roadmap-advanced-head-note`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:178-199`
Why it matters: Reference note on which advanced heads are structurally live and which carriers are active or staged.

## Artifact 07 — Live code: allowed advanced losses in train entrypoint
Artifact id: `loss-policy-allowed-heads`
Source label: LOSSPOL
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs:1-61`
Why it matters: Shows DeltaQ and SafetyResidual are admitted while belief/mixture/opponent-hand-type remain blocked.

## Artifact 08 — Live code: DeltaQ and SafetyResidual loss terms
Artifact id: `losses-deltaq-safety`
Source label: LOSSES
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:620-679`
Why it matters: Current masked regression wiring for DeltaQ and SafetyResidual inside the shared loss breakdown.

## Artifact 09 — Loss breakdown and optional-head contribution surface
Artifact id: `loss-breakdown-surface`
Source label: LBD
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:520-679`
Why it matters: Shared loss breakdown surface showing how optional advanced targets contribute and remain zero when absent.

## Artifact 10 — Optional-head and SafetyResidual loss tests
Artifact id: `loss-optional-and-safety-tests`
Source label: LOPT
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:947-1096`
Why it matters: Dense unit-test surface covering missing-target zeroing, optional-head activation, and SafetyResidual-specific mask semantics.

## Artifact 11 — Live code: pooled-vs-spatial head routing
Artifact id: `model-routing-live`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:419-438`
Why it matters: Current forward path showing pooled routing for policy, DeltaQ, and SafetyResidual versus spatial routing for tile-structured heads.

## Artifact 12 — Live code: sample carriers and batch collation for ExIt and DeltaQ
Artifact id: `sample-carriers-deltaq-exit`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:260-419`
Why it matters: Shows action-vector augmentation, target/mask pair safety, and batch collation for ExIt, DeltaQ, and adjacent advanced targets.

## Artifact 13 — Sample tests for SafetyResidual and DeltaQ carriage
Artifact id: `safetyresidual-sample-tests`
Source label: SAMPTEST
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:863-1029`
Why it matters: Concrete sample-collation and augmentation tests for SafetyResidual and DeltaQ target/mask behavior.

## Artifact 14 — Bootstrap loading of ExIt and DeltaQ sidecars
Artifact id: `bootstrap-sidecar-loading`
Source label: BOOT
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/bootstrap.rs:120-158`
Why it matters: Train bootstrap path that loads replay ExIt and DeltaQ sidecars into the streaming loader configuration.

## Artifact 15 — Borrowed vs owned collation parity for SafetyResidual and DeltaQ
Artifact id: `borrowed-collate-tests`
Source label: BORROW
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:1236-1319`
Why it matters: Additional collation-parity test surface proving SafetyResidual and DeltaQ survive both borrowed and owned batch collation paths consistently.

## Artifact 16 — SafetyResidual target builder and replay-loader context
Artifact id: `safetyresidual-target-builder`
Source label: SRBUILD
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:283-429`
Why it matters: Core builder for signed replay-derived SafetyResidual semantics plus surrounding replay loader context.

## Artifact 17 — Loader tests for SafetyResidual semantics
Artifact id: `safetyresidual-loader-tests`
Source label: SRLOAD
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:838-909`
Why it matters: Concrete tests showing discard-only SafetyResidual population and signed exact_safety - public_score semantics.

## Artifact 18 — Loss tests for SafetyResidual behavior
Artifact id: `safetyresidual-loss-tests`
Source label: SRLOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:1000-1093`
Why it matters: Unit tests proving SafetyResidual loss is nonzero when enabled and present, and zero when mask semantics are broken.

## Artifact 19 — BC test: advanced auxiliary targets change loss
Artifact id: `bc-advanced-loss-test`
Source label: BCTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/bc.rs:857-944`
Why it matters: Behavior-level BC test showing advanced auxiliary targets materially change the loss surface.

## Artifact 20 — Target presence extraction for DeltaQ and SafetyResidual
Artifact id: `head-presence-extraction`
Source label: PRES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:243-359`
Why it matters: Exact presence-counting logic used by the head activation controller for sparse and dense advanced lanes.

## Artifact 21 — Head activation controller state machine
Artifact id: `head-gates-state-machine`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:556-735`
Why it matters: Core gating logic for coverage, sparse density, warmup, and deferred conflict checks across advanced heads.

## Artifact 22 — Coverage/conflict/controller tests for DeltaQ and SafetyResidual
Artifact id: `head-coverage-tests`
Source label: COVTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:886-1099`
Why it matters: Dense test surface for label density, sparse SPP, dense rho, conflict tracking, and activation behavior.

## Artifact 23 — Head-gate tests for SafetyResidual presence and gating
Artifact id: `safetyresidual-presence-gates`
Source label: SRGATE
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1325-1469`
Why it matters: Presence extraction and gating tests specific to SafetyResidual, including nonzero-mask row counting and gated pass-through.

## Artifact 24 — Train entrypoint tests for SafetyResidual-only activation and blocked belief activation
Artifact id: `train-safetyresidual-config-tests`
Source label: TRAINTEST
Type: `file_range`
Source: `crates/hydra-train/src/bin/train.rs:1018-1059`
Why it matters: Train-binary tests showing SafetyResidual is admitted while belief remains blocked.

## Artifact 25 — Advanced loss config surface
Artifact id: `advanced-loss-config-surface`
Source label: ADVCFG
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/config.rs:173-182`
Why it matters: Exact config surface for the advanced loss knobs that runtime/train validation admits or blocks.

## Artifact 26 — Live code: current DeltaQ validation harness
Artifact id: `deltaq-validation-current`
Source label: DQV
Type: `file_range`
Source: `crates/hydra-train/src/training/delta_q_validation.rs:174-396`
Why it matters: Shows that current DeltaQ validation is structural (sample size, emission, coverage, support) rather than direct decision-quality proof.

## Artifact 27 — Live code: ExIt validation contrast
Artifact id: `exit-validation-contrast`
Source label: EXITV
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:218-358`
Why it matters: ExIt already has top-1 agreement and KL-style criteria, which makes it the right comparison point for what DeltaQ promotion still lacks.

## Artifact 28 — Train binary validation runner
Artifact id: `validation-runner-current`
Source label: VALRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/validation.rs:1-142`
Why it matters: Current validation pass and scoring surface in the train binary; useful for seeing what already exists versus what a DeltaQ promotion gate would still need.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_32.md">
  <![CDATA[## Recommendation

Use **DeltaQ’s promotion scaffolding**, **not** DeltaQ’s semantics. SafetyResidual already has real label semantics and real train-time plumbing; what it lacks is a **lane-specific validation/report gate**. The right shape is:

1. make SafetyResidual’s target/mask discipline as strict as DeltaQ’s,
2. add a **DeltaQ-style structural validator** for SafetyResidual,
3. then add one **narrow SafetyResidual-specific behavioral check** based on the **recomposed safety score** `public_score + residual`.

Do **not** turn SafetyResidual into a search-sidecar or live-RL lane. Do **not** reuse ExIt’s KL gate on the raw residual vector.

That recommendation is justified by the semantics of the lane. Expert Iteration is explicitly a tree-search-to-policy imitation setup, so KL and top-1 agreement are natural when the target is an improved policy distribution. AlphaZero-style work also distinguishes policy-improvement targets from value-like targets. Meanwhile, multi-task literature is very clear that mixed regression/classification objectives are sensitive to loss scaling, and negative gradient cosine is a standard signal of interference; that supports keeping Hydra’s existing density/conflict head-gate machinery instead of replacing it with “just tune the weight.” ([arXiv][1])

By analogy, adjacent search-learning work also shows that a strong outer system can hide defects in an internal learned component. That is a good reason not to stop at “labels exist” or “aux loss is nonzero”: once the structural gate exists, SafetyResidual should eventually be checked at the **decision object it induces**. 

## What SafetyResidual already has today

**Direct artifact support, high confidence:**

SafetyResidual is already a real replay-derived lane, not a placeholder. The loader computes a signed target on legal discard actions only:

[
r(a) = \text{exact_safety}(a) - \text{public_score}(a)
]

with `exact_safety = 1 - exact_dealin`, and masks only legal discards. Red fives are normalized before scoring. That is explicit in `build_safety_residual_targets` (Artifact 16, SRBUILD L0283-L0307). Loader tests confirm discard-only masking, finite masked values, bounded masked residuals in `[-1, 1]`, and the existence of positive/nonzero replay residuals (Artifact 17, SRLOAD L0838-L0896).

The carriage path is also real. `MjaiSample`/`MjaiBatch` carry `safety_residual_target` and `safety_residual_mask`; augmentation permutes the action-vector target under suit symmetry; borrowed and owned collation agree; and `HydraTargets` receives the tensors (Artifacts 12, 13, 15).

Loss wiring is closed. `losses.rs` applies `masked_action_mse(outputs.safety_residual, target, mask)` and returns zero when target/mask are absent (Artifacts 08-10, 18). Tests show the loss is nonzero when enabled and present, and zero when the mask is absent or all-zero (Artifacts 10 and 18).

Admission is closed in the train entrypoint. `advanced_loss.safety_residual` is allowed in `build_loss_config()` and `build_rl_loss_config()`, while belief, mixture, and opponent-hand-type remain blocked (Artifact 07). Train-binary tests explicitly allow SafetyResidual-only activation (Artifact 24). The config surface includes `safety_residual` as a first-class knob (Artifact 25).

Routing is also real. The model sends `safety_residual` through a pooled 46-way head, alongside policy and delta_q, not through a spatial tile head (Artifact 11). Generic head-gate extraction counts SafetyResidual presence from nonzero mask rows, and controller tests show full dense-head warmup/active lifecycle for SafetyResidual (Artifacts 20, 22, 23).

One important limit is also explicit: the **shipped closure is the narrow replay/supervised BC lane**, not a live-RL SafetyResidual producer. The doctrine repeatedly calls SafetyResidual “replay-derived and narrow,” while DeltaQ is the lane explicitly described as closed across live RL and replay/offline BC (Artifacts 02-04). `build_rl_loss_config()` admits a weight, but I do **not** see direct artifact evidence of a live RL SafetyResidual label producer analogous to DeltaQ’s root-search path. That gap is real.

## What SafetyResidual still lacks compared with DeltaQ

**Direct artifact support, high confidence:**

DeltaQ already has a dedicated validation module with thresholds, result objects, and a pass/fail evaluator over structural metrics like sample size, emission rate, coverage, and supported actions (Artifact 26). ExIt likewise has a dedicated validator with KL and top-1 agreement criteria (Artifact 27).

There is **no analogous SafetyResidual validator** in the artifact set.

The generic train validation runner only exposes `total_loss`, `policy_loss`, `agreement`, and `samples` in its returned summary (Artifact 28). So even if SafetyResidual is active and training, there is no evidenced lane-specific validation output or promotion report for it.

There is also a subtle plumbing asymmetry: DeltaQ and ExIt collation enforce target/mask pairing with a panic on mismatch, but SafetyResidual collation currently copies target and mask independently and only sets `any_safety_residual = true` if either exists (Artifact 12, SAMPLE L0295-L0305 vs. L0306-L0315). That means a future mismatch can silently degrade into an all-zero mask instead of failing fast. DeltaQ already has stronger pair discipline here.

So the missing pieces are:

1. **Dedicated validator/report/thresholds** for SafetyResidual.
2. **Fail-fast target/mask parity** in sample collation.
3. **Runner-visible output** for the lane.
4. **Semantics-aware promotion criteria** beyond the generic density/conflict gate.

## Why the right next move is not “clone DeltaQ semantics” or “reuse ExIt wholesale”

Clone **DeltaQ’s structure**. Do **not** clone DeltaQ’s search semantics.

RECON is explicit: SafetyResidual should stay replay-derived and narrow; it should not drift into search-derived semantics (Artifact 04, RECON L0442-L0449). So no sidecar, no search-root producer, no live-RL broadening.

Also do **not** reuse ExIt’s behavioral criteria wholesale. ExIt KL/top-1 is meaningful because ExIt’s target is an improved policy. SafetyResidual’s target is a signed correction term, and raw residuals are not a probability distribution.

A tiny worked example shows why KL/top-1 on the residual itself is the wrong object:

[
p = [0.95, 0.50], \quad e = [0.80, 0.75], \quad r = e-p = [-0.15, +0.25]
]

If you look only at residuals, action 2 “wins” because `+0.25 > -0.15`. But the actual safest discard is still action 1 because `0.80 > 0.75`. So the right behavioral object is **not** `r`; it is

[
\hat e(a)=p(a)+\hat r(a)
]

and any ExIt-like agreement criterion has to be applied **after recomposition**.

So the best answer is:

* **DeltaQ-style validator/report/threshold plumbing**
* **SafetyResidual-specific metrics**
* **only one borrowed ExIt idea:** agreement on the induced decision object, not on the raw residual vector

## Exact blueprint

### Phase 1: bring SafetyResidual up to literal DeltaQ-grade discipline

This is the minimal, high-confidence tranche.

### 1) Tighten pair discipline first

Change `crates/hydra-train/src/data/sample.rs` first.

Today, SafetyResidual target and mask are copied independently. Make it match DeltaQ/ExIt:

```rust
match (safety_residual, safety_residual_mask) {
    (Some(target), Some(mask)) => {
        self.safety_residual_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
            .copy_from_slice(&target);
        self.safety_residual_mask_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
            .copy_from_slice(&mask);
        self.any_safety_residual = true;
    }
    (None, None) => {}
    _ => panic!("safety_residual target/mask mismatch for sample collation"),
}
```

Why first: it closes the only obvious plumbing-discipline asymmetry versus DeltaQ, and it prevents future silent regressions.

Tests to add immediately in the same file:

* `batch_to_hydra_targets_rejects_safety_residual_when_pair_is_incomplete`
* keep existing `batch_to_hydra_targets_carries_safety_residual`
* keep existing augmentation/parity tests unchanged

### 2) Add a dedicated structural validator

Create `crates/hydra-train/src/training/safety_residual_validation.rs`.

Use the existing sample/target/mask carriers. Phase 1 does **not** need new model heads, new sample fields, or new loader semantics.

Suggested report:

```rust
pub struct SafetyResidualValidationReport {
    pub compatible_discard_samples: u64,
    pub labels_emitted: u64,
    pub supported_actions_sum: u64,
    pub coverage_sum: f64,

    pub masked_entry_count: u64,
    pub masked_nonzero_count: u64,
    pub masked_positive_count: u64,
    pub masked_negative_count: u64,
    pub masked_abs_sum: f64,
    pub out_of_range_count: u64,
}
```

Derived metrics:

[
\text{emission_rate} = \frac{\text{labels_emitted}}{\text{compatible_discard_samples}}
]

[
\text{coverage}(s)=\frac{\sum_a m_s(a)}{#{\text{legal discard actions in }s}}
]

[
\text{mean_coverage}=\frac{1}{\text{labels_emitted}}\sum_s \text{coverage}(s)
]

[
\text{mean_supported_actions}=\frac{\text{supported_actions_sum}}{\text{labels_emitted}}
]

[
\text{nonzero_rate}=\frac{\text{masked_nonzero_count}}{\text{masked_entry_count}}
]

[
\text{out_of_range_fraction}=\frac{\text{out_of_range_count}}{\text{masked_entry_count}}
]

Collector rule:

* a sample is “compatible discard” if it has at least one legal discard in `legal_mask[..=DISCARD_END]`
* a label is “emitted” if the SafetyResidual mask row has any nonzero entry
* coverage/support are computed on discard actions only

That denominator is important. SafetyResidual should be judged against **compatible discard samples**, not all replay events, because the lane is discard-only by doctrine.

Suggested hard thresholds for v1:

```rust
pub struct SafetyResidualValidationThresholds {
    pub min_sample_size: u64,              // 1000
    pub min_emission_rate: f64,            // 0.99
    pub min_mean_coverage: f64,            // 0.99
    pub min_mean_supported_actions: f64,   // 3.0
    pub max_out_of_range_fraction: f64,    // 0.0
}
```

Why these are justified:

* `1000` matches the order of magnitude already used by DeltaQ/ExIt.
* `0.99` emission/coverage are justified because the label is directly computable from replay for legal discards; this is not a sparse search lane.
* `3.0` keeps parity with the existing DeltaQ/ExIt “supported actions” floor.
* `0.0` out-of-range is an invariant: the label definition itself implies masked targets should stay within `[-1,1]`, and the loader tests already check boundedness on real replay data.

Also log, but do not hard-gate yet:

* `nonzero_rate`
* `positive_rate`
* `negative_rate`
* `mean_abs_residual`

Those are useful for “is there signal here?” but I do not have artifact evidence to set universal hard thresholds for them.

Tests for the new module:

* `evaluate_report_passes_dense_replay_defaults`
* `evaluate_report_fails_low_emission`
* `evaluate_report_fails_low_coverage`
* `evaluate_report_fails_out_of_range`
* `collect_metrics_counts_only_nonzero_mask_rows`
* `labels_emitted_zero_forces_fail_on_coverage_and_support`

### 3) Make the report runner-visible, but do not let it choose checkpoints

The directly evidenced validation entrypoint is `crates/hydra-train/src/bin/train/validation.rs` (Artifact 28). That file already streams validation buffers and has access to `targets` batch-by-batch, so it is the safest first integration point.

Recommended change:

* extend `ValidationSummary` with `safety_residual_validation: Option<SafetyResidualValidationResult>`
* accumulate the report during the existing validation loop when `w_safety_residual > 0.0`
* print/report it
* **do not** change `is_better_validation()` yet

That last part matters. Current best-checkpoint selection is still policy-loss/agreement-driven (Artifact 28). Keep it that way until SafetyResidual proves stable. The lane should become **visible and reviewable** before it becomes **checkpoint-authoritative**.

## Phase 2: the stronger SafetyResidual-specific promotion gate

This is the part I recommend before calling the lane truly “promoted,” even though literal DeltaQ parity stops at Phase 1.

For this phase, refactor `crates/hydra-train/src/data/mjai_loader.rs` so the label builder can expose not just the residual and mask, but also the two quantities it is defined from.

Suggested internal bundle:

```rust
pub struct SafetyResidualLabelBundle {
    pub public_score: [f32; HYDRA_ACTION_SPACE],
    pub exact_safety: [f32; HYDRA_ACTION_SPACE],
    pub residual: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],
}
```

Then keep the training path narrow:

```rust
fn build_safety_residual_targets(...) -> ([f32; 46], [f32; 46]) {
    let bundle = build_safety_residual_label_bundle(...);
    (bundle.residual, bundle.mask)
}
```

No new head. No new broad carrier. Just shared semantics.

Then add a **behavior report** in `training/safety_residual_validation.rs` or a sibling module.

For a model prediction (\hat r(a)), define:

[
\hat e_{\text{raw}}(a)=p(a)+\hat r(a)
]

and optionally

[
\hat e_{\text{clip}}(a)=\text{clamp}(\hat e_{\text{raw}}(a), 0, 1)
]

Recommended metrics:

* **exact MAE** on masked actions: (\text{MAE}(\hat e_{\text{clip}}, e))
* **sign accuracy** on nontrivial residuals: `sign(\hat r) == sign(r)` for `|r| > ε`
* **pairwise rank agreement** across legal discards
* **top1 exact agreement**:
  [
  \arg\max_a \hat e_{\text{raw}}(a) = \arg\max_a e(a)
  ]
* **flip recovery** on the hard subset where public score is wrong:
  [
  \arg\max_a p(a) \neq \arg\max_a e(a)
  ]
  and the corrected prediction fixes it
* **overshoot rate**:
  [
  \hat e_{\text{raw}}(a)\notin[0,1]
  ]

This is the one place where borrowing an ExIt-like “agreement” idea is right: apply it to the **recomposed exact-safety ranking**, not to the raw residual vector.

I would make these **log-only on the first landing**, then freeze thresholds after one or two representative held-out replay manifests. I do not have enough visible evidence to defend universal hard numbers here, so any fixed threshold I gave now would be a proposal, not a fact.

## What should remain deliberately narrower than DeltaQ

**High confidence:**

Keep SafetyResidual narrower than DeltaQ in these ways:

1. **Replay-derived only.** Do not add search-sidecar semantics or a live root-search producer. RECON explicitly says not to drift SafetyResidual into search-derived semantics (Artifact 04).
2. **Discard-only.** Do not widen to call/kan/belief/opponent work.
3. **Dense-head activation logic.** SafetyResidual already fits the dense `rho_h` path in `head_gates.rs`; do not reclassify it as a sparse-search head.
4. **No checkpoint-selection authority yet.** Log/report it; do not let it override policy-loss/agreement model selection until the behavior report is stable.

## Simpler and stronger local alternatives

The **simpler** local path is:

* pair-discipline fix in `sample.rs`
* structural validator only
* runner-visible pass/fail report

That gets you literal DeltaQ-grade discipline quickly.

The **stronger** local path, and the one I recommend, is:

* the same structural validator
* plus the recomposed behavior report

That second piece is what turns “labels exist” into “this correction head is actually aligned with the decision object it claims to improve.”

## Confidence / blocked surfaces

**High confidence**

* SafetyResidual already has real labels, masks, loss wiring, train admission, pooled routing, and dense head-gate support.
* It lacks a dedicated validator.
* It should stay replay-derived and narrow.

**Medium confidence**

* The exact runner integration file after `validation.rs` is not fully proven from the artifacts. `validation.rs` is the safest directly evidenced place to wire first. If there is already a lane-specific validator orchestration path elsewhere, the artifact set does not show it.

**Blocked**

* I cannot honestly claim that DeltaQ validation is already wired into a train/orchestrator log path, because the call site is not in the provided artifacts.
* I cannot defend fixed hard thresholds for the Phase 2 behavior metrics without at least one held-out replay baseline run.

If you want the shortest action order: **`sample.rs` pair fix -> new `safety_residual_validation.rs` -> `bin/train/validation.rs` report wiring -> optional `mjai_loader.rs` bundle refactor for recomposed behavior metrics`.**

[1]: https://arxiv.org/abs/1705.08439?utm_source=chatgpt.com "[1705.08439] Thinking Fast and Slow with Deep Learning and Tree Search"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
