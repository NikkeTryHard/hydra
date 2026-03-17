<combined_run_record run_id="answer_29" variant_id="deltaq-promotion-gate" schema_version="1">
  <metadata>
    <notes>Self-contained combined record for Agent 29. It preserves the compact prompt shell and artifact manifest generated from the authoritative prompt config, plus the preserved answer text.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — DeltaQ decision-quality promotion gate

<role>
Example role placeholder.
Replace this with the role that fits your actual prompt.
Keep it short and task-specific.
You are writing an implementation-ready research blueprint for a narrow Hydra validation/promotion task.
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
Determine the strongest exact blueprint for adding a real decision-quality promotion gate for the already-closed DeltaQ lane.

We want a concrete answer that makes clear:
- what parts of the DeltaQ lane are already closed in live code
- what the current validation harness measures versus what it does not measure
- what exact promotion object should be added next so DeltaQ can be promoted honestly
- which files should change and in what order
- what measurable acceptance criteria should control promotion
- what proof object is still missing if the task cannot yet be called 90%-safe
- why this task should or should not outrank the strongest routing challenger right now

Treat this as a buildable validation-ready blueprint, not a memo. Use the artifacts below to derive your conclusions.
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

## Artifact 03 — Promoted doctrine: ranked next-step recommendations
Artifact id: `recon-ranked-next`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:136-255`
Why it matters: Promoted execution doctrine for what Hydra should build next and what remains reserve or later.

## Artifact 04 — Promoted doctrine: file-by-file first tranche coding spec
Artifact id: `recon-first-tranche-spec`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:420-544`
Why it matters: Concrete promoted coding spec showing the live DeltaQ and SafetyResidual closure context, acceptance checklist, and no-new-heads rule.

## Artifact 05 — Archive roadmap: phase-next survivors
Artifact id: `archive-phase-next`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:108-125`
Why it matters: Derived archive view showing the strongest surviving later lanes, including tile-aware routing correction, Hand-EV repair, and completed DeltaQ closure context. Treat as archive evidence, not promoted doctrine.

## Artifact 06 — Archive raw detail: narrow DeltaQ object and producer envelope
Artifact id: `answer23-deltaq-blueprint`
Source label: ANS23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:503-619`
Why it matters: Historical narrowing of the honest DeltaQ object and its shared root-search producer semantics. Use as archive evidence only.

## Artifact 07 — Live code: replay-sidecar DeltaQ join into samples
Artifact id: `deltaq-join-loader`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:430-471`
Why it matters: Replay/offline DeltaQ closure in the loader with provenance/version-aware sidecar join.

## Artifact 08 — Live code: sample carriers and batch collation for ExIt and DeltaQ
Artifact id: `sample-carriers-deltaq-exit`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:260-419`
Why it matters: Shows action-vector augmentation, target/mask pair safety, and batch collation for ExIt, DeltaQ, and adjacent advanced targets.

## Artifact 09 — Live code: replay DeltaQ sidecar producer and contract validation
Artifact id: `replay-deltaq-producer-full`
Source label: RQ
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_delta_q.rs:1-320`
Why it matters: Full offline DeltaQ producer, sidecar lookup, provenance/version checks, and contract validation logic.

## Artifact 10 — Live code: replay ExIt sidecar producer contrast
Artifact id: `replay-exit-contrast`
Source label: REXIT
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_exit.rs:1-260`
Why it matters: Replay ExIt producer and sidecar join path for contrast with DeltaQ and to show the stronger behavior-facing validation/report surfaces ExIt already has.

## Artifact 11 — Live code: allowed advanced losses in train entrypoint
Artifact id: `loss-policy-allowed-heads`
Source label: LOSSPOL
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs:1-61`
Why it matters: Shows DeltaQ and SafetyResidual are admitted while belief/mixture/opponent-hand-type remain blocked.

## Artifact 12 — Live code: DeltaQ and SafetyResidual loss terms
Artifact id: `losses-deltaq-safety`
Source label: LOSSES
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:620-679`
Why it matters: Current masked regression wiring for DeltaQ and SafetyResidual inside the shared loss breakdown.

## Artifact 13 — Head activation controller state machine
Artifact id: `head-gates-state-machine`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:556-735`
Why it matters: Core gating logic for coverage, sparse density, warmup, and deferred conflict checks across advanced heads.

## Artifact 14 — Live ExIt/DeltaQ shared root-search producer
Artifact id: `live-exit-deltaq-producer`
Source label: LIVEX
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:320-409`
Why it matters: Shared live self-play producer that emits visit-based ExIt and DeltaQ labels from the same AFBS root search.

## Artifact 15 — Integration proof: RL batch carries ExIt and DeltaQ labels
Artifact id: `integration-deltaq-exit`
Source label: INTEG
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:149-250`
Why it matters: End-to-end proof that live RL batch conversion carries both ExIt and DeltaQ labels with the expected target/mask contents.

## Artifact 16 — RL test: DeltaQ activates and participates after gating
Artifact id: `rl-deltaq-activation-test`
Source label: RLTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:726-775`
Why it matters: Behavior-level RL test showing DeltaQ enters warmup and affects the RL path after activation.

## Artifact 17 — BC test: advanced auxiliary targets change loss
Artifact id: `bc-advanced-loss-test`
Source label: BCTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/bc.rs:857-944`
Why it matters: Behavior-level BC test showing advanced auxiliary targets materially change the loss surface.

## Artifact 18 — SaF fast-path consumer for search-derived per-action context
Artifact id: `saf-fastpath-consumer`
Source label: SAF
Type: `file_range`
Source: `crates/hydra-train/src/saf.rs:48-138`
Why it matters: Shows that live inference can already consume search-derived delta_q-like Group C context through the SaF fast path.

## Artifact 19 — Bridge search features and observation encoding with search context
Artifact id: `bridge-search-deltaq`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:341-417`
Why it matters: Shows how DeltaQ-like search features are emitted into the fixed-superset observation and when search context is considered present.

## Artifact 20 — RL runner: DeltaQ activation, logging, and runtime state
Artifact id: `rl-runner-deltaq-state`
Source label: RLRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/rl_runner.rs:55-166`
Why it matters: Operational RL loop integration for DeltaQ activation and logging inside the train binary.

## Artifact 21 — Runtime autotune path also activates DeltaQ
Artifact id: `runtime-autotune-deltaq`
Source label: AUTO
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/runtime_autotune.rs:181-234`
Why it matters: Shows DeltaQ activation is already wired into runtime probing/autotune, not just the main RL runner.

## Artifact 22 — Live code: current DeltaQ validation harness
Artifact id: `deltaq-validation-current`
Source label: DQV
Type: `file_range`
Source: `crates/hydra-train/src/training/delta_q_validation.rs:174-396`
Why it matters: Shows that current DeltaQ validation is structural (sample size, emission, coverage, support) rather than direct decision-quality proof.

## Artifact 23 — Live code: ExIt validation contrast
Artifact id: `exit-validation-contrast`
Source label: EXITV
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:218-358`
Why it matters: ExIt already has top-1 agreement and KL-style criteria, which makes it the right comparison point for what DeltaQ promotion still lacks.

## Artifact 24 — Live code: current evaluation surface
Artifact id: `eval-surface-current`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/eval.rs:1-220`
Why it matters: Existing evaluation helpers and benchmark gates. Useful for seeing what a DeltaQ decision-quality promotion gate can and cannot currently reuse.

## Artifact 25 — Train binary validation runner
Artifact id: `validation-runner-current`
Source label: VALRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/validation.rs:1-142`
Why it matters: Current validation pass and scoring surface in the train binary; useful for seeing what already exists versus what a DeltaQ promotion gate would still need.

## Artifact 26 — Live code: arena summary helpers
Artifact id: `arena-summaries-current`
Source label: ARENA
Type: `file_range`
Source: `crates/hydra-core/src/arena.rs:231-289`
Why it matters: Shows current summary surfaces available for future paired or fixed-compute evaluation logic.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_29.md">
  <![CDATA[**Verdict**

DeltaQ is **code-closed** but **not promotion-closed**. The live repo evidence is strong that the lane exists end to end; the evidence is weak that it improves decisions. The next exact thing to add is **not more DeltaQ plumbing**. It is a **`DeltaQPromotionReport`** that proves three distinct facts: the head fits the search teacher in a decision-relevant way, that fit transfers into better policy decisions, and the trained policy does not regress full-game strength under fixed compute.

## 1) What is already closed in live code

**Direct artifact support**

DeltaQ is already wired through the full narrow lane:

* repo status says DeltaQ is closed across live RL and replay/offline BC, using a shared root-search producer for `Q(child)-Q(root)` and replay sidecars with provenance/version checks (Artifact 02; Artifact 05 L0120-L0124).
* the live self-play producer emits both ExIt and DeltaQ from the same AFBS root search (Artifact 14, LIVEX L0349-L0365).
* replay/offline sidecars exist, carry semantics/provenance/source-net identity, and are contract-validated on lookup (Artifact 09, RQ L0034-L0107, L0171-L0303).
* the loader joins DeltaQ sidecars back into `MjaiSample` with source/version/legal-mask checks (Artifact 07, LOADER L0439-L0471).
* sample collation carries `delta_q_target` and `delta_q_mask`, augments them under suit permutation, and panics on target/mask mismatch instead of zero-filling (Artifact 08, SAMPLE L0269-L0315, L0351-L0400).
* shared losses already consume masked DeltaQ regression; train entrypoints allow DeltaQ while still blocking belief/mixture/opponent-hand-type (Artifact 12, LOSSES L0630-L0661; Artifact 11, LOSSPOL L0024-L0035, L0052-L0060).
* RL integration, RL activation, and BC loss-surface tests all exist (Artifacts 15, 16, 17, 20, 21).

So the correct repo-level statement today is:

> **DeltaQ is implemented as a real, provenance-explicit auxiliary supervision lane.**

That claim is well supported.

**Inference**

What is **not** shown in the artifacts is a live consumer of the **learned `outputs.delta_q` head** in runtime action selection. Artifact 18 shows a consumer for **search-derived DeltaQ planes inside observation context**, not the learned head output itself. So the honest promotion question is:

> **Does training with DeltaQ supervision improve downstream decisions enough to justify promotion/default-on status?**

Not:

> “Is the learned DeltaQ head already a live inference-time decision surface?”

That latter claim is not supported by the artifacts.

## 2) What the current validation harness measures, and what it does not

**Direct artifact support**

Current DeltaQ validation is structural.

`delta_q_validation.rs` only thresholds:

* `sample_size`
* `emission_rate`
* `mean_coverage`
* `mean_supported_actions`

(Artifact 22, DQV L0174-L0287).

Its per-step collector:

* checks compatible discard state,
* checks hard-state gating,
* reads the already-attached `step.delta_q_label`,
* counts support/coverage/sign statistics,
* sums a budget-derived root-visit quantity,

but it **never evaluates `outputs.delta_q`** from the trained head (Artifact 22, DQV L0290-L0357).

That means it measures:

* label emission frequency,
* mask density/support,
* rejection reasons,
* coarse structural viability.

It does **not** measure:

* whether the DeltaQ head learned the teacher,
* whether the DeltaQ head preserves teacher action ordering,
* whether the policy improves because of DeltaQ supervision,
* whether arena strength improves or at least does not regress,
* whether DeltaQ beats a simpler proxy like masked policy logits.

There are two extra weaknesses worth calling out.

First, the default structural thresholds are too weak for promotion. With `min_sample_size = 1000` and `min_emission_rate = 0.01`, a run can pass with only **10 emitted labels**. That is enough for plumbing sanity, not decision-quality promotion. (Artifact 22, DQV L0183-L0190, L0248-L0287.)

Second, DeltaQ’s diagnostic surface is thinner than ExIt’s. `ReplayExitRecordV1` stores `root_visit_count`, `supported_actions`, `coverage`, and `kl_to_base`; `ReplayDeltaQRecordV1` stores none of those, only target/mask plus identity fields (Artifact 10, REXIT L0044-L0061 vs. Artifact 09, RQ L0043-L0055). ExIt validation also already has behavior-facing criteria (`mean_kl`, `top1_agreement`) while DeltaQ validation does not (Artifact 23, EXITV L0218-L0358 vs. Artifact 22).

Also, the train binary’s ordinary validation surface still selects “best” by policy loss and policy agreement only; no DeltaQ decision metric participates (Artifact 25, VALRUN L0050-L0061, L0126-L0141).

So the honest summary is:

> **Current DeltaQ validation proves structural truth and contract safety, not decision quality.**

## 3) The exact promotion object to add next

Add a new object:

```rust
pub struct DeltaQPromotionReport {
    pub structural: DeltaQValidationResult,      // existing structural gate
    pub head_fit: DeltaQHeadFitReport,           // single-model: delta_q head vs teacher
    pub policy_transfer: DeltaQPolicyTransferReport, // candidate policy vs baseline policy on same teacher-labeled holdout
    pub arena: DeltaQArenaReport,                // fixed-compute paired no-regression / lift
}
```

That is the right object because the search teacher is the right reference object for this lane, but static supervised fit alone is not enough, and advantage-like targets should be judged by ordering/regret, not only raw MSE. ExIt and AlphaGo Zero both use search as a stronger policy-improvement object projected back into the network; DAgger is the warning that sequential-control systems must be judged under the learner’s induced distribution, not only on i.i.d.-style label fit; and action-gap theory plus dueling networks both point toward ordering/gap-sensitive evaluation for relative action values. ([NeurIPS Papers][1])

### The core metric: teacher decision regret

For each validation state (s) with DeltaQ target (t_s[a]) and mask (m_s[a] \in {0,1}), define the supported action set

[
A_s = { a ;|; m_s[a] = 1 }.
]

Ignore states with (|A_s| < 2).

Let

[
a^**s = \arg\max*{a \in A_s} t_s[a]
]

be the teacher-best discard under search.

For any chooser (c_s[a]) (the DeltaQ head, or masked policy logits), define

[
\hat a_s(c) = \arg\max_{a \in A_s} c_s[a]
]

and the **teacher decision regret**

[
R_s(c) = t_s[a^*_s] - t_s[\hat a_s(c)].
]

This is the metric the current harness is missing.

Why it is the right one:

* it is on the teacher’s own value scale,
* it directly measures loss from the chosen action,
* it does not require absolute calibration of the student head,
* it matches the lane semantics (`Q(child)-Q(root)` over supported discards).

Worked example:

* teacher target on three masked discards: `[0.40, 0.10, -0.30]`
* head picks action 2 → regret = `0.40 - 0.10 = 0.30`
* head picks action 1 → regret = `0.00`

That is decision-quality, not just regression fit.

### What `head_fit` should contain

```rust
pub struct DeltaQHeadFitReport {
    pub emitted_labels: u64,
    pub compared_states: u64,        // emitted states with >=2 masked actions
    pub masked_entries: u64,
    pub supported_pairs: u64,

    pub head_top1_agreement: f64,
    pub proxy_top1_agreement: f64,   // masked policy-logit proxy

    pub head_mean_teacher_regret: f64,
    pub proxy_mean_teacher_regret: f64,

    pub head_high_gap_top1: f64,     // top quartile by teacher action-gap
    pub proxy_high_gap_top1: f64,

    pub head_weighted_pair_acc: f64, // optional diagnostic
    pub proxy_weighted_pair_acc: f64,
}
```

The **policy-logit proxy** is the strongest simple local challenger inside the same lane. If the dedicated DeltaQ head cannot beat masked policy logits on teacher regret/order, promotion should stop.

### What `policy_transfer` should contain

This is the missing proof that matters most.

Run two matched training jobs from the same init/data/order:

* **baseline**: current training without DeltaQ loss
* **candidate**: same training, only DeltaQ enabled with the intended weight/controller

Then evaluate **policy logits**, not the DeltaQ head, on the same fixed teacher-labeled holdout.

```rust
pub struct DeltaQPolicyTransferReport {
    pub compared_states: u64,

    pub baseline_policy_top1_to_teacher: f64,
    pub candidate_policy_top1_to_teacher: f64,

    pub baseline_policy_mean_teacher_regret: f64,
    pub candidate_policy_mean_teacher_regret: f64,

    pub delta_policy_agreement_pp: f64,   // ordinary validation agreement delta
    pub delta_policy_loss: f64,
}
```

If DeltaQ only teaches the auxiliary head and does not improve policy decisions, it is not honestly promotable as a training-strength lane.

### What `arena` should contain

Use a fixed-opponent, fixed-compute paired A/B setup:

* same seeds,
* same seat-rotation schedule,
* same search/ponder budget,
* same temperature,
* same frozen opponent pool.

That is the Mahjong analogue of controlled engine testing. Lc0’s public testing guidance recommends controlled match setups with opening control, and the LCZero org exposes OpenBench specifically as an SPRT-based testing framework; the operational lesson transfers even though the game differs. ([Leela Chess Zero][2])

```rust
pub struct DeltaQArenaReport {
    pub games: u64,

    pub baseline_mean_placement: f64,
    pub candidate_mean_placement: f64,
    pub delta_mean_placement: f64,   // candidate - baseline; lower is better

    pub baseline_stable_dan: f64,
    pub candidate_stable_dan: f64,
    pub delta_stable_dan: f64,

    pub baseline_fourth_rate: f64,
    pub candidate_fourth_rate: f64,
    pub delta_fourth_rate: f64,

    pub ci95_low_mean_placement: f64,
    pub ci95_high_mean_placement: f64,
}
```

## 4) Which files should change, and in what order

### Step 0: freeze semantics

Do **not** touch the lane plumbing first. The lane is already closed.

Do not change, in the first pass:

* `hydra-train/src/model.rs`
* `hydra-train/src/data/sample.rs`
* `hydra-train/src/data/mjai_loader.rs`
* `hydra-core/src/afbs.rs`
* `hydra-core/src/bridge.rs`

That matches the repo doctrine: no new heads, no AFBS broadening, no semantic drift (Artifact 04, RECON L0491-L0500, L0511-L0518, L0527-L0540).

### Step 1: add the new promotion module

**New file**

* `crates/hydra-train/src/training/delta_q_promotion.rs`

Put here:

* `DeltaQHeadFitReport`
* `DeltaQPolicyTransferReport`
* `DeltaQArenaReport`
* `DeltaQPromotionReport`
* threshold structs
* pure metric collectors
* paired bootstrap / summary helpers
* pass/fail evaluator

### Step 2: add a single-model holdout runner

**Modify minimally or add sibling**

* preferably a new promotion runner under `crates/hydra-train/src/bin/train/`
* reuse the streaming loader path used by ordinary validation

Goal:

* load a fixed replay holdout manifest with DeltaQ sidecars,
* forward the model,
* accumulate `head_fit` metrics from `output.delta_q`,
* accumulate policy-proxy metrics from `output.policy_logits`.

You may touch `src/bin/train/validation.rs` only to extract shared streaming helpers if reuse is easier; do not overload ordinary best-checkpoint validation logic with promotion semantics yet.

### Step 3: add paired baseline-vs-candidate comparison

**New CLI / subcommand**

* something like `delta_q_promotion`

Inputs:

* `baseline_checkpoint`
* `candidate_checkpoint`
* fixed holdout manifest
* paired arena config
* thresholds

Output:

* one persisted JSON report
* one clear PASS / FAIL verdict

### Step 4: add paired arena helper

**Modify**

* `crates/hydra-train/src/eval.rs`

Add:

* fixed-opponent paired evaluation config/result
* bootstrap CI or SPRT wrapper for mean-placement delta

Only touch `hydra-core/src/arena.rs` if you need a small accessor for per-game rows; otherwise leave it alone.

### Step 5: tests

Add:

* unit tests in `delta_q_promotion.rs` for teacher regret / high-gap slicing / pair metrics
* integration test for sidecar-backed holdout evaluation
* paired arena smoke test with deterministic seeds
* CLI pass/fail fixture tests

### Step 6: only after passing the gate, change defaults

**Then modify**

* `crates/hydra-train/src/bin/train/loss_policy.rs`
* promotion/default config files

Until then, DeltaQ stays **implemented but unpromoted**.

## 5) Measurable acceptance criteria

Only the structural thresholds below are artifact-backed today. The rest are proposed thresholds for the new promotion gate.

### G0 — structural truth (must pass)

Reuse current structural pass, but add real support floors:

* current `DeltaQValidationThresholds` pass
* `emitted_labels >= 5_000`
* `compared_states >= 3_000`
* `masked_entries >= 20_000`

Why this extra floor: the current repo threshold floor can pass with 10 emitted labels, which is nowhere near enough for promotion.

### G1 — head decision fit (must pass)

On the fixed holdout set, compare:

* chooser 1: `output.delta_q`
* chooser 2: masked `output.policy_logits`

Pass if all are true:

* `head_mean_teacher_regret <= 0.90 * proxy_mean_teacher_regret`
* paired 95% bootstrap CI for `(head_regret - proxy_regret)` is entirely `< 0`
* `head_high_gap_top1 >= 0.85` on the top quartile of teacher-gap states
* `head_top1_agreement > proxy_top1_agreement`

This is the “separate head beats the simpler local alternative” gate.

### G2 — policy transfer (must pass)

Across matched baseline-vs-candidate training runs:

* `candidate_policy_mean_teacher_regret <= 0.95 * baseline_policy_mean_teacher_regret`
* paired 95% bootstrap CI for `(candidate_policy_regret - baseline_policy_regret)` is entirely `< 0`
* ordinary validation policy agreement does not drop by more than `0.25` percentage points

This is the real proof that the auxiliary helps the policy rather than only teaching the auxiliary head.

### G3 — arena non-regression (must pass)

Using fixed opponents, fixed compute, same seeds, same seat rotations:

* pass if upper 95% CI for `delta_mean_placement` is `<= +0.025`

Because current eval defines

[
\text{stable_dan} = 10 - 4(\text{mean_placement}-1),
]

that non-inferiority band is equivalent to

[
\Delta \text{stable_dan} \ge -0.10.
]

For **strong promotion**, require either:

* upper 95% CI for `delta_mean_placement <= 0`, or
* your existing match infra accepts a positive SPRT-style improvement verdict.

### 90%-safe bar

To call the lane **90%-safe to promote**, do not rely on one lucky run.

Require:

* **3 matched seed pairs** of baseline vs candidate training runs,
* at least **2 of 3** satisfy G2,
* **0 of 3** violate G3.

If you only do one matched pair, you can decide whether to kill or continue the lane, but you cannot honestly call it 90%-safe.

## 6) The missing proof object right now

Right now the missing proof object is:

> **a persisted matched A/B DeltaQ promotion report showing policy-transfer improvement on a fixed search-labeled holdout and no paired arena regression at fixed compute.**

Until that exists, the honest statement is:

> **DeltaQ is implemented and structurally validated, but not yet decision-quality promoted.**

My confidence levels:

* **High**: the lane is closed in code.
* **High**: the current validation harness is structural, not decision-quality.
* **Medium-high**: the right next object is `DeltaQPromotionReport` built around teacher regret + policy transfer + paired arena.
* **Low**: any claim that DeltaQ is already safe to promote/default-on.

## 7) Should this outrank the strongest routing challenger right now?

Yes, **for one bounded tranche**, it should outrank the routing challenger.

**Direct artifact support**

The strongest routing challenger in the provided archive is the tile-aware spatial/global routing correction, but the archive explicitly says it is a strong survivor **not current doctrine** (Artifact 05, ROADMAP L0115-L0118). The promoted doctrine says Hydra’s active mainline is still supervision-first, no giant architecture expansion, and truthful advanced-target discipline before broader architectural/search projects (Artifact 03, RECON L0217-L0235).

**Why DeltaQ gate should go first**

* it resolves truth on an already-shipped lane,
* it requires no new heads and almost no semantic risk,
* it produces an immediate keep / narrow / kill decision,
* it prevents the team from claiming more than the repo has earned.

So the priority order I would use is:

1. build the DeltaQ promotion gate,
2. run one matched baseline-vs-candidate smoke pair,
3. if DeltaQ fails G1 or G2, stop promotion work immediately and move to routing correction,
4. if DeltaQ passes the smoke pair, finish the 3-seed promotion suite.

That makes this a **high-decision-value, short, falsifiable tranche**. It outranks the routing challenger **because it closes an honesty gap on live code**. It does **not** outrank it indefinitely.

## 8) The practical end state

If the gate passes, flip DeltaQ from “implemented but selective” to “promoted auxiliary lane,” and only then consider default-on configs.

If the gate fails:

* keep the lane closed but unpromoted,
* leave DeltaQ weight at `0` in defaults,
* preserve the code and sidecar machinery,
* move the team to the routing challenger.

That is the cleanest, most buildable, least hand-wavy next move.

[1]: https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf "https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf"
[2]: https://lczero.org/dev/wiki/testing-guide/ "https://lczero.org/dev/wiki/testing-guide/"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
