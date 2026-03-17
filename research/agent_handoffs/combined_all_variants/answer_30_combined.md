<combined_run_record run_id="answer_30" variant_id="routing-vs-deltaq-headtohead" schema_version="1">
  <metadata>
    <notes>Self-contained combined record for Agent 30. It preserves the compact prompt shell and artifact manifest generated from the authoritative prompt config, plus the preserved answer text.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — tile-aware routing correction vs DeltaQ promotion gate

<role>
Example role placeholder.
Replace this with the role that fits your actual prompt.
Keep it short and task-specific.
You are doing a strict Hydra head-to-head between two narrow next-build candidates.
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
Compare exactly two candidate next tasks and pick one:
(A) add a real DeltaQ decision-quality promotion gate for the already-closed DeltaQ lane
(B) apply the narrow tile-aware spatial/global routing correction for tile-indexed outputs

We want a hard winner with explicit reasoning on:
- doctrine alignment
- live code readiness
- blueprint precision
- measurable acceptance criteria
- likely strength leverage
- why the loser loses right now

Do not widen back into broad architecture or broad supervision work. Use the artifacts below to derive the strongest exact next-build recommendation.
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

## Artifact 04 — Archive roadmap: phase-next survivors
Artifact id: `archive-phase-next`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:108-125`
Why it matters: Derived archive view showing the strongest surviving later lanes, including tile-aware routing correction, Hand-EV repair, and completed DeltaQ closure context. Treat as archive evidence, not promoted doctrine.

## Artifact 05 — Archive canonical rendered row: strongest surviving architecture claim
Artifact id: `archive-routing-claim`
Source label: ARCHIVE
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md:68-68`
Why it matters: Canonical archive row for the routing/history architecture survivor. Use as archive evidence only.

## Artifact 06 — Archive raw detail: tile-indexed heads should read spatial features
Artifact id: `answer21-routing-detail`
Source label: ANS21
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_21_combined.md:647-689`
Why it matters: High-signal raw archive detail behind the routing correction claim. Preserve as evidence, not doctrine.

## Artifact 07 — Archive raw detail: narrow DeltaQ object and producer envelope
Artifact id: `answer23-deltaq-blueprint`
Source label: ANS23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:503-619`
Why it matters: Historical narrowing of the honest DeltaQ object and its shared root-search producer semantics. Use as archive evidence only.

## Artifact 08 — Promoted architecture doctrine: SaF, ExIt, and validation gates
Artifact id: `hydra-final-validation-gates`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:249-365`
Why it matters: North-star validation language for search-as-feature, ExIt, and measurable improvement gates.

## Artifact 09 — Live code: pooled-vs-spatial head routing
Artifact id: `model-routing-live`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:419-438`
Why it matters: Current forward path showing pooled routing for policy, DeltaQ, and SafetyResidual versus spatial routing for tile-structured heads.

## Artifact 10 — Live code: head definitions for pooled and spatial branches
Artifact id: `heads-routing-live`
Source label: HEADS
Type: `file_range`
Source: `crates/hydra-train/src/heads.rs:1-160`
Why it matters: Defines which heads are linear-on-pooled versus conv-on-spatial today.

## Artifact 11 — Backbone split into spatial and pooled outputs
Artifact id: `backbone-spatial-pooled`
Source label: BACKBONE
Type: `file_range`
Source: `crates/hydra-train/src/backbone.rs:137-145`
Why it matters: Minimal live seam showing the backbone already exposes both spatial and pooled representations.

## Artifact 12 — Head init and shape tests for DeltaQ and SafetyResidual
Artifact id: `heads-shape-tests`
Source label: HEADTEST
Type: `file_range`
Source: `crates/hydra-train/src/heads.rs:281-425`
Why it matters: Current test surface for pooled head construction and output shape expectations.

## Artifact 13 — Model output tests and finiteness checks
Artifact id: `model-output-tests`
Source label: MODELTEST
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:548-677`
Why it matters: Current model-level test surface showing output shapes and finiteness but not a routing-behavior oracle.

## Artifact 14 — Implementation roadmap: current head/output surface reference
Artifact id: `impl-roadmap-heads-context`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:172-329`
Why it matters: Reference-only live-snapshot context for current head surfaces, routing, and model output shapes. Subordinate to reconciliation and code.

## Artifact 15 — Live code: sample carriers and batch collation for ExIt and DeltaQ
Artifact id: `sample-carriers-deltaq-exit`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:260-419`
Why it matters: Shows action-vector augmentation, target/mask pair safety, and batch collation for ExIt, DeltaQ, and adjacent advanced targets.

## Artifact 16 — Live code: replay DeltaQ sidecar producer and contract validation
Artifact id: `replay-deltaq-producer-full`
Source label: RQ
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_delta_q.rs:1-320`
Why it matters: Full offline DeltaQ producer, sidecar lookup, provenance/version checks, and contract validation logic.

## Artifact 17 — Head activation controller state machine
Artifact id: `head-gates-state-machine`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:556-735`
Why it matters: Core gating logic for coverage, sparse density, warmup, and deferred conflict checks across advanced heads.

## Artifact 18 — Integration proof: RL batch carries ExIt and DeltaQ labels
Artifact id: `integration-deltaq-exit`
Source label: INTEG
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:149-250`
Why it matters: End-to-end proof that live RL batch conversion carries both ExIt and DeltaQ labels with the expected target/mask contents.

## Artifact 19 — RL test: DeltaQ activates and participates after gating
Artifact id: `rl-deltaq-activation-test`
Source label: RLTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:726-775`
Why it matters: Behavior-level RL test showing DeltaQ enters warmup and affects the RL path after activation.

## Artifact 20 — BC test: advanced auxiliary targets change loss
Artifact id: `bc-advanced-loss-test`
Source label: BCTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/bc.rs:857-944`
Why it matters: Behavior-level BC test showing advanced auxiliary targets materially change the loss surface.

## Artifact 21 — Bridge search features and observation encoding with search context
Artifact id: `bridge-search-deltaq`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:341-417`
Why it matters: Shows how DeltaQ-like search features are emitted into the fixed-superset observation and when search context is considered present.

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

## Artifact 24 — Live code: DeltaQ and SafetyResidual loss terms
Artifact id: `losses-deltaq-safety`
Source label: LOSSES
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:620-679`
Why it matters: Current masked regression wiring for DeltaQ and SafetyResidual inside the shared loss breakdown.

## Artifact 25 — Live code: current evaluation surface
Artifact id: `eval-surface-current`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/eval.rs:1-220`
Why it matters: Existing evaluation helpers and benchmark gates. Useful for seeing what a DeltaQ decision-quality promotion gate can and cannot currently reuse.

## Artifact 26 — Train binary validation runner
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
  <answer_text status="preserved" source_path="agent_30.md">
  <![CDATA[**Hard winner: A — add a real DeltaQ decision-quality promotion gate.**

My confidence is **high** on the *next-build choice*, and **medium** on whether DeltaQ will ultimately deserve permanent promotion. The reason those differ is simple: the repo already has a closed DeltaQ lane, but it still lacks the proof object that would justify trusting that lane. Building that proof object is the sharper next move than rewriting head routing right now.

**Why A wins on doctrine alignment**

Direct artifact support is unusually strong here. The repo status says DeltaQ is already “closed across both live RL and replay/offline BC paths” (README L72). The promoted reconciliation doctrine then says the mainline priority is **truthful advanced-target / activation discipline after the shipped supervision tranche** (RECON L0138-L0152, L0217-L0220), not a fresh architecture turn. DeltaQ fits that exactly: it is already in code, already wired into BC/RL/losses, and still missing truthful promotion criteria. By contrast, the routing correction is explicitly labeled a **strong archive architecture survivor; not current doctrine** (ROADMAP L0116).

That doctrine reading is also the one most consistent with primary-source precedent. AlphaZero-style training improves policies by matching search-improved targets rather than raw network priors, ExIt is explicitly about improving the expert and then distilling that stronger expert into the apprentice, and safe-policy-improvement work defines improvement relative to a baseline policy rather than relative to structural target availability. ([ar5iv][1])

**Why A wins on live code readiness**

Direct artifact support again favors A. DeltaQ already has:

* a precise target object: masked `Q(child)-Q(root)` over discard-compatible actions (ANS23 L0511-L0517, L0551-L0603),
* live RL carriage (INTEG L0149-L0250),
* replay/offline sidecar generation with provenance/version checks (RQ L0034-L0108, L0171-L0302),
* BC/RL loss wiring (LOSSES L0630-L0661),
* activation/warmup logic (GATES L0559-L0565, L0691-L0729),
* and behavior tests showing it enters training once activated (RLTEST L0727-L0775, BCTEST L0857-L0944).

What is missing is not more plumbing. What is missing is a gate that answers: **does this lane improve decisions enough to deserve promotion?**

The current validator does not answer that. `delta_q_validation.rs` only checks sample size, emission rate, coverage, and supported-action count (DQV L0174-L0287). ExIt already has a more decision-facing validator with KL and top-1 agreement (EXITV L0218-L0358). That asymmetry is the gap.

**Why A wins on blueprint precision**

A has a cleaner measurable object than B because DeltaQ’s semantics are already exact. For a fixed root state,

[
\Delta Q^*_s(a) = Q_s(\text{child}_a) - Q_s(\text{root})
]

So for any two compared actions (a,b),

[
\Delta Q^*_s(a)-\Delta Q^*_s(b)=Q_s(\text{child}_a)-Q_s(\text{child}_b)
]

The root term cancels. That means DeltaQ already gives you an honest **per-root decision-quality comparator** on supported actions. You do not need to invent a vague surrogate.

That lets you build a promotion gate directly from Hydra’s own validation doctrine. `HYDRA_FINAL` G0 says a decision-improvement gate should require **mean (\Delta > 0)** and **<40% negative** (FINAL L0350-L0358). For DeltaQ, the exact local version is:

1. Build a held-out bank (S) of **hard, compatible discard states** with nonzero `delta_q_mask`.
2. For each state, restrict comparison to the teacher-supported mask (M_s={a:m_s(a)=1}).
3. Let `a_base` be the control policy’s argmax on (M_s), and `a_cand` the candidate’s argmax on (M_s).
4. Define per-state lift:

[
\delta_s = \Delta Q^**s(a*{\text{cand}})-\Delta Q^**s(a*{\text{base}})
]

5. Promote only if:

   * the existing structural validator passes first,
   * (\frac{1}{|S|}\sum_s \delta_s > 0),
   * (\Pr(\delta_s<0) < 0.40),
   * and the candidate is not worse in downstream arena/self-play metrics.

That is much sharper than B, because B still needs unresolved semantic choices before you can even write the acceptance test.

**Why B loses on blueprint precision right now**

The routing correction is real, but it is not as implementation-ready as it looks.

Direct artifact support shows the seam: the backbone already exposes both `spatial` and `pooled` (BACKBONE L0138-L0144), yet `policy`, `delta_q`, and `safety_residual` all currently read from `pooled` (MODEL L0424-L0437; HEADS L0010-L0018, L0141-L0160). So the defect is visible.

But the local patch is not semantically closed:

* Hydra’s external action space is **46**, not a clean 34-tile discard head (Artifact 6; MODELTEST L0548-L0563).
* The spatial surface is width **34** (BACKBONE L0142-L0144).
* The archive’s proposed internal split is **37 discard + 9 global** (ANS21 L0675-L0680), which immediately raises an explicit unresolved mapping problem for the 3 aka actions.
* DeltaQ’s current contract **forbids aka support** and only validates masked entries on non-aka discard indices (RQ L0153-L0163).
* Bridge-side search DeltaQ features are populated only over `NUM_TILE_TYPES` / 34 tile types (BRIDGE L0341-L0353).

So B is not “swap one layer and ship.” It needs an explicit red-five policy, explicit treatment of tile-indexed auxiliaries whose support semantics differ, and new routing-behavior tests. Today’s tests are shape/finiteness tests, not routing or semantic tests (HEADTEST L0305-L0425; MODELTEST L0548-L0667).

**Why A wins on measurable acceptance criteria**

A can be promoted with exact, reviewable gates.

A minimal implementation would add this to `crates/hydra-train/src/training/delta_q_validation.rs`:

```rust
pub struct DeltaQPromotionThresholds {
    pub structural: DeltaQValidationThresholds,
    pub min_mean_decision_lift: f64,   // 0.0
    pub max_negative_lift_frac: f64,   // 0.40
    pub require_top1_non_regression: bool,
}

pub struct DeltaQDecisionQualityReport {
    pub eligible_states: u64,
    pub mean_decision_lift: f64,
    pub negative_lift_frac: f64,
    pub base_top1: f64,
    pub cand_top1: f64,
}
```

and then compute:

```rust
delta_s = target[a_cand] - target[a_base];
best_s  = argmax_a target[a] over supported mask;
```

with `a_base` and `a_cand` chosen by policy logits restricted to `legal && delta_q_mask`.

The right file-level build is narrow:

1. Keep the existing structural report as a prerequisite.
2. Add a held-out RL-first state bank builder from existing search-derived DeltaQ labels.
3. Add candidate-vs-control comparison over that bank.
4. Wire a promotion summary into `src/bin/train/validation.rs`.
5. Treat self-play/arena non-regression as the second-stage gate.

The only materially missing proof object for A is the visible arena comparator. `eval.rs` exposes metrics and thresholds, but the artifact set does not show a concrete head-to-head runner (EVAL L0001-L0220). So the honest statement is: **offline decision-quality gating is ready to specify now; full promotion-to-default-on still needs a visible comparison runner if one does not already exist elsewhere.**

**Why A wins on likely leverage**

A has better *expected value per engineering hour* because it is useful no matter what the answer turns out to be.

If the gate passes, you have a justified reason to keep or strengthen an already-shipped supervision lane. If it fails, you stop pretending a structurally healthy lane deserves promotion. That is exactly the “treat artifacts as evidence, not truth” posture from Artifact 1.

B has real upside, and the outside evidence says so. Suphx is especially important here: it uses tile-column semantics, keeps a separate 34-way discard model, and explicitly avoids pooling in that model because pooling would lose tile-column meaning. But that same paper also attributes its gains to **global reward prediction, oracle guiding, and runtime policy adaptation**; it is not evidence that a mixed single-head 46-action routing correction should outrank target-truthfulness work in a repo where DeltaQ is already closed and routing semantics are still partially unresolved. AlphaZero also weakens the urgency case for B a bit: it uses geometry-matched action representations, but reports that a flat action output for chess/shogi reached almost the same final result, only training somewhat slower. ([ar5iv][2])

**Why the loser loses right now**

B loses **now**, not forever.

Direct support says it is a real defect: the current policy path is pooled-only, and the archive correctly preserved “stop routing tile-indexed policy through pooled-only features” as the strongest architecture survivor (ARCHIVE L0068; ANS21 L0647-L0680). Mahjong-specific primary evidence from Suphx makes that survivor stronger, not weaker. ([ar5iv][2])

But the loser loses on three concrete points:

1. **Not mainline doctrine now.** The promoted reconciliation doc puts truthful target promotion on the critical path and leaves routing correction as archive-side later work.
2. **Not semantically closed.** The 34-column spatial surface does not directly resolve Hydra’s 46-action mixed interface, especially aka handling and DeltaQ’s non-aka contract.
3. **No crisp strength gate yet.** There is no visible routing-behavior oracle or no-regression harness tailored to this patch.

**The better narrowed version of B, later**

If you ignore my recommendation and insist on the routing lane, do **not** take the full “tile-indexed outputs” bundle first.

The better local version is:

* policy discard split first,
* keep the fixed 46 external interface,
* leave history-sidecar work out,
* leave DeltaQ and SafetyResidual alone until their action-index semantics are made explicit.

That is narrower and stronger than touching all tile-indexed outputs at once.

**Bottom line**

Keep **A**. Narrow it to an **RL-first DeltaQ decision-quality promotion gate** built on the existing search-derived masked target, with structural thresholds as prerequisites and a G0-style decision-lift test as the core acceptance criterion.

Do **not** promote B next. Keep it as the strongest architecture-side reserve patch, and when you come back to it, narrow it to **policy discard routing only** before touching the rest of the 46-action sur

face.

[1]: https://ar5iv.org/pdf/1712.01815 "https://ar5iv.org/pdf/1712.01815"
[2]: https://ar5iv.org/abs/2003.13590 "[2003.13590] Suphx: Mastering Mahjong with Deep Reinforcement Learning1footnote 11footnote 1This work was conducted at Microsoft Research Asia. The 2nd, 4th, 5th, and 6th authors were interns at Microsoft Research Asia then."
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
