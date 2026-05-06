<combined_run_record run_id="answer_30" variant_id="routing-vs-deltaq-headtohead" schema_version="1">
<metadata>
<notes>Self-contained combined record for Agent 30. Keeps compact prompt shell, artifact manifest from authoritative prompt config, plus preserved answer text.</notes>
<layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
<![CDATA[# Hydra prompt — tile-aware routing correction vs DeltaQ promotion gate

<role>
Example role placeholder.
Replace with role fitting actual prompt.
Keep short, task-specific.
Strict Hydra head-to-head between two narrow next-build candidates.
</role>

<task>
Example task placeholder.
Replace with actual job agent should do.

Example task block may ask for:
- what artifacts directly support
- what is inference only
- what confidence each major conclusion deserves
- what simpler or stronger local alternatives exist inside same lane
- what to keep, narrow, or remove
- why confident answer parts are justified
- how to implement or validate surviving path with minimal guesswork

Use artifacts below to derive conclusions.
Compare exactly two candidate next tasks and pick one:
add real DeltaQ decision-quality promotion gate for already-closed DeltaQ lane
(B) apply narrow tile-aware spatial/global routing correction for tile-indexed outputs

Need hard winner with explicit reasoning on:
- doctrine alignment
- live code readiness
- blueprint precision
- measurable acceptance criteria
- likely strength leverage
- why loser loses now

Do not widen into broad architecture or broad supervision work. Use artifacts below to derive strongest exact next-build rec.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections; customize when prompt needs
- distinguish direct artifact support from own inference
- use search/browse aggressively when it strengthens answer: find original paper, adjacent papers, official docs, repos, other primary sources; use abstracts or summaries mostly for discovery, not final evidence
- use bash tool to run Python for light research support when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, validation
- do not dump logic; every important mechanism, threshold, or rec should be inferable from evidence or explicit in blueprint so it can be validated and reproduced
- if claiming path works, survives, or is impl-ready, show why confidence justified and how claim can be validated or falsified later
- inspect own draft before finishing: if confident claim lacks objective visible evidence, downgrade to inference, proposal, or blocked
- do not finish early; keep looping through discovery, thinking, testing, validation until information saturated, falsified, or truly blocked, and do not stop because first pass produced plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when sounding confident, show justification for confidence level
- for every important claim, make validation path visible enough for later reviewer testing
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail for us to validate, reproduce, or falsify ourselves (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now. Not guaranteed fully correct. Treat as evidence to inspect and critique, not truth to inherit. High chance some are incomplete, misleading, stale, or semantically wrong, so validate all.
</artifact_note>

<artifacts_manifest>

## Artifact 01 — Prompt-packing reminder
Artifact id: `prompt-packing-reminder`
Type: `literal`
Why it matters: Task reminder matching prompt style guide.

## Artifact 02 — Repo status snapshot
Artifact id: `repo-status`
Source label: README
Type: `file_range`
Source: `README.md:70-72`
Why it matters: Current repo truth on shipped lanes vs still selective or staged lanes.

## Artifact 03 — Promoted doctrine: ranked next-step recommendations
Artifact id: `recon-ranked-next`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:136-255`
Why it matters: Promoted execution doctrine for Hydra next build and reserve/later work.

## Artifact 04 — Archive roadmap: phase-next survivors
Artifact id: `archive-phase-next`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:108-125`
Why it matters: Derived archive view of strongest surviving later lanes, including tile-aware routing correction, Hand-EV repair, completed DeltaQ closure context. Archive evidence only, not promoted doctrine.

## Artifact 05 — Archive canonical rendered row: strongest surviving architecture claim
Artifact id: `archive-routing-claim`
Source label: ARCHIVE
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md:68-68`
Why it matters: Canonical archive row for routing/history architecture survivor. Archive evidence only.

## Artifact 06 — Archive raw detail: tile-indexed heads should read spatial features
Artifact id: `answer21-routing-detail`
Source label: ANS21
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_21_combined.md:647-689`
Why it matters: High-signal raw archive detail behind routing correction claim. Preserve as evidence, not doctrine.

## Artifact 07 — Archive raw detail: narrow DeltaQ object and producer envelope
Artifact id: `answer23-deltaq-blueprint`
Source label: ANS23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:503-619`
Why it matters: Historical narrowing of honest DeltaQ object and shared root-search producer semantics. Archive evidence only.

## Artifact 08 — Promoted architecture doctrine: SaF, ExIt, and validation gates
Artifact id: `hydra-final-validation-gates`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:249-365`
Why it matters: North-star validation language for search-as-feature, ExIt, measurable improvement gates.

## Artifact 09 — Live code: pooled-vs-spatial head routing
Artifact id: `model-routing-live`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:419-438`
Why it matters: Current forward path showing pooled routing for policy, DeltaQ, SafetyResidual, spatial routing for tile-structured heads.

## Artifact 10 — Live code: head definitions for pooled and spatial branches
Artifact id: `heads-routing-live`
Source label: HEADS
Type: `file_range`
Source: `crates/hydra-train/src/heads.rs:1-160`
Why it matters: Defines which heads are linear-on-pooled vs conv-on-spatial today.

## Artifact 11 — Backbone split into spatial and pooled outputs
Artifact id: `backbone-spatial-pooled`
Source label: BACKBONE
Type: `file_range`
Source: `crates/hydra-train/src/backbone.rs:137-145`
Why it matters: Minimal live seam showing backbone already exposes both spatial and pooled representations.

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
Why it matters: Current model-level test surface showing output shapes and finiteness, not routing-behavior oracle.

## Artifact 14 — Implementation roadmap: current head/output surface reference
Artifact id: `impl-roadmap-heads-context`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:172-329`
Why it matters: Reference-only live-snapshot context for current head surfaces, routing, model output shapes. Subordinate to reconciliation and code.

## Artifact 15 — Live code: sample carriers and batch collation for ExIt and DeltaQ
Artifact id: `sample-carriers-deltaq-exit`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:260-419`
Why it matters: Shows action-vector augmentation, target/mask pair safety, batch collation for ExIt, DeltaQ, adjacent advanced targets.

## Artifact 16 — Live code: replay DeltaQ sidecar producer and contract validation
Artifact id: `replay-deltaq-producer-full`
Source label: RQ
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_delta_q.rs:1-320`
Why it matters: Full offline DeltaQ producer, sidecar lookup, provenance/version checks, contract validation logic.

## Artifact 17 — Head activation controller state machine
Artifact id: `head-gates-state-machine`
Source label: GATES
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:556-735`
Why it matters: Core gating logic for coverage, sparse density, warmup, deferred conflict checks across advanced heads.

## Artifact 18 — Integration proof: RL batch carries ExIt and DeltaQ labels
Artifact id: `integration-deltaq-exit`
Source label: INTEG
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:149-250`
Why it matters: End-to-end proof that live RL batch conversion carries both ExIt and DeltaQ labels with expected target/mask contents.

## Artifact 19 — RL test: DeltaQ activates and participates after gating
Artifact id: `rl-deltaq-activation-test`
Source label: RLTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:726-775`
Why it matters: Behavior-level RL test showing DeltaQ enters warmup and affects RL path after activation.

## Artifact 20 — BC test: advanced auxiliary targets change loss
Artifact id: `bc-advanced-loss-test`
Source label: BCTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/bc.rs:857-944`
Why it matters: Behavior-level BC test showing advanced auxiliary targets materially change loss surface.

## Artifact 21 — Bridge search features and observation encoding with search context
Artifact id: `bridge-search-deltaq`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:341-417`
Why it matters: Shows how DeltaQ-like search features emit into fixed-superset observation and when search context counts as present.

## Artifact 22 — Live code: current DeltaQ validation harness
Artifact id: `deltaq-validation-current`
Source label: DQV
Type: `file_range`
Source: `crates/hydra-train/src/training/delta_q_validation.rs:174-396`
Why it matters: Shows current DeltaQ validation is structural (sample size, emission, coverage, support), not direct decision-quality proof.

## Artifact 23 — Live code: ExIt validation contrast
Artifact id: `exit-validation-contrast`
Source label: EXITV
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:218-358`
Why it matters: ExIt already has top-1 agreement and KL-style criteria, so right comparison point for what DeltaQ promotion still lacks.

## Artifact 24 — Live code: DeltaQ and SafetyResidual loss terms
Artifact id: `losses-deltaq-safety`
Source label: LOSSES
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:620-679`
Why it matters: Current masked regression wiring for DeltaQ and SafetyResidual inside shared loss breakdown.

## Artifact 25 — Live code: current evaluation surface
Artifact id: `eval-surface-current`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/eval.rs:1-220`
Why it matters: Existing evaluation helpers and benchmark gates. Useful for seeing what DeltaQ decision-quality promotion gate can and cannot reuse now.

## Artifact 26 — Train binary validation runner
Artifact id: `validation-runner-current`
Source label: VALRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/validation.rs:1-142`
Why it matters: Current validation pass and scoring surface in train binary; useful for seeing what already exists vs what DeltaQ promotion gate still needs.

</artifacts_manifest>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_30.md">
<![CDATA[**Hard winner: — add real DeltaQ decision-quality promotion gate.**

Confidence = **high** on *next-build choice*, **medium** on whether DeltaQ ultimately deserves permanent promotion. Why split: repo already has closed DeltaQ lane, but still lacks proof object justifying trust in that lane. Building that proof object is sharper next move than rewriting head routing now.

**Why wins on doctrine alignment**

Direct artifact support strong. Repo status says DeltaQ already “closed across both live RL and replay/offline BC paths” (README L72). Promoted reconciliation doctrine then says mainline priority is **truthful advanced-target / activation discipline after shipped supervision tranche** (RECON L0138-L0152, L0217-L0220), not fresh architecture turn. DeltaQ fits exactly: already in code, already wired into BC/RL/losses, still missing truthful promotion criteria. By contrast, routing correction explicitly labeled **strong archive architecture survivor; not current doctrine** (ROADMAP L0116).

That doctrine reading also best matches primary-source precedent. AlphaZero-style training improves policies by matching search-improved targets, not raw network priors; ExIt explicitly improves expert then distills stronger expert into apprentice; safe-policy-improvement work defines improvement relative to baseline policy, not structural target availability. ([ar5iv][1])

**Why wins on live code readiness**

Direct artifact support again favors DeltaQ already has:

* precise target object: masked `Q(child)-Q(root)` over discard-compatible actions (ANS23 L0511-L0517, L0551-L0603),
* live RL carriage (INTEG L0149-L0250),
* replay/offline sidecar generation with provenance/version checks (RQ L0034-L0108, L0171-L0302),
* BC/RL loss wiring (LOSSES L0630-L0661),
* activation/warmup logic (GATES L0559-L0565, L0691-L0729),
* behavior tests showing it enters training once activated (RLTEST L0727-L0775, BCTEST L0857-L0944).

Missing piece is not more plumbing. Missing piece is gate answering: **does this lane improve decisions enough to deserve promotion?**

Current validator does not answer that. `delta_q_validation.rs` only checks sample size, emission rate, coverage, supported-action count (DQV L0174-L0287). ExIt already has more decision-facing validator with KL and top-1 agreement (EXITV L0218-L0358). That asymmetry is gap.

**Why wins on blueprint precision**

has cleaner measurable object than B because DeltaQ semantics already exact. For fixed root state,

[
\Delta Q^*_s(a) = Q_s(\text{child}_a) - Q_s(\text{root})
]

So for any two compared actions (a,b),

[
\Delta Q^*_s(a)-\Delta Q^*_s(b)=Q_s(\text{child}_a)-Q_s(\text{child}_b)
]

Root term cancels. So DeltaQ already gives honest **per-root decision-quality comparator** on supported actions. No need invent vague surrogate.

That lets you build promotion gate directly from Hydra’s own validation doctrine. `HYDRA_FINAL` G0 says decision-improvement gate should require **mean (\Delta > 0)** and **<40% negative** (FINAL L0350-L0358). For DeltaQ, exact local version is:

1. Build held-out bank (S) of **hard, compatible discard states** with nonzero `delta_q_mask`.
2. For each state, restrict comparison to teacher-supported mask (M_s={a:m_s(a)=1}).
3. Let `a_base` be control policy’s argmax on (M_s), and `a_cand` candidate’s argmax on (M_s).
4. Define per-state lift:

[
\delta_s = \Delta Q^**s(a*{\text{cand}})-\Delta Q^**s(a*{\text{base}})
]

5. Promote only if:

   * existing structural validator passes first,
   * (\frac{1}{|S|}\sum_s \delta_s > 0),
   * (\Pr(\delta_s<0) < 0.40),
   * candidate not worse in downstream arena/self-play metrics.

That is much sharper than B, because B still needs unresolved semantic choices before you can even write acceptance test.

**Why B loses on blueprint precision right now**

Routing correction is real, but not as impl-ready as it looks.

Direct artifact support shows seam: backbone already exposes both `spatial` and `pooled` (BACKBONE L0138-L0144), yet `policy`, `delta_q`, `safety_residual` all read from `pooled` (MODEL L0424-L0437; HEADS L0010-L0018, L0141-L0160). So defect is visible.

But local patch is not semantically closed:

* Hydra’s external action space is **46**, not clean 34-tile discard head (Artifact 6; MODELTEST L0548-L0563).
* spatial surface is width **34** (BACKBONE L0142-L0144).
* archive’s proposed internal split is **37 discard + 9 global** (ANS21 L0675-L0680), which immediately raises explicit unresolved mapping problem for 3 aka actions.
* DeltaQ’s current contract **forbids aka support** and only validates masked entries on non-aka discard indices (RQ L0153-L0163).
* Bridge-side search DeltaQ features populate only over `NUM_TILE_TYPES` / 34 tile types (BRIDGE L0341-L0353).

So B is not “swap one layer and ship.” It needs explicit red-five policy, explicit treatment of tile-indexed auxiliaries whose support semantics differ, new routing-behavior tests. Today’s tests are shape/finiteness tests, not routing or semantic tests (HEADTEST L0305-L0425; MODELTEST L0548-L0667).

**Why wins on measurable acceptance criteria**

can be promoted with exact, reviewable gates.

Minimal impl would add this to `crates/hydra-train/src/training/delta_q_validation.rs`:

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

Right file-level build is narrow:

1. Keep existing structural report as prerequisite.
2. Add held-out RL-first state bank builder from existing search-derived DeltaQ labels.
3. Add candidate-vs-control comparison over that bank.
4. Wire promotion summary into `src/bin/train/validation.rs`.
5. Treat self-play/arena non-regression as second-stage gate.

Only materially missing proof object for is visible arena comparator. `eval.rs` exposes metrics and thresholds, but artifact set does not show concrete head-to-head runner (EVAL L0001-L0220). Honest statement: **offline decision-quality gating is ready to specify now; full promotion-to-default-on still needs visible comparison runner if one does not already exist elsewhere.**

**Why wins on likely leverage**

has better *expected value per engineering hour* because useful no matter answer.

If gate passes, you get justified reason to keep or strengthen already-shipped supervision lane. If gate fails, you stop pretending structurally healthy lane deserves promotion. That exactly matches “treat artifacts as evidence, not truth” posture from Artifact 1.

B has real upside, and outside evidence says so. Suphx especially matters here: it uses tile-column semantics, keeps separate 34-way discard model, explicitly avoids pooling in that model because pooling would lose tile-column meaning. But same paper also attributes gains to **global reward prediction, oracle guiding, runtime policy adaptation**; it is not evidence that mixed single-head 46-action routing correction should outrank target-truthfulness work in repo where DeltaQ already closed and routing semantics still partly unresolved. AlphaZero also weakens urgency case for B bit: it uses geometry-matched action representations, but reports flat action output for chess/shogi reached almost same final result, only training somewhat slower. ([ar5iv][2])

**Why loser loses right now**

B loses **now**, not forever.

Direct support says real defect: current policy path is pooled-only, and archive correctly preserved “stop routing tile-indexed policy through pooled-only features” as strongest architecture survivor (ARCHIVE L0068; ANS21 L0647-L0680). Mahjong-specific primary evidence from Suphx makes that survivor stronger, not weaker. ([ar5iv][2])

But loser loses on three concrete points:

1. **Not mainline doctrine now.** Promoted reconciliation doc puts truthful target promotion on critical path and leaves routing correction as archive-side later work.
2. **Not semantically closed.** 34-column spatial surface does not directly resolve Hydra’s 46-action mixed interface, especially aka handling and DeltaQ’s non-aka contract.
3. **No crisp strength gate yet.** No visible routing-behavior oracle or no-regression harness tailored to this patch.

better narrowed version of B, later**

If you ignore this rec and insist on routing lane, do **not** take full “tile-indexed outputs” bundle first.

Better local version is:

* policy discard split first,
* keep fixed 46 external interface,
* leave history-sidecar work out,
* leave DeltaQ and SafetyResidual alone until their action-index semantics are explicit.

That is narrower and stronger than touching all tile-indexed outputs at once.

**Bottom line**

Keep Narrow it to **RL-first DeltaQ decision-quality promotion gate** built on existing search-derived masked target, with structural thresholds as prerequisites and G0-style decision-lift test as core acceptance criterion.

Do **not** promote B next. Keep it as strongest architecture-side reserve patch, and when you come back to it, narrow it to **policy discard routing only** before touching rest of 46-action sur

face.

[1]: https://ar5iv.org/pdf/1712.01815 "https://ar5iv.org/pdf/1712.01815"
[2]: https://ar5iv.org/abs/2003.13590 "[2003.13590] Suphx: Mastering Mahjong with Deep Reinforcement Learning1footnote 11footnote 1This work was conducted at Microsoft Research Asia. 2nd, 4th, 5th, and 6th authors were interns at Microsoft Research Asia then."
]]>
</answer_text>
</answer_section>
</combined_run_record>