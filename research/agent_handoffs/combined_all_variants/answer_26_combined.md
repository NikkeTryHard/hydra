<combined_run_record run_id="answer_26" variant_id="delta-q-provenance-balanced" schema_version="1">
  <metadata>
    <notes>Compact self-contained combined record for Agent 26 replay/offline delta_q provenance closure research. It preserves the rendered prompt shell reconstructed from the authoritative generator config, the preserved Agent 26 answer.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — replay/offline delta_q provenance closure blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for replay/offline delta_q provenance and staged activation closure in Hydra, but do not treat the current framing as automatically complete or correct.

We want a detailed answer that makes clear:
- what the authority docs directly require versus what the current code merely makes possible
- what the existing live RL delta_q lane proves and what it still does not prove
- what exact replay/offline provenance object would be needed to close the lane without semantic guessing
- whether that object can now be specified implementation-ready from current Hydra evidence alone
- what should be kept, narrowed, or rejected relative to the replay ExIt sidecar pattern
- what confidence level each major conclusion deserves and why
- how to implement or validate the surviving path with minimal guesswork and no doctrinal drift

This is not a broad future-of-Hydra prompt.
Stay inside the current supervision-first direction.
Do not restart architecture search.
Do not widen the model surface.
Do not broaden AFBS.
If the lane still cannot be fully closed, identify the smallest decisive missing contract artifact or design spec that would unblock it.

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- distinguish direct artifact support from inference
- include formulas when needed
- include code-like detail when helpful
- keep the answer actionable and auditable
- for important claims, make the validation path visible enough that a reviewer can test it later
- if confidence is high, justify that confidence from evidence
- do not treat typed target surfaces, heads, or existing tensors as proof that a teacher object is semantically valid or activation-ready
- do not let archive artifacts silently upgrade a missing repo contract into a settled plan
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.

Special caution:
- authority order is README.md -> research/design/HYDRA_FINAL.md -> research/design/HYDRA_RECONCILIATION.md -> docs/GAME_ENGINE.md
- archive/handoff artifacts are evidence only and must not outrank the authority chain
- current typed target support in train-side structs is not by itself proof of semantic closure
</artifact_note>

<scope_note>
Broad exploration has already happened. Do not redo repo-wide future-planning from scratch.
Start from the current authority docs, the live code paths, the replay ExIt sidecar contract, the RL-only delta_q lane, and the cleaned archive evidence.
Use new retrieval only to validate, falsify, or sharpen the exact replay/offline delta_q provenance closure question.
</scope_note>

<artifacts_manifest>

## Artifact 01 — Task scope and required outcome
Artifact id: `task-scope`
Source label: TASK
Type: `literal`
Why it matters: Pins the research task to the exact open Hydra lane: close the replay/offline delta_q provenance question completely, or prove the exact blocker and smallest decisive next design artifact.

## Artifact 02 — Current repo status and immediate needs
Artifact id: `authority-readme`
Source label: README
Type: `file_range`
Source: `README.md:60-63`
Why it matters: High-authority current status: replay/sample ExIt lane is real, live RL delta_q lane is real, replay/offline delta_q remains blocked, and immediate needs are stronger belief-teacher semantics plus staged delta_q closure.

## Artifact 03 — Current tranche doctrine for mjai_loader, losses, BC, RL, and bridge
Artifact id: `authority-reconciliation`
Source label: REC
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:432-542`
Why it matters: This is the active execution doctrine for the exact open lane. It says delta_q staging must remain explicit, replay/sample delta_q is still absent, belief stays default-off until stronger teachers exist, and no new heads or broad AFBS rewrite belong in this tranche.

## Artifact 04 — Architecture target for belief/search features and Hand-EV oracle surface
Artifact id: `authority-final-hand-ev`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:78-88`
Why it matters: Provides the north-star semantics for Group C dynamic search/belief features and Group D Hand-EV. Useful contrast so the agent does not confuse a live interface with a semantically closed teacher object.

## Artifact 05 — Live 192x34 runtime encoder contract
Artifact id: `runtime-game-engine`
Source label: ENG
Type: `file_range`
Source: `docs/GAME_ENGINE.md:122-177`
Why it matters: Shows the current fixed-shape runtime surface and makes clear that Group C/D are already live fixed extensions, which prevents the agent from pretending this is still a blank-slate architecture task.

## Artifact 06 — Bridge-side CT-SMC-weighted Hand-EV and search feature construction
Artifact id: `core-bridge-hand-ev`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:263-347`
Why it matters: Shows the live runtime split between CT-SMC-weighted Hand-EV and Group C search/belief features. Useful to keep the research grounded in which signals are real runtime inputs versus train-side targets.

## Artifact 07 — Current Hand-EV implementation semantics
Artifact id: `core-hand-ev-live`
Source label: HEV
Type: `file_range`
Source: `crates/hydra-core/src/hand_ev.rs:253-309`
Why it matters: This gives the exact current Hand-EV math so the agent can compare a live but heuristic feature lane against the semantically blocked teacher lanes.

## Artifact 08 — Current Stage-A belief teacher object
Artifact id: `train-belief-stage-a`
Source label: BELIEF
Type: `file_range`
Source: `crates/hydra-train/src/teacher/belief.rs:1-152`
Why it matters: Critical contrast artifact. The belief lane already has a placeholder teacher object, but it is intentionally weak: uniform kernel, equal hidden-zone split, heuristic trust gating. This helps the agent distinguish carrier readiness from teacher credibility.

## Artifact 09 — Live ExIt target semantics and safety gates
Artifact id: `train-exit-contract`
Source label: EXIT
Type: `file_range`
Source: `crates/hydra-train/src/training/exit.rs:1-220`
Why it matters: Gold-standard in-repo reference for what a semantically closed teacher lane looks like: exact object, exact mask, exact coverage gates, exact safety valves, and compatible-discard constraints.

## Artifact 10 — Canonical delta_q builder and batch collation
Artifact id: `train-delta-q-builder`
Source label: DQOBJ
Type: `file_range`
Source: `crates/hydra-train/src/training/exit.rs:229-357`
Why it matters: This is the exact current delta_q object contract in code: masked [46], discard-compatible only, Q(child)-Q(root), with zero rows for absent samples at batch collation. It is the starting point for any replay/offline provenance closure blueprint.

## Artifact 11 — Exit and delta_q unit tests around gating and masked object shape
Artifact id: `train-exit-and-deltaq-tests`
Source label: EXITTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/exit.rs:378-620`
Why it matters: These tests prove what the current builder paths already enforce and what they do not. Good for separating mathematically closed local object semantics from still-open replay/offline provenance semantics.

## Artifact 12 — Replay loader builds Stage-A belief and ExIt sidecar joins but leaves delta_q absent
Artifact id: `train-replay-loader-gap`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:320-459`
Why it matters: This is the exact replay/offline delta_q gap in current code. Replay samples already build Stage-A belief targets and joined ExIt sidecar labels, but still hardcode delta_q_target and delta_q_mask to None.

## Artifact 13 — Train-bin activation policy for advanced losses
Artifact id: `train-loss-policy-gates`
Source label: POLICY
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs:1-59`
Why it matters: Shows the exact policy boundary today: BC/train.rs rejects belief_fields, mixture_weight, opponent_hand_type, and delta_q, while RL loss config can carry delta_q. This is the policy seam any closure blueprint must respect or intentionally change.

## Artifact 14 — Train-bin tests that hard-block belief and delta_q activation
Artifact id: `train-loss-policy-tests`
Source label: TPOLTEST
Type: `file_range`
Source: `crates/hydra-train/src/bin/train.rs:1039-1102`
Why it matters: This makes the current staging policy concrete: train.rs rejects belief and delta_q even at zero weight, so any blueprint that proposes activation has to justify changing a deliberate guardrail rather than pretending it is an accidental omission.

## Artifact 15 — Replay ExIt sidecar contract and provenance-enforced join pattern
Artifact id: `train-replay-exit-contract`
Source label: REXIT
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_exit.rs:1-260`
Why it matters: This is the exact offline sidecar pattern that already works in Hydra: semantics string, provenance label, replay decision identity, action key, legal-mask digest, source net hash/version, and join-time rejection on mismatch. Use it as a comparison target, not a loose analogy.

## Artifact 16 — Replay ExIt sidecar tests for contract matching and loader population
Artifact id: `train-replay-exit-tests`
Source label: REXITTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/replay_exit.rs:330-507`
Why it matters: These tests show the exact sort of contract-checking behavior replay/offline delta_q would need if it follows the ExIt sidecar pattern honestly: mismatch rejection, action disambiguation, provenance tagging, and loader join success only under matching keys.

## Artifact 17 — RL-only delta_q validation harness and thresholds
Artifact id: `train-delta-q-validation`
Source label: DQV
Type: `file_range`
Source: `crates/hydra-train/src/training/delta_q_validation.rs:1-260`
Why it matters: Shows that delta_q is not fully fake: the live RL lane already has a structural validation harness, coverage metrics, emission-rate thresholds, and acceptance logic. The question is what exact replay/offline provenance contract should sit in front of staged activation.

## Artifact 18 — Shared live search-label producer for ExIt and delta_q
Artifact id: `train-live-search-producer`
Source label: LIVE
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:211-420`
Why it matters: This is the reusable root-only producer envelope that already emits both ExIt and delta_q from the same search call. It matters because any replay/offline closure that diverges from this envelope needs a strong reason.

## Artifact 19 — Live ExIt and delta_q producer tests
Artifact id: `train-live-search-tests`
Source label: LIVETEST
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:720-839`
Why it matters: These tests prove the shared live producer emits valid masked labels, that delta_q ordering behaves sanely on good input, and that visit-based ExIt labels are not q-softmax aliases. Good reality check for any replay/offline reuse proposal.

## Artifact 20 — Head-gating core: advanced-head taxonomy, target presence, and coverage accounting
Artifact id: `train-head-gates-core`
Source label: GATEA
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1-320`
Why it matters: This is the real gate pack already in repo. It matters because replay/offline delta_q activation cannot be discussed honestly without the implemented sparse-head accounting and target-presence logic.

## Artifact 21 — Head-gating controller: density gates, conflict checks, and activation transitions
Artifact id: `train-head-gates-controller`
Source label: GATEB
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:321-680`
Why it matters: This extends the gate pack into the actual controller logic. Useful for asking whether delta_q replay/offline closure should plug into the existing activation controller unchanged or whether some narrower gating policy is needed.

## Artifact 22 — Head-gates behavior tests for activation, sparse delta_q rejection, and approved weights
Artifact id: `train-head-gates-tests`
Source label: GATETEST
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:1000-1259`
Why it matters: This test block gives the most concrete evidence about how the implemented controller behaves for sparse delta_q and dense residual heads. Useful for determining whether replay/offline delta_q closure can rely on the current gate pack as-is.

## Artifact 23 — RL-side gating, delta_q batch stats, and microbatched train-step path
Artifact id: `train-rl-gating-path`
Source label: RL
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:1-320`
Why it matters: Shows the live RL consumption path, optional-target pair validation, head-gating hookup, and aux-loss integration. Important for distinguishing what already works in RL from what is still blocked in replay/offline and BC/train-bin.

## Artifact 24 — ExIt validation harness core report and thresholds
Artifact id: `train-exit-validation-core`
Source label: EVALA
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:1-320`
Why it matters: This is the strongest in-repo example of a validation harness for a search-derived label lane. It defines metrics, thresholds, rejection accounting, and report structure that replay/offline delta_q closure would likely need to mirror or consciously differ from.

## Artifact 25 — ExIt validation step collector and observational runner
Artifact id: `train-exit-validation-runner`
Source label: EVALB
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:321-559`
Why it matters: Shows how Hydra runs an observational validation pass without letting labels train the model. Good pattern reference if replay/offline delta_q needs a similar shadow-validation tranche before activation.

## Artifact 26 — ExIt validation harness tests
Artifact id: `train-exit-validation-tests`
Source label: EVALTEST
Type: `file_range`
Source: `crates/hydra-train/src/training/exit_validation.rs:560-811`
Why it matters: Concrete proof of the report/default/pass-fail semantics Hydra already considers sufficient for a search-label validator. Helps a research agent avoid inventing a validation style detached from repo norms.

## Artifact 27 — Arena-side delta_q carried-label invariants
Artifact id: `core-arena-deltaq-invariants`
Source label: ARENA
Type: `file_range`
Source: `crates/hydra-core/src/arena.rs:559-600`
Why it matters: Strong invariant block for carried delta_q labels: binary mask, finite values, legal-only, discard-only, no aka discard slots, no nonzero values outside mask, and non-empty support. Great evidence for the narrow v1 object contract.

## Artifact 28 — Self-play RL batch carries delta_q targets today
Artifact id: `train-selfplay-delta-q`
Source label: RLBATCH
Type: `file_range`
Source: `crates/hydra-train/src/selfplay.rs:436-491`
Why it matters: Proof that the live self-play RL path already carries delta_q_target and delta_q_mask into HydraTargets, so the open problem is not output surface existence but replay/offline provenance closure.

## Artifact 29 — Batch carrier and collation support for delta_q, belief, and ExIt
Artifact id: `train-sample-carriers`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:150-454`
Why it matters: Important typed-surface artifact: the carrier layer already supports all these targets, including augmentation and collation. The agent must not mistake that for semantic closure.

## Artifact 30 — HydraTargets and advanced loss surface
Artifact id: `train-loss-target-surface`
Source label: LOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:1-191`
Why it matters: Shows the exact training target and loss-weight surface for advanced heads. Useful for keeping the blueprint inside the current model surface with no head expansion.

## Artifact 31 — Belief BCE, mixture CE, dense regression, and masked action MSE semantics
Artifact id: `train-loss-masked-semantics`
Source label: LOSSSEM
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:276-330`
Why it matters: High-value semantic artifact: belief currently uses BCE-style field loss, while delta_q has a masked action MSE path. This helps the research agent reason about why belief and delta_q are blocked for very different reasons.

## Artifact 32 — Integration test proof that RL delta_q and ExIt carriers already work
Artifact id: `train-integration-proof`
Source label: ITEST
Type: `file_range`
Source: `crates/hydra-train/tests/integration_pipeline.rs:149-250`
Why it matters: Fast end-to-end proof that trajectory labels, RL batch collation, exit_target, and delta_q_target/mask already survive into testable tensors. Again: typed proof, not replay/offline semantic closure proof.

## Artifact 33 — Orchestrator maintenance plan and live enablement seam
Artifact id: `train-orchestrator-enable-seam`
Source label: ORCH
Type: `file_range`
Source: `crates/hydra-train/src/training/orchestrator.rs:162-208`
Why it matters: Small but useful seam showing how ExIt producer enablement is already wired through the maintenance plan. Helps the research agent discuss whether replay/offline delta_q should mirror that style or require a more explicit staging path.

## Artifact 34 — Replay regression test that keeps delta_q absent
Artifact id: `train-replay-deltaq-absence-test`
Source label: LOADTEST
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:783-792`
Why it matters: Tiny but important artifact: replay/offline delta_q absence is not accidental; it is regression-tested. Any closure blueprint has to account for deliberately changing this invariant.

## Artifact 35 — Testing doctrine: why bad labels are catastrophic
Artifact id: `testing-doctrine`
Source label: TEST
Type: `file_range`
Source: `research/design/TESTING.md:1-15`
Why it matters: Short but high-value doctrine reminder: training-data bugs are worse than ordinary app bugs because they silently create confidently wrong models. Good prompt anchor for why provenance and validation matter so much here.

## Artifact 36 — Current archive-derived tracking roadmap with stale rows already corrected
Artifact id: `archive-roadmap-current`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:1-220`
Why it matters: This is not SSOT, but it is the cleaned archive tracking surface. Use it as evidence about how prior research has been normalized, not as authority over README or reconciliation.

## Artifact 37 — Prior narrow delta_q validation and do-not-do blueprint
Artifact id: `archive-delta-q-blueprint`
Source label: A23
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_23_combined.md:820-1249`
Why it matters: High-signal archive evidence, but not doctrine. This artifact is useful because it already sharpens object correctness, carrier correctness, activation correctness, and a strong do-not-do list for delta_q. It still must be revalidated against current code and docs.

## Artifact 38 — Prior RL-only delta_q validation-and-enable contract
Artifact id: `archive-delta-q-rl-contract`
Source label: A25
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_25_combined.md:1-180`
Why it matters: Useful archive artifact for the narrow live RL lane. It should help the agent distinguish what is already fairly settled in the RL-only delta_q contract from what is still missing in replay/offline provenance closure.

## Artifact 39 — Prior RL-only delta_q artifact manifest extension
Artifact id: `archive-delta-q-manifest-extension`
Source label: A25X
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_25_combined.md:180-290`
Why it matters: Extra archive context that points at ExIt validation harnesses, RL tests, replay-absence guards, train-bin rejection tests, and canonical archive rows. Useful as an evidence map, but still subordinate to the live repo files it points at.

## Artifact 40 — Prior target-provenance registry and semantic-state proposal
Artifact id: `archive-provenance-registry-idea`
Source label: A18
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_18_combined.md:1167-1290`
Why it matters: Not doctrine, but a high-signal artifact about making semantic status explicit. Helpful if the agent wants a registry-like mechanism to separate Ready, SemanticallyWrong, TypedHole, and DiagnosticOnly targets without upgrading archive proposals into truth.

## Artifact 41 — Archive warning example and historical target-provenance artifact
Artifact id: `archive-belief-warning`
Source label: A15
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md:1-40`
Why it matters: Short artifact showing the warning pattern added to stale combined files. It reinforces the rule that archive artifacts are evidence only and may be historically useful while still stale on live-state framing.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_26.md">
  <![CDATA[The public repo still exposes the same governing order and status that your artifact bundle describes: `HYDRA_FINAL.md`, then `HYDRA_RECONCILIATION.md`, then `docs/GAME_ENGINE.md`, and the README still says replay/sample ExIt is live, live self-play RL `delta_q` is live, while replay/offline `delta_q` and train-bin activation remain blocked pending a separate provenance/validation tranche. ([GitHub][1])

## 1. Decision

Implement replay/offline `delta_q` as a **search-derived replay sidecar** that reuses the **existing shared root-search producer** and joins into replay samples with the **same replay identity / legal-mask / source-version discipline** as ExIt.

Do **not**:

* build `delta_q` inside `mjai_loader.rs` from replay/public data,
* lift the runtime Group-C `[34]` `delta_q` feature plane into a teacher,
* create a second search semantics for replay,
* add heads,
* broaden AFBS,
* unblock `train.rs` in the same patch.

### Final verdict

* **Replay/offline provenance closure:** **repo-backed enough to implement now**.
* **Full BC/train-bin activation closure:** **still blocked**, but the blocker is no longer `delta_q` semantics or replay provenance.
* **Smallest decisive missing artifact:** a narrow **BC/train advanced-head activation hook spec** that wires `HeadActivationController` into BC/train and defines warmup-time trunk detachment for sparse heads like `DeltaQ`.

That blocker is real: public `bc.rs` still uses `HydraLoss` directly and has no visible `HeadActivationController`, `approved_loss_config`, or `extract_target_presence` integration, so BC-side `delta_q` enablement would currently bypass Hydra’s own sparse-head activation discipline. ([GitHub][2])

---

## 2. Authority boundary: what is required vs what is merely possible

### Direct artifact support — high confidence

1. **Replay/offline `delta_q` is intentionally absent today.**
   Support: README status, REC staging rule, loader hardcoded `None`, replay absence regression test, train-bin rejection tests.
   Validation path: `README L0062`, `REC L0435-L0449`, `LOADER L0451-L0454`, `LOADTEST L0783-L0792`, `POLICY L0018-L0023`, `TPOLTEST L1083-L1102`.

2. **The surviving teacher object is already semantically narrow.**
   It is masked action-space `[46]`, discard-compatible only, `Q(child)-Q(root)`, emitted by the shared root-search producer.
   Validation path: `DQOBJ L0263-L0303`, `LIVE L0281-L0366`, `ARENA L0559-L0600`, `LIVETEST L0761-L0781`.

3. **No new heads / no AFBS broadening / no runtime-bridge lift are allowed in this tranche.**
   Validation path: `REC L0493-L0498`, `REC L0504-L0508`, `REC L0512-L0515`, `REC L0537-L0538`.

4. **Loss surface and carriers already exist.**
   `HydraTargets`, `MjaiSample`, `MjaiBatch`, model head, masked MSE, RL batch collation all already carry `delta_q`.
   Validation path: `LOSS L0021-L0032`, `SAMPLE L0157-L0164`, `SAMPLE L0399-L0414`, `LOSSSEM L0321-L0330`, `RLBATCH L0477-L0485`, `ITEST L0229-L0250`.

### Code only makes this possible — not closed

1. **Typed carriers are not proof of replay/offline semantic closure.**
   `HydraTargets.delta_q_target` existing does not prove a teacher object exists for replay.
   Validation path: contrast `LOSS L0027-L0028` with `LOADER L0453-L0454`.

2. **The live RL lane proves object correctness and RL transport, not replay provenance.**
   Validation path: `DQV L0001-L0260`, `RL L0026-L0079`, `RLBATCH L0437-L0490`.

3. **The runtime Group-C `delta_q` feature plane is not a replay teacher.**
   It is a live encoder/runtime feature family, not an offline provenance contract.
   Validation path: `FINAL L0078-L0088`, `BRIDGE L0301-L0347`, `REC L0504-L0508`.

### Inference — medium confidence

1. **Replay/offline closure should follow the ExIt sidecar pattern, but not identically.**
   Keep the replay identity / provenance / version checks. Narrow the schema to signed regression semantics.
   Why only medium: authority docs require provenance-explicit closure, but they do not spell out the exact `delta_q` record shape.
   Validation path: implement roundtrip tests against ExIt-style join discipline.

---

## 3. Exact surviving v1 `delta_q` teacher object

### 3.1 Object semantics — direct support, high confidence

Let `s` be a replay decision state after replay reconstruction, and let `root` be the AFBS root built by the existing shared search producer.

Define the supported action set

[
\mathcal{A}_{\Delta q}(s)=\left{a \mid
\begin{array}{l}
a \le \texttt{DISCARD_END},\
a \notin {\texttt{AKA_5M},\texttt{AKA_5P},\texttt{AKA_5S}},\
\text{legal}(a)=1,\
\text{child}(a)\ \text{exists at the seeded root},\
N(\text{child}(a)) > 0,\
Q(root),Q(\text{child}(a))\in \mathbb{R}
\end{array}
\right}
]

Then the target is

[
m_a = \mathbf{1}[a\in \mathcal{A}_{\Delta q}(s)]
]

[
\Delta q^*_a = m_a \cdot \left(Q(\text{child}(a)) - Q(root)\right)
]

All unmasked actions must satisfy:

[
m_a = 0 \implies \Delta q^*_a = 0
]

This is not a distribution. It is not normalized. It is signed regression over a masked subset of the `[46]` action surface.
Support: `DQOBJ L0263-L0303`, `ARENA L0559-L0600`, `LOSSSEM L0321-L0330`.
Confidence is high because builder code, invariants, and loss semantics all agree.

### 3.2 Emission envelope — direct support, high confidence

The object above is only valid when emitted through the current shared root-search lane:

* compatible discard-only state,
* at least 2 legal discards,
* base policy computed from raw logits,
* hard-state gate,
* all legal discard children seeded at the root,
* learner-only value-head child evaluation,
* root-only AFBS search,
* `None` on any failed gate.

Support: `LIVE L0294-L0366`.
Confidence high because the producer code already enforces this exact envelope.

### 3.3 Worked example

If legal discard actions are `{1,2,5}`, root search yields:

* `Q(root)=0.40`
* `Q(child_1)=0.90`
* `Q(child_2)=0.10`
* child `5` exists but has `visit_count=0`

then

* `mask[1]=1`, `target[1]=+0.50`
* `mask[2]=1`, `target[2]=-0.30`
* `mask[5]=0`, `target[5]=0.0`
* all non-discard / aka / illegal actions: `mask=0`, `target=0`

Coverage is

[
\text{coverage} = \frac{2}{3}
]

This matches the builder semantics in `DQOBJ L0284-L0302`.

---

## 4. What the existing live RL lane proves, and what it does not prove

### Proved now — direct support, high confidence

1. **The teacher object is exact enough to name.**
   `Q(child)-Q(root)`, masked discard-only, finite, legal-only.
   Support: `DQOBJ`, `ARENA`.

2. **One search can emit both ExIt and `delta_q`.**
   Support: `LIVE L0349-L0366`.

3. **RL transport works.**
   Sidecar-free self-play already collates `delta_q_target`/`delta_q_mask` into `HydraTargets`.
   Support: `RLBATCH L0437-L0485`, `ITEST L0229-L0250`.

4. **Masked regression already exists.**
   Support: `LOSSSEM L0321-L0330`.

5. **There is already a structural `delta_q` validator.**
   Support: `DQV L0014-L0260`.

### Not proved now — direct support, high confidence

1. **Replay/offline provenance is not closed.**
   Loader still assigns `None`.
   Support: `LOADER L0451-L0454`, `LOADTEST L0783-L0792`.

2. **BC/train-bin activation is not safe just because the tensors exist.**
   `train.rs` still rejects `advanced_loss.delta_q`.
   Support: `POLICY L0018-L0023`, `TPOLTEST L1083-L1102`.

3. **Typed surfaces do not prove semantic readiness.**
   Support: `LOSS`, `SAMPLE`, contrasted with loader absence and policy rejection.

4. **Runtime `[34]` `delta_q` feature planes are not replay teachers.**
   Support: `FINAL L0078-L0088`, `REC L0504-L0508`.

---

## 5. Replay/offline provenance object: exact v1 contract

## 5.1 Chosen v1 schema

```rust
pub const REPLAY_DELTA_Q_SEMANTICS_V1: &str =
    "delta_q_root_child_minus_root_q_v1";
pub const REPLAY_DELTA_Q_PROVENANCE: &str = "search-derived";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDeltaQLookupKey {
    pub replay: ReplayDecisionKey, // keep ExIt replay identity exactly
    pub action: u8,                // sample discriminator only
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDeltaQRecordV1 {
    pub version: u32,              // schema version; must be 1
    pub semantics: String,         // must equal REPLAY_DELTA_Q_SEMANTICS_V1
    pub provenance: String,        // must equal REPLAY_DELTA_Q_PROVENANCE

    pub key: ReplayDecisionKey,    // {source_hash, event_index, actor, obs_hash}
    pub action: u8,                // replay action id; not teacher-defining
    pub legal_mask_digest: u64,    // digest of replay sample legal mask
    pub source_net_hash: u64,      // checkpoint identity used for search/value eval
    pub source_version: u32,       // producer/model version

    pub search_budget: u32,        // exact known quantity from budget_from_legal_count()
    pub legal_discard_count: u8,
    pub supported_actions: u8,
    pub coverage: f32,             // supported_actions / legal_discard_count

    pub target: Vec<f32>,          // len == HYDRA_ACTION_SPACE
    pub mask: Vec<f32>,            // len == HYDRA_ACTION_SPACE
}
```

### Why this exact object survives

**Keep from ExIt** — direct/inference mix, confidence high-to-medium:

* `ReplayDecisionKey`
* action discriminator
* `legal_mask_digest`
* `source_net_hash`
* `source_version`
* explicit `semantics`
* explicit `provenance`
* vector `target` + vector `mask`
* loader-time mismatch rejection

**Narrow from ExIt** — medium confidence, justified by semantics:

* `search_budget` instead of blindly reusing ExIt’s audit field shape as “root visits”; budget is the exact quantity exposed by the current replay-sidecar producer code path.
* signed regression target, not a normalized policy distribution.
* no KL field.
* no top-1 agreement field in the record.

**Remove from ExIt** — high confidence:

* `kl_to_base`
* any visit-distribution interpretation
* any q-softmax alias
* any action-probability semantics

### Why `action` stays in v1

This is **inference**, confidence **medium**:

* semantically, `delta_q` is state-rooted, not action-conditioned;
* operationally, keeping `action` in the lookup key preserves loader symmetry with ExIt and avoids a wider replay-join refactor.

If later deduplication matters, `v2` can drop `action` and key only by `ReplayDecisionKey`.
For `v1`, keeping it is the lower-risk patch.

### Simpler local alternative

Key only by `ReplayDecisionKey` and make `action` audit-only.
Not chosen in `v1` because it changes the current loader/join symmetry for little immediate gain.

---

## 6. Exact join contract

## 6.1 Join-time lookup

```rust
pub fn lookup_label(
    &self,
    key: &ReplayDecisionKey,
    action: u8,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    source_net_hash: u64,
    source_version: u32,
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>
```

## 6.2 Required checks

Reject the record unless **all** checks pass:

```rust
record.version == 1
record.semantics == REPLAY_DELTA_Q_SEMANTICS_V1
record.provenance == REPLAY_DELTA_Q_PROVENANCE
record.legal_mask_digest == legal_mask_digest_from_f32(legal_mask)
record.source_net_hash == source_net_hash
record.source_version == source_version
record.target.len() == HYDRA_ACTION_SPACE
record.mask.len() == HYDRA_ACTION_SPACE
```

Then perform **delta_q-specific structural validation** before accepting the label:

```rust
fn delta_q_contract_ok(
    target: &[f32; HYDRA_ACTION_SPACE],
    mask: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
) -> bool {
    let mut saw_masked = false;

    for a in 0..HYDRA_ACTION_SPACE {
        let m = mask[a];
        let t = target[a];

        if !(m == 0.0 || (m - 1.0).abs() < 1e-6) {
            return false;
        }
        if !t.is_finite() {
            return false;
        }

        if m > 0.5 {
            saw_masked = true;

            if legal_mask[a] <= 0.0 {
                return false;
            }
            if a > DISCARD_END as usize {
                return false;
            }
            if matches!(a as u8, AKA_5M | AKA_5P | AKA_5S) {
                return false;
            }
        } else if t.abs() > 1e-5 {
            return false;
        }
    }

    saw_masked
}
```

Also verify the derived metadata:

```rust
let supported_actual = mask.iter().filter(|&&m| m > 0.5).count() as u8;
if supported_actual != record.supported_actions {
    return None;
}
if record.legal_discard_count == 0 {
    return None;
}
let expected_coverage = supported_actual as f32 / record.legal_discard_count as f32;
if (record.coverage - expected_coverage).abs() > 1e-4 {
    return None;
}
if record.search_budget == 0 {
    return None;
}
```

### Confidence

High.
Why: every structural check above is already implied by the current arena invariants or builder semantics; this adds no new teacher meaning, only explicit provenance rejection.
Validation path: `ARENA L0559-L0600`, `DQOBJ L0263-L0303`.

---

## 7. File-level implementation

## 7.1 `crates/hydra-train/src/training/replay_delta_q.rs` — new file

### Required content

1. Reuse from `replay_exit.rs`:

   * `ReplayDecisionKey`
   * `source_hash_from_identity`
   * `source_net_hash_from_checkpoint_identity`
   * `legal_mask_digest_from_f32`

2. Define:

   * `REPLAY_DELTA_Q_SEMANTICS_V1`
   * `REPLAY_DELTA_Q_PROVENANCE`
   * `ReplayDeltaQLookupKey`
   * `ReplayDeltaQRecordV1`
   * `DeltaQSidecarIndex`
   * `lookup_label`
   * `from_jsonl_reader`
   * `from_jsonl_path`

3. Add producer:

```rust
pub fn generate_replay_delta_q_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,      // keep shared producer config exactly
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)>
```

### Producer algorithm

* reconstruct replay state exactly like `replay_exit.rs`
* for each sampled event:

  * build `obs`
  * map replay action to Hydra action
  * build `legal_mask`
  * encode observation
  * create `RootDecisionContext`
  * create `ReplayDecisionKey`
  * reconstruct validator counters:

    * `compatible_discard_state`
    * legal discard count
    * hard-state gate from base logits
  * call **shared** producer:

```rust
let labels = try_search_labels_from_context(
    &state,
    &obs,
    &ctx,
    &safety[actor],
    exit_cfg,
    &mut |obs_encoded| model.policy_value_cpu(obs_encoded, device),
    &mut adapter,
);
```

* take `labels.and_then(|l| l.delta_q)`
* if `None`, update rejection counters and continue
* if `Some(delta_q)`:

  * compute `supported_actions`
  * compute `coverage`
  * update `DeltaQValidationReport`
  * emit `ReplayDeltaQRecordV1`

### Keep / narrow / remove at producer

**Keep**

* shared search call
* learner-only value head
* same hard-state gate
* same all-legal discard seeding
* `None` on failed gate

**Narrow**

* record only `delta_q`, not ExIt target
* audit field `search_budget`, not `kl_to_base`

**Remove**

* no second search
* no replay-derived fallback label
* no bridge-plane fallback

### Simpler and stronger local alternatives

* **Simpler**: implement `replay_delta_q.rs` as a sibling of `replay_exit.rs` and tolerate duplicate search when ExIt and `delta_q` sidecars are generated separately.
  Confidence: high. Low churn.

* **Stronger**: factor a shared internal replay-search helper returning `TrajectorySearchLabels`, and let ExIt / `delta_q` writers consume one search result.
  Confidence: medium. Better compute hygiene, bigger patch.

Choose the **simpler** one for the first patch unless replay-sidecar generation is routinely dual-lane.

---

## 7.2 `crates/hydra-train/src/data/mjai_loader.rs`

### New loader entrypoint

Add a new narrow API, keep the existing ExIt-only and plain-loader APIs intact:

```rust
pub struct ReplaySidecars<'a> {
    pub exit: Option<&'a ExitSidecarIndex>,
    pub delta_q: Option<&'a DeltaQSidecarIndex>,
}
```

or, if you want less API surface churn, add:

```rust
pub fn load_game_from_events_with_sidecars(
    source_identity: &str,
    source_net_hash: u64,
    source_version: u32,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame>
```

### Join logic

Inside the sampled replay path:

```rust
let replay_key = source_hash.map(|source_hash| ReplayDecisionKey {
    source_hash,
    event_index: idx as u32,
    actor: actor as u8,
    obs_hash: crate::training::live_exit::obs_hash(&obs_encoded),
});

let joined_delta_q = replay_key.and_then(|key| {
    delta_q_sidecar.and_then(|sidecar| {
        source_net_hash.zip(source_version).and_then(|(source_net_hash, source_version)| {
            sidecar.lookup_label(
                &key,
                hydra_action.id(),
                &legal_mask,
                source_net_hash,
                source_version,
            )
        })
    })
});
```

Populate **both** fields from the **same** `Option`:

```rust
delta_q_target: joined_delta_q.map(|(target, _)| target),
delta_q_mask: joined_delta_q.map(|(_, mask)| mask),
```

### Loader invariants

* no sidecar => `None, None`
* any contract mismatch => `None, None`
* plain replay loader remains unchanged and keeps `delta_q` absent
* do not compute `delta_q` in loader code
* do not silently fabricate a zero mask with nonzero target

### Confidence

High.
Why: this is exactly the already-proven ExIt replay join pattern, narrowed to the `delta_q` object.
Validation path: mirror `REXITTEST` with `delta_q`-specific contract tests.

---

## 7.3 `crates/hydra-train/src/data/sample.rs`

This file needs the most important **local correction**.

### Problem in current code — direct support, high confidence

Current `sample.rs` stores `delta_q_target` and `delta_q_mask` in **parallel flat buffers** with a single `any_delta_q` flag:

* `SAMPLE L0185-L0187`
* `SAMPLE L0314-L0323`
* `SAMPLE L0399-L0414`

That means a per-sample mismatch such as:

```rust
sample.delta_q_target = Some([...]);
sample.delta_q_mask   = None;
```

can become:

* batch-level `Some(delta_q_target)`
* batch-level `Some(delta_q_mask)` (because some other row may have a mask, or because `any_delta_q` is true)
* zero mask row for the broken sample

`validate_optional_target_pairs` in `rl.rs` only checks **batch-level** `(Some,Some)` vs `(None,None)`, not per-row mismatch. `masked_action_mse` then returns zero on a zero-mask row even if the target row contains garbage.
Support: `RL L0060-L0067`, `LOSSSEM L0321-L0330`.
Confidence high because this follows directly from the current write/collate logic.

### Worked failure example

If a row has:

```text
target = [1.2, -0.7, ...]
mask   = [0.0, 0.0, ...]
```

then

[
L_{\Delta q} =
\frac{\sum \frac12 (\hat{\Delta q}-\Delta q)^2 \cdot m}{\max(1,\sum m)} = 0
]

even though the target row is invalid.

### Required fix

Replace the parallel-flat-buffer path with the already-existing canonical option-pair path used by ExIt.

#### Change `CollateBuffers`

```rust
struct CollateBuffers {
    // ...
    delta_q_samples: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}
```

#### Change `write_sample`

```rust
self.delta_q_samples[index] = match (delta_q_target, delta_q_mask) {
    (Some(target), Some(mask)) => Some((target.to_vec(), mask.to_vec())),
    (None, None) => None,
    _ => panic!("delta_q target/mask mismatch at sample {index}"),
};
```

#### Change `into_batch`

```rust
let (delta_q_target, delta_q_mask) =
    collate_delta_q_targets::<B>(&self.delta_q_samples, device);

MjaiBatch {
    // ...
    delta_q_target,
    delta_q_mask,
    // ...
}
```

### Why this is the chosen local fix

* `collate_delta_q_targets` already exists (`DQOBJ L0335-L0357`)
* it matches the ExIt absent-row semantics
* it makes per-sample mismatch impossible to hide
* it is narrower than inventing a new validator layer

### Confidence

High.
Validation path:

* add `delta_q_sample_pair_mismatch_panics`
* add `delta_q_collation_roundtrips_absent_rows_as_zero_mask_rows`
* add `delta_q_target_without_mask_cannot_reach_MjaiBatch`

---

## 7.4 `crates/hydra-train/src/training/delta_q_validation.rs`

Keep the existing report and threshold types. Do **not** invent new global thresholds in this tranche.

### Reuse unchanged

```rust
DeltaQValidationReport
DeltaQValidationThresholds::default()
evaluate_report(...)
```

Current default thresholds already give the conservative structural gate:

* `sample_size >= 1000`
* `emission_rate >= 0.01`
* `mean_coverage >= 0.70`
* `mean_supported_actions >= 3.0`

Support: `DQV L0174-L0259`.

### Add replay/offline runner

```rust
pub fn run_replay_delta_q_validation<B: Backend>(
    replays: &[(String, Vec<MjaiEvent>)],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> DeltaQValidationReport
```

This runner should:

* reconstruct replay states,
* call the same shared producer,
* aggregate the same report fields,
* evaluate with the same `DeltaQValidationThresholds`.

### Add replay roundtrip validator

This is the **new provenance-specific validator**:

```rust
pub struct ReplayDeltaQRoundtripReport {
    pub total_records: u64,
    pub matched_records: u64,
    pub missing_records: u64,
    pub contract_rejections: u64,
}
```

Pass condition:

```text
matched_records == total_records
missing_records == 0
contract_rejections == 0
```

Run it on a self-generated sidecar over the exact same replay corpus and source identity/version.

### Why this survives

The existing `DeltaQValidationReport` is structural and search-lane-oriented.
The new roundtrip report is the smallest replay/offline addition that actually tests provenance closure, not just teacher sparsity.

### Confidence

* Reusing existing thresholds: **medium**.
  Why not high: current artifacts define them for the RL lane, not explicitly for replay/offline BC.
  Why still acceptable: they are already conservative and lane-local.
* Adding exact roundtrip match: **high**.
  Why: it directly tests the sidecar contract you are introducing.

### Stronger local alternative

Compare replay sidecar labels against a deeper learner-only AFBS reference on sampled replay states:

* masked sign agreement,
* top-discard agreement,
* masked MAE.

This is useful, but it is **archive-supported only**, not current authority doctrine. Keep it out of the v1 blocking gate.

---

## 8. Tests that must land with the patch

## 8.1 `training/replay_delta_q.rs`

Add:

* `delta_q_sidecar_lookup_requires_matching_contract`
* `delta_q_sidecar_lookup_rejects_non_discard_mask`
* `delta_q_sidecar_lookup_rejects_aka_mask`
* `delta_q_sidecar_lookup_rejects_nonzero_target_outside_mask`
* `delta_q_sidecar_lookup_rejects_nonfinite_target`
* `replay_delta_q_records_are_tagged_search_derived`
* `loader_with_delta_q_sidecar_populates_delta_q_fields`
* `self_generated_delta_q_sidecar_roundtrips_exactly`

## 8.2 `data/sample.rs`

Add:

* `delta_q_sample_pair_mismatch_panics`
* `delta_q_collation_uses_zero_rows_for_absent_samples`
* `delta_q_collation_preserves_present_row_values`

## 8.3 `training/delta_q_validation.rs`

Add:

* `replay_delta_q_validation_thresholds_pass_on_passing_report`
* `replay_delta_q_roundtrip_report_requires_exact_match`

## 8.4 Keep these existing tests unchanged

* replay plain-loader absence test
* train-bin rejection tests

This preserves the deliberate guardrail until activation hook lands.

---

## 9. Staged activation closure

## Stage A — implement now

Land all of this now:

1. `replay_delta_q.rs`
2. loader join path
3. `sample.rs` pair-safety fix
4. replay/offline `delta_q` validation runner
5. replay roundtrip validator
6. tests above

Keep `train.rs` rejection unchanged.

### Confidence

High.
Why: no new semantics, no new heads, no AFBS broadening, all changes are direct extensions of already-closed ExIt and RL `delta_q` lanes.

---

## Stage B — shadow validation gate

Require all of the following before any BC/train enablement discussion:

1. `DeltaQValidationResult.passed == true`
2. `ReplayDeltaQRoundtripReport` exact match pass
3. at least one self-generated replay corpus shows nonzero joined `delta_q_target` coverage
4. plain replay loader still keeps `delta_q` absent without sidecar
5. no per-sample target/mask mismatch can reach `MjaiBatch`

### Confidence

High.
Why: this is exactly the missing provenance/validation tranche the authority docs call for; it is not broader than the current lane.

---

## Stage C — still blocked

Do **not** unblock `train.rs` yet.

### Exact blocker

A narrow **BC/train advanced-head activation hook** is still missing.

The hook must define:

1. how BC/train records per-batch target presence,
2. how BC/train applies `approved_loss_config`,
3. how BC/train handles warmup-time trunk detachment for `warmup_heads()`.

The current controller docs explicitly require the caller/orchestrator to do that, and public `bc.rs` still does not. Support: `GATEA L0031-L0042`, `GATEB L0531-L0572`, plus public `bc.rs` check. ([GitHub][2])

### Smallest decisive missing contract artifact

Implement or write down this exact narrow spec:

```rust
pub struct BcAdvancedHeadActivationHook {
    pub controller: HeadActivationController,
}

pub fn bc_apply_head_gating<B: Backend>(
    controller: &mut HeadActivationController,
    base_loss: &HydraLossConfig,
    targets: &HydraTargets<B>,
) -> HydraLossConfig {
    let presence = extract_target_presence(targets);
    controller.record_batch(&presence);
    controller.approved_loss_config(base_loss)
}
```

and add the missing model-side warmup API:

```rust
pub fn forward_with_warmup_detach<B: Backend>(
    &self,
    obs: Tensor<B, 3>,
    loss_cfg: &HydraLossConfig,
    warmup_heads: &[AdvancedHead],
) -> HydraOutput<B>;
```

`DeltaQ` warmup behavior in that API must be:

* **no new head**
* **no trunk change**
* feed the existing `delta_q` head from the current shared representation,
* but if `AdvancedHead::DeltaQ` is in `warmup_heads`, detach that shared representation before the `delta_q` head so the head trains while trunk stays frozen.

Without that hook, flipping `advanced_loss.delta_q` on in BC/train would bypass Hydra’s own sparse-head activation doctrine.

### Confidence

High.
Why:

* the controller contract already says the caller must do this,
* the roadmap already flags orchestration/warmup integration as open,
* public `bc.rs` shows no controller hookup.

---

## 10. Keep / narrow / remove summary

| Piece                                       | Verdict              | Why                                       |
| ------------------------------------------- | -------------------- | ----------------------------------------- |
| `ReplayDecisionKey`                         | **Keep**             | already canonical replay identity         |
| `source_net_hash` / `source_version`        | **Keep**             | already canonical producer provenance     |
| `legal_mask_digest`                         | **Keep**             | already canonical join guard              |
| action discriminator                        | **Keep, but narrow** | join-only, not teacher-defining           |
| `search-derived` provenance tag             | **Keep**             | matches ExIt and lane semantics           |
| shared root-search producer                 | **Keep**             | already emits ExIt + `delta_q` together   |
| sidecar JSONL / index pattern               | **Keep**             | proven by ExIt                            |
| `kl_to_base`                                | **Remove**           | not meaningful for signed regression      |
| visit-distribution teacher                  | **Remove**           | wrong teacher object                      |
| q-softmax teacher                           | **Remove**           | wrong teacher object                      |
| runtime `[34]` bridge plane as label source | **Remove**           | feature lane, not teacher lane            |
| replay-derived loader builder               | **Remove**           | violates provenance-explicit rule         |
| current `sample.rs` flat `delta_q` buffers  | **Remove**           | weaker than existing option-pair collator |

---

## 11. Confidence ledger

| Conclusion                                                      | Support type       |  Confidence | Why                                                                 | Falsify by                                               |
| --------------------------------------------------------------- | ------------------ | ----------: | ------------------------------------------------------------------- | -------------------------------------------------------- |
| v1 teacher is masked discard-only `Q(child)-Q(root)`            | direct             |        high | builder, live producer, invariants, tests all agree                 | break `DQOBJ`/arena tests                                |
| replay/offline must be search-derived sidecar, not loader-built | direct             |        high | README/REC/loader absence/test/train rejection all align            | produce authority text blessing replay-built `delta_q`   |
| ExIt sidecar is the right provenance template                   | inference + direct | medium-high | already working in same repo, same replay identity problem          | implement and fail exact roundtrip                       |
| `sample.rs` must switch to option-pair collation                | direct reasoning   |        high | current code can hide per-row mismatch behind zero mask             | add mismatch test and observe current silent pass        |
| full BC/train activation is still blocked on activation hook    | direct + web check |        high | controller docs require caller integration; public `bc.rs` lacks it | show existing BC controller hookup and warmup detach API |

---

## 12. Final decision

Implement **Stage A** now:

* new `replay_delta_q.rs`,
* loader join,
* `sample.rs` pair-safety fix,
* replay validation runner,
* exact roundtrip validator,
* tests.

That closes the **replay/offline provenance question** without semantic guessing.

Do **not** claim the lane is fully activation-closed yet.
It is not.
The remaining blocker is narrow and explicit:

> **Missing artifact:** BC/train advanced-head activation hook spec (`HeadActivationController` integration + warmup detach API).

Once that generic hook exists, `delta_q` no longer needs new semantics, new provenance, or new search doctrine. It only needs the already-built controller to be honored in BC/train.

[1]: https://github.com/NikkeTryHard/hydra "https://github.com/NikkeTryHard/hydra"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/crates/hydra-train/src/training/bc.rs "raw.githubusercontent.com"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
