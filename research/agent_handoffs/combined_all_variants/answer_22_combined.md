Read rest file. Then compress whole body exact-shape.<combined_run_record run_id="answer_22" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 22 + returned agent answer. Prompt side exactly recoverable from authoritative generator config at `research/agent_handoffs/agent22_exit_eval_prompt_config.json` plus surviving rendered prompt file at `research/agent_handoffs/agent22_exit_live_afbs_evaluator_blueprint.md`. Answer side only partly recoverable because original uncommitted `answer_22.md` got overwritten during normalization; surviving original prefix preserved, missing continuation explicitly marked grounded reconstruction.</notes>
<layout>exact_prompt_source_plus_partially_reconstructed_answer_record</layout>
</metadata>

<warning_note status="historical_partially_stale" added="2026-03-15">
<![CDATA[
Warning: core evaluator/teacher-shape reasoning here still useful, but old live-lane status wording partly stale.

Known stale area:
- older default-off framing for live ExIt producer. Current code/docs already flipped live self-play ExIt lane to default-on, while keeping narrower visit-based teacher semantics + validation history.

Use this file for:
- learner-only, root-only AFBS evaluator reasoning
- visit-based teacher semantics
- rejection of `root_exit_policy()` / q-softmax as teacher object

Do not use this file as current truth for:
- present-tense default-off / enablement status

Validate against live authority chain before reuse:
README.md -> research/design/HYDRA_FINAL.md -> research/design/HYDRA_RECONCILIATION.md -> docs/GAME_ENGINE.md
]]>
</warning_note>

<prompt_section>
<prompt_text status="exact_source_reference" source_path="agent22_exit_live_afbs_evaluator_blueprint.md">
<![CDATA[# Hydra prompt — agent 22 ExIt live AFBS evaluator blueprint

<role>
Produce impl-ready blueprint.
No memo.
Answer itself must be blueprint.
</role>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- separate direct artifact support from own inference
- use search/browse to find original paper, then inspect full PDF with skill; use abstracts or summaries only for discovery, not final evidence base
- use bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not dump logic; every important mechanism, threshold, or rec must be inferable from evidence or explicit in blueprint so it can be validated and reproduced
- do not stop early; keep looping through discovery, thinking, testing, and validation until info saturated or blocked, and not before at least 20+ such loops (more if possible)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now. Not guaranteed fully correct. Treat as evidence to inspect + critique, not truth to inherit. High chance some are incomplete, misleading, stale, or semantically wrong, so validate all.
</artifact_note>

<direction>
Aim for strongest exact blueprint for still-unresolved evaluator/value source inside Hydra's live AFBS -> ExIt self-play producer loop.

We do NOT want broad ExIt memo. Carrier seam already mostly closed. Unresolved question narrower: if Hydra runs AFBS at decision time during self-play to emit `exit_target` / `exit_mask`, what evaluator or value source should AFBS use so resulting labels are semantically defensible in current repo reality?

Need detailed answer making clear:
- what candidate evaluator/value sources exist or almost exist in current Hydra
- which are semantically valid vs fake / misleading / too weak for training labels
- whether live decision-time ExIt generation should use current AFBS shell semantics, model value head, root-child visits only, bridge/runtime signal, tiny rollout/value evaluator, or some other narrow source
- what must stay narrow / deferred / rejected
- what exact producer algorithm should run from decision-time state to `TrajectoryExitLabel` with minimal guesswork
- what acceptance tests or experiments are minimum needed if evidence still underdetermined

Use artifacts below to derive conclusions.
</direction>

<scope_note>
Keep narrow.
Do not redesign Hydra broadly.
Do not widen into belief, Hand-EV, delta_q, oracle-path architecture, or broad AFBS identity changes.
Focus only evaluator/value source needed for real live AFBS ExIt producer during self-play.
</scope_note>

<hard_guardrails>
1. Do not assume current self-play ExIt hook is enough because carrier seam exists.
2. Do not assume `root_exit_policy()` is valid training teacher; if it survives, justify with artifact support + tests.
3. Do not bless fake evaluator because it makes loop easy.
4. Do not silently convert plumbing question into broad AFBS redesign.
5. Do not widen into offline relabel infra unless online path truly blocked by current repo surfaces.
6. If candidate evaluator only acceptable for smoke-testing or closure tests, say that explicitly and do not present it as real training doctrine.
7. If evidence insufficient to pick one evaluator confidently, say underdetermined and give smallest decisive experiment matrix instead of faking certainty.
</hard_guardrails>

<output_requirements>
Answer must separate:
- direct artifact support
- external source support
- inference
- proposal
- blocked / missing surface

And must end with one buildable surviving blueprint, not only menu of options.
</output_requirements>

<artifacts_manifest>

[Normalization note: original inline artifact bodies intentionally omitted here. Hydra combined artifacts now preserve compact manifest entries only, because referenced files already exist locally and duplicating full contents in this record is no longer archive format.]

## Artifact 01 — Reference narrow prompt shape
Artifact id: `reference-narrow-shape`
Source label: REF
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/reference_prompt_example_001_narrow_focused.md:9-51`
Why it matters: Canon narrow artifact-first blueprint family this prompt should resemble.

## Artifact 02 — Canonical shell doctrine
Artifact id: `prompt-guide-shell`
Source label: GUIDE
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:110-155`
Why it matters: Use this to keep prompt shell aligned with Hydra's artifact-first doctrine.

## Artifact 03 — Prompt generator workflow and authoring rules
Artifact id: `prompt-guide-generator`
Source label: GUIDE
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:528-724`
Why it matters: Documents intended JSON-driven workflow, artifact density requirements, and final checklist.

[Prompt source note: full authoritative prompt can be regenerated exactly from `research/agent_handoffs/agent22_exit_eval_prompt_config.json` and matches surviving rendered output at `research/agent_handoffs/agent22_exit_live_afbs_evaluator_blueprint.md`. This combined record keeps prompt prefix + exact source refs rather than duplicating full 4,957-line rendered prompt body inline.]
</artifacts_manifest>]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="partially_reconstructed" source_path="answer_22.md">
<![CDATA[# Live AFBS ExIt evaluator blueprint

## Decision

Use **visit-based ExIt labels** only.
Label object stays:

* `exit_target[a] = normalized root child visits over masked discard actions`
* `exit_mask[a] = 1` only for legal discard children with enough support

Only surviving narrow evaluator/value source for generating those visits in current Hydra:

* **current public model value head, used only as leaf scorer inside learner-only, root-only AFBS search**

But evaluator **not yet blessed as always-on doctrine**. It stays **default-off until it passes small validation matrix**. If it fails validation, live producer must emit `None`, not fall back to `root_exit_policy()`, q-softmax labels, bridge heuristics, or oracle rollouts.

---

## Direct artifact support

### 1) Current repo ExIt teacher semantics are child-visit semantics, not q-softmax semantics

Canonical target builder is `make_exit_target_from_child_visits()` and canonical AFBS bridge is `build_exit_from_afbs_tree()`. They:

* require compatible discard-only state
* read **root child visit counts**
* mask only legal discard actions with `visits >= 2`
* normalize by covered visit mass
* require coverage `>= 0.60`
* require average root visits per legal discard `>= 8.0`
* require KL safety check against `base_pi`

That is current teacher meaning. `q_value()` not used in target builder.
Direct artifact support: `hydra-train/src/training/exit.rs:157-261`.

### 2) `root_exit_policy()` is a different object

`AfbsTree::root_exit_policy()` computes **softmax over child q-values**. It is not current ExIt target builder.
Direct artifact support: `hydra-core/src/afbs.rs:265-305`.

`PonderResult::from_tree()` stores that q-softmax as `exit_policy`, with `trust_level = LearnerOnly`.
Direct artifact support: `hydra-core/src/afbs.rs:419-438`.

So in current repo reality:

* `build_exit_from_afbs_tree()` = training-teacher path
* `root_exit_policy()` = ponder/cache/reporting path

Not same object.

### 3) AFBS currently has no built-in evaluator

`run_search_iterations()` requires external `eval_fn: Fn(NodeIdx) -> f32`. Shell does selection + backprop only; it does not define value source.
Direct artifact support: `hydra-core/src/afbs.rs:246-263`.

So unresolved question is real: live producer must supply evaluator.

### 4) Self-play carrier seam now exists

Self-play loop now has decision-time hook that can attach `TrajectoryExitLabel` to each step, trajectory validator enforces strict ExIt invariants, batch collation forwards labels into `RlBatch`, and RL already consumes `exit_target`/`exit_mask` when present.
Direct artifact support: `hydra-train/src/selfplay.rs:264-273,365-366,412-464,467-516`; `hydra-core/src/arena.rs:7-29`; `hydra-train/src/training/rl.rs:148-156`.

So unresolved piece no longer carrier plumbing; it is search evaluator/value source.

### 5) Current model surface exposes value output, with no new head required

Reconciliation explicitly says no new heads in this tranche, and model already exposes needed surfaces. Integration test also confirms forward output includes `value`.
Direct artifact support: `research/design/HYDRA_RECONCILIATION.md:470-479`; `hydra-train/src/model.rs:9-23,253-286`.

So model-value evaluator is **available or nearly available**.

### 6) Current self-play value supervision looks weak as search evaluator

In current self-play batch construction, `value_target` filled with `step.reward`, and `step.reward` produced by splitting each player’s final score evenly across that player’s steps.
Direct artifact support: `hydra-train/src/selfplay.rs:388-390,546-556`.

That does **not** prove value head useless, but does mean current value path is **not already evidenced as strong search evaluator**.

### 7) `expand_node()` is incompatible with broad ExIt coverage

AFBS root expansion truncates to `TOP_K = 5`. ExIt target construction requires `coverage >= 0.60`.
Direct artifact support: `hydra-core/src/afbs.rs:188-219`; `hydra-train/src/training/exit.rs:8-10,210-213`.

If legal discard count is `L`, then using `expand_node()` makes max possible coverage `5 / L`.

So:

* feasible only if `5 / L >= 0.60`
* i.e. only if `L <= 8`

For `L = 9`, max coverage is `0.556`; for `L = 14`, max coverage is `0.357`.
Therefore **live ExIt producer must not use `expand_node()` for root label generation**. It must seed all legal discard children itself.

### 8) Archive doctrine already warned against broadening and against weak targets

Reconciliation says: do not broaden AFBS, expose only minimum outputs needed for ExIt, and do not fabricate weak labels. Archive guidance further says ExIt should activate only after trust-gated AFBS label building with explicit support masks + coverage logging.
Direct artifact support: `research/design/HYDRA_RECONCILIATION.md:423-429,490-497`; `research/agent_handoffs/combined_all_variants/answer_15_combined.md:479-483`.

---

## External source support

ExIt’s original paper is clear that apprentice should imitate **root tree policy** `n(s,a)/n(s)`, not only chosen move, because target is cost-sensitive and better aligned with future search guidance. Same paper also uses value network to score expanded leaves and back those estimates through tree; when exact expert value too costly, it approximates expert value with apprentice value (`https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf`).

AlphaZero uses joint network `(p, v)` to guide MCTS, where `v` estimates expected outcome from position. Search returns policy `π` from **root visit counts**, and training matches policy head to those search probabilities while matching value head to game outcome. Strongest canonical support for “value network as evaluator, visits as teacher” (`https://arxiv.org/abs/1712.01815`).

Grill et al. show AlphaZero’s **empirical visit distribution** tracks regularized policy-improvement objective, while exact reversed-KL solution is **not** generic q-softmax. They also show using exact solution can outperform raw visits when simulation budgets are low. Strong support against treating naked q-softmax like `root_exit_policy()` as doctrinal teacher, especially in low-budget search (`https://proceedings.mlr.press/v119/grill20a.html`).

---

## Inference

### 1) `root_exit_policy()` should be rejected as training teacher

This follows from both repo semantics + literature.

Repo’s own canonical teacher path is visit-based, not q-softmax-based.
Literature says ExIt/AlphaZero train against search-improved policies tied to **visits**, and Grill 2020 says exact regularized improvement object is not generic q-softmax.

Concrete repo-native mismatch already appears in current test tree.

From `EXIT`’s AFBS test tree:

* child visits = `[10, 8, 6]`
* child q-values = `[9/10, 4/8, 0.6/6] = [0.9, 0.5, 0.1]`

Then:

* canonical visit target = `[10, 8, 6] / 24 = [0.417, 0.333, 0.250]`
* `root_exit_policy(tau=1)` = `softmax([0.9, 0.5, 0.1]) = [0.472, 0.316, 0.212]`

L1 gap about `0.110`.

So even on current repo-style numbers, `root_exit_policy()` is **not** same object as current teacher.

### 2) “root-child visits only” is correct as target semantics, but not as evaluator semantics

Visits are right **output object**. They are **not** value source.

If AFBS has no meaningful evaluator, visit counts collapse into prior/exploration bookkeeping. Replaying repo PUCT rule with priors `[0.5, 0.3, 0.2]`, constant zero values, and 24 simulations yields visits `[12, 7, 5]`, prior-shaped exploration. Useful for smoke-testing plumbing, but not semantically defensible as training teacher.

So:

* **visits as teacher**: yes
* **visits with no evaluator**: no

### 3) The only current public-compatible evaluator that stays narrow is the model value head

Candidates that fail:

* **Exact hidden-state rollout / oracle evaluator**: invalid for student labels in this imperfect-information setting; label would depend on privileged hidden state.
* **Bridge/runtime signals (`risk_score`, `ΔQ` plane, Hand-EV summaries)**: feature-side/runtime summaries, not already-defined student teacher.
* **`root_exit_policy()` / q-softmax**: wrong teacher object.
* **Prior-only AFBS visits**: smoke-test only.

Candidate that survives:

* **current public model value head**, used only to drive search visits

Why it survives:

* public-compatible
* already exists
* matches ExIt/AlphaZero pattern “value head evaluates leaves, visits become teacher”
* needs no new head or broad AFBS redesign

Why still not blessed:

* current repo artifacts do not yet show this head is calibrated strongly enough for search

### 4) The current value head may be too weak or too small-scale to move PUCT visits

Main caution.

If value head is trained against current self-play `value_target` path, its target scale plausibly small. Example:

[
\text{step.reward} = \frac{\text{final_score}}{100000 \cdot \text{player_step_count}}
]

Typical value might look like:

* `25000 / 100000 / 40 = 0.00625`

But AFBS exploration bonus is:

[
U(a) = 2.5 \cdot P(a) \cdot \frac{\sqrt{N}}{1+n_a}
]

At `P(a)=0.1`, `N=80`, `n_a=8`, that is about:

[
U \approx 2.5 \cdot 0.1 \cdot \frac{\sqrt{80}}{9} \approx 0.25
]

So if value head lives near `0.005-0.02`, raw q may be order-of-magnitude too small to materially change visits.

That creates clear rule:

* **do not invent new `value_scale` knob in this tranche**
* first test whether raw value-head AFBS beats alternatives
* if not, keep producer off

### 5) The live producer should stay root-only in this tranche

Current AFBS shell is missing fully evidenced deeper transition/evaluation stack in artifacts. Narrow, repo-compatible interpretation:

* **root-only AFBS bandit search over legal discard children**
* one public leaf value per child
* no deeper opponent-tree expansion
* no belief-stack redesign
* no q-target activation

That keeps producer narrow and makes teacher meaning explicit.

---

## Proposal

## Provenance

`exit_target` / `exit_mask` produced by this blueprint are:

* **search-derived**
* **learner-only**
* **visit-based**
* **public-evaluator-driven**
* **discard-only**
* **hard-state-gated**

They are **not** replay-derived, **not** bridge-derived, and **not** oracle-derived.

---

## Candidate verdicts

### Reject now

`root_exit_policy()`
Reason: wrong teacher object; q-softmax path; no perspective contract; no visit-match test; contradicts current canonical builder.

`exit_policy_from_q()` / `make_exit_target()` mainline use
Reason: q-target path is not current doctrine for live ExIt. Keep for tests or future delta-q work only.

Bridge/runtime summary signals as evaluator
Reason: features/heuristics, not closed teacher semantics.

Exact hidden-state rollouts from live simulator
Reason: privileged/oracle teacher for public student.

### Smoke-test only

AFBS visits with constant/zero evaluator
Reason: proves carrier + batching only.

### Defer

Public CT-SMC rollout / belief evaluator
Reason: semantically interesting, but widens into belief machinery and violates requested narrow scope for first closure.

### Survive

Current public model value head as AFBS leaf evaluator
Reason: only narrow public-compatible evaluator already on surface.
Status: **implementable, but default-off until accepted by experiment matrix below**.

---

## Surviving producer algorithm

### Semantics

For decision-time state `s` and legal discard action set `A_disc(s)`:

1. compute masked base prior `base_pi`
2. run learner-only root-only AFBS over all legal discard children
3. evaluate each child once with current public value head
4. let AFBS turn those child values + priors into **root visit counts**
5. call existing visit-based target builder
6. emit `TrajectoryExitLabel` only if all existing gates pass

No q-softmax distillation. No oracle rollout. No bridge heuristic target.

---

## Exact algorithm

### Step 0: entry gate

Run producer only inside current self-play decision-time hook, on pre-transition state that already feeds `StepRecord`.
Direct artifact support: `hydra-train/src/selfplay.rs:467-516`.

### Step 1: state compatibility gate

Build:

```rust
let legal_f32: [f32; HYDRA_ACTION_SPACE] =
    step.legal_mask.map(|b| if b { 1.0 } else { 0.0 });
```

Reject unless:

```rust
compatible_discard_state(&legal_f32)
```

and at least 2 legal discard actions.

Use same discard-only legality rules as current `exit.rs`.
Direct artifact support: `hydra-train/src/training/exit.rs:141-155,172-179`.

### Step 2: base policy and hard-state gate

Compute base prior from raw logits, **not** from `step.pi_old`:

```rust
let base_pi = softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0);
```

[Reconstruction note: original uncommitted `answer_22.md` content after this point was overwritten locally during normalization. Continuation below is grounded reconstruction based on recoverable original prefix + current repo sources.]

Use current ExIt hard-state predicate as narrow tranche gate, not broader search router:

```rust
let exit_cfg = ExitConfig::default_phase3();
if !is_hard_state(&base_pi, exit_cfg.hard_state_threshold) {
    return None;
}
```

Why this gate survives:

* already exists in ExIt surface (`hydra-train/src/training/exit.rs:12-24,34-47,78-85`)
* keeps producer aligned with current “hard-state-gated” doctrine (`research/design/HYDRA_RECONCILIATION.md:490-497`)
* avoids silently widening this into broad always-on AFBS

### Step 3: build a learner-only root-only AFBS tree

Do **not** call `expand_node()` for label generation. That helper truncates to `TOP_K = 5`, which can make `build_exit_from_afbs_tree()` fail coverage even on otherwise valid states (`hydra-core/src/afbs.rs:188-219`; `hydra-train/src/training/exit.rs:8-10,210-213`).

Instead, construct root + all legal discard children explicitly from masked base prior:

```rust
let mut tree = AfbsTree::new();
let root = tree.add_node(info_state_hash, 1.0, false);

for action in 0..=DISCARD_END {
    let idx = action as usize;
    if !step.legal_mask[idx] {
        continue;
    }
    let prior = base_pi[idx];
    let child = tree.add_node(predicted_child_hash(info_state_hash, action), prior, false);
    tree.nodes[root as usize].children.push((action, child));
}
```

This is smallest honest change because AFBS already exposes `add_node`, `predicted_child_hash`, child lists, `run_search_iterations`, and root-visit extraction (`hydra-core/src/afbs.rs:153-165,246-314`). Only thing avoided is lossy top-k expansion helper.

### Step 4: evaluator contract

Evaluator must be:

* public-compatible
* learner-only
* leaf-scoring only
* root-search narrow

Surviving evaluator is current model value head:

```rust
fn eval_child_with_value_head(model: &HydraModel<B>, child_obs: Tensor<B, 3>) -> f32 {
    let out = model.forward(child_obs);
    out.value_scalar().unwrap_or(0.0)
}
```

Grounding:

* `HydraOutput` already includes `value` and exposes `value_scalar()` (`hydra-train/src/model.rs:9-24,43-49`)
* `HydraModel::forward()` already computes `value` from pooled public representation (`hydra-train/src/model.rs:253-271`)
* value head already exists as standard head, not new architectural addition (`hydra-train/src/heads.rs:206-210,314-327`)

Blocked surface:

* repo material in scope does **not** show ready-made helper that takes discard child node and directly emits child observation tensor for value evaluation

So buildable doctrine is: **producer stays default-off until that child-observation seam is implemented and validation matrix below passes**. Still narrower + more honest than blessing q-softmax or zero-value visits as real teacher.

### Step 5: root-only AFBS iterations

Once child observations exist, run only root-bandit search:

```rust
tree.run_search_iterations(root, num_iters, &|leaf_idx| {
    let child_action = /* recover action for this root child */;
    let child_obs = build_child_observation(state, obs, child_action)?;
    eval_child_with_value_head(model, child_obs)
});
```

Why root-only:

* `run_search_iterations()` is only selection/backprop shell and agnostic to evaluator semantics (`hydra-core/src/afbs.rs:246-263`)
* nothing in scoped artifacts proves deeper transition/evaluator stack already semantically closed for live ExIt
* user explicitly wanted narrow unresolved evaluator question, not broader AFBS redesign

### Step 6: canonical teacher build

After root visits exist, use existing canonical teacher path without inventing new label object:

```rust
let built = build_exit_from_afbs_tree(
    &tree,
    root,
    &base_pi,
    &legal_f32,
    exit_cfg.min_visits,
    exit_cfg.safety_valve_max_kl,
);
let (target, mask) = built?;
let label = TrajectoryExitLabel::from_slices(&target, &mask)?;
return Some(label);
```

Grounding: `build_exit_from_afbs_tree()` is canonical producer entrypoint; `TrajectoryExitLabel` is self-play carrier object; RL consumes `exit_target`/`exit_mask` when present (`hydra-train/src/training/exit.rs:229-291`; `hydra-core/src/arena.rs:7-29`; `hydra-train/src/selfplay.rs:365-366,412-464`; `hydra-train/src/training/rl.rs:148-156`).

### Step 7: failure semantics

If any gate fails, emit `None`.

That includes:

* not compatible discard-only state
* not hard state
* child-observation/evaluator seam unavailable
* root visits below threshold
* coverage below threshold
* KL valve failure
* non-finite value outputs

This is doctrine, not style. Reconciliation explicitly says leave clearly unavailable targets absent rather than fabricate weak labels (`research/design/HYDRA_RECONCILIATION.md:424-429`).

---

## Blocked / missing surface

Current repo still lacks one crucial evidential seam for always-on activation:

1. clean, explicit helper that converts candidate legal discard child into public observation tensor value head should score

Until that exists, live producer is **implementable in blueprint form** but not yet fully evidenced as completed path. That does **not** invalidate blueprint; it means blueprint’s default state remains off.

---

## Minimum experiment matrix

Value-head evaluator survives only if it beats fake alternatives on smallest decisive matrix.

### E0 — Carrier smoke test

Goal: prove self-play seam attaches labels and RL receives them.

Pass if:

* `run_self_play_game_with_exit_labels()` emits at least some non-`None` `TrajectoryExitLabel`s on forced hard-state fixture (`hydra-train/src/selfplay.rs:264-273,467-516`)
* `trajectories_to_rl_batch()` collates non-`None` `exit_target` and `exit_mask` (`hydra-train/src/selfplay.rs:365-366,412-464`)
* RL step consumes them without NaN/Inf (`hydra-train/src/training/rl.rs:146-156`)

### E1 — Teacher semantics equivalence

Goal: prove producer still emits canonical teacher object.

Pass if, on fixed fixtures:

* emitted labels match `build_exit_from_afbs_tree()` exactly
* masked target sums to 1 when emitted
* support + coverage behavior match current tests in `exit.rs` (`hydra-train/src/training/exit.rs:382-605`)

### E2 — Evaluator necessity test

Compare three candidates on same hard discard-only states:

1. zero/constant evaluator
2. q-softmax teacher shortcut via `root_exit_policy()`
3. value-head evaluator

Pass condition for value-head candidate:

* it changes root visits materially relative to zero-evaluator search on meaningful fraction of hard states
* it preserves canonical visit-built teacher semantics because teacher is still `build_exit_from_afbs_tree()`
* it avoids doctrinal mismatch of q-softmax-as-teacher

If it does **not** change visits materially, keep producer off instead of inventing scale knob.

### E3 — Short training sanity

Run small RL/BC sanity slice with producer enabled only on hard discard-only states.

Require:

* finite loss
* nonzero exit auxiliary contribution
* no collapse in label coverage to near-zero
* no evidence labels are only priors in disguise

Last check can be approximated by measuring whether emitted exit targets differ materially from masked base priors on accepted states.

---

## Final surviving blueprint

Use this and only this as buildable narrow blueprint:

1. trigger only from existing self-play decision-time hook
2. require current discard-only compatibility
3. require current ExIt hard-state gate
4. build learner-only root-only AFBS tree over **all** legal discard children, not `expand_node()`
5. score child leaves with **current public model value head**
6. derive labels only through `build_exit_from_afbs_tree()`
7. wrap them in `TrajectoryExitLabel`
8. emit `None` on any failed gate
9. keep producer **default-off** until experiment matrix passes

What stays rejected:

* `root_exit_policy()` as teacher
* q-softmax teacher shortcuts
* bridge heuristics as evaluator
* oracle/hidden-state rollouts
* broad AFBS redesign

What stays deferred:

* public CT-SMC / belief evaluator variants
* any value scaling knob
* deeper than root-only live AFBS for this producer

That is narrowest semantically defensible live ExIt producer in current Hydra reality: **visit-based labels from learner-only root-only AFBS, driven by existing public value head, default-off until validated**.]]>
</answer_text>
  # Live AFBS ExIt evaluator blueprint

## Decision

Use **visit-based ExIt labels** only.
Label object stays:

* `exit_target[a] = normalized root child visits over masked discard actions`
* `exit_mask[a] = 1` only for legal discard children with enough support

Only surviving narrow evaluator/value source for generating those visits in current Hydra:

* **current public model value head, used only as leaf scorer inside learner-only, root-only AFBS search**

But evaluator **not yet blessed as always-on doctrine**. It stays **default-off until it passes small validation matrix**. If it fails validation, live producer must emit `None`, not fall back to `root_exit_policy()`, q-softmax labels, bridge heuristics, or oracle rollouts.

---

## Direct artifact support

### 1) Current repo ExIt teacher semantics are child-visit semantics, not q-softmax semantics

Canonical target builder is `make_exit_target_from_child_visits()` and canonical AFBS bridge is `build_exit_from_afbs_tree()`. They:

* require compatible discard-only state
* read **root child visit counts**
* mask only legal discard actions with `visits >= 2`
* normalize by covered visit mass
* require coverage `>= 0.60`
* require average root visits per legal discard `>= 8.0`
* require KL safety check against `base_pi`

That is current teacher meaning. `q_value()` not used in target builder.
Direct artifact support: `EXIT L0157-L0261`.

### 2) `root_exit_policy()` is a different object

`AfbsTree::root_exit_policy()` computes **softmax over child q-values**. It is not current ExIt target builder.
Direct artifact support: `AFBS L0265-L0305`.

`PonderResult::from_tree()` stores that q-softmax as `exit_policy`, with `trust_level = LearnerOnly`.
Direct artifact support: `AFBS L0419-L0438`.

So in current repo reality:

* `build_exit_from_afbs_tree()` = training-teacher path
* `root_exit_policy()` = ponder/cache/reporting path

Not same object.

### 3) AFBS currently has no built-in evaluator

`run_search_iterations()` requires external `eval_fn: Fn(NodeIdx) -> f32`. Shell does selection + backprop only; it does not define value source.
Direct artifact support: `AFBS L0246-L0263`.

So unresolved question is real: live producer must supply evaluator.

### 4) Self-play carrier seam now exists

Self-play loop now has decision-time hook that can attach `TrajectoryExitLabel` to each step, trajectory validator enforces strict ExIt invariants, batch collation forwards labels into `RlBatch`, and RL already consumes `exit_target`/`exit_mask` when present.
Direct artifact support: `SELF L0264-L0273`, `SELF L0501-L0516`, `SELF L0411-L0464`, `ARENA L0483-L0530`, `RL L0148-L0156`.

So unresolved piece no longer carrier plumbing; it is search evaluator/value source.

### 5) The current model surface exposes a value output, with no new head required

Reconciliation explicitly says no new heads in this tranche, and model already exposes needed surfaces. Integration test also confirms that forward output includes `value`.
Direct artifact support: `RECON L0470-L0477`, `ITEST L0031-L0039`.

So model-value evaluator is **available or nearly available**.

### 6) Current self-play value supervision looks weak as search evaluator

In current self-play batch construction, `value_target` filled with `step.reward`, and `step.reward` produced by splitting each player’s final score evenly across that player’s steps.
Direct artifact support: `SELF L0388-L0390`, `SELF L0546-L0556`.

That does **not** prove value head useless, but does mean current value path is **not already evidenced as strong search evaluator**.

### 7) `expand_node()` is incompatible with broad ExIt coverage

AFBS root expansion truncates to `TOP_K = 5`. ExIt target construction requires `coverage >= 0.60`.
Direct artifact support: `AFBS L0015-L0016`, `AFBS L0188-L0219`, `EXIT L0210-L0213`.

If legal discard count is `L`, then using `expand_node()` makes max possible coverage `5 / L`.

So:

* feasible only if `5 / L >= 0.60`
* i.e. only if `L <= 8`

For `L = 9`, max coverage is `0.556`; for `L = 14`, max coverage is `0.357`.
Therefore **live ExIt producer must not use `expand_node()` for root label generation**. It must seed all legal discard children itself.

### 8) Archive doctrine already warned against broadening and against weak targets

Reconciliation says: do not broaden AFBS, expose only minimum outputs needed for ExIt, and do not fabricate weak labels. Archive guidance further says ExIt should activate only after trust-gated AFBS label building with explicit support masks + coverage logging.
Direct artifact support: `RECON L0490-L0497`, `RECON L0424-L0429`, `A15 L0479-L0483`.

---

## External source support

ExIt’s original paper is clear that apprentice should imitate **root tree policy** `n(s,a)/n(s)`, not only chosen move, because target is cost-sensitive and better aligned with future search guidance. Same paper also uses value network to score expanded leaves and back those estimates through tree; when exact expert value too costly, it approximates expert value with apprentice value. ([NeurIPS Papers][1])

AlphaZero uses joint network `(p, v)` to guide MCTS, where `v` estimates expected outcome from position. Search returns policy `π` from **root visit counts**, and training matches policy head to those search probabilities while matching value head to game outcome. Strongest canonical support for “value network as evaluator, visits as teacher.” ([arXiv][2])

Grill et al. show that AlphaZero’s **empirical visit distribution** tracks regularized policy-improvement objective, while exact reversed-KL solution is **not** generic q-softmax. They also show using exact solution can outperform raw visits when simulation budgets are low. Strong support against treating naked q-softmax like `root_exit_policy()` as doctrinal teacher, especially in low-budget search. ([Proceedings of Machine Learning Research][3])

---

## Inference

### 1) `root_exit_policy()` should be rejected as the training teacher

This follows from both repo semantics + literature.

Repo’s own canonical teacher path is visit-based, not q-softmax-based.
Literature says ExIt/AlphaZero train against search-improved policies tied to **visits**, and Grill 2020 says exact regularized improvement object is not generic q-softmax.

Concrete repo-native mismatch already appears in current test tree.

From `EXIT`’s AFBS test tree:

* child visits = `[10, 8, 6]`
* child q-values = `[9/10, 4/8, 0.6/6] = [0.9, 0.5, 0.1]`

Then:

* canonical visit target = `[10, 8, 6] / 24 = [0.417, 0.333, 0.250]`
* `root_exit_policy(tau=1)` = `softmax([0.9, 0.5, 0.1]) = [0.472, 0.316, 0.212]`

L1 gap about `0.110`.

So even on current repo-style numbers, `root_exit_policy()` is **not** same object as current teacher.

### 2) “root-child visits only” is correct as target semantics, but not as evaluator semantics

Visits are right **output object**. They are **not** value source.

If AFBS has no meaningful evaluator, visit counts collapse into prior/exploration bookkeeping. Replaying repo PUCT rule with priors `[0.5, 0.3, 0.2]`, constant zero values, and 24 simulations yields visits `[12, 7, 5]`, prior-shaped exploration. Useful for smoke-testing plumbing, but not semantically defensible as training teacher.

So:

* **visits as teacher**: yes
* **visits with no evaluator**: no

### 3) The only current public-compatible evaluator that stays narrow is the model value head

Candidates that fail:

* **Exact hidden-state rollout / oracle evaluator**: invalid for student labels in this imperfect-information setting; label would depend on privileged hidden state.
* **Bridge/runtime signals (`risk_score`, `ΔQ` plane, Hand-EV summaries)**: feature-side/runtime summaries, not already-defined student teacher.
* **`root_exit_policy()` / q-softmax**: wrong teacher object.
* **Prior-only AFBS visits**: smoke-test only.

Candidate that survives:

* **current public model value head**, used only to drive search visits

Why it survives:

* public-compatible
* already exists
* matches ExIt/AlphaZero pattern “value head evaluates leaves, visits become teacher”
* needs no new head or broad AFBS redesign

Why still not blessed:

* current repo artifacts do not yet show this head is calibrated strongly enough for search

### 4) The current value head may be too weak or too small-scale to move PUCT visits

Main caution.

If value head is trained against current self-play `value_target` path, its target scale plausibly small. Example:

[
\text{step.reward} = \frac{\text{final_score}}{100000 \cdot \text{player_step_count}}
]

Typical value might look like:

* `25000 / 100000 / 40 = 0.00625`

But AFBS exploration bonus is:

[
U(a) = 2.5 \cdot P(a) \cdot \frac{\sqrt{N}}{1+n_a}
]

At `P(a)=0.1`, `N=80`, `n_a=8`, that is about:

[
U \approx 2.5 \cdot 0.1 \cdot \frac{\sqrt{80}}{9} \approx 0.25
]

So if value head lives near `0.005-0.02`, raw q may be order-of-magnitude too small to materially change visits.

That creates clear rule:

* **do not invent new `value_scale` knob in this tranche**
* first test whether raw value-head AFBS beats alternatives
* if not, keep producer off

### 5) The live producer should stay root-only in this tranche

Current AFBS shell is missing fully evidenced deeper transition/evaluation stack in artifacts. Narrow, repo-compatible interpretation:

* **root-only AFBS bandit search over legal discard children**
* one public leaf value per child
* no deeper opponent-tree expansion
* no belief-stack redesign
* no q-target activation

That keeps producer narrow and makes teacher meaning explicit.

---

## Proposal

## Provenance

`exit_target` / `exit_mask` produced by this blueprint are:

* **search-derived**
* **learner-only**
* **visit-based**
* **public-evaluator-driven**
* **discard-only**
* **hard-state-gated**

They are **not** replay-derived, **not** bridge-derived, and **not** oracle-derived.

---

## Candidate verdicts

### Reject now

`root_exit_policy()`
Reason: wrong teacher object; q-softmax path; no perspective contract; no visit-match test; contradicts current canonical builder.

`exit_policy_from_q()` / `make_exit_target()` mainline use
Reason: q-target path is not current doctrine for live ExIt. Keep for tests or future delta-q work only.

Bridge/runtime summary signals as evaluator
Reason: features/heuristics, not closed teacher semantics.

Exact hidden-state rollouts from live simulator
Reason: privileged/oracle teacher for public student.

### Smoke-test only

AFBS visits with constant/zero evaluator
Reason: proves carrier + batching only.

### Defer

Public CT-SMC rollout / belief evaluator
Reason: semantically interesting, but widens into belief machinery and violates requested narrow scope for first closure.

### Survive

Current public model value head as AFBS leaf evaluator
Reason: only narrow public-compatible evaluator already on surface.
Status: **implementable, but default-off until accepted by experiment matrix below**.

---

## Surviving producer algorithm

### Semantics

For decision-time state `s` and legal discard action set `A_disc(s)`:

1. compute masked base prior `base_pi`
2. run learner-only root-only AFBS over all legal discard children
3. evaluate each child once with current public value head
4. let AFBS turn those child values + priors into **root visit counts**
5. call existing visit-based target builder
6. emit `TrajectoryExitLabel` only if all existing gates pass

No q-softmax distillation. No oracle rollout. No bridge heuristic target.

---

## Exact algorithm

### Step 0: entry gate

Run producer only inside current self-play decision-time hook, on pre-transition state that already feeds `StepRecord`.
Direct artifact support: `SELF L0467-L0516`.

### Step 1: state compatibility gate

Build:

```rust
let legal_f32: [f32; HYDRA_ACTION_SPACE] =
    step.legal_mask.map(|b| if b { 1.0 } else { 0.0 });
```

Reject unless:

```rust
compatible_discard_state(&legal_f32)
```

and at least 2 legal discard actions.

Use same discard-only legality rules as current `exit.rs`.
Direct artifact support: `EXIT L0141-L0155`, `EXIT L0172-L0179`.

### Step 2: base policy and hard-state gate

Compute base prior from raw logits, **not** from `step.pi_old`:

```rust
let base_pi = softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0);
```

Reason:

* `step.pi_old` is action-sampling policy after self-play temperature
* search prior and KL safety valve should compare against raw network prior, not exploration temperature

Then extract only legal discard probabilities and apply current hard-state helper:

```rust
let legal_discards: Vec<usize> = (0..=DISCARD_END as usize)
    .filter(|&a| step.legal_mask[a])
    .collect();

let hard_slice: Vec<f32> = legal_discards.iter().map(|&a| base_pi[a]).collect();

if !is_hard_state(&hard_slice, cfg.hard_state_threshold) {
    return None;
}
```

Use existing threshold default `0.1`.
Direct artifact support: `EXIT L0012-L0024`, `EXIT L0078-L0085`, `ARENA L0537-L0565`.

### Step 3: dynamic visit budget from existing gates

Do not guess new search budget. Use current gate itself:

[
N_{\text{budget}} =
\max\left(
\texttt{cfg.min_visits},
\left\lceil 8.0 \cdot |A_{\text{disc}}(s)| \right\rceil
\right)
]

Rust:

```rust
let budget = cfg.min_visits.max(
    (MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD
        * legal_discards.len() as f32)
        .ceil() as u32
);
```

This is minimal budget that can satisfy existing average-visits gate without inventing new multiplier.
Direct artifact support: `EXIT L0008-L0010`, `EXIT L0181-L0185`.

### Step 4: seed the AFBS root with **all** legal discard children

Do **not** call `expand_node()`.

Instead manually seed root with every legal discard action so coverage is even possible:

```rust
fn seed_root_children_all_legal(
    tree: &mut AfbsTree,
    root: NodeIdx,
    root_hash: u64,
    priors: &[(u8, f32)],
) {
    let z = priors.iter().map(|(_, p)| *p).sum::<f32>().max(1e-8);
    for &(action, prior) in priors {
        let child = tree.add_node(
            predicted_child_hash(root_hash, action),
            prior / z,
            false,
        );
        tree.nodes[root as usize].children.push((action, child));
    }
}
```

This is mandatory. If `expand_node()` is used, many discard states can never meet coverage because of `TOP_K = 5`.

### Step 5: child public observation adapter

For each legal discard action `a`, create **public child observation** for same root player.

Required contract:

```rust
trait ExitSearchAdapter {
    fn root_hash(&self, state: &GameState, player: u8, step: &StepRecord) -> u64;

    fn child_public_obs_after_discard(
        &mut self,
        state: &GameState,
        obs: &Observation,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]>;
}
```

Required semantics of `child_public_obs_after_discard`:

* clone current simulator state
* map hydra discard `action` back to riichienv `Action` using same context path as `NnActionSelector` (`infer_action_phase`, `hand_from_observation`, `hydra_to_riichienv`)
* apply only root player’s discard into next public state boundary needed to form root-player observation
* do **not** roll through hidden-state-contingent opponent actions for teacher construction
* re-encode with same public bridge path and `SafetyInfo`

This keeps evaluator public-compatible.

If adapter cannot produce such child observation, return `None` and emit no label.

### Step 6: evaluate each child once with the model value head

Use joint policy/value inference closure:

```rust
type PolicyValueFn =
    dyn FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32);
```

Only scalar value is used in this tranche:

```rust
let mut child_value_by_node = HashMap::<NodeIdx, f32>::new();

for &a in &legal_discards {
    let child_obs = adapter.child_public_obs_after_discard(
        state, obs, step.player_id, a as u8, safety
    )?;
    let (_child_logits, v_child) = model_pv(&child_obs);
    child_value_by_node.insert(child_node, v_child);
}
```

This is evaluator choice.
Child policy output ignored for now.

### Step 7: run root-only AFBS

Run AFBS search iterations with cached child values:

```rust
tree.run_search_iterations(root, budget, &|child_idx| {
    child_value_by_node[&child_idx]
});
```

Because this is root-only, repeated visits to child reapply cached public value. Intentional in this tranche.

Selection is exactly current repo PUCT:

[
\text{score}(a)=Q_a + 2.5 \cdot P_a \cdot \frac{\sqrt{N_{\text{root}}}}{1+n_a}
]

Direct artifact support: `AFBS L0153-L0186`, `AFBS L0246-L0263`.

### Step 8: build the label using the existing canonical helper

Do not reimplement target math.

Call:

```rust
let (target, mask) = build_exit_from_afbs_tree(
    &tree,
    root,
    &base_pi,
    &legal_f32,
    budget,
    cfg.safety_valve_max_kl,
)?;
```

Then:

```rust
TrajectoryExitLabel::from_slices(&target, &mask)
```

This preserves current semantics + existing validation gates.
Direct artifact support: `EXIT L0238-L0261`, `ARENA L0012-L0029`.

### Step 9: if any gate fails, emit `None`

No fallback teacher.

Specifically, do **not** fall back to:

* `root_exit_policy()`
* `exit_policy_from_q()`
* constant-eval AFBS visits
* bridge risk/ΔQ heuristics
* exact hidden-state rollouts

---

## Code skeleton

```rust
pub fn try_live_exit_label<M, A>(
    state: &GameState,
    obs: &Observation,
    step: &StepRecord,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
{
    let legal_f32 = step.legal_mask.map(|b| if b { 1.0 } else { 0.0 });
    if !compatible_discard_state(&legal_f32) {
        return None;
    }

    let legal_discards: Vec<usize> = (0..=DISCARD_END as usize)
        .filter(|&a| step.legal_mask[a])
        .collect();
    if legal_discards.len() < 2 {
        return None;
    }

    let base_pi = softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0);

    let hard_slice: Vec<f32> = legal_discards.iter().map(|&a| base_pi[a]).collect();
    if !is_hard_state(&hard_slice, cfg.hard_state_threshold) {
        return None;
    }

    let budget = cfg.min_visits.max(
        (MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD
            * legal_discards.len() as f32)
            .ceil() as u32
    );

    let root_hash = adapter.root_hash(state, step.player_id, step);
    let mut tree = AfbsTree::new();
    let root = tree.add_node(root_hash, 1.0, false);

    let priors: Vec<(u8, f32)> = legal_discards
        .iter()
        .map(|&a| (a as u8, base_pi[a]))
        .collect();
    seed_root_children_all_legal(&mut tree, root, root_hash, &priors);

    let mut value_by_child = std::collections::HashMap::<NodeIdx, f32>::new();
    for &(action, child) in &tree.nodes[root as usize].children {
        let child_obs = adapter.child_public_obs_after_discard(
            state, obs, step.player_id, action, safety
        )?;
        let (_logits, v) = model_pv(&child_obs);
        value_by_child.insert(child, v);
    }

    tree.run_search_iterations(root, budget, &|child| value_by_child[&child]);

    let (target, mask) = build_exit_from_afbs_tree(
        &tree,
        root,
        &base_pi,
        &legal_f32,
        budget,
        cfg.safety_valve_max_kl,
    )?;

    TrajectoryExitLabel::from_slices(&target, &mask)
}
```

---

## File-level implementation plan

### `hydra-train/src/selfplay.rs`

Change exit hook signature to pass current player’s `SafetyInfo` snapshot:

```rust
E: FnMut(
    &GameState,
    &Observation,
    &StepRecord,
    &SafetyInfo,
    u32
) -> Option<TrajectoryExitLabel>
```

At call site:

```rust
let exit_label = exit_label_fn(
    env.state,
    &obs,
    &step_record,
    env.selector.safety(step_record.player_id),
    turn,
);
```

Reason: child observation encoding needs same public safety context as actor path.

### New module: `hydra-train/src/training/live_exit.rs`

Add:

* `ExitSearchAdapter`
* `seed_root_children_all_legal`
* `try_live_exit_label`
* small helpers:

  * `legal_discard_actions(step)`
  * `base_pi_from_logits(step)`
  * `budget_from_legal_count(cfg, n_legal)`

### No required doctrinal changes in `training/exit.rs`

Reuse existing:

* `compatible_discard_state`
* `is_hard_state`
* `build_exit_from_afbs_tree`

### No required doctrinal changes in `afbs.rs`

Do **not** change AFBS shell semantics in this tranche.

### Do not use current `PonderResult::from_tree()` for label caching

That constructor stores `root_exit_policy()` q-softmax, which is not teacher.
If caching needed later, add **separate learner-target cache object** that stores visit-based `target/mask`, same-net hash/version, and generation. Not in this tranche.

---

## Blocked / missing surface

### 1) Public child-state adapter

Artifacts do not prove ready-made helper that takes:

* current `GameState`
* root player id
* candidate discard action

and returns next **public root-player observation** after only that discard.

This is main missing surface. Buildable fix is tiny adapter/helper, not AFBS redesign.

### 2) Consistent child encoding context

Child encoding needs root player’s `SafetyInfo`.
Current hook does not pass it. That is why self-play signature change above is required.

### 3) Value-head strength is not yet evidenced

Current artifacts do not prove model value head is calibrated enough, or large enough in scale, to move PUCT visits meaningfully. That is why producer must be validation-gated.

### 4) Do not use `PonderCache` for labels yet

Current cache/result surfaces encode q-softmax `exit_policy`, not visit teacher semantics.

---

## Minimum acceptance tests

## A. Unit / integration tests that must be added immediately

### A1. `root_exit_policy` stays rejected

Construct existing test tree and assert visit target and `root_exit_policy(tau=1)` are not numerically identical.

This is not for math correctness; it prevents future semantic drift.

### A2. All-legal root seeding is mandatory

Test two producers on state with 9+ legal discard actions:

* producer using `expand_node()`
* producer using `seed_root_children_all_legal()`

Expected:

* `expand_node()` path returns `None`
* all-legal seeding path can pass coverage if visits support it

### A3. Producer uses visit builder, not q builder

Stub deterministic child values, run producer, and assert:

* output equals `build_exit_from_afbs_tree(...)`
* output does not route through `root_exit_policy()` or `exit_policy_from_q()`

### A4. Structural reject tests

Add tests that producer returns `None` on:

* incompatible state
* fewer than 2 legal discards
* not hard state
* KL reject
* insufficient coverage
* missing child observation

---

## B. Small decisive evaluator-selection matrix

This is minimum experiment matrix if evaluator remains underdetermined.

### Dataset

Collect held-out set of compatible discard states from self-play **after**:

* `compatible_discard_state`
* legal discard count `>= 2`
* hard-state gate

### Candidate teachers

Evaluate these four candidates on same states:

1. **proposed**: value-head AFBS visits
2. **baseline**: raw `base_pi`
3. **smoke control**: prior-only AFBS visits (`eval_fn = 0`)
4. **rejected control**: `root_exit_policy()` from same searched tree

### Evaluation target

For **evaluation only**, estimate child-action quality with privileged continuation sampling from simulator:

[
G(s,a) = \frac{1}{K}\sum_{k=1}^K \text{final_normalized_score}_{\text{root}}^{(k)}(s,a)
]

Important:

* privileged rollout is **only** for evaluator selection
* it must **not** be used as training label source

### Metrics

Report:

[
\mathbb{E}*s[G(s,\arg\max_a \pi*{\text{teacher}}(a|s))]
]

and pairwise action-order accuracy against `G(s,a)`.

### Enable criterion

Enable live producer only if proposed value-head AFBS beats:

* raw `base_pi`
* prior-only AFBS visits
* `root_exit_policy()`

on expected privileged continuation score, with bootstrap confidence interval excluding zero improvement over `base_pi`.

No arbitrary score threshold needed. Choice is relative.

---

## C. Short training ablation

After B passes, run short same-budget ablation:

* no exit labels
* prior-only AFBS exit labels
* value-head AFBS exit labels

Accept always-on training activation only if:

* value-head AFBS beats both alternatives on held-out self-play metrics
* prior-only labels do **not** produce same gain

This is minimum test that distinguishes “real teacher” from “plumbing noise”.

---

## What stays narrow / deferred / rejected

### Stay narrow now

* discard-only states
* hard-state gate
* root-only AFBS
* visit-based labels only
* public model value head as evaluator
* manual all-legal root seeding
* existing coverage / child-visit / KL gates
* learner-only provenance

### Deferred

* deeper AFBS tree expansion
* CT-SMC public rollout evaluator
* delta-q target activation
* cache reuse for learner targets
* any change to value-head training semantics
* any explicit regularized-policy operator (`\bar{\pi}`) replacing visits

### Rejected

* `root_exit_policy()` as teacher
* q-softmax labels
* bridge/runtime heuristics as teacher
* exact hidden-state/oracle rollouts as teacher
* prior-only AFBS labels as real doctrine

---

## Buildable surviving blueprint

Implement **opt-in learner-only live ExIt producer** that:

1. runs only on compatible discard-only hard states,
2. computes `base_pi` from raw masked policy logits at temperature `1.0`,
3. seeds AFBS root with **all** legal discard actions,
4. scores each child once with **current public model value head** on public child observation,
5. runs **root-only** AFBS for
[
   N_{\text{budget}}=\max(64,\lceil 8 \cdot |A_{\text{disc}}|\rceil)
]
6. converts root child visits into `TrajectoryExitLabel` via existing canonical `build_exit_from_afbs_tree()`,
7. emits `None` on any gate failure,
8. stays **default-off for real training** until it beats `base_pi`, prior-only AFBS visits, and `root_exit_policy()` in evaluator-selection matrix.

If validation fails, do **not** rescue lane with `root_exit_policy()`, q-softmax, bridge heuristics, value rescaling knobs, or oracle rollouts. Keep producer off and leave ExIt labels absent.

[1]: https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf "https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf"
[2]: https://arxiv.org/pdf/1712.01815.pdf "https://arxiv.org/pdf/1712.01815.pdf"
[3]: https://proceedings.mlr.press/v119/grill20a/grill20a.pdf "https://proceedings.mlr.press/v119/grill20a/grill20a.pdf"

</answer_section>
</combined_run_record>