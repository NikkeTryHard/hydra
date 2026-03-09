<combined_run_record run_id="answer_22" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 22 and its returned agent answer. The prompt side is exactly recoverable from the authoritative generator config at `research/agent_handoffs/agent22_exit_eval_prompt_config.json` plus the surviving rendered prompt file at `research/agent_handoffs/agent22_exit_live_afbs_evaluator_blueprint.md`. The answer side is only partially recoverable because the original uncommitted `answer_22.md` was overwritten during normalization; the surviving original prefix is preserved and the missing continuation remains explicitly marked as a grounded reconstruction.</notes>
    <layout>exact_prompt_source_plus_partially_reconstructed_answer_record</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="exact_source_reference" source_path="agent22_exit_live_afbs_evaluator_blueprint.md">
  <![CDATA[# Hydra prompt — agent 22 ExIt live AFBS evaluator blueprint

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
</artifact_note>

<direction>
Work toward the strongest exact blueprint for the still-unresolved evaluator/value source inside Hydra's live AFBS -> ExIt self-play producer loop.

We do NOT want a broad ExIt memo. The carrier seam is already largely closed. The unresolved question is narrower: if Hydra runs AFBS at decision time during self-play in order to emit `exit_target` / `exit_mask`, what evaluator or value source should AFBS use so that the resulting labels are semantically defensible in current repo reality?

We want a detailed answer that makes clear:
- what candidate evaluator/value sources are actually available or nearly available in current Hydra
- which of those are semantically valid versus fake / misleading / too weak for training labels
- whether live decision-time ExIt generation should use current AFBS shell semantics, model value head, root-child visits only, a bridge/runtime signal, a tiny rollout/value evaluator, or some other narrow source
- what must stay narrow / deferred / rejected
- what exact producer algorithm should be used from decision-time state to `TrajectoryExitLabel` with minimal guesswork
- what acceptance tests or experiments are the minimum needed if the evidence is still underdetermined

Use the artifacts below to derive your conclusions.
</direction>

<scope_note>
Keep this narrow.
Do not redesign Hydra broadly.
Do not widen into belief, Hand-EV, delta_q, oracle-path architecture, or broad AFBS identity changes.
Focus only on the evaluator/value source needed for a real live AFBS ExIt producer during self-play.
</scope_note>

<hard_guardrails>
1. Do not assume the current self-play ExIt hook is enough just because the carrier seam now exists.
2. Do not assume `root_exit_policy()` is a valid training teacher; if it survives, justify it with artifact support and tests.
3. Do not bless a fake evaluator just because it makes the loop easy to implement.
4. Do not silently convert a plumbing question into a broad AFBS redesign.
5. Do not widen into offline relabel infrastructure unless the online path is truly blocked by current repo surfaces.
6. If a candidate evaluator is only acceptable for smoke-testing or closure tests, say that explicitly and do not present it as real training doctrine.
7. If the evidence is insufficient to pick one evaluator with confidence, say underdetermined and specify the smallest decisive experiment matrix instead of faking certainty.
</hard_guardrails>

<output_requirements>
The answer must separate:
- direct artifact support
- external source support
- inference
- proposal
- blocked / missing surface

And it must end with one buildable surviving blueprint, not just a menu of options.
</output_requirements>

<artifacts>

## Artifact 01 — Reference narrow prompt shape
Artifact id: `reference-narrow-shape`
Source label: REF
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/reference_prompt_example_001_narrow_focused.md:9-51`
Why it matters: Canonical example of the narrow artifact-first blueprint family this prompt should resemble.

```text
[REF L0009]   <![CDATA[# Reference example — narrow focused artifact-first blueprint
[REF L0010] 
[REF L0011] <role>
[REF L0012] Produce an implementation-ready blueprint.
[REF L0013] Do not give a memo.
[REF L0014] Your answer itself must be the blueprint.
[REF L0015] </role>
[REF L0016] 
[REF L0017] <direction>
[REF L0018] Work toward the strongest exact blueprint for a single narrow implementation or validation lane.
[REF L0019] 
[REF L0020] We want a detailed answer that makes clear:
[REF L0021] - what the current quantities or mechanisms really mean
[REF L0022] - what is semantically broken or misleading
[REF L0023] - what the clean repaired meanings should be
[REF L0024] - what should stay exact, what should stay approximate, and what should be dropped or demoted
[REF L0025] - how to implement or validate the surviving path with minimal guesswork
[REF L0026] 
[REF L0027] Use the artifacts below to derive your conclusions.
[REF L0028] </direction>
[REF L0029] 
[REF L0030] <style>
[REF L0031] - no high-level survey
[REF L0032] - no vague answer
[REF L0033] - include reasoning
[REF L0034] - include formulas when needed
[REF L0035] - include code-like detail when helpful (python or rust)
[REF L0036] - include worked examples when helpful
[REF L0037] - include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
[REF L0038] - distinguish direct artifact support from your own inference
[REF L0039] - use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
[REF L0040] - use the bash tool to run Python for calculations, math checks, and validation when rigor matters
[REF L0041] - do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
[REF L0042] - do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
[REF L0043] </style>
[REF L0044] 
[REF L0045] <artifact_note>
[REF L0046] The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
[REF L0047] </artifact_note>
[REF L0048] 
[REF L0049] <artifacts>
[REF L0050] [Insert dense task-specific code/doc/test/formula artifacts here.]
[REF L0051] </artifacts>]]>
```

## Artifact 02 — Canonical shell doctrine
Artifact id: `prompt-guide-shell`
Source label: GUIDE
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:110-155`
Why it matters: Use this to keep the prompt shell aligned with Hydra's artifact-first doctrine.

```text
[GUIDE L0110] ## 2. Default top-of-prompt shell
[GUIDE L0111] 
[GUIDE L0112] Use this shell for most serious Hydra prompts.
[GUIDE L0113] 
[GUIDE L0114] ```xml
[GUIDE L0115] <role>
[GUIDE L0116] Produce an implementation-ready blueprint.
[GUIDE L0117] Do not give a memo.
[GUIDE L0118] Your answer itself must be the blueprint.
[GUIDE L0119] </role>
[GUIDE L0120] 
[GUIDE L0121] <direction>
[GUIDE L0122] Work toward the strongest exact blueprint for [TASK].
[GUIDE L0123] 
[GUIDE L0124] We want a detailed answer that makes clear:
[GUIDE L0125] - [decision point 1]
[GUIDE L0126] - [decision point 2]
[GUIDE L0127] - [decision point 3]
[GUIDE L0128] - [what must stay narrow / deferred / rejected]
[GUIDE L0129] - [how to implement or validate the surviving path with minimal guesswork]
[GUIDE L0130] 
[GUIDE L0131] Use the artifacts below to derive your conclusions.
[GUIDE L0132] </direction>
[GUIDE L0133] 
[GUIDE L0134] <style>
[GUIDE L0135] - no high-level survey
[GUIDE L0136] - no vague answer
[GUIDE L0137] - include reasoning
[GUIDE L0138] - include formulas when needed
[GUIDE L0139] - include code-like detail when helpful (python or rust)
[GUIDE L0140] - include worked examples when helpful
[GUIDE L0141] - include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
[GUIDE L0142] - distinguish direct artifact support from your own inference
[GUIDE L0143] - use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
[GUIDE L0144] - use the bash tool to run Python for calculations, math checks, and validation when rigor matters
[GUIDE L0145] - do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
[GUIDE L0146] </style>
[GUIDE L0147] 
[GUIDE L0148] <artifact_note>
[GUIDE L0149] The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
[GUIDE L0150] </artifact_note>
[GUIDE L0151] 
[GUIDE L0152] <artifacts>
[GUIDE L0153] ...
[GUIDE L0154] </artifacts>
[GUIDE L0155] ```
```

## Artifact 03 — Prompt generator workflow and authoring rules
Artifact id: `prompt-guide-generator`
Source label: GUIDE
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:528-724`
Why it matters: Documents the intended JSON-driven workflow, artifact density requirements, and final checklist.

```text
[GUIDE L0528] ## 12. Prompt generator tool
[GUIDE L0529] 
[GUIDE L0530] For repeated prompt authoring, use `scripts/generate_prompt.py` instead of hand-assembling every long prompt from scratch.
[GUIDE L0531] 
[GUIDE L0532] The tool is not a prompt framework.
[GUIDE L0533] It is a small JSON-driven utility for generating Hydra-style artifact-first prompts faster and more consistently.
[GUIDE L0534] 
[GUIDE L0535] Use it when:
[GUIDE L0536] 
[GUIDE L0537] - you want multiple prompt variants from one shared artifact packet
[GUIDE L0538] - you want reusable shell blocks like `role`, `direction`, `style`, and `artifact_note`
[GUIDE L0539] - you want line-ranged code/doc excerpts without manual copy-paste
[GUIDE L0540] - you want per-artifact labels and explanations so the artifact body has useful context
[GUIDE L0541] - you want to regenerate prompts quickly after changing the artifact set
[GUIDE L0542] 
[GUIDE L0543] Do not use it as an excuse to stop thinking about prompt quality.
[GUIDE L0544] The generator speeds up assembly.
[GUIDE L0545] It does not decide what artifacts belong in the prompt.
```

<artifacts>
[Prompt source note: the full authoritative prompt can be regenerated exactly from `research/agent_handoffs/agent22_exit_eval_prompt_config.json` and matches the surviving rendered output at `research/agent_handoffs/agent22_exit_live_afbs_evaluator_blueprint.md`. This combined record keeps a prompt prefix plus exact source references rather than duplicating the full 4,957-line rendered prompt body inline.]
</artifacts>]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="partially_reconstructed" source_path="answer_22.md">
  <![CDATA[# Live AFBS ExIt evaluator blueprint

## Decision

Use **visit-based ExIt labels** and **only** visit-based ExIt labels.
The label object should remain:

* `exit_target[a] = normalized root child visits over masked discard actions`
* `exit_mask[a] = 1` only for legal discard children with enough support

The **only surviving narrow evaluator/value source** for generating those visits in current Hydra is:

* **the current public model value head, used only as a leaf scorer inside a learner-only, root-only AFBS search**

But that evaluator is **not yet blessed as always-on doctrine**. It is **default-off until it passes a small validation matrix**. If it fails validation, the live producer must emit `None`, not fall back to `root_exit_policy()`, q-softmax labels, bridge heuristics, or oracle rollouts.

---

## Direct artifact support

### 1) Current repo ExIt teacher semantics are child-visit semantics, not q-softmax semantics

The canonical target builder is `make_exit_target_from_child_visits()` and the canonical AFBS bridge is `build_exit_from_afbs_tree()`. They:

* require a compatible discard-only state
* read **root child visit counts**
* mask only legal discard actions with `visits >= 2`
* normalize by covered visit mass
* require coverage `>= 0.60`
* require average root visits per legal discard `>= 8.0`
* require KL safety check against `base_pi`

That is the current teacher meaning. `q_value()` is not used in the target builder.
Direct artifact support: `hydra-train/src/training/exit.rs:157-261`.

### 2) `root_exit_policy()` is a different object

`AfbsTree::root_exit_policy()` computes a **softmax over child q-values**. It is not the current ExIt target builder.
Direct artifact support: `hydra-core/src/afbs.rs:265-305`.

`PonderResult::from_tree()` stores that q-softmax as `exit_policy`, with `trust_level = LearnerOnly`.
Direct artifact support: `hydra-core/src/afbs.rs:419-438`.

So in current repo reality:

* `build_exit_from_afbs_tree()` = training-teacher path
* `root_exit_policy()` = ponder/cache/reporting path

Those are not the same thing.

### 3) AFBS currently has no built-in evaluator

`run_search_iterations()` requires an external `eval_fn: Fn(NodeIdx) -> f32`. The shell does selection and backprop only; it does not define what the value source is.
Direct artifact support: `hydra-core/src/afbs.rs:246-263`.

So the unresolved question is real: a live producer must supply an evaluator.

### 4) Self-play carrier seam now exists

The self-play loop now has a decision-time hook that can attach `TrajectoryExitLabel` to each step, the trajectory validator enforces strict ExIt invariants, batch collation forwards labels into `RlBatch`, and RL already consumes `exit_target`/`exit_mask` when present.
Direct artifact support: `hydra-train/src/selfplay.rs:264-273,365-366,412-464,467-516`; `hydra-core/src/arena.rs:7-29`; `hydra-train/src/training/rl.rs:148-156`.

So the unresolved piece is not carrier plumbing anymore; it is the search evaluator/value source.

### 5) The current model surface exposes a value output, with no new head required

Reconciliation explicitly says no new heads in this tranche, and the model already exposes the needed surfaces. The integration test also confirms that forward output includes `value`.
Direct artifact support: `research/design/HYDRA_RECONCILIATION.md:470-479`; `hydra-train/src/model.rs:9-23,253-286`.

So a model-value evaluator is **available or nearly available**.

### 6) Current self-play value supervision looks weak as a search evaluator

In current self-play batch construction, `value_target` is filled with `step.reward`, and `step.reward` is produced by splitting each player’s final score evenly across that player’s steps.
Direct artifact support: `hydra-train/src/selfplay.rs:388-390,546-556`.

That does **not** directly prove the value head is useless, but it does mean the current value path is **not already evidenced as a strong search evaluator**.

### 7) `expand_node()` is incompatible with broad ExIt coverage

AFBS root expansion truncates to `TOP_K = 5`. ExIt target construction requires `coverage >= 0.60`.
Direct artifact support: `hydra-core/src/afbs.rs:188-219`; `hydra-train/src/training/exit.rs:8-10,210-213`.

If legal discard count is `L`, then using `expand_node()` makes maximum possible coverage `5 / L`.

So:

* feasible only if `5 / L >= 0.60`
* i.e. only if `L <= 8`

For `L = 9`, max coverage is `0.556`; for `L = 14`, max coverage is `0.357`.
Therefore **a live ExIt producer must not use `expand_node()` for root label generation**. It must seed all legal discard children itself.

### 8) Archive doctrine already warned against broadening and against weak targets

Reconciliation says: do not broaden AFBS, expose only the minimum outputs needed for ExIt, and do not fabricate weak labels. Archive guidance further says ExIt should activate only after trust-gated AFBS label building with explicit support masks and coverage logging.
Direct artifact support: `research/design/HYDRA_RECONCILIATION.md:423-429,490-497`; `research/agent_handoffs/combined_all_variants/answer_15_combined.md:479-483`.

---

## External source support

ExIt’s original paper is clear that the apprentice should imitate the **root tree policy** `n(s,a)/n(s)`, not just the chosen move, because that target is cost-sensitive and better aligned with future search guidance. The same paper also uses a value network to score expanded leaves and backs those estimates up through the tree; when exact expert value is too costly, it approximates expert value with the apprentice value (`https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf`).

AlphaZero uses a joint network `(p, v)` to guide MCTS, where `v` estimates expected outcome from the position. The search returns a policy `π` from **root visit counts**, and training matches the policy head to those search probabilities while matching the value head to game outcome. That is the strongest canonical support for “value network as evaluator, visits as teacher” (`https://arxiv.org/abs/1712.01815`).

Grill et al. show that AlphaZero’s **empirical visit distribution** tracks a regularized policy-improvement objective, while the exact reversed-KL solution is **not** a generic q-softmax. They also show that using the exact solution can outperform raw visits when simulation budgets are low. That is strong support against treating a naked q-softmax like `root_exit_policy()` as the doctrinal teacher, especially in low-budget search (`https://proceedings.mlr.press/v119/grill20a.html`).

---

## Inference

### 1) `root_exit_policy()` should be rejected as the training teacher

This follows from both repo semantics and the literature.

The repo’s own canonical teacher path is visit-based, not q-softmax-based.
The literature says ExIt/AlphaZero train against search-improved policies tied to **visits**, and Grill 2020 says the exact regularized improvement object is not generic q-softmax.

A concrete repo-native mismatch already appears in the current test tree.

From `EXIT`’s AFBS test tree:

* child visits = `[10, 8, 6]`
* child q-values = `[9/10, 4/8, 0.6/6] = [0.9, 0.5, 0.1]`

Then:

* canonical visit target = `[10, 8, 6] / 24 = [0.417, 0.333, 0.250]`
* `root_exit_policy(tau=1)` = `softmax([0.9, 0.5, 0.1]) = [0.472, 0.316, 0.212]`

The L1 gap is about `0.110`.

So even on current repo-style numbers, `root_exit_policy()` is **not** the same object as the current teacher.

### 2) “root-child visits only” is correct as target semantics, but not as evaluator semantics

Visits are the right **output object**. They are **not** a value source.

If AFBS has no meaningful evaluator, visit counts collapse into prior/exploration bookkeeping. Replaying the repo PUCT rule with priors `[0.5, 0.3, 0.2]`, constant zero values, and 24 simulations yields visits `[12, 7, 5]`, which is basically just prior-shaped exploration. That is useful for smoke-testing the plumbing, but it is not semantically defensible as a training teacher.

So:

* **visits as teacher**: yes
* **visits with no evaluator**: no

### 3) The only current public-compatible evaluator that stays narrow is the model value head

Candidates that fail:

* **Exact hidden-state rollout / oracle evaluator**: invalid for student labels in this imperfect-information setting; the label would depend on privileged hidden state.
* **Bridge/runtime signals (`risk_score`, `ΔQ` plane, Hand-EV summaries)**: these are feature-side/runtime summaries, not an already-defined student teacher.
* **`root_exit_policy()` / q-softmax**: wrong teacher object.
* **Prior-only AFBS visits**: smoke-test only.

Candidate that survives:

* **current public model value head**, used only to drive search visits

Why it survives:

* it is public-compatible
* it exists already
* it matches the ExIt/AlphaZero pattern “value head evaluates leaves, visits become teacher”
* it does not require a new head or broad AFBS redesign

Why it is still not blessed:

* current repo artifacts do not yet show that this head is calibrated strongly enough for search

### 4) The current value head may be too weak or too small-scale to move PUCT visits

This is the main caution.

If the value head is trained against the current self-play `value_target` path, its target scale is plausibly very small. Example:

[
\text{step.reward} = \frac{\text{final_score}}{100000 \cdot \text{player_step_count}}
]

A typical value might look like:

* `25000 / 100000 / 40 = 0.00625`

But AFBS exploration bonus is:

[
U(a) = 2.5 \cdot P(a) \cdot \frac{\sqrt{N}}{1+n_a}
]

At `P(a)=0.1`, `N=80`, `n_a=8`, that is about:

[
U \approx 2.5 \cdot 0.1 \cdot \frac{\sqrt{80}}{9} \approx 0.25
]

So if the value head really lives near `0.005-0.02`, raw q may be an order of magnitude too small to materially change visits.

That creates a clear rule:

* **do not invent a new `value_scale` knob in this tranche**
* first test whether raw value-head AFBS beats the alternatives
* if it does not, keep the producer off

### 5) The live producer should stay root-only in this tranche

Current AFBS shell is missing a fully evidenced deeper transition/evaluation stack in the artifacts. The narrow, repo-compatible interpretation is:

* **root-only AFBS bandit search over legal discard children**
* one public leaf value per child
* no deeper opponent-tree expansion
* no belief-stack redesign
* no q-target activation

That keeps the producer narrow and makes the teacher meaning explicit.

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
Reason: they are features/heuristics, not a closed teacher semantics.

Exact hidden-state rollouts from the live simulator
Reason: privileged/oracle teacher for a public student.

### Smoke-test only

AFBS visits with constant/zero evaluator
Reason: proves carrier and batching only.

### Defer

Public CT-SMC rollout / belief evaluator
Reason: semantically interesting, but widens into belief machinery and violates the requested narrow scope for first closure.

### Survive

Current public model value head as AFBS leaf evaluator
Reason: only narrow public-compatible evaluator already on the surface.
Status: **implementable, but default-off until accepted by the experiment matrix below**.

---

## Surviving producer algorithm

### Semantics

For a decision-time state `s` and legal discard action set `A_disc(s)`:

1. compute a masked base prior `base_pi`
2. run learner-only root-only AFBS over all legal discard children
3. evaluate each child once with the current public value head
4. let AFBS turn those child values + priors into **root visit counts**
5. call the existing visit-based target builder
6. emit `TrajectoryExitLabel` only if all existing gates pass

No q-softmax distillation. No oracle rollout. No bridge heuristic target.

---

## Exact algorithm

### Step 0: entry gate

Run the producer only inside the current self-play decision-time hook, on the pre-transition state that already feeds `StepRecord`.
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

and there are at least 2 legal discard actions.

Use the same discard-only legality rules as current `exit.rs`.
Direct artifact support: `hydra-train/src/training/exit.rs:141-155,172-179`.

### Step 2: base policy and hard-state gate

Compute the base prior from raw logits, **not** from `step.pi_old`:

```rust
let base_pi = softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0);
```

[Reconstruction note: the original uncommitted `answer_22.md` content after this point was overwritten locally during normalization. The continuation below is a grounded reconstruction based on the recoverable original prefix plus current repo sources.]

Use the current ExIt hard-state predicate as the narrow tranche gate, not a broader search router:

```rust
let exit_cfg = ExitConfig::default_phase3();
if !is_hard_state(&base_pi, exit_cfg.hard_state_threshold) {
    return None;
}
```

Why this gate survives:

* it already exists in the ExIt surface (`hydra-train/src/training/exit.rs:12-24,34-47,78-85`)
* it keeps the producer aligned with the current “hard-state-gated” doctrine (`research/design/HYDRA_RECONCILIATION.md:490-497`)
* it avoids silently widening this into broad always-on AFBS

### Step 3: build a learner-only root-only AFBS tree

Do **not** call `expand_node()` for label generation. That helper truncates to `TOP_K = 5`, which can make `build_exit_from_afbs_tree()` fail coverage even on otherwise valid states (`hydra-core/src/afbs.rs:188-219`; `hydra-train/src/training/exit.rs:8-10,210-213`).

Instead, construct the root and all legal discard children explicitly from the masked base prior:

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

This is the smallest honest change because AFBS already exposes `add_node`, `predicted_child_hash`, child lists, `run_search_iterations`, and root-visit extraction (`hydra-core/src/afbs.rs:153-165,246-314`). The only thing it avoids is the lossy top-k expansion helper.

### Step 4: evaluator contract

The evaluator must be:

* public-compatible
* learner-only
* leaf-scoring only
* root-search narrow

The surviving evaluator is the current model value head:

```rust
fn eval_child_with_value_head(model: &HydraModel<B>, child_obs: Tensor<B, 3>) -> f32 {
    let out = model.forward(child_obs);
    out.value_scalar().unwrap_or(0.0)
}
```

Grounding:

* `HydraOutput` already includes `value` and exposes `value_scalar()` (`hydra-train/src/model.rs:9-24,43-49`)
* `HydraModel::forward()` already computes `value` from the pooled public representation (`hydra-train/src/model.rs:253-271`)
* the value head already exists as a standard head, not a new architectural addition (`hydra-train/src/heads.rs:206-210,314-327`)

Blocked surface:

* the repo material in scope does **not** show a ready-made helper that takes a discard child node and directly emits the child observation tensor for value evaluation

So the buildable doctrine is: **producer stays default-off until that child-observation seam is implemented and the validation matrix below passes**. That is still narrower and more honest than blessing q-softmax or zero-value visits as a real teacher.

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

* `run_search_iterations()` is only a selection/backprop shell and is agnostic to evaluator semantics (`hydra-core/src/afbs.rs:246-263`)
* nothing in the scoped artifacts proves a deeper transition/evaluator stack is already semantically closed for live ExIt
* the user explicitly wanted the narrow unresolved evaluator question, not a broader AFBS redesign

### Step 6: canonical teacher build

After root visits exist, use the existing canonical teacher path without inventing a new label object:

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

Grounding: `build_exit_from_afbs_tree()` is the canonical producer entrypoint; `TrajectoryExitLabel` is the self-play carrier object; RL consumes `exit_target`/`exit_mask` when present (`hydra-train/src/training/exit.rs:229-291`; `hydra-core/src/arena.rs:7-29`; `hydra-train/src/selfplay.rs:365-366,412-464`; `hydra-train/src/training/rl.rs:148-156`).

### Step 7: failure semantics

If any gate fails, emit `None`.

That includes:

* not a compatible discard-only state
* not a hard state
* child-observation/evaluator seam unavailable
* root visits below threshold
* coverage below threshold
* KL valve failure
* non-finite value outputs

This is important doctrine, not style. Reconciliation explicitly says to leave clearly unavailable targets absent rather than fabricate weak labels (`research/design/HYDRA_RECONCILIATION.md:424-429`).

---

## Blocked / missing surface

The current repo still lacks one crucial evidential seam for always-on activation:

1. a clean, explicit helper that converts a candidate legal discard child into the public observation tensor the value head should score

Until that exists, the live producer is **implementable in blueprint form** but not yet fully evidenced as a completed path. That does **not** invalidate the blueprint; it means the blueprint’s default state remains off.

---

## Minimum experiment matrix

The value-head evaluator survives only if it beats the fake alternatives on the smallest decisive matrix.

### E0 — Carrier smoke test

Goal: prove the self-play seam attaches labels and RL receives them.

Pass if:

* `run_self_play_game_with_exit_labels()` emits at least some non-`None` `TrajectoryExitLabel`s on a forced hard-state fixture (`hydra-train/src/selfplay.rs:264-273,467-516`)
* `trajectories_to_rl_batch()` collates non-`None` `exit_target` and `exit_mask` (`hydra-train/src/selfplay.rs:365-366,412-464`)
* RL step consumes them without NaN/Inf (`hydra-train/src/training/rl.rs:146-156`)

### E1 — Teacher semantics equivalence

Goal: prove the producer still emits the canonical teacher object.

Pass if, on fixed fixtures:

* emitted labels match `build_exit_from_afbs_tree()` exactly
* masked target sums to 1 when emitted
* support and coverage behavior match current tests in `exit.rs` (`hydra-train/src/training/exit.rs:382-605`)

### E2 — Evaluator necessity test

Compare three candidates on the same hard discard-only states:

1. zero/constant evaluator
2. q-softmax teacher shortcut via `root_exit_policy()`
3. value-head evaluator

Pass condition for the value-head candidate:

* it changes root visits materially relative to zero-evaluator search on a meaningful fraction of hard states
* it preserves canonical visit-built teacher semantics because the teacher is still `build_exit_from_afbs_tree()`
* it avoids the doctrinal mismatch of q-softmax-as-teacher

If it does **not** change visits materially, keep the producer off instead of inventing a scale knob.

### E3 — Short training sanity

Run a small RL/BC sanity slice with the producer enabled only on hard discard-only states.

Require:

* finite loss
* nonzero exit auxiliary contribution
* no collapse in label coverage to near-zero
* no evidence that labels are just priors in disguise

The last check can be approximated by measuring whether emitted exit targets differ materially from masked base priors on accepted states.

---

## Final surviving blueprint

Use this and only this as the buildable narrow blueprint:

1. trigger only from the existing self-play decision-time hook
2. require current discard-only compatibility
3. require current ExIt hard-state gate
4. build a learner-only root-only AFBS tree over **all** legal discard children, not `expand_node()`
5. score child leaves with the **current public model value head**
6. derive labels only through `build_exit_from_afbs_tree()`
7. wrap them in `TrajectoryExitLabel`
8. emit `None` on any failed gate
9. keep the producer **default-off** until the experiment matrix passes

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

That is the narrowest semantically defensible live ExIt producer in current Hydra reality: **visit-based labels from learner-only root-only AFBS, driven by the existing public value head, default-off until validated**.]]>
  </answer_text>
  # Live AFBS ExIt evaluator blueprint

## Decision

Use **visit-based ExIt labels** and **only** visit-based ExIt labels.
The label object should remain:

* `exit_target[a] = normalized root child visits over masked discard actions`
* `exit_mask[a] = 1` only for legal discard children with enough support

The **only surviving narrow evaluator/value source** for generating those visits in current Hydra is:

* **the current public model value head, used only as a leaf scorer inside a learner-only, root-only AFBS search**

But that evaluator is **not yet blessed as always-on doctrine**. It is **default-off until it passes a small validation matrix**. If it fails validation, the live producer must emit `None`, not fall back to `root_exit_policy()`, q-softmax labels, bridge heuristics, or oracle rollouts.

---

## Direct artifact support

### 1) Current repo ExIt teacher semantics are child-visit semantics, not q-softmax semantics

The canonical target builder is `make_exit_target_from_child_visits()` and the canonical AFBS bridge is `build_exit_from_afbs_tree()`. They:

* require a compatible discard-only state
* read **root child visit counts**
* mask only legal discard actions with `visits >= 2`
* normalize by covered visit mass
* require coverage `>= 0.60`
* require average root visits per legal discard `>= 8.0`
* require KL safety check against `base_pi`

That is the current teacher meaning. `q_value()` is not used in the target builder.
Direct artifact support: `EXIT L0157-L0261`.

### 2) `root_exit_policy()` is a different object

`AfbsTree::root_exit_policy()` computes a **softmax over child q-values**. It is not the current ExIt target builder.
Direct artifact support: `AFBS L0265-L0305`.

`PonderResult::from_tree()` stores that q-softmax as `exit_policy`, with `trust_level = LearnerOnly`.
Direct artifact support: `AFBS L0419-L0438`.

So in current repo reality:

* `build_exit_from_afbs_tree()` = training-teacher path
* `root_exit_policy()` = ponder/cache/reporting path

Those are not the same thing.

### 3) AFBS currently has no built-in evaluator

`run_search_iterations()` requires an external `eval_fn: Fn(NodeIdx) -> f32`. The shell does selection and backprop only; it does not define what the value source is.
Direct artifact support: `AFBS L0246-L0263`.

So the unresolved question is real: a live producer must supply an evaluator.

### 4) Self-play carrier seam now exists

The self-play loop now has a decision-time hook that can attach `TrajectoryExitLabel` to each step, the trajectory validator enforces strict ExIt invariants, batch collation forwards labels into `RlBatch`, and RL already consumes `exit_target`/`exit_mask` when present.
Direct artifact support: `SELF L0264-L0273`, `SELF L0501-L0516`, `SELF L0411-L0464`, `ARENA L0483-L0530`, `RL L0148-L0156`.

So the unresolved piece is not carrier plumbing anymore; it is the search evaluator/value source.

### 5) The current model surface exposes a value output, with no new head required

Reconciliation explicitly says no new heads in this tranche, and the model already exposes the needed surfaces. The integration test also confirms that forward output includes `value`.
Direct artifact support: `RECON L0470-L0477`, `ITEST L0031-L0039`.

So a model-value evaluator is **available or nearly available**.

### 6) Current self-play value supervision looks weak as a search evaluator

In current self-play batch construction, `value_target` is filled with `step.reward`, and `step.reward` is produced by splitting each player’s final score evenly across that player’s steps.
Direct artifact support: `SELF L0388-L0390`, `SELF L0546-L0556`.

That does **not** directly prove the value head is useless, but it does mean the current value path is **not already evidenced as a strong search evaluator**.

### 7) `expand_node()` is incompatible with broad ExIt coverage

AFBS root expansion truncates to `TOP_K = 5`. ExIt target construction requires `coverage >= 0.60`.
Direct artifact support: `AFBS L0015-L0016`, `AFBS L0188-L0219`, `EXIT L0210-L0213`.

If legal discard count is `L`, then using `expand_node()` makes maximum possible coverage `5 / L`.

So:

* feasible only if `5 / L >= 0.60`
* i.e. only if `L <= 8`

For `L = 9`, max coverage is `0.556`; for `L = 14`, max coverage is `0.357`.
Therefore **a live ExIt producer must not use `expand_node()` for root label generation**. It must seed all legal discard children itself.

### 8) Archive doctrine already warned against broadening and against weak targets

Reconciliation says: do not broaden AFBS, expose only the minimum outputs needed for ExIt, and do not fabricate weak labels. Archive guidance further says ExIt should activate only after trust-gated AFBS label building with explicit support masks and coverage logging.
Direct artifact support: `RECON L0490-L0497`, `RECON L0424-L0429`, `A15 L0479-L0483`.

---

## External source support

ExIt’s original paper is clear that the apprentice should imitate the **root tree policy** `n(s,a)/n(s)`, not just the chosen move, because that target is cost-sensitive and better aligned with future search guidance. The same paper also uses a value network to score expanded leaves and backs those estimates up through the tree; when exact expert value is too costly, it approximates expert value with the apprentice value. ([NeurIPS Papers][1])

AlphaZero uses a joint network `(p, v)` to guide MCTS, where `v` estimates expected outcome from the position. The search returns a policy `π` from **root visit counts**, and training matches the policy head to those search probabilities while matching the value head to game outcome. That is the strongest canonical support for “value network as evaluator, visits as teacher.” ([arXiv][2])

Grill et al. show that AlphaZero’s **empirical visit distribution** tracks a regularized policy-improvement objective, while the exact reversed-KL solution is **not** a generic q-softmax. They also show that using the exact solution can outperform raw visits when simulation budgets are low. That is strong support against treating a naked q-softmax like `root_exit_policy()` as the doctrinal teacher, especially in low-budget search. ([Proceedings of Machine Learning Research][3])

---

## Inference

### 1) `root_exit_policy()` should be rejected as the training teacher

This follows from both repo semantics and the literature.

The repo’s own canonical teacher path is visit-based, not q-softmax-based.
The literature says ExIt/AlphaZero train against search-improved policies tied to **visits**, and Grill 2020 says the exact regularized improvement object is not generic q-softmax.

A concrete repo-native mismatch already appears in the current test tree.

From `EXIT`’s AFBS test tree:

* child visits = `[10, 8, 6]`
* child q-values = `[9/10, 4/8, 0.6/6] = [0.9, 0.5, 0.1]`

Then:

* canonical visit target = `[10, 8, 6] / 24 = [0.417, 0.333, 0.250]`
* `root_exit_policy(tau=1)` = `softmax([0.9, 0.5, 0.1]) = [0.472, 0.316, 0.212]`

The L1 gap is about `0.110`.

So even on current repo-style numbers, `root_exit_policy()` is **not** the same object as the current teacher.

### 2) “root-child visits only” is correct as target semantics, but not as evaluator semantics

Visits are the right **output object**. They are **not** a value source.

If AFBS has no meaningful evaluator, visit counts collapse into prior/exploration bookkeeping. Replaying the repo PUCT rule with priors `[0.5, 0.3, 0.2]`, constant zero values, and 24 simulations yields visits `[12, 7, 5]`, which is basically just prior-shaped exploration. That is useful for smoke-testing the plumbing, but it is not semantically defensible as a training teacher.

So:

* **visits as teacher**: yes
* **visits with no evaluator**: no

### 3) The only current public-compatible evaluator that stays narrow is the model value head

Candidates that fail:

* **Exact hidden-state rollout / oracle evaluator**: invalid for student labels in this imperfect-information setting; the label would depend on privileged hidden state.
* **Bridge/runtime signals (`risk_score`, `ΔQ` plane, Hand-EV summaries)**: these are feature-side/runtime summaries, not an already-defined student teacher.
* **`root_exit_policy()` / q-softmax**: wrong teacher object.
* **Prior-only AFBS visits**: smoke-test only.

Candidate that survives:

* **current public model value head**, used only to drive search visits

Why it survives:

* it is public-compatible
* it exists already
* it matches the ExIt/AlphaZero pattern “value head evaluates leaves, visits become teacher”
* it does not require a new head or broad AFBS redesign

Why it is still not blessed:

* current repo artifacts do not yet show that this head is calibrated strongly enough for search

### 4) The current value head may be too weak or too small-scale to move PUCT visits

This is the main caution.

If the value head is trained against the current self-play `value_target` path, its target scale is plausibly very small. Example:

[
\text{step.reward} = \frac{\text{final_score}}{100000 \cdot \text{player_step_count}}
]

A typical value might look like:

* `25000 / 100000 / 40 = 0.00625`

But AFBS exploration bonus is:

[
U(a) = 2.5 \cdot P(a) \cdot \frac{\sqrt{N}}{1+n_a}
]

At `P(a)=0.1`, `N=80`, `n_a=8`, that is about:

[
U \approx 2.5 \cdot 0.1 \cdot \frac{\sqrt{80}}{9} \approx 0.25
]

So if the value head really lives near `0.005-0.02`, raw q may be an order of magnitude too small to materially change visits.

That creates a clear rule:

* **do not invent a new `value_scale` knob in this tranche**
* first test whether raw value-head AFBS beats the alternatives
* if it does not, keep the producer off

### 5) The live producer should stay root-only in this tranche

Current AFBS shell is missing a fully evidenced deeper transition/evaluation stack in the artifacts. The narrow, repo-compatible interpretation is:

* **root-only AFBS bandit search over legal discard children**
* one public leaf value per child
* no deeper opponent-tree expansion
* no belief-stack redesign
* no q-target activation

That keeps the producer narrow and makes the teacher meaning explicit.

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
Reason: they are features/heuristics, not a closed teacher semantics.

Exact hidden-state rollouts from the live simulator
Reason: privileged/oracle teacher for a public student.

### Smoke-test only

AFBS visits with constant/zero evaluator
Reason: proves carrier and batching only.

### Defer

Public CT-SMC rollout / belief evaluator
Reason: semantically interesting, but widens into belief machinery and violates the requested narrow scope for first closure.

### Survive

Current public model value head as AFBS leaf evaluator
Reason: only narrow public-compatible evaluator already on the surface.
Status: **implementable, but default-off until accepted by the experiment matrix below**.

---

## Surviving producer algorithm

### Semantics

For a decision-time state `s` and legal discard action set `A_disc(s)`:

1. compute a masked base prior `base_pi`
2. run learner-only root-only AFBS over all legal discard children
3. evaluate each child once with the current public value head
4. let AFBS turn those child values + priors into **root visit counts**
5. call the existing visit-based target builder
6. emit `TrajectoryExitLabel` only if all existing gates pass

No q-softmax distillation. No oracle rollout. No bridge heuristic target.

---

## Exact algorithm

### Step 0: entry gate

Run the producer only inside the current self-play decision-time hook, on the pre-transition state that already feeds `StepRecord`.
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

and there are at least 2 legal discard actions.

Use the same discard-only legality rules as current `exit.rs`.
Direct artifact support: `EXIT L0141-L0155`, `EXIT L0172-L0179`.

### Step 2: base policy and hard-state gate

Compute the base prior from raw logits, **not** from `step.pi_old`:

```rust
let base_pi = softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0);
```

Reason:

* `step.pi_old` is action-sampling policy after self-play temperature
* the search prior and KL safety valve should compare against the raw network prior, not exploration temperature

Then extract only legal discard probabilities and apply the current hard-state helper:

```rust
let legal_discards: Vec<usize> = (0..=DISCARD_END as usize)
    .filter(|&a| step.legal_mask[a])
    .collect();

let hard_slice: Vec<f32> = legal_discards.iter().map(|&a| base_pi[a]).collect();

if !is_hard_state(&hard_slice, cfg.hard_state_threshold) {
    return None;
}
```

Use the existing threshold default `0.1`.
Direct artifact support: `EXIT L0012-L0024`, `EXIT L0078-L0085`, `ARENA L0537-L0565`.

### Step 3: dynamic visit budget from existing gates

Do not guess a new search budget. Use the current gate itself:

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

This is the minimal budget that can satisfy the existing average-visits gate without inventing a new multiplier.
Direct artifact support: `EXIT L0008-L0010`, `EXIT L0181-L0185`.

### Step 4: seed the AFBS root with **all** legal discard children

Do **not** call `expand_node()`.

Instead manually seed the root with every legal discard action so coverage is even possible:

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

For each legal discard action `a`, create the **public child observation** for the same root player.

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

* clone the current simulator state
* map hydra discard `action` back to the riichienv `Action` using the same context path as `NnActionSelector` (`infer_action_phase`, `hand_from_observation`, `hydra_to_riichienv`)
* apply only the root player’s discard into the next public state boundary needed to form a root-player observation
* do **not** roll through hidden-state-contingent opponent actions for teacher construction
* re-encode with the same public bridge path and `SafetyInfo`

This keeps the evaluator public-compatible.

If the adapter cannot produce such a child observation, return `None` and emit no label.

### Step 6: evaluate each child once with the model value head

Use a joint policy/value inference closure:

```rust
type PolicyValueFn =
    dyn FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32);
```

Only the scalar value is used in this tranche:

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

This is the evaluator choice.
The child policy output is ignored for now.

### Step 7: run root-only AFBS

Run AFBS search iterations with the cached child values:

```rust
tree.run_search_iterations(root, budget, &|child_idx| {
    child_value_by_node[&child_idx]
});
```

Because this is root-only, repeated visits to a child simply reapply its cached public value. That is intentional in this tranche.

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

This preserves the current semantics and existing validation gates.
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

Change the exit hook signature to pass the current player’s `SafetyInfo` snapshot:

```rust
E: FnMut(
    &GameState,
    &Observation,
    &StepRecord,
    &SafetyInfo,
    u32
) -> Option<TrajectoryExitLabel>
```

At the call site:

```rust
let exit_label = exit_label_fn(
    env.state,
    &obs,
    &step_record,
    env.selector.safety(step_record.player_id),
    turn,
);
```

Reason: child observation encoding needs the same public safety context as the actor path.

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

Do **not** change the AFBS shell semantics in this tranche.

### Do not use current `PonderResult::from_tree()` for label caching

That constructor stores `root_exit_policy()` q-softmax, which is not the teacher.
If caching is needed later, add a **separate learner-target cache object** that stores visit-based `target/mask`, same-net hash/version, and generation. Not in this tranche.

---

## Blocked / missing surface

### 1) Public child-state adapter

The artifacts do not prove a ready-made helper that takes:

* current `GameState`
* root player id
* candidate discard action

and returns the next **public root-player observation** after only that discard.

This is the main missing surface. The buildable fix is a tiny adapter/helper, not an AFBS redesign.

### 2) Consistent child encoding context

Child encoding needs the root player’s `SafetyInfo`.
Current hook does not pass it. That is why the self-play signature change above is required.

### 3) Value-head strength is not yet evidenced

The current artifacts do not prove that the model value head is calibrated enough, or large enough in scale, to move PUCT visits meaningfully. That is why the producer must be validation-gated.

### 4) Do not use `PonderCache` for labels yet

Current cache/result surfaces encode q-softmax `exit_policy`, not visit teacher semantics.

---

## Minimum acceptance tests

## A. Unit / integration tests that must be added immediately

### A1. `root_exit_policy` stays rejected

Construct the existing test tree and assert that visit target and `root_exit_policy(tau=1)` are not numerically identical.

This is not to test math correctness; it is to prevent future semantic drift.

### A2. All-legal root seeding is mandatory

Test two producers on a state with 9+ legal discard actions:

* producer using `expand_node()`
* producer using `seed_root_children_all_legal()`

Expected:

* `expand_node()` path returns `None`
* all-legal seeding path can pass coverage if visits support it

### A3. Producer uses visit builder, not q builder

Stub deterministic child values, run the producer, and assert:

* output equals `build_exit_from_afbs_tree(...)`
* output does not route through `root_exit_policy()` or `exit_policy_from_q()`

### A4. Structural reject tests

Add tests that the producer returns `None` on:

* incompatible state
* fewer than 2 legal discards
* not hard state
* KL reject
* insufficient coverage
* missing child observation

---

## B. Small decisive evaluator-selection matrix

This is the minimum experiment matrix if the evaluator remains underdetermined.

### Dataset

Collect a held-out set of compatible discard states from self-play **after**:

* `compatible_discard_state`
* legal discard count `>= 2`
* hard-state gate

### Candidate teachers

Evaluate these four candidates on the same states:

1. **proposed**: value-head AFBS visits
2. **baseline**: raw `base_pi`
3. **smoke control**: prior-only AFBS visits (`eval_fn = 0`)
4. **rejected control**: `root_exit_policy()` from the same searched tree

### Evaluation target

For **evaluation only**, estimate child-action quality with privileged continuation sampling from the simulator:

[
G(s,a) = \frac{1}{K}\sum_{k=1}^K \text{final_normalized_score}_{\text{root}}^{(k)}(s,a)
]

Important:

* this privileged rollout is **only** for evaluator selection
* it must **not** be used as the training label source

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

No arbitrary score threshold is needed. The choice is relative.

---

## C. Short training ablation

After B passes, run a short same-budget ablation:

* no exit labels
* prior-only AFBS exit labels
* value-head AFBS exit labels

Accept always-on training activation only if:

* value-head AFBS beats both alternatives on held-out self-play metrics
* prior-only labels do **not** produce the same gain

This is the minimum test that distinguishes “real teacher” from “plumbing noise”.

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

Implement an **opt-in learner-only live ExIt producer** that:

1. runs only on compatible discard-only hard states,
2. computes `base_pi` from raw masked policy logits at temperature `1.0`,
3. seeds AFBS root with **all** legal discard actions,
4. scores each child once with the **current public model value head** on a public child observation,
5. runs **root-only** AFBS for
   [
   N_{\text{budget}}=\max(64,\lceil 8 \cdot |A_{\text{disc}}|\rceil)
   ]
6. converts root child visits into `TrajectoryExitLabel` via the existing canonical `build_exit_from_afbs_tree()`,
7. emits `None` on any gate failure,
8. stays **default-off for real training** until it beats `base_pi`, prior-only AFBS visits, and `root_exit_policy()` in the evaluator-selection matrix.

If that validation fails, do **not** rescue the lane with `root_exit_policy()`, q-softmax, bridge heuristics, value rescaling knobs, or oracle rollouts. Keep the producer off and leave ExIt labels absent.

[1]: https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf "https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf"
[2]: https://arxiv.org/pdf/1712.01815.pdf "https://arxiv.org/pdf/1712.01815.pdf"
[3]: https://proceedings.mlr.press/v119/grill20a/grill20a.pdf "https://proceedings.mlr.press/v119/grill20a/grill20a.pdf"

  </answer_section>
</combined_run_record>
