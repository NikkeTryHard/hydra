<combined_run_record run_id="answer_15" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 15 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_15_VALIDATE_ADVANCED_TARGET_PROVENANCE_AND_PUBLIC_TEACHERS.md">
  <![CDATA[# Hydra prompt — validate advanced target provenance and public-teacher belief supervision closure

Primary source material lives in the raw GitHub links below.

## Critical directive

This is a narrow closure prompt for one of Hydra's most important unfinished seams: exactly which advanced targets are allowed, where they come from, and how public-teacher belief supervision should work without leaking hidden-state nonsense.

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior combined answers and use new retrieval only to validate, falsify, or sharpen the target-provenance and public-teacher pipeline.

Do not treat this as a broad breakthrough prompt. This is a technical closure and anti-hallucination prompt.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md`
5. `research/design/TESTING.md`
6. `research/design/SEEDING.md`
7. `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md`
8. `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md`
9. code-grounding files
10. outside retrieval only if needed to validate target semantics

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs
- `hydra-train/src/training/losses.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/training/losses.rs
- `hydra-train/src/model.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/model.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs

Relevant prior answers and variant references:
- `research/agent_handoffs/combined_all_variants/answer_2-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md
- `research/agent_handoffs/combined_all_variants/answer_3-1_combined.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_002_repo_aware_next_tranche.md

You are validating Hydra's advanced-target and public-teacher doctrine, specifically:
- replay-derived vs bridge-derived vs search-derived vs privileged-only targets
- exact presence/absence semantics for optional advanced targets
- whether `belief_fields_target`, `mixture_weight_target`, and related belief supervision should use projected/public-teacher objects or gauge-fixed marginals
- whether any currently imagined target path is semantically wrong and should stay absent

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<verbosity_controls>
- Prefer full technical exposition over compressed summary.
- Do not omit equations, tensor shapes, target semantics, presence masks, or provenance boundaries when they matter.
</verbosity_controls>

<tool_persistence_rules>
- Do not restart broad belief-search or breakthrough exploration.
- New retrieval should only validate, falsify, or sharpen target provenance and public-teacher semantics.
- Use Python in bash only if numerical sanity checks materially help target semantics or tensor interpretation.
</tool_persistence_rules>

<dependency_checks>
- Verify which targets are already represented in `HydraTargets`, which are still `None`, and which can already be built credibly.
- Verify which target families require CT-SMC/search-grade context versus replay-only information.
- Verify whether any candidate teacher object leaks realized hidden state or non-identifiable latent allocations.
</dependency_checks>

<grounding_rules>
- Ground all Hydra-specific claims in the provided docs/code.
- Mark any unevidenced target path, label source, or teacher object as `inference` or `[blocked]`.
</grounding_rules>

<self_red_team_rules>
- Ask explicitly:
  - Is this target replay-credible, or is it fake because it quietly depends on hidden/search-only state?
  - Does this target supervise the wrong semantic object?
  - Does this belief target collapse into raw Sinkhorn/table supervision that Hydra doctrine already warns against?
  - Would this target train a student on something the runtime policy can never observe or reconstruct?
</self_red_team_rules>

<minimum_falsification_rules>
- If a target path cannot be given a clean provenance and public/student semantics, reject it.
- If a belief target cannot be expressed as a projected/public teacher or gauge-fixed marginal, reject it.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not turn this into a broad “future of belief learning” memo.
- Stay inside target provenance, target semantics, and public-teacher closure.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current advanced-target doctrine and the tranche ordering already decided.
2. Produce the exact target taxonomy Hydra should obey now.
3. For each major advanced target, classify it as:
   - replay-derived now
   - bridge/search-derived later
   - public-teacher only
   - privileged-only / not student-facing
   - reject / keep absent
4. Write down the exact tensor/interface shapes and presence semantics.
5. For belief supervision, specify the exact target object that is semantically correct.

## Deliverables
1. Hydra posture reconstruction for advanced targets
2. Canonical target provenance table
3. Exact public-teacher vs privileged-teacher boundary
4. Exact belief supervision object and why raw alternatives are wrong
5. Tensor shapes / masks / presence semantics
6. Dependency closure table
7. Minimum falsification checks
8. Final recommendation: what Hydra should activate now, later, or never

## Hard constraints
- no new heads
- no broad architecture redesign
- no raw hidden-allocation supervision as student targets
- no vague “belief supervision should be better” answers without exact target objects
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_15.md">
  <![CDATA[## 1. Hydra posture reconstruction for advanced targets

Hydra’s current doctrine is narrower than “turn on every dormant head.” The reconciliation memo makes the immediate move a supervision-closure tranche: populate advanced targets where feasible, keep presence-gated optional losses explicit, prefer ExIt + ΔQ + safety-residual first, do not add new heads, and do not widen privileged pathways before the public-target path is closed. HYDRA_FINAL states the governing principle even more sharply: any target used to improve the deployable policy must be a function of the public information state, while perfect-information objects are allowed only for diagnostics or variance reduction. ([GitHub][1])

The prior combined handoffs already pushed in that same direction, but with two different emphases. `answer_2-1_combined` argued for a narrow tranche with ExIt, ΔQ, and safety residual, while deferring belief/mix/hand-type until a credible public teacher exists. `answer_3-1_combined` sharpened the belief side further: replace raw belief-field supervision with projected belief supervision, and separate replay-safe from search-derived provenance rather than silently mixing them. ([GitHub][2])

The strict repo re-check changes one important thing: some reconciliation prose is stale relative to current master. In current code, `HydraTargets` already has optional slots for `oracle_target`, `belief_fields_target`, `mixture_weight_target`, `opponent_hand_type_target`, `delta_q_target`, and `safety_residual_target`; `HydraOutput` already has matching heads; advanced loss weights still default to zero; and the replay pipeline already populates `oracle_target`, `safety_residual_target`, `belief_fields_target`, and `mixture_weight_target`. By contrast, `opponent_hand_type_target` and `delta_q_target` are still explicitly left `None` in the sample-to-target path. So “wired” and “doctrinally ready” are not the same thing. ([GitHub][3])

That matters because the reconciliation note also contains older language describing `sample.rs` and the loader path as baseline-only. That description is no longer true of master, but the doctrinal warning behind it is still right: activation should follow clean provenance and correct semantics, not mere tensor existence. ([GitHub][1])

After the stricter validation pass, Hydra’s posture narrows to this:

1. keep the head surface frozen;
2. separate replay-derived, public-teacher, search-derived, and privileged-only labels explicitly;
3. treat current Stage A belief labels as semantically unfit even though they exist in code;
4. treat ΔQ and ExIt as the right later targets, but not yet concretely closed in master;
5. keep oracle-side supervision on the privileged side;
6. keep opponent-hand-type absent until its semantics are replaced, not merely implemented. ([GitHub][1])

## 2. Canonical target provenance table

| Surface                         |  Present in current repo? | Current source path                                                               | Allowed provenance class                                    | Strict verdict now                          |
| ------------------------------- | ------------------------: | --------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------- |
| `safety_residual_target`        |                       Yes | replay loader builds it from public safety heuristic vs exact replay waits        | replay-derived now                                          | **Activate now** as narrow masked auxiliary |
| `delta_q_target`                |                       Yes | no loader path; runtime bridge only exposes discard-level feature summary         | bridge/search-derived later                                 | **Keep absent**                             |
| ExIt / exit policy              | No separate optional slot | no current explicit carrier; conceptually reuses policy head                      | search-derived later                                        | **Keep absent**                             |
| `belief_fields_target`          |                       Yes | replay Stage A builder populates `[16,34]` tensor                                 | public-teacher only                                         | **Current path reject; keep absent**        |
| `mixture_weight_target`         |                       Yes | replay Stage A builder may populate `[4]` tensor                                  | public-teacher only                                         | **Current path reject; keep absent**        |
| `opponent_hand_type_target`     |                       Yes | sample path hardcodes `None`; doc-proposed labels are eventual-yaku / oracle-yaku | reject current path; later public-teacher only if redefined | **Keep absent**                             |
| `oracle_target` / oracle critic |                       Yes | replay final-score oracle path                                                    | privileged-only / not student-facing                        | **Privileged-side only**                    |

This table is the strict synthesis of HYDRA_FINAL’s information-state rule, HYDRA_RECONCILIATION’s sequencing, the prior combined handoffs, and the actual current code surfaces. The “present in repo” column comes from `HydraTargets`, `HydraOutput`, the sample collation path, and the loader; the “allowed provenance class” column comes from the docs’ public-vs-privileged doctrine rather than from current implementation convenience. ([GitHub][4])

The one immediate survivor is `safety_residual_target`. It is already concrete, cheap, discard-masked, and replay-buildable. It is not a public teacher, but it is a narrow replay-derived calibration auxiliary with explicit action masking, which is why it survives this pass while broader hidden-state belief supervision does not. ([GitHub][5])

`belief_fields_target` and `mixture_weight_target` are the opposite case: they already exist and are already populated, but the current Stage A teacher is not semantically the right object. So these paths fail on correctness, not plumbing. ([GitHub][5])

`delta_q_target` and ExIt survive only as later target families. The bridge/runtime side has a real AFBS-facing concept for them, but master does not yet close them into a credible training target path: the bridge emits only discard-level `ΔQ` features, the training head is `[B,46]`, and there is no current `delta_q_mask`. ExIt is even less concrete in current surfaces because there is no separate optional `exit_target`; it would have to reuse the policy head or a training-only carrier without changing heads. ([GitHub][4])

`opponent_hand_type_target` fails for semantic reasons, not because the head is missing. OPPONENT_MODELING’s current label story is either eventual winning yaku class in phase 1, masked to a narrow winner subset, or oracle-visible exact yaku potential in later phases. Neither is the right student-facing object. ([GitHub][6])

Compressed classification:

* **Replay-derived now:** `safety_residual_target`
* **Bridge/search-derived later:** `delta_q_target`, ExIt
* **Public-teacher only:** the canonical belief object, and only any later belief/mixture supervision derived from it
* **Privileged-only / not student-facing:** `oracle_target` / oracle critic
* **Reject / keep absent:** current Stage A `belief_fields_target`, current Stage A `mixture_weight_target`, current `opponent_hand_type_target` label path

## 3. Exact public-teacher vs privileged-teacher boundary

Let `I_t` denote Hydra’s public information state at time `t`: public action history, public board state, scores, discards, melds, dora indicators, turn/wall metadata, and any belief/search computation that itself depends only on those public inputs. A **public teacher** is any target of the form

[
T_{\text{pub}}(I_t).
]

It may be expensive. It may use Mixture-SIB, CT-SMC, AFBS, bridge code, or offline search. But it must still be a function only of public information. HYDRA_FINAL’s rule is exactly this: search targets for the deployable policy must optimize the information state, not the hidden state. ([GitHub][4])

A **privileged teacher** is anything of the form

[
T_{\text{priv}}(I_t, X_t^*, W_t^*, Y_{\text{future}}),
]

where `X_t^*` is realized concealed allocation, `W_t^*` is realized wall/hidden future state, or `Y_future` includes end-of-hand outcomes not public at `t`. Exact reconstructed waits from logs, actual concealed hands, exact yaku potential, final-score oracle targets, and realized hidden allocations all live here. These can still be useful for diagnostics, oracle baselines, or carefully fenced auxiliary labels, but they are not the public-teacher path. ([GitHub][4])

A critical closure point is that **replay reconstructibility is not the same thing as public legitimacy**. OPPONENT_MODELING is explicit that `start_kyoku` plus the replay log lets you reconstruct all concealed hands offline and obtain exact tenpai/wait labels. That makes those labels available for offline supervision, but it does **not** make them public-state objects at decision time. So any claim of the form “this is fine because the logs let us reconstruct it” is invalid for belief-target doctrine unless the target can also be written as `T_pub(I_t)`. ([GitHub][6])

This boundary produces an important asymmetry:

* current `safety_residual_target` is **privileged but narrow**: it uses exact replay waits, so it is not a public teacher, but it can still function as a small replay-derived calibration auxiliary;
* current Stage A belief labels are **publicly computable but semantically wrong**: they do not leak hidden state, but they also do not represent the correct public posterior object Hydra doctrine calls for.

So “not hidden-state leakage” is necessary but not sufficient. A publicly computable toy target can still be wrong. A privileged replay target can still be useful but must never be mistaken for the public-teacher doctrine. ([GitHub][5])

## 4. Exact belief supervision object and why raw alternatives are wrong

### 4.1 The semantically correct object

HYDRA_FINAL defines the belief stack around constrained hidden-tile allocation over a transportation polytope. Let hidden zones be

[
z \in {1,2,3,w},
]

corresponding to the three opponents’ concealed hands and the live wall. Let

[
X_t(k,z)
]

be the random hidden allocation count for tile type `k ∈ {1,\dots,34}` into zone `z`. HYDRA_FINAL defines the constrained belief object

[
B_t \in \mathcal U(r_t,s_t)
= {B \ge 0 : B\mathbf 1 = r_t,; B^\top \mathbf 1 = s_t},
]

where `r_t` are row sums from public remaining tile counts and `s_t` are hidden-zone sizes. OPPONENT_MODELING makes the intended zone semantics explicit: the column marginals are opponent concealed hand sizes plus wall remainder, which are public-state quantities. ([GitHub][4])

The correct student-facing belief teacher is therefore not a realized hidden allocation and not an arbitrary latent field tensor. It is the **public posterior expected allocation**

[
\bar B_t(k,z)
= \mathbb E[X_t(k,z)\mid I_t],
]

or an equivalent deterministic representation of that same object. For rowwise classification-style supervision, define

[
P_t(z\mid k)
= \frac{\bar B_t(k,z)}{\sum_{z'} \bar B_t(k,z')}
\quad\text{for rows with } r_t(k)>0.
]

Rows with `r_t(k)=0` should be masked out. ([GitHub][4])

If Hydra later needs a field-like representation while keeping the semantic target public and identifiable, the acceptable compromise is a **gauge-fixed row logit** derived from `\bar B_t`, for example

[
g_t(k,z)
========

## \log(\bar B_t(k,z)+\varepsilon)

\frac14\sum_{z'}
\log(\bar B_t(k,z')+\varepsilon).
]

This is still a deterministic transform of the projected public-teacher belief object. It is not raw Sinkhorn-field supervision. The prior handoff explicitly proposed this kind of gauge-fixed target as the acceptable alternative to raw field labels. ([GitHub][7])

### 4.2 Why raw realized hidden-allocation supervision is wrong

A realized replay allocation `X_t^*` is just one hidden realization consistent with the public history. Supervising the student directly on `X_t^*` asks it to predict something the deployed policy can never observe and, in general, can never reconstruct from public information alone. That is exactly what HYDRA_FINAL forbids for policy-improving search/belief targets. So raw hidden-allocation supervision is out. ([GitHub][4])

This is also why “but the logs contain `start_kyoku`” does not rescue the target. Offline exact reconstruction is a privileged replay affordance, not a runtime student semantic. ([GitHub][6])

### 4.3 Why raw Sinkhorn external-field supervision is wrong

HYDRA_FINAL describes Mixture-SIB in terms of external fields `F_\theta(k,z)` and the projected constrained belief `B_t(k,z)` over the transportation polytope. The prior combined answer correctly points out that the raw field parameterization is not the right supervision object: the projection is what has semantic meaning; the fields do not. In other words, multiple field/scaling parameterizations can represent the same projected constrained belief, so the raw field tensor is not an identifiable student target. ([GitHub][4])

A strict repo-reality nuance: the current tensor named `belief_fields_target` is already **not** raw-field supervision in practice. The current belief teacher writes projected `mixture.components[...].belief[...]` values into that slot, and the bridge does the same for runtime feature planes. So the real closure question is not “fields or projected beliefs?”; current code is already using projected cell beliefs under a misleading name. The actual question is whether those projected beliefs come from the correct public teacher. Right now they do not. ([GitHub][5])

### 4.4 Why the current Stage A belief path is semantically wrong

The `teacher/belief.rs` file you provided makes the failure mode concrete. Stage A:

* clips public remaining counts into row sums,
* collapses all hidden zones into a single `hidden_tiles` count,
* splits that total equally across four columns,
* uses a uniform kernel,
* then emits the resulting projected component beliefs and sometimes component weights.

That is not a public posterior over hidden zones. It ignores publicly known per-zone sizes, ignores event likelihoods from the action history, and has no CT-SMC posterior semantics at all. HYDRA_FINAL says CT-SMC is the search-grade belief and Mixture-SIB is the fast amortized summary; Stage A is neither. OPPONENT_MODELING’s own Sinkhorn discussion says the column marginals should be zone sizes, not an equal split of total hidden mass. ([GitHub][4])

A small sanity check makes the gate problem visible. Under the symmetric Stage A setup, a uniform 4-component mixture has entropy `ln 4 ≈ 1.386`, which is above the default `mixture_entropy_threshold = 1.15`, while the current trust formula gives about `0.70`, which is above the default `trust_threshold = 0.55`. So the default configuration can emit “belief” broadly while withholding mixture weights, even when the mixture is not carrying meaningful multimodal evidence. That is a poor credibility gate for student belief supervision. **[inference from current code and a direct numerical sanity check]**

### 4.5 Why current `belief_fields_target` and `mixture_weight_target` should stay absent

There are two independent reasons.

First, the teacher is wrong. Stage A does not compute the correct public posterior object. So even though it is public-side rather than hidden-state leaking, it still fails the semantic-object test.

Second, even a correct public posterior `\bar B_t(k,z)` does **not** uniquely determine a 4-component mixture decomposition. If the student head outputs component beliefs `B^{(\ell)}(k,z)` and weights `w_\ell`, then the aggregate marginal is

[
\bar B_t(k,z)
=============

\sum_{\ell=1}^4 w_\ell, B^{(\ell)}_t(k,z).
]

But the inverse problem is non-unique: many different mixtures can produce the same aggregate marginal. So direct supervision of per-component `belief_fields_target` and `mixture_weight_target` is underidentified unless Hydra first defines a **canonical public-teacher mixture fitting procedure** and a **canonical component ordering**. Without that, even a correct public marginal teacher does not justify raw component-wise supervision. ([GitHub][7])

There is also a loss mismatch in current code. `belief_fields_target` currently carries projected nonnegative cell values, but the training loss is BCE-with-logits over `[B,16,34]`. BCE is a natural fit for Bernoulli-like targets, not for transportation-polytope mass tables or row-normalized posterior conditionals. So even the loss/object pairing is not closed today. ([GitHub][3])

Finally, if Hydra later builds a real public belief teacher from CT-SMC, it should use weighted posterior expectations, not an unweighted particle average. In `ct_smc.rs`, `weighted_mean_tile_count(tile,col)` respects particle weights; `mean_allocation()` is just a simple particle average. A future public belief teacher should use the weighted path unless weights have been deliberately equalized by resampling. ([GitHub][8])

**Bottom line for belief supervision:** the semantically correct target is the projected public posterior `\bar B_t(k,z)` or a gauge-fixed transform of it. The currently imagined student paths through `belief_fields_target` and `mixture_weight_target` are not clean enough to activate, and raw hidden allocations or raw external fields should never be student targets.

## 5. Tensor shapes / masks / presence semantics

### 5.1 Current training-head surfaces

| Surface                 | Model output shape |           Current target slot shape |                                              Current mask shape | Current repo status  |
| ----------------------- | -----------------: | ----------------------------------: | --------------------------------------------------------------: | -------------------- |
| `oracle_critic`         |            `[B,4]` |              `oracle_target: [B,4]` |                                     `oracle_guidance_mask: [B]` | populated            |
| `belief_fields`         |        `[B,16,34]` |   `belief_fields_target: [B,16,34]` |                                       `belief_fields_mask: [B]` | populated by Stage A |
| `mixture_weight_logits` |            `[B,4]` |      `mixture_weight_target: [B,4]` |                                      `mixture_weight_mask: [B]` | populated by Stage A |
| `opponent_hand_type`    |           `[B,24]` | `opponent_hand_type_target: [B,24]` | none dedicated; current loss reuses `oracle_guidance_mask: [B]` | always `None`        |
| `delta_q`               |           `[B,46]` |            `delta_q_target: [B,46]` |                                                        **none** | always `None`        |
| `safety_residual`       |           `[B,46]` |    `safety_residual_target: [B,46]` |                                  `safety_residual_mask: [B,46]` | populated            |

These shapes come directly from `HydraOutput`, the model tests, `HydraTargets`, and the batch collation path. The `opponent_hand_type` width is `3 opponents × 8 classes = 24`, matching OPPONENT_MODELING’s `[B × 3 × K]` idea with `K=8`. ([GitHub][9])

The current loss semantics are uneven:

* `belief_fields_target` uses per-sample BCE-with-logits masked by `[B]`;
* `mixture_weight_target` uses soft cross-entropy masked by `[B]`;
* `opponent_hand_type_target` would use soft cross-entropy but only sample-level `oracle_guidance_mask`;
* `delta_q_target` uses dense MSE with **no** sample or action mask;
* `safety_residual_target` uses masked action MSE over `[B,46]`. ([GitHub][3])

### 5.2 Current sample/batch presence semantics

At the sample level, replay examples already carry:

* `oracle_target: Option<[4]>`
* `safety_residual: Option<[46]>`
* `safety_residual_mask: Option<[46]>`
* `belief_fields: Option<[16*34]>`
* `mixture_weights: Option<[4]>`
* `belief_fields_present: bool`
* `mixture_weights_present: bool` ([GitHub][10])

At batch collation time:

* if any sample has safety residual, the batch materializes `Some([B,46])` target and `Some([B,46])` mask;
* if any sample has `belief_fields` or `belief_fields_present`, the batch materializes `Some([B,16,34])` plus `belief_fields_mask [B]`;
* if any sample has `mixture_weights` or `mixture_weights_present`, the batch materializes `Some([B,4])` plus `mixture_weight_mask [B]`;
* `opponent_hand_type_target` and `delta_q_target` are still hardcoded to `None` in sample-to-target conversion. ([GitHub][10])

### 5.3 Runtime Group C / bridge-side shapes

At runtime, Group C search/belief feature planes remain fixed-superset tensors with explicit presence masks. The bridge/search side currently carries:

* `belief_fields: [16,34]`
* `mixture_weights: [4]`
* `mixture_entropy: scalar`
* `mixture_ess: scalar`
* `delta_q: [34]`
* `opponent_risk: [3,34]`
* `opponent_stress: [3]`
* belief/search/robust/context presence booleans. ([GitHub][11])

This is where one of the strongest strict-pass corrections lands: **runtime `delta_q` is `[34]`, not `[46]`**. It is a discard-level summary, not a full action-surface target. So the training head shape and runtime feature shape do not presently close into a valid supervision path. ([GitHub][4])

### 5.4 Exact presence semantics Hydra should obey

For any optional advanced target family `k`, the correct rule is:

[
L_k
===

\begin{cases}
w_k \cdot \dfrac{\sum m_k ,\ell_k}{\max(1,\sum m_k)}, & \text{if target exists and has valid support},[6pt]
0, & \text{if absent.}
\end{cases}
]

Presence controls whether the loss exists at all. It is not enough to leave a target tensor zero-filled and hope the weight handles absence. HYDRA_RECONCILIATION and the prior handoff both insist on this. Current code mostly satisfies this for safety residual and the belief/mix masks, but not for `delta_q_target` and not cleanly for opponent-hand-type partial masking. ([GitHub][1])

Strict shape pruning from this pass:

* `safety_residual` is **not** `[B,3,34]`; in current Hydra it is `[B,46]` with a discard-action mask.
* the canonical belief object is `[B,34,4]` at the semantic level; current repo surfaces are all 34-tile based, so the looser `[B,37,4]` variant from earlier brainstorming does **not** survive strict repo grounding here. ([GitHub][9])

## 6. Dependency closure table

| Target family               | Represented in current `HydraTargets`? | Produced today? |                    Credible today? | Missing dependency / exact blocker                                                                                    | Verdict              |
| --------------------------- | -------------------------------------: | --------------: | ---------------------------------: | --------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `oracle_target`             |                                    Yes |             Yes | Yes, but only as privileged oracle | must stay on privileged side                                                                                          | keep privileged-only |
| `safety_residual_target`    |                                    Yes |             Yes |                      Yes, narrowly | no blocker beyond provenance logging and low-weight activation                                                        | activate now         |
| `belief_fields_target`      |                                    Yes |             Yes |                                 No | wrong teacher object; wrong zone semantics; no public-event conditioning; component underidentification; BCE mismatch | keep absent          |
| `mixture_weight_target`     |                                    Yes |          Partly |                                 No | no canonical mixture decomposition; component IDs not stable; current teacher not credible                            | keep absent          |
| `delta_q_target`            |                                    Yes |              No |                                 No | no builder, no trust gate, no `[B,46]` action mask, bridge only has `[34]` discard summary                            | keep absent          |
| ExIt                        |              No separate optional slot |              No |                            Not yet | needs explicit search-policy carrier/provenance path reusing policy head; hard-state gated AFBS labels                | later                |
| `opponent_hand_type_target` |                                    Yes |              No |                                 No | current labels are survivorship-biased or privileged; no per-opponent mask semantics                                  | keep absent          |

Two closure facts matter most.

First, the code already closes `safety_residual_target` end to end: replay builder, batch tensor, per-action mask, head, and masked loss. That is why it survives. ([GitHub][5])

Second, the code does **not** close belief supervision even though tensors exist. The missing dependency is not one thing; it is a stack:

1. correct public teacher object,
2. correct zone sizes `s_t(z)`,
3. public-history likelihood conditioning,
4. canonical mapping from teacher object to student slot,
5. loss semantics matched to that object,
6. stable component ordering if mixture slots are used. ([GitHub][4])

For `delta_q_target`, the blocker is simpler and harsher: master lacks both a credible label producer and the mask semantics the current head would need. The bridge can produce a discard-level `ΔQ` feature summary, but that is not the same thing as a trainable `[B,46]` target with explicit support. ([GitHub][4])

ExIt is doctrinally alive but interface-blocked. Because there is no new-head allowance, the right later path is to reuse the policy head and feed it search-derived target distributions on explicitly marked search-label batches. But that carrier/provenance split is not concretely closed in current master. ([GitHub][1])

## 7. Minimum falsification checks

Anything that fails any one of these stays off.

1. **Public-state test:** can the target be written as `T(I_t)`? If it needs actual concealed hands, realized wall contents, or future outcomes, it is not a public teacher. This kills raw hidden-allocation belief labels and current hand-type oracle labels. ([GitHub][4])

2. **Replay-reconstructible is not public:** if the only reason the label exists is that MJAI logs expose `start_kyoku` and let you replay hidden state offline, it is still privileged. This kills “exact belief from replay hand reconstruction” as a student-facing doctrine. ([GitHub][6])

3. **Belief conservation test:** the teacher must represent `\bar B_t \in \mathcal U(r_t,s_t)` using true public row sums and true public zone sizes. Equal four-way splitting of total hidden mass fails this. This kills current Stage A belief supervision. ([GitHub][4])

4. **Public-history likelihood test:** a belief teacher must condition on the public action history, not just on static remaining counts. Uniform-kernel projection with no event-likelihood update fails this. This kills current Stage A as a doctrinal public teacher. ([GitHub][4])

5. **Identifiability test:** if the supervised object is not unique up to representation gauge or component permutation, do not supervise it directly. This kills raw external-field targets and non-canonical mixture weights. ([GitHub][7])

6. **Component-canonicalization test:** if runtime and teacher can order mixture components differently, direct component-wise supervision is fake. The bridge already rank-sorts components by weight for runtime feature planes, so future component supervision would need explicit canonicalization. Until then, keep it absent. ([GitHub][12])

7. **Mask completeness test:** sparse or partial labels require explicit masks at the same granularity as support. This kills current `delta_q_target` activation because there is no `[B,46]` mask and current bridge support is discard-only. It also blocks opponent-hand-type partial labeling because current loss uses only sample-level oracle masking. ([GitHub][3])

8. **Loss-object fit test:** the loss must match the target semantics. BCE on projected transportation-mass cells is not obviously the right loss; CE on arbitrary mixture IDs is not valid without canonical decomposition. This kills current belief/mix activation. ([GitHub][3])

9. **Runtime reconstructibility test:** would the deployed policy ever be able to infer the supervised object from public information, perhaps with compute? If not, reject it. This kills realized hidden allocation and exact yaku potential as student targets. ([GitHub][4])

10. **Sequencing / compute test:** if activation requires turning full public-belief search into the immediate mainline, reject it for now. The reconciliation memo explicitly says not to do that yet. This kills any “just make CT-SMC belief search the whole next tranche” answer. ([GitHub][1])

Ideas that do **not** survive this pass:

* **Current Stage A belief-field supervision:** fails checks 3, 4, 5, 8.
* **Current Stage A mixture-weight supervision:** fails checks 5, 6, 8.
* **Using current bridge `ΔQ` planes as dense `[B,46]` labels:** fails check 7.
* **Eventual winning-yaku / exact-yaku hand-type labels:** fail checks 1, 2, 9, and are especially weak in a 4-player general-sum setting because they collapse latent opponent state into a winner-conditioned future outcome. ([GitHub][6])

Ideas that **do** survive:

* **`safety_residual_target` as narrow replay-derived masked auxiliary**
* **later AFBS-derived `ΔQ` and ExIt, but only after trust-gated search-label plumbing exists**
* **public-posterior belief supervision as a semantic target object, not as a currently activatable code path through Stage A**

## 8. Final recommendation: what Hydra should activate now, later, or never

### Activate now

**Activate only `safety_residual_target`, and activate it narrowly.** It is the only advanced target that is already end-to-end concrete in current master: replay builder, explicit `[B,46]` tensor, explicit `[B,46]` action mask, existing head, and masked loss. Its target semantics are simple and local:

[
s_t^*(a)
========

\operatorname{clip}(u_H(a)-d^*(a),,0,,1),
]

where `u_H(a)` is Hydra’s public safety score for a discard action and `d^\*(a)` is the exact replay-hidden immediate ron indicator for that tile type. Only legal discard actions are masked in; aka discards are mapped back to base tile type for scoring. ([GitHub][5])

This activation should carry an explicit provenance label: **replay-derived, privileged, discard-only auxiliary**. It is allowed now because it is narrow and already concretely wired, not because it satisfies the public-teacher doctrine. Do not use it as precedent for belief supervision. Its minimum falsifiable gate is simple: held-out masked regression quality on valid discard actions must improve over a zero baseline or naive public-score baseline, and primary policy metrics must not regress. ([GitHub][4])

### Activate later

**Activate ExIt and `delta_q_target` later, not now.** They are the right next target families, but only after Hydra has a trust-gated AFBS label builder that produces either full `[B,46]` support or an explicit action mask. The current bridge `ΔQ` plane is a discard-only runtime feature summary, not a trainable `[B,46]` target. ExIt can reuse the policy head, but current master does not yet have a clean search-policy carrier/provenance split distinct from ordinary replay one-hot `policy_target`. Minimum gate: hard-state AFBS only, search trust threshold, explicit support mask, and coverage logging. ([GitHub][4])

**Activate belief supervision only after the teacher object is fixed, and even then not as the current Stage A path.** The semantic target must be the public posterior marginal `\bar B_t(k,z)` or a gauge-fixed transform of it, built from a public teacher. In current repo reality, that means Stage A must stay off. A future activation would need, at minimum: true public column sizes, public-history likelihood conditioning, weighted CT-SMC posterior expectations or an equivalent valid public teacher, and a canonical mapping into the existing head surface. Minimum gate: row/column conservation residuals near zero, teacher log-likelihood/calibration better than Stage A, and no component-supervision unless decomposition canonicalization is demonstrated. ([GitHub][4])

### Never as student-facing targets

Never use these as student targets:

* raw realized hidden allocations `X_t^\*`
* raw Sinkhorn external fields as direct labels
* the current Stage A equal-split / uniform-kernel belief path
* the current Stage A mixture weights
* eventual winning-yaku class labels as the meaning of `opponent_hand_type_target`
* oracle exact-yaku-potential labels as student targets

These all fail either the public-state test, the semantic-object test, or the identifiability test. ([GitHub][4])

### Privileged-side only

Keep `oracle_target` / oracle critic on the privileged side only. The model already has a detached oracle input path, and the docs explicitly defer widening the oracle/privileged alignment work until after public-target closure. So the strict answer is not “delete oracle,” but “do not treat it as part of advanced student-target closure.” ([GitHub][1])

### Net recommendation

The stricter validation pass narrows the earlier provisional tranche. The earlier “ExIt + ΔQ + safety residual now” recommendation does **not** survive current repo reality unchanged. The evidence supports this narrower closure:

* **Now:** `safety_residual_target` only
* **Later:** ExIt and `delta_q_target`, after search-label plumbing and masks exist
* **Later, but only after semantic closure:** public-teacher belief supervision
* **Never as student targets:** raw hidden-state belief objects, raw fields, current Stage A belief/mix path, current hand-type label path
* **Privileged-only:** oracle critic

That is the tightest target-provenance and public-teacher doctrine consistent with the current Hydra docs, current master code, and the prior handoff material.

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_2-1_combined.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs"
[4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[5]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md"
[7]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_3-1_combined.md"
[8]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs"
[10]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[11]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[12]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
