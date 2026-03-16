<combined_run_record run_id="answer_27" variant_id="belief-teacher-repo-blueprint" schema_version="1">
  <metadata>
    <notes>Self-contained combined record for Agent 27. It preserves the compact prompt shell and artifact manifest generated from the authoritative prompt config, plus the preserved answer text.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — stronger public-teacher belief semantics: repo-grounded implementation blueprint

<role>
You are a Hydra research agent producing a repo-grounded implementation blueprint.
Do not give a memo. Your answer itself must be the blueprint.
</role>

<task>
Your task is only the repo-grounded implementation and validation blueprint for the stronger public-teacher belief-semantics lane.
Assume the semantic-object question is being handled by a separate research agent. You should not spend your answer re-deriving the whole belief theory from scratch unless needed for a concrete file-level decision.
We want a detailed answer that makes clear:
- exact files and functions to change first
- which existing tensor contracts, masks, and output shapes must stay unchanged
- what minimal rollout order preserves current train-bin safety gates
- what tests should be added or changed before any activation policy changes
- whether `belief_fields` can move before `mixture_weight` and what gate should keep mixture off
- what should be explicitly deferred or rejected in v1
- a step-by-step implementation-ready blueprint that minimizes guesswork and avoids widening the tranche
Prefer the smallest buildable path. Reuse existing carrier/mask/loss plumbing whenever possible. Do not invent new heads or a broad train/infer parity project unless the artifacts force it.
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

## Artifact 01 — Task scope and split
Artifact id: `belief-scope-note`
Type: `literal`
Why it matters: Explains why this prompt family is split into two narrow research agents instead of one blended packet.

## Artifact 02 — Repo status and immediate need
Artifact id: `status-readme`
Source label: README
Type: `file_range`
Source: `README.md:70-72`
Why it matters: Top-level status says the immediate project need is stronger belief-teacher semantics after recent supervision closure work.

## Artifact 03 — Reconciliation critical path
Artifact id: `reconciliation-critical-path`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:218-233`
Why it matters: Promoted execution doctrine keeps supervision closure first, Hand-EV second, and selective search later.

## Artifact 04 — Archive roadmap phase-next ranking
Artifact id: `archive-roadmap-phase-next`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:112-118`
Why it matters: Canonical archive prioritization ranks public-posterior belief teacher closure ahead of H1a Hand-EV and later selective lanes.

## Artifact 05 — Suggested execution order
Artifact id: `archive-roadmap-order`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:176-186`
Why it matters: Archive-derived execution order says strengthen public-teacher belief semantics before H1a and later selective lanes.

## Artifact 06 — Canonical belief keep-off rows
Artifact id: `archive-belief-jsonl`
Source label: JSONL
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl:10-13`
Why it matters: Canonical archive SSOT says the current Stage-A belief and mixture teachers are projection-grade and should remain off until repaired.

## Artifact 07 — Belief doctrine and CT-SMC north star
Artifact id: `final-belief-doctrine`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:121-161`
Why it matters: Promoted architecture doctrine says advanced activation remains staged because public-teacher belief semantics are not equally closed, and gives the intended Mixture-SIB plus CT-SMC belief story.

## Artifact 08 — Live encoder shape and runtime reality
Artifact id: `game-engine-shape`
Source label: ENGINE
Type: `file_range`
Source: `docs/GAME_ENGINE.md:122-124`
Why it matters: Confirms the fixed 192x34 compatibility surface and current runtime-reality authority note.

## Artifact 09 — Current Stage-A belief teacher implementation
Artifact id: `teacher-stage-a`
Source label: BELIEF
Type: `file_full`
Source: `crates/hydra-train/src/teacher/belief.rs`
Why it matters: This is the exact weak seam. It shows uniform kernel, equal hidden-zone totals, trust/ESS heuristics, output contract, and current tests.

## Artifact 10 — Replay loader belief target builder seam
Artifact id: `loader-belief-builder`
Source label: LOADER
Type: `file_range`
Source: `crates/hydra-train/src/data/mjai_loader.rs:308-335`
Why it matters: Shows the current Stage-A teacher is already invoked in the replay loader and exactly where a stronger teacher can replace it.

## Artifact 11 — Belief and mixture batch collation
Artifact id: `sample-belief-collation`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:316-427`
Why it matters: Shows the batch carrier, masks, and exact tensor contract already exist for belief_fields and mixture_weight.

## Artifact 12 — Belief, mixture, and hand-type loss functions
Artifact id: `loss-belief-fns`
Source label: LOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:276-314`
Why it matters: Shows the current per-sample loss objects and helps separate teacher-object questions from existing loss plumbing.

## Artifact 13 — Masked belief and mixture loss use
Artifact id: `loss-belief-mask-use`
Source label: LOSS
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:594-615`
Why it matters: Shows how belief_fields and mixture_weight are consumed when targets/masks exist and how hand-type differs.

## Artifact 14 — Train-bin advanced-loss policy block
Artifact id: `train-policy-block`
Source label: POLICY
Type: `file_full`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs`
Why it matters: Confirms belief_fields and mixture_weight are intentionally blocked in train.rs today.

## Artifact 15 — Train tests proving belief and mixture remain blocked
Artifact id: `train-tests-block`
Source label: TRAINTEST
Type: `file_range`
Source: `crates/hydra-train/src/bin/train.rs:1043-1073`
Why it matters: Tests prove the current keep-off posture is intentional and enforced, not accidental missing work.

## Artifact 16 — Current CT-SMC bridge seam
Artifact id: `bridge-ctsmc-counts`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:263-299`
Why it matters: Shows what the runtime currently extracts from CT-SMC today: weighted-mean counts collapsed before downstream use.

## Artifact 17 — CT-SMC weighted-mean tile count API
Artifact id: `ctsmc-weighted-mean`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:303-323`
Why it matters: Shows the current public CT-SMC summary API available at the bridge seam.

## Artifact 18 — Prompt-style guidance for Hydra research packets
Artifact id: `prompting-rules`
Source label: PROMPT
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:144-230`
Why it matters: Reminder that the agent should treat artifacts as evidence, distinguish direct support from inference, and produce an implementation-ready blueprint rather than a memo.

## Artifact 19 — Belief activation and file-by-file closure checklist
Artifact id: `reconciliation-belief-activation`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:402-480`
Why it matters: Promoted execution doctrine for when belief targets are allowed, what must stay off, and which files own the closure work.

## Artifact 20 — Opponent-modeling Sinkhorn semantics and public marginals
Artifact id: `opponent-modeling-sinkhorn`
Source label: OPPMODEL
Type: `file_range`
Source: `research/design/OPPONENT_MODELING.md:660-689`
Why it matters: Spells out row marginals, zone-size column marginals, and the intended public-state semantics for constrained belief outputs.

## Artifact 21 — Advanced head surfaces and activation note
Artifact id: `implementation-roadmap-head-surfaces`
Source label: ROADMAPREF
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:190-199`
Why it matters: Reference-only but useful summary of live advanced heads, shapes, and uneven target closure.

## Artifact 22 — Rendered canonical belief and mixture keep-off rows
Artifact id: `archive-rendered-belief-rows`
Source label: RENDERED
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md:36-39`
Why it matters: Human-readable mirror of the canonical archive rows explaining why current Stage-A belief and mixture paths stay off.

## Artifact 23 — Answer 15 belief object closure excerpt
Artifact id: `archive-answer15-belief-object`
Source label: A15
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md:232-351`
Why it matters: High-signal archive excerpt reconstructing the semantically correct public belief object, why Stage A is weak, and why current belief/mix stay off.

## Artifact 24 — Answer 18 loss and batch surface excerpt
Artifact id: `archive-answer18-loss-batch-surfaces`
Source label: A18
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_18_combined.md:232-351`
Why it matters: Archive excerpt summarizing how optional advanced targets are consumed in losses and batches, useful for spotting belief-object versus loss-shape mismatches.

## Artifact 25 — Encoder search and belief contract
Artifact id: `encoder-search-contract`
Source label: ENC
Type: `file_range`
Source: `crates/hydra-core/src/encoder.rs:1-153`
Why it matters: Defines the fixed 192x34 search/belief channel contract, including belief fields, mixture weights, entropy, ESS, and masks.

## Artifact 26 — Encoder writes belief/search planes and masks
Artifact id: `encoder-search-encode`
Source label: ENC
Type: `file_range`
Source: `crates/hydra-core/src/encoder.rs:795-830`
Why it matters: Shows exactly how search/belief planes and presence masks are written into the live encoder surface.

## Artifact 27 — Bridge search feature builder
Artifact id: `bridge-search-feature-builder`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:301-370`
Why it matters: Runtime bridge code that turns Mixture-SIB and AFBS context into belief fields, mixture weights, entropy, ESS, delta_q, and masks.

## Artifact 28 — Bridge search-feature tests
Artifact id: `bridge-search-tests`
Source label: BRIDGETEST
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:879-1015`
Why it matters: Tests proving runtime belief/search feature construction and encoder Group C population already work.

## Artifact 29 — Bridge CT-SMC context test
Artifact id: `bridge-ctsmc-hand-ev-test`
Source label: BRIDGETEST
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:1015-1064`
Why it matters: Shows CT-SMC context already flows into runtime feature production in a tested way, useful for grounding later stronger teacher ideas.

## Artifact 30 — Mixture-SIB implementation and tests
Artifact id: `sinkhorn-mixture-full`
Source label: SINK
Type: `file_full`
Source: `crates/hydra-core/src/sinkhorn.rs`
Why it matters: Full local implementation of Sinkhorn projection and Mixture-SIB weight/ESS/split-merge behavior.

## Artifact 31 — CT-SMC implementation and tests
Artifact id: `ctsmc-full`
Source label: CTSMC
Type: `file_full`
Source: `crates/hydra-core/src/ct_smc.rs`
Why it matters: Full local CT-SMC implementation showing particle structure, exact DP sampling, ESS behavior, weighted means, and tests relevant to stronger public posterior teachers.

## Artifact 32 — CT-SMC weighted mean and summary APIs
Artifact id: `ctsmc-summary-apis`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:280-379`
Why it matters: Shows the weighted-mean API, ESS gating, and mean-allocation contrast relevant to a stronger public teacher object.

## Artifact 33 — Hydra model advanced output surface
Artifact id: `model-advanced-output-surface`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:1-245`
Why it matters: Defines the live advanced output heads and model config surface that belief-related supervision must plug into without widening heads.

## Artifact 34 — Heads config and belief-related head init
Artifact id: `heads-belief-surfaces`
Source label: HEADS
Type: `file_range`
Source: `crates/hydra-train/src/heads.rs:163-309`
Why it matters: Shows belief component counts, opponent-hand-type classes, and the exact head construction surface.

## Artifact 35 — Sample and batch structures for advanced targets
Artifact id: `sample-structures`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:1-220`
Why it matters: Defines the sample-level carriers, batch target slots, masks, and advanced target storage surface.

## Artifact 36 — Advanced target collation path
Artifact id: `sample-advanced-collation`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:221-520`
Why it matters: Shows how belief, mixture, delta_q, exit, and safety residual targets/masks are carried through collation into HydraTargets.

## Artifact 37 — Train loss policy gating
Artifact id: `train-loss-policy-full`
Source label: POLICY
Type: `file_full`
Source: `crates/hydra-train/src/bin/train/loss_policy.rs`
Why it matters: Full small file showing BC/RL policy gates that block belief, mixture, and opponent-hand-type activation in train.rs today.

## Artifact 38 — Integration pipeline and target plumbing tests
Artifact id: `integration-pipeline-full`
Source label: INTEG
Type: `file_full`
Source: `crates/hydra-train/tests/integration_pipeline.rs`
Why it matters: Large end-to-end integration test surface proving what the training/runtime pipeline already closes and how advanced targets are exercised today.

## Artifact 39 — SaF observation decoder for Group C context
Artifact id: `saf-observation-decoder`
Source label: SAF
Type: `file_range`
Source: `crates/hydra-train/src/saf.rs:1-180`
Why it matters: Shows downstream train-side consumption of Group C entropy, ESS, delta_q, risk, and stress planes from the observation tensor.

</artifacts_manifest>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_27.md">
  <![CDATA[Here is the smallest repo-grounded path I would take.

`belief_fields` can move before `mixture_weight`, but only if you stop treating the 16×34 surface as a true per-component mixture teacher and instead use it as a carrier for a **single public-posterior belief object** replicated across the four existing component blocks. `mixture_weight` should stay fully off in v1, both at the producer and at `train.rs` policy-gate layers.

That recommendation is grounded by four direct repo facts.

First, the current loader seam already supports belief present while mixture absent. `build_stage_a_belief_targets` returns belief and mixture independently as `(Option<[f32;16*34]>, Option<[f32;4]>, bool, bool)`, so you already have the carrier needed for “belief on, mixture off” with no batch-schema change (Artifact 10 LOADER L0308-L0335).

Second, the batch and model contracts are already the right size. `belief_fields_target` is `[B,16,34]`, `mixture_weight_target` is `[B,4]`, with sample-level masks `[B]`; the model outputs the same shapes; no new head is required (Artifacts 11, 25, 33, 35, 36).

Third, the current Stage-A mixture is weaker than it looks. `build_stage_a_teacher` calls `MixtureSib::new(...)` and then never calls `bayesian_update`, `posterior_step`, `reproject`, split, or merge before reading back beliefs and weights (Artifact 09 BELIEF L0096-L0153; Artifact 30 SINK L0196-L0210, L0347-L0361). `MixtureSib::new` initializes **all components with the same projected belief** and equal log-weights (Artifact 30 SINK L0202-L0207). So Stage-A is already emitting identical component beliefs with uniform weights. I also checked the current trust formula numerically: with 4 uniform weights, entropy is `ln 4 ≈ 1.386294` and trust is about `0.700001`, above the default `0.55` threshold, while mixture emission is blocked by the `1.15` entropy threshold. So the current code can emit “belief” broadly while the “mixture” is uninformative and suppressed. That makes the existing trust/entropy heuristics the wrong thing to preserve.

Fourth, `mixture_weight` has an explicit identity problem in current runtime code. `build_search_features` sorts components by current weight before writing them into the fixed planes (Artifact 27 BRIDGE L0316-L0335). That means component slot 0 is “top-ranked this sample,” not a canonical cross-sample identity. So even before doctrine, the repo already treats component order as local/ranked rather than stable.

That is why I would split the work into a substrate phase and an activation phase.

## 1. Substrate phase: land the stronger lane with train gates unchanged

This is the part I would do first, before touching `train.rs` policy gates.

### 1.1 `crates/hydra-core/src/ct_smc.rs`

**Direct support:** current API exposes `weighted_mean_tile_count(tile,col)` and `mean_allocation()`; the latter is unweighted, and the archive explicitly warns that a future public belief teacher should use the weighted path, not simple particle averaging (Artifact 17 CTSMC L0303-L0323; Artifact 32 CTSMC L0341-L0359; Artifact 23 A15 L0331-L0334).

**Change first:**
Add a narrow helper:

```rust
impl CtSmc {
    pub fn weighted_mean_allocation(&self) -> [[f32; 4]; 34] {
        let mut out = [[0.0f32; 4]; 34];
        if self.particles.is_empty() {
            return out;
        }

        let max_w = self.max_log_weight();
        let mut w_sum = 0.0f64;

        for p in &self.particles {
            let w = (p.log_weight - max_w).exp();
            w_sum += w;
            for tile in 0..34 {
                for zone in 0..4 {
                    out[tile][zone] += (w * p.allocation[tile][zone] as f64) as f32;
                }
            }
        }

        if w_sum > 0.0 {
            let inv = (1.0 / w_sum) as f32;
            for row in &mut out {
                for v in row {
                    *v *= inv;
                }
            }
        }
        out
    }
}
```

**Why this is justified:** it is the smallest change that closes the weighted-posterior seam without touching heads, loaders, or runtime encoder layout.

**Tests to add before anything else:**

* `weighted_mean_allocation_respects_particle_weights`
* `weighted_mean_allocation_differs_from_mean_allocation_when_weights_skewed`

Validation is simple: create two particles with different allocations and log weights `0.0` and `-10.0`; `weighted_mean_allocation()` should sit near the first particle, while `mean_allocation()` sits near the arithmetic average.

### 1.2 `crates/hydra-train/src/teacher/belief.rs`

This is the main file to change first.

**What stays exact:**

* `BELIEF_COMPONENTS = 4`
* `BELIEF_ZONES = 4`
* `BELIEF_TILES = 34`
* flattened carrier size `16 * 34`

**What should be removed from the active path:**

* `build_uniform_kernel()`
* `project_hidden_count_to_col_sums(hidden_tiles)`
* `build_stage_a_teacher(...)` as the loader’s default path

Keep them only as legacy helpers if you want archived regression tests, but get them out of the active loader path.

**What to add:**
A narrow, semantics-agnostic encoder from a stronger public teacher object into the existing 16×34 carrier. The teacher object coming from Agent A can still vary; the encoding path does not need to.

I would standardize the train-time object as rowwise public posteriors:

[
P_t(z \mid k) = \frac{\bar B_t(k,z)}{\sum_{z'} \bar B_t(k,z')}
]

for rows with positive remaining mass, and zero rows otherwise (Artifact 23 A15 L0264-L0272).

Then encode that into the existing 16×34 slot by **repeating the same 4-zone row distribution across all four component blocks**:

```rust
pub fn encode_repeated_public_belief(
    row_post: &[[f32; BELIEF_ZONES]; BELIEF_TILES],
) -> [f32; BELIEF_FIELDS_SIZE] {
    let mut out = [0.0f32; BELIEF_FIELDS_SIZE];
    for component in 0..BELIEF_COMPONENTS {
        for zone in 0..BELIEF_ZONES {
            let ch = component * BELIEF_ZONES + zone;
            for tile in 0..BELIEF_TILES {
                out[ch * BELIEF_TILES + tile] = row_post[tile][zone];
            }
        }
    }
    out
}
```

and a normalizer:

```rust
pub fn posterior_counts_to_row_post(
    counts: &[[f32; BELIEF_ZONES]; BELIEF_TILES],
) -> [[f32; BELIEF_ZONES]; BELIEF_TILES] {
    let mut out = [[0.0f32; BELIEF_ZONES]; BELIEF_TILES];
    for tile in 0..BELIEF_TILES {
        let row_sum: f32 = counts[tile].iter().sum();
        if row_sum > 0.0 {
            for zone in 0..BELIEF_ZONES {
                out[tile][zone] = counts[tile][zone] / row_sum;
            }
        }
    }
    out
}
```

**Why I recommend duplication across component blocks:**
This is the least invasive way to reuse `[16,34]` without inventing a new head or pretending component identity exists. It is also not a semantic regression relative to the current Stage-A path, because Stage-A already initializes identical component beliefs and never differentiates them before emission.

**What to emit in v1:**

* `belief_fields = Some(encode_repeated_public_belief(...))`
* `mixture_weights = None`

Do not try to fit or emit component weights in this file in v1.

**Deterministic validity checks instead of trust heuristics:**
Replace the current `trust_threshold` / `mixture_entropy_threshold` gating with hard validity checks:

* finite
* nonnegative
* if counts are available before normalization: row sums match public remaining counts within tolerance
* if counts are available before normalization: column sums match public zone sizes within tolerance
* after normalization: each valid row sums to `1 ± eps`; invalid rows are exactly zero

That is a better gate than the current entropy/trust pair, which is not tied to teacher correctness.

**Tests to replace/add here:**

* replace `stage_a_teacher_can_emit_mixture_weights` with `public_belief_target_omits_mixture_weights_in_v1`
* replace the current “finite and nonnegative” belief test with:

  * `public_belief_target_rows_normalize_or_zero`
  * `public_belief_target_repeats_across_component_blocks`
  * `public_belief_target_rejects_nonfinite_or_negative_inputs`
  * if counts path is available: `public_belief_target_preserves_row_and_col_marginals_before_normalization`

## 2. Loader phase: swap the teacher, not the carriers

### 2.1 `crates/hydra-train/src/data/mjai_loader.rs`

The exact seam is `build_stage_a_belief_targets(...)` (Artifact 10 LOADER L0308-L0335).

**Change:**
Rename it to something like `build_belief_targets(...)` and stop computing a single `hidden_tiles` total as the active teacher input.

The loader already has the public row marginals:

* `remaining = extract_public_remaining_counts(...)`

It also already has the public zone-size pieces:

* each opponent concealed size from `state.players[*].hand_len`
* wall remainder from `state.wall.remaining()`

So the loader should compute exact zone sizes in canonical zone order and pass those to the stronger teacher provider instead of collapsing them to one total.

Do **not** restate the zone order from memory. Pin it by test. The artifact packet does not show the full table, and this is exactly the sort of thing that should be frozen by an asymmetric unit test rather than verbal recall.

Suggested helper:

```rust
fn hidden_zone_sizes(state: &GameState, actor: usize) -> [usize; 4] {
    let opp = canonical_belief_zone_order(actor); // test this explicitly
    [
        state.players[opp[0]].hand_len as usize,
        state.players[opp[1]].hand_len as usize,
        state.players[opp[2]].hand_len as usize,
        state.wall.remaining(),
    ]
}
```

**Critical v1 rule:** if the stronger teacher is not available, return `(None, None, false, false)`.
Do not silently fall back to Stage-A. That matches the doctrine to leave unavailable targets absent rather than fabricating weak labels (Artifact 19 RECON L0445-L0455).

**This is the exact v1 return you want when belief is available but mixture stays off:**

```rust
(
    Some(belief_fields),
    None,
    true,
    false,
)
```

That asymmetry is already supported by the tuple and by downstream collation.

## 3. Loss phase: repair the loss/object pairing before any activation

### `crates/hydra-train/src/training/losses.rs`

This is the second must-change file.

**Direct repo fact:** current belief loss is BCE-with-logits over `[B,16,34]` (Artifact 12 LOSS L0276-L0289; Artifact 13 LOSS L0594-L0600). The archive already flags this as a mismatch for transport-style mass tables or row-normalized posterior conditionals (Artifact 23 A15 L0331-L0332).

**Recommended v1 loss:**
Use rowwise soft cross-entropy / KL over the 4 zones for each tile, not BCE.

Why this is the smallest viable repair:

* the head already outputs raw logits
* rowwise posterior targets are probabilities over 4 zones
* no new output activation is needed
* no new batch mask tensor is needed

Implement it by reshaping `[B,16,34] -> [B,4,4,34]`, then for each component/tile row:

* compute `row_mass = sum_z target`
* if `row_mass == 0`, ignore the row
* else normalize target row and apply soft cross-entropy against the 4 logits

Conceptually:

[
L_{\text{belief}}(b)
====================

\frac{1}{|V_b|}
\sum_{c=1}^{4}
\sum_{k \in V_b}
-\sum_{z=1}^{4} P_b(z\mid k),\log \operatorname{softmax}(Y_{b,c,:,k})_z
]

where `V_b` is the set of valid tile rows for sample `b`.

That keeps the existing sample-level `belief_fields_mask: [B]` exactly as-is; the per-row validity mask is derived from the target inside the loss.

**Why no new mask carrier is needed:**
The only rows that need masking are the `r_t(k)=0` rows, and those are visible directly from the normalized target row sum. So this can be handled inside `belief_fields_*_per_sample()`.

**Tests to add here before any policy change:**

* `belief_fields_row_ce_zero_when_logits_match_target`
* `belief_fields_row_ce_ignores_zero_rows`
* `belief_fields_row_ce_is_finite_with_belief_only_targets`
* `advanced_loss_breakdown_belief_zero_when_target_absent_or_mask_zero`

A small but important detail: mean over valid rows and components so duplication across the four component blocks does **not** multiply the sample loss by 4.

## 4. Sample/collation phase: keep contracts, add assertions

### `crates/hydra-train/src/data/sample.rs`

I would avoid contract changes here. The current structures are already what you need (Artifacts 35 and 36).

**No functional surface change:**

* keep `MjaiSample.belief_fields: Option<[f32;16*34]>`
* keep `MjaiSample.mixture_weights: Option<[f32;4]>`
* keep `belief_fields_present: bool`
* keep `mixture_weights_present: bool`

**One narrow safety improvement I would add before activation:**
assert target/presence consistency for belief and mixture, the same way exit and delta-Q already panic on target/mask mismatch.

Something like:

```rust
assert_eq!(
    belief_fields.is_some(),
    sample.belief_fields_present,
    "belief_fields target/presence mismatch"
);
assert_eq!(
    sample.mixture_weights.is_some(),
    sample.mixture_weights_present,
    "mixture_weight target/presence mismatch"
);
```

Reason: today a bad producer could copy a belief target into the batch while leaving the sample mask at zero, and the loss would silently skip it. The doctrine explicitly says target presence should govern loss existence and plumbing should not silently disappear (Artifact 19 RECON L0463-L0469).

**Tests to add:**

* `collate_batch_supports_belief_without_mixture`
* `belief_fields_target_presence_mismatch_panics`
* `mixture_weight_target_presence_mismatch_panics`
* `belief_fields_suit_augmentation_preserves_repeated_component_pattern`

## 5. Activation policy: change this last

### Phase 0: keep `train.rs` gates exactly as they are

`crates/hydra-train/src/bin/train/loss_policy.rs` should remain unchanged until the new teacher encoding, loss, and tests land. The current rejection of `advanced_loss.belief_fields` and `advanced_loss.mixture_weight` is deliberate and tested (Artifacts 14, 15, 37).

This preserves today’s train-bin safety.

### Phase 1: unlock belief only, still default-off

After the substrate tests pass, do one narrow policy move:

* in `build_loss_config(...)`, allow `belief_fields`
* continue rejecting `mixture_weight`
* continue rejecting `opponent_hand_type`
* leave RL-side `build_rl_loss_config(...)` unchanged in v1 to avoid widening the tranche

That makes the first activation BC/train-only and opt-in.

I would not make belief default-nonzero in the same PR. Keep the default at `None` or `0.0`; require explicit config for pilot runs.

**Tests to change only at this point:**

* replace `build_loss_config_rejects_belief_fields_activation`
* replace `build_loss_config_rejects_belief_fields_even_at_zero`
  with:
* `build_loss_config_accepts_belief_fields_activation`
* `build_loss_config_accepts_zero_belief_fields_key`

and keep:

* `build_loss_config_rejects_mixture_weight_activation`

unchanged.

### Phase 2: pilot belief only

Only after Phase 1, run a narrow BC pilot with:

* belief target present
* mixture absent
* small auxiliary weight

The only numeric prior I see in the repo docs is the older opponent-modeling suggestion of `0.02` for Sinkhorn-style belief supervision (Artifact 20 OPPMODEL L0676-L0679). I would treat `0.02` as a reasonable **starting** belief weight for pilot runs, not as a doctrinal constant.

## 6. Why `belief_fields` can move before `mixture_weight`

Yes, it can, and the repo already supports the asymmetry.

**Direct support:**

* loader return tuple already separates belief from mixture (Artifact 10)
* sample collation already separates belief from mixture (Artifacts 11 and 36)
* loss breakdown already separates `l_belief` from `l_mix` (Artifact 13)
* train policy already blocks them independently (Artifacts 14 and 37)

**Why mixture must still stay off:**

1. No canonical cross-sample component identity is closed in current repo doctrine or code (Artifacts 6, 22, 23).
2. Runtime bridge code explicitly re-ranks components by weight per sample before encoding (Artifact 27).
3. Stage-A components are currently identical anyway, so the component axis is not an informative teacher surface today (Artifacts 9 and 30).

So the v1 gate for mixture should be two-layered:

* producer gate: always emit `mixture_weights = None`, `mixture_weights_present = false`
* train gate: keep `reject_blocked_advanced_loss_presence("mixture_weight", ...)` unchanged

That is the narrowest safe gate.

The external literature also supports this split. Sinkhorn-style OT is about a regularized transport/projection object computed by matrix scaling, not about magically giving you a canonical latent component identity; if you later want stable component alignment, that becomes a separate assignment/permutation problem, exactly the kind of problem adjacent Sinkhorn work treats explicitly rather than implicitly. ([NeurIPS Proceedings][1])

## 7. What I would explicitly defer or reject in v1

Reject in v1:

* no new 4×34 aggregate belief head
* no new mixture canonicalization scheme
* no `mixture_weight` activation
* no `opponent_hand_type` activation
* no live RL belief-label path
* no “student belief output feeds runtime encoder” parity project
* no broad rename of `belief_fields` / `mixture_weight_logits` surfaces
* no Stage-A fallback when stronger labels are absent

Defer until later:

* canonical mixture fitting / component ordering
* using true multimodal belief decompositions rather than repeated aggregate posterior
* any train/infer use of student belief head outputs as live search features
* any belief-side search/ponder integration beyond the existing CT-SMC/search seams

If later you need canonical mixture identity, treat that as a separate lane. The adjacent literature frames that as a latent alignment/permutation problem, not as something to smuggle into a weak label surface. ([OpenReview][2])

## 8. What is safe to implement now even before Agent A finishes

These changes are robust no matter which stronger teacher object Agent A chooses:

1. add `CtSmc::weighted_mean_allocation()`
2. add belief target/presence assertions in `sample.rs`
3. add rowwise belief loss tests in `losses.rs`
4. add a new belief-only integration test in `integration_pipeline.rs`
5. stop planning any `mixture_weight` activation in this tranche

The only part that depends on Agent A is the actual provider of the `34×4` public posterior counts or row distributions:

* if it is reconstructible locally, wire it in `teacher/belief.rs` and call it from `mjai_loader.rs`
* if it needs replay/search provenance, materialize it as a sidecar and still terminate in the same existing `[16*34]` carrier

That stays aligned with the repo’s sidecar-first pattern for other advanced labels and avoids widening the batch surface (README status note; Artifact 19 RECON).

## Confidence

High confidence:

* no new heads or carrier shapes are needed
* `mixture_weight` should stay off in v1
* loss repair must happen before any belief activation
* `train.rs` policy gates should move last

Medium confidence:

* the best smallest path is repeated aggregate posterior across the four component blocks
* rowwise soft CE/KL is the right v1 loss repair

Blocked on Agent A’s output:

* whether the stronger public teacher is locally reconstructible in loader code or should come from a search/CT-SMC sidecar

The SMC side of this is also on solid ground: ESS-based gating/resampling with a particle-count threshold is standard, with `N/2` a common heuristic, so Hydra’s existing `0.4 * P` CT-SMC threshold is already in the normal range and does not need a new belief-specific retune in this tranche. ([Oxford Statistics][3])

[1]: https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf "https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf"
[2]: https://openreview.net/forum?id=Byt3oJ-0W "https://openreview.net/forum?id=Byt3oJ-0W"
[3]: https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf "https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
