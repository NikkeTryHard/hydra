<combined_run_record run_id="answer_28" variant_id="belief-teacher-semantic-object" schema_version="1">
  <metadata>
    <notes>Self-contained combined record for Agent 28. It preserves the compact prompt shell and artifact manifest generated from the authoritative prompt config, plus the preserved answer text.</notes>
    <layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
  <![CDATA[# Hydra prompt — stronger public-teacher belief semantics: semantic object blueprint

<role>
You are a Hydra research agent producing a semantically exact belief-teacher blueprint.
Do not give a memo. Your answer itself must be the blueprint.
</role>

<task>
Your task is only the semantic teacher object.
Produce the strongest exact blueprint for a stronger public-posterior belief teacher that can replace the current Stage-A projection object in `crates/hydra-train/src/teacher/belief.rs`.
Do not produce a broad repo implementation plan. Do not split attention across AFBS, Hand-EV, or unrelated lanes.
We want a detailed answer that makes clear:
- what the current Stage-A teacher is really doing mathematically
- which parts are directly supported by artifacts versus inference
- what the stronger public teacher object should be for `belief_fields`
- whether `mixture_weight` should remain off for now and why
- what invariants the replacement teacher must satisfy
- what should be kept, narrowed, deferred, or rejected
- the narrowest semantically honest v1 teacher object that can be implemented now with minimal guesswork
Treat the artifacts as evidence, not truth. If the final teacher object is still partly underdetermined, make that visible instead of faking certainty.
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

## Artifact 08 — Current Stage-A belief teacher implementation
Artifact id: `teacher-stage-a`
Source label: BELIEF
Type: `file_full`
Source: `crates/hydra-train/src/teacher/belief.rs`
Why it matters: This is the exact weak seam. It shows uniform kernel, equal hidden-zone totals, trust/ESS heuristics, output contract, and current tests.

## Artifact 09 — Current CT-SMC bridge seam
Artifact id: `bridge-ctsmc-counts`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:263-299`
Why it matters: Shows what the runtime currently extracts from CT-SMC today: weighted-mean counts collapsed before downstream use.

## Artifact 10 — CT-SMC weighted-mean tile count API
Artifact id: `ctsmc-weighted-mean`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:303-323`
Why it matters: Shows the current public CT-SMC summary API available at the bridge seam.

## Artifact 11 — Prompt-style guidance for Hydra research packets
Artifact id: `prompting-rules`
Source label: PROMPT
Type: `file_range`
Source: `research/agent_handoffs/PROMPT_STYLE_GUIDE.md:144-230`
Why it matters: Reminder that the agent should treat artifacts as evidence, distinguish direct support from inference, and produce an implementation-ready blueprint rather than a memo.

## Artifact 12 — Belief activation and file-by-file closure checklist
Artifact id: `reconciliation-belief-activation`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:402-480`
Why it matters: Promoted execution doctrine for when belief targets are allowed, what must stay off, and which files own the closure work.

## Artifact 13 — Opponent-modeling Sinkhorn semantics and public marginals
Artifact id: `opponent-modeling-sinkhorn`
Source label: OPPMODEL
Type: `file_range`
Source: `research/design/OPPONENT_MODELING.md:660-689`
Why it matters: Spells out row marginals, zone-size column marginals, and the intended public-state semantics for constrained belief outputs.

## Artifact 14 — Advanced head surfaces and activation note
Artifact id: `implementation-roadmap-head-surfaces`
Source label: ROADMAPREF
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:190-199`
Why it matters: Reference-only but useful summary of live advanced heads, shapes, and uneven target closure.

## Artifact 15 — Rendered canonical belief and mixture keep-off rows
Artifact id: `archive-rendered-belief-rows`
Source label: RENDERED
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md:36-39`
Why it matters: Human-readable mirror of the canonical archive rows explaining why current Stage-A belief and mixture paths stay off.

## Artifact 16 — Answer 15 belief object closure excerpt
Artifact id: `archive-answer15-belief-object`
Source label: A15
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_15_combined.md:232-351`
Why it matters: High-signal archive excerpt reconstructing the semantically correct public belief object, why Stage A is weak, and why current belief/mix stay off.

## Artifact 17 — Answer 18 loss and batch surface excerpt
Artifact id: `archive-answer18-loss-batch-surfaces`
Source label: A18
Type: `file_range`
Source: `research/agent_handoffs/combined_all_variants/answer_18_combined.md:232-351`
Why it matters: Archive excerpt summarizing how optional advanced targets are consumed in losses and batches, useful for spotting belief-object versus loss-shape mismatches.

## Artifact 18 — Encoder search and belief contract
Artifact id: `encoder-search-contract`
Source label: ENC
Type: `file_range`
Source: `crates/hydra-core/src/encoder.rs:1-153`
Why it matters: Defines the fixed 192x34 search/belief channel contract, including belief fields, mixture weights, entropy, ESS, and masks.

## Artifact 19 — Encoder writes belief/search planes and masks
Artifact id: `encoder-search-encode`
Source label: ENC
Type: `file_range`
Source: `crates/hydra-core/src/encoder.rs:795-830`
Why it matters: Shows exactly how search/belief planes and presence masks are written into the live encoder surface.

## Artifact 20 — Bridge search feature builder
Artifact id: `bridge-search-feature-builder`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:301-370`
Why it matters: Runtime bridge code that turns Mixture-SIB and AFBS context into belief fields, mixture weights, entropy, ESS, delta_q, and masks.

## Artifact 21 — Bridge search-feature tests
Artifact id: `bridge-search-tests`
Source label: BRIDGETEST
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:879-1015`
Why it matters: Tests proving runtime belief/search feature construction and encoder Group C population already work.

## Artifact 22 — Mixture-SIB implementation and tests
Artifact id: `sinkhorn-mixture-full`
Source label: SINK
Type: `file_full`
Source: `crates/hydra-core/src/sinkhorn.rs`
Why it matters: Full local implementation of Sinkhorn projection and Mixture-SIB weight/ESS/split-merge behavior.

## Artifact 23 — CT-SMC implementation and tests
Artifact id: `ctsmc-full`
Source label: CTSMC
Type: `file_full`
Source: `crates/hydra-core/src/ct_smc.rs`
Why it matters: Full local CT-SMC implementation showing particle structure, exact DP sampling, ESS behavior, weighted means, and tests relevant to stronger public posterior teachers.

## Artifact 24 — CT-SMC weighted mean and summary APIs
Artifact id: `ctsmc-summary-apis`
Source label: CTSMC
Type: `file_range`
Source: `crates/hydra-core/src/ct_smc.rs:280-379`
Why it matters: Shows the weighted-mean API, ESS gating, and mean-allocation contrast relevant to a stronger public teacher object.

## Artifact 25 — Hydra model advanced output surface
Artifact id: `model-advanced-output-surface`
Source label: MODEL
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:1-245`
Why it matters: Defines the live advanced output heads and model config surface that belief-related supervision must plug into without widening heads.

## Artifact 26 — Heads config and belief-related head init
Artifact id: `heads-belief-surfaces`
Source label: HEADS
Type: `file_range`
Source: `crates/hydra-train/src/heads.rs:163-309`
Why it matters: Shows belief component counts, opponent-hand-type classes, and the exact head construction surface.

## Artifact 27 — Sample and batch structures for advanced targets
Artifact id: `sample-structures`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:1-220`
Why it matters: Defines the sample-level carriers, batch target slots, masks, and advanced target storage surface.

## Artifact 28 — Advanced target collation path
Artifact id: `sample-advanced-collation`
Source label: SAMPLE
Type: `file_range`
Source: `crates/hydra-train/src/data/sample.rs:221-520`
Why it matters: Shows how belief, mixture, delta_q, exit, and safety residual targets/masks are carried through collation into HydraTargets.

## Artifact 29 — Integration pipeline and target plumbing tests
Artifact id: `integration-pipeline-full`
Source label: INTEG
Type: `file_full`
Source: `crates/hydra-train/tests/integration_pipeline.rs`
Why it matters: Large end-to-end integration test surface proving what the training/runtime pipeline already closes and how advanced targets are exercised today.

## Artifact 30 — SaF observation decoder for Group C context
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
  <answer_text status="preserved" source_path="agent_28.md">
  <![CDATA[The replacement object should be **a single public posterior over hidden allocation**, not a fake 4-component mixture.

## 1. What the current Stage-A teacher is actually doing

**Direct artifact support.** `belief.rs` takes `remaining[34]`, clips it to nonnegative row sums, turns one scalar `hidden_tiles` into four equal column sums, uses an all-ones kernel, calls `MixtureSib::new`, and then writes the returned component beliefs into the `[4 components × 4 zones × 34 tiles]` tensor. `MixtureSib::new` itself computes **one** Sinkhorn projection and clones that same table into every component with equal log-weights; Stage A never calls `bayesian_update`. (Artifact 08, BELIEF L0077-L0152; Artifact 22, SINK L0196-L0209.)

From that code, the Stage-A object has a closed form.

Let

[
r_k := \max(\texttt{remaining}[k], 0), \qquad R := \sum_k r_k,
]
[
c_z := H/4 \quad \text{for } z\in{1,2,3,w}, \qquad H:=\texttt{hidden_tiles},
]
[
K_{kz} := 1.
]

Because the kernel is constant, Sinkhorn returns the rank-1 table

[
B_{kz}=\frac{r_k c_z}{R}=\frac{r_k H}{4R}.
]

And because `MixtureSib::new` clones that same `B` into every component,

[
B^{(1)}=B^{(2)}=B^{(3)}=B^{(4)}=B,
\qquad
w_\ell = \frac14.
]

So the emitted `belief_fields` tensor is just **four identical copies** of the same 4-zone table. This is not a multimodal teacher. It is one outer-product table duplicated four times. (Artifact 08, BELIEF L0111-L0144; Artifact 22, SINK L0196-L0209.)

The row-conditional object makes the weakness even clearer:

[
P(z\mid k)=\frac{B_{kz}}{\sum_{z'} B_{kz'}} = \frac14
]

for every tile row with positive mass. So Stage A’s hidden-zone belief is **uniform across the 4 zones for every tile type**. The only nontrivial quantity it carries is the public row magnitude (r_k), which is already determined by visible tiles. That is why the archive calls it a projection artifact rather than a posterior. Confidence: **high**. (Artifact 06/15; Artifact 16 L0300-L0314.)

There is a second, narrower bug-like seam in the current tests. The test case uses `remaining = [1;34]`, so (R=34), but passes `hidden_tiles=40`, so (\sum_z c_z = 40). Under the formula above, each row becomes (40/34 \approx 1.17647), not 1. So the current tests do **not** validate conservation against the supplied row marginal; they only check finiteness/nonnegativity. That is a direct consequence of the artifact code and the literal test inputs, not a guess about production behavior. Confidence: **high**. (Artifact 08, BELIEF L0165-L0187.)

The trust gate is also much weaker than it looks. Since all component weights are always uniform at construction time, Stage A always has

[
\text{ESS}=L,\qquad H_w=\log L.
]

For the default (L=4),

[
\texttt{trust}
= 0.7\cdot \frac{\text{ESS}}{L}

* 0.3\cdot \left(1-\frac{H_w}{1.3863}\right)
  \approx 0.700001.
  ]

That is above the default threshold (0.55), so for any sample with `hidden_tiles > 0` and positive row mass, belief is basically emitted by default. But `mixture_weights` are suppressed by default because (\log 4 \approx 1.38629 > 1.15). So the default behavior is effectively **belief on / mixture off**, regardless of any real posterior evidence. Confidence: **high**. (Artifact 08, BELIEF L0067-L0075, L0113-L0144; Artifact 22, SINK L0367-L0441.)

## 2. What the stronger public teacher object should be

The semantically correct object is the **public posterior expected hidden allocation**.

Let (I_t) be the public information state. Let hidden allocation be a fixed-margin contingency table

[
X_t \in \mathbb{Z}_{\ge 0}^{34\times 4},
]

with row sums equal to public remaining counts (r_t(k)) and column sums equal to public hidden-zone sizes (s_t(z)) (three opponent concealed hands plus wall). Hydra’s own doctrine already points to exactly this transport/polytope view, and in partially observed control more generally the correct belief object is a probability distribution over hidden states, not a single hidden realization. Sinkhorn’s role in the fast path is to compute a KL / entropic projection onto those fixed marginals; the projected plan is the semantically meaningful object. ([MIT CSAIL][1])

So the canonical teacher should be

[
q_t(X) := p_{\text{teacher}}(X\mid I_t),
\qquad
\bar B_t(k,z) := \mathbb{E}_{q_t}[X_t(k,z)].
]

That is the object that should replace the current Stage-A pseudo-posterior. It is:

* public-side,
* identifiable,
* permutation-invariant,
* and still lives in the same 34×4 hidden-zone space.

This is exactly the object the archive excerpt reconstructs as the “projected public posterior expected allocation,” and it is the object that stays meaningful even if the underlying posterior engine is CT-SMC or Mixture-SIB. Confidence: **high**. (Artifact 07, FINAL L0147-L0161; Artifact 13, OPPMODEL L0663-L0672; Artifact 16, A15 L0234-L0286.)

## 3. What `belief_fields` should mean

For the **canonical** teacher, keep (\bar B_t) as the SSOT object.

For the **`belief_fields` carrier**, the cleanest deterministic representation is the row-conditional version:

[
P_t(z\mid k)=
\begin{cases}
\bar B_t(k,z) / r_t(k), & r_t(k)>0 \
\text{masked}, & r_t(k)=0.
\end{cases}
]

Why this is the right `belief_fields` object:

1. It preserves exactly the same posterior semantics as (\bar B_t).
2. It removes the trivial public row-marginal factor, which the observation already determines.
3. It turns each tile row into a genuine 4-way uncertainty target.
4. It fits the doctrinal rule “projected/public-teacher belief objects or gauge-fixed marginals, not raw fields.” (Artifact 12, RECON L0407-L0409, L0467-L0469; Artifact 16, A15 L0257-L0286.)

I would **not** supervise raw Sinkhorn external fields (F_\theta). Those are gauge-dependent: adding row and column potentials can leave the projected plan unchanged after scaling. More broadly, matrix-scaling theory identifies the scaled plan, not raw pre-scaling fields; even the diagonal scalings are unique only up to scalar factors. That is why the projected table or a gauge-fixed transform is the identifiable object, not the raw field tensor. ([ScienceDirect][2])

If Hydra later wants a logit-like representation instead of probabilities, the acceptable transform is the archive’s gauge-fixed row logit:

[
g_t(k,z)=
\log(\bar B_t(k,z)+\varepsilon)
-\frac14\sum_{z'}\log(\bar B_t(k,z')+\varepsilon).
]

But that is a **derived encoding** of the same (\bar B_t), not a different teacher.

## 4. What the teacher source should be

The strongest exact teacher source is:

[
\bar B_t(k,z)=\texttt{ct_smc.weighted_mean_tile_count}(k,z).
]

That is already available cellwise from current CT-SMC APIs, and the repo artifacts explicitly distinguish that weighted path from the unweighted `mean_allocation()` helper. The weighted path is the correct one whenever particle weights still encode posterior likelihood. Confidence: **high**. (Artifact 10, CTSMC L0303-L0323; Artifact 24, CTSMC L0303-L0359; Artifact 16, A15 L0331-L0334.)

The **narrowest semantically honest v1** is therefore:

* **use CT-SMC weighted posterior mean when CT-SMC is present;**
* **otherwise emit no belief label.**

I would **not** let Stage A survive as a fallback. Artifact doctrine is explicit that unavailable targets should stay absent rather than be replaced by weak fabricated labels. Confidence: **high**. (Artifact 12, RECON L0444-L0455.)

There is one acceptable **later** fallback inside the same lane: if Hydra has a real posterior-updated Mixture-SIB object with correct public marginals and nontrivial kernels, then the permutation-invariant aggregate

[
\bar B_t = \sum_{\ell} w_\ell B_t^{(\ell)}
]

is a semantically valid approximate teacher. But that is a coverage-expansion fallback, not the narrowest honest v1. Confidence: **medium**. (Artifact 07, FINAL L0133-L0141; Artifact 22, SINK L0416-L0425.)

## 5. `mixture_weight` should remain off

This is the clearest answer in the packet: **keep `mixture_weight` off for now.**

Reason 1: even with a correct aggregate posterior (\bar B_t), the decomposition

[
\bar B_t = \sum_{\ell=1}^4 w_\ell B_t^{(\ell)}
]

is non-unique. Aggregate posterior does **not** identify a unique 4-component fit. Confidence: **high**. (Artifact 16, A15 L0318-L0335.)

Reason 2: the runtime side currently sorts components by descending weight before encoding them into search features. That is a within-sample ranking convention, not a stable cross-sample component-identity contract. Confidence: **high**. (Artifact 20, BRIDGE L0315-L0335.)

Reason 3: mixture models have a standard label-switching problem. Stan’s guide states that mixture components are exchangeable and that only label-switching-invariant inferences are sound; Stephens shows componentwise posterior means and marginal summaries can become nonsensical under label switching. So until Hydra defines a **canonical public-teacher mixture fit** and a **canonical component ordering**, `mixture_weight` supervision is not semantically closed. ([Stan][3])

So the right policy is:

* `belief_fields`: can be repaired around the aggregate public posterior.
* `mixture_weight`: **stay off** until canonical mixture identity exists.

Confidence: **high**.

## 6. Invariants the replacement teacher must satisfy

1. **Public-only semantics.**
   The emitted target must be a function of (I_t) and a teacher-side posterior built from public information, never the realized hidden allocation.
   Validation: hold public history fixed, vary hidden realization among states consistent with it; the teacher must not change.
   Confidence: **high**. (Artifact 16, A15 L0288-L0298.) ([MIT CSAIL][1])

2. **Exact margin conservation.**
   [
   \sum_z \bar B_t(k,z)=r_t(k), \qquad
   \sum_k \bar B_t(k,z)=s_t(z).
   ]
   Validation: row/column sums checked to tolerance on every emitted sample.
   Confidence: **high**. (Artifact 13, OPPMODEL L0663-L0668.)

3. **Weighted posterior expectation, not unweighted particle average.**
   Validation: construct a nonuniform-weight particle set and verify teacher differs from `mean_allocation()`.
   Confidence: **high**. (Artifact 24, CTSMC L0303-L0359.)

4. **Permutation-invariant v1 semantics.**
   The v1 teacher must not depend on arbitrary component IDs.
   Validation: permuting mixture components leaves the aggregate teacher unchanged.
   Confidence: **high**. ([Stan][3])

5. **Zero-row masking.**
   Rows with (r_t(k)=0) are masked, not supervised.
   Validation: row mask equals `1[r_t(k)>0]`.
   Confidence: **high**. (Artifact 16, A15 L0264-L0272.)

6. **Audit metrics are not semantic gates.**
   ESS / entropy can be logged, but a concentrated posterior is not an invalid posterior.
   Validation: a single-particle / collapsed posterior should still emit a label if numerically sane and margin-consistent.
   Confidence: **medium-high**. This is a proposal, but it follows directly from how posterior concentration works and from the current Stage-A gate’s perversity.

## 7. What to keep, narrow, defer, reject

**Keep**

* the transport-polytope framing;
* the 4-zone public semantics including wall;
* the doctrine “projected/public belief object, not hidden realization”;
* CT-SMC weighted means as the search-grade source.
  Confidence: **high**.

**Narrow**

* `belief_fields` to the **aggregate public posterior** only;
* the `belief_fields` carrier to row-conditional (P(z\mid k)) or a gauge-fixed transform of that same object;
* audit stats to diagnostics only.
  Confidence: **high**.

**Defer**

* any per-component belief supervision;
* `mixture_weight` activation;
* canonical mixture fitting / relabeling / ordering contracts.
  Confidence: **high**.

**Reject**

* equalized hidden-zone totals;
* uniform-kernel Stage A as “posterior”;
* raw external fields as targets;
* realized hidden allocations as targets;
* unweighted CT-SMC averages;
* activating belief loss just because a repaired teacher exists.
  Confidence: **high**.

## 8. The narrowest semantically honest v1

This is the v1 I would actually endorse.

### Canonical semantic object

[
\bar B_t(k,z)=\mathbb{E}[X_t(k,z)\mid I_t]
]

computed from **CT-SMC weighted particles only**.

### `belief_fields`-specific teacher

[
P_t(z\mid k)=\bar B_t(k,z)/r_t(k)
]

for (r_t(k)>0), with row mask otherwise.

### Legacy `[16,34]` compatibility shim

If the current surface must stay unchanged, use:

[
\texttt{belief_fields}[c,z,k] = P_t(z\mid k)
\quad \text{for all } c\in{0,1,2,3}.
]

That is: **repeat the same repaired 4×34 posterior across all 4 component blocks** and document explicitly that the component axis is a dummy carrier in v1, not mixture semantics.

### `mixture_weight`

Always `None`.

### Presence rule

Emit target **iff** CT-SMC posterior exists and passes numeric/margin checks; otherwise `None`.

### Activation rule

Keep `w_belief_fields = 0` until the loss is changed to a rowwise CE/KL-style objective over (P_t(z\mid k)). Burn already has separate APIs for binary cross-entropy and for cross-entropy with target probabilities; the latter matches a rowwise 4-way posterior target much better than BCE-on-cells. ([Burn][4])

In code-like form:

```rust
// canonical meaning: posterior over 34 x 4 hidden-allocation marginals
for tile in 0..34 {
    for zone in 0..4 {
        bbar[tile][zone] = ct_smc.weighted_mean_tile_count(tile as u8, zone as u8);
    }
}

assert_rows_match_public_remaining(&bbar, &public_r, tol)?;
assert_cols_match_public_zone_sizes(&bbar, &public_s, tol)?;

for tile in 0..34 {
    let row = public_r[tile];
    row_mask[tile] = if row > 0.0 { 1.0 } else { 0.0 };
    for zone in 0..4 {
        p[tile][zone] = if row > 0.0 { bbar[tile][zone] / row } else { 0.0 };
    }
}

// legacy carrier shim only: no mixture semantics in v1
for component in 0..4 {
    for zone in 0..4 {
        for tile in 0..34 {
            belief_fields[(component * 4 + zone) * 34 + tile] = p[tile][zone];
        }
    }
}

mixture_weights = None;
```

That is the narrowest v1 that is both **semantically honest** and **implementable with minimal guesswork** from the current artifacts.

My bottom line:

* **Replace Stage A with CT-SMC-weighted public posterior mean semantics.**
* **Make `belief_fields` mean row-conditional posterior over the 4 public hidden zones.**
* **Leave `mixture_weight` off.**
* **Do not activate belief loss until the loss matches the repaired object.**

Confidence overall: **high on the canonical object and on keeping `mixture_weight` off; medium-high on the temporary 16-channel carrier shim; high that Stage A as it stands should not be trusted as an active teacher.**

[1]: https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf "https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf"
[2]: https://www.sciencedirect.com/science/article/pii/S0024379597000104 "https://www.sciencedirect.com/science/article/pii/S0024379597000104"
[3]: https://mc-stan.org/docs/2_20/stan-users-guide/label-switching-problematic-section.html "https://mc-stan.org/docs/2_20/stan-users-guide/label-switching-problematic-section.html"
[4]: https://burn.dev/docs/burn/nn/loss/struct.BinaryCrossEntropyLoss.html "https://burn.dev/docs/burn/nn/loss/struct.BinaryCrossEntropyLoss.html"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
