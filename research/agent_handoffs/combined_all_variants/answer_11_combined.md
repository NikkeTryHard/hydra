Fixing single URL mismatch in compressed file.Patch missed exact line form. Reading target line, then applying minimal fix.<combined_run_record run_id="answer_11" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 11 + returned agent answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<warning_note status="historical_partially_stale" added="2026-03-15">
<![CDATA[
Warning: combined archive artifact still useful for narrow CT-SMC / Hand-EV benchmark reasoning, but some train-side `delta_q` claims stale vs current code.

Known stale area:
- File speaks like masked RL-side `delta_q` handling/export still future-only. Current Hydra already has narrow RL-only masked `delta_q` lane, including target/mask carriage + masked-loss handling.

Still useful here:
- caution that replay/offline `delta_q` closure not solved by RL-only lane
- discard-centric runtime-vs-[46]-head mismatch discussion

Do not use file as current truth for:
- present-tense RL-side `delta_q` plumbing status

Validate against live authority chain before reuse:
README.md -> research/design/HYDRA_FINAL.md -> research/design/HYDRA_RECONCILIATION.md -> docs/GAME_ENGINE.md
]]>
</warning_note>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_11_VALIDATE_ACTION_SUFFICIENT_WORLD_COMPRESSION.md">
<![CDATA[# Hydra prompt — validate action-sufficient CT-SMC world compression

Primary source material in raw GitHub links below.

## Critical directive

Read core Hydra docs holistically before proposing compression scheme. Do not jump from generic POMDP compression papers to Hydra recommendations.

## Reading order
1. `research/design/HYDRA_RECONCILIATION.md`
2. `research/design/HYDRA_FINAL.md`
3. `docs/GAME_ENGINE.md`
4. `research/design/OPPONENT_MODELING.md`
5. `research/design/TESTING.md`
6. code-grounding files
7. outside retrieval

## Raw GitHub links
- `research/design/HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `research/design/HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `research/design/OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `docs/GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `hydra-core/src/ct_smc.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs
- `hydra-core/src/bridge.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs
- `hydra-core/src/hand_ev.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/hand_ev.rs
- `hydra-core/src/endgame.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/endgame.rs
- `hydra-train/src/data/sample.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/sample.rs
- `hydra-train/src/data/mjai_loader.rs` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs

Relevant prior variant writeups + prompt refs:
- `research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md

Validate whether action-sufficient world compression real long-run Hydra path.

Focus on:
- compressing CT-SMC worlds by decision relevance, not probability mass
- using current Hand-EV / endgame evaluators as local regret geometry
- whether this is cheap + real seam before deeper AFBS expansion

Broad exploration already done in `research/agent_handoffs/combined_all_variants/`. Do not redo broad work. Start from prior DEBC-AR + 8-series materials. Search only enough to validate, falsify, or tighten this specific compression lane.

<output_contract>
- Return exactly requested sections, in requested order.
- Be as detailed + explicit as needed; do not optimize for brevity.
- Return full technical treatment, not compressed memo.
- Short answer usually failure mode for this prompt.
</output_contract>

<calculation_validation_rules>
- Use Python in bash for compression-ratio arithmetic, evaluator-call accounting, + toy regret-clustering sanity checks.
- Do not leave claims like “8 worlds replaces 50-100 worlds” uncomputed.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart broad belief-compression survey.
- New retrieval should only validate, falsify, or sharpen action-sufficient compression for Hydra's actual runtime seam.
</tool_persistence_rules>

<dependency_checks>
- Verify what CT-SMC exposes today, what Hand-EV/endgame evaluators exist, + whether current runtime already supports relevant insertion points.
- Verify whether later `delta_q` export real or still future-only.
</dependency_checks>

<self_red_team_rules>
- Ask explicitly:
  - Does this only preserve noisy evaluator mistakes more efficiently?
  - Is posterior quality actual bottleneck, not compression quality?
  - Does this fail if Hand-EV realism not improved first?
  - Does result reduce to top-mass particle pruning with fancier vocabulary?
</self_red_team_rules>

<minimum_falsification_rules>
- Define minimum offline benchmark that compares top-mass particle pruning against action-sufficient compression on decision regret at equal evaluator budget.
- Reject method if it cannot beat simple top-mass pruning offline.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not drift into generic POMDP compression literature review.
- Stay inside CT-SMC world compression for Hydra's current Hand-EV / endgame seam.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current CT-SMC / Hand-EV / endgame posture.
2. Validate whether action-sufficient compression timely under reconciliation.
3. Write exact objective, compression rule, tensor/interfaces, + pseudocode.
4. Separate MVP runtime-only compression from later training export ideas.
5. Decide if this serious second-wave investment or still too early.

## Deliverables
1. Hydra posture reconstruction for CT-SMC / Hand-EV / endgame
2. Existing repo surfaces + blockers
3. Exact mathematical formulation of compression criterion
4. Tensor shapes / runtime payloads
5. Exact pseudocode
6. Dependency closure table
7. Offline falsification benchmark
8. Failure modes + kill criteria
9. Final rec: worth now, later, or not worth it

## Hard constraints
- no broad AFBS rewrite
- no new belief stack
- no fake novelty by renaming belief compression
- no training-first rec if runtime-only falsification not proven
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_11.md">
<![CDATA[## 1. Hydra posture reconstruction for CT-SMC / Hand-EV / endgame

Hydra reconciled doctrine not “search first.” Active path = supervision-first, with Hand-EV realism before deeper AFBS semantics, AFBS specialist / hard-state gated, unified belief story where Mixture-SIB = amortized belief object and CT-SMC = search-grade posterior. `OPPONENT_MODELING.md` says defer to `HYDRA_FINAL.md` and `HYDRA_RECONCILIATION.md` when active impl shape differs, and treat heavier posterior / deception extensions as reserve-shelf unless promoted by reconciled doctrine. ([GitHub][1])

`HYDRA_FINAL.md` makes CT-SMC role precise: exact contingency-table sampling replaces generic particle proposal; resampling when ESS below `0.4P`; posterior quality judged by Gate (posterior log-likelihood) and Gate B (pairwise MI calibration); late-game exactification = Pure PIMC over CT-SMC particles with 95% top-mass reducer, usually leaving about 50–100 worlds in wall-≤10 threat slice. That already probability-mass world compression policy, not decision-aware. ([GitHub][2])

Live code mostly matches posture. `ct_smc.rs` exposes exact DP sampling, default `num_particles = 128`, `ess_threshold = 0.4`, particle log-weights, `weighted_mean_tile_count`, and `ess_ratio`. Bridge already compresses CT-SMC into belief-weighted remaining counts then runs `compute_hand_ev`, while AFBS-derived `delta_q` features only populated for `0..NUM_TILE_TYPES` tile-type discard actions. `hand_ev.rs` defines per-discard tenpai / win / expected-score / ukeire features from `(hand, remaining)`. `endgame.rs` activates on `wall_remaining <= max_wall && has_threat`, and `pimc_endgame_q_topk` chooses selected particles, normalizes weights, and averages expensive `eval_fn(particle, action)` over legal actions. ([GitHub][3])

So Hydra already has two probability-oriented compression moves in runtime reality: first-moment CT-SMC→Hand-EV in bridge, and top-mass CT-SMC→PIMC in endgame. Missing object not new belief stack. It decision-relevant representative-world selector that can replace one existing compression only if it preserves action quality better than weight-only pruning at same expensive-evaluator budget. That exactly lane prior DEBC-AR / 8-series writeups circled, but reconciliation forces it much narrower than general belief-compression program. (hydra-core/src/bridge.rs:263-299; hydra-core/src/endgame.rs:80-87,90-184; research/design/HYDRA_RECONCILIATION.md:149-160,221-243)

## 2. Existing repo surfaces and blockers

Real surfaces exist today. CT-SMC gives weighted posterior over valid hidden worlds, not marginals. Bridge already knows how to turn CT-SMC into hand-aware local features. Endgame helper already has “select worlds, then aggregate expensive per-world action values” structure. AFBS / inference already expose hard-state signals like top-2 policy gap and particle ESS that could later host compression-escalation rule. Genuine seams; not fantasy “invent new belief system” proposal. (hydra-core/src/ct_smc.rs:228-258; hydra-core/src/bridge.rs:263-299; hydra-core/src/endgame.rs:80-87,90-184; hydra-core/src/afbs.rs:472-507)

First blocker = interface shape. `solve_with_particles` in `endgame.rs` gets `(particles, legal_mask, eval_fn)`, but `compute_hand_ev` needs `(hand, remaining)`. So action-sufficient selector based on current Hand-EV cannot drop in as literal one-line swap. At minimum, selector needs current hand threaded into endgame seam. Stronger production version also wants cheap safety-side vector because endgame activation threat-gated. In surfaced files, hand/safety context available around bridge / encoder code, but not yet in endgame helper API. (hydra-core/src/endgame.rs:80-87; hydra-core/src/hand_ev.rs:253-309; hydra-core/src/bridge.rs:251-299)

Second blocker = action-space realism. Cheap local geometry Hydra exposes today discard / tile-type centric, not full-action. `compute_hand_ev` indexed by 34 tile types; bridge fills `delta_q` only for `0..NUM_TILE_TYPES`; encoder’s search `delta_q` plane = `[34]`; but model’s `delta_q` head = `[46]`. That kills any honest claim Hydra already has cheap current local scorer for full 46-action space, especially call-phase and declaration actions. Surviving scope = discard-phase world compression over tile-type discard classes, not full-action compression. (hydra-core/src/hand_ev.rs:6-10,253-309; hydra-core/src/bridge.rs:343-355; hydra-train/src/model.rs:22,98,241,269,297)

There also action-identity mismatch around red fives. Current target-building code already collapses aka 5m / 5p / 5s to base tile types for safety residual targets. Another sign practical current signature space = tile-type discard classes, not raw action IDs. For this lane, fine; means exact MVP must state collapse-to-tile-type semantics instead of pretending to preserve full raw-action ontology it does not have. (hydra-train/src/data/mjai_loader.rs:283-299)

Third blocker = evaluator realism. `compute_hand_ev` offense-oriented: shanten improvement, ukeire, multi-horizon tenpai / win probabilities, and conditional score estimate. Defensive info exists elsewhere (`danger_from_particles`, safety features in bridge / encoder), not in Hand-EV scalar itself. Because endgame helper threat-gated, pure Hand-EV-only production compressor would use offense-biased signature exactly where defensive tail misses matter most. Here reconciliation matters: broad deployment should wait for better Hand-EV realism or threaded safety side-channel. (hydra-core/src/hand_ev.rs:24-43,132-141,253-309; hydra-core/src/bridge.rs:301-360; research/design/HYDRA_RECONCILIATION.md:135-146,278-283)

Fourth blocker kills full-tree version. AFBS has `particle_handle`, but it `Option<u32>` placeholder initialized as `None`; live hard-state machinery = root-exit policy, ponder reuse, and priority from top-2 gap, risk score, and particle ESS. Enough to host future escalation signal. Not enough to justify node-level world compression or broad AFBS rewrite. Under hard constraints, that version does not survive. (hydra-core/src/afbs.rs:106-114,265-314,398-459,472-507)

Fifth blocker kills training-first versions. `sample.rs` still writes `opponent_hand_type_target: None` and `delta_q_target: None`. `HydraLoss` applies dense unmasked MSE to `delta_q` if target ever appears, while only `safety_residual` has action mask. Combined with runtime `delta_q` payload being `[34]` tile-type discard-level while model head `[46]`, later `delta_q` export still future-only. Not cheap current extension. (hydra-train/src/data/sample.rs:178-181,210-213; hydra-train/src/training/losses.rs:247-262; hydra-train/src/model.rs:22,98,241,269,297)

After stricter pass, four ideas dead for now: full-tree AFBS compression, full 46-action compression, training-first compressed `delta_q` export, and any “compression” that top-mass pruning or first-moment averaging under new name. Sole survivor = runtime-only, discard-phase representative selector over CT-SMC particles, benchmarked against top-mass at equal expensive-evaluator budget.

## 3. Exact mathematical formulation of the compression criterion

Conceptual family legitimate: decision-aware / value-directed compression rather than raw belief approximation. But Hydra actual code narrows it sharply. Exact survivor = discard-phase representative-world selector over CT-SMC particles, where cheap geometry comes from current per-world Hand-EV and final decision still comes from existing expensive endgame evaluator. ([Google Research][9])

Let CT-SMC posterior at current state be particles ({(X_i,\ell_i)}_{i=1}^P), where (X_i \in {0,\dots,4}^{34\times 4}) = hidden-world contingency table and (\ell_i) = log-weight. Normalize:

[
w_i = \frac{\exp(\ell_i - \ell_{\max})}{\sum_{j=1}^{P} \exp(\ell_j - \ell_{\max})}, \qquad \sum_i w_i = 1.
]

For each particle, define world-specific unseen-tile count vector used by Hand-EV:

[
r_i[t] = \sum_{z=0}^{3} X_i[t,z], \qquad t \in {0,\dots,33}.
]

This exactly mirrors what bridge already does in expectation when it sums CT-SMC hidden columns into remaining counts; difference = do it per world instead of after first-moment averaging. (hydra-core/src/bridge.rs:263-283)

Define signature action set (A_{\text{sig}}(s)) as set of unique legal discard tile types in current hand, after collapsing aka discard actions to base tile types. In strict surviving MVP, this only honest signature action set, because current cheap geometry tile-type discard indexed. Let (|A_{\text{sig}}| = m \le 14).

For each particle (i), compute current Hand-EV features:

[
f_i = \texttt{compute_hand_ev}(h, r_i),
]

where (h \in {0,\dots,4}^{34}) = player’s current hand. Exact MVP scalar signature deliberately simple:

[
q_i^{\text{sig}}(a) = f_i.\texttt{expected_score}[a], \qquad \in A_{\text{sig}}.
]

I am not baking defensive penalty into exact MVP formula because current runtime does not expose clean endgame-side safety scalar yet. That omission production blocker, not reason to lie about math. Later risk-aware variant can replace scalar with
[
q_i^{\text{sig}}(a)= f_i.\texttt{expected_score}[a]-\lambda_{\text{risk}} d_{\text{pub}}(a),
]
but that not proven current MVP.

Now define posterior-mean signature values

[
\bar q^{\text{sig}}(a) = \sum_{i=1}^{P} w_i q_i^{\text{sig}}(a),
]

and let signature frontier (F \subseteq A_{\text{sig}}) be top (k_F) discard tile types by (\bar q^{\text{sig}}), with strict MVP default (k_F = 3).

Define per-world regret vectors on that frontier:

[
R_i(a) = \max_{b \in A_{\text{sig}}} q_i^{\text{sig}}(b) - q_i^{\text{sig}}(a), \qquad \in F.
]

Decision-aware world distance then

[
d(i,j) = \frac{1}{|F|} \sum_{a \in F} \left| R_i(a) - R_j(a) \right|.
]

Core point: posterior mass still weights objective, but geometry = action distortion, not raw hidden-world distance. Top-mass baseline sorts by (w_i) alone. This objective clusters worlds by how much they disagree about discard frontier.

Tail protection explicit. For each frontier action \in F), define (T_\alpha(a)) as smallest set of highest-regret particles whose total posterior mass at least (\alpha), with strict MVP default (\alpha = 0.10). Medoid initialization must include at least one seed from union of these tail sets before ordinary weighted k-medoids proceeds. This prevents method from degenerating into “top-mass with fancier vocabulary.”

Weighted medoid problem

[
\min_{M,\phi} J(M,\phi)
\quad\text{where}\quad
J(M,\phi)=\sum_{i=1}^{P} w_i, d(i,\phi(i)),
]

subject to (|M|=K) and (\phi(i)\in M).

If (M={m_1,\dots,m_K}) selected medoids, compressed cluster weights

[
W_k = \sum_{i:\phi(i)=m_k} w_i.
]

Final action values are **not** taken from cheap signature. They come from Hydra’s existing expensive evaluator on medoids only:

[
\hat Q_K(u)=\sum_{k=1}^{K} W_k, Q^{\text{eval}}(X_{m_k},u),
\qquad u \in A_{\text{eval}}(s),
]
where (A_{\text{eval}}(s)) = actual legal-action set under 46-action mask, and (Q^{\text{eval}}(X,u)) = current `eval_fn(&Particle, action)` used by endgame helper.

Cheap compression certificate signature-level, not truth-level:

[
\varepsilon_{\text{sig}}
========================

\max_{a \in F}
\left|
\sum_{i=1}^{P} w_i R_i(a)
-------------------------

\sum_{k=1}^{K} W_k R_{m_k}(a)
\right|.
]

Let (\Delta_{\text{sig}}) be top-2 gap in compressed signature frontier:
[
\Delta_{\text{sig}}=
\bar q^{\text{comp}}*{(1)}-\bar q^{\text{comp}}*{(2)},
\qquad
\bar q^{\text{comp}}(a)=\sum_{k=1}^{K} W_k q_{m_k}^{\text{sig}}(a).
]

If
[
\Delta_{\text{sig}} > 2 \varepsilon_{\text{sig}},
]
then signature winner stable under this compression. But this only certifies clustering preserved cheap local geometry. It does **not** certify true endgame optimality, because (q^{\text{sig}}) only proxy for (Q^{\text{eval}}). That why real go / no-go criterion must be offline full-evaluator regret, not certificate itself.

Arithmetic only works if scope stays narrow. At (P=128) and (m=14), pairwise regret matrix has 229,376 entries, and rough (K=8) PAM / local-search pass about 1.72M absolute-difference operations. If you stored richer 3-feature action embedding, `P × P × m × 3` float32 = 2.625 MiB. Cheap. Bogus full-action (m=46) variant would be 753,664 entries and 8.625 MiB; memory still tolerable, but semantics not. On expensive side, if current 95%-mass reducer leaves 50–100 worlds and 14 discard-like legal actions, evaluation costs 700–1400 `eval_fn` calls; (K=8) representatives cost 112. That 6.25×–12.5× reduction.

My toy sanity check does show criterion not tautological top-mass pruning. In 11-world / 3-action toy with low-mass catastrophic cluster, full expected (Q) was ([-0.055,\ 0.797,\ 0.174]), so action 1 correct. (K=3) regret-medoids recovered medoids ([0,4,7]), cluster weights ([0.44,0.30,0.26]), and reproduced full expected (Q) / regret exactly with (\varepsilon_{\text{sig}} \approx 2.8\times10^{-17}). Equal-budget top-mass (K=3) kept worlds ([0,1,2]), chose action 0, and incurred full-reference regret (1.014). So lane not fake by construction. Still must beat top-mass on real Hydra states.

## 4. Tensor shapes / runtime payloads

Relevant live shapes already present. CT-SMC particles = `allocation[34][4]` plus `log_weight`; live encoder / model contract = `NUM_CHANNELS = 192`, `OBS_SIZE = 192 * 34`; search-side `delta_q` input plane = `[34]`; model outputs include `belief_fields [B,16,34]`, `mixture_weight_logits [B,4]`, `delta_q [B,46]`, and `safety_residual [B,46]`. `sample.rs` and tests enforce same `192 × 34` observation shape. ([GitHub][3])

For surviving runtime-only MVP, payloads should be:

* Existing inputs:

  * `allocation`: `u8[P, 34, 4]`
  * `log_weight`: `f64[P]`
  * `hand`: `u8[34]`
  * `legal_mask`: `bool[46]`
  * optional `public_danger`: `f32[34]` later, not required by strict MVP
  * `wall_remaining`: scalar
  * `has_threat`: scalar / bool

* Derived compression buffers:

  * `w`: `f32[P]`
  * `remaining_world`: `f32[P, 34]`, with `remaining_world[i,t] = sum_z allocation[i,t,z]`
  * `sig_action_tile`: `u8[m]`, where `m = |A_sig| <= 14`
  * `q_sig`: `f32[P, m]`
  * `regret`: `f32[P, m]` or `f32[P, |F|]` after frontier restriction
  * optional `dist`: `f32[P, P]`
  * `medoid_idx`: `usize[K]`
  * `assign`: `u16[P]`
  * `cluster_weight`: `f32[K]`
  * `rep_q_eval`: `f32[K, a_eval]`, where `a_eval = number of legal actions actually evaluated`
  * `q_out`: `f32[46]`
  * diagnostics: `epsilon_sig`, `gap_sig`, `fallback_reason`, `num_reps`

Minimal Rust-side interface:

```rust
pub struct ActionCompressionConfig {
    pub max_reps: usize,        // e.g. 8
    pub frontier_k: usize,      // e.g. 3
    pub tail_alpha: f32,        // e.g. 0.10
    pub split_once: bool,       // true
    pub require_discard_phase: bool, // true
}

pub struct ActionCompressionContext<'a> {
    pub hand: &'a [u8; 34],
    pub legal_mask: &'a [bool; 46],
    pub public_danger: Option<&'a [f32; 34]>, // future / optional
}

pub struct CompressionDiagnostics {
    pub medoid_indices: Vec<usize>,
    pub cluster_weight: Vec<f32>,
    pub epsilon_sig: f32,
    pub gap_sig: f32,
    pub fallback_reason: Option<FallbackReason>,
}
```

impl note: do **not** use `mean_allocation()` for posterior means in this path. In live code it averages particles uniformly by count, not by log-weight. Use normalized `log_weight` directly, or existing weighted helpers. ([GitHub][3])

Later training export different payload problem. Runtime search-side `delta_q` tile-type `[34]`, while model head `[46]`, `sample.rs` emits `delta_q_target: None`, and `HydraLoss` has no `delta_q_mask`. So later export would need either new `delta_q_mask [46]`, separate discard-only head, or explicit semantic remapping from tile-type to full action IDs. Future work, not part of MVP. ([GitHub][10])

## 5. Exact pseudocode

This pseudocode intentionally limited to survivor: discard-phase representative selection over CT-SMC particles, with current expensive endgame `eval_fn` left untouched. I am not including AFBS-node compression or train-side `delta_q` export pseudocode because those variants did not survive dependency pass. ([GitHub][4])

```text
function normalize_log_weights(logw[0..P-1]):
    maxw = max(logw)
    tmp[i] = exp(logw[i] - maxw)
    z = sum_i tmp[i]
    return w[i] = tmp[i] / z
```

```text
function unique_discard_tile_types(hand[34], legal_mask[46]):
    # Strict MVP scope:
    # - only states where meaningful decisions are discard-like
    # - collapse aka discard actions to base tile types
    tiles = empty ordered set
    for raw_action in legal discard-like actions:
        tile = collapse_action_to_tile_type(raw_action)   # aka 5m/5p/5s -> 4/13/22
        if hand[tile] > 0:
            insert tile into tiles
    return tiles
```

```text
function build_signature_scores(hand[34], particles[0..P-1], sig_tiles[0..m-1]):
    q_sig = zeros(P, m)
    for i in 0..P-1:
        remaining[34] = 0
        for t in 0..33:
            remaining[t] = particles[i].allocation[t][0]
                         + particles[i].allocation[t][1]
                         + particles[i].allocation[t][2]
                         + particles[i].allocation[t][3]
        hev = compute_hand_ev(hand, remaining)
        for u in 0..m-1:
            tile = sig_tiles[u]
            q_sig[i,u] = hev.expected_score[tile]
    return q_sig
```

```text
function worst_alpha_tail_indices(regret_col[0..P-1], w[0..P-1], alpha):
    # sort by descending regret
    idx = argsort_desc(regret_col)
    acc = 0
    tail = []
    for i in idx:
        tail.push(i)
        acc += w[i]
        if acc >= alpha:
            break
    return tail
```

```text
function weighted_farthest_seed(current_medoids, regret[P, f], w[P]):
    # choose point with largest weighted distance to nearest current medoid
    best_idx = 0
    best_score = -inf
    for i in 0..P-1:
        dmin = +inf
        for m in current_medoids:
            d = mean_abs(regret[i,:] - regret[m,:])
            dmin = min(dmin, d)
        score = w[i] * dmin
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
```

```text
function weighted_local_medoids(initial_medoids, regret[P, f], w[P], K, max_iters):
    medoids = dedupe(initial_medoids)
    while len(medoids) < K:
        medoids.push(weighted_farthest_seed(medoids, regret, w))

    for iter in 1..max_iters:
        changed = false

        # assign each particle to nearest medoid
        assign[P] = 0
        for i in 0..P-1:
            assign[i] = argmin_k mean_abs(regret[i,:] - regret[medoids[k],:])

        # update medoid of each cluster by weighted 1-median over cluster members
        for k in 0..K-1:
            members = { i : assign[i] == k }
            if members empty:
                continue
            best = medoids[k]
            best_cost = +inf
            for cand in members:
                cost = sum_{i in members} w[i] * mean_abs(regret[i,:] - regret[cand,:])
                if cost < best_cost:
                    best_cost = cost
                    best = cand
            if best != medoids[k]:
                medoids[k] = best
                changed = true

        if not changed:
            break

    return medoids
```

```text
function compress_particles_action_sufficient_mvp(ctx, particles, cfg):
    P = len(particles)
    if P == 0:
        return FALLBACK(EmptyParticles)
    if P <= cfg.max_reps:
        return FALLBACK(NoCompressionNeeded)

    if cfg.require_discard_phase and has_meaningful_nondiscard_choice(ctx.legal_mask):
        return FALLBACK(UnsupportedActionSet)

    sig_tiles = unique_discard_tile_types(ctx.hand, ctx.legal_mask)
    m = len(sig_tiles)
    if m <= 1:
        return FALLBACK(NoDecisionBranching)

    w = normalize_log_weights([p.log_weight for p in particles])
    q_sig = build_signature_scores(ctx.hand, particles, sig_tiles)

    # posterior-mean signature values
    q_bar[m] = 0
    for u in 0..m-1:
        q_bar[u] = sum_i w[i] * q_sig[i,u]

    frontier = top_k_indices(q_bar, min(cfg.frontier_k, m))
    f = len(frontier)

    regret[P, f] = 0
    for i in 0..P-1:
        best_i = max_u q_sig[i,u]
        for j in 0..f-1:
            u = frontier[j]
            regret[i,j] = best_i - q_sig[i,u]

    # tail seeds
    seeds = []
    for j in 0..f-1:
        tail = worst_alpha_tail_indices(regret[:,j], w, cfg.tail_alpha)
        # simplest seed: heaviest particle in the tail
        seed = argmax_{i in tail} w[i]
        seeds.push(seed)

    medoids = weighted_local_medoids(seeds, regret, w, cfg.max_reps, max_iters=8)

    # final assignment
    assign[P] = 0
    for i in 0..P-1:
        assign[i] = argmin_k mean_abs(regret[i,:] - regret[medoids[k],:])

    cluster_weight[K] = 0
    for i in 0..P-1:
        cluster_weight[assign[i]] += w[i]

    # signature certificate
    eps_sig = 0
    q_comp[f] = 0
    for j in 0..f-1:
        lhs = sum_i w[i] * regret[i,j]
        rhs = sum_k cluster_weight[k] * regret[medoids[k],j]
        eps_sig = max(eps_sig, abs(lhs - rhs))
        q_comp[j] = sum_k cluster_weight[k] * q_sig[medoids[k], frontier[j]]

    gap_sig = top1(q_comp) - top2(q_comp)

    if cfg.split_once and gap_sig <= 2 * eps_sig and K < P:
        # optional one-shot refinement:
        # split heaviest / highest-distortion cluster once, then recompute
        # omitted here for brevity but mechanically straightforward
        attempt_single_split(...)

    if gap_sig <= 2 * eps_sig:
        return FALLBACK(UncertainCompression)

    return SUCCESS(medoids, cluster_weight, eps_sig, gap_sig)
```

```text
function pimc_endgame_q_action_compressed(ctx, particles, legal_mask, eval_fn, cfg):
    comp = compress_particles_action_sufficient_mvp(ctx, particles, cfg)

    if comp is FALLBACK:
        return pimc_endgame_q_topk(particles, legal_mask, eval_fn, threshold=0.95)

    q_out[46] = 0
    for action in 0..45:
        if not legal_mask[action]:
            continue
        q_out[action] = 0
        for k in 0..K-1:
            idx = comp.medoids[k]
            q_out[action] += comp.cluster_weight[k] * eval_fn(particles[idx], action)
    return q_out
```

Two non-negotiable gates belong around this pseudocode. First: own-turn discard-phase only for MVP. Second: if method cannot produce confident signature certificate, it falls back to current top-mass path rather than pretending certainty.

## 6. Dependency closure table

* **CT-SMC weighted particle posterior.**
**Status:** already exists.
**Evidence:** exact particle allocations, log-weights, `weighted_mean_tile_count`, and `ess_ratio` live.
**Implication:** no new belief stack needed; compressor can operate directly on Hydra’s chosen search-grade posterior. ([GitHub][3])

* **Bridge seam from CT-SMC to hand-aware local features.**
**Status:** already exists.
**Evidence:** `extract_ct_smc_remaining_counts` and `compute_ct_smc_hand_ev` live.
**Implication:** per-world Hand-EV signature mechanically easy to derive once hand + particles available together. ([GitHub][10])

* **Expensive world-aggregating evaluator seam.**
**Status:** already exists.
**Evidence:** `pimc_endgame_q_topk` selects particles, normalizes weights, and averages `eval_fn(particle, action)` over legal actions.
**Implication:** replace selector, not evaluator. Right narrow seam. ([GitHub][4])

* **Hand input at selector seam.**
**Status:** missing but cheap to expose.
**Evidence:** `compute_hand_ev` needs `(hand, remaining)`; `solve_with_particles` takes only `(particles, legal_mask, eval_fn)`.
**Implication:** MVP needs small API extension before runtime wiring. ([GitHub][5])

* **Cheap full 46-action local geometry.**
**Status:** missing.
**Evidence:** current Hand-EV and bridge-side `delta_q` discard / tile-type centric, while model head 46-action.
**Implication:** honest MVP discard-phase only. Full-action compression does not survive. ([GitHub][5])

* **Threat-aware defensive side-channel for compression.**
**Status:** partial.
**Evidence:** safety / opponent-risk features exist in bridge / encoder; endgame exactification threat-gated; Hand-EV itself offense-side.
**Implication:** benchmark pure Hand-EV first, but production deployment in threat states likely needs safety threaded in. ([GitHub][10])

* **AFBS node-world semantics.**
**Status:** missing / placeholder.
**Evidence:** `particle_handle: Option<u32>` exists but initialized `None`; current AFBS features = root-exit policy and ponder priority from gap / risk / ESS.
**Implication:** full-tree AFBS compression out of scope and should stay out. ([GitHub][7])

* **Runtime hard-state gating hook.**
**Status:** already exists.
**Evidence:** inference uses top-2 policy gap; AFBS pondering already scores particle ESS.
**Implication:** later, `epsilon_sig` can become one more narrow gating signal instead of forcing new control framework. ([GitHub][11])

* **Posterior-quality diagnostics beyond ESS.**
**Status:** partial.
**Evidence:** FINAL defines Gate A/B, but live code surfaces ESS much more clearly than online Gate A/B metrics.
**Implication:** offline benchmark must stratify by posterior-quality proxies; if posterior failure dominates, compression not bottleneck. ([GitHub][2])

* **Train-side `delta_q` target path.**
**Status:** absent for current runtime semantics.
**Evidence:** `sample.rs` emits `delta_q_target: None`; `HydraLoss` uses dense unmasked MSE if present; runtime `delta_q` context 34 tile types while model head 46 actions.
**Implication:** later export future-only and requires semantic / masking work first. ([GitHub][8])

## 7. Offline falsification benchmark

Minimum honest benchmark = runtime-only, equal-budget endgame benchmark. Do **not** start in training loader, and do **not** use singleton reference. Reference must be full-posterior `pimc_endgame_q` over all available particles using same expensive `eval_fn` compressed method will use. That stays inside Hydra’s actual seam and respects no-training-first constraint. ([GitHub][4])

Use sidecar dump of at least 10k states satisfying:

* wall remaining `<= 10`
* threat flag true
* current player’s decision discard-phase only for MVP
* CT-SMC snapshot available
* current hand available
* legal mask available
* ESS ratio logged
* actor policy top-2 gap logged
* optional safety cache logged if available

Compare these methods at **equal expensive-evaluator budget**:

1. **Full reference:** all particles, same `eval_fn`.
2. **Top-mass-K:** highest-weight (K) particles, (K \in {4,8,12,16}).
3. **Action-sufficient-K:** representative selector above, same (K).
4. **Context only:** current 95%-mass top-mass variable-(K) operating point, reported for context but not fairness baseline.

For each state and each (K):

* compute (Q_{\text{ref}}), (Q_{\text{mass},K}), (Q_{\text{as},K})
* let (a_{\text{ref}} = \arg\max Q_{\text{ref}})
* measure **reference regret**
[
\text{regret}*{\text{method}} =
Q*{\text{ref}}(a_{\text{ref}}) - Q_{\text{ref}}(a_{\text{method}})
]
* record `eval_fn_calls = K * num_legal_actions` for compressed methods
* record wall-clock latency including clustering overhead
* record fallback rate
* record certificate calibration on slice with (\Delta_{\text{sig}} > 2\varepsilon_{\text{sig}})

Benchmark primary metrics:

* mean reference regret
* p95 reference regret
* p99 reference regret
* action agreement vs full reference
* evaluator calls
* end-to-end latency
* fallback rate
* certified-slice mismatch rate

**Minimum go / no-go rule** harsh on purpose: if no (K \in {4,8,12,16}) where action-sufficient compression beats top-mass-(K) on **both mean regret and p95 regret** under paired evaluation, kill method. If p99 regret worsens materially, kill it even if mean regret improves. This prompt’s required falsification bar.

I would also stratify benchmark by:

* ESS bucket
* top-2 policy-gap bucket
* wall bucket
* optional safety / risk bucket

That answers “posterior quality vs compression quality” directly. If both top-mass and action-sufficient compression fail mainly in low-quality-posterior slices, posterior quality bottleneck, not compression.

Concrete budget claim easy to audit. If current 95%-mass reducer leaves 50, 64, or 100 worlds and 14 discard-like legal actions, expensive evaluator cost = 700, 896, or 1400 calls. (K=8) representatives cost 112 calls. Reduction = 6.25×, 8×, and 12.5× respectively. Real seam only if regret improves or at least stays better than top-mass at same call counts.

## 8. Failure modes and kill criteria

**Does this only preserve noisy evaluator mistakes more efficiently?**
It can. Main red-team risk. Survivor mitigates it only partially: final action values still come from existing expensive `eval_fn`, not cheap Hand-EV signature, so method not directly replacing evaluator. But selector still chooses which worlds get evaluated. If signature misaligned, wrong representatives get chosen more efficiently. That exactly why training-first distillation from compressed worlds dead for now, and why offline reference benchmark non-negotiable.

**Is posterior quality actual bottleneck, not compression quality?**
Possibly. FINAL explicitly defines posterior-quality gates beyond ESS, and live code clearly exposes ESS. If benchmark shows both top-mass and action-sufficient methods fail in same low-ESS / low-posterior-quality slices, then compression secondary and right next move = posterior improvement, not better clustering. ([GitHub][2])

**Does this fail if Hand-EV realism not improved first?**
For broad deployment, yes. For narrow runtime-only benchmark, not necessarily. Benchmark can still run now because final action values come from existing expensive evaluator, and question whether Hand-EV good enough to choose representatives. But production go-live in threat states without defensive side-channel much less convincing. If pure Hand-EV signature loses heavily on threat-tail slices, that not reason to expand method; reason to either thread safety into selector or kill lane.

**Does result reduce to top-mass pruning with fancier vocabulary?**
It does if any following happen:

* medoids almost highest-weight particles,
* tail seeds do not matter,
* method does not beat top-mass-(K) at equal (K),
* or only winning variants secretly use too much expensive evaluator work before selection.

If that happens, kill it. Do not rename pruning as novelty.

There also Hydra-specific multiplayer caveat. This method preserves **current-player scalar action ranking** under Hydra’s existing evaluator API. It not multiplayer equilibrium abstraction and does not solve 4-player general-sum reasoning by itself. Acceptable for current seam because Hydra’s endgame helper already written around current-player scalar `Q(action)`, but kills any stronger game-theoretic claim. ([GitHub][4])

Ideas that did **not** survive:

* **Full AFBS / node-level world compression.** Failed because node-world semantics still placeholder-only.
* **Full 46-action action-sufficient compression.** Failed because current cheap geometry discard / tile-type only.
* **Training-first compressed `delta_q` export.** Failed because `delta_q_target` still absent, semantics 34-vs-46 mismatched, and no `delta_q_mask`.
* **Pure offense-only production deployment in threat states.** Failed as deployment rec because Hand-EV not yet defensive enough; survives only as benchmark ablation or after adding safety side-channel.

My explicit kill criteria:

1. No (K \in {4,8,12,16}) beats top-mass-(K) on both mean and p95 reference regret.
2. p99 regret worsens materially even when mean improves.
3. Certified slice ((\Delta_{\text{sig}} > 2\varepsilon_{\text{sig}})) still has high mismatch rate, so certificate useless.
4. Fallback rate so high method almost never applies on target slice.
5. Gains vanish once clustering overhead included.
6. Best-performing variant requires broad AFBS or train-side changes prompt forbids.

## 9. Final recommendation: worth it now, later, or not worth it

My final rec: **worth it now only as narrow, runtime-only falsification project**. Not broad AFBS program. Not new belief stack. Not training-first `delta_q` export. Surviving MVP = discard-phase representative selector over CT-SMC particles, attached to current endgame/top-mass seam after small hand-aware API extension, and judged only by whether it beats equal-budget top-mass pruning on full-reference decision regret. That consistent with reconciliation’s sequencing and current repo surfaces. ([GitHub][1])

Broader investment = **later, conditional, and only if MVP wins**. If equal-budget benchmark shows real regret-vs-calls frontier improvement, then action-sufficient compression becomes legitimate second-wave Hydra path: first wire runtime helper, then thread cheap safety side-channel, then revisit root-only discard-phase version, and only much later reconsider masked / semantically aligned `delta_q` export. If benchmark does not beat top-mass, kill lane completely rather than renaming top-mass pruning. That narrowest rec evidence supports after stricter repo pass.

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/ct_smc.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/endgame.rs"
[5]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/hand_ev.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/hand_ev.rs"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/data/mjai_loader.rs"
[7]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[8]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[9]: https://research.google/pubs/value-directed-compression-of-pomdps/ "https://research.google/pubs/value-directed-compression-of-pomdps/"
[10]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/bridge.rs"
[11]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/inference.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/inference.rs"
]]>
</answer_text>
</answer_section>
</combined_run_record>