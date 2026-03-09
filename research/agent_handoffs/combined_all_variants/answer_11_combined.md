<combined_run_record run_id="answer_11" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 11 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_11_VALIDATE_ACTION_SUFFICIENT_WORLD_COMPRESSION.md">
  <![CDATA[# Hydra prompt — validate action-sufficient CT-SMC world compression

Primary source material lives in the raw GitHub links below.

## Critical directive

Read the core Hydra docs holistically before proposing any compression scheme. Do not jump straight from generic POMDP compression papers to Hydra recommendations.

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

Relevant prior variant writeups and prompt references:
- `research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/006_followup_debc_ar.md
- `research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/008_variant_agent_8new1.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_pack_006.md
- `research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/reference_prompt_template_004_outside_the_box_but_grounded.md

You are validating whether action-sufficient world compression is a real long-run Hydra path.

Focus on:
- compressing CT-SMC worlds by decision relevance, not probability mass
- using current Hand-EV / endgame evaluators as local regret geometry
- whether this is a cheap and real seam before deeper AFBS expansion

Broad exploration has already been done in `research/agent_handoffs/combined_all_variants/`. Do not redo that broad work. Start from the prior DEBC-AR and 8-series materials and only search enough to validate, falsify, or tighten this specific compression lane.

<output_contract>
- Return exactly the requested sections, in the requested order.
- Be as detailed and explicit as necessary; do not optimize for brevity.
- Return a full technical treatment, not a compressed memo.
- A short answer is usually a failure mode for this prompt.
</output_contract>

<calculation_validation_rules>
- Use Python in bash for compression-ratio arithmetic, evaluator-call accounting, and toy regret-clustering sanity checks.
- Do not leave claims like “8 worlds replaces 50-100 worlds” uncomputed.
</calculation_validation_rules>

<tool_persistence_rules>
- Do not restart a broad belief-compression survey.
- New retrieval should only validate, falsify, or sharpen action-sufficient compression for Hydra's actual runtime seam.
</tool_persistence_rules>

<dependency_checks>
- Verify what CT-SMC exposes today, what Hand-EV/endgame evaluators exist, and whether current runtime already supports the relevant insertion points.
- Verify whether later `delta_q` export is real or still future-only.
</dependency_checks>

<self_red_team_rules>
- Ask explicitly:
  - Does this only preserve noisy evaluator mistakes more efficiently?
  - Is posterior quality the actual bottleneck, not compression quality?
  - Does this fail if Hand-EV realism is not improved first?
  - Does the result reduce to top-mass particle pruning with fancier vocabulary?
</self_red_team_rules>

<minimum_falsification_rules>
- Define the minimum offline benchmark that compares top-mass particle pruning against action-sufficient compression on decision regret at equal evaluator budget.
- Reject the method if it cannot beat simple top-mass pruning offline.
</minimum_falsification_rules>

<anti_survey_rules>
- Do not drift into generic POMDP compression literature review.
- Stay inside CT-SMC world compression for Hydra's current Hand-EV / endgame seam.
</anti_survey_rules>

## What to do
1. Reconstruct Hydra's current CT-SMC / Hand-EV / endgame posture.
2. Validate whether action-sufficient compression is timely under reconciliation.
3. Write down the exact objective, compression rule, tensor/interfaces, and pseudocode.
4. Separate MVP runtime-only compression from later training export ideas.
5. Decide if this is a serious second-wave investment or still too early.

## Deliverables
1. Hydra posture reconstruction for CT-SMC / Hand-EV / endgame
2. Existing repo surfaces and blockers
3. Exact mathematical formulation of the compression criterion
4. Tensor shapes / runtime payloads
5. Exact pseudocode
6. Dependency closure table
7. Offline falsification benchmark
8. Failure modes and kill criteria
9. Final recommendation: worth it now, later, or not worth it

## Hard constraints
- no broad AFBS rewrite
- no new belief stack
- no fake novelty by renaming belief compression
- no training-first recommendation if runtime-only falsification has not been proven
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_11.md">
  <![CDATA[## 1. Hydra posture reconstruction for CT-SMC / Hand-EV / endgame

Hydra’s reconciled doctrine is not “search first.” The active path is supervision-first, with Hand-EV realism ahead of deeper AFBS semantics, AFBS kept specialist / hard-state gated, and a unified belief story in which Mixture-SIB is the amortized belief object and CT-SMC is the search-grade posterior. `OPPONENT_MODELING.md` explicitly says to defer to `HYDRA_FINAL.md` and `HYDRA_RECONCILIATION.md` when active implementation shape differs, and to treat heavier posterior / deception extensions as reserve-shelf unless promoted by the reconciled doctrine. ([GitHub][1])

`HYDRA_FINAL.md` makes CT-SMC’s intended role precise: exact contingency-table sampling replaces the generic particle proposal; resampling happens when ESS falls below `0.4P`; posterior quality is supposed to be judged by Gate A (posterior log-likelihood) and Gate B (pairwise MI calibration); and late-game exactification is Pure PIMC over CT-SMC particles with a 95% top-mass reducer, typically leaving about 50–100 worlds in the wall-≤10 threat slice. That is already a probability-mass world compression policy, just not a decision-aware one. ([GitHub][2])

The live code mostly matches that posture. `ct_smc.rs` exposes exact DP sampling, default `num_particles = 128`, `ess_threshold = 0.4`, particle log-weights, `weighted_mean_tile_count`, and `ess_ratio`. The bridge already compresses CT-SMC into belief-weighted remaining counts and then runs `compute_hand_ev`, while AFBS-derived `delta_q` features are only populated for `0..NUM_TILE_TYPES` tile-type discard actions. `hand_ev.rs` defines per-discard tenpai / win / expected-score / ukeire features from `(hand, remaining)`. `endgame.rs` activates on `wall_remaining <= max_wall && has_threat`, and `pimc_endgame_q_topk` simply chooses selected particles, normalizes their weights, and averages the expensive `eval_fn(particle, action)` over legal actions. ([GitHub][3])

So Hydra already has two probability-oriented compression moves in runtime reality: first-moment CT-SMC→Hand-EV in the bridge, and top-mass CT-SMC→PIMC in endgame. The missing object is not a new belief stack. It is a decision-relevant representative-world selector that can replace one of those existing compressions only if it preserves action quality better than weight-only pruning at the same expensive-evaluator budget. That is exactly the lane the prior DEBC-AR / 8-series writeups were circling, but reconciliation forces it to be much narrower than a general belief-compression program. (hydra-core/src/bridge.rs:263-299; hydra-core/src/endgame.rs:80-87,90-184; research/design/HYDRA_RECONCILIATION.md:149-160,221-243)

## 2. Existing repo surfaces and blockers

There are real surfaces today. CT-SMC gives a weighted posterior over valid hidden worlds, not just marginals. The bridge already knows how to turn CT-SMC into hand-aware local features. The endgame helper already has a “select worlds, then aggregate expensive per-world action values” structure. AFBS / inference already expose hard-state signals such as top-2 policy gap and particle ESS that could later host a compression-escalation rule. Those are genuine seams; this is not a fantasy “invent a new belief system” proposal. (hydra-core/src/ct_smc.rs:228-258; hydra-core/src/bridge.rs:263-299; hydra-core/src/endgame.rs:80-87,90-184; hydra-core/src/afbs.rs:472-507)

The first blocker is interface shape. `solve_with_particles` in `endgame.rs` gets `(particles, legal_mask, eval_fn)`, but `compute_hand_ev` needs `(hand, remaining)`. So an action-sufficient selector based on current Hand-EV cannot be inserted as a literal one-line swap. At minimum, the selector needs the current hand threaded into the endgame seam. A stronger production version would also want a cheap safety-side vector because endgame activation is threat-gated. In the surfaced files, that hand/safety context is available around bridge / encoder code, but not yet in the endgame helper API. (hydra-core/src/endgame.rs:80-87; hydra-core/src/hand_ev.rs:253-309; hydra-core/src/bridge.rs:251-299)

The second blocker is action-space realism. The cheap local geometry that Hydra exposes today is discard / tile-type centric, not full-action. `compute_hand_ev` is indexed by 34 tile types; the bridge fills `delta_q` only for `0..NUM_TILE_TYPES`; the encoder’s search `delta_q` plane is `[34]`; but the model’s `delta_q` head is `[46]`. That kills any honest claim that Hydra already has a cheap current local scorer for the full 46-action space, especially call-phase and declaration actions. The surviving scope is discard-phase world compression over tile-type discard classes, not full-action compression. (hydra-core/src/hand_ev.rs:6-10,253-309; hydra-core/src/bridge.rs:343-355; hydra-train/src/model.rs:22,98,241,269,297)

There is also an action-identity mismatch around red fives. Current target-building code already collapses aka 5m / 5p / 5s to base tile types for safety residual targets. That is another sign that the practical current signature space is tile-type discard classes, not raw action IDs. For this lane, that is fine; it just means the exact MVP has to be explicit about collapse-to-tile-type semantics instead of pretending to preserve a full raw-action ontology it does not currently have. (hydra-train/src/data/mjai_loader.rs:283-299)

The third blocker is evaluator realism. `compute_hand_ev` is offense-oriented: shanten improvement, ukeire, multi-horizon tenpai / win probabilities, and a conditional score estimate. Defensive information exists elsewhere (`danger_from_particles`, safety features in bridge / encoder), not in the Hand-EV scalar itself. Because the endgame helper is threat-gated, a pure Hand-EV-only production compressor would be using an offense-biased signature exactly where defensive tail misses matter most. This is where reconciliation matters: broad deployment should wait for better Hand-EV realism or a threaded safety side-channel. (hydra-core/src/hand_ev.rs:24-43,132-141,253-309; hydra-core/src/bridge.rs:301-360; research/design/HYDRA_RECONCILIATION.md:135-146,278-283)

The fourth blocker kills the full-tree version. AFBS has a `particle_handle`, but it is an `Option<u32>` placeholder initialized as `None`; the live hard-state machinery is root-exit policy, ponder reuse, and priority from top-2 gap, risk score, and particle ESS. That is enough to host a future escalation signal. It is not enough to justify node-level world compression or any broad AFBS rewrite. Under your hard constraints, that version does not survive. (hydra-core/src/afbs.rs:106-114,265-314,398-459,472-507)

The fifth blocker kills training-first versions. `sample.rs` still writes `opponent_hand_type_target: None` and `delta_q_target: None`. `HydraLoss` applies dense unmasked MSE to `delta_q` if a target ever appears, while only `safety_residual` has an action mask. Combined with the runtime `delta_q` payload being `[34]` tile-type discard-level while the model head is `[46]`, later `delta_q` export is still future-only. It is not a cheap current extension. (hydra-train/src/data/sample.rs:178-181,210-213; hydra-train/src/training/losses.rs:247-262; hydra-train/src/model.rs:22,98,241,269,297)

After this stricter pass, four ideas are dead for now: full-tree AFBS compression, full 46-action compression, training-first compressed `delta_q` export, and any “compression” that is just top-mass pruning or first-moment averaging under a new name. The sole survivor is a runtime-only, discard-phase representative selector over CT-SMC particles, benchmarked against top-mass at equal expensive-evaluator budget.

## 3. Exact mathematical formulation of the compression criterion

The conceptual family here is legitimate: decision-aware / value-directed compression rather than raw belief approximation. But Hydra’s actual code narrows it sharply. The exact survivor is a discard-phase representative-world selector over CT-SMC particles, where the cheap geometry comes from current per-world Hand-EV and the final decision still comes from the existing expensive endgame evaluator. ([Google Research][9])

Let the CT-SMC posterior at the current state be particles ({(X_i,\ell_i)}_{i=1}^P), where (X_i \in {0,\dots,4}^{34\times 4}) is the hidden-world contingency table and (\ell_i) is its log-weight. Normalize:

[
w_i = \frac{\exp(\ell_i - \ell_{\max})}{\sum_{j=1}^P \exp(\ell_j - \ell_{\max})}, \qquad \sum_i w_i = 1.
]

For each particle, define the world-specific unseen-tile count vector used by Hand-EV:

[
r_i[t] = \sum_{z=0}^{3} X_i[t,z], \qquad t \in {0,\dots,33}.
]

This exactly mirrors what the bridge already does in expectation when it sums CT-SMC hidden columns into remaining counts; the difference is that we do it per world instead of after first-moment averaging. (hydra-core/src/bridge.rs:263-283)

Define the signature action set (A_{\text{sig}}(s)) as the set of unique legal discard tile types in the current hand, after collapsing aka discard actions to base tile types. In the strict surviving MVP, this is the only signature action set that is honest, because current cheap geometry is tile-type discard indexed. Let (|A_{\text{sig}}| = m \le 14).

For each particle (i), compute current Hand-EV features:

[
f_i = \texttt{compute_hand_ev}(h, r_i),
]

where (h \in {0,\dots,4}^{34}) is the player’s current hand. The exact MVP scalar signature is deliberately simple:

[
q_i^{\text{sig}}(a) = f_i.\texttt{expected_score}[a], \qquad a \in A_{\text{sig}}.
]

I am not baking a defensive penalty into the exact MVP formula because the current runtime does not expose a clean endgame-side safety scalar yet. That omission is a production blocker, not a reason to lie about the math. A later risk-aware variant can replace the scalar with
[
q_i^{\text{sig}}(a)= f_i.\texttt{expected_score}[a]-\lambda_{\text{risk}} d_{\text{pub}}(a),
]
but that is not the proven current MVP.

Now define posterior-mean signature values

[
\bar q^{\text{sig}}(a) = \sum_{i=1}^{P} w_i q_i^{\text{sig}}(a),
]

and let the signature frontier (F \subseteq A_{\text{sig}}) be the top (k_F) discard tile types by (\bar q^{\text{sig}}), with strict MVP default (k_F = 3).

Define per-world regret vectors on that frontier:

[
R_i(a) = \max_{b \in A_{\text{sig}}} q_i^{\text{sig}}(b) - q_i^{\text{sig}}(a), \qquad a \in F.
]

The decision-aware world distance is then

[
d(i,j) = \frac{1}{|F|} \sum_{a \in F} \left| R_i(a) - R_j(a) \right|.
]

This is the core point: posterior mass still weights the objective, but geometry is action distortion, not raw hidden-world distance. A top-mass baseline sorts by (w_i) alone. This objective clusters worlds by how much they disagree about the discard frontier.

Tail protection is explicit. For each frontier action (a \in F), define (T_\alpha(a)) as the smallest set of highest-regret particles whose total posterior mass is at least (\alpha), with strict MVP default (\alpha = 0.10). The medoid initialization must include at least one seed from the union of these tail sets before ordinary weighted k-medoids proceeds. This is what prevents the method from degenerating into “top-mass with fancier vocabulary.”

The weighted medoid problem is

[
\min_{M,\phi} J(M,\phi)
\quad\text{where}\quad
J(M,\phi)=\sum_{i=1}^{P} w_i, d(i,\phi(i)),
]
subject to (|M|=K) and (\phi(i)\in M).

If (M={m_1,\dots,m_K}) are the selected medoids, the compressed cluster weights are

[
W_k = \sum_{i:\phi(i)=m_k} w_i.
]

The final action values are **not** taken from the cheap signature. They come from Hydra’s existing expensive evaluator on medoids only:

[
\hat Q_K(u)=\sum_{k=1}^{K} W_k , Q^{\text{eval}}(X_{m_k},u),
\qquad u \in A_{\text{eval}}(s),
]
where (A_{\text{eval}}(s)) is the actual legal-action set under the 46-action mask, and (Q^{\text{eval}}(X,u)) is the current `eval_fn(&Particle, action)` used by the endgame helper.

The cheap compression certificate is signature-level, not truth-level:

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

Let (\Delta_{\text{sig}}) be the top-2 gap in the compressed signature frontier:
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
then the signature winner is stable under this compression. But this only certifies that the clustering preserved the cheap local geometry. It does **not** certify true endgame optimality, because (q^{\text{sig}}) is only a proxy for (Q^{\text{eval}}). That is why the real go / no-go criterion must be offline full-evaluator regret, not the certificate itself.

The arithmetic only works if the scope stays narrow. At (P=128) and (m=14), the pairwise regret matrix has 229,376 entries, and a rough (K=8) PAM / local-search pass is about 1.72M absolute-difference operations. If you stored a richer 3-feature action embedding, `P × P × m × 3` float32 is 2.625 MiB. That is cheap. A bogus full-action (m=46) variant would be 753,664 entries and 8.625 MiB; memory is still tolerable, but semantics are not. On the expensive side, if the current 95%-mass reducer leaves 50–100 worlds and there are 14 discard-like legal actions, evaluation costs 700–1400 `eval_fn` calls; (K=8) representatives cost 112. That is a 6.25×–12.5× reduction.

My toy sanity check does show that the criterion is not tautological top-mass pruning. In an 11-world / 3-action toy with a low-mass catastrophic cluster, the full expected (Q) was ([-0.055,\ 0.797,\ 0.174]), so action 1 was correct. (K=3) regret-medoids recovered medoids ([0,4,7]), cluster weights ([0.44,0.30,0.26]), and reproduced the full expected (Q) / regret exactly with (\varepsilon_{\text{sig}} \approx 2.8\times10^{-17}). Equal-budget top-mass (K=3) kept worlds ([0,1,2]), chose action 0, and incurred full-reference regret (1.014). So the lane is not fake by construction. It still has to beat top-mass on real Hydra states.

## 4. Tensor shapes / runtime payloads

The relevant live shapes are already present. CT-SMC particles are `allocation[34][4]` plus `log_weight`; the live encoder / model contract is `NUM_CHANNELS = 192`, `OBS_SIZE = 192 * 34`; the search-side `delta_q` input plane is `[34]`; and the model outputs include `belief_fields [B,16,34]`, `mixture_weight_logits [B,4]`, `delta_q [B,46]`, and `safety_residual [B,46]`. `sample.rs` and tests enforce the same `192 × 34` observation shape. ([GitHub][3])

For the surviving runtime-only MVP, the payloads should be:

* Existing inputs:

  * `allocation`: `u8[P, 34, 4]`
  * `log_weight`: `f64[P]`
  * `hand`: `u8[34]`
  * `legal_mask`: `bool[46]`
  * optional `public_danger`: `f32[34]` later, not required by the strict MVP
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

A minimal Rust-side interface is:

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

Implementation note: do **not** use `mean_allocation()` for posterior means in this path. In the live code it averages particles uniformly by count, not by log-weight. Use normalized `log_weight` directly, or the existing weighted helpers. ([GitHub][3])

Later training export is a different payload problem. Runtime search-side `delta_q` is currently tile-type `[34]`, while the model head is `[46]`, `sample.rs` emits `delta_q_target: None`, and `HydraLoss` has no `delta_q_mask`. So later export would need either a new `delta_q_mask [46]`, a separate discard-only head, or an explicit semantic remapping from tile-type to full action IDs. That is future work, not part of the MVP. ([GitHub][10])

## 5. Exact pseudocode

This pseudocode is intentionally limited to the survivor: discard-phase representative selection over CT-SMC particles, with the current expensive endgame `eval_fn` left untouched. I am not including AFBS-node compression or train-side `delta_q` export pseudocode because those variants did not survive the dependency pass. ([GitHub][4])

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

Two non-negotiable gates belong around this pseudocode. First: own-turn discard-phase only for the MVP. Second: if the method cannot produce a confident signature certificate, it falls back to the current top-mass path rather than pretending certainty.

## 6. Dependency closure table

* **CT-SMC weighted particle posterior.**
  **Status:** already exists.
  **Evidence:** exact particle allocations, log-weights, `weighted_mean_tile_count`, and `ess_ratio` are live.
  **Implication:** no new belief stack is needed; the compressor can operate directly on Hydra’s chosen search-grade posterior. ([GitHub][3])

* **Bridge seam from CT-SMC to hand-aware local features.**
  **Status:** already exists.
  **Evidence:** `extract_ct_smc_remaining_counts` and `compute_ct_smc_hand_ev` are live.
  **Implication:** a per-world Hand-EV signature is mechanically easy to derive once hand and particles are available together. ([GitHub][10])

* **Expensive world-aggregating evaluator seam.**
  **Status:** already exists.
  **Evidence:** `pimc_endgame_q_topk` selects particles, normalizes their weights, and averages `eval_fn(particle, action)` over legal actions.
  **Implication:** replace the selector, not the evaluator. That is the right narrow seam. ([GitHub][4])

* **Hand input at the selector seam.**
  **Status:** missing but cheap to expose.
  **Evidence:** `compute_hand_ev` needs `(hand, remaining)`; `solve_with_particles` currently takes only `(particles, legal_mask, eval_fn)`.
  **Implication:** the MVP needs a small API extension before runtime wiring. ([GitHub][5])

* **Cheap full 46-action local geometry.**
  **Status:** missing.
  **Evidence:** current Hand-EV and bridge-side `delta_q` are discard / tile-type centric, while the model head is 46-action.
  **Implication:** the honest MVP is discard-phase only. Full-action compression does not survive. ([GitHub][5])

* **Threat-aware defensive side-channel for compression.**
  **Status:** partial.
  **Evidence:** safety / opponent-risk features exist in bridge / encoder; endgame exactification is threat-gated; Hand-EV itself is offense-side.
  **Implication:** benchmark pure Hand-EV first, but production deployment in threat states likely needs safety threaded in. ([GitHub][10])

* **AFBS node-world semantics.**
  **Status:** missing / placeholder.
  **Evidence:** `particle_handle: Option<u32>` exists but is initialized `None`; current AFBS features are root-exit policy and ponder priority from gap / risk / ESS.
  **Implication:** full-tree AFBS compression is out of scope and should stay out. ([GitHub][7])

* **Runtime hard-state gating hook.**
  **Status:** already exists.
  **Evidence:** inference uses top-2 policy gap; AFBS pondering already scores particle ESS.
  **Implication:** later, `epsilon_sig` can become one more narrow gating signal instead of forcing a new control framework. ([GitHub][11])

* **Posterior-quality diagnostics beyond ESS.**
  **Status:** partial.
  **Evidence:** FINAL defines Gate A/B, but live code surfaces ESS much more clearly than online Gate A/B metrics.
  **Implication:** offline benchmark must stratify by posterior-quality proxies; if posterior failure dominates, compression is not the bottleneck. ([GitHub][2])

* **Train-side `delta_q` target path.**
  **Status:** absent for current runtime semantics.
  **Evidence:** `sample.rs` emits `delta_q_target: None`; `HydraLoss` uses dense unmasked MSE if present; runtime `delta_q` context is 34 tile types while the model head is 46 actions.
  **Implication:** later export is future-only and requires semantic / masking work first. ([GitHub][8])

## 7. Offline falsification benchmark

The minimum honest benchmark is a runtime-only, equal-budget endgame benchmark. Do **not** start in the training loader, and do **not** use a singleton reference. The reference must be full-posterior `pimc_endgame_q` over all available particles using the same expensive `eval_fn` that the compressed method will use. That stays inside Hydra’s actual seam and respects the no-training-first constraint. ([GitHub][4])

Use a sidecar dump of at least 10k states satisfying:

* wall remaining `<= 10`
* threat flag true
* current player’s decision is discard-phase only for the MVP
* CT-SMC snapshot available
* current hand available
* legal mask available
* ESS ratio logged
* actor policy top-2 gap logged
* optional safety cache logged if available

Compare these methods at **equal expensive-evaluator budget**:

1. **Full reference:** all particles, same `eval_fn`.
2. **Top-mass-K:** highest-weight (K) particles, (K \in {4,8,12,16}).
3. **Action-sufficient-K:** the representative selector above, same (K).
4. **Context only:** current 95%-mass top-mass variable-(K) operating point, reported for context but not as the fairness baseline.

For each state and each (K):

* compute (Q_{\text{ref}}), (Q_{\text{mass},K}), (Q_{\text{as},K})
* let (a_{\text{ref}} = \arg\max Q_{\text{ref}})
* measure **reference regret**
  [
  \text{regret}*{\text{method}} =
  Q*{\text{ref}}(a_{\text{ref}}) - Q_{\text{ref}}(a_{\text{method}})
  ]
* record `eval_fn_calls = K * num_legal_actions` for the compressed methods
* record wall-clock latency including clustering overhead
* record fallback rate
* record certificate calibration on the slice with (\Delta_{\text{sig}} > 2\varepsilon_{\text{sig}})

The benchmark’s primary metrics are:

* mean reference regret
* p95 reference regret
* p99 reference regret
* action agreement vs full reference
* evaluator calls
* end-to-end latency
* fallback rate
* certified-slice mismatch rate

The **minimum go / no-go rule** is harsh on purpose: if there is no (K \in {4,8,12,16}) for which action-sufficient compression beats top-mass-(K) on **both mean regret and p95 regret** under paired evaluation, kill the method. If p99 regret worsens materially, kill it even if mean regret improves. This is the prompt’s required falsification bar.

I would also stratify the benchmark by:

* ESS bucket
* top-2 policy-gap bucket
* wall bucket
* optional safety / risk bucket

That answers the “posterior quality vs compression quality” question directly. If both top-mass and action-sufficient compression fail mainly in low-quality-posterior slices, then posterior quality is the bottleneck, not compression.

The concrete budget claim is easy to audit. If the current 95%-mass reducer leaves 50, 64, or 100 worlds and there are 14 discard-like legal actions, the expensive evaluator cost is 700, 896, or 1400 calls. (K=8) representatives cost 112 calls. The reduction is 6.25×, 8×, and 12.5× respectively. That is a real seam only if regret improves or at least stays better than top-mass at those same call counts.

## 8. Failure modes and kill criteria

**Does this only preserve noisy evaluator mistakes more efficiently?**
It can. That is the main red-team risk. The survivor mitigates it only partially: the final action values still come from the existing expensive `eval_fn`, not from the cheap Hand-EV signature, so the method is not directly replacing the evaluator. But the selector still chooses which worlds get evaluated. If the signature is misaligned, the wrong representatives get chosen more efficiently. That is exactly why training-first distillation from compressed worlds is dead for now, and why the offline reference benchmark is non-negotiable.

**Is posterior quality the actual bottleneck, not compression quality?**
Possibly. FINAL explicitly defines posterior-quality gates beyond ESS, and the live code clearly exposes ESS. If the benchmark shows that both top-mass and action-sufficient methods fail in the same low-ESS / low-posterior-quality slices, then compression is secondary and the right next move is posterior improvement, not better clustering. ([GitHub][2])

**Does this fail if Hand-EV realism is not improved first?**
For broad deployment, yes. For the narrow runtime-only benchmark, not necessarily. The benchmark can still be run now because the final action values come from the existing expensive evaluator, and the question is whether Hand-EV is good enough to choose representatives. But a production go-live in threat states without a defensive side-channel is much less convincing. If the pure Hand-EV signature loses heavily on threat-tail slices, that is not a reason to expand the method; it is a reason to either thread safety into the selector or kill the lane.

**Does the result reduce to top-mass pruning with fancier vocabulary?**
It does if any of the following happen:

* medoids are almost always just the highest-weight particles,
* tail seeds do not matter,
* the method does not beat top-mass-(K) at equal (K),
* or the only winning variants secretly use too much expensive evaluator work before selection.

If that happens, kill it. Do not rename pruning as novelty.

There is also a Hydra-specific multiplayer caveat. This method preserves a **current-player scalar action ranking** under Hydra’s existing evaluator API. It is not a multiplayer equilibrium abstraction and does not solve 4-player general-sum reasoning by itself. That is acceptable for the current seam because Hydra’s endgame helper is already written around a current-player scalar `Q(action)`, but it kills any stronger game-theoretic claim. ([GitHub][4])

The ideas that did **not** survive are:

* **Full AFBS / node-level world compression.** Failed because node-world semantics are still placeholder-only.
* **Full 46-action action-sufficient compression.** Failed because the current cheap geometry is discard / tile-type only.
* **Training-first compressed `delta_q` export.** Failed because `delta_q_target` is still absent, semantics are 34-vs-46 mismatched, and there is no `delta_q_mask`.
* **Pure offense-only production deployment in threat states.** Failed as a deployment recommendation because Hand-EV is not yet defensive enough; it survives only as a benchmark ablation or after adding a safety side-channel.

My explicit kill criteria are:

1. No (K \in {4,8,12,16}) beats top-mass-(K) on both mean and p95 reference regret.
2. p99 regret worsens materially even when mean improves.
3. Certified slice ((\Delta_{\text{sig}} > 2\varepsilon_{\text{sig}})) still has a high mismatch rate, so the certificate is useless.
4. Fallback rate is so high that the method almost never applies on the target slice.
5. Gains vanish once clustering overhead is included.
6. The best-performing variant requires the broad AFBS or train-side changes that the prompt forbids.

## 9. Final recommendation: worth it now, later, or not worth it

My final recommendation is: **worth it now only as a very narrow, runtime-only falsification project**. Not as a broad AFBS program. Not as a new belief stack. Not as a training-first `delta_q` export. The surviving MVP is a discard-phase representative selector over CT-SMC particles, attached to the current endgame/top-mass seam after a small hand-aware API extension, and judged only by whether it beats equal-budget top-mass pruning on full-reference decision regret. That is consistent with reconciliation’s sequencing and with the current repo surfaces. ([GitHub][1])

Broader investment is **later, conditional, and only if the MVP wins**. If the equal-budget benchmark shows a real regret-vs-calls frontier improvement, then action-sufficient compression becomes a legitimate second-wave Hydra path: first wire the runtime helper, then thread a cheap safety side-channel, then revisit a root-only discard-phase version, and only much later reconsider masked / semantically aligned `delta_q` export. If the benchmark does not beat top-mass, kill the lane completely rather than renaming top-mass pruning. That is the narrowest recommendation the evidence supports after the stricter repo pass.

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
