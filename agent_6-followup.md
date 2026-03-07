## 1. what DEBC-AR is trying to buy Hydra that current AFBS + CT-SMC does not

Hydra already has the right ingredients for search-grade belief work — CT-SMC, Mixture-SIB, AFBS, predictive pondering, Hand-EV plumbing, and search-as-feature channels — but today those pieces are still arranged as a **specialist shell** rather than a belief-compression layer that actively removes redundant particle work. Reconciliation is explicit that AFBS should stay specialist / hard-state gated for now, and that CT-SMC should be treated as a credible belief source rather than as a broad new search project. DEBC-AR’s job is to add one missing capability at the root: **collapse particles that are decision-equivalent, keep apart particles that imply different root actions or different catastrophe structure, and spend root search only where that distinction could change the action.** That is the concrete asymmetry current AFBS + CT-SMC does not yet provide. ([GitHub][1])

The outside transfer is not “more search.” It is the combination of problem-driven scenario reduction, decision-based k-medoids, and abstraction-refining tree search: cluster uncertain scenarios by the **decisions they induce**, use true scenarios as representatives, and only refine abstractions when they can still change the current choice. That is a much better fit for Hydra than generic wider search or generic belief averaging, and it directly counters the determinization/pathology problem that hidden-world search can otherwise suffer from. ([arXiv][2])

## 2. revised final mechanism

**Final revision:** make DEBC-AR a **root-only, discard-phase belief compressor** for v0.

Not full-tree. Not call-phase. Not wall-exact. Root-only discard-phase is the highest-upside / lowest-scope cut that matches the repo Hydra actually has: bridge-side Hand-EV is discard-centric, current search features expose discard-level (\Delta Q), AFBS is still a shell, and inference is still fast-path actor+SaF with ponder-cache reuse. So DEBC-AR should first sit **between CT-SMC and root AFBS/ponder generation**, not inside every tree layer. ([GitHub][3])

Mechanically, v0 does four things:

1. Select the top posterior mass particles from CT-SMC.
2. Compute a cheap per-particle discard-action signature (Q_{\text{fast}}(I,X_p,a)).
3. Hard-partition particles by a catastrophic-defense bit pattern, then run weighted k-medoids **inside** each bucket.
4. Search only medoid representatives; re-split a cluster only if its current representative could still flip the global root winner.

I reject two tempting variants for v0. First, I reject wall-only Hand-EV per particle: Hydra’s live belief guidance still treats the fourth CT zone as **unseen**, with live/dead wall split delayed to late exactification, so wall-only counts would be a weak unsupported assumption. Second, I reject recursive DEBC-AR inside the whole tree: that is not aligned with the current “AFBS specialist shell” repo reality. ([GitHub][4])

## 3. exact state/action/proxy definitions

**State.**
Use
[
I = (\texttt{hand_counts}[34],\ \texttt{legal_mask}[46],\ \pi_{\text{base}}[46],\ \texttt{SafetyInfo}),
]
where (\pi_{\text{base}}) is the actor policy after legal masking / renormalization, and
[
X_p = \texttt{Particle.allocation}[34][4].
]
For v0, interpret the 4 CT columns as “three opponent hidden zones + unseen pool,” not as “three opponents + exact live wall.” That is the safest reading of the current belief stack. ([GitHub][5])

**Action-space assumptions.**
Hydra’s full action space is 46 actions: discards (0..33), aka discards (34..36), then riichi, chi-left, chi-mid, chi-right, pon, kan, agari, ryuukyoku, pass. `build_legal_mask` produces a `[bool; 46]`; in `RiichiSelect` and `KanSelect` phases, only discard actions are marked legal. v0 DEBC-AR supports only roots where the effective legal set is discard-only (`0..=36`), plus the existing upstream `AGARI` guard if agari is legal. If any legal `RIICHI`, `CHI_*`, `PON`, `KAN`, `RYUUKYOKU`, or `PASS` coexists at the root, DEBC-AR **falls back** to the current fast path / baseline AFBS. That is deliberate: the current reusable offensive and risk signals are discard-centric. ([GitHub][5])

**Cheap remaining-count proxy.**
For v0,
[
\texttt{remaining}*p[t] = \sum*{z=0}^{3} X_p[t][z].
]
Do **not** use only column 3. This keeps (Q_{\text{fast}}) aligned with Hydra’s current CT-SMC → Hand-EV usage and avoids pretending that the always-on CT table already has exact live-wall semantics. ([GitHub][4])

**Cheap offensive proxy.**
Compute
[
\texttt{HEV}*p = \texttt{compute_hand_ev}(\texttt{hand_counts},\texttt{remaining}*p),
]
reusing `hydra-core/src/hand_ev.rs`. For discard tile (t),
[
\text{off}*p(t)=
0.35,P*{\text{ten}}^{(1)}(t)
+0.45,P*{\text{win}}^{(2)}(t)
+0.20,P*{\text{win}}^{(3)}(t)
+0.15,\min!\left(1,\frac{\sum_u \text{ukeire}[t][u]}{16}\right)
+0.15,\min!\left(1.5,\frac{\mathbb E[\text{score}]_t}{8000}\right).
]
This uses only existing Hand-EV outputs. ([GitHub][1])

**Cheap defensive proxy.**
Move the existing loader-side `public_safety_score` logic into `hydra-core::safety` unchanged as a small shared extension. Define:

* `public_risk(t) = 1 - public_safety_score(safety, t)`.
* `threat_i = 1.0` if opponent (i) is riichi, else `cached_tenpai_prob[i]`.
* `hidden_risk_p(t) = max_i threat_i * 1[X_p[t][i] > 0]`.

Then
[
\text{risk}_p(t)=0.70,\text{public_risk}(t)+0.30,\text{hidden_risk}_p(t).
]

**Exact (Q_{\text{fast}}).**
Let (A) be the supported legal discard actions, and let (t(a)) be `discard_tile_type(a)` (so aka actions map back to 4/13/22). Let
[
\text{prior_bonus}(a)=0.25\left(\log\max(\pi_{\text{base}}(a),10^{-6})-\frac{1}{|A|}\sum_{b\in A}\log\max(\pi_{\text{base}}(b),10^{-6})\right).
]
Then for every supported action (a\in A),
[
Q_{\text{fast}}(I,X_p,a)
========================

\text{prior_bonus}(a)
+\text{off}_p(t(a))
-0.90,\text{risk}_p(t(a)).
]

This is the exact v0 proxy. It is cheap, action-complete over discard roots, and uses only signals Hydra already has or can expose with one tiny core-side move (`public_safety_score`). ([GitHub][5])

## 4. exact data structures and API sketch

Use these constants:

```rust
pub const DEBCAR_MAX_ACTIONS: usize = 14;   // max discard-like choices in a 14-tile hand
pub const DEBCAR_MAX_PARTICLES: usize = 64; // top-mass subset cap
pub const DEBCAR_MAX_CLUSTERS: usize = 8;   // total root reps after bucketing
```

Use these structs:

```rust
pub struct ParticleSignature {
    pub q_fast: [f32; DEBCAR_MAX_ACTIONS],   // valid prefix [0..n_actions)
    pub sigma: [f32; DEBCAR_MAX_ACTIONS],    // softmax(q_fast / tau_signature)
    pub risk: [f32; DEBCAR_MAX_ACTIONS],     // per-action risk proxy
    pub best_slot: u8,                       // index into supported_actions
    pub second_slot: u8,                     // index into supported_actions
    pub best_margin: f32,                    // q_fast[best] - q_fast[second]
    pub catastrophe_bits: u16,               // bit i => action i is catastrophe-preserving
}

pub struct ParticleSummary {
    pub particle_idx: u16,                   // index into CtSmc.particles
    pub weight: f32,                         // normalized posterior weight
    pub signature: ParticleSignature,
}

pub struct MedoidRecord {
    pub particle_idx: u16,                   // representative particle
    pub supported_actions: [u8; DEBCAR_MAX_ACTIONS],
    pub n_actions: u8,
    pub q_fast: [f32; DEBCAR_MAX_ACTIONS],
    pub sigma: [f32; DEBCAR_MAX_ACTIONS],
    pub catastrophe_bits: u16,
    pub total_weight: f32,                   // cluster posterior mass
}

pub struct ClusterRecord {
    pub cluster_id: u16,
    pub member_indices: smallvec::SmallVec<[u16; 16]>,
    pub total_weight: f32,
    pub medoid_particle_idx: u16,
    pub catastrophe_bits: u16,
    pub mean_sigma: [f32; DEBCAR_MAX_ACTIONS],
    pub intra_js: f32,                       // weighted mean JS to mean_sigma
    pub rep_q: [f32; DEBCAR_MAX_ACTIONS],    // root evaluator output for medoid
    pub rep_visits: [u16; DEBCAR_MAX_ACTIONS],
    pub split_depth: u8,
}

pub struct RefinementQueueItem {
    pub priority: f32,
    pub cluster_id: u16,
    pub flip_potential: f32,
    pub intra_js: f32,
    pub split_depth: u8,
    pub cluster_size: u16,
}
```

Root planner surface:

```rust
pub struct DebcarConfig {
    pub top_mass_threshold: f32,
    pub particle_cap: usize,
    pub tau_signature: f32,
    pub tau_js_single: f32,
    pub tau_js_extra: f32,
    pub max_clusters_total: usize,
    pub max_clusters_per_bucket: usize,
    pub lambda_margin: f32,
    pub lambda_alloc: f32,
    pub tau_split: f32,
    pub min_cluster_mass: f32,
    pub min_cluster_size: usize,
    pub min_visits_per_cluster: u32,
    pub public_risk_cat: f32,
    pub hidden_risk_cat: f32,
    pub delta_cat_keep: f32,
}

pub struct DebcarContext<'a> {
    pub info_state_hash: u64,
    pub hand_counts: [u8; 34],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub base_policy: [f32; HYDRA_ACTION_SPACE],  // already masked + normalized
    pub safety: &'a SafetyInfo,
    pub ct_smc: &'a CtSmc,
}

pub struct RepresentativeEval {
    pub q: [f32; DEBCAR_MAX_ACTIONS],
    pub visits: [u16; DEBCAR_MAX_ACTIONS],
    pub expanded_any: bool,
}

pub trait DebcarRootBackend {
    fn evaluate_medoid_root(
        &mut self,
        ctx: &DebcarContext<'_>,
        rep: &MedoidRecord,
        visit_budget: u32,
    ) -> RepresentativeEval;
}

pub struct DebcarRootResult {
    pub supported_actions: [u8; DEBCAR_MAX_ACTIONS],
    pub n_actions: u8,
    pub q_values: [f32; HYDRA_ACTION_SPACE],
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub total_visits: u32,
    pub selected_particle_count: u16,
    pub cluster_count: u8,
    pub refined_cluster_count: u8,
}
```

**File-level insertion points.**

* `hydra-core/src/ct_smc.rs`: add `normalized_top_mass_indices()` and `particle_remaining_counts()`.
* `hydra-core/src/hand_ev.rs`: no model change; reuse `compute_hand_ev()`. Optional tiny helper: `sum_ukeire_row(&HandEvFeatures, tile) -> f32`.
* `hydra-core/src/safety.rs` (preferred) or `bridge.rs`: move the existing loader-side `public_safety_score` into core.
* `hydra-core/src/afbs.rs`: add `DebcarConfig`, `DebcarContext`, the five structs above, `DebcarRootBackend`, `DebcarRootPlanner`, and tests.
* `hydra-core/src/endgame.rs`: no v0 change.
* `hydra-train/src/data/sample.rs`, `mjai_loader.rs`, `losses.rs`, `model.rs`: **no v0 changes**. Runtime first; distillation / labels later. That matches the repo’s current “close target loops first, keep AFBS specialist” posture. ([GitHub][6])

## 5. search loop pseudocode

```text
fn plan_root(ctx, backend, cfg) -> DebcarRootResult {
    // 0. support + fallback
    if !validate_legal_mask(ctx.legal_mask) => fallback_baseline();
    if legal AGARI => return upstream agari guard result;
    supported_actions = legal discard actions in 0..=36;
    if supported_actions.len() < 2 => fallback_baseline();
    if any legal unsupported non-discard action => fallback_baseline();
    if ctx.ct_smc.is_empty() => fallback_baseline();

    // 1. particle subset
    selected = ct_smc.normalized_top_mass_indices(cfg.top_mass_threshold, cfg.particle_cap);
    if selected.len() < 16 => fallback_baseline();

    // 2. signature generation
    summaries = [];
    for p_idx in selected:
        remaining_p[t] = sum_z particle[p_idx].allocation[t][z];
        hev_p = compute_hand_ev(ctx.hand_counts, remaining_p);
        for action slot j in supported_actions:
            tile = discard_tile_type(action);
            q_fast[j] = exact formula from section 3
            sigma[j]  = softmax(q_fast / cfg.tau_signature)
            risk[j]   = exact risk formula
        best, second = argmax2(q_fast)
        catastrophe_bits = 0
        for j in actions:
            if public_risk(tile_j) >= cfg.public_risk_cat
               && hidden_risk_p(tile_j) >= cfg.hidden_risk_cat
               && q_fast[j] >= q_fast[best] - cfg.delta_cat_keep:
                   catastrophe_bits |= (1 << j)
        push ParticleSummary

    // 3. hard catastrophe partition
    buckets = group summaries by catastrophe_bits

    // 4. initial clustering inside each bucket
    clusters = []
    for bucket in buckets:
        uniq_best = count_unique(best_slot)
        mean_sigma = weighted_mean(bucket.sigma)
        mean_js = weighted_mean(JS(summary.sigma, mean_sigma))
        if uniq_best == 1 && mean_js < cfg.tau_js_single:
            k = 1
        else:
            k = min(cfg.max_clusters_per_bucket,
                    uniq_best + 1[mean_js > cfg.tau_js_extra])

        run weighted PAM / k-medoids inside bucket
            distance(p, m) =
                JS(p.sigma, m.sigma)
                + cfg.lambda_margin * abs(p.best_margin - m.best_margin)

        for each cluster:
            medoid = argmin weighted distance
            build ClusterRecord with rep_q = zeros for now

    if clusters.len() > cfg.max_clusters_total:
        merge smallest-mass same-bit-pattern clusters first

    // 5. first representative evaluation pass
    allocate initial budgets:
        budget0[c] ∝ cluster.total_weight, floor = cfg.min_visits_per_cluster
    for cluster c:
        rep = MedoidRecord from medoid particle
        eval = backend.evaluate_medoid_root(ctx, rep, budget0[c])
        if !eval.expanded_any:
            eval.q = rep.q_fast   // cheap fallback inside the planner
        cluster.rep_q = eval.q
        cluster.rep_visits = eval.visits

    // 6. aggregate root action values
    Q[a] = sum_c cluster.total_weight * cluster.rep_q[a] / sum_c cluster.total_weight
    a_star = argmax_a Q[a]

    // 7. refinement queue
    for cluster c:
        b_star = argmax_a c.rep_q[a]
        flip_potential =
            c.total_weight * max(0, c.rep_q[b_star] - c.rep_q[a_star])
        priority = flip_potential + 0.5 * c.intra_js
        push RefinementQueueItem(priority, ...)

    // 8. one-step split / refine loop (v0 = at most one split round)
    while queue not empty:
        top = pop max priority
        if top.priority < cfg.tau_split: break
        if top.cluster_size < cfg.min_cluster_size: continue
        if top.split_depth >= 1: continue

        split target cluster into 2 medoids, same catastrophe bucket only
        child_budget[k] ∝ child_weight + cfg.lambda_alloc * child_priority
        re-evaluate both children
        replace parent by children
        recompute Q and queue

    // 9. output
    exit_policy over supported actions = softmax(Q_supported)
    scatter into [46], zero unsupported
    return DebcarRootResult

    // 10. planner-level fallback conditions (abort to baseline)
    if any non-finite q/risk/sigma
       or sum(selected weights) not in [0.99, 1.01]
       or clusters.is_empty()
       or best output illegal:
           fallback_baseline();
}
```

This is the intended v0 loop. It is root-only, split-once, and discard-phase-only by design. The split trigger is myopic / decision-focused: keep refining only while a cluster can still change the global winner by enough to justify more compute. That is the right metareasoning shape for Hydra’s current “AFBS specialist” posture. ([OpenReview][7])

## 6. defaults and thresholds

Use these defaults first:

```text
top_mass_threshold      = 0.97   // safe range: 0.95 .. 0.99
particle_cap            = 64     // safe range: 32 .. 128
tau_signature           = 0.35   // safe range: 0.25 .. 0.50
tau_js_single           = 0.05   // safe range: 0.03 .. 0.07
tau_js_extra            = 0.12   // safe range: 0.08 .. 0.15
max_clusters_total      = 8      // safe range: 4 .. 12
max_clusters_per_bucket = 4      // safe range: 2 .. 6
lambda_margin           = 0.25   // safe range: 0.10 .. 0.50
lambda_alloc            = 2.00   // safe range: 1.00 .. 3.00
tau_split               = 0.08   // safe range: 0.05 .. 0.15
min_cluster_mass        = 0.05   // safe range: 0.03 .. 0.08
min_cluster_size        = 2
min_visits_per_cluster  = 8      // safe range: 4 .. 16
public_risk_cat         = 0.55   // safe range: 0.45 .. 0.65
hidden_risk_cat         = 0.65   // safe range: 0.55 .. 0.80
delta_cat_keep          = 0.15   // safe range: 0.10 .. 0.25
```

Interpretation:

* `top_mass_threshold=0.97` matches the existing endgame-style “top posterior mass first” instinct.
* `max_clusters_total=8` and split-once are deliberate: v0 should prove compression value before it grows into a deeper policy.
* `public_risk_cat` + `hidden_risk_cat` are intentionally conservative: tail protection should trip early. ([GitHub][1])

## 7. failure modes and safeguards

**Failure 1: wrong wall semantics sneaks in.**
Guardrail: `remaining_p[t]` must use row-sum unseen counts, not “column 3 = live wall.” If later exact wall split is needed, that belongs only in late-mode / endgame exactification. Do not silently change (Q_{\text{fast}}) to wall-only counts in v0. ([GitHub][4])

**Failure 2: catastrophic branches get averaged away.**
Guardrail: the tail rule is hard, not soft. Particles with different `catastrophe_bits` may not cluster together. More precisely, for supported slot (j), set catastrophe bit (j) iff:
[
\text{public_risk}(t_j)\ge 0.55,\quad
\text{hidden_risk}*p(t_j)\ge 0.65,\quad
Q*{\text{fast}}(j)\ge Q_{\text{fast}}(\text{best})-0.15.
]
That is the exact catastrophic-defense tail rule. It preserves “near-best but dangerous under some hidden worlds” structure. ([arXiv][2])

**Failure 3: medoid mismatch — representative search does not reflect its members.**
Guardrail: after representative evaluation, compute
[
\Delta_{\text{medoid}} = \max_j |,\text{rep_q}[j] - \text{weighted_mean}*{p\in c} Q*{\text{fast},p}[j],|.
]
If (\Delta_{\text{medoid}} > 0.20) on the cluster winner, force one split if allowed; otherwise fallback.

**Failure 4: compression buys no compute.**
Guardrail: if selected particles (\le 8), or if cluster count == selected particle count, or if after one split round the cluster count is still (> 0.75 \times) selected-particle count, abort DEBC-AR and use the baseline path.

**Failure 5: unsupported root types leak in.**
Guardrail: call-phase / mixed-phase roots fall back immediately. No partial support. That keeps v0 aligned with the repo’s current signal surface rather than pretending Hydra already has good call-phase per-particle value proxies. ([GitHub][5])

## 8. benchmark design and kill criteria

**Cheapest serious benchmark.**
Do a **root-only approximation benchmark** against a singleton-particle reference on a 10k discard-phase state slice.

**State slice filter.**

* legal discard actions in ([2,14])
* no legal unsupported non-discard action at the same root
* selected CT-SMC particles in ([16,64])
* hard-state bias: either `policy_top2_gap(base_policy) < 0.10` or max public discard risk (> 0.25)

**Compare three methods under the same representative evaluator backend.**

1. **Reference:** evaluate every selected particle as its own singleton cluster; aggregate by posterior weight.
2. **DEBC-AR:** the mechanism above.
3. **Naive baseline:** top-8 particles or one global medoid, no catastrophe buckets, no refinement.

**Primary metrics.**

* `root_action_agreement`: DEBC-AR vs singleton reference
* `mean_abs_q_error`: mean absolute error on supported discard actions
* `rep_eval_reduction`: ((N_{\text{singleton}} - N_{\text{DEBCAR}})/N_{\text{singleton}})
* `tail_state_agreement`: same as above on tail states only
* `tail_miss_rate`: fraction of tail states where DEBC-AR chooses an action whose catastrophe mass exceeds the reference winner’s catastrophe mass by (> 0.10)

Define
[
\text{cat_mass}(j)=\sum_p w_p,1[\text{catastrophe_bit}_{p,j}=1].
]
A state is a **tail state** if (\max_j \text{cat_mass}(j)\ge 0.10).

**Success thresholds.**

* overall action agreement (\ge 97%)
* tail-state action agreement (\ge 95%)
* mean absolute (Q) error (\le 0.03)
* representative-eval reduction (\ge 35%)
* tail miss rate (\le 0.5%)

**Kill thresholds.**

* overall agreement (< 97%), or tail agreement (< 93%)
* eval reduction (< 20%)
* tail miss rate (> 1.0%)
* fallback rate (> 30%) on the intended slice
* any non-finite / weight-mass conservation failure on (> 0.5%) of benchmark states

**Precheck before the serious benchmark.**
Run the same benchmark with representative eval temporarily set to medoid `q_fast` only. That is not the success gate; it is only the sanity gate that the clustering / catastrophe logic is behaving before wiring into real root search. The serious go/no-go decision must use the singleton-vs-clustered comparison under the same actual backend evaluator. ([arXiv][2])

## 9. final verdict: prototype now / reserve shelf / reject

**Prototype now.**

But prototype the **revised** version above: root-only, discard-phase-only, split-once, catastrophe-bucketed, no train-side changes. That version is strong enough to matter, small enough to wire into the current repo, and honest about what Hydra already has versus what it does not. The original broader DEBC-AR idea was too ambitious for the current AFBS shell. This revised DEBC-AR is not.

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[2]: https://arxiv.org/abs/2404.07810 "https://arxiv.org/abs/2404.07810"
[3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/encoder.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/encoder.rs"
[4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md"
[5]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/action.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/action.rs"
[6]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-core/src/ct_smc.rs"
[7]: https://openreview.net/forum?id=0qnPBmvJSaf "https://openreview.net/forum?id=0qnPBmvJSaf"
