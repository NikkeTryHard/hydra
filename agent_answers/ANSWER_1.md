<start-prompt>
## Hydra deep-agent handoff

## Primary working package

I have attached a zip file to this prompt called `hydra_agent_handoff_docs_only.zip`.

Use that zip as the **primary docs package** instead of trying to reconstruct the project from browsing.

Expected workflow:
1. Open / extract `hydra_agent_handoff_docs_only.zip` and treat the extracted markdown docs as the primary source material.
2. Read the included docs carefully before forming conclusions.
3. Use the raw GitHub markdown links below only as supplemental reference / cross-check material.
4. Use the attached PDF package as the primary paper attachment set.

If you cannot access the attached zip for any reason, fall back to fetching the markdown docs directly from the **raw GitHub links** in this document.

Important:
- Do **not** rely on normal GitHub browsing/search to reconstruct the repo.
- Do **not** rely on generic/plain web search to discover the project files.
- If the zip is unavailable, fetch the raw markdown docs directly from the raw GitHub links in this handoff instead.

You are a deep-thinking **research and design advisor** for **Hydra**, a Riichi Mahjong AI project whose goal is to reach or exceed **LuckyJ-level** strength.

Your job is **not** to inspect source code, browse loosely, or directly modify anything. Your job is to read the design docs and papers, think very hard about the hardest and most underspecified parts, and then produce the strongest possible technical guidance for a separate coding agent to implement later.

Treat the following as the governing hierarchy:

1. `research/design/HYDRA_FINAL.md` = the architectural SSOT for final strength
2. `research/design/IMPLEMENTATION_ROADMAP.md` = the implementation ordering and gates
3. `research/BUILD_AGENT_PROMPT.md` = execution-discipline and rigor overlay on the other docs
4. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

**Important:** use `research/BUILD_AGENT_PROMPT.md` as a cross-cutting overlay on the rest of the docs, but do so in a **design/advisory** sense rather than a code-execution sense. In other words, use it to understand ordering, rigor, and what must not be hand-waved.

### What you should focus on

Focus on the parts of Hydra that are:
- most important for actual strength
- technically difficult
- underspecified or only partially specified in the docs
- likely to require precise formulas, update rules, approximations, thresholds, and engineering choices

You should especially think hard about topics such as:
- belief inference and belief-state representations
- AFBS / search-as-feature / pondering / ExIt-style improvement loops
- Hand-EV or offensive oracle features
- endgame exactification / late-game solving
- robust opponent modeling under uncertainty
- target generation for advanced auxiliary heads
- practical approximations that preserve strength without blowing up runtime
- evaluation and ablation plans that would let a coding agent verify the choices

### What kind of answer is wanted

Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
- formulas over vague prose
- precise algorithms over general suggestions
- concrete thresholds/hyperparameters over hand-waving
- pseudocode or compact code snippets where implementation detail matters
- explicit tradeoff analysis
- ablation/evaluation plans tied to Hydra’s stated architecture
- discussion of what the docs leave unspecified and the best way to fill those gaps

Avoid spending your budget on:
- generic motivational advice
- broad summaries of Mahjong AI history unless they directly affect design choices
- telling the user to “just run more experiments” without specifying which ones and why
- pretending to have implemented or validated anything
- spending time trying to inspect code rather than thinking from the docs and references

### Required deliverables

Produce a technical design package for a separate coding agent. Your deliverables should be:

1. A prioritized analysis of the most important hard/underspecified Hydra problems.
2. For each major problem, a concrete proposed solution including:
   - the exact algorithmic idea
   - formulas / objective functions / update rules
   - recommended constants, thresholds, and hyperparameters
   - what data or targets would be needed
   - runtime/inference-time behavior
   - training-time behavior
   - likely failure modes and mitigations
   - evaluation criteria and ablations
3. Pseudocode or compact illustrative code snippets for the hardest parts, especially where the docs are too abstract.
4. Recommended module boundaries / interfaces / data flows a coding agent should implement.
5. A practical implementation order for a separate coding agent to follow.
6. A concise risk assessment identifying which proposals are strongest, which are fragile, and which are likely not worth the complexity.

### Success condition

Your output should make it materially easier for a separate coding agent to implement the remaining Hydra improvements correctly and efficiently. The goal is to produce the best possible design guidance for making Hydra genuinely stronger, especially where the docs are ambitious but underspecified.

Start by reading the attached `hydra_agent_handoff_docs_only.zip` carefully. Use the raw GitHub markdown links and external paper links below as supplementary context, not as your primary reading path. If the zip is inaccessible, fetch the markdown docs directly from the raw links below rather than relying on GitHub search or generic web search.

<references-are-omitted>

## Practical attachment order if the external agent has a hard upload limit
1. Upload the 20 PDFs from the attached `deep_agent_20_pdfs` package.
2. Send the high-priority raw GitHub markdown links first.
3. If more context is needed, send the remaining markdown raw links and any especially relevant external PDF links.

## Notes
- `research/BUILD_AGENT_PROMPT.md` should be treated as an execution overlay on top of the architecture/design docs.
- Do not use AGPL code as implementation source.
- Keep `.cargo/config.toml` local-only; it is not part of the handoff.
</start-prompt>

<start-answer>
I treated the docs zip as the primary source, then cross-checked the key claims against the raw markdown and the cited papers.

# Hydra design package

## 1. Priority order

1. **Make the belief posterior good enough that search is worth paying for.**
2. **Search over public-event trees, not per-determinization perfect-info trees.**
3. **Move Hand-EV up the schedule; it is cheaper and higher-ROI than deep search.**
4. **Replace “independent danger” with a factorized threat model.**
5. **Use exact chance only where it matters most: the endgame.**
6. **Only trust search labels when search quality is measurable.**
7. **Train a small bank of opponent archetypes before chasing fancy population diversity.**
8. **Evaluate with duplicate seed bundles and slice metrics, or the project will lie to you.**

Hydra’s own SSOT already points in this direction: CT-SMC exact contingency-table belief, AFBS, robust opponent nodes, Hand-EV features, endgame exactification, SaF, and ExIt/pondering are the central strength stack, while the opponent-modeling doc adds wait-set, value-conditioned threat, call-intent, safety reserve, and CVaR. Suphx’s look-ahead features and Microsoft’s GRP/oracle-guiding stack, RVR’s relative-value/expected-reward variance reduction, OLSS’s limited-opponent search, and ISMCTS’s strategy-fusion warning all support that emphasis. ([GitHub][1])

---

## 2. Belief inference: one real posterior, not two half-posteriors

### What is underspecified

The docs say “Mixture-SIB for fast amortized belief” and “CT-SMC exact contingency-table sampler for search,” but they do not pin down:

* hidden-state granularity,
* sequential update mechanics,
* the event likelihood model,
* how opponent-reading heads feed back into the particle filter,
* how to keep one coherent belief representation instead of a Mixture-SIB path and a separate Sinkhorn/tile-allocation path.

The biggest design mistake available here is building **two unrelated belief systems**. I would not do that.

### Recommendation

Use **one belief stack**:

* **Amortized layer:** Mixture-SIB outputs per-component entropic fields and mixture weights.
* **Search-grade layer:** CT-DP / CT-SMC samples exact hidden allocations from those fields and updates them sequentially.
* **No second standalone Sinkhorn head.** Reuse the Mixture-SIB kernel/fields as the only entropic projection mechanism.

Hydra FINAL already says Mixture-SIB stays as the fast network-input belief and CT-SMC becomes the search-grade posterior; the later “constraint-consistent Sinkhorn projection head” is conceptually overlapping. Unify them now instead of paying double calibration cost later. ([GitHub][1])

### Hidden state representation

Use **37 hidden tile types**, not 34:

* 34 normal tile kinds
* 3 explicit aka rows: red 5m, red 5p, red 5s

The engine/action space already distinguishes red fives, so collapsing them inside belief throws away exactly the information needed for hand value and endgame push/fold. ([GitHub][2])

Use **4 CT-DP zones** for speed:

* left concealed
* across concealed
* right concealed
* wall bucket

Then attach a **dead-wall reservoir** sampled from the wall bucket:

* `deadwall_hidden ~ MultiHyper(wall_counts, n = dead_wall_unseen)`
* `live_wall = wall_counts - deadwall_hidden`

This keeps the DP 3D as in Hydra FINAL, but restores live-wall vs dead-wall semantics for draws, rinshan, and ura/dora handling.

### Exact CT-DP sampler

For each component `l`, let `F_l[k,z]` be the log-field for tile row `k` and zone `z`, with `k in {1..37}`, `z in {L,A,R,W}`.

Let `r_k` be the remaining count of tile type `k`, and residual zone capacities `(cL,cA,cR)`.

Use the exact partition recursion:

[
\log Z_k(c_L,c_A,c_R)=\log\sum_{x\in\mathcal C(r_k;c_L,c_A,c_R)}
\exp\Big(\sum_z x_z F_l[k,z]+\log Z_{k+1}(c_L-x_L,c_A-x_A,c_R-x_R)\Big)
]

with `cW` derived from remaining total mass, exactly as the SSOT suggests, but over 37 rows. Backward sample each row from:

[
p(x_k=x\mid c)=\exp\Big(\sum_z x_zF_l[k,z]+\log Z_{k+1}(c-x)-\log Z_k(c)\Big)
]

Precompute all row compositions for `r in {0,1,2,3,4}`.

### Sequential particle update

The practical update is:

[
\log w_n \leftarrow \log w_n

* \log p_\phi(e_t\mid I_t, X_t^{(n)})
* \lambda_{\text{tp}}\log p(\hat y^{\text{tp}}_t\mid X_t^{(n)})
* \lambda_{\text{ws}}\log p(\hat Y^{\text{ws}}_t\mid X_t^{(n)})
* \lambda_{\text{val}}\log p(\hat y^{\text{val}}_t\mid X_t^{(n)})
  ]

where:

* `e_t` is the actual public event,
* `p_phi` is a **privileged public-event likelihood model**,
* the hat terms are current public-head predictions,
* the last three terms are weak pseudo-likelihood corrections.

Use conservative weights:

* `lambda_tp = 0.10`
* `lambda_ws = 0.05`
* `lambda_val = 0.02`

and clip the event log-likelihood increment to `[-10, 0]` per event to prevent catastrophic particle collapse from an overconfident event model.

Resample when:

[
\text{ESS}=\frac{1}{\sum_n \bar w_n^2} < 0.4P
]

Then do `3` MH rejuvenation moves per particle using margin-preserving 2x2 cycle swaps.

### Event likelihood model

Build a small frozen module just for filtering:

```python
class BeliefLikelihoodNet:
    # public embedding cached once per event
    pub_embed: R^d
    # hidden features from particle: hand counts, live wall, dead wall, meta
    hidden_proj: MLP(82 -> 128)
    joint: MLP((d+128) -> 128)
    discard_head: Linear(128 -> 37)
    tedashi_head: Linear(128 -> 1)
    riichi_head: Linear(128 -> 1)
    call_head: Linear(128 -> K_call)
```

Train it on oracle-visible Phase 1/2 states using actual public events. It does **not** have to be a huge net; it just has to provide a calibrated ranking over observed public events.

### Pseudocode

```python
def update_posterior(public_state, event, mixture_sib, particles):
    h_pub = likelihood_net.public_embed(public_state, actor=event.actor)

    for p in particles:
        p = propagate_particle_one_event(p, event)   # draw/call bookkeeping
        loglik = likelihood_net.logprob(h_pub, p.hidden_features(event.actor), event)
        loglik += 0.10 * tenpai_consistency(public_state, p)
        loglik += 0.05 * waitset_consistency(public_state, p)
        loglik += 0.02 * value_consistency(public_state, p)
        p.logw += max(-10.0, min(0.0, loglik))

    normalize(particles)

    if ess(particles) < 0.4 * len(particles):
        particles = stratified_resample(particles)
        particles = mh_rejuvenate(particles, steps=3)

    return particles
```

### Runtime and training behavior

* **On-turn:** `L=4`, `P=192`
* **Ponder:** `L=8`, `P=1536`
* CT-DP itself should remain sub-millisecond-class if implemented in log-space with precomputed row compositions, which is exactly why Hydra FINAL likes it. ([GitHub][1])

Train with:

[
L_{\text{belief}} = L_{\text{event-NLL}} + 0.25L_{\text{rowKL}} + 0.25L_{\text{colKL}} + 0.10L_{\text{posterior-LL}}
]

using oracle-visible states from Phase 1 and later self-play oracle traces.

### Failure modes

* **Particle collapse:** overconfident event model.
  Fix with log-likelihood clipping, temperature scaling, ESS floor, and weak auxiliary corrections.
* **Wrong red-five or dead-wall accounting:** threat value and endgame errors.
  Fix with 37-type rows and dead-wall reservoir.
* **Two competing belief implementations:** chronic calibration drift.
  Fix by not implementing separate Sinkhorn and Mixture-SIB pipelines.

### Evaluation

Use Hydra FINAL’s own gates, but make them concrete:

* posterior NLL against held-out hidden allocations,
* MI error on top 200 correlated tile-pair events,
* wait-set calibration against true ron-eligible waits,
* ESS profile by turn bucket. ([GitHub][1])

---

## 3. AFBS: search on public events, not on determinizations

### What is underspecified

Hydra FINAL gives beam widths, depths, particles, robust opponent nodes, and caches, but not the actual tree semantics. That matters.

A bad implementation would do ordinary PIMC over perfect-information worlds and average them. That is exactly where strategy fusion bites. ISMCTS’s whole warning is that determinization-based methods get fooled when plans differ across hidden states. OLSS’s contribution is that limiting the opponent strategy set can make large imperfect-information search practical. The right Hydra compromise is a **public-event tree with particle-attached nodes**. ([White Rose Research Online][3])

### Recommendation

Use a **public-event expectimax / robust-min tree**.

#### Node types

1. `MyDecision`
2. `MyFutureDrawChance`
3. `OpponentPublicEvent`
4. `Terminal`

Each node stores:

* public state hash,
* belief signature,
* particle handle,
* cached network eval,
* aggregated risk/value summaries.

#### Key idea

* Opponent hidden draws are **not** separate chance nodes.
* At opponent turns, branch on **public events** directly: discard type, riichi+discard, call, pass.
* The distribution over those public events comes from particles + archetype-conditioned opponent policies.

That preserves information-state semantics.

### Opponent-node backup

For opponent `j`, with archetypes `i=1..K`:

[
V_i = \min_{q: D_{\mathrm{KL}}(q|p_i)\le \varepsilon_c} \sum_e q(e),Q_e
]

with solution

[
q_{\tau,i}(e)\propto p_i(e)\exp(-Q_e/\tau)
]

and `tau` found by bisection so the KL ball is tight. Then combine archetypes by soft-min:

[
V_{\text{opp}} =
-\tau_{\text{arch}}
\log\sum_i w_i \exp(-V_i/\tau_{\text{arch}})
]

Recommended starting values:

* `K = 4` archetypes
* `epsilon_c = q95_contextual_KL`, clipped to `[0.02, 0.20]`
* `tau_arch = 0.5`

The docs already want both KL-robust opponent nodes and an OLSS-style archetype set; the formula above is the cleanest way to combine them. ([GitHub][1])

### Search expansion policy

At `MyDecision` nodes, expand:

* top `6` legal actions by policy prior,
* `+1` minimum-risk discard,
* `+1` top Hand-EV discard,
* always include agari/riichi if legal.

So practical branching is usually `6–9`.

At `OpponentPublicEvent` nodes, expand:

* top `5` public events by robust mass,
* plus an `"other"` bucket if residual mass `> 0.05`.

At `MyFutureDrawChance` nodes:

* if `wall > 6`: top `8` effective draws + residual bucket,
* if `wall <= 6`: exact branching over all live draws with nonzero count.

### Beam schedule

I would pin the defaults to the middle of Hydra FINAL’s ranges:

* **on-turn:** `W=96, D=5, P=192`
* **ponder:** `W=512, D=12, P=1536`

That is easier to benchmark than ranges and still consistent with the SSOT. ([GitHub][1])

### Frontier score

Use a beam score that prefers value, uncertainty reduction, and reach mass:

[
S(n)=\mu(n)+0.5,\sigma(n)+0.2\log(P_{\text{reach}}(n)+10^{-8})-0.05,d(n)
]

where `mu` and `sigma` are particle-aggregated mean/std of leaf value.

### Pseudocode

```python
def afbs(root, cfg):
    frontier = [root]

    for depth in range(cfg.depth):
        cand = []
        for node in frontier:
            children = expand_public_children(node, cfg)
            batched_leaf_eval(children)
            cand.extend(children)

        frontier = top_k(cand, cfg.beam_width, key=lambda n: n.beam_score)

    return robust_backup_to_root(frontier)
```

### Runtime behavior

* `MyDecision`: batched learner/value eval
* `OpponentPublicEvent`: robust event aggregation
* `MyFutureDrawChance`: exact or truncated chance
* reuse subtrees aggressively after actual events

Hydra FINAL already wants transposition, NN cache, Sinkhorn warm starts, and predictive ponder cache; those are all much easier when the tree is over public events rather than private-world nodes. ([GitHub][1])

### Failure modes

* **Too much residual bucket mass:** search is pretending to be exact while ignoring too much branch mass.
  Require expanded mass `>= 0.85` before using search as a training target.
* **Wrong archetype posterior:** over-robust or mis-robust backups.
  Use posterior smoothing and never let any `w_i` exceed `0.95`.
* **Search too shallow to matter:** then SaF and ExIt will learn noise.
  Gate search labels by visit count and expanded mass.

### Evaluation

Hydra FINAL’s G0 and G3 are the right top-line criteria; make them paired, duplicate, and slice-based:

* mean decision improvement on 200k hard states,
* fraction of negative deltas,
* search gain vs latency curve,
* “SaF without live search” recovery fraction. ([GitHub][1])

---

## 4. Hand-EV: move it earlier than deep AFBS

### Why

Hydra FINAL calls Hand-EV one of the core OMEGA additions, the roadmap puts it in the final block, and Suphx explicitly says its 100+ look-ahead features were a major practical gain. This is exactly the sort of cheap, CPU-side strength that should come **before** deep search, not after it. ([GitHub][1])

### Recommendation

Implement a **3-self-draw offensive recursion** on CPU, using the existing shanten batch cache and exact scoring on terminal wins. Keep it greedy but exact within that greedy policy.

### State and outputs

```python
HandEvFeatures:
    p_tenpai[37][3]      # discard x horizon {1,2,3}
    p_win[37][3]
    e_score[37]
    ukeire_vec[37][37]   # discard x draw-tile
```

### Recursion

For a post-discard 13-tile hand `h`, remaining live-wall counts `r`, and horizon `d`:

* If draw `t` makes agari, score exactly.
* Else choose the best discard after drawing `t` using a recursive proxy.

Use the recursive tuple ordering:

1. maximize `p_win`
2. then `p_tenpai`
3. then `ukeire_1`
4. then `e_score`

That gives a deterministic greedy continuation policy.

### Exact-ish formula

[
\text{Eval}(h,r,d)=
\sum_{t: r_t>0} \frac{r_t}{R},\text{BestAfterDraw}(h+t, r-e_t, d)
]

with

[
\text{BestAfterDraw}(h+t,r',d)=
\max_{b\in \mathcal D(h+t)} \text{Eval}(h+t-b,r',d-1)
]

where the max is lexicographic over `(p_win, p_tenpai, ukeire_1, e_score)`.

### Simplified ron model

Do not try to solve full multi-agent ron dynamics here. Use a cheap approximation only once the hand is tenpai.

If `W` is the wait set and `p_i^{disc}(w)` is the next-discard model for opponent `i`:

[
P_{\text{ron}}^{(d)}(W)=
1-\prod_i
\left(1-\sum_{w\in W} p_i^{disc}(w)\right)^{m_i(d)}
]

where `m_i(d)` is the expected number of discard chances opponent `i` gets before our `d`-th next self-draw.

Then

[
P_{\text{win}} = 1-(1-P_{\text{tsumo}})(1-P_{\text{ron}})
]

Use `kappa_ron = 1.0` when mixing ron mass into score expectation.

### Practical pruning

* If current shanten `<= 1`, expand all effective draws.
* Else only expand top `12` draws by `remaining[t] * immediate_gain(t)`.
* At depth `>=2`, keep only top `3` post-draw discard branches.

That keeps CPU cost sane.

### Pseudocode

```python
@memoize
def eval_hand(hand13, live_counts, depth):
    if depth == 0:
        return Metrics(
            p_tenpai=float(shanten(hand13) == 0),
            p_win=0.0,
            e_score=0.0,
            ukeire1=ukeire_mass(hand13, live_counts),
        )

    out = zero_metrics()
    R = sum(live_counts)

    for t in candidate_draws(hand13, live_counts):
        p = live_counts[t] / R
        hand14 = add_tile(hand13, t)

        if is_agari(hand14):
            s = exact_score(hand14, win_tile=t)
            out += p * Metrics(1.0, 1.0, s, 0.0)
            continue

        best = None
        for b in legal_discards(hand14):
            child = eval_hand(discard(hand14, b), live_counts - onehot(t), depth - 1)
            best = lexmax(best, child, key=("p_win", "p_tenpai", "ukeire1", "e_score"))

        out += p * best

    return out
```

### Failure modes

* **Greedy continuation bias:** it is not full search.
  Fine. These are features, not the final policy.
* **Ron approximation too optimistic:**
  Benchmark `kappa_ron` on held-out exact/MC slices.
* **Too slow:**
  Cache by canonical hand + quantized live-wall counts + riichi state.

### Evaluation

* Correlation with exact wall-small solver on `wall <= 12`
* Correlation with Monte Carlo rollouts on midgame slices
* Pure ablation: add only Hand-EV planes to the base net before any search

A strong result here is often a better use of time than deeper AFBS.

---

## 5. Opponent modeling: factorize threat instead of learning danger from scratch

### What is underspecified

Hydra’s opponent-modeling doc is rich, but the target generation is still fuzzy in the places that matter most:

* danger labels,
* value-conditioned threat,
* call-intent supervision,
* how these heads should interact.

The cleanest improvement is to stop making the danger head learn everything at once.

Hydra already wants tenpai, wait-set, value-conditioned tenpai, call-intent, safety reserve, and CVaR-style tail control. The easiest way to cash that out is a factorized threat model. ([GitHub][4])

### Recommendation: danger = tenpai × wait-set × residual

Define, per opponent `i` and tile `t`:

[
\text{logit};p^{\text{danger}}_{i,t}
====================================

\text{logit};p^{\text{tenpai}}*i
+
\text{logit};p^{\text{wait}}*{i,t}
+
r_{i,t}
]

where `r_{i,t}` is a small residual head conditioned by call-intent and value-threat features.

This is much easier to train than a monolithic danger head because:

* tenpai is dense,
* wait-set is denser than actual deal-in events,
* the residual only has to learn the leftover calibration error.

### Dense targets

#### Tenpai

[
y^{\text{tp}}_i = \mathbf{1}[\text{shanten}_i = 0]
]

#### Wait-set

[
y^{\text{ws}}_{i,t} = \mathbf{1}[t \in W_i^{\text{ron-eligible}}]
]

with furiten exclusion applied.

#### Value-conditioned threat

Do **not** use “eventual winning hand value” as the main label. That is survivorship-biased.

Use the **current exact threat value** when the opponent is tenpai:

[
\tilde p_i(w)\propto \text{live}*w + \kappa*{\text{ron}},p^{disc}_{\text{all}}(w)
]

[
v_i = \sum_{w \in W_i} \tilde p_i(w), \text{score}_i(w)
]

with `kappa_ron = 1.0`.

Then bin `v_i` into:

* `<=1300`
* `1301–2600`
* `2601–5200`
* `5201–8000`
* `8001–12000`
* `12001–18000`
* `18000+`

That produces a dense, current-state target instead of a winner-only target.

#### Call-intent

For Phase 1, I would use a **hybrid target**:

[
y^{intent} = \lambda_{rule},y^{rule} + (1-\lambda_{rule}),y^{outcome}
]

with

* `lambda_rule = 0.7` before the first open call,
* `lambda_rule = 0.5` after the first open call.

`y_rule` is a rule-based soft label from current meld pattern, suit concentration, value-honor retention, chi/pon ratio, and terminal/honor density.
`y_outcome` is the eventual winning-yaku archetype when available.

That reduces survivorship bias without pretending the current plan is perfectly observable.

### Losses

Recommended starting weights:

[
L_{aux} =
0.03L_{tp}
+0.02L_{ws}
+0.02L_{val}
+0.015L_{intent}
+0.05L_{danger}
+10^{-3}|r|_2^2
]

Use focal BCE for wait-set/danger and plain CE for value/intent.

Add a gradient guard: if total aux gradient norm exceeds `0.35` of the policy gradient norm, downscale all aux losses for that batch.

### FiLM conditioning

The call-intent -> danger FiLM design in the doc is fine. Keep it, but apply FiLM to the **residual danger block**, not the whole danger output. That preserves the factorized base.

### Safety reserve

Upgrade the doc’s reserve feature from tenpai-only to value-weighted:

[
reserve(t) = \sum_i
\mathbf{1}[t \text{ safe vs } i]
\cdot
\frac{p^{tp}_i,(1+v_i/8000)}{1+n^{safe}_i}
]

and apply

[
\ell'*a = \ell_a - \alpha*{reserve},reserve(a)
]

with `alpha_reserve = 0.15` to start.

### Failure modes

* **Correlated errors in tenpai and wait-set:**
  residual head handles this; calibrate the final danger output, not just the components.
* **Call-intent weak labels are noisy:**
  keep weight low; use it mainly as a feature.
* **Aux heads overpower policy learning:**
  use the gradient cap.

### Evaluation

Measure:

* tenpai ECE and Brier,
* wait-set PR-AUC,
* danger ECE and PR-AUC,
* “second-riichi deal-in rate,” which the reserve feature should directly reduce. ([GitHub][4])

---

## 6. Opponent archetypes: build them from data before making them from diversity regularizers

Hydra FINAL wants `N=4` archetypes for OLSS-style robust search. The lowest-risk way to get them is not fancy PBT first. It is:

1. cluster opponent behavior from logs,
2. distill one light adapter per cluster,
3. maintain an online Bayesian posterior over adapters per opponent.

### Recommendation

Use `K=4` archetypes:

* speed
* value
* defense
* balanced/aggressive

Train them as small LoRA/adapter deltas on top of the shared policy/event model.

### Offline training

Build clustering features from human logs:

* call rate,
* riichi timing,
* fold rate under threat,
* average hand value at win,
* average speed to tenpai,
* deal-in rate.

Cluster players/hanchan segments into 4 groups. Distill one adapter per group.

This is far easier than training a diversity league first, and it gives the search system a meaningful style bank immediately.

### Online posterior

For opponent `j`:

[
\log w_{j,i}^{(t+1)}
====================

0.98,\log w_{j,i}^{(t)}
+
\log p_i(e_t\mid I_t)
]

then normalize.

The `0.98` factor stops a few odd events from instantly collapsing the posterior.

### Why this is better than starting with diversity RL

It gives you:

* interpretable styles,
* immediate search utility,
* easier debugging,
* fewer moving parts.

Population diversity is still useful later, but it is not the fastest path to a useful OLSS-style search bank.

---

## 7. Endgame exactification: pure one-sequence PIMC is the fallback, not the final answer

### What is underspecified

Hydra FINAL proposes pure PIMC in wall-small states because full expectimax is too slow. I would not stop there. The key late-game gain comes from **exact chance on our future draws**, not from pretending opponent behavior becomes exact. ([GitHub][1])

### Recommendation: two-level solver

#### Level 1: `EndgameLite`

Trigger when:

* `wall <= 10`, and
* any of:

  * opponent riichi,
  * `max p_tenpai > 0.65`,
  * `min_safe_tile_risk > 0.12`,
  * or orasu / south-4 high-swing context.

Algorithm:

* keep top-mass particle subset covering `95%` mass, capped at `64`,
* exact branch on our future draws,
* robust branch on opponent public events,
* use GRP/CVaR leaf utility.

#### Level 2: `EndgameExact`

Trigger when:

* `wall <= 6`, or
* orasu with `|score_gap_to_next_rank| <= 8000`.

Algorithm:

* exact enumeration over all live-wall draws with nonzero count,
* no residual draw bucket,
* terminal utility exact when the hanchan can end this kyoku,
* otherwise leaf to GRP.

### Leaf utility

Normal mode:

[
U = \mathbb E[\text{placement points}]
]

South-round tail-control mode:

[
U = (1-\beta),\mathbb E[\text{placement points}] + \beta,\text{CVaR}_\alpha
]

Good starting schedule:

* East / early South: `beta=0`
* South 1-3 while leading by `>8000`: `alpha=0.2, beta=0.25`
* South 4 while in 1st: `alpha=0.1, beta=0.4`
* South 4 while 2nd within `8000`: `alpha=0.4, beta=0.2`

This is fully consistent with the GRP/CVaR direction in the opponent-modeling doc, but I would keep it inference-only at first. ([GitHub][4])

### Cache key

Cache endgame states by:

* canonicalized own hand,
* live-wall count vector,
* dead-wall hidden count vector,
* visible riichi/call state,
* turn index,
* score context hash.

### Failure modes

* **Trying to exactify opponent behavior too:** runtime blowup.
  Exactify chance first.
* **Using too many particles late:** duplicated work.
  Use top-mass particle subset only.
* **No exact terminal utility in orasu:** you leave the biggest gain on the table.

### Evaluation

Use a dedicated last-10-draw suite and report:

* deal-in rate,
* win conversion,
* 1st/4th swing,
* mean placement delta.

Hydra FINAL already asks for a 50k endgame comparison; keep that, but run it as duplicate paired evaluation. ([GitHub][1])

---

## 8. ExIt, SaF, and pondering: add trust gates or they will teach noise

### Recommendation

Search should not supervise the learner unconditionally.

Define a **search trust weight**:

[
\lambda_{exit}=
clip!\left(\frac{N_{root}-64}{256-64},0,1\right)
\cdot
clip!\left(\frac{m_{expanded}-0.85}{0.10},0,1\right)
\cdot
\exp(-\sigma_{top1}/0.15)
\cdot
clip!\left(\frac{ESS}{0.6P},0,1\right)
]

where:

* `N_root` = root visits,
* `m_expanded` = expanded branch mass,
* `sigma_top1` = std of the best action value across particles,
* `ESS/P` = posterior quality.

Then define:

[
\pi^* = normalize\Big((1-\lambda_{exit})\pi_{base} + \lambda_{exit},softmax(Q/0.25)\Big)
]

This is much safer than blindly replacing the policy target with search.

### SaF

Keep the doc’s tiny MLP, but normalize features and clip the residual logit:

[
\ell_{final}(a)=\ell_\theta(a)+clip(g_\psi(\tilde f(a)),-2,2)
]

with feature vector

* `delta_Q`
* `log(r_boole + 1e-6)`
* `log(r_hunter + 1e-6)`
* `log(r_robust + 1e-6)`
* `entropy_drop`
* `log(tau_robust + 1e-6)`
* `log(var_Q + 1e-6)`
* `log(ESS/P + 1e-6)`

Train in two stages:

1. Huber regression to `log pi_search - log pi_base`
2. joint fine-tune with the policy

Use SaF dropout `0.3`, as the SSOT suggests. ([GitHub][1])

### Hard-state gate for running search

Run deep search only if at least two of these are true:

* top-2 policy gap `< 0.10`
* best safe discard risk `> 0.08`
* `ESS/P < 0.55`
* `wall <= 12`
* South 4 / orasu

Otherwise use network + Hand-EV + opponent heads.

### Pondering queue

Score predicted child states by:

[
priority(child)=
p(child)\cdot
\Big(0.5+\sigma_Q+0.5,risk+0.25,placement_swing+0.25,\mathbf{1}_{cache_miss}\Big)
]

Maintain the top `64` queued future states.

### Why

Hydra FINAL explicitly makes pondering label-amplification central, and Suphx’s runtime adaptation result is a reminder that online adaptation can help, but only if it is compute-selective. ([GitHub][1])

---

## 9. Module boundaries and interfaces

The current engine is `85 x 34` with a 46-action space, but the docs are explicit that the target architecture should become a **fixed-shape superset** with zero-filled dynamic planes and presence masks. Keep that invariant. Do not let search/belief features change model shape at runtime. ([GitHub][2])

### Core interfaces

```rust
struct BeliefSummary {
    marginals: [[f32; 4]; 37],   // L, A, R, W
    mixture_w: [f32; 8],         // max size, mask unused
    entropy_zone: [f32; 4],
    ess: f32,
    signature: u64,
}

struct HiddenParticle {
    counts: [[u8; 4]; 37],
    deadwall: [u8; 37],
    arch_post: [[f32; 4]; 3],    // 3 opponents x 4 archetypes
    logw: f32,
}

struct OpponentReadout {
    p_tenpai: [f32; 3],
    p_wait: [[f32; 37]; 3],
    value_bins: [[f32; 7]; 3],
    intent: [[f32; 8]; 3],
    p_danger: [[f32; 37]; 3],
    reserve: [f32; 37],
}

struct HandEvFeatures {
    p_tenpai: [[f32; 3]; 37],
    p_win: [[f32; 3]; 37],
    e_score: [f32; 37],
    ukeire_vec: [[f32; 37]; 37],
}

struct SearchFeatures {
    delta_q: [f32; 46],
    risk_boole: [f32; 46],
    risk_hunter: [f32; 46],
    risk_robust: [f32; 46],
    entropy_drop: [f32; 46],
    tau_robust: [f32; 46],
    var_q: [f32; 46],
    ess: f32,
}
```

### Data flow

`public state -> backbone -> {policy, value, grp, tenpai, wait-set, value threat, intent} -> BeliefSummary -> CT-SMC particles -> HandEv + AFBS -> SearchFeatures -> SaF adaptor / ExIt target builder`

That keeps each subsystem separately testable.

---

## 10. Practical implementation order

This is the order I would give the coding agent, even though it slightly reorders the roadmap.

### Stage A

Implement the **dense opponent targets** first:

1. tenpai
2. wait-set
3. value-conditioned threat from current exact waits
4. factorized danger residual

Reason: search quality depends on these.

### Stage B

Implement the **37-type CT-DP sampler** and the **belief likelihood net**, but do **not** add full search yet.

Pass gate:

* better posterior NLL than simple mean-field / CMPS baseline,
* acceptable ESS profile.

### Stage C

Implement **Hand-EV** next.

Pass gate:

* positive standalone ablation over base net,
* good correlation with wall-small exact/MC rollouts.

### Stage D

Implement **shallow AFBS** over public events, but **without** archetype robustness first.

Pass gate:

* duplicate paired improvement at 80–120 ms.

### Stage E

Add **archetype adapters + KL robust backup**.

Pass gate:

* better held-out-opponent robustness,
* no collapse against balanced anchors.

### Stage F

Add **SaF**, then **trust-gated ExIt**.

Pass gate:

* “SaF without live search” recovers a meaningful fraction of search gain.

### Stage G

Add **EndgameLite**, then **EndgameExact**.

Pass gate:

* positive endgame-suite deltas.

### Stage H

Only after all of that, revisit:

* population-diversity training,
* richer CVaR schedules,
* deception/RSA,
* domain adaptation.

This order is closer to “cheap strength first, expensive strength second.”

---

## 11. Risk assessment

### Strongest bets

These are the ones I would commit to:

* **37-type CT-SMC with event likelihood filtering**
* **public-event AFBS**
* **Hand-EV before deep search**
* **factorized tenpai/wait/value/danger**
* **duplicate-seed evaluation**

Those are the highest-strength-per-engineering-hour items.

### Medium-risk but worth doing

* **archetype adapter bank**
* **KL-robust opponent backups**
* **SaF + trust-gated ExIt**
* **two-level endgame solver**

These can swing strength, but only after the foundations are real.

### Fragile or likely not worth the complexity yet

* a **second standalone Sinkhorn/tile-allocation head**
* **RSA/deception** as an RL objective
* **state-conditioned lambda** beyond the existing reward/value context
* **full domain-adversarial regularization**
* **training-time CVaR optimization** before inference-time CVaR proves useful

The docs themselves already flag some of these as future/deferred ideas; I agree with that ranking. ([GitHub][4])

### Bottom line

If I had to bet on where Hydra can get a real LuckyJ-level jump, it is not from a more exotic optimizer. It is from this stack:

1. **belief that respects tile conservation and actual public behavior,**
2. **search over public events with robust opponent nodes,**
3. **cheap offensive look-ahead features,**
4. **a factorized, calibrated threat model,**
5. **late-game exact chance control,**
6. **strict duplicate evaluation.**

That is the part of the design that is both most underspecified and most likely to produce a genuine strength breakthrough if implemented cleanly. ([GitHub][1])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[3]: https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf "https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf"
[4]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md"
</start-answer>