# HYDRA: A Maximum-Ceiling 4-Player Riichi Mahjong AI

**Promoted architecture doctrine summary.** This doc = Hydra architecture north star after canonical-archive SSOT filter + current repo/code validation. Supersedes prior internal variants: throughput-first "compute-constrained elegance" and "information-geometric / all-out". Keeps best parts, removes ceilings, adds grounded robustness layer.

This file owns target architecture, not live repo status board. Current shipped/staged status: read `docs/CURRENT_STATUS.md`. Active-path / staged-vs-reserve execution: read `research/design/HYDRA_RECONCILIATION.md`. Runtime compatibility/runtime reality: read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.

---

## 0. Abstract

4-player Riichi Mahjong = large general-sum imperfect-information game with **finite shared hidden pool** (multivariate hypergeometric), **hard conservation constraints**, and **decision-critical correlations** that grow late game.

Hydra centered on one engine:

> **ExIt + Pondering + Search-as-Feature (SaF)**
> Deep anytime belief-search continuously generates training targets during self-play, amplified by opponent-turn idle time; targets amortized back into policy/value nets so inference stays fast.

System couples engine with:

1. **Belief correctness with constraints**: SIB / Mixture-SIB (Sinkhorn KL projection) + **CT-SMC exact contingency-table sampler** using Mahjong small row counts ($r \le 4$) for correlation-faithful beliefs via 3,375-state DP (~4M ops, <1ms in Rust).
2. **Anytime Factored-Belief Search (AFBS)**: top-k pruning, heavy caching, incremental reuse, predictive pondering, and **endgame exactification** (exact chance enumeration when wall $\le 10$).
3. **Robust opponent modeling inside search**: opponent nodes solved as distributionally robust soft-min inside KL uncertainty set around learned opponent policy.
4. **Conservative safety math tight enough to matter**: Negative dependence / Strongly Rayleigh + Hunter/Kounias union tightening + bounded-error Monte Carlo intersections.
5. **Hand-EV oracle features**: CPU-precomputed per-discard tenpai probability, win probability, expected score, and ukeire; Suphx showed biggest practical gain.
6. **ACH training** (Actor-Critic Hedge, LuckyJ algorithm): +0.4 fan over PPO via Hedge-derived conservative clipping. Global $\eta$, per-(s,a) gating, standard GAE, one epoch per batch. Compatible with oracle guiding via CTDE.
7. **Two-tier network** (12-block actor / 24-block learner): 40-block teacher too data-starved at 7 spp on hard states only. 24-block learner (245 spp) handles training + deep AFBS. Continuous distillation learner -> actor.

Goal: **maximize expected Tenhou stable rank**; LuckyJ 10.68 stable dan = current public benchmark.

---

## 1. Design principles

### P1. Ceiling first, then amortize
If mechanism raises ceiling but too slow at inference, put it in pondering, deep search, offline solvers, or distillation targets, not critical inference loop.

### P2. Search targets must optimize information state, not hidden state
Any training target for deployable policy must be function of public/information state, not privileged knowledge. Perfect-information nets allowed for variance reduction and diagnostics, but improvement operator must respect information constraints.

### P3. Every "guarantee-like" claim must be theorem, bound, or empirical gate
Each guarantee-like claim must be theorem (with conditions), bound (with explicit constants), or empirical gate with measurable pass/fail threshold.

### P4. Robustness is not optional in 4-player general-sum
Instead of equilibrium-style guarantees (not clean in 4p), use distributional robustness: robust to belief error, opponent policy misspecification, and population shift.

---

## 2. Game model and notation

- Tile types: $k \in \{1,\dots,34\}$, multiplicity 4, total 136 tiles.
- Hidden locations: $z \in \{1,2,3,W\}$: three opponent concealed hands + wall remainder.
- Public information state at time $t$: $I_t$ (our hand, discards/melds, riichi, dora, scores, round meta).
- Remaining tile counts: $r_t(k) = 4 - \mathrm{visible}_t(k)$.
- Hidden location sizes: $s_t(z) \in \mathbb{Z}_{\ge 0}$, $\sum_z s_t(z) = \sum_k r_t(k)$.
- Hidden allocation matrix: $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$, $\sum_z X_t(k,z) = r_t(k)$, $\sum_k X_t(k,z)=s_t(z)$.

Under purely random dealing, $X_t$ is multivariate hypergeometric; under strategic play, $p(X_t\mid I_t)$ is shaped by action likelihoods.

---

## 3. System overview -- four interacting loops

**Loop Belief loop** -- Mixture-SIB for fast marginal updates under constraints, particle SMC for joint correlation capture.

**Loop B: Search loop** -- AFBS on $I_t$ with belief $q_t$: on-turn (shallow, feature-producing), off-turn/pondering (deep, cached, predictive).

**Loop C: Distillation loop** -- Train policy/value to predict $\pi^{\text{ExIt}}$, $V^{\text{ExIt}}$, and calibrated safety features.

**Loop D: Population loop** -- League with self-play variants, human-style anchors, adversarial exploiters.

---

## 4. Neural architecture

### 4.1 Input tensor

**Group -- Public encoding (~80-120 planes):** Hand, ordered discards (recency), open melds, riichi state, dora, round/scoring context, shanten/uke-ire summaries.

**Group B -- Safety planes (~23 planes):** Tenpai hints, furiten, genbutsu/suji/kabe safe-tile masks.

**Group C -- Search and belief features (dynamic, ~60-200 planes):** Belief marginals $B_t(k,z)$, mixture weights/entropy/ESS, AFBS action deltas $\Delta Q(a)$, risk estimates, robust opponent stress indicators. Zeroed with presence mask when unavailable.

**Group D -- Hand-EV oracle features (~34-68 planes, CPU-precomputed):** For each discard candidate $a$ (34 tile types), precompute look-ahead analysis on existing 42-plane interface:
- $P_{\text{tenpai}}^{(d)}(a)$: probability of reaching tenpai within $d \in \{1,2,3\}$ self-draws.
- $P_{\text{win}}^{(d)}(a)$: probability of winning within $d$ draws (tsumo + simplified ron model).
- $\mathbb{E}[\text{score} \mid \text{win}, expected hand value (han/fu/score) if we win after discarding $a$.
- Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.

These features computed by CPU-side hand analyzer (`shanten_batch.rs` + scoring engine) using belief-weighted remaining tile counts from CT-SMC. Zero GPU cost; CPU precomputes during game-step processing. Suphx reported these look-ahead features as single biggest practical gain (Li et al. 2020).

Runtime reality note: live repo already carries same 42-plane Hand-EV surface through `HandEvFeatures`, bridge code, and encoder channels. Runtime bridge uses public remaining counts by default and CT-SMC wall-weighted remaining counts when search context present. For shipped/staged status of surface, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md`.

### 4.2 Two-tier architecture

**Why not monolithic 40-block?** At 2000 GPU hours, self-play yields ~2.45B decisions (35M games). Samples-per-parameter ratio:

| Config | Params | Samples/param | vs Mortal (514) | Verdict |
|--------|-------:|-------------:|----------------:|---------|
| 40-block mono | 16.5M | 148 | 0.29x | Undertrained AND too slow for rollouts |
| 24-block | 10M | 245 | 0.48x | Viable with ExIt quality boost |
| 12-block | 5M | 490 | 0.95x | Well-trained, fast inference |

(Based on ~35M games * 70 decisions = 2.45B total samples.)

40-block teacher trained only on hard states (1-5%) gets ~7 spp; catastrophic data starvation. **Two-tier architecture avoids paradox:**

| Network | Blocks | Params | Role | Runtime placement |
|---------|-------:|-------:|------|-------------------|
| **LearnerNet** | 24 | ~10M | Training (ACH/ExIt) + deep AFBS on hard positions | Main Delta A100 training resources |
| **ActorNet** | 12 | ~5M | Self-play data generation + shallow SaF features | Fast rollout / self-play generation resources |

All use SE-ResNet with GroupNorm(32) and Mish. Target deploy precision = bf16-capable, but current repo remains fp32-first unless backend autocast wired explicitly. **Continuous distillation**: Learner -> Actor (every 1-2 minutes, IMPALA-style). ActorNet inference: ~0.2ms. LearnerNet inference: ~0.35ms. LearnerNet runs deeper AFBS only on hard-position ExIt labels when throughput budget allows.

### 4.3 Heads (multi-task)

**Core decision heads:** (1) Policy $\pi_\theta(a\mid I_t)$, 46 actions. (2) Value $V_\theta(I_t)$, scalar. (3) Score distribution: pdf + cdf (64 bins, KataGo-style).

**Opponent and safety heads:** (4) Opponent tenpai (3 sigmoids). (5) Opponent next discard (3x34). (6) Danger: per-tile deal-in probability (3x34).

**Belief heads:** (7) Mixture-SIB external fields $F_\theta^{(\ell)}(k,z)$ and mixture weight logits. (8) Opponent hand-type latent predictor.

**Search distillation heads:** (9) $\Delta Q$ regression (predict search advantage over baseline). (10) Safety bound residual (predict conservatism gap).

Runtime reality note: live model already exposes these advanced output families structurally in one output contract (`belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`). For which surfaces are shipped baseline vs implemented-but-staged vs implemented-but-not-default-on, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md`, not this architecture file.

---

## 5. Belief inference: SIB, Mixture-SIB, and particle posterior

### 5.1 SIB as KL projection

Let

$$K_\theta(k,z)=\exp(F_\theta(k,z))>0$$

transportation polytope is

$$\mathcal{U}(r_t,s_t)=\{B\ge 0: B\mathbf{1}=r_t, B^\top \mathbf{1}=s_t\}$$

**SIB operator:**

$$\mathrm{SIB}(K_\theta;r_t,s_t):= \arg\min_{B\in\mathcal{U}} D_{\mathrm{KL}}(B\|K_\theta)$$

Sinkhorn-Knopp gives solution

$$B^*=\mathrm{diag}(u)\cdot K_\theta\cdot\mathrm{diag}(v)$$

### 5.2 Mixture-SIB for multimodality

With $L$ components, mixture posterior is

$$q_t(X)=\sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X)$$

Each component marginal is

$$B_t^{(\ell)}=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t)$$

Weight update (Bayes):

$$w_{t+1}^{(\ell)}\propto w_t^{(\ell)} \cdot p_\phi(e_t\mid I_t, B_t^{(\ell)}, \ell)$$

Here, $e_t$ = observed public event (opponent discard, call, riichi, or pass). Anti-collapse via entropy regularizer, split-merge on low ESS, and diversity penalty between components.

### 5.3 Particle posterior (SMC) for joint structure

Particles $\{X_t^{(p)},\alpha_t^{(p)}\}_{p=1}^P$ target $p(X_t\mid I_t)$. Proposal via constrained sequential fill guided by mixture component. Resample when $\mathrm{ESS}<0.4P$. Rejuvenate via Metropolis-Hastings swap moves preserving row/col sums.

### 5.4 Correlation scale diagnostic

Correlation scale is

$$|\rho_{ij}|=\sqrt{K_i K_j} / \sqrt{(H-K_i)(H-K_j)}$$

At $H=50$ and $K=4$, $|\rho|=4/46=0.087$. At $H=25$, $|\rho|=0.190$. Late-game correlations motivate Mixture-SIB + particles over first moment alone.

### 5.5 CT-SMC: Exact contingency-table sampling (replaces generic particle proposals)

Hidden allocation $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$ = **fixed-margin contingency table**. Key Mahjong insight: each row sum $r_t(k) \le 4$, so per-row compositions tiny ($\binom{r+3}{3} \le 35$).

**Exact DP partition function.** Order tile types $k=1,\dots,34$. Let residual capacities be $\mathbf{c}=(c_1,c_2,c_3,c_W)$. Define:

$$Z_k(\mathbf{c}) = \sum_{x \in \mathcal{X}_k(\mathbf{c})} \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x), \quad Z_{35}(\mathbf{0})=1$$

Learned field weight for each row is

$$\phi_k(x)=\prod_j \omega_{kj}^{x_j}$$

Wall residual derived from other capacities:

$$c_W = R_k - (c_1+c_2+c_3)$$

Here, $R_k = \sum_{t \ge k} r_t$ = remaining hidden tile count at DP step $k$. So DP state = 3D: $(c_1,c_2,c_3)$. State count: $\le (15)^3 = 3{,}375$ (max 14 tiles after draw, before discard). Each transition enumerates $\le 35$ compositions. Total: $\sim 34 \times 3375 \times 35 \approx 4.0M$ ops; **trivially sub-millisecond in Rust**. Use log-space DP for numerical stability.

**Exact backward sampling:**

$$p(x_k = x \mid \mathbf{c}) = \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x) / Z_k(\mathbf{c})$$

This gives **exact samples with correct correlations** from conservation-constrained distribution, not mean-field approximations.

**SMC integration.** Full posterior is

$$p(X \mid \mathcal{O}_{1:t}) \propto p_0(X) \cdot L(X)$$

Here, $L(X)$ = opponent action likelihood. Sample $X^{(n)} \sim p_0$ via CT-DP (fast, correlation-correct), then assign weights with

$$w^{(n)} \leftarrow L(X^{(n)})$$

Normalize and resample. Proposal already respects hardest constraint (tile conservation) exactly, so ESS stays high.

**What CT-SMC replaces:** generic particle proposal from Section 5.3. Mixture-SIB stays as fast amortized belief head for network input; CT-SMC = search-grade belief for AFBS and safety queries.

**Validation gates:**
- **Gate (posterior log-likelihood):** At hand end, evaluate $\log p(X^* \mid \mathcal{O}_{1:t})$ under CT-SMC vs generic CMPS. CT-SMC must win.
- **Gate B (pairwise MI calibration):** Compare estimated mutual information between whether tile $A$ is in hidden hand $z$ and whether tile $B$ is in hidden hand $z$ against empirical values. Must capture correlations generic CMPS misses.

---

## 6. Conservative safety estimates without over-folding

### 6.1 Strongly Rayleigh / negative dependence foundations
Remaining-tile distribution under "draw without replacement" is Strongly Rayleigh (BBL 2009), implying strong negative dependence. Use only for bounding monotone danger events.

### 6.2 Hunter bound (spanning tree correction)
For threat events $A_1, \ldots, A_J$ and any spanning tree $T$:

$$P\left(\bigcup_{j=1}^{J} A_j\right) \le \sum_{j=1}^{J} P(A_j) - \sum_{(u,v)\in T} P(A_u \cap A_v)$$

Maximum-weight spanning tree gives tightest bound. Kounias (1968) bound is member; take minimum computable bound.

### 6.3 Computing intersections reliably
Use analytic formulas for simple events; particle estimates with Hoeffding CIs otherwise. Never use intersection estimate unless CI half-width $<\delta_\cap$ (e.g., 0.01). Fall back to conservative Boole if CI fails.

---

## 7. Anytime Factored Belief Search (AFBS)

### 7.1 Tree structure
Node state: $(I, \mathcal{B}, \mathcal{P})$ -- info state, Mixture-SIB summary, particle set handle.

### 7.2 Beam parameters

| Mode | Beam W | Depth D | Particles P | Mixture L |
|------|-------:|--------:|------------:|----------:|
| On-turn | 64-128 | 4-6 | 128-256 | 4-8 |
| Ponder | 256-1024 | 10-14 | 1024-4096 | 8-32 |

### 7.3 Caches
Transposition table (public hash + belief signature), neural eval cache (batched GPU, LRU), Sinkhorn warm-start cache (`u,v` scalings), predictive ponder cache (subtrees for top-M predicted opponent actions).

### 7.4 Incremental reuse across turns
On event: lookup predicted child key; if match, shift root and keep statistics; else reuse TT/NN cache and rebuild shallow frontier.

### 7.5 Endgame exactification (wall-small solver)

Runtime reality note: live repo implements selective particle-weighted PIMC shell here, not full exact multiplayer endgame solver. Keep exactification as target direction; defer current shipped/staged status to `docs/CURRENT_STATUS.md` and runtime semantics to `docs/GAME_ENGINE.md`.

**Trigger:** Activate when remaining wall is 10 tiles or fewer and at least one threatening signal exists (riichi, open tenpai, high-tempo opponent).

**PIMC with top-k draw pruning.** Full Expectimax over wall=10 too slow (~661K paths per particle at 0.1ms each = 66s). Instead use **Pure PIMC**: for each CT-SMC particle, sample ONE draw sequence (weighted by hypergeometric probabilities) and ONE opponent action sequence (from ActorNet policy). Average over P particles. This reduces to P forward passes per endgame eval. With top-mass particle reduction (keep particles covering 95% weight, typically P=50-100): **5-10ms per decision**, within budget. Top-k draw pruning (branch only on 2-3 most likely draws at our nodes) gives middle ground between PIMC and full Expectimax when more precision needed.

$$Q(a) \approx \frac{1}{P}\sum_{p=1}^{P} PIMC(a \mid X^{(p)})$$

Inner value exact over wall draws; opponent actions still modeled by robust policy (KL ball). This removes chance-uncertainty variance at most sensitive game phase (oorasu placement swings).

**Caching.** Late-game states repeat structurally across particles. Cache by: our hand canonicalization + remaining wall multiset signature (34-count vector) + riichi state + turn index. DP results reused heavily.

**Why this matters:** Late-game decisions are disproportionately high-EV. One wrong fold or push in oorasu can flip placement from 1st to 4th (~90,000 point swing in uma). Exact computation removes approximation error where cost highest.

**Validation gate:** Collect 50K endgame positions (last 10 draws). Compare deal-in rate, win conversion rate, and placement swings between standard AFBS vs endgame-exact mode. Endgame mode must improve all three.

---

## 8. Robust opponent modeling inside search

### 8.1 Opponent uncertainty set
Learned opponent policy is $p(a)$. True policy $q(a)$ lies in KL ball

$$\mathcal{Q}_\varepsilon(p)=\{q: D_{\mathrm{KL}}(q\|p)\le \varepsilon\}$$

$\varepsilon$ calibrated from data as empirical upper quantile of observed KL, bucketed by context.

### 8.2 Robust value at opponent nodes
Robust value at opponent nodes is

$$V_{\text{rob}}=\min_{q\in \mathcal{Q}_\varepsilon(p)} \sum_a q(a) Q(a)$$

Solution has form

$$q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)$$

Choose $\tau$ so $D_{\mathrm{KL}}(q_\tau\|p)=\varepsilon$.

**Contract.** For any opponent policy $q$ in KL ball, AFBS robust backup gives lower bound on expected value against $q$.

### 8.3 OLSS-style opponent strategy set
Besides continuous KL robustness, maintain $N$ discrete opponent archetypes $\{\sigma_1,\dots,\sigma_N\}$ (e.g., aggressive/defensive/speed/value, $N=4$). At opponent nodes, evaluate:
$$Q(a) = -\tau_{\text{arch}} \log \sum_{i=1}^N w_i \exp(-Q^{\sigma_i}(a)/\tau_{\text{arch}})$$

where $w_i$ = archetype weights (uniform $1/N$ initially, updated by posterior over opponent type) and $\tau_{\text{arch}}$ = archetype soft-min temperature (distinct from Section 8.2 $\tau$ found by binary search).

This soft-min over archetypes directly mirrors LuckyJ OLSS-II approach (Liu et al., ICML 2023) and hardens against wrong-opponent-model failure, dominant multiplayer-search failure mode. Archetypes trained as lightweight shared-backbone adapters during population training.

---

## 9. Search-as-Feature (SaF)

For each legal action $a$, AFBS returns: $\Delta Q(a)$, deal-in risk estimates (Boole/Hunter/robust), epistemic terms (entropy drop), robust stress ($\tau$), uncertainty (variance, ESS).

**Logit-residual policy:**

$$\ell_{\text{final}}(a)=\ell_\theta(a) + \alpha_{\text{SaF}}\cdot g_\psi(f(a))\cdot m(a)$$

Here, $m(a)\in\{0,1\}$ marks whether features are present. $g_\psi$ = tiny shared MLP (hidden dim 32-64).

**SaF-dropout:** during training, randomly zero $m$ even when features exist ($p_{\text{drop}}=0.3$) to prevent over-reliance. Train $g_\psi$ first via supervised regression on $\delta(a)=\log\pi_{\text{search}}(a)-\log\pi_{\text{base}}(a)$, then switch to joint end-to-end.

---

## 10. ExIt + Pondering as the central training engine

### 10.1 ExIt targets
Current Hydra doctrine and impl direction use masked, visit-based root-child distribution as ExIt teacher object. `root_exit_policy()` / q-softmax is not teacher object for live AFBS-generated ExIt lane.

### 10.2 Pondering = label amplification
75% idle time used for: deepen current root search + precompute searches for predicted near-future states. Every completed search yields extra labeled training examples.

### 10.3 Playout cap randomization
More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.

---

## 11. Training pipeline

### Compute budget (about 2000 GPU-hours on Delta GPU `gpuA100x4` with 1 shared A100)

| Phase | GPU-hrs | Nets trained | Games | Key output |
|-------|--------:|-------------|------:|-----------|
| Phase -1: Benchmarks | 150 | All nets | N/A | Latency/throughput/distill gates |
| Phase 0: BC | 50 | LearnerNet (24-block) | N/A (5-6M expert) | Initialize from human data |
| Phase 1: Oracle guiding | 200 | LearnerNet + oracle critic | ~5M | Oracle-calibrated beliefs/danger |
| Phase 2: DRDA-wrapped ACH | 800 | LearnerNet via ACH+DRDA | ~18M | Game-theoretic base + early ExIt |
| Phase 3: ExIt + Pondering | 800 | LearnerNet (deep AFBS on hard positions) | ~12M | Deep search ExIt + endgame |
| **Total** | **2000** |                                                                                                                                | **~35M** |                                                                                                                                |

Logical role split: training, self-play generation, and pondering/search amplification should be partitioned across available Delta A100 budget as throughput permits. Treat as workload roles, not claim of exclusive full-node use. Distillation: Learner -> Actor continuously (IMPALA-style).

### Phase -1: Hard reality benchmarks (150 GPU hours reserve)
Unlocked BEFORE full-budget commit. Must pass:
- **Latency gate**: AFBS on-turn < 150ms, CT-SMC DP < 1ms, endgame solver < 100ms
- **Throughput gate**: ActorNet self-play > 20 games/sec sustained
- **Distillation gate**: Learner->Actor KL drift < threshold over 100 updates
- **Hyperparameter sweep**: ACH eta, DRDA tau_drda, beam W, depth D, particles P
If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.

### Phase 0: BC warm start (50 GPU hours)
Train LearnerNet (24-block) on 5-6M expert games (Tenhou Houou + Majsoul). 24x augmentation (6 suit perms x 4 seat rotations). All heads supervised. Distill to ActorNet (12-block) at end.

### Phase 1: Oracle-visible supervision (200 GPU hours)
Self-play with full hidden state access. Train oracle critic under zero-sum constraint

$$\sum_i V_i = 0$$

and train belief likelihood model alongside it.

Use Suphx-style Bernoulli dropout schedule

$$\gamma_t: 1 \to 0$$

Post-oracle stability uses LR decay by $\times 0.1$ plus importance weight rejection when $\gamma_t$ reaches 0.

### Phase 2: DRDA-wrapped ACH self-play (800 GPU hours)

**DRDA-wrapped ACH**: ACH = LuckyJ inner optimizer (+0.4 fan over PPO) but theory covers only 2-player zero-sum. For 4-player stability, wrap it in DRDA multi-round structure (ICLR 2025).

Policy is

$$\pi_\theta(a|x) = \mathrm{softmax}(\ell_{\text{base}}(x,a) + y_\theta(x,a)/\tau_{\text{drda}})$$

Here, $\ell_{\text{base}}$ = frozen checkpoint, $y_\theta$ = trainable residual, and $\tau_{\text{drda}} \in \{2, 4, 8\}$ (tune via Phase -1; target median KL to base in $[0.05, 0.20]$).

**Rebase rule (CRITICAL):** Every 25-50 GPU hours, fold residual into base with

$$\ell_{\text{base}} \leftarrow \ell_{\text{base}} + y_\theta/\tau_{\text{drda}}$$

Then zero $y_\theta` and reset optimizer moments. This preserves $\pi$ exactly across boundaries and prevents double-counting accumulated regret.

ACH update (per-(s,a) sample):
$$L_\pi(s,a) = -c(s,a) \cdot \eta \cdot \frac{y(a|s;\theta)}{\pi_{\text{old}}(a|s)} \cdot A(s,a)$$

- $\eta$: global scalar hyperparameter (try $\eta \in \{1,2,3\}$), NOT state-dependent in practice
- $c(s,a) \in \{0,1\}$: per-sample gate zeroing update when ratio exceeds $1\pm\epsilon$ OR centered logit exceeds $\pm l_{\text{th}}$
- Uses **logits** $y(a)$ (not log-probs), centered by $\bar{y}(s)$ and clamped to $[-l_{\text{th}}, l_{\text{th}}]$
- Standard GAE for advantages (per-player $V_i$, $\lambda=0.95$, $\gamma=0.995$)
- **One update epoch per batch** (not PPO 3-10 epochs)
- Recommended: $\epsilon=0.5$, $l_{\text{th}}=8$, $\beta_{\text{ent}}=5\times10^{-4}$, LR $2.5\times10^{-4}$

Oracle critic provides advantages via CTDE: actor conditions on public info only. Normalize advantages per minibatch for scale stability.

**Start cheap ExIt mid-Phase 2**: From ~400 GPU hours, run shallow AFBS (depth 3-4, P=64) on 20% of states. Do not wait for Phase 3 to begin amortizing search into learner.

**Fallback:** If DRDA-wrapped ACH proves unstable, fall back to PPO with entropy 0.05-0.1.

### Phase 2 (continuous): Distill rollout net

**RolloutNet** (ActorNet-sized, 12 blocks): LuckyJ "environmental model" concept. Policy + value for fast AFBS rollouts. Distilled from LearnerNet **continuously** (not every 50h; confirmed too stale). Same input encoding. Run distillation worker on spare GPU cycles.

### Phase 3: ExIt + AFBS + Pondering (800 GPU hours)
LearnerNet runs deep AFBS for **hard positions only** (top-2 policy gap < 10%, high-risk defense, low particle ESS) when available Delta GPU A100 throughput budget allows. ExIt targets distilled into LearnerNet training loss (ACH + ExIt + SaF auxiliary regression). ActorNet updated from LearnerNet continuously.

### Population training
League: latest ActorNet, trailing checkpoints, human-style anchors (BC-heavy), adversarial exploiters.

---

## 12. Risk, information, and placement

### 12.1 Distributional value and CVaR
Score pdf/cdf heads. CVaR for "avoid 4th" objectives.

### 12.2 Information-Value Decomposition (IVD)
Full decomposition is

$$Q^{\text{total}}(I,a)=Q^{\text{inst}}(I,a)+\beta_{\text{epi}} Q^{\text{epi}}(I,a)+\xi Q^{\text{str}}(I,a)$$

Here, instrumental = score utility, epistemic = posterior entropy decrease, strategic = concealment or leakage penalty. Note $\beta_{\text{epi}}$ = epistemic weight, distinct from ACH $\eta$.

### 12.3 Primal-dual risk constraints
Constraints keep deal-in risk below $\kappa_{\text{deal}}$ and information leakage below $\kappa_{\text{leak}}$.

Dual updates use

$$\lambda \leftarrow [\lambda+\alpha(\hat{C}-\kappa)]_+$$

### DeltaQ lane runtime note

Target architecture still includes DeltaQ supervision family. For current repo maturity and promotion state, defer to `docs/CURRENT_STATUS.md` and `research/design/HYDRA_RECONCILIATION.md`, not this architecture summary.

---

## 13. Validation gates

**G0:** Does Mixture-SIB + particles + AFBS produce positive decision improvement? 200K stratified states, mean $\Delta>0$, <40% negative.

**G1:** Robustness calibration. KL deviations between opponent model and held-out opponents at 95th percentile.

**G2:** Safety bound usefulness. Hunter reduces over-folding without underestimating risk beyond CI.

**G3:** SaF amortization. Shallow search + SaF must dominate shallow search alone.

---

## 14. Deployment profile

**Fast path:** Network forward + SaF adaptor. **Slow path:** Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time. Agari guard active.

---

## 15. Heritage from prior Hydra variants

**From throughput-first plan:** Asynchronous pondering as "free" label compute, distributional value heads, oracle guiding/critic, PPO hyperparameters (entropy coeff 0.05+), double-buffered weight sync, ExIt safety valves.

**From all-out plan:** Mixture-SIB, anytime FBS, SaF, Hunter/Kounias tightening, ExIt+Pondering centrality, SR concentration.

**OMEGA additions:** CT-SMC exact contingency-table belief sampler, robust opponent nodes (KL-uncertainty soft-min + OLSS-style archetype set), hand-EV oracle features, endgame exactification, DRDA-wrapped ACH training with explicit rebase rule, 2-tier network (12/24), early ExIt from mid-Phase 2, explicit calibration gates.

**Verified ablation data (Suphx Figure 8):** SL baseline ~7.65 dan, +RL basic +0.41, +GRP +0.18, +oracle guiding +0.12. Oracle guiding alone modest; stack is what matters.

---

## 16. Limitations

1. **4-player general-sum has no clean exploitability target.** Use robustness + population training instead.
2. **Belief model misspecification** remains core risk; G0 detects early.
3. **Compute allocation**: deep AFBS expensive; depends on caching, pondering hit rate, distillation efficiency.
4. **Strategy fusion / determinization pitfalls**: particles + robust opponent nodes mitigate but do not remove all pathologies.

---

## 17. References

1. Sinkhorn, Knopp. "Doubly Stochastic Matrices." *Pacific J. Math*, 1967.
2. Hunter. "Upper Bound for Union." *J. Applied Probability*, 1976.
3. Kounias. "Bounds for Union." *Annals Math Stat*, 1968.
4. Borcea, Branden, Liggett. "SR and Geometry of Polynomials." *JAMS*, 2009.
5. Bardenet, Maillard. "Concentration for Sampling Without Replacement." *Bernoulli*, 2015.
6. Anthony, Tian, Barber. "Expert Iteration." *NeurIPS*, 2017.
7. Silver et al. "Mastering Go Without Human Knowledge." *Nature* 550, 2017.
8. Wu. "Accelerating Self-Play Learning in Go (KataGo)." *arXiv 1902.10565*, 2020.
9. Li et al. "Suphx: Mastering Mahjong with Deep RL." *arXiv 2003.13590*, 2020.
10. Li et al. "Speedup Training via Reward Variance Reduction." *IEEE CoG*, 2022.
11. Farina et al. "DRDA for Multiplayer POSGs." *ICLR*, 2025.
12. Rudolph et al. "Reevaluating PG Methods in IIGs." *arXiv 2502.08938*, 2025.
13. Kalogiannis, Farina. "PG Converge in IIEFGs." *NeurIPS*, 2024.
14. Schulman et al. "Proximal Policy Optimization." *arXiv 1707.06347*, 2017.
15. Perolat et al. "Mastering Stratego (DeepNash)." *Science*, 2022.
16. Boney et al. "Learning to Play IIGs by Imitating Oracle Planner." *IEEE Trans. Games*, 2021.
17. Abbasi-Yadkori et al. "POLITEX." *ICML*, 2019.
18. Cuturi. "Sinkhorn Distances." *NeurIPS*, 2013.
19. Chen, Diaconis, Holmes, Liu. "Sequential Monte Carlo Methods for Statistical Analysis of Tables." *JASA*, 2005.
20. Patefield. "Algorithm AS 159: Efficient Method of Generating R x C Tables with Given Row and Column Totals." *Applied Statistics*, 1981.
21. Fu et al. "Actor-Critic Hedge for Imperfect-Information Games (ACH)." *ICLR*, 2022.
22. Liu et al. "OLSS: Opponent-Limited Online Search for Imperfect-Information Games." *ICML*, 2023.