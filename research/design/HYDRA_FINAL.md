# HYDRA-OMEGA: A Maximum-Ceiling 4-Player Riichi Mahjong AI (Complexity-Free)

**Single-source-of-truth (SSOT).** This document supersedes the two prior internal variants: the throughput-first "compute-constrained elegance" plan and the "information-geometric / all-out" plan. HYDRA-OMEGA keeps their best ideas, removes their ceilings, and adds a rigorously-grounded robustness layer.

---

## 0. Abstract

4-player Riichi Mahjong is a large, general-sum, imperfect-information game with a **finite shared hidden pool** (multivariate hypergeometric), **hard conservation constraints**, and **decision-critical correlations** that strengthen late game.

HYDRA-OMEGA is built around one central engine:

> **ExIt + Pondering + Search-as-Feature (SaF)**
> Deep anytime belief-search generates training targets continuously during self-play, amplified by opponent-turn idle time; those targets are amortized back into the policy/value networks so inference remains fast.

The system couples this engine with:

1. **Belief correctness with constraints**: SIB / Mixture-SIB (Sinkhorn KL projection) + **CT-SMC exact contingency-table sampler** exploiting Mahjong's small row counts ($r \le 4$) for correlation-faithful beliefs via a 50K-state DP (<1ms in Rust).
2. **Anytime Factored-Belief Search (AFBS)**: top-k pruning, heavy caching, incremental reuse, predictive pondering, and **endgame exactification** (exact chance enumeration when wall $\le 10$).
3. **Robust opponent modeling inside search**: opponent nodes solved as distributionally robust soft-min within a KL uncertainty set around the learned opponent policy.
4. **Conservative safety math that is tight enough to matter**: Negative dependence / Strongly Rayleigh + Hunter/Kounias union tightening + bounded-error Monte Carlo intersections.
5. **Hand-EV oracle features**: CPU-precomputed per-discard tenpai probability, win probability, expected score, and ukeire -- proven by Suphx as their biggest practical win.
6. **ACH training** (Actor-Critic Hedge, LuckyJ's algorithm): +0.4 fan over PPO via Hedge-derived conservative clipping. Global $\eta$, per-(s,a) gating, standard GAE, one epoch per batch. Compatible with oracle guiding via CTDE.
7. **Two-tier network** (12-block actor / 24-block learner): 40-block teacher data-starved at 7 spp on hard states only. 24-block learner (245 spp) handles both training and deep AFBS. Continuous distillation learner -> actor.

Goal: **maximize expected Tenhou stable rank**; LuckyJ's 10.68 stable dan is the current public benchmark.

---

## 1. Design principles

### P1. Ceiling first, then amortize
If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, teacher search, offline solvers, or distillation targets -- not in the critical inference loop.

### P2. The "teacher" must optimize the information state, not the hidden state
Any training target used to update the deployable policy must be a function of the public/information state, not privileged knowledge. We allow perfect-information networks for variance reduction and diagnostics, but the improvement operator must respect the information constraints.

### P3. Every "guarantee-like" claim must be either a theorem (with conditions), a bound (with explicit constants), or an empirical gate with a measurable pass/fail threshold.

### P4. Robustness is not optional in 4-player general-sum
Instead of equilibrium-style guarantees (which do not cleanly extend to 4p), we use distributional robustness: robust to belief error, robust to opponent policy misspecification, robust to population shifts.

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

**Loop A: Belief loop** -- Mixture-SIB for fast marginal updates under constraints, particle SMC for joint correlation capture.

**Loop B: Search loop** -- AFBS on $I_t$ with belief $q_t$: on-turn (shallow, feature-producing), off-turn/pondering (deep, cached, predictive).

**Loop C: Distillation loop** -- Train policy/value to predict $\pi^{\text{ExIt}}$, $V^{\text{ExIt}}$, and calibrated safety features.

**Loop D: Population loop** -- League with self-play variants, human-style anchors, adversarial exploiters.

---

## 4. Neural architecture

### 4.1 Input tensor

**Group A -- Public encoding (~80-120 planes):** Hand, ordered discards (recency), open melds, riichi state, dora, round/scoring context, shanten/uke-ire summaries.

**Group B -- Safety planes (~23 planes):** Tenpai hints, furiten, genbutsu/suji/kabe safe-tile masks.

**Group C -- Search and belief features (dynamic, ~60-200 planes):** Belief marginals $B_t(k,z)$, mixture weights/entropy/ESS, AFBS action deltas $\Delta Q(a)$, risk estimates, robust opponent stress indicators. Zeroed with presence mask when unavailable.

**Group D -- Hand-EV oracle features (~34-68 planes, CPU-precomputed):** For each discard candidate $a$ (34 tile types), pre-compute exact look-ahead analysis:
- $P_{\text{tenpai}}^{(d)}(a)$: probability of reaching tenpai within $d \in \{1,2,3\}$ self-draws.
- $P_{\text{win}}^{(d)}(a)$: probability of winning within $d$ draws (tsumo + simplified ron model).
- $\mathbb{E}[\text{score} \mid \text{win}, a]$: expected hand value (han/fu/score) if we win after discarding $a$.
- Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.

These features are computed by the CPU-side hand analyzer (`shanten_batch.rs` + scoring engine) using belief-weighted remaining tile counts from CT-SMC. Zero GPU cost -- CPU pre-computes during game step processing. Suphx reported these look-ahead features as their single biggest practical improvement (Li et al. 2020).

### 4.2 Three-network architecture

**Why not monolithic 40-block?** At 2000 GPU hours, self-play generates ~2.45B decisions (35M games). Samples-per-parameter ratio:

| Config | Params | Samples/param | vs Mortal (514) | Verdict |
|--------|-------:|-------------:|----------------:|---------|
| 40-block mono | 16.5M | 148 | 0.29x | Undertrained AND too slow for rollouts |
| 24-block | 10M | 245 | 0.48x | Viable with ExIt quality boost |
| 12-block | 5M | 490 | 0.95x | Well-trained, fast inference |

(Based on ~35M games * 70 decisions = 2.45B total samples.)

A 40-block teacher trained only on hard states (1-5%) gets just ~7 spp -- catastrophic data starvation. **Two-tier architecture avoids this paradox:**

| Network | Blocks | Params | Role | GPU |
|---------|-------:|-------:|------|-----|
| **LearnerNet** | 24 | ~10M | Training (ACH/ExIt) + deep AFBS on hard positions | GPU 0-1 (train), GPU 3 (search) |
| **ActorNet** | 12 | ~5M | Self-play data generation + shallow SaF features | GPU 2 |

All use SE-ResNet with GroupNorm(32) and Mish. bf16 precision. **Continuous distillation**: Learner -> Actor (every 1-2 minutes, IMPALA-style). ActorNet inference: ~0.2ms. LearnerNet inference: ~0.35ms. LearnerNet runs deep AFBS on GPU 3 for hard-position ExIt labels.

### 4.3 Heads (multi-task)

**Core decision heads:** (1) Policy $\pi_\theta(a\mid I_t)$, 46 actions. (2) Value $V_\theta(I_t)$, scalar. (3) Score distribution: pdf + cdf (64 bins, KataGo-style).

**Opponent and safety heads:** (4) Opponent tenpai (3 sigmoids). (5) Opponent next discard (3x34). (6) Danger: per-tile deal-in probability (3x34).

**Belief heads:** (7) Mixture-SIB external fields $F_\theta^{(\ell)}(k,z)$ and mixture weight logits. (8) Opponent hand-type latent predictor.

**Search distillation heads:** (9) $\Delta Q$ regression (predict search advantage over baseline). (10) Safety bound residual (predict conservatism gap).

---

## 5. Belief inference: SIB, Mixture-SIB, and particle posterior

### 5.1 SIB as KL projection

Let $K_\theta(k,z)=\exp(F_\theta(k,z))>0$. The transportation polytope: $\mathcal{U}(r_t,s_t)=\{B\ge0: B\mathbf{1}=r_t, B^\top \mathbf{1}=s_t\}$.

**SIB operator:** $\mathrm{SIB}(K_\theta;r_t,s_t) := \arg\min_{B\in\mathcal{U}} D_{\mathrm{KL}}(B\|K_\theta)$. Solution: $B^*=\mathrm{diag}(u)\cdot K_\theta\cdot\mathrm{diag}(v)$ via Sinkhorn-Knopp.

### 5.2 Mixture-SIB for multimodality

$L$ components: $q_t(X)=\sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X)$, each $B_t^{(\ell)}=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t)$.

Weight update (Bayes): $w_{t+1}^{(\ell)}\propto w_t^{(\ell)} \cdot p_\phi(e_t\mid I_t, B_t^{(\ell)}, \ell)$. Anti-collapse via entropy regularizer, split-merge on low ESS, diversity penalty between components.

### 5.3 Particle posterior (SMC) for joint structure

Particles $\{X_t^{(p)},\alpha_t^{(p)}\}_{p=1}^P$ targeting $p(X_t\mid I_t)$. Proposal via constrained sequential fill guided by mixture component. Resample when $\mathrm{ESS}<0.4P$. Rejuvenation via Metropolis-Hastings swap moves preserving row/col sums.

### 5.4 Correlation scale diagnostic

$|\rho_{ij}|=\sqrt{K_i K_j} / \sqrt{(H-K_i)(H-K_j)}$. At $H=50$, $K=4$: $|\rho|=4/46=0.087$; at $H=25$: $|\rho|=0.190$. Late-game correlations motivate Mixture-SIB + particles over first-moment alone.

### 5.5 CT-SMC: Exact contingency-table sampling (replaces generic particle proposals)

The hidden allocation $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$ is a **fixed-margin contingency table**. Key Mahjong insight: each row sum $r_t(k) \le 4$, so per-row compositions are tiny ($\binom{r+3}{3} \le 35$).

**Exact DP partition function.** Order tile types $k=1,\dots,34$. Let residual capacities be $\mathbf{c}=(c_1,c_2,c_3,c_W)$. Define:

$$Z_k(\mathbf{c}) = \sum_{x \in \mathcal{X}_k(\mathbf{c})} \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x), \quad Z_{35}(\mathbf{0})=1$$

where $\phi_k(x)=\prod_j \omega_{kj}^{x_j}$ is the learned field weight per row. Key insight: $c_W = R_k - (c_1+c_2+c_3)$ is **derived**, so the DP state is 3D: $(c_1,c_2,c_3)$. State count: $\le (15)^3 = 3{,}375$ (max 14 tiles after draw, before discard). Each transition enumerates $\le 35$ compositions. Total: $\sim 34 \times 3375 \times 35 \approx 4.0M$ ops -- **trivially sub-millisecond in Rust**. Use log-space DP for numerical stability.

**Exact backward sampling:** $p(x_k = x \mid \mathbf{c}) = \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x) / Z_k(\mathbf{c})$. This gives **exact samples with correct correlations** from the conservation-constrained distribution -- not mean-field approximations.

**SMC integration.** The full posterior is $p(X \mid \mathcal{O}_{1:t}) \propto p_0(X) \cdot L(X)$ where $L(X)$ is the opponent action likelihood. Sample $X^{(n)} \sim p_0$ via CT-DP (fast, correlation-correct), weight $w^{(n)} \leftarrow L(X^{(n)})$, normalize and resample. The proposal already respects the hardest constraint (tile conservation) exactly, so ESS stays high.

**What CT-SMC replaces:** The generic particle proposal from Section 5.3. Mixture-SIB is KEPT as the fast amortized belief head for network input; CT-SMC is the search-grade belief for AFBS and safety queries.

**Validation gates:**
- **Gate A (posterior log-likelihood):** At end of hand, evaluate $\log p(X^* \mid \mathcal{O}_{1:t})$ under CT-SMC vs generic CMPS. CT-SMC must win.
- **Gate B (pairwise MI calibration):** Compare estimated $I(\mathbf{1}\{A \in H_z\}; \mathbf{1}\{B \in H_z\})$ vs empirical. Must capture correlations generic CMPS misses.

---

## 6. Conservative safety estimates without over-folding

### 6.1 Strongly Rayleigh / negative dependence foundations

The remaining-tile distribution under "draw without replacement" is Strongly Rayleigh (BBL 2009), implying strong negative dependence. Used only for bounding monotone danger events.

### 6.2 Hunter bound (spanning tree correction)

For threat events $\{A_j\}_{j=1}^J$ and any spanning tree $\mathcal{T}$: $P(\bigcup_j A_j) \le \sum_j P(A_j) - \sum_{(u,v)\in\mathcal{T}} P(A_u\cap A_v)$. Maximum-weight spanning tree gives the tightest bound. Kounias (1968) bound is a member; we take the minimum computable bound.

### 6.3 Computing intersections reliably

Analytic formulas for simple events; particle estimates with Hoeffding CIs otherwise. Never use an intersection estimate unless CI half-width $<\delta_\cap$ (e.g., 0.01). Fall back to conservative Boole if CI not met.

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

Transposition table (public hash + belief signature), neural eval cache (batched GPU, LRU), Sinkhorn warm-start cache (u,v scalings), predictive ponder cache (subtrees for top-M predicted opponent actions).

### 7.4 Incremental reuse across turns

On event: lookup predicted child key; if match, shift root and keep statistics; else reuse TT/NN cache and rebuild shallow frontier.

### 7.5 Endgame exactification (wall-small solver)

**Trigger:** Activate when remaining wall $\le W^* = 10$ tiles AND at least one threatening signal (riichi, open tenpai, high-tempo opponent).

**PIMC with top-k draw pruning.** Full Expectimax over wall=10 is too slow (~661K paths per particle at 0.1ms each = 66s). Instead, use **Pure PIMC**: for each CT-SMC particle, sample ONE draw sequence (weighted by hypergeometric probabilities) and ONE opponent action sequence (from ActorNet policy). Average over P particles. This reduces to P forward passes per endgame evaluation. With top-mass particle reduction (keep particles covering 95% weight, typically P=50-100): **5-10ms per decision**, well within budget. Top-k draw pruning (branch only on the 2-3 most likely draws at our nodes) provides a middle ground between PIMC and full Expectimax when more precision is needed.

$$Q(a) = \mathbb{E}_{X \sim p(X)} \left[ \text{ExactChanceValue}(a \mid X) \right]$$

The inner value is exact over wall draws; opponent actions remain modeled by the robust policy (KL ball). This removes chance uncertainty variance at the most sensitive game phase (oorasu placement swings).

**Caching.** Late-game states repeat structurally across particles. Cache by: our hand canonicalization + remaining wall multiset signature (34-count vector) + riichi state + turn index. DP results reused heavily.

**Why this matters:** Late-game decisions are disproportionately high-EV. A single wrong fold or push in oorasu can flip placement from 1st to 4th (~90,000 point swing in uma). Exact computation eliminates the approximation error precisely where it's most costly.

**Validation gate:** Collect 50K endgame positions (last 10 draws). Compare deal-in rate, win conversion rate, and placement swings between standard AFBS vs endgame-exact mode. Endgame mode must improve all three.

---

## 8. Robust opponent modeling inside search

### 8.1 Opponent uncertainty set

Learned opponent policy $p(a)$. True policy $q(a)$ lies in KL ball: $\mathcal{Q}_\varepsilon(p)=\{q: D_{\mathrm{KL}}(q\|p)\le \varepsilon\}$. $\varepsilon$ calibrated from data as empirical upper quantile of observed KL, bucketed by context.

### 8.2 Robust value at opponent nodes

$V_{\text{rob}}=\min_{q\in \mathcal{Q}_\varepsilon(p)} \sum_a q(a) Q(a)$. Solution: $q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)$ for $\tau$ chosen so $D_{\mathrm{KL}}(q_\tau\|p)=\varepsilon$.

**Contract.** For any opponent policy $q$ in the KL ball, AFBS's robust backup gives a lower bound on expected value against $q$.

### 8.3 OLSS-style opponent strategy set

In addition to continuous KL robustness, maintain $N$ discrete opponent archetypes $\{\sigma_1,\dots,\sigma_N\}$ (e.g., aggressive/defensive/speed/value, $N=4$). At opponent nodes, evaluate:
$$Q(a) = -\tau \log \sum_{i=1}^N w_i \exp(-Q^{\sigma_i}(a)/\tau)$$

This soft-min over archetypes directly mirrors LuckyJ's OLSS-II approach (Liu et al., ICML 2023) and hardens against "wrong opponent model" -- a dominant failure mode in multiplayer search. Archetypes are trained as lightweight shared-backbone adapters during population training.

---

## 9. Search-as-Feature (SaF)

For each legal action $a$, AFBS returns: $\Delta Q(a)$, deal-in risk estimates (Boole/Hunter/robust), epistemic terms (entropy drop), robust stress ($\tau$), uncertainty (variance, ESS).

**Logit-residual policy:** $\ell_{\text{final}}(a)=\ell_\theta(a) + \alpha_{\text{SaF}}\cdot g_\psi(f(a))\cdot m(a)$ where $m(a)\in\{0,1\}$ indicates features present. $g_\psi$ is a tiny shared MLP (hidden dim 32-64). **SaF-dropout**: during training, randomly zero $m$ even when features are available ($p_{\text{drop}}=0.3$) to prevent over-reliance. Train $g_\psi$ first via supervised regression on $\delta(a)=\log\pi_{\text{search}}(a)-\log\pi_{\text{base}}(a)$, then switch to joint end-to-end.

---

## 10. ExIt + Pondering as the central training engine

### 10.1 ExIt targets

From AFBS: $\pi^{\text{ExIt}}(\cdot\mid I)=\mathrm{Softmax}(Q(I,\cdot)/\tau_{\text{ExIt}})$.

### 10.2 Pondering = label amplification

75% idle time used for: deepening current root search + precomputing searches for predicted near-future states. Every completed search yields additional labeled training examples.

### 10.3 Playout cap randomization

More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.

---

## 11. Training pipeline

### Compute budget (2000 GPU hours on 4x RTX 5000 Ada)

| Phase | GPU-hrs | Nets trained | Games | Key output |
|-------|--------:|-------------|------:|-----------|
| Phase -1: Benchmarks | 150 | All nets | N/A | Latency/throughput/distill gates |
| Phase 0: BC | 50 | LearnerNet (24-block) | N/A (5-6M expert) | Initialize from human data |
| Phase 1: Oracle guiding | 200 | LearnerNet + oracle critic | ~5M | Oracle-calibrated beliefs/danger |
| Phase 2: DRDA-wrapped ACH | 800 | LearnerNet via ACH+DRDA | ~18M | Game-theoretic base + early ExIt |
| Phase 3: ExIt + Pondering | 800 | LearnerNet + TeacherNet | ~12M | Deep search ExIt + endgame |
| **Total** | **2000** | | **~35M** | |

GPU allocation: GPU 0-1 training (LearnerNet), GPU 2 self-play (ActorNet), GPU 3 pondering/teacher (TeacherNet). Distillation: Learner -> Actor continuously (IMPALA-style), Teacher -> Learner on hard-mined positions.

### Phase -1: Hard reality benchmarks (150 GPU hours reserve)
Unlocked BEFORE committing the full budget. Must pass:
- **Latency gate**: AFBS on-turn < 150ms, CT-SMC DP < 1ms, endgame solver < 100ms
- **Throughput gate**: ActorNet self-play > 20 games/sec sustained
- **Distillation gate**: Teacher->Learner->Actor KL drift < threshold over 100 updates
- **Hyperparameter sweep**: ACH eta, DRDA tau_drda, beam W, depth D, particles P
If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.

### Phase 0: BC warm start (50 GPU hours)
Train LearnerNet (24-block) on 5-6M expert games (Tenhou Houou + Majsoul). 24x augmentation (6 suit perms x 4 seat rotations). All heads supervised. Distill to ActorNet (12-block) at end.

### Phase 1: Oracle-visible supervision (200 GPU hours)
Self-play with full hidden state access. Train oracle critic (zero-sum constraint $\sum_i V_i = 0$) and belief likelihood model. Suphx-style Bernoulli dropout $\gamma_t: 1 \to 0$. Post-oracle stability: LR decay $\times 0.1$ + importance weight rejection when $\gamma_t$ reaches 0.

### Phase 2: DRDA-wrapped ACH self-play (800 GPU hours)

**DRDA-wrapped ACH**: ACH is LuckyJ's inner optimizer (+0.4 fan over PPO) but its theory covers only 2-player zero-sum. For 4-player stability, wrap in DRDA's multi-round structure (ICLR 2025). Policy: $\pi_\theta(a|x) = \mathrm{softmax}(\ell_{\text{base}}(x,a) + y_\theta(x,a)/\tau_{\text{drda}})$ where $\ell_{\text{base}}$ is a frozen checkpoint, $y_\theta$ is a trainable residual, and $\tau_{\text{drda}} \in \{2, 4, 8\}$ (tune via Phase -1; target median KL to base in $[0.05, 0.20]$).

**Rebase rule (CRITICAL):** Every 25-50 GPU hours: (1) fold residual into base: $\ell_{\text{base}} \leftarrow \ell_{\text{base}} + y_\theta/\tau_{\text{drda}}$, (2) zero $y_\theta$ and reset optimizer moments. This preserves $\pi$ exactly across boundaries and prevents double-counting accumulated regret.

ACH update (per-(s,a) sample):
$$L_\pi(s,a) = -c(s,a) \cdot \eta \cdot \frac{y(a|s;\theta)}{\pi_{\text{old}}(a|s)} \cdot A(s,a)$$

- $\eta$: global scalar hyperparameter (try $\eta \in \{1,2,3\}$), NOT state-dependent in practice
- $c(s,a) \in \{0,1\}$: per-sample gate zeroing update when ratio exceeds $1\pm\epsilon$ OR centered logit exceeds $\pm l_{\text{th}}$
- Uses **logits** $y(a)$ (not log-probs), centered by $\bar{y}(s)$ and clamped to $[-l_{\text{th}}, l_{\text{th}}]$
- Standard GAE for advantages (per-player $V_i$, $\lambda=0.95$, $\gamma=0.995$)
- **One update epoch per batch** (not PPO's 3-10 epochs)
- Recommended: $\epsilon=0.5$, $l_{\text{th}}=8$, $\beta_{\text{ent}}=5\times10^{-4}$, LR $2.5\times10^{-4}$

Oracle critic provides advantages via CTDE: actor conditions on public info only. Normalize advantages per-minibatch for scale stability.

**Start cheap ExIt mid-Phase 2**: From ~400 GPU hours, run shallow AFBS (depth 3-4, P=64) on 20% of states. Don't wait for Phase 3 to begin amortizing search into the learner.

**Fallback:** If DRDA-wrapped ACH proves unstable, fall back to PPO with entropy 0.05-0.1.

### Phase 2 (continuous): Distill rollout net

**RolloutNet** (ActorNet-sized, 12 blocks): LuckyJ's "environmental model" concept. Policy + value for fast AFBS rollouts. Distilled from LearnerNet **continuously** (not every 50h -- confirmed too stale). Same input encoding. Run distillation worker on spare GPU cycles.

### Phase 3: ExIt + AFBS + Pondering (800 GPU hours)

TeacherNet (40-block) activated on **hard positions only** (top-2 policy gap < 10%, high-risk defense, low particle ESS). Runs deep AFBS to generate best ExIt labels. LearnerNet trains on: ACH loss + ExIt distillation from TeacherNet + SaF auxiliary regression. ActorNet updated from LearnerNet continuously.

### Population training
League: latest ActorNet, trailing checkpoints, human-style anchors (BC-heavy), adversarial exploiters.

---

## 12. Risk, information, and placement

### 12.1 Distributional value and CVaR
Score pdf/cdf heads. CVaR for "avoid 4th" objectives.

### 12.2 Information-Value Decomposition (IVD)
$Q^{\text{total}}(I,a)=Q^{\text{inst}}(I,a)+\eta Q^{\text{epi}}(I,a)+\xi Q^{\text{str}}(I,a)$ where instrumental = score utility, epistemic = posterior entropy decrease, strategic = concealment/leakage penalty.

### 12.3 Primal-dual risk constraints
Constraints: deal-in risk below $\kappa_{\text{deal}}$, info leakage below $\kappa_{\text{leak}}$. Dual updates: $\lambda \leftarrow [\lambda+\alpha(\hat{C}-\kappa)]_+$.

---

## 13. Validation gates

**G0:** Does Mixture-SIB + particles + AFBS produce positive decision improvement? 200K stratified states, mean $\Delta>0$, <40% negative.

**G1:** Robustness calibration. KL deviations between opponent model and held-out opponents at 95th percentile.

**G2:** Safety bound usefulness. Hunter reduces over-folding without underestimating risk beyond CI.

**G3:** SaF amortization. Shallow search + SaF must dominate shallow search alone.

---

## 14. Deployment profile

**Fast path:** Network forward + SaF adaptor. **Slow path:** Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time. Agari guard always active.

---

## 15. Heritage from prior Hydra variants

**From the throughput-first plan:** Asynchronous pondering as "free" label compute, distributional value heads, oracle guiding/critic, PPO hyperparameters (entropy coeff 0.05+), double-buffered weight sync, ExIt safety valves.

**From the all-out plan:** Mixture-SIB, anytime FBS, SaF, Hunter/Kounias tightening, ExIt+Pondering centrality, SR concentration.

**OMEGA additions:** CT-SMC exact contingency-table belief sampler, robust opponent nodes (KL-uncertainty soft-min + OLSS-style archetype set), hand-EV oracle features, endgame exactification, DRDA-wrapped ACH training with explicit rebase rule, 2-tier network (12/24), early ExIt from mid-Phase 2, explicit calibration gates.

**Verified ablation data (Suphx Figure 8):** SL baseline ~7.65 dan, +RL basic +0.41, +GRP +0.18, +oracle guiding +0.12. Oracle guiding alone is modest; the stack is what matters.

---

## 16. Limitations

1. **4-player general-sum has no clean exploitability target.** We use robustness + population training instead.
2. **Belief model misspecification** remains the core risk; G0 detects it early.
3. **Compute allocation**: deep AFBS is expensive; depends on caching, pondering hit rate, distillation efficiency.
4. **Strategy fusion / determinization pitfalls**: particles + robust opponent nodes mitigate but do not eliminate all pathologies.

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
16. Boney et al. "Learning to Play IIGs by Imitating an Oracle Planner." *IEEE Trans. Games*, 2021.
17. Abbasi-Yadkori et al. "POLITEX." *ICML*, 2019.
18. Cuturi. "Sinkhorn Distances." *NeurIPS*, 2013.
19. Chen, Diaconis, Holmes, Liu. "Sequential Monte Carlo Methods for Statistical Analysis of Tables." *JASA*, 2005.
20. Patefield. "Algorithm AS 159: An Efficient Method of Generating R x C Tables with Given Row and Column Totals." *Applied Statistics*, 1981.
21. Fu et al. "Actor-Critic Hedge for Imperfect-Information Games (ACH)." *ICLR*, 2022.
22. Liu et al. "OLSS: Opponent-Limited Online Search for Imperfect-Information Games." *ICML*, 2023.
