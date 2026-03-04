# HYDRA: Compute-Constrained Elegance for 4-Player Riichi Mahjong

**A 9-technique system for LuckyJ-level play within 2000 GPU hours. Mortal-speed inference with ExIt-quality training via pondering and oracle guiding.**

> **SSOT Notice**: This document is the single source of truth for HYDRA's architecture. `TRAINING.md`, `HYDRA_SPEC.md`, and `ALLOUT_PLAN.md` describe earlier design iterations and will be updated during implementation to match this document.

---

## 0. Abstract

4-player Riichi Mahjong is a large, general-sum, imperfect-information game. Existing top systems either rely on expensive inference-time search or on pure evaluation with massive compute budgets. HYDRA takes a third route: **train like AlphaZero, run like Mortal**.

**Budget**: 2000 GPU hours on 4x RTX 5000 (667 node hours).

**Core insight**: During self-play, each agent is idle ~75% of the time (the other 3 players' turns). Pondering converts this dead time into search-informed ExIt training targets -- achieving search-quality policy improvement without any inference latency cost. The network runs a single forward pass (~0.5ms) at decision time.

This document describes a 9-technique system that replaces a previous 22-technique design. The simplification was driven by a hard constraint: at 2000 GPU hours, self-play throughput dominates all other considerations. Every millisecond of inference latency costs thousands of training games. The 9-technique system generates 10-35x more self-play games than the 22-technique system at the same budget (~35M vs ~1-3M games), and volume decisively beats per-game quality at this scale.

**Target**: beat LuckyJ (10.68 stable dan on Tenhou).

---

## 1. Design Principles

**Principle 1: Self-play throughput is king.** Every millisecond of inference latency costs thousands of training games over the full budget. A technique that improves per-game quality by 5% but halves throughput is a net loss. The 22-technique system's ~50-200ms inference made it infeasible at 2000 GPU hours.

**Principle 2: One-in, one-out.** Each technique must replace or simplify something. No stacking complexity for marginal gains. If a technique cannot justify its existence against the simplest alternative, it is cut.

**Principle 3: Train the network, not the runtime.** All intelligence is baked into weights via training. The forward pass at inference is the entire computation -- no search trees, no belief updates, no KL computations. Anything the old system computed at runtime, the new system learns to predict directly.

**Principle 4: Pondering is free compute.** Opponent idle time (~75% of game time in 4-player Mahjong) is wasted in standard self-play. Pondering converts it into ExIt training targets at zero cost to self-play throughput. This is the mechanism that lets HYDRA achieve ExIt-quality training at Mortal-speed inference.

---

## 2. Architecture

**Backbone**: 40-block SE-ResNet with GroupNorm(32) and Mish activation. ~16.5M parameters.

**Input**: 85x34 tensor.
- 62 base channels: hand, discards, melds, riichi, dora, round metadata, shanten.
- 23 safety channels: genbutsu (9), suji (9), kabe (2), tenpai hints (3).

**Stem**: Conv1d(85 -> 256, kernel=3, pad=1). Each residual block: Conv1d(256 -> 128 -> 256) with SE gate and GroupNorm. Output: feature map $\mathbf{h}_t \in \mathbb{R}^{256 \times 34}$ plus pooled $\bar{\mathbf{h}}_t \in \mathbb{R}^{256}$.

**Output heads** (8 inference + oracle critic for training):

| Head | Shape | Loss | Purpose |
|------|-------|------|---------|
| Policy | 46 actions | Cross-entropy | Action selection |
| Value | scalar | MSE | Expected game outcome |
| GRP | 24-way | Cross-entropy | Rank probability distribution |
| Tenpai | 3 sigmoids | BCE | Opponent tenpai detection |
| Danger | 3x34 | BCE | Per-tile deal-in risk |
| Opp-next | 3x34 softmax | Cross-entropy | Opponent discard prediction |
| Score-pdf | B bins softmax | Cross-entropy | Score outcome distribution |
| Score-cdf | B bins sigmoid | BCE per bin | Cumulative score distribution |

**Inference**: single forward pass, ~0.5ms on RTX 5000 in bf16 precision. No search, no belief computation, no post-processing beyond argmax. The agari guard (always take legal ron/tsumo) is the only rule-based override. bf16 provides 2x throughput over fp32 without requiring GradScaler.

---

## 3. The 9 Techniques

**1. SE-ResNet backbone (GroupNorm + Mish).** The proven architecture from Mortal and LuckyJ. SE gates provide channel attention. GroupNorm avoids batch-size sensitivity. Mish activation for smooth gradients. 40 blocks at 256 channels is the sweet spot: deep enough for complex pattern recognition, narrow enough for fast inference.

**2. Score belief distribution (pdf + cdf heads).** From KataGo (Wu 2020). Instead of a scalar value, predict the full distribution over game outcomes. The pdf head gives placement-conditional strategy ("need 1st" vs "avoid 4th"). The cdf head enables CVaR risk management: $\text{CVaR}_\alpha = \mathbb{E}[S \mid S \le F^{-1}(\alpha)]$, computed directly from sigmoid outputs. Zero additional inference cost -- just two more linear projections on the shared backbone.

**3. PPO training with 8 heads + oracle critic + multi-lambda TD.** Proximal Policy Optimization with clipped objective ($\epsilon = 0.1$), **high entropy coefficient** ($\beta_{\text{ent}} = 0.05$-$0.1$, critical for IIGs -- standard defaults of 0.01 cause catastrophic failure per Rudolph et al. 2025), LR $2.5 \times 10^{-4}$, max gradient norm 0.5. All 8 heads trained jointly. During training only, advantage estimation uses an **oracle value critic** (separate lightweight network observing all 4 hands, Li et al. 2022, RVR) with zero-sum constraint ($\sum_i V_i = 0$), dramatically reducing PPO variance. An **Expected Reward Network** (ERN, Li et al. 2022) predicts the expected reward from the T-1 state (before final tile draw), replacing the raw game outcome as the RL reward signal. This filters out last-tile luck -- a major variance source in Mahjong where one tile can flip a 30,000-point swing. Multi-lambda TD targets ($\lambda \in \{0.0, 0.2, ..., 1.0\}$) supervise the value head at multiple horizons (KataGo). Zero inference cost for all training enhancements.

**4. Pondering ExIt (core innovation).** During opponent turns in self-play (~75% of game time), run shallow PUCT search using the oracle teacher for leaf evaluation (perfect information available during self-play). The search results become ExIt training targets: $\pi^{\text{ExIt}}(\cdot \mid I) = \text{softmax}(Q^{\text{oracle}}(I, \cdot) / \tau)$. Non-blocking -- self-play continues at network speed regardless. At deployment, Sinkhorn belief + blind evaluation replaces the oracle. Detailed in Section 6.

**5. Playout cap randomization (KataGo).** Not every pondered state gets the same search depth. Some states get deep search (hard positions with ambiguous policy), others get zero search (obvious decisions). This amortizes search cost across the training run and prevents overfitting to easy positions. Following Wu (2020), the playout cap is drawn from a distribution that allocates more compute to states where the network is uncertain.

**6. SR concentration for danger calibration (zero inference cost).** The multivariate hypergeometric tile distribution is Strongly Rayleigh (BBL 2009). This yields Gaussian concentration bounds and Bernstein-Serfling finite-population bounds that are used during training to validate the danger head's calibration. No runtime cost -- purely a training-time quality gate. Detailed in Section 7.

**7. Progressive training (BC -> Oracle -> PPO -> PPO+ExIt).** Phase 1 initializes from expert data. Phase 1.5 adds oracle guiding (Suphx's most impactful technique: train with perfect-info features, gradually mask). Phase 2 builds self-play strength. Phase 3 adds pondering ExIt targets. Each phase builds on the previous.

**8. Oracle guiding.** Suphx (Li et al. 2020) reached 8.74 stable rank WITHOUT search, ~1.7 dan above Mortal, primarily from oracle guiding. During early self-play, provide perfect-information features (opponents' actual hands + wall composition) as extra input channels, masked via Bernoulli dropout $\gamma_t$ decaying from 1 to 0. Critical: apply learning rate decay ($\times 0.1$) and importance weight rejection when transitioning from oracle to normal agent (without these tricks, post-oracle training is unstable -- Li et al. 2020). Zero inference cost (oracle channels removed after training).

**9. Auxiliary target distillation.** The opp-next head learns Glosten-Milgrom-style discard inference implicitly from training data -- no runtime GM computation needed. The danger head learns SR-calibrated safety from training-time validation. The score heads learn distributional value estimates. All of this intelligence is distilled into weights, not computed at runtime.

**Optional: "Think harder" trigger.** When the top-2 policy gap is less than 5% (the network is genuinely uncertain), run a 50ms shallow search to break the tie. This fires on approximately 10% of decisions and adds negligible average latency (~5ms amortized). This is the only runtime search, and it is optional -- the system works without it.

---

## 4. What Was Ablated

| Removed Technique | Why Removed | Replacement |
|---|---|---|
| Runtime FBS search (50-200ms/decision) | Kills self-play throughput; 10-35x fewer games | Pondering ExIt (asynchronous, non-blocking) |
| Mixture-SIB runtime belief (10-50ms) | Adds inference latency for marginal belief quality | Network learns beliefs implicitly from training |
| Glosten-Milgrom discard inference | Extra computation per discard observation | Opp-next head learns GM patterns from data |
| IVD / EFE decomposition | Requires runtime KL computation | Network learns unified Q-value directly |
| Rao-Blackwellized belief decomposition | Complex sampling machinery during inference | Oracle evaluation during training; Sinkhorn for deployment pondering |
| Adaptive leaf evaluation | No search tree at runtime = no leaves to evaluate | N/A (no runtime search) |
| Search-as-feature amortization | No runtime search = no search features to amortize | Network output is the only feature |
| Oracle distillation phase | Extra training phase with diminishing returns | BC warm-start feeds directly into PPO |
| DRDA-M / ACH dynamics | Convergence theory, but adds training complexity | Standard PPO self-play (empirically sufficient) |
| POT (population training) | 30% human-opponent mix dilutes self-play signal | Pure self-play; evaluate against humans post-training |
| Hard position mining | Oversampling adds data pipeline complexity | Playout cap randomization handles this naturally |
| Progressive model scaling | Small-to-large curriculum adds engineering burden | Train full 40-block model from the start |
| Nested bottleneck blocks | Marginal parameter efficiency gain | Standard residual blocks (simpler, proven) |
| 24x data augmentation (full) | Full 24x only needed for BC; during self-play, 6x suit permutation applied to replay buffer at negligible cost | 6x suit augmentation during PPO (tensor-level, ~0 overhead) |

**The argument**: At 2000 GPU hours, the 22-technique system achieves ~1-3M self-play games with rich per-game signal. The 9-technique system achieves ~35M self-play games with simpler per-game signal. Per 4/5 expert evaluations, the volume advantage decisively wins.

---

## 5. Compute Budget

| Phase | GPU Hours | Games | Purpose |
|---|---|---|---|
| Phase 1: Behavioral Cloning | 50 | N/A (5-6M expert games) | Initialize network from human expert data |
| Phase 1.5: Oracle Guiding | 200 | ~5M | Train with perfect-info features, then mask |
| Phase 2: PPO Self-Play | 750 | ~17M | Build self-play strength at full throughput |
| Phase 3: PPO + Pondering ExIt | 1000 | ~13M | Add search-informed ExIt training targets |
| **Total** | **2000** | **~35M** | |

**Throughput estimate**: With 4 GPUs split (2 training, 1 self-play, 1 pondering). RTX 5000 Ada provides 261 TFLOPS bf16 tensor throughput. Running hundreds of games in parallel and batching inference (batch size 256+), the self-play GPU achieves ~500-1000 games/second in ideal conditions. Conservatively estimating ~100 games/second after accounting for data pipeline overhead, GPU-CPU coordination, and training data writing. Phase 1 (BC): 50 GPU hours, 24 epochs over 5-6M expert games. Phase 1.5 (oracle): 200 GPU hours, ~5M games. Phase 2 (PPO): 750 GPU hours = 188 wall hours at 25 games/sec = ~17M games. Phase 3 (ExIt): 1000 GPU hours = 250 wall hours, slightly slower due to pondering sync = ~13M games. **Total: ~35M self-play games.** Note: BC and oracle phases can use all 4 GPUs for training (no self-play needed), so their wall time is 4x faster than listed GPU hours.

---

## 6. Pondering ExIt

This is HYDRA's central mechanism. It solves the fundamental tension in game AI: search improves decisions but costs inference time, which reduces the number of self-play games, which reduces training quality.

### 6.1 The idle time opportunity

In 4-player Mahjong, each player acts on roughly 25% of turns. The remaining 75% is opponent turns where the agent waits. In standard self-play, this time is wasted. HYDRA's pondering system uses it productively.

During each opponent turn, the pondering thread:

**During self-play training** (the primary mode, ~2000 GPU hours): The simulator has full game state. No Sinkhorn or hand sampling needed.
1. Runs PUCT search directly: expand top-K=5 actions, simulate opponent responses from the oracle policy (which knows all hands). Evaluate resulting states with the **oracle teacher** on GPU 3 using perfect information. Depth 2, ~100 leaf evaluations (~1ms batched).
2. Produces $Q^{\text{oracle}}(I, a)$ -- strictly better than blind evaluation. The oracle model on GPU 3 shares the same backbone as the blind student but with extra input channels for opponent hands + wall (same architecture as Phase 1.5 oracle guiding). It is continuously updated alongside the student via weight sync. This is CTDE: centralized evaluation for training, decentralized execution at deployment.

**During deployment** (Tenhou, evaluation): Sinkhorn belief inference (10-30 iterations, ~2ms) computes opponent hand marginals. Hands are sampled for PIMC-style search. The blind student evaluates leaves. Note: PIMC is subject to strategy fusion bias (over-committing to actions that require hidden knowledge). Mitigated by averaging over many hand samples. Future: consider EPIMC (Ackermann et al. 2024) for postponed resolution.

### 6.2 ExIt target generation

The search results become training labels via softmax:

$$\pi^{\text{ExIt}}(\cdot \mid I) = \text{softmax}\!\left(\frac{Q^{\text{search}}(I, \cdot)}{\tau}\right)$$

where $\tau$ is a temperature parameter controlling target sharpness. These ExIt targets replace the network's own policy as training labels for the states where pondering was performed. **KL-prioritized weighting** (enabled after 50 GPU hours of Phase 3, not at start): the ExIt loss for each state is weighted by $w(I) = \mathrm{KL}(\pi^{\text{ExIt}}(\cdot|I) \| \pi_\theta(\cdot|I))$. States where search most disagrees with the network receive higher weight. **Safety valve**: discard any ExIt target where the search value is worse than the network's own value by more than $\epsilon$ (i.e., $Q_{\text{search}}(\text{best}) < V_\theta - \epsilon$), preventing poisoned targets from buggy search. KL-weighting is disabled for the first 50 GPU hours of Phase 3 (uniform weighting) until ExIt quality is validated.

### 6.3 Playout cap randomization

Not all states deserve equal search effort. Following Wu (2020, KataGo), HYDRA draws a playout cap $C$ per state from a distribution that favors hard positions:

- States where the network's top-2 policy gap < 10%: full search budget (depth 2, 256 belief samples x N=4 opponents).
- States where the gap is 10-30%: medium budget (depth 2, 32 belief samples x N=4 opponents).
- States where the gap > 30%: zero search (use network policy directly as ExIt target).

This ensures compute is concentrated on positions where search actually helps, rather than wasting cycles confirming obvious decisions.

### 6.4 Non-blocking architecture

The pondering system runs on separate CPU/GPU threads from the main self-play loop. **GPU allocation (4x RTX 5000)**: GPU 0-1 for training (forward+backward), GPU 2 for self-play inference (batched), GPU 3 for pondering leaf evaluation (batched). **Weight sync**: double-buffered async copy from training GPUs to pondering GPU every 30-60 seconds. Pondering reads from buffer A while buffer B receives the update, then swaps atomically. Each ExIt target is tagged with `weight_version` for staleness tracking. **Staleness budget**: discard ExIt targets older than 500 gradient steps. **Batch coordination**: lockless concurrent hashmap keyed by (game_id, turn, seat) with 30-second TTL. Pondering writes; trajectory finalization reads and consumes. Absent = use on-policy target. Self-play continues at network speed regardless of whether pondering finishes. If pondering completes before the next decision point, its results are used. If not, the network's own policy is used. This guarantees that pondering never reduces self-play throughput.

### 6.5 Approximate policy iteration contract

ExIt with search targets implements approximate policy iteration. If each iteration produces an $\varepsilon$-approximate greedy policy:

$$Q^\pi(I, \pi'(I)) \ge \max_a Q^\pi(I, a) - \varepsilon$$

then the performance loss is bounded: $V^{\pi^*}(I_0) - V^{\pi'}(I_0) \le O(H\varepsilon)$ (Abbasi-Yadkori et al. 2019, POLITEX). For imperfect-information EFGs, Kalogiannis and Farina (2024) prove policy gradient methods converge globally at rate $O(1/\sqrt{T})$. HYDRA reduces $\varepsilon$ by increasing search depth during pondering as the network improves -- a virtuous cycle where better networks produce better search targets produce better networks.

### 6.6 Label amplification

Every pondered state with a computed $\pi^{\text{ExIt}}$ is a valid training example. The ratio of ExIt-labeled states to environment steps, $N_{\text{lab}} / N_{\text{env}}$, measures the amplification factor. With oracle pondering during training (~1ms per state vs 4ms with Sinkhorn+blind), completion rate increases to ~70-80%. HYDRA achieves $N_{\text{lab}} / N_{\text{env}} \approx 0.5$-$0.6$ -- meaning 50-60% of all training states have oracle-quality search labels at no throughput cost.

---

## 7. SR Concentration for Danger Calibration

This technique has **zero inference cost**. It is used exclusively during training to validate that the danger head produces well-calibrated safety estimates. The mathematical framework is retained from the predecessor design because it is rigorous, cheap, and directly useful.

### 7.1 Strongly Rayleigh property

**Definition (BBL 2009).** A measure $\mu$ on $\{0,1\}^n$ is *strongly Rayleigh* (SR) if its generating polynomial $g_\mu(z) = \sum_S \mu(S) \prod_{i \in S} z_i$ is real stable.

**Theorem 7.1 (BBL 2009, Theorem 3.8).** The multivariate hypergeometric distribution over tile assignments is SR. This places it at the top of the negative dependence hierarchy: SR $\Rightarrow$ CNA+ $\Rightarrow$ CNA $\Rightarrow$ NA $\Rightarrow$ pairwise negative correlation.

### 7.2 Gaussian concentration bounds

**Theorem 7.2 (Pemantle-Peres 2014).** For a $k$-homogeneous SR measure and Lipschitz-1 function $f$:

$$P(f - \mathbb{E}f \ge a) \le \exp\!\left(-\frac{a^2}{8k}\right)$$

For an opponent hand of $k = 13$ tiles, any Lipschitz-1 danger function concentrates as $\exp(-a^2/104)$.

### 7.3 Bernstein-Serfling finite-population bounds

**Theorem 7.3 (Bardenet-Maillard 2015).** For tiles drawn without replacement from $N$ tiles, with variance $\sigma^2$:

$$P\!\left(\frac{1}{n}\sum_{t=1}^n (X_t - \mu) \ge \epsilon\right) \le \exp\!\left(-\frac{n\epsilon^2 / 2}{\sigma^2 \rho_n + \frac{2}{3}(b-a)\epsilon \kappa_n}\right)$$

where $\rho_n = 1 - (n-1)/N$ is the Serfling correction. Late game ($n$ close to $N$): $\rho_n \to 0$, giving near-perfect concentration. This captures the strategic intuition that late-game defense is more precise.

### 7.4 SR closure under conditioning

**Proposition 7.4 (BBL 2009, Theorem 4.9).** SR is closed under conditioning ($X_i = 0$ or $X_i = 1$), external fields, projections, and truncation. When tiles are revealed (discards, calls, draws), the posterior over remaining tiles stays SR, so all bounds remain valid after every observation.

### 7.5 Validation gate: danger head calibration

During training, compare the danger head's outputs against analytical SR bounds on a held-out set of 50K stratified states:

1. For each state, compute the SR analytical bound on deal-in probability per tile.
2. Compare against the danger head's predicted probability.
3. Flag systematic over/under-estimation (KS statistic > 0.05).

The danger head is not required to match the SR bounds exactly -- it can be sharper (using patterns the bounds cannot capture). But it must not violate the bounds systematically, which would indicate miscalibration.

---

## 8. Score Belief Distribution

Instead of a scalar value estimate, HYDRA predicts the full distribution over game outcomes. This follows KataGo (Wu 2020) and adds zero inference cost beyond two extra linear projections.

### 8.1 Architecture

Two heads on the shared backbone. $B = 64$ bins spanning $[-50000, +60000]$ score delta (~1700 pts/bin):

- **Score-pdf head**: softmax over $B$ bins. Outputs $p_\theta(s \mid I_t)$.
- **Score-cdf head**: sigmoid per bin. Outputs $F_\theta(s \mid I_t) = P_\theta(S \le s \mid I_t)$.

### 8.2 Training loss

$$\mathcal{L}_{\text{score}} = -\sum_{b=1}^{B} \hat{p}_b \log p_\theta(b \mid I_t) + \lambda_{\text{cdf}} \sum_{b=1}^{B} \text{BCE}(F_\theta(b \mid I_t),\ \mathbf{1}[S^* \le b])$$

where $S^*$ is the realized outcome and $\hat{p}_b$ is the empirical target.

### 8.3 CVaR for placement-aware play

The cdf head enables direct CVaR computation:

$$\text{CVaR}_\alpha = \mathbb{E}[S \mid S \le F^{-1}(\alpha)]$$

This supports placement-conditional strategy without any additional inference cost. "Avoid 4th place" = optimize $\text{CVaR}_{0.25}$. "Need 1st place" = optimize upper tail. The pdf head provides the conditional distribution for either scenario.

---

## 9. Training Plan

### 9.1 Phase 1: Behavioral Cloning (50 GPU hours)

Train on 5-6M expert games (Tenhou Houou + Majsoul high-rank). All 9 heads supervised. 24x data augmentation (6 suit permutations x 4 seat rotations) applied at this phase only. The goal is a competent initial policy that plays legal, sensible Mahjong -- roughly 4-5 dan level.

**Loss weights**: $\mathcal{L} = 1.0 \cdot \mathcal{L}_\pi + 0.5 \cdot \mathcal{L}_V + 0.2 \cdot \mathcal{L}_{\text{GRP}} + 0.1 \cdot \mathcal{L}_{\text{tenpai}} + 0.1 \cdot \mathcal{L}_{\text{danger}} + 0.1 \cdot \mathcal{L}_{\text{opp-next}} + 0.05 \cdot \mathcal{L}_{\text{score}} - \beta_{\text{ent}} \cdot H(\pi)$. These initial weights follow Mortal's ordering (policy > value > auxiliary). Tune during Phase 1 BC via validation loss.

### 9.2 Phase 1.5: Oracle Guiding (200 GPU hours, ~5M games)

The critical missing technique from Mortal that explains Suphx's 1.7 dan advantage: **oracle guiding** (Li et al. 2020). During early self-play, provide the agent with perfect-information features (opponents' actual hands, wall composition) as additional input channels. Gradually mask these channels over training (linear schedule over 200 hours), forcing the network to internalize the patterns it learned with oracle access.

Oracle guiding works because it gives the network a "teacher's answer key" for defensive play, opponent modeling, and danger assessment during the critical early training phase. The network learns WHAT to predict before learning HOW to predict it from imperfect information. Suphx reports this as their single most impactful technique.

**Implementation**: Add oracle channels (opponent hand indicators + wall composition) to the input tensor during this phase. Oracle features are masked via Bernoulli dropout $\delta_t$ with $P(\delta_t = 1) = \gamma_t$ decaying from 1 to 0 over the phase (Li et al. 2020). **Critical stability tricks** (from Suphx): when $\gamma_t$ reaches 0, (1) decay learning rate to $\frac{1}{10}$ of its value, (2) reject state-action pairs with importance weight above a threshold. Without these, post-oracle training is unstable. Cost: zero inference overhead since oracle channels are removed.

### 9.3 Phase 2: PPO Self-Play (750 GPU hours, ~17M games)

Pure self-play at full inference speed. The oracle-guided network plays against copies of itself. PPO with clipped objective, GAE for advantage estimation. All 9 heads continue training with self-play outcomes as targets.

**Key metric**: self-play Elo gain per GPU hour. Expected trajectory: rapid improvement for ~200 hours (benefiting from oracle pretraining), then diminishing returns as PPO approaches its ceiling without search. 6x suit permutation applied to the replay buffer at negligible cost. **Temperature-varied self-play**: each of the 4 seats draws $\tau \sim \mathrm{Uniform}(0.5, 1.5)$ at game start, creating implicit strategy diversity without population-based training infrastructure. This prevents convergence to a single playstyle and makes the agent robust to varied opponents.

### 9.4 Phase 3: PPO + Pondering ExIt (1000 GPU hours, ~13M games)

Activate the pondering system. Self-play continues as in Phase 2, but now idle threads run oracle-evaluated PUCT search (no Sinkhorn needed during training -- oracle has full game state). **ExIt warmup**: start with 10% ExIt targets / 90% on-policy, linearly increase to 40% ExIt over the first 200 GPU hours. **Rollback safety**: save a checkpoint at Phase 3 start. Monitor self-play Elo every 25 GPU hours. If Elo drops below the Phase 2 endpoint, kill Phase 3, rollback to checkpoint, and debug pondering before retrying.

**ExIt loss**:

$$\mathcal{L} = \lambda_\pi \text{CE}(\pi_\theta, \pi^{\text{ExIt}}) + \lambda_V \|V_\theta - V^{\text{ExIt}}\|_2^2 + \sum_{h \in \text{aux}} \lambda_h \mathcal{L}_h$$

**Expected outcome**: the ExIt targets provide a consistent policy improvement signal beyond PPO's ceiling. The virtuous cycle (better network -> better search -> better targets -> better network) should push performance into the 8-10 dan range.

**Checkpoint opponent pool**: Save a checkpoint every 50 GPU hours. In each self-play game, 3 of the 4 seats are randomly assigned to recent checkpoints (last 10 saved), 1 seat is the current policy. This prevents strategy cycling and creates implicit opponent diversity beyond temperature variation. Storage cost: ~33MB per checkpoint (bf16), negligible. This is "league play lite" without the engineering complexity of full population-based training.

---

## 10. Validation Gates

Five gates, checked at training milestones. Each is a go/no-go decision.

**G0: Does pondering search improve actions?** (Highest uncertainty)
- Collect 200K stratified states (early/mid/late, offense/defense).
- Compare search policy $\pi^{\text{ExIt}}$ against network policy $\pi_\theta$ using oracle evaluation.
- Go: mean improvement $\Delta > 0$ with < 40% states showing negative $\Delta$.
- If G0 fails: fix the search before proceeding to Phase 3.

**G1: Sinkhorn convergence (deployment mode).**
- Residuals $\varepsilon_{\text{row}}, \varepsilon_{\text{col}} < 10^{-4}$ within 30 iterations on 95% of states.
- If G1 fails: increase iterations or switch to log-domain stabilized Sinkhorn.
- Note: Sinkhorn is only used during deployment pondering, not during training (oracle mode).

**G2: Danger head calibration.**
- Compare danger head outputs against SR analytical bounds (Section 7.5).
- KS statistic < 0.05 across tile types and game phases.
- If G2 fails: increase danger head capacity or add SR-bound auxiliary loss.

**G3: ExIt label amplification.**
- Pondering completion rate $N_{\text{lab}} / N_{\text{env}} > 0.3$.
- If G3 fails: reduce pondering search depth to fit within opponent turn time.

**G4: Score belief calibration.**
- Predicted cdf $F_\theta(s)$ vs empirical outcome frequencies.
- KS statistic < 0.05 across score bins.
- If G4 fails: increase score head capacity or adjust bin discretization.

---

### 10.2 Evaluation Protocol

**Internal evaluation (during training):**
- Self-play Elo tracked every 50 GPU hours (1v1v1v1 against checkpoints from 50h ago)
- Policy agreement rate on 10K held-out Houou games (target: >72% by end of Phase 2)
- Value prediction MSE on held-out games (tracked for convergence monitoring)

**External evaluation (after training):**
- **Head-to-head vs Mortal**: 1000+ games, Hydra vs 3x Mortal instances. Baseline: Mortal ~7 dan. If Hydra wins >55% of placements, Hydra > 7 dan.
- **Tenhou deployment**: play on Tenhou via bot API. Target: reach Houou lobby (requires 7+ dan), track stable rank. Need ~1000-2000 games for reliable stable rank.
- **Ablation tests**: disable pondering ExIt (Phase 3 contribution), oracle guiding (Phase 1.5 contribution), and score distributions independently to measure each technique's contribution.
- **Encoding ablation**: test adding explicit "remaining tile count" channels (85 -> 89 channels); the network may already learn this from discard/meld channels, but explicit encoding could accelerate early training.

---

## 11. Limitations

1. **No runtime opponent modeling.** The network learns opponent patterns implicitly through the opp-next and tenpai heads, but cannot adapt to novel opponent strategies at inference time. Against adversarial or highly unconventional opponents, this is a ceiling.

2. **Pondering search quality.** During training, oracle evaluation eliminates belief noise. During deployment, the shallow search (depth 2) with Sinkhorn beliefs is a crude approximation. ExIt targets during training are high quality (oracle), but deployment pondering may be noisy in late-game high-correlation states.

3. **No exploitability guarantees.** PPO self-play optimizes strength, not exploitability bounds. HYDRA may develop exploitable patterns that a targeted adversary could discover. This is shared with all pure-RL Mahjong systems including Mortal and LuckyJ.

4. **Budget ceiling.** 2000 GPU hours is a hard constraint. If Phase 3 convergence is slower than projected, the system may plateau below target. The progressive phase design provides natural early stopping points.

5. **BC data dependency.** The Phase 1 warm-start relies on 5-6M expert games of reasonable quality. Poor data quality (misranked games, cheaters, bots) would propagate errors into later phases.

6. **Single-policy limitation.** HYDRA trains a single policy for all game situations. It cannot condition on opponent skill level or tournament context at inference time. The score distribution heads partially address this (CVaR for risk management), but full context-awareness requires additional mechanism.

---

## 12. References

1. Borcea, Branden, Liggett. "Negative Dependence and the Geometry of Polynomials." *JAMS* 22(2), 2009.
2. Pemantle, Peres. "Concentration of Lipschitz Functionals of Determinantal and Other Strong Rayleigh Measures." *CPC* 23(1), 2014.
3. Bardenet, Maillard. "Concentration Inequalities for Sampling Without Replacement." *Bernoulli* 21(3), 2015.
4. Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." *NeurIPS*, 2013.
5. Sinkhorn, Knopp. "Concerning Nonnegative Matrices and Doubly Stochastic Matrices." *Pacific J. Math*, 1967.
6. Wu. "Accelerating Self-Play Learning in Go." *arXiv 1902.10565*, 2020 (KataGo).
7. Anthony, Tian, Barber. "Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration)." *NeurIPS*, 2017.
8. Abbasi-Yadkori, Bartlett, Bhatia, Lazic. "POLITEX: Regret Bounds for Policy Iteration Using Expert Prediction." *ICML*, 2019.
9. Kalogiannis, Farina. "Policy Gradient Methods Converge in Imperfect Information Extensive-Form Games." *NeurIPS*, 2024.
10. Schulman, Wolski, Dhariwal, Radford, Klimov. "Proximal Policy Optimization Algorithms." *arXiv 1707.06347*, 2017.
11. Glosten, Milgrom. "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *J. Financial Economics* 14(1), 1985.
12. Perolat et al. "Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning (DeepNash)." *Science*, 2022.
13. Hu, Lerer, Peysakhovich, Foerster. "Other-Play for Zero-Shot Coordination." *ICML*, 2020.
14. Hermon, Salez. "Modified Log-Sobolev Inequalities for Strong-Rayleigh Measures." *AAP* 33(2), 2023.
15. Dubhashi, Panconesi. *Concentration of Measure for the Analysis of Randomized Algorithms.* Cambridge, 2009.
16. Rudolph et al. "Reevaluating Policy Gradient Methods for Imperfect Information Games." *arXiv 2502.08938*, 2025.
17. Silver et al. "Mastering the Game of Go Without Human Knowledge." *Nature* 550, 2017.
18. Li et al. "Suphx: Mastering Mahjong with Deep Reinforcement Learning." *arXiv 2003.13590*, 2020.
19. Li, Wu, Fu, Fu, Zhao, Xing. "Speedup Training AI for Mahjong via Reward Variance Reduction." *IEEE CoG*, 2022.
20. Boney, Ilin, Kannala, Seppanen. "Learning to Play Imperfect-Information Games by Imitating an Oracle Planner." *IEEE Trans. Games* 14(3), 2021.

---

## 13. Upgrade Path

When more compute becomes available (5000+ GPU hours), the following techniques from the predecessor design should be added in priority order:

**Priority 1: Mixture-SIB belief inference (add at 5000 GPU hours).** Replace the single Sinkhorn belief with $L = 4$-$8$ mixture components during pondering. Each component models a distinct opponent hand hypothesis (honitsu, toitoi, damaten, etc.). This improves ExIt target quality in late-game states where pairwise tile correlations exceed 15%. Cost: ~3x pondering compute, offset by better ExIt targets.

**Priority 2: Full Factored Belief Search (add at 10000 GPU hours).** Replace shallow pondering search with depth-8+ FBS using belief sampling. This dramatically improves ExIt target quality but requires significant pondering compute. At this budget, the network is strong enough that deeper search provides consistent improvement.

**Priority 3: Rao-Blackwellized belief decomposition (add with FBS).** Factor hidden state into suit-group totals (sampled) and within-group assignments (analytically marginalized). Reduces FBS belief sampling variance by an expected factor of 2-5x. Only worthwhile when FBS is deep enough to benefit from reduced variance.

**Not recommended at any budget**: IVD/EFE decomposition (adds runtime complexity for uncertain gain), DRDA-M dynamics (PPO self-play is empirically sufficient), adaptive leaf evaluation (marginal benefit over simpler leaf evaluation).

The 9-technique system is designed as a complete, self-contained architecture that achieves competitive performance at 2000 GPU hours. The upgrade path exists for teams with larger budgets, not as a requirement for the base system.

