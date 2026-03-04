# Critique Agent Deep Dive v2: Implementation Uncertainties

## YOUR ROLE
You are an expert who IMPLEMENTS game AI systems (not just designs them). You have built RL training pipelines, search algorithms, and belief inference systems. You can read papers and verify math. When I ask "how does X work," I want pseudocode-level detail, not hand-waves.

## CONTEXT
HYDRA-OMEGA is a 4-player Riichi Mahjong AI. Full design: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md

We're implementing in Rust (game engine) + Burn framework (ML). Budget: 2000 GPU hours on 4x RTX 5000 Ada (261 TFLOPS bf16 each). Target: beat LuckyJ (10.68 dan).

I have 7 areas of uncertainty. For each one, I need you to either resolve my confusion with concrete math/pseudocode, or tell me "this is a real problem and here's how to fix it."

## REFERENCE PAPERS (read these to answer accurately)

**Core LuckyJ stack:**
- ACH (Actor-Critic Hedge): https://openreview.net/pdf?id=DTXZqTNV5nW (ICLR 2022, Algorithm 2, Eq. 29, Section 4)
- OLSS (Opponent-Limited Online Search): https://proceedings.mlr.press/v202/liu23k/liu23k.pdf (ICML 2023, Appendix D.1 for training details, Table 4 for hyperparams)
- GSCU (opponent modeling): https://proceedings.mlr.press/v162/fu22b/fu22b.pdf (ICML 2022)

**Mahjong AI references:**
- Suphx (oracle guiding, pMCPA, GRP): https://arxiv.org/pdf/2003.13590 (Section 3-4 for oracle guiding, Footnote 8 for pMCPA not deployed)
- Mortal (DQN, 12-block ResNet): https://arxiv.org/pdf/2404.12877
- Mortal source code (architecture): https://github.com/Equim-chan/Mortal
- RVR (oracle critic, expected reward network): search IEEE CoG 2022 "Speedup Training AI for Mahjong"

**RL training algorithms:**
- PPO: https://arxiv.org/pdf/1707.06347
- PPO reevaluation in IIGs (5600 runs): https://arxiv.org/pdf/2502.08938
- PG convergence in IIEFGs: https://arxiv.org/pdf/2408.00751
- DRDA (multiplayer POSGs): https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf
- R-NaD / DeepNash: https://www.science.org/doi/10.1126/science.add4679
- DDCFR (learned discounting): https://openreview.net/forum?id=6PbvbLyqT6 (ICLR 2024)
- GAE (Generalized Advantage Estimation): https://arxiv.org/pdf/1506.02438

**Belief inference / sampling:**
- Sinkhorn-Knopp: https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
- SMC for contingency tables: Chen, Diaconis, Holmes, Liu (2005) "Sequential Monte Carlo Methods for Statistical Analysis of Tables" in JASA
- Patefield AS159 (random table generation): https://www.jstor.org/stable/2346669
- Strongly Rayleigh / negative dependence: https://www.ams.org/jams/2009-22-02/S0894-0347-08-00618-8/
- Concentration for sampling w/o replacement: https://projecteuclid.org/journals/bernoulli/volume-21/issue-3/Concentration-inequalities-for-sampling-without-replacement/10.3150/14-BEJ605.pdf

**Search / ExIt / MCTS:**
- Expert Iteration: https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html
- KataGo (playout cap, auxiliary targets, score distributions): https://arxiv.org/pdf/1902.10565
- AlphaZero: https://www.science.org/doi/10.1126/science.aar6404
- Oracle planner imitation: https://ieeexplore.ieee.org/document/9382923 (Boney et al. 2021)
- PIMC strategy fusion: https://arxiv.org/pdf/2408.02380

**Game theory / safety:**
- Hunter bound: https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/an-upper-bound-for-the-probability-of-a-union/092D711504BA968EF0D1D903A2685D60
- Kounias bound: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-39/issue-6/Bounds-for-the-Probability-of-a-Union-with-Applications/10.1214/aoms/1177698049.full
- Distributionally robust optimization (KL ball): https://proceedings.mlr.press/v202/liu23k/liu23k.pdf (Section 4)

**Scaling / compute:**
- KataGo compute efficiency: https://arxiv.org/pdf/1902.10565 (Section 6-7)
- Chinchilla scaling laws: https://arxiv.org/pdf/2203.15556
- Scaling laws for board games: https://arxiv.org/pdf/2104.03113

**Our codebase:**
- Full repo: https://github.com/NikkeTryHard/hydra
- HYDRA_FINAL design doc: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- Implementation roadmap: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- BPR feasibility report: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/BPR_FEASIBILITY.md
- Burn RL examples (PPO in Rust): https://github.com/yunjhongwu/burn-rl-examples
- Burn framework docs: https://burn.dev/docs/burn/

---

## UNCERTAINTY 1: ACH Implementation Details (highest priority)

I have ACH's loss from the paper (ICLR 2022, Eq. 29):
```
L_π(s) = -c · η(s) · y(a|s;θ)/π_old(a|s) · A(s,a)
```

But I'm missing critical details:

**1a. What is η(s)?** The paper calls it a "Hedge coefficient." Is it:
- A learned parameter per state?
- A function of the cumulative regret at state s?
- A fixed hyperparameter?
- Derived from the policy entropy at s?

Give me the EXACT formula for η(s) from the paper. If it's from the Hedge algorithm, I expect something like η(s) = sqrt(ln(|A|) / T) but adapted for the actor-critic setting. What is it?

**1b. The gate c -- per-action or per-state?** The 4 conditions for c involve both A(s,a) and y(a|s;θ). This means c depends on the specific action a, not just the state s. So the loss L_π(s) should actually be L_π(s,a) right? And the full loss is E_a~π_old[L_π(s,a)]?

**1c. How does ACH compute advantages in 4-player Mahjong?** Standard GAE uses V(s) as a baseline. In 4-player, whose value function? Each player has their own V_i(s). Does ACH use:
- Per-player advantages A_i(s,a) = Q_i(s,a) - V_i(s)?
- Counterfactual values like in CFR?
- Something else?

The OLSS paper (ICML 2023, Appendix D) says "blueprint models are trained using ACH" and gives hyperparams (GAE λ=0.95, γ=0.995, α=0.5). Does this mean ACH uses standard GAE, NOT counterfactual values?

**1d. Can I implement ACH as "PPO with a modified loss"?** Specifically: same training loop, same data collection, same GAE computation, just swap the loss function? Or does ACH require a fundamentally different data collection / trajectory structure?

**Read these to answer:**
- ACH paper: https://openreview.net/pdf?id=DTXZqTNV5nW (Algorithm 2, Eq. 29, Section 4, Appendix A for Hedge derivation)
- ACH applied to Mahjong: same paper Section 7.2 + Table 4 (hyperparams)
- OLSS paper Appendix D.1 for ACH training details in Mahjong: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- Hedge algorithm background: https://en.wikipedia.org/wiki/Multiplicative_weights_update_method
- Original Hedge: Freund & Schapire 1997 "A Decision-Theoretic Generalization of On-Line Learning"

---

## UNCERTAINTY 2: CT-SMC Exact DP Feasibility

I claimed the DP has ~50K states and runs in <1ms. Let me stress-test this.

**2a. State space size.** Column capacities are opponent hand sizes + wall. Typical mid-game: s = (13, 13, 13, 11) = 50 hidden tiles. The DP state is the residual capacity vector c = (c_1, c_2, c_3, c_W). At step k, residual capacities range from 0 to their initial values. State count at step k: product of (c_j + 1) for j in {1,2,3,W}.

Worst case (all tiles hidden, early game): c = (13, 13, 13, 97). State count: 14 * 14 * 14 * 98 = 268,912. That's 5x my earlier estimate of 50K. With 34 tile types and up to 35 compositions per type: 34 * 269K * 35 = ~320M ops. Is this still <1ms?

At 10 GFLOPS throughput for integer DP in Rust: 320M / 10G = 32ms. That's NOT <1ms. **Am I wrong about the state space, or is there a pruning trick I'm missing?**

**2b. Late game (the important case).** When H=25 (late game, correlations matter most): c = (13, 13, 13, -14)... wait, wall can't be negative. Let me reconsider. If 111 tiles are visible, 25 are hidden. Opponent hands: 13+13+13 = 39 > 25. That's impossible -- opponent hands shrink as tiles are drawn.

Actually in late game: each opponent has ~7-10 concealed tiles (some were discarded/melded). Wall has ~4-8 tiles. So c = (8, 9, 7, 4) or similar. State count: 9 * 10 * 8 * 5 = 3,600. That's TINY. 34 * 3600 * 35 = 4.5M ops = definitely <1ms.

**So the DP is fast in late game (where it matters most) but potentially slow in early game (where correlations don't matter). Is this correct? Should we only run CT-SMC when H < some threshold (e.g., H < 50) and use Mixture-SIB for early game?**

**2c. Numerical stability.** The partition function Z_k(c) involves products of ω_kj^{x_j}. If some ω values are very large or very small, Z can overflow/underflow. Standard fix: work in log-space. But log-space makes the sum-over-compositions harder (log-sum-exp per row). Is this a real issue or am I overthinking it?

**Read these to answer:**
- Chen, Diaconis, Holmes, Liu (2005) "Sequential Monte Carlo Methods for Statistical Analysis of Tables" (JASA)
- Patefield AS159: https://www.jstor.org/stable/2346669
- Multivariate hypergeometric: https://en.wikipedia.org/wiki/Multivariate_hypergeometric_distribution
- Integer partition enumeration: https://en.wikipedia.org/wiki/Composition_(combinatorics)
- Rust SIMD performance guide: https://doc.rust-lang.org/std/simd/

---

## UNCERTAINTY 3: AFBS + Rollout Net Interaction

**3a. What does the rollout net replace?** In AFBS, there are two uses of neural evaluation:
- Leaf value estimation (V(state) at the bottom of the search tree)
- Opponent action sampling (π_opp(action | state) to simulate opponent responses)

Does the rollout net replace BOTH? Or just the opponent action sampling? LuckyJ's "environmental model" predicts "action probabilities" only -- so it's the opponent action sampler, not the value estimator?

**3b. Architecture mismatch.** The rollout net (3-6 blocks) has different capacity than the learner (40 blocks). When we distill learner -> rollout:
- Do we distill the POLICY head only? Or all heads?
- Do we use the same input encoding (85x34)? Or a simplified one?
- How much training data do we need per distillation step?

**3c. Staleness.** The rollout net is distilled every 50 GPU hours. During those 50 hours, the learner improves. The rollout net's opponent model becomes stale. How bad is this? Is 50 hours too infrequent?

LuckyJ's OLSS paper doesn't mention re-distilling. Does their environmental model stay fixed after initial training?

**Read these to answer:**
- OLSS paper Appendix D.1 (environmental model training): https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- Knowledge distillation survey: https://arxiv.org/pdf/2006.05525
- AlphaZero self-play actor/learner split: https://www.science.org/doi/10.1126/science.aar6404
- KataGo distributed training (multiple net sizes): https://katagotraining.org/

---

## UNCERTAINTY 4: Oracle Guiding + ACH Coexistence

**4a. Standard CTDE says:** oracle critic provides advantages, actor conditions on public info only. But ACH's loss uses y(a|s;θ) (logits) and π_old(a|s). These are from the ACTOR (public info only). The oracle is used ONLY for computing A(s,a) via GAE with V_oracle. Is this correct?

**4b. Zero-sum oracle critic.** We enforce sum_i V_i = 0. In ACH, the Hedge coefficient η(s) might depend on the value scale. If the oracle critic has different scale than the blind critic, does η(s) need recalibration?

**4c. Oracle guiding masks.** During Phase 1 (oracle guiding), we provide oracle channels with Bernoulli dropout decaying from 1 to 0. During Phase 2 (ACH self-play), oracle channels are gone. But the oracle CRITIC still sees all hands. Is there a training instability when the actor loses oracle input but the critic keeps it?

**Read these to answer:**
- Suphx paper (oracle guiding + pMCPA): https://arxiv.org/pdf/2003.13590
- CTDE survey: https://arxiv.org/pdf/2409.03052
- MAPPO (multi-agent PPO with centralized critic): https://arxiv.org/pdf/2103.01955
- RVR (oracle critic + zero-sum): search "Speedup Training AI for Mahjong via Reward Variance Reduction" IEEE CoG 2022

---

## UNCERTAINTY 5: Endgame Solver Scaling

**5a. How many draw sequences with wall=10?** The wall is a multiset of 10 tiles (not all distinct). The number of distinct orderings is 10!/∏(n_k!) where n_k is the count of tile type k in the wall. Worst case (all distinct): 10! = 3.6M. Best case (many repeats): much less. What's the TYPICAL case in Mahjong?

**5b. Can we actually enumerate in real-time?** With 3.6M sequences, each requiring ~100ns to evaluate (game logic): 360ms. That's too slow for a 100ms decision budget. But with DP over (remaining_tiles, turn_number), the state space is much smaller. What's the actual DP state count?

**5c. Opponent actions interleave with draws.** It's not just "enumerate draw sequences." Between draws, opponents take actions (discard, call, riichi). How do we handle this? Monte Carlo over opponent actions + exact enumeration over draws? Or do we model opponents as a fixed policy and enumerate the full game tree?

**Read these to answer:**
- Endgame solving in imperfect info: https://arxiv.org/pdf/2007.13544 (ReBeL, Section 3 on depth-limited solving)
- Expectiminimax: https://en.wikipedia.org/wiki/Expectiminimax
- PIMC for card games: https://arxiv.org/pdf/2408.02380
- Bridge endgame solvers: search "double dummy solver bridge" for precedent on exact endgame in card games
- Mahjong tile draw mechanics: https://riichi.wiki/Haitei_and_houtei (last draw rules)

---

## UNCERTAINTY 6: Network Sizing (40 blocks vs smaller)

**6a. The critique agent flagged 40 blocks as potentially too big for 2000 GPU hours.** LuckyJ used 3 blocks. Mortal uses 12 blocks. Our 40 blocks is 3-13x larger than proven systems.

What's the compute-optimal network size for 2000 GPU hours of self-play training? Chinchilla scaling for RL is different from language models, but the principle (model size should scale with data volume) applies. With ~35M self-play games producing ~2.5B decisions, what's the optimal parameter count?

**6b. Two-network training.** The critique agent suggested: small actor (fast, for self-play data generation) + big learner (slow, for accuracy) with periodic distillation big->small. This is like AlphaZero's approach.

But our pondering ExIt already provides search-quality targets to the big learner. Does two-network training add value on top of pondering ExIt? Or is it redundant?

**6c. Concrete recommendation.** Should we:
- (A) Keep 40 blocks, accept slower self-play throughput
- (B) Use 20 blocks (still 2x Mortal), get 2x more self-play games
- (C) Two-network: 12-block actor + 40-block learner
- (D) Something else

For each option, estimate: games/GPU-hour, total games at 2000 hours, expected dan level.

**Read these to answer:**
- Chinchilla scaling laws: https://arxiv.org/pdf/2203.15556
- Scaling laws for board games: https://arxiv.org/pdf/2104.03113 (Jones 2021)
- KataGo training efficiency: https://arxiv.org/pdf/1902.10565 (Section 6-7 on compute efficiency)
- Mortal architecture (12 blocks, 256ch): https://github.com/Equim-chan/Mortal (model.py)
- LuckyJ architecture (3 blocks): https://proceedings.mlr.press/v202/liu23k/liu23k.pdf (Appendix D.1)
- AlphaZero actor/learner split: https://www.science.org/doi/10.1126/science.aar6404
- IMPALA (distributed actor-learner): https://arxiv.org/pdf/1802.01561

---

## UNCERTAINTY 7: SaF (Search-as-Feature) Training

**7a. The adaptor g_ψ.** Our design says: ℓ_final(a) = ℓ_θ(a) + α_SaF · g_ψ(f(a)) · m(a). What should g_ψ be?
- A single linear layer? (simplest)
- A small MLP (2-3 layers)?
- Something that processes ALL actions jointly (attention over action features)?

**7b. Training regime.** When do we train g_ψ?
- Jointly with the main network from the start?
- Only in Phase 3 (when search features exist)?
- With a separate loss (regression to ΔQ) or end-to-end through the policy loss?

**7c. The presence mask m(a).** When SaF features are absent (no search was run), m(a)=0 and the adaptor is zeroed. But during training, features are present for ~50-60% of states. Does the network learn to rely too heavily on SaF features, then degrade when they're absent at deployment?

**Read these to answer:**
- KataGo search-informed training targets: https://arxiv.org/pdf/1902.10565 (Section 4-5)
- FiLM conditioning (feature-wise linear modulation): https://arxiv.org/pdf/1709.07871
- Dropout as feature absence training: https://arxiv.org/pdf/1207.0580
- Residual adaptor design patterns: https://arxiv.org/pdf/2106.09685 (LoRA paper for adaptor sizing)
- Mixture of experts gating: https://arxiv.org/pdf/1701.06538 (Shazeer et al. 2017)

---

## OUTPUT FORMAT
For each uncertainty (1-7), provide:
1. **Resolution** (concrete math, pseudocode, or "this is a real problem")
2. **Confidence** (low/medium/high)
3. **Key reference** (paper + section/equation number)
4. **Implementation recommendation** (what to actually code)
