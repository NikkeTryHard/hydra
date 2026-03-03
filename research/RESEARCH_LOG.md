# Research Log: LuckyJ Analysis and Hydra Spec Updates

Date: 2026-03-03
Context: Exhaustive research loop (30+ subagent runs, 4 independent judge panels across 8+ rounds) evaluating Hydra's probability of matching LuckyJ (10.68 stable dan). Cold research judges with tool access converged at ~20-25% for 10.68, ~55% for 9+ dan.

## Key Findings

### 1. ACH Loss Function (Replaces PPO as Primary Training Loss)

**Discovery:** ACH (Actor-Critic Hedge, ICLR 2022, Fu et al.) is a ~20-line drop-in replacement for PPO's clipped surrogate loss. Same GAE, same value function, same entropy, same actor-learner infrastructure. The only change is the policy gradient computation.

**Evidence:**
- Official code: github.com/Liuweiming/ACH_poker (C++/OpenSpiel)
- Third-party impl: github.com/sbl1996/ygo-agent (JAX, actively maintained for Yu-Gi-Oh! AI)
- Key hyperparams from ACH paper's Mahjong experiments: eta=1.0, logits_threshold=6.0, clip=0.5, entropy=0.01

**What ACH does differently:** Instead of maximizing returns (PPO), it minimizes weighted cumulative counterfactual regret via the Hedge algorithm (multiplicative weights). The network outputs cumulative regret values, and the policy is derived via softmax over regrets.

**Convergence:** Provably converges to Nash equilibrium in 2-player zero-sum at O(T^{-1/2}). No formal guarantee for 4-player (same as PPO). However, regret-weighted updates focus training on high-leverage decisions, which may produce qualitatively better policies empirically.

**Critical note:** ACH's Nash guarantee is 2-PLAYER ONLY. LuckyJ's published ACH results are on 1v1 Mahjong, not 4-player. The 4-player LuckyJ system is unpublished. Every top 4-player Mahjong AI (Suphx, Mortal, LuckyJ) uses RL self-play without formal equilibrium guarantees.

**Action:** Replace PPO clipped surrogate loss with ACH loss in Phase 2 and Phase 3. Keep PPO as documented fallback.

### 2. Oracle Distillation Claim Correction (0.06 dan, not 0.6)

**Error found:** Multiple docs cited Suphx oracle contribution as "+0.6 dan." This is WRONG. TRAINING.md line 197 correctly documents: the FULL RL pipeline gained ~0.63 dan (SL->RL+GRP->RL+GRP+Oracle). Oracle guiding SPECIFICALLY contributed ~0.06 dan (RL-2 vs RL-1 median gap in Figure 8). The +0.63 includes techniques BEYOND oracle (GRP, RL tuning).

**Action:** Ensure all cross-references cite the correct 0.06 dan oracle-specific contribution. The +0.63 full-pipeline gain remains valid but must not be attributed to oracle alone.

### 3. PGOI Estimate Revision (+0.3-0.8 central, not +1.0-1.5)

**Previous spec (SEARCH_PGOI.md):** Listed optimistic +1.0-1.5, moderate +0.3-0.7, pessimistic +0.0-0.2.

**Judge feedback:** Four independent research judges with tool access consistently estimated +0.3-0.8 realistic, citing:
- Suphx oracle-student gap was only ~0.6 dan. PGOI cannot exceed 100% of the gap it approximates.
- Strategy fusion is worst on push/fold decisions (bimodal payoffs), which are exactly the high-entropy decisions PGOI targets.
- Marginal-only sampling (Variant A) misses inter-opponent correlations.
- Oracle was trained on TRUE hidden states; querying with SAMPLED states introduces distributional shift.

**Action:** SEARCH_PGOI.md estimates remain as-is (they are already labeled optimistic/moderate/pessimistic). Add a note that independent evaluation centers on the moderate range (+0.3-0.7) as the expected case, with +1.0-1.5 requiring good Sinkhorn quality AND Variant B learned sampler.

### 4. pMCPA Removed from Inference Plans

**Previous spec:** SEARCH_PGOI.md and other docs mentioned Suphx's Monte Carlo Policy Adaptation (pMCPA) as a potential idle-time technique.

**Why removed:** Judges confirmed pMCPA requires ~100K trajectories per adaptation round (Suphx paper Section 4.3). Even with 90 seconds of idle time, generating 100K rollouts on a single GPU is infeasible. Suphx itself could not deploy pMCPA in real-time on Tenhou.

**The +66% offline WR claim:** This was from offline evaluation against a fixed opponent, NOT online play. The technique was never deployed.

**Action:** Remove pMCPA from inference plans. Keep the Suphx reference in COMMUNITY_INSIGHTS.md as historical context.

### 5. OLSS Naming Removed from PGOI

**Previous issue:** Earlier proposals called Hydra's inference search "OLSS-I" to borrow credibility from the OLSS paper (ICML 2023). Judges correctly identified this as misleading. Real OLSS is game-theoretic subgame solving with pUCT (1000 simulations, tree search). PGOI is oracle consultation with belief sampling -- fundamentally different.

**Action:** Never refer to PGOI as OLSS. PGOI is its own technique: neural posterior-guided oracle inference. The OLSS paper remains a reference for future work on actual subgame solving.

### 6. Sinkhorn Go/No-Go Validation Gate

**New addition:** After Phase 1 BC training (~500 A100-hrs, ~1.6% of total budget), validate Sinkhorn head quality:
- Measure MAE of Sinkhorn marginals vs true tile counts on held-out games
- If MAE < 0.3: PGOI is viable. Continue full pipeline.
- If MAE 0.3-0.5: PGOI may work with more samples (128+). Test before committing.
- If MAE > 0.5: Skip PGOI. Reallocate compute to longer self-play.

This is the cheapest possible validation of the most important novel component.

### 7. Compute Resources Identified

| Resource | Amount | GPU | Status |
|----------|--------|-----|--------|
| TACC Frontera | 667 node-hrs (2,668 GPU-hrs) | RTX 5000 16GB | Applied |
| ACCESS Explore | 400K credits (~6,000 A100-hrs) | A100 40GB (Delta) | Not yet applied, auto-approved |
| ACCESS Discover | 1.5M credits (~22,500 A100-hrs) | A100 40GB (Delta) | Needs 1-page writeup |
| TACC Lonestar6 | Startup allocation (~300 A100-hrs) | A100 40GB / H100 80GB | Not yet applied |
| NAIRR Pilot | ~2,000 GPU-hrs | Various | Needs short form |
| Local | 1x RTX 5070 + 64GB RAM | RTX 5070 12GB | Available |

All sources stackable. Realistic near-term: ~31K A100-equiv hours.

### 8. Multiplayer Convergence (CFR/Nash in 4-Player)

**Finding:** No algorithm has proven Nash convergence for 4-player Mahjong. CFR converges to Coarse Correlated Equilibrium (CCE) in multiplayer, NOT Nash. ACH's Nash guarantee is 2-player zero-sum only.

**Implication:** The "PPO has no Nash guarantee" criticism applies equally to ACH in 4-player, and to LuckyJ's unpublished 4-player system. Every top 4-player Mahjong AI runs on empirics, not theory. This is NOT a weakness specific to Hydra -- it's a universal limitation.

**Source:** Comprehensive survey of Risk & Szafron 2010, QP paper 2025, OMD/MMD results (NeurIPS 2022), DeepNash/R-NaD (ICLR 2023), IESL (IEEE TNNLS 2025). Full analysis at /tmp/multiplayer_convergence.md.

## Honest Assessment

| Target | Probability | Evidence |
|--------|------------|---------|
| 7+ dan (beat Mortal) | ~70% | Proven BC pipeline + larger model + more data |
| 8+ dan (approach Suphx) | ~55% | Requires oracle-critic + ACH self-play to work |
| 9+ dan (strongest OSS) | ~40-50% | Requires significant self-play + some PGOI gains |
| 10+ dan | ~20-30% | Requires PGOI at moderate range + everything else working |
| 10.68+ dan (match LuckyJ) | ~20-25% | Requires optimistic PGOI + strong base + favorable variance |

The plan is GOOD. The target is HARD. These are not contradictory.
