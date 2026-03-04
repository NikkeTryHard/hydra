# BPR Feasibility Report for Hydra Mahjong AI

## TL;DR

**Verdict: BPR in its original form is NOT worth it for Hydra. But a modified "GSL-style" approach with shared backbone + specialist heads IS feasible and evidence-backed within 2000 GPU-hours.**

The original BPR was tested on toy domains (golf, surveillance grids), never on complex imperfect-info games. The compute cost of training 32+ fully separate specialist policies would blow the budget. However, the Generalist-Specialist Learning (GSL) framework from ICML 2022 shows you can get the benefits of specialist policies WITHOUT multiplying training cost, by initializing specialists from a shared generalist and distilling back.

---

## 1. Original BPR Paper (Rosman & Ramamoorthy, 2016)

**Paper**: "Bayesian Policy Reuse" -- Machine Learning journal, Springer  
**Links**: [arXiv](https://arxiv.org/abs/1505.00284) | [Springer](https://link.springer.com/article/10.1007/s10994-016-5547-y)

### Domains Tested
| Domain | Library Size | Complexity |
|--------|-------------|------------|
| Golf Club Selection | **4 policies** | Toy -- pick 1 of 4 clubs for unknown course |
| Online Personalisation | **20 policies** | Medium -- identify user language preference |
| Surveillance/Drone | **68 policies** | Grid world (26x26) -- navigate to 1 of 68 targets |

### Key Results
- BPR converged in ~15 episodes for the 68-policy surveillance task, roughly **4x faster than brute-force** testing each policy
- BPR-EI (Expected Improvement) hit near-zero regret within 5 episodes vs UCB1 needing 50+
- With the golf task, BPR got within 10-15 yards by shot 2, while single-club generalist stayed at 25-50 yards

### Critical Limitations for Hydra
- **All domains were fully observable** or had simple partial observability (noisy distance readings)
- **Policies were pre-trained offline** on known task distributions -- the paper doesn't address the cost of building the library
- **No imperfect-information games tested** -- no hidden opponent hands, no strategic deception
- The Bayesian belief update assumes you can estimate P(signal | policy, task) -- in Mahjong with hidden info, this distribution is extremely noisy

---

## 2. BPR in Imperfect-Info Games: Almost No Evidence

### What Exists
- **CABPR** (Context-Aware BPR, 2022): Extends BPR with intra-episode belief updates for tracking multi-strategic opponents. Published in Applied Soft Computing. Tested on **simple repeated games** (like iterated prisoner's dilemma style), NOT complex card/board games.
  - Paper: [ACM DL](https://dl.acm.org/doi/10.1016/j.asoc.2022.108715)

- **Efficient BPR with Scalable Observation Model** (IEEE TNNLS, 2023): Proposes using state-transition samples instead of episodic returns for observation signals. Claims better scalability but tests on **standard RL benchmarks**, not games.
  - Paper: [IEEE](https://ieeexplore.ieee.org/document/10149182)

- **Multi-Agent BPR** (Neural Networks, 2026): Extends BPR to multi-agent settings with partial observability. The most recent work, but focused on general MARL settings.
  - Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608026002170)

### What Doesn't Exist
- **No BPR applied to Mahjong, poker, or any complex card game**
- **No BPR with deep RL policy libraries at game-scale** (millions of parameters per policy)
- **No evidence BPR's Bayesian belief mechanism works when observation signals are as noisy as Mahjong discards**

### Closest Relevant Work: Opponent Modeling in Poker/Mahjong
- Poker opponent modeling uses different approaches entirely: style classification + counter-strategy, not BPR
- Suphx (Mahjong) uses **runtime policy adaptation** (see section 4 below) which is conceptually similar but architecturally very different from BPR

---

## 3. Cheap Policy Libraries: The GSL Approach (Best Evidence)

**Paper**: "Improving Policy Optimization with Generalist-Specialist Learning" (ICML 2022)  
**Links**: [arXiv](https://arxiv.org/abs/2206.12984) | [ICML](https://proceedings.mlr.press/v162/jia22a.html)

This is the strongest evidence for whether a specialist library can beat a generalist, and how to build one cheaply.

### How GSL Works
1. Train a **generalist** policy on all task variations
2. When generalist plateaus, **clone its weights** to initialize N specialists
3. Each specialist trains on a **small subset** of variations (e.g., 4 levels each)
4. Specialists generate demonstrations, which are **distilled back** into the generalist via imitation learning (DAPG/GAIL)

### Key Numbers

| Benchmark | Specialists | Vars/Specialist | Generalist Score | GSL Score | Improvement |
|-----------|-------------|-----------------|-----------------|-----------|-------------|
| BigFish (Procgen) | 75 | 4 | 24.6 | **31.1** | +26% |
| StarPilot (Procgen) | 75 | 4 | 39.4 | **49.5** | +26% |
| BossFight (Procgen) | 75 | 4 | 8.6 | **11.3** | +31% |
| MT-10 (Meta-World) | varied | varied | 58.4% | **77.5%** | +19pp |
| MT-50 (Meta-World) | varied | varied | 31.1% | **43.5%** | +12pp |
| PushChair (ManiSkill) | 8 | varied | -2.97k | **-2.78k** | improved |

### Critical Finding: Fixed Sample Budget
The specialists DON'T cost extra samples. From the paper: "Each of the N_s specialists has a budget of 200 million/N_s samples to train PPO, so that the total sample budget is fixed." The training compute is **redistributed**, not multiplied.

### Optimal Specialist Count
From ablation (Section 6.1, Figure 6):
- K=4 variations per specialist (= 64 specialists for 256 levels) achieved best final performance
- K=256 (pure generalist) learns faster initially but **plateaus earlier**
- K=16 is a reasonable middle ground with good final performance and reasonable setup
- **Sweet spot: 4-16 variations per specialist**

---

## 4. Suphx: What Actually Works for Mahjong

**Paper**: "Suphx: Mastering Mahjong with Deep Reinforcement Learning" (Microsoft Research)  
**Link**: [arXiv](https://arxiv.org/abs/2003.13590)

### Architecture: 5 Action-Type Specialists (NOT opponent-style)
Suphx does NOT use opponent-style specialists. It uses **5 separate models** for different action types:
1. Discard model (which tile to throw)
2. Riichi model (declare riichi or not)
3. Chow model (claim chow or not)
4. Pong model (claim pong or not)
5. Kong model (claim kong or not)

Each has its own network (50-block CNN with 256 filters). Plus a rule-based winning model.

### Runtime Policy Adaptation (pMCPA)
Instead of selecting from a library of pre-trained strategies, Suphx does:
1. When a round starts, sample 100,000 random trajectories using Monte Carlo
2. Fine-tune the offline policy **specifically for this hand** using those trajectories
3. Use the fine-tuned policy for this round only
4. Reset to base policy for next round

Result: **66% win rate** for adapted version vs non-adapted version. This is massive.

### Training Cost
- 44 GPUs (4 Titan XP + 40 K80) for 2 days = ~2,112 GPU-hours (K80-equivalent)
- In modern A100 equivalents: roughly **200-300 A100-hours** (K80 is ~10x slower)
- This is for ONE generalist agent, not a library

### Key Insight for Hydra
Suphx's approach shows that **runtime adaptation per-hand beats having pre-trained opponent specialists**. The game state varies so much per hand that adapting to the current situation is more valuable than adapting to opponent style.

---

## 5. Shared-Backbone Specialists (LoRA-style): State of the Art

### Does Anyone Do BPR + LoRA? 
**No.** I found zero papers combining BPR with LoRA-style adapters specifically. This is a gap in the literature.

### What Exists in Adjacent Areas
- **Multi-Task RL with MoE** (IEEE TNNLS, 2023): Attention-based Mixture of Experts for multi-task RL. Shared expert networks with attention-weighted routing. Tested on Meta-World (50 tasks) and MuJoCo (4 tasks). Shows that soft expert routing beats naive parameter sharing.
  - [IEEE](https://ieeexplore.ieee.org/document/10111062)

- **Efficient Multi-Task RL with Cross-Task Policy Guidance** (NeurIPS 2025): Shared features across tasks with adaptive experience replay. Shows shared feature extraction works when tasks are related.
  - [OpenReview](https://openreview.net/forum?id=3qUks3wrnH)

- **RLDG** (RSS 2025): Distills multiple RL specialist policies into a single generalist robot policy. Shows the specialist-to-generalist distillation pipeline works in practice.
  - [arXiv](https://arxiv.org/html/2412.09858)

- **LoRA for RL in LLMs** (verl/mLoRA): The LLM world has solved multi-adapter serving. SLoRA can serve thousands of concurrent LoRA adapters from one base model. mLoRA can train multiple adapters concurrently on shared base weights. This infrastructure exists but hasn't been applied to game RL.

### The Obvious Architecture Nobody Has Published
Shared SE-ResNet backbone + per-situation LoRA adapters (rank 4-16) for specialization. Each adapter adds ~0.1-1% parameters. Training cost: one full backbone training + cheap adapter fine-tuning per specialist. This is the approach Hydra should consider.

---

## 6. How Many Specialists Do You Actually Need?

### Evidence Summary
| Source | Domain | Optimal Count | Finding |
|--------|--------|---------------|---------|
| BPR (original) | Surveillance | 68 | More policies = less regret, but diminishing returns |
| GSL (ICML 2022) | Procgen | 64 (K=4) | Best final performance, but K=16 is good tradeoff |
| GSL (ICML 2022) | ManiSkill | 8 | Sufficient for manipulation tasks |
| Suphx | Mahjong | 5 | Action-type specialists, not opponent-type |
| MoE-MTRL | Meta-World | ~8 experts | Attention-weighted soft routing |

### Practical Recommendation for Hydra
- **2 specialists** = too few, barely distinguishes from generalist
- **8 specialists** = sweet spot for cost/benefit in game AI
- **16 specialists** = diminishing returns unless task variation is extreme
- **32+ specialists** = only justified if you can amortize via shared backbone
- **128 specialists** = no evidence this helps; likely overfits to narrow situations

---

## 7. Budget Feasibility: 2000 GPU-Hours

### Reference Points
| System | GPU-Hours (modern equiv) | What It Trained |
|--------|------------------------|-----------------|
| Suphx | ~200-300 A100-hrs | 1 generalist Mahjong agent (5 action models) |
| Mortal | Unknown (self-play based) | 1 generalist Riichi agent |
| Hydra target | 2000 GPU-hrs | SE-ResNet 40-block, 16.5M params |

### Scenario A: Naive BPR (32 separate specialists)
- 32 x full training runs = 32 x ~500-600 GPU-hrs = **16,000-19,200 GPU-hrs**
- **VERDICT: WAY over budget. Not feasible.**

### Scenario B: GSL-style (shared init, parallel training, distillation)
- Phase 1: Train generalist backbone: **800-1000 GPU-hrs**
- Phase 2: Fork 8-16 specialists (init from generalist), train on subsets: **400-600 GPU-hrs** (shared sample budget)
- Phase 3: Distill specialist knowledge back to generalist: **100-200 GPU-hrs**
- Phase 4: Final fine-tuning of generalist: **100-200 GPU-hrs**
- **TOTAL: ~1400-2000 GPU-hrs. FEASIBLE but tight.**

### Scenario C: Shared backbone + LoRA adapters (recommended)
- Phase 1: Train generalist backbone fully: **1000-1200 GPU-hrs**
- Phase 2: Train 8-16 LoRA adapters (rank 8-16, ~0.5% params each): **200-400 GPU-hrs total**
- Phase 3: Runtime adapter selection or merging: **~0 additional training**
- **TOTAL: ~1200-1600 GPU-hrs. FEASIBLE with headroom.**

### Scenario D: Suphx-style runtime adaptation (simplest)
- Phase 1: Train single strong generalist with oracle guiding: **1200-1500 GPU-hrs**
- Phase 2: Runtime Monte-Carlo policy adaptation (no extra training): **0 GPU-hrs**
- Phase 3: Oracle critic training: **300-500 GPU-hrs**
- **TOTAL: ~1500-2000 GPU-hrs. FEASIBLE. Already in Hydra's design.**

---

## 8. Recommendations for Hydra

### Don't Do
1. **Don't implement BPR as described in the original paper** -- it was designed for task identification in simple domains, not for adapting to opponent strategies in imperfect-info games
2. **Don't train 32+ fully separate specialist networks** -- blows the compute budget by 8-10x
3. **Don't use opponent-style specialists** (aggressive/defensive/etc.) -- Suphx showed per-hand adaptation beats per-opponent adaptation in Mahjong

### Consider Doing (Priority Order)
1. **Suphx-style runtime pMCPA** (already aligned with Hydra's oracle pondering ExIt design) -- strongest evidence for Mahjong specifically, 66% winrate improvement
2. **GSL-style specialist-to-generalist distillation** during training -- 20-30% improvement on hard benchmarks, fixed sample budget
3. **LoRA adapter library** for situation-type specialization (e.g., defensive play, tenpai racing, early-game tile efficiency) -- novel but theoretically sound, compute-efficient

### The Honest Assessment
The Hydra design doc already includes **oracle pondering ExIt**, which is essentially a more principled version of runtime policy adaptation. The oracle critic sees perfect information and guides the main policy -- this is conceptually similar to Suphx's oracle guiding. Adding BPR on top of this would be engineering complexity for uncertain marginal gains.

**Bottom line: Spend the 2000 GPU-hours training one excellent generalist with oracle pondering, not a library of mediocre specialists.**

---

## Sources
1. Rosman & Ramamoorthy. "Bayesian Policy Reuse." Machine Learning, 2016. [arXiv:1505.00284](https://arxiv.org/abs/1505.00284)
2. Jia et al. "Improving Policy Optimization with Generalist-Specialist Learning." ICML 2022. [arXiv:2206.12984](https://arxiv.org/abs/2206.12984)
3. Li et al. "Suphx: Mastering Mahjong with Deep Reinforcement Learning." 2020. [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
4. Yang et al. "Context-Aware Bayesian Policy Reuse." Applied Soft Computing, 2022. [ACM](https://dl.acm.org/doi/10.1016/j.asoc.2022.108715)
5. Cheng et al. "Efficient Bayesian Policy Reuse With a Scalable Observation Model." IEEE TNNLS, 2023. [IEEE](https://ieeexplore.ieee.org/document/10149182)
6. "Multi-Task RL With Attention-Based MoE." IEEE TNNLS, 2023. [IEEE](https://ieeexplore.ieee.org/document/10111062)
7. "Efficient Multi-Agent Policy Adaptation with BPR." Neural Networks, 2026. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608026002170)
