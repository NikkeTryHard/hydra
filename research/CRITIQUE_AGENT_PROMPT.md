# Critique Agent Deep Analysis Prompt

## YOUR ROLE
You are an expert AI researcher with deep knowledge of game theory, reinforcement learning, and imperfect-information games. You have access to the internet and can read papers. You think slowly and carefully. You are known for finding things other researchers miss.

## CONTEXT
We are building HYDRA, a 4-player Riichi Mahjong AI targeting LuckyJ's 10.68 stable dan on Tenhou. Our current design (HYDRA-OMEGA) is at:
- https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md

LuckyJ's stack is: ACH (Actor-Critic Hedge, ICLR 2022) + OLSS-II search (ICML 2023). No oracle guiding. 3-block ResNet. Pure self-play.

Our stack is: 40-block SE-ResNet + PPO + Oracle Guiding + Oracle Critic + AFBS search + SaF + Mixture-SIB + CT-SMC beliefs + Endgame Solver + Population Training.

Budget: 2000 GPU hours on 4x RTX 5000 Ada (261 TFLOPS bf16 tensor each).

---

## QUESTIONS FOR YOU (think deeply on each)

### Q1: Why did you recommend BPR (Bayesian Policy Reuse) and Test-Time Adaptation?

Our research found:
- BPR has NEVER been applied to imperfect-info games at Mahjong's scale (only toy domains like golf club selection)
- Training 32+ specialist policies costs ~16,000 GPU hours (8x our budget)
- Test-time adaptation: ZERO deployed game AI systems update weights at test time (not AlphaGo, AlphaStar, Suphx, Mortal, Pluribus, OpenAI Five)
- Suphx's pMCPA was NOT deployed in their 10-dan Tenhou run (footnote 8 of their paper)
- The search/rollout part of pMCPA (which our AFBS already provides) matters more than the gradient update

Given this evidence, **are you still confident these two approaches are worth building?** If so, explain specifically:
- What evidence did we miss?
- How would you make BPR work within 2000 GPU hours?
- What specific failure mode of our current design does test-time adaptation solve that pondering search doesn't?

References to read:
- Suphx paper (pMCPA details): https://arxiv.org/pdf/2003.13590
- BPR original: Rosman & Ramamoorthy 2016, "Bayesian Policy Reuse"
- GSL (cheap specialists): https://proceedings.mlr.press/v162/ (ICML 2022, "Generalist-Specialist Learning")
- Our BPR feasibility report: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/BPR_FEASIBILITY.md

### Q2: Can we use ACH instead of PPO? How?

ACH (Actor-Critic Hedge) is LuckyJ's training algorithm and gives **+0.4 fan over PPO** (~1-1.5 dan). This is the SINGLE BIGGEST technique gap between us and LuckyJ.

We chose PPO because:
- ACH is complex (weighted CFR + actor-critic hybrid)
- Rudolph et al. 2025 showed tuned PPO is competitive with game-theoretic methods
- We compensate with oracle guiding (+1.7 dan) which LuckyJ doesn't use

But if complexity is free, should we use ACH? Specifically:
1. **Is ACH implementable in Rust/Burn?** It requires: counterfactual value computation, Hedge-style regret minimization, weighted policy updates. Can this be done with standard autodiff?
2. **Does ACH work in 4-player general-sum?** The original paper tests on 2-player and Mahjong. Are there any theoretical issues extending to 4-player?
3. **Can ACH and oracle guiding coexist?** LuckyJ uses ACH WITHOUT oracle guiding. Could we combine both for an even stronger system?
4. **What is the actual ACH update rule?** Give me the exact equations so I can assess implementation difficulty.
5. **Is there a simpler alternative that captures ACH's game-theoretic stability?** E.g., DRDA (which we already plan to use), R-NaD (from DeepNash), or something else?

References to read:
- ACH paper (ICLR 2022): https://openreview.net/pdf?id=DTXZqTNV5nW
- OLSS paper (ICML 2023, has ACH training details): https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- DRDA paper (ICLR 2025): search for "Divergence-Regularized Discounted Aggregation"
- R-NaD / DeepNash: https://www.science.org/doi/10.1126/science.add4679
- PPO reevaluation in IIGs (5600 runs): https://arxiv.org/abs/2502.08938
- Policy gradient convergence in IIGs: https://arxiv.org/abs/2408.00751

### Q3: What are we STILL missing?

Look at our full design (HYDRA_FINAL.md linked above) and LuckyJ's known techniques. What specific weaknesses remain that could prevent us from reaching 10.68 dan? Consider:

1. **The ACH gap** (0.4 fan / ~1.5 dan) -- is our oracle guiding advantage enough to close it?
2. **Search quality** -- is AFBS with robust opponent modeling truly comparable to OLSS-II? OLSS uses 1000 sims with N=1-4 opponent strategies. How does our KL soft-min approach compare?
3. **Training stability** -- DRDA vs ACH for 4-player dynamics. Which is actually more stable?
4. **What does LuckyJ's environmental model do?** They train a separate model with 2400 CPUs + 8 V100s. What is it for? Should we have one?
5. **Network capacity** -- LuckyJ uses 3 blocks and reaches 10.68 dan with search. We use 40 blocks. Is our network too big for 2000 GPU hours? Would a smaller network + more self-play be better?
6. **Any completely new technique** from 2024-2025 papers that neither we nor LuckyJ have?

References for LuckyJ's full stack:
- ACH: https://openreview.net/pdf?id=DTXZqTNV5nW
- OLSS: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- GSCU (opponent modeling, may be used): https://proceedings.mlr.press/v162/fu22b/fu22b.pdf
- Tenhou performance: https://technode.com/2023/07/12/tencents-mahjong-ai-sets-new-gaming-record-on-international-mahjong-platform/

### Q4: The "one weird trick" question

If you could add EXACTLY ONE more technique to HYDRA-OMEGA that would have the highest expected impact on dan rating, what would it be? Not something we already have -- something genuinely new.

Constraints:
- Must be implementable in code (no uncomputable functions)
- Must work within 2000 GPU hours total training budget
- Must not add more than 5ms to inference latency
- Must have at least some theoretical or empirical justification

---

## OUTPUT FORMAT
For each question, provide:
1. Your answer (be specific, not vague)
2. Your confidence level (low/medium/high)
3. The key reference that supports your answer
4. What experiment would prove you right or wrong
