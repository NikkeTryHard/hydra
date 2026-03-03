# Cross-Disciplinary Paradigm Shifts for Mahjong AI

**Date**: 2026-03-02
**Scope**: 7 fields outside game AI evaluated for structural advantages in 4-player Riichi Mahjong
**Verdict**: Active Inference (Friston's Expected Free Energy) is the single most promising paradigm shift

---

## Executive Summary

After searching across information theory, neuroscience, swarm intelligence, causal inference, active inference, game theory, and compression theory, one framework stands above all others: **Friston's Active Inference with explicit epistemic value decomposition**.

The core insight is embarrassingly simple once you see it:

> **Every discard in Mahjong is simultaneously a move AND a signal. The optimal play isn't the one that maximizes your hand's expected value -- it's the one that maximizes expected value PLUS expected information gain about hidden states. Current Mahjong AI has no explicit concept of this second term.**

Current state-of-the-art (Mortal, Suphx) uses a single scalar objective: maximize expected placement score. Active inference decomposes this into two explicit terms that sum naturally, and the framework automatically shifts from information-seeking (early game) to score-maximizing (late game) -- exactly matching expert human play.

---
## Tier 1: Paradigm Shifts

### 1. ACTIVE INFERENCE -- Expected Free Energy Decomposition (THE ONE)

**Source**: Friston, Rigoli, Ognibene, Mathys, Fitzgerald & Pezzulo (2015). "Active inference and epistemic value." *Cognitive Neuroscience*, 6(4), 187-214.
**URL**: https://www.fil.ion.ucl.ac.uk/~karl/Active%20inference%20and%20epistemic%20value.pdf

**Also**: Smith, Friston & Whyte (2022). "A step-by-step tutorial on active inference as a POMDP." *J. Math Psychology*, 107.
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8956124/

**Also**: Parr, Da Costa & Friston (2019). "The value of uncertainty: An active inference perspective." *Behavioral Brain Research*.
**URL**: https://europepmc.org/article/MED/30940252

#### The Math

In active inference, an agent selects policies (action sequences) by minimizing **Expected Free Energy (EFE)**. The negative EFE (policy quality) decomposes cleanly:

```
Q(pi) = E[extrinsic_value] + E[epistemic_value]
```

Formally, for policy pi at future time tau:

```
Q_tau(pi) = E_Q(o|pi)[ln P(o|m)]           -- extrinsic: prefer winning outcomes
           + E_Q(o|pi)[KL(Q(s|o,pi)||Q(s|pi))]  -- epistemic: information gain about hidden states
```

The epistemic term is the **expected KL divergence** between your posterior beliefs about hidden states WITH vs WITHOUT future observations. It quantifies: "how much will I learn about the game state by taking this action and observing the result?"

#### Why This Is Revolutionary for Mahjong

Current Mahjong AI asks: "Which discard maximizes my expected score?"

Active inference asks: "Which discard maximizes my expected score AND maximizes what I learn about opponent hands?"

Consider two discards that are EV-equivalent for hand development:
- **Discard A**: 3-man (a tile near what South might need based on their melds)
- **Discard B**: North wind (safe, reveals nothing)

Current AI sees these as equal. Active inference sees Discard A as superior because:
- If South calls chi: you learn their hand direction (high epistemic value realized)
- If South doesn't call: you learn they're NOT pursuing that shape ("absence of evidence" -- see Section 2 below)
- Either way, your belief state about South's hand gets sharper

The framework ALSO naturally handles the flip side: it penalizes actions that leak information about YOUR hand. Making a risky chi call reveals your hand direction, which reduces YOUR epistemic advantage.

#### Phase Dynamics

The EFE framework naturally produces the early-game vs late-game shift that experts exhibit:
- **Early game** (high uncertainty): epistemic value dominates. Optimal play is information-seeking.
- **Late game** (beliefs sharpened): extrinsic value dominates. Optimal play is score-maximizing.

This is exactly what pro players describe: "In early rounds I'm reading, in later rounds I'm executing."

#### Implementation Sketch

1. Maintain explicit **belief distributions** over opponent hand compositions (not just a neural net hidden state)
2. For each candidate discard, compute:
   - Extrinsic value: expected hand improvement + expected score (existing approach)
   - Epistemic value: expected information gain = how much the posterior over opponent hands will change
3. Weight by game phase (or let the EFE handle it naturally through decreasing uncertainty)
4. Select the discard that maximizes the sum

The epistemic value could be computed via Monte Carlo sampling of possible opponent responses to each discard, measuring the average KL divergence in belief state.


### 2. BAYESIAN ABSENT-EVIDENCE REASONING -- "The Dog That Didn't Bark"

**Source**: Hsu, Griffiths & Schreiber (2017). "When Absence of Evidence Is Evidence of Absence: Rational Inferences From Absent Data." *Cognitive Science*, 41(5), 1155-1167.
**URL**: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12356

**Also**: "The Dog that Didn't Bark: Bayesian Approaches to Reasoning from Censored Data"
**URL**: https://www.researchgate.net/publication/371180677

#### The Core Insight

In Mahjong, the MOST informative signals are often the calls that DIDN'T happen. Current neural networks learn correlational patterns from observed events but cannot explicitly reason about the informational content of non-events.

The Bayesian framework formalizes this:

```
P(opponent_has_X | didn't_call_on_Y) proportional to
    P(didn't_call_on_Y | has_X) * P(has_X)
```

Where `P(didn't_call_on_Y | has_X)` depends on how natural the call would be. If calling chi on 5-man would be the obvious play for someone pursuing pinfu, then NOT calling is very strong evidence they don't have that shape.

Key finding from the cognitive science: **The informativeness of absence scales with the expected probability of the event**. Rare non-events tell you little. Common non-events tell you a lot.

#### Mahjong Application

Turn-by-turn "absence tracking" for each opponent:
- Track which tiles were discarded that could have been called (chi/pon/kan)
- For each non-call, compute: "How surprising is it that they didn't call this?"
- High-surprise non-calls = strong evidence about what they DON'T have
- Accumulate these as Bayesian updates to opponent hand distributions

This connects directly to Pearl's Causal Hierarchy (Level 3 -- counterfactual reasoning):
"IF opponent had been waiting on X, THEN they WOULD have called chi on Y, BUT they didn't, THEREFORE they likely don't have X."

Current AI gets at this implicitly through neural network pattern matching, but explicit Bayesian absent-evidence tracking could be more sample-efficient and more transparent.

### 3. GAME THEORY OF MIND -- Recursive Opponent Modeling

**Source**: Yoshida, Dolan & Friston (2008). "Game Theory of Mind." *PLoS Computational Biology*, 4(12), e1000254.
**URL**: https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1000254&type=printable

#### The Framework

Formalizes recursive belief reasoning (I think they think I think...) as nested value functions:
- **Level 0**: Play myopically, ignore opponents
- **Level 1**: Best response to Level-0 opponents
- **Level 2**: Best response to Level-1 opponents (who model you as Level-0)
- **Level K**: Best response to Level-(K-1) opponents

The paper shows that in sequential games, agents can INFER the sophistication level of opponents from observed play, then adapt by playing one level above.

#### Mahjong Application

Current Mahjong AI has ZERO recursive reasoning. It plays its hand against a statistical model of "average" opponents. But expert Mahjong involves:
- "They see I called pon on chun -- they think I'm going for honitsu"
- "Since they think I'm going for honitsu, they'll hold back honor tiles"  
- "Since they're holding honor tiles, I can safely discard them for tanyao instead"

Even 2-level ToM would be a structural advantage. The paper's finding that subjects played at relatively high sophistication levels suggests this is what humans actually do.

---

## Tier 2: Significant but More Incremental

### 4. MAHJONG IS NOT ZERO-SUM

**Source**: Riichi Wiki on Oka/Uma placement scoring
**URL**: https://riichi.wiki/Oka_and_uma

**Also**: Computing Nash Equilibria in Multiplayer DAG-Structured Stochastic Games
**URL**: https://link.springer.com/chapter/10.1007/978-3-030-90370-1_1

**Also**: Opponent Modeling in Multiplayer Imperfect-Information Games
**URL**: https://dl.acm.org/doi/10.1145/3719545.3721108 (notes NE is PPAD-hard for multiplayer non-zero-sum)

The uma/oka placement bonus system means Mahjong is formally a **general-sum** game, not zero-sum. The implications are deep:

- **Nash equilibrium is PPAD-hard** for 4-player general-sum games. No efficient algorithm exists.
- The standard CFR approach (which converges for 2-player zero-sum) has no convergence guarantees.
- **Correlated equilibrium** might be the correct solution concept (polynomial-time computable).
- Placement incentives create non-obvious dynamics: sometimes optimal play is to ENSURE 2nd place rather than risk 3rd chasing 1st. This is already somewhat handled by reward shaping in current AI, but the formal implications run deeper.

### 5. PEARL'S CAUSAL HIERARCHY

**Source**: "From Probability to Counterfactuals: the Increasing Complexity in Pearl's Causal Hierarchy"
**URL**: https://arxiv.org/abs/2405.07373

**Also**: "Reasoning about causality in games"
**URL**: https://www.sciencedirect.com/science/article/pii/S0004370223000656

Three levels of reasoning:
1. **Observational** (association): P(opponent has X | discards seen) -- current AI does this
2. **Interventional**: "If I discard Y, how does the probability change?" -- partially captured by lookahead
3. **Counterfactual**: "If they HAD X, they WOULD have done Y by now" -- not captured at all

The complexity result: counterfactual satisfiability is NEXP-complete vs NP^PP for observational (under summation languages). This means full counterfactual reasoning is computationally brutal, but approximate counterfactual reasoning (which is what humans do) could be tractable and powerful.

### 6. EXPERTISE = DEEPER SEARCH, NOT BETTER HEURISTICS

**Source**: van Opheusden, Kuperwajs, Galbiati et al. (2023). "Expertise increases planning depth in human gameplay." *Nature*, 620, 1004-1008.
**URL**: https://www.nature.com/articles/s41586-023-06124-2

Key quantitative findings:
- Planning depth vs Elo: rho = 0.62, p < 0.001 (strong positive)
- Feature-drop rate vs Elo: rho = -0.73, p < 0.001 (experts miss fewer key features)
- **Heuristic quality vs Elo: rho = 0.11, p = 0.088 (NOT significant)**

Translation: Better players search DEEPER with more reliable feature detection, NOT with better position evaluation. This suggests that for Mahjong AI, investing in search depth (MCTS/lookahead) may matter more than investing in a better value network. Mortal currently does zero search -- it's pure policy network. Adding even shallow search could be disproportionately valuable.

---

## Tier 3: Interesting but Not Paradigm-Shifting

### 7. SWARM / ENSEMBLE APPROACHES

**Source**: "Ensemble strategy learning for imperfect information games." *Neurocomputing*, 2023.
**URL**: https://www.sciencedirect.com/science/article/pii/S0925231223003648

The idea of multiple specialized agents (offense, defense, calling) composed into one player is validated by this paper. Multiple paradigms (rule-based, game-theoretic, RL) combined outperform any single paradigm. However, this is an engineering improvement, not a conceptual breakthrough.

### 8. COMPRESSION / MDL

**Source**: "Bridging Kolmogorov Complexity and Deep Learning"
**URL**: https://arxiv.org/abs/2509.22445

MDL could help identify the *simplest* strategy that explains expert play, which would be useful for interpretability and knowledge distillation. But it's an analysis tool, not a training paradigm.

### 9. MAHJONG BRAIN IMAGING

**Source**: "Comparison of Cortical Activation during Mahjong Game Play" (fNIRS study)
**URL**: https://www.walshmedicalmedia.com/open-access/comparison-of-cortical-activation-during-mahjong-game-play-in-a-video-game-setting-and-a-reallife-setting-2161-1009-1000164.pdf

Found: Real-life Mahjong activates Broca's area, somatosensory cortex, angular gyrus, and Wernicke's area more than digital Mahjong. This suggests Mahjong involves significant *linguistic/symbolic processing* (angular gyrus is involved in number/symbol manipulation). However, this study compared settings, not expertise levels -- no expert vs novice comparison exists in the Mahjong neuroscience literature.

---

## The Unified Framework: Active Inference + Absent Evidence + ToM

The three Tier 1 insights aren't independent -- they're facets of the same underlying framework:

```
Active Inference (Friston 2015)
    |
    +-- Epistemic Value: "What do I learn from this action?"
    |       |
    |       +-- Positive evidence: Opponent calls/doesn't call on my discard
    |       +-- Absent evidence (Hsu 2017): What they DIDN'T do is informative
    |
    +-- Extrinsic Value: "Does this move me toward winning?"
    |
    +-- Game Theory of Mind (Yoshida/Dolan/Friston 2008):
            "What do THEY think I'm doing? How does that affect their actions?"
            This feeds back into epistemic value -- my discards are probes
            whose informativeness depends on opponent sophistication
```

The Yoshida/Dolan/Friston 2008 paper was literally co-authored by Friston. It's all the same program. Active inference IS theory of mind IS epistemic foraging. They're not separate insights -- they're one coherent mathematical framework for decision-making under uncertainty with other agents.

---

## Concrete Recommendation for Hydra

### Phase 1: Explicit Belief States (low-hanging fruit)
Add a parallel belief module that maintains probability distributions over opponent hand compositions, updated each turn via Bayesian inference from observed discards + non-calls. This already exists implicitly in the neural network hidden state but making it explicit enables:
- Entropy-based phase detection (early = high entropy, late = low entropy)
- Targeted uncertainty reduction

### Phase 2: Epistemic Value Head (the paradigm shift)
Add a 6th output head: **Epistemic Value**. For each of the 34 possible discards, predict the expected information gain (reduction in opponent hand entropy). Train this with a self-play objective where the reward includes KL divergence between pre-discard and post-discard belief states.

### Phase 3: Absent Evidence Module
Track per-opponent "absence surprisal" -- for each opponent, accumulate the information content of non-calls as the game progresses. Feed this as an additional feature channel.

### Phase 4: Recursive ToM (aspirational)
Model opponents as having their own policies that depend on what they think you're doing. Even level-1 ToM (modeling opponents as modeling you) would be novel in Mahjong AI.

---

## Key References

1. Friston et al. (2015). "Active inference and epistemic value." Cognitive Neuroscience, 6(4).
2. Parr, Da Costa & Friston (2019). "The value of uncertainty." Behavioral Brain Research.
3. Smith, Friston & Whyte (2022). "Active inference as POMDP." J. Math Psychology.
4. Yoshida, Dolan & Friston (2008). "Game Theory of Mind." PLoS Comp Bio.
5. Hsu, Griffiths & Schreiber (2017). "When Absence of Evidence Is Evidence of Absence." Cognitive Science.
6. van Opheusden et al. (2023). "Expertise increases planning depth." Nature, 620.
7. Pearl's Causal Hierarchy -- arXiv:2405.07373
8. Information gathering in POMDPs using active inference -- Springer JAAMAS 2024
