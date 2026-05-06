# Cross-Disciplinary Paradigm Shifts for Mahjong AI

**Date**: 2026-03-02
**Scope**: 7 non-game-AI fields checked for structural edge in 4-player Riichi Mahjong
**Verdict**: Active Inference (Friston's Expected Free Energy) = strongest paradigm shift

---

## Executive Summary

After search across information theory, neuroscience, swarm intelligence, causal inference, active inference, game theory, compression theory, one framework above rest: **Friston's Active Inference with explicit epistemic value decomposition**.

Core insight simple once seen:

> **Every discard in Mahjong is both move AND signal. Optimal play not move maximizing hand EV alone -- it maximizes EV PLUS expected information gain about hidden states. Current Mahjong AI lacks explicit concept of second term.**

Current SOTA (Mortal, Suphx) use one scalar objective: maximize expected placement score. Active inference splits this into two explicit terms that sum cleanly, framework then shifts automatically from information-seeking (early game) to score-maximizing (late game) -- same as expert human play.

---
## Tier 1: Paradigm Shifts

### 1. ACTIVE INFERENCE -- Expected Free Energy Decomposition (THE ONE)

**Source**: Friston, Rigoli, Ognibene, Mathys, Fitzgerald & Pezzulo (2015). "Active inference and epistemic value." *Cognitive Neuroscience*, 6(4), 187-214.
**URL**: https://www.fil.ion.ucl.ac.uk/~karl/Active%20inference%20and%20epistemic%20value.pdf

**Also**: Smith, Friston & Whyte (2022). step-by-step tutorial on active inference as POMDP." *J. Math Psychology*, 107.
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8956124/

**Also**: Parr, Da Costa & Friston (2019). value of uncertainty: active inference perspective." *Behavioral Brain Research*.
**URL**: https://europepmc.org/article/MED/30940252

#### The Math

In active inference, agent selects policies (action sequences) by minimizing **Expected Free Energy (EFE)**. Negative EFE (policy quality) decomposes cleanly:

```
Q(pi) = E[extrinsic_value] + E[epistemic_value]
```

Formally, for policy pi at future time tau:

```
Q_tau(pi) = E_Q(o|pi)[ln P(o|m)]           -- extrinsic: prefer winning outcomes
           + E_Q(o|pi)[KL(Q(s|o,pi)||Q(s|pi))]  -- epistemic: information gain about hidden states
```

Epistemic term = **expected KL divergence** between posterior beliefs about hidden states WITH vs WITHOUT future observations. Quantifies: "how much will I learn about game state by taking this action and observing result?"

#### Why This Is Revolutionary for Mahjong

Current Mahjong AI asks: "Which discard maximizes expected score?"

Active inference asks: "Which discard maximizes expected score AND maximizes what I learn about opponent hands?"

Consider two discards EV-equivalent for hand development:
- **Discard 3-man (tile near what South may need from their melds)
- **Discard B**: North wind (safe, reveals nothing)

Current AI sees equality. Active inference sees Discard superior because:
- If South calls chi: learn their hand direction (high epistemic value realized)
- If South does not call: learn they are NOT pursuing that shape ("absence of evidence" -- see Section 2 below)
- Either way, belief state about South's hand sharpens

Framework ALSO handles flip side naturally: penalizes actions leaking information about YOUR hand. Risky chi call reveals hand direction, reducing YOUR epistemic edge.

#### Phase Dynamics

EFE framework naturally yields early-game vs late-game shift experts show:
- **Early game** (high uncertainty): epistemic value dominates. Optimal play = information-seeking.
- **Late game** (beliefs sharpened): extrinsic value dominates. Optimal play = score-maximizing.

Exactly what pros describe: "In early rounds I'm reading, in later rounds I'm executing."

#### Implementation Sketch

1. Maintain explicit **belief distributions** over opponent hand compositions (not only neural-net hidden state)
2. For each candidate discard, compute:
   - Extrinsic value: expected hand improvement + expected score (existing approach)
   - Epistemic value: expected information gain = how much posterior over opponent hands changes
3. Weight by game phase (or let EFE handle naturally through falling uncertainty)
4. Select discard maximizing sum

Epistemic value could be computed via Monte Carlo sampling of possible opponent responses to each discard, measuring average KL divergence in belief state.


### 2. BAYESIAN ABSENT-EVIDENCE REASONING -- "The Dog That Didn't Bark"

**Source**: Hsu, Griffiths & Schreiber (2017). "When Absence of Evidence Is Evidence of Absence: Rational Inferences From Absent Data." *Cognitive Science*, 41(5), 1155-1167.
**URL**: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12356

**Also**: Dog that Didn't Bark: Bayesian Approaches to Reasoning from Censored Data"
**URL**: https://www.researchgate.net/publication/371180677

#### The Core Insight

In Mahjong, most informative signals often calls that DIDN'T happen. Current neural nets learn correlational patterns from observed events but cannot explicitly reason about informational content of non-events.

Bayesian framework formalizes this:

```
P(opponent_has_X | didn't_call_on_Y) proportional to
    P(didn't_call_on_Y | has_X) * P(has_X)
```

Where `P(didn't_call_on_Y | has_X)` depends on how natural call would be. If calling chi on 5-man would be obvious play for someone pursuing pinfu, then NOT calling is strong evidence they lack that shape.

Key finding from cognitive science: **Informativeness of absence scales with expected probability of event**. Rare non-events say little. Common non-events say much.

#### Mahjong Application

Turn-by-turn "absence tracking" for each opponent:
- Track which tiles were discarded that could have been called (chi/pon/kan)
- For each non-call, compute: "How surprising that they didn't call this?"
- High-surprise non-calls = strong evidence about what they DON'T have
- Accumulate as Bayesian updates to opponent hand distributions

Connects directly to Pearl's Causal Hierarchy (Level 3 -- counterfactual reasoning):
"IF opponent had been waiting on X, THEN they WOULD have called chi on Y, BUT they didn't, THEREFORE they likely don't have X."

Current AI gets this implicitly through neural-net pattern matching, but explicit Bayesian absent-evidence tracking may be more sample-efficient and transparent.

### 3. GAME THEORY OF MIND -- Recursive Opponent Modeling

**Source**: Yoshida, Dolan & Friston (2008). "Game Theory of Mind." *PLoS Computational Biology*, 4(12), e1000254.
**URL**: https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1000254&type=printable

#### The Framework

Formalizes recursive belief reasoning (I think they think I think...) as nested value functions:
- **Level 0**: Play myopically, ignore opponents
- **Level 1**: Best response to Level-0 opponents
- **Level 2**: Best response to Level-1 opponents (who model you as Level-0)
- **Level K**: Best response to Level-(K-1) opponents

Paper shows that in sequential games, agents can INFER sophistication level of opponents from observed play, then adapt by playing one level above.

#### Mahjong Application

Current Mahjong AI has ZERO recursive reasoning. It plays its hand against statistical model of "average" opponents. But expert Mahjong involves:
- "They see I called pon on chun -- they think I'm going for honitsu"
- "Since they think I'm going for honitsu, they'll hold back honor tiles"
- "Since they're holding honor tiles, I can safely discard them for tanyao instead"

Even 2-level ToM would be structural edge. Paper's finding that subjects played at relatively high sophistication levels suggests humans do this.

---

## Tier 2: Significant but More Incremental

### 4. MAHJONG IS NOT ZERO-SUM

**Source**: Riichi Wiki on Oka/Uma placement scoring
**URL**: https://riichi.wiki/Oka_and_uma

**Also**: Computing Nash Equilibria in Multiplayer DAG-Structured Stochastic Games
**URL**: https://link.springer.com/chapter/10.1007/978-3-030-90370-1_1

**Also**: Opponent Modeling in Multiplayer Imperfect-Information Games
**URL**: https://dl.acm.org/doi/10.1145/3719545.3721108 (notes NE is PPAD-hard for multiplayer non-zero-sum)

Uma/oka placement bonus system means Mahjong formally **general-sum**, not zero-sum. Implications deep:

- **Nash equilibrium is PPAD-hard** for 4-player general-sum games. No efficient algorithm known.
- Standard CFR approach (converges for 2-player zero-sum) has no convergence guarantees.
- **Correlated equilibrium** may be correct solution concept (polynomial-time computable).
- Placement incentives create non-obvious dynamics: sometimes optimal play = ENSURE 2nd rather than risk 3rd chasing 1st. Current AI partly handles this via reward shaping, but formal implications go deeper.

### 5. PEARL'S CAUSAL HIERARCHY

**Source**: "From Probability to Counterfactuals: Increasing Complexity in Pearl's Causal Hierarchy"
**URL**: https://arxiv.org/abs/2405.07373

**Also**: "Reasoning about causality in games"
**URL**: https://www.sciencedirect.com/science/article/pii/S0004370223000656

Three levels of reasoning:
1. **Observational** (association): P(opponent has X | discards seen) -- current AI does this
2. **Interventional**: "If I discard Y, how does probability change?" -- partly captured by lookahead
3. **Counterfactual**: "If they HAD X, they WOULD have done Y by now" -- not captured at all

Complexity result: counterfactual satisfiability is NEXP-complete vs NP^PP for observational (under summation languages). So full counterfactual reasoning is computationally brutal, but approximate counterfactual reasoning (what humans do) may be tractable and powerful.

### 6. EXPERTISE = DEEPER SEARCH, NOT BETTER HEURISTICS

**Source**: van Opheusden, Kuperwajs, Galbiati et al. (2023). "Expertise increases planning depth in human gameplay." *Nature*, 620, 1004-1008.
**URL**: https://www.nature.com/articles/s41586-023-06124-2

Key quantitative findings:
- Planning depth vs Elo: rho = 0.62, p < 0.001 (strong positive)
- Feature-drop rate vs Elo: rho = -0.73, p < 0.001 (experts miss fewer key features)
- **Heuristic quality vs Elo: rho = 0.11, p = 0.088 (NOT significant)**

Translation: Better players search DEEPER with more reliable feature detection, NOT with better position evaluation. Suggests for Mahjong AI, investment in search depth (MCTS/lookahead) may matter more than better value net. Mortal does zero search -- pure policy net. Even shallow search may give outsized value.

---

## Tier 3: Interesting but Not Paradigm-Shifting

### 7. SWARM / ENSEMBLE APPROACHES

**Source**: "Ensemble strategy learning for imperfect information games." *Neurocomputing*, 2023.
**URL**: https://www.sciencedirect.com/science/article/pii/S0925231223003648

Idea of multiple specialized agents (offense, defense, calling) composed into one player is validated by this paper. Multiple paradigms (rule-based, game-theoretic, RL) combined outperform any single paradigm. But this is engineering improvement, not conceptual breakthrough.

### 8. COMPRESSION / MDL

**Source**: "Bridging Kolmogorov Complexity and Deep Learning"
**URL**: https://arxiv.org/abs/2509.22445

MDL could help identify *simplest* strategy explaining expert play, useful for interpretability and knowledge distillation. But this is analysis tool, not training paradigm.

### 9. MAHJONG BRAIN IMAGING

**Source**: "Comparison of Cortical Activation during Mahjong Game Play" (fNIRS study)
**URL**: https://www.walshmedicalmedia.com/open-access/comparison-of-cortical-activation-during-mahjong-game-play-in-a-video-game-setting-and-a-reallife-setting-2161-1009-1000164.pdf

Found: Real-life Mahjong activates Broca's area, somatosensory cortex, angular gyrus, Wernicke's area more than digital Mahjong. Suggests Mahjong involves major *linguistic/symbolic processing* (angular gyrus involved in number/symbol manipulation). But study compared settings, not expertise levels -- no expert-vs-novice comparison exists in Mahjong neuroscience literature.

---

## The Unified Framework: Active Inference + Absent Evidence + ToM

Tier 1 insights not independent -- facets of one deeper framework:

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

Yoshida/Dolan/Friston 2008 paper was literally co-authored by Friston. Same program. Active inference IS theory of mind IS epistemic foraging. Not separate insights -- one coherent mathematical framework for decision-making under uncertainty with other agents.

---

## Concrete Recommendation for Hydra

### Phase 1: Explicit Belief States (low-hanging fruit)
Add parallel belief module maintaining probability distributions over opponent hand compositions, updated each turn via Bayesian inference from observed discards + non-calls. This already exists implicitly in neural-net hidden state, but making it explicit enables:
- Entropy-based phase detection (early = high entropy, late = low entropy)
- Targeted uncertainty reduction

### Phase 2: Epistemic Value Head (the paradigm shift)
Add 6th output head: **Epistemic Value**. For each of 34 possible discards, predict expected information gain (reduction in opponent hand entropy). Train with self-play objective where reward includes KL divergence between pre-discard and post-discard belief states.

### Phase 3: Absent Evidence Module
Track per-opponent "absence surprisal" -- for each opponent, accumulate information content of non-calls as game progresses. Feed as additional feature channel.

### Phase 4: Recursive ToM (aspirational)
Model opponents as having their own policies depending on what they think you're doing. Even level-1 ToM (modeling opponents as modeling you) would be novel in Mahjong AI.

---

## Key References

1. Friston et al. (2015). "Active inference and epistemic value." Cognitive Neuroscience, 6(4).
2. Parr, Da Costa & Friston (2019). value of uncertainty." Behavioral Brain Research.
3. Smith, Friston & Whyte (2022). "Active inference as POMDP." J. Math Psychology.
4. Yoshida, Dolan & Friston (2008). "Game Theory of Mind." PLoS Comp Bio.
5. Hsu, Griffiths & Schreiber (2017). "When Absence of Evidence Is Evidence of Absence." Cognitive Science.
6. van Opheusden et al. (2023). "Expertise increases planning depth." Nature, 620.
7. Pearl's Causal Hierarchy -- arXiv:2405.07373
8. Information gathering in POMDPs using active inference -- Springer JAAMAS 2024