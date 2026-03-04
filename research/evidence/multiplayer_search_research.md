# Search in Multiplayer Imperfect-Information Games
## Research Report for Hydra (4-Player Riichi Mahjong)

---

## The Core Problem: What Breaks in Multiplayer?

Every algorithm reviewed (ReBeL, Student of Games, GT-CFR) shares the same fundamental
wall when moving from 2-player to N-player. Three things break simultaneously:

### 1. No Unique Nash Equilibrium Value

In 2-player zero-sum (2p0s) games, every state has exactly ONE Nash equilibrium value
(the minimax value). This means you can train a value network V(s) that outputs a
single scalar -- the "true value" of the position.

In multiplayer games, there are **infinitely many Nash equilibria**, each assigning
DIFFERENT values to the same state. If Player A picks their strategy from equilibrium
E1 and Player B picks from E2, the combined play is NOT an equilibrium -- players
have incentive to deviate.

**Source**: ReBeL paper (Brown et al. 2020), Section 4, page 5:
> "In 2p0s games... every PBS beta has a unique value V_i(beta) for each agent i,
> where V_1(beta) = -V_2(beta)."

Brown thesis, Section 6.6 (p.189):
> "If each player in such a game independently computes and plays a Nash equilibrium,
> the resulting joint strategy that they play may not be a Nash equilibrium and players
> might have an incentive to deviate to a different strategy."

**Why this matters for Hydra**: You can't train a value head V(s) -> R^4 that outputs
"the Nash value" for each of 4 players. No such unique value exists.

### 2. Loss of Convexity (CFR Breaks)

ReBeL converts imperfect-info games into continuous optimization over Public Belief
States (PBS). In 2p0s, this is a **convex optimization problem**, so iterative
algorithms like CFR provably converge at rate O(1/sqrt(T)).

In multiplayer, the optimization landscape is **non-convex**. CFR's convergence
guarantee vanishes. You might oscillate, diverge, or converge to a non-equilibrium
fixed point.

**Source**: ReBeL paper, Section 4, page 5:
> "Fortunately, in 2p0s games, these high-dimensional belief representations are
> convex optimization problems. ReBeL leverages this fact by conducting search via
> an iterative gradient-ascent-like algorithm."

### 3. PPAD-Completeness

Computing a Nash equilibrium in multiplayer or general-sum games is **PPAD-complete**
-- widely conjectured to be computationally intractable. Even approximate Nash is hard.

**Source**: Opponent Modeling paper (ACM 2025):
> "While computing a Nash equilibrium can be done in polynomial time in two-player
> zero-sum games, it is PPAD-hard for non-zero-sum and multiplayer games."

Brown thesis, Section 6.6 (p.189): confirms this is why Pluribus abandons
theoretical guarantees for multiplayer.

---

## Algorithm-by-Algorithm Analysis

### 1. ReBeL (Brown et al., NeurIPS 2020)
**Paper**: https://arxiv.org/abs/2007.13544

**What it does**: Converts imperfect-info games into a "PBS game" -- a
perfect-information game over Public Belief States. Then applies AlphaZero-style
self-play RL + search on the PBS game.

**Key innovation**: The Public Belief State (PBS):
- beta = (Delta_S1(s_pub), ..., Delta_SN(s_pub))
- A probability distribution over each player's possible infostates, given public info
- Think of it as: "everyone knows the probability distribution of everyone's hands"
- The value network outputs a VECTOR (one value per possible infostate), not a scalar
- These infostate values are actually supergradients of the PBS value function (Thm 1)

**Why 2-player only**:
1. Unique value requires 2p0s (V_1 = -V_2)
2. Convexity of PBS optimization requires 2p0s
3. CFR convergence theorem requires convexity
4. Value network training requires a single "correct" target value

**Convergence**: O(1/sqrt(T)) to Nash (Theorem 2), but ONLY in 2p0s.

**Authors' own words** (Section 9):
> "ReBeL's theoretical guarantees are limited only to two-player zero-sum games."

**Verdict for Hydra**: Cannot be applied directly. The PBS representation could
theoretically be constructed for 4 players, but you lose all convergence guarantees,
and the value network has no well-defined training target.

---

### 2. Student of Games / Player of Games (Schmid et al., Science Advances 2023)
**Paper**: https://arxiv.org/abs/2112.03178

Note: "Player of Games" was the earlier arxiv name; "Student of Games" is the
published version. Same paper, same algorithm.

**What it does**: Combines GT-CFR search with a Counterfactual Value-Policy
Network (CVPN) and safe re-solving. Aims to be a single algorithm for both
perfect-info (Chess, Go) and imperfect-info (poker, Scotland Yard).

**GT-CFR (Growing-Tree CFR)**:
1. Maintains a partial subgame tree (not the full game tree)
2. Alternates between two phases:
   - **Regret Update**: Run CFR iterations on current tree. At leaves, query the CVPN
     for counterfactual values.
   - **Expansion**: Sample a trajectory using a 50/50 mix of PUCT and CFR policies.
     When hitting a state not in tree, add it. Top-k actions expanded per node.
3. For imperfect-info: k=infinity (must allow mixing over all actions)
4. For perfect-info: k=1 (just find the best action)
5. Complexity: O(kT^2) CVPN calls for T iterations

**The CVPN**:
- Input: PBS beta = (s_pub, r) where r is the "range" (belief distribution)
- Output: (v, p) = counterfactual values + prior policies for each infostate
- This is like a combined value+policy network but outputting VECTORS, not scalars

**Safe Re-solving**:
- Uses a "gadget" (binary opponent decision node) to ensure local search is consistent
  with global equilibrium
- Can re-solve from the nearest previously-solved state if exact match unavailable

**Multiplayer support**: NO.
> "The theoretical guarantee of Nash equilibria outside of this setting is less
> meaningful and it is unclear how effective they would be (for example, in games
> with more than two players)." -- Section "Background", page 4

**Games tested**: Chess, Go, HUNL Poker, Scotland Yard -- all 2-player.

**Verdict for Hydra**: Same wall as ReBeL. GT-CFR is interesting as a search mechanism
(anytime, non-uniform tree growth), but the CFR convergence + safe re-solving both
require 2p0s. The CVPN architecture concept might be adaptable.

---

### 3. Pluribus (Brown & Sandholm, Science 2019)
**Paper**: https://www.science.org/doi/10.1126/science.aay2400
**Thesis**: https://noambrown.github.io/thesis.pdf

**THE multiplayer success story.** First superhuman AI in 6-player poker.

**How it works -- the key innovation is "just don't prove anything"**:

**Phase 1: Blueprint Strategy (Offline)**
- Compute an approximate blueprint strategy for the full 6-player game via Linear MCCFR
- Self-play: 6 copies of itself playing 12,400 CPU-core-hours (~$144, 8 days)
- Uses action abstraction (bucketed bet sizes) and card abstraction
- Stores in <512GB (vs Libratus's 18TB for 2-player!)

**Phase 2: Real-Time Depth-Limited Search (Online)**
- At decision time, construct a subgame from beginning of current betting round
- Solve this subgame in real time using Linear MCCFR
- CRITICAL DIFFERENCE from 2-player: at leaf nodes (depth limit), ALL players
  (not just opponents) choose among k continuation strategies
- This is called "Multi-Valued States" (MVS)

**Why MVS matters**:
In 2-player depth-limited solving, only the OPPONENT gets choice at leaves
(ensures robustness against opponent adaptation). In multiplayer, the searcher
itself must also choose, because:
- The searcher might want to change strategy beyond the search horizon
- A fixed searcher strategy at leaves makes it too "predictable/conservative"

Brown thesis (p.190):
> "Pluribus addressed this weakness by having the searcher also choose among
> continuation strategies... this approach is more effective, easier, and more elegant."

**Belief tracking across 6 players**:
- Uses PBS representation: board cards + pot size + binary acting-player flag +
  probability distribution over all 1,326 possible hole-card combos for EACH player
- All players observe the same public actions -> Bayesian update on beliefs
- Key simplification: poker has SMALL private info (2 cards) vs Mahjong (13+ tiles)

**Computational cost comparison**:
| Metric            | Libratus (2p)          | Pluribus (6p)        |
|-------------------|------------------------|----------------------|
| Training          | Millions core-hours    | 12,400 core-hours    |
| Memory            | 18 TB                  | <512 GB              |
| Real-time HW      | Supercomputer nodes    | 28-core CPU, 128GB   |
| Search depth       | Full game              | Depth-limited        |

Brown thesis (p.190):
> "Depth-limited search reduces the computational resources and memory needed
> by probably at least five orders of magnitude."

**NO theoretical guarantees**: Pluribus explicitly does NOT claim to play Nash in
6-player. It just works empirically. The blueprint is trained via MCCFR which has
no convergence guarantee in multiplayer.

**Verdict for Hydra**: The most relevant model. BUT Mahjong's private info space
is VASTLY larger than poker's. Poker: 1,326 hole card combos. Mahjong: info set
size ~10^48. You cannot enumerate belief distributions over all possible hands.
The MVS approach at leaves is interesting but would need a learned value function
instead of blueprint rollouts.

---

### 4. EPIMC -- Extended Perfect Information Monte Carlo (Amouret et al., 2024)
**Paper**: https://arxiv.org/abs/2408.02380

**The strategy fusion problem (what's wrong with PIMC)**:
When you determinize (sample a possible world and solve as if it were perfect-info),
each sampled world gets a DIFFERENT optimal strategy. But the real player can't
distinguish between worlds in the same information set -- they must use the SAME
strategy for all of them. PIMC ignores this constraint.

Example: In Rock-Paper-Scissors where you "sample" the opponent's move, PIMC would
play Paper when it samples Rock, Scissors when it samples Paper, etc. -- getting
value 1.0 (always win). But the real value is 0.0 because you can't condition on
what you don't know.

**EPIMC's fix -- "postpone the perfect info assumption"**:
1. Sample world states from your information set
2. Simulate forward for d steps WITHOUT using perfect info
3. Only at depth d, switch to perfect-info solving for leaf evaluation
4. Solve the d-step subgame using an infostate-respecting solver (CFR, info-set search)

This creates a "buffer zone" of d steps where strategy fusion is prevented.

**Theoretical guarantees**:
- Increasing depth d never makes strategy fusion worse (Proposition 1)
- There always exists a depth d that strictly reduces fusion (Proposition 2)
- For finite games, setting d = game length eliminates fusion entirely (Proposition 3)

**Multiplayer**: Framework is general for N players, but the SUBGAME SOLVERS used
(CFR, info-set search) only have guarantees for 2 players. Experiments are all 2-player.

**Verdict for Hydra**: Interesting concept for reducing strategy fusion in
determinization-based approaches. Could potentially combine with PIMC/ISMCTS in
Mahjong. The "postpone reasoning" idea is orthogonal to the multiplayer problem.

---

### 5. MCTS for Imperfect-Info Multiplayer

**ISMCTS (Information Set MCTS)** -- Cowling et al., 2012:
- The standard approach: at each MCTS iteration, sample a determinization consistent
  with the acting player's information set, then run normal MCTS on it
- Aggregates statistics across many determinizations
- Works for N players out of the box (no 2-player assumption)
- Problem: slow convergence, strategy fusion still present (each determinization is
  solved independently)

**Hybrid Multi-Agent AI/MCTS** (2024/2025 -- Authorea):
- Applied to "28" (4-player trick-taking card game)
- Uses belief networks + MCTS + heuristic search
- Dynamically switches between approaches based on game phase
- No formal guarantees, pure engineering

**Key insight**: ISMCTS is the ONLY search algorithm that naturally handles
multiplayer imperfect info without requiring 2p0s assumptions. Everything else
(ReBeL, SoG, GT-CFR, EPIMC) is fundamentally 2-player.

---

### 6. Mahjong AI -- State of the Art (IJCAI 2024 Competition)
**Paper**: https://www.ijcai.org/proceedings/2024/1020.pdf

**Critical context for Hydra**: The Mahjong AI competition reveals that NOBODY uses
search at test time. The state of the art is:

1. **Supervised Learning (dominant)**: CNN/ResNet trained on human game logs to predict
   actions. Most top-16 agents in 2023 used this approach.
2. **Reinforcement Learning**: PPO or IMPALA with self-play. Won the first competition.
3. **Heuristic methods**: Shanten-based search trees + hand-coded rules.

**Why no search?** The paper explains:
- Information set size ~10^48 (vs poker's much smaller space)
- "Much larger than games like Poker and Bridge, making standard algorithms like
  CFR difficult to apply"
- High variance from tile draws destabilizes learning
- 81 different scoring patterns make evaluation complex

**What the winners do**: SL (behavior cloning from human data) + high-level
features (shanten value). The best approach combines learned policies with
hand-crafted features but NO online search.

---

## Recent Work (2024-2025)

### "Last-Iterate Convergence to Approximate Nash in Multiplayer IIGs" (IEEE 2024)
- Proposes IESL (Imperfect-info Exponential-decay Score-based Learning)
- Proves last-iterate convergence to approximate Nash in multiplayer IIGs
- Uses Nash Distribution (a type of Quantal Response Equilibrium)
- This is a TRAINING algorithm, not a search algorithm
- Significance: first last-iterate convergence result for multiplayer IIGs

### "Quadratic Programming Approach for Nash in Multiplayer" (Games journal 2026)
- Exact Nash computation in multiplayer normal-form games
- Not scalable to extensive-form games like Mahjong

### "Look-ahead search on top of policy networks in IIGs" (IJCAI 2024)
- Test-time search added to policy-gradient algorithms
- Still focused on two-player adversarial games
- Uses sampled public state approach

### "Belief Stochastic Game model" (arxiv 2507.19263, 2025)
- Delegates state estimation to the game model itself
- Players operate on externally provided belief states
- Interesting for Mahjong: could reduce need for game-specific inference

---

## Synthesis: What This Means for Hydra

### The Landscape

| Algorithm       | Multiplayer? | Online Search? | Theoretical Guarantee? | Tested On            |
|-----------------|-------------|----------------|----------------------|----------------------|
| ReBeL           | No (2p0s)   | Yes            | Yes (Nash conv.)     | Poker, Liar's Dice   |
| Student of Games| No (2p0s)   | Yes (GT-CFR)   | Yes (exploitability) | Chess, Go, Poker, SY |
| Pluribus        | Yes (6p)    | Yes (DLS)      | **None**             | 6-player Poker       |
| ISMCTS          | Yes (Np)    | Yes            | None formal          | Various card games    |
| EPIMC           | Framework yes, solvers no | Yes | 2p only        | Dark Chess, etc.     |
| Mahjong SOTA    | Yes (4p)    | **No**         | None                 | 4-player Mahjong     |

### The Hard Truth

**No algorithm exists that provides both:**
1. Theoretical guarantees (convergence to equilibrium)
2. Multiplayer support (3+ players)

...for imperfect-information games. This is a fundamental open problem in game theory/AI.

### Practical Options for Hydra (4-player Riichi Mahjong)

**Option A: No Search (Current Mahjong SOTA)**
- Train policy network via SL on game logs + RL via PPO/IMPALA
- No online search at all
- Proven to work (IJCAI competition winners)
- Con: leaves performance on the table. A search-enhanced agent should beat pure policy.

**Option B: Pluribus-Style Depth-Limited Search (adapted)**
- Blueprint policy via self-play RL
- Real-time depth-limited subgame solving
- Challenge: Mahjong info sets are 10^48 -- can't enumerate belief distributions
- Possible adaptation: use LEARNED belief representations instead of explicit distributions
  (encode beliefs as neural network embeddings rather than probability vectors over all hands)

**Option C: ISMCTS + Policy Network (hybrid)**
- Use ISMCTS for search, guided by a learned policy/value network
- Determinize by sampling opponent hands consistent with observations
- Use policy network as prior (like PUCT in AlphaZero) and value network for rollouts
- Naturally handles 4 players
- Strategy fusion is the known weakness -- could use EPIMC's "postpone reasoning"
  idea to partially mitigate

**Option D: "Just RL + Search" (no guarantees, like Pluribus)**
- Accept that no theoretical guarantees exist for 4-player
- Train a strong policy+value network via self-play RL
- At test time, do some form of search to improve on the policy:
  - Sample N possible worlds (opponent hands) from a belief model
  - For each world, evaluate candidate actions using value network
  - Choose action with best expected value across worlds
- This is basically PIMC + neural evaluation + belief modeling
- Simple, fast, scales well

### Recommendation

**Option D is the most practical path for Hydra.** Here's why:

1. The Mahjong SOTA (no search) already works -- adding even simple search should help
2. Pluribus proved that "no guarantees, but it works" is fine for multiplayer
3. The belief modeling problem (what tiles do opponents hold?) is separate from
   the search problem and can be addressed with a dedicated neural network
4. Strategy fusion from PIMC is mitigated in practice because:
   - Mahjong has high branching factor but relatively low information asymmetry
     per decision (you see ~70% of discards)
   - A good belief model narrows the sampling space dramatically
   - Short search horizons (1-3 moves) reduce fusion impact

The theoretical open problems (convergence in multiplayer, convexity, unique Nash)
are unsolved in the literature and likely won't be solved soon. Every practical
system that works in multiplayer (Pluribus, Mahjong winners) uses heuristic/learned
approaches without formal guarantees.

---

## Key Papers Referenced

1. **ReBeL**: Brown et al. "Combining Deep RL and Search for Imperfect-Information Games." NeurIPS 2020. https://arxiv.org/abs/2007.13544
2. **Student of Games**: Schmid et al. "Student of Games: A unified learning algorithm for both perfect and imperfect information games." Science Advances, 2023. https://arxiv.org/abs/2112.03178
3. **Pluribus**: Brown & Sandholm. "Superhuman AI for multiplayer poker." Science, 2019. https://www.science.org/doi/10.1126/science.aay2400
4. **Brown Thesis**: Brown. "Equilibrium Finding for Large Adversarial Imperfect-Information Games." CMU PhD Thesis. https://noambrown.github.io/thesis.pdf
5. **EPIMC**: Amouret et al. "Perfect Information Monte Carlo with Postponing Reasoning." 2024. https://arxiv.org/abs/2408.02380
6. **Mahjong AI Competition**: IJCAI 2024. https://www.ijcai.org/proceedings/2024/1020.pdf
7. **IESL**: Lu & Zhu. "Last-Iterate Convergence to Approximate Nash Equilibria in Multiplayer IIGs." IEEE 2024.
8. **Depth-Limited Solving**: Brown et al. NeurIPS 2018. https://noambrown.github.io/papers/18-NIPS-Depth.pdf
