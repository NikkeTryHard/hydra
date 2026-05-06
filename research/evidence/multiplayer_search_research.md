# Search in Multiplayer Imperfect-Information Games
## Research Report for Hydra (4-Player Riichi Mahjong)

---

## The Core Problem: What Breaks in Multiplayer?

All reviewed algos (ReBeL, Student of Games, GT-CFR) hit same wall moving 2-player -> N-player. Three breaks:

### 1. No Unique Nash Equilibrium Value

In 2-player zero-sum (2p0s), each state has ONE Nash value
(minimax). Thus value net `V(s)` can output single scalar:
"true value" of position.

In multiplayer, **infinitely many Nash equilibria** exist, giving
DIFFERENT values at same state. If Player uses equilibrium
E1 and Player B uses E2, combined play is NOT equilibrium --
players may deviate.

**Source**: ReBeL paper (Brown et al. 2020), Section 4, page 5:
> "In 2p0s games... every PBS beta has unique value V_i(beta) for each agent i,
> where V_1(beta) = -V_2(beta)."

Brown thesis, Section 6.6 (p.189):
> "If each player in such game independently computes and plays Nash equilibrium,
> resulting joint strategy that they play may not be Nash equilibrium and players
> might have incentive to deviate to different strategy."

**Why this matters for Hydra**: Cannot train value head `V(s) -> R^4` to output
Nash value" for 4 players. No unique target exists.

### 2. Loss of Convexity (CFR Breaks)

ReBeL maps imperfect-info games into continuous optimization over Public Belief
States (PBS). In 2p0s, this is **convex**, so iterative algos like CFR provably
converge at rate O(1/sqrt(T)).

In multiplayer, landscape is **non-convex**. CFR convergence guarantee dies.
May oscillate, diverge, or reach non-equilibrium fixed point.

**Source**: ReBeL paper, Section 4, page 5:
> "Fortunately, in 2p0s games, these high-dimensional belief representations are
> convex optimization problems. ReBeL leverages this fact by conducting search via
> iterative gradient-ascent-like algorithm."

### 3. PPAD-Completeness

Computing Nash in multiplayer or general-sum games is **PPAD-complete**
-- widely believed intractable. Even approximate Nash is hard.

**Source**: Opponent Modeling paper (ACM 2025):
> "While computing Nash equilibrium can be done in polynomial time in two-player
> zero-sum games, it is PPAD-hard for non-zero-sum and multiplayer games."

Brown thesis, Section 6.6 (p.189): confirms this is why Pluribus drops
theoretical guarantees in multiplayer.

---

## Algorithm-by-Algorithm Analysis

### 1. ReBeL (Brown et al., NeurIPS 2020)
**Paper**: https://arxiv.org/abs/2007.13544

**What it does**: Converts imperfect-info games into "PBS game" --
perfect-info game over Public Belief States. Then applies AlphaZero-style
self-play RL + search on PBS game.

**Key innovation**: Public Belief State (PBS):
- beta = (Delta_S1(s_pub),..., Delta_SN(s_pub))
- Probability distribution over each player's possible infostates, given public info
- Think: "all know probability distribution of all hands"
- Value net outputs VECTOR (one value per infostate), not scalar
- These infostate values are supergradients of PBS value fn (Thm 1)

**Why 2-player only**:
1. Unique value needs 2p0s (`V_1 = -V_2`)
2. PBS convexity needs 2p0s
3. CFR convergence theorem needs convexity
4. Value-net training needs single "correct" target

**Convergence**: O(1/sqrt(T)) to Nash (Theorem 2), but ONLY in 2p0s.

**Authors' own words** (Section 9):
> "ReBeL's theoretical guarantees are limited only to two-player zero-sum games."

**Verdict for Hydra**: Cannot apply directly. PBS representation could
exist for 4 players, but all convergence guarantees vanish,
and value net loses well-defined target.

---

### 2. Student of Games / Player of Games (Schmid et al., Science Advances 2023)
**Paper**: https://arxiv.org/abs/2112.03178

Note: "Player of Games" = earlier arxiv name; "Student of Games" = published name.
Same paper, same algo.

**What it does**: Combines GT-CFR search with Counterfactual Value-Policy
Network (CVPN) and safe re-solving. Tries to unify perfect-info
(Chess, Go) and imperfect-info (poker, Scotland Yard).

**GT-CFR (Growing-Tree CFR)**:
1. Keeps partial subgame tree (not full tree)
2. Alternates two phases:
   - **Regret Update**: Run CFR on current tree. At leaves, query CVPN
for counterfactual values.
   - **Expansion**: Sample trajectory using 50/50 mix of PUCT and CFR policies.
On new state, add it. Top-k actions expanded per node.
3. For imperfect-info: k=infinity (must allow mixing over all actions)
4. For perfect-info: k=1 (find best action)
5. Complexity: O(kT^2) CVPN calls for T iterations

CVPN**:
- Input: PBS beta = (s_pub, r) where r = "range" (belief distribution)
- Output: (v, p) = counterfactual values + prior policies for each infostate
- Like combined value+policy net, but outputs VECTORS, not scalars

**Safe Re-solving**:
- Uses "gadget" (binary opponent decision node) to keep local search
consistent with global equilibrium
- Can re-solve from nearest previously-solved state if exact match missing

**Multiplayer support**: NO.
> theoretical guarantee of Nash equilibria outside of this setting is less
> meaningful and it is unclear how effective they would be (e.g., in games
> with more than two players)." -- Section "Background", page 4

**Games tested**: Chess, Go, HUNL Poker, Scotland Yard -- all 2-player.

**Verdict for Hydra**: Same wall as ReBeL. GT-CFR interesting as search mech
(anytime, non-uniform tree growth), but CFR convergence + safe re-solving
need 2p0s. CVPN architecture idea may adapt.

---

### 3. Pluribus (Brown & Sandholm, Science 2019)
**Paper**: https://www.science.org/doi/10.1126/science.aay2400
**Thesis**: https://noambrown.github.io/thesis.pdf

multiplayer success story.** First superhuman AI in 6-player poker.

**How it works -- key innovation = "do not prove anything"**:

**Phase 1: Blueprint Strategy (Offline)**
- Compute approximate blueprint strategy for full 6-player game via Linear MCCFR
- Self-play: 6 copies of itself playing 12,400 CPU-core-hours (~$144, 8 days)
- Uses action abstraction (bucketed bet sizes) and card abstraction
- Stores in <512GB (vs Libratus's 18TB for 2-player!)

**Phase 2: Real-Time Depth-Limited Search (Online)**
- At decision time, construct subgame from start of current betting round
- Solve subgame in real time using Linear MCCFR
- CRITICAL DIFFERENCE from 2-player: at leaf nodes, ALL players
(not only opponents) choose among k continuation strategies
- Called "Multi-Valued States" (MVS)

**Why MVS matters**:
In 2-player depth-limited solving, only OPPONENT gets choice at leaves
(ensures robustness vs opponent adaptation). In multiplayer, searcher
must also choose, because:
- Searcher may want strategy change beyond horizon
- Fixed searcher leaf strategy makes play too "predictable/conservative"

Brown thesis (p.190):
> "Pluribus addressed this weakness by having searcher also choose among
> continuation strategies... this approach is more effective, easier, and more elegant."

**Belief tracking across 6 players**:
- Uses PBS representation: board cards + pot size + binary acting-player flag +
probability distribution over all 1,326 possible hole-card combos for EACH player
- All players observe same public actions -> Bayesian belief update
- Key simplification: poker has SMALL private info (2 cards) vs Mahjong (13+ tiles)

**Computational cost comparison**:
| Metric            | Libratus (2p)          | Pluribus (6p)        |
|-------------------|------------------------|----------------------|
| Training          | Millions core-hours    | 12,400 core-hours    |
| Memory            | 18 TB                  | <512 GB              |
| Real-time HW      | Supercomputer nodes    | 28-core CPU, 128GB   |
| Search depth       | Full game              | Depth-limited        |

Brown thesis (p.190):
> "Depth-limited search reduces computational resources and memory needed
> by probably at least five orders of magnitude."

**NO theoretical guarantees**: Pluribus does NOT claim Nash play in
6-player. Blueprint trained via MCCFR, which has no multiplayer
convergence guarantee.

**Verdict for Hydra**: Most relevant model. BUT Mahjong private info space
is VASTLY larger than poker's. Poker: 1,326 hole-card combos. Mahjong: info set
size ~10^48. Cannot enumerate belief distributions over all hands.
MVS leaf idea interesting, but likely needs learned value fn instead
of blueprint rollouts.

---

### 4. EPIMC -- Extended Perfect Information Monte Carlo (Amouret et al., 2024)
**Paper**: https://arxiv.org/abs/2408.02380

**Strategy fusion problem (why PIMC fails)**:
When determinizing (sampling possible world and solving as perfect-info),
each sampled world gets DIFFERENT optimal strategy. Real player cannot
distinguish worlds inside same information set -- must use SAME strategy
for all. PIMC ignores this.

Example: In Rock-Paper-Scissors where you "sample" opponent move, PIMC would
play Paper when it samples Rock, Scissors when it samples Paper, etc. --
getting value 1.0 win). Real value is 0.0 because you cannot condition
on hidden info.

**EPIMC fix -- "postpone perfect info assumption"**:
1. Sample world states from your information set
2. Simulate forward for d steps WITHOUT perfect info
3. Only at depth d, switch to perfect-info solving for leaf eval
4. Solve d-step subgame using infostate-respecting solver (CFR, info-set search)

This creates "buffer zone" of d steps where strategy fusion is blocked.

**Theoretical guarantees**:
- Increasing depth d never worsens strategy fusion (Proposition 1)
- There exists depth d that strictly reduces fusion (Proposition 2)
- For finite games, d = game length removes fusion entirely (Proposition 3)

**Multiplayer**: Framework general for N players, but SUBGAME SOLVERS used
(CFR, info-set search) only have 2-player guarantees. Experiments all 2-player.

**Verdict for Hydra**: Interesting concept for reducing strategy fusion in
determinization-based methods. Could combine with PIMC/ISMCTS in
Mahjong. "Postpone reasoning" idea is orthogonal to multiplayer issue.

---

### 5. MCTS for Imperfect-Info Multiplayer

**ISMCTS (Information Set MCTS)** -- Cowling et al., 2012:
- Standard approach: each MCTS iteration samples determinization consistent
with acting player's information set, then runs normal MCTS on it
- Aggregates stats across many determinizations
- Works for N players out of box (no 2-player assumption)
- Problem: slow convergence, strategy fusion still present (each determinization
solved independently)

**Hybrid Multi-Agent AI/MCTS** (2024/2025 -- Authorea):
- Applied to "28" (4-player trick-taking card game)
- Uses belief nets + MCTS + heuristic search
- Dynamically switches approach by game phase
- No formal guarantees, pure engineering

**Key insight**: ISMCTS is ONLY search algo that naturally handles
multiplayer imperfect info without 2p0s assumptions. Everything else
(ReBeL, SoG, GT-CFR, EPIMC) is fundamentally 2-player.

---

### 6. Mahjong AI -- State of the Art (IJCAI 2024 Competition)
**Paper**: https://www.ijcai.org/proceedings/2024/1020.pdf

**Critical context for Hydra**: Mahjong AI competition shows NOBODY uses
search at test time. State of art is:

1. **Supervised Learning (dominant)**: CNN/ResNet trained on human game logs to predict
actions. Most top-16 agents in 2023 used this approach.
2. **Reinforcement Learning**: PPO or IMPALA with self-play. Won first competition.
3. **Heuristic methods**: Shanten-based search trees + hand-coded rules.

**Why no search?** Paper explains:
- Information set size ~10^48 (vs poker much smaller)
- "Much larger than games like Poker and Bridge, making standard algorithms like
CFR difficult to apply"
- High variance from tile draws destabilizes learning
- 81 different scoring patterns make eval complex

**What winners do**: SL (behavior cloning from human data) + high-level
features (shanten value). Best approach combines learned policies with
hand-crafted features but NO online search.

---

## Recent Work (2024-2025)

### "Last-Iterate Convergence to Approximate Nash in Multiplayer IIGs" (IEEE 2024)
- Proposes IESL (Imperfect-info Exponential-decay Score-based Learning)
- Proves last-iterate convergence to approximate Nash in multiplayer IIGs
- Uses Nash Distribution (kind of Quantal Response Equilibrium)
- This is TRAINING algo, not search algo
- Significance: first last-iterate convergence result for multiplayer IIGs

### "Quadratic Programming Approach for Nash in Multiplayer" (Games journal 2026)
- Exact Nash computation in multiplayer normal-form games
- Not scalable to extensive-form games like Mahjong

### "Look-ahead search on top of policy networks in IIGs" (IJCAI 2024)
- Test-time search added to policy-gradient algos
- Still focused on 2-player adversarial games
- Uses sampled public state approach

### "Belief Stochastic Game model" (arxiv 2507.19263, 2025)
- Delegates state estimation to game model itself
- Players operate on externally provided belief states
- Interesting for Mahjong: may reduce game-specific inference need

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

**No algo exists that gives both:**
1. Theoretical guarantees (equilibrium convergence)
2. Multiplayer support (3+ players)

...for imperfect-information games. This is fundamental open problem in game theory/AI.

### Practical Options for Hydra (4-player Riichi Mahjong)

**Option No Search (Current Mahjong SOTA)**
- Train policy net via SL on game logs + RL via PPO/IMPALA
- No online search
- Proven to work (IJCAI competition winners)
- Con: leaves performance on table. Search-enhanced agent should beat pure policy.

**Option B: Pluribus-Style Depth-Limited Search (adapted)**
- Blueprint policy via self-play RL
- Real-time depth-limited subgame solving
- Challenge: Mahjong info sets are 10^48 -- cannot enumerate belief distributions
- Possible adaptation: use LEARNED belief representations instead of explicit distributions
(encode beliefs as neural embeddings, not probability vectors over all hands)

**Option C: ISMCTS + Policy Network (hybrid)**
- Use ISMCTS for search, guided by learned policy/value net
- Determinize by sampling opponent hands consistent with observations
- Use policy net as prior (like PUCT in AlphaZero) and value net for rollouts
- Naturally handles 4 players
- Strategy fusion = known weakness -- could use EPIMC "postpone reasoning"
idea to partially mitigate

**Option D: RL + Search" (no guarantees, like Pluribus)**
- Accept no theoretical guarantees for 4-player
- Train strong policy+value net via self-play RL
- At test time, do some search to improve policy:
  - Sample N possible worlds (opponent hands) from belief model
  - For each world, evaluate candidate actions using value net
  - Choose action with best expected value across worlds
- This is PIMC + neural eval + belief modeling
- Simple, fast, scales well

### Recommendation

**Option D is most practical path for Hydra.** Why:

1. Mahjong SOTA (no search) already works -- even simple search should help
2. Pluribus proved "no guarantees, but works" is acceptable in multiplayer
3. Belief modeling problem (what tiles opponents hold?) is separate from
search problem and can be handled by dedicated neural net
4. Strategy fusion from PIMC is mitigated in practice because:
   - Mahjong has high branching factor but relatively low info asymmetry
per decision (you see ~70% of discards)
   - Good belief model narrows sampling space sharply
   - Short search horizons (1-3 moves) reduce fusion impact

Theoretical open problems (multiplayer convergence, convexity, unique Nash)
remain unsolved in literature, likely not soon solved. Every practical
multiplayer system that works (Pluribus, Mahjong winners) uses heuristic/learned
methods without formal guarantees.

---

## Key Papers Referenced

1. **ReBeL**: Brown et al. "Combining Deep RL and Search for Imperfect-Information Games." NeurIPS 2020. https://arxiv.org/abs/2007.13544
2. **Student of Games**: Schmid et al. "Student of Games: unified learning algorithm for both perfect and imperfect information games." Science Advances, 2023. https://arxiv.org/abs/2112.03178
3. **Pluribus**: Brown & Sandholm. "Superhuman AI for multiplayer poker." Science, 2019. https://www.science.org/doi/10.1126/science.aay2400
4. **Brown Thesis**: Brown. "Equilibrium Finding for Large Adversarial Imperfect-Information Games." CMU PhD Thesis. https://noambrown.github.io/thesis.pdf
5. **EPIMC**: Amouret et al. "Perfect Information Monte Carlo with Postponing Reasoning." 2024. https://arxiv.org/abs/2408.02380
6. **Mahjong AI Competition**: IJCAI 2024. https://www.ijcai.org/proceedings/2024/1020.pdf
7. **IESL**: Lu & Zhu. "Last-Iterate Convergence to Approximate Nash Equilibria in Multiplayer IIGs." IEEE 2024.
8. **Depth-Limited Solving**: Brown et al. NeurIPS 2018. https://noambrown.github.io/papers/18-NIPS-Depth.pdf