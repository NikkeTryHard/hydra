# Cutting-Edge Game AI Algorithms (2024-2025)
## For 4-Player Mahjong AI -- Genuinely Novel Algorithmic Contributions

Research compiled March 2026. Focus: algorithms (not recipes) for imperfect-information multiplayer games.

---

## TIER 1: HIGHEST RELEVANCE TO 4P MAHJONG (Directly Applicable Algorithms)

---

### 1. DDCFR -- Dynamic Discounted Counterfactual Regret Minimization
- **Paper**: Xu, Li, Fu, Fu, Xing, Cheng. **ICLR 2024 (Spotlight)**
- **Source**: https://proceedings.iclr.cc/paper_files/paper/2024/hash/a89331e2ecbbd9d9618bc94717a91155-Abstract-Conference.html
- **Code**: https://github.com/rpSebastian/DDCFR
- **Team**: Tencent AI (Haobo Fu's group -- same team behind LuckyJ mahjong)

**What it does**: Previous CFR variants (DCFR, Linear CFR, CFR+) all use FIXED discounting
schemes -- e.g., weight iteration t by t^alpha. DDCFR replaces this with a LEARNED dynamic
discounting function. A meta-learner (neural network) observes the current game-solving state
and outputs the discount weights for each iteration, adapting in real-time.

**Key innovation (ALGORITHM, not recipe)**:
- Formulates CFR discounting as a sequential decision problem
- Trains a policy network via RL that outputs (alpha_t, beta_t, gamma_t) at each iteration
- The discount parameters are STATE-DEPENDENT, not fixed constants
- Converges faster than any fixed scheme because it adapts to game structure

**Mathematical core**: At iteration t, instead of fixed weights:
```
R_i^t(I,a) = max(R_i^{t-1}(I,a) * w_pos(t) + r_i^t(I,a), 0)  -- positive regrets
              R_i^{t-1}(I,a) * w_neg(t) + r_i^t(I,a)           -- negative regrets
```
DDCFR replaces w_pos(t), w_neg(t), w_avg(t) with pi_theta(state_t) where state_t encodes
convergence progress, exploitability estimates, etc.

**Multiplayer applicability**: YES -- CFR family extends to multiplayer. DDCFR's meta-learning
framework is game-agnostic.

**Why this matters for Hydra**: This is literally from LuckyJ's team. If you want to go BEYOND
LuckyJ, you need to either (a) extend DDCFR itself, or (b) combine it with something orthogonal.

---

### 2. Deep DDCFR (VR-DeepDCFR+) -- Neural CFR with Advanced Discounting
- **Paper**: Xu, Li, Fu et al. **arxiv 2511.08174** (Nov 2025)
- **Source**: https://arxiv.org/abs/2511.08174
- **Team**: Same Tencent AI group

**What it does**: The sequel to DDCFR. Existing neural CFR methods (Deep CFR, DREAM) only
approximate VANILLA CFR. They can't capture the faster convergence of DCFR+/DDCFR because
the discounting + clipping operations don't translate cleanly to neural function approximation.
Deep DDCFR solves this.

**Key innovation (ALGORITHM)**:
Bootstrapped cumulative advantage estimation that lets neural nets approximate DCFR+ behavior:

1. Train a value network for variance-reduced advantage sampling
2. Maintain a cumulative-advantage network R(I,a|theta) that bootstraps from previous iteration
3. Apply discount+clip inside the learning target:

```
L(theta_t) = E[ sum_a (max(R(I,a|theta_{t-1}), 0) * (t-1)^alpha/((t-1)^alpha + 1)
              + r_bar(I,a) - R(I,a|theta_t))^2 ]
```

**Key equations**:
- Advantage: A_i^sigma(I,a) = u_i^sigma(I,a) - u_i^sigma(I)
- Bootstrap target: prev_cumulative * discount_factor + current_advantage
- Discount factor: (t-1)^alpha / ((t-1)^alpha + 1)
- Clipping: max(cumulative, 0) before discounting (DCFR+ style)
- Strategy: sigma(I,a) = max(0, R(I,a)) / sum_a' max(0, R(I,a'))

**Multiplayer**: YES -- same CFR foundation. Neural approximation is player-independent.

**Why novel**: First method to successfully combine neural function approximation with
advanced CFR variants (discount + clip). All prior neural CFR was stuck on vanilla CFR.

---

### 3. PDCFR+ -- Predictive Discounted CFR+
- **Paper**: Xu, Li, Liu, Fu, Fu, Xing, Cheng. **IJCAI 2024**
- **Source**: https://www.ijcai.org/proceedings/2024/0583.pdf
- **Team**: Same Tencent AI / CAS group

**What it does**: Combines two orthogonal acceleration ideas in CFR that had never been
merged before: (1) optimistic/predictive updates (PCFR+) and (2) regret discounting (DCFR).

**Key innovation (ALGORITHM)**:
Optimistic Online Mirror Descent applied to weighted counterfactual regret:

```
R_j^t = [R_j^{t-1} * (t-1)^alpha/((t-1)^alpha+1) + r_j^t]+    -- discounted regret
R_tilde_j^{t+1} = [R_j^t * t^alpha/(t^alpha+1) + v_j^{t+1}]+   -- predictive step
x_j^{t+1} = R_tilde_j^{t+1} / ||R_tilde_j^{t+1}||_1            -- regret matching
X^t = ((t-1)/t)^gamma * X^{t-1} + x_dot^t                       -- weighted averaging
```

**Mathematical formulation**: PWCFR+ is the general framework. PDCFR+ is the concrete
instantiation with discount-style weights. The theory requires that regret weights discount
MORE aggressively than average-strategy weights.

**Multiplayer**: YES

**Why novel**: First principled combination of prediction and discounting in CFR.
Provable convergence rate improvement over both DCFR and PCFR+ individually.

---

### 4. DRDA -- Divergence-Regularized Discounted Aggregation
- **Paper**: ICLR 2025 (accepted)
- **Source**: https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf

**What it does**: A new equilibrium-finding algorithm for MULTIPLAYER partially observable
stochastic games (POSGs) -- which includes extensive-form games with imperfect info.
Provably converges to EXACT Nash equilibrium in multiplayer settings via multi-round learning.

**Key innovation (ALGORITHM)**:
Multi-round discounted FTRL with KL-divergence regularization to a base policy:

```
Single-round dynamics:
  y_dot_t = v_i(pi_t) - y_t                           -- advantage accumulation
  pi_t = argmax_pi <pi, y_t> - eps * KL(pi || pi_base) -- regularized policy

Closed form:
  sigma(y_t)(a) = pi_base(a) * exp(y_t(a)/eps) / Z

Multi-round:
  Round l: set pi_base = pi_from_round_{l-1}, run single-round to rest point
  Repeat -> converges to exact Nash equilibrium
```

**Mathematical core**:
- Advantage value: v_i(pi)(x,a) = sum_h Pr(h|pi) A_i(h,a) / sum_h Pr(h|pi)
- Policy with KL regularization from base policy
- Linear convergence to rest point under local lambda-hypomonotonicity
- Rest points = generalized QRE (Nash distribution)
- Multi-round iteration converges to exact NE

**Multiplayer**: YES -- this is SPECIFICALLY designed for multiplayer games (not just 2-player)

**Why this is huge for Mahjong**: Most game-solving algorithms either (a) only work for
2-player zero-sum, or (b) converge to approximate/weak equilibrium in multiplayer. DRDA
provably finds EXACT Nash equilibrium in multiplayer POSGs. 4-player Mahjong is exactly
this class of game.

---

## TIER 2: HIGH RELEVANCE (Novel Algorithms, Needs Adaptation for Mahjong)

---

### 5. LAMIR -- Look-Ahead with Model in Imperfect-information Reasoning
- **Paper**: arxiv 2510.05048 (Oct 2025)
- **Source**: https://arxiv.org/abs/2510.05048

**What it does**: Learns an ABSTRACT MODEL of the imperfect-information game from interaction,
then uses it for look-ahead search at test time. No hand-built game model needed.

**Key innovation**:
- Learns a compressed/abstracted game model via agent-environment interaction
- At test time, does principled look-ahead in the learned abstract space
- Abstraction keeps subgames tractable for search
- With enough capacity, recovers exact game structure; with limited capacity, still useful

**Multiplayer**: Not explicitly demonstrated, but the framework is general
**Mahjong angle**: Could learn an abstract model of mahjong game dynamics for test-time search

---

### 6. Obscuro / KLUSS -- Knowledge-Limited Unfrozen Subgame Solving
- **Paper**: Zhang, Sandholm. arxiv 2506.01242 (June 2025)
- **Source**: https://arxiv.org/abs/2506.01242

**What it does**: General search techniques for imperfect-information games WITHOUT requiring
common knowledge (a standard assumption in prior subgame solving that fails in many real games).
Applied to create first superhuman Fog of War chess AI.

**Key innovation**:
- KLUSS: constructs imperfect-information subgames from sampled additional positions
- Does NOT require common knowledge assumption (which standard OLSS requires)
- Enables scalable search in games where players don't share knowledge of game state structure
- Significant advance over Sandholm's own prior work (Libratus, Pluribus OLSS)

**Multiplayer**: Framework is general; FoW chess is 2-player but technique extends
**Mahjong angle**: Mahjong lacks common knowledge (players don't know each other's hands).
KLUSS directly addresses this limitation of standard subgame solving.

---

### 7. Embedding CFR -- Continuous Embedding for Information Set Abstraction
- **Paper**: arxiv 2511.12083 (Nov 2025)
- **Source**: https://arxiv.org/abs/2511.12083

**What it does**: Replaces hard discrete clustering for information set abstraction with
pre-trained continuous embeddings (like word embeddings, but for game states).

**Key innovation**:
- Pre-train low-dimensional continuous embeddings for information sets
- Run CFR in embedding space instead of over hard clusters
- Soft/continuous abstraction preserves fine-grained distinctions lost by clustering
- Theoretical regret reduction guarantee
- First poker AI using pre-trained embeddings for abstraction

**Multiplayer**: Yes (CFR-based, extends to multiplayer)
**Mahjong angle**: Mahjong has ~10^48 information sets. Continuous embeddings could provide
much better abstraction than discrete tile-pattern clustering.

---

### 8. GPU-Accelerated CFR (MatrixCFR)
- **Paper**: Kim. arxiv 2408.14778 (Aug 2024)
- **Source**: https://arxiv.org/abs/2408.14778

**What it does**: Reformulates CFR as dense+sparse matrix/vector operations for massive GPU
parallelism. Up to 400x speedup vs OpenSpiel Python, 200x vs C++.

**Key innovation**:
- CFR tree traversal recast as matrix multiplications
- Sparse matrices for game tree structure, dense for strategy/regret tables
- All CFR variants (vanilla, CFR+, DCFR, Linear CFR) can be expressed this way
- Higher memory cost, but dramatically faster iteration

**Multiplayer**: Yes (CFR is multiplayer-compatible)
**Mahjong angle**: Training speed is a bottleneck. GPU-accelerated CFR could make
CFR-based approaches feasible for Mahjong's massive game tree.

---

## TIER 3: RELEVANT CONTEXT (From Same Research Groups / Adjacent Work)

---

### 9. Opponent Modeling with In-Context Search (NeurIPS 2024)
- **Team**: Haobo Fu et al. (Tencent AI -- LuckyJ group)
- **Note**: Builds on "Towards Offline Opponent Modeling with In-context Learning" (ICLR 2024)
- Uses in-context learning (transformer-style) to model opponents at test time
- Relevant for Mahjong: opponent modeling in 4-player setting is critical

### 10. RegFTRL -- Regularized Follow-the-Regularized-Leader (AAMAS 2025)
- **Source**: https://dl.acm.org/doi/abs/10.1145/3719545.3719556
- Adaptive regularization for last-iterate convergence to Nash equilibrium in 2p zero-sum
- Relevant as a building block for equilibrium computation

### 11. Student of Games (SoG) -- Science Advances 2023
- **Source**: https://www.science.org/doi/10.1126/sciadv.adg3256
- Unified algorithm: guided search + self-play + game-theoretic reasoning
- Works in BOTH perfect and imperfect information games
- DeepMind work -- benchmark for "general game AI"
- Already published (2023) so not novel for 2024-2025, but important baseline

### 12. MCU: Evaluation Framework for Open-Ended Game Agents (ICML 2025)
- **Team**: Haobo Fu et al. (Tencent AI)
- Evaluation framework, not an algorithm -- but shows where the LuckyJ team is heading

---

## SYNTHESIS: WHAT'S GENUINELY NOVEL FOR 4-PLAYER MAHJONG

The key insight from this survey: **the frontier has moved from "better CFR variants"
to three genuinely new algorithmic directions**:

### Direction A: Meta-Learned Equilibrium Finding
**Algorithms**: DDCFR -> Deep DDCFR -> PDCFR+
**Core idea**: Don't hand-tune CFR parameters. Learn them.
**Novel extension opportunity**: DDCFR learns discount weights. DRDA learns KL regularization
strength. Nobody has combined meta-learned dynamics with multiplayer-specific equilibrium
concepts (correlated equilibrium, team-maxmin, etc.)

### Direction B: Search Without Common Knowledge
**Algorithms**: KLUSS/Obscuro, LAMIR
**Core idea**: Do real-time search in imperfect-info games without assuming players share
knowledge of game structure.
**Novel extension opportunity**: Apply KLUSS-style search to 4-player Mahjong. Prior Mahjong
AIs (Suphx, LuckyJ) do NO search at test time -- they're pure policy networks. Adding
principled search that handles Mahjong's lack of common knowledge would be genuinely new.

### Direction C: Continuous Abstraction + Neural Equilibrium
**Algorithms**: Embedding CFR, DRDA
**Core idea**: Replace discrete game abstractions with continuous representations.
**Novel extension opportunity**: Learn continuous information-set embeddings for Mahjong
hands/states, then run equilibrium-finding in embedding space. This sidesteps the
intractable full game tree while preserving fine-grained hand distinctions.

---

## NOVELTY ASSESSMENT: What Would Pass a Reviewer's "Novel Algorithm" Bar

| Idea | Novel? | Risk |
|------|--------|------|
| "We apply DDCFR to Mahjong" | NO -- just application | Low novelty |
| "We combine policy gradient with CFR" | NO -- Dream/NFSP exist | Known recipe |
| "We extend DRDA to 4-player Mahjong with partial observability" | MAYBE -- if you prove new convergence bounds | Medium |
| "We design test-time search for Mahjong using KLUSS without common knowledge" | YES -- nobody has done this for Mahjong | High novelty |
| "We learn continuous info-set embeddings for Mahjong + run CFR in embedding space" | YES -- Embedding CFR is new and Mahjong is untouched | High novelty |
| "We meta-learn the equilibrium-finding dynamics for 4-player games (extending DDCFR to multiplayer-specific solution concepts)" | YES -- DDCFR is 2p, extending the meta-learning to multiplayer equilibria is open | High novelty |
| "We combine policy network with learned-model test-time search (LAMIR) for Mahjong" | YES -- first test-time search for Mahjong with learned models | High novelty |

---

## THE TOP 3 MOST PROMISING NOVEL ALGORITHM IDEAS FOR HYDRA

### Idea 1: KLUSS-Mahjong -- Test-Time Search Without Common Knowledge
Take KLUSS (Obscuro's core algorithm) and adapt it for 4-player Mahjong.
- Current SOTA Mahjong AIs are pure policy networks (no search)
- KLUSS handles imperfect info without common knowledge (perfect for Mahjong)
- You'd need: belief network over opponent hands + KLUSS subgame construction
- This is the AlphaGo->AlphaZero moment: adding search to a policy-only system
- Contribution: First test-time search algorithm for N-player imperfect info tile games

### Idea 2: Continuous Belief Equilibrium (DRDA + Embedding CFR hybrid)
Combine DRDA's multiplayer equilibrium finding with Embedding CFR's continuous abstraction.
- Learn continuous embeddings for Mahjong information sets (hand + discards + context)
- Run DRDA dynamics in embedding space for multiplayer equilibrium
- KL regularization from a pre-trained policy network (your Hydra policy becomes pi_base)
- Contribution: First algorithm for multiplayer equilibrium finding in continuous
  information-set embedding spaces. New mathematical framework.

### Idea 3: Meta-Learned Multiplayer Equilibrium Dynamics
Extend DDCFR's meta-learning to multiplayer-specific solution concepts.
- DDCFR learns discount schedules for 2-player CFR
- In 4-player games, the solution concept itself changes (no unique Nash, need selection)
- Meta-learn not just weights but the equilibrium selection criterion
- Contribution: First meta-learned equilibrium-finding framework for N>2 player games

---

## PAPER LINKS SUMMARY

| # | Algorithm | Venue | Year | URL |
|---|-----------|-------|------|-----|
| 1 | DDCFR | ICLR (Spotlight) | 2024 | https://github.com/rpSebastian/DDCFR |
| 2 | Deep DDCFR | arxiv | 2025 | https://arxiv.org/abs/2511.08174 |
| 3 | PDCFR+ | IJCAI | 2024 | https://www.ijcai.org/proceedings/2024/0583.pdf |
| 4 | DRDA | ICLR | 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf |
| 5 | LAMIR | arxiv | 2025 | https://arxiv.org/abs/2510.05048 |
| 6 | KLUSS/Obscuro | arxiv | 2025 | https://arxiv.org/abs/2506.01242 |
| 7 | Embedding CFR | arxiv | 2025 | https://arxiv.org/abs/2511.12083 |
| 8 | GPU-CFR | arxiv | 2024 | https://arxiv.org/abs/2408.14778 |
| 9 | Opponent Modeling InContext | NeurIPS | 2024 | (Haobo Fu, Tencent) |
| 10 | RegFTRL | AAMAS | 2025 | https://dl.acm.org/doi/abs/10.1145/3719545.3719556 |
