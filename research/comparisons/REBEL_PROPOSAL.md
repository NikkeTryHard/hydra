# Recursive Belief-based Learning: Sound Search in Imperfect-Information Games via Public Belief States

---

## 1. Introduction

We propose a framework for imperfect-information games that resolves a fundamental challenge: how to apply the powerful combination of self-play and search (AlphaZero paradigm) to games with hidden information.

The core insight: any imperfect-information game can be reformulated as a continuous-state, continuous-action **perfect-information game** over Public Belief States (PBS). In this transformed game, the state is no longer the hidden game state but the probability distribution over hidden states -- which is common knowledge. Search algorithms designed for perfect-information games (MCTS, CFR) then apply directly to the belief space.

This yields **ReBeL** (Recursive Belief-based Learning): a self-play algorithm that interleaves depth-limited search in belief space with value network training, achieving sound play with formal exploitability bounds.

---

## 2. Public Belief States (PBS)

### 2.1 Definition

Let s_pub be the public state (all commonly known information). Let S_i(s_pub) be the set of infostates player i may hold. A PBS is a joint probability distribution over all players' infostates:

$$\beta = \bigl(\Delta S_1(s_{pub}),\;\ldots,\;\Delta S_N(s_{pub})\bigr)$$

where Delta S_i(s_pub) is a probability distribution over player i's possible private information.

### 2.2 The Transformation Theorem

Any imperfect-information game becomes a perfect-information game when the state space is replaced by the PBS space. In this transformed game:
- **States** are probability distributions (continuous, common knowledge)
- **Actions** are the same as the original game
- **Transitions** are deterministic: given PBS beta and action a, the next PBS is uniquely determined by Bayes' rule
- **A value network V(beta)** can be trained via standard self-play

This is the key insight: the information barrier vanishes when you reason about beliefs rather than states.

### 2.3 PBS Value Function

The value of a PBS for agent i under policy profile pi:

$$V_i^\pi(\beta) = \sum_{h \in \mathcal{H}(s_{pub}(\beta))} p(h|\beta) \cdot v_i^\pi(h)$$

Key structural properties (Lemmas 1-2):
- V_1^{pi_2}(beta) is **linear** in beta_1 (fixing opponent strategy)
- V_1(beta) = min_{pi_2} V_1^{pi_2}(beta) is **concave** (the game value is concave in beliefs)
- The minimizing pi_2 are precisely the Nash equilibrium policies at beta

These properties enable gradient-based optimization and ensure the value function is well-behaved for neural network approximation.

### 2.4 The Supergradient Connection

The relationship between infostate values and the PBS value function:

$$v_1^{\pi^*}(s_1|\beta) = V_1(\beta) + g \cdot \hat{s}_1$$

where g is a supergradient of V_1(beta) with respect to the unnormalized belief distribution. This equation is the theoretical foundation for learning a VECTOR-valued network: the value at each infostate is the PBS value plus a linear correction determined by the gradient of the value landscape.

---

## 3. Search: CFR-AVG

### 3.1 Algorithm

At each decision point, ReBeL constructs a depth-limited subgame and solves it via CFR-AVG (a variant of CFR Decomposition using average strategy):

```
function SEARCH(beta_r, V_net, depth D, iterations T):
    G = construct_subgame(beta_r, depth=D)
    
    for t = 1 to T:
        pi^t = CFR_update(G, pi^{t-1})          // standard regret matching
        pi_bar^t = (t/(t+1)) * pi_bar^{t-1} + (1/(t+1)) * pi^t
        
        // Leaf evaluation using value network on AVERAGE-strategy PBS
        for each leaf PBS beta_z in G:
            beta_z_avg = compute_PBS(beta_z, pi_bar^t)  // beliefs under avg policy
            v^t(s_i) = t * V_net(s_i | beta_z_avg) - (t-1) * V_net(s_i | beta_z_avg_prev)
        
        v(beta_r) = (t/(t+1)) * v(beta_r) + (1/(t+1)) * compute_EV(G, pi^t)
    
    return pi_bar^T, v(beta_r)
```

The modified leaf value (line with t * V_net - (t-1) * V_net) is a key empirical improvement: it debiases the leaf evaluation toward the current iteration's beliefs, preventing stale estimates from earlier iterations from polluting the search.

### 3.2 Warm Start

Initialize CFR regrets using a policy network's best-response computation, scaled by factor 15 (simulating 15 CFR iterations). This reduces the number of actual iterations needed from ~1000 to ~100-200.

### 3.3 Test-Time Policy

At test time, sample a random iteration t ~ Uniform{1, T} and play pi^t (NOT the average policy pi_bar^T). Pass the beliefs from that iteration's policy to the next subgame. This maintains theoretical convergence properties.

---

## 4. Self-Play Training Loop

ReBeL's training interleaves search with value network updates:

```
Repeat until convergence:
    1. Start from initial PBS beta_0
    2. While game not terminal:
        a. Run SEARCH(beta_current, V_net, D, T) to get pi_bar, v(beta)
        b. Add {beta_current, v(beta_current)} to value training buffer
        c. Add {beta, pi_bar(beta)} for all beta in subgame to policy buffer
        d. Sample leaf PBS beta' by playing in the subgame with exploration
        e. beta_current = beta'
    3. Train V_net on value buffer (Huber loss)
    4. Train pi_net on policy buffer (for warm-starting future searches)
```

The recursive structure: the value network is trained on search outputs, and the search uses the value network for leaf evaluation. This creates a bootstrap where each improves the other -- analogous to AlphaZero's self-play + MCTS loop but extended to imperfect information.

---

## 5. Formal Guarantees

### Theorem 2: Value Approximation

Running the self-play loop with T iterations of CFR per subgame produces a value network with error at most C/sqrt(T) for any PBS encountered during play, where C is a game-dependent constant.

### Theorem 3: Test-Time Exploitability

With a value network of error at most delta and T search iterations at test time, the algorithm plays a:

$$(C_1 \cdot \delta + C_2 \cdot \delta / \sqrt{T})\text{-Nash equilibrium}$$

This is the key result: exploitability decreases with BOTH value network quality AND search depth. More search at test time PROVABLY reduces exploitability.

### Theorem 5: CFR-AVG Convergence

T iterations of CFR-AVG in a depth-limited subgame (with exactly-solved leaf subgames) produces a C/sqrt(T)-Nash equilibrium.

---

## 6. Theoretical Strengths

### 6.1 The Belief-Space Transformation

The transformation from imperfect-info to perfect-info over beliefs is not a heuristic -- it's a mathematical equivalence. Any imperfect-info game IS a perfect-info game over beliefs. This means:
- The entire AlphaZero paradigm (self-play + search + neural value function) applies
- Sound search algorithms (CFR) work directly on the belief space
- No approximation is introduced by the transformation itself

### 6.2 Formal Exploitability Bounds

ReBeL is one of very few systems with FORMAL bounds on how exploitable the resulting strategy is. The bound tightens with more compute (both training and inference), providing a clear compute-quality tradeoff.

### 6.3 No Abstraction Required

Unlike prior poker AI (Libratus, DeepStack), ReBeL requires no hand-crafted action abstraction or information abstraction. The value network learns directly on the full game representation.

### 6.4 Generality

The framework applies to ANY imperfect-information game with perfect recall. No domain-specific adaptation is required for the core algorithm -- only the network architecture and input representation change.

---

## 7. Limitations

1. **2-player only**: ReBeL's theoretical guarantees (Theorems 2, 3, 5) hold for 2-player zero-sum games. Extension to N>2 players breaks the minimax structure that CFR depends on. The concavity of V_1(beta) (Lemma 2) relies on the zero-sum property.

2. **Exact PBS is intractable for large games**: For poker, |S_i| = 1326 (possible hole card combinations), so the PBS is a 1326-dimensional vector per player. For games with larger information state spaces (e.g., Mahjong with ~10^10 possible hands), exact PBS representation is computationally infeasible.

3. **CFR convergence rate**: O(1/sqrt(T)) convergence requires many iterations. Each iteration requires traversing the subgame tree and evaluating the value network at every leaf. For games with large branching factors, this is expensive.

4. **No opponent exploitation**: ReBeL converges toward Nash equilibrium. Against non-Nash opponents (e.g., humans), it does not adapt or exploit -- it plays the game-theoretically safe strategy regardless of opponent behavior.

5. **No belief constraints**: PBS distributions are learned implicitly through the value network. There is no mechanism to enforce domain-specific constraints on beliefs (e.g., tile conservation in Mahjong, card counting in bridge). Beliefs may be internally inconsistent.

6. **No information-theoretic reasoning**: Actions are evaluated on expected value only. There is no mechanism to reason about information gain ("will this action teach me about opponent's hand?") or information concealment ("will this action reveal my hand?").

7. **Search is expensive**: 90 DGX-1 machines (720 V100 GPUs) for training on heads-up poker. Inference requires running ~100-200 CFR iterations per decision, each requiring multiple value network evaluations.

---

## 8. Architecture

| Component | Specification |
|-----------|--------------|
| Value network | 6-layer MLP, 1536 units, GeLU, LayerNorm |
| Input | 1 + 1 + 1 + 5 + 2*1326 = 2660 dimensions (agent + pot + board + beliefs) |
| Output | R^{|S_1| + |S_2|} (infostate values, not scalar) |
| Loss | Pointwise Huber |
| Optimizer | Adam, lr=3e-4, halved every 800 epochs |
| Replay buffer | 12M examples, circular |
| CFR iterations | T=1750 per subgame (training), ~200 (inference) |
| Exploration | epsilon=0.25 (uniform random action with 25% probability) |
| Action space | 9 discrete bet sizes (fixed pot fractions) |

---

## References

1. Brown, Sandholm. "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games." NeurIPS 2020. arXiv: 2007.13544.
2. Zinkevich et al. "Regret Minimization in Games with Incomplete Information." NeurIPS 2007.
3. Brown et al. "Deep Counterfactual Regret Minimization." ICML 2019.
4. Silver et al. "A General RL Algorithm that Masters Chess, Shogi, and Go." Science 2018.
5. Brown, Sandholm. "Superhuman AI for Multiplayer Poker." Science 2019.
6. Moravcik et al. "DeepStack: Expert-Level AI in Heads-Up No-Limit Poker." Science 2017.

</content>
