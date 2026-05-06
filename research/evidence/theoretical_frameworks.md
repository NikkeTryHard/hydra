# Novel Theoretical Frameworks for Mahjong AI

## Compilation of Mathematical Foundations for Imperfect-Information Game Play

**Context**: 4-player imperfect-info, stochastic draws, simultaneous offense/defense, hidden tiles, partial observability.

---

## Table of Contents

1. [Information-Theoretic Decision Making](#1-information-theoretic-decision-making)
2. [Bayesian Opponent Modeling](#2-bayesian-opponent-modeling-via-particle-filters)
3. [Online Learning & Regret Minimization](#3-online-learning--regret-minimization-cfr)
4. [Differential Game Theory](#4-differential-game-theory)
5. [Causal Inference in Games](#5-causal-inference-in-games)
6. [Information Geometry of Games](#6-information-geometry-of-games)
7. [Free Energy Principle & Active Inference](#7-free-energy-principle--active-inference)
8. [Algebraic / Compositional Game Theory](#8-algebraic--compositional-game-theory)
9. [Synthesis: A Unified Framework for Mahjong](#9-synthesis-a-unified-framework-for-mahjong)

---

## 1. Information-Theoretic Decision Making

### Core Idea
Do not maximize expected reward directly. Choose actions by *information ratio*: immediate regret vs information gained about hidden state.

### Source
Russo & Van Roy, "Learning to Optimize via Information-Directed Sampling" (2018).
[PDF](https://web.stanford.edu/~bvr/pubs/IDS.pdf) |
[arXiv:1403.5556](https://arxiv.org/abs/1403.5556)

### Key Formulations

**Instantaneous expected regret** of action `a` at time `t`:

```
Delta_t(a) := E[R_{t,A*} - R_{t,a} | F_t]
```

For sampling distribution pi over actions:

```
Delta_t(pi) = sum_a pi(a) * Delta_t(a)
```

**Information gain** from taking action `a` (mutual information between
optimal action and observation Y_{t,a}):

```
g_t(a) := I_t(A*; Y_{t,a})
        = D_KL( P((A*,Y_{t,a}) in . | F_t) || P(A* in . | F_t) * P(Y_{t,a} in . | F_t) )
```

Entropy view: expected uncertainty reduction about

```
g_t(a) = E[ H(alpha_t) - H(alpha_{t+1}) | F_t, A_t = a ]
```

where `alpha_t(a) = P(A* = a | F_t)` is posterior belief.

Information Ratio** (key concept):

```
Psi_t(pi) := Delta_t(pi)^2 / g_t(pi)
```

Meaning: squared expected regret per unit information.

**IDS Objective** -- at each step, solve:

```
pi_t^IDS = argmin_{pi in D(A)} { Psi_t(pi) }
```

**Regret Bound**:

```
If Psi_t(pi_t) <= lambda a.s. for all t, then:
    E[Regret(T, pi)] <= sqrt(lambda * H(alpha_1) * T)
```

### Application to Mahjong

In Mahjong, discard information is dual-use:
- **Forward info**: what discard reveals about my hand
- **Backward info**: what opponent discard reveals about theirs

IDS suggests discards minimizing `Psi_t`:
- Low Delta: low expected-value cost
- High g: high opponent-state information

This balances exploitation vs exploration. Defense emerges when `g_t` is high for safe tiles: cheap info, low cost.

---

## 2. Bayesian Opponent Modeling via Particle Filters

### Core Idea
Represent belief over each opponent's hidden state as weighted particles. Update online from discards and calls via Bayesian filtering.

### Source
Southey et al., "Particle Filtering for Dynamic Agent Modelling in
Simplified Poker" (AAAI 2007).
[PDF](https://webdocs.cs.ualberta.ca/~mbowling/papers/07aaai-om.pdf)

Also: Ganzfried & Sandholm, "Bayesian Opponent Modeling in Multiplayer
Imperfect-Information Games" (2022).
[arXiv:2212.06027](https://arxiv.org/abs/2212.06027)

### Key Formulations

**State-estimation target** -- posterior over opponent state given observations:

```
P(x_t | z_{1:t})
```

where `x_t` = opponent hidden state at time t, `z_{1:t}` = observations through time t.

**Recursive Bayesian filter**:

```
P(x_t | z_{1:t}) = eta * P(z_t | x_t) * integral[ P(x_t | x_{t-1}) * P(x_{t-1} | z_{1:t-1}) dx_{t-1} ]
```

where eta is normalization constant.

**Particle filter algorithm**:

1. **Proposal**: For each particle i, sample from motion model:
   ```
   x_tilde^(i) ~ P(x_t | x^(i)_{t-1})
   ```

2. **Importance weighting** (observation likelihood under particle):
   ```
   w_t^(i) proportional_to P(z_t | x_tilde^(i)_t)
   ```

3. **Resampling** (select particles proportional to weights):
   ```
   P(x_t^(i) = x_tilde^(j)) = w_t^(j) / sum_k w_t^(k)
   ```

**Motion models for opponent dynamics**:

*Switching model* (opponent may abruptly change strategy):
```
x_t = {
    Uniform random strategy,   with prob rho
    x_{t-1},                   with prob 1 - rho
}
```

*Drift model* (opponent gradually adapts):
```
x_t ~ N(x_{t-1}, sigma^2 * I)   [truncated to valid range]
```

*Combined model*:
```
x_t ~ {
    Uniform,                           with prob rho
    Truncated N(x_{t-1}, sigma^2 I),   with prob 1 - rho
}
```

**Rao-Blackwellized extension** (estimate dynamics params too):
```
theta_tilde ~ P(theta | s^(i)_{t-1})
x_tilde^(i) ~ P(x_t | x^(i)_{t-1}, theta_tilde)
s^(i)_t = UPDATE(s^(i)_{t-1}, x^(i)_{t-1} -> x_tilde^(i))
```

With conjugate priors:
```
rho ~ Beta(alpha, beta)
sigma^2 ~ InvGamma(v, w)
```

Updated on each transition:
```
If x_{t-1} = x_t:  beta <- beta + 1
If x_{t-1} != x_t: alpha <- alpha + 1
w <- w + ||x_t - x_{t-1}||^2 / 2
v <- v + d/2
```

### Application to Mahjong

Each opponent state `x_t` may encode:
- Hand-composition distribution
- Strategy params: aggression, tenpai probability, etc.
- Estimated shanten

Observation `z_t` may include:
- Discarded tile
- Whether they called pon/chi/kan
- Decision timing, riichi, etc.

Likelihood `P(z_t | x_t)` asks: given hidden state x, how likely was this observed action? This matches Hydra's forward-modeling need.

---

## 3. Online Learning & Regret Minimization (CFR)

### Core Idea
Rather than solve equilibrium directly, iteratively minimize *counterfactual regret*: gain missed by choosing differently at each decision point. Converges to Nash equilibrium.

### Sources
Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
(NeurIPS 2007). Formulations from
[labml.ai/cfr](https://nn.labml.ai/cfr/index.html) and
[stevengong.co/notes/CFR](https://stevengong.co/notes/Counterfactual-Regret-Minimization)

### Key Formulations

**Information set**: partition of game histories player i cannot distinguish:
```
I_i is a partition of { h in H : P(h) = i }
such that A(h) = A(h') whenever h,h' are in the same info set.
```

**Behavioral strategy**: action distribution at each info set:
```
sigma_i(I, a) = Pr(a | I),   a in A(I),  I in I_i
```

**Reach probability** (product of action probs reaching h):
```
pi^sigma(h) = pi^sigma_i(h) * pi^sigma_{-i}(h)
```

**Expected utility**:
```
u_i(sigma) = sum_{h in Z} u_i(h) * pi^sigma(h)
```

**Counterfactual value** of information set I under strategy sigma:
```
v_i(sigma, I) = sum_{z in Z_I} pi^sigma_{-i}(z[I]) * pi^sigma(z[I], z) * u_i(z)
```

Key insight: weight terminal states by opponent reach, factoring out player i's own contribution.

**Instantaneous counterfactual regret** for action at info set I:
```
r_i^t(I, a) = v_i(sigma^t |_{I->a}, I) - v_i(sigma^t, I)
```

**Cumulative regret**:
```
R_i^T(I, a) = (1/T) * sum_{t=1}^{T} r_i^t(I, a)
```

**Regret matching** (strategy update rule):
```
R_i^{T,+}(I,a) = max(R_i^T(I,a), 0)

sigma_i^{T+1}(I)(a) = {
    R_i^{T,+}(I,a) / sum_{a'} R_i^{T,+}(I,a'),   if sum > 0
    1 / |A(I)|,                                      otherwise
}
```

**Average strategy** (converges to equilibrium):
```
sigma_bar_i^T(I)(a) = sum_{t=1}^T pi_i^{sigma^t}(I) * sigma^t(I)(a)
                      / sum_{t=1}^T pi_i^{sigma^t}(I)
```

**Nash equilibrium convergence**: If R_i^T < epsilon for all players,
then sigma_bar^T is 2*epsilon-Nash equilibrium.

**Best response and exploitability**:
```
b_i(sigma_{-i}) = max_{sigma_i'} u_i(sigma_i', sigma_{-i})
exploitability = b_1(sigma_2) + b_2(sigma_1)
```

### Application to Mahjong

CFR powers poker AI like Libratus, Pluribus. For Mahjong:
- Information sets much larger
- 4-player equilibrium non-unique, less cleanly exploitable
- But counterfactual regret still valid training signal
- Deep CFR may scale with function approximation

Main problem: state space enormous. Need abstraction: group similar hands into equivalence classes.

---

## 4. Differential Game Theory

### Core Idea
Model game as continuous-time dynamics. Player strategies are controls; state evolves by differential equation. Value function satisfies PDE: Hamilton-Jacobi-Isaacs equation.

### Sources
- Evans & Souganidis, "Differential Games and Representation Formulas
for Solutions of Hamilton-Jacobi-Isaacs Equations" (1984).
[JSTOR](https://www.jstor.org/stable/45010271)
- "Stochastic Differential Games: Sampling Approach"
[PDF](https://dcsl.gatech.edu/papers/dgaa17%20(Printed).pdf)

### Key Formulations

**State dynamics** (N-player stochastic differential game):
```
dX_t = f(X_t, u_1, ..., u_N) dt + sigma(X_t) dW_t
```

where X_t is game state, u_i player i control, W_t Brownian motion.

**Hamilton-Jacobi-Isaacs (HJI) equation** (2-player zero-sum):
```
dV/dt + min_{u_2} max_{u_1} [ f(x,u_1,u_2) . grad_x V
                              + (1/2) tr(sigma sigma^T Hess_x V)
                              + L(x,u_1,u_2) ] = 0
```

with terminal condition V(T, x) = g(x).

**N-player generalization** (each player has own value function V_i):
```
dV_i/dt + H_i(x, grad V_1, ..., grad V_N) = 0
```

where H_i is player i's Hamiltonian:
```
H_i(x, p_1,...,p_N) = opt_{u_i} [ f(x,u) . p_i + L_i(x,u) ]
```

subject to others also optimizing; yields coupled PDEs.

**Isaacs condition** (sufficient for value existence):
```
min_{u_2} max_{u_1} H(x, p, u_1, u_2) = max_{u_1} min_{u_2} H(x, p, u_1, u_2)
```

**Viscosity solution** handles non-smooth value functions; practical notion of solution.

### Application to Mahjong

Treat turns as continuous progression. State X may include:
- Wall depletion
- Each player's threat level / distance to tenpai
- Point differentials

HJI view says value surface V(state) exists; optimal play follows its gradient. This justifies value-head approximation and gives:
1. Theoretical grounding for value approximation
2. Structure for attack/defense switching surfaces
3. Stochastic-control view where tile draws are noise `sigma*dW`

4-player Mahjong harder because system needs 4 coupled PDEs, not 1.

---

## 5. Causal Inference in Games

### Core Idea
Replace purely utility-max reasoning with *causal* reasoning via structural causal models. Enables counterfactual questions: "If opponent had different hand, would discard stay same?" Useful for stronger opponent modeling.

### Source
Bareinboim, Forney, Pearl, "Counterfactual Rationality: Causal Approach
to Game Theory" (Causal AI Lab).
[PDF](https://causalai.net/r125.pdf)

Also: Ibeling et al., "Reasoning about causality in games" (AIJ 2023).
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370223000656)

### Key Formulations

**Structural Causal Model (SCM)**:
```
M = <U, V, F, P(U)>
```
- U: exogenous hidden vars
- V: endogenous observed vars
- F: structural functions V <- f_V(Pa(V), U_V)
- P(U): distribution over exogenous vars

**Intervention (do-operator)**: `do(X = x)` replaces structural equation
for X with constant X <- x. Produces interventional distribution:
```
P_x(Y) = P(Y_x) = sum_{u: Y_x(u)=y} P(u)
```

**Causal Multi-Agent System (CMAS)**:
```
<M, N, X, Y>
```
- N: agents
- X = (X_1,..., X_n): disjoint action nodes
- Y = (Y_1,..., Y_n): reward nodes
- R_i: D(Y_i) -> R is reward function

**Three layers of causal reasoning in games**:

L1 (Observational): Follow natural mechanism f_{X_i}.
Action space A_1 = {a_0} (singleton -- "be yourself")

L2 (Interventional): Hard intervention do(x_i).
Standard game theory -- deliberate action choice.
Action space A_2 = D(X_i)

L3 (Counterfactual): Function h: D(X_i) -> D(X_i).
"What would I naturally do, how should I deviate?"
Natural tendency X_i^-> = f_{X_i}(U_i), executed as X_i <- h(X_i^->)
Special cases: h(x) = x is L1, constant h is L2.

**Causal Nash Equilibrium (CNE)**: Two-stage concept:

1. Layer Selection Game L^up: Each player chooses reasoning *layer*.
   ```
   u(A) = NE(Game(A_1, ..., A_n))
   ```

2. CNE: Let s^up be NE of L^up, and A_i^up = union of supports.
Then omega^up is CNE if it's Nash equilibrium of Game(A^up).

**Key theorem**: CNE exists; payoff weakly dominates unilateral layer-switch alternatives.

### Application to Mahjong

L3 counterfactual layer fits Mahjong defense exactly:

- **L1 thinking**: natural greedy offense discards 3m
- **L2 thinking**: deliberate switch to 7z
- **L3 thinking**: natural move would be 3m, but transform it into 7z given opponent model

This captures:
- **Reading opponents**: if they had X, would they discard Y? This is `P(Y_x)`
- **Signaling awareness**: my discard may change opponent behavior
- **Defense as causal intervention**: switching strategy is do-operation on my node

SCM can model full game causally:
Wall -> Draws -> Hands -> Discards -> Melds -> Outcomes, with hidden confounders like wall order and opponent hands.

---

## 6. Information Geometry of Games

### Core Idea
Game strategy spaces are probability simplices. Natural geometry is Fisher information, not Euclidean. Under this geometry, replicator dynamics become gradient flow of fitness; KL divergence becomes natural Lyapunov function.

### Source
Harper, "Information Geometry and Evolutionary Game Theory" (2009).
[arXiv:0911.1383](https://ar5iv.labs.arxiv.org/html/0911.1383)

Also: Jost & Li, "Natural gradient ascent in evolutionary games" (2024).
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0303264724000121)

### Key Formulations

**Strategy simplex**:
```
Delta^n = { x in R^n : sum_i x_i = 1, x_i >= 0 }
```

**Fisher information metric** (for categorical distributions):
```
g_{ij}(x) = (1/x_i) * delta_{ij}
```

This is unique up to scale among metrics invariant under sufficient statistics.

**Shahshahani metric** (on R_+^n, restricts to Fisher on simplex):
```
g_{ij}(x) = (||x|| / x_i) * delta_{ij},   ||x|| = sum_i x_i
```

**Replicator equation** (fundamental strategy-evolution dynamics):
```
x_dot_i = x_i * (f_i(x) - f_bar(x))
f_bar(x) = sum_i x_i * f_i(x)   [mean fitness]
```

**KEY THEOREM: Replicator = Shahshahani gradient ascent of fitness**

If f_i = dV/dx_i for some potential V, then replicator dynamics are
gradient flow of V under Shahshahani/Fisher metric:
```
x_hat_i(x) = x_i * (f_i(x) - f_bar(x))   [Shahshahani gradient]
```

**Exponential-map representation** (dual coordinates):
```
x_i = exp(v_i - G)
v_dot_i = f_i(x)
G_dot = f_bar(x)
```

This yields replicator dynamics. `v_i` are log-odds coordinates.

**KL divergence as Lyapunov function**:
```
V(x) = D_KL(x_hat || x) = sum_i x_hat_i * log(x_hat_i / x_i)
```

Time derivative along replicator flow:
```
V_dot(x) = -(x_hat . f(x) - x . f(x))
```

If x_hat is ESS, then V_dot < 0, so KL to equilibrium shrinks monotonically.

**Fisher information variance identity**:
```
Var_p[g] = || (dE[g])_p ||_p^2 = || (grad E[g])_p ||_p^2
```

Fitness variance equals squared norm of fitness gradient in Fisher geometry. This is natural selection's fundamental theorem in info-geometric form.

**Two-population dynamics** (attacker vs defender):
```
p_dot_i = p_i * (f_i(p,q) - E_p[f(p,q)])
q_dot_j = q_j * (g_j(p,q) - E_q[g(p,q)])
```

Block metric:
```
G_{ij}(p,q) = {
    1/p_i,  if i=j <= n
    1/q_i,  if i=j > n
    0,      otherwise
}
```

**KL divergence induced metric** (showing Fisher = KL Hessian):
```
g_{ij}^(D) = d^2 D / (dx_i dy_j) |_{x=y}

For KL: d^2 D_KL(x||y) / (dx_i dy_j) |_{x=y} = (1/x_i) * delta_{ij}
```

Thus Fisher metric is infinitesimal KL divergence.

### Application to Mahjong

This view gives:

1. **Natural gradient for policy learning**: use Fisher metric instead of vanilla SGD.
   ```
   theta_{t+1} = theta_t - alpha * F^{-1}(theta) * grad L(theta)
   ```
where F is policy Fisher matrix.

2. **KL regularization is natural**: KL penalty, as in PPO, matches strategy-manifold geometry.

3. **Strategy dynamics interpretation**: Mahjong meta evolution follows replicator dynamics on Fisher manifold. Convergence measured by KL.

4. **Fitness landscape**: strategy fitness = expected score vs population. Replicator dynamics predict frequency shifts; useful for curriculum design.

---

## 7. Free Energy Principle & Active Inference

### Core Idea
Model Mahjong agent as minimizing *free energy*, upper-bounding surprise. Agent keeps generative model of game; both belief update and action selection come from one objective: reduce mismatch between prediction and reality.

### Sources
- Parr & Friston, "Generalised free energy and active inference" (2019).
[PMC6848054](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/)
- Da Costa et al., "Distributionally robust free energy principle for
decision-making" (Nature Comms, 2025).
[Nature](https://www.nature.com/articles/s41467-025-67348-6)

### Key Formulations

**Variational Free Energy** (core objective):
```
F = E_Q[ ln Q - ln P(o, s, pi) ] >= -ln P(o)
```

Free energy upper-bounds surprise `-ln P(observations)`. Minimizing F performs approximate Bayesian inference.

**Mean-field factorization**:
```
Q(s_tilde, pi) approx Q(pi) * prod_tau Q(s_tau | pi)
```

**Free energy decomposition by policy**:
```
F = E_{Q(pi)}[F_pi] + D_KL(Q(pi) || P(pi))
```

where F_pi is free energy conditioned on policy pi.

**Policy belief update** (softmax of negative free energy):
```
Q(pi) proportional_to exp(ln P(pi) - F_pi)
Q(pi) = softmax(ln P(pi) - F_pi)
```

Policies predicting observations well (low F_pi) get higher probability.

**Expected Free Energy (EFE)** -- key quantity for future decisions:
```
G_pi = E_{Q(o,s|pi)}[ ln Q(s|pi) - ln P(o,s|pi) ]
```

**EFE decomposition into risk and ambiguity**:
```
G_pi = D_KL(Q(o|pi) || P(o))     [risk / pragmatic value]
     + E_{Q(s|pi)}[ H(P(o|s)) ]  [ambiguity / epistemic value]
```

- **Risk**: predicted outcomes vs preferred outcomes
- **Ambiguity**: expected observation uncertainty after action

**Combined policy selection** (with expected free energy as prior):
```
Q(pi) = softmax( ln E(pi) - F_pi - G_pi )
```

where E(pi) encodes habits/preferences.

**DR-FREE: Distributionally Robust Free Energy** (model uncertainty):
Joint trajectory distribution:
```
p_{0:N} = p_0(x_0) * prod_{k=1}^{N} p_k(x_k | x_{k-1}, u_k) * pi_k(u_k | x_{k-1})
```

robust objective (minimax over model uncertainty):
```
min_{pi_k} max_{p_k in B_eta(p_bar_k)}
    [ D_KL(p_{0:N} || q_{0:N}) + E_{p_{0:N}}[ sum_k (c_k^x(X_k) + c_k^u(U_k)) ] ]
```

where B_eta is ambiguity set (KL ball around trained model):
```
B_eta(p_bar_k) = { p_k : D_KL(p_k || p_bar_k) <= eta_k }
```

**Optimal robust policy** (Gibbs/softmax form):
```
pi_k*(u_k | x_{k-1}) proportional_to
    q_k(u_k | x_{k-1}) * exp(- c_k^u(u_k) - eta_k(x_{k-1}, u_k) - c_tilde(x_{k-1}, u_k))
```

### Application to Mahjong

Active inference fits Mahjong unusually well:

1. **Generative model = game engine**: predicts draws, opponent discards, outcomes. This is what hydra-core simulates.

2. **EFE balances offense and defense**:
   - Risk: does action lead toward preferred outcomes?
   - Ambiguity: does action reduce uncertainty about opponents?
   - Safe discard may be low risk and low ambiguity.

3. **Robustness**: DR-FREE handles wrong opponent models. Larger eta = more paranoia = more defense.

4. **Unified perception-action loop**: belief update and action choice become same optimization.

---

## 8. Algebraic / Compositional Game Theory

### Core Idea
Build games compositionally from small pieces using categorical algebra, not monolithically. Games are morphisms in symmetric monoidal category; sequence = composition, simultaneity = tensor product. Good fit for decomposing Mahjong into reusable parts.

### Source
Ghani, Hedges, Winschel, Zahn, "Compositional Game Theory" (LICS 2018).
[arXiv:1603.04641](https://arxiv.org/abs/1603.04641)

Hedges, "Towards Compositional Game Theory" (PhD thesis, Oxford 2016).
[PDF](https://www.cs.ox.ac.uk/people/julian.hedges/papers/Thesis.pdf)

### Key Formulations

**Open game** (type signature): open game G has type:
```
G : X (x) S* -> Y (x) R*
```

where:
- X: input from environment
- Y: output/actions to environment
- R: utility flowing back from environment
- S: utility passed backward to earlier stages

open game is specified by 4 components:

1. **Strategy set**: Sigma_G

2. **Play** (forward information flow):
   ```
   P_G : Sigma_G -> Hom(X, Y)
   ```

3. **Coplay** (backward utility flow / coutility):
   ```
   C_G : Sigma_G -> Hom(X (x) R, S)
   ```

4. **Best response relation**:
   ```
   B_G : Hom(I, X) x Hom(Y, R) -> Sigma_G -> P(Sigma_G)
   ```

**Lens/optic structure**: For fixed strategy sigma, pair (play, coplay) forms lens:
```
(P_G(sigma), C_G(sigma))  :  (X, S) <-> (Y, R)
```

Play goes forward (X -> Y), coplay backward (X x R -> S).

**Sequential composition** (G then H):
```
H . G : X (x) S* -> Z (x) T*

Sigma_{H.G} = Sigma_G x Sigma_H
```

**Parallel composition** (G and H simultaneously):
```
G (x) H : (X1 (x) X2) (x) (S1 (x) S2)* -> (Y1 (x) Y2) (x) (R1 (x) R2)*

Sigma_{G(x)H} = Sigma_G x Sigma_H
```

**Nash equilibrium** (categorical best-response fixed point):

Given context (h, k) with h: I -> X (history), k: Y -> R (continuation):
```
sigma is equilibrium  iff  sigma in B_G(h, k)(sigma)
```

Strategy sigma is equilibrium when it is fixed point of best response.

**Coutility**: backward-flowing value:
```
C_G(sigma) : X (x) R -> S
```

Takes current state/history plus future utility R, returns utility S passed back to earlier stages.

### Application to Mahjong

Compositional game theory breaks Mahjong into modular subgames:

Mahjong round as composition**:
```
Round = Deal ; (Turn_1 (x) Turn_2 (x) Turn_3 (x) Turn_4)^{*n} ; Score
```

Where:
- Deal: I -> HandState (x) WallState
- Turn_i: GameState -> GameState
- Score: GameState -> Points (x) Points (x) Points (x) Points

Each Turn is itself composed:
```
Turn = Draw ; Evaluate ; (Discard | Call | Win)
```

**Why this matters**:

1. **Modularity**: reason about `Turn_i` alone, then compose. Defensive and aggressive subpolicies combine cleanly.

2. **Coutility = downstream impact**: coplay function C propagates future cost backward through composition, analogous to backprop.

3. **Equilibrium preservation**: if each subgame has equilibrium, composition preserves equilibrium properties.

4. **Formal verification**: categorical structure may prove whole-game properties from subparts, useful for rule-respecting AI.

---

## 9. Synthesis: A Unified Framework for Mahjong

### The Big Picture

These 8 frameworks form one stack, not 8 isolated ideas:

```
Layer 4: ALGEBRAIC STRUCTURE (Compositional Game Theory)
         Decomposes the full game into modular subgames.
         Provides formal guarantees about composition.
              |
Layer 3: DECISION CRITERION (choose one or combine)
         [Information-Directed Sampling] -- minimize information ratio
         [Active Inference / Free Energy] -- minimize expected free energy
         [Counterfactual Regret]         -- minimize cumulative regret
              |
Layer 2: BELIEF REPRESENTATION
         [Particle Filters]     -- nonparametric belief over opponent states
         [Bayesian Networks]    -- structured probabilistic model of game
         [Causal SCM]           -- interventional/counterfactual reasoning
              |
Layer 1: GEOMETRY & DYNAMICS
         [Information Geometry]   -- Fisher metric on strategy space
         [Differential Games]    -- continuous-time value function PDEs
         [Replicator Dynamics]   -- population strategy evolution
```

### Concrete Proposal: Information-Theoretic Active Inference for Mahjong

Combine strongest parts into one framework:

**State**: At each decision point, maintain:
```
b_t = {
    hand:     own tiles (known),
    wall:     posterior over remaining wall (Bayesian),
    opp[i]:   particle cloud over opponent i's hand + strategy,
    value:    estimated game value V(state)
}
```

**Decision criterion**: Expected Free Energy with information-directed twist:
```
a* = argmin_a [ G(a) / (1 + lambda * g(a)) ]
```

where:
- G(a) = risk(a) + ambiguity(a) is expected free energy
- g(a) = mutual information gain about opponents from action
- lambda controls exploration-exploitation balance

This reduces to:
- Pure active inference when lambda = 0
- Pure information-directed when risk is constant
- Hybrid balancing offense and intelligence gathering

**Belief update**: After each observation (opponent discard, call, etc.):
```
For each opponent i:
    For each particle j:
        w_j <- P(observation | particle_j)   [likelihood]
    Resample particles proportional to weights
    Apply motion model (drift/switch) to capture strategy shifts
```

**Policy learning** (training time): Use natural gradient (Fisher geometry)
on policy network, with:
- CFR-style counterfactual regret as training signal
- KL regularization against reference policy
- Causal opponent reasoning: "given opponent discarded X,
what interventional distribution over their hand follows?"

**Strategy update rule** (combines information geometry + regret matching):
```
theta_{t+1} = theta_t - alpha * F^{-1}(theta) * grad[ L_policy + beta * R_cfr + gamma * D_KL(pi || pi_ref) ]
```

where:
- L_policy = standard policy gradient loss
- R_cfr = counterfactual regret term
- D_KL = KL divergence regularization
- F^{-1} = inverse Fisher information matrix (natural gradient)

### What's Genuinely Novel Here

Current Mahjong AI landscape (Suphx, Mortal) mainly uses:
- Standard RL (PPO, actor-critic)
- Neural function approximation
- Hand-crafted features or self-play

Framework additions here:

1. **Information-theoretic action selection**: optimize information ratio, not only value.

2. **Particle filter opponent tracking**: adapt online to specific opponent tendencies.

3. **Causal counterfactual reasoning**: formalize "if opponent had Y, would they discard Z?"

4. **Free energy as unified objective**: one principled objective for perception and action, not ad-hoc head/loss mixing.

5. **Natural gradient on strategy manifold**: geometrically correct updates on probability simplices, unlike crude trust-region approximations.

---

## References

1. Russo & Van Roy. "Learning to Optimize via Information-Directed Sampling."
NeurIPS 2018. https://web.stanford.edu/~bvr/pubs/IDS.pdf

2. Southey et al. "Particle Filtering for Dynamic Agent Modelling."
AAAI 2007. https://webdocs.cs.ualberta.ca/~mbowling/papers/07aaai-om.pdf

3. Ganzfried & Sandholm. "Bayesian Opponent Modeling in Multiplayer
Imperfect-Information Games." 2022. https://arxiv.org/abs/2212.06027

4. Zinkevich et al. "Regret Minimization in Games with Incomplete
Information." NeurIPS 2007.
Formulations: https://nn.labml.ai/cfr/index.html

5. Evans & Souganidis. "Differential Games and HJI Equations." 1984.
https://www.jstor.org/stable/45010271

6. Bareinboim et al. "Counterfactual Rationality: Causal Approach to
Game Theory." https://causalai.net/r125.pdf

7. Harper. "Information Geometry and Evolutionary Game Theory." 2009.
https://ar5iv.labs.arxiv.org/html/0911.1383

8. Parr & Friston. "Generalised Free Energy and Active Inference." 2019.
https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/

9. Da Costa et al. "Distributionally Robust Free Energy Principle for
Decision-Making." Nature Comms 2025.
https://www.nature.com/articles/s41467-025-67348-6

10. Ghani, Hedges et al. "Compositional Game Theory." LICS 2018.
https://arxiv.org/abs/1603.04641

11. Hedges. "Towards Compositional Game Theory." Oxford PhD thesis 2016.
https://www.cs.ox.ac.uk/people/julian.hedges/papers/Thesis.pdf

12. Farina. "Game-Theoretic Decision Making in Imperfect-Information Games."
MIT PhD thesis 2023.
https://www.mit.edu/~gfarina/2023/phd_thesis/FARINA-Thesis-2023.pdf