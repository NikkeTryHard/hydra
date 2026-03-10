# Novel Theoretical Frameworks for Mahjong AI

## Compilation of Mathematical Foundations for Imperfect-Information Game Play

**Context**: 4-player imperfect information, stochastic draws, simultaneous
exploitation/defense, hidden tiles, partial observability.

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
Instead of maximizing expected reward directly, choose actions that optimize the
*information ratio* -- the tradeoff between immediate regret and information
gained about the hidden game state.

### Source
Russo & Van Roy, "Learning to Optimize via Information-Directed Sampling" (2018).
[PDF](https://web.stanford.edu/~bvr/pubs/IDS.pdf) |
[arXiv:1403.5556](https://arxiv.org/abs/1403.5556)

### Key Formulations

**Instantaneous expected regret** of action `a` at time `t`:

```
Delta_t(a) := E[R_{t,A*} - R_{t,a} | F_t]
```

For a sampling distribution pi over actions:

```
Delta_t(pi) = sum_a pi(a) * Delta_t(a)
```

**Information gain** from taking action `a` (mutual information between
optimal action A* and observation Y_{t,a}):

```
g_t(a) := I_t(A*; Y_{t,a})
        = D_KL( P((A*,Y_{t,a}) in . | F_t) || P(A* in . | F_t) * P(Y_{t,a} in . | F_t) )
```

Entropy interpretation -- how much uncertainty about A* we expect to resolve:

```
g_t(a) = E[ H(alpha_t) - H(alpha_{t+1}) | F_t, A_t = a ]
```

where `alpha_t(a) = P(A* = a | F_t)` is the posterior belief.

**The Information Ratio** (the key concept):

```
Psi_t(pi) := Delta_t(pi)^2 / g_t(pi)
```

This is "squared expected regret per unit of information gained."

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

In Mahjong, the "information gain" of a discard is dual-purpose:
- **Forward info**: what does this discard reveal about my hand to opponents?
- **Backward info**: what does an opponent's discard tell me about their tiles?

The IDS framework suggests choosing discards that minimize `Psi_t`:
- Low Delta (not too costly in expected value)
- High g (maximally informative about opponent states)

This naturally balances exploitation (play for points) vs. exploration
(learn what opponents hold). Defensive play emerges when `g_t` is high
for safe tiles -- you learn opponent intent cheaply.


---

## 2. Bayesian Opponent Modeling via Particle Filters

### Core Idea
Represent beliefs about each opponent's hidden state (their hand, their
strategy) as a cloud of weighted particles. Update beliefs in real-time
using Bayesian filtering as you observe their discards and claims.

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

where `x_t` = opponent's hidden state at time t, `z_{1:t}` = all
observations up to time t.

**Recursive Bayesian filter**:

```
P(x_t | z_{1:t}) = eta * P(z_t | x_t) * integral[ P(x_t | x_{t-1}) * P(x_{t-1} | z_{1:t-1}) dx_{t-1} ]
```

where eta is a normalization constant.

**Particle filter algorithm**:

1. **Proposal**: For each particle i, sample from motion model:
   ```
   x_tilde^(i) ~ P(x_t | x^(i)_{t-1})
   ```

2. **Importance weighting** (likelihood of observation given particle):
   ```
   w_t^(i) proportional_to P(z_t | x_tilde^(i)_t)
   ```

3. **Resampling** (select particles proportional to weights):
   ```
   P(x_t^(i) = x_tilde^(j)) = w_t^(j) / sum_k w_t^(k)
   ```

**Motion models for opponent dynamics**:

*Switching model* (opponent may suddenly change strategy):
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

**Rao-Blackwellized extension** (estimate dynamics parameters too):
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

Each opponent's state `x_t` could encode:
- Estimated hand composition (probability over tile sets)
- Current strategy parameters (aggression level, tenpai probability, etc.)
- Estimated shanten count

The observation `z_t` at each step includes:
- Which tile was discarded
- Whether they called pon/chi/kan
- Timing of their decisions (riichi declaration, etc.)

The likelihood `P(z_t | x_t)` asks: "Given that opponent has hand state x,
how likely is this observed discard?" -- this is exactly the kind of
forward model Hydra's encoder already captures.


---

## 3. Online Learning & Regret Minimization (CFR)

### Core Idea
Instead of computing an equilibrium directly, iteratively minimize
*counterfactual regret* -- the amount you would have gained by playing
differently at each decision point. Converges to Nash equilibrium.

### Sources
Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
(NeurIPS 2007). Formulations from
[labml.ai/cfr](https://nn.labml.ai/cfr/index.html) and
[stevengong.co/notes/CFR](https://stevengong.co/notes/Counterfactual-Regret-Minimization)

### Key Formulations

**Information set**: partition of game histories where player i cannot
distinguish states:
```
I_i is a partition of { h in H : P(h) = i }
such that A(h) = A(h') whenever h,h' are in the same info set.
```

**Behavioral strategy**: probability distribution over actions at each info set:
```
sigma_i(I, a) = Pr(a | I),   a in A(I),  I in I_i
```

**Reach probability** (product of all players' action probabilities to reach h):
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

Key insight: this weights terminal states by *opponent* reach probability,
factoring out player i's own contribution.

**Instantaneous counterfactual regret** for action a at info set I:
```
r_i^t(I, a) = v_i(sigma^t |_{I->a}, I) - v_i(sigma^t, I)
```

**Cumulative regret**:
```
R_i^T(I, a) = (1/T) * sum_{t=1}^{T} r_i^t(I, a)
```

**Regret matching** (the strategy update rule):
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
then sigma_bar^T is a 2*epsilon-Nash equilibrium.

**Best response and exploitability**:
```
b_i(sigma_{-i}) = max_{sigma_i'} u_i(sigma_i', sigma_{-i})
exploitability = b_1(sigma_2) + b_2(sigma_1)
```

### Application to Mahjong

CFR is the backbone of poker AI (Libratus, Pluribus). For Mahjong:
- Information sets are much larger (136 tiles vs 52 cards)
- 4-player means Nash equilibrium is not unique and may not be exploitable
- But *counterfactual regret* is still a valid training signal
- Deep CFR (neural network function approximation) could scale

The key challenge: Mahjong's information sets are enormous. Abstraction
is mandatory -- grouping similar hands into equivalence classes.


---

## 4. Differential Game Theory

### Core Idea
Model the game as a continuous-time dynamical system where players'
strategies are controls, and the game state evolves according to a
differential equation. The value function satisfies a PDE (the
Hamilton-Jacobi-Isaacs equation).

### Sources
- Evans & Souganidis, "Differential Games and Representation Formulas
  for Solutions of Hamilton-Jacobi-Isaacs Equations" (1984).
  [JSTOR](https://www.jstor.org/stable/45010271)
- "Stochastic Differential Games: A Sampling Approach"
  [PDF](https://dcsl.gatech.edu/papers/dgaa17%20(Printed).pdf)

### Key Formulations

**State dynamics** (N-player stochastic differential game):
```
dX_t = f(X_t, u_1, ..., u_N) dt + sigma(X_t) dW_t
```

where X_t is the game state, u_i is player i's control, W_t is Brownian motion.

**Hamilton-Jacobi-Isaacs (HJI) equation** (2-player zero-sum):
```
dV/dt + min_{u_2} max_{u_1} [ f(x,u_1,u_2) . grad_x V
                              + (1/2) tr(sigma sigma^T Hess_x V)
                              + L(x,u_1,u_2) ] = 0
```

with terminal condition V(T, x) = g(x).

**N-player generalization** (each player has their own value function V_i):
```
dV_i/dt + H_i(x, grad V_1, ..., grad V_N) = 0
```

where H_i is player i's Hamiltonian:
```
H_i(x, p_1,...,p_N) = opt_{u_i} [ f(x,u) . p_i + L_i(x,u) ]
```

subject to other players also optimizing (coupled system of PDEs).

**Isaacs condition** (sufficient for value to exist):
```
min_{u_2} max_{u_1} H(x, p, u_1, u_2) = max_{u_1} min_{u_2} H(x, p, u_1, u_2)
```

**Viscosity solution** concept handles non-smooth value functions --
the correct notion of "solution" for these PDEs in practice.

### Application to Mahjong

Think of the "game clock" as turns progressing continuously. The state
X includes:
- Wall depletion rate (tiles remaining)
- Each player's "threat level" (how close to tenpai)
- Point differentials

The HJI perspective says: there exists a value surface V(state) such that
optimal play is the gradient of V. This is exactly what a neural network
value head approximates! The differential game formulation gives us:

1. **Theoretical justification** for value function approximation
2. **Structure of optimal switching** -- when to switch from attack to defense
   (the "switching surfaces" in the value function)
3. **Stochastic control** interpretation -- tile draws are the noise term sigma*dW

The multi-player HJI system shows why 4-player Mahjong is fundamentally
harder: you need 4 coupled PDEs, not one.


---

## 5. Causal Inference in Games

### Core Idea
Replace standard game-theoretic reasoning (which assumes rational utility
maximization) with *causal* reasoning using structural causal models. This
lets you ask counterfactual questions: "If opponent had a different hand,
would they still have made that discard?" -- and use the answers for
stronger opponent modeling.

### Source
Bareinboim, Forney, Pearl, "Counterfactual Rationality: A Causal Approach
to Game Theory" (Causal AI Lab).
[PDF](https://causalai.net/r125.pdf)

Also: Ibeling et al., "Reasoning about causality in games" (AIJ 2023).
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370223000656)

### Key Formulations

**Structural Causal Model (SCM)**:
```
M = <U, V, F, P(U)>
```
- U: exogenous (hidden) variables
- V: endogenous (observed) variables
- F: structural functions V <- f_V(Pa(V), U_V)
- P(U): distribution over exogenous variables

**Intervention (do-operator)**: `do(X = x)` replaces the structural equation
for X with the constant X <- x. Produces interventional distribution:
```
P_x(Y) = P(Y_x) = sum_{u: Y_x(u)=y} P(u)
```

**Causal Multi-Agent System (CMAS)**:
```
<M, N, X, Y>
```
- N: set of agents
- X = (X_1, ..., X_n): disjoint action nodes (each player controls theirs)
- Y = (Y_1, ..., Y_n): reward nodes
- R_i: D(Y_i) -> R is reward function

**Three layers of causal reasoning in games**:

L1 (Observational): Follow natural mechanism f_{X_i}.
   Action space A_1 = {a_0} (singleton -- just "be yourself")

L2 (Interventional): Hard intervention do(x_i).
   This is standard game theory -- choose actions deliberately.
   Action space A_2 = D(X_i)

L3 (Counterfactual): Function h: D(X_i) -> D(X_i).
   "What would I naturally do, and how should I deviate from that?"
   Natural tendency X_i^-> = f_{X_i}(U_i), executed as X_i <- h(X_i^->)
   Special cases: h(x) = x is L1, constant h is L2.

**Causal Nash Equilibrium (CNE)**: A two-stage concept:

1. Layer Selection Game L^up: Each player chooses which *layer* to reason at.
   ```
   u(A) = NE(Game(A_1, ..., A_n))
   ```

2. CNE: Let s^up be the NE of L^up, and A_i^up = union of supports.
   Then omega^up is CNE if it's a Nash equilibrium of Game(A^up).

**Key theorem**: CNE always exists, and CNE payoff weakly dominates
unilateral layer-switch alternatives.

### Application to Mahjong

The L3 (counterfactual) layer is *precisely* what Mahjong defense needs:

- **L1 thinking**: "My natural play (greedy offense) would discard 3m"
- **L2 thinking**: "I deliberately choose to discard 7z instead" (standard)
- **L3 thinking**: "My natural play *would be* 3m, but given what I know
  about opponent's state, I *transform* my natural choice into 7z"

The counterfactual structure captures:
- **Reading opponents**: "If they had tiles X, would they have discarded Y?"
  This is literally P(Y_x) -- the interventional/counterfactual distribution.
- **Signaling awareness**: "My discard of 3m would *signal* that I don't
  need it -- does that change opponent behavior?" (causal chain)
- **Defense as causal intervention**: Switching to defense is a do-operation
  on your own strategy node.

The SCM framework can model the *entire game* as a causal graph:
Wall -> Draws -> Hands -> Discards -> Melds -> Outcomes, with hidden
confounders (the wall order, opponents' hands).


---

## 6. Information Geometry of Games

### Core Idea
Strategy spaces in games are probability simplices. The *natural* geometry
on these spaces is not Euclidean -- it's the Fisher information metric
(a.k.a. Shahshahani metric). Under this geometry, the replicator equation
(evolution of strategies) is a *gradient flow* of fitness, and KL divergence
is the natural Lyapunov function.

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

This is the *unique* (up to scale) Riemannian metric that is invariant
under sufficient statistics.

**Shahshahani metric** (on R_+^n, restricts to Fisher on simplex):
```
g_{ij}(x) = (||x|| / x_i) * delta_{ij},   ||x|| = sum_i x_i
```

**Replicator equation** (fundamental dynamics of strategy evolution):
```
x_dot_i = x_i * (f_i(x) - f_bar(x))
f_bar(x) = sum_i x_i * f_i(x)   [mean fitness]
```

**KEY THEOREM: Replicator = Shahshahani gradient ascent of fitness**

If f_i = dV/dx_i for some potential V, then the replicator dynamics are
the gradient flow of V with respect to the Shahshahani/Fisher metric:
```
x_hat_i(x) = x_i * (f_i(x) - f_bar(x))   [Shahshahani gradient]
```

**Exponential-map representation** (dual coordinates):
```
x_i = exp(v_i - G)
v_dot_i = f_i(x)
G_dot = f_bar(x)
```

This yields the replicator equation. The `v_i` are "log-odds" coordinates.

**KL divergence as Lyapunov function**:
```
V(x) = D_KL(x_hat || x) = sum_i x_hat_i * log(x_hat_i / x_i)
```

Time derivative along replicator flow:
```
V_dot(x) = -(x_hat . f(x) - x . f(x))
```

If x_hat is an ESS (evolutionarily stable strategy), then V_dot < 0,
meaning KL divergence to the equilibrium monotonically decreases.

**Fisher information variance identity**:
```
Var_p[g] = || (dE[g])_p ||_p^2 = || (grad E[g])_p ||_p^2
```

The variance of fitness equals the squared norm of the fitness gradient
in Fisher geometry. This is the *fundamental theorem of natural selection*
restated in information-geometric language.

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

**KL divergence induced metric** (proving Fisher = KL Hessian):
```
g_{ij}^(D) = d^2 D / (dx_i dy_j) |_{x=y}

For KL: d^2 D_KL(x||y) / (dx_i dy_j) |_{x=y} = (1/x_i) * delta_{ij}
```

This proves Fisher metric is the *infinitesimal* version of KL divergence.

### Application to Mahjong

The information-geometric perspective gives us:

1. **Natural gradient for policy learning**: Instead of vanilla SGD on
   the policy network, use the Fisher information matrix (natural gradient).
   This is parameterization-invariant and converges faster.
   ```
   theta_{t+1} = theta_t - alpha * F^{-1}(theta) * grad L(theta)
   ```
   where F is the Fisher information matrix of the policy.

2. **KL regularization is geometrically natural**: Penalizing KL divergence
   from a reference policy (as in PPO) is not arbitrary -- it's the natural
   distance measure on strategy manifolds.

3. **Strategy dynamics interpretation**: The evolution of Mahjong meta
   (how strategies shift over time in a population) follows replicator
   dynamics on the Fisher manifold. Convergence to equilibrium is measured
   by KL divergence.

4. **Fitness landscape**: The "fitness" of a Mahjong strategy is its
   expected score against the population. The replicator equation predicts
   how strategy frequencies evolve -- useful for training curriculum design.


---

## 7. Free Energy Principle & Active Inference

### Core Idea
Model the Mahjong agent as a system that minimizes *free energy* -- a
quantity that bounds surprise. The agent maintains a generative model of the
game, and both perception (belief updating) and action (tile selection) arise
from a single objective: minimize the divergence between predictions and reality.

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

Free energy is an upper bound on "surprise" = -ln P(observations).
Minimizing F is equivalent to approximate Bayesian inference.

**Mean-field factorization**:
```
Q(s_tilde, pi) approx Q(pi) * prod_tau Q(s_tau | pi)
```

**Free energy decomposition by policy**:
```
F = E_{Q(pi)}[F_pi] + D_KL(Q(pi) || P(pi))
```

where F_pi is the free energy conditioned on policy pi.

**Policy belief update** (softmax of negative free energy):
```
Q(pi) proportional_to exp(ln P(pi) - F_pi)
Q(pi) = softmax(ln P(pi) - F_pi)
```

Policies that predict observations well (low F_pi) get higher probability.

**Expected Free Energy (EFE)** -- the key quantity for *future* decisions:
```
G_pi = E_{Q(o,s|pi)}[ ln Q(s|pi) - ln P(o,s|pi) ]
```

**EFE decomposition into risk and ambiguity**:
```
G_pi = D_KL(Q(o|pi) || P(o))     [risk / pragmatic value]
     + E_{Q(s|pi)}[ H(P(o|s)) ]  [ambiguity / epistemic value]
```

- **Risk**: KL divergence between predicted outcomes and preferred outcomes.
  "Am I likely to get outcomes I want?"
- **Ambiguity**: Expected entropy of observations given states.
  "How uncertain will I be about the state after observing?"

**Combined policy selection** (with expected free energy as prior):
```
Q(pi) = softmax( ln E(pi) - F_pi - G_pi )
```

where E(pi) encodes habitual preferences.

**DR-FREE: Distributionally Robust Free Energy** (handles model uncertainty):

Joint trajectory distribution:
```
p_{0:N} = p_0(x_0) * prod_{k=1}^{N} p_k(x_k | x_{k-1}, u_k) * pi_k(u_k | x_{k-1})
```

The robust objective (minimax over model uncertainty):
```
min_{pi_k} max_{p_k in B_eta(p_bar_k)}
    [ D_KL(p_{0:N} || q_{0:N}) + E_{p_{0:N}}[ sum_k (c_k^x(X_k) + c_k^u(U_k)) ] ]
```

where B_eta is the ambiguity set (KL ball around trained model):
```
B_eta(p_bar_k) = { p_k : D_KL(p_k || p_bar_k) <= eta_k }
```

**Optimal robust policy** (Gibbs/softmax form):
```
pi_k*(u_k | x_{k-1}) proportional_to
    q_k(u_k | x_{k-1}) * exp(- c_k^u(u_k) - eta_k(x_{k-1}, u_k) - c_tilde(x_{k-1}, u_k))
```

### Application to Mahjong

Active inference is *shockingly* well-suited to Mahjong:

1. **Generative model = game engine**: The agent's generative model predicts
   what tiles will appear, what opponents will discard, what outcomes will
   occur. This is exactly what hydra-core simulates.

2. **EFE naturally balances offense and defense**:
   - Risk term: "Does this action lead to preferred outcomes (winning)?"
   - Ambiguity term: "Does this action reduce my uncertainty about
     opponents' hands?"
   - A safe discard might have low risk (opponents unlikely to win from it)
     AND low ambiguity (doesn't reveal much about game state).

3. **Robustness**: DR-FREE handles the fact that the agent's model of
   opponents is *wrong*. The ambiguity set eta controls how paranoid
   the agent is about model misspecification -- larger eta = more defensive.

4. **Unified perception-action loop**: Belief updating (what tiles do
   opponents have?) and action selection (what should I discard?) are
   the SAME optimization: minimize free energy.


---

## 8. Algebraic / Compositional Game Theory

### Core Idea
Instead of defining games monolithically, build them *compositionally*
from small pieces using categorical algebra. Games are morphisms in a
symmetric monoidal category; sequential play is composition, simultaneous
play is tensor product. This lets you decompose a complex game like
Mahjong into reusable building blocks.

### Source
Ghani, Hedges, Winschel, Zahn, "Compositional Game Theory" (LICS 2018).
[arXiv:1603.04641](https://arxiv.org/abs/1603.04641)

Hedges, "Towards Compositional Game Theory" (PhD thesis, Oxford 2016).
[PDF](https://www.cs.ox.ac.uk/people/julian.hedges/papers/Thesis.pdf)

### Key Formulations

**Open game** (type signature): An open game G has type:
```
G : X (x) S* -> Y (x) R*
```

where:
- X: input (information flowing in from environment)
- Y: output (information/actions flowing out)
- R: utility flowing back from environment
- S: utility passed back to earlier stages

An open game is specified by 4 components:

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

**Lens/optic structure**: For fixed strategy sigma, the (play, coplay) pair
forms a lens:
```
(P_G(sigma), C_G(sigma))  :  (X, S) <-> (Y, R)
```

Play goes forward (X -> Y), coplay goes backward (X x R -> S).

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

Strategy sigma is an equilibrium when it's a fixed point of the best
response: no player can improve by unilateral deviation.

**Coutility**: The backward-flowing value:
```
C_G(sigma) : X (x) R -> S
```
Takes current state/history plus future utility R, returns utility S
passed back to earlier game stages.

### Application to Mahjong

Compositional game theory decomposes Mahjong into modular subgames:

**A Mahjong round as composition**:
```
Round = Deal ; (Turn_1 (x) Turn_2 (x) Turn_3 (x) Turn_4)^{*n} ; Score
```

Where:
- Deal: I -> HandState (x) WallState
- Turn_i: GameState -> GameState (one player's draw-discard cycle)
- Score: GameState -> Points (x) Points (x) Points (x) Points

Each Turn is itself composed:
```
Turn = Draw ; Evaluate ; (Discard | Call | Win)
```

**Why this matters**:

1. **Modularity**: You can reason about Turn_i in isolation, then compose.
   A defensive strategy for Turn_i composes seamlessly with an aggressive
   strategy for Turn_j.

2. **Coutility = downstream impact**: The coplay function C propagates
   "how much did this discard cost future-me?" backward through the
   composition. This is analogous to backpropagation!

3. **Equilibrium preservation**: If each sub-game has a Nash equilibrium,
   the composition preserves equilibrium properties (by the main theorem).

4. **Formal verification**: The categorical structure allows proving
   properties about the full game from properties of its parts --
   potentially useful for verifying that an AI respects game rules.


---

## 9. Synthesis: A Unified Framework for Mahjong

### The Big Picture

These 8 frameworks are not independent -- they form a coherent stack:

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

Combine the most promising elements into a single framework:

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
- G(a) = risk(a) + ambiguity(a) is the expected free energy
- g(a) = mutual information gain about opponents from action a
- lambda controls exploration-exploitation balance

This reduces to:
- Pure active inference when lambda = 0
- Pure information-directed when risk is constant
- A hybrid that naturally balances offense (minimize risk) with
  intelligence gathering (maximize info gain)

**Belief update**: After each observation (opponent discard, call, etc.):
```
For each opponent i:
    For each particle j:
        w_j <- P(observation | particle_j)   [likelihood]
    Resample particles proportional to weights
    Apply motion model (drift/switch) to capture strategy shifts
```

**Policy learning** (training time): Use natural gradient (Fisher geometry)
on the policy network, with:
- CFR-style counterfactual regret as training signal
- KL regularization (geometrically natural) against reference policy
- Causal reasoning for opponent modeling: "given opponent discarded X,
  what interventional distribution over their hand is implied?"

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

The existing Mahjong AI landscape (Suphx, Mortal) uses:
- Standard RL (PPO, actor-critic)
- Neural network function approximation
- Hand-crafted features or self-play

What the frameworks above add:

1. **Information-theoretic action selection**: No existing Mahjong AI
   explicitly optimizes the information ratio. Discards are chosen for
   value, not for their information content.

2. **Particle filter opponent tracking**: Current AIs use fixed neural
   networks for opponent modeling. Particle filters adapt in real-time
   to THIS specific opponent's tendencies.

3. **Causal counterfactual reasoning**: "If opponent X had tiles Y,
   would they have discarded Z?" -- no current AI asks this question
   formally. The SCM framework makes it rigorous.

4. **Free energy as unified objective**: Rather than separate value/policy
   heads with ad-hoc loss weighting, free energy minimization gives a
   principled single objective that handles both perception and action.

5. **Natural gradient on strategy manifold**: PPO uses clipping as a
   crude trust region. Natural gradient is the *geometrically correct*
   way to update on probability simplices.

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

6. Bareinboim et al. "Counterfactual Rationality: A Causal Approach to
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
