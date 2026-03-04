# ACH (Actor-Critic Hedge) -- Research Report for Hydra

**Date**: 2026-03-02
**Paper**: "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game" (ICLR 2022)
**Authors**: Tencent AI Lab (Weiming Liu et al.)
**OpenReview**: https://openreview.net/forum?id=DTXZqTNV5nW
**arXiv**: https://arxiv.org/abs/2201.12113

---

## TL;DR

ACH is a **drop-in replacement for the PPO policy gradient loss** (~20 lines of code).
The training infrastructure (actors, learners, GAE, value function, entropy) stays identical.
It provides CFR-style Nash convergence guarantees for 2-player zero-sum games.

---

## 1. What is ACH?

ACH is a practical implementation of **NW-CFR (Neural-based Weighted CFR)**.
It bridges deep RL (actor-critic) and game theory (Counterfactual Regret Minimization).

**Core idea**: Instead of training a network to maximize returns (PPO),
train it to track **cumulative counterfactual regret**, then derive the
policy via the **Hedge algorithm** (softmax over regrets).

The Hedge algorithm (Multiplicative Weights Update) converts cumulative regrets
into a policy:

```
pi_{t+1}(a|s) = exp(eta * R_t(s,a)) / sum_a' exp(eta * R_t(s,a'))
```

This is literally softmax with learning rate eta. The network y(a|s;theta)
approximates R_t(s,a) -- the cumulative advantage sum across iterations.

---

## 2. ACH vs PPO -- Side by Side

| Aspect | PPO | ACH |
|--------|-----|-----|
| **Objective** | Maximize discounted returns | Minimize weighted cumulative counterfactual regret |
| **Network output** | Log-probability of actions | Cumulative regret y(a\|s;theta) |
| **Policy derivation** | Direct softmax of logits | Hedge: pi(a\|s) = softmax(eta * y(a\|s;theta)) |
| **Loss gradient term** | -log_prob * advantage (or ratio-clipped) | -c * eta * y(a\|s) / pi_old(a\|s) * advantage |
| **Clipping** | Ratio clipping only | Ratio clipping AND logit threshold clipping |
| **Convergence** | No Nash guarantee in multiplayer | Nash Equilibrium (2p zero-sum, O(T^-1/2)) |
| **Infrastructure** | Actor-critic + GAE | Identical actor-critic + GAE |
| **Entropy** | Optional regularization | Mandatory (required for convergence) |

The key structural difference: PPO uses `log_prob(a)` in its loss,
ACH uses the raw **centered logit** `y(a) - mean(y)` divided by `pi_old(a)`.

---

## 3. The ACH Loss Function

### Paper Formula (Equation 29)

```
L_ACH = -c * eta(s) * [y(a|s;theta) / pi_old(a|s)] * A(a,s)
        + (alpha/2) * [V(s;omega) - G]^2
        + beta * sum_a pi(a|s;theta) * log(pi(a|s;theta))
```

### Clipping Mask c

```
if A(s,a) >= 0:
    c = 1{ratio < 1+eps} * 1{y(a|s) - mean(y) < logit_threshold}
if A(s,a) < 0:
    c = 1{ratio > 1-eps} * 1{y(a|s) - mean(y) > -logit_threshold}
```

The clipping mask combines TWO safety mechanisms:
1. **Ratio clipping** (same as PPO, prevents too-large policy updates)
2. **Logit thresholding** (NEW -- prevents any single regret/logit from dominating)

### Rust Pseudocode for Hydra

```rust
fn ach_policy_loss(
    old_logits: &Tensor,    // logits from behavior policy
    new_logits: &Tensor,    // logits from current policy
    actions: &Tensor,       // sampled actions
    advantages: &Tensor,    // GAE advantages
    clip_coef: f32,         // e.g., 0.5
    logit_threshold: f32,   // e.g., 6.0
) -> Tensor {
    let old_probs = softmax(old_logits);
    let old_prob_a = gather(old_probs, actions).clamp_min(0.001);

    let new_logit_a = gather(new_logits, actions);
    let mean_logit = mean_over_legal(new_logits);
    let centered_logit = new_logit_a - mean_logit;

    let new_probs = softmax(new_logits);
    let ratio = gather(new_probs, actions) / old_prob_a;

    // Combined clipping mask
    let c = where(
        advantages >= 0.0,
        (ratio < 1.0 + clip_coef) & (centered_logit < logit_threshold),
        (ratio > 1.0 - clip_coef) & (centered_logit > -logit_threshold),
    ).detach().to_float();

    // ACH policy gradient loss
    -c * centered_logit / old_prob_a * advantages.detach()
}

// Total loss (same structure as PPO):
// loss = ach_policy_loss + vf_coef * value_loss - ent_coef * entropy
```

---

## 4. Algorithm Pseudocode (from paper)

### Algorithm 1: NW-CFR (Theoretical Foundation)

```
function NW-CFR(player p, iterations T, episodes M, theta, omega):
    for t = 1 to T:
        # Derive policy from regret network via Hedge
        pi_t = Softmax(eta(s) * y(a|s; theta_{t-1}))

        # Train value/Q network
        Collect M self-play trajectories with (pi_p, pi_{-p})
        Train omega on MSE loss: E[(Q(s,a;omega) - G)^2]

        # Train regret/policy network
        Collect M trajectories with exploration policy
        Estimate advantages A(s,a) for ALL actions using Q network
        Sum-aggregate advantages by state
        Train theta on policy loss L_pi(s) (Eq 2 in paper)
```

### Algorithm 2: ACH (Practical Implementation)

```
Initialize policy params theta, critic params omega
Start multiple actor threads and learner threads

ACTORS (parallel):
    loop:
        Fetch latest model from learners
        Generate self-play samples: [action, state, A(s,a), G, pi_old(a|s)]
        Push samples to replay buffer

LEARNERS:
    for t = 1, 2, 3, ...:
        Fetch mini-batch from replay buffer
        L_sum = 0
        for each sample [a, s, A(s,a), G, pi_old(a|s)]:
            Compute clipping mask c (ratio clip + logit threshold)
            L_sum += -c*eta * y(a|s;theta)/pi_old * A(a,s)    // policy loss
                     + alpha/2 * [V(s;omega) - G]^2            // value loss
                     + beta * sum_a pi(a|s) * log(pi(a|s))     // entropy
        Update theta, omega with gradient of L_sum
```

This is structurally identical to PPO/IMPALA actor-learner loops.
The ONLY difference is the loss function in the learner.

---

## 5. Infrastructure Requirements

**No special infrastructure beyond standard PPO/IMPALA.**

Evidence from ygo-agent (cleanba.py L837-864):
The entire training loop is standard PPO. ACH is activated by a single flag:

```python
if args.logits_threshold is not None:    # ACH mode
    pg_loss = ach_loss(actions, logits, new_logits, advantages,
                       args.logits_threshold, args.clip_coef)
elif args.ppo_clip:                      # PPO mode
    pg_loss = clipped_surrogate_pg_loss(ratios, advantages, args.clip_coef)
```

Source: https://github.com/sbl1996/ygo-agent/blob/dbf5142/scripts/cleanba.py#L837-L839

What stays identical:
- Actor-learner architecture (IMPALA/cleanba)
- GAE advantage estimation (lambda=0.95)
- Value function training (MSE loss)
- Entropy bonus computation
- Trajectory collection / replay buffers
- Self-play framework

What changes:
- Policy gradient loss function (~20 lines)
- One new hyperparameter: logits_threshold

---

## 6. Convergence Guarantees

### Theorem 1 (from paper, Page 5)

For 2-player zero-sum games, with appropriate eta:

```
eta(s) = sqrt(8 * ln|A(s)| / (w_h^2 * Delta^2 * T))
```

The average policy has exploitability epsilon after T iterations:

```
epsilon <= |S| * Delta * sqrt(ln|A| / (2T))
           + Delta * sum_s (w_h(s) - w_l(s)) / w_h(s)
```

- **First term**: Converges at O(T^{-1/2}) -- standard CFR rate
- **Second term**: Constant error from weight (reach probability) variation

### Caveats for 4-Player Mahjong

The convergence proof is for **2-player zero-sum** only.
4-player Riichi Mahjong is NOT 2-player zero-sum.

LuckyJ likely handles this by:
1. Training in a 1v3 framework (treating 3 opponents as "the environment")
2. Or accepting the theoretical gap and relying on empirical convergence
3. The paper's own experiments were on 1-on-1 Mahjong (2-player reduction)

For Hydra: this means ACH gives you the same theoretical standing as LuckyJ
(converges in 2p zero-sum, empirically strong in 4p), which is strictly
better than PPO (no convergence guarantee at all in multi-agent settings).

---

## 7. Hyperparameters (from Paper's Mahjong Experiments)

Table 5, Page 26 of the paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| eta (Hedge LR) | 1.0 | Can be absorbed into logit scale |
| logit_threshold | 6.0 | Prevents logit explosion |
| clip_coef (epsilon) | 0.5 | Wider than PPO's typical 0.2 |
| entropy_coef (beta) | 0.01 | Mandatory, not optional |
| value_coef (alpha) | 0.5 | Same as PPO |
| GAE lambda | 0.95 | Same as PPO |
| learning_rate | 2.5e-4 | Same range as PPO |
| batch_size | 8192 | Same range as PPO |
| gamma | 0.995 | Slightly higher than typical 0.99 |

From the official poker implementation (ach.sh):

| Parameter | Value |
|-----------|-------|
| ach_alpha | 2 |
| ach_beta | 0.03 |
| ach_thres | 2 |
| ach_reward_scale | 0.002 |

Source: https://github.com/Liuweiming/ACH_poker/blob/2f8613f/ach.sh

---

## 8. Existing Implementations

### Official (Tencent)
- **ACH_poker** (C++/OpenSpiel): https://github.com/Liuweiming/ACH_poker
  - Commit: 2f8613f73749117ee2156aa549773e2fbffca98e
  - Full external-sampling MCCFR solver with neural function approximation
  - Uses TensorFlow 1.x, OpenSpiel framework

### Third-Party
- **ygo-agent** (JAX/Python): https://github.com/sbl1996/ygo-agent
  - Commit: dbf5142d49aab2e6beb4150788d4fffec39ae3e5
  - Clean ~20-line ACH loss implementation in JAX
  - Used for Yu-Gi-Oh! card game AI (another imperfect-info game)
  - Shows ACH works as drop-in loss swap in standard PPO loop
  - **Best reference for Hydra's implementation**

- **CR-PPO** (hnsqdtt, deleted/private)
  - Was described as "Counterfactual Regret-Weighted PPO"
  - Adapted ACH for personal computing resources
  - Confirms the PPO-modification approach works

---

## 9. Implementation Plan for Hydra

### What to Change

In Hydra's training pipeline (hydra-train), the ONLY change needed is
replacing the PPO clipped surrogate loss with the ACH loss.

```
// BEFORE (PPO):
loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)

// AFTER (ACH):
centered_logit = new_logit[a] - mean(new_logits)
c = match adv >= 0 {
    true  => (ratio < 1+eps) && (centered_logit < threshold),
    false => (ratio > 1-eps) && (centered_logit > -threshold),
};
loss = -c * centered_logit / old_prob[a] * adv
```

### What Stays the Same

- SE-ResNet architecture (40 blocks, 256 channels)
- 85x34 input encoding
- 46-action output head
- Value head, GRP head, Tenpai head, Danger head
- GroupNorm(32), Mish activation
- GAE advantage estimation
- Self-play trajectory collection
- Replay buffers
- Actor-learner parallelism

### Recommended Approach

1. Implement ACH loss as an **alternative** to PPO loss (flag-switchable)
2. Use ygo-agent's implementation as reference (cleanest code)
3. Start with paper's mahjong hyperparameters (eta=1, threshold=6, clip=0.5)
4. Entropy coefficient is MANDATORY -- ensure it's never set to 0
5. Can A/B test ACH vs PPO with identical infrastructure

### Risk Assessment

- **Low risk**: ACH is well-understood (ICLR 2022, multiple implementations)
- **Low effort**: ~20 lines of loss function code
- **High reward**: Closes the "PPO lacks Nash convergence" criticism
- **Caveat**: 4-player convergence is empirical, not proven (same as LuckyJ)

---

## 10. Key Evidence Links

| What | Link |
|------|------|
| Paper (OpenReview) | https://openreview.net/forum?id=DTXZqTNV5nW |
| Paper PDF | https://openreview.net/pdf?id=DTXZqTNV5nW |
| ICLR 2022 Slides | https://iclr.cc/media/iclr-2022/Slides/6627.pdf |
| Official poker code | https://github.com/Liuweiming/ACH_poker |
| ACH solver (C++) | https://github.com/Liuweiming/ACH_poker/blob/2f8613f/algorithms/ach_solver.cc |
| ygo-agent ACH loss (JAX) | https://github.com/sbl1996/ygo-agent/blob/dbf5142/ygoai/rl/jax/__init__.py#L99-L118 |
| ygo-agent training loop | https://github.com/sbl1996/ygo-agent/blob/dbf5142/scripts/cleanba.py#L837-L864 |
