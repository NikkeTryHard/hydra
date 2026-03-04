# Non-Convex Placement Games: Multi-Objective RL for Mahjong (D+)

Placement-conditioned multi-objective reinforcement learning for 4-player Riichi Mahjong. Addresses the fundamental limitation that scalar reward functions cannot represent optimal placement strategies in all game states.

---

## Related Documents

- [TRAINING.md](TRAINING.md) -- PPO pipeline, reward function, PID-Lagrangian, CRC risk control
- [REWARD_DESIGN.md](REWARD_DESIGN.md) -- GRP-based reward, RVR variance reduction, placement point vectors
- [HYDRA_SPEC.md](HYDRA_SPEC.md) -- Architecture, GRP head, value head
- [SEARCH_PGOI.md](SEARCH_PGOI.md) -- Inference-time oracle reuse (complementary, independent module)
- [OPPONENT_MODELING.md](OPPONENT_MODELING.md) -- Sinkhorn belief, danger head

---

## 1. Motivation: Why Scalar Rewards Fail

### Hydra's Current Reward Design

Hydra uses a GRP-based scalar reward ([TRAINING.md S Reward Function](TRAINING.md#reward-function)):

```
r_k = E[pts]_after_kyoku_k - E[pts]_before_kyoku_k
pts_vector = [3, 1, -1, -3]  (symmetric placement points)
```

This collapses placement optimization into a single scalar: expected placement points. The PID-Lagrangian ([TRAINING.md S PID-Lagrangian](TRAINING.md#pid-lagrangian-lambda-auto-tuning)) adds a secondary deal-in constraint, and CRC ([TRAINING.md S CRC](TRAINING.md#conformal-risk-control-inference-time-defense-governor)) adds inference-time risk budgeting with Mondrian buckets.

**The fundamental limitation:** All of these are LINEAR combinations or ADDITIVE adjustments of a scalar objective. Linear scalarization can only find solutions on the CONVEX HULL of the Pareto front. If the Pareto front has concave regions, these solutions are invisible to any scalar-reward method.

### The Non-Convexity Claim

**Claim:** In 4-player Riichi Mahjong with placement-based scoring (uma/oka), the Pareto front of (score-maximization, 4th-avoidance) strategies is non-convex. Specifically, there exist game states where the relationship between aggression and safety is non-monotone.

### Constructive Example: South 4, Dealer, 4th Place

Consider this specific game state:
- Round: South 4 (final round or later)
- Seat: East (dealer)
- Position: 4th place, 20,000 points behind 3rd
- Hand: 2-shanten, moderate potential (mangan-class if completed)

Three strategies with different aggression levels:

| Strategy | Aggression | Expected Score Gain | P(escape 4th) |
|-|-|-|-|
| S1: Pure fold | 0.0 | 0 pts | ~0% (guaranteed 4th) |
| S2: Moderate push | 0.5 | +3,000 pts expected | ~8% (might score small, still probably 4th) |
| S3: Maximum push | 1.0 | +1,500 pts expected (high deal-in risk) | ~18% (frequent deal-ins BUT dealer repeats create extra rounds) |

The critical observation: S3 has LOWER expected score than S2 (because of deal-ins) but HIGHER P(escape 4th) than S2. This happens because:

1. As dealer, winning any hand triggers a dealer repeat (renchan)
2. Each extra round is another chance to score
3. The CUMULATIVE probability of eventually scoring enough across multiple rounds exceeds the single-round probability
4. Deal-ins against opponents who are already ahead don't change placement much (you're already 4th)

This creates a non-monotone relationship:
```
Aggression:     0.0 ---- 0.5 ---- 1.0
P(escape 4th):  0%  ---- 8%  ---- 18%   (non-monotone: curves UP at high aggression)
E[score]:       0   ---- +3000 -- +1500  (monotone then dropping: standard risk)
```

The Pareto front in (E[score], P(escape 4th)) space has a concave region between S2 and S3. No linear combination of score and safety objectives can discover S3 — it's Pareto-dominated in EXPECTED SCORE but Pareto-optimal in PLACEMENT.

> **[estimated -- formal proof requires modeling]** The above argument is a proof sketch. A rigorous version requires:
> 1. A Markov chain model of round-to-round score transitions under dealer repeats
> 2. Formal computation of P(escape 4th | aggression level, rounds remaining)
> 3. Demonstration that this function is non-monotone for specific parameter regimes
> 4. The non-monotonicity must survive expectation over opponent strategies and tile draws
>
> **Risk:** The non-monotonicity might vanish when accounting for the counter-effect: dealer repeats also expose the dealer to MORE rounds of potential deal-ins from 3 opponents. The proof must show that the escape-via-scoring effect DOMINATES the exposure-to-danger effect under realistic mahjong parameters.

### Generalization Beyond Mahjong

This non-convexity is structural to ANY placement-based competitive game:

| Domain | Mechanism | Non-Convexity Source |
|-|-|-|
| Riichi Mahjong | Dealer repeats | Aggressive play creates extra rounds |
| Poker tournaments (ICM) | All-in dynamics | Doubling stack gives non-linear ICM equity gain |
| Formula 1 points | Risk/reward in final laps | Overtake attempts: moderate risk = crash, high risk = pass |
| Academic competitions | Score thresholds | Attempting harder problems: medium difficulty = partial credit, hard = full points or zero |

A result proving non-convexity for ONE of these domains implies it for the general class of placement games.

---

## 2. Integration with Hydra's Existing Pipeline

### What Already Exists (Don't Reinvent)

Hydra already has primitive multi-objective machinery:

| Component | Location | What It Does | MORL Relationship |
|-|-|-|-|
| PID-Lagrangian | [TRAINING.md S 460](TRAINING.md) | Auto-tunes offense/defense lambda | Becomes ONE axis of the preference vector |
| CRC Mondrian buckets | [TRAINING.md S 519](TRAINING.md) | State-dependent risk budgets at inference | Provides the state-dependent preference SELECTION (meta-selector ground truth) |
| GRP placement points | [TRAINING.md S 304](TRAINING.md) | Configurable `[3, 1, -1, -3]` vector | Becomes the REWARD DECOMPOSITION into per-placement objectives |
| Combined advantage | [TRAINING.md S 492](TRAINING.md) | `A_combined = (A^R - lambda * A^C) / (1 + lambda)` | Becomes Chebyshev or weighted-sum scalarization |

**Key insight:** Hydra's PID-Lagrangian is already a TWO-objective system (reward vs deal-in cost). MORL generalizes this to THREE+ objectives with a learned preference selector instead of PID control.

### Objective Decomposition

Decompose Hydra's scalar reward into 3 independent objectives:

| Objective | Symbol | Reward Signal | Source |
|-|-|-|-|
| Score gain | O1 | Raw point delta per kyoku | Game outcome |
| 4th avoidance | O2 | -1 if final placement is 4th, 0 otherwise | Game outcome |
| Placement optimization | O3 | GRP delta E[pts] (existing) | GRP model |

The preference vector `w = [w1, w2, w3]` lives on the 3-simplex (w1 + w2 + w3 = 1, all non-negative).

### Preference Conditioning Architecture

The preference vector `w` is injected via **concatenation at the head inputs** (following PD-MORL). The shared backbone is preference-agnostic -- only the heads see `w`.

```
Shared Backbone (40 SE-ResBlocks, unchanged):
  Input: [B x 85 x 34] -> GAP -> [B x 256] backbone embedding

Policy Head (preference-conditioned):
  Input: concat([B x 256] backbone, [B x 3] preference w) -> [B x 259]
  FC(259, 256) -> ReLU -> FC(256, 46) -> softmax

Value Head (multi-objective, preference-conditioned):
  Input: concat([B x 256] backbone, [B x 3] preference w) -> [B x 259]
  FC(259, 256) -> ReLU -> FC(256, 3) -> 3 scalar values (one per objective)
```

**Why concatenation, not FiLM:** FiLM modulates every ResBlock (40x overhead). Concatenation at heads only adds 3 input dims. The backbone learns general mahjong features; heads learn to weight them per preference. PD-MORL found late-stage conditioning sufficient.

> **Architectural cost:** +3 input dimensions to policy and value heads. Zero backbone changes. Zero inference overhead when preference is fixed.

### Training Loop Changes

**Phase 3 only.** Phases 1-2 are unchanged. MORL is a Phase 3 modification.

Per training episode (per kyoku):
1. **Sample preference:** `w ~ Dirichlet(alpha=1)` on the 3-simplex (uniform random preference)
2. **Condition policy:** Feed `w` to policy and value heads alongside backbone output
3. **Collect trajectory:** Play kyoku with preference-conditioned policy
4. **Compute multi-objective returns:** `R = [R_O1, R_O2, R_O3]` (3-vector of objective returns)
5. **Scalarize for PPO:** Use Chebyshev scalarization (handles non-convex regions):

```
scalarized_return = min_i(w_i * (R_i - z_i*)) + rho * sum(w_i * (R_i - z_i*))
```

Where `z*` is the ideal point (best known value per objective), and `rho = 0.1` is a small augmentation term that ensures unique solutions. Chebyshev scalarization can find ALL Pareto-optimal points including unsupported ones.

6. **PPO update:** Standard PPO with scalarized_return as the reward signal

**What replaces the PID-Lagrangian (MORL variant only):** In the MORL Phase 3 variant, the PID auto-tuning of lambda is replaced by the preference vector `w`. The defense/offense tradeoff becomes one dimension of `w` instead of a PID-controlled scalar. **Standard Phase 3 training uses PID as specified in [TRAINING.md](TRAINING.md).** MORL is a Phase 3 alternative, not the default.

> **Compatibility note:** Phases 1-2 use the existing scalar pipeline (no MORL). Phase 3 starts from the Phase 2 checkpoint and adds preference conditioning. The Phase 2 model is equivalent to the MORL model with a fixed preference `w = [0.33, 0.33, 0.33]` (balanced).

### Meta-Selector (Inference-Time Preference Picker)

At game time, a small network maps game state to the optimal preference vector.

**Architecture:** MLP taking score context as input:
```
Input: [round, honba, seat_wind, score_self/10000, score_opp1/10000,
        score_opp2/10000, score_opp3/10000, placement_rank]  -> [8]
FC(8, 64) -> ReLU -> FC(64, 3) -> softmax -> w*
```

**Training:** Supervised on the Pareto front. After MORL Phase 3 training, evaluate every preference `w` in a grid (e.g., 100 points on the simplex) across 10K games. For each game state encountered, record which `w` achieved the best final placement. Train the meta-selector to predict this optimal `w*` from score context.

**Bootstrapping from CRC Mondrian buckets:** The CRC's existing Mondrian buckets ([TRAINING.md S 533](TRAINING.md)) provide a coarse initial mapping:

| CRC Bucket | Approximate w* |
|-|-|
| Leading (1st, gap >= 8000) | [0.1, 0.1, 0.8] (protect placement) |
| Comfortable (1st/2nd, big gap) | [0.2, 0.2, 0.6] |
| Middle (typical) | [0.33, 0.33, 0.33] (balanced) |
| Trailing (3rd/4th, big gap) | [0.6, 0.2, 0.2] (score-heavy) |
| Desperate (4th, South 3+) | [0.5, 0.4, 0.1] (push for points, avoid 4th) |

The meta-selector REFINES these coarse buckets into a continuous mapping. The CRC buckets serve as initialization / sanity check.

---

## 3. Risks and Open Questions

| Risk | Severity | Mitigation |
|-|-|-|
| Preference space is trivially small (w* nearly constant) | HIGH | Run cheap validation first: train 3 fixed-preference models, check if their policies diverge meaningfully |
| Non-convexity proof doesn't survive formal scrutiny | MEDIUM | Use constructive two-point counterexample + Jensen's inequality on dealer repeat function g(p) = p/(1-p). Cite White 1982, Wakuta 1999, Roijers 2013. |
| Chebyshev scalarization is unstable with PPO | MEDIUM | Fall back to constraint-based MORL (RCPO-style) if Chebyshev oscillates. Hydra's existing PID-Lagrangian is already constraint-based. |
| Meta-selector learns the wrong preferences | MEDIUM | Bootstrap from CRC Mondrian buckets. Validate: meta-selector output should correlate with CRC bucket assignments. |
| MORL slows convergence vs scalar RL | LOW | Expected ~1.5x training cost (sampling over preference simplex). Acceptable given Phase 3 is already the longest phase. |
| Dealer repeat non-monotonicity vanishes under realistic opponent play | MEDIUM | The proof must model opponent response to aggressive dealer play. If 3 opponents target the dealer, extra rounds may be net-negative. Empirical validation required. |

---

## 4. Implementation Plan

### Prerequisites

| Prerequisite | Status | Notes |
|-|-|-|
| Phase 2 complete | Required | MORL is a Phase 3 modification |
| Scalar Phase 3 baseline working | Recommended | Need comparison point |
| GRP reward model trained | Required | Provides O3 signal |

### Steps

| Step | Effort | Description |
|-|-|-|
| 1. Cheap validation | 3-5 days | Train 3 fixed-preference models (aggressive/balanced/defensive). If policies diverge, MORL is justified. If identical, stop. |
| 2. Multi-objective reward decomposition | 2-3 days | Split scalar reward into [O1, O2, O3] vector. Log per-objective returns. |
| 3. Preference-conditioned heads | 2-3 days | Add 3-dim concat to policy and value head inputs. |
| 4. Chebyshev scalarization | 2-3 days | Replace scalar PPO reward with Chebyshev-scalarized return. |
| 5. Preference sampling | 1-2 days | Dirichlet sampling in training loop. |
| 6. Full MORL Phase 3 training | 2-4 weeks | Train on 4-8 A100s with diverse preferences. |
| 7. Meta-selector training | 1 week | Supervised on Pareto-front evaluation data. |
| 8. Evaluation | 1-2 weeks | A/B test MORL vs scalar baseline across 10K+ games. |
| **Total** | **~6-8 weeks** | After scalar Phase 3 baseline exists. |

### Ablation Plan

| Experiment | Tests | Success Criterion |
|-|-|-|
| M1: Fixed preferences | 3 models (agg/bal/def) | Policies diverge meaningfully in same game states |
| M2: Preference diversity | Dirichlet(0.5) vs Dirichlet(1.0) vs uniform grid | Higher diversity -> better Pareto coverage |
| M3: Chebyshev vs linear | Same training, different scalarization | Chebyshev finds strategies linear misses |
| M4: Meta-selector vs CRC buckets | Learned w* vs CRC-derived fixed w | Meta-selector outperforms coarse buckets |
| M5: MORL vs scalar + CRC | Full comparison | MORL improves 4th-avoidance in desperate states |

---

## 5. Theoretical Foundation (Proof Sketch)

### Formal Setup

Model the hanchan as a multi-objective MDP (MOMDP). State: `(score_vector, round, dealer, honba, ...)`. Actions: discard/call/riichi. Objectives: `f(pi) = (E[O1], E[O2], E[O3])`.

### Non-Convexity Proof (Two-Point Counterexample)

**Step 1:** Invoke Wakuta (1999) -- the value set of deterministic stationary policies in a vector-valued MDP is discrete and generically non-convex.

**Step 2:** Construct two policies:
- `pi_A` (dealer aggression): pushes every hand. Exploits renchan compounding via g(p) = p/(1-p).
- `pi_B` (consistent defense): folds to all threats.

**Step 3:** Show `0.5*V(pi_A) + 0.5*V(pi_B)` is dominated by `pi_C` (state-dependent switching). Jensen's inequality on the convex function g(p) = p/(1-p) gives pure strategies a premium over mixing.

**Step 4:** Show no linear weight `w > 0` makes `pi_C` a scalarized maximizer, establishing the unsupported Pareto point.

### Key Tools

| Tool | Source | Role |
|-|-|-|
| CCS vs PCS | Roijers et al. 2013 | When linear scalarization is insufficient |
| Value space geometry | White 1982, Wakuta 1999 | Generic non-convexity of deterministic policies |
| Jensen on g(p) = p/(1-p) | Direct calculation | Pure strategies outperform mixtures for dealer bonus |
| Multi-surface structure | NeurIPS 2023 | Extends to parameterized (neural) policies |
| ICM analogy | Malmuth-Harville | Placement equity non-linearity in poker |

---

## 6. References

| Reference | Relevance |
|-|-|
| Roijers et al., "Survey of Multi-Objective Sequential Decision-Making", JAIR 2013 | CCS/PCS theory, supported vs unsupported Pareto points |
| White, "Vector-Valued Dynamic Programming", SIAM 1982 | Value space geometry in MOMDPs |
| Wakuta, "Structure of Value Spaces in Vector-Valued MDPs", 1999 | Deterministic policy non-convexity |
| Xin et al., "Revisiting Scalarization in Multi-Task Learning", NeurIPS 2023 | Parameterized non-convexity |
| Lu et al., "IPRO: Provably Unveiling the Pareto Front", 2024 | Non-linear scalarization algorithm |
| Xu et al., "PD-MORL", ICLR 2023 | Preference conditioning architecture |
| Malmuth & Harville, ICM Model | Poker placement equity non-linearity |
| Stooke et al., "PID-Lagrangian for Constrained MDPs", ICML 2020 | Hydra's existing constraint framework |
| IJCAI 2024 Mahjong AI Competition Survey | Open problem: multi-objective strategy in mahjong |
| Li et al., GenBR, IJCAI 2025 | Learned belief for imperfect-info games |
| MORL-Baselines library (LucasAlegre/morl-baselines) | Reference implementation for MO-PPO |