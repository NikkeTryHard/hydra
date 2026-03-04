# Hydra All-Out Plan: Beat LuckyJ

> **OUTDATED**: This document describes the 22-technique "kitchen sink" approach. It was replaced by the compute-constrained 9-technique design in **HYDRA_FINAL.md** after 5/5 expert judges unanimously preferred the simpler system at our 2000 GPU-hour budget.

## The Core Engine: ExIt + Pondering Virtuous Cycle

Everything else is secondary to this. This is what made AlphaZero dominate Go/Chess/Shogi, and we're applying the exact same principle to imperfect-info Mahjong.

```
FBS search (with pondering) produces better action values than raw policy
  -> ExIt: train policy on search-improved soft targets
    -> Policy improves
      -> FBS is even better (better V_phi for leaf eval)
        -> ExIt targets are even better
          -> (cycle compounds)
```

The twist: PONDERING amplifies the cycle. During training self-play, each agent is idle 75% of the time (3 other players' turns). That idle time runs FBS to produce high-quality ExIt targets. So pondering doesn't just help inference -- it ACCELERATES TRAINING.

Without pondering: FBS runs 1-5ms per decision. D=4, 16 samples. OK targets.
With pondering: FBS runs 5-15 SECONDS per decision. D=8-12, 100-1000 samples. Excellent targets.

This is why our plan can match LuckyJ without matching OLSS's game-theoretic soundness. OLSS is a better SINGLE-SHOT search. But our cycle produces a better POLICY through training, because every self-play game generates thousands of high-quality ExIt targets from pondering.

---

## The Full Stack (16 Techniques, No Complexity Limit)

### TIER 1: Foundation (proven, non-negotiable)

| # | Technique | What It Does | Evidence |
|---|-----------|-------------|---------|
| 1 | 40-block SE-ResNet + nested bottleneck | 1.5x effective depth free | KataGo: bottleneck matches 2x plain |
| 2 | 85x34 encoding + 23 safety channels | Pre-computed suji/kabe/genbutsu | "Highest leverage Mahjong-specific input" |
| 3 | 7 aux heads (policy, value, GRP, tenpai, danger, opp-next, score) | Dense training gradients | KataGo: -190 Elo without aux |
| 4 | 24x data augmentation (6 suit x 4 seat) | Free training data | Standard in Go |
| 5 | Agari guard | Never miss a win | Prevents catastrophic misplay |
| 6 | BC warm-start on 5-6M expert games | Fast initial learning | Standard |

### TIER 2: The Engine (ExIt cycle + beliefs + search)

| # | Technique | What It Does | Evidence |
|---|-----------|-------------|---------|
| 7 | Oracle distillation (Phase 1-2) | Teacher sees all hands, student learns | Suphx: core innovation, 10 dan |
| 8 | ExIt training signal | FBS Q-values as soft policy targets | AlphaZero paradigm, 5x in Hex |
| 9 | SIB + Mixture-SIB (L=8) | Constraint-consistent beliefs with correlations | Novel. NA-backed, KL-projected |
| 10 | FBS search (D=4 on-turn, D=8+ pondered) | Factored belief search | Novel. Conservative by NA |
| 11 | Pondering | 52K searches/round from idle time | +20-66 Elo chess. 260s idle. |

### TIER 3: Training Convergence + Exploitation

| # | Technique | What It Does | Evidence |
|---|-----------|-------------|---------|
| 12 | DRDA-M (or ACH fallback) | Convergent multiplayer training | R-NaD at 1024 TPUs (DeepNash) |
| 13 | POT (30% HPM mix) | Exploit measured human biases | Naga data: +4.4% overcall, -1.5% under-riichi |
| 14 | IVD primal-dual | Info gain + concealment as constrained objectives | 3 direct precedents. MI identities. |
| 15 | Hard position mining | Train more on weaknesses | AlphaStar league exploiters |
| 16 | Progressive model scaling | Start 20-block, upscale at 30% | ~20-40% compute savings |

---

## Implementation Phases

### Phase 0: Foundation (Month 1-2) -- Local RTX 5070
- Build architecture (#1-6)
- BC training with all 7 heads + 24x augmentation
- Validate encoder, training loop, data pipeline
- **Target: ~5-6 dan base**

### Phase 1: Oracle + ExIt Bootstrap (Month 3-4) -- Frontera 2000 SU
- Oracle distillation (#7)
- ExIt with oracle as expert (oracle evaluates top-K, student learns soft targets)
- Train V_phi on oracle-visible true beliefs
- Validate SIB: MAE < 0.3
- **Target: ~8-9 dan (Suphx-level)**

### Phase 2: Search + Pondering + Cycle (Month 5-6) -- ACCESS Explore 6K hrs
- FBS implementation (#10)
- Pondering infrastructure (#11)
- Switch ExIt expert from oracle to FBS+Pondering
- ExIt virtuous cycle begins: FBS -> train -> better FBS -> train -> ...
- Mixture-SIB (#9)
- **Target: ~9-10 dan (cycle accelerating)**

### Phase 3: Full Stack (Month 7-9) -- Remaining compute
- DRDA-M self-play (#12) with POT (#13)
- IVD primal-dual (#14)
- Hard position mining (#15)
- Extended training with full ExIt cycle
- **Target: ~10-11+ dan**

---

## Why This Beats LuckyJ (The Argument)

LuckyJ: ACH (good training) + OLSS (heavy search) = 10.68 dan

Hydra: ExIt cycle (search TEACHES policy) + Pondering (free compute amplifies cycle) + 16 stacked techniques = ?

The key difference: LuckyJ's search (OLSS) only helps at INFERENCE TIME. Our search (FBS+Pondering) helps at BOTH training time (via ExIt) AND inference time. This means our POLICY NETWORK is trained on better data than LuckyJ's, producing a stronger base policy that ALSO gets search at inference.

LuckyJ's raw policy (without OLSS): estimated ~9-9.5 dan
Hydra's raw policy (trained via ExIt cycle): estimated ~10-10.5 dan

Then at inference:
LuckyJ + OLSS: 10.68 dan
Hydra + FBS + Pondering: 10.5-11.5 dan

---

## Confidence: 55-65% with equal compute

After devil's advocacy on every claim:
- The ExIt + Pondering cycle is HARD TO ARGUE AGAINST (it's AlphaZero's principle)
- FBS's conservative bias is a FEATURE for Mahjong (deal-in costs >> fold costs)
- Pondering gives 70-80% hit rate on top-10 predictions
- Efficiency multiplier is realistically 2-3x (not 4-8x)
- Base policy is probably 8.5-9.5 (conservative, not 10+)
- IVD is a gamble with a gate
- FBS vs OLSS in 4-player is arguable (both approximate in 4p)
