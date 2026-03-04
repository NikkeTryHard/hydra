# Math & Code Verification Prompt

## YOUR ROLE
You are a mathematical verification agent AND a systems programmer. You verify proofs, check equations, and assess whether designs are implementable in code. You are adversarial -- you WANT to find errors.

## TASK
Read HYDRA-OMEGA: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md

Verify EVERY mathematical claim, equation, and compute estimate. For each, state: CORRECT, WRONG (with fix), or UNVERIFIABLE.

## REFERENCES
- ACH paper: https://openreview.net/pdf?id=DTXZqTNV5nW
- OLSS paper: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- Suphx: https://arxiv.org/pdf/2003.13590
- KataGo: https://arxiv.org/pdf/1902.10565
- DRDA: https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf
- PPO in IIGs: https://arxiv.org/pdf/2502.08938
- Sinkhorn: https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
- SR/BBL: https://www.ams.org/jams/2009-22-02/S0894-0347-08-00618-8/
- Concentration w/o replacement: https://projecteuclid.org/journals/bernoulli/volume-21/issue-3/Concentration-inequalities-for-sampling-without-replacement/10.3150/14-BEJ605.pdf
- Hunter bound: https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/an-upper-bound-for-the-probability-of-a-union/092D711504BA968EF0D1D903A2685D60
- Board game scaling: https://arxiv.org/pdf/2104.03113
- Chinchilla: https://arxiv.org/pdf/2203.15556
- GAE: https://arxiv.org/pdf/1506.02438
- IMPALA: https://arxiv.org/pdf/1802.01561
- Mortal source: https://github.com/Equim-chan/Mortal
- Burn framework: https://burn.dev/docs/burn/
- Our codebase: https://github.com/NikkeTryHard/hydra

---

## VERIFY THESE SPECIFIC CLAIMS

### V1: Samples-per-parameter proof (Section 4.2)
We claim: 2000 GPU hours -> ~3.15B decisions -> 40-block (16.5M params) gets 191 samples/param = 0.37x Mortal's 514. Therefore undertrained.
- Is the 3.15B decision count correct? (25 games/sec * 3600 * 500 wall-hrs * 70 decisions/game)
- Is Mortal's ~514 samples/param estimate reasonable? (80M games * 70 / 10.9M params)
- Is 0.37x actually "undertrained"? What's the minimum viable ratio in RL?
- Does 6x suit augmentation count as real data or inflated? (we count raw, not augmented)

### V2: CT-SMC 3D DP (Section 5.5)
We claim: c_W = R_k - (c1+c2+c3), so state is 3D with 2,744 states max, 3.3M ops total.
- Verify: is c_W truly derivable? Are there edge cases where this identity breaks?
- Verify: (14)^3 = 2,744 assumes max opponent hand = 13. Can opponents have 14 tiles? (after kan)
- Verify: 34 * 2744 * 35 = 3,265,960. Is this right?
- Can the DP be done in log-space without losing the ability to backward-sample?

### V3: ACH loss function (Section 11, Phase 2)
We reproduce ACH Eq.29 as: L = -c * eta * y(a)/pi_old(a) * A(s,a)
- Verify against the actual paper (ICLR 2022, Eq.29). Is our reproduction exact?
- The gate c has 4 conditions. Write them out explicitly and verify each.
- We claim eta is a global scalar. The paper's theory says eta(s) is state-dependent. Is global eta a valid simplification? What do we lose?
- We claim "one epoch per batch." Cite the exact line in Algorithm 2 that confirms this.

### V4: Endgame multiset DP (Section 7.5)
We claim: wall=10, all distinct -> 2^10 = 1024 multiset states.
- Verify: the DP state is the remaining multiset. With k distinct tile types each having n_k copies, submultiset count = prod(n_k + 1). For all distinct (n_k=1): 2^10 = 1024. Correct?
- What if tiles repeat? e.g., 2 copies of 5m in wall. Then that type contributes (2+1)=3 states instead of 2. Total could be > 1024?
- With opponent actions interleaved: the DP state must also include game state (whose turn, riichi status, etc.). How much does this expand the state space?

### V5: Correlation formula (Section 5.4)
We claim: |rho_ij| = K_i * K_j / sqrt((H-K_i)*(H-K_j))
- Verify against the exact multivariate hypergeometric correlation formula.
- The standard formula is: Cov(X_i,X_j) = -n*K_i*K_j*(N-n)/(N^2*(N-1)). How does this relate to our rho formula?
- Plug in H=50, K=4: does |rho| = 0.087?

### V6: Compute estimates (Section 11 budget table)
We claim: Phase 2 (750 GPU-hrs) produces ~20M games.
- At 25 games/sec on 1 GPU: 750 * 3600 * 25 / 4 GPUs... wait, only 1 GPU does self-play. So 750 GPU-hrs = 187.5 wall-hrs. 187.5 * 3600 * 25 = 16.9M games. Is 20M correct?
- Phase 3 claims ~15M games but has pondering overhead. What's the realistic throughput?
- Total: 50+200+750+1000 = 2000 GPU-hrs. But is there overhead for distillation? Does distillation eat into the budget?

### V7: Sinkhorn KL projection (Section 5.1)
We claim: SIB = argmin KL(B||K) subject to B in U(r,s). Solution: B* = diag(u)*K*diag(v).
- Verify this is the correct Sinkhorn-Knopp formulation. Some formulations minimize KL(K||B), not KL(B||K). Which is ours and does it matter?
- Is convergence guaranteed for all K > 0? What about numerical issues when K has very small entries?

### V8: Hunter bound (Section 6.2)
We claim: P(union A_j) <= sum P(A_j) - sum_{(u,v) in T} P(A_u cap A_v) for spanning tree T.
- Verify this is Hunter's bound (not a different inequality).
- Is the maximum-weight spanning tree truly optimal for tightness?
- Can we compute this in real-time during a Mahjong decision? How many events J typically?

### V9: Robust opponent modeling (Section 8)
We claim: V_rob = min_{q in KL-ball} sum q(a)*Q(a), solution q_tau(a) propto p(a)*exp(-Q(a)/tau).
- Verify this is the correct dual form. Derive it from the Lagrangian.
- Is there a closed-form for tau given epsilon? Or must we binary-search?
- What happens when epsilon=0 (no robustness)? Does it reduce to expectation under p?

### V10: Code implementability
- Can ACH's gated loss be implemented in Burn's autodiff? The hard gate c is non-differentiable. Does this break backprop? (Hint: c gates the loss, not a network layer. Gradient flows through y and pi, not through c.)
- Can CT-SMC's log-space DP be done with Rust's f64? Or do we need arbitrary precision?
- Can the 3-tier distillation work with Burn's model serialization? Teacher and learner have different architectures.

---

## OUTPUT FORMAT
For each V1-V10:
1. **Verdict**: CORRECT / WRONG / PARTIALLY CORRECT
2. **Details**: Show your work. If wrong, give the fix.
3. **Severity**: If wrong, how bad? (cosmetic / affects estimates / breaks the design)
4. **Confidence**: low/medium/high
