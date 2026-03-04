# Proposal Evolution Log

This documents the iterative development of Hydra's novel approach proposal.
The definitive version is `HYDRA_FINAL.md` in the parent directory.

## Iteration History

| Version | Name | Core Idea | Judge Avg | Outcome |
|---------|------|-----------|-----------|---------|
| v1 | BFN (Bayesian Flow Networks) | 5 components: SIB + IVD + BGSS + HPM + R-NaD | 19% (match LuckyJ) | "Kitchen sink" -- too many unvalidated components |
| v2 | POT (Population-Optimal Training) | Exploitation-focused: train vs human population | 16% | Suphx counter-evidence: human data didn't help Suphx beat LuckyJ |
| v3 | FBS (Factored Belief Search) | Novel search algorithm via mean-field factorization | 14% | "Mean-field is known technique" but best focused idea |
| v4 | Combined | Best of v1-v3: SIB + POT + FBS + DRDA-M | 28% | Highest rated; R3-5 holistic judge recommended this combination |
| v5 | HYDRA_PROPOSAL | v4 evolved for H2H judging with IVD + Pondering added | 77-83% (vs frontier) | Consistent wins on all axes except math rigor |
| v6 | HYDRA_FINAL | Hardened math (5 formal results), polished standalone | 38/50 avg (vs ReBeL) | Unanimous 5/5 wins against the most novel approach in game AI |

## Key Pivot Points

1. **v1->v2**: Judges said "too many components." Pivoted to single-contribution exploitation focus.
2. **v2->v3**: Judges said "not novel enough." Pivoted to novel ALGORITHM (search).
3. **v3->v4**: Holistic judge said "combine the best of each." Returned to multi-component but curated.
4. **v4->v5**: Added IVD (most praised component from v1) and Pondering (new). Research-backed every claim.
5. **v5->v6**: Hardened math after judges consistently scored rigor at 6/10. Added 5 formal results.

## Unique Details from Earlier Versions (Preserved Here)

### Belief Divergence Safeguards (from v1)

Three mechanisms to prevent SIB from diverging:

1. **Periodic reset**: Every N=10 turns, run full Sinkhorn prediction from backbone. If L1 distance from incremental B_t exceeds 0.5, REPLACE B_t. Cost: one extra forward pass every 10 turns.
2. **Entropy floor**: Clamp B_t[k,m] >= epsilon/M (epsilon=0.01). Prevents beliefs from reaching 0 or 1.
3. **Phase 1 validation gate**: MAE < 0.3 on held-out games with known hands.

### Search Comparison Table (from v3)

| Algorithm | Fusion-Free? | Multiplayer? | Cost | Game-Theoretic? |
|-----------|-------------|-------------|------|----------------|
| OLSS | Yes | Limited (2p tested) | O(exp) | Yes (Nash) |
| PIMC | No (fusion) | Yes | O(N*A^D) | No |
| ISMCTS | Partial | Yes | O(N*sims) | No |
| ReBeL | Yes | 2p only | O(|hands|^2) | Yes |
| Student of Games | Yes | Untested | O(large) | Yes |
| FBS (ours) | Yes | Yes (native 4p) | O(A*D*K*M) | No (mean-field) |
