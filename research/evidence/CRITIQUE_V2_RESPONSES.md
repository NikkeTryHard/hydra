# Critique Agent v2 Responses -- Implementation Uncertainties Resolved

## Summary of Key Resolutions

### U1: ACH -- RESOLVED
- eta(s) = global scalar hyperparameter (try eta in {1,2,3})
- Gate c is per-(s,a) sample
- Uses standard GAE, per-player advantages
- Implementable as "PPO with modified loss" -- same pipeline
- Key diff: logits not log-probs, hard gates, centered+clipped logits, 1 epoch

### U2: CT-SMC -- RESOLVED (critical fix)
- 4th dim (wall) is DERIVED: c_W = R_k - (c1+c2+c3)
- State space is 3D: (14)^3 = 2,744 states (not 269K!)
- 34 x 2744 x 35 = 3.3M ops = sub-millisecond in Rust
- Can run at ANY game phase. Use log-space for stability.

### U3: Rollout net -- RESOLVED
- LuckyJ env model = policy only (action predictor)
- Recommended: policy+value rollout net for fastest search
- Distill CONTINUOUSLY, not every 50h
- Same input encoding as big net

### U4: Oracle+ACH -- RESOLVED
- Oracle only for advantages. Actor uses public info only.
- Normalize advantages per-minibatch for scale stability.
- Phase 1: oracle-guided pretrain. Phase 2: ACH with oracle critic.

### U5: Endgame -- RESOLVED
- Use multiset DP, not sequence enumeration
- Wall=10, all distinct: 2^10 = 1024 states (tiny)
- Exact draws + approximate actions (rollout policy)

### U6: Network sizing -- SURPRISE CHANGE
- 40 blocks monolithic = TOO BIG for 2000 hours
- Recommended: 12-block actor + 24-block learner + 40-block teacher
- Teacher used only for hard-position mining, not everything

### U7: SaF -- RESOLVED
- g_psi = tiny MLP (hidden 32-64), shared per-action
- Train: first supervised on delta(a), then joint end-to-end
- MUST use SaF-dropout to prevent over-reliance
