# FINAL Critique Agent Review: Is HYDRA-OMEGA Ready to Build?

## YOUR ROLE
You are the final reviewer before implementation begins. You've seen this design evolve through multiple iterations. This is the last chance to catch problems. Be ruthless.

## THE DESIGN
https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md

## PREVIOUS REVIEW HISTORY
- v1: 5/5 judges unanimously chose 9-technique system over 22-technique
- v2: Found ACH implementation details, CT-SMC 3D fix, 3-tier architecture
- v3: Added DRDA-wrapped ACH, opponent archetype set, early ExIt
- Math review: Fixed correlation sqrt, CT-SMC 15^3 states, Phase 2 game count

## WHAT I NEED FROM YOU

### SECTION A: Final Probability Assessment

Read the complete design. Given:
- 2000 GPU hours on 4x RTX 5000 Ada
- 3-tier network (12-block actor / 24-block learner / 40-block teacher)
- DRDA-wrapped ACH training
- Oracle guiding + oracle critic
- CT-SMC exact beliefs + Mixture-SIB
- AFBS with opponent archetypes + KL robustness
- SaF amortization + endgame solver
- Hand-EV oracle features
- Population training with league

Give me your honest win probability against each:

| Opponent | Your estimate (0-100%) | Key reasoning (1 sentence) |
|----------|----------------------|--------------------------|
| Mortal (7 dan) | ? | ? |
| NAGA (8-9 dan) | ? | ? |
| Suphx (8.74 dan) | ? | ? |
| LuckyJ (10.68 dan) | ? | ? |

### SECTION B: The 3 Biggest Remaining Risks

What are the THREE most likely ways this project fails to reach 10+ dan? For each:
1. What goes wrong
2. How likely (%)
3. How to detect it early (which validation gate catches it)
4. The fallback plan

### SECTION C: Missing Pieces Check

Go through this checklist. For each, answer YES (covered) or NO (gap):

1. [ ] Can we actually train ACH in Burn/Rust? (no PyTorch reference implementation exists)
2. [ ] Is DRDA's base-policy-round-update actually compatible with ACH's Hedge accumulator?
3. [ ] Does CT-SMC's 4.0M ops DP actually run in <1ms on real hardware? (not just theoretical)
4. [ ] Is the 3-tier distillation (12/24/40) stable? Has anyone done this exact pattern?
5. [ ] Does SaF-dropout (p=0.3) prevent degradation when search features are absent at deployment?
6. [ ] Is the endgame solver (1024 multiset states * 40x game state) fast enough for 100ms decisions?
7. [ ] Can AFBS root-sampling (particles propagated, not recomputed) maintain accuracy over depth 10-14?
8. [ ] Is the opponent archetype set (N=4) sufficient? Does OLSS show diminishing returns past N=4?
9. [ ] Does the budget (50+200+800+950) leave enough for hyperparameter tuning and failed experiments?
10. [ ] Is there a clear "minimum viable product" that works if half the components fail?

### SECTION D: The MVP Question

If we could only build 5 of the following 12 components (the rest are cut), which 5 maximize our dan rating?

1. 3-tier network architecture
2. DRDA-wrapped ACH
3. Oracle guiding + oracle critic
4. CT-SMC exact beliefs
5. AFBS search
6. SaF amortization
7. Endgame solver
8. Hand-EV oracle features
9. Opponent archetype set
10. Population training / league
11. Mixture-SIB
12. Hunter/Kounias safety bounds

### SECTION E: What Would You Change?

If you could make ONE final change to the design before we start coding, what would it be?

## REFERENCES
- ACH: https://openreview.net/pdf?id=DTXZqTNV5nW
- OLSS: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- Suphx: https://arxiv.org/pdf/2003.13590
- DRDA: https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf
- PPO in IIGs: https://arxiv.org/pdf/2502.08938
- KataGo: https://arxiv.org/pdf/1902.10565
- Our repo: https://github.com/NikkeTryHard/hydra
- Implementation roadmap: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- BPR feasibility: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/BPR_FEASIBILITY.md

## OUTPUT FORMAT
Be concise. Use tables. No fluff. This is the last review before we spend 2000 GPU hours.
