# Critique Agent Deep Dive v3: Follow-ups + Confirm Prior Findings

## YOUR ROLE
Same as before: you IMPLEMENT game AI systems. Pseudocode-level answers. No hand-waves.

## CONTEXT
HYDRA-OMEGA design: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
Previous critique responses: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/evidence/CRITIQUE_V2_RESPONSES.md
Full repo: https://github.com/NikkeTryHard/hydra

We just made major changes based on your v2 answers:
- Switched to 3-network architecture (12 actor / 24 learner / 40 teacher)
- Fixed CT-SMC to 3D DP (2,744 states, 3.3M ops)
- Added SaF-dropout (p=0.3)
- ACH as primary training algo with global eta

Now I need follow-ups on the things I'm STILL unsure about after your answers, plus confirmation of some earlier findings.

## REFERENCE PAPERS (use these)

**ACH + LuckyJ:**
- ACH paper: https://openreview.net/pdf?id=DTXZqTNV5nW (ICLR 2022)
- OLSS paper: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf (ICML 2023)
- GSCU: https://proceedings.mlr.press/v162/fu22b/fu22b.pdf (ICML 2022)

**Training:**
- PPO: https://arxiv.org/pdf/1707.06347
- IMPALA: https://arxiv.org/pdf/1802.01561
- GAE: https://arxiv.org/pdf/1506.02438
- DRDA: https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf
- R-NaD/DeepNash: https://www.science.org/doi/10.1126/science.add4679
- PPO in IIGs (5600 runs): https://arxiv.org/pdf/2502.08938
- Chinchilla scaling: https://arxiv.org/pdf/2203.15556

**Mahjong AI:**
- Suphx: https://arxiv.org/pdf/2003.13590
- Mortal: https://github.com/Equim-chan/Mortal
- Mortal paper: https://arxiv.org/pdf/2404.12877

**Search + beliefs:**
- KataGo: https://arxiv.org/pdf/1902.10565
- ReBeL: https://arxiv.org/pdf/2007.13544
- AlphaZero: https://www.science.org/doi/10.1126/science.aar6404
- PIMC strategy fusion: https://arxiv.org/pdf/2408.02380

**Other:**
- LoRA: https://arxiv.org/pdf/2106.09685
- FiLM: https://arxiv.org/pdf/1709.07871
- Distillation survey: https://arxiv.org/pdf/2006.05525
- Board game scaling laws: https://arxiv.org/pdf/2104.03113

---

## FOLLOW-UP 1: ACH -- The "one update epoch" detail

You said ACH uses ONE update epoch per minibatch, not PPO's typical 3-10 epochs. This is a huge throughput difference.

**1a.** Is this correct? Cite the exact line from Algorithm 2 that says one epoch. Because PPO's advantage comes partly from reusing each batch for multiple gradient steps.

**1b.** If ACH is truly one-epoch, then with the same data budget, ACH makes 3-10x FEWER gradient updates than PPO. How does it still converge faster? Is the Hedge-derived conservative gating so much more stable that it compensates?

**1c.** Can we safely do 2-3 epochs with ACH (like PPO) without breaking the Hedge guarantees? Or does the regret bound require single-epoch updates?

**1d.** What happens if we use ACH's loss function but PPO's multi-epoch training loop? Is that "ACH-flavored PPO" or does it break something?

---

## FOLLOW-UP 2: 3-Network Distillation -- The Practical Details

You recommended: 12-block actor + 24-block learner + 40-block teacher.

**2a. Distillation schedule.** You said "continuous distillation." But the teacher (40 blocks) is expensive to run forward passes on. If the teacher only processes "hard positions," how do we SELECT hard positions? Concrete criteria:
- Top-2 policy gap < X%?
- High particle ESS collapse?
- Endgame positions?
- Random subsample?
What fraction of positions should the teacher process? 1%? 5%? 20%?

**2b. GPU allocation with 3 networks.** We have 4x RTX 5000. Old plan: GPU 0-1 training, GPU 2 self-play, GPU 3 pondering. New plan with 3 nets:
- Where does each net run?
- How do we fit teacher inference + learner training + actor self-play on 4 GPUs?
- Does the teacher need a dedicated GPU? Or can it time-share?

**2c. Learner -> Actor distillation frequency.** You said "every few minutes, IMPALA-style." But in IMPALA, actors run the SAME architecture as learner -- they just have stale weights. Here our actor is a DIFFERENT architecture (12 blocks vs 24). So we can't just "copy weights" -- we must run actual distillation (forward passes + KD loss). How often is often enough? Every N games? Every M gradient steps?

**2d. Is the 3-network approach proven?** Has anyone actually done 3-tier distillation (actor/learner/teacher) in game AI? Or is this novel? I want to know if there's precedent.

---

## FOLLOW-UP 3: ACH vs DRDA -- Do we need both?

Our design currently says "Phase 2: ACH (primary) or DRDA/PPO (fallback)." But in v1, DRDA was the primary stability backbone.

**3a.** ACH and DRDA solve DIFFERENT problems right? ACH = game-theoretic policy updates. DRDA = stable multi-agent dynamics. Can they be COMBINED? E.g., use DRDA's aggregation rule to combine multiple ACH-trained agents?

**3b.** Or does ACH REPLACE DRDA entirely? Since ACH already has game-theoretic stability (Hedge-based), does DRDA add anything on top?

**3c.** The DRDA paper (ICLR 2025) claims convergence in multiplayer POSGs. Does ACH have similar multiplayer convergence guarantees? Or only 2-player?

**3d.** Concrete recommendation: should we implement ACH only, DRDA only, or both? If both, how do they interact in the training loop?

Read: ACH paper Section 3-4 for convergence analysis, DRDA paper for multiplayer POSG guarantees.

---

## FOLLOW-UP 4: CT-SMC -- Integrating with AFBS

The 3D DP is sub-millisecond. Great. But how does it plug into AFBS?

**4a.** At the AFBS root, we compute the CT-DP partition function and sample N particles. But as AFBS expands the search tree (depth 2-14), the game state changes at each node. Do we need to RECOMPUTE the DP at every search node? That would be 2744-state DP * thousands of nodes = back to being slow.

**4b.** Or do we sample particles ONCE at the root and propagate them through the tree? If so, how do we handle the fact that each particle assigns specific tiles to specific opponents, and those assignments affect the search dynamics?

**4c.** When we observe an opponent action during search (e.g., opponent discards tile X in a rollout), how do we UPDATE the particle weights? Is it just importance reweighting, or do we need to resample?

**4d.** For the ENDGAME solver specifically: we said "for each CT-SMC particle, enumerate wall draws exactly." But if we have P=1024 particles and wall=10 with 1024 multiset states each, that's 1M DP evaluations. Is this still fast enough (<100ms)?

---

## FOLLOW-UP 5: Training Budget Allocation (the real plan)

With the 3-network architecture, the compute budget changes.

**5a.** Give me a CONCRETE allocation of 2000 GPU hours across phases:

| Phase | GPU hours | What happens | Which nets train |
|-------|-----------|-------------|-----------------|
| Phase 0: BC | ? | ? | ? |
| Phase 1: Oracle guiding | ? | ? | ? |
| Phase 2: ACH self-play | ? | ? | ? |
| Phase 3: ExIt + pondering | ? | ? | ? |

Account for:
- Teacher training/inference time (40 blocks is expensive)
- Distillation overhead
- Self-play throughput at each phase

**5b.** How many total self-play games do we generate? How does 3-network change this vs our old estimate of ~35M games?

**5c.** At what point in training should the teacher (40-block) be introduced? From the start? After the learner has converged on BC? Only in Phase 3?

---

## FOLLOW-UP 6: Confirm or Challenge These Earlier Claims

From our earlier research sprint, we made several claims. Tell me if each is correct or wrong:

**6a.** "Oracle guiding adds ~1.7 dan" (from Suphx 7.0 -> 8.74). Is this a fair characterization? Or was the improvement from MULTIPLE techniques (oracle + GRP + other) and we're overcrediting oracle?
Read: https://arxiv.org/pdf/2003.13590 Figure 8 for ablation.

**6b.** "PPO entropy coefficient must be 0.05-0.1 for IIGs" (from Rudolph et al. 2025). Does this apply to ACH too? ACH's recommended entropy coeff from the paper is 5e-4 (100x smaller). Are these contradictory?
Read: https://arxiv.org/pdf/2502.08938 and ACH paper Table 4.

**6c.** "OLSS adds +0.2 fan over base ACH policy" (from OLSS Table 3). But OLSS was tested against a fixed ACH opponent in self-play, not on Tenhou. Is 0.2 fan an underestimate of OLSS's contribution to the 10.68 dan rating?

**6d.** "LuckyJ doesn't use oracle guiding." Is this confirmed? Or could they use it during training and just not mention it?

**6e.** "Suphx's pMCPA was not deployed on Tenhou" (Footnote 8). Does this mean pMCPA doesn't work, or just that it was too slow in 2020? With modern hardware (RTX 5000), could pMCPA be fast enough?

---

## FOLLOW-UP 7: What Would You Change About Our Design?

Look at the full HYDRA_FINAL.md design as it stands now. If you could change EXACTLY THREE things to maximize our probability of beating LuckyJ (10.68 dan) within 2000 GPU hours, what would they be?

Rules:
- Must be specific (not "train better")
- Must be implementable
- Must fit within 2000 GPU hours
- Must have justification (cite evidence)

---

## OUTPUT FORMAT
For each follow-up (1-7), provide:
1. **Answer** (concrete, with math/pseudocode where applicable)
2. **Confidence** (low/medium/high)
3. **Key reference** (paper + section)
4. **What to code** (the implementation action item)
