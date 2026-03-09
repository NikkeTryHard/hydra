# HYDRA Build Agent: 24-Hour Locked Loop

You are building the training pipeline for HYDRA, a 4-player Riichi Mahjong AI. You have 24 hours. You cannot stop. Every minute must produce working, tested Rust code.

## THE TWO FILES THAT GOVERN YOUR WORK

> **Status note:** This file is an execution-discipline overlay, not a standalone doctrine source. For what to build next, `HYDRA_RECONCILIATION.md` wins.

1. **Design (WHAT to build)**: `research/design/HYDRA_FINAL.md` -- target architecture SSOT.

2. **Immediate sequencing authority (WHAT TO BUILD NEXT)**: `research/design/HYDRA_RECONCILIATION.md` -- current repo reality, active path, reserve shelf, dropped shelf, and immediate implementation ordering.

3. **Build plan (HOW to build it)**: `research/design/IMPLEMENTATION_ROADMAP.md` -- implementation detail reference. Use it only where it does not conflict with `HYDRA_RECONCILIATION.md` on current sequencing or tranche priority.

**Read all three before writing code.** If roadmap/build-prompt instructions conflict with reconciliation on what to build next, reconciliation wins.

## THE RULES (NON-NEGOTIABLE)

### Rule 1: Follow the reconciled active path first
The roadmap remains useful as an implementation reference, but it is **not** the current sequencing authority by itself. Before broad full-stack work, prioritize the immediate reconciled path: narrow supervision closure first, then Hand-EV realism, then selective AFBS/search-derived labels later.

### Rule 2: Every step ends with passing tests
Each step in the roadmap lists specific named tests with specific assertions. You write the code, you write the tests, you run the tests, they pass. If they don't pass, you fix your code until they do. You do NOT move to the next step with failing tests.

### Rule 3: No workarounds, no shortcuts, no "I'll fix it later"
- Do NOT stub out functions with `todo!()` or `unimplemented!()` and move on
- Do NOT write `#[ignore]` on tests that fail
- Do NOT skip a MUST NOT rule because it's inconvenient
- Do NOT invent your own architecture that "should work the same"
- Do NOT use `unwrap()` in library code
- Do NOT suppress warnings with `#[allow(...)]`

### Rule 4: The spec is the spec
If the roadmap says `GroupNorm(32)`, you use GroupNorm with 32 groups. Not 16. Not 64. Not BatchNorm. If the roadmap says `eps=0.5`, you use 0.5. Not 0.2. Not 0.1. If the roadmap says "ONE epoch per batch", you do one epoch. Not two. Not four. The numbers are not suggestions.

### Rule 5: Use worktrees
All code goes in a git worktree at `.worktrees/build`. Do NOT edit the main repo working tree directly. Create the worktree, do your work there, commit frequently.

### Rule 6: Commit after every passing gate
When a gate passes, commit with a descriptive message. Do not accumulate 8 steps of uncommitted work. If the session dies, we should be able to resume from the last passing gate.

### Rule 7: Verify with cargo at every step
After every step, run ALL THREE:
```
cargo build --release
cargo nextest run --release
cargo clippy --all-targets -- -D warnings
```
All three must pass. If clippy warns, fix it. If tests fail, fix them. If it doesn't compile, fix it. Do not proceed with any failures.

## WHAT YOU ARE BUILDING

A Rust crate called `hydra-train` plus supporting `hydra-core` interfaces, but do **not** treat this as license to widen the current tranche. The immediate work should stay on the reconciled active path.

Long-range target surface includes:
- SE-ResNet backbone (24-block LearnerNet + 12-block ActorNet)
- 8 inference heads + oracle critic
- All loss functions with exact weights
- MJAI data loader with 24x augmentation
- Behavioral cloning training loop
- GAE advantage computation
- ACH (Actor-Critic Hedge) loss with gate c
- DRDA wrapper with rebase rule
- CT-SMC exact DP belief sampler
- AFBS search engine with PUCT
- Robust opponent modeling (KL-ball + archetypes)
- Self-play arena with batch simulation
- ExIt target pipeline with safety valve
- Distillation worker (learner -> actor)
- Endgame PIMC solver
- Sinkhorn belief projection
- Hand-EV oracle features
- Search-as-Feature MLP adaptor
- Population league
- Evaluation harness
- Inference server

All tested. All compiling. All clippy-clean.

## CRATE PLACEMENT

**hydra-core** (pure Rust, no Burn): ct_smc, afbs, robust_opponent, endgame, sinkhorn, hand_ev, arena
**hydra-train** (Burn-dependent): model, backbone, heads, config, ach, drda, gae, bc, exit, distill, saf, league, eval, inference, losses

## THE 12 STEPS (summary -- full details in IMPLEMENTATION_ROADMAP.md)

**Immediate tranche note:** do not assume Step 8+ sequencing is current just because it exists in the roadmap. For current execution order, defer to `HYDRA_RECONCILIATION.md`.

1. Create hydra-train crate scaffold
2. SE-ResNet backbone (SEBlock, SEResBlock, SEResNet)
3. Output heads (8 inference + oracle critic)
4. Full model (backbone + heads combined)
5. Loss functions (all head losses + total weighted)
6. MJAI data loader + augmentation
7. BC training loop
8. GAE + ACH + DRDA (the critical step)
9. CT-SMC exact DP sampler
10. AFBS search + robust opponent
11. Self-play arena + distillation + ExIt
12. Remaining components (endgame, sinkhorn, hand-ev, saf, league, eval, inference)

Final: Integration test chaining all 12 steps end-to-end.

## WHAT "DONE" LOOKS LIKE

```
cargo build --release          # zero errors
cargo nextest run --release    # all ~55 tests pass
cargo clippy -- -D warnings    # zero warnings
```

Plus the integration test `full_pipeline_integration` passes: BC -> ACH -> AFBS -> ExIt -> distill -> inference, all producing valid outputs with no NaN and no panics.

## COMMON FAILURE MODES (things that WILL go wrong)

1. **Burn API changes**: The roadmap assumes burn 0.16+. If APIs differ, check `burn.dev/docs/burn` and adapt. Do NOT switch to a different framework.

2. **Tensor shape mismatches**: The #1 bug. Print shapes at every layer boundary during development. The roadmap has exact shape traces -- use them.

3. **ACH loss sign errors**: The loss is `- c * eta * (y[a] / pi_old) * advantage`. The negative sign matters. The division by pi_old (not subtraction) matters. This is NOT PPO.

4. **DRDA rebase breaks policy**: After rebase, the policy MUST be identical (KL < 1e-6). If it's not, your fold + zero procedure is wrong. Test this BEFORE using rebase in training.

5. **CT-SMC overflow**: Use log-space DP. If you see NaN or Inf, you're not using logsumexp.

6. **Illegal action selection**: The policy MUST never select an illegal action. Legal masking (set illegal logits to -1e9 before softmax) must be applied EVERYWHERE -- in inference, in ACH, in ExIt, in arena.

7. **Forgetting to read existing code**: `hydra-core` already has an encoder, action space, safety module, bridge, simulator, shanten, tiles, seeding, and MJAI events. USE THEM. Do not reimplement what exists.

## TIME MANAGEMENT

You have 24 hours. Here's a realistic pace:

| Steps | Time | What |
|-------|------|------|
| 1-4 | Hours 1-6 | Foundation: crate + model + heads |
| 5-7 | Hours 7-12 | Data + losses + BC loop |
| 8 | Hours 13-16 | ACH + DRDA + GAE (hardest step) |
| 9-10 | Hours 17-20 | CT-SMC + AFBS |
| 11-12 | Hours 21-23 | Arena + ExIt + remaining |
| Final | Hour 24 | Integration test + cleanup |

If you're ahead of schedule, use extra time to:
- Add more edge case tests
- Improve error messages
- Add logging/tracing instrumentation
- Write rustdoc for public items

If you're behind schedule, do NOT skip steps. Do fewer tests per step but still pass the gate.

## START NOW

1. Read `research/design/HYDRA_FINAL.md`
2. Read `research/design/HYDRA_RECONCILIATION.md`
3. Read `docs/GAME_ENGINE.md`
4. Use `research/design/IMPLEMENTATION_ROADMAP.md` only as a reference for the currently selected tranche
