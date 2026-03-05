# HYDRA Code Build Plan

**Goal**: Write all Rust/Burn code until `hydra-train` compiles, tests pass, and is ready to train.
**No GPU hours. No training. Just code that works.**

**End state**: `cargo build --release && cargo test --release && cargo clippy -- -D warnings` all green. Every HYDRA_FINAL component implemented and tested.

**Dev agent rules**:
- Every task ends with tests that PASS
- No `unwrap()` in library code
- `cargo clippy -- -D warnings` clean after every task
- Read the referenced files before writing code

---

## Existing code (read these first)

| Module | Path |
|--------|------|
| Encoder | `hydra-core/src/encoder.rs` (85x34 obs, 31 tests) |
| Actions | `hydra-core/src/action.rs` (46 actions, 18 tests) |
| Safety | `hydra-core/src/safety.rs` (23 channels, 18 tests) |
| Bridge | `hydra-core/src/bridge.rs` (game state -> encoder) |
| Simulator | `hydra-core/src/simulator.rs` (batch sim, 29K g/s) |
| Shanten | `hydra-core/src/shanten_batch.rs` (shanten + 34 discards) |
| Tiles | `hydra-core/src/tile.rs` (suit permutation) |
| Seeding | `hydra-core/src/seeding.rs` (BLAKE3 RNG) |
| MJAI | `hydra-engine/src/mjai_event.rs` (event replay) |

---

## STEP 1: Create hydra-train crate

**Creates**: `hydra-train/Cargo.toml`, `hydra-train/src/lib.rs`
**Read first**: root `Cargo.toml` (workspace members)

- Add `hydra-train` to workspace in root `Cargo.toml`
- Dependencies: `burn`, `burn-tch`, `hydra-core`, `anyhow`, `serde`
- Module structure:
```
hydra-train/src/
  lib.rs
  model/         -- neural network code
    mod.rs
    backbone.rs  -- SE-ResNet
    heads.rs     -- 8 output heads + oracle critic
    saf.rs       -- SaF adaptor
  data/          -- data loading + features
    mod.rs
    mjai_loader.rs  -- MJAI file -> training batches
    hand_ev.rs      -- hand-EV oracle features
    augment.rs      -- suit permutation + seat rotation
  training/      -- training loops + losses
    mod.rs
    losses.rs    -- all loss functions
    ach.rs       -- ACH update with gate c
    drda.rs      -- DRDA base policy wrapper
    gae.rs       -- GAE advantage computation
    bc.rs        -- behavioral cloning loop
    exit.rs      -- ExIt target pipeline
    distill.rs   -- learner -> actor distillation
  search/        -- belief + search
    mod.rs
    ct_smc.rs    -- CT-SMC exact DP sampler
    afbs.rs      -- AFBS search engine
    robust.rs    -- robust opponent modeling
    endgame.rs   -- endgame PIMC solver
    sinkhorn.rs  -- Sinkhorn for deployment beliefs
  arena/         -- self-play + evaluation
    mod.rs
    selfplay.rs  -- batch self-play arena
    league.rs    -- population training
    eval.rs      -- evaluation suite
  inference/     -- deployment
    mod.rs
    server.rs    -- inference server
```

**Gate**: `cargo build` succeeds. `cargo test` runs (even if 0 tests).

---

## STEP 2: SE-ResNet backbone

**Creates**: `hydra-train/src/model/backbone.rs`
**Read first**: HYDRA_FINAL Section 4.2, Burn docs for Conv1d/GroupNorm

Implement:
- `SEBlock<B>`: global avg pool -> Linear(C, C/16) -> Mish -> Linear(C/16, C) -> sigmoid -> scale
- `SEResBlock<B>`: Conv1d(C,C,k=3,pad=1) -> GroupNorm(32) -> Mish -> Conv1d(C,C,k=3,pad=1) -> GroupNorm -> + SE -> + skip
- `SEResNet<B>`: stem Conv1d(input_ch, 256, k=3, pad=1) + N blocks + global avg pool
- Constructor takes `n_blocks: usize`, `input_channels: usize`
- LearnerNet = `SEResNet::new(24, 85)`, ActorNet = `SEResNet::new(12, 85)`
- Forward returns: `(feature_map: [B, 256, 34], pooled: [B, 256])`

**Tests**:
- `test_se_block_shapes`: input [4, 256, 34] -> output [4, 256, 34]
- `test_resblock_skip`: output shape matches input shape
- `test_backbone_24_forward`: [4, 85, 34] -> ([4, 256, 34], [4, 256])
- `test_backbone_12_forward`: same shapes, fewer blocks
- `test_param_count_24`: assert ~10M params (within 20%)
- `test_param_count_12`: assert ~5M params (within 20%)

**Gate**: all 6 tests pass. `cargo clippy` clean.

---

## STEP 3: Output heads

**Creates**: `hydra-train/src/model/heads.rs`
**Read first**: HYDRA_FINAL Section 4.3 (head table), Section 8 (score bins)

Implement `HydraHeads<B>` with methods:
- `policy(pooled, feature_map, legal_mask) -> [B, 46]` (masked softmax)
- `value(pooled) -> [B, 1]`
- `grp(pooled) -> [B, 24]` (softmax)
- `tenpai(pooled) -> [B, 3]` (sigmoid)
- `danger(pooled) -> [B, 3, 34]` (sigmoid)
- `opp_next(pooled) -> [B, 3, 34]` (per-opponent softmax)
- `score_pdf(pooled) -> [B, 64]` (softmax)
- `score_cdf(pooled) -> [B, 64]` (sigmoid)

Implement `OracleCritic<B>` separately:
- Small SEResNet(6 blocks) with extended input (85 + oracle_channels) * 34
- Output: `[B, 4]` (one value per player)
- Zero-sum normalization: subtract mean across player dim

**Tests**:
- `test_policy_legal_mask`: illegal actions get probability 0.0
- `test_policy_sums_to_one`: probabilities sum to 1.0 (within 1e-5)
- `test_all_head_shapes`: verify every head output shape
- `test_oracle_critic_zero_sum`: verify sum of 4 values = 0.0 (within 1e-5)
- `test_oracle_critic_extended_input`: verify accepts wider input tensor

**Gate**: all 5 tests pass.

---

## STEP 4: Full model (backbone + heads combined)

**Creates**: `hydra-train/src/model/mod.rs`
**Read first**: backbone.rs, heads.rs

Implement `HydraModel<B>`:
- Wraps `SEResNet` + `HydraHeads`
- `forward(obs: [B, 85, 34], legal_mask: [B, 46]) -> HydraOutput`
- `HydraOutput` struct: policy, value, grp, tenpai, danger, opp_next, score_pdf, score_cdf
- Implement `forward_with_logits` variant that returns raw logits (for ACH)

**Tests**:
- `test_full_model_forward`: random input -> all outputs valid shapes
- `test_full_model_backward`: loss.backward() produces non-zero gradients
- `test_model_save_load`: save checkpoint, load, verify identical outputs
- `test_learner_vs_actor`: 24-block and 12-block produce same-shape outputs

**Gate**: all 4 tests pass. Forward + backward works in bf16.

---

## STEP 5: Loss functions

**Creates**: `hydra-train/src/training/losses.rs`
**Read first**: HYDRA_FINAL Section 4.3 (loss table), Section 11 (loss weights)

Implement `HydraLoss<B>`:
- `policy_ce(pred_logits, target, weight) -> Tensor` (CE with soft target support)
- `value_mse(pred, target) -> Tensor`
- `grp_ce(pred, target) -> Tensor`
- `tenpai_bce(pred, target) -> Tensor`
- `danger_bce(pred, target) -> Tensor`
- `opp_next_ce(pred, target) -> Tensor`
- `score_pdf_ce(pred, target) -> Tensor`
- `score_cdf_bce(pred, target) -> Tensor`
- `total(outputs, targets) -> Tensor` with weights: pi=1.0, V=0.5, GRP=0.2, rest=0.1, score=0.05

**Tests**:
- `test_policy_ce_gradient`: gradient flows through logits
- `test_soft_target_ce`: CE with [0.3, 0.7, 0, ...] target (for ExIt)
- `test_total_loss_backward`: full loss produces gradients on all params
- `test_loss_weights_configurable`: can override default weights

**Gate**: all 4 tests pass.

---

## STEP 6: Data loader (MJAI -> training batches)

**Creates**: `hydra-train/src/data/mjai_loader.rs`, `augment.rs`
**Read first**: `hydra-engine/src/mjai_event.rs`, `hydra-core/src/encoder.rs`, `tile.rs`

Implement `MjaiDataset`:
- Parse `.mjson.gz` files into `Vec<MjaiEvent>`
- Replay game via `hydra-engine` state machine
- At each decision point: encode obs via `hydra-core::encoder`
- Extract: `MjaiSample { obs: [85,34], action: u8, legal_mask: [46], placement: u8, score_delta: i32 }`

Implement `Augmenter`:
- 6 suit permutations via `tile::SUIT_PERMUTATIONS`
- 4 seat rotations (view from each seat)
- Apply online during batch construction

Implement `MjaiDataLoader`:
- Shuffled batching with configurable batch size
- Outputs `MjajBatch { obs, actions, masks, placements, scores }` as Burn tensors

**Tests**:
- `test_load_single_game`: parse 1 .mjson.gz, verify >50 decision points
- `test_augment_6x`: suit permutation produces 6 distinct obs for same state
- `test_batch_shapes`: batch of 32 has correct tensor dimensions
- `test_legal_mask_valid`: no sample has all-zero mask

**Gate**: all 4 tests pass. Can load 1000 games without panic.

---

## STEP 7: BC training loop

**Creates**: `hydra-train/src/training/bc.rs`
**Read first**: losses.rs, mjai_loader.rs, model/mod.rs

Implement `BCTrainer`:
- Adam optimizer, LR 2.5e-4, batch 2048
- Epoch loop: load batch -> forward -> loss -> backward -> step
- Checkpoint save every epoch
- Eval: policy agreement on held-out set

**Tests**:
- `test_bc_one_step`: single gradient step reduces loss
- `test_bc_overfit_10_samples`: overfit to 10 samples in <100 steps (loss < 0.1)
- `test_checkpoint_save_load`: save, load, verify loss is identical

**Gate**: all 3 tests pass. BC can train on real data for 1 epoch without crash.

---

## STEP 8: GAE + ACH + DRDA

**Creates**: `hydra-train/src/training/gae.rs`, `ach.rs`, `drda.rs`
**Read first**: HYDRA_FINAL Section 11 Phase 2 (ACH update rule + DRDA wrapper)

Implement `compute_gae(rewards, values, dones, gamma=0.995, lambda=0.95) -> (advantages, returns)`:
- Per-player GAE computation
- Normalize advantages per-minibatch

Implement `ACHLoss`:
```rust
// Pseudocode -- implement exactly:
// y_centered = clamp(logits - mean(logits[legal]), -L_TH, L_TH)
// pi = masked_softmax(y_centered, legal_mask)
// ratio = pi[action] / pi_old
// gate_ratio = if adv >= 0 { ratio < 1+EPS } else { ratio > 1-EPS }
// gate_logit = if adv >= 0 { y[action] < L_TH } else { y[action] > -L_TH }
// c = gate_ratio && gate_logit  (as f32: 0.0 or 1.0)
// loss = -c * ETA * (y[action] / pi_old) * adv
```
- Hyperparams: ETA=1.0, EPS=0.5, L_TH=8.0, BETA_ENT=5e-4

Implement `DRDAWrapper`:
- Stores frozen `base_logits: Tensor` (detached)
- `combined_logits(base, residual, tau_drda) -> Tensor`
- `rebase(model)`: fold residual into base, zero residual, reset optimizer

**Tests**:
- `test_gae_simple`: 5-step trajectory, verify advantages match hand-computed
- `test_ach_gate_positive_adv`: c=1 when ratio and logit within bounds
- `test_ach_gate_clips`: c=0 when ratio exceeds 1+eps
- `test_ach_one_epoch`: verify loss changes after exactly 1 update
- `test_drda_rebase_preserves_policy`: pi before rebase == pi after rebase (KL < 1e-6)
- `test_drda_residual_zeroed`: after rebase, residual weights are all 0

**Gate**: all 6 tests pass.

---

## STEP 9: CT-SMC exact DP sampler

**Creates**: `hydra-train/src/search/ct_smc.rs`
**Read first**: HYDRA_FINAL Section 5.5. Pure Rust, no Burn needed.

Implement `CTDP`: 3D state (c1,c2,c3), c_W derived. Log-space partition function. Backward sampling.

Implement `CTSMCSampler`: Sample N particles, weight by opponent likelihood, resample at ESS < 0.4N.

**Tests**:
- `test_dp_marginals_match_hypergeometric`: uniform weights -> exact hypergeometric marginals
- `test_dp_backward_sample_valid`: sampled table has correct row/col sums
- `test_dp_no_nan`: no NaN/Inf on 1000 random states
- `test_dp_under_1ms`: benchmark median < 1ms

**Gate**: all 4 tests pass.

---

## STEP 10: AFBS search + robust opponent

**Creates**: `hydra-train/src/search/afbs.rs`, `robust.rs`
**Read first**: HYDRA_FINAL Sections 7-8. Depends on: model, ct_smc.

Implement `AFBSTree`: PUCT selection, top-K=5 expansion, batched leaf eval. Root sampling with particles.

Implement `RobustOpponent`: KL-ball soft-min, binary search for tau, N=4 archetypes.

Implement `PonderingManager`: Async, priority queue, lockless hashmap, 30s TTL.

**Tests**:
- `test_puct_selects_high_prior`: uniform Q -> picks highest prior
- `test_afbs_depth2`: search changes policy vs raw network
- `test_robust_epsilon_zero`: reduces to expectation under p
- `test_pondering_nonblocking`: self-play not blocked

**Gate**: all 4 tests pass. AFBS depth-4 < 150ms.

---

## STEP 11: Self-play arena + distillation + ExIt

**Creates**: `arena/selfplay.rs`, `training/exit.rs`, `training/distill.rs`
**Read first**: HYDRA_FINAL Sections 10-11. Depends on: model, ach, drda, gae, afbs.

Implement `SelfPlayArena`: Batch 500+ games, ActorNet on GPU 2, temperature-varied seats, checkpoint pool.

Implement `Distiller`: KD loss learner->actor, async every 1-2 min.

Implement `ExItPipeline`: Soft targets from search Q, KL-weighting, safety valve, warmup schedule.

**Tests**:
- `test_arena_10_games`: 10 games produce valid trajectories
- `test_distill_reduces_kl`: KL decreases after 10 steps
- `test_exit_soft_target`: valid probability distribution
- `test_exit_safety_valve`: bad targets filtered

**Gate**: all 4 tests pass.

---

## STEP 12: Remaining components

**12a: Endgame** (`search/endgame.rs`): PIMC, wall<=10, top-mass P=50-100.
Tests: `test_endgame_valid`, `test_endgame_under_50ms`

**12b: Sinkhorn** (`search/sinkhorn.rs`): 34x4 matrix, 10-30 iterations.
Tests: `test_sinkhorn_converges`, `test_sinkhorn_marginals`

**12c: Hand-EV** (`data/hand_ev.rs`): Per-discard tenpai/win/score/ukeire.
Tests: `test_hand_ev_shapes`, `test_hand_ev_ukeire`

**12d: SaF** (`model/saf.rs`): MLP adaptor, dropout p=0.3.
Tests: `test_saf_zero_absent`, `test_saf_modifies_present`

**12e: League** (`arena/league.rs`): Checkpoint pool, diverse sampling.
Tests: `test_league_diverse`

**12f: Eval** (`arena/eval.rs`): Policy agreement, Elo tracker.
Tests: `test_eval_computes`

**12g: Inference** (`inference/server.rs`): ActorNet bf16, agari guard.
Tests: `test_inference_legal`, `test_inference_under_1ms`

**Gate**: all component tests pass.

---

## FINAL GATE

```bash
cargo build --release
cargo test --release
cargo clippy --all-targets -- -D warnings
```

Integration test: `test_full_pipeline` -- BC 1 epoch on 100 games -> ACH 10 games -> AFBS 5 states -> ExIt targets -> distill -> inference returns valid actions.

**When this passes: code is ready. Plug in data and 2000 GPU hours.**
