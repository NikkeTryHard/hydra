# HYDRA Code Build Plan

**Goal**: Write all Rust/Burn code until `hydra-train` compiles, tests pass, and is ready to train.
**No GPU hours. No training. Just code that works.**

**End state**: `cargo build --release && cargo test --release && cargo clippy -- -D warnings` all green. Every HYDRA_FINAL component implemented and tested.

> **Current sequencing note:** This roadmap is an implementation-detail reference, not the sole authority on what to build next. For the immediate execution order, `research/design/HYDRA_RECONCILIATION.md` wins when it conflicts with this roadmap's broader full-stack sequencing.
>
> **Demotion note:** Treat this file as a reference/backlog map, not as the default build order for the strongest current Hydra. Any section that conflicts with reconciliation or current runtime reality should be treated as archived or future-use material, not immediate marching orders.

**Dev agent rules**:
- Every task ends with tests that PASS
- No `unwrap()` in library code (use `?` or explicit error handling)
- `cargo clippy -- -D warnings` clean after every task
- Read the referenced files BEFORE writing code
- Do NOT skip steps or proceed without passing the gate
- Do NOT invent your own approach -- follow the spec EXACTLY

**Design SSOT**: `research/design/HYDRA_FINAL.md`.

**Current execution authority**: `research/design/HYDRA_RECONCILIATION.md` for immediate tranche ordering.

---

# HYDRA Build Plan: Steps 1-4 (Neural Network Foundation)

> For an LLM coding agent. Every signature, shape, and value is EXACT.
> If something is not specified here, ASK -- do not invent.

---

## Step 1: Create hydra-train Crate

**Ref: HYDRA_FINAL Section 4, Section 11 (Training pipeline)**

### 1.1 Workspace Registration

Edit `/home/nikketryhard/dev/hydra/Cargo.toml` to add `hydra-train` to `[workspace] members`.

### 1.2 Crate Cargo.toml

Create `hydra-train/Cargo.toml`. Use whatever the latest `burn` version on crates.io is at build time.

**Required dependencies**:
- `hydra-core = { path = "../hydra-core" }`
- `burn = { version = "0.16", features = ["train", "tch"] }`
- `serde = { version = "1.0", features = ["derive"] }`

**Dev dependencies**: `burn = { version = "0.16", features = ["ndarray"] }`

**IMPORTANT**: If `burn` 0.16 doesn't exist yet, use whatever the latest stable release is. Check `crates.io/crates/burn` first.

### 1.3 Module Structure

```
hydra-train/
  Cargo.toml
  src/
    lib.rs          # pub mod declarations
    model.rs        # Full HydraModel combining backbone + heads
    backbone.rs     # SEResNet, SEResBlock, SEBlock
    heads.rs        # All 9 output heads
    config.rs       # HydraModelConfig (hyperparameters)
```

`lib.rs` exports: `pub mod backbone; pub mod config; pub mod heads; pub mod model;`

**Note**: Search/belief code goes in `hydra-core` (pure Rust, no Burn dep): `ct_smc.rs`, `afbs.rs`, `robust_opponent.rs`, `endgame.rs`, `sinkhorn.rs`, `hand_ev.rs`, `arena.rs`. Training code goes in `hydra-train` (depends on Burn): `ach.rs`, `drda.rs`, `gae.rs`, `bc.rs`, `exit.rs`, `distill.rs`, `saf.rs`, `league.rs`, `eval.rs`, `inference.rs`.

### 1.4 Constants (in `config.rs`)

**Canonical SSOT note:** `HYDRA_FINAL.md` is the governing architecture doc, and `HYDRA_RECONCILIATION.md` is the current repo-wide execution memo. The old `85 x 34` tensor describes the baseline prefix, not the full live encoder. The current code already uses the **fixed-shape 192 x 34 superset** of Groups A/B/C/D from `HYDRA_FINAL`, with zero-filled dynamic features plus presence-mask channels when search/belief/Hand-EV features are unavailable.

These are the exact defaults for the **current live implementation snapshot**, but this roadmap remains subordinate to `HYDRA_FINAL.md` and `HYDRA_RECONCILIATION.md` when they disagree on overall direction or priority:

| Constant | Value | Source |
|----------|-------|--------|
| `INPUT_CHANNELS` | 192 | `hydra_core::encoder::NUM_CHANNELS` |
| `TILE_DIM` | 34 | `hydra_core::encoder::NUM_TILES` |
| `HIDDEN_CHANNELS` | 256 | SE-ResNet channel width |
| `SE_REDUCTION` | 4 | SE squeeze ratio: 256/4 = 64 |
| `SE_BOTTLENECK` | `HIDDEN_CHANNELS / SE_REDUCTION` = 64 | |
| `NUM_GROUPS` | 32 | GroupNorm groups (Section 4.2) |
| `ACTION_SPACE` | 46 | `hydra_core::action::HYDRA_ACTION_SPACE` |
| `SCORE_BINS` | 64 | KataGo-style score dist |
| `NUM_OPPONENTS` | 3 | 4-player game minus self |
| `GRP_CLASSES` | 24 | Global reward prediction classes |

### 1.5 MUST NOT

- Do NOT add `burn-wgpu`, `burn-candle`, or any backend other than `tch` and `ndarray`.
- Do NOT use `BatchNorm`. HYDRA_FINAL Section 4.2 says GroupNorm(32).
- Do NOT use `Relu` or `Gelu`. Use `burn::tensor::activation::mish`.
- Do NOT create a `main.rs`. This is a library crate only.
- Do NOT import anything from `Mortal-Policy/`.
- Do NOT add `tokio`, `async-std`, or any async runtime.

---

## Step 2: SE-ResNet Backbone

**Ref: HYDRA_FINAL Section 4.2**

All code goes in `hydra-train/src/backbone.rs`.

### 2.1 SEBlock (Squeeze-and-Excitation)

**SEBlock<B: Backend>**: Squeeze-and-Excitation gate.
- Fields: `fc1: Linear<B>` (256, 64), `fc2: Linear<B>` (64, 256)
- **SEBlockConfig**: `channels: usize` (default 256), `bottleneck: usize` (default 64)
- `init<B>(&self, device) -> SEBlock<B>`: Create fc1 = Linear(channels, bottleneck), fc2 = Linear(bottleneck, channels).
- `forward(&self, x: [B, 256, 34]) -> [B, 256, 34]`: Global avg pool (mean_dim 2) -> squeeze -> fc1 -> mish -> fc2 -> sigmoid -> unsqueeze -> broadcast multiply with input.

Shape trace: `[B,256,34] -> mean(dim=2) -> [B,256,1] -> squeeze -> [B,256] -> fc1 -> [B,64] -> mish -> fc2 -> [B,256] -> sigmoid -> unsqueeze -> [B,256,1] -> broadcast * x -> [B,256,34]`

**MUST NOT:**
- Do NOT use `AdaptiveAvgPool1d` for the global pool. Use `mean_dim(2)`.
- Do NOT use `Relu` inside SE. Use `mish` for fc1 activation, `sigmoid` for gate.
- Do NOT forget the `x.clone()` -- Burn tensors move on use.

### 2.2 SEResBlock (Single Residual Block)

**SEResBlock<B: Backend>**: Pre-activation residual block with SE gate.
- Fields: `gn1: GroupNorm<B>` (32, 256), `conv1: Conv1d<B>` (256, 256, k=3, pad=Same), `gn2: GroupNorm<B>` (32, 256), `conv2: Conv1d<B>` (256, 256, k=3, pad=Same), `se: SEBlock<B>`
- **SEResBlockConfig**: `channels: usize` (256), `num_groups: usize` (32), `se_bottleneck: usize` (64)
- `init<B>(&self, device) -> SEResBlock<B>`: Init all fields from config.
- `forward(&self, x: [B, 256, 34]) -> [B, 256, 34]`: Pre-activation: GN1 -> Mish -> Conv1 -> GN2 -> Mish -> Conv2 -> SE -> add residual.

**MUST NOT:**
- Do NOT use post-activation (Conv -> GN -> Mish). Use PRE-activation.
- Do NOT add a projection/downsample path. All blocks are same-dim (256->256).
- Do NOT use `bias=false` on Conv1d. Keep default `bias=true`.
- The spatial dim MUST stay 34 through every block (padding=Same guarantees this).

### 2.3 SEResNet (Full Backbone)

**SEResNet<B: Backend>**: Full backbone.
- Fields: `input_conv: Conv1d<B>` (`INPUT_CHANNELS`, 256, k=3, pad=Same), `input_gn: GroupNorm<B>` (32, 256), `blocks: Vec<SEResBlock<B>>`, `final_gn: GroupNorm<B>` (32, 256)
- **SEResNetConfig**: `num_blocks: usize` (12 for ActorNet, 24 for LearnerNet), `input_channels: usize` (baseline 85; final target = fixed superset per `HYDRA_FINAL`), `hidden_channels: usize` (256), `num_groups: usize` (32), `se_bottleneck: usize` (64)
- `init<B>(&self, device) -> SEResNet<B>`: Create input_conv, input_gn, N residual blocks, final_gn.
- `forward(&self, x: [B, INPUT_CHANNELS, 34]) -> (spatial: [B, 256, 34], pooled: [B, 256])`:
  Input conv -> input_gn -> mish -> N residual blocks -> final_gn -> mish -> (spatial, global_avg_pool).

Shape trace: `[B,INPUT_CHANNELS,34] -> Conv1d -> [B,256,34] -> GN -> mish -> N blocks -> [B,256,34] -> final_gn -> mish -> spatial [B,256,34]; mean(dim=2) -> squeeze -> pooled [B,256]`

**Returns a TUPLE**: `(spatial, pooled)` because heads need both:
- `spatial: [B, 256, 34]` -- for per-tile heads (danger, opp_next_discard)
- `pooled: [B, 256]` -- for scalar/vector heads (policy, value, etc.)

**MUST NOT:**
- Do NOT flatten the spatial tensor. Keep it as [B, 256, 34].
- Do NOT apply final_gn INSIDE the loop. Apply it ONCE after all blocks.
- Do NOT return only pooled. Both spatial and pooled are needed.
- Do NOT hardcode num_blocks. It's configurable (12 or 24).

### 2.4 Tests for Step 2

File: `hydra-train/src/backbone.rs` (in `#[cfg(test)] mod tests`). Use `burn::backend::NdArray`.

| Test | Assertion |
|------|-----------|
| `se_block_preserves_shape` | SEBlock(256,64) on [4,256,34] -> output [4,256,34] |
| `se_res_block_preserves_shape` | SEResBlock(256,32,64) on [4,256,34] -> output [4,256,34] |
| `backbone_output_shapes_12_blocks` | SEResNet(12,85,256,32,64) on [4,85,34] -> spatial [4,256,34], pooled [4,256] |
| `backbone_output_shapes_24_blocks` | SEResNet(24,85,256,32,64) on [2,85,34] -> spatial [2,256,34], pooled [2,256] |

---

## Step 3: Output Heads

**Ref: HYDRA_FINAL Section 4.3**

All code goes in `hydra-train/src/heads.rs`.

There are 8 inference heads + 1 oracle-only head = 9 total. Each head is a separate struct. They share NO weights. Every head takes either `pooled: [B, 256]` or `spatial: [B, 256, 34]` from the backbone.

### 3.1 Head Summary Table

| # | Name | Input | Layer(s) | Output shape | Activation |
|---|------|-------|----------|--------------|------------|
| 1 | PolicyHead | pooled [B,256] | Linear(256,46) | [B, 46] | none (raw logits) |
| 2 | ValueHead | pooled [B,256] | Linear(256,1) | [B, 1] | tanh |
| 3 | ScorePdfHead | pooled [B,256] | Linear(256,64) | [B, 64] | log_softmax (at loss time) |
| 4 | ScoreCdfHead | pooled [B,256] | Linear(256,64) | [B, 64] | sigmoid |
| 5 | OppTenpaiHead | pooled [B,256] | Linear(256,3) | [B, 3] | sigmoid |
| 6 | GrpHead | pooled [B,256] | Linear(256,24) | [B, 24] | none (raw logits) |
| 7 | OppNextDiscardHead | spatial [B,256,34] | Conv1d(256,3,1) | [B, 3, 34] | none (raw logits) |
| 8 | DangerHead | spatial [B,256,34] | Conv1d(256,3,1) | [B, 3, 34] | sigmoid |
| 9 | OracleCriticHead | pooled [B,256] | Linear(256,4) | [B, 4] | none (raw) |

Note: Heads 7-10 from HYDRA_FINAL S4.3 (Mixture-SIB fields, hand-type latent, delta-Q regression, safety bound residual) are deferred to Step 12 (SaF + belief heads).

### 3.2 Struct Definitions

Each head is a `#[derive(Module, Debug)]` struct with a single `linear: Linear<B>` or `conv: Conv1d<B>` field. Forward methods are trivial pass-through with activation:
- **PolicyHead**: `forward(pooled: [B,256]) -> [B,46]` -- linear only, no activation
- **ValueHead**: `forward(pooled: [B,256]) -> [B,1]` -- linear -> tanh
- **ScorePdfHead**: `forward(pooled: [B,256]) -> [B,64]` -- linear only (log_softmax at loss time)
- **ScoreCdfHead**: `forward(pooled: [B,256]) -> [B,64]` -- linear only (sigmoid via bce_with_logits at loss time)
- **OppTenpaiHead**: `forward(pooled: [B,256]) -> [B,3]` -- linear only (sigmoid via bce_with_logits at loss time)
- **GrpHead**: `forward(pooled: [B,256]) -> [B,24]` -- linear only
- **OppNextDiscardHead**: `forward(spatial: [B,256,34]) -> [B,3,34]` -- conv1d(k=1) only
- **DangerHead**: `forward(spatial: [B,256,34]) -> [B,3,34]` -- conv1d(k=1) only (sigmoid via bce_with_logits at loss time)
- **OracleCriticHead**: `forward(pooled: [B,256]) -> [B,4]` -- linear only (4 values, one per player, zero-sum normalized in loss)

### 3.3 Config and Init

**HeadsConfig** (`#[derive(Config, Debug)]`):
- Fields: `hidden_channels: usize` (256), `action_space: usize` (46), `score_bins: usize` (64), `num_opponents: usize` (3), `grp_classes: usize` (24)
- Provide `init_*` methods for each head (e.g. `init_policy`, `init_value`, etc.)
- For Conv1d heads (OppNextDiscard, Danger), kernel_size=1, NO padding needed.

### 3.4 MUST NOT

- Do NOT apply softmax to PolicyHead output. That happens at loss/sampling time.
- Do NOT apply softmax to GrpHead output. Same reason.
- Do NOT apply log_softmax to ScorePdfHead in forward. Only at loss time.
- Do NOT use kernel_size=3 for OppNextDiscard/Danger. Use kernel_size=1 (pointwise).
- Do NOT share any weights between heads. Each is independent.
- Do NOT add hidden layers inside heads. Each head is a SINGLE Linear or Conv1d.
- Do NOT apply activation to OracleCriticHead. It's a raw value estimate.
- Value head uses `tanh`, NOT `sigmoid`. Value range is [-1, 1] not [0, 1].

### 3.5 Tests for Step 3

Use `burn::backend::NdArray`. Create a `default_config() -> HeadsConfig` helper.

| Test | Assertion |
|------|-----------|
| `policy_head_shape` | [4,256] -> [4,46] |
| `value_head_shape_and_range` | [4,256] -> [4,1], all values in [-1, 1] |
| `score_pdf_head_shape` | [4,256] -> [4,64] |
| `score_cdf_head_range` | [4,256] -> [4,64] |
| `opp_tenpai_head_shape` | [4,256] -> [4,3] |
| `grp_head_shape` | [4,256] -> [4,24] |
| `opp_next_discard_head_shape` | [4,256,34] -> [4,3,34] |
| `danger_head_shape_and_range` | [4,256,34] -> [4,3,34] |
| `oracle_critic_head_shape` | [4,256] -> [4,4] |

---

## Step 4: Full HydraModel

**Ref: HYDRA_FINAL Section 4.2 + 4.3**

All code goes in `hydra-train/src/model.rs`.

### 4.1 HydraOutput Struct

The model returns ALL head outputs in a single struct, NOT a tuple:

**HydraOutput<B: Backend>**:
- `policy_logits: Tensor<B, 2>` -- [B, 46]. Apply masked softmax externally.
- `value: Tensor<B, 2>` -- [B, 1]. In [-1, 1].
- `score_pdf: Tensor<B, 2>` -- [B, 64]. Apply log_softmax at loss time.
- `score_cdf: Tensor<B, 2>` -- [B, 64].
- `opp_tenpai: Tensor<B, 2>` -- [B, 3].
- `grp: Tensor<B, 2>` -- [B, 24].
- `opp_next_discard: Tensor<B, 3>` -- [B, 3, 34].
- `danger: Tensor<B, 3>` -- [B, 3, 34].
- `oracle_critic: Tensor<B, 2>` -- [B, 4]. Only meaningful during oracle training.

### 4.2 HydraModel Struct

**HydraModel<B: Backend>** (`#[derive(Module, Debug)]`):
- Fields: `backbone: SEResNet<B>`, plus one field for each of the 9 heads (policy, value, score_pdf, score_cdf, opp_tenpai, grp, opp_next_discard, danger, oracle_critic).

### 4.3 HydraModelConfig

**HydraModelConfig** (`#[derive(Config, Debug)]`):
- `num_blocks: usize` -- 12 for ActorNet, 24 for LearnerNet (NO default -- must be specified)
- `input_channels: usize` (default = current baseline 85; migrate to final fixed-superset channel count from `HYDRA_FINAL`), `hidden_channels: usize` (default 256), `num_groups: usize` (default 32), `se_bottleneck: usize` (default 64)
- `action_space: usize` (default 46), `score_bins: usize` (default 64), `num_opponents: usize` (default 3), `grp_classes: usize` (default 24)
- Convenience constructors: `actor() -> Self` (num_blocks=12, ~5M params), `learner() -> Self` (num_blocks=24, ~10M params)
- Usage: `HydraModelConfig::new(12)` for ActorNet, `HydraModelConfig::new(24)` for LearnerNet

### 4.4 Init and Forward

- `init<B>(&self, device) -> HydraModel<B>`: Create backbone from SEResNetConfig, create HeadsConfig, init all heads.
- `forward(&self, x: [B, INPUT_CHANNELS, 34]) -> HydraOutput<B>`: Backbone -> (spatial, pooled) -> feed each head. Clone `pooled` 6 times (7 uses, last is move), clone `spatial` 1 time (2 uses, last is move).

**CRITICAL**: The last uses of `pooled` and `spatial` must NOT be cloned (Burn moves on use).

### 4.5 MUST NOT

- Do NOT run the backbone twice (once per head). Run it ONCE, share outputs.
- Do NOT add dropout to the model. Dropout is NOT in HYDRA_FINAL architecture.
- Do NOT forget to clone tensors. Burn moves on use. Last consumer gets the move.
- Do NOT make HydraOutput generic over which heads are present. Always return all 9.
- Do NOT add a `training: bool` parameter to forward. Oracle critic is always computed; it's simply ignored at inference time.
- Do NOT reshape the input inside forward. The caller provides [B, 85, 34] directly.

### 4.6 Tests for Step 4

| Test | Assertion |
|------|-----------|
| `actor_net_all_output_shapes` | HydraModelConfig::new(12) on [4,85,34] -> all 9 head shapes correct |
| `learner_net_all_output_shapes` | HydraModelConfig::new(24) on [2,85,34] -> all 9 head shapes correct |
| `value_head_bounded` | Random input -> all value outputs in [-1, 1] |
| `danger_head_bounded` | Random input -> all danger outputs in [0, 1] |

Expected output shapes: policy=[B,46], value=[B,1], score_pdf=[B,64], score_cdf=[B,64], opp_tenpai=[B,3], grp=[B,24], opp_next_discard=[B,3,34], danger=[B,3,34], oracle_critic=[B,4]

---

## Global MUST NOT (All Steps 1-4)

1. **No BatchNorm.** Always GroupNorm(32). Ref: Section 4.2.
2. **No Relu/Gelu.** Always Mish. Ref: Section 4.2.
3. **No bf16 yet.** Build everything in f32 first. bf16 is a later optimization.
4. **No async.** All forward passes are synchronous.
5. **No Python.** 100% Rust. Ref: AGENTS.md.
6. **No Mortal code.** Do not look at or copy from `Mortal-Policy/`. Ref: AGENTS.md.
7. **Test with NdArray backend.** Tests use `burn::backend::NdArray`, not `Tch`.
8. **Burn API pitfall**: `Tensor` moves on use. Clone before reuse.
9. **Burn API pitfall**: `GroupNormConfig::new(num_groups, num_channels)` -- groups first, channels second. Do NOT reverse them.
10. **Burn API pitfall**: `Conv1dConfig::new(channels_in, channels_out, kernel_size)` -- these are positional, NOT named.

---

## Quick Reference: Import Map

From `hydra-core`: `encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE}` (85, 34, 2890), `action::HYDRA_ACTION_SPACE` (46)

From `burn`: `prelude::*` (Backend, Tensor, Module, Config), `nn::{Linear, LinearConfig, GroupNorm, GroupNormConfig}`, `nn::conv::{Conv1d, Conv1dConfig}`, `nn::PaddingConfig1d`, `tensor::activation` (mish, sigmoid, tanh)

# Hydra Implementation Roadmap: Steps 5-8

## STEP 5: Loss Functions

**File**: `hydra-train/src/training/losses.rs`
**Ref**: HYDRA_FINAL S4.3, S11 Phase 0/2; HYDRA_SPEC Value/GRP/Tenpai/Danger heads

### 5.1 HydraLoss Struct

**HydraLoss<B: Backend>**: Holds loss weight hyperparameters.
- Fields: `w_pi: f32` (1.0), `w_v: f32` (0.5), `w_grp: f32` (0.2), `w_tenpai: f32` (0.1), `w_danger: f32` (0.1), `w_opp: f32` (0.1), `w_score: f32` (0.05 -- covers BOTH pdf + cdf, split 50/50 internally), `_backend: PhantomData<B>`

### 5.2 Per-Head Loss Functions

| Function | Signature | Loss Type | Notes |
|----------|-----------|-----------|-------|
| `policy_ce` | `(logits: [B,46], target: [B,46], mask: [B,46]) -> [B]` | CE with soft targets | Mask illegal logits to -1e9 before softmax. `loss = -sum(target * log_softmax(masked_logits))` |
| `value_mse` | `(pred: [B], target: [B]) -> [B]` | MSE | `0.5 * (pred - target)^2` |
| `grp_ce` | `(logits: [B,24], target: [B,24]) -> [B]` | CE with soft targets | Same as policy_ce over 24 classes |
| `tenpai_bce` | `(logits: [B,3], target: [B,3]) -> [B]` | BCE | Per-opponent independent sigmoid |
| `danger_focal_bce` | `(logits: [B,3,34], target: [B,3,34], mask: [B,3,34]) -> [B]` | Focal BCE | alpha=0.25, gamma=2.0. Mask=0 where label unavailable |
| `opp_next_ce` | `(logits: [B,3,34], target: [B,3,34]) -> [B]` | CE per opponent | 3 independent 34-way CEs, summed |
| `score_pdf_ce` | `(logits: [B,64], target: [B,64]) -> [B]` | CE | Softmax over 64 score bins |
| `score_cdf_bce` | `(logits: [B,64], target: [B,64]) -> [B]` | BCE | Each bin: P(score <= threshold_i) |

### 5.3 Oracle Critic Loss (Phase 1+)

- Signature: `oracle_critic_loss(v_oracle: [B,4], target: [B,4]) -> scalar`
- MSE loss: `0.5 * mean((v_oracle - target)^2)`
- Zero-sum penalty: `mean((v_oracle.sum(dim=1))^2) * 10.0` -- enforces V1+V2+V3+V4 = 0
- Zero-sum normalization: `v_norm = v_oracle - v_oracle.mean(dim=1, keepdim=true)` before MSE

### 5.4 Total Loss

- Signature: `total_loss(outputs: &HydraOutput<B>, targets: &HydraTargets<B>) -> scalar`
- Weighted sum: `l_pi*1.0 + l_v*0.5 + l_grp*0.2 + l_tenpai*0.1 + l_danger*0.1 + l_opp*0.1 + (l_pdf + l_cdf)*0.025`

### 5.5 MUST NOT

- MUST NOT use label smoothing on policy head (soft targets handle this for ExIt)
- MUST NOT use reduction='sum' -- always mean-reduce per sample, then mean over batch
- MUST NOT forget the legal_mask on policy_ce (illegal actions -> -1e9 before softmax)
- MUST NOT apply focal loss to tenpai head (only danger head gets focal)
- MUST NOT let oracle critic loss backprop into the student backbone

### 5.6 Tests

| Test | Assertion |
|------|-----------|
| `test_policy_ce_gradient` | logits.requires_grad=true, backward produces non-zero grad on all 46 dims |
| `test_soft_target_ce` | target=[0.3,0.7,0,...] loss differs from hard-target [0,1,0,...] by >0.01 |
| `test_total_loss_backward` | all 8 head losses > 0, total loss backward fills all param grads |
| `test_loss_weights_configurable` | custom weights -> total changes proportionally |
| `test_oracle_critic_zero_sum` | v_oracle=[1,-1,2,-2] -> zero_sum_penalty = 0.0 (within 1e-5) |
| `test_focal_bce_vs_bce` | focal(alpha=0.25,gamma=2) < standard_bce for high-confidence correct preds |

---

## STEP 6: MJAI Data Loader + Augmentation

**Files**: `hydra-train/src/data/mjai_loader.rs`, `hydra-train/src/data/augment.rs`
**Ref**: HYDRA_FINAL S11 Phase 0; hydra-engine/src/mjai_event.rs; hydra-core/src/encoder.rs; hydra-core/src/tile.rs

### 6.1 Parsing .mjson.gz

- Signature: `parse_mjson_gz(path: &Path) -> Result<Vec<Vec<serde_json::Value>>>`
- Dependencies: `flate2::read::GzDecoder`, `serde_json`
- Each line in .mjson.gz is one JSON event (MJAI protocol). Group by game (split on "start_game" events).

### 6.2 Replay via hydra-engine

For each game: create GameState from start_kyoku -> step through events -> at each decision point:
1. Call `state.observe()` to get Observation
2. Encode via `bridge::encode_observation(&mut encoder, &obs, &safety, drawn)`
3. Map MJAI event to HydraAction via `action::mjai_to_hydra()`
4. Compute legal_mask via `action::compute_legal_mask(&obs)`
5. Store as MjaiSample

### 6.3 MjaiSample Struct

**MjaiSample**: One decision point.
- Fields: `obs: [f32; 2890]` (85*34 flat), `action: u8` (0-45), `legal_mask: [f32; 46]`, `placement: u8` (0-3), `score_delta: i32`, `grp_label: u8` (0-23), `tenpai: [f32; 3]`, `opp_next: [u8; 3]` (0-33 or 255=none), `danger: [f32; 3*34]`, `danger_mask: [f32; 3*34]`
- Size: ~12.4 KB per sample. At 70 decisions/game * 6M games = 420M samples.
- Stream from disk, do NOT load all into RAM.

### 6.4 GRP Label Construction

- 4! = 24 permutations. Sort players by score descending (ties broken by seat order).
- Precomputed lookup: `GRP_PERM_TABLE: [[u8; 4]; 24]` = all permutations of [0,1,2,3]
- Signature: `scores_to_grp_index(scores: [i32; 4]) -> u8`

### 6.5 Suit Permutation (6x augmentation)

- `augment_obs_suit(obs: &[f32; 2890], perm: &[u8; 3]) -> [f32; 2890]`: Permute suit indices for all 85 channels x 34 tiles using `permute_tile_type`.
- `augment_action_suit(action: u8, perm: &[u8; 3]) -> u8`: Permute discard actions (0-36) only; non-discard actions (37-45) unchanged.
- `augment_mask_suit(mask: &[f32; 46], perm: &[u8; 3]) -> [f32; 46]`: Permute first 37 entries; keep 37-45 unchanged.
- Use `hydra_core::tile::{ALL_PERMUTATIONS, permute_tile_type, permute_tile_extended}`. ALL_PERMUTATIONS has 6 entries (index 0 = identity).

### 6.6 Seat Rotation (4x augmentation)

Seat rotation is NOT a post-hoc transform on the obs tensor -- it requires RE-ENCODING the observation from scratch with a different observer. During replay, for each decision point, encode from ALL 4 seats. Channels that change: hand/drawn (0-8), discards (11-22), melds (23-34), metadata (43-61), safety (62-84). Total augmentation: 6 suits * 4 seats = 24x.

### 6.7 Batch Collation into Burn Tensors

**MjaiBatch<B: Backend>**:
- Fields: `obs: [batch, 85, 34]`, `actions: [batch] Int`, `legal_mask: [batch, 46]`, `value_target: [batch]`, `grp_target: [batch, 24]`, `tenpai_target: [batch, 3]`, `danger_target: [batch, 3, 34]`, `danger_mask: [batch, 3, 34]`, `opp_next_target: [batch, 3, 34]`, `score_pdf_target: [batch, 64]`, `score_cdf_target: [batch, 64]`

Collation: stack obs arrays -> reshape [B,85,34], convert action to Int tensor, grp_label -> one_hot [B,24], opp_next -> one_hot per opponent (mask 255 as zero), score_delta -> value target (div by 100,000), score_delta -> bin index for pdf, score_delta -> step function for cdf.

### 6.8 MUST NOT

- MUST NOT load entire dataset into RAM (stream via memory-mapped files or chunked reads)
- MUST NOT apply suit permutation to honor tiles (indices 27-33 pass through unchanged)
- MUST NOT confuse seat rotation with channel permutation (re-encode, don't shuffle channels)
- MUST NOT include samples where legal_mask is all zeros (skip non-decision states)
- MUST NOT forget to permute the action index when doing suit augmentation

### 6.9 Tests

| Test | Assertion |
|------|-----------|
| `test_load_single_game` | parse 1 .mjson.gz -> >50 decision points extracted |
| `test_augment_6x` | 6 suit perms on same obs -> 6 distinct obs (identity included) |
| `test_augment_preserves_honors` | suit perm does not move channels 27-33 |
| `test_batch_shapes` | batch=32 -> obs=[32,85,34], actions=[32], mask=[32,46] |
| `test_legal_mask_valid` | no sample in batch has all-zero mask |
| `test_seat_rotation_4x` | same decision point from 4 seats -> 4 different obs, hand channels differ |

---

## STEP 7: BC Training Loop

**File**: `hydra-train/src/training/bc.rs`
**Ref**: HYDRA_FINAL S11 Phase 0 (50 GPU hours, 24-block LearnerNet, 5-6M expert games)

### 7.1 BCTrainer Struct

**BCTrainer<B: AutodiffBackend>**:
- Fields: `model: HydraModel<B>`, `optimizer` (Adam), `lr: f64` (2.5e-4), `batch_size: usize` (2048), `loss: HydraLoss<B>`, `device: B::Device`

### 7.2 Optimizer Config

- Adam: beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-5 (light L2)
- LR scheduler: CosineAnnealing from 2.5e-4 to 1e-6

### 7.3 Epoch Loop

- Signature: `train_epoch(&mut self, loader: &mut MjaiDataLoader<B>) -> EpochStats`
- For each batch: forward -> total_loss (all 8 heads) -> backward -> gradient clip (max_norm=1.0) -> optimizer step -> track metrics (avg loss, policy agreement).
- `EpochStats`: `avg_loss: f64`, `policy_agreement: f64`

### 7.4 Policy Agreement Metric

- Signature: `policy_agreement(logits: [B,46], mask: [B,46], targets: [B] Int) -> f64`
- Mask illegal logits to -1e9 -> argmax -> compare to target -> mean accuracy.
- Expected range: 60-70% for strong BC on Houou data (Mortal reports ~65%).

### 7.5 Checkpoint Format (Burn Record)

- Use Burn's `NamedMpkFileRecorder::<FullPrecisionSettings>` -- saves all model weights + optimizer state as MessagePack (~40MB for 24-block).
- Save every epoch. Keep last 3 + best (by policy_agreement on eval set).
- **CheckpointMeta** (JSON sidecar): `epoch: u32`, `train_loss: f64`, `eval_agreement: f64`, `timestamp: u64`, `model_config: HydraModelConfig`

### 7.6 MUST NOT

- MUST NOT use multi-epoch over same batch (one pass per batch, reshuffle between epochs)
- MUST NOT skip gradient clipping (grad norm can spike on early batches)
- MUST NOT evaluate on training data (hold out 5% of games for eval)
- MUST NOT forget to apply legal_mask before argmax in agreement computation
- MUST NOT save optimizer state in bf16 (use f32 for optimizer moments)

### 7.7 Tests

| Test | Assertion |
|------|-----------|
| `test_bc_one_step` | single gradient step on random batch -> loss decreases |
| `test_bc_overfit_10_samples` | overfit 10 fixed samples in <100 steps -> loss < 0.1 |
| `test_checkpoint_save_load` | save, load, forward same input -> identical output (bitwise) |
| `test_policy_agreement_range` | random model agreement ~1/46 (~2.2%), not 0% or 100% |

---

## STEP 8: GAE + ACH + DRDA (CRITICAL)

**Files**: `hydra-train/src/training/gae.rs`, `ach.rs`, `drda.rs`
**Ref**: HYDRA_FINAL S11 Phase 2 (verbatim ACH update rule + DRDA wrapper)

### 8.1 GAE (Generalized Advantage Estimation)

**File**: `hydra-train/src/training/gae.rs`

- Signature: `compute_gae(rewards: &[f32], values: &[f32], dones: &[bool], gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>)` -- returns (advantages[T], returns[T])
- Hyperparameters: gamma=0.995, lambda=0.95
- `rewards`: [T] per-step rewards. `values`: [T+1] including bootstrap V(s_{T+1}). `dones`: [T] termination flags.
- Algorithm: reverse sweep. `delta = r[t] + gamma * V[t+1] * mask - V[t]`, `gae = delta + gamma * lambda * mask * gae`. Returns = advantages + values[0..T].
- `normalize_advantages(advantages: &mut [f32])`: subtract mean, divide by (std + 1e-8).
- Per-player GAE: run independently for each of 4 players, then normalize per-minibatch.

### 8.2 ACH (Actor-Critic Hedge) -- VERBATIM from HYDRA_FINAL S11 Phase 2

**File**: `hydra-train/src/training/ach.rs`

**Hyperparameters** (HYDRA_FINAL S11 Phase 2, line 310):
- `ETA = 1.0` (try {1, 2, 3}), `EPS = 0.5`, `L_TH = 8.0`, `BETA_ENT = 5e-4`, `LR = 2.5e-4`

**AchConfig**: `eta: f32` (1.0), `eps: f32` (0.5), `l_th: f32` (8.0), `beta_ent: f32` (5e-4)

**ACH loss formula**: `L_pi(s,a) = -c(s,a) * eta * (y(a|s;theta) / pi_old(a|s)) * A(s,a)`

**CRITICAL**: Uses raw logits `y(a)`, NOT log-probs. The ratio `y(a)/pi_old` is intentional -- this is what makes ACH different from PPO.

- Signature: `ach_policy_loss<B: AutodiffBackend>(logits: [B,46], legal_mask: [B,46], actions: [B] Int, pi_old: [B], advantages: [B], cfg: &AchConfig) -> scalar`

**ACH algorithm steps**:
1. Mask illegal actions to -1e9
2. Center logits: subtract mean of LEGAL logits
3. Clamp centered logits to [-L_TH, L_TH]
4. Softmax for current pi (over clamped+masked logits)
5. Gather y[a] and pi[a] for taken action
6. Ratio = pi(a|s) / pi_old(a|s)
7. Gate c (per-sample, 0 or 1):
   - When adv >= 0: c=1 iff ratio < 1+eps AND y[a] < l_th
   - When adv < 0: c=1 iff ratio > 1-eps AND y[a] > -l_th
8. ACH loss: `-mean(c * eta * (y[a] / pi_old) * advantage)`
9. Entropy bonus: `-beta_ent * mean(H(pi))` where `H = -sum(pi * log(pi) * legal_mask)`

**ONE epoch per batch** (HYDRA_FINAL S11 Phase 2, line 309): Collect batch -> compute GAE -> one forward+backward -> discard batch -> collect new batch.

### 8.3 DRDA (Dilated Regularized Dual Averaging)

**File**: `hydra-train/src/training/drda.rs`
**Ref**: HYDRA_FINAL S11 Phase 2 (lines 298-300); Farina et al. ICLR 2025

**Policy composition**: `pi_theta(a|x) = softmax(l_base(x,a) + y_theta(x,a) / tau_drda)`
- `l_base`: frozen logits from a checkpoint (detached, no grad)
- `y_theta`: trainable residual (model's output logits)
- `tau_drda = 4.0` (tune in {2, 4, 8}; target median KL to base in [0.05, 0.20])

**DrdaWrapper<B: Backend>**:
- Fields: `base_model: HydraModel<B>` (frozen, no grad), `tau_drda: f32` (4.0), `base_logits_cache: Option<Tensor<B, 2>>`
- `combined_logits(base: [B,46], residual: [B,46]) -> [B,46]`: Return `base + residual / tau_drda` (pre-softmax).

**Rebase procedure** (every 25-50 GPU hours -- CRITICAL):
1. Fold residual into base: `base_weight += model_weight / tau_drda`
2. Zero y_theta: set policy head final Linear weights/biases to 0
3. Reset optimizer moments (re-initialize Adam)
- This preserves pi exactly across boundaries (verify KL < 1e-6 before/after).

### 8.4 Full RL Training Step

- Signature: `rl_step<B: AutodiffBackend>(model, drda, batch, ach_cfg, optimizer)`
- Steps: compute base logits (frozen, no_grad) -> compute residual logits (trainable) -> combined_logits -> ACH loss -> value loss (w_v) -> auxiliary head losses -> total -> backward -> clip(1.0) -> optimizer step. ONE epoch, ONE pass.

### 8.5 MUST NOT

- MUST NOT use multiple epochs per batch (ACH theory requires ONE epoch)
- MUST NOT use log-probs in ACH loss (uses raw logits y(a), not log pi(a))
- MUST NOT forget to detach base_logits (no gradient through frozen model)
- MUST NOT skip advantage normalization (per-minibatch mean=0, std=1)
- MUST NOT apply DRDA rebase more often than every 25 GPU hours
- MUST NOT let tau_drda < 2.0 (policy diverges too far from base)
- MUST NOT confuse pi_old (rollout probability) with pi (current probability)
- MUST NOT backprop through the gate c (it's a hard 0/1 mask, not differentiable)

### 8.6 Tests

| Test | Assertion |
|------|-----------|
| `test_gae_simple` | 5-step trajectory with known r/V -> advantages match hand-computed values within 1e-4 |
| `test_gae_done_resets` | done=true at step 2 -> advantage at step 1 does NOT see reward at step 3 |
| `test_ach_gate_positive_adv` | adv>0, ratio=1.0, y_a=0.0 -> c=1.0 |
| `test_ach_gate_clips_ratio` | adv>0, ratio=1.6 (>1+0.5) -> c=0.0 |
| `test_ach_gate_clips_logit` | adv>0, y_a=9.0 (>L_TH=8) -> c=0.0 |
| `test_ach_gate_negative_adv` | adv<0, ratio=0.4 (<1-0.5) -> c=0.0 |
| `test_ach_one_epoch` | one forward+backward changes weights, second pass on same data FORBIDDEN |
| `test_drda_combined_logits` | base=[1,2,3], residual=[4,8,12], tau=4 -> combined=[2,4,6] |
| `test_drda_rebase_preserves_pi` | pi before rebase == pi after rebase (KL < 1e-6) |
| `test_drda_residual_zeroed` | after rebase, policy head final layer weights all 0.0 |
| `test_drda_optimizer_reset` | after rebase, Adam m and v moments all 0.0 |

# Hydra Build Plan: Steps 9-12 + Final Gate

## Step 9: CT-SMC Exact DP Sampler (Pure Rust, No Burn)

### File: `hydra-core/src/ct_smc.rs`

### 9.1 State Space

Hidden allocation matrix X: shape 34x4 (tile types x locations: opp1, opp2, opp3, wall).
Row sums r(k) = 4 - visible(k), column sums s(z) = concealed hand sizes + wall size.

**Key insight**: c_W = R_k - (c1 + c2 + c3) is derived, so DP state is 3D: (c1, c2, c3).
- Max concealed hand = 14 tiles. State space: each c_i in [0, 14], <= 15^3 = 3,375 states per DP layer.
- 34 tile-type layers, <= 35 compositions per layer. Total: ~4.0M ops.

### 9.2 Data Structures

**CtSmcConfig**: `num_particles: usize` (128-4096), `ess_threshold: f32` (0.4), `rng_seed: u64`

**Particle**: `allocation: [[u8; 4]; 34]` (X[k][z], z=0,1,2 are opponents, z=3 is wall), `log_weight: f64`

**Precomputed compositions**: `COMPOSITIONS: LazyLock<[Vec<[u8; 4]>; 5]>` -- for r in 0..=4, all (x0,x1,x2,x3) with sum=r. Sizes: r=0->1, r=1->4, r=2->10, r=3->20, r=4->35. Total: 70.

### 9.3 Exact DP Recurrence (Log-Space)

- Signature: `forward_dp(row_sums: &[u8; 34], col_sums: &[usize; 4], log_omega: &[[f64; 4]; 34]) -> Vec<FxHashMap<(u8,u8,u8), f64>>`
- Base: dp[34][(0,0,0)] = 0.0 (log 1), all others -inf.
- Recurrence: dp[k][(c1,c2,c3)] = logsumexp over compositions x of r(k) where `log_phi_k(x) = sum_j x[j] * log(omega[k][j])`, checking capacity constraints.
- Helper: `logsumexp2(a, b)` -- numerically stable log(exp(a) + exp(b)).

### 9.4 Exact Backward Sampling

- Signature: `backward_sample(dp, row_sums, col_sums, log_omega, rng) -> [[u8; 4]; 34]`
- For k=0..34: sample composition x_k ~ p(x_k | remaining capacity) where `p(x_k = x | c) = exp(log_phi_k(x) + dp[k+1][c-x] - dp[k][c])`.
- Exact: no rejection, no MCMC. One pass through 34 tile types.

### 9.5 SMC Reweighting + ESS + Resampling

**CtSmc**: `config: CtSmcConfig`, `particles: Vec<Particle>`, `dp_cache: Option<...>`
- `update(&mut self, row_sums, col_sums, log_omega, likelihood_fn, rng)`: Forward DP -> sample P particles -> weight by likelihood -> normalize -> ESS check -> systematic resample if ESS < threshold * P.
- `ess(&self) -> f32`: Standard ESS = 1 / sum(w_i^2).
- `systematic_resample(&mut self, rng)`: O(N), low variance.

### 9.6 Public API

In `hydra-core/src/lib.rs`: `pub mod ct_smc;`
Exports: `CtSmc`, `CtSmcConfig`, `Particle`

### 9.7 Tests + Benchmark

| Test | Assertion |
|------|-----------|
| `uniform_omega_matches_hypergeometric` | For r=[1,1,0,...] with uniform omega, marginals match analytic hypergeometric |
| `particles_satisfy_constraints` | Every particle satisfies row and column sum constraints |
| `uniform_likelihood_high_ess` | Uniform likelihood -> ESS close to P |

Benchmark: `bench_ct_smc_full_pipeline` -- median < 1ms over 1000 runs (forward DP + 128 backward samples).

---

## Step 10: AFBS Search Engine + Robust Opponent

### Files: `hydra-core/src/afbs.rs`, `hydra-core/src/robust_opponent.rs`

### 10.1 Search Tree Node

**AfbsNode**:
- Fields: `info_state_hash: u64` (Zobrist), `visit_count: u32`, `total_value: f64`, `prior: f32`, `children: SmallVec<[(u8, NodeIdx); 5]>`, `is_opponent: bool`, `particle_handle: Option<u32>`

**PUCT selection**: `UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))` where Q(a) = W(a)/N(a). c_puct = 2.5.

### 10.2 Top-K Expansion + Root Sampling

- `expand_node(node_idx, tree, policy_logits: [f32; 46], legal_mask: [bool; 46])`: Masked softmax -> take top K=5 by prior -> allocate child nodes.
- For opponent nodes: sample CT-SMC particle, condition ActorNet on particle's hand.

### 10.3 Batched Leaf Evaluation

**LeafBatch**: `obs_buffer: Vec<f32>` (N * 85 * 34), `node_indices: Vec<NodeIdx>`, `batch_size: usize`
- `MIN_BATCH = 32` for GPU efficiency. Returns (policy_logits[46], value[1]) per leaf.

### 10.4 Robust KL-Ball Opponent Nodes

**RobustOpponentConfig**: `epsilon: f32` (0.1-0.5), `tau_search_iters: u8` (20), `num_archetypes: usize` (4: aggressive, defensive, speed, value), `tau_arch: f32` (1.0)

- `find_robust_tau(p: &[f32], q_values: &[f32], epsilon: f32, iters: u8) -> (f32, Vec<f32>)`: Binary search for tau satisfying KL(q_tau || p) = epsilon. Returns (tau, q_tau).
- q_tau(a) = p(a) * exp(-Q(a)/tau) / Z(tau). KL is monotone decreasing in tau.
- Soft-min over archetypes: `Q_final(a) = -tau_arch * log(sum_i w_i * exp(-Q_i(a) / tau_arch))`

### 10.5 Pondering Manager

**PonderManager**: `cache: DashMap<u64, PonderResult>`, `priority_queue: BinaryHeap<PonderTask>`, `worker_handle: Option<JoinHandle<()>>`
**PonderResult**: `exit_policy: [f32; 46]`, `value: f32`, `search_depth: u8`, `visit_count: u32`, `timestamp: Instant`
**PonderTask**: `info_state_hash: u64`, `priority_score: f32`, `game_state_snapshot: GameStateSnapshot`
- Priority: high when top-2 policy gap < 0.1, risk > threshold, ESS low.

### 10.6 Tests

| Test | Assertion |
|------|-----------|
| `puct_selects_high_prior_unvisited` | UCB prefers high-prior unvisited child |
| `robust_tau_converges` | KL(q_tau||p) within 1% of epsilon after 20 iters |
| `archetype_softmin_reduces_to_uniform` | equal Q -> uniform weights |
| `ponder_cache_hit_reuses_search` | lookup returns stored result |
| `batched_eval_correct_size` | batch of 32 produces 32 outputs |

---

## Step 11: Self-Play Arena + Distillation + ExIt Pipeline

### Files: `hydra-core/src/arena.rs`, `hydra-train/src/distill.rs`, `hydra-train/src/exit.rs`

### 11.1 Arena: BatchSimulator Integration

**Arena**: `simulator: BatchSimulator`, `actor_weights: Arc<RwLock<ActorWeights>>`, `config: ArenaConfig`, `trajectory_buffer: Vec<Trajectory>`

**ArenaConfig**: `num_parallel_games: usize` (500+), `game_mode: u8` (0=hanchan), `temperature_range: (f32, f32)` (0.5, 1.5), `exit_fraction: f32` (0.2), `max_trajectory_buffer: usize`

Action sampling: per-seat temperature T ~ Uniform(0.5, 1.5), sampled once at game start. `a ~ Categorical(softmax(logits / T))`, illegal actions set to -inf.

### 11.2 Trajectory Struct

**TrajectoryStep** (`#[repr(C)]`):
- Fields: `obs: [f32; 85*34]`, `action: u8`, `pi_old: [f32; 46]`, `reward: f32`, `done: bool`, `player_id: u8`, `game_id: u32`, `turn: u16`, `temperature: f32`

**Trajectory**: `steps: Vec<TrajectoryStep>`, `final_scores: [i32; 4]`, `game_id: u32`, `seed: u64`

### 11.3 Distiller: Learner -> Actor

**DistillConfig**: `kd_kl_weight: f32` (1.0), `kd_mse_weight: f32` (0.5), `distill_lr: f32` (1e-4), `distill_batch_size: usize` (256), `update_interval_secs: u64` (60-120), `ema_decay: f32` (0.999)

Loss: `L_kd = KL(sg(learner_pi) || actor_pi) + 0.5 * MSE(sg(learner_v), actor_v)`

### 11.4 ExIt Pipeline

**ExitConfig**: `tau_exit: f32` (1.0), `exit_weight: f32` (0.5, annealed up in Phase 3), `min_visits: u32` (64), `hard_state_threshold: f32` (top-2 gap < 0.1), `safety_valve_max_kl: f32` (2.0)

- ExIt policy: `pi_exit(a|I) = softmax(Q(I,a) / tau_exit)` from AFBS.
- Safety valve: skip if visit_count < min_visits OR KL(exit||base) > max_kl.
- Combined loss: `L = L_ach + exit_weight*L_exit + saf_weight*L_saf + aux_weight*L_aux`

### 11.5 Tests

| Test | Assertion |
|------|-----------|
| `trajectory_roundtrip` | serialize/deserialize preserves all fields |
| `temperature_sampling_legal_only` | illegal action never selected |
| `distill_loss_zero_when_identical` | same weights -> L=0 |
| `exit_safety_valve_skips_low_visits` | N<64 -> ignored |
| `arena_500_games_completes` | 500 games with dummy policy complete without panic |

---

## Step 12: Remaining 7 Components

### 12.1 Endgame Solver

**File**: `hydra-core/src/endgame.rs`

**EndgameSolver**: `max_wall: u8`, `mass_threshold: f32`
- PIMC endgame solver. Trigger: wall <= 10 AND threat signal active.
- For each CT-SMC particle: sample ONE draw sequence + ONE opponent action sequence.
- Average Q over P particles (top-mass subset covering 95% weight, P~50-100).
- Signature: `solve_endgame(particles, game: &GameState, opponent: &RobustOpponent) -> [f32; 46]`
- Test: `endgame_improves_placement_accuracy` (50K positions, must beat vanilla AFBS)

### 12.2 Sinkhorn Belief (Mixture-SIB)

**File**: `hydra-core/src/sinkhorn.rs`

**SinkhornConfig**: `max_iters: u16`, `tol: f64`, `num_components: u8`
**MixtureSib**: `components: Vec<SibComponent>`, `weights: Vec<f64>`
- SIB: B* = diag(u) * K * diag(v) via Sinkhorn-Knopp iterations.
- Mixture-SIB: L components, Bayesian weight updates on observed events.
- Signature: `sinkhorn_project(kernel: &[f64; 34*4], row_sums, col_sums) -> [f64; 34*4]`
- Tests: `sinkhorn_converges_to_margins`, `mixture_weight_update_is_bayesian`

### 12.3 Hand-EV Oracle Features

**File**: `hydra-core/src/hand_ev.rs` (extends `shanten_batch.rs`)

**HandEvFeatures**: `tenpai_prob: [[f32;3];34]`, `win_prob: [[f32;3];34]`, `expected_score: [f32;34]`, `ukeire: [[f32;34];34]`
- CPU-precomputed per-discard features: P_tenpai(a,d) for d in {1,2,3}, P_win(a,d), E[score|win,a], ukeire vector.
- Signature: `compute_hand_ev(hand: &[u8;34], belief_remaining: &[f32;34]) -> HandEvFeatures`
- Tests: `tenpai_hand_has_high_p_tenpai`, `ukeire_sums_match_acceptance_count`

### 12.4 Search-as-Feature (SaF)

**File**: `hydra-train/src/saf.rs`

- Logit-residual: `l_final(a) = l_theta(a) + alpha * g_psi(f(a)) * m(a)`
- `f(a)` = [delta_Q, boole_risk, hunter_risk, robust_risk, entropy_drop, tau_robust, variance, ess] (8 features)
- `g_psi`: MLP(8->32->1). SaF-dropout p=0.3 during training.
- **SafConfig**: `alpha: f32`, `dropout: f32`, `hidden_dim: usize`
- Tests: `saf_dropout_zeros_features_at_rate`, `saf_logit_addition_correct`

### 12.5 Population League

**File**: `hydra-train/src/league.rs`

**League**: `agents: Vec<LeagueAgent>`, `elo_ratings: Vec<f32>`
**LeagueAgent**: `weights_path: PathBuf`, `agent_type: AgentType`
**AgentType**: `Current | Checkpoint(u32) | BcAnchor | Exploiter`
- Latest ActorNet, 3 trailing checkpoints, 2 BC-anchors, 1 exploiter. Uniform random matchmaking.
- Tests: `league_matchmaking_is_uniform`, `elo_updates_correctly`

### 12.6 Evaluation Harness

**File**: `hydra-train/src/eval.rs`

**EvalConfig**: `num_games: usize`, `opponents: Vec<PathBuf>`, `seed: u64`
**EvalResult**: `mean_placement: f32`, `stable_dan: f32`, `win_rate: f32`, `deal_in_rate: f32`, `tsumo_rate: f32`
- Signature: `evaluate(agent: &ActorNet, config: &EvalConfig) -> EvalResult`
- N=1000 hanchan games against fixed opponents (BC baseline, Mortal-level).
- Tests: `eval_deterministic_with_seed`, `eval_reports_all_metrics`

### 12.7 Inference Server

**File**: `hydra-train/src/inference.rs`

**InferenceServer**: `actor: ActorNet`, `ponder_cache: Arc<DashMap<u64, PonderResult>>`, `saf_mlp: SafMlp`
- Fast path: ActorNet forward + SaF adaptor (< 5ms). Slow path: reuse pondered AFBS subtree.
- On-turn budget: 80-150ms. Call reactions: 20-50ms. Agari guard: always-on final check.
- Signature: `infer(server, obs: &[f32; 85*34], legal: &[bool; 46]) -> (u8, [f32; 46])`
- Tests: `inference_respects_time_budget`, `agari_guard_prevents_illegal`

---

## Final Gate: End-to-End Integration Test

### File: `tests/integration_pipeline.rs`

Chains the entire pipeline with tiny models (2-block, 32 channels) on CPU:

**`full_pipeline_integration` test** -- verifies 8 stages:
1. BC warm start: train tiny LearnerNet on 100 synthetic games -> no NaN weights
2. Distill LearnerNet -> ActorNet -> no NaN weights
3. Self-play arena: 10 games with ActorNet -> 10 non-empty trajectories
4. ACH training step on trajectories -> no NaN weights
5. CT-SMC belief sampling -> 32 particles, all satisfy row/col sum constraints
6. AFBS search (beam_w=8, depth=2, top_k=3) -> visit_count >= 8, exit_policy sums to 1.0
7. ExIt training step -> no NaN weights
8. Final distill + inference -> picks legal action, policy sums to 1.0

**`edge_case_smoke_test`** -- near-endgame state (wall=5), CT-SMC + endgame solver produce finite Q values.

### Gate Pass Criteria

| Check | Criterion |
|-------|-----------|
| No panics | All 8 pipeline stages complete without panic |
| No NaN | Every weight tensor and output tensor is finite |
| Conservation | Every CT-SMC particle satisfies row/col sum constraints |
| Legal actions | Inference never selects an illegal action |
| Policy valid | All output policies sum to 1.0 (within epsilon=0.01) |
| Trajectory non-empty | Every arena game produces >= 1 decision step |
| Determinism | Same seed produces identical trajectories across 2 runs |
