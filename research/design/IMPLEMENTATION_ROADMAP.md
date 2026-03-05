# HYDRA Code Build Plan

**Goal**: Write all Rust/Burn code until `hydra-train` compiles, tests pass, and is ready to train.
**No GPU hours. No training. Just code that works.**

**End state**: `cargo build --release && cargo test --release && cargo clippy -- -D warnings` all green. Every HYDRA_FINAL component implemented and tested.

**Dev agent rules**:
- Every task ends with tests that PASS
- No `unwrap()` in library code (use `?` or explicit error handling)
- `cargo clippy -- -D warnings` clean after every task
- Read the referenced files BEFORE writing code
- Do NOT skip steps or proceed without passing the gate
- Do NOT invent your own approach -- follow the spec EXACTLY

**Design SSOT**: `research/design/HYDRA_FINAL.md` (406 lines)

---

# HYDRA Build Plan: Steps 1-4 (Neural Network Foundation)

> For an LLM coding agent. Every signature, shape, and value is EXACT.
> If something is not specified here, ASK -- do not invent.

---

## Step 1: Create hydra-train Crate

**Ref: HYDRA_FINAL Section 4, Section 11 (Training pipeline)**

### 1.1 Workspace Registration

Edit `/home/nikketryhard/dev/hydra/Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["hydra-core", "hydra-engine", "hydra-train"]
```

### 1.2 Crate Cargo.toml

Create `hydra-train/Cargo.toml`. Use whatever the latest `burn` version on
crates.io is at build time. The features below are the MINIMUM required:
```toml
[package]
name = "hydra-train"
version = "0.1.0"
edition = "2024"
license-file = "LICENSE"
description = "Hydra Riichi Mahjong AI - neural network and training"

[lib]
crate-type = ["rlib"]

[dependencies]
hydra-core = { path = "../hydra-core" }
burn = { version = "0.16", features = ["train", "tch"] }
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
burn = { version = "0.16", features = ["ndarray"] }
```

**IMPORTANT**: If `burn` 0.16 doesn't exist yet, use whatever the latest
stable release is. Check `crates.io/crates/burn` first. The API examples
below use `burn::prelude::*` which re-exports `Backend`, `Tensor`, `Module`,
`Config`, and common nn modules.

### 1.3 Module Structure

Create these files:
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

`src/lib.rs` contents:
```rust
//! Hydra neural network: SE-ResNet backbone with multi-task heads.
pub mod backbone;
pub mod config;
pub mod heads;
pub mod model;
```

### 1.4 Constants (in `config.rs`)

These are the EXACT defaults from HYDRA_FINAL Section 4:
```rust
pub const INPUT_CHANNELS: usize = 85;   // from hydra_core::encoder::NUM_CHANNELS
pub const TILE_DIM: usize = 34;         // from hydra_core::encoder::NUM_TILES
pub const HIDDEN_CHANNELS: usize = 256; // SE-ResNet channel width
pub const SE_REDUCTION: usize = 4;      // SE squeeze ratio: 256/4 = 64
pub const SE_BOTTLENECK: usize = HIDDEN_CHANNELS / SE_REDUCTION; // = 64
pub const NUM_GROUPS: usize = 32;       // GroupNorm groups (Section 4.2)
pub const ACTION_SPACE: usize = 46;     // from hydra_core::action::HYDRA_ACTION_SPACE
pub const SCORE_BINS: usize = 64;       // Section 4.3: KataGo-style score dist
pub const NUM_OPPONENTS: usize = 3;     // 4-player game minus self
pub const GRP_CLASSES: usize = 24;      // Global reward prediction classes
```

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

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct SEBlock<B: Backend> {
    fc1: Linear<B>,  // Linear(256, 64)
    fc2: Linear<B>,  // Linear(64, 256)
}

#[derive(Config, Debug)]
pub struct SEBlockConfig {
    #[config(default = "256")]
    pub channels: usize,
    #[config(default = "64")]
    pub bottleneck: usize,
}
```

**`init` method:**
```rust
impl SEBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEBlock<B> {
        SEBlock {
            fc1: LinearConfig::new(self.channels, self.bottleneck).init(device),
            fc2: LinearConfig::new(self.bottleneck, self.channels).init(device),
        }
    }
}
```

**`forward` method -- EXACT tensor shapes in comments:**
```rust
impl<B: Backend> SEBlock<B> {
    /// Input: [B, 256, 34] -> Output: [B, 256, 34]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [B, 256, 34] -> mean over dim 2 -> [B, 256, 1]
        let se = x.clone().mean_dim(2);
        // [B, 256, 1] -> squeeze dim 2 -> [B, 256]
        let se = se.squeeze::<2>(2);
        // [B, 256] -> Linear(256, 64) -> [B, 64]
        let se = self.fc1.forward(se);
        // [B, 64] -> Mish -> [B, 64]
        let se = burn::tensor::activation::mish(se);
        // [B, 64] -> Linear(64, 256) -> [B, 256]
        let se = self.fc2.forward(se);
        // [B, 256] -> Sigmoid -> [B, 256]
        let se = burn::tensor::activation::sigmoid(se);
        // [B, 256] -> unsqueeze dim 2 -> [B, 256, 1]
        let se = se.unsqueeze_dim::<3>(2);
        // [B, 256, 34] * [B, 256, 1] -> broadcast -> [B, 256, 34]
        x * se
    }
}
```

**MUST NOT:**
- Do NOT use `AdaptiveAvgPool1d` for the global pool. Use `mean_dim(2)`.
- Do NOT use `Relu` inside SE. Use `mish` for fc1 activation, `sigmoid` for gate.
- Do NOT forget the `x.clone()` -- Burn tensors move on use.

### 2.2 SEResBlock (Single Residual Block)

```rust
use burn::nn::{GroupNorm, GroupNormConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;

#[derive(Module, Debug)]
pub struct SEResBlock<B: Backend> {
    gn1: GroupNorm<B>,   // GroupNorm(32, 256)
    conv1: Conv1d<B>,    // Conv1d(256, 256, kernel=3, padding=Same)
    gn2: GroupNorm<B>,   // GroupNorm(32, 256)
    conv2: Conv1d<B>,    // Conv1d(256, 256, kernel=3, padding=Same)
    se: SEBlock<B>,
}

#[derive(Config, Debug)]
pub struct SEResBlockConfig {
    #[config(default = "256")]
    pub channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
}
```

**`init` method:**
```rust
impl SEResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEResBlock<B> {
        let ch = self.channels;
        SEResBlock {
            gn1: GroupNormConfig::new(self.num_groups, ch).init(device),
            conv1: Conv1dConfig::new(ch, ch, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            gn2: GroupNormConfig::new(self.num_groups, ch).init(device),
            conv2: Conv1dConfig::new(ch, ch, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            se: SEBlockConfig { channels: ch, bottleneck: self.se_bottleneck }
                .init(device),
        }
    }
}
```

**`forward` method -- pre-activation residual (GroupNorm -> Mish -> Conv):**
```rust
impl<B: Backend> SEResBlock<B> {
    /// Input: [B, 256, 34] -> Output: [B, 256, 34]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        // Pre-activation: GN -> Mish -> Conv -> GN -> Mish -> Conv -> SE
        let out = self.gn1.forward(x);
        let out = burn::tensor::activation::mish(out);
        let out = self.conv1.forward(out);
        let out = self.gn2.forward(out);
        let out = burn::tensor::activation::mish(out);
        let out = self.conv2.forward(out);
        let out = self.se.forward(out);
        // Residual connection
        out + residual
    }
}
```

**MUST NOT:**
- Do NOT use post-activation (Conv -> GN -> Mish). Use PRE-activation.
- Do NOT add a projection/downsample path. All blocks are same-dim (256->256).
- Do NOT use `bias=false` on Conv1d. Keep default `bias=true`.
- The spatial dim MUST stay 34 through every block (padding=Same guarantees this).

### 2.3 SEResNet (Full Backbone)

```rust
#[derive(Module, Debug)]
pub struct SEResNet<B: Backend> {
    /// Initial projection: Conv1d(85, 256, kernel=3, padding=Same)
    input_conv: Conv1d<B>,
    input_gn: GroupNorm<B>,  // GroupNorm(32, 256)
    /// N residual blocks (N=12 for ActorNet, N=24 for LearnerNet)
    blocks: Vec<SEResBlock<B>>,
    /// Final normalization before heads
    final_gn: GroupNorm<B>,  // GroupNorm(32, 256)
}

#[derive(Config, Debug)]
pub struct SEResNetConfig {
    /// Number of residual blocks. 12 for ActorNet, 24 for LearnerNet.
    pub num_blocks: usize,
    #[config(default = "85")]
    pub input_channels: usize,
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
}
```

**`init` method:**
```rust
impl SEResNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SEResNet<B> {
        let ch = self.hidden_channels;
        let blocks = (0..self.num_blocks)
            .map(|_| SEResBlockConfig {
                channels: ch,
                num_groups: self.num_groups,
                se_bottleneck: self.se_bottleneck,
            }.init(device))
            .collect();
        SEResNet {
            input_conv: Conv1dConfig::new(self.input_channels, ch, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            input_gn: GroupNormConfig::new(self.num_groups, ch).init(device),
            blocks,
            final_gn: GroupNormConfig::new(self.num_groups, ch).init(device),
        }
    }
}
```

**`forward` method:**
```rust
impl<B: Backend> SEResNet<B> {
    /// Input: [B, 85, 34] -> spatial: [B, 256, 34], pooled: [B, 256]
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        // [B, 85, 34] -> Conv1d(85, 256, k=3, pad=Same) -> [B, 256, 34]
        let mut x = self.input_conv.forward(x);
        x = self.input_gn.forward(x);
        x = burn::tensor::activation::mish(x);
        // N residual blocks: [B, 256, 34] -> [B, 256, 34]
        for block in &self.blocks {
            x = block.forward(x);
        }
        // Final norm + activation
        let spatial = self.final_gn.forward(x);
        let spatial = burn::tensor::activation::mish(spatial);
        // Global average pool: [B, 256, 34] -> mean(dim=2) -> [B, 256, 1] -> squeeze -> [B, 256]
        let pooled = spatial.clone().mean_dim(2).squeeze::<2>(2);
        (spatial, pooled)
    }
}
```

**Returns a TUPLE**: `(spatial, pooled)` because heads need both:
- `spatial: [B, 256, 34]` -- for per-tile heads (danger, opp_next_discard)
- `pooled: [B, 256]` -- for scalar/vector heads (policy, value, etc.)

**MUST NOT:**
- Do NOT flatten the spatial tensor. Keep it as [B, 256, 34].
- Do NOT apply final_gn INSIDE the loop. Apply it ONCE after all blocks.
- Do NOT return only pooled. Both spatial and pooled are needed.
- Do NOT hardcode num_blocks. It's configurable (12 or 24).

### 2.4 Tests for Step 2

File: `hydra-train/src/backbone.rs` (in `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray;

    #[test]
    fn se_block_preserves_shape() {
        let device = Default::default();
        let se = SEBlockConfig { channels: 256, bottleneck: 64 }.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        let out = se.forward(x);
        assert_eq!(out.dims(), [4, 256, 34]);
    }

    #[test]
    fn se_res_block_preserves_shape() {
        let device = Default::default();
        let block = SEResBlockConfig {
            channels: 256, num_groups: 32, se_bottleneck: 64,
        }.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &device);
        let out = block.forward(x);
        assert_eq!(out.dims(), [4, 256, 34]);
    }

    #[test]
    fn backbone_output_shapes_12_blocks() {
        let device = Default::default();
        let net = SEResNetConfig {
            num_blocks: 12,
            input_channels: 85,
            hidden_channels: 256,
            num_groups: 32,
            se_bottleneck: 64,
        }.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 85, 34], &device);
        let (spatial, pooled) = net.forward(x);
        assert_eq!(spatial.dims(), [4, 256, 34]);
        assert_eq!(pooled.dims(), [4, 256]);
    }

    #[test]
    fn backbone_output_shapes_24_blocks() {
        let device = Default::default();
        let net = SEResNetConfig {
            num_blocks: 24,
            input_channels: 85,
            hidden_channels: 256,
            num_groups: 32,
            se_bottleneck: 64,
        }.init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, 85, 34], &device);
        let (spatial, pooled) = net.forward(x);
        assert_eq!(spatial.dims(), [2, 256, 34]);
        assert_eq!(pooled.dims(), [2, 256]);
    }
}
```

---

## Step 3: Output Heads

**Ref: HYDRA_FINAL Section 4.3**

All code goes in `hydra-train/src/heads.rs`.

There are 8 inference heads + 1 oracle-only head = 9 total. Each head is a
separate struct. They share NO weights. Every head takes either `pooled: [B, 256]`
or `spatial: [B, 256, 34]` from the backbone.

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
| 9 | OracleCriticHead | pooled [B,256] | Linear(256,1) | [B, 1] | none (raw) |

### 3.2 Struct Definitions (ALL heads)

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};

// --- Head 1: Policy ---
#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 46)
}
// forward: pooled [B, 256] -> linear -> [B, 46]  (raw logits, no activation)

// --- Head 2: Value ---
#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 1)
}
// forward: pooled [B, 256] -> linear -> [B, 1] -> tanh -> [B, 1]

// --- Head 3: Score PDF ---
#[derive(Module, Debug)]
pub struct ScorePdfHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 64)
}
// forward: pooled [B, 256] -> linear -> [B, 64]  (apply log_softmax at loss time)

// --- Head 4: Score CDF ---
#[derive(Module, Debug)]
pub struct ScoreCdfHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 64)
}
// forward: pooled [B, 256] -> linear -> [B, 64] -> sigmoid -> [B, 64]

// --- Head 5: Opponent Tenpai ---
#[derive(Module, Debug)]
pub struct OppTenpaiHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 3)
}
// forward: pooled [B, 256] -> linear -> [B, 3] -> sigmoid -> [B, 3]

// --- Head 6: Global Reward Prediction ---
#[derive(Module, Debug)]
pub struct GrpHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 24)
}
// forward: pooled [B, 256] -> linear -> [B, 24]  (raw logits for classification)

// --- Head 7: Opponent Next Discard (per-tile) ---
#[derive(Module, Debug)]
pub struct OppNextDiscardHead<B: Backend> {
    conv: Conv1d<B>,  // Conv1d(256, 3, kernel_size=1)
}
// forward: spatial [B, 256, 34] -> conv1d(k=1) -> [B, 3, 34]

// --- Head 8: Danger (per-tile deal-in probability) ---
#[derive(Module, Debug)]
pub struct DangerHead<B: Backend> {
    conv: Conv1d<B>,  // Conv1d(256, 3, kernel_size=1)
}
// forward: spatial [B, 256, 34] -> conv1d(k=1) -> [B, 3, 34] -> sigmoid

// --- Head 9: Oracle Critic (training only) ---
#[derive(Module, Debug)]
pub struct OracleCriticHead<B: Backend> {
    linear: Linear<B>,  // Linear(256, 1)
}
// forward: pooled [B, 256] -> linear -> [B, 1]  (no activation -- raw value)
```

### 3.3 Config and Init

Create a single config that initializes ALL heads:

```rust
#[derive(Config, Debug)]
pub struct HeadsConfig {
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "46")]
    pub action_space: usize,
    #[config(default = "64")]
    pub score_bins: usize,
    #[config(default = "3")]
    pub num_opponents: usize,
    #[config(default = "24")]
    pub grp_classes: usize,
}
```

Each head has a trivial init. Example for PolicyHead:
```rust
impl HeadsConfig {
    pub fn init_policy<B: Backend>(&self, device: &B::Device) -> PolicyHead<B> {
        PolicyHead {
            linear: LinearConfig::new(self.hidden_channels, self.action_space)
                .init(device),
        }
    }
    // ... same pattern for all 9 heads
}
```

For Conv1d heads (OppNextDiscard, Danger), kernel_size=1, NO padding needed:
```rust
    pub fn init_opp_next_discard<B: Backend>(&self, d: &B::Device) -> OppNextDiscardHead<B> {
        OppNextDiscardHead {
            conv: Conv1dConfig::new(self.hidden_channels, self.num_opponents, 1)
                .init(d),
        }
    }
```

### 3.4 Forward Methods -- EXACT Signatures

```rust
impl<B: Backend> PolicyHead<B> {
    /// pooled [B, 256] -> [B, 46] raw logits
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

impl<B: Backend> ValueHead<B> {
    /// pooled [B, 256] -> [B, 1] tanh-squashed value
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::tanh(self.linear.forward(pooled))
    }
}

impl<B: Backend> ScorePdfHead<B> {
    /// pooled [B, 256] -> [B, 64] raw logits (apply log_softmax at loss time)
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

impl<B: Backend> ScoreCdfHead<B> {
    /// pooled [B, 256] -> [B, 64] monotonic CDF via sigmoid
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::sigmoid(self.linear.forward(pooled))
    }
}

impl<B: Backend> OppTenpaiHead<B> {
    /// pooled [B, 256] -> [B, 3] independent probabilities
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::sigmoid(self.linear.forward(pooled))
    }
}

impl<B: Backend> GrpHead<B> {
    /// pooled [B, 256] -> [B, 24] raw logits
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}

impl<B: Backend> OppNextDiscardHead<B> {
    /// spatial [B, 256, 34] -> [B, 3, 34] raw logits per opponent per tile
    pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
        self.conv.forward(spatial)
    }
}

impl<B: Backend> DangerHead<B> {
    /// spatial [B, 256, 34] -> [B, 3, 34] deal-in probabilities
    pub fn forward(&self, spatial: Tensor<B, 3>) -> Tensor<B, 3> {
        burn::tensor::activation::sigmoid(self.conv.forward(spatial))
    }
}

impl<B: Backend> OracleCriticHead<B> {
    /// pooled [B, 256] -> [B, 1] raw value (oracle-only, not used at inference)
    pub fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(pooled)
    }
}
```

### 3.5 MUST NOT

- Do NOT apply softmax to PolicyHead output. That happens at loss/sampling time.
- Do NOT apply softmax to GrpHead output. Same reason.
- Do NOT apply log_softmax to ScorePdfHead in forward. Only at loss time.
- Do NOT use kernel_size=3 for OppNextDiscard/Danger. Use kernel_size=1 (pointwise).
- Do NOT share any weights between heads. Each is independent.
- Do NOT add hidden layers inside heads. Each head is a SINGLE Linear or Conv1d.
- Do NOT apply activation to OracleCriticHead. It's a raw value estimate.
- Value head uses `tanh`, NOT `sigmoid`. Value range is [-1, 1] not [0, 1].

### 3.6 Tests for Step 3

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray;

    fn default_config() -> HeadsConfig {
        HeadsConfig {
            hidden_channels: 256,
            action_space: 46,
            score_bins: 64,
            num_opponents: 3,
            grp_classes: 24,
        }
    }

    #[test]
    fn policy_head_shape() {
        let d = Default::default();
        let head = default_config().init_policy::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        assert_eq!(head.forward(x).dims(), [4, 46]);
    }

    #[test]
    fn value_head_shape_and_range() {
        let d = Default::default();
        let head = default_config().init_value::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        let out = head.forward(x);
        assert_eq!(out.dims(), [4, 1]);
        // tanh output must be in [-1, 1]
        let data = out.into_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn score_pdf_head_shape() {
        let d = Default::default();
        let head = default_config().init_score_pdf::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        assert_eq!(head.forward(x).dims(), [4, 64]);
    }

    #[test]
    fn score_cdf_head_range() {
        let d = Default::default();
        let head = default_config().init_score_cdf::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        let out = head.forward(x);
        assert_eq!(out.dims(), [4, 64]);
        // sigmoid output in [0, 1]
        let data = out.into_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn opp_tenpai_head_shape() {
        let d = Default::default();
        let head = default_config().init_opp_tenpai::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        assert_eq!(head.forward(x).dims(), [4, 3]);
    }

    #[test]
    fn grp_head_shape() {
        let d = Default::default();
        let head = default_config().init_grp::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        assert_eq!(head.forward(x).dims(), [4, 24]);
    }

    #[test]
    fn opp_next_discard_head_shape() {
        let d = Default::default();
        let head = default_config().init_opp_next_discard::<B>(&d);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &d);
        assert_eq!(head.forward(x).dims(), [4, 3, 34]);
    }

    #[test]
    fn danger_head_shape_and_range() {
        let d = Default::default();
        let head = default_config().init_danger::<B>(&d);
        let x = Tensor::<B, 3>::zeros([4, 256, 34], &d);
        let out = head.forward(x);
        assert_eq!(out.dims(), [4, 3, 34]);
        let data = out.into_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn oracle_critic_head_shape() {
        let d = Default::default();
        let head = default_config().init_oracle_critic::<B>(&d);
        let x = Tensor::<B, 2>::zeros([4, 256], &d);
        assert_eq!(head.forward(x).dims(), [4, 1]);
    }
}
```

---

## Step 4: Full HydraModel

**Ref: HYDRA_FINAL Section 4.2 + 4.3**

All code goes in `hydra-train/src/model.rs`.

### 4.1 HydraOutput Struct

The model returns ALL head outputs in a single struct, NOT a tuple:

```rust
/// All outputs from a single forward pass.
pub struct HydraOutput<B: Backend> {
    /// Policy logits: [B, 46]. Apply masked softmax externally.
    pub policy_logits: Tensor<B, 2>,
    /// Value in [-1, 1]: [B, 1].
    pub value: Tensor<B, 2>,
    /// Score PDF logits: [B, 64]. Apply log_softmax at loss time.
    pub score_pdf: Tensor<B, 2>,
    /// Score CDF probabilities: [B, 64]. Already sigmoided.
    pub score_cdf: Tensor<B, 2>,
    /// Opponent tenpai probabilities: [B, 3]. Already sigmoided.
    pub opp_tenpai: Tensor<B, 2>,
    /// Global reward prediction logits: [B, 24].
    pub grp: Tensor<B, 2>,
    /// Opponent next discard logits: [B, 3, 34].
    pub opp_next_discard: Tensor<B, 3>,
    /// Danger (deal-in) probabilities: [B, 3, 34]. Already sigmoided.
    pub danger: Tensor<B, 3>,
    /// Oracle critic value: [B, 1]. Only meaningful during oracle training.
    pub oracle_critic: Tensor<B, 2>,
}
```

### 4.2 HydraModel Struct

```rust
use crate::backbone::{SEResNet, SEResNetConfig};
use crate::heads::*;

#[derive(Module, Debug)]
pub struct HydraModel<B: Backend> {
    pub backbone: SEResNet<B>,
    pub policy: PolicyHead<B>,
    pub value: ValueHead<B>,
    pub score_pdf: ScorePdfHead<B>,
    pub score_cdf: ScoreCdfHead<B>,
    pub opp_tenpai: OppTenpaiHead<B>,
    pub grp: GrpHead<B>,
    pub opp_next_discard: OppNextDiscardHead<B>,
    pub danger: DangerHead<B>,
    pub oracle_critic: OracleCriticHead<B>,
}
```

### 4.3 HydraModelConfig

```rust
#[derive(Config, Debug)]
pub struct HydraModelConfig {
    /// 12 for ActorNet, 24 for LearnerNet
    pub num_blocks: usize,
    #[config(default = "85")]
    pub input_channels: usize,
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
    #[config(default = "46")]
    pub action_space: usize,
    #[config(default = "64")]
    pub score_bins: usize,
    #[config(default = "3")]
    pub num_opponents: usize,
    #[config(default = "24")]
    pub grp_classes: usize,
}

impl HydraModelConfig {
    /// Convenience: create ActorNet config (12 blocks, ~5M params)
    pub fn actor() -> Self {
        Self { num_blocks: 12, ..Default::default() }
    }

    /// Convenience: create LearnerNet config (24 blocks, ~10M params)
    pub fn learner() -> Self {
        Self { num_blocks: 24, ..Default::default() }
    }
}
```

**NOTE:** `HydraModelConfig` must `impl Default` manually OR the `#[config]`
defaults handle it. If using Burn's `Config` derive, provide a `new(num_blocks)`
constructor since `num_blocks` has no default:

```rust
// Usage:
let config = HydraModelConfig::new(12);  // ActorNet
let config = HydraModelConfig::new(24);  // LearnerNet
```

### 4.4 Init Method

```rust
impl HydraModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HydraModel<B> {
        let heads_cfg = HeadsConfig {
            hidden_channels: self.hidden_channels,
            action_space: self.action_space,
            score_bins: self.score_bins,
            num_opponents: self.num_opponents,
            grp_classes: self.grp_classes,
        };
        HydraModel {
            backbone: SEResNetConfig {
                num_blocks: self.num_blocks,
                input_channels: self.input_channels,
                hidden_channels: self.hidden_channels,
                num_groups: self.num_groups,
                se_bottleneck: self.se_bottleneck,
            }.init(device),
            policy: heads_cfg.init_policy(device),
            value: heads_cfg.init_value(device),
            score_pdf: heads_cfg.init_score_pdf(device),
            score_cdf: heads_cfg.init_score_cdf(device),
            opp_tenpai: heads_cfg.init_opp_tenpai(device),
            grp: heads_cfg.init_grp(device),
            opp_next_discard: heads_cfg.init_opp_next_discard(device),
            danger: heads_cfg.init_danger(device),
            oracle_critic: heads_cfg.init_oracle_critic(device),
        }
    }
}
```

### 4.5 Forward Method

```rust
impl<B: Backend> HydraModel<B> {
    /// Full forward pass.
    /// Input: [B, 85, 34] observation tensor from hydra_core::encoder.
    /// Returns: HydraOutput with all 9 head outputs.
    pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
        // Backbone: [B, 85, 34] -> spatial [B, 256, 34] + pooled [B, 256]
        let (spatial, pooled) = self.backbone.forward(x);
        HydraOutput {
            policy_logits: self.policy.forward(pooled.clone()),
            value: self.value.forward(pooled.clone()),
            score_pdf: self.score_pdf.forward(pooled.clone()),
            score_cdf: self.score_cdf.forward(pooled.clone()),
            opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
            grp: self.grp.forward(pooled.clone()),
            opp_next_discard: self.opp_next_discard.forward(spatial.clone()),
            danger: self.danger.forward(spatial),
            oracle_critic: self.oracle_critic.forward(pooled),
        }
    }
}
```

**CRITICAL**: The last uses of `pooled` and `spatial` must NOT be cloned
(Burn moves on use). Count the clones: `pooled` is used 7 times (clone 6,
move last), `spatial` is used 2 times (clone 1, move last).

### 4.6 MUST NOT

- Do NOT run the backbone twice (once per head). Run it ONCE, share outputs.
- Do NOT add dropout to the model. Dropout is NOT in HYDRA_FINAL architecture.
- Do NOT forget to clone tensors. Burn moves on use. Last consumer gets the move.
- Do NOT make HydraOutput generic over which heads are present. Always return all 9.
- Do NOT add a `training: bool` parameter to forward. Oracle critic is always computed;
  it's simply ignored at inference time.
- Do NOT reshape the input inside forward. The caller provides [B, 85, 34] directly.

### 4.7 Tests for Step 4

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray;

    #[test]
    fn actor_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::new(12).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([4, 85, 34], &device);
        let out = model.forward(x);
        assert_eq!(out.policy_logits.dims(), [4, 46]);
        assert_eq!(out.value.dims(), [4, 1]);
        assert_eq!(out.score_pdf.dims(), [4, 64]);
        assert_eq!(out.score_cdf.dims(), [4, 64]);
        assert_eq!(out.opp_tenpai.dims(), [4, 3]);
        assert_eq!(out.grp.dims(), [4, 24]);
        assert_eq!(out.opp_next_discard.dims(), [4, 3, 34]);
        assert_eq!(out.danger.dims(), [4, 3, 34]);
        assert_eq!(out.oracle_critic.dims(), [4, 1]);
    }

    #[test]
    fn learner_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::new(24).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, 85, 34], &device);
        let out = model.forward(x);
        assert_eq!(out.policy_logits.dims(), [2, 46]);
        assert_eq!(out.value.dims(), [2, 1]);
        assert_eq!(out.score_pdf.dims(), [2, 64]);
        assert_eq!(out.score_cdf.dims(), [2, 64]);
        assert_eq!(out.opp_tenpai.dims(), [2, 3]);
        assert_eq!(out.grp.dims(), [2, 24]);
        assert_eq!(out.opp_next_discard.dims(), [2, 3, 34]);
        assert_eq!(out.danger.dims(), [2, 3, 34]);
        assert_eq!(out.oracle_critic.dims(), [2, 1]);
    }

    #[test]
    fn value_head_bounded() {
        let device = Default::default();
        let model = HydraModelConfig::new(12).init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [8, 85, 34], burn::tensor::Distribution::Normal(0.0, 1.0), &device,
        );
        let out = model.forward(x);
        let data = out.value.into_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v >= -1.0 && v <= 1.0, "value {} out of tanh range", v);
        }
    }

    #[test]
    fn danger_head_bounded() {
        let device = Default::default();
        let model = HydraModelConfig::new(12).init::<B>(&device);
        let x = Tensor::<B, 3>::random(
            [4, 85, 34], burn::tensor::Distribution::Normal(0.0, 1.0), &device,
        );
        let out = model.forward(x);
        let data = out.danger.into_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v >= 0.0 && v <= 1.0, "danger {} out of sigmoid range", v);
        }
    }
}
```

---

## Global MUST NOT (All Steps)

1. **No BatchNorm.** Always GroupNorm(32). Ref: Section 4.2.
2. **No Relu/Gelu.** Always Mish. Ref: Section 4.2.
3. **No bf16 yet.** Build everything in f32 first. bf16 is a later optimization.
4. **No async.** All forward passes are synchronous.
5. **No Python.** 100% Rust. Ref: AGENTS.md.
6. **No Mortal code.** Do not look at or copy from `Mortal-Policy/`. Ref: AGENTS.md.
7. **Test with NdArray backend.** Tests use `burn::backend::NdArray`, not `Tch`.
8. **Burn API pitfall**: `Tensor` moves on use. Clone before reuse.
9. **Burn API pitfall**: `GroupNormConfig::new(num_groups, num_channels)` -- groups
   first, channels second. Do NOT reverse them.
10. **Burn API pitfall**: `Conv1dConfig::new(channels_in, channels_out, kernel_size)` --
    these are positional, NOT named. Do NOT pass `[in, out]` as an array.

---

## Quick Reference: Import Map

From `hydra-core` (used in future steps, not needed for model definition):
```rust
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};  // 85, 34, 2890
use hydra_core::action::HYDRA_ACTION_SPACE;                     // 46
```

From `burn`:
```rust
use burn::prelude::*;  // Backend, Tensor, Module, Config
use burn::nn::{Linear, LinearConfig, GroupNorm, GroupNormConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation;  // activation::mish, activation::sigmoid, activation::tanh
```
# Hydra Implementation Roadmap: Steps 5-8

## STEP 5: Loss Functions

**File**: `hydra-train/src/training/losses.rs`
**Ref**: HYDRA_FINAL S4.3, S11 Phase 0/2; HYDRA_SPEC Value/GRP/Tenpai/Danger heads

### 5.1 Struct

```rust
pub struct HydraLoss<B: Backend> {
    pub w_pi:     f32,  // 1.0
    pub w_v:      f32,  // 0.5
    pub w_grp:    f32,  // 0.2
    pub w_tenpai: f32,  // 0.1
    pub w_danger: f32,  // 0.1
    pub w_opp:    f32,  // 0.1
    pub w_score:  f32,  // 0.05  (covers BOTH pdf + cdf, split 50/50 internally)
    _backend: PhantomData<B>,
}
```

### 5.2 Per-Head Loss Functions

| Function | Signature | Shape In | Loss Type | Notes |
|----------|-----------|----------|-----------|-------|
| `policy_ce` | `(logits: [B,46], target: [B,46], mask: [B,46]) -> [B]` | logits=[B,46], target=[B,46] soft | CE with soft targets | `target` is one-hot for BC, soft distribution for ExIt. Masked softmax: set illegal logit to -1e9 before softmax. `loss = -sum(target * log_softmax(masked_logits))` |
| `value_mse` | `(pred: [B], target: [B]) -> [B]` | scalars | MSE | `0.5 * (pred - target)^2` |
| `grp_ce` | `(logits: [B,24], target: [B,24]) -> [B]` | 24-way permutation | CE with soft targets | Same formula as policy_ce but over 24 classes. Target is one-hot (BC) or soft (ExIt). |
| `tenpai_bce` | `(logits: [B,3], target: [B,3]) -> [B]` | 3 opponents | BCE | `F::binary_cross_entropy_with_logits`. Per-opponent independent sigmoid. |
| `danger_focal_bce` | `(logits: [B,3,34], target: [B,3,34], mask: [B,3,34]) -> [B]` | 3x34 | Focal BCE | alpha=0.25, gamma=2.0. Ref: OPPONENT_MODELING.md danger head. Mask=0 where label unavailable. |
| `opp_next_ce` | `(logits: [B,3,34], target: [B,3,34]) -> [B]` | per-opp 34-way | CE per opponent | 3 independent 34-way CEs, summed. target is one-hot of next discard tile type. |
| `score_pdf_ce` | `(logits: [B,64], target: [B,64]) -> [B]` | 64 bins | CE | Softmax over 64 score bins. Target is one-hot bin index. |
| `score_cdf_bce` | `(logits: [B,64], target: [B,64]) -> [B]` | 64 bins | BCE | Each bin: P(score <= threshold_i). Target is step function. |

### 5.3 Oracle Critic Loss (Phase 1+)

```rust
pub fn oracle_critic_loss(v_oracle: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1>
// v_oracle: [B, 4] -- one value per player from oracle critic
// target:   [B, 4] -- actual per-player rewards
// 1) MSE loss: 0.5 * mean((v_oracle - target)^2)
// 2) Zero-sum penalty: mean((v_oracle.sum(dim=1))^2) * 10.0
//    Enforces V1+V2+V3+V4 = 0 (HYDRA_FINAL S11 Phase 1, HYDRA_SPEC S Value Head)
// 3) Normalize: subtract mean across players before MSE
//    v_norm = v_oracle - v_oracle.mean(dim=1, keepdim=true)
//    This is the "zero-sum normalization" -- forces predictions to be relative
```

### 5.4 Total Loss

```rust
pub fn total_loss(outputs: &HydraOutput<B>, targets: &HydraTargets<B>) -> Tensor<B, 1> {
    let l_pi     = policy_ce(outputs.policy_logits, targets.policy, targets.legal_mask);
    let l_v      = value_mse(outputs.value, targets.value);
    let l_grp    = grp_ce(outputs.grp_logits, targets.grp);
    let l_tenpai = tenpai_bce(outputs.tenpai_logits, targets.tenpai);
    let l_danger = danger_focal_bce(outputs.danger_logits, targets.danger, targets.danger_mask);
    let l_opp    = opp_next_ce(outputs.opp_next_logits, targets.opp_next);
    let l_pdf    = score_pdf_ce(outputs.score_pdf_logits, targets.score_pdf);
    let l_cdf    = score_cdf_bce(outputs.score_cdf_logits, targets.score_cdf);

    l_pi.mean() * 1.0
        + l_v.mean() * 0.5
        + l_grp.mean() * 0.2
        + l_tenpai.mean() * 0.1
        + l_danger.mean() * 0.1
        + l_opp.mean() * 0.1
        + (l_pdf.mean() + l_cdf.mean()) * 0.025  // 0.05 total, split evenly
}
```

### 5.5 MUST NOT

- MUST NOT use label smoothing on policy head (soft targets handle this for ExIt)
- MUST NOT use reduction='sum' -- always mean-reduce per sample, then mean over batch
- MUST NOT forget the legal_mask on policy_ce (illegal actions -> -1e9 before softmax)
- MUST NOT apply focal loss to tenpai head (only danger head gets focal)
- MUST NOT let oracle critic loss backprop into the student backbone

### 5.6 Tests

| Test | Assertion |
|------|-----------|
| `test_policy_ce_gradient` | `logits.requires_grad=true`, backward produces non-zero grad on all 46 logit dims |
| `test_soft_target_ce` | target=[0.3, 0.7, 0, ...], loss differs from hard-target [0, 1, 0, ...] by >0.01 |
| `test_total_loss_backward` | all 8 head losses > 0, total loss backward fills all param grads |
| `test_loss_weights_configurable` | construct with custom weights, verify total changes proportionally |
| `test_oracle_critic_zero_sum` | v_oracle=[1,-1,2,-2] -> zero_sum_penalty = 0.0 (within 1e-5) |
| `test_focal_bce_vs_bce` | focal(alpha=0.25,gamma=2) < standard_bce for high-confidence correct preds |

---

## STEP 6: MJAI Data Loader + Augmentation

**Files**: `hydra-train/src/data/mjai_loader.rs`, `hydra-train/src/data/augment.rs`
**Ref**: HYDRA_FINAL S11 Phase 0; hydra-engine/src/mjai_event.rs; hydra-core/src/encoder.rs; hydra-core/src/tile.rs

### 6.1 Parsing .mjson.gz

```rust
// Dependencies: flate2::read::GzDecoder, serde_json
// Each line in .mjson.gz is one JSON event (MJAI protocol)
pub fn parse_mjson_gz(path: &Path) -> Result<Vec<Vec<serde_json::Value>>> {
    let file = File::open(path)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    // Each line is a JSON object. Group by game (split on "start_game" events).
    // Returns Vec<Game> where Game = Vec<serde_json::Value>
}
```

### 6.2 Replay via hydra-engine

```rust
// For each game in the parsed MJAI log:
// 1. Create riichienv_core::state::GameState from start_kyoku event
// 2. Step through events, calling state.step() with the corresponding Action
// 3. At each decision point (state.waiting_action() for the observer):
//    a. Call state.observe() to get Observation
//    b. Encode via bridge::encode_observation(&mut encoder, &obs, &safety, drawn)
//    c. Map the actual MJAI event to HydraAction via action::mjai_to_hydra()
//    d. Compute legal_mask via action::compute_legal_mask(&obs)
//    e. Store as MjaiSample
```

### 6.3 MjaiSample Struct

```rust
pub struct MjaiSample {
    pub obs:         [f32; 2890],   // 85 * 34, flat row-major
    pub action:      u8,            // HydraAction index 0-45
    pub legal_mask:  [f32; 46],     // 1.0 = legal, 0.0 = illegal
    pub placement:   u8,            // 0-3 (final placement this game, 0=1st)
    pub score_delta: i32,           // observer's score change this round
    pub grp_label:   u8,            // 0-23 index into 4! permutations
    pub tenpai:      [f32; 3],      // ground truth opponent tenpai (0 or 1)
    pub opp_next:    [u8; 3],       // next discard tile type per opponent (0-33, 255=none)
    pub danger:      [f32; 3 * 34], // ground truth deal-in labels (0 or 1)
    pub danger_mask: [f32; 3 * 34], // 1.0 where label available, 0.0 otherwise
}
// Size: ~12.4 KB per sample. At 70 decisions/game * 6M games = 420M samples.
// Stream from disk, do NOT load all into RAM.
```

### 6.4 GRP Label Construction

```rust
// 4! = 24 permutations. Given final scores [s0, s1, s2, s3]:
// 1. Sort players by score descending (ties broken by seat order)
// 2. The resulting rank permutation (e.g., [2,0,3,1] meaning P2 got 1st, P0 got 2nd...)
//    maps to one of 24 indices via a precomputed lookup table.
pub const GRP_PERM_TABLE: [[u8; 4]; 24] = /* all 24 permutations of [0,1,2,3] */;
pub fn scores_to_grp_index(scores: [i32; 4]) -> u8 { /* ... */ }
```

### 6.5 Suit Permutation (6x augmentation)

```rust
use hydra_core::tile::{ALL_PERMUTATIONS, permute_tile_type};

pub fn augment_obs_suit(obs: &[f32; 2890], perm: &[u8; 3]) -> [f32; 2890] {
    let mut out = [0.0f32; 2890];
    for ch in 0..85 {
        for tile in 0..34u8 {
            let new_tile = permute_tile_type(tile, perm) as usize;
            out[ch * 34 + new_tile] = obs[ch * 34 + tile as usize];
        }
    }
    out
}

pub fn augment_action_suit(action: u8, perm: &[u8; 3]) -> u8 {
    // Only discard actions (0-36) need permutation
    if action <= 36 {
        hydra_core::tile::permute_tile_extended(action, perm)
    } else {
        action // chi/pon/kan/etc unchanged
    }
}

pub fn augment_mask_suit(mask: &[f32; 46], perm: &[u8; 3]) -> [f32; 46] {
    let mut out = [0.0f32; 46];
    for i in 0..37u8 {
        let new_i = hydra_core::tile::permute_tile_extended(i, perm);
        out[new_i as usize] = mask[i as usize];
    }
    for i in 37..46 { out[i] = mask[i]; } // non-discard actions unchanged
    out
}
// ALL_PERMUTATIONS has 6 entries. Index 0 = identity. Apply all 6 per sample.
```

### 6.6 Seat Rotation (4x augmentation)

```rust
// Seat rotation: view the game from each of the 4 players.
// This is NOT a post-hoc transform on the obs tensor -- it requires
// RE-ENCODING the observation from scratch with a different observer.
// During replay (6.2), for each decision point, encode from ALL 4 seats:
//   for seat in 0..4 {
//       obs_copy.player_id = seat;
//       encode_observation(&mut enc, &obs_copy, &safety_for_seat, drawn);
//   }
// Channels that flip per seat rotation:
//   - Ch 0-8: hand/drawn -> different player's hand
//   - Ch 11-22: discards -> relative order changes (self=idx0 always)
//   - Ch 23-34: melds -> relative order changes
//   - Ch 43-61: metadata (scores, gaps) -> recomputed relative to new observer
//   - Ch 62-84: safety -> recomputed for new observer
// Total augmentation: 6 suits * 4 seats = 24x (HYDRA_FINAL S11 Phase 0)
```

### 6.7 Batch Collation into Burn Tensors

```rust
pub struct MjaiBatch<B: Backend> {
    pub obs:         Tensor<B, 3>,  // [batch, 85, 34]
    pub actions:     Tensor<B, 1, Int>, // [batch] -- u8 action indices
    pub legal_mask:  Tensor<B, 2>,  // [batch, 46]
    pub value_target:Tensor<B, 1>,  // [batch] -- score_delta normalized
    pub grp_target:  Tensor<B, 2>,  // [batch, 24] -- one-hot
    pub tenpai_target: Tensor<B, 2>, // [batch, 3]
    pub danger_target: Tensor<B, 3>, // [batch, 3, 34]
    pub danger_mask:   Tensor<B, 3>, // [batch, 3, 34]
    pub opp_next_target: Tensor<B, 3>, // [batch, 3, 34] -- one-hot per opponent
    pub score_pdf_target: Tensor<B, 2>, // [batch, 64] -- one-hot bin
    pub score_cdf_target: Tensor<B, 2>, // [batch, 64] -- step function
}

pub fn collate(samples: &[MjaiSample], device: &B::Device) -> MjaiBatch<B> {
    // 1. Stack obs arrays -> Tensor::from_data([B, 2890]).reshape([B, 85, 34])
    // 2. Convert action u8 -> Tensor<Int>
    // 3. Convert grp_label u8 -> one_hot([B, 24])
    // 4. Convert opp_next u8 -> one_hot per opponent (mask 255 as zero vector)
    // 5. score_delta -> normalized value target (divide by 100_000.0)
    // 6. score_delta -> bin index for pdf (quantize into 64 bins)
    // 7. score_delta -> step function for cdf
}
```

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

```rust
pub struct BCTrainer<B: AutodiffBackend> {
    model: HydraModel<B>,
    optimizer: AdamConfig::init::<B>(), // Adam with default betas (0.9, 0.999)
    lr: f64,           // 2.5e-4 (HYDRA_FINAL S11 Phase 2 recommends this; same for BC)
    batch_size: usize, // 2048
    loss: HydraLoss<B>,
    device: B::Device,
}
```

### 7.2 Optimizer Config

```rust
let optimizer_config = AdamConfig::new()
    .with_beta_1(0.9)
    .with_beta_2(0.999)
    .with_epsilon(1e-8)
    .with_weight_decay(Some(WeightDecayConfig::new(1e-5))); // light L2
let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(2.5e-4, num_steps)
    .with_min_lr(1e-6);
```

### 7.3 Epoch Loop

```rust
pub fn train_epoch(&mut self, loader: &mut MjaiDataLoader<B>) -> EpochStats {
    let mut total_loss = 0.0;
    let mut total_correct = 0usize;
    let mut total_samples = 0usize;

    for batch in loader.iter_batches(self.batch_size) {
        // 1. Forward
        let output = self.model.forward(batch.obs.clone(), batch.legal_mask.clone());

        // 2. Loss (all 8 heads)
        let loss = self.loss.total_loss(&output, &batch.into_targets());

        // 3. Backward
        let grads = loss.backward();

        // 4. Gradient clipping (max_norm = 1.0)
        let grads = GradientsParams::from_grads(grads, &self.model)
            .clip_by_norm(1.0);

        // 5. Optimizer step
        self.model = self.optimizer.step(self.lr, self.model, grads);

        // 6. Metrics
        total_loss += loss.into_scalar() as f64;
        let pred_actions = output.policy_logits.argmax(1);
        total_correct += pred_actions.equal(batch.actions).int().sum().into_scalar() as usize;
        total_samples += batch.actions.dims()[0];
    }

    EpochStats {
        avg_loss: total_loss / (total_samples as f64 / self.batch_size as f64),
        policy_agreement: total_correct as f64 / total_samples as f64,
    }
}
```

### 7.4 Policy Agreement Metric

```rust
// Exact computation:
// pred = argmax(masked_softmax(policy_logits, legal_mask))
// agreement = mean(pred == target_action)  -- over entire eval set
// Expected range: 60-70% for strong BC on Houou data (Mortal reports ~65%)
// Masked: set illegal logits to -inf before argmax
pub fn policy_agreement(logits: Tensor<B, 2>, mask: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f64 {
    let masked = logits + (mask.ones_like() - mask) * (-1e9);
    let preds = masked.argmax(1).squeeze(1);
    preds.equal(targets).float().mean().into_scalar() as f64
}
```

### 7.5 Checkpoint Format (Burn Record)

```rust
// Use Burn's built-in record system:
// model.save_file("checkpoint_epoch_N", &NamedMpkFileRecorder::<FullPrecisionSettings>::new());
// Saves: all model weights + optimizer state
// File format: MessagePack (.mpk), ~40MB for 24-block LearnerNet
// Save every epoch. Keep last 3 + best (by policy_agreement on eval set).
pub struct CheckpointMeta {
    pub epoch: u32,
    pub train_loss: f64,
    pub eval_agreement: f64,
    pub timestamp: u64,
    pub model_config: HydraModelConfig,  // for reproducibility
}
// Save meta as JSON sidecar: checkpoint_epoch_N.meta.json
```

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

```rust
/// Compute GAE advantages and returns for a single player's trajectory.
///
/// gamma = 0.995, lambda = 0.95 (HYDRA_FINAL S11 Phase 2)
pub fn compute_gae(
    rewards: &[f32],      // [T] per-step rewards
    values:  &[f32],      // [T+1] value estimates (includes bootstrap V(s_{T+1}))
    dones:   &[bool],     // [T] episode termination flags
    gamma:   f32,         // 0.995
    lambda:  f32,         // 0.95
) -> (Vec<f32>, Vec<f32>) // (advantages[T], returns[T])
{
    let t_len = rewards.len();
    let mut advantages = vec![0.0f32; t_len];
    let mut gae = 0.0f32;

    // Reverse sweep
    for t in (0..t_len).rev() {
        let mask = if dones[t] { 0.0 } else { 1.0 };
        let delta = rewards[t] + gamma * values[t + 1] * mask - values[t];
        gae = delta + gamma * lambda * mask * gae;
        advantages[t] = gae;
    }

    // returns = advantages + values[0..T]
    let returns: Vec<f32> = advantages.iter()
        .zip(values[..t_len].iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

/// Per-player GAE: run compute_gae independently for each of 4 players.
/// Then normalize advantages per-minibatch (subtract mean, divide by std+1e-8).
pub fn normalize_advantages(advantages: &mut [f32]) {
    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt() + 1e-8;
    for a in advantages.iter_mut() {
        *a = (*a - mean) / std;
    }
}
// END gae.rs
```

### 8.2 ACH (Actor-Critic Hedge) -- VERBATIM from HYDRA_FINAL S11 Phase 2

**File**: `hydra-train/src/training/ach.rs`

**Hyperparameters** (HYDRA_FINAL S11 Phase 2, line 310):
- `ETA = 1.0` (global scalar, try {1, 2, 3})
- `EPS = 0.5` (clipping threshold for ratio)
- `L_TH = 8.0` (logit clamp threshold)
- `BETA_ENT = 5e-4` (entropy bonus coefficient)
- `LR = 2.5e-4` (Adam learning rate)

**ACH loss formula** (HYDRA_FINAL S11 Phase 2, line 303):
`L_pi(s,a) = -c(s,a) * eta * (y(a|s;theta) / pi_old(a|s)) * A(s,a)`

**CRITICAL**: Uses raw logits `y(a)`, NOT log-probs. The ratio `y(a)/pi_old` is intentional -- this is what makes ACH different from PPO.

```rust
pub struct AchConfig {
    pub eta: f32,      // 1.0
    pub eps: f32,      // 0.5
    pub l_th: f32,     // 8.0
    pub beta_ent: f32, // 5e-4
}

pub fn ach_policy_loss<B: AutodiffBackend>(
    logits:     Tensor<B, 2>,      // [B, 46] current model raw logits
    legal_mask: Tensor<B, 2>,      // [B, 46] 1.0=legal, 0.0=illegal
    actions:    Tensor<B, 1, Int>, // [B] taken action indices
    pi_old:     Tensor<B, 1>,      // [B] pi_old(a|s) from rollout
    advantages: Tensor<B, 1>,      // [B] GAE advantages (normalized)
    cfg: &AchConfig,
) -> Tensor<B, 1> {
    // 1. Mask illegal actions to -1e9
    let neg_inf = (Tensor::ones_like(&legal_mask) - legal_mask.clone()) * (-1e9);
    let masked_logits = logits.clone() + neg_inf.clone();

    // 2. Mean of LEGAL logits (for centering)
    let legal_sum = (logits.clone() * legal_mask.clone()).sum_dim(1);
    let legal_count = legal_mask.clone().sum_dim(1).clamp_min(1.0);
    let logit_mean = legal_sum / legal_count; // [B, 1]

    // 3. Center logits, clamp to [-L_TH, L_TH]
    let y_centered = (masked_logits - logit_mean).clamp(-cfg.l_th, cfg.l_th);

    // 4. Softmax for current pi (used for ratio + entropy)
    let pi = activation::softmax(y_centered.clone() + neg_inf, 1);

    // 5. Gather y[a] and pi[a] for the taken action
    let idx = actions.clone().reshape([-1, 1]);
    let y_a = y_centered.gather(1, idx.clone()).flatten(0, 1);  // [B]
    let pi_a = pi.clone().gather(1, idx).flatten(0, 1);         // [B]

    // 6. Ratio = pi(a|s) / pi_old(a|s)
    let ratio = pi_a / pi_old.clone().clamp_min(1e-8);

    // 7. Gate c: ALL 4 conditions as Rust boolean expressions
    //    HYDRA_FINAL S11 Phase 2, line 306:
    //    "c(s,a) in {0,1}: per-sample gate zeroing update when
    //     ratio exceeds 1+/-eps OR centered logit exceeds +/-l_th"
    let adv = advantages.clone();
    let adv_nonneg = adv.clone().greater_equal(Tensor::zeros_like(&adv)).float();
    let adv_neg    = adv.clone().lower(Tensor::zeros_like(&adv)).float();

    // When adv >= 0: gate ON iff ratio < 1+eps AND y[a] < l_th
    let g_ratio_pos = ratio.clone().lower_elem(1.0 + cfg.eps).float();
    let g_logit_pos = y_a.clone().lower_elem(cfg.l_th).float();
    let gate_pos = adv_nonneg * g_ratio_pos * g_logit_pos;

    // When adv < 0: gate ON iff ratio > 1-eps AND y[a] > -l_th
    let g_ratio_neg = ratio.clone().greater_elem(1.0 - cfg.eps).float();
    let g_logit_neg = y_a.clone().greater_elem(-cfg.l_th).float();
    let gate_neg = adv_neg * g_ratio_neg * g_logit_neg;

    let c = gate_pos + gate_neg; // [B], each element 0.0 or 1.0

    // 8. ACH loss: -c * eta * (y[a] / pi_old) * advantage
    let ach = -(c * cfg.eta * (y_a / pi_old.clamp_min(1e-8)) * adv).mean();

    // 9. Entropy bonus: -beta_ent * H(pi)
    let log_pi = pi.clone().clamp_min(1e-8).log();
    let entropy = -(pi * log_pi * legal_mask).sum_dim(1).mean();

    ach + (-cfg.beta_ent * entropy)
}
```

**ONE epoch per batch** (HYDRA_FINAL S11 Phase 2, line 309): "One update epoch per batch (not PPO's 3-10 epochs)". The agent MUST NOT iterate multiple epochs over the same batch of rollout data. Collect batch -> compute GAE -> one forward+backward -> discard batch -> collect new batch.

### 8.3 DRDA (Dilated Regularized Dual Averaging)

**File**: `hydra-train/src/training/drda.rs`
**Ref**: HYDRA_FINAL S11 Phase 2 (lines 298-300); Farina et al. ICLR 2025

**Policy composition** (HYDRA_FINAL S11 Phase 2, line 298):
`pi_theta(a|x) = softmax(l_base(x,a) + y_theta(x,a) / tau_drda)`

- `l_base`: frozen logits from a checkpoint (detached, no grad)
- `y_theta`: trainable residual (the model's output logits)
- `tau_drda = 4.0` (tune in {2, 4, 8}; target median KL to base in [0.05, 0.20])

```rust
pub struct DrdaWrapper<B: Backend> {
    /// Frozen base logits (detached). Initially = BC checkpoint logits.
    /// Updated only during rebase.
    base_logits_cache: Option<Tensor<B, 2>>, // lazily computed per batch
    /// Frozen base model (no grad). Used to compute l_base on the fly.
    base_model: HydraModel<B>,
    /// Temperature for residual mixing.
    pub tau_drda: f32, // 4.0
}

impl<B: AutodiffBackend> DrdaWrapper<B> {
    /// Compute combined logits for a batch.
    pub fn combined_logits(
        &self,
        base_logits: Tensor<B, 2>,     // [B, 46] from frozen base_model
        residual_logits: Tensor<B, 2>, // [B, 46] from trainable model
    ) -> Tensor<B, 2> {
        // pi = softmax(l_base + y_theta / tau_drda)
        // Return the PRE-softmax combined logits (softmax applied in ACH loss)
        base_logits + residual_logits / self.tau_drda
    }
```

**Rebase procedure** (HYDRA_FINAL S11 Phase 2, line 300 -- CRITICAL):

> "Every 25-50 GPU hours: (1) fold residual into base: l_base <- l_base + y_theta/tau_drda,
> (2) zero y_theta and reset optimizer moments. This preserves pi exactly across boundaries
> and prevents double-counting accumulated regret."

```rust
    /// Rebase: fold current residual into the frozen base.
    /// MUST preserve pi exactly (KL < 1e-6 before/after).
    pub fn rebase(&mut self, model: &mut HydraModel<B>, optimizer: &mut impl Optimizer<B>) {
        // STEP 1: Fold residual into base
        //   For each parameter in the final linear layer that produces logits:
        //   base_weight += model_weight / tau_drda
        //   base_bias   += model_bias   / tau_drda
        // Alternatively (simpler): replace base_model with a model that
        // produces combined_logits directly, then zero the residual.

        // STEP 2: Zero y_theta (the trainable model's final logit layer)
        //   Set all weights and biases of the policy head's final Linear to 0.
        //   This makes y_theta(x,a) = 0 for all x,a immediately after rebase.

        // STEP 3: Reset optimizer moments
        //   Adam's m (first moment) and v (second moment) accumulators
        //   must be zeroed. In Burn: re-initialize the optimizer.
        //   *optimizer = AdamConfig::new().init();
    }
}
```

### 8.4 Full RL Training Step (ACH + DRDA combined)

```rust
pub fn rl_step<B: AutodiffBackend>(
    model: &mut HydraModel<B>,
    drda: &DrdaWrapper<B>,
    batch: &RolloutBatch<B>,
    ach_cfg: &AchConfig,
    optimizer: &mut impl Optimizer<B>,
) {
    // 1. Compute base logits (frozen, no grad)
    let base_logits = Tensor::no_grad(|| {
        drda.base_model.forward_logits(batch.obs.clone(), batch.legal_mask.clone())
    });

    // 2. Compute residual logits (trainable)
    let residual_logits = model.forward_logits(batch.obs.clone(), batch.legal_mask.clone());

    // 3. Combined logits
    let combined = drda.combined_logits(base_logits, residual_logits);

    // 4. ACH loss on combined logits
    let loss = ach_policy_loss(combined, batch.legal_mask, batch.actions,
                                batch.pi_old, batch.advantages, ach_cfg);

    // 5. Value loss (on trainable model's value head)
    let value_loss = value_mse(model.forward_value(batch.obs.clone()), batch.returns) * 0.5;

    // 6. Auxiliary head losses (same weights as BC)
    let aux_loss = compute_auxiliary_losses(model, batch); // GRP + tenpai + danger + opp + score

    // 7. Total
    let total = loss + value_loss + aux_loss;

    // 8. Backward + clip + step (ONE epoch, ONE pass)
    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, model).clip_by_norm(1.0);
    *model = optimizer.step(2.5e-4, model.clone(), grads);
}
```

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
| `test_gae_simple` | 5-step trajectory: r=[1,0,0,0,1], V=[0.5]*6, gamma=0.995, lambda=0.95 -> verify advantages match hand-computed values within 1e-4 |
| `test_gae_done_resets` | done=true at step 2 -> advantage at step 1 does NOT see reward at step 3 |
| `test_ach_gate_positive_adv` | adv>0, ratio=1.0, y_a=0.0 -> c=1.0 (all conditions met) |
| `test_ach_gate_clips_ratio` | adv>0, ratio=1.6 (>1+0.5) -> c=0.0 (gate kills update) |
| `test_ach_gate_clips_logit` | adv>0, y_a=9.0 (>L_TH=8) -> c=0.0 (gate kills update) |
| `test_ach_gate_negative_adv` | adv<0, ratio=0.4 (<1-0.5) -> c=0.0 |
| `test_ach_one_epoch` | one forward+backward changes model weights, second pass on same data is FORBIDDEN |
| `test_drda_combined_logits` | base=[1,2,3], residual=[4,8,12], tau=4 -> combined=[2,4,6] |
| `test_drda_rebase_preserves_pi` | compute pi before rebase, rebase, compute pi after -> KL < 1e-6 |
| `test_drda_residual_zeroed` | after rebase, model's policy head final layer weights are all 0.0 |
| `test_drda_optimizer_reset` | after rebase, Adam m and v moments are all 0.0 |
# Hydra Build Plan: Steps 9-12 + Final Gate

## Step 9: CT-SMC Exact DP Sampler (Pure Rust, No Burn)

### File: `hydra-core/src/ct_smc.rs`

### 9.1 State Space

The hidden allocation matrix X has shape 34x4 (tile types x locations: opp1, opp2, opp3, wall).
Row sums r(k) = 4 - visible(k), column sums s(z) = concealed hand sizes + wall size.

**Key insight**: c_W = R_k - (c1 + c2 + c3) is derived, so DP state is 3D: (c1, c2, c3).
- Max concealed hand = 14 tiles (after draw, before discard)
- State space: each c_i in [0, 14], so <= 15^3 = 3,375 states per DP layer
- 34 tile-type layers, <= 35 compositions per layer
- Total: ~34 * 3,375 * 35 = ~4.0M ops

### 9.2 Data Structures

```rust
/// Per-tile-type learned field weights from Mixture-SIB component.
/// omega[k][j] = exp(F_theta(k, j)) for location j in {0,1,2,3}.
pub struct CtSmcConfig {
    pub num_particles: usize,       // P, typically 128-4096
    pub ess_threshold: f32,         // resample when ESS < threshold * P, default 0.4
    pub rng_seed: u64,
}

/// A single CT-SMC particle: full 34x4 allocation + log-weight.
#[derive(Clone)]
pub struct Particle {
    pub allocation: [[u8; 4]; 34],  // X[k][z], z=0,1,2 are opponents, z=3 is wall
    pub log_weight: f64,
}

/// Precomputed compositions for r in 0..=4.
/// COMPOSITIONS[r] = Vec of (x0, x1, x2, x3) with sum = r.
/// Sizes: r=0 -> 1, r=1 -> 4, r=2 -> 10, r=3 -> 20, r=4 -> 35. Total: 70.
static COMPOSITIONS: LazyLock<[Vec<[u8; 4]>; 5]> = LazyLock::new(precompute_compositions);

fn precompute_compositions() -> [Vec<[u8; 4]>; 5] {
    let mut result: [Vec<[u8; 4]>; 5] = Default::default();
    for r in 0..=4u8 {
        for x0 in 0..=r {
            for x1 in 0..=(r - x0) {
                for x2 in 0..=(r - x0 - x1) {
                    let x3 = r - x0 - x1 - x2;
                    result[r as usize].push([x0, x1, x2, x3]);
                }
            }
        }
    }
    result
}
```

### 9.3 Exact DP Recurrence (Log-Space)

```rust
/// Forward pass: compute log Z_k(c1, c2, c3) for k = 34 down to 0.
/// Base: dp[34][(0,0,0)] = 0.0 (log 1), all others -inf.
/// Recurrence: dp[k][(c1,c2,c3)] = logsumexp over compositions x of r(k):
///   log_phi_k(x) + dp[k+1][(c1-x0, c2-x1, c3-x2)]
///   where x3 = r(k)-x0-x1-x2, require x_j <= c_j, x3 <= c_W = R_k - sum(c).
/// log_phi_k(x) = sum_j x[j] * log(omega[k][j])
fn forward_dp(
    row_sums: &[u8; 34], col_sums: &[usize; 4], log_omega: &[[f64; 4]; 34],
) -> Vec<FxHashMap<(u8, u8, u8), f64>> {
    let mut dp: Vec<FxHashMap<(u8,u8,u8), f64>> = vec![FxHashMap::default(); 35];
    dp[34].insert((0, 0, 0), 0.0);
    let mut r_suffix = [0u16; 35];
    for k in (0..34).rev() { r_suffix[k] = r_suffix[k+1] + row_sums[k] as u16; }

    for k in (0..34).rev() {
        let comps = &COMPOSITIONS[row_sums[k] as usize];
        for (&(c1, c2, c3), &log_z_next) in &dp[k + 1] {
            for comp in comps {
                let (nc1, nc2, nc3) = (c1+comp[0], c2+comp[1], c3+comp[2]);
                let nc_sum = nc1 as u16 + nc2 as u16 + nc3 as u16;
                if nc_sum > r_suffix[k] { continue; }
                if (comp[3] as u16) > r_suffix[k] - nc_sum { continue; }
                let log_phi = comp.iter().enumerate()
                    .map(|(j, &x)| x as f64 * log_omega[k][j]).sum::<f64>();
                let entry = dp[k].entry((nc1, nc2, nc3)).or_insert(f64::NEG_INFINITY);
                *entry = logsumexp2(*entry, log_phi + log_z_next);
            }
        }
    }
    dp
}
#[inline] fn logsumexp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let m = a.max(b); m + ((a-m).exp() + (b-m).exp()).ln()
}
```

### 9.4 Exact Backward Sampling

```rust
/// Sample one complete allocation X from the DP table.
/// For k = 0..34: sample composition x_k ~ p(x_k | remaining capacity).
///   p(x_k = x | c) = exp(log_phi_k(x) + dp[k+1][c - x] - dp[k][c])
/// This is exact: no rejection, no MCMC. One pass through 34 tile types.
fn backward_sample(
    dp: &[FxHashMap<(u8,u8,u8), f64>],
    row_sums: &[u8; 34],
    log_omega: &[[f64; 4]; 34],
    rng: &mut impl Rng,
) -> [[u8; 4]; 34] {
    let mut allocation = [[0u8; 4]; 34];
    let mut c = (col_sums[0] as u8, col_sums[1] as u8, col_sums[2] as u8);

    for k in 0..34 {
        let r = row_sums[k] as usize;
        let comps = &COMPOSITIONS[r];
        let log_z_k = dp[k][&c]; // partition at current state

        // Build categorical distribution over valid compositions
        let mut log_probs: SmallVec<[f64; 35]> = SmallVec::new();
        let mut valid_comps: SmallVec<[[u8; 4]; 35]> = SmallVec::new();
        for comp in comps {
            if comp[0] > c.0 || comp[1] > c.1 || comp[2] > c.2 { continue; }
            let next_c = (c.0 - comp[0], c.1 - comp[1], c.2 - comp[2]);
            if let Some(&log_z_next) = dp[k + 1].get(&next_c) {
                let log_phi = /* same as forward */;
                log_probs.push(log_phi + log_z_next - log_z_k);
                valid_comps.push(*comp);
            }
        }
        // Sample from categorical (log-space -> normalized probabilities)
        let idx = log_categorical_sample(&log_probs, rng);
        allocation[k] = valid_comps[idx];
        c = (c.0 - allocation[k][0], c.1 - allocation[k][1], c.2 - allocation[k][2]);
    }
    allocation
}
```

### 9.5 SMC Reweighting + ESS + Resampling

```rust
/// Full CT-SMC pipeline:
/// 1. Run forward DP once per belief update (when omega changes).
/// 2. Sample P particles via backward_sample (exact from prior p_0).
/// 3. Weight each particle by opponent action likelihood L(X).
/// 4. Compute ESS = 1 / sum(w_i^2). If ESS < threshold * P, resample.
/// 5. Rejuvenate via Metropolis-Hastings row-swap moves (optional).
pub struct CtSmc {
    pub config: CtSmcConfig,
    pub particles: Vec<Particle>,
    dp_cache: Option<Vec<FxHashMap<(u8,u8,u8), f64>>>,
}

impl CtSmc {
    /// Update beliefs: recompute DP, resample particles.
    pub fn update(
        &mut self,
        row_sums: &[u8; 34],
        col_sums: &[usize; 4],
        log_omega: &[[f64; 4]; 34],
        likelihood_fn: &dyn Fn(&[[u8; 4]; 34]) -> f64, // log L(X)
        rng: &mut impl Rng,
    ) {
        let dp = forward_dp(row_sums, col_sums, log_omega);
        self.particles.clear();
        for _ in 0..self.config.num_particles {
            let alloc = backward_sample(&dp, row_sums, log_omega, rng);
            let log_w = likelihood_fn(&alloc);
            self.particles.push(Particle { allocation: alloc, log_weight: log_w });
        }
        self.normalize_weights();
        if self.ess() < self.config.ess_threshold * self.config.num_particles as f32 {
            self.systematic_resample(rng);
        }
        self.dp_cache = Some(dp);
    }

    /// Effective sample size: 1 / sum(w_i^2) in normalized weight space.
    pub fn ess(&self) -> f32 { /* standard ESS formula */ }

    /// Systematic resampling (O(N), low variance).
    fn systematic_resample(&mut self, rng: &mut impl Rng) { /* standard */ }
}
```

### 9.6 Public API

```rust
// In hydra-core/src/lib.rs:
pub mod ct_smc;

// Key exports:
pub use ct_smc::{CtSmc, CtSmcConfig, Particle};
```

### 9.7 Tests + Benchmark

```rust
#[cfg(test)]
mod tests {
    /// Correctness: for r=[1,1,0,...] with uniform omega, verify marginals
    /// match hypergeometric analytically.
    #[test]
    fn uniform_omega_matches_hypergeometric() { /* ... */ }

    /// Conservation: every particle satisfies row and column sum constraints.
    #[test]
    fn particles_satisfy_constraints() { /* ... */ }

    /// ESS: with uniform likelihood, ESS should be close to P.
    #[test]
    fn uniform_likelihood_high_ess() { /* ... */ }
}

/// Benchmark: median < 1ms over 1000 runs (forward DP + 128 backward samples).
/// Use Criterion. Typical game state: ~20 hidden tiles, r(k) in 0..4.
#[bench] fn bench_ct_smc_full_pipeline(b: &mut Bencher) { /* ... */ }
```

---

## Step 10: AFBS Search Engine + Robust Opponent

### Files: `hydra-core/src/afbs.rs`, `hydra-core/src/robust_opponent.rs`

### 10.1 Search Tree Node

```rust
pub struct AfbsNode {
    pub info_state_hash: u64,       // Zobrist hash of public info state
    pub visit_count: u32,           // N(node)
    pub total_value: f64,           // W(node), sum of backed-up values
    pub prior: f32,                 // P(a) from policy network
    pub children: SmallVec<[(u8, NodeIdx); 5]>, // (action_id, child_index), top-K only
    pub is_opponent: bool,          // true = opponent decision node
    pub particle_handle: Option<u32>, // index into particle pool for this branch
}

/// PUCT selection: pick child maximizing UCB.
/// UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
/// where Q(a) = W(a) / N(a) (mean value of child).
/// c_puct = 2.5 (tunable in Phase -1 sweep).
fn select_child(node: &AfbsNode, nodes: &[AfbsNode], c_puct: f32) -> usize {
    let sqrt_n_parent = (node.visit_count as f32).sqrt();
    node.children.iter()
        .enumerate()
        .max_by(|(_, (_, a_idx)), (_, (_, b_idx))| {
            let a = &nodes[*a_idx as usize];
            let b = &nodes[*b_idx as usize];
            let ucb_a = (a.total_value as f32 / a.visit_count.max(1) as f32)
                + c_puct * a.prior * sqrt_n_parent / (1.0 + a.visit_count as f32);
            let ucb_b = (b.total_value as f32 / b.visit_count.max(1) as f32)
                + c_puct * b.prior * sqrt_n_parent / (1.0 + b.visit_count as f32);
            ucb_a.partial_cmp(&ucb_b).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

### 10.2 Top-K Expansion + Root Sampling

```rust
/// Expand node: query policy network for action priors, keep top K=5.
/// For our nodes: legal actions from game state, priors from ActorNet.
/// For opponent nodes: sample CT-SMC particle, condition ActorNet on particle's hand.
const TOP_K: usize = 5;

fn expand_node(
    node_idx: NodeIdx,
    tree: &mut AfbsTree,
    policy_logits: &[f32; 46],  // from batched NN eval
    legal_mask: &[bool; 46],
) {
    // Masked softmax over legal actions
    let priors = masked_softmax(policy_logits, legal_mask);
    // Take top-K by prior probability
    let mut top_k: SmallVec<[(u8, f32); 5]> = SmallVec::new();
    // ... argpartition top K, store (action_id, prior) ...
    for (action_id, prior) in top_k {
        let child_idx = tree.allocate_node(prior);
        tree.nodes[node_idx].children.push((action_id, child_idx));
    }
}
```

### 10.3 Batched Leaf Evaluation

```rust
/// Collect pending leaf evaluations into a batch (min 32 for GPU efficiency).
/// Each leaf needs: encoded observation tensor (85x34 from encoder.rs).
/// Returns: (policy_logits[46], value[1]) per leaf.
pub struct LeafBatch {
    pub obs_buffer: Vec<f32>,       // N * 85 * 34 contiguous
    pub node_indices: Vec<NodeIdx>, // which tree node each leaf corresponds to
    pub batch_size: usize,
}

impl LeafBatch {
    pub const MIN_BATCH: usize = 32;
    pub fn is_ready(&self) -> bool { self.batch_size >= Self::MIN_BATCH }
}
```

### 10.4 Robust KL-Ball Opponent Nodes

```rust
/// File: hydra-core/src/robust_opponent.rs
///
/// At opponent decision nodes, compute robust value:
///   V_rob = min_{q in KL-ball} sum_a q(a) * Q(a)
/// Solution: q_tau(a) = p(a) * exp(-Q(a)/tau) / Z(tau)
/// Find tau via binary search such that KL(q_tau || p) = epsilon.
///
/// Then soft-min over N=4 archetypes:
///   Q_final(a) = -tau_arch * log(sum_i w_i * exp(-Q_i(a) / tau_arch))

pub struct RobustOpponentConfig {
    pub epsilon: f32,              // KL ball radius, calibrated from data (~0.1-0.5)
    pub tau_search_iters: u8,      // binary search iterations for tau (default: 20)
    pub num_archetypes: usize,     // N=4: aggressive, defensive, speed, value
    pub tau_arch: f32,             // archetype soft-min temperature (default: 1.0)
}

/// Binary search for tau satisfying KL(q_tau || p) = epsilon.
/// KL is monotone decreasing in tau, so binary search works.
/// Returns (tau, q_tau) where q_tau is the robust opponent policy.
fn find_robust_tau(
    p: &[f32],        // base opponent policy (from ActorNet)
    q_values: &[f32], // Q(a) for each action at this node
    epsilon: f32,
    iters: u8,
) -> (f32, Vec<f32>) {
    let mut lo = 1e-4_f32;
    let mut hi = 100.0_f32;
    for _ in 0..iters {
        let mid = (lo + hi) / 2.0;
        let q_tau = compute_q_tau(p, q_values, mid);
        let kl = kl_divergence(&q_tau, p);
        if kl > epsilon { lo = mid; } else { hi = mid; }
    }
    let tau = (lo + hi) / 2.0;
    (tau, compute_q_tau(p, q_values, tau))
}
```

### 10.5 Pondering Manager

```rust
/// File: hydra-core/src/pondering.rs
/// Async task that runs AFBS during opponent turns (75% idle time).
/// Priority queue: states sorted by (policy_gap, risk_level, ess_inverse).
/// Results stored in lockless DashMap<InfoStateHash, PonderResult>.

pub struct PonderManager {
    pub cache: DashMap<u64, PonderResult>,  // info_state_hash -> search result
    priority_queue: BinaryHeap<PonderTask>,
    worker_handle: Option<JoinHandle<()>>,
}

pub struct PonderResult {
    pub exit_policy: [f32; 46],   // pi_ExIt from deep AFBS
    pub value: f32,                // V from deep AFBS
    pub search_depth: u8,
    pub visit_count: u32,
    pub timestamp: Instant,
}

/// Priority: high when top-2 policy gap < 0.1, risk > threshold, ESS low.
pub struct PonderTask {
    pub info_state_hash: u64,
    pub priority_score: f32,       // higher = more urgent
    pub game_state_snapshot: GameStateSnapshot,
}
```

### 10.6 Tests

```rust
#[cfg(test)]
mod tests {
    #[test] fn puct_selects_high_prior_unvisited() { /* UCB should prefer high-prior unvisited */ }
    #[test] fn robust_tau_converges() { /* KL(q_tau||p) within 1% of epsilon after 20 iters */ }
    #[test] fn archetype_softmin_reduces_to_uniform() { /* equal Q -> uniform weights */ }
    #[test] fn ponder_cache_hit_reuses_search() { /* lookup returns stored result */ }
    #[test] fn batched_eval_correct_size() { /* batch of 32 produces 32 outputs */ }
}
```

---

## Step 11: Self-Play Arena + Distillation + ExIt Pipeline

### Files: `hydra-core/src/arena.rs`, `hydra-train/src/distill.rs`, `hydra-train/src/exit.rs`

### 11.1 Arena: BatchSimulator Integration

```rust
/// Extends existing BatchSimulator (simulator.rs) with NN-driven action selection.
/// 500+ parallel games, each seat driven by ActorNet with temperature sampling.
pub struct Arena {
    pub simulator: BatchSimulator,     // reuse existing rayon pool
    pub actor_weights: Arc<RwLock<ActorWeights>>, // hot-swapped by distiller
    pub config: ArenaConfig,
    pub trajectory_buffer: Vec<Trajectory>,
}

pub struct ArenaConfig {
    pub num_parallel_games: usize,     // 500+ (bounded by GPU batch throughput)
    pub game_mode: u8,                 // 0 = hanchan
    pub temperature_range: (f32, f32), // per-seat: rand::Uniform(0.5, 1.5)
    pub exit_fraction: f32,            // fraction of states sent to AFBS (0.2 in Phase 2)
    pub max_trajectory_buffer: usize,  // flush to trainer when full
}

/// Per-seat temperature: sampled once at game start, fixed for that seat.
/// T ~ Uniform(0.5, 1.5). Action sampling: a ~ Categorical(softmax(logits / T)).
fn sample_action_with_temperature(logits: &[f32; 46], legal: &[bool; 46], temp: f32) -> u8 {
    let scaled: Vec<f32> = logits.iter().zip(legal)
        .map(|(&l, &ok)| if ok { l / temp } else { f32::NEG_INFINITY })
        .collect();
    categorical_sample(&softmax(&scaled))
}
```

### 11.2 Trajectory Struct

```rust
/// One decision point. Stored contiguously for efficient batching.
#[derive(Clone)]
#[repr(C)]
pub struct TrajectoryStep {
    pub obs: [f32; 85 * 34],          // encoded observation (from encoder.rs)
    pub action: u8,                    // action taken (0..45)
    pub pi_old: [f32; 46],            // policy at time of action
    pub reward: f32,                   // 0 during game, final placement reward at done
    pub done: bool,
    pub player_id: u8,                 // 0..3
    pub game_id: u32,
    pub turn: u16,
    pub temperature: f32,
}

pub struct Trajectory {
    pub steps: Vec<TrajectoryStep>,
    pub final_scores: [i32; 4],
    pub game_id: u32,
    pub seed: u64,
}
```

### 11.3 Distiller: Learner -> Actor

```rust
/// File: hydra-train/src/distill.rs
/// Continuous distillation (IMPALA-style, every 60-120s).
/// L_kd = KL(sg(learner_pi) || actor_pi) + 0.5 * MSE(sg(learner_v), actor_v)

pub struct DistillConfig {
    pub kd_kl_weight: f32,            // 1.0
    pub kd_mse_weight: f32,           // 0.5
    pub distill_lr: f32,              // 1e-4
    pub distill_batch_size: usize,    // 256
    pub update_interval_secs: u64,    // 60-120
    pub ema_decay: f32,               // 0.999
}
```

### 11.4 ExIt Pipeline

```rust
/// File: hydra-train/src/exit.rs
/// pi_exit(a|I) = softmax(Q(I,a) / tau_exit) from AFBS.
/// Safety valve: skip if visit_count < min_visits OR KL(exit||base) > max_kl.
pub struct ExitConfig {
    pub tau_exit: f32,                 // 1.0
    pub exit_weight: f32,             // 0.5 (annealed up during Phase 3)
    pub min_visits: u32,              // 64
    pub hard_state_threshold: f32,    // top-2 gap < 0.1 -> hard state
    pub safety_valve_max_kl: f32,     // 2.0
}
/// Combined loss (Phase 2-3):
///   L = L_ach + exit_weight*L_exit + saf_weight*L_saf + aux_weight*L_aux
```

### 11.5 Tests

```rust
#[cfg(test)]
mod tests {
    #[test] fn trajectory_roundtrip() { /* serialize/deserialize */ }
    #[test] fn temperature_sampling_legal_only() { /* illegal never selected */ }
    #[test] fn distill_loss_zero_when_identical() { /* same weights -> L=0 */ }
    #[test] fn exit_safety_valve_skips_low_visits() { /* N<64 ignored */ }
    #[test] fn arena_500_games_completes() { /* 500 games with dummy policy */ }
}
```

---

## Step 12: Remaining 7 Components

### 12.1 Endgame Solver

**File**: `hydra-core/src/endgame.rs`

```rust
/// PIMC endgame solver. Trigger: wall <= 10 AND threat signal active.
/// For each CT-SMC particle: sample ONE draw sequence + ONE opponent action sequence.
/// Average Q over P particles (top-mass subset covering 95% weight, P~50-100).
pub struct EndgameSolver { pub max_wall: u8, pub mass_threshold: f32 }
pub fn solve_endgame(particles: &[Particle], game: &GameState, actor: &ActorNet) -> [f32; 46];
// Tests: endgame_improves_placement_accuracy (50K positions, must beat vanilla AFBS on all 3 metrics)
```

### 12.2 Sinkhorn Belief (Mixture-SIB)

**File**: `hydra-core/src/sinkhorn.rs`

```rust
/// SIB: B* = diag(u) * K * diag(v) via Sinkhorn-Knopp iterations.
/// Mixture-SIB: L components, Bayesian weight updates on observed events.
pub struct SinkhornConfig { pub max_iters: u16, pub tol: f64, pub num_components: u8 }
pub struct MixtureSib { pub components: Vec<SibComponent>, pub weights: Vec<f64> }
pub fn sinkhorn_project(kernel: &[f64; 34*4], row_sums: &[u8;34], col_sums: &[usize;4]) -> [f64;34*4];
// Tests: sinkhorn_converges_to_margins, mixture_weight_update_is_bayesian
```

### 12.3 Hand-EV Oracle Features

**File**: `hydra-core/src/hand_ev.rs` (extends `shanten_batch.rs`)

```rust
/// CPU-precomputed per-discard features (Group D of input tensor):
///   P_tenpai(a, d) for d in {1,2,3} draws
///   P_win(a, d) for d in {1,2,3} draws
///   E[score | win, a] (han/fu/score lookup)
///   Ukeire vector: 34-element acceptance weighted by belief-remaining counts
/// Uses batch_discard_shanten() for shanten, scoring engine for E[score].
/// Zero GPU cost -- CPU side during game step.
pub struct HandEvFeatures { pub tenpai_prob: [[f32;3];34], pub win_prob: [[f32;3];34],
    pub expected_score: [f32;34], pub ukeire: [[f32;34];34] }
pub fn compute_hand_ev(hand: &[u8;34], belief_remaining: &[f32;34]) -> HandEvFeatures;
// Tests: tenpai_hand_has_high_p_tenpai, ukeire_sums_match_acceptance_count
```

### 12.4 Search-as-Feature (SaF)

**File**: `hydra-train/src/saf.rs`

```rust
/// Logit-residual: l_final(a) = l_theta(a) + alpha * g_psi(f(a)) * m(a)
/// f(a) = [delta_Q(a), risk(a), entropy_drop(a), tau_robust(a), ess(a)]
/// g_psi: MLP(input=5, hidden=32, output=1). SaF-dropout p=0.3 during training.
pub struct SafConfig { pub alpha: f32, pub dropout: f32, pub hidden_dim: usize }
pub struct SafMlp { /* tiny 5->32->1 MLP */ }
// Tests: saf_dropout_zeros_features_at_rate, saf_logit_addition_correct
```

### 12.5 Population League

**File**: `hydra-train/src/league.rs`

```rust
/// League: latest ActorNet, 3 trailing checkpoints, 2 BC-anchors, 1 exploiter.
/// Matchmaking: uniform random from pool. Elo tracking per agent.
pub struct League { pub agents: Vec<LeagueAgent>, pub elo_ratings: Vec<f32> }
pub struct LeagueAgent { pub weights_path: PathBuf, pub agent_type: AgentType }
pub enum AgentType { Current, Checkpoint(u32), BcAnchor, Exploiter }
// Tests: league_matchmaking_is_uniform, elo_updates_correctly
```

### 12.6 Evaluation Harness

**File**: `hydra-train/src/eval.rs`

```rust
/// Run N=1000 hanchan games against fixed opponents (BC baseline, Mortal-level).
/// Report: mean placement, stable dan estimate, win/deal-in/tsumo rates.
pub struct EvalConfig { pub num_games: usize, pub opponents: Vec<PathBuf>, pub seed: u64 }
pub struct EvalResult { pub mean_placement: f32, pub stable_dan: f32,
    pub win_rate: f32, pub deal_in_rate: f32, pub tsumo_rate: f32 }
pub fn evaluate(agent: &ActorNet, config: &EvalConfig) -> EvalResult;
// Tests: eval_deterministic_with_seed, eval_reports_all_metrics
```

### 12.7 Inference Server

**File**: `hydra-train/src/inference.rs`

```rust
/// Fast path: ActorNet forward + SaF adaptor (< 5ms).
/// Slow path: reuse pondered AFBS subtree if cache hit.
/// On-turn budget: 80-150ms. Call reactions: 20-50ms.
/// Agari guard: always-on final check before committing action.
pub struct InferenceServer {
    pub actor: ActorNet,
    pub ponder_cache: Arc<DashMap<u64, PonderResult>>,
    pub saf_mlp: SafMlp,
}
pub fn infer(server: &InferenceServer, obs: &[f32; 85*34], legal: &[bool;46]) -> (u8, [f32;46]);
// Tests: inference_respects_time_budget, agari_guard_prevents_illegal
```

---

## Final Gate: End-to-End Integration Test

### File: `tests/integration_pipeline.rs`

This test chains the entire pipeline: BC -> ACH -> AFBS -> ExIt -> Distill -> Inference.
Runs with tiny models (2-block, 32 channels) on CPU to verify correctness, not performance.

```rust
/// Integration test: full pipeline chain on toy-sized models.
/// Verifies data flows correctly through every stage without panics or NaN.
#[test]
fn full_pipeline_integration() {
    // 1. BC warm start: train tiny LearnerNet on 100 synthetic expert games
    let learner = TinyLearnerNet::new(blocks=2, channels=32);
    let bc_data = generate_synthetic_expert_games(100);
    let learner = bc_train(&learner, &bc_data, epochs=2);
    assert!(!has_nan_params(&learner), "BC produced NaN weights");

    // 2. Distill LearnerNet -> ActorNet
    let actor = TinyActorNet::new(blocks=1, channels=32);
    let actor = distill(&learner, &actor, steps=10);
    assert!(!has_nan_params(&actor), "Distillation produced NaN weights");

    // 3. Self-play arena: 10 games with ActorNet
    let arena = Arena::new(actor.clone(), ArenaConfig { num_parallel_games: 10, .. });
    let trajectories = arena.run_one_batch();
    assert!(trajectories.len() == 10, "Arena must produce 10 game trajectories");
    assert!(trajectories.iter().all(|t| !t.steps.is_empty()), "No empty trajectories");

    // 4. ACH training step on trajectories
    let learner = ach_update(&learner, &trajectories, AchConfig::default());
    assert!(!has_nan_params(&learner), "ACH produced NaN weights");

    // 5. CT-SMC belief sampling
    let game_state = trajectories[0].steps[10].reconstruct_game_state();
    let ct_smc = CtSmc::new(CtSmcConfig { num_particles: 32, .. });
    let particles = ct_smc.sample(&game_state);
    assert_eq!(particles.len(), 32);
    for p in &particles {
        assert_row_col_sums_valid(&p.allocation, &game_state);
    }

    // 6. AFBS search on one position
    let afbs = AfbsTree::new(AfbsConfig { beam_w: 8, depth: 2, top_k: 3, .. });
    let search_result = afbs.search(&game_state, &learner, &particles);
    assert!(search_result.visit_count >= 8, "AFBS must expand nodes");
    let exit_policy = search_result.exit_policy();
    assert!((exit_policy.iter().sum::<f32>() - 1.0).abs() < 0.01, "ExIt policy must sum to 1");

    // 7. ExIt training step
    let learner = exit_update(&learner, &[(game_state.obs(), exit_policy)], ExitConfig::default());
    assert!(!has_nan_params(&learner), "ExIt produced NaN weights");

    // 8. Final distillation + inference
    let actor = distill(&learner, &actor, steps=5);
    let server = InferenceServer::new(actor);
    let (action, policy) = server.infer(&game_state.obs(), &game_state.legal_mask());
    assert!(game_state.legal_mask()[action as usize], "Inference must pick legal action");
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01, "Policy must sum to 1");
}

/// Smoke test: no component panics on edge cases (empty hand, all tiles visible, etc.)
#[test]
fn edge_case_smoke_test() {
    // Game with only 1 round remaining, wall nearly empty
    let endgame_state = create_near_endgame_state(wall_remaining=5);
    let ct_smc = CtSmc::new(CtSmcConfig { num_particles: 16, .. });
    let particles = ct_smc.sample(&endgame_state);
    assert!(!particles.is_empty());

    let endgame_solver = EndgameSolver::new(max_wall=10);
    let q_values = endgame_solver.solve(&particles, &endgame_state, &dummy_actor());
    assert!(q_values.iter().all(|v| v.is_finite()), "No NaN/Inf in endgame Q");
}
```

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

