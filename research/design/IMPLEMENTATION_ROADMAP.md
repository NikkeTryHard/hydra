# HYDRA Implementation Roadmap

Maps HYDRA_FINAL.md design to concrete Rust/Burn code tasks.
Ordered by dependency -- each task builds on the previous.

## Phase 0: Model Architecture (hydra-train)

### Burn API Notes
- Custom training loop: `loss.backward()` -> `GradientsParams::from_grads()` -> `optim.step()`
- PPO requires manual loop (not `SupervisedTraining` / `Learner`)
- Backend: `burn-tch` with `AutodiffBackend` for training, `Backend` for inference
- bf16: `set_default_dtypes_unchecked(&device, FloatDType::BF16, IntDType::I32)`
- Conv1d, GroupNorm, Linear all available in `burn::nn`

### T0.1: SE-ResNet in Burn
- Implement 40-block SE-ResNet with GroupNorm(32) + Mish
- Input: Tensor<B, 2> shape [batch, 85*34] -> reshape to [batch, 85, 34]
- Stem: Conv1d(85, 256, k=3, pad=1)
- Block: Conv1d(256,256,k=3) -> GN -> Mish -> Conv1d(256,256,k=3) -> GN + SE + skip
- Output: feature map [batch, 256, 34] + pooled [batch, 256]
- Verify: forward pass compiles, bf16 works on burn-tch

### T0.2: 9 Output Heads
- Policy: Linear(256*34, 46) with legal action masking
- Value: Linear(256, 1)
- GRP: Linear(256, 24)
- Tenpai: Linear(256, 3) with sigmoid
- Danger: Linear(256, 3*34) reshaped to [batch, 3, 34]
- Opp-next: Linear(256, 34) with softmax
- Score-pdf: Linear(256, B) with softmax (B=64 bins)
- Score-cdf: Linear(256, B) with sigmoid
- Oracle critic (training only): separate small net, input [batch, (85+oracle)*34]

### T0.3: Loss Functions
- Policy: CE with ExIt target support (weighted by KL divergence)
- Value: MSE + multi-lambda TD targets (lambda in {0.0, 0.2, ..., 1.0})
- GRP: CE over 24 ranking permutations
- Others: BCE/MSE as specified in HYDRA_FINAL Section 2
- Oracle critic: MSE with zero-sum constraint
- Total loss: weighted sum with configurable lambdas

## Already Implemented (in hydra-core + hydra-engine)
- 85x34 observation encoder (encoder.rs, 1450 lines, 31 tests)
- 46-action space mapping (action.rs, Mortal-compatible)
- Safety module: genbutsu, suji, kabe, one-chance (safety.rs, 23 channels)
- Bridge: game state -> encoder input (bridge.rs, 465 lines)
- Batch simulator with rayon (simulator.rs, 29K games/sec)
- Shanten computation with caching (shanten_batch.rs)
- Tile system with suit permutation support (tile.rs)
- Deterministic seeding hierarchy (seeding.rs)
- MJAI event types + replay (hydra-engine mjai_event.rs)
- Batch encoder with zero-alloc (batch_encoder.rs)

## Phase 1: Behavioral Cloning Pipeline

### T1.1: Data Loader
- Read MJAI .mjson.gz files from tenhou-to-mjai dataset
- For each game: replay states, encode 85x34 obs via hydra-core encoder
- Extract: obs, action, mask, game_result, scores, placement
- 24x augmentation (6 suit perms x 4 seat rotations) applied online
- Output: batched tensors ready for Burn training loop

### T1.2: BC Training Loop
- Standard supervised training on expert data
- All 9 heads supervised (action labels, game outcomes, etc.)
- Adam optimizer, LR 2.5e-4, batch 2048
- 24 epochs over 5-6M games
- Checkpoint every epoch, evaluate policy agreement on held-out set

## Phase 2: Oracle Guiding

### T2.1: Oracle Feature Encoder
- Extend encoder with oracle channels (opponent hands + wall)
- Bernoulli dropout mask gamma_t decaying from 1.0 to 0.0
- Same architecture, just wider input during this phase

### T2.2: Oracle Self-Play
- 4-player self-play with oracle features visible
- PPO training with oracle critic for advantage estimation
- Entropy coefficient: 0.05-0.1 (CRITICAL -- standard defaults will fail)
- LR decay to 0.1x when gamma_t reaches 0
- Importance weight rejection for post-oracle stability

## Phase 3: PPO Self-Play

### T3.1: Self-Play Arena
- Batch hundreds of games in parallel
- Collect decision requests into GPU batches (size 256+)
- Temperature-varied seats: tau ~ Uniform(0.5, 1.5)
- 6x suit permutation on replay buffer
- Checkpoint opponent pool: save every 50 GPU hours, 3/4 seats from pool

### T3.2: PPO Training
- Clipped objective, epsilon=0.1
- GAE with oracle critic baseline (zero-sum constraint)
- Multi-lambda TD value targets
- Entropy bonus 0.05-0.1
- Max gradient norm 0.5, 4 update epochs per batch

## Phase 4: Pondering ExIt

### T4.1: Oracle Pondering (training mode)
- Oracle teacher model loaded on GPU 3
- PUCT search with K=5, oracle policy for opponent responses
- Oracle evaluates leaves with perfect information (full game state)
- No Sinkhorn needed -- oracle bypasses belief inference
- ~1ms per pondered state, ~70-80% completion rate

### T4.1b: Sinkhorn Pondering (deployment mode)
- CPU-side Sinkhorn iterations on 3x34 matrix (10-30 iterations)
- Sample opponent hands from doubly-stochastic marginals
- Blind student evaluates leaves
- Used only for Tenhou/evaluation, not training

### T4.2: PUCT Search (shared by both modes)
- Top-K=5 actions from policy
- Depth-2 search with batched GPU leaf evaluation
- Playout cap: uncertain states get full budget, obvious get zero

### T4.3: ExIt Target Pipeline
- Compute pi_ExIt from search Q-values
- KL-prioritized weighting for training
- ExIt warmup: 10% -> 40% over first 200 GPU hours
- Non-blocking: self-play never waits for pondering

## Phase 5: Evaluation

### T5.1: Internal Metrics
- Self-play Elo tracker (vs checkpoint pool)
- Policy agreement on held-out Houou games
- Value prediction MSE

### T5.2: External Evaluation  
- Head-to-head vs Mortal (1000+ games)
- Tenhou bot API integration
