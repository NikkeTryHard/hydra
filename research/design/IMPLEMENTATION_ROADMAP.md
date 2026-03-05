# HYDRA Implementation Roadmap

**SSOT**: This roadmap maps `HYDRA_FINAL.md` to sequential Rust/Burn code tasks.
Each phase has a hard GATE -- do NOT proceed to the next phase until all gate checks pass.

**Architecture**: 2-tier (24-block LearnerNet + 12-block ActorNet). No 40-block teacher.
**Training**: DRDA-wrapped ACH with oracle guiding + pondering ExIt.
**Budget**: 2000 GPU hours on 4x RTX 5000 Ada.

---

## Already Implemented (hydra-core + hydra-engine)

| Module | File | Lines | Tests | What it does |
|--------|------|------:|------:|-------------|
| Encoder | `hydra-core/src/encoder.rs` | 1450 | 31 | 85x34 observation tensor |
| Actions | `hydra-core/src/action.rs` | ~500 | 18 | 46-action space, Mortal-compatible |
| Safety | `hydra-core/src/safety.rs` | ~400 | 18 | 23 channels: genbutsu, suji, kabe |
| Bridge | `hydra-core/src/bridge.rs` | 465 | - | Game state -> encoder input |
| Simulator | `hydra-core/src/simulator.rs` | ~300 | - | Batch sim, 29K games/sec |
| Shanten | `hydra-core/src/shanten_batch.rs` | ~500 | - | Base + all 34 discards in one pass |
| Tiles | `hydra-core/src/tile.rs` | ~260 | - | Suit permutation for augmentation |
| Seeding | `hydra-core/src/seeding.rs` | ~130 | - | BLAKE3 deterministic RNG hierarchy |
| MJAI | `hydra-engine/src/mjai_event.rs` | - | - | Event types + replay |

---

## PHASE 0: Foundation (hydra-train crate)

**Goal**: Build the neural network and verify it compiles and runs on GPU.
**Time estimate**: 1-2 weeks dev time. 0 GPU hours.

### Task 0.1: Create hydra-train crate
- Add `hydra-train` to workspace in root `Cargo.toml`
- Dependencies: `burn`, `burn-tch`, `burn-train`, `hydra-core`
- Structure: `src/model/`, `src/training/`, `src/data/`, `src/search/`

### Task 0.2: SE-ResNet backbone
- File: `hydra-train/src/model/backbone.rs`
- Implement `SEResBlock`: Conv1d(256,256,k=3) -> GroupNorm(32) -> Mish -> Conv1d -> GroupNorm + SE + skip
- Implement `SEResNet<B>` parameterized by block count
- LearnerNet: `SEResNet::<B>::new(24)` (~10M params)
- ActorNet: `SEResNet::<B>::new(12)` (~5M params)
- Reference: `burn::nn::{Conv1d, GroupNorm, Linear}`, `burn::tensor::activation::mish`

### Task 0.3: Output heads
- File: `hydra-train/src/model/heads.rs`
- 8 inference heads + oracle critic (see HYDRA_FINAL Section 4.3)
- Policy: `Linear(256*34, 46)` with legal action masking (masked softmax)
- Value: `Linear(256, 1)` 
- GRP: `Linear(256, 24)` 
- Tenpai: `Linear(256, 3)` + sigmoid
- Danger: `Linear(256, 3*34)` reshaped to `[batch, 3, 34]`
- Opp-next: `Linear(256, 3*34)` + per-opponent softmax
- Score-pdf: `Linear(256, 64)` + softmax (B=64 bins, range [-50K, +60K])
- Score-cdf: `Linear(256, 64)` + sigmoid
- Oracle critic: separate `SEResNet(6)` with extended input `(85+oracle_channels)*34`

### Task 0.4: Loss functions
- File: `hydra-train/src/training/losses.rs`
- Loss weights: L_pi=1.0, L_V=0.5, L_GRP=0.2, L_tenpai=0.1, L_danger=0.1, L_opp=0.1, L_score=0.05
- Policy: CE with label smoothing support (for ExIt soft targets later)
- Value: MSE (multi-lambda TD targets come in Phase 2)
- Oracle critic: MSE with zero-sum normalization

### Task 0.5: Forward pass smoke test
- File: `hydra-train/tests/smoke_test.rs`
- Random input tensor `[batch=4, 85, 34]` in bf16
- Forward through both LearnerNet and ActorNet
- Verify output shapes match head specs
- Verify backward pass produces gradients
- Run on CPU first, then GPU if available

### GATE 0 (must pass before Phase 1):
- [ ] `cargo build --release` succeeds for hydra-train
- [ ] `cargo test --release` passes all smoke tests
- [ ] `cargo clippy --all-targets -- -D warnings` clean
- [ ] Forward+backward on random data: no NaN, no panic
- [ ] bf16 forward pass produces same-shape output as f32

---

## PHASE 1: Behavioral Cloning (50 GPU hours)

**Goal**: Train LearnerNet on expert data. Get a baseline that plays legal Mahjong.
**Depends on**: GATE 0 passed.
**Read**: HYDRA_FINAL Section 11 Phase 0, SEEDING.md, TESTING.md

### Task 1.1: Data loader
- File: `hydra-train/src/data/mjai_loader.rs`
- Input: `.mjson.gz` files from `tenhou-to-mjai` dataset (2.5M+ Houou games)
- For each game: replay via `hydra-engine::apply_mjai_event()`
- At each decision point: encode via `hydra-core::encoder::Observation::encode()`
- Extract: obs `[85,34]`, action `u8`, legal_mask `[46]`, game_result, placement, scores
- 24x online augmentation: `tile::SUIT_PERMUTATIONS` (6) x 4 seat rotations
- Output: `DataLoader<MahjongBatch>` yielding batched tensors
- Test: load 100 games, verify obs shapes, action validity, mask correctness

### Task 1.2: BC training loop
- File: `hydra-train/src/training/bc.rs`
- Adam optimizer, LR 2.5e-4, batch 2048 per GPU (4096 total on 2 GPUs)
- Train LearnerNet (24-block) on all 8 heads
- Labels: action (policy), game outcome (value), final placement (GRP), etc.
- 24 epochs over 5-6M games
- Checkpoint every epoch to `checkpoints/phase1/epoch_{N}.bin`
- Evaluate: policy agreement on 10K held-out Houou games after each epoch

### Task 1.3: Distill to ActorNet
- File: `hydra-train/src/training/distill.rs`
- KD loss: KL(LearnerNet policy || ActorNet policy) + MSE(values)
- Train ActorNet (12-block) on 1M states from the BC replay buffer
- 3 epochs, LR 1e-4

### GATE 1 (must pass before Phase 2):
- [ ] Policy agreement on held-out Houou games > 55% (Mortal-level baseline)
- [ ] Value prediction MSE < 0.5 on held-out games
- [ ] ActorNet policy agreement within 3% of LearnerNet (distillation quality)
- [ ] No NaN/Inf in any head output over 100K random states
- [ ] Legal action masking: agent never selects illegal actions (0% violation rate)

---

## PHASE 2: DRDA-Wrapped ACH Self-Play (800 GPU hours)

**Goal**: Train via game-theoretic self-play. This is the core training phase.
**Depends on**: GATE 1 passed.
**Read**: HYDRA_FINAL Section 11 Phase 2, ACH paper Algorithm 2 (Eq. 29)

### Task 2.1: Self-play arena
- File: `hydra-train/src/arena/selfplay.rs`
- Run 500+ games in parallel using `hydra-core::simulator::BatchSimulator`
- ActorNet (12-block) on GPU 2 for inference (batched, size 256+)
- Collect: (obs, action, pi_old, reward, done, player_id) trajectories
- Temperature-varied seats: each seat draws tau ~ Uniform(0.5, 1.5)
- Checkpoint opponent pool: save ActorNet every 50 GPU hours, 3/4 seats from pool
- 6x suit permutation applied to replay buffer entries

### Task 2.2: GAE advantage computation
- File: `hydra-train/src/training/gae.rs`
- Per-player advantages: A_i(s,a) = sum (gamma*lambda)^k * delta_{t+k}
- Use oracle critic V_oracle(s_full) for tighter baselines
- Normalize advantages per-minibatch: A = (A - mean) / (std + 1e-8)
- GAE params: lambda=0.95, gamma=0.995

### Task 2.3: ACH loss function
- File: `hydra-train/src/training/ach.rs`
- Implement EXACTLY per HYDRA_FINAL Section 11 Phase 2:
```
y_raw = net.forward(obs)  // raw logits
y_mean = mean(y_raw[legal_actions])
y = clamp(y_raw - y_mean, -l_th, +l_th)  // centered + clamped
pi = softmax_masked(y, legal_mask)
r = pi[a] / pi_old

if A >= 0:
    c = (r < 1+eps) AND (y[a] - y_mean < l_th)
else:
    c = (r > 1-eps) AND (y[a] - y_mean > -l_th)

L_policy = -c * eta * (y[a] / pi_old) * A
L_value = 0.5 * alpha * (V - G)^2
L_entropy = beta * sum(pi * log(pi))
```
- Hyperparams: eta in {1,2,3} (tune), eps=0.5, l_th=8, beta=5e-4, LR=2.5e-4
- ONE update epoch per batch (mandatory -- multi-epoch breaks Hedge)

### Task 2.4: DRDA wrapper
- File: `hydra-train/src/training/drda.rs`
- Maintain frozen `base_policy_logits` (detached tensor)
- Combined policy: pi = softmax(l_base + y_theta / tau_drda)
- tau_drda in {2, 4, 8} (tune in Phase -1; target median KL to base in [0.05, 0.20])
- REBASE RULE (every 25-50 GPU hours):
  1. l_base <- l_base + y_theta / tau_drda (fold residual into base)
  2. y_theta <- 0 (zero residual weights)
  3. Reset optimizer moments for y_theta parameters
- Monitor: KL(pi || pi_base) per minibatch. Alert if 95th percentile > 0.5

### Task 2.5: Oracle critic training
- File: `hydra-train/src/training/oracle_critic.rs`
- Separate small network seeing full game state (all 4 hands + wall)
- Zero-sum constraint: sum_i V_i = 0 (normalize after forward pass)
- Train alongside LearnerNet using same replay data
- Used ONLY for advantage computation, never for inference

### Task 2.6: Start cheap ExIt mid-phase
- From ~400 GPU hours into Phase 2:
- Run shallow AFBS (depth 2, P=64 particles) on 20% of states (top-2 gap < 10%)
- Store ExIt soft targets in replay buffer alongside ACH targets
- LearnerNet loss adds ExIt CE term with weight ramping 0.0 -> 0.3

### Task 2.7: Continuous distillation
- Run async distillation worker: LearnerNet -> ActorNet every 1-2 minutes
- KD loss on recent 8192 states from replay buffer
- Broadcast ActorNet weights to self-play workers after each distillation step

### GATE 2 (must pass before Phase 3):
- [ ] Self-play Elo gain: LearnerNet > ActorNet by 200+ Elo (search value confirmed)
- [ ] KL(pi || pi_base) median in [0.05, 0.20] (DRDA not too tight or loose)
- [ ] ACH gate fraction E[c] > 0.15 (updates not being over-clipped)
- [ ] Head-to-head vs BC baseline: win > 65% of placements
- [ ] Policy agreement on Houou held-out > 62% (improved from Phase 1)
- [ ] No training instability: loss variance < 3x mean over last 100 batches
- [ ] ActorNet distillation: KL drift from LearnerNet < 0.05 sustained

---

## PHASE 3: ExIt + Pondering + Full Stack (800 GPU hours)

**Goal**: Add deep search, CT-SMC beliefs, endgame solver, SaF. Push toward 10 dan.
**Depends on**: GATE 2 passed.
**Read**: HYDRA_FINAL Sections 5-9

### Task 3.1: CT-SMC exact DP sampler
- File: `hydra-train/src/search/ct_smc.rs` (pure Rust, no ML dependency)
- Implement 3D DP: state = (c1, c2, c3), c_W derived. Max 15^3 = 3,375 states.
- Log-space DP with logsumexp for numerical stability
- Backward sampling for exact correlation-correct particles
- SMC weighting: w_p = L(X_p) where L is opponent action likelihood
- Resample when ESS < 0.4P
- Test: verify marginals match Sinkhorn on 1000 random game states
- Benchmark: must run < 1ms per belief computation

### Task 3.2: AFBS search engine
- File: `hydra-train/src/search/afbs.rs`
- PUCT-guided tree search with top-K=5 actions
- LearnerNet on GPU 3 for leaf evaluation (batched)
- Root sampling: sample particles once at root, propagate through tree
- Opponent actions sampled from ActorNet policy conditioned on particle's private hand
- Caches: transposition table (public hash + belief signature), neural eval LRU
- On-turn mode: depth 4-6, P=128-256, W=64-128
- Pondering mode: depth 10-14, P=1024-4096, W=256-1024

### Task 3.3: Robust opponent modeling
- File: `hydra-train/src/search/robust_opponent.rs`
- KL-ball robustness: q_tau(a) = p(a)*exp(-Q(a)/tau) / Z
- Binary search for tau to satisfy KL(q||p) = epsilon
- N=4 opponent archetypes with soft-min: Q(a) = -tau_arch * log(sum w_i * exp(-Q_i/tau_arch))
- Epsilon calibrated from held-out opponent KL deviations (95th percentile)

### Task 3.4: Search-as-Feature (SaF) adaptor
- File: `hydra-train/src/model/saf.rs`
- g_psi: small MLP (input: search features, hidden: 32-64, output: 1 per action)
- l_final(a) = l_theta(a) + alpha_saf * g_psi(f(a)) * m(a)
- SaF-dropout: randomly zero m during training (p=0.3) even when features available
- Train g_psi first via supervised regression on delta(a) = log(pi_search) - log(pi_base)
- Then switch to joint end-to-end training

### Task 3.5: Endgame solver
- File: `hydra-train/src/search/endgame.rs`
- Trigger: wall <= 10 AND threatening signal (riichi, open tenpai)
- Pure PIMC: sample 1 draw sequence + 1 opponent action sequence per particle
- Top-mass reduction: keep particles covering 95% weight (typically P=50-100)
- Average Q over particles: Q(a) = (1/P) * sum PIMC_Rollout(a | X_p)
- Benchmark: must complete in < 50ms per decision

### Task 3.6: Hand-EV oracle features
- File: `hydra-train/src/data/hand_ev.rs` (extends encoder)
- For each discard candidate: P_tenpai(d), P_win(d), E[score|win,d], ukeire vector
- Computed by CPU using shanten_batch.rs + scoring engine
- Fed as extra input channels (Group D, ~34-68 planes)
- Belief-aware: use CT-SMC posterior expected remaining counts

### Task 3.7: ExIt target pipeline (full)
- File: `hydra-train/src/training/exit.rs`
- pi_ExIt = softmax(Q_search / tau_exit)
- KL-prioritized weighting: w(I) = KL(pi_ExIt || pi_theta). Enabled after 50 GPU hours.
- Safety valve: discard targets where Q_search(best) < V_theta - epsilon
- ExIt warmup: 10% -> 40% mixing over first 200 GPU hours of Phase 3
- Rollback safety: save Phase 2 checkpoint. Monitor Elo every 25h. Kill + rollback if Elo drops.

### Task 3.8: Population training
- File: `hydra-train/src/arena/league.rs`
- League: latest ActorNet + trailing checkpoints (last 10) + BC anchor
- 3/4 seats from checkpoint pool, 1/4 current ActorNet
- Track win rates per opponent type for diversity monitoring

### GATE 3 (must pass before deployment):
- [ ] CT-SMC DP < 1ms on real hardware (benchmark, not theoretical)
- [ ] AFBS on-turn latency < 150ms (depth 4, P=128)
- [ ] Endgame solver < 50ms (wall=10, P=100)
- [ ] Self-play Elo: LearnerNet > Phase 2 endpoint by 100+ Elo
- [ ] SaF: shallow search + SaF > shallow search alone on 50K held-out states
- [ ] ExIt labels are net-positive: mean delta(I) > 0 over 200K states
- [ ] Head-to-head vs Mortal: win > 55% of placements (1000+ games)
- [ ] Policy agreement on Houou held-out > 68%

---

## PHASE 4: Deployment + Evaluation

**Goal**: Deploy on Tenhou, measure dan, iterate.
**Depends on**: GATE 3 passed.

### Task 4.1: Inference server
- File: `hydra-train/src/inference/server.rs`
- Load ActorNet (12-block) in bf16
- Single forward pass: < 0.5ms
- Agari guard: always take legal ron/tsumo
- SaF adaptor with deployment-mode AFBS (Sinkhorn beliefs, blind evaluation)

### Task 4.2: Tenhou bot integration
- MJAI protocol over Tenhou bot API
- Time management: on-turn < 150ms, call reactions < 50ms
- Pondering during opponent turns (Sinkhorn + blind AFBS)

### Task 4.3: Evaluation suite
- Head-to-head vs Mortal (1000+ games, automated)
- Self-play Elo tracking
- Policy agreement on held-out Houou games
- Ablation tests: disable each component independently

### GATE 4 (ship criteria):
- [ ] Beat Mortal in head-to-head (> 55% placement win rate)
- [ ] Tenhou: reach 7+ dan within 500 games
- [ ] No illegal moves in 10K+ games
- [ ] Inference latency < 0.5ms sustained

---

## PHASE -1: Hard Reality Benchmarks (150 GPU hours reserve)

**Goal**: Verify assumptions BEFORE committing the full budget.
**Run BEFORE Phase 2 starts consuming serious GPU hours.

### Benchmarks to run:
- [ ] CT-SMC DP wall-clock: < 1ms for 1000 random states
- [ ] ActorNet sustained throughput: > 20 games/sec with batching
- [ ] Learner->Actor distillation: KL drift < 0.05 over 100 updates
- [ ] ACH loss: no NaN/divergence over 1000 gradient steps
- [ ] DRDA rebase: policy preserved exactly across boundary (KL < 1e-6)
- [ ] Hyperparameter sweep: eta in {1,2,3}, tau_drda in {2,4,8}

### If gates fail:
- CT-SMC slow -> fall back to Sinkhorn + generic CMPS
- Throughput low -> reduce ActorNet to 8 blocks
- ACH unstable -> fall back to PPO (entropy 0.05-0.1)
- Distillation unstable -> increase distill frequency, reduce LR

---

## Reference Implementations (DO NOT COPY CODE, reference only)
- PPO in Burn: `yunjhongwu/burn-rl-examples` (GitHub)
- Mortal training: `Equim-chan/Mortal` (AGPL, reference only)
- KataGo training: `lightvector/KataGo` (ExIt + MCTS patterns)
- ACH paper: Algorithm 2, Eq. 29 in `openreview.net/pdf?id=DTXZqTNV5nW`

## Burn API Quick Reference
- Custom training: `loss.backward()` -> `GradientsParams::from_grads()` -> `optim.step()`
- Backend: `burn-tch` with `AutodiffBackend` for training
- bf16: `set_default_dtypes_unchecked(&device, FloatDType::BF16, IntDType::I32)`
- Key modules: `burn::nn::{Conv1d, GroupNorm, Linear}`, `burn::tensor::activation::mish`
