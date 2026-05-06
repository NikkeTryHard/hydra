# R-NaD vs DRDA: Neural-Scale Evidence Report

## TL;DR

**Judges right on DRDA -- it has ZERO neural-scale experiments.** But this gap can become strength: R-NaD (algo family DRDA extends) IS proven at neural scale via DeepNash. Gap sits in DRDA paper, not in regularized dynamics approach itself.

| Algorithm | Neural-Scale Proof? | Largest Game | Compute | Open-Source? |
|-----------|-------------------|--------------|---------|--------------|
| **R-NaD** (DeepNash) | YES -- 1024 TPUs, Stratego (10^535 states) | Stratego (full 10x10) | 768 learner + 256 actor TPU nodes | NO (DeepMind internal) |
| **DRDA** (ICLR 2025) | NO -- purely tabular | 4-player Kuhn poker (6 ranks) | Dynamic programming | NO |
| **NFSP** | YES -- Texas Hold'em | Limit Texas Hold'em | Single GPU | YES (OpenSpiel) |

---

## 1. R-NaD at Neural Scale: DeepNash (Perolat et al., Science 2022)

**Paper**: [arXiv:2206.15378](https://arxiv.org/abs/2206.15378) / [Science](https://www.science.org/doi/10.1126/science.add4679)

### Architecture

DeepNash uses **U-Net / Pyramid convolutional network** with residual blocks + skip connections:

- **Input**: 10x10x82 tensor (82 stacked frames encoding private info, public info, move history)
- **Torso**: Pyramid Module with N=2 outer blocks, M=2 inner blocks
  - Outer channels: **256**
  - Inner channels: **320**
  - Uses Conv ResBlocks (3x3 kernels, bottleneck at C//2) + Deconv ResBlocks with symmetric skip connections
- **4 Output Heads**:
  1. **Value head** (Pyramid N=0, M=0) -> scalar
  2. **Deployment policy** (Pyramid N=1, M=0) -> 10x10 distribution
  3. **Piece-selection policy** (Pyramid N=1, M=0) -> 10x10 distribution
  4. **Piece-displacement policy** (Pyramid N=1, M=0) -> 10x10 distribution
- **Parameter count**: Not stated explicitly, but from architecture (256/320 channel U-Net with ~14 ResBlocks across torso + 4 heads), likely **low millions** range

### Training Infrastructure

- **Hardware**: 768 TPU nodes (learners) + 256 TPU nodes (actors) = **1,024 TPU nodes total**
- **Pipeline**: Sebulba/Podracer architecture
  - Actors: C++ environment loop with OpenSpiel interfaces
  - Replay buffer: Full-game replay, variable-length trajectories
  - Learner: JAX distributed synchronous training via `pmap`
- **Total compute**: Not stated in TPU-hours, but 1,024 TPU nodes = DeepMind scale

### Training Hyperparameters (Table 2 of paper)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (b1=0.0, b2=0.999, eps=1e-8) |
| Learning rate | 5e-5 |
| Batch size | 768 trajectories/step |
| Max training steps | **7.21 million** |
| Trajectory length | 3600 |
| eta (regularization) | **0.2** |
| Gamma averaging | 0.001 |
| Logit threshold (beta) | 2 |
| NeuRD clip | 1000 |
| Gradient clip | 1000 |

### R-NaD Iteration Schedule

R-NaD outer loop ran **165+ iterations** with this schedule:
- m <= 100: delta_m = 10,000 steps per iteration
- 100 < m <= 165: delta_m = 100,000 steps per iteration
- m > 165: delta_m = 35,000 steps per iteration

Regularization target network updates at each iteration boundary, with smooth alpha interpolation between old + new regularization targets.

### Training Stability

Key stability mechanisms:
1. **Lyapunov function**: R-NaD defines dynamics with provably decreasing Lyapunov function
2. **Exponential target averaging**: gamma=0.001 for soft target network updates
3. **V-trace**: Adapted for two-player imperfect-info (off-policy correction)
4. **NeuRD update**: Neural Replicator Dynamics for policy gradient
5. **Post-processing**: Thresholding, discretization (n=32), repetition-avoidance heuristics at test time

---

## 2. DRDA (ICLR 2025) -- PURELY TABULAR

**Paper**: [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html)
**Authors**: Runyu Lu, Yuanheng Zhu, Dongbin Zhao

### Critical Finding: NO Neural Experiments

DRDA is **purely tabular**. Paper explicitly uses dynamic programming for value computation:

> "Since per-iteration time complexity of discrete-time DRDA (SDRDA; see Algorithm 1) is standard O(|H|) when we use **dynamic programming** to compute advantage value..." (Section 5, page 8)

> "Since evaluation of value functions requires repeated dynamic programming in each iteration, we only run total of 100 iterations..." (Section 5.2, page 10)

Algorithm 1 (page 29), step 3 explicitly says: "Compute all Pr(h|pi_m) and A^i_{pi_m}(h, a^i) for all i in N (**using dynamic programming**)."

### Tested Environments (ALL small/tabular)

| Game | Type | Size |
|------|------|------|
| 2-action matrix game | NFG | 2x2 |
| 3-action bimatrix game | NFG | 3x3 |
| 3-action 3-player game | NFG | 3x3x3 (27 joint actions) |
| 3-player Kuhn poker (5 ranks) | EFG | Small |
| 4-player Kuhn poker (6 ranks) | EFG | Small-medium |
| Leduc poker variants | EFG | Small |
| Soccer grid-world | MG | ~5x4 grid |
| Adversarial/Competitive Tiger | POSG | H=2,3,4 |

### R-NaD IS a baseline in the paper

DRDA compares against R-NaD in EFG experiments. Paper notes:
> "R-NaD has multi-round learning pattern close to DRDA, but process is much slower and suffers from oscillation in 4-player scenario." (page 10)

### Neural Scaling Mentioned Only as Future Work

Paper motivation acknowledges neural scaling:
> "Last-iterate convergence... is **ideal property for further extension to deep reinforcement learning (DRL)**, as it is intractable to time-average function approximators like neural networks." (page 2)

But **no neural experiments were conducted**.

---

## 3. Open-Source Implementations

### R-NaD Implementations

baskuit/R-NaD** (50 stars, PyTorch)
- **URL**: https://github.com/baskuit/R-NaD
- **Permalink (rnad.py)**: https://github.com/baskuit/R-NaD/blob/0d163921bc597405040c33c89e151b18da68fa6e/learn/rnad.py
- **Permalink (net.py)**: https://github.com/baskuit/R-NaD/blob/0d163921bc597405040c33c89e151b18da68fa6e/nn/net.py
- **What it is**: R-NaD on abstract stochastic matrix trees (GPU-accelerated)
- **Neural nets**: MLP (2-layer) + ConvNet (ResBlock tower) implementations
- **Uses**: Full R-NaD loop with v-trace, NeuRD, entropy scheduling, identical hyperparams to DeepNash paper
- **Scale**: Consumer hardware -- built for accessible experimentation, NOT game-scale

**b) AbhinavPeri/DeepNash** (7 stars, PyTorch)
- **URL**: https://github.com/AbhinavPeri/DeepNash
- **Permalink (network.py)**: https://github.com/AbhinavPeri/DeepNash/blob/94925a04ae547a282d91e83eb48091dac65825e9/deep_nash/network.py
- **Permalink (rnad.py)**: https://github.com/AbhinavPeri/DeepNash/blob/94925a04ae547a282d91e83eb48091dac65825e9/deep_nash/rnad.py
- **What it is**: Full DeepNash architecture recreation for Stratego (4x4 variant)
- **Neural nets**: PyramidModule U-Net with ConvResBlock/DeconvResBlock, 4 heads (deployment, selection, movement, value)
- **Scale**: 4x4 Stratego variant (reduced from 10x10)

**c) Other community implementations**:
- **spktrm/pokesim**: R-NaD applied to Pokemon battles (1 star)
- **JimZhouZZY/RNaD-JunQi**: R-NaD for JunQi (Chinese military chess)
- **valvarl/deepnash-torchrl**: DeepNash using TorchRL
- **nathanlct/IIG-RL-Benchmark**: R-NaD benchmark for imperfect info games

### OpenSpiel Status

**OpenSpiel does NOT have official R-NaD impl.**

Searched exhaustively:
- `open_spiel/python/algorithms/` -- no rnad directory or file
- `open_spiel/python/jax/` -- has NFSP but no R-NaD
- GitHub code search across all google-deepmind repos -- zero R-NaD results

OpenSpiel has NFSP (`open_spiel/python/jax/nfsp.py`) + various CFR variants, but R-NaD stayed internal to DeepMind's DeepNash project.

### DRDA Implementations

**No open-source impl of DRDA exists.** GitHub search returned zero results for DRDA in game-theory context.

---

## 4. R-NaD vs NFSP at Neural Scale

No direct published comparison exists between R-NaD and NFSP at neural scale. But:

| Dimension | NFSP (Heinrich & Silver, 2016) | R-NaD (DeepNash, 2022) |
|-----------|-------------------------------|------------------------|
| Largest game | Limit Texas Hold'em (~10^18 states) | Stratego (~10^535 states) |
| Architecture | DQN + supervised policy net | U-Net Pyramid + 4 heads |
| Convergence target | Average-iterate NE | Last-iterate NE |
| Compute | Single GPU | 1,024 TPU nodes |
| Open-source | YES (OpenSpiel) | NO |
| Key advantage | Simple, well-understood | No cycling, last-iterate convergence |

R-NaD/DRDA family's key theoretical advantage is **last-iterate convergence** -- no need to average policies across training, which is impractical with neural nets. NFSP works around this with separate supervised network tracking average policy, but this is approximation.

---

## 5. Follow-Up Work After DeepNash

No published work has applied R-NaD to another game at DeepNash-scale. Community implementations (baskuit, AbhinavPeri, etc.) are all smaller-scale experiments. DeepMind itself noted R-NaD "can be directly applied to other two-player zero-sum games" but has not published such follow-ups.

DRDA (ICLR 2025) is main theoretical follow-up, extending R-NaD's regularized dynamics to multiplayer POSGs, but stays purely tabular.

---

## 6. Strategic Implications for Your Proposal

### The judges' critique is valid but addressable:

1. **DRDA itself is unproven at neural scale** -- factually correct. DRDA paper has zero neural experiments.

2. **BUT R-NaD (parent algorithm) IS proven at neural scale** -- DeepNash is proof. 1,024 TPUs, Stratego (10^535 states), published in Science.

3. **DRDA's key contribution is theoretical** -- it provides last-iterate convergence guarantees + extends to multiplayer, which R-NaD alone does not formally provide for general POSG case.

### Suggested response to reviewers:

> "While DRDA itself has only been validated in tabular settings (Lu et al., ICLR 2025), closely related R-NaD algorithm -- which DRDA directly extends -- has been validated at unprecedented neural scale in DeepNash (Perolat et al., Science 2022), using U-Net pyramid architecture trained on 1,024 TPU nodes for 7.21M steps on Stratego (10^535 states). DRDA's theoretical extensions (last-iterate convergence, multiplayer POSG support) are algorithmically lightweight modifications to R-NaD's reward transformation scheme, and neural-scale infrastructure (v-trace, NeuRD updates, entropy scheduling) transfers directly. Our approach specifically leverages these proven neural-scale components while incorporating DRDA's improved convergence properties."

### What you could also do:
- Reference baskuit/R-NaD as evidence algorithm is implementable outside DeepMind
- Note OpenSpiel lacks R-NaD, making community implementations even more valuable
- Emphasize that YOUR contribution would be one of first neural-scale DRDA implementations -- this is FEATURE, not bug