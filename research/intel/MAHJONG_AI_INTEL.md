# Mahjong AI Intel

Agent routing doc. Owns competitor/community intel for Mortal, NAGA, LuckyJ, tactical gaps, source volatility, and cross-field transfer ideas. Current impl truth lives in `../../docs/` and `../design/`; this file is research context, not code contract.

## Hard boundary: AGPL / Mortal-Policy

- `Equim-chan/Mortal`, `Nitasurin/Mortal-Policy`, `smly/mjai.app`: AGPL-3.0. **No code copy, no translation, no derived impl.**
- Safe use: observe public behavior, cite facts, compare architectures, run tools as external programs if license path approved separately.
- Unsafe use: porting structs/functions/loss code, copying encoders, copying PPO fork details, or using AGPL snippets as impl template.
- Prefer MIT/Apache/BSD refs for code: `xiangting`, `mahc`, `agari`, `mahjax`, `RiichiEnv`, `OpenSpiel`, `CleanRL`, `ort`, `tract`, `candle`, `burn`.

## Competitor map

| AI | Level / status | Method | Source | Hydra lesson |
|---|---:|---|---|---|
| Mortal | ~7 dan; best open-source reviewer | SE-ResNet + Dueling DQN + CQL + GRP reward | AGPL, public | Strong baseline; gaps = opponent modeling, damaten, placement thresholds, multi-threat defense. Ref only. |
| Mortal-Policy | PPO fork of Mortal | actor/critic, GroupNorm, policy loss | AGPL, public | PPO transition proves route exists; do not derive. Use only as high-level signal. |
| NAGA | ~9 dan stable; paid | 4 independent CNNs; pure supervised Houou imitation; confidence calibration; guided backprop | proprietary | Strong analysis UX + styles; imitation ceiling. Agreement metrics not strength truth. |
| Suphx | 8.74 stable; first 10-dan | imitation -> RL, GRP, oracle guiding, pMCPA historically | papers only | GRP and privileged training useful; pMCPA not viable for realtime. |
| LuckyJ | 10.68 stable; 10-dan in 1,321 games | ACH + OLSS + search-as-feature [inference] | papers/articles, no code | Strongest signal: game-theoretic RL + imperfect-info search beats pure imitation/RL. |
| Akochan | ~8 dan | EV heuristics + explicit betaori | open | Defense formulas useful as concept; verify license before code use. |
| Kanachan | unknown | large transformer, no handcrafted features | public, no license | Research ref only; likely impractical for Hydra online RL. |

## Mortal: compact facts

- Obs v4: `(1012, 34)` public feature planes. Oracle infra can make `(1229, 34)` by appending `(217,34)` hidden/wall planes, but public train path does not enable oracle mode.
- Action space: 46. `0..36` discard / kan-tile select incl aka; `37` riichi; `38..40` chi low/mid/high; `41` pon; `42` kan trigger; `43` agari; `44` ryuukyoku; `45` pass. Hydra uses same map.
- Backbone: ResNet + channel attention. Dueling DQN: `Q = V + A - mean(A)`. v4 uses one `Linear(1024, 47)`, then splits V/A post-hoc.
- Reward: GRP predicts placement distribution; reward = delta expected placement value with `[3, 1, -1, -3]` vector.
- Offline loss: DQN + CQL. CQL term is `logsumexp(q_out) - mean(q_taken)`; off during online mode.
- Q-targets: Monte Carlo returns using final kyoku reward; no TD bootstrap, no target net, no double-DQN.
- GRP: pretrained separately on 24 placement permutations, then frozen during DQN train.
- Shanten: tomohxx table method, base-5 indexing. Suit table 1,940,777 entries; honor table 78,032 entries.
- Single-player EV: per-discard tenpai, win, point-EV curves over future turns. Strong weighted-efficiency feature, but ignores opponents and ron (`is_ron: false` in SP calc).
- Suit augmentation: 6x man/pin/sou permutation during train. Hydra should keep.
- Training fragility: online train can hang; long self-play can catastrophically forget; Q-values collapse toward equal values.

## Mortal known gaps

| Gap | Evidence / symptom | Hydra action |
|---|---|---|
| No explicit opponent model | Single-player tables; no tendency/profile tracking | Add belief/tenpai/danger heads; keep opponent history sequence-aware. |
| Damaten blind spot | No silent-tenpai estimator; community reports silent high-value deal-ins | Train tenpai labels from hidden info; expose calibrated per-opponent tenpai prob. |
| Multi-threat defense weak | Can fold vs one player into danger vs another | Danger = 3 opponents x 34 tiles; decision uses max/mixture risk. |
| Orras / placement threshold errors | Issue #111; over-defensive as 2nd/3rd, imprecise score thresholds | Encode exact overtake/avoid-last thresholds, not only normalized scores/rank. |
| Early-riichi push errors | Underestimates early riichi danger [community] | Threat model uses turn, riichi timing, wait prior. |
| Efficiency over yaku/value | Drops value tiles for shanten; bad in comeback spots | Value/yaku planning features; placement-weighted reward. |
| Coarse score encoding | dual scale 100k/30k loses fine resolution high-score late game | Use exact delta-to-rank and uma/oka config. |
| Reviewer rating over-penalizes near ties | Boltzmann over Q spreads mass; human choice can be equivalent | Report EV bands / near-equivalence, not one true move. |

## NAGA intel

- Architecture: pure supervised learning; 4 CNNs for discard, call, riichi, kan. CNN details undisclosed.
- Training: Tenhou Houou imitation, no self-play/RL. Cannot exceed data except via better calibration/ensembling [inference].
- Features: confidence estimation from DeVries/Taylor; guided backprop for interpretability.
- Heuristics: final-round winning judgment to avoid wins causing last; otherwise CNN output.
- Styles: Omega aggressive calling; Gamma defensive; Nishiki balanced; Hibakari closed-hand; Kagashi extreme calling.
- Metrics caution: NAGA match %, bad-move rate, and Rating are agreement-with-NAGA metrics. Suphx and LuckyJ can score low agreement while being strong.
- Product lesson: analysis UX matters. Users value explanations, style labels, and visible alternatives. Hydra should output decision factors and uncertainty bands.

## LuckyJ intel

- Identity: Tencent / JueYi brand, researcher Haobo Fu. Tenhou 10-dan reached 2023-05-30 in 1,321 games; reported stable dan 10.68.
- Public method reconstruction:
  - ACH: Actor-Critic Hedge, deep RL + weighted CFR; pure self-play, no human data.
  - OLSS: Opponent-Limited Online Search; imperfect-info subgame solving with opponent-limited pruning, tested publicly on 2-player mahjong.
  - Search-as-feature: search outputs feed policy NN, not hard override [article-based inference].
  - RVR: reward variance reduction work from same team.
- Playstyle reports: high meld rate (~35.9%), keeps safe tiles before tenpai, early folds poor hands, shanten backtracking for honitsu/sanshoku/ittsuu, dama on some central double-musuji waits, strong South-round score adaptation.
- Unknowns: exact NN, 4-player OLSS adaptation, search feature schema, compute/latency, single vs multiple models.
- Hydra implication: OLSS-like features are research target; direct CFR on full 4p mahjong is too large. Use belief-limited / opponent-limited search, then let policy consume search summaries.

## Tactical gaps: what is solved vs open

| Domain | Current state | Gap severity | Hydra route |
|---|---|---:|---|
| Shanten | Solved by DP/table lookup. | Low | Use exact lib/table; do not learn shanten. |
| Ukeire | Solved count, but unweighted. | Low | Exact visible-count acceptance. |
| Weighted efficiency | Mortal SP EV strong but single-player. | Medium | Combine SP EV with opponent danger and placement. |
| Suji/kabe/genbutsu | Heuristics known; Mortal implicit; Akochan explicit risk arrays. | Med-high | Bootstrap danger head with empirical priors; update from visible counts. |
| Betaori | Akochan has explicit fold sequence EV; Mortal implicit. | High | Learned danger + explicit push/fold comparison. |
| Damaten detection | No public explicit estimator. | High | Tenpai head trained from perfect-info logs; calibrate by turn/call/discard shift. |
| Placement-aware play | GRP helps; edge cases fail. | Medium | Exact rank-threshold features + uma/oka parametrization. |
| Yaku planning | Mostly implicit; Tjong fan-backward is rare explicit attempt. | Low-med | Keep implicit initially; add yaku target aux if value errors persist. |
| Call efficiency | Common spots learned; rare tails weak. | Medium | EV delta: speed gain - riichi loss - info leak - defense loss. |
| Riichi timing | Easy spots solved; marginal late/bad-wait spots hard. | Low-med | Compare EV(riichi), EV(dama), EV(fold) under placement. |
| Opponent hand reading | NN implicit; exact Bayes intractable. | Medium | Sequence model over tedashi/tsumogiri, calls, riichi tile; belief head. |

Defense priors worth preserving:

| Tile class vs riichi | Approx deal-in rate |
|---|---:|
| Genbutsu | 0% |
| 3rd visible honor | ~0.3% |
| Suji terminal 1/9 | ~1.9% |
| Nakasuji 4/5/6 | ~2.4% |
| Suji 2/8 | ~4.0% |
| Suji 3/7 | ~5.6% |
| Non-suji terminal | ~8.0% |
| Half-suji 4/5/6 | ~8.1% |
| Non-suji 4/5/6 | ~13.9% |

Push/fold community baseline: push if at least 2 of 3 hold: tenpai, good wait (5+ outs), high value. Then adjust by placement, tile danger, multiple threats, ippatsu, remaining draws, noten penalty.

## Cross-field transfer ideas

Ranked by direct usefulness:

1. **Glosten-Milgrom sequential trade model**: exact structural match. Discards = trades revealing private info; not discarding tile is adverse-selection signal. Implement as Bayesian update over `P(opponent holds tile j | action history)`, with per-player "strategic vs noise" parameter.
2. **Rao-Blackwellized particle filter**: sample remaining tile-count profiles, analytically integrate opponent assignments via hypergeometric probabilities. Big variance reduction vs sampling full hidden hands.
3. **Active inference / expected free energy**: action value = pragmatic reward plus information gain minus ambiguity. Useful for training objective/search features; compute risk high.
4. **Sparse Bayesian hand recovery**: opponent hand vector sparse over 34 tile types. Classical compressed sensing not directly valid because observations are strategic/nonlinear, but L1/sparsity regularization can prevent diffuse belief heads.
5. **Low-rank / tomography analogy**: weak. Maybe model opponent style as low-rank latent types; skip unless needed.

## Ecosystem / source volatility

- Source volatility: Reddit, note.com, hatenablog, Ghost, nicovideo blomaga, modern-jan links may rot. Critical numeric claims are preserved here; re-verify before publication or major design decision.
- `REFERENCES.md` remains citation/source ledger; detailed bibliography stays there.
- Dataset summary preserved: training-data plan referenced ~6.6M high-rank 4p hanchan across Tenhou Houou, Majsoul Throne, Majsoul Jade; verify against current data docs before using as operational fact.
- MJAI compatibility remains important because Mortal/mjai.app ecosystem uses it.
- Synthetic envs: `mjx` (~100x faster than Mjai, Gym/gRPC), `mahjax` (~1.6M steps/s on 8xA100 reported) useful for self-play experiments.
- Eval: `mjai-reviewer` Apache-2.0 can analyze move quality; 1v3 duplicate evaluation reduces variance.

## Safe integration priority

P0 direct/use soon:
- `xiangting` (MIT): shanten.
- `MahjongRepository/mahjong` (MIT): Python scoring oracle/dev dep.
- `ort` (Apache-2.0): Rust ONNX inference when deploy path exists.
- `CleanRL` (MIT): PPO reference patterns.

P1 references:
- `agari` (Cargo says MIT; no LICENSE): scoring ref only unless license verified.
- `mahc` (BSD-3): scoring/fu enum ref.
- `mahjax` (Apache-2.0), `RiichiEnv` (Apache-2.0): env/training loop references.
- `OpenSpiel` (Apache-2.0): self-play/ELO/opponent sampling architecture.
- `Meowjong` (MIT): sanma reference.
- `Microsoft Olive` (MIT), ONNX Runtime quantization, TensorRT Model Optimizer: inference optimization.

P2 watch/ref:
- `mahjong_ev` unlicensed: concepts only unless licensed.
- `torchdistill` MIT: oracle -> blind distillation.
- `tempai-core` MIT: cross-lang rules architecture.
- `lizhisim` MIT: simulator watch.
- `burn`/`candle`: long-term native Rust ML options.

## Training red flags / practices

- PPO self-play can produce "fearful agent": folds all, stops pursuing wins. Causes: heavy loss penalties, sparse rewards, catastrophic forgetting.
- Stabilizers: opponent pool, random/weak opponents for base competency, reward normalization, asymmetric winner bonus, freeze opponent weights during rollout windows, entropy/exploration schedule.
- ELO > cumulative reward for adversarial progress tracking.
- `sample_reuse`: avoid stale sample reuse unless measured safe.
- Online phase may peak then degrade; keep checkpoints, evaluate vs fixed pool, stop by held-out eval not train reward.
- 100% AI agreement is suspicious and not quality target; many riichi decisions are preference-equivalent.

## Agent read order

- Need competitor/tactical gap context: read this file.
- Need citations/source bibliography: read `REFERENCES.md` (may still contain old intel links).
- Need current code behavior/contracts: read `../../docs/GAME_ENGINE.md`, `../../docs/TRAINING_RUNBOOK.md`, crate READMEs.
- Need research architecture rationale: read `../design/HYDRA_FINAL.md`, `../design/HYDRA_RECONCILIATION.md`, `../design/HYDRA_ARCHIVE.md`.
