# Community Insights: Mahjong AI Discussions

Research comp. from Reddit, JP blogs, RL comm, public AI analysis. Focus: insights direct relevant to Hydra.

> **Source volatility note:** Some refs = personal blogs (note.com, hatenablog, Ghost, nicovideo blomaga, modern-jan.com) may die. All critical datapoints (stats, arch details, p-values) reproduced inline so doc stays self-contained if links rot. Last verified: 2026-02-11.
> **Maintenance cadence:** Re-check external links + source-backed numeric claims quarterly (or before major doc releases), update timestamp after verify.

---

## 1. Mortal Strengths & Weaknesses (r/Mahjong, r/mahjongsoul)

### Confirmed Strengths

| Strength | Evidence | Source |
|----------|----------|--------|
| **~7 dan level play** | Above most Tokujou players on Tenhou | [r/Mahjong](https://www.reddit.com/r/Mahjong/comments/14ex61l/) |
| **Error detection** | Good at spotting clearly bad discards; big eval gaps = real mistake | Same thread |
| **Free & accessible** | Supports Tenhou, Mahjong Soul, Riichi City log analysis | Multiple sources |
| **4th-place avoidance** | Trained on uma 90/45/0/−135 (close to MJS ranked) | Mortal docs |

### Confirmed Weaknesses

| Weakness | Details | Hydra Relevance |
|----------|---------|-----------------|
| **Cannot explain reasoning** | No interpretable output; user must infer "why" from raw Q-values | Hydra should add explainability hooks |
| **Poor future planning** | Weak at "reading wall" and multi-turn planning; no lookahead search | Chance for search-augmented approach |
| **Sub-optimal multi-threat defense** | With multiple pushers, may choose tile safe now but dangerous if second riichi appears | Gap in multi-player defense modeling |
| **Conservative bias** | Folds more than NAGA or Akochan in same spots | Different training goals → different style |
| **Rule-based agari guard required** | NN can miss basic winning decisions; heuristic override needed | Raw NN can miss trivial game logic |
| **Not "source of truth"** | Unlike Stockfish, many decisions are preference-based; high-level players often disagree | Mahjong has many valid plays |
| **Fixed uma optimization** | Trained for one point spread; not adaptive to tournament rules | Hydra should parameterize scoring context |
| **No opponent modeling** | Treats all opponents same; cannot exploit tendencies or detect damaten | Core Hydra target gap |

### Key Quote
> "In Mahjong, there are many different perfectly playable options. Mortal may have preferences that match with certain high-level players' decisions and not with others." — r/Mahjong community

---

## 2. NAGA vs Mortal Comparison

### NAGA Architecture (Confirmed)

NAGA = **pure supervised learning**. No self-play, no RL. Uses **4 independent CNNs** (discard, call, riichi, kan), each trained on Tenhou Houou logs via imitation learning. CNN details (layers, filters, input shape) never disclosed publicly. [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) = only official technical doc; no papers, patents, conference talks found.

**Key technical features:**
- **Confidence estimation** (DeVries & Taylor 2018) — during training, low-confidence preds get penalty, corrected toward ground truth, improving calibration
- **Guided Backpropagation** (Springenberg 2014) — for interpretability, showing which input features drove each choice
- **Heuristics** — only for final-round winning judgment (avoid win causing last place); all else purely CNN output

**5 playstyle variants**, each trained on different player logs:

| Model | Style | Training Source |
|-------|-------|----------------|
| **Omega (オメガ)** | Aggressive calling | Watanabe Futoshi (M-League pro) — 100% |
| **Gamma (ガンマ)** | Defensive | One undisclosed private player |
| **Nishiki (ニシキ)** | Balanced | Multiple players (~1/3 Watanabe Futoshi) |
| **Hibakari (ヒバカリ)** | Closed-hand focused | One undisclosed private player |
| **Kagashi (カガシ)** | Extreme aggressive calling | One undisclosed private player (furo rate >40%) |

**Performance:** Current models estimated ~9-dan stable on Tenhou. Original NAGA25 reached 10-dan in 26,598 games (source unverified; this number absent from DMV article and other locatable public sources). All 5 current models reportedly beat original NAGA25. Action with NAGA rec rate <5% marked "bad move" (悪手) — stylistic judgment, not mathematical optimality claim.

**Critical implication for Hydra:** Because NAGA = pure imitation learning, it **cannot exceed training data**. Output = probability distribution of what top humans likely choose, not optimized strategy. Long-term strategy (folding, round-aware play) learned only implicitly from behavior patterns. This ceiling is why RL-based paths (Suphx, LuckyJ, Hydra) have higher upside despite NAGA polish.

**Sources:** [DMV official article](https://dmv.nico/en/articles/mahjong_ai_naga/), [note.com analysis](https://note.com/bold_myrtle4902/n/n8015e4508fe3), [witchverse.hatenablog.com](https://witchverse.hatenablog.com/entry/2025/06/02/124431), [KADOKAWA book](https://www.kadokawa.co.jp/product/322311000197) (co-authored by developer Odagiri Yuuri and pro player Watanabe Futoshi)

### Head-to-Head Differences

| Dimension | Mortal | NAGA |
|-----------|--------|------|
| **Playstyle** | More conservative/defensive | More aggressive push |
| **Riichi decisions** | Hesitant in marginal spots | Strong riichi bias when +EV |
| **Kan decisions** | Mortal, NAGA often disagree on kan timing | NAGA more aggressive on kan |
| **Accessibility** | Free, open-source | Paid, proprietary |
| **Explanation** | None (raw values only) | Human-readable analysis per discard |
| **Calibration** | 7 dan equivalent | 10 dan, with NAGA Rating metrics |
| **Push/fold** | Conservative — values position safety | Calibrated to 4th-avoidance at 7-dan rates |

### NAGA Rating System Limitations
- NAGA "match%" and "bad move rate" = imperfect strength proxies
- Suphx (9–10 dan) scored only match% 74.4 and avg NAGA Rating 86.3 — comparable to avg 7-dan in 2020
- Tencent's LuckyJ hit 10 dan with bad move rates >10% in many games (riichinotes quote: "...LuckyJ hit 10 Dan with bad move rates of >10% in many games.")
- **Takeaway**: Agreement with one AI = poor absolute-strength metric

Source: [riichinotes.blogspot.com](https://riichinotes.blogspot.com/2023/06/reviewing-my-first-50-houou-games-with.html)

---

## 3. Push/Fold Mathematics (r/Mahjong)

### Poker Pot Odds Framework for Riichi
Community member adapted poker pot odds into riichi mahjong EV framework:

- **Round EV** = expected point outcome per hand (not per game)
- **Decision**: Push if Round EV > 0 in flat spots (East 1–3)
- **Deal-in rate thresholds**: Based on tile danger (suji, kabe, genbutsu)
- **Good shape**: Tenpai with 5+ acceptance tiles → easier push
- **Bad shape**: Tenpai with ≤4 tiles → need higher reward

### Factors Beyond Round EV
NAGA accounts for 4th-avoidance, but base math starts with Round EV. Human exceptions:
1. **Exploitative folding** — opponent tendency reads
2. **Lateral movement** — point flow between other players
3. **Negative rates** — specific statistical disadvantages in current spot

**Hydra Relevance**: Score-aware + placement-aware adjustment on top of base tile EV is exact jump from "good" to "great." Confirmed Mortal gap.

Source: [r/Mahjong Push/Fold thread](https://www.reddit.com/r/Mahjong/comments/17rgvq3/)

---

## 4. LuckyJ (Tencent)

### Identity

LuckyJ (ⓃLuckyJ on Tenhou, 绝艺/JueYi brand) built by **Tencent** (AI Platform Department). Key researcher: **Haobo Fu** (Principal Research Scientist, Tencent). 绝艺 brand shared with Tencent Go AI from intl Go comps. LuckyJ reached **10-dan on Tenhou on May 30, 2023** in **1,321 games** — fastest known path to 10-dan by any AI.

### Performance

| Metric | Value | Source |
|--------|-------|--------|
| Peak Tenhou rank | 10-dan | All sources |
| Stable dan | **10.68** | [Tencent official](https://sports.sina.com.cn/go/2023-07-12/doc-imzamafw0364307.shtml) |
| Games to 10-dan | **1,321** | [haobofu.github.io](https://haobofu.github.io/) |
| vs Suphx | Statistically significantly stronger (p=0.02883) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |
| vs NAGA | Statistically significantly stronger (p=0.00003) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |

Early stats (370 games, from pro player Kihara): Avg rank 2.259, stable dan 11.25, 1st place 31.3%, last place 15.9%. Source: [ch.nicovideo.jp/kihara/blomaga/ar2149306](https://ch.nicovideo.jp/kihara/blomaga/ar2149306)

### Architecture (Reconstructed from Published Papers)

No single "LuckyJ" paper, but arch can be reconstructed from Haobo Fu publication trail:

**Component 1 — Offline Training: ACH (Actor-Critic Hedge)**
- Paper: [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) — "Actor-Critic Policy Optimization in Large-Scale Imperfect-Information Game"
- Merges deep RL with Weighted CFR for Nash Equilibrium convergence
- **Pure self-play, zero human data** — trains fully from scratch
- Lower variance than prior sampled regret methods

**Component 2 — Online Search: OLSS (Opponent-Limited Subgame Solving)**
- Paper: [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) — "Opponent-Limited Online Search for Imperfect Information Games"
- Imperfect-info subgame solving with opponent-limited tree pruning
- Orders faster than common-knowledge subgame solving
- Explicitly tested on 2-player mahjong

**Component 3 — Search-as-Feature Integration (Unpublished)**
- Search results used as **features** into policy NN; they do not directly override policy (unlike AlphaGo-style MCTS)
- Lets learned integration combine search info with trained policy for real-time strategy shifts
- Source: [Tencent official article](https://modern-jan.com/blog/luckyj_article_ja/)

**Component 4 — Training Acceleration: RVR**
- Paper: [IEEE CoG 2022](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) — "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction"
- Same team (Li, Wu, Fu, Fu, Zhao, Xing)

### Observed Playstyle

From [note.com analysis](https://note.com/comtefurapote/n/ne7c3668b6e09) and [doramahjong.org](https://doramahjong.org/?p=11393):
- **High meld rate (~35.9%)** — aggressive calling for yakuhai, honitsu, toitoi
- **Defensive priority** — keeps 2 safe tiles at 2-shanten, 1 at 1-shanten; early folds poor hands
- **Shanten backtracking** — sacrifices efficiency for expensive hands (honitsu, sanshoku, ittsuu)
- **Dama over riichi** on double-mushuji 4-5-6 waits
- **Situational play** shifts hard by rank/score from South 2 onward

### What Remains Unknown

1. Exact neural net architecture (layers, embedding dims, input encoding)
2. How ACH and OLSS adapted from 2-player to 4-player mahjong (papers show 2-player)
3. Search-as-feature integration details
4. Compute requirements and inference latency
5. Whether it uses separate models (like NAGA's 4 CNNs) or unified arch

### Comparison Table

| Aspect | NAGA | Suphx | LuckyJ |
|--------|------|-------|--------|
| **Training data** | Human expert logs | Human logs + self-play RL | **Pure self-play, zero human data** |
| **Method** | Imitation learning | Imitation → RL | Game-theoretic RL (ACH) |
| **Search** | None | Monte Carlo Policy Adaptation | **OLSS (subgame solving)** |

> **Deprecated (2026-03-03):** pMCPA (Monte Carlo Policy Adaptation) removed from inference plans. Requires ~100K trajectories per round, infeasible in real-time even with 90s idle. See RESEARCH_LOG.md entry 4.
| **Theory** | None (pattern matching) | Partial (oracle guiding) | **Nash Equilibrium convergence** |
| **Games to 10-dan** | 26,598 | 5,373 | **1,321** |
| **Stable dan** | ~9.0 (current v2) | 8.74 | **10.68** |

Source: [modern-jan.com](https://modern-jan.com/2023/09/06/luckyj_vs_naga_and_suphx/)

### Hydra Relevance

LuckyJ shows game-theoretic RL + imperfect-information online search gives far better sample efficiency and stable strength than pure RL (Suphx) or pure imitation (NAGA). Search-as-feature integration — search outputs become NN inputs instead of direct overrides — is most novel and least documented part. If Hydra adds search, OLSS = start point.

---

## 5. AI Analysis Best Practices (Community Guide)

### How to Properly Use AI Review
Key insights from [Riichi City analysis guide](https://gamesoftrobo.ghost.io/untitled-6/):

1. **Focus on process, not results**: AI can make "correct" move and still deal in — not mistake
2. **Don't aim for 100% accuracy**: Mortal authors warn accuracy % is bad skill metric; 100% match = cheating red flag
3. **Supplement with human reasoning**: AI can't explain "why" — use community + theory to fill gap
4. **Efficiency vs Value trade-off**: Mortal often picks most efficient wait, but humans may pick less efficient wait for higher value (dora targeting)
5. **Hindsight bias is enemy**: Judge decisions with info available at decision time

### Mortal Analysis Modes
- **"Last Avoidance Type" (ラス回避)**: Optimized for Mahjong Soul ranked
- **Multiple model versions**: v1 through v4 with evolving architecture
- **Integration**: Built into Riichi City as official AI analysis tool (v4)

---

## 6. Imperfect Information Game RL (r/reinforcementlearning)

### Approaches Discussed

| Approach | Description | Applicability to Mahjong |
|----------|-------------|-------------------------|
| **CFR (Counterfactual Regret Minimization)** | Poker standard; computes Nash equilibria | Game tree too huge for direct CFR in mahjong |
| **Standard RL (DQN, PPO, A2C)** | Train against static/self environment | What Mortal uses (DQN) |
| **MARL (Multi-Agent RL)** | Full multi-agent training | Expensive but theory-best |
| **Opponent modeling** | Train against hardcoded/top-tier/human policies | Avoids full MARL complexity |

### ReBeL (Meta AI)
- **Paper**: [arxiv.org/abs/2007.13544](https://arxiv.org/abs/2007.13544)
- **Key innovation**: Combines deep RL + search for imperfect information games
- **Concept**: Expands "state" into probabilistic beliefs about true state from common knowledge
- **Limitation**: Proven convergent only for 2-player zero-sum; mahjong = 4-player
- **Hydra Relevance**: Belief-state approach for opponent hand estimation aligns with Hydra opponent-model goals

---

## 7. PPO Self-Play Challenges (r/reinforcementlearning)

### The "Fearful Agent" Problem
With PPO self-play, critical failure mode:

**Symptoms**:
- Agent becomes too conservative after losses
- Focus shifts fully to loss avoidance, not winning
- In mahjong terms: folds all, never pushes

**Root Causes**:
1. **Large reward disparity** — heavy losing penalties overwhelm heuristic rewards
2. **Catastrophic forgetting** — agent forgets winning tactics while adapting to specific opponents
3. **Sparse rewards** — long games (1000+ actions) need heuristics, but these can break zero-sum balance

**Community Solutions**:

| Solution | Description |
|----------|-------------|
| **Opponent pool** | Sample from past N network states, not only latest |                                                                                                                                
| **Random opponents** | Periodically play vs random to keep base competency |
| **Reward normalization** | Balance gradual heuristics with win/loss bonuses |
| **Asymmetric bonuses** | Bonus only winner; no loser penalty |
| **Weight freezing** | Freeze opponent weights during training passes |
| **Increased exploration** | Higher entropy to find new winning lines |

**Hydra Relevance**: Mortal already shows catastrophic forgetting in [GitHub Discussion #64](https://github.com/Equim-chan/Mortal/discussions/64). Opponent pool + reward normalization apply directly.

Source: [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/comments/1c2ym5s/)

---

## 8. Self-Play Training Best Practices (HuggingFace Deep RL Course)

### Key Hyperparameters for Opponent Pool

| Parameter | Effect |
|-----------|--------|
| `window` | Saved opponent policy count. Larger = more diverse training |
| `save_steps` | Steps between saves. Higher = wider skill range in pool |
| `play_against_latest_ratio` | Prob facing current vs historical policy |
| `swap_steps` | How often opponents rotate |

### ELO as Training Metric
- **Why ELO > cumulative reward**: In adversarial games, reward depends on opponent skill. ELO measures relative skill in zero-sum setting
- **K-factor**: Max adjustment per game; controls rating volatility
- **Self-correcting**: Better opponents yield more points on win

### Core Trade-off
> Balance final policy's **skill level** and **generality** against **training stability**.

Training against slowly changing adversaries = more stable, but risks overfit to specific behaviors.

Source: [HuggingFace Deep RL Course Unit 7](https://huggingface.co/learn/deep-rl-course/unit7/self-play)

---

## 9. Japanese Community Sources

### Shanten Algorithm (Qiita — tomohxx)

Standard shanten algorithm used by Mortal and most mahjong AIs:

**Mathematical Foundation**:
- Shanten S(h) = T(h) − 1, where T = min tile exchanges to tenpai
- Distance fn: d(h, g) = ½ Σ(|h_i − g_i| + h_i − g_i) over 34 tile types
- Special-case formulas for Chiitoitsu (7 pairs) and Kokushi (13 orphans)

**DP Algorithm for Regular Hands**:
1. Break hand into 4 groups (man, pin, sou, honors)
2. Precompute partial replacement numbers for all suit combos (~5^9 states)
3. Merge groups via DP: t^(n+1)_m = min over splits of meld counts
4. Result: t^(3)_4 = shanten for 4 melds + 1 pair

**Performance**: O(1) after precompute; independent of hand size or shanten value.

Source: [Qiita (tomohxx)](https://qiita.com/tomohxx/items/75b5f771285e1334c0a5), [GitHub](https://github.com/tomohxx/shanten-number)

### Japanese Mahjong AI Development Blog (TadaoYamaoka)

Indie dev doc of building mahjong AI from scratch with PPO:

**Key Technical Points**:
- Uses **PPO** (vs Mortal's DQN) as baseline
- **Reward variance reduction**: Value model uses "global information" (incl opponent private tiles) to reduce noise from random opening hands
- **Zero-sum property**: Loss fn designed so sum of 4 players' predicted values = 0
- Referenced **LuckyJ** (Tencent unpublished AI) using search-based methods for higher strength
- **Search excluded from baseline** due to impl complexity

**Hydra Relevance**: Confirms PPO viable alt to DQN; validates reward variance reduction with global info.

Source: [TadaoYamaoka's blog](https://tadaoyamaoka.hatenablog.com/entry/2023/10/03/233925)

### Mortal User Reviews (note.com, ai-bo.jp)

JP community consensus:
- Mortal rated "excellent" (優秀) by regular NAGA users
- Main value: Free + supports Mahjong Soul log import
- Main frustration: No reasoning explanation (user must infer intent)
- Comparison verdict: NAGA stronger analysis, but costs money

---

## 10. Mortal Architecture Deep Dive

> See [MORTAL_ANALYSIS.md](MORTAL_ANALYSIS.md) for full arch analysis incl DQN head evolution (v1–v4), training loss components, distributed training, and 1v3 duplicate eval protocol.

---

## 11. Defense & Betaori Analysis

### Standard Defense Framework (riichi.wiki, community)

**Tile Safety Hierarchy** (safest → least safe):
1. **Genbutsu**: 100% safe (already discarded by riichi declarer)
2. **Suji**: ~94% safe vs riichi
3. **Kabe (wall)**: Safe when all 4 copies of connecting tiles visible
4. **Honor tiles**: Variable safety by game state
5. **Middle tiles (4-5-6)**: Most dangerous

### AI Defense Limitations
- **No damaten detection**: AIs cannot reliably detect hidden tenpai (opponent waiting without riichi)
- **Multi-player defense**: Folding vs one opponent can push danger into another
- **Score context**: Push timing depends heavily on scores/placement — Mortal uses fixed uma

### Push/Fold Decision Framework
Community consensus ("2 of 3" rule):
1. Am I in **tenpai**?
2. Do I have **good wait** (5+ tiles)?
3. Is hand **high value**?

If 2 of 3 → push. Else → fold. Extra factors: round number, current scores, tile danger while pushing.

---

## 12. Mahjong AI Landscape Summary

| AI | Level | Architecture | Open Source | Analysis | Key Trait |
|----|-------|-------------|-------------|----------|-----------|
| **Mortal** | ~7 dan | SE-ResNet + Dueling DQN | ✅ Yes | Free log review | Best open-source option |
| **NAGA** | ~9 dan (stable) | 4 CNNs, pure imitation learning | ❌ No | Paid, detailed | 5 playstyle variants trained on different players |
| **Suphx** | 8.74 dan (stable) | ResNet + Oracle guiding | ❌ No | Replay viewing only | First 10 dan; GRP + oracle pioneer |
| **LuckyJ** | **10.68 dan (stable)** | ACH (RL+CFR) + OLSS (search) | ❌ No | None | Strongest known; game-theoretic RL + online search |
| **Kanachan** | Unknown (no benchmarks) | Transformer (BERT, ~90-310M params) | ✅ Yes (⚠️ no LICENSE file) | None | Zero hand-crafted features; impractical for online RL |
| **Akochan** | ~8 dan | EV-based heuristic (not ML) | ✅ Yes | Reviewer tool | Explicit suji/kabe/genbutsu defense logic |
| **Bakuuchi** | 9 dan | ISMCTS | ❌ No | None | Legacy, outperformed |

---

## 13. Key Takeaways for Hydra

> **Ownership note:** This section captures community-observed signals + hypotheses. Canonical Mortal limitation statements live in `MORTAL_ANALYSIS.md`; current Hydra arch deltas live across `README.md`, `HYDRA_FINAL.md`, `HYDRA_RECONCILIATION.md`, and focused design docs like `OPPONENT_MODELING.md`.

### Confirmed Gaps in Existing AIs (Opportunities for Hydra)

1. **Opponent Modeling**: No existing AI models opponent tendencies or detects damaten
2. **Score/Placement Awareness**: Mortal uses fixed uma; dynamic adjustment remains open
3. **Multi-Turn Planning**: LuckyJ uses online search (OLSS, ICML 2023) and is strongest AI — but 4-player adaptation and search-as-feature integration remain unpublished. No open-source AI uses search.
4. **Explainability**: All AIs black-box; interpretable decision factors would be novel
5. **Multi-Player Defense**: Simultaneous defense vs 2+ threats poorly handled
6. **Adaptive Playstyle**: NAGA offers multiple styles but does not adapt dynamically per game

### Training Methodology Recommendations

1. **PPO over DQN**: TadaoYamaoka + community discussion suggest PPO viable, maybe preferable, for policy-based mahjong AI
2. **Reward Variance Reduction**: Use global info in value model to separate skill from luck
3. **Opponent Pool**: Essential to prevent catastrophic forgetting and "fearful agent" failure
4. **CQL for Offline**: Mortal's CQL integration prevents Q-value overestimation on unseen actions
5. **ELO Tracking**: Better progress metric than cumulative reward during self-play training
6. **1v3 Duplicate**: Gold-standard eval method; removes variance

### Community Red Flags

- **100% AI accuracy = cheating indicator**: Mortal used for real-time assistance (Akagi tool); known anti-cheat concern
- **Playstyle subjectivity**: No one "correct" play in many mahjong spots; AI agreement weak quality proxy
- **AI metrics are imperfect**: NAGA Rating, match%, bad move rate do not reliably predict real playing strength