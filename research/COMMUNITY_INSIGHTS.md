# Community Insights: Mahjong AI Discussions

Research compilation from Reddit, Japanese blogs, RL communities, and public AI analysis discussions. Focused on insights directly relevant to Hydra's development.

> **Source volatility note:** Several references link to personal blogs (note.com, hatenablog, Ghost, nicovideo blomaga, modern-jan.com) that may go offline. All critical data points (statistics, architecture details, p-values) are reproduced inline so this document remains self-contained even if external links rot. Last verified: 2026-02-11.
> **Maintenance cadence:** Re-verify external links and source-backed numeric claims quarterly (or before major documentation releases), and update this timestamp when verification is completed.

---

## 1. Mortal Strengths & Weaknesses (r/Mahjong, r/mahjongsoul)

### Confirmed Strengths

| Strength | Evidence | Source |
|----------|----------|--------|
| **~7 dan level play** | Better than vast majority of Tokujou players on Tenhou | [r/Mahjong](https://www.reddit.com/r/Mahjong/comments/14ex61l/) |
| **Error detection** | Consistent at identifying clearly bad discards — large eval differences = real mistakes | Same thread |
| **Free & accessible** | Supports Tenhou, Mahjong Soul, and Riichi City log analysis | Multiple sources |
| **4th-place avoidance** | Trained on uma distribution 90/45/0/−135 (similar to MJS ranked) | Mortal documentation |

### Confirmed Weaknesses

| Weakness | Details | Hydra Relevance |
|----------|---------|-----------------|
| **Cannot explain reasoning** | No interpretable output — users must infer "why" from raw Q-values | Hydra should consider explainability hooks |
| **Poor future planning** | Struggles with "reading the wall" and multi-turn planning; no lookahead search | Opportunity for search-augmented approach |
| **Sub-optimal multi-threat defense** | When multiple opponents push, may pick tiles safe for now but dangerous if second riichi appears | Multi-player defense modeling gap |
| **Conservative bias** | Recommends folding more often than NAGA or Akochan in equivalent spots | Different training objectives lead to different playstyles |
| **Rule-based agari guard required** | Neural network occasionally fails at basic winning decisions; heuristic override is needed | Raw NN may miss trivial game logic |
| **Not a "source of truth"** | Unlike Stockfish in chess, many decisions are preference-based; high-level players frequently disagree | Mahjong has inherently multiple "correct" plays |
| **Fixed uma optimization** | Trained for one specific point spread; doesn't adapt to different tournament rules | Hydra should parameterize scoring context |
| **No opponent modeling** | Treats all opponents identically; cannot exploit tendencies or detect damaten | Core gap Hydra aims to fill |

### Key Quote
> "In Mahjong, there are many different perfectly playable options. Mortal may have preferences that match with certain high-level players' decisions and not with others." — r/Mahjong community

---

## 2. NAGA vs Mortal Comparison

### NAGA Architecture (Confirmed)

NAGA is a **pure supervised learning system** — no self-play, no reinforcement learning. It uses **4 independent CNNs** (discard, call, riichi, kan), each trained on Tenhou Houou table game logs via imitation learning. The CNN architecture details (layers, filters, input shape) have never been publicly disclosed. The [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) is the sole official technical document; there are no academic papers, patents, or conference presentations.

**Key technical features:**
- **Confidence estimation** (DeVries & Taylor 2018) — during training, low-confidence predictions incur a penalty and are corrected toward ground truth, improving calibration
- **Guided Backpropagation** (Springenberg 2014) — used for interpretability, visualizing which input features drove each decision
- **Heuristics** — only for final-round winning judgment (avoiding wins that result in last place); everything else is purely CNN output

**5 playstyle variants**, each trained on different players' game records:

| Model | Style | Training Source |
|-------|-------|----------------|
| **Omega (オメガ)** | Aggressive calling | Watanabe Futoshi (M-League pro) — 100% |
| **Gamma (ガンマ)** | Defensive | One undisclosed private player |
| **Nishiki (ニシキ)** | Balanced | Multiple players (~1/3 Watanabe Futoshi) |
| **Hibakari (ヒバカリ)** | Closed-hand focused | One undisclosed private player |
| **Kagashi (カガシ)** | Extremely aggressive calling | One undisclosed private player (furo rate >40%) |

**Performance:** Current models estimated ~9-dan stable on Tenhou. The original NAGA25 reached 10-dan in 26,598 games (source unverified — this number does not appear in the DMV article or any locatable public source). All 5 current models reportedly outperform the original NAGA25. An action with NAGA recommendation rate <5% is flagged as a "bad move" (悪手) — this is a stylistic judgment, not a mathematical optimality claim.

**Critical implication for Hydra:** Because NAGA is pure imitation learning, it **cannot exceed its training data**. Its output is a probability distribution reflecting what top humans would likely choose, not an optimized strategy. Long-term strategy (folding, round-aware play) is learned implicitly from behavioral patterns. This fundamental ceiling is why RL-based approaches (Suphx, LuckyJ, and Hydra) have higher potential despite NAGA's commercial polish.

**Sources:** [DMV official article](https://dmv.nico/en/articles/mahjong_ai_naga/), [note.com analysis](https://note.com/bold_myrtle4902/n/n8015e4508fe3), [witchverse.hatenablog.com](https://witchverse.hatenablog.com/entry/2025/06/02/124431), [KADOKAWA book](https://www.kadokawa.co.jp/product/322311000197) (co-authored by developer Odagiri Yuuri and pro player Watanabe Futoshi)

### Head-to-Head Differences

| Dimension | Mortal | NAGA |
|-----------|--------|------|
| **Playstyle** | More conservative/defensive | More aggressive push tendencies |
| **Riichi decisions** | Hesitant in marginal spots | Strongly favors riichi when +EV |
| **Kan decisions** | Mortal and NAGA frequently disagree on kan timing | NAGA tends more toward aggressive kan |
| **Accessibility** | Free, open-source | Paid, proprietary |
| **Explanation** | None (raw values only) | Human-readable analysis per discard |
| **Calibration** | 7 dan equivalent | 10 dan, with NAGA Rating metrics |
| **Push/fold** | Conservative — values position safety | Calibrated to 4th-avoidance at 7-dan rates |

### NAGA Rating System Limitations
- NAGA's "match%" and "bad move rate" metrics are imperfect proxies for actual strength
- Suphx (9–10 dan) only scored match% of 74.4 and average NAGA Rating of 86.3 — stats comparable to average 7-dan in 2020
- Tencent's LuckyJ hit 10 dan with bad move rates >10% in many games (riichinotes quote: "...LuckyJ hit 10 Dan with bad move rates of >10% in many games.")
- **Takeaway**: Agreement with a specific AI is a poor metric for absolute strength

Source: [riichinotes.blogspot.com](https://riichinotes.blogspot.com/2023/06/reviewing-my-first-50-houou-games-with.html)

---

## 3. Push/Fold Mathematics (r/Mahjong)

### Poker Pot Odds Framework for Riichi
A community member adapted poker pot odds into a riichi mahjong EV estimation framework:

- **Round EV** = expected point outcome per hand (not per game)
- **Decision**: Push if Round EV > 0 in flat positions (East 1–3)
- **Deal-in rate thresholds**: Based on tile danger level (suji, kabe, genbutsu)
- **Good shape**: Tenpai with 5+ tiles acceptance → more pushable
- **Bad shape**: Tenpai with ≤4 tiles → requires higher reward to justify

### Factors Beyond Round EV
NAGA accounts for 4th-avoidance but base math starts with Round EV. Human exceptions:
1. **Exploitative folding** — opponent tendency reads
2. **Lateral movement** — how points flow between other players
3. **Negative rates** — specific statistical disadvantages in the current position

**Hydra Relevance**: Score-aware and placement-aware adjustments on top of base tile EV is exactly what makes an AI jump from "good" to "great." This is a confirmed gap in Mortal.

Source: [r/Mahjong Push/Fold thread](https://www.reddit.com/r/Mahjong/comments/17rgvq3/)

---

## 4. LuckyJ (Tencent AI Lab)

### Identity

LuckyJ (ⓝLuckyJ on Tenhou, 绝艺/JueYi brand) is developed by **Tencent AI Lab**. Key researcher: **Haobo Fu** (Principal Research Scientist, Tencent AI Lab). The 绝艺 brand is shared with Tencent's Go AI that competed in international Go competitions. LuckyJ achieved **10-dan on Tenhou on May 30, 2023** in only **1,321 games** — the most efficient path to 10-dan by any AI.

### Performance

| Metric | Value | Source |
|--------|-------|--------|
| Peak Tenhou rank | 10-dan | All sources |
| Stable dan | **10.68** | [Tencent official](https://sports.sina.com.cn/go/2023-07-12/doc-imzamafw0364307.shtml) |
| Games to 10-dan | **1,321** | [haobofu.github.io](https://haobofu.github.io/) |
| vs Suphx | Statistically significantly stronger (p=0.02883) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |
| vs NAGA | Statistically significantly stronger (p=0.00003) | [modern-jan.com](https://modern-jan.com/blog/luckyj_article_ja/) |

Early stats (370 games, from pro player Kihara): Average rank 2.259, stable dan 11.25, 1st place 31.3%, last place 15.9%. Source: [ch.nicovideo.jp/kihara/blomaga/ar2149306](https://ch.nicovideo.jp/kihara/blomaga/ar2149306)

### Architecture (Reconstructed from Published Papers)

There is **no single "LuckyJ" paper**, but the architecture is reconstructable from Haobo Fu's publication trail:

**Component 1 — Offline Training: ACH (Actor-Critic Hedge)**
- Paper: [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) — "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"
- Merges deep RL with Weighted CFR for Nash Equilibrium convergence
- **Pure self-play, zero human data** — trains entirely from scratch
- Lower variance than previous sampled regret methods

**Component 2 — Online Search: OLSS (Opponent-Limited Subgame Solving)**
- Paper: [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) — "Opponent-Limited Online Search for Imperfect Information Games"
- Imperfect-info subgame solving with opponent-limited tree pruning
- Orders of magnitude faster than common-knowledge subgame solving
- Explicitly tested on 2-player mahjong

**Component 3 — Search-as-Feature Integration (Unpublished)**
- Search results are input as **features** into the policy neural network — they don't directly override the policy (unlike AlphaGo-style MCTS)
- Enables learned integration of search information with trained policy for real-time strategy adjustment
- Source: [Tencent official article](https://modern-jan.com/blog/luckyj_article_ja/)

**Component 4 — Training Acceleration: RVR**
- Paper: [IEEE CoG 2022](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) — "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction"
- Same team (Li, Wu, Fu, Fu, Zhao, Xing)

### Observed Playstyle

From [note.com analysis](https://note.com/comtefurapote/n/ne7c3668b6e09) and [doramahjong.org](https://doramahjong.org/?p=11393):
- **High meld rate (~35.9%)** — aggressive calling for yakuhai, honitsu, toitoi
- **Defensive priority** — keeps 2 safe tiles at 2-shanten, 1 at 1-shanten; practices early folding on poor hands
- **Shanten backtracking** — reduces efficiency to pursue expensive hands (honitsu, sanshoku, ittsuu)
- **Dama over riichi** on double-mushuji 4-5-6 waits
- **Situational play** shifts dramatically based on rank/score from South 2 onwards

### What Remains Unknown

1. Exact neural network architecture (layers, embedding dims, input encoding)
2. How ACH and OLSS were adapted from 2-player to 4-player mahjong (the papers demonstrate on 2-player)
3. Search-as-feature integration details
4. Compute requirements and inference latency
5. Whether it uses separate models (like NAGA's 4 CNNs) or a unified architecture

### Comparison Table

| Aspect | NAGA | Suphx | LuckyJ |
|--------|------|-------|--------|
| **Training data** | Human expert logs | Human logs + self-play RL | **Pure self-play, zero human data** |
| **Method** | Imitation learning | Imitation → RL | Game-theoretic RL (ACH) |
| **Search** | None | Monte Carlo Policy Adaptation | **OLSS (subgame solving)** |
| **Theory** | None (pattern matching) | Partial (oracle guiding) | **Nash Equilibrium convergence** |
| **Games to 10-dan** | 26,598 | 5,373 | **1,321** |
| **Stable dan** | ~9.0 (current v2) | 8.74 | **10.68** |

Source: [modern-jan.com](https://modern-jan.com/2023/09/06/luckyj_vs_naga_and_suphx/)

### Hydra Relevance

LuckyJ proves that combining game-theoretic RL with imperfect-information online search yields dramatically better sample efficiency and higher stable performance than pure RL (Suphx) or pure imitation (NAGA). The search-as-feature integration — where search outputs become neural network inputs rather than direct policy overrides — is the most novel and least documented piece. If Hydra ever adds search, OLSS is the starting point.

---

## 5. AI Analysis Best Practices (Community Guide)

### How to Properly Use AI Review
Key insights from the [Riichi City analysis guide](https://gamesoftrobo.ghost.io/untitled-6/):

1. **Focus on process, not results**: AI makes "correct" moves that sometimes deal into hands — that's not a mistake
2. **Don't aim for 100% accuracy**: Mortal's authors warn against using accuracy % as skill metric; 100% match = cheating red flag
3. **Supplement with human reasoning**: AI can't explain "why" — use community and theory to fill gaps
4. **Efficiency vs Value trade-off**: Mortal often picks most efficient wait, but humans may correctly choose less efficient wait for higher point value (dora targeting)
5. **Hindsight bias is the enemy**: Evaluate decisions with info available at decision time

### Mortal Analysis Modes
- **"Last Avoidance Type" (ラス回避)**: Optimized for Mahjong Soul ranked play
- **Multiple model versions**: v1 through v4 with evolving architecture
- **Integration**: Built into Riichi City as official AI analysis tool (v4)

---

## 6. Imperfect Information Game RL (r/reinforcementlearning)

### Approaches Discussed

| Approach | Description | Applicability to Mahjong |
|----------|-------------|-------------------------|
| **CFR (Counterfactual Regret Minimization)** | Standard for poker; computes Nash equilibria | Game tree too large for direct CFR in mahjong |
| **Standard RL (DQN, PPO, A2C)** | Train against static/self environment | What Mortal uses (DQN) |
| **MARL (Multi-Agent RL)** | Full multi-agent training | Expensive but theoretically ideal |
| **Opponent modeling** | Train against hardcoded/top-tier/human policies | Avoids full MARL complexity |

### ReBeL (Meta AI)
- **Paper**: [arxiv.org/abs/2007.13544](https://arxiv.org/abs/2007.13544)
- **Key innovation**: Combines deep RL + search for imperfect information games
- **Concept**: Expands "state" to probabilistic beliefs about actual state based on common knowledge
- **Limitation**: Proven convergent only for 2-player zero-sum; mahjong is 4-player
- **Hydra Relevance**: Belief-state approach for opponent hand estimation aligns with Hydra's opponent modeling goals

---

## 7. PPO Self-Play Challenges (r/reinforcementlearning)

### The "Fearful Agent" Problem
When using PPO with self-play, a critical failure mode occurs:

**Symptoms**:
- Agent becomes overly conservative after experiencing losses
- Focuses entirely on loss avoidance rather than winning
- In mahjong terms: folds everything, never pushes for wins

**Root Causes**:
1. **Large reward disparity** — heavy penalties for losing overwhelm heuristic rewards
2. **Catastrophic forgetting** — agent forgets winning tactics as it adapts to specific opponents
3. **Sparse rewards** — long games (1000+ actions) need heuristics but these can break zero-sum balance

**Community Solutions**:

| Solution | Description |
|----------|-------------|
| **Opponent pool** | Sample from past N network states, not just latest | 
| **Random opponents** | Periodically play vs random to maintain basic competency |
| **Reward normalization** | Balance gradual heuristics with win/loss bonuses |
| **Asymmetric bonuses** | Bonus only to winner; no penalty to loser |
| **Weight freezing** | Freeze opponent weights during training passes |
| **Increased exploration** | Higher entropy to discover new winning strategies |

**Hydra Relevance**: Mortal already has catastrophic forgetting issues documented in [GitHub Discussion #64](https://github.com/Equim-chan/Mortal/discussions/64). The opponent pool and reward normalization techniques are directly applicable.

Source: [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/comments/1c2ym5s/)

---

## 8. Self-Play Training Best Practices (HuggingFace Deep RL Course)

### Key Hyperparameters for Opponent Pool

| Parameter | Effect |
|-----------|--------|
| `window` | Number of saved opponent policies. Larger = more diverse training |
| `save_steps` | Steps between saves. Higher = wider skill range in pool |
| `play_against_latest_ratio` | Probability of facing current vs historical policy |
| `swap_steps` | How often opponents rotate |

### ELO as Training Metric
- **Why ELO > cumulative reward**: In adversarial games, reward depends on opponent skill. ELO measures relative skill in zero-sum context
- **K-factor**: Maximum adjustment per game; controls rating volatility
- **Self-correcting**: Better opponents yield more points on victory

### Core Trade-off
> Balance final policy's **skill level** and **generality** against **training stability**.

Training against slowly-changing adversaries = more stable but risk of overfitting to specific behaviors.

Source: [HuggingFace Deep RL Course Unit 7](https://huggingface.co/learn/deep-rl-course/unit7/self-play)

---

## 9. Japanese Community Sources

### Shanten Algorithm (Qiita — tomohxx)

The standard shanten algorithm used by Mortal and most other mahjong AIs:

**Mathematical Foundation**:
- Shanten S(h) = T(h) − 1, where T = minimum tile exchanges to tenpai
- Distance function: d(h, g) = ½ Σ(|h_i − g_i| + h_i − g_i) over 34 tile types
- Special-case formulas for Chiitoitsu (7 pairs) and Kokushi (13 orphans)

**DP Algorithm for Regular Hands**:
1. Break hand into 4 groups (man, pin, sou, honors)
2. Precompute partial replacement numbers for all possible suit combinations (~5^9 states)
3. Merge groups via DP: t^(n+1)_m = min over splits of meld counts
4. Result: t^(3)_4 = shanten for 4 melds + 1 pair

**Performance**: O(1) after precomputation; independent of hand size or shanten value.

Source: [Qiita (tomohxx)](https://qiita.com/tomohxx/items/75b5f771285e1334c0a5), [GitHub](https://github.com/tomohxx/shanten-number)

### Japanese Mahjong AI Development Blog (TadaoYamaoka)

An independent developer documenting their attempt to build a mahjong AI from scratch using PPO:

**Key Technical Points**:
- Uses **PPO** (vs Mortal's DQN) as the baseline algorithm
- **Reward variance reduction**: Value model uses "global information" (including opponent private tiles) to reduce noise from random initial hands
- **Zero-sum property**: Loss function designed so sum of 4 players' predicted values = 0
- Referenced **LuckyJ** (Tencent's unpublished AI) which uses search-based techniques for higher performance
- **Search excluded from baseline** due to implementation complexity

**Hydra Relevance**: Confirms PPO as viable alternative to DQN for mahjong; validates reward variance reduction with global info.

Source: [TadaoYamaoka's blog](https://tadaoyamaoka.hatenablog.com/entry/2023/10/03/233925)

### Mortal User Reviews (note.com, ai-bo.jp)

Japanese community consensus:
- Mortal rated as "excellent" (優秀) by regular NAGA users
- Primary value: Free + supports Mahjong Soul log import
- Primary frustration: No explanation of reasoning (users must infer intent)
- Comparison verdict: NAGA has higher analysis power but costs money

---

## 10. Mortal Architecture Deep Dive

> See [MORTAL_ANALYSIS.md](MORTAL_ANALYSIS.md) for the full architecture analysis including DQN head evolution (v1–v4), training loss components, distributed training, and 1v3 duplicate evaluation protocol.

---

## 11. Defense & Betaori Analysis

### Standard Defense Framework (riichi.wiki, community)

**Tile Safety Hierarchy** (from safest to least safe):
1. **Genbutsu**: 100% safe (already discarded by riichi declarer)
2. **Suji**: ~94% safe against riichi
3. **Kabe (wall)**: Safe when all 4 copies of connecting tiles are visible
4. **Honor tiles**: Variable safety based on game state
5. **Middle tiles (4-5-6)**: Most dangerous

### AI Defense Limitations
- **No damaten detection**: AIs can't reliably detect hidden tenpai (opponent waiting without riichi)
- **Multi-player defense**: Folding against one opponent may push dangerous tiles toward another
- **Score context**: When to push depends heavily on current scores/placement — Mortal uses fixed uma

### Push/Fold Decision Framework
Community consensus ("2 of 3" rule):
1. Am I in **tenpai**?
2. Do I have a **good wait** (5+ tiles)?
3. Is my hand **high value**?

If 2 of 3 → push. Otherwise → fold. Additional factors: round number, current scores, danger level of tiles to push.

---

## 12. Mahjong AI Landscape Summary

| AI | Level | Architecture | Open Source | Analysis | Key Trait |
|----|-------|-------------|-------------|----------|-----------|
| **Mortal** | ~7 dan | SE-ResNet + Dueling DQN | ✅ Yes | Free log review | Best open-source option |
| **NAGA** | ~9 dan (stable) | 4 CNNs, pure imitation learning | ❌ No | Paid, detailed | 5 playstyle variants trained on different players |
| **Suphx** | 8.74 dan (stable) | ResNet + Oracle guiding | ❌ No | Replay viewing only | First to reach 10 dan; GRP + oracle pioneering |
| **LuckyJ** | **10.68 dan (stable)** | ACH (RL+CFR) + OLSS (search) | ❌ No | None | Strongest known; game-theoretic RL + online search |
| **Kanachan** | Unknown (no benchmarks) | Transformer (BERT, ~90-310M params) | ✅ Yes (⚠️ no LICENSE file) | None | Zero hand-crafted features; impractical for online RL |
| **Akochan** | ~8 dan | EV-based heuristic (not ML) | ✅ Yes | Reviewer tool | Explicit suji/kabe/genbutsu defense logic |
| **Bakuuchi** | 9 dan | ISMCTS | ❌ No | None | Legacy, outperformed |

---

## 13. Key Takeaways for Hydra

> **Ownership note:** This section captures community-observed signals and hypotheses. Canonical Mortal limitation statements live in `MORTAL_ANALYSIS.md`; architecture-level Hydra-vs-Mortal deltas live in `HYDRA_SPEC.md`.

### Confirmed Gaps in Existing AIs (Opportunities for Hydra)

1. **Opponent Modeling**: No existing AI models opponent tendencies or detects damaten
2. **Score/Placement Awareness**: Mortal uses fixed uma; dynamic adjustment is an open problem  
3. **Multi-Turn Planning**: LuckyJ uses online search (OLSS, ICML 2023) and is the strongest AI — but the 4-player adaptation and search-as-feature integration are unpublished. No open-source AI uses search.
4. **Explainability**: All AIs are black boxes; interpretable decision factors would be novel
5. **Multi-Player Defense**: Simultaneous defense against 2+ threats is poorly handled
6. **Adaptive Playstyle**: NAGA offers multiple styles but doesn't adapt dynamically per-game

### Training Methodology Recommendations

1. **PPO over DQN**: TadaoYamaoka's work and community discussion suggest PPO is viable and may be preferable for policy-based mahjong AI
2. **Reward Variance Reduction**: Use global info in value model to distinguish skill from luck
3. **Opponent Pool**: Essential for preventing catastrophic forgetting and the "fearful agent" problem
4. **CQL for Offline**: Mortal's CQL integration prevents Q-value overestimation on unseen actions
5. **ELO Tracking**: Better progress metric than cumulative reward during self-play training
6. **1v3 Duplicate**: Gold standard evaluation method; eliminates variance

### Community Red Flags

- **100% AI accuracy = cheating indicator**: Mortal is used for real-time assistance (Akagi tool); this is a known anti-cheat concern
- **Playstyle subjectivity**: No single "correct" play in many mahjong situations; AI agreement is a weak proxy for quality
- **AI metrics are imperfect**: NAGA Rating, match%, and bad move rate don't reliably predict actual playing strength
