# Mahjong-Specific AI Techniques: Gaps Between Current Play and Optimal

Research report covering 10 domains of Riichi Mahjong AI strategy.
Evidence sourced from Mortal, Suphx, akochan, academic papers, and community analysis.

---

## Table of Contents

1. [Suji/Kabe Defense](#1-sujikabe-defense)
2. [Damaten Detection](#2-damaten-silent-tenpai-detection)
3. [Betaori (Defensive Retreat)](#3-betaori-defensive-retreat)
4. [Placement-Aware Play](#4-placement-aware-play)
5. [Yaku Selection & Hand Planning](#5-yaku-selection--hand-planning)
6. [Call Efficiency](#6-call-efficiency-chipon-decisions)
7. [Riichi Timing](#7-riichi-timing)
8. [Tile Efficiency (Shanten)](#8-tile-efficiency-shanten)
9. [Opponent Hand Reading](#9-opponent-hand-reading-from-discard-patterns)
10. [Disproportionate-Gain Tricks](#10-disproportionate-gain-mahjong-specific-tricks)

---

## 1. Suji/Kabe Defense

### How It Works

Suji exploits the **furiten rule**: if an opponent discarded tile X, they cannot ron on tiles that would form a ryanmen wait with X. The intervals are 1-4-7, 2-5-8, 3-6-9. Kabe ("wall") uses tile exhaustion -- if all 4 copies of a tile are visible, certain adjacent waits become impossible.

### Empirical Safety Data (from Houou-level games)

| Category | Deal-in Rate | Relative Danger |
|----------|-------------|-----------------|
| Genbutsu | 0% | Absolute safe |
| 3rd visible honor | ~0.3% | Near-absolute |
| Suji terminal (1/9) | ~1.9% | Very safe |
| Nakasuji 4/5/6 | ~2.4% | Very safe |
| Suji 2/8 | ~4.0% | Medium safe |
| Suji 3/7 | ~5.6% | Medium |
| Non-suji terminal | ~8.0% | Dangerous |
| Half suji 4/5/6 | ~8.1% | Dangerous |
| Non-suji 4/5/6 | ~13.9% | Very dangerous |

*Source: [riichi.wiki/Defense](https://riichi.wiki/Defense), [pathofhouou.blogspot.com kabe analysis](https://pathofhouou.blogspot.com/2020/07/guideanalysis-defense-techniques-kabe.html) (1.2M Houou games)*

### How Current AIs Handle It

**Mortal**: NO explicit suji/kabe computation. The neural network learns safety patterns implicitly from the observation encoding. The obs includes:
- `kawa_overview` (per-player discard sets)
- `tiles_seen` (global visibility counts)
- `riichi_declared`/`riichi_accepted` flags
- Opponent discard sequences with recency-weighted encoding

Evidence: [Mortal obs_repr.rs L299-301](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L299-L301) -- tiles_seen encoded as count/4.

**Akochan**: Uses explicit `houjuu_hai_prob[38]` (per-tile deal-in probability) arrays computed elsewhere, then consumed by the betaori module. Evidence: [akochan betaori.hpp L23-38](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.hpp#L23-L38).

**Suphx**: Learns implicitly. Achieves 10.06% deal-in rate vs 12.16% for Bakuuchi. Evidence: Suphx paper Table 5, Page 21.

### The Gap

**Current state**: All top AIs either learn safety implicitly (Mortal, Suphx) or use simplified probability tables (akochan). None compute a true Bayesian posterior over opponent waits given all observable information.

**Theoretical optimal**: A perfect defense algorithm would maintain a probability distribution over each opponent's possible waiting tiles, updated in real-time as each discard, call, and riichi declaration is observed. This would account for:
- Suji/kabe/one-chance/no-chance simultaneously
- Which tiles the opponent chose NOT to discard (tedashi vs tsumogiri distinction)
- Call patterns revealing hand structure
- Turn-by-turn Bayesian updates

**Gap severity**: MEDIUM-HIGH. The empirical data shows suji alone drops danger from ~14% to ~2-6%, but the gap between "suji-level heuristic" and "true Bayesian" is estimated at 1-3% deal-in rate improvement. For a 10% base deal-in rate, that's a 10-30% relative improvement.

**Hydra opportunity**: An explicit safety head (Danger head outputting 3x34 danger estimates) could bootstrap from these empirical distributions and be refined via RL, getting the best of both worlds.


---

## 2. Damaten (Silent Tenpai) Detection

### What It Is

Damaten = reaching tenpai but NOT declaring riichi. The player stays "hidden" -- opponents don't know they're in tenpai. This is both an offensive choice (when should I damaten vs riichi?) and a defensive challenge (detecting when an opponent is in damaten).

### How Current AIs Handle It

**Mortal**: Riichi is action index 37 in the 46-action space. The model learns when to riichi vs not-riichi purely through RL Q-values. There is NO explicit damaten detection for opponents.

Evidence: [Mortal obs_repr.rs L478-483](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L478-L483):
```rust
if cans.can_riichi {
    self.arr.fill(self.idx, 1.);
    if !self.at_kan_select {
        self.mask[37] = true;
    }
}
```

**Suphx**: Has a separate "Riichi model" that decides whether to declare riichi. The decision is a learned trade-off. Damaten detection for opponents is implicit in the observation encoding. Evidence: Suphx paper Table 1 (Page 4).

### The Gap

**Why damaten is hard to detect**: Unlike riichi (which is explicitly declared), damaten opponents provide no signal. The only clues are:
- Sudden shift in discard patterns (tedashi to tsumogiri)
- Repeated safe-tile discards suggesting they no longer care about hand building
- Call patterns that complete a hand with yaku
- Turn count (late game damaten is more common)

**Current gap**: This is widely cited as a MAJOR weakness of all current Mahjong AI. No public AI has an explicit damaten probability estimator. The theoretical approach would be:
1. Track opponent tenpai probability via a dedicated model/head
2. Input: discard pattern changes, call timing, turn number, hand composition constraints
3. Output: per-opponent tenpai probability (even without riichi)

**Gap severity**: HIGH. Damaten accounts for roughly 30-40% of ron losses in high-level play. Being able to estimate opponent damaten probability with even 60% accuracy would significantly improve fold timing.

**Hydra opportunity**: The Tenpai head (3 outputs, one per opponent) in Hydra's design already targets this. The key insight is that this head should be trained with supervision from perfect-information labels (we KNOW who was in tenpai in training data) -- this is essentially a form of oracle guiding applied specifically to tenpai detection.


---

## 3. Betaori (Defensive Retreat)

### The State of the Art: Akochan's Implementation

Akochan has the most explicit betaori implementation in any open-source Mahjong AI. The algorithm:

**Per-tile risk coefficient** ([akochan betaori.cpp](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.cpp)):
```
risk_coeff = houjuu_prob * (other_value - houjuu_value) / (houjuu_prob + beta - houjuu_prob * beta)
```
Where:
- `houjuu_prob` = deal-in probability for that tile
- `houjuu_value` = expected loss if you deal in with that tile
- `other_value` = expected value if no deal-in occurs
- `beta = 1 - 0.9^(tile_count_in_hand)`

Tiles are sorted by risk_coeff (lowest first = safest), then discarded in order.

**Fold EV calculation**: Accumulates total deal-in probability and expected loss across the folding sequence, producing `betaori_exp` (expected points from folding line).

**Key limitation**: The betaori module does NOT decide WHEN to fold -- it only computes the fold-line EV. The attack-vs-fold decision must be made elsewhere by comparing attack EV vs betaori_exp.

Evidence: [akochan betaori.hpp](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.hpp#L23-L38)

### How Mortal Handles Betaori

Mortal has NO explicit betaori module. The RL-trained policy implicitly learns when to fold and what to discard defensively. This is both a strength (no handcrafted heuristics to be wrong) and a weakness (no guarantee of optimal defensive play).

### What "Optimal Betaori" Would Look Like

An optimal betaori algorithm would:

1. **Maintain per-opponent threat models**: Probability of each opponent being in tenpai, and conditional distribution over their waiting tiles
2. **Compute per-discard deal-in EV**: For each tile in hand, P(deal-in to opponent i) * E[loss | deal-in to i]
3. **Compare against attack EV**: Fold when fold_EV > attack_EV, considering:
   - Multiple opponents simultaneously (3-way danger)
   - Turn progression (remaining draw count)
   - Exhaustive-draw probability and noten penalty
   - Ippatsu timing
4. **Mawashi (defensive shaping)**: Find discards that are safe-ish AND maintain hand progress -- this is the hardest part, as it requires optimizing two objectives

### Gap Severity: HIGH

Mortal is known to occasionally push in spots where folding is clearly correct, and to fold in spots where pushing is EV-positive. The lack of explicit push/fold framework means the RL policy can be unstable in these critical decision points. Akochan's explicit framework is more principled but uses simplified probability estimates.

**Hydra opportunity**: The multi-head design (Danger head + Value head) provides the ingredients for explicit push/fold computation. During inference, the danger head estimates per-opponent tile danger, the value head estimates attack EV, and an explicit comparison determines the action mode. This hybrid approach (learned estimates + explicit decision logic) could bridge the gap.


---

## 4. Placement-Aware Play

### Why It Matters

Riichi Mahjong scores by PLACEMENT (1st through 4th), not total points. The uma/oka system means:
- 1st place: +90 pts, 2nd: +45, 3rd: 0, 4th: -135 (typical)
- Avoiding 4th is 2-3x more important than climbing from 3rd to 2nd
- South 3-4 require fundamentally different strategy than East rounds

### Suphx: The Gold Standard

Suphx (Microsoft, 2020) introduced **Global Reward Prediction** specifically for this:

1. **GRU-based reward predictor**: A 2-layer GRU + 2 FC layers that takes round-level features (score delta, accumulated scores, dealer position, honba/kyotaku) and predicts final game ranking.
2. **Per-round reward attribution**: Round k reward = Phi(x^k) - Phi(x^{k-1}), where Phi is the predictor output.
3. **Key insight**: "A negative round score may not necessarily mean a poor policy -- it may sometimes reflect certain tactics" (Suphx paper, Page 10). A player with a massive lead should play DEFENSIVELY to protect 1st, even if that means losing points in a single round.

Evidence: Suphx paper Section 3.2, Pages 10-11, Figure 9 (Page 16) showing defensive play to protect 1st place lead.

### How Mortal Encodes Placement

Mortal encodes placement context in its observation:
- **Scores**: Each player's score normalized to [0, 100000] range, plus RBF encoding
- **Rank**: One-hot encoding of current rank (0-3)
- **Round**: kyoku (0-3) as one-hot, plus combined round indicator for v2+
- **Honba/kyotaku**: Integer encoded with RBF

Evidence: [Mortal obs_repr.rs L149-194](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L149-L194)

### The Gap

**Current gap**: Mortal provides placement info to the NN but relies on RL to learn the right strategy. The problem is that placement-critical situations (South 4 comebacks, avoiding last) are RARE in training data, so the policy may not be well-calibrated for these critical moments.

**Theoretical optimal**: A placement-aware agent would:
1. Compute exact placement probabilities given current scores + remaining rounds
2. Adjust the value function to optimize expected placement rather than expected points
3. Make explicit strategy switches (e.g., "this is a must-win round" vs "protect placement")

**Gap severity**: MEDIUM. Current AIs handle common placement situations reasonably well but fail at edge cases. The Suphx global reward predictor is a good approach but not publicly available. For Hydra, training with placement-weighted rewards (rather than raw point rewards) is the key lever.


---

## 5. Yaku Selection & Hand Planning

### The Problem

Given a starting hand + visible information, which yaku (winning pattern) should you aim for? Examples:
- Tanyao (all simples) vs Pinfu (all sequences, no yakuhai pair)
- Going for Honitsu (half flush) vs balanced hand
- Riichi-only vs building value

### Current State

**No formal framework exists.** All current top AIs (Mortal, Suphx) learn yaku selection implicitly through RL/SL. There's no published "optimal hand planning algorithm."

**Why it's hard**: The space of possible yaku combinations is large, and the optimal target depends on:
- Current hand (starting tiles)
- Visible information (discards, calls, dora)
- Turn count (how much time to build)
- Opponent threats
- Placement needs

**Closest approach**: Tjong (transformer-based Mahjong AI, 2024) uses "fan backward" -- a technique where yaku (fan) targets are considered in reverse to guide discard decisions. This is the only published attempt at explicit yaku-aware planning.

Evidence: [Tjong paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298) -- "decouples the decision process into two distinct stages: action decision and tile decision."

### Gap Severity: LOW-MEDIUM

Neural networks learn reasonable yaku selection through pattern recognition. The gap is mainly in:
- Rare yaku (yakuman, complex honitsu decisions)
- Multi-step planning (e.g., deliberately keeping tiles for future value)
- Yaku compatibility analysis (which yaku combos are achievable given constraints)

**Hydra opportunity**: The GRP head (24-way Global Rank Prediction) indirectly encourages yaku awareness by predicting game-level outcomes, but explicit yaku selection remains learned implicitly.

---

## 6. Call Efficiency (Chi/Pon Decisions)

### The Strategic Axis

Calling (chi/pon) speeds up your hand but:
- Opens your hand (visible melds give opponents information)
- Eliminates riichi option (biggest single-yaku point source)
- May destroy yaku eligibility (menzen-only yaku like pinfu, ippatsu)
- Reveals hand direction (opponents adjust defense)

### How Current AIs Handle It

**Mortal**: Chi (actions 38-40), Pon (41), Kan (42), Pass (45) are all in the 46-action space. The DQN learns when to call purely from Q-values. No explicit call-efficiency heuristic.

Evidence: [Mortal obs_repr.rs L486-511](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L486-L511)

**Akochan**: Evaluates calls by computing expected value with and without the call, comparing attack speed gain vs information leak cost.

### The Gap

**Current gap**: RL-based call decisions tend to over-call in some situations (especially low-value hands where closing quickly doesn't compensate for lost riichi value) and under-call in others (high-value open hands where speed is critical).

**Theoretical optimal**: The decision framework should weigh:
1. Shanten reduction (how much does this call speed up the hand?)
2. Value impact (riichi eligibility loss, yaku destruction, dora exposure)
3. Information leak (what does this call tell opponents about my hand?)
4. Defensive flexibility (open hand reduces safe tile options)

**Gap severity**: MEDIUM. Call decisions are one of the highest-variance strategic choices, but current RL approaches handle common cases well. The tail of rare/subtle call decisions is where the gap lives.


---

## 7. Riichi Timing

### The EV Landscape

Riichi adds approximately **1.5 han on average** (1 han base + ippatsu/ura chances), often doubling or tripling hand value below mangan. But it costs:
- 1000-point stick (recovered only on win)
- Hand is locked (no flexibility)
- Opponents know you're in tenpai (may fold)
- Can't change waits

### Decision Framework (from riichi.wiki analysis)

**Riichi favored when**:
- First to tenpai, good wait (6+ outs), hand <= 5200 before riichi
- Early game (before turn 12)
- Chasing riichi with decent hand

**Damaten favored when**:
- Bad wait + riichi-only hand
- Already haneman+ (marginal value of riichi bonus is low past mangan ceiling)
- Late game leading (protect placement)
- All-last where placement is locked regardless of riichi

Evidence: [riichi.wiki/Riichi_strategy](https://riichi.wiki/Riichi_strategy)

### Current AI Handling

Mortal: Riichi is action 37. Learned via RL. Generally good but known to occasionally riichi with bad waits or in placement-losing situations.

### The Gap

**No optimal riichi solver exists.** The theoretical approach would be:
1. Compute EV(riichi) = P(win|riichi) * E[points|riichi_win] - P(deal-in|riichi) * E[loss] - 1000*(1-P(win))
2. Compute EV(dama) = P(win|dama) * E[points|dama_win] + P(change_wait) * delta_EV
3. Riichi iff EV(riichi) > EV(dama), adjusted for placement effects

**Gap severity**: LOW-MEDIUM. Most riichi decisions are straightforward (riichi is almost always correct in early-game with decent waits). The gap is in marginal cases: 4-han hands, bad waits, late game, and placement-sensitive situations. These marginal cases matter more at high-level play.

---

## 8. Tile Efficiency (Shanten)

### The Solved Problem

**Shanten computation is exactly solved.** The state of the art is the Nyanten algorithm by Cryolite, also implemented by tomohxx.

**Theory** (from [Cryolite's Nyanten writeup](https://qiita.com/Cryolite/items/75d504c7489426806b87)):
- Define replacement number r(h) = minimum self-draws to reach a winning hand
- shanten = r(h) - 1
- Key optimization: decompose by suit (man/pin/sou/honors), compute per-suit partial replacement numbers independently, then minimize over all valid meld/pair allocations
- Uses enumerative coding / minimal perfect hash for compact table indexing

**Mortal's implementation**: Direct port of tomohxx's C++ algorithm into Rust. Uses precomputed lookup tables (1.94M entries for suits, 78K for honors), compressed with gzip.

Evidence: [Mortal shanten.rs L1-4](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/algo/shanten.rs#L1-L4):
```rust
//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>
```

Table sizes: `JIHAI_TABLE_SIZE = 78,032`, `SUHAI_TABLE_SIZE = 1,940,777`.

### But Tile Efficiency != Optimal Play

**Shanten**: Solved. Exact minimum draws to tenpai.
**Ukeire (acceptance count)**: Solved. Exact count of tiles that improve shanten.
**Weighted tile efficiency**: NOT solved. This requires weighting by:
- Probability of drawing each improving tile (depends on visible tiles)
- VALUE of resulting hands (a 1-han tenpai is worth less than a mangan tenpai)
- Wait quality after tenpai (ryanmen >> kanchan >> tanki)
- Tenpai probability over multiple turns (not just immediate improvement)

### Mortal's Approach to Weighted Efficiency

Mortal v4 has a **single-player expected value table** that goes beyond raw shanten:

Evidence: [Mortal obs_repr.rs L564-623](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L564-L623) -- encodes per-discard tenpai probability curves, win probability curves, and EV curves over remaining turns.

The `sp` (single-player) module computes:
- `tenpai_probs[turn]`: probability of reaching tenpai by each future turn
- `win_probs[turn]`: probability of winning by each future turn
- `exp_values[turn]`: expected point value of winning

This is computed via dynamic programming with memoization across shanten states. This IS a form of weighted tile efficiency, and it's one of the most sophisticated such implementations.

### Gap Severity: LOW for pure shanten, MEDIUM for weighted efficiency

Pure shanten is solved. Weighted efficiency is well-handled by Mortal's SP module but still assumes single-player optimal play (no opponent interaction). The gap is in incorporating opponent models into efficiency calculations.


---

## 9. Opponent Hand Reading from Discard Patterns

### What's Known

Human experts read opponents' hands via:
- **Tedashi vs tsumogiri**: Hand-cut tiles (tedashi) reveal more about hand structure than draw-and-discard (tsumogiri)
- **Discard order**: Early honor discards suggest a speed-oriented hand. Keeping honors suggests yakuhai or honitsu direction.
- **Missing tiles**: If an opponent never discards tiles in a suit, they may be collecting that suit (honitsu/chinitsu)
- **Call patterns**: Chi/pon reveals exact tiles, constraining possible hand configurations
- **Riichi timing + riichi tile**: The tile discarded with riichi declaration is often the "last useless tile," which narrows possible waits

### Current AI Approaches

**Mortal**: Encodes opponent discards with:
- First 6 and last 18 discards per opponent
- Tedashi flag per discard
- Recency-weighted encoding: `v = exp(-0.2 * (max_kawa_len - 1 - turn))`
- Riichi tile tracking

Evidence: [Mortal obs_repr.rs L235-277](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L235-L277)

The NN then implicitly learns to read hands from these features. NO explicit hand-reading algorithm exists.

**Suphx**: Similarly relies on NN implicit learning. The oracle guiding phase gives the model access to opponent hands during training, which helps it learn correlations between discard patterns and actual hands.

### Are There Better Explicit Algorithms?

**Short answer: No.** No published algorithm outperforms neural network implicit learning for hand reading. The reasons:
1. The space of possible opponent hands is enormous (~10^48 information sets per IJCAI 2024)
2. Exact Bayesian inference is computationally intractable
3. Heuristic hand reading (suji counting, suit analysis) captures only a small fraction of the available signal

### The Gap

**Theoretical optimal**: Maintain a full probability distribution over each opponent's possible hand configurations, updated via Bayesian inference after each action. This is computationally intractable for exact computation but could be approximated via:
- Particle filtering (sample possible hands, weight by consistency with observations)
- Learned latent representations of opponent hand state
- Explicit belief tracking networks

**Gap severity**: MEDIUM. Neural networks capture the MOST COMMON patterns well but may miss subtle signals in rare configurations. The biggest gap is in MULTI-STEP reasoning: "opponent discarded X, then called Y, then discarded Z" chains that require tracking sequential dependencies.

**Hydra opportunity**: The observation encoding already includes tedashi/tsumogiri distinction and recency weighting. Adding explicit attention over opponent discard sequences (transformer-style) could improve hand reading beyond what pure CNN architectures capture.

---

## 10. Disproportionate-Gain Mahjong-Specific Tricks

These are techniques where small implementation effort yields outsized performance gains:

### Trick 1: Oracle Guiding (Suphx)

**What**: Train with perfect information (see all hands + wall), then gradually drop out oracle features.
**Why it's powerful**: The oracle agent learns WHAT GOOD PLAY LOOKS LIKE with full information, then transfers that knowledge to the imperfect-information agent. This drastically speeds up RL training.
**Evidence**: Suphx paper Section 3.3 -- the oracle features are dropped via a decay parameter gamma_t that goes from 1 to 0 over training. "With the help of the oracle agent, our normal agent improves much faster than standard RL training."
**Hydra relevance**: DIRECT. Hydra's IVD (Invisible Value Decomposition) is related -- using privileged information during training that isn't available at inference.

### Trick 2: Single-Player EV Tables (Mortal v4)

**What**: Precompute expected value curves (tenpai prob, win prob, point EV over turns) for each possible discard, using dynamic programming in a single-player model.
**Why it's powerful**: Gives the NN a "cheat sheet" of optimal single-player play to start from. The NN then only needs to learn deviations caused by opponent interaction.
**Evidence**: [Mortal obs_repr.rs L564-611](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L564-L611)
**Hydra relevance**: HIGH. This is essentially what Hydra's FBS (Feature-Based Shaping) aims to provide.

### Trick 3: Auxiliary Prediction Heads

**What**: Add prediction targets beyond the main policy: tenpai probability, danger estimates, rank prediction.
**Why it's powerful**: Forces the network to build representations that explicitly capture safety and hand-state information, rather than hoping the policy gradient discovers them.
**Evidence**: Hydra's 5-head design (Policy + Value + GRP + Tenpai + Danger) is exactly this pattern. The Tenpai head is trained with supervision from perfect-info labels.
**Hydra relevance**: CORE ARCHITECTURE.

### Trick 4: Explicit Safety Channel Encoding

**What**: Instead of raw tile observations, pre-compute safety features (genbutsu masks, suji relationships, visible-tile counts) and include them as dedicated input channels.
**Why it's powerful**: Reduces the learning burden on the NN. Instead of learning "if opponent discarded 1m and I have 4m, 4m is suji-safe" from scratch, the input directly encodes "4m is suji-safe vs opponent 1."
**Evidence**: Hydra's 23 safety channels (62 base + 23 safety = 85 total) in the encoding spec.
**Hydra relevance**: CORE DESIGN. This is the single highest-leverage Mahjong-specific optimization.

### Trick 5: Agari Guard (Rule-Based Override)

**What**: Hard-code that the AI always wins when it can (never passes on tsumo/ron).
**Why it's powerful**: Even strong RL policies occasionally learn to pass on winning hands in certain situations. A simple rule override prevents this catastrophic error.
**Evidence**: [DeepWiki on Mortal](https://deepwiki.com/Equim-chan/Mortal/3-mortal-ai-system) -- "A rule-based agari guard can override suboptimal winning decisions."
**Hydra relevance**: Trivial to implement, prevents rare but devastating mistakes.

### Trick 6: Global Reward Prediction (Suphx)

**What**: Use an RNN to predict final game placement from round-by-round features, then use the per-round delta as the RL reward signal.
**Why it's powerful**: Standard RL uses round-level point deltas as rewards, which can be misleading (losing points to protect 1st place is actually GOOD strategy). Global reward prediction correctly attributes game-level success to individual round decisions.
**Evidence**: Suphx paper Section 3.2 -- GRU-based predictor with round-level features.
**Hydra relevance**: Phase 3 of Hydra's training pipeline should incorporate this.


---

## Summary: Gap Severity Rankings

| Topic | Gap Severity | Current Best | Theoretical Optimal | Hydra Leverage |
|-------|-------------|-------------|--------------------|--------------------|
| 1. Suji/Kabe Defense | MEDIUM-HIGH | Implicit NN learning | Bayesian posterior over waits | Danger head + safety channels |
| 2. Damaten Detection | HIGH | No explicit detection | Tenpai probability estimator | Tenpai head with oracle labels |
| 3. Betaori | HIGH | Akochan explicit, Mortal implicit | Multi-opponent threat + attack EV comparison | Danger head + Value head comparison |
| 4. Placement-Aware | MEDIUM | Suphx GRP, Mortal score encoding | Exact placement probability optimization | GRP head + placement-weighted rewards |
| 5. Yaku Selection | LOW-MEDIUM | Implicit NN learning | No formal framework exists | Indirect via GRP/Value heads |
| 6. Call Efficiency | MEDIUM | RL Q-values | Information-theoretic call framework | Standard RL + better encoding |
| 7. Riichi Timing | LOW-MEDIUM | RL + heuristic rules | Full EV comparison with placement | Policy action + Value head |
| 8. Tile Efficiency | LOW (shanten solved) | Mortal SP tables | Single-player is solved; multi-player gap | FBS (Feature-Based Shaping) |
| 9. Hand Reading | MEDIUM | Implicit NN + recency encoding | Bayesian belief tracking | Attention over discard sequences |
| 10. Special Tricks | HIGH ROI | Scattered across AIs | Combined approach | All 6 tricks applicable |

## Key Takeaway for Hydra

The biggest performance gaps in current Mahjong AI are in **defense** (topics 1-3) and **placement awareness** (topic 4). Hydra's multi-head architecture (Policy + Value + GRP + Tenpai + Danger) directly targets the top 4 gaps. The 23 safety channels in the encoder provide the safety-specific input that no other AI has in explicit form.

The disproportionate-gain tricks (Section 10) are all applicable to Hydra and should be implemented in order:
1. **Safety channel encoding** (already designed) -- highest immediate ROI
2. **Auxiliary prediction heads** (already designed) -- forces good representations
3. **Oracle guiding / IVD** (designed as IVD) -- dramatically speeds up RL
4. **SP EV tables / FBS** (designed as FBS) -- provides single-player optimal baseline
5. **Global reward prediction** (Phase 3) -- correct reward attribution
6. **Agari guard** (trivial) -- prevents rare catastrophic errors

---

## Sources

- Mortal source: https://github.com/Equim-chan/Mortal (commit 0cff2b52)
- Suphx paper: https://arxiv.org/abs/2003.13590
- Akochan source: https://github.com/critter-mj/akochan
- Tjong paper: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298
- Nyanten theory: https://qiita.com/Cryolite/items/75d504c7489426806b87
- Shanten calculator: https://github.com/tomohxx/shanten-number
- Defense data: https://riichi.wiki/Defense, https://riichi.wiki/Suji
- Kabe analysis: https://pathofhouou.blogspot.com/2020/07/guideanalysis-defense-techniques-kabe.html
- Riichi strategy: https://riichi.wiki/Riichi_strategy
- Mahjong EV engine: https://github.com/CharlesC63/mahjong_ev
- IJCAI Mahjong competition: https://www.ijcai.org/proceedings/2024/1020.pdf
