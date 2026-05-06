# Mahjong-Specific AI Techniques: Gaps Between Current Play and Optimal

Research report on 10 Riichi Mahjong AI domains.
Evidence: Mortal, Suphx, akochan, papers, community analysis.

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

Suji uses **furiten rule**: if opponent discarded tile X, cannot ron on tiles making ryanmen wait with X. Intervals: 1-4-7, 2-5-8, 3-6-9. Kabe ("wall") uses tile exhaustion: if all 4 copies visible, some adjacent waits impossible.

### Empirical Safety Data (from Houou-level games)

| Category | Deal-in Rate | Relative Danger |
|----------|-------------|-----------------|
| Genbutsu | 0% | Full safe |
| 3rd visible honor | ~0.3% | Near-full safe |
| Suji terminal (1/9) | ~1.9% | safe |
| Nakasuji 4/5/6 | ~2.4% | safe |
| Suji 2/8 | ~4.0% | Mid safe |
| Suji 3/7 | ~5.6% | Mid |
| Non-suji terminal | ~8.0% | Dangerous |
| Half suji 4/5/6 | ~8.1% | Dangerous |
| Non-suji 4/5/6 | ~13.9% | dangerous |

*Source: [riichi.wiki/Defense](https://riichi.wiki/Defense), [pathofhouou.blogspot.com kabe analysis](https://pathofhouou.blogspot.com/2020/07/guideanalysis-defense-techniques-kabe.html) (1.2M Houou games)*

### How Current AIs Handle It

**Mortal**: No explicit suji/kabe compute. NN learns safety implicitly from observation encoding. Obs includes:
- `kawa_overview` (per-player discard sets)
- `tiles_seen` (global visibility counts)
- `riichi_declared`/`riichi_accepted` flags
- Opponent discard sequences with recency-weighted encoding

Evidence: [Mortal obs_repr.rs L299-301](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L299-L301) -- `tiles_seen` encoded as count/4.

**Akochan**: Uses explicit `houjuu_hai_prob[38]` (per-tile deal-in probability) arrays computed elsewhere, then consumed by betaori module. Evidence: [akochan betaori.hpp L23-38](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.hpp#L23-L38).

**Suphx**: Learns implicitly. Reaches 10.06% deal-in vs 12.16% for Bakuuchi. Evidence: Suphx paper Table 5, Page 21.

### The Gap

**Current state**: Top AIs either learn safety implicitly (Mortal, Suphx) or use simplified probability tables (akochan). None compute true Bayesian posterior over opponent waits from full visible info.

**Theoretical optimal**: Perfect defense keeps probability distribution over each opponent's waiting tiles, updates in real time from each discard, call, riichi. Combines:
- Suji/kabe/one-chance/no-chance
- Tiles opponent chose NOT to discard (tedashi vs tsumogiri)
- Call patterns revealing hand shape
- Turn-by-turn Bayesian updates

**Gap severity**: MEDIUM-HIGH. Data shows suji alone cuts danger from ~14% to ~2-6%, but gap from "suji heuristic" to "true Bayesian" likely gives 1-3% deal-in improvement. At 10% base deal-in, that is 10-30% relative gain.

**Hydra opportunity**: Explicit safety head (Danger head outputting 3x34 danger estimates) can bootstrap from these empirical distributions, then refine via RL.

---

## 2. Damaten (Silent Tenpai) Detection

### What It Is

Damaten = reach tenpai without declaring riichi. Opponents get no explicit signal. Matters both offensively (when to damaten vs riichi) and defensively (detecting opponent damaten).

### How Current AIs Handle It

**Mortal**: Riichi is action index 37 in 46-action space. Model learns riichi vs no-riichi only through RL Q-values. No explicit opponent damaten detection.

Evidence: [Mortal obs_repr.rs L478-483](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L478-L483):
```rust
if cans.can_riichi {
    self.arr.fill(self.idx, 1.);
    if !self.at_kan_select {
        self.mask[37] = true;
    }
}
```

**Suphx**: Has separate "Riichi model" deciding whether to declare riichi. Trade-off learned. Opponent damaten detection stays implicit in observation encoding. Evidence: Suphx paper Table 1 (Page 4).

### The Gap

**Why damaten hard**: Unlike riichi, damaten gives no declaration. Only clues:
- Sudden discard-pattern shift (tedashi to tsumogiri)
- Repeated safe-tile discards showing hand-building no longer matters
- Call patterns completing yaku
- Turn count (late-game damaten more common)

**Current gap**: Major weakness of all current Mahjong AI. No public AI has explicit damaten probability estimator. Theoretical path:
1. Track opponent tenpai probability with dedicated model/head
2. Input: discard-pattern shifts, call timing, turn number, hand-composition constraints
3. Output: per-opponent tenpai probability, even without riichi

**Gap severity**: HIGH. Damaten causes roughly 30-40% of ron losses in high-level play. Even ~60% accurate damaten estimation would improve fold timing lot.

**Hydra opportunity**: Hydra Tenpai head (3 outputs, one per opponent) already targets this. Key move: train with perfect-information labels since training data shows exactly who was in tenpai. This is oracle guiding focused on tenpai detection.

---

## 3. Betaori (Defensive Retreat)

### The State of the Art: Akochan's Implementation

Akochan has most explicit betaori impl among open-source Mahjong AIs. Algorithm:

**Per-tile risk coefficient** ([akochan betaori.cpp](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.cpp)):
```
risk_coeff = houjuu_prob * (other_value - houjuu_value) / (houjuu_prob + beta - houjuu_prob * beta)
```
Where:
- `houjuu_prob` = deal-in probability for that tile
- `houjuu_value` = expected loss if you deal in with that tile
- `other_value` = expected value if no deal-in occurs
- `beta = 1 - 0.9^(tile_count_in_hand)`

Tiles sort by `risk_coeff` (lowest first = safest), then discard in that order.

**Fold EV calculation**: Accumulates total deal-in probability and expected loss across fold sequence, producing `betaori_exp` (expected points from fold line).

**Key limitation**: Betaori module does NOT decide WHEN to fold. It only computes fold-line EV. Attack-vs-fold decision happens elsewhere by comparing attack EV vs `betaori_exp`.

Evidence: [akochan betaori.hpp](https://github.com/critter-mj/akochan/blob/master/ai_src/betaori.hpp#L23-L38)

### How Mortal Handles Betaori

Mortal has no explicit betaori module. RL policy implicitly learns both when to fold and which defensive discard to choose. Strength: no handcrafted heuristic ceiling. Weakness: no guarantee of optimal defense.

### What "Optimal Betaori" Would Look Like

Optimal betaori would:
1. **Maintain per-opponent threat models**: probability each opponent is in tenpai, plus conditional distribution over waiting tiles
2. **Compute per-discard deal-in EV**: for each tile in hand, P(deal-in to opponent i) * E[loss | deal-in to i]
3. **Compare against attack EV**: fold when fold_EV > attack_EV, considering:
   - Multiple opponents at once (3-way danger)
   - Turn progression (remaining draws)
   - Exhaustive-draw probability and noten penalty
   - Ippatsu timing
4. **Mawashi (defensive shaping)**: find discards that are safe-ish AND preserve hand progress; hardest part since it is two-objective optimization

### Gap Severity: HIGH

Mortal sometimes pushes when fold is clearly right, and folds when push is EV-positive. No explicit push/fold framework means RL policy can be unstable in these critical spots. Akochan is more principled here but still uses simplified probabilities.

**Hydra opportunity**: Multi-head design (Danger head + Value head) gives needed pieces for explicit push/fold. At inference: danger head estimates per-opponent tile danger, value head estimates attack EV, explicit comparison picks mode. Hybrid = learned estimates + explicit decision logic.

---

## 4. Placement-Aware Play

### Why It Matters

Riichi Mahjong rewards PLACEMENT, not raw points. Uma/oka means:
- 1st place: +90 pts, 2nd: +45, 3rd: 0, 4th: -135 (typical)
- Avoiding 4th matters 2-3x more than climbing 3rd to 2nd
- South 3-4 need fundamentally different strategy than East rounds

### Suphx: The Gold Standard

Suphx (Microsoft, 2020) introduced **Global Reward Prediction** for this:

1. **GRU-based reward predictor**: 2-layer GRU + 2 FC layers takes round-level features (score delta, accumulated scores, dealer position, honba/kyotaku) and predicts final ranking.
2. **Per-round reward attribution**: Round k reward = Phi(x^k) - Phi(x^{k-1}), where Phi is predictor output.
3. **Key insight**: negative round score may not necessarily mean poor policy -- it may sometimes reflect certain tactics" (Suphx paper, Page 10). Huge leader should often play DEFENSIVELY to preserve 1st, even if one round loses points.

Evidence: Suphx paper Section 3.2, Pages 10-11, Figure 9 (Page 16) showing defensive play to protect 1st place lead.

### How Mortal Encodes Placement

Mortal encodes placement context in observation:
- **Scores**: each player's score normalized to [0, 100000], plus RBF encoding
- **Rank**: one-hot current rank (0-3)
- **Round**: kyoku (0-3) as one-hot, plus combined round indicator for v2+
- **Honba/kyotaku**: integer encoded with RBF

Evidence: [Mortal obs_repr.rs L149-194](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L149-L194)

### The Gap

**Current gap**: Mortal feeds placement info into NN but relies on RL to learn strategy. Problem: placement-critical spots (South 4 comeback, avoiding last) are rare in training, so calibration may be weak exactly where stakes peak.

**Theoretical optimal**: Placement-aware agent would:
1. Compute exact placement probabilities from current scores + remaining rounds
2. Optimize expected placement, not expected raw points
3. Switch strategy explicitly ("must-win round" vs "protect placement")

**Gap severity**: MEDIUM. Current AIs handle common placement spots OK, fail more on edge cases. Suphx global reward predictor is strong but not public. For Hydra, main lever is placement-weighted rewards instead of raw point rewards.

---

## 5. Yaku Selection & Hand Planning

### The Problem

Given starting hand + visible info, which yaku should you target? Examples:
- Tanyao (all simples) vs Pinfu (all sequences, no yakuhai pair)
- Go for Honitsu (half flush) vs balanced hand
- Riichi-only vs build value

### Current State

**No formal framework exists.** Top AIs (Mortal, Suphx) learn yaku selection implicitly through RL/SL. No published "optimal hand planning algorithm."

**Why hard**: Space of possible yaku combinations is large, and best target depends on:
- Current hand (starting tiles)
- Visible info (discards, calls, dora)
- Turn count (time left to build)
- Opponent threats
- Placement needs

**Closest approach**: Tjong (transformer-based Mahjong AI, 2024) uses "fan backward" -- yaku (fan) targets considered in reverse to guide discard decisions. Only published explicit yaku-aware planning attempt.

Evidence: [Tjong paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298) -- "decouples decision process into two distinct stages: action decision and tile decision."

### Gap Severity: LOW-MEDIUM

Neural nets learn decent yaku selection by pattern recognition. Main gap areas:
- Rare yaku (yakuman, tricky honitsu decisions)
- Multi-step planning (e.g., keep tiles now for later value)
- Yaku compatibility analysis (which combos remain achievable under constraints)

**Hydra opportunity**: GRP head (24-way Global Rank Prediction) indirectly pushes yaku awareness by predicting game outcomes, but explicit yaku selection still stays implicit.

---

## 6. Call Efficiency (Chi/Pon Decisions)

### The Strategic Axis

Calling (chi/pon) speeds hand up but:
- Opens hand (melds expose info)
- Removes riichi option (largest single-yaku point source)
- May kill yaku eligibility (menzen-only yaku like pinfu, ippatsu)
- Reveals hand direction (opponents adjust defense)

### How Current AIs Handle It

**Mortal**: Chi (actions 38-40), Pon (41), Kan (42), Pass (45) all sit in 46-action space. DQN learns call timing only from Q-values. No explicit call-efficiency heuristic.

Evidence: [Mortal obs_repr.rs L486-511](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L486-L511)

**Akochan**: Evaluates calls by computing expected value with and without call, balancing speed gain vs information-leak cost.

### The Gap

**Current gap**: RL call decisions tend to over-call in some spots (especially low-value hands where speed does not repay lost riichi value) and under-call in others (high-value open hands where speed is urgent).

**Theoretical optimal**: Framework should weigh:
1. Shanten reduction (how much speed gain?)
2. Value impact (lose riichi eligibility, kill yaku, expose dora)
3. Information leak (what opponents learn from this call?)
4. Defensive flexibility (open hand reduces safe-tile options)

**Gap severity**: MEDIUM. Call decisions are high-variance strategic choices, but RL handles common spots well. Gap lives in rare and subtle tail cases.

---

## 7. Riichi Timing

### The EV Landscape

Riichi adds about **1.5 han on average** (1 han base + ippatsu/ura chances), often doubling or tripling sub-mangan hands. Costs:
- 1000-point stick (recovered only on win)
- Hand locks
- Opponents know tenpai and may fold
- Can't change waits

### Decision Framework (from riichi.wiki analysis)

**Riichi favored when**:
- First to tenpai, good wait (6+ outs), hand <= 5200 before riichi
- Early game (before turn 12)
- Chasing riichi with decent hand

**Damaten favored when**:
- Bad wait + riichi-only hand
- Already haneman+ (riichi bonus marginal once past mangan ceiling)
- Late game while leading (protect placement)
- All-last where placement stays locked either way

Evidence: [riichi.wiki/Riichi_strategy](https://riichi.wiki/Riichi_strategy)

### Current AI Handling

Mortal: Riichi is action 37. Learned via RL. Usually good, but known to sometimes riichi bad waits or riichi in placement-losing spots.

### The Gap

**No optimal riichi solver exists.** Theoretical path:
1. Compute EV(riichi) = P(win|riichi) * E[points|riichi_win] - P(deal-in|riichi) * E[loss] - 1000*(1-P(win))
2. Compute EV(dama) = P(win|dama) * E[points|dama_win] + P(change_wait) * delta_EV
3. Riichi iff EV(riichi) > EV(dama), adjusted for placement

**Gap severity**: LOW-MEDIUM. Most riichi calls are easy (early-game decent wait usually means riichi). Gap is in marginal spots: 4-han hands, bad waits, late game, placement-sensitive cases. Those marginal spots matter more at high level.

---

## 8. Tile Efficiency (Shanten)

### The Solved Problem

**Shanten computation is exactly solved.** State of art: Nyanten algorithm by Cryolite, also implemented by tomohxx.

**Theory** (from [Cryolite's Nyanten writeup](https://qiita.com/Cryolite/items/75d504c7489426806b87)):
- Define replacement number r(h) = minimum self-draws to reach winning hand
- shanten = r(h) - 1
- Key optimization: decompose by suit (man/pin/sou/honors), compute per-suit partial replacement numbers independently, then minimize over valid meld/pair allocations
- Uses enumerative coding / minimal perfect hash for compact table indexing

**Mortal's impl**: Direct Rust port of tomohxx's C++ algorithm. Uses precomputed lookup tables (1.94M suit entries, 78K honor entries), compressed with gzip.

Evidence: [Mortal shanten.rs L1-4](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/algo/shanten.rs#L1-L4):
```rust
//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>
```

Table sizes: `JIHAI_TABLE_SIZE = 78,032`, `SUHAI_TABLE_SIZE = 1,940,777`.

### But Tile Efficiency != Optimal Play

**Shanten**: Solved. Exact minimum draws to tenpai.
**Ukeire (acceptance count)**: Solved. Exact count of tiles improving shanten.
**Weighted tile efficiency**: NOT solved. Must weight by:
- Probability of drawing each improving tile (depends on visible tiles)
- VALUE of resulting hands (1-han tenpai < mangan tenpai)
- Wait quality after tenpai (ryanmen >> kanchan >> tanki)
- Tenpai probability over multiple future turns, not only immediate improvement

### Mortal's Approach to Weighted Efficiency

Mortal v4 has **single-player expected value table** beyond raw shanten:

Evidence: [Mortal obs_repr.rs L564-623](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L564-L623) -- encodes per-discard tenpai probability curves, win probability curves, and EV curves over remaining turns.

`sp` (single-player) module computes:
- `tenpai_probs[turn]`: probability of reaching tenpai by each future turn
- `win_probs[turn]`: probability of winning by each future turn
- `exp_values[turn]`: expected point value of winning

Computed via dynamic programming with memoization across shanten states. This IS weighted tile efficiency, but in single-player setting.

### Gap Severity: LOW for pure shanten, MEDIUM for weighted efficiency

Pure shanten solved. Weighted efficiency handled well by Mortal SP module, but still assumes single-player optimal play with no opponent interaction. Gap is opponent-aware efficiency.

---

## 9. Opponent Hand Reading from Discard Patterns

### What's Known

Human experts read hands from:
- **Tedashi vs tsumogiri**: Hand-cut tiles reveal more than draw-and-discard
- **Discard order**: Early honor discards suggest speed hand. Keeping honors suggests yakuhai or honitsu
- **Missing tiles**: If suit never discarded, opponent may be collecting that suit (honitsu/chinitsu)
- **Call patterns**: Chi/pon exposes exact tiles, constraining hand configs
- **Riichi timing + riichi tile**: Riichi discard often is "last useless tile," narrowing waits

### Current AI Approaches

**Mortal**: Encodes opponent discards with:
- First 6 and last 18 discards per opponent
- Tedashi flag per discard
- Recency-weighted encoding: `v = exp(-0.2 * (max_kawa_len - 1 - turn))`
- Riichi tile tracking

Evidence: [Mortal obs_repr.rs L235-277](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L235-L277)

NN then learns hand reading implicitly from these features. No explicit hand-reading algorithm exists.

**Suphx**: Same pattern: implicit NN learning. Oracle guiding phase exposes opponent hands during training, helping model learn links between discard patterns and actual hands.

### Are There Better Explicit Algorithms?

**Short answer: No.** No published algorithm beats NN implicit learning for hand reading. Reasons:
1. Opponent-hand space is enormous (~10^48 information sets per IJCAI 2024)
2. Exact Bayesian inference is computationally intractable
3. Heuristic hand reading (suji counting, suit reading) captures only small part of signal

### The Gap

**Theoretical optimal**: Keep full probability distribution over each opponent's possible hand configs, updated by Bayesian inference after each action. Exact compute is intractable, but approximations could include:
- Particle filtering (sample possible hands, weight by observation consistency)
- Learned latent representations of opponent hand state
- Explicit belief tracking networks

**Gap severity**: MEDIUM. NNs capture common patterns well but may miss subtle rare signals. Biggest gap is MULTI-STEP reasoning: "opponent discarded X, then called Y, then discarded Z" chains needing sequential dependency tracking.

**Hydra opportunity**: Observation encoding already includes tedashi/tsumogiri split and recency weighting. Explicit attention over opponent discard sequences (transformer-style) could improve hand reading past pure CNN capture.

---

## 10. Disproportionate-Gain Mahjong-Specific Tricks

These techniques give outsized strength for small impl cost:

### Trick 1: Oracle Guiding (Suphx)

**What**: Train with perfect information (all hands + wall visible), then gradually drop oracle features.
**Why it's powerful**: Oracle agent learns WHAT GOOD PLAY LOOKS LIKE with full info, then transfers that to imperfect-information agent. This greatly speeds RL.
**Evidence**: Suphx paper Section 3.3 -- oracle features drop out via decay parameter gamma_t from 1 to 0 over training. "With help of oracle agent, our normal agent improves much faster than standard RL training."
**Hydra relevance**: DIRECT. Hydra's IVD (Invisible Value Decomposition) is related -- privileged info during training, unavailable at inference.

### Trick 2: Single-Player EV Tables (Mortal v4)

**What**: Precompute expected value curves (tenpai prob, win prob, point EV over turns) for each discard using dynamic programming in single-player model.
**Why it's powerful**: Gives NN "cheat sheet" of optimal single-player play. NN then mainly learns deviations from opponent interaction.
**Evidence**: [Mortal obs_repr.rs L564-611](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L564-L611)
**Hydra relevance**: HIGH. This is Hydra's FBS (Feature-Based Shaping).

### Trick 3: Auxiliary Prediction Heads

**What**: Add targets beyond main policy: tenpai probability, danger estimates, rank prediction.
**Why it's powerful**: Forces network to build representations that explicitly capture safety and hand-state info instead of hoping policy gradient discovers them.
**Evidence**: Hydra's 5-head design (Policy + Value + GRP + Tenpai + Danger) matches