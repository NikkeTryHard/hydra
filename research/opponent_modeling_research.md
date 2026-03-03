# Online Opponent Modeling for Search -- Research Survey

**Context**: Hydra's OLSS-II fixes opponents to a single blueprint policy during search.
This survey investigates whether modeling opponents more accurately during search
could improve search quality, potentially yielding +1 to +3 dan improvement.

---

## 1. OMIS -- Opponent Modeling with In-context Search (NeurIPS 2024)

**Paper**: [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/710445227fa8c1b6a9ceada902dd4741-Abstract-Conference.html)
| [OpenReview](https://openreview.net/forum?id=bGhsbfyg3b)
| [Code](https://github.com/renicechin/omis)

**What it does**: OMIS trains a Transformer with three in-context components:

1. **Actor** -- learns best responses to opponent policies via in-context learning
2. **Opponent Imitator** -- mimics opponent actions from observed interaction history
3. **Critic** -- estimates state values conditioned on opponent behavior

**How it works**:
- **Pretraining**: The Transformer sees diverse opponent policies during training,
  learning to *read* opponents from interaction context (like few-shot prompting).
- **Decision-Time Search**: At test time, given an unknown opponent, OMIS feeds
  recent interaction history into the Transformer. The opponent imitator predicts
  what the opponent will do; the actor + critic search for the best response
  against that predicted opponent.

**Key theoretical claims**:
- Without search: convergence in opponent policy recognition + generalization
- With search: improvement guarantees and performance stability

**Tested on**: Predator-Prey, Level-Based Foraging, Overcooked (cooperative/competitive/mixed)

**Relevance to Hydra**:
- OMIS is the closest thing to "plug opponent modeling into search" that exists.
- The three-component architecture (actor/imitator/critic) maps naturally to Hydra's
  search: the imitator replaces the fixed blueprint assumption for opponents.
- **Limitation**: Tested only on small cooperative/mixed games, not adversarial
  4-player games with the complexity of Mahjong. Scaling is unknown.
- **Opportunity**: The *concept* -- using interaction history to condition opponent
  predictions during search -- is directly applicable to OLSS-II.

---

## 2. Bayesian Opponent Modeling (Ganzfried et al., DAI 2024)

**Paper**: [DAI 2024](https://www.adai.ai/dai/2024/dai_papers/DAI2024_paper_10.pdf)
| [arXiv (earlier version)](https://arxiv.org/abs/2212.06027)

**Core approach**: Maintain a posterior distribution over opponent strategies and
update it as you observe their play across rounds.

**How it works**:

1. **Type representation**: Sample k candidate opponent strategies from Dirichlet priors
   at each information set. Each sample is an opponent "type."

2. **Belief update** (Bayes rule): For each joint sample index tuple s over opponents:
   ```
   q(s) = p(s) * z_s / sum_s'(p(s') * z_s')
   ```
   where p(s) = prior/previous posterior, z_s = likelihood of observed actions under
   sample s.

3. **Opponent model**: Posterior-weighted mixture of sampled strategies:
   ```
   m(a | q, i, j) = sum_s p(s) * tau_{i,j,s}(a | q)
   ```
   for opponent i, position j, information set q, action a.

4. **Best response**: Play best response to the posterior mixture opponent model.

5. **Warm-up phase**: First H rounds, play Nash (default) while collecting observations.
   After H rounds, switch to Bayesian best response.

**Partial observability**: The likelihood z_s naturally handles partial reveals (e.g.,
in poker, private cards only revealed at showdown). In Mahjong, this maps to: you
see discards and calls but not opponents' closed hands.

**Results in 3-player Kuhn poker**: MBBR ranked 1st with 48 millichips/hand,
beating all Nash agents (next best: 37). Prior quality matters enormously --
uninformed priors caused performance collapse to -54.

**Relevance to Hydra**:
- This is the most directly applicable approach. Mahjong has repeated games
  (hanchan = 4-8+ hands). Each hand reveals opponent tendencies.
- The "warm-up then exploit" pattern matches naturally: first 1-2 hands observe,
  then adjust search assumptions for remaining hands.
- Position-dependent modeling is built in (dealer vs non-dealer play styles).
- **Critical finding**: Prior quality matters hugely. For Hydra, the blueprint
  policy IS the prior -- start with "opponents play like our policy" and update.

---

## 3. Opponent Style Detection from Observed Play

**Key finding from mahjongAI project** ([GitHub](https://github.com/lxwooxy/mahjongAI)):

- Used generative model + particle-based importance sampling to infer opponent hand
  targets from observed discards.
- Found opponent modeling helps most against strategic/predictable opponents (67-94%
  win rate improvements), but high draw rates limit impact in symmetric matchups.

**Observable signals for style detection in Riichi Mahjong**:
- **Discard patterns**: Early honor discards = going for speed/tanyao.
  Holding honors late = defensive or going for yakuhai.
- **Riichi timing**: Early riichi = aggressive. Late/no riichi = defensive or
  going for damaten (hidden tenpai).
- **Call frequency**: Frequent pon/chi = aggressive hand-building, likely open hand.
  No calls = closed hand strategy (menzen), possibly going for riichi.
- **Defensive tells**: Cutting safe tiles after someone riichis = defensive.
  Pushing through = aggressive or desperate.

**Calibration approach for Hydra**:
- Hand 1: Play blueprint (Nash-like). Observe all opponents' discard/call patterns.
- Hand 2: Begin forming opponent type estimates. Adjust search weights slightly.
- Hand 3+: Full opponent model active. Search uses per-opponent predictions.
- Between hands: Update posterior over opponent types based on cumulative evidence.

**Adaptive Opponent Policy Detection** (Mridul & Khan, [arXiv 2406.06500](https://arxiv.org/abs/2406.06500)):
- OPS-DeMo approach outperforms PPO in dynamic scenarios with sudden policy shifts.
- Real-time strategy change detection -- relevant for detecting when an opponent
  shifts from aggressive to defensive mid-game (e.g., after taking a big hit).

---

## 4. How Pluribus Handles Multiple Opponent Types

**Paper**: [Science 2019](https://www.science.org/doi/10.1126/science.aay2400)
| [PDF Mirror](https://knowen-production.s3.amazonaws.com/uploads/attachment/file/3485/AI%2Bfor%2Bmultiplayer%2Bpoker.pdf)

**Critical answer**: Pluribus does NOT do per-opponent modeling.

**What Pluribus actually does**:

1. **Blueprint strategy**: Computed offline via Monte Carlo CFR self-play. All players
   use the SAME blueprint. No opponent differentiation.

2. **Real-time search**: During play, Pluribus searches from the current state.
   At depth limits, it models that EACH player may independently switch to one
   of k=4 continuation strategies:
   - The blueprint itself
   - Blueprint biased toward folding
   - Blueprint biased toward calling
   - Blueprint biased toward raising

3. **No adaptation**: The paper explicitly states Pluribus "plays a fixed strategy
   that does not adapt to the observed tendencies of the opponents."

**Why this matters for Hydra**:
- Pluribus proves you can be superhuman in 6-player poker WITHOUT opponent modeling.
  This is the conservative baseline -- and it works.
- BUT poker has much less per-opponent signal than Mahjong. In Mahjong, you see
  every discard, every call, and the full discard river. This is WAY more info
  per hand than poker gives you.
- The k=4 continuation strategies idea IS a form of type modeling -- just not
  adapted to specific opponents. Hydra could use a similar approach but make the
  continuation strategy weights opponent-specific.

**Opportunity**: Take Pluribus's k-strategy approach and make it adaptive:
- Instead of uniform weights over k strategies, use Bayesian weights updated
  from observed play.
- Opponent who has been calling light? Weight "biased to call" higher in search.
- Opponent who folded to every riichi? Weight "biased to fold/defend" higher.

---

## 5. Type-Based Opponent Modeling for Multiplayer Games

### 5a. Model-Based Opponent Modeling -- MBOM (NeurIPS 2022)

**Paper**: [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/b528459c99e929718a7d7e1697253d7f-Paper-Conference.pdf)

**Recursive Imagination**: MBOM learns an environment model, then:
1. Start with level-0 opponent policy from real data
2. Simulate interactions in the learned environment model
3. Compute opponent's imagined best response -> level-1 opponent model
4. Repeat for level-2, level-3, etc.
5. **Bayesian mixing**: Combine multiple levels with weights based on how
   well each matches recent real opponent actions.

This handles fixed opponents (level-0 sufficient), naive learners (level-1),
and reasoning opponents (higher levels) -- all in one framework.

### 5b. Opponent-Model Search in Games with Incomplete Information (AAAI 2024)

**Paper**: [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28844)
| [HAL PDF](https://hal.science/hal-04100646/document)

**The most formally rigorous type-based framework found.** Uses "vector games"
formulation and provides algorithms for multiple scenarios:

| Opponent Knowledge | Algorithm | Complexity |
|---|---|---|
| No model (pure maxmin) | MiniMax with set-valued domain | NP-hard |
| Single known OM | Best response via modified MiniMax | O(t*|T|) |
| Probabilistic OMs (weighted types) | Merged mixed OM | O(m*t*|T|) |
| Lexicographic OMs (priority ordering) | Lexicographic MiniMax | O(m*t*|T|) |
| Nondeterministic OMs (set of possible types) | Robust max-min over set | NP-hard (pure), poly (mixed) |
| Uncertain OM (with probability p_inf of wrong) | Interpolation: exploit vs robust | Poly (mixed) |

**Level-k recursive framework**: Defines base strategies Sigma_0, then recursively:
```
Sigma_k = BR(aggregate(Sigma_{k-1}, ..., Sigma_0))
```
A level-k player best-responds to opponents modeled at lower levels.

**Relevance to Hydra**: The "uncertain OM with p_inf" formulation is EXACTLY what
Hydra needs. It answers: "With probability (1-p_inf), opponent plays like our model.
With probability p_inf, opponent plays adversarially." The parameter p_inf is the
confidence dial -- high confidence in model = low p_inf = more exploitation.

---

## 6. Running Different Searches for Different Assumed Opponent Types

**Short answer**: Yes, this is well-supported by the literature. Multiple approaches:

### Approach A: Weighted ensemble (from OMS / Bayesian OM)
- Maintain k opponent type models with posterior weights
- Run search once but weight opponent actions by posterior mixture
- Equivalent to OMS "probabilistic OMs" -- O(k * t * |T|)
- **Most practical for Hydra**: One search pass, weighted opponent predictions

### Approach B: Parallel independent searches (from OMS "nondeterministic OMs")
- Run k separate searches, one per assumed opponent type
- Select the action that maximizes worst-case or expected payoff across types
- More robust but k times the compute
- **Use case**: When opponent types are very different (aggressive vs defensive)
  and you want the action that's decent against all of them

### Approach C: Robust interpolation (from OMS "uncertain OM with p_inf")
- Run one search that interpolates between "exploit the model" and "be safe"
- Parameter p_inf controls the exploitation/safety tradeoff
- As confidence in opponent model grows, reduce p_inf to exploit more
- **Best balance** for Hydra's compute budget

### Approach D: Pluribus-style continuation strategies
- Don't model individual opponents, but allow search to consider k different
  "behavioral modes" at depth limits
- Each opponent independently chooses from {aggressive, defensive, balanced, tricky}
- Simple to implement, already proven in Pluribus

**Recommendation for Hydra**: Start with Approach D (easiest, proven), then graduate
to Approach A (weighted mixture) as opponent modeling matures. Approach C is the
endgame -- confidence-based interpolation between exploitation and safety.

---

## 7. Safe Strategy vs Exploitative Strategy -- When to Exploit

Three key papers define the state of the art:

### 7a. Restricted Nash Response (RNR)
**Source**: [Lecture by Kuhnle](https://www.alankuhnle.com/teaching/f25-631/lecture-slides/m8-opponent-modeling/robust-exploit-v2.pdf)

**Core idea**: Instead of pure best-response to opponent model (risky), optimize:
```
max_{sigma_i} min_{sigma_{-i} in S_R} u_i(sigma_i, sigma_{-i})
```
where S_R is a "plausible set" around the opponent model (e.g., KL-ball of
radius epsilon).

**Tuning knobs**:
- lambda (blending): Higher = more exploitation, lower = safer/Nash-like
  - sigma_blend = lambda * hat_sigma + (1-lambda) * sigma_NE
- epsilon (RNR radius): Larger = more conservative
- Confidence-based: More data -> more exploitation

### 7b. Adaptation Safety (NeurIPS 2025)
**Paper**: [OpenReview](https://openreview.net/forum?id=JV84NVo1em)

**Key insight**: Safety should NOT mean "never be exploitable." In imperfect-info
games, all strategies are somewhat exploitable. Safety should mean: "your
exploitation attempt doesn't make you MORE exploitable than your non-exploiting
baseline."

This is huge for Mahjong -- pure Nash is impossible in 4-player Mahjong anyway.
So "adaptation safety" means: don't make yourself worse by trying to exploit.

### 7c. Safe Opponent Exploitation for Epsilon-Equilibrium (arXiv 2307.12338)
**Paper**: [arXiv](https://arxiv.org/abs/2307.12338)

**Prime-safe exploitation**: Redefines the "game value floor" as the worst-case
payoff your current strategy is vulnerable to. Even approximate equilibria can
be exploited safely with a practical guaranteed lower bound.

**Limitation**: Only proven for small two-player zero-sum games. Scaling to
Mahjong's 4-player setting is open research.

### When to exploit detected weaknesses (practical guidelines):

| Signal Strength | Confidence | Action |
|---|---|---|
| 1-2 hands observed | Low | Play blueprint (Nash-like) |
| 3-4 hands, clear pattern | Medium | Blend: 70% blueprint, 30% exploit |
| 5+ hands, consistent pattern | High | Blend: 40% blueprint, 60% exploit |
| Pattern breaks (opponent adapts) | Dropping | Revert toward blueprint |

**Mahjong-specific exploit examples**:
- Opponent never defends after riichi? Push more marginal hands.
- Opponent always folds to riichi? Riichi with worse waits (lower value, higher win%).
- Opponent calls too much (open hand addiction)? They'll have fewer safe tiles
  in endgame -- push harder when they're in tenpai.
- Opponent is too defensive? Bluff-riichi more (if applicable to scoring).

---

## Synthesis: Recommendations for Hydra's OLSS-II Evolution

### Phase 1: Low-Hanging Fruit (OLSS-II.1)
**Pluribus-style k-continuation strategies**

Instead of assuming all opponents play the blueprint, allow k=4 behavioral modes
at search depth limits:
- Blueprint (balanced)
- Blueprint biased toward defensive play (more fold/safe discard)
- Blueprint biased toward aggressive play (more riichi/push)
- Blueprint biased toward open-hand play (more calling)

Weight uniformly at first. This alone should improve search accuracy because
real opponents aren't all balanced.

**Estimated effort**: Small. Just modify the search to sample opponent actions
from k weighted policies instead of 1.

### Phase 2: Bayesian Opponent Typing (OLSS-II.2)
**Ganzfried-style posterior updates across hands**

- Prior: Blueprint policy (all opponents assumed balanced)
- After each hand: Update posterior weights on k opponent types based on
  observed discards, calls, riichi timing
- Feed updated weights into Phase 1's k-continuation strategies

Observable features per opponent per hand:
1. Discard speed (riichi timing percentile)
2. Call frequency (pon/chi rate)
3. Defense rate after riichi (fold % when someone declares riichi)
4. Tenpai rate (how often they reach tenpai)

**Estimated effort**: Medium. Need to define type features, implement Bayesian
updates, and wire updated weights into search. But the search itself doesn't change
much -- just the weights on existing k-modes.

### Phase 3: Confidence-Based Exploitation (OLSS-II.3)
**OMS-style uncertain model with p_inf**

- Track confidence in opponent model (function of # hands observed, consistency
  of observed patterns)
- Low confidence (early game): p_inf high, search is near-blueprint (safe)
- High confidence (late game, consistent opponent): p_inf low, search exploits
  detected weaknesses
- Use RNR-style blending: sigma = lambda * exploit_policy + (1-lambda) * blueprint
  where lambda = f(confidence)

**Safety guarantee**: Adaptation Safety principle -- never become more exploitable
than the non-exploiting baseline. If exploitation would make us worse, don't do it.

### Phase 4: Full OMIS-style In-Context Opponent Modeling (OLSS-III, future)
**Transformer-based opponent prediction from interaction history**

- Train a lightweight Transformer to predict opponent actions given the history
  of their discards/calls across the current game
- Use as the opponent imitator in search (replaces fixed k-modes)
- This is the most powerful but most expensive approach

**Estimated effort**: Large. Requires training a separate opponent model,
significant compute for data generation, and integration with the search pipeline.
Worth exploring only after Phases 1-3 are validated.

---

## Key Papers Reference Table

| Paper | Venue | Key Contribution | Relevance |
|---|---|---|---|
| OMIS | NeurIPS 2024 | In-context opponent modeling + decision-time search | Architecture pattern |
| Ganzfried et al. | DAI 2024 | Bayesian OM in multiplayer imperfect-info games | Direct application |
| MBOM | NeurIPS 2022 | Recursive imagination + Bayesian mixing of opponent levels | Type hierarchy |
| OMS (Li et al.) | AAAI 2024 | Formal algorithms for search with various opponent knowledge types | Theoretical foundation |
| Pluribus | Science 2019 | k-continuation strategies, no per-opponent modeling | Proven baseline |
| HORSE-CFR | ESWA 2025 | Hierarchical reasoning + safe exploitation CFR | Safe/exploit balance |
| Adaptation Safety | NeurIPS 2025 | Safety = don't become more exploitable than baseline | Safety principle |
| Safe OE (epsilon) | arXiv 2023 | Safe exploitation with approximate equilibria | Formal guarantees |
| OPS-DeMo | arXiv 2024 | Real-time opponent policy change detection | Drift detection |

---

## Bottom Line for Hydra

The research strongly supports that opponent modeling during search CAN improve
performance significantly, especially in information-rich games like Mahjong
where you observe extensive opponent behavior (discards, calls, timing).

**The phased approach (Phases 1-3) is the right path**:
1. Start with what Pluribus already proved works (k-strategies)
2. Add Bayesian adaptation (Ganzfried's approach)
3. Gate exploitation on confidence (OMS uncertain-model interpolation)

**The critical insight**: Don't go straight to full opponent modeling. The biggest
gains come from simply acknowledging that opponents aren't all identical. Even
k=4 unweighted behavioral modes (Pluribus-style) is a major improvement over
k=1 (current OLSS-II assumption).

**Risk**: Over-exploitation. If opponent model is wrong and you exploit hard,
you lose more than playing safe. The Adaptation Safety principle is the guard
rail -- never be worse than the non-exploiting baseline.
