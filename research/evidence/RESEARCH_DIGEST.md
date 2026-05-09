# Research Digest: Evidence Core

Purpose: route agents to current research contracts. This file preserves compact conclusions from deleted evidence notes:

- `theoretical_frameworks.md` -- merged here; deleted.
- `ivd_evidence_report.md` -- merged here; deleted.
- `population-exploitation-survey.md` -- merged here; deleted.
- `BPR_FEASIBILITY.md` -- merged here; deleted.
- `EXPLOITATION_VS_NASH.md` -- merged here; deleted.

- `TRAINING_TECHNIQUES_SURVEY.md` -- merged here; deleted.
Status key: **PROMOTE** = use in current Hydra design. **KEEP-CONTEXT** = useful model/analogy, not primary route. **DEMOTE** = historical/speculative; do not build from it without fresh proof.

## Promoted contracts

### 1. Value decomposition / IVD

**Status: PROMOTE as design lens, not standalone algorithm.**

Core claim: Mahjong action value is not only immediate EV. Useful split:

```text
Q(action) ~= instrumental value + epistemic value + strategic/signaling value
```

Evidence:

- Factorised Active Inference for strategic multi-agent games (`arXiv:2411.07362`) directly decomposes expected free energy:
  - `rho`: pragmatic / instrumental payoff.
  - `varsigma`: salience, info about state.
  - `eta`: novelty, info about model params.
  - Tested on iterated Chicken, Stag Hunt, Prisoner's Dilemma; supports multi-agent decomposition.
- BAD / Hanabi (`arXiv:1811.01458`, ICML 2019): actions carry task value plus information value; public belief update from actions made conventions work.
- ERSAC (ICML 2023): converts epistemic uncertainty into value; min-max form gives competitive framing.
- Strategic ULCB / Strategic Nash-Q (UAI 2021): strategic exploration bonuses beat optimistic exploration in zero-sum games; robust to irrelevant states.
- LOLA (`arXiv:1709.04326`, AAMAS 2018): opponent-learning-awareness term is precedent for strategic component: action value includes impact on other learner.
- IDS (Russo & Van Roy): information ratio `regret^2 / information_gain`; independent support for reward/info tradeoff.
- VIME: information gain as intrinsic reward; single-agent but mechanism matches epistemic term.
- Suphx: oracle guiding and runtime adaptation imply hidden-information value matters in Mahjong, though not explicit IVD.

Hydra use:

- Keep multi-head/value design honest: value head must not collapse uncertainty, danger, and opponent belief into one opaque scalar if caller needs decomposition.
- Treat epistemic outputs as decision support for search/belief update, not as decorative auxiliary loss.
- Strategic/signaling value matters for discards: what move reveals to opponents and what it causes them to believe.

Demotions:

- **DEMOTE Active Inference philosophy wars.** useful part is math: EFE / risk / ambiguity / info gain. Do not depend on broad Friston claims.
- **DEMOTE claims that IVD is uniquely novel.** Better claim: independent convergence across AIF, IDS, BAD, LOLA, ERSAC.

### 2. Active inference / epistemic value

**Status: KEEP-CONTEXT; PROMOTE only bounded pieces.**

Useful equations:

```text
F = E_Q[ln Q - ln P(o,s,pi)] >= -ln P(o)
G_pi = E_Q(o,s|pi)[ln Q(s|pi) - ln P(o,s|pi)]
G_pi = risk/pragmatic + ambiguity/epistemic
Q(pi) = softmax(ln P(pi) - F_pi - G_pi)
```

Mahjong mapping:

- Generative model = engine + opponent model + wall posterior.
- Risk = outcome preference / placement EV / deal-in cost.
- Ambiguity = uncertainty after action: hand states, tenpai, waits, opponent type.
- Robust free energy maps to paranoia under model error: larger ambiguity set => more defensive play.

Use in Hydra:

- Good language for unifying belief update and action selection.
- Good source for risk/ambiguity split in search features.
- Not required as full control stack; current Rust engine + neural heads can implement useful terms directly.

### 3. Safe exploitation vs Nash

**Status: PROMOTE as inference/research direction; require safety bounds.**

Core claim: In 4-player Mahjong, pure Nash/self-play is not enough. Multiplayer Nash lacks 2-player zero-sum safety. Human population exploitation can beat near-equilibrium play if bounded.

Evidence:

- Pluribus: 6-player poker dropped strict guarantees, used practical self-play + subgame solving, beat pros (`+48 mbb/hand` in 1-AI-vs-5-human setup; `+32 mbb/hand` in 5-AI-vs-1-human setup).
- Ganzfried multiplayer opponent modeling (`arXiv:2212.06027`): opponent modeling significantly beat exact Nash strategies in 3-player Kuhn poker.
- Equal Share (`arXiv:2406.04201`): in multiplayer symmetric constant-sum games, self-play/Nash-like methods can converge to exploitable strategies and fail to secure equal share.
- Safe Opponent Exploitation / RNR / DBR: interpolation between Nash and best response gives explicit safety/exploitation tradeoff.
- SES (NeurIPS 2022): real-time safe exploitation search; parameter `alpha` trades safety vs exploitation; keeps exploitability bounded while exploiting non-NE opponents.
- Adaptation Safety (ICML 2024): adds robustness to opponent-model errors and deceptive shifts.
- ABD (AAMAS 2025): exploits mistakes beyond depth limit; relevant to depth-limited search.
- KataGo adversarial policies: even superhuman fixed policies have blind spots; exploitation cuts both ways.

Hydra contract:

- Base policy should be strong and population-aware.
- Exploitation must be bounded: expose `alpha`/confidence-like control; lower alpha when model confidence low or opponents strong/unknown.
- Population exploitation is main Tenhou lever because opponents are anonymous; per-hanchan adaptation is secondary.
- Track simple per-opponent features first: riichi rate, push/fold, call timing, dama/tenpai signals, score-conditional risk.

Estimated gain from old notes:

- Population-level Houou exploitation: `+0.5` to `+1.0` dan [inference/speculative].
- Per-hanchan adaptation: `+0.3` to `+0.7` dan [inference/speculative].
- Combined plausible range: `+0.8` to `+1.7` dan; needs Mahjong-specific measurement. Do not present as proven.

Demotions:

- **DEMOTE pure Nash as final objective for 4-player Mahjong.** Use as baseline/safety reference, not sole target.
- **DEMOTE unbounded best response.** It overfits opponent model and invites counter-exploitation.
- **DEMOTE pMCPA as online plan.** It is situation adaptation, not opponent exploitation; old note says deprecated 2026-03-03 due ~100k trajectories/round infeasible in real-time.

### 4. BPR / GSL verdict

**Status: DEMOTE original BPR; PROMOTE GSL-style shared specialist training as optional training tactic.**

Verdict:

- Original Bayesian Policy Reuse (Rosman & Ramamoorthy 2016) is not worth implementing for Hydra as-is.
- GSL-style generalist-specialist training is feasible within budget if specialists share backbone/init and distill back.
- Suphx-style/oracle-guided strong generalist remains higher-confidence path than large policy library.

Why original BPR fails Hydra:

- Tested on toy/simple domains: golf club, online personalization, surveillance grid.
- Requires pre-trained policy library; ignores cost of building it.
- No evidence in Mahjong, poker, or complex hidden-info games.
- Observation model `P(signal | policy, task)` likely too noisy for Mahjong discards/calls.
- 32+ full specialist nets would blow budget (`~16k-19k GPU-hours` in old estimate).

What to keep:

- GSL (ICML 2022): train generalist, fork specialists on subsets, distill back. Reported +19pp to +31% style gains on Procgen/Meta-World/ManiSkill under fixed sample budget.
- Good specialist count heuristic from old survey: 8-16 specialists or 4-16 variations/specialist; 32+ only if cheap adapters/shared backbone.
- Shared-backbone LoRA/adapters are plausible but unpublished for Mahjong RL; treat as engineering experiment, not evidence-backed core.

Hydra rec:

1. Train one excellent generalist with oracle/belief/search support.
2. If specialization needed, use shared-backbone adapters or GSL distillation.
3. Do not build opponent-style policy library (`aggressive`, `defensive`, etc.) unless new data proves routing works.

### 5. Training-time techniques

**Status: PROMOTE cheap/high-signal pieces; DEMOTE heavyweight league/ensemble until training proves need.**

Use now:

- **Auxiliary targets.** KataGo ownership+score heads: +190 Elo at 2.5G queries and ~1.65x convergence vs no value aux. Suphx global reward prediction improved rank distribution qualitatively; Mortal uses next-rank aux loss in production. Hydra action: keep GRP/tenpai/danger heads first-class; consider opponent discard / future-state aux only if data path stays simple.
- **Suit permutation augmentation.** Riichi suits are exact `man/pin/sou` symmetry: 3! = 6 legal variants. Hydra action: implement in loader/batch transform. Treat seat rotation as multi-perspective extraction, not exact symmetry; use all four seats where labels are valid.
- **Reward shaping.** OpenAI Five dense rewards gave ~10x faster 1v1 training and higher TrueSkill; Mahjong ShangTing+bonus shaping reported +$1.37/game in single-player setting. Hydra action: only use potential-based shaping (`gamma*Phi(s') - Phi(s)`) for early training, anneal shaped rewards to zero, final objective stays placement/score. Non-potential bonuses can teach reckless riichi, over-folding, or hand-value chasing.
- **Distillation.** Rusu policy distillation: 25% Atari student reached 108.3% teacher; multi-game KL distillation 116.9% of separate teachers. Hydra action: use human/teacher soft targets for supervised pretrain, compression, and generation-to-generation smoothing.

Consider later:

- **Curriculum.** Endgame-first AlphaZero-style curriculum showed faster early convergence but same asymptote; Mahjong endgame definition is noisy. Use only if training bottleneck is convergence speed.
- **PBT / league.** AlphaStar exploiters added +284 Elo over main agents and improved robustness; cost/ops high. Hydra action: start with simpler self-play/PFSP; revisit exploiters if regression suite finds exploitable patterns or catastrophic forgetting.
- **SPR / self-supervised representation.** SPR beat CURL on Atari 100k (`0.415` vs `0.175` median HNS), but evidence targets pixel/state-representation bottlenecks. For Hydra's hand-crafted tensor, keep only principle: predict future discard/state latents if it cheaply improves shared features.
- **Ensembles.** Inference ensembles violate deployment cost; training ensemble distillation is valid but costs N models. Use only when strong teacher pool already exists.

Action order: aux heads + suit augmentation + PBRS anneal; then distillation; then curriculum/league/SPR only after measured bottleneck.

## Other framework notes from merged sources

### Bayesian opponent modeling / particles

**Status: PROMOTE for belief-state research.**

- Maintain posterior over opponent hidden state and strategy params.
- Update from discards/calls/timing/riichi via likelihood `P(observation | particle)`.
- Use drift/switch model for opponent strategy changes.
- Useful state fields: hand distribution, tenpai probability, aggression, danger, shanten estimate.

### CFR / regret

**Status: KEEP-CONTEXT.**

- CFR is proven base for imperfect-info poker; Mahjong state/action/info sets too large for direct tabular use.
- Use regret concepts for training/search diagnostics; not as direct full-game solver.
- In 4-player games, equilibrium target weaker/non-unique; pair with population/safety framing.

### Causal inference

**Status: KEEP-CONTEXT.**

- Useful for counterfactual opponent reading: “if opponent held X, would they discard Y?”
- Natural fit for defense as intervention: greedy discard transformed by opponent model.
- Do not build full SCM unless concrete caller needs it.

### Information geometry / natural gradient

**Status: KEEP-CONTEXT.**

- Policy simplex geometry explains KL regularization/PPO trust regions.
- Natural gradient is principled but existing optimizer choices may already approximate enough.

### Differential / compositional game theory

**Status: DEMOTE to background.**

- Interesting formal language for value surfaces and modular game composition.
- No direct impl route for Hydra current code. Keep as analogy only.

## Source map

Primary promoted citations:

- Factorised Active Inference for Strategic Multi-Agent Interactions, `arXiv:2411.07362`.
- Bayesian Action Decoder / BAD, `arXiv:1811.01458`.
- O'Donoghue, ERSAC, ICML 2023.
- Loftin et al., Strategically Efficient Exploration in Competitive MARL, UAI 2021.
- Foerster et al., LOLA, `arXiv:1709.04326`.
- Russo & Van Roy, Information-Directed Sampling, `arXiv:1403.5556`.
- Johanson et al., Restricted Nash Response, NIPS 2007.
- Safe Exploitation Search, NeurIPS 2022.
- Ge et al., Safe and Robust Subgame Exploitation / Adaptation Safety, ICML 2024.
- Ganzfried, Opponent Modeling in Multiplayer Imperfect-Information Games, `arXiv:2212.06027`.
- Equal Share, `arXiv:2406.04201`.
- Jia et al., Generalist-Specialist Learning, ICML 2022, `arXiv:2206.12984`.
- Li et al., Suphx, `arXiv:2003.13590`.
- Wu, KataGo / Accelerating Self-Play Learning in Go, `arXiv:1902.10565`.
- Vinyals et al., AlphaStar, Nature 2019.
- Rusu et al., Policy Distillation, `arXiv:1511.06295`.
- Schwarzer et al., SPR, ICLR 2021, `arXiv:2007.05929`.
- OpenAI Five, 2018; Chen & Lai Mahjong reward shaping, `arXiv:2305.04145`; McAleer et al. curriculum, `arXiv:1903.12328`.

Historical/demoted citations retained for context:

- Rosman & Ramamoorthy, Bayesian Policy Reuse, Machine Learning 2016.
- CABPR, Applied Soft Computing 2022.
- Efficient BPR with Scalable Observation Model, IEEE TNNLS 2023.
- VIME, NeurIPS 2016.
- DeepNash, Science 2022.
- Pluribus, Science 2019.
- KataGo adversarial policies, `arXiv:2211.00241`.
- QRE / Magnetic Mirror Descent, Sokota et al. 2022.
- Camerer-Ho-Chong Cognitive Hierarchy, QJE 2004.
