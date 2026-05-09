# Algorithm Watch

Purpose: one routing doc for Hydra algorithm choices. Replaces standalone ACH/LuckyJ, training-paradigm, R-NaD/DRDA, CFR-frontier, 2024-2025 watch notes.

Status keys: **proven** = strong external evidence at useful scale; **speculative** = good idea, weak scale evidence; **reserve** = keep for later phase/watch; **rejected** = do not spend current Hydra effort.

## Current Hydra action

| Area | Status | Action |
|---|---|---|
| Suphx-style asymmetric oracle distillation | proven | Keep Phase 2. Oracle sees hidden tiles during train; student sees legal public/private obs at test. Distill oracle value/GRP signals; no hidden-info leak in exported policy. |
| ACH / LuckyJ-style self-play | proven for 2p; speculative for 4p proof | Keep as Phase 3 candidate loss beside PPO. Implement as flag, not fork. Use same actor/learner, GAE, value, entropy, replay path. |
| ExIt / search-guided training | proven in perfect-info + some IIG; speculative for 4p riichi | Reserve until Hydra has real search/value plumbing. Once search exists, use it to create training targets, not only inference actions. |
| R-NaD | proven neural-scale in Stratego; not open-source official | Reserve. Good evidence for last-iterate neural equilibrium dynamics; too much infra/compute for current path. |
| DRDA | speculative | Watch only. Theory useful for multiplayer POSGs, but paper evidence is tabular only. Do not claim neural-scale DRDA. |
| CFR variants: DDCFR/PDCFR+/Deep DDCFR, MatrixCFR, Embedding CFR | reserve/speculative | Watch for search/resolving modules. Do not replace current BC/oracle/self-play pipeline now. |
| LAMIR/KLUSS/Obscuro | speculative | Watch for future imperfect-info search. Mahjong chance nodes/common-knowledge issues remain open. |
| Offline RL CQL/IQL/DT | rejected as main trainer | Use BC/offline only for warm-start. Dataset ceiling; cannot replace online self-play/oracle phases. |
| Inverse RL | rejected | No strong competitive-game evidence; high compute; Hydra auxiliary heads already cover most useful learned signals. |

## ACH / LuckyJ contract

**What exists.** ACH = actor-critic + Hedge/regret update. Policy logits approximate cumulative regret; policy = `softmax(eta * regret)`. Loss swaps PPO policy term for ACH term; value loss, entropy, GAE, actors, learners stay same.

**Evidence.** ICLR 2022 Tencent/LuckyJ group. Guarantees `O(T^-1/2)` average-policy Nash convergence in 2-player zero-sum. Mahjong experiments were 2-player reduction; 4-player riichi has no formal convergence guarantee. Third-party ygo-agent shows practical drop-in ACH loss in normal PPO loop. Official ACH_poker exists but old C++/OpenSpiel/TensorFlow path.

**Hydra action.** Implement ACH only as alternative loss behind config. Start params from paper/report: `eta=1`, `logit_threshold=6`, `clip=0.5`, entropy nonzero, value coef normal, GAE lambda `0.95`. A/B vs PPO under identical self-play. Claim: “game-theoretic self-play candidate with 2p guarantee, empirical 4p risk,” not “proved 4p Nash.”

**LuckyJ reconstruction.** LuckyJ likely used ACH + league/frozen opponents + RVR; inference search via OLSS and possible search-as-feature. OLSS evidence is 2-player Mahjong and needs explicit subgame solving; reported compute class was thousands CPUs + V100s. For Hydra: do not chase OLSS now. If search enters, make consistency boundary explicit: subgame/resolving module owns search targets; policy module owns distilled fast action.

## Asymmetric oracle + ExIt

**Oracle distillation: proven.** Suphx trained perfect-information oracle/value guidance, distilled into normal partial-information student, reached top 0.01% Tenhou / stable 10-dan-level result. This directly supports Hydra Phase 2.

**Hydra contract.** Oracle may see all tiles only in train jobs. Student inputs/test artifacts must not include hidden tiles. Distill value/GRP/aux targets; audit feature schema at export.

**ExIt: reserve.** Expert Iteration loop: search expert generates stronger policy/value targets; apprentice imitates; improved apprentice guides next search. Proven pattern in AlphaZero/Hex; IIG version needs CFR/IS-MCTS/LAMIR/SoG-like search, not vanilla MCTS. Use after search exists. Search-as-training-target is likely higher value than search-as-only-inference.

**Student of Games context.** Strong general template: sound search + self-play + game-theoretic reasoning across perfect/imperfect info. Reserve as architecture reference, not current impl target.

## R-NaD / DRDA

**R-NaD: proven but expensive.** DeepNash used R-NaD at neural scale on Stratego (`10^535` states), 1024 TPU nodes, U-Net/pyramid conv net, V-trace, NeuRD, regularization target updates, last-iterate dynamics. No official OpenSpiel R-NaD. Community PyTorch impls exist but small-scale.

**Hydra action.** Cite R-NaD as evidence that regularized Nash dynamics can scale with massive infra. Do not use as near-term dependency. If revisited, isolate as new trainer backend; do not blur with PPO/ACH semantics.

**DRDA: speculative for Hydra.** ICLR 2025 DRDA extends regularized dynamics to multiplayer POSGs and claims exact Nash convergence via multi-round KL-regularized discounted aggregation. But experiments are small/tabular/dynamic-programming only: matrix games, Kuhn/Leduc variants, tiny POSGs/grid worlds. No neural experiments, no open impl observed in prior report.

**Hydra action.** Watch. Do not claim “DRDA proven at neural scale.” Accurate claim: “DRDA is theoretical multiplayer follow-up to R-NaD; neural-scale validation remains open.”

## CFR variants watch

| Algorithm | Status | Keep conclusion | Hydra action |
|---|---|---|---|
| DDCFR | reserve | Tencent 2024 ICLR Spotlight; meta-learns CFR discounting instead of fixed schedules. Game-agnostic CFR improvement. | Watch for search/resolving, not BC/oracle path. |
| Deep DDCFR / VR-DeepDCFR+ | speculative/reserve | Claimed neural approximation for advanced discounted/clipped CFR. Newer and less battle-tested. | Track; maybe useful if Hydra builds neural CFR resolver. |
| PDCFR+ | reserve | Combines predictive/optimistic update with discounted CFR+. | Candidate for solver core if tabular/abstracted subgames exist. |
| MatrixCFR / GPU CFR | reserve | Recasts CFR traversal as sparse/dense GPU ops; speedup potential, memory-heavy. | Consider only for bounded abstractions/subgames. |
| Embedding CFR | speculative | Continuous info-set embeddings instead of hard clusters. Strong fit to huge Mahjong info sets, but new. | Research-only; possible future continuous-belief resolver. |
| ReBeL / DeepStack-style CVNs | reserve | Depth-limited CFR with learned public-state values; proven poker lineage. | Good template if public-belief/value state is built. |

Boundary: CFR extends beyond 2p, but strong exploitability/Nash guarantees are cleanest in 2p zero-sum. For 4p riichi, state solution concept and evaluation metric before claiming convergence.

## 2024-2025 algorithm watch

**High-interest, not current build:**
- **KLUSS/Obscuro**: search without common-knowledge assumption. Strong conceptual fit because Mahjong players do not share hidden-hand knowledge. Evidence mostly 2p/Fog-of-War-chess style; 4p Mahjong adaptation open.
- **LAMIR**: learned abstract model + CFR+ resolving for imperfect-info games; reported large wins vs R-NaD on Goofspiel variants. Limitations: no explicit chance-node modeling in report, imperfect-recall abstraction risk, no Mahjong-scale evidence.
- **Opponent modeling with in-context search**: Tencent/LuckyJ direction. Relevant to 4-player exploitation; watch, but avoid ungrounded architecture claims.
- **RegFTRL/adaptive regularization**: useful theory for last-iterate convergence, mainly 2p zero-sum building block.
- **MCU/eval frameworks**: evaluation infrastructure, not training algorithm.

**Novelty filter.** “Apply DDCFR/DRDA to Mahjong” alone is weak novelty. Stronger research claims require one new axis: no-common-knowledge 4p search, continuous info-set equilibrium, or meta-learned multiplayer equilibrium selection. For Hydra product path, strength matters more than paper novelty: keep pipeline simple until search module earns cost.

## Rejected / demoted

- **Offline RL as main path**: CQL/IQL/DT useful for warm-start only. Dataset ceiling, conservative bias, no self-improvement. Mortal-style CQL offline then online RL supports demotion.
- **Decision Transformer for play policy**: awkward return conditioning in multiplayer competitive setting; no strong Mahjong evidence.
- **Inverse RL**: high compute, reward non-identifiability, no demonstrated game-AI gain over hand-built placement reward + aux heads.
- **Pure OLSS now**: sound idea but compute-heavy, 2p evidence, explicit tree build/solve burden. Reserve for future search effort.

## Read routing

Read this file first for algorithm choice. Then:
- Current implemented contracts: `docs/CURRENT_STATUS.md`, crate READMEs.
- Training workflow mechanics: `docs/TRAINING_RUNBOOK.md`.
- Architecture/spec context: `research/design/HYDRA_FINAL.md`, `research/design/HYDRA_RECONCILIATION.md`, `research/design/HYDRA_ARCHIVE.md`.
- Reward/variance: see `research/evidence/RESEARCH_DIGEST.md` and `research/design/HYDRA_ARCHIVE.md` for retired RVR notes.

Historical source docs removed after merge:
- `research/evidence/rnad_drda_report.md`
- `research/evidence/game-ai-papers-2024-2025.md`
- `research/intel/ACH_RESEARCH.md`
- `research/comparisons/TRAINING_PARADIGMS.md`
- `research/comparisons/LUCKYJ_PROPOSAL.md`
