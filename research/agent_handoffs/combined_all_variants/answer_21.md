# Hydra architecture selection prompt — broad artifact-first first-principles search

<role>
Produce an implementation-ready blueprint for determining the best architecture for Hydra.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Determine the logically best architecture for Hydra from first principles.
It does not have to be SE-ResNet.
Assume implementation capability is unconstrained: we can code anything reasonable.
Do not reject an architecture family just because the current repository or current stack is shaped differently.
However, do not ignore real runtime, training, sample-efficiency, deployment, search-integration, and compute-budget constraints.
Treat the current Hydra plan as one candidate family and one evidence packet, not as the answer.

We want a detailed answer that makes clear:
- what the irreducible problem constraints of 4-player Riichi Mahjong actually are
- which current Hydra assumptions are hard constraints versus contestable doctrine
- whether a single architecture should be used for actor and learner, or whether they should differ
- whether the best answer is convolutional, attention-based, recurrent, state-space, hybrid, entity-based, set-based, graph-based, or something else
- whether the best answer should keep a fixed tile-tensor path, add an event-history path, or replace the whole representation
- what the minimum decisive experiments are if the evidence is still underdetermined
- what should be rejected, deferred, or kept only as reserve-shelf ideas
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- no architecture fashion takes
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- after grounding in the artifacts, explore many adjacent fields for competing formulations of the same problem, keep searching for interesting fragments worth fusing together, and continue the explore -> think hard -> validate loop until the strongest fused formulation either survives or is killed by the artifact constraints
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not dump logic; every important mechanism, threshold, recommendation, and architecture move should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now.
They are not guaranteed to be fully correct.
Treat them as evidence to inspect and critique, not truth to inherit.
High chance some of them are incomplete, misleading, stale, semantically wrong, or overconfident by omission, so validate everything.
</artifact_note>

<hard_guardrails>
1. Do not assume the best Hydra architecture is the current one.
2. Do not assume the best Hydra architecture is a single backbone plus many shallow heads; prove it or reject it.
3. Do not assume the best Hydra architecture must preserve the current 192x34 tensor representation unchanged; classify that representation as hard constraint, soft constraint, convenience, or liability.
4. Do not assume the best Hydra architecture must preserve the current multi-head split unchanged; evaluate whether some tasks should move into trunk, adapters, auxiliary branches, or teacher-only modules.
5. Do not assume the best Hydra architecture must be symmetric between fast actor and slow learner/search-side model.
6. Do not assume search/belief/Hand-EV dynamic features must be injected at the input; consider late fusion, cross-attention, sidecar encoders, or separate planning modules.
7. Do not reject transformer, recurrent, state-space, graph, set, hybrid, or dual-path architectures just because the repo is currently conv-centric.
8. Do not accept transformer, recurrent, state-space, graph, set, hybrid, or dual-path architectures just because they sound more modern.
9. Do not let implementation convenience beat logical fit.
10. Do not let novelty beat strength-per-effort unless the evidence says the simpler family is capped.
11. If the evidence is insufficient, say underdetermined and specify the smallest decisive experiment matrix instead of faking certainty.
12. If the best answer differs for ideal unconstrained architecture versus best architecture under Hydra’s stated runtime/compute goals, explicitly separate those answers.
</hard_guardrails>

<assumption_protocol>
Before comparing architecture families, build an assumption ledger with four buckets:
- bucket A: irreducible game/problem facts
- bucket B: explicit project objectives and runtime constraints
- bucket C: current repo implementation realities
- bucket D: contestable doctrine / hypotheses / design bets

You must not blur these buckets together.
Anything in bucket C or D may be overturned if a stronger architecture survives the evaluation.
Only bucket A and truly binding parts of bucket B should be treated as hard constraints.
</assumption_protocol>

<anti_anchor_protocol>
You must use the following anti-anchor sequence:
step 1: derive requirements from the game, action structure, information structure, and objectives BEFORE reading doctrine excerpts as conclusions
step 2: define candidate architecture families in abstract terms
step 3: steelman the current Hydra plan as one candidate, not the baseline truth
step 4: steelman at least one serious non-conv family and at least one hybrid family
step 5: compare all families under the same rubric
step 6: perform a red-team pass against the leading candidate
step 7: only then state the recommendation, if any

You must explicitly answer: “what would have made me incorrectly choose the current doctrine by default?”
</anti_anchor_protocol>

<minimum_candidate_family_set>
At minimum, compare all of these families unless an artifact-grounded reason makes one literally inapplicable:
family 1: pure fixed-tensor residual conv family
family 2: SE-ResNet or related channel-attention conv family
family 3: ConvNeXt-style or modernized conv family over tile axis
family 4: event-sequence transformer family
family 5: tile-token or entity-token transformer/set-transformer family
family 6: recurrent or state-space event-history family
family 7: dual-path hybrid family (fixed tensor trunk + history encoder)
family 8: graph/entity/set family over players, melds, discards, and tile groups
family 9: asymmetric actor/learner family where fast actor and slow learner/search-side network differ materially
family 10: any adjacent-field formulation that survives validation and does not collapse into one of the above

For each family, you must say whether it is:
- best overall
- best under ideal-but-still-realistic Hydra objectives
- best under current runtime/search constraints
- only good as a subsystem
- only good as a teacher-side model
- reserve shelf only
- or reject
</minimum_candidate_family_set>

<evaluation_rubric>
Use a weighted rubric with explicit scoring or explicit pairwise dominance logic.
First-order dimensions that must be evaluated:
- representation fit to Mahjong’s public state and partial observability
- ability to exploit tile geometry and local combinatorics
- ability to capture temporal opponent-read patterns and discard/call sequences
- sample efficiency under Hydra-like compute budgets
- fast-path inference latency for actor/runtime use
- slower learner/search-side usefulness under pondering/distillation
- compatibility with multi-head supervision (policy, value, GRP, danger, tenpai, belief, search residuals, etc.)
- robustness when dynamic search/belief/Hand-EV features are absent or stale
- ease of distilling search/oracle signals into the deployable policy
- support for selective search rather than universal expensive search
- calibration potential for safety and belief outputs
- scaling path if Hydra later earns more compute

Tie-breakers that may matter only after first-order comparison:
- implementation complexity
- maintenance burden
- stack compatibility
- debugging surface area
- profiling predictability
- licensing or ecosystem friction if relevant

Do not let a tie-breaker decide the winner if a family is materially worse on first-order fit.
</evaluation_rubric>

<required_questions>
You must answer all of these:
Q1. What information patterns actually dominate strong Mahjong play: local tile-shape reasoning, cross-player relational reasoning, temporal opponent modeling, search-conditioned adaptation, or some mixture?
Q2. Which of those patterns need to live in the deployable fast actor, and which can be outsourced to teacher/search/pondering/distillation machinery?
Q3. Is the best architecture likely to be single-path or multi-path?
Q4. Should actor and learner share architecture at all, or only share some representation ideas?
Q5. Is the current 192x34 tensor a core strength, a neutral compatibility layer, or an anchor holding the project back?
Q6. Is opponent-history modeling central enough to require a dedicated sequence module?
Q7. Is the best architecture likely to preserve explicit safety channels and structured Hand-EV/belief features, or absorb them into a different representation?
Q8. Is the best architecture likely to be end-to-end monolithic, or should it be modular with specialized trunks or sidecars?
Q9. What is the smallest architecture leap that has a realistic chance to beat the current plan?
Q10. What would falsify the recommended architecture quickly and cheaply?
</required_questions>

<required_output_shape>
The answer must be a blueprint with these practical deliverables:
- an assumption ledger
- a hard-facts section
- a contestable-doctrine section
- a candidate-family generation section
- a family-by-family evaluation table
- a steelman for SE-ResNet
- a steelman for the strongest non-SE alternative
- a comparison of ideal architecture vs best practical Hydra architecture if they differ
- a recommended architecture or an explicit underdetermined verdict
- a decisive experiment matrix
- a migration blueprint if the recommendation differs from current doctrine
- revisit triggers that would cause the decision to change later

The answer must feel buildable or directly auditable.
</required_output_shape>

<failure_modes_to_ban>
- do not answer with “SE-ResNet is good enough” unless you show why stronger candidates fail
- do not answer with “transformers are better” unless you show why they win under Hydra’s actual objectives
- do not let the current codebase shape masquerade as proof of optimality
- do not treat old design docs as ground truth if live runtime docs or code disagree
- do not cite architecture papers abstractly without method details, scope limits, or failure cases
- do not declare exact thresholds or budgets without visible support or explicit proposal status
- do not force a single-family winner if the real answer is asymmetric actor/learner or hybrid trunk/sidecar
- do not confuse teacher-side architecture freedom with fast actor requirements
- do not confuse a better opponent-modeling subsystem with a better full-agent backbone
</failure_modes_to_ban>

<architecture_search_notes>
Important: “best architecture for Hydra” may mean one of the following, and you must disentangle them:
- best full system architecture if we were rebuilding from scratch
- best deployable fast actor architecture
- best slow learner / teacher architecture
- best search-side value / policy / belief helper architecture
- best migration target from the current repo state

You may recommend different architectures for different roles.
You may recommend preserving the current actor while changing the learner, or vice versa.
You may recommend a hybrid where the current SE-ResNet remains only as one component.
You may recommend rejecting SE-ResNet entirely.
You may recommend keeping SE-ResNet.
You may recommend something more exotic if the evidence survives scrutiny.
But every important move must be justified.
</architecture_search_notes>

<artifacts>
The artifacts below are dense starting evidence packets.
They are not truth.
They are not the answer.
They are here so you can critique, validate, and outgrow them if needed.

## Artifact 01 — research/design/HYDRA_FINAL.md lines 1-220
Source label: AFINAL_A
Path: research/design/HYDRA_FINAL.md
Use: treat this as evidence, not truth.
```text
[AFINAL_A L0001] # HYDRA-OMEGA: A Maximum-Ceiling 4-Player Riichi Mahjong AI (Complexity-Free)
[AFINAL_A L0002] 
[AFINAL_A L0003] **Single-source-of-truth (SSOT).** This document supersedes the two prior internal variants: the throughput-first "compute-constrained elegance" plan and the "information-geometric / all-out" plan. HYDRA-OMEGA keeps their best ideas, removes their ceilings, and adds a rigorously-grounded robustness layer.
[AFINAL_A L0004] 
[AFINAL_A L0005] ---
[AFINAL_A L0006] 
[AFINAL_A L0007] ## 0. Abstract
[AFINAL_A L0008] 
[AFINAL_A L0009] 4-player Riichi Mahjong is a large, general-sum, imperfect-information game with a **finite shared hidden pool** (multivariate hypergeometric), **hard conservation constraints**, and **decision-critical correlations** that strengthen late game.
[AFINAL_A L0010] 
[AFINAL_A L0011] HYDRA-OMEGA is built around one central engine:
[AFINAL_A L0012] 
[AFINAL_A L0013] > **ExIt + Pondering + Search-as-Feature (SaF)**
[AFINAL_A L0014] > Deep anytime belief-search generates training targets continuously during self-play, amplified by opponent-turn idle time; those targets are amortized back into the policy/value networks so inference remains fast.
[AFINAL_A L0015] 
[AFINAL_A L0016] The system couples this engine with:
[AFINAL_A L0017] 
[AFINAL_A L0018] 1. **Belief correctness with constraints**: SIB / Mixture-SIB (Sinkhorn KL projection) + **CT-SMC exact contingency-table sampler** exploiting Mahjong's small row counts ($r \le 4$) for correlation-faithful beliefs via a 3,375-state DP (~4M ops, <1ms in Rust).
[AFINAL_A L0019] 2. **Anytime Factored-Belief Search (AFBS)**: top-k pruning, heavy caching, incremental reuse, predictive pondering, and **endgame exactification** (exact chance enumeration when wall $\le 10$).
[AFINAL_A L0020] 3. **Robust opponent modeling inside search**: opponent nodes solved as distributionally robust soft-min within a KL uncertainty set around the learned opponent policy.
[AFINAL_A L0021] 4. **Conservative safety math that is tight enough to matter**: Negative dependence / Strongly Rayleigh + Hunter/Kounias union tightening + bounded-error Monte Carlo intersections.
[AFINAL_A L0022] 5. **Hand-EV oracle features**: CPU-precomputed per-discard tenpai probability, win probability, expected score, and ukeire -- proven by Suphx as their biggest practical win.
[AFINAL_A L0023] 6. **ACH training** (Actor-Critic Hedge, LuckyJ's algorithm): +0.4 fan over PPO via Hedge-derived conservative clipping. Global $\eta$, per-(s,a) gating, standard GAE, one epoch per batch. Compatible with oracle guiding via CTDE.
[AFINAL_A L0024] 7. **Two-tier network** (12-block actor / 24-block learner): 40-block teacher data-starved at 7 spp on hard states only. 24-block learner (245 spp) handles both training and deep AFBS. Continuous distillation learner -> actor.
[AFINAL_A L0025] 
[AFINAL_A L0026] Goal: **maximize expected Tenhou stable rank**; LuckyJ's 10.68 stable dan is the current public benchmark.
[AFINAL_A L0027] 
[AFINAL_A L0028] ---
[AFINAL_A L0029] 
[AFINAL_A L0030] ## 1. Design principles
[AFINAL_A L0031] 
[AFINAL_A L0032] ### P1. Ceiling first, then amortize
[AFINAL_A L0033] If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, deep search, offline solvers, or distillation targets -- not in the critical inference loop.
[AFINAL_A L0034] 
[AFINAL_A L0035] ### P2. Search targets must optimize the information state, not the hidden state
[AFINAL_A L0036] Any training target used to update the deployable policy must be a function of the public/information state, not privileged knowledge. We allow perfect-information networks for variance reduction and diagnostics, but the improvement operator must respect the information constraints.
[AFINAL_A L0037] 
[AFINAL_A L0038] ### P3. Every "guarantee-like" claim must be either a theorem (with conditions), a bound (with explicit constants), or an empirical gate with a measurable pass/fail threshold.
[AFINAL_A L0039] 
[AFINAL_A L0040] ### P4. Robustness is not optional in 4-player general-sum
[AFINAL_A L0041] Instead of equilibrium-style guarantees (which do not cleanly extend to 4p), we use distributional robustness: robust to belief error, robust to opponent policy misspecification, robust to population shifts.
[AFINAL_A L0042] 
[AFINAL_A L0043] ---
[AFINAL_A L0044] 
[AFINAL_A L0045] ## 2. Game model and notation
[AFINAL_A L0046] 
[AFINAL_A L0047] - Tile types: $k \in \{1,\dots,34\}$, multiplicity 4, total 136 tiles.
[AFINAL_A L0048] - Hidden locations: $z \in \{1,2,3,W\}$: three opponent concealed hands + wall remainder.
[AFINAL_A L0049] - Public information state at time $t$: $I_t$ (our hand, discards/melds, riichi, dora, scores, round meta).
[AFINAL_A L0050] - Remaining tile counts: $r_t(k) = 4 - \mathrm{visible}_t(k)$.
[AFINAL_A L0051] - Hidden location sizes: $s_t(z) \in \mathbb{Z}_{\ge 0}$, $\sum_z s_t(z) = \sum_k r_t(k)$.
[AFINAL_A L0052] - Hidden allocation matrix: $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$, $\sum_z X_t(k,z) = r_t(k)$, $\sum_k X_t(k,z)=s_t(z)$.
[AFINAL_A L0053] 
[AFINAL_A L0054] Under purely random dealing, $X_t$ is multivariate hypergeometric; under strategic play, $p(X_t\mid I_t)$ is shaped by action likelihoods.
[AFINAL_A L0055] 
[AFINAL_A L0056] ---
[AFINAL_A L0057] 
[AFINAL_A L0058] ## 3. System overview -- four interacting loops
[AFINAL_A L0059] 
[AFINAL_A L0060] **Loop A: Belief loop** -- Mixture-SIB for fast marginal updates under constraints, particle SMC for joint correlation capture.
[AFINAL_A L0061] 
[AFINAL_A L0062] **Loop B: Search loop** -- AFBS on $I_t$ with belief $q_t$: on-turn (shallow, feature-producing), off-turn/pondering (deep, cached, predictive).
[AFINAL_A L0063] 
[AFINAL_A L0064] **Loop C: Distillation loop** -- Train policy/value to predict $\pi^{\text{ExIt}}$, $V^{\text{ExIt}}$, and calibrated safety features.
[AFINAL_A L0065] 
[AFINAL_A L0066] **Loop D: Population loop** -- League with self-play variants, human-style anchors, adversarial exploiters.
[AFINAL_A L0067] 
[AFINAL_A L0068] ---
[AFINAL_A L0069] 
[AFINAL_A L0070] ## 4. Neural architecture
[AFINAL_A L0071] 
[AFINAL_A L0072] ### 4.1 Input tensor
[AFINAL_A L0073] 
[AFINAL_A L0074] **Group A -- Public encoding (~80-120 planes):** Hand, ordered discards (recency), open melds, riichi state, dora, round/scoring context, shanten/uke-ire summaries.
[AFINAL_A L0075] 
[AFINAL_A L0076] **Group B -- Safety planes (~23 planes):** Tenpai hints, furiten, genbutsu/suji/kabe safe-tile masks.
[AFINAL_A L0077] 
[AFINAL_A L0078] **Group C -- Search and belief features (dynamic, ~60-200 planes):** Belief marginals $B_t(k,z)$, mixture weights/entropy/ESS, AFBS action deltas $\Delta Q(a)$, risk estimates, robust opponent stress indicators. Zeroed with presence mask when unavailable.
[AFINAL_A L0079] 
[AFINAL_A L0080] **Group D -- Hand-EV oracle features (~34-68 planes, CPU-precomputed):** For each discard candidate $a$ (34 tile types), pre-compute exact look-ahead analysis:
[AFINAL_A L0081] - $P_{\text{tenpai}}^{(d)}(a)$: probability of reaching tenpai within $d \in \{1,2,3\}$ self-draws.
[AFINAL_A L0082] - $P_{\text{win}}^{(d)}(a)$: probability of winning within $d$ draws (tsumo + simplified ron model).
[AFINAL_A L0083] - $\mathbb{E}[\text{score} \mid \text{win}, a]$: expected hand value (han/fu/score) if we win after discarding $a$.
[AFINAL_A L0084] - Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.
[AFINAL_A L0085] 
[AFINAL_A L0086] These features are computed by the CPU-side hand analyzer (`shanten_batch.rs` + scoring engine) using belief-weighted remaining tile counts from CT-SMC. Zero GPU cost -- CPU pre-computes during game step processing. Suphx reported these look-ahead features as their single biggest practical improvement (Li et al. 2020).
[AFINAL_A L0087] 
[AFINAL_A L0088] ### 4.2 Two-tier architecture
[AFINAL_A L0089] 
[AFINAL_A L0090] **Why not monolithic 40-block?** At 2000 GPU hours, self-play generates ~2.45B decisions (35M games). Samples-per-parameter ratio:
[AFINAL_A L0091] 
[AFINAL_A L0092] | Config | Params | Samples/param | vs Mortal (514) | Verdict |
[AFINAL_A L0093] |--------|-------:|-------------:|----------------:|---------|
[AFINAL_A L0094] | 40-block mono | 16.5M | 148 | 0.29x | Undertrained AND too slow for rollouts |
[AFINAL_A L0095] | 24-block | 10M | 245 | 0.48x | Viable with ExIt quality boost |
[AFINAL_A L0096] | 12-block | 5M | 490 | 0.95x | Well-trained, fast inference |
[AFINAL_A L0097] 
[AFINAL_A L0098] (Based on ~35M games * 70 decisions = 2.45B total samples.)
[AFINAL_A L0099] 
[AFINAL_A L0100] A 40-block teacher trained only on hard states (1-5%) gets just ~7 spp -- catastrophic data starvation. **Two-tier architecture avoids this paradox:**
[AFINAL_A L0101] 
[AFINAL_A L0102] | Network | Blocks | Params | Role | GPU |
[AFINAL_A L0103] |---------|-------:|-------:|------|-----|
[AFINAL_A L0104] | **LearnerNet** | 24 | ~10M | Training (ACH/ExIt) + deep AFBS on hard positions | GPU 0-1 (train), GPU 3 (search) |
[AFINAL_A L0105] | **ActorNet** | 12 | ~5M | Self-play data generation + shallow SaF features | GPU 2 |
[AFINAL_A L0106] 
[AFINAL_A L0107] All use SE-ResNet with GroupNorm(32) and Mish. bf16 precision. **Continuous distillation**: Learner -> Actor (every 1-2 minutes, IMPALA-style). ActorNet inference: ~0.2ms. LearnerNet inference: ~0.35ms. LearnerNet runs deep AFBS on GPU 3 for hard-position ExIt labels.
[AFINAL_A L0108] 
[AFINAL_A L0109] ### 4.3 Heads (multi-task)
[AFINAL_A L0110] 
[AFINAL_A L0111] **Core decision heads:** (1) Policy $\pi_\theta(a\mid I_t)$, 46 actions. (2) Value $V_\theta(I_t)$, scalar. (3) Score distribution: pdf + cdf (64 bins, KataGo-style).
[AFINAL_A L0112] 
[AFINAL_A L0113] **Opponent and safety heads:** (4) Opponent tenpai (3 sigmoids). (5) Opponent next discard (3x34). (6) Danger: per-tile deal-in probability (3x34).
[AFINAL_A L0114] 
[AFINAL_A L0115] **Belief heads:** (7) Mixture-SIB external fields $F_\theta^{(\ell)}(k,z)$ and mixture weight logits. (8) Opponent hand-type latent predictor.
[AFINAL_A L0116] 
[AFINAL_A L0117] **Search distillation heads:** (9) $\Delta Q$ regression (predict search advantage over baseline). (10) Safety bound residual (predict conservatism gap).
[AFINAL_A L0118] 
[AFINAL_A L0119] ---
[AFINAL_A L0120] 
[AFINAL_A L0121] ## 5. Belief inference: SIB, Mixture-SIB, and particle posterior
[AFINAL_A L0122] 
[AFINAL_A L0123] ### 5.1 SIB as KL projection
[AFINAL_A L0124] 
[AFINAL_A L0125] Let $K_\theta(k,z)=\exp(F_\theta(k,z))>0$. The transportation polytope: $\mathcal{U}(r_t,s_t)=\{B\ge0: B\mathbf{1}=r_t, B^\top \mathbf{1}=s_t\}$.
[AFINAL_A L0126] 
[AFINAL_A L0127] **SIB operator:** $\mathrm{SIB}(K_\theta;r_t,s_t) := \arg\min_{B\in\mathcal{U}} D_{\mathrm{KL}}(B\|K_\theta)$. Solution: $B^*=\mathrm{diag}(u)\cdot K_\theta\cdot\mathrm{diag}(v)$ via Sinkhorn-Knopp.
[AFINAL_A L0128] 
[AFINAL_A L0129] ### 5.2 Mixture-SIB for multimodality
[AFINAL_A L0130] 
[AFINAL_A L0131] $L$ components: $q_t(X)=\sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X)$, each $B_t^{(\ell)}=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t)$.
[AFINAL_A L0132] 
[AFINAL_A L0133] Weight update (Bayes): $w_{t+1}^{(\ell)}\propto w_t^{(\ell)} \cdot p_\phi(e_t\mid I_t, B_t^{(\ell)}, \ell)$ where $e_t$ is the observed public event (opponent discard, call, riichi, or pass). Anti-collapse via entropy regularizer, split-merge on low ESS, diversity penalty between components.
[AFINAL_A L0134] 
[AFINAL_A L0135] ### 5.3 Particle posterior (SMC) for joint structure
[AFINAL_A L0136] 
[AFINAL_A L0137] Particles $\{X_t^{(p)},\alpha_t^{(p)}\}_{p=1}^P$ targeting $p(X_t\mid I_t)$. Proposal via constrained sequential fill guided by mixture component. Resample when $\mathrm{ESS}<0.4P$. Rejuvenation via Metropolis-Hastings swap moves preserving row/col sums.
[AFINAL_A L0138] 
[AFINAL_A L0139] ### 5.4 Correlation scale diagnostic
[AFINAL_A L0140] 
[AFINAL_A L0141] $|\rho_{ij}|=\sqrt{K_i K_j} / \sqrt{(H-K_i)(H-K_j)}$. At $H=50$, $K=4$: $|\rho|=4/46=0.087$; at $H=25$: $|\rho|=0.190$. Late-game correlations motivate Mixture-SIB + particles over first-moment alone.
[AFINAL_A L0142] 
[AFINAL_A L0143] ### 5.5 CT-SMC: Exact contingency-table sampling (replaces generic particle proposals)
[AFINAL_A L0144] 
[AFINAL_A L0145] The hidden allocation $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$ is a **fixed-margin contingency table**. Key Mahjong insight: each row sum $r_t(k) \le 4$, so per-row compositions are tiny ($\binom{r+3}{3} \le 35$).
[AFINAL_A L0146] 
[AFINAL_A L0147] **Exact DP partition function.** Order tile types $k=1,\dots,34$. Let residual capacities be $\mathbf{c}=(c_1,c_2,c_3,c_W)$. Define:
[AFINAL_A L0148] 
[AFINAL_A L0149] $$Z_k(\mathbf{c}) = \sum_{x \in \mathcal{X}_k(\mathbf{c})} \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x), \quad Z_{35}(\mathbf{0})=1$$
[AFINAL_A L0150] 
[AFINAL_A L0151] where $\phi_k(x)=\prod_j \omega_{kj}^{x_j}$ is the learned field weight per row. Key insight: $c_W = R_k - (c_1+c_2+c_3)$ is **derived** (where $R_k = \sum_{t \ge k} r_t$ is the remaining hidden tile count at DP step $k$), so the DP state is 3D: $(c_1,c_2,c_3)$. State count: $\le (15)^3 = 3{,}375$ (max 14 tiles after draw, before discard). Each transition enumerates $\le 35$ compositions. Total: $\sim 34 \times 3375 \times 35 \approx 4.0M$ ops -- **trivially sub-millisecond in Rust**. Use log-space DP for numerical stability.
[AFINAL_A L0152] 
[AFINAL_A L0153] **Exact backward sampling:** $p(x_k = x \mid \mathbf{c}) = \phi_k(x) \cdot Z_{k+1}(\mathbf{c}-x) / Z_k(\mathbf{c})$. This gives **exact samples with correct correlations** from the conservation-constrained distribution -- not mean-field approximations.
[AFINAL_A L0154] 
[AFINAL_A L0155] **SMC integration.** The full posterior is $p(X \mid \mathcal{O}_{1:t}) \propto p_0(X) \cdot L(X)$ where $L(X)$ is the opponent action likelihood. Sample $X^{(n)} \sim p_0$ via CT-DP (fast, correlation-correct), weight $w^{(n)} \leftarrow L(X^{(n)})$, normalize and resample. The proposal already respects the hardest constraint (tile conservation) exactly, so ESS stays high.
[AFINAL_A L0156] 
[AFINAL_A L0157] **What CT-SMC replaces:** The generic particle proposal from Section 5.3. Mixture-SIB is KEPT as the fast amortized belief head for network input; CT-SMC is the search-grade belief for AFBS and safety queries.
[AFINAL_A L0158] 
[AFINAL_A L0159] **Validation gates:**
[AFINAL_A L0160] - **Gate A (posterior log-likelihood):** At end of hand, evaluate $\log p(X^* \mid \mathcal{O}_{1:t})$ under CT-SMC vs generic CMPS. CT-SMC must win.
[AFINAL_A L0161] - **Gate B (pairwise MI calibration):** Compare estimated $I(\mathbf{1}\{A \in H_z\}; \mathbf{1}\{B \in H_z\})$ vs empirical. Must capture correlations generic CMPS misses.
[AFINAL_A L0162] 
[AFINAL_A L0163] ---
[AFINAL_A L0164] 
[AFINAL_A L0165] ## 6. Conservative safety estimates without over-folding
[AFINAL_A L0166] 
[AFINAL_A L0167] ### 6.1 Strongly Rayleigh / negative dependence foundations
[AFINAL_A L0168] 
[AFINAL_A L0169] The remaining-tile distribution under "draw without replacement" is Strongly Rayleigh (BBL 2009), implying strong negative dependence. Used only for bounding monotone danger events.
[AFINAL_A L0170] 
[AFINAL_A L0171] ### 6.2 Hunter bound (spanning tree correction)
[AFINAL_A L0172] 
[AFINAL_A L0173] For threat events $\{A_j\}_{j=1}^J$ and any spanning tree $\mathcal{T}$: $P(\bigcup_j A_j) \le \sum_j P(A_j) - \sum_{(u,v)\in\mathcal{T}} P(A_u\cap A_v)$. Maximum-weight spanning tree gives the tightest bound. Kounias (1968) bound is a member; we take the minimum computable bound.
[AFINAL_A L0174] 
[AFINAL_A L0175] ### 6.3 Computing intersections reliably
[AFINAL_A L0176] 
[AFINAL_A L0177] Analytic formulas for simple events; particle estimates with Hoeffding CIs otherwise. Never use an intersection estimate unless CI half-width $<\delta_\cap$ (e.g., 0.01). Fall back to conservative Boole if CI not met.
[AFINAL_A L0178] 
[AFINAL_A L0179] ---
[AFINAL_A L0180] 
[AFINAL_A L0181] ## 7. Anytime Factored Belief Search (AFBS)
[AFINAL_A L0182] 
[AFINAL_A L0183] ### 7.1 Tree structure
[AFINAL_A L0184] 
[AFINAL_A L0185] Node state: $(I, \mathcal{B}, \mathcal{P})$ -- info state, Mixture-SIB summary, particle set handle.
[AFINAL_A L0186] 
[AFINAL_A L0187] ### 7.2 Beam parameters
[AFINAL_A L0188] 
[AFINAL_A L0189] | Mode | Beam W | Depth D | Particles P | Mixture L |
[AFINAL_A L0190] |------|-------:|--------:|------------:|----------:|
[AFINAL_A L0191] | On-turn | 64-128 | 4-6 | 128-256 | 4-8 |
[AFINAL_A L0192] | Ponder | 256-1024 | 10-14 | 1024-4096 | 8-32 |
[AFINAL_A L0193] 
[AFINAL_A L0194] ### 7.3 Caches
[AFINAL_A L0195] 
[AFINAL_A L0196] Transposition table (public hash + belief signature), neural eval cache (batched GPU, LRU), Sinkhorn warm-start cache (u,v scalings), predictive ponder cache (subtrees for top-M predicted opponent actions).
[AFINAL_A L0197] 
[AFINAL_A L0198] ### 7.4 Incremental reuse across turns
[AFINAL_A L0199] 
[AFINAL_A L0200] On event: lookup predicted child key; if match, shift root and keep statistics; else reuse TT/NN cache and rebuild shallow frontier.
[AFINAL_A L0201] 
[AFINAL_A L0202] ### 7.5 Endgame exactification (wall-small solver)
[AFINAL_A L0203] 
[AFINAL_A L0204] **Trigger:** Activate when remaining wall $\le W^* = 10$ tiles AND at least one threatening signal (riichi, open tenpai, high-tempo opponent).
[AFINAL_A L0205] 
[AFINAL_A L0206] **PIMC with top-k draw pruning.** Full Expectimax over wall=10 is too slow (~661K paths per particle at 0.1ms each = 66s). Instead, use **Pure PIMC**: for each CT-SMC particle, sample ONE draw sequence (weighted by hypergeometric probabilities) and ONE opponent action sequence (from ActorNet policy). Average over P particles. This reduces to P forward passes per endgame evaluation. With top-mass particle reduction (keep particles covering 95% weight, typically P=50-100): **5-10ms per decision**, well within budget. Top-k draw pruning (branch only on the 2-3 most likely draws at our nodes) provides a middle ground between PIMC and full Expectimax when more precision is needed.
[AFINAL_A L0207] 
[AFINAL_A L0208] $$Q(a) \approx \frac{1}{P}\sum_{p=1}^{P} \text{PIMC\_Rollout}(a \mid X^{(p)})$$
[AFINAL_A L0209] 
[AFINAL_A L0210] The inner value is exact over wall draws; opponent actions remain modeled by the robust policy (KL ball). This removes chance uncertainty variance at the most sensitive game phase (oorasu placement swings).
[AFINAL_A L0211] 
[AFINAL_A L0212] **Caching.** Late-game states repeat structurally across particles. Cache by: our hand canonicalization + remaining wall multiset signature (34-count vector) + riichi state + turn index. DP results reused heavily.
[AFINAL_A L0213] 
[AFINAL_A L0214] **Why this matters:** Late-game decisions are disproportionately high-EV. A single wrong fold or push in oorasu can flip placement from 1st to 4th (~90,000 point swing in uma). Exact computation eliminates the approximation error precisely where it's most costly.
[AFINAL_A L0215] 
[AFINAL_A L0216] **Validation gate:** Collect 50K endgame positions (last 10 draws). Compare deal-in rate, win conversion rate, and placement swings between standard AFBS vs endgame-exact mode. Endgame mode must improve all three.
[AFINAL_A L0217] 
[AFINAL_A L0218] ---
[AFINAL_A L0219] 
[AFINAL_A L0220] ## 8. Robust opponent modeling inside search
```

## Artifact 02 — research/design/HYDRA_FINAL.md lines 221-405
Source label: AFINAL_B
Path: research/design/HYDRA_FINAL.md
Use: treat this as evidence, not truth.
```text
[AFINAL_B L0221] 
[AFINAL_B L0222] ### 8.1 Opponent uncertainty set
[AFINAL_B L0223] 
[AFINAL_B L0224] Learned opponent policy $p(a)$. True policy $q(a)$ lies in KL ball: $\mathcal{Q}_\varepsilon(p)=\{q: D_{\mathrm{KL}}(q\|p)\le \varepsilon\}$. $\varepsilon$ calibrated from data as empirical upper quantile of observed KL, bucketed by context.
[AFINAL_B L0225] 
[AFINAL_B L0226] ### 8.2 Robust value at opponent nodes
[AFINAL_B L0227] 
[AFINAL_B L0228] $V_{\text{rob}}=\min_{q\in \mathcal{Q}_\varepsilon(p)} \sum_a q(a) Q(a)$. Solution: $q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)$ for $\tau$ chosen so $D_{\mathrm{KL}}(q_\tau\|p)=\varepsilon$.
[AFINAL_B L0229] 
[AFINAL_B L0230] **Contract.** For any opponent policy $q$ in the KL ball, AFBS's robust backup gives a lower bound on expected value against $q$.
[AFINAL_B L0231] 
[AFINAL_B L0232] ### 8.3 OLSS-style opponent strategy set
[AFINAL_B L0233] 
[AFINAL_B L0234] In addition to continuous KL robustness, maintain $N$ discrete opponent archetypes $\{\sigma_1,\dots,\sigma_N\}$ (e.g., aggressive/defensive/speed/value, $N=4$). At opponent nodes, evaluate:
[AFINAL_B L0235] $$Q(a) = -\tau_{\text{arch}} \log \sum_{i=1}^N w_i \exp(-Q^{\sigma_i}(a)/\tau_{\text{arch}})$$
[AFINAL_B L0236] 
[AFINAL_B L0237] where $w_i$ are archetype weights (uniform $1/N$ initially, updated by posterior over opponent type) and $\tau_{\text{arch}}$ is the archetype soft-min temperature (distinct from Section 8.2's $\tau$ found by binary search).
[AFINAL_B L0238] 
[AFINAL_B L0239] This soft-min over archetypes directly mirrors LuckyJ's OLSS-II approach (Liu et al., ICML 2023) and hardens against "wrong opponent model" -- a dominant failure mode in multiplayer search. Archetypes are trained as lightweight shared-backbone adapters during population training.
[AFINAL_B L0240] 
[AFINAL_B L0241] ---
[AFINAL_B L0242] 
[AFINAL_B L0243] ## 9. Search-as-Feature (SaF)
[AFINAL_B L0244] 
[AFINAL_B L0245] For each legal action $a$, AFBS returns: $\Delta Q(a)$, deal-in risk estimates (Boole/Hunter/robust), epistemic terms (entropy drop), robust stress ($\tau$), uncertainty (variance, ESS).
[AFINAL_B L0246] 
[AFINAL_B L0247] **Logit-residual policy:** $\ell_{\text{final}}(a)=\ell_\theta(a) + \alpha_{\text{SaF}}\cdot g_\psi(f(a))\cdot m(a)$ where $m(a)\in\{0,1\}$ indicates features present. $g_\psi$ is a tiny shared MLP (hidden dim 32-64). **SaF-dropout**: during training, randomly zero $m$ even when features are available ($p_{\text{drop}}=0.3$) to prevent over-reliance. Train $g_\psi$ first via supervised regression on $\delta(a)=\log\pi_{\text{search}}(a)-\log\pi_{\text{base}}(a)$, then switch to joint end-to-end.
[AFINAL_B L0248] 
[AFINAL_B L0249] ---
[AFINAL_B L0250] 
[AFINAL_B L0251] ## 10. ExIt + Pondering as the central training engine
[AFINAL_B L0252] 
[AFINAL_B L0253] ### 10.1 ExIt targets
[AFINAL_B L0254] 
[AFINAL_B L0255] From AFBS: $\pi^{\text{ExIt}}(\cdot\mid I)=\mathrm{Softmax}(Q(I,\cdot)/\tau_{\text{ExIt}})$.
[AFINAL_B L0256] 
[AFINAL_B L0257] ### 10.2 Pondering = label amplification
[AFINAL_B L0258] 
[AFINAL_B L0259] 75% idle time used for: deepening current root search + precomputing searches for predicted near-future states. Every completed search yields additional labeled training examples.
[AFINAL_B L0260] 
[AFINAL_B L0261] ### 10.3 Playout cap randomization
[AFINAL_B L0262] 
[AFINAL_B L0263] More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.
[AFINAL_B L0264] 
[AFINAL_B L0265] ---
[AFINAL_B L0266] 
[AFINAL_B L0267] ## 11. Training pipeline
[AFINAL_B L0268] 
[AFINAL_B L0269] ### Compute budget (2000 GPU hours on 4x RTX 5000 Ada)
[AFINAL_B L0270] 
[AFINAL_B L0271] | Phase | GPU-hrs | Nets trained | Games | Key output |
[AFINAL_B L0272] |-------|--------:|-------------|------:|-----------|
[AFINAL_B L0273] | Phase -1: Benchmarks | 150 | All nets | N/A | Latency/throughput/distill gates |
[AFINAL_B L0274] | Phase 0: BC | 50 | LearnerNet (24-block) | N/A (5-6M expert) | Initialize from human data |
[AFINAL_B L0275] | Phase 1: Oracle guiding | 200 | LearnerNet + oracle critic | ~5M | Oracle-calibrated beliefs/danger |
[AFINAL_B L0276] | Phase 2: DRDA-wrapped ACH | 800 | LearnerNet via ACH+DRDA | ~18M | Game-theoretic base + early ExIt |
[AFINAL_B L0277] | Phase 3: ExIt + Pondering | 800 | LearnerNet (deep AFBS on GPU 3) | ~12M | Deep search ExIt + endgame |
[AFINAL_B L0278] | **Total** | **2000** | | **~35M** | |
[AFINAL_B L0279] 
[AFINAL_B L0280] GPU allocation: GPU 0-1 training (LearnerNet), GPU 2 self-play (ActorNet), GPU 3 pondering (LearnerNet inference for deep AFBS). Distillation: Learner -> Actor continuously (IMPALA-style).
[AFINAL_B L0281] 
[AFINAL_B L0282] ### Phase -1: Hard reality benchmarks (150 GPU hours reserve)
[AFINAL_B L0283] Unlocked BEFORE committing the full budget. Must pass:
[AFINAL_B L0284] - **Latency gate**: AFBS on-turn < 150ms, CT-SMC DP < 1ms, endgame solver < 100ms
[AFINAL_B L0285] - **Throughput gate**: ActorNet self-play > 20 games/sec sustained
[AFINAL_B L0286] - **Distillation gate**: Learner->Actor KL drift < threshold over 100 updates
[AFINAL_B L0287] - **Hyperparameter sweep**: ACH eta, DRDA tau_drda, beam W, depth D, particles P
[AFINAL_B L0288] If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.
[AFINAL_B L0289] 
[AFINAL_B L0290] ### Phase 0: BC warm start (50 GPU hours)
[AFINAL_B L0291] Train LearnerNet (24-block) on 5-6M expert games (Tenhou Houou + Majsoul). 24x augmentation (6 suit perms x 4 seat rotations). All heads supervised. Distill to ActorNet (12-block) at end.
[AFINAL_B L0292] 
[AFINAL_B L0293] ### Phase 1: Oracle-visible supervision (200 GPU hours)
[AFINAL_B L0294] Self-play with full hidden state access. Train oracle critic (zero-sum constraint $\sum_i V_i = 0$) and belief likelihood model. Suphx-style Bernoulli dropout $\gamma_t: 1 \to 0$. Post-oracle stability: LR decay $\times 0.1$ + importance weight rejection when $\gamma_t$ reaches 0.
[AFINAL_B L0295] 
[AFINAL_B L0296] ### Phase 2: DRDA-wrapped ACH self-play (800 GPU hours)
[AFINAL_B L0297] 
[AFINAL_B L0298] **DRDA-wrapped ACH**: ACH is LuckyJ's inner optimizer (+0.4 fan over PPO) but its theory covers only 2-player zero-sum. For 4-player stability, wrap in DRDA's multi-round structure (ICLR 2025). Policy: $\pi_\theta(a|x) = \mathrm{softmax}(\ell_{\text{base}}(x,a) + y_\theta(x,a)/\tau_{\text{drda}})$ where $\ell_{\text{base}}$ is a frozen checkpoint, $y_\theta$ is a trainable residual, and $\tau_{\text{drda}} \in \{2, 4, 8\}$ (tune via Phase -1; target median KL to base in $[0.05, 0.20]$).
[AFINAL_B L0299] 
[AFINAL_B L0300] **Rebase rule (CRITICAL):** Every 25-50 GPU hours: (1) fold residual into base: $\ell_{\text{base}} \leftarrow \ell_{\text{base}} + y_\theta/\tau_{\text{drda}}$, (2) zero $y_\theta$ and reset optimizer moments. This preserves $\pi$ exactly across boundaries and prevents double-counting accumulated regret.
[AFINAL_B L0301] 
[AFINAL_B L0302] ACH update (per-(s,a) sample):
[AFINAL_B L0303] $$L_\pi(s,a) = -c(s,a) \cdot \eta \cdot \frac{y(a|s;\theta)}{\pi_{\text{old}}(a|s)} \cdot A(s,a)$$
[AFINAL_B L0304] 
[AFINAL_B L0305] - $\eta$: global scalar hyperparameter (try $\eta \in \{1,2,3\}$), NOT state-dependent in practice
[AFINAL_B L0306] - $c(s,a) \in \{0,1\}$: per-sample gate zeroing update when ratio exceeds $1\pm\epsilon$ OR centered logit exceeds $\pm l_{\text{th}}$
[AFINAL_B L0307] - Uses **logits** $y(a)$ (not log-probs), centered by $\bar{y}(s)$ and clamped to $[-l_{\text{th}}, l_{\text{th}}]$
[AFINAL_B L0308] - Standard GAE for advantages (per-player $V_i$, $\lambda=0.95$, $\gamma=0.995$)
[AFINAL_B L0309] - **One update epoch per batch** (not PPO's 3-10 epochs)
[AFINAL_B L0310] - Recommended: $\epsilon=0.5$, $l_{\text{th}}=8$, $\beta_{\text{ent}}=5\times10^{-4}$, LR $2.5\times10^{-4}$
[AFINAL_B L0311] 
[AFINAL_B L0312] Oracle critic provides advantages via CTDE: actor conditions on public info only. Normalize advantages per-minibatch for scale stability.
[AFINAL_B L0313] 
[AFINAL_B L0314] **Start cheap ExIt mid-Phase 2**: From ~400 GPU hours, run shallow AFBS (depth 3-4, P=64) on 20% of states. Don't wait for Phase 3 to begin amortizing search into the learner.
[AFINAL_B L0315] 
[AFINAL_B L0316] **Fallback:** If DRDA-wrapped ACH proves unstable, fall back to PPO with entropy 0.05-0.1.
[AFINAL_B L0317] 
[AFINAL_B L0318] ### Phase 2 (continuous): Distill rollout net
[AFINAL_B L0319] 
[AFINAL_B L0320] **RolloutNet** (ActorNet-sized, 12 blocks): LuckyJ's "environmental model" concept. Policy + value for fast AFBS rollouts. Distilled from LearnerNet **continuously** (not every 50h -- confirmed too stale). Same input encoding. Run distillation worker on spare GPU cycles.
[AFINAL_B L0321] 
[AFINAL_B L0322] ### Phase 3: ExIt + AFBS + Pondering (800 GPU hours)
[AFINAL_B L0323] 
[AFINAL_B L0324] LearnerNet runs deep AFBS on GPU 3 for **hard positions only** (top-2 policy gap < 10%, high-risk defense, low particle ESS). ExIt targets distilled into LearnerNet's own training loss (ACH + ExIt + SaF auxiliary regression). ActorNet updated from LearnerNet continuously.
[AFINAL_B L0325] 
[AFINAL_B L0326] ### Population training
[AFINAL_B L0327] League: latest ActorNet, trailing checkpoints, human-style anchors (BC-heavy), adversarial exploiters.
[AFINAL_B L0328] 
[AFINAL_B L0329] ---
[AFINAL_B L0330] 
[AFINAL_B L0331] ## 12. Risk, information, and placement
[AFINAL_B L0332] 
[AFINAL_B L0333] ### 12.1 Distributional value and CVaR
[AFINAL_B L0334] Score pdf/cdf heads. CVaR for "avoid 4th" objectives.
[AFINAL_B L0335] 
[AFINAL_B L0336] ### 12.2 Information-Value Decomposition (IVD)
[AFINAL_B L0337] $Q^{\text{total}}(I,a)=Q^{\text{inst}}(I,a)+\beta_{\text{epi}} Q^{\text{epi}}(I,a)+\xi Q^{\text{str}}(I,a)$ where instrumental = score utility, epistemic = posterior entropy decrease, strategic = concealment/leakage penalty. (Note: $\beta_{\text{epi}}$ is the epistemic weight, distinct from ACH's $\eta$.)
[AFINAL_B L0338] 
[AFINAL_B L0339] ### 12.3 Primal-dual risk constraints
[AFINAL_B L0340] Constraints: deal-in risk below $\kappa_{\text{deal}}$, info leakage below $\kappa_{\text{leak}}$. Dual updates: $\lambda \leftarrow [\lambda+\alpha(\hat{C}-\kappa)]_+$.
[AFINAL_B L0341] 
[AFINAL_B L0342] ---
[AFINAL_B L0343] 
[AFINAL_B L0344] ## 13. Validation gates
[AFINAL_B L0345] 
[AFINAL_B L0346] **G0:** Does Mixture-SIB + particles + AFBS produce positive decision improvement? 200K stratified states, mean $\Delta>0$, <40% negative.
[AFINAL_B L0347] 
[AFINAL_B L0348] **G1:** Robustness calibration. KL deviations between opponent model and held-out opponents at 95th percentile.
[AFINAL_B L0349] 
[AFINAL_B L0350] **G2:** Safety bound usefulness. Hunter reduces over-folding without underestimating risk beyond CI.
[AFINAL_B L0351] 
[AFINAL_B L0352] **G3:** SaF amortization. Shallow search + SaF must dominate shallow search alone.
[AFINAL_B L0353] 
[AFINAL_B L0354] ---
[AFINAL_B L0355] 
[AFINAL_B L0356] ## 14. Deployment profile
[AFINAL_B L0357] 
[AFINAL_B L0358] **Fast path:** Network forward + SaF adaptor. **Slow path:** Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time. Agari guard always active.
[AFINAL_B L0359] 
[AFINAL_B L0360] ---
[AFINAL_B L0361] 
[AFINAL_B L0362] ## 15. Heritage from prior Hydra variants
[AFINAL_B L0363] 
[AFINAL_B L0364] **From the throughput-first plan:** Asynchronous pondering as "free" label compute, distributional value heads, oracle guiding/critic, PPO hyperparameters (entropy coeff 0.05+), double-buffered weight sync, ExIt safety valves.
[AFINAL_B L0365] 
[AFINAL_B L0366] **From the all-out plan:** Mixture-SIB, anytime FBS, SaF, Hunter/Kounias tightening, ExIt+Pondering centrality, SR concentration.
[AFINAL_B L0367] 
[AFINAL_B L0368] **OMEGA additions:** CT-SMC exact contingency-table belief sampler, robust opponent nodes (KL-uncertainty soft-min + OLSS-style archetype set), hand-EV oracle features, endgame exactification, DRDA-wrapped ACH training with explicit rebase rule, 2-tier network (12/24), early ExIt from mid-Phase 2, explicit calibration gates.
[AFINAL_B L0369] 
[AFINAL_B L0370] **Verified ablation data (Suphx Figure 8):** SL baseline ~7.65 dan, +RL basic +0.41, +GRP +0.18, +oracle guiding +0.12. Oracle guiding alone is modest; the stack is what matters.
[AFINAL_B L0371] 
[AFINAL_B L0372] ---
[AFINAL_B L0373] 
[AFINAL_B L0374] ## 16. Limitations
[AFINAL_B L0375] 
[AFINAL_B L0376] 1. **4-player general-sum has no clean exploitability target.** We use robustness + population training instead.
[AFINAL_B L0377] 2. **Belief model misspecification** remains the core risk; G0 detects it early.
[AFINAL_B L0378] 3. **Compute allocation**: deep AFBS is expensive; depends on caching, pondering hit rate, distillation efficiency.
[AFINAL_B L0379] 4. **Strategy fusion / determinization pitfalls**: particles + robust opponent nodes mitigate but do not eliminate all pathologies.
[AFINAL_B L0380] 
[AFINAL_B L0381] ---
[AFINAL_B L0382] 
[AFINAL_B L0383] ## 17. References
[AFINAL_B L0384] 
[AFINAL_B L0385] 1. Sinkhorn, Knopp. "Doubly Stochastic Matrices." *Pacific J. Math*, 1967.
[AFINAL_B L0386] 2. Hunter. "Upper Bound for Union." *J. Applied Probability*, 1976.
[AFINAL_B L0387] 3. Kounias. "Bounds for Union." *Annals Math Stat*, 1968.
[AFINAL_B L0388] 4. Borcea, Branden, Liggett. "SR and Geometry of Polynomials." *JAMS*, 2009.
[AFINAL_B L0389] 5. Bardenet, Maillard. "Concentration for Sampling Without Replacement." *Bernoulli*, 2015.
[AFINAL_B L0390] 6. Anthony, Tian, Barber. "Expert Iteration." *NeurIPS*, 2017.
[AFINAL_B L0391] 7. Silver et al. "Mastering Go Without Human Knowledge." *Nature* 550, 2017.
[AFINAL_B L0392] 8. Wu. "Accelerating Self-Play Learning in Go (KataGo)." *arXiv 1902.10565*, 2020.
[AFINAL_B L0393] 9. Li et al. "Suphx: Mastering Mahjong with Deep RL." *arXiv 2003.13590*, 2020.
[AFINAL_B L0394] 10. Li et al. "Speedup Training via Reward Variance Reduction." *IEEE CoG*, 2022.
[AFINAL_B L0395] 11. Farina et al. "DRDA for Multiplayer POSGs." *ICLR*, 2025.
[AFINAL_B L0396] 12. Rudolph et al. "Reevaluating PG Methods in IIGs." *arXiv 2502.08938*, 2025.
[AFINAL_B L0397] 13. Kalogiannis, Farina. "PG Converge in IIEFGs." *NeurIPS*, 2024.
[AFINAL_B L0398] 14. Schulman et al. "Proximal Policy Optimization." *arXiv 1707.06347*, 2017.
[AFINAL_B L0399] 15. Perolat et al. "Mastering Stratego (DeepNash)." *Science*, 2022.
[AFINAL_B L0400] 16. Boney et al. "Learning to Play IIGs by Imitating an Oracle Planner." *IEEE Trans. Games*, 2021.
[AFINAL_B L0401] 17. Abbasi-Yadkori et al. "POLITEX." *ICML*, 2019.
[AFINAL_B L0402] 18. Cuturi. "Sinkhorn Distances." *NeurIPS*, 2013.
[AFINAL_B L0403] 19. Chen, Diaconis, Holmes, Liu. "Sequential Monte Carlo Methods for Statistical Analysis of Tables." *JASA*, 2005.
[AFINAL_B L0404] 20. Patefield. "Algorithm AS 159: An Efficient Method of Generating R x C Tables with Given Row and Column Totals." *Applied Statistics*, 1981.
[AFINAL_B L0405] 21. Fu et al. "Actor-Critic Hedge for Imperfect-Information Games (ACH)." *ICLR*, 2022.
```

## Artifact 03 — research/design/HYDRA_RECONCILIATION.md lines 1-260
Source label: ARECON_A
Path: research/design/HYDRA_RECONCILIATION.md
Use: treat this as evidence, not truth.
```text
[ARECON_A L0001] # Hydra Reconciliation
[ARECON_A L0002] 
[ARECON_A L0003] > **Current execution doctrine.**
[ARECON_A L0004] >
[ARECON_A L0005] > If any implementation/reference doc conflicts with this file on sequencing, tranche priority, or active-vs-reserve status, this file wins.
[ARECON_A L0006] 
[ARECON_A L0007] This memo reconciles the strongest design inputs, the actual repository state, and the best immediate next move.
[ARECON_A L0008] 
[ARECON_A L0009] It is intentionally opinionated:
[ARECON_A L0010] - keep the strongest old ideas in a reserve shelf so they are not lost
[ARECON_A L0011] - remove weak or distracting ideas from the active path
[ARECON_A L0012] - define the clearest version of Hydra to build right now
[ARECON_A L0013] 
[ARECON_A L0014] Scope:
[ARECON_A L0015] - Target architecture authority: `research/design/HYDRA_FINAL.md`
[ARECON_A L0016] - Verified code reality: `hydra-core/`, `hydra-train/`
[ARECON_A L0017] - Deep-agent inputs: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
[ARECON_A L0018] - High-impact drift only; not a full doc rewrite
[ARECON_A L0019] 
[ARECON_A L0020] ## 1. Executive synthesis
[ARECON_A L0021] 
[ARECON_A L0022] The three answer files point in the same broad direction, but they are useful for different reasons.
[ARECON_A L0023] 
[ARECON_A L0024] - `ANSWER_1.md` is strongest on technical fill-ins:
[ARECON_A L0025]   - unified belief stack
[ARECON_A L0026]   - AFBS node semantics
[ARECON_A L0027]   - factorized threat modeling
[ARECON_A L0028]   - Hand-EV formulas
[ARECON_A L0029]   - endgame trigger ideas
[ARECON_A L0030] - `ANSWER_2.md` is strongest on repo-aware loop closure:
[ARECON_A L0031]   - advanced heads exist but are not fully trained from real targets
[ARECON_A L0032]   - AFBS exists as a shell, not a full information-state search runtime
[ARECON_A L0033]   - Hand-EV and endgame are present but still too shallow
[ARECON_A L0034] - `ANSWER_3.md` is strongest on strategic pruning:
[ARECON_A L0035]   - do not let novelty outrank strength-per-effort
[ARECON_A L0036]   - do not center the roadmap on broad expensive search
[ARECON_A L0037]   - reduce architectural confusion before scaling implementation
[ARECON_A L0038] 
[ARECON_A L0039] Main consensus:
[ARECON_A L0040] - Do not restart Hydra from zero.
[ARECON_A L0041] - The biggest blocker is not missing files; it is partially closed loops plus doc drift.
[ARECON_A L0042] - Stronger target generation is a better immediate lever than a giant search rewrite.
[ARECON_A L0043] - AFBS should be selective and specialist, not the default path everywhere.
[ARECON_A L0044] - Hand-EV is worth moving earlier than deeper AFBS expansion.
[ARECON_A L0045] 
[ARECON_A L0046] Main disagreements:
[ARECON_A L0047] - How aggressively to keep DRDA/ACH on the critical path.
[ARECON_A L0048] - How early opponent-latent and robust-opponent logic should move from helper math into the main runtime.
[ARECON_A L0049] - How much to invest now in search semantics versus cheaper supervision and feature realism.
[ARECON_A L0050] 
[ARECON_A L0051] Best combined reading:
[ARECON_A L0052] - Keep `HYDRA_FINAL.md` as the north star.
[ARECON_A L0053] - Treat current code as a strong baseline with real advanced components already present.
[ARECON_A L0054] - Fix guidance drift first.
[ARECON_A L0055] - Make the first coding tranche close advanced target-generation and supervision loops before spending more engineering on deeper AFBS integration.
[ARECON_A L0056] 
[ARECON_A L0057] Working principle for this memo:
[ARECON_A L0058] - **active path** = what the team should optimize for now
[ARECON_A L0059] - **reserve shelf** = good ideas kept for later if the active path underdelivers
[ARECON_A L0060] - **drop shelf** = ideas that should stop consuming mainline attention for now
[ARECON_A L0061] 
[ARECON_A L0062] ## Doctrine routing
[ARECON_A L0063] 
[ARECON_A L0064] | Need | Primary file |
[ARECON_A L0065] |---|---|
[ARECON_A L0066] | Architecture north star | `HYDRA_FINAL.md` |
[ARECON_A L0067] | Current execution doctrine | `HYDRA_RECONCILIATION.md` |
[ARECON_A L0068] | Current runtime reality | `docs/GAME_ENGINE.md` |
[ARECON_A L0069] | Historical / reserve-only planning surfaces | `HYDRA_ARCHIVE.md` |
[ARECON_A L0070] 
[ARECON_A L0071] ## 2. Verified repo reality
[ARECON_A L0072] 
[ARECON_A L0073] What is confirmed in code today:
[ARECON_A L0074] 
[ARECON_A L0075] - Real advanced modules exist:
[ARECON_A L0076]   - `hydra-core/src/ct_smc.rs`
[ARECON_A L0077]   - `hydra-core/src/sinkhorn.rs`
[ARECON_A L0078]   - `hydra-core/src/afbs.rs`
[ARECON_A L0079]   - `hydra-core/src/robust_opponent.rs`
[ARECON_A L0080]   - `hydra-core/src/hand_ev.rs`
[ARECON_A L0081]   - `hydra-core/src/endgame.rs`
[ARECON_A L0082] - The encoder already moved beyond the old baseline and now exposes a fixed-superset tensor:
[ARECON_A L0083]   - `hydra-core/src/encoder.rs`
[ARECON_A L0084]   - `NUM_CHANNELS = 192`
[ARECON_A L0085] - The train model already includes advanced heads structurally:
[ARECON_A L0086]   - `hydra-train/src/model.rs`
[ARECON_A L0087]   - `hydra-train/src/heads.rs`
[ARECON_A L0088] 
[ARECON_A L0089] What is only partially true:
[ARECON_A L0090] 
[ARECON_A L0091] - Advanced losses exist, but default advanced loss weights are zero:
[ARECON_A L0092]   - `hydra-train/src/training/losses.rs`
[ARECON_A L0093] - Advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets:
[ARECON_A L0094]   - `hydra-train/src/data/sample.rs`
[ARECON_A L0095]   - `hydra-train/src/data/mjai_loader.rs`
[ARECON_A L0096] - AFBS exists as a search shell, but not as a fully integrated public-belief search runtime:
[ARECON_A L0097]   - `hydra-core/src/afbs.rs`
[ARECON_A L0098] - Hand-EV exists, but is still heuristic rather than a full offensive oracle:
[ARECON_A L0099]   - `hydra-core/src/hand_ev.rs`
[ARECON_A L0100] - Endgame exists, but as weighted particle/PIMC evaluation rather than true exactification:
[ARECON_A L0101]   - `hydra-core/src/endgame.rs`
[ARECON_A L0102] 
[ARECON_A L0103] What is outdated, wrong, or overstated in docs:
[ARECON_A L0104] 
[ARECON_A L0105] - `README.md` authority routing has been fixed, but secondary docs still need cleanup discipline to avoid reintroducing stale guidance.
[ARECON_A L0106] - `research/design/HYDRA_SPEC.md` is explicitly outdated but still heavily referenced.
[ARECON_A L0107] - `research/infrastructure/INFRASTRUCTURE.md` still operationalizes old PPO-era assumptions.
[ARECON_A L0108] - `docs/GAME_ENGINE.md` and `hydra-core/README.md` now correctly describe `85x34` only as the baseline prefix, not the full live encoder.
[ARECON_A L0109] 
[ARECON_A L0110] Doc drift that materially affects decisions:
[ARECON_A L0111] 
[ARECON_A L0112] - target architecture says two-tier 12/24-block + ExIt-centered training
[ARECON_A L0113] - secondary reference docs can still imply older 40-block / PPO / dead-TRAINING assumptions if they are read without authority routing
[ARECON_A L0114] - implementation roadmap partially reflects the new world, but supporting docs often do not
[ARECON_A L0115] 
[ARECON_A L0116] ## 3. Ranked next-step recommendations
[ARECON_A L0117] 
[ARECON_A L0118] ### Recommendation 1
[ARECON_A L0119] - recommendation: Close advanced target generation and supervision loops
[ARECON_A L0120] - evidence basis:
[ARECON_A L0121]   - strongest support from `ANSWER_2.md`
[ARECON_A L0122]   - external evidence favors stronger teacher targets over immediate broad search expansion
[ARECON_A L0123] - support from answers: `ANSWER_2.md`, `ANSWER_3.md`, with `ANSWER_1.md` providing useful target semantics
[ARECON_A L0124] - repo verification status:
[ARECON_A L0125]   - model heads exist in `hydra-train/src/model.rs`
[ARECON_A L0126]   - advanced loss support exists in `hydra-train/src/training/losses.rs`
[ARECON_A L0127]   - main data path still underpopulates advanced targets in `hydra-train/src/data/sample.rs`
[ARECON_A L0128] - expected upside: high
[ARECON_A L0129] - difficulty: medium
[ARECON_A L0130] - risk: medium-low
[ARECON_A L0131] - do now / later / drop: do now
[ARECON_A L0132] 
[ARECON_A L0133] ### Recommendation 2
[ARECON_A L0134] - recommendation: Rework Hand-EV realism before deeper AFBS expansion
[ARECON_A L0135] - evidence basis:
[ARECON_A L0136]   - `ANSWER_1.md` and `ANSWER_2.md` both rank this as a cheaper, higher-ROI upgrade than broader search
[ARECON_A L0137]   - external evidence says auxiliary/offensive target generation is a good medium-cost multiplier
[ARECON_A L0138] - support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
[ARECON_A L0139] - repo verification status:
[ARECON_A L0140]   - `hydra-core/src/hand_ev.rs` exists, but expected score and win modeling remain heuristic
[ARECON_A L0141]   - `hydra-core/src/bridge.rs` already threads Hand-EV into encoder paths
[ARECON_A L0142] - expected upside: medium-high
[ARECON_A L0143] - difficulty: medium
[ARECON_A L0144] - risk: low-medium
[ARECON_A L0145] - do now / later / drop: do soon after recommendation 1
[ARECON_A L0146] 
[ARECON_A L0147] ### Recommendation 3
[ARECON_A L0148] - recommendation: Keep AFBS specialist and hard-state gated
[ARECON_A L0149] - evidence basis:
[ARECON_A L0150]   - all three answers warn against broad expensive search
[ARECON_A L0151]   - external evidence supports selective exactification more than universal belief search
[ARECON_A L0152] - support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
[ARECON_A L0153] - repo verification status:
[ARECON_A L0154]   - `hydra-core/src/afbs.rs` is a useful shell but not yet a fully integrated runtime
[ARECON_A L0155]   - `hydra-train/src/inference.rs` already has a fast-path vs ponder-cache split
[ARECON_A L0156] - expected upside: medium-high
[ARECON_A L0157] - difficulty: high
[ARECON_A L0158] - risk: medium-high
[ARECON_A L0159] - do now / later / drop: later, after recommendation 1
[ARECON_A L0160] 
[ARECON_A L0161] ### Recommendation 4
[ARECON_A L0162] - recommendation: Integrate robust opponent logic at search backup/runtime level only after supervision and feature realism improve
[ARECON_A L0163] - evidence basis:
[ARECON_A L0164]   - useful, but downstream of better targets and better local evaluators
[ARECON_A L0165] - support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
[ARECON_A L0166] - repo verification status:
[ARECON_A L0167]   - helper math exists in `hydra-core/src/robust_opponent.rs`
[ARECON_A L0168]   - not yet deeply wired into `hydra-core/src/afbs.rs`
[ARECON_A L0169] - expected upside: medium
[ARECON_A L0170] - difficulty: medium-high
[ARECON_A L0171] - risk: medium
[ARECON_A L0172] - do now / later / drop: later
[ARECON_A L0173] 
[ARECON_A L0174] ### Recommendation 5
[ARECON_A L0175] - recommendation: Do not make full public-belief search the immediate mainline
[ARECON_A L0176] - evidence basis:
[ARECON_A L0177]   - external evidence says it is real but expensive
[ARECON_A L0178]   - repo does not yet have the closed loops needed to justify that jump
[ARECON_A L0179] - support from answers: mostly `ANSWER_3.md`, with `ANSWER_1.md` and `ANSWER_2.md` supporting narrower/search-grade use
[ARECON_A L0180] - repo verification status: not ready for this as the first move
[ARECON_A L0181] - expected upside: potentially high, but too delayed
[ARECON_A L0182] - difficulty: very high
[ARECON_A L0183] - risk: very high
[ARECON_A L0184] - do now / later / drop: later research branch, not now
[ARECON_A L0185] 
[ARECON_A L0186] ## 4. Active Hydra vs reserve shelf vs dropped shelf
[ARECON_A L0187] 
[ARECON_A L0188] ### 4.1 Active Hydra mainline
[ARECON_A L0189] 
[ARECON_A L0190] This is the strongest version of Hydra to code right now.
[ARECON_A L0191] 
[ARECON_A L0192] #### Core identity
[ARECON_A L0193] - two-tier / ExIt-centered target direction from `HYDRA_FINAL.md`
[ARECON_A L0194] - current repo code treated as a partially built advanced baseline, not a restart point
[ARECON_A L0195] - supervision-first before search-expansion-first
[ARECON_A L0196] - AFBS used selectively on hard states, not as the universal default engine
[ARECON_A L0197] 
[ARECON_A L0198] #### What stays on the critical path
[ARECON_A L0199] 1. **Advanced target-generation / supervision loop closure**
[ARECON_A L0200]    - because the heads and losses already exist in code, but are not fully fed or activated
[ARECON_A L0201] 2. **Hand-EV realism improvements**
[ARECON_A L0202]    - because it is already wired into the bridge/encoder path and is cheaper than deeper search work
[ARECON_A L0203] 3. **Selective endgame / AFBS improvement only after supervision is alive end-to-end**
[ARECON_A L0204]    - because deeper search without strong targets risks expensive confusion
[ARECON_A L0205] 4. **Unified belief story**
[ARECON_A L0206]    - Mixture-SIB for amortized belief, CT-SMC for search-grade posterior
[ARECON_A L0207] 
[ARECON_A L0208] #### What this means in plain English
[ARECON_A L0209] Hydra should become:
[ARECON_A L0210] - a strong learned policy/value system
[ARECON_A L0211] - with real advanced auxiliary targets
[ARECON_A L0212] - with better public-belief-quality features
[ARECON_A L0213] - and with selective search layered on top only where it clearly pays
[ARECON_A L0214] 
[ARECON_A L0215] Not:
[ARECON_A L0216] - a giant search project first
[ARECON_A L0217] - a giant theory project first
[ARECON_A L0218] - or a giant “invent ten new heads” project first
[ARECON_A L0219] 
[ARECON_A L0220] ### 4.2 Reserve shelf: good old ideas worth preserving
[ARECON_A L0221] 
[ARECON_A L0222] These ideas should stay documented so the team can come back to them if active Hydra tops out too early.
[ARECON_A L0223] 
[ARECON_A L0224] #### Keep in reserve
[ARECON_A L0225] - **DRDA/ACH as stronger game-theoretic optimizer direction**
[ARECON_A L0226]   - worth preserving because it may become important once the target pipeline is healthy
[ARECON_A L0227] - **Robust-opponent search backups / safe exploitation layers**
[ARECON_A L0228]   - worth preserving because multiplayer Mahjong is exploitable and these may matter at the last strength mile
[ARECON_A L0229] - **Richer latent opponent posterior / more unified opponent modeling**
[ARECON_A L0230]   - worth preserving because the current head surface may eventually want stronger coupling
[ARECON_A L0231] - **Deeper AFBS semantics and hard-state expansion policies**
[ARECON_A L0232]   - worth preserving because `ANSWER_1.md` contains good structure ideas here
[ARECON_A L0233] - **Selective exactification and stronger endgame resolvers**
[ARECON_A L0234]   - worth preserving because late-game precision is plausible high leverage once earlier loops are stable
[ARECON_A L0235] - **Incremental / structured belief-network ideas**
[ARECON_A L0236]   - worth preserving as a research branch, especially if current belief machinery is too slow or too blurry
[ARECON_A L0237] 
[ARECON_A L0238] #### Why these are reserve, not active
[ARECON_A L0239] - they are not obviously wrong
[ARECON_A L0240] - they may matter later
[ARECON_A L0241] - but they add enough complexity that they should not steer the next coding tranche
[ARECON_A L0242] 
[ARECON_A L0243] ### 4.3 Dropped shelf: stop letting these drive current planning
[ARECON_A L0244] 
[ARECON_A L0245] These are not “forbidden forever.” They are just bad uses of mainline attention right now.
[ARECON_A L0246] 
[ARECON_A L0247] For preserved historical context and reserve-only planning surfaces, see `HYDRA_ARCHIVE.md`.
[ARECON_A L0248] 
[ARECON_A L0249] #### Drop from the active path for now
[ARECON_A L0250] - **full public-belief search as the immediate project identity**
[ARECON_A L0251] - **broad “search everywhere” AFBS rollout**
[ARECON_A L0252] - **duplicated belief stacks with overlapping responsibilities**
[ARECON_A L0253] - **adding more output heads before existing advanced heads are properly trained**
[ARECON_A L0254] - **big optimizer-theory detours before target-generation is closed**
[ARECON_A L0255] - **speculative novelty that has weak evidence and no clear repo insertion point**
[ARECON_A L0256] 
[ARECON_A L0257] #### Why these are dropped for now
[ARECON_A L0258] - too compute-heavy
[ARECON_A L0259] - too architecturally confusing
[ARECON_A L0260] - too weakly grounded in the repo's current bottlenecks
```

## Artifact 04 — research/design/HYDRA_RECONCILIATION.md lines 261-532
Source label: ARECON_B
Path: research/design/HYDRA_RECONCILIATION.md
Use: treat this as evidence, not truth.
```text
[ARECON_B L0261] - likely to delay actual strength gains
[ARECON_B L0262] 
[ARECON_B L0263] ## 5. Conflict resolutions
[ARECON_B L0264] 
[ARECON_B L0265] ### Unified belief stack vs duplicated belief machinery
[ARECON_B L0266] Decision:
[ARECON_B L0267] - Use Mixture-SIB as the amortized belief representation and CT-SMC as the search-grade posterior.
[ARECON_B L0268] - Do not create a separate competing belief stack as the next move.
[ARECON_B L0269] 
[ARECON_B L0270] Why:
[ARECON_B L0271] - `HYDRA_FINAL.md` already supports this split.
[ARECON_B L0272] - `hydra-core/src/sinkhorn.rs` and `hydra-core/src/ct_smc.rs` already exist.
[ARECON_B L0273] - Another parallel belief system would increase drift and calibration cost.
[ARECON_B L0274] 
[ARECON_B L0275] ### Hand-EV earlier vs deeper AFBS earlier
[ARECON_B L0276] Decision:
[ARECON_B L0277] - Move Hand-EV realism earlier than deeper AFBS work.
[ARECON_B L0278] 
[ARECON_B L0279] Why:
[ARECON_B L0280] - It is cheaper.
[ARECON_B L0281] - It already has plumbing in `hydra-core/src/bridge.rs` and `hydra-core/src/encoder.rs`.
[ARECON_B L0282] - The current Hand-EV is clearly under-realistic, so there is a high-confidence improvement path.
[ARECON_B L0283] 
[ARECON_B L0284] ### AFBS broad vs AFBS specialist
[ARECON_B L0285] Decision:
[ARECON_B L0286] - AFBS should be specialist / hard-state gated, not broad default runtime.
[ARECON_B L0287] 
[ARECON_B L0288] Why:
[ARECON_B L0289] - Verified code already has fast-path inference and ponder hooks.
[ARECON_B L0290] - External evidence supports selective exactification and planning more than universal expensive search.
[ARECON_B L0291] 
[ARECON_B L0292] ### DRDA/ACH on critical path vs challenger status
[ARECON_B L0293] Decision:
[ARECON_B L0294] - Keep DRDA/ACH as the intended target architecture direction from `HYDRA_FINAL.md`, but do not make immediate implementation decisions depend on resolving every optimizer-level debate first.
[ARECON_B L0295] 
[ARECON_B L0296] Why:
[ARECON_B L0297] - The first coding tranche is about target-generation/supervision closure, which is more robust to this uncertainty.
[ARECON_B L0298] 
[ARECON_B L0299] ### Oracle guidance alignment
[ARECON_B L0300] Decision:
[ARECON_B L0301] - Oracle guidance should teach, not dominate.
[ARECON_B L0302] - Immediate repo move: keep oracle-related supervision connected to the same representation learning plan, but do not expand privileged pathways before the public-target path is closed.
[ARECON_B L0303] 
[ARECON_B L0304] Why:
[ARECON_B L0305] - `hydra-train/src/model.rs` currently detaches the oracle critic path from the shared pooled representation.
[ARECON_B L0306] - That is a repo-verified issue to address deliberately in later coding, not by bolting on more teacher complexity first.
[ARECON_B L0307] 
[ARECON_B L0308] ### Opponent modeling as unified latent posterior vs many disconnected heads
[ARECON_B L0309] Decision:
[ARECON_B L0310] - Long-term direction: more unified.
[ARECON_B L0311] - Immediate move: do not expand head count further; first feed the existing advanced heads with better targets.
[ARECON_B L0312] 
[ARECON_B L0313] Why:
[ARECON_B L0314] - The repo already has enough surface area. The bottleneck is not a lack of outputs.
[ARECON_B L0315] 
[ARECON_B L0316] ### Must-have vs speculative
[ARECON_B L0317] Must-have now:
[ARECON_B L0318] - reconciliation of doc authority
[ARECON_B L0319] - advanced target-generation / supervision loop closure
[ARECON_B L0320] - Hand-EV realism improvements
[ARECON_B L0321] 
[ARECON_B L0322] Strong multipliers later:
[ARECON_B L0323] - AFBS integration improvements
[ARECON_B L0324] - robust-opponent search backups
[ARECON_B L0325] - selective endgame exactification improvements
[ARECON_B L0326] 
[ARECON_B L0327] Speculative / not worth current complexity:
[ARECON_B L0328] - broad public-belief search as immediate mainline
[ARECON_B L0329] - major new latent machinery before existing heads are trained properly
[ARECON_B L0330] 
[ARECON_B L0331] ### Old good parts to explicitly keep available
[ARECON_B L0332] Keep these old ideas documented, but demote them from the current coding path:
[ARECON_B L0333] - optimizer/game-theory upgrades that depend on a healthier training loop
[ARECON_B L0334] - advanced search backup rules and exploitation layers
[ARECON_B L0335] - deeper belief-model experiments
[ARECON_B L0336] - richer ToM / latent-opponent modeling ideas
[ARECON_B L0337] 
[ARECON_B L0338] These should remain as fallback or phase-next material, not be deleted from project memory.
[ARECON_B L0339] 
[ARECON_B L0340] ## 6. Best next action
[ARECON_B L0341] 
[ARECON_B L0342] Best immediate next move for Hydra:
[ARECON_B L0343] 
[ARECON_B L0344] 1. Reconcile the repo around one truthful planning artifact.
[ARECON_B L0345] 2. Use that artifact to pin the first coding tranche as advanced target-generation / supervision loop closure.
[ARECON_B L0346] 
[ARECON_B L0347] Why this beats a direct broad implementation tranche:
[ARECON_B L0348] - The repo already contains a lot of the advanced surfaces.
[ARECON_B L0349] - The current highest-leverage missing piece is that those surfaces are not consistently fed real targets and active losses.
[ARECON_B L0350] - Fixing that gives the project learning signal now, with less architecture risk than a large AFBS rewrite.
[ARECON_B L0351] 
[ARECON_B L0352] First concrete execution tranche:
[ARECON_B L0353] 
[ARECON_B L0354] - training target generation and activation, centered on:
[ARECON_B L0355]   - `hydra-train/src/data/sample.rs`
[ARECON_B L0356]   - `hydra-train/src/data/mjai_loader.rs`
[ARECON_B L0357]   - `hydra-train/src/training/losses.rs`
[ARECON_B L0358]   - `hydra-train/src/training/bc.rs`
[ARECON_B L0359]   - `hydra-train/src/training/rl.rs`
[ARECON_B L0360]   - `hydra-train/src/model.rs`
[ARECON_B L0361] - supporting bridge/search context review in:
[ARECON_B L0362]   - `hydra-core/src/bridge.rs`
[ARECON_B L0363]   - `hydra-core/src/ct_smc.rs`
[ARECON_B L0364]   - `hydra-core/src/afbs.rs`
[ARECON_B L0365] 
[ARECON_B L0366] Exact tranche intent:
[ARECON_B L0367] - populate advanced targets where feasible from existing replay/context machinery
[ARECON_B L0368] - turn on nonzero advanced loss weights in a controlled staged way
[ARECON_B L0369] - keep AFBS deeper integration for the following tranche, not this one
[ARECON_B L0370] 
[ARECON_B L0371] ### First tranche coding spec
[ARECON_B L0372] 
[ARECON_B L0373] The goal is not “make AFBS smarter.”
[ARECON_B L0374] The goal is:
[ARECON_B L0375] - make existing advanced model surfaces receive real targets
[ARECON_B L0376] - make those targets participate in training with nonzero but staged weights
[ARECON_B L0377] - verify the full path from data -> targets -> losses -> train step -> metrics
[ARECON_B L0378] 
[ARECON_B L0379] #### Concrete coding objectives
[ARECON_B L0380] 1. **Audit and populate advanced targets in sample construction**
[ARECON_B L0381]    - confirm where `HydraTargets` fields are still `None`
[ARECON_B L0382]    - populate fields that can already be built from existing replay/search/belief context
[ARECON_B L0383] 2. **Stage loss activation in one place**
[ARECON_B L0384]    - make advanced loss weights move from zero to small nonzero defaults only when their targets exist and are numerically sane
[ARECON_B L0385] 3. **Keep the rollout narrow**
[ARECON_B L0386]    - prefer ExIt target + delta-Q + safety-residual activation first
[ARECON_B L0387]    - bring belief-field / mixture / hand-type targets online only where labels are credible
[ARECON_B L0388]    - if belief supervision is activated, supervise projected/public-teacher belief objects or gauge-fixed marginals, not raw Sinkhorn fields and not realized hidden allocations as direct student targets
[ARECON_B L0389] 4. **Do not expand model surface in this tranche**
[ARECON_B L0390]    - no new heads
[ARECON_B L0391]    - no new broad search engine
[ARECON_B L0392]    - no new optimizer family
[ARECON_B L0393] 
[ARECON_B L0394] #### Success criteria for the first tranche
[ARECON_B L0395] - advanced targets are produced deterministically where expected
[ARECON_B L0396] - losses are nonzero only when targets are present
[ARECON_B L0397] - RL/BC steps consume those targets without NaN or silent skipping
[ARECON_B L0398] - tests cover the new target plumbing explicitly
[ARECON_B L0399] 
[ARECON_B L0400] ### File-by-file implementation checklist
[ARECON_B L0401] 
[ARECON_B L0402] Use this as the concrete coding handoff for the first tranche.
[ARECON_B L0403] 
[ARECON_B L0404] #### `hydra-train/src/data/sample.rs`
[ARECON_B L0405] - **Current state**
[ARECON_B L0406]   - `MjaiSample` only stores baseline targets: policy, GRP, tenpai, danger, opp-next, score targets
[ARECON_B L0407]   - `MjaiBatch` only collates those baseline tensors
[ARECON_B L0408] - **Required changes**
[ARECON_B L0409]   1. decide whether advanced targets should live directly in `MjaiSample` or be introduced as a parallel advanced-target carrier
[ARECON_B L0410]   2. extend batch collation so the advanced target tensors needed by `HydraTargets` can be created deterministically
[ARECON_B L0411]   3. keep augmentation behavior correct for any tile-indexed advanced targets
[ARECON_B L0412] - **Do not do**
[ARECON_B L0413]   - do not invent new model heads here
[ARECON_B L0414]   - do not mix search-only targets into baseline batches unless provenance is explicit
[ARECON_B L0415] 
[ARECON_B L0416] #### `hydra-train/src/data/mjai_loader.rs`
[ARECON_B L0417] - **Current state**
[ARECON_B L0418]   - builds only baseline labels from replay + exact waits + next discard lookahead
[ARECON_B L0419]   - has no production path for `exit_target`, `delta_q_target`, `belief_fields_target`, `mixture_weight_target`, `opponent_hand_type_target`, or `safety_residual_target`
[ARECON_B L0420] - **Required changes**
[ARECON_B L0421]   1. define which advanced targets can be built from replay-only information versus which require search/belief context
[ARECON_B L0422]   2. add a narrow advanced-target builder path for replay-safe labels first
[ARECON_B L0423]   3. leave clearly unavailable targets as absent rather than fabricating weak labels
[ARECON_B L0424]   4. document provenance inline: replay-derived, bridge-derived, or search-derived
[ARECON_B L0425] - **Preferred order**
[ARECON_B L0426]   - first: `safety_residual_target`, `delta_q_target`, and any replay-credible target that can be computed without new search infra
[ARECON_B L0427]   - later in same tranche if credible: `belief_fields_target`, `mixture_weight_target`, `opponent_hand_type_target`
[ARECON_B L0428] 
[ARECON_B L0429] #### `hydra-train/src/training/losses.rs`
[ARECON_B L0430] - **Current state**
[ARECON_B L0431]   - `HydraTargets` already exposes all advanced target slots
[ARECON_B L0432]   - advanced weights default to `0.0`
[ARECON_B L0433] - **Required changes**
[ARECON_B L0434]   1. add a single, clear activation policy for advanced losses
[ARECON_B L0435]   2. ensure each optional target contributes loss only when target data exists
[ARECON_B L0436]   3. ensure breakdowns make missing-target behavior obvious during debugging
[ARECON_B L0437] 4. keep default behavior conservative: no accidental activation without valid labels
[ARECON_B L0438] - **Key rule**
[ARECON_B L0439]   - target presence should control whether an advanced loss exists at all; weight alone should not hide broken plumbing
[ARECON_B L0440]   - future belief supervision must target projected belief objects, not raw field regression
[ARECON_B L0441] 
[ARECON_B L0442] #### `hydra-train/src/training/bc.rs`
[ARECON_B L0443] - **Current state**
[ARECON_B L0444]   - BC already routes through `HydraTargets`
[ARECON_B L0445]   - it can benefit from advanced targets as soon as batches provide them
[ARECON_B L0446] - **Required changes**
[ARECON_B L0447]   1. add tranche-specific tests showing BC consumes advanced targets when present
[ARECON_B L0448]   2. confirm policy-agreement and oracle-guiding paths behave sanely when optional advanced targets are activated
[ARECON_B L0449]   3. make failures obvious if target tensors are shape-inconsistent
[ARECON_B L0450] 
[ARECON_B L0451] #### `hydra-train/src/training/rl.rs`
[ARECON_B L0452] - **Current state**
[ARECON_B L0453]   - `RlBatch` can already carry `targets: HydraTargets` and `exit_target: Option<Tensor<...>>`
[ARECON_B L0454]   - RL only gets ExIt signal if upstream code produces it
[ARECON_B L0455] - **Required changes**
[ARECON_B L0456]   1. make upstream production of `exit_target` part of the tranche, not a future assumption
[ARECON_B L0457]   2. add tests for mixed cases:
[ARECON_B L0458]      - baseline targets only
[ARECON_B L0459]      - baseline + exit
[ARECON_B L0460]      - baseline + selected advanced auxiliary targets
[ARECON_B L0461]   3. verify staged exit/aux weighting remains numerically stable
[ARECON_B L0462] 
[ARECON_B L0463] #### `hydra-train/src/model.rs`
[ARECON_B L0464] - **Current state**
[ARECON_B L0465]   - model already exposes advanced surfaces: `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`
[ARECON_B L0466]   - oracle path is still detached from pooled representation
[ARECON_B L0467] - **Required changes in this tranche**
[ARECON_B L0468]   1. no new heads
[ARECON_B L0469]   2. no architectural expansion
[ARECON_B L0470]   3. use existing output surface exactly as-is for target/loss closure work
[ARECON_B L0471] - **Deferred explicitly**
[ARECON_B L0472]   - oracle-path detachment review is later, after supervision plumbing is healthier
[ARECON_B L0473] 
[ARECON_B L0474] #### `hydra-core/src/bridge.rs`
[ARECON_B L0475] - **Current state**
[ARECON_B L0476]   - already builds search/belief feature planes from `MixtureSib`, `CtSmc`, and `AfbsTree`
[ARECON_B L0477]   - already threads CT-SMC-weighted Hand-EV into encoder-side features
[ARECON_B L0478] - **Required changes**
[ARECON_B L0479]   1. identify which bridge-side signals can be promoted into training labels without inventing new semantics
[ARECON_B L0480]   2. define a clean mapping from bridge/search features to train-side targets where credible
[ARECON_B L0481]   3. avoid coupling replay-only loading to runtime-only search context unless there is an explicit offline generation path
[ARECON_B L0482] 
[ARECON_B L0483] #### `hydra-core/src/afbs.rs`
[ARECON_B L0484] - **Current state**
[ARECON_B L0485]   - search shell exists with root exit policy, visit counts, priors, Q summaries
[ARECON_B L0486] - **Required changes in this tranche**
[ARECON_B L0487]   1. do not broaden AFBS itself yet
[ARECON_B L0488]   2. only expose the minimum target-generation outputs needed for ExIt / delta-Q supervision
[ARECON_B L0489]   3. keep hard-state-gated philosophy intact
[ARECON_B L0490] 
[ARECON_B L0491] #### `hydra-core/src/ct_smc.rs`
[ARECON_B L0492] - **Current state**
[ARECON_B L0493]   - exact contingency-table belief sampler already exists
[ARECON_B L0494] - **Required changes in this tranche**
[ARECON_B L0495]   1. treat CT-SMC as a source of credible belief-weighted features/targets, not as a new search project
[ARECON_B L0496]   2. use it where it improves label quality for belief-related supervision
[ARECON_B L0497]   3. avoid turning this tranche into a sampler redesign
[ARECON_B L0498] 
[ARECON_B L0499] ### Suggested execution order inside the first tranche
[ARECON_B L0500] 1. `losses.rs`: make target-presence/activation behavior explicit
[ARECON_B L0501] 2. `sample.rs`: extend data containers/collation for advanced targets
[ARECON_B L0502] 3. `mjai_loader.rs`: add replay-credible advanced target generation
[ARECON_B L0503] 4. `rl.rs` and `bc.rs`: prove train-step consumption with tests
[ARECON_B L0504] 5. `bridge.rs` / `afbs.rs` / `ct_smc.rs`: only add the minimum plumbing needed for ExIt / belief-grade labels
[ARECON_B L0505] 
[ARECON_B L0506] ### Minimal tranche acceptance checklist
[ARECON_B L0507] - `MjaiSample` / batch collation can carry the tranche-selected advanced targets
[ARECON_B L0508] - `HydraTargets` fields used in the tranche are populated by real code paths, not left as always-`None`
[ARECON_B L0509] - at least one train path produces nonzero advanced auxiliary loss contributions in tests
[ARECON_B L0510] - `exit_target` is produced by a real upstream path, not just by unit-test fixtures
[ARECON_B L0511] - no new heads, no broad AFBS rewrite, no duplicated belief stack
[ARECON_B L0512] 
[ARECON_B L0513] ## 7. Final handoff / progress report
[ARECON_B L0514] 
[ARECON_B L0515] What I concluded:
[ARECON_B L0516] - Hydra should not restart from zero.
[ARECON_B L0517] - The best immediate next move is reconciliation plus a supervision-first first coding tranche.
[ARECON_B L0518] - The repo is closer to “advanced baseline with partially inactive loops” than to “missing everything.”
[ARECON_B L0519] 
[ARECON_B L0520] What changed:
[ARECON_B L0521] - Added this reconciliation memo.
[ARECON_B L0522] - Next step also updates the repo root README so it no longer routes readers into stale or missing guidance first.
[ARECON_B L0523] 
[ARECON_B L0524] What was verified:
[ARECON_B L0525] - doc drift across `HYDRA_FINAL.md`, `HYDRA_SPEC.md`, `INFRASTRUCTURE.md`, `README.md`, and runtime docs
[ARECON_B L0526] - code reality in `hydra-core` and `hydra-train`
[ARECON_B L0527] - advanced heads/losses present but partially inactive
[ARECON_B L0528] - AFBS, Hand-EV, endgame, and robust-opponent modules present but not fully integrated end-to-end
[ARECON_B L0529] 
[ARECON_B L0530] What remains next:
[ARECON_B L0531] - implement the first coding tranche described above
[ARECON_B L0532] - then reassess Hand-EV realism and selective AFBS improvements against that stronger supervision base
```

## Artifact 05 — docs/GAME_ENGINE.md lines 1-220
Source label: AGAME
Path: docs/GAME_ENGINE.md
Use: treat this as evidence, not truth.
```text
[AGAME L0001] # Hydra Game Engine (hydra-core)
[AGAME L0002] 
[AGAME L0003] Reference documentation for the `hydra-core` Rust crate, the game engine powering the Hydra Riichi Mahjong AI.
[AGAME L0004] 
[AGAME L0005] ## Overview
[AGAME L0006] 
[AGAME L0007] `hydra-core` is a Rust library that provides everything the Hydra training pipeline and runtime need from the game side: a complete Riichi Mahjong simulator, observation encoding, safety analysis, search/belief feature bridging, and batch execution. It wraps `riichienv-core` as the underlying game engine and layers Hydra-specific encoding, seeding, and orchestration on top.
[AGAME L0008] 
[AGAME L0009] Core responsibilities:
[AGAME L0010] 
[AGAME L0011] - Tile representation and suit permutation for data augmentation
[AGAME L0012] - A 46-action space with bidirectional conversion to/from `riichienv` actions
[AGAME L0013] - A currently implemented **192-channel x 34-tile fixed-superset observation encoder**, whose first 85 channels preserve the original public+safety baseline while Groups C/D add search/belief and Hand-EV planes
[AGAME L0014] - Tile safety analysis (genbutsu, suji, kabe, one-chance)
[AGAME L0015] - Deterministic seeding via SHA-256 KDF + ChaCha8Rng
[AGAME L0016] - Parallel batch simulation with `rayon`
[AGAME L0017] - A game loop abstraction with pluggable action selection
[AGAME L0018] 
[AGAME L0019] Hydra uses a 100% Rust stack (see `research/RUST_STACK.md`). The training pipeline (hydra-train, using Burn framework) consumes hydra-core directly -- same process, same memory, zero IPC.
[AGAME L0020] 
[AGAME L0021] ## Foundation: RiichiEnv
[AGAME L0022] 
[AGAME L0023] The game engine is built on top of [smly/RiichiEnv](https://github.com/smly/RiichiEnv) (`riichienv-core` crate, Apache-2.0 license).
[AGAME L0024] 
[AGAME L0025] RiichiEnv provides:
[AGAME L0026] 
[AGAME L0027] - Full 4-player and 3-player Riichi Mahjong rules
[AGAME L0028] - Red dora (aka-dora) support for all three suits
[AGAME L0029] - All kan types: ankan (closed), daiminkan (open), shouminkan (added)
[AGAME L0030] - Native MJAI protocol compatibility for game state representation
[AGAME L0031] - Correctness verified by running MortalAgent (AGPL, used as a black-box MJAI player -- no code shared) over 1M+ hanchan without errors ([source: RiichiEnv README](https://github.com/smly/RiichiEnv#-features))
[AGAME L0032] 
[AGAME L0033] Hydra treats `riichienv-core` as a black-box game engine. All game state progression, legality checks, and rule enforcement happen inside RiichiEnv. Hydra's own code handles encoding, analysis, and orchestration only.
[AGAME L0034] 
[AGAME L0035] Because riichienv-core's correctness is already verified upstream -- smly ran Mortal as a black-box MJAI player (separate process, no linking) over 1M+ hanchan on RiichiEnv with zero errors ([source](https://github.com/smly/RiichiEnv)) -- Hydra does not need its own cross-engine validation. The correctness guarantee is inherited through the dependency. No Mortal code exists in RiichiEnv or Hydra.
[AGAME L0036] 
[AGAME L0037] ## Module Reference
[AGAME L0038] 
[AGAME L0039] | Module | File | Description |
[AGAME L0040] |--------|------|-------------|
[AGAME L0041] | `tile` | `tile.rs` | Tile types (0-33), 136-format representation, aka-dora handling, suit permutation |
[AGAME L0042] | `action` | `action.rs` | 46-action space, `HydraAction` enum, bidirectional riichienv conversion, legal mask builder |
[AGAME L0043] | `encoder` | `encoder.rs` | 192x34 fixed-superset observation tensor, `ObservationEncoder`, incremental encoding with `DirtyFlags` |
[AGAME L0044] | `safety` | `safety.rs` | `SafetyInfo` per-opponent tile safety: genbutsu, suji, kabe, one-chance |
[AGAME L0045] | `simulator` | `simulator.rs` | `BatchSimulator` with rayon thread pool, `BatchConfig`, `GameResult` collection |
[AGAME L0046] | `seeding` | `seeding.rs` | SHA-256 KDF, `SessionRng`, deterministic wall generation, Fisher-Yates shuffle |
[AGAME L0047] | `bridge` | `bridge.rs` | Converts riichienv `Observation` into encoder-ready data via `extract_*` functions |
[AGAME L0048] | `game_loop` | `game_loop.rs` | `GameRunner`, `ActionSelector` trait, step-by-step or run-to-completion execution |
[AGAME L0049] | `batch_encoder` | `batch_encoder.rs` | Pre-allocated contiguous buffer for encoding N observations without per-obs allocation |
[AGAME L0050] | `shanten_batch` | `shanten_batch.rs` | Batch shanten with hierarchical hash caching (base + all 34 discards in one pass) |
[AGAME L0051] 
[AGAME L0052] 
[AGAME L0053] ## Tile System (`tile.rs`)
[AGAME L0054] 
[AGAME L0055] ### TileType
[AGAME L0056] 
[AGAME L0057] All tiles use a `TileType(u8)` newtype representing the 34 distinct Mahjong tile kinds:
[AGAME L0058] 
[AGAME L0059] | Range | Tiles | Count |
[AGAME L0060] |-------|-------|-------|
[AGAME L0061] | 0-8 | 1m through 9m (manzu/characters) | 9 |
[AGAME L0062] | 9-17 | 1p through 9p (pinzu/circles) | 9 |
[AGAME L0063] | 18-26 | 1s through 9s (souzu/bamboo) | 9 |
[AGAME L0064] | 27-33 | East, South, West, North, Haku, Hatsu, Chun | 7 |
[AGAME L0065] 
[AGAME L0066] The physical game uses 136 tiles (4 copies of each type). The 136-format index identifies a specific physical tile, while `TileType` identifies its kind. Converting between them is a simple `tile136 / 4` truncation.
[AGAME L0067] 
[AGAME L0068] ### Aka-Dora (Red Fives)
[AGAME L0069] 
[AGAME L0070] Three tiles in the 136-format set are designated red dora (aka-dora):
[AGAME L0071] 
[AGAME L0072] - Red 5m (manzu)
[AGAME L0073] - Red 5p (pinzu)
[AGAME L0074] - Red 5s (souzu)
[AGAME L0075] 
[AGAME L0076] These are the 0th copy (index 0 within each group of 4) of the respective 5-tiles: 136-format indices 16 (5m), 52 (5p), 88 (5s). Extended tile type indices 34-36 represent aka variants in the action space. The encoder and action space both handle aka-dora as distinct from regular fives where needed.
[AGAME L0077] 
[AGAME L0078] ### Suit Permutation
[AGAME L0079] 
[AGAME L0080] For data augmentation during training, `tile.rs` provides suit permutation functions. There are 6 permutations of the three numbered suits (manzu, pinzu, souzu), leaving honor tiles untouched. Given a permutation index (0-5), the module remaps all tile types in a hand/observation to the permuted suit assignment. This 6x data augmentation helps the model learn suit-invariant patterns.
[AGAME L0081] 
[AGAME L0082] ## Action Space (`action.rs`)
[AGAME L0083] 
[AGAME L0084] ### 46-Action Space
[AGAME L0085] 
[AGAME L0086] Hydra uses a fixed 46-action output space. Every decision point in the game maps to one of these action indices:
[AGAME L0087] 
[AGAME L0088] | Index | Action | Notes |
[AGAME L0089] |-------|--------|-------|
[AGAME L0090] | 0-33 | Discard tile type 0-33 | Standard discard (non-red) |
[AGAME L0091] | 34-36 | Discard aka 5m, 5p, 5s | Discard a specific red five |
[AGAME L0092] | 37 | Declare riichi | Announces riichi; tile selection follows |
[AGAME L0093] | 38-40 | Chi (3 variants) | Left/middle/right chi calls |
[AGAME L0094] | 41 | Pon | Open pon call |
[AGAME L0095] | 42 | Kan | Any kan type (ankan, daiminkan, shouminkan) |
[AGAME L0096] | 43 | Agari | Win declaration (tsumo or ron) |
[AGAME L0097] | 44 | Ryuukyoku | Abortive draw declaration (kyuushu kyuuhai, etc.) |
[AGAME L0098] | 45 | Pass | Decline a call opportunity |
[AGAME L0099] 
[AGAME L0100] ### Two-Phase Actions
[AGAME L0101] 
[AGAME L0102] Riichi and kan use a two-phase selection process. The model first outputs a phase-1 action (index 37 for riichi, 42 for kan). Then the game engine presents the legal tile choices and the model picks which specific tile to discard (riichi) or which specific kan to declare. This keeps the action space compact at 46 while supporting the full combinatorial range.
[AGAME L0103] 
[AGAME L0104] ### HydraAction
[AGAME L0105] 
[AGAME L0106] `HydraAction` is a validated newtype wrapper around `u8`:
[AGAME L0107] 
[AGAME L0108] ```rust
[AGAME L0109] pub struct HydraAction(u8);
[AGAME L0110] ```
[AGAME L0111] 
[AGAME L0112] It validates the index is in range 0-45 on construction via `HydraAction::new(id) -> Option<Self>`. Methods like `is_discard()`, `is_aka_discard()`, and `discard_tile_type()` provide type-safe access. Bidirectional conversion functions `hydra_to_riichienv()` and `riichienv_to_hydra()` translate between Hydra's compact action space and riichienv-core's `Action` struct, using a `GameContext` to resolve context-dependent actions (chi consume tiles, tsumo vs ron, kan type).
[AGAME L0113] 
[AGAME L0114] ### Legal Action Mask
[AGAME L0115] 
[AGAME L0116] The `build_legal_action_mask` function takes the current riichienv game state and returns a `[bool; 46]` array. Each slot is `true` if that action is legal in the current state. The training pipeline uses this mask to zero out illegal actions before softmax, guaranteeing the model never selects an impossible move.
[AGAME L0117] 
[AGAME L0118] ## Observation Encoder (`encoder.rs`)
[AGAME L0119] 
[AGAME L0120] ### Tensor Shape
[AGAME L0121] 
[AGAME L0122] **Canonical SSOT note:** `research/design/HYDRA_FINAL.md` is the governing architecture doc, and `research/design/HYDRA_RECONCILIATION.md` is the current repo-wide decision memo. The original `85 x 34` tensor now describes the **baseline prefix** of the live encoder, not the full live encoder. The current implementation is already a **fixed-shape 192 x 34 superset** with Groups C/D plus presence-mask channels.
[AGAME L0123] 
[AGAME L0124] Each observation is a `192 x 34` float tensor (6,528 values). The first 85 channels retain the baseline public+safety encoding; the remaining channels provide fixed-shape search/belief and Hand-EV context with zero-fill plus explicit presence masks when dynamic features are unavailable. This full shape feeds directly into the current SE-ResNet model input.
[AGAME L0125] 
[AGAME L0126] ### Baseline Prefix Channel Layout (channels 0-84)
[AGAME L0127] 
[AGAME L0128] The 85 channels break down into these groups:
[AGAME L0129] 
[AGAME L0130] | Channels | Name | Encoding |
[AGAME L0131] |----------|------|----------|
[AGAME L0132] | 0-3 | Closed hand | Thresholded: ch N is 1.0 if tile count >= N+1 |
[AGAME L0133] | 4-7 | Open meld hand | Same thresholding for tiles exposed in open melds |
[AGAME L0134] | 8 | Drawn tile | One-hot: 1.0 at the tile type just drawn (tsumo only) |
[AGAME L0135] | 9-10 | Shanten masks | Ch 9: keep-shanten (tiles whose discard does not increase shanten). Ch 10: next-shanten (tiles whose discard decreases shanten) |
[AGAME L0136] | 11-13 | Player 0 discards | Presence (1.0 if discarded), tedashi flag (1.0 if from hand, not tsumogiri), temporal weight (exp(-0.2 * age)) |
[AGAME L0137] | 14-16 | Player 1 discards | Same three channels, relative to seat |
[AGAME L0138] | 17-19 | Player 2 discards | Same three channels, relative to seat |
[AGAME L0139] | 20-22 | Player 3 discards | Same three channels, relative to seat |
[AGAME L0140] | 23-25 | Player 0 melds | Chi (1.0 for tiles in chi melds), pon (tiles in pon), kan (tiles in kan) |
[AGAME L0141] | 26-28 | Player 1 melds | Same three channels |
[AGAME L0142] | 29-31 | Player 2 melds | Same three channels |
[AGAME L0143] | 32-34 | Player 3 melds | Same three channels |
[AGAME L0144] | 35-39 | Dora indicators | Thermometer encoding: ch N is 1.0 if N+1 or more dora indicators revealed |
[AGAME L0145] | 40-42 | Aka dora flags | Per-suit plane: ch 40 = manzu red five, ch 41 = pinzu, ch 42 = souzu. 1.0 at the 5-tile column if that red five is visible |
[AGAME L0146] | 43-46 | Riichi flags | One channel per player. Entire plane is 1.0 if that player has declared riichi |
[AGAME L0147] | 47-50 | Scores | One channel per player. Entire plane filled with score / 100,000 |
[AGAME L0148] | 51-54 | Relative score gaps | One channel per player. Filled with (player_score - my_score) / 30,000 |
[AGAME L0149] | 55-58 | Shanten one-hot | Ch 55 = tenpai (shanten 0), ch 56 = iishanten (1), ch 57 = ryanshanten (2), ch 58 = 3+ shanten. Entire plane is 1.0 for the matching shanten count |
[AGAME L0150] | 59 | Round number | Entire plane filled with kyoku / 8.0 (normalized round index) |
[AGAME L0151] | 60 | Honba count | Entire plane filled with honba / 10.0 |
[AGAME L0152] | 61 | Kyotaku (riichi sticks) | Entire plane filled with kyotaku / 10.0 |
[AGAME L0153] | 62-84 | Safety channels | 23 channels of per-opponent tile safety data (see Safety System section) |
[AGAME L0154] 
[AGAME L0155] **Safety channel breakdown (channels 62-84):**
[AGAME L0156] 
[AGAME L0157] | Channels | Name |
[AGAME L0158] |----------|------|
[AGAME L0159] | 62-64 | Genbutsu (all): 1.0 for tiles each opponent discarded (one ch per opponent) |
[AGAME L0160] | 65-67 | Genbutsu (tedashi): restricted to tiles discarded from hand (not tsumogiri) |
[AGAME L0161] | 68-70 | Genbutsu (riichi-era): restricted to tiles discarded after opponent's riichi |
[AGAME L0162] | 71-73 | Suji: float 0.0-1.0 for suji-inferred safety against each opponent |
[AGAME L0163] | 74-79 | Reserved suji context (zeros) |
[AGAME L0164] | 80 | Kabe: 1.0 for tiles with all 4 copies visible (global, not per-opponent) |
[AGAME L0165] | 81 | One-chance: 1.0 for tiles where exactly 3 of 4 copies are visible |
[AGAME L0166] | 82-84 | Tenpai hints | Opponent tenpai hints (implemented baseline use: riichi or cached tenpai prediction threshold) |
[AGAME L0167] 
[AGAME L0168] ### ObservationEncoder
[AGAME L0169] 
[AGAME L0170] `ObservationEncoder` is the main struct for building observation tensors. In the current implementation it holds a pre-allocated `[f32; 192 * 34]` buffer marked `#[repr(C)]` for predictable memory layout. The baseline public+safety channels remain intact in the first 85 planes; Groups C/D are already present as fixed-shape extensions.
[AGAME L0171] 
[AGAME L0172] ```rust
[AGAME L0173] #[repr(C)]
[AGAME L0174]     pub struct ObservationEncoder {
[AGAME L0175]     buffer: [f32; 6528],  // 192 channels x 34 tiles, row-major
[AGAME L0176] }
[AGAME L0177] ```
[AGAME L0178] 
[AGAME L0179] ### Incremental Encoding with DirtyFlags
[AGAME L0180] 
[AGAME L0181] `DirtyFlags` is a bitflags struct where each bit corresponds to a channel group (hand, discards, melds, dora, scores, safety, etc.). When the game state changes, only the relevant flags are set. On the next `encode()` call, only flagged channel groups are recomputed. Unchanged channels keep their previous values in the buffer.
[AGAME L0182] 
[AGAME L0183] This matters for performance: a single discard only dirties the discard and safety channels, skipping the more expensive hand/meld/dora re-encoding. During batch simulation of thousands of games, these savings compound.
[AGAME L0184] 
[AGAME L0185] ## Safety System (`safety.rs`)
[AGAME L0186] 
[AGAME L0187] The safety module computes per-opponent, per-tile safety information used to populate encoder channels 62-84 and to inform defensive play decisions.
[AGAME L0188] 
[AGAME L0189] ### SafetyInfo
[AGAME L0190] 
[AGAME L0191] `SafetyInfo` holds safety data from one player's perspective against all 3 opponents:
[AGAME L0192] 
[AGAME L0193] ```rust
[AGAME L0194] #[repr(C)]
[AGAME L0195] pub struct SafetyInfo {
[AGAME L0196]     pub genbutsu_all: [[bool; 34]; 3],       // per-opponent
[AGAME L0197]     pub genbutsu_tedashi: [[bool; 34]; 3],   // per-opponent
[AGAME L0198]     pub genbutsu_riichi_era: [[bool; 34]; 3], // per-opponent
[AGAME L0199]     pub suji: [[f32; 34]; 3],                // per-opponent, float 0.0-1.0
[AGAME L0200]     pub kabe: [bool; 34],                     // global
[AGAME L0201]     pub one_chance: [bool; 34],               // global
[AGAME L0202]     pub visible_counts: [u8; 34],             // global tile visibility
[AGAME L0203]     pub opponent_riichi: [bool; 3],           // per-opponent riichi status
[AGAME L0204] }
[AGAME L0205] ```
[AGAME L0206] 
[AGAME L0207] **Genbutsu** (safe tiles) tracks tiles that a specific opponent cannot ron:
[AGAME L0208] 
[AGAME L0209] - `genbutsu_all`: any tile the opponent discarded (always safe against that player)
[AGAME L0210] - `genbutsu_tedashi`: only tiles discarded from the opponent's hand (not tsumogiri), indicating intentional discards
[AGAME L0211] - `genbutsu_riichi_era`: only tiles discarded after the opponent declared riichi, relevant for reading post-riichi waits
[AGAME L0212] 
[AGAME L0213] **Suji** inference identifies tiles protected by the 1-4-7 / 2-5-8 / 3-6-9 suji relationship. If an opponent discarded a 4m, then 1m and 7m get suji safety (float 1.0) against that opponent. Suji only applies to suited tiles (indices 0-26); honors have no suji. Values update incrementally as new discards appear.
[AGAME L0214] 
[AGAME L0215] **Kabe** (wall block) marks tiles where all 4 copies are accounted for in visible information (discards, melds, own hand). A tile with all copies visible can't be part of any opponent's winning hand.
[AGAME L0216] 
[AGAME L0217] **One-chance** marks tiles where exactly 3 of 4 copies are visible, meaning only one unknown copy remains. These tiles carry reduced but nonzero risk.
[AGAME L0218] 
[AGAME L0219] All safety arrays update incrementally. When a new discard or meld occurs, only the affected opponent's `SafetyInfo` is recomputed.
[AGAME L0220] 
```

## Artifact 06 — research/infrastructure/RUST_STACK.md lines 1-220
Source label: ARUST
Path: research/infrastructure/RUST_STACK.md
Use: treat this as evidence, not truth.
```text
[ARUST L0001] # RUST_STACK.md -- 100% Rust Training Stack Decision
[ARUST L0002] 
[ARUST L0003] > Decision Record: Hydra will use a 100% Rust stack for training, inference,
[ARUST L0004] > and self-play. No Python dependency at any point in the pipeline.
[ARUST L0005] >
[ARUST L0006] > **Status note:** this is primarily a decision-record / rationale doc, not the current implementation SSOT. Current Hydra truth is routed by `README.md` -> `research/design/HYDRA_FINAL.md` -> `research/design/HYDRA_RECONCILIATION.md` -> `docs/GAME_ENGINE.md`.
[ARUST L0007] >
[ARUST L0008] > Keep the Rust-stack rationale here. Treat older `85x34`, monolithic-40-block, and PPO-loop-first language as legacy planning unless the active doctrine explicitly promotes it.
[ARUST L0009] 
[ARUST L0010] ## 1. Executive Summary
[ARUST L0011] 
[ARUST L0012] Hydra adopts **Burn** (tracel-ai/burn) as its deep learning framework with
[ARUST L0013] the **`burn-tch` backend** (libtorch/cuDNN) for production training, and
[ARUST L0014] **`burn-cuda`** (CubeCL JIT) as a future upgrade path.
[ARUST L0015] 
[ARUST L0016] This eliminates Python entirely. The game engine (riichienv-core), observation
[ARUST L0017] encoder (hydra-core), training loop, self-play arena, and inference all run
[ARUST L0018] in a single Rust binary with zero IPC, zero GIL, and zero interpreter overhead.
[ARUST L0019] 
[ARUST L0020] ### Why 100% Rust
[ARUST L0021] 
[ARUST L0022] - **Same GPU performance**: `burn-tch` wraps libtorch, which calls the same
[ARUST L0023]   cuDNN/cuBLAS CUDA kernels as Python PyTorch. Identical GPU compute.
[ARUST L0024] - **3.5-4x less CPU overhead**: Benchmarked: C++ LibTorch trains ResNet18
[ARUST L0025]   3.56x faster than Python PyTorch (same model, same GPU).
[ARUST L0026]   Python's `torch.compile` exists to claw back this overhead. Rust never has it.
[ARUST L0027] - **Self-play integration**: any future self-play stage can run in the same Rust process as simulation and training, without Python IPC overhead.
[ARUST L0028]   In Python, this requires subprocess/IPC. In Rust, same process, same memory.
[ARUST L0029] - **Single binary**: No pip, no conda, no virtualenv, no dependency hell.
[ARUST L0030]   `cargo build --release` produces one artifact.
[ARUST L0031] 
[ARUST L0032] ### Why Burn over raw tch-rs
[ARUST L0033] 
[ARUST L0034] - Built-in training infrastructure (Learner, DataLoader, metrics, checkpointing)
[ARUST L0035] - Built-in DDP with NCCL for multi-GPU
[ARUST L0036] - Built-in LR schedulers (cosine annealing, linear warmup, exponential, noam, step, composed)
[ARUST L0037] - Built-in gradient clipping (by value and by norm)
[ARUST L0038] - Backend-generic code: swap `burn-tch` to `burn-cuda` by changing one type parameter
[ARUST L0039] - CubeCL JIT fusion as future upgrade (Burn's answer to torch.compile)
[ARUST L0040] 
[ARUST L0041] ## 2. Framework Comparison
[ARUST L0042] 
[ARUST L0043] ### Rust ML Frameworks Evaluated
[ARUST L0044] 
[ARUST L0045] | Framework | Stars | Training | GPU | Autograd | cuDNN | JIT Fusion | Verdict |
[ARUST L0046] |---|---|---|---|---|---|---|---|
[ARUST L0047] | **Burn** | 9.5k | Full | CUDA/WGPU/Metal | Native autodiff | Via burn-tch | CubeCL | **Selected** |
[ARUST L0048] | tch-rs | 4.4k | Full | CUDA | libtorch autograd | Yes (libtorch) | No | Backend only |
[ARUST L0049] | Candle | 16k | Basic | CUDA/Metal | Basic | No | No | Inference-focused |
[ARUST L0050] | Linfa | -- | No | No | No | No | No | Classical ML only |
[ARUST L0051] 
[ARUST L0052] ### Why Burn Wins
[ARUST L0053] 
[ARUST L0054] 1. **Backend abstraction**: Same model code runs on `burn-tch` (cuDNN) or `burn-cuda` (CubeCL).
[ARUST L0055]    Swap one generic parameter, zero code changes.
[ARUST L0056] 2. **Training infrastructure**: `burn-train` crate provides Learner, DataLoader (multi-threaded),
[ARUST L0057]    metric tracking, and checkpointing out of the box.
[ARUST L0058] 3. **DDP built-in**: `burn-collective` with NCCL for CUDA, AllReduce/AllGather/Broadcast.
[ARUST L0059]    Multi-node via WebSocket. Feature flag: `collective`.
[ARUST L0060] 4. **CubeCL JIT fusion**: Burn's answer to torch.compile. Serializes tensor ops into symbolic
[ARUST L0061]    graph, fuses elementwise ops, auto-tunes kernels for hardware. Works for training + inference.
[ARUST L0062] 5. **All required layers exist**: GroupNorm, Mish, AdaptiveAvgPool2d, Conv2d, Linear, residual
[ARUST L0063]    connections, SE blocks (compose from primitives). Verified in source.
[ARUST L0064] 
[ARUST L0065] ### burn-tch Backend (Production Config)
[ARUST L0066] 
[ARUST L0067] The `burn-tch` backend wraps tch-rs which wraps libtorch. This gives us:
[ARUST L0068] - cuDNN conv2d (same algorithms as Python PyTorch)
[ARUST L0069] - cuBLAS matmul (identical performance)
[ARUST L0070] - libtorch autograd (battle-tested by millions of users)
[ARUST L0071] - libtorch CUDA caching allocator (proven memory management)
[ARUST L0072] - `tch::autocast` for bf16 mixed precision
[ARUST L0073] - `cudnn_benchmark` for convolution algorithm autotuning
[ARUST L0074] 
[ARUST L0075] ### burn-cuda Backend (Future Upgrade)
[ARUST L0076] 
[ARUST L0077] The `burn-cuda` backend uses CubeCL to JIT-generate CUDA kernels:
[ARUST L0078] - Implicit GEMM conv2d with tensor cores (CMMA + MMA) and autotuning
[ARUST L0079] - Operator fusion (fuse-on-read, fuse-on-write) for elementwise chains
[ARUST L0080] - 3-tier memory pool (SlicedPool, ExclusivePool, PersistentPool)
[ARUST L0081] - Published benchmarks: matches cuBLAS on matmul, 3-33x faster than libtorch on CPU ops
[ARUST L0082] - Upgrade path: swap `Burn<LibTorch>` to `Burn<CudaRuntime>`, zero code changes
[ARUST L0083] 
[ARUST L0084] ## 3. hydra-core vs Mortal's libriichi
[ARUST L0085] 
[ARUST L0086] ### Architecture: Delegation vs Monolith
[ARUST L0087] 
[ARUST L0088] hydra-core delegates game logic to riichienv-core. libriichi is fully self-contained.
[ARUST L0089] 
[ARUST L0090] | Area | hydra-core | libriichi |
[ARUST L0091] |---|---|---|
[ARUST L0092] | Game state | Delegates to riichienv-core | Own PlayerState (10 files) |
[ARUST L0093] | Shanten | Delegates to riichienv-core | Own solver with lookup tables |
[ARUST L0094] | Scoring | Delegates to riichienv-core | Own agari + point calculation |
[ARUST L0095] | Encoder | **192x34 fixed-superset tensor** (own; first 85 channels preserve the old baseline prefix) | **1012x34** tensor (own) |
[ARUST L0096] | Safety | **Dedicated module** (23 channels) | Embedded in state |
[ARUST L0097] | Action space | 46 actions (Mortal-compatible) | 46 actions |
[ARUST L0098] | Tile encoding | TileType(0-33) + suit permutation | Tile(0-37) incl aka+unknown |
[ARUST L0099] | Augmentation | **6-way suit permutation** | m/p swap only |
[ARUST L0100] | Encoding | **Incremental with DirtyFlags** | Full re-encode per turn |
[ARUST L0101] | Seeding | **SHA-256 KDF + vendored Fisher-Yates** | Standard RNG |
[ARUST L0102] | Arena | Partial training/runtime scaffolding exists; broad self-play mainline is not the current active tranche | Built-in self-play |
[ARUST L0103] | Dataset | MJAI log reader + batch pipeline scaffold exists | mjai log reader + batch pipeline |
[ARUST L0104] | Inference | Burn/runtime path exists; see `docs/GAME_ENGINE.md` and `hydra-train/src/inference.rs` | tch (libtorch) |
[ARUST L0105] | LoC | ~3,500 | ~15,000+ |
[ARUST L0106] | License | MIT | AGPL-3.0 |
[ARUST L0107] 
[ARUST L0108] ### What hydra-core Owns (unique to Hydra)
[ARUST L0109] 
[ARUST L0110] - 192x34 fixed-superset observation encoding, with the old 85-channel public+safety view preserved as the baseline prefix
[ARUST L0111] - Incremental encoding with DirtyFlags (skip unchanged channels)
[ARUST L0112] - 6-way suit permutation for data augmentation
[ARUST L0113] - SHA-256 KDF wall generation for cross-version determinism
[ARUST L0114] - ActionSelector trait for pluggable policies
[ARUST L0115] - Bridge layer between riichienv-core and encoder
[ARUST L0116] 
[ARUST L0117] ## 4. Performance Analysis: Max Rust vs Max Python
[ARUST L0118] 
[ARUST L0119] ### Python 3.2x Speedup Breakdown (ResNet50, CIFAR-10, Nsight-profiled)
[ARUST L0120] 
[ARUST L0121] | Optimization | img/s | Gain | Share of Total |
[ARUST L0122] |---|---:|---:|---:|
[ARUST L0123] | Baseline (eager Python) | 994 | -- | -- |
[ARUST L0124] | Fix .item() sync | 1,049 | +5.5% | 2.5% |
[ARUST L0125] | pin_memory + non_blocking | 1,063 | +1.3% | 0.6% |
[ARUST L0126] | cudnn.benchmark = True | 1,093 | +2.9% | 1.4% |
[ARUST L0127] | torch.compile() default | 1,290 | +18.1% | 9.0% |
[ARUST L0128] | torch.compile(max-autotune) | 1,393 | +8.0% | 4.7% |
[ARUST L0129] | Inductor exhaustive search | 1,427 | +2.4% | 1.6% |
[ARUST L0130] | **AMP (mixed precision)** | **3,026** | **+112%** | **73.2%** |
[ARUST L0131] | Channels-last memory | 3,178 | +5.0% | 7.0% |
[ARUST L0132] | **Total** | **3,178** | | **3.2x** |
[ARUST L0133] 
[ARUST L0134] Key insight: **73% of all gains come from bf16 tensor cores.** This is
[ARUST L0135] hardware-level, not Python-specific. Rust gets it identically.
[ARUST L0136] 
[ARUST L0137] ### C++ LibTorch vs Python PyTorch (no torch.compile)
[ARUST L0138] 
[ARUST L0139] | Model | Python (s/epoch) | C++ LibTorch (s/epoch) | C++ Speedup |
[ARUST L0140] |---|---:|---:|---:|
[ARUST L0141] | ResNet18 | 25.78 | 7.24 | **3.56x** |
[ARUST L0142] | ResNet34 | 45.24 | 11.06 | **4.09x** |
[ARUST L0143] 
[ARUST L0144] C++/Rust is 3.5-4x faster than Python for training **before any
[ARUST L0145] optimizations**. torch.compile exists to close this gap for Python.
[ARUST L0146] Rust starts where torch.compile tries to get to.
[ARUST L0147] 
[ARUST L0148] ### Head-to-Head: Max Python vs Max Rust
[ARUST L0149] 
[ARUST L0150] | Factor | Python (max perf) | Rust (burn-tch) |
[ARUST L0151] |---|---|---|
[ARUST L0152] | Conv2d kernels | cuDNN | **cuDNN (identical)** |
[ARUST L0153] | Matmul kernels | cuBLAS | **cuBLAS (identical)** |
[ARUST L0154] | bf16 tensor cores | autocast | **autocast (identical)** |
[ARUST L0155] | Operator fusion | torch.compile Inductor | None needed (no Python dispatch) |
[ARUST L0156] | CUDA graphs | Built into Inductor | Manual (or async execution) |
[ARUST L0157] | Interpreter overhead | ~3.5x penalty, clawed back by compile | **Zero** |
[ARUST L0158] | Self-play integration | Subprocess + IPC | **Same process, zero-copy** |
[ARUST L0159] | GIL contention | Yes (data loading, logging) | **None** |
[ARUST L0160] | Build reproducibility | pip/conda dependency tree | **cargo build** |
[ARUST L0161] 
[ARUST L0162] **Verdict: Max Rust >= Max Python.** Same CUDA kernels, zero Python tax,
[ARUST L0163] zero IPC for self-play. For RL workloads where self-play dominates wall
[ARUST L0164] clock time, Rust wins by a significant margin.
[ARUST L0165] 
[ARUST L0166] ## 5. All Concerns Raised and Resolutions
[ARUST L0167] 
[ARUST L0168] ### Resolved: "Just Write Rust" (17 concerns)
[ARUST L0169] 
[ARUST L0170] These were removed because they require only Rust code, not framework features.
[ARUST L0171] 
[ARUST L0172] | # | Concern | Resolution |
[ARUST L0173] |---|---|---|
[ARUST L0174] | 1 | No ONNX export | Use Burn directly for inference. Or save via burn-tch as TorchScript. |
[ARUST L0175] | 2 | No param groups | Multiple optimizers on different param subsets. |
[ARUST L0176] | 3 | Legacy RL-loop design gap | Keep as reserve context only; current active work is not “write PPO loop first.” |
[ARUST L0177] | 4 | Compile times | Pin to stable Burn version. Incremental builds mitigate. |
[ARUST L0178] | 5 | DDP is new | burn-tch backend's DDP tested. Manual NCCL via cudarc as fallback. |
[ARUST L0179] | 6 | No W&B | Hit W&B REST API via reqwest. Or use tensorboard-rs. |
[ARUST L0180] | 7 | Debugging grads | Write our own gradcheck (finite difference vs analytical). |
[ARUST L0181] | 8 | CUDA profiling | Nsight Systems/Compute work on any CUDA calls. |
[ARUST L0182] | 9 | ONNX import limited | Training from scratch. Not importing. |
[ARUST L0183] | 10 | bf16 not autocast | Whole model bf16 via device policy. bf16 has fp32 range. |
[ARUST L0184] | 11 | Checkpoint format | Burn's Record system. Converter if needed. |
[ARUST L0185] | 12 | Binary size | Acceptable for server-side training. |
[ARUST L0186] | 13 | API stability | Pin Burn version. Upgrade deliberately. |
[ARUST L0187] | 14 | Ecosystem lock-in | Accepted. Committed to Rust. |
[ARUST L0188] | 15 | Data pipeline | Burn's MultiThreadDataLoader. Custom Dataset for mjai logs. |
[ARUST L0189] | 16 | Legacy self-play loop plan | Keep as reserve context only; do not treat manual PPO loop work as the current default tranche. |
[ARUST L0190] | 17 | No no_grad scope | Use non-Autodiff backend for inference during rollout. |
[ARUST L0191] 
[ARUST L0192] ### Bridged: Framework-Level Concerns (6 concerns)
[ARUST L0193] 
[ARUST L0194] These required research to confirm Rust solutions exist.
[ARUST L0195] 
[ARUST L0196] | # | Concern | Bridge | Effort |
[ARUST L0197] |---|---|---|---|
[ARUST L0198] | 1 | conv2d perf vs cuDNN | burn-tch uses cuDNN. Identical perf. burn-cuda has implicit GEMM + tensor cores + autotuning. | Benchmark to verify |
[ARUST L0199] | 2 | Autodiff correctness | Write gradcheck. Cross-backend gradient comparison (burn-tch vs burn-cuda). Burn has per-op gradient tests. | ~100 lines Rust |
[ARUST L0200] | 3 | Kernel fusion bugs | Cross-backend tensor comparison for our architecture. Burn's test suite runs across all backends. | ~200 lines Rust |
[ARUST L0201] | 4 | GPU memory allocator | CubeCL has 3-tier memory pool (Sliced + Exclusive ring buffer + Persistent). burn-tch uses libtorch caching allocator. | Profile at first run |
[ARUST L0202] | 5 | Numerical divergence | Accepted. Tolerance-based assertions (1e-5 fp32, 1e-3 bf16). Different kernels = different FP rounding. | None needed |
[ARUST L0203] | 6 | Small community | burn-tch fallback. Contribute fixes upstream (Rust is readable). Pin versions. Dual-backend CI. | Ongoing |
[ARUST L0204] 
[ARUST L0205] ### New Concerns from Research (7 concerns)
[ARUST L0206] 
[ARUST L0207] | # | Concern | Resolution |
[ARUST L0208] |---|---|---|
[ARUST L0209] | 1 | Small tensor overhead (192x34 input) | Profile CubeCL JIT latency. Use burn-tch if JIT overhead dominates for Hydra’s fixed-superset inputs. |
[ARUST L0210] | 2 | Backend swap mid-training | Validate numerical stability of burn-tch to burn-cuda switch. May need retrain. |
[ARUST L0211] | 3 | Thread safety of autodiff | Clone model for inference threads. Single model for training. Standard RL pattern. |
[ARUST L0212] | 4 | Generic compile errors | Rust limitation. Mitigate with type aliases and wrapper types. |
[ARUST L0213] | 5 | Tensor creation from raw slices | Verify Burn's Tensor::from_data is zero-copy on CPU. Profile if bottleneck. |
[ARUST L0214] | 6 | NaN gradient detection | Write custom detect_anomaly. Hook into backward pass. |
[ARUST L0215] | 7 | Checkpoint compat across versions | Pin Burn version for training duration. Test checkpoint load before upgrading. |
[ARUST L0216] 
[ARUST L0217] ### Hard Blockers Found: Zero
[ARUST L0218] 
[ARUST L0219] No concern was identified that cannot be solved in Rust. Every PyTorch
[ARUST L0220] capability needed for Hydra has a Rust equivalent or can be built.
```

## Artifact 07 — research/infrastructure/COMPUTE_FEASIBILITY.md lines 1-220
Source label: ACOMP
Path: research/infrastructure/COMPUTE_FEASIBILITY.md
Use: treat this as evidence, not truth.
```text
[ACOMP L0001] # Compute Budget Feasibility Analysis: Hydra on 2668 GPU-hours
[ACOMP L0002] 
[ACOMP L0003] ## Executive Summary
[ACOMP L0004] 
[ACOMP L0005] **Bottom line: 2668 GPU-hours on Quadro RTX 5000 is sufficient for reaching strong amateur/low-dan play (Phase 1-2 of training), but fundamentally insufficient for LuckyJ-level (10+ dan) play without radical efficiency innovations. The budget is comparable to what Suphx used per RL agent on much weaker hardware, putting us in "one good shot" territory.**
[ACOMP L0006] 
[ACOMP L0007] **Recommendation: Option (b) -- pursue radically more efficient approach. BC-heavy pipeline with targeted RL fine-tuning is the only viable path at this budget.**
[ACOMP L0008] 
[ACOMP L0009] ---
[ACOMP L0010] 
[ACOMP L0011] ## 1. Hard Compute Data from Mahjong AI Systems
[ACOMP L0012] 
[ACOMP L0013] ### 1.1 Suphx (Microsoft Research, 2020)
[ACOMP L0014] 
[ACOMP L0015] **Source**: [arXiv:2003.13590](https://arxiv.org/abs/2003.13590), Section 4.2
[ACOMP L0016] 
[ACOMP L0017] | Metric | Value |
[ACOMP L0018] |--------|-------|
[ACOMP L0019] | Training hardware | 4 Titan XP (param server) + 40 Tesla K80 (self-play) |
[ACOMP L0020] | Training time | **2 days per RL agent** |
[ACOMP L0021] | Total GPU-hours | **2,112 GPU-hours per agent** (44 GPUs x 48h) |
[ACOMP L0022] | Self-play games | **1.5M games per agent** |
[ACOMP L0023] | Evaluation cost | 20 K80s for 2 additional days |
[ACOMP L0024] | SL training data | 15M discard + 5M riichi + 10M chow + 10M pong + 4M kong samples |
[ACOMP L0025] | Model | 50 residual blocks, 256 channels, 5 separate networks |
[ACOMP L0026] | Result | **10 dan** (Tenhou, somewhat unstable) |
[ACOMP L0027] 
[ACOMP L0028] **FLOPS-normalized to RTX 5000:**
[ACOMP L0029] - K80 die: ~2.9 TFLOPS FP32; Titan XP: ~12.1 TFLOPS; Quadro RTX 5000: **11.2 TFLOPS**
[ACOMP L0030] - Suphx effective compute: (40 x 2.9 + 4 x 12.1) x 48h = ~7,900 TFLOPS-hours
[ACOMP L0031] - Our budget: 2668 x 11.2 = **~29,900 TFLOPS-hours**
[ACOMP L0032] - **We have ~3.8x one Suphx agent's compute budget in raw FLOPS**
[ACOMP L0033] 
[ACOMP L0034] ### 1.2 LuckyJ / JueJong (Tencent AI Lab, 2022-2023)
[ACOMP L0035] 
[ACOMP L0036] **No public training compute data exists for LuckyJ itself.** However, we have data from the two published papers underlying its techniques:
[ACOMP L0037] 
[ACOMP L0038] **ACH / JueJong (ICLR 2022)** -- predecessor, 1-on-1 Mahjong only:
[ACOMP L0039] 
[ACOMP L0040] | Metric | Value |
[ACOMP L0041] |--------|-------|
[ACOMP L0042] | Hardware | 800 CPUs, 3200 GB RAM, 8 NVIDIA M40 GPUs |
[ACOMP L0043] | Training steps | 1,000,000 |
[ACOMP L0044] | Batch size | 8,192 |
[ACOMP L0045] | Model | 3 stages x 3 res blocks = 9 blocks, channels 64->128->32 |
[ACOMP L0046] | Eval games | 7,700 games vs 157 humans + 1,000 vs champion |
[ACOMP L0047] | Result | Beat 2014 World Mahjong champion in 1v1 |
[ACOMP L0048] 
[ACOMP L0049] **OLSS (ICML 2023)** -- online search for 4-player Mahjong:
[ACOMP L0050] 
[ACOMP L0051] | Metric | Value |
[ACOMP L0052] |--------|-------|
[ACOMP L0053] | Blueprint training | 8 V100 GPUs + 1,200 CPUs for 2 days = **384 V100-hours** |
[ACOMP L0054] | Environmental model | 8 V100 GPUs + 2,400 CPUs |
[ACOMP L0055] | Key innovation | 100x faster than common-knowledge subgame solving |
[ACOMP L0056] | Search at inference | pUCT (much cheaper than CFR) |
[ACOMP L0057] 
[ACOMP L0058] **Estimated LuckyJ total**: Given Tencent AI Lab's resources, likely **10,000-50,000+ GPU-hours** (V100/A100 class) across multiple training phases, hyperparameter sweeps, and iterative self-play leagues. The OLSS paper alone needed ~768 V100-hours for two model components, and LuckyJ would have required many iterations plus the league training described in their NeurIPS 2023 StarCraft work.
[ACOMP L0059] 
[ACOMP L0060] ### 1.3 Mortal (Individual Developer, Open Source)
[ACOMP L0061] 
[ACOMP L0062] **No published total compute budget.** Key data points:
[ACOMP L0063] 
[ACOMP L0064] | Metric | Value |
[ACOMP L0065] |--------|-------|
[ACOMP L0066] | Self-play throughput | 40K hanchans/hour on RTX 4090 + Ryzen 9 7950X |
[ACOMP L0067] | Default config | 192 channels, 40 res blocks (~our Hydra spec) |
[ACOMP L0068] | Batch size | 512 |
[ACOMP L0069] | Developer note | "cost me far more time and money than I anticipated for a hobby" |
[ACOMP L0070] 
[ACOMP L0071] **Estimated compute**: Based on architecture similarity (192ch/40 blocks is close to our 256ch/40 blocks) and the developer running on consumer hardware for months, likely **500-2,000 GPU-hours equivalent** on a 4090 over the full training run. Mortal achieves strong play but not 10-dan level.
[ACOMP L0072] 
[ACOMP L0073] ### 1.4 LsAc*-MJ (Low-Resource Mahjong, 2024)
[ACOMP L0074] 
[ACOMP L0075] **Source**: [Wiley 10.1155/2024/4558614](https://onlinelibrary.wiley.com/doi/full/10.1155/2024/4558614)
[ACOMP L0076] 
[ACOMP L0077] | Metric | Value |
[ACOMP L0078] |--------|-------|
[ACOMP L0079] | Training time | **51.4 hours** (vs DQN 277h, NFSP 355h, Dueling 1105h) |
[ACOMP L0080] | Parameters | 308K (much smaller model) |
[ACOMP L0081] | Key technique | Knowledge-guided pretraining + A2C self-play |
[ACOMP L0082] | Result | Beats DQN/NFSP/Dueling baselines (not competitive with Suphx/Mortal) |
[ACOMP L0083] 
[ACOMP L0084] This is the only paper explicitly targeting **low-resource Mahjong training**. Their two-stage approach (knowledge-guided + self-play) is conceptually similar to our BC + RL pipeline.
[ACOMP L0085] 
[ACOMP L0086] ---
[ACOMP L0087] 
[ACOMP L0088] ## 2. Comparison Game AI Compute Budgets
[ACOMP L0089] 
[ACOMP L0090] | System | Game | Hardware | Training Time | Effective Compute | Cost | Result |
[ACOMP L0091] |--------|------|----------|--------------|-------------------|------|--------|
[ACOMP L0092] | **Suphx** | Mahjong | 44 GPUs (K80+TitanXP) | 2 days/agent | ~2,100 GPU-hr/agent | Unknown | 10 dan |
[ACOMP L0093] | **LuckyJ** | Mahjong | V100s + thousands CPUs | Unknown | Est. 10K-50K+ GPU-hr | Unknown | 10.68 dan (stable) |
[ACOMP L0094] | **Mortal** | Mahjong | Consumer GPU (4090) | Months | Est. 500-2,000 GPU-hr | Personal funds | Strong amateur |
[ACOMP L0095] | **Pluribus** | Poker | 64-core CPU server | 8 days | 12,400 CPU-hr | **$144** | Superhuman |
[ACOMP L0096] | **AlphaStar** | StarCraft II | 16 TPUs/agent x 600 agents | 14-44 days | ~Millions TPU-hr | ~$Millions | Grandmaster |
[ACOMP L0097] | **OpenAI Five** | Dota 2 | 256 P100 + 128K CPUs | Months | ~Millions GPU-hr | ~$Millions | Beat pros |
[ACOMP L0098] | **Hydra (ours)** | Mahjong | Quadro RTX 5000 | TBD | **2,668 GPU-hr** | Grant-funded | Target: 80%+ |
[ACOMP L0099] 
[ACOMP L0100] ## 3. Scaling Laws for Game AI
[ACOMP L0101] 
[ACOMP L0102] **Source**: [arXiv:2301.13442](https://arxiv.org/abs/2301.13442) -- "Scaling Laws for Single-Agent RL"
[ACOMP L0103] 
[ACOMP L0104] Key findings relevant to us:
[ACOMP L0105] 
[ACOMP L0106] 1. **RL performance follows power laws** in both model size (N) and environment interactions (E): `I^(-beta) = (Nc/N)^alpha_N + (Ec/E)^alpha_E`
[ACOMP L0107] 
[ACOMP L0108] 2. **Optimal model size scales with compute budget**: The exponent ranges from 0.40-0.80 depending on domain. For Dota 2: ~0.76. This means as compute doubles, optimal model size should increase by ~70%.
[ACOMP L0109] 
[ACOMP L0110] 3. **Environment cost dominates**: "It is usually inefficient to use a model that is much cheaper to run than the environment." For Mahjong, our engine is fast (Rust), so we can afford a bigger model.
[ACOMP L0111] 
[ACOMP L0112] 4. **Dota 2 vs MNIST**: Same model needs ~2000x more training on Dota 2 than MNIST. Game complexity matters enormously.
[ACOMP L0113] 
[ACOMP L0114] 5. **Sample efficiency vs humans**: RL needs 100-10,000x more interactions than humans to reach the same level.
[ACOMP L0115] 
[ACOMP L0116] **No Mahjong-specific scaling laws exist.** But Mahjong complexity is between poker (simpler) and StarCraft/Dota (much more complex). The hidden information aspect adds sample complexity beyond what game-tree size alone would suggest.
[ACOMP L0117] 
[ACOMP L0118] ---
[ACOMP L0119] 
[ACOMP L0120] ## 4. Sample Efficiency Techniques (Dan-per-GPU-hour)
[ACOMP L0121] 
[ACOMP L0122] Ranked by expected impact:
[ACOMP L0123] 
[ACOMP L0124] ### 4.1 Oracle Guiding (Suphx, highest impact)
[ACOMP L0125] Train with perfect information first, then distill to imperfect-information agent. Suphx showed this dramatically accelerates early RL training. **Estimated 3-10x speedup** in early phases.
[ACOMP L0126] 
[ACOMP L0127] ### 4.2 Behavioral Cloning Pretraining (our Phase 1)
[ACOMP L0128] BC from expert games gives a strong initialization. Instead of learning from scratch via RL, start from a competent policy. **Estimated 5-20x speedup** vs pure RL from random.
[ACOMP L0129] 
[ACOMP L0130] ### 4.3 Global Reward Prediction (Suphx)
[ACOMP L0131] Predict final game outcome from intermediate states. Reduces credit assignment problem in long-horizon games. **Estimated 2-5x improvement** in value estimation quality.
[ACOMP L0132] 
[ACOMP L0133] ### 4.4 CQL (Conservative Q-Learning) -- Mortal's approach
[ACOMP L0134] Mortal uses offline RL (CQL) which is dramatically more sample-efficient than online RL because it reuses logged data. **This is our biggest efficiency lever** -- no wasted self-play games, every sample is reused.
[ACOMP L0135] 
[ACOMP L0136] ### 4.5 Knowledge Distillation
[ACOMP L0137] Train a small "fast" model for self-play generation, distill into the large model. Reduces self-play GPU cost by 4-8x.
[ACOMP L0138] 
[ACOMP L0139] ### 4.6 Prioritized Experience Replay
[ACOMP L0140] Re-weight training samples by TD error or novelty. Standard 1.5-2x improvement.
[ACOMP L0141] 
[ACOMP L0142] ---
[ACOMP L0143] 
[ACOMP L0144] ## 5. BC Data Scaling Saturation
[ACOMP L0145] 
[ACOMP L0146] No Mahjong-specific study exists, but from the general imitation learning literature and the Suphx data:
[ACOMP L0147] 
[ACOMP L0148] - **Suphx SL data**: 15M discard samples, 5M riichi, 10M each for chow/pong, 4M kong = ~44M total samples
[ACOMP L0149] - **Mortal**: Trained on years of Tenhou log data (millions of games available)
[ACOMP L0150] - **General pattern**: BC performance follows a log-linear curve -- each 10x increase in data gives a roughly constant improvement. Saturation typically occurs when the policy captures ~95% of expert behavior variance.
[ACOMP L0151] 
[ACOMP L0152] **Estimated saturation point for Mahjong BC**: ~500K-2M expert games (10-40M decision samples). Beyond this, additional data yields diminishing returns on action prediction accuracy. The remaining gap must be closed by RL.
[ACOMP L0153] 
[ACOMP L0154] **Key insight**: BC gets you to ~dan level cheaply. The jump from dan to 10-dan requires RL, which is where compute really matters.
[ACOMP L0155] 
[ACOMP L0156] ## 6. Optimal BC/RL Compute Split (the "Chinchilla" for Game AI)
[ACOMP L0157] 
[ACOMP L0158] No formal equivalent exists, but we can derive estimates from what worked:
[ACOMP L0159] 
[ACOMP L0160] **Suphx approach**: ~20% SL, ~80% RL (SL was a pretraining step, bulk of compute in RL self-play)
[ACOMP L0161] 
[ACOMP L0162] **Mortal approach**: Primarily offline RL on logged data (CQL). Essentially 100% data-reuse, minimal fresh self-play. This is massively more compute-efficient.
[ACOMP L0163] 
[ACOMP L0164] **AlphaStar**: ~5% imitation learning (from replays), ~95% RL (league self-play)
[ACOMP L0165] 
[ACOMP L0166] **Recommended split for Hydra at 2668 GPU-hours**:
[ACOMP L0167] 
[ACOMP L0168] | Phase | Budget | GPU-hours | What it buys |
[ACOMP L0169] |-------|--------|-----------|-------------|
[ACOMP L0170] | Phase 1: BC Pretraining | 10-15% | 250-400h | Train on expert Tenhou logs to ~dan prediction accuracy |
[ACOMP L0171] | Phase 2: Offline RL (CQL) | 40-50% | 1,000-1,300h | Fine-tune with conservative Q-learning on same data |
[ACOMP L0172] | Phase 3: Online RL Self-play | 30-40% | 800-1,000h | Self-play with PPO to go beyond expert data |
[ACOMP L0173] | Evaluation + Tuning | 5-10% | 130-270h | 1v3 testing, hyperparameter sweeps |
[ACOMP L0174] 
[ACOMP L0175] ---
[ACOMP L0176] 
[ACOMP L0177] ## 7. Minimum Viable Compute for Superhuman Mahjong
[ACOMP L0178] 
[ACOMP L0179] Nobody has studied this directly. But we can triangulate:
[ACOMP L0180] 
[ACOMP L0181] - **Pluribus** (poker, simpler): $144 on CPUs. Poker is *much* simpler than Mahjong.
[ACOMP L0182] - **Suphx** (Mahjong, 2020): ~2,100 GPU-hours on old hardware per agent, reached 10 dan (unstable). Multiple agents trained iteratively.
[ACOMP L0183] - **Mortal** (Mahjong, ongoing): Hundreds to low-thousands of GPU-hours on modern consumer hardware, reached strong amateur.
[ACOMP L0184] - **LuckyJ** (Mahjong, 2023): Unknown but almost certainly >>10K GPU-hours, reached stable 10.68 dan.
[ACOMP L0185] 
[ACOMP L0186] **Estimated minimum for 10+ dan Mahjong AI**:
[ACOMP L0187] - With state-of-the-art techniques (OLSS, oracle guiding, CQL): **5,000-15,000 RTX 5000-equivalent GPU-hours**
[ACOMP L0188] - Without advanced techniques (basic PPO self-play): **50,000-100,000+ GPU-hours**
[ACOMP L0189] - For 80% agreement (strong dan, not necessarily 10 dan): **1,500-5,000 GPU-hours**
[ACOMP L0190] 
[ACOMP L0191] ---
[ACOMP L0192] 
[ACOMP L0193] ## 8. Feasibility Assessment
[ACOMP L0194] 
[ACOMP L0195] ### What 2668 GPU-hours CAN achieve:
[ACOMP L0196] - Full BC pretraining to expert-level prediction accuracy
[ACOMP L0197] - Substantial offline RL (CQL) fine-tuning
[ACOMP L0198] - Limited online self-play (~10-20M games at our engine speed)
[ACOMP L0199] - A model that plays at strong amateur / low-dan level
[ACOMP L0200] - Probably 70-80% agreement with expert play
[ACOMP L0201] 
[ACOMP L0202] ### What 2668 GPU-hours CANNOT achieve:
[ACOMP L0203] - LuckyJ-level (10+ dan) performance
[ACOMP L0204] - Extensive hyperparameter search
[ACOMP L0205] - League-based training with multiple agents
[ACOMP L0206] - Many iterations of the train->evaluate->iterate cycle
[ACOMP L0207] 
[ACOMP L0208] ### Hardware context:
[ACOMP L0209] - Quadro RTX 5000: 11.2 TFLOPS FP32, 16GB GDDR6, 384 Tensor Cores
[ACOMP L0210] - 4 per Frontera node, likely running 1 training job across all 4
[ACOMP L0211] - bf16 training supported via Tensor Cores
[ACOMP L0212] 
[ACOMP L0213] ### The verdict:
[ACOMP L0214] 
[ACOMP L0215] **Option (b) is the answer: pursue radically more efficient approach.**
[ACOMP L0216] 
[ACOMP L0217] Specifically:
[ACOMP L0218] 1. **Maximize BC phase**: Use ALL available Tenhou expert data. This is the cheapest path to competence.
[ACOMP L0219] 2. **CQL over PPO**: Mortal's offline RL approach is dramatically more sample-efficient than online self-play. Reuse logged data.
[ACOMP L0220] 3. **Oracle guiding**: Train with perfect-information oracle first, distill to imperfect-information policy.
```

## Artifact 08 — research/intel/MAHJONG_TECHNIQUES.md lines 380-506
Source label: ATECH
Path: research/intel/MAHJONG_TECHNIQUES.md
Use: treat this as evidence, not truth.
```text
[ATECH L0380] - **Riichi timing + riichi tile**: The tile discarded with riichi declaration is often the "last useless tile," which narrows possible waits
[ATECH L0381] 
[ATECH L0382] ### Current AI Approaches
[ATECH L0383] 
[ATECH L0384] **Mortal**: Encodes opponent discards with:
[ATECH L0385] - First 6 and last 18 discards per opponent
[ATECH L0386] - Tedashi flag per discard
[ATECH L0387] - Recency-weighted encoding: `v = exp(-0.2 * (max_kawa_len - 1 - turn))`
[ATECH L0388] - Riichi tile tracking
[ATECH L0389] 
[ATECH L0390] Evidence: [Mortal obs_repr.rs L235-277](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L235-L277)
[ATECH L0391] 
[ATECH L0392] The NN then implicitly learns to read hands from these features. NO explicit hand-reading algorithm exists.
[ATECH L0393] 
[ATECH L0394] **Suphx**: Similarly relies on NN implicit learning. The oracle guiding phase gives the model access to opponent hands during training, which helps it learn correlations between discard patterns and actual hands.
[ATECH L0395] 
[ATECH L0396] ### Are There Better Explicit Algorithms?
[ATECH L0397] 
[ATECH L0398] **Short answer: No.** No published algorithm outperforms neural network implicit learning for hand reading. The reasons:
[ATECH L0399] 1. The space of possible opponent hands is enormous (~10^48 information sets per IJCAI 2024)
[ATECH L0400] 2. Exact Bayesian inference is computationally intractable
[ATECH L0401] 3. Heuristic hand reading (suji counting, suit analysis) captures only a small fraction of the available signal
[ATECH L0402] 
[ATECH L0403] ### The Gap
[ATECH L0404] 
[ATECH L0405] **Theoretical optimal**: Maintain a full probability distribution over each opponent's possible hand configurations, updated via Bayesian inference after each action. This is computationally intractable for exact computation but could be approximated via:
[ATECH L0406] - Particle filtering (sample possible hands, weight by consistency with observations)
[ATECH L0407] - Learned latent representations of opponent hand state
[ATECH L0408] - Explicit belief tracking networks
[ATECH L0409] 
[ATECH L0410] **Gap severity**: MEDIUM. Neural networks capture the MOST COMMON patterns well but may miss subtle signals in rare configurations. The biggest gap is in MULTI-STEP reasoning: "opponent discarded X, then called Y, then discarded Z" chains that require tracking sequential dependencies.
[ATECH L0411] 
[ATECH L0412] **Hydra opportunity**: The observation encoding already includes tedashi/tsumogiri distinction and recency weighting. Adding explicit attention over opponent discard sequences (transformer-style) could improve hand reading beyond what pure CNN architectures capture.
[ATECH L0413] 
[ATECH L0414] ---
[ATECH L0415] 
[ATECH L0416] ## 10. Disproportionate-Gain Mahjong-Specific Tricks
[ATECH L0417] 
[ATECH L0418] These are techniques where small implementation effort yields outsized performance gains:
[ATECH L0419] 
[ATECH L0420] ### Trick 1: Oracle Guiding (Suphx)
[ATECH L0421] 
[ATECH L0422] **What**: Train with perfect information (see all hands + wall), then gradually drop out oracle features.
[ATECH L0423] **Why it's powerful**: The oracle agent learns WHAT GOOD PLAY LOOKS LIKE with full information, then transfers that knowledge to the imperfect-information agent. This drastically speeds up RL training.
[ATECH L0424] **Evidence**: Suphx paper Section 3.3 -- the oracle features are dropped via a decay parameter gamma_t that goes from 1 to 0 over training. "With the help of the oracle agent, our normal agent improves much faster than standard RL training."
[ATECH L0425] **Hydra relevance**: DIRECT. Hydra's IVD (Invisible Value Decomposition) is related -- using privileged information during training that isn't available at inference.
[ATECH L0426] 
[ATECH L0427] ### Trick 2: Single-Player EV Tables (Mortal v4)
[ATECH L0428] 
[ATECH L0429] **What**: Precompute expected value curves (tenpai prob, win prob, point EV over turns) for each possible discard, using dynamic programming in a single-player model.
[ATECH L0430] **Why it's powerful**: Gives the NN a "cheat sheet" of optimal single-player play to start from. The NN then only needs to learn deviations caused by opponent interaction.
[ATECH L0431] **Evidence**: [Mortal obs_repr.rs L564-611](https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/state/obs_repr.rs#L564-L611)
[ATECH L0432] **Hydra relevance**: HIGH. This is essentially what Hydra's FBS (Feature-Based Shaping) aims to provide.
[ATECH L0433] 
[ATECH L0434] ### Trick 3: Auxiliary Prediction Heads
[ATECH L0435] 
[ATECH L0436] **What**: Add prediction targets beyond the main policy: tenpai probability, danger estimates, rank prediction.
[ATECH L0437] **Why it's powerful**: Forces the network to build representations that explicitly capture safety and hand-state information, rather than hoping the policy gradient discovers them.
[ATECH L0438] **Evidence**: Hydra's 5-head design (Policy + Value + GRP + Tenpai + Danger) is exactly this pattern. The Tenpai head is trained with supervision from perfect-info labels.
[ATECH L0439] **Hydra relevance**: CORE ARCHITECTURE.
[ATECH L0440] 
[ATECH L0441] ### Trick 4: Explicit Safety Channel Encoding
[ATECH L0442] 
[ATECH L0443] **What**: Instead of raw tile observations, pre-compute safety features (genbutsu masks, suji relationships, visible-tile counts) and include them as dedicated input channels.
[ATECH L0444] **Why it's powerful**: Reduces the learning burden on the NN. Instead of learning "if opponent discarded 1m and I have 4m, 4m is suji-safe" from scratch, the input directly encodes "4m is suji-safe vs opponent 1."
[ATECH L0445] **Evidence**: Hydra's 23 safety channels (62 base + 23 safety = 85 total) in the encoding spec.
[ATECH L0446] **Hydra relevance**: CORE DESIGN. This is the single highest-leverage Mahjong-specific optimization.
[ATECH L0447] 
[ATECH L0448] ### Trick 5: Agari Guard (Rule-Based Override)
[ATECH L0449] 
[ATECH L0450] **What**: Hard-code that the AI always wins when it can (never passes on tsumo/ron).
[ATECH L0451] **Why it's powerful**: Even strong RL policies occasionally learn to pass on winning hands in certain situations. A simple rule override prevents this catastrophic error.
[ATECH L0452] **Evidence**: [DeepWiki on Mortal](https://deepwiki.com/Equim-chan/Mortal/3-mortal-ai-system) -- "A rule-based agari guard can override suboptimal winning decisions."
[ATECH L0453] **Hydra relevance**: Trivial to implement, prevents rare but devastating mistakes.
[ATECH L0454] 
[ATECH L0455] ### Trick 6: Global Reward Prediction (Suphx)
[ATECH L0456] 
[ATECH L0457] **What**: Use an RNN to predict final game placement from round-by-round features, then use the per-round delta as the RL reward signal.
[ATECH L0458] **Why it's powerful**: Standard RL uses round-level point deltas as rewards, which can be misleading (losing points to protect 1st place is actually GOOD strategy). Global reward prediction correctly attributes game-level success to individual round decisions.
[ATECH L0459] **Evidence**: Suphx paper Section 3.2 -- GRU-based predictor with round-level features.
[ATECH L0460] **Hydra relevance**: Phase 3 of Hydra's training pipeline should incorporate this.
[ATECH L0461] 
[ATECH L0462] 
[ATECH L0463] ---
[ATECH L0464] 
[ATECH L0465] ## Summary: Gap Severity Rankings
[ATECH L0466] 
[ATECH L0467] | Topic | Gap Severity | Current Best | Theoretical Optimal | Hydra Leverage |
[ATECH L0468] |-------|-------------|-------------|--------------------|--------------------|
[ATECH L0469] | 1. Suji/Kabe Defense | MEDIUM-HIGH | Implicit NN learning | Bayesian posterior over waits | Danger head + safety channels |
[ATECH L0470] | 2. Damaten Detection | HIGH | No explicit detection | Tenpai probability estimator | Tenpai head with oracle labels |
[ATECH L0471] | 3. Betaori | HIGH | Akochan explicit, Mortal implicit | Multi-opponent threat + attack EV comparison | Danger head + Value head comparison |
[ATECH L0472] | 4. Placement-Aware | MEDIUM | Suphx GRP, Mortal score encoding | Exact placement probability optimization | GRP head + placement-weighted rewards |
[ATECH L0473] | 5. Yaku Selection | LOW-MEDIUM | Implicit NN learning | No formal framework exists | Indirect via GRP/Value heads |
[ATECH L0474] | 6. Call Efficiency | MEDIUM | RL Q-values | Information-theoretic call framework | Standard RL + better encoding |
[ATECH L0475] | 7. Riichi Timing | LOW-MEDIUM | RL + heuristic rules | Full EV comparison with placement | Policy action + Value head |
[ATECH L0476] | 8. Tile Efficiency | LOW (shanten solved) | Mortal SP tables | Single-player is solved; multi-player gap | FBS (Feature-Based Shaping) |
[ATECH L0477] | 9. Hand Reading | MEDIUM | Implicit NN + recency encoding | Bayesian belief tracking | Attention over discard sequences |
[ATECH L0478] | 10. Special Tricks | HIGH ROI | Scattered across AIs | Combined approach | All 6 tricks applicable |
[ATECH L0479] 
[ATECH L0480] ## Key Takeaway for Hydra
[ATECH L0481] 
[ATECH L0482] The biggest performance gaps in current Mahjong AI are in **defense** (topics 1-3) and **placement awareness** (topic 4). Hydra's multi-head architecture (Policy + Value + GRP + Tenpai + Danger) directly targets the top 4 gaps. The 23 safety channels in the encoder provide the safety-specific input that no other AI has in explicit form.
[ATECH L0483] 
[ATECH L0484] The disproportionate-gain tricks (Section 10) are all applicable to Hydra and should be implemented in order:
[ATECH L0485] 1. **Safety channel encoding** (already designed) -- highest immediate ROI
[ATECH L0486] 2. **Auxiliary prediction heads** (already designed) -- forces good representations
[ATECH L0487] 3. **Oracle guiding / IVD** (designed as IVD) -- dramatically speeds up RL
[ATECH L0488] 4. **SP EV tables / FBS** (designed as FBS) -- provides single-player optimal baseline
[ATECH L0489] 5. **Global reward prediction** (Phase 3) -- correct reward attribution
[ATECH L0490] 6. **Agari guard** (trivial) -- prevents rare catastrophic errors
[ATECH L0491] 
[ATECH L0492] ---
[ATECH L0493] 
[ATECH L0494] ## Sources
[ATECH L0495] 
[ATECH L0496] - Mortal source: https://github.com/Equim-chan/Mortal (commit 0cff2b52)
[ATECH L0497] - Suphx paper: https://arxiv.org/abs/2003.13590
[ATECH L0498] - Akochan source: https://github.com/critter-mj/akochan
[ATECH L0499] - Tjong paper: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298
[ATECH L0500] - Nyanten theory: https://qiita.com/Cryolite/items/75d504c7489426806b87
[ATECH L0501] - Shanten calculator: https://github.com/tomohxx/shanten-number
[ATECH L0502] - Defense data: https://riichi.wiki/Defense, https://riichi.wiki/Suji
[ATECH L0503] - Kabe analysis: https://pathofhouou.blogspot.com/2020/07/guideanalysis-defense-techniques-kabe.html
[ATECH L0504] - Riichi strategy: https://riichi.wiki/Riichi_strategy
[ATECH L0505] - Mahjong EV engine: https://github.com/CharlesC63/mahjong_ev
[ATECH L0506] - IJCAI Mahjong competition: https://www.ijcai.org/proceedings/2024/1020.pdf
```

## Artifact 09 — research/evidence/game-ai-papers-2024-2025.md lines 230-319
Source label: APAPERS
Path: research/evidence/game-ai-papers-2024-2025.md
Use: treat this as evidence, not truth.
```text
[APAPERS L0230] 
[APAPERS L0231] ## TIER 3: RELEVANT CONTEXT (From Same Research Groups / Adjacent Work)
[APAPERS L0232] 
[APAPERS L0233] ---
[APAPERS L0234] 
[APAPERS L0235] ### 9. Opponent Modeling with In-Context Search (NeurIPS 2024)
[APAPERS L0236] - **Team**: Haobo Fu et al. (Tencent AI -- LuckyJ group)
[APAPERS L0237] - **Note**: Builds on "Towards Offline Opponent Modeling with In-context Learning" (ICLR 2024)
[APAPERS L0238] - Uses in-context learning (transformer-style) to model opponents at test time
[APAPERS L0239] - Relevant for Mahjong: opponent modeling in 4-player setting is critical
[APAPERS L0240] 
[APAPERS L0241] ### 10. RegFTRL -- Regularized Follow-the-Regularized-Leader (AAMAS 2025)
[APAPERS L0242] - **Source**: https://dl.acm.org/doi/abs/10.1145/3719545.3719556
[APAPERS L0243] - Adaptive regularization for last-iterate convergence to Nash equilibrium in 2p zero-sum
[APAPERS L0244] - Relevant as a building block for equilibrium computation
[APAPERS L0245] 
[APAPERS L0246] ### 11. Student of Games (SoG) -- Science Advances 2023
[APAPERS L0247] - **Source**: https://www.science.org/doi/10.1126/sciadv.adg3256
[APAPERS L0248] - Unified algorithm: guided search + self-play + game-theoretic reasoning
[APAPERS L0249] - Works in BOTH perfect and imperfect information games
[APAPERS L0250] - DeepMind work -- benchmark for "general game AI"
[APAPERS L0251] - Already published (2023) so not novel for 2024-2025, but important baseline
[APAPERS L0252] 
[APAPERS L0253] ### 12. MCU: Evaluation Framework for Open-Ended Game Agents (ICML 2025)
[APAPERS L0254] - **Team**: Haobo Fu et al. (Tencent AI)
[APAPERS L0255] - Evaluation framework, not an algorithm -- but shows where the LuckyJ team is heading
[APAPERS L0256] 
[APAPERS L0257] ---
[APAPERS L0258] 
[APAPERS L0259] ## SYNTHESIS: WHAT'S GENUINELY NOVEL FOR 4-PLAYER MAHJONG
[APAPERS L0260] 
[APAPERS L0261] The key insight from this survey: **the frontier has moved from "better CFR variants"
[APAPERS L0262] to three genuinely new algorithmic directions**:
[APAPERS L0263] 
[APAPERS L0264] ### Direction A: Meta-Learned Equilibrium Finding
[APAPERS L0265] **Algorithms**: DDCFR -> Deep DDCFR -> PDCFR+
[APAPERS L0266] **Core idea**: Don't hand-tune CFR parameters. Learn them.
[APAPERS L0267] **Novel extension opportunity**: DDCFR learns discount weights. DRDA learns KL regularization
[APAPERS L0268] strength. Nobody has combined meta-learned dynamics with multiplayer-specific equilibrium
[APAPERS L0269] concepts (correlated equilibrium, team-maxmin, etc.)
[APAPERS L0270] 
[APAPERS L0271] ### Direction B: Search Without Common Knowledge
[APAPERS L0272] **Algorithms**: KLUSS/Obscuro, LAMIR
[APAPERS L0273] **Core idea**: Do real-time search in imperfect-info games without assuming players share
[APAPERS L0274] knowledge of game structure.
[APAPERS L0275] **Novel extension opportunity**: Apply KLUSS-style search to 4-player Mahjong. Prior Mahjong
[APAPERS L0276] AIs (Suphx, LuckyJ) do NO search at test time -- they're pure policy networks. Adding
[APAPERS L0277] principled search that handles Mahjong's lack of common knowledge would be genuinely new.
[APAPERS L0278] 
[APAPERS L0279] ### Direction C: Continuous Abstraction + Neural Equilibrium
[APAPERS L0280] **Algorithms**: Embedding CFR, DRDA
[APAPERS L0281] **Core idea**: Replace discrete game abstractions with continuous representations.
[APAPERS L0282] **Novel extension opportunity**: Learn continuous information-set embeddings for Mahjong
[APAPERS L0283] hands/states, then run equilibrium-finding in embedding space. This sidesteps the
[APAPERS L0284] intractable full game tree while preserving fine-grained hand distinctions.
[APAPERS L0285] 
[APAPERS L0286] ---
[APAPERS L0287] 
[APAPERS L0288] ## NOVELTY ASSESSMENT: What Would Pass a Reviewer's "Novel Algorithm" Bar
[APAPERS L0289] 
[APAPERS L0290] | Idea | Novel? | Risk |
[APAPERS L0291] |------|--------|------|
[APAPERS L0292] | "We apply DDCFR to Mahjong" | NO -- just application | Low novelty |
[APAPERS L0293] | "We combine policy gradient with CFR" | NO -- Dream/NFSP exist | Known recipe |
[APAPERS L0294] | "We extend DRDA to 4-player Mahjong with partial observability" | MAYBE -- if you prove new convergence bounds | Medium |
[APAPERS L0295] | "We design test-time search for Mahjong using KLUSS without common knowledge" | YES -- nobody has done this for Mahjong | High novelty |
[APAPERS L0296] | "We learn continuous info-set embeddings for Mahjong + run CFR in embedding space" | YES -- Embedding CFR is new and Mahjong is untouched | High novelty |
[APAPERS L0297] | "We meta-learn the equilibrium-finding dynamics for 4-player games (extending DDCFR to multiplayer-specific solution concepts)" | YES -- DDCFR is 2p, extending the meta-learning to multiplayer equilibria is open | High novelty |
[APAPERS L0298] | "We combine policy network with learned-model test-time search (LAMIR) for Mahjong" | YES -- first test-time search for Mahjong with learned models | High novelty |
[APAPERS L0299] 
[APAPERS L0300] ---
[APAPERS L0301] 
[APAPERS L0302] ## THE TOP 3 MOST PROMISING NOVEL ALGORITHM IDEAS FOR HYDRA
[APAPERS L0303] 
[APAPERS L0304] ### Idea 1: KLUSS-Mahjong -- Test-Time Search Without Common Knowledge
[APAPERS L0305] Take KLUSS (Obscuro's core algorithm) and adapt it for 4-player Mahjong.
[APAPERS L0306] - Current SOTA Mahjong AIs are pure policy networks (no search)
[APAPERS L0307] - KLUSS handles imperfect info without common knowledge (perfect for Mahjong)
[APAPERS L0308] - You'd need: belief network over opponent hands + KLUSS subgame construction
[APAPERS L0309] - This is the AlphaGo->AlphaZero moment: adding search to a policy-only system
[APAPERS L0310] - Contribution: First test-time search algorithm for N-player imperfect info tile games
[APAPERS L0311] 
[APAPERS L0312] ### Idea 2: Continuous Belief Equilibrium (DRDA + Embedding CFR hybrid)
[APAPERS L0313] Combine DRDA's multiplayer equilibrium finding with Embedding CFR's continuous abstraction.
[APAPERS L0314] - Learn continuous embeddings for Mahjong information sets (hand + discards + context)
[APAPERS L0315] - Run DRDA dynamics in embedding space for multiplayer equilibrium
[APAPERS L0316] - KL regularization from a pre-trained policy network (your Hydra policy becomes pi_base)
[APAPERS L0317] - Contribution: First algorithm for multiplayer equilibrium finding in continuous
[APAPERS L0318]   information-set embedding spaces. New mathematical framework.
[APAPERS L0319] 
```

## Artifact 10 — research/design/OPPONENT_MODELING.md lines 641-719
Source label: AOPP
Path: research/design/OPPONENT_MODELING.md
Use: treat this as evidence, not truth.
```text
[AOPP L0641] ### Discard Sequence Encoder (GRU)
[AOPP L0642] 
[AOPP L0643] Per-opponent GRU over the full discard history to capture temporal patterns (tedashi/tsumogiri sequences, call interruptions). This remains a reserve ablation idea rather than part of the current active path.
[AOPP L0644] 
[AOPP L0645] ### Constraint-Consistent Belief via Sinkhorn Projection (Tile Allocation Head)
[AOPP L0646] 
[AOPP L0647] **Cross-field import:** Optimal Transport / differentiable matrix scaling (Cuturi, NeurIPS 2013; Mena et al., ICLR 2018).
[AOPP L0648] 
[AOPP L0649] **Problem:** Mahjong's hidden state is *constrained unknown*, not just unknown. At any time, the remaining tile multiset is fixed by counts (4 copies per type minus visible tiles). Opponents' concealed hands are strongly anti-correlated through shared tile availability. Neural heads that predict per-opponent tile distributions independently can output *inconsistent marginals* (e.g., "each opponent probably has 2x 5p" when only one 5p remains unseen). This miscalibration is systematic and worst in exactly the situations where defense/offense hinges on 1-2 tile copies.
[AOPP L0650] 
[AOPP L0651] **Solution:** Add a differentiable Sinkhorn projection layer that enforces global tile conservation as a hard structural constraint inside the forward pass.
[AOPP L0652] 
[AOPP L0653] **Architecture:**
[AOPP L0654] 1. **TileAllocationHead**: `Conv1d(256 -> 4, kernel_size=1)` producing logits `[B x 4 x 34]` where Z=4 zones are:
[AOPP L0655] 
[AOPP L0656] | Zone | Content | `zone_size[z]` computation |
[AOPP L0657] |------|---------|---------------------------|
[AOPP L0658] | 0 | Opponent Left (shimocha) concealed hand | 13 - open_meld_tiles[left] - kans[left] |
[AOPP L0659] | 1 | Opponent Cross (toimen) concealed hand | 13 - open_meld_tiles[cross] - kans[cross] |
[AOPP L0660] | 2 | Opponent Right (kamicha) concealed hand | 13 - open_meld_tiles[right] - kans[right] |
[AOPP L0661] | 3 | Wall remainder (live wall + dead wall unseen) | 136 - 4*13 - visible_tiles - dead_wall_revealed |
[AOPP L0662] 
[AOPP L0663] > `remaining[t] = 4 - visible_count[t]` for each of 34 tile types. Visible tiles include: own hand, all discards, all open melds, all dora indicators. The sum `sum_z zone_size[z]` must equal `sum_t remaining[t]` (total unseen tiles) -- this is guaranteed by construction and serves as a runtime sanity check.
[AOPP L0664] 2. Convert logits to positive matrix `A = softplus(logits)` (or `exp(logits/tau)` with tau=1.0 default).
[AOPP L0665] 3. Run **Sinkhorn-Knopp iterations** (20 iterations default, range 10-30) in **log-domain** to find matrix `X` whose:
[AOPP L0666]    - Row sums match the **remaining count** of each tile type (known exactly from visible tiles): `sum_z X[t,z] = remaining[t]`
[AOPP L0667]    - Column sums match each zone's **unknown tile count** (known from public state: meld counts, hand sizes, wall size): `sum_t X[t,z] = zone_size[z]`
[AOPP L0668] 4. Output: consistent expected tile counts per zone per tile type.
[AOPP L0669] 
[AOPP L0670] **Mathematical basis:** The standard entropic optimal transport problem minimizes `<C, P>` subject to `P*1 = r` and `P^T*1 = c`, where r and c are arbitrary non-negative marginals. Sinkhorn iterations alternate row and column normalization: `u^(l+1) = r / (K * v^(l))`, `v^(l+1) = c / (K^T * u^(l+1))`, where `K = exp(-C/epsilon)`. Convergence to the unique solution is guaranteed for positive matrices (Sinkhorn, 1964). Cuturi (NeurIPS 2013, [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)) showed this can be computed efficiently and differentiated through. Mena et al. (ICLR 2018, [arXiv:1802.08665](https://arxiv.org/abs/1802.08665)) demonstrated differentiable Sinkhorn layers inside neural networks with backpropagation through the iterations.
[AOPP L0671] 
[AOPP L0672] **For Hydra's mahjong case:** Row marginals = remaining tile counts per type (34 values, known exactly). Column marginals = zone sizes (opponent concealed hand sizes + wall remainder, known from public state). The Sinkhorn projection enforces that the belief over hidden tiles is globally consistent with tile conservation -- something no existing mahjong AI does.
[AOPP L0673] 
[AOPP L0674] **Computational cost:** 10-30 iterations of matrix-vector multiply on a [34 x 4] matrix. Microseconds per forward pass, negligible relative to Hydra's backbone inference budget.
[AOPP L0675] 
[AOPP L0676] **Training signal:**
[AOPP L0677] - **Phase 1:** Labels from log-reconstructed opponent hands (same infrastructure as tenpai/danger/wait-set labels). Target: per-opponent concealed tile count vectors (34-dim).
[AOPP L0678] - **Phase 2-3:** Oracle teacher sees exact hands. Dense, noise-free supervision.
[AOPP L0679] - **Loss:** KL divergence on Sinkhorn-projected marginals vs ground truth counts, weight 0.02. Same gradient magnitude caution as dense danger labels.
[AOPP L0680] 
[AOPP L0681] **Integration with existing heads:** The Sinkhorn belief output serves as a force multiplier for all downstream opponent modeling:
[AOPP L0682] - **Danger head:** calibrate per-tile danger with "can they even structurally support this wait?"
[AOPP L0683] - **Wait-set head:** constrain wait predictions to be consistent with available tiles.
[AOPP L0684] - **Tenpai head:** if the belief assigns near-zero probability to tenpai-enabling tiles being in an opponent's hand, tenpai probability should be low.
[AOPP L0685] - Feed belief marginals (3x34 opponent tile probabilities) as extra channels into policy/danger heads *after the backbone*, not into the 85-channel observation.
[AOPP L0686] 
[AOPP L0687] **Stability notes:** Log-domain Sinkhorn (log-sum-exp formulation) is required for numerical stability with small epsilon. Well-documented in the literature (Peyre & Cuturi, "Computational Optimal Transport", 2019). Known issues: gradient vanishing for very small epsilon (too peaked); gradient explosion for very large epsilon (too uniform). **Default: epsilon = 0.05** (midpoint of 0.01–0.1 range). Tune by monitoring row/column sum constraint residuals during training.
[AOPP L0688] 
[AOPP L0689] > **Novelty note:** No published mahjong AI or poker AI uses a Sinkhorn/OT projection layer for belief inference inside the agent network. The closest adjacent works are: (1) diffusion-based mahjong hand generation ([DMV Nico case study](https://dmv.nico/en/casestudy/mahjong_tehai_generation/)), which generates hands but requires post-hoc greedy discretization to enforce tile counts -- proving the constraint problem exists; (2) LinSATNet (Wang et al., ICML 2023, [GitHub](https://github.com/Thinklab-SJTU/LinSATNet)), a differentiable Sinkhorn-based constraint satisfaction layer proven to work for routing, graph matching, and portfolio allocation -- proving the mechanism works. The specific intersection of "differentiable Sinkhorn constraint layer inside a game-playing agent for hand inference" is empty in the literature. This is a genuine cross-field import from optimal transport / constrained structured prediction into game AI.
[AOPP L0690] 
[AOPP L0691] > **Consensus note:** This approach was independently proposed as the #1 recommendation by two separate frontier AI analyses (GPT-5.2 Pro, two independent runs) without seeing each other's output. Both identified mahjong's tile-count conservation as the key structural property that makes Sinkhorn uniquely appropriate.
[AOPP L0692] 
[AOPP L0693] ### Pragmatic Deception via Rational Speech Acts (Phase 3 Module)
[AOPP L0694] 
[AOPP L0695] **Cross-field import:** Cognitive linguistics / computational pragmatics (Frank & Goodman, "Predicting Pragmatic Reasoning in Language Games", Science 336:998, 2012).
[AOPP L0696] 
[AOPP L0697] **Problem:** Riichi discards are a 34-dimensional "language" that opponents read to infer your hand. Strong opponents (and search-based agents like LuckyJ) use your discard sequence to narrow down your possible hands. A predictable agent is an exploitable agent -- especially against inference-time search, which samples hands consistent with your observed behavior.
[AOPP L0698] 
[AOPP L0699] **Solution:** Train Hydra to choose discards that actively minimize an observer model's ability to predict its true waiting tiles. In RSA terms: Hydra becomes a "pragmatic speaker" that selects "utterances" (discards) to manipulate the "listener's" (opponent's) Bayesian posterior away from truth.
[AOPP L0700] 
[AOPP L0701] #### L0 Public-Only Observer (Architecture)
[AOPP L0702] 
[AOPP L0703] A separate lightweight network trained to predict Hydra's wait from PUBLIC information only.
[AOPP L0704] 
[AOPP L0705] | Property | Specification |
[AOPP L0706] |----------|---------------|
[AOPP L0707] | Architecture | 10-block SE-ResNet, **96 channels**, same block structure as Hydra's backbone (pre-activation, GroupNorm(32), Mish, dual-pool SE ratio=16). Stem: `Conv1d(73, 96, 3, padding=1, bias=False)`. Output head: `GAP(96×34 → 96) → FC(96 → 34) → Sigmoid`. |
[AOPP L0708] | Input shape | `[B x 73 x 34]` (Hydra's 85 public channels MINUS 11 private hand/draw channels + 1 player-perspective channel) |
[AOPP L0709] | Output | `[B x 34]` sigmoid -- per-tile probability that the tile is in Hydra's waiting set |
[AOPP L0710] | Parameters | ~3.2M (10 blocks × ~300K/block at 96ch + stem ~21K + head ~3.3K) |
[AOPP L0711] | Training data | Phase 1 game logs. For each state where the acting player is in tenpai, label = binary wait mask (34-dim). Input = public info only. |
[AOPP L0712] | Training | Supervised BCE, 3 epochs on Phase 1 data. **[estimated]** Convergence accuracy unknown -- no published mahjong AI has measured wait prediction from public info only. Measure L0's top-3 accuracy on a held-out eval set after training; this becomes the empirical WOR baseline. |
[AOPP L0713] | Freeze point | After Phase 1 training. **Never updated during Phase 3 self-play.** |
[AOPP L0714] | Storage | `checkpoints/l0_observer/` -- single file, frozen. |
[AOPP L0715] 
[AOPP L0716] #### Deception Reward (Phase 3 PPO Integration)
[AOPP L0717] 
[AOPP L0718] Added as an auxiliary reward term during Phase 3 self-play:
[AOPP L0719] 
```

## Artifact 11 — research/design/HYDRA_SPEC.md lines 114-199
Source label: ASPEC
Path: research/design/HYDRA_SPEC.md
Use: treat this as evidence, not truth.
```text
[ASPEC L0114] Hydra uses a **Unified Multi-Head SE-ResNet** architecture. A single deep convolutional backbone extracts features from the game state, and five specialized heads branch from the shared latent representation to produce all outputs simultaneously.
[ASPEC L0115] 
[ASPEC L0116] The input observation tensor has shape `[Batch × 85 × 34]`, encoding 85 feature channels across the 34 tile types. A convolutional stem projects this into 256 channels using a 3×1 kernel. The representation then flows through 40 pre-activation SE-ResNet blocks — each applying GroupNorm, Mish activation, two 3×1 convolutions, and a squeeze-and-excitation attention gate — producing a shared latent tensor of shape `[B × 256 × 34]`. No pooling is applied anywhere in the backbone, preserving the full 34-tile spatial geometry.
[ASPEC L0117] 
[ASPEC L0118] For Phase 2 Oracle Distillation, the Teacher network uses the same backbone but with a wider stem: `Conv1d(290, 256, 3)` instead of `Conv1d(85, 256, 3)`. The 290-channel input is the public observation (85ch) concatenated with the oracle observation (205ch: opponent hands, wall draw order, dora/ura indicators). All 40 ResBlock weights are identical and transferable between teacher and student — only the stem Conv1d differs. See [Phase 2: Oracle Distillation RL](TRAINING.md#phase-2-oracle-distillation-rl) for the full oracle encoding specification.
[ASPEC L0119] 
[ASPEC L0120] From this shared representation, output heads operate from the backbone: the Policy Head selects the next action, the Value Head estimates expected round outcome, the GRP Head predicts final game placement distribution, the Tenpai Head estimates opponent tenpai probabilities, and the Danger Head estimates per-tile deal-in risk per opponent. The baseline five heads run in parallel. When extended heads are active (call-intent, wait-set, value-tenpai, Sinkhorn), the call-intent head runs first and its output conditions the Danger Head via FiLM — a minor sequential dependency with negligible latency impact (~0.1ms). See [OPPONENT_MODELING § 4.7](OPPONENT_MODELING.md#47-call-intent--yaku-plan-inference-head) for details.
[ASPEC L0121] 
[ASPEC L0122] ```mermaid
[ASPEC L0123] graph TB
[ASPEC L0124]     subgraph "Input Layer"
[ASPEC L0125]         INPUT["Observation Tensor<br/>[Batch × 85 × 34]<br/>62 base + 23 safety channels"]
[ASPEC L0126]     end
[ASPEC L0127] 
[ASPEC L0128]     subgraph "Stem"
[ASPEC L0129]         STEM["Conv1D Stem<br/>3×1 kernel, padding 1, no bias<br/>85→256 channels, stride 1"]
[ASPEC L0130]     end
[ASPEC L0131] 
[ASPEC L0132]     subgraph "Backbone"
[ASPEC L0133]         RES["40× SE-ResNet Blocks<br/>256ch, GroupNorm(32)<br/>SE-ratio=16, Mish activation<br/>Pre-activation, No Pooling"]
[ASPEC L0134]     end
[ASPEC L0135] 
[ASPEC L0136]     subgraph "Shared Representation"
[ASPEC L0137]         LATENT["Latent Features<br/>[B × 256 × 34]"]
[ASPEC L0138]     end
[ASPEC L0139] 
[ASPEC L0140]     subgraph "Output Heads"
[ASPEC L0141]         POLICY["Policy Head<br/>Softmax(46)"]
[ASPEC L0142]         VALUE["Value Head<br/>Scalar"]
[ASPEC L0143]         GRP["GRP Head<br/>Softmax(24)"]
[ASPEC L0144]         TENPAI["Tenpai Head<br/>Sigmoid(3)"]
[ASPEC L0145]         DANGER["Danger Head<br/>Sigmoid(3×34)"]
[ASPEC L0146]     end
[ASPEC L0147] 
[ASPEC L0148]     INPUT --> STEM
[ASPEC L0149]     STEM --> RES
[ASPEC L0150]     RES --> LATENT
[ASPEC L0151]     LATENT --> POLICY
[ASPEC L0152]     LATENT --> VALUE
[ASPEC L0153]     LATENT --> GRP
[ASPEC L0154]     LATENT --> TENPAI
[ASPEC L0155]     LATENT --> DANGER
[ASPEC L0156] ```
[ASPEC L0157] 
[ASPEC L0158] ---
[ASPEC L0159] 
[ASPEC L0160] ## Backbone Specification
[ASPEC L0161] 
[ASPEC L0162] ### Why SE-ResNet?
[ASPEC L0163] 
[ASPEC L0164] SE-ResNet captures global board state (e.g., "expensive field," dora density) via channel-wise squeeze-and-excitation attention while maintaining the spatial tile geometry that matters for shape recognition. Mortal already uses dual-pool SE-style channel attention (`model.py:L10-28`, at commit `0cff2b5`); Hydra retains this proven design but replaces BatchNorm with GroupNorm for batch-size independence during RL training. Suphx uses a plain deep residual CNN without channel attention.
[ASPEC L0165] 
[ASPEC L0166] | Architecture | Pros | Cons | Used By |
[ASPEC L0167] |--------------|------|------|---------|
[ASPEC L0168] | ResNet | Fast, proven for spatial | Limited global context | Suphx (50 blocks, 256 filters) |
[ASPEC L0169] | ResNet + Channel Attention | Global context via squeeze-excite | Slightly more params | Mortal v1–v4 (dual-pool SE) |
[ASPEC L0170] | Transformer | Long-range dependencies | ~90-310M params (45-155× larger than ResNet); no published mahjong performance benchmarks despite multi-year Kanachan development (public repo created 2021-08-05); impractical for online RL self-play | Kanachan (no results), Tjong |
[ASPEC L0171] | Hybrid | Best of both | Complexity, unproven | — |
[ASPEC L0172] 
[ASPEC L0173] ### Block Structure
[ASPEC L0174] 
[ASPEC L0175] Each SE-ResNet block uses pre-activation ordering: GroupNorm → Mish → Conv1D → GroupNorm → Mish → Conv1D → SE Attention → residual add. Both convolutions use 3×1 kernels with padding 1 and no bias (GroupNorm handles centering). The residual connection bypasses the entire block, enabling gradient flow through 40 layers.
[ASPEC L0176] 
[ASPEC L0177] ```mermaid
[ASPEC L0178] graph LR
[ASPEC L0179]     subgraph "SE-ResBlock (Pre-Activation)"
[ASPEC L0180]         IN[Input] --> GN1[GroupNorm 32]
[ASPEC L0181]         GN1 --> ACT1[Mish]
[ASPEC L0182]         ACT1 --> CONV1["Conv1D 3×1<br/>256ch, no bias"]
[ASPEC L0183]         CONV1 --> GN2[GroupNorm 32]
[ASPEC L0184]         GN2 --> ACT2[Mish]
[ASPEC L0185]         ACT2 --> CONV2["Conv1D 3×1<br/>256ch, no bias"]
[ASPEC L0186]         CONV2 --> SE[SE Attention]
[ASPEC L0187]         SE --> ADD((+))
[ASPEC L0188]         IN --> ADD
[ASPEC L0189]     end
[ASPEC L0190] ```
[ASPEC L0191] 
[ASPEC L0192] ### SE Attention Module
[ASPEC L0193] 
[ASPEC L0194] The squeeze-and-excitation module uses dual-pool channel attention (inspired by the channel attention component of CBAM, Woo et al. 2018), matching Mortal's implementation exactly. The feature tensor is independently average-pooled and max-pooled to single values per channel, each passed through a **shared MLP** (same weights for both paths), then **element-wise added** (not concatenated) before sigmoid. This means the FC input dimension remains C (not 2C), and the bottleneck is C/r = 256/16 = **16**.
[ASPEC L0195] 
[ASPEC L0196] ```mermaid
[ASPEC L0197] graph LR
[ASPEC L0198]     subgraph "Squeeze-and-Excitation (CBAM-style)"
[ASPEC L0199]         F[Features] --> GAP[Global Avg Pool]
```

## Artifact 12 — hydra-train/src/backbone.rs lines 1-145
Source label: ABACK
Path: hydra-train/src/backbone.rs
Use: treat this as evidence, not truth.
```text
[ABACK L0001] //! SE-ResNet backbone: SEBlock, SEResBlock, and SEResNet.
[ABACK L0002] 
[ABACK L0003] use burn::nn::{
[ABACK L0004]     GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig1d,
[ABACK L0005]     conv::{Conv1d, Conv1dConfig},
[ABACK L0006] };
[ABACK L0007] use burn::prelude::*;
[ABACK L0008] use burn::tensor::activation;
[ABACK L0009] 
[ABACK L0010] #[derive(Config, Debug)]
[ABACK L0011] pub struct SEBlockConfig {
[ABACK L0012]     pub channels: usize,
[ABACK L0013]     pub bottleneck: usize,
[ABACK L0014] }
[ABACK L0015] 
[ABACK L0016] #[derive(Module, Debug)]
[ABACK L0017] pub struct SEBlock<B: Backend> {
[ABACK L0018]     fc1: Linear<B>,
[ABACK L0019]     fc2: Linear<B>,
[ABACK L0020] }
[ABACK L0021] 
[ABACK L0022] impl SEBlockConfig {
[ABACK L0023]     pub fn init<B: Backend>(&self, device: &B::Device) -> SEBlock<B> {
[ABACK L0024]         SEBlock {
[ABACK L0025]             fc1: LinearConfig::new(self.channels, self.bottleneck).init(device),
[ABACK L0026]             fc2: LinearConfig::new(self.bottleneck, self.channels).init(device),
[ABACK L0027]         }
[ABACK L0028]     }
[ABACK L0029] }
[ABACK L0030] 
[ABACK L0031] impl<B: Backend> SEBlock<B> {
[ABACK L0032]     pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
[ABACK L0033]         let scale = x.clone().mean_dim(2).squeeze_dim::<2>(2);
[ABACK L0034]         let scale = activation::mish(self.fc1.forward(scale));
[ABACK L0035]         let scale = activation::sigmoid(self.fc2.forward(scale));
[ABACK L0036]         let scale = scale.unsqueeze_dim::<3>(2);
[ABACK L0037]         x.mul(scale)
[ABACK L0038]     }
[ABACK L0039] }
[ABACK L0040] 
[ABACK L0041] #[derive(Config, Debug)]
[ABACK L0042] pub struct SEResBlockConfig {
[ABACK L0043]     pub channels: usize,
[ABACK L0044]     pub num_groups: usize,
[ABACK L0045]     pub se_bottleneck: usize,
[ABACK L0046] }
[ABACK L0047] 
[ABACK L0048] #[derive(Module, Debug)]
[ABACK L0049] pub struct SEResBlock<B: Backend> {
[ABACK L0050]     gn1: GroupNorm<B>,
[ABACK L0051]     conv1: Conv1d<B>,
[ABACK L0052]     gn2: GroupNorm<B>,
[ABACK L0053]     conv2: Conv1d<B>,
[ABACK L0054]     se: SEBlock<B>,
[ABACK L0055] }
[ABACK L0056] 
[ABACK L0057] impl SEResBlockConfig {
[ABACK L0058]     pub fn init<B: Backend>(&self, device: &B::Device) -> SEResBlock<B> {
[ABACK L0059]         let conv_cfg =
[ABACK L0060]             Conv1dConfig::new(self.channels, self.channels, 3).with_padding(PaddingConfig1d::Same);
[ABACK L0061]         let gn_cfg = GroupNormConfig::new(self.num_groups, self.channels);
[ABACK L0062]         let se_cfg = SEBlockConfig::new(self.channels, self.se_bottleneck);
[ABACK L0063]         SEResBlock {
[ABACK L0064]             gn1: gn_cfg.init(device),
[ABACK L0065]             conv1: conv_cfg.init(device),
[ABACK L0066]             gn2: GroupNormConfig::new(self.num_groups, self.channels).init(device),
[ABACK L0067]             conv2: Conv1dConfig::new(self.channels, self.channels, 3)
[ABACK L0068]                 .with_padding(PaddingConfig1d::Same)
[ABACK L0069]                 .init(device),
[ABACK L0070]             se: se_cfg.init(device),
[ABACK L0071]         }
[ABACK L0072]     }
[ABACK L0073] }
[ABACK L0074] 
[ABACK L0075] impl<B: Backend> SEResBlock<B> {
[ABACK L0076]     pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
[ABACK L0077]         let residual = x.clone();
[ABACK L0078]         let out = activation::mish(self.gn1.forward(x));
[ABACK L0079]         let out = self.conv1.forward(out);
[ABACK L0080]         let out = activation::mish(self.gn2.forward(out));
[ABACK L0081]         let out = self.conv2.forward(out);
[ABACK L0082]         let out = self.se.forward(out);
[ABACK L0083]         out + residual
[ABACK L0084]     }
[ABACK L0085] }
[ABACK L0086] 
[ABACK L0087] #[derive(Config, Debug)]
[ABACK L0088] pub struct SEResNetConfig {
[ABACK L0089]     pub num_blocks: usize,
[ABACK L0090]     pub input_channels: usize,
[ABACK L0091]     pub hidden_channels: usize,
[ABACK L0092]     pub num_groups: usize,
[ABACK L0093]     pub se_bottleneck: usize,
[ABACK L0094] }
[ABACK L0095] 
[ABACK L0096] impl SEResNetConfig {
[ABACK L0097]     pub fn validate(&self) -> Result<(), &'static str> {
[ABACK L0098]         if self.num_blocks == 0 {
[ABACK L0099]             return Err("num_blocks > 0");
[ABACK L0100]         }
[ABACK L0101]         if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
[ABACK L0102]             return Err("hidden_channels % num_groups != 0");
[ABACK L0103]         }
[ABACK L0104]         Ok(())
[ABACK L0105]     }
[ABACK L0106] }
[ABACK L0107] 
[ABACK L0108] #[derive(Module, Debug)]
[ABACK L0109] pub struct SEResNet<B: Backend> {
[ABACK L0110]     input_conv: Conv1d<B>,
[ABACK L0111]     input_gn: GroupNorm<B>,
[ABACK L0112]     blocks: Vec<SEResBlock<B>>,
[ABACK L0113]     final_gn: GroupNorm<B>,
[ABACK L0114] }
[ABACK L0115] 
[ABACK L0116] impl SEResNetConfig {
[ABACK L0117]     pub fn init<B: Backend>(&self, device: &B::Device) -> SEResNet<B> {
[ABACK L0118]         let input_conv = Conv1dConfig::new(self.input_channels, self.hidden_channels, 3)
[ABACK L0119]             .with_padding(PaddingConfig1d::Same)
[ABACK L0120]             .init(device);
[ABACK L0121]         let input_gn = GroupNormConfig::new(self.num_groups, self.hidden_channels).init(device);
[ABACK L0122]         let block_cfg =
[ABACK L0123]             SEResBlockConfig::new(self.hidden_channels, self.num_groups, self.se_bottleneck);
[ABACK L0124]         let blocks = (0..self.num_blocks)
[ABACK L0125]             .map(|_| block_cfg.init(device))
[ABACK L0126]             .collect();
[ABACK L0127]         let final_gn = GroupNormConfig::new(self.num_groups, self.hidden_channels).init(device);
[ABACK L0128]         SEResNet {
[ABACK L0129]             input_conv,
[ABACK L0130]             input_gn,
[ABACK L0131]             blocks,
[ABACK L0132]             final_gn,
[ABACK L0133]         }
[ABACK L0134]     }
[ABACK L0135] }
[ABACK L0136] 
[ABACK L0137] impl<B: Backend> SEResNet<B> {
[ABACK L0138]     pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
[ABACK L0139]         let x = self.input_conv.forward(x);
[ABACK L0140]         let x = activation::mish(self.input_gn.forward(x));
[ABACK L0141]         let x = self.blocks.iter().fold(x, |acc, block| block.forward(acc));
[ABACK L0142]         let spatial = activation::mish(self.final_gn.forward(x));
[ABACK L0143]         let pooled = spatial.clone().mean_dim(2).squeeze_dim::<2>(2);
[ABACK L0144]         (spatial, pooled)
[ABACK L0145]     }
```

## Artifact 13 — hydra-train/src/model.rs lines 83-262
Source label: AMODEL
Path: hydra-train/src/model.rs
Use: treat this as evidence, not truth.
```text
[AMODEL L0083] #[derive(Module, Debug)]
[AMODEL L0084] pub struct HydraModel<B: Backend> {
[AMODEL L0085]     backbone: SEResNet<B>,
[AMODEL L0086]     policy: PolicyHead<B>,
[AMODEL L0087]     value: ValueHead<B>,
[AMODEL L0088]     score_pdf: ScorePdfHead<B>,
[AMODEL L0089]     score_cdf: ScoreCdfHead<B>,
[AMODEL L0090]     opp_tenpai: OppTenpaiHead<B>,
[AMODEL L0091]     grp: GrpHead<B>,
[AMODEL L0092]     opp_next_discard: OppNextDiscardHead<B>,
[AMODEL L0093]     danger: DangerHead<B>,
[AMODEL L0094]     oracle_critic: OracleCriticHead<B>,
[AMODEL L0095]     belief_field: BeliefFieldHead<B>,
[AMODEL L0096]     mixture_weight: MixtureWeightHead<B>,
[AMODEL L0097]     opponent_hand_type: OpponentHandTypeHead<B>,
[AMODEL L0098]     delta_q: DeltaQHead<B>,
[AMODEL L0099]     safety_residual: SafetyResidualHead<B>,
[AMODEL L0100] }
[AMODEL L0101] 
[AMODEL L0102] #[derive(Config, Debug)]
[AMODEL L0103] pub struct HydraModelConfig {
[AMODEL L0104]     pub num_blocks: usize,
[AMODEL L0105]     #[config(default = "192")]
[AMODEL L0106]     pub input_channels: usize,
[AMODEL L0107]     #[config(default = "256")]
[AMODEL L0108]     pub hidden_channels: usize,
[AMODEL L0109]     #[config(default = "32")]
[AMODEL L0110]     pub num_groups: usize,
[AMODEL L0111]     #[config(default = "64")]
[AMODEL L0112]     pub se_bottleneck: usize,
[AMODEL L0113]     #[config(default = "46")]
[AMODEL L0114]     pub action_space: usize,
[AMODEL L0115]     #[config(default = "64")]
[AMODEL L0116]     pub score_bins: usize,
[AMODEL L0117]     #[config(default = "3")]
[AMODEL L0118]     pub num_opponents: usize,
[AMODEL L0119]     #[config(default = "24")]
[AMODEL L0120]     pub grp_classes: usize,
[AMODEL L0121]     #[config(default = "4")]
[AMODEL L0122]     pub num_belief_components: usize,
[AMODEL L0123]     #[config(default = "8")]
[AMODEL L0124]     pub opponent_hand_type_classes: usize,
[AMODEL L0125] }
[AMODEL L0126] 
[AMODEL L0127] impl HydraModelConfig {
[AMODEL L0128]     pub fn summary(&self) -> String {
[AMODEL L0129]         let kind = if self.num_blocks <= 12 {
[AMODEL L0130]             "actor"
[AMODEL L0131]         } else {
[AMODEL L0132]             "learner"
[AMODEL L0133]         };
[AMODEL L0134]         format!(
[AMODEL L0135]             "{}(blocks={}, ch={})",
[AMODEL L0136]             kind, self.num_blocks, self.hidden_channels
[AMODEL L0137]         )
[AMODEL L0138]     }
[AMODEL L0139] 
[AMODEL L0140]     pub fn is_actor(&self) -> bool {
[AMODEL L0141]         self.num_blocks == 12
[AMODEL L0142]     }
[AMODEL L0143]     pub fn is_learner(&self) -> bool {
[AMODEL L0144]         self.num_blocks == 24
[AMODEL L0145]     }
[AMODEL L0146] 
[AMODEL L0147]     pub fn validate(&self) -> Result<(), &'static str> {
[AMODEL L0148]         if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
[AMODEL L0149]             return Err("hidden_channels must be divisible by num_groups");
[AMODEL L0150]         }
[AMODEL L0151]         if self.num_blocks == 0 {
[AMODEL L0152]             return Err("num_blocks must be > 0");
[AMODEL L0153]         }
[AMODEL L0154]         if self.se_bottleneck == 0 {
[AMODEL L0155]             return Err("se_bottleneck must be > 0");
[AMODEL L0156]         }
[AMODEL L0157]         if self.num_belief_components == 0 {
[AMODEL L0158]             return Err("num_belief_components must be > 0");
[AMODEL L0159]         }
[AMODEL L0160]         if self.opponent_hand_type_classes == 0 {
[AMODEL L0161]             return Err("opponent_hand_type_classes must be > 0");
[AMODEL L0162]         }
[AMODEL L0163]         Ok(())
[AMODEL L0164]     }
[AMODEL L0165] 
[AMODEL L0166]     pub fn actor() -> Self {
[AMODEL L0167]         Self::new(12).with_input_channels(INPUT_CHANNELS)
[AMODEL L0168]     }
[AMODEL L0169] 
[AMODEL L0170]     pub fn estimated_params(&self) -> usize {
[AMODEL L0171]         let h = self.hidden_channels;
[AMODEL L0172]         let se_b = self.se_bottleneck;
[AMODEL L0173]         let input_conv = self.input_channels * h * 3 + h;
[AMODEL L0174]         let gn = h * 2;
[AMODEL L0175]         let block = (h * h * 3 + h) * 2 + gn * 2 + (h * se_b + se_b) + (se_b * h + h);
[AMODEL L0176]         let backbone = input_conv + gn + block * self.num_blocks + gn;
[AMODEL L0177]         let policy = h * self.action_space + self.action_space;
[AMODEL L0178]         let value = h + 1;
[AMODEL L0179]         let score = (h * self.score_bins + self.score_bins) * 2;
[AMODEL L0180]         let tenpai = h * self.num_opponents + self.num_opponents;
[AMODEL L0181]         let grp = h * self.grp_classes + self.grp_classes;
[AMODEL L0182]         let opp_next = h * self.num_opponents + self.num_opponents;
[AMODEL L0183]         let danger = h * self.num_opponents + self.num_opponents;
[AMODEL L0184]         let oracle = h * 4 + 4;
[AMODEL L0185]         let belief_field = h * (self.num_belief_components * 4) + (self.num_belief_components * 4);
[AMODEL L0186]         let mixture_weight = h * self.num_belief_components + self.num_belief_components;
[AMODEL L0187]         let opponent_hand_type = h * (self.num_opponents * self.opponent_hand_type_classes)
[AMODEL L0188]             + (self.num_opponents * self.opponent_hand_type_classes);
[AMODEL L0189]         let delta_q = h * self.action_space + self.action_space;
[AMODEL L0190]         let safety_residual = h * self.action_space + self.action_space;
[AMODEL L0191]         backbone
[AMODEL L0192]             + policy
[AMODEL L0193]             + value
[AMODEL L0194]             + score
[AMODEL L0195]             + tenpai
[AMODEL L0196]             + grp
[AMODEL L0197]             + opp_next
[AMODEL L0198]             + danger
[AMODEL L0199]             + oracle
[AMODEL L0200]             + belief_field
[AMODEL L0201]             + mixture_weight
[AMODEL L0202]             + opponent_hand_type
[AMODEL L0203]             + delta_q
[AMODEL L0204]             + safety_residual
[AMODEL L0205]     }
[AMODEL L0206] 
[AMODEL L0207]     pub fn learner() -> Self {
[AMODEL L0208]         Self::new(24).with_input_channels(INPUT_CHANNELS)
[AMODEL L0209]     }
[AMODEL L0210] 
[AMODEL L0211]     pub fn init<B: Backend>(&self, device: &B::Device) -> HydraModel<B> {
[AMODEL L0212]         let backbone_cfg = SEResNetConfig::new(
[AMODEL L0213]             self.num_blocks,
[AMODEL L0214]             self.input_channels,
[AMODEL L0215]             self.hidden_channels,
[AMODEL L0216]             self.num_groups,
[AMODEL L0217]             self.se_bottleneck,
[AMODEL L0218]         );
[AMODEL L0219]         let heads_cfg = HeadsConfig::new()
[AMODEL L0220]             .with_hidden_channels(self.hidden_channels)
[AMODEL L0221]             .with_action_space(self.action_space)
[AMODEL L0222]             .with_score_bins(self.score_bins)
[AMODEL L0223]             .with_num_opponents(self.num_opponents)
[AMODEL L0224]             .with_grp_classes(self.grp_classes)
[AMODEL L0225]             .with_num_belief_components(self.num_belief_components)
[AMODEL L0226]             .with_opponent_hand_type_classes(self.opponent_hand_type_classes);
[AMODEL L0227]         HydraModel {
[AMODEL L0228]             backbone: backbone_cfg.init(device),
[AMODEL L0229]             policy: heads_cfg.init_policy(device),
[AMODEL L0230]             value: heads_cfg.init_value(device),
[AMODEL L0231]             score_pdf: heads_cfg.init_score_pdf(device),
[AMODEL L0232]             score_cdf: heads_cfg.init_score_cdf(device),
[AMODEL L0233]             opp_tenpai: heads_cfg.init_opp_tenpai(device),
[AMODEL L0234]             grp: heads_cfg.init_grp(device),
[AMODEL L0235]             opp_next_discard: heads_cfg.init_opp_next_discard(device),
[AMODEL L0236]             danger: heads_cfg.init_danger(device),
[AMODEL L0237]             oracle_critic: heads_cfg.init_oracle_critic(device),
[AMODEL L0238]             belief_field: heads_cfg.init_belief_field(device),
[AMODEL L0239]             mixture_weight: heads_cfg.init_mixture_weight(device),
[AMODEL L0240]             opponent_hand_type: heads_cfg.init_opponent_hand_type(device),
[AMODEL L0241]             delta_q: heads_cfg.init_delta_q(device),
[AMODEL L0242]             safety_residual: heads_cfg.init_safety_residual(device),
[AMODEL L0243]         }
[AMODEL L0244]     }
[AMODEL L0245] }
[AMODEL L0246] 
[AMODEL L0247] impl<B: Backend> HydraModel<B> {
[AMODEL L0248]     pub fn policy_logits_for(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
[AMODEL L0249]         let (_, pooled) = self.backbone.forward(x);
[AMODEL L0250]         self.policy.forward(pooled)
[AMODEL L0251]     }
[AMODEL L0252] 
[AMODEL L0253]     pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
[AMODEL L0254]         let (spatial, pooled) = self.backbone.forward(x);
[AMODEL L0255]         let oracle_input = pooled.clone().detach();
[AMODEL L0256]         HydraOutput {
[AMODEL L0257]             policy_logits: self.policy.forward(pooled.clone()),
[AMODEL L0258]             value: self.value.forward(pooled.clone()),
[AMODEL L0259]             score_pdf: self.score_pdf.forward(pooled.clone()),
[AMODEL L0260]             score_cdf: self.score_cdf.forward(pooled.clone()),
[AMODEL L0261]             opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
[AMODEL L0262]             grp: self.grp.forward(pooled.clone()),
```

## Artifact 14 — hydra-core/src/encoder.rs lines 1-200
Source label: AENC
Path: hydra-core/src/encoder.rs
Use: treat this as evidence, not truth.
```text
[AENC L0001] //! Fixed-superset observation tensor encoder for neural network input.
[AENC L0002] //!
[AENC L0003] //! Encodes the full game state into a flat `[f32; NUM_CHANNELS * 34]` array
[AENC L0004] //! (row-major) that serves as input to the Hydra SE-ResNet model.
[AENC L0005] //!
[AENC L0006] //! The currently implemented baseline channels remain intact in the first 85
[AENC L0007] //! planes. Additional Group C / Group D planes provide a fixed-shape superset
[AENC L0008] //! for search/belief and Hand-EV context, with zero-filled planes plus
[AENC L0009] //! presence-mask channels when those dynamic features are unavailable.
[AENC L0010] //!
[AENC L0011] //! Channels are grouped:
[AENC L0012] //!
[AENC L0013] //! - 0..3:   closed hand (thresholded tile counts)
[AENC L0014] //! - 4..7:   open meld hand counts (thresholded)
[AENC L0015] //! - 8:      drawn tile one-hot
[AENC L0016] //! - 9..10:  shanten masks (keep / next)
[AENC L0017] //! - 11..22: discards per player (presence, tedashi, temporal)
[AENC L0018] //! - 23..34: melds per player (chi, pon, kan)
[AENC L0019] //! - 35..39: dora indicator thermometer
[AENC L0020] //! - 40..42: aka dora flags (per suit plane)
[AENC L0021] //! - 43..61: game metadata (riichi, scores, gaps, shanten, round, honba, kyotaku)
[AENC L0022] //! - 62..84: safety channels (genbutsu, suji, kabe, one-chance, tenpai)
[AENC L0023] //! - 85..149: Group C search/belief context + presence masks + reserved slots
[AENC L0024] //! - 150..191: Group D Hand-EV context + presence mask
[AENC L0025] use crate::hand_ev::HandEvFeatures;
[AENC L0026] use crate::safety::SafetyInfo;
[AENC L0027] use crate::tile::NUM_TILE_TYPES;
[AENC L0028] 
[AENC L0029] // ---------------------------------------------------------------------------
[AENC L0030] // Constants
[AENC L0031] // ---------------------------------------------------------------------------
[AENC L0032] 
[AENC L0033] /// Baseline observation channels (public + safety).
[AENC L0034] pub const BASELINE_CHANNELS: usize = 85;
[AENC L0035] 
[AENC L0036] /// Group C search/belief context channels.
[AENC L0037] pub const SEARCH_CONTEXT_CHANNELS: usize = 65;
[AENC L0038] 
[AENC L0039] /// Group D Hand-EV channels.
[AENC L0040] pub const HAND_EV_CHANNELS: usize = 42;
[AENC L0041] 
[AENC L0042] /// First Group C search/belief channel.
[AENC L0043] pub const SEARCH_CHANNEL_START: usize = BASELINE_CHANNELS;
[AENC L0044] 
[AENC L0045] /// First Group C belief-field channel.
[AENC L0046] pub const SEARCH_BELIEF_CHANNEL_START: usize = SEARCH_CHANNEL_START;
[AENC L0047] 
[AENC L0048] /// First Group D Hand-EV channel.
[AENC L0049] pub const HAND_EV_CHANNEL_START: usize = SEARCH_CHANNEL_START + SEARCH_CONTEXT_CHANNELS;
[AENC L0050] 
[AENC L0051] /// Group C discard-level delta-Q channel.
[AENC L0052] pub const SEARCH_DELTA_Q_CHANNEL: usize = SEARCH_CHANNEL_START + 22;
[AENC L0053] 
[AENC L0054] /// Group C mixture-entropy scalar channel.
[AENC L0055] pub const SEARCH_MIXTURE_ENTROPY_CHANNEL: usize = SEARCH_CHANNEL_START + 20;
[AENC L0056] 
[AENC L0057] /// Group C mixture-ESS scalar channel.
[AENC L0058] pub const SEARCH_MIXTURE_ESS_CHANNEL: usize = SEARCH_CHANNEL_START + 21;
[AENC L0059] 
[AENC L0060] /// First Group C opponent-risk channel.
[AENC L0061] pub const SEARCH_RISK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 23;
[AENC L0062] 
[AENC L0063] /// First Group C opponent-stress channel.
[AENC L0064] pub const SEARCH_STRESS_CHANNEL_START: usize = SEARCH_CHANNEL_START + 26;
[AENC L0065] 
[AENC L0066] /// First Group C presence-mask channel.
[AENC L0067] pub const SEARCH_MASK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 29;
[AENC L0068] 
[AENC L0069] /// Final Group D Hand-EV presence-mask channel.
[AENC L0070] pub const HAND_EV_MASK_CHANNEL: usize = HAND_EV_CHANNEL_START + HAND_EV_CHANNELS - 1;
[AENC L0071] 
[AENC L0072] /// Total observation channels.
[AENC L0073] pub const NUM_CHANNELS: usize = BASELINE_CHANNELS + SEARCH_CONTEXT_CHANNELS + HAND_EV_CHANNELS;
[AENC L0074] 
[AENC L0075] /// Tiles per channel (one per tile type).
[AENC L0076] pub const NUM_TILES: usize = NUM_TILE_TYPES; // 34
[AENC L0077] 
[AENC L0078] /// Total elements in the flat observation buffer.
[AENC L0079] pub const OBS_SIZE: usize = NUM_CHANNELS * NUM_TILES;
[AENC L0080] 
[AENC L0081] // -- Channel group starts --
[AENC L0082] 
[AENC L0083] const CH_HAND: usize = 0; // 0..3   (4 channels)
[AENC L0084] const CH_OPEN_MELD: usize = 4; // 4..7   (4 channels)
[AENC L0085] const CH_DRAWN: usize = 8; // 8      (1 channel)
[AENC L0086] const CH_SHANTEN_MASK: usize = 9; // 9..10  (2 channels)
[AENC L0087] const CH_DISCARDS: usize = 11; // 11..22 (12 channels: 3 per player)
[AENC L0088] const CH_MELDS: usize = 23; // 23..34 (12 channels: 3 per player)
[AENC L0089] const CH_DORA: usize = 35; // 35..39 (5 channels)
[AENC L0090] const CH_AKA: usize = 40; // 40..42 (3 channels)
[AENC L0091] const CH_META: usize = 43; // 43..61 (19 channels)
[AENC L0092] const CH_SAFETY: usize = 62; // 62..84 (23 channels)
[AENC L0093] const CH_SEARCH: usize = SEARCH_CHANNEL_START; // 85..149 (65 channels)
[AENC L0094] const CH_HAND_EV: usize = HAND_EV_CHANNEL_START; // 150..191 (42 channels)
[AENC L0095] 
[AENC L0096] const SEARCH_BELIEF_CHANNELS: usize = 16;
[AENC L0097] const SEARCH_MIXTURE_WEIGHT_CHANNELS: usize = 4;
[AENC L0098] const SEARCH_RISK_CHANNELS: usize = 3;
[AENC L0099] const SEARCH_STRESS_CHANNELS: usize = 3;
[AENC L0100] const SEARCH_MASK_CHANNELS: usize = 4;
[AENC L0101] const SEARCH_RESERVED_CHANNELS: usize = 32;
[AENC L0102] 
[AENC L0103] const CH_SEARCH_BELIEF: usize = CH_SEARCH; // 85..100
[AENC L0104] const CH_SEARCH_MIXTURE_WEIGHT: usize = CH_SEARCH_BELIEF + SEARCH_BELIEF_CHANNELS; // 101..104
[AENC L0105] const CH_SEARCH_MIXTURE_ENTROPY: usize = CH_SEARCH_MIXTURE_WEIGHT + SEARCH_MIXTURE_WEIGHT_CHANNELS; // 105
[AENC L0106] const CH_SEARCH_MIXTURE_ESS: usize = CH_SEARCH_MIXTURE_ENTROPY + 1; // 106
[AENC L0107] const CH_SEARCH_DELTA_Q: usize = CH_SEARCH_MIXTURE_ESS + 1; // 107
[AENC L0108] const CH_SEARCH_RISK: usize = CH_SEARCH_DELTA_Q + 1; // 108..110
[AENC L0109] const CH_SEARCH_STRESS: usize = CH_SEARCH_RISK + SEARCH_RISK_CHANNELS; // 111..113
[AENC L0110] const CH_SEARCH_MASKS: usize = CH_SEARCH_STRESS + SEARCH_STRESS_CHANNELS; // 114..117
[AENC L0111] const CH_SEARCH_RESERVED: usize = CH_SEARCH_MASKS + SEARCH_MASK_CHANNELS; // 118..149
[AENC L0112] 
[AENC L0113] const CH_HAND_EV_TENPAI: usize = CH_HAND_EV; // 150..152
[AENC L0114] const CH_HAND_EV_WIN: usize = CH_HAND_EV_TENPAI + 3; // 153..155
[AENC L0115] const CH_HAND_EV_SCORE: usize = CH_HAND_EV_WIN + 3; // 156
[AENC L0116] const CH_HAND_EV_UKEIRE: usize = CH_HAND_EV_SCORE + 1; // 157..190
[AENC L0117] const CH_HAND_EV_MASK: usize = CH_HAND_EV_UKEIRE + NUM_TILES; // 191
[AENC L0118] 
[AENC L0119] /// Number of players at the table.
[AENC L0120] const NUM_PLAYERS: usize = 4;
[AENC L0121] 
[AENC L0122] /// Fixed-shape Group C search/belief context planes.
[AENC L0123] #[derive(Debug, Clone)]
[AENC L0124] pub struct SearchFeaturePlanes {
[AENC L0125]     pub belief_fields: [[f32; NUM_TILES]; SEARCH_BELIEF_CHANNELS],
[AENC L0126]     pub mixture_weights: [f32; SEARCH_MIXTURE_WEIGHT_CHANNELS],
[AENC L0127]     pub mixture_entropy: f32,
[AENC L0128]     pub mixture_ess: f32,
[AENC L0129]     pub delta_q: [f32; NUM_TILES],
[AENC L0130]     pub opponent_risk: [[f32; NUM_TILES]; SEARCH_RISK_CHANNELS],
[AENC L0131]     pub opponent_stress: [f32; SEARCH_STRESS_CHANNELS],
[AENC L0132]     pub belief_features_present: bool,
[AENC L0133]     pub search_features_present: bool,
[AENC L0134]     pub robust_features_present: bool,
[AENC L0135]     pub context_features_present: bool,
[AENC L0136] }
[AENC L0137] 
[AENC L0138] impl Default for SearchFeaturePlanes {
[AENC L0139]     fn default() -> Self {
[AENC L0140]         Self {
[AENC L0141]             belief_fields: [[0.0; NUM_TILES]; SEARCH_BELIEF_CHANNELS],
[AENC L0142]             mixture_weights: [0.0; SEARCH_MIXTURE_WEIGHT_CHANNELS],
[AENC L0143]             mixture_entropy: 0.0,
[AENC L0144]             mixture_ess: 0.0,
[AENC L0145]             delta_q: [0.0; NUM_TILES],
[AENC L0146]             opponent_risk: [[0.0; NUM_TILES]; SEARCH_RISK_CHANNELS],
[AENC L0147]             opponent_stress: [0.0; SEARCH_STRESS_CHANNELS],
[AENC L0148]             belief_features_present: false,
[AENC L0149]             search_features_present: false,
[AENC L0150]             robust_features_present: false,
[AENC L0151]             context_features_present: false,
[AENC L0152]         }
[AENC L0153]     }
[AENC L0154] }
[AENC L0155] 
[AENC L0156] // ---------------------------------------------------------------------------
[AENC L0157] // ObservationEncoder
[AENC L0158] // ---------------------------------------------------------------------------
[AENC L0159] 
[AENC L0160] /// Pre-allocated encoder buffer for the fixed-superset observation tensor.
[AENC L0161] ///
[AENC L0162] /// Reuse across turns to avoid per-turn allocation. Call [`clear`] then
[AENC L0163] /// the individual `encode_*` methods, or use [`encode`] as the one-shot
[AENC L0164] /// entry point.
[AENC L0165] #[derive(Clone)]
[AENC L0166] #[repr(C)]
[AENC L0167] pub struct ObservationEncoder {
[AENC L0168]     /// Flat buffer: `NUM_CHANNELS` channels x 34 tiles, row-major.
[AENC L0169]     buffer: [f32; OBS_SIZE],
[AENC L0170] }
[AENC L0171] 
[AENC L0172] impl ObservationEncoder {
[AENC L0173]     /// Create a new encoder with a zeroed buffer.
[AENC L0174]     #[inline]
[AENC L0175]     pub fn new() -> Self {
[AENC L0176]         Self {
[AENC L0177]             buffer: [0.0; OBS_SIZE],
[AENC L0178]         }
[AENC L0179]     }
[AENC L0180] 
[AENC L0181]     /// Zero the entire buffer.
[AENC L0182]     #[inline]
[AENC L0183]     pub fn clear(&mut self) {
[AENC L0184]         self.buffer.fill(0.0);
[AENC L0185]     }
[AENC L0186] 
[AENC L0187]     /// Zero only the channels in range `[start_ch, end_ch)` (exclusive end).
[AENC L0188]     #[inline]
[AENC L0189]     pub fn clear_range(&mut self, start_ch: usize, end_ch: usize) {
[AENC L0190]         let start = start_ch * NUM_TILES;
[AENC L0191]         let end = end_ch * NUM_TILES;
[AENC L0192]         self.buffer[start..end].fill(0.0);
[AENC L0193]     }
[AENC L0194] 
[AENC L0195]     /// Read-only view of the flat observation buffer.
[AENC L0196]     #[inline]
[AENC L0197]     pub fn as_slice(&self) -> &[f32; OBS_SIZE] {
[AENC L0198]         &self.buffer
[AENC L0199]     }
[AENC L0200] 
```

</artifacts>

<final_reminders>
Your job is not to defend the current doctrine.
Your job is not to be contrarian for sport either.
Your job is to determine the best architecture for Hydra as rigorously as possible.
It does not have to be SE-ResNet.
Assume we can code anything.
Still respect actual problem constraints, actual runtime goals, and actual compute realities.
If the answer is “keep SE-ResNet but add a dedicated history sidecar,” say that.
If the answer is “SE-ResNet actor, different learner,” say that.
If the answer is “full replacement,” say that.
If the answer is “underdetermined; run these experiments,” say that.
But do not stop before the reasoning is saturated or blocked.
</final_reminders>

# Hydra architecture blueprint

## 0. Decision

Do **not** make Hydra a pure SE-ResNet project, and do **not** replace it with a pure transformer project.

The best architecture family for Hydra is an **asymmetric multi-path system**:

* a **convolutional static-state path** over the 34-tile axis for local tile combinatorics and cheap public-structure processing,
* a **dedicated public-event history path** for discard/call/riichi order and opponent reading,
* **late-fused optional belief/search inputs** rather than treating dynamic planning outputs as the canonical state,
* **tile-aware heads** for tile-indexed outputs and **global heads** for non-tile actions and scalar predictions,
* a **materially different learner/search-side model** from the fast actor.

In concrete terms:

* **Fast actor**: keep the current 12-block SE-style 1D residual trunk as the static path, add a **small recurrent history sidecar** (default: GRU), split policy into **discard-tile spatial head + global-action head**.
* **Learner / teacher / search helper**: keep the 24-block SE-style static trunk, add a **richer event-history transformer sidecar**, and late-fuse belief/search tokens there.
* **Do not replace the 192×34 path.** Reclassify it as the **static compatibility path**, not the whole information state.

That is the best practical Hydra architecture under current objectives.
The best ideal learner/search-side architecture, if Hydra later earns more data/compute, is a **three-path hybrid**: static conv path + event-history transformer + optional entity/belief/search token path.

The key principle is simple:

> **Use convolution where Mahjong is local and structured; use sequence modeling where Mahjong is ordered and lossy under fixed tensors.**

---

## 1. Assumption ledger

| Bucket                                                   | Contents                                                                                                                                                                                                                           |         Hardness | What follows                                                                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------: | ---------------------------------------------------------------------------------------------------------------------------- |
| **A. Irreducible game facts**                            | 4-player general-sum imperfect information; multi-round placement-aware scoring; huge hidden-state uncertainty; interrupted action flow; conserved shared tile pool; strong local tile-shape structure on 34 tile types            |         **Hard** | Architecture must handle both **local tile geometry** and **temporal public history**, while respecting hidden-tile coupling |
| **B. Explicit project objectives / runtime constraints** | Maximize stable rank, support fast deployable actor, use selective search rather than universal heavy search, operate under moderate compute and real inference budgets, preserve deployability in the current Rust stack          |  **Mostly hard** | Actor latency matters; learner/search model can be richer; selective search/distillation is allowed and desirable            |
| **C. Current repo realities**                            | Live encoder is 192×34 fixed-superset; current trunk is SE-style residual conv; actor/learner are 12/24 blocks; many advanced heads exist; advanced supervision is only partially active; current policy path uses pooled features |         **Soft** | These are starting constraints for migration, not proof of optimality                                                        |
| **D. Contestable doctrine / design bets**                | “SE-ResNet is the answer,” “192×34 is the state,” “single backbone + shallow heads is best,” “actor and learner should match,” “search/belief should be input planes”                                                              | **Overturnable** | These must be proved, not inherited                                                                                          |

Bucket A comes from the game itself and the strongest external evidence: Suphx describes Riichi Mahjong as a four-player imperfect-information game with multi-round placement-aware rewards, irregular interruptions from meld/win actions, and more than (10^{48}) hidden states per information set on average. The artifacts add the other hard fact: hidden tiles sit in a conserved shared pool, so beliefs are coupled by exact tile counts rather than independently factorized. ([arXiv][1]) (Artifact AFINAL_A L0047-L0055)

---

## 2. Hard facts

### 2.1 Static tile geometry is real, frequent, and cheap to exploit

Suphx used 34-column convolutional models, added 100+ look-ahead features as 34-dimensional vectors, and explicitly said it avoided pooling because each tile column has semantic meaning. JueJong, though 1v1 rather than 4p Riichi, still stayed in the residual/CNN family and explicitly encoded the latest 24 discards **in order** rather than abandoning structure for a pure sequence backbone. ([arXiv][1])

A 1D (3)-kernel trunk over width (34) does **not** lack global static reach. With one stem conv and two (3\times1) convs per block, receptive field is

[
\text{RF} = 1 + 2(1 + 2B).
]

So:

* (B=12 \Rightarrow \text{RF}=51)
* (B=24 \Rightarrow \text{RF}=99)

Both exceed width (34).
So the static conv trunk can already integrate whole-board **static** context. If attention helps Hydra, it is not because the conv trunk “cannot see the whole board.” It is because the missing information is **ordered history** and **optional modality fusion**.

### 2.2 Ordered public history is the missing modality

The strongest evidence packet points the same way. JueJong devotes 24 feature maps to the latest 24 discards in order. Suphx uses a recurrent GRU model for game-level reward prediction across rounds. Your own artifact on Mahjong techniques identifies the gap as multi-step reasoning over discard/call chains, and explicitly flags attention over discard sequences as a plausible upgrade over pure CNN treatment. ([arXiv][1]) (Artifact ATECH L0405-L0413)

This is the central architecture fact: **Mahjong’s missing signal in fixed tile tensors is not static board context; it is event order.**

### 2.3 Search matters, but as a selective overlay, not as the actor’s identity

OLSS’s Mahjong experiments used a learned blueprint and environmental model, both based on small residual networks, and then used **pUCT** because CFR-style search was too simulation-hungry in that setting; they report meaningful gains at 1000 simulations while CFR at 5000 was still inadequate. ReBeL and Student of Games reinforce the same system pattern in imperfect-information games more broadly: strong learned blueprint + search + self-play + distillation, not “one giant inference-time planner everywhere.” ([Proceedings of Machine Learning Research][2])

So Hydra’s architecture should be designed to support **selective search and distillation**, not to make every deployable actor forward pass depend on heavy planner-state inputs.

### 2.4 Token/transformer Mahjong is plausible, but the strongest evidence still does not make it the default winner

Kanachan is the best public steelman for raw-token transformers in Riichi Mahjong: it explicitly argues that much larger datasets make more expressive models like transformers viable, represents many aspects of state as tokens and sequences instead of human-crafted planes, and frames that as a conscious trade of feature engineering for data and model scale. ([GitHub][3])

Tjong is relevant but weak evidence here because only abstract-level access was available in this session. Its abstract reports a 15M-parameter transformer with hierarchical decision-making, trained on roughly 0.5M data over 7 days, outperforming multiple baselines in its environment. That makes transformer formulations credible; it does **not** outweigh the stronger Riichi-specific conv evidence from Suphx plus the structured residual evidence from JueJong. ([Directory of Open Access Journals][4])

### 2.5 The repo itself already proves the current doctrine is not a hard constraint

The live encoder is already a **192×34** fixed-superset, not the old 85×34 monolith, and it already carries dynamic feature presence masks. The current code also exposes both `spatial` and `pooled` trunk outputs. But the current model feeds pooled features into the policy head, while Suphx explicitly avoided pooling for tile-semantic reasons. So the repo is **not** a proof that “one pooled shared trunk to shallow heads” is the best Hydra architecture; it is a partial implementation with at least one architecturally meaningful simplification still present. (AGAME L0122-L0125; AENC L0023-L0117; AMODEL L0247-L0258; ABACK L0138-L0144) ([arXiv][1])

---

## 3. Contestable doctrine

| Doctrine                                                | Verdict                                                          | Reason                                                                                                                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **“SE-ResNet is the best Hydra architecture.”**         | **Reject as whole-system claim; keep as static trunk candidate** | SE-style conv is strong for the static tile path, but it does not solve ordered history on its own                                                                  |
| **“192×34 is the state.”**                              | **Reject**                                                       | It is a useful **static compatibility layer**, not the whole information state                                                                                      |
| **“One backbone plus shallow pooled heads is enough.”** | **Reject**                                                       | Tile-indexed outputs should read spatial tile features; pooling everything is the wrong default for discard policy                                                  |
| **“Actor and learner should share architecture.”**      | **Reject**                                                       | Role constraints differ: actor needs streaming latency and robustness without search; learner can absorb richer history and optional dynamic tokens                 |
| **“Belief/search/Hand-EV should all be input planes.”** | **Split**                                                        | Safety and Hand-EV are worth explicit early structure; optional belief/search signals should move toward late fusion                                                |
| **“The exact SE flavor is sacred.”**                    | **Reject**                                                       | Even the repo differs from the older spec: code uses mean-pool SE, while older doctrine described dual-pool CBAM-style SE (ASPEC L0192-L0194 vs. ABACK L0031-L0037) |
| **“More heads are the bottleneck.”**                    | **Reject**                                                       | The artifacts already show the bigger issue is inactive supervision and lossy representation routing, not missing head count                                        |

---

## 4. Candidate-family generation

The families to compare are:

1. **Pure fixed-tensor residual conv**
2. **SE-ResNet / channel-attention conv**
3. **ConvNeXt-style / modernized conv**
4. **Pure event-sequence transformer**
5. **Pure tile-token or entity-token transformer / set-transformer**
6. **Pure recurrent or state-space event-history**
7. **Dual-path hybrid: fixed tensor + history encoder**
8. **Graph/entity/set backbone**
9. **Asymmetric actor/learner system**
10. **Adjacent formulation overlay: search/distillation/belief sidecars**

Load-bearing external evidence for this comparison is Suphx, JueJong/ACH, OLSS, Kanachan, and ConvNeXt. Tjong, graph/set/SSM papers are useful but lighter evidence here. ([arXiv][1])

---

## 5. Evaluation rubric

Weights below are explicit. Scores are coarse and only meant to separate families, not pretend statistical certainty.

| Dimension                                                          | Weight |
| ------------------------------------------------------------------ | -----: |
| Representation fit to public state + partial observability         |     15 |
| Ability to exploit tile geometry / local combinatorics             |     12 |
| Ability to capture temporal opponent modeling                      |     12 |
| Sample efficiency under Hydra-like compute                         |     12 |
| Fast-path actor latency                                            |     10 |
| Learner / search-side usefulness                                   |     10 |
| Compatibility with multi-head supervision                          |      8 |
| Robustness when dynamic search/belief features are absent or stale |      6 |
| Ease of search/oracle distillation                                 |      5 |
| Support for selective search                                       |      4 |
| Calibration potential for safety/belief                            |      3 |
| Scaling path if compute grows                                      |      3 |

Rule for interpretation: a family does **not** win on tie-breakers if it loses materially on representation fit, temporal modeling, or sample efficiency.

---

## 6. Family-by-family evaluation

These totals are coarse weighted judgments from the rubric above.

| Family                                                |     Approx. score / 100 | Best role                        | Verdict                                                |
| ----------------------------------------------------- | ----------------------: | -------------------------------- | ------------------------------------------------------ |
| 1. Pure fixed-tensor residual conv                    |                  **80** | Fast actor baseline              | Strong reserve baseline; not best overall              |
| 2. SE-ResNet / channel-attention conv                 |                  **82** | Static trunk for actor + learner | **Keep as component**, not as the whole answer         |
| 3. ConvNeXt-style conv                                |                  **74** | Later trunk challenger           | Reserve shelf only                                     |
| 4. Pure event-sequence transformer                    |                  **63** | Teacher/history module           | Reject as full backbone under current Hydra objectives |
| 5. Pure tile/entity transformer or set-transformer    |                  **72** | Learner-side research challenger | Teacher-side / reserve shelf only for now              |
| 6. Pure recurrent/state-space history family          | **66** as full backbone | Actor-side history subsystem     | Good **subsystem**, reject as full replacement         |
| 7. Dual-path hybrid (static tensor + history encoder) |                  **90** | Main backbone family             | **Best practical backbone family**                     |
| 8. Graph/entity/set backbone                          |                  **70** | Opponent/belief helper           | Subsystem or reserve shelf only                        |
| 9. Asymmetric actor/learner family                    |                  **93** | Whole system                     | **Best full-system answer**                            |
| 10. Search/distillation/belief overlay                |                     n/a | System overlay                   | Necessary overlay, not a backbone choice               |

### Pairwise dominance

* **7 dominates 1/2** because it keeps the static conv strengths and adds first-class temporal modeling at modest cost.
* **5 is the strongest non-SE challenger**, but loses to 7 on sample efficiency, migration risk, and current-budget plausibility.
* **6 beats 4 on actor practicality**, because a recurrent sidecar can be streamed incrementally, but it loses to 4 in learner-side global event interaction.
* **8 does not beat 7** because graph bias helps relations, but does not eliminate the need for ordered event modeling.
* **9 wins system-level design** because actor and learner have genuinely different jobs.

---

## 7. Steelman for SE-ResNet

If forced to choose **one single-family Hydra architecture** and ban sidecars, the best answer is still **SE-ResNet over the fixed tile tensor**, not a transformer.

Why that steelman is real:

1. **It matches the most universal signal.** Every decision uses local tile-shape reasoning; not every decision needs deep history or search.
2. **It is well-supported by strong Mahjong evidence.** Suphx’s strongest public Riichi result used convolutional networks over 34-tile columns and explicitly preserved tile-column semantics. JueJong’s strong 1v1 result also stayed in the residual/CNN family. ([arXiv][1])
3. **It is sample-efficient under moderate compute.** Kanachan’s own argument for transformers is “huge data + large expressive models,” which is not Hydra’s current budget regime. ([GitHub][3])
4. **It already solves the static-axis global-context problem.** On width 34, the static trunk’s receptive field is already global.
5. **SE itself is low-risk.** Channel reweighting is cheap and current Hydra already has it.

So the answer is **not** “SE-ResNet is wrong.”
The answer is: **SE-ResNet is only the static half of the answer.**

---

## 8. Steelman for the strongest non-SE alternative

The strongest non-SE alternative is **not** ConvNeXt, graph networks, or Mamba as a whole-agent backbone.

It is:

> **An entity-token / event-token transformer (or set-transformer hybrid) that represents tiles, players, melds, discards, dora/meta, and optional belief/search tokens in one unified token space.**

Why this is the strongest challenger:

* It can model **cross-player relations** directly.
* It can treat **ordered public history** as first-class.
* It can ingest optional belief/search tokens elegantly.
* It aligns with Kanachan’s explicit claim that raw tokenization plus huge data should let more expressive models beat feature-engineered CNNs. ([GitHub][3])
* Set-transformer style modules are a natural fit for unordered subsets such as tile multisets or meld collections, and attention-based set models were designed precisely to model interactions while preserving permutation structure. ([Cool Papers][5])
* Graph-network thinking also supports this challenger by emphasizing relational inductive bias over entities and relations. ([Google Research][6])

Why it still loses **for Hydra now**:

1. **Evidence quality is weaker.** The strongest public Riichi evidence is still conv-centric.
2. **The local tile prior matters.** A pure token model has to relearn suit-local combinatorics that conv gets almost for free.
3. **Hydra’s budget is not a “just scale it” regime.** Kanachan’s own README frames raw-token modeling as a data/compute trade. ([GitHub][3])
4. **Migration and debugging risk are much higher** in the present stack.
5. **The actual missing modality is history**, not static-tile global reach.

So this challenger belongs on the **learner-side reserve shelf**, not on the mainline actor path.

---

## 9. Red-team pass against the leading candidate

The leading candidate is the **asymmetric hybrid**. The strongest arguments against it are real:

* Maybe the current 192×34 tensor already captures enough order through recency planes and tedashi flags.
* Maybe a history sidecar improves auxiliary heads but not actual policy strength.
* Maybe the added path hurts actor latency more than it helps.
* Maybe the true missing fix is not a new path at all, but simply routing tile-indexed heads to spatial features instead of pooled ones.
* Maybe a pure token model only looks expensive because the current benchmarks are unfair.

Those objections change the rollout order, but not the family ranking:

1. **Fix tile/global head routing first** so the baseline is not artificially handicapped.
2. **Validate history with a collision benchmark** where identical static tensors map to divergent targets because history differed.
3. **Require order-sensitivity** via history-shuffle ablations.
4. **Keep actor-side history cheap and incremental** unless a small transformer proves latency-safe.

### What would have made me incorrectly choose the current doctrine by default?

Three things:

1. Treating the repo shape as proof of optimality rather than implementation history.
2. Looking at Suphx and seeing “CNNs win,” while missing that Suphx also used **separate action-type models, no pooling on tile semantics, look-ahead features, oracle guiding, and a recurrent reward model**. ([arXiv][1])
3. Treating 192×34 as the whole information state instead of the **static view** of the information state.

---

## 10. Direct answers to Q1–Q10

**Q1. What information patterns actually dominate strong Mahjong play?**
A mixture. **Local tile-shape reasoning** is the universal base; **temporal opponent modeling** is the highest-leverage missing public signal; **cross-player relational reasoning** matters but is usually mediated through public history and score context; **search-conditioned adaptation** matters on a hard minority of states, not every state.

**Q2. Which patterns need to live in the deployable fast actor?**
The actor must carry **local tile geometry**, **basic public-history opponent reading**, **explicit safety structure**, and **placement-aware value tendencies**. Deep belief/search-conditioned adaptation can stay in the learner/search/teacher stack.

**Q3. Is the best architecture single-path or multi-path?**
**Multi-path.** Static tiles and ordered public history are different modalities and should not be forced through one representation.

**Q4. Should actor and learner share architecture?**
They should share **representation ideas, event schema, and distillation interfaces**, not identical architecture.

**Q5. Is the current 192×34 tensor a strength or an anchor?**
It is a **strong compatibility layer** and becomes an anchor only if treated as the whole state.

**Q6. Is opponent-history modeling central enough to require a dedicated sequence module?**
**Yes.** At least learner-side definitely; actor-side probably yes in lightweight recurrent form.

**Q7. Preserve explicit safety and Hand-EV / belief features, or absorb them?**
Preserve **safety** and **Hand-EV** explicitly. Keep **belief/search** structured too, but move them toward **late fusion** rather than mandatory early input.

**Q8. Monolithic or modular?**
**Modular.** Static trunk + history sidecar + optional planning/belief sidecars.

**Q9. Smallest architecture leap with realistic win chance?**
First: **tile-aware spatial/global head split**.
Second: **dedicated history sidecar**, learner-first, actor-next.

**Q10. What falsifies the recommendation quickly and cheaply?**
If a hybrid cannot beat conv-only on a **same-static-tensor/different-history collision benchmark**, or if **order-shuffling the history input barely changes performance**, the history path is not earning its keep.

---

## 11. Ideal architecture vs. best practical Hydra architecture

| Role                  | Ideal if Hydra later earns more compute/data                                            | Best practical Hydra architecture now                                                            |
| --------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Fast actor**        | Static conv path + tiny event transformer or GRU + spatial/global split heads           | **12-block SE-style static trunk + cached GRU history sidecar + spatial/global split heads**     |
| **Learner / teacher** | Static conv path + event-history transformer + optional entity/belief/search token path | **24-block SE-style static trunk + event-history transformer + late-fused belief/search tokens** |
| **Search helper**     | Learner-style hybrid with richer optional entity/belief modules                         | **Learner-style hybrid first; no new full replacement backbone yet**                             |

The ideal and practical answers differ in **learner richness**, not in the family-level choice. Even with more compute, I would still keep a static conv path.

---

## 12. Recommended architecture

## 12.1 System choice

**Winner:**
**Family 9 built on Family 7.**

That means:

* **System-level winner**: asymmetric actor/learner
* **Backbone-family winner**: static tile-tensor conv path + dedicated history encoder
* **Reserve learner challenger**: optional entity-token path later
* **Rejected mainline replacements**: pure transformer actor, pure graph actor, pure SSM actor

## 12.2 Static path

Keep the current fixed-shape static path for now:

* input: current **192×34** tensor
* actor trunk: **12-block** 1D pre-activation SE-style residual conv
* learner/search trunk: **24-block** version
* keep GroupNorm-style normalization and current deployment-friendly stack

But reclassify the planes:

* **canonical early input**: public state, safety, Hand-EV
* **compatibility/optional input**: search/belief planes with masks

This is not because belief/search are unimportant. It is because optional/stale dynamic features should not define the base actor representation. (AGAME L0122-L0125; AENC L0023-L0117; AFINAL_B L0245-L0248)

## 12.3 History path

### Actor history path: recurrent, cached, incremental

Default actor choice:

* **1-layer GRU**, hidden size **128**
* sequence cap **96** public events
* persistent hidden state updated on each public event
* reset at hand boundary

Why GRU on actor:

* event histories are naturally streaming,
* hidden state can be cached and updated incrementally,
* cost is tiny relative to the conv trunk,
* latency is predictable.

A rough compute check puts a 12-block conv trunk around **165M MACs**, a tiny (T=96,d=128) 2-layer transformer sidecar around **42M MACs**, and a GRU sidecar around **9–10M MACs**. So the actor-side history module is affordable either way, but cached recurrence is the cleanest default fast path.

### Learner history path: attention over full public history

Default learner choice:

* **3-layer transformer**
* (d_{\text{model}} = 128) or (192)
* 4 heads
* seat embeddings + relative/causal position encoding
* full per-event hidden states retained for cross-attention

Why transformer on learner:

* the learner benefits from richer pairwise event interactions,
* sequence lengths are short enough that quadratic cost is not the bottleneck,
* the transformer is better suited to distilling opponent-read patterns and optional side tokens than a single recurrent summary.

Mamba/SSM stays reserve-shelf only: its main advertised advantage is long-sequence throughput, and Hydra’s public histories are not long enough for that to dominate the design decision. ([arXiv][7])

## 12.4 Event schema

Use a real event vocabulary, not only recency-weighted discard planes.

Minimum event token fields:

```rust
struct EventToken {
    kind: u8,          // draw, discard, chi, pon, kan, riichi, agari, pass, dora_reveal, score_update
    actor_rel: u8,     // 0=self, 1=left, 2=across, 3=right
    target_rel: u8,    // source/target seat when relevant
    tile: u8,          // 0..33, plus none
    aka: bool,
    tedashi: bool,
    from_riichi_player: bool,
    wall_left_bucket: u8,
    turn_index_bucket: u8,
    score_rank_bucket: u8,
    open_meld_count_bucket: u8,
    riichi_mask: u8,
}
```

This schema is enough to make **order**, **seat**, **call interruptions**, and **riichi timing** first-class.

## 12.5 Fusion

Use **late fusion**, not “concatenate everything at the input and hope.”

Let

* (Z \in \mathbb{R}^{34 \times C}): static tile features from the conv trunk
* (\bar Z \in \mathbb{R}^{C}): pooled static summary
* (H): history summary or history token matrix
* (D): optional late-fused belief/search summary

Then use:

[
\tilde Z = Z + \text{CrossAttn}(Q=W_Q Z,\ K=H,\ V=H) + g(H,D)\odot Z
]

for learner, and a lighter gated affine version for actor:

[
\tilde z_k = z_k + A(H,D) + g(H,D)\odot z_k.
]

The key is that **history modulates tile features**, not just the final scalar head.

## 12.6 Heads

This is the most important architectural correction after adding history.

### Tile-indexed heads must read spatial features

Use spatial tile embeddings (\tilde z_k) for:

* discard logits (34 normal + 3 aka)
* danger (3 \times 34)
* opponent next discard (3 \times 34)
* any belief-marginal or search-residual outputs that are tile-indexed

Example:

[
\ell_{\text{discard}}(k) = w^\top \phi([\tilde z_k,\ \text{HandEV}_k,\ \text{Safety}*k,\ h*{\text{opp}}])
]

Current Hydra code already surfaces `spatial`, but the policy path is currently fed pooled features. That should change. (AMODEL L0247-L0258)

### Global heads should read fused pooled context

Use pooled fused context for:

* riichi / chi / pon / kan / agari / ryuukyoku / pass logits
* value
* score distribution / CDF-PDF
* GRP / placement
* opponent tenpai summary
* mixture weights / meta uncertainty summaries

So the policy becomes a **factorized 46-action head**:

* **37 discard logits** from tile branch
* **9 global-action logits** from global branch

Keep the fixed 46-action interface externally. Change the internal head semantics.

### Learner-only or teacher-biased heads

Keep these primarily on learner / teacher unless proved actor-useful:

* `delta_q`
* `safety_residual`
* raw belief-field / mixture / opponent-hand-type auxiliaries

And if belief supervision is used, target **projected public-teacher belief objects**, not raw Sinkhorn potentials, which matches the reconciliation artifact’s caution. (ARECON_B L0386-L0441)

## 12.7 Safety, Hand-EV, belief, and search features

### Keep explicit early:

* **safety channels**
* **Hand-EV features**

Reason: these are cheap, structured, high-value transforms of public state or near-solved single-player subproblems. Suphx explicitly reports look-ahead features as important, and your artifact correctly treats safety encoding as high-ROI domain structure. ([arXiv][1]) (Artifact ATECH L0441-L0490)

### Move toward late fusion:

* **belief marginals**
* **search deltas**
* **ESS / entropy / robust stress**
* **search-only risk summaries**

Reason: they are optional, sometimes stale, and selective-search dependent.

Hydra already discovered the right idea with presence masks. Keep the masks; move the fusion later.

## 12.8 Search-side role

Use the learner-style hybrid as the **search blueprint/value/prior model**.

Do **not** make raw search dependence a mandatory actor input.
Do **distill** search residuals and policy improvements into the actor.

This is aligned with both the Hydra artifacts and external imperfect-information search systems. ([Proceedings of Machine Learning Research][2])

---

## 13. Reject, defer, keep

**Reject now**: pure transformer actor replacement; pure graph/entity backbone as mainline; identical actor and learner; pooled-only policy head; universal early-fusion dependence on search/belief features.

**Defer**: ConvNeXt-style trunk rewrite; learner-only entity-token challenger; Mamba/SSM history sidecar; graph-based opponent/belief helper.

**Keep**: SE-style static trunk; explicit safety channels; explicit Hand-EV; selective search; search distillation; structured belief modules outside the actor core.

---

## 14. Decisive experiment matrix

These are the minimum experiments that actually decide the remaining uncertainty.

| ID     | Question                                                                    | Compare                                                                     | Budget                           | Proposed pass / fail gate                                                                                            |
| ------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **E0** | Is the current head routing itself leaving strength on the table?           | pooled policy head vs **spatial discard + global action** split, same trunk | cheapest                         | Pass if discard CE or matched-latency self-play improves; fail kills head-rewrite priority                           |
| **E1** | Does history contain actor-useful information beyond current static tensor? | conv-only vs conv+GRU(actor) vs learner+history                             | cheap offline + short self-play  | Pass only if hybrid wins on **collision benchmark** and temporal-slice danger/opp-next/policy metrics                |
| **E2** | Is the actor sidecar better as GRU or tiny transformer?                     | same trunk, same event schema                                               | medium                           | Pick transformer only if it clears actor latency gate with measurable strength gain; otherwise GRU                   |
| **E3** | Should belief/search stay early or move late?                               | early planes vs late tokens vs both                                         | medium                           | Pass if late fusion is more robust under absent/stale dynamic features                                               |
| **E4** | Is a full replacement transformer actually better?                          | matched-param pure entity/tile transformer vs hybrid                        | medium-high, one challenger only | Promote only if it wins offline temporal slices **and** matched-budget self-play **and** latency-adjusted deployment |
| **E5** | Is ConvNeXt-style modernization worth a trunk rewrite?                      | matched-param SE trunk vs modernized conv trunk                             | optional                         | Only pursue if hybrid is already validated and trunk still looks like the bottleneck                                 |

### The two cheapest falsifiers

**Falsifier A: static-collision benchmark**

* Zero dynamic search/belief channels.
* Hash the remaining actor input.
* Collect clusters where the same static tensor appears with materially different targets because history differed.
* If the history model does **not** win here, it is not solving the problem it was added for.

**Falsifier B: order-shuffle ablation**

* Keep event multiset and event identities fixed.
* Randomly shuffle order in the last (N) public events.
* If performance barely changes, the history module is not using order and should be killed or simplified.

---

## 15. Migration blueprint

The architecture decision does **not** require restarting Hydra from zero. It changes what gets built next.

### Phase 1 — correct the existing trunk/head interface before family expansion

Keep the current supervision-first execution order from reconciliation, but make the baseline architecturally honest.

Touch points:

* `hydra-train/src/model.rs`
  Add `DiscardTileHead`, `GlobalActionHead`, `FusionBlock`, `HistoryEncoder`.
* `hydra-train/src/heads.rs`
  Route tile-indexed heads from spatial features.
* `hydra-train/src/backbone.rs`
  Keep current SE-style trunk for now.
* `hydra-train/src/training/losses.rs`
  Keep optional-head gating explicit.

This phase alone may yield real strength, because current policy pooling is likely too lossy for tile-indexed actions. (AMODEL L0247-L0258) ([arXiv][1])

### Phase 2 — add event-history plumbing, learner first

Touch points:

* `hydra-core/src/bridge.rs`
  Emit incremental `EventToken`s from public game events.
* `hydra-train/src/data/sample.rs`
  Add padded event sequences + masks to batches.
* `hydra-train/src/data/mjai_loader.rs`
  Reconstruct event histories from logs; create temporal benchmark slices.
* `hydra-train/src/model.rs`
  Add learner-side transformer history encoder and late fusion.

Do **not** break the 192×34 interface yet.

### Phase 3 — distill history-aware learner into lightweight actor

Touch points:

* `hydra-train/src/model.rs`
  Actor uses GRU history sidecar.
* `hydra-train/src/inference.rs`
  Cache recurrent hidden state across public events.
* `hydra-core/src/game_loop.rs` or equivalent runtime path
  Reset hidden state on hand start; update on each public event.

This phase is where actor/learner asymmetry becomes structural rather than just “12 blocks vs 24 blocks.”

### Phase 4 — move optional dynamic features to late fusion

Touch points:

* `hydra-core/src/encoder.rs`
  Keep current planes for compatibility, but stop treating them as semantically canonical.
* `hydra-train/src/model.rs`
  Add `BeliefSearchAdapter` that turns optional search/belief summaries into late-fused tokens or summaries.
* `losses.rs` / distillation code
  Train with feature dropout and presence-gated fusion.

### Phase 5 — reserve challengers only if the hybrid validates

Only after the hybrid wins E0–E3:

* learner-only pure entity/token challenger
* ConvNeXt trunk challenger
* graph-based belief helper
* SSM history challenger

---

## 16. Revisit triggers

| Trigger                                                                              | What changes                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Hydra gets **much larger data** (tens of millions of strong rounds with good labels) | Re-open pure token/entity learner challenger                         |
| Actor latency budget loosens or batching changes materially                          | Re-run GRU vs tiny transformer actor sidecar                         |
| Dynamic search/belief features become cheap, ubiquitous, and reliable                | Strengthen late-fused planning adapter                               |
| Hybrid fails collision benchmark / order-ablation tests                              | Drop or learner-limit the history path                               |
| Pure entity/token transformer wins matched experiments                               | Re-open conv-trunk assumption                                        |
| Tile-aware head rewrite alone captures most of the gain                              | Delay history-sidecar migration on actor, keep it learner-side first |

---

## 17. Final recommendation

**Best full-system architecture for Hydra:**
**Asymmetric dual-path hybrid.**

* **Keep** an SE-style 1D residual trunk over the 192×34 static path.
* **Add** a dedicated public-history encoder.
* **Use GRU on the actor, transformer on the learner.**
* **Split policy and other tile-indexed heads into spatial tile heads + global heads.**
* **Keep safety and Hand-EV explicit.**
* **Move belief/search toward late fusion.**
* **Do not force actor and learner to match.**
* **Do not make pure transformer or pure graph backbones the mainline now.**

The strongest concise statement is:

> **Hydra should stop asking one pooled conv trunk to do three different jobs.**
> Use **conv for static tile structure**, **sequence modeling for public history**, and **late-fused side modules for belief/search**.

[1]: https://arxiv.org/pdf/2003.13590.pdf "https://arxiv.org/pdf/2003.13590.pdf"
[2]: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf "https://proceedings.mlr.press/v202/liu23k/liu23k.pdf"
[3]: https://github.com/Cryolite/kanachan "https://github.com/Cryolite/kanachan"
[4]: https://doaj.org/article/59647e6aed75405f880c94040d2b053a "https://doaj.org/article/59647e6aed75405f880c94040d2b053a"
[5]: https://papers.cool/arxiv/1810.00825 "https://papers.cool/arxiv/1810.00825"
[6]: https://research.google/pubs/relational-inductive-biases-deep-learning-and-graph-networks/ "https://research.google/pubs/relational-inductive-biases-deep-learning-and-graph-networks/"
[7]: https://arxiv.gg/abs/2312.00752 "https://arxiv.gg/abs/2312.00752"
