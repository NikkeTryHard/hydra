# Hydra breakthrough prompt 1 — technical ceiling

Attach this zip to the model:
- `hydra_breakthrough_docs_pack.zip`

Zip structure the model should expect:
- core design docs under `research/design/`
- runtime summary under `docs/`
- infra summary under `research/infrastructure/`
- prior answer archive under `research/agent_handoffs/prior_answers/`
- prompt files under `research/agent_handoffs/prompts/`

The zip should be sufficient by itself. Use raw links only if the attachment is inaccessible or corrupted.
For this task, do not go looking for the thin-source validation pack unless the docs pack is missing and you have a specific reason to believe source validation is necessary.

You are a research advisor for Hydra, a Riichi Mahjong AI project trying to become stronger than LuckyJ.

Your task is to identify the **single strongest technical breakthrough path** still available inside Hydra's reconciled architecture.

<memo_mode>
- Write in a polished, professional memo style.
- Prefer exact, evidence-backed conclusions over generic hedging.
- Synthesize across sources rather than summarizing each one separately.
</memo_mode>

<output_contract>
- Return exactly the sections requested, in the requested order.
- Keep the answer compact but information-dense.
- Do not repeat the prompt or restate settled assumptions unless they change the conclusion.
</output_contract>

<verbosity_controls>
- Prefer concise, high-density writing.
- Avoid filler, repetition, and generic motivation.
- Do not shorten so aggressively that formulas, evidence, or failure modes disappear.
</verbosity_controls>

<research_mode>
- Work in 3 passes:
  1. Plan: identify the 3-5 technical questions that matter most for the breakthrough candidate.
  2. Retrieve: collect evidence from the provided package and follow 1-2 second-order leads if needed.
  3. Synthesize: choose one main path, resolve contradictions, and produce the final memo.
- Stop only when more searching is unlikely to change the conclusion.
</research_mode>

<citation_rules>
- Only cite sources available in the provided package or explicitly supplied links.
- Never fabricate citations, URLs, or quote spans.
- Attach citations to the specific claims they support.
</citation_rules>

<grounding_rules>
- Base claims only on provided documents, supplied links, or explicit evidence gathered during the task.
- If a claim is an inference rather than directly stated, label it as an inference.
- If sources conflict, state the conflict explicitly and resolve it.
</grounding_rules>

<completeness_contract>
- The task is incomplete until you have:
  1. named one main breakthrough path,
  2. specified its technical mechanism,
  3. identified failure modes,
  4. proposed a decisive cheap test,
  5. explained why the main alternatives lose.
</completeness_contract>

<verification_loop>
- Before finalizing, check:
  - Did you pick one main breakthrough rather than a menu?
  - Are the key claims grounded in the provided materials?
  - Are formulas/thresholds concrete enough to guide implementation?
</verification_loop>

<dig_deeper_nudge>
- Do not stop at the first plausible technical upgrade.
- Look for second-order constraints, hidden costs, and failure-triggering assumptions.
</dig_deeper_nudge>

Use these as primary sources:
- `research/design/HYDRA_FINAL.md`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/design/OPPONENT_MODELING.md`
- `docs/GAME_ENGINE.md`
- `research/design/SEEDING.md`
- `research/design/TESTING.md`
- prior answer files in `research/agent_handoffs/prior_answers/`

Raw GitHub fallback references:
- `HYDRA_FINAL.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- `HYDRA_RECONCILIATION.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md
- `IMPLEMENTATION_ROADMAP.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/IMPLEMENTATION_ROADMAP.md
- `OPPONENT_MODELING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/OPPONENT_MODELING.md
- `GAME_ENGINE.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md
- `SEEDING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/SEEDING.md
- `TESTING.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/TESTING.md
- `ANSWER_1-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_1-1.md
- `ANSWER_3-1.md` — https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/prior_answers/ANSWER_3-1.md

Assume these are already settled:
- unified belief stack = Mixture-SIB + CT-SMC
- Hand-EV before deeper AFBS
- AFBS is selective/specialist, not broad default runtime
- DRDA/ACH is off the critical path
- supervision-first before search-expansion-first

Your job:
1. Identify the highest-upside remaining breakthrough that is still compatible with the reconciled active path.
2. Explain why it could matter specifically in Mahjong.
3. Give the exact technical shape:
   - formulas
   - update rules
   - thresholds
   - approximations
   - failure modes
4. Give the cheapest meaningful experiment that would falsify or support it.
5. Compare it against the best obvious alternative and say why your chosen path wins.

Constraints:
- Do not inspect code unless a raw source file is explicitly included in the package.
- Do not propose broad architecture sprawl.
- Do not give a menu of many equal ideas.
- Pick one main breakthrough path and defend it.

Output format:
1. Executive claim
2. Why this is the best remaining breakthrough
3. Technical specification
4. Failure modes
5. Cheapest decisive test
6. Why the main alternatives lose

Success means your answer is concrete enough that a coding agent could start designing toward it immediately.


## 1. Executive claim

The strongest remaining breakthrough path is **information-state-correct belief distillation**: use the settled **Mixture-SIB + CT-SMC** stack to generate **public-teacher posterior targets**, and train Hydra’s existing `belief_fields` / mixture surfaces on **Sinkhorn-projected posteriors with gauge fixing**, not on raw Sinkhorn fields and not on realized hidden allocations. This is the highest-ceiling path that still fits the reconciled architecture because it is supervision-first, uses already-existing model surfaces and encoder channels, treats CT-SMC as a label source rather than a new search project, and strengthens the deployable policy everywhere rather than only in rare search slices. (HYDRA_FINAL.md:35-36, 115-117, 123-157; HYDRA_RECONCILIATION.md:105-118, 175-194, 446-453, 474-480; GAME_ENGINE.md:122-124)

## 2. Why this is the best remaining breakthrough

This path wins because Mahjong is unusually belief-sensitive and unusually belief-teachable. Hydra’s own design says the hidden allocation is a fixed-margin (34\times 4) contingency table with per-tile row sum at most 4, which makes exact CT-style sampling cheap enough to use as a strong teacher; the docs estimate about (4.0)M DP ops and sub-millisecond Rust runtime. At the same time, the correlation diagnostic shows late-hand dependence is not small: (|\rho|\approx 0.087) at (H=50) and (|\rho|\approx 0.190) at (H=25). That is exactly the regime where first-moment beliefs stop being good enough and posterior shape begins to drive real decisions. (HYDRA_FINAL.md:139-161)

It also compounds across the whole active path. AFBS nodes are explicitly belief-conditioned; danger composition in the prior work is belief-factorized; and the stronger Hand-EV proposal is supposed to average over CT-SMC worlds rather than a single marginal count vector. The live encoder already reserves fixed-shape search/belief and Hand-EV planes. So a sharper amortized posterior is not “one more auxiliary”; it is the common substrate underneath defense, offense, and selective search. By contrast, Hand-EV realism is a strong local multiplier, but it is still downstream of belief quality. (HYDRA_FINAL.md:183-185; ANSWER_1-1.md:246-258, 328-339; GAME_ENGINE.md:122-124; HYDRA_RECONCILIATION.md:121-132)

The key doctrinal point is that Hydra’s deployable targets must live on the **information state**, not the hidden state. `HYDRA_FINAL.md` makes that a design principle, and `ANSWER_2.md` is explicit that supervising belief heads from the realized hidden world is the central conceptual trap. There is a mild source conflict because `ANSWER_3-1.md` allows reconstructed hidden-state fallback; I would resolve that conflict in favor of `HYDRA_FINAL` P2 and `ANSWER_2`: use realized hidden allocations only for **evaluation** and for fitting/calibrating the teacher, never as the direct student target. (HYDRA_FINAL.md:35-36; ANSWER_2.md:171-245; ANSWER_3-1.md:407-421)

## 3. Technical specification

**Teacher.** For each training state (I_t), run a **public** teacher with the settled unified update:
[
w'*\ell \propto w*\ell, p_\phi(e_t\mid I_t,B_\ell,\ell), \qquad
\alpha'_p \propto \alpha_p, L(e_t\mid X^{(p)},I_t).
]
Use (L=4) components and (P=512) particles for offline labels, (P=256) for online/teacher-side generation. Resample when (\mathrm{ESS}<0.4P). Until the event model is better, use the hybrid likelihood
[
\log L
======

1.0\log p_{\text{next}}
+0.5\log p_{\text{tenpai}}
+0.5\log h_{\text{call}}
+4.0\log \mathbf 1[\text{legal}],
]
with (\Delta\log L) clipped to ([-10,0]). This keeps the breakthrough inside the existing belief stack rather than creating a second one. (HYDRA_FINAL.md:131-157; ANSWER_1-1.md:184-220; HYDRA_RECONCILIATION.md:250-258, 474-480)

**Stage A: projected-marginal supervision.** First train only the mixture-averaged projected marginal:
[
\bar P^*(z\mid k)=\sum_p \alpha_p \frac{X^{(p)}*{kz}}{r_t(k)}.
]
Let the student emit fields (F*\theta^{(\ell)}) and mixture logits; project each field through Sinkhorn,
[
\hat B_\ell=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t), \qquad
\hat w=\mathrm{softmax}(a_\theta),
]
and define
[
\bar P_\theta(z\mid k)=\sum_{\ell=1}^L \hat w_\ell \frac{\hat B_\ell(k,z)}{r_t(k)}.
]
Train with
[
L_{\text{marg}}=\sum_k r_t(k),\mathrm{KL}!\big(\bar P^*(\cdot\mid k),|,\bar P_\theta(\cdot\mid k)\big).
]
Do **not** regress raw (F_\theta); the identifiable object is the projected belief, not the unconstrained field. (HYDRA_FINAL.md:125-133; ANSWER_3-1.md:407-419)

**Stage B: modal mixture supervision.** After Stage A is stable, activate component-wise supervision only on genuinely multimodal states. Cluster teacher particles with weighted k-medoids into (L=4) modes:
[
w_\ell^*=\sum_{p\in C_\ell}\alpha_p,\qquad
B_\ell^*(k,z)=\frac{1}{w_\ell^*}\sum_{p\in C_\ell}\alpha_p\frac{X^{(p)}*{kz}}{r_t(k)}.
]
Align teacher and student components by Hungarian assignment with
[
C*{ij}
======

\sum_k r_t(k),\mathrm{KL}!\big(P_i^*(\cdot\mid k),|,\hat P_j(\cdot\mid k)\big)
+
0.25\big(\log(w_i^*+10^{-6})-\log(\hat w_j+10^{-6})\big)^2,
]
[
L_{\text{mix}}=\min_{\sigma\in S_4}\sum_i C_{i,\sigma(i)}.
]
**Inference:** only emit `mixture_weight_target` when the teacher mode effective size
[
N_{\text{mode}}=\frac{1}{\sum_\ell (w_\ell^*)^2}
]
is at least (1.5); otherwise leave `mixture_weight_target=None` and keep Stage A only. That is the cleanest way to respect the reconciliation memo’s “bring belief / mixture targets online only where labels are credible” constraint. If a field-space stabilizer is needed, use the gauge-fixed row logit
[
g_{k,z}=\log(B_{k,z}+10^{-8})-\frac14\sum_{z'}\log(B_{k,z'}+10^{-8}),
]
not raw field regression. (ANSWER_2.md:206-245; ANSWER_3-1.md:423-437; HYDRA_RECONCILIATION.md:405-411)

**Activation and boundaries.** Use explicit presence-gated losses:
[
m_j=\mathbf 1[\text{tensor exists} \land \text{finite} \land \text{sane range}],
\qquad
\tilde L_j=\frac{L_j}{\mathrm{EMA}(L_j)+10^{-6}},
]
[
L_{\text{total}}
================

L_{\text{base}}
+
m_B,w_B(t),\tilde L_{\text{marg}}
+
m_M,w_M(t),\tilde L_{\text{mix}}.
]
Use (w_B^{\max}=0.02), (w_M^{\max}=0.01), and cap auxiliary gradients at (0.35\times) the core gradient norm. **Inference:** reuse the trust logic already proposed for search labels and require teacher (\mathrm{ESS}/P \ge 0.6) before emitting belief targets. Implementation should stay narrow: `sample.rs` / `mjai_loader.rs` carry optional belief targets with explicit provenance, `losses.rs` adds the gated losses, `model.rs` stays unchanged, and `ct_smc.rs` is exposed only as a teacher/label source. The same posterior can later feed the existing `opponent_hand_type` head with soft targets, but that is a second activation, not the first proof. (ANSWER_3-1.md:357-395, 480-507; ANSWER_2.md:247-266; HYDRA_RECONCILIATION.md:400-412, 446-453, 474-480)

## 4. Failure modes

The first failure mode is **wrong supervision object**: if Hydra regresses raw Sinkhorn fields or the realized hidden world, it will optimize a non-identifiable or non-deployable target and can look “better” offline while teaching the wrong representation. The second is **weak teacher posterior**: if CT-SMC plus the event model is still too blurry, the student just learns noise; watch for posterior NLL failing to improve, median ESS dropping below (0.3P), or no pairwise-MI gain. The third is **mode thrash or collapse**: forcing four components on effectively unimodal states produces assignment noise; if early mixture collapse exceeds about 15% before late hand, keep only Stage A. The fourth is **auxiliary loss takeover**: if the aux/core gradient ratio breaches (0.35) for sustained windows, or belief metrics improve while duplicate play stays flat, the breakthrough is not paying for itself. Late-wall label corruption is a fifth, narrower risk; keep live/dead-wall refinement late-only ((wall\le10)) so the always-on belief target stays cheap and semantically clean. (ANSWER_1-1.md:222-244, 291-310; ANSWER_3-1.md:357-395, 681-697)

## 5. Cheapest decisive test

Run a **three-arm ablation on the same model surface**: A = no belief auxiliary, B = naive raw-field regression, C = the proposed projected/gauge-fixed posterior distillation. For the first test, precompute a fixed offline label cache with the public teacher rather than putting CT-SMC generation inside the training loop. Evaluate on fixed held-out suites covering hard defense, hand-building, last-10-draw endgames, and South-4 close-placement states. The key offline metrics are rowwise posterior NLL against the revealed hidden allocation **for evaluation only**, pairwise MI calibration, danger ECE in riichi and multi-threat buckets, and hard-state policy agreement. **Inference:** treat “support” as C beating B by at least ~2% on rowwise posterior NLL and ~10% on MI-calibration error with no danger-ECE regression. (HYDRA_FINAL.md:159-161; ANSWER_3-1.md:305-313, 643-670)

Then run the cheapest online discriminator Hydra already standardizes: **1,000 duplicate seed sets × 4 seat rotations = 4,000 games** against the current mainline. Support means C shows a positive rank-point trend with no 4th-place or deal-in regression; falsification means raw-field regression ties or beats it offline, or C improves offline belief metrics but is non-positive in duplicate play. Only if the 4,000-game quick gate is positive should this path graduate to the full 200,000-game duplicate tier. (INFRASTRUCTURE.md:696-712; ANSWER_3-1.md:672-697)

## 6. Why the main alternatives lose

The best obvious alternative is **Hand-EV realism**. It is real, it is cheaper, and it should still be the next local multiplier. But it loses as the single strongest breakthrough because it is **narrower and upstream-limited by belief quality**: the current interface already consumes `belief_remaining`, and the stronger Hand-EV proposal explicitly says to average over CT-SMC worlds rather than a single marginal count vector. So Hand-EV without sharper belief distillation is still asking a weak posterior to carry offense. Better belief supervision improves Hand-EV too; the reverse is not true. (HYDRA_RECONCILIATION.md:121-132; IMPLEMENTATION_ROADMAP.md:779-785; ANSWER_1-1.md:328-423)

**Endgame exactification** loses because its trigger is intentionally narrow ((wall\le10) with threat signals): it is a high-leverage specialist, not an always-on representation gain. **Trust-gated AFBS/ExIt** loses because AFBS is explicitly later, harder, and still not a full public-belief runtime; without a sharper posterior, its labels remain expensive and noisy. **Robust-opponent / exploitation layers** lose because the reconciliation memo itself places them downstream of better targets and better local evaluators. In ceiling terms, belief distillation is the one path that improves the baseline actor, the feature planes, the search substrate, and the offensive oracle at once. (HYDRA_FINAL.md:202-216; HYDRA_RECONCILIATION.md:134-159; ANSWER_3-1.md:335-343)
