Inspecting exact code block diffs in compressed file. Then patch only those blocks.Found original code blocks. Reading matching compressed sections now.Compressed file truncates before first code block. Restoring exact missing code blocks in place from original.<combined_run_record run_id="answer_3" variant_id="agent_answers_mixed_transcript" schema_version="1">
<metadata>
<notes>Mixed transcript-style file from agent_answers with both prompt + answer bodies.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="agent_answers/ANSWER_3.md" extracted_from="mixed_transcript">
<![CDATA[You deep-thinking research + design advisor for Hydra, Riichi Mahjong AI project targeting LuckyJ-level or stronger play.

I attached:
    - `hydra_agent_handoff_docs_only.zip` — primary source material
    - `deep_agent_20_pdfs` — primary paper/reference package

Use docs zip first. If cannot access zip, use raw GitHub markdown links I provide separately. Do NOT rely on normal GitHub browsing/search or generic web search to reconstruct project context.

Job in this prompt is NOT inspect source code and NOT write or integrate code. Job is look at Hydra’s CURRENT PLAN and make it:
    1. stronger,
    2. more novel,
    3. more likely produce real breakthroughs in Mahjong AI,
    4. still grounded enough for separate coding agent to implement later.

I do NOT want generic brainstorming. I want hard, evidence-backed redesign/upgrade pass on plan.

Read docs as coherent program, especially:
    - `research/design/HYDRA_FINAL.md`
    - `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md` (historical; removed later, see `combined_all_variants/README.md`)
    - `research/design/OPPONENT_MODELING.md`
    - `research/design/TESTING.md`
    - `research/infrastructure/INFRASTRUCTURE.md`
    - `docs/GAME_ENGINE.md`
    - `research/design/SEEDING.md`
    - plus any evidence/comparison docs that materially affect conclusions

Treat old `research/BUILD_AGENT_PROMPT.md` reference here as historical execution-rigor context only; for current routing chain, use `combined_all_variants/README.md` and live authority docs.

    ## What I want you to do
Critique CURRENT Hydra plan, then propose how make it substantially stronger + more breakthrough-oriented.

Specifically:
    ### Part 1 — Critique the current plan
Identify where current Hydra plan is:
    - too derivative,
    - too incremental,
    - too fragile,
    - too underspecified,
    - too compute-inefficient,
    - too likely underperform despite sounding ambitious,
    - or missing real bottleneck for beating LuckyJ-level systems.
Do not say “this is good” or “this is ambitious.” I want hardest critique.

    ### Part 2 — Propose stronger and more novel directions
Propose best high-upside improvements to CURRENT plan.

These should not be random ideas. They should be:
    - evidence-backed when possible,
    - justified from Mahjong-specific or adjacent imperfect-information-game research,
    - realistically compatible with Hydra’s overall constraints,
    - strong enough to plausibly matter for actual strength.

I especially want search for improvements in areas like:
    - belief modeling,
    - search / ExIt / pondering,
    - opponent modeling,
    - endgame solving,
    - offensive oracle features,
    - uncertainty calibration,
    - training paradigms,
    - self-play / league design,
    - auxiliary heads and target generation,
    - exploitability vs robustness tradeoffs,
    - multi-timescale planning,
    - cross-disciplinary techniques unusually well-suited to Mahjong.

    ### Part 3 — Find potential breakthroughs, not just upgrades
Explicitly identify candidate “breakthrough bets”:
    - ideas not standard baseline upgrades,
    - ideas that could create real edge in Mahjong,
    - ideas maybe underexplored publicly,
    - ideas risky but high-upside.
For each breakthrough bet, tell me:
    - why it might matter,
    - what evidence supports it,
    - what assumptions it relies on,
    - why it might fail,
    - how test it cheaply before full commitment.

    ### Part 4 — Fill in what docs did not specify
For any major proposal, fill missing technical details docs leave abstract:
    - formulas,
    - objective functions,
    - update rules,
    - priors,
    - thresholds,
    - hyperparameters,
    - approximate algorithms,
    - fallback approximations,
    - calibration procedures,
    - evaluation metrics,
    - ablation structure.
If current docs vague in exactly wrong places, point that out and fix it.

    ## Constraints
    - Do NOT inspect source code.
    - Do NOT pretend you implemented or validated anything.
    - Do NOT give broad generic summaries of Mahjong AI history unless directly relevant.
    - Do NOT recommend things that obviously blow up latency/compute without addressing feasibility.
    - Do NOT rely on AGPL code or impl borrowing.
    - Keep proposals compatible with separate coding agent implementing them later.

Assume separate coding agent will take response and implement strongest parts.
    ## How to reason about evidence

Use strict evidence hierarchy:
    1. direct Mahjong evidence,
    2. direct imperfect-information game AI evidence,
    3. adjacent multiplayer/search/belief modeling evidence,
    4. cross-disciplinary evidence that transfers unusually well.

When evidence weak, say so clearly.
When idea speculative, quantify that.
When you think something novel but unproven, separate that from evidence-backed recommendations.

    ## Required output format
Give answer in this structure:
    ### 1. Executive verdict
    - Is current Hydra plan likely strong enough as-is?
    - Where is it strongest?
    - Where is it most likely fail?
    - How much genuinely differentiated vs mostly sophisticated recombination?

    ### 2. Top weaknesses in the current plan
Rank most important weaknesses or blind spots.
    ### 3. Best upgrades to the current plan
For each:
    - concept
    - why it helps
    - evidence basis
    - exact technical proposal
    - feasibility
    - expected upside
    - risk
    ### 4. Breakthrough bets
Give best 3–7 high-upside ideas.
For each:
    - novelty level
    - evidence level
    - why it could matter specifically in Mahjong
    - minimum viable experiment
    - what success would look like

    ### 5. Concrete technical fill-ins
Where current docs too vague, provide:
    - formulas
    - losses
    - update rules
    - thresholds
    - hyperparameters
    - pseudocode / compact illustrative code snippets
    - interface/data-flow guidance for coding agent

    ### 6. Recommended revised Hydra research agenda
Give revised prioritized roadmap:
    - must-have
    - strong multipliers
    - speculative bets
    - likely dead ends / not worth complexity
    ### 7. Evaluation plan
Tell me how know revised plan better:
    - ablations
    - matchups
    - metrics
    - failure signals
    - stopping criteria
    ### 8. Final recommendation
If you had reshape Hydra into strongest + most novel feasible version of itself, what exact direction would you choose and why?

    ## Style requirements
I want:
    - depth,
    - specificity,
    - formulas,
    - precise proposals,
    - evidence-backed reasoning,
    - clear separation between proven, plausible, and speculative ideas.
I do NOT want:
    - fluff,
    - generic “future work,”
    - vague references to “using transformers” or “more compute” without real reasoning,
    - shallow novelty for novelty’s sake.

Goal: help make Hydra not more complete, but more genuinely formidable + more likely discover real Mahjong breakthroughs.

Be willing say parts of current Hydra plan should be simplified, replaced, or dropped if elegant but unlikely produce real strength. I care more about actual breakthrough potential than attachment to
current design.
</start-prompt>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_answers/ANSWER_3.md" extracted_from="mixed_transcript">
<![CDATA[### 1. Executive verdict

Treated docs zip + paper bundle as primary record, using attached reference map only as index.

  * **Is current Hydra plan likely to be strong enough as-is?**
No for stated target. As written, could plausibly produce strong **Mortal++ / Suphx-inspired** system, but I would **not** be confident it reaches **LuckyJ-level or stronger** under stated compute + current design coherence. Biggest reason not lack of ideas; plan spreads novelty budget across too many fronts while real bottleneck stays under-specified. (`HYDRA_FINAL.md` §4.2, §11, §16; `COMPUTE_FEASIBILITY.md` §8; `INFRASTRUCTURE.md` §Compute Budget)

  * **Where is it strongest?**
Hydra strongest where most Mahjong-specific:

    1. **CT-SMC exact contingency-table sampler** exploiting fixed-margin tile allocation structure,
    2. idea that **pondering + ExIt + Search-as-Feature** should amortize expensive reasoning into fast policy,
    3. explicit emphasis on **placement-aware value** + opponent-aware defense.
Those parts feel like real ceiling-raisers, not polish. (`HYDRA_FINAL.md` §5.5, §9, §10, §12)

  * **Where is it most likely to fail?**
In three places:

    1. **event-likelihood / opponent-action model** supposed make belief state correct,
    2. decision make **DRDA-wrapped ACH** load-bearing despite weak direct evidence in 4-player neural-scale settings,
    3. attempt run broad **AFBS everywhere** instead of highly selective teacher/search regime. (`HYDRA_FINAL.md` §5, §7, §11; `rnad_drda_report.md`; `ACH_RESEARCH.md`; `multiplayer_search_research.md`)

  * **How much is genuinely differentiated vs mostly sophisticated recombination?**
My read roughly **one-third genuinely differentiated, two-thirds sophisticated recombination**.
Differentiated core = CT-SMC + belief/search coupling + amortized pondering/ExIt.
Rest = intelligent recombination of **Suphx** (oracle guiding, placement-aware reward prediction), **Mortal** (Mahjong-specific engineering + compact inference), **LuckyJ/ACH/OLSS** (self-play + search-as-feature), and general imperfect-information-game ideas. Not criticism by itself. Problem: current plan behaves as if all layers will compound cleanly. They probably will not. (`HYDRA_FINAL.md` §15; `LUCKYJ_PROPOSAL.md`; `MORTAL_ANALYSIS.md`; Suphx 2020; ACH 2022; OLSS 2023)

---

  ### 2. Top weaknesses in the current plan

  1. **There is no single executable Hydra plan right now.**
`HYDRA_FINAL.md` declares itself SSOT and explicitly moves to **two-tier 24-block learner / 12-block actor** because old 40-block monolith undertrained at available sample budget. But `HYDRA_SPEC.md` and parts of `INFRASTRUCTURE.md` still describe **40-block ~16.5M single-model baseline**, and `INFRASTRUCTURE.md` still frames Phase 3 around **PPO league self-play** instead of ExIt-centered plan. `BUILD_AGENT_PROMPT.md` then tells coding agent not question spec. Not harmless drift; guaranteed impl hazard. (`HYDRA_FINAL.md` §4.2, §11; `HYDRA_SPEC.md` §Architecture Overview; `INFRASTRUCTURE.md` §Phase 3; `BUILD_AGENT_PROMPT.md`)

  2. **Plan over-invests in exact constraint mechanics and under-specifies thing that determines belief quality: event model.**
CT-SMC elegant and likely useful. But exact fixed-margin sampling does not help much if
[
p(e_t \mid I_t, X_t)
]
weak, miscalibrated, or missing persistent opponent-plan structure. In Mahjong, hard part not “respect 4-copy constraints.” Hard part = “interpret discard/call/pass timing as evidence about coherent hidden hand plan.” Hydra talks about this, but current plan still treats it too locally. (`HYDRA_FINAL.md` §5; `OPPONENT_MODELING.md` §3–§5; `COMMUNITY_INSIGHTS.md` §1, §11)

  3. **Hydra betting project on training core whose strongest direct evidence is for wrong game setting.**
ACH real result, but evidence is **2-player zero-sum Mahjong**, not 4-player general-sum Riichi. OLSS likewise strongest in 2-player settings, and OLSS-II in 2-player Mahjong itself useful but not magic bullet. DRDA even riskier: own evidence file blunt that it has **zero neural-scale experiments**. Putting **DRDA-wrapped ACH** at center of Hydra is exactly kind of elegant research bet that can consume budget without producing strength. (`ACH 2022`; `OLSS 2023`; `rnad_drda_report.md`; `HYDRA_FINAL.md` §11)

  4. **Search plan too broad for budget.**
Current AFBS proposal ambitious enough be research program by itself: belief mixtures, particles, robust opponent nodes, endgame exactification, pondering, and ExIt all at once. OLSS’s empirical result in 2-player Mahjong was modest but real improvement, roughly **0.14–0.22 fan**, and even there Tencent used **pUCT** because CFR-style online solving too expensive. That argues for **selective specialist search**, not generalized online belief solving as universal layer. (`OLSS 2023` Table 3, Appendix D; `HYDRA_FINAL.md` §7, §10)

  5. **Hydra’s current opponent modeling fragmented into heads, when it should be single latent inference problem.**
Tenpai head, danger head, wait-set head, call-intent head, value-conditioned tenpai: each reasonable, but together still look like collection of correlated diagnostics rather than coherent posterior over “what hand plan is this opponent pursuing?” Strong Mahjong play needs persistent plan inference, not local risk scoring. (`OPPONENT_MODELING.md` §3–§4)

  6. **Current danger formulation weaker than data available to Hydra.**
Plan still leans too much on sparse “actual deal-in” labels and downstream action score like
[
\log \pi(a) - \lambda \log p_{\text{danger}}(a).
]
That throws away structure. In Riichi logs, concealed hands can be reconstructed exactly from `start_kyoku` and subsequent events, so Hydra can generate **dense targets** for tenpai, wait-set membership, value bins, ron-eligibility, and intent. Using only sparse deal-in outcomes is one of biggest “vague in exactly wrong place” issues in current docs. (`OPPONENT_MODELING.md` §3.5, §4.4, §4.6; `HYDRA_FINAL.md` §5; `GAME_ENGINE.md`)

  7. **Compute story internally contradictory.**
`HYDRA_FINAL.md` presents 2000-GPU-hour path aimed at LuckyJ-level strength. `INFRASTRUCTURE.md` budgets 2668 GPU-hours. `COMPUTE_FEASIBILITY.md` says plainly that budget probably **not enough for LuckyJ-level without radical efficiency innovation**. That means Hydra cannot afford be merely “ambitious.” It needs sharply prioritized breakthrough path. (`HYDRA_FINAL.md` §11; `INFRASTRUCTURE.md` §Compute Budget; `COMPUTE_FEASIBILITY.md` §8)

  8. **Too much novelty budget spent on second-order ideas.**
RSA-style deception, generic IVD bonuses, primal-dual risk constraints, broad Hunter-bound machinery, and other elegant modules may all be interesting. But they are downstream of real problem: infer opponents better, search only where it pays, and amortize that reasoning into stable deployable policy. As written, Hydra risks becoming clever stack of second-order improvements on top of only partially solved first-order problem. (`OPPONENT_MODELING.md` §7.6; `HYDRA_FINAL.md` §6, §12)

---

  ### 3. Best upgrades to the current plan

  #### 3.1 De-risk the training core: make the optimizer boring and let the novelty live in belief/search

  * **Concept**
Move **DRDA-wrapped ACH** off critical path. Keep as challenger branch, not foundation.

  * **Why it helps**
Hydra’s real chance at breakthrough not “better policy update rule.” It is “better public-belief inference + better opponent modeling + more efficient search amortization.” If optimizer unstable, none of that matters.

  * **Evidence basis**
Direct Mahjong evidence for ACH limited to **2-player zero-sum**. DRDA has no neural-scale evidence in own survey. By contrast, strongest direct Mahjong gains come from **Suphx-style guidance and value shaping**, **RVR-style variance reduction**, and selective online search, not optimizer novelty. (ACH 2022; `rnad_drda_report.md`; Suphx 2020; RVR 2022)

  * **Exact technical proposal**
Mainline Hydra should use:
[
    L = L_{\text{AC}} + \beta_{\text{KL}}, \mathrm{KL}(\pi_\theta,|, \pi_{\text{ref}}) + \beta_{\text{ExIt}}, \mathrm{CE}(\pi_{\text{deep}}, \pi_\theta) + L_{\text{aux}}
]
where:

    * (L_{\text{AC}}) standard clipped actor-critic objective,
    * (\pi_{\text{ref}}) previous checkpoint or teacher policy,
    * (\pi_{\text{deep}}) selective search teacher target.

Start with:

    * clip (\epsilon = 0.10),
    * target median policy KL to reference in ([0.02, 0.08]),
    * (\beta_{\text{ExIt}}) annealed from 0.0 to 0.5 after warm start,
    * entropy coefficient (5\times10^{-4}) to (1\times10^{-3}).

Run ACH/DRDA only as controlled A/B branch once rest of system working.

  * **Feasibility**
High. Simpler than current mainline.

  * **Expected upside**
Mainly **risk reduction**, but that matters lot here.

  * **Risk**
You might leave some performance on table if ACH/DRDA truly transfers well to 4-player Riichi. But that challenger question, not reason make it load-bearing.

---

  #### 3.2 Replace plain oracle dropout with a constrained guider–learner setup

  * **Concept**
Keep privileged training, but upgrade from naive oracle distillation to **GPO-like “possibly-good” guider**.

  * **Why it helps**
Suphx shows oracle-style guidance helps. PerfectDou shows **perfect-training / imperfect-execution** can be sample-efficient in multiplayer imperfect-information game. But Mortal’s internal analysis says plain oracle guiding did not help enough in practice. Missing piece likely **imitation gap**: teacher too strong + too privileged. GPO’s key idea: keep guider ahead** of learner, not far beyond it. (Suphx 2020; PerfectDou 2022; `MORTAL_ANALYSIS.md`; GPO 2025)

  * **Evidence basis**
Direct Mahjong: medium. Adjacent imperfect-information multiplayer: medium-strong.

  * **Exact technical proposal**
Use two policies sharing most weights:
[
\pi_g(a \mid o_{\text{pub}}, o_{\text{priv}}), \qquad \pi_l(a \mid o_{\text{pub}})
]
with inputs
[
o_g=[o_{\text{pub}},o_{\text{priv}},1], \qquad o_l=[o_{\text{pub}},0,0].
]

Train guider with RL plus barrier:
[
    L_g = L_{\text{RL}}(\pi_g) + \lambda_{\text{bar}} \max!\big(0,\mathrm{KL}(\pi_g,|, \pi_l)-\delta\big)^2.
]

Train learner with imitation plus light RL:
[
L_l = \lambda_{\text{BC}}, \mathrm{CE}(\pi_g^\tau,\pi_l) + \lambda_{\text{RL}} L_{\text{RL}}(\pi_l) + L_{\text{aux}}.
]

Starting schedule:

    * (\delta = 0.40 \rightarrow 0.25 \rightarrow 0.15) nats/state over oracle phase,
    * (\lambda_{\text{BC}} = 1.0 \rightarrow 0.3),
    * (\lambda_{\text{RL}} = 0.1 \rightarrow 0.5),
    * distillation temperature (\tau=2.0) or (3.0).

  * **Feasibility**
High. Hydra already has oracle/student machinery in docs.

  * **Expected upside**
Medium-high. One of cleaner ways turn Hydra’s oracle phase from “nice idea” into real sample-efficiency lever.

  * **Risk**
Moderate. If guider still becomes too privileged or privileged features badly chosen, you can still get poor transfer.

---

  #### 3.3 Make opponent modeling a unified latent posterior, not a pile of heads

  * **Concept**
Turn tenpai, wait-set, danger, yaku-plan, and next-discard prediction into one coherent **latent-opponent-plan posterior**.

  * **Why it helps**
This real bottleneck. Strong Riichi AI largely about reading **persistent hand plans** from partial public evidence. Player discarding tile not revealing local risk score; they are expressing latent hand trajectory. Hydra’s current multi-head design sees that, but still too indirectly.

  * **Evidence basis**
Direct Mahjong: strong that opponent reading matters, but weak on exact architecture. Imperfect-information-game evidence: medium. Cross-disciplinary transfer: strong for sequential Bayesian inference.

  * **Exact technical proposal**
Introduce per-opponent latent plan (z_i) with (K=8) to (16) categories or small continuous embedding. Maintain
[
q_t(X, z_{1:3}) \propto p_0(X)\prod_i p(z_i)\prod_{\tau \le t} p_\phi(e_\tau \mid I_\tau, X, z_{1:3}).
]

Where:

    * (X) fixed-margin concealed-tile allocation,
    * (z_i) captures persistent intent/style/hand-class.

Use CT-SMC for (X), but weight particles by learned event model:
[
w^{(p)} \leftarrow w^{(p)} \cdot p_\phi(e_t \mid I_t, X^{(p)}, z).
]

Crucially, train event model on **dense reconstructible labels** from logs:

    * exact tenpai indicator,
    * exact wait-set (W_i(s)),
    * exact ron-eligibility/furiten state,
    * hand-value bin if won now,
    * next discard/call.

Then compute risk compositionally:
[
\hat r_i(k\mid s) \approx \hat p(T_i=1\mid s)\cdot \hat p(k \in W_i \mid s)\cdot \hat p(\text{ron-legal} \mid k,s).
]

Make current “wait-set head” **core** component, not extension.

  * **Feasibility**
Medium. But Mahjong’s logs make this unusually implementable because concealed hands can be reconstructed exactly from `start_kyoku` plus replay. Major advantage Hydra should exploit much harder. (`OPPONENT_MODELING.md` §3.5)

  * **Expected upside**
high. Most important upgrade in whole answer.

  * **Risk**
Medium-high. Latent plans can collapse or become hard interpret. But even weak version likely better than sparse deal-in supervision alone.

---

  #### 3.4 Refactor AFBS into selective specialist search, and use paired worlds to reduce label variance

  * **Concept**
Do **less search**, but make it more surgical + more label-efficient.

  * **Why it helps**
Suphx’s run-time adaptation beat non-adapted policy **66%** in fixed-initial-hand experiments, but too expensive deploy directly. OLSS-II improved 2-player Mahjong but by modest amount and without full guarantees. Lesson not “search more.” It is “search only where it pays, and amortize it.” (Suphx 2020 §4.3; OLSS 2023)

  * **Evidence basis**
Direct Mahjong: strong that selective adaptation/search helps. IIG evidence: medium.

  * **Exact technical proposal**
Split search into three modes:

    1. **Defense search** for push/fold and mawashi-uchi states,
    2. **Close-offense search** when top candidate discards near-tied but differ in expected value/yaku path,
    3. **Endgame solver** when wall small.

Add value-of-computation trigger:
[
\text{VOC}(s)=\eta_0 + \eta_1\Delta_{12}^{-1} + \eta_2 H(B_s) + \eta_3 p_\cup(a^*) + \eta_4 \text{placement_leverage}(s) + \eta_5 \mathbf{1}[W\le 8].
]
Search only if (\text{VOC}(s)>\tau), with (\tau) tuned hit fixed compute budget.

Starting thresholds:

    * search if top-2 base-policy gap (< 0.05),
    * or calibrated union danger (> 0.08),
    * or wall ( \le 8),
    * or posterior entropy above late-game threshold.

Then generate lower-variance labels with **common-random-number worlds**:
[
\hat Q(a)=\frac{1}{M}\sum_{m=1}^{M}R(s,a;\omega_m,\xi_m),
]
using **same** hidden world (\omega_m) and downstream random seeds (\xi_m) across top-(K) candidate actions. This reduces variance of (\Delta Q(a,b)), which ExIt needs.

  * **Feasibility**
Medium-high.

  * **Expected upside**
High, mainly through **better teacher labels per millisecond**, not through stronger online search alone.

  * **Risk**
Moderate. Trigger can miss useful states or over-search noisy ones if not well calibrated.

---

  #### 3.5 Replace “robust opponent soft-min” as whole story with true safe-exploitation layer

  * **Concept**
Keep robustness, but add explicit **safe-vs-exploit interpolation** over opponent-style posterior.

  * **Why it helps**
4-player general-sum self-play does not inherit comforting logic of 2-player zero-sum equilibrium training. Own evidence package explicit about this. Hydra’s current KL-ball / archetype soft-min good start for robustness, but not same thing as strategically exploiting real opponent populations. (`population-exploitation-survey.md`; `multiplayer_search_research.md`)

  * **Evidence basis**
Direct Mahjong: medium through Suphx/Mortal population pretraining. Multiplayer game-theory evidence: strong.

  * **Exact technical proposal**
Keep:
[
Q_{\text{safe}}(a)
]
from robust search / KL-ball opponent uncertainty.

Add:
[
Q_{\text{exploit}}(a)=\sum_m q(m\mid h_{1:t})Q(a\mid \sigma_m),
]
where (m) indexes opponent-style models or QRE temperature bins.

Then score actions by:
[
Q_{\text{mix}}(a)=(1-\alpha(s))Q_{\text{safe}}(a)+\alpha(s)Q_{\text{exploit}}(a)-\lambda_{\text{OOD}}(s)U(a).
]

Set
[
\alpha(s)=\mathrm{clip}!\left(\sigma(\beta_0+\beta_1 C_{\text{opp}}(s)-\beta_2 \text{OOD}(s)-\beta_3 p_\cup(a^*)),,0,,\alpha_{\max}\right).
]

Start with (\alpha_{\max}=0.35) online. Only allow larger values in offline evaluation or clearly in-distribution settings.

Also: use human logs not for BC warm start, but as standing corpus for **style posterior training**.

  * **Feasibility**
Medium.

  * **Expected upside**
High against real human populations; moderate against strong self-play bots.

  * **Risk**
Moderate-high. Overfitting to population can create new exploitability holes. That why interpolation must be confidence-gated.

---

  #### 3.6 Make placement/tail-risk + offensive value explicit, and calibrate them

  * **Concept**
Replace abstract IVD/CVaR aspirations with concrete **multi-timescale distributional decision stack**.

  * **Why it helps**
Mortal’s known weaknesses—south-4 cowardice, efficiency-over-yaku bias, and coarse placement sensitivity—are exactly places where stronger multi-timescale value model should help. Suphx also got lot of mileage from global reward prediction. (`MORTAL_ANALYSIS.md`; Suphx 2020; `HYDRA_FINAL.md` §12)

  * **Evidence basis**
Direct Mahjong: strong. Distributional RL evidence: strong.

  * **Exact technical proposal**
Use three decision-value objects:

    1. (V_{\text{kyoku}}(s)): round-level expected point transfer,
    2. (Z_{\text{rank}}(s)): 24-class final placement distribution,
    3. (Z_{\text{tail}}(s)): quantile utility head for tail risk.

Score each action by:
[
S(a)=w_1(s)Q_{\text{kyoku}}(a)+w_2(s)\mathbb{E}[u(\text{rank})\mid a]+w_3(s)\mathrm{CVaR}_{\alpha(s)}(u\mid
]

Add **offensive oracle heads** for candidate discard actions:

    * (P(\text{tenpai in }1,2,3\text{ draws}\mid
    * (P(\text{mangan+}\mid
    * (E[\text{score}\mid \text{win},a]),
    * (\Delta)uke-ire / path-value features.

These should be learned from hidden-state teacher data and selective deep-search labels.

  * **Feasibility**
High.

  * **Expected upside**
Medium-high, especially in late-round + comeback states.

  * **Risk**
Low-moderate. Mostly engineering + calibration problem.

---

  ### 4. Breakthrough bets

  #### 4.1 Hierarchical latent-opponent posterior over persistent hand plans

  * **Novelty level**
High for Riichi Mahjong.

  * **Evidence level**
Medium. Strong direct evidence that opponent reading matters; weaker direct evidence on this exact architecture. (`OPPONENT_MODELING.md`; `COMMUNITY_INSIGHTS.md`)

  * **Why it could matter specifically in Mahjong**
Mahjong unusually rich in persistent hidden-hand plans: flush lines, toitoi-ish shapes, speed vs value tradeoffs, riichi timing, safety reserve behavior. Plan-aware posterior can turn seemingly local discard evidence into coherent long-horizon inference.

  * **Minimum viable experiment**
On reconstructed logs, compare:

    1. separate tenpai/danger heads,
    2. unified latent-plan posterior model,
on hidden-hand NLL, wait-set recall, and downstream rank-points in duplicate evaluation.

  * **What success would look like**
Better posterior log-likelihood, noticeably better wait-set calibration, and measurable drop in damaten deal-ins without overall push-collapse.

---

  #### 4.2 Incremental belief accumulator (“belief-NNUE” for Mahjong)

  * **Novelty level**
high.

  * **Evidence level**
Weak-to-medium. Strong structural intuition; limited direct game evidence. (`incremental-belief-survey.md`)

  * **Why it could matter specifically in Mahjong**
Mahjong public state changes sparsely. Most turns reveal one tile and maybe one action. If belief updates can be done incrementally, Hydra gets cheaper real-time inference and much better pondering reuse.

  * **Minimum viable experiment**
Train event-updated belief accumulator and compare it to full recomputation on:

    * hidden-hand NLL,
    * wait-set calibration,
    * latency,
    * ponder-reuse hit rate.

  * **What success would look like**
Less than 1% degradation in posterior quality for large gain in latency or root-reuse.

---

  #### 4.3 Common-random-number ExIt labels / duplicate-world action evaluation

  * **Novelty level**
High.

  * **Evidence level**
Weak direct Mahjong evidence; strong transfer from variance-reduction in stochastic simulation. RVR also strongly suggests Hydra should care about variance. (RVR 2022)

  * **Why it could matter specifically in Mahjong**
Root-action evaluation in Mahjong noisy because most downstream randomness shared across candidate actions. Evaluating all candidate root actions on same sampled hidden worlds should reduce variance of action differences dramatically.

  * **Minimum viable experiment**
For hard states, compare independent rollout labels vs paired-world labels for same top-3 actions. Measure:

    * variance of (\Delta Q),
    * teacher-label rank stability,
    * downstream ExIt performance per unit compute.

  * **What success would look like**
Big reduction in label variance and better policy improvement at same search budget.

---

  #### 4.4 Safe exploitation with online opponent-style posterior and QRE temperature

  * **Novelty level**
Medium-high.

  * **Evidence level**
Medium. Theory side strong in adjacent domains; Mahjong-specific validation weak. (`population-exploitation-survey.md`)

  * **Why it could matter specifically in Mahjong**
Real Mahjong populations noisy + style-diverse. Modeling them as one “robust adversary” leaves value on table. Style posterior plus confidence-gated exploitation could produce real edge, especially against LuckyJ-level systems if they have systematic biases.

  * **Minimum viable experiment**
Train style/posterior models on human/self-play logs, then evaluate safe–exploit frontier against held-out style mixtures.

  * **What success would look like**
Clear gains on in-distribution populations with only small worst-case degradation under held-out mixtures.

---

  #### 4.5 Tiny exact/near-exact endgame solver on public-belief slices

  * **Novelty level**
Medium-high.

  * **Evidence level**
Medium. Late-game exactification well-motivated; exact form not established. (`HYDRA_FINAL.md` §7.5; `mean_field_analysis.md`)

  * **Why it could matter specifically in Mahjong**
Late game = hidden-state correlations strengthen, push/fold becomes razor-thin, and placement leverage spikes. This exactly where small exact or near-exact solver can beat generic policy.

  * **Minimum viable experiment**
Restrict to wall ( \le 6) or ( \le 8), high-threat states only, and compare:

    * generic policy,
    * PIMC,
    * paired-world PIMC,
    * tiny exact/branch-and-bound solver.

  * **What success would look like**
Better late-game rank-point outcomes with bounded latency and small activation rate.

---

  ### 5. Concrete technical fill-ins

  #### 5.1 Belief model: posterior, losses, updates

Use joint posterior over hidden allocation (X) and opponent plans (z_{1:3}):
[
q_t(X,z_{1:3}) \propto p_0(X)\prod_{i=1}^3 p(z_i)\prod_{\tau \le t} p_\phi(e_\tau \mid I_\tau, X, z_{1:3}).
]

For online inference:

[
w_t^{(p)} \leftarrow w_{t-1}^{(p)} \cdot \sum_{z} q_{t-1}(z), p_\phi(e_t \mid I_t, X^{(p)}, z)
]
followed by normalization and resampling when
[
\mathrm{ESS} < 0.5 P.
]

Per-opponent plan update:
[
q_t(z_i) \propto q_{t-1}(z_i),\mathbb{E}*{X\sim q*{t-1}}[p_\phi(e_t \mid I_t, X, z_i)].
]

Make opponent learning task dense:

[
L_{\text{opp}}=
\lambda_T L_{\text{BCE}}(\hat T,T)
+\lambda_W L_{\text{BCE}}(\hat W,W)
+\lambda_V L_{\text{CE}}(\hat V,V)
+\lambda_A L_{\text{CE}}(\hat a,a)
+\lambda_P L_{\text{CE}}(\hat z,z^*)
]

with starting weights:

  * (\lambda_T = 1.0)
  * (\lambda_W = 1.0)
  * (\lambda_V = 0.5)
  * (\lambda_A = 0.5)
  * (\lambda_P = 0.25)

Use exact hidden-hand replay to derive:

  * (T): exact tenpai,
  * (W): exact wait-set,
  * (V): current hand-value bin if won on each wait,
  * next actual public action.

**Fallback approximations**

  * **Stage 0**: one-component Sinkhorn marginals + dense oracle targets, no particles.
  * **Stage 1**: CT-SMC with (P=96) on-turn, (P=384) in pondering.
  * **Stage 2**: add latent-plan mixture only if held-out event NLL + wait-set calibration improve materially.

---

  #### 5.2 Risk aggregation and calibration

Do not use raw
[
\log \pi(a)-\lambda \log p_{\text{danger}}(a)
]
as main decision rule.

Instead compute calibrated per-opponent risk:
[
\tilde r_i(k\mid s)=\tilde p(T_i=1\mid s)\cdot \tilde p(k\in W_i\mid s)\cdot \tilde p(\text{ron-legal}\mid k,s).
]

Then aggregate:
[
p_\cup(k\mid s)=1-\prod_{i=1}^{3}(1-\tilde r_i(k\mid s)).
]

If pairwise threat interaction calibration available:
[
p_\cup \approx \sum_i \tilde r_i - \sum_{i<j}\kappa_{ij}(s)\tilde r_i\tilde r_j.
]

Post-hoc calibration:

  * per-bucket temperature scaling for tenpai, wait-set, and danger,
  * buckets by early/mid/late hand, riichi/open/closed field, and number of threatening opponents.

Targets:

  * tenpai ECE (< 0.02),
  * danger ECE (< 0.03) overall and (< 0.05) in high-threat buckets.

---

  #### 5.3 Search trigger, budgets, and paired-world labels

Use small trigger head or explicit score:

[
\text{VOC}(s)=\eta_0+\eta_1 \mathbf{1}[\Delta_{12}<0.05]+\eta_2 H(q_t)+\eta_3 p_\cup(a^*)+\eta_4 \mathbf{1}[W\le 8]+\eta_5 \text{placement_leverage}(s).
]

Initial budgets:

  * close-offense search: **4–12 ms**
  * defense search: **8–24 ms**
  * endgame search: **20–80 ms**
  * pondering: all remaining idle time

Use paired worlds for top-(K) action teacher labels:

  ```python
  def paired_world_q(state, topk_actions, belief, M):
      returns = {a: [] for a in topk_actions}
      worlds = sample_hidden_worlds(belief, M)  # CT-SMC particles / weighted worlds

      for m, world in enumerate(worlds):
          draw_seed = seed_from(world, m)
          opp_seed = seed_from("opp", m)

          for a in topk_actions:
              r = rollout_same_world(
                  state=state,
                  root_action=a,
                  hidden_world=world,
                  draw_seed=draw_seed,
                  opp_seed=opp_seed,
              )
              returns[a].append(r)

      return {a: mean(returns[a]) for a in topk_actions}
  ```

That should be default label generator for hard states, not independent rollouts.

---

  #### 5.4 Safe-exploitation decision rule

Maintain:

  * (Q_{\text{safe}}): robust value against uncertainty set,
  * (Q_{\text{exploit}}): value against posterior opponent-style model.

Then:
[
Q_{\text{mix}}(a)=(1-\alpha(s))Q_{\text{safe}}(a)+\alpha(s)Q_{\text{exploit}}(a)-\lambda_{\text{OOD}}(s)U(a).
]

Set
[
\alpha(s)=\min\Big(\alpha_{\max},\sigma(\beta_0+\beta_1 C_{\text{opp}}-\beta_2 \text{OOD}-\beta_3 p_\cup)\Big).
]

Starting points:

  * (\alpha_{\max}=0.35),
  * (\lambda_{\text{OOD}}=0.25),
  * exploitation disabled entirely when OOD score exceeds conservative threshold.

---

  #### 5.5 Multi-timescale critic and offensive oracle heads

Use:

  * (V_{\text{kyoku}}): scalar,
  * (Z_{\text{rank}}): 24-way rank distribution,
  * (Z_{\text{tail}}): 32-quantile utility head,
  * (O(a)): offensive oracle vector for top discard actions.

Quantile loss:
[
L_Q=\frac{1}{N}\sum_{i,j}\rho_{\tau_i}^{\kappa}(y_j-\theta_i), \qquad N=32,\ \kappa=1.
]

Action score:
[
S(a)=w_{\text{kyoku}}Q_{\text{kyoku}}(a)+w_{\text{rank}}\mathbb{E}[u(\text{rank})\mid a]+w_{\text{tail}}\mathrm{CVaR}*{0.1}(u\mid a)-\lambda*{\text{risk}}p_\cup(a).
]

Initial state-dependent weights:

  * opening: ((0.70, 0.25, 0.05))
  * south rounds, moderate leverage: ((0.35, 0.40, 0.25))
  * south-4 / all-last / big leverage: ((0.20, 0.45, 0.35))

Offensive oracle vector for action
[
O(a)=\big[
P(\text{tenpai in 1 draw}),
P(\text{tenpai in 3 draws}),
P(\text{mangan+}),
E[\text{score}\mid \text{win}],
\Delta \text{ukeire},
P(\text{retain safe reserve})
\big].
]

These should be supervised from hidden-state teacher data and selective deep-search labels.

---

  #### 5.6 Suggested interface / data flow for coding agent

  ```python
  class BeliefState:
      tile_marginals: Float[34, 4]        # opp1, opp2, opp3, wall
      plan_posteriors: Float[3, K]        # K latent styles/plans
      particles: Optional[ParticleSet]
      entropy: float
      ess: float
      ood_score: float

  class OpponentReadout:
      tenpai_prob: Float[3]
      waitset_prob: Float[3, 34]
      value_bin_prob: Float[3, B]
      next_action_prob: Float[3, A]

  class SearchFeatures:
      q_delta: Float[A]
      risk_union: Float[A]
      tail_risk: Float[A]
      belief_value: Float[A]
      search_uncertainty: Float[A]

  def decide(obs_public, belief_state):
      base = model(obs_public, belief_state)
      if trigger_search(base, belief_state):
          sf = specialist_search(obs_public, belief_state, mode=pick_mode(base, belief_state))
          logits = base.policy_logits + saf_adapter(sf)
      else:
          logits = base.policy_logits
      return masked_softmax(logits)
  ```

That level of interface separation Hydra’s current docs need more of.

---

  ### 6. Recommended revised Hydra research agenda

  #### Must-have

  * **Unify docs first.** Archive old 40-block monolith + PPO-centric phase descriptions. Make one SSOT.
  * **Move DRDA/ACH off critical path.** Mainline should be stable actor-critic + ExIt distillation.
  * **Make dense hidden-state target generation core.** Exact tenpai, exact wait-set, exact value bins, exact plan labels.
  * **Upgrade oracle phase to guider–learner, not plain oracle dropout.**
  * **Make belief/opponent model hierarchical + calibrated.**
  * **Refactor search into selective defense/offense/endgame specialists.**

  #### Strong multipliers

  * **Safe-exploitation layer** over opponent-style posterior.
  * **Multi-timescale distributional critic** with explicit tail-risk control.
  * **Paired-world ExIt labels** for lower-variance teacher data.
  * **Human-log population modeling** as permanent training stream, not warm start.
  * **Diversity-aware league**, but only after core opponent model works.

  #### Speculative bets

  * Incremental belief accumulator.
  * Tiny exact endgame solver beyond simple PIMC.
  * QRE temperature inference / level-k opponent layers.
  * Domain-adversarial human-vs-self-play representation alignment, if domain shift severe.

  #### Likely dead ends / not worth complexity right now

  * **Mainlining DRDA-wrapped ACH**
  * **Broad AFBS everywhere**
  * **Early RSA/deception modules**
  * **Generic IVD bonuses before core belief/search stack working**
  * **Large Mixture-SIB complexity before proving multimodality decision-relevant**
  * **Heavy Hunter-bound machinery outside multi-threat late-game defense**

---

  ### 7. Evaluation plan

  #### Ablations

  1. **Base**: BC + standard policy/value + GRP
  2. * dense tenpai/wait-set/value opponent supervision
  3. * guider–learner privileged phase
  4. * calibrated latent opponent posterior
  5. * selective specialist search
  6. * paired-world teacher labels
  7. * safe-exploitation layer
  8. * diversity-aware league

Do not run speculative bets before 1–7 stable.

  #### Matchups

  * Current Hydra baseline vs revised Hydra
  * Revised Hydra vs BC anchors
  * Revised Hydra vs frozen older checkpoints
  * Revised Hydra vs style-perturbed anchors
  * Revised Hydra vs exploiters trained against its policy family
  * If legally clean + operationally isolated, external benchmark bots can be used for evaluation-only, but Hydra should not depend on them for design validation

  #### Metrics

**Strength**

  * mean rank points/game
  * mean placement
  * 1st and 4th rates
  * deal-in rate overall
  * deal-in rate conditioned on actual opponent tenpai
  * damaten deal-in rate
  * riichi win rate / value per win

**Belief / opponent modeling**

  * posterior log-likelihood of true hidden hands
  * tenpai AUROC / Brier
  * wait-set recall@k and calibration
  * danger ECE / Brier
  * style posterior accuracy / calibration

**Search**

  * gain per searched state
  * teacher-label variance
  * ponder reuse hit rate
  * ms spent per useful search call
  * percent of searched states where deep search changes top action

**Robustness / exploitation**

  * performance under held-out opponent mixtures
  * safe–exploit frontier as (\alpha) varies
  * OOD gating quality

  #### Failure signals

  * Belief NLL improves but match strength does not
  * Search produces lots of KL from base policy but little Elo/rank-point gain
  * Exploitation gains collapse on held-out opponent mixtures
  * Danger/tenpai calibration remains poor in high-threat buckets
  * League becomes non-transitive + brittle without corresponding population gains

  #### Stopping criteria

  * Module should not be promoted unless it clears **at least one** of:

    * (>1) rank-point/game gain at 95% confidence on large duplicate eval,
    * or smaller gain plus major auxiliary improvement that strongly predictive of strength
  * Kill or demote any module that adds:

    * (>10%) latency,
    * (>20%) training cost,
    * with (<0.5) rank-point/game gain after serious evaluation

Use evaluation scale already suggested in `INFRASTRUCTURE.md` as statistical backbone; that part of current plan is good.

---

  ### 8. Final recommendation

If I were reshaping Hydra into strongest + most novel feasible version of itself, I would make it **belief-centric, not optimizer-centric**.

Exact direction I would choose:

  1. **Commit to two-tier 24/12 architecture and delete old 40-block spec.**
  2. **Mainline stable actor-critic + ExIt training core.**
  3. **Upgrade oracle phase into constrained guider–learner setup.**
  4. **Make dense hidden-state supervision heart of project:** exact tenpai, wait-set, value, and plan targets from reconstructed logs.
  5. **Build latent opponent-plan posterior on top of CT-SMC**, because that where real Mahjong breakthrough most plausible.
  6. **Refactor search into selective defense/offense/endgame teachers** and use paired-world labels so search compute buys learning signal, not online strength.
  7. **Add confidence-gated safe-exploitation layer** once opponent posterior trustworthy.
  8. **Postpone DRDA/ACH, RSA deception, generic IVD, and other elegant tails** until above stack demonstrably winning.

In one sentence: **Hydra’s best shot at becoming genuinely formidable is to win information war—better public-belief inference, better opponent reading, and better selective search amortization—not to win optimizer war.**
</start-answer>
]]>
</answer_text>
</answer_section>
</combined_run_record>