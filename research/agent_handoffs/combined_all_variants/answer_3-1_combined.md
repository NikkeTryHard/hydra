<combined_run_record run_id="answer_3-1" variant_id="agent_answers_mixed_transcript" schema_version="1">
<metadata>
<notes>Mixed transcript-style file from agent_answers. Contains prompt + answer bodies.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="agent_answers/ANSWER_3-1.md" extracted_from="mixed_transcript">
<![CDATA[# Hydra deep-agent follow-up for ANSWER_3-style agent

Attached:
  - `hydra_agent_handoff_docs_only.zip` — primary source material
  - `deep_agent_20_pdfs.zip` — primary paper/reference package

Use docs zip first. If zip inaccessible, use raw GitHub markdown links provided separately. Do NOT use normal GitHub browsing/search or generic web search to reconstruct project context.

Job in this prompt is NOT inspect source code, NOT write/integrate code. Job: inspect Hydra’s CURRENT PLAN, make it:
  1. stronger,
  2. more coherent,
  3. more likely produce real breakthroughs in Mahjong AI,
  4. still grounded enough for separate coding agent to implement later.

Do NOT give generic brainstorming. Need hard, evidence-backed **second pruning / prioritization pass** on plan *after* several key strategic questions already settled.

Read docs as coherent program, especially:
  - `research/design/HYDRA_FINAL.md`
  - `research/design/HYDRA_RECONCILIATION.md`
  - `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md` (historical; removed later, see `combined_all_variants/README.md`)
  - `research/design/OPPONENT_MODELING.md`
  - `research/design/TESTING.md`
  - `research/infrastructure/INFRASTRUCTURE.md`
  - `docs/GAME_ENGINE.md`
  - `research/design/SEEDING.md`
  - plus any evidence/comparison docs that materially affect conclusions

Treat old `research/BUILD_AGENT_PROMPT.md` reference here as historical execution-rigor context only; for current routing chain, use `combined_all_variants/README.md` plus live authority docs.

  ## Settled decisions you should treat as fixed inputs

Do **not** spend time re-arguing these unless one looks clearly catastrophic:

  - Hydra should **not** restart from scratch.
  - Hydra’s active path is **supervision-first before search-expansion-first**.
  - Use **one unified belief stack**: Mixture-SIB + CT-SMC; no duplicate standalone belief machinery.
  - **Hand-EV comes before deeper AFBS**.
  - **AFBS is selective/specialist**, not broad default runtime.
  - **DRDA/ACH moves off critical path** and lives on reserve shelf.
  - **Oracle guidance should be aligned** so teacher stays teachable.
  - Broad “search everywhere,” duplicated belief stacks, early optimizer-theory detours not on active path.

External evidence already supports several pattern-level choices here:
  - unified public-belief-style state abstractions are real, not made-up Hydra weirdness
  - aligned oracle/teacher guidance more defensible than unconstrained privileged distillation
  - robustness belongs in core solving/objective layer, though Hydra-specific placement details still need judgment

No need spend time re-proving those pattern-level claims unless strong contrary argument.

  ## What I want you to do

Critique **current reconciled Hydra plan**. Make one more hard call on what remains:

  ### Part 1 — Critique the reconciled active path
Identify where reconciled active Hydra path is still:
  - too fragile,
  - too underspecified,
  - too compute-inefficient,
  - too likely stall before real strength,
  - or still carrying too much reserve-shelf baggage in disguised form.

  ### Part 2 — Re-rank the reserve shelf
Reconciliation memo keeps several old good ideas on reserve shelf. Sort them harder:

  - which reserve ideas are strong **phase-next** candidates,
  - which are long-shot but worth preserving,
  - which should probably be demoted further,
  - which one or two reserve ideas have best “if active Hydra underdelivers, try this next” upside.

Focus especially on:
  - robust-opponent search backups vs confidence-gated safe exploitation
  - richer latent opponent posterior
  - deeper AFBS semantics
  - stronger endgame exactification
  - incremental/structured belief updates
  - any remaining optimizer/game-theory ideas

  ### Part 3 — Identify the strongest breakthrough bets that still survive the pruning
Do not list cool ideas. Need **best surviving breakthrough bets after active-path cuts already happened**.

For each surviving bet, tell me:
  - why still alive after pruning,
  - why it might matter specifically in Mahjong,
  - what evidence supports it,
  - what assumption it relies on,
  - why it might still fail,
  - cheapest meaningful experiment to test it.

  ### Part 4 — Fill in the remaining strategic blanks
For any reserve or breakthrough idea kept alive, fill missing technical details docs still leave abstract:
  - formulas,
  - objective functions,
  - update rules,
  - gating criteria,
  - thresholds,
  - approximate algorithms,
  - calibration procedures,
  - evaluation metrics,
  - stopping rules.

  ## Constraints

  - Do NOT inspect source code.
  - Do NOT pretend you implemented or validated anything.
  - Do NOT give broad generic summaries of Mahjong AI history unless directly relevant.
  - Do NOT recommend things that obviously blow up latency/compute without addressing feasibility.
  - Do NOT rely on AGPL code or impl borrowing.
  - Keep proposals compatible with separate coding agent implementing them later.

Assume separate coding agent will take response, use it as strategic decision layer above concrete impl work.

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
  - Is reconciled active Hydra plan strong enough to pursue as mainline?
  - Where strongest?
  - Where still most likely fail?
  - Which reserve-shelf idea most likely matter next if mainline underdelivers?

  ### 2. Hardest remaining weaknesses in the reconciled active path
Rank most important remaining weaknesses or blind spots.

  ### 3. Re-ranked reserve shelf
For each major reserve idea:
  - keep / demote / mostly drop
  - why
  - evidence basis
  - feasibility
  - upside
  - risk

  ### 4. Best surviving breakthrough bets
Give best 3–5 surviving high-upside ideas.
For each:
  - novelty level
  - evidence level
  - why it could matter specifically in Mahjong
  - minimum viable experiment
  - what success would look like

  ### 5. Concrete technical fill-ins
Where remaining reserve/breakthrough ideas too vague, provide:
  - formulas
  - losses
  - update rules
  - thresholds
  - hyperparameters
  - pseudocode / compact illustrative snippets
  - interface/data-flow guidance for coding agent

  ### 6. Recommended revised research agenda after reconciliation
Give revised prioritized roadmap:
  - active mainline must-have
  - phase-next multipliers
  - reserve shelf worth preserving
  - likely dead ends / no-longer-worth-it complexity

  ### 7. Evaluation plan
Tell me how know whether pruned/revised plan is better:
  - ablations
  - matchups
  - metrics
  - failure signals
  - stopping criteria

  ### 8. Final recommendation
If you had to reshape Hydra into strongest, most coherent, feasible version of itself **after this reconciliation pass**, what exact direction would you choose and why?

  ## Style requirements

Want:
  - depth,
  - specificity,
  - formulas,
  - precise proposals,
  - evidence-backed reasoning,
  - clear separation between proven, plausible, speculative ideas.

Do NOT want:
  - fluff,
  - generic “future work,”
  - shallow novelty for novelty’s sake,
  - re-arguing already-settled choices unless genuinely strong reason.

Goal: make Hydra not only more complete, but more formidable and coherent after active-path / reserve-shelf split already made.
</start-prompt>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_answers/ANSWER_3-1.md" extracted_from="mixed_transcript">
<![CDATA[A quick note before main answer: raw `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md` links returned 404 here, so I treated `HYDRA_RECONCILIATION.md` as authoritative summary of those three answer files. It explicitly names them as deep-agent inputs and synthesizes active-path / reserve-shelf split.

  ## 1. Executive verdict

Yes: reconciled active Hydra plan strong enough to pursue as mainline. Strongest parts now coherent: one unified belief story, supervision-first before search-expansion-first, Hand-EV before deeper AFBS, AFBS as specialist hard-state tool, oracle guidance that teaches rather than dominates. This is materially more coherent than earlier “everything advanced at once” version. ([GitHub][1])

Strongest part is not abstract “Mahjong AI novelty.” It is combination of: existing advanced model surfaces already present in repo, reconciliation memo’s explicit focus on target-generation / supervision loop closure, fact that Hydra’s intended ceiling still comes from ExIt + pondering + search-as-feature layered on strong learned base rather than broad always-on search. ([GitHub][2])

Most likely failure point is not ideas themselves, but **program coherence**. Branch still contains mutually inconsistent governing docs: `HYDRA_RECONCILIATION.md` says first tranche should be target/loss closure with no architectural expansion and no broad AFBS rewrite, while `BUILD_AGENT_PROMPT.md` and `IMPLEMENTATION_ROADMAP.md` still describe from-scratch full build, strict roadmap obedience, early ACH/DRDA impl, broad AFBS/robust-opponent work, even stale shapes like `85*34` trajectory observations. If conflict not resolved before impl, coding agent can “correctly” execute wrong plan. ([GitHub][2])

Reserve-shelf idea most likely matter next if active mainline underdelivers is **stronger endgame exactification**. It is narrow, high-leverage, compute-bounded, already endorsed by `HYDRA_FINAL.md` as core ceiling-raising addition. Next most important fallback is **robust-opponent search backups**, but only if selective AFBS already proves valuable on hard states. ([GitHub][2])

  ## 2. Hardest remaining weaknesses in the reconciled active path

**1. Branch still has doc-authority bug, not only design bug.**
`HYDRA_RECONCILIATION.md` says first tranche is target-generation / supervision closure using existing output surfaces, no new heads, no broad AFBS rewrite. `BUILD_AGENT_PROMPT.md` still says `HYDRA_FINAL.md` plus `IMPLEMENTATION_ROADMAP.md` are final and must be followed line by line; roadmap still encodes full from-scratch build sequence and stale interfaces. This is single most dangerous remaining weakness because it can silently reintroduce reserve-shelf baggage into active path. ([GitHub][2])

**2. “Close advanced targets” is right mainline, but target taxonomy still underspecified.**
Reconciliation memo correctly says populate advanced targets, stage nonzero loss activation, distinguish replay-derived, bridge-derived, search-derived labels. But it still leaves ambiguous which targets are replay-safe, which require CT-SMC / AFBS context, and in what order they should be activated. In particular, `delta_q_target` and `exit_target` are not replay-safe by default, while memo’s “preferred order” still groups them too close to replay-only targets. ([GitHub][2])

**3. Belief-target story still mathematically sloppy.**
`HYDRA_FINAL.md` defines belief machinery in terms of Mixture-SIB fields and CT-SMC search-grade posteriors, and reconciliation memo lists `belief_fields_target` and `mixture_weight_target` as possible advanced labels. Problem: raw Sinkhorn external fields are not right supervision object; they are gauge-like and non-identifiable up to row/column reparameterizations. Right supervised object is projected belief itself—marginals or gauge-fixed row logits—not raw fields. This is most important technical blank still abstract. ([GitHub][1])

**4. Hand-EV is correctly early, but still not operationally specified enough to avoid becoming “better heuristics” instead of real offensive oracle.**
Both `HYDRA_FINAL.md` and reconciliation memo elevate Hand-EV, and opponent-modeling stack already assumes value-aware push/fold math. But branch still does not lock down exact limited-depth recursion, simplified ron approximation, caching key, or calibration procedure. That is how promising offensive features turn into underpowered heuristics that never fully cash out. ([GitHub][1])

**5. Selective AFBS is right posture, but branch still lacks hard trust policy for when search labels are usable.**
`HYDRA_FINAL.md` centers ExIt + pondering + SaF and already includes safety valves like visit-count thresholds and KL caps. Reconciliation memo says AFBS should stay specialist and hard-state gated. Still missing: concrete trust rule based on expanded branch mass, particle ESS, visit count, cross-particle variance. Without that, “selective search” will still supervise noise. ([GitHub][1])

**6. Opponent-modeling doc still leaks extra complexity into design space.**
It is useful rationale doc, but explicitly says defer to `HYDRA_FINAL.md` and `HYDRA_RECONCILIATION.md` on impl direction. It still contains attractive-but-weakly-grounded ideas like deception rewards with arbitrary tuning guidance. Those should not re-enter mainline through “interesting auxiliary” creep. ([GitHub][3])

  ## 3. Re-ranked reserve shelf

  ### Stronger endgame exactification — **keep, phase-next #1**

**Why:** narrow, compatible with selective compute, directly aligned with `HYDRA_FINAL.md`’s endgame-exactification rationale and validation gate.
**Evidence basis:** strongest direct Mahjong-specific support among reserve ideas; doc explicitly says late-game decisions are disproportionately high-EV and proposes exact chance enumeration when wall is short.
**Feasibility:** medium.
**Upside:** high.
**Risk:** moderate but bounded. ([GitHub][2])

  ### Robust-opponent search backups — **keep, phase-next #2**

**Why:** one of few reserve ideas also part of Hydra’s intended final identity. But it only pays once local evaluators and selective AFBS are already useful.
**Evidence basis:** strong inside Hydra docs, weaker as immediate mainline sequencing decision.
**Feasibility:** medium-high if AFBS already positive; poor otherwise.
**Upside:** medium-high at last strength mile.
**Risk:** medium-high because it can become expensive sophistication on top of noisy search. ([GitHub][1])

  ### Confidence-gated safe exploitation — **keep, but as cheaper challenger, not coequal mainline**

**Why:** If mainline underdelivers and AFBS still too expensive, root-level exploitation layer is plausible lower-cost alternative to full robust-opponent search.
**Evidence basis:** indirect; reconciliation memo groups “robust-opponent search backups / safe exploitation” together as worth preserving, but does not yet separate them.
**Feasibility:** medium.
**Upside:** medium.
**Risk:** high if opponent posterior calibration poor; lower if gated hard by uncertainty. ([GitHub][2])

  ### Incremental / structured belief updates — **keep high as main contingency branch**

**Why:** best reserve response if unified Mixture-SIB + CT-SMC stack proves too slow or too blurry.
**Evidence basis:** medium; Hydra already treats belief misspecification as core risk and keeps structured belief ideas on reserve shelf.
**Feasibility:** medium.
**Upside:** high if current belief quality tops out early.
**Risk:** medium because belief research can sprawl. ([GitHub][2])

  ### Richer latent opponent posterior — **keep, but demote below endgame and belief contingencies**

**Why:** long-term direction toward more unified opponent modeling is reasonable, but reconciliation memo is right that immediate bottleneck is not lack of outputs.
**Evidence basis:** low-medium.
**Feasibility:** medium.
**Upside:** medium.
**Risk:** high risk of inventing new machinery before existing heads are properly trained. ([GitHub][2])

  ### Deeper AFBS semantics — **keep, but narrow scope**

**Why:** preserve one specific upgrade path—public-event semantics and better hard-state expansion rules—but do not preserve “make AFBS deeper/broader” as vague bucket.
**Evidence basis:** medium.
**Feasibility:** medium-high if kept narrow.
**Upside:** medium-high.
**Risk:** high if it drifts back into broad-search identity. ([GitHub][2])

  ### Optimizer / game-theory additions beyond existing ACH/DRDA assumption — **demote hard**

**Why:** `HYDRA_FINAL.md` still includes ACH and DRDA as part of final architecture, but reconciliation memo is explicit that immediate progress should not depend on resolving optimizer-level debates.
**Evidence basis:** mixed.
**Feasibility:** high in impl terms, low in “best next use of attention.”
**Upside:** unclear short term.
**Risk:** high opportunity cost. ([GitHub][1])

  ### Deception reward / novelty-heavy ToM extras — **mostly drop**

**Why:** opponent-modeling doc itself admits arbitrary starting gamma and no prior basis for tuning. That is exactly kind of reserve-shelf baggage reconciliation memo says to stop letting drive mainline.
**Evidence basis:** weak.
**Feasibility:** medium.
**Upside:** speculative.
**Risk:** high. ([GitHub][3])

  ## 4. Best surviving breakthrough bets

  ### A. Gauge-fixed belief supervision on top of unified Mixture-SIB + CT-SMC stack

**Novelty level:** medium.
**Evidence level:** medium-high inside Hydra’s own design.
**Why it survives pruning:** strengthens mainline instead of creating second belief stack.
**Why it matters in Mahjong:** tile conservation and correlation structure are unusually central here; bad posterior shape poisons both search and safety.
**Assumption:** CT-SMC already produces meaningfully better search-grade posterior than generic mean-field approximations.
**Why it might fail:** if posterior still too blurry or expensive, supervision target is not strong enough to teach backbone anything useful.
**Cheapest meaningful experiment:** compare three auxiliary schemes on same model surface: raw field regression, projected marginal supervision, no belief auxiliary. Measure posterior NLL, pairwise MI calibration, wait-set calibration, downstream hard-state policy gain. Success looks like projected marginals clearly beating raw-field regression. ([GitHub][1])

  ### B. Hand-EV realism as first serious offensive multiplier

**Novelty level:** low-medium.
**Evidence level:** high by Hydra’s own evidence standard.
**Why it survives pruning:** already wired into architecture, cheaper than deeper search, directly endorsed by both `HYDRA_FINAL.md` and reconciliation memo.
**Why it matters in Mahjong:** offensive tempo, tenpai timing, value realization are central; weak offensive forecasts distort both push/fold and search priors.
**Assumption:** limited-depth self-draw recursion with simplified ron model is enough to improve decision quality before full AFBS expansion.
**Why it might fail:** if recursion too greedy or ron approximation badly miscalibrated.
**Cheapest meaningful experiment:** add only improved Hand-EV planes to current learned policy/value stack and run duplicate evaluation against same baseline, plus offline correlation against small-wall exact or MC rollout slices. Success looks like online gain without needing live search. ([GitHub][1])

  ### C. Stronger endgame exactification

**Novelty level:** medium.
**Evidence level:** medium-high.
**Why it survives pruning:** selective, bounded, consistent with Hydra’s “specialist search” identity.
**Why it matters in Mahjong:** late-game push/fold and placement swings are massively leverage-heavy relative to average decisions.
**Assumption:** exact chance over live wall and better terminal utility matter more than broader midgame search.
**Why it might fail:** if trigger policy too broad and compute blows up, or too narrow and gains are negligible.
**Cheapest meaningful experiment:** last-10-draw exactification benchmark with duplicate pairing and slice metrics for orasu / riichi-defense contexts. Success looks like simultaneous improvement in deal-in rate, win conversion, placement swing. ([GitHub][2])

  ### D. Trust-gated selective AFBS feeding ExIt / delta-Q supervision

**Novelty level:** medium.
**Evidence level:** medium.
**Why it survives pruning:** AFBS remains core to Hydra’s ceiling, but only if converted from “broad expensive search” into reliable hard-state teacher.
**Why it matters in Mahjong:** hardest decisions are sparse but decisive; specialist search can change them without dominating total latency.
**Assumption:** small fraction of states carry most of search value, and Hydra can identify them well enough.
**Why it might fail:** if hard-state gate too loose, or if search labels too noisy across particles/archetypes.
**Cheapest meaningful experiment:** generate ExIt labels only on hard-state slice with strict trust filters; compare policy-learning gain against same compute spent on more BC/RL. Success looks like positive duplicate match delta and high “search gain recaptured by SaF/offline distillation.” ([GitHub][1])

  ### E. Confidence-gated exploitation as cheaper “next after mainline” challenger

**Novelty level:** medium.
**Evidence level:** low-medium.
**Why it survives pruning:** one of few reserve ideas with real upside that does not force Hydra back into broad search or extra head sprawl.
**Why it matters in Mahjong:** opponent styles are exploitable, but brittle exploitation is deadly in multiplayer general-sum settings.
**Assumption:** Hydra can produce calibrated enough style posterior to exploit only when confidence high.
**Why it might fail:** posterior confidence may be overestimated; exploitation may overfit anchor styles.
**Cheapest meaningful experiment:** root-level exploitation layer only, tested against style-specific anchor pools and balanced anchors, with hard uncertainty gating. Success looks like gain against style-biased pools without measurable collapse against balanced opponents. ([GitHub][2])

  ## 5. Concrete technical fill-ins

  ### 5.1 Target presence and staged activation

Tranche should use explicit presence-gated loss policy, not “weights default to zero and maybe later become nonzero.” Reconciliation memo already points this way; I would make it formal. ([GitHub][2])

Define, for each auxiliary target (j):

[
m_j = \mathbf{1}{\text{tensor exists} \land \text{finite} \land \text{sane range}}
]

[
w_j(t) = w^{\max}_j \cdot \mathrm{clip}!\left(\frac{t-s_j}{r_j}, 0, 1\right)
]

[
L_{\text{total}} = L_{\text{base}} + \sum_j m_j, w_j(t), \tilde L_j
]

with normalized auxiliary loss

[
\tilde L_j = \frac{L_j}{\operatorname{EMA}(L_j)+10^{-6}}
]

and gradient cap

[
                                                                                                                                                                                                                                                                |\nabla L_{\text{aux}}| \le 0.35,|\nabla L_{\text{base}}|
]

Recommended first-tranche maxima:

  * `safety_residual`: `w_max = 0.02`
  * `belief_marginal`: `0.02`
  * `mixture_weight`: `0.01`
  * `delta_q`: `0.05`
  * `exit_target`: `0.10`

with replay-safe targets activated first and search-derived targets only after provenance is explicit.

  ```python
  for name, target in aux_targets.items():
      present = target is not None and isfinite(target).all() and sane(target)
      if not present:
          continue
      loss = aux_loss[name](pred[name], target)
      loss = loss / (ema[name] + 1e-6)
      total += ramp_weight(name, step) * loss
  ```

  ### 5.2 Belief supervision: do not train raw Sinkhorn fields

`HYDRA_FINAL.md` defines belief in terms of SIB/Mixture-SIB fields and projected beliefs, while reconciliation memo tentatively lists `belief_fields_target` as candidate advanced label. I would not supervise raw fields. ([GitHub][1])

Use projected belief (B_t(k,z)) as supervised object:

[
P_t(z\mid k)=\frac{B_t(k,z)}{\sum_{z'} B_t(k,z')}
]

[
  L_{\text{belief}} = \sum_k r_t(k),\mathrm{KL}!\left(P_t^*(\cdot\mid k),|,P_\theta(\cdot\mid k)\right)
]

where (P_t^*) comes from reconstructed hidden state or CT-SMC-weighted posterior targets.

If mixture supervision is credible later:

[
L_{\text{mix}} = \mathrm{CE}(w^*, w_\theta)
]

but only if component labels come from consistent offline fitting procedure. Otherwise leave `mixture_weight_target` absent.

Good compromise target for “field-like” supervision is gauge-fixed row logit:

[
g_{k,z} = \log(B_{k,z}+10^{-8}) - \frac{1}{4}\sum_{z'}\log(B_{k,z'}+10^{-8})
]

This preserves rowwise relative preference without trying supervise non-identifiable raw fields.

  ### 5.3 Hand-EV realism

Use bounded-depth self-draw recursion over discard candidates. This operationalizes Hand-EV planes already defined in `HYDRA_FINAL.md`. ([GitHub][1])

For discard horizon (d\in{1,2,3}), and live-wall counts (r):

[
\mathrm{Eval}(h_a,r,d)=\sum_{t:r_t>0}\frac{r_t}{R}\cdot \mathrm{BestAfterDraw}(h_a+t, r-e_t, d)
]

If (h_a+t) is agari, compute exact score. Otherwise:

[
\mathrm{BestAfterDraw}(h',r',d)=\max_{b\in\mathcal D(h')} \mathrm{Eval}(h'-b,r',d-1)
]

Use lexicographic continuation order:

  1. (P_{\text{win}})
  2. (P_{\text{tenpai}})
  3. one-step ukeire mass
  4. expected score

Simplified ron approximation once tenpai:

[
P_{\text{ron}}^{(d)}(W)=1-\prod_i\left(1-\sum_{w\in W} p_i^{disc}(w)\right)^{m_i(d)}
]

[
P_{\text{win}} = 1-(1-P_{\text{tsumo}})(1-P_{\text{ron}})
]

Use `kappa_ron = 1.0` initially, calibrate against exact/MC slices.

Pruning:

  * if shanten (\le 1): expand all effective draws
  * else: top 12 draws by `remaining[t] * immediate_gain(t)`
  * depth (\ge 2): keep top 3 discard continuations

  ### 5.4 Search trust gate and ExIt eligibility

`HYDRA_FINAL.md` already gives `min_visits`, KL safety valves, hard-state signals, playout-cap randomization cues. Missing piece is unified trust weight. ([GitHub][1])

Run selective AFBS only if:

[
g_{\text{search}}=\mathbf{1}[
(\Delta_{\text{top2}}<0.10)
\lor (risk_{\max}>0.08)
\lor (ESS/P<0.55)
\lor (wall\le 10)
\lor (\text{orasu})
]
]

Then define label trust

[
\lambda_{\text{exit}}=
\mathrm{clip}!\left(\frac{N_{\text{root}}-64}{256-64},0,1\right)
\cdot
\mathrm{clip}!\left(\frac{m_{\text{expanded}}-0.85}{0.10},0,1\right)
\cdot
\exp(-\sigma_Q/0.15)
\cdot
\mathrm{clip}!\left(\frac{ESS}{0.6P},0,1\right)
]

and only emit `exit_target` or `delta_q_target` when (\lambda_{\text{exit}}>0.5).

[
\pi^* = \mathrm{normalize}\left((1-\lambda_{\text{exit}})\pi_{\text{base}} + \lambda_{\text{exit}}\operatorname{softmax}(Q/0.25)\right)
]

This keeps ExIt specialist, stops noisy search from pretending be ground truth.

  ### 5.5 Endgame exactification

`HYDRA_FINAL.md` already fixes motivation and suggests exactification once wall is short. I would turn that into two-trigger policy. ([GitHub][1])

**Trigger:**

  * `EndgameLite` if `wall <= 10` and any of:

    * orasu,
    * opponent riichi,
    * `max p_tenpai > 0.65`,
    * safe-tile inventory `<= 2`,
    * `top2_gap < 0.08`
  * `EndgameExact` if `wall <= 6` or `(orasu and rank_gap_to_next <= 8000)`

Utility:

[
U=(1-\beta),\mathbb E[\text{placement}] + \beta,\mathrm{CVaR}_\alpha(\text{placement})
]

Recommended schedule:

  * normal rounds: (\beta=0)
  * South 3/4 while leading: (\alpha=0.2,\beta=0.25)
  * South 4 in 1st: (\alpha=0.1,\beta=0.4)

Cache key:
`(canonical_hand, live_wall_counts, riichi_state, turn_idx, score_context_hash)`

  ### 5.6 Confidence-gated exploitation

Keep this cheap, root-level first. It should not require full robust-opponent search to be useful. It is challenger, not mainline. ([GitHub][2])

Let (w_i) be posterior mass over opponent archetypes and (Q_i(a)) archetype-conditional action value. Define exploitation advantage relative to balanced archetype:

[
\Delta_{\text{exploit}}(a)=\sum_i w_i,[Q_i(a)-Q_{\text{bal}}(a)]
]

Confidence gate:

[
g_{\text{exploit}}=
\mathbf{1}[\max_i w_i \ge 0.70]
\cdot
\mathbf{1}[ESS/P \ge 0.60]
\cdot
\mathbf{1}[H(w)/\log K \le 0.80]
]

Apply only bounded residual:

[
\ell'*a=\ell_a + 0.25,g*{\text{exploit}},\mathrm{clip}(\Delta_{\text{exploit}}(a),-1.5,1.5)
]

This prevents brittle overreaction when style uncertainty is high.

  ### 5.7 Recommended module / interface boundaries

  ```rust
  struct ReplaySafeTargets {
      safety_residual: Option<Tensor>,   // [B,3,34]
      belief_marginal: Option<Tensor>,   // [B,34,4] or [B,37,4]
  }

  struct SearchDerivedTargets {
      delta_q: Option<Tensor>,           // [B,46]
      exit_policy: Option<Tensor>,       // [B,46]
  }

  struct TargetPresenceMask {
      safety_residual: bool,
      belief_marginal: bool,
      delta_q: bool,
      exit_policy: bool,
  }
  ```

Data flow:

`replay/log -> replay-safe label builder -> batch collation -> presence-gated losses`

`hard-state offline search/bridge -> search-derived label builder -> merged HydraTargets -> same loss machinery`

This keeps replay-safe and search-derived provenance separated instead of silently mixing them.

  ## 6. Recommended revised research agenda after reconciliation

  ### Active mainline must-have

  1. Make `HYDRA_RECONCILIATION.md` temporary sequencing authority wherever it conflicts with roadmap/build prompt.
  2. Close replay-safe advanced targets and presence-gated loss activation.
  3. Replace raw belief-field supervision with projected belief supervision.
  4. Rework Hand-EV into real bounded offensive oracle.
  5. Add tranche-specific duplicate evaluation gates before any deeper search work. ([GitHub][2])

  ### Phase-next multipliers

  1. Stronger endgame exactification.
  2. Trust-gated selective AFBS for ExIt / delta-Q labels.
  3. Robust-opponent backups, only if AFBS is already clearly positive.
  4. Confidence-gated exploitation as cheaper challenger. ([GitHub][2])

  ### Reserve shelf worth preserving

  * Incremental / structured belief updates
  * Narrower richer latent opponent posterior
  * Public-event AFBS semantics and better hard-state expansion policy
  * ACH/DRDA as existing architectural assumption, but not new theory agenda ([GitHub][2])

  ### Likely dead ends / no-longer-worth-it complexity

  * broad “search everywhere” AFBS
  * duplicated belief stacks
  * new heads before old heads get real labels
  * deeper optimizer-theory detours right now
  * deception reward and similar speculative extras ([GitHub][2])

  ## 7. Evaluation plan

Use **three levels of gating**.

  ### A. Tranche-health metrics

Before online play, require:

  * target coverage rate by auxiliary target
  * fraction of minibatches with nonzero auxiliary contribution
  * no NaN / Inf / silent-all-None paths
  * gradient-ratio dashboard: aux/core norm ratio
  * belief marginal KL / posterior NLL
  * danger / tenpai / wait-set calibration

This follows reconciliation memo’s acceptance-checklist spirit but makes it quantitative. ([GitHub][2])

  ### B. Slice-based offline evaluation

Maintain fixed state suites for:

  * hard defensive states
  * hand-building / offensive states
  * last-10-draw endgame states
  * South-4 close-placement states

For each suite report:

  * policy agreement vs teacher
  * value error
  * tenpai ECE
  * danger ECE / PR-AUC
  * Hand-EV correlation with exact or MC rollout
  * search gain over base on hard-state slices

  ### C. Duplicate online evaluation

Every promoted change should clear:

  * duplicate paired hanchan vs previous mainline
  * seat-rotated / duplicate seed bundles
  * slice matchups against balanced anchors and style-biased anchors
  * mean placement, 1st/4th rate, deal-in rate, win rate, average score

  ### Failure signals

Stop or revert if any of these happen:

  * replay-safe target coverage stays below `90%` for target meant to be “active”
  * aux/core gradient ratio exceeds `0.35` for sustained windows
  * search-labeled tranche improves offline search agreement but not duplicate online play
  * Hand-EV improves offensive metrics but worsens deal-in or placement enough to erase gain
  * robust or exploitative additions gain against one anchor pool but collapse against balanced anchors

  ### Stopping criteria

Promote tranche only if:

  * offline tranche-health metrics improve on intended targets,
  * duplicate online delta is positive with confidence interval above zero,
  * no major slice regresses catastrophically.

For endgame exactification, I would require explicit `HYDRA_FINAL.md` criterion: improve deal-in, win conversion, placement swing on last-10-draw suites before promotion. ([GitHub][1])

  ## 8. Final recommendation

If I had to reshape Hydra into strongest coherent feasible version of itself after this reconciliation pass, I would make it this:

**Hydra is strong learned policy/value system with calibrated opponent and belief supervision, real offensive Hand-EV oracle, selective hard-state search/endgame layer—not broad search-first project and not theory-first optimizer project.** ([GitHub][1])

Concretely, I would freeze architecture surface for now, make `HYDRA_RECONCILIATION.md` override conflicting sequencing in build prompt and roadmap, close replay-safe advanced labels, supervise projected beliefs instead of raw fields, bring Hand-EV up to “real offensive oracle” level, only then spend major effort on selective AFBS label generation and endgame exactification. If that mainline still underdelivers, I would try **endgame exactification first**, then **robust-opponent / confidence-gated exploitation**, only after that reopen richer latent opponent or structured-belief research. ([GitHub][2])

Strongest surviving insight after pruning is not “Hydra needs more ideas.” It is that Hydra already has more ideas than it can digest. Breakthrough path is making existing advanced surfaces receive **credible targets, calibrated losses, narrow high-leverage search**—then let late-game and opponent-robust layers come in only where they can be trusted. ([GitHub][2])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_FINAL.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/OPPONENT_MODELING.md "raw.githubusercontent.com"
</start-answer>
]]>
</answer_text>
</answer_section>
</combined_run_record>