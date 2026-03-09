<combined_run_record run_id="answer_3-1" variant_id="agent_answers_mixed_transcript" schema_version="1">
  <metadata>
    <notes>Mixed transcript-style file from agent_answers containing both prompt and answer bodies.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="agent_answers/ANSWER_3-1.md" extracted_from="mixed_transcript">
  <![CDATA[# Hydra deep-agent follow-up for ANSWER_3-style agent

  I have attached:
  - `hydra_agent_handoff_docs_only.zip` — this is the primary source material
  - `deep_agent_20_pdfs.zip` — the primary paper/reference package

  Use the docs zip first. If you cannot access the zip, use the raw GitHub markdown links I provide separately. Do NOT rely on normal GitHub browsing/search or generic web search to reconstruct the project context.

  Your job in this prompt is NOT to inspect source code and NOT to write or integrate code. Your job is to look at Hydra’s CURRENT PLAN and make it:
  1. stronger,
  2. more coherent,
  3. more likely to produce real breakthroughs in Mahjong AI,
  4. still grounded enough to be implementable by a separate coding agent later.

  I do NOT want generic brainstorming. I want a hard, evidence-backed **second pruning / prioritization pass** on the plan *after* several key strategic questions have already been settled.

  You should read the docs as a coherent program, especially:
  - `research/design/HYDRA_FINAL.md`
  - `research/design/HYDRA_RECONCILIATION.md`
  - `research/design/IMPLEMENTATION_ROADMAP.md`
- `research/BUILD_AGENT_PROMPT.md` (historical; removed later, see `combined_all_variants/README.md`)
  - `research/design/OPPONENT_MODELING.md`
  - `research/design/TESTING.md`
  - `research/infrastructure/INFRASTRUCTURE.md`
  - `docs/GAME_ENGINE.md`
  - `research/design/SEEDING.md`
  - plus any evidence/comparison docs that materially affect your conclusions

Treat the old `research/BUILD_AGENT_PROMPT.md` reference here as historical execution-rigor context only; for the current routing chain, use `combined_all_variants/README.md` and the live authority docs.

  ## Settled decisions you should treat as fixed inputs

  Do **not** spend time re-arguing these unless you think one is clearly catastrophic:

  - Hydra should **not** restart from scratch.
  - Hydra’s active path is **supervision-first before search-expansion-first**.
  - Use **one unified belief stack**: Mixture-SIB + CT-SMC; no duplicate standalone belief machinery.
  - **Hand-EV comes before deeper AFBS**.
  - **AFBS is selective/specialist**, not broad default runtime.
  - **DRDA/ACH moves off the critical path** and lives on the reserve shelf.
  - **Oracle guidance should be aligned** so the teacher stays teachable.
  - Broad “search everywhere,” duplicated belief stacks, and early optimizer-theory detours are not on the active path.

  External evidence also already supports several pattern-level choices here:
  - unified public-belief-style state abstractions are real, not made-up Hydra weirdness
  - aligned oracle/teacher guidance is more defensible than unconstrained privileged distillation
  - robustness belongs in the core solving/objective layer, though Hydra-specific placement details still require judgment

  You do not need to spend time re-proving those pattern-level claims unless you have a very strong contrary argument.

  ## What I want you to do

  I want you to critique the **current reconciled Hydra plan** and make one more hard call on what remains:

  ### Part 1 — Critique the reconciled active path
  Identify where the reconciled active Hydra path is still:
  - too fragile,
  - too underspecified,
  - too compute-inefficient,
  - too likely to stall before real strength,
  - or still carrying too much reserve-shelf baggage in disguised form.

  ### Part 2 — Re-rank the reserve shelf
  The reconciliation memo keeps several old good ideas on the reserve shelf. I want you to sort them harder:

  - which reserve ideas are actually strong **phase-next** candidates,
  - which are long-shot but worth preserving,
  - which should probably be demoted even further,
  - and which one or two reserve ideas have the best “if active Hydra underdelivers, try this next” upside.

  Focus especially on:
  - robust-opponent search backups vs confidence-gated safe exploitation
  - richer latent opponent posterior
  - deeper AFBS semantics
  - stronger endgame exactification
  - incremental/structured belief updates
  - any remaining optimizer/game-theory ideas

  ### Part 3 — Identify the strongest breakthrough bets that still survive the pruning
  Do not just list cool ideas. I want the **best surviving breakthrough bets after the active-path cuts have already happened**.

  For each surviving bet, tell me:
  - why it is still alive after pruning,
  - why it might matter specifically in Mahjong,
  - what evidence supports it,
  - what assumption it relies on,
  - why it might still fail,
  - and the cheapest meaningful experiment to test it.

  ### Part 4 — Fill in the remaining strategic blanks
  For any reserve or breakthrough idea you keep alive, fill in the missing technical details that the docs still leave abstract:
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
  - Do NOT rely on AGPL code or implementation borrowing.
  - Keep proposals compatible with a separate coding agent implementing them later.

  Assume a separate coding agent will take your response and use it as the strategic decision layer above concrete implementation work.

  ## How to reason about evidence

  Use a strict evidence hierarchy:
  1. direct Mahjong evidence,
  2. direct imperfect-information game AI evidence,
  3. adjacent multiplayer/search/belief modeling evidence,
  4. cross-disciplinary evidence that transfers unusually well.

  When evidence is weak, say so clearly.
  When an idea is speculative, quantify that.
  When you think something is novel but unproven, separate that from evidence-backed recommendations.

  ## Required output format

  Give me the answer in this structure:

  ### 1. Executive verdict
  - Is the reconciled active Hydra plan strong enough to pursue as the mainline?
  - Where is it strongest?
  - Where is it still most likely to fail?
  - Which reserve-shelf idea is most likely to matter next if mainline underdelivers?

  ### 2. Hardest remaining weaknesses in the reconciled active path
  Rank the most important remaining weaknesses or blind spots.

  ### 3. Re-ranked reserve shelf
  For each major reserve idea:
  - keep / demote / mostly drop
  - why
  - evidence basis
  - feasibility
  - upside
  - risk

  ### 4. Best surviving breakthrough bets
  Give me your best 3–5 surviving high-upside ideas.
  For each:
  - novelty level
  - evidence level
  - why it could matter specifically in Mahjong
  - minimum viable experiment
  - what success would look like

  ### 5. Concrete technical fill-ins
  Where the remaining reserve/breakthrough ideas are too vague, provide:
  - formulas
  - losses
  - update rules
  - thresholds
  - hyperparameters
  - pseudocode / compact illustrative snippets
  - interface/data-flow guidance for a coding agent

  ### 6. Recommended revised research agenda after reconciliation
  Give me a revised prioritized roadmap:
  - active mainline must-have
  - phase-next multipliers
  - reserve shelf worth preserving
  - likely dead ends / no-longer-worth-it complexity

  ### 7. Evaluation plan
  Tell me how to know whether the pruned/revised plan is actually better:
  - ablations
  - matchups
  - metrics
  - failure signals
  - stopping criteria

  ### 8. Final recommendation
  If you had to reshape Hydra into the strongest and most coherent feasible version of itself **after this reconciliation pass**, what exact direction would you choose and why?

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
  - shallow novelty for novelty’s sake,
  - or re-arguing already-settled choices unless you have a genuinely strong reason.

  Your goal is to help make Hydra not just more complete, but more formidable and more coherent after the active-path / reserve-shelf split has already been made.
  </start-prompt>
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_answers/ANSWER_3-1.md" extracted_from="mixed_transcript">
  <![CDATA[A quick note before the main answer: the raw `ANSWER_1.md`, `ANSWER_2.md`, and `ANSWER_3.md` links you provided returned 404s in the browser here, so I treated `HYDRA_RECONCILIATION.md` as the authoritative summary of those three answer files, which it explicitly names as deep-agent inputs and synthesizes into the active-path / reserve-shelf split. 

  ## 1. Executive verdict

  Yes: the reconciled active Hydra plan is strong enough to pursue as the mainline. Its strongest parts are now coherent: one unified belief story, supervision-first before search-expansion-first, Hand-EV before deeper AFBS, AFBS as a specialist hard-state tool, and oracle guidance that teaches rather than dominates. That is materially more coherent than the earlier “everything advanced at once” version. ([GitHub][1])

  Where it is still strongest is not “Mahjong AI novelty” in the abstract. It is the combination of: existing advanced model surfaces already present in the repo, the reconciliation memo’s explicit focus on target-generation / supervision loop closure, and the fact that Hydra’s intended ceiling still comes from ExIt + pondering + search-as-feature layered on top of a strong learned base rather than from broad always-on search. ([GitHub][2])

  Where it is most likely to fail is not on the ideas themselves, but on **program coherence**. The branch still contains mutually inconsistent governing docs: `HYDRA_RECONCILIATION.md` says the first tranche should be target/loss closure with no architectural expansion and no broad AFBS rewrite, while `BUILD_AGENT_PROMPT.md` and `IMPLEMENTATION_ROADMAP.md` still describe a from-scratch full build, strict roadmap obedience, early ACH/DRDA implementation, broad AFBS/robust-opponent work, and even stale shapes like `85*34` trajectory observations. If that conflict is not resolved before implementation, the coding agent can “correctly” execute the wrong plan. ([GitHub][2])

  The reserve-shelf idea most likely to matter next if the active mainline underdelivers is **stronger endgame exactification**. It is narrow, high-leverage, compute-bounded, and already endorsed by `HYDRA_FINAL.md` as one of the core ceiling-raising additions. The next most important fallback is **robust-opponent search backups**, but only if selective AFBS already proves valuable on hard states. ([GitHub][2])

  ## 2. Hardest remaining weaknesses in the reconciled active path

  **1. The branch still has a doc-authority bug, not just a design bug.**
  `HYDRA_RECONCILIATION.md` says the first tranche is target-generation / supervision closure using existing output surfaces, no new heads, and no broad AFBS rewrite. `BUILD_AGENT_PROMPT.md` still says `HYDRA_FINAL.md` plus `IMPLEMENTATION_ROADMAP.md` are final and must be followed line by line; the roadmap still encodes a full from-scratch build sequence and stale interfaces. This is the single most dangerous remaining weakness because it can silently reintroduce reserve-shelf baggage into the active path. ([GitHub][2])

  **2. “Close advanced targets” is the right mainline, but the target taxonomy is still underspecified.**
  The reconciliation memo correctly says to populate advanced targets, stage nonzero loss activation, and distinguish replay-derived, bridge-derived, and search-derived labels. But it still leaves ambiguous which targets are actually replay-safe, which require CT-SMC / AFBS context, and in what order they should be activated. In particular, `delta_q_target` and `exit_target` are not replay-safe by default, while the memo’s “preferred order” still groups them too close to replay-only targets. ([GitHub][2])

  **3. The belief-target story is still mathematically sloppy.**
  `HYDRA_FINAL.md` defines the belief machinery in terms of Mixture-SIB fields and CT-SMC search-grade posteriors, and the reconciliation memo lists `belief_fields_target` and `mixture_weight_target` as possible advanced labels. The problem is that raw Sinkhorn external fields are not the right supervision object: they are gauge-like and non-identifiable up to row/column reparameterizations. The right supervised object is the projected belief itself—marginals or gauge-fixed row logits—not raw fields. This is the most important technical blank still left abstract. ([GitHub][1])

  **4. Hand-EV is correctly early, but still not operationally specified enough to avoid becoming “better heuristics” instead of a real offensive oracle.**
  Both `HYDRA_FINAL.md` and the reconciliation memo elevate Hand-EV, and the opponent-modeling stack already assumes value-aware push/fold math. But the branch still does not lock down the exact limited-depth recursion, the simplified ron approximation, the caching key, or the calibration procedure. That is how promising offensive features turn into underpowered heuristics that never quite cash out. ([GitHub][1])

  **5. Selective AFBS is the right posture, but the branch still lacks a hard trust policy for when search labels are usable.**
  `HYDRA_FINAL.md` centers ExIt + pondering + SaF and already includes safety valves like visit-count thresholds and KL caps. The reconciliation memo says AFBS should stay specialist and hard-state gated. What is still missing is a concrete trust rule based on expanded branch mass, particle ESS, visit count, and cross-particle variance. Without that, “selective search” will still supervise noise. ([GitHub][1])

  **6. The opponent-modeling doc still leaks extra complexity into the design space.**
  It is a useful rationale document, but it explicitly says to defer to `HYDRA_FINAL.md` and `HYDRA_RECONCILIATION.md` on implementation direction. It still contains attractive-but-weakly-grounded ideas like deception rewards with arbitrary tuning guidance. Those should not be allowed to re-enter the mainline through “interesting auxiliary” creep. ([GitHub][3])

  ## 3. Re-ranked reserve shelf

  ### Stronger endgame exactification — **keep, phase-next #1**

  **Why:** It is narrow, compatible with selective compute, and directly aligned with `HYDRA_FINAL.md`’s endgame-exactification rationale and validation gate.
  **Evidence basis:** strongest direct Mahjong-specific support among reserve ideas; the doc explicitly says late-game decisions are disproportionately high-EV and proposes exact chance enumeration when the wall is short.
  **Feasibility:** medium.
  **Upside:** high.
  **Risk:** moderate but bounded. ([GitHub][2])

  ### Robust-opponent search backups — **keep, phase-next #2**

  **Why:** This is one of the few reserve ideas that is also part of Hydra’s intended final identity. But it only pays once local evaluators and selective AFBS are already useful.
  **Evidence basis:** strong inside Hydra docs, weaker as an immediate mainline sequencing decision.
  **Feasibility:** medium-high if AFBS is already positive; poor otherwise.
  **Upside:** medium-high at the last strength mile.
  **Risk:** medium-high because it can become expensive sophistication on top of noisy search. ([GitHub][1])

  ### Confidence-gated safe exploitation — **keep, but as a cheaper challenger, not as a coequal mainline**

  **Why:** If the mainline underdelivers and AFBS is still too expensive, a root-level exploitation layer is a plausible lower-cost alternative to full robust-opponent search.
  **Evidence basis:** indirect; the reconciliation memo groups “robust-opponent search backups / safe exploitation” together as worth preserving, but does not yet separate them.
  **Feasibility:** medium.
  **Upside:** medium.
  **Risk:** high if opponent posterior calibration is poor; lower if gated hard by uncertainty. ([GitHub][2])

  ### Incremental / structured belief updates — **keep high as the main contingency branch**

  **Why:** This is the best reserve response if the unified Mixture-SIB + CT-SMC stack proves too slow or too blurry.
  **Evidence basis:** medium; Hydra already treats belief misspecification as the core risk and keeps structured belief ideas on the reserve shelf.
  **Feasibility:** medium.
  **Upside:** high if current belief quality tops out early.
  **Risk:** medium because belief research can sprawl. ([GitHub][2])

  ### Richer latent opponent posterior — **keep, but demote below endgame and belief contingencies**

  **Why:** The long-term direction toward more unified opponent modeling is reasonable, but the reconciliation memo is right that the immediate bottleneck is not lack of outputs.
  **Evidence basis:** low-medium.
  **Feasibility:** medium.
  **Upside:** medium.
  **Risk:** high risk of inventing new machinery before existing heads are properly trained. ([GitHub][2])

  ### Deeper AFBS semantics — **keep, but narrow the scope**

  **Why:** Preserve one specific upgrade path—public-event semantics and better hard-state expansion rules—but do not preserve “make AFBS deeper/broader” as a vague bucket.
  **Evidence basis:** medium.
  **Feasibility:** medium-high if kept narrow.
  **Upside:** medium-high.
  **Risk:** high if it drifts back into broad-search identity. ([GitHub][2])

  ### Optimizer / game-theory additions beyond the existing ACH/DRDA assumption — **demote hard**

  **Why:** `HYDRA_FINAL.md` still includes ACH and DRDA as part of the final architecture, but the reconciliation memo is explicit that immediate progress should not depend on resolving optimizer-level debates.
  **Evidence basis:** mixed.
  **Feasibility:** high in implementation terms, low in “best next use of attention.”
  **Upside:** unclear in the short term.
  **Risk:** very high opportunity cost. ([GitHub][1])

  ### Deception reward / novelty-heavy ToM extras — **mostly drop**

  **Why:** The opponent-modeling doc itself admits arbitrary starting gamma and no prior basis for tuning. That is exactly the kind of reserve-shelf baggage the reconciliation memo says to stop letting drive the mainline.
  **Evidence basis:** weak.
  **Feasibility:** medium.
  **Upside:** speculative.
  **Risk:** very high. ([GitHub][3])

  ## 4. Best surviving breakthrough bets

  ### A. Gauge-fixed belief supervision on top of the unified Mixture-SIB + CT-SMC stack

  **Novelty level:** medium.
  **Evidence level:** medium-high inside Hydra’s own design.
  **Why it survives pruning:** it strengthens the mainline instead of creating a second belief stack.
  **Why it matters in Mahjong:** tile conservation and correlation structure are unusually central here; bad posterior shape poisons both search and safety.
  **Assumption:** CT-SMC already produces a meaningfully better search-grade posterior than generic mean-field approximations.
  **Why it might fail:** if the posterior is still too blurry or expensive, the supervision target is not strong enough to teach the backbone anything useful.
  **Cheapest meaningful experiment:** compare three auxiliary schemes on the same model surface: raw field regression, projected marginal supervision, and no belief auxiliary. Measure posterior NLL, pairwise MI calibration, wait-set calibration, and downstream hard-state policy gain. Success looks like projected marginals clearly beating raw-field regression. ([GitHub][1])

  ### B. Hand-EV realism as the first serious offensive multiplier

  **Novelty level:** low-medium.
  **Evidence level:** high by Hydra’s own evidence standard.
  **Why it survives pruning:** it is already wired into the architecture, cheaper than deeper search, and directly endorsed by both `HYDRA_FINAL.md` and the reconciliation memo.
  **Why it matters in Mahjong:** offensive tempo, tenpai timing, and value realization are central; weak offensive forecasts distort both push/fold and search priors.
  **Assumption:** limited-depth self-draw recursion with a simplified ron model is enough to improve decision quality before full AFBS expansion.
  **Why it might fail:** if the recursion is too greedy or the ron approximation is badly miscalibrated.
  **Cheapest meaningful experiment:** add only improved Hand-EV planes to the current learned policy/value stack and run duplicate evaluation against the same baseline, plus offline correlation against small-wall exact or MC rollout slices. Success looks like online gain without needing live search. ([GitHub][1])

  ### C. Stronger endgame exactification

  **Novelty level:** medium.
  **Evidence level:** medium-high.
  **Why it survives pruning:** it is selective, bounded, and consistent with Hydra’s “specialist search” identity.
  **Why it matters in Mahjong:** late-game push/fold and placement swings are massively leverage-heavy relative to average decisions.
  **Assumption:** exact chance over the live wall and better terminal utility matter more than broader midgame search.
  **Why it might fail:** if the trigger policy is too broad and compute blows up, or too narrow and the gains are negligible.
  **Cheapest meaningful experiment:** a last-10-draw exactification benchmark with duplicate pairing and slice metrics for orasu / riichi-defense contexts. Success looks like simultaneous improvement in deal-in rate, win conversion, and placement swing. ([GitHub][2])

  ### D. Trust-gated selective AFBS feeding ExIt / delta-Q supervision

  **Novelty level:** medium.
  **Evidence level:** medium.
  **Why it survives pruning:** AFBS remains core to Hydra’s ceiling, but only if converted from “broad expensive search” into a reliable hard-state teacher.
  **Why it matters in Mahjong:** the hardest decisions are sparse but decisive; specialist search can change them without dominating total latency.
  **Assumption:** a small fraction of states carry most of the search value, and Hydra can identify them well enough.
  **Why it might fail:** if the hard-state gate is too loose, or if search labels are too noisy across particles/archetypes.
  **Cheapest meaningful experiment:** generate ExIt labels only on a hard-state slice with strict trust filters; compare policy-learning gain against the same compute spent on more BC/RL. Success looks like positive duplicate match delta and high “search gain recaptured by SaF/offline distillation.” ([GitHub][1])

  ### E. Confidence-gated exploitation as the cheaper “next after mainline” challenger

  **Novelty level:** medium.
  **Evidence level:** low-medium.
  **Why it survives pruning:** it is one of the few reserve ideas with real upside that does not force Hydra back into broad search or extra head sprawl.
  **Why it matters in Mahjong:** opponent styles are exploitable, but brittle exploitation is deadly in multiplayer general-sum settings.
  **Assumption:** Hydra can produce a calibrated enough style posterior to exploit only when confidence is high.
  **Why it might fail:** posterior confidence may be overestimated; exploitation may simply overfit anchor styles.
  **Cheapest meaningful experiment:** root-level exploitation layer only, tested against style-specific anchor pools and balanced anchors, with hard uncertainty gating. Success looks like gain against style-biased pools without measurable collapse against balanced opponents. ([GitHub][2])

  ## 5. Concrete technical fill-ins

  ### 5.1 Target presence and staged activation

  The tranche should use an explicit presence-gated loss policy, not “weights default to zero and maybe later become nonzero.” The reconciliation memo already points in this direction; I would make it formal. ([GitHub][2])

  Define, for each auxiliary target (j):

  [
  m_j = \mathbf{1}{\text{tensor exists} \land \text{finite} \land \text{sane range}}
  ]

  [
  w_j(t) = w^{\max}_j \cdot \mathrm{clip}!\left(\frac{t-s_j}{r_j}, 0, 1\right)
  ]

  [
  L_{\text{total}} = L_{\text{base}} + \sum_j m_j , w_j(t), \tilde L_j
  ]

  with normalized auxiliary loss

  [
  \tilde L_j = \frac{L_j}{\operatorname{EMA}(L_j)+10^{-6}}
  ]

  and a gradient cap

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

  `HYDRA_FINAL.md` defines belief in terms of SIB/Mixture-SIB fields and projected beliefs, while the reconciliation memo tentatively lists `belief_fields_target` as a candidate advanced label. I would not supervise raw fields. ([GitHub][1])

  Use the projected belief (B_t(k,z)) as the supervised object:

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

  but only if component labels come from a consistent offline fitting procedure. Otherwise leave `mixture_weight_target` absent.

  A good compromise target for “field-like” supervision is a gauge-fixed row logit:

  [
  g_{k,z} = \log(B_{k,z}+10^{-8}) - \frac{1}{4}\sum_{z'}\log(B_{k,z'}+10^{-8})
  ]

  This preserves rowwise relative preference without trying to supervise non-identifiable raw fields.

  ### 5.3 Hand-EV realism

  Use a bounded-depth self-draw recursion over discard candidates. This operationalizes the Hand-EV planes already defined in `HYDRA_FINAL.md`. ([GitHub][1])

  For discard (a), horizon (d\in{1,2,3}), and live-wall counts (r):

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

  Use `kappa_ron = 1.0` initially and calibrate against exact/MC slices.

  Pruning:

  * if shanten (\le 1): expand all effective draws
  * else: top 12 draws by `remaining[t] * immediate_gain(t)`
  * depth (\ge 2): keep top 3 discard continuations

  ### 5.4 Search trust gate and ExIt eligibility

  `HYDRA_FINAL.md` already gives `min_visits`, KL safety valves, hard-state signals, and playout-cap randomization cues. The missing piece is a unified trust weight. ([GitHub][1])

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

  This keeps ExIt specialist and stops noisy search from pretending to be ground truth.

  ### 5.5 Endgame exactification

  `HYDRA_FINAL.md` already fixes the motivation and suggests exactification once the wall is short. I would turn that into a two-trigger policy. ([GitHub][1])

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

  Keep this cheap and root-level first. It should not require full robust-opponent search to be useful. It is a challenger, not the mainline. ([GitHub][2])

  Let (w_i) be posterior mass over opponent archetypes and (Q_i(a)) the archetype-conditional action value. Define exploitation advantage relative to the balanced archetype:

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

  Apply only a bounded residual:

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

  That keeps replay-safe and search-derived provenance separated instead of silently mixing them.

  ## 6. Recommended revised research agenda after reconciliation

  ### Active mainline must-have

  1. Make `HYDRA_RECONCILIATION.md` the temporary sequencing authority wherever it conflicts with the roadmap/build prompt.
  2. Close replay-safe advanced targets and presence-gated loss activation.
  3. Replace raw belief-field supervision with projected belief supervision.
  4. Rework Hand-EV into a real bounded offensive oracle.
  5. Add tranche-specific duplicate evaluation gates before any deeper search work. ([GitHub][2])

  ### Phase-next multipliers

  1. Stronger endgame exactification.
  2. Trust-gated selective AFBS for ExIt / delta-Q labels.
  3. Robust-opponent backups, only if AFBS is already clearly positive.
  4. Confidence-gated exploitation as a cheaper challenger. ([GitHub][2])

  ### Reserve shelf worth preserving

  * Incremental / structured belief updates
  * Narrower richer latent opponent posterior
  * Public-event AFBS semantics and better hard-state expansion policy
  * ACH/DRDA as existing architectural assumption, but not a new theory agenda ([GitHub][2])

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

  This follows the reconciliation memo’s acceptance-checklist spirit but makes it quantitative. ([GitHub][2])

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

  * replay-safe target coverage stays below `90%` for a target meant to be “active”
  * aux/core gradient ratio exceeds `0.35` for sustained windows
  * search-labeled tranche improves offline search agreement but not duplicate online play
  * Hand-EV improves offensive metrics but worsens deal-in or placement enough to erase gain
  * robust or exploitative additions gain against one anchor pool but collapse against balanced anchors

  ### Stopping criteria

  Promote a tranche only if:

  * offline tranche-health metrics improve on intended targets,
  * duplicate online delta is positive with a confidence interval above zero,
  * no major slice regresses catastrophically.

  For endgame exactification, I would require the explicit `HYDRA_FINAL.md` criterion: improve deal-in, win conversion, and placement swing on last-10-draw suites before promotion. ([GitHub][1])

  ## 8. Final recommendation

  If I had to reshape Hydra into the strongest coherent feasible version of itself after this reconciliation pass, I would make it this:

  **Hydra is a strong learned policy/value system with calibrated opponent and belief supervision, a real offensive Hand-EV oracle, and a selective hard-state search/endgame layer—not a broad search-first project and not a theory-first optimizer project.** ([GitHub][1])

  Concretely, I would freeze the architecture surface for now, make `HYDRA_RECONCILIATION.md` override conflicting sequencing in the build prompt and roadmap, close replay-safe advanced labels, supervise projected beliefs instead of raw fields, bring Hand-EV up to “real offensive oracle” level, and only then spend major effort on selective AFBS label generation and endgame exactification. If that mainline still underdelivers, I would try **endgame exactification first**, then **robust-opponent / confidence-gated exploitation**, and only after that reopen richer latent opponent or structured-belief research. ([GitHub][2])

  The strongest surviving insight after pruning is not “Hydra needs more ideas.” It is that Hydra already has more ideas than it can currently digest. The breakthrough path is to make the existing advanced surfaces receive **credible targets, calibrated losses, and narrow high-leverage search**—then let the late-game and opponent-robust layers come in only where they can actually be trusted. ([GitHub][2])

  [1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_FINAL.md "raw.githubusercontent.com"
  [2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md "raw.githubusercontent.com"
  [3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/OPPONENT_MODELING.md "raw.githubusercontent.com"
  </start-answer>
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
