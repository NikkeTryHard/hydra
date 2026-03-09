<combined_run_record run_id="answer_1-1" variant_id="agent_answers_mixed_transcript" schema_version="1">
  <metadata>
    <notes>Mixed transcript-style file from agent_answers containing both prompt and answer bodies.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="agent_answers/ANSWER_1-1.md" extracted_from="mixed_transcript">
  <![CDATA[## Primary working package

  I have attached a zip file to this prompt called `hydra_agent_handoff_docs_only.zip`.

  Use that zip as the **primary docs package** instead of trying to reconstruct the project from browsing.

  Expected workflow:
  1. Open / extract `hydra_agent_handoff_docs_only.zip` and treat the extracted markdown docs as the primary source material.
  2. Read the included docs carefully before forming conclusions.
  3. Use the raw GitHub markdown links below only as supplemental reference / cross-check material.
  4. Use the attached PDF package as the primary paper attachment set.

  If you cannot access the attached zip for any reason, fall back to fetching the markdown docs directly from the **raw GitHub links** in this document.

  Important:
  - Do **not** rely on normal GitHub browsing/search to reconstruct the repo.
  - Do **not** rely on generic/plain web search to discover the project files.
  - If the zip is unavailable, fetch the raw markdown docs directly from the raw links in this handoff instead.

  You are a deep-thinking **research and design advisor** for **Hydra**, a Riichi Mahjong AI project whose goal is to reach or exceed **LuckyJ-level** strength.

  Your job is **not** to inspect source code, browse loosely, or directly modify anything. Your job is to read the design docs and papers, think very hard about the hardest and most underspecified parts, and then produce the strongest possible technical guidance for a separate coding agent to implement later.

  Treat the following as the governing hierarchy:

  1. `research/design/HYDRA_FINAL.md` = the architectural SSOT for final strength
  2. `research/design/HYDRA_RECONCILIATION.md` = the reconciled active-path / reserve-shelf / dropped-shelf decision memo
  3. `research/design/IMPLEMENTATION_ROADMAP.md` = the implementation ordering and gates
4. `research/BUILD_AGENT_PROMPT.md` = historical execution-discipline overlay on the other docs (removed later; see `combined_all_variants/README.md` for the current routing chain)
  5. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

  ## Resolved decisions you should treat as fixed inputs

  These are no longer open questions for this follow-up:

  - **Unified belief stack:** Mixture-SIB = amortized belief, CT-SMC = search-grade posterior, no duplicated standalone belief pipeline.
  - **Hand-EV ordering:** Hand-EV comes before deeper AFBS expansion.
  - **AFBS scope:** AFBS is selective / specialist / hard-state-gated, not broad default runtime.
  - **Training-core status:** DRDA/ACH is not on the critical path; keep it as reserve/challenger direction.
  - **Oracle guidance:** privileged teacher for oracle critic, public teacher for belief/search targets, aligned guider/learner setup.
  - **Robust opponent logic:** eventually inside search backup/opponent-node semantics, not merely helper math.

  External evidence already supports these directions at least at the pattern level:
  - unified public-belief-style representations are a serious imperfect-information design pattern
  - aligned oracle/teacher guidance should use the same query/target semantics as the learner where possible
  - robustness belongs in the solver/search objective layer, not as a random bolt-on heuristic

  Do not spend your budget re-proving those broad patterns. Focus on the Hydra-specific technical fill-ins.

  ## What you should focus on

  You were strongest on **technical fill-ins** and precise algorithms. Focus only on the still-hard technical gaps that remain after the above decisions are fixed:

  1. **Unified belief stack mechanics**
     - exact Mixture-SIB -> CT-SMC data flow
     - event-likelihood update mechanics
     - final hidden-state granularity (34 vs 37 tile rows, aka handling, dead-wall handling)
     - what should be cached, quantized, and hashed for runtime reuse

  2. **Hand-EV as a real offensive oracle**
     - particle-averaged offensive evaluator over CT-SMC worlds
     - exact recurrence / DP structure
     - minimum viable ron model vs stronger version
     - practical runtime shortcuts that still preserve strength

  3. **Selective AFBS internals**
     - exact hard-state / VOC gating formulas
     - shallow specialist AFBS semantics for offense / defense / endgame
     - what the minimum viable public-event AFBS should contain in v1 and what to defer

  4. **Exactification boundaries**
     - when Hand-EV yields to endgame solving
     - when CT-SMC is enough vs when a more exact late-game solver should activate
     - exact trigger thresholds and fallback behavior

  5. **Failure modes and calibration**
     - belief collapse
     - overconfident event models
     - CPU blow-up in Hand-EV / AFBS
     - target leakage from hidden-state labels into information-state heads

  ## What kind of answer is wanted

  Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
  - formulas over vague prose
  - precise algorithms over general suggestions
  - concrete thresholds/hyperparameters over hand-waving
  - pseudocode or compact code snippets where implementation detail matters
  - explicit tradeoff analysis
  - ablation/evaluation plans tied to Hydra’s stated architecture
  - discussion of what remains underspecified and the best way to fill those gaps

  Avoid spending your budget on:
  - re-arguing already-settled high-level direction
  - broad summaries of Mahjong AI history unless they directly affect design choices
  - generic “run more experiments” suggestions without exact experiments
  - pretending to have implemented or validated anything
  - trying to inspect source code instead of reasoning from docs, evidence, and the reconciled decisions above

  ## Required deliverables

  Produce a technical design package for a separate coding agent. Your deliverables should be:

  1. A prioritized analysis of the **remaining** hard/underspecified Hydra technical problems after the resolved decisions above are treated as fixed.
  2. For each major problem, a concrete proposed solution including:
     - the exact algorithmic idea
     - formulas / objective functions / update rules
     - recommended constants, thresholds, and hyperparameters
     - what data or targets would be needed
     - runtime/inference-time behavior
     - training-time behavior
     - likely failure modes and mitigations
     - evaluation criteria and ablations
  3. Pseudocode or compact illustrative code snippets for the hardest parts.
  4. Recommended module boundaries / interfaces / data flows a coding agent should implement.
  5. A practical implementation order for the remaining technical stack.
  6. A concise risk assessment identifying which proposals are strongest, which are fragile, and which are likely not worth the complexity.

  ## Success condition

  Your output should make it materially easier for a separate coding agent to implement the remaining Hydra improvements correctly and efficiently, especially:
  - unified belief mechanics
  - particle-averaged Hand-EV
  - selective AFBS semantics
  - endgame trigger logic

  The goal is to produce the best possible technical fill-ins for the **hard math / algorithm** parts that still matter after the strategic direction has already been settled.

  ## Additional context references

  Use these as first-class references in addition to the original hierarchy:
  - `ANSWER_1.md`
  - `ANSWER_2.md`
  - `ANSWER_3.md`
  - `research/design/HYDRA_RECONCILIATION.md`

  Treat `HYDRA_RECONCILIATION.md` as the active-path decision memo, not as optional commentary.
  </start-prompt>
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_answers/ANSWER_1-1.md" extracted_from="mixed_transcript">
  <![CDATA[I’m using the attached Hydra handoff plus the reconciliation memo and answer files as the governing brief for this pass.  The remaining blockers are not missing modules; they are partially closed loops: belief updates are still under-pinned by the event model, Hand-EV is still too heuristic, AFBS is still scaffold rather than a real public-belief improvement engine, and endgame exactification is still weighted aggregation rather than a finite-horizon solver.

  ## Remaining technical problems

  1. **Belief update mechanics are still the real bottleneck.**
     The settled stack is correct — Mixture-SIB for amortized belief and CT-SMC for search-grade posterior — but the missing part is the exact event-likelihood bridge and the runtime update rule. The reconciliation memo is explicit that CT-SMC should be treated as a source of credible belief-weighted targets and search-grade posterior, not as a new parallel belief project.

  2. **Hand-EV is not yet a real offensive oracle.**
     The active path explicitly moves Hand-EV realism ahead of deeper AFBS, and the answer files are consistent that the current heuristic should be replaced by a short-horizon particle-averaged offensive DP with at least a cheap ron model.

  3. **AFBS needs exact gating and minimum viable semantics.**
     The active path says AFBS must stay specialist and hard-state gated; the technical gap is the exact trigger, node semantics, root object, and backup rule.

  4. **Endgame exactification needs hard boundaries.**
     The current “wall-small” idea is directionally right, but the actual trigger, exact-vs-sparse split, and leaf utility schedule need to be fixed precisely before a coding agent can implement it correctly.

  5. **Runtime-safe approximations and search trust are under-specified.**
     Hydra needs explicit rules for when to approximate, when to trust search labels, and when to abort expensive branches. Otherwise the system will either explode latency or teach itself noise.

  ---

  ## 1. Exact recommended solution: unified belief mechanics

  ### State representation

  Use one belief object:

  [
  \mathcal B_t=\big({(w_\ell,B_\ell)}*{\ell=1}^L,\ {(\alpha_p,X^{(p)})}*{p=1}^P,\ \text{ESS},\ H(w),\ n_{\text{live}},n_{\text{dead}},a_{\text{aka}}\big)
  ]

  with:

  * (L=4) mixture components on-turn, (L=8) in ponder
  * (P=128) particles on-turn, (P=512) in ponder
  * core CT state (X^{(p)} \in \mathbb Z_+^{34\times 4}) over zones `{oppL, oppA, oppR, unseen}`
  * **do not** upgrade the core CT table to 37 rows right now
  * instead keep aka information as side metadata (a_{\text{aka}}\in{0,1}^3)
  * keep live/dead wall split as delayed refinement, not part of the always-on CT table

  That is the cleanest implementable decision. It preserves the settled unified belief stack, avoids a sampler redesign, and still supports late-game exactification.

  ### Event update rule

  Use a two-level update.

  Component update:
  [
  w'*\ell \propto w*\ell , p_\phi(e_t \mid I_t,B_\ell,\ell)
  ]

  Particle update:
  [
  \alpha'_p \propto \alpha_p , L(e_t \mid X^{(p)},I_t)
  ]

  Until a dedicated event model is fully trained, use the hybrid likelihood from the answer files:

  [
  \log L(e\mid X,I)=
  1.0 \log p_{\text{next}}(e)
  +0.5 \log p_{\text{tenpai}}(e)
  +0.5 \log h_{\text{call}}(e\mid X,I)
  +4.0 \log \mathbf 1[\text{event legal under }X]
  ]

  Then clamp the total increment:

  [
  \Delta \log L \leftarrow \operatorname{clip}(\Delta \log L,,-10,,0)
  ]

  Resample when:

  [
  \mathrm{ESS} = \frac{1}{\sum_p \bar\alpha_p^2} < 0.4P
  ]

  and run 2 MH-style rejuvenation sweeps on-turn, 4 in ponder, using only margin-preserving row/column cycle moves. The source material already fixes the key thresholds here: ESS at 0.4P and the hybrid likelihood decomposition are the right starting point.

  ### Late live/dead split

  Only activate when `wall <= 10` or exact score/ura/rinshan logic is needed.

  Given unseen row counts (u_k) and desired live count (N_{\text{live}}), sample a live/dead split (y_k) with:

  [
  \sum_k y_k = N_{\text{live}}, \qquad 0 \le y_k \le u_k
  ]

  using the DP:

  [
  Z_k(c)=\sum_{y=0}^{\min(u_k,c)} \binom{u_k}{y} Z_{k+1}(c-y), \qquad Z_{35}(0)=1
  ]

  and backward sample (y_k) from:

  [
  p(y_k=y \mid c) \propto \binom{u_k}{y} Z_{k+1}(c-y)
  ]

  This keeps the always-on belief cheap and pushes exact wall semantics only into the late mode.

  ### Risk composition from belief

  Danger should not be one monolithic scalar. Use:

  [
  r_i(k\mid s)=\hat p(T_i=1\mid s)\cdot \hat p(k\in W_i\mid s)\cdot \hat p(\text{ron-legal}\mid k,s)
  ]

  and aggregate:

  [
  p_\cup(k\mid s)=1-\prod_{i=1}^{3}(1-r_i(k\mid s))
  ]

  Make the wait-set head core, not optional. That is one of the few genuinely high-leverage opponent-modeling decisions that survives all the pruning.

  ### Pseudocode

  ```python
  def update_belief(public_state, event, mixture, particles):
      # component update
      for l in range(len(mixture)):
          mixture[l].logw += log_event_component(public_state, event, mixture[l])

      mixture = normalize_components(mixture)

      # particle update
      for p in particles:
          ll = (
              1.0 * log_p_next(public_state, event, p)
            + 0.5 * log_p_tenpai(public_state, event, p)
            + 0.5 * log_call_heuristic(public_state, event, p)
            + 4.0 * log_legality(event, p)
          )
          p.logw += clip(ll, -10.0, 0.0)

      particles = normalize_particles(particles)

      if ess(particles) < 0.4 * len(particles):
          particles = stratified_resample(particles)
          particles = rejuvenate_margin_preserving(particles, steps=2)

      return mixture, particles
  ```

  ### Failure modes

  * **Belief collapse**: event model too sharp.
    Mitigation: clip event increments, temperature-scale event heads, enforce minimum component weight 0.03.

  * **Wall semantics wrong late**: exact draw and score paths drift.
    Mitigation: delayed live/dead split only in late mode.

  * **Duplicated belief implementations**: calibration drift and debugging chaos.
    Mitigation: one belief object only; no parallel belief stack.

  ### Evaluation gate

  Ship belief changes only if all are true:

  * posterior hidden-state NLL improves over the current baseline
  * wait-set Brier / PR-AUC improves
  * danger ECE improves in riichi and multi-threat buckets
  * ESS stays above 0.3P median after update on held-out games
  * no more than 15% mixture collapse into a single component before late hand

  ---

  ## 2. Exact recommended solution: Hand-EV recurrence and approximations

  ### Output

  For each legal discard (a), emit:

  [
  \mathrm{HEV}(a)=\big(
  P_{\text{ten}}^{(1)},P_{\text{ten}}^{(2)},P_{\text{ten}}^{(3)},
  P_{\text{win}}^{(1)},P_{\text{win}}^{(2)},P_{\text{win}}^{(3)},
  \mathbb E[\text{score}\mid \text{win}],\ \text{ukeire}*1,\ v*{\text{off}}
  \big)
  ]

  ### Particle averaging

  Do not compute Hand-EV from one marginal count vector if CT-SMC exists. Use:

  [
  \mathrm{HEV}(I,a)=\sum_{p=1}^{P_{\text{hand}}}\alpha_p ,\mathrm{HEV}(I,X^{(p)},a)
  ]

  with:

  * (P_{\text{hand}}=16) on-turn
  * (P_{\text{hand}}=64) in ponder / offline labels

  ### Per-particle DP

  For a post-discard hand (h), exact live-wall multiset (w), and horizon (d\in{1,2,3}), define:

  [
  T(h,w,d)=\left(v_{\text{off}},P_{\text{tenpai}},P_{\text{win}},S_{\text{win}},u_1\right)
  ]

  Base:

  * if agari: (P_{\text{tenpai}}=1,\ P_{\text{win}}=1,\ S_{\text{win}}=\text{score}_{\text{exact}})
  * if (d=0): (P_{\text{tenpai}}=\mathbf 1[\text{tenpai}],\ P_{\text{win}}=0,\ S_{\text{win}}=0)

  Recurrence:
  [
  T(h,w,d)=
  \sum_{u:w_u>0}\frac{w_u}{|w|}
  \begin{cases}
  (\frac{s(u)}{12000},1,1,s(u),0) & \text{if } h+u \text{ is agari}\
  T(h+u-b^*,w-e_u,d-1) & \text{otherwise}
  \end{cases}
  ]

  with:
  [
  b^*=\arg\max_b v_{\text{off}}(h+u-b,w-e_u,d-1)
  ]

  and scalar:
  [
  v_{\text{off}}
  ==============

  P_{\text{win}}\cdot \frac{\mathbb E[\text{score}\mid \text{win}]}{12000}
  +0.05,P_{\text{tenpai}}
  +0.01,\frac{u_1}{20}
  ]

  Tie-break equal (v_{\text{off}}) by higher (P_{\text{win}}), then (P_{\text{tenpai}}), then (u_1), then expected score. This is the most stable choice for a feature oracle: scalar primary, lexicographic tie-break.

  ### Minimum viable ron model

  Do not ship tsumo-only Hand-EV.

  For wait set (W), define:

  [
  P_i^{\le m}(t)=1-(1-p_{\text{next},i}(t))^m
  ]

  [
  m_i(d)=\left\lceil 0.8d + 0.4\cdot \text{open_melds}_i + 0.8\cdot \mathbf 1[\text{riichi}_i]\right\rceil
  ]

  Then:
  [
  P_{\text{ron}}^{(d)}(W)=
  1-\prod_{i=1}^{3}\left(1-\sum_{t\in W} P_i^{\le m_i(d)}(t)\right)
  ]

  [
  P_{\text{win}}^{(d)}=
  1-(1-P_{\text{tsumo}}^{(d)})(1-P_{\text{ron}}^{(d)})
  ]

  [
  \mathbb E[\text{score}\mid \text{win}] =
  \frac{P_{\text{tsumo}}\mathbb E[s_{\text{tsumo}}] + P_{\text{ron}}\mathbb E[s_{\text{ron}}]}
  {\max(P_{\text{win}},10^{-6})}
  ]

  Use (\kappa_{\text{ron}}=1.0). Exact scoring calls go through the engine, not heuristics.

  ### Runtime-safe pruning

  Use exactly these rules:

  * if current shanten (\le 1): expand all effective draws
  * else: expand top 12 draws or until cumulative mass (\ge 0.97)
  * at depth (\ge 2): keep only top 3 post-draw discards
  * memoize by `(hand_signature, wall_signature, dora, seat, riichi_flags, depth)`
  * if CT-SMC unavailable: fall back to public remaining counts
  * if wall (\le 10): Hand-EV is advisory only; endgame solver overrides

  ### Pseudocode

  ```python
  @memoize
  def offensive_stats(hand13, wall, depth, ctx):
      if is_agari(hand13, ctx):
          s = exact_score(hand13, win_mode="tsumo", ctx=ctx)
          return Stats.win_now(s)

      if depth == 0:
          return Stats.from_tenpai(is_tenpai(hand13), ukeire_mass(hand13, wall))

      acc = Stats.zero()
      for u, p_u in top_mass_draws(wall, mass=0.97, cap=12):
          hand14 = add_tile(hand13, u)

          if is_agari(hand14, ctx):
              s_t = exact_score(hand14, win_mode="tsumo", ctx=ctx)
              ron = simplified_ron_hazard(hand14, depth, ctx)
              acc += p_u * Stats.agari(s_t, ron)
              continue

          best = None
          for b in top_discards(hand14, cap=3 if depth <= 2 else None):
              nxt = offensive_stats(remove_tile(hand14, b), wall - onehot(u), depth - 1, ctx)
              best = choose_better(best, nxt)  # v_off primary, lexicographic tiebreak
          acc += p_u * best

      return acc
  ```

  ### Failure modes

  * **Greedy continuation bias**: acceptable because Hand-EV is a feature oracle, not the final solver.
  * **Ron too optimistic**: calibrate (m_i(d)) and (p_{\text{next},i}) buckets on held-out data.
  * **CPU blow-up**: hard-cap draws at 12 / 0.97 mass and post-draw discards at 3.

  ### Evaluation gate

  Promote Hand-EV only if:

  * top-1 discard agreement vs deeper AFBS improves
  * shanten-1 and shanten-0 win conversion improves
  * expected score on tenpai-entry states improves
  * p95 CPU latency stays within budget
  * tsumo+ron beats tsumo-only and current heuristic in ablation

  ---

  ## 3. Exact recommended solution: AFBS gating and minimum viable public-event semantics

  ### Hard-state gate

  Use one deterministic gate, not multiple alternatives.

  Run AFBS iff one of these mode predicates is true:

  **Defense mode**

  * (p_\cup(a^*) \ge 0.08), or
  * any threat present and (p_\cup(a^*) \ge 0.06)

  **Close-offense mode**

  * top-2 policy gap (< 0.05), and
  * top-2 Hand-EV gap (< 0.02), and
  * (p_\cup(a^*) < 0.08)

  **Endgame mode**

  * wall (\le 10), and
  * any of: riichi present, ( \max p_{\text{tenpai}} \ge 0.35), oorasu, (|\text{score to next place}| \le 12000)

  Otherwise: no AFBS. This matches the settled “specialist, not default runtime” direction and is concrete enough to implement immediately.

  ### Tree semantics

  Use a public-event tree with four node types:

  1. `MyDecision`
  2. `MyFutureDrawChance`
  3. `OpponentPublicEvent`
  4. `Terminal`

  Do **not** branch on opponent hidden draws. Opponent turns branch on public events only: discard, riichi+discard, call, pass. That is the minimum viable information-state search semantics and avoids determinization-style strategy fusion.

  ### Root result

  Make `SearchRootResult` first-class and feed all three consumers: ExIt, SaF, and ponder/inference cache.

  ```rust
  struct SearchRootResult {
      q: [f32; 46],
      visits: [u32; 46],
      exit_policy: [f32; 46],
      risk_upper: [f32; 46],
      risk_est: [f32; 46],
      entropy_drop: [f32; 46],
      robust_tau: [f32; 46],
      q_var: [f32; 46],
      ess_after: [f32; 46],
      root_value: f32,
  }
  ```

  ### Backup rules

  At self nodes:
  [
  a^*=\arg\max_a \left(\bar Q(a)+2.5,P(a)\frac{\sqrt{N}}{1+n(a)}\right)
  ]

  At opponent nodes:

  [
  q_{m,\tau}(b)\propto p_m(b)\exp(-Q_m(b)/\tau)
  ]
  with (\tau) chosen so that
  [
  D_{\mathrm{KL}}(q_{m,\tau}|p_m)=\varepsilon_{\mathrm{eff}}(\text{ctx})
  ]

  [
  V_m=\sum_b q_{m,\tau}(b)Q_m(b)
  ]

  [
  V_{\text{opp}}=-0.7\log\sum_m w_m\exp(-V_m/0.7)
  ]

  Use:

  * (\varepsilon(\text{ctx}) \in [0.03,0.30])
  * (\varepsilon_{\text{eff}}=\operatorname{clip}(\varepsilon(\text{ctx})+0.10(1-c(I)),0.03,0.35))
  * 12 bisection iterations on-turn, 20 in ponder

  ### Expansion and budgets

  Root action expansion:

  * top 5 legal actions by policy prior
  * plus top Hand-EV discard if not already included
  * plus minimum-risk discard if not already included
  * always include agari/riichi if legal
  * hard cap 7 root actions

  Opponent public events:

  * top 3 on-turn, top 5 in ponder
  * add `"other"` bucket if residual public-event mass (>0.05)

  Chance draws:

  * if wall (> 6): top 8 effective draws + residual bucket
  * if wall (\le 6): exact branching over all live draws with nonzero count

  On-turn defaults:

  * particles 64–128
  * depth 4–6
  * visits 64 base, then +64 for each of:

    * top2_gap < 0.10
    * max_risk > 0.15
    * ESS/P < 0.45
  * clamp visits at 256

  Ponder defaults:

  * particles 256–1024
  * depth 8–12
  * visits 512–2048

  Leaf batching:

  * `MIN_BATCH = 32`
  * `MAX_BATCH = 128`
  * flush early if remaining budget < 20%

  ### Search trust for ExIt

  Do not emit pure search targets unconditionally. Use:

  [
  \lambda_{\text{exit}}=
  \operatorname{clip}\left(\frac{N_{\text{root}}-64}{256-64},0,1\right)
  \cdot
  \operatorname{clip}\left(\frac{m_{\text{expanded}}-0.85}{0.10},0,1\right)
  \cdot
  \exp(-\sigma_{\text{top1}}/0.15)
  \cdot
  \operatorname{clip}\left(\frac{\mathrm{ESS}}{0.6P},0,1\right)
  ]

  [
  \pi^*=\operatorname{normalize}\Big((1-\lambda_{\text{exit}})\pi_{\text{base}}+\lambda_{\text{exit}}\operatorname{softmax}(Q/0.25)\Big)
  ]

  This should be the only ExIt target rule. It is concrete and already aligned with the search-quality concerns in the earlier analysis.

  ### Pseudocode

  ```python
  def run_afbs(state, belief, budget_ms):
      mode = choose_mode(state, belief)
      if mode is None:
          return None

      tree = AfbsTree(root_state=state, belief_sig=belief.signature())
      leaf_batch = []
      deadline = now_ms() + budget_ms

      while now_ms() < deadline:
          path, leaf = tree.select_with_belief(mode, belief)

          if leaf.is_terminal():
              tree.backup_terminal(path, leaf.utility())
              continue

          leaf_batch.append((path, leaf.snapshot()))
          if len(leaf_batch) >= 32 or time_left_ms(deadline) < budget_ms * 0.2:
              encoded = encode_leaf_batch(leaf_batch)
              evals = learner_forward(encoded)
              tree.expand_and_backup_with_belief(leaf_batch, evals, belief)
              leaf_batch.clear()

      return tree.root_result()
  ```

  ### Failure modes

  * **Residual bucket too large**: search is pretending to be exact while dropping too much mass.
    Mitigation: never emit ExIt if expanded mass < 0.85.

  * **Robust backup too pessimistic**: defense over-folds.
    Mitigation: calibrate (\varepsilon(\text{ctx})) by bucket, not globally.

  * **AFBS too shallow**: labels become noise.
    Mitigation: trust gate above, not just visit count.

  ### Evaluation gate

  Promote AFBS only if:

  * mean decision improvement > 0
  * negative fraction < 0.40
  * SaF with AFBS features beats shallow search alone
  * on-turn p95 latency < 150 ms
  * root reuse rate after observed child event is material
  * belief transition calibration improves, not worsens

  ---

  ## 4. Exact recommended solution: endgame trigger boundaries and solver

  ### Trigger boundary

  Use two explicit modes.

  **EndgameLite**

  * wall (\le 10)
  * and any of:

    * riichi present
    * (\max p_{\text{tenpai}} \ge 0.35)
    * oorasu
    * (|\text{score to next place}| \le 12000)

  **EndgameExact**

  * wall (\le 6), or
  * oorasu with (|\text{score gap to next rank}| \le 8000)

  This is the cleanest boundary set in the source material and is strong enough to implement directly.

  ### Solver structure

  Use:

  [
  Q_{\text{end}}(a)=\sum_{p\in \mathcal P_{0.95}} \alpha_p,V(a\mid X^{(p)})
  ]

  where (\mathcal P_{0.95}) is the top-mass particle subset covering 95% posterior mass.

  Inner solver:

  * if `wall <= 6`: exact draw branching
  * if `7 <= wall <= 10`: top-3 draw branching or 0.90 draw-mass
  * opponent nodes: top 2 archetypes by posterior mass, and top 2–3 robust actions within each archetype

  ### Utility

  Use one schedule.

  Normal endgame:
  [
  U(a)=\mathbb E[\Delta pts(a)] + 0.25,\mathbb E[\Delta rank(a)]
  ]

  Tail-risk mode:
  [
  U(a)=\mathbb E[\Delta pts(a)] - \lambda_{4th},\mathrm{CVaR}*{\alpha_t}(L*{4th}(a))
  ]

  with:

  * if oorasu and (P(4th)>0.35): (\alpha_t=0.25), (\lambda_{4th}=1.0)
  * else if oorasu: (\alpha_t=0.10), (\lambda_{4th}=0.5)
  * else: no CVaR term

  This is the strongest simple schedule. It reserves tail complexity for the states where it actually matters.

  ### Runtime caps

  * on-turn particles: top-mass subset capped at 64
  * ponder particles: capped at 256
  * exact node budget: 30k
  * hard wall-clock cap: 100 ms

  ### Failure modes

  * **Trying to exactify opponent behavior too**: branch explosion.
    Mitigation: exactify chance first; keep opponent branch cap hard.

  * **Too many particles late**: duplicated work.
    Mitigation: top-mass subset only.

  * **No exact terminal utility in oorasu**: you miss the biggest endgame edge.
    Mitigation: exact terminal payoffs whenever the hanchan can end in the current kyoku.

  ### Evaluation gate

  Use a dedicated last-10-draw benchmark and require improvement in:

  * placement EV
  * 4th-place avoidance
  * deal-in rate
  * p95 latency

  with ablations:

  1. current weighted particle mean
  2. PIMC rollout
  3. PIMC + exact draw branching
  4. * opponent limitation
  5. * CVaR late mode

  ---

  ## 5. Runtime-safe approximations and label-trust rules

  These are the rules that keep Hydra implementable without giving away too much strength.

  ### Rule 1: do not promote search to teacher unless it clears trust gate

  Use the (\lambda_{\text{exit}}) rule above. Reject any root label if (\lambda_{\text{exit}} < 0.35).

  ### Rule 2: paired-world labels for hard states

  For top-(K) candidate root actions, evaluate on the same hidden worlds and same downstream RNG:

  [
  \hat Q(a)=\frac{1}{M}\sum_{m=1}^{M}R(s,a;\omega_m,\xi_m)
  ]

  with:

  * (K=3)
  * (M=16) on-turn teacher generation
  * (M=64) offline hard-state label generation

  This is the best low-risk variance reduction move once AFBS produces labels. It reduces label noise without forcing a broader search engine.

  ### Rule 3: calibration before courage

  Temperature-scale:

  * tenpai
  * wait-set
  * danger
  * event heads

  by buckets:

  * early / mid / late
  * closed / open / riichi
  * single-threat / multi-threat

  Promotion thresholds:

  * tenpai ECE < 0.02
  * danger ECE < 0.03 overall
  * danger ECE < 0.05 in riichi buckets

  ### Rule 4: keep the first coding tranche narrow

  Before deeper AFBS integration, ensure:

  * advanced targets are present where expected
  * losses are nonzero only when targets are present
  * RL/BC consume them without NaN or silent skipping
  * no new heads, no duplicated belief stack, no broad AFBS rewrite

  ---

  ## Minimal module / data-flow contract

  ```python
  class BeliefState:
      mix_weights: Float[L]
      sib_fields: Float[L, 34, 4]
      particles: ParticleSet
      ess: float
      entropy: float
      n_live: int
      n_dead: int
      aka_meta: Int[3]

  class HandEvOut:
      p_tenpai: Float[A, 3]
      p_win: Float[A, 3]
      e_score: Float[A]
      ukeire1: Float[A]
      v_off: Float[A]

  class SearchRootResult:
      q: Float[A]
      visits: Int[A]
      exit_policy: Float[A]
      risk_upper: Float[A]
      risk_est: Float[A]
      entropy_drop: Float[A]
      robust_tau: Float[A]
      q_var: Float[A]
      ess_after: Float[A]
      root_value: float
  ```

  Decision path:

  1. update belief
  2. compute Hand-EV
  3. apply AFBS gate
  4. if gate false: use network + Hand-EV + calibrated risk
  5. if gate true and wall > 10: AFBS specialist mode
  6. if gate true and wall <= 10: endgame mode
  7. emit `SearchRootResult` only through one path; do not create parallel search outputs

  ---

  ## Evaluation gates

  A coding agent should stop and not promote a module unless it clears these:

  **Belief**

  * hidden-state NLL down
  * wait-set calibration up
  * no collapse / no latency blow-up

  **Hand-EV**

  * offense-only duplicate eval improves
  * p95 CPU within budget
  * tsumo+ron > tsumo-only > current heuristic

  **AFBS**

  * mean decision improvement > 0
  * negative fraction < 0.40
  * on-turn p95 < 150 ms
  * trust-gated ExIt beats no-ExIt

  **Endgame**

  * last-10-draw placement EV up
  * 4th avoidance up
  * exact mode activation rate low but valuable
  * p95 < 100 ms in active endgame mode

  That is the most concrete, lowest-ambiguity version of the remaining Hydra technical stack I would hand to the coding agent now.
  </start-answer>
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
