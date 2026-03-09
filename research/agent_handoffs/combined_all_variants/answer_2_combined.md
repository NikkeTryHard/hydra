<combined_run_record run_id="answer_2" variant_id="agent_answers_mixed_transcript" schema_version="1">
  <metadata>
    <notes>Mixed transcript-style file from agent_answers containing both prompt and answer bodies.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="agent_answers/ANSWER_2.md" extracted_from="mixed_transcript">
  <![CDATA[## Hydra deep-agent handoff

  ## Primary working package

  I have attached a zip file to this prompt called `hydra_agent_handoff_source_only.zip`.

  Use that zip as the **primary codebase snapshot** instead of trying to discover or clone the repository first.

  Expected workflow:
  1. Open / extract `hydra_agent_handoff_source_only.zip` and treat the extracted contents as the working project.
  2. Read the included markdown docs from the zip first.
  3. Use the raw GitHub links below only as supplemental reference / cross-check material.
  4. Use the attached PDF package as the primary paper attachment set.

  If you cannot access the attached zip for any reason, fall back to fetching the repository files directly from the **raw GitHub file links** in this document.

  Important:
  - Do **not** rely on normal GitHub browsing/search to reconstruct the repo.
  - Do **not** rely on generic/plain web search to discover the project files.
  - If the zip is unavailable, fetch the raw files directly from the raw GitHub links in this handoff instead.

  You are a deep-thinking **research and design advisor** working on **Hydra**, a Rust-first Riichi Mahjong AI whose goal is to reach or exceed **LuckyJ-level** strength.

  Your job is **not** to browse loosely or brainstorm from scratch, and it is also **not** to directly integrate changes into the repository. Your job is to think very hard about the remaining unsolved high-leverage problems, then produce the strongest possible technical guidance for a separate coding agent to implement.

  Treat the following as the governing hierarchy:

  1. `research/design/HYDRA_FINAL.md` = the architectural SSOT for final strength
  2. `research/design/IMPLEMENTATION_ROADMAP.md` = step-by-step implementation and gates
3. `research/BUILD_AGENT_PROMPT.md` = historical execution-discipline overlay on all docs (removed later; see `combined_all_variants/README.md` for the current routing chain)
  4. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

**Important:** interpret the old `research/BUILD_AGENT_PROMPT.md` reference here as historical context only. For the current routing chain, use `combined_all_variants/README.md` and the live authority docs:
  - follow the roadmap in dependency order
  - do not stub or defer with `todo!()` / `unimplemented!()`
  - do not bypass failing tests
  - do not invent alternate architecture if the spec is explicit
  - validate with `cargo build --release`, `cargo test --release`, and `cargo clippy --all-targets -- -D warnings`
  - use worktrees if your environment supports them
  - commit after meaningful passing gates

  ### Current repo reality you must account for

  Hydra already has many advanced modules by name and partial implementation:
  - fixed-superset 192-channel encoder with Group C / D presence masks
  - CT-SMC exact DP sampler
  - Mixture-SIB / Sinkhorn support code
  - AFBS tree scaffolding
  - Hand-EV module
  - endgame module
  - robust opponent math utilities
  - train-side model/head/loss scaffolding

  But the main remaining blocker is **integration and realism**, not mere file absence.

  ### Highest-priority gaps you must analyze deeply

  1. **Dormant advanced supervision**
     - Advanced heads/targets exist structurally, but the real data/training path appears to populate mostly baseline targets.
     - Ensure real end-to-end target generation and training for:
       - oracle critic
       - belief fields / mixture weights
       - opponent hand type
       - search distillation (`ΔQ`, safety residual)
       - ExIt targets

  2. **AFBS is still mostly scaffold**
     - Convert AFBS from a light PUCT skeleton into a real belief-conditioned search/improvement system.
     - Integrate:
       - real leaf evaluation batching
       - belief-conditioned transitions
       - opponent-node handling
       - pondering/reuse
       - ExIt target generation
       - search-as-feature outputs that are genuinely useful

  3. **Hand-EV oracle is too heuristic**
     - Current Hand-EV is largely shanten/ukeire/cumulative-draw heuristics.
     - Replace/upgrade it toward the spec’s intended per-discard offensive oracle:
       - tenpai/win probabilities
       - expected score
       - belief-weighted counts from CT-SMC
       - simplified ron support at minimum
       - far better late-hand realism

  4. **Endgame exactification is not really there**
     - Current endgame code is mostly weighted particle aggregation.
     - Implement the late-game solver mode the docs imply for small wall counts, and integrate it into decision-time search/runtime.

  5. **Robust opponent modeling is not truly inside search**
     - The KL-ball / archetype math exists, but it does not appear deeply integrated into AFBS opponent nodes/backups.
     - Make robust opponent handling real in search, not just exposed as helper math or shallow features.

  ### Additional constraints

  - **Do not copy or derive code from `Mortal-Policy/`** or other AGPL sources.
  - Reference-only is fine; code derivation is not.
  - Maintain Hydra’s Rust conventions, zero-warning policy, and library-code safety rules.
  - Preserve engine performance. Do not casually add hot-path regressions.
  - Respect existing architecture unless the docs clearly require correction.

  ### What kind of answer is wanted

  Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
  - formulas over vague prose
  - precise algorithms over general suggestions
  - concrete thresholds/hyperparameters over hand-waving
  - pseudocode / compact code snippets where edge cases matter
  - explicit tradeoff analysis
  - ablation/evaluation plans tied to Hydra’s actual architecture

  Avoid spending your budget on:
  - generic motivational advice
  - broad summaries of known Mahjong AI history
  - telling the user to “just run more experiments” without specifying which ones
  - pretending to have implemented or validated code changes

  Assume that a separate coding agent will use your response as the implementation blueprint.

  ### Required deliverables

  Produce a technical design package for a separate coding agent. Do **not** try to directly modify the repository. Your deliverables should be:

  1. A prioritized analysis of the remaining highest-leverage weaknesses in Hydra.
  2. For each major gap, a concrete proposed solution including:
     - the exact algorithmic idea
     - formulas / objective functions / update rules
     - recommended constants, thresholds, and hyperparameters
     - data/target requirements
     - runtime/inference-time behavior
     - training-time behavior
     - evaluation criteria and ablations
  3. Pseudocode or compact code snippets for the hardest pieces, especially where implementation detail matters.
  4. Recommended interfaces between modules where integration is unclear.
  5. A practical implementation order for a coding agent to follow.
  6. A concise risk assessment explaining what is most likely to fail, overfit, be too slow, or be too weak.

  ### Success condition

  Your output should make it materially easier for a separate coding agent to implement the remaining Hydra improvements correctly and efficiently. The goal is to produce the best possible design guidance for making Hydra genuinely stronger, not to directly patch the repo yourself.

  Start by unzipping the attached `hydra_agent_handoff_source_only.zip`, reading the core docs inside it completely, then inspect the current implementation and identify the highest-leverage gaps first. Use the raw GitHub links and external paper links below as supplementary context, not as the primary project source. If the zip is inaccessible, fetch the raw files directly from the raw GitHub links below rather than relying on GitHub search or generic web search.

  <references-are-omitted>

  ## Practical attachment order if the external agent has a hard upload limit
  1. Upload the 20 PDFs from the attached `deep_agent_20_pdfs` package.
  2. Send the high-priority raw GitHub markdown links first.
  3. If more context is needed, send the remaining markdown raw links and any especially relevant external PDF links.

  ## Notes
- historical `research/BUILD_AGENT_PROMPT.md` references should now be interpreted through the current routing chain described in `combined_all_variants/README.md`.
  - Do not use AGPL code as implementation source.
  - Keep `.cargo/config.toml` local-only; it is not part of the handoff.
  </start-prompt>
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_answers/ANSWER_2.md" extracted_from="mixed_transcript">
  <![CDATA[I unpacked the repo snapshot, read the SSOT docs first, then inspected the current source. I also cross-checked the design logic against Suphx, ReBeL/ISMCTS, RVR, PerfectDou, GPO, OLSS, distributional RL, and a couple of robustness papers. One honest limitation: I could not rerun `cargo build/test/clippy` here because this environment does not have a Rust toolchain, so the repo assessment below is from source inspection of the attached snapshot rather than a fresh compile.

  The main conclusion is blunt: Hydra is no longer bottlenecked by missing modules. It is bottlenecked by **five loops that are present on paper but not yet closed in code**.

  1. The **advanced supervision loop is dead**. The model has the extra heads; the loss struct has the extra fields; but the data path still emits almost entirely baseline labels, the advanced loss weights default to zero, and the current oracle-critic path is detached from the backbone.
  2. **AFBS is not yet an information-state search**. It is a top-K PUCT shell with callback leaf values, not a real belief-conditioned search engine.
  3. **Hand-EV is not an offensive oracle**. It is still a shanten/ukeire heuristic with `p_draw`-style proxies.
  4. **Endgame exactification is absent**. The current module is weighted particle averaging, not a small-wall solver.
  5. **Robust opponent math is not inside search**. The KL-ball and archetype code exists, but it is not driving opponent-node backups.

  That makes the right strategy very clear: stop adding more heads and close those five loops in order.

  ## 1. Highest-leverage weakness: advanced supervision is targeting the wrong thing, or nothing

  The single biggest conceptual trap is to supervise the belief heads from the **realized hidden state**. Do not do that. Hydra’s own principle P2 is right: these heads should learn the **information state**, not the hidden state. ReBeL and ISMCTS point in the same direction from different angles: imperfect-information search must operate on public beliefs / information sets, not on determinizations of the hidden state, otherwise you get duplicated compute and strategy-fusion errors. ([arXiv][1])

  Suphx and PerfectDou also make the same broader training point: privileged information is useful during training, but it has to be used in a way that still teaches an imperfect-information policy what it can actually infer at execution time. GPO strengthens that idea with a co-trained privileged guider that stays aligned with the learner instead of becoming too strong to imitate. ([arXiv][2])

  ### Proposed fix: split targets into privileged and public-teacher targets

  Use two teachers, not one.

  **Privileged teacher**
  Used only for `oracle_critic` and optional aligned oracle guidance.

  * Input: public observation plus exact concealed hands and wall.
  * Output: `v_oracle ∈ R^4`, zero-sum normalized.

  Target:
  [
  y^{oracle}_i(s_t)=\mathrm{clip}\left(\frac{\hat V_i^{priv}(s_t)-\frac14\sum_j \hat V_j^{priv}(s_t)}{100000},-1,1\right)
  ]

  Use either:

  * exact terminal score deltas on replay/self-play states, or
  * a privileged bootstrap teacher ( \hat V^{priv} ) trained on self-play rollouts.

  **Public teacher**
  Used for `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`, and `ExIt`.

  * Input: public state only.
  * Engine: teacher CT-SMC + teacher event-likelihood model + teacher AFBS.
  * Output: posterior objects and search objects defined over the public information state.

  This distinction matters. `oracle_critic` can be privileged. `belief_fields`, `ΔQ`, `safety_residual`, and `π_exit` must be public-teacher targets.

  ### Belief targets: posterior clusters, not one-hot hidden allocations

  For each training state (I_t):

  1. Run teacher CT-SMC with a strong event-likelihood model and (P_{teacher}=512) particles.
  2. Cluster weighted particles into (L=4) modes with weighted k-medoids.
  3. Convert each cluster into a component belief target.

  Let particle (p) have weight (\alpha_p) and exact allocation (X^{(p)}*{tz}) for tile (t), zone (z\in{opp1,opp2,opp3,wall}).
  For cluster (C*\ell),

  [
  w_\ell^*=\sum_{p\in C_\ell}\alpha_p
  ]

  [
  B_\ell^*(t,z)=\frac{1}{w_\ell^*}\sum_{p\in C_\ell}\alpha_p\frac{X^{(p)}_{tz}}{r_t}
  ]

  where (r_t) is the public remaining count of tile (t). This makes each row (B_\ell^*(t,\cdot)) a 4-way zone distribution, which is exactly what the belief head should learn.

  Because component order is arbitrary, match predicted and teacher components with Hungarian assignment:

  [
  C_{ij}=\sum_t r_t , \mathrm{KL}!\left(B_i^*(t,\cdot),|,\hat B_j(t,\cdot)\right)
  +\lambda_w\left(\log(w_i^*+\epsilon)-\log(\hat w_j+\epsilon)\right)^2
  ]

  Then
  [
  L_{belief+mix}=\min_{\sigma\in S_L}\sum_{i=1}^L C_{i,\sigma(i)}
  ]

  Recommended constants:

  * `L = 4`
  * `lambda_w = 0.25`
  * `epsilon = 1e-6`
  * cluster distance: weighted L1 over flattened (34\times 4) counts
  * teacher particle count: `512` offline, `256` online

  ### Opponent-hand-type targets: posterior, not hard labels

  Use the archetype set already implied by `OPPONENT_MODELING.md`:

  0 yakuhai-speed
  1 honitsu/chinitsu
  2 toitoi/honroutou
  3 tanyao-speed
  4 sanshoku/ittsuu
  5 chanta/junchan
  6 menzen/riichi
  7 other

  For opponent (o),

  [
  h_{o,k}^*=\sum_p \alpha_p , \mathbf 1[\mathrm{arch}(X_o^{(p)})=k]
  ]

  Train with per-opponent soft CE. This is much better than labeling from the realized winning hand only.

  ### Search distillation targets

  For hard states only, run teacher AFBS and define:

  [
  \delta Q^*(a)=\mathrm{clip}\left(\frac{Q_{search}(a)-\sum_b \pi_{base}(b)Q_{search}(b)}{\tau_{\delta}},-1,1\right)
  ]

  Use `tau_delta = 0.15` in normalized utility units.

  For safety residual, do **not** target exact danger directly. Target the removable slack in the conservative bound:

  [
  s^*(a)=\mathrm{clip}\left(u_{Hunter}(a)-p_{deal,true}(a), 0, 0.35\right)
  ]

  That lets the policy learn where Hunter is overly conservative without giving up the hard safety bound at decision time.

  ExIt target:

  [
  \pi_{exit}(a)=\mathrm{Softmax}\left(Q_{search}(a)/\tau_{exit}\right),\quad \tau_{exit}=1.0
  ]

  Only emit it when:

  * root visits `>= 64`
  * `KL(π_exit || π_base) <= 2.0`
  * hard-state predicate is true

  Those thresholds match the roadmap’s intended shape.

  ### Training behavior

  Turn these on in stages.

  Stage A: belief/oracle only

  * `w_oracle = 0.25`
  * `w_belief = 0.05`
  * `w_mix = 0.02`
  * `w_hand_type = 0.02`

  Stage B: once teacher AFBS is stable

  * `w_delta_q = 0.075`
  * `w_safety_residual = 0.03`
  * `w_exit = 0.5`

  Hard-state predicate:
  [
  \mathbf 1[\text{top2 gap}<0.10 ;\lor; \max risk>0.15 ;\lor; ESS/P<0.45 ;\lor; wall\le 12]
  ]

  Also: remove the current oracle-critic `detach()` or make it scheduled. In the snapshot, the oracle head currently cannot shape the shared representation.

  ### GPO-style alignment for the privileged guider

  Suphx explicitly notes that simple distillation from a strong oracle is hard; GPO gives a better way to keep the guider useful but imitable. Keep Hydra’s oracle-guiding plan, but add an alignment penalty:

  [
  L_{guider}=L_{RL}^{priv} + \lambda_{align} , \mathrm{KL}(\pi_g(\cdot|s^{priv})|\mathrm{sg}[\pi_s(\cdot|o)])
  ]

  [
  L_{student}=L_{base}+ \lambda_{kd},\mathrm{KL}(\mathrm{sg}[\pi_g(\cdot|s^{priv})]|\pi_s(\cdot|o))
  ]

  Adapt `lambda_align` to keep median guider-student KL in `[0.05, 0.20]`. That keeps the oracle from becoming “too good to teach.” ([ar5iv][3])

  ### Evaluation

  Use separate gates:

  * belief marginal NLL / Brier
  * pairwise MI calibration on held-out reveals
  * mixture collapse rate `< 15%`
  * `corr(pred_delta_q, teacher_delta_q) > 0.5` on hard states
  * oracle head adds policy gain, not just lower oracle loss

  ---

  ## 2. AFBS must become a real public-belief search, not a callback-driven PUCT shell

  The core architectural correction is this:

  **Node identity should be `(public_state, belief_state)`**, not a sampled hidden state.

  ReBeL’s lesson is that search in imperfect information should operate over public-belief objects. ISMCTS gives the same warning from the tree-search side: determinization duplicates work and induces strategy fusion. Hydra is 4-player general-sum, so the theorem does not transfer directly, but the architectural lesson absolutely does. ([arXiv][1])

  ### Node key

  Use:

  ```rust
  struct NodeKey {
      public_hash: u64,
      actor: u8,
      phase: PhaseTag,        // SelfDecision, OppDecision, ChanceDraw, Terminal
      belief_sig: u64,        // compressed mixture + ESS + archetype posterior
      wall_bucket: u8,        // 0..10 exact, >10 coarse bucket
  }
  ```

  `belief_sig` should hash:

  * rounded mixture weights
  * normalized entropy
  * normalized ESS
  * top archetype posterior masses
  * a compressed particle-cluster signature

  Do not hash raw particles directly.

  ### Real search object

  Move from node-only stats to edge stats:

  ```rust
  struct ChildStats {
      action: u8,
      prior: f32,
      visits: u32,
      q_sum: f32,
      q_sq_sum: f32,
      risk_sum: f32,
      entropy_drop_sum: f32,
      robust_tau_sum: f32,
  }
  ```

  Root output should be first-class:

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

  This should feed **all three** consumers:

  * ExIt target generation
  * SaF runtime features
  * inference/ponder cache

  ### Search recursion

  At self nodes:
  [
  a^*=\arg\max_a \left(\bar Q(a)+c_{puct}P(a)\frac{\sqrt{N}}{1+n(a)}\right), \quad c_{puct}=2.5
  ]

  At opponent nodes, do not use plain PUCT. Use robust expectation:

  For archetype (m),
  [
  q_{m,\tau}(b)\propto p_m(b)\exp(-Q_m(b)/\tau)
  ]
  with (\tau) chosen so that
  [
  D_{KL}(q_{m,\tau}|p_m)=\varepsilon(\text{ctx})
  ]

  Then
  [
  V_m=\sum_b q_{m,\tau}(b)Q_m(b)
  ]

  Across archetypes:
  [
  V_{opp}=-\tau_{arch}\log \sum_m w_m \exp(-V_m/\tau_{arch})
  ]

  Recommended:

  * `tau_arch = 0.7`
  * `iters_tau = 12` on-turn, `20` in ponder
  * `epsilon(ctx)` clipped to `[0.03, 0.30]`

  Across particles:
  [
  Q(a)=\sum_p \alpha_p V_p(a)
  ]

  ### Belief-conditioned transitions

  For any public event (e):

  Mixture update:
  [
  w'*\ell \propto w*\ell , p_\phi(e \mid I, B_\ell, \ell)
  ]

  Particle update:
  [
  \alpha'_p \propto \alpha_p , L(e \mid X^{(p)}, I)
  ]

  Resample when `ESS < 0.4 P`.

  Until a dedicated opponent event model exists, use a hybrid likelihood:

  [
  \log L(e|X,I)=
  1.0 \log p_{next}(e)
  +0.5 \log p_{tenpai}(e)
  +0.5 \log h_{call}(e|X,I)
  +4.0 \log \mathbf 1[\text{event legal under } X]
  ]

  That is good enough to make CT-SMC and AFBS actually interact before the full learned event model lands.

  ### Leaf batching

  Use batched leaf evaluation for real.

  * `MIN_BATCH = 32`
  * `MAX_BATCH = 128`
  * flush early if remaining budget `< 20%`

  On-turn defaults:

  * particles `64-128`
  * self top-K `5`
  * opponent top-K `3`
  * depth `4-6`
  * visits `64`, then +64 for each active trigger:

    * `top2_gap < 0.10`
    * `max_risk > 0.15`
    * `ESS/P < 0.45`
  * clamp at `256`

  Ponder defaults:

  * particles `256-1024`
  * self top-K `8`
  * opponent top-K `5`
  * depth `8-12`
  * visits `512-2048`

  ### Pseudocode

  ```rust
  fn afbs_search(root_state: &PublicState, belief: &BeliefState, budget_ms: u64) -> SearchRootResult {
      let mut tree = AfbsTree::new(root_state, belief.signature());
      let deadline = now_ms() + budget_ms;
      let mut leaf_batch = Vec::new();

      while now_ms() < deadline {
          let path = tree.select_path_with_belief(root_state, belief);
          let leaf = path.last().unwrap();

          if leaf.is_terminal() {
              tree.backup_terminal(&path, leaf.utility());
              continue;
          }

          leaf_batch.push((path, leaf.snapshot()));
          if leaf_batch.len() >= 32 || time_left_ms(deadline) < budget_ms / 5 {
              let encoded = encode_leaf_batch(&leaf_batch);           // bridge::SearchContext
              let evals = learner.forward(encoded);                   // batched GPU call
              tree.expand_and_backup_with_belief(&leaf_batch, evals, belief);
              leaf_batch.clear();
          }
      }

      tree.root_result()
  }
  ```

  ### Evaluation

  Use the gates Hydra already wants:

  * `mean_decision_improvement > 0`
  * negative fraction `< 0.40`
  * SaF beats shallow search alone
  * on-turn p95 latency `< 150ms`
  * CT-SMC DP `< 1ms`

  And add:

  * belief transition calibration after public events
  * ponder-cache hit value
  * root reuse rate after observed child event

  ---

  ## 3. Hand-EV should be a particle-averaged short-horizon offensive oracle

  Suphx explicitly used 100+ look-ahead features built from the probability and score of winning after discarding a tile and drawing replacements; that is very close to Hydra’s Group D intent. The current snapshot’s Hand-EV is not that yet. ([ar5iv][3])

  ### Core design

  Do **not** compute Hand-EV from one marginal remaining-count vector if you can avoid it. Compute it as a **particle average** over exact hidden allocations:

  [
  \mathrm{HandEV}(I,a)=\sum_p \alpha_p , \mathrm{HandEV}(I, X^{(p)}, a)
  ]

  Use top particles only:

  * `P_hand = 16` on-turn
  * `P_hand = 64` ponder / offline labels

  ### Per-particle short-horizon DP

  For each discard (a), after discard you have a 13-tile hand (h_a) and exact wall multiset (w).

  Define a recursive tuple
  [
  T(h,w,d)=\left(V_{off}, P_{tenpai}, P_{win}, S_{win}\right)
  ]

  with horizon (d\in{1,2,3}).

  Base:

  * if agari: `P_tenpai=1, P_win=1, S_win=score_exact`
  * if `d=0`: `P_tenpai = 1[tenpai]`, `P_win=0`, `S_win=0`

  Recurrence:
  [
  T(h,w,d)=\sum_{u:w_u>0}\frac{w_u}{|w|}
  \begin{cases}
  (\frac{s(u)}{S_{ref}},1,1,s(u)) & \text{if } h+u \text{ is agari}\
  T(h+u-b^*, w-e_u, d-1) & \text{otherwise}
  \end{cases}
  ]

  where
  [
  b^*=\arg\max_b V_{off}(h+u-b, w-e_u, d-1)
  ]

  Use
  [
  V_{off}=P_{win}\cdot \frac{\mathbb E[\text{score}\mid win]}{S_{ref}}+\lambda_{tenpai}P_{tenpai}
  ]
  with:

  * `S_ref = 12000`
  * `lambda_tenpai = 0.05`

  This gives one consistent offensive utility and still exposes the three features Hydra wants.

  ### Simplified ron support

  Minimum viable ron model:

  If state after discard is tenpai with wait set (W), define for opponent (i):

  [
  P_i^{disc \le m}(t)=1-\left(1-p_{next,i}(t)\right)^m
  ]

  with
  [
  m_i(d)=\left\lceil 0.8d + 0.4\cdot open_melds_i + 0.8\cdot \mathbf 1[\text{riichi}_i] \right\rceil
  ]

  Then
  [
  P_{ron}^{(d)}=1-\prod_{i=1}^3\prod_{t\in W}\left(1-P_i^{disc\le m_i(d)}(t)\cdot P_i(wait=t)\right)
  ]

  Use:

  * `p_next_i(t)` from opponent-next-discard head
  * `P_i(wait=t)` from danger/wait-set model if available, else tenpai-head-gated danger proxy

  Combine with tsumo:
  [
  P_{win}^{(d)}=1-\left(1-P_{tsumo}^{(d)}\right)\left(1-P_{ron}^{(d)}\right)
  ]

  For score,
  [
  \mathbb E[\text{score}|win]=
  \frac{P_{tsumo}\mathbb E[s_{tsumo}] + P_{ron}\mathbb E[s_{ron}]}{\max(P_{win},10^{-6})}
  ]

  The exact scoring calls should go through `hydra-engine`, not heuristics.

  ### Runtime behavior

  * Compute once per decision on CPU.
  * Memoize by `(hand_signature, wall_signature, dora, seat, riichi_flags, d)`.
  * Probability-mass prune draw branches:

    * max `12` draw tiles
    * or until cumulative mass `>= 0.97`

  Fallback policy:

  * if CT-SMC unavailable, use public remaining counts
  * if CT-SMC available, use particle average
  * if wall `<= 10`, let endgame solver override Hand-EV

  ### Pseudocode

  ```rust
  fn offensive_stats(hand13: Counts34, wall: Counts34, d: u8, ctx: &RoundCtx) -> Stats {
      if is_agari(&hand13, ctx) {
          let s = exact_score(&hand13, WinMode::Tsumo, ctx);
          return Stats::win_now(s);
      }
      if d == 0 {
          return Stats::from_tenpai(is_tenpai(&hand13));
      }

      let mut acc = Stats::zero();
      for (u, p_u) in top_mass_draws(&wall, 0.97, 12) {
          let hand14 = add_tile(hand13, u);
          if is_agari(&hand14, ctx) {
              let s_tsumo = exact_score(&hand14, WinMode::Tsumo, ctx);
              let ron = simplified_ron_hazard(&hand14, d, ctx);
              acc += p_u * Stats::agari(s_tsumo, ron);
          } else {
              let mut best = None;
              for b in legal_discards(&hand14) {
                  let next = offensive_stats(remove_tile(hand14, b), wall - u, d - 1, ctx);
                  if best.as_ref().is_none_or(|x| next.v_off > x.v_off) {
                      best = Some(next);
                  }
              }
              acc += p_u * best.unwrap();
          }
      }
      acc
  }
  ```

  ### Evaluation

  * top-1 discard agreement vs deep AFBS on offense-only states
  * win conversion from shanten 1 and 0
  * expected score on tenpai-entry states
  * p95 CPU latency
  * ablate:

    1. current heuristic
    2. exact tsumo only
    3. tsumo + simplified ron
    4. marginal counts vs particle-average

  ---

  ## 4. Endgame exactification should be a finite-horizon solver, not weighted averaging

  The current endgame module is only:
  [
  Q(a)=\sum_p \alpha_p , f(X^{(p)},a)
  ]

  That is useful plumbing, but it is not the small-wall solver the docs describe.

  ### Trigger

  Activate exact endgame mode when:

  * `wall <= 10`
  * and at least one:

    * any riichi present
    * `max p_tenpai >= 0.35`
    * `oorasu`
    * `abs(score_to_next_place) <= 12000`

  ### Solver design

  Use a **particle outer loop + exact/sparse inner loop**.

  Outer:
  [
  Q_{end}(a)=\sum_{p\in \mathcal P_{0.95}} \alpha_p V(a \mid X^{(p)})
  ]
  where (\mathcal P_{0.95}) is the top-mass particle set covering 95% posterior mass.

  Inner (V(\cdot|X^{(p)})):

  * exact draw branching when `wall <= 6`
  * top-3 draw branching or 0.90 mass when `7 <= wall <= 10`
  * opponent nodes restricted with OLSS-style strategy limitation

  OLSS is directly relevant here: limiting the opponent strategy space can reduce online search size dramatically while preserving safety in the limit, and the paper reports gains in two-player Mahjong. Hydra is 4-player, so import the principle, not the proof. ([Proceedings of Machine Learning Research][4])

  ### Opponent limitation in late solver

  At opponent nodes:

  * keep top `K_opp = 2` archetypes by posterior mass
  * within each archetype, keep top `K_action = 2 or 3` robust-policy actions

  That gives a tractable branch factor while still respecting opponent uncertainty.

  ### Distributional late-game utility

  This is where Hydra should finally use its score-pdf/cdf and GRP machinery.

  Default leaf utility:
  [
  U(a)=\mathbb E[\Delta pts(a)] + \lambda_{grp},\mathbb E[\Delta rank(a)]
  ]

  Late-game tail-risk mode:
  [
  U(a)=\mathbb E[\Delta pts(a)] - \lambda_{4th},\mathrm{CVaR}*{\alpha_t}(L*{4th}(a))
  ]

  Suggested schedule:

  * if `oorasu` and `P(4th) > 0.35`, use `alpha_t = 0.25`
  * else if `oorasu`, use `alpha_t = 0.10`
  * else `alpha_t = 0`

  This is exactly the kind of place where distributional value pays off, because the important object is not just the mean return. ([arXiv][5])

  ### Cache

  Cache by:

  ```text
  (public_hash, wall_multiset_hash, hand_signature, seat_to_act, horizon, riichi_flags)
  ```

  Recommended caps:

  * on-turn top-mass particles: `<= 64`
  * ponder: `<= 256`
  * exact node budget: `<= 30k`
  * hard time cap: `100ms`

  ### Evaluation

  Use a dedicated last-10-draw benchmark:

  * placement EV
  * 4th-place avoidance
  * deal-in rate
  * p95 latency
  * ablate:

    1. weighted particle mean only
    2. PIMC rollout
    3. PIMC + exact draw branching
    4. * opponent limitation
    5. * CVaR late mode

  ---

  ## 5. Robust opponent modeling must move from helper math into the AFBS backup rule

  The current robust-opponent code is mathematically fine. The problem is placement: it is not yet governing opponent-node values.

  ### Calibration

  Estimate KL-ball size by context bucket:

  [
  \varepsilon(\text{ctx}) = \mathrm{clip}\left(Q_{0.95}\left(D_{KL}(\pi_{real}|\pi_{model}) \mid \text{ctx}\right), 0.03, 0.30\right)
  ]

  Bucket by:

  * stage: early / mid / late
  * openness: closed / open / riichi
  * pressure: neutral / placement-critical

  Then inflate under uncertainty:
  [
  \varepsilon_{eff} = \mathrm{clip}\left(\varepsilon(\text{ctx}) + 0.10\cdot (1-c(I)), 0.03, 0.35\right)
  ]
  where (c(I)\in[0,1]) is a confidence score.

  That makes low-confidence states automatically more pessimistic.

  ### Backup rule

  At opponent node with base prior (p):

  [
  q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)
  ]
  [
  D_{KL}(q_\tau|p)=\varepsilon_{eff}
  ]
  [
  V_{rob}=\sum_a q_\tau(a)Q(a)
  ]

  Then, once league archetype adapters exist, wrap with archetype soft-min:
  [
  V=-\tau_{arch}\log \sum_m w_m \exp(-V_m/\tau_{arch})
  ]

  Use:

  * `tau_arch = 0.7`
  * `iters_tau = 12` on-turn
  * `iters_tau = 20` ponder
  * `num_archetypes = 4`

  ### Where archetypes come from

  Do not block KL-ball integration on full population training. Roll it out in two phases.

  **Phase 1**
  Use KL-ball only.

  **Phase 2**
  Train 4 lightweight archetype adapters from the league:

  * aggressive
  * defensive
  * speed
  * value

  Weight them with the posterior from `opponent_hand_type` and recent public tempo stats.

  This gives the OLSS-style strategy set a real implementation path without inventing an entirely separate search stack.

  ### Evaluation

  * held-out opponent next-discard NLL
  * p95 KL calibration error
  * search decision gain against style-diverse opponents
  * regret of robust vs non-robust root actions on adversarially selected opponent styles

  ---

  ## Interfaces I would standardize immediately

  Do not invent a new integration path; extend the ones already present.

  ```rust
  pub struct AdvancedTargetBlock {
      pub oracle_v: Option<[f32; 4]>;
      pub belief_fields: Option<[[f32; 34]; 16]>;
      pub mixture_weights: Option<[f32; 4]>;
      pub opp_hand_type: Option<[f32; 24]>;
      pub delta_q: Option<[f32; 46]>;
      pub safety_residual: Option<[f32; 46]>;
      pub exit_policy: Option<[f32; 46]>;
      pub availability_mask: u32;
  }

  pub struct SearchRootResult {
      pub q: [f32; 46];
      pub visits: [u32; 46];
      pub exit_policy: [f32; 46];
      pub risk_upper: [f32; 46];
      pub risk_est: [f32; 46];
      pub entropy_drop: [f32; 46];
      pub robust_tau: [f32; 46];
      pub q_var: [f32; 46];
      pub ess_after: [f32; 46];
  }

  pub trait BeliefTeacher {
      fn posterior_targets(&self, public_state: &PublicState) -> AdvancedTargetBlock;
  }

  pub trait SearchEngine {
      fn search(&mut self, public_state: &PublicState, belief: &BeliefState, budget_ms: u64)
          -> SearchRootResult;
  }
  ```

  Three concrete integration rules:

  1. `bridge::SearchContext` should remain the encoder-side entry point. Extend it to accept `SearchRootResult`, not ad hoc per-feature arguments.
  2. `MjaiSample` should gain `AdvancedTargetBlock`; do not create a second disconnected sample type.
  3. SaF should have two input paths:

     * observation-decoded fallback
     * direct `SearchRootResult` path for all 46 actions when live search exists

  That avoids the current “first 34 actions get real features, the rest get zeros” limitation.

  ---

  ## Practical implementation order for the coding agent

  1. **Turn the dead heads on correctly.**
     Add `AdvancedTargetBlock`, populate it in the data pipeline, set nonzero advanced loss weights, remove the oracle-critic `detach()` or gate it by phase.

  2. **Make belief supervision information-state correct.**
     Build the teacher posterior pipeline: CT-SMC teacher, particle clustering, Hungarian-matched belief/mix targets, posterior hand-type targets.

  3. **Replace Hand-EV heuristics with the short-horizon particle oracle.**
     This is the fastest direct strength win and it immediately improves Group D.

  4. **Rewrite AFBS around public-belief nodes.**
     Add real transitions, leaf batching, root result struct, belief updates, and opponent-node robust backup.

  5. **Turn on real ExIt and SaF.**
     Generate `π_exit`, `ΔQ`, `safety_residual`, and root risk stats from AFBS. Feed them into both training and inference.

  6. **Implement the wall<=10 endgame solver.**
     Integrate it as a special leaf evaluator / mode switch inside AFBS.

  7. **Add second-wave robustness.**
     OOD confidence gating, domain-adversarial source/self-play invariance, and late-game CVaR mode.

  That order respects the roadmap’s dependency logic while fixing the actual ceiling blockers first.

  ---

  ## What is most likely to fail

  1. **Belief targets trained to realized hidden state.**
     This is the easiest mistake and the most damaging one. It teaches the wrong object.

  2. **AFBS latency blow-up.**
     The danger is branching on hidden-state details instead of searching public belief and limiting opponent actions.

  3. **Oracle guider becoming inimitable.**
     Suphx already hints at this; GPO-style alignment is the cleanest fix. ([ar5iv][3])

  4. **Hand-EV CPU explosion.**
     Solve with particle caps, memoization, and draw-mass pruning.

  5. **Robust backup becoming over-pessimistic.**
     Context-bucket calibration and confidence-conditioned inflation help; do not use one global epsilon.

  6. **Offline-to-selfplay / platform shift.**
     Domain-adversarial training and explicit confidence estimation are sensible second-wave defenses here. ([arXiv][6])

  The deepest design recommendation is this: **Hydra should stop treating AFBS, CT-SMC, Mixture-SIB, Hand-EV, and the advanced heads as separate modules. They need to become one closed loop**:

  [
  \text{belief teacher} \rightarrow \text{better encoder features} \rightarrow \text{better AFBS}
  \rightarrow \text{better ExIt/}\Delta Q/\text{safety targets} \rightarrow \text{better actor}
  \rightarrow \text{better event likelihoods and search priors}
  ]

  Right now Hydra has most of the pieces for that loop. It does not yet have the loop itself. That is the real remaining blocker.

  [1]: https://arxiv.org/abs/2007.13544 "https://arxiv.org/abs/2007.13544"
  [2]: https://arxiv.org/abs/2003.13590 "https://arxiv.org/abs/2003.13590"
  [3]: https://ar5iv.org/pdf/2003.13590 "https://ar5iv.org/pdf/2003.13590"
  [4]: https://proceedings.mlr.press/v202/liu23k.html "https://proceedings.mlr.press/v202/liu23k.html"
  [5]: https://arxiv.org/abs/1710.10044 "https://arxiv.org/abs/1710.10044"
  [6]: https://arxiv.org/abs/1505.07818 "https://arxiv.org/abs/1505.07818"
  </start-answer>
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
