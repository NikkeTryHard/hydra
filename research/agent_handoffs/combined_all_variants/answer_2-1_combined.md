<combined_run_record run_id="answer_2-1" variant_id="agent_answers_mixed_transcript" schema_version="1">
  <metadata>
    <notes>Mixed transcript-style file from agent_answers containing both prompt and answer bodies.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="agent_answers/ANSWER_2-1.md" extracted_from="mixed_transcript">
  <![CDATA[# Hydra deep-agent follow-up for ANSWER_2-style agent

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
  2. `research/design/HYDRA_RECONCILIATION.md` = the active-path / reserve-shelf / dropped-shelf decision memo
  3. `research/design/IMPLEMENTATION_ROADMAP.md` = step-by-step implementation and gates
4. `research/BUILD_AGENT_PROMPT.md` = historical execution-discipline overlay on all docs (removed later; see `combined_all_variants/README.md` for the current routing chain)
  5. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs and constraints

  ## Current repo reality you must account for

  Hydra already has many advanced modules by name and partial implementation:
  - fixed-superset 192-channel encoder with Group C / D presence masks
  - CT-SMC exact DP sampler
  - Mixture-SIB / Sinkhorn support code
  - AFBS tree scaffolding
  - Hand-EV module
  - endgame module
  - robust opponent math utilities
  - train-side model/head/loss scaffolding

  But the main remaining blocker is still **integration and realism**, not mere file absence.

  ## Resolved decisions you should treat as fixed inputs

  These are no longer open questions for this follow-up:

  - **Unified belief stack:** Mixture-SIB + CT-SMC, no duplicate standalone belief path.
  - **Hand-EV timing:** Hand-EV realism comes before deeper AFBS expansion.
  - **AFBS scope:** selective / specialist / hard-state-gated, not broad mainline search.
  - **Training-core status:** DRDA/ACH is not the mainline foundation; keep as reserve/challenger branch.
  - **Oracle guidance:** privileged oracle teacher only for oracle-critic-like targets; public teacher for belief/search targets; aligned guider/learner setup.

  External evidence already supports these broad patterns:
  - public-belief-style representations are a valid main substrate for learning/search in imperfect-information systems
  - aligned privileged guidance should stay close to the learner's query/target semantics
  - robustness should live in search/solver logic rather than only as shallow post-hoc heuristics

  Do not spend your budget re-litigating those high-level patterns. Focus on Hydra-specific implementation closure.

  ## Highest-priority gaps you must analyze deeply

  You were strongest on **repo-aware loop closure**. Focus on turning the resolved direction above into a concrete implementation blueprint for the current codebase.

  1. **Advanced supervision loop closure**
     - specify the exact data path needed to make advanced targets real end-to-end
     - identify which advanced targets are replay-credible now vs which require teacher search/belief generation
     - define staged activation order for:
       - oracle critic
       - belief fields / mixture weights
       - opponent hand type
       - `ΔQ`
       - safety residual
       - ExIt targets

  2. **Canonical data/target boundaries**
     - define the exact boundary between replay-derived labels, bridge-derived labels, teacher-generated labels, and runtime-only features
     - specify what should flow through `MjaiSample`, batch collation, `HydraTargets`, and any new helper structs
     - make presence/absence semantics explicit for optional advanced targets

  3. **Public-teacher vs privileged-teacher pipeline**
     - define exactly which targets are privileged-only and which must be information-state/public-teacher targets
     - give a concrete teacher-generation workflow that a coding agent could implement in phases

  4. **AFBS loop-closing as an implementation problem**
     - not broad redesign-from-scratch
     - instead: what exact interfaces, caches, labels, and leaf outputs are needed so AFBS becomes useful to training and inference in stages

  5. **Hand-EV / endgame / robust-opponent rollout order**
     - given the current codebase state, what exact tranche ordering would close loops fastest without fake progress?

  ## Additional constraints

  - **Do not copy or derive code from `Mortal-Policy/`** or other AGPL sources.
  - Reference-only is fine; code derivation is not.
  - Maintain Hydra’s Rust conventions, zero-warning policy, and library-code safety rules.
  - Preserve engine performance. Do not casually add hot-path regressions.
  - Respect the reconciled architecture unless the docs clearly require a correction.

  ## What kind of answer is wanted

  Your answer should be optimized for **technical depth and implementation usefulness**, not repo edits. Prefer:
  - formulas where target definitions matter
  - precise dataflow / interface guidance
  - concrete thresholds/hyperparameters over hand-waving
  - pseudocode / compact code snippets where edge cases matter
  - explicit tradeoff analysis
  - ablation/evaluation plans tied to Hydra’s actual architecture

  Avoid spending your budget on:
  - re-litigating already-resolved architecture choices
  - generic motivational advice
  - broad summaries of known Mahjong AI history
  - pretending to have implemented or validated code changes

  Assume that a separate coding agent will use your response as the implementation blueprint.

  ## Required deliverables

  Produce a technical design package for a separate coding agent. Your deliverables should be:

  1. A prioritized analysis of the remaining highest-leverage integration/realism weaknesses in Hydra.
  2. For each major gap, a concrete proposed solution including:
     - exact data/target requirements
     - exact interface boundaries
     - runtime vs training-time behavior
     - staged rollout order
     - evaluation criteria and ablations
  3. Pseudocode or compact code snippets for the hardest integration points.
  4. Recommended interfaces between modules where integration is unclear.
  5. A practical implementation order for a coding agent to follow now.
  6. A concise risk assessment explaining what is most likely to fail, overfit, be too slow, or be too weak.

  ## Success condition

  Your output should make it materially easier for a separate coding agent to implement the next Hydra tranches correctly and efficiently. The goal is to produce the best possible **repo-aware implementation blueprint** for closing the loops that are still only half-alive in code.

  ## Additional context references

  Use these as first-class references in addition to the original hierarchy:
  - `ANSWER_1.md`
  - `ANSWER_2.md`
  - `ANSWER_3.md`
  - `research/design/HYDRA_RECONCILIATION.md`

  Treat `HYDRA_RECONCILIATION.md` as the active-path decision memo and assume the codebase has already been reconciled against the most dangerous doc drift.
  <start-prompt>
  ]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_answers/ANSWER_2-1.md" extracted_from="mixed_transcript">
  <![CDATA[I checked the raw files on `reconcile-hydra-plan` directly. The branch is very consistent on the next move: the best immediate tranche is **not** a broader AFBS push, not Hand-EV realism, and not a belief-teacher program. It is a **narrow advanced-target generation + activation tranche** centered on `sample.rs`, `mjai_loader.rs`, `losses.rs`, `bc.rs`, and `rl.rs`, with only minimal support from existing AFBS/bridge outputs. The memo is explicit that the rollout should stay narrow, prefer `ExIt + ΔQ + safety_residual` first, and only bring belief/mix/hand-type online when labels are actually credible. ([GitHub][1])

  ## Best single next tranche

  **Tranche name:** `Advanced-target closure, narrow edition`

  **Goal:** make three already-existing advanced training surfaces receive real targets and real nonzero losses, with explicit provenance and presence semantics, while leaving every larger search/belief/oracle integration question untouched for the following tranche. ([GitHub][1])

  **The three active targets in this tranche:**

  * `exit_target`
  * `delta_q_target`
  * `safety_residual_target`

  Everything else stays structurally present in the codebase but **inactive** unless a later tranche adds credible labels. That is the smallest step that closes a real loop instead of starting a second half-built loop. ([GitHub][1])

  ## Build-now vs later vs not-this-tranche

  The table below is my branch-aligned classification for each target/surface, using the reconciliation memo’s preferred order and the explicit deferrals around belief supervision, oracle-path detachment, Hand-EV, endgame, and robust-opponent search. ([GitHub][1])

  | Surface                            | Decision                                         | Why                                                                                                                                                             |
  | ---------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | **ExIt targets**                   | **Build now**                                    | Existing AFBS shell already has root exit policy / visit summaries; memo says upstream `exit_target` production belongs in this tranche.                        |
  | **ΔQ**                             | **Build now**                                    | Existing AFBS shell already has Q summaries; memo explicitly prefers `delta_q_target` early.                                                                    |
  | **Safety residual**                | **Build now**                                    | Memo explicitly prefers it early; can be defined from replay-credible exact immediate danger plus existing upper-bound signal without a new search project.     |
  | **Oracle critic**                  | **Not part of this tranche**                     | The model’s oracle path/detachment issue is explicitly deferred; pulling oracle in now widens this from data-plumbing into privileged-path representation work. |
  | **Belief fields**                  | **Build only after teacher/search infra exists** | These must be public-teacher targets, not realized-hidden-state labels.                                                                                         |
  | **Mixture weights**                | **Build only after teacher/search infra exists** | Same reason as belief fields.                                                                                                                                   |
  | **Opponent hand type**             | **Build only after teacher/search infra exists** | Needs posterior-soft teacher labels, not realized final-hand labels.                                                                                            |
  | **Hand-EV labels/features**        | **Not part of this tranche**                     | Hand-EV realism is the next tranche after supervision closure, not inside this one.                                                                             |
  | **Endgame-driven targets**         | **Not part of this tranche**                     | Explicitly later than supervision closure and Hand-EV realism.                                                                                                  |
  | **Robust-opponent search outputs** | **Not part of this tranche**                     | Later AFBS/runtime backup work; not for the first coding pass.                                                                                                  |

  Two clarifications matter. First, belief-related heads are later not because they are unimportant, but because `ANSWER_2` is right that supervising them from realized hidden state would be conceptually wrong. Second, oracle is not “later because impossible”; it is out of this tranche because the memo explicitly defers oracle-path detachment/alignment review and wants the first pass to stay narrow. ([GitHub][2])

  ## Exact tranche boundary

  ### What enters

  This tranche should add exactly one new training capability: **optional advanced-target plumbing with explicit provenance**, then use it to activate only the three selected targets. ([GitHub][1])

  The targets should be defined as follows.

  For **ExIt**:
  [
  \pi_{\text{exit}}^*(a)=
  \begin{cases}
  \pi_{\text{AFBS-root}}(a) & \text{if root policy already exists} \
  \mathrm{Softmax}(Q(a)/1.0) & \text{otherwise}
  \end{cases}
  ]
  only when:

  * hard-state predicate is true,
  * root visits `>= 64`,
  * and `KL(\pi_exit \,\|\, \pi_base) <= 2.0`. ([GitHub][2])

  For **ΔQ**:
  [
  \Delta Q^*(a)=\mathrm{clip}\left(\frac{Q(a)-\sum_b \pi_{\text{base}}(b)Q(b)}{0.15},-1,1\right)
  ]
  with an **action mask** keeping only actions with meaningful search support. I would use `visits(a) >= 4` as the minimal support threshold. This is a recommendation, not a branch quote; the branch supplies the AFBS-Q path and the hard-state gating, and the visit cutoff is the smallest extra guard that keeps labels from being pure noise. ([GitHub][2])

  For **safety residual**, keep it **replay-derived and privileged** in this tranche rather than inventing a public teacher. Let
  [
  d^*(a)=\mathbf 1{\text{discard } a \text{ would immediately ron into any opponent under the exact replay-hidden state}}
  ]
  using the exact reconstructed tenpai / wait / ron-legal state already available from logs, and let (u_H(a)) be the existing conservative upper-bound danger signal from Hydra’s current stack. Then define
  [
  s^*(a)=\mathrm{clip}(u_H(a)-d^*(a),0,1).
  ]
  This is deliberately narrower than the later probabilistic teacher version. It is still useful because `ANSWER_3` explicitly argues Hydra is leaving dense exact risk labels on the table, and the memo explicitly prioritizes `safety_residual_target` early. ([GitHub][3])

  For hard-state gating, use the existing shape from `ANSWER_2`:
  [
  \mathbf 1[\text{top2 gap}<0.10 ;\lor; \max risk>0.15 ;\lor; ESS/P<0.45 ;\lor; wall\le 12].
  ]
  That keeps AFBS specialist and avoids quietly expanding this into broad search supervision. ([GitHub][2])

  ### What structs/interfaces change

  Keep this minimal. `HydraTargets` already has advanced slots; do not invent new heads or a second sample type. Add one optional carrier at the sample/batch boundary so presence and provenance are explicit. `ANSWER_2`’s `AdvancedTargetBlock` is the right starting point, but populate only the three selected fields in this tranche. ([GitHub][2])

  A minimal shape is:

  ```rust
  pub enum TargetProvenance {
      Replay,
      Search,
  }

  pub struct Tranche1AdvancedTargets {
      pub exit_policy: Option<[f32; 46]>,
      pub exit_action_mask: Option<[bool; 46]>,
      pub delta_q: Option<[f32; 46]>,
      pub delta_q_action_mask: Option<[bool; 46]>,
      pub safety_residual: Option<[f32; 46]>,
      pub safety_action_mask: Option<[bool; 46]>,
      pub presence_mask: u32,
      pub provenance_mask: u32,
  }
  ```

  Then collate those into masked tensors at batch time. The important rule is:
  [
  L_k=
  \begin{cases}
  w_k \cdot \dfrac{\sum m\ell(\hat y,y)}{\max(1,\sum m)} & \text{if any valid targets exist} \
  0 \text{ and logged as absent} & \text{otherwise}
  \end{cases}
  ]
  where (m) is the sample/action mask. The memo is explicit that target presence, not just weight, must control whether a loss exists at all. ([GitHub][1])

  ### What targets are produced in this tranche

  Produced:

  * `exit_target`
  * `delta_q_target`
  * `safety_residual_target`

  Structurally present but deliberately **not produced**:

  * `oracle_v`
  * `belief_fields_target`
  * `mixture_weight_target`
  * `opponent_hand_type_target` ([GitHub][1])

  ### What is explicitly out of scope

  Out of scope means “do not sneak it in.”

  * No new heads.
  * No oracle-path detachment/alignment work.
  * No belief-teacher pipeline.
  * No CT-SMC redesign.
  * No Hand-EV rewrite.
  * No endgame target generation.
  * No robust-opponent search backup integration.
  * No AFBS semantics rewrite; only minimal root-output export if needed. ([GitHub][1])

  ## Minimal file-level implementation plan

  This is the smallest credible file plan that matches the branch.

  **1. `hydra-train/src/training/losses.rs`**
  Make target presence explicit and central. Turn on only:

  * `w_exit = 0.25`
  * `w_delta_q = 0.05`
  * `w_safety_residual = 0.02`

  Keep every other advanced weight at `0.0`. Add per-target counters so logs show:

  * present rows,
  * absent rows,
  * masked action count,
  * nonzero auxiliary loss contributions.
    This is the first file because the memo explicitly wants activation policy in one place. ([GitHub][1])

  **2. `hydra-train/src/data/sample.rs`**
  Add the narrow advanced-target carrier to `MjaiSample` and `MjaiBatch`. Do not generalize beyond the three active targets yet. Preserve augmentation correctness for action-indexed tensors. The memo explicitly warns against mixing search-only targets into baseline batches without explicit provenance. ([GitHub][1])

  **3. `hydra-train/src/data/mjai_loader.rs`**
  Add one builder path for:

  * search-derived `exit_target`,
  * search-derived `delta_q_target`,
  * replay-derived `safety_residual_target`.

  Leave all unavailable advanced targets as absent. Do not fabricate belief/mix/hand-type labels. This is the real core of the tranche. ([GitHub][1])

  **4. `hydra-train/src/training/rl.rs`**
  Make real upstream `exit_target` consumption part of the tranche. Add mixed-batch tests for:

  * baseline only,
  * baseline + exit,
  * baseline + exit + delta_q + safety_residual. ([GitHub][1])

  **5. `hydra-train/src/training/bc.rs`**
  Mirror the same mixed-batch coverage for BC. This is where silent shape/presence bugs will show up early. ([GitHub][1])

  **6. `hydra-core/src/afbs.rs`**
  Touch only if current root outputs are not already accessible. Export the minimum needed:

  * root exit policy,
  * root Q summary,
  * root visits.
    No node semantics, no opponent-node rewrite, no pondering changes. ([GitHub][1])

  **7. `hydra-core/src/bridge.rs` only if necessary**
  Only add a narrow accessor if the existing conservative danger upper bound used for `safety_residual_target` is not already reachable in the loader path. Otherwise do not touch it. The memo explicitly says bridge should only promote existing credible signals, not invent new semantics. ([GitHub][1])

  Files that should stay untouched in this tranche unless compilation forces trivial edits:

  * `hydra-train/src/model.rs`
  * `hydra-core/src/ct_smc.rs`
  * `hydra-core/src/hand_ev.rs`
  * `hydra-core/src/endgame.rs`
  * `hydra-core/src/robust_opponent.rs` ([GitHub][1])

  ## Why each deferred item stays deferred

  **Oracle critic:** defer it. The branch explicitly says oracle-path detachment review is later, and the first tranche should not widen into privileged-path / alignment work. ([GitHub][1])

  **Belief fields / mixture weights / opponent hand type:** defer them until a credible public-teacher path exists. `ANSWER_2` is explicit that belief heads should not be trained from realized hidden state. ([GitHub][2])

  **Hand-EV features/labels:** next tranche, not this one. The branch ranks Hand-EV realism immediately after supervision closure, not inside it. ([GitHub][1])

  **Endgame-driven targets and robust-opponent outputs:** later, after supervision and Hand-EV are alive. Both are described as downstream multipliers, not first-tranche work. ([GitHub][1])

  ## Top risks if scope expands

  The biggest risk is turning this narrow tranche into a hidden second tranche. The memo repeatedly warns against exactly that. ([GitHub][1])

  If scope expands, the likely failure modes are:

  1. **Adding belief/mix/hand-type now** turns a plumbing tranche into a teacher-posterior tranche and raises the “wrong labels from realized hidden state” risk immediately. ([GitHub][2])

  2. **Pulling oracle critic in now** turns a data/loss closure pass into an architecture/alignment pass because the detached oracle path is explicitly deferred. ([GitHub][1])

  3. **Touching Hand-EV now** mixes feature realism with supervision closure and makes it much harder to tell whether training improved because labels became real or because bridge features changed. ([GitHub][1])

  4. **Expanding AFBS now** turns a medium-risk tranche into a large runtime/search tranche, which the branch explicitly rejects as the immediate move. ([GitHub][1])

  5. **Trying to activate every dormant head** recreates the repo’s current failure mode: lots of sophisticated surfaces, partially closed loops, and unclear training signal. ([GitHub][2])

  My strongest recommendation is to keep the next step almost boring: **one optional advanced-target carrier, one activation policy, and exactly three live targets: ExIt, ΔQ, safety residual.** That is the smallest tranche that closes a real loop and leaves Hydra in a much better place for the very next move.

  [1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md "raw.githubusercontent.com"
  [2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/ANSWER_2.md "raw.githubusercontent.com"
  [3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/ANSWER_3.md "raw.githubusercontent.com"
  </start-answer>
  ]]>
  </answer_text>
  </answer_section>
</combined_run_record>
