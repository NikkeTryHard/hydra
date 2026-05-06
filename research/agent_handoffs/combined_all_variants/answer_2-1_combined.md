<combined_run_record run_id="answer_2-1" variant_id="agent_answers_mixed_transcript" schema_version="1">
<metadata>
<notes>Mixed transcript file from agent_answers. Contains prompt + answer bodies.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="agent_answers/ANSWER_2-1.md" extracted_from="mixed_transcript">
<![CDATA[# Hydra deep-agent follow-up for ANSWER_2-style agent

  ## Primary working package

Attached zip `hydra_agent_handoff_source_only.zip`.

Use zip as **primary codebase snapshot**. Do not first discover/clone repo.

Expected workflow:
  1. Open / extract `hydra_agent_handoff_source_only.zip`; treat extracted contents as working project.
  2. Read included markdown docs first.
  3. Use raw GitHub links below only as supplement / cross-check.
  4. Use attached PDF package as primary paper set.

If attached zip inaccessible, fall back to fetching repo files directly from **raw GitHub file links** in this document.

Important:
  - Do **not** use normal GitHub browsing/search to reconstruct repo.
  - Do **not** use generic/plain web search to discover project files.
  - If zip unavailable, fetch raw files directly from raw GitHub links in this handoff.

You are deep-thinking **research and design advisor** for **Hydra**, Rust-first Riichi Mahjong AI targeting **LuckyJ-level** strength or higher.

Job is **not** loose browsing or fresh brainstorming. Also **not** direct repo integration. Job = think hard about remaining unsolved high-leverage problems, then produce strongest technical guidance for separate coding agent.

Treat this governing hierarchy:

  1. `research/design/HYDRA_FINAL.md` = architectural SSOT for final strength
  2. `research/design/HYDRA_RECONCILIATION.md` = active-path / reserve-shelf / dropped-shelf decision memo
  3. `research/design/IMPLEMENTATION_ROADMAP.md` = step-by-step impl + gates
4. `research/BUILD_AGENT_PROMPT.md` = historical execution-discipline overlay on all docs (removed later; see `combined_all_variants/README.md` for current routing chain)
  5. `research/design/OPPONENT_MODELING.md`, `research/design/TESTING.md`, `research/infrastructure/INFRASTRUCTURE.md`, `docs/GAME_ENGINE.md`, `research/design/SEEDING.md` = supporting specs + constraints

  ## Current repo reality you must account for

Hydra already has many advanced modules by name + partial impl:
  - fixed-superset 192-channel encoder with Group C / D presence masks
  - CT-SMC exact DP sampler
  - Mixture-SIB / Sinkhorn support code
  - AFBS tree scaffolding
  - Hand-EV module
  - endgame module
  - robust opponent math utilities
  - train-side model/head/loss scaffolding

Main blocker now = **integration + realism**, not file absence.

  ## Resolved decisions you should treat as fixed inputs

These no longer open questions for this follow-up:

  - **Unified belief stack:** Mixture-SIB + CT-SMC, no duplicate standalone belief path.
  - **Hand-EV timing:** Hand-EV realism before deeper AFBS expansion.
  - **AFBS scope:** selective / specialist / hard-state-gated, not broad mainline search.
  - **Training-core status:** DRDA/ACH not mainline foundation; keep as reserve/challenger branch.
  - **Oracle guidance:** privileged oracle teacher only for oracle-critic-like targets; public teacher for belief/search targets; aligned guider/learner setup.

External evidence already supports these broad patterns:
  - public-belief-style representations valid main substrate for learning/search in imperfect-information systems
  - aligned privileged guidance should stay close to learner query/target semantics
  - robustness should live in search/solver logic, not only shallow post-hoc heuristics

Do not spend budget re-litigating those high-level patterns. Focus on Hydra-specific impl closure.

  ## Highest-priority gaps you must analyze deeply

You were strongest on **repo-aware loop closure**. Focus on turning resolved direction above into concrete impl blueprint for current codebase.

  1. **Advanced supervision loop closure**
     - specify exact data path needed to make advanced targets real end-to-end
     - identify which advanced targets are replay-credible now vs which require teacher search/belief generation
     - define staged activation order for:
       - oracle critic
       - belief fields / mixture weights
       - opponent hand type
       - `ΔQ`
       - safety residual
       - ExIt targets

  2. **Canonical data/target boundaries**
     - define exact boundary between replay-derived labels, bridge-derived labels, teacher-generated labels, and runtime-only features
     - specify what should flow through `MjaiSample`, batch collation, `HydraTargets`, and any new helper structs
     - make presence/absence semantics explicit for optional advanced targets

  3. **Public-teacher vs privileged-teacher pipeline**
     - define exactly which targets are privileged-only and which must be information-state/public-teacher targets
     - give concrete teacher-generation workflow coding agent could implement in phases

  4. **AFBS loop-closing as impl problem**
     - not broad redesign-from-scratch
     - instead: exact interfaces, caches, labels, and leaf outputs needed so AFBS becomes useful to training + inference in stages

  5. **Hand-EV / endgame / robust-opponent rollout order**
     - given current codebase state, what exact tranche ordering closes loops fastest without fake progress?

  ## Additional constraints

  - **Do not copy or derive code from `Mortal-Policy/`** or other AGPL sources.
  - Reference-only OK; code derivation not OK.
  - Maintain Hydra Rust conventions, zero-warning policy, library-code safety rules.
  - Preserve engine performance. Do not casually add hot-path regressions.
  - Respect reconciled architecture unless docs clearly require correction.

  ## What kind of answer is wanted

Answer should optimize for **technical depth + impl usefulness**, not repo edits. Prefer:
  - formulas where target definitions matter
  - precise dataflow / interface guidance
  - concrete thresholds/hyperparameters over hand-waving
  - pseudocode / compact code snippets where edge cases matter
  - explicit tradeoff analysis
  - ablation/evaluation plans tied to Hydra actual architecture

Avoid spending budget on:
  - re-litigating already-resolved architecture choices
  - generic motivational advice
  - broad summaries of known Mahjong AI history
  - pretending to have implemented or validated code changes

Assume separate coding agent will use response as impl blueprint.

  ## Required deliverables

Produce technical design package for separate coding agent. Deliverables:

  1. Prioritized analysis of remaining highest-leverage integration/realism weaknesses in Hydra.
  2. For each major gap, concrete proposed solution including:
     - exact data/target requirements
     - exact interface boundaries
     - runtime vs training-time behavior
     - staged rollout order
     - evaluation criteria + ablations
  3. Pseudocode or compact code snippets for hardest integration points.
  4. Recommended interfaces between modules where integration unclear.
  5. Practical impl order for coding agent to follow now.
  6. Concise risk assessment explaining what most likely fails, overfits, runs too slow, or stays too weak.

  ## Success condition

Output should make it materially easier for separate coding agent to implement next Hydra tranches correctly + efficiently. Goal = best possible **repo-aware impl blueprint** for closing loops still half-alive in code.

  ## Additional context references

Use these as first-class references in addition to original hierarchy:
  - `ANSWER_1.md`
  - `ANSWER_2.md`
  - `ANSWER_3.md`
  - `research/design/HYDRA_RECONCILIATION.md`

Treat `HYDRA_RECONCILIATION.md` as active-path decision memo and assume codebase already reconciled against most dangerous doc drift.
<start-prompt>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_answers/ANSWER_2-1.md" extracted_from="mixed_transcript">
<![CDATA[I checked raw files on `reconcile-hydra-plan` directly. Branch consistent on next move: best immediate tranche is **not** broader AFBS push, not Hand-EV realism, not belief-teacher program. It is **narrow advanced-target generation + activation tranche** centered on `sample.rs`, `mjai_loader.rs`, `losses.rs`, `bc.rs`, and `rl.rs`, with only minimal support from existing AFBS/bridge outputs. Memo explicit: rollout stays narrow, prefer `ExIt + ΔQ + safety_residual` first, bring belief/mix/hand-type online only when labels credible. ([GitHub][1])

  ## Best single next tranche

**Tranche name:** `Advanced-target closure, narrow edition`

**Goal:** make three already-existing advanced training surfaces receive real targets + real nonzero losses, with explicit provenance + presence semantics, while leaving every larger search/belief/oracle integration question untouched for following tranche. ([GitHub][1])

three active targets in this tranche:**

  * `exit_target`
  * `delta_q_target`
  * `safety_residual_target`

Everything else stays structurally present in codebase but **inactive** unless later tranche adds credible labels. Smallest step that closes real loop instead of starting second half-built loop. ([GitHub][1])

  ## Build-now vs later vs not-this-tranche

Table below = branch-aligned classification for each target/surface, using reconciliation memo preferred order and explicit deferrals around belief supervision, oracle-path detachment, Hand-EV, endgame, and robust-opponent search. ([GitHub][1])

                                                                                                                                                                                                                                                                | Surface                            | Decision                                         | Why                                                                                                                                                             |
  | ---------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ExIt targets** | **Build now** | Existing AFBS shell already has root exit policy / visit summaries; memo says upstream `exit_target` production belongs in this tranche. |
| **ΔQ** | **Build now** | Existing AFBS shell already has Q summaries; memo explicitly prefers `delta_q_target` early. |
                                                                                                                                                                                                                                                                | **Safety residual**                | **Build now**                                    | Memo explicitly prefers it early; can define from replay-credible exact immediate danger plus existing upper-bound signal without new search project.           |
                                                                                                                                                                                                                                                                | **Oracle critic**                  | **Not part of this tranche**                     | Model oracle path/detachment issue explicitly deferred; pulling oracle now widens this from data-plumbing into privileged-path representation work.            |
                                                                                                                                                                                                                                                                | **Belief fields**                  | **Build only after teacher/search infra exists** | These must be public-teacher targets, not realized-hidden-state labels.                                                                                         |
                                                                                                                                                                                                                                                                | **Mixture weights**                | **Build only after teacher/search infra exists** | Same reason as belief fields.                                                                                                                                   |
                                                                                                                                                                                                                                                                | **Opponent hand type**             | **Build only after teacher/search infra exists** | Needs posterior-soft teacher labels, not realized final-hand labels.                                                                                            |
                                                                                                                                                                                                                                                                | **Hand-EV labels/features**        | **Not part of this tranche**                     | Hand-EV realism is next tranche after supervision closure, not inside this one.                                                                                 |
                                                                                                                                                                                                                                                                | **Endgame-driven targets**         | **Not part of this tranche**                     | Explicitly later than supervision closure and Hand-EV realism.                                                                                                  |
                                                                                                                                                                                                                                                                | **Robust-opponent search outputs** | **Not part of this tranche**                     | Later AFBS/runtime backup work; not for first coding pass.                                                                                                      |

Two clarifications matter. First, belief-related heads are later not because unimportant, but because `ANSWER_2` is right that supervising them from realized hidden state would be conceptually wrong. Second, oracle is not “later because impossible”; it is out of this tranche because memo explicitly defers oracle-path detachment/alignment review and wants first pass to stay narrow. ([GitHub][2])

  ## Exact tranche boundary

  ### What enters

This tranche should add exactly one new training capability: **optional advanced-target plumbing with explicit provenance**, then use it to activate only three selected targets. ([GitHub][1])

Targets should be defined as follows.

For **ExIt**:
[
\pi_{\text{exit}}^*(a)=
\begin{cases}
\pi_{\text{AFBS-root}}(a) & \text{if root policy already exists} \
\mathrm{Softmax}(Q(a)/1.0) & \text{otherwise}
\end{cases}
]
only when:

  * hard-state predicate true,
  * root visits `>= 64`,
  * and `KL(\pi_exit \,\|\, \pi_base) <= 2.0`. ([GitHub][2])

For **ΔQ**:
[
\Delta Q^*(a)=\mathrm{clip}\left(\frac{Q(a)-\sum_b \pi_{\text{base}}(b)Q(b)}{0.15},-1,1\right)
]
with **action mask** keeping only actions with meaningful search support. I would use `visits(a) >= 4` as minimal support threshold. This rec, not branch quote; branch supplies AFBS-Q path + hard-state gating, and visit cutoff is smallest extra guard keeping labels from pure noise. ([GitHub][2])

For **safety residual**, keep it **replay-derived and privileged** in this tranche rather than inventing public teacher. Let
[
d^*(a)=\mathbf 1{\text{discard } \text{ would immediately ron into any opponent under exact replay-hidden state}}
]
using exact reconstructed tenpai / wait / ron-legal state already available from logs, and let (u_H(a)) be existing conservative upper-bound danger signal from Hydra current stack. Then define
[
s^*(a)=\mathrm{clip}(u_H(a)-d^*(a),0,1).
]
This deliberately narrower than later probabilistic teacher version. Still useful because `ANSWER_3` explicitly argues Hydra is leaving dense exact risk labels on table, and memo explicitly prioritizes `safety_residual_target` early. ([GitHub][3])

For hard-state gating, use existing shape from `ANSWER_2`:
[
\mathbf 1[\text{top2 gap}<0.10 ;\lor; \max risk>0.15 ;\lor; ESS/P<0.45 ;\lor; wall\le 12].
]
That keeps AFBS specialist and avoids quietly expanding this into broad search supervision. ([GitHub][2])

  ### What structs/interfaces change

Keep this minimal. `HydraTargets` already has advanced slots; do not invent new heads or second sample type. Add one optional carrier at sample/batch boundary so presence + provenance explicit. `ANSWER_2`’s `AdvancedTargetBlock` is right starting point, but populate only three selected fields in this tranche. ([GitHub][2])

Minimal shape:

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

Then collate into masked tensors at batch time. Important rule:
[
L_k=
\begin{cases}
w_k \cdot \dfrac{\sum m\ell(\hat y,y)}{\max(1,\sum m)} & \text{if any valid targets exist} \
0 \text{ and logged as absent} & \text{otherwise}
\end{cases}
]
where (m) = sample/action mask. Memo explicit that target presence, not only weight, must control whether loss exists at all. ([GitHub][1])

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

This is smallest credible file plan matching branch.

**1. `hydra-train/src/training/losses.rs`**
Make target presence explicit + central. Turn on only:

  * `w_exit = 0.25`
  * `w_delta_q = 0.05`
  * `w_safety_residual = 0.02`

Keep every other advanced weight at `0.0`. Add per-target counters so logs show:

  * present rows,
  * absent rows,
  * masked action count,
  * nonzero auxiliary loss contributions.
This first file because memo explicitly wants activation policy in one place. ([GitHub][1])

**2. `hydra-train/src/data/sample.rs`**
Add narrow advanced-target carrier to `MjaiSample` and `MjaiBatch`. Do not generalize beyond three active targets yet. Preserve augmentation correctness for action-indexed tensors. Memo explicitly warns against mixing search-only targets into baseline batches without explicit provenance. ([GitHub][1])

**3. `hydra-train/src/data/mjai_loader.rs`**
Add one builder path for:

  * search-derived `exit_target`,
  * search-derived `delta_q_target`,
  * replay-derived `safety_residual_target`.

Leave all unavailable advanced targets absent. Do not fabricate belief/mix/hand-type labels. This real core of tranche. ([GitHub][1])

**4. `hydra-train/src/training/rl.rs`**
Make real upstream `exit_target` consumption part of tranche. Add mixed-batch tests for:

  * baseline only,
  * baseline + exit,
  * baseline + exit + delta_q + safety_residual. ([GitHub][1])

**5. `hydra-train/src/training/bc.rs`**
Mirror same mixed-batch coverage for BC. Here silent shape/presence bugs will show early. ([GitHub][1])

**6. `hydra-core/src/afbs.rs`**
Touch only if current root outputs not already accessible. Export minimum needed:

  * root exit policy,
  * root Q summary,
  * root visits.
No node semantics, no opponent-node rewrite, no pondering changes. ([GitHub][1])

**7. `hydra-core/src/bridge.rs` only if necessary**
Only add narrow accessor if existing conservative danger upper bound used for `safety_residual_target` not already reachable in loader path. Otherwise do not touch it. Memo explicitly says bridge should only promote existing credible signals, not invent new semantics. ([GitHub][1])

Files that should stay untouched in this tranche unless compilation forces trivial edits:

  * `hydra-train/src/model.rs`
  * `hydra-core/src/ct_smc.rs`
  * `hydra-core/src/hand_ev.rs`
  * `hydra-core/src/endgame.rs`
  * `hydra-core/src/robust_opponent.rs` ([GitHub][1])

  ## Why each deferred item stays deferred

**Oracle critic:** defer. Branch explicitly says oracle-path detachment review later, and first tranche should not widen into privileged-path / alignment work. ([GitHub][1])

**Belief fields / mixture weights / opponent hand type:** defer until credible public-teacher path exists. `ANSWER_2` explicit that belief heads should not train from realized hidden state. ([GitHub][2])

**Hand-EV features/labels:** next tranche, not this one. Branch ranks Hand-EV realism immediately after supervision closure, not inside it. ([GitHub][1])

**Endgame-driven targets and robust-opponent outputs:** later, after supervision + Hand-EV alive. Both described as downstream multipliers, not first-tranche work. ([GitHub][1])

  ## Top risks if scope expands

Biggest risk = turning this narrow tranche into hidden second tranche. Memo repeatedly warns against exactly this. ([GitHub][1])

If scope expands, likely failure modes:

  1. **Adding belief/mix/hand-type now** turns plumbing tranche into teacher-posterior tranche and raises “wrong labels from realized hidden state” risk immediately. ([GitHub][2])

  2. **Pulling oracle critic in now** turns data/loss closure pass into architecture/alignment pass because detached oracle path explicitly deferred. ([GitHub][1])

  3. **Touching Hand-EV now** mixes feature realism with supervision closure and makes it much harder to tell whether training improved because labels became real or because bridge features changed. ([GitHub][1])

  4. **Expanding AFBS now** turns medium-risk tranche into large runtime/search tranche, which branch explicitly rejects as immediate move. ([GitHub][1])

  5. **Trying to activate every dormant head** recreates repo current failure mode: many sophisticated surfaces, partially closed loops, unclear training signal. ([GitHub][2])

Strongest rec: keep next step almost boring: **one optional advanced-target carrier, one activation policy, and exactly three live targets: ExIt, ΔQ, safety residual.** Smallest tranche that closes real loop and leaves Hydra in much better place for next move.

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/research/design/HYDRA_RECONCILIATION.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/ANSWER_2.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/NikkeTryHard/hydra/reconcile-hydra-plan/ANSWER_3.md "raw.githubusercontent.com"
</start-answer>
]]>
</answer_text>
</answer_section>
</combined_run_record>