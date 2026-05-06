<combined_run_record run_id="answer_31" variant_id="runtime-authority-promotion" schema_version="1">
<metadata>
<notes>Self-contained combined record for Agent 31. Keeps compact prompt shell, artifact manifest from authoritative prompt config, plus preserved answer text.</notes>
<layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">
<![CDATA[# Hydra prompt — selective runtime search authority promotion

<role>
Example role placeholder.
Replace with role fitting actual prompt.
Keep short, task-specific.
You evaluate narrow runtime/search challenger for Hydra, not whole architecture.
</role>

<task>
Example task placeholder.
Replace with actual agent job.

Example task block may ask:
- what artifacts directly support
- what is only inference
- what confidence each major conclusion deserves
- what simpler or stronger local alternatives exist in same lane
- what to keep, narrow, remove
- why confident answer parts are justified
- how to implement or validate surviving path with minimal guesswork

Use artifacts below to derive conclusions.
Drive toward strongest exact blueprint for selective runtime search authority promotion.

Need concrete answer showing:
- what live runtime authority seam already exists today
- why current search outputs stay learner-only
- what exact producer, trust, freshness conditions required before runtime-authoritative reuse is safe
- what evaluation harness needed to prove this lane should outrank current DeltaQ-centered path
- whether this lane is truly next-build ready or still later challenger

Do not assume code presence = runtime authority. Derive from artifacts below.
</task>

<rules>
- treat `role` and `task` as task-specific shell sections; customize when prompt needs it
- separate direct artifact support from your inference
- use search/browse aggressively when it strengthens answer: find original paper, adjacent papers, official docs, repos, other primary sources; use abstracts/summaries mainly for discovery, not final evidence
- use bash tool to run Python for light research support when useful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, validation
- do not logic-dump; every important mechanism, threshold, rec must be inferable from evidence or explicit in blueprint so it can be validated and reproduced
- if you claim path works, survives, or is impl-ready, show why confidence is justified and how claim can later be validated or falsified
- inspect own draft before finishing: if confident claim lacks objective visible support, downgrade to inference, proposal, or blocked
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturates, falsifies, or is truly blocked; do not stop because first pass gave plausible answer
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when sounding confident, show confidence basis
- for every important claim, make validation path visible enough for later reviewer testing
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail for validation, reproduction, or falsification (pdfs, sources, links, similar projects, concrete checks)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now. They are not guaranteed correct. Treat as evidence to inspect and critique, not truth to inherit. High chance some are incomplete, misleading, stale, or semantically wrong, so validate all.
</artifact_note>

<artifacts_manifest>

## Artifact 01 — Prompt-packing reminder
Artifact id: `prompt-packing-reminder`
Type: `literal`
Why it matters: Task-specific reminder matching prompt style guide.

## Artifact 02 — Repo status snapshot
Artifact id: `repo-status`
Source label: README
Type: `file_range`
Source: `README.md:70-72`
Why it matters: Current repo truth on shipped lanes vs still selective or staged lanes.

## Artifact 03 — Promoted doctrine: ranked next-step recommendations
Artifact id: `recon-ranked-next`
Source label: RECON
Type: `file_range`
Source: `research/design/HYDRA_RECONCILIATION.md:136-255`
Why it matters: Promoted execution doctrine for what Hydra should build next and what stays reserve/later.

## Artifact 04 — Archive roadmap: phase-next survivors
Artifact id: `archive-phase-next`
Source label: ROADMAP
Type: `file_range`
Source: `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md:108-125`
Why it matters: Derived archive view of strongest surviving later lanes, incl tile-aware routing correction, Hand-EV repair, completed DeltaQ closure context. Treat as archive evidence, not promoted doctrine.

## Artifact 05 — Live ExIt/DeltaQ shared root-search producer
Artifact id: `live-exit-deltaq-producer`
Source label: LIVEX
Type: `file_range`
Source: `crates/hydra-train/src/training/live_exit.rs:320-409`
Why it matters: Shared live self-play producer emitting visit-based ExIt and DeltaQ labels from same AFBS root search.

## Artifact 06 — SaF fast-path consumer for search-derived per-action context
Artifact id: `saf-fastpath-consumer`
Source label: SAF
Type: `file_range`
Source: `crates/hydra-train/src/saf.rs:48-138`
Why it matters: Shows live inference already consumes search-derived delta_q-like Group C context through SaF fast path.

## Artifact 07 — Bridge search features and observation encoding with search context
Artifact id: `bridge-search-deltaq`
Source label: BRIDGE
Type: `file_range`
Source: `crates/hydra-core/src/bridge.rs:341-417`
Why it matters: Shows how DeltaQ-like search features emit into fixed-superset observation and when search context counts as present.

## Artifact 08 — Live code: runtime authority gate and fast path
Artifact id: `runtime-authority-gate`
Source label: INFER
Type: `file_range`
Source: `crates/hydra-train/src/inference.rs:151-196`
Why it matters: Current runtime decision seam: authoritative ponder reuse gated off; else inference uses actor logits plus SaF fast path.

## Artifact 09 — Live code: ponder result provenance and trust gating
Artifact id: `ponder-trust-surface`
Source label: AFBS
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:397-526`
Why it matters: Current runtime/search challenger seam with LearnerOnly default and trust-gated cache semantics.

## Artifact 10 — Inference tests for runtime cache authority and budget behavior
Artifact id: `inference-runtime-tests`
Source label: INFTEST
Type: `file_range`
Source: `crates/hydra-train/src/inference.rs:563-699`
Why it matters: Concrete runtime tests showing authoritative cache reuse, learner-only exclusion, cache invalidation.

## Artifact 11 — Ponder cache generation and trust semantics
Artifact id: `ponder-cache-generation-tests`
Source label: CACHE
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:533-680`
Why it matters: Cache insertion, generation invalidation, trusted lookups, manager helpers for runtime authority lane.

## Artifact 12 — AFBS cache and ponder behavior tests
Artifact id: `afbs-cache-tests`
Source label: AFBSTEST
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:882-1003`
Why it matters: Concrete test surface for cache hits, predictive child caching, ponder priority, from_tree semantics.

## Artifact 13 — AFBS cache/trust tests continued
Artifact id: `afbs-cache-tests-continued`
Source label: AFBST2
Type: `file_range`
Source: `crates/hydra-core/src/afbs.rs:1000-1179`
Why it matters: More dense runtime/cache/trust test surface incl trust ordering, generation invalidation, authoritative filtering.

## Artifact 14 — Train runtime config gates for ExIt and DeltaQ sidecars
Artifact id: `config-runtime-sidecar-gates`
Source label: CFGRT
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/config_runtime.rs:160-183`
Why it matters: Shows explicit runtime-config enforcement that DeltaQ and ExIt replay losses require sidecar-backed labels.

## Artifact 15 — Advanced loss config surface
Artifact id: `advanced-loss-config-surface`
Source label: ADVCFG
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/config.rs:173-182`
Why it matters: Exact config surface for advanced loss knobs that runtime/train validation admits or blocks.

## Artifact 16 — Bootstrap loading of ExIt and DeltaQ sidecars
Artifact id: `bootstrap-sidecar-loading`
Source label: BOOT
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/bootstrap.rs:120-158`
Why it matters: Train bootstrap path loading replay ExIt and DeltaQ sidecars into streaming loader config.

## Artifact 17 — Implementation roadmap live snapshot note
Artifact id: `impl-roadmap-live-snapshot`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:12-24`
Why it matters: Reference-only live snapshot clarifying current crate surfaces and which advanced lanes are already shipped or staged.

## Artifact 18 — Implementation roadmap runtime/search snapshot context
Artifact id: `impl-roadmap-runtime-challenger-context`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:12-18`
Why it matters: Short reference snapshot reinforcing that runtime/eval/inference surfaces already exist in live tree.

## Artifact 19 — Implementation roadmap runtime/search tail context
Artifact id: `impl-roadmap-runtime-tail`
Source label: IMPL
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:19-24`
Why it matters: Extra small reference slice on live runtime/search surfaces to keep prompt floor without filler.

## Artifact 20 — Promoted architecture doctrine: SaF, ExIt, and validation gates
Artifact id: `hydra-final-validation-gates`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:249-365`
Why it matters: North-star validation language for search-as-feature, ExIt, measurable improvement gates.

## Artifact 21 — Promoted architecture doctrine: endgame exactification target
Artifact id: `hydra-final-endgame-risk`
Source label: FINAL
Type: `file_range`
Source: `research/design/HYDRA_FINAL.md:206-223`
Why it matters: Shows gap between current endgame reality and stronger target direction.

## Artifact 22 — Live code: current endgame solver shell
Artifact id: `endgame-live-current`
Source label: ENDGAME
Type: `file_range`
Source: `crates/hydra-core/src/endgame.rs:1-220`
Why it matters: Current endgame impl reality for comparison with promoted doctrine and later challenger lanes.

## Artifact 23 — Live code: current Hand-EV local evaluator logic
Artifact id: `handev-live-current`
Source label: HANDEV
Type: `file_range`
Source: `crates/hydra-core/src/hand_ev.rs:150-309`
Why it matters: Current Hand-EV reality for separating shipped baseline realism from later refinement lanes.

## Artifact 24 — Live code: robust opponent math surface
Artifact id: `robust-opponent-live-current`
Source label: ROBUST
Type: `file_range`
Source: `crates/hydra-core/src/robust_opponent.rs:150-269`
Why it matters: Useful reserve-shelf/runtime challenger surface for context on why broader runtime/search work stays later.

## Artifact 25 — RL runner: DeltaQ activation, logging, and runtime state
Artifact id: `rl-runner-deltaq-state`
Source label: RLRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/rl_runner.rs:55-166`
Why it matters: Operational RL loop integration for DeltaQ activation and logging inside train binary.

## Artifact 26 — Runtime autotune path also activates DeltaQ
Artifact id: `runtime-autotune-deltaq`
Source label: AUTO
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/runtime_autotune.rs:181-234`
Why it matters: Shows DeltaQ activation already wired into runtime probing/autotune, not only main RL runner.

## Artifact 27 — Live code: current evaluation surface
Artifact id: `eval-surface-current`
Source label: EVAL
Type: `file_range`
Source: `crates/hydra-train/src/eval.rs:1-220`
Why it matters: Existing evaluation helpers and benchmark gates. Useful for seeing what DeltaQ decision-quality promotion gate can and cannot reuse.

## Artifact 28 — Train binary validation runner
Artifact id: `validation-runner-current`
Source label: VALRUN
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/validation.rs:1-142`
Why it matters: Current validation pass and scoring surface in train binary; useful for seeing what already exists vs what DeltaQ promotion gate still needs.

## Artifact 29 — Live code: arena summary helpers
Artifact id: `arena-summaries-current`
Source label: ARENA
Type: `file_range`
Source: `crates/hydra-core/src/arena.rs:231-289`
Why it matters: Shows current summary surfaces available for future paired or fixed-compute evaluation logic.

## Artifact 30 — Loss breakdown and optional-head contribution surface
Artifact id: `loss-breakdown-surface`
Source label: LBD
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:520-679`
Why it matters: Shared loss breakdown surface showing how optional advanced targets contribute and stay zero when absent.

</artifacts_manifest>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_31.md">
<![CDATA[**Verdict**

High confidence: Hydra already has real **runtime search authority seam**, but **dormant by design**. Search also already has **live runtime advisory seam**. Authority seam = `Authoritative` ponder-cache short-circuit in inference; advisory seam = active SaF logit-residual path fed by search-derived context. What evidence does *not* show: producer deserving `Authoritative`, or promotion harness strong enough to move this lane ahead of now-closed DeltaQ-centered path. So this is **not next-build ready as new mainline**. It is **next-build ready only as narrow proving experiment**: exact observed-root, discard-only, calibrated authority promotion with hard falsification harness. (Artifacts 02–11, 13–18, 25–30)

### 1) The live runtime authority seam that already exists today

**Direct artifact support, high confidence**

Hydra already has two distinct runtime seams:

1. **Advisory runtime seam (already live):** search-derived per-action context gets encoded into observation, decoded into SaF features, and used by small MLP to shift actor logits before action selection. Real runtime use of search outputs today, but not full action authority. (Artifacts 06, 07, 08)

2. **Authority seam (implemented but inert):** `infer_with_budget()` first tries `lookup_ponder_trusted(info_state_hash, TrustLevel::Authoritative)`. If hit exists, it returns cached `exit_policy` as acting policy; else falls back to actor logits plus SaF. Tests prove manually inserted `Authoritative` cache entry changes runtime action selection, while `LearnerOnly` entry does not, and cache invalidation rejects stale generations. (Artifacts 08, 10, 11, 13)

There is also already **selective routing surface** for which states merit pondering: priority score is function of small top-2 gap, high risk, low particle ESS. That is already right kind of “hard-state” selector for selective authority challenger. (Artifact 09)

So seam is not hypothetical. Precise statement:

* **search already runtime-advisory**
* **cache-based runtime override already exists**
* **missing piece = trustworthy promotion into `Authoritative`**

### 2) Why current search outputs remain learner-only

**Direct artifact support, high confidence**

Four local reasons.

First, code says so explicitly: inference comments say only `Authoritative` cache hits may drive runtime action selection, and **nothing qualifies**, which keeps ponder outputs learner-only. Archive roadmap says same. (Artifacts 08, 04)

Second, default producer semantics say same: `PonderResult::from_tree()` records provenance, but defaults `trust_level` to `LearnerOnly`. Cache layer enforces trust thresholds and generation freshness, but artifact set shows no real producer promoting result past `LearnerOnly`. (Artifacts 09, 11, 13)

Third, only concrete shared search producer shown in evidence is **teacher-side**, not deployment-side: live self-play producer builds root with legal discard children, evaluates each child with model value head, runs **root-only AFBS**, then emits visit-based ExIt labels and DeltaQ labels from that tree. Narrow label-amplification object, not validated runtime solver. (Artifact 05)

Fourth, promoted doctrine still **supervision-first and selective-search-later**. Reconciliation doc ranks truthful target generation and evaluator realism ahead of AFBS runtime promotion, describes `afbs.rs` as shell rather than fully integrated runtime, and explicitly rejects broad search-first or full public-belief search as immediate move. (Artifact 03)

**External evidence, medium-high confidence**

That local read matches literature. In systems where search is truly authoritative, search is not merely labeler: AlphaZero acts by MCTS and then trains network back toward search policy; ExIt explicitly splits planning from generalization similarly. Reanalyse and targeted search-control work strengthen *teacher/data-selection* side of loop by generating improved targets or revisiting better states. That fits Hydra’s shipped DeltaQ/ExIt/sidecar regime far better than unvalidated direct cache override. ([arXiv][1])

Caution also matches literature. Gumbel planning gives policy-improvement guarantee only when action-values are correctly evaluated. Recent MuZero analysis finds learned models struggle to evaluate unseen policies accurately, error grows as evaluated policy moves away from data-collection policy, and policy priors help because they regularize search toward regions where model is more accurate. Exactly pattern where same-net approximate planner is useful as teacher or advisory bias, but not automatically safe as blanket runtime authority. ([OpenReview][2])

Hydra’s domain strengthens that caution. Cleanest search-with-guarantees result in imperfect information is ReBeL, and that is for **two-player zero-sum** public-belief search. Multiplayer imperfect-information systems such as Pluribus can work well, but even there lack of strong general guarantees outside two-player zero-sum is part of story, and recent multi-agent planning papers still frame planning/search integration as intrinsically difficult. Hydra is four-player imperfect-information game, so “code exists” is nowhere near enough to justify runtime authority. ([arXiv][3])

### 3) What exact producer, trust, and freshness conditions would be required before runtime-authoritative reuse is safe

This part is partly **direct support** and partly **proposal**.

#### 3.1 Producer classes

**Direct support**

Artifact set already implies at least three producer classes:

* **`LiveExitRootOnly`**: current label producer in `live_exit.rs`. Root-only, discard-only, same-net child evaluation. This should stay **LearnerOnly**. (Artifact 05)
* **`ObservedRoot` ponder result**: cache entry for exact current observed state. Only class that is plausible future candidate for `Authoritative`. (Artifacts 09, 11)
* **`SpeculativeChildHint`**: predicted-child cache entries already have own namespace. These should not be first-wave authoritative results. (Artifact 11)

**rec**

Start with **exact observed-root, discard-only** authority promotion. Do **not** start with speculative-child authority, and do **not** start with full action-space authority. That narrowing is justified because strongest live search-derived surfaces in evidence are discard-compatible: live producer is built around legal discards, surviving honest DeltaQ object is discard-compatible, and SaF’s current per-action delta-q path is tile/discard oriented. (Artifacts 04, 05, 06)

#### 3.2 Trust assignment

**Proposal, but strongly grounded**

Do **not** let raw search code set `Authoritative` directly.

Use separate **promotion gate**:

* raw producer writes cache result plus quality/provenance metadata
* promoter classifies into `LearnerOnly`, `Advisory`, `WarmStart`, or `Authoritative`
* inference obeys only `Authoritative`

That separation matters because current evidence proves dangerous fact: runtime seam is strong enough that *any* cache entry marked `Authoritative` can steer action selection, while current cache semantics shown in evidence enforce only generation freshness and minimum trust. That is why promotion cannot be “set enum.”

Minimal acceptance rule should look like:

```rust
fn accept_authoritative(r: &PonderResult, ctx: &RuntimeCtx) -> bool {
    r.cache_namespace == CacheNamespace::ObservedRoot &&
    r.generation == ctx.current_generation &&
    r.source_net_hash == ctx.live_net_hash &&
    r.source_version == ctx.live_version &&
    r.info_state_hash == ctx.info_state_hash &&
    r.legal_mask_hash == hash_legal(ctx.legal_mask) &&
    r.obs_schema_hash == OBS_SCHEMA_HASH &&
    r.search_algo_version == AFBS_VERSION &&
    r.posterior_hash == ctx.posterior_hash &&
    r.visit_count >= calibrated_min_visits(ctx.bucket) &&
    r.search_depth >= calibrated_min_depth(ctx.bucket) &&
    r.repeat_jsd <= calibrated_max_jsd(ctx.bucket) &&
    r.expected_gain_lcb95 > 0.0
}
```

**What is directly missing today**

Artifact set shows:

* **enforced now:** trust threshold, generation freshness, info-state lookup key, legality masking *after* lookup (Artifacts 08, 11, 13)
* **recorded but not shown enforced at lookup:** `source_net_hash`, `source_version`, `timestamp`, `cache_namespace` (Artifacts 09, 11)
* **not shown anywhere:** `legal_mask_hash`, `obs_schema_hash`, `search_algo_version`, `posterior_hash`, repeated-search stability, calibrated expected-gain score

That is cleanest statement of missing proof objects.

Key blocked surface: `infer_with_budget()` hashes only `obs` and passes `legal` separately. Artifact set does **not** prove legality is part of cache identity. Until that proof exists, authoritativeness should require explicit legal-mask hash. (Artifact 08)

Another blocked surface: `PonderCache::get_trusted()` filters by generation and trust only. Artifact set does **not** show runtime lookup layer additionally verifying `source_net_hash`, `source_version`, or `cache_namespace`. So either add that check, or proof it already happens elsewhere is missing. (Artifact 11)

#### 3.3 Freshness

**High confidence on current reality**

Current freshness is **generation-based**. Good, but too coarse for authority. (Artifacts 09, 11, 13)

**rec**

Freshness for authority should be:

* **semantic first:** exact observed-root identity, legal-mask identity, posterior identity
* **version first:** net hash/version, search algorithm version, encoder/schema version
* **generation second:** checkpoint invalidation
* **timestamp only as audit/TTL**, never primary authority criterion

This also matches practical tree-reuse systems: KataGo reuses search structure, but explicitly discusses stale nodes and update handling rather than treating reuse as inherently trustworthy. ([GitHub][4])

### 4) The evaluation harness needed to prove this lane should outrank the DeltaQ-centered path

**Direct artifact support, high confidence**

Hydra’s current eval surface is not enough. `eval.rs` gives coarse game metrics and latency/throughput gates; validation gives loss and policy agreement; unit tests prove cache semantics. None of that is decision-quality authority-promotion harness. (Artifacts 27, 28, 29, 30)

So missing harness must do four jobs.

#### A. State-level decision-quality bench

Use Hydra’s own validation style as template: **200K stratified-state** bench is already doctrine for testing positive decision improvement. Reuse that scale and make comparator explicit. (Artifact 20)

For each state (s), evaluate:

* (a_0): actor only
* (a_1): current actor + SaF / DeltaQ path
* (a_2): candidate authoritative reuse

Against stronger reference policy/value object (Q_{\text{ref}}(s,a)), compute:

[
\Delta_{\text{DeltaQ}} = \mathbb{E}[Q_{\text{ref}}(s,a_1)-Q_{\text{ref}}(s,a_0)]
]

[
\Delta_{\text{Auth}} = \mathbb{E}[Q_{\text{ref}}(s,a_2)-Q_{\text{ref}}(s,a_1)]
]

Lane should not outrank DeltaQ unless:

* lower 95% confidence bound of (\Delta_{\text{Auth}}) is (>0)
* negative-share stays below Hydra’s own G0-style tolerance
* win is present on slices Hydra routes to pondering: low top-2 gap, high risk, low ESS, late wall

This is right comparison because search is **already** helping runtime through SaF, so challenger must beat shipped advisory path, not straw-man actor-only baseline. (Artifacts 06–08, 20)

Reference objects should be slice-specific:

* **late wall / threat**: strongest local endgame oracle available
* **ordinary hard states**: much higher-budget repeated AFBS / consensus search
* **safety slices**: best available risk/deal-in reference

If that oracle stack does not exist yet, say so. Real blocker, not detail.

#### B. Freshness and trust falsification bench

Current tests already cover:

* authoritative cache reuse works
* learner-only does not influence runtime
* generation invalidation works

Re-run those, then add tests that matter for promotion:

* `authoritative_rejects_speculative_namespace`
* `authoritative_rejects_legal_mask_mismatch`
* `authoritative_rejects_source_version_mismatch`
* `authoritative_rejects_posterior_hash_mismatch`
* `authoritative_rejects_schema_version_mismatch`

Until those exist, “safe authority” is not proven.

#### C. Paired arena under fixed compute

Run paired, seat-balanced matches between:

* current baseline: actor + SaF / DeltaQ
* candidate: exact-root authoritative reuse

under same deployment envelope Hydra already names:

* on-turn budget: 80–150 ms
* call reactions: 20–50 ms
* same idle-ponder hardware budget charged to both arms

Do **not** let authority win by quietly spending more background compute.

Use at least:

* mean placement
* stable dan
* top-2 rate
* 4th rate
* deal-in rate
* hit rate of authoritative reuse
* conditional gain given hit
* net gain (=) hit rate × conditional gain

Useful paired-sample sizing rule:

[
n \approx \left(\frac{1.96 + 0.84}{\delta} \sigma_d \right)^2
]

where (\delta) is mean-placement effect you want to detect and (\sigma_d) is paired SD. If (\sigma_d \in [1.0, 1.2]) and you care about (\delta=0.02), ballpark is **20k–28k paired games**.

#### D. Calibration bench

Authority needs calibrated acceptance model, not hand-wavy thresholds.

At minimum, for every cache candidate log:

* namespace
* generation
* source hash/version
* visits
* depth
* search-policy entropy
* top-1 / top-2 gap
* repeated-search JSD
* route bucket (gap/risk/ESS/wall)

Then fit simple calibrator for “does authority beat current SaF on this bucket?” and set thresholds from held-out lower-confidence-bound performance, not intuition.

### 5) Keep, narrow, remove

**Keep**

* current DeltaQ/ExIt/sidecar path
* current advisory SaF runtime path
* trust/provenance/generation substrate
* selective routing by gap/risk/ESS

(Artifacts 02, 05–08, 09–11, 25–26)

**Narrow**

* first authority experiment to **exact observed-root, discard-only**
* require explicit promotion gate
* require exact freshness/identity checks beyond current generation gate

**Remove or delay**

* speculative-child direct authority
* blanket cache authority for any `Authoritative` enum value
* any claim current root-only label producer is already runtime-authoritative solver
* any reprioritization above DeltaQ-centered path before new harness wins

### 6) Final readiness call

**Mainline readiness:** **No**
**Narrow experiment readiness:** **Yes**

More precisely:

* **High confidence:** authority seam exists.
* **High confidence:** current outputs are intentionally learner-only for runtime override.
* **Medium-high confidence:** current evidence supports keeping DeltaQ/SaF ahead of this lane.
* **Medium confidence:** best surviving next step in this lane is narrow exact-root authority experiment, not broad runtime search promotion.

So strongest exact blueprint is:

1. keep DeltaQ/SaF as incumbent,
2. add real promotion gate,
3. restrict first-wave authority to exact observed-root discard states,
4. add missing provenance/freshness fields,
5. run 200K-state + paired-arena harness,
6. only then ask whether this lane deserves to outrank current path.

Until that proof exists, this is **later challenger**, not next Hydra mainline.

[1]: https://arxiv.org/pdf/1712.01815 "https://arxiv.org/pdf/1712.01815"
[2]: https://openreview.net/pdf?id=bERaNdoegnO "https://openreview.net/pdf?id=bERaNdoegnO"
[3]: https://arxiv.org/abs/2007.13544 "https://arxiv.org/abs/2007.13544"
[4]: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md "https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md"
]]>
</answer_text>
</answer_section>
</combined_run_record>