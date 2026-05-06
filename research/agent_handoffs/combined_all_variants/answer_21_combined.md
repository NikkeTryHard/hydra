Need full file first. Then compress natural language only.<combined_run_record run_id="answer_21" variant_id="reconstructed_prompt_and_answer_pair" schema_version="1">
<metadata>
<notes>Rebuilt combined record for Prompt 21 + inline answer, normalized to shared XML-wrapped combined format. Internal repo cites in answer resolved to exact file paths + line ranges.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="combined.md">
<![CDATA[# Hydra architecture selection prompt — broad artifact-first first-principles search

<role>
Produce impl-ready blueprint for best Hydra architecture.
No memo.
Answer itself = blueprint.
</role>

<direction>
Determine logically best Hydra architecture from first principles.
Need not be SE-ResNet.
Assume impl capability unconstrained: can code anything reasonable.
Do not reject family because current repo/stack shaped different.
Do not ignore real runtime, training, sample-efficiency, deployment, search-integration, compute-budget constraints.
Treat current Hydra plan as one candidate family + one evidence packet, not answer.

Need detailed answer showing:
- irreducible problem constraints of 4-player Riichi Mahjong
- which current Hydra assumptions are hard constraints vs contestable doctrine
- whether actor + learner should share one architecture or differ
- whether best answer is convolutional, attention-based, recurrent, state-space, hybrid, entity-based, set-based, graph-based, or other
- whether best answer should keep fixed tile-tensor path, add event-history path, or replace whole representation
- minimum decisive experiments if evidence still underdetermined
- what to reject, defer, or keep only as reserve-shelf ideas
- how to implement or validate surviving path with minimal guesswork

Use artifacts below to derive conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- no architecture fashion takes
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your inference
- use search/browse to find original paper, then inspect full PDF with skill; use abstracts/summaries only for discovery, not final evidence base
- after grounding in artifacts, explore adjacent fields for competing formulations, keep searching for fragments worth fusing, continue explore -> think hard -> validate loop until strongest fused formulation survives or dies by artifact constraints
- use bash tool to run Python for calculations, math checks, validation when rigor matters
- do not dump logic; every important mechanism, threshold, rec, architecture move should be inferable from evidence or explicit in blueprint so it can be validated + reproduced
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturates or blocks, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem to say now.
They may be wrong.
Treat as evidence to inspect + critique, not truth to inherit.
High chance some incomplete, misleading, stale, semantically wrong, or overconfident by omission, so validate all.
</artifact_note>

<hard_guardrails>
1. Do not assume best Hydra architecture = current one.
2. Do not assume best Hydra architecture = single backbone + many shallow heads; prove or reject.
3. Do not assume best Hydra architecture must preserve current 192x34 tensor unchanged; classify it as hard constraint, soft constraint, convenience, or liability.
4. Do not assume best Hydra architecture must preserve current multi-head split unchanged; evaluate whether some tasks should move into trunk, adapters, auxiliary branches, or teacher-only modules.
5. Do not assume best Hydra architecture must be symmetric between fast actor and slow learner/search-side model.
6. Do not assume search/belief/Hand-EV dynamic features must be injected at input; consider late fusion, cross-attention, sidecar encoders, or separate planning modules.
7. Do not reject transformer, recurrent, state-space, graph, set, hybrid, dual-path architectures because repo conv-centric.
8. Do not accept transformer, recurrent, state-space, graph, set, hybrid, dual-path architectures because they sound modern.
9. Do not let impl convenience beat logical fit.
10. Do not let novelty beat strength-per-effort unless evidence says simpler family capped.
11. If evidence insufficient, say underdetermined and specify smallest decisive experiment matrix instead of faking certainty.
12. If best answer differs for ideal unconstrained architecture vs best architecture under Hydra’s runtime/compute goals, separate answers explicitly.
</hard_guardrails>

<assumption_protocol>
Before comparing families, build assumption ledger with four buckets:
- bucket irreducible game/problem facts
- bucket B: explicit project objectives + runtime constraints
- bucket C: current repo impl realities
- bucket D: contestable doctrine / hypotheses / design bets

Do not blur buckets.
Anything in bucket C or D may be overturned if stronger architecture survives evaluation.
Only bucket and truly binding parts of bucket B count as hard constraints.
</assumption_protocol>

<anti_anchor_protocol>
Use this anti-anchor sequence:
step 1: derive requirements from game, action structure, information structure, objectives BEFORE reading doctrine excerpts as conclusions
step 2: define candidate architecture families in abstract terms
step 3: steelman current Hydra plan as one candidate, not baseline truth
step 4: steelman at least one serious non-conv family and at least one hybrid family
step 5: compare all families under same rubric
step 6: do red-team pass against leading candidate
step 7: only then state rec, if any

Must answer explicitly: “what would have made me incorrectly choose current doctrine by default?”
</anti_anchor_protocol>

<minimum_candidate_family_set>
At minimum, compare all families below unless artifact-grounded reason makes one literally inapplicable:
family 1: pure fixed-tensor residual conv family
family 2: SE-ResNet or related channel-attention conv family
family 3: ConvNeXt-style or modernized conv family over tile axis
family 4: event-sequence transformer family
family 5: tile-token or entity-token transformer/set-transformer family
family 6: recurrent or state-space event-history family
family 7: dual-path hybrid family (fixed tensor trunk + history encoder)
family 8: graph/entity/set family over players, melds, discards, tile groups
family 9: asymmetric actor/learner family where fast actor and slow learner/search-side network differ materially
family 10: any adjacent-field formulation that survives validation and does not collapse into one of above

For each family, say whether it is:
- best overall
- best under ideal-but-still-realistic Hydra objectives
- best under current runtime/search constraints
- only good as subsystem
- only good as teacher-side model
- reserve shelf only
- or reject
</minimum_candidate_family_set>

<evaluation_rubric>
Use weighted rubric with explicit scoring or explicit pairwise dominance logic.
First-order dimensions that must be evaluated:
- representation fit to Mahjong’s public state + partial observability
- ability to exploit tile geometry + local combinatorics
- ability to capture temporal opponent-read patterns + discard/call sequences
- sample efficiency under Hydra-like compute budgets
- fast-path inference latency for actor/runtime use
- slower learner/search-side usefulness under pondering/distillation
- compatibility with multi-head supervision (policy, value, GRP, danger, tenpai, belief, search residuals, etc.)
- robustness when dynamic search/belief features absent or stale
- ease of distilling search/oracle signals into deployable policy
- support for selective search rather than universal expensive search
- calibration potential for safety + belief outputs
- scaling path if Hydra later earns more compute

Tie-breakers that may matter only after first-order comparison:
- impl complexity
- maintenance burden
- stack compatibility
- debugging surface area
- profiling predictability
- licensing or ecosystem friction if relevant

Do not let tie-breaker decide winner if family materially worse on first-order fit.
</evaluation_rubric>

<required_questions>
Must answer all:
Q1. What info patterns dominate strong Mahjong play: local tile-shape reasoning, cross-player relational reasoning, temporal opponent modeling, search-conditioned adaptation, or mixture?
Q2. Which patterns must live in deployable fast actor, and which can be outsourced to teacher/search/pondering/distillation machinery?
Q3. Is best architecture likely single-path or multi-path?
Q4. Should actor and learner share architecture at all, or only share some representation ideas?
Q5. Is current 192x34 tensor core strength, neutral compatibility layer, or anchor holding project back?
Q6. Is opponent-history modeling central enough to require dedicated sequence module?
Q7. Is best architecture likely to preserve explicit safety channels + structured Hand-EV/belief features, or absorb them into different representation?
Q8. Is best architecture likely end-to-end monolithic, or modular with specialized trunks or sidecars?
Q9. What smallest architecture leap has realistic chance to beat current plan?
Q10. What would falsify recommended architecture quickly + cheaply?
</required_questions>

<required_output_shape>
Answer must be blueprint with these practical deliverables:
- assumption ledger
- hard-facts section
- contestable-doctrine section
- candidate-family generation section
- family-by-family evaluation table
- steelman for SE-ResNet
- steelman for strongest non-SE alternative
- comparison of ideal architecture vs best practical Hydra architecture if they differ
- recommended architecture or explicit underdetermined verdict
- decisive experiment matrix
- migration blueprint if rec differs from current doctrine
- revisit triggers that would change decision later

Answer must feel buildable or directly auditable.
</required_output_shape>

<failure_modes_to_ban>
- do not answer with “SE-ResNet is good enough” unless you show why stronger candidates fail
- do not answer with “transformers are better” unless you show why they win under Hydra’s actual objectives
- do not let current codebase shape masquerade as proof of optimality
- do not treat old design docs as ground truth if live runtime docs or code disagree
- do not cite architecture papers abstractly without method details, scope limits, or failure cases
- do not declare exact thresholds or budgets without visible support or explicit proposal status
- do not force single-family winner if real answer is asymmetric actor/learner or hybrid trunk/sidecar
- do not confuse teacher-side architecture freedom with fast actor requirements
- do not confuse better opponent-modeling subsystem with better full-agent backbone
</failure_modes_to_ban>

<architecture_search_notes>
Important: “best architecture for Hydra” may mean one of:
- best full system architecture if rebuilding from scratch
- best deployable fast actor architecture
- best slow learner / teacher architecture
- best search-side value / policy / belief helper architecture
- best migration target from current repo state

You may recommend different architectures for different roles.
You may recommend preserving current actor while changing learner, or vice versa.
You may recommend hybrid where current SE-ResNet remains only one component.
You may recommend rejecting SE-ResNet entirely.
You may recommend keeping SE-ResNet.
You may recommend something more exotic if evidence survives scrutiny.
But every important move must be justified.
</architecture_search_notes>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use paired answer body + repository source files for audit and follow-up.
</artifacts>

<final_reminders>
Job not to defend current doctrine.
Job not to be contrarian for sport either.
Job to determine best Hydra architecture as rigorously as possible.
Need not be SE-ResNet.
Assume we can code anything.
Still respect actual problem constraints, runtime goals, compute realities.
If answer is “keep SE-ResNet but add dedicated history sidecar,” say that.
If answer is “SE-ResNet actor, different learner,” say that.
If answer is “full replacement,” say that.
If answer is “underdetermined; run these experiments,” say that.
Do not stop before reasoning saturates or blocks.
</final_reminders>
]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="combined.md">
<![CDATA[# Hydra architecture blueprint

## 0. Decision

Do **not** make Hydra pure SE-ResNet, and do **not** replace it with pure transformer.

Best Hydra family = **asymmetric multi-path system**:

* **convolutional static-state path** over 34-tile axis for local tile combinatorics + cheap public-structure processing,
* **dedicated public-event history path** for discard/call/riichi order + opponent reading,
* **late-fused optional belief/search inputs** instead of treating dynamic planning outputs as canonical state,
* **tile-aware heads** for tile-indexed outputs + **global heads** for non-tile actions and scalar predictions,
* **materially different learner/search-side model** from fast actor.

Concretely:

* **Fast actor**: keep current 12-block SE-style 1D residual trunk as static path, add **small recurrent history sidecar** (default: GRU), split policy into **discard-tile spatial head + global-action head**.
* **Learner / teacher / search helper**: keep 24-block SE-style static trunk, add **richer event-history transformer sidecar**, and late-fuse belief/search tokens there.
* **Do not replace 192×34 path.** Reclassify it as **static compatibility path**, not whole information state.

This is best practical Hydra architecture under current objectives.
Best ideal learner/search-side architecture, if Hydra later gets more data/compute, is **three-path hybrid**: static conv path + event-history transformer + optional entity/belief/search token path.

Key principle:

> **Use convolution where Mahjong is local and structured; use sequence modeling where Mahjong is ordered and lossy under fixed tensors.**

---

## 1. Assumption ledger

| Bucket                                                   | Contents                                                                                                                                                                                                              |         Hardness | What follows                                                                                                              |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------: | ------------------------------------------------------------------------------------------------------------------------- |
| Irreducible game facts**                            | 4-player general-sum imperfect information; multi-round placement-aware scoring; huge hidden-state uncertainty; interrupted action flow; conserved shared tile pool; strong local tile-shape structure on 34 tile types |         **Hard** | Architecture must handle both **local tile geometry** and **temporal public history**, while respecting hidden-tile coupling |
| **B. Explicit project objectives / runtime constraints** | Maximize stable rank, support fast deployable actor, use selective search not universal heavy search, operate under moderate compute + real inference budgets, preserve deployability in current Rust stack            |  **Mostly hard** | Actor latency matters; learner/search model can be richer; selective search/distillation allowed + desirable             |
| **C. Current repo realities**                            | Live encoder is 192×34 fixed-superset; current trunk is SE-style residual conv; actor/learner are 12/24 blocks; many advanced heads exist; advanced supervision only partly active; current policy path uses pooled features |         **Soft** | Starting constraints for migration, not proof of optimality                                                             |
| **D. Contestable doctrine / design bets**                | “SE-ResNet is answer,” “192×34 is state,” “single backbone + shallow heads is best,” “actor and learner should match,” “search/belief should be input planes”                                                       | **Overturnable** | Must be proved, not inherited                                                                                            |

Bucket comes from game itself + strongest external evidence: Suphx describes Riichi Mahjong as four-player imperfect-information game with multi-round placement-aware rewards, irregular interruptions from meld/win actions, and more than (10^{48}) hidden states per information set on average. Artifacts add other hard fact: hidden tiles sit in conserved shared pool, so beliefs are coupled by exact tile counts, not independently factorized. ([arXiv][1]) (research/design/HYDRA_FINAL.md:47-55)

---

## 2. Hard facts

### 2.1 Static tile geometry is real, frequent, and cheap to exploit

Suphx used 34-column convolutional models, added 100+ look-ahead features as 34-dimensional vectors, and said it avoided pooling because each tile column has semantic meaning. JueJong, though 1v1 not 4p Riichi, still stayed residual/CNN family and encoded latest 24 discards **in order** rather than abandoning structure for pure sequence backbone. ([arXiv][1])

1D (3)-kernel trunk over width (34) does **not** lack global static reach. With one stem conv and two (3\times1) convs per block, receptive field is

[
\text{RF} = 1 + 2(1 + 2B).
]

So:

* (B=12 \Rightarrow \text{RF}=51)
* (B=24 \Rightarrow \text{RF}=99)

Both exceed width (34).
So static conv trunk can already integrate whole-board **static** context. If attention helps Hydra, not because conv trunk “cannot see whole board.” Because missing info is **ordered history** + **optional modality fusion**.

### 2.2 Ordered public history is missing modality

Strongest evidence packet points same way. JueJong devotes 24 feature maps to latest 24 discards in order. Suphx uses recurrent GRU model for game-level reward prediction across rounds. Your own artifact on Mahjong techniques identifies gap as multi-step reasoning over discard/call chains, and flags attention over discard sequences as plausible upgrade over pure CNN treatment. ([arXiv][1]) (research/intel/MAHJONG_TECHNIQUES.md:405-413)

Central architecture fact: **Mahjong’s missing signal in fixed tile tensors is not static board context; it is event order.**

### 2.3 Search matters, but as selective overlay, not actor identity

OLSS’s Mahjong experiments used learned blueprint + environmental model, both based on small residual networks, then used **pUCT** because CFR-style search was too simulation-hungry there; they report meaningful gains at 1000 simulations while CFR at 5000 still inadequate. ReBeL and Student of Games reinforce same system pattern in imperfect-information games broadly: strong learned blueprint + search + self-play + distillation, not “one giant inference-time planner everywhere.” ([Proceedings of Machine Learning Research][2])

So Hydra architecture should support **selective search + distillation**, not make every deployable actor forward pass depend on heavy planner-state inputs.

### 2.4 Token/transformer Mahjong is plausible, but strongest evidence still does not make it default winner

Kanachan is best public steelman for raw-token transformers in Riichi Mahjong: it argues much larger datasets make more expressive models like transformers viable, represents many state aspects as tokens + sequences instead of human-crafted planes, and frames that as conscious trade of feature engineering for data + model scale. ([GitHub][3])

Tjong matters but weak evidence here because only abstract-level access available in this session. Its abstract reports 15M-parameter transformer with hierarchical decision-making, trained on roughly 0.5M data over 7 days, outperforming multiple baselines in its environment. That makes transformer formulations credible; it does **not** outweigh stronger Riichi-specific conv evidence from Suphx plus structured residual evidence from JueJong. ([Directory of Open Access Journals][4])

### 2.5 Repo itself already proves current doctrine is not hard constraint

Live encoder already **192×34** fixed-superset, not old 85×34 monolith, and already carries dynamic feature presence masks. Current code also exposes both `spatial` and `pooled` trunk outputs. But current model feeds pooled features into policy head, while Suphx explicitly avoided pooling for tile-semantic reasons. So repo is **not** proof that “one pooled shared trunk to shallow heads” is best Hydra architecture; it is partial impl with at least one architecturally meaningful simplification still present. (docs/GAME_ENGINE.md:122-125; hydra-core/src/encoder.rs:23-117; hydra-train/src/model.rs:247-258; hydra-train/src/backbone.rs:138-144) ([arXiv][1])

---

## 3. Contestable doctrine

| Doctrine                                                | Verdict                                                          | Reason                                                                                                                                                 |
| ------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **“SE-ResNet is best Hydra architecture.”**         | **Reject as whole-system claim; keep as static trunk candidate** | SE-style conv is strong for static tile path, but does not solve ordered history by itself                                                           |
| **“192×34 is state.”**                              | **Reject**                                                       | Useful **static compatibility layer**, not whole information state                                                                                    |
| **“One backbone plus shallow pooled heads is enough.”** | **Reject**                                                       | Tile-indexed outputs should read spatial tile features; pooling all is wrong default for discard policy                                              |
| **“Actor and learner should share architecture.”**      | **Reject**                                                       | Role constraints differ: actor needs streaming latency + robustness without search; learner can absorb richer history + optional dynamic tokens      |
| **“Belief/search/Hand-EV should all be input planes.”** | **Split**                                                        | Safety + Hand-EV worth explicit early structure; optional belief/search signals should move toward late fusion                                       |
| exact SE flavor is sacred.”**                    | **Reject**                                                       | Even repo differs from older spec: code uses mean-pool SE, while older doctrine described dual-pool CBAM-style SE (research/design/HYDRA_SPEC.md:192-194 vs. hydra-train/src/backbone.rs:31-37) |
| **“More heads are bottleneck.”**                        | **Reject**                                                       | Artifacts already show bigger issue is inactive supervision + lossy representation routing, not missing head count                                   |

---

## 4. Candidate-family generation

Families to compare:

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

Load-bearing external evidence for this comparison is Suphx, JueJong/ACH, OLSS, Kanachan, ConvNeXt. Tjong, graph/set/SSM papers useful but lighter evidence here. ([arXiv][1])

---

## 5. Evaluation rubric

Weights below explicit. Scores coarse; meant to separate families, not fake statistical certainty.

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

Rule for interpretation: family does **not** win on tie-breakers if it loses materially on representation fit, temporal modeling, or sample efficiency.

---

## 6. Family-by-family evaluation

These totals are coarse weighted judgments from rubric above.

| Family                                                |     Approx. score / 100 | Best role                        | Verdict                                                |
| ----------------------------------------------------- | ----------------------: | -------------------------------- | ------------------------------------------------------ |
| 1. Pure fixed-tensor residual conv                    |                  **80** | Fast actor baseline              | Strong reserve baseline; not best overall              |
| 2. SE-ResNet / channel-attention conv                 |                  **82** | Static trunk for actor + learner | **Keep as component**, not whole answer                |
| 3. ConvNeXt-style conv                                |                  **74** | Later trunk challenger           | Reserve shelf only                                     |
| 4. Pure event-sequence transformer                    |                  **63** | Teacher/history module           | Reject as full backbone under current Hydra objectives |
| 5. Pure tile/entity transformer or set-transformer    |                  **72** | Learner-side research challenger | Teacher-side / reserve shelf only for now              |
| 6. Pure recurrent/state-space history family          | **66** as full backbone | Actor-side history subsystem     | Good **subsystem**, reject as full replacement         |
| 7. Dual-path hybrid (static tensor + history encoder) |                  **90** | Main backbone family             | **Best practical backbone family**                     |
| 8. Graph/entity/set backbone                          |                  **70** | Opponent/belief helper           | Subsystem or reserve shelf only                        |
| 9. Asymmetric actor/learner family                    |                  **93** | Whole system                     | **Best full-system answer**                            |
| 10. Search/distillation/belief overlay                |                     n/a | System overlay                   | Necessary overlay, not backbone choice                 |

### Pairwise dominance

* **7 dominates 1/2** because it keeps static conv strengths and adds first-class temporal modeling at modest cost.
* **5 is strongest non-SE challenger**, but loses to 7 on sample efficiency, migration risk, current-budget plausibility.
* **6 beats 4 on actor practicality**, because recurrent sidecar can be streamed incrementally, but loses to 4 in learner-side global event interaction.
* **8 does not beat 7** because graph bias helps relations, but does not remove need for ordered event modeling.
* **9 wins system-level design** because actor and learner have genuinely different jobs.

---

## 7. Steelman for SE-ResNet

If forced to choose **one single-family Hydra architecture** and ban sidecars, best answer still **SE-ResNet over fixed tile tensor**, not transformer.

Why steelman is real:

1. **Matches most universal signal.** Every decision uses local tile-shape reasoning; not every decision needs deep history or search.
2. **Well-supported by strong Mahjong evidence.** Suphx’s strongest public Riichi result used convolutional networks over 34-tile columns and preserved tile-column semantics explicitly. JueJong’s strong 1v1 result also stayed residual/CNN family. ([arXiv][1])
3. **Sample-efficient under moderate compute.** Kanachan’s own argument for transformers is “huge data + large expressive models,” which is not Hydra’s current budget regime. ([GitHub][3])
4. **Already solves static-axis global-context problem.** On width 34, static trunk’s receptive field already global.
5. **SE itself low-risk.** Channel reweighting cheap and current Hydra already has it.

So answer is **not** “SE-ResNet is wrong.”
Answer is: **SE-ResNet is only static half of answer.**

---

## 8. Steelman for the strongest non-SE alternative

Strongest non-SE alternative is **not** ConvNeXt, graph networks, or Mamba as whole-agent backbone.

It is:

> entity-token / event-token transformer (or set-transformer hybrid) that represents tiles, players, melds, discards, dora/meta, and optional belief/search tokens in one unified token space.**

Why this is strongest challenger:

* It can model **cross-player relations** directly.
* It can treat **ordered public history** as first-class.
* It can ingest optional belief/search tokens elegantly.
* It aligns with Kanachan’s explicit claim that raw tokenization plus huge data should let more expressive models beat feature-engineered CNNs. ([GitHub][3])
* Set-transformer style modules fit unordered subsets such as tile multisets or meld collections, and attention-based set models were designed to model interactions while preserving permutation structure. ([Cool Papers][5])
* Graph-network thinking also supports this challenger by emphasizing relational inductive bias over entities + relations. ([Google Research][6])

Why it still loses **for Hydra now**:

1. **Evidence quality weaker.** Strongest public Riichi evidence still conv-centric.
2. **Local tile prior matters.** Pure token model must relearn suit-local combinatorics that conv gets almost free.
3. **Hydra’s budget not scale it” regime.** Kanachan’s own README frames raw-token modeling as data/compute trade. ([GitHub][3])
4. **Migration + debugging risk much higher** in present stack.
5. **Actual missing modality is history**, not static-tile global reach.

So challenger belongs on **learner-side reserve shelf**, not mainline actor path.

---

## 9. Red-team pass against the leading candidate

Leading candidate = **asymmetric hybrid**. Strongest arguments against it are real:

* Maybe current 192×34 tensor already captures enough order through recency planes + tedashi flags.
* Maybe history sidecar improves auxiliary heads but not actual policy strength.
* Maybe added path hurts actor latency more than helps.
* Maybe true missing fix is not new path at all, but routing tile-indexed heads to spatial features instead of pooled ones.
* Maybe pure token model only looks expensive because current benchmarks unfair.

Those objections change rollout order, not family ranking:

1. **Fix tile/global head routing first** so baseline not artificially handicapped.
2. **Validate history with collision benchmark** where identical static tensors map to divergent targets because history differed.
3. **Require order-sensitivity** via history-shuffle ablations.
4. **Keep actor-side history cheap + incremental** unless small transformer proves latency-safe.

### What would have made me incorrectly choose the current doctrine by default?

Three things:

1. Treating repo shape as proof of optimality instead of impl history.
2. Looking at Suphx and seeing “CNNs win,” while missing that Suphx also used **separate action-type models, no pooling on tile semantics, look-ahead features, oracle guiding, and recurrent reward model**. ([arXiv][1])
3. Treating 192×34 as whole information state instead of **static view** of information state.

---

## 10. Direct answers to Q1–Q10

**Q1. What information patterns dominate strong Mahjong play?**
Mixture. **Local tile-shape reasoning** = universal base; **temporal opponent modeling** = highest-leverage missing public signal; **cross-player relational reasoning** matters but usually mediated through public history + score context; **search-conditioned adaptation** matters on hard minority of states, not every state.

**Q2. Which patterns need to live in deployable fast actor?**
Actor must carry **local tile geometry**, **basic public-history opponent reading**, **explicit safety structure**, and **placement-aware value tendencies**. Deep belief/search-conditioned adaptation can stay in learner/search/teacher stack.

**Q3. Is best architecture single-path or multi-path?**
**Multi-path.** Static tiles + ordered public history are different modalities and should not be forced through one representation.

**Q4. Should actor and learner share architecture?**
They should share **representation ideas, event schema, distillation interfaces**, not identical architecture.

**Q5. Is current 192×34 tensor strength or anchor?**
It is **strong compatibility layer** and becomes anchor only if treated as whole state.

**Q6. Is opponent-history modeling central enough to require dedicated sequence module?**
**Yes.** At least learner-side definitely; actor-side probably yes in lightweight recurrent form.

**Q7. Preserve explicit safety and Hand-EV / belief features, or absorb them?**
Preserve **safety** + **Hand-EV** explicitly. Keep **belief/search** structured too, but move them toward **late fusion** rather than mandatory early input.

**Q8. Monolithic or modular?**
**Modular.** Static trunk + history sidecar + optional planning/belief sidecars.

**Q9. Smallest architecture leap with realistic win chance?**
First: **tile-aware spatial/global head split**.
Second: **dedicated history sidecar**, learner-first, actor-next.

**Q10. What falsifies rec quickly + cheaply?**
If hybrid cannot beat conv-only on **same-static-tensor/different-history collision benchmark**, or if **order-shuffling history input barely changes performance**, history path is not earning keep.

---

## 11. Ideal architecture vs. best practical Hydra architecture

| Role                  | Ideal if Hydra later earns more compute/data                                            | Best practical Hydra architecture now                                                            |
| --------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Fast actor**        | Static conv path + tiny event transformer or GRU + spatial/global split heads           | **12-block SE-style static trunk + cached GRU history sidecar + spatial/global split heads**     |
| **Learner / teacher** | Static conv path + event-history transformer + optional entity/belief/search token path | **24-block SE-style static trunk + event-history transformer + late-fused belief/search tokens** |
| **Search helper**     | Learner-style hybrid with richer optional entity/belief modules                         | **Learner-style hybrid first; no new full replacement backbone yet**                             |

Ideal and practical answers differ in **learner richness**, not family-level choice. Even with more compute, I would still keep static conv path.

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

Keep current fixed-shape static path for now:

* input: current **192×34** tensor
* actor trunk: **12-block** 1D pre-activation SE-style residual conv
* learner/search trunk: **24-block** version
* keep GroupNorm-style normalization + current deployment-friendly stack

But reclassify planes:

* **canonical early input**: public state, safety, Hand-EV
* **compatibility/optional input**: search/belief planes with masks

This not because belief/search unimportant. Because optional/stale dynamic features should not define base actor representation. (docs/GAME_ENGINE.md:122-125; hydra-core/src/encoder.rs:23-117; research/design/HYDRA_FINAL.md:245-248)

## 12.3 History path

### Actor history path: recurrent, cached, incremental

Default actor choice:

* **1-layer GRU**, hidden size **128**
* sequence cap **96** public events
* persistent hidden state updated on each public event
* reset at hand boundary

Why GRU on actor:

* event histories naturally streaming,
* hidden state can be cached + updated incrementally,
* cost tiny relative to conv trunk,
* latency predictable.

Rough compute check puts 12-block conv trunk around **165M MACs**, tiny (T=96,d=128) 2-layer transformer sidecar around **42M MACs**, and GRU sidecar around **9–10M MACs**. So actor-side history module affordable either way, but cached recurrence is cleanest default fast path.

### Learner history path: attention over full public history

Default learner choice:

* **3-layer transformer**
* (d_{\text{model}} = 128) or (192)
* 4 heads
* seat embeddings + relative/causal position encoding
* full per-event hidden states retained for cross-attention

Why transformer on learner:

* learner benefits from richer pairwise event interactions,
* sequence lengths short enough that quadratic cost not bottleneck,
* transformer better suited to distilling opponent-read patterns + optional side tokens than single recurrent summary.

Mamba/SSM stays reserve-shelf only: its main advertised advantage is long-sequence throughput, and Hydra’s public histories are not long enough for that to dominate design decision. ([arXiv][7])

## 12.4 Event schema

Use real event vocabulary, not only recency-weighted discard planes.

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

This schema enough to make **order**, **seat**, **call interruptions**, and **riichi timing** first-class.

## 12.5 Fusion

Use **late fusion**, not “concatenate everything at input and hope.”

Let

* (Z \in \mathbb{R}^{34 \times C}): static tile features from conv trunk
* (\bar Z \in \mathbb{R}^{C}): pooled static summary
* (H): history summary or history token matrix
* (D): optional late-fused belief/search summary

Then use:

[
\tilde Z = Z + \text{CrossAttn}(Q=W_Q Z,\ K=H,\ V=H) + g(H,D)\odot Z
]

for learner, and lighter gated affine version for actor:

[
\tilde z_k = z_k + A(H,D) + g(H,D)\odot z_k.
]

Key is **history modulates tile features**, not only final scalar head.

## 12.6 Heads

This most important architecture correction after adding history.

### Tile-indexed heads must read spatial features

Use spatial tile embeddings (\tilde z_k) for:

* discard logits (34 normal + 3 aka)
* danger (3 \times 34)
* opponent next discard (3 \times 34)
* any belief-marginal or search-residual outputs that are tile-indexed

Example:

[
\ell_{\text{discard}}(k) = w^\top \phi([\tilde z_k,\ \text{HandEV}_k,\ \text{Safety}*k,\ h*_{\text{opp}}])
]

Current Hydra code already surfaces `spatial`, but policy path fed pooled features. That should change. (hydra-train/src/model.rs:247-258)

### Global heads should read fused pooled context

Use pooled fused context for:

* riichi / chi / pon / kan / agari / ryuukyoku / pass logits
* value
* score distribution / CDF-PDF
* GRP / placement
* opponent tenpai summary
* mixture weights / meta uncertainty summaries

So policy becomes **factorized 46-action head**:

* **37 discard logits** from tile branch
* **9 global-action logits** from global branch

Keep fixed 46-action interface externally. Change internal head semantics.

### Learner-only or teacher-biased heads

Keep these primarily on learner / teacher unless proved actor-useful:

* `delta_q`
* `safety_residual`
* raw belief-field / mixture / opponent-hand-type auxiliaries

And if belief supervision used, target **projected public-teacher belief objects**, not raw Sinkhorn potentials, matching reconciliation artifact’s caution. (research/design/HYDRA_RECONCILIATION.md:386-441)

## 12.7 Safety, Hand-EV, belief, and search features

### Keep explicit early:

* **safety channels**
* **Hand-EV features**

Reason: cheap, structured, high-value transforms of public state or near-solved single-player subproblems. Suphx explicitly reports look-ahead features as important, and your artifact correctly treats safety encoding as high-ROI domain structure. ([arXiv][1]) (research/intel/MAHJONG_TECHNIQUES.md:441-490)

### Move toward late fusion:

* **belief marginals**
* **search deltas**
* **ESS / entropy / robust stress**
* **search-only risk summaries**

Reason: optional, sometimes stale, selective-search dependent.

Hydra already discovered right idea with presence masks. Keep masks; move fusion later.

## 12.8 Search-side role

Use learner-style hybrid as **search blueprint/value/prior model**.

Do **not** make raw search dependence mandatory actor input.
Do **distill** search residuals + policy improvements into actor.

This aligns with both Hydra artifacts + external imperfect-information search systems. ([Proceedings of Machine Learning Research][2])

---

## 13. Reject, defer, keep

**Reject now**: pure transformer actor replacement; pure graph/entity backbone as mainline; identical actor + learner; pooled-only policy head; universal early-fusion dependence on search/belief features.

**Defer**: ConvNeXt-style trunk rewrite; learner-only entity-token challenger; Mamba/SSM history sidecar; graph-based opponent/belief helper.

**Keep**: SE-style static trunk; explicit safety channels; explicit Hand-EV; selective search; search distillation; structured belief modules outside actor core.

---

## 14. Decisive experiment matrix

These are minimum experiments that decide remaining uncertainty.

| ID     | Question                                                                    | Compare                                                                     | Budget                           | Proposed pass / fail gate                                                                                            |
| ------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **E0** | Is current head routing itself leaving strength on table?                   | pooled policy head vs **spatial discard + global action** split, same trunk | cheapest                         | Pass if discard CE or matched-latency self-play improves; fail kills head-rewrite priority                          |
| **E1** | Does history contain actor-useful information beyond current static tensor? | conv-only vs conv+GRU(actor) vs learner+history                             | cheap offline + short self-play  | Pass only if hybrid wins on **collision benchmark** and temporal-slice danger/opp-next/policy metrics              |
| **E2** | Is actor sidecar better as GRU or tiny transformer?                        | same trunk, same event schema                                               | medium                           | Pick transformer only if it clears actor latency gate with measurable strength gain; otherwise GRU                  |
| **E3** | Should belief/search stay early or move late?                              | early planes vs late tokens vs both                                         | medium                           | Pass if late fusion more robust under absent/stale dynamic features                                                 |
| **E4** | Is full replacement transformer better?                           | matched-param pure entity/tile transformer vs hybrid                        | medium-high, one challenger only | Promote only if it wins offline temporal slices **and** matched-budget self-play **and** latency-adjusted deployment |
| **E5** | Is ConvNeXt-style modernization worth trunk rewrite?                       | matched-param SE trunk vs modernized conv trunk                             | optional                         | Only pursue if hybrid already validated and trunk still looks like bottleneck                                       |

### The two cheapest falsifiers

**Falsifier static-collision benchmark**

* Zero dynamic search/belief channels.
* Hash remaining actor input.
* Collect clusters where same static tensor appears with materially different targets because history differed.
* If history model does **not** win here, it is not solving problem it was added for.

**Falsifier B: order-shuffle ablation**

* Keep event multiset + event identities fixed.
* Randomly shuffle order in last (N) public events.
* If performance barely changes, history module is not using order and should be killed or simplified.

---

## 15. Migration blueprint

Architecture decision does **not** require restarting Hydra from zero. It changes what gets built next.

### Phase 1 — correct the existing trunk/head interface before family expansion

Keep current supervision-first execution order from reconciliation, but make baseline architecturally honest.

Touch points:

* `hydra-train/src/model.rs`
Add `DiscardTileHead`, `GlobalActionHead`, `FusionBlock`, `HistoryEncoder`.
* `hydra-train/src/heads.rs`
Route tile-indexed heads from spatial features.
* `hydra-train/src/backbone.rs`
Keep current SE-style trunk for now.
* `hydra-train/src/training/losses.rs`
Keep optional-head gating explicit.

This phase alone may yield real strength, because current policy pooling likely too lossy for tile-indexed actions. (hydra-train/src/model.rs:247-258) ([arXiv][1])

### Phase 2 — add event-history plumbing, learner first

Touch points:

* `hydra-core/src/bridge.rs`
Emit incremental `EventToken`s from public game events.
* `hydra-train/src/data/sample.rs`
Add padded event sequences + masks to batches.
* `hydra-train/src/data/mjai_loader.rs`
Reconstruct event histories from logs; create temporal benchmark slices.
* `hydra-train/src/model.rs`
Add learner-side transformer history encoder + late fusion.

Do **not** break 192×34 interface yet.

### Phase 3 — distill history-aware learner into lightweight actor

Touch points:

* `hydra-train/src/model.rs`
Actor uses GRU history sidecar.
* `hydra-train/src/inference.rs`
Cache recurrent hidden state across public events.
* `hydra-core/src/game_loop.rs` or equivalent runtime path
Reset hidden state on hand start; update on each public event.

This phase is where actor/learner asymmetry becomes structural rather than only “12 blocks vs 24 blocks.”

### Phase 4 — move optional dynamic features to late fusion

Touch points:

* `hydra-core/src/encoder.rs`
Keep current planes for compatibility, but stop treating them as semantically canonical.
* `hydra-train/src/model.rs`
Add `BeliefSearchAdapter` that turns optional search/belief summaries into late-fused tokens or summaries.
* `losses.rs` / distillation code
Train with feature dropout + presence-gated fusion.

### Phase 5 — reserve challengers only if hybrid validates

Only after hybrid wins E0–E3:

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
| Hybrid fails collision benchmark / order-ablation tests                              | Drop or learner-limit history path                                   |
| Pure entity/token transformer wins matched experiments                               | Re-open conv-trunk assumption                                        |
| Tile-aware head rewrite alone captures most of gain                                  | Delay history-sidecar migration on actor, keep it learner-side first |

---

## 17. Final recommendation

**Best full-system architecture for Hydra:**
**Asymmetric dual-path hybrid.**

* **Keep** SE-style 1D residual trunk over 192×34 static path.
* **Add** dedicated public-history encoder.
* **Use GRU on actor, transformer on learner.**
* **Split policy and other tile-indexed heads into spatial tile heads + global heads.**
* **Keep safety and Hand-EV explicit.**
* **Move belief/search toward late fusion.**
* **Do not force actor and learner to match.**
* **Do not make pure transformer or pure graph backbones mainline now.**

Strongest concise statement:

> **Hydra should stop asking one pooled conv trunk to do three different jobs.**
> Use **conv for static tile structure**, **sequence modeling for public history**, and **late-fused side modules for belief/search**.

[1]: https://arxiv.org/pdf/2003.13590.pdf "https://arxiv.org/pdf/2003.13590.pdf"
[2]: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf "https://proceedings.mlr.press/v202/liu23k/liu23k.pdf"
[3]: https://github.com/Cryolite/kanachan "https://github.com/Cryolite/kanachan"
[4]: https://doaj.org/article/59647e6aed75405f880c94040d2b053a "https://doaj.org/article/59647e6aed75405f880c94040d2b053a"
[5]: https://papers.cool/arxiv/1810.00825 "https://papers.cool/arxiv/1810.00825"
[6]: https://research.google/pubs/relational-inductive-biases-deep-learning-and-graph-networks/ "https://research.google/pubs/relational-inductive-biases-deep-learning-and-graph-networks/"
[7]: https://arxiv.gg/abs/2312.00752 "https://arxiv.gg/abs/2312.00752"
]]>
</answer_text>
</answer_section>
</combined_run_record>