<combined_run_record run_id="answer_21" variant_id="reconstructed_prompt_and_answer_pair" schema_version="1">
  <metadata>
    <notes>Reconstructed combined record for Prompt 21 and its inline answer, normalized to the shared XML-wrapped combined format. Internal repo citations in the answer body are resolved to exact file paths and line ranges.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="combined.md">
  <![CDATA[# Hydra architecture selection prompt — broad artifact-first first-principles search

<role>
Produce an implementation-ready blueprint for determining the best architecture for Hydra.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Determine the logically best architecture for Hydra from first principles.
It does not have to be SE-ResNet.
Assume implementation capability is unconstrained: we can code anything reasonable.
Do not reject an architecture family just because the current repository or current stack is shaped differently.
However, do not ignore real runtime, training, sample-efficiency, deployment, search-integration, and compute-budget constraints.
Treat the current Hydra plan as one candidate family and one evidence packet, not as the answer.

We want a detailed answer that makes clear:
- what the irreducible problem constraints of 4-player Riichi Mahjong actually are
- which current Hydra assumptions are hard constraints versus contestable doctrine
- whether a single architecture should be used for actor and learner, or whether they should differ
- whether the best answer is convolutional, attention-based, recurrent, state-space, hybrid, entity-based, set-based, graph-based, or something else
- whether the best answer should keep a fixed tile-tensor path, add an event-history path, or replace the whole representation
- what the minimum decisive experiments are if the evidence is still underdetermined
- what should be rejected, deferred, or kept only as reserve-shelf ideas
- how to implement or validate the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- no architecture fashion takes
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- after grounding in the artifacts, explore many adjacent fields for competing formulations of the same problem, keep searching for interesting fragments worth fusing together, and continue the explore -> think hard -> validate loop until the strongest fused formulation either survives or is killed by the artifact constraints
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not dump logic; every important mechanism, threshold, recommendation, and architecture move should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now.
They are not guaranteed to be fully correct.
Treat them as evidence to inspect and critique, not truth to inherit.
High chance some of them are incomplete, misleading, stale, semantically wrong, or overconfident by omission, so validate everything.
</artifact_note>

<hard_guardrails>
1. Do not assume the best Hydra architecture is the current one.
2. Do not assume the best Hydra architecture is a single backbone plus many shallow heads; prove it or reject it.
3. Do not assume the best Hydra architecture must preserve the current 192x34 tensor representation unchanged; classify that representation as hard constraint, soft constraint, convenience, or liability.
4. Do not assume the best Hydra architecture must preserve the current multi-head split unchanged; evaluate whether some tasks should move into trunk, adapters, auxiliary branches, or teacher-only modules.
5. Do not assume the best Hydra architecture must be symmetric between fast actor and slow learner/search-side model.
6. Do not assume search/belief/Hand-EV dynamic features must be injected at the input; consider late fusion, cross-attention, sidecar encoders, or separate planning modules.
7. Do not reject transformer, recurrent, state-space, graph, set, hybrid, or dual-path architectures just because the repo is currently conv-centric.
8. Do not accept transformer, recurrent, state-space, graph, set, hybrid, or dual-path architectures just because they sound more modern.
9. Do not let implementation convenience beat logical fit.
10. Do not let novelty beat strength-per-effort unless the evidence says the simpler family is capped.
11. If the evidence is insufficient, say underdetermined and specify the smallest decisive experiment matrix instead of faking certainty.
12. If the best answer differs for ideal unconstrained architecture versus best architecture under Hydra’s stated runtime/compute goals, explicitly separate those answers.
</hard_guardrails>

<assumption_protocol>
Before comparing architecture families, build an assumption ledger with four buckets:
- bucket A: irreducible game/problem facts
- bucket B: explicit project objectives and runtime constraints
- bucket C: current repo implementation realities
- bucket D: contestable doctrine / hypotheses / design bets

You must not blur these buckets together.
Anything in bucket C or D may be overturned if a stronger architecture survives the evaluation.
Only bucket A and truly binding parts of bucket B should be treated as hard constraints.
</assumption_protocol>

<anti_anchor_protocol>
You must use the following anti-anchor sequence:
step 1: derive requirements from the game, action structure, information structure, and objectives BEFORE reading doctrine excerpts as conclusions
step 2: define candidate architecture families in abstract terms
step 3: steelman the current Hydra plan as one candidate, not the baseline truth
step 4: steelman at least one serious non-conv family and at least one hybrid family
step 5: compare all families under the same rubric
step 6: perform a red-team pass against the leading candidate
step 7: only then state the recommendation, if any

You must explicitly answer: “what would have made me incorrectly choose the current doctrine by default?”
</anti_anchor_protocol>

<minimum_candidate_family_set>
At minimum, compare all of these families unless an artifact-grounded reason makes one literally inapplicable:
family 1: pure fixed-tensor residual conv family
family 2: SE-ResNet or related channel-attention conv family
family 3: ConvNeXt-style or modernized conv family over tile axis
family 4: event-sequence transformer family
family 5: tile-token or entity-token transformer/set-transformer family
family 6: recurrent or state-space event-history family
family 7: dual-path hybrid family (fixed tensor trunk + history encoder)
family 8: graph/entity/set family over players, melds, discards, and tile groups
family 9: asymmetric actor/learner family where fast actor and slow learner/search-side network differ materially
family 10: any adjacent-field formulation that survives validation and does not collapse into one of the above

For each family, you must say whether it is:
- best overall
- best under ideal-but-still-realistic Hydra objectives
- best under current runtime/search constraints
- only good as a subsystem
- only good as a teacher-side model
- reserve shelf only
- or reject
</minimum_candidate_family_set>

<evaluation_rubric>
Use a weighted rubric with explicit scoring or explicit pairwise dominance logic.
First-order dimensions that must be evaluated:
- representation fit to Mahjong’s public state and partial observability
- ability to exploit tile geometry and local combinatorics
- ability to capture temporal opponent-read patterns and discard/call sequences
- sample efficiency under Hydra-like compute budgets
- fast-path inference latency for actor/runtime use
- slower learner/search-side usefulness under pondering/distillation
- compatibility with multi-head supervision (policy, value, GRP, danger, tenpai, belief, search residuals, etc.)
- robustness when dynamic search/belief/Hand-EV features are absent or stale
- ease of distilling search/oracle signals into the deployable policy
- support for selective search rather than universal expensive search
- calibration potential for safety and belief outputs
- scaling path if Hydra later earns more compute

Tie-breakers that may matter only after first-order comparison:
- implementation complexity
- maintenance burden
- stack compatibility
- debugging surface area
- profiling predictability
- licensing or ecosystem friction if relevant

Do not let a tie-breaker decide the winner if a family is materially worse on first-order fit.
</evaluation_rubric>

<required_questions>
You must answer all of these:
Q1. What information patterns actually dominate strong Mahjong play: local tile-shape reasoning, cross-player relational reasoning, temporal opponent modeling, search-conditioned adaptation, or some mixture?
Q2. Which of those patterns need to live in the deployable fast actor, and which can be outsourced to teacher/search/pondering/distillation machinery?
Q3. Is the best architecture likely to be single-path or multi-path?
Q4. Should actor and learner share architecture at all, or only share some representation ideas?
Q5. Is the current 192x34 tensor a core strength, a neutral compatibility layer, or an anchor holding the project back?
Q6. Is opponent-history modeling central enough to require a dedicated sequence module?
Q7. Is the best architecture likely to preserve explicit safety channels and structured Hand-EV/belief features, or absorb them into a different representation?
Q8. Is the best architecture likely to be end-to-end monolithic, or should it be modular with specialized trunks or sidecars?
Q9. What is the smallest architecture leap that has a realistic chance to beat the current plan?
Q10. What would falsify the recommended architecture quickly and cheaply?
</required_questions>

<required_output_shape>
The answer must be a blueprint with these practical deliverables:
- an assumption ledger
- a hard-facts section
- a contestable-doctrine section
- a candidate-family generation section
- a family-by-family evaluation table
- a steelman for SE-ResNet
- a steelman for the strongest non-SE alternative
- a comparison of ideal architecture vs best practical Hydra architecture if they differ
- a recommended architecture or an explicit underdetermined verdict
- a decisive experiment matrix
- a migration blueprint if the recommendation differs from current doctrine
- revisit triggers that would cause the decision to change later

The answer must feel buildable or directly auditable.
</required_output_shape>

<failure_modes_to_ban>
- do not answer with “SE-ResNet is good enough” unless you show why stronger candidates fail
- do not answer with “transformers are better” unless you show why they win under Hydra’s actual objectives
- do not let the current codebase shape masquerade as proof of optimality
- do not treat old design docs as ground truth if live runtime docs or code disagree
- do not cite architecture papers abstractly without method details, scope limits, or failure cases
- do not declare exact thresholds or budgets without visible support or explicit proposal status
- do not force a single-family winner if the real answer is asymmetric actor/learner or hybrid trunk/sidecar
- do not confuse teacher-side architecture freedom with fast actor requirements
- do not confuse a better opponent-modeling subsystem with a better full-agent backbone
</failure_modes_to_ban>

<architecture_search_notes>
Important: “best architecture for Hydra” may mean one of the following, and you must disentangle them:
- best full system architecture if we were rebuilding from scratch
- best deployable fast actor architecture
- best slow learner / teacher architecture
- best search-side value / policy / belief helper architecture
- best migration target from the current repo state

You may recommend different architectures for different roles.
You may recommend preserving the current actor while changing the learner, or vice versa.
You may recommend a hybrid where the current SE-ResNet remains only as one component.
You may recommend rejecting SE-ResNet entirely.
You may recommend keeping SE-ResNet.
You may recommend something more exotic if the evidence survives scrutiny.
But every important move must be justified.
</architecture_search_notes>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use the paired answer body plus the repository source files for audit and follow-up.
</artifacts>

<final_reminders>
Your job is not to defend the current doctrine.
Your job is not to be contrarian for sport either.
Your job is to determine the best architecture for Hydra as rigorously as possible.
It does not have to be SE-ResNet.
Assume we can code anything.
Still respect actual problem constraints, actual runtime goals, and actual compute realities.
If the answer is “keep SE-ResNet but add a dedicated history sidecar,” say that.
If the answer is “SE-ResNet actor, different learner,” say that.
If the answer is “full replacement,” say that.
If the answer is “underdetermined; run these experiments,” say that.
But do not stop before the reasoning is saturated or blocked.
</final_reminders>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="combined.md">
  <![CDATA[# Hydra architecture blueprint

## 0. Decision

Do **not** make Hydra a pure SE-ResNet project, and do **not** replace it with a pure transformer project.

The best architecture family for Hydra is an **asymmetric multi-path system**:

* a **convolutional static-state path** over the 34-tile axis for local tile combinatorics and cheap public-structure processing,
* a **dedicated public-event history path** for discard/call/riichi order and opponent reading,
* **late-fused optional belief/search inputs** rather than treating dynamic planning outputs as the canonical state,
* **tile-aware heads** for tile-indexed outputs and **global heads** for non-tile actions and scalar predictions,
* a **materially different learner/search-side model** from the fast actor.

In concrete terms:

* **Fast actor**: keep the current 12-block SE-style 1D residual trunk as the static path, add a **small recurrent history sidecar** (default: GRU), split policy into **discard-tile spatial head + global-action head**.
* **Learner / teacher / search helper**: keep the 24-block SE-style static trunk, add a **richer event-history transformer sidecar**, and late-fuse belief/search tokens there.
* **Do not replace the 192×34 path.** Reclassify it as the **static compatibility path**, not the whole information state.

That is the best practical Hydra architecture under current objectives.
The best ideal learner/search-side architecture, if Hydra later earns more data/compute, is a **three-path hybrid**: static conv path + event-history transformer + optional entity/belief/search token path.

The key principle is simple:

> **Use convolution where Mahjong is local and structured; use sequence modeling where Mahjong is ordered and lossy under fixed tensors.**

---

## 1. Assumption ledger

| Bucket                                                   | Contents                                                                                                                                                                                                                           |         Hardness | What follows                                                                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------: | ---------------------------------------------------------------------------------------------------------------------------- |
| **A. Irreducible game facts**                            | 4-player general-sum imperfect information; multi-round placement-aware scoring; huge hidden-state uncertainty; interrupted action flow; conserved shared tile pool; strong local tile-shape structure on 34 tile types            |         **Hard** | Architecture must handle both **local tile geometry** and **temporal public history**, while respecting hidden-tile coupling |
| **B. Explicit project objectives / runtime constraints** | Maximize stable rank, support fast deployable actor, use selective search rather than universal heavy search, operate under moderate compute and real inference budgets, preserve deployability in the current Rust stack          |  **Mostly hard** | Actor latency matters; learner/search model can be richer; selective search/distillation is allowed and desirable            |
| **C. Current repo realities**                            | Live encoder is 192×34 fixed-superset; current trunk is SE-style residual conv; actor/learner are 12/24 blocks; many advanced heads exist; advanced supervision is only partially active; current policy path uses pooled features |         **Soft** | These are starting constraints for migration, not proof of optimality                                                        |
| **D. Contestable doctrine / design bets**                | “SE-ResNet is the answer,” “192×34 is the state,” “single backbone + shallow heads is best,” “actor and learner should match,” “search/belief should be input planes”                                                              | **Overturnable** | These must be proved, not inherited                                                                                          |

Bucket A comes from the game itself and the strongest external evidence: Suphx describes Riichi Mahjong as a four-player imperfect-information game with multi-round placement-aware rewards, irregular interruptions from meld/win actions, and more than (10^{48}) hidden states per information set on average. The artifacts add the other hard fact: hidden tiles sit in a conserved shared pool, so beliefs are coupled by exact tile counts rather than independently factorized. ([arXiv][1]) (research/design/HYDRA_FINAL.md:47-55)

---

## 2. Hard facts

### 2.1 Static tile geometry is real, frequent, and cheap to exploit

Suphx used 34-column convolutional models, added 100+ look-ahead features as 34-dimensional vectors, and explicitly said it avoided pooling because each tile column has semantic meaning. JueJong, though 1v1 rather than 4p Riichi, still stayed in the residual/CNN family and explicitly encoded the latest 24 discards **in order** rather than abandoning structure for a pure sequence backbone. ([arXiv][1])

A 1D (3)-kernel trunk over width (34) does **not** lack global static reach. With one stem conv and two (3\times1) convs per block, receptive field is

[
\text{RF} = 1 + 2(1 + 2B).
]

So:

* (B=12 \Rightarrow \text{RF}=51)
* (B=24 \Rightarrow \text{RF}=99)

Both exceed width (34).
So the static conv trunk can already integrate whole-board **static** context. If attention helps Hydra, it is not because the conv trunk “cannot see the whole board.” It is because the missing information is **ordered history** and **optional modality fusion**.

### 2.2 Ordered public history is the missing modality

The strongest evidence packet points the same way. JueJong devotes 24 feature maps to the latest 24 discards in order. Suphx uses a recurrent GRU model for game-level reward prediction across rounds. Your own artifact on Mahjong techniques identifies the gap as multi-step reasoning over discard/call chains, and explicitly flags attention over discard sequences as a plausible upgrade over pure CNN treatment. ([arXiv][1]) (research/intel/MAHJONG_TECHNIQUES.md:405-413)

This is the central architecture fact: **Mahjong’s missing signal in fixed tile tensors is not static board context; it is event order.**

### 2.3 Search matters, but as a selective overlay, not as the actor’s identity

OLSS’s Mahjong experiments used a learned blueprint and environmental model, both based on small residual networks, and then used **pUCT** because CFR-style search was too simulation-hungry in that setting; they report meaningful gains at 1000 simulations while CFR at 5000 was still inadequate. ReBeL and Student of Games reinforce the same system pattern in imperfect-information games more broadly: strong learned blueprint + search + self-play + distillation, not “one giant inference-time planner everywhere.” ([Proceedings of Machine Learning Research][2])

So Hydra’s architecture should be designed to support **selective search and distillation**, not to make every deployable actor forward pass depend on heavy planner-state inputs.

### 2.4 Token/transformer Mahjong is plausible, but the strongest evidence still does not make it the default winner

Kanachan is the best public steelman for raw-token transformers in Riichi Mahjong: it explicitly argues that much larger datasets make more expressive models like transformers viable, represents many aspects of state as tokens and sequences instead of human-crafted planes, and frames that as a conscious trade of feature engineering for data and model scale. ([GitHub][3])

Tjong is relevant but weak evidence here because only abstract-level access was available in this session. Its abstract reports a 15M-parameter transformer with hierarchical decision-making, trained on roughly 0.5M data over 7 days, outperforming multiple baselines in its environment. That makes transformer formulations credible; it does **not** outweigh the stronger Riichi-specific conv evidence from Suphx plus the structured residual evidence from JueJong. ([Directory of Open Access Journals][4])

### 2.5 The repo itself already proves the current doctrine is not a hard constraint

The live encoder is already a **192×34** fixed-superset, not the old 85×34 monolith, and it already carries dynamic feature presence masks. The current code also exposes both `spatial` and `pooled` trunk outputs. But the current model feeds pooled features into the policy head, while Suphx explicitly avoided pooling for tile-semantic reasons. So the repo is **not** a proof that “one pooled shared trunk to shallow heads” is the best Hydra architecture; it is a partial implementation with at least one architecturally meaningful simplification still present. (docs/GAME_ENGINE.md:122-125; hydra-core/src/encoder.rs:23-117; hydra-train/src/model.rs:247-258; hydra-train/src/backbone.rs:138-144) ([arXiv][1])

---

## 3. Contestable doctrine

| Doctrine                                                | Verdict                                                          | Reason                                                                                                                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **“SE-ResNet is the best Hydra architecture.”**         | **Reject as whole-system claim; keep as static trunk candidate** | SE-style conv is strong for the static tile path, but it does not solve ordered history on its own                                                                  |
| **“192×34 is the state.”**                              | **Reject**                                                       | It is a useful **static compatibility layer**, not the whole information state                                                                                      |
| **“One backbone plus shallow pooled heads is enough.”** | **Reject**                                                       | Tile-indexed outputs should read spatial tile features; pooling everything is the wrong default for discard policy                                                  |
| **“Actor and learner should share architecture.”**      | **Reject**                                                       | Role constraints differ: actor needs streaming latency and robustness without search; learner can absorb richer history and optional dynamic tokens                 |
| **“Belief/search/Hand-EV should all be input planes.”** | **Split**                                                        | Safety and Hand-EV are worth explicit early structure; optional belief/search signals should move toward late fusion                                                |
| **“The exact SE flavor is sacred.”**                    | **Reject**                                                       | Even the repo differs from the older spec: code uses mean-pool SE, while older doctrine described dual-pool CBAM-style SE (research/design/HYDRA_SPEC.md:192-194 vs. hydra-train/src/backbone.rs:31-37) |
| **“More heads are the bottleneck.”**                    | **Reject**                                                       | The artifacts already show the bigger issue is inactive supervision and lossy representation routing, not missing head count                                        |

---

## 4. Candidate-family generation

The families to compare are:

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

Load-bearing external evidence for this comparison is Suphx, JueJong/ACH, OLSS, Kanachan, and ConvNeXt. Tjong, graph/set/SSM papers are useful but lighter evidence here. ([arXiv][1])

---

## 5. Evaluation rubric

Weights below are explicit. Scores are coarse and only meant to separate families, not pretend statistical certainty.

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

Rule for interpretation: a family does **not** win on tie-breakers if it loses materially on representation fit, temporal modeling, or sample efficiency.

---

## 6. Family-by-family evaluation

These totals are coarse weighted judgments from the rubric above.

| Family                                                |     Approx. score / 100 | Best role                        | Verdict                                                |
| ----------------------------------------------------- | ----------------------: | -------------------------------- | ------------------------------------------------------ |
| 1. Pure fixed-tensor residual conv                    |                  **80** | Fast actor baseline              | Strong reserve baseline; not best overall              |
| 2. SE-ResNet / channel-attention conv                 |                  **82** | Static trunk for actor + learner | **Keep as component**, not as the whole answer         |
| 3. ConvNeXt-style conv                                |                  **74** | Later trunk challenger           | Reserve shelf only                                     |
| 4. Pure event-sequence transformer                    |                  **63** | Teacher/history module           | Reject as full backbone under current Hydra objectives |
| 5. Pure tile/entity transformer or set-transformer    |                  **72** | Learner-side research challenger | Teacher-side / reserve shelf only for now              |
| 6. Pure recurrent/state-space history family          | **66** as full backbone | Actor-side history subsystem     | Good **subsystem**, reject as full replacement         |
| 7. Dual-path hybrid (static tensor + history encoder) |                  **90** | Main backbone family             | **Best practical backbone family**                     |
| 8. Graph/entity/set backbone                          |                  **70** | Opponent/belief helper           | Subsystem or reserve shelf only                        |
| 9. Asymmetric actor/learner family                    |                  **93** | Whole system                     | **Best full-system answer**                            |
| 10. Search/distillation/belief overlay                |                     n/a | System overlay                   | Necessary overlay, not a backbone choice               |

### Pairwise dominance

* **7 dominates 1/2** because it keeps the static conv strengths and adds first-class temporal modeling at modest cost.
* **5 is the strongest non-SE challenger**, but loses to 7 on sample efficiency, migration risk, and current-budget plausibility.
* **6 beats 4 on actor practicality**, because a recurrent sidecar can be streamed incrementally, but it loses to 4 in learner-side global event interaction.
* **8 does not beat 7** because graph bias helps relations, but does not eliminate the need for ordered event modeling.
* **9 wins system-level design** because actor and learner have genuinely different jobs.

---

## 7. Steelman for SE-ResNet

If forced to choose **one single-family Hydra architecture** and ban sidecars, the best answer is still **SE-ResNet over the fixed tile tensor**, not a transformer.

Why that steelman is real:

1. **It matches the most universal signal.** Every decision uses local tile-shape reasoning; not every decision needs deep history or search.
2. **It is well-supported by strong Mahjong evidence.** Suphx’s strongest public Riichi result used convolutional networks over 34-tile columns and explicitly preserved tile-column semantics. JueJong’s strong 1v1 result also stayed in the residual/CNN family. ([arXiv][1])
3. **It is sample-efficient under moderate compute.** Kanachan’s own argument for transformers is “huge data + large expressive models,” which is not Hydra’s current budget regime. ([GitHub][3])
4. **It already solves the static-axis global-context problem.** On width 34, the static trunk’s receptive field is already global.
5. **SE itself is low-risk.** Channel reweighting is cheap and current Hydra already has it.

So the answer is **not** “SE-ResNet is wrong.”
The answer is: **SE-ResNet is only the static half of the answer.**

---

## 8. Steelman for the strongest non-SE alternative

The strongest non-SE alternative is **not** ConvNeXt, graph networks, or Mamba as a whole-agent backbone.

It is:

> **An entity-token / event-token transformer (or set-transformer hybrid) that represents tiles, players, melds, discards, dora/meta, and optional belief/search tokens in one unified token space.**

Why this is the strongest challenger:

* It can model **cross-player relations** directly.
* It can treat **ordered public history** as first-class.
* It can ingest optional belief/search tokens elegantly.
* It aligns with Kanachan’s explicit claim that raw tokenization plus huge data should let more expressive models beat feature-engineered CNNs. ([GitHub][3])
* Set-transformer style modules are a natural fit for unordered subsets such as tile multisets or meld collections, and attention-based set models were designed precisely to model interactions while preserving permutation structure. ([Cool Papers][5])
* Graph-network thinking also supports this challenger by emphasizing relational inductive bias over entities and relations. ([Google Research][6])

Why it still loses **for Hydra now**:

1. **Evidence quality is weaker.** The strongest public Riichi evidence is still conv-centric.
2. **The local tile prior matters.** A pure token model has to relearn suit-local combinatorics that conv gets almost for free.
3. **Hydra’s budget is not a “just scale it” regime.** Kanachan’s own README frames raw-token modeling as a data/compute trade. ([GitHub][3])
4. **Migration and debugging risk are much higher** in the present stack.
5. **The actual missing modality is history**, not static-tile global reach.

So this challenger belongs on the **learner-side reserve shelf**, not on the mainline actor path.

---

## 9. Red-team pass against the leading candidate

The leading candidate is the **asymmetric hybrid**. The strongest arguments against it are real:

* Maybe the current 192×34 tensor already captures enough order through recency planes and tedashi flags.
* Maybe a history sidecar improves auxiliary heads but not actual policy strength.
* Maybe the added path hurts actor latency more than it helps.
* Maybe the true missing fix is not a new path at all, but simply routing tile-indexed heads to spatial features instead of pooled ones.
* Maybe a pure token model only looks expensive because the current benchmarks are unfair.

Those objections change the rollout order, but not the family ranking:

1. **Fix tile/global head routing first** so the baseline is not artificially handicapped.
2. **Validate history with a collision benchmark** where identical static tensors map to divergent targets because history differed.
3. **Require order-sensitivity** via history-shuffle ablations.
4. **Keep actor-side history cheap and incremental** unless a small transformer proves latency-safe.

### What would have made me incorrectly choose the current doctrine by default?

Three things:

1. Treating the repo shape as proof of optimality rather than implementation history.
2. Looking at Suphx and seeing “CNNs win,” while missing that Suphx also used **separate action-type models, no pooling on tile semantics, look-ahead features, oracle guiding, and a recurrent reward model**. ([arXiv][1])
3. Treating 192×34 as the whole information state instead of the **static view** of the information state.

---

## 10. Direct answers to Q1–Q10

**Q1. What information patterns actually dominate strong Mahjong play?**
A mixture. **Local tile-shape reasoning** is the universal base; **temporal opponent modeling** is the highest-leverage missing public signal; **cross-player relational reasoning** matters but is usually mediated through public history and score context; **search-conditioned adaptation** matters on a hard minority of states, not every state.

**Q2. Which patterns need to live in the deployable fast actor?**
The actor must carry **local tile geometry**, **basic public-history opponent reading**, **explicit safety structure**, and **placement-aware value tendencies**. Deep belief/search-conditioned adaptation can stay in the learner/search/teacher stack.

**Q3. Is the best architecture single-path or multi-path?**
**Multi-path.** Static tiles and ordered public history are different modalities and should not be forced through one representation.

**Q4. Should actor and learner share architecture?**
They should share **representation ideas, event schema, and distillation interfaces**, not identical architecture.

**Q5. Is the current 192×34 tensor a strength or an anchor?**
It is a **strong compatibility layer** and becomes an anchor only if treated as the whole state.

**Q6. Is opponent-history modeling central enough to require a dedicated sequence module?**
**Yes.** At least learner-side definitely; actor-side probably yes in lightweight recurrent form.

**Q7. Preserve explicit safety and Hand-EV / belief features, or absorb them?**
Preserve **safety** and **Hand-EV** explicitly. Keep **belief/search** structured too, but move them toward **late fusion** rather than mandatory early input.

**Q8. Monolithic or modular?**
**Modular.** Static trunk + history sidecar + optional planning/belief sidecars.

**Q9. Smallest architecture leap with realistic win chance?**
First: **tile-aware spatial/global head split**.
Second: **dedicated history sidecar**, learner-first, actor-next.

**Q10. What falsifies the recommendation quickly and cheaply?**
If a hybrid cannot beat conv-only on a **same-static-tensor/different-history collision benchmark**, or if **order-shuffling the history input barely changes performance**, the history path is not earning its keep.

---

## 11. Ideal architecture vs. best practical Hydra architecture

| Role                  | Ideal if Hydra later earns more compute/data                                            | Best practical Hydra architecture now                                                            |
| --------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Fast actor**        | Static conv path + tiny event transformer or GRU + spatial/global split heads           | **12-block SE-style static trunk + cached GRU history sidecar + spatial/global split heads**     |
| **Learner / teacher** | Static conv path + event-history transformer + optional entity/belief/search token path | **24-block SE-style static trunk + event-history transformer + late-fused belief/search tokens** |
| **Search helper**     | Learner-style hybrid with richer optional entity/belief modules                         | **Learner-style hybrid first; no new full replacement backbone yet**                             |

The ideal and practical answers differ in **learner richness**, not in the family-level choice. Even with more compute, I would still keep a static conv path.

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

Keep the current fixed-shape static path for now:

* input: current **192×34** tensor
* actor trunk: **12-block** 1D pre-activation SE-style residual conv
* learner/search trunk: **24-block** version
* keep GroupNorm-style normalization and current deployment-friendly stack

But reclassify the planes:

* **canonical early input**: public state, safety, Hand-EV
* **compatibility/optional input**: search/belief planes with masks

This is not because belief/search are unimportant. It is because optional/stale dynamic features should not define the base actor representation. (docs/GAME_ENGINE.md:122-125; hydra-core/src/encoder.rs:23-117; research/design/HYDRA_FINAL.md:245-248)

## 12.3 History path

### Actor history path: recurrent, cached, incremental

Default actor choice:

* **1-layer GRU**, hidden size **128**
* sequence cap **96** public events
* persistent hidden state updated on each public event
* reset at hand boundary

Why GRU on actor:

* event histories are naturally streaming,
* hidden state can be cached and updated incrementally,
* cost is tiny relative to the conv trunk,
* latency is predictable.

A rough compute check puts a 12-block conv trunk around **165M MACs**, a tiny (T=96,d=128) 2-layer transformer sidecar around **42M MACs**, and a GRU sidecar around **9–10M MACs**. So the actor-side history module is affordable either way, but cached recurrence is the cleanest default fast path.

### Learner history path: attention over full public history

Default learner choice:

* **3-layer transformer**
* (d_{\text{model}} = 128) or (192)
* 4 heads
* seat embeddings + relative/causal position encoding
* full per-event hidden states retained for cross-attention

Why transformer on learner:

* the learner benefits from richer pairwise event interactions,
* sequence lengths are short enough that quadratic cost is not the bottleneck,
* the transformer is better suited to distilling opponent-read patterns and optional side tokens than a single recurrent summary.

Mamba/SSM stays reserve-shelf only: its main advertised advantage is long-sequence throughput, and Hydra’s public histories are not long enough for that to dominate the design decision. ([arXiv][7])

## 12.4 Event schema

Use a real event vocabulary, not only recency-weighted discard planes.

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

This schema is enough to make **order**, **seat**, **call interruptions**, and **riichi timing** first-class.

## 12.5 Fusion

Use **late fusion**, not “concatenate everything at the input and hope.”

Let

* (Z \in \mathbb{R}^{34 \times C}): static tile features from the conv trunk
* (\bar Z \in \mathbb{R}^{C}): pooled static summary
* (H): history summary or history token matrix
* (D): optional late-fused belief/search summary

Then use:

[
\tilde Z = Z + \text{CrossAttn}(Q=W_Q Z,\ K=H,\ V=H) + g(H,D)\odot Z
]

for learner, and a lighter gated affine version for actor:

[
\tilde z_k = z_k + A(H,D) + g(H,D)\odot z_k.
]

The key is that **history modulates tile features**, not just the final scalar head.

## 12.6 Heads

This is the most important architectural correction after adding history.

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

Current Hydra code already surfaces `spatial`, but the policy path is currently fed pooled features. That should change. (hydra-train/src/model.rs:247-258)

### Global heads should read fused pooled context

Use pooled fused context for:

* riichi / chi / pon / kan / agari / ryuukyoku / pass logits
* value
* score distribution / CDF-PDF
* GRP / placement
* opponent tenpai summary
* mixture weights / meta uncertainty summaries

So the policy becomes a **factorized 46-action head**:

* **37 discard logits** from tile branch
* **9 global-action logits** from global branch

Keep the fixed 46-action interface externally. Change the internal head semantics.

### Learner-only or teacher-biased heads

Keep these primarily on learner / teacher unless proved actor-useful:

* `delta_q`
* `safety_residual`
* raw belief-field / mixture / opponent-hand-type auxiliaries

And if belief supervision is used, target **projected public-teacher belief objects**, not raw Sinkhorn potentials, which matches the reconciliation artifact’s caution. (research/design/HYDRA_RECONCILIATION.md:386-441)

## 12.7 Safety, Hand-EV, belief, and search features

### Keep explicit early:

* **safety channels**
* **Hand-EV features**

Reason: these are cheap, structured, high-value transforms of public state or near-solved single-player subproblems. Suphx explicitly reports look-ahead features as important, and your artifact correctly treats safety encoding as high-ROI domain structure. ([arXiv][1]) (research/intel/MAHJONG_TECHNIQUES.md:441-490)

### Move toward late fusion:

* **belief marginals**
* **search deltas**
* **ESS / entropy / robust stress**
* **search-only risk summaries**

Reason: they are optional, sometimes stale, and selective-search dependent.

Hydra already discovered the right idea with presence masks. Keep the masks; move the fusion later.

## 12.8 Search-side role

Use the learner-style hybrid as the **search blueprint/value/prior model**.

Do **not** make raw search dependence a mandatory actor input.
Do **distill** search residuals and policy improvements into the actor.

This is aligned with both the Hydra artifacts and external imperfect-information search systems. ([Proceedings of Machine Learning Research][2])

---

## 13. Reject, defer, keep

**Reject now**: pure transformer actor replacement; pure graph/entity backbone as mainline; identical actor and learner; pooled-only policy head; universal early-fusion dependence on search/belief features.

**Defer**: ConvNeXt-style trunk rewrite; learner-only entity-token challenger; Mamba/SSM history sidecar; graph-based opponent/belief helper.

**Keep**: SE-style static trunk; explicit safety channels; explicit Hand-EV; selective search; search distillation; structured belief modules outside the actor core.

---

## 14. Decisive experiment matrix

These are the minimum experiments that actually decide the remaining uncertainty.

| ID     | Question                                                                    | Compare                                                                     | Budget                           | Proposed pass / fail gate                                                                                            |
| ------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **E0** | Is the current head routing itself leaving strength on the table?           | pooled policy head vs **spatial discard + global action** split, same trunk | cheapest                         | Pass if discard CE or matched-latency self-play improves; fail kills head-rewrite priority                           |
| **E1** | Does history contain actor-useful information beyond current static tensor? | conv-only vs conv+GRU(actor) vs learner+history                             | cheap offline + short self-play  | Pass only if hybrid wins on **collision benchmark** and temporal-slice danger/opp-next/policy metrics                |
| **E2** | Is the actor sidecar better as GRU or tiny transformer?                     | same trunk, same event schema                                               | medium                           | Pick transformer only if it clears actor latency gate with measurable strength gain; otherwise GRU                   |
| **E3** | Should belief/search stay early or move late?                               | early planes vs late tokens vs both                                         | medium                           | Pass if late fusion is more robust under absent/stale dynamic features                                               |
| **E4** | Is a full replacement transformer actually better?                          | matched-param pure entity/tile transformer vs hybrid                        | medium-high, one challenger only | Promote only if it wins offline temporal slices **and** matched-budget self-play **and** latency-adjusted deployment |
| **E5** | Is ConvNeXt-style modernization worth a trunk rewrite?                      | matched-param SE trunk vs modernized conv trunk                             | optional                         | Only pursue if hybrid is already validated and trunk still looks like the bottleneck                                 |

### The two cheapest falsifiers

**Falsifier A: static-collision benchmark**

* Zero dynamic search/belief channels.
* Hash the remaining actor input.
* Collect clusters where the same static tensor appears with materially different targets because history differed.
* If the history model does **not** win here, it is not solving the problem it was added for.

**Falsifier B: order-shuffle ablation**

* Keep event multiset and event identities fixed.
* Randomly shuffle order in the last (N) public events.
* If performance barely changes, the history module is not using order and should be killed or simplified.

---

## 15. Migration blueprint

The architecture decision does **not** require restarting Hydra from zero. It changes what gets built next.

### Phase 1 — correct the existing trunk/head interface before family expansion

Keep the current supervision-first execution order from reconciliation, but make the baseline architecturally honest.

Touch points:

* `hydra-train/src/model.rs`
  Add `DiscardTileHead`, `GlobalActionHead`, `FusionBlock`, `HistoryEncoder`.
* `hydra-train/src/heads.rs`
  Route tile-indexed heads from spatial features.
* `hydra-train/src/backbone.rs`
  Keep current SE-style trunk for now.
* `hydra-train/src/training/losses.rs`
  Keep optional-head gating explicit.

This phase alone may yield real strength, because current policy pooling is likely too lossy for tile-indexed actions. (hydra-train/src/model.rs:247-258) ([arXiv][1])

### Phase 2 — add event-history plumbing, learner first

Touch points:

* `hydra-core/src/bridge.rs`
  Emit incremental `EventToken`s from public game events.
* `hydra-train/src/data/sample.rs`
  Add padded event sequences + masks to batches.
* `hydra-train/src/data/mjai_loader.rs`
  Reconstruct event histories from logs; create temporal benchmark slices.
* `hydra-train/src/model.rs`
  Add learner-side transformer history encoder and late fusion.

Do **not** break the 192×34 interface yet.

### Phase 3 — distill history-aware learner into lightweight actor

Touch points:

* `hydra-train/src/model.rs`
  Actor uses GRU history sidecar.
* `hydra-train/src/inference.rs`
  Cache recurrent hidden state across public events.
* `hydra-core/src/game_loop.rs` or equivalent runtime path
  Reset hidden state on hand start; update on each public event.

This phase is where actor/learner asymmetry becomes structural rather than just “12 blocks vs 24 blocks.”

### Phase 4 — move optional dynamic features to late fusion

Touch points:

* `hydra-core/src/encoder.rs`
  Keep current planes for compatibility, but stop treating them as semantically canonical.
* `hydra-train/src/model.rs`
  Add `BeliefSearchAdapter` that turns optional search/belief summaries into late-fused tokens or summaries.
* `losses.rs` / distillation code
  Train with feature dropout and presence-gated fusion.

### Phase 5 — reserve challengers only if the hybrid validates

Only after the hybrid wins E0–E3:

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
| Hybrid fails collision benchmark / order-ablation tests                              | Drop or learner-limit the history path                               |
| Pure entity/token transformer wins matched experiments                               | Re-open conv-trunk assumption                                        |
| Tile-aware head rewrite alone captures most of the gain                              | Delay history-sidecar migration on actor, keep it learner-side first |

---

## 17. Final recommendation

**Best full-system architecture for Hydra:**
**Asymmetric dual-path hybrid.**

* **Keep** an SE-style 1D residual trunk over the 192×34 static path.
* **Add** a dedicated public-history encoder.
* **Use GRU on the actor, transformer on the learner.**
* **Split policy and other tile-indexed heads into spatial tile heads + global heads.**
* **Keep safety and Hand-EV explicit.**
* **Move belief/search toward late fusion.**
* **Do not force actor and learner to match.**
* **Do not make pure transformer or pure graph backbones the mainline now.**

The strongest concise statement is:

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
