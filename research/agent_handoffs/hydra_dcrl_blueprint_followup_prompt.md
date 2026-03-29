# Hydra prompt — DCRL follow-up implementation blueprint

<role>
You are producing an implementation-ready Hydra blueprint for a narrow follow-up lane.
Assume the earlier worth-it triage is already done; your job is to turn that conclusion into the strongest exact plan.
</role>

<task>
Produce the exact blueprint for how a DCRL-style mixed-advantage critic lane would map onto Hydra if Hydra wanted the smallest rigorous follow-up beyond the prior docs-positioning verdict.

We need a buildable answer that makes clear:
- the exact formula Hydra should test, including whether and how to define A_std, A_orc, beta, and any detach / stop-gradient boundaries
- whether the proposal should reuse existing value/oracle heads or add a new head, target, or loss surface
- the exact code files and functions to change, in what order, and why
- whether this belongs in baseline, implemented-but-not-default-on, implemented-but-staged, or reserve shelf
- the exact TrainingPhase where it should first appear, and how it should move through phase gating if it survives
- how HYDRA_FINAL.md and any other docs should be updated to describe the lane without overwriting Hydra's current identity
- what train mode, artifact, or promotion path should exist if this should mirror the DeltaQ-style promotion workflow
- what measurable acceptance criteria should control promotion or rejection
- what falsifies the lane and what the fallback should be if the lane fails

Do not redo the broad worth-it debate. Start from the prior answer and produce the narrowest implementation-ready blueprint that fits current Hydra doctrine and code reality. Use the artifacts below to derive your conclusions.
</task>

<rules>
- distinguish direct artifact support from your own inference
- use search/browse aggressively when it can strengthen the answer: find the original paper, adjacent papers, official docs, repos, and other primary sources; use abstracts or summaries mainly for discovery, not as the final evidence base
- use the bash tool to run Python for lightweight research support work when helpful: calculations, math checks, parsers, data inspection, sanity checks, quick experiments, and validation
- do not dump logic; every important mechanism, threshold, or recommendation should be inferable from evidence or made explicit in the blueprint so it can be validated and reproduced
- if you claim a path works, survives, or is implementation-ready, show why that confidence is justified and how the claim can be validated or falsified later
- inspect your own draft before finishing: if a confident claim is not objectively justified by visible evidence, downgrade it to inference, proposal, or blocked
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated, falsified, or truly blocked, and do not stop just because the first pass produced a plausible answer
- separate 'docs positioning patch' from 'code/phase blueprint' so the answer does not blur them together
- if the best answer is 'ablation only, not mainline', still give the exact ablation blueprint rather than stopping at policy advice
- prefer reusing existing Hydra surfaces over inventing new heads or runtime complexity unless the artifacts force a new surface
- make every proposed file touchpoint concrete enough that a coding agent could open the files and start implementing in order
- when you recommend a phase or lane status, tie it to the existing TrainingPhase / PipelineState / head-gating machinery rather than vague project language
</rules>

<style>
- no high-level survey
- no vague answer
- include reasoning
- when you sound confident, show the justification for that confidence level
- for every important claim, make the validation path visible enough that a reviewer can test it later
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate, reproduce, or falsify it ourselves (pdfs, sources, links, similar projects, concrete checks)
- lead with a one-paragraph verdict on the exact smallest viable Hydra blueprint
- include formulas and pseudocode when they help make the update object unambiguous
- include a compact file-by-file change plan and a compact promotion-gate checklist
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>

## Artifact 01 — Follow-up context and hard scope
Artifact id: `followup-context`
Source label: META
Type: `literal`
Why it matters: Pins the task to the already-narrowed question so the agent does not waste time re-arguing whether DCRL is worth major method attention.

```text
Prior answer is already available in the packet and should be treated as the narrowed starting point.
The previous verdict was: docs yes lightly, method no active reaction now, but a future ablation note could be justified if Hydra's actor-critic RL lane becomes active and unstable.
This follow-up is not asking whether DCRL is important. It is asking for the exact Hydra blueprint if we wanted the smallest rigorous implementation-ready plan for that lane.
We need the exact formula, the exact code surfaces, how it would coexist with the current baseline and staged lanes, what phase it belongs in, what docs should change, and what gates/fallback would control it.
```

## Artifact 02 — Prior narrowed answer
Artifact id: `prior-answer`
Source label: PREV
Type: `file_full`
Source: `research/agent_handoffs/agent_answer.md`
Why it matters: This is the already-completed worth-it answer. The new task must start from this conclusion rather than redoing the worth-it analysis.

```markdown
[PREV L0001] ## Verdict
[PREV L0002] 
[PREV L0003] **docs-worth-it:** **Yes, lightly.** Hydra should explicitly position its oracle-critic / CTDE surfaces against the asymmetric-critic lineage, but the right patch is small: cite **UAAC** and **DCRL** (and optionally the earlier asymmetric actor-critic paper), then state clearly that Hydra’s project-defining lane is still **public-state ExIt/search-as-feature for Mahjong**, not “dual critic” as a standalone novelty. **Confidence: medium-high.**
[PREV L0004] 
[PREV L0005] **method-worth-it:** **No material reaction now.** No architecture reprioritization, no new reserve lane, no roadmap reshuffle. At most, keep a **future ablation note** for a DCRL-style mixed-advantage critic only if Hydra’s actor-critic RL lane becomes active and empirically unstable. **Confidence: high.**
[PREV L0006] 
[PREV L0007] The reason these diverge is simple: Hydra already has real oracle-critic and CTDE-adjacent surfaces, but its promoted method identity is centered on **ExIt + search-as-feature + Mahjong-specific belief/Hand-EV/search plumbing**, and its active roadmap explicitly says not to broaden architecture before the baseline loop is alive. ([GitHub][1])
[PREV L0008] 
[PREV L0009] ## What the repo already proves
[PREV L0010] 
[PREV L0011] **Artifact-supported repo reality:** Hydra is not merely “thinking about” oracle/search surfaces. Public docs show a live **192x34** encoder with **Group C search/belief** and **Group D Hand-EV** planes; current status marks the stronger belief tranche, Hand-EV realism, `safety_residual`, and an end-to-end **ExIt carrier** as shipped; **DeltaQ** is implemented but not default-on; and broader public-belief search remains reserve-shelf. README-level routing also foregrounds opponent modeling and inference-time search as the long-run differentiators. ([GitHub][2])
[PREV L0012] 
[PREV L0013] **Code-level proof:** the model surface already includes both a public `value` head and an `oracle_critic`, plus `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`. The oracle critic is a distinct **4-output** head, it is fed a **detached** backbone tensor, oracle-only loss is tested not to backprop through the shared backbone, and the default loss weights for oracle and advanced heads are **0.0** in the exposed loss config. That proves the oracle lane is structurally real, but also that it is currently treated as an **auxiliary / gated** surface rather than a default project identity. ([GitHub][3])
[PREV L0014] 
[PREV L0015] **Doctrine-level proof:** Hydra’s architecture doc makes two important commitments at once:
[PREV L0016] 
[PREV L0017] 1. perfect-information networks are allowed for **variance reduction and diagnostics**, but
[PREV L0018] 2. the improvement operator for the deployable policy should respect the **public/information state**.
[PREV L0019]    That is stricter than “samplewise hidden-state critic is unbiased in expectation,” and it matters for the DCRL comparison. Hydra also explicitly says the oracle critic supplies advantages via **CTDE**, while the actor conditions on public info only. ([GitHub][1])
[PREV L0020] 
[PREV L0021] ## Overlap with DCRL: real vs superficial
[PREV L0022] 
[PREV L0023] | Dimension                                                  | DCRL / adjacent prior art | Hydra                                      | Verdict                     |
[PREV L0024] | ---------------------------------------------------------- | ------------------------- | ------------------------------------------ | --------------------------- |
[PREV L0025] | Privileged critic during training                          | Yes                       | Yes                                        | **Real overlap**            |
[PREV L0026] | Actor deploys without privileged state                     | Yes                       | Yes                                        | **Real overlap**            |
[PREV L0027] | Standard partial/history critic + oracle critic            | DCRL: core mechanism      | Hydra: surfaces exist                      | **Real overlap**            |
[PREV L0028] | DCRL-style mixed actor advantage is the main update object | Yes                       | Not shown as active project identity       | **Not established overlap** |
[PREV L0029] | Public-state search / ExIt teacher object                  | No                        | Central Hydra doctrine                     | **Major divergence**        |
[PREV L0030] | Search-as-feature, belief planes, Hand-EV, DeltaQ          | No                        | Core Hydra surfaces                        | **Major divergence**        |
[PREV L0031] | Mahjong, 4-player, rank-aware objective                    | No                        | Yes                                        | **Major divergence**        |
[PREV L0032] | “dual critic” as generic naming similarity                 | N/A                       | Many heads / 2-tier nets / auxiliary lanes | **Mostly superficial**      |
[PREV L0033] 
[PREV L0034] The most important distinction is **training trust object**. DCRL’s main idea is still an actor-critic update built from a weighted combination of oracle and standard advantages. Hydra’s stated north star is different: **search-generated public-state teacher targets** are the central engine, while privileged critics are auxiliary CTDE-style tools. So the overlap is real, but narrow. ([NeurIPS Proceedings][4])
[PREV L0035] 
[PREV L0036] ## Which external sources actually move the answer
[PREV L0037] 
[PREV L0038] **1) UAAC matters more than DCRL for literature positioning.**
[PREV L0039] UAAC is the paper that established the theoretically sound **history+state oracle critic** under partial observability and explicitly contrasted it with biased state-only asymmetric critics. If Hydra mentions DCRL, it should almost certainly mention **UAAC** too, because DCRL is better read as a variance-reduced refinement of that lineage than as the root prior-art family. ([IFAAMAS][5])
[PREV L0040] 
[PREV L0041] **2) DCRL matters, but as a narrow extension.**
[PREV L0042] DCRL’s primary contribution is: keep a **standard critic** and an **oracle critic**, combine them with a weighting/gating scheme, and retain unbiasedness while reducing variance under partial observability. That is relevant to Hydra’s oracle-critic lane, but it does **not** address Hydra’s central differentiators: Mahjong-specific belief/search integration, ExIt/search targets, or 4-player rank-aware decision making. Its published evaluation is on partial-observation control benchmarks, not Mahjong or multiplayer imperfect-information search. ([NeurIPS Proceedings][4])
[PREV L0043] 
[PREV L0044] **3) Same-domain Mahjong work is still more important for Hydra than DCRL.**
[PREV L0045] Hydra’s own docs already call **RVR** “the most directly relevant work,” specifically because it is same-domain Mahjong, uses an oracle/relative-value network, enforces a zero-sum constraint, and adds an expected reward network for variance reduction. The RVR paper itself describes a relative value network using global information plus an expected reward network to stabilize Mahjong RL. That is simply more on-point for Hydra than DCRL. ([GitHub][6])
[PREV L0046] 
[PREV L0047] **4) The older asymmetric-critic root should also be acknowledged.**
[PREV L0048] The 2018 asymmetric actor-critic paper is the clean historical root for “actor on partial observations, critic on full state during training only.” If Hydra wants one umbrella sentence, the clean lineage is **AACC/AAC → UAAC → DCRL**, with **RVR/Suphx** as the Mahjong-specific branch. ([Robotics Proceedings][7])
[PREV L0049] 
[PREV L0050] **5) Hydra already foregrounds stronger adjacent families.**
[PREV L0051] Its own reference/comparison surfaces emphasize **RVR**, **ACH**, **OLSS**, **ExIt**, and **oracle distillation** as the most actionable or central ideas. The training-paradigms document literally says **ExIt** and **asymmetric oracle training** are the two most actionable paradigms for Hydra, and treats oracle distillation as already being Hydra’s Phase 2 concept. That lowers the method-level importance of DCRL further. ([GitHub][8])
[PREV L0052] 
[PREV L0053] ## Decision-ready recommendation
[PREV L0054] 
[PREV L0055] ### Docs-worth-it: **Yes, but tiny**
[PREV L0056] 
[PREV L0057] The omission is worth fixing because Hydra publicly discusses oracle critics / CTDE / partial observability, and missing the **UAAC/DCRL** branch creates an avoidable “nearby prior art not acknowledged” hole. But this is a **positioning fix**, not a re-architecture.
[PREV L0058] 
[PREV L0059] ### Smallest justified docs patch set
[PREV L0060] 
[PREV L0061] **Minimum patch**
[PREV L0062] 
[PREV L0063] 1. **`research/intel/REFERENCES.md`**
[PREV L0064]    Add entries for:
[PREV L0065] 
[PREV L0066]    * **Asymmetric Actor-Critic for Image-Based Robot Learning** (2018)
[PREV L0067]    * **Unbiased Asymmetric Reinforcement Learning under Partial Observability** (UAAC, 2022)
[PREV L0068]    * **Dual Critic Reinforcement Learning under Partial Observability** (DCRL, 2024)
[PREV L0069] 
[PREV L0070] 2. **`research/design/HYDRA_FINAL.md`**
[PREV L0071]    Add one short prior-art note near **P2** or **Phase 1 oracle supervision**:
[PREV L0072] 
[PREV L0073]    > Prior-art positioning: Hydra’s oracle-critic / CTDE surfaces are adjacent to the asymmetric-critic lineage (Pinto et al. 2018; Baisero & Amato 2022; Li et al. 2024). Hydra does not claim privileged critics as a standalone novelty. The promoted delta is the combination with Mahjong-specific ExIt/search-as-feature, belief/search surfaces, Hand-EV features, and 4-player rank-aware training.
[PREV L0074] 
[PREV L0075] 3. **Optional README sentence**
[PREV L0076]    Only if you want the public front door to inoculate against shallow critique:
[PREV L0077] 
[PREV L0078]    > Hydra’s oracle-critic lane sits in the asymmetric-critic line (AACC / UAAC / DCRL), while Hydra’s main architecture identity is public-state ExIt/search-as-feature for Mahjong.
[PREV L0079] 
[PREV L0080] **Do not** edit `HYDRA_RECONCILIATION.md` unless you also change roadmap priority. Right now, you should not.
[PREV L0081] 
[PREV L0082] ## Method-worth-it: **No active reaction**
[PREV L0083] 
[PREV L0084] Hydra’s roadmap explicitly says:
[PREV L0085] 
[PREV L0086] * baseline first,
[PREV L0087] * close loops before expanding architecture,
[PREV L0088] * search-strength lanes stay selective,
[PREV L0089] * broader identity debates should not outrank the live training loop. ([GitHub][9])
[PREV L0090] 
[PREV L0091] So DCRL does **not** justify:
[PREV L0092] 
[PREV L0093] * changing architecture priorities,
[PREV L0094] * adding a new reserve lane now,
[PREV L0095] * bumping DeltaQ/search-strength work,
[PREV L0096] * or rewriting Hydra’s method story around “dual critics.”
[PREV L0097] 
[PREV L0098] ### Exact no-op rationale
[PREV L0099] 
[PREV L0100] 1. **Hydra already tracks more direct prior art.**
[PREV L0101]    For Hydra’s actual domain and ambitions, **RVR / Suphx / ACH / OLSS / ExIt** are more direct than DCRL. ([GitHub][6])
[PREV L0102] 
[PREV L0103] 2. **Hydra’s main trust object is different.**
[PREV L0104]    DCRL’s center of gravity is a dual-critic actor-critic update. Hydra’s is a **public-state ExIt/search teacher**. DCRL can only touch a secondary lane. ([GitHub][1])
[PREV L0105] 
[PREV L0106] 3. **Current code exposes an oracle surface, but as an auxiliary, not a promoted shared training identity.**
[PREV L0107]    Detached oracle input, optional oracle loss, default zero oracle weight, and auxiliary-head gating all point to “use carefully when activated,” not “make this the project’s next big fight.” ([GitHub][3])
[PREV L0108] 
[PREV L0109] ## If someone insists on a method follow-up anyway
[PREV L0110] 
[PREV L0111] Do **one ablation**, not a new lane:
[PREV L0112] 
[PREV L0113] ```text
[PREV L0114] A_std  = delta_phi(h_t, a_t)
[PREV L0115] A_orc  = delta_psi(h_t, s_t, a_t)
[PREV L0116] beta_t = 0 if A_std <= 0 else beta0      # beta0 in {1/3, 1/2}
[PREV L0117] A_mix  = (1 - beta_t) * A_orc + beta_t * A_std
[PREV L0118] 
[PREV L0119] L_pi = - A_mix * log pi(a_t | h_t)
[PREV L0120] ```
[PREV L0121] 
[PREV L0122] Only run this **after** Hydra’s actor-critic RL loop is live. Promotion gate:
[PREV L0123] 
[PREV L0124] * lower advantage variance on matched public histories,
[PREV L0125] * no regression on duplicate/arena strength,
[PREV L0126] * no degradation to ExIt / DeltaQ label quality,
[PREV L0127] * no instability in KL drift or training loss.
[PREV L0128] 
[PREV L0129] If it fails any of those, close the experiment and keep it out of doctrine.
[PREV L0130] 
[PREV L0131] ## What would falsify this recommendation
[PREV L0132] 
[PREV L0133] Any of these would move me off the current verdict:
[PREV L0134] 
[PREV L0135] 1. **Repo evidence** that Hydra already has a DCRL-style mixed oracle/public advantage update in the active RL path, with good results.
[PREV L0136] 2. **Mahjong- or multiplayer-IIG-specific evidence** that DCRL-style mixed critics beat the current same-domain anchors (RVR/Suphx-style oracle/value or ExIt/search-guided training) in the regime Hydra cares about.
[PREV L0137] 3. **Active-path instability** in Hydra’s future RL lane where oracle-only CTDE is clearly the bottleneck and a mixed critic fixes it.
[PREV L0138] 4. **Promoted docs already updated** to include UAAC/DCRL positioning; in that case, docs-worth-it drops to no-op.
[PREV L0139] 
[PREV L0140] ## Bottom line
[PREV L0141] 
[PREV L0142] Hydra should **mention DCRL/UAAC explicitly**, but only as a **small literature-positioning patch**.
[PREV L0143] 
[PREV L0144] Hydra should **not** materially react at the method level right now.
[PREV L0145] 
[PREV L0146] The missing citation is a **docs hygiene issue**, not a sign that Hydra’s architecture priorities should move.
[PREV L0147] 
[PREV L0148] [1]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md"
[PREV L0149] [2]: https://github.com/NikkeTryHard/hydra "https://github.com/NikkeTryHard/hydra"
[PREV L0150] [3]: https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs"
[PREV L0151] [4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html"
[PREV L0152] [5]: https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf "https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf"
[PREV L0153] [6]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md"
[PREV L0154] [7]: https://www.roboticsproceedings.org/rss14/p08.pdf "https://www.roboticsproceedings.org/rss14/p08.pdf"
[PREV L0155] [8]: https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md "https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md"
[PREV L0156] [9]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md"
```

## Artifact 03 — Training crate ownership and shipped lane summary
Artifact id: `train-readme`
Source label: DOC
Type: `file_full`
Source: `crates/hydra-train/README.md`
Why it matters: Fast orientation for what hydra-train owns and which training lanes are already baseline versus staged or promotion-gated.

```markdown
[DOC L0001] # hydra-train
[DOC L0002] 
[DOC L0003] Training crate for the Hydra Riichi Mahjong AI. It owns the model stack, replay/self-play data plumbing, target construction, evaluation harnesses, and the training/data utility binaries that turn `hydra-core` encoder/runtime signals into checkpoints or replay-side artifacts.
[DOC L0004] 
[DOC L0005] ## Overview
[DOC L0006] 
[DOC L0007] `hydra-train` is the workspace layer that sits above `hydra-core` and `hydra-engine`.
[DOC L0008] 
[DOC L0009] - `hydra-engine` owns low-level Riichi rules and replay parsing
[DOC L0010] - `hydra-core` owns runtime bridging, encoding, simulation, seeding, and search/runtime feature plumbing
[DOC L0011] - `hydra-train` owns model definition, losses, BC/RL/self-play orchestration, sidecar generation, and training/evaluation utilities
[DOC L0012] 
[DOC L0013] The crate is built around Burn and the current Hydra training baseline. The shipped baseline already includes the live `192x34` encoder/model contract, replay-derived `safety_residual`, the stronger public-teacher belief semantics tranche, and the ExIt carrier across both live self-play and replay/sample sidecar-first lanes. Promotion-gated DeltaQ tooling also lives here, but it is not the default-on training lane.
[DOC L0014] 
[DOC L0015] For current shipped-vs-staged status, read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
[DOC L0016] For active-path sequencing, read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).
[DOC L0017] 
[DOC L0018] ## What this crate owns
[DOC L0019] 
[DOC L0020] `hydra-train` is responsible for:
[DOC L0021] 
[DOC L0022] - model/backbone/head definitions
[DOC L0023] - BC and RL optimization loops
[DOC L0024] - replay data loading and sample collation
[DOC L0025] - self-play batch generation and evaluation harnesses
[DOC L0026] - preflight/runtime autotuning and resume compatibility checks
[DOC L0027] - replay sidecar generation for ExIt and DeltaQ-style lanes
[DOC L0028] - workspace binaries and utilities like `train`, `mjai_audit`, `recompress`, `repack_tar`, replay sidecar builders, and replay failure inspection tools
[DOC L0029] 
[DOC L0030] It does **not** own the Riichi rules engine itself. When rule semantics drift, `hydra-engine` and `docs/GAME_ENGINE.md` are the runtime authority.
[DOC L0031] 
[DOC L0032] ## Module Reference
[DOC L0033] 
[DOC L0034] | Module | Description |
[DOC L0035] |--------|-------------|
[DOC L0036] | `backbone` | Backbone building blocks for Hydra's network stack |
[DOC L0037] | `config` | Shared training/runtime config types and parsing helpers |
[DOC L0038] | `data` | Replay loading, data-source scanning, augmentation, and batch/sample plumbing |
[DOC L0039] | `eval` | Arena/evaluation helpers and training/eval metric summaries |
[DOC L0040] | `heads` | Policy / value / auxiliary head definitions |
[DOC L0041] | `inference` | Train-side model inference helpers |
[DOC L0042] | `league` | League-style model coordination and related utilities |
[DOC L0043] | `model` | Top-level `HydraModel` assembly and config surface |
[DOC L0044] | `preflight` | Probe/preflight configuration for runtime selection and autotune flows |
[DOC L0045] | `saf` | SAF-related train-side helpers |
[DOC L0046] | `selfplay` | Self-play orchestration and mixed-policy game execution |
[DOC L0047] | `selfplay_batch` | Batched self-play data plumbing |
[DOC L0048] | `teacher` | Teacher-side feature/label helpers, including belief surfaces |
[DOC L0049] | `training` | BC, RL, ACH, DRDA, ExIt, DeltaQ promotion/validation, losses, gates, and orchestrators |
[DOC L0050] 
[DOC L0051] ## Workspace binaries
[DOC L0052] 
[DOC L0053] The crate currently exposes these workspace binaries:
[DOC L0054] 
[DOC L0055] | Binary | Purpose |
[DOC L0056] |--------|---------|
[DOC L0057] | `train` | Main training entrypoint; supports normal training, preflight, probe, and DeltaQ-promotion modes |
[DOC L0058] | `mjai_audit` | Audits replay datasets and archives, including failure bucketing and optional failure inventories |
[DOC L0059] | `recompress` | Recompression utility for replay/data artifacts |
[DOC L0060] | `repack_tar` | Repack utility for tar-based replay corpora |
[DOC L0061] | `build_replay_delta_q_sidecar` | Builds replay-side DeltaQ sidecars |
[DOC L0062] | `build_replay_exit_sidecar` | Builds replay-side ExIt sidecars |
[DOC L0063] | `mjai_debug_failure` | Debug helper for replay failures |
[DOC L0064] | `mjai_first_failure` | Finds/inspects the first replay failure in a dataset |
[DOC L0065] 
[DOC L0066] The main training entrypoint lives at [`src/bin/train.rs`](src/bin/train.rs).
[DOC L0067] 
[DOC L0068] ## Runtime and data contract
[DOC L0069] 
[DOC L0070] The training crate consumes the same live runtime surface as the rest of Hydra:
[DOC L0071] 
[DOC L0072] - encoder/model contract: `192x34`
[DOC L0073] - action space: 46 actions
[DOC L0074] - replay input support: flat MJAI directories plus `.tar.zst` archives
[DOC L0075] - default workspace test path: `cargo nextest run --release`
[DOC L0076] 
[DOC L0077] The Docker/container-facing training contract is documented in [`docker/train/README.md`](../../docker/train/README.md).
[DOC L0078] 
[DOC L0079] ## Training flow at a glance
[DOC L0080] 
[DOC L0081] At a high level, `hydra-train` does four things:
[DOC L0082] 
[DOC L0083] 1. reads config and chooses runtime/preflight behavior
[DOC L0084] 2. loads replay or self-play data through `data::*`
[DOC L0085] 3. builds targets/losses and runs BC/RL/update loops through `training::*`
[DOC L0086] 4. evaluates, checkpoints, and reports metrics through the train binary and eval helpers
[DOC L0087] 
[DOC L0088] That split is intentional: runtime semantics stay below this crate, while optimization policy and target construction stay here.
[DOC L0089] 
[DOC L0090] ## Where to read next
[DOC L0091] 
[DOC L0092] - Need runtime truth? Read [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md) and [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
[DOC L0093] - Need current shipped/staged status? Read [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md).
[DOC L0094] - Need the active Hydra v1 roadmap? Read [`research/design/HYDRA_RECONCILIATION.md`](../../research/design/HYDRA_RECONCILIATION.md).
[DOC L0095] - Need container execution details? Read [`docker/train/README.md`](../../docker/train/README.md).
[DOC L0096] 
[DOC L0097] ## License
[DOC L0098] 
[DOC L0099] Business Source License 1.1 (BSL). See the repo-root [LICENSE](../../LICENSE).
[DOC L0100] 
[DOC L0101] - Free for personal, non-commercial, and academic use
[DOC L0102] - Commercial mahjong AI services require a paid license from the Licensor
[DOC L0103] - Converts to Apache-2.0 on 2031-03-02
[DOC L0104] 
[DOC L0105] For commercial licensing inquiries, contact Sho Kaneko.
```

## Artifact 04 — Current shipped vs staged status
Artifact id: `current-status`
Source label: DOC
Type: `file_full`
Source: `docs/CURRENT_STATUS.md`
Why it matters: Needed so the blueprint can say whether a DCRL-style lane belongs in baseline, implemented-but-not-default-on, staged, or reserve shelf.

```markdown
[DOC L0001] # Hydra Current Status
[DOC L0002] 
[DOC L0003] Current shipped/staged status for Hydra's already-built surfaces.
[DOC L0004] 
[DOC L0005] This file is Hydra's promoted current-status snapshot for things that already exist in code or are partially implemented in code. Use it to answer questions like "what is shipped today?", "what is implemented but still staged?", and "what is implemented but not default-on yet?"
[DOC L0006] 
[DOC L0007] This file reports shipped/staged status only.
[DOC L0008] 
[DOC L0009] - For the roadmap to Hydra v1, read `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0010] - For runtime semantics and compatibility truth, read `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.
[DOC L0011] 
[DOC L0012] When this file and current code disagree, current code wins. When this file and `HYDRA_RECONCILIATION.md` disagree on active vs reserve vs staged priority, refresh reconciliation and then refresh this file. When reconciliation or current status drift from the archive root, refresh the promoted docs rather than demoting the canonical archive source ledger.
[DOC L0013] 
[DOC L0014] ## Status vocabulary
[DOC L0015] 
[DOC L0016] This file uses the status vocabulary defined in `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0017] 
[DOC L0018] | Term | Meaning |
[DOC L0019] |---|---|
[DOC L0020] | `shipped baseline` | implemented and part of the current live baseline |
[DOC L0021] | `implemented but not default-on` | implemented and validated enough to exist in-code, but intentionally not the default runtime/training path |
[DOC L0022] | `implemented but staged` | core code path exists, but promotion/activation is still intentionally deferred |
[DOC L0023] | `reserve shelf` | documented later-work direction, not current mainline priority |
[DOC L0024] | `historical` | preserved context only; not current governing truth |
[DOC L0025] 
[DOC L0026] ## Runtime and training snapshot
[DOC L0027] 
[DOC L0028] ### Shipped baseline
[DOC L0029] 
[DOC L0030] - `hydra-core` is a real first-party runtime/encoder/simulator crate.
[DOC L0031] - The live encoder/model contract is `192x34`; the old `85x34` view is baseline-prefix only.
[DOC L0032] - The fixed runtime action space is 46 actions with two-phase riichi and kan handling.
[DOC L0033] - BC training now supports **epoch-boundary-only** reuse of matching preflight-selected runtime for the selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, derived `accum_steps`), while fresh runs remain config-derived, partial-epoch resumes still require identical runtime, and loader-runtime stays config-derived.
[DOC L0034] - The stronger public-teacher belief-semantics tranche is shipped as part of the current training baseline.
[DOC L0035] - The current Hand-EV realism upgrade is shipped as part of the live baseline surface.
[DOC L0036] - Replay-derived `safety_residual` is shipped as a narrow supervised lane.
[DOC L0037] - ExIt now has an end-to-end carrier across the live self-play lane and the replay/sample sidecar-first lane.
[DOC L0038] 
[DOC L0039] ### Implemented but not default-on
[DOC L0040] 
[DOC L0041] - The narrow DeltaQ supervision lane is implemented in code and promotion-gated through an arena-confirmation path.
[DOC L0042] - DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but the lane is still **not** default-on.
[DOC L0043] 
[DOC L0044] ### Implemented but staged
[DOC L0045] 
[DOC L0046] - `mixture_weight` promotion remains staged.
[DOC L0047] - Richer opponent-target closure remains staged.
[DOC L0048] - Representative-world / per-particle CT-SMC Hand-EV remains staged.
[DOC L0049] - Selective AFBS / endgame deepening remains staged.
[DOC L0050] 
[DOC L0051] ### Reserve shelf
[DOC L0052] 
[DOC L0053] - Broader public-belief search as project identity remains reserve-shelf, not active-path.
[DOC L0054] - Deeper robust-opponent search backups remain reserve-shelf.
[DOC L0055] - Larger latent-opponent / richer auxiliary-head expansion remains reserve-shelf until existing target closure improves.
[DOC L0056] 
[DOC L0057] ## Area-by-area summary
[DOC L0058] 
[DOC L0059] | Area | Current status | Notes |
[DOC L0060] |---|---|---|
[DOC L0061] | Runtime encoder / action semantics | shipped baseline | See `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` |
[DOC L0062] | Hand-EV baseline surface | shipped baseline | Stronger local evaluator is live; representative-world CT-SMC Hand-EV remains staged |
[DOC L0063] | Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche is in the live baseline |
[DOC L0064] | BC runtime authority | shipped baseline | Fresh runs are config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
[DOC L0065] | `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
[DOC L0066] | ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
[DOC L0067] | DeltaQ lane | implemented but not default-on | Arena-confirmation path implemented; promotion artifact now records pre-arena recommendation plus final `arena_decision`/`arena_report` |
[DOC L0068] | `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
[DOC L0069] | `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
[DOC L0070] | AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |
[DOC L0071] 
[DOC L0072] ## Where to read next
[DOC L0073] 
[DOC L0074] - Need the current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
[DOC L0075] - Need the roadmap to Hydra v1 or the active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
[DOC L0076] - Need the north-star architecture rather than current shipped status? Read `research/design/HYDRA_FINAL.md`.
```

## Artifact 05 — Historical head layout and activation notes
Artifact id: `implementation-roadmap-step3`
Source label: ROADMAP
Type: `file_range`
Source: `research/design/IMPLEMENTATION_ROADMAP.md:166-244`
Why it matters: Useful compressed reference for the current head set, including oracle, delta_q, and advanced-head staging notes.

```markdown
[ROADMAP L0166] ## Step 3: Output Heads
[ROADMAP L0167] 
[ROADMAP L0168] **Ref: HYDRA_FINAL Section 4.3**
[ROADMAP L0169] 
[ROADMAP L0170] All code goes in `hydra-train/src/heads.rs`.
[ROADMAP L0171] 
[ROADMAP L0172] Current code reality is broader than the original 9-head bootstrap plan. The live model exposes the original 9-output core (policy/value/score/tenpai/GRP/opp-next/danger/oracle) plus five advanced outputs already present structurally in `model.rs`: `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`. They still share the same backbone and remain staged at the data/loss level rather than all being equally active today.
[ROADMAP L0173] 
[ROADMAP L0174] ### 3.1 Head Summary Table
[ROADMAP L0175] 
[ROADMAP L0176] | # | Name | Input | Layer(s) | Output shape | Activation |
[ROADMAP L0177] |---|------|-------|----------|--------------|------------|
[ROADMAP L0178] | 1 | PolicyHead | pooled [B,256] | Linear(256,46) | [B, 46] | none (raw logits) |
[ROADMAP L0179] | 2 | ValueHead | pooled [B,256] | Linear(256,1) | [B, 1] | tanh |
[ROADMAP L0180] | 3 | ScorePdfHead | pooled [B,256] | Linear(256,64) | [B, 64] | log_softmax (at loss time) |
[ROADMAP L0181] | 4 | ScoreCdfHead | pooled [B,256] | Linear(256,64) | [B, 64] | sigmoid |
[ROADMAP L0182] | 5 | OppTenpaiHead | pooled [B,256] | Linear(256,3) | [B, 3] | sigmoid |
[ROADMAP L0183] | 6 | GrpHead | pooled [B,256] | Linear(256,24) | [B, 24] | none (raw logits) |
[ROADMAP L0184] | 7 | OppNextDiscardHead | spatial [B,256,34] | Conv1d(256,3,1) | [B, 3, 34] | none (raw logits) |
[ROADMAP L0185] | 8 | DangerHead | spatial [B,256,34] | Conv1d(256,3,1) | [B, 3, 34] | sigmoid |
[ROADMAP L0186] | 9 | OracleCriticHead | pooled [B,256] | Linear(256,4) | [B, 4] | none (raw) |
[ROADMAP L0187] | 10 | BeliefFieldHead | spatial [B,256,34] | Conv1d(256,16,1) | [B, 16, 34] | none (raw) |
[ROADMAP L0188] | 11 | MixtureWeightHead | pooled [B,256] | Linear(256,4) | [B, 4] | none (raw logits) |
[ROADMAP L0189] | 12 | OpponentHandTypeHead | pooled [B,256] | Linear(256,24) | [B, 24] | none (raw logits) |
[ROADMAP L0190] | 13 | DeltaQHead | pooled [B,256] | Linear(256,46) | [B, 46] | none (raw) |
[ROADMAP L0191] | 14 | SafetyResidualHead | pooled [B,256] | Linear(256,46) | [B, 46] | none (raw) |
[ROADMAP L0192] 
[ROADMAP L0193] Current activation note: the advanced heads above are structurally live, and the current shipped baseline now includes the stronger public-teacher belief-semantics tranche across the belief carrier / loss / presence path. In the normal supervised path, `belief_fields`, `mixture_weight`, and `safety_residual` already have concrete carriers, replay/sample ExIt now flows through a separate sidecar-backed carrier path, replay/offline `delta_q` now also flows through a parallel sidecar-backed carrier path plus BC/train activation-hook closure, while `opponent_hand_type_target` still remains absent in the standard sample-to-target conversion. `mixture_weight` promotion remains staged and should not be inferred from the stronger shipped belief carrier semantics.
[ROADMAP L0194] 
[ROADMAP L0195] Current DeltaQ status note: the repo has moved past plain structural DeltaQ validation. The current shipped DeltaQ follow-up includes an offline promotion stack with teacher-regret metrics, paired holdout policy-transfer reporting, a dedicated `--delta-q-promotion` train mode, persisted promotion artifacts, a paired arena executor, and paired-arena helper/config/report objects under matched seeds / fixed seat rotation / matched temperature / frozen baseline opponents. Promotion artifacts now keep the pre-arena `recommendation` as the stage-aware next-step signal and persist the final arena proof separately as `arena_decision` plus `arena_report`.
[ROADMAP L0196] 
[ROADMAP L0197] ### 3.2 Struct Definitions
[ROADMAP L0198] 
[ROADMAP L0199] Each head is a `#[derive(Module, Debug)]` struct with a single `linear: Linear<B>` or `conv: Conv1d<B>` field. Forward methods are trivial pass-through with activation:
[ROADMAP L0200] - **PolicyHead**: `forward(pooled: [B,256]) -> [B,46]` -- linear only, no activation
[ROADMAP L0201] - **ValueHead**: `forward(pooled: [B,256]) -> [B,1]` -- linear -> tanh
[ROADMAP L0202] - **ScorePdfHead**: `forward(pooled: [B,256]) -> [B,64]` -- linear only (log_softmax at loss time)
[ROADMAP L0203] - **ScoreCdfHead**: `forward(pooled: [B,256]) -> [B,64]` -- linear only (sigmoid via bce_with_logits at loss time)
[ROADMAP L0204] - **OppTenpaiHead**: `forward(pooled: [B,256]) -> [B,3]` -- linear only (sigmoid via bce_with_logits at loss time)
[ROADMAP L0205] - **GrpHead**: `forward(pooled: [B,256]) -> [B,24]` -- linear only
[ROADMAP L0206] - **OppNextDiscardHead**: `forward(spatial: [B,256,34]) -> [B,3,34]` -- conv1d(k=1) only
[ROADMAP L0207] - **DangerHead**: `forward(spatial: [B,256,34]) -> [B,3,34]` -- conv1d(k=1) only (sigmoid via bce_with_logits at loss time)
[ROADMAP L0208] - **OracleCriticHead**: `forward(pooled: [B,256]) -> [B,4]` -- linear only (4 values, one per player, zero-sum normalized in loss)
[ROADMAP L0209] 
[ROADMAP L0210] ### 3.3 Config and Init
[ROADMAP L0211] 
[ROADMAP L0212] **HeadsConfig** (`#[derive(Config, Debug)]`):
[ROADMAP L0213] - Fields: `hidden_channels: usize` (256), `action_space: usize` (46), `score_bins: usize` (64), `num_opponents: usize` (3), `grp_classes: usize` (24)
[ROADMAP L0214] - Provide `init_*` methods for each head (e.g. `init_policy`, `init_value`, etc.)
[ROADMAP L0215] - For Conv1d heads (OppNextDiscard, Danger), kernel_size=1, NO padding needed.
[ROADMAP L0216] 
[ROADMAP L0217] ### 3.4 MUST NOT
[ROADMAP L0218] 
[ROADMAP L0219] - Do NOT apply softmax to PolicyHead output. That happens at loss/sampling time.
[ROADMAP L0220] - Do NOT apply softmax to GrpHead output. Same reason.
[ROADMAP L0221] - Do NOT apply log_softmax to ScorePdfHead in forward. Only at loss time.
[ROADMAP L0222] - Do NOT use kernel_size=3 for OppNextDiscard/Danger. Use kernel_size=1 (pointwise).
[ROADMAP L0223] - Do NOT share any weights between heads. Each is independent.
[ROADMAP L0224] - Do NOT add hidden layers inside heads. Each head is a SINGLE Linear or Conv1d.
[ROADMAP L0225] - Do NOT apply activation to OracleCriticHead. It's a raw value estimate.
[ROADMAP L0226] - Value head uses `tanh`, NOT `sigmoid`. Value range is [-1, 1] not [0, 1].
[ROADMAP L0227] 
[ROADMAP L0228] ### 3.5 Tests for Step 3
[ROADMAP L0229] 
[ROADMAP L0230] Use `burn::backend::NdArray`. Create a `default_config() -> HeadsConfig` helper.
[ROADMAP L0231] 
[ROADMAP L0232] | Test | Assertion |
[ROADMAP L0233] |------|-----------|
[ROADMAP L0234] | `policy_head_shape` | [4,256] -> [4,46] |
[ROADMAP L0235] | `value_head_shape_and_range` | [4,256] -> [4,1], all values in [-1, 1] |
[ROADMAP L0236] | `score_pdf_head_shape` | [4,256] -> [4,64] |
[ROADMAP L0237] | `score_cdf_head_range` | [4,256] -> [4,64] |
[ROADMAP L0238] | `opp_tenpai_head_shape` | [4,256] -> [4,3] |
[ROADMAP L0239] | `grp_head_shape` | [4,256] -> [4,24] |
[ROADMAP L0240] | `opp_next_discard_head_shape` | [4,256,34] -> [4,3,34] |
[ROADMAP L0241] | `danger_head_shape_and_range` | [4,256,34] -> [4,3,34] |
[ROADMAP L0242] | `oracle_critic_head_shape` | [4,256] -> [4,4] |
[ROADMAP L0243] 
[ROADMAP L0244] ---
```

## Artifact 06 — RL update composition and auxiliary-loss mixing
Artifact id: `rl-loop`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/training/rl.rs:141-400`
Why it matters: Primary code surface for where a DCRL-style mixed-advantage update could actually land in Hydra's RL loop.

```rust
[CODE L0141] pub struct RlConfig {
[CODE L0142]     pub tau_drda: f32,
[CODE L0143]     pub ach_cfg: AchConfig,
[CODE L0144]     pub lr: f64,
[CODE L0145]     pub exit_weight: f32,
[CODE L0146]     pub aux_weight: f32,
[CODE L0147]     pub microbatch_size: Option<usize>,
[CODE L0148] }
[CODE L0149] 
[CODE L0150] impl RlConfig {
[CODE L0151]     pub fn default_phase2() -> Self {
[CODE L0152]         Self {
[CODE L0153]             tau_drda: 4.0,
[CODE L0154]             ach_cfg: AchConfig::new(),
[CODE L0155]             lr: 2.5e-4,
[CODE L0156]             exit_weight: DEFAULT_EXIT_WEIGHT,
[CODE L0157]             aux_weight: 0.1,
[CODE L0158]             microbatch_size: None,
[CODE L0159]         }
[CODE L0160]     }
[CODE L0161] 
[CODE L0162]     pub fn with_lr(mut self, lr: f64) -> Self {
[CODE L0163]         self.lr = lr;
[CODE L0164]         self
[CODE L0165]     }
[CODE L0166]     pub fn with_exit_weight(mut self, w: f32) -> Self {
[CODE L0167]         self.exit_weight = w;
[CODE L0168]         self
[CODE L0169]     }
[CODE L0170]     pub fn with_aux_weight(mut self, w: f32) -> Self {
[CODE L0171]         self.aux_weight = w;
[CODE L0172]         self
[CODE L0173]     }
[CODE L0174] 
[CODE L0175]     pub fn default_phase3() -> Self {
[CODE L0176]         Self {
[CODE L0177]             tau_drda: 4.0,
[CODE L0178]             ach_cfg: AchConfig::new(),
[CODE L0179]             lr: 1e-4,
[CODE L0180]             exit_weight: 0.5,
[CODE L0181]             aux_weight: 0.1,
[CODE L0182]             microbatch_size: None,
[CODE L0183]         }
[CODE L0184]     }
[CODE L0185] 
[CODE L0186]     pub fn summary(&self) -> String {
[CODE L0187]         format!(
[CODE L0188]             "rl(tau={:.1}, lr={:.1e}, exit_w={:.2}, aux_w={:.2})",
[CODE L0189]             self.tau_drda, self.lr, self.exit_weight, self.aux_weight
[CODE L0190]         )
[CODE L0191]     }
[CODE L0192] 
[CODE L0193]     pub fn effective_exit_weight(&self, phase: u8, progress: f32) -> f32 {
[CODE L0194]         crate::training::exit::anneal_exit_weight(self.exit_weight, phase, progress)
[CODE L0195]     }
[CODE L0196] 
[CODE L0197]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0198]         if self.tau_drda < crate::training::drda::MIN_TAU_DRDA {
[CODE L0199]             return Err("tau_drda below minimum");
[CODE L0200]         }
[CODE L0201]         self.ach_cfg.validate()?;
[CODE L0202]         if self.lr <= 0.0 {
[CODE L0203]             return Err("lr must be positive");
[CODE L0204]         }
[CODE L0205]         Ok(())
[CODE L0206]     }
[CODE L0207] }
[CODE L0208] 
[CODE L0209] pub fn rl_step<B: AutodiffBackend>(
[CODE L0210]     model: HydraModel<B>,
[CODE L0211]     batch: &RlBatch<B>,
[CODE L0212]     cfg: &RlConfig,
[CODE L0213]     loss_fn: &HydraLoss<B>,
[CODE L0214]     optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
[CODE L0215] ) -> (HydraModel<B>, f64) {
[CODE L0216]     rl_step_with_phase_progress_and_controller(model, batch, cfg, 3, 1.0, loss_fn, optimizer, None)
[CODE L0217] }
[CODE L0218] 
[CODE L0219] pub fn rl_step_with_phase_progress<B: AutodiffBackend>(
[CODE L0220]     model: HydraModel<B>,
[CODE L0221]     batch: &RlBatch<B>,
[CODE L0222]     cfg: &RlConfig,
[CODE L0223]     phase: u8,
[CODE L0224]     progress: f32,
[CODE L0225]     loss_fn: &HydraLoss<B>,
[CODE L0226]     optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
[CODE L0227] ) -> (HydraModel<B>, f64) {
[CODE L0228]     rl_step_with_phase_progress_and_controller(
[CODE L0229]         model, batch, cfg, phase, progress, loss_fn, optimizer, None,
[CODE L0230]     )
[CODE L0231] }
[CODE L0232] 
[CODE L0233] /// Single RL training step with gradient accumulation across microbatches.
[CODE L0234] ///
[CODE L0235] /// Advantages are normalized over the full batch before splitting so the
[CODE L0236] /// statistics match the non-microbatched path exactly. Each microbatch
[CODE L0237] /// runs forward+backward independently; gradients are accumulated and
[CODE L0238] /// applied in one optimizer step at the end.
[CODE L0239] #[allow(clippy::too_many_arguments)]
[CODE L0240] pub fn rl_step_with_phase_progress_and_controller<B: AutodiffBackend>(
[CODE L0241]     model: HydraModel<B>,
[CODE L0242]     batch: &RlBatch<B>,
[CODE L0243]     cfg: &RlConfig,
[CODE L0244]     phase: u8,
[CODE L0245]     progress: f32,
[CODE L0246]     loss_fn: &HydraLoss<B>,
[CODE L0247]     optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
[CODE L0248]     mut controller: Option<&mut HeadActivationController>,
[CODE L0249] ) -> (HydraModel<B>, f64) {
[CODE L0250]     let effective_loss_fn;
[CODE L0251]     let active_loss_fn = if let Some(ctrl) = controller.as_mut() {
[CODE L0252]         let (effective_cfg, _) = apply_head_gating_to_batch(ctrl, &loss_fn.config, &batch.targets)
[CODE L0253]             .expect("validated optional targets before RL step");
[CODE L0254]         effective_loss_fn = HydraLoss::<B>::new(effective_cfg);
[CODE L0255]         &effective_loss_fn
[CODE L0256]     } else {
[CODE L0257]         loss_fn
[CODE L0258]     };
[CODE L0259] 
[CODE L0260]     // Normalize advantages over the full batch before splitting.
[CODE L0261]     let adv = batch.advantages.clone();
[CODE L0262]     let adv_mean = adv.clone().mean();
[CODE L0263]     let adv_var = (adv.clone() - adv_mean.clone()).powf_scalar(2.0).mean();
[CODE L0264]     let adv_std = (adv_var + 1e-8).sqrt();
[CODE L0265]     let advantages_normed = (adv - adv_mean) / adv_std;
[CODE L0266] 
[CODE L0267]     let total_samples = batch.batch_size();
[CODE L0268]     let mb_size = cfg.microbatch_size.unwrap_or(total_samples);
[CODE L0269] 
[CODE L0270]     if mb_size >= total_samples {
[CODE L0271]         // Fast path: entire batch fits in one shot (original behavior).
[CODE L0272]         return rl_microbatch_forward(
[CODE L0273]             model,
[CODE L0274]             batch,
[CODE L0275]             &advantages_normed,
[CODE L0276]             0,
[CODE L0277]             total_samples,
[CODE L0278]             cfg,
[CODE L0279]             phase,
[CODE L0280]             progress,
[CODE L0281]             active_loss_fn,
[CODE L0282]             optimizer,
[CODE L0283]         );
[CODE L0284]     }
[CODE L0285] 
[CODE L0286]     // Microbatch accumulation path.
[CODE L0287]     let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
[CODE L0288]     let mut num_chunks = 0usize;
[CODE L0289]     let mut m = model;
[CODE L0290]     let device = batch.obs.device();
[CODE L0291]     let mut total_loss_tensor = Tensor::<B, 1>::zeros([1], &device);
[CODE L0292] 
[CODE L0293]     let mut start = 0;
[CODE L0294]     while start < total_samples {
[CODE L0295]         let end = (start + mb_size).min(total_samples);
[CODE L0296]         let mb_batch = batch.slice(start, end);
[CODE L0297]         #[allow(clippy::single_range_in_vec_init)]
[CODE L0298]         let mb_adv = advantages_normed.clone().slice([start..end]);
[CODE L0299] 
[CODE L0300]         let output = m.forward_active(mb_batch.obs.clone(), &active_loss_fn.config);
[CODE L0301]         let combined = drda::combined_logits(
[CODE L0302]             mb_batch.base_logits.clone(),
[CODE L0303]             output.policy_logits.clone(),
[CODE L0304]             cfg.tau_drda,
[CODE L0305]         );
[CODE L0306]         let ach_loss = ach_policy_loss(
[CODE L0307]             combined,
[CODE L0308]             mb_batch.targets.legal_mask.clone(),
[CODE L0309]             mb_batch.actions.clone(),
[CODE L0310]             mb_batch.pi_old.clone(),
[CODE L0311]             mb_adv,
[CODE L0312]             &cfg.ach_cfg,
[CODE L0313]         );
[CODE L0314]         let aux = active_loss_fn.total_loss(&output, &mb_batch.targets);
[CODE L0315]         let mut chunk_total = ach_loss + aux.total * cfg.aux_weight;
[CODE L0316]         if let (Some(exit_target), Some(exit_mask)) = (&mb_batch.exit_target, &mb_batch.exit_mask) {
[CODE L0317]             let exit_weight = cfg.effective_exit_weight(phase, progress);
[CODE L0318]             let exit_loss = crate::training::exit::exit_loss(
[CODE L0319]                 output.policy_logits,
[CODE L0320]                 exit_target.clone(),
[CODE L0321]                 exit_mask.clone(),
[CODE L0322]                 exit_weight,
[CODE L0323]             );
[CODE L0324]             chunk_total = chunk_total + exit_loss;
[CODE L0325]         }
[CODE L0326] 
[CODE L0327]         total_loss_tensor = total_loss_tensor + chunk_total.clone().detach();
[CODE L0328]         let grads = chunk_total.backward();
[CODE L0329]         let grads = GradientsParams::from_grads(grads, &m);
[CODE L0330]         accumulator.accumulate(&m, grads);
[CODE L0331]         num_chunks += 1;
[CODE L0332]         start = end;
[CODE L0333]     }
[CODE L0334] 
[CODE L0335]     let grads = accumulator.grads();
[CODE L0336]     m = optimizer.step(cfg.lr, m, grads);
[CODE L0337]     let avg_loss = if num_chunks > 0 {
[CODE L0338]         total_loss_tensor
[CODE L0339]             .into_data()
[CODE L0340]             .convert::<f64>()
[CODE L0341]             .as_slice::<f64>()
[CODE L0342]             .expect("rl aggregated loss should be readable as f64")[0]
[CODE L0343]             / num_chunks as f64
[CODE L0344]     } else {
[CODE L0345]         0.0
[CODE L0346]     };
[CODE L0347]     (m, avg_loss)
[CODE L0348] }
[CODE L0349] 
[CODE L0350] /// Run a single (micro)batch forward+backward+step. Used for the fast
[CODE L0351] /// path when the entire batch fits in VRAM without splitting.
[CODE L0352] #[allow(clippy::too_many_arguments)]
[CODE L0353] fn rl_microbatch_forward<B: AutodiffBackend>(
[CODE L0354]     model: HydraModel<B>,
[CODE L0355]     batch: &RlBatch<B>,
[CODE L0356]     advantages_normed: &Tensor<B, 1>,
[CODE L0357]     _start: usize,
[CODE L0358]     _end: usize,
[CODE L0359]     cfg: &RlConfig,
[CODE L0360]     phase: u8,
[CODE L0361]     progress: f32,
[CODE L0362]     loss_fn: &HydraLoss<B>,
[CODE L0363]     optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
[CODE L0364] ) -> (HydraModel<B>, f64) {
[CODE L0365]     let output = model.forward_active(batch.obs.clone(), &loss_fn.config);
[CODE L0366]     let combined = drda::combined_logits(
[CODE L0367]         batch.base_logits.clone(),
[CODE L0368]         output.policy_logits.clone(),
[CODE L0369]         cfg.tau_drda,
[CODE L0370]     );
[CODE L0371]     let ach_loss = ach_policy_loss(
[CODE L0372]         combined,
[CODE L0373]         batch.targets.legal_mask.clone(),
[CODE L0374]         batch.actions.clone(),
[CODE L0375]         batch.pi_old.clone(),
[CODE L0376]         advantages_normed.clone(),
[CODE L0377]         &cfg.ach_cfg,
[CODE L0378]     );
[CODE L0379]     let aux = loss_fn.total_loss(&output, &batch.targets);
[CODE L0380]     let mut total = ach_loss + aux.total * cfg.aux_weight;
[CODE L0381]     if let (Some(exit_target), Some(exit_mask)) = (&batch.exit_target, &batch.exit_mask) {
[CODE L0382]         let exit_weight = cfg.effective_exit_weight(phase, progress);
[CODE L0383]         let exit_loss = crate::training::exit::exit_loss(
[CODE L0384]             output.policy_logits,
[CODE L0385]             exit_target.clone(),
[CODE L0386]             exit_mask.clone(),
[CODE L0387]             exit_weight,
[CODE L0388]         );
[CODE L0389]         total = total + exit_loss;
[CODE L0390]     }
[CODE L0391]     let loss_val = total
[CODE L0392]         .clone()
[CODE L0393]         .into_data()
[CODE L0394]         .convert::<f64>()
[CODE L0395]         .as_slice::<f64>()
[CODE L0396]         .expect("rl microbatch total loss should be readable as f64")[0];
[CODE L0397]     let grads = total.backward();
[CODE L0398]     let grads = GradientsParams::from_grads(grads, &model);
[CODE L0399]     let model = optimizer.step(cfg.lr, model, grads);
[CODE L0400]     (model, loss_val)
```

## Artifact 07 — Target surfaces and loss weights
Artifact id: `loss-config`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/training/losses.rs:8-193`
Why it matters: Shows what targets and loss knobs already exist, including oracle critic and advanced-head weights defaulting to zero.

```rust
[CODE L0008] #[derive(Clone)]
[CODE L0009] pub struct HydraTargets<B: Backend> {
[CODE L0010]     pub policy_target: Tensor<B, 2>,
[CODE L0011]     pub legal_mask: Tensor<B, 2>,
[CODE L0012]     pub value_target: Tensor<B, 1>,
[CODE L0013]     pub grp_target: Tensor<B, 2>,
[CODE L0014]     pub tenpai_target: Tensor<B, 2>,
[CODE L0015]     pub danger_target: Tensor<B, 3>,
[CODE L0016]     pub danger_mask: Tensor<B, 3>,
[CODE L0017]     pub opp_next_target: Tensor<B, 3>,
[CODE L0018]     pub score_pdf_target: Tensor<B, 2>,
[CODE L0019]     pub score_cdf_target: Tensor<B, 2>,
[CODE L0020]     pub oracle_target: Option<Tensor<B, 2>>,
[CODE L0021]     pub belief_fields_target: Option<Tensor<B, 3>>,
[CODE L0022]     pub belief_fields_mask: Option<Tensor<B, 1>>,
[CODE L0023]     pub mixture_weight_target: Option<Tensor<B, 2>>,
[CODE L0024]     pub mixture_weight_mask: Option<Tensor<B, 1>>,
[CODE L0025]     pub opponent_hand_type_target: Option<Tensor<B, 2>>,
[CODE L0026]     pub delta_q_target: Option<Tensor<B, 2>>,
[CODE L0027]     pub delta_q_mask: Option<Tensor<B, 2>>,
[CODE L0028]     pub safety_residual_target: Option<Tensor<B, 2>>,
[CODE L0029]     pub safety_residual_mask: Option<Tensor<B, 2>>,
[CODE L0030]     pub oracle_guidance_mask: Option<Tensor<B, 1>>,
[CODE L0031]     pub target_presence: Option<TargetPresence>,
[CODE L0032] }
[CODE L0033] 
[CODE L0034] impl<B: Backend> HydraTargets<B> {
[CODE L0035]     /// Slice all target tensors along the batch dimension (dim 0).
[CODE L0036]     ///
[CODE L0037]     /// Produces a sub-batch covering `[start..end)`. Used by microbatch
[CODE L0038]     /// accumulation to split a full RL batch into VRAM-friendly chunks.
[CODE L0039]     #[allow(clippy::single_range_in_vec_init)]
[CODE L0040]     pub fn slice_batch(&self, start: usize, end: usize) -> Self {
[CODE L0041]         let r1 = [start..end];
[CODE L0042]         let r2 = [start..end];
[CODE L0043]         let r3 = [start..end];
[CODE L0044]         Self {
[CODE L0045]             policy_target: self.policy_target.clone().slice(r1.clone()),
[CODE L0046]             legal_mask: self.legal_mask.clone().slice(r1.clone()),
[CODE L0047]             value_target: self.value_target.clone().slice(r2.clone()),
[CODE L0048]             grp_target: self.grp_target.clone().slice(r1.clone()),
[CODE L0049]             tenpai_target: self.tenpai_target.clone().slice(r1.clone()),
[CODE L0050]             danger_target: self.danger_target.clone().slice(r3.clone()),
[CODE L0051]             danger_mask: self.danger_mask.clone().slice(r3.clone()),
[CODE L0052]             opp_next_target: self.opp_next_target.clone().slice(r3.clone()),
[CODE L0053]             score_pdf_target: self.score_pdf_target.clone().slice(r1.clone()),
[CODE L0054]             score_cdf_target: self.score_cdf_target.clone().slice(r1.clone()),
[CODE L0055]             oracle_target: self
[CODE L0056]                 .oracle_target
[CODE L0057]                 .as_ref()
[CODE L0058]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0059]             belief_fields_target: self
[CODE L0060]                 .belief_fields_target
[CODE L0061]                 .as_ref()
[CODE L0062]                 .map(|t| t.clone().slice(r3.clone())),
[CODE L0063]             belief_fields_mask: self
[CODE L0064]                 .belief_fields_mask
[CODE L0065]                 .as_ref()
[CODE L0066]                 .map(|t| t.clone().slice(r2.clone())),
[CODE L0067]             mixture_weight_target: self
[CODE L0068]                 .mixture_weight_target
[CODE L0069]                 .as_ref()
[CODE L0070]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0071]             mixture_weight_mask: self
[CODE L0072]                 .mixture_weight_mask
[CODE L0073]                 .as_ref()
[CODE L0074]                 .map(|t| t.clone().slice(r2.clone())),
[CODE L0075]             opponent_hand_type_target: self
[CODE L0076]                 .opponent_hand_type_target
[CODE L0077]                 .as_ref()
[CODE L0078]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0079]             delta_q_target: self
[CODE L0080]                 .delta_q_target
[CODE L0081]                 .as_ref()
[CODE L0082]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0083]             delta_q_mask: self
[CODE L0084]                 .delta_q_mask
[CODE L0085]                 .as_ref()
[CODE L0086]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0087]             safety_residual_target: self
[CODE L0088]                 .safety_residual_target
[CODE L0089]                 .as_ref()
[CODE L0090]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0091]             safety_residual_mask: self
[CODE L0092]                 .safety_residual_mask
[CODE L0093]                 .as_ref()
[CODE L0094]                 .map(|t| t.clone().slice(r1.clone())),
[CODE L0095]             oracle_guidance_mask: self
[CODE L0096]                 .oracle_guidance_mask
[CODE L0097]                 .as_ref()
[CODE L0098]                 .map(|t| t.clone().slice(r2)),
[CODE L0099]             target_presence: None,
[CODE L0100]         }
[CODE L0101]     }
[CODE L0102] }
[CODE L0103] 
[CODE L0104] #[derive(Config, Debug)]
[CODE L0105] pub struct HydraLossConfig {
[CODE L0106]     #[config(default = "1.0")]
[CODE L0107]     pub w_pi: f32,
[CODE L0108]     #[config(default = "0.5")]
[CODE L0109]     pub w_v: f32,
[CODE L0110]     #[config(default = "0.2")]
[CODE L0111]     pub w_grp: f32,
[CODE L0112]     #[config(default = "0.1")]
[CODE L0113]     pub w_tenpai: f32,
[CODE L0114]     #[config(default = "0.1")]
[CODE L0115]     pub w_danger: f32,
[CODE L0116]     #[config(default = "0.1")]
[CODE L0117]     pub w_opp: f32,
[CODE L0118]     #[config(default = "0.025")]
[CODE L0119]     pub w_score: f32,
[CODE L0120]     #[config(default = "0.0")]
[CODE L0121]     pub w_oracle_critic: f32,
[CODE L0122]     #[config(default = "0.0")]
[CODE L0123]     pub w_belief_fields: f32,
[CODE L0124]     #[config(default = "0.0")]
[CODE L0125]     pub w_mixture_weight: f32,
[CODE L0126]     #[config(default = "0.0")]
[CODE L0127]     pub w_opponent_hand_type: f32,
[CODE L0128]     #[config(default = "0.0")]
[CODE L0129]     pub w_delta_q: f32,
[CODE L0130]     #[config(default = "0.0")]
[CODE L0131]     pub w_safety_residual: f32,
[CODE L0132] }
[CODE L0133] 
[CODE L0134] impl HydraLossConfig {
[CODE L0135]     pub fn total_weight(&self) -> f32 {
[CODE L0136]         self.w_pi
[CODE L0137]             + self.w_v
[CODE L0138]             + self.w_grp
[CODE L0139]             + self.w_tenpai
[CODE L0140]             + self.w_danger
[CODE L0141]             + self.w_opp
[CODE L0142]             + self.w_score * 2.0
[CODE L0143]             + self.w_oracle_critic
[CODE L0144]             + self.w_belief_fields
[CODE L0145]             + self.w_mixture_weight
[CODE L0146]             + self.w_opponent_hand_type
[CODE L0147]             + self.w_delta_q
[CODE L0148]             + self.w_safety_residual
[CODE L0149]     }
[CODE L0150] 
[CODE L0151]     pub fn scale_all(&self, factor: f32) -> Self {
[CODE L0152]         Self::new()
[CODE L0153]             .with_w_pi(self.w_pi * factor)
[CODE L0154]             .with_w_v(self.w_v * factor)
[CODE L0155]             .with_w_grp(self.w_grp * factor)
[CODE L0156]             .with_w_tenpai(self.w_tenpai * factor)
[CODE L0157]             .with_w_danger(self.w_danger * factor)
[CODE L0158]             .with_w_opp(self.w_opp * factor)
[CODE L0159]             .with_w_score(self.w_score * factor)
[CODE L0160]             .with_w_oracle_critic(self.w_oracle_critic * factor)
[CODE L0161]             .with_w_belief_fields(self.w_belief_fields * factor)
[CODE L0162]             .with_w_mixture_weight(self.w_mixture_weight * factor)
[CODE L0163]             .with_w_opponent_hand_type(self.w_opponent_hand_type * factor)
[CODE L0164]             .with_w_delta_q(self.w_delta_q * factor)
[CODE L0165]             .with_w_safety_residual(self.w_safety_residual * factor)
[CODE L0166]     }
[CODE L0167] 
[CODE L0168]     pub fn summary(&self) -> String {
[CODE L0169]         format!(
[CODE L0170]             "loss(pi={:.1}, v={:.1}, grp={:.1})",
[CODE L0171]             self.w_pi, self.w_v, self.w_grp
[CODE L0172]         )
[CODE L0173]     }
[CODE L0174] 
[CODE L0175]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0176]         if self.w_pi < 0.0
[CODE L0177]             || self.w_v < 0.0
[CODE L0178]             || self.w_grp < 0.0
[CODE L0179]             || self.w_tenpai < 0.0
[CODE L0180]             || self.w_danger < 0.0
[CODE L0181]             || self.w_opp < 0.0
[CODE L0182]             || self.w_score < 0.0
[CODE L0183]             || self.w_oracle_critic < 0.0
[CODE L0184]             || self.w_belief_fields < 0.0
[CODE L0185]             || self.w_mixture_weight < 0.0
[CODE L0186]             || self.w_opponent_hand_type < 0.0
[CODE L0187]             || self.w_delta_q < 0.0
[CODE L0188]             || self.w_safety_residual < 0.0
[CODE L0189]         {
[CODE L0190]             return Err("loss weights must be non-negative");
[CODE L0191]         }
[CODE L0192]         Ok(())
[CODE L0193]     }
```

## Artifact 08 — Model outputs and head structure
Artifact id: `model-surface`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/model.rs:11-220`
Why it matters: Shows the exact output surface already available to the model, including value and oracle critic heads.

```rust
[CODE L0011] pub struct HydraOutput<B: Backend> {
[CODE L0012]     pub policy_logits: Tensor<B, 2>,
[CODE L0013]     pub value: Tensor<B, 2>,
[CODE L0014]     pub score_pdf: Tensor<B, 2>,
[CODE L0015]     pub score_cdf: Tensor<B, 2>,
[CODE L0016]     pub opp_tenpai: Tensor<B, 2>,
[CODE L0017]     pub grp: Tensor<B, 2>,
[CODE L0018]     pub opp_next_discard: Tensor<B, 3>,
[CODE L0019]     pub danger: Tensor<B, 3>,
[CODE L0020]     pub oracle_critic: Tensor<B, 2>,
[CODE L0021]     pub belief_fields: Tensor<B, 3>,
[CODE L0022]     pub mixture_weight_logits: Tensor<B, 2>,
[CODE L0023]     pub opponent_hand_type: Tensor<B, 2>,
[CODE L0024]     pub delta_q: Tensor<B, 2>,
[CODE L0025]     pub safety_residual: Tensor<B, 2>,
[CODE L0026] }
[CODE L0027] 
[CODE L0028] pub type ActorNet<B> = HydraModel<B>;
[CODE L0029] pub type LearnerNet<B> = HydraModel<B>;
[CODE L0030] 
[CODE L0031] impl<B: Backend> HydraOutput<B> {
[CODE L0032]     pub fn masked_policy(&self, legal_mask: Tensor<B, 2>) -> Tensor<B, 2> {
[CODE L0033]         let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
[CODE L0034]         self.policy_logits.clone() + neg_inf
[CODE L0035]     }
[CODE L0036] 
[CODE L0037]     pub fn policy_logits_cpu(&self) -> Option<Vec<f32>> {
[CODE L0038]         self.policy_logits
[CODE L0039]             .to_data()
[CODE L0040]             .convert::<f32>()
[CODE L0041]             .as_slice::<f32>()
[CODE L0042]             .ok()
[CODE L0043]             .map(|s| s.to_vec())
[CODE L0044]     }
[CODE L0045] 
[CODE L0046]     pub fn value_scalar(&self) -> Option<f32> {
[CODE L0047]         self.value
[CODE L0048]             .to_data()
[CODE L0049]             .convert::<f32>()
[CODE L0050]             .as_slice::<f32>()
[CODE L0051]             .ok()
[CODE L0052]             .and_then(|s| s.first().copied())
[CODE L0053]     }
[CODE L0054] 
[CODE L0055]     pub fn is_finite(&self) -> bool {
[CODE L0056]         let check2 = |t: &Tensor<B, 2>| -> bool {
[CODE L0057]             if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
[CODE L0058]                 s.iter().all(|v| v.is_finite())
[CODE L0059]             } else {
[CODE L0060]                 false
[CODE L0061]             }
[CODE L0062]         };
[CODE L0063]         let check3 = |t: &Tensor<B, 3>| -> bool {
[CODE L0064]             if let Ok(s) = t.to_data().convert::<f32>().as_slice::<f32>() {
[CODE L0065]                 s.iter().all(|v| v.is_finite())
[CODE L0066]             } else {
[CODE L0067]                 false
[CODE L0068]             }
[CODE L0069]         };
[CODE L0070]         check2(&self.policy_logits)
[CODE L0071]             && check2(&self.value)
[CODE L0072]             && check2(&self.score_pdf)
[CODE L0073]             && check2(&self.score_cdf)
[CODE L0074]             && check2(&self.opp_tenpai)
[CODE L0075]             && check2(&self.grp)
[CODE L0076]             && check2(&self.oracle_critic)
[CODE L0077]             && check3(&self.opp_next_discard)
[CODE L0078]             && check3(&self.danger)
[CODE L0079]             && check3(&self.belief_fields)
[CODE L0080]             && check2(&self.mixture_weight_logits)
[CODE L0081]             && check2(&self.opponent_hand_type)
[CODE L0082]             && check2(&self.delta_q)
[CODE L0083]             && check2(&self.safety_residual)
[CODE L0084]     }
[CODE L0085] }
[CODE L0086] 
[CODE L0087] fn zero_linear_head<B: Backend>(batch: usize, width: usize, device: &B::Device) -> Tensor<B, 2> {
[CODE L0088]     Tensor::<B, 2>::zeros([batch, width], device)
[CODE L0089] }
[CODE L0090] 
[CODE L0091] fn zero_spatial_head<B: Backend>(
[CODE L0092]     batch: usize,
[CODE L0093]     channels: usize,
[CODE L0094]     width: usize,
[CODE L0095]     device: &B::Device,
[CODE L0096] ) -> Tensor<B, 3> {
[CODE L0097]     Tensor::<B, 3>::zeros([batch, channels, width], device)
[CODE L0098] }
[CODE L0099] 
[CODE L0100] #[derive(Module, Debug)]
[CODE L0101] pub struct HydraModel<B: Backend> {
[CODE L0102]     backbone: SEResNet<B>,
[CODE L0103]     policy: PolicyHead<B>,
[CODE L0104]     value: ValueHead<B>,
[CODE L0105]     score_pdf: ScorePdfHead<B>,
[CODE L0106]     score_cdf: ScoreCdfHead<B>,
[CODE L0107]     opp_tenpai: OppTenpaiHead<B>,
[CODE L0108]     grp: GrpHead<B>,
[CODE L0109]     opp_next_discard: OppNextDiscardHead<B>,
[CODE L0110]     danger: DangerHead<B>,
[CODE L0111]     oracle_critic: OracleCriticHead<B>,
[CODE L0112]     belief_field: BeliefFieldHead<B>,
[CODE L0113]     mixture_weight: MixtureWeightHead<B>,
[CODE L0114]     opponent_hand_type: OpponentHandTypeHead<B>,
[CODE L0115]     delta_q: DeltaQHead<B>,
[CODE L0116]     safety_residual: SafetyResidualHead<B>,
[CODE L0117] }
[CODE L0118] 
[CODE L0119] #[derive(Config, Debug)]
[CODE L0120] pub struct HydraModelConfig {
[CODE L0121]     pub num_blocks: usize,
[CODE L0122]     #[config(default = "192")]
[CODE L0123]     pub input_channels: usize,
[CODE L0124]     #[config(default = "256")]
[CODE L0125]     pub hidden_channels: usize,
[CODE L0126]     #[config(default = "32")]
[CODE L0127]     pub num_groups: usize,
[CODE L0128]     #[config(default = "64")]
[CODE L0129]     pub se_bottleneck: usize,
[CODE L0130]     #[config(default = "46")]
[CODE L0131]     pub action_space: usize,
[CODE L0132]     #[config(default = "64")]
[CODE L0133]     pub score_bins: usize,
[CODE L0134]     #[config(default = "3")]
[CODE L0135]     pub num_opponents: usize,
[CODE L0136]     #[config(default = "24")]
[CODE L0137]     pub grp_classes: usize,
[CODE L0138]     #[config(default = "4")]
[CODE L0139]     pub num_belief_components: usize,
[CODE L0140]     #[config(default = "8")]
[CODE L0141]     pub opponent_hand_type_classes: usize,
[CODE L0142] }
[CODE L0143] 
[CODE L0144] impl HydraModelConfig {
[CODE L0145]     pub fn summary(&self) -> String {
[CODE L0146]         let kind = if self.num_blocks <= 12 {
[CODE L0147]             "actor"
[CODE L0148]         } else {
[CODE L0149]             "learner"
[CODE L0150]         };
[CODE L0151]         format!(
[CODE L0152]             "{}(blocks={}, ch={})",
[CODE L0153]             kind, self.num_blocks, self.hidden_channels
[CODE L0154]         )
[CODE L0155]     }
[CODE L0156] 
[CODE L0157]     pub fn is_actor(&self) -> bool {
[CODE L0158]         self.num_blocks == 12
[CODE L0159]     }
[CODE L0160]     pub fn is_learner(&self) -> bool {
[CODE L0161]         self.num_blocks == 24
[CODE L0162]     }
[CODE L0163] 
[CODE L0164]     pub fn validate(&self) -> Result<(), &'static str> {
[CODE L0165]         if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
[CODE L0166]             return Err("hidden_channels must be divisible by num_groups");
[CODE L0167]         }
[CODE L0168]         if self.num_blocks == 0 {
[CODE L0169]             return Err("num_blocks must be > 0");
[CODE L0170]         }
[CODE L0171]         if self.se_bottleneck == 0 {
[CODE L0172]             return Err("se_bottleneck must be > 0");
[CODE L0173]         }
[CODE L0174]         if self.num_belief_components == 0 {
[CODE L0175]             return Err("num_belief_components must be > 0");
[CODE L0176]         }
[CODE L0177]         if self.opponent_hand_type_classes == 0 {
[CODE L0178]             return Err("opponent_hand_type_classes must be > 0");
[CODE L0179]         }
[CODE L0180]         Ok(())
[CODE L0181]     }
[CODE L0182] 
[CODE L0183]     pub fn actor() -> Self {
[CODE L0184]         Self::new(12).with_input_channels(INPUT_CHANNELS)
[CODE L0185]     }
[CODE L0186] 
[CODE L0187]     pub fn estimated_params(&self) -> usize {
[CODE L0188]         let h = self.hidden_channels;
[CODE L0189]         let se_b = self.se_bottleneck;
[CODE L0190]         let input_conv = self.input_channels * h * 3 + h;
[CODE L0191]         let gn = h * 2;
[CODE L0192]         let block = (h * h * 3 + h) * 2 + gn * 2 + (h * se_b + se_b) + (se_b * h + h);
[CODE L0193]         let backbone = input_conv + gn + block * self.num_blocks + gn;
[CODE L0194]         let policy = h * self.action_space + self.action_space;
[CODE L0195]         let value = h + 1;
[CODE L0196]         let score = (h * self.score_bins + self.score_bins) * 2;
[CODE L0197]         let tenpai = h * self.num_opponents + self.num_opponents;
[CODE L0198]         let grp = h * self.grp_classes + self.grp_classes;
[CODE L0199]         let opp_next = h * self.num_opponents + self.num_opponents;
[CODE L0200]         let danger = h * self.num_opponents + self.num_opponents;
[CODE L0201]         let oracle = h * 4 + 4;
[CODE L0202]         let belief_field = h * (self.num_belief_components * 4) + (self.num_belief_components * 4);
[CODE L0203]         let mixture_weight = h * self.num_belief_components + self.num_belief_components;
[CODE L0204]         let opponent_hand_type = h * (self.num_opponents * self.opponent_hand_type_classes)
[CODE L0205]             + (self.num_opponents * self.opponent_hand_type_classes);
[CODE L0206]         let delta_q = h * self.action_space + self.action_space;
[CODE L0207]         let safety_residual = h * self.action_space + self.action_space;
[CODE L0208]         backbone
[CODE L0209]             + policy
[CODE L0210]             + value
[CODE L0211]             + score
[CODE L0212]             + tenpai
[CODE L0213]             + grp
[CODE L0214]             + opp_next
[CODE L0215]             + danger
[CODE L0216]             + oracle
[CODE L0217]             + belief_field
[CODE L0218]             + mixture_weight
[CODE L0219]             + opponent_hand_type
[CODE L0220]             + delta_q
```

## Artifact 09 — Training phase enum and budgets
Artifact id: `training-phase-config`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/config.rs:78-159`
Why it matters: Critical for deciding where a DCRL-style lane belongs in Hydra's phase schedule.

```rust
[CODE L0078] #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
[CODE L0079] pub enum TrainingPhase {
[CODE L0080]     BenchmarkGates,
[CODE L0081]     BcWarmStart,
[CODE L0082]     OracleGuiding,
[CODE L0083]     DrdaAchSelfPlay,
[CODE L0084]     ExitPondering,
[CODE L0085] }
[CODE L0086] 
[CODE L0087] impl TrainingPhase {
[CODE L0088]     pub fn gpu_hours_budget(self) -> u32 {
[CODE L0089]         match self {
[CODE L0090]             Self::BenchmarkGates => 150,
[CODE L0091]             Self::BcWarmStart => 50,
[CODE L0092]             Self::OracleGuiding => 200,
[CODE L0093]             Self::DrdaAchSelfPlay => 800,
[CODE L0094]             Self::ExitPondering => 800,
[CODE L0095]         }
[CODE L0096]     }
[CODE L0097] 
[CODE L0098]     pub fn cumulative_budget_before(self) -> u32 {
[CODE L0099]         match self {
[CODE L0100]             Self::BenchmarkGates => 0,
[CODE L0101]             Self::BcWarmStart => Self::BenchmarkGates.gpu_hours_budget(),
[CODE L0102]             Self::OracleGuiding => {
[CODE L0103]                 Self::BenchmarkGates.gpu_hours_budget() + Self::BcWarmStart.gpu_hours_budget()
[CODE L0104]             }
[CODE L0105]             Self::DrdaAchSelfPlay => {
[CODE L0106]                 Self::BenchmarkGates.gpu_hours_budget()
[CODE L0107]                     + Self::BcWarmStart.gpu_hours_budget()
[CODE L0108]                     + Self::OracleGuiding.gpu_hours_budget()
[CODE L0109]             }
[CODE L0110]             Self::ExitPondering => {
[CODE L0111]                 Self::BenchmarkGates.gpu_hours_budget()
[CODE L0112]                     + Self::BcWarmStart.gpu_hours_budget()
[CODE L0113]                     + Self::OracleGuiding.gpu_hours_budget()
[CODE L0114]                     + Self::DrdaAchSelfPlay.gpu_hours_budget()
[CODE L0115]             }
[CODE L0116]         }
[CODE L0117]     }
[CODE L0118] 
[CODE L0119]     pub fn cumulative_budget_through(self) -> u32 {
[CODE L0120]         self.cumulative_budget_before() + self.gpu_hours_budget()
[CODE L0121]     }
[CODE L0122] 
[CODE L0123]     pub fn exit_schedule_phase(self) -> u8 {
[CODE L0124]         match self {
[CODE L0125]             Self::BenchmarkGates | Self::BcWarmStart | Self::OracleGuiding => 1,
[CODE L0126]             Self::DrdaAchSelfPlay => 2,
[CODE L0127]             Self::ExitPondering => 3,
[CODE L0128]         }
[CODE L0129]     }
[CODE L0130] 
[CODE L0131]     pub fn next(self) -> Option<Self> {
[CODE L0132]         match self {
[CODE L0133]             Self::BenchmarkGates => Some(Self::BcWarmStart),
[CODE L0134]             Self::BcWarmStart => Some(Self::OracleGuiding),
[CODE L0135]             Self::OracleGuiding => Some(Self::DrdaAchSelfPlay),
[CODE L0136]             Self::DrdaAchSelfPlay => Some(Self::ExitPondering),
[CODE L0137]             Self::ExitPondering => None,
[CODE L0138]         }
[CODE L0139]     }
[CODE L0140] 
[CODE L0141]     pub fn uses_exit(self) -> bool {
[CODE L0142]         matches!(self, Self::DrdaAchSelfPlay | Self::ExitPondering)
[CODE L0143]     }
[CODE L0144] 
[CODE L0145]     pub fn uses_oracle(self) -> bool {
[CODE L0146]         matches!(
[CODE L0147]             self,
[CODE L0148]             Self::OracleGuiding | Self::DrdaAchSelfPlay | Self::ExitPondering
[CODE L0149]         )
[CODE L0150]     }
[CODE L0151] 
[CODE L0152]     pub fn phase_index(self) -> u8 {
[CODE L0153]         match self {
[CODE L0154]             Self::BenchmarkGates => 0,
[CODE L0155]             Self::BcWarmStart => 1,
[CODE L0156]             Self::OracleGuiding => 2,
[CODE L0157]             Self::DrdaAchSelfPlay => 3,
[CODE L0158]             Self::ExitPondering => 4,
[CODE L0159]         }
```

## Artifact 10 — Pipeline phase state and progress logic
Artifact id: `pipeline-state`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/config.rs:262-353`
Why it matters: Lets the agent tie any proposed lane to the actual phase-advance machinery Hydra already uses.

```rust
[CODE L0262] pub struct PipelineState {
[CODE L0263]     pub phase: TrainingPhase,
[CODE L0264]     pub gpu_hours_used: f32,
[CODE L0265]     pub total_games: u64,
[CODE L0266]     pub total_samples: u64,
[CODE L0267]     pub learner_version: u32,
[CODE L0268]     pub actor_version: u32,
[CODE L0269] }
[CODE L0270] 
[CODE L0271] impl Default for PipelineState {
[CODE L0272]     fn default() -> Self {
[CODE L0273]         Self {
[CODE L0274]             phase: TrainingPhase::BenchmarkGates,
[CODE L0275]             gpu_hours_used: 0.0,
[CODE L0276]             total_games: 0,
[CODE L0277]             total_samples: 0,
[CODE L0278]             learner_version: 0,
[CODE L0279]             actor_version: 0,
[CODE L0280]         }
[CODE L0281]     }
[CODE L0282] }
[CODE L0283] 
[CODE L0284] impl PipelineState {
[CODE L0285]     pub fn advance_phase(&mut self) {
[CODE L0286]         self.phase = match self.phase {
[CODE L0287]             TrainingPhase::BenchmarkGates => TrainingPhase::BcWarmStart,
[CODE L0288]             TrainingPhase::BcWarmStart => TrainingPhase::OracleGuiding,
[CODE L0289]             TrainingPhase::OracleGuiding => TrainingPhase::DrdaAchSelfPlay,
[CODE L0290]             TrainingPhase::DrdaAchSelfPlay => TrainingPhase::ExitPondering,
[CODE L0291]             TrainingPhase::ExitPondering => TrainingPhase::ExitPondering,
[CODE L0292]         };
[CODE L0293]     }
[CODE L0294] 
[CODE L0295]     pub fn remaining_budget(&self) -> f32 {
[CODE L0296]         2000.0 - self.gpu_hours_used
[CODE L0297]     }
[CODE L0298] 
[CODE L0299]     pub fn total_budget() -> f32 {
[CODE L0300]         2000.0
[CODE L0301]     }
[CODE L0302] 
[CODE L0303]     pub fn overall_progress(&self) -> f32 {
[CODE L0304]         (self.gpu_hours_used / Self::total_budget()).min(1.0)
[CODE L0305]     }
[CODE L0306] 
[CODE L0307]     pub fn phase_progress(&self) -> f32 {
[CODE L0308]         let budget = self.phase.gpu_hours_budget() as f32;
[CODE L0309]         if budget == 0.0 {
[CODE L0310]             return 0.0;
[CODE L0311]         }
[CODE L0312]         self.phase_hours_used() / budget
[CODE L0313]     }
[CODE L0314] 
[CODE L0315]     pub fn phase_hours_used(&self) -> f32 {
[CODE L0316]         let phase_start = self.phase.cumulative_budget_before() as f32;
[CODE L0317]         let phase_budget = self.phase.gpu_hours_budget() as f32;
[CODE L0318]         (self.gpu_hours_used - phase_start).clamp(0.0, phase_budget)
[CODE L0319]     }
[CODE L0320] 
[CODE L0321]     pub fn increment_learner_version(&mut self) {
[CODE L0322]         self.learner_version += 1;
[CODE L0323]     }
[CODE L0324]     pub fn increment_actor_version(&mut self) {
[CODE L0325]         self.actor_version += 1;
[CODE L0326]     }
[CODE L0327] 
[CODE L0328]     pub fn record_game(&mut self, num_samples: usize) {
[CODE L0329]         self.total_games += 1;
[CODE L0330]         self.total_samples += num_samples as u64;
[CODE L0331]     }
[CODE L0332] 
[CODE L0333]     pub fn tick_gpu_hours(&mut self, hours: f32) {
[CODE L0334]         self.gpu_hours_used += hours;
[CODE L0335]     }
[CODE L0336] 
[CODE L0337]     pub fn should_advance_phase(&self) -> bool {
[CODE L0338]         self.gpu_hours_used >= self.phase.cumulative_budget_through() as f32
[CODE L0339]     }
[CODE L0340] 
[CODE L0341]     pub fn progress_summary(&self) -> String {
[CODE L0342]         format!(
[CODE L0343]             "phase={:?} phase_hours={:.1}/{} total_hours={:.1} games={} v{}->v{}",
[CODE L0344]             self.phase,
[CODE L0345]             self.phase_hours_used(),
[CODE L0346]             self.phase.gpu_hours_budget(),
[CODE L0347]             self.gpu_hours_used,
[CODE L0348]             self.total_games,
[CODE L0349]             self.learner_version,
[CODE L0350]             self.actor_version
[CODE L0351]         )
[CODE L0352]     }
[CODE L0353] }
```

## Artifact 11 — Advanced-head activation and gating controller
Artifact id: `head-gates`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/training/head_gates.rs:577-860`
Why it matters: High-value surface for deciding how a DCRL-related lane should be staged, warmed up, and gated instead of made baseline immediately.

```rust
[CODE L0577] // HeadActivationController -- orchestrates density, conflict, and warmup
[CODE L0578] // ---------------------------------------------------------------------------
[CODE L0579] 
[CODE L0580] /// Manages per-head activation state, density tracking, gradient conflict
[CODE L0581] /// monitoring, and warmup-to-active transitions.
[CODE L0582] ///
[CODE L0583] /// All advanced heads start in [`HeadState::Off`]. The caller requests
[CODE L0584] /// activation via [`try_activate`](Self::try_activate), which checks the
[CODE L0585] /// density gate and transitions to [`HeadState::Warmup`] if it passes.
[CODE L0586] /// During warmup, the caller should freeze trunk gradient flow for the
[CODE L0587] /// head's loss. After the warmup countdown completes,
[CODE L0588] /// [`tick_warmup`](Self::tick_warmup) checks the gradient conflict gate
[CODE L0589] /// and transitions to [`HeadState::Active`] or back to [`HeadState::Off`].
[CODE L0590] #[derive(Clone, Debug)]
[CODE L0591] pub struct HeadActivationController {
[CODE L0592]     coverage: HeadCoverage,
[CODE L0593]     conflict: GradConflictTracker,
[CODE L0594]     config: HeadActivationConfig,
[CODE L0595]     states: [HeadState; NUM_ADVANCED_HEADS],
[CODE L0596]     warmup_steps_remaining: [usize; NUM_ADVANCED_HEADS],
[CODE L0597] }
[CODE L0598] 
[CODE L0599] impl HeadActivationController {
[CODE L0600]     /// Creates a controller with all heads in [`HeadState::Off`].
[CODE L0601]     pub fn new(config: HeadActivationConfig) -> Self {
[CODE L0602]         Self {
[CODE L0603]             coverage: HeadCoverage::new(),
[CODE L0604]             conflict: GradConflictTracker::new(),
[CODE L0605]             config,
[CODE L0606]             states: [HeadState::Off; NUM_ADVANCED_HEADS],
[CODE L0607]             warmup_steps_remaining: [0; NUM_ADVANCED_HEADS],
[CODE L0608]         }
[CODE L0609]     }
[CODE L0610] 
[CODE L0611]     // -- Data collection ---------------------------------------------------
[CODE L0612] 
[CODE L0613]     /// Records per-head target presence from one training batch.
[CODE L0614]     pub fn record_batch(&mut self, presence: &TargetPresence) {
[CODE L0615]         self.coverage.record_batch(presence);
[CODE L0616]     }
[CODE L0617] 
[CODE L0618]     /// Records a shared-trunk gradient cosine measurement for `head`.
[CODE L0619]     pub fn record_grad_cosine(&mut self, head: AdvancedHead, cosine: f32) {
[CODE L0620]         self.conflict.record(head, cosine);
[CODE L0621]     }
[CODE L0622] 
[CODE L0623]     // -- State queries -----------------------------------------------------
[CODE L0624] 
[CODE L0625]     /// Returns the current activation state of `head`.
[CODE L0626]     pub fn head_state(&self, head: AdvancedHead) -> HeadState {
[CODE L0627]         self.states[head.index()]
[CODE L0628]     }
[CODE L0629] 
[CODE L0630]     /// Returns all heads currently in [`HeadState::Warmup`].
[CODE L0631]     ///
[CODE L0632]     /// The caller should detach trunk outputs for these heads so only head
[CODE L0633]     /// parameters receive gradients.
[CODE L0634]     pub fn warmup_heads(&self) -> Vec<AdvancedHead> {
[CODE L0635]         AdvancedHead::ALL
[CODE L0636]             .iter()
[CODE L0637]             .copied()
[CODE L0638]             .filter(|h| self.states[h.index()] == HeadState::Warmup)
[CODE L0639]             .collect()
[CODE L0640]     }
[CODE L0641] 
[CODE L0642]     /// Returns a reference to the underlying coverage tracker.
[CODE L0643]     pub fn coverage(&self) -> &HeadCoverage {
[CODE L0644]         &self.coverage
[CODE L0645]     }
[CODE L0646] 
[CODE L0647]     /// Returns a reference to the underlying conflict tracker.
[CODE L0648]     pub fn conflict(&self) -> &GradConflictTracker {
[CODE L0649]         &self.conflict
[CODE L0650]     }
[CODE L0651] 
[CODE L0652]     // -- Gate evaluation ---------------------------------------------------
[CODE L0653] 
[CODE L0654]     /// Evaluates all applicable gates for `head` without changing state.
[CODE L0655]     pub fn evaluate(&self, head: AdvancedHead) -> HeadGateReport {
[CODE L0656]         let mut failures = Vec::new();
[CODE L0657]         let rho = self.coverage.rho(head);
[CODE L0658]         let spp = match head.kind() {
[CODE L0659]             HeadKind::SparseSearch => Some(self.coverage.spp(head, self.config.learner_params)),
[CODE L0660]             HeadKind::Dense => None,
[CODE L0661]         };
[CODE L0662]         let negative_frac = self.conflict.negative_fraction(head);
[CODE L0663] 
[CODE L0664]         // Check minimum samples.
[CODE L0665]         if self.coverage.total_samples() < self.config.min_eval_samples {
[CODE L0666]             failures.push("insufficient_samples");
[CODE L0667]         }
[CODE L0668] 
[CODE L0669]         // Density gate.
[CODE L0670]         match head.kind() {
[CODE L0671]             HeadKind::Dense => {
[CODE L0672]                 if rho < self.config.min_dense_rho {
[CODE L0673]                     failures.push("density_rho_below_threshold");
[CODE L0674]                 }
[CODE L0675]             }
[CODE L0676]             HeadKind::SparseSearch => {
[CODE L0677]                 if let Some(s) = spp
[CODE L0678]                     && s < self.config.min_sparse_spp
[CODE L0679]                 {
[CODE L0680]                     failures.push("density_spp_below_threshold");
[CODE L0681]                 }
[CODE L0682]             }
[CODE L0683]         }
[CODE L0684] 
[CODE L0685]         // Gradient conflict gate (only if enough checks).
[CODE L0686]         if self.conflict.total_checks(head) >= self.config.min_conflict_checks
[CODE L0687]             && self
[CODE L0688]                 .conflict
[CODE L0689]                 .is_conflicting(head, self.config.max_negative_frac)
[CODE L0690]         {
[CODE L0691]             failures.push("gradient_conflict");
[CODE L0692]         }
[CODE L0693] 
[CODE L0694]         HeadGateReport {
[CODE L0695]             head,
[CODE L0696]             approved: failures.is_empty(),
[CODE L0697]             state: self.states[head.index()],
[CODE L0698]             rho,
[CODE L0699]             spp,
[CODE L0700]             negative_frac,
[CODE L0701]             failures,
[CODE L0702]         }
[CODE L0703]     }
[CODE L0704] 
[CODE L0705]     /// Evaluates all advanced heads.
[CODE L0706]     pub fn evaluate_all(&self) -> Vec<HeadGateReport> {
[CODE L0707]         AdvancedHead::ALL
[CODE L0708]             .iter()
[CODE L0709]             .map(|&h| self.evaluate(h))
[CODE L0710]             .collect()
[CODE L0711]     }
[CODE L0712] 
[CODE L0713]     // -- State transitions -------------------------------------------------
[CODE L0714] 
[CODE L0715]     /// Attempts to activate `head`.
[CODE L0716]     ///
[CODE L0717]     /// - If the head is [`HeadState::Off`] and the density gate passes,
[CODE L0718]     ///   transitions to [`HeadState::Warmup`] with the configured warmup
[CODE L0719]     ///   countdown.
[CODE L0720]     /// - If the head is already in `Warmup` or `Active`, returns a report
[CODE L0721]     ///   reflecting the current state without changing it.
[CODE L0722]     pub fn try_activate(&mut self, head: AdvancedHead) -> HeadGateReport {
[CODE L0723]         let idx = head.index();
[CODE L0724]         match self.states[idx] {
[CODE L0725]             HeadState::Warmup | HeadState::Active => {
[CODE L0726]                 // Already activated or activating.
[CODE L0727]                 return self.evaluate(head);
[CODE L0728]             }
[CODE L0729]             HeadState::Off => {}
[CODE L0730]         }
[CODE L0731] 
[CODE L0732]         let report = self.evaluate(head);
[CODE L0733] 
[CODE L0734]         // Only check density gate for Off -> Warmup transition.
[CODE L0735]         // Gradient conflict is checked after warmup completes.
[CODE L0736]         let density_ok = !report.failures.contains(&"insufficient_samples")
[CODE L0737]             && !report.failures.contains(&"density_rho_below_threshold")
[CODE L0738]             && !report.failures.contains(&"density_spp_below_threshold");
[CODE L0739] 
[CODE L0740]         if density_ok {
[CODE L0741]             self.states[idx] = HeadState::Warmup;
[CODE L0742]             self.warmup_steps_remaining[idx] = self.config.warmup_steps;
[CODE L0743]             // Return updated report reflecting new state.
[CODE L0744]             let mut updated = report;
[CODE L0745]             updated.state = HeadState::Warmup;
[CODE L0746]             // For Off -> Warmup, density passed so we approve the transition.
[CODE L0747]             // The conflict gate is deferred until warmup completes.
[CODE L0748]             updated.approved = true;
[CODE L0749]             updated.failures.retain(|f| *f != "gradient_conflict");
[CODE L0750]             return updated;
[CODE L0751]         }
[CODE L0752] 
[CODE L0753]         report
[CODE L0754]     }
[CODE L0755] 
[CODE L0756]     /// Attempts to activate all heads that are currently [`HeadState::Off`].
[CODE L0757]     pub fn try_activate_all(&mut self) -> Vec<HeadGateReport> {
[CODE L0758]         AdvancedHead::ALL
[CODE L0759]             .iter()
[CODE L0760]             .map(|&h| self.try_activate(h))
[CODE L0761]             .collect()
[CODE L0762]     }
[CODE L0763] 
[CODE L0764]     /// Advances warmup countdowns by one step and handles transitions.
[CODE L0765]     ///
[CODE L0766]     /// For each head in [`HeadState::Warmup`]:
[CODE L0767]     /// - If warmup steps remain, decrements the counter.
[CODE L0768]     /// - If warmup is complete and sufficient gradient conflict data exists,
[CODE L0769]     ///   transitions to [`HeadState::Active`] (conflict passes) or
[CODE L0770]     ///   [`HeadState::Off`] (conflict fails).
[CODE L0771]     /// - If warmup is complete but insufficient conflict data, stays in
[CODE L0772]     ///   `Warmup` until enough data accumulates.
[CODE L0773]     pub fn tick_warmup(&mut self) {
[CODE L0774]         for &head in &AdvancedHead::ALL {
[CODE L0775]             let idx = head.index();
[CODE L0776]             if self.states[idx] != HeadState::Warmup {
[CODE L0777]                 continue;
[CODE L0778]             }
[CODE L0779] 
[CODE L0780]             if self.warmup_steps_remaining[idx] > 0 {
[CODE L0781]                 self.warmup_steps_remaining[idx] -= 1;
[CODE L0782]                 if self.warmup_steps_remaining[idx] > 0 {
[CODE L0783]                     continue;
[CODE L0784]                 }
[CODE L0785]             }
[CODE L0786] 
[CODE L0787]             // Warmup countdown complete. Check conflict gate if we have
[CODE L0788]             // enough gradient cosine data.
[CODE L0789]             if self.conflict.total_checks(head) < self.config.min_conflict_checks {
[CODE L0790]                 // Not enough data yet; stay in warmup.
[CODE L0791]                 continue;
[CODE L0792]             }
[CODE L0793] 
[CODE L0794]             if self
[CODE L0795]                 .conflict
[CODE L0796]                 .is_conflicting(head, self.config.max_negative_frac)
[CODE L0797]             {
[CODE L0798]                 self.states[idx] = HeadState::Off;
[CODE L0799]             } else {
[CODE L0800]                 self.states[idx] = HeadState::Active;
[CODE L0801]             }
[CODE L0802]         }
[CODE L0803]     }
[CODE L0804] 
[CODE L0805]     /// Forces a head back to [`HeadState::Off`].
[CODE L0806]     pub fn force_off(&mut self, head: AdvancedHead) {
[CODE L0807]         let idx = head.index();
[CODE L0808]         self.states[idx] = HeadState::Off;
[CODE L0809]         self.warmup_steps_remaining[idx] = 0;
[CODE L0810]     }
[CODE L0811] 
[CODE L0812]     // -- Loss config integration -------------------------------------------
[CODE L0813] 
[CODE L0814]     /// Returns a [`HydraLossConfig`] with unapproved heads zeroed out.
[CODE L0815]     ///
[CODE L0816]     /// - [`HeadState::Off`] heads get weight `0.0`.
[CODE L0817]     /// - [`HeadState::Warmup`] and [`HeadState::Active`] heads keep
[CODE L0818]     ///   their weight from `base`.
[CODE L0819]     ///
[CODE L0820]     /// Baseline heads (policy, value, grp, tenpai, danger, opp_next, score)
[CODE L0821]     /// are always passed through unchanged.
[CODE L0822]     pub fn approved_loss_config(&self, base: &HydraLossConfig) -> HydraLossConfig {
[CODE L0823]         let gate = |head: AdvancedHead, w: f32| -> f32 {
[CODE L0824]             match self.states[head.index()] {
[CODE L0825]                 HeadState::Off => 0.0,
[CODE L0826]                 HeadState::Warmup | HeadState::Active => w,
[CODE L0827]             }
[CODE L0828]         };
[CODE L0829]         HydraLossConfig::new()
[CODE L0830]             .with_w_pi(base.w_pi)
[CODE L0831]             .with_w_v(base.w_v)
[CODE L0832]             .with_w_grp(base.w_grp)
[CODE L0833]             .with_w_tenpai(base.w_tenpai)
[CODE L0834]             .with_w_danger(base.w_danger)
[CODE L0835]             .with_w_opp(base.w_opp)
[CODE L0836]             .with_w_score(base.w_score)
[CODE L0837]             .with_w_oracle_critic(gate(AdvancedHead::OracleCritic, base.w_oracle_critic))
[CODE L0838]             .with_w_belief_fields(gate(AdvancedHead::BeliefFields, base.w_belief_fields))
[CODE L0839]             .with_w_mixture_weight(gate(AdvancedHead::MixtureWeight, base.w_mixture_weight))
[CODE L0840]             .with_w_opponent_hand_type(gate(
[CODE L0841]                 AdvancedHead::OpponentHandType,
[CODE L0842]                 base.w_opponent_hand_type,
[CODE L0843]             ))
[CODE L0844]             .with_w_delta_q(gate(AdvancedHead::DeltaQ, base.w_delta_q))
[CODE L0845]             .with_w_safety_residual(gate(AdvancedHead::SafetyResidual, base.w_safety_residual))
[CODE L0846]     }
[CODE L0847] 
[CODE L0848]     /// Returns a multi-line summary of all heads for logging.
[CODE L0849]     pub fn summary(&self) -> String {
[CODE L0850]         let mut lines = Vec::new();
[CODE L0851]         lines.push(format!(
[CODE L0852]             "HeadActivationController (samples={})",
[CODE L0853]             self.coverage.total_samples()
[CODE L0854]         ));
[CODE L0855]         for &head in &AdvancedHead::ALL {
[CODE L0856]             let report = self.evaluate(head);
[CODE L0857]             lines.push(format!("  {}", report.summary()));
[CODE L0858]         }
[CODE L0859]         lines.join("\n")
[CODE L0860]     }
```

## Artifact 12 — Phase-aware orchestration and gate evaluation
Artifact id: `orchestrator`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/training/orchestrator.rs:1-260`
Why it matters: Needed for blueprinting the exact promotion gates, benchmark gates, and validation gates a new lane would need to obey.

```rust
[CODE L0001] //! Phase-aware training orchestration and gate evaluation.
[CODE L0002] 
[CODE L0003] use burn::prelude::*;
[CODE L0004] use burn::tensor::backend::AutodiffBackend;
[CODE L0005] 
[CODE L0006] use crate::config::{OracleGuidingConfig, PipelineState, TrainingPhase};
[CODE L0007] use crate::data::sample::MjaiBatch;
[CODE L0008] use crate::model::HydraModel;
[CODE L0009] use crate::training::bc::{
[CODE L0010]     BcExitConfig, bc_train_step, oracle_guiding_train_step, phase_learning_rate,
[CODE L0011] };
[CODE L0012] use crate::training::distill::{DistillConfig, DistillState};
[CODE L0013] use crate::training::drda::RebaseTracker;
[CODE L0014] use crate::training::exit::ExitConfig;
[CODE L0015] use crate::training::head_gates::HeadActivationController;
[CODE L0016] use crate::training::live_exit::LiveExitConfig;
[CODE L0017] use crate::training::losses::{HydraLoss, HydraTargets};
[CODE L0018] use crate::training::rl::{RlBatch, RlConfig, rl_step_with_phase_progress_and_controller};
[CODE L0019] 
[CODE L0020] #[derive(Debug, Clone, Copy, PartialEq)]
[CODE L0021] pub struct BenchmarkGateMetrics {
[CODE L0022]     pub afbs_on_turn_ms: f32,
[CODE L0023]     pub ct_smc_dp_ms: f32,
[CODE L0024]     pub endgame_exact_ms: f32,
[CODE L0025]     pub self_play_games_per_sec: f32,
[CODE L0026]     pub distill_kl_drift: f32,
[CODE L0027] }
[CODE L0028] 
[CODE L0029] #[derive(Debug, Clone, Copy, PartialEq)]
[CODE L0030] pub struct ValidationGateMetrics {
[CODE L0031]     pub mean_decision_improvement: f32,
[CODE L0032]     pub negative_decision_fraction: f32,
[CODE L0033]     pub opponent_kl_p95: f32,
[CODE L0034]     pub opponent_kl_p95_limit: f32,
[CODE L0035]     pub hunter_overfold_reduction: f32,
[CODE L0036]     pub danger_underestimate_rate: f32,
[CODE L0037]     pub max_danger_underestimate_rate: f32,
[CODE L0038]     pub saf_advantage_over_shallow: f32,
[CODE L0039] }
[CODE L0040] 
[CODE L0041] #[derive(Debug, Clone, PartialEq, Eq)]
[CODE L0042] pub struct GateReport {
[CODE L0043]     pub passed: bool,
[CODE L0044]     pub failures: Vec<&'static str>,
[CODE L0045] }
[CODE L0046] 
[CODE L0047] #[derive(Debug, Clone, Copy, PartialEq)]
[CODE L0048] pub struct PhaseTrainReport {
[CODE L0049]     pub phase: TrainingPhase,
[CODE L0050]     pub skipped: bool,
[CODE L0051]     pub loss: Option<f64>,
[CODE L0052]     pub effective_lr: f64,
[CODE L0053]     pub oracle_keep_prob: Option<f32>,
[CODE L0054]     pub kept_oracle_fraction: Option<f32>,
[CODE L0055]     pub exit_weight: Option<f32>,
[CODE L0056] }
[CODE L0057] 
[CODE L0058] #[derive(Debug, Clone, Copy, PartialEq)]
[CODE L0059] pub struct MaintenancePlan {
[CODE L0060]     pub should_rebase: bool,
[CODE L0061]     pub should_distill: bool,
[CODE L0062]     pub distill_warning: bool,
[CODE L0063]     pub shallow_exit_enabled: bool,
[CODE L0064]     pub deep_exit_enabled: bool,
[CODE L0065] }
[CODE L0066] 
[CODE L0067] pub fn evaluate_benchmark_gates(
[CODE L0068]     metrics: &BenchmarkGateMetrics,
[CODE L0069]     max_distill_kl_drift: f32,
[CODE L0070] ) -> GateReport {
[CODE L0071]     let mut failures = Vec::new();
[CODE L0072]     if metrics.afbs_on_turn_ms >= 150.0 {
[CODE L0073]         failures.push("latency_afbs_on_turn");
[CODE L0074]     }
[CODE L0075]     if metrics.ct_smc_dp_ms >= 1.0 {
[CODE L0076]         failures.push("latency_ct_smc_dp");
[CODE L0077]     }
[CODE L0078]     if metrics.endgame_exact_ms >= 100.0 {
[CODE L0079]         failures.push("latency_endgame_exact");
[CODE L0080]     }
[CODE L0081]     if metrics.self_play_games_per_sec <= 20.0 {
[CODE L0082]         failures.push("throughput_self_play");
[CODE L0083]     }
[CODE L0084]     if metrics.distill_kl_drift > max_distill_kl_drift {
[CODE L0085]         failures.push("distill_kl_drift");
[CODE L0086]     }
[CODE L0087]     GateReport {
[CODE L0088]         passed: failures.is_empty(),
[CODE L0089]         failures,
[CODE L0090]     }
[CODE L0091] }
[CODE L0092] 
[CODE L0093] pub fn evaluate_validation_gates(metrics: &ValidationGateMetrics) -> GateReport {
[CODE L0094]     let mut failures = Vec::new();
[CODE L0095]     if metrics.mean_decision_improvement <= 0.0 {
[CODE L0096]         failures.push("g0_mean_decision_improvement");
[CODE L0097]     }
[CODE L0098]     if metrics.negative_decision_fraction >= 0.40 {
[CODE L0099]         failures.push("g0_negative_fraction");
[CODE L0100]     }
[CODE L0101]     if metrics.opponent_kl_p95 > metrics.opponent_kl_p95_limit {
[CODE L0102]         failures.push("g1_robustness_calibration");
[CODE L0103]     }
[CODE L0104]     if metrics.hunter_overfold_reduction <= 0.0 {
[CODE L0105]         failures.push("g2_hunter_overfold_reduction");
[CODE L0106]     }
[CODE L0107]     if metrics.danger_underestimate_rate > metrics.max_danger_underestimate_rate {
[CODE L0108]         failures.push("g2_danger_underestimate_rate");
[CODE L0109]     }
[CODE L0110]     if metrics.saf_advantage_over_shallow <= 0.0 {
[CODE L0111]         failures.push("g3_saf_amortization");
[CODE L0112]     }
[CODE L0113]     GateReport {
[CODE L0114]         passed: failures.is_empty(),
[CODE L0115]         failures,
[CODE L0116]     }
[CODE L0117] }
[CODE L0118] 
[CODE L0119] pub fn phase_advance_report(
[CODE L0120]     state: &PipelineState,
[CODE L0121]     benchmark_report: Option<&GateReport>,
[CODE L0122]     validation_report: Option<&GateReport>,
[CODE L0123] ) -> GateReport {
[CODE L0124]     let mut failures = Vec::new();
[CODE L0125]     match state.phase {
[CODE L0126]         TrainingPhase::BenchmarkGates => match benchmark_report {
[CODE L0127]             Some(report) if report.passed => {}
[CODE L0128]             Some(report) => failures.extend(report.failures.iter().copied()),
[CODE L0129]             None => failures.push("missing_benchmark_report"),
[CODE L0130]         },
[CODE L0131]         TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
[CODE L0132]             if !state.should_advance_phase() {
[CODE L0133]                 failures.push("phase_budget_incomplete");
[CODE L0134]             }
[CODE L0135]             match validation_report {
[CODE L0136]                 Some(report) if report.passed => {}
[CODE L0137]                 Some(report) => failures.extend(report.failures.iter().copied()),
[CODE L0138]                 None => failures.push("missing_validation_report"),
[CODE L0139]             }
[CODE L0140]         }
[CODE L0141]         _ => {
[CODE L0142]             if !state.should_advance_phase() {
[CODE L0143]                 failures.push("phase_budget_incomplete");
[CODE L0144]             }
[CODE L0145]         }
[CODE L0146]     }
[CODE L0147]     GateReport {
[CODE L0148]         passed: failures.is_empty(),
[CODE L0149]         failures,
[CODE L0150]     }
[CODE L0151] }
[CODE L0152] 
[CODE L0153] pub fn maybe_advance_phase(state: &mut PipelineState, advance_report: &GateReport) -> bool {
[CODE L0154]     if advance_report.passed {
[CODE L0155]         state.advance_phase();
[CODE L0156]         true
[CODE L0157]     } else {
[CODE L0158]         false
[CODE L0159]     }
[CODE L0160] }
[CODE L0161] 
[CODE L0162] pub fn maintenance_plan(
[CODE L0163]     state: &PipelineState,
[CODE L0164]     rebase_tracker: &RebaseTracker,
[CODE L0165]     distill_state: &DistillState,
[CODE L0166]     distill_cfg: &DistillConfig,
[CODE L0167]     elapsed_secs: u64,
[CODE L0168]     max_distill_kl_drift: f32,
[CODE L0169] ) -> MaintenancePlan {
[CODE L0170]     let phase_progress = state.phase_progress();
[CODE L0171]     let shallow_exit_enabled = match state.phase {
[CODE L0172]         TrainingPhase::DrdaAchSelfPlay => phase_progress > 0.5,
[CODE L0173]         TrainingPhase::ExitPondering => true,
[CODE L0174]         _ => false,
[CODE L0175]     };
[CODE L0176]     let deep_exit_enabled = matches!(state.phase, TrainingPhase::ExitPondering);
[CODE L0177]     let should_rebase = matches!(
[CODE L0178]         state.phase,
[CODE L0179]         TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering
[CODE L0180]     ) && rebase_tracker.should_rebase();
[CODE L0181]     let should_distill = match state.phase {
[CODE L0182]         TrainingPhase::BenchmarkGates => false,
[CODE L0183]         TrainingPhase::BcWarmStart => state.should_advance_phase(),
[CODE L0184]         TrainingPhase::OracleGuiding => false,
[CODE L0185]         TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
[CODE L0186]             distill_state.should_distill(distill_cfg, elapsed_secs)
[CODE L0187]         }
[CODE L0188]     };
[CODE L0189] 
[CODE L0190]     MaintenancePlan {
[CODE L0191]         should_rebase,
[CODE L0192]         should_distill,
[CODE L0193]         distill_warning: distill_state.should_warn(max_distill_kl_drift),
[CODE L0194]         shallow_exit_enabled,
[CODE L0195]         deep_exit_enabled,
[CODE L0196]     }
[CODE L0197] }
[CODE L0198] 
[CODE L0199] /// Builds the live ExIt producer config from the current maintenance plan.
[CODE L0200] ///
[CODE L0201] /// Returns a [`LiveExitConfig`] with `enabled` set according to the plan's
[CODE L0202] /// exit flags. The producer remains default-off when neither shallow nor
[CODE L0203] /// deep exit is active.
[CODE L0204] pub fn live_exit_config_from_plan(plan: &MaintenancePlan) -> LiveExitConfig {
[CODE L0205]     LiveExitConfig {
[CODE L0206]         enabled: plan.shallow_exit_enabled || plan.deep_exit_enabled,
[CODE L0207]         exit_config: ExitConfig::default_phase3(),
[CODE L0208]     }
[CODE L0209] }
[CODE L0210] 
[CODE L0211] #[allow(clippy::too_many_arguments)]
[CODE L0212] pub fn supervised_phase_train_step<B: AutodiffBackend>(
[CODE L0213]     state: &PipelineState,
[CODE L0214]     model: HydraModel<B>,
[CODE L0215]     obs: Tensor<B, 3>,
[CODE L0216]     targets: &HydraTargets<B>,
[CODE L0217]     loss_fn: &HydraLoss<B>,
[CODE L0218]     optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
[CODE L0219]     oracle_cfg: &OracleGuidingConfig,
[CODE L0220]     step: usize,
[CODE L0221]     total_steps: usize,
[CODE L0222]     importance_weight: f32,
[CODE L0223]     max_importance_weight: f32,
[CODE L0224]     rng_values: &[f32],
[CODE L0225] ) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
[CODE L0226]     let batch_size = obs.dims()[0];
[CODE L0227]     let device = obs.device();
[CODE L0228]     let empty_batch = MjaiBatch {
[CODE L0229]         obs: Tensor::zeros([batch_size, crate::config::INPUT_CHANNELS, 34], &device),
[CODE L0230]         actions: Tensor::zeros([batch_size], &device),
[CODE L0231]         legal_mask: targets.legal_mask.clone(),
[CODE L0232]         value_target: targets.value_target.clone(),
[CODE L0233]         grp_target: targets.grp_target.clone(),
[CODE L0234]         oracle_target: targets.oracle_target.clone(),
[CODE L0235]         oracle_target_mask: targets
[CODE L0236]             .oracle_guidance_mask
[CODE L0237]             .clone()
[CODE L0238]             .unwrap_or_else(|| Tensor::zeros([batch_size], &device)),
[CODE L0239]         tenpai_target: targets.tenpai_target.clone(),
[CODE L0240]         danger_target: targets.danger_target.clone(),
[CODE L0241]         danger_mask: targets.danger_mask.clone(),
[CODE L0242]         safety_residual_target: targets.safety_residual_target.clone(),
[CODE L0243]         safety_residual_mask: targets.safety_residual_mask.clone(),
[CODE L0244]         exit_target: None,
[CODE L0245]         exit_mask: None,
[CODE L0246]         belief_fields_target: targets.belief_fields_target.clone(),
[CODE L0247]         mixture_weight_target: targets.mixture_weight_target.clone(),
[CODE L0248]         delta_q_target: targets.delta_q_target.clone(),
[CODE L0249]         delta_q_mask: targets.delta_q_mask.clone(),
[CODE L0250]         belief_fields_mask: targets.belief_fields_mask.clone(),
[CODE L0251]         mixture_weight_mask: targets.mixture_weight_mask.clone(),
[CODE L0252]         opp_next_target: targets.opp_next_target.clone(),
[CODE L0253]         score_pdf_target: targets.score_pdf_target.clone(),
[CODE L0254]         score_cdf_target: targets.score_cdf_target.clone(),
[CODE L0255]         target_presence: targets.target_presence.clone(),
[CODE L0256]     };
[CODE L0257]     match state.phase {
[CODE L0258]         TrainingPhase::BenchmarkGates => Ok((
[CODE L0259]             model,
[CODE L0260]             PhaseTrainReport {
```

## Artifact 13 — Training mode and promotion-mode surfaces
Artifact id: `train-modes`
Source label: CODE
Type: `file_range`
Source: `crates/hydra-train/src/bin/train/modes.rs:267-499`
Why it matters: Shows how Hydra currently exposes RL training and a promotion-gated experimental lane (DeltaQ), which is a strong pattern candidate for any DCRL-style follow-up lane.

```rust
[CODE L0267] pub(super) fn handle_training_mode(
[CODE L0268]     config_path: &std::path::Path,
[CODE L0269]     config: TrainConfig,
[CODE L0270] ) -> Result<(), String> {
[CODE L0271]     println!(
[CODE L0272]         "{}",
[CODE L0273]         format_warning_line(explicit_preflight_recommendation())
[CODE L0274]     );
[CODE L0275]     if let Some(rl_cfg) = config.rl.clone() {
[CODE L0276]         if matches!(
[CODE L0277]             config.precision_mode,
[CODE L0278]             crate::config::PrecisionMode::Bf16Autocast
[CODE L0279]         ) {
[CODE L0280]             return Err(
[CODE L0281]                 "precision_mode=bf16_autocast is not supported for RL training yet".to_string(),
[CODE L0282]             );
[CODE L0283]         }
[CODE L0284]         let (bootstrap, runtime) = initialize_rl_training_bootstrap(config_path, config, rl_cfg)?;
[CODE L0285]         let RlTrainingBootstrap {
[CODE L0286]             config: _,
[CODE L0287]             rl_config,
[CODE L0288]             artifacts,
[CODE L0289]             model_config,
[CODE L0290]             device_name,
[CODE L0291]             ..
[CODE L0292]         } = &bootstrap;
[CODE L0293]         println!(
[CODE L0294]             "{}",
[CODE L0295]             timestamped(format!(
[CODE L0296]                 "{} mode=rl phase={:?} games_per_batch={} device={} artifacts={} model={} ",
[CODE L0297]                 "Hydra RL training".bold().cyan(),
[CODE L0298]                 rl_config.phase,
[CODE L0299]                 rl_config.games_per_batch,
[CODE L0300]                 device_name,
[CODE L0301]                 artifacts.root.display(),
[CODE L0302]                 if model_config.is_learner() {
[CODE L0303]                     "learner"
[CODE L0304]                 } else {
[CODE L0305]                     "actor"
[CODE L0306]                 },
[CODE L0307]             ))
[CODE L0308]         );
[CODE L0309]         let _runtime: RlTrainingRuntime = runtime;
[CODE L0310]         return run_rl_training_loop(bootstrap, _runtime);
[CODE L0311]     }
[CODE L0312]     match config.precision_mode {
[CODE L0313]         crate::config::PrecisionMode::Fp32 => {
[CODE L0314]             let (bootstrap, runtime) = initialize_training_bootstrap(config_path, config)?;
[CODE L0315]             run_bc_training_mode_for_backend::<TrainBackend>(bootstrap, runtime)
[CODE L0316]         }
[CODE L0317]         crate::config::PrecisionMode::Bf16Autocast => {
[CODE L0318]             let (bootstrap, runtime) = initialize_training_bootstrap_bf16(config_path, config)?;
[CODE L0319]             run_bc_training_mode_for_backend::<Bf16TrainBackend>(bootstrap, runtime)
[CODE L0320]         }
[CODE L0321]     }
[CODE L0322] }
[CODE L0323] 
[CODE L0324] pub(super) fn handle_delta_q_promotion_mode(
[CODE L0325]     config_path: &std::path::Path,
[CODE L0326]     config: TrainConfig,
[CODE L0327]     baseline_checkpoint: Option<PathBuf>,
[CODE L0328] ) -> Result<(), String> {
[CODE L0329]     if matches!(
[CODE L0330]         config.precision_mode,
[CODE L0331]         crate::config::PrecisionMode::Bf16Autocast
[CODE L0332]     ) {
[CODE L0333]         return Err(
[CODE L0334]             "precision_mode=bf16_autocast is not supported for delta_q promotion yet".to_string(),
[CODE L0335]         );
[CODE L0336]     }
[CODE L0337]     let (bootstrap, runtime) = initialize_training_bootstrap(config_path, config)?;
[CODE L0338]     let TrainingBootstrap {
[CODE L0339]         config,
[CODE L0340]         artifacts,
[CODE L0341]         loader_config,
[CODE L0342]         manifest,
[CODE L0343]         model_config,
[CODE L0344]         device_name,
[CODE L0345]         train_device,
[CODE L0346]         valid_loss_fn,
[CODE L0347]         bc_exit_cfg,
[CODE L0348]         ..
[CODE L0349]     } = bootstrap;
[CODE L0350]     let TrainingRuntime {
[CODE L0351]         model,
[CODE L0352]         mut head_controller,
[CODE L0353]         ..
[CODE L0354]     } = runtime;
[CODE L0355]     let baseline_checkpoint = baseline_checkpoint.as_ref().ok_or_else(|| {
[CODE L0356]         "delta_q promotion mode requires --delta-q-baseline-checkpoint for arena confirmation"
[CODE L0357]             .to_string()
[CODE L0358]     })?;
[CODE L0359]     let checkpoint_base = checkpoint_base_from_path(baseline_checkpoint);
[CODE L0360]     let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
[CODE L0361]     let baseline_model = HydraModelConfig::learner()
[CODE L0362]         .init::<super::TrainBackend>(&train_device)
[CODE L0363]         .load_file(&checkpoint_base, &recorder, &train_device)
[CODE L0364]         .map_err(|err| {
[CODE L0365]             format!(
[CODE L0366]                 "failed to load delta_q baseline checkpoint {}: {err}",
[CODE L0367]                 checkpoint_base.display()
[CODE L0368]             )
[CODE L0369]         })?;
[CODE L0370] 
[CODE L0371]     println!(
[CODE L0372]         "{}",
[CODE L0373]         timestamped(format!(
[CODE L0374]             "{} device={} artifacts={} model={}",
[CODE L0375]             "Hydra DeltaQ offline/transfer gate".bold().cyan(),
[CODE L0376]             device_name,
[CODE L0377]             artifacts.root.display(),
[CODE L0378]             if model_config.is_learner() {
[CODE L0379]                 "learner"
[CODE L0380]             } else {
[CODE L0381]                 "actor"
[CODE L0382]             },
[CODE L0383]         ))
[CODE L0384]     );
[CODE L0385] 
[CODE L0386]     let summary = run_validation_with_policy_baseline(
[CODE L0387]         &model,
[CODE L0388]         &baseline_model,
[CODE L0389]         ValidationContext {
[CODE L0390]             config: &config,
[CODE L0391]             loader_config: &loader_config,
[CODE L0392]             manifest: &manifest,
[CODE L0393]             cached_samples: None,
[CODE L0394]             device: &train_device,
[CODE L0395]             loss_fn: &valid_loss_fn,
[CODE L0396]             exit_cfg: &bc_exit_cfg,
[CODE L0397]         },
[CODE L0398]         ValidationRuntime {
[CODE L0399]             head_controller: Some(&mut head_controller),
[CODE L0400]             progress: None,
[CODE L0401]         },
[CODE L0402]     )?;
[CODE L0403] 
[CODE L0404]     let (Some(report), Some(result), Some(snapshot), transfer_result) = (
[CODE L0405]         summary.delta_q_promotion.as_ref(),
[CODE L0406]         summary.delta_q_promotion_result.as_ref(),
[CODE L0407]         summary.delta_q_promotion_snapshot,
[CODE L0408]         summary.delta_q_policy_transfer_result.as_ref(),
[CODE L0409]     ) else {
[CODE L0410]         return Err(
[CODE L0411]             "delta_q promotion mode requires active delta_q targets in validation batches"
[CODE L0412]                 .to_string(),
[CODE L0413]         );
[CODE L0414]     };
[CODE L0415]     let pre_arena_recommendation =
[CODE L0416]         pre_arena_recommendation(result.passed, transfer_result.map(|r| r.passed));
[CODE L0417] 
[CODE L0418]     let arena_confirmation_request = default_arena_confirmation_request(pre_arena_recommendation);
[CODE L0419]     let arena_config = arena_confirmation_request.as_ref().map(|request| {
[CODE L0420]         PairedArenaEvalConfig::new()
[CODE L0421]             .with_min_games(request.min_games as usize)
[CODE L0422]             .with_seed(config.seed)
[CODE L0423]             .with_same_seeds(request.same_seeds)
[CODE L0424]             .with_same_seat_rotation_schedule(request.same_seat_rotation_schedule)
[CODE L0425]             .with_same_search_budget(request.same_search_budget)
[CODE L0426]             .with_same_temperature(request.same_temperature)
[CODE L0427]             .with_same_frozen_opponent_pool(request.same_frozen_opponent_pool)
[CODE L0428]     });
[CODE L0429]     let arena_eval = arena_config.as_ref().map(|arena_config| {
[CODE L0430]         run_paired_delta_q_arena_confirmation(
[CODE L0431]             &model,
[CODE L0432]             &baseline_model,
[CODE L0433]             &train_device,
[CODE L0434]             arena_config,
[CODE L0435]             config.rl.as_ref().map(|rl| rl.temperature).unwrap_or(1.0),
[CODE L0436]         )
[CODE L0437]     });
[CODE L0438]     let arena_report = arena_eval.as_ref().map(|outcome| {
[CODE L0439]         DeltaQArenaReport::from_paired_eval(
[CODE L0440]             &outcome.paired_result,
[CODE L0441]             outcome.lower_confidence_bound_mean_placement,
[CODE L0442]         )
[CODE L0443]     });
[CODE L0444]     let arena_decision = arena_eval.as_ref().map(|outcome| {
[CODE L0445]         outcome.paired_result.recommendation(
[CODE L0446]             arena_config
[CODE L0447]                 .as_ref()
[CODE L0448]                 .expect("arena config exists when arena eval exists"),
[CODE L0449]         )
[CODE L0450]     });
[CODE L0451] 
[CODE L0452]     write_delta_q_promotion_artifact(
[CODE L0453]         &artifacts.delta_q_promotion_path,
[CODE L0454]         &PersistedDeltaQPromotionArtifact {
[CODE L0455]             scope: "promotion_mode",
[CODE L0456]             step_or_epoch: 0,
[CODE L0457]             recommendation: pre_arena_recommendation,
[CODE L0458]             stage: delta_q_promotion_stage(arena_report.is_some()),
[CODE L0459]             arena_confirmation: arena_confirmation_request.clone(),
[CODE L0460]             arena_decision,
[CODE L0461]             arena_report: arena_report.as_ref(),
[CODE L0462]             report,
[CODE L0463]             result,
[CODE L0464]             policy_transfer: summary.delta_q_policy_transfer.as_ref(),
[CODE L0465]             policy_transfer_result: transfer_result,
[CODE L0466]         },
[CODE L0467]     )?;
[CODE L0468] 
[CODE L0469]     println!(
[CODE L0470]         "{}",
[CODE L0471]         format_delta_q_offline_gate_message(
[CODE L0472]             summary.samples,
[CODE L0473]             snapshot,
[CODE L0474]             pre_arena_recommendation,
[CODE L0475]             &delta_q_arena_requirement_summary(arena_confirmation_request.as_ref()),
[CODE L0476]             &artifacts.delta_q_promotion_path,
[CODE L0477]         )
[CODE L0478]     );
[CODE L0479]     if let Some(outcome) = arena_eval.as_ref() {
[CODE L0480]         println!(
[CODE L0481]             "{}",
[CODE L0482]             timestamped(format!(
[CODE L0483]                 "{} {} lower_ci={:.3}",
[CODE L0484]                 "DeltaQ arena confirmation".bold().green(),
[CODE L0485]                 outcome.paired_result.summary(
[CODE L0486]                     arena_config
[CODE L0487]                         .as_ref()
[CODE L0488]                         .expect("arena config exists when arena eval exists"),
[CODE L0489]                 ),
[CODE L0490]                 outcome.lower_confidence_bound_mean_placement,
[CODE L0491]             ))
[CODE L0492]         );
[CODE L0493]         if let Some(decision) = arena_decision {
[CODE L0494]             println!(
[CODE L0495]                 "{}",
[CODE L0496]                 timestamped(format!(
[CODE L0497]                     "{} {}",
[CODE L0498]                     "DeltaQ arena decision".bold().green(),
[CODE L0499]                     decision.summary(),
```

## Artifact 14 — Opponent modeling rationale surface
Artifact id: `opponent-modeling`
Source label: DESIGN
Type: `file_range`
Source: `research/design/OPPONENT_MODELING.md:1-140`
Why it matters: Useful context for whether any DCRL-style lane should attach to Hydra's actual differentiator stack rather than float as an abstract RL trick.

```markdown
[DESIGN L0001] # Hydra Opponent Modeling
[DESIGN L0002] 
[DESIGN L0003] Opponent modeling is Hydra's primary differentiator from existing Mahjong AIs. This document covers every aspect of how Hydra reads opponents — from explicit safety plane encoding through auxiliary prediction heads to implicit learning via oracle distillation.
[DESIGN L0004] 
[DESIGN L0005] ---
[DESIGN L0006] 
[DESIGN L0007] ## 1. The Problem: Why Current AIs Fail at Opponent Modeling
[DESIGN L0008] 
[DESIGN L0009] ### Mortal's Blind Spot
[DESIGN L0010] 
[DESIGN L0011] > **Ownership note:** This document is the detailed rationale/reference surface for opponent modeling. For active-path priority, use `HYDRA_RECONCILIATION.md`. For what is actually shipped today, use `docs/CURRENT_STATUS.md`. For live runtime/channel truth, use `docs/GAME_ENGINE.md` and current code.
[DESIGN L0012] 
[DESIGN L0013] > Reserve/future ideas are kept here as rationale, but they do not become active just because they are described in this file.
[DESIGN L0014] 
[DESIGN L0015] Mortal uses `SinglePlayerTables` for EV calculation, assuming no opponent interaction. There are no safety features (suji, kabe, genbutsu) pre-computed, no opponent tenpai estimation, and no aggression or tendency profiling. The network must learn all opponent-relevant patterns implicitly through raw observation channels — and the evidence shows it fails at the hardest cases.
[DESIGN L0016] 
[DESIGN L0017] ```mermaid
[DESIGN L0018] graph LR
[DESIGN L0019]     subgraph "Mortal's Approach"
[DESIGN L0020]         M_OBS[Observation] --> M_NET[Network]
[DESIGN L0021]         M_NET --> M_POL[Policy]
[DESIGN L0022]         M_NOTE["No explicit opponent model<br/>Uses SinglePlayerTables"]
[DESIGN L0023]     end
[DESIGN L0024] 
[DESIGN L0025]     subgraph "What's Missing"
[DESIGN L0026]         MISS1["Tenpai probability estimation"]
[DESIGN L0027]         MISS2["Danger level per tile"]
[DESIGN L0028]         MISS3["Opponent wait prediction"]
[DESIGN L0029]         MISS4["Damaten detection"]
[DESIGN L0030]         MISS5["Threat severity (hand value)"]
[DESIGN L0031]         MISS6["Yaku-plan / call-intent reading"]
[DESIGN L0032]     end
[DESIGN L0033] ```
[DESIGN L0034] 
[DESIGN L0035] ### Evidence from the Community
[DESIGN L0036] 
[DESIGN L0037] **Damaten detection failures** are Mortal's most cited weakness in the Japanese mahjong community. Community reports confirm Mortal frequently deals into obvious damaten (silent tenpai) hands because it has no explicit tenpai detection mechanism. The AI relies entirely on explicit signals like riichi declarations and open melds — when an opponent reaches tenpai silently, Mortal has no mechanism to detect the increased danger.
[DESIGN L0038] 
[DESIGN L0039] Specific documented issues:
[DESIGN L0040] 
[DESIGN L0041] - **GitHub Issue #111** — Overtake score miscalculation; Mortal plays too safe when trailing, missing opportunities to overtake, partly because it cannot read opponent hand danger accurately.
[DESIGN L0042] - **GitHub Discussion #102** — Equim-chan (Mortal's creator) confirmed that oracle guiding "didn't bring improvements in practice" and was removed in v3, replaced with a next-rank prediction auxiliary task (implemented as `AuxNet` in `mortal/model.py`; rationale in Discussion #52). This suggests Mortal's architecture may not be structured to benefit from opponent-aware signals.
[DESIGN L0043] 
[DESIGN L0044] **Community-identified weaknesses related to opponent reading:**
[DESIGN L0045] 
[DESIGN L0046] 1. **Early riichi push errors** — Underestimates the threat of early (turn 1–6) riichi, pushing with sub-optimal hands against unknown waits.
[DESIGN L0047] 2. **Damaten detection failures** — No intent reading for silent tenpai. Relies on explicit signals (riichi, melds). Deals into high-value silent hands.
[DESIGN L0048] 3. **Coarse placement sensitivity** — Same playstyle regardless of point spread; doesn't adjust aggression based on how dangerous opponents are.
[DESIGN L0049] 4. **場況 (bakyou) blindness** — Struggles with field status and table flow reading, as noted in Japanese mahjong blogs on Note.com and Reddit r/Mahjong.
[DESIGN L0050] 
[DESIGN L0051] ### What Hydra Adds
[DESIGN L0052] 
[DESIGN L0053] Hydra addresses the opponent modeling gap through a mix of implemented rationale and later extensions:
[DESIGN L0054] 
[DESIGN L0055] 1. **Explicit Safety Planes** — core Hydra opponent-read rationale with shipped baseline encoding support
[DESIGN L0056] 2. **Tenpai Predictor Head** — rationale for opponent-readiness prediction
[DESIGN L0057] 3. **Danger Head** — rationale for per-tile defensive modeling
[DESIGN L0058] 4. **Value-Conditioned Tenpai** — later extension unless promoted by current doctrine (§ 3.7)
[DESIGN L0059] 5. **Wait-Set Belief Head** — later extension unless promoted by current doctrine (§ 4.6)
[DESIGN L0060] 6. **Call-Intent Head** — later extension unless promoted by current doctrine (§ 4.7)
[DESIGN L0061] 7. **Oracle Distillation** — later extension; implementation priority still comes from `HYDRA_RECONCILIATION.md`
[DESIGN L0062] 
[DESIGN L0063] ---
[DESIGN L0064] 
[DESIGN L0065] ## 2. Safety Planes: Explicit Defensive Encoding
[DESIGN L0066] 
[DESIGN L0067] > For the live channel-level summary, see [docs/GAME_ENGINE.md § Baseline Prefix Channel Layout](../../docs/GAME_ENGINE.md#baseline-prefix-channel-layout-channels-0-84) and the safety-system section there. This file provides the deeper design rationale and encoding logic.
[DESIGN L0068] 
[DESIGN L0069] Hydra dedicates 23 input channels (channels 62–84) to safety information — a novel addition absent from Mortal's 1012-channel encoding. These planes pre-compute traditional Japanese mahjong defensive concepts, giving the network structured safety data rather than forcing it to rediscover these patterns implicitly.
[DESIGN L0070] 
[DESIGN L0071] > **Quantitative basis:** The 23-channel safety encoding is grounded in mahjong theory (genbutsu, suji, kabe are the foundation of all human defensive play) but the specific channel design (9 genbutsu, 9 suji, 2 kabe, 3 tenpai hints) and encoding choices (suji float values, 3 sub-channels per genbutsu opponent) are based on domain analysis, not empirical ablation. Mortal achieves ~11% deal-in rate without any explicit safety planes, relying on implicit learning from 1012 raw channels. Whether pre-computed safety planes improve over implicit learning is an open empirical question that should be evaluated through the active ablation/testing process rather than the now-missing historical ablation-plan doc. The safety plane design will be validated or revised based on those results. The conservative channel counts (9+9+2+3=23) were chosen to minimize parameter overhead (~0% increase to backbone) while covering the complete human defensive vocabulary.
[DESIGN L0072] 
[DESIGN L0073] ### 2.1 Genbutsu (絶対安全牌) — Channels 62–70
[DESIGN L0074] 
[DESIGN L0075] **Definition:** Tiles that are 100% safe against a specific riichi player. Any tile discarded by the riichi player after their riichi declaration is genbutsu — they cannot win on a tile they themselves threw after declaring riichi.
[DESIGN L0076] 
[DESIGN L0077] **Encoding:** 9 binary channels, 3 per opponent. The 3 channels per opponent encode three semantically distinct safety signals:
[DESIGN L0078] 
[DESIGN L0079] | Sub-channel | Content | Encoding |
[DESIGN L0080] |-------------|---------|----------|
[DESIGN L0081] | +0 | All genbutsu | Binary mask: 1 if tile is 100% safe against this opponent. Union of discard-furiten genbutsu (tile in their river) and riichi-furiten genbutsu (tile discarded by anyone after their riichi, not ron'd). |
[DESIGN L0082] | +1 | Tedashi genbutsu | Binary mask: subset of +0 where tile was specifically hand-discarded (tedashi) by this opponent, not tsumogiri. Carries hand-shape information — tedashi implies the opponent evaluated and rejected this tile. |
[DESIGN L0083] | +2 | Riichi-era genbutsu | Binary mask: subset of +0 where tile became safe AFTER this opponent declared riichi (any player's post-riichi discard). Only non-zero when opponent is in riichi. Separates pre-riichi safety (mutable hand) from post-riichi safety (locked hand). |
[DESIGN L0084] 
[DESIGN L0085] **Calculation:**
[DESIGN L0086] 
[DESIGN L0087] ```mermaid
[DESIGN L0088] graph TB
[DESIGN L0089]     subgraph "Genbutsu Sub-Channel Calculation"
[DESIGN L0090]         DISCARD[Opponent discards tile] --> ALL["+0: All genbutsu<br/>Mark tile safe"]
[DESIGN L0091]         DISCARD --> CHECK_TEDASHI{Tedashi?}
[DESIGN L0092]         CHECK_TEDASHI -->|Yes| TEDASHI["+1: Tedashi genbutsu<br/>Hand-discarded → hand-shape info"]
[DESIGN L0093]         CHECK_TEDASHI -->|No| SKIP_T["Skip +1<br/>(tsumogiri — no hand info)"]
[DESIGN L0094]         RIICHI[Opponent declares riichi] --> POST[Track all post-riichi discards]
[DESIGN L0095]         POST --> ERA["+2: Riichi-era genbutsu<br/>Post-riichi safety (locked hand)"]
[DESIGN L0096]     end
[DESIGN L0097] ```
[DESIGN L0098] 
[DESIGN L0099] **Why 3 channels per opponent, not 1:** While genbutsu is binary (safe or not), the sub-channel decomposition provides the network with pre-computed hand-reading signals. Tedashi genbutsu reveals which tiles the opponent actively rejected from their hand (matagi-suji and sotogawa inferences follow). Riichi-era genbutsu separates the temporal regime where the opponent's hand is locked. This mirrors Mortal v4's 3-channel kawa summary (all discards / tedashi-only / riichi-tile) but pre-computes the safety derivation. No existing mahjong AI pre-computes genbutsu channels — Mortal, Suphx, and Mjx all rely on the network to derive safety from raw discard data. Hydra's explicit encoding is a deliberate advantage.
[DESIGN L0100] 
[DESIGN L0101] ### 2.2 Suji (筋) — Channels 71–79
[DESIGN L0102] 
[DESIGN L0103] **Definition:** Probabilistic safety based on ryanmen (two-sided) wait patterns. When an opponent discards a tile, certain numerically related tiles become safer because common wait patterns involving the discarded tile become less likely.
[DESIGN L0104] 
[DESIGN L0105] **Logic:** In ryanmen waits, tiles are linked in 1-4-7, 2-5-8, and 3-6-9 sequences. If a player discards one tile in a sequence, the paired tiles at the opposite end become safer.
[DESIGN L0106] 
[DESIGN L0107] **Suji Logic Table:**
[DESIGN L0108] 
[DESIGN L0109] | Discarded Tile | Safer Tiles | Reasoning |
[DESIGN L0110] |----------------|-------------|-----------|
[DESIGN L0111] | 1 or 4 | 7 | No 4-7 ryanmen wait |
[DESIGN L0112] | 2 or 5 | 8 | No 5-8 ryanmen wait |
[DESIGN L0113] | 3 or 6 | 9 | No 6-9 ryanmen wait |
[DESIGN L0114] | 4 or 7 | 1 | No 1-4 ryanmen wait |
[DESIGN L0115] | 5 or 8 | 2 | No 2-5 ryanmen wait |
[DESIGN L0116] | 6 or 9 | 3 | No 3-6 ryanmen wait |
[DESIGN L0117] 
[DESIGN L0118] **Half-suji vs Full-suji:** Half-suji means only one side of the sequence has been discarded. Full-suji means both sides are visible, providing stronger safety.
[DESIGN L0119] 
[DESIGN L0120] **Encoding:** 9 float channels, 3 per opponent (one channel per suit: manzu, pinzu, souzu). Per-tile suji coverage is computed as follows:
[DESIGN L0121] 
[DESIGN L0122] For each opponent and each numbered tile (1–9) in each suit, count how many of that tile's suji pairs have been discarded by that opponent:
[DESIGN L0123] 
[DESIGN L0124] | Suji pairs for tile | Pairs | Coverage value |
[DESIGN L0125] |---------------------|-------|----------------|
[DESIGN L0126] | 1 | (4) | 0.0 if neither 4 discarded; 1.0 if 4 discarded |
[DESIGN L0127] | 2 | (5) | 0.0 if neither 5 discarded; 1.0 if 5 discarded |
[DESIGN L0128] | 3 | (6) | 0.0 if neither 6 discarded; 1.0 if 6 discarded |
[DESIGN L0129] | 4 | (1, 7) | 0.0 / 0.5 / 1.0 for 0 / 1 / 2 partners discarded |
[DESIGN L0130] | 5 | (2, 8) | 0.0 / 0.5 / 1.0 for 0 / 1 / 2 partners discarded |
[DESIGN L0131] | 6 | (3, 9) | 0.0 / 0.5 / 1.0 for 0 / 1 / 2 partners discarded |
[DESIGN L0132] | 7 | (4) | 0.0 if neither 4 discarded; 1.0 if 4 discarded |
[DESIGN L0133] | 8 | (5) | 0.0 if neither 5 discarded; 1.0 if 5 discarded |
[DESIGN L0134] | 9 | (6) | 0.0 if neither 6 discarded; 1.0 if 6 discarded |
[DESIGN L0135] 
[DESIGN L0136] Tiles 1/2/3/7/8/9 have only one suji partner, so they are binary (0.0 or 1.0). Tiles 4/5/6 have two partners, so they use half-suji = 0.5 when one partner is discarded. Honor tiles always have suji value 0.0 (no suji relationship).
[DESIGN L0137] 
[DESIGN L0138] > **Design note:** This encoding is purely structural (based on which tiles are in the discard pile) with no temporal decay — suji status does not change over time. The backbone learns to weight suji evidence against other signals (kabe, genbutsu, tedashi patterns). Mortal has no explicit suji channels; Hydra's explicit encoding is a hypothesis that pre-computed suji accelerates safety learning and should be tested through the active ablation/testing workflow.
[DESIGN L0139] 
[DESIGN L0140] **Caveats — Suji is NOT 100% safe:**
```

</artifacts>
