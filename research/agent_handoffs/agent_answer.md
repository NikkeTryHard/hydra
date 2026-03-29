## Verdict

**docs-worth-it:** **Yes, lightly.** Hydra should explicitly position its oracle-critic / CTDE surfaces against the asymmetric-critic lineage, but the right patch is small: cite **UAAC** and **DCRL** (and optionally the earlier asymmetric actor-critic paper), then state clearly that Hydra’s project-defining lane is still **public-state ExIt/search-as-feature for Mahjong**, not “dual critic” as a standalone novelty. **Confidence: medium-high.**

**method-worth-it:** **No material reaction now.** No architecture reprioritization, no new reserve lane, no roadmap reshuffle. At most, keep a **future ablation note** for a DCRL-style mixed-advantage critic only if Hydra’s actor-critic RL lane becomes active and empirically unstable. **Confidence: high.**

The reason these diverge is simple: Hydra already has real oracle-critic and CTDE-adjacent surfaces, but its promoted method identity is centered on **ExIt + search-as-feature + Mahjong-specific belief/Hand-EV/search plumbing**, and its active roadmap explicitly says not to broaden architecture before the baseline loop is alive. ([GitHub][1])

## What the repo already proves

**Artifact-supported repo reality:** Hydra is not merely “thinking about” oracle/search surfaces. Public docs show a live **192x34** encoder with **Group C search/belief** and **Group D Hand-EV** planes; current status marks the stronger belief tranche, Hand-EV realism, `safety_residual`, and an end-to-end **ExIt carrier** as shipped; **DeltaQ** is implemented but not default-on; and broader public-belief search remains reserve-shelf. README-level routing also foregrounds opponent modeling and inference-time search as the long-run differentiators. ([GitHub][2])

**Code-level proof:** the model surface already includes both a public `value` head and an `oracle_critic`, plus `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`. The oracle critic is a distinct **4-output** head, it is fed a **detached** backbone tensor, oracle-only loss is tested not to backprop through the shared backbone, and the default loss weights for oracle and advanced heads are **0.0** in the exposed loss config. That proves the oracle lane is structurally real, but also that it is currently treated as an **auxiliary / gated** surface rather than a default project identity. ([GitHub][3])

**Doctrine-level proof:** Hydra’s architecture doc makes two important commitments at once:

1. perfect-information networks are allowed for **variance reduction and diagnostics**, but
2. the improvement operator for the deployable policy should respect the **public/information state**.
   That is stricter than “samplewise hidden-state critic is unbiased in expectation,” and it matters for the DCRL comparison. Hydra also explicitly says the oracle critic supplies advantages via **CTDE**, while the actor conditions on public info only. ([GitHub][1])

## Overlap with DCRL: real vs superficial

| Dimension                                                  | DCRL / adjacent prior art | Hydra                                      | Verdict                     |
| ---------------------------------------------------------- | ------------------------- | ------------------------------------------ | --------------------------- |
| Privileged critic during training                          | Yes                       | Yes                                        | **Real overlap**            |
| Actor deploys without privileged state                     | Yes                       | Yes                                        | **Real overlap**            |
| Standard partial/history critic + oracle critic            | DCRL: core mechanism      | Hydra: surfaces exist                      | **Real overlap**            |
| DCRL-style mixed actor advantage is the main update object | Yes                       | Not shown as active project identity       | **Not established overlap** |
| Public-state search / ExIt teacher object                  | No                        | Central Hydra doctrine                     | **Major divergence**        |
| Search-as-feature, belief planes, Hand-EV, DeltaQ          | No                        | Core Hydra surfaces                        | **Major divergence**        |
| Mahjong, 4-player, rank-aware objective                    | No                        | Yes                                        | **Major divergence**        |
| “dual critic” as generic naming similarity                 | N/A                       | Many heads / 2-tier nets / auxiliary lanes | **Mostly superficial**      |

The most important distinction is **training trust object**. DCRL’s main idea is still an actor-critic update built from a weighted combination of oracle and standard advantages. Hydra’s stated north star is different: **search-generated public-state teacher targets** are the central engine, while privileged critics are auxiliary CTDE-style tools. So the overlap is real, but narrow. ([NeurIPS Proceedings][4])

## Which external sources actually move the answer

**1) UAAC matters more than DCRL for literature positioning.**
UAAC is the paper that established the theoretically sound **history+state oracle critic** under partial observability and explicitly contrasted it with biased state-only asymmetric critics. If Hydra mentions DCRL, it should almost certainly mention **UAAC** too, because DCRL is better read as a variance-reduced refinement of that lineage than as the root prior-art family. ([IFAAMAS][5])

**2) DCRL matters, but as a narrow extension.**
DCRL’s primary contribution is: keep a **standard critic** and an **oracle critic**, combine them with a weighting/gating scheme, and retain unbiasedness while reducing variance under partial observability. That is relevant to Hydra’s oracle-critic lane, but it does **not** address Hydra’s central differentiators: Mahjong-specific belief/search integration, ExIt/search targets, or 4-player rank-aware decision making. Its published evaluation is on partial-observation control benchmarks, not Mahjong or multiplayer imperfect-information search. ([NeurIPS Proceedings][4])

**3) Same-domain Mahjong work is still more important for Hydra than DCRL.**
Hydra’s own docs already call **RVR** “the most directly relevant work,” specifically because it is same-domain Mahjong, uses an oracle/relative-value network, enforces a zero-sum constraint, and adds an expected reward network for variance reduction. The RVR paper itself describes a relative value network using global information plus an expected reward network to stabilize Mahjong RL. That is simply more on-point for Hydra than DCRL. ([GitHub][6])

**4) The older asymmetric-critic root should also be acknowledged.**
The 2018 asymmetric actor-critic paper is the clean historical root for “actor on partial observations, critic on full state during training only.” If Hydra wants one umbrella sentence, the clean lineage is **AACC/AAC → UAAC → DCRL**, with **RVR/Suphx** as the Mahjong-specific branch. ([Robotics Proceedings][7])

**5) Hydra already foregrounds stronger adjacent families.**
Its own reference/comparison surfaces emphasize **RVR**, **ACH**, **OLSS**, **ExIt**, and **oracle distillation** as the most actionable or central ideas. The training-paradigms document literally says **ExIt** and **asymmetric oracle training** are the two most actionable paradigms for Hydra, and treats oracle distillation as already being Hydra’s Phase 2 concept. That lowers the method-level importance of DCRL further. ([GitHub][8])

## Decision-ready recommendation

### Docs-worth-it: **Yes, but tiny**

The omission is worth fixing because Hydra publicly discusses oracle critics / CTDE / partial observability, and missing the **UAAC/DCRL** branch creates an avoidable “nearby prior art not acknowledged” hole. But this is a **positioning fix**, not a re-architecture.

### Smallest justified docs patch set

**Minimum patch**

1. **`research/intel/REFERENCES.md`**
   Add entries for:

   * **Asymmetric Actor-Critic for Image-Based Robot Learning** (2018)
   * **Unbiased Asymmetric Reinforcement Learning under Partial Observability** (UAAC, 2022)
   * **Dual Critic Reinforcement Learning under Partial Observability** (DCRL, 2024)

2. **`research/design/HYDRA_FINAL.md`**
   Add one short prior-art note near **P2** or **Phase 1 oracle supervision**:

   > Prior-art positioning: Hydra’s oracle-critic / CTDE surfaces are adjacent to the asymmetric-critic lineage (Pinto et al. 2018; Baisero & Amato 2022; Li et al. 2024). Hydra does not claim privileged critics as a standalone novelty. The promoted delta is the combination with Mahjong-specific ExIt/search-as-feature, belief/search surfaces, Hand-EV features, and 4-player rank-aware training.

3. **Optional README sentence**
   Only if you want the public front door to inoculate against shallow critique:

   > Hydra’s oracle-critic lane sits in the asymmetric-critic line (AACC / UAAC / DCRL), while Hydra’s main architecture identity is public-state ExIt/search-as-feature for Mahjong.

**Do not** edit `HYDRA_RECONCILIATION.md` unless you also change roadmap priority. Right now, you should not.

## Method-worth-it: **No active reaction**

Hydra’s roadmap explicitly says:

* baseline first,
* close loops before expanding architecture,
* search-strength lanes stay selective,
* broader identity debates should not outrank the live training loop. ([GitHub][9])

So DCRL does **not** justify:

* changing architecture priorities,
* adding a new reserve lane now,
* bumping DeltaQ/search-strength work,
* or rewriting Hydra’s method story around “dual critics.”

### Exact no-op rationale

1. **Hydra already tracks more direct prior art.**
   For Hydra’s actual domain and ambitions, **RVR / Suphx / ACH / OLSS / ExIt** are more direct than DCRL. ([GitHub][6])

2. **Hydra’s main trust object is different.**
   DCRL’s center of gravity is a dual-critic actor-critic update. Hydra’s is a **public-state ExIt/search teacher**. DCRL can only touch a secondary lane. ([GitHub][1])

3. **Current code exposes an oracle surface, but as an auxiliary, not a promoted shared training identity.**
   Detached oracle input, optional oracle loss, default zero oracle weight, and auxiliary-head gating all point to “use carefully when activated,” not “make this the project’s next big fight.” ([GitHub][3])

## If someone insists on a method follow-up anyway

Do **one ablation**, not a new lane:

```text
A_std  = delta_phi(h_t, a_t)
A_orc  = delta_psi(h_t, s_t, a_t)
beta_t = 0 if A_std <= 0 else beta0      # beta0 in {1/3, 1/2}
A_mix  = (1 - beta_t) * A_orc + beta_t * A_std

L_pi = - A_mix * log pi(a_t | h_t)
```

Only run this **after** Hydra’s actor-critic RL loop is live. Promotion gate:

* lower advantage variance on matched public histories,
* no regression on duplicate/arena strength,
* no degradation to ExIt / DeltaQ label quality,
* no instability in KL drift or training loss.

If it fails any of those, close the experiment and keep it out of doctrine.

## What would falsify this recommendation

Any of these would move me off the current verdict:

1. **Repo evidence** that Hydra already has a DCRL-style mixed oracle/public advantage update in the active RL path, with good results.
2. **Mahjong- or multiplayer-IIG-specific evidence** that DCRL-style mixed critics beat the current same-domain anchors (RVR/Suphx-style oracle/value or ExIt/search-guided training) in the regime Hydra cares about.
3. **Active-path instability** in Hydra’s future RL lane where oracle-only CTDE is clearly the bottleneck and a mixed critic fixes it.
4. **Promoted docs already updated** to include UAAC/DCRL positioning; in that case, docs-worth-it drops to no-op.

## Bottom line

Hydra should **mention DCRL/UAAC explicitly**, but only as a **small literature-positioning patch**.

Hydra should **not** materially react at the method level right now.

The missing citation is a **docs hygiene issue**, not a sign that Hydra’s architecture priorities should move.

[1]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md"
[2]: https://github.com/NikkeTryHard/hydra "https://github.com/NikkeTryHard/hydra"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs"
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html"
[5]: https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf "https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf"
[6]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md"
[7]: https://www.roboticsproceedings.org/rss14/p08.pdf "https://www.roboticsproceedings.org/rss14/p08.pdf"
[8]: https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md "https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md"
