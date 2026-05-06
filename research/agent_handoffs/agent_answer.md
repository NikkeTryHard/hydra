## Verdict

**docs-worth-it:** **Yes, lightly.** Hydra should state its oracle-critic / CTDE surfaces sit in asymmetric-critic lineage, but patch should stay small: cite **UAAC** and **DCRL** (optionally earlier asymmetric actor-critic), then state clearly Hydra’s core lane remains **public-state ExIt/search-as-feature for Mahjong**, not “dual critic” as standalone novelty. **Confidence: medium-high.**

**method-worth-it:** **No material reaction now.** No architecture reprioritization, no new reserve lane, no roadmap reshuffle. At most, keep **future ablation note** for DCRL-style mixed-advantage critic only if Hydra’s actor-critic RL lane becomes active and empirically unstable. **Confidence: high.**

Reason split simple: Hydra already has real oracle-critic and CTDE-adjacent surfaces, but promoted method identity centers on **ExIt + search-as-feature + Mahjong-specific belief/Hand-EV/search plumbing**, and active roadmap explicitly says do not broaden architecture before baseline loop lives. ([GitHub][1])

## What the repo already proves

**Artifact-supported repo reality:** Hydra not merely “thinking about” oracle/search surfaces. Public docs show live **192x34** encoder with **Group C search/belief** and **Group D Hand-EV** planes; current status marks stronger belief tranche, Hand-EV realism, `safety_residual`, and end-to-end **ExIt carrier** as shipped; **DeltaQ** implemented but not default-on; broader public-belief search remains reserve-shelf. README-level routing also foregrounds opponent modeling and inference-time search as long-run differentiators. ([GitHub][2])

**Code-level proof:** model surface already includes public `value` head and `oracle_critic`, plus `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, and `safety_residual`. Oracle critic is distinct **4-output** head, fed detached backbone tensor, oracle-only loss tested not to backprop through shared backbone, and default loss weights for oracle and advanced heads are **0.0** in exposed loss config. This proves oracle lane is structurally real, but treated as **auxiliary / gated** surface, not default project identity. ([GitHub][3])

**Doctrine-level proof:** Hydra’s architecture doc makes two commitments:

1. perfect-information networks allowed for **variance reduction and diagnostics**, but
2. improvement operator for deployable policy should respect **public/information state**.

That stricter than “samplewise hidden-state critic unbiased in expectation,” and matters for DCRL comparison. Hydra also explicitly says oracle critic supplies advantages via **CTDE**, while actor conditions on public info only. ([GitHub][1])

## Overlap with DCRL: real vs superficial

| Dimension                                                  | DCRL / adjacent prior art | Hydra                                | Verdict                     |
| ---------------------------------------------------------- | ------------------------- | ------------------------------------ | --------------------------- |
| Privileged critic during training                          | Yes                       | Yes                                  | **Real overlap**            |
| Actor deploys without privileged state                     | Yes                       | Yes                                  | **Real overlap**            |
| Standard partial/history critic + oracle critic            | DCRL: core mechanism      | Hydra: surfaces exist                | **Real overlap**            |
| DCRL-style mixed actor advantage is main update object     | Yes                       | Not shown as active project identity | **Not established overlap** |
| Public-state search / ExIt teacher object                  | No                        | Central Hydra doctrine               | **Major divergence**        |
| Search-as-feature, belief planes, Hand-EV, DeltaQ          | No                        | Core Hydra surfaces                  | **Major divergence**        |
| Mahjong, 4-player, rank-aware objective                    | No                        | Yes                                  | **Major divergence**        |
| “dual critic” as generic naming similarity                 | N/A                       | Many heads / 2-tier nets / aux lanes | **Mostly superficial**      |

Most important distinction: **training trust object**. DCRL’s main idea remains actor-critic update built from weighted combination of oracle and standard advantages. Hydra’s stated north star differs: **search-generated public-state teacher targets** are central engine, while privileged critics are auxiliary CTDE-style tools. Overlap real, but narrow. ([NeurIPS Proceedings][4])

## Which external sources actually move the answer

**1) UAAC matters more than DCRL for literature positioning.**
UAAC established theoretically sound **history+state oracle critic** under partial observability and explicitly contrasted it with biased state-only asymmetric critics. If Hydra mentions DCRL, it should almost certainly mention **UAAC** too, because DCRL reads better as variance-reduced refinement of that lineage than root prior-art family. ([IFAAMAS][5])

**2) DCRL matters, but as narrow extension.**
DCRL’s primary contribution: keep **standard critic** and **oracle critic**, combine with weighting/gating scheme, retain unbiasedness while reducing variance under partial observability. Relevant to Hydra’s oracle-critic lane, but it does **not** address Hydra’s central differentiators: Mahjong-specific belief/search integration, ExIt/search targets, or 4-player rank-aware decision making. Published evaluation is on partial-observation control benchmarks, not Mahjong or multiplayer imperfect-information search. ([NeurIPS Proceedings][4])

**3) Same-domain Mahjong work still matters more for Hydra than DCRL.**
Hydra’s own docs already call **RVR** “most directly relevant work,” specifically because it is same-domain Mahjong, uses oracle/relative-value network, enforces zero-sum constraint, and adds expected reward network for variance reduction. RVR paper itself describes relative value network using global information plus expected reward network to stabilize Mahjong RL. more on-point for Hydra than DCRL. ([GitHub][6])

**4) Older asymmetric-critic root should also be acknowledged.**
2018 asymmetric actor-critic paper is clean historical root for “actor on partial observations, critic on full state during training only.” If Hydra wants one umbrella sentence, clean lineage is **AACC/AAC → UAAC → DCRL**, with **RVR/Suphx** as Mahjong-specific branch. ([Robotics Proceedings][7])

**5) Hydra already foregrounds stronger adjacent families.**
Its own reference/comparison surfaces emphasize **RVR**, **ACH**, **OLSS**, **ExIt**, and **oracle distillation** as most actionable or central ideas. Training-paradigms doc literally says **ExIt** and **asymmetric oracle training** are two most actionable paradigms for Hydra, and treats oracle distillation as already Hydra’s Phase 2 concept. That lowers method-level importance of DCRL further. ([GitHub][8])

## Decision-ready recommendation

### Docs-worth-it: **Yes, but tiny**

Omission worth fixing because Hydra publicly discusses oracle critics / CTDE / partial observability, and missing **UAAC/DCRL** branch creates avoidable “nearby prior art not acknowledged” hole. But this is **positioning fix**, not re-architecture.

### Smallest justified docs patch set

**Minimum patch**

1. **`research/intel/REFERENCES.md`**
Add entries for:

   * **Asymmetric Actor-Critic for Image-Based Robot Learning** (2018)
   * **Unbiased Asymmetric Reinforcement Learning under Partial Observability** (UAAC, 2022)
   * **Dual Critic Reinforcement Learning under Partial Observability** (DCRL, 2024)

2. **`research/design/HYDRA_FINAL.md`**
Add one short prior-art note near **P2** or **Phase 1 oracle supervision**:

> Prior-art positioning: Hydra’s oracle-critic / CTDE surfaces are adjacent to asymmetric-critic lineage (Pinto et al. 2018; Baisero & Amato 2022; Li et al. 2024). Hydra does not claim privileged critics as standalone novelty. Promoted delta is combination with Mahjong-specific ExIt/search-as-feature, belief/search surfaces, Hand-EV features, and 4-player rank-aware training.

3. **Optional README sentence**
Only if you want public front door to inoculate against shallow critique:

> Hydra’s oracle-critic lane sits in asymmetric-critic line (AACC / UAAC / DCRL), while Hydra’s main architecture identity is public-state ExIt/search-as-feature for Mahjong.

**Do not** edit `HYDRA_RECONCILIATION.md` unless you also change roadmap priority. Right now, not.

## Method-worth-it: **No active reaction**

Hydra’s roadmap explicitly says:

* baseline first,
* close loops before expanding architecture,
* search-strength lanes stay selective,
* broader identity debates should not outrank live training loop. ([GitHub][9])

So DCRL does **not** justify:

* changing architecture priorities,
* adding new reserve lane now,
* bumping DeltaQ/search-strength work,
* or rewriting Hydra’s method story around “dual critics.”

### Exact no-op rationale

1. **Hydra already tracks more direct prior art.**
For Hydra’s actual domain and ambitions, **RVR / Suphx / ACH / OLSS / ExIt** are more direct than DCRL. ([GitHub][6])

2. **Hydra’s main trust object differs.**
DCRL’s center of gravity is dual-critic actor-critic update. Hydra’s is **public-state ExIt/search teacher**. DCRL can only touch secondary lane. ([GitHub][1])

3. **Current code exposes oracle surface, but as auxiliary, not promoted shared training identity.**
Detached oracle input, optional oracle loss, default zero oracle weight, and auxiliary-head gating all point to “use carefully when activated,” not “make this project’s next big fight.” ([GitHub][3])

## If someone insists on a method follow-up anyway

Do **one ablation**, not new lane:

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

If it fails any, close experiment and keep it out of doctrine.

## What would falsify this recommendation

Any of these would move me off current verdict:

1. **Repo evidence** that Hydra already has DCRL-style mixed oracle/public advantage update in active RL path, with good results.
2. **Mahjong- or multiplayer-IIG-specific evidence** that DCRL-style mixed critics beat current same-domain anchors (RVR/Suphx-style oracle/value or ExIt/search-guided training) in regime Hydra cares about.
3. **Active-path instability** in Hydra’s future RL lane where oracle-only CTDE is clearly bottleneck and mixed critic fixes it.
4. **Promoted docs already updated** to include UAAC/DCRL positioning; in that case, docs-worth-it drops to no-op.

## Bottom line

Hydra should **mention DCRL/UAAC explicitly**, but only as **small literature-positioning patch**.

Hydra should **not** materially react at method level right now.

Missing citation is **docs hygiene issue**, not sign Hydra’s architecture priorities should move.

[1]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_FINAL.md"
[2]: https://github.com/NikkeTryHard/hydra "https://github.com/NikkeTryHard/hydra"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/crates/hydra-train/src/model.rs"
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/d399b67fa017f0f7670102c88507720c-Abstract-Conference.html"
[5]: https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf "https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf"
[6]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/REWARD_DESIGN.md"
[7]: https://www.roboticsproceedings.org/rss14/p08.pdf "https://www.roboticsproceedings.org/rss14/p08.pdf"
[8]: https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md "https://github.com/NikkeTryHard/hydra/blob/master/research/intel/REFERENCES.md"
[9]: https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md "https://github.com/NikkeTryHard/hydra/blob/master/research/design/HYDRA_RECONCILIATION.md"