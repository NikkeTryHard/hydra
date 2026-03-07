## Hydra posture reconstruction

1. **Mainline doctrine is “close loops first,” not “restart Hydra around global search.”** The current execution doctrine says the biggest blocker is that several advanced training/teacher loops are only partially closed, and it explicitly ranks “stronger target generation and supervision closure” ahead of a giant search rewrite. It also says not to restart Hydra and not to make full public-belief search the immediate mainline identity. ([GitHub][1])

2. **Hydra already has the architecture surfaces people would normally propose as future work.** The repo already carries a 192×34 fixed-shape encoder as single source of truth, a 12-block actor / 24-block learner configuration, and output heads for policy, value, score distribution, opponent tenpai, GRP, next discard, danger, oracle critic, belief fields, mixture weights, opponent hand type, `delta_q`, and `safety_residual`. This means the bottleneck is not “invent more heads”; it is whether those heads are fed credible targets and runtime hooks. ([GitHub][2])

3. **Several advanced loops are partially closed already.** `mjai_loader.rs` already builds Stage-A belief targets and `safety_residual` targets, and the model already detaches the oracle input path, which is consistent with the doctrine that oracle guidance should teach rather than dominate. This is not a blank slate. ([GitHub][3])

4. **The important open loops are concrete, not abstract.** In the current training path, `opponent_hand_type_target` and `delta_q_target` are still set to `None` during sample conversion, and the advanced loss weights for oracle critic, belief fields, mixture weights, opponent hand type, `delta_q`, and `safety_residual` all default to 0.0. So Hydra already has dormant capability surfaces whose teacher/label path is not yet fully activated. ([GitHub][4])

5. **AFBS, CT-SMC, robust-opponent, pondering, and search context are real code surfaces, but not yet the repo’s governing identity.** `bridge.rs` already exposes optional search/belief/risk/stress context; `afbs.rs` has a real top-k truncated expansion rule and pondering priority function; `ct_smc.rs` has a real particle structure and config; `robust_opponent.rs` already implements KL-ball adversary calculations. The doctrine’s point is not that these ideas are fake, but that they should remain specialist and gated until the higher-ROI loops are closed. ([GitHub][5])

6. **Current sequencing matters: Hand-EV realism comes before deeper AFBS reliance.** The reconciliation document explicitly ranks “rework Hand-EV realism before deeper AFBS” and “keep AFBS specialist / hard-state gated” ahead of robust-opponent and full public-belief-search ambitions. ([GitHub][1])

7. **Reserve shelf ideas are acknowledged, not denied.** DRDA/ACH, robust-opponent safe exploitation, richer latent opponent posterior, deeper AFBS semantics, and selective exactification remain on the reserve shelf. The doctrine is sequencing them, not erasing them. ([GitHub][1])

8. **There are explicit non-goals right now.** Broad “search everywhere,” duplicate belief stacks, more heads before current ones are trained, full public-belief search as immediate identity, and large optimizer side-quests are all deprioritized. That constrains what counts as a real breakthrough candidate here. ([GitHub][1])

---

## Candidate 1

### 1. name

**Lower-confidence KL-capped teacher distillation**

### 2. problem solved

Hydra already has stronger-than-main-policy information sources—Stage-A belief teachers, `safety_residual`, endgame solving, AFBS/CT-SMC search context, and a dormant `delta_q` head—but it does **not** yet have a mathematically explicit rule for when a stronger teacher should be trusted enough to move the policy, and by how much. Right now, the repo is structurally ready for advanced supervision, yet `delta_q_target` is still nulled and the advanced loss weights default to zero. That creates a ceiling: Hydra can possess stronger local evaluators without turning them into stable policy improvement. ([GitHub][4])

The failure mode I am targeting is not “Hydra lacks search.” It is “Hydra lacks a conservative improvement operator for noisy, partial-support, partially observed teachers in a 4-player general-sum game.” If you simply turn on teacher imitation, you over-trust hidden-information noise. If you wait for full ACH/DRDA or search-everywhere, you violate current doctrine and outrun the data/teacher closure problem. ([GitHub][1])

### 3. outside ingredients and exact sources

This candidate borrows from five places, each for a different reason.

First, **Romain Laroche, Paul Trichelair, and Rémi Tachet des Combes, “Safe Policy Improvement with Baseline Bootstrapping,” ICML 2019**: the important transferable idea is to stay close to a baseline when uncertainty is high, rather than switching wholesale to a supposedly better target. ([arXiv][6])

Second, **Matteo Pirotta, Marcello Restelli, Alessio Pecorino, and Daniele Calandriello, “Safe Policy Iteration,” ICML 2013**: the usable ingredient is the trust-region / bounded-step view of policy improvement, where you move toward a better policy by a constrained amount instead of replacing the baseline outright. ([Proceedings of Machine Learning Research][7])

Third, **Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel, “Overcoming Exploration in Reinforcement Learning with Demonstrations,” 2017**: the relevant idea is the Q-filter notion that demonstrations should shape policy only when they appear better than the current policy. ([arXiv][8])

Fourth, **Yujie Zhu, Charles A. Hepburn, Matthew Thorpe, and Giovanni Montana, “Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations,” 2025**: the key transfer is replacing brittle binary trust with a continuous uncertainty-aware weighting rule. That is especially relevant for Hydra because teacher quality varies with belief/search quality. ([arXiv][9])

Fifth, **Wen Sun, Arun Venkatraman, Geoffrey J. Gordon, Byron Boots, and J. Andrew Bagnell, “Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction,” ICML 2017**: the transferable principle is that training-time access to a stronger cost-to-go oracle can be converted into a better student policy, provided the oracle signal is used as a structured policy-improvement target rather than naive cloning. ([Proceedings of Machine Learning Research][10])

### 4. what is borrowed unchanged

Borrowed essentially unchanged:

* **Baseline anchoring under uncertainty** from SPIBB: when you are not sure, do not move far from the current policy. ([arXiv][6])
* **Conservative step sizing / trust-region view** from Safe Policy Iteration: policy improvement should be bounded, not absolute. ([Proceedings of Machine Learning Research][7])
* **“Only imitate when teacher seems better”** from Q-filter. ([arXiv][11])
* **Smooth uncertainty weighting instead of binary gating** from SPReD. ([arXiv][9])
* **Training-time teacher cost-to-go as a policy-improvement signal** from AggreVaTeD. ([Proceedings of Machine Learning Research][10])

### 5. what is adapted for Hydra

Three adaptations are Hydra-specific.

First, Hydra’s teachers are not full omniscient experts over the whole 46-action simplex. AFBS is top-k truncated, some teachers will naturally be discard-only, and runtime search support may be partial. So the target builder must work with a **partial support set** and leave unsupported actions at baseline mass. Hydra’s AFBS code already truncates to top-k and renormalizes, which makes this a real constraint rather than a hypothetical one. ([GitHub][12])

Second, Hydra’s uncertainty is not just “dataset count.” It can come from search-grade belief quality and hidden-information stress. `SearchContext` already has hooks for CT-SMC, AFBS, opponent risk, and opponent stress, and CT-SMC already carries an ESS-style config concept. So Hydra can drive the trust rule from **belief/search quality proxies**, not from fantasy observability. ([GitHub][5])

Third, Hydra already has a dormant `delta_q` head and a detached oracle path. I am adapting the teacher rule so that it does **not** require a new architecture. It turns the existing `delta_q` head into a learnable “teacher-improvement score” head and uses the current policy as a stop-gradient anchor when forming the teacher target. ([GitHub][13])

### 6. what is genuinely novel synthesis

The novel synthesis is **not** “safe RL plus imitation.” It is this specific operator:

* store **sparse teacher action scores** rather than dense soft targets,
* reconstruct a **current-policy-anchored** teacher target on the fly,
* use **statewise lower-confidence shrinkage** from Hydra-native uncertainty signals,
* keep unsupported actions at baseline mass by construction,
* and use the same sparse teacher payload to supervise the dormant `delta_q` head.

That combination is the important part. It is what lets Hydra exploit expensive, local, high-value teacher computations without promoting search-everywhere or pretending the teacher is globally correct.

### 7. why it transfers to Hydra specifically

Hydra is unusually well positioned for this candidate because the repo already contains all of the pieces that ordinary projects would still need to invent: advanced heads, optional search context, belief/search modules, a detached oracle path, and partial teacher generation in the data loader. The problem is the **conversion rule** from stronger local teacher information into stable policy improvement. That is exactly what this candidate supplies. ([GitHub][13])

It also matches the doctrine almost perfectly. The reconciliation document says the first tranche should center on `sample.rs`, `mjai_loader.rs`, `losses.rs`, `model.rs`, and bridge/search-context review, and it explicitly prefers “ExIt target + delta-Q + safety residual first.” This candidate is exactly that, made explicit enough to prototype. ([GitHub][1])

### 8. exact mathematical formulation

Let the legal action set in state (s) be (A_{\text{leg}}(s)\subseteq{1,\dots,46}). Let the current student logits be (z_\theta(s)\in\mathbb{R}^{46}). Define the **stop-gradient** baseline policy

[
\pi_0(a\mid s)=\operatorname{sg}!\left(\operatorname{softmax}*{A*{\text{leg}}}(z_\theta(s))\right)_a.
]

For this state, suppose the teacher provides scores on a sparse support set (S(s)\subseteq A_{\text{leg}}(s)), with teacher payload
[
{(i_k, q_k)}_{k=1}^{K}, \quad i_k\in S(s), ; q_k\in\mathbb{R}.
]

Here (K\le K_{\max}), and unsupported actions are simply absent. Define the baseline mass on teacher-supported actions:
[
P_S(s)=\sum_{a\in S(s)} \pi_0(a\mid s).
]

If (K<2) or (P_S(s)<p_{\min}), we do **no teacher update** on this state.

Otherwise define the support-renormalized baseline:
[
\tilde{\pi}_0(a\mid s)=\frac{\pi_0(a\mid s)}{P_S(s)} \quad \text{for } a\in S(s).
]

Define the teacher-centered baseline score:
[
\bar q(s)=\sum_{a\in S(s)} \tilde{\pi}_0(a\mid s), q_T(a\mid s).
]

To normalize scale across heterogeneous teachers, define a robust dispersion estimate over supported scores:
[
\sigma_q(s)=\max!\left(\sigma_{\min}, 1.4826 \cdot \operatorname{median}*{a\in S(s)} \left|q_T(a\mid s)-\operatorname{median}*{b\in S(s)} q_T(b\mid s)\right|\right).
]

Then the normalized teacher improvement score on supported actions is
[
\hat A(a\mid s)=\operatorname{clip}!\left(\frac{q_T(a\mid s)-\bar q(s)}{\sigma_q(s)}, -A_{\max}, A_{\max}\right), \qquad a\in S(s),
]
and (\hat A(a\mid s)=0) for (a\notin S(s)).

Now define a Hydra-native uncertainty scalar (u(s)\in[0,1]). For the **minimum prototype**, use only signals that already exist or are cheap to expose:

* AFBS / CT-SMC teacher state: (u(s)=\operatorname{clip}(\alpha_{\text{ess}}(1-\text{ESS}(s))+\alpha_{\text{stress}}\cdot \text{stress}(s),0,1)),
* endgame teacher state: (u(s)=0) initially,
* Hand-EV teacher: **defer** until Hand-EV realism is reworked, per doctrine. ([GitHub][5])

Form the pessimistic improvement score
[
g(a\mid s)=\max\bigl(0,\hat A(a\mid s)-\lambda_u u(s)\bigr).
]

Define a statewise KL budget
[
\kappa(s)=\kappa_{\max}(1-u(s)).
]

The teacher target is the solution of the trust-region problem
[
\pi_T(\cdot\mid s)=\arg\max_{\mu\in\Delta(A_{\text{leg}})}
\left[
\sum_{a} \mu(a), g(a\mid s) - \frac{1}{\eta_s}\operatorname{KL}(\mu|\pi_0)
\right]
]
with the additional constraint
[
\operatorname{KL}(\pi_T(\cdot\mid s)|\pi_0(\cdot\mid s))\le \kappa(s).
]

The closed form is
[
\pi_{T,\eta}(a\mid s)=\frac{\pi_0(a\mid s)\exp(\eta g(a\mid s))}{Z_\eta(s)},
]
with (\eta_s) chosen by binary search so that the KL budget is met. Unsupported actions have (g=0), so they retain baseline relative mass automatically.

I numerically sanity-checked the binary-search solver on random masked 46-action cases; the KL rises monotonically with (\eta) in the nondegenerate cases that matter here, so a simple bounded binary search is enough for the prototype.

Define the state guidance mask
[
m_{\text{teach}}(s)=\mathbf{1}!\left[\max_{a\in S(s)} g(a\mid s)>\tau_{\text{imp}} ;\land; P_S(s)\ge p_{\min}\right].
]

Then use two new losses:

**Teacher policy loss**
[
L_{\text{teach-pi}}
===================

\frac{1}{\sum_i m_{\text{teach}}(s_i)}
\sum_i
m_{\text{teach}}(s_i);
\mathrm{CE}!\left(z_\theta(s_i), \pi_T(\cdot\mid s_i)\right).
]

**Delta-Q loss** on supported actions only:
[
L_{\Delta Q}
============

\frac{1}{\sum_i \sum_{a\in S(s_i)} m_{\text{teach}}(s_i)}
\sum_i
m_{\text{teach}}(s_i)(1-u(s_i))
\sum_{a\in S(s_i)}
\operatorname{Huber}!\left(\Delta Q_\theta(a\mid s_i)-\hat A(a\mid s_i)\right).
]

Total loss:
[
L = L_{\text{Hydra-current}} + \lambda_{\text{teach}} L_{\text{teach-pi}} + \lambda_{\Delta Q} L_{\Delta Q}.
]

I would **not** regress raw point-unit (q_T-\bar q) in the first prototype. The normalization above is deliberate; Hydra’s teachers are heterogeneous, and a normalized improvement target is more stable across source types.

### 9. tensor shapes and affected network interfaces

Existing Hydra interfaces:

* observation tensor: **`[B, 192, 34]`**. ([GitHub][2])
* policy logits: **`[B, 46]`**. ([GitHub][13])
* `delta_q`: **`[B, 46]`**. ([GitHub][13])
* `safety_residual`: **`[B, 46]`**. ([GitHub][13])
* belief fields head: effectively **`[B, 16, 34]`** from 4 components × 4 channels. ([GitHub][14])
* mixture weights: **`[B, 4]`**. ([GitHub][13])
* opponent hand type head: **`[B, 24]`** = 3 opponents × 8 classes. ([GitHub][13])

New per-sample sparse teacher payload for this candidate:

* `teacher_action_idx`: **`[B, K_max]`**, integer padded with sentinel.
* `teacher_action_score`: **`[B, K_max]`**, float32.
* `teacher_action_mask`: **`[B, K_max]`**, bool.
* `teacher_uncertainty`: **`[B, 1]`**, float32.
* `teacher_source_id` (optional but recommended): **`[B, 1]`**, small integer.
* gathered baseline support probs during loss: **`[B, K_max]`**.
* reconstructed full target policy: **`[B, 46]`** transient, not stored.

I recommend **`K_max = 16`** for the first prototype. I checked the storage arithmetic explicitly: a dense 46-float target costs 184 bytes/sample; a dense target plus dense `delta_q_target` costs 368 bytes/sample. A sparse payload with `K_max=16` costs about **100 bytes/sample** with uint8 indices, or about **116 bytes/sample** with int16 indices. That matters because the final KL-capped target is built on the fly around the *current* student policy anyway, so storing dense soft targets would be both heavier and conceptually stale.

### 10. exact algorithm pseudocode

```python
# Inputs per batch item i:
# logits_i: [46]
# legal_mask_i: [46] bool
# teacher_idx_i: [K_max] int
# teacher_score_i: [K_max] float
# teacher_mask_i: [K_max] bool
# teacher_uncertainty_i: scalar in [0,1]

def build_teacher_target(logits, legal_mask, teacher_idx, teacher_score, teacher_mask, u,
                         p_min=0.15, sigma_min=1e-3, A_max=5.0,
                         lambda_u=1.0, kappa_max=0.10, tau_imp=0.15):
    p0 = stop_grad(masked_softmax(logits, legal_mask))          # [46]

    idx = teacher_idx[teacher_mask]                            # [K]
    q   = teacher_score[teacher_mask]                          # [K]
    if len(idx) < 2:
        return None

    p_support = p0[idx]                                        # [K]
    P = p_support.sum()
    if P < p_min:
        return None

    p_support = p_support / P
    q_bar = (p_support * q).sum()

    med = median(q)
    mad = median(abs(q - med))
    sigma = max(sigma_min, 1.4826 * mad)

    A = clip((q - q_bar) / sigma, -A_max, A_max)               # [K]
    g = maximum(A - lambda_u * u, 0.0)                         # [K]

    if g.max() <= tau_imp:
        return None

    g_full = zeros_like(p0)                                    # [46]
    g_full[idx] = g

    kappa = kappa_max * (1.0 - u)

    # binary search eta for KL budget
    lo, hi = 0.0, 64.0
    for _ in range(20):
        eta = 0.5 * (lo + hi)
        p_t = p0 * exp(eta * g_full)
        p_t = masked_normalize(p_t, legal_mask)
        kl = kl_divergence(p_t, p0)
        if kl > kappa:
            hi = eta
        else:
            lo = eta

    eta = lo
    p_t = p0 * exp(eta * g_full)
    p_t = masked_normalize(p_t, legal_mask)

    return {
        "pi_target": p_t,              # [46]
        "delta_q_idx": idx,            # [K]
        "delta_q_target": A,           # [K]
        "weight": 1.0 - u
    }

def training_step(batch, model, hydra_loss, lambda_teach, lambda_dq):
    out = model(batch.obs)

    base_loss = hydra_loss(out, batch.targets)

    pi_losses = []
    dq_losses = []

    for i in range(batch_size):
        tgt = build_teacher_target(
            out.policy_logits[i], batch.legal_mask[i],
            batch.teacher_action_idx[i],
            batch.teacher_action_score[i],
            batch.teacher_action_mask[i],
            batch.teacher_uncertainty[i]
        )
        if tgt is None:
            continue

        pi_losses.append(
            cross_entropy_with_soft_target(
                out.policy_logits[i], batch.legal_mask[i], tgt["pi_target"]
            )
        )

        pred = out.delta_q[i][tgt["delta_q_idx"]]
        dq_losses.append(
            tgt["weight"] * huber(pred - tgt["delta_q_target"]).mean()
        )

    L_teach = mean(pi_losses) if pi_losses else 0.0
    L_dq = mean(dq_losses) if dq_losses else 0.0

    return base_loss + lambda_teach * L_teach + lambda_dq * L_dq
```

### 11. exact Hydra surfaces it would touch

* **`hydra-train/src/data/sample.rs`**: add the sparse teacher payload to `MjaiSample` / batch conversion, instead of only dense one-hot-style labels. Today the sample conversion still nulls `delta_q_target`; that is the concrete gap being closed. ([GitHub][4])
* **`hydra-train/src/data/mjai_loader.rs`**: generate sparse teacher score supports on selected states. This is already the natural place because it already builds Stage-A belief targets and `safety_residual` targets. ([GitHub][3])
* **`hydra-train/src/training/losses.rs`**: add the on-the-fly KL-capped soft teacher loss and sparse `delta_q` Huber loss. There is already a `soft_target_from_exit(...)` helper, so the codebase already has the pattern of mixing model policy with a soft target; this candidate generalizes that from a scalar mix to a statewise KL-capped target. ([GitHub][15])
* **`hydra-train/src/model.rs`**: no new backbone or head is required; reuse the existing `delta_q` head. The oracle path already being detached is compatible with this teaching posture. ([GitHub][13])
* **`hydra-core/src/bridge.rs`**: expose `ESS`, opponent stress, and any search-derived uncertainty signals already present in `SearchContext`. ([GitHub][5])
* **`hydra-core/src/afbs.rs` / `endgame.rs` / `ct_smc.rs`**: wrap teacher score extraction for supported actions and uncertainty payloads. AFBS top-k support and CT-SMC particle machinery already exist. ([GitHub][12])
* **Do not put Hand-EV on the critical path in v0.** The doctrine explicitly says Hand-EV realism should be reworked before deeper AFBS reliance, so the first prototype should use endgame and AFBS teachers before relying on Hand-EV semantics. ([GitHub][1])

### 12. prototype path

1. **Start narrow: discard-only late/endgame bank.** Use states where endgame PIMC or existing exact-risk-style calculations already make sense. That minimizes support-mismatch and uncertainty problems on day 1. Doctrine-wise, this is consistent with using stronger target-generation loops before trying to make AFBS the mainline identity. ([GitHub][1])

2. **Implement sparse teacher payload with `K_max=16`.** This covers discard-only support and avoids stale dense soft targets.

3. **Add on-the-fly KL-capped target reconstruction in `losses.rs`.** Keep `pi_target` transient, built around stop-gradient current policy.

4. **Turn on only two new loss weights at first:** `lambda_teach` and `lambda_dq`. Leave other advanced heads alone.

5. **Teacher sources in order:** endgame first, then AFBS on hard states. Hand-EV waits.

6. **Only after offline bank metrics pass** do one full training run and then self-play eval.

### 13. benchmark plan

The first benchmark should isolate the mechanism from self-play noise.

**Benchmark A: held-out teacher-lift on a fixed state bank**

* Build a fixed late/endgame state bank from self-play / logs under deterministic seeds. Hydra already has deterministic seed hierarchy guidance for fair comparisons. ([GitHub][16])
* For each state, generate sparse teacher scores with the chosen teacher.
* Train four variants at equal teacher compute:

  1. current baseline,
  2. naive teacher imitation,
  3. binary Q-filter-style imitation,
  4. proposed lower-confidence KL-capped distillation.
* Evaluate on a held-out bank using the teacher scores as an **off-policy policy-improvement proxy**:
  [
  \text{Lift} = \mathbb{E}*{s}\left[\sum_a \pi*{\text{new}}(a\mid s) q_T(a\mid s) - \sum_a \pi_{\text{base}}(a\mid s) q_T(a\mid s)\right].
  ]
* Stratify lift by uncertainty bucket.

This tells you whether the policy-improvement operator itself works before you pay for long self-play.

**Benchmark B: downstream self-play**

* Equal total teacher compute.
* Same seed bank and deterministic comparison protocol.
* Track rating / average placement / deal-in-related regressions.
* Also track `delta_q` rank correlation against held-out teacher scores on supported actions.

### 14. what success would look like

Success has three parts.

1. **On the held-out teacher bank,** the proposed method should dominate naive imitation and binary Q-filter on the low- and medium-uncertainty buckets, and it should avoid regression on the high-uncertainty bucket.

2. **`delta_q` should learn ordering, not just noise.** On supported actions, the head should show useful rank correlation with normalized teacher improvement.

3. **In self-play at equal teacher compute,** gains should survive paired evaluation. If it only improves the teacher-lift proxy but not actual play, the operator is not strong enough.

### 15. what would kill the idea quickly

Kill it quickly if any of the following happen:

* It is **not better than binary Q-filter** at equal teacher compute.
* Guidance coverage collapses because (P_S) is too small on most states.
* High-uncertainty states still regress, which means the uncertainty proxy is not actually protecting anything.
* `delta_q` does not learn stable supported-action ordering.
* The method only matches “just turn on a scalar soft-target mix” using the existing `soft_target_from_exit(...)` pattern. If that simpler baseline gets the same result, this is not separator-level. ([GitHub][15])

### 16. red-team failure analysis

**How does this break in a 4-player general-sum game?**
The SPIBB / Safe-PI intuitions do **not** carry over as guarantees. Hydra is not in a stationary single-agent MDP. Other players adapt, information is hidden, and values are not zero-sum. So I am not claiming monotonic policy improvement. I am claiming a better *operator* than naive distillation in Hydra’s actual setting. That distinction matters. ([arXiv][6])

**Does it violate partial observability?**
No, provided the teacher uncertainty signal is based only on search/belief quality or on specialized endgame teachers, not on realized hidden state. The danger would be supervising on raw hidden allocations or omniscient labels. Hydra’s doctrine explicitly warns against that and prefers projected/public-teacher belief objects instead. ([GitHub][1])

**Does it require labels Hydra does not have?**
No for v0. It needs sparse teacher action scores and a scalar uncertainty. Sparse scores are cheap to expose from endgame / AFBS wrappers, and uncertainty can initially be zero for endgame or derived from existing CT-SMC / stress signals for search-backed teachers. `delta_q_target` is currently absent, but that is precisely one of the concrete open loops. ([GitHub][4])

**Is this secretly weaker than a simpler selective-compute move already on the mainline?**
Potentially. If the main gain comes entirely from “some states got better labels,” then this collapses into data-selection. That is why Benchmark A matters: it isolates the trust-rule itself against naive imitation and binary gating.

**Could partial teacher support make it inert?**
Yes. That is why the first prototype should live on discard-only late/endgame states or AFBS-supported hard states, and why (P_S) is explicitly in the math.

**Could heterogeneous teacher scales break it?**
Yes. That is the biggest technical risk. The MAD normalization reduces, but does not eliminate, cross-teacher semantic mismatch.

### 17. why this is more likely to matter than the strongest simpler mainline alternative

The strongest simpler mainline alternative is:

> “Close the loop the easy way: turn on more advanced losses, maybe add a scalar soft-target mix, maybe teach `delta_q` directly, but do not add a new trust rule.”

I do not think that is enough. Hydra’s actual bottleneck is not only the absence of targets; it is the absence of a **statewise trust mechanism** for teacher quality under partial observability and partial action support. The existing scalar-mix helper in `losses.rs` is too blunt for that. The proposed operator is more likely to matter because it decides **how far** to move, **when** to move, and **how to preserve baseline mass** on unsupported actions. That is the separator, not merely “another auxiliary loss.” ([GitHub][15])

### 18. closest known baseline and why this does not reduce to it

**Closest known baseline:** a blend of SPIBB, Safe Policy Iteration / CPI-style trust regions, Q-filter, and SPReD.

**Classification:** **B** — known mechanisms with a Hydra-specific adaptation that plausibly changes capability.

**Why it does not reduce to a `C`-class rename:**

* SPIBB assumes batch-RL-style baseline safety on uncertain state-action pairs; Hydra uses **search-quality / hidden-information uncertainty**, not dataset-count uncertainty. ([arXiv][6])
* Q-filter is binary; Hydra needs continuous distrust because teacher quality is graded and support is sparse. ([arXiv][11])
* Safe-PI is about bounded improvement in an MDP; Hydra adapts the trust-region shape but drops the guarantee claim because the game is 4-player general-sum with hidden information.
* The sparse partial-support construction—unsupported actions keep baseline mass—is a Hydra-specific necessity because AFBS is top-k truncated and not all teacher sources score the full 46-action simplex. ([GitHub][12])

### 19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker

| Requirement                       | Status             | Evidence or blocker                                                                                            |
| --------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| legal action mask                 | already exists     | present in current training targets / model usage pattern. ([GitHub][15])                                      |
| policy logits `[B,46]`            | already exists     | model output already has policy logits. ([GitHub][13])                                                         |
| `delta_q` head `[B,46]`           | already exists     | model output already has `delta_q`. ([GitHub][13])                                                             |
| `delta_q_target` path             | cheap to expose    | target field exists but sample conversion still sets it to `None`. ([GitHub][15])                              |
| sparse teacher action scores      | cheap to expose    | AFBS, endgame, and search context are real surfaces; needs wrapper code, not a new architecture. ([GitHub][5]) |
| teacher uncertainty scalar        | cheap to expose    | ESS/stress-style signals exist or are adjacent to existing runtime state. ([GitHub][5])                        |
| `safety_residual` target/mask     | already exists     | built in `mjai_loader.rs`. ([GitHub][3])                                                                       |
| statewise soft teacher CE in loss | missing but cheap  | `losses.rs` already has soft-target helper patterns; new KL-capped builder needed. ([GitHub][15])              |
| deterministic comparison harness  | already exists     | testing + seeding docs already define deterministic comparison discipline. ([GitHub][16])                      |
| Hand-EV as trusted teacher        | **blocked for v0** | doctrine says Hand-EV realism should be reworked before deeper reliance. ([GitHub][1])                         |

### 20. minimum falsifiable prototype

A **minimum falsifiable prototype** is:

* late/endgame discard-only state bank,
* sparse teacher payload from endgame teacher only,
* no Hand-EV,
* no runtime pondering changes,
* only two new losses: KL-capped teacher CE and supported-action `delta_q` Huber,
* compared against naive imitation and binary Q-filter at equal teacher compute,
* first judged on held-out teacher-lift before any full self-play.

If this setup does not beat binary Q-filter and scalar-mix soft targeting, the idea should be downgraded immediately.

---

## Candidate 2

### 1. name

**Value-of-search archive gating**

### 2. problem solved

Hydra’s doctrine is explicit that AFBS should remain **specialist / hard-state gated**, not broad search-everywhere. That means search strength is constrained less by raw search code and more by **where compute is spent**. Right now Hydra already has a hand-built pondering priority based on top-2 gap, risk, and ESS-like belief quality, plus a real `PonderManager`. But a three-term hand heuristic is not yet a learned value-of-computation operator. ([GitHub][1])

So the problem solved here is: **How do you make sparse search budget behave like a strategically important operator when doctrine forbids making search the everywhere-mainline?** The answer is to learn where search most changes policy quality per unit of compute, then use that both offline (for teacher generation) and online (for runtime pondering). ([GitHub][1])

### 3. outside ingredients and exact sources

This candidate mainly combines two sources.

First, **Alexandre Trudeau and Michael Bowling, “Targeted Search Control in AlphaZero for Effective Policy Improvement,” AAMAS 2023**: the central lesson is that search is only useful if it is an effective policy-improvement operator on the states you actually spend it on, and that an archive of states of interest can improve sample efficiency. ([arXiv][17])

Second, **Erez Karpas, Oded Betzalel, Solomon Eyal Shimony, David Tolpin, and Ariel Felner, “Rational deployment of multiple heuristics in optimal state-space search,” Artificial Intelligence 2018**: the relevant transferable idea is metareasoning—compute an expensive heuristic only when predicted myopic regret justifies the cost. ([Researchr][18])

### 4. what is borrowed unchanged

Borrowed unchanged:

* **archive-of-interest state selection** from Go-Exploit, except the archive is for search/teacher queries rather than self-play restart states. ([arXiv][17])
* **myopic value-of-computation logic** from rational metareasoning: spend expensive compute only when predicted regret of skipping it is high enough. ([ScienceDirect][19])

### 5. what is adapted for Hydra

Hydra-specific adaptations:

* The archive is **not** “states to restart self-play from.” It is “states worth teacher/search budget.”
* The label is not generic search value. It is **Hydra-specific search impact on current policy**, measured from sparse teacher scores.
* The runtime integration target is not a new planner; it is Hydra’s existing pondering infrastructure and hard-state gating logic. `compute_ponder_priority` and `PonderManager` already exist. ([GitHub][12])

### 6. what is genuinely novel synthesis

The real synthesis is to turn search allocation into a **learned operator tied to realized policy lift**, not a fixed heuristic. In Hydra’s setting, that matters more than in many engines because doctrine already rejects broad search-everywhere. Once that doctrine is fixed, learning where search is worth spending becomes a first-class capability question, not just tuning.

### 7. why it transfers to Hydra specifically

Hydra already has exactly the prerequisite surfaces:

* a real but heuristic pondering priority,
* a real pondering queue/cache,
* real search/belief context,
* and search modules that are supposed to stay specialist. ([GitHub][12])

That makes Hydra a much better host for this idea than a system already committed to uniform heavy search. In Hydra, search allocation is not a footnote; it is central to whether the specialist-search doctrine actually becomes strong in practice.

### 8. exact mathematical formulation

For a state (s), let the current baseline policy be the stop-gradient masked policy
[
\pi_0(a\mid s)=\operatorname{sg}!\left(\operatorname{softmax}*{A*{\text{leg}}}(z_\theta(s))\right)_a.
]

Suppose a budgeted teacher run with budget (b) returns sparse action support (S_b(s)) and teacher scores (q_T^b(a\mid s)) on that support. Define support mass
[
P_S^b(s)=\sum_{a\in S_b(s)} \pi_0(a\mid s).
]

If (P_S^b(s)<p_{\min}) or (|S_b(s)|<2), discard the label.

Otherwise define support-renormalized baseline
[
\tilde{\pi}_0^b(a\mid s)=\frac{\pi_0(a\mid s)}{P_S^b(s)},\qquad a\in S_b(s).
]

Define the **realized policy-improvement proxy**
[
\Delta_{\text{imp}}^b(s)=
P_S^b(s)\cdot
\left[
\max_{a\in S_b(s)} q_T^b(a\mid s)
---------------------------------

\sum_{a\in S_b(s)} \tilde{\pi}*0^b(a\mid s) q_T^b(a\mid s)
\right]*+.
]

Interpretation: this is the supported-set expected gain available if search reveals a better-supported action than the current policy’s supported-set expectation. Multiplying by (P_S^b) discounts states where the teacher only touched a tiny fraction of the student’s probability mass.

Let (c_b(s)) be search cost in normalized budget units (iterations, node expansions, or fixed budget quanta). Define
[
\mathrm{VOS}*b(s)=\frac{\Delta*{\text{imp}}^b(s)}{c_b(s)+c_0}.
]

Now define the pre-search feature vector
[
x(s)\in\mathbb{R}^{8}
]
as

[
x(s)=
\big[
H(\pi_0),;
\text{gap}_{1,2},;
\max(\text{danger}),;
\frac{#\text{legal}}{46},;
\text{risk_score},;
\text{particle_ess},;
\mathbf{1}[\text{risk_score present}],;
\mathbf{1}[\text{ess present}]
\big].
]

This uses already-existing policy/danger/legal outputs and the same risk / ESS flavor already present in the current pondering heuristic. ([GitHub][13])

Train a tiny gate (g_\phi:\mathbb{R}^8\to\mathbb{R}) to predict
[
y(s)=\log(1+\mathrm{VOS}_b(s)).
]

Loss:
[
L_{\text{gate}}
===============

\mathbb{E}*{s\sim \mathcal{D}*{\text{archive}}}
\left[
\operatorname{Huber}\left(g_\phi(x(s)) - y(s)\right)
\right].
]

Allocation rule under a fixed budget:

* offline teacher labeling: sample archive states in descending (g_\phi(x(s))),
* runtime pondering: push or rank queue items by (g_\phi(x(s))) instead of the current hand heuristic alone.

For safety, the first runtime version should be a **calibrated replacement**, e.g.
[
\text{priority}(s)=\beta_0\cdot \text{current_heuristic}(s)+\beta_1\cdot g_\phi(x(s)),
]
not a hard swap on day 1.

### 9. tensor shapes and affected network interfaces

Minimal prototype shapes:

* policy logits: **`[B,46]`** existing. ([GitHub][13])
* danger head: **`[B,46]`** existing. ([GitHub][13])
* legal mask: **`[B,46]`** existing in training/runtime logic. ([GitHub][15])
* gate feature vector (x): **`[B,8]`**
* gate hidden layer 1: **`[B,32]`**
* gate hidden layer 2: **`[B,32]`**
* gate output: **`[B,1]`**
* archive label `log(1+VOS)`: **`[B,1]`**

No Hydra backbone change is required for v0. This can be a side MLP or even a calibrated linear model first. That is important: the first test is about whether the *label* is learnable and useful, not about architecting a new trunk.

### 10. exact algorithm pseudocode

```python
# Phase 1: collect archive
archive = Reservoir(capacity=N)

for state in self_play_or_logs:
    x = make_gate_features(state)          # [8]
    archive.add(state, x)

# Phase 2: label a subset with search impact
labeled = []
for state, x in sample_subset(archive, M):
    teacher = run_budgeted_teacher(state, budget=b)  # AFBS or endgame
    if teacher.support_size < 2:
        continue

    p0 = stop_grad(masked_softmax(policy_logits(state), legal_mask(state)))
    idx, q = teacher.idx, teacher.score
    P = p0[idx].sum()
    if P < p_min:
        continue

    p_s = p0[idx] / P
    delta_imp = P * max(0.0, q.max() - (p_s * q).sum())
    vos = delta_imp / (teacher.cost_units + c0)

    labeled.append((x, log1p(vos)))

# Phase 3: train gate
for minibatch in labeled_loader(labeled):
    pred = gate(minibatch.x)               # [B,1]
    loss = huber(pred - minibatch.log1p_vos).mean()
    update(gate, loss)

# Phase 4: use gate for future allocation
def priority(state):
    x = make_gate_features(state)
    return beta0 * current_ponder_priority(state) + beta1 * gate(x)

# offline label allocation:
states_for_teacher = top_k(archive, key=priority, K=teacher_budget_count)

# runtime pondering:
ponder_queue.push(state, priority=priority(state))
```

### 11. exact Hydra surfaces it would touch

* **`hydra-core/src/afbs.rs`**: current pondering/search priority is heuristic; this candidate augments or calibrates it. AFBS already has real selection logic and top-k expansion. ([GitHub][12])
* **`hydra-core/src/bridge.rs`**: natural place to assemble risk/stress/ESS-backed feature inputs for the gate because `SearchContext` already aggregates them. ([GitHub][5])
* **`hydra-core` pondering path**: `PonderManager` queue/cache already exists. This is where runtime priority replacement lands. ([GitHub][12])
* **`hydra-train` data side**: a small archive-labeling pipeline, probably adjacent to current sample generation tooling, to store (x(s)) and search-impact labels.
* **No main-model surgery in v0.** That is deliberate.

### 12. prototype path

1. **Offline only first.** Do not touch runtime pondering yet.
2. Build a fixed archive from self-play / logs.
3. Label a subset with a cheap teacher budget.
4. Train the 8→32→32→1 gate.
5. Compare gate ranking against:

   * random selection,
   * current heuristic `compute_ponder_priority`.
6. Only if it wins offline, integrate into runtime `PonderManager`.

### 13. benchmark plan

**Benchmark A: equal-budget archive ranking**

* Fixed archive.
* Label a held-out portion with teacher runs.
* Compare top-(K) states selected by:

  1. random,
  2. current ponder heuristic,
  3. learned gate.
* Metric:
  [
  \text{Captured-VOS@K}
  =
  \frac{1}{K}\sum_{s\in \text{top-}K} \mathrm{VOS}(s).
  ]

Also report total captured (\Delta_{\text{imp}}) per cost unit.

**Benchmark B: downstream label allocation**

* Use each selector to decide which states get teacher labels during training.
* Equal total teacher compute.
* Then compare downstream training improvement.

**Benchmark C: runtime pondering**

Only after A and B pass:

* swap or blend into queue priority,
* equal runtime compute,
* paired self-play.

### 14. what success would look like

* Offline, the learned gate clearly out-ranks the current heuristic on held-out Captured-VOS@K.
* With the same teacher budget, label allocation chosen by the gate produces better downstream policy improvement than the current heuristic.
* Runtime pondering integration produces strength gains without increasing total search cost.

### 15. what would kill the idea quickly

* If the learned gate does **not** beat the current heuristic on held-out Captured-VOS@K.
* If it beats the heuristic offline but yields no downstream training advantage.
* If it only works by rediscovering one trivial feature such as entropy and nothing else.
* If search impact labels are too noisy to learn any ranking better than the hand heuristic.

### 16. red-team failure analysis

**Is this secretly just prioritization?**
Yes, and that is the main danger. If all it does is modestly reorder obvious hard states, it is a tuning trick, not a breakthrough. That is why it only survives here because Hydra has already decided not to search everywhere. In that doctrine, compute allocation can become capability-defining. Otherwise I would reject it.

**Does this depend on candidate 1?**
No. It can use the simpler supported-set (\Delta_{\text{imp}}) label above, independent of the KL-capped trust rule. Candidate 1 and Candidate 2 compound well, but the second is testable on its own.

**Does it assume hidden information away?**
No. The label is built from Hydra’s own budgeted teacher/search outputs and supported-action scores, not realized hidden state.

**Can it fail because AFBS itself is not mature enough?**
Absolutely. If the underlying teacher/search operator is weak, learning where to call it does not save it. That is one reason this candidate ranks below Candidate 1.

### 17. why this is more likely to matter than the strongest simpler mainline alternative

The strongest simpler alternative is to keep the current hand-crafted priority:
[
(0.1-\text{gap}*{1,2})*+ \cdot 10 + \text{risk}*+ + (1-\text{ESS})*+.
]
That heuristic already exists. ([GitHub][12])

I think the learned gate has a real shot to matter because the existing heuristic is not trained against **realized search impact**. It is trained against intuition. In a project whose doctrine says search must remain specialist, that distinction matters more than usual. But I rank this below Candidate 1 because it is still easier for it to collapse into “better tuning” than for Candidate 1 to do so.

### 18. closest known baseline and why this does not reduce to it

**Closest known baseline:** Go-Exploit plus rational metareasoning.

**Classification:** **B**.

**Why it does not reduce to a `C`-class rename:**

* Go-Exploit changes self-play start-state distribution in AlphaZero-style training; this candidate changes **teacher-query / pondering allocation inside Hydra’s specialist-search regime**. ([arXiv][17])
* Rational metareasoning typically reasons about whether to compute a heuristic; Hydra adapts it to **budgeted imperfect-information search impact on current policy**. ([ScienceDirect][19])
* The label is not generic “hardness.” It is policy-improvement-per-cost, derived from Hydra’s own sparse teacher scores.

### 19. dependency closure table: required signal / label / hook / teacher / runtime state | already exists / cheap to expose / missing | evidence or blocker

| Requirement                               | Status                             | Evidence or blocker                                                                                                          |
| ----------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| current ponder heuristic                  | already exists                     | `compute_ponder_priority` is already implemented. ([GitHub][12])                                                             |
| pondering queue/cache                     | already exists                     | `PonderManager` exists. ([GitHub][12])                                                                                       |
| policy entropy / top-2 gap                | cheap to expose / already implicit | both are derivable from existing policy logits. ([GitHub][13])                                                               |
| max danger                                | already exists                     | danger head already present. ([GitHub][13])                                                                                  |
| legal-action count                        | already exists                     | derivable from legal mask. ([GitHub][15])                                                                                    |
| risk score / ESS-style signal             | already exists or cheap to expose  | current heuristic already uses risk and particle ESS flavor; `SearchContext` carries related runtime context. ([GitHub][12]) |
| sparse teacher action scores for labeling | cheap to expose                    | AFBS/endgame surfaces already exist. ([GitHub][12])                                                                          |
| archive builder                           | missing but cheap                  | implementation work only; no architecture blocker evidenced                                                                  |
| tiny gate model                           | missing but cheap                  | implementation work only; no architecture blocker evidenced                                                                  |
| deterministic comparison discipline       | already exists                     | testing/seeding docs support fair comparison. ([GitHub][16])                                                                 |

### 20. minimum falsifiable prototype

A minimum falsifiable prototype is:

* build a fixed archive,
* label 50k states with a shallow teacher budget,
* train the 8→32→32→1 gate,
* compare captured VOS at equal top-(K) against random and current heuristic,
* and stop there.

If the gate does not beat the hand heuristic **offline**, do not integrate it into runtime pondering.

---

## the single best candidate to try first

**Candidate 1: Lower-confidence KL-capped teacher distillation.**

It attacks the exact mainline bottleneck: Hydra already has advanced teacher/search surfaces, but it does not yet have the trust rule that turns them into stable policy improvement. It also fits the reconciliation doctrine almost line by line. ([GitHub][1])

## the single best cheap benchmark to run first

**Held-out teacher-lift on a fixed late/endgame discard-only state bank for Candidate 1**, comparing:

1. naive imitation,
2. binary Q-filter,
3. scalar soft-target mixing,
4. KL-capped lower-confidence distillation.

That benchmark is cheap, isolates the mechanism, uses already-credible teacher surfaces, and will tell you quickly whether the trust rule is doing real work.

## the single biggest hidden implementation risk

**Cross-teacher score semantics.** Endgame, AFBS, and later Hand-EV may not naturally live on the same score scale or even the same horizon semantics. The MAD normalization helps, but if source-specific semantics differ too much, the target builder can become silently miscalibrated. That is the biggest risk because it can produce plausible-looking training while teaching the wrong comparison.

## the 2-5 most tempting rejected directions and exactly why they were rejected

1. **Broad ACH/DRDA revival now.** Rejected because the doctrine explicitly pushes DRDA/ACH to the reserve shelf until stronger target-generation and supervision closure are done. Reviving it now would outrun the existing label/teacher bottlenecks. ([GitHub][1])

2. **Full public-belief search as Hydra’s immediate identity.** Rejected because the reconciliation document explicitly says not to make full public-belief search the immediate mainline, and because that would violate the closure-first sequencing. ([GitHub][1])

3. **Posterior-conditioned robust exploitation / archetype opponent optimization.** Tempting because `robust_opponent.rs` is real and `opponent_hand_type` exists in the model, but rejected because `opponent_hand_type_target` is still nulled in the current sample path, the doctrine places robust-opponent later, and recent imperfect-information opponent-modeling work explicitly warns that even consistency is hard in simpler settings and loses convexity beyond two players. In Hydra’s 4-player general-sum setting, this is too diffuse for the current stage. ([GitHub][4])

4. **“Just add more heads / bigger model.”** Rejected because Hydra already has a rich head set and the doctrine explicitly deprioritizes adding more heads before current ones are trained with credible supervision. ([GitHub][1])

5. **Hand-EV-centered breakthrough right now.** Rejected not because Hand-EV is unimportant, but because the doctrine explicitly says Hand-EV realism should be reworked before deeper reliance. Making it the breakthrough bet before that realism pass would be backwards sequencing. ([GitHub][1])

[1]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_RECONCILIATION.md"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/docs/GAME_ENGINE.md"
[3]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/mjai_loader.rs"
[4]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/data/sample.rs"
[5]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/bridge.rs"
[6]: https://arxiv.org/abs/1712.06924 "https://arxiv.org/abs/1712.06924"
[7]: https://proceedings.mlr.press/v28/pirotta13.html "https://proceedings.mlr.press/v28/pirotta13.html"
[8]: https://arxiv.org/abs/1709.10089 "https://arxiv.org/abs/1709.10089"
[9]: https://arxiv.org/abs/2509.15981 "https://arxiv.org/abs/2509.15981"
[10]: https://proceedings.mlr.press/v70/sun17d.html "https://proceedings.mlr.press/v70/sun17d.html"
[11]: https://arxiv.org/pdf/1709.10089 "https://arxiv.org/pdf/1709.10089"
[12]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-core/src/afbs.rs"
[13]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/model.rs"
[14]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/heads.rs "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/hydra-train/src/heads.rs"
[15]: https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs "https://github.com/NikkeTryHard/hydra/blob/master/hydra-train/src/training/losses.rs"
[16]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md "https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/agent_handoffs/combined_all_variants/answer_1-1_combined.md"
[17]: https://arxiv.org/abs/2302.12359 "https://arxiv.org/abs/2302.12359"
[18]: https://researchr.org/publication/KarpasBSTF18 "https://researchr.org/publication/KarpasBSTF18"
[19]: https://www.sciencedirect.com/science/article/pii/S0004370217301261 "https://www.sciencedirect.com/science/article/pii/S0004370217301261"
