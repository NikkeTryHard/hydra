# Hydra hidden-world pass-two — external design-closure artifact bank

This file is intentionally small and pass-two-specific. The first hidden-world packet already carried a broad cross-field discovery bank. This follow-up bank is narrower. It exists to help the genie choose the actual winning design stack, training recipe, evaluation gates, and kill criteria rather than re-running general discovery.

The selection rule is strict:

- include only outside artifacts that sharpen a concrete pass-two decision,
- especially algorithm choice, teacher/student distillation, calibration-gate design, and offline-to-online policy improvement,
- avoid broad survey artifacts that merely repeat pass-one context.

## Artifact F01 — Estimating Expected Calibration Errors

- URL: https://arxiv.org/abs/2109.03480
- Domain: calibration evaluation
- Type: primary paper
- Suggested label: `ext_pass2_ece_estimation`

Why it matters in pass two:

Pass one already established that calibration matters. Pass two needs something stricter: calibration **evaluation** itself can be misleading if the metric is chosen or estimated poorly. This paper directly sharpens the pass-two job of defining promotion gates and kill criteria for belief-adjacent confidence outputs, danger outputs, tenpai probability, trust scores, or search-deferral triggers.

Exact pass-two use:

- force the genie to distinguish “measure calibration” from “measure calibration well”
- justify richer gate design than a single naive ECE scalar
- encourage phase-conditioned, bucket-sensitive, and estimator-aware calibration checks

What it should influence:

- `Evaluation Gates`
- `Kill Criteria`
- any design that uses belief confidence to gate runtime search, abstention, or action trust

## Artifact F02 — Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization

- URL: https://www.ijcai.org/proceedings/2024/0477
- Domain: offline-to-online RL transition
- Type: primary conference paper
- Suggested label: `ext_pass2_offline_to_online_cpr`

Why it matters in pass two:

The hidden-world lane probably will not go straight from a perfect teacher object to a strong live policy. There is a transition problem: how to turn offline or replay-side supervision into online improvement without policy lock-in, instability, or brittle imitation. CPR is useful because it is not another generic RL paper; it explicitly focuses on stable fine-tuning after an offline-pretrained policy and on keeping improvement stable when the starting policy is already overtrained or brittle.

Exact pass-two use:

- sharpen the `Training Recipe` section for a hidden-world lane that starts from offline teacher signals and then moves into online or self-play refinement
- help the genie specify whether and how Hydra should separate feature reuse from policy revitalization
- provide a useful external precedent for why “good offline teacher” is not automatically “good online improvement path”

What it should influence:

- `Training Recipe`
- `Shortest Honest Tranche`
- `Kill Criteria` for offline-to-online instability

## Artifact F03 — Privileged Information Distillation for Language Models

- URL: https://arxiv.org/abs/2602.04942
- Domain: privileged-information teacher/student distillation
- Type: primary paper
- Suggested label: `ext_pass2_privileged_distillation`

Why it matters in pass two:

Pass one already made room for privileged or oracle objects as teacher-only tools. Pass two must decide how that actually turns into a legal student. This paper is useful because it is not just “teacher distillation is good”; it gives a clearer pass-two precedent for a privileged teacher, an unconditioned student, and a training process that keeps the teacher/student split explicit instead of hoping the transfer happens by magic.

Exact pass-two use:

- sharpen the `Teacher Hierarchy`
- strengthen the `Training Recipe` where oracle or privileged hidden-state teachers are allowed during training but must disappear at deployment
- encourage an explicit answer on whether Hydra should use joint teacher-student objectives, staged masking, or on-policy self-distillation once the hidden-world teacher exists

What it should influence:

- `Teacher Hierarchy`
- `Training Recipe`
- `Where Hydra Is Wrong` if current Hydra underuses privileged-teacher distillation patterns

## Artifact F04 — Optional calibration-gating appendix slot

- Status: intentionally left as an appendix slot rather than a locked artifact

Rationale:

Pass two may want a selective-classification / abstention paper if the genie strongly recommends confidence-gated search deferral or abstain-to-search behavior. But do not force this into the generated packet unless the exact paper is chosen and actually sharpens the design. The first three artifacts above already add value without dragging the packet back into broad-search mode.

## Minimal inclusion rule

If the generated pass-two packet needs to stay focused, include only:

1. `ext_pass2_ece_estimation`
2. `ext_pass2_offline_to_online_cpr`
3. `ext_pass2_privileged_distillation`

That is enough external pressure for pass two. Everything else should only be added if the generated packet still lacks the evidence needed to decide the winning design, training loop, or gate structure.
