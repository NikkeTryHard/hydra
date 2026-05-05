# Hydra hidden-world pass-two — external design-closure artifact bank

File intentionally small, pass-two-specific. First hidden-world packet already carried broad cross-field discovery bank. This follow-up bank narrower. Purpose: help genie choose winning design stack, training recipe, evaluation gates, kill criteria, not rerun general discovery.

Selection rule strict:

- include only outside artifacts that sharpen concrete pass-two decision
- especially algorithm choice, teacher/student distillation, calibration-gate design, offline-to-online policy improvement
- avoid broad survey artifacts that only repeat pass-one context

## Artifact F01 — Estimating Expected Calibration Errors

- URL: https://arxiv.org/abs/2109.03480
- Domain: calibration evaluation
- Type: primary paper
- Suggested label: `ext_pass2_ece_estimation`

Why pass two care:

Pass one already proved calibration matters. Pass two needs stricter point: calibration **evaluation** itself can mislead if metric chosen badly or estimated badly. Paper sharpens pass-two job of defining promotion gates and kill criteria for belief-adjacent confidence outputs, danger outputs, tenpai probability, trust scores, search-deferral triggers.

Exact pass-two use:

- force genie distinguish “measure calibration” vs “measure calibration well”
- justify richer gate design than single naive ECE scalar
- push phase-conditioned, bucket-sensitive, estimator-aware calibration checks

What should influence:

- `Evaluation Gates`
- `Kill Criteria`
- any design using belief confidence to gate runtime search, abstention, or action trust

## Artifact F02 — Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization

- URL: https://www.ijcai.org/proceedings/2024/0477
- Domain: offline-to-online RL transition
- Type: primary conference paper
- Suggested label: `ext_pass2_offline_to_online_cpr`

Why pass two care:

Hidden-world lane likely not go straight from perfect teacher object to strong live policy. Transition problem exists: turn offline or replay-side supervision into online improvement without policy lock-in, instability, brittle imitation. CPR useful because not generic RL paper; explicitly targets stable fine-tuning after offline-pretrained policy, keeping improvement stable when start policy already overtrained or brittle.

Exact pass-two use:

- sharpen `Training Recipe` section for hidden-world lane starting from offline teacher signals, then moving into online or self-play refinement
- help genie specify whether and how Hydra should separate feature reuse from policy revitalization
- give external precedent for why “good offline teacher” != “good online improvement path”

What should influence:

- `Training Recipe`
- `Shortest Honest Tranche`
- `Kill Criteria` for offline-to-online instability

## Artifact F03 — Privileged Information Distillation for Language Models

- URL: https://arxiv.org/abs/2602.04942
- Domain: privileged-information teacher/student distillation
- Type: primary paper
- Suggested label: `ext_pass2_privileged_distillation`

Why pass two care:

Pass one already allowed privileged or oracle objects as teacher-only tools. Pass two must decide how that becomes legal student. Paper useful because not only “teacher distillation good”; gives clearer pass-two precedent for privileged teacher, unconditioned student, training process that keeps teacher/student split explicit instead of hoping transfer happens magically.

Exact pass-two use:

- sharpen `Teacher Hierarchy`
- strengthen `Training Recipe` where oracle or privileged hidden-state teachers allowed during training but must disappear at deployment
- push explicit answer on whether Hydra should use joint teacher-student objectives, staged masking, or on-policy self-distillation once hidden-world teacher exists

What should influence:

- `Teacher Hierarchy`
- `Training Recipe`
- `Where Hydra Is Wrong` if current Hydra underuses privileged-teacher distillation patterns

## Artifact F04 — Optional calibration-gating appendix slot

- Status: intentionally left appendix slot, not locked artifact

Rationale:

Pass two may want selective-classification / abstention paper if genie strongly recommends confidence-gated search deferral or abstain-to-search behavior. But do not force this into generated packet unless exact paper chosen and actually sharpens design. First three artifacts already add value without dragging packet back into broad-search mode.

## Minimal inclusion rule

If generated pass-two packet must stay focused, include only:

1. `ext_pass2_ece_estimation`
2. `ext_pass2_offline_to_online_cpr`
3. `ext_pass2_privileged_distillation`

Enough external pressure for pass two. Everything else add only if generated packet still lacks evidence needed to decide winning design, training loop, or gate structure.