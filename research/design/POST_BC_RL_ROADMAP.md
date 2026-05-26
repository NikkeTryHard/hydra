# Post-BC RL Roadmap

SSOT for post-BC RL status and next phases. Keep local ToT files as evidence/prompts only; do not treat them as operator docs.

## Phase status

| Phase | Status | Scope |
|---|---|---|
| 0 | Done | PPO/GAE tensors and checkpoint init. |
| 1 | Done | PPO batch/artifact and one train step. |
| 2 | Done | Rust real-game smoke into Python PPO update. |
| 3A | Done | Direct sampled ACH objective. |
| 3B | Done | Same-artifact PPO vs ACH comparison. |
| 4A | Done | Checkpoint paired-eval decision surface. |
| 4B | Done | Control-run harness with injected eval. |
| 4B2 | Done | Native arena/eval adapter path for control-run/eval spine. |
| 4C | Done | Default-off/fail-closed GRP-derived `Phi` validation/PBRS substrate. |
| 4E | Done | Default-off narrowed PyTorch DRDA residual ACH adapter; exact neural rebase and export remain fail-closed. |
| 4D | Research only | Privileged/asymmetric critic is not builder-ready; hidden-feature contract and isolation proof missing. |
| 5 | Later | ExIt/DeltaQ Python target consumption/promotion after earlier gates. |
| 6 | Later | Population/league after single-agent objective and eval spine prove useful. |

## Built now

- PPO substrate: public actor path, terminal `U_A`, GAE/RL batch/artifact train path.
- Direct sampled ACH and narrowed DRDA residual ACH: same artifact/control surface as PPO. DRDA is default-off, no-rebase, and not export/native-arena compatible yet.
- Control-run/eval spine: paired checkpoint evaluation and control-run comparison, with native arena/eval path available for checkpoint evidence.
- Phase 4C substrate: GRP-derived public `Phi` validation/PBRS metadata path exists dormant. Defaults preserve terminal `U_A`; nonzero PBRS fails closed without validated activation artifact.

## Deliberately inactive

- Nonzero PBRS without validation artifact. Thresholds-absent reports cannot activate beta.
- Privileged critic. Current deploy actor remains public `obs [192,34] -> policy_logits [46]`; no hidden tensor may reach actor/export.
- NeuRD. Needs all-action Q/regret targets before it is meaningful.
- ExIt/DeltaQ Python targets. Sidecar lanes exist elsewhere, but Python default RL target consumption is not active.
- Population/league. Waits for stable paired arena promotion discipline.
- Belief/search stack: AFBS, CT-SMC, broad search-as-feature, robust opponent search. Later ceiling work, not current post-BC gate.

## Old Hydra mapping

| Old lane | Current mapping |
|---|---|
| DRDA-wrapped ACH | Narrowed PyTorch residual/tau adapter built default-off. Full exact neural rebase remains deferred/fail-closed. |
| ExIt/DeltaQ | Deferred for Python RL target path; sidecar/provenance rules still govern future use. |
| GRP / NextRank | Scaffolded as public `Phi` validation and default-off PBRS substrate. Not active shaping. |
| Search / belief / AFBS / CT-SMC | Later reserve/ceiling work after control spine wins. |
| Population / league | Later promotion infrastructure, not first post-BC RL step. |

## Train ladder

1. BC checkpoint baseline.
2. PPO control.
3. Direct sampled ACH.
4. DRDA residual ACH (no-rebase adapter first; full rebase later only if exact fold representation exists).
5. PBRS beta sweep only after validated public `Phi` artifact.
6. Privileged critic only after hidden-feature contract + actor/export isolation proof.
7. ExIt/DeltaQ Python targets with strict sidecar provenance.
8. Population/league.

## Gates

- Paired arena evidence against previous winner; validation/offline loss alone is not promotion.
- No regression in fourth-rate, top-2 decision quality, legal-mask behavior, or KL-to-baseline discipline.
- New modules default off until promoted; activation requires explicit config and matching metadata.
- No hidden actor path: deploy/export actor accepts public obs only; private/teacher data may not affect actor logits except through approved training losses with isolation proof.
- Reward contracts fail closed on rank utility, gamma/lambda, state boundary, encoder/action, or shaping metadata mismatch.

## References

- `local/post-bc-rl-roadmap.md` — superseded scratch roadmap.
- `local/tree of thoughts agent/tot/post-bc-rl-next-phases.md` — earlier phase plan and research prompts.
- `local/tree of thoughts agent/tot/phase-4b-control-run.md` — Phase 4B control-run notes.
- `local/tree of thoughts agent/tot/phase-4c-all-builder-readiness-tot.md` — all-4C readiness audit.
- `local/tree of thoughts agent/tot/phase-4c-4d-final-tot-research.md` — 4C/4D reconciliation.
- `local/tree of thoughts agent/tot/phase-4c-grp-nextrank-potential-shaping-spec.md` — GRP/PBRS research spec.
- `local/tree of thoughts agent/tot/pytorch-drda-residual-ach-spec.md` — narrowed PyTorch DRDA residual ACH spec.
- `local/tree of thoughts agent/tot/phase-4d-privileged-critic.md` — privileged critic research spec.
- `research/design/HYDRA_FINAL.md` — long-term max-ceiling architecture.
- `research/design/HYDRA_RECONCILIATION.md` — v1 active path and reserve shelf.
- `docs/CURRENT_STATUS.md` — shipped/staged status.
- `docs/TRAINING_RUNBOOK.md` — operator-facing training behavior.
