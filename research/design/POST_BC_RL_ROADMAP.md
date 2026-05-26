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
| 5 | Done | Dormant/default-off Python ExIt + DeltaQ target consumption. |
| 6 | Done | Minimal population/promotion ledger with registry, seed bank, opponent pool, and fail-closed promotion evidence. |

## Built now

- PPO substrate/operator path: public actor path, terminal `U_A`, GAE/RL batch/artifact train path, plus long-run `rl.phase=ppo_control` using native ONNX/PyO3 real-game rollout collection.
- Direct sampled ACH and narrowed DRDA residual ACH: same artifact/control surface as PPO. DRDA is default-off, no-rebase, and not export/native-arena compatible yet.
- Control-run/eval spine: paired checkpoint evaluation and control-run comparison, with native arena/eval path available for checkpoint evidence.
- Phase 4C substrate: GRP-derived public `Phi` validation/PBRS metadata path exists dormant. Defaults preserve terminal `U_A`; nonzero PBRS fails closed without validated activation artifact.
- Phase 5 target consumption: Python compact-shard path carries ExIt/DeltaQ target+mask lanes dormant/default-off; ExIt loss can consume validated labels when explicitly weighted; DeltaQ target carrier/metadata exists, and positive DeltaQ loss fails closed without reviewed output-head contract.

## Deliberately inactive

- Nonzero PBRS without validation artifact. Thresholds-absent reports cannot activate beta.
- Privileged critic. Current deploy actor remains public `obs [192,34] -> policy_logits [46]`; no hidden tensor may reach actor/export.
- NeuRD. Needs all-action Q/regret targets before it is meaningful.
- DeltaQ positive loss/head remains inactive until reviewed output-head contract exists; search generation, PIMC, hard action override, and Python promotion authority remain inactive.
- Broad league/PSRO/PFSP/exploiters remain inactive. Minimal Phase 6 registry/promotion ledger exists; offline loss alone cannot promote.
- Belief/search stack: AFBS, CT-SMC, broad search-as-feature, robust opponent search. Later ceiling work, not current post-BC gate.

## Old Hydra mapping

| Old lane | Current mapping |
|---|---|
| DRDA-wrapped ACH | Narrowed PyTorch residual/tau adapter built default-off. Full exact neural rebase remains deferred/fail-closed. |
| ExIt/DeltaQ | Carrier/consumption scaffolded in Python default-off. Search generation, teacher production, and promotion authority remain later; Rust sidecar/provenance remains authority. |
| GRP / NextRank | Scaffolded as public `Phi` validation and default-off PBRS substrate. Not active shaping. |
| Search / belief / AFBS / CT-SMC | Later reserve/ceiling work after control spine wins. |
| Population / league | Minimal registry/promotion ledger built. Full PSRO/PFSP/exploiters remain later reserve. |

## Train ladder

1. BC checkpoint baseline.
2. PPO control.
3. Direct sampled ACH.
4. DRDA residual ACH (no-rebase adapter first; full rebase later only if exact fold representation exists).
5. PBRS beta sweep only after validated public `Phi` artifact.
6. Privileged critic only after hidden-feature contract + actor/export isolation proof.
7. Optional ExIt/DeltaQ target consumption only after PBRS/privileged/DRDA evidence; not default training lane.
8. Minimal population/promotion ledger is available for evidence tracking; full league/PSRO only later if needed.

## Training roadmap

### Campaign artifact layout

`output_dir` is campaign root, not single-run directory. Campaign root owns `campaign.json`, `registry/`, and `stages/`. Python BC, Python PPO T1, and Rust RL T2-T7 runs write under `stages/<stage>/runs/<run_id>/` with run-local `config.yaml`, `launch_metadata.json`, `logs/`, `checkpoints/`, `exports/`, `rollouts/`, `eval/`, `tensorboard/`, `summary.json`, and learner result files. `stages/<stage>/latest_run` points resume/latest at active run.

Principle: train one new capability at time. Keep previous winner as baseline, use same campaign registry/seed bank/eval gate, and promote only on paired arena evidence.

| Step | Candidate | Enable | Keep off | Promotion gate |
|---|---|---|---|---|
| T0 | BC baseline | Existing Python BC checkpoint. | RL updates. | Register immutable baseline; arena sanity vs known baseline if available. |
| T1 | PPO control | Terminal `U_A`, GAE, legal masks, BC KL/entropy controls; wired as `rl.phase=ppo_control` long-run operator path. | ACH, DRDA, PBRS, ExIt/DeltaQ positive weights. | Beats or does not regress BC on `U_A`, fourth-rate, top-2, illegal count. |
| T2 | Direct sampled ACH | Direct ACH objective on same rollout/eval substrate. | DRDA, PBRS, ExIt/DeltaQ positive weights. | Beats PPO under same seed/opponent/eval budget with no safety regression. |
| T3 | DRDA residual ACH | Default-off no-rebase residual/tau adapter. | Full rebase/export, PBRS, ExIt/DeltaQ positive weights. | Beats direct ACH; KL/gate/entropy buckets stable; export limitation recorded. |
| T4 | PBRS beta sweep | Only after validated public `Phi` artifact authorizes nonzero beta. | Privileged critic, ExIt/DeltaQ positive weights. | Beats previous winner; no fourth-rate/top-2/bucket bias regression. |
| T5 | ExIt auxiliary | ExIt loss only with validated sidecar/provenance and explicit positive weight. | DeltaQ positive loss/head, search generation, hard overrides. | Improves arena result; no overfit/off-mask/sidecar mismatch. |
| T6 | DeltaQ experiment | Only after reviewed DeltaQ output/head contract exists. | Python promotion authority, hard overrides. | Existing DeltaQ promotion flow + paired arena confirmation. |
| T7 | Population window | Bounded recent snapshots/opponent pool. | PFSP/PSRO/exploiters/search teacher. | More robust than active-baseline-only without instability. |

Stop expanding when candidate fails gates. Fix data/objective/eval cause before adding another feature.

### Pre-train checklist

Stage directory names / implemented phase mapping: `T0_bc_baseline`, `T1_ppo_control`, `T2_direct_sampled_ach`, `T3_drda_residual_ach`, `T4_pbrs_beta_sweep`, `T5_exit_auxiliary`, `T6_delta_q_experiment`, `T7_population_window`. Top-level `stage:` may override layout stage name for run; it does not change algorithm/capability being trained.

- [ ] Immutable BC checkpoint registered in Phase 6 registry.
- [ ] Seed bank and active-baseline-only opponent pool written.
- [ ] Native arena path works for baseline checkpoint/export profile.
- [ ] Operator path chosen and documented: current T1 production RL path is Python `rl.phase: ppo_control`; Rust `rl.phase: drda_ach_self_play` and later Rust RL lanes remain legacy/advanced, but they still use `stages/<stage>/runs/<run_id>/` instead of `<output_dir>/rl`.
- [ ] Raw-MJAI resume cursor or shard-cache path selected before long run.
- [ ] Metrics sink writes train, eval, promotion, and registry artifacts under campaign root; Python BC/RL and Rust RL run logs/checkpoints/exports/rollouts/eval stay under `stages/<stage>/runs/<run_id>/`.

### Train-run rules

- Change exactly one capability per run.
- Preserve terminal `U_A` as promotion metric even when training reward changes.
- Do not compare checkpoints from different seed banks, opponent pools, or reward contracts as promotion evidence.
- Treat `insufficient_games` and `reject` as non-promotion.
- Never promote from training loss, validation loss, objective-comparison metrics, or sidecar validity alone.
- Record disabled capabilities explicitly so absent metadata cannot be misread as default-on.

## Gates

- Paired arena evidence against previous winner; validation/offline loss alone is not promotion.
- No regression in fourth-rate, top-2 decision quality, legal-mask behavior, or KL-to-baseline discipline.
- New modules default off until promoted; activation requires explicit config and matching metadata.
- No hidden actor path: deploy/export actor accepts public obs only; private/teacher data may not affect actor logits except through approved training losses with isolation proof.
- Reward contracts fail closed on rank utility, gamma/lambda, state boundary, encoder/action, or shaping metadata mismatch.

## References

Scratch evidence/prompts live under `local/tree of thoughts agent/tot/`; names may contain historical phase labels because they are archival inputs, not operator docs.

- `local/post-bc-rl-roadmap.md` — superseded scratch pointer to this SSOT.
- `local/tree of thoughts agent/tot/post_bc_rl_next_stages.md` — earlier plan/research prompts.
- `local/tree of thoughts agent/tot/control_run.md` — control-run evidence prompt.
- `local/tree of thoughts agent/tot/reward_shaping_builder_readiness_tot.md` — reward-shaping readiness audit.
- `local/tree of thoughts agent/tot/reward_shaping_privileged_critic_final_tot_research.md` — reward-shaping / privileged-critic reconciliation.
- `local/tree of thoughts agent/tot/grp_nextrank_potential_shaping_spec.md` — GRP/PBRS research spec.
- `local/tree of thoughts agent/tot/pytorch-drda-residual-ach-spec.md` — narrowed PyTorch DRDA residual ACH spec.
- `local/tree of thoughts agent/tot/population_league_promotion_spec.md` — population registry/promotion evidence prompt.
- `local/tree of thoughts agent/tot/privileged_critic.md` — privileged critic research spec.
- `research/design/HYDRA_FINAL.md` — long-term max-ceiling architecture.
- `research/design/HYDRA_RECONCILIATION.md` — v1 active path and reserve shelf.
- `docs/CURRENT_STATUS.md` — shipped/staged status.
- `docs/TRAINING_RUNBOOK.md` — operator-facing training behavior.
