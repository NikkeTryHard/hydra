# Hydra2 Training Plan (Tenhou 4p hanchan, offline-only)

## 1. Vibe: win the night, not the hand

A hanchan is a long night, not one exciting hand. First place in one hand means
little if the next two hands give it all back. So we train players that protect
their tournament standing: prefer safe seconds over risky firsts, fold bad hands
early, and only push when the payoff is worth the danger. Every training step
below is judged by average final placement over many nights, not by flashy wins.

## 2. NOW — gates before baseline training starts

- [x] Wave-2 Lean lemmas green, named: `corpus_identity` (`TrainingData.lean`),
  `wall_block_variance_additive_finset` (`Evaluation.lean`),
  `legalSoftmax_sum_one` (`PPO.lean`), `cql_penalty_nonneg` (`CQL.lean`).
  Gate: `cd lean && lake build` clean with no sorry/axiom in the touched files
  (build gate MAIN-owned; oleans fresh at audit, grep sorry/axiom empty).
- [x] 3 tiny fixtures green. Gate: `tests/unit/test_supervised_loop_wp05b.py`
  and `tests/unit/test_baseline_wp05c.py` pass locally (data loads, loss goes
  down, checkpoints resume); duplicate/schedule guards
  `tests/integration/test_duplicate_block_wp06.py` and
  `tests/integration/test_schedule_wp03b.py` hold. MAIN-verified: 48 scoped
  tests green, lint+format+typecheck clean.
- [ ] Oracle real-data fixture. Gate: one small fixture built from a real log
  excerpt runs end to end through the teacher/oracle path (refs
  `lean/Formal/Blueprint/Opponent.lean`, `lean/Formal/Blueprint/Belief.lean`).
- [ ] Spec re-pin. Gate: `docs/IMPLEMENTATION_SPEC.md` matches the frozen Python
  surface (known drift: tracking tree still lists
  `tracking/{protocol.py,wandb_mirror.py}`; actual is
  `tracking/{__init__.py,clearml_mirror.py}`). No training runs until this matches.

## 3. BASELINE — first real training run

- [x] Behaviour cloning day-one slice (BC + placement + value heads wired).
  `TrainingLoopConfig` / replay config carry `w_policy`/`w_placement`/`w_value`
  (`loop.py:126-128`, `replay.py:130-132`) into
  `objectives.compute_supervised_loss`; Wave-E input bridge (`_model_forward`,
  `_move_batch_to_device` in loop.py only; replay stays legacy dict path by design).
  Gate: model imitates expert moves on held-out nights; loss and accuracy reported.
- [x] Manual selection gate (no auto-wire: train, then score held-out walls by
  hand). Gate: `evaluate_selection` on a wall-disjoint held-out BlockSplit
  (`loop.py:898`, `replay.py:740`), `maybe_promote_best` copies best-ckpt.pt by
  hand (`loop.py:907`, `replay.py:748`); both docstrings pin that `train` never
  calls selection.
- [x] ClearML mirror (Wave-D, observer-only). `tracking/clearml_mirror.py` is the
  sole `import clearml` owner (lazy); disabled by default (`HYDRA2_CLEARML_ENABLED`
  opt-in, `HYDRA2_CLEARML_DISABLED` kill-switch, `CLEARML_OFFLINE_MODE=1` offline
  sessions under `<artifact_root>/clearml_offline`); loop/replay construct via
  `make_mirror` with `log_update`+`log_checkpoint` per published checkpoint.
  Local manifests/checkpoints stay authoritative; the mirror never feeds back
  into training or RNG.
- [ ] RL-ready placeholders (hooks, no learning yet). Honest state: policy /
  placement / value heads emitted (`models/model.py`), losses implemented AND
  wired, mirror logging observer-only default-off; reward plumbing NOT YET —
  ticketed, no code paths (flag inventory has no RL/CQL/PPO flags). Gate: heads
  train with zero effect on BC numbers until enabled.

## 4. NEXT — one upgrade at a time, each with its gate

- [ ] Play-vs-teacher difference check (GRP-Phi style). Gate: promotion only if
  head-to-head placement improves over baseline with statistical significance;
  metric defined in `lean/Formal/Blueprint/Evaluation.lean`.
- [ ] Conservative learning (CQL style: stay close to known-good play).
  Penalty math already proved (`cql_penalty_nonneg`, `cql_scaled_nonneg`);
  wiring is future. Gate: offline check passes — no value over-claim on unseen
  moves; spec in `lean/Formal/Blueprint/CQL.lean`.
- [ ] Small stable policy steps (GAE/MPPO, ~5-6% demo mix i.e. 80 self-play +
  10 LfD actors — a data ratio, not a lift target). Ratio-correctness already
  proved (`ratio_of_legalSoftmax_eq`, `clippedRatio_mem_Icc`,
  `maskedPPO_zero_grad_illegal_logits`). Gate: promote only on head-to-head
  placement gain with significance held across two seeds with no safety regression.
- [ ] Value network with counterfactuals (CVPN: "what if I had folded?").
  Gate: prediction error drops on held-out nights before it is allowed to steer play.
- [ ] Promotion harness (PrPl / fixed-N). Mirror already exposes
  `log_eval_report` / `log_promotion` / `log_duplicate_audit` (no call sites
  yet); fixed-N formula proved (`Implementation/Evaluation.lean`); thresholds
  pre-registered from `lean/Formal/Blueprint/Curriculum.lean` and
  `tests/integration/test_schedule_wp03b.py`. Gate: fixed game count, frozen
  opponent pool; no ad-hoc rematches.

## 5. NOT NOW — explicitly parked

Flag state: `HYDRA2_*` inventory is `CLEARML_ENABLED` / `CLEARML_DISABLED` /
`CLEARML_OFFLINE_DIR`, `ARTIFACT_ROOT`, `REQUIRE_PIXI_LOCK`, `SKIP_GPU_SOAK`,
`TENHOU_MOUNT` — none of the items below have code paths yet, so any future
path MUST land behind a default-off flag (schedule/block tests guard
walls/schedules, not flags).

- [ ] Online search learning (pMCPA). Gate to revisit: only after baseline + one NEXT item are stable; needs live-search budget first.
- [ ] Decision-Transformer / ARDT sequence models. Gate to revisit: only if cloning + conservative steps plateau for two full rounds.
- [ ] Search-leaf swaps (changing what the search evaluates at leaves). Gate to revisit: only after value network (CVPN) is trusted; hygiene rules in `lean/Formal/Blueprint/SearchHygiene.lean` apply.
- [ ] Hash-fallback removal timing. Gate to revisit: separate deprecation ticket with its own duplicate-block proof (`tests/integration/test_duplicate_block_wp06.py`); never bundled with a training change.
