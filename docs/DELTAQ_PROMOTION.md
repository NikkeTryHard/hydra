# DeltaQ Promotion

Operator guide for running Hydra's DeltaQ promotion path and interpreting the resulting promotion artifact.

This document explains the current CLI contract, the offline and policy-transfer gates, the optional arena confirmation step, and the shape of the persisted `delta_q_promotion.json` artifact. For the main training entrypoint, read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). For current shipped-vs-staged truth, read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).

## What DeltaQ promotion is for

Hydra's DeltaQ lane is implemented, but it is intentionally not default-on. Promotion exists to answer a narrow question:

> Is this candidate checkpoint good enough on the DeltaQ-specific validation surfaces to justify further confirmation and possible adoption?

This is not normal training. It is a gated evaluation workflow that compares a candidate against a baseline checkpoint and records a structured decision artifact.

## Current status

Current promoted status is:

- DeltaQ is implemented but not default-on
- promotion artifacts persist explicit `arena_decision` and `arena_report`
- BF16 is not supported in DeltaQ promotion mode yet

That status is intentional. Do not treat this workflow as the baseline default path for Hydra training.

## Invocation shape

Hydra runs DeltaQ promotion through the main `train` binary, not a separate standalone command.

CLI shape:

```bash
train <config.yaml> --delta-q-promotion --delta-q-baseline-checkpoint <path>
```

The baseline checkpoint is required because promotion is explicitly comparative. Hydra loads the baseline checkpoint and evaluates the candidate relative to it.

### Example

```bash
cargo run -p hydra-train --bin train -- \
  /config/train.yaml \
  --delta-q-promotion \
  --delta-q-baseline-checkpoint /models/baseline/model_base
```

## Hard gates before promotion runs

### BF16 is rejected

DeltaQ promotion mode currently rejects:

```yaml
precision_mode: bf16_autocast
```

The runtime error is explicit: BF16/autocast is not supported for DeltaQ promotion yet.

### Baseline checkpoint is mandatory

If `--delta-q-baseline-checkpoint` is missing, Hydra fails the promotion run. This is not optional because arena confirmation and offline comparison both depend on a baseline model.

## High-level decision flow

The promotion path has up to three stages:

1. offline DeltaQ gate
2. DeltaQ policy-transfer gate
3. arena confirmation

Hydra first runs validation with a policy baseline, then inspects the DeltaQ-specific summaries produced by that validation pass.

### 1) Offline DeltaQ gate

The offline gate consumes a `DeltaQPromotionReport` and a `DeltaQPromotionResult`.

Important metrics in the report include:

- eligible and compared states
- candidate vs baseline top-1 agreement
- candidate vs baseline mean regret
- mean decision lift
- negative lift fraction
- candidate-beats-baseline rates
- high-gap behavior summaries

The report answers a narrow question:

- does the candidate look better than the baseline on replay-derived DeltaQ decision quality?

### 2) Policy-transfer gate

Hydra can also compute a policy-transfer report and result.

This gate exists because a candidate that looks better on one narrow DeltaQ metric can still transfer policy quality badly.

Important policy-transfer metrics include:

- compared states
- candidate/baseline top-1-to-teacher rates
- candidate/baseline mean teacher regret
- candidate-beats-baseline count/rate
- negative transfer fraction

Operationally, this means:

- passing the offline gate alone is not enough if the policy-transfer gate says the candidate still transfers badly
- promotion remains conservative by design

### 3) Arena confirmation

If the pre-arena recommendation says the candidate is promising enough, Hydra creates an arena-confirmation request and runs paired arena confirmation.

The arena step exists because offline replay-derived signals are not treated as the final authority for promotion.

The resulting arena report captures:

- compared games
- baseline mean placement
- candidate mean placement
- delta mean placement
- baseline and candidate stable-dan scores
- lower and upper confidence bounds for mean placement

## Recommendation states you will see

Hydra exposes a small recommendation vocabulary through the promotion result and artifact.

The two operator-critical states are:

- `RejectAtOfflineGate`
- `RequiresArenaConfirmation`

Interpret them literally:

- `RejectAtOfflineGate` means the candidate should not advance further on current evidence.
- `RequiresArenaConfirmation` means the candidate passed enough pre-arena criteria to justify the paired arena step.

If the arena step runs, the final persisted artifact also records the arena-side decision.

## The persisted artifact

Promotion writes a pretty-printed JSON artifact at:

```text
<output_dir>/delta_q_promotion.json
```

The persisted artifact currently includes:

- `scope`
- `step_or_epoch`
- `recommendation`
- `stage`
- `arena_confirmation`
- `arena_decision`
- `arena_report`
- `report`
- `result`
- `policy_transfer`
- `policy_transfer_result`

This artifact is the operator-facing truth for what happened during the promotion run.

### Key fields to read first

#### `recommendation`

This is the pre-arena recommendation computed from the offline and policy-transfer gates.

#### `stage`

This tells you whether the artifact reflects:

- offline-only gating
- offline + transfer + arena confirmation

Treat this as the fastest way to tell whether the run stopped before arena confirmation or continued through it.

#### `arena_confirmation`

If present, this records the arena configuration request Hydra constructed for paired confirmation.

#### `arena_decision`

If present, this is the arena-side promotion decision derived from the paired evaluation result.

#### `arena_report`

If present, this is the paired arena summary itself and should be used to judge whether the candidate actually held up beyond replay-derived metrics.

#### `policy_transfer` and `policy_transfer_result`

These explain whether the candidate's policy behavior remained acceptable even if the narrower DeltaQ metrics improved.

## Console output you should expect

Hydra prints a few specific promotion phases:

- DeltaQ offline/transfer gate banner
- offline gate summary
- optional arena confirmation summary
- optional arena decision summary
- optional policy-transfer holdout summary
- policy-transfer gate summary

Operationally, the console output is the fast human read, while `delta_q_promotion.json` is the durable artifact you can archive and compare later.

## Recommended operator workflow

1. Choose the candidate checkpoint you want to evaluate.
2. Choose a baseline checkpoint that represents the current accepted policy.
3. Run `train config.yaml --delta-q-promotion --delta-q-baseline-checkpoint <baseline>`.
4. Read the console summaries for a quick decision path.
5. Inspect `delta_q_promotion.json` for the durable structured result.
6. Do not treat `RequiresArenaConfirmation` as acceptance; it means the candidate earned the arena step, not that it has already won it.

## Common failure modes and interpretation mistakes

- Missing baseline checkpoint is a hard invocation error, not a soft warning.
- BF16/autocast is currently unsupported here even though BC preflight and BC training support BF16 paths.
- Passing the offline gate does not guarantee final promotion if policy transfer or arena confirmation disagrees.
- A persisted `recommendation` is not the same thing as a final arena-backed promotion decision.
- The DeltaQ lane is implemented, but it is still intentionally not default-on.

## Relationship to the main training docs

`docs/TRAINING_WORKFLOWS.md` already lists DeltaQ promotion as one training mode. This document exists to explain the operational meaning of that mode and the artifact it emits; it does not replace the top-level mode overview.

## Where to read next

- Need the main training mode overview? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need current shipped/staged truth? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).
- Need replay-side supervision lanes rather than promotion? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).
