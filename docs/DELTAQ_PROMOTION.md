# DeltaQ Promotion

Op guide for Hydra DeltaQ promotion path, read resulting promotion artifact.

Doc explains current CLI contract, offline + policy-transfer gates, optional arena confirmation, shape of persisted `delta_q_promotion.json` artifact. Main training entrypoint: [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md). Current shipped-vs-staged truth: [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).

## What DeltaQ promotion is for

Hydra DeltaQ lane implemented, intentionally not default-on. Promotion answers narrow question:

> Is this candidate checkpoint good enough on the DeltaQ-specific validation surfaces to justify further confirmation and possible adoption?

Not normal training. Gated eval workflow. Compares candidate vs baseline checkpoint, records structured decision artifact.

## Current status

Current promoted status:

- DeltaQ implemented, not default-on
- promotion artifacts persist explicit `arena_decision` and `arena_report`
- BF16 not supported in DeltaQ promotion mode yet

Status intentional. Do not treat workflow as baseline default Hydra training path.

## Invocation shape

Hydra runs DeltaQ promotion through main `train` binary, not separate standalone command.

CLI shape:

```bash
train <config.yaml> --delta-q-promotion --delta-q-baseline-checkpoint <path>
```

Baseline checkpoint required because promotion is comparative. Hydra loads baseline checkpoint, evaluates candidate relative to it.

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

Runtime error explicit: BF16/autocast not supported for DeltaQ promotion yet.

### Baseline checkpoint is mandatory

If `--delta-q-baseline-checkpoint` missing, Hydra fails promotion run. Not optional because arena confirmation and offline comparison both depend on baseline model.

## High-level decision flow

Promotion path has up to three stages:

1. offline DeltaQ gate
2. DeltaQ policy-transfer gate
3. arena confirmation

Hydra first runs validation with policy baseline, then inspects DeltaQ-specific summaries from that validation pass.

### 1) Offline DeltaQ gate

Offline gate consumes `DeltaQPromotionReport` and `DeltaQPromotionResult`.

Important metrics in report:

- eligible and compared states
- candidate vs baseline top-1 agreement
- candidate vs baseline mean regret
- mean decision lift
- negative lift fraction
- candidate-beats-baseline rates
- high-gap behavior summaries

Report answers narrow question:

- does candidate look better than baseline on replay-derived DeltaQ decision quality?

### 2) Policy-transfer gate

Hydra can also compute policy-transfer report and result.

Gate exists because candidate can look better on narrow DeltaQ metric yet transfer policy quality badly.

Important policy-transfer metrics:

- compared states
- candidate/baseline top-1-to-teacher rates
- candidate/baseline mean teacher regret
- candidate-beats-baseline count/rate
- negative transfer fraction

Operational meaning:

- passing offline gate alone not enough if policy-transfer gate says candidate still transfers badly
- promotion stays conservative by design

### 3) Arena confirmation

If pre-arena recommendation says candidate promising enough, Hydra creates arena-confirmation request and runs paired arena confirmation.

Arena step exists because offline replay-derived signals are not final authority for promotion.

Resulting arena report captures:

- compared games
- baseline mean placement
- candidate mean placement
- delta mean placement
- baseline and candidate stable-dan scores
- lower and upper confidence bounds for mean placement

## Recommendation states you will see

Hydra exposes small recommendation vocabulary through promotion result and artifact.

Two operator-critical states:

- `RejectAtOfflineGate`
- `RequiresArenaConfirmation`

Interpret literally:

- `RejectAtOfflineGate` = candidate should not advance further on current evidence.
- `RequiresArenaConfirmation` = candidate passed enough pre-arena criteria to justify paired arena step.

If arena step runs, final persisted artifact also records arena-side decision.

## The persisted artifact

Promotion writes pretty-printed JSON artifact at:

```text
<output_dir>/delta_q_promotion.json
```

Persisted artifact currently includes:

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

Artifact = operator-facing truth for what happened during promotion run.

### Key fields to read first

#### `recommendation`

Pre-arena recommendation computed from offline and policy-transfer gates.

#### `stage`

Tells whether artifact reflects:

- offline-only gating
- offline + transfer + arena confirmation

Fastest way to tell whether run stopped before arena confirmation or continued through it.

#### `arena_confirmation`

If present, records arena configuration request Hydra constructed for paired confirmation.

#### `arena_decision`

If present, arena-side promotion decision derived from paired evaluation result.

#### `arena_report`

If present, paired arena summary itself. Use it to judge whether candidate held up beyond replay-derived metrics.

#### `policy_transfer` and `policy_transfer_result`

These explain whether candidate policy behavior stayed acceptable even if narrower DeltaQ metrics improved.

## Console output you should expect

Hydra prints few specific promotion phases:

- DeltaQ offline/transfer gate banner
- offline gate summary
- optional arena confirmation summary
- optional arena decision summary
- optional policy-transfer holdout summary
- policy-transfer gate summary

Operationally: console output = fast human read. `delta_q_promotion.json` = durable artifact for archive and later comparison.

## Recommended operator workflow

1. Choose candidate checkpoint to evaluate.
2. Choose baseline checkpoint representing current accepted policy.
3. Run `train config.yaml --delta-q-promotion --delta-q-baseline-checkpoint <baseline>`.
4. Read console summaries for quick decision path.
5. Inspect `delta_q_promotion.json` for durable structured result.
6. Do not treat `RequiresArenaConfirmation` as acceptance; it means candidate earned arena step, not that it already won it.

## Common failure modes and interpretation mistakes

- Missing baseline checkpoint is hard invocation error, not soft warning.
- BF16/autocast currently unsupported here even though BC preflight and BC training support BF16 paths.
- Passing offline gate does not guarantee final promotion if policy transfer or arena confirmation disagrees.
- Persisted `recommendation` not same thing as final arena-backed promotion decision.
- DeltaQ lane implemented, but still intentionally not default-on.

## Relationship to the main training docs

`docs/TRAINING_WORKFLOWS.md` already lists DeltaQ promotion as one training mode. This doc explains operational meaning of that mode and emitted artifact; it does not replace top-level mode overview.

## Where to read next

- Need main training mode overview? Read [`docs/TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md).
- Need current shipped/staged truth? Read [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md).
- Need replay-side supervision lanes rather than promotion? Read [`docs/REPLAY_SIDECARS.md`](REPLAY_SIDECARS.md).