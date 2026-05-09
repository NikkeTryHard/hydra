# Hydra Archive: Demoted Design/Planning Index

Purpose: keep historical signal from removed design docs without letting old doctrine compete with current repo truth.

## Read order / authority

1. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` — canonical archive SSOT / upstream research intake. Do not edit in doc cleanup.
2. `research/design/HYDRA_FINAL.md` — promoted architecture doctrine summary.
3. `research/design/HYDRA_RECONCILIATION.md` — promoted execution doctrine / active path.
4. `docs/CURRENT_STATUS.md` — shipped/staged status.
5. `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, current code — runtime truth.
6. This file — historical parking lot only. Not impl authority.

## Demoted source docs now parked here

### HYDRA_SPEC.md — old architecture specification

Summary: historical 5-head / inference-search architecture snapshot: Tenhou Houou target, SE-ResNet rationale, 46-action Mortal-compatible mapping, safety-plane idea, licensing constraints, latency/VRAM targets. Contains stale assumptions like `85 x 34` full-input framing, 40-block baseline, old head set, and design-target benchmarks.

Why demoted: duplicates promoted architecture docs and conflicts with current code/reconciliation in model shape and priority. Useful only for rationale: Tenhou edge cases, no-pooling reason, explicit safety-plane motivation, AGPL/Mortal-policy boundary.

Promoted replacements: `HYDRA_FINAL.md` for architecture; `HYDRA_RECONCILIATION.md` for active scope; `docs/GAME_ENGINE.md` + code for encoder/runtime details; `docs/COMPATIBILITY_SURFACE.md` for public compatibility.

### IMPLEMENTATION_ROADMAP.md — old code build plan

Summary: historical step-by-step build checklist for `hydra-train` / `hydra-core`: backbone, heads, model, dataset, training loop, teacher/oracle, CT-SMC/AFBS, ExIt/self-play, endgame/Sinkhorn/Hand-EV/SaF/eval/inference. Preserves useful task granularity, tests, and Rust/Burn impl cautions.

Why demoted: too large and too prescriptive for current agents; many items are already shipped, staged differently, or superseded by reconciliation/status docs. Treat as archived backlog map, not build order.

Promoted replacements: `HYDRA_RECONCILIATION.md` for what to build; `docs/CURRENT_STATUS.md` for shipped/staged reality; crate READMEs and current source for module contracts; `docs/TRAINING_WORKFLOWS.md` / `docs/PREFLIGHT_AND_RUNTIME_SELECTION.md` for runbooks.

### OPPONENT_MODELING.md — expanded opponent-model rationale

Summary: detailed rationale for Hydra's edge over Mortal: explicit genbutsu/suji/kabe/one-chance safety planes, tenpai/danger heads, damaten detection, wait-set/value-conditioned/call-intent extensions, belief/search reserve ideas, CVaR/deception/Sinkhorn research parking lot. Strongest preserved facts: Mortal lacks explicit opponent model; damaten and multi-threat defense are known weak spots; safety-plane counts were hypotheses needing ablation.

Why demoted: mixes live rationale, active-ish architecture, and speculative reserve modules. Keeping it active invites accidental scope creep: extra heads, belief stacks, deception/RSA, CVaR, broad AFBS/search before base pipeline is healthy.

Promoted replacements: `HYDRA_FINAL.md` for current head/encoder doctrine; `HYDRA_RECONCILIATION.md` for promoted opponent-model scope; `docs/GAME_ENGINE.md` for live channel layout and safety semantics; `docs/DELTAQ_PROMOTION.md` / `docs/REPLAY_SIDECARS.md` where sidecar targets are active.

### REWARD_DESIGN.md — reward evidence and old decision note

Summary: compact reward-analysis source: RVR Mahjong paper uses oracle relative value + expected reward at `T-1` to reduce hidden-information and last-tile variance; Hydra reward notes favored per-kyoku GRP delta, placement vector swaps, frozen GRP lifecycle, running-std normalization, zero-sum/placement-aware rewards, and no extra shaping/intrinsic motivation.

Why demoted: combines evidence, historical final-decision language, and dead references. Reward truth now belongs in promoted doctrine and training/runtime docs; this survives as evidence index only.

Promoted replacements: `HYDRA_RECONCILIATION.md` for reward/training doctrine; `docs/TRAINING_WORKFLOWS.md` for current workflow; `docs/GAME_ENGINE.md` for scoring/runtime; `research/evidence/` and `research/intel/` for paper/community evidence.

## Reserve ideas preserved, not active defaults

- Explicit safety planes: genbutsu, suji, kabe, one-chance, tenpai hints. Keep as rationale; validate through current ablation/test path.
- Opponent auxiliary heads: tenpai, danger, next-discard, value-conditioned tenpai, wait-set, call-intent. Only current promoted heads/contracts count.
- Oracle/RVR variance reduction: oracle critic, zero-sum value baseline, expected reward at final pre-terminal state. Evidence useful; impl priority must come from reconciliation/current code.
- Search/belief extensions: CT-SMC, AFBS, robust opponent nodes, Sinkhorn tile allocation, Hand-EV, SaF, ExIt. Reserve unless promoted by current docs/status.
- Risk/exploitation reserve: dynamic risk lambda, CVaR-on-GRP, safety reserve feature, lateral movement predictor, deception/RSA. Do not add by default.

## Anti-drift rules for agents

- If this archive conflicts with `HYDRA_FINAL.md`, `HYDRA_RECONCILIATION.md`, docs under `docs/`, or code, archive loses.
- Do not resurrect old `85 x 34` full-input doctrine; current live encoder uses fixed-shape `192 x 34` superset unless code says otherwise.
- Do not use AGPL/Mortal-Policy code. Mortal/Mortal-Policy facts here are comparison/rationale only.
- Do not treat deleted source names as active docs. They were intentionally removed from active Markdown tree and summarized here.
