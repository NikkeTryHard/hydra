# Hydra2 Algorithm Experiment Blueprint

**Status:** Research design draft. No Hydra2 planner, model, belief system, simulator adapter, or evaluation runner exists yet.  
**Date:** 2026-07-28  
**Rules authority:** Tenhou four-player hanchan rules.  
**Companion direction plan:** [PROJECT_PLAN.md](./PROJECT_PLAN.md)  
**Normative build backlog:** [BUILD_EXECUTION_PLAN.md](./BUILD_EXECUTION_PLAN.md)  
**Canonical implementation APIs:** [IMPLEMENTATION_SPEC.md](./IMPLEMENTATION_SPEC.md)

This document specifies research targets and measurements required before claiming an algorithmic result. `BUILD_EXECUTION_PLAN.md` controls executable sequencing and evidence; `IMPLEMENTATION_SPEC.md` controls exact builder-facing schemas, APIs, and pseudocode. Every planner has a target, inputs, state, transition semantics, correctness tests, comparison arms, stop conditions, and promotion record.

## 1. Locked Decisions and Non-Claims

### 1.1 Rules and platform

Hydra2 targets `tenhou_4p_hanchan_v1`: standard four-player Tenhou hanchan, not a generic or engine-default notion of riichi.

The authoritative source is Tenhou's rule/manual page: <https://tenhou.net/man/>. The initial manifest must preserve the source URL, retrieval date, content digest, and every selected rule flag. The official page establishes at least:

- 25,000 starting points; 30,000 return points;
- Tenhou 10-20 uma; three red fives, one in each suit;
- multiple ron; abortive draws including nine terminals, four winds, four kans, four riichi, and three ron;
- nagashi mangan; bankruptcy at negative score; zero continues;
- standard `5+10` and fast `3+5` clocks;
- Tenhou end/renchan behavior described by the current manual.

The first rules artifact, `tenhou_4p_hanchan_v1.json`, must additionally record every scoring and match flag not safely inferred from prose: kuitan, kuikae, furiten, chankan, rinshan, kan-dora/ura timing, pao, yakuman/kazoe policy, multiple-ron stick allocation, all-last, agari-yame, tobi, sudden-death, tie-break, and exact placement conversion. Engines may not fill omitted fields from their defaults.

`RiichiEnv==0.4.8` is the first reference-engine adapter, not the rules authority. MahJax at commit `3fa282699e5786d165216578bc8e213f96a0dca5` is an optional accelerator after conformance. Neither may silently redefine Tenhou rules.

### 1.2 LuckyJ and deployment boundary

Hydra2 has no authorized LuckyJ model, source, checkpoint, API, or private training data. Do not claim to reproduce, query, clone, reverse engineer, or directly control LuckyJ. Hydra2 is offline research/simulation only: no live Tenhou, Mahjong Soul, or other platform client. Public LuckyJ logs support observational policy/distribution analysis only; they do not reveal hidden state, policy, value, search, training, internal latency, or causal playing strength.

### 1.3 Novelty boundary

The central research candidate is a **public-belief reusable event forest** (PBRF): one parent belief population fans into predicted actor-visible event children, computes during opponent time, commits only the realized transition-correct child, and semantically discards incompatible work.

Do not claim novelty for Bayes conditioning, particle filtering, ESS, common random numbers, MCTS rerooting, pondering, GPU batching, or speculative execution individually. The defensible claim, if independently verified, is an implementation and empirical composition: provenance-safe speculative public-belief planning for four-player Tenhou riichi.

### 1.4 Objective, compute, and data boundary

Primary utility is expected final placement under the declared Tenhou utility manifest. Preserve raw score/rank vectors; points and first/fourth-place rates remain diagnostics. Development order: supervised policy/value baseline, then belief/search, then optional self-play RL.

`gameplay_5s` has a 5,000 ms own-turn hard deadline including synchronization, planner overhead, and fallback margin. `ponder` uses opponent/public-event time but commits only transition-correct actor-visible work. `analysis` declares a larger finite budget separately and changes compute only, never information or semantics. RTX 5070 is default; A100 use charges a 2,000 GPU-hour reserve and requires measured local bottleneck/value justification.

Dataset provenance is confidential. User attests authority for model training and internal evaluation. Artifacts record a non-identifying source ID, authorization/purpose/disclosure fields, acquisition metadata if known, and content hashes; never sponsor identity or raw samples.

### 1.5 Prohibited claims and defaults

- No claim that any algorithm beats LuckyJ; this offline plan has no direct causal comparison protocol.
- No ReBeL equilibrium guarantee in four-player general-sum riichi.
- No learned latent dynamics as a replacement for exact rules, legality, tile movement, scoring, or settlement.
- No normalized finite-particle SMC mean called unbiased.
- No self-normalized or clipped importance estimate called unbiased.
- No parent-only likelihood reweighting called a successor belief after hidden state moves.
- No adaptive search statistic called confirmatory evidence.
- No blind estimator portfolio or unconstrained learned value-of-computation scheduler as a default.
- No inference from duplicate walls that divergent games are identical counterfactuals.

### 1.6 Hydra1 transfer boundary

Hydra1 commit `166e3ebd31b45db614695bccda3308d626200d98` was audited by direct source reading. This is design evidence only: Hydra2 does not import that checkout, accept its tests as Hydra2 qualification, or inherit its names as implementation status.

| Hydra1 mechanism | Decision | Hydra2 boundary |
| --- | --- | --- |
| Exact checkpoint/optimizer/RNG resume; frozen policy-snapshot identity; strict rollout validation; ONNX hash/parity fixture | ADOPT semantics, independently implement | Canonical Hydra2 envelopes/hashes, plain-PyTorch/Fabric equivalence, and fresh runtime parity remain authoritative. |
| Decision-before-transition capture; explicit omitted-pass reconstruction; red-aware action mapping | ADAPT contracts/tests | Rebuild through Hydra2 canonical actions, packet priority, visibility types, strict quarantine, and qualified RiichiEnv adapter. |
| Direct-sampled Actor-Critic Hedge loss | ADAPT as optional WP-11 objective | Same frozen rollout as PPO; exact formula/mask/gradient fixtures; no strength claim before held-out duplicate-block evidence. |
| Exact actor-visible shanten, own waits, public unseen counts, ukeire | ADAPT as optional model-input ablation | Availability/shape only; new schema identity; hidden-permutation and playing-strength gates. |
| Exact opponent waits and safety residuals from concealed hands | ADAPT only as privileged training labels | Separate privileged namespace; never inference input, runtime safety rule, or unqualified target. |
| Batched all-legal one-step afterstate values and provenance-keyed sidecars | ADAPT plumbing only | May inform teacher-record generation, but Hydra1's root-only fixed-value PUCT is not a Candidate 1-6 planner or five-gate Candidate 7 teacher. |
| CT-SMC contingency-table DP | REJECT current probability claim/implementation | Current weights omit physical allocation multiplicities; derive target law and exact finite tests before any independent proposal. |
| Mixture-SIB teacher/filter | REJECT | Uniform identical components, heuristic trust, and no qualified sequential posterior. |
| AFBS tree and action-XOR ponder cache | REJECT as search/persistence | No hidden-world transition, opponent information-set rollout, event packet partition, belief epoch, or transition-correct reuse. |
| Robust-opponent KL/archetype module | REJECT for Candidate 8 | Support/KL and feasibility gaps; type-only update lacks joint type/world posterior and coherent trajectory policy. |
| Hand-EV `tenpai_prob`, `win_prob`, `expected_score` | REJECT | Multi-draw continuation factors and score bonuses are hand-built heuristics, not calibrated probabilities or exact EV. |
| Hydra1 engine, 46-action ABI, permissive replay, FNV/path split, compact shards | REJECT direct reuse | Incomplete Tenhou authority, contextual visibility safety, silent replay omissions, and insufficient lineage/integrity. Use edge cases only as fixture inventory. |
| Hydra1 paired arena/bootstrap and stable-dan transform | REJECT as formal evaluation | Seat groups use different walls; game-level bootstrap ignores wall clustering; raw vectors/provenance/resources are incomplete. Hydra2 wall-block protocol supersedes it. |
| Burn trainer, dual Rust/Python configuration, probe/autotune orchestration | REJECT | Project-owned PyTorch loop and one configuration authority; add optimization only from measured bottlenecks. |

No Hydra1 component satisfies a Hydra2 milestone. Transfer means reimplement the narrow invariant, run Hydra2 fixtures, and publish Hydra2 completion evidence. Archived performance, novelty, or maturity prose is noncausal and nonauthoritative.

## 2. Beginner Terms and Visibility Rules

| Term | Meaning in Hydra2 |
| --- | --- |
| Public state | Facts every player can infer: round, scores, visible tiles, calls, discards, riichi, revealed dora, turn/phase, and rules. Concealed tile identities are excluded. |
| Actor observation | Public state plus one seat's concealed hand, own drawn tile where visible, and seat-specific legal facts. |
| Full world | A complete tile-conserving hidden completion: all concealed hands, wall/dead-wall order, unrevealed dora/ura, and model latent state if used. |
| Belief | Probability law over full worlds consistent with one actor observation. |
| Particle | One sampled full world plus provenance and, if proposal-sampled, its density information. |
| Actor-visible packet | A sanitized event delivered to one actor. It includes only fields that actor may observe. |
| Public packet | A packet every seat may observe. An opponent's drawn tile is never public. |
| Event forest | Predicted children indexed by possible next actor-visible packets. |
| Nonanticipativity | Same actor observation implies the same action distribution, regardless of hidden world. |
| Strategy fusion | Illegal behavior: choosing different actions in worlds the actor cannot distinguish. |
| Natural sample | A world sampled directly from the frozen target belief. |
| Proposal sample | A world sampled from another law. Corrected estimators require the full target/proposal ratio exactly once. |
| Confirmation | Fresh natural, full-fidelity evaluation after candidate actions are frozen. |
| Pondering | Computation after Hydra2 emits an action and before its next actor-visible packet. |
| Commit/squash | Promote only the realized transition-correct child; invalidate sibling-specific estimates and caches. |

A turn transition can be inferable from public order while its drawn tile remains hidden. The event schema therefore distinguishes `turn_advance(actor)` from `draw_tile(actor, tile)`. Only the drawing actor receives the tile identity. A server-private event must never enter an actor observation, planner key, cache, log, or model input.

## 3. Formal Objective

At a committed history `h`, let root seat be `i`.

```text
I_i(h)       actor i's observation
X(h)         full worlds consistent with I_i(h)
b_h(x)       frozen target belief over X(h)
A_i(h)       exact legal root actions
R_a          terminal raw score/rank outcome vector in R^4 after forcing a
U_T(R_a)     declared expected-placement utility vector in R^4
s_i(U)       acting seat i's expected-placement scalar U[i]
Q_i(h,a)     E_{P_h^a}[s_i(U_T(R_a))]
Delta_ab     Q_i(h,a) - Q_i(h,b)
```

`R_a` and `U_T(R_a)` remain vectors throughout simulation, training targets, backups, and result logs. Only root selection uses `s_i`. Do not assume a utility vector sums to zero unless its manifest explicitly proves it.

For a coupling `Gamma_h^{ab}` with correct action-conditional marginals,

```text
Delta_ab = E_{Gamma_h^{ab}}[
  s_i(U_T(R_a)) - s_i(U_T(R_b))
].
```

A coupling may share exogenous random numbers to reduce variance. It must not force later public events, opponent actions, or legal choices to be identical when the forced root actions change their information. Never optimize `E[max_a s_i(U_T(R_a))]`; that is clairvoyant and differs from `max_a E[s_i(U_T(R_a))]`.

### 3.1 Successor belief

Let `e` be the next packet visible to root actor `i`. The exact transition kernel includes hidden draws, actor-information-measurable policies, call/pass priority, and physical tile movement:

```text
K_h^a(dx', de | x) = probability of successor world x' and packet e
                     after forced action a from world x.

P_h^a(dx', e) = integral b_h(dx) K_h^a(dx', de | x)
Z_e            = integral integral b_h(dx) K_h^a(dx', de | x)
b_{h,a,e}(dx') = P_h^a(dx', e) / Z_e.
```

A draw, call, kan, rinshan draw, or hidden tile transfer normally changes `x` into `x'`. Reweighting parent particles by `P(e | x)` without the pushforward through `K_h^a` is not a valid successor belief.

### 3.2 Information-set invariant

For every actor `j` and compatible worlds `x, x'`:

```text
I_j(x) = I_j(x')  =>  pi_j(. | x) = pi_j(. | x') = pi_j(. | I_j).
```

A policy key may include actor `j`'s private hand and public facts available to `j`. It must not include root particle ID, wall order, another player's concealed hand, future packets, server-private events, or posterior features conditioned on unavailable information.

### 3.3 Confirmation estimator

For frozen candidates `C`, draw independent natural worlds and branch-correct continuation randomness:

```text
for k in 1..n:
  X_k ~ b_h
  for a in C:
    R_k[a] <- exact_rules_rollout(X_k, forced_action=a, structural_coupling)

D_k[a,b] = s_i(U_T(R_k[a])) - s_i(U_T(R_k[b]))
hat_Delta_ab = mean_k D_k[a,b]
```

Each action retains its own correct conditional future law. Shared randomness changes variance only. Confirmation uses independent IID pairs, independent randomized-QMC scrambles, or independent SMC populations as its uncertainty unit, as declared before results are inspected.

## 4. Required Interfaces and Isolation Boundary

The following semantics are required even if Rust, Python, shared memory, RPC, or CUDA implementations differ.

```text
ActorObservation:
  public_state, own_hand, own_private_events, seat, legal_facts,
  rules_hash, observation_hash

FullWorld:
  complete tile assignment and engine state; simulation-sandbox only

EventEnvelope:
  sequence, actor, kind, visibility in {public, actor_private, server_private},
  red-aware payload, public_state_delta

Transition:
  next_world, emitted_envelopes, next_actor, terminal, raw_score_vector

Model.evaluate(obs, legal_mask, utility_id) ->
  {policy_logits, value_vector_distribution, event_likelihoods, diagnostics}

Belief.begin(obs, model, rng) -> BeliefEpoch
Belief.sample_natural(epoch, n, rng) -> [FullWorld]
Belief.sample_proposal(epoch, proposal_id, n, rng) ->
  [{world, log_target, log_proposal, provenance}]
Belief.transition_and_condition(epoch, actor_visible_packet, simulator, model, rng) -> BeliefEpoch
Belief.actor_conditional(epoch, actor_observation, immutable_constraints, rng) ->
  {world, log_target, log_proposal, provenance}

Simulator.actor_view(world, seat) -> ActorObservation       # sandbox only
Simulator.legal_actions(obs) -> LegalMask
Simulator.apply_action(world, seat, action, rng) -> Transition
Simulator.advance_until_actor_packet(world, root_seat, rng) -> Transition
Simulator.terminal_return(world) -> RawScoreVector
Simulator.rules_hash() -> Digest

Planner.observe(actor_visible_packet, deadline) -> PlannerState
Planner.act(actor_observation, deadline) -> ActionDecision
Planner.ponder(window) -> PonderReport
Planner.metrics() -> PlannerMetrics

ExperimentRunner.confirm(case, frozen_candidates, plan) -> ConfirmationRecord
ExperimentRunner.run_block(block, agents, latency_schedule) -> BlockRecord
```

Only simulation workers receive `FullWorld`. The root-facing planner receives `ActorObservation`, legal masks, and aggregate values. A process/API boundary and hidden-tile canaries must enforce this distinction.

Planner mode is explicit in every call/specification: `gameplay_5s`, `ponder`, or `analysis`. A mode may change deadline/resource allocation only. Same observation, rules, utility, legal set, model identities, and estimator semantics remain mandatory.

## 5. Reproducibility Contract

Every formal run writes canonical JSON according to RFC 8785, hashes it with SHA-256, and records:

```text
rules_hash, engine_adapter_hash, model_checkpoint_hash, belief_model_hash,
utility_id, observation_schema_hash, action_schema_hash, proposal_id,
continuation_policy_ids, candidate_spec_hash, case_manifest_hash,
seed_manifest_hash, pixi_lock_hash, Python/CUDA/driver/PyTorch/JAX versions,
runtime_adapter_id, runtime_adapter_version, fabric_version_or_null,
compile_backend, compile_mode, compile_fullgraph, compile_dynamic,
backward_pass_autocast, compile_options_hash, compile_artifact_identity, extension_versions,
GPU name/compute capability, cuDNN version, container_or_wheel_digest,
hardware/environment manifest, source/dataset hashes
```

Derive counter-based random streams from immutable labels, never call order:

```text
seed = SHA256(master_seed || experiment_id || split_id || case_id ||
              wall_seed || root_seat || candidate || belief_epoch ||
              purpose || population_id || scenario_id || action_id ||
              depth || replicate_id)
```

Allowed `purpose` values include `belief_natural`, `proposal`, `transition`, `policy`, `coupling`, `search`, `pilot`, `confirmation`, and `evaluation`. Retries retain logical seeds and add an attempt identifier.

A candidate specification must exist before an experiment:

```json
{
  "candidate_id": "pbrf_core_v1",
  "case_manifest": "sha256:...",
  "oracle_or_confirmation": "exact_tiny_v1 | natural_full_fidelity_v1",
  "primary_metric": "expected_final_placement",
  "resource_view": {"mode": "gameplay_5s | ponder | analysis", "deadline_ms": 0, "model_calls": 0, "transitions": 0, "joules": 0, "gpu_hours": 0},
  "hardware": "rtx_5070 | a100",
  "runtime": {"adapter": "fabric_2.6.5 | plain_pytorch", "torch": "2.13.x", "compile_backend": "inductor", "compile_mode": "eager | default | max-autotune-no-cudagraphs | max-autotune", "backward_pass_autocast": "off | null", "compile_identity": "sha256:..."},
  "uncertainty_unit": "case | scramble | smc_population | wall_block",
  "pass_rule": "predeclared inequality and interval/CS",
  "failure_rule": "illegal | leak | replay_drift | deadline_policy_violation | uncharged_compute"
}
```

Numeric values are selected from a blinded pilot and frozen in this artifact. A prose phrase such as "improves" is not a promotion gate.

## 6. Prerequisites and Candidate Crosswalk

| Artifact/gate | Required by |
| --- | --- |
| Tenhou rules manifest, red-aware action codec, actor-visible event schema | Every candidate |
| Qualified RiichiEnv reference adapter and actor boundary | Every candidate |
| Frozen policy/value checkpoint with legal mask | Candidates 0-8 |
| Belief density/sampler and event-kernel qualification | Candidates 1-8 |
| Decision-case corpus and natural confirmation runner | Candidates 1-8 |
| Exact tiny-state oracle suite | Candidates 1-6 and PBRF modules |
| Duplicate-block evaluator | Promotion to match-strength claims |
| Confidential authorization attestation and content-addressed dataset manifest | Any corpus-backed training, calibration, or evaluation |

No candidate becomes executable merely because this document exists.

## 7. Candidate 0: Frozen Policy/Value

**Purpose:** establish legal, reproducible, low-latency baseline and fallback.

**Inputs/state:** `ActorObservation`, exact legal mask, frozen model, Tenhou utility ID, immutable history cache.

```text
observe(packet):
  require packet visible to root actor
  cache <- append_visible_packet(cache, packet)

act(obs):
  legal <- Simulator.legal_actions(obs)
  out <- Model.evaluate(obs, legal, utility_id)
  return predeclared_masked_choice(out.policy_logits, legal)
```

**Compute:** one cached model evaluation. No particles, search, pondering, online learning, or hidden state.

**Tests:** legal action on every qualification case; hidden-tile permutation invariance; cache/full-encode equality; deterministic replay; deadline fallback; red-five action round trip.

**Arms:** greedy policy; fixed-temperature stochastic policy; predeclared value tie break.

**Promotion:** zero legality/leak/replay failures and a frozen baseline record. This establishes no strength claim.

## 8. Candidate 1: Natural-Particle ISMCTS

**Purpose:** test information-set tree search without proposal correction, persistence, or adaptive estimator modules.

**Inputs/state:** Candidate 0; natural belief sampler; exact simulator; frozen actor-view continuation policies with versioned RNG semantics; vector leaf value; fresh root-seat information-set tree.

**Estimator rule:** baseline worlds are sampled only from `b_h`; backups carry no importance ratio. Root-seat information nodes use the declared UCT/selection rule and root scalar `s_i`. Every other seat samples its frozen legal-masked continuation policy `pi_j(. | I_j)` from that seat's actor view and named policy RNG stream. This defines the action-conditional continuation law; policy IDs and RNG semantics are part of provenance. A proposal-corrected variant is a separate ablation.

```text
repeat until declared simulation or deadline budget:
  world <- Belief.sample_natural(epoch, 1, rng)[0]
  path <- []
  while not terminal(world) and depth < declared_depth:
    seat <- actor_to_move(world)
    sim_obs <- Simulator.actor_view(world, seat)
    legal <- Simulator.legal_actions(sim_obs)
    if seat == root_seat:
      node <- tree.get_or_create(key=hash(sim_obs), seat=seat)
      action <- root_information_set_selection(node, legal, scalar=s_i)
      path.append(node, action)
    else:
      action <- sample(frozen_continuation_policy[seat](sim_obs, legal), policy_rng)
    world <- Simulator.apply_action(world, seat, action, rng).next_world
  if terminal(world):
    u <- U_T(Simulator.terminal_return(world))
  else:
    leaf_obs <- Simulator.actor_view(world, actor_to_move(world))
    leaf_legal <- Simulator.legal_actions(leaf_obs)
    u <- Model.evaluate(leaf_obs, leaf_legal, utility_id).value_vector_distribution.mean
  backup_vector(path, u)
return root_action_with_declared_scalar_rule(tree, root_seat=i)
```

At another actor's node, their private hand is available only inside the simulator sandbox because it is part of their own information state. Root selection never receives it. Optional re-determinization must be specified as a named conditional law `q_j(x | I_j, immutable_constraints)` with `b(x | I_j, immutable_constraints)/q_j(...)` applied exactly once. It must preserve public reach, root-known tiles, and every already observed packet. Until that law has a tiny-state proof, re-determinization is disabled.

**Tests:** equal actor observations map to equal root-node keys/actions and equal non-root policy distributions; opponents with changed actor-visible information may change distribution; policy RNG replay; root-known/public constraints survive conditional sampling; a two-world unequal-probability oracle detects double weighting; vector backup preserves raw settlement and utility-schema identity; no forbidden field appears in a tree key or continuation-policy input.

**Arms:** policy-only; natural ISMCTS; explicitly qualified conditional re-determinization; no root determinization only as a negative diagnostic.

**Promotion:** candidate spec reports noninferior natural-confirmation root decision metric at matched resource view, with zero invariant failures.

## 9. Candidate 2: Natural-Scenario DESPOT

**Purpose:** test sparse scenario-contingent planning before persistent belief reuse.

**Inputs/state:** Candidate 0; natural belief sampler; exact simulator; semantic random-stream generator; feasible blueprint policy; decision-case/confirmation runner; fresh scenario action-observation tree.

To retain ordinary DESPOT-style scenario interpretation, each scenario is sampled naturally:

```text
sigma_k = (X_k, xi_k),  X_k ~ b_h, xi_k ~ target continuation randomness.
```

No proposal-sampled or importance-corrected scenarios are allowed in this baseline. A weighted scenario-tree experiment must give a separate objective and proof; it must not call an arbitrary weighted number an upper bound.

```text
S <- [natural_scenario(epoch, rng) for k in 1..K]
root.lower_policy_value <- empirical_value_of_blueprint(S)
root.priority_proxy <- declared_search_priority(S)  # not called a bound unless proved
while budget remains:
  fringe <- select_declared_priority(root)
  expand_action_packet_children(fringe, S, exact_simulator)
  update_feasible_policy_values_and_priority(fringe)
return best_feasible_root_action(root)
```

A feasible policy value is an estimate of that policy, not a general-sum optimality certificate. An upper bound may be labeled only after a candidate-specific proof identifies its target and assumptions.

**Tests:** scenario replay; packet partition; lower-policy action legality; tiny POMDP/game exact comparison; proposal-reversal fixture proving that unweighted non-natural scenarios can choose the wrong action; packet aliasing rejection.

**Arms:** policy-only; ISMCTS; DESPOT; declared regularization/no-regularization; equal calls/transitions/joules views.

**Promotion:** only after natural confirmation wins the predeclared decision metric under at least one declared resource view.

## 10. Candidate 3: PBRF Core

### 10.1 Core state and equations

PBRF keeps an immutable parent population and event-conditioned child views. Let natural parents be `x_i ~ b_h`, `i=1..N`. For action `a`, let disjoint actor-visible packets form `E_a`. The exact finite transition construction is:

```text
gamma_hat[a,e](g) = (1/N) sum_i sum_{x'} K_h^a(x', e | x_i) g(x')
Z_hat[a,e]        = gamma_hat[a,e](1)
child_hat[a,e](g) = gamma_hat[a,e](g) / Z_hat[a,e]     when Z_hat[a,e] > 0
ESS[a,e]          = 1 / sum_r normalized_weight_r^2
```

`child_hat` is a finite-particle ratio estimator: useful for search, not an unbiased confirmation claim. Each child references a parent identifier plus transition delta when possible; it does not share child-specific statistics with siblings after commit.

Packets must be mutually exclusive and exhaustive under the declared next-packet semantics. A call/pass packet includes the complete priority resolution needed to make the successor unambiguous.

### 10.2 Exact algorithm

```text
parent <- Belief.sample_natural(epoch, N, rng)
candidates <- freeze_root_candidate_generator(parent, policy, legal_actions)
for a in candidates:
  for x in parent:
    for successor in enumerate_disjoint_next_packet_kernel(x, a):
      child[a, successor.packet].append(
        parent_id=x.id,
        successor_delta=successor.delta,
        raw_weight=(1/N) * successor.probability,
        provenance=target_id)
  require abs(sum_e Z_hat[a,e] - 1) <= declared_kernel_tolerance

The immediate next-packet kernel must be exhaustively enumerated in Candidate 3. Sampling is permitted only after an event child is constructed. A later sampled-fanout module must declare packet proposal `q(e,x' | x,a)`, use `K_h^a(x',e | x)/q(e,x' | x,a)` exactly once, and replace the exact per-parent partition assertion with a proposal-corrected Monte Carlo diagnostic.

fixed_allocate_search_batches(child, declared_schedule)
freeze candidates and every search-derived object
return ExperimentRunner.confirm(case, candidates, natural_full_fidelity_plan)
```

At a real packet `e_star`, recompute or verify the authoritative transition, create `b_{h,a,e_star}`, increment the belief epoch, admit only artifacts whose target/provenance remains valid, and squash every sibling-specific value, visit, posterior, or paired statistic.

**Tests:** finite enumerated kernel equals exact tiny-game probabilities; child normalizers partition one; pushforward equals rebuild; child/proposal provenance rejects stale versions; event packet duplicate/missing detection; structural coupling preserves branch marginals; commit result equals fresh rebuild within declared Monte Carlo tolerance.

**Arms:** natural parent-only sampled event; PBRF immediate fanout; independent coupling; semantic common-random-number coupling; ISMCTS; DESPOT.

**Promotion:** exact-tiny oracle pass plus frozen natural confirmation at matched calls/transitions/joules. A confirmation reversal, missing packet mass, or stale-child admission is a hard failure.

## 11. Candidate 4: PBRF Modules and Persistent Forest

PBRF core remains the control. Test one module, return to core, then test the next. Only separately promoted modules may enter a cumulative build. Every module uses frozen pilot parameters and a named `CandidateSpec`.

### 11.1 Transition Rao-Blackwellization

If a finite variable `Y` can be enumerated conditional on retained state `X`, replace sampled `g(X,Y)` with:

```text
RB(X) = sum_y P(Y=y | X) g(X,y).
E[RB(X)] = E[g(X,Y)].
```

**Steps:** choose a declared tractable variable; enumerate all legal `y`; apply exact transition/policy likelihood; sum; sample only residual randomness.

**Tiny test:** two-state, two-draw oracle with exact enumerated expectation; verify seeded sampled and RB estimators have equal expectation, and charge every policy/transition evaluation.

### 11.2 Defensive targeted MIS

For inherited proposal `q0`, targeted child proposal `q1`, and deterministic counts `n0,n1`, use:

```text
m(x) = (n0*q0(x) + n1*q1(x)) / (n0+n1)
gamma_hat_e(g) = (1/(n0+n1)) sum_r [b_h(x_r)L_e(x_r)g(x_r)/m(x_r)]
Z_hat_e        = gamma_hat_e(1)
```

`L_e` includes physical transition and actor-policy likelihood. The conditional ratio `gamma_hat_e(g)/Z_hat_e` is consistent but finite-sample biased; it is search-only unless a valid alternative estimator is supplied. Targeted sample counts never estimate `P(e)`.

**Steps:** trigger only from frozen support/ESS/normalizer rule; sample `q1`; retain source density for every sample; use the one balance denominator above; preserve a natural floor `epsilon`; prohibit clipping and a second correction.

**Tiny test:** unequal two-state law where applying `b/q` twice gives a known wrong value; verify expected unnormalized numerator and normalizer from repeated runs; verify a zero-support proposal is rejected.

### 11.3 Structural common random numbers

For each branch use shared primitive uniforms `u`, mapped through each branch's own conditional distribution:

```text
z_a = F_a^{-1}(u),  z_b = F_b^{-1}(u).
```

**Steps:** declare coupled primitive IDs; map each through branch-specific legal categorical/transition kernel; record empirical covariance; retain independent control.

**Tiny test:** empirical marginal frequencies match each target categorical law; negative-covariance fixture selects independent coupling; never force equal opponent actions after divergent observations.

### 11.4 Fixed MLMC

Let `D_L` be full fidelity and adjacent levels share declared semantic randomness:

```text
E[D_L] = E[D_0] + sum_{ell=1..L} E[D_ell - D_(ell-1)].
hat_D = mean(D_0) + sum_ell mean(D_ell - D_(ell-1)).
```

**Steps:** choose fidelity ladder and counts from a disjoint pilot; allocate independent groups by level; execute paired adjacent levels; report residual full-fidelity bias as zero only when level `L` is exact.

**Tiny test:** deterministic three-level telescope with signed corrections; deliberately omit one correction and verify failure; reject outcome-dependent extra-level allocation.

### 11.5 Randomized QMC

**Steps:** generate independently scrambled low-discrepancy points; map each uniform coordinate through a declared inverse-CDF or categorical partition; use one scramble as one dependent replicate; retain separate scrambles for uncertainty.

**Tiny test:** across scrambles, categorical frequencies converge to declared probabilities; one-scramble IID interval attempt fails a test; rare discontinuity is compared to IID rather than assumed better.

### 11.6 Scenario coreset

**Steps:** select a weighted subset only from the current search population; store original scenario IDs and nonnegative weights summing to one; use the weighted objective for search; never for confirmation.

**Tiny test:** weighted replay equals selected empirical objective; held-out tail/rare-scenario error is reported; unweighted selected subset intentionally fails a fixture.

### 11.7 Primal-dual pruning

Prune candidate `b` only when a valid simultaneous one-sided confidence statement proves:

```text
U_b < L_a
```

where bounds use the declared uncertainty unit and multiplicity correction. A sampled mean, approximate critic, or unrelated optimistic difference is not a bound.

**Tiny test:** a noisy two-action case where sample means favor pruning but intervals overlap; pruning must not occur. A certified case must prune only after the simultaneous inequality holds.

### 11.8 Controlled SMC

For incremental weights `G_t`, an unnormalized Feynman-Kac functional is:

```text
gamma_T(f) = E_q[f(X_0:T) product_t G_t].
```

With exact ratios and conditionally unbiased resampling, use an unnormalized population estimator `gamma_hat_T(f)`. Do not divide by random `gamma_hat_T(1)` and call the ratio unbiased.

**Steps:** freeze twist/proposal off evaluation data; propagate; multiply exact incremental ratios; resample with declared unbiased scheme; retain genealogy; treat independent populations, not descendants, as uncertainty units.

**Tiny test:** exact two-stage finite law checks unnormalized expectation across populations; normalization-bias fixture fails; resampling offspring frequencies match declared scheme.

### 11.9 Persistent event forest

**Steps:** after Hydra2 emits an action, speculate only from current actor-visible history; expand predicted packet children; on the next actual packet, verify/rebuild transition, rekey epoch, promote matching child, transport only target-identical artifacts, and delete/squash all incompatible statistics.

**Tiny test:** each packet-child commit matches a from-scratch posterior rebuild; sibling visit/value statistics cannot be queried after commit; hidden-tile canary changes leave root-facing output invariant; surprise/miss path recovers from fresh rebuild.

### 11.10 Constrained value-of-computation routing

This is the final optional module and remains heuristic.

**Steps:** fit routing score on pilot data; reserve a natural/coverage floor for every live child; cap any child; allocate remaining fixed budget using frozen score; log all overhead and missed branches. No outcome-derived allocation in confirmation.

**Tiny test:** every legal child receives its floor; no child exceeds cap; total charged work equals budget; miscalibration fixture cannot starve a rare branch.

### 11.11 Persistence factorial: B/F/R/P/C

| Arm | Definition |
| --- | --- |
| B | Frozen policy, no search. |
| F | Fresh search at each own decision; discard state; no opponent-time compute. |
| R | Retain compatible state but pause all search during opponent turns. |
| P | Retain state and ponder only after Hydra2 emits an action until its next actor-visible packet. |
| C | Laboratory-only fresh-search control. At an own decision it starts from the new actor observation with no retained state and receives an extended, predeclared allowance equal to the arm-invariant own deadline plus assigned intervening wait-window allowance. It sees no intervening packet before the decision. |

`B/F/R/P` share the deployment deadline selected from the target protocol and measured fallback margin. Tenhou standard permits `5+10`; Hydra2's formal deadline must be `<= 5000 ms` minus a recorded margin when testing standard tables. Fast tables use their own `<= 3000 ms` margin. `C` is not Tenhou-deployable and must be labeled a mechanism control.

`P-F` measures deployment benefit; `R-F` reuse; `P-R` pondering conditional on reuse; `P-C` asks whether a fresh search with comparable *scheduled maximum opportunity* explains P. P and C cannot be perfectly resource-identical because P conditions work on intervening visible packets; report actual calls, transitions, and joules rather than claiming exact equivalence.

## 12. Candidate 5: Public-Belief Local Resolving

**Purpose:** test shallow strategic improvement around high-leverage Tenhou decisions without importing a two-player zero-sum guarantee.

**Inputs/state:** qualified belief, exact simulator, blueprint continuation, action abstraction, actor-information-node strategy tables, vector values.

```text
subgame <- build_from_public_history(epoch, declared_horizon, abstraction)
initialize every actor information-node policy from blueprint
for t in 1..declared_iterations:
  traverse exact-rule sampled public histories
  compute vector continuation returns
  update each actor only at that actor's information nodes
average declared strategy sequence
execute root marginal for seat i only
confirm frozen root candidates naturally
```

Any regret/minimization update is an empirical optimizer, not an equilibrium certificate in this game. The local subgame, horizon, abstraction, leaf model, update rule, and averaging rule must all be frozen.

**Tests:** same-information policy equality; raw settlement conservation; utility-schema identity; tiny general-sum game against exhaustive reference; cycle detection; leaf reproducibility; action-abstraction round trip.

**Arms:** policy; ISMCTS; PBRF; local resolving with/without PBRF warm start; declared update variants.

**Promotion:** held-out natural confirmation gain under the declared resource view, without equilibrium or exploitability claim.

## 13. Candidate 6: Exact-Rules Belief-MuZero With Gumbel Search

**Purpose:** test low-simulation root policy improvement while preserving exact Tenhou rules.

**Inputs/state:** qualified belief; exact simulator; model policy/event/value heads; cached actor-visible representation; Gumbel root stream.

```text
belief <- Belief.begin(root_observation, model, rng)
legal <- Simulator.legal_actions(root_observation)
gumbels <- deterministic_root_gumbels(seed, legal)
survivors <- legal
for each declared sequential-halving round:
  allocate declared visits to survivors
  simulate exact world transitions and actor-visible packets
  evaluate leaves with Model.evaluate(actor_view, legal_mask, utility_id)
  back up U_T vectors
  remove candidates only by declared Gumbel score rule
return declared root selection or confirmation candidate set
```

The model predicts priors, beliefs, opponent behavior, and leaf values. It never replaces exact simulator transitions.

**Tests:** exact rule parity; cached/full history equality; hidden-tile invariance; deterministic Gumbel replay; vector backup; call/transition accounting; forbidden learned-rules negative control.

**Arms:** policy; PUCT-style natural-particle search; Gumbel search; fresh/persistent variants only after PBRF persistence gates.

**Promotion:** natural-confirmation root metric at matched model calls and exact-rule parity.

## 14. Candidate 7: Search Distillation and Population Training

**Purpose:** amortize a teacher that passed all five §18 gates into a faster frozen model.

**Inputs/state:** five-gate-promoted search teacher; qualified self-play or logged trajectory source; actor-visible replay; checkpoint population; exact rules simulator.

```text
for declared training source only:
  generate or replay actor-visible decision records
  store search policy, vector return, event/belief labels, provenance, budget
train student on frozen train split
retain behavior-cloning anchors and legal masking
freeze checkpoint and calibration
compare teacher, student, teacher+same_search, student+same_search
```

The stored full world may produce supervised labels inside a privileged training namespace. It may never enter student inference features.

**Tests:** split/wall overlap; actor-view audit; hidden permutation; replay/checkpoint determinism; population identity; fresh-process inference; no evaluation seed reuse.

**Arms:** pre-distillation policy; student; teacher search; student with same search; population variants.

**Promotion:** duplicate-block gain or noninferiority under predeclared block metric plus calibration/legality gates. Candidate 7 follows a teacher already passing contract, exact, search, match, and analysis gates; it is not the earlier oracle/belief distillation milestone in the project plan.

## 15. Candidate 8: Observation-Based Opponent-Type Robustness

**Purpose:** test a generic, legally observable opponent model without LuckyJ internals.

**Status:** research-only until sufficient authorized Tenhou or internal logs exist. It is not a LuckyJ reconstruction or a promise of targeted exploitation.

For opponent seat `j`, define a behavioral policy `q_j(a | I_j, theta)` keyed only by `j`'s information. Root does not observe `I_j`. Maintain the joint posterior over opponent type and hidden world because an observed action correlates them. For pre-packet joint law `p_h(theta,x)` and observed actor-visible packet `e`:

```text
p_next(theta, x') proportional
  integral p_h(theta, x) K_h(dx', e | x, q_j(. | I_j(x), theta)).

p_next(theta) = integral p_next(theta, x') dx'
b_next(x' | theta) = p_next(theta, x') / p_next(theta).
```

The kernel includes opponent-policy likelihood, exact physical transition, and call/pass resolution. Updating only a type marginal against a type-independent current belief is prohibited: it loses the induced type/world correlation and can double-condition on the observed action.

A robust response chooses root actions against a declared uncertainty set over coherent information-set policies, not independent worst distributions at each tree node:

```text
Q_set = {q_j(. | I_j):
  q_j respects legal masks and same-information equality,
  divergence(q_j || nominal_j) <= rho,
  q_j = (1-epsilon) q_nominal + epsilon r_j,
  r_j belongs to declared support class}
```

Any near-rationality condition, divergence direction, `rho`, `epsilon`, support class, and nonempty-set proof must be frozen. The exact simulator induces the joint trajectory law from these behavioral policies.

**Tests:** same-information equality; nominal inclusion; feasible-set nonemptiness; coherent trajectory generation; exact finite joint type/world posterior update; sequential actions preserve induced correlation; hidden-hand marginalization; held-out calibration; no test leakage; synthetic recovery.

**Promotion:** confidential held-out improvement over generic model under the declared uncertainty set. Inadequate data, bad calibration, or training-only gain keeps it disabled.

## 16. Offline Evaluation and Compute Protocol

### 16.1 Internal controlled evidence

The main causal comparison runs in an exact Tenhou-rule simulator. Use committed wall/seat/latency schedules; report block-level results. Duplicate walls reduce randomness but calls alter later draw ownership, so the wall block, not individual games, is the independent unit.

For symmetric 2-v-2 comparison, use all six assignments of agent A to two of four seats per wall seed. For a 1-v-3 diagnostic, rotate focal A through every seat. Expected final placement is primary. Also report raw score, first/fourth/deal-in/riichi/call rates, legal failures, timeouts, and energy/latency distributions.

A blinded pilot estimates block variance and execution loss. Before arm labels are unblinded, freeze practical margin `delta`, multiplicity rule, maximum blocks, and either:

```text
fixed N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)^2)
```

with pilot standard deviation `s`, or a predeclared time-uniform confidence sequence. Bootstrap and sign-flip analyses resample whole wall blocks only.

### 16.2 Compute modes, runtime adapter, and hardware ledger

`gameplay_5s`: 5,000 ms own-turn hard deadline. CandidateSpec freezes synchronization, planner overhead, fallback margin/policy, and timeout accounting. `ponder`: computation only between actor-visible packets; commit requires a transition-correct child and valid belief epoch/provenance. `analysis`: separately frozen larger finite deadline/resource cap; same actor observation, rules, utility, models, legal set, and estimator semantics.

RTX 5070 is the default optimization and formal single-device target. Standalone Lightning Fabric 2.6.5 may be used only as a thin device/precision/backward adapter around project-owned loops; it does not own simulator semantics, objectives, optimizer policy, accumulation, schedules, evaluation, or checkpoints. Every arm retains a plain-PyTorch single-device fallback with identical tensors, state, seeds, and gates. Runtime identity records `fabric_2.6.5` or `plain_pytorch`, and adapter equivalence is measured rather than assumed.

A100 access is a transactional reserve, not a second rank. Before each allocation, atomically append a ledger entry with request ID, local RTX 5070 bottleneck evidence, hypothesis, frozen arm/corpus, requested hours, transfer/data-loader cost, compile-amortization plan, and approval. On completion append actual start/stop, charged GPU-hours, retries/failures, artifact hashes, and disposition, then atomically decrement the 2,000 GPU-hour balance; failed and aborted runs are charged. Never infer distributed execution or NVLink from a local RTX 5070 plus a separately reserved A100. Device-specific kernels, results, compile caches, and artifacts remain separate absent explicit compatibility proof.

### 16.3 PyTorch 2.13 Performance Qualification

PyTorch 2.13.x is the semantic authority. These are qualification experiments, not free-speed claims. Published kernel results motivate probes only; they are not Hydra2, RTX 5070, or A100 promises.

#### 16.3.1 Eager simulator, tensor boundary, and Fabric

The exact simulator, actor visibility, legal-action computation, decoding, control flow, objectives, and loops stay eager. Only actor-visible pure tensor encodings enter a compiled model. Fabric is optional and standalone; plain PyTorch is mandatory fallback. For compiled AMP, the `backward_pass_autocast="off"` patch spans both initial compilation and Fabric's setup-time unwrap/reapplication because Fabric does not capture that global compiler configuration.

```python
def build_runtime(adapter, model, optimizer, precision, device, compile_mode,
                  compiled_amp=False):
    def compile_model(model):
        if compile_mode is None:
            return model
        return torch.compile(model, backend="inductor", mode=compile_mode,
                             fullgraph=True, dynamic=None)

    def setup_adapter(model, optimizer):
        if adapter == "fabric_2.6.5":
            fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
            model, optimizer = fabric.setup(model, optimizer)
            return model, optimizer, fabric.backward, fabric.device
        if adapter == "plain_pytorch":
            model = model.to(device)
            return model, optimizer, lambda loss: loss.backward(), device
        raise ValueError(adapter)

    if compiled_amp and compile_mode is not None:
        # Keep the patch active if Fabric unwraps and reapplies compilation.
        with torch._functorch.config.patch(backward_pass_autocast="off"):
            model = compile_model(model)
            return setup_adapter(model, optimizer)

    model = compile_model(model)
    return setup_adapter(model, optimizer)

for request in project_owned_training_loop():
    exact = simulator.transition_and_legal_actions(request.full_world)  # eager oracle
    actor_view = exact.actor_observation(request.actor)
    assert exact.legal_actions == legal_actions_from_rules(actor_view)
    batch = encode_actor_view_to_tensors(actor_view, exact.legal_mask)
    batch = tree_to(batch, runtime_device, non_blocking=qualified_nonblocking)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.features, batch.history_mask, batch.legal_mask)
    loss = project_training_objective(output, batch.targets)
    backward(loss)
    optimizer.step()
```

`fullgraph=True` is a gate for the intentionally compiled tensor region so breaks fail visibly. A separately named regional-compilation arm may compile repeated tensor-only blocks if whole-model cold start is excessive; it never absorbs simulator work or changes information semantics.

#### 16.3.2 Compile and attention arms

Run this ladder in order on identical checkpoints, corpus, buckets, and seeds. `max-autotune` includes CUDA Graphs in 2.13; `max-autotune-no-cudagraphs` isolates autotuning from graph capture.

```python
COMPILE_ARMS = [
    {"name": "eager", "compile": False},
    {"name": "default", "compile": True, "mode": "default", "dynamic": None},
    {"name": "max-autotune-no-cudagraphs", "compile": True,
     "mode": "max-autotune-no-cudagraphs", "dynamic": False},
    {"name": "max-autotune", "compile": True,
     "mode": "max-autotune", "dynamic": False},
]
for arm in COMPILE_ARMS:
    candidate = clone_from_identical_checkpoint(reference_model)
    if arm["compile"]:
        candidate = torch.compile(candidate, backend="inductor", mode=arm["mode"],
                                  fullgraph=True, dynamic=arm["dynamic"])
    qualify(candidate, fixed_batches, cold_runs=True, warm_runs=True)
```

Use `dynamic=None` first; bounded buckets are conditional; `dynamic=True` is diagnostic. Record `TORCH_LOGS="graph_breaks,recompiles,dynamic,perf_hints"`, `torch._dynamo.explain`, and access-controlled `TORCH_TRACE`/`tlparse` when needed because traces can contain source. Profiler-disabled synchronized timings are authoritative. CUDA Graphs require stable CUDA-only buckets, safe addresses/lifetimes/mutation, bounded recordings, and acceptable memory.

Standard dense/causal attention uses SDPA; evaluation dropout is exactly zero and boolean-mask `True` means participate.

```python
def standard_attention(q, k, v, *, attn_mask, training, dropout_p):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )
```

FlexAttention is a prototype, conditional only for custom score semantics or useful block sparsity. Dense patterns stay on SDPA. Cache `BlockMask` by stable semantic/shape bucket outside each step.

```python
BLOCK_MASKS = {}
def cached_block_mask(bucket, device):
    key = (bucket.semantic_id, bucket.q_len, bucket.kv_len, str(device))
    if key not in BLOCK_MASKS:
        def mask_mod(b, h, q_idx, kv_idx):
            return bucket.pure_tensor_mask_rule(q_idx, kv_idx)
        BLOCK_MASKS[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=bucket.q_len,
            KV_LEN=bucket.kv_len, device=device)
    return BLOCK_MASKS[key]

@torch.compile(fullgraph=True, dynamic=False)
def flex_region(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)
```

#### 16.3.3 Precision, input, and optimizer arms

FP32 eager is the semantic oracle. FP16 and BF16 are separate device-specific AMP candidates. Autocast wraps forward/loss, not backward; FP16 scales and unscales before clipping/finite checks. BF16 normally omits scaling. For a compiled AMP arm whose backward intentionally runs outside autocast, scope `torch._functorch.config.patch(backward_pass_autocast="off")` around `torch.compile`, as shown in `build_runtime`. Record that value in the CandidateSpec and compile identity. Changing it requires recompilation; cached artifacts compiled under another value are invalid. Eager and non-AMP arms retain their existing behavior and record `null`.

```python
def amp_step(model, optimizer, batch, dtype, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=dtype):
        loss = project_training_objective(model(**batch.model_inputs), batch.targets)
    if dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert_all_gradients_finite(model)
        clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        assert_all_gradients_finite(model)
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    return loss

fp16_scaler = torch.amp.GradScaler("cuda")
```

TF32 is a fine-grained FP32 candidate, never assumed on. Do not mix legacy `allow_tf32` flags with the 2.13 precision controls.

```python
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "tf32"  # candidate
torch.backends.cudnn.fp32_precision = "ieee"
```

Pinned memory/nonblocking transfer is conditional on GPU-idle/data-wait evidence, is not always faster, and does not by itself prove overlap. Workers return CPU tensors; custom batches implement `pin_memory()`.

```python
loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                    pin_memory=pin_memory_arm,
                    persistent_workers=workers > 0,
                    prefetch_factor=2 if workers > 0 else None,
                    collate_fn=collate)
for cpu_batch in loader:
    cuda_batch = tree_map(
        lambda x: x.to("cuda", non_blocking=nonblocking_arm), cpu_batch)
    train_step(cuda_batch)
```

Compare AdamW default selection, explicit for-loop, and fused. On eligible CUDA tensors unset flags attempt foreach; foreach can add about one parameter-sized intermediate peak. Fused is newer, not automatic, and cannot combine with `foreach=True`.

```python
OPTIMIZER_ARMS = {
    "default": lambda p: torch.optim.AdamW(p, lr=lr),
    "for_loop": lambda p: torch.optim.AdamW(p, lr=lr, foreach=False),
    "fused": lambda p: torch.optim.AdamW(p, lr=lr, fused=True),
}
```

#### 16.3.4 Common acceptance gates

Every arm uses identical representative samples, buckets, legal masks, targets, initialization/checkpoint, logical seeds, update count, and protocol. Dtype/task tolerances come from a blinded pilot and freeze before labels are inspected. Reject unless all applicable gates pass:

1. **Semantic oracle:** eager/FP32 exact transitions, actor observations, visibility, settlement, and information-set semantics remain authoritative.
2. **Legal/action parity:** exact legal-mask equality and legal chosen-action parity; any illegal action or hidden-information sensitivity is a hard failure.
3. **Training/checkpoint parity:** outputs/loss meet frozen tolerances; finite gradients cover every trainable path; short-run parameter updates, optimizer/scheduler state, checkpoint save/resume, and continued trajectory meet declared parity.
4. **Numerics:** no unexplained NaN/Inf in inputs, outputs, loss, gradients, parameters, optimizer state, or metrics; FP16 scaler/skipped updates are logged.
5. **Compiler health:** zero unexpected graph breaks in the intended region, bounded recompiles and shape recordings after warmup, and exact compile/cache identity.
6. **Cold/warm cost:** separately report process startup, first-call compile/autotune/capture, cache state, warmup, steady state, and measured amortization count; do not assume amortization.
7. **Timing/throughput:** synchronized median and p95 end-to-end step latency plus examples, decisions, or transitions per second. Include applicable encoding boundary, transfer, mask construction, optimizer, synchronization, adapter overhead, and fallback. Kernel-only timing is diagnostic.
8. **Memory:** peak CUDA allocated and reserved memory; host memory when workers/pinning matter. Memory-only changes additionally require demonstrated OOM risk or meaningful frozen peak-memory reduction.
9. **Determinism:** repeated declared configuration and quantified residual variance; nondeterministic backends are labeled and may not invalidate replay, checkpoint, legality, or comparison.
10. **End-to-end strength:** same held-out policy/value metrics and, when deployed decisions may change, frozen duplicate-wall/block evaluation. Faster kernels fail if quality or expected-placement evidence crosses the frozen regression bound.
11. **Hardware separation:** independent RTX 5070 and A100 tables, tolerances, artifacts, caches, and conclusions. Missing data is unqualified; no cross-architecture extrapolation.
12. **Operational burden:** exact dependency/wheel/container digest, failure rate, cache portability, cold-start risk, and eager fallback. Accept only a statistically credible end-to-end bottleneck improvement without unacceptable semantic, numerical, strength, memory, reproducibility, or maintenance regression.

#### 16.3.5 Explicit classifications and corrections

| Candidate or claim | Hydra2 classification | Correction or required gate |
| --- | --- | --- |
| `torch.compile(mode="default")` on pure tensor model | Default-safe qualification arm, not promised win | Simulator stays eager; measure cold/warm cost, breaks, recompiles, memory, and strength. |
| `max-autotune-no-cudagraphs` | Conditional after `default` | Isolates autotuning from CUDA Graphs. |
| `max-autotune` | Conditional after no-graphs arm | Includes CUDA Graphs in PyTorch 2.13; qualify capture, recordings, address/mutation safety, and memory. |
| CuTeDSL / `CUTEDSL` | Inapplicable on both targets | Internal Inductor autotune candidate, not public replacement backend. Named templates are SM100-SM109; RTX 5070 is SM120, A100 SM80. |
| AOTInductor | Deferred/inapplicable to current training | Exported inference/non-Python deployment, not the normal training path; training compile uses Dynamo/AOTAutograd/Inductor. |
| SDPA | Default-safe standard attention | Qualify dispatch, mask meaning, dtype, numerical/determinism behavior, and evaluation dropout zero. |
| FlexAttention / cached `BlockMask` | Prototype conditional experiment | Only custom/block-sparse semantics; cache stable masks and compare dense cases with SDPA. |
| FlashAttention-4 Flex backend | Experimental/external; A100 inapplicable; RTX 5070 unproven | Official support excludes Ampere. Consumer SM120 coverage/backward/value require quarantined local proof; no borrowed numeric promise. |
| `nn.LinearCrossEntropyLoss` | Unsuitable for current small action head | Keep `Linear`, legal mask, and `cross_entropy`; revisit only for a genuinely large projection with migration, loss/gradient/update, compile, and memory gates. |
| Nested/jagged tensors | Conditional | Only padding-dominated variable histories after op/compile audit; compare padded/bucketed baseline and respect one ragged dimension and offset-identity constraints. |
| Activation checkpointing | Conditional memory experiment | Only after activation-driven memory evidence; use non-reentrant form and require recomputation/RNG parity. Selective checkpointing is deferred further. |
| 2:4 semi-structured sparsity | Deferred conditional inference for large eligible linears | Lossy pruning/quality recovery and supported ops required. FP16/BF16 documented path needs 2-D contiguous CUDA weights and dimensions divisible by 64; small heads are likely ineligible/unamortized. No training-speed claim. |
| torchao FP8 training | Deferred conditional research for large GEMMs only | Separate dependency; exclude small/output heads. Local kernel coverage, convergence, strength, and end-to-end gain required independently per device. |
| Torch-TensorRT NVFP4 | Deferred RTX inference-only probe; A100 inapplicable | Not training. A100 lacks Blackwell FP4. Exact RTX 5070 Torch-TensorRT support is unproven; require local dry run, partition coverage, calibration quality, build/cold/warm timing, and fallback. |
| FSDP2, DTensor, torchcomms, DCP | Deferred until real concurrent `world_size > 1` | One RTX 5070 plus time-separated A100 reserve is not distributed. Keep ordinary single-device `torch.save` checkpoints. |
| Pinned memory/nonblocking | Conditional | Benchmark only with data-wait evidence; not always faster and no overlap or NVLink claim follows. RTX 5070 officially has no NVLink; A100 reservation proves no topology. |
| NGC PyTorch container | Digest-qualified environment experiment only | Pin image digest, exact PyTorch commit/version and libraries, host-driver compatibility, and compare semantics/performance against qualified 2.13.x wheels. It is not semantic authority or a speed feature. |

No candidate changes simulator transitions, rules, actor visibility, legal sets, objective, or estimator semantics. A performance arm that requires such a change is a different algorithm experiment and cannot pass this section.
### 16.4 Confidential data and leakage checks

Use held-out actor-visible logs. Measure event/action log loss, Brier score, calibration, tenpai/wait/tile-marginal predictions where later revealed, and score/placement distribution quality. Cluster uncertainty by game/player, never individual decision.

Stop a run for any unrevealed-tile permutation sensitivity, hidden-canary sensitivity, actor-view serialization leak, duplicate train/test game, wall/seed overlap, checkpoint-selection leakage, illegal action, nondeterministic replay, unresolved rules mismatch, or missing telemetry beyond the predeclared tolerance.

## 17. Toy-Only Evidence and Constraints

These local synthetic checks motivate safeguards. They do not predict Mahjong strength. Paths are local to the earlier theory workspace and are not Hydra2 dependencies.

| Warning/module | Observed synthetic result | Required consequence |
| --- | --- | --- |
| Nonanticipativity | Clairvoyant value `4.7`; legal shared optimum `2.2`; optimism `2.5`. | Information-set keys; never world-specific root actions. |
| Successor transition | Parent-only L1 `0.738691`; fanout `0`. | Push forward and condition, not parent-only reweight. |
| Targeted rare event | IID accepted `1593/300000` (`0.5310%`); targeted MIS ESS `12977.7/20000` (`64.89%`). | ESS is diagnostic, not proof. |
| Rao-Blackwellization | Equal 72 policy calls: sampled RMSE `0.1715`; RB `0.04009`. | Charge enumeration and use only tractable variables. |
| Multifidelity | Wrong-action: expensive `0.52533`; MLMC `0.33867`; MLMC RMSE `0.07793`. | Full top level; fixed pilot allocation; signed telescope. |
| Random depth | Good RR wrong `0.522`; bad `0.692`; bad max `29.475`. | Inspect tails; no aggressive default schedule. |
| RQMC | Smooth LHS gain `4.024`; rare discontinuous `1.041`. | Independent scrambles; no assumed rare-event gain. |
| Causal coupling | Independent variance `1.090416`; common uniform `2.241801`; causal OT `0.000821385`. | Measure covariance; retain independent control. |
| Decision IS | IID RMSE `0.004123`; defensive `0.0007804`; unweighted bias `0.2614`. | Full mixture denominator and one correction. |
| Twisted SMC | Bootstrap `0.2589`; aggressive `0.5939`; aggressive coverage `0.0002`. | Natural floor and independent populations. |
| Scenario coreset | Decision error `0`; unseen MAE `1.9271`; tail RMSE `3.1253`. | Search-only; tail/held-out validation. |
| Primal-dual | Simultaneous-bound false-prune `0`. | Prune only simultaneous `U_b < L_a`. |
| Information relaxation | Exact `3.1875`; perfect-information `4.0`. | Exact centering required for a guarantee. |
| Consensus | Limited single-PH failure `3/240`; portfolio-3 `0/240` with more work. | Candidate generation is not confirmation. |
| VOC | No-floor starvation `0.999`; floor `0`; floor regret `0.040619`; uniform `0.038544`. | Floors/caps and charged overhead. |
| Hybrid | Paired MF gap RMSE `0.00784`; blind `0.01167`; pilot-gated `0.01224`. | No blind hybrid default. |

## 18. Promotion Gates and Stop Conditions

A candidate promotion record must name the exact fixture/case manifest, resource view, uncertainty unit, result-table hash, and pass inequality. The minimum suite is:

1. **Contract gate:** rule/action/event/visibility schemas and actor-isolation tests pass.
2. **Exact gate:** versioned finite tiny-state corpus passes its declared equality/tolerance rules.
3. **Search gate:** candidate beats or matches its declared comparator on fresh natural confirmation under fixed resources.
4. **Match gate:** qualified candidate survives duplicate-block evaluation with whole-block uncertainty.
5. **Analysis gate:** larger-budget mode changes only charged computation and preserves gameplay semantics/information.
For Candidates 0-6, "promoted teacher" means all five gates passed. Analysis qualification therefore precedes Candidate 7 teacher selection; Candidate 7's own later promotion is a separate student result.

Hard failures: illegal action, wrong rules transition, packet mass error, double correction, hidden-information leak, stale posterior admission, nondeterministic replay, unaccounted deadline fallback, or use of confirmation outcomes to change candidates/statistics without valid sequential design.

## 19. Primary References

Rules and Mahjong context:

- Tenhou manual and AI policy: <https://tenhou.net/man/>
- Mizukami and Tsuruoka, *Building a Computer Mahjong Player Based on Monte Carlo Simulation and Opponent Models*, CIG 2015, DOI: <https://doi.org/10.1109/CIG.2015.7317959>
- Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, arXiv:2003.13590: <https://arxiv.org/abs/2003.13590>
- Kurita and Hoki, *Mahjong MDP Abstractions*, arXiv:1904.07491: <https://arxiv.org/abs/1904.07491>
- Tencent public LuckyJ description: <https://www.tencent.com/en-us/articles/2201746.html>

Imperfect-information planning:

- Silver and Veness, *Monte-Carlo Planning in Large POMDPs*, NeurIPS 2010: <https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html>
- Cowling, Powley, and Whitehouse, *Information Set Monte Carlo Tree Search*, IEEE TCIAIG 2012, DOI: <https://doi.org/10.1109/TCIAIG.2012.2200894>
- Somani et al., *DESPOT*, JAIR 2017: <https://arxiv.org/abs/1609.03250>
- Sunberg and Kochenderfer, *Online Algorithms for POMDPs with Continuous State, Action, and Observation Spaces*, ICAPS 2018: <https://arxiv.org/abs/1709.06196>
- Brown et al., *ReBeL*, NeurIPS 2020: <https://arxiv.org/abs/2007.13544>
- Zinkevich et al., *Regret Minimization in Games with Incomplete Information*, NeurIPS 2007: <https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information>
- Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022: <https://arxiv.org/abs/2112.03178>
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*, Nature 2020: <https://arxiv.org/abs/1911.08265>

Particles, variance, and inference:

- Pitt and Shephard, *Filtering via Simulation: Auxiliary Particle Filters*, JASA 1999, DOI: <https://doi.org/10.1080/01621459.1999.10474153>
- Veness et al., *Variance Reduction in Monte Carlo Tree Search*, NeurIPS 2011: <https://papers.nips.cc/paper/4288-variance-reduction-in-monte-carlo-tree-search>
- Giles, *Multilevel Monte Carlo Path Simulation*, Operations Research 2008, DOI: <https://doi.org/10.1287/opre.1070.0496>
- Owen, *Monte Carlo Variance of Scrambled Net Quadrature*, SIAM J. Numer. Anal. 1997, DOI: <https://doi.org/10.1137/S0036142994278852>
- Howard et al., *Time-Uniform, Nonparametric, Nonasymptotic Confidence Sequences*, Annals of Statistics 2021: <https://arxiv.org/abs/1810.08240>

PyTorch 2.13 performance and runtime:

- `torch.compile` API and mode definitions: <https://docs.pytorch.org/docs/2.13/generated/torch.compile.html>
- Compiler troubleshooting, graph breaks, recompiles, traces, and profiling caveats: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_troubleshooting.html>
- Where to apply compilation: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html>
- `fullgraph=True` programming model: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html>
- Dynamic shapes: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html>
- CUDA Graph Trees: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>
- Compiler FAQ, including AOTAutograd distinction: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_faq.html>
- Compiled backward and `backward_pass_autocast`: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_backward.html>
- AOTInductor inference deployment: <https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_aot_inductor.html>
- PyTorch 2.13 Inductor configuration source, including CUTEDSL restrictions: <https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/config.py>
- PyTorch 2.13 compile mode source: <https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/__init__.py>
- SDPA API: <https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html>
- FlexAttention and `BlockMask`: <https://docs.pytorch.org/docs/2.13/nn.attention.flex_attention.html>
- FlashAttention-4 Flex backend status and limitations: <https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/>
- AMP API and examples: <https://docs.pytorch.org/docs/2.13/amp.html> and <https://docs.pytorch.org/docs/2.13/notes/amp_examples.html>
- TF32 controls: <https://docs.pytorch.org/docs/2.13/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices>
- AdamW default, foreach, and fused behavior: <https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html>
- DataLoader memory pinning: <https://docs.pytorch.org/docs/2.13/data.html#memory-pinning>
- Pinned/nonblocking transfer tutorial: <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- Nested/jagged tensors: <https://docs.pytorch.org/docs/2.13/nested.html>
- Activation checkpointing: <https://docs.pytorch.org/docs/2.13/checkpoint.html>
- Semi-structured sparsity: <https://docs.pytorch.org/docs/2.13/sparse.html#sparse-semi-structured-tensors>
- `LinearCrossEntropyLoss` PyTorch 2.13 source: <https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/modules/loss.py#L1410-L1584>
- Standalone Lightning Fabric 2.6.5 setup/compile behavior: <https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.5/src/lightning/fabric/fabric.py>
- torchao float8 training workflow and conversion API: <https://docs.pytorch.org/ao/stable/workflows/training.html> and <https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.float8.convert_to_float8_training.html>
- Torch-TensorRT quantization/NVFP4 workflow: <https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html>
- FSDP2, DTensor, distributed communication, and DCP: <https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html>, <https://docs.pytorch.org/docs/2.13/distributed.tensor.html>, <https://docs.pytorch.org/docs/2.13/distributed.html>, and <https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html>
- NVIDIA CUDA GPU compute capabilities: <https://developer.nvidia.com/cuda-gpus>
- NVIDIA RTX 5070 specifications, including no NVLink: <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/>
- NVIDIA TensorRT FP4 on RTX 50-series: <https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/>
- NVIDIA NGC PyTorch catalog, release notes, and support matrix: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>, <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html>, and <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>

The document is self-contained for implementation choices. These sources provide proofs, historical context, and alternate formulations; they do not override the declared Tenhou rules manifest or Hydra2 provenance contract.

## 20. Status Addendum (2026-09-04) — Implementation Alignment

Normative §§1–19 are unchanged. This section records where the implementation
has concretized what the blueprint leaves schematic, and where the toolchain
has drifted from the blueprint's pins. File:line citations are worktree state
on 2026-09-04 (including the ~51-file uncommitted worktree); the last clean
commit is Loop-4 (`602746a`). Nothing here promotes a candidate or relaxes §18.

### 20.1 Action-space authority and mask widths

- Legal-mask width is `BASELINE_ACTION_COUNT = 6792`
  (`src/hydra2/models/schema.py:55`), enforced at encode time
  (`src/hydra2/models/encoder.py:214-216`) and model construction
  (`src/hydra2/models/model.py:192-193`). The canonical authority is
  `configs/contracts/action_table_v1.json` `payload.actions` (6792 entries);
  engines must not fill gaps from defaults (§1.1 still applies).
- `0xFFFF` is NOT a mask width. It is the PBRF fallback action-id hash mask
  (`src/hydra2/search/pbrf.py:200`). Gumbel planners use a `0x7FFFFFFF`
  action-id domain via salted SHA-256 (`src/hydra2/search/gumbel.py:853-857`,
  `src/hydra2/search/gumbel.py:1365-1369`); the bare-`hash()` fallback is gone
  because `hash()` is per-process seeded and nondeterministic across runs.

### 20.2 Candidate 7 teacher: real path, 48-dim features, StudentModel (§14)

- The teacher uses the REAL `model_input_v1` encoder path
  (`encode_observations`) for priors, never a hash-derived sketch
  (`src/hydra2/distillation/teacher.py:614-643`,
  `src/hydra2/distillation/teacher.py:1012-1047`).
- Student features are a fixed 48-dim actor-visible vector: 34 concealed-tile
  counts ([-1,1]) + 4 scores (/40000) + 1 live-wall remaining (/136) + 4 actor
  one-hot + 4 turn-actor one-hot + 1 ippatsu-any flag
  (`src/hydra2/distillation/teacher.py:59-60`,
  `src/hydra2/distillation/teacher.py:1016-1041`). A dim guard fails closed
  (`src/hydra2/distillation/teacher.py:1042-1044`). There is no 16-dim student
  feature variant; the `16` in `Hydra2BaselineModel.placement_head`
  (`src/hydra2/models/model.py:260`) is the unrelated 4x4 placement matrix.
- `StudentModel` defaults to `d_model=32`: `Linear(48→32)` → `Linear(32→32)`,
  policy head `Linear(32→num_actions)`, value head `Linear(32→4)`
  (`src/hydra2/distillation/teacher.py:970-987`). `num_actions` resolves from
  the canonical action table via `_action_table()`; `_DEFAULT_NUM_ACTIONS=32`
  is import-time fallback only (`src/hydra2/distillation/teacher.py:62`).
  A missing/malformed table blocks WP-10 instead of shrinking the mask
  (`src/hydra2/distillation/teacher.py:124-148`).
- Synthetic gate digests are removed: every gate digest is content-addressed
  to a real CandidateSpec/analysis-gate artifact, and missing prerequisites
  raise `ContractError` ("WP-10 blocked"), never a synthesized digest
  (`src/hydra2/distillation/teacher.py:41-42`). The WP-12 analysis gate loads
  fail-closed through `analysis_gate_for` on
  `work_packages/WP-12/analysis_gates.json`, requiring eligible + compute-only
  + deterministic-replay pass
  (`src/hydra2/distillation/teacher.py:158-198`). Teacher→analysis identity
  maps `candidate3` to `candidate3_pbrf_core_v1` (`_ANALYSIS_ID_FOR_TEACHER`);
  unmapped ids are blocked, never defaulted.

### 20.3 Planner/qualification hash authorities (§5 reproducibility contract)

- `gumbel`, `pbrf`, and `local_resolving` no longer carry `PLACEHOLDER_*`
  constant hashes. File-backed config hashes load from disk with per-key
  `MISSING_HASH` fallback; the model digest binds through the candidate0
  authority (import, mirror on failure) and the utility-manifest digest
  derives from the live model, raising instead of faking:
  `src/hydra2/search/gumbel.py:1676-1718`,
  `src/hydra2/search/pbrf.py:998-1080`,
  `src/hydra2/search/local_resolving.py:983` (`_model_hash_from_identity`,
  `_derive_utility_manifest_hash`, `_canonical_hashes` in each module).
  RNG/stream/case digests come verbatim from the candidate0 canonical
  descriptors (`counter_based_v1`, `random_stream_v1` with
  `candidate0_tie` purpose, empty case manifest).
- Qualification mirrors the same rule: candidate0 canonical descriptors
  verbatim, never constant hashes
  (`src/hydra2/analysis/qualification.py:762`,
  `src/hydra2/analysis/qualification.py:800-824`), and every gate/record
  carries the live `gameplay_spec_hash`
  (`src/hydra2/analysis/qualification.py:669`,
  `src/hydra2/analysis/qualification.py:1013-1049`). Stale-spec guards compare
  factory CandidateSpec digests against the WP-12 gate
  (`src/hydra2/distillation/teacher.py:304-310`).

### 20.4 Candidate 0 binding (§7)

- `frozen_choice` runs under `@torch.inference_mode()`
  (`src/hydra2/search/candidate0.py:46`) with a device-bound
  `torch.Generator` for the temperature arms; `make_candidate0_spec`
  unconditionally narrows `rules_hash`/`action_table_hash`/`model_hash` so no
  `str|None` reaches `CandidateSpec`. Descriptor semantics (greedy /
  fixed-temperature / value tie-break, §7 arms) are unchanged.

### 20.5 Synthetic-path hardening (§18 hard failures)

- Teacher, loop, replay, and completion paths fail closed where the blueprint
  demands hard failures: "WP-10 blocked" `ContractError`s replace synthetic
  fallbacks (`src/hydra2/distillation/teacher.py:138-148,376,503,580-643,1028,
  1044,1068-1087,1387-1396`); completion declares "missing/ineligible gate
  raises ContractError — never synthetic fallback"
  (`src/hydra2/completion.py:108`) with WP-10 gated on WP-09C/D/E + WP-12 and
  WP-12 gated on WP-08A/B/C + WP-09A/B/C/D/E
  (`src/hydra2/completion.py:109-117`); replay rejects privileged fields in
  actor inputs (`src/hydra2/training/replay.py:167`). This is the
  WP-10+/WP-12 hardening the blueprint's §18 requires; it changes no estimator
  semantics (§§3, 10–11).

### 20.6 Belief kernel and eval statistics (§§10, 16.1)

- `NaturalPacketKernel` (`src/hydra2/belief/kernel.py:135`) keeps the §10.2
  exact per-parent partition assertion; the worktree adds cached
  packet-chain/observation-hash/successor-ref helpers (same digests, no
  semantic change).
- `eval/statistics.py` keeps the §16.1 fixed-N formula and whole-block
  resampling units; the worktree hoists a module-level `_STD = NormalDist()`
  (`src/hydra2/eval/statistics.py:54`) and vectorizes `bootstrap_blocks` via
  seeded numpy PCG64 (statistically equivalent; exact Monte Carlo draws differ
  from the per-element loop — do not compare old/new bootstrap draws
  bit-for-bit).

### 20.7 Toolchain drift (§§5, 16.2–16.3)

- Blueprint pins PyTorch `2.13.x` (including the §5 CandidateSpec example
  `"torch": "2.13.x"` and all §16.3 references). Worktree authority is
  `torch == 2.14.0` (`pyproject.toml:33`) with pixi as sole environment/lock
  authority and `uv.lock` banned (`pyproject.toml:16`). `config_check.py:5`
  still names 2.13.0 and is stale — pixi/pyproject wins on conflict.
- §16.3 remains valid as a qualification *protocol* (eager oracle, fullgraph
  gate, cold/warm accounting, hardware-separation gate §16.3.4-11), but every
  measured table attributed to 2.13.x must be re-run under 2.14.0 per device;
  no cross-version extrapolation.
- Hardware posture is unchanged: RTX 5070 (sm_120) default with a CUDA-wheel
  sm_120 kernel probe (`src/hydra2/_probe_support.py:64-77`), A100 (sm_80) as
  a charged 2,000 GPU-hour transactional reserve per §16.2. `Trainer`
  packages remain absent; Fabric 2.6.5 stays a thin standalone adapter with a
  mandatory plain-PyTorch fallback (§16.2–16.3.1 normative text stands).

### 20.8 Repeatability evidence (no strength claim)

- Tiny-shard overfit plus deterministic-repeat fixtures
  (`tests/unit/test_baseline_wp05c.py:122-135,240`) cover the §18
  contract/exact/search-gate replay leg of the loop; they establish
  repeatability only, exactly as §7/§18 require before any strength claim.
