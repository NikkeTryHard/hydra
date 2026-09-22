//! joint_world: frozen Candidate 8 epoch-increment core + numeric config gates.
//!
//! DAG: pyo3 ONLY (frozen `u64` epoch arithmetic + finite/comparison config
//! gates; no feed/shard/search-crate edge — the joint likelihood + normalize
//! lanes already ride `search::joint_posterior_weights` /
//! `joint_normalize_weights` / `joint_posterior_fused` (`search.rs:1418-1596`,
//! reused, never reimplemented here), the canon bytes live `feed::canon` via
//! that fused path, and the live dataclass orchestration stays Python in
//! `joint_types.py` / `joint_uncertainty.py` / `joint_planner.py`). FORBIDS:
//! canon/JCS re-printing, digest/hash math (`sha2`/`hashlib` — the owners
//! hash), live `JointPosterior` / `OpponentTypePolicy` / `FullWorld` IO
//! (info-key assembly, hidden-permutation validators, the belief sampler
//! `NaturalBelief.begin` + tiny-corpus build, and the hash-draw sampler stay
//! Python), second weight implementations (the posterior lanes above are
//! reused by reference), float `sum()` compensation (the posterior mass audit
//! `sum(p.weight)` stays Python per the wave-8 rule — this file holds only
//! `is_finite` + comparison gates, never a sum), `ContractError` shaping (the
//! bridge raises `ValueError` or returns plain `bool`; thin Python
//! translators map to `ContractError` with byte-identical oracle text), and
//! clocks/time (`time.monotonic_ns` telemetry stays Python).
//!
//! TABLE ported here — frozen values + pure compute, oracle
//! `python/hydra2/search/joint_type_world.py` (facade over the split modules,
//! no logic of its own) with the leaf oracles in `joint_types.py`,
//! `joint_uncertainty.py`, `joint_planner.py` (HEAD `src/hydra2/search/
//! joint_*.py`, equals worktree at port time except the `hydra2_replay_rs` ->
//! `hydra2._native` import heads):
//! - `JOINT_WORLD_FIRST_EPOCH` <- epoch origin `0` (fresh: no Rust owner —
//!   see NONE note; oracle cites `joint_types.py:484-485` `epoch < 0` rejects
//!   so `0` is the first valid epoch, `joint_uncertainty.py:74-75` the prior
//!   epoch gate, `joint_planner.py:164`
//!   `epoch_id = int(getattr(epoch, "epoch", 0))` the `0` default).
//! - `joint_world_next_epoch` <- the implied caller-increments-epoch rule
//!   (`joint_uncertainty.py:70` "Returns normalized JointPosterior at same
//!   epoch (caller increments epoch if committing packet)"): `prior + 1`
//!   through `u64::checked_add` (overflow is `ValueError`, never wrap).
//!   Shape mirrors the crate-ready `pbrf.rs:183-194` `is_target_compatible`
//!   `authoritative == first.epoch + 1` arm and `pbrf.rs:209-211`
//!   `commit_gate` `packet_epoch != forest_epoch + 1` stale arm — integer
//!   increment only, never the packet binding (packets stay Python).
//! - `joint_world_is_next_epoch` <- same rule as a bool predicate
//!   (`next == prior.checked_add(1)`, `false` on overflow): the stale-reject
//!   half of the `pbrf.rs` pair above. Returns `bool`, never raises.
//! - `joint_world_check_config` <- `JointTypeWorldConfig.__post_init__`
//!   numeric lanes (`joint_uncertainty.py:503-516`: `rho`/`epsilon`/
//!   `calibration_threshold` finite `float`, `epsilon in [0,1]`, `rho >= 0`,
//!   `max_particles` positive `int` bool-excluded) fused with the
//!   `UncertaintySet` numeric lanes (`:344-351` `rho` finite `>= 0`,
//!   `epsilon` finite in `[0,1]`; `:364-366` the finite re-audit) and
//!   `is_nonempty` (`:384-386` `rho >= 0 and 0 <= epsilon <= 1`). The string
//!   vocabulary lanes (`divergence_direction` / `support_class` /
//!   `rationality_rule` membership, `theta_ids` subset) stay Python — owned
//!   by the `JOINT_*` consts in `joint_types.rs:106-110`, referenced, never
//!   retyped. Returns `bool`, never raises (the translator maps `false` to
//!   the oracle `ContractError` text).
//!
//! TABLED (stays Python, per-fn reasons):
//! - `joint_type_world.py` facade re-exports (`:12-92`: `JointParticle`,
//!   `JointPosterior`, `OpponentTypePolicy`, `JointTypeWorldConfig`,
//!   `UncertaintySet`, `JointTypeWorldPlanner`, the six fn re-exports): no
//!   logic of its own — the translators below are the only new names.
//! - `JointParticle` / `JointPosterior` dataclasses (`joint_types.py:463-516`
//!   incl. `__post_init__` epoch/type/mass gates): frozen records stay
//!   Python; only the epoch origin (`0`) and the `+1` increment move. The
//!   `normalized` mass audit (`:512-514` compensated `sum()` vs a naive fold
//!   diverges 1 ulp — wave-8 rule, sums stay Python) and the per-particle
//!   `epoch`/`target_id` coherence (`:506-509`) stay Python with the records.
//! - `OpponentTypePolicy` + `distribution_for` / `log_prob`
//!   (`joint_types.py:324-408` hash-seeded Dirichlet pseudo-counts): the
//!   likelihood lane already rides `joint_posterior_weights` via the private
//!   `joint_row_likelihood` core (`search.rs:1191-1265`), reused by
//!   reference, never a second impl.
//! - `info_key_for_observation` (`:240-266`), `validate_hidden_permutation_`
//!   `invariance` (`:269-316`), `validate_same_information_equality`
//!   (`:411-455`): live `ActorObservation` / `FullWorld` orchestration over
//!   caller-side worlds through `feed::canon` (fused joint path), bool
//!   verdicts, never plain values.
//! - `exact_joint_posterior_oracle` (`joint_uncertainty.py:52-175`): already
//!   bridged — `search::joint_posterior_weights` runs likelihood + normalize
//!   detached; the remainder is live-world assembly (`world_actor_
//!   observation` + `info_key_for_observation` memoized per `world_ref`),
//!   `sum()` mass audits (compensated `sum()` stays Python), and
//!   `JointParticle` provenance rebuild. Reused, never a second impl.
//! - `hidden_marginalization` (`:178-180` alias for `marginal_theta`),
//!   `preserve_correlation_check` (`:183-214` live-posterior orchestration
//!   with `marginal_theta()` renormalization `tot = sum(out.values())`
//!   plus `+=` accumulation and the `0.05`/`0.01` `isclose` scan — sums stay
//!   Python), `sequential_joint_update` (`:217-237` driver looping the
//!   oracle), `coherent_trajectory` (`:240-292` hash-draw sampler explicitly
//!   OUT — generic `sha256(rng_seed + …)` draws over live worlds/policy
//!   tables, not CTR `natural_indices`/`sampled_draws`; no pyfn covers it
//!   per recon B4).
//! - `UncertaintySet.divergence` / `is_feasible` (`:388-469`): already
//!   bridged via `joint_uncertainty_divergence` /
//!   `joint_uncertainty_is_feasible` (`joint_uncertainty.rs:251-278`),
//!   reused, never reimplemented. `contains_nominal` (`:368-382` live
//!   `distribution_for` per theta + `1e-9` sum audit) stays Python.
//! - `JointTypeWorldConfig` string lanes + `make_joint_type_world_candidate_`
//!   `spec` (`:477-601` digest binding via `contracts::make_digest_text`,
//!   budget assembly, `CandidateSpec` construction): frozen records + live
//!   orchestration stay Python; only the numeric gates move.
//! - `JointTypeWorldPlanner._ensure_joint_prior` (`joint_planner.py:123-191`
//!   live belief sampler `NaturalBelief.begin` + tiny-corpus build +
//!   `target_id`/`epoch_id` reads off the live epoch), `act` (`:193-410`
//!   hash-leaf proxy + `deterministic_joint_gumbel` perturbation — already
//!   bridged via `joint_types_gumbel` — + torch `UtilityVector` + telemetry
//!   + `time.monotonic_ns` clocks OUT), `observe` (`:412-457` live posterior
//!     threading driver, not frozen consts + plain values), `ponder`
//!     (`:459-461` deterministic no-op).
//! - `joint::ensure_prior` / `exact_posterior` / `robust_select`
//!   (`joint.rs:59-82`, `:89-143`, `:189-217`): live world-handle (`u64`
//!   arena refs) + likelihood-threaded posterior + expected-value selection
//!   over caller vectors — reachable through the existing
//!   `joint_posterior_weights` / `joint_normalize_weights` /
//!   `joint_posterior_fused` lanes by reference, never a second impl here.
//!
//! NONE-duplication note: `rg` for `JOINT_WORLD_|joint_world_next_epoch|
//! joint_world_is_next_epoch|joint_world_check_config` over `crates/` hits
//! nothing before this file (no owner exists — frozen here with oracle line
//! cites, never recopied elsewhere); `rg` for `FIRST_EPOCH` over
//! `crates/bridge/src` hits only this file (the `0` default in
//! `joint_planner.py:164` stays the Python fallback literal); the posterior
//! lanes (`joint_posterior_weights`, `joint_normalize_weights`,
//! `joint_posterior_fused`) are reused by reference (`search.rs:1418-1596`),
//! never reimplemented; the `JOINT_*` vocabulary consts live in
//! `joint_types.rs:104-112` and the `JOINT_UNCERTAINTY_*` consts in
//! `joint_uncertainty.rs:130-148` (this file stores neither).
//!
//! Shape per fn: Python validates with the oracle `ContractError` text first
//! (byte-identical HEAD text for the epoch gates, fresh `prior_epoch must be
//! non-negative int` text for the new increment helpers sharing the
//! `joint_types.py:485` shape), then calls positionally for plain domains
//! only — `u64`-range non-`bool` `int` epochs, `float` (never `int`) rho /
//! epsilon / calibration plus non-`bool` `int` max_particles. Exotic domains
//! (`bool`, negative, non-`int` epochs; `int` rho/epsilon/calibration; `bool`
//! max_particles) and `u64::MAX` prior (unbounded Python `+1` vs checked
//! overflow) keep the pure-Python fallback below so no new `ContractError`
//! is introduced, mirroring `joint_types_gumbel` (`joint_types.rs:71-74`);
//! attached staging here is vacuous (typed `u64`/`f64` extraction), ONE
//! `py.detach` over owned plain data with zero Python API inside, `ValueError`
//! on the overflow reject only (per `contracts.rs:117-123`), `bool` gates
//! never raise.
//!
//! Single-cdylib tree: registers its const + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `joint_uncertainty::register`, `joint_uncertainty.rs:283-324`); no new
//! entry point. MAIN wiring: `pub mod joint_world;` in `lib.rs`
//! (alphabetical, between the `joint_uncertainty` and `local_graph` entries)
//! plus `crate::joint_world::register(&sub)?;` in `search::register` next to
//! the `joint_uncertainty` line (`crates/bridge/src/search.rs:1747`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `joint_uncertainty::register`
//! (`joint_uncertainty.rs:283-324`); `sub.add` plain const mirrors
//! `search_shared::register` (`search_shared.rs:69-87`); `#[pyfunction]` +
//! `py.detach` + `ValueError` map mirror `search::gumbel_for_action`
//! (`search.rs:254-274`); `Ok(py.detach(|| …))` bool return mirrors the
//! `pbrf_forest` bool gates (`pbrf_forest.rs:79-96`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Frozen first epoch (`joint_planner.py:164` `getattr(epoch, "epoch", 0)`;
/// `joint_types.py:484-485` admits `0`, `joint_uncertainty.py:74-75` gates
/// below it).
const JOINT_WORLD_FIRST_EPOCH: u64 = 0;

/// Pure next-epoch core (caller-increments-epoch rule,
/// `joint_uncertainty.py:70`, via the `pbrf.rs:183-194` / `pbrf.rs:209-211`
/// checked `+1` shape). Owned `u64` only — detach-safe.
fn next_epoch_core(prior_epoch: u64) -> Result<u64, String> {
    prior_epoch
        .checked_add(1)
        .ok_or_else(|| format!("joint world next epoch overflows u64 for prior {prior_epoch}"))
}

/// Pure is-next-epoch predicate (stale-reject half of the `pbrf.rs` pair:
/// `next == prior + 1`, `false` on overflow). Owned `u64`s only.
fn is_next_epoch_core(prior_epoch: u64, next_epoch: u64) -> bool {
    match prior_epoch.checked_add(1) {
        Some(expect) => expect == next_epoch,
        None => false,
    }
}

/// Pure numeric config gate (`JointTypeWorldConfig.__post_init__` numeric
/// lanes, `joint_uncertainty.py:503-516`, fused with the `UncertaintySet`
/// numeric lanes, `:344-351` + `:364-366`, and `is_nonempty`, `:384-386`):
/// `rho` finite `>= 0`, `epsilon` finite in `[0,1]`, `calibration_threshold`
/// finite, `max_particles >= 1` (`u64` typing already excludes negative /
/// `bool` / non-`int` — the translator guards those to the fallback).
/// Owned plain data only — detach-safe.
fn check_config_core(rho: f64, epsilon: f64, max_particles: u64, calibration: f64) -> bool {
    if !rho.is_finite() || rho < 0.0 {
        return false;
    }
    if !epsilon.is_finite() || epsilon < 0.0 || epsilon > 1.0 {
        return false;
    }
    if max_particles == 0 {
        return false;
    }
    if !calibration.is_finite() {
        return false;
    }
    true
}

/// Next epoch over an owned `u64` (`joint_uncertainty.py:70` caller-increments
/// rule via `pbrf.rs:183-194` + `pbrf.rs:209-211`).
///
/// Positional `(prior_epoch,)` matches the translator order; the translator
/// validates with the oracle text first and only rides the bridge for
/// `u64`-range non-`bool` `int` priors below `u64::MAX` (the `MAX` fallback
/// computes unbounded `+1`). Compute runs detached; `ValueError` on
/// overflow, never a default.
#[pyfunction]
#[pyo3(signature = (prior_epoch,))]
fn joint_world_next_epoch(py: Python<'_>, prior_epoch: u64) -> PyResult<u64> {
    py.detach(|| next_epoch_core(prior_epoch))
        .map_err(PyValueError::new_err)
}

/// Is-next-epoch predicate over owned `u64`s (stale-reject half).
///
/// Positional `(prior_epoch, next_epoch)`; the translator only rides the
/// bridge for `u64`-range non-`bool` `int` pairs (exotic domains read `false`
/// via the fallback). Compute runs detached; never raises.
#[pyfunction]
#[pyo3(signature = (prior_epoch, next_epoch))]
fn joint_world_is_next_epoch(py: Python<'_>, prior_epoch: u64, next_epoch: u64) -> PyResult<bool> {
    Ok(py.detach(|| is_next_epoch_core(prior_epoch, next_epoch)))
}

/// Numeric config gate over owned scalars (`JointTypeWorldConfig`
/// `:503-516` + `UncertaintySet` `:344-351` / `:364-366` / `:384-386`).
///
/// Positional `(rho, epsilon, max_particles, calibration_threshold)` in
/// `JointTypeWorldConfig` field order; the translator only rides the bridge
/// for `float` rho/epsilon/calibration plus non-`bool` `int` max_particles
/// (plain domains — `int` floats and `bool` counts keep the fallback so the
/// oracle's `isinstance(v, float)` shape is preserved). Compute runs
/// detached; returns `bool`, never raises.
#[pyfunction]
#[pyo3(signature = (rho, epsilon, max_particles, calibration_threshold))]
fn joint_world_check_config(
    py: Python<'_>,
    rho: f64,
    epsilon: f64,
    max_particles: u64,
    calibration_threshold: f64,
) -> PyResult<bool> {
    Ok(py.detach(|| check_config_core(rho, epsilon, max_particles, calibration_threshold)))
}

/// Register the joint-world const + pyfns on the EXISTING `search` submodule
/// (mirrors `joint_uncertainty::register` on the shared submodule; MAIN
/// calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("JOINT_WORLD_FIRST_EPOCH", JOINT_WORLD_FIRST_EPOCH)?;
    sub.add_function(wrap_pyfunction!(joint_world_next_epoch, sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_world_is_next_epoch, sub)?)?;
    sub.add_function(wrap_pyfunction!(joint_world_check_config, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_first_epoch_is_zero() {
        assert_eq!(JOINT_WORLD_FIRST_EPOCH, 0);
    }

    #[test]
    fn next_epoch_increments() {
        assert_eq!(next_epoch_core(0).expect("0 + 1"), 1);
        assert_eq!(next_epoch_core(41).expect("41 + 1"), 42);
        assert_eq!(
            next_epoch_core(u64::MAX - 1).expect("MAX - 1 + 1"),
            u64::MAX
        );
    }

    #[test]
    fn next_epoch_overflow_rejects() {
        assert!(next_epoch_core(u64::MAX).is_err());
    }

    #[test]
    fn is_next_epoch_predicate() {
        assert!(is_next_epoch_core(0, 1));
        assert!(is_next_epoch_core(7, 8));
        assert!(!is_next_epoch_core(7, 7));
        assert!(!is_next_epoch_core(7, 9));
        assert!(!is_next_epoch_core(0, 0));
        // Overflow can never be "next": MAX + 1 is unrepresentable.
        assert!(!is_next_epoch_core(u64::MAX, 0));
        assert!(!is_next_epoch_core(u64::MAX, u64::MAX));
    }

    #[test]
    fn check_config_accepts_head_defaults() {
        // HEAD defaults (`joint_uncertainty.py:482-488`): rho 0.15, epsilon
        // 0.05, max_particles 16, calibration 0.05.
        assert!(check_config_core(0.15, 0.05, 16, 0.05));
        assert!(check_config_core(0.0, 0.0, 1, 0.0));
        assert!(check_config_core(0.0, 1.0, 1, f64::MIN_POSITIVE));
    }

    #[test]
    fn check_config_rejects_each_lane() {
        // rho lane: negative / non-finite.
        assert!(!check_config_core(-0.5, 0.05, 16, 0.05));
        assert!(!check_config_core(f64::NAN, 0.05, 16, 0.05));
        assert!(!check_config_core(f64::INFINITY, 0.05, 16, 0.05));
        // epsilon lane: outside [0,1] / non-finite.
        assert!(!check_config_core(0.15, -0.1, 16, 0.05));
        assert!(!check_config_core(0.15, 1.5, 16, 0.05));
        assert!(!check_config_core(0.15, f64::NAN, 16, 0.05));
        assert!(!check_config_core(0.15, f64::INFINITY, 16, 0.05));
        // max_particles lane: zero.
        assert!(!check_config_core(0.15, 0.05, 0, 0.05));
        // calibration lane: non-finite only (no range gate in the oracle).
        assert!(!check_config_core(0.15, 0.05, 16, f64::NAN));
        assert!(!check_config_core(0.15, 0.05, 16, f64::INFINITY));
        assert!(!check_config_core(0.15, 0.05, 16, f64::NEG_INFINITY));
        // Negative-zero rho is valid (>= 0 holds, mirroring the oracle).
        assert!(check_config_core(-0.0, 0.05, 16, 0.05));
    }
}
