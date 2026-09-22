//! qual_replay: deterministic replay-hash leaves over `hydra-feed` + `sha2`.
//!
//! DAG: this module depends on the feed crate (`canon`/`digest`/`partition`),
//! `serde_json`, `sha2`, + pyo3 ONLY (same edge as the rest of the bridge;
//! no new dependency). FORBIDS: canon re-printing (the feed `canon` owner is
//! the single printer), Python-object orchestration (spec/observation/legal
//! objects stay Python-side; only canonical scalars cross), wall-clock/RNG,
//! file/artifact IO, and the privilege/compute-only judges (those stay in
//! `analysis/qual_budget.py`).
//!
//! Rust port of the pure compute in `python/hydra2/analysis/qual_replay.py`
//! (deterministic replay hash + value-vector core). The `_aid`
//! repr/getattr canonicalization (`:80-87`) stays Python-side: Python `repr`
//! has no Rust equivalent, and the only nearby crate owner
//! (`hydra_search::persistence_kernel::action_key`,
//! `persistence_kernel.rs:335-351`) maps frozen kind strings to ordinals
//! while the oracle hashes them via `repr` — a different mapping, so there
//! is NO compatible aid owner (lesson-2 grep evidence, not an oversight).
//! The `_pick` index parse and the `value_l2` accumulation stay Python-side
//! too: the former is a two-line modulo (a bridge round-trip exceeds its
//! cost), the latter ends in `** 0.5` (C `pow`, no bit-exact Rust
//! counterpart), so Python remains the combining judge while Rust owns the
//! digest and the vector core.
//!
//! Hash routing (never reimplemented): the replay payload digests through
//! `hydra_feed::digest::of_canonical` (the single canon+hash site,
//! `feed/src/digest.rs:46`), and the digest shape gate is
//! `hydra_feed::partition::is_digest_text` (`feed/src/partition.rs:583`,
//! `packet_decode.rs:419` precedent); the strict lowercase-only validation
//! stays on the translator's already-bridged `validate_digest` call
//! (`qual_replay.py:75`). The value-vector preimage `f"{model_hash}:{aid}"`
//! has no crate owner (`hydra_search::eval::sha256_bytes` is `pub(crate)`;
//! `ismcts_driver`'s u16/65535 mapping uses a different domain —
//! `sha256(wid:leaf_kind)`, little-endian words, zero-sum centered —
//! `ismcts_driver.rs:213-229`), so the single `sha2` fold lives here
//! (`search.rs:1231-1234` streaming precedent, `search.rs:86` import
//! precedent).
//!
//! Single-cdylib tree: registers its pyfns on the shared
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_shared::register` on `search`, `search_shared.rs:58-60`); no new
//! entry point.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};

/// Frozen mode literals (`qual_replay.py:76` — the translator owns the
/// `ContractError` surface; this is the detached fail-closed mirror).
const REPLAY_MODES: [&str; 3] = ["gameplay_5s", "ponder", "analysis"];

/// Mode gate (attached call-site + kernel; unit-tested below): mirrors the
/// frozen `("gameplay_5s", "ponder", "analysis")` literal.
fn check_mode(mode: &str) -> Result<(), String> {
    if REPLAY_MODES.contains(&mode) {
        Ok(())
    } else {
        Err(format!(
            "mode must be gameplay_5s/ponder/analysis, got {mode:?}"
        ))
    }
}

/// Digest-shape gate: the crate owner (`feed::partition`), never a local
/// printer. The strict lowercase-only validation stays on the translator's
/// already-bridged `validate_digest` call.
fn check_digest_shape(text: &str) -> Result<(), String> {
    if hydra_feed::partition::is_digest_text(text) {
        Ok(())
    } else {
        Err(format!(
            "observation_hash must be sha256:<64 hex>, got {text:?}"
        ))
    }
}

/// Pure replay-digest kernel (`deterministic_replay_hash`, `:89-96`): sort
/// the canonical aids, stage the frozen payload in sorted-key order, digest
/// through the feed canon+hash owner. `aids` arrive canonicalized
/// (translator-owned `_aid` + sort); the re-sort here is idempotent, so
/// caller order never affects the digest.
fn replay_digest_kernel(
    candidate_id: &str,
    observation_hash: &str,
    aids: &[i64],
    case_id: &str,
    mode: &str,
    seed_extra: &str,
) -> Result<String, String> {
    check_mode(mode)?;
    check_digest_shape(observation_hash)?;
    let mut sorted: Vec<i64> = aids.to_vec();
    sorted.sort_unstable();
    let aids_json: Vec<JsonValue> = sorted.iter().map(|aid| json!(aid)).collect();
    let mut payload = BTreeMap::new();
    payload.insert("candidate_id".to_string(), json!(candidate_id));
    payload.insert("case_id".to_string(), json!(case_id));
    payload.insert("legal_action_ids".to_string(), JsonValue::Array(aids_json));
    payload.insert("mode".to_string(), json!(mode));
    payload.insert("observation_hash".to_string(), json!(observation_hash));
    payload.insert("seed_extra".to_string(), json!(seed_extra));
    hydra_feed::digest::of_canonical(&payload)
        .map_err(|err| format!("qual_replay payload failed canon: {err}"))
}

/// Pure value-vector kernel (`compare_gameplay_analysis._value_for`,
/// `:192-200`): `sha256(f"{model_hash}:{aid}")`, four big-endian u16 words
/// mapped `(w / 65535.0) * 2.0 - 1.0` in oracle op order — bit-identical f64
/// (exact division rounding, exact `*2`, exact `-1`; no `±0`/NaN edge: only
/// `w = 32767.5` could yield zero, unrepresentable in u16).
fn replay_value_vector_kernel(model_hash: &str, aid: i64) -> (f64, f64, f64, f64) {
    let preimage = format!("{model_hash}:{aid}");
    let sum = Sha256::digest(preimage.as_bytes());
    let mut out = [0.0f64; 4];
    let mut i = 0;
    while i < 4 {
        let word = u16::from_be_bytes([sum[2 * i], sum[2 * i + 1]]);
        out[i] = f64::from(word) / 65535.0 * 2.0 - 1.0;
        i += 1;
    }
    (out[0], out[1], out[2], out[3])
}

/// Deterministic replay hash (`deterministic_replay_hash`, `:57-96`).
///
/// Attached arg-checks mirror the translator (fail closed before the
/// detach); compute runs detached; only owned scalars cross
/// (`search.rs:253-274` precedent).
#[pyfunction]
#[pyo3(signature = (candidate_id, observation_hash, legal_aids, case_id, mode, seed_extra))]
fn qual_replay_hash(
    py: Python<'_>,
    candidate_id: String,
    observation_hash: String,
    legal_aids: Vec<i64>,
    case_id: String,
    mode: String,
    seed_extra: String,
) -> PyResult<String> {
    check_mode(&mode).map_err(PyValueError::new_err)?;
    check_digest_shape(&observation_hash).map_err(PyValueError::new_err)?;
    let out = py.detach(|| {
        replay_digest_kernel(
            &candidate_id,
            &observation_hash,
            &legal_aids,
            &case_id,
            &mode,
            &seed_extra,
        )
    });
    out.map_err(PyValueError::new_err)
}

/// Replay value vector (`compare_gameplay_analysis._value_for`, `:192-200`).
///
/// Pure sha draw — identical inputs give identical outputs regardless of
/// call order or global RNG. Compute runs detached; only owned scalars
/// cross.
#[pyfunction]
#[pyo3(signature = (model_hash, aid))]
fn qual_replay_value_vector(
    py: Python<'_>,
    model_hash: String,
    aid: i64,
) -> PyResult<(f64, f64, f64, f64)> {
    Ok(py.detach(|| replay_value_vector_kernel(&model_hash, aid)))
}

/// Register the qual-replay leaves on the shared `search` submodule
/// (mirrors `search_shared::register`); MAIN wires one line in
/// `search::register`.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(qual_replay_hash, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_replay_value_vector, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod qual_replay_tests {
    use super::*;

    fn zeroes() -> String {
        format!("sha256:{}", "0".repeat(64))
    }

    fn model_a() -> String {
        format!("sha256:{}", "a".repeat(64))
    }

    fn assert_bits_eq(got: (f64, f64, f64, f64), exp: (f64, f64, f64, f64)) {
        assert_eq!(got.0.to_bits(), exp.0.to_bits());
        assert_eq!(got.1.to_bits(), exp.1.to_bits());
        assert_eq!(got.2.to_bits(), exp.2.to_bits());
        assert_eq!(got.3.to_bits(), exp.3.to_bits());
    }

    #[test]
    fn replay_hash_golden_analysis_matches_oracle() {
        // Oracle: `deterministic_replay_hash(candidate_id="candidate0",
        // observation_hash=sha256:0*64, legal_actions=(3, 1, 2),
        // case_id="analysis_replay_case", mode="analysis", seed_extra="")`
        // captured pre-edit from the HEAD Python oracle.
        let got = replay_digest_kernel(
            "candidate0",
            &zeroes(),
            &[3, 1, 2],
            "analysis_replay_case",
            "analysis",
            "",
        )
        .unwrap();
        assert_eq!(
            got,
            "sha256:66ab305fdc89bbb41d301b1480c652fb1f072db0d412330cf0aca0be2491b25a"
        );
    }

    #[test]
    fn replay_hash_golden_gameplay_matches_oracle() {
        // Oracle: same candidate, `observation_hash=sha256:ab*32`,
        // `case_id="c0_gate"`, `mode="gameplay_5s"`, `seed_extra="x"`.
        let obs = format!("sha256:{}", "ab".repeat(32));
        let got = replay_digest_kernel(
            "candidate0",
            &obs,
            &[3, 1, 2],
            "c0_gate",
            "gameplay_5s",
            "x",
        )
        .unwrap();
        assert_eq!(
            got,
            "sha256:780b5ddf853ba9bc19467bd8a6021948ae2d084764f28fc02b0e439ba17333da"
        );
    }

    #[test]
    fn replay_hash_golden_ponder_exotic_aids_matches_oracle() {
        // Oracle: `candidate_id="candidate1"`, `mode="ponder"`, aids already
        // canonicalized by the translator (`[5, 9, 2322915238]` — the last is
        // the `repr`-fallback branch, u32 range, carried losslessly in i64).
        let got = replay_digest_kernel(
            "candidate1",
            &zeroes(),
            &[5, 9, 2322915238],
            "analysis_replay_case",
            "ponder",
            "",
        )
        .unwrap();
        assert_eq!(
            got,
            "sha256:14d6f76e9252188299d260bb96e82d486723db254932075006d28125cf53d1b7"
        );
    }

    #[test]
    fn replay_hash_sorts_legal_set() {
        // Order-insensitivity (oracle `sorted(_aid(...))`): any permutation
        // of the same aids digests identically.
        let obs = zeroes();
        let a =
            replay_digest_kernel("candidate0", &obs, &[3, 1, 2], "case", "analysis", "").unwrap();
        let b =
            replay_digest_kernel("candidate0", &obs, &[1, 2, 3], "case", "analysis", "").unwrap();
        let c =
            replay_digest_kernel("candidate0", &obs, &[2, 3, 1], "case", "analysis", "").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn replay_hash_splits_modes() {
        // The mode label is load-bearing: identical inputs under different
        // modes digest differently (both deterministic).
        let obs = zeroes();
        let gp =
            replay_digest_kernel("candidate0", &obs, &[1, 2], "case", "gameplay_5s", "").unwrap();
        let an = replay_digest_kernel("candidate0", &obs, &[1, 2], "case", "analysis", "").unwrap();
        assert_ne!(gp, an);
        assert_eq!(
            gp,
            replay_digest_kernel("candidate0", &obs, &[1, 2], "case", "gameplay_5s", "").unwrap()
        );
    }

    #[test]
    fn replay_hash_rejects_bad_mode() {
        let err =
            replay_digest_kernel("candidate0", &zeroes(), &[1], "case", "blitz", "").unwrap_err();
        assert!(
            err.contains("gameplay_5s/ponder/analysis"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn replay_hash_rejects_bad_digest() {
        assert!(replay_digest_kernel("c", "", &[1], "case", "analysis", "").is_err());
        assert!(replay_digest_kernel("c", "sha256:zz", &[1], "case", "analysis", "").is_err());
        assert!(replay_digest_kernel("c", &"0".repeat(64), &[1], "case", "analysis", "").is_err());
    }

    #[test]
    fn value_vector_goldens_are_bit_identical_to_oracle() {
        // Oracle: `compare._value_for` replica over `model_hash=sha256:a*64`,
        // captured pre-edit from the HEAD Python oracle.
        assert_bits_eq(
            replay_value_vector_kernel(&model_a(), 0),
            (
                -0.7840848401617456,
                0.2271915770199131,
                -0.9305409323262379,
                -0.1564812695506218,
            ),
        );
        assert_bits_eq(
            replay_value_vector_kernel(&model_a(), 5),
            (
                -0.5267566948958572,
                -0.4228122377355612,
                -0.8361486228732739,
                0.29643701838712144,
            ),
        );
        assert_bits_eq(
            replay_value_vector_kernel(&model_a(), 70000),
            (
                0.8701457236591135,
                0.7482871747920958,
                0.9541313801785305,
                0.3095292591744869,
            ),
        );
    }

    #[test]
    fn value_vector_real_model_hash_golden() {
        // Oracle: full `compare_gameplay_analysis` fixture (`candidate0`
        // gameplay spec, aid 0 via the default-`0` branch) — both value
        // vectors in the captured comparison equal this.
        let got = replay_value_vector_kernel(
            "sha256:8ebcb2164d0c4c54cb92513fcfedae57597ae61f6b5b9fb68f63ec0fdf1797dc",
            0,
        );
        assert_bits_eq(
            got,
            (
                0.14998092622262904,
                -0.7999847409781033,
                -0.2990920881971466,
                -0.9184252689402609,
            ),
        );
    }

    #[test]
    fn value_vector_range_and_determinism() {
        // Four-seat placement range `[-1, 1]`; pure draw replays exactly.
        for aid in [0, 1, 5, 70000, 2322915238] {
            let v = replay_value_vector_kernel(&model_a(), aid);
            for x in [v.0, v.1, v.2, v.3] {
                assert!((-1.0..=1.0).contains(&x), "out of range: {x}");
                assert!(x.is_finite(), "non-finite: {x}");
            }
            assert_bits_eq(v, replay_value_vector_kernel(&model_a(), aid));
        }
    }
}
