//! training_leaves: pure training-config leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (frozen literals + detached int/string/float-closed-form
//! math; no feed/shard/search edge, no new dependency). FORBIDS: torch/CUDA/
//! triton/JAX/riichienv/arrow/parquet/zstd/jsonschema/requests/file-IO/RNG/
//! clocks/mutable-state (all stay Python), float sums over data (stay Python
//! per the wave-16 contract — the scheduler closed form below returns one
//! `f64` factor per step int, never a reduction), and `ContractError` shaping
//! (the bridge raises `ValueError`; thin Python translators map to
//! `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/training/` (HEAD text at port
//! time; consts frozen from it, never redefined):
//! - `training_minibatch_size` <- `TrainingLoopConfig.optimizer_minibatch_size`
//!   (`loop_state.py:210-212`) shared with `ReplayConfig.optimizer_minibatch_size`
//!   (`replay_state.py:120-122`): plain `microbatch * accumulation` int shape.
//! - `training_sizes_gate` <- the three int-positivity heads of
//!   `TrainingLoopConfig.validate` (`loop_state.py:225-230`) shared with the
//!   `ReplayConfig.validate` prefix (`replay_state.py:134-139`).
//! - `training_prefetch_gate` <- the prefetch-int gate (`loop_state.py:272-279`:
//!   bool-rejecting, `1 <= depth <= 16`).
//! - `loop_precision_gate` <- the loop precision enum (`loop_state.py:238-239`:
//!   `"fp32"` / `"bf16_mixed"`).
//! - `replay_precision_gate` <- the replay fp32-only gate
//!   (`replay_state.py:140-144`). This port claims the replay-twin names the
//!   `loop_state.rs:36-38` NONE-owner note parked in Python.
//! - `replay_require_sha256` <- `replay_state._require_sha256`
//!   (`replay_state.py:54-57`): the same loose `startswith("sha256:")` +
//!   `len == 71` shape the loop twin owns (no hex check — `"sha256:" +
//!   "Z" * 64` passes by oracle parity, pinned below).
//! - const `REPLAY_FORBIDDEN_BATCH_KEYS` <- `FORBIDDEN_REPLAY_KEYS`
//!   (`replay_state.py:23-37`: 11 entries; the loop 9-tuple stays owned by
//!   `loop_state.rs:58-69`).
//! - `scheduler_lr_factor` <- NEW pure projection of `_build_scheduler`
//!   (`stream_build.py:209-259`): warmup `LinearLR` + `cosine` / `constant` /
//!   `linear` decay as a closed-form `f64` multiplier of one step int (torch
//!   construction itself stays Python). Unknown-parameter-key and
//!   `end_factor`-range failures raise the oracle-identical texts.
//! - `prefix_hashes_name` <- `_prefix_hashes_name` (`stream_build.py:372-374`:
//!   `f"prefix-hashes-{update:06d}.txt"`).
//! - `backward_autocast_for` <- `_backward_pass_autocast_for`
//!   (`stream_resume.py:712-721`: compiled non-fp32 -> `"off"`, else `None`).
//! - `require_replay_backend` <- `_require_replay_backend`
//!   (`stream_expand.py:106-112`). NOTE the near-twin in `rc_parse.rs:1114`
//!   gates `data.replay_backend` with a different text (`data.replay_backend
//!   must be 'rust'`) and a `"rust"` default — that gate stays
//!   the owner there; this one mirrors the stream-expand text verbatim.
//! - `plane_suffix` <- `_plane_suffix` (`shard_reader.py:105-110`): ordered
//!   suffix match over the lowercased filename (`.arrow.ipc` before `.ipc`).
//! - consts `TRAINING_REPLAY_BACKENDS` (`stream_expand.py:103`),
//!   `TRAINING_PLANE_SUFFIXES` (`shard_reader.py:107` order),
//!   `TRAINING_SINGLETON_WORLD` / `TRAINING_SINGLETON_RANK`
//!   (`stream_expand.py:283-284`), `TRAINING_PREFETCH_MIN` / `MAX`
//!   (`loop_state.py:275`), `TRAINING_LOOP_PRECISIONS` (`loop_state.py:238`).
//!
//! LOGIC staying Python (TABLE verdict, per-fn reasons):
//! - `loop_state.TrainingLoopConfig` float/bool/map sections (`w_policy`,
//!   `w_event`/`w_belief` maps, `label_smoothing`, `sampling_ratios`,
//!   `stratified_sampling`, `log_per_type_metrics`, `fit_temperature`):
//!   float-map validation/sums stay Python per the wave-16 contract.
//! - `loop_batch` (`_window_means`, `_model_forward`, `_move_batch_to_device`,
//!   telemetry quantiles/summaries): torch/CUDA owned; float percentile sums
//!   stay Python.
//! - `loop_checkpoint` / `loop_engine` / `loop_train`, `loop_loss`,
//!   `objectives_loss`, `objectives_metrics`, `fused_ce`, `adapters`,
//!   `dataset_encode` (incl. its `_require_action_width` vocab gate, which
//!   stays with the torch encode caller): torch/triton owned.
//! - `dataset_store` (`AuthoritativeParquetDataset`, `_row_action_kind`,
//!   `_validate_sampling_ratios`, `_validate_kind_by_id`,
//!   `build_stratified_order`): parquet IO + float-map validation + `round()`
//!   banker's-rounding parity risk stay Python.
//! - `pinned_ring` (slot shapes/dtypes/layout, `PinnedRing`): torch dtypes +
//!   CUDA ring owned.
//! - `replay_engine` / `replay_checkpoint` / replay batch helpers:
//!   torch + checkpoint IO owned.
//! - `rust_batch` / `rust_stream` (freshness mtime, extension load, torch
//!   assembly): file-IO + torch owned.
//! - `shard_reader` (`ShardReader`, `_sha256_file`, `_canonical_perm`,
//!   `dataset_hash_of`, `_bucket_ceil`, manifest consts): file-IO + torch RNG
//!   forbidden; `dataset_hash_of` + `_bucket_ceil` are plan-item-6 verbatim
//!   duplicates (columnar shim / `ring.bucket_for_length` own the ceil).
//! - `stream_build` (`_build_model`, `_build_optimizer`, `_build_scheduler`
//!   torch construction, `_manifest_hashes_for_loop`, `_rng_anchors`,
//!   `_verify_rng_anchors`, `_sidecar_cursor`, `_load_prefix_hashes`):
//!   torch + file-IO + live-RNG/cursor objects stay Python.
//! - `stream_checkpoint`, `stream_dataset`, `stream_dataset_buffer`,
//!   `stream_resume` payload/feed/profiler paths, `stream_train`: IO + torch.
//! - `stream_expand` (`_quarantine_class` regex chain, `_needs_privileged_labels`
//!   / `_split_ratios` float weights, `_action_kind_for_id` JSON load + mutable
//!   cache, `_history_bucket_of` bucket-triple duplicate, `_wall_override`,
//!   row/walk assembly): regex/file-IO/float/mutable-state stay Python.
//! - `stream_scan` + `_rc_*` family: already owned by `stream_scan.rs` /
//!   `rc_*.rs` — not re-ported.
//! - replay `_REQUIRED_MANIFEST_KEYS` (`replay_state.py:40-51`): verbatim
//!   duplicate of `LOOP_REQUIRED_MANIFEST_KEYS` (`loop_state.rs:71-83`) —
//!   deleted-duplicate per plan item 6, not re-ported.
//!
//! Shape per fn: attached staging (exact `repr(value)` text via the live
//! Python API, so `{value!r}` renders byte-exact per `rc_require.rs:60-65`)
//! -> ONE `py.detach(|| ...)` over owned plain data with zero Python API
//! inside (per `contracts.rs:434-437`) -> attached wrap as `PyValueError`
//! (per `contracts.rs:117-123`). Consts via `sub.add` (`PyTuple` per
//! `contracts.rs:1156`). `bool` maps to `0`/`1` for the size/product math
//! (the oracle comparisons admit `True == 1`); the prefetch gate rejects
//! `bool` first exactly like the oracle's `isinstance(..., bool)` head
//! (per `contracts.rs:69-74`).
//!
//! Tolerances: int/string gates are bit-exact (repr-staged messages);
//! `scheduler_lr_factor` is float-bit parity CLAIMED at `1e-12` absolute vs
//! the torch `LinearLR`/`CosineAnnealingLR`/`SequentialLR` composition and vs
//! the Python fallback closed form (same formula, `math.cos`); NaN inputs
//! fail closed, never propagate.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `loop_state.rs:132-144` — there is NO
//! `hydra2._native.training` submodule in `lib.rs:99-118`, so the contracts
//! site carries these leaves); no new entry point. MAIN wiring:
//! `pub mod training_leaves;` in `lib.rs` (alphabetical, after `tiles`) plus
//! `crate::training_leaves::register(&sub)?;` in `contracts.rs` next to the
//! `crate::loop_state::register(&sub)?;` line (`contracts.rs:1311`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyModule, PyTuple};
use std::f64::consts::PI;

/// Replay backends (`stream_expand.py:103`, frozen order).
pub const TRAINING_REPLAY_BACKENDS: [&str; 1] = ["rust"];

/// Plane storage suffixes in oracle match order (`shard_reader.py:107`:
/// `.arrow.ipc` precedes `.ipc`).
pub const TRAINING_PLANE_SUFFIXES: [&str; 8] = [
    ".arrow.ipc",
    ".ipc",
    ".arrow",
    ".tensor",
    ".npy",
    ".bin",
    ".raw",
    ".dat",
];

/// Single-process world size (`stream_expand.py:283`).
pub const TRAINING_SINGLETON_WORLD: i64 = 1;

/// Single-process rank (`stream_expand.py:284`).
pub const TRAINING_SINGLETON_RANK: i64 = 0;

/// Prefetch-depth bounds (`loop_state.py:275`: `1 <= depth <= 16`).
pub const TRAINING_PREFETCH_MIN: i64 = 1;
/// Prefetch-depth bounds (`loop_state.py:275`: `1 <= depth <= 16`).
pub const TRAINING_PREFETCH_MAX: i64 = 16;

/// Loop precisions (`loop_state.py:238`).
pub const TRAINING_LOOP_PRECISIONS: [&str; 2] = ["fp32", "bf16_mixed"];

/// Replay privileged batch fields (`replay_state.py:23-37`, 11 entries).
pub const REPLAY_FORBIDDEN_BATCH_KEYS: [&str; 11] = [
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
    "privileged_label",
    "wall_remaining",
    "hidden",
    "privileged_labels",
    "opponent_hidden",
];

/// Attached staging helper: exact `repr(value)` text (per `rc_require.rs:60-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Stage an int-or-other value: `bool` maps to `0`/`1` (oracle arithmetic
/// admits `True == 1`); anything else is `extract::<i64>()` or `None`.
fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<(String, Option<i64>)> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() {
        let flag: bool = obj.extract()?;
        return Ok((repr, Some(i64::from(flag))));
    }
    Ok((repr, obj.extract::<i64>().ok()))
}

/// Stage a str-or-other value (`None` for non-`str`, matching the oracle
/// `!=` semantics for the enum gates).
fn stage_str(obj: &Bound<'_, PyAny>) -> PyResult<(String, Option<String>)> {
    let repr = py_repr(obj)?;
    Ok((repr, obj.extract::<String>().ok()))
}

/// Python `repr()` for a Rust `str`: single-quoted, backslash- and
/// single-quote-escaped (matches `{key!r}` / `{filename!r}` for the ascii
/// ids and filenames these gates carry).
fn py_str_repr(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('\'');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out.push('\'');
    out
}

/// Python `str(float)` for an `f64`: `Debug` already prints `2.0`-style with
/// a trailing `.0`; only `NaN` needs remapping to Python's lowercase `nan`
/// (`inf` / `-inf` already agree).
fn py_float_str(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    format!("{value:?}")
}

/// Python `repr(list[str])`: `['a', 'b']` with `py_str_repr` elements.
fn py_str_list_repr(keys: &[String]) -> String {
    let parts: Vec<String> = keys.iter().map(|k| py_str_repr(k)).collect();
    format!("[{}]", parts.join(", "))
}

/// Pure shape core: `microbatch * accumulation` (`loop_state.py:212`,
/// `replay_state.py:122`).
fn minibatch_impl(microbatch: i64, accumulation: i64) -> Option<i64> {
    microbatch.checked_mul(accumulation)
}

/// Pure gate core: the three int-positivity heads in oracle order
/// (`loop_state.py:225-230`, `replay_state.py:134-139`).
fn sizes_gate_impl(
    micro: i64,
    accum: i64,
    max_updates: i64,
    ckpt_freq: i64,
) -> Option<&'static str> {
    if micro <= 0 || accum <= 0 || max_updates <= 0 {
        return Some("microbatch/accumulation/max_updates must be positive");
    }
    if ckpt_freq <= 0 {
        return Some("checkpoint_frequency_updates must be positive");
    }
    let product = (micro as i128) * (accum as i128);
    if product <= 0 {
        return Some("optimizer_minibatch_size must be positive");
    }
    None
}

/// Pure shape core: the oracle-loose `sha256:` prefix + 71-char gate
/// (mirrors `loop_state.py:52` / `replay_state.py:55` exactly — no hex
/// validity check). Char count (not byte length) so non-ASCII inputs agree
/// with the oracle's `len()`.
fn sha_shape_ok(text: &str) -> bool {
    text.starts_with("sha256:") && text.chars().count() == 71
}

/// Oracle-identical error text (`loop_state.py:53`, `replay_state.py:56`).
fn require_message(name: &str, repr: &str) -> String {
    format!("{name} must be sha256:<64 hex>, got {repr}")
}

/// Pure math core: linear ramp from `start` to `end` over `total` steps
/// (torch `LinearLR` factor semantics; `step` clamped into `[0, total]`).
fn linear_ramp(step: i64, start: f64, end: f64, total: i64) -> f64 {
    if total <= 0 {
        return end;
    }
    // proof: `total > 0` (checked above) and `clamped` in [0,total]; small step counts (< 2^53), exact; ramp tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let clamped_f: f64 = step.clamp(0, total) as f64;
    #[allow(clippy::cast_precision_loss)]
    let total_f: f64 = total as f64;
    let clamped = clamped_f / total_f;
    start + (end - start) * clamped
}

/// Pure math core: cosine decay from `1.0` to `final_factor` over `t_max`
/// steps (torch `CosineAnnealingLR` factor semantics, peak normalised to
/// `1.0`; at/past `t_max` the factor rests at `final_factor`).
fn cosine_decay(step: i64, t_max: i64, final_factor: f64) -> f64 {
    if t_max <= 0 {
        return final_factor;
    }
    let clamped = step.clamp(0, t_max);
    if clamped >= t_max {
        return final_factor;
    }
    // proof: `clamped` in [0,t_max] and `t_max > 0` (checked above); small step counts (< 2^53), exact; cosine tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let clamped_f: f64 = clamped as f64;
    #[allow(clippy::cast_precision_loss)]
    let t_max_f: f64 = t_max as f64;
    final_factor + (1.0 - final_factor) * 0.5 * (1.0 + (PI * clamped_f / t_max_f).cos())
}

/// Pure closed form: warmup ramp + main decay selector. `t_max_resolved`
/// and `end_resolved` are already defaulted (`main_span` / `final_factor`);
/// `span` is the warmup-adjusted horizon (`max(1, horizon - warmup)`).
fn lr_factor_impl(
    scheduler: &str,
    step: i64,
    warmup: i64,
    horizon: i64,
    final_factor: f64,
    warmup_start: f64,
    span: i64,
    t_max_resolved: i64,
    end_resolved: f64,
) -> f64 {
    if warmup > 0 && warmup >= horizon {
        return linear_ramp(step, warmup_start, 1.0, horizon);
    }
    if warmup > 0 && step < warmup {
        return linear_ramp(step, warmup_start, 1.0, warmup);
    }
    let after = if warmup > 0 { step - warmup } else { step };
    match scheduler {
        "cosine" => cosine_decay(after, t_max_resolved, final_factor),
        "linear" => linear_ramp(after, 1.0, end_resolved, span),
        _ => 1.0,
    }
}

/// Validated minibatch product (`loop_state.py:210-212`,
/// `replay_state.py:120-122`). Translators call positionally; the bridge
/// raises `ValueError`, Python maps to `ContractError`. Compute runs
/// detached with zero Python API inside; counter-free deterministic.
#[pyfunction]
#[pyo3(signature = (microbatch_size, accumulation_steps))]
fn training_minibatch_size(
    py: Python<'_>,
    microbatch_size: Bound<'_, PyAny>,
    accumulation_steps: Bound<'_, PyAny>,
) -> PyResult<i64> {
    let (micro_repr, micro) = stage_int(&microbatch_size)?;
    let (accum_repr, accum) = stage_int(&accumulation_steps)?;
    let (Some(micro), Some(accum)) = (micro, accum) else {
        let bad = if micro.is_none() {
            micro_repr
        } else {
            accum_repr
        };
        return Err(PyValueError::new_err(format!(
            "optimizer_minibatch_size requires ints, got {bad}"
        )));
    };
    let product = py.detach(|| minibatch_impl(micro, accum));
    match product {
        Some(value) => Ok(value),
        None => Err(PyValueError::new_err(
            "optimizer_minibatch_size overflow".to_owned(),
        )),
    }
}

/// Int-positivity gate (`loop_state.py:225-230`, `replay_state.py:134-139`).
/// Static oracle texts, raised in oracle order.
#[pyfunction]
#[pyo3(signature = (microbatch_size, accumulation_steps, max_updates, checkpoint_frequency_updates))]
fn training_sizes_gate(
    py: Python<'_>,
    microbatch_size: Bound<'_, PyAny>,
    accumulation_steps: Bound<'_, PyAny>,
    max_updates: Bound<'_, PyAny>,
    checkpoint_frequency_updates: Bound<'_, PyAny>,
) -> PyResult<()> {
    let (_, micro) = stage_int(&microbatch_size)?;
    let (_, accum) = stage_int(&accumulation_steps)?;
    let (max_repr, max) = stage_int(&max_updates)?;
    let (ckpt_repr, ckpt) = stage_int(&checkpoint_frequency_updates)?;
    // Non-int sizes fail closed on the staged repr (new-projection text:
    // the oracle fields are pre-validated ints, so only the bridge sees raw).
    let (Some(micro), Some(accum), Some(max), Some(ckpt)) = (micro, accum, max, ckpt) else {
        let bad = if micro.is_none() {
            py_repr(&microbatch_size)?
        } else if accum.is_none() {
            py_repr(&accumulation_steps)?
        } else if max.is_none() {
            max_repr
        } else {
            ckpt_repr
        };
        return Err(PyValueError::new_err(format!(
            "microbatch/accumulation/max_updates/checkpoint_frequency_updates must be ints, got {bad}"
        )));
    };
    let err = py.detach(|| sizes_gate_impl(micro, accum, max, ckpt));
    match err {
        Some(text) => Err(PyValueError::new_err(text.to_owned())),
        None => Ok(()),
    }
}

/// Prefetch-depth gate (`loop_state.py:272-279`): bool-rejecting,
/// `1 <= depth <= 16`, oracle-identical text.
#[pyfunction]
#[pyo3(signature = (fetch_prefetch_depth,))]
fn training_prefetch_gate(py: Python<'_>, fetch_prefetch_depth: Bound<'_, PyAny>) -> PyResult<i64> {
    let repr_text = py_repr(&fetch_prefetch_depth)?;
    let is_bool = fetch_prefetch_depth.is_instance_of::<PyBool>();
    let staged: Option<i64> = fetch_prefetch_depth.extract().ok();
    let ok = py.detach(|| {
        !is_bool
            && staged.is_some_and(|v| (TRAINING_PREFETCH_MIN..=TRAINING_PREFETCH_MAX).contains(&v))
    });
    if ok {
        Ok(staged.unwrap_or(TRAINING_PREFETCH_MIN))
    } else {
        Err(PyValueError::new_err(format!(
            "fetch_prefetch_depth must be an int in [1, 16], got {repr_text}"
        )))
    }
}

/// Loop precision enum (`loop_state.py:238-239`), oracle-identical text.
#[pyfunction]
#[pyo3(signature = (precision,))]
fn loop_precision_gate(py: Python<'_>, precision: Bound<'_, PyAny>) -> PyResult<String> {
    let (repr_text, staged) = stage_str(&precision)?;
    let ok = py.detach(|| {
        staged
            .as_deref()
            .is_some_and(|s| TRAINING_LOOP_PRECISIONS.contains(&s))
    });
    if ok {
        Ok(staged.unwrap_or_default())
    } else {
        Err(PyValueError::new_err(format!(
            "precision must be 'fp32' or 'bf16_mixed', got {repr_text}"
        )))
    }
}

/// Replay fp32-only gate (`replay_state.py:140-144`), oracle-identical text.
#[pyfunction]
#[pyo3(signature = (precision,))]
fn replay_precision_gate(py: Python<'_>, precision: Bound<'_, PyAny>) -> PyResult<String> {
    let (repr_text, staged) = stage_str(&precision)?;
    let ok = py.detach(|| staged.as_deref() == Some("fp32"));
    if ok {
        Ok(staged.unwrap_or_default())
    } else {
        Err(PyValueError::new_err(format!(
            "ActorLearnerReplay is fp32-only, got precision {repr_text}; replay never runs bf16 (supervised-bf16/replay-fp32 incomparable)"
        )))
    }
}

/// Replay sha gate (`replay_state.py:54-57`): same loose shape as the loop
/// twin (`loop_state.rs:90-92`), own name per the `loop_state.rs:36-38`
/// claim note. Translators call positionally; the bridge raises
/// `ValueError`, Python maps to `ContractError`.
#[pyfunction]
#[pyo3(signature = (name, value))]
fn replay_require_sha256(
    py: Python<'_>,
    name: Bound<'_, PyAny>,
    value: Bound<'_, PyAny>,
) -> PyResult<String> {
    let name_text = name.str()?.to_str()?.to_owned();
    let repr_text = value.repr()?.to_str()?.to_owned();
    let staged: Option<String> = value.extract().ok();
    let ok = py.detach(|| staged.as_deref().is_some_and(sha_shape_ok));
    if ok {
        match staged {
            Some(text) => Ok(text),
            None => Err(PyValueError::new_err(require_message(
                &name_text, &repr_text,
            ))),
        }
    } else {
        Err(PyValueError::new_err(require_message(
            &name_text, &repr_text,
        )))
    }
}

/// Scheduler closed-form factor: pure projection of `_build_scheduler`
/// (`stream_build.py:209-259`) onto one step int. `param_keys` carries
/// `list(parameters)` so the unknown-key gates stay Rust-side with the
/// oracle-identical text; `t_max`/`end_factor` are `None` when the oracle
/// would default (`main_span` / `final_factor`). NaN factors fail closed.
/// Compute runs detached; tolerance `1e-12` absolute vs torch/`math.cos`.
#[pyfunction]
#[pyo3(signature = (scheduler, step, warmup_updates, max_updates, final_factor, warmup_start_factor, param_keys, t_max, end_factor))]
#[allow(clippy::too_many_arguments)]
fn scheduler_lr_factor(
    py: Python<'_>,
    scheduler: Bound<'_, PyAny>,
    step: Bound<'_, PyAny>,
    warmup_updates: Bound<'_, PyAny>,
    max_updates: Bound<'_, PyAny>,
    final_factor: Bound<'_, PyAny>,
    warmup_start_factor: Bound<'_, PyAny>,
    param_keys: Bound<'_, PyAny>,
    t_max: Bound<'_, PyAny>,
    end_factor: Bound<'_, PyAny>,
) -> PyResult<f64> {
    let (_sched_repr, sched) = stage_str(&scheduler)?;
    let (step_repr, step_val) = stage_int(&step)?;
    let (warm_repr, warm_val) = stage_int(&warmup_updates)?;
    let (horizon_repr, horizon_val) = stage_int(&max_updates)?;
    let (final_repr, final_val) = (py_repr(&final_factor)?, final_factor.extract::<f64>().ok());
    let (wstart_repr, wstart_val) = (
        py_repr(&warmup_start_factor)?,
        warmup_start_factor.extract::<f64>().ok(),
    );
    let keys_repr = py_repr(&param_keys)?;
    let staged_keys: Option<Vec<String>> = param_keys.extract().ok();
    let (tmax_repr, tmax_val) = (py_repr(&t_max)?, t_max.extract::<i64>().ok());
    let tmax_is_none = t_max.is_none();
    let (end_repr, end_val) = (py_repr(&end_factor)?, end_factor.extract::<f64>().ok());
    let end_is_none = end_factor.is_none();
    // Attached validation that needs Python-rendered reprs; the closed-form
    // math itself runs detached below.
    let sched_text = match sched {
        Some(s) if ["cosine", "constant", "linear"].contains(&s.as_str()) => s,
        _ => {
            return Err(PyValueError::new_err(
                "scheduler.id must be one of ['cosine', 'constant', 'linear']".to_owned(),
            ));
        }
    };
    let (Some(step), Some(warm), Some(horizon)) = (step_val, warm_val, horizon_val) else {
        let bad = if step_val.is_none() {
            step_repr
        } else if warm_val.is_none() {
            warm_repr
        } else {
            horizon_repr
        };
        return Err(PyValueError::new_err(format!(
            "scheduler step/warmup_updates/max_updates must be ints, got {bad}"
        )));
    };
    if step < 0 {
        return Err(PyValueError::new_err(format!(
            "scheduler step must be a non-negative int, got {step_repr}"
        )));
    }
    if warm < 0 {
        return Err(PyValueError::new_err(format!(
            "scheduler warmup_updates must be a non-negative int, got {warm_repr}"
        )));
    }
    if horizon <= 0 {
        return Err(PyValueError::new_err(format!(
            "scheduler max_updates must be a positive int, got {horizon_repr}"
        )));
    }
    let (Some(final_factor), Some(warmup_start)) = (final_val, wstart_val) else {
        let bad = if final_val.is_none() {
            final_repr
        } else {
            wstart_repr
        };
        return Err(PyValueError::new_err(format!(
            "scheduler final_factor/warmup_start_factor must be numbers, got {bad}"
        )));
    };
    if !final_factor.is_finite() || !warmup_start.is_finite() {
        return Err(PyValueError::new_err(
            "scheduler final_factor/warmup_start_factor must be finite".to_owned(),
        ));
    }
    let keys = match staged_keys {
        Some(keys) => keys,
        None => {
            return Err(PyValueError::new_err(format!(
                "scheduler.parameters must be a string-key list, got {keys_repr}"
            )));
        }
    };
    let mut unknown: Vec<String> = match sched_text.as_str() {
        "cosine" => keys.into_iter().filter(|k| k != "T_max").collect(),
        "linear" => keys.into_iter().filter(|k| k != "end_factor").collect(),
        _ => keys,
    };
    unknown.sort();
    if !unknown.is_empty() {
        return Err(PyValueError::new_err(format!(
            "scheduler.parameters unknown keys {}",
            py_str_list_repr(&unknown)
        )));
    }
    // `T_max` is read only by the cosine branch (`parameters.get("T_max",
    // main_span)` at `stream_build.py:234`) and `end_factor` only by the
    // linear branch (`stream_build.py:246`); other schedulers ignore them
    // exactly like the oracle.
    if sched_text == "cosine" {
        if !tmax_is_none && tmax_val.is_none() {
            return Err(PyValueError::new_err(format!(
                "scheduler T_max must be a positive int or none, got {tmax_repr}"
            )));
        }
        if let Some(t) = tmax_val
            && t <= 0
        {
            return Err(PyValueError::new_err(format!(
                "scheduler T_max must be a positive int or none, got {tmax_repr}"
            )));
        }
    }
    if sched_text == "linear" {
        if !end_is_none && end_val.is_none() {
            return Err(PyValueError::new_err(format!(
                "scheduler end_factor must lie in [0, 1], got {end_repr}"
            )));
        }
        if let Some(e) = end_val
            && !(0.0..=1.0).contains(&e)
        {
            return Err(PyValueError::new_err(format!(
                "scheduler end_factor must lie in [0, 1], got {}",
                py_float_str(e)
            )));
        }
    }
    let span = if warm < horizon {
        (horizon - warm).max(1)
    } else {
        horizon
    };
    let t_max_resolved = tmax_val.unwrap_or(span);
    let end_resolved = end_val.unwrap_or(final_factor);
    let owned_name = sched_text.clone();
    let factor = py.detach(|| {
        lr_factor_impl(
            &owned_name,
            step,
            warm,
            horizon,
            final_factor,
            warmup_start,
            span,
            t_max_resolved,
            end_resolved,
        )
    });
    if !factor.is_finite() {
        return Err(PyValueError::new_err(
            "scheduler lr factor is non-finite".to_owned(),
        ));
    }
    Ok(factor)
}

/// Prefix-hash filename (`stream_build.py:372-374`:
/// `f"prefix-hashes-{update:06d}.txt"`; Rust `{:06}` pads identically,
/// including the `-00001` negative shape).
#[pyfunction]
#[pyo3(signature = (update,))]
fn prefix_hashes_name(py: Python<'_>, update: Bound<'_, PyAny>) -> PyResult<String> {
    let (repr_text, staged) = stage_int(&update)?;
    let Some(value) = staged else {
        return Err(PyValueError::new_err(format!(
            "prefix update must be an int, got {repr_text}"
        )));
    };
    Ok(py.detach(|| format!("prefix-hashes-{value:06}.txt")))
}

/// Backward-shim derivation (`stream_resume.py:712-721`): compiled non-fp32
/// -> `"off"`, every other pair -> `None`. Non-`str` inputs compare
/// unequal exactly like the oracle `!=`.
#[pyfunction]
#[pyo3(signature = (precision, compile_mode))]
fn backward_autocast_for(
    py: Python<'_>,
    precision: Bound<'_, PyAny>,
    compile_mode: Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let (_, precision_text) = stage_str(&precision)?;
    let (_, compile_text) = stage_str(&compile_mode)?;
    Ok(py.detach(|| {
        if precision_text.as_deref() != Some("fp32") && compile_text.as_deref() != Some("eager") {
            Some("off".to_owned())
        } else {
            None
        }
    }))
}

/// Replay-backend gate (`stream_expand.py:106-112`), oracle-identical text.
#[pyfunction]
#[pyo3(signature = (backend,))]
fn require_replay_backend(py: Python<'_>, backend: Bound<'_, PyAny>) -> PyResult<String> {
    let (repr_text, staged) = stage_str(&backend)?;
    let ok = py.detach(|| {
        staged
            .as_deref()
            .is_some_and(|s| TRAINING_REPLAY_BACKENDS.contains(&s))
    });
    if ok {
        Ok(staged.unwrap_or_default())
    } else {
        Err(PyValueError::new_err(format!(
            "replay_backend must be 'rust', got {repr_text}"
        )))
    }
}

/// Plane-suffix gate (`shard_reader.py:105-110`): ordered match over the
/// lowercased filename, oracle-identical error with Python-repr filename.
#[pyfunction]
#[pyo3(signature = (filename,))]
fn plane_suffix(py: Python<'_>, filename: Bound<'_, PyAny>) -> PyResult<String> {
    let (repr_text, staged) = stage_str(&filename)?;
    let Some(name) = staged else {
        return Err(PyValueError::new_err(format!(
            "plane file {repr_text} has unknown storage suffix"
        )));
    };
    let lowered = name.to_lowercase();
    let hit = py.detach(|| {
        TRAINING_PLANE_SUFFIXES
            .iter()
            .find(|suffix| lowered.ends_with(*suffix))
            .copied()
    });
    match hit {
        Some(suffix) => Ok(suffix.to_owned()),
        None => Err(PyValueError::new_err(format!(
            "plane file {} has unknown storage suffix",
            py_str_repr(&name)
        ))),
    }
}

/// Register the training-config leaves on the shared `contracts` submodule
/// (mirrors `loop_state.rs:132-144`): `let py = sub.py();` (per
/// `contracts.rs:1115`), fns via `wrap_pyfunction!(f, sub)` (per
/// `belief_leaves.rs:194`), consts via `sub.add` (per
/// `contracts.rs:1153-1156`); single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(training_minibatch_size, sub)?)?;
    sub.add_function(wrap_pyfunction!(training_sizes_gate, sub)?)?;
    sub.add_function(wrap_pyfunction!(training_prefetch_gate, sub)?)?;
    sub.add_function(wrap_pyfunction!(loop_precision_gate, sub)?)?;
    sub.add_function(wrap_pyfunction!(replay_precision_gate, sub)?)?;
    sub.add_function(wrap_pyfunction!(replay_require_sha256, sub)?)?;
    sub.add_function(wrap_pyfunction!(scheduler_lr_factor, sub)?)?;
    sub.add_function(wrap_pyfunction!(prefix_hashes_name, sub)?)?;
    sub.add_function(wrap_pyfunction!(backward_autocast_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(require_replay_backend, sub)?)?;
    sub.add_function(wrap_pyfunction!(plane_suffix, sub)?)?;
    sub.add(
        "TRAINING_REPLAY_BACKENDS",
        PyTuple::new(py, TRAINING_REPLAY_BACKENDS)?,
    )?;
    sub.add(
        "TRAINING_PLANE_SUFFIXES",
        PyTuple::new(py, TRAINING_PLANE_SUFFIXES)?,
    )?;
    sub.add("TRAINING_SINGLETON_WORLD", TRAINING_SINGLETON_WORLD)?;
    sub.add("TRAINING_SINGLETON_RANK", TRAINING_SINGLETON_RANK)?;
    sub.add("TRAINING_PREFETCH_MIN", TRAINING_PREFETCH_MIN)?;
    sub.add("TRAINING_PREFETCH_MAX", TRAINING_PREFETCH_MAX)?;
    sub.add(
        "TRAINING_LOOP_PRECISIONS",
        PyTuple::new(py, TRAINING_LOOP_PRECISIONS)?,
    )?;
    sub.add(
        "REPLAY_FORBIDDEN_BATCH_KEYS",
        pyo3::types::PyFrozenSet::new(py, REPLAY_FORBIDDEN_BATCH_KEYS)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Hand-derived from the oracle literals; any drift fails here first.
        assert_eq!(TRAINING_REPLAY_BACKENDS, ["rust"]);
        assert_eq!(
            TRAINING_PLANE_SUFFIXES,
            [
                ".arrow.ipc",
                ".ipc",
                ".arrow",
                ".tensor",
                ".npy",
                ".bin",
                ".raw",
                ".dat"
            ]
        );
        assert_eq!(TRAINING_SINGLETON_WORLD, 1);
        assert_eq!(TRAINING_SINGLETON_RANK, 0);
        assert_eq!((TRAINING_PREFETCH_MIN, TRAINING_PREFETCH_MAX), (1, 16));
        assert_eq!(TRAINING_LOOP_PRECISIONS, ["fp32", "bf16_mixed"]);
        assert_eq!(
            REPLAY_FORBIDDEN_BATCH_KEYS,
            [
                "hidden_tiles",
                "wall",
                "dead_wall",
                "opponent_hand",
                "privileged",
                "full_world",
                "privileged_label",
                "wall_remaining",
                "hidden",
                "privileged_labels",
                "opponent_hidden",
            ]
        );
    }

    #[test]
    fn minibatch_and_sizes_match_oracle() {
        assert_eq!(minibatch_impl(4, 8), Some(32));
        assert_eq!(minibatch_impl(1, 1), Some(1));
        assert_eq!(sizes_gate_impl(4, 8, 10, 5), None);
        assert_eq!(
            sizes_gate_impl(0, 8, 10, 5),
            Some("microbatch/accumulation/max_updates must be positive")
        );
        assert_eq!(
            sizes_gate_impl(4, 8, 10, 0),
            Some("checkpoint_frequency_updates must be positive")
        );
    }

    #[test]
    fn sha_shape_ok_matches_loose_oracle_gate() {
        // Same loose gate as `loop_state.rs:203-217`: uppercase-hex passes.
        assert!(sha_shape_ok(&format!("sha256:{}", "a".repeat(64))));
        assert!(sha_shape_ok(&format!("sha256:{}", "Z".repeat(64))));
        assert!(!sha_shape_ok("sha256:abc"));
        assert!(!sha_shape_ok(&format!("sha256:{}", "a".repeat(63))));
        assert_eq!(
            require_message("run_spec_hash", "'abc'"),
            "run_spec_hash must be sha256:<64 hex>, got 'abc'"
        );
    }

    #[test]
    fn lr_closed_forms_match_torch_shapes() {
        // Warmup ramp endpoints: step 0 rests at warmup_start, the warmup
        // boundary hands 1.0 to the decay tail.
        assert!((lr_factor_impl("cosine", 0, 5, 10, 0.0, 0.1, 5, 5, 0.0) - 0.1).abs() < 1e-12);
        assert!((lr_factor_impl("cosine", 5, 5, 10, 0.0, 0.1, 5, 5, 0.0) - 1.0).abs() < 1e-12);
        // Cosine tail: midpoint is the half-cosine (0.5), horizon rests at final.
        assert!((cosine_decay(0, 4, 0.0) - 1.0).abs() < 1e-12);
        assert!((cosine_decay(2, 4, 0.0) - 0.5).abs() < 1e-12);
        assert!((cosine_decay(4, 4, 0.0) - 0.0).abs() < 1e-12);
        assert!((cosine_decay(9, 4, 0.0) - 0.0).abs() < 1e-12);
        // Linear tail endpoints and constant unity.
        assert!((linear_ramp(0, 1.0, 0.2, 5) - 1.0).abs() < 1e-12);
        assert!((linear_ramp(5, 1.0, 0.2, 5) - 0.2).abs() < 1e-12);
        assert!((lr_factor_impl("constant", 7, 0, 10, 0.0, 1.0, 10, 10, 0.0) - 1.0).abs() < 1e-12);
        // Whole-horizon warmup: single ramp, no decay tail.
        assert!((lr_factor_impl("linear", 10, 10, 10, 0.0, 0.5, 10, 10, 0.0) - 1.0).abs() < 1e-12);
        // Prefix render incl. the negative `-00001` shape.
        assert_eq!(
            format!("prefix-hashes-{:06}.txt", 7),
            "prefix-hashes-000007.txt"
        );
        assert_eq!(
            format!("prefix-hashes-{:06}.txt", -1),
            "prefix-hashes--00001.txt"
        );
    }

    #[test]
    fn py_render_helpers_match_python_repr() {
        assert_eq!(py_str_repr("a/b.ipc"), "'a/b.ipc'");
        assert_eq!(
            py_str_list_repr(&["b".to_owned(), "a".to_owned()]),
            "['b', 'a']"
        );
        assert_eq!(py_float_str(2.0), "2.0");
        assert_eq!(py_float_str(1.5), "1.5");
        assert_eq!(py_float_str(f64::NAN), "nan");
    }
}
