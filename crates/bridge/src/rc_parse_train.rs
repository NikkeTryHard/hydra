//! Run-config training-section parsers (split from `rc_parse` to hold the gate).
//!
//! Owns weights, optimizer, scheduler, and runtime sections plus their staged
//! and valid records. Staging helpers (`stage_*`, `check_*`, `section_dict`,
//! `field_slot`) stay in `crate::rc_parse`; this file imports them one way
//! with no new dependency. Failure mode is `String` reject with field path,
//! never silent default; unknown keys reject via `stage_unknown`.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use crate::rc_parse::{
    ADAPTER_IDS, COMPILE_MODES, FloatEnds, HugeInt, IntSlot, LABEL_SMOOTHING_ENDS, MapStage,
    NumView, OPTIMIZER_IDS, OptStr, PinStage, RUNTIME_PRECISIONS, SCHEDULER_IDS, UnknownView,
    WEIGHTS_ALLOWED, check_bounded_float, check_nonnegative_float, check_nonnegative_int,
    check_pin, check_unknown, check_weight_map, cuda_device_ok, field_slot, one_of_list, py_repr,
    section_dict, stage_int_slot, stage_map, stage_num, stage_opt_str, stage_pin, stage_unknown,
};
struct WeightsStaged {
    unknown: UnknownView,
    w_policy: Option<NumView>,
    w_placement: Option<NumView>,
    w_value: Option<NumView>,
    w_event: MapStage,
    w_belief: MapStage,
    privileged_source_hash: PinStage,
    label_smoothing_present: bool,
    label_smoothing_is_none: bool,
    label_smoothing: NumView,
}

struct WeightsValid {
    w_policy: f64,
    w_placement: f64,
    w_value: f64,
    w_event: Option<Vec<(String, f64)>>,
    w_belief: Option<Vec<(String, f64)>>,
    privileged_source_hash: Option<String>,
    label_smoothing: f64,
}

/// `_parse_weights` (oracle `_rc_parse.py:196-230`): objective weights with
/// gate-evaluated defaults in constructor order.
#[pyfunction]
pub(crate) fn rc_parse_weights(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "weights")?;
    let unknown = stage_unknown(&dict, &WEIGHTS_ALLOWED, "weights")?;
    let w_policy = stage_opt_num(field_slot(&dict, "w_policy", "rc_parse_weights")?)?;
    let w_placement = stage_opt_num(field_slot(&dict, "w_placement", "rc_parse_weights")?)?;
    let w_value = stage_opt_num(field_slot(&dict, "w_value", "rc_parse_weights")?)?;
    let w_event = stage_map(field_slot(&dict, "w_event", "rc_parse_weights")?)?;
    let w_belief = stage_map(field_slot(&dict, "w_belief", "rc_parse_weights")?)?;
    let privileged_source_hash = stage_pin(field_slot(
        &dict,
        "privileged_source_hash",
        "rc_parse_weights",
    )?)?;
    // Oracle: present-and-not-None gates, else the `0.03` default.
    let label_slot = field_slot(&dict, "label_smoothing", "rc_parse_weights")?;
    let (label_smoothing_present, label_smoothing_is_none, label_smoothing): (bool, bool, NumView) =
        match label_slot {
            None => (
                false,
                false,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) if obj.is_none() => (
                true,
                true,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) => (true, false, stage_num(&obj)?),
        };
    let staged = WeightsStaged {
        unknown,
        w_policy,
        w_placement,
        w_value,
        w_event,
        w_belief,
        privileged_source_hash,
        label_smoothing_present,
        label_smoothing_is_none,
        label_smoothing,
    };
    let valid = py
        .detach(|| validate_weights(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("w_policy", valid.w_policy)?;
    out.set_item("w_placement", valid.w_placement)?;
    out.set_item("w_value", valid.w_value)?;
    set_weight_map(&out, "w_event", valid.w_event.as_ref())?;
    set_weight_map(&out, "w_belief", valid.w_belief.as_ref())?;
    out.set_item("privileged_source_hash", valid.privileged_source_hash)?;
    out.set_item("label_smoothing", valid.label_smoothing)?;
    Ok(out.unbind())
}

/// Stage an optional gated float: `None` when absent (caller substitutes the
/// default), the number view when present (including present-`None`, which the
/// oracle gates — and therefore rejects — except where noted).
fn stage_opt_num(slot: Option<Bound<'_, PyAny>>) -> PyResult<Option<NumView>> {
    match slot {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(Some(NumView {
            repr: "None".to_owned(),
            number: None,
        })),
        Some(obj) => Ok(Some(stage_num(&obj)?)),
    }
}

fn set_weight_map(
    out: &Bound<'_, PyDict>,
    key: &str,
    pairs: Option<&Vec<(String, f64)>>,
) -> PyResult<()> {
    match pairs {
        None => {
            let none = out.py().None();
            out.set_item(key, &none)?;
        }
        Some(entries) => {
            let dict = PyDict::new(out.py());
            for (head, number) in entries {
                dict.set_item(head, *number)?;
            }
            out.set_item(key, dict)?;
        }
    }
    Ok(())
}

fn validate_weights(staged: &WeightsStaged) -> Result<WeightsValid, String> {
    check_unknown(&staged.unknown)?;
    let w_policy = match &staged.w_policy {
        None => 1.0,
        Some(view) => check_nonnegative_float("weights", "w_policy", view)?,
    };
    let w_placement = match &staged.w_placement {
        None => 0.0,
        Some(view) => check_nonnegative_float("weights", "w_placement", view)?,
    };
    let w_value = match &staged.w_value {
        None => 0.0,
        Some(view) => check_nonnegative_float("weights", "w_value", view)?,
    };
    let w_event = match &staged.w_event.entries {
        None => {
            if staged.w_event.type_name.is_some() {
                return Err(format!(
                    "weights.w_event must be a mapping or null, got {}",
                    staged.w_event.type_name.clone().unwrap_or_default()
                ));
            }
            None
        }
        Some(entries) => Some(check_weight_map("weights", "w_event", entries)?),
    };
    let w_belief = match &staged.w_belief.entries {
        None => {
            if staged.w_belief.type_name.is_some() {
                return Err(format!(
                    "weights.w_belief must be a mapping or null, got {}",
                    staged.w_belief.type_name.clone().unwrap_or_default()
                ));
            }
            None
        }
        Some(entries) => Some(check_weight_map("weights", "w_belief", entries)?),
    };
    let privileged_source_hash = check_pin(
        "weights",
        "privileged_source_hash",
        &staged.privileged_source_hash,
    )?;
    let label_smoothing = if staged.label_smoothing_present && !staged.label_smoothing_is_none {
        check_bounded_float(
            "weights",
            "label_smoothing",
            &staged.label_smoothing,
            &LABEL_SMOOTHING_ENDS,
        )?
    } else {
        0.03
    };
    Ok(WeightsValid {
        w_policy,
        w_placement,
        w_value,
        w_event,
        w_belief,
        privileged_source_hash,
        label_smoothing,
    })
}

// ---------------------------------------------------------------------------
// optimizer
// ---------------------------------------------------------------------------

pub(crate) const OPTIMIZER_ALLOWED: [&str; 5] =
    ["id", "lr", "betas", "weight_decay", "head_lr_mult"];

struct OptimizerStaged {
    unknown: UnknownView,
    name: OptStr,
    lr_repr: String,
    lr: Option<f64>,
    betas_repr: String,
    betas_is_pair: bool,
    betas: Option<(f64, f64)>,
    betas_tuple_repr: String,
    weight_decay_repr: String,
    weight_decay: Option<f64>,
    head_lr_mult_repr: String,
    head_lr_mult: Option<f64>,
}

struct OptimizerValid {
    name: String,
    lr: f64,
    betas: (f64, f64),
    weight_decay: f64,
    head_lr_mult: f64,
}

/// `_parse_optimizer` (oracle `_rc_parse.py:233-282`): registered id plus the
/// numeric battery. `betas` converts each entry with the live `float()`
/// builtin at stage time, so numeric strings, bools, and conversion errors
/// behave exactly like the oracle's `float(betas_raw[i])`.
#[pyfunction]
pub(crate) fn rc_parse_optimizer(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "optimizer")?;
    let unknown = stage_unknown(&dict, &OPTIMIZER_ALLOWED, "optimizer")?;
    let name = stage_opt_str(field_slot(&dict, "id", "rc_parse_optimizer")?, "adamw")?;
    let lr_slot = field_slot(&dict, "lr", "rc_parse_optimizer")?;
    let (lr_repr, lr): (String, Option<f64>) = match lr_slot {
        None => (String::new(), Some(3e-4)),
        Some(obj) => {
            let view = stage_num(&obj)?;
            (view.repr.clone(), view.number)
        }
    };
    // `betas`: 2-element list/tuple gate, then live-`float()` conversion.
    let betas_slot = field_slot(&dict, "betas", "rc_parse_optimizer")?;
    let (betas_repr, betas_is_pair, betas, betas_tuple_repr): (
        String,
        bool,
        Option<(f64, f64)>,
        String,
    ) = match betas_slot {
        None => (String::new(), true, Some((0.9, 0.999)), String::new()),
        Some(obj) => {
            let repr = py_repr(&obj)?;
            if !(obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>())
                || obj.len().unwrap_or(usize::MAX) != 2
            {
                (repr, false, None, String::new())
            } else {
                // Conversion runs through the live `float()` builtin, so numeric
                // strings, bools, and conversion errors (`ValueError` /
                // `OverflowError` / `TypeError`) behave exactly like the oracle's
                // `float(betas_raw[i])`: every error propagates unwrapped.
                let float_type = py.import("builtins")?.getattr("float")?;
                let first: f64 = float_type.call1((obj.get_item(0)?,))?.extract()?;
                let second: f64 = float_type.call1((obj.get_item(1)?,))?.extract()?;
                let tuple_repr = PyTuple::new(py, [first, second])?
                    .repr()?
                    .to_str()?
                    .to_owned();
                (repr, true, Some((first, second)), tuple_repr)
            }
        }
    };
    let decay_slot = field_slot(&dict, "weight_decay", "rc_parse_optimizer")?;
    let (weight_decay_repr, weight_decay): (String, Option<f64>) = match decay_slot {
        None => (String::new(), Some(0.01)),
        Some(obj) => {
            let view = stage_num(&obj)?;
            (view.repr.clone(), view.number)
        }
    };
    let head_slot = field_slot(&dict, "head_lr_mult", "rc_parse_optimizer")?;
    let (head_lr_mult_repr, head_lr_mult): (String, Option<f64>) = match head_slot {
        None => (String::new(), Some(3.0)),
        Some(obj) => {
            let view = stage_num(&obj)?;
            (view.repr.clone(), view.number)
        }
    };
    let staged = OptimizerStaged {
        unknown,
        name,
        lr_repr,
        lr,
        betas_repr,
        betas_is_pair,
        betas,
        betas_tuple_repr,
        weight_decay_repr,
        weight_decay,
        head_lr_mult_repr,
        head_lr_mult,
    };
    let valid = py
        .detach(|| validate_optimizer(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("id", valid.name)?;
    out.set_item("lr", valid.lr)?;
    let betas_list = PyTuple::new(py, [valid.betas.0, valid.betas.1])?;
    out.set_item("betas", betas_list)?;
    out.set_item("weight_decay", valid.weight_decay)?;
    out.set_item("head_lr_mult", valid.head_lr_mult)?;
    Ok(out.unbind())
}

fn validate_optimizer(staged: &OptimizerStaged) -> Result<OptimizerValid, String> {
    check_unknown(&staged.unknown)?;
    match &staged.name.text {
        Some(text) if OPTIMIZER_IDS.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "optimizer.id must be one of {}, got {}",
                one_of_list(&OPTIMIZER_IDS),
                staged.name.repr
            ));
        }
    }
    let Some(lr) = staged.lr else {
        return Err(format!(
            "optimizer.lr must be positive and finite, got {}",
            staged.lr_repr
        ));
    };
    if !lr.is_finite() || lr <= 0.0 {
        return Err(format!(
            "optimizer.lr must be positive and finite, got {}",
            staged.lr_repr
        ));
    }
    if !staged.betas_is_pair {
        return Err(format!(
            "optimizer.betas must be a 2-element list, got {}",
            staged.betas_repr
        ));
    }
    let Some((beta0, beta1)) = staged.betas else {
        return Err(format!(
            "optimizer.betas must be a 2-element list, got {}",
            staged.betas_repr
        ));
    };
    for beta in [beta0, beta1] {
        if !(0.0..1.0).contains(&beta) || beta.is_nan() {
            return Err(format!(
                "optimizer.betas entries must lie in [0, 1), got {}",
                staged.betas_tuple_repr
            ));
        }
    }
    let Some(weight_decay) = staged.weight_decay else {
        return Err(format!(
            "optimizer.weight_decay must be finite non-negative, got {}",
            staged.weight_decay_repr
        ));
    };
    if !weight_decay.is_finite() || weight_decay < 0.0 {
        return Err(format!(
            "optimizer.weight_decay must be finite non-negative, got {}",
            staged.weight_decay_repr
        ));
    }
    let Some(head_lr_mult) = staged.head_lr_mult else {
        return Err(format!(
            "optimizer.head_lr_mult must be positive and finite, got {}",
            staged.head_lr_mult_repr
        ));
    };
    if !head_lr_mult.is_finite() || head_lr_mult <= 0.0 {
        return Err(format!(
            "optimizer.head_lr_mult must be positive and finite, got {}",
            staged.head_lr_mult_repr
        ));
    }
    Ok(OptimizerValid {
        name: staged
            .name
            .text
            .clone()
            .unwrap_or_else(|| "adamw".to_owned()),
        lr,
        betas: (beta0, beta1),
        weight_decay,
        head_lr_mult,
    })
}

// ---------------------------------------------------------------------------
// scheduler
// ---------------------------------------------------------------------------

pub(crate) const SCHEDULER_ALLOWED: [&str; 5] = [
    "id",
    "warmup_updates",
    "parameters",
    "final_factor",
    "warmup_start_factor",
];

pub(crate) const FINAL_FACTOR_ENDS: FloatEnds = FloatEnds {
    lo: 0.0,
    lo_repr: "0.0",
    lo_open: false,
    hi: 1.0,
    hi_repr: "1.0",
    hi_open: false,
};

pub(crate) const WARMUP_START_ENDS: FloatEnds = FloatEnds {
    lo: 0.0,
    lo_repr: "0.0",
    lo_open: true,
    hi: 1.0,
    hi_repr: "1.0",
    hi_open: false,
};

struct SchedulerStaged {
    unknown: UnknownView,
    name: OptStr,
    warmup_updates: IntSlot,
    parameters_present: bool,
    parameters_is_dict: bool,
    parameters: Option<Py<PyDict>>,
    final_factor_present: bool,
    final_factor_is_none: bool,
    final_factor: NumView,
    warmup_start_present: bool,
    warmup_start_is_none: bool,
    warmup_start: NumView,
}

struct SchedulerValid {
    name: String,
    warmup_updates: HugeInt,
    final_factor: f64,
    warmup_start_factor: f64,
}

/// `_parse_scheduler` (oracle `_rc_parse.py:285-314`): registered id plus
/// warmup and factor gates.
#[pyfunction]
pub(crate) fn rc_parse_scheduler(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "scheduler")?;
    let unknown = stage_unknown(&dict, &SCHEDULER_ALLOWED, "scheduler")?;
    let name = stage_opt_str(field_slot(&dict, "id", "rc_parse_scheduler")?, "cosine")?;
    let warmup_updates = stage_int_slot(
        py,
        field_slot(&dict, "warmup_updates", "rc_parse_scheduler")?,
    )?;
    let parameters_slot = field_slot(&dict, "parameters", "rc_parse_scheduler")?;
    let (parameters_present, parameters_is_dict, parameters): (bool, bool, Option<Py<PyDict>>) =
        match parameters_slot {
            None => (false, true, None),
            Some(obj) => match obj.extract::<Bound<'_, PyDict>>() {
                Ok(params) => (true, true, Some(params.unbind())),
                Err(_) => (true, false, None),
            },
        };
    let final_slot = field_slot(&dict, "final_factor", "rc_parse_scheduler")?;
    let (final_factor_present, final_factor_is_none, final_factor): (bool, bool, NumView) =
        match final_slot {
            None => (
                false,
                false,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) if obj.is_none() => (
                true,
                true,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) => (true, false, stage_num(&obj)?),
        };
    let warmup_slot = field_slot(&dict, "warmup_start_factor", "rc_parse_scheduler")?;
    let (warmup_start_present, warmup_start_is_none, warmup_start): (bool, bool, NumView) =
        match warmup_slot {
            None => (
                false,
                false,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) if obj.is_none() => (
                true,
                true,
                NumView {
                    repr: "None".to_owned(),
                    number: None,
                },
            ),
            Some(obj) => (true, false, stage_num(&obj)?),
        };
    let staged = SchedulerStaged {
        unknown,
        name,
        warmup_updates,
        parameters_present,
        parameters_is_dict,
        parameters,
        final_factor_present,
        final_factor_is_none,
        final_factor,
        warmup_start_present,
        warmup_start_is_none,
        warmup_start,
    };
    let valid = py
        .detach(|| validate_scheduler(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("id", valid.name)?;
    match valid.warmup_updates {
        Some(value) => {
            out.set_item("warmup_updates", value)?;
        }
        None => {
            if let Some(handle) = staged.warmup_updates.handle.as_ref() {
                out.set_item("warmup_updates", handle.bind(py))?;
            } else {
                out.set_item("warmup_updates", 100)?;
            }
        }
    }
    let params = PyDict::new(py);
    if let Some(stored) = staged.parameters.as_ref() {
        for (head, value) in stored.bind(py).iter() {
            params.set_item(head, value)?;
        }
    }
    out.set_item("parameters", params)?;
    out.set_item("final_factor", valid.final_factor)?;
    out.set_item("warmup_start_factor", valid.warmup_start_factor)?;
    Ok(out.unbind())
}

fn validate_scheduler(staged: &SchedulerStaged) -> Result<SchedulerValid, String> {
    check_unknown(&staged.unknown)?;
    match &staged.name.text {
        Some(text) if SCHEDULER_IDS.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "scheduler.id must be one of {}, got {}",
                one_of_list(&SCHEDULER_IDS),
                staged.name.repr
            ));
        }
    }
    let warmup_updates: HugeInt = if !staged.warmup_updates.present {
        Some(100)
    } else {
        check_nonnegative_int("scheduler", "warmup_updates", &staged.warmup_updates.view)?;
        staged.warmup_updates.view.as_i64
    };
    if staged.parameters_present && !staged.parameters_is_dict {
        return Err("scheduler.parameters must be a mapping".to_owned());
    }
    let final_factor = if staged.final_factor_present && !staged.final_factor_is_none {
        check_bounded_float(
            "scheduler",
            "final_factor",
            &staged.final_factor,
            &FINAL_FACTOR_ENDS,
        )?
    } else {
        0.0
    };
    let warmup_start_factor = if staged.warmup_start_present && !staged.warmup_start_is_none {
        check_bounded_float(
            "scheduler",
            "warmup_start_factor",
            &staged.warmup_start,
            &WARMUP_START_ENDS,
        )?
    } else {
        0.01
    };
    Ok(SchedulerValid {
        name: staged
            .name
            .text
            .clone()
            .unwrap_or_else(|| "cosine".to_owned()),
        warmup_updates,
        final_factor,
        warmup_start_factor,
    })
}

// ---------------------------------------------------------------------------
// runtime
// ---------------------------------------------------------------------------

pub(crate) const RUNTIME_ALLOWED: [&str; 4] = ["adapter_id", "device", "precision", "compile_mode"];

struct RuntimeStaged {
    unknown: UnknownView,
    adapter_id: OptStr,
    device: OptStr,
    precision: OptStr,
    compile_mode: OptStr,
}

/// `_parse_runtime` (oracle `_rc_parse.py:317-360`): mirrors `RuntimeSpec`.
/// The `validate_runtime_spec` protocol check stays Python (import-guarded
/// patch-point).
#[pyfunction]
pub(crate) fn rc_parse_runtime(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "runtime")?;
    let unknown = stage_unknown(&dict, &RUNTIME_ALLOWED, "runtime")?;
    let adapter_id = stage_opt_str(
        field_slot(&dict, "adapter_id", "rc_parse_runtime")?,
        "plain_pytorch",
    )?;
    let device = stage_opt_str(field_slot(&dict, "device", "rc_parse_runtime")?, "cuda")?;
    let precision = stage_opt_str(field_slot(&dict, "precision", "rc_parse_runtime")?, "fp32")?;
    let compile_mode = stage_opt_str(
        field_slot(&dict, "compile_mode", "rc_parse_runtime")?,
        "eager",
    )?;
    let staged = RuntimeStaged {
        unknown,
        adapter_id,
        device,
        precision,
        compile_mode,
    };
    let valid = py
        .detach(|| validate_runtime(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("adapter_id", valid.0)?;
    out.set_item("device", valid.1)?;
    out.set_item("precision", valid.2)?;
    out.set_item("compile_mode", valid.3)?;
    Ok(out.unbind())
}

fn validate_runtime(staged: &RuntimeStaged) -> Result<(String, String, String, String), String> {
    check_unknown(&staged.unknown)?;
    match &staged.adapter_id.text {
        Some(text) if ADAPTER_IDS.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "runtime.adapter_id must be one of {}, got {}",
                one_of_list(&ADAPTER_IDS),
                staged.adapter_id.repr
            ));
        }
    }
    match &staged.device.text {
        Some(text) if text == "cpu" || cuda_device_ok(text) => {}
        _ => {
            return Err(format!(
                "runtime.device must be 'cpu' or cuda[:N], got {}",
                staged.device.repr
            ));
        }
    }
    match &staged.precision.text {
        Some(text) if RUNTIME_PRECISIONS.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "runtime.precision must be one of {} (fp16 excluded by design), got {}",
                one_of_list(&RUNTIME_PRECISIONS),
                staged.precision.repr
            ));
        }
    }
    match &staged.compile_mode.text {
        Some(text) if COMPILE_MODES.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "runtime.compile_mode must be one of {}, got {}",
                one_of_list(&COMPILE_MODES),
                staged.compile_mode.repr
            ));
        }
    }
    Ok((
        staged
            .adapter_id
            .text
            .clone()
            .unwrap_or_else(|| "plain_pytorch".to_owned()),
        staged
            .device
            .text
            .clone()
            .unwrap_or_else(|| "cuda".to_owned()),
        staged
            .precision
            .text
            .clone()
            .unwrap_or_else(|| "fp32".to_owned()),
        staged
            .compile_mode
            .text
            .clone()
            .unwrap_or_else(|| "eager".to_owned()),
    ))
}

/// Register weights, optimizer, scheduler, and runtime parsers.
pub fn register_train(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_parse_weights, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_optimizer, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_scheduler, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_runtime, sub)?)?;
    Ok(())
}
