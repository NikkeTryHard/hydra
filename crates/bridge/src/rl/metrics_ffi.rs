//! Trace reduction stays native; Python owns the Torch profiler because
//! moving tensor execution into a callback would alter the measured surface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn analyze_trace<'py>(py: Python<'py>, trace_path: &str, scope: &str) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = std::fs::read(trace_path).map_err(|e| PyValueError::new_err(format!("read CUDA trace: {e}")))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| PyValueError::new_err(format!("invalid CUDA trace JSON: {e}")))?;
    let result = super::metrics::analyze_trace(&document, scope).map_err(PyValueError::new_err)?;
    let encoded = hydra_feed::canon::canonical_bytes(&result, "rl-device-timeline")
        .map_err(|e| PyValueError::new_err(format!("CUDA metric identity: {e}")))?;
    Ok(PyBytes::new(py, &encoded))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_trace, module)?)?;
    Ok(())
}
