//! Python bindings for the Hydra Riichi Mahjong game engine.

use pyo3::prelude::*;

mod env;

#[pymodule]
fn _hydra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<env::HydraEnv>()?;
    m.add_class::<env::HydraVectorEnv>()?;
    Ok(())
}
