//! Hydra3 native batch arena.
//!
//! The extension module exposes one class, [`Arena`], which owns full red
//! four-player hanchan simulations from riichienv-core.
//! Rows are encoded by the shared feature module and returned as four owned
//! little-endian byte arrays, one row per active game.
//!
//! The Python package imports this module as `hydra3._arena` (the crate name
//! and the module name intentionally differ from the reference `_native`).

mod arena;
mod features;
mod view;

use pyo3::prelude::*;

#[pymodule]
fn _arena(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<arena::Arena>()?;
    Ok(())
}
