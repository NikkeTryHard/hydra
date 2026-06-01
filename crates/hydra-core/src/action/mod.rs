//! Hydra 46-action space mapping, Mortal-compatible.
//!
//! Maps between Hydra's compact 46-action representation and
//! riichienv-core's ActionType/Action structs.

mod context;
mod from_riichienv;
mod legal_mask;
mod to_riichienv;

pub use context::{ActionPhase, GameContext};
pub use from_riichienv::riichienv_to_hydra;
pub use hydra_runtime_types::action::{
    AGARI, AKA_5M, AKA_5P, AKA_5S, CHI_LEFT, CHI_MID, CHI_RIGHT, DISCARD_END, DISCARD_START,
    HYDRA_ACTION_SPACE, HydraAction, KAN, PASS, PON, RIICHI, RYUUKYOKU,
};
pub use legal_mask::build_legal_mask;
pub use to_riichienv::hydra_to_riichienv;

#[cfg(test)]
use hydra_runtime_types::tile::AKA_MANZU_136;
#[cfg(test)]
use riichienv_core::action::{Action, ActionType};

#[cfg(test)]
mod tests;
