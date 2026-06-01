use riichienv_core::action::Action;

use super::context::ActionPhase;
use super::{HYDRA_ACTION_SPACE, riichienv_to_hydra};

/// Builds a boolean mask over the 46-action space from riichienv legal actions.
///
/// Each entry in the returned array is `true` if the corresponding Hydra
/// action index is legal. Actions that fail conversion (e.g. Kita) are skipped.
pub fn build_legal_mask(
    legal_actions: &[Action],
    phase: ActionPhase,
) -> [bool; HYDRA_ACTION_SPACE] {
    let mut mask = [false; HYDRA_ACTION_SPACE];
    for action in legal_actions {
        if let Ok(hydra) = riichienv_to_hydra(action) {
            let idx = hydra.id() as usize;
            if idx >= HYDRA_ACTION_SPACE {
                continue;
            }
            match phase {
                ActionPhase::Normal => {
                    mask[idx] = true;
                }
                ActionPhase::RiichiSelect | ActionPhase::KanSelect => {
                    if hydra.is_discard() {
                        mask[idx] = true;
                    }
                }
            }
        }
    }
    mask
}
