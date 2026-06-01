use anyhow::{Result, bail};
use riichienv_core::action::{Action, ActionType};

use hydra_runtime_types::tile::{AKA_MANZU_136, AKA_PINZU_136, AKA_SOUZU_136};

use super::{
    AGARI, AKA_5M, AKA_5P, AKA_5S, CHI_LEFT, CHI_MID, CHI_RIGHT, HydraAction, KAN, PASS, PON,
    RIICHI, RYUUKYOKU,
};

/// Convert a riichienv Action to a HydraAction.
///
/// Uses 136-format tile IDs from the Action to determine the correct Hydra
/// action index. Aka tiles (136-indices 16, 52, 88) map to Hydra 34-36.
pub fn riichienv_to_hydra(action: &Action) -> Result<HydraAction> {
    let id = match action.action_type {
        ActionType::Discard => {
            let tile = action
                .tile
                .ok_or_else(|| anyhow::anyhow!("Discard action missing tile"))?;
            // Check if this is an aka tile in 136-format
            match tile {
                AKA_MANZU_136 => AKA_5M,
                AKA_PINZU_136 => AKA_5P,
                AKA_SOUZU_136 => AKA_5S,
                _ => tile / 4, // 136-format -> 34-format tile type
            }
        }
        ActionType::Riichi => RIICHI,
        ActionType::Chi => {
            // Determine chi variant from called tile position among sorted tiles
            let target = action
                .tile
                .ok_or_else(|| anyhow::anyhow!("Chi action missing target tile"))?;
            let target_34 = target / 4;
            let slice = action.consume_slice();
            let mut tiles_34 = [0u8; 4];
            for (i, &t) in slice.iter().enumerate() {
                tiles_34[i] = t / 4;
            }
            let tile_count = slice.len();
            tiles_34[tile_count] = target_34;
            let total = tile_count + 1;
            let used = &mut tiles_34[..total];
            used.sort();
            if target_34 == used[0] {
                CHI_LEFT // called tile is lowest
            } else if target_34 == used[1] {
                CHI_MID // called tile is middle
            } else {
                CHI_RIGHT // called tile is highest
            }
        }
        ActionType::Pon => PON,
        ActionType::Daiminkan | ActionType::Ankan | ActionType::Kakan => KAN,
        ActionType::Ron | ActionType::Tsumo => AGARI,
        ActionType::KyushuKyuhai => RYUUKYOKU,
        ActionType::Pass => PASS,
        ActionType::Kita => bail!("Kita not supported in Hydra 4-player action space"),
    };
    HydraAction::new(id).ok_or_else(|| anyhow::anyhow!("computed invalid HydraAction id: {id}"))
}
