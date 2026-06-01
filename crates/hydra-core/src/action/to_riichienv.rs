use anyhow::{Result, bail};
use riichienv_core::action::{Action, ActionType};

use hydra_runtime_types::tile::{AKA_MANZU_136, AKA_PINZU_136, AKA_SOUZU_136};

use super::HydraAction;
use super::context::{ActionPhase, GameContext};

/// Find a 136-format tile of the given type in a hand.
fn find_tile_in_hand(hand: &[u8], tile_type: u8) -> Result<u8> {
    hand.iter()
        .find(|&&t| t / 4 == tile_type)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tile type {} not in hand", tile_type))
}

fn chi_consume_type(called_type: u8, offset: i8) -> Result<u8> {
    if called_type >= 27 {
        bail!("chi requires suited tile, got tile type {called_type}");
    }
    let consume_type = i16::from(called_type) + i16::from(offset);
    if !(0..=26).contains(&consume_type) || consume_type / 9 != i16::from(called_type / 9) {
        bail!("chi variant out of suit range: called tile type {called_type}, offset {offset}");
    }
    Ok(consume_type as u8)
}

/// Convert a HydraAction to a riichienv Action using game context.
///
/// For discard actions, converts tile type (0-33) to 136-format. For 5m/5p/5s
/// (types 4,13,22), uses copy 1 to avoid the aka slot. Aka discards (34-36)
/// use the known aka 136-indices directly.
///
/// Chi, kan, and agari actions use the `GameContext` to resolve the full
/// action details (consume tiles, kan type, tsumo vs ron).
pub fn hydra_to_riichienv(hydra: HydraAction, ctx: &GameContext) -> Result<Action> {
    let id = hydra.id();
    match id {
        // Normal discards: tile type 0-33 -> 136-format = type * 4 + copy
        // For 5m/5p/5s (types 4,13,22), copy 0 is the aka tile.
        // Use copy 1 for normal discards to avoid the collision.
        0..=33 => {
            let copy = if matches!(id, 4 | 13 | 22) { 1 } else { 0 };
            Ok(Action::new(
                ActionType::Discard,
                Some(id * 4 + copy),
                &[],
                None,
            ))
        }
        // Aka 5m discard
        34 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_MANZU_136),
            &[],
            None,
        )),
        // Aka 5p discard
        35 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_PINZU_136),
            &[],
            None,
        )),
        // Aka 5s discard
        36 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_SOUZU_136),
            &[],
            None,
        )),
        // Riichi declaration (tile selection is a separate phase)
        37 => Ok(Action::new(ActionType::Riichi, None, &[], None)),
        // Chi variants -- resolved using last_discard and hand from context
        38..=40 => {
            let called = ctx
                .last_discard
                .ok_or_else(|| anyhow::anyhow!("chi requires last_discard"))?;
            let called_type = called / 4;
            let (offset_a, offset_b) = match id {
                38 => (1i8, 2i8),  // left: called is lowest
                39 => (-1i8, 1i8), // mid: called is middle
                _ => (-2i8, -1i8), // right: called is highest
            };
            let type_a = chi_consume_type(called_type, offset_a)?;
            let type_b = chi_consume_type(called_type, offset_b)?;
            let tile_a = find_tile_in_hand(&ctx.hand[..ctx.hand_len as usize], type_a)?;
            let tile_b = find_tile_in_hand(&ctx.hand[..ctx.hand_len as usize], type_b)?;
            Ok(Action::new(
                ActionType::Chi,
                Some(called),
                &[tile_a, tile_b],
                None,
            ))
        }
        // Pon
        41 => Ok(Action::new(ActionType::Pon, None, &[], None)),
        // Kan -- resolved from game phase
        42 => {
            let action_type = match ctx.phase {
                ActionPhase::Normal => ActionType::Ankan,
                _ => ActionType::Daiminkan,
            };
            Ok(Action::new(action_type, None, &[], None))
        }
        // Agari -- tsumo during own turn, ron during response
        43 => {
            let action_type = match ctx.phase {
                ActionPhase::Normal => ActionType::Tsumo,
                _ => ActionType::Ron,
            };
            Ok(Action::new(action_type, None, &[], None))
        }
        // Kyushu kyuhai (abortive draw)
        44 => Ok(Action::new(ActionType::KyushuKyuhai, None, &[], None)),
        // Pass
        45 => Ok(Action::new(ActionType::Pass, None, &[], None)),
        _ => bail!("invalid HydraAction id: {id}"),
    }
}
