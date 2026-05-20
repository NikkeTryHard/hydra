#[macro_use]
pub mod mjai_event;
mod agari;
pub mod errors;
pub mod hand_evaluator;
pub mod hand_evaluator_3p;
pub mod score;
#[cfg(test)]
pub(crate) mod test_support {
    pub(crate) fn parsed_tile(text: &str) -> u8 {
        crate::parser::parse_tile(text).expect("test tile should parse")
    }

    pub(crate) fn tiles_to_u32(tiles: &[u8]) -> Vec<u32> {
        tiles.iter().copied().map(u32::from).collect()
    }
}
mod tests;
pub mod types;
pub mod yaku;
mod yaku_3p;

pub mod action;
pub mod game_variant;
pub mod observation;
pub mod observation_3p;
pub mod observation_ref;
pub mod parser;
pub mod replay;
pub mod rule;
pub mod shanten;
pub mod state;
pub mod state_3p;

pub use hand_evaluator::check_riichi_candidates;
