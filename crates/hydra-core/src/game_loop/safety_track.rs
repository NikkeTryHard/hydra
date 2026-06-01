use crate::safety::bit_test;
use riichienv_core::action::{Action, ActionType};

use super::runner::GameRunner;

impl GameRunner {
    /// Update safety info when an action is taken.
    pub(super) fn track_action(&mut self, actor: u8, action: &Action) {
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile136) = action.tile {
                    let tile_type = tile136 / 4;
                    // Tedashi = discarded from hand (not the just-drawn tile).
                    // drawn_tile is still set here because track_action runs
                    // BEFORE state.step() clears it.
                    let is_tsumogiri = self.state.drawn_tile == Some(tile136);
                    let is_tedashi = !is_tsumogiri;
                    // Update safety from each OTHER player's perspective
                    for observer in 0..4u8 {
                        if observer == actor {
                            continue;
                        }
                        // Relative-opponent slots are ordered as left-to-right opponents
                        // from the observer's perspective: [observer+1, +2, +3] mod 4.
                        let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                        if opp_idx < 3 {
                            let safety = &mut self.safety[observer as usize];
                            let already_visible =
                                bit_test(safety.genbutsu_all[opp_idx], tile_type as usize);
                            safety.on_discard(tile_type, opp_idx, is_tedashi);
                            if already_visible {
                                safety.visible_counts[tile_type as usize] =
                                    safety.visible_counts[tile_type as usize].saturating_sub(1);
                            }
                        }
                    }
                }
            }
            ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                let mut tile_types = [0u8; 4];
                let count = action.consume_count as usize;
                for (i, &t) in action.consume_slice().iter().enumerate() {
                    tile_types[i] = t / 4;
                }
                if count > 0 {
                    for s in &mut self.safety {
                        s.on_call(&tile_types[..count]);
                    }
                }
            }
            ActionType::Riichi => {
                for observer in 0..4u8 {
                    if observer == actor {
                        continue;
                    }
                    let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                    if opp_idx < 3 {
                        self.safety[observer as usize].on_riichi(opp_idx);
                    }
                }
            }
            _ => {}
        }
    }
}
