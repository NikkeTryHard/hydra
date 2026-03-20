#[cfg(feature = "python")]
mod encode;
#[cfg(feature = "python")]
pub(crate) mod helpers;
#[cfg(feature = "python")]
mod python;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::Meld;

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all)
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Player-facing snapshot of the game state at a decision point.
pub struct Observation {
    /// Seat index of the observing player (0-3).
    pub player_id: u8,
    /// Hands for all four players (tile IDs as u32).
    pub hands: [Vec<u32>; 4],
    /// Open melds declared by each player.
    pub melds: [Vec<Meld>; 4],
    /// Discard ponds for all four players (tile IDs as u32).
    pub discards: [Vec<u32>; 4],
    /// Revealed dora indicator tiles (u32).
    pub dora_indicators: Vec<u32>,
    /// Current point totals for each player.
    pub scores: [i32; 4],
    /// Whether each player has declared riichi.
    pub riichi_declared: [bool; 4],

    pub(crate) _legal_actions: Vec<Action>,

    pub(crate) events: Vec<String>,

    /// Current honba (repeat) counter.
    pub honba: u8,
    /// Number of riichi sticks on the table.
    pub riichi_sticks: u32,
    /// Round wind tile index (27=East, 28=South, 29=West, 30=North).
    pub round_wind: u8,
    /// Seat index of the current dealer.
    pub oya: u8,
    /// Current round number within the game.
    pub kyoku_index: u8,
    /// Tile indices the player is waiting on for tenpai.
    pub waits: Vec<u8>,
    /// Whether the observing player is currently tenpai.
    pub is_tenpai: bool,
    /// Per-player flags indicating tsumogiri (drawn-tile discard) for each discard.
    pub tsumogiri_flags: [Vec<bool>; 4],
    /// Tile discarded to declare riichi for each player, if any.
    pub riichi_sutehais: [Option<u8>; 4],
    /// Last non-tsumogiri discard tile for each player, if any.
    pub last_tedashis: [Option<u8>; 4],
    /// Most recent discard tile on the table, if any.
    pub last_discard: Option<u32>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation {
    /// Create a new observation from raw game state components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: [Vec<u8>; 4],
        melds: [Vec<Meld>; 4],
        discards: [Vec<u8>; 4],
        dora_indicators: Vec<u8>,
        scores: [i32; 4],
        riichi_declared: [bool; 4],
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: [Option<u8>; 4],
        last_tedashis: [Option<u8>; 4],
        last_discard: Option<u32>,
    ) -> Self {
        let hands_u32 = hands.map(|h| h.into_iter().map(|x| x as u32).collect());
        let discards_u32 = discards.map(|d| d.into_iter().map(|x| x as u32).collect());
        let dora_u32 = dora_indicators.iter().map(|&x| x as u32).collect();

        Self {
            player_id,
            hands: hands_u32,
            melds,
            discards: discards_u32,
            dora_indicators: dora_u32,
            scores,
            riichi_declared,
            _legal_actions: legal_actions,
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            tsumogiri_flags: Default::default(),
            riichi_sutehais,
            last_tedashis,
            last_discard,
        }
    }

    /// Return a cloned list of legal actions.
    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    /// Returns a reference to legal actions without cloning.
    pub fn legal_actions_ref(&self) -> &[Action] {
        &self._legal_actions
    }

    /// Find a legal action by its encoded action ID.
    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        let encoder = ActionEncoder::FourPlayer;
        self._legal_actions
            .iter()
            .find(|a| {
                if let Ok(idx) = encoder.encode(a) {
                    (idx as usize) == action_id
                } else {
                    false
                }
            })
            .cloned()
    }

    /// Return a cloned list of MJAI event strings.
    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    /// Serialize this Observation to a base64-encoded JSON string.
    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        let json = serde_json::to_vec(self).map_err(|e| RiichiError::Serialization {
            message: format!("serialization failed: {e}"),
        })?;
        Ok(BASE64.encode(&json))
    }

    /// Deserialize an Observation from a base64-encoded JSON string.
    pub fn deserialize_from_base64(s: &str) -> RiichiResult<Self> {
        let bytes = BASE64.decode(s).map_err(|e| RiichiError::Serialization {
            message: format!("base64 decode failed: {e}"),
        })?;
        let obs: Observation =
            serde_json::from_slice(&bytes).map_err(|e| RiichiError::Serialization {
                message: format!("JSON deserialize failed: {e}"),
            })?;
        Ok(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ActionType};

    fn sample_observation() -> Observation {
        Observation::new(
            1,
            [vec![0, 1], vec![36], vec![], vec![]],
            Default::default(),
            [vec![4], vec![], vec![], vec![]],
            vec![12],
            [25_000, 24_000, 26_000, 25_000],
            [false, true, false, false],
            vec![
                Action::new(ActionType::Discard, Some(4), &[], Some(1)),
                Action::new(ActionType::Riichi, None, &[], Some(1)),
            ],
            vec!["event-a".to_string(), "event-b".to_string()],
            2,
            1,
            27,
            0,
            3,
            vec![7, 8],
            true,
            [None, Some(4), None, None],
            [Some(0), None, None, None],
            Some(4),
        )
    }

    #[test]
    fn new_converts_input_collections_and_preserves_metadata() {
        let obs = sample_observation();
        assert_eq!(obs.player_id, 1);
        assert_eq!(obs.hands[0], vec![0, 1]);
        assert_eq!(obs.hands[1], vec![36]);
        assert_eq!(obs.discards[0], vec![4]);
        assert_eq!(obs.dora_indicators, vec![12]);
        assert_eq!(obs.honba, 2);
        assert_eq!(obs.riichi_sticks, 1);
        assert!(obs.is_tenpai);
    }

    #[test]
    fn legal_action_helpers_clone_find_and_roundtrip_events() {
        let obs = sample_observation();
        let legal = obs.legal_actions_method();
        assert_eq!(legal.len(), 2);
        assert_eq!(obs.legal_actions_ref().len(), 2);

        let discard = obs.find_action(1).expect("discard action should be found");
        assert_eq!(discard.action_type, ActionType::Discard);
        assert_eq!(discard.tile, Some(4));
        assert!(obs.find_action(99).is_none());

        assert_eq!(
            obs.new_events(),
            vec!["event-a".to_string(), "event-b".to_string()]
        );
    }

    #[test]
    fn base64_serialization_roundtrips_and_invalid_input_fails() {
        let obs = sample_observation();
        let encoded = obs.serialize_to_base64().expect("serialize observation");
        let decoded =
            Observation::deserialize_from_base64(&encoded).expect("deserialize observation");
        assert_eq!(decoded.player_id, obs.player_id);
        assert_eq!(decoded.scores, obs.scores);
        assert_eq!(decoded.waits, obs.waits);
        assert_eq!(decoded.last_discard, obs.last_discard);

        let err = Observation::deserialize_from_base64("not-base64")
            .expect_err("invalid base64 should fail");
        assert!(matches!(err, RiichiError::Serialization { .. }));
    }
}
