use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::Meld;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation3P {
    pub player_id: u8,
    pub hands: [Vec<u32>; 3],
    pub melds: [Vec<Meld>; 3],
    pub discards: [Vec<u32>; 3],
    pub dora_indicators: Vec<u32>,
    pub scores: [i32; 3],
    pub riichi_declared: [bool; 3],

    pub(crate) _legal_actions: Vec<Action>,

    pub(crate) events: Vec<String>,

    pub honba: u8,
    pub riichi_sticks: u32,
    pub round_wind: u8,
    pub oya: u8,
    pub kyoku_index: u8,
    pub waits: Vec<u8>,
    pub is_tenpai: bool,
    pub tsumogiri_flags: [Vec<bool>; 3],
    pub riichi_sutehais: [Option<u8>; 3],
    pub last_tedashis: [Option<u8>; 3],
    pub last_discard: Option<u32>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation3P {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: [Vec<u8>; 3],
        melds: [Vec<Meld>; 3],
        discards: [Vec<u8>; 3],
        dora_indicators: Vec<u8>,
        scores: [i32; 3],
        riichi_declared: [bool; 3],
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: [Option<u8>; 3],
        last_tedashis: [Option<u8>; 3],
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

    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        let encoder = ActionEncoder::ThreePlayer;
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

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    /// Serialize this Observation3P to a base64-encoded JSON string.
    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        let json = serde_json::to_vec(self).map_err(|e| RiichiError::Serialization {
            message: format!("serialization failed: {e}"),
        })?;
        Ok(BASE64.encode(&json))
    }

    /// Deserialize an Observation3P from a base64-encoded JSON string.
    pub fn deserialize_from_base64(s: &str) -> RiichiResult<Self> {
        let bytes = BASE64.decode(s).map_err(|e| RiichiError::Serialization {
            message: format!("base64 decode failed: {e}"),
        })?;
        let obs: Observation3P =
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

    fn sample_observation() -> Observation3P {
        Observation3P::new(
            2,
            [vec![0, 1], vec![36], vec![72]],
            Default::default(),
            [vec![4], vec![], vec![]],
            vec![12],
            [35_000, 28_000, 37_000],
            [false, true, false],
            vec![
                Action::new(ActionType::Discard, Some(0), &[], Some(2)),
                Action::new(ActionType::Kita, None, &[], Some(2)),
            ],
            vec!["event-a".to_string(), "event-b".to_string()],
            1,
            2,
            27,
            1,
            4,
            vec![3, 6],
            true,
            [None, Some(4), None],
            [Some(0), None, None],
            Some(4),
        )
    }

    #[test]
    fn new_converts_input_collections_and_preserves_metadata() {
        let obs = sample_observation();
        assert_eq!(obs.player_id, 2);
        assert_eq!(obs.hands[0], vec![0, 1]);
        assert_eq!(obs.hands[2], vec![72]);
        assert_eq!(obs.discards[0], vec![4]);
        assert_eq!(obs.dora_indicators, vec![12]);
        assert_eq!(obs.riichi_sticks, 2);
        assert!(obs.is_tenpai);
    }

    #[test]
    fn legal_action_helpers_and_base64_roundtrip_work_in_three_player_mode() {
        let obs = sample_observation();
        let legal = obs.legal_actions_method();
        assert_eq!(legal.len(), 2);

        let discard = obs.find_action(0).expect("discard action should be found");
        assert_eq!(discard.action_type, ActionType::Discard);
        assert!(obs.find_action(999).is_none());
        assert_eq!(obs.new_events().len(), 2);

        let encoded = obs.serialize_to_base64().expect("serialize observation3p");
        let decoded =
            Observation3P::deserialize_from_base64(&encoded).expect("deserialize observation3p");
        assert_eq!(decoded.player_id, obs.player_id);
        assert_eq!(decoded.scores, obs.scores);
        assert_eq!(decoded.last_discard, obs.last_discard);
    }

    #[test]
    fn invalid_base64_returns_serialization_error() {
        let err = Observation3P::deserialize_from_base64("not-base64").expect_err("invalid base64");
        assert!(matches!(err, RiichiError::Serialization { .. }));
    }
}
