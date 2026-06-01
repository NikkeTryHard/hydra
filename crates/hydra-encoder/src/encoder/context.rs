use hydra_belief_search::hand_ev::HandEvFeatures;
use hydra_belief_search::shanten_batch::BatchShantenResult;
use hydra_safety::SafetyInfo;

use super::ObservationEncoder;
use super::layout::*;
use super::types::*;

impl ObservationEncoder {
    /// Encode fixed-shape Group C search/belief context planes.
    #[inline]
    pub fn encode_search_features(&mut self, features: &SearchFeaturePlanes) {
        self.clear_range(CH_SEARCH, CH_SEARCH + SEARCH_CONTEXT_CHANNELS);

        for (idx, plane) in features.belief_fields.iter().enumerate() {
            self.copy_channel(CH_SEARCH_BELIEF + idx, plane);
        }
        for (idx, &weight) in features.mixture_weights.iter().enumerate() {
            self.fill_channel(CH_SEARCH_MIXTURE_WEIGHT + idx, weight);
        }
        self.fill_channel(CH_SEARCH_MIXTURE_ENTROPY, features.mixture_entropy);
        self.fill_channel(CH_SEARCH_MIXTURE_ESS, features.mixture_ess);
        self.copy_channel(CH_SEARCH_DELTA_Q, &features.delta_q);
        for (idx, plane) in features.opponent_risk.iter().enumerate() {
            self.copy_channel(CH_SEARCH_RISK + idx, plane);
        }
        for (idx, &stress) in features.opponent_stress.iter().enumerate() {
            self.fill_channel(CH_SEARCH_STRESS + idx, stress);
        }
        if features.belief_features_present {
            self.fill_channel(CH_SEARCH_MASKS, 1.0);
        }
        if features.search_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 1, 1.0);
        }
        if features.robust_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 2, 1.0);
        }
        if features.context_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 3, 1.0);
        }

        let _ = CH_SEARCH_RESERVED;
        let _ = SEARCH_RESERVED_CHANNELS;
    }

    /// Encode fixed-shape Group D Hand-EV context planes.
    #[inline]
    pub fn encode_hand_ev_features(&mut self, hand_ev: &HandEvFeatures) {
        self.clear_range(CH_HAND_EV, CH_HAND_EV + HAND_EV_CHANNELS);

        let tenpai0 = (CH_HAND_EV_TENPAI) * NUM_TILES;
        let tenpai1 = tenpai0 + NUM_TILES;
        let tenpai2 = tenpai1 + NUM_TILES;
        let win0 = (CH_HAND_EV_WIN) * NUM_TILES;
        let win1 = win0 + NUM_TILES;
        let win2 = win1 + NUM_TILES;
        let score = CH_HAND_EV_SCORE * NUM_TILES;

        for discard in 0..NUM_TILES {
            let tenpai = hand_ev.tenpai_prob[discard];
            let win = hand_ev.win_prob[discard];
            self.buffer[tenpai0 + discard] = tenpai[0];
            self.buffer[tenpai1 + discard] = tenpai[1];
            self.buffer[tenpai2 + discard] = tenpai[2];
            self.buffer[win0 + discard] = win[0];
            self.buffer[win1 + discard] = win[1];
            self.buffer[win2 + discard] = win[2];
            self.buffer[score + discard] = hand_ev.expected_score[discard];
        }
        for draw_tile in 0..NUM_TILES {
            let row = (CH_HAND_EV_UKEIRE + draw_tile) * NUM_TILES;
            for discard in 0..NUM_TILES {
                self.buffer[row + discard] = hand_ev.ukeire[discard][draw_tile];
            }
        }
        self.fill_channel(CH_HAND_EV_MASK, 1.0);
    }

    /// Encode a complete observation plus optional Group C / Group D context.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_with_context(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.encode(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        if let Some(features) = search_features {
            self.encode_search_features(features);
        }
        if let Some(features) = hand_ev {
            self.encode_hand_ev_features(features);
        }
        self.as_slice()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_with_context_and_shanten_batch(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        shanten_batch: &BatchShantenResult,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_baseline_prefix_from_batch(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
            shanten_batch,
        );
        if let Some(features) = search_features {
            self.encode_search_features(features);
        }
        if let Some(features) = hand_ev {
            self.encode_hand_ev_features(features);
        }
        self.as_slice()
    }
}
