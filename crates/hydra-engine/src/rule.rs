use serde::{Deserialize, Serialize};

/// Configuration for game rules and yakuman/scoring variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    /// Whether ron is allowed on an ankan that completes kokushi musou.
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    /// Whether 13-wait kokushi musou counts as double yakuman.
    pub is_kokushi_musou_13machi_double: bool,
    /// Whether single-wait suuankou (tanki) counts as double yakuman.
    pub is_suuankou_tanki_double: bool,
    /// Whether pure nine gates (junsei chuurenpoutou) counts as double yakuman.
    pub is_junsei_chuurenpoutou_double: bool,
    /// Whether big four winds (daisuushii) counts as double yakuman.
    pub is_daisuushii_double: bool,
    /// Whether yakuman pao only applies to the liable player (not split).
    pub yakuman_pao_is_liability_only: bool,
    /// Whether triple ron (sanchaho) results in an abortive draw.
    pub sanchaho_is_draw: bool,

    /// Whether swap-calling (kuikae) is forbidden.
    pub kuikae_forbidden: bool,

    /// Whether open kan (Daiminkan/Kakan) dora is revealed after the discard.
    /// - `true`: dora revealed after discard (Tenhou / Mahjong Soul style)
    /// - `false`: dora revealed before discard (Mortal mjai protocol style)
    ///
    /// Note: Ankan (closed kan) always reveals dora immediately (before rinshan tsumo),
    /// regardless of this flag.
    pub open_kan_dora_after_discard: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_mortal()
    }
}

impl GameRule {
    /// Returns the default Tenhou 4-player rule set.
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: true,
        }
    }

    /// Returns the default Mahjong Soul 4-player rule set.
    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: true,
        }
    }

    /// Returns the default Mortal-compatible 4-player rule set.
    pub fn default_mortal() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: false,
        }
    }

    /// Returns the default Mahjong Soul 3-player (sanma) rule set.
    pub fn default_mjsoul_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: true,
        }
    }

    /// Returns the default Tenhou 3-player (sanma) rule set.
    pub fn default_tenhou_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: true,
        }
    }

    /// Returns the default Mortal-compatible 3-player (sanma) rule set.
    pub fn default_mortal_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rules_match_expected_open_kan_and_draw_policies() {
        let tenhou = GameRule::default_tenhou();
        let mortal = GameRule::default_mortal();
        let mjsoul = GameRule::default_mjsoul();

        assert!(tenhou.open_kan_dora_after_discard);
        assert!(tenhou.sanchaho_is_draw);
        assert!(!mortal.open_kan_dora_after_discard);
        assert!(mortal.sanchaho_is_draw);
        assert!(mjsoul.open_kan_dora_after_discard);
        assert!(!mjsoul.sanchaho_is_draw);
        assert!(mjsoul.is_daisuushii_double);
        assert!(mjsoul.yakuman_pao_is_liability_only);
    }

    #[test]
    fn sanma_defaults_keep_three_player_specific_policy_split() {
        let tenhou = GameRule::default_tenhou_sanma();
        let mortal = GameRule::default_mortal_sanma();
        let mjsoul = GameRule::default_mjsoul_sanma();

        assert!(tenhou.open_kan_dora_after_discard);
        assert!(!mortal.open_kan_dora_after_discard);
        assert!(mjsoul.open_kan_dora_after_discard);

        assert!(!tenhou.sanchaho_is_draw);
        assert!(!mortal.sanchaho_is_draw);
        assert!(!mjsoul.sanchaho_is_draw);
        assert!(mjsoul.allows_ron_on_ankan_for_kokushi_musou);
        assert!(!mortal.allows_ron_on_ankan_for_kokushi_musou);
    }

    #[test]
    fn default_impl_matches_mortal_rule_set() {
        assert_eq!(GameRule::default(), GameRule::default_mortal());
    }
}
