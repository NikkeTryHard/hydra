use crate::agari::{self, Division, Mentsu};
use crate::types::{Hand, Meld};

// Re-export yaku IDs from the main yaku module
pub use crate::yaku::{
    YakuResult, ID_AKADORA, ID_BAKAZE, ID_CHANKAN, ID_CHANTA, ID_CHIHO, ID_CHINITSU, ID_CHINROUTO,
    ID_CHITOITSU, ID_CHUN, ID_CHUUREN, ID_DAISANGEN, ID_DAISUUSHI, ID_DORA, ID_DOUBLE_RIICHI,
    ID_HAITEI, ID_HAKU, ID_HATSU, ID_HONITSU, ID_HONROUTO, ID_HOUTEI, ID_IPEIKO, ID_IPPATSU,
    ID_ITTSU, ID_JIKAZE, ID_JUNCHAN, ID_JUNSEI_CHUUREN, ID_KOKUSHI, ID_KOKUSHI_13, ID_NUKIDORA,
    ID_PINFU, ID_RIICHI, ID_RINSHAN, ID_RYANPEIKO, ID_RYUISOU, ID_SANANKOU, ID_SANKANTSU,
    ID_SANSHOKU, ID_SANSHOKU_DOKO, ID_SHOSANGEN, ID_SHOUSUUSHI, ID_SUANKO, ID_SUANKO_TANKI,
    ID_SUKANTSU, ID_TANYAO, ID_TENHO, ID_TOITOI, ID_TSUISO, ID_TSUMO, ID_URADORA,
};

#[derive(Debug)]
pub struct YakuContext3P {
    pub is_menzen: bool,
    pub is_reach: bool,
    pub is_ippatsu: bool,
    pub is_tsumo: bool,
    pub is_haitei: bool,
    pub is_houtei: bool,
    pub is_rinshan: bool,
    pub is_chankan: bool,
    pub is_tsumo_first_turn: bool,
    pub is_daburu_reach: bool,
    pub dora_count: u8,
    pub aka_dora: u8,
    pub ura_dora_count: u8,
    pub nukidora_count: u8,
    pub round_wind: u8, // 27=East, 28=South, etc.
    pub seat_wind: u8,
}

impl Default for YakuContext3P {
    fn default() -> Self {
        Self {
            is_menzen: true,
            is_reach: false,
            is_ippatsu: false,
            is_tsumo: false,
            is_haitei: false,
            is_houtei: false,
            is_rinshan: false,
            is_chankan: false,
            is_tsumo_first_turn: false,
            is_daburu_reach: false,
            dora_count: 0,
            aka_dora: 0,
            ura_dora_count: 0,
            nukidora_count: 0,
            round_wind: 27,
            seat_wind: 27,
        }
    }
}

pub fn calculate_yaku_3p(
    hand: &Hand,
    melds: &[Meld],
    ctx: &YakuContext3P,
    win_tile: u8,
) -> YakuResult {
    let divisions = agari::find_divisions(hand);
    let mut best_res = YakuResult::default();

    if divisions.is_empty() {
        if agari::is_kokushi(hand) {
            let is_13_wait = hand.counts[win_tile as usize] == 2;
            if is_13_wait {
                best_res.han = 26;
                best_res.yakuman_count = 2;
                best_res.push_yaku_id(ID_KOKUSHI_13);
                best_res
                    .yaku_names
                    .push("Kokushi Musou 13-wait".to_string());
            } else {
                best_res.han = 13;
                best_res.yakuman_count = 1;
                best_res.push_yaku_id(ID_KOKUSHI);
                best_res.yaku_names.push("Kokushi Musou".to_string());
            }
            return best_res;
        }
        if agari::is_chiitoitsu(hand) {
            best_res.han = 2;
            best_res.fu = 25;
            best_res.push_yaku_id(ID_CHITOITSU);
            best_res.yaku_names.push("Chiitoitsu".to_string());

            if is_tanyao(hand, melds) {
                best_res.han += 1;
                best_res.push_yaku_id(12);
                best_res.yaku_names.push("Tanyao".to_string());
            }
            if is_chinitsu(hand, melds) {
                best_res.han += 6;
                best_res.push_yaku_id(29);
                best_res.yaku_names.push("Chinitsu".to_string());
            } else if is_honitsu(hand, melds) {
                best_res.han += 3;
                best_res.push_yaku_id(27);
                best_res.yaku_names.push("Honitsu".to_string());
            }
            if is_honroutou(hand, melds) {
                best_res.han += 2;
                best_res.push_yaku_id(24);
                best_res.yaku_names.push("Honroutou".to_string());
            }

            apply_yakuman(
                &mut best_res,
                hand,
                melds,
                ctx,
                &Division {
                    head: 0,
                    body: Vec::new(),
                },
                None,
                win_tile,
            );
            apply_static_yaku(&mut best_res, ctx);
            return best_res;
        }
        return best_res;
    }

    for div in &divisions {
        let mut win_group_indices = Vec::new();
        if div.head == win_tile {
            win_group_indices.push(None);
        }
        for (idx, m) in div.body.iter().enumerate() {
            match m {
                Mentsu::Koutsu(t) => {
                    if *t == win_tile {
                        win_group_indices.push(Some(idx));
                    }
                }
                Mentsu::Shuntsu(t) => {
                    if win_tile >= *t && win_tile <= *t + 2 {
                        win_group_indices.push(Some(idx));
                    }
                }
            }
        }

        if win_group_indices.is_empty() {
            continue;
        }

        for wg_idx in win_group_indices {
            let mut res = YakuResult::default();

            apply_yakuman(&mut res, hand, melds, ctx, div, wg_idx, win_tile);
            if res.han >= 13 {
                if res.han > best_res.han {
                    best_res = res;
                }
                continue;
            }

            apply_static_yaku(&mut res, ctx);

            // Tanyao
            if is_tanyao(hand, melds) {
                res.han += 1;
                res.push_yaku_id(ID_TANYAO);
                res.yaku_names.push("Tanyao".to_string());
            }

            // Pinfu check
            if check_pinfu(div, melds, ctx, wg_idx, win_tile) {
                res.han += 1;
                res.push_yaku_id(ID_PINFU);
                res.yaku_names.push("Pinfu".to_string());
                res.fu = if ctx.is_tsumo { 20 } else { 30 };
            } else {
                res.fu = calculate_fu_with_waiting(div, melds, ctx, wg_idx, win_tile);
            }

            // Yakuhai
            let yakuhai_tiles = [31, 32, 33, ctx.round_wind, ctx.seat_wind];
            for (i, &t) in yakuhai_tiles.iter().enumerate() {
                let count = div
                    .body
                    .iter()
                    .filter(|m| matches!(m, Mentsu::Koutsu(tile) if *tile == t))
                    .count()
                    + melds
                        .iter()
                        .filter(|m| m.tiles[0] == t && m.meld_type != crate::types::MeldType::Chi)
                        .count();
                if count > 0 {
                    res.han += count as u8;
                    let id = match t {
                        31 => ID_HAKU,
                        32 => ID_HATSU,
                        33 => ID_CHUN,
                        _ => {
                            if i == 3 {
                                ID_BAKAZE
                            } else {
                                ID_JIKAZE
                            }
                        }
                    };
                    res.push_yaku_id(id);
                    res.yaku_names.push("Yakuhai".to_string());
                }
            }

            // Dragons check (Daisangen / Shousangen)
            let haku_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 31,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 31 && m.meld_type != crate::types::MeldType::Chi);
            let hatsu_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 32,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 32 && m.meld_type != crate::types::MeldType::Chi);
            let chun_koutsu = div.body.iter().any(|m| match m {
                Mentsu::Koutsu(t) => *t == 33,
                _ => false,
            }) || melds
                .iter()
                .any(|m| m.tiles[0] == 33 && m.meld_type != crate::types::MeldType::Chi);

            if haku_koutsu && hatsu_koutsu && chun_koutsu {
                // Already handled in apply_yakuman (Daisangen)
            } else {
                let dragon_koutsu_count = (if haku_koutsu { 1 } else { 0 })
                    + (if hatsu_koutsu { 1 } else { 0 })
                    + (if chun_koutsu { 1 } else { 0 });
                let dragon_pair_count = (if div.head == 31 { 1 } else { 0 })
                    + (if div.head == 32 { 1 } else { 0 })
                    + (if div.head == 33 { 1 } else { 0 });
                if dragon_koutsu_count == 2 && dragon_pair_count == 1 {
                    res.han += 2;
                    res.push_yaku_id(ID_SHOSANGEN);
                    res.yaku_names.push("Shousangen".to_string());
                }
            }

            // Toitoi
            let koutsu_total = div
                .body
                .iter()
                .filter(|m| matches!(m, Mentsu::Koutsu(_)))
                .count()
                + melds
                    .iter()
                    .filter(|m| m.meld_type != crate::types::MeldType::Chi)
                    .count();
            if koutsu_total == 4 {
                res.han += 2;
                res.push_yaku_id(ID_TOITOI);
                res.yaku_names.push("Toitoi".to_string());
            }

            // San Ankou
            let mut closed_koutsu_count = 0;
            for (idx, m) in div.body.iter().enumerate() {
                if let Mentsu::Koutsu(_) = m {
                    if !ctx.is_tsumo && Some(idx) == wg_idx {
                        continue;
                    }
                    closed_koutsu_count += 1;
                }
            }
            for m in melds {
                if m.meld_type == crate::types::MeldType::Ankan {
                    closed_koutsu_count += 1;
                }
            }
            if closed_koutsu_count == 3 {
                res.han += 2;
                res.push_yaku_id(ID_SANANKOU);
                res.yaku_names.push("San Ankou".to_string());
            }

            // San Kantsu
            let kantsu_count = melds
                .iter()
                .filter(|m| {
                    m.meld_type == crate::types::MeldType::Daiminkan
                        || m.meld_type == crate::types::MeldType::Ankan
                        || m.meld_type == crate::types::MeldType::Kakan
                })
                .count();
            if kantsu_count == 3 {
                res.han += 2;
                res.push_yaku_id(ID_SANKANTSU);
                res.yaku_names.push("San Kantsu".to_string());
            }

            // Iipeiko / Ryanpeikou (Closed only)
            if ctx.is_menzen {
                let mut shuntsu_tiles = Vec::new();
                for m in &div.body {
                    if let Mentsu::Shuntsu(t) = m {
                        shuntsu_tiles.push(*t);
                    }
                }
                shuntsu_tiles.sort();
                let mut identical_pairs = 0;
                let mut i = 0;
                while i + 1 < shuntsu_tiles.len() {
                    if shuntsu_tiles[i] == shuntsu_tiles[i + 1] {
                        identical_pairs += 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if identical_pairs == 2 {
                    res.han += 3;
                    res.push_yaku_id(ID_RYANPEIKO);
                    res.yaku_names.push("Ryanpeikou".to_string());
                } else if identical_pairs == 1 {
                    res.han += 1;
                    res.push_yaku_id(ID_IPEIKO);
                    res.yaku_names.push("Iipeiko".to_string());
                }
            }

            // Ittsu / Sanshoku Doujun
            if check_ittsu(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.push_yaku_id(ID_ITTSU);
                res.yaku_names.push("Ittsu".to_string());
            }
            if is_sanshoku_doujun(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.push_yaku_id(ID_SANSHOKU);
                res.yaku_names.push("Sanshoku Doujun".to_string());
            }
            if is_sanshoku_doukou(div, melds) {
                res.han += 2;
                res.push_yaku_id(ID_SANSHOKU_DOKO);
                res.yaku_names.push("Sanshoku Doukou".to_string());
            }

            // Honitsu / Chinitsu
            if is_chinitsu(hand, melds) {
                res.han += if ctx.is_menzen { 6 } else { 5 };
                res.push_yaku_id(ID_CHINITSU);
                res.yaku_names.push("Chinitsu".to_string());
            } else if is_honitsu(hand, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.push_yaku_id(ID_HONITSU);
                res.yaku_names.push("Honitsu".to_string());
            }

            // Chantai / Junchan / Honroutou
            if is_honroutou(hand, melds) {
                res.han += 2;
                res.push_yaku_id(ID_HONROUTO);
                res.yaku_names.push("Honroutou".to_string());
            } else if is_junchan(div, melds) {
                res.han += if ctx.is_menzen { 3 } else { 2 };
                res.push_yaku_id(ID_JUNCHAN);
                res.yaku_names.push("Junchan".to_string());
            } else if is_chantai(div, melds) {
                res.han += if ctx.is_menzen { 2 } else { 1 };
                res.push_yaku_id(ID_CHANTA);
                res.yaku_names.push("Chantai".to_string());
            }

            if res.han > best_res.han || (res.han == best_res.han && res.fu > best_res.fu) {
                best_res = res;
            }
        }
    }

    best_res
}

fn calculate_fu_with_waiting(
    div: &Division,
    melds: &[Meld],
    ctx: &YakuContext3P,
    wg_idx: Option<usize>,
    win_tile: u8,
) -> u8 {
    let mut fu: u8 = 20;
    if ctx.is_tsumo {
        fu += 2;
    } else if ctx.is_menzen {
        fu += 10;
    }

    if div.head == ctx.round_wind {
        fu += 2;
    }
    if div.head == ctx.seat_wind {
        fu += 2;
    }
    if div.head >= 31 {
        fu += 2;
    }

    // Waiting fu
    match wg_idx {
        None => fu += 2, // Tanki
        Some(idx) => match div.body[idx] {
            Mentsu::Koutsu(_) => {}
            Mentsu::Shuntsu(t) => {
                if win_tile == t + 1
                    || (win_tile == t + 2 && (t % 9 == 0))
                    || (win_tile == t && (t % 9 == 6))
                {
                    fu += 2;
                }
            }
        },
    }

    for (idx, m) in div.body.iter().enumerate() {
        if let Mentsu::Koutsu(t) = m {
            let mut f = 4;
            if !ctx.is_tsumo && Some(idx) == wg_idx {
                f = 2;
            }
            if is_terminal(*t) {
                f *= 2;
            }
            fu += f;
        }
    }
    for m in melds {
        if m.tile_count as usize >= 3 && m.tiles[0] == m.tiles[1] {
            let mut f = 2;
            if !m.opened {
                f = 4;
            }
            if is_terminal(m.tiles[0]) {
                f *= 2;
            }
            if m.meld_type == crate::types::MeldType::Daiminkan
                || m.meld_type == crate::types::MeldType::Ankan
                || m.meld_type == crate::types::MeldType::Kakan
            {
                f *= 4;
            }
            fu += f;
        }
    }

    if fu == 20 && !ctx.is_tsumo {
        fu = 30;
    }

    fu.div_ceil(10) * 10
}

fn check_pinfu(
    div: &Division,
    melds: &[Meld],
    ctx: &YakuContext3P,
    wg_idx: Option<usize>,
    win_tile: u8,
) -> bool {
    if !ctx.is_menzen {
        return false;
    }
    if !melds.is_empty() {
        return false;
    }
    for m in &div.body {
        if let Mentsu::Koutsu(_) = m {
            return false;
        }
    }
    if is_yakuhai_tile(div.head, ctx) {
        return false;
    }

    if let Some(idx) = wg_idx {
        if let Mentsu::Shuntsu(t) = div.body[idx] {
            if win_tile == t {
                if t % 9 == 6 {
                    return false;
                }
                return true;
            }
            if win_tile == t + 2 {
                if t % 9 == 0 {
                    return false;
                }
                return true;
            }
        }
    }
    false
}

fn is_yakuhai_tile(tile: u8, ctx: &YakuContext3P) -> bool {
    tile >= 31 || tile == ctx.round_wind || tile == ctx.seat_wind
}

fn is_honroutou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !is_terminal(i as u8) {
            return false;
        }
    }
    if melds
        .iter()
        .any(|m| m.tiles_slice().iter().any(|&t| !is_terminal(t)))
    {
        return false;
    }
    true
}

fn is_junchan(div: &Division, melds: &[Meld]) -> bool {
    if !is_number_terminal(div.head) {
        return false;
    }
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => {
                if !is_number_terminal(*t) {
                    return false;
                }
            }
            Mentsu::Shuntsu(t) => {
                if !is_number_terminal(*t) && !is_number_terminal(t + 2) {
                    return false;
                }
            }
        }
    }
    for m in melds {
        if m.tiles_slice().iter().all(|&t| !is_number_terminal(t)) {
            return false;
        }
    }
    true
}

fn is_chantai(div: &Division, melds: &[Meld]) -> bool {
    if !is_terminal(div.head) {
        return false;
    }
    let mut has_honor = is_honor(div.head);
    for m in &div.body {
        match m {
            Mentsu::Koutsu(t) => {
                if !is_terminal(*t) {
                    return false;
                }
                if is_honor(*t) {
                    has_honor = true;
                }
            }
            Mentsu::Shuntsu(t) => {
                if !is_terminal(*t) && !is_terminal(t + 2) {
                    return false;
                }
            }
        }
    }
    for m in melds {
        if m.tiles_slice().iter().all(|&t| !is_terminal(t)) {
            return false;
        }
        if m.tiles_slice().iter().any(|&t| is_honor(t)) {
            has_honor = true;
        }
    }
    has_honor
}

fn is_terminal(tile: u8) -> bool {
    tile >= 27 || tile.is_multiple_of(9) || (tile % 9 == 8)
}
fn is_number_terminal(tile: u8) -> bool {
    tile < 27 && (tile.is_multiple_of(9) || tile % 9 == 8)
}
fn is_honor(tile: u8) -> bool {
    tile >= 27
}

fn is_honitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    let mut has_honor = false;
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i < 9 {
                suits[0] = true;
            } else if i < 18 {
                suits[1] = true;
            } else if i < 27 {
                suits[2] = true;
            } else {
                has_honor = true;
            }
        }
    }
    for meld in melds {
        for &t in meld.tiles_slice() {
            let idx = t as usize;
            if idx < 9 {
                suits[0] = true;
            } else if idx < 18 {
                suits[1] = true;
            } else if idx < 27 {
                suits[2] = true;
            } else {
                has_honor = true;
            }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1 && has_honor
}

fn is_chinitsu(hand: &Hand, melds: &[Meld]) -> bool {
    let mut suits = [false; 3];
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i >= 27 {
                return false;
            }
            if i < 9 {
                suits[0] = true;
            } else if i < 18 {
                suits[1] = true;
            } else if i < 27 {
                suits[2] = true;
            }
        }
    }
    for meld in melds {
        for &t in meld.tiles_slice() {
            let idx = t as usize;
            if idx >= 27 {
                return false;
            }
            if idx < 9 {
                suits[0] = true;
            } else if idx < 18 {
                suits[1] = true;
            } else if idx < 27 {
                suits[2] = true;
            }
        }
    }
    suits.iter().filter(|&&b| b).count() == 1
}

fn apply_static_yaku(res: &mut YakuResult, ctx: &YakuContext3P) {
    // Riichi
    if ctx.is_reach && !ctx.is_daburu_reach {
        res.han += 1;
        res.push_yaku_id(ID_RIICHI);
    }
    if ctx.is_daburu_reach {
        res.han += 2;
        res.push_yaku_id(ID_DOUBLE_RIICHI);
    }
    if ctx.is_ippatsu {
        res.han += 1;
        res.push_yaku_id(ID_IPPATSU);
    }
    // Menzen Tsumo
    if ctx.is_menzen && ctx.is_tsumo {
        res.han += 1;
        res.push_yaku_id(ID_TSUMO);
    }
    if ctx.is_haitei && ctx.is_tsumo {
        res.han += 1;
        res.push_yaku_id(ID_HAITEI);
    }
    if ctx.is_houtei && !ctx.is_tsumo {
        res.han += 1;
        res.push_yaku_id(ID_HOUTEI);
    }
    if ctx.is_rinshan && ctx.is_tsumo {
        res.han += 1;
        res.push_yaku_id(ID_RINSHAN);
    }
    if ctx.is_chankan && !ctx.is_tsumo {
        res.han += 1;
        res.push_yaku_id(ID_CHANKAN);
    }

    if ctx.dora_count > 0 {
        res.han += ctx.dora_count;
        res.push_yaku_id(ID_DORA);
    }
    if ctx.aka_dora > 0 {
        res.han += ctx.aka_dora;
        res.push_yaku_id(ID_AKADORA);
    }
    if ctx.ura_dora_count > 0 {
        res.han += ctx.ura_dora_count;
        res.push_yaku_id(ID_URADORA);
    }
    if ctx.nukidora_count > 0 {
        res.han += ctx.nukidora_count;
        res.push_yaku_id(ID_NUKIDORA);
    }
}

fn apply_yakuman(
    res: &mut YakuResult,
    hand: &Hand,
    melds: &[Meld],
    ctx: &YakuContext3P,
    div: &Division,
    wg_idx: Option<usize>,
    win_tile: u8,
) {
    let mut yakuman_count = 0;

    if div.head == 0 && div.body.is_empty() {
        // Special Case: Kokushi / Chiitoitsu call from start of calculate_yaku_3p
    }

    // Tsuu iisou (All Honors)
    if is_tsuu_iisou(hand, melds) {
        yakuman_count += 1;
        res.push_yaku_id(ID_TSUISO);
        res.yaku_names.push("Tsuu iisou".to_string());
    }

    // Chinroutou (All Terminals)
    if is_chinroutou(hand, melds) {
        yakuman_count += 1;
        res.push_yaku_id(ID_CHINROUTO);
        res.yaku_names.push("Chinroutou".to_string());
    }

    // Ryuu iisou (All Green)
    if is_ryuu_iisou(hand, melds) {
        yakuman_count += 1;
        res.push_yaku_id(ID_RYUISOU);
        res.yaku_names.push("Ryuu iisou".to_string());
    }

    // Su Kantsu (Four Kans)
    if melds
        .iter()
        .filter(|m| {
            m.meld_type == crate::types::MeldType::Daiminkan
                || m.meld_type == crate::types::MeldType::Ankan
                || m.meld_type == crate::types::MeldType::Kakan
        })
        .count()
        == 4
    {
        yakuman_count += 1;
        res.push_yaku_id(ID_SUKANTSU);
        res.yaku_names.push("Su Kantsu".to_string());
    }

    // Chuuren Poutou
    if ctx.is_menzen && (div.body.len() + melds.len()) == 4 && is_chuuren_poutou(hand) {
        let is_9_wait = is_chuuren_9_wait(hand, win_tile);
        if is_9_wait {
            yakuman_count += 2;
            res.push_yaku_id(ID_JUNSEI_CHUUREN);
            res.yaku_names.push("Chuuren Poutou 9-wait".to_string());
        } else {
            yakuman_count += 1;
            res.push_yaku_id(ID_CHUUREN);
            res.yaku_names.push("Chuuren Poutou".to_string());
        }
    }

    // Tenhou / Chiihou
    if ctx.is_tsumo_first_turn && ctx.is_menzen && ctx.is_tsumo {
        if ctx.seat_wind == 27 {
            yakuman_count += 1;
            res.push_yaku_id(ID_TENHO);
            res.yaku_names.push("Tenhou".to_string());
        } else {
            yakuman_count += 1;
            res.push_yaku_id(ID_CHIHO);
            res.yaku_names.push("Chiihou".to_string());
        }
    }

    // Su Ankou
    let mut closed_koutsu_count = 0;
    for (idx, m) in div.body.iter().enumerate() {
        if let Mentsu::Koutsu(_) = m {
            if !ctx.is_tsumo && Some(idx) == wg_idx {
                continue;
            }
            closed_koutsu_count += 1;
        }
    }
    for m in melds {
        if m.meld_type == crate::types::MeldType::Ankan {
            closed_koutsu_count += 1;
        }
    }
    if closed_koutsu_count == 4 {
        if wg_idx.is_none() {
            yakuman_count += 2;
            res.push_yaku_id(ID_SUANKO_TANKI);
            res.yaku_names.push("Su Ankou Tanki".to_string());
        } else {
            yakuman_count += 1;
            res.push_yaku_id(ID_SUANKO);
            res.yaku_names.push("Su Ankou".to_string());
        }
    }

    // Daisangen
    let haku_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 31,
        _ => false,
    }) || melds.iter().any(|m| m.tiles_slice().contains(&31));
    let hatsu_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 32,
        _ => false,
    }) || melds.iter().any(|m| m.tiles_slice().contains(&32));
    let chun_koutsu = div.body.iter().any(|m| match m {
        Mentsu::Koutsu(t) => *t == 33,
        _ => false,
    }) || melds.iter().any(|m| m.tiles_slice().contains(&33));

    if haku_koutsu && hatsu_koutsu && chun_koutsu {
        yakuman_count += 1;
        res.push_yaku_id(ID_DAISANGEN);
        res.yaku_names.push("Daisangen".to_string());
    }

    // Winds
    let mut wind_koutsu_count = 0;
    let mut wind_pair_count = 0;
    for w in 27..=30 {
        let has_koutsu = div.body.iter().any(|m| match m {
            Mentsu::Koutsu(t) => *t == w,
            _ => false,
        }) || melds
            .iter()
            .any(|m| m.tiles[0] == w && m.meld_type != crate::types::MeldType::Chi);
        if has_koutsu {
            wind_koutsu_count += 1;
        } else if div.head == w {
            wind_pair_count += 1;
        }
    }
    if wind_koutsu_count == 4 {
        yakuman_count += 2;
        res.push_yaku_id(ID_DAISUUSHI);
        res.yaku_names.push("Daisushii".to_string());
    } else if wind_koutsu_count == 3 && wind_pair_count == 1 {
        yakuman_count += 1;
        res.push_yaku_id(ID_SHOUSUUSHI);
        res.yaku_names.push("Shousushii".to_string());
    }

    if yakuman_count > 0 {
        res.han = 13 * yakuman_count;
        res.yakuman_count = yakuman_count;
    }
}

fn is_tsuu_iisou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && i < 27 {
            return false;
        }
    }
    for m in melds {
        if m.tiles_slice().iter().any(|&t| t < 27) {
            return false;
        }
    }
    true
}

fn is_chinroutou(hand: &Hand, melds: &[Meld]) -> bool {
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !is_number_terminal(i as u8) {
            return false;
        }
    }
    for m in melds {
        if m.tiles_slice().iter().any(|&t| !is_number_terminal(t)) {
            return false;
        }
    }
    true
}

fn is_ryuu_iisou(hand: &Hand, melds: &[Meld]) -> bool {
    let green_tiles = [19, 20, 21, 23, 25, 32];
    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 && !green_tiles.contains(&(i as u8)) {
            return false;
        }
    }
    for m in melds {
        if m.tiles_slice().iter().any(|&t| !green_tiles.contains(&t)) {
            return false;
        }
    }
    true
}

fn is_chuuren_poutou(hand: &Hand) -> bool {
    let mut counts = [0u8; 9];
    let mut suit = None;

    for (i, &count) in hand.counts.iter().enumerate() {
        if count > 0 {
            if i >= 27 {
                return false;
            }
            let s = i / 9;
            if let Some(prev_s) = suit {
                if prev_s != s {
                    return false;
                }
            } else {
                suit = Some(s);
            }
            counts[i % 9] = count;
        }
    }

    if counts[0] < 3 || counts[8] < 3 {
        return false;
    }
    if counts[1..8].contains(&0) {
        return false;
    }
    true
}

fn is_chuuren_9_wait(hand: &Hand, win_tile: u8) -> bool {
    if win_tile >= 27 {
        return false;
    }
    let val = (win_tile % 9) as usize;
    let counts = &hand.counts[(win_tile / 9 * 9) as usize..(win_tile / 9 * 9 + 9) as usize];

    if val == 0 || val == 8 {
        counts[val] == 4
    } else {
        counts[val] == 2
    }
}

fn check_ittsu(div: &Division, melds: &[Meld]) -> bool {
    for suit_offset in [0, 9, 18] {
        let mut has_123 = false;
        let mut has_456 = false;
        let mut has_789 = false;

        for m in &div.body {
            if let Mentsu::Shuntsu(t) = m {
                if *t == suit_offset {
                    has_123 = true;
                } else if *t == suit_offset + 3 {
                    has_456 = true;
                } else if *t == suit_offset + 6 {
                    has_789 = true;
                }
            }
        }

        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == suit_offset {
                    has_123 = true;
                } else if t == suit_offset + 3 {
                    has_456 = true;
                } else if t == suit_offset + 6 {
                    has_789 = true;
                }
            }
        }

        if has_123 && has_456 && has_789 {
            return true;
        }
    }
    false
}

fn is_sanshoku_doujun(div: &Division, melds: &[Meld]) -> bool {
    for i in 0..7 {
        let mut has_man = false;
        let mut has_pin = false;
        let mut has_sou = false;
        for m in &div.body {
            if let Mentsu::Shuntsu(t) = m {
                if *t == i {
                    has_man = true;
                }
                if *t == i + 9 {
                    has_pin = true;
                }
                if *t == i + 18 {
                    has_sou = true;
                }
            }
        }
        for m in melds {
            if m.meld_type == crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i {
                    has_man = true;
                }
                if t == i + 9 {
                    has_pin = true;
                }
                if t == i + 18 {
                    has_sou = true;
                }
            }
        }
        if has_man && has_pin && has_sou {
            return true;
        }
    }
    false
}

fn is_tanyao(hand: &Hand, melds: &[Meld]) -> bool {
    let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    for &t in &terminals {
        if hand.counts[t] > 0 {
            return false;
        }
    }
    for meld in melds {
        for &t in meld.tiles_slice() {
            if terminals.contains(&(t as usize)) {
                return false;
            }
        }
    }
    true
}

fn is_sanshoku_doukou(div: &Division, melds: &[Meld]) -> bool {
    for i in 0..9 {
        let mut has_man = false;
        let mut has_pin = false;
        let mut has_sou = false;
        for m in &div.body {
            if let Mentsu::Koutsu(t) = m {
                if *t == i {
                    has_man = true;
                }
                if *t == i + 9 {
                    has_pin = true;
                }
                if *t == i + 18 {
                    has_sou = true;
                }
            }
        }
        for m in melds {
            if m.meld_type != crate::types::MeldType::Chi {
                let t = m.tiles[0];
                if t == i {
                    has_man = true;
                }
                if t == i + 9 {
                    has_pin = true;
                }
                if t == i + 18 {
                    has_sou = true;
                }
            }
        }
        if has_man && has_pin && has_sou {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agari::Mentsu;
    use crate::types::MeldType;

    fn hand_with_tiles(tiles: &[u8]) -> Hand {
        Hand::new(Some(tiles.to_vec()))
    }

    fn meld(meld_type: MeldType, tiles: &[u8], opened: bool) -> Meld {
        Meld::new(meld_type, tiles, opened, -1, None)
    }

    #[test]
    fn apply_static_yaku_adds_all_enabled_flags() {
        let mut res = YakuResult::default();
        let ctx = YakuContext3P {
            is_menzen: true,
            is_reach: true,
            is_ippatsu: true,
            is_tsumo: true,
            is_haitei: true,
            is_houtei: false,
            is_rinshan: true,
            is_chankan: false,
            is_tsumo_first_turn: false,
            is_daburu_reach: false,
            dora_count: 2,
            aka_dora: 1,
            ura_dora_count: 3,
            nukidora_count: 2,
            round_wind: 27,
            seat_wind: 28,
        };

        apply_static_yaku(&mut res, &ctx);

        assert_eq!(res.han, 13);
        assert!(res.yaku_ids.contains(&ID_RIICHI));
        assert!(res.yaku_ids.contains(&ID_IPPATSU));
        assert!(res.yaku_ids.contains(&ID_TSUMO));
        assert!(res.yaku_ids.contains(&ID_HAITEI));
        assert!(res.yaku_ids.contains(&ID_RINSHAN));
        assert!(res.yaku_ids.contains(&ID_DORA));
        assert!(res.yaku_ids.contains(&ID_AKADORA));
        assert!(res.yaku_ids.contains(&ID_URADORA));
        assert!(res.yaku_ids.contains(&ID_NUKIDORA));
    }

    #[test]
    fn apply_static_yaku_prefers_double_reach_and_non_tsumo_branches() {
        let mut res = YakuResult::default();
        let ctx = YakuContext3P {
            is_menzen: false,
            is_reach: true,
            is_ippatsu: false,
            is_tsumo: false,
            is_haitei: false,
            is_houtei: true,
            is_rinshan: false,
            is_chankan: true,
            is_tsumo_first_turn: false,
            is_daburu_reach: true,
            dora_count: 0,
            aka_dora: 0,
            ura_dora_count: 0,
            nukidora_count: 0,
            round_wind: 27,
            seat_wind: 27,
        };

        apply_static_yaku(&mut res, &ctx);

        assert_eq!(res.han, 4);
        assert!(res.yaku_ids.contains(&ID_DOUBLE_RIICHI));
        assert!(!res.yaku_ids.contains(&ID_RIICHI));
        assert!(res.yaku_ids.contains(&ID_HOUTEI));
        assert!(res.yaku_ids.contains(&ID_CHANKAN));
    }

    #[test]
    fn calculate_fu_with_waiting_handles_tanki_and_rounding() {
        let div = Division {
            head: 27,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Koutsu(31),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: false,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };

        let fu = calculate_fu_with_waiting(&div, &[], &ctx, None, 27);
        assert_eq!(fu, 50);
    }

    #[test]
    fn check_pinfu_accepts_closed_ryanmen_and_rejects_yakuhai_pair() {
        let good_div = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let good_ctx = YakuContext3P {
            is_menzen: true,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };
        assert!(check_pinfu(&good_div, &[], &good_ctx, Some(0), 0));

        let bad_div = Division {
            head: 27,
            body: good_div.body.clone(),
        };
        assert!(!check_pinfu(&bad_div, &[], &good_ctx, Some(0), 2));
    }

    #[test]
    fn suit_and_terminal_helpers_distinguish_flush_and_terminal_patterns() {
        let honitsu_hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 27, 27, 28, 28, 29, 29, 31, 31]);
        let chinitsu_hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 3]);
        let honroutou_hand = hand_with_tiles(&[0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 26, 26, 27, 27]);
        let tanyao_hand = hand_with_tiles(&[1, 1, 2, 2, 10, 10, 11, 11, 19, 19, 20, 20, 21, 21]);

        assert!(is_honitsu(&honitsu_hand, &[]));
        assert!(!is_chinitsu(&honitsu_hand, &[]));
        assert!(is_chinitsu(&chinitsu_hand, &[]));
        assert!(is_honroutou(&honroutou_hand, &[]));
        assert!(is_tanyao(&tanyao_hand, &[]));
        assert!(!is_tanyao(&honroutou_hand, &[]));
    }

    #[test]
    fn sequence_and_triplet_helpers_detect_cross_suit_patterns() {
        let doujun = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
                Mentsu::Shuntsu(3),
            ],
        };
        let doukou = Division {
            head: 31,
            body: vec![
                Mentsu::Koutsu(0),
                Mentsu::Koutsu(9),
                Mentsu::Koutsu(18),
                Mentsu::Shuntsu(3),
            ],
        };
        let ittsu = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(6),
                Mentsu::Koutsu(31),
            ],
        };

        assert!(is_sanshoku_doujun(&doujun, &[]));
        assert!(is_sanshoku_doukou(&doukou, &[]));
        assert!(check_ittsu(&ittsu, &[]));
    }

    #[test]
    fn chuuren_helpers_distinguish_base_and_nine_wait_shapes() {
        let base = hand_with_tiles(&[0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8]);
        let nine_wait = hand_with_tiles(&[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);

        assert!(is_chuuren_poutou(&base));
        assert!(is_chuuren_9_wait(&base, 4));
        assert!(is_chuuren_poutou(&nine_wait));
        assert!(is_chuuren_9_wait(&nine_wait, 0));
    }

    #[test]
    fn calculate_fu_with_waiting_counts_open_and_closed_meld_fu() {
        let div = Division {
            head: 1,
            body: vec![
                Mentsu::Koutsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let melds = [Meld::new(
            MeldType::Ankan,
            &[31, 31, 31, 31],
            false,
            -1,
            None,
        )];
        let ctx = YakuContext3P {
            is_menzen: false,
            is_tsumo: true,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };

        let fu = calculate_fu_with_waiting(&div, &melds, &ctx, Some(0), 0);
        assert_eq!(fu, 70);
    }

    #[test]
    fn outside_and_yakuman_shape_helpers_detect_expected_patterns() {
        let tsuuiisou = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31, 31]);
        let chinroutou = hand_with_tiles(&[0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18, 18]);
        let ryuuiisou = hand_with_tiles(&[19, 19, 20, 20, 21, 21, 23, 23, 25, 25, 32, 32, 32, 32]);

        assert!(is_tsuu_iisou(&tsuuiisou, &[]));
        assert!(is_chinroutou(&chinroutou, &[]));
        assert!(is_ryuu_iisou(&ryuuiisou, &[]));
        assert!(!is_tsuu_iisou(&chinroutou, &[]));
        assert!(!is_chinroutou(&tsuuiisou, &[]));
        assert!(!is_ryuu_iisou(&chinroutou, &[]));
    }

    #[test]
    fn junchan_and_chantai_distinguish_number_terminals_from_honor_mix() {
        let junchan_div = Division {
            head: 0,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(6),
                Mentsu::Shuntsu(9),
                Mentsu::Koutsu(26),
            ],
        };
        let chantai_div = Division {
            head: 27,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(6),
                Mentsu::Koutsu(8),
                Mentsu::Koutsu(31),
            ],
        };
        let bad_div = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(1),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Koutsu(18),
            ],
        };

        assert!(is_junchan(&junchan_div, &[]));
        assert!(is_chantai(&chantai_div, &[]));
        assert!(!is_junchan(&chantai_div, &[]));
        assert!(!is_chantai(&bad_div, &[]));
    }

    #[test]
    fn apply_yakuman_detects_tenhou_and_double_wind_yakuman() {
        let hand = hand_with_tiles(&[27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33]);
        let mut res = YakuResult::default();
        let div = Division {
            head: 27,
            body: vec![
                Mentsu::Koutsu(28),
                Mentsu::Koutsu(29),
                Mentsu::Koutsu(30),
                Mentsu::Koutsu(31),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: true,
            is_tsumo_first_turn: true,
            seat_wind: 27,
            round_wind: 27,
            ..Default::default()
        };

        apply_yakuman(&mut res, &hand, &[], &ctx, &div, None, 27);

        assert!(res.yakuman_count >= 3);
        assert_eq!(res.han, 13 * res.yakuman_count);
        assert!(res.yaku_ids.contains(&ID_TENHO));
        assert!(res.yaku_ids.contains(&ID_SHOUSUUSHI));
        assert!(res.yaku_ids.contains(&ID_TSUISO));
    }

    #[test]
    fn apply_yakuman_detects_suuankou_tanki_shape() {
        let hand = hand_with_tiles(&[31, 31, 32, 32, 32, 33, 33, 33, 0, 0, 0, 9, 9, 9]);
        let mut res = YakuResult::default();
        let div = Division {
            head: 31,
            body: vec![
                Mentsu::Koutsu(32),
                Mentsu::Koutsu(33),
                Mentsu::Koutsu(0),
                Mentsu::Koutsu(9),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: false,
            ..Default::default()
        };

        apply_yakuman(&mut res, &hand, &[], &ctx, &div, None, 31);

        assert!(res.yakuman_count >= 2);
        assert_eq!(res.han, 13 * res.yakuman_count);
        assert!(res.yaku_ids.contains(&ID_SUANKO_TANKI));
        assert!(!res.yaku_ids.is_empty());
    }

    #[test]
    fn apply_yakuman_detects_daisangen_and_daisuushii_variants() {
        let dragon_hand = hand_with_tiles(&[31, 31, 31, 32, 32, 32, 33, 33, 33, 0, 1, 2, 27, 27]);
        let dragon_div = Division {
            head: 27,
            body: vec![
                Mentsu::Koutsu(31),
                Mentsu::Koutsu(32),
                Mentsu::Koutsu(33),
                Mentsu::Shuntsu(0),
            ],
        };
        let mut dragon_res = YakuResult::default();
        apply_yakuman(
            &mut dragon_res,
            &dragon_hand,
            &[],
            &YakuContext3P::default(),
            &dragon_div,
            Some(3),
            2,
        );
        assert!(dragon_res.yaku_ids.contains(&ID_DAISANGEN));

        let wind_hand = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31]);
        let wind_div = Division {
            head: 31,
            body: vec![
                Mentsu::Koutsu(27),
                Mentsu::Koutsu(28),
                Mentsu::Koutsu(29),
                Mentsu::Koutsu(30),
            ],
        };
        let mut wind_res = YakuResult::default();
        apply_yakuman(
            &mut wind_res,
            &wind_hand,
            &[],
            &YakuContext3P::default(),
            &wind_div,
            Some(0),
            27,
        );
        assert!(wind_res.yaku_ids.contains(&ID_DAISUUSHI));
    }

    #[test]
    fn calculate_yaku_3p_surfaces_daisangen_and_daisuushii_results() {
        let dragon_hand = hand_with_tiles(&[31, 31, 31, 32, 32, 32, 33, 33, 33, 0, 1, 2, 27, 27]);
        let dragon_res = calculate_yaku_3p(&dragon_hand, &[], &YakuContext3P::default(), 2);
        assert!(dragon_res.han >= 13);
        assert!(dragon_res.yaku_ids.contains(&ID_DAISANGEN));

        let wind_hand = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31]);
        let wind_res = calculate_yaku_3p(&wind_hand, &[], &YakuContext3P::default(), 31);
        assert!(wind_res.han >= 26);
        assert!(wind_res.yaku_ids.contains(&ID_DAISUUSHI));
    }

    #[test]
    fn calculate_yaku_3p_distinguishes_kokushi_wait_variants() {
        let thirteen_wait = hand_with_tiles(&[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33]);
        let thirteen_wait_res =
            calculate_yaku_3p(&thirteen_wait, &[], &YakuContext3P::default(), 33);

        assert_eq!(thirteen_wait_res.han, 26);
        assert_eq!(thirteen_wait_res.yakuman_count, 2);
        assert!(thirteen_wait_res.yaku_ids.contains(&ID_KOKUSHI_13));

        let regular = hand_with_tiles(&[0, 0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]);
        let regular_res = calculate_yaku_3p(&regular, &[], &YakuContext3P::default(), 33);

        assert_eq!(regular_res.han, 13);
        assert_eq!(regular_res.yakuman_count, 1);
        assert!(regular_res.yaku_ids.contains(&ID_KOKUSHI));
    }

    #[test]
    fn calculate_yaku_3p_scores_chiitoitsu_chinitsu_path() {
        let hand = hand_with_tiles(&[1, 1, 2, 2, 4, 4, 5, 5, 7, 7, 8, 8, 6, 6]);

        let res = calculate_yaku_3p(&hand, &[], &YakuContext3P::default(), 7);

        assert_eq!(res.han, 8);
        assert_eq!(res.fu, 25);
        assert!(res.yaku_ids.contains(&ID_CHITOITSU));
        assert!(res.yaku_ids.contains(&ID_CHINITSU));
        assert!(!res.yaku_ids.contains(&ID_HONITSU));
        assert!(!res.yaku_ids.contains(&ID_HONROUTO));
    }

    #[test]
    fn calculate_yaku_3p_accumulates_standard_triplet_yaku_and_double_winds() {
        let hand = hand_with_tiles(&[0, 0, 0, 27, 27, 27, 31, 31, 31, 32, 32, 32, 33, 33]);
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: false,
            round_wind: 27,
            seat_wind: 27,
            ..Default::default()
        };

        let res = calculate_yaku_3p(&hand, &[], &ctx, 0);

        assert_eq!(res.yakuman_count, 0);
        assert!(res.han >= 12);
        assert!(res.yaku_ids.contains(&ID_HAKU));
        assert!(res.yaku_ids.contains(&ID_HATSU));
        assert!(res.yaku_ids.contains(&ID_BAKAZE));
        assert!(res.yaku_ids.contains(&ID_JIKAZE));
        assert!(res.yaku_ids.contains(&ID_SHOSANGEN));
        assert!(res.yaku_ids.contains(&ID_TOITOI));
        assert!(res.yaku_ids.contains(&ID_SANANKOU));
        assert!(res.yaku_ids.contains(&ID_HONITSU));
        assert!(res.yaku_ids.contains(&ID_HONROUTO));
    }

    #[test]
    fn check_pinfu_rejects_open_triplet_and_bad_wait_shapes() {
        let ctx = YakuContext3P {
            is_menzen: true,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };
        let div = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(6),
                Mentsu::Shuntsu(18),
            ],
        };
        let triplet_div = Division {
            head: 1,
            body: vec![
                Mentsu::Koutsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(6),
                Mentsu::Shuntsu(18),
            ],
        };

        assert!(!check_pinfu(
            &div,
            &[],
            &YakuContext3P {
                is_menzen: false,
                ..ctx
            },
            Some(0),
            0,
        ));
        assert!(!check_pinfu(
            &div,
            &[meld(MeldType::Chi, &[0, 1, 2], true)],
            &ctx,
            Some(0),
            0,
        ));
        assert!(!check_pinfu(&triplet_div, &[], &ctx, Some(0), 0));
        assert!(!check_pinfu(&div, &[], &ctx, Some(0), 1));
        assert!(!check_pinfu(&div, &[], &ctx, Some(0), 2));
        assert!(!check_pinfu(&div, &[], &ctx, Some(2), 6));
    }

    #[test]
    fn calculate_fu_with_waiting_distinguishes_ryanmen_kanchan_and_penchan() {
        let div = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(6),
                Mentsu::Shuntsu(18),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: false,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };

        assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 3), 30);
        assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 4), 40);
        assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(0), 2), 40);
        assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(2), 6), 40);
    }

    #[test]
    fn calculate_fu_with_waiting_promotes_open_twenty_fu_to_thirty() {
        let div = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: false,
            is_tsumo: false,
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };

        let fu = calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 3);

        assert_eq!(fu, 30);
    }

    #[test]
    fn calculate_fu_with_waiting_stacks_round_seat_and_dragon_pair_fu() {
        let div = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: false,
            round_wind: 31,
            seat_wind: 31,
            ..Default::default()
        };

        let fu = calculate_fu_with_waiting(&div, &[], &ctx, None, 31);

        assert_eq!(fu, 40);
    }

    #[test]
    fn sequence_and_triplet_helpers_consider_meld_types() {
        let ittsu_div = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Koutsu(31),
                Mentsu::Koutsu(32),
            ],
        };
        let doujun_div = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(9),
                Mentsu::Koutsu(31),
                Mentsu::Koutsu(32),
            ],
        };
        let doukou_div = Division {
            head: 31,
            body: vec![
                Mentsu::Koutsu(0),
                Mentsu::Koutsu(9),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(12),
            ],
        };

        assert!(check_ittsu(
            &ittsu_div,
            &[meld(MeldType::Chi, &[6, 7, 8], true)]
        ));
        assert!(is_sanshoku_doujun(
            &doujun_div,
            &[meld(MeldType::Chi, &[18, 19, 20], true)],
        ));
        assert!(!is_sanshoku_doujun(
            &doujun_div,
            &[meld(MeldType::Pon, &[18, 18, 18], true)],
        ));
        assert!(is_sanshoku_doukou(
            &doukou_div,
            &[meld(MeldType::Pon, &[18, 18, 18], true)],
        ));
        assert!(!is_sanshoku_doukou(
            &doukou_div,
            &[meld(MeldType::Chi, &[18, 19, 20], true)],
        ));
    }

    #[test]
    fn outside_helpers_consider_meld_terminal_and_honor_rules() {
        let junchan_div = Division {
            head: 0,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(6),
                Mentsu::Shuntsu(9),
                Mentsu::Koutsu(26),
            ],
        };

        assert!(is_junchan(
            &junchan_div,
            &[meld(MeldType::Pon, &[8, 8, 8], true)]
        ));
        assert!(!is_junchan(
            &junchan_div,
            &[meld(MeldType::Pon, &[31, 31, 31], true)]
        ));
        assert!(!is_chantai(&junchan_div, &[]));
        assert!(is_chantai(
            &junchan_div,
            &[meld(MeldType::Pon, &[31, 31, 31], true)],
        ));
    }

    #[test]
    fn apply_yakuman_detects_chiihou_without_other_yakuman() {
        let hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 31, 31]);
        let div = Division {
            head: 31,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            is_tsumo: true,
            is_tsumo_first_turn: true,
            seat_wind: 28,
            ..Default::default()
        };
        let mut res = YakuResult::default();

        apply_yakuman(&mut res, &hand, &[], &ctx, &div, Some(0), 2);

        assert_eq!(res.han, 13);
        assert_eq!(res.yakuman_count, 1);
        assert!(res.yaku_ids.contains(&ID_CHIHO));
    }

    #[test]
    fn apply_yakuman_detects_suukantsu_from_four_kans() {
        let hand = hand_with_tiles(&[1, 1]);
        let melds = [
            meld(MeldType::Ankan, &[0, 0, 0, 0], false),
            meld(MeldType::Daiminkan, &[9, 9, 9, 9], true),
            meld(MeldType::Kakan, &[18, 18, 18, 18], true),
            meld(MeldType::Ankan, &[27, 27, 27, 27], false),
        ];
        let mut res = YakuResult::default();

        apply_yakuman(
            &mut res,
            &hand,
            &melds,
            &YakuContext3P::default(),
            &Division {
                head: 1,
                body: Vec::new(),
            },
            None,
            1,
        );

        assert_eq!(res.han, 13);
        assert_eq!(res.yakuman_count, 1);
        assert!(res.yaku_ids.contains(&ID_SUKANTSU));
    }

    #[test]
    fn apply_yakuman_distinguishes_chuuren_and_junsei_variants() {
        let pure_hand = hand_with_tiles(&[0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);
        let pure_div = Division {
            head: 1,
            body: vec![
                Mentsu::Koutsu(0),
                Mentsu::Shuntsu(2),
                Mentsu::Shuntsu(5),
                Mentsu::Koutsu(8),
            ],
        };
        let ctx = YakuContext3P {
            is_menzen: true,
            ..Default::default()
        };
        let mut pure_res = YakuResult::default();

        apply_yakuman(&mut pure_res, &pure_hand, &[], &ctx, &pure_div, Some(1), 1);

        assert_eq!(pure_res.han, 26);
        assert_eq!(pure_res.yakuman_count, 2);
        assert!(pure_res.yaku_ids.contains(&ID_JUNSEI_CHUUREN));

        let regular_hand = hand_with_tiles(&[0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);
        let mut regular_res = YakuResult::default();

        apply_yakuman(
            &mut regular_res,
            &regular_hand,
            &[],
            &ctx,
            &pure_div,
            Some(1),
            7,
        );

        assert_eq!(regular_res.han, 13);
        assert_eq!(regular_res.yakuman_count, 1);
        assert!(regular_res.yaku_ids.contains(&ID_CHUUREN));
    }

    #[test]
    fn tile_classifier_helpers_distinguish_terminals_honors_and_yakuhai() {
        let ctx = YakuContext3P {
            round_wind: 28,
            seat_wind: 30,
            ..Default::default()
        };

        assert!(is_terminal(0));
        assert!(is_terminal(33));
        assert!(is_number_terminal(8));
        assert!(!is_number_terminal(27));
        assert!(is_honor(31));
        assert!(!is_honor(9));
        assert!(is_yakuhai_tile(31, &ctx));
        assert!(is_yakuhai_tile(28, &ctx));
        assert!(is_yakuhai_tile(30, &ctx));
        assert!(!is_yakuhai_tile(29, &ctx));
        assert!(!is_yakuhai_tile(8, &ctx));
    }

    #[test]
    fn flush_and_outside_helpers_reject_mixed_inside_shapes() {
        let mixed_flush = hand_with_tiles(&[0, 1, 2, 9, 10, 11, 27, 27, 28, 28, 29, 29, 31, 31]);
        let inside_shape = Division {
            head: 1,
            body: vec![
                Mentsu::Shuntsu(1),
                Mentsu::Shuntsu(4),
                Mentsu::Shuntsu(10),
                Mentsu::Koutsu(18),
            ],
        };

        assert!(!is_honitsu(&mixed_flush, &[]));
        assert!(!is_chinitsu(&mixed_flush, &[]));
        assert!(!is_honroutou(&mixed_flush, &[]));
        assert!(!is_junchan(&inside_shape, &[]));
        assert!(!is_chantai(&inside_shape, &[]));
    }

    #[test]
    fn calculate_yaku_3p_returns_empty_result_for_non_agari_shape() {
        let hand = hand_with_tiles(&[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 18, 19, 27, 31]);
        let res = calculate_yaku_3p(&hand, &[], &YakuContext3P::default(), 31);

        assert_eq!(res.han, 0);
        assert_eq!(res.fu, 0);
        assert_eq!(res.yakuman_count, 0);
        assert!(res.yaku_names.is_empty() || res.yaku_ids.len() == res.yaku_names.len());
    }

    #[test]
    fn yakuhai_and_terminal_helpers_reject_non_matching_tiles() {
        let ctx = YakuContext3P {
            round_wind: 27,
            seat_wind: 28,
            ..Default::default()
        };

        assert!(!is_terminal(1));
        assert!(!is_number_terminal(1));
        assert!(!is_honor(8));
        assert!(!is_yakuhai_tile(29, &ctx));
        assert!(!is_yakuhai_tile(0, &ctx));
    }

    #[test]
    fn tsuuiisou_chinroutou_and_ryuuiisou_reject_mixed_hands() {
        let mixed = hand_with_tiles(&[0, 0, 8, 8, 27, 27, 31, 31, 19, 19, 20, 20, 25, 25]);
        assert!(!is_tsuu_iisou(&mixed, &[]));
        assert!(!is_chinroutou(&mixed, &[]));
        assert!(!is_ryuu_iisou(&mixed, &[]));
    }

    #[test]
    fn chuuren_helpers_reject_honor_and_multi_suit_hands() {
        let honors = hand_with_tiles(&[27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 33, 33]);
        let mixed = hand_with_tiles(&[0, 0, 0, 1, 2, 3, 9, 9, 9, 10, 11, 12, 18, 18]);

        assert!(!is_chuuren_poutou(&honors));
        assert!(!is_chuuren_poutou(&mixed));
        assert!(!is_chuuren_9_wait(&mixed, 0));
    }

    #[test]
    fn apply_yakuman_leaves_result_empty_for_non_yakuman_hand() {
        let hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27]);
        let div = Division {
            head: 27,
            body: vec![
                Mentsu::Shuntsu(0),
                Mentsu::Shuntsu(3),
                Mentsu::Shuntsu(9),
                Mentsu::Shuntsu(18),
            ],
        };
        let mut res = YakuResult::default();

        apply_yakuman(
            &mut res,
            &hand,
            &[],
            &YakuContext3P::default(),
            &div,
            Some(0),
            2,
        );

        assert_eq!(res.han, 0);
        assert_eq!(res.yakuman_count, 0);
        assert!(res.yaku_names.is_empty() || res.yaku_ids.len() == res.yaku_names.len());
    }
}
