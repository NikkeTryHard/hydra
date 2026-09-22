use super::scoring_tables::{
    HAN_YAKUMAN, HanFu, HoraInput, MeldKind, OpenMeld, Payment, TILE_E, TILE_HAKU, TILE_N,
    WaitKind, WinKind, is_honor, is_simple, is_terminal_type,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosedShape {
    Shuntsu,
    Koutsu,
}

#[derive(Debug, Clone)]
pub(crate) struct Decomp {
    pub(crate) pair: u8,
    pub(crate) melds: Vec<(u8, ClosedShape)>,
}

fn split_into(
    counts: &mut [u8; 34],
    pair: Option<u8>,
    melds: &mut Vec<(u8, ClosedShape)>,
    target: usize,
    out: &mut Vec<Decomp>,
) {
    if out.len() >= 64 {
        return;
    }
    let mut t = 0usize;
    while t < 34 && counts[t] == 0 {
        t += 1;
    }
    if t == 34 {
        if melds.len() == target
            && let Some(p) = pair
        {
            out.push(Decomp {
                pair: p,
                melds: melds.clone(),
            });
        }
        return;
    }
    if pair.is_none() && counts[t] >= 2 {
        counts[t] -= 2;
        // proof: `t` in 0..34 (recursion guard `t == 34` returns above), fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t_u: u8 = t as u8;
        split_into(counts, Some(t_u), melds, target, out);
        counts[t] += 2;
    }
    if melds.len() < target && counts[t] >= 3 {
        counts[t] -= 3;
        // proof: `t` in 0..34 (recursion guard above), fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t_k: u8 = t as u8;
        melds.push((t_k, ClosedShape::Koutsu));
        split_into(counts, pair, melds, target, out);
        melds.pop();
        counts[t] += 3;
    }
    if melds.len() < target && t < 27 && t % 9 <= 6 && counts[t + 1] > 0 && counts[t + 2] > 0 {
        counts[t] -= 1;
        counts[t + 1] -= 1;
        counts[t + 2] -= 1;
        // proof: `t` in 0..27 here (`t < 27` guard above), fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t_s: u8 = t as u8;
        melds.push((t_s, ClosedShape::Shuntsu));
        split_into(counts, pair, melds, target, out);
        melds.pop();
        counts[t] += 1;
        counts[t + 1] += 1;
        counts[t + 2] += 1;
    }
}

/// Standard-form decompositions of the concealed takes into pair + `target`
/// closed melds (no chiitoi/kokushi here; callers check those first).
pub(crate) fn decompose(counts: &[u8; 34], target: usize) -> Vec<Decomp> {
    let mut counts = *counts;
    let mut melds = Vec::with_capacity(4);
    let mut out = Vec::new();
    split_into(&mut counts, None, &mut melds, target, &mut out);
    out
}

pub(crate) fn is_chiitoi(counts: &[u8; 34]) -> bool {
    let mut pairs = 0u8;
    for &c in counts {
        match c {
            2 => pairs += 1,
            0 => {}
            _ => return false,
        }
    }
    pairs == 7
}

pub(crate) fn is_kokushi(counts: &[u8; 34]) -> bool {
    const NEED: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut sum = 0u32;
    for &c in counts {
        sum += u32::from(c);
    }
    if sum != 14 {
        return false;
    }
    let mut doubled = 0u8;
    for (t, &c) in counts.iter().enumerate() {
        // proof: `t` in 0..34 (loop over `counts`), fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t_n: u8 = t as u8;
        if NEED.contains(&t_n) {
            if c == 2 {
                doubled += 1;
            } else if c != 1 {
                return false;
            }
        } else if c != 0 {
            return false;
        }
    }
    doubled == 1
}

pub(crate) fn is_chuuren(counts: &[u8; 34]) -> bool {
    let mut sum = 0u32;
    for &c in counts {
        sum += u32::from(c);
    }
    if sum != 14 {
        return false;
    }
    for suit in 0..3u8 {
        let b = (suit * 9) as usize;
        let c = &counts[b..b + 9];
        if c.iter().all(|&x| x == 0) {
            continue;
        }
        if c[0] >= 3 && c[8] >= 3 && c[1..8].iter().all(|&x| x >= 1) {
            return true;
        }
        return false;
    }
    false
}

pub(crate) fn is_all_green(counts: &[u8; 34]) -> bool {
    const GREEN: [u8; 6] = [19, 20, 21, 23, 25, 32];
    let mut sum = 0u32;
    for (t, &c) in counts.iter().enumerate() {
        sum += u32::from(c);
        // proof: `t` in 0..34 (loop over `counts`), fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t_g: u8 = t as u8;
        if c > 0 && !GREEN.contains(&t_g) {
            return false;
        }
    }
    sum == 14
}

fn pair_value_han(pair: u8, input: &HoraInput) -> u8 {
    let mut han = 0u8;
    if pair >= TILE_HAKU {
        han += 1;
    }
    if pair == input.seat_wind {
        han += 1;
    }
    if pair == input.round_wind {
        han += 1;
    }
    han
}

fn meld_koutsu_fu(tile_type: u8, closed: bool) -> u16 {
    let big = is_terminal_type(tile_type) || is_honor(tile_type);
    match (closed, big) {
        (true, true) => 8,
        (true, false) => 4,
        (false, true) => 4,
        (false, false) => 2,
    }
}

fn open_meld_fu(m: OpenMeld) -> u16 {
    let big = is_terminal_type(m.tile_type) || is_honor(m.tile_type);
    match m.kind {
        MeldKind::Chi => 0,
        MeldKind::Pon => {
            if big {
                4
            } else {
                2
            }
        }
        MeldKind::Daiminkan | MeldKind::Kakan => {
            if big {
                16
            } else {
                8
            }
        }
        MeldKind::Ankan => {
            if big {
                32
            } else {
                16
            }
        }
    }
}

/// Yaku + fu for one standard interpretation. `exclude_triplet` names a
/// closed triplet of the winning type treated as the ron-open meld for the
/// sanankou count (`None` = count all closed). Returns `None` when the
/// interpretation carries no yaku.
pub(crate) fn eval_standard(
    decomp: &Decomp,
    input: &HoraInput,
    exclude_triplet: Option<u8>,
    hand_open: bool,
) -> Option<HanFu> {
    let closed = &decomp.melds;
    // Yakuman scan (closed melds + open melds + pair).
    let mut closed_quads = 0u8;
    let mut quads = 0u8;
    for m in &input.open {
        match m.kind {
            MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                quads += 1;
                if m.is_closed() {
                    closed_quads += 1;
                }
            }
            MeldKind::Chi | MeldKind::Pon => {}
        }
    }
    let mut closed_triplets = 0u8;
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            if Some(*t) == exclude_triplet {
                continue;
            }
            closed_triplets += 1;
        }
    }
    closed_triplets += closed_quads;
    // Wind triplet tallies (open or closed).
    let mut wind3 = [0u8; 4];
    let mut drag3 = [0u8; 3];
    let mut all_meld_koutsu = true;
    for (t, shape) in closed {
        if *shape == ClosedShape::Shuntsu {
            all_meld_koutsu = false;
        } else if *t >= TILE_E && *t <= TILE_N {
            wind3[(t - TILE_E) as usize] += 1;
        } else if *t >= TILE_HAKU {
            drag3[(t - TILE_HAKU) as usize] += 1;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => all_meld_koutsu = false,
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if m.tile_type >= TILE_E && m.tile_type <= TILE_N {
                    wind3[(m.tile_type - TILE_E) as usize] += 1;
                } else if m.tile_type >= TILE_HAKU {
                    drag3[(m.tile_type - TILE_HAKU) as usize] += 1;
                }
            }
        }
    }
    let mut yakuman = 0u8;
    if drag3 == [1, 1, 1] {
        yakuman += 1;
    }
    if closed_triplets + quads - closed_quads >= 4
        && input.win == WinKind::Tsumo
        && quads == closed_quads
    {
        // Four closed triplets/quads on tsumo (quads all ankan).
        yakuman += 1;
        if input.wait == WaitKind::Tanki {
            yakuman += 1;
        }
    }
    if quads == 4 {
        yakuman += 1;
    }
    if wind3 == [1, 1, 1, 1] {
        yakuman += 1;
    }
    if (wind3[0] + wind3[1] + wind3[2] + wind3[3] == 3)
        && (decomp.pair >= TILE_E && decomp.pair <= TILE_N)
    {
        yakuman += 1;
    }
    // All-terminals / all-honors / all-green over concealed + open takes.
    let mut concealed_all_term = true;
    let mut concealed_all_honor = true;
    for t in 0..34u8 {
        if input.concealed[t as usize] == 0 {
            continue;
        }
        if !is_terminal_type(t) {
            concealed_all_term = false;
        }
        if !is_honor(t) {
            concealed_all_honor = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                concealed_all_term = false;
                concealed_all_honor = false;
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if !is_terminal_type(m.tile_type) {
                    concealed_all_term = false;
                }
                if !is_honor(m.tile_type) {
                    concealed_all_honor = false;
                }
            }
        }
    }
    if concealed_all_term && input.open.iter().all(|m| m.kind != MeldKind::Chi) {
        yakuman += 1;
    }
    if concealed_all_honor {
        yakuman += 1;
    }
    if yakuman > 0 {
        return Some(HanFu {
            han: HAN_YAKUMAN.saturating_mul(yakuman),
            fu: 25,
        });
    }

    // Normal yaku.
    let mut han: u8 = 0;
    let closed_hand = !hand_open;
    // Pinfu first (fu depends on it).
    let pair_valueless = pair_value_han(decomp.pair, input) == 0;
    let all_seq = closed.iter().all(|(_, s)| *s == ClosedShape::Shuntsu)
        && input.open.iter().all(|m| m.kind == MeldKind::Chi);
    let pinfu = closed_hand && all_seq && pair_valueless && input.wait == WaitKind::Ryanmen;
    if pinfu {
        han += 1;
    }
    if input.win == WinKind::Tsumo && closed_hand {
        han += 1;
    }
    if input.riichi && closed_hand {
        han += if input.double_riichi { 2 } else { 1 };
    }
    if input.ippatsu && closed_hand && (input.riichi || input.double_riichi) {
        han += 1;
    }
    // Tanyao: every set all-simples.
    let mut meld_simple = true;
    for (t, _) in closed {
        if !is_simple(*t) {
            meld_simple = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                if !is_simple(m.tile_type)
                    || !is_simple(m.tile_type + 1)
                    || !is_simple(m.tile_type + 2)
                {
                    meld_simple = false;
                }
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if !is_simple(m.tile_type) {
                    meld_simple = false;
                }
            }
        }
    }
    if meld_simple && is_simple(decomp.pair) {
        han += 1;
    }
    // Yakuhai triplets/quads (open or closed).
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            han += pair_value_han(*t, input);
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {}
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                han += pair_value_han(m.tile_type, input);
            }
        }
    }
    // Toitoi / sanankou / sankantsu.
    if all_meld_koutsu {
        han += 2;
    }
    if closed_triplets == 3 {
        han += 2;
    }
    if quads == 3 {
        han += 2;
    }
    // Sanshoku doujun (open or closed sequences count; han drops when open).
    let mut seq_present = [[false; 7]; 3];
    for (t, shape) in closed {
        if *shape == ClosedShape::Shuntsu && *t < 27 {
            seq_present[(t / 9) as usize][(t % 9) as usize] = true;
        }
    }
    for m in &input.open {
        if m.kind == MeldKind::Chi && m.tile_type < 27 {
            seq_present[(m.tile_type / 9) as usize][(m.tile_type % 9) as usize] = true;
        }
    }
    let mut sanshoku = false;
    for (n, _) in seq_present[0].iter().enumerate().take(7) {
        if seq_present[0][n] && seq_present[1][n] && seq_present[2][n] {
            sanshoku = true;
        }
    }
    if sanshoku {
        han += if closed_hand { 2 } else { 1 };
    }
    // Ittsu.
    let mut ittsu = false;
    for s in 0..3u8 {
        if seq_present[s as usize][0] && seq_present[s as usize][3] && seq_present[s as usize][6] {
            ittsu = true;
        }
    }
    if ittsu {
        han += if closed_hand { 2 } else { 1 };
    }
    // Chanta / junchan / honroutou.
    let set_has_term_honor = |t: u8, seq: bool| -> bool {
        if seq {
            is_terminal_type(t) || is_terminal_type(t + 2)
        } else {
            is_terminal_type(t) || is_honor(t)
        }
    };
    let set_all_term = |t: u8, seq: bool| -> bool {
        if seq {
            t.is_multiple_of(9) || (t % 9 == 6)
        } else {
            is_terminal_type(t)
        }
    };
    let mut chanta = set_has_term_honor(decomp.pair, false);
    let mut junchan = is_terminal_type(decomp.pair);
    let mut honroutou = is_terminal_type(decomp.pair) || is_honor(decomp.pair);
    for (t, shape) in closed {
        let seq = *shape == ClosedShape::Shuntsu;
        chanta &= set_has_term_honor(*t, seq);
        junchan &= set_all_term(*t, seq);
        honroutou &= is_terminal_type(*t) || is_honor(*t);
        if seq {
            honroutou = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                chanta &= set_has_term_honor(m.tile_type, true);
                junchan &= set_all_term(m.tile_type, true);
                honroutou = false;
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                chanta &= set_has_term_honor(m.tile_type, false);
                junchan &= set_all_term(m.tile_type, false);
                honroutou &= is_terminal_type(m.tile_type) || is_honor(m.tile_type);
            }
        }
    }
    if honroutou {
        han += 2;
        chanta = true;
        junchan = false;
    }
    if junchan {
        han += if closed_hand { 3 } else { 2 };
    } else if chanta {
        han += if closed_hand { 2 } else { 1 };
    }
    // Iipeikou / ryanpeikou (closed sequences only).
    if closed_hand {
        let mut seq_counts = [0u8; 27];
        for (t, shape) in closed {
            if *shape == ClosedShape::Shuntsu {
                seq_counts[*t as usize] += 1;
            }
        }
        let mut pairs2 = 0u8;
        let mut single = false;
        for &c in &seq_counts {
            if c == 2 {
                pairs2 += 1;
            } else if c == 4 {
                pairs2 += 2;
            } else if c > 2 {
                single = true;
            }
        }
        if pairs2 == 2 && !single {
            han += 3;
        } else if pairs2 == 1 || single {
            han += 1;
        }
    }
    // Honitsu / chinitsu.
    let mut suits = [false; 4];
    for t in 0..34u8 {
        if input.concealed[t as usize] > 0 {
            suits[if t < 27 { (t / 9) as usize } else { 3 }] = true;
        }
    }
    for m in &input.open {
        if m.kind == MeldKind::Chi {
            suits[(m.tile_type / 9) as usize] = true;
        } else if is_honor(m.tile_type) {
            suits[3] = true;
        } else {
            suits[(m.tile_type / 9) as usize] = true;
        }
    }
    let suit_count = suits[0] as u8 + suits[1] as u8 + suits[2] as u8;
    if suit_count == 1 && suits[3] {
        han += if closed_hand { 3 } else { 2 };
    } else if suit_count == 1 {
        han += if closed_hand { 6 } else { 5 };
    }
    // Shousangen: two dragon triplets + dragon pair.
    let mut drag_pongs = 0u8;
    for d in 0..3u8 {
        if drag3[d as usize] > 0 {
            drag_pongs += 1;
        }
    }
    if drag_pongs == 2 && decomp.pair >= TILE_HAKU {
        han += 2;
    }
    // Situational 1-han.
    if input.rinshan {
        han += 1;
    }
    if input.chankan {
        han += 1;
    }
    if input.haitei {
        han += 1;
    }
    if input.houtei {
        han += 1;
    }
    han += input.dora;
    if han == 0 {
        return None;
    }
    if han >= HAN_YAKUMAN {
        return Some(HanFu {
            han: HAN_YAKUMAN,
            fu: 25,
        });
    }

    // Fu.
    let mut fu: u16 = 20;
    if input.win == WinKind::Ron && closed_hand {
        fu += 10;
    }
    if input.win == WinKind::Tsumo && !(pinfu && closed_hand) {
        fu += 2;
    }
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            let ron_open = Some(*t) == exclude_triplet;
            fu += meld_koutsu_fu(*t, !ron_open);
        }
    }
    for m in &input.open {
        fu += open_meld_fu(*m);
    }
    fu += u16::from(pair_value_han(decomp.pair, input)) * 2;
    match input.wait {
        WaitKind::Ryanmen | WaitKind::Shanpon => {}
        WaitKind::Kanchan | WaitKind::Penchan | WaitKind::Tanki => fu += 2,
    }
    fu = fu.div_ceil(10) * 10;
    if fu < 30 {
        fu = 30;
    }
    Some(HanFu { han, fu })
}

/// Chiitoi scoring (closed only): 25 fu + 2 han + combinable subset
/// (tanyao / honitsu / chinitsu / riichi family / tsumo / dora).
pub(crate) fn eval_chiitoi(input: &HoraInput) -> Option<HanFu> {
    if !input.open.is_empty() {
        return None;
    }
    let mut han: u8 = 2;
    let mut all_simple = true;
    let mut suits = [false; 4];
    for t in 0..34u8 {
        if input.concealed[t as usize] > 0 {
            if !is_simple(t) {
                all_simple = false;
            }
            suits[if t < 27 { (t / 9) as usize } else { 3 }] = true;
        }
    }
    if all_simple {
        han += 1;
    }
    let suit_count = suits[0] as u8 + suits[1] as u8 + suits[2] as u8;
    if suit_count == 1 && suits[3] {
        han += 3;
    } else if suit_count == 1 {
        han += 6;
    }
    if input.win == WinKind::Tsumo {
        han += 1;
    }
    if input.riichi {
        han += if input.double_riichi { 2 } else { 1 };
    }
    if input.ippatsu && (input.riichi || input.double_riichi) {
        han += 1;
    }
    if input.rinshan || input.chankan || input.haitei || input.houtei {
        han += 1;
    }
    han += input.dora;
    if han >= HAN_YAKUMAN {
        return Some(HanFu {
            han: HAN_YAKUMAN,
            fu: 25,
        });
    }
    Some(HanFu { han, fu: 25 })
}

pub(crate) fn winner_receipt(payment: Payment, dealer: bool, tsumo: bool) -> u32 {
    if !tsumo {
        payment.ron
    } else if dealer {
        payment.tsumo_oya * 3
    } else {
        payment.tsumo_ko * 2 + payment.tsumo_oya
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::super::scoring_tables::TILE_S;
    use super::*;
    use crate::sink::{Reconcile, ScoreOutcome, hora_payment, reconcile_hora, score_hora};

    // -- han/fu pins (standard scoring anchors) --

    fn base_input() -> HoraInput {
        HoraInput {
            concealed: [0u8; 34],
            win_type: 0,
            win: WinKind::Ron,
            ron_target: Some(1),
            wait: WaitKind::Ryanmen,
            open: Vec::new(),
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            dora: 0,
            rinshan: false,
            chankan: false,
            haitei: false,
            houtei: false,
            seat_wind: TILE_S,
            round_wind: TILE_E,
            dealer: false,
            dealer_seat: 1,
        }
    }

    fn set_counts(input: &mut HoraInput, types: &[u8]) {
        for t in types {
            input.concealed[*t as usize] += 1;
        }
    }

    #[test]
    fn pinfu_closed_ron_scores_1han_30fu() {
        // 234m 123p 345s 678s + pair 5m, ron on 8s ryanmen.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 25;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 1, fu: 30 });
                assert_eq!(payment.ron, 1000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
        // Same hand + 2 dora = 3 han 30 fu ron = 3900.
        input.dora = 2;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 3, fu: 30 });
                assert_eq!(payment.ron, 3900);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn tanyao_closed_tsumo_scores_2han_30fu() {
        // 234m 345p 555s 678s + pair 6p, tsumo on 8s ryanmen.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 10, 11, 12, 21, 21, 21, 23, 24, 25, 14, 14],
        );
        input.win_type = 25;
        input.win = WinKind::Tsumo;
        input.ron_target = None;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 2, fu: 30 });
                assert_eq!(payment.tsumo_ko, 500);
                assert_eq!(payment.tsumo_oya, 1000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn yakuhai_triplet_scores_fu_and_han() {
        // Haku pon + 234m + 345p + 567s + pair 6p, ron on 7s.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[31, 31, 31, 1, 2, 3, 10, 11, 12, 22, 23, 24, 14, 14],
        );
        input.win_type = 24;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 1, fu: 40 });
                assert_eq!(payment.ron, 1300);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn chiitoi_scores_fixed_25fu() {
        // Seven pairs incl honors, ron tanki.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 4, 4, 10, 10, 14, 14, 20, 20, 24, 24, 27, 27],
        );
        input.win_type = 27;
        input.wait = WaitKind::Tanki;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 2, fu: 25 });
                assert_eq!(payment.ron, 1600);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }
    #[test]
    fn reconcile_tsumo_applies_honba_and_kyotaku() {
        // Chiitoi tsumo, no riichi: 2 + tsumo = 3 han 25 fu -> ko 800 /
        // oya 1600. Winner seat 0 (child), dealer seat 1, honba 1 (+100 per
        // payer), kyotaku 2 (+2000 to winner).
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 4, 4, 10, 10, 14, 14, 20, 20, 24, 24, 27, 27],
        );
        input.win_type = 27;
        input.wait = WaitKind::Tanki;
        input.win = WinKind::Tsumo;
        input.ron_target = None;
        assert_eq!(
            reconcile_hora(&input, 0, [5500, -1700, -900, -900], 1, 2),
            Reconcile::Match { han: 3, fu: 25 }
        );
        // Dealer/consistency mismatch fails loud.
        match reconcile_hora(&input, 1, [5500, -1700, -900, -900], 1, 2) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
    }

    #[test]
    fn toitoi_ron_downgrade_scores_sanankou_mangan() {
        // Four closed triplets + pair, ron: suuankou shape downgrades to
        // sanankou + toitoi = 4 han; fu: 20+10+(8+4+4)+4 ron-open -> 50 -> mangan 8000.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 0, 10, 10, 10, 20, 20, 20, 27, 27, 27, 5, 5],
        );
        input.win_type = 27;
        input.wait = WaitKind::Shanpon;
        input.seat_wind = TILE_S;
        input.round_wind = TILE_N;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 4, fu: 50 });
                assert_eq!(payment.ron, 8000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn payment_table_pins_mangan_family() {
        assert_eq!(hora_payment(3, 60, false, false).ron, 7700);
        assert_eq!(hora_payment(3, 70, false, false).ron, 8000);
        assert_eq!(hora_payment(5, 30, false, false).ron, 8000);
        assert_eq!(hora_payment(5, 30, true, false).ron, 12000);
        assert_eq!(hora_payment(6, 40, false, false).ron, 12000);
        assert_eq!(hora_payment(8, 30, false, false).ron, 16000);
        assert_eq!(hora_payment(11, 30, false, false).ron, 24000);
        assert_eq!(hora_payment(13, 25, false, false).ron, 32000);
        assert_eq!(hora_payment(13, 25, true, false).ron, 48000);
        let mangan_child = hora_payment(3, 70, false, true);
        assert_eq!(
            (mangan_child.tsumo_ko, mangan_child.tsumo_oya),
            (2000, 4000)
        );
        let mangan_dealer = hora_payment(4, 40, true, true);
        assert_eq!(
            (mangan_dealer.tsumo_ko, mangan_dealer.tsumo_oya),
            (2000, 4000)
        );
    }

    #[test]
    fn reconcile_matches_logged_deltas() {
        // Yakuhai vector above: child ron 1300 off seat 1.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[31, 31, 31, 1, 2, 3, 10, 11, 12, 22, 23, 24, 14, 14],
        );
        input.win_type = 24;
        assert_eq!(
            reconcile_hora(&input, 0, [1300, -1300, 0, 0], 0, 0),
            Reconcile::Match { han: 1, fu: 40 }
        );
        // Wrong deltas -> Mismatch with the expected movement.
        match reconcile_hora(&input, 0, [8000, -8000, 0, 0], 0, 0) {
            Reconcile::Mismatch { han, fu, expected } => {
                assert_eq!((han, fu), (1, 40));
                assert_eq!(expected, [1300, -1300, 0, 0]);
            }
            other => panic!("must mismatch, got {other:?}"),
        }
    }

    #[test]
    fn open_yakuless_hand_needs_manual() {
        // Open pinfu shape, no dora, no yaku: fail-loud, never a guess.
        let mut input = base_input();
        set_counts(&mut input, &[1, 2, 3, 10, 11, 12, 23, 24, 25, 4, 4]);
        input.win_type = 25;
        input.open.push(OpenMeld {
            kind: MeldKind::Pon,
            tile_type: TILE_E,
        });
        input.seat_wind = TILE_S;
        input.round_wind = TILE_N;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "no-yaku"),
            ScoreOutcome::Known { han_fu, .. } => panic!("must not score: {han_fu:?}"),
        }
        match reconcile_hora(&input, 0, [1000, -1000, 0, 0], 0, 0) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "no-yaku"),
            other => panic!("must propagate manual, got {other:?}"),
        }
    }

    #[test]
    fn bad_inputs_need_manual_never_panic() {
        let mut input = base_input();
        // Concealed sum != 14.
        set_counts(&mut input, &[1, 2, 3]);
        input.win_type = 3;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
        // Win tile absent from concealed.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 5;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
        // Winner seat out of range reconciles manual.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 25;
        match reconcile_hora(&input, 9, [0, 0, 0, 0], 0, 0) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
    }
}
