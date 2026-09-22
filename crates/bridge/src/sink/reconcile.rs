use super::scoring_eval::{
    ClosedShape, decompose, eval_chiitoi, eval_standard, is_all_green, is_chiitoi, is_chuuren,
    is_kokushi, winner_receipt,
};
use super::scoring_tables::{
    HAN_YAKUMAN, HanFu, HoraInput, MeldKind, Payment, Reconcile, ScoreOutcome, TILE_E, TILE_HATSU,
    TILE_N, WinKind, hora_payment,
};

/// Cold terminal-hora recompute: enumerate closed-form interpretations of
/// the concealed takes and score the best by winner receipt. `NeedsManual`
/// on bad input, yaku-less hands, or unimplemented (exotic) shapes —
/// fail-loud, never a guess.
pub fn score_hora(input: &HoraInput) -> ScoreOutcome {
    if input.win_type >= 34
        || input.open.len() > 4
        || input.dealer_seat > 3
        || !(TILE_E..=TILE_N).contains(&input.seat_wind)
        || !(TILE_E..=TILE_N).contains(&input.round_wind)
        || input.concealed[input.win_type as usize] == 0
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    if (input.haitei && input.win != WinKind::Tsumo)
        || (input.houtei && input.win != WinKind::Ron)
        || (input.rinshan && input.win != WinKind::Tsumo)
        || (input.chankan && input.win != WinKind::Ron)
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    if (input.win == WinKind::Tsumo && input.ron_target.is_some())
        || (input.win == WinKind::Ron && input.ron_target.map(|s| s > 3).unwrap_or(true))
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    let mut open_takes = 0u32;
    for m in &input.open {
        if m.tile_type >= 34 {
            return ScoreOutcome::NeedsManual { why: "bad-input" };
        }
        open_takes += u32::from(m.takes());
    }
    let mut concealed_sum = 0u32;
    for t in 0..34 {
        let c = input.concealed[t];
        if c > 4 {
            return ScoreOutcome::NeedsManual { why: "bad-input" };
        }
        concealed_sum += u32::from(c);
    }
    if concealed_sum + open_takes != 14 {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    let hand_open = !input.open.is_empty();

    // Kokushi (closed thirteen-orphans): immediate yakuman.
    if !hand_open && is_kokushi(&input.concealed) {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // Chuuren (closed nine-gates): immediate yakuman.
    if !hand_open && is_chuuren(&input.concealed) {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // All-green (ryuuiisou): immediate yakuman.
    if is_all_green(&input.concealed)
        && input.open.iter().all(|m| {
            m.kind == MeldKind::Chi
                || ((m.tile_type == 19
                    || m.tile_type == 20
                    || m.tile_type == 21
                    || m.tile_type == 23
                    || m.tile_type == 25
                    || m.tile_type == TILE_HATSU)
                    && m.kind != MeldKind::Chi)
        })
    {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // Chiitoi (closed seven-pairs).
    if !hand_open && is_chiitoi(&input.concealed) {
        if let Some(han_fu) = eval_chiitoi(input) {
            let payment = hora_payment(
                han_fu.han,
                han_fu.fu,
                input.dealer,
                input.win == WinKind::Tsumo,
            );
            return ScoreOutcome::Known { han_fu, payment };
        }
        return ScoreOutcome::NeedsManual { why: "no-yaku" };
    }

    // Standard form: closed melds + pair.
    let target = 4usize.saturating_sub(input.open.len());
    let decomps = decompose(&input.concealed, target);
    if decomps.is_empty() {
        return ScoreOutcome::NeedsManual { why: "exotic" };
    }
    let tsumo = input.win == WinKind::Tsumo;
    let mut best: Option<(HanFu, Payment)> = None;
    for decomp in &decomps {
        // Ron openness variants for the sanankou count: the winning tile
        // may complete a triplet (open) or a sequence (triplets all closed).
        let mut variants = vec![None];
        if !tsumo {
            for (t, shape) in &decomp.melds {
                if *shape == ClosedShape::Koutsu && *t == input.win_type {
                    variants.push(Some(*t));
                    break;
                }
            }
        }
        for exclude in variants {
            if let Some(han_fu) = eval_standard(decomp, input, exclude, hand_open) {
                let payment = hora_payment(han_fu.han, han_fu.fu, input.dealer, tsumo);
                let receipt = winner_receipt(payment, input.dealer, tsumo);
                let replace = match best {
                    None => true,
                    Some((b, p)) => {
                        receipt > winner_receipt(p, input.dealer, tsumo)
                            || (receipt == winner_receipt(p, input.dealer, tsumo)
                                && han_fu.fu > b.fu)
                    }
                };
                if replace {
                    best = Some((han_fu, payment));
                }
            }
        }
    }
    match best {
        Some((han_fu, payment)) => ScoreOutcome::Known { han_fu, payment },
        None => ScoreOutcome::NeedsManual { why: "no-yaku" },
    }
}

/// G4 scores-vs-delta reconciliation (cold, report-only): recompute the
/// terminal hora and compare the winner-indexed score movement against the
/// logged `deltas` (honba/kyotaku sticks applied standardly). `Mismatch` is
/// cold review, never a hot verdict.
pub fn reconcile_hora(
    input: &HoraInput,
    winner: usize,
    deltas: [i32; 4],
    honba: u32,
    kyotaku: u32,
) -> Reconcile {
    if winner > 3 || (winner == usize::from(input.dealer_seat)) != input.dealer {
        return Reconcile::NeedsManual { why: "bad-input" };
    }
    let (han_fu, payment) = match score_hora(input) {
        ScoreOutcome::Known { han_fu, payment } => (han_fu, payment),
        ScoreOutcome::NeedsManual { why } => return Reconcile::NeedsManual { why },
    };
    let tsumo = input.win == WinKind::Tsumo;
    let mut expected = [0i32; 4];
    if tsumo {
        for p in 0..4 {
            if p == winner {
                continue;
            }
            let base = if p == usize::from(input.dealer_seat) {
                payment.tsumo_oya
            } else {
                payment.tsumo_ko
            };
            let pay = base + 100 * honba;
            // proof: hora payments are small (< 100k) + sticks, far below 2^31, fit `i32`.
            #[allow(clippy::cast_possible_wrap)]
            let pay_i: i32 = pay as i32;
            expected[p] -= pay_i;
            expected[winner] += pay_i;
        }
        // proof: kyotaku sticks are small counts, `1000*kyotaku` far below 2^31, fits `i32`.
        #[allow(clippy::cast_possible_wrap)]
        let kyo_i: i32 = (1000 * kyotaku) as i32;
        expected[winner] += kyo_i;
    } else {
        let target = match input.ron_target {
            Some(s) if (s as usize) < 4 && (s as usize) != winner => s as usize,
            _ => return Reconcile::NeedsManual { why: "bad-input" },
        };
        let gain = payment.ron + 300 * honba + 1000 * kyotaku;
        // proof: hora payments + sticks are small (< 100k), far below 2^31, fit `i32`.
        #[allow(clippy::cast_possible_wrap)]
        let gain_i: i32 = gain as i32;
        expected[target] -= gain_i;
        expected[winner] += gain_i;
    }
    if expected == deltas {
        Reconcile::Match {
            han: han_fu.han,
            fu: han_fu.fu,
        }
    } else {
        Reconcile::Mismatch {
            han: han_fu.han,
            fu: han_fu.fu,
            expected,
        }
    }
}
