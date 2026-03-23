use criterion::{black_box, criterion_group, criterion_main, Criterion};
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::hand_evaluator_3p::HandEvaluator3P;
use riichienv_core::score;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};
use serde::Deserialize;
use std::fs;

// --- Deserialization types for benchmark data ---

#[derive(Deserialize)]
struct BenchData {
    cases: Vec<AgariCase>,
}

#[derive(Deserialize)]
struct AgariCase {
    tiles_136: Vec<u8>,
    melds: Vec<BenchMeld>,
    win_tile_136: u8,
    dora_indicators: Vec<u8>,
    ura_indicators: Vec<u8>,
    conditions: BenchConditions,
    #[allow(dead_code)]
    expected: BenchExpected,
}

#[derive(Deserialize)]
struct BenchMeld {
    meld_type: String,
    tiles: Vec<u8>,
    opened: bool,
    from_who: i8,
}

#[derive(Deserialize)]
struct BenchConditions {
    tsumo: bool,
    riichi: bool,
    double_riichi: bool,
    ippatsu: bool,
    haitei: bool,
    houtei: bool,
    rinshan: bool,
    chankan: bool,
    tsumo_first_turn: bool,
    player_wind: u8,
    round_wind: u8,
    honba: u32,
    #[serde(default)]
    kita_count: u8,
    #[serde(default)]
    is_sanma: bool,
    #[serde(default = "default_num_players")]
    num_players: u8,
}

fn default_num_players() -> u8 {
    4
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct BenchExpected {
    is_win: bool,
    han: u32,
    fu: u32,
    yaku: Vec<u32>,
}

type HandEvaluatorBenchInput = (HandEvaluator, u8, Vec<u8>, Vec<u8>, Conditions);
type HandEvaluator3PBenchInput = (HandEvaluator3P, u8, Vec<u8>, Vec<u8>, Conditions);
type ScoreBenchCase = (u8, u8, bool, bool, u32, u8);

// --- Helper functions ---

fn parse_meld_type(s: &str) -> MeldType {
    match s {
        "chi" => MeldType::Chi,
        "pon" => MeldType::Pon,
        "daiminkan" => MeldType::Daiminkan,
        "ankan" => MeldType::Ankan,
        "kakan" => MeldType::Kakan,
        _ => panic!("Unknown meld type: {}", s),
    }
}

fn to_meld(bm: &BenchMeld) -> Meld {
    Meld::new(
        parse_meld_type(&bm.meld_type),
        &bm.tiles,
        bm.opened,
        bm.from_who,
        None,
    )
}

fn to_conditions(bc: &BenchConditions) -> Conditions {
    Conditions {
        tsumo: bc.tsumo,
        riichi: bc.riichi,
        double_riichi: bc.double_riichi,
        ippatsu: bc.ippatsu,
        haitei: bc.haitei,
        houtei: bc.houtei,
        rinshan: bc.rinshan,
        chankan: bc.chankan,
        tsumo_first_turn: bc.tsumo_first_turn,
        player_wind: Wind::from(bc.player_wind),
        round_wind: Wind::from(bc.round_wind),
        honba: bc.honba,
        riichi_sticks: 0,
        kita_count: bc.kita_count,
        is_sanma: bc.is_sanma,
        num_players: bc.num_players,
    }
}

fn load_agari_data(path: &str) -> BenchData {
    let data =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("Failed to parse {}: {}", path, e))
}

// --- Benchmarks ---

fn bench_hand_evaluator_calc_4p(c: &mut Criterion) {
    let data = load_agari_data("benches/data/agari_4p.json");

    // Pre-build evaluator inputs
    let inputs: Vec<HandEvaluatorBenchInput> = data
        .cases
        .iter()
        .map(|case| {
            let melds: Vec<Meld> = case.melds.iter().map(to_meld).collect();
            let he = HandEvaluator::new(&case.tiles_136, &melds);
            let conds = to_conditions(&case.conditions);
            (
                he,
                case.win_tile_136,
                case.dora_indicators.clone(),
                case.ura_indicators.clone(),
                conds,
            )
        })
        .collect();

    c.bench_function("hand_evaluator/calc_4p", |b| {
        b.iter(|| {
            for (he, win_tile, dora, ura, conds) in &inputs {
                black_box(he.calc(*win_tile, dora, ura, Some(conds.clone())));
            }
        })
    });
}

fn bench_hand_evaluator_calc_3p(c: &mut Criterion) {
    let path = "benches/data/agari_3p.json";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let data = load_agari_data(path);

    let inputs: Vec<HandEvaluator3PBenchInput> = data
        .cases
        .iter()
        .map(|case| {
            let melds: Vec<Meld> = case.melds.iter().map(to_meld).collect();
            let he = HandEvaluator3P::new(&case.tiles_136, &melds);
            let conds = to_conditions(&case.conditions);
            (
                he,
                case.win_tile_136,
                case.dora_indicators.clone(),
                case.ura_indicators.clone(),
                conds,
            )
        })
        .collect();

    c.bench_function("hand_evaluator/calc_3p", |b| {
        b.iter(|| {
            for (he, win_tile, dora, ura, conds) in &inputs {
                black_box(he.calc(*win_tile, dora, ura, Some(conds.clone())));
            }
        })
    });
}

fn bench_calculate_score(c: &mut Criterion) {
    // Representative han/fu combinations covering all scoring tiers
    let test_cases: Vec<ScoreBenchCase> = vec![
        // (han, fu, is_oya, is_tsumo, honba, num_players)
        (1, 30, false, false, 0, 4), // 1 han 30 fu
        (1, 40, true, true, 0, 4),   // 1 han 40 fu oya tsumo
        (2, 30, false, false, 1, 4), // 2 han 30 fu 1 honba
        (2, 40, false, true, 0, 4),  // 2 han 40 fu tsumo
        (3, 30, false, false, 0, 4), // 3 han 30 fu
        (3, 40, true, false, 0, 4),  // 3 han 40 fu oya
        (4, 30, false, false, 0, 4), // 4 han 30 fu (mangan boundary)
        (4, 40, false, true, 2, 4),  // 4 han 40 fu tsumo
        (5, 0, false, false, 0, 4),  // mangan
        (6, 0, true, true, 0, 4),    // haneman oya tsumo
        (8, 0, false, false, 0, 4),  // baiman
        (11, 0, false, true, 0, 4),  // sanbaiman tsumo
        (13, 0, false, false, 0, 4), // yakuman
        (13, 0, true, true, 0, 4),   // yakuman oya tsumo
        (26, 0, false, false, 0, 4), // double yakuman ron
        (26, 0, true, true, 0, 4),   // double yakuman oya tsumo
        (26, 0, false, true, 0, 4),  // double yakuman ko tsumo
        (26, 0, false, false, 2, 4), // double yakuman ron 2 honba
        (39, 0, false, false, 0, 4), // triple yakuman ron
        (39, 0, true, true, 0, 4),   // triple yakuman oya tsumo
        (52, 0, false, false, 0, 4), // quadruple yakuman ron
        (65, 0, false, true, 0, 4),  // quintuple yakuman tsumo
        // 3-player variants
        (5, 0, false, false, 0, 3),  // mangan 3p
        (5, 0, true, true, 0, 3),    // mangan oya tsumo 3p
        (13, 0, false, false, 0, 3), // yakuman 3p
        (26, 0, false, false, 0, 3), // double yakuman 3p
        (26, 0, true, true, 0, 3),   // double yakuman oya tsumo 3p
        (39, 0, false, true, 0, 3),  // triple yakuman ko tsumo 3p
    ];

    c.bench_function("calculate_score", |b| {
        b.iter(|| {
            for &(han, fu, is_oya, is_tsumo, honba, num_players) in &test_cases {
                black_box(score::calculate_score(
                    han,
                    fu,
                    is_oya,
                    is_tsumo,
                    honba,
                    num_players,
                ));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_hand_evaluator_calc_4p,
    bench_hand_evaluator_calc_3p,
    bench_calculate_score,
);
criterion_main!(benches);
