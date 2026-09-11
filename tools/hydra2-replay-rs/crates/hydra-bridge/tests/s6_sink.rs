//! S6 sink end-to-end (P4-A): feed `QuarantineStub` → bridge `FeedQuar`/`PyQuar`
//! + lineage file + `games_quarantined` accounting.
//!
//! - Full drain of the workspace s4 corpus balances
//!   (`games_ok + games_quarantined == consumed`, staged 0) via the
//!   [`check_accounting`](hydra2_replay_rs::sink::check_accounting) assert
//!   path, every quarantine code joins the closed G5 vocabulary (no
//!   `"other"`), and the quarantine artifact round-trips the lineage file +
//!   game join.
//! - A synthetic post-claim game (claimer tsumo after a non-kan claim, no
//!   intervening discard — the Wave-C pon_ext shape) quarantines end-to-end
//!   with `tile-conservation`, zero rows, balanced accounting, and
//!   post-close readable quarantines.

use std::path::PathBuf;

use hydra2_replay_rs::sink::{
    QuarantineLineage, check_accounting, join_lineage, quarantine_reason_name,
    read_lineage_file, sink_bucket, write_lineage_file,
};
use hydra2_replay_rs::stream::{FeedOpen, FeedStream};

/// Max row stride per plane (fixed planes + T256 history pair), mirroring
/// the §7 plane table: `[34,34,20,16,2048,256,8,8,8,8,8,32,136,8,8,4,4,4,4,4,4,4,4,32,1,1]`.
const MAX_STRIDE: [usize; 26] = [
    34, 34, 20, 16, 2048, 256, 8, 8, 8, 8, 8, 32, 136, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4,
    32, 1, 1,
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/s4")
}

fn open_dir(dir: PathBuf, batch: usize) -> FeedStream {
    FeedStream::open(FeedOpen {
        data_dirs: vec![dir],
        batch_rows: batch,
        split: "train".to_string(),
        source_hash: "s6-fixture-v1".to_string(),
        rules_hash: "rules-fixture-v1".to_string(),
        action_table_hash: "table-fixture-v1".to_string(),
    })
    .expect("open stream")
}

fn full_caps(batch: usize) -> [usize; 26] {
    std::array::from_fn(|i| batch * MAX_STRIDE[i] + 64)
}

struct CallerBufs {
    bufs: [Vec<u8>; 26],
}

impl CallerBufs {
    fn canary(caps: [usize; 26]) -> Self {
        Self {
            bufs: std::array::from_fn(|i| vec![0xABu8; caps[i]]),
        }
    }

    fn ptrs(&mut self) -> [u64; 26] {
        std::array::from_fn(|i| self.bufs[i].as_mut_ptr() as u64)
    }
}

fn drain_all(stream: &mut FeedStream, batch: usize) -> (u64, u64, u64) {
    let caps = full_caps(batch);
    let mut rows_total = 0u64;
    let mut consumed_total = 0u64;
    let mut quars_total = 0u64;
    loop {
        let mut caller = CallerBufs::canary(caps);
        let ptrs = caller.ptrs();
        let fill = stream.next_into(ptrs, caps).expect("fill");
        rows_total += u64::from(fill.rows);
        consumed_total += u64::from(fill.games_consumed);
        quars_total += u64::from(fill.games_quarantined);
        if fill.rows == 0 && fill.games_consumed == 0 {
            break;
        }
    }
    (rows_total, consumed_total, quars_total)
}

#[test]
fn s6_corpus_drain_balances_and_lineage_round_trips() {
    let batch = 16;
    let mut stream = open_dir(fixture_dir(), batch);
    let (rows, consumed, quars) = drain_all(&mut stream, batch);
    assert!(rows > 0, "good fixtures must stage rows");
    // 10 corpus members: 2 good + 8 quarantined.
    assert_eq!(consumed, 10, "every corpus member consumed exactly once");
    let stats = stream.stats();
    assert_eq!(stats.rows_out, rows);
    assert_eq!(stats.games_ok + stats.games_quarantined, consumed);
    assert_eq!(quars, stats.games_quarantined);
    // G5 assert path: ok + quarantined + staged(0) == consumed.
    check_accounting(stats.games_ok, stats.games_quarantined, 0, consumed)
        .expect("accounting must balance at drain");
    // Every quarantine code joins the closed vocabularies (G5: no "other").
    let records: Vec<QuarantineLineage> = stream
        .quarantines()
        .iter()
        .map(|q| {
            assert_ne!(q.reason_code, "other", "code escaped: {}", q.game_id);
            assert_eq!(q.reason_code, quarantine_reason_name(q.reason));
            let bucket = sink_bucket(q.reason);
            assert_ne!(bucket, "other", "bucket escaped: {}", q.game_id);
            QuarantineLineage {
                game_id: q.game_id.clone(),
                event_idx: q.event_idx,
                reason: q.reason,
                obs_hash: q.obs_hash,
            }
        })
        .collect();
    assert_eq!(records.len() as u64, stats.games_quarantined);
    // Quarantine artifact round-trips the lineage file.
    let dir = std::env::temp_dir().join(format!("hydra-s6-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("quarantine.lineage");
    write_lineage_file(&path, &records).expect("lineage write");
    let back = read_lineage_file(&path).expect("lineage read");
    assert_eq!(back, records);
    // Game lineage join: every quarantined game hits, in order.
    let games: Vec<(String, u64)> = records
        .iter()
        .map(|r| (r.game_id.clone(), r.obs_hash))
        .collect();
    let joined = join_lineage(&games, &back);
    assert_eq!(joined.len(), games.len());
    for ((game, _), hit) in joined {
        let hit = hit.expect("quarantined game must join");
        assert_eq!(&hit.game_id, game);
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

fn pon_ext_game() -> Vec<u8> {
    // Wave-C pon_ext shape (mirrors the feed walk vector): seat 1 ponns
    // seat 0's 1m, then draws 6p with no intervening discard.
    let tehais = [
        ["1m", "1m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p", "3p", "4p"],
        ["1m", "1m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "7s", "7s", "7s"],
        ["2m", "2m", "2m", "2m", "3m", "3m", "3m", "3m", "4p", "4p", "4p", "5s", "5s"],
        ["7m", "7m", "7m", "7m", "8m", "8m", "8m", "8m", "E", "E", "E", "E", "F"],
    ];
    let mut seats = Vec::new();
    for hand in tehais {
        let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
        seats.push(format!("[{}]", tiles.join(",")));
    }
    let kyoku = format!(
        "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"F\",\"honba\":0,\"kyoku\":1,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
        seats.join(",")
    );
    let lines = [
        "{\"type\":\"start_game\"}".to_string(),
        kyoku,
        "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"5p\"}".to_string(),
        "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1m\",\"tsumogiri\":false}".to_string(),
        "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
            .to_string(),
        "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"6p\"}".to_string(),
        "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"6p\",\"tsumogiri\":true}".to_string(),
        "{\"type\":\"end_game\"}".to_string(),
    ];
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn s6_post_claim_draw_quarantines_end_to_end() {
    let dir = std::env::temp_dir().join(format!("hydra-s6-postclaim-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join("post-claim.jsonl"), pon_ext_game()).expect("fixture write");
    let batch = 16;
    let mut stream = open_dir(dir.clone(), batch);
    let (rows, consumed, quars) = drain_all(&mut stream, batch);
    // Conservation reject: zero rows, one quarantine, balanced.
    assert_eq!(rows, 0, "quarantined game emits zero rows");
    assert_eq!(consumed, 1);
    assert_eq!(quars, 1);
    let stats = stream.stats();
    assert_eq!((stats.games_ok, stats.games_quarantined), (0, 1));
    check_accounting(stats.games_ok, stats.games_quarantined, 0, consumed)
        .expect("accounting must balance");
    // Feed stub crossed the boundary intact (G5 histogram agreement).
    let quar = stream.quarantines().to_vec();
    assert_eq!(quar.len(), 1);
    assert_eq!(quar[0].game_id, "post-claim");
    assert_eq!(quar[0].reason_code, "tile-conservation");
    assert_eq!(quar[0].reason, 11);
    assert_eq!(quar[0].event_idx, 5);
    assert_eq!(quar[0].detail, "event_idx=5 game_idx=0");
    assert_eq!(sink_bucket(quar[0].reason), "conservation");
    // Lineage artifact round-trips this quarantine.
    let record = QuarantineLineage {
        game_id: quar[0].game_id.clone(),
        event_idx: quar[0].event_idx,
        reason: quar[0].reason,
        obs_hash: quar[0].obs_hash,
    };
    let path = dir.join("quarantine.lineage");
    write_lineage_file(&path, &[record.clone()]).expect("lineage write");
    let back = read_lineage_file(&path).expect("lineage read");
    assert_eq!(back, vec![record]);
    // Readable post-close (plan §6.5).
    stream.close();
    assert!(stream.is_closed());
    let tail = stream.quarantines().to_vec();
    assert_eq!(tail, quar);
    let tail_stats = stream.stats();
    assert_eq!(tail_stats.games_quarantined, 1);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(dir.join("post-claim.jsonl"));
    let _ = std::fs::remove_dir(&dir);
}
