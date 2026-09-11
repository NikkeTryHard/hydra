//! Bridge S5 gates: `next_into` fills the caller prefix exactly,
//! `BufferTooSmall` drains nothing and moves no counter (reporting the
//! failing plane), close is idempotent with post-close readable stats.
//!
//! Fixtures stay at the workspace `tests/fixtures/s4` corpus (kept in place
//! so `feed::ingest`'s `include_str!` paths keep working).

use std::path::PathBuf;

use hydra2_replay_rs::stream::{FeedOpen, FeedStream, StreamError};

/// Max row stride per plane (fixed planes + T256 history pair), mirroring
/// the §7 plane table: `[34,34,20,16,2048,256,8,8,8,8,8,32,136,8,8,4,4,4,4,4,4,4,4,32,1,1]`.
const MAX_STRIDE: [usize; 26] = [
    34, 34, 20, 16, 2048, 256, 8, 8, 8, 8, 8, 32, 136, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4,
    32, 1, 1,
];
/// Fixed row stride per plane (`#4`/`#5` filled per `t_len`).
const FIXED_STRIDE: [usize; 26] = [
    34, 34, 20, 16, 0, 0, 8, 8, 8, 8, 8, 32, 136, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 32,
    1, 1,
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/s4")
}

fn open(batch: usize) -> FeedStream {
    FeedStream::open(FeedOpen {
        data_dirs: vec![fixture_dir()],
        batch_rows: batch,
        split: "train".to_string(),
        source_hash: "s4-fixture-v1".to_string(),
        rules_hash: "rules-fixture-v1".to_string(),
        action_table_hash: "table-fixture-v1".to_string(),
    })
    .expect("open fixture stream")
}

/// Generous caps covering `batch` rows at any bucket (T256 history pair).
fn full_caps(batch: usize) -> [usize; 26] {
    std::array::from_fn(|i| batch * MAX_STRIDE[i] + 64)
}

/// Canary-filled caller buffers + pointer/cap views (buffers must outlive
/// the fill call).
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
fn bridge_next_into_fills_prefix_exactly() {
    let batch = 16;
    let mut stream = open(batch);
    let caps = full_caps(batch);
    let mut caller = CallerBufs::canary(caps);
    let ptrs = caller.ptrs();
    let fill = stream.next_into(ptrs, caps).expect("first fill");
    assert!(fill.rows > 0, "good fixtures must stage rows");
    assert!(
        [32, 64, 128, 256].contains(&fill.t_len),
        "t_len must be a frozen bucket, got {}",
        fill.t_len
    );
    let rows = fill.rows as usize;
    let t = fill.t_len as usize;
    // Exact-prefix: every byte past rows*stride keeps the canary.
    for (i, buf) in caller.bufs.iter().enumerate() {
        let stride = if i == 4 {
            t * 8
        } else if i == 5 {
            t
        } else {
            FIXED_STRIDE[i]
        };
        assert!(
            buf[rows * stride..].iter().all(|b| *b == 0xAB),
            "plane {i}: beyond-rows bytes touched"
        );
    }
    // Prefix landed: chosen ids (i64 LE, small ints) are not all canary.
    assert!(
        !caller.bufs[6][..rows * 8].iter().all(|b| *b == 0xAB),
        "plane 6 prefix must hold chosen ids"
    );
    // Dora intact: every entry is -1 (tail sentinel) or a physical id.
    for r in 0..rows {
        for k in 0..5 {
            let off = r * 20 + k * 4;
            let v = i32::from_le_bytes(caller.bufs[2][off..off + 4].try_into().unwrap());
            assert!(
                v == -1 || (0..136).contains(&v),
                "dora[{r}][{k}] implausible ({v}): T-tail touched dora"
            );
        }
    }
}

#[test]
fn bridge_small_buffer_drains_nothing_and_reports_plane() {
    let batch = 16;
    // Sabotage exactly one plane: history-kind cap fits no bucket.
    let mut sabotaged = full_caps(batch);
    sabotaged[4] = 1;
    let mut stream = open(batch);
    let mut caller = CallerBufs::canary(sabotaged);
    let ptrs = caller.ptrs();
    let err = stream
        .next_into(ptrs, sabotaged)
        .expect_err("tiny plane-4 cap must fail");
    match err {
        StreamError::BufferTooSmall { plane, needed, capacity } => {
            assert_eq!(plane, 4, "must report the failing plane index");
            assert_eq!(capacity, 1, "capacity echoes the caller byte_cap");
            assert!(needed > capacity, "needed must exceed capacity");
        }
        other => panic!("expected BufferTooSmall, got {other:?}"),
    }
    // Counters unmoved, quarantines unlisted.
    let stats = stream.stats();
    assert_eq!(stats.rows_out, 0, "rows_out moved on failed fill");
    assert_eq!(stats.games_ok, 0, "games_ok moved on failed fill");
    assert_eq!(stats.games_quarantined, 0, "games_quarantined moved on failed fill");
    assert!(stream.quarantines().is_empty(), "quarantines listed on failed fill");
    // Retry with full caps replays byte-identical rows to a fresh stream:
    // the failed call consumed nothing (cursor restored).
    let caps = full_caps(batch);
    let mut retry = CallerBufs::canary(caps);
    let ptrs = retry.ptrs();
    let fill = stream.next_into(ptrs, caps).expect("retry fill");
    assert!(fill.rows > 0);
    let mut fresh = open(batch);
    let mut fresh_caller = CallerBufs::canary(caps);
    let fresh_ptrs = fresh_caller.ptrs();
    let fresh_fill = fresh.next_into(fresh_ptrs, caps).expect("fresh fill");
    assert_eq!(fill.rows, fresh_fill.rows, "retry diverged: failed call drained");
    assert_eq!(fill.t_len, fresh_fill.t_len);
    for i in 0..13 {
        assert_eq!(retry.bufs[i], fresh_caller.bufs[i], "plane {i} diverged after rollback");
    }
}


#[test]
fn bridge_full_drain_accounting_balances() {
    let batch = 16;
    let mut stream = open(batch);
    let (rows, consumed, _quars) = drain_all(&mut stream, batch);
    assert!(rows > 0);
    let stats = stream.stats();
    assert_eq!(stats.open_count, 1);
    assert_eq!(stats.rows_out, rows, "rows_out must equal drained rows");
    assert_eq!(
        stats.games_ok + stats.games_quarantined,
        consumed,
        "ok + quarantined must equal consumed files"
    );
    // Exhaustion signal observed (loop ended on rows==0 && consumed==0).
    assert!(consumed > 0);
}

#[test]
fn bridge_close_is_idempotent_and_stats_stay_readable() {
    let batch = 16;
    let mut stream = open(batch);
    let (rows, _, _) = drain_all(&mut stream, batch);
    assert!(rows > 0);
    stream.close();
    assert!(stream.is_closed());
    stream.close();
    let stats = stream.stats();
    assert_eq!(stats.rows_out, rows);
    assert!(!stream.quarantines().is_empty() || stats.games_quarantined == 0);
    let caps = full_caps(batch);
    let mut caller = CallerBufs::canary(caps);
    let ptrs = caller.ptrs();
    assert_eq!(
        stream.next_into(ptrs, caps),
        Err(StreamError::Closed),
        "next_into after close must fail Closed"
    );
}

#[test]
fn bridge_open_validates() {
    let dir = fixture_dir();
    let good = || FeedOpen {
        data_dirs: vec![dir.clone()],
        batch_rows: 4,
        split: "train".to_string(),
        source_hash: "s".to_string(),
        rules_hash: "r".to_string(),
        action_table_hash: "a".to_string(),
    };
    assert!(FeedStream::open(good()).is_ok());
    let mut bad = good();
    bad.data_dirs.clear();
    assert!(FeedStream::open(bad).is_err());
    let mut bad = good();
    bad.batch_rows = 0;
    assert!(FeedStream::open(bad).is_err());
    let mut bad = good();
    bad.split = "test".to_string();
    assert!(FeedStream::open(bad).is_err());
    let mut bad = good();
    bad.source_hash.clear();
    assert!(FeedStream::open(bad).is_err());
    let mut bad = good();
    bad.data_dirs = vec![dir.join("good-a.jsonl")];
    assert!(FeedStream::open(bad).is_err());
}

#[test]
fn bridge_discovers_mjai_suffixes_with_stemmed_ids() {
    // Scratch dir (outside the repo): mjai-named copies of one good game
    // (rows must flow) and one framing quarantine (stem must survive).
    let dir = std::env::temp_dir().join(format!("hydra-bridge-mjai-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let good = std::fs::read(fixture_dir().join("good-a.jsonl")).expect("good fixture read");
    let bad = std::fs::read(fixture_dir().join("q-framing.jsonl")).expect("bad fixture read");
    std::fs::write(dir.join("tenhou-game.mjai.json"), &good).expect("good copy");
    std::fs::write(dir.join("broken-game.mjai.json.zst"), &bad).expect("bad copy");
    // Decoys the discovery filter must skip.
    std::fs::write(dir.join("notes.txt"), b"not a game").expect("decoy");
    std::fs::write(dir.join("game.json"), b"{}").expect("json decoy");
    let mut stream = FeedStream::open(FeedOpen {
        data_dirs: vec![dir.clone()],
        batch_rows: 64,
        split: "train".to_string(),
        source_hash: "s".to_string(),
        rules_hash: "r".to_string(),
        action_table_hash: "a".to_string(),
    })
    .expect("open mjai dir");
    let caps = full_caps(64);
    let mut caller = CallerBufs::canary(caps);
    let ptrs = caller.ptrs();
    let mut rows_total = 0u64;
    loop {
        let fill = stream.next_into(ptrs, caps).expect("mjai fill");
        rows_total += u64::from(fill.rows);
        if fill.rows == 0 && fill.games_consumed == 0 {
            break;
        }
    }
    assert!(rows_total > 0, "mjai-suffixed good game must stage rows");
    let stats = stream.stats();
    assert_eq!(stats.games_ok, 1, "exactly the good mjai file: {stats:?}");
    assert_eq!(stats.games_quarantined, 1, "exactly the bad mjai file: {stats:?}");
    let quars = stream.quarantines();
    assert_eq!(quars.len(), 1);
    assert_eq!(quars[0].game_id, "broken-game", "stem must strip .mjai.json.zst");
    assert_eq!(quars[0].reason_code, "framing");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bridge_multi_game_pack_keys_subgames_without_collision() {
    // One file, three chunks (good, framing, good): siblings must sample
    // around the bad chunk, and ok + quarantined must equal 3 sub-games
    // (base advances by staged count, never by 1).
    let dir = std::env::temp_dir().join(format!("hydra-bridge-pack-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut pack = Vec::new();
    for name in ["good-a.jsonl", "q-framing.jsonl", "good-b.jsonl"] {
        pack.extend_from_slice(&std::fs::read(fixture_dir().join(name)).expect("fixture read"));
    }
    std::fs::write(dir.join("pack-3.jsonl"), &pack).expect("pack write");
    let mut stream = FeedStream::open(FeedOpen {
        data_dirs: vec![dir.clone()],
        batch_rows: 64,
        split: "train".to_string(),
        source_hash: "s".to_string(),
        rules_hash: "r".to_string(),
        action_table_hash: "a".to_string(),
    })
    .expect("open pack dir");
    let caps = full_caps(64);
    let mut caller = CallerBufs::canary(caps);
    let ptrs = caller.ptrs();
    let mut rows_total = 0u64;
    let mut consumed_total = 0u64;
    loop {
        let fill = stream.next_into(ptrs, caps).expect("pack fill");
        rows_total += u64::from(fill.rows);
        consumed_total += u64::from(fill.games_consumed);
        if fill.rows == 0 && fill.games_consumed == 0 {
            break;
        }
    }
    // Frozen fixtures: good-a + good-b stage 9 rows; the framing chunk in
    // the middle quarantines without killing its siblings.
    assert_eq!(rows_total, 9, "bad chunk must not poison siblings");
    assert_eq!(consumed_total, 3, "three sub-games staged from one file");
    let stats = stream.stats();
    assert_eq!(stats.games_ok, 2, "two ok sub-games: {stats:?}");
    assert_eq!(stats.games_quarantined, 1, "one quarantined sub-game: {stats:?}");
    let quars = stream.quarantines();
    assert_eq!(quars.len(), 1);
    assert_eq!(quars[0].game_id, "pack-3", "stem shared, detail carries sub-game idx");
    assert_eq!(quars[0].reason_code, "framing");
    let _ = std::fs::remove_dir_all(&dir);
}
