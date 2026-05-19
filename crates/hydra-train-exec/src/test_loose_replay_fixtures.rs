use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::data_pipeline::{DataManifest, DataSource};

fn unique_test_root(label: &str) -> PathBuf {
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("tmp");
    fs::create_dir_all(&base).expect("shared loose replay fixture root should be creatable");
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    base.join(format!("hydra-loose-replay-{label}-{unique}"))
}

pub(super) fn tiny_real_mjai_replay() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

pub(super) fn write_real_probe_fixture(label: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = unique_test_root(label);
    fs::create_dir_all(&root).expect("create real probe fixture dir");
    let replay_path = root.join("game.mjai.json");
    fs::write(&replay_path, tiny_real_mjai_replay()).expect("write real probe replay");
    let result_path = root.join("probe-result.json");
    (root, replay_path, result_path)
}

pub(super) fn loose_file_manifest(
    replay_path: PathBuf,
    train_count: usize,
    val_count: usize,
) -> DataManifest {
    DataManifest {
        sources: vec![DataSource::LooseFile(replay_path)],
        total_games: train_count + val_count,
        train_count,
        val_count,
        counts_exact: true,
    }
}

pub(super) fn single_loose_train_manifest(label: &str) -> (DataManifest, PathBuf) {
    let path = unique_test_root(label).with_extension("mjai.json");
    fs::write(&path, tiny_real_mjai_replay())
        .expect("runtime autotune replay fixture should be writable");
    (loose_file_manifest(path.clone(), 1, 0), path)
}
