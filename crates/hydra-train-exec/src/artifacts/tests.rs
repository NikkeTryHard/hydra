use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_checkpoint_base(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("hydra_exec_{label}_{unique}"))
}

#[test]
fn checkpoint_meta_semantic_match_ignores_timestamp_only() {
    let mut existing = CheckpointMeta::new(3, 1.25, Some(0.5), Some(1.0), Some(2.0));
    let mut candidate = existing.clone();
    candidate.timestamp = existing.timestamp.saturating_add(10);

    assert!(checkpoint_meta_semantically_matches(&existing, &candidate));

    existing.hidden_channels += 1;
    assert!(!checkpoint_meta_semantically_matches(&existing, &candidate));
}

#[test]
fn write_checkpoint_meta_preserves_existing_semantic_match() {
    let base = temp_checkpoint_base("checkpoint_meta_preserve");
    let meta_path = base.with_extension("meta.json");
    let meta = CheckpointMeta::new(4, 2.5, None, None, None);

    write_checkpoint_meta(&base, &meta).expect("write checkpoint metadata");
    let first_raw = fs::read_to_string(&meta_path).expect("read checkpoint metadata");
    let mut same_semantics = meta.clone();
    same_semantics.timestamp = meta.timestamp.saturating_add(60);
    write_checkpoint_meta(&base, &same_semantics).expect("rewrite matching checkpoint metadata");

    assert_eq!(
        fs::read_to_string(&meta_path).expect("read preserved checkpoint metadata"),
        first_raw
    );
    let _ = fs::remove_file(meta_path);
}
