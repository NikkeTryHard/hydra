use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::CampaignRunLayout;

fn temp_dir(name: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be monotonic")
        .as_millis();
    std::env::temp_dir().join(format!("hydra-campaign-layout-{name}-{millis}"))
}

#[test]
fn campaign_layout_creates_run_skeleton_and_metadata() {
    let root = temp_dir("skeleton");
    let layout = CampaignRunLayout::new(
        &root,
        Some("T5_exit_auxiliary"),
        Some("run_002"),
        "T1_ppo_control",
    );
    layout
        .ensure()
        .expect("campaign skeleton should be created");
    layout
        .ensure_tensorboard_dir()
        .expect("tensorboard dir should be created on demand");

    let run_dir = root.join("stages/T5_exit_auxiliary/runs/run_002");
    assert_eq!(layout.run_dir, run_dir);
    assert!(root.join("campaign.json").is_file());
    assert!(root.join("registry/opponent_pools").is_dir());
    assert_eq!(
        fs::read_to_string(root.join("stages/T5_exit_auxiliary/latest_run"))
            .expect("read latest run"),
        "run_002\n"
    );
    for dir in [
        "logs",
        "checkpoints",
        "exports",
        "rollouts",
        "eval",
        "tensorboard",
    ] {
        assert!(run_dir.join(dir).is_dir(), "missing {dir}");
    }

    let metadata: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(run_dir.join("launch_metadata.json")).expect("read launch metadata"),
    )
    .expect("parse launch metadata");
    assert_eq!(metadata["stage"], "T5_exit_auxiliary");
    assert_eq!(metadata["run_id"], "run_002");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn campaign_layout_detects_already_resolved_run_dir() {
    let root = temp_dir("resolved");
    let run_dir = root.join("stages/T1_ppo_control/runs/existing");
    let layout = CampaignRunLayout::new(&run_dir, Some("ignored"), Some("ignored"), "ignored");

    assert_eq!(layout.campaign_root, root);
    assert_eq!(layout.stage, "T1_ppo_control");
    assert_eq!(layout.run_id, "existing");
    assert_eq!(layout.run_dir, run_dir);
}
