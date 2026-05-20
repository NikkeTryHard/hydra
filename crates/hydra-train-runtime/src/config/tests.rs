use super::*;

#[test]
fn parse_pf_repetitions_aliases_required_successes() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--pf-repetitions".to_string(),
        "5".to_string(),
        "--pf-candidate-tuples".to_string(),
        "1024:2:1:1,2048:4:2:2".to_string(),
    ])
    .expect("pf repetitions should parse");
    let preflight = cli.preflight.expect("preflight options should be present");
    assert_eq!(preflight.preflight_config.required_successes, 5);
    assert_eq!(preflight.preflight_config.bench_candidate_tuples.len(), 2);
}

#[test]
fn usage_lists_all_probe_kinds() {
    let text = usage("train");
    assert!(text.contains("--probe-kind <train|validation|rl_games|rl_microbatch>"));
}

#[test]
fn parse_args_rejects_partial_probe_flags() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
    ];
    let err = parse_args(args).expect_err("partial probe args should fail");
    assert!(
        err.contains("probe-only mode requires both --probe-kind and --probe-candidate-microbatch")
    );
}

#[test]
fn parse_args_accepts_rl_probe_kinds_advertised_in_usage() {
    for kind in ["rl_games", "rl_microbatch"] {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--probe-kind".to_string(),
            kind.to_string(),
            "--probe-candidate-microbatch".to_string(),
            "16".to_string(),
        ];
        let cli = parse_args(args).expect("advertised probe kind should parse");
        assert!(cli.probe_only.is_some());
    }
}
