use std::fs::File;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::action::ActionType;
use crate::errors::RiichiError;
use crate::replay::mjai_replay::{
    load_mjai_events_from_path, mjai_event_actor, mjai_event_to_action, read_mjai_events, MjaiEvent,
};

#[test]
fn read_mjai_events_parses_jsonl() {
    let log = concat!(
        "{\"type\":\"start_game\"}\n",
        "{\"type\":\"reach\",\"actor\":2}\n",
        "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"5pr\",\"tsumogiri\":true}\n"
    );
    let events = read_mjai_events(Cursor::new(log)).expect("read events");
    assert_eq!(events.len(), 3);
    assert_eq!(mjai_event_actor(&events[1]), Some(2));
}

#[test]
fn load_mjai_events_from_gzip_path_parses_jsonl() {
    let path = std::env::temp_dir().join(format!(
        "hydra_engine_mjai_events_{}_{}.json.gz",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let file = File::create(&path).expect("create gzip log");
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n")
        .expect("write gzip log");
    encoder.finish().expect("finish gzip log");

    let events = load_mjai_events_from_path(&path).expect("load gz events");
    std::fs::remove_file(&path).expect("remove temp log");

    assert_eq!(events.len(), 2);
}

#[test]
fn mjai_event_to_action_preserves_tiles_and_actor() {
    let discard = MjaiEvent::Dahai {
        actor: 1,
        pai: "5pr".to_string(),
        tsumogiri: true,
    };
    let action = mjai_event_to_action(&discard)
        .expect("convert discard")
        .expect("discard action");
    assert_eq!(action.actor, Some(1));
    assert_eq!(action.action_type, ActionType::Discard);
    assert_eq!(action.tile, Some(52));

    let hora = MjaiEvent::Hora {
        actor: 3,
        target: 1,
        pai: Some("C".to_string()),
        uradora_markers: None,
        yaku: None,
        fu: None,
        han: None,
        scores: None,
        delta: None,
    };
    let action = mjai_event_to_action(&hora)
        .expect("convert hora")
        .expect("hora action");
    assert_eq!(action.actor, Some(3));
    assert_eq!(action.action_type, ActionType::Ron);
    assert_eq!(action.tile, Some(132));
}

#[test]
fn mjai_event_actor_and_action_ignore_non_actionable_events() {
    let start = MjaiEvent::StartGame {
        names: None,
        id: None,
    };
    assert_eq!(mjai_event_actor(&start), None);
    assert_eq!(
        mjai_event_to_action(&start).expect("non-action event should parse"),
        None
    );

    let end = MjaiEvent::EndGame;
    assert_eq!(mjai_event_actor(&end), None);
    assert_eq!(
        mjai_event_to_action(&end).expect("end_game should be ignored"),
        None
    );
}

#[test]
fn mjai_event_to_action_rejects_invalid_tile_strings() {
    let bad_discard = MjaiEvent::Dahai {
        actor: 0,
        pai: "??".to_string(),
        tsumogiri: false,
    };
    let err = mjai_event_to_action(&bad_discard).expect_err("bad tile should fail");
    assert!(matches!(err, RiichiError::Parse { .. }));

    let bad_ankan = MjaiEvent::Ankan {
        actor: 1,
        consumed: vec!["5m".to_string(), "oops".to_string()],
    };
    let err = mjai_event_to_action(&bad_ankan).expect_err("bad consumed tile should fail");
    assert!(matches!(err, RiichiError::Parse { .. }));
}

#[test]
fn read_and_load_mjai_events_report_bad_json_and_missing_paths() {
    let err =
        read_mjai_events(Cursor::new("{not-json}\n")).expect_err("invalid json line should fail");
    assert!(matches!(err, RiichiError::Parse { .. }));

    let missing = PathBuf::from(format!(
        "/home/nikketryhard/tmp/missing-mjai-replay-{}-{}.jsonl",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let err = load_mjai_events_from_path(&missing).expect_err("missing path should fail");
    assert!(matches!(err, RiichiError::Serialization { .. }));
}
