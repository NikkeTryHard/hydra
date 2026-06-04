use super::*;

#[allow(
    clippy::too_many_arguments,
    reason = "loader seam carries target and sidecar policy"
)]
pub(super) fn load_game_from_events_internal(
    source_hash: Option<u64>,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let mut sink = VecReplaySampleSink::with_capacity(events.len());
    let final_scores = load_game_from_events_into_sink(
        source_hash,
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
        &mut sink,
    )?;
    Ok(MjaiGame {
        samples: sink.samples,
        final_scores,
    })
}

pub(super) fn load_game_from_events(events: Vec<MjaiEvent>) -> io::Result<MjaiGame> {
    load_game_from_events_internal(
        None,
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        ReplayTargetProfile::minimal_bc(),
        ReplayObservationProfile::BcMinimal,
        events,
        None,
        None,
    )
}

fn load_game_from_events_into_sink_strict<S: ReplaySampleSink>(
    events: Vec<MjaiEvent>,
    sink: &mut S,
) -> io::Result<[i32; 4]> {
    let mut stats = ReplayProfileStats::default();
    let t_precompute = Instant::now();
    let final_scores = final_scores(&events)?;
    stats.precompute_ns += t_precompute.elapsed().as_nanos();
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let decision_options = ReplayDecisionOptions {
        observation_profile: ReplayObservationProfile::BcMinimal,
        strict_replay_legality: true,
    };

    if events.iter().any(|event| matches!(event, MjaiEvent::Other)) {
        return Err(invalid_data("unsupported MJAI event type"));
    }

    for (idx, event) in events.iter().enumerate() {
        stats.event_count += 1;
        let t_prepare = Instant::now();
        let decisions = prepare_replay_decisions_with_options(
            idx,
            event,
            &mut state,
            &safety,
            &mut encoder,
            decision_options,
        )?;
        stats.prepare_decisions_ns += t_prepare.elapsed().as_nanos();
        for decision in decisions {
            stats.decision_count += 1;
            sink.push_sample(ReplaySampleRecord {
                trace: decision.trace.into(),
                obs: decision.obs_encoded,
                compact_facts: decision.compact_facts,
                action: decision.action_id,
                legal_mask: decision.legal_mask_f32,
                placement: 0,
                score_delta: 0,
                grp_label: 0,
                oracle_target: None,
                tenpai: [0.0; 3],
                opp_next: [MISSING_TILE_TARGET; 3],
                danger: [0.0; 102],
                danger_mask: [0.0; 102],
                safety_residual: None,
                safety_residual_mask: None,
                exit_target: None,
                exit_mask: None,
                delta_q_target: None,
                delta_q_mask: None,
                belief_fields: None,
                mixture_weights: None,
                belief_fields_present: false,
                mixture_weights_present: false,
            })?;
        }
        let t_safety = Instant::now();
        update_safety(&mut safety, event)?;
        stats.update_safety_ns += t_safety.elapsed().as_nanos();
        validate_terminal_event(event, &state)?;
        let t_apply = Instant::now();
        state
            .try_apply_mjai_event(event.clone())
            .map_err(|err| invalid_data(format!("replay state update failed: {err}")))?;
        stats.apply_event_ns += t_apply.elapsed().as_nanos();
    }

    record_replay_materialization_stats(ReplayMaterializationStats {
        decompress_ns: 0,
        json_parse_ns: 0,
        replay_update_ns: stats.update_safety_ns.saturating_add(stats.apply_event_ns),
        observation_encode_ns: stats
            .replay_observation_ns
            .saturating_add(stats.encode_observation_ns),
        mask_build_ns: stats
            .legal_mask_build_ns
            .saturating_add(stats.legal_mask_convert_ns),
        target_synthesis_ns: stats
            .precompute_ns
            .saturating_add(stats.prepare_decisions_ns),
        event_count: stats.event_count,
        decision_count: stats.decision_count,
    });

    Ok(final_scores)
}

#[allow(
    clippy::too_many_arguments,
    reason = "public test/helper seam carries target and sidecar policy"
)]
pub fn load_game_from_events_with_sidecar(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let source_hash = source_hash_from_identity(source_identity);
    load_game_from_events_internal(
        Some(source_hash),
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_reader<R: BufRead>(reader: R) -> io::Result<MjaiGame> {
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    let parse_ns = t_parse.elapsed().as_nanos();
    let game = load_game_from_events(events)?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: parse_ns,
        ..ReplayMaterializationStats::default()
    });
    if !game.samples.is_empty() {
        let stats = ReplayProfileStats {
            parse_events_ns: parse_ns,
            ..ReplayProfileStats::default()
        };
        maybe_print_replay_profile(&stats);
    }
    Ok(game)
}
fn load_game_from_reader_strict_into_sink<R: BufRead, S: ReplaySampleSink>(
    reader: R,
    sink: &mut S,
) -> io::Result<[i32; 4]> {
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    let parse_ns = t_parse.elapsed().as_nanos();
    let final_scores = load_game_from_events_into_sink_strict(events, sink)?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: parse_ns,
        ..ReplayMaterializationStats::default()
    });
    Ok(final_scores)
}

pub fn load_game_from_reader_strict<R: BufRead>(reader: R) -> io::Result<MjaiGame> {
    let mut sink = VecReplaySampleSink::with_capacity(0);
    let final_scores = load_game_from_reader_strict_into_sink(reader, &mut sink)?;
    Ok(MjaiGame {
        samples: sink.samples,
        final_scores,
    })
}

pub fn debug_first_replay_failure_from_reader<R: BufRead>(reader: R) -> io::Result<Option<String>> {
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;

    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut legal_buf = Vec::with_capacity(64);

    for (idx, event) in events.iter().enumerate() {
        match prepare_replay_decision(event, &mut state, &safety, &mut encoder) {
            Ok(_) => {}
            Err(err) => {
                let actor = mjai_event_actor(event).map(|actor| actor as u8);
                let env_action = state.replay_action_for_mjai_event(event).map_err(|conv| {
                    invalid_data(format!("replay action conversion failed: {conv}"))
                })?;
                let legal_actions = if let Some(actor) = actor {
                    state.get_legal_actions_into(actor, &mut legal_buf);
                    format!("{:?}", legal_buf)
                } else {
                    "<actor unavailable>".to_string()
                };
                return Ok(Some(format!(
                    "EVENT_INDEX: {idx}\nEVENT: {:?}\nEVENT_ACTOR: {:?}\nENV_ACTION: {:?}\nSTATE_PHASE: {:?}\nSTATE_DRAWN: {:?}\nACTIVE_PLAYERS: {:?}\nLEGAL_ACTIONS: {}\nERROR: {}",
                    event,
                    actor,
                    env_action,
                    state.phase,
                    state.drawn_tile,
                    state.active_player_slice(),
                    legal_actions,
                    err
                )));
            }
        }

        update_safety(&mut safety, event)?;
        if let Err(err) = validate_terminal_event(event, &state) {
            return Ok(Some(format!(
                "EVENT_INDEX: {idx}\nEVENT: {:?}\nSTATE_PHASE: {:?}\nSTATE_DRAWN: {:?}\nLAST_DISCARD: {:?}\nERROR: {}",
                event, state.phase, state.drawn_tile, state.last_discard, err
            )));
        }
        if let Err(err) = state.try_apply_mjai_event(event.clone()) {
            return Ok(Some(format!(
                "EVENT_INDEX: {idx}\nEVENT: {:?}\nERROR: replay state update failed: {}",
                event, err
            )));
        }
    }

    Ok(None)
}

#[allow(
    clippy::too_many_arguments,
    reason = "reader seam carries target and sidecar policy"
)]
pub fn load_game_from_reader_with_sidecar<R: BufRead>(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    reader: R,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: t_parse.elapsed().as_nanos(),
        ..ReplayMaterializationStats::default()
    });
    load_game_from_events_with_sidecar(
        source_identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
    )
}

/// Loads one already-decompressed MJAI stream into a caller-owned sample sink.
///
/// Samples are emitted in replay order without building `MjaiGame.samples`; the
/// returned scores are the final game scores. `source_identity` is used only for
/// joined sidecar replay-key hashing when `policy` contains sidecar indexes.
pub fn load_game_from_reader_into_sink<R, S>(
    source_identity: &str,
    reader: R,
    policy: Option<&ReplayLoadPolicy<'_>>,
    sink: &mut S,
) -> io::Result<[i32; 4]>
where
    R: BufRead,
    S: ReplaySampleSink,
{
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: t_parse.elapsed().as_nanos(),
        ..ReplayMaterializationStats::default()
    });

    let (
        source_hash,
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        exit_sidecar,
        delta_q_sidecar,
    ) = match policy {
        Some(policy) => (
            policy
                .has_joined_sidecars()
                .then(|| source_hash_from_identity(source_identity)),
            policy.exit_provenance,
            policy.delta_q_provenance,
            policy.profile,
            policy.observation_profile,
            policy.exit_sidecar,
            policy.delta_q_sidecar,
        ),
        None => (
            None,
            SidecarProvenance::default(),
            SidecarProvenance::default(),
            ReplayTargetProfile::minimal_bc(),
            ReplayObservationProfile::BcMinimal,
            None,
            None,
        ),
    };
    load_game_from_events_into_sink(
        source_hash,
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
        sink,
    )
}

pub fn load_game_from_stream_into_sink<R, S>(
    source_identity: &str,
    reader: R,
    policy: Option<&ReplayLoadPolicy<'_>>,
    sink: &mut S,
) -> io::Result<[i32; 4]>
where
    R: Read,
    S: ReplaySampleSink,
{
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;

    match compression {
        StreamCompression::Gzip => {
            let (timed, elapsed_ns) = TimedRead::new(GzDecoder::new(reader));
            let result = load_game_from_reader_into_sink(
                source_identity,
                BufReader::new(timed),
                policy,
                sink,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Zstd => {
            let zstd = ZstdDecoder::new(reader)
                .map_err(|err| invalid_data(format!("failed to open zstd MJAI stream: {err}")))?;
            let (timed, elapsed_ns) = TimedRead::new(zstd);
            let result = load_game_from_reader_into_sink(
                source_identity,
                BufReader::new(timed),
                policy,
                sink,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Plain => {
            load_game_from_reader_into_sink(source_identity, reader, policy, sink)
        }
    }
}

pub fn load_game_from_stream_into_sink_strict<R, S>(reader: R, sink: &mut S) -> io::Result<[i32; 4]>
where
    R: Read,
    S: ReplaySampleSink,
{
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;
    match compression {
        StreamCompression::Plain => load_game_from_reader_into_sink_strict(reader, sink),
        StreamCompression::Gzip => {
            load_game_from_reader_into_sink_strict(BufReader::new(GzDecoder::new(reader)), sink)
        }
        StreamCompression::Zstd => {
            load_game_from_reader_into_sink_strict(BufReader::new(ZstdDecoder::new(reader)?), sink)
        }
    }
}

pub fn load_game_from_reader_into_sink_strict<R, S>(reader: R, sink: &mut S) -> io::Result<[i32; 4]>
where
    R: BufRead,
    S: ReplaySampleSink,
{
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: t_parse.elapsed().as_nanos(),
        ..ReplayMaterializationStats::default()
    });
    load_game_from_events_into_sink_strict(events, sink)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamCompression {
    Plain,
    Gzip,
    Zstd,
}

struct TimedRead<R> {
    inner: R,
    elapsed_ns: Rc<Cell<u64>>,
}

impl<R> TimedRead<R> {
    fn new(inner: R) -> (Self, Rc<Cell<u64>>) {
        let elapsed_ns = Rc::new(Cell::new(0));
        (
            Self {
                inner,
                elapsed_ns: Rc::clone(&elapsed_ns),
            },
            elapsed_ns,
        )
    }
}

impl<R: Read> Read for TimedRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let start = Instant::now();
        let result = self.inner.read(buf);
        let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.elapsed_ns
            .set(self.elapsed_ns.get().saturating_add(elapsed));
        result
    }
}

fn inspect_stream_compression<R: BufRead>(reader: &mut R) -> io::Result<StreamCompression> {
    let buf = reader
        .fill_buf()
        .map_err(|err| invalid_data(format!("failed to inspect MJAI stream: {err}")))?;
    if buf.starts_with(&[0x1f, 0x8b]) {
        Ok(StreamCompression::Gzip)
    } else if buf.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Ok(StreamCompression::Zstd)
    } else {
        Ok(StreamCompression::Plain)
    }
}

fn record_decompression_result<T>(
    result: &io::Result<T>,
    elapsed_ns: &Cell<u64>,
    compression: StreamCompression,
) {
    if result.is_ok() && !matches!(compression, StreamCompression::Plain) {
        record_replay_materialization_stats(ReplayMaterializationStats {
            decompress_ns: u128::from(elapsed_ns.get()),
            ..ReplayMaterializationStats::default()
        });
    }
}

pub fn load_game_from_stream<R: Read>(reader: R) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;

    match compression {
        StreamCompression::Gzip => {
            let (timed, elapsed_ns) = TimedRead::new(GzDecoder::new(reader));
            let result = load_game_from_reader(BufReader::new(timed));
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Zstd => {
            let zstd = ZstdDecoder::new(reader)
                .map_err(|err| invalid_data(format!("failed to open zstd MJAI stream: {err}")))?;
            let (timed, elapsed_ns) = TimedRead::new(zstd);
            let result = load_game_from_reader(BufReader::new(timed));
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Plain => load_game_from_reader(reader),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "stream seam carries target and sidecar policy"
)]
pub fn load_game_from_stream_with_sidecar<R: Read>(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    reader: R,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;

    match compression {
        StreamCompression::Gzip => {
            let (timed, elapsed_ns) = TimedRead::new(GzDecoder::new(reader));
            let result = load_game_from_reader_with_sidecar(
                source_identity,
                exit_provenance,
                delta_q_provenance,
                profile,
                observation_profile,
                BufReader::new(timed),
                exit_sidecar,
                delta_q_sidecar,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Zstd => {
            let zstd = ZstdDecoder::new(reader)
                .map_err(|err| invalid_data(format!("failed to open zstd MJAI stream: {err}")))?;
            let (timed, elapsed_ns) = TimedRead::new(zstd);
            let result = load_game_from_reader_with_sidecar(
                source_identity,
                exit_provenance,
                delta_q_provenance,
                profile,
                observation_profile,
                BufReader::new(timed),
                exit_sidecar,
                delta_q_sidecar,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Plain => load_game_from_reader_with_sidecar(
            source_identity,
            exit_provenance,
            delta_q_provenance,
            profile,
            observation_profile,
            reader,
            exit_sidecar,
            delta_q_sidecar,
        ),
    }
}

pub fn load_game_from_path(path: impl AsRef<Path>) -> io::Result<MjaiGame> {
    let file = fs::File::open(path)?;
    load_game_from_stream(file)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))
}

pub fn load_game_from_path_strict(path: impl AsRef<Path>) -> io::Result<MjaiGame> {
    let file = fs::File::open(path)?;
    load_game_from_stream_strict(file)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))
}

pub fn load_game_from_stream_strict<R: Read>(reader: R) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;
    match compression {
        StreamCompression::Plain => load_game_from_reader_strict(reader),
        StreamCompression::Gzip => {
            load_game_from_reader_strict(BufReader::new(GzDecoder::new(reader)))
        }
        StreamCompression::Zstd => {
            load_game_from_reader_strict(BufReader::new(ZstdDecoder::new(reader)?))
        }
    }
}

pub fn load_game_from_stream_strict_into_sink<R: Read, S: ReplaySampleSink>(
    reader: R,
    sink: &mut S,
) -> io::Result<[i32; 4]> {
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;
    match compression {
        StreamCompression::Plain => load_game_from_reader_strict_into_sink(reader, sink),
        StreamCompression::Gzip => {
            load_game_from_reader_strict_into_sink(BufReader::new(GzDecoder::new(reader)), sink)
        }
        StreamCompression::Zstd => {
            load_game_from_reader_strict_into_sink(BufReader::new(ZstdDecoder::new(reader)?), sink)
        }
    }
}

pub fn load_game_from_path_with_sidecar(
    path: impl AsRef<Path>,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let path = path.as_ref();
    let identity = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data(format!("invalid filename {}", path.display())))?;
    let file = fs::File::open(path)?;
    load_game_from_stream_with_sidecar(
        identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        observation_profile,
        file,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_path_with_policy(
    path: impl AsRef<Path>,
    policy: Option<&ReplayLoadPolicy<'_>>,
) -> io::Result<MjaiGame> {
    let path = path.as_ref();
    match policy {
        Some(policy) => load_game_from_path_with_sidecar(
            path,
            policy.exit_provenance,
            policy.delta_q_provenance,
            policy.profile,
            policy.observation_profile,
            policy.exit_sidecar,
            policy.delta_q_sidecar,
        ),
        None => load_game_from_path(path),
    }
}

pub fn load_game_from_stream_with_policy<R: Read>(
    source_identity: &str,
    reader: R,
    policy: Option<&ReplayLoadPolicy<'_>>,
) -> io::Result<MjaiGame> {
    match policy {
        Some(policy) => load_game_from_stream_with_sidecar(
            source_identity,
            policy.exit_provenance,
            policy.delta_q_provenance,
            policy.profile,
            policy.observation_profile,
            reader,
            policy.exit_sidecar,
            policy.delta_q_sidecar,
        ),
        None => load_game_from_stream(reader),
    }
}
