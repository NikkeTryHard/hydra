//! rl/core/tests: native behavior tests for the hanchan driver.
//!
//! Seating accepts only cyclic rotations; bit packing is least-significant-bit
//! first over canonical action order; a full game terminates with dense hand
//! ordinals `0..N-1` and valid placement ranks; replaying the applied prefix
//! against identical walls reproduces the state digest.

    use super::*;
    use sha2::Digest as _;

    fn test_context() -> GameContext {
        GameContext {
            rules_id: "test-rules".to_string(),
            rules_hash: "sha256:00".to_string(),
            action_table_hash: "sha256:01".to_string(),
            event_schema_hash: "sha256:02".to_string(),
            observation_schema_hash: "sha256:03".to_string(),
            packet_boundary_hash: "sha256:04".to_string(),
            adapter_version: "2.0.0".to_string(),
            schedule_id: "test-schedule".to_string(),
            schedule_digest: "sha256:ab".to_string(),
            schedule_tiles: std::array::from_fn(|i| i as u8),
            seat_perm: [0, 1, 2, 3],
            rank_values: [1.0, 0.0, -0.5, -1.0],
            behavior_ckpt: [7u8; 32],
        }
    }

    #[test]
    fn seating_allows_only_rotations() {
        assert!(check_cyclic_rotation([0, 1, 2, 3]).is_ok());
        assert!(check_cyclic_rotation([2, 3, 0, 1]).is_ok());
        assert!(check_cyclic_rotation([0, 2, 1, 3]).is_err());
        assert!(check_cyclic_rotation([0, 1, 2, 2]).is_err());
    }

    #[test]
    fn bit_packing_is_lsb_first() {
        let bits = pack_bits_lsb(&[true, false, true, false, false, false, false, false, true]);
        assert_eq!(bits, vec![0b0000_0101, 0b0000_0001]);
        assert_eq!(bits.len(), 2);
    }

    #[test]
    fn full_game_terminates_with_dense_ordinals() {
        let ctx = test_context();
        let mut game = NativeGame::open(ctx, 25000).expect("open");
        let mut ordinals = vec![game.hand_index()];
        let mut steps = 0usize;
        while !game.is_terminal() {
            let decision = game.drain().expect("drain").expect("pending decision");
            let seat = match decision {
                PendingDecision::Draw { seat } => seat,
                PendingDecision::Window { pending } => pending[0],
            };
            let obs = game.observe(seat).expect("observe");
            assert!(obs.legal_mask.iter().any(|f| *f));
            let (ordered, _) = game.view_for(seat).expect("view");
            let selected = ordered[0].1.clone();
            game.apply(&selected).expect("apply");
            if game.hand_index() != *ordinals.last().expect("ordinal") {
                ordinals.push(game.hand_index());
            }
            steps += 1;
            assert!(steps < 20000, "hanchan must terminate");
        }
        let dense: Vec<u64> = (0..ordinals.len() as u64).collect();
        assert_eq!(ordinals, dense);
        let outcome = game.outcome().expect("outcome");
        let mut ranks = outcome.ranks.to_vec();
        ranks.sort_unstable();
        assert_eq!(ranks, vec![1, 2, 3, 4]);
    }

    #[test]
    fn replay_prefix_reproduces_state() {
        let ctx = test_context();
        let mut game = NativeGame::open(ctx.clone(), 25000).expect("open");
        for _ in 0..12 {
            let decision = game.drain().expect("drain").expect("pending");
            let seat = match decision {
                PendingDecision::Draw { seat } => seat,
                PendingDecision::Window { pending } => pending[0],
            };
            let (ordered, _) = game.view_for(seat).expect("view");
            game.apply(&ordered[0].1.clone()).expect("apply");
            if game.is_terminal() {
                break;
            }
        }
        let digest_before = game.state_digest();
        let prefix = game.applied_prefix().to_vec();
        let mut fresh = NativeGame::open(ctx, 25000).expect("reopen");
        fresh.replay_prefix(&prefix).expect("replay");
        assert_eq!(fresh.state_digest(), digest_before);
    }

    #[test]
    fn replay_keeps_a_partially_answered_claim_window() {
        let mut ctx = test_context();
        ctx.schedule_tiles = std::array::from_fn(|i| ((i * 37) % 136) as u8);
        let mut game = NativeGame::open(ctx.clone(), 25000).expect("open");
        for _ in 0..1200 {
            let decision = game.drain().expect("drain").expect("pending");
            let seat = match &decision {
                PendingDecision::Draw { seat } => *seat,
                PendingDecision::Window { pending } => pending[0],
            };
            let (ordered, _) = game.view_for(seat).expect("view");
            let action = ordered[0].1.clone();
            game.apply(&action).expect("answer");
            if matches!(decision, PendingDecision::Window { ref pending } if pending.len() > 1) {
                // A checkpoint between replies must restore the unanswered
                // seats, not ask an already committed responder to draw again.
                let pending = game.drain().expect("partial window");
                let mut restored = NativeGame::open(ctx, 25000).expect("restore open");
                restored.replay_prefix(game.applied_prefix()).expect("replay");
                assert_eq!(restored.drain().expect("restored window"), pending);
                assert_eq!(restored.state_digest(), game.state_digest());
                return;
            }
            if game.is_terminal() {
                break;
            }
        }
        panic!("named wall must exercise a multi-responder claim window");
    }

    #[test]
    fn frontier_advances_only_on_resolved_decisions() {
        // Buffered partial window answers resolve nothing: the counter holds
        // across them and advances once on the final response. Draws resolve
        // one each. Uses the named wall known to open a multi-responder
        // claim window within budget.
        let mut ctx = test_context();
        ctx.schedule_tiles = std::array::from_fn(|i| ((i * 37) % 136) as u8);
        let mut game = NativeGame::open(ctx, 25000).expect("open");
        // New-game boundary: a fresh open resolves nothing and captures
        // nothing; the drive loop below covers the first draw onward.
        assert_eq!(game.frontier_count(), 0);
        assert_eq!(game.captured_count(), 0);
        let mut saw_window = false;
        for _ in 0..2000 {
            if game.is_terminal() {
                break;
            }
            let revision = game.frontier_count();
            let seats: Vec<u8> = match game.drain().expect("drain").expect("pending") {
                PendingDecision::Draw { seat } => vec![seat],
                PendingDecision::Window { pending } => {
                    if pending.len() > 1 {
                        saw_window = true;
                    }
                    pending
                }
            };
            for (i, seat) in seats.iter().enumerate() {
                let (ordered, _) = game.view_for(*seat).expect("view");
                game.apply(&ordered[0].1.clone()).expect("answer");
                if i + 1 < seats.len() {
                    assert_eq!(game.frontier_count(), revision);
                }
            }
            assert_eq!(game.frontier_count(), revision + 1);
        }
        assert!(saw_window, "named wall must exercise a multi-responder claim window");
    }

    #[test]
    fn replay_prefix_rebuilds_frontier_count() {
        // The scheduling frontier needs no stored counter: replaying the
        // applied prefix through identical walls re-fires the same resolving
        // applies (partial answers early-return again), so a fresh game
        // converges to the same count alongside the same state digest.
        let mut ctx = test_context();
        ctx.schedule_tiles = std::array::from_fn(|i| ((i * 37) % 136) as u8);
        let mut game = NativeGame::open(ctx.clone(), 25000).expect("open");
        for _ in 0..1200 {
            let decision = game.drain().expect("drain").expect("pending");
            let seat = match &decision {
                PendingDecision::Draw { seat } => *seat,
                PendingDecision::Window { pending } => pending[0],
            };
            let (ordered, _) = game.view_for(seat).expect("view");
            let action = ordered[0].1.clone();
            game.apply(&action).expect("answer");
            if matches!(decision, PendingDecision::Window { ref pending } if pending.len() > 1) {
                let mut restored = NativeGame::open(ctx, 25000).expect("restore open");
                restored.replay_prefix(game.applied_prefix()).expect("replay");
                assert_eq!(restored.frontier_count(), game.frontier_count());
                assert_eq!(restored.state_digest(), game.state_digest());
                return;
            }
            if game.is_terminal() {
                break;
            }
        }
        panic!("named wall must exercise a multi-responder claim window");
    }

    #[test]
    fn coalesced_window_captures_carry_consecutive_distinct_provenance() {
        // Two same-window seats captured back-to-back with no applies between
        // (exactly a coalesced commit) share one decision identity but take
        // consecutive trajectory steps with distinct provenances. Drives the
        // named multi-responder wall; fails loudly if it never appears.
        let mut ctx = test_context();
        ctx.schedule_tiles = std::array::from_fn(|i| ((i * 37) % 136) as u8);
        let mut game = NativeGame::open(ctx, 25000).expect("open");
        for _ in 0..1200 {
            if game.is_terminal() {
                break;
            }
            let responders: Vec<u8> = match game.drain().expect("drain").expect("pending") {
                PendingDecision::Draw { seat } => {
                    let (ordered, _) = game.view_for(seat).expect("view");
                    game.apply(&ordered[0].1.clone()).expect("answer");
                    continue;
                }
                PendingDecision::Window { pending } => pending,
            };
            if responders.len() < 2 {
                let (ordered, _) = game.view_for(responders[0]).expect("view");
                game.apply(&ordered[0].1.clone()).expect("answer");
                continue;
            }
            // No applies land between these captures (exactly a coalesced
            // commit), so the live applied-action count is frozen here while
            // the capture ordinal advances per row. Record the frozen count:
            // the OLD law would stamp every row below with it identically.
            let frozen_applied = game.applied_prefix().len() as u64;
            let mut rows = vec![];
            let mut decision = String::new();
            for seat in &responders {
                let obs = game.observe(*seat).expect("observe");
                let (ordered, _) = game.view_for(*seat).expect("view");
                let selected = ordered[0].0;
                let row = game
                    .capture_row(
                        &obs,
                        selected,
                        0.0,
                        [0.0; 4],
                        "w-test",
                        [0xAA; 32],
                        [0xBB; 32],
                        0,
                    )
                    .expect("capture");
                decision = obs.decision_id.clone();
                rows.push(row);
            }
            assert!(!decision.is_empty());
            // Steps advance consecutively from the first capture, independent
            // of the many unrecorded applies this test drove to get here, so
            // no global applied-length equality is asserted on this path.
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(row.trajectory_step, i as u64);
                assert_eq!(row.decision_id, decision);
            }
            // Each row re-derives EXACTLY from its own preimage with
            // applied == its capture ordinal (its trajectory step here).
            for row in &rows {
                let doc = serde_json::json!({
                    "game": row.game_id.clone(),
                    "ordinal": row.round_index,
                    "prefix": row.event_prefix_len,
                    "applied": row.trajectory_step,
                });
                let bytes =
                    hydra_feed::canon::canonical_bytes_value(&doc, "rl:transition-provenance")
                        .expect("canon preimage");
                let expected: [u8; 32] = sha2::Sha256::digest(&bytes).into();
                assert_eq!(row.transition_provenance, expected);
            }
            // Distinctness with teeth: the provenances differ from each other
            // AND from the old law's single shared value (frozen applied
            // count), which would have stamped every row identically.
            let mut sorted: Vec<[u8; 32]> =
                rows.iter().map(|row| row.transition_provenance).collect();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), rows.len());
            let stale_doc = serde_json::json!({
                "game": rows[0].game_id.clone(),
                "ordinal": rows[0].round_index,
                "prefix": rows[0].event_prefix_len,
                "applied": frozen_applied,
            });
            let stale_bytes =
                hydra_feed::canon::canonical_bytes_value(&stale_doc, "rl:transition-provenance")
                    .expect("canon stale preimage");
            let stale: [u8; 32] = sha2::Sha256::digest(&stale_bytes).into();
            for row in &rows {
                assert_ne!(row.transition_provenance, stale);
            }
            return;
        }
        panic!("named wall must exercise a multi-responder claim window");
    }
