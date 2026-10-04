//! rl/core/tests: native behavior tests for the hanchan driver.
//!
//! Seating accepts only cyclic rotations; bit packing is least-significant-bit
//! first over canonical action order; a full game terminates with dense hand
//! ordinals `0..N-1` and valid placement ranks; replaying the applied prefix
//! against identical walls reproduces the state digest.

    use super::*;

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
