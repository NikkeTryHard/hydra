use hydra_rules::census::ActionKind;

use super::super::{ACTION, TOKENS, WIDTH};
use super::support::{Log, action, draw, encode_pair, legal, observation};

fn non_id_equal(left: &[i32], right: &[i32], row: usize) {
    for column in 1..ACTION {
        assert_eq!(
            left[row * ACTION + column],
            right[row * ACTION + column],
            "descriptor column {column} differs at row {row}"
        );
    }
}

#[test]
fn physical_copy_relabeling_preserves_observed_semantics() {
    let mut first_log = Log::new();
    first_log.push(draw(0, 54));
    let mut first = observation(0, "draw_decision");
    first.concealed_hand = vec![53];
    first.own_drawn_tile = Some(54);
    first.history_indices = vec![0, 1];
    let first_legal = legal(vec![
        action(ActionKind::Discard, Some(53), None, &[], None),
        action(ActionKind::Tsumogiri, Some(54), None, &[], None),
    ]);

    let mut second_log = Log::new();
    second_log.push(draw(0, 53));
    let mut second = observation(0, "draw_decision");
    second.concealed_hand = vec![54];
    second.own_drawn_tile = Some(53);
    second.history_indices = vec![0, 1];
    let second_legal = legal(vec![
        action(ActionKind::Discard, Some(54), None, &[], None),
        action(ActionKind::Tsumogiri, Some(53), None, &[], None),
    ]);

    let (first_tokens, first_candidates) = encode_pair(&first, first_log.arena(), &first_legal);
    let (second_tokens, second_candidates) =
        encode_pair(&second, second_log.arena(), &second_legal);

    assert_eq!(first_tokens, second_tokens);
    non_id_equal(&first_candidates, &second_candidates, 0);
    non_id_equal(&first_candidates, &second_candidates, 1);
    assert_ne!(first_candidates[0], second_candidates[0]);
    assert_ne!(first_candidates[ACTION], second_candidates[ACTION]);
    assert_eq!(first_tokens.len(), TOKENS * WIDTH);
}
