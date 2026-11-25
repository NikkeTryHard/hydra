use crate::view::EventArena;
use hydra_rules::census::ActionKind;

use super::super::ACTION;
use super::super::layout::desc;
use super::support::{action, descriptor, encode_pair, legal, observation};

struct Expected {
    id: i32,
    kind: ActionKind,
    source: i32,
    face: i32,
    red: i32,
    from_draw: i32,
    riichi: i32,
    contested: i32,
    removals: [i32; 4],
    removal_red: [i32; 4],
}

fn expect(candidates: &[i32], row: usize, expected: Expected) {
    assert_eq!(descriptor(candidates, row, desc::ID), expected.id);
    assert_eq!(
        descriptor(candidates, row, desc::KIND),
        i32::from(expected.kind.ordinal())
    );
    assert_eq!(descriptor(candidates, row, desc::SOURCE), expected.source);
    assert_eq!(descriptor(candidates, row, desc::FACE), expected.face);
    assert_eq!(descriptor(candidates, row, desc::RED), expected.red);
    assert_eq!(
        descriptor(candidates, row, desc::FROM_DRAW),
        expected.from_draw
    );
    assert_eq!(descriptor(candidates, row, desc::RIICHI), expected.riichi);
    assert_eq!(
        descriptor(candidates, row, desc::CONTESTED),
        expected.contested
    );
    for slot in 0..4 {
        assert_eq!(
            descriptor(candidates, row, desc::REMOVALS + slot),
            expected.removals[slot]
        );
        assert_eq!(
            descriptor(candidates, row, desc::REMOVAL_RED + slot),
            expected.removal_red[slot]
        );
    }
}

#[test]
fn descriptors_cover_first_real_template_of_every_kind() {
    let actions = legal(vec![
        action(ActionKind::Pass, None, None, &[], None),
        action(ActionKind::Discard, Some(0), None, &[], None),
        action(ActionKind::Tsumogiri, Some(0), None, &[], None),
        action(ActionKind::RiichiDiscard, Some(0), None, &[], None),
        action(ActionKind::Chi, None, Some(0), &[4, 8], Some(3)),
        action(ActionKind::Pon, None, Some(0), &[1, 2], Some(3)),
        action(ActionKind::Daiminkan, None, Some(0), &[1, 2, 3], Some(3)),
        action(ActionKind::Ankan, None, None, &[0, 1, 2, 3], None),
        action(ActionKind::Kakan, Some(0), None, &[], None),
        action(ActionKind::Ron, Some(0), None, &[], Some(3)),
        action(ActionKind::Tsumo, Some(0), None, &[], None),
        action(ActionKind::AbortNineTerminals, None, None, &[], None),
        action(ActionKind::AcceptAbortiveDraw, None, None, &[], None),
    ]);
    let obs = observation(0, "draw_decision");
    let arena: EventArena = Vec::new();
    let (_, candidates) = encode_pair(&obs, &arena, &actions);
    assert_eq!(candidates.len(), actions.len() * ACTION);

    let expected = [
        Expected {
            id: 0,
            kind: ActionKind::Pass,
            source: 4,
            face: -1,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [-1; 4],
            removal_red: [0; 4],
        },
        Expected {
            id: 4,
            kind: ActionKind::Discard,
            source: 4,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [0, -1, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 140,
            kind: ActionKind::Tsumogiri,
            source: 4,
            face: 0,
            red: 0,
            from_draw: 1,
            riichi: 0,
            contested: 0,
            removals: [0, -1, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 276,
            kind: ActionKind::RiichiDiscard,
            source: 4,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 1,
            contested: 0,
            removals: [0, -1, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 412,
            kind: ActionKind::Chi,
            source: 3,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 1,
            removals: [1, 2, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 4444,
            kind: ActionKind::Pon,
            source: 3,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 1,
            removals: [0, 0, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 5668,
            kind: ActionKind::Daiminkan,
            source: 3,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 1,
            removals: [0, 0, 0, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 6076,
            kind: ActionKind::Ankan,
            source: 4,
            face: -1,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [0, 0, 0, 0],
            removal_red: [0; 4],
        },
        Expected {
            id: 6110,
            kind: ActionKind::Kakan,
            source: 4,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [0, -1, -1, -1],
            removal_red: [0; 4],
        },
        Expected {
            id: 6246,
            kind: ActionKind::Ron,
            source: 3,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 1,
            removals: [-1; 4],
            removal_red: [0; 4],
        },
        Expected {
            id: 6654,
            kind: ActionKind::Tsumo,
            source: 4,
            face: 0,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [-1; 4],
            removal_red: [0; 4],
        },
        Expected {
            id: 6790,
            kind: ActionKind::AbortNineTerminals,
            source: 4,
            face: -1,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [-1; 4],
            removal_red: [0; 4],
        },
        Expected {
            id: 6791,
            kind: ActionKind::AcceptAbortiveDraw,
            source: 4,
            face: -1,
            red: 0,
            from_draw: 0,
            riichi: 0,
            contested: 0,
            removals: [-1; 4],
            removal_red: [0; 4],
        },
    ];
    for (row, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            descriptor(&candidates, row, desc::ID),
            actions[row].0 as i32
        );
        expect(&candidates, row, expected);
    }
}
