//! Action census: frozen kind ordinals + full canonical action-template enumeration (every structurally valid template exactly once; canonical bytes and digest freeze the indices).
//!
//! Mirrors `src/hydra2/contracts/action_kinds.py` (`ACTION_KIND_ORDINALS`,
//! never reordered) and `src/hydra2/contracts/action_model.py`
//! (`generate_action_templates`, `template_sort_key`) without copying them:
//! the analytic census (pass 4, discards 136 x3, chi 4032, pon 1224,
//! daiminkan 408, ankan 34, kakan 136, ron 408, tsumo 136, aborts 1 x2 =
//! 6792 templates) is regenerated here from the same loops, and the sort key
//! (ordinal, none-first tile/called/offset, consumed tuple, riichi/meld
//! flags) is reproduced as a total [`Ord`] so generation order is
//! bit-identical to the Python census.
//!
//! Design notes (Wave 4 rank 4, census):
//! - The census is a pure function of nothing (no seed, no clock, no I/O):
//!   counter-free determinism by construction.
//! - Search is key-based: [`census_index_of`] binary-searches a caller-built
//!   probe, exactly like `ActionTable.index_of` (bisect on sort keys, `None`
//!   when absent). Shape validation lives in the Python record constructor;
//!   the probe carries kind-derived riichi/meld flags, so only genuinely
//!   present templates hit.
//! - Counts come from the generated census (never hardcoded), so a loop edit
//!   that changes the census fails the count tests instead of drifting.

use core::cmp::Ordering;
use std::sync::LazyLock;

/// Analytic census size: 4 + 3*136 + 4032 + 1224 + 408 + 34 + 136 + 408 + 136 + 2.
pub const CENSUS_TOTAL: usize = 6792;

/// Process-once frozen census: `generate_census()` computed a single time per
/// process (feed/validate.rs `SCHEMA_OK` precedent). `Sync` + immutable, so
/// sharing `&'static [ActionTemplate]` across threads (incl. `py.detach`
/// workers) is safe; fork children re-init on first touch if untouched.
static FROZEN_CENSUS: LazyLock<Vec<ActionTemplate>> = LazyLock::new(generate_census);

/// Borrow the frozen process-once census (identical by construction to
/// `generate_census()` output, so indices coincide bit-for-bit).
pub fn frozen_census() -> &'static [ActionTemplate] {
    &FROZEN_CENSUS
}

/// Action kinds in frozen ordinal order (pass 0, discard 1, tsumogiri 2, riichi_discard 3, chi 4, pon 5, daiminkan 6, ankan 7, kakan 8, ron 9, tsumo 10, abort_nine_terminals 11, accept_abortive_draw 12; `action_kinds.py:71-85`).
/// Discriminants ARE the ordinals: never reordered, never extended in place.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Pass = 0,
    Discard = 1,
    Tsumogiri = 2,
    RiichiDiscard = 3,
    Chi = 4,
    Pon = 5,
    Daiminkan = 6,
    Ankan = 7,
    Kakan = 8,
    Ron = 9,
    Tsumo = 10,
    AbortNineTerminals = 11,
    AcceptAbortiveDraw = 12,
}

impl ActionKind {
/// Frozen kind ordinal (the discriminant: pass 0 through accept_abortive_draw 12, never reordered).
    pub fn ordinal(self) -> u8 {
        self as u8
    }

    /// Frozen kind literal.
    pub fn name(self) -> &'static str {
        match self {
            ActionKind::Pass => "pass",
            ActionKind::Discard => "discard",
            ActionKind::Tsumogiri => "tsumogiri",
            ActionKind::RiichiDiscard => "riichi_discard",
            ActionKind::Chi => "chi",
            ActionKind::Pon => "pon",
            ActionKind::Daiminkan => "daiminkan",
            ActionKind::Ankan => "ankan",
            ActionKind::Kakan => "kakan",
            ActionKind::Ron => "ron",
            ActionKind::Tsumo => "tsumo",
            ActionKind::AbortNineTerminals => "abort_nine_terminals",
            ActionKind::AcceptAbortiveDraw => "accept_abortive_draw",
        }
    }

    /// Ordinal to kind; `None` outside `0..=12`.
    pub fn from_ordinal(ordinal: u8) -> Option<Self> {
        match ordinal {
            0 => Some(ActionKind::Pass),
            1 => Some(ActionKind::Discard),
            2 => Some(ActionKind::Tsumogiri),
            3 => Some(ActionKind::RiichiDiscard),
            4 => Some(ActionKind::Chi),
            5 => Some(ActionKind::Pon),
            6 => Some(ActionKind::Daiminkan),
            7 => Some(ActionKind::Ankan),
            8 => Some(ActionKind::Kakan),
            9 => Some(ActionKind::Ron),
            10 => Some(ActionKind::Tsumo),
            11 => Some(ActionKind::AbortNineTerminals),
            12 => Some(ActionKind::AcceptAbortiveDraw),
            _ => None,
        }
    }

    /// Kind literal to kind; `None` for unknown literals.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "pass" => Some(ActionKind::Pass),
            "discard" => Some(ActionKind::Discard),
            "tsumogiri" => Some(ActionKind::Tsumogiri),
            "riichi_discard" => Some(ActionKind::RiichiDiscard),
            "chi" => Some(ActionKind::Chi),
            "pon" => Some(ActionKind::Pon),
            "daiminkan" => Some(ActionKind::Daiminkan),
            "ankan" => Some(ActionKind::Ankan),
            "kakan" => Some(ActionKind::Kakan),
            "ron" => Some(ActionKind::Ron),
            "tsumo" => Some(ActionKind::Tsumo),
            "abort_nine_terminals" => Some(ActionKind::AbortNineTerminals),
            "accept_abortive_draw" => Some(ActionKind::AcceptAbortiveDraw),
            _ => None,
        }
    }
}

/// One canonical action slot, mirroring `CanonicalActionTemplate` (kind, tile, called tile, consumed tiles, source offset relative modulo four, riichi flag, meld-ref flag; actor is contextual and excluded from template identity).
///
/// `consumed_len` is always `<= 4` (chi/pon 2, daiminkan 3, ankan 4, else 0);
/// only `consumed[..consumed_len]` is significant. `declares_riichi` is true
/// exactly for riichi-discard and `meld_ref_required` exactly for kakan, as
/// enforced by the Python record constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionTemplate {
    pub kind: ActionKind,
    pub tile: Option<u8>,
    pub called_tile: Option<u8>,
    pub consumed: [u8; 4],
    pub consumed_len: u8,
    pub source_offset: Option<i8>,
    pub declares_riichi: bool,
    pub meld_ref_required: bool,
}

impl ActionTemplate {
    /// Significant consumed tiles (unique ascending, like the Python tuple).
    pub fn consumed_slice(&self) -> &[u8] {
        // `consumed_len <= 4` at every construction site (generator loops and
        // the bridge probe gate), so this slice never overruns.
        &self.consumed[..self.consumed_len as usize]
    }
}

/// Generation order: lexicographic over (kind ordinal, tile, called tile, consumed tiles, source offset, riichi flag, meld-ref flag) with `None` before integers
/// (`template_sort_key`). `Option`'s `Ord` (`None < Some`) is exactly the
/// `none_first_int` mapping, and slice `Ord` is tuple-lexicographic, so this
/// `Ord` is the Python key bit-for-bit.
impl Ord for ActionTemplate {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.kind.ordinal(), &self.tile, &self.called_tile)
            .cmp(&(
                other.kind.ordinal(),
                &other.tile,
                &other.called_tile,
            ))
            .then_with(|| self.consumed_slice().cmp(other.consumed_slice()))
            .then_with(|| self.source_offset.cmp(&other.source_offset))
            .then_with(|| {
                self.declares_riichi
                    .cmp(&other.declares_riichi)
                    .then_with(|| self.meld_ref_required.cmp(&other.meld_ref_required))
            })
    }
}

impl PartialOrd for ActionTemplate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Append one template; `consumed` holds at most 4 tiles at every call site.
fn push(
    templates: &mut Vec<ActionTemplate>,
    kind: ActionKind,
    tile: Option<u8>,
    called_tile: Option<u8>,
    consumed: &[u8],
    source_offset: Option<i8>,
) {
    let mut arr = [0u8; 4];
    let mut n = 0usize;
    for &t in consumed {
        // Every call site passes 0, 2, 3, or 4 tiles; the census never holds
        // a longer group (ankan's full quad is the maximum).
        arr[n] = t;
        n += 1;
    }
    templates.push(ActionTemplate {
        kind,
        tile,
        called_tile,
        consumed: arr,
        consumed_len: u8::try_from(n).unwrap_or(u8::MAX),
        source_offset,
        declares_riichi: kind == ActionKind::RiichiDiscard,
        meld_ref_required: kind == ActionKind::Kakan,
    });
}

/// Enumerate every and only structurally valid template once, in generation
/// order (mirrors `generate_action_templates`, `action_model.py:276-404`).
pub fn generate_census() -> Vec<ActionTemplate> {
    let mut templates: Vec<ActionTemplate> = Vec::with_capacity(CENSUS_TOTAL);
    // pass x 4 offsets (None, -1, 1, 2).
    for offset in [None, Some(-1), Some(1), Some(2)] {
        push(&mut templates, ActionKind::Pass, None, None, &[], offset);
    }
    // discard / tsumogiri / riichi_discard x 136 tiles.
    for kind in [
        ActionKind::Discard,
        ActionKind::Tsumogiri,
        ActionKind::RiichiDiscard,
    ] {
        for tile in 0..136u8 {
            push(&mut templates, kind, Some(tile), None, &[], None);
        }
    }
    // chi x 4032: 3 suits x 7 lows x 3 called positions x 4^3 copies.
    for suit_base in [0u8, 9, 18] {
        for low in 0..7u8 {
            let run = [suit_base + low, suit_base + low + 1, suit_base + low + 2];
            for position in 0..3usize {
                let called_type = run[position];
                // The two remaining run types, ascending (Python slices the
                // run around `position`, which preserves ascending order).
                let mut others = [0u8; 2];
                let mut k = 0usize;
                for (i, &t) in run.iter().enumerate() {
                    if i != position {
                        others[k] = t;
                        k += 1;
                    }
                }
                for called_copy in 0..4u8 {
                    let called = 4 * called_type + called_copy;
                    for copy_a in 0..4u8 {
                        for copy_b in 0..4u8 {
                            let a = 4 * others[0] + copy_a;
                            let b = 4 * others[1] + copy_b;
                            // Distinct physical ids (different types), ascending.
                            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                            push(
                                &mut templates,
                                ActionKind::Chi,
                                None,
                                Some(called),
                                &[lo, hi],
                                Some(-1),
                            );
                        }
                    }
                }
            }
        }
    }
    // pon x 1224 + daiminkan x 408: called tile x 3 pairs/triple x 3 offsets.
    for called in 0..136u8 {
        let ctype = called / 4;
        let base = 4 * ctype;
        // The three same-type siblings, ascending.
        let mut others = [0u8; 3];
        let mut k = 0usize;
        for c in 0..4u8 {
            let t = base + c;
            if t != called {
                others[k] = t;
                k += 1;
            }
        }
        for offset in [Some(-1), Some(1), Some(2)] {
            // Python emits the three pon pairs then the daiminkan per offset.
            push(
                &mut templates,
                ActionKind::Pon,
                None,
                Some(called),
                &[others[0], others[1]],
                offset,
            );
            push(
                &mut templates,
                ActionKind::Pon,
                None,
                Some(called),
                &[others[0], others[2]],
                offset,
            );
            push(
                &mut templates,
                ActionKind::Pon,
                None,
                Some(called),
                &[others[1], others[2]],
                offset,
            );
            push(
                &mut templates,
                ActionKind::Daiminkan,
                None,
                Some(called),
                &[others[0], others[1], others[2]],
                offset,
            );
        }
    }
    // ankan x 34: the full quad of each logical type.
    for tile_type in 0..34u8 {
        let base = 4 * tile_type;
        push(
            &mut templates,
            ActionKind::Ankan,
            None,
            None,
            &[base, base + 1, base + 2, base + 3],
            None,
        );
    }
    // kakan x 136: added tile only.
    for tile in 0..136u8 {
        push(&mut templates, ActionKind::Kakan, Some(tile), None, &[], None);
    }
    // ron x 408: offered winning tile x 3 offsets.
    for tile in 0..136u8 {
        for offset in [Some(-1), Some(1), Some(2)] {
            push(&mut templates, ActionKind::Ron, Some(tile), None, &[], offset);
        }
    }
    // tsumo x 136.
    for tile in 0..136u8 {
        push(&mut templates, ActionKind::Tsumo, Some(tile), None, &[], None);
    }
    // Both abort kinds x 1: parameterless.
    push(
        &mut templates,
        ActionKind::AbortNineTerminals,
        None,
        None,
        &[],
        None,
    );
    push(
        &mut templates,
        ActionKind::AcceptAbortiveDraw,
        None,
        None,
        &[],
        None,
    );
    // Stable sort into generation order (keys are total, so stability only
    // mirrors Python's `sorted` for identical keys, which cannot occur for
    // distinct templates).
    templates.sort();
    templates
}

/// Per-kind template counts in frozen ordinal order, counted from the
/// generated census (never hardcoded, so loop edits fail loudly here).
pub fn census_counts(census: &[ActionTemplate]) -> [(ActionKind, usize); 13] {
    let mut counts = [0usize; 13];
    for template in census {
        counts[template.kind.ordinal() as usize] += 1;
    }
    [
        (ActionKind::Pass, counts[0]),
        (ActionKind::Discard, counts[1]),
        (ActionKind::Tsumogiri, counts[2]),
        (ActionKind::RiichiDiscard, counts[3]),
        (ActionKind::Chi, counts[4]),
        (ActionKind::Pon, counts[5]),
        (ActionKind::Daiminkan, counts[6]),
        (ActionKind::Ankan, counts[7]),
        (ActionKind::Kakan, counts[8]),
        (ActionKind::Ron, counts[9]),
        (ActionKind::Tsumo, counts[10]),
        (ActionKind::AbortNineTerminals, counts[11]),
        (ActionKind::AcceptAbortiveDraw, counts[12]),
    ]
}

/// Generation-order index of `probe`, or `None` when absent (mirrors
/// `ActionTable.index_of`: bisect on sort keys, never a partial output).
pub fn census_index_of(census: &[ActionTemplate], probe: &ActionTemplate) -> Option<usize> {
    census.binary_search(probe).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic per-kind census (`test_action_wp02c.py::EXPECTED_COUNTS`).
    const EXPECTED: [(u8, usize); 13] = [
        (0, 4),
        (1, 136),
        (2, 136),
        (3, 136),
        (4, 4032),
        (5, 1224),
        (6, 408),
        (7, 34),
        (8, 136),
        (9, 408),
        (10, 136),
        (11, 1),
        (12, 1),
    ];

    #[test]
    fn census_total_is_6792() {
        assert_eq!(generate_census().len(), CENSUS_TOTAL);
    }

    #[test]
    fn census_per_kind_counts_match_analytic() {
        let census = generate_census();
        let counts = census_counts(&census);
        for (i, (kind, count)) in counts.iter().enumerate() {
            assert_eq!(kind.ordinal(), EXPECTED[i].0);
            assert_eq!(*count, EXPECTED[i].1, "kind {}", kind.name());
        }
    }

    #[test]
    fn census_is_in_generation_order() {
        let census = generate_census();
        for pair in census.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn census_generation_is_deterministic() {
        assert_eq!(generate_census(), generate_census());
    }

    #[test]
    fn kind_round_trips_by_ordinal_and_name() {
        for ordinal in 0..13u8 {
            let kind = ActionKind::from_ordinal(ordinal).unwrap();
            assert_eq!(kind.ordinal(), ordinal);
            assert_eq!(ActionKind::from_name(kind.name()).unwrap(), kind);
        }
        assert_eq!(ActionKind::from_ordinal(13), None);
        assert_eq!(ActionKind::from_name("pung"), None);
    }

    #[test]
    fn index_of_finds_present_templates() {
        let census = generate_census();
        // pass with offset None sorts first (`None` before integers).
        let probe = ActionTemplate {
            kind: ActionKind::Pass,
            tile: None,
            called_tile: None,
            consumed: [0u8; 4],
            consumed_len: 0,
            source_offset: None,
            declares_riichi: false,
            meld_ref_required: false,
        };
        assert_eq!(census_index_of(&census, &probe), Some(0));
        // pass with offset -1 sorts second.
        let probe_m1 = ActionTemplate {
            source_offset: Some(-1),
            ..probe
        };
        assert_eq!(census_index_of(&census, &probe_m1), Some(1));
        // daiminkan(408) + ankan(34) precede kakan in ordinal order.
        let kakan = ActionTemplate {
            kind: ActionKind::Kakan,
            tile: Some(0),
            called_tile: None,
            consumed: [0u8; 4],
            consumed_len: 0,
            source_offset: None,
            declares_riichi: false,
            meld_ref_required: true,
        };
        let idx = census_index_of(&census, &kakan).unwrap();
        assert_eq!(census[idx], kakan);
        assert_eq!(idx, 4 + 3 * 136 + 4032 + 1224 + 408 + 34);
    }

    #[test]
    fn index_of_misses_absent_shape() {
        let census = generate_census();
        // chi with a non-previous-seat offset is never generated.
        let probe = ActionTemplate {
            kind: ActionKind::Chi,
            tile: None,
            called_tile: Some(4),
            consumed: [0, 8, 0, 0],
            consumed_len: 2,
            source_offset: Some(1),
            declares_riichi: false,
            meld_ref_required: false,
        };
        assert_eq!(census_index_of(&census, &probe), None);
    }
}
