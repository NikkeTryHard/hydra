//! action_model: frozen SPEC 6.3 template sort-key leaf on the shared `contracts` submodule.
//!
//! TABLE owner for `python/hydra2/contracts/action_model.py` pure template-math:
//! the SPEC 6.3 generation-order key (`template_sort_key`, `action_model.py:261-276`).
//! The census combinatorics already live in `hydra-feed` (`census::generate_census`,
//! `census.rs:214`; bridged as `action_census` at `contracts.rs:320`) and the kind
//! ordinals already live in `contracts::ACTION_KIND_TABLE` (`contracts.rs:175-189`),
//! so this module ports ONLY the key assembly over plain values (no second census,
//! no second kind table). There are NO module-level frozen UPPER consts in
//! `action_model.py` (`rg '^[A-Z_]+ ?=' returns nothing) — values table is NONE.
//!
//! - `template_sort_key_for` <- `template_sort_key` (`action_model.py:261-276`)
//!   value-level half: the caller passes the seven record parts as scalars
//!   (kind + tile/called/consumed/offset + two flags, never live
//!   `CanonicalActionTemplate` objects — the `PacketSuccessor` precedent at
//!   `contracts.rs:259-261`); Rust assembles the oracle-identical key tuple
//!   `(ordinal, none-first tile, none-first called, consumed tuple,
//!   none-first offset, declares, meld)`. Ordinals resolve through the feed
//!   owner (`ActionKind::from_name` + `ordinal`, `census.rs:62-66,108-125` —
//!   never a second table); `None` maps to `(0, 0)` and `Some(v)` to `(1, v)`
//!   exactly like the oracle `none_first_int` closure. Range authority stays
//!   in the record constructor: tile/offset values assemble verbatim (no
//!   `0..=135` / `-1..=2` re-check) so ordering matches the oracle for every
//!   plain int; only malformed shapes fail closed (`ValueError` for unknown
//!   kinds and bool/non-int tile parts, `TypeError` for non-str kinds and
//!   non-bool flags via the extractors).
//!
//! LOGIC staying Python (`action_model.py`): the `CanonicalAction` /
//! `CanonicalActionTemplate` dataclasses with their SPEC 6.2/6.3 `__post_init__`
//! validators (record validating roots per the `contracts.rs:21-26` rank-3 note),
//! the `SourceOffset` Literal, `generate_action_templates` (already a thin
//! hard-Rust delegate over `action_census`), `_require_census_bridge`, and every
//! `__all__` entry. `template_sort_key` itself stays Python as a thin translator
//! over this leaf (same signature, positional call, `ImportError` on stale `.so`).
//!
//! NONE-owner note: `rg 'template_sort|CENSUS_TOTAL|action_model' over
//! `crates/bridge/src` hits only the feed `census.rs` Ord docs plus the shard
//! `table.rs` private sorter (no bridge pyfn, no const); no crate owns the key
//! tuple, so this leaf is fresh.
//!
//! Shape per fn: attached staging (plain-int gates with bool excluded, kind
//! resolution) -> ONE `py.detach(|| ...)` over owned plain data with zero Python
//! API inside (per `contracts.rs:434-437`, `rc_require.rs:827-844`) -> attached
//! wrap as nested `PyTuple`s (per `contracts.rs:1156`, `contracts.rs:1281-1287`)
//! with `ValueError` on any reject (per `contracts.rs:117-123`). Fns via
//! `wrap_pyfunction!(f, sub)` (per `validate.rs:123`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_require.rs:846-862`); no new entry
//! point. MAIN wiring: `pub mod action_model;` in `lib.rs` plus
//! `crate::action_model::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1300`).

// Precedent `use pyo3::prelude::*;` -> contracts.rs:52.
use pyo3::prelude::*;
// Precedent `PyAny` + `PyBool` + `PyModule` + `PyTuple` imports -> contracts.rs:53.
use pyo3::types::{PyAny, PyBool, PyModule, PyTuple};
// Precedent `PyValueError` -> contracts.rs:51.
use pyo3::exceptions::PyValueError;

/// Plain-int view of a Python value: `bool` is excluded first (it subclasses
/// `int`), non-ints and out-of-`i64` values map to `None` (callers raise).
/// Precedent -> contracts.rs:69-74.
fn plain_int_value(obj: &Bound<'_, PyAny>) -> Option<i64> {
    if obj.is_instance_of::<PyBool>() {
        return None;
    }
    obj.extract::<i64>().ok()
}

/// Fail-closed plain-int gate. Precedent -> contracts.rs:77-84.
fn plain_int_or_err(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    match plain_int_value(obj) {
        Some(value) => Ok(value),
        None => Err(PyValueError::new_err(format!(
            "contracts {name} must be a plain int (bool excluded)"
        ))),
    }
}

/// Optional plain-int gate shared by tile/called/offset (`None` stays `None`).
/// Precedent `opt_tile` -> contracts.rs:344-357 (without the range re-check:
/// range authority stays in the record constructor, see module docs).
fn opt_int(value: Option<Bound<'_, PyAny>>, name: &str) -> PyResult<Option<i64>> {
    match value {
        None => Ok(None),
        Some(obj) => plain_int_or_err(&obj, name).map(Some),
    }
}

/// Consumed-tiles gate: every entry must be a plain int (bool excluded).
/// Precedent consumed loop -> contracts.rs:382-391 (without the range re-check).
fn consumed_ints(values: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<i64>> {
    values
        .iter()
        .map(|entry| plain_int_or_err(entry, "consumed_tiles"))
        .collect()
}

/// `None`-before-integers mapping (`none_first_int`, `action_model.py:265-266`).
/// `Option`'s `Ord` (`None < Some`) is exactly this mapping (per `census.rs:155-158`);
/// pure owned-data fn, detach-safe. Precedent -> shard `table.rs:96-98`.
fn none_first(value: Option<i64>) -> (u8, i64) {
    match value {
        None => (0, 0),
        Some(v) => (1, v),
    }
}

/// SPEC 6.3 generation-order key over the seven record parts (mirrors
/// `template_sort_key`, `action_model.py:261-276`, without the live object —
/// callers pass scalars, the `PacketSuccessor` precedent at `contracts.rs:259-261`).
/// Ordinals resolve through the feed owner (`ActionKind::from_name` +
/// `ordinal`); unknown kinds fail closed. Staging runs attached, the tuple
/// mapping runs under ONE detach with zero Python API inside, and the nested
/// `PyTuple` wrap runs attached. Fail-closed: every shape violation is
/// `PyValueError` (or extractor `TypeError`), never a default.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn template_sort_key_for<'py>(
    py: Python<'py>,
    kind: &str,
    tile: Option<Bound<'py, PyAny>>,
    called_tile: Option<Bound<'py, PyAny>>,
    consumed_tiles: Vec<Bound<'py, PyAny>>,
    source_offset: Option<Bound<'py, PyAny>>,
    declares_riichi: bool,
    meld_ref_required: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Attached staging (precedent -> contracts.rs:372-403): resolve the ordinal
    // through the feed owner, gate every int part with bool excluded.
    let ordinal: u8 = hydra_feed::census::ActionKind::from_name(kind)
        .map(|k| k.ordinal())
        // Precedent error text -> contracts.rs:199-204.
        .ok_or_else(|| PyValueError::new_err(format!("contracts unknown action kind {kind:?}")))?;
    let tile = opt_int(tile, "tile")?;
    let called_tile = opt_int(called_tile, "called_tile")?;
    let consumed = consumed_ints(consumed_tiles)?;
    let source_offset = opt_int(source_offset, "source_offset")?;
    // ONE detach over owned plain data with zero Python API inside
    // (precedent -> rc_require.rs:827-844).
    let (ordinal, tile_key, called_key, consumed, offset_key, declares_riichi, meld_ref_required) =
        py.detach(|| {
            let tile_key = none_first(tile);
            let called_key = none_first(called_tile);
            let offset_key = none_first(source_offset);
            (
                ordinal,
                tile_key,
                called_key,
                consumed,
                offset_key,
                declares_riichi,
                meld_ref_required,
            )
        });
    // Attached wrap as nested tuples (precedents -> contracts.rs:1156 for
    // `PyTuple::new(py, [...])`; contracts.rs:1281-1287 for tuple-of-tuples;
    // contracts.rs:1101-1102 for `into_pyobject(py)?.into_any()` mixed elems).
    let tile_py = PyTuple::new(py, [tile_key.0 as i64, tile_key.1])?;
    let called_py = PyTuple::new(py, [called_key.0 as i64, called_key.1])?;
    let consumed_py = PyTuple::new(py, consumed)?;
    let offset_py = PyTuple::new(py, [offset_key.0 as i64, offset_key.1])?;
    let outer = PyTuple::new(
        py,
        [
            ordinal.into_pyobject(py)?.into_any().unbind(),
            tile_py.into_any().unbind(),
            called_py.into_any().unbind(),
            consumed_py.into_any().unbind(),
            offset_py.into_any().unbind(),
            declares_riichi
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind(),
            meld_ref_required
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind(),
        ],
    )?;
    Ok(outer.into_any())
}

/// Register the action-model sort-key leaf on the shared `contracts` submodule
/// (mirrors `validate.rs:111-127`): compute detached, wrap attached; single
/// cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(template_sort_key_for, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sort-key tuple: `(ordinal, tile, called, consumed, offset, declares, meld)`.
    type SortKey = (u8, (i64, i64), (i64, i64), Vec<i64>, (i64, i64), bool, bool);

    #[test]
    fn kind_ordinals_match_frozen_spec_order() {
        // Hand-derived from the HEAD oracle (`action_kinds.py:71-85`,
        // `contracts.rs:175-189`): frozen SPEC 6.1 order, never reordered.
        let want: [(&str, u8); 13] = [
            ("pass", 0),
            ("discard", 1),
            ("tsumogiri", 2),
            ("riichi_discard", 3),
            ("chi", 4),
            ("pon", 5),
            ("daiminkan", 6),
            ("ankan", 7),
            ("kakan", 8),
            ("ron", 9),
            ("tsumo", 10),
            ("abort_nine_terminals", 11),
            ("accept_abortive_draw", 12),
        ];
        for (name, ordinal) in want {
            let kind = hydra_feed::census::ActionKind::from_name(name).unwrap();
            assert_eq!(kind.ordinal(), ordinal);
            assert_eq!(kind.name(), name);
        }
        assert!(hydra_feed::census::ActionKind::from_name("bogus").is_none());
        assert!(hydra_feed::census::ActionKind::from_name("Pass").is_none());
    }

    #[test]
    fn none_first_maps_none_before_integers() {
        // Hand-derived from the oracle closure (`action_model.py:265-266`).
        assert_eq!(none_first(None), (0, 0));
        assert_eq!(none_first(Some(0)), (1, 0));
        assert_eq!(none_first(Some(135)), (1, 135));
        assert_eq!(none_first(Some(-1)), (1, -1));
        assert_eq!(none_first(Some(2)), (1, 2));
    }

    #[test]
    fn sort_key_leaf_matches_oracle_tuples() {
        Python::initialize();
        Python::attach(|py| {
            // pass, parameterless with no offset (oracle: ordinal 0, all
            // none-first empty, both flags false).
            let got: SortKey =
                template_sort_key_for(py, "pass", None, None, vec![], None, false, false)
                    .unwrap()
                    .extract()
                    .unwrap();
            assert_eq!(got, (0, (0, 0), (0, 0), vec![], (0, 0), false, false));
            // discard tile 0 (oracle: ordinal 1, tile (1, 0), rest empty).
            let got: SortKey = template_sort_key_for(
                py,
                "discard",
                Some(0i64.into_pyobject(py).unwrap().into_any()),
                None,
                vec![],
                None,
                false,
                false,
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(got, (1, (1, 0), (0, 0), vec![], (0, 0), false, false));
            // chi run with previous-seat offset (oracle: ordinal 4, called
            // (1, 24), consumed (28, 32), offset (1, -1)). Tiles 24/28/32 are
            // types 6/7/8 (same-suit consecutive), a genuinely valid template.
            let consumed = vec![
                28i64.into_pyobject(py).unwrap().into_any(),
                32i64.into_pyobject(py).unwrap().into_any(),
            ];
            let got: SortKey = template_sort_key_for(
                py,
                "chi",
                None,
                Some(24i64.into_pyobject(py).unwrap().into_any()),
                consumed,
                Some((-1i64).into_pyobject(py).unwrap().into_any()),
                false,
                false,
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(
                got,
                (4, (0, 0), (1, 24), vec![28, 32], (1, -1), false, false)
            );
            // riichi_discard carries declares_riichi (oracle embeds flags verbatim).
            let got: SortKey = template_sort_key_for(
                py,
                "riichi_discard",
                Some(5i64.into_pyobject(py).unwrap().into_any()),
                None,
                vec![],
                None,
                true,
                false,
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(got, (3, (1, 5), (0, 0), vec![], (0, 0), true, false));
        });
    }

    #[test]
    fn sort_key_leaf_rejects_malformed_shapes() {
        Python::initialize();
        Python::attach(|py| {
            // Unknown kind fails closed (precedent -> contracts.rs:199-204).
            assert!(
                template_sort_key_for(py, "bogus", None, None, vec![], None, false, false).is_err()
            );
            // Bool tile is not a plain int (precedent -> contracts.rs:69-74).
            let bool_tile = true.into_pyobject(py).unwrap().to_owned().into_any();
            assert!(
                template_sort_key_for(
                    py,
                    "discard",
                    Some(bool_tile),
                    None,
                    vec![],
                    None,
                    false,
                    false
                )
                .is_err()
            );
            // String tile is not a plain int.
            let str_tile = "0".into_pyobject(py).unwrap().into_any();
            assert!(
                template_sort_key_for(
                    py,
                    "discard",
                    Some(str_tile),
                    None,
                    vec![],
                    None,
                    false,
                    false
                )
                .is_err()
            );
            // Bool inside consumed fails closed.
            let bad_consumed = vec![true.into_pyobject(py).unwrap().to_owned().into_any()];
            assert!(
                template_sort_key_for(py, "chi", None, None, bad_consumed, None, false, false)
                    .is_err()
            );
        });
    }
}
