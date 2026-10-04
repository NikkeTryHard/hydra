//! Tensor-plane geometry for actor observations.
//!
//! This module owns the ordered plane table, the aligned batch geometry, and
//! the splitter that carves one caller-owned byte slab into per-plane views.
//!
//! The table holds 26 feature planes followed by one trailing uniform plane.
//! The 26 feature planes pack tightly to exactly `7254 + 9 * T` bytes per row
//! (`T` is the history bucket: 32, 64, 128, or 256). The uniform plane is a
//! live-input convenience only and is never part of sealed bytes. On the live
//! slab each plane region starts at a multiple of its element width (1, 4, or
//! 8 bytes) so consumers can map typed views directly over the slab; the gaps
//! between regions are bounded deterministic zero padding, and there is never
//! trailing padding. At batch sizes that are multiples of 8 every region is
//! already a multiple of 8, so the aligned total equals the tight total.
//!
//! Plane order below is load-bearing: the row fill, the canonical document
//! encoder, the direct facts encoder, and the learner-side transpose all
//! assume this exact order. Adding, removing, or reordering planes changes
//! the encoded bytes and is never done silently.

use super::ObservationError;
use crate::rl::actions::BASELINE_ACTIONS;

/// Number of planes in the live input table: 26 feature planes plus one
/// trailing uniform plane. The sealed feature bytes cover only the first 26.
pub const PLANE_COUNT: usize = 27;

/// Ordered per-row plane table for one history bucket: each entry is
/// (plane name, dtype string, elements per row). The uniform plane is last.
pub fn plane_layout(bucket_t: usize) -> Vec<(&'static str, &'static str, usize)> {
    vec![
        ("actor", "int64", 1),
        ("actor_can_riichi", "bool", 1),
        ("actor_can_tsumo", "bool", 1),
        ("actor_furiten", "int64", 1),
        ("actor_seats", "int64", 1),
        ("concealed_hand_counts", "int32", 34),
        ("dealer", "int64", 1),
        ("dora_indicators", "int32", 5),
        ("hand_number", "int32", 1),
        ("history_event_kind", "int64", bucket_t),
        ("history_mask", "bool", bucket_t),
        ("honba", "int32", 1),
        ("ippatsu_active", "bool", 4),
        ("kan_count", "int32", 1),
        ("legal_mask", "bool", BASELINE_ACTIONS),
        ("live_wall_tiles_remaining", "int32", 1),
        ("own_drawn_tile", "int32", 1),
        ("phase", "int64", 1),
        ("riichi_states", "int64", 4),
        ("riichi_sticks", "int32", 1),
        ("round_index", "int32", 1),
        ("round_wind", "int64", 1),
        ("scores", "int32", 4),
        ("seat_winds", "int64", 4),
        ("turn_actor", "int64", 1),
        ("visible_discards_counts", "int32", 34),
        ("uniform", "float64", 1),
    ]
}

/// Byte width of one element for a plane dtype string. Unknown dtypes fall
/// back to 1 byte; the table above only uses int64, float64, int32, and bool.
pub(super) fn dtype_width(dtype: &str) -> usize {
    match dtype {
        "int64" | "float64" => 8,
        "int32" => 4,
        _ => 1,
    }
}

/// Round `offset` up to a multiple of `width`. Widths are always 1, 4, or 8,
/// hence powers of two. Every region base produced by the layout satisfies
/// `offset % width == 0`, which typed byte-view consumers require.
const fn align_up(offset: usize, width: usize) -> usize {
    let mask = width - 1;
    (offset + mask) & !mask
}

/// Owned per-bucket plane spec `(name, dtype, elements per row)`, cached once
/// per bucket and shared by every layout call. Offsets still scale with the
/// batch size, so they are computed per call; only the spec table is shared.
static PLANE_SPEC_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<usize, Vec<(String, String, usize)>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Clone the cached spec for `bucket_t`, building and storing it on first use.
pub(super) fn plane_spec(bucket_t: usize) -> Vec<(String, String, usize)> {
    if let Ok(guard) = PLANE_SPEC_CACHE.lock() {
        if let Some(spec) = guard.get(&bucket_t) {
            return spec.clone();
        }
    }
    let spec: Vec<(String, String, usize)> = plane_layout(bucket_t)
        .into_iter()
        .map(|(name, dtype, per_row)| (name.to_string(), dtype.to_string(), per_row))
        .collect();
    if let Ok(mut guard) = PLANE_SPEC_CACHE.lock() {
        guard.insert(bucket_t, spec.clone());
    }
    spec
}

/// Byte offsets for `n_valid` active rows: aligned plane-major packing where
/// each region base is a multiple of its dtype width and gaps are
/// deterministic zero padding. The total is the last region end, with no tail
/// padding and no dummy rows. Returns (name, dtype, shape, byte_offset).
/// When `include_uniform` is false the trailing uniform plane is skipped, so
/// the table covers exactly the 26 sealed feature planes.
pub fn active_layout(
    bucket_t: usize,
    n_valid: usize,
    include_uniform: bool,
) -> Vec<(String, String, Vec<usize>, usize)> {
    let mut out = Vec::with_capacity(PLANE_COUNT);
    let mut offset = 0usize;
    // The tight per-row feature law is unchanged (sealed bytes stay tight;
    // only the live slab carries bounded alignment gaps). At batch sizes that
    // are multiples of 8 every region is already a multiple of 8, so the
    // aligned total equals the tight total and capacities are unchanged.
    for (name, dtype, per_row) in plane_spec(bucket_t) {
        if name == "uniform" && !include_uniform {
            continue;
        }
        let width = dtype_width(&dtype);
        offset = align_up(offset, width);
        let region = n_valid * per_row * width;
        let shape = if per_row == 1 {
            vec![n_valid]
        } else {
            vec![n_valid, per_row]
        };
        out.push((name, dtype, shape, offset));
        offset += region;
    }
    out
}

/// Total live input bytes for `n_valid` rows at bucket `T` (26 feature planes
/// plus uniform): the aligned last-region end, including bounded inter-region
/// padding but never tail padding. Every slab size, copy length, and capacity
/// computation must use this value, never `n_valid * (row_bytes + 8)`.
pub fn active_input_bytes(bucket_t: usize, n_valid: usize) -> usize {
    let layout = active_layout(bucket_t, n_valid, true);
    match layout.last() {
        None => 0,
        Some((_, dtype, shape, off)) => {
            off + shape.iter().product::<usize>() * dtype_width(dtype)
        }
    }
}

/// Total feature bytes for `n_valid` rows at bucket `T` (26 planes, uniform
/// excluded): the aligned last-region end of the uniform-free table. Rows stay
/// contiguous inside regions with bounded zero-filled inter-region padding
/// and never tail padding.
pub fn active_feature_bytes(bucket_t: usize, n_valid: usize) -> usize {
    let layout = active_layout(bucket_t, n_valid, false);
    match layout.last() {
        None => 0,
        Some((_, dtype, shape, off)) => {
            off + shape.iter().product::<usize>() * dtype_width(dtype)
        }
    }
}

/// Mutable plane views for one fill, borrowed from the caller slab. Field
/// order follows the plane table exactly.
pub struct PlaneBuffers<'a> {
    pub actor: &'a mut [u8],
    pub can_riichi: &'a mut [u8],
    pub can_tsumo: &'a mut [u8],
    pub furiten: &'a mut [u8],
    pub seats: &'a mut [u8],
    pub concealed: &'a mut [u8],
    pub dealer: &'a mut [u8],
    pub dora: &'a mut [u8],
    pub hand_number: &'a mut [u8],
    pub history_kind: &'a mut [u8],
    pub history_mask: &'a mut [u8],
    pub honba: &'a mut [u8],
    pub ippatsu: &'a mut [u8],
    pub kan_count: &'a mut [u8],
    pub legal: &'a mut [u8],
    pub live_wall: &'a mut [u8],
    pub own_drawn: &'a mut [u8],
    pub phase: &'a mut [u8],
    pub riichi_states: &'a mut [u8],
    pub riichi_sticks: &'a mut [u8],
    pub round_index: &'a mut [u8],
    pub round_wind: &'a mut [u8],
    pub scores: &'a mut [u8],
    pub seat_winds: &'a mut [u8],
    pub turn_actor: &'a mut [u8],
    pub visible: &'a mut [u8],
    pub uniform: &'a mut [u8],
}

/// Split one aligned slab into mutable plane views in table order. Alignment
/// gaps between regions are zeroed deterministically first, so stale slab
/// bytes can never leak into padding that downstream readers may observe.
/// Rejects slabs smaller than the layout total and tables whose regions
/// overlap or escape the slab.
pub fn split_planes<'a>(
    slab: &'a mut [u8],
    bucket_t: usize,
    n_valid: usize,
    include_uniform: bool,
) -> Result<PlaneBuffers<'a>, ObservationError> {
    let layout = active_layout(bucket_t, n_valid, include_uniform);
    let total = layout
        .last()
        .map_or(0, |(_, dtype, shape, off)| off + shape.iter().product::<usize>() * dtype_width(dtype));
    if slab.len() < total {
        return Err(ObservationError(format!(
            "input slab {} < active {total} for B={n_valid} T={bucket_t}",
            slab.len()
        )));
    }
    // Zero alignment gaps first (plain disjoint writes, no views alive yet),
    // rejecting any overlapping table (fail loud, never silent aliasing).
    let mut prev_end = 0usize;
    for (_, dtype, shape, off) in &layout {
        let region: usize = shape.iter().product::<usize>() * dtype_width(dtype);
        if *off < prev_end || off.checked_add(region).is_none_or(|end| end > slab.len()) {
            return Err(ObservationError(format!(
                "plane table overlaps or escapes the slab at offset {off}"
            )));
        }
        slab[prev_end..*off].fill(0);
        prev_end = off + region;
    }
    // Carve disjoint region views (in-order, non-overlapping by the check
    // above; raw slices keep the borrow checker out of the gap logic).
    let mut views: Vec<&mut [u8]> = Vec::with_capacity(layout.len());
    let base = slab.as_mut_ptr();
    for (_, dtype, shape, off) in &layout {
        let region: usize = shape.iter().product::<usize>() * dtype_width(dtype);
        // SAFETY: `off..off + region` lies inside the slab and regions are
        // pairwise disjoint (verified above); no other borrow of the slab
        // is alive across this loop.
        let view = unsafe { std::slice::from_raw_parts_mut(base.add(*off), region) };
        views.push(view);
    }
    let mut it = views.into_iter();
    // Inline macro (not a closure): the closure boundary forces the returned
    // slice lifetime onto the `&mut it` borrow, which cannot outlive the
    // iterator borrow; expanding inline keeps the slice tied to the views.
    macro_rules! take {
        () => {
            it.next()
                .ok_or_else(|| ObservationError("plane split underflow".to_string()))
        };
    }
    // Order follows `plane_layout` exactly.
    let actor = take!()?;
    let can_riichi = take!()?;
    let can_tsumo = take!()?;
    let furiten = take!()?;
    let seats = take!()?;
    let concealed = take!()?;
    let dealer = take!()?;
    let dora = take!()?;
    let hand_number = take!()?;
    let history_kind = take!()?;
    let history_mask = take!()?;
    let honba = take!()?;
    let ippatsu = take!()?;
    let kan_count = take!()?;
    let legal = take!()?;
    let live_wall = take!()?;
    let own_drawn = take!()?;
    let phase = take!()?;
    let riichi_states = take!()?;
    let riichi_sticks = take!()?;
    let round_index = take!()?;
    let round_wind = take!()?;
    let scores = take!()?;
    let seat_winds = take!()?;
    let turn_actor = take!()?;
    let visible = take!()?;
    let uniform = if include_uniform { take!()? } else { &mut [] as &mut [u8] };
    Ok(PlaneBuffers {
        actor,
        can_riichi,
        can_tsumo,
        furiten,
        seats,
        concealed,
        dealer,
        dora,
        hand_number,
        history_kind,
        history_mask,
        honba,
        ippatsu,
        kan_count,
        legal,
        live_wall,
        own_drawn,
        phase,
        riichi_states,
        riichi_sticks,
        round_index,
        round_wind,
        scores,
        seat_winds,
        turn_actor,
        visible,
        uniform,
    })
}
