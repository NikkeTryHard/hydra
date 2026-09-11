//! Cold collate: compact rows → full-26 planes (P4-B, cold only — NEVER hot).
//!
//! Reference pattern (read-only): hydra1 `bc-shards` host `Scratch`
//! (`new` / `reset` / `swap_batch`) + reader `collate_into`. The scratch owns
//! alloc-once plane buffers (caps kept across files); `collate_into` decodes
//! every compact row via [`crate::full26::decode_row_into_planes`] (legal
//! bits expand to `u8`, counts widen `u8 → int32`); `swap_batch` hands the
//! filled batch out while recycling the spare buffers.
//!
//! G2 parity ([`verify_hot_subset`]): the collated hot subset is byte-identical
//! to the hot fill output at the same `T`. The schema carries no float fields,
//! so the `allclose 1e-6` leg is vacuous; integers compare EXACT post-cast
//! (hot `u8` counts cast to `int32` first, then compared).

use hydra_feed::fill::N_PLANES;
use hydra_feed::ledger::{
    PLANE_ACTOR, PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DEALER, PLANE_DORA, PLANE_HIST_KIND,
    PLANE_HIST_MASK, PLANE_LEGAL, PLANE_PHASE, PLANE_ROUND_WIND, PLANE_SCORES, PLANE_SEAT_WINDS,
    PLANE_VISIBLE,
};

use crate::full26::{
    Full26Error, compact_row_bytes, decode_row_into_planes, full26_plane,
    full26_plane_row_bytes,
};
use crate::reader::ShardReader;

/// One collated batch: 26 full-26 planes (schema-alphabetical, LE bytes,
/// row-major, history planes `T`-padded) + the `i64` label + join keys.
#[derive(Debug, Default)]
pub struct CollatedBatch {
    /// Full-26 planes in [`crate::full26::FULL26_FIELDS`] order.
    pub planes: [Vec<u8>; 26],
    /// Training labels (`chosen_action_id`, `i64` LE, one per row).
    pub label_chosen: Vec<u8>,
    /// Join keys in row order (string ids derive cold via `crate::ids`).
    pub keys_game: Vec<u32>,
    /// Per-row sequence within the game.
    pub keys_seq: Vec<u32>,
    /// Committed row count.
    pub rows: u32,
    /// History bucket `T` of this batch.
    pub t_len: usize,
}

/// G2 mismatch: hot plane index + row + what diverged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G2Mismatch {
    /// Hot §7 plane index (`usize::MAX` = row-count/shape mismatch).
    pub hot_plane: usize,
    pub row: u32,
    pub what: &'static str,
}

impl core::fmt::Display for G2Mismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "G2 hot-plane {} row {}: {}",
            self.hot_plane, self.row, self.what
        )
    }
}

impl std::error::Error for G2Mismatch {}

/// Alloc-once collate scratch (caps kept across files and batches).
#[derive(Debug, Default)]
pub struct CollateScratch {
    batch: CollatedBatch,
}

impl CollateScratch {
    /// Empty scratch (no allocation; buffers grow on first collate).
    pub fn new() -> Self {
        Self::default()
    }

    /// Truncate every plane/label/key buffer (`len = 0`, capacities kept).
    pub fn reset(&mut self) {
        let mut i = 0usize;
        while i < 26 {
            self.batch.planes[i].clear();
            i += 1;
        }
        self.batch.label_chosen.clear();
        self.batch.keys_game.clear();
        self.batch.keys_seq.clear();
        self.batch.rows = 0;
        self.batch.t_len = 0;
    }

    /// Borrow the last collated batch.
    pub fn batch(&self) -> &CollatedBatch {
        &self.batch
    }

    /// Exchange the filled batch with `out` (buffer caps recycle both ways:
    /// `out`'s old buffers become the next collate's spare).
    pub fn swap_batch(&mut self, out: &mut CollatedBatch) {
        let mut i = 0usize;
        while i < 26 {
            core::mem::swap(&mut self.batch.planes[i], &mut out.planes[i]);
            i += 1;
        }
        core::mem::swap(&mut self.batch.label_chosen, &mut out.label_chosen);
        core::mem::swap(&mut self.batch.keys_game, &mut out.keys_game);
        core::mem::swap(&mut self.batch.keys_seq, &mut out.keys_seq);
        core::mem::swap(&mut self.batch.rows, &mut out.rows);
        core::mem::swap(&mut self.batch.t_len, &mut out.t_len);
    }

    /// Decode every compact row of `reader` into the scratch planes
    /// (legal expands to `u8`, counts widen `u8 → int32`, kinds `u8 → i64`).
    pub fn collate_into(&mut self, reader: &ShardReader) -> Result<(), Full26Error> {
        self.reset();
        let t = reader.t_bucket();
        let rows = reader.rows();
        let stride = compact_row_bytes(t);
        let mut i = 0usize;
        while i < 26 {
            self.batch.planes[i]
                .reserve(rows as usize * full26_plane_row_bytes(i, t));
            i += 1;
        }
        self.batch.label_chosen.reserve(rows as usize * 8);
        self.batch.keys_game.reserve(rows as usize);
        self.batch.keys_seq.reserve(rows as usize);
        let payload = reader.payload();
        let mut r = 0u64;
        while r < rows {
            let base = r as usize * stride;
            let row = payload.get(base..base + stride).ok_or(Full26Error::RowLength {
                expected: stride,
                got: payload.len().saturating_sub(base),
            })?;
            let game_idx = u32::from_le_bytes([row[2], row[3], row[4], row[5]]);
            let seq = u32::from_le_bytes([row[6], row[7], row[8], row[9]]);
            decode_row_into_planes(row, t, &mut self.batch.planes, &mut self.batch.label_chosen)?;
            self.batch.keys_game.push(game_idx);
            self.batch.keys_seq.push(seq);
            r += 1;
        }
        self.batch.rows = rows as u32;
        self.batch.t_len = t;
        Ok(())
    }
}

/// Plane index of a full-26 field (panics never: unknown names map to `None`
/// and surface as a `G2Mismatch`).
fn f26(name: &str) -> Result<usize, G2Mismatch> {
    full26_plane(name).ok_or(G2Mismatch {
        hot_plane: usize::MAX,
        row: u32::MAX,
        what: "unknown full-26 field",
    })
}

/// G2 parity: the collated hot subset is byte-identical to the hot fill
/// output (`hot_filled`: 13 caller-pinned planes at `t_len`, exactly what the
/// bridge hands torch). Counts compare EXACT post-cast (`u8` → `int32`
/// first); legal compares as set-membership (every hot id set in the full
/// mask, `legal_len <= popcount`, since production sidecars carry offers past
/// the hot `K = 32` truncation).
pub fn verify_hot_subset(
    coll: &CollatedBatch,
    hot_filled: &[Vec<u8>; N_PLANES],
    t_len: usize,
) -> Result<u32, G2Mismatch> {
    let rows = coll.rows as usize;
    let stride = |plane: usize| -> usize {
        match plane {
            PLANE_HIST_KIND => t_len * 8,
            PLANE_HIST_MASK => t_len,
            PLANE_CONCEALED | PLANE_VISIBLE => 34,
            PLANE_DORA => 20,
            PLANE_SCORES => 16,
            PLANE_LEGAL => 136,
            PLANE_SEAT_WINDS => 32,
            _ => 8,
        }
    };
    let hot_row = |plane: usize, r: usize| -> Result<&[u8], G2Mismatch> {
        let s = stride(plane);
        hot_filled[plane].get(r * s..(r + 1) * s).ok_or(G2Mismatch {
            hot_plane: plane,
            row: r as u32,
            what: "hot fill footprint short",
        })
    };
    let coll_row = |field: &str, r: usize| -> Result<&[u8], G2Mismatch> {
        let p = f26(field)?;
        let s = full26_plane_row_bytes(p, t_len);
        coll.planes[p].get(r * s..(r + 1) * s).ok_or(G2Mismatch {
            hot_plane: usize::MAX,
            row: r as u32,
            what: "collated footprint short",
        })
    };
    if coll.t_len != t_len {
        return Err(G2Mismatch {
            hot_plane: usize::MAX,
            row: u32::MAX,
            what: "bucket mismatch",
        });
    }
    let mut r = 0usize;
    while r < rows {
        // Counts: EXACT post-cast (hot u8 -> int32 LE vs collated int32 LE).
        for (hot_plane, field) in [
            (PLANE_CONCEALED, "concealed_hand_counts"),
            (PLANE_VISIBLE, "visible_discards_counts"),
        ] {
            let h = hot_row(hot_plane, r)?;
            let c = coll_row(field, r)?;
            let mut k = 0usize;
            while k < 34 {
                let expect = i32::from(h[k]).to_le_bytes();
                if c[k * 4..k * 4 + 4] != expect {
                    return Err(G2Mismatch {
                        hot_plane,
                        row: r as u32,
                        what: "count widen mismatch (post-cast)",
                    });
                }
                k += 1;
            }
        }
        // Verbatim planes: dora / scores / kinds / mask / scalars / seats.
        for (hot_plane, field) in [
            (PLANE_DORA, "dora_indicators"),
            (PLANE_SCORES, "scores"),
            (PLANE_HIST_KIND, "history_event_kind"),
            (PLANE_HIST_MASK, "history_mask"),
            (PLANE_ACTOR, "actor"),
            (PLANE_DEALER, "dealer"),
            (PLANE_ROUND_WIND, "round_wind"),
            (PLANE_PHASE, "phase"),
            (PLANE_SEAT_WINDS, "seat_winds"),
        ] {
            let h = hot_row(hot_plane, r)?;
            let c = coll_row(field, r)?;
            if h.len() != c.len() || h != c {
                return Err(G2Mismatch {
                    hot_plane,
                    row: r as u32,
                    what: "byte mismatch",
                });
            }
        }
        // Label: hot chosen vs collated label.
        {
            let h = hot_row(PLANE_CHOSEN, r)?;
            let got = coll.label_chosen.get(r * 8..(r + 1) * 8).ok_or(G2Mismatch {
                hot_plane: PLANE_CHOSEN,
                row: r as u32,
                what: "label footprint short",
            })?;
            if h != got {
                return Err(G2Mismatch {
                    hot_plane: PLANE_CHOSEN,
                    row: r as u32,
                    what: "label mismatch",
                });
            }
        }
        // Legal: every hot id set in the full mask; len <= popcount.
        {
            let h = hot_row(PLANE_LEGAL, r)?;
            let mask = coll_row("legal_mask", r)?;
            let len = i64::from_le_bytes(h[128..136].try_into().map_err(|_| G2Mismatch {
                hot_plane: PLANE_LEGAL,
                row: r as u32,
                what: "legal_len unreadable",
            })?);
            if len < 0 || len > 32 {
                return Err(G2Mismatch {
                    hot_plane: PLANE_LEGAL,
                    row: r as u32,
                    what: "legal_len out of K=32",
                });
            }
            let mut k = 0i64;
            while k < len {
                let id = i32::from_le_bytes(h[k as usize * 4..k as usize * 4 + 4].try_into().map_err(
                    |_| G2Mismatch {
                        hot_plane: PLANE_LEGAL,
                        row: r as u32,
                        what: "legal id unreadable",
                    },
                )?) as usize;
                if id >= mask.len() || mask[id] != 1 {
                    return Err(G2Mismatch {
                        hot_plane: PLANE_LEGAL,
                        row: r as u32,
                        what: "hot legal id missing from cold mask",
                    });
                }
                k += 1;
            }
            let mut pop = 0i64;
            for b in mask {
                pop += i64::from(*b);
            }
            if pop < len {
                return Err(G2Mismatch {
                    hot_plane: PLANE_LEGAL,
                    row: r as u32,
                    what: "cold mask popcount below hot len",
                });
            }
        }
        r += 1;
    }
    Ok(rows as u32)
}
