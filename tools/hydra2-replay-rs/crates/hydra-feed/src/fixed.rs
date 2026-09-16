//! Fixed-point utility math: validated ranks + exact zero-sum (no float).
//!
//! Mirrors `src/hydra2/contracts/utility.py` (`_validated_ranks`, `_exact_total`,
//! `utility`) without copying it: Python sums `Fraction(item)` per float and
//! raises `ContractError` on ties/gaps; here all accumulation is integer-only.
//!
//! - Ranks are `u8` and validated as a strict permutation of `1..=4` (M15):
//!   `0` underflow and `5+` OOB become `Err`, never a panic — Python raises
//!   `ContractError` (`utility.py:170-177`), so Rust MUST return `Result`.
//! - Exact zero-sum over `f64` uses `i128` checked shifts over a common
//!   power-of-two denominator (m8): `1e12` mixed with a subnormal needs a
//!   1100+ bit shift, which MUST surface as `Err(Overflow)` (fallback, never
//!   wrap, never assume zero). No `float` accumulation, no epsilon.
//! - [`utility_fixed`] maps validated ranks onto canonical zero-sum fixed
//!   placement points; [`utility_for_ranks_fixed`] maps them through a caller
//!   manifest's fixed `rank_values` (mirrors `utility()` indexing).

/// Fixed scale: micro-placement points (1 point = 1_000_000 units).
pub const FIXED_SCALE: i64 = 1_000_000;

// ---------------------------------------------------------------------------
// Errors (fail-closed, numeric, never a `String`)
// ---------------------------------------------------------------------------

/// Fixed-point failure taxonomy: mirrors `ContractError` edges, never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedError {
    /// Rank byte outside `1..=4` (`0` underflows, `5+` OOBs in Python too).
    InvalidRank {
        /// Position in the ranks quad.
        index: usize,
        /// Rejected byte.
        value: u8,
    },
    /// Ranks are individually in range but not a permutation of `1..=4`
    /// (tie or gap — Python `_validated_ranks` rejects `sorted != [1,2,3,4]`).
    DuplicateRank {
        /// First duplicated (or gap-implying) value found.
        value: u8,
    },
    /// `f64` input is NaN or infinite (Python `_require_finite_float` rejects).
    NonFiniteValue {
        /// Position in the values quad.
        index: usize,
    },
    /// Exact-sum alignment overflows `i128` (m8 fallback: e.g. `1e12` mixed
    /// with a subnormal needs a 1100+ bit shift — never wrap, never assume).
    Overflow,
}

impl core::fmt::Display for FixedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            FixedError::InvalidRank { index, value } => {
                write!(f, "ranks[{index}]={value} outside 1..=4")
            }
            FixedError::DuplicateRank { value } => {
                write!(f, "ranks must be a strict permutation of 1..=4 (tie/gap at {value})")
            }
            FixedError::NonFiniteValue { index } => {
                write!(f, "rank_values[{index}] must be finite")
            }
            FixedError::Overflow => {
                write!(f, "exact total overflows i128 (exponent spread too wide)")
            }
        }
    }
}

impl std::error::Error for FixedError {}

// ---------------------------------------------------------------------------
// Ranks validation (M15)
// ---------------------------------------------------------------------------

/// Validate a strict permutation of `1..=4` (no ties, no gaps).
///
/// Mirrors `_validated_ranks` (`utility.py:170-177`): each byte in `1..=4`
/// and `sorted == [1,2,3,4]`, else `Err` (Python raises `ContractError`).
pub fn validate_ranks(ranks: &[u8; 4]) -> Result<(), FixedError> {
    let mut i = 0;
    while i < 4 {
        let v = ranks[i];
        if v < 1 || v > 4 {
            return Err(FixedError::InvalidRank { index: i, value: v });
        }
        i += 1;
    }
    let mut counts = [0u8; 5];
    let mut j = 0;
    while j < 4 {
        let v = ranks[j] as usize;
        counts[v] += 1;
        if counts[v] > 1 {
            return Err(FixedError::DuplicateRank { value: ranks[j] });
        }
        j += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fixed mappings (NO float)
// ---------------------------------------------------------------------------

/// Canonical zero-sum fixed placement points for validated ranks.
///
/// Rank `r` maps to `(5 - 2*r) * FIXED_SCALE`: rank 1 → `+3·S`, 2 → `+1·S`,
/// 3 → `-1·S`, 4 → `-3·S`. Every permutation sums to exactly zero (integer
/// sum, no float). Default/test manifest; caller manifests go through
/// [`utility_for_ranks_fixed`].
pub fn utility_fixed(ranks: [u8; 4]) -> Result<[i64; 4], FixedError> {
    validate_ranks(&ranks)?;
    let mut out = [0i64; 4];
    let mut i = 0;
    while i < 4 {
        let r = ranks[i] as i64;
        out[i] = (5 - 2 * r) * FIXED_SCALE;
        i += 1;
    }
    Ok(out)
}

/// Map ranks through a caller manifest's fixed `rank_values`.
///
/// Mirrors `utility()` indexing (`manifest.rank_values[rank-1]`): validated
/// ranks index `rank_values`; out-of-range indexing is `Err` by construction
/// (defensive `.get`, unreachable after validation — never a panic).
pub fn utility_for_ranks_fixed(
    rank_values: &[i64; 4],
    ranks: &[u8; 4],
) -> Result<[i64; 4], FixedError> {
    validate_ranks(ranks)?;
    let mut out = [0i64; 4];
    let mut i = 0;
    while i < 4 {
        let idx = (ranks[i] - 1) as usize;
        match rank_values.get(idx) {
            Some(&v) => out[i] = v,
            None => {
                return Err(FixedError::InvalidRank {
                    index: i,
                    value: ranks[i],
                });
            }
        }
        i += 1;
    }
    Ok(out)
}

/// Exact integer sum of fixed values (`i128` accumulator, infallible:
/// `4 * i64::MAX` fits `i128`).
pub fn exact_total_fixed(values: &[i64; 4]) -> i128 {
    let mut acc: i128 = 0;
    let mut i = 0;
    while i < 4 {
        acc += values[i] as i128;
        i += 1;
    }
    acc
}

// ---------------------------------------------------------------------------
// Exact f64 zero-sum via checked shifts (m8, NO float accumulation)
// ---------------------------------------------------------------------------

/// Decompose a finite `f64` into exact `(mantissa: i128, exp2: i32)` with
/// `value = mantissa * 2^exp2` (sign folded into the mantissa).
fn decompose_f64(value: f64) -> (i128, i32) {
    const EXP_MASK: u64 = 0x7FF0_0000_0000_0000;
    const MANT_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
    const EXP_BIAS: i32 = 1023;
    const MANT_BITS: i32 = 52;
    let bits = value.to_bits();
    let negative = (bits >> 63) != 0;
    let raw_exp = ((bits & EXP_MASK) >> MANT_BITS) as i32;
    let raw_mant = bits & MANT_MASK;
    let (mag, exp): (u64, i32) = if raw_exp == 0 {
        // Subnormal or ±0: value = mantissa * 2^-1074.
        (raw_mant, -1074)
    } else {
        // Normal: implicit leading 1, value = (2^52 | mantissa) * 2^(exp-1075).
        (raw_mant | (1u64 << MANT_BITS), raw_exp - EXP_BIAS - MANT_BITS)
    };
    let signed = if negative { -(mag as i128) } else { mag as i128 };
    (signed, exp)
}

/// Exact zero-sum over four `f64` values (integer-only, checked).
///
/// Equivalent to Python `Fraction` sum `== 0` for the 4-element check:
/// aligns all mantissae to the minimum exponent with `checked_shl` and
/// accumulates with `checked_add`. Exponent spreads wider than `i128`
/// (m8: `1e12` + subnormal ⇒ 1100+ bit shift) yield `Err(Overflow)` —
/// the checked-shift fallback, never wrapping, never assuming.
/// Non-finite inputs are `Err` (Python `_require_finite_float` rejects).
/// No float summation, no epsilon.
pub fn exact_total_is_zero(vals: &[f64; 4]) -> Result<bool, FixedError> {
    let mut i = 0;
    while i < 4 {
        if !vals[i].is_finite() {
            return Err(FixedError::NonFiniteValue { index: i });
        }
        i += 1;
    }
    let (m0, e0) = decompose_f64(vals[0]);
    let (m1, e1) = decompose_f64(vals[1]);
    let (m2, e2) = decompose_f64(vals[2]);
    let (m3, e3) = decompose_f64(vals[3]);
    let mut emin = e0;
    if e1 < emin {
        emin = e1;
    }
    if e2 < emin {
        emin = e2;
    }
    if e3 < emin {
        emin = e3;
    }
    let ms = [m0, m1, m2, m3];
    let es = [e0, e1, e2, e3];
    let mut acc: i128 = 0;
    let mut k = 0;
    while k < 4 {
        if ms[k] == 0 {
            k += 1;
            continue;
        }
        let shift = (es[k] - emin) as u32;
        match ms[k].checked_shl(shift) {
            Some(term) => match acc.checked_add(term) {
                Some(next) => acc = next,
                None => return Err(FixedError::Overflow),
            },
            None => return Err(FixedError::Overflow),
        }
        k += 1;
    }
    Ok(acc == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_valid_permutation_passes() {
        assert!(validate_ranks(&[1, 2, 3, 4]).is_ok());
        assert!(validate_ranks(&[4, 1, 3, 2]).is_ok());
    }

    #[test]
    fn ranks_reject_zero_and_five_m15() {
        // M15: `0` underflows `(r-1)` indexing, `5+` OOBs — both `Err`.
        assert!(matches!(
            validate_ranks(&[0, 2, 3, 4]),
            Err(FixedError::InvalidRank { index: 0, value: 0 })
        ));
        assert!(matches!(
            validate_ranks(&[1, 2, 3, 5]),
            Err(FixedError::InvalidRank { index: 3, value: 5 })
        ));
        assert!(utility_fixed([0, 2, 3, 4]).is_err());
        assert!(utility_for_ranks_fixed(&[10, 20, 30, 40], &[1, 2, 3, 5]).is_err());
    }

    #[test]
    fn ranks_reject_ties_and_gaps() {
        // Python `_validated_ranks`: sorted must equal [1,2,3,4].
        assert!(matches!(
            validate_ranks(&[1, 1, 3, 4]),
            Err(FixedError::DuplicateRank { .. })
        ));
        assert!(matches!(
            validate_ranks(&[1, 2, 2, 4]),
            Err(FixedError::DuplicateRank { .. })
        ));
    }

    #[test]
    fn utility_fixed_canonical_points_and_zero_sum() {
        let out = utility_fixed([1, 2, 3, 4]).unwrap();
        assert_eq!(out, [3 * FIXED_SCALE, FIXED_SCALE, -FIXED_SCALE, -3 * FIXED_SCALE]);
        assert_eq!(exact_total_fixed(&out), 0);
        // Permuted ranks permute the points; sum stays exactly zero.
        let perm = utility_fixed([4, 1, 3, 2]).unwrap();
        assert_eq!(
            perm,
            [-3 * FIXED_SCALE, 3 * FIXED_SCALE, -FIXED_SCALE, FIXED_SCALE]
        );
        assert_eq!(exact_total_fixed(&perm), 0);
    }

    #[test]
    fn utility_for_ranks_fixed_mirrors_python_indexing() {
        let rank_values = [10i64, 20, 30, 40];
        assert_eq!(
            utility_for_ranks_fixed(&rank_values, &[4, 3, 2, 1]).unwrap(),
            [40, 30, 20, 10]
        );
        assert_eq!(
            utility_for_ranks_fixed(&rank_values, &[2, 1, 4, 3]).unwrap(),
            [20, 10, 40, 30]
        );
    }

    #[test]
    fn exact_total_f64_zero_and_nonzero() {
        assert_eq!(exact_total_is_zero(&[1.0, -1.0, 2.0, -2.0]).unwrap(), true);
        assert_eq!(exact_total_is_zero(&[1.5, 2.5, -1.0, -2.0]).unwrap(), false);
        assert_eq!(exact_total_is_zero(&[0.0, 0.0, 0.0, 0.0]).unwrap(), true);
        // Exactness, not float rounding: 0.1+0.2-0.1-0.2 cancels exactly.
        assert_eq!(exact_total_is_zero(&[0.1, 0.2, -0.1, -0.2]).unwrap(), true);
    }

    #[test]
    fn exact_total_f64_rejects_nonfinite() {
        assert!(matches!(
            exact_total_is_zero(&[f64::NAN, 0.0, 0.0, 0.0]),
            Err(FixedError::NonFiniteValue { index: 0 })
        ));
        assert!(matches!(
            exact_total_is_zero(&[1.0, f64::INFINITY, -1.0, 0.0]),
            Err(FixedError::NonFiniteValue { index: 1 })
        ));
    }

    #[test]
    fn exact_total_f64_overflow_is_err_m8() {
        // m8: 1e12-scale normal mixed with a subnormal needs a 1100+ bit
        // alignment shift — checked fallback, never wrap, never assume zero.
        assert_eq!(
            exact_total_is_zero(&[1e12, -1e12, 5e-324, -5e-324]),
            Err(FixedError::Overflow)
        );
    }
}
