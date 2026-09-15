import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! # Hydra2 observation wire tables (observation-spec)

Do NOT copy Suphx 838/958 blind — they stay OPAQUE undecomposed bounds (no
per-group sum theorem exists until the source tarball unblocks).
What IS pinned: kanachan padded-cap arithmetic,
Mortal versioned wire shapes as opaque table data, and the
raw-vs-cap discipline (caps dominate raw; using raw under-allocates and fails closed).
The 34 tile axis itself lives in
`Formal.Mahjong.Tile` (`TileType := Fin 34`); this module only tables.
-/

namespace Hydra2.Mahjong.ObsWire

section KanachanCaps

/-- Kanachan encoder padded caps: sparse 33, numeric 6, progression 113,
action-candidates 32; wrong caps break the encoder width and fail closed. -/
def kanachanSparseCap : ℕ := 33
def kanachanNumericCap : ℕ := 6
def kanachanProgCap : ℕ := 113
def kanachanActCap : ℕ := 32

/-- `ENCODER_WIDTH` 184; wrong width breaks the encoder shape and fails closed. -/
def kanachanEncoderWidth : ℕ := 184

/-- Cap sum: `184 = 33+6+113+32` padded caps, NOT raw counts. -/
theorem kanachan_width_eq :
    33 + 6 + 113 + 32 = kanachanEncoderWidth := by decide

/-- Raw-vs-cap discipline (progression actually 106, set to 113; actions actually 30,
set to 32). Caps dominate raw; using raw under-allocates and fails closed. -/
theorem kanachan_prog_raw_le_cap : 106 ≤ kanachanProgCap := by decide

theorem kanachan_act_raw_le_cap : 30 ≤ kanachanActCap := by decide

end KanachanCaps

section MortalShapes

/-- Mortal obs rows by version: `(938,34)/(942,34)/(934,34)/(1012,34)`;
wrong rows break the wire shape and fail closed. -/
def mortalObsRows : Fin 4 → ℕ
  | 0 => 938
  | 1 => 942
  | 2 => 934
  | 3 => 1012

/-- Mortal oracle rows: `(211,34)` v1, `(217,34)` v2-4. -/
def mortalOracleRows : Fin 4 → ℕ
  | 0 => 211
  | 1 => 217
  | 2 => 217
  | 3 => 217

/-- Mortal action space: `37+1+3+1+1+1+1+1 = 46`, `GRP_SIZE = 7`. -/
def mortalActionSpace : ℕ := 46
def mortalGrpSize : ℕ := 7

theorem mortal_action_space_eq : 37 + 1 + 3 + 1 + 1 + 1 + 1 + 1 = mortalActionSpace := by
  decide

/-- Obs-row version table, closed lookup. -/
theorem mortal_obs_rows_table :
    mortalObsRows 0 = 938 ∧ mortalObsRows 1 = 942 ∧
    mortalObsRows 2 = 934 ∧ mortalObsRows 3 = 1012 := by
  decide

/-- Oracle-row version table, closed lookup. -/
theorem mortal_oracle_rows_table :
    mortalOracleRows 0 = 211 ∧ mortalOracleRows 1 = 217 ∧
    mortalOracleRows 2 = 217 ∧ mortalOracleRows 3 = 217 := by
  decide

end MortalShapes

section SuphxOpaque

/-- Suphx totals: `34x838` discard-phase input, `34x958` call-phase input.
OPAQUE — decomposing these into per-group sums is forbidden,
so no such theorem appears here. -/
def suphxDiscardPlanes : ℕ := 838
def suphxCallPlanes : ℕ := 958

end SuphxOpaque

end Hydra2.Mahjong.ObsWire
