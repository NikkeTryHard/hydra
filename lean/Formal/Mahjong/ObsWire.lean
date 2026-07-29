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

Mirrors `ideas/observation-spec/planes.md` verdict (§5, L300-308): do NOT
copy Suphx 838/958 blind — they stay OPAQUE undecomposed bounds (no
per-group sum theorem exists until the `.tex` tarball / Gao PDF unblock,
doc §6). What IS pinned: kanachan padded-cap arithmetic (L252-281),
Mortal's versioned wire shapes as opaque table data (L199-229), and the
raw-vs-cap discipline (L260-267). The 34 tile axis itself lives in
`Formal.Mahjong.Tile` (`TileType := Fin 34`); this module only tables.
-/

namespace Hydra2.Mahjong.ObsWire

section KanachanCaps

/-- Kanachan encoder padded caps (`planes.md` L252, L257, L260-262,
L265-267): sparse 33, numeric 6, progression 113, action-candidates 32. -/
def kanachanSparseCap : ℕ := 33
def kanachanNumericCap : ℕ := 6
def kanachanProgCap : ℕ := 113
def kanachanActCap : ℕ := 32

/-- `ENCODER_WIDTH` (`planes.md` L269-270, L281). -/
def kanachanEncoderWidth : ℕ := 184

/-- Cap sum (`planes.md` L281): `184 = 33+6+113+32` padded caps, NOT raw
counts. -/
theorem kanachan_width_eq :
    33 + 6 + 113 + 32 = kanachanEncoderWidth := by decide

/-- Raw-vs-cap discipline (`planes.md` L260-262: progression actually 106,
set to 113; L264-267: actions actually 30, set to 32). Caps dominate raw. -/
theorem kanachan_prog_raw_le_cap : 106 ≤ kanachanProgCap := by decide

theorem kanachan_act_raw_le_cap : 30 ≤ kanachanActCap := by decide

end KanachanCaps

section MortalShapes

/-- Mortal obs rows by version (`planes.md` L216-219):
`(938,34)/(942,34)/(934,34)/(1012,34)`. -/
def mortalObsRows : Fin 4 → ℕ
  | 0 => 938
  | 1 => 942
  | 2 => 934
  | 3 => 1012

/-- Mortal oracle rows (`planes.md` L228-229): `(211,34)` v1, `(217,34)`
v2-4. -/
def mortalOracleRows : Fin 4 → ℕ
  | 0 => 211
  | 1 => 217
  | 2 => 217
  | 3 => 217

/-- Mortal action space (`planes.md` L201-210): `37+1+3+1+1+1+1+1 = 46`,
`GRP_SIZE = 7`. -/
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

/-- Suphx totals (`planes.md` L50-51): `34x838` discard-phase input,
`34x958` call-phase input. OPAQUE — the spec's own verdict (§5) forbids
decomposing these into per-group sums, so no such theorem appears here. -/
def suphxDiscardPlanes : ℕ := 838
def suphxCallPlanes : ℕ := 958

end SuphxOpaque

end Hydra2.Mahjong.ObsWire
