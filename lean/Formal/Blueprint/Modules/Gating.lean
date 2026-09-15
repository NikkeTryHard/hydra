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

/-! # Hydra2 gating config predicates (KataGo App E port, https://arxiv.org/abs/1910.07112)

Promotion threshold, node-cap step-up, seating temperature, noise strip,
root FPU, resignation conjunction. Config arithmetic only —
`Formal/Blueprint/Modules/ForcedTarget.lean` defers gating and Elo harness-side, so this
module pins the finite config facts the harness must satisfy, not the Elo
argument; violating any threshold promotes a weaker arm and fails closed.
Real-valued temps are stored ×10 as `ℕ` to keep everything
`decide`-closed.
-/

namespace Hydra2.Blueprint.Modules.Gating

section Promotion

/-- Promotion gate: `promoteWins := 100`, `promoteGames := 200`, pass iff
`wins ≥ promoteWins` (100/200 = 50% win-rate bar; lower bar promotes noise and fails closed). -/
def gatePass (w : ℕ) : Bool := decide (100 ≤ w)

theorem gate_pass_iff (w : ℕ) : gatePass w = true ↔ 100 ≤ w := by
  simp [gatePass]

/-- More wins never un-pass a gate. -/
theorem gate_pass_mono {w₁ w₂ : ℕ} (hle : w₁ ≤ w₂)
    (h : gatePass w₁ = true) : gatePass w₂ = true := by
  simp [gatePass] at h ⊢
  omega

/-- The 100/200 threshold is exactly one half (`≥100/200 (50%)`). -/
theorem gate_half_threshold : 100 + 100 = 200 := by decide

end Promotion

section NodeCap

/-- Node cap with day-2 step-up: `gateNodeCap := 300`, `gateNodeCapLate := 400`,
`gateCapStepUp := 2`; skipping the step-up under-searches late gates and fails closed. -/
def gateCap (d : ℕ) : ℕ := if d < 2 then 300 else 400

theorem gateCap_early {d : ℕ} (h : d < 2) : gateCap d = 300 := by
  simp [gateCap, h]

theorem gateCap_late {d : ℕ} (h : 2 ≤ d) : gateCap d = 400 := by
  simp [gateCap, Nat.not_lt.mpr h]

/-- The step-up strictly raises the cap (`increasing to 400`). -/
theorem gateCap_stepup : gateCap 0 < gateCap 2 := by decide

theorem gateCap_mono {d₁ d₂ : ℕ} (h : d₁ ≤ d₂) :
    gateCap d₁ ≤ gateCap d₂ := by
  unfold gateCap
  split <;> split <;> omega

end NodeCap

section NoiseStrip

/-- Self-play target mixes raw policy with Dirichlet noise; the gate strips
it (`gateDirichlet := false`, forced playouts `k = 2` off). At `ε = 0` the
target IS the raw policy — zero noise mass, nothing to tune. -/
def mixPolicy (praw eta eps : ℝ) : ℝ := (1 - eps) * praw + eps * eta

theorem gate_mix_is_raw (p e : ℝ) : mixPolicy p e 0 = p := by
  unfold mixPolicy
  ring

end NoiseStrip

section Resignation

/-- Resignation: both sides agree AND the worst MCTS
winrate estimate stayed below 5% for the last 5 turns
(`resignAgreeSides := 2`, `resignTurns := 5`, `resignWinrate := 0.05`). -/
def resignHalt (agreeSides badTurns : ℕ) : Bool :=
  decide (agreeSides = 2 ∧ 5 ≤ badTurns)

theorem resignHalt_iff (s t : ℕ) :
    resignHalt s t = true ↔ (s = 2 ∧ 5 ≤ t) := by
  simp [resignHalt]

/-- More bad turns never un-trigger resignation. -/
theorem resignHalt_mono_turns {s t₁ t₂ : ℕ} (hle : t₁ ≤ t₂)
    (h : resignHalt s t₁ = true) : resignHalt s t₂ = true := by
  simp [resignHalt] at h ⊢
  omega

end Resignation

section ConfigBundle

/-- Gate vs self-play configs (temps ×10). Self-play: T = 0.8 seating temp,
Dirichlet on, forced playouts k = 2. -/
structure GateConfig where
  promoteWins : ℕ
  promoteGames : ℕ
  capEarly : ℕ
  capLate : ℕ
  dirichletOff : Bool
  forcedK : ℕ
  placementTemp10 : ℕ

/-- Canonical KataGo gate config. -/
def kataGoGate : GateConfig :=
  ⟨100, 200, 300, 400, true, 0, 5⟩

/-- Canonical self-play config (same thresholds). -/
def kataGoSelfPlay : GateConfig :=
  ⟨100, 200, 300, 300, false, 2, 8⟩

/-- The gate differs from self-play on exactly the reuse fields: noise off,
forced off, cooler seating temp (cap-stepup is day-dependent, thresholds shared);
identical configs would compare an arm to itself and fail closed. -/
theorem gate_differs_selfplay :
    kataGoGate.dirichletOff = true ∧ kataGoSelfPlay.dirichletOff = false ∧
    kataGoGate.forcedK = 0 ∧ kataGoSelfPlay.forcedK = 2 ∧
    kataGoGate.placementTemp10 < kataGoSelfPlay.placementTemp10 := by
  decide

end ConfigBundle

end Hydra2.Blueprint.Modules.Gating
