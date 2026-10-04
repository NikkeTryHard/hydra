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

/-- Threshold boundary pins: 99 fails, 100 passes — the bar is exactly
100 with no off-by-one. Practical core: promotion-harness unit tests
assert both sides of the boundary, so a threshold regression fails
closed instead of silently promoting at 99. -/
theorem gate_fail_99 : gatePass 99 = false := by decide

theorem gate_pass_100 : gatePass 100 = true := by decide

end Promotion

section NodeCap

/-- Node cap with day-2 step-up: `gateNodeCap := 300`, `gateNodeCapLate := 400`,
`gateCapStepUp := 2`; skipping the step-up under-searches late gates and fails closed. -/
def gateCap (d : ℕ) : ℕ := if d < 2 then 300 else 400

/-- Day 1 runs the early cap: the step-up fires at day 2, so day 1 still
budgets 300 sims — the finite reason overnight self-play ledgers price
day 1 at the early rate. Pair with `gateCap_early`/`gateCap_late` (the
general bounds): together the schedule is pinned at the boundary day. -/
theorem gateCap_day1 : gateCap 1 = 300 := by decide

theorem gateCap_early {d : ℕ} (h : d < 2) : gateCap d = 300 := by
  simp [gateCap, h]

theorem gateCap_late {d : ℕ} (h : 2 ≤ d) : gateCap d = 400 := by
  simp [gateCap, Nat.not_lt.mpr h]

/-- The step-up strictly raises the cap (`increasing to 400`). -/
theorem gateCap_stepup : gateCap 0 < gateCap 2 := by decide

/-- The late cap dominates the early cap everywhere: no day gets fewer sims
    than day 0, so late-gate searches are never starved relative to the
    opening — the finite core behind budgeting more compute where the
    placement decision is sharpest (limited-compute allocation lemma).
    Day-one matches day-zero (proved next): the first 48 hours run at
    identical budget before the day-2 step-up. -/
theorem gateCap_day1_eq_day0 : gateCap 1 = gateCap 0 := by
  simp [gateCap]

/-- Opening three days cost 1000 sims exactly: 300 + 300 + 400 — the
finite reason the ledger's camp-opening budget is a closed count, not
an estimate: two floor days plus the step-up day sum to one thousand.
Pairs `gateCap_day1_eq_day0` (first 48h flat) with `gateCap_stepup`
(day-2 rise): together the opening is pinned day-by-day and in total. -/
theorem gateCap_opening_three_days : gateCap 0 + gateCap 1 + gateCap 2 = 1000 := by
  decide

/-- Day 2 runs the late cap: the step-up fires exactly here, so day 2
budgets 400 sims — the mirror of `gateCap_day0`/`gateCap_day1` (early
pins), completing the day-by-day pins across the step-up boundary.
Together all three opening days are pinned individually AND in total
(`gateCap_opening_three_days`), so the ledger never interpolates a
day's budget. -/
theorem gateCap_day2 : gateCap 2 = 400 := by
  decide

theorem gateCap_late_dominates {d : ℕ} : gateCap 0 ≤ gateCap d := by
  by_cases h : d < 2
  · have h0 : gateCap d = 300 := gateCap_early h
    have h00 : gateCap 0 = 300 := gateCap_early (by norm_num)
    omega
  · have h1 : gateCap d = 400 := gateCap_late (Nat.le_of_not_lt h)
    have h00 : gateCap 0 = 300 := gateCap_early (by norm_num)
    omega

/-- Universal budget cap: no day exceeds 400 sims, so a full D-day camp
costs at most `400 * D` node expansions — the finite envelope behind
sizing the 5 s search budget and the overnight self-play ledger.
Failure mode it kills: an uncapped day silently eating the compute
budget of every later gate. Day-zero costs the floor (proved next):
the opening days price exactly. -/
theorem gateCap_day0 : gateCap 0 = 300 := by
  unfold gateCap
  simp

theorem gateCap_le_400 {d : ℕ} : gateCap d ≤ 400 := by
  unfold gateCap
  split <;> omega

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

/-- Endpoints agree only at agreement: `mix p e 0 = mix p e 1` iff `p = e`
— the finite reason distinct endpoints give a genuine sweep range:
degenerate sweeps (zero range) happen exactly when raw policy already
equals noise, so a flat ablation ladder diagnoses endpoint collapse,
never a broken dial. Cross-checks both endpoint pins at once. -/
theorem gate_mix_endpoints_eq_iff (p e : ℝ) :
    mixPolicy p e 0 = mixPolicy p e 1 ↔ p = e := by
  unfold mixPolicy
  constructor
  · intro h; linarith
  · intro h; rw [h]; ring

/-- At `ε = 1` the target IS the noise draw: full-noise endpoint mirrors
`gate_mix_is_raw` (the zero-noise endpoint). Together the endpoints pin
the segment: the sweep range is exactly [raw, noise] with no overshoot
in either direction — the finite reason epsilon-greedy-style tuning can
only interpolate. -/
theorem gate_mix_is_noise (p e : ℝ) : mixPolicy p e 1 = e := by
  unfold mixPolicy
  ring

/-- Half-noise is the midpoint: at `ε = 1/2` the mix is exactly the average
of raw policy and noise — the finite reason epsilon sweeps are symmetric:
the halfway point treats both endpoints equally, so ablation ladders read
the noise effect linearly around the center. -/
theorem gate_mix_midpoint (p e : ℝ) : mixPolicy p e (1 / 2) = (p + e) / 2 := by
  unfold mixPolicy
  ring

/-- Midpoint halves the deviation: the mix at `ε = 1/2` is exactly halfway
from raw policy to noise — the finite reason the center of an epsilon
sweep is the fair compromise point: deviations shrink by exactly half,
so two-sided ablations (raw-leaning vs noise-leaning) are equidistant
from the midpoint. Cross-checks `gate_mix_midpoint` via subtraction. -/
theorem gate_mix_midpoint_halves (p e : ℝ) :
    mixPolicy p e (1 / 2) - p = (e - p) / 2 := by
  unfold mixPolicy
  ring

/-- Mix is antisymmetric in its endpoints at the midpoint: swapping raw
policy and noise negates the deviation — the finite reason paired
ablations (raw-leaning vs noise-leaning) are exact mirrors: the
two-sided sweep reads symmetric deviations, so an asymmetric response
is signal, never sweep artifact. Cross-checks `gate_mix_midpoint`
through subtraction. -/
theorem gate_mix_midpoint_swap (p e : ℝ) :
    mixPolicy e p (1 / 2) - e = -(mixPolicy p e (1 / 2) - p) := by
  unfold mixPolicy
  ring

/-- Mixture stays on the segment between raw policy and noise: for
    `ε ∈ [0,1]` the mix is a convex combination, so it never leaves the
    `[min praw eta, max praw eta]` range. Finite core behind blending the
    Dirichlet-noise target with the clean policy: partial noise can only
    interpolate, never extrapolate beyond either endpoint. -/
theorem gate_mix_segment (praw eta eps : ℝ) (h0 : 0 ≤ eps) (h1 : eps ≤ 1) :
    min praw eta ≤ mixPolicy praw eta eps ∧ mixPolicy praw eta eps ≤ max praw eta := by
  unfold mixPolicy
  constructor
  · rcases le_total praw eta with h | h
    · rw [min_eq_left h]
      have hle : eps * praw ≤ eps * eta := mul_le_mul_of_nonneg_left h h0
      have h2 : (1 - eps) * praw + eps * praw = praw := by ring
      linarith
    · rw [min_eq_right h]
      have hle : (1 - eps) * eta ≤ (1 - eps) * praw :=
        mul_le_mul_of_nonneg_left h (sub_nonneg.mpr h1)
      have h2 : (1 - eps) * eta + eps * eta = eta := by ring
      linarith
  · rcases le_total praw eta with h | h
    · rw [max_eq_right h]
      have hle : (1 - eps) * praw ≤ (1 - eps) * eta :=
        mul_le_mul_of_nonneg_left h (sub_nonneg.mpr h1)
      have h2 : (1 - eps) * eta + eps * eta = eta := by ring
      linarith
    · rw [max_eq_left h]
      have hle : eps * eta ≤ eps * praw := mul_le_mul_of_nonneg_left h h0
      have h2 : (1 - eps) * praw + eps * praw = praw := by ring
      linarith

/-- Mixture is monotone in epsilon at fixed endpoints: more noise weight
can only move the mix toward eta (up when `eta ≥ praw`, down when
`praw ≥ eta`). Practical core: the noise dial moves one way per
ordering — sweeping epsilon never oscillates the target, so the
noise-ablation ladder reads cleanly. -/
theorem gate_mix_mono_eps (praw eta eps₁ eps₂ : ℝ) (h : eps₁ ≤ eps₂) :
    (if eta ≤ praw then mixPolicy praw eta eps₂ ≤ mixPolicy praw eta eps₁
     else mixPolicy praw eta eps₁ ≤ mixPolicy praw eta eps₂) := by
  unfold mixPolicy
  by_cases hle : eta ≤ praw
  · simp only [hle, ↓reduceIte]
    have : (eps₂ - eps₁) * (praw - eta) ≥ 0 :=
      mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr hle)
    linarith
  · simp only [hle, ↓reduceIte]
    have hlt : praw < eta := lt_of_not_ge hle
    have : (eps₂ - eps₁) * (eta - praw) ≥ 0 :=
      mul_nonneg (sub_nonneg.mpr h) (by linarith)
    linarith

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

/-- Partial agreement never halts: a single agreeing side (`s = 1`) is
insufficient regardless of bad turns — the finite reason the resign
trigger is consensus, not majority. Pair with `resignHalt_needs_agreement`
(halt implies full agreement): together the trigger is exactly `s = 2`,
so lone-estimator resignation is impossible by construction. -/
theorem resignHalt_single_side_never (t : ℕ) :
    resignHalt 1 t = false := by
  simp [resignHalt]

/-- More bad turns never un-trigger resignation. -/
theorem resignHalt_mono_turns {s t₁ t₂ : ℕ} (hle : t₁ ≤ t₂)
    (h : resignHalt s t₁ = true) : resignHalt s t₂ = true := by
  simp [resignHalt] at h ⊢
  omega

/-- Resignation needs a strict minority of good news: the trigger fires only
    when bad turns reach the 5-turn bar, so any halt implies the bar was met
    — the finite core behind cutting losses instead of playing out dead
    positions (compute saved compounds over thousands of self-play games). -/
theorem resignHalt_needs_bar {s t : ℕ} (h : resignHalt s t = true) : 5 ≤ t := by
  simp [resignHalt] at h
  exact h.2

/-- Resignation also needs full agreement: a halt implies both sides agreed
(`s = 2`). Pair with `resignHalt_needs_bar` (the 5-turn bar): together the
trigger is a conjunction with no silent single-side path — one dissenting
side can never halt the search, so early resignation can only cut
mutually-dead positions. Failure mode killed: a lone noisy estimator
abandoning a live game. -/
theorem resignHalt_needs_agreement {s t : ℕ} (h : resignHalt s t = true) : s = 2 := by
  simp [resignHalt] at h
  exact h.1

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

/-- Gate and self-play share promotion thresholds: the same 100/200 bar judges both, so a promotion means beating the bar, never moving it. Pair with `gate_differs_selfplay` (reuse fields differ): together the configs differ exactly where they must (noise/forcing/temp) and agree exactly where they must (the bar) -- the finite reason gate comparisons are apples-to-apples. -/
theorem gate_shares_thresholds :
    kataGoGate.promoteWins = kataGoSelfPlay.promoteWins ∧
    kataGoGate.promoteGames = kataGoSelfPlay.promoteGames := by
  decide

end ConfigBundle

end Hydra2.Blueprint.Modules.Gating
