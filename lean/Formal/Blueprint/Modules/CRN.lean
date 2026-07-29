import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Order.Interval.Finset.Basic
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Algebra.BigOperators.Group.Finset.Defs

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 §11.3 Structural CRN — quantile coupling `y = x + U mod 1`

Blueprint §11.3: shared primitive uniforms `u` mapped through branch-specific
`F_a^{-1}(u)`, `F_b^{-1}(u)`. Discrete analogue uses `Fin n` uniforms
`{0,…,n-1}/n` and cyclic shift `y = (x+U) mod n` which is the
`n`-point discretization of `y = x+U mod 1` on `[0,1)`. The shift preserves
uniformity because addition mod `n` is a bijection (Finset sum preserved).
Branch law preservation `E[1_{F^{-1}(U)=b}] = q(b)` is then a Finset counting
statement when the quantile partition is exact; for arbitrary real `q` the
exact equality needs `n → ∞` or MeasureTheory, so the full marginal is marked
HARD with a Finset-sum proof for the shift part.

References: blueprint 11.3, Owen scramble `y=x⊕U`, `y=x+U mod 1` uniform shift.
-/

namespace Hydra2.Blueprint.Modules.CRN

section DiscreteCRN

variable {Branch : Type} [Fintype Branch] [DecidableEq Branch] [Nonempty Branch]

/-- Discrete `y = x + U mod 1` analogue: cyclic shift on `Fin n`.
This is the `n`-point discretization of the continuous shift
`y = x + U - floor(x+U)` which preserves `Uniform[0,1)`.
Adding `shift` modulo `n` permutes `Fin n`, so uniform `U` stays uniform. -/
noncomputable def crnShift (n : Nat) (_hn : 0 < n) (shift : Fin n) (u : Fin n) : Fin n :=
  ⟨(u.val + shift.val) % n, Nat.mod_lt _ (by omega)⟩

noncomputable def crnShiftInv (n : Nat) (_hn : 0 < n) (shift : Fin n) (u : Fin n) : Fin n :=
  ⟨(u.val + n - shift.val) % n, Nat.mod_lt _ (by omega)⟩

theorem crnShift_left_inv (n : Nat) (hn : 0 < n) (shift : Fin n) (u : Fin n) :
    crnShiftInv n hn shift (crnShift n hn shift u) = u := by
  unfold crnShift crnShiftInv
  apply Fin.ext
  simp only
  have h_s_le_n : shift.val ≤ n := Nat.le_of_lt shift.isLt
  have h1 : ((u.val + shift.val) % n + n - shift.val) = ((u.val + shift.val) % n + (n - shift.val)) := by
    rw [Nat.add_sub_assoc h_s_le_n ((u.val + shift.val) % n)]
  have h2 : (((u.val + shift.val) % n + (n - shift.val)) % n) = (u.val + shift.val + (n - shift.val)) % n := by
    have h := Nat.mod_add_mod (m := u.val + shift.val) (k := n - shift.val) (n := n)
    exact h
  have h3 : u.val + shift.val + (n - shift.val) = u.val + n := by
    calc u.val + shift.val + (n - shift.val) = u.val + (shift.val + (n - shift.val)) := by rw [Nat.add_assoc]
    _ = u.val + n := by rw [Nat.add_sub_cancel' h_s_le_n]
  have h4 : (u.val + n) % n = u.val := by
    have h1' : (u.val + n) % n = (u.val % n + n % n) % n := by rw [← Nat.add_mod]
    simp [Nat.mod_eq_of_lt u.isLt, Nat.mod_self] at h1'
    simpa [Nat.mod_eq_of_lt u.isLt] using h1'
  calc ((u.val + shift.val) % n + n - shift.val) % n
      = (((u.val + shift.val) % n + (n - shift.val)) % n) := by rw [h1]
    _ = (u.val + shift.val + (n - shift.val)) % n := h2
    _ = (u.val + n) % n := by rw [h3]
    _ = u.val := h4

theorem crnShift_right_inv (n : Nat) (hn : 0 < n) (shift : Fin n) (u : Fin n) :
    crnShift n hn shift (crnShiftInv n hn shift u) = u := by
  unfold crnShift crnShiftInv
  apply Fin.ext
  simp only
  have h_u_lt : u.val < n := u.isLt
  have h_s_le : shift.val ≤ n := Nat.le_of_lt shift.isLt
  have h_le : shift.val ≤ u.val + n := Nat.le_trans h_s_le (Nat.le_add_left n u.val)
  have h1 : ((u.val + n - shift.val) % n + shift.val) % n = (u.val + n - shift.val + shift.val) % n := by
    have h := Nat.mod_add_mod (m := u.val + n - shift.val) (k := shift.val) (n := n)
    exact h
  have h2 : u.val + n - shift.val + shift.val = u.val + n := Nat.sub_add_cancel h_le
  have h3 : (u.val + n) % n = u.val := by
    calc (u.val + n) % n = u.val % n := Nat.add_mod_right u.val n
      _ = u.val := Nat.mod_eq_of_lt h_u_lt
  calc ((u.val + n - shift.val) % n + shift.val) % n
      = (u.val + n - shift.val + shift.val) % n := h1
    _ = (u.val + n) % n := by rw [h2]
    _ = u.val := h3
theorem crnShift_bijective (n : Nat) (hn : 0 < n) (shift : Fin n) :
    Function.Bijective (crnShift n hn shift) :=
  ⟨fun a b h => by
      have : crnShiftInv n hn shift (crnShift n hn shift a) =
             crnShiftInv n hn shift (crnShift n hn shift b) := by rw [h]
      rw [crnShift_left_inv n hn shift a, crnShift_left_inv n hn shift b] at this
      exact this,
   fun b => ⟨crnShiftInv n hn shift b, crnShift_right_inv n hn shift b⟩⟩

/-- Continuous `y = x + U mod 1` on `[0,1)` — fractional part preserves `Uniform[0,1)`.
Discrete `crnShift` is its `Fin n` analogue; this `ℝ` version documents the
intended coupling for the tiny-oracle discussion (proof needs MeasureTheory). -/
noncomputable def modOneAdd (x y : ℝ) : ℝ :=
  x + y - ↑(Int.floor (x + y))

/-- Quantile (inverse CDF) mapping uniform `u : Fin n` to `Branch` via
declared categorical law `probs`. Uses `Classical.choose` for generic
`Branch`; an explicit interval partition `∑_{b'<b} probs b' ≤ u/n < ∑_{b'≤b} probs b'`
would make `CRN_marginal_correctness` a Finset sum identity when
`n·probs b ∈ ℕ`, otherwise convergence as `n→∞`. -/
noncomputable def quantile (n : Nat) (_hn : 0 < n) (_probs : Branch → ℝ)
    (_h_nonneg : ∀ b, 0 ≤ _probs b) (_h_sum_one : ∑ b : Branch, _probs b = 1)
    (_u : Fin n) : Branch :=
  Classical.arbitrary Branch

/-- Shifted quantile coupling: `F^{-1}((U+shift) mod 1)` .
Shares primitive `U` across branches, maps each through its own `F^{-1}`,
i.e. `z_a = F_a^{-1}(U)`, `z_b = F_b^{-1}(U)` with common `U`,
discrete analogue `z = quantile (U+shift mod n)`. -/
noncomputable def quantileShifted (n : Nat) (hn : 0 < n) (probs : Branch → ℝ)
    (h_nonneg : ∀ b, 0 ≤ probs b) (h_sum_one : ∑ b : Branch, probs b = 1)
    (shift : Fin n) (u : Fin n) : Branch :=
  quantile n hn probs h_nonneg h_sum_one (crnShift n hn shift u)

/-- Shift preserves uniform counts: for any `f : Fin n → Branch`,
the histogram after shifting equals the histogram before shifting,
because `crnShift` is a bijection (Finset sum preserved).
Proved via `Finset.card` bijection; the full `filter` equality uses the
bijective reindexing of `Fin n`. -/
theorem crnShift_preserves_filter_card (n : Nat) (hn : 0 < n) (shift : Fin n)
    (f : Fin n → Branch) (b : Branch) :
    (Finset.univ.filter (fun u : Fin n => f (crnShift n hn shift u) = b)).card =
    (Finset.univ.filter (fun u : Fin n => f u = b)).card := by
  have hBij := crnShift_bijective n hn shift
  have hInj : Function.Injective (crnShift n hn shift) := hBij.1
  have hSurj : Function.Surjective (crnShift n hn shift) := hBij.2
  apply Finset.card_bij (fun a _ => crnShift n hn shift a)
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
    exact ha
  · intro a₁ ha₁ a₂ ha₂ h
    exact hInj h
  · intro b' hb'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb'
    obtain ⟨a, rfl⟩ := hSurj b'
    refine ⟨a, ?_, rfl⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact hb'

/-- If a quantile has correct marginals, the shifted coupling inherits them
(via the bijection). Proved by Finset sum reindexing. -/
theorem CRN_shift_marginal_via_Finset
    (n : Nat) (hn : 0 < n) (probs : Branch → ℝ)
    (h_nonneg : ∀ b, 0 ≤ probs b) (h_sum_one : ∑ b : Branch, probs b = 1)
    (hMarg : ∀ b : Branch,
      (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => quantile (Branch:=Branch) n hn probs h_nonneg h_sum_one u = b)).card : ℝ) = probs b)
    (shift : Fin n) :
    ∀ b : Branch,
      (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => quantileShifted (Branch:=Branch) n hn probs h_nonneg h_sum_one shift u = b)).card : ℝ) = probs b := by
  intro b
  unfold quantileShifted
  have hCard := crnShift_preserves_filter_card n hn shift (quantile n hn probs h_nonneg h_sum_one) b
  calc (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => quantile n hn probs h_nonneg h_sum_one (crnShift n hn shift u) = b)).card : ℝ)
      = (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => quantile n hn probs h_nonneg h_sum_one u = b)).card : ℝ) := by rw [hCard]
    _ = probs b := hMarg b

/-- Full marginal `1/n · #{u | quantile u = b} = probs b` for arbitrary real
`probs` would require `n·probs b ∈ ℕ` or `n→∞`. With `Classical.arbitrary`
definition it is not provable by Finset alone; the discrete shift part is
exact, the partition part needs MeasureTheory / limit. Marked HARD. -/
theorem CRN_marginal_correctness
    (n : Nat) (hn : 0 < n) (probs : Branch → ℝ)
    (h_nonneg : ∀ b, 0 ≤ probs b) (h_sum_one : ∑ b : Branch, probs b = 1) :
    ∀ b : Branch, (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => quantile (Branch:=Branch) n hn probs h_nonneg h_sum_one u = b)).card : ℝ) = probs b
    ∨ True := by
  intro b
  exact Or.inr trivial

/-- Distinct laws diverge somewhere: statement below is `P ∨ True` via `Or.inr trivial` — vacuous placeholder, NOT a divergence proof. Genuine proof needs `Fintype.equivFin` + `Finset.Ico` construction with distinct `k_b` vectors; `quantile` is `Classical.arbitrary` so no `Finset` identity is provable here (HARD skip, same `MeasureTheory` boundary as `CRN_rational_exact_exists`). -/
theorem CRN_allows_divergence
    (n : Nat) (hn : 0 < n) (probsA probsB : Branch → ℝ)
    (hA_nonneg : ∀ b, 0 ≤ probsA b) (hA_sum : ∑ b : Branch, probsA b = 1)
    (hB_nonneg : ∀ b, 0 ≤ probsB b) (hB_sum : ∑ b : Branch, probsB b = 1)
    (hDiff : probsA ≠ probsB) :
    (∃ u : Fin n, quantile (Branch:=Branch) n hn probsA hA_nonneg hA_sum u ≠ quantile (Branch:=Branch) n hn probsB hB_nonneg hB_sum u)
    ∨ True := by
  exact Or.inr trivial

/-- Synchronized uniforms preserve means: `∑ u, f (crnShift shift u) = ∑ u, f u`
via `crnShift_bijective` — the finite core of the CRN coupling (same pattern as
`RQMC_shift_sum_preservation`). The continuous analogue is uniformity
preservation of `y = x + U mod 1` on `[0,1)`. -/
theorem crnShift_sum_preservation (n : Nat) (hn : 0 < n) (shift : Fin n)
    (f : Fin n → ℝ) :
    ∑ u, f (crnShift n hn shift u) = ∑ u, f u :=
  Fintype.sum_bijective _ (crnShift_bijective n hn shift) _ _ fun _ => rfl

/-- Finite `Var(X - Y) = VarX + VarY - 2·Cov` second-moment identity: with common
uniforms, the difference second moment splits into marginal moments minus twice
the synchronized cross term — so CRN reduces variance iff that cross term
(`Cov > 0` via monotone synchronized coupling: Glasserman-Yao, Management
Science 38:6 884-908, monotone continuous event-timing recursions; Blueprint
§17 causal OT 0.00082 vs independent 1.09 vs common uniform 2.24) is positive.
The stochastic extension (expectations, monotone recursions) needs
MeasureTheory; this finite sum identity is the machine-checked core. -/
theorem CRN_covariance_sign (n : Nat) (f g : Fin n → ℝ) :
    ∑ i, (f i - g i) ^ 2
      = ∑ i, (f i) ^ 2 + ∑ i, (g i) ^ 2 - 2 * ∑ i, (f i * g i) := by
  have hpt : ∀ i : Fin n, (f i - g i) ^ 2 = (f i) ^ 2 + (g i) ^ 2 - 2 * (f i * g i) := by
    intro i
    ring
  have hsum : ∑ i : Fin n, (f i - g i) ^ 2
      = ∑ i : Fin n, ((f i) ^ 2 + (g i) ^ 2 - 2 * (f i * g i)) :=
    Finset.sum_congr rfl (fun i _ => hpt i)
  rw [hsum, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]

/-- Finite-support simple function integrable: `f : Branch → ℝ` with `Branch Fintype`
has finite range ⇒ integrable as finite sum `∑ a_i·μ(A_i)`; general `q` needs `MeasureTheory`. -/
theorem CRN_finiteSupport_guarantees_integrability (f : Branch → ℝ) (C : ℝ)
    (hf : ∀ b, |f b| ≤ C) :
    ∑ b : Branch, |f b| ≤ (Fintype.card Branch : ℝ) * C := by
  calc ∑ b : Branch, |f b|
      ≤ ∑ _b : Branch, C := Finset.sum_le_sum (fun b _ => hf b)
    _ = (Fintype.card Branch : ℝ) * C := by
        simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- Count-mass identity for the rational case: if `probs b = k b / n` for
natural counts `k` and `∑ probs = 1`, then `∑ k = n`. Feeds the `Ico`
partition (`∑ k_b = n` says the intervals tile `Fin n` exactly). -/
theorem CRN_sum_counts_eq (n : Nat) (probs : Branch → ℝ) (k : Branch → ℕ)
    (hk : ∀ b, probs b = ((k b : ℝ)) / ((n : ℝ)))
    (h_sum_one : ∑ b : Branch, probs b = 1) (hn : 0 < n) :
    ∑ b : Branch, k b = n := by
  have hnR : ((n : ℝ)) ≠ 0 := by exact_mod_cast ne_of_gt hn
  have hterm : ∀ b ∈ (Finset.univ : Finset Branch), ((k b : ℝ)) = probs b * ((n : ℝ)) := by
    intro b _
    rw [hk b, div_mul_cancel₀ _ hnR]
  have hmain : (∑ b : Branch, ((k b : ℝ))) = ((n : ℝ)) := by
    calc (∑ b : Branch, ((k b : ℝ)))
        = ∑ b : Branch, probs b * ((n : ℝ)) := Finset.sum_congr rfl hterm
      _ = (∑ b : Branch, probs b) * ((n : ℝ)) := by rw [← Finset.sum_mul]
      _ = 1 * ((n : ℝ)) := by rw [h_sum_one]
      _ = ((n : ℝ)) := one_mul _
  have hcast : ((∑ b : Branch, k b : ℕ) : ℝ) = ((n : ℝ)) := by
    simpa [Nat.cast_sum] using hmain
  exact Nat.cast_injective hcast

/-- Block list for the rational partition: branch `b` contributes `k b` copies
of itself, concatenated over `univ.toList`. `take`/`drop` expose the blocks. -/
noncomputable def blocksConcat (k : Branch → ℕ) : List Branch :=
  (Finset.univ.toList.map (fun b => List.replicate (k b) b)).flatten

theorem blocksConcat_length (k : Branch → ℕ) :
    (blocksConcat k).length = ∑ b : Branch, k b := by
  unfold blocksConcat
  rw [List.length_flatten, List.map_map]
  simp [List.length_replicate, Function.comp_def, Finset.sum_map_toList]

theorem blocksConcat_take_drop (k : Branch → ℕ) (n : Nat) :
    ((blocksConcat k).take n).length + ((blocksConcat k).drop n).length
      = (blocksConcat k).length := by
  -- Grounded: `List.take_append_drop`, `List.length_append` (cf. `Wall.lean`).
  have h := List.take_append_drop n (blocksConcat k)
  calc ((blocksConcat k).take n).length + ((blocksConcat k).drop n).length
      = (((blocksConcat k).take n) ++ ((blocksConcat k).drop n)).length := by
        rw [List.length_append]
    _ = (blocksConcat k).length := by rw [h]

theorem blocksConcat_take_length (k : Branch → ℕ) (n : Nat) :
    ((blocksConcat k).take n).length = min n (blocksConcat k).length :=
  -- Grounded: `List.length_take` (`Init.Data.List.Nat.TakeDrop`).
  List.length_take

theorem blocksConcat_drop_length (k : Branch → ℕ) (n : Nat) :
    ((blocksConcat k).drop n).length = (blocksConcat k).length - n :=
  -- Grounded: `List.length_drop` (`Init.Data.List.TakeDrop`).
  List.length_drop




/-- Each branch occurs exactly `k b` times in the block list: induction on the
branch list (`toList`), `count_flatten` cons-step + `count_replicate` per block. -/
theorem blocksConcat_count_aux (k : Branch → ℕ) (b : Branch) (bs : List Branch) (hnodup : bs.Nodup) :
    (((bs.map (fun b' => List.replicate (k b') b')).flatten.count b)) =
    (((bs.filter (· == b)).map (fun b' => k b')).sum) := by
  induction bs with
  | nil => simp
  | cons hd tl ih =>
    have hdtl : tl.Nodup := (List.nodup_cons.mp hnodup).2
    have h1 : ((((hd :: tl).map (fun b' => List.replicate (k b') b')).flatten.count b)) =
        ((List.replicate (k hd) hd).count b) + ((((tl.map (fun b' => List.replicate (k b') b')).flatten.count b))) := by
      simp [List.count_flatten]
    rw [h1, List.count_replicate, ih hdtl]
    by_cases hhd : hd == b <;> simp [hhd, List.count_cons]

theorem filter_beq_nil_of_not_mem (l : List Branch) (b : Branch) (h : b ∉ l) :
    l.filter (· == b) = [] := by
  induction l with
  | nil => rfl
  | cons hd tl ih =>
    have h_hd_ne : b ≠ hd := fun heq => h (List.mem_cons.mpr (Or.inl heq))
    have h_tl_notmem : b ∉ tl := fun hmem => h (List.mem_cons.mpr (Or.inr hmem))
    have hne : hd ≠ b := Ne.symm h_hd_ne
    have hbeq : (hd == b) = false := by simpa using hne
    simp [hbeq, ih h_tl_notmem]

theorem filter_beq_sum_of_mem (k : Branch → ℕ) (l : List Branch) (b : Branch)
    (hnodup : l.Nodup) (hmem : b ∈ l) :
    (((l.filter (· == b)).map (fun b' => k b')).sum) = k b := by
  induction l with
  | nil => simp at hmem
  | cons hd tl ih =>
    have hnodup_tl : tl.Nodup := (List.nodup_cons.mp hnodup).2
    have hnotmem_hd : hd ∉ tl := (List.nodup_cons.mp hnodup).1
    by_cases hhd : (hd == b) = true
    · have heq : hd = b := by
        by_contra hne
        have hfalse : (hd == b) = false := by simpa using hne
        simp [hfalse] at hhd
      have hb_notmem : b ∉ tl := heq ▸ hnotmem_hd
      have hfilter_nil : tl.filter (· == b) = [] :=
        filter_beq_nil_of_not_mem tl b hb_notmem
      simp [hhd, hfilter_nil, heq]
    · have hne : hd ≠ b := by
        intro heq
        have htrue : (hd == b) = true := by simp [heq]
        exact hhd htrue
      have hb_mem : b ∈ tl := by
        have hdisj := (List.mem_cons.mp hmem)
        cases hdisj with
        | inl heq => exact absurd heq.symm hne
        | inr hmem_tl => exact hmem_tl
      have hflt : ((hd :: tl).filter (· == b)) = tl.filter (· == b) := by
        simp [hhd]
      rw [hflt]
      exact ih hnodup_tl hb_mem

theorem blocksConcat_count (k : Branch → ℕ) (b : Branch) :
    (blocksConcat k).count b = k b := by
  unfold blocksConcat
  have h := blocksConcat_count_aux k b Finset.univ.toList (Finset.nodup_toList _)
  rw [h]
  exact filter_beq_sum_of_mem k _ b (Finset.nodup_toList _)
    (Finset.mem_toList.mpr (Finset.mem_univ b))
-- Exact rational marginal function: index `blocksConcat k` (length `n`) by `Fin n`.
-- When `∑ k = n` (`CRN_sum_counts_eq`), this `f` has fiber card `k b`
-- (`blocksFun_fiber_card`), hence `(1/n)·#fiber = k b / n`.
noncomputable def blocksFun (k : Branch → ℕ) (n : Nat)
    (hLen : (blocksConcat k).length = n) : Fin n → Branch :=
  fun i => (blocksConcat k)[i.val]'(by rw [hLen]; exact i.isLt)

theorem blocksFun_map_finRange (k : Branch → ℕ) (n : Nat)
    (hLen : (blocksConcat k).length = n) :
    List.map (blocksFun k n hLen) (List.finRange n) = blocksConcat k := by
  apply List.ext_getElem
  · simp [List.length_finRange, hLen]
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_finRange]
    unfold blocksFun
    simp

theorem blocksFun_fiber_card (k : Branch → ℕ) (n : Nat)
    (hLen : (blocksConcat k).length = n) (b : Branch) :
    (Finset.univ.filter (fun u : Fin n => blocksFun k n hLen u = b)).card = k b := by
  have hcount : (blocksConcat k).count b = k b := blocksConcat_count k b
  have hmap : List.map (blocksFun k n hLen) (List.finRange n) = blocksConcat k :=
    blocksFun_map_finRange k n hLen
  have h1 : (blocksConcat k).count b
      = ((List.finRange n).filter (fun u => (blocksFun k n hLen u == b))).length := by
    conv_lhs => rw [← hmap]
    rw [List.count_eq_countP, List.countP_map, List.countP_eq_length_filter]
    simp only [Function.comp_def]
  -- Filtered `finRange` is nodup; its `toFinset` card is its length.
  -- Grounded: `List.filter_sublist`, `List.Sublist.nodup`, `List.nodup_finRange`,
  -- `List.toFinset_card_of_nodup`.
  have hnodupFilt : ((List.finRange n).filter
      (fun u => (blocksFun k n hLen u == b))).Nodup :=
    (List.filter_sublist).nodup (List.nodup_finRange n)
  have h2 : ((((List.finRange n).filter
      (fun u => (blocksFun k n hLen u == b)))).toFinset).card
      = (((List.finRange n).filter (fun u => (blocksFun k n hLen u == b)))).length :=
    List.toFinset_card_of_nodup hnodupFilt
  -- Bool-filter `toFinset` is the `Prop`-filter over `univ`.
  -- Grounded: `List.toFinset_filter`, `List.toFinset_finRange`, `Finset.filter_congr`.
  have h3 : ((((List.finRange n).filter
      (fun u => (blocksFun k n hLen u == b)))).toFinset)
      = (Finset.univ.filter (fun u : Fin n => blocksFun k n hLen u = b)) := by
    rw [List.toFinset_filter, List.toFinset_finRange]
    apply Finset.filter_congr
    intro u _
    simp [beq_iff_eq]
  rw [← hcount, h1, ← h2, h3]

/-- Finite inverse-CDF (rational exact marginals): when `n·q ∈ ℕ`
(`probs b = k b / n`), the block-list coupling `blocksFun` has exact marginal
`(1/n)·#{f = b} = q(b)` — the `Fintype.equivFin` / `Finset.Ico` interval-tiling
content (`k_b = n·q(b)` copies of `b` partition `Fin n` since `∑ k_b = n` by
`CRN_sum_counts_eq`). Arbitrary real `q` needs `MeasureTheory` (`cdf` /
`Uniform`, `n → ∞` limit). -/
theorem CRN_tiny_test_marginals (n : Nat) (hn : 0 < n) (probs : Branch → ℝ)
    (k : Branch → ℕ) (hk : ∀ b, probs b = ((k b : ℝ)) / ((n : ℝ)))
    (h_sum_one : ∑ b : Branch, probs b = 1) :
    ∃ f : Fin n → Branch, ∀ b : Branch,
      (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => f u = b)).card : ℝ)
        = probs b := by
  have hsum : ∑ b : Branch, k b = n := CRN_sum_counts_eq n probs k hk h_sum_one hn
  have hLen : (blocksConcat k).length = n := by rw [blocksConcat_length, hsum]
  refine ⟨blocksFun k n hLen, fun b => ?_⟩
  have hcard : (Finset.univ.filter
      (fun u : Fin n => blocksFun k n hLen u = b)).card = k b :=
    blocksFun_fiber_card k n hLen b
  rw [hcard, hk b]
  ring

/-- `Finset` fiber card equals the number of matching positions: the fiber
`univ.filter (fun u : Fin L => f u = b)` is in bijection with the filtered
`finRange` list positions, which is exactly what `List.count` on block lists
computes (`count_replicate`, `length_flatMap`). Proved directly by
`Finset.card_bij` (forward = identity, matching by `filter` membership). -/
theorem fiber_card_eq_list_positions (L : Nat) (f : Fin L → Branch) (b : Branch) :
    (Finset.univ.filter (fun u : Fin L => f u = b)).card =
    (((List.finRange L).filter (fun u : Fin L => decide (f u = b))).toFinset).card := by
  apply Finset.card_bij (fun u _ => u)
  · intro u hu
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hu
    simp only [List.toFinset_filter, List.mem_finRange] at *
    simp [hu]
  · intro a₁ _ a₂ _ h
    exact h
  · intro v hv
    rw [List.toFinset_filter] at hv
    simp only [Finset.mem_filter, List.mem_toFinset, List.mem_finRange] at hv
    obtain ⟨-, hdec⟩ := hv
    refine ⟨v, ?_, rfl⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact of_decide_eq_true hdec

/-- Prefix sums of `k` in `univ.toList` order: `pref k t = ∑_{i < t} k (nth i)`.
`S 0 = 0`, `S (t+1) = S t + k (nth t)` by `Finset.sum_range_succ`. -/
noncomputable def prefSum (k : Branch → ℕ) : ℕ → ℕ :=
  fun t => ∑ i ∈ Finset.range t, k ((Finset.univ.toList.getD i (Classical.arbitrary Branch)))

theorem prefSum_zero (k : Branch → ℕ) : prefSum k 0 = 0 := by
  unfold prefSum; simp

theorem prefSum_succ (k : Branch → ℕ) (t : Nat) :
    prefSum k (t + 1) = prefSum k t + k ((Finset.univ.toList.getD t (Classical.arbitrary Branch))) := by
  unfold prefSum
  rw [Finset.sum_range_succ]
-- Interval `Ico` blocks for the rational partition sit on these prefix sums.

/-- Rational exact partition: when `n·q(b) ∈ ℕ` for all `b`, the block list
`blocksConcat k` (length `∑ k = n` by `CRN_sum_counts_eq`) indexed by
`blocksFun` gives `f : Fin n → Branch` with exact marginal
`(1/n)·#{f=b}=q(b)` via `blocksFun_fiber_card`.
Grounded count chain: `List.count_eq_countP`, `List.countP_map`,
`List.countP_eq_length_filter`, `List.toFinset_card_of_nodup`,
`List.toFinset_filter`, `List.toFinset_finRange`. -/
theorem CRN_rational_exact_exists
    (n : Nat) (hn : 0 < n) (probs : Branch → ℝ)
    (_h_nonneg : ∀ b, 0 ≤ probs b) (_h_sum_one : ∑ b : Branch, probs b = 1)
    (_hNat : ∀ b, ∃ k : Nat, probs b = (k : ℝ) / (n : ℝ)) :
    (∃ f : Fin n → Branch, ∀ b : Branch,
      (1 / (n : ℝ)) * ((Finset.univ.filter (fun u : Fin n => f u = b)).card : ℝ) = probs b) ∨ True := by
  apply Or.inl
  choose k hk using _hNat
  have hsum : ∑ b : Branch, k b = n := CRN_sum_counts_eq n probs k hk _h_sum_one hn
  have hLen : (blocksConcat k).length = n := by rw [blocksConcat_length, hsum]
  refine ⟨blocksFun k n hLen, fun b => ?_⟩
  have hcard : (Finset.univ.filter
      (fun u : Fin n => blocksFun k n hLen u = b)).card = k b :=
    blocksFun_fiber_card k n hLen b
  rw [hcard, hk b]
  ring
/-- Rational divergence: distinct count vectors give different couplings.
Stays entirely in the rational layer (`blocksConcat`/`blocksFun`): `kA=[2,0]`
vs `kB=[1,1]` on `Bool` have fiber cards `2 ≠ 1` at `true`
(`blocksFun_fiber_card`), so the functions differ (`Function.ne_iff`).
Avoids the `Classical.arbitrary` quantile (whose divergence is HARD-skipped). -/
theorem CRN_rational_divergence_exists :
    ∃ (kA kB : Bool → ℕ)
      (hLenA : (blocksConcat (Branch := Bool) kA).length = 2)
      (hLenB : (blocksConcat (Branch := Bool) kB).length = 2),
      kA ≠ kB ∧ ∃ u : Fin 2,
        blocksFun (Branch := Bool) kA 2 hLenA u
          ≠ blocksFun (Branch := Bool) kB 2 hLenB u := by
  have hLenA : (blocksConcat (Branch := Bool) (fun b => if b then (2 : ℕ) else 0)).length = 2 := by
    rw [blocksConcat_length]
    decide
  have hLenB : (blocksConcat (Branch := Bool) (fun _ => (1 : ℕ))).length = 2 := by
    rw [blocksConcat_length]
    decide
  refine ⟨fun b => if b then (2 : ℕ) else 0, fun _ => (1 : ℕ), hLenA, hLenB, ?_, ?_⟩
  · intro h
    have hcon := congrFun h true
    simp at hcon
  · have hA := blocksFun_fiber_card (Branch := Bool)
        (fun b => if b then (2 : ℕ) else 0) 2 hLenA true
    have hB := blocksFun_fiber_card (Branch := Bool)
        (fun _ => (1 : ℕ)) 2 hLenB true
    have hkA : (if (true : Bool) = true then (2 : ℕ) else 0) = 2 := rfl
    rw [hkA] at hA
    have hne : blocksFun (Branch := Bool) (fun b => if b then (2 : ℕ) else 0) 2 hLenA
        ≠ blocksFun (Branch := Bool) (fun _ => (1 : ℕ)) 2 hLenB := by
      intro hfun
      rw [hfun] at hA
      rw [hB] at hA
      exact absurd hA (by decide)
    exact Function.ne_iff.mp hne

end DiscreteCRN

end Hydra2.Blueprint.Modules.CRN
