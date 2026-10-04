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

/-! # Hydra2 offline-reanalyze-only reuse

Offline-reanalyze-only reuse (ReZero backward-view port): backward-view and subtree
reuse are allowed in offline replay and reanalyze ONLY and are FORBIDDEN on the live
hanchan path — live allows at most within-kyoku draw-subtree carry, never backward-view.
Violation leaks future search into live play and fails closed by rejecting the line.
Shard identity: a length-`k+1` backward view partitions batch `B` into shards of width
`B/(k+1)`. Shard math and the reuse predicate are finite (`ℕ`/`Bool`/`Finset`);
return-optimality over cadence needs an env model and stays open.
-/

namespace Hydra2.Blueprint.SearchHygiene

section ReuseRule

/-- Search phase: live hanchan path vs offline replay/reanalyze. -/
inductive Phase where
  | Live
  | Offline
  deriving DecidableEq

/-- Reuse kind: none, within-kyoku draw-subtree carry (maximal live reuse boundary),
or full backward-view subtree reuse (ReZero, offline only). -/
inductive ReuseKind where
  | None
  | WithinKyoku
  | Backward
  deriving DecidableEq

/-- Offline-reanalyze-only reuse rule: offline allows everything; live allows at most
within-kyoku carry, never backward-view. Live backward-view leaks future search
and fails closed. -/
def ReuseOK : Phase → ReuseKind → Bool
  | .Offline, _ => true
  | .Live, .None => true
  | .Live, .WithinKyoku => true
  | .Live, .Backward => false

/-- Offline reanalyze admits every reuse kind. -/
theorem reuseOK_offline_all (r : ReuseKind) :
    ReuseOK .Offline r = true := by
  cases r <;> rfl

/-- Live hanchan FORBIDS backward-view reuse. -/
theorem live_forbids_backward :
    ReuseOK .Live .Backward = false := rfl

/-- Live-allowed reuse is always offline-allowed: the live rule is strictly
    narrower, so anything surviving live gating passes offline replay.
    Practical core: prove a reuse safe for live once, reuse it offline free. -/
theorem live_allowed_implies_offline (r : ReuseKind)
    (h : ReuseOK .Live r = true) : ReuseOK .Offline r = true := by
  cases r <;> simp [ReuseOK]

/-- Live permits the within-kyoku draw-subtree carry (the maximal live reuse,
a boundary case — not a contradiction). -/
theorem live_allows_within_kyoku :
    ReuseOK .Live .WithinKyoku = true := rfl

/-- Live admissibility is exact: the live rule admits precisely
`.None` and `.WithinKyoku` — the finite reason live-gating audits are
decidable by enumeration: only two kinds pass, so the audit checks
membership in a two-element set, never a property. Pairs
`live_forbids_backward` (the excluded end) with the two admission
theorems: together the live gate is fully characterized. -/
theorem live_admissible_iff (r : ReuseKind) :
    ReuseOK .Live r = true ↔ r = .None ∨ r = .WithinKyoku := by
  cases r <;> simp [ReuseOK]

/-- Live permits doing nothing: `.None` reuse is always allowed live.
Completes the live-admissible pair with `live_allows_within_kyoku`:
live reuse is exactly {None, WithinKyoku} — the finite reason the re-solve
loop's default posture (fresh solve, no carry) never needs a gate check.
Failure mode killed: a reuse audit flagging the clean default. -/
theorem live_allows_none :
    ReuseOK .Live .None = true := rfl

end ReuseRule

section ShardWidth

/-- Shard identity: a backward view over a length-`k+1` trajectory partitions
batch `B` into shards of width `B/(k+1)`; shards never exceed the parent batch. -/
def shardWidth (B k : ℕ) : ℕ := B / (k + 1)

/-- A shard never exceeds its parent batch. -/
theorem shard_le (B k : ℕ) : shardWidth B k ≤ B :=
  Nat.div_le_self B (k + 1)

/-- Even splits shard exactly: a batch of `B * (k+1)` over history depth
`k` gives shards of exactly `B` — the finite reason the harness sizes
batches as multiples of (depth+1) whenever it wants zero remainder:
no straggler shard, no padding branch. Pairs `shard_le` (shards never
exceed the parent): together the split is both bounded above AND exact
at multiples. -/
theorem shardWidth_exact_split (B k : ℕ) :
    shardWidth (B * (k + 1)) k = B :=
  Nat.mul_div_cancel B (Nat.succ_pos k)

/-- Wider batches never narrow shards: shard width is monotone in batch
width at fixed view — the mirror of `shard_antitone_k` (antitone in
view). Together the width moves the right way in EACH argument, so the
ledger can trade batch size against view depth monotonically: widen the
batch or shorten the view, both widen shards, never a surprise. -/
theorem shard_mono_B (B₁ B₂ k : ℕ) (h : B₁ ≤ B₂) :
    shardWidth B₁ k ≤ shardWidth B₂ k := by
  unfold shardWidth
  exact Nat.div_le_div_right h

/-- Longer backward views never widen shards: shard width is antitone in
view length. Practical core: deepening the backward view can only narrow
(or hold) the shard — parallel throughput never improves by looking
further back, so budget view length against shard width explicitly. -/
theorem shard_antitone_k (B k₁ k₂ : ℕ) (h : k₁ ≤ k₂) :
    shardWidth B k₂ ≤ shardWidth B k₁ := by
  unfold shardWidth
  exact Nat.div_le_div_left (Nat.succ_le_succ h) (Nat.succ_pos _)
/-- Live width-1 end (`moves arrive one at a time`): at width 1 every shard is
degenerate — the width-1 end of the tradeoff. -/
theorem live_width_degenerate (k : ℕ) : shardWidth 1 k ≤ 1 :=
  shard_le 1 k

/-- Empty batch shards empty: `shardWidth 0 k = 0` at any view — the
finite reason the ledger's zero-batch corner needs no special case:
no games, no shard, so empty training steps cost nothing in width.
Pairs `shard_zero_full` (zero view keeps everything): together both
zero corners are pinned (empty batch, empty view). -/
theorem shard_empty_batch (k : ℕ) : shardWidth 0 k = 0 := by
  unfold shardWidth
  simp

/-- Zero-length backward view keeps the full batch: no history means no
partitioning. Practical core: the shard-width knob's neutral element —
shallow views cost nothing in batch width, so keep views short where
parallel throughput matters. History deeper than the batch kills the
shard (proved below): both history ends pinned, full batch to empty. -/
theorem shard_zero_of_deep_history (B k : ℕ) (h : B ≤ k) :
    shardWidth B k = 0 := by
  unfold shardWidth
  rw [Nat.div_eq_zero_iff]
  right
  omega

theorem shard_zero_full (B : ℕ) : shardWidth B 0 = B := by
  unfold shardWidth
  simp

/-- Shards stay nonempty iff the batch covers the view: `k + 1 ≤ B` keeps
every shard at width `≥ 1`. Failure mode killed: a backward view longer
than the batch silently starves parallel workers with empty shards —
size views against batch width before paying for the parallelism. -/
theorem shard_nonempty_of_cover (B k : ℕ) (h : k + 1 ≤ B) :
    0 < shardWidth B k := by
  unfold shardWidth
  exact Nat.div_pos h (Nat.succ_pos k)

/-- Batch meets the maximal live view: at `k = B - 1` the divisor is
`B`, so the shard is `B / B` — the boundary case of
`shard_nonempty_of_cover` where the view is as deep as the batch
allows while staying live. Pairs `shard_zero_of_deep_history` (one
deeper kills it): together the live/dead threshold sits exactly
between `k = B - 1` and `k = B`. Batch one wider than the view gives
unit shards (proved next): the live region bottoms at width one. -/
theorem shard_unit_of_cover_plus_one (B : ℕ) :
    shardWidth (B + 1) B = 1 := by
  unfold shardWidth
  apply Nat.div_self
  omega

theorem shard_live_boundary (B : ℕ) (hB : 0 < B) :
    shardWidth B (B - 1) = B / B := by
  unfold shardWidth
  have h : B - 1 + 1 = B := Nat.succ_pred_eq_of_pos hB
  rw [h]

end ShardWidth

section Decoupling

/-- Decoupling fix: per-iteration training is sample + gradient descent with ZERO MCTS;
MCTS concentrates in periodic whole-buffer reanalyze; per-iteration MCTS in training
re-couples search and training and fails closed. Training-iteration MCTS calls by phase. -/
def trainIterMCTS : Phase → ℕ
  | .Offline => 0
  | .Live => 1

/-- Decoupled offline training invokes no MCTS per iteration. -/
theorem decouple_zeroes_train_mcts : trainIterMCTS .Offline = 0 := rfl

/-- Live training keeps one MCTS call per iteration: the online leg stays
coupled while offline stays decoupled — the finite reason the two phases
have different compute profiles. Pair with `decouple_zeroes_train_mcts`:
together the phase split is pinned at both ends, so a phase-confusion bug
(offline running live's loop or vice versa) fails closed on sight. -/
theorem live_keeps_train_mcts : trainIterMCTS .Live = 1 := rfl

/-- Decoupling saves exactly one MCTS call per iteration: live spends one,
offline zero — the finite reason the offline throughput win is countable
per step, so reanalyze budgeting can price the saved call precisely.
Pairs `offline_train_mcts_zero` (offline is zero) with
`live_keeps_train_mcts` (live is one): together the per-iteration gap
is exactly unity, never approximate. -/
theorem offline_saves_one_mcts :
    trainIterMCTS .Live - trainIterMCTS .Offline = 1 := by
  decide

/-- Saved calls compound linearly: over `n` iterations offline saves
exactly `n` MCTS calls — the finite reason reanalyze budgeting scales
the per-iteration unity gap by the step count with no compounding
surprise. Lifts `offline_saves_one_mcts` from one step to the whole
run, so the budget planner multiplies, never re-derives. -/
theorem offline_saves_n_mcts (n : ℕ) :
    n * trainIterMCTS .Live - n * trainIterMCTS .Offline = n := by
  simp [trainIterMCTS]

/-- Live spends exactly one MCTS call per iteration: over `n` iterations
the live loop pays `n` — the mirror of `offline_saves_n_mcts` (offline
saves `n`). Together the per-run ledger is pinned on both sides
(spend `n`, save `n`), so the reanalyze budget converts saved calls
one-for-one with no exchange rate. -/
theorem live_spends_n_mcts (n : ℕ) :
    n * trainIterMCTS .Live = n := by
  simp [trainIterMCTS]

/-- Per-iteration budget across both modes is exactly one: live spends
one, offline spends zero, total one — the finite reason the mode switch
is a pure reallocation, never a budget change: the iteration's single
MCTS call moves to reanalyze under offline with nothing created or
destroyed. Pairs `live_spends_n_mcts` (live side) with
`offline_train_mcts_zero` (offline side): together the two-sided
ledger closes. -/
theorem trainIterMCTS_modes_sum_one :
    trainIterMCTS .Live + trainIterMCTS .Offline = 1 := by
  simp [trainIterMCTS]

/-- Offline strictly undercuts live on per-iteration MCTS: decoupling is a
strict saving, never a tie. Practical core: the offline throughput win is
provable per iteration — budget the saved MCTS calls into reanalyze or
larger batches. Zero-at-offline is the sharp form (proved below):
decoupling doesn't just save, it eliminates per-iteration MCTS outright. -/
theorem offline_train_mcts_zero : trainIterMCTS .Offline = 0 := rfl

theorem offline_lt_live_train_mcts : trainIterMCTS .Offline < trainIterMCTS .Live := by
  decide

/-- Reanalyze cadence menu (replay ratio 0.25, reanalyze ratio 1). Length only —
no return-optimality claim (needs an env model, stays open); claiming optimality
without a model fails closed. -/
def reanalyzeGrid : List ℚ := [1 / 3, 1, 2]

theorem reanalyzeGrid_length : reanalyzeGrid.length = 3 := rfl

/-- Reanalyze grid ascends strictly: the cadence menu sweeps sparse to
dense with no ties — the finite reason the replay-ratio sweep never
revisits a setting and ablation results compare along one ordered axis.
Pair with `reanalyzeGrid_length` (three settings): together the sweep is
a fixed strict menu, so each step strictly densifies reanalysis. -/
theorem reanalyzeGrid_strict_asc : reanalyzeGrid[0]? = some (1 / 3 : ℚ)
    ∧ reanalyzeGrid[1]? = some (1 : ℚ) ∧ reanalyzeGrid[2]? = some (2 : ℚ)
    ∧ (1 / 3 : ℚ) < 1 ∧ (1 : ℚ) < 2 := by
  refine ⟨rfl, rfl, rfl, by norm_num, by norm_num⟩

/-- Unit replay ratio stays on the menu: the baseline reanalyze ratio 1
is always swept — the finite reason ablations never drop the no-reanalyze
control. Pair with `reanalyzeGrid_strict_asc` (strict order): together
the menu is an ordered sweep that always includes the identity point.
The densest ratio 2 is pinned too (proved next): identity through
double-density covered. -/
theorem reanalyzeGrid_contains_two : (2 : ℚ) ∈ reanalyzeGrid := by
  unfold reanalyzeGrid
  simp

/-- Sparse endpoint stays on the menu: ratio 1/3 is always swept — the
mirror of `reanalyzeGrid_contains_two` (ratio 2 always swept). Together
all three menu points are pinned (sparse, identity, dense), so the
sweep covers the full ordered axis with no droppable endpoint. -/
theorem reanalyzeGrid_contains_third : (1 / 3 : ℚ) ∈ reanalyzeGrid := by
  unfold reanalyzeGrid
  simp

theorem reanalyzeGrid_contains_unit : (1 : ℚ) ∈ reanalyzeGrid := by
  unfold reanalyzeGrid
  simp

end Decoupling

end Hydra2.Blueprint.SearchHygiene
