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

/-- Live permits the within-kyoku draw-subtree carry (the maximal live reuse,
a boundary case — not a contradiction). -/
theorem live_allows_within_kyoku :
    ReuseOK .Live .WithinKyoku = true := rfl

end ReuseRule

section ShardWidth

/-- Shard identity: a backward view over a length-`k+1` trajectory partitions
batch `B` into shards of width `B/(k+1)`; shards never exceed the parent batch. -/
def shardWidth (B k : ℕ) : ℕ := B / (k + 1)

/-- A shard never exceeds its parent batch. -/
theorem shard_le (B k : ℕ) : shardWidth B k ≤ B :=
  Nat.div_le_self B (k + 1)

/-- Longer backward views mean narrower shards (narrower shards diminish
the benefits of parallelized search). -/
theorem shard_succ_le (B k : ℕ) :
    shardWidth B (k + 1) ≤ shardWidth B k := by
  unfold shardWidth
  exact Nat.div_le_div_left (Nat.le_succ _) (Nat.succ_pos _)
/-- Live width-1 end (`moves arrive one at a time`): at width 1 every shard is
degenerate — the width-1 end of the tradeoff. -/
theorem live_width_degenerate (k : ℕ) : shardWidth 1 k ≤ 1 :=
  shard_le 1 k

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

/-- Reanalyze cadence menu (replay ratio 0.25, reanalyze ratio 1). Length only —
no return-optimality claim (needs an env model, stays open); claiming optimality
without a model fails closed. -/
def reanalyzeGrid : List ℚ := [1 / 3, 1, 2]

theorem reanalyzeGrid_length : reanalyzeGrid.length = 3 := rfl

end Decoupling

end Hydra2.Blueprint.SearchHygiene
