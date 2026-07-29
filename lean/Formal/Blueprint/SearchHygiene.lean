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

/-! # Hydra2 search-hygiene: offline-reanalyze-only reuse (requiem)

Mirrors `ideas/search-hygiene/requiem.md` (ReZero backward-view port): the
RULE (L172-173) — backward-view/subtree reuse allowed in offline
replay/reanalyze ONLY, FORBIDDEN on the live hanchan path — plus the shard
identity behind it (L35-36: a length-`k+1` backward view splits batch `B`
into shards of width `B/(k+1)`, L21-23). Shard math and the reuse predicate
are finite (`ℕ`/`Bool`/`Finset`); the U-curve device argmax (L78-106) stays
a documented re-sweep obligation since Fig.12 values are not transcribed
(doc §5 Gap), and return-optimality over cadence needs an env model.
Boundary note: `gating.md` L107 `gateTreeReuse = true` (within-kyoku
draw-subtree carry) is the maximal live reuse this rule permits.
-/

namespace Hydra2.Blueprint.SearchHygiene

section ReuseRule

/-- Search phase: live hanchan path vs offline replay/reanalyze. -/
inductive Phase where
  | Live
  | Offline
  deriving DecidableEq

/-- Reuse kind: none, within-kyoku draw-subtree carry (gating L107
boundary), or full backward-view subtree reuse (ReZero). -/
inductive ReuseKind where
  | None
  | WithinKyoku
  | Backward
  deriving DecidableEq

/-- The requiem RULE (`requiem.md` L172-173, legs L177-188): offline allows
everything; live allows at most within-kyoku carry, never backward-view. -/
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

/-- Live permits the within-kyoku draw-subtree carry (gating L107: the
maximal live reuse, a boundary case — not a contradiction). -/
theorem live_allows_within_kyoku :
    ReuseOK .Live .WithinKyoku = true := rfl

end ReuseRule

section ShardWidth

/-- Shard identity (`requiem.md` L35-36, paper L21): a backward view over a
length-`k+1` trajectory splits batch `B` into shards of width `B/(k+1)`. -/
def shardWidth (B k : ℕ) : ℕ := B / (k + 1)

/-- A shard never exceeds its parent batch. -/
theorem shard_le (B k : ℕ) : shardWidth B k ≤ B :=
  Nat.div_le_self B (k + 1)

/-- Longer backward views mean narrower shards (the L22-23 slowdown:
`diminishes benefits of parallelized search`). -/
theorem shard_succ_le (B k : ℕ) :
    shardWidth B (k + 1) ≤ shardWidth B k := by
  unfold shardWidth
  exact Nat.div_le_div_left (Nat.le_succ _) (Nat.succ_pos _)
/-- Live leg L177 (`moves arrive one at a time`): at width 1 every shard is
degenerate — the width-1 end of the tradeoff. -/
theorem live_width_degenerate (k : ℕ) : shardWidth 1 k ≤ 1 :=
  shard_le 1 k

end ShardWidth

section Decoupling

/-- Decoupling fix (`requiem.md` L44-47, L61-62): per-iteration training is
sample + gradient descent with ZERO MCTS; MCTS concentrates in periodic
whole-buffer reanalyze. Training-iteration MCTS calls by phase. -/
def trainIterMCTS : Phase → ℕ
  | .Offline => 0
  | .Live => 1

/-- Decoupled offline training invokes no MCTS per iteration. -/
theorem decouple_zeroes_train_mcts : trainIterMCTS .Offline = 0 := rfl

/-- Reanalyze cadence menu, §5.3 ablation (`requiem.md` L68-73; replay
ratio 0.25, reanalyze ratio 1). Length only — no return-optimality
claim (needs an env model, doc §5 Gap). -/
def reanalyzeGrid : List ℚ := [1 / 3, 1, 2]

theorem reanalyzeGrid_length : reanalyzeGrid.length = 3 := rfl

end Decoupling

end Hydra2.Blueprint.SearchHygiene
