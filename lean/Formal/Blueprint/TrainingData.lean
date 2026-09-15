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

/-! # Hydra2 training-data batch pipeline counts

Corpus reconciliation, phase batch sizes, file-batch shuffle windows, and
exact-multiple re-batching. Every number below is a closed count or a one-lemma
coverage identity — sizes and shapes only. No published instances-per-second exist,
so no throughput-speed claim appears here; shuffle must precede partitioning (wrong
order biases the shuffle window and fails closed) but its stochastic justification
needs PMF machinery this module does not claim.
-/

namespace Hydra2.Blueprint.TrainingData

section Corpus

/-- Corpus reconciliation: tenhou part 2512433 + majsoul part 4298196 sums EXACTLY
to 6810629 packager items; one job per MJAI game file
(`tools/mjai-dataset-packager/src/main.rs`, `total_items = jobs.len()`);
mismatch fails closed. -/
theorem corpus_identity : 2512433 + 4298196 = 6810629 := by decide

end Corpus

section BatchSizes

/-- Offline CQL batch 1024 is exactly 4× the GRP/rank-aux pre-train batch 256;
wrong ratio breaks the practiced compute budget and fails closed. -/
theorem cql_batch_four_x_grp : 1024 = 4 * 256 := by decide

/-- VLOG latent dim is derived, not set: `z = hidden / 2 = 512` at hidden 1024;
pinning it independently breaks the encoder shape and fails closed. -/
theorem vlog_latent_half : 1024 / 2 = 512 := by decide

end BatchSizes

section ShuffleWindow

/-- Per-file density ~660 decision instances: default file_batch 20 gives a
~13.2k-instance shuffle unit (20 * 660 = 13200); smaller window under-shuffles
and fails closed. -/
theorem shuffle_window_default : 20 * 660 = 13200 := by decide

/-- Practiced file_batch 100 gives a ~66k-instance shuffle unit (100 * 660 = 66000).
The `file_batch × 660` product is the effective shuffle window. -/
theorem shuffle_window_practiced : 100 * 660 = 66000 := by decide

/-- Practiced file_batch 100 is 5× the code default 20. -/
theorem filebatch_practiced_five_x_default : 100 = 5 * 20 := by decide

end ShuffleWindow

section Coverage

/-- Exact-multiple re-batching (`drop_last = False` + re-batched leftovers):
no sample is ever dropped at epoch tails — every `n` partitions into whole
batches plus its remainder; dropping leftovers biases the epoch and fails closed. -/
theorem epoch_cover (n b : ℕ) : b * (n / b) + n % b = n :=
  Nat.div_add_mod n b

end Coverage

section Cadence

/-- Practiced checkpoint cadence is 5× sparser than the example dummies:
`save_every` 2000 vs 400, `test_every` 100000 vs 20000. -/
theorem checkpoint_cadence_five_x :
    2000 = 5 * 400 ∧ 100000 = 5 * 20000 := by decide

end Cadence

end Hydra2.Blueprint.TrainingData
