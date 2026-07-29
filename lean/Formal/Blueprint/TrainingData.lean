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

/-! # Hydra2 training-data batch pipeline counts (batch-guidance)

Mirrors `ideas/training-data/batch-guidance.md`: corpus reconciliation
(§1), phase batch sizes (§2), file_batch shuffle windows (§3), and
exact-multiple re-batching (§3). Every number below is a closed count from
the doc or a one-lemma coverage identity — sizes and shapes only. The doc's
own GAP (§3, L95-97) states no published instances/sec exist, so no
throughput-speed claim appears here; the shuffle-before-split ORDER rule
(§4, L104-109) stays doc-level since its stochastic justification needs PMF
machinery this module does not claim.
-/

namespace Hydra2.Blueprint.TrainingData

section Corpus

/-- D-017 reconciliation (`batch-guidance.md` L17-18, L23-28): tenhou part +
majsoul part sums EXACTLY to `packager_items`; one job per MJAI game file
(`tools/mjai-dataset-packager/src/main.rs`, `total_items = jobs.len()`). -/
theorem corpus_identity : 2512433 + 4298196 = 6810629 := by decide

end Corpus

section BatchSizes

/-- Offline CQL batch 1024 (Mortal-298k / VLOG practice, L56-60) is exactly
4× the GRP/rank-aux pre-train batch 256 (L61). -/
theorem cql_batch_four_x_grp : 1024 = 4 * 256 := by decide

/-- VLOG latent dim is derived, not set (`batch-guidance.md` L62-63 and
do-not-pin L133): `z = hidden / 2 = 512` at hidden 1024. -/
theorem vlog_latent_half : 1024 / 2 = 512 := by decide

end BatchSizes

section ShuffleWindow

/-- Per-file density ~660 decision instances (`dataloader.py` comment, L81):
default file_batch 20 ⇒ ~13.2k-instance shuffle unit (L83). -/
theorem shuffle_window_default : 20 * 660 = 13200 := by decide

/-- Practiced file_batch 100 (298k) ⇒ ~66k-instance shuffle unit (L84).
The `file_batch × 660` product is the effective shuffle window (L117-118). -/
theorem shuffle_window_practiced : 100 * 660 = 66000 := by decide

/-- Practiced file_batch is 5× the code default (L83-84). -/
theorem filebatch_practiced_five_x_default : 100 = 5 * 20 := by decide

end ShuffleWindow

section Coverage

/-- Exact-multiple re-batching (`batch-guidance.md` L88-89: `drop_last =
False` + `train.py` re-batches leftovers): no sample is ever dropped at
epoch tails — every `n` splits into whole batches plus its remainder. -/
theorem epoch_cover (n b : ℕ) : b * (n / b) + n % b = n :=
  Nat.div_add_mod n b

end Coverage

section Cadence

/-- Practiced checkpoint cadence is 5× sparser than the example-toml
dummies (L90-91): `save_every` 2000 vs 400, `test_every` 100000 vs 20000. -/
theorem checkpoint_cadence_five_x :
    2000 = 5 * 400 ∧ 100000 = 5 * 20000 := by decide

end Cadence

end Hydra2.Blueprint.TrainingData
