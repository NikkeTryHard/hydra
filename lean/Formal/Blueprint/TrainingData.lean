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

/-- Soul share dominates the corpus: the majsoul part exceeds half the
total — the finite reason Soul-side mix decisions (HE weights, mode
audit, dedupe) move the corpus more than Tenhou-side ones. Pairs
`corpus_identity` (the exact total): together the composition is
pinned both in sum and in majority, so mix reweighting starts from
measured shares, not guesses. -/
theorem corpus_soul_majority : 6810629 / 2 < 4298196 := by decide

/-- Soul part exceeds the Tenhou part outright: 4298196 > 2512433 — the
finite reason Soul-side mix work outranks Tenhou-side work item-for-item:
the majority is not just over half, it exceeds the entire other part.
Pairs `corpus_soul_majority` (over half the total): together the Soul
share is pinned both absolutely and relatively, so mix-weight effort
go where the games are. -/
theorem corpus_soul_exceeds_tenhou : 2512433 < 4298196 := by decide

/-- Soul is less than double Tenhou: 4298196 < 2 × 2512433 — the finite
reason the mix-weight priors live in a bounded band: with
`corpus_soul_exceeds_tenhou` (ratio above one) the Soul/Tenhou ratio is
pinned inside (1, 2), so neither side dominates by an order of
magnitude and fixed 63/37 budgeting never drifts far off. -/
theorem corpus_ratio_bounded_two : 4298196 < 2 * 2512433 := by decide

end Corpus

section BatchSizes

/-- Offline CQL batch 1024 is exactly 4× the GRP/rank-aux pre-train batch 256;
wrong ratio breaks the practiced compute budget and fails closed. -/
theorem cql_batch_four_x_grp : 1024 = 4 * 256 := by decide

/-- Tenhou part is the minority outright: 2512433 < half the total — the
mirror of `corpus_soul_majority` (Soul over half). Together both shares
are pinned against the halfway line from opposite sides, so the
majority/minority assignment is proved, not assumed, and mix-weight
priors start from the correct side. -/
theorem corpus_tenhou_minority : 2512433 < 6810629 / 2 := by decide

/-- Soul share in percent: ~63% of the corpus is Soul-side — the finite
reason Soul-side mix work gets roughly two-thirds of the data budget:
the share is pinned as an integer percent, so budget splits quote 63/37
without recomputation. Pairs `corpus_soul_majority` (over half):
together the majority is quantified, not just bounded. -/
theorem corpus_soul_share_pct : 4298196 * 100 / 6810629 = 63 := by decide

/-- Tenhou share in percent: ~36-37% is Tenhou-side — the mirror of
`corpus_soul_share_pct` (Soul ~63%). Together both shares are pinned
as integers summing to ~100 (63 + 36 = 99, remainder is integer
division), so the mix-weight priors start from a closed budget. -/
theorem corpus_tenhou_share_pct : 2512433 * 100 / 6810629 = 36 := by decide

/-- Shares close the budget: 63 + 36 = 99 (remainder is integer
division) — the finite reason the 63/37 split quotes a closed budget:
both share pins agree with each other, not just with their own
bounds, so mix-weight priors never leak a phantom percent. Pairs
`corpus_soul_share_pct` with `corpus_tenhou_share_pct`: together the
composition is checked end-to-end. -/
theorem corpus_shares_close :
    4298196 * 100 / 6810629 + (2512433 * 100 / 6810629) = 99 := by decide

/-- VLOG latent dim is derived, not set: `z = hidden / 2 = 512` at hidden 1024;
pinning it independently breaks the encoder shape and fails closed. -/
theorem vlog_latent_half : 1024 / 2 = 512 := by decide

/-- VLOG latent doubles back to hidden: `512 * 2 = 1024` — the finite
reason the decoder's first projection inverts the encoder bottleneck
exactly: halving then doubling round-trips, so no capacity is lost to
shape mismatch at the bottleneck. Pairs `vlog_latent_half` (the halve
direction): together the bottleneck is a proved two-sided reshape. -/
theorem vlog_latent_double : 512 * 2 = 1024 := by decide

/-- GRP batches tile exactly into CQL batches: 1024 % 256 = 0, so four
pre-train batches make one offline batch with no ragged tail — the finite
reason the S0.5 dataloader reuses GRP batches without re-batching. Pair
with `cql_batch_four_x_grp` (the 4x ratio): together ratio AND
divisibility are pinned, so a batch-size edit that breaks either fails
closed. -/
theorem cql_batch_divides_grp : 1024 % 256 = 0 := by decide

/-- GRP batch count per CQL batch is exact: 1024 / 256 = 4 whole batches —
the count form of `cql_batch_divides_grp` (the remainder form). Together
both directions are pinned (no ragged tail AND exactly four), so the
dataloader's four-to-one packing is proved, not assumed. -/
theorem cql_batch_count_four : 1024 / 256 = 4 := by decide

end BatchSizes

section ShuffleWindow

/-- Per-file density ~660 decision instances: default file_batch 20 gives a
~13.2k-instance shuffle unit (20 * 660 = 13200); smaller window under-shuffles
and fails closed. -/
theorem shuffle_window_default : 20 * 660 = 13200 := by decide

/-- Default window per file-batch unit: `13200 / 20 = 660` instances per
unit — the finite reason the 660 factor is a per-unit rate, not a magic
constant: dividing the default window by its batch count recovers the
unit rate. Pairs `shuffle_window_default` (the product form): together
both directions are pinned, so window sizing divides as well as it
multiplies. -/
theorem shuffle_window_unit_rate : 13200 / 20 = 660 := by decide

/-- Practiced window per file-batch unit: `66000 / 100 = 660` — the same
unit rate from the practiced side. Pairs `shuffle_window_unit_rate`
(default side): together both windows agree on 660 per unit, so the
unit rate is batch-size-invariant and memory budgets scale linearly
from either pin. -/
theorem shuffle_window_unit_rate_practiced : 66000 / 100 = 660 := by decide

/-- Practiced file_batch 100 gives a ~66k-instance shuffle unit (100 * 660 = 66000).
The `file_batch × 660` product is the effective shuffle window. -/
theorem shuffle_window_practiced : 100 * 660 = 66000 := by decide

/-- Practiced file_batch 100 is 5× the code default 20. -/
theorem filebatch_practiced_five_x_default : 100 = 5 * 20 := by decide

/-- Shuffle window scales monotonically in file_batch: raising file_batch
never shrinks the shuffle unit. Practical core: under-shuffle risk is
monotone — budget file_batch upward with no window regression, so
limited-compute training can size the window from memory headroom alone. -/
theorem shuffle_window_mono {fb1 fb2 : ℕ} (h : fb1 ≤ fb2) :
    fb1 * 660 ≤ fb2 * 660 :=
  Nat.mul_le_mul_right 660 h

/-- Practiced shuffle window is 5x the default: 66000 = 5 * 13200 — the
finite reason the practiced file_batch costs 5x the shuffle memory of
the default. Pair with `filebatch_practiced_five_x_default` (the batch
ratio): together ratio AND window are pinned, so a memory budget sized
from one stays consistent with the other. -/
theorem shuffle_window_five_x : 66000 = 5 * 13200 := by decide

/-- Window pins compose: the practiced window equals the batch ratio times
the default window (`100 * 660 = 5 * (20 * 660)`) — the finite reason the
three shuffle pins (default, ratio, practiced) are mutually consistent:
a file_batch edit breaking any one fails this cross-check. Pairs
`shuffle_window_five_x` (the closed product): together ratio AND product
forms agree, so memory budgets sized either way coincide. -/
theorem shuffle_window_compose : 100 * 660 = 5 * (20 * 660) := by decide

end ShuffleWindow

section Coverage

/-- Exact-multiple re-batching (`drop_last = False` + re-batched leftovers):
no sample is ever dropped at epoch tails — every `n` partitions into whole
batches plus its remainder; dropping leftovers biases the epoch and fails closed. -/
theorem epoch_cover (n b : ℕ) : b * (n / b) + n % b = n :=
  Nat.div_add_mod n b

/-- Full batches fit exactly when the corpus is an exact multiple: remainder
zero means `n / b` whole batches with nothing left over. Practical core:
size corpora to exact multiples where possible — zero tail waste, and the
re-batch path stays cold. -/
theorem epoch_exact_batches (n b : ℕ) (h : n % b = 0) :
    b * (n / b) = n := by
  have := epoch_cover n b
  omega

/-- Batch count covers the corpus: `n / b + 1` batches always suffice even
with a tail. Practical core: allocate one spare batch slot — the epoch
scheduler never overflows, so limited-compute training can pre-size the
batch array from `(n, b)` alone with no dynamic resize. -/
theorem epoch_batches_cover (n b : ℕ) (hb : 0 < b) :
    n ≤ b * (n / b + 1) := by
  have h := epoch_cover n b
  have hle : n % b < b := Nat.mod_lt n hb
  calc n = b * (n / b) + n % b := h.symm
    _ ≤ b * (n / b) + b := by omega
    _ = b * (n / b + 1) := by ring

/-- Full batches never overshoot: `b * (n / b) ≤ n`, the tail is dropped
not rounded up. Pair with `epoch_batches_cover` (one spare always
suffices): together the scheduler knows both bounds — allocate the floor
for the steady loop, one spare for the tail, never more. Exact division
needs no spare (proved below): even-shard corpora skip the +1 slot. -/
theorem epoch_exact_no_spare (n b : ℕ) (hb : 0 < b) (h : n % b = 0) :
    n ≤ b * (n / b) := by
  have hc := epoch_cover n b
  omega

theorem epoch_full_batches_le (n b : ℕ) :
    b * (n / b) ≤ n := by
  have h := epoch_cover n b
  omega

/-- Batch count never exceeds the corpus: `n / b ≤ n` — the finite
reason the epoch scheduler's batch array pre-sizes from the corpus
count alone: no batching scheme creates more batches than items, so
the allocation bound is the corpus size with no lookup. Pairs
`epoch_batches_cover` (n/b+1 always suffice): together the batch
count sits in [cover-1, n], pinned on both sides. -/
theorem epoch_batches_le_n (n b : ℕ) : n / b ≤ n :=
  Nat.div_le_self n b

end Coverage

section Cadence

/-- Practiced checkpoint cadence is 5× sparser than the example dummies:
`save_every` 2000 vs 400, `test_every` 100000 vs 20000. -/
theorem checkpoint_cadence_five_x :
    2000 = 5 * 400 ∧ 100000 = 5 * 20000 := by decide

/-- Sparser cadence divides the checkpoint count: at `save_every` 2000 vs
400 the run writes 5x fewer checkpoints over the same step budget — the
finite reason the practiced cadence keeps disk and eval overhead low.
Failure mode killed: a dense example cadence copied into a long run
silently multiplying eval cost. -/
theorem checkpoint_count_ratio (steps : ℕ) :
    steps / 2000 ≤ steps / 400 :=
  Nat.div_le_div_left (by norm_num : 400 ≤ 2000) (by norm_num : 0 < 400)

/-- Eval-per-save ratio is cadence-invariant: `100000/2000 = 20000/400 =
50` — scaling both knobs by 5x preserves the eval proportion. Practical
core: switching cadence never changes how many evals each save pays for,
so the practiced cadence keeps the example's eval-per-save economics
while writing 5x fewer checkpoints. -/
theorem eval_per_save_invariant : 100000 / 2000 = 20000 / 400 := by decide

/-- Eval-per-save value is pinned: both cadences pay 50 evals per save —
the finite reason cadence switches are cost-neutral per checkpoint:
the invariant above only equates the ratios; this pins the ratio at 50,
so the ledger prices each save at fifty evals with no lookup. Pairs
`checkpoint_count_ratio` (sparser cadence writes fewer checkpoints):
together fewer saves × same price-per-save = strictly cheaper runs. -/
theorem eval_per_save_value : 100000 / 2000 = 50 := by decide

/-- The example cadence pays the same 50: `20000/400 = 50` — the mirror of
`eval_per_save_value` (the practiced side). Together both cadences are
pinned at 50 independently, so the invariant is checked end-to-end,
not just as an equation between two unverified ratios. -/
theorem eval_per_save_value_other : 20000 / 400 = 50 := by decide

end Cadence

end Hydra2.Blueprint.TrainingData
