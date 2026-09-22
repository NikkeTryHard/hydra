"""Baseline metric math: masked NLL, top-k, calibration, held-out split.

Tiny-shard overfit to NLL 0.15 / top-1 0.90; complete reference games with
zero illegal actions/timeouts; hidden permutation and canary tests; eager
FP32 oracle recorded.

Owns the deterministic kernels over the legal subspace: masked cross-entropy
with illegal logits excluded, legal-uniform comparison, top-k accuracy,
frozen-bin calibration error, the metric bundle with its stable digest, and
the disjoint train/held-out partition. Process-level evaluation
(fresh-process repeat, tiny-shard overfit, reference games, canonical report)
lives in :mod:`hydra2.eval.baseline_eval`.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError, DigestText

__all__ = [
    "BASELINE_METRICS_VERSION",
    "COMPILE_ORDER",
    "EAGER_ORACLE_ID",
    "OVERFIT_NLL_THRESHOLD",
    "OVERFIT_TOP1_THRESHOLD",
    "BaselineMetrics",
    "HeldOutSplit",
    "compute_baseline_metrics",
    "expected_calibration_error",
    "legal_uniform_nll",
    "masked_cross_entropy",
    "split_held_out",
    "top_k_accuracy",
    "verify_held_out_disjoint",
]


BASELINE_METRICS_VERSION = "1.0.0"
OVERFIT_NLL_THRESHOLD = 0.15
OVERFIT_TOP1_THRESHOLD = 0.90
# Compile ladder order is fixed — eager is the semantic oracle (eager FP32
# exact simulator, plain PyTorch, padded/bucketed histories, SDPA, dense
# action head, AdamW default, ordinary checkpoint); later arms qualify only
# in order with the eager comparator.
COMPILE_ORDER = ("eager", "default", "max-autotune-no-cudagraphs", "max-autotune")
EAGER_ORACLE_ID = "eager_fp32"


def _require_finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{name} must be finite, got {value!r}")
    return float(value)


def _require_legal_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ContractError(f"legal_mask must be a Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise ContractError(f"legal_mask must be bool dtype, got {mask.dtype}")
    if mask.dim() != 2:
        raise ContractError(f"legal_mask must be [B,A], got shape {tuple(mask.shape)}")
    if mask.shape[0] == 0:
        raise ContractError("legal_mask batch dimension must be > 0")
    # Nonterminal all-false is hard error (mask before softmax/loss/argmax;
    # illegal probability exactly zero; dora shape fixed at (5,), never a padded (4,)).
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            mask.any(dim=1),
            lambda: "nonterminal all-false legal row is hard error",
        )
    elif torch.all(mask.any(dim=1)).item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for contract; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("nonterminal all-false legal row is hard error")
    return mask


def _require_targets(targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(targets, torch.Tensor):
        raise ContractError(f"targets must be Tensor, got {type(targets).__name__}")
    if targets.dim() != 1:
        raise ContractError(f"targets must be [B], got shape {tuple(targets.shape)}")
    if targets.shape[0] != legal_mask.shape[0]:
        raise ContractError(
            f"targets batch {targets.shape[0]} != legal_mask batch {legal_mask.shape[0]}"
        )
    if targets.dtype not in (torch.int32, torch.int64):
        raise ContractError(f"targets must be int dtype, got {targets.dtype}")
    # Each target must be legal per its row — hoisted for compile compatibility.
    if torch.compiler.is_compiling():
        torch._check_tensor_all(targets >= 0, lambda: "target out of range (negative)")
        torch._check_tensor_all(targets < legal_mask.shape[1], lambda: "target out of range (>= A)")
        # Indexing after range check is safe; gather legality via vectorized gather.
        _ar = torch.arange(targets.shape[0], device=targets.device)
        torch._check_tensor_all(
            legal_mask[_ar, targets.to(torch.long)],
            lambda: "target is illegal (masked)",
        )
    else:
        for i in range(targets.shape[0]):
            a = int(targets[i].item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for row error; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            if a < 0 or a >= legal_mask.shape[1]:
                raise ContractError(f"target {a} out of range [0,{legal_mask.shape[1]}) at row {i}")
            if not bool(legal_mask[i, a].item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for legality check; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                raise ContractError(f"target {a} is illegal at row {i} (masked)")
    return targets


def _require_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ContractError(f"logits must be Tensor, got {type(logits).__name__}")
    if logits.dim() != 2:
        raise ContractError(f"logits must be [B,A], got shape {tuple(logits.shape)}")
    if logits.shape != legal_mask.shape:
        raise ContractError(
            f"logits shape {tuple(logits.shape)} != legal_mask shape {tuple(legal_mask.shape)}"
        )
    if not logits.dtype.is_floating_point:
        raise ContractError(f"logits must be floating dtype, got {logits.dtype}")
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            torch.isfinite(logits), lambda: "logits must be finite (no inf/nan)"
        )
    elif torch.isfinite(logits).all().item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for finiteness check; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("logits must be finite (no inf/nan)")
    return logits


def _seed_everything(seed: int) -> None:
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic algorithms so library callers match the test
    # fixture without pytest.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor
) -> float:
    """Mean cross-entropy over legal actions only, target must be legal."""
    mask = _require_legal_mask(legal_mask)
    t = _require_targets(targets, mask)
    logits_v = _require_logits(logits, mask)
    # Mask illegal logits to -inf before log-softmax so they get zero prob.
    masked: torch.Tensor = logits_v.masked_fill(~mask, float("-inf"))
    log_probs: torch.Tensor = torch.log_softmax(masked, dim=-1)
    # Gather target log-prob; illegal targets already rejected.
    per_row = -log_probs[torch.arange(logits_v.shape[0], device=logits_v.device), t]
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            torch.isfinite(per_row),
            lambda: "masked NLL produced non-finite value",
        )
    elif torch.isfinite(per_row).all().item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for NLL check; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("masked NLL produced non-finite value")
    return float(per_row.mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for metric reporting; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html


def legal_uniform_nll(targets: torch.Tensor, legal_mask: torch.Tensor) -> float:
    """NLL of the legal-uniform baseline: -log(1 / num_legal)."""
    mask = _require_legal_mask(legal_mask)
    _ = _require_targets(targets, mask)
    # Uniform prob is 1 / num_legal per row; NLL = log(num_legal) independent of target choice
    # as long as target is legal (already validated).
    counts = mask.sum(dim=1).to(torch.float64)
    nll = torch.log(counts)
    return float(nll.mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for metric reporting; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html


def top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor, k: int
) -> float:
    """Accuracy of target in top-k legal logits."""
    if not isinstance(k, int) or k <= 0:
        raise ContractError(f"k must be positive int, got {k!r}")
    mask = _require_legal_mask(legal_mask)
    t = _require_targets(targets, mask)
    logits_v = _require_logits(logits, mask)
    masked = logits_v.masked_fill(~mask, float("-inf"))
    # Number of legal per row may be < k; then top-k is at most that many.
    # torch.topk with k > A would error, so clamp k to A.
    k_clamped = min(k, logits_v.shape[1])
    _, topk_idx = torch.topk(masked, k=k_clamped, dim=-1)
    hits = (topk_idx == t.unsqueeze(1)).any(dim=1).to(torch.float64)
    return float(hits.mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for metric reporting; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html


def expected_calibration_error(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor, num_bins: int = 10
) -> float:
    """ECE over legal softmax confidence (max prob), frozen 10-bin grid.

    The default matches the frozen ``_ECE_NUM_BINS`` grid in
    ``training/objectives.py`` so train and eval calibration agree; pass an
    explicit ``num_bins`` only for ablations (digests then differ).
    """
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ContractError(f"num_bins must be positive int, got {num_bins!r}")
    mask = _require_legal_mask(legal_mask)
    t = _require_targets(targets, mask)
    logits_v = _require_logits(logits, mask)
    masked = logits_v.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    # Confidence = max prob among legal; prediction = argmax among legal.
    conf, pred = probs.max(dim=-1)
    correct = (pred == t).to(torch.float64)
    conf_f = conf.to(torch.float64)
    ece = 0.0
    n = float(logits_v.shape[0])
    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins
        # Include high edge in last bin.
        if b == num_bins - 1:
            in_bin = (conf_f >= low) & (conf_f <= high)
        else:
            in_bin = (conf_f >= low) & (conf_f < high)
        bin_count = int(in_bin.sum().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for bin count; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        if bin_count == 0:
            continue
        bin_acc = float(correct[in_bin].mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for bin accuracy; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        bin_conf = float(conf_f[in_bin].mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for bin confidence; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        ece += abs(bin_acc - bin_conf) * (bin_count / n)
    if not math.isfinite(ece):
        raise ContractError(f"ECE must be finite, got {ece!r}")
    return ece


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """Computed baseline metrics for one evaluation slice."""

    masked_nll: float
    top1_accuracy: float
    top3_accuracy: float
    ece: float
    legal_uniform_nll: float
    nll_vs_uniform_delta: float
    count: int
    compile_mode: str = EAGER_ORACLE_ID
    digest: DigestText = field(
        # Default digest; the factory overwrites it with the computed digest.
        default=DigestText("sha256:" + "0" * 64)
    )

    def __post_init__(self) -> None:
        _ = _require_finite(self.masked_nll, "masked_nll")
        _ = _require_finite(self.top1_accuracy, "top1_accuracy")
        _ = _require_finite(self.top3_accuracy, "top3_accuracy")
        _ = _require_finite(self.ece, "ece")
        _ = _require_finite(self.legal_uniform_nll, "legal_uniform_nll")
        _ = _require_finite(self.nll_vs_uniform_delta, "nll_vs_uniform_delta")
        if not 0 <= self.top1_accuracy <= 1:
            raise ContractError(f"top1_accuracy must be in [0,1], got {self.top1_accuracy!r}")
        if not 0 <= self.top3_accuracy <= 1:
            raise ContractError(f"top3_accuracy must be in [0,1], got {self.top3_accuracy!r}")
        if not 0 <= self.ece <= 1:
            raise ContractError(f"ece must be in [0,1], got {self.ece!r}")
        if not isinstance(self.count, int) or self.count <= 0:
            raise ContractError(f"count must be positive int, got {self.count!r}")
        if self.compile_mode not in COMPILE_ORDER and self.compile_mode != EAGER_ORACLE_ID:
            # Allow eager_fp32 alias.
            raise ContractError(
                f"compile_mode {self.compile_mode!r} not in {COMPILE_ORDER} nor {EAGER_ORACLE_ID!r}"
            )
        _bridge_contracts.make_digest_text(self.digest)


def compute_baseline_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    compile_mode: str = EAGER_ORACLE_ID,
    num_bins: int = 10,
) -> BaselineMetrics:
    """Compute the full baseline metric bundle deterministically."""
    nll = masked_cross_entropy(logits, targets, legal_mask)
    uniform_nll = legal_uniform_nll(targets, legal_mask)
    top1 = top_k_accuracy(logits, targets, legal_mask, k=1)
    top3 = top_k_accuracy(logits, targets, legal_mask, k=3)
    ece = expected_calibration_error(logits, targets, legal_mask, num_bins=num_bins)
    delta = uniform_nll - nll  # positive means model beats uniform
    count = logits.shape[0]
    # Stable digest over the numeric values (rounded to avoid float noise beyond 1e-9).
    payload = {
        "masked_nll": round(nll, 9),
        "top1_accuracy": round(top1, 9),
        "top3_accuracy": round(top3, 9),
        "ece": round(ece, 9),
        "legal_uniform_nll": round(uniform_nll, 9),
        "count": count,
        "compile_mode": compile_mode,
    }
    digest = of_canonical(payload)
    return BaselineMetrics(
        masked_nll=nll,
        top1_accuracy=top1,
        top3_accuracy=top3,
        ece=ece,
        legal_uniform_nll=uniform_nll,
        nll_vs_uniform_delta=delta,
        count=count,
        compile_mode=compile_mode,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Held-out split — disjoint train / held-out, no leakage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeldOutSplit:
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    seed: int
    held_out_ratio: float
    digest: DigestText

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ContractError(f"seed must be int, got {self.seed!r}")
        if not 0 < self.held_out_ratio < 1:
            raise ContractError(f"held_out_ratio must be in (0,1), got {self.held_out_ratio!r}")
        if len(self.train_ids) == 0 or len(self.held_out_ids) == 0:
            raise ContractError("both splits must be non-empty")
        _bridge_contracts.make_digest_text(self.digest)
        verify_held_out_disjoint(self)


def verify_held_out_disjoint(split: HeldOutSplit) -> None:
    """Fail closed when train and held-out share an id.

    Thin bridge delegate: ``hydra2._native.eval.record_verify_held_out_disjoint``
    decides detached over translator-staged id lists (owner
    ``hydra_search::eval::partition`` split membership); this function keeps
    the live-``HeldOutSplit`` shape, the ``ContractError`` shaping, and the
    ``__all__`` name. Extraction ``TypeError`` falls back to the oracle body.
    """
    try:
        from hydra2._native import eval as _eval_bridge  # pyrefly: ignore[missing-import]
    except ImportError:
        _eval_bridge = None  # type: ignore[assignment]  # reason: stale import keeps fallback contract
    leaf = (
        getattr(_eval_bridge, "record_verify_held_out_disjoint", None)
        if _eval_bridge is not None
        else None
    )
    if leaf is not None:
        try:
            leaf(list(split.train_ids), list(split.held_out_ids))
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except TypeError:
            pass
        else:
            return
    train = set(split.train_ids)
    held = set(split.held_out_ids)
    overlap = train & held
    if len(overlap) != 0:
        raise ContractError(f"held-out leakage: overlap {sorted(overlap)[:5]!r}")
    if len(train) + len(held) != len(train | held):
        raise ContractError("split sizes inconsistent (duplicate ids)")
    # Ensure every id appears exactly once across the two splits vs union — callers
    # provide the universe; we at least assert disjointness here. Coverage is
    # validated by the caller comparing to the original universe.


def split_held_out(
    all_ids: Sequence[str],
    *,
    held_out_ratio: float = 0.2,
    seed: int = 0,
) -> HeldOutSplit:
    """Deterministic held-out split: shuffle via seeded RNG, then slice.

    Held-out partition is NEVER exposed to training. Leakage fails closed.
    The split is deterministic in seed and held_out_ratio; identical inputs
    produce identical partition and digest.
    """
    if not isinstance(all_ids, Sequence) or len(all_ids) == 0:
        raise ContractError("all_ids must be non-empty sequence")
    if len(set(all_ids)) != len(all_ids):
        raise ContractError("all_ids must contain unique ids")
    if not 0 < held_out_ratio < 1:
        raise ContractError(f"held_out_ratio must be in (0,1), got {held_out_ratio!r}")
    if not isinstance(seed, int):
        raise ContractError(f"seed must be int, got {seed!r}")
    # Deterministic shuffle via torch Generator (counter-based, not call-order).
    n = len(all_ids)
    held_n = max(1, min(n - 1, round(n * held_out_ratio)))
    generator = torch.Generator().manual_seed(seed)
    perm: list[int] = torch.randperm(n, generator=generator).tolist()  # type: ignore[assignment]  # reason: tolist() statically Any; ints validated by construction
    # Held-out is the first held_n of permuted order (deterministic choice).
    ids: list[str] = list(all_ids)
    held = tuple(ids[i] for i in perm[:held_n])
    train = tuple(ids[i] for i in perm[held_n:])
    payload = {
        "all_ids_sorted": sorted(ids),
        "held_out_ids": sorted(held),
        "train_ids": sorted(train),
        "held_out_ratio": held_out_ratio,
        "seed": seed,
    }
    digest = of_canonical(payload)
    return HeldOutSplit(
        train_ids=train, held_out_ids=held, seed=seed, held_out_ratio=held_out_ratio, digest=digest
    )
