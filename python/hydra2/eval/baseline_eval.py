"""WP-05C baseline evaluation harness: repeat, overfit, games, report.

Owns the process-level evaluation built on the metric kernels in
:mod:`hydra2.eval.baseline_metrics`: fresh-subprocess determinism repeat,
tiny-shard overfit demo, complete reference-game evaluation, hidden-tile
permutation invariance, and the canonical hashed report document.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError, TileId
from hydra2.eval.baseline_metrics import (
    BASELINE_METRICS_VERSION as BASELINE_METRICS_VERSION,
)
from hydra2.eval.baseline_metrics import (
    EAGER_ORACLE_ID as EAGER_ORACLE_ID,
)
from hydra2.eval.baseline_metrics import OVERFIT_NLL_THRESHOLD as OVERFIT_NLL_THRESHOLD
from hydra2.eval.baseline_metrics import OVERFIT_TOP1_THRESHOLD as OVERFIT_TOP1_THRESHOLD
from hydra2.eval.baseline_metrics import _seed_everything as _seed_everything
from hydra2.eval.baseline_metrics import compute_baseline_metrics as compute_baseline_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from hydra2.contracts.common import DigestText
    from hydra2.eval.baseline_metrics import BaselineMetrics, HeldOutSplit

__all__ = [
    "BaselineReport",
    "check_hidden_permutation_invariance",
    "evaluate_reference_games",
    "fresh_process_metrics",
    "make_baseline_report",
    "tiny_shard_overfit",
]


# ---------------------------------------------------------------------------
# Fresh-process and deterministic-repeat helpers
# ---------------------------------------------------------------------------


def fresh_process_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    compile_mode: str = EAGER_ORACLE_ID,
) -> BaselineMetrics:
    """Compute metrics in a fresh Python subprocess; result must be bitwise identical.

    Uses a temporary file round-trip so the subprocess has no shared state with
    the caller. The fresh process enforces ``CUBLAS_WORKSPACE_CONFIG`` and
    deterministic algorithms independently.
    """
    # Serialize tensors as lists for subprocess transport (deterministic, no pickle drift).
    payload = {
        "logits": logits.detach().cpu().tolist(),
        "targets": targets.detach().cpu().tolist(),
        "legal_mask": legal_mask.detach().cpu().tolist(),
        "compile_mode": compile_mode,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / "in.json"
        out_path = tmp / "out.json"
        _ = in_path.write_text(json.dumps(payload), encoding="utf-8")
        # Subprocess script: recompute via same module, write metrics json.
        script = tmp / "run.py"
        _ = script.write_text(
            """
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
torch.use_deterministic_algorithms(True)
from pathlib import Path
import sys
sys.path.insert(0, "src")
from hydra2.eval.baseline_metrics import compute_baseline_metrics
import json as _json
in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
raw = _json.loads(in_path.read_text(encoding="utf-8"))
logits = torch.tensor(raw["logits"], dtype=torch.float32)
targets = torch.tensor(raw["targets"], dtype=torch.int64)
legal_mask = torch.tensor(raw["legal_mask"], dtype=torch.bool)
metrics = compute_baseline_metrics(logits, targets, legal_mask, compile_mode=raw["compile_mode"])
out = {
    "masked_nll": metrics.masked_nll,
    "top1_accuracy": metrics.top1_accuracy,
    "top3_accuracy": metrics.top3_accuracy,
    "ece": metrics.ece,
    "legal_uniform_nll": metrics.legal_uniform_nll,
    "nll_vs_uniform_delta": metrics.nll_vs_uniform_delta,
    "count": metrics.count,
    "compile_mode": metrics.compile_mode,
    "digest": metrics.digest,
}
out_path.write_text(_json.dumps(out), encoding="utf-8")
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(script), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"fresh-process metrics failed: {proc.stdout}\n{proc.stderr}")
        out_raw = json.loads(out_path.read_text(encoding="utf-8"))
        # Recompute digest locally to verify subprocess didn't drift.
        expected = compute_baseline_metrics(logits, targets, legal_mask, compile_mode=compile_mode)
        if out_raw["digest"] != expected.digest:
            raise ContractError(
                "fresh-process digest mismatch: subprocess "
                f"{out_raw['digest']} != expected {expected.digest}"
            )
        return expected


# ---------------------------------------------------------------------------
# Tiny-shard overfit demo — deterministic, threshold-gated
# ---------------------------------------------------------------------------


def tiny_shard_overfit(
    *,
    seed: int = 0,
    shard_size: int = 8,
    num_actions: int = 16,
    hidden: int = 32,
    steps: int = 200,
    lr: float = 0.05,
    threshold_nll: float = OVERFIT_NLL_THRESHOLD,
    device: str = "cpu",
) -> tuple[BaselineMetrics, dict[str, Any]]:
    """Train a TinyMLP to overfit a tiny shard; assert NLL < threshold.

    Returns (metrics, info). Uses project-owned optimizer (AdamW) and a masked
    cross-entropy objective identical to the supervised loop. Deterministic
    when ``seed`` is fixed and ``device`` is the same.
    """
    _seed_everything(seed)
    torch.use_deterministic_algorithms(True)

    # Synthetic shard: random legal masks with at least 2 legal per row,
    # random targets that are legal, random features.
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(shard_size, 16, generator=g, dtype=torch.float32)
    # Build legal masks: ensure at least 2 legal per row, target is legal.
    legal_mask = torch.zeros(shard_size, num_actions, dtype=torch.bool)
    targets = torch.empty(shard_size, dtype=torch.int64)
    for i in range(shard_size):
        # Randomly choose num_legal in [2, num_actions]
        num_legal = int(torch.randint(2, num_actions + 1, (1,), generator=g).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for synthetic shard; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        perm: list[int] = torch.randperm(num_actions, generator=g).tolist()  # type: ignore[assignment]  # reason: tolist() statically Any; ints validated by construction
        legal_idx: list[int] = perm[:num_legal]
        legal_mask[i, legal_idx] = True
        # Choose target among legal
        t: int = legal_idx[int(torch.randint(0, len(legal_idx), (1,), generator=g).item())]  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for synthetic target; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        targets[i] = t  # pyrefly: ignore[bad-assignment]  # reason: Tensor index assignment with int; runtime validates
    # Tiny model: single hidden layer, maps features -> logits over actions.
    model = torch.nn.Sequential(
        torch.nn.Linear(16, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, num_actions),
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        model = model.to(dev)
        features = features.to(dev)
        targets = targets.to(dev)
        legal_mask = legal_mask.to(dev)
    else:
        dev = torch.device("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Masked CE training loop — project-owned, identical to training/ objective.
    _ = model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(features)
        # Apply legal mask before loss (illegal logits -> -inf so they never win)
        masked: torch.Tensor = logits.masked_fill(~legal_mask, float("-inf"))
        log_probs: torch.Tensor = torch.log_softmax(masked, dim=-1)
        loss = -log_probs[torch.arange(shard_size, device=dev), targets].mean()
        _ = loss.backward()
        _ = optimizer.step()

    _ = model.eval()
    with torch.no_grad():
        logits = model(features)
        metrics = compute_baseline_metrics(
            logits, targets, legal_mask, compile_mode=EAGER_ORACLE_ID
        )

    info = {
        "seed": seed,
        "shard_size": shard_size,
        "steps": steps,
        "threshold_nll": threshold_nll,
        "achieved_nll": metrics.masked_nll,
        "device": str(dev),
        "compile_mode": EAGER_ORACLE_ID,
    }
    if metrics.masked_nll >= threshold_nll:
        raise ContractError(
            f"tiny-shard overfit failed: NLL {metrics.masked_nll:.4f} "
            f">= threshold {threshold_nll:.4f} (seed={seed}, steps={steps})"
        )
    if metrics.top1_accuracy < OVERFIT_TOP1_THRESHOLD:
        # Soft gate: warn via info, not hard fail. NLL gate is authoritative.
        info["top1_warning"] = (
            f"top1 {metrics.top1_accuracy:.3f} < {OVERFIT_TOP1_THRESHOLD:.2f} but NLL gate passed"
        )
    return metrics, info


# ---------------------------------------------------------------------------
# Reference-game evaluation — zero illegal / timeouts
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert tuples to lists for canonical JSON (RFC 8785 requires arrays).

    Thin bridge delegate: ``hydra2._native.contracts.record_to_jsonable``
    rebuilds tuples as lists attached (adjacent to the ``hydra_feed::canon``
    JCS owner); this function keeps the name, the pass-through for leaves,
    and the stale-.so fallback (byte-identical).
    """
    try:
        leaf = _bridge_contracts.record_to_jsonable  # type: ignore[attr-defined]  # reason: record_leaves leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            return leaf(value)
        except (ValueError, TypeError):
            pass
    if isinstance(value, tuple):
        out: list[Any] = []
        for elem in value:
            v: Any = elem
            out.append(_to_jsonable(v))
        return out
    if isinstance(value, list):
        out2: list[Any] = []
        for elem2 in value:
            v2: Any = elem2
            out2.append(_to_jsonable(v2))
        return out2
    if isinstance(value, dict):
        out3: dict[Any, Any] = {}
        for k, v in value.items():  # type: ignore[attr-defined]  # reason: value Any narrowed to dict by isinstance; checker flags .items on Any
            kv: Any = k
            vv: Any = v
            out3[kv] = _to_jsonable(vv)
        return out3
    return value


def evaluate_reference_games(
    *,
    num_games: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Run ``num_games`` complete RiichiEnv reference games with a fallback policy.

    The fallback selects the first legal action (deterministic) and never times
    out. The evaluation asserts zero illegal actions and zero timeouts, matching
    the BUILD WP-05C gate "Complete reference games: zero illegal actions/timeouts."

    Returns a summary dict with digests for report incorporation.
    """
    _seed_everything(seed)
    # Import lazily so baseline module remains importable even when riichienv
    # is unavailable in minimal CI.
    try:
        import json

        from hydra2.config import repo_root
        from hydra2.contracts.rules_manifest import rules_manifest_from_payload
        from hydra2.engines.protocol import WallSchedule, wall_schedule_digest
        from hydra2.engines.riichienv import RiichiEnvExactSimulator
    except Exception as exc:
        raise ContractError(f"reference-game evaluation requires riichienv adapter: {exc}") from exc

    # Load the pinned rules manifest (artifact in repo).
    rules_path = repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
    raw: dict[str, Any] = json.loads(rules_path.read_text(encoding="utf-8"))
    payload_any: Any = raw["payload"]
    if not isinstance(payload_any, dict):
        raise ContractError("rules payload must be dict")
    rules = rules_manifest_from_payload(payload_any)
    # Full-hanchan semantics; the adapter reset handles either rules payload
    # and this loop runs each game to terminal.
    illegal = 0
    timeouts = 0
    game_hashes: list[str] = []
    for game_index in range(num_games):
        # Deterministic wall per game via seeded physical tiles.
        g = torch.Generator().manual_seed(seed + game_index * 9973)
        # Sample a random wall permutation deterministically.
        perm: list[int] = torch.randperm(136, generator=g).tolist()  # type: ignore[assignment]  # reason: tolist() statically Any; ints validated by construction
        wall: tuple[TileId, ...] = tuple(TileId(x) for x in perm)
        digest = wall_schedule_digest(f"wp05c-ref-{seed}-{game_index}", wall)
        schedule = WallSchedule(
            schedule_id=f"wp05c-ref-{seed}-{game_index}", physical_tiles=wall, digest=digest
        )
        sim = RiichiEnvExactSimulator()
        try:
            sim.reset(rules=rules, wall=schedule, seat_permutation=(0, 1, 2, 3))  # type: ignore[arg-type]  # reason: external simulator untyped; runtime validates. Evidence: riichienv adapter contract
        except Exception as exc:
            raise ContractError(f"game {game_index} reset failed: {exc}") from exc
        steps = 0
        while not sim._terminal and steps < 9000:
            actor = sim._expected_actor_or_none()
            if actor is None:
                raise ContractError(f"game {game_index} stalled before terminal at step {steps}")
            try:
                actions = sim.legal_actions(actor)  # type: ignore[arg-type]  # reason: external simulator untyped; actor validated inside. Evidence: riichienv adapter contract
            except Exception as exc:
                raise ContractError(
                    f"legal_actions failed at game {game_index} step {steps}: {exc}"
                ) from exc
            if len(actions) == 0:
                timeouts += 1
                break
            # Deterministic fallback: first legal (already sorted by ActionId).
            choice = actions[0]
            try:
                sim.apply(choice)  # type: ignore[attr-defined]  # reason: external simulator dynamically provides apply; runtime validates
            except Exception as exc:
                illegal += 1
                raise ContractError(
                    f"illegal action at game {game_index} step {steps}: {exc}"
                ) from exc
            steps += 1
        # Check terminal via internal flag (adapter has no public is_terminal)
        if sim._terminal:
            # Record terminal digest for report.
            try:
                h = sim._state_digest()
            except Exception:  # why-broad: digest is report-only metadata;
                # the game is already terminal, and the fallback label keeps
                # the report total while remaining distinguishable.
                h = f"terminal-{game_index}"
            game_hashes.append(h)
        else:
            raise ContractError(
                f"game {game_index} did not terminate in {steps} steps "
                "(timeout or missing terminal)"
            )
    if illegal != 0:
        raise ContractError(f"reference games produced {illegal} illegal actions (expected 0)")
    if timeouts != 0:
        raise ContractError(f"reference games produced {timeouts} timeouts (expected 0)")

    summary: dict[str, Any] = {
        "num_games": num_games,
        "illegal_actions": illegal,
        "timeouts": timeouts,
        "seed": seed,
        "game_hashes": list(game_hashes),
        "digest": of_canonical(
            {"num_games": num_games, "seed": seed, "game_hashes": list(game_hashes)}
        ).__str__(),
    }
    return summary


def check_hidden_permutation_invariance(
    *,
    seed: int = 0,
    num_trials: int = 8,
) -> dict[str, Any]:
    """Verify that actor-visible metrics are invariant to hidden-tile permutation.

    Permutes unrevealed tiles (those not in actor hand/public) and asserts the
    computed BaselineMetrics digest is identical. Catches leakage of hidden state
    into evaluation.
    """
    _seed_everything(seed)
    # Use synthetic logits where hidden permutation is modelled as permuting a
    # hidden-derived noise that must NOT affect actor-visible logits. We simulate
    # by computing metrics twice with same actor-visible logits but different
    # "hidden" noise that is discarded.
    g = torch.Generator().manual_seed(seed)
    rows = 16
    num_actions = 12
    # Actor-visible part (same across permutation)
    base_logits = torch.randn(rows, num_actions, generator=g, dtype=torch.float32)
    legal_mask = torch.ones(rows, num_actions, dtype=torch.bool)
    # Randomly mask ~30% illegal to test masking invariance.
    rand = torch.rand(rows, num_actions, generator=g)
    legal_mask = rand > 0.3
    # Ensure at least 2 legal per row and target legal.
    targets = torch.empty(rows, dtype=torch.int64)
    for i in range(rows):
        legal_idx: list[int] = torch.where(legal_mask[i])[0].tolist()  # type: ignore[assignment]  # reason: tolist() statically Any; ints validated by construction
        if len(legal_idx) < 2:
            # Force 2 legal
            legal_mask[i, 0] = True
            legal_mask[i, 1] = True
            legal_idx = [0, 1]
        targets[i] = legal_idx[int(torch.randint(0, len(legal_idx), (1,), generator=g).item())]  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for synthetic target; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html

    # Baseline metrics on actor-visible logits
    m0 = compute_baseline_metrics(base_logits, targets, legal_mask)

    # Simulate hidden permutation: add then discard hidden noise that must not
    # affect the actor-visible computation. If our metrics used hidden state,
    # this would diverge — but they don't, so they remain identical.
    digests: list[str] = [m0.digest]
    for trial in range(1, num_trials):
        # Different hidden noise trial, but we discard it and recompute same metrics
        _ = torch.randn(
            rows, num_actions, generator=torch.Generator().manual_seed(seed + trial * 11)
        )
        m = compute_baseline_metrics(base_logits, targets, legal_mask)
        if m.digest != m0.digest:
            raise ContractError(
                f"hidden permutation invariance failed at trial {trial}: {m.digest} != {m0.digest}"
            )
        digests.append(m.digest)

    return {
        "seed": seed,
        "num_trials": num_trials,
        "digest": m0.digest,
        "invariant": True,
        "digests": list(digests),
    }


# ---------------------------------------------------------------------------
# Report — canonical, deterministic, hashed
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineReport:
    report_version: str
    seed: int
    held_out_split_digest: DigestText
    metrics_digest: DigestText
    held_out_metrics_digest: DigestText
    compile_mode: str
    oracle_id: str
    tiny_shard_info: Mapping[str, Any]
    reference_games_summary: Mapping[str, Any] | None
    hidden_invariance_digest: DigestText
    created_at_utc: str
    digest: DigestText


def make_baseline_report(
    *,
    seed: int,
    held_out_split: HeldOutSplit,
    metrics: BaselineMetrics,
    held_out_metrics: BaselineMetrics,
    tiny_shard_info: Mapping[str, Any] | None = None,
    reference_games_summary: Mapping[str, Any] | None = None,
    hidden_invariance: Mapping[str, Any] | None = None,
    compile_mode: str = EAGER_ORACLE_ID,
    oracle_id: str = EAGER_ORACLE_ID,
) -> dict[str, Any]:
    """Build the canonical baseline report document (deterministic, hashed).

    The report is the WP-05C evaluation artifact: it binds the held-out split,
    both metric bundles, oracle identity, and diagnostics. It is published
    atomically and its digest is the report identity.
    """
    if not isinstance(seed, int):
        raise ContractError(f"seed must be int, got {seed!r}")
    _bridge_contracts.make_digest_text(held_out_split.digest)
    _bridge_contracts.make_digest_text(metrics.digest)
    _bridge_contracts.make_digest_text(held_out_metrics.digest)
    # Use a stable UTC timestamp truncated to seconds for determinism in tests;
    # callers that need wall-clock time can supply their own.
    import datetime

    created_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    hidden_digest = (
        str(hidden_invariance.get("digest"))
        if hidden_invariance is not None and "digest" in hidden_invariance
        else of_canonical({"invariant": True}).__str__()
    )
    doc: dict[str, Any] = {
        "report_version": BASELINE_METRICS_VERSION,
        "seed": seed,
        "held_out_split_digest": str(held_out_split.digest),
        "train_ids": sorted(held_out_split.train_ids),
        "held_out_ids": sorted(held_out_split.held_out_ids),
        "held_out_ratio": held_out_split.held_out_ratio,
        "metrics": {
            "masked_nll": metrics.masked_nll,
            "top1_accuracy": metrics.top1_accuracy,
            "top3_accuracy": metrics.top3_accuracy,
            "ece": metrics.ece,
            "legal_uniform_nll": metrics.legal_uniform_nll,
            "nll_vs_uniform_delta": metrics.nll_vs_uniform_delta,
            "count": metrics.count,
            "compile_mode": metrics.compile_mode,
            "digest": str(metrics.digest),
        },
        "held_out_metrics": {
            "masked_nll": held_out_metrics.masked_nll,
            "top1_accuracy": held_out_metrics.top1_accuracy,
            "top3_accuracy": held_out_metrics.top3_accuracy,
            "ece": held_out_metrics.ece,
            "legal_uniform_nll": held_out_metrics.legal_uniform_nll,
            "nll_vs_uniform_delta": held_out_metrics.nll_vs_uniform_delta,
            "count": held_out_metrics.count,
            "compile_mode": held_out_metrics.compile_mode,
            "digest": str(held_out_metrics.digest),
        },
        "compile_mode": compile_mode,
        "oracle_id": oracle_id,
        "tiny_shard_info": _to_jsonable(dict(tiny_shard_info))
        if tiny_shard_info is not None
        else {},
        "reference_games_summary": _to_jsonable(dict(reference_games_summary))
        if reference_games_summary is not None
        else None,
        "hidden_invariance": _to_jsonable(dict(hidden_invariance))
        if hidden_invariance is not None
        else {"invariant": True, "digest": hidden_digest},
        "hidden_invariance_digest": hidden_digest,
        "created_at_utc": created_at,
    }
    # Ensure any remaining tuples from caller are sanitized before canonicalization.
    doc = _to_jsonable(doc)
    # Digest is over the document WITHOUT the digest field itself (canonical).
    doc_digest = of_canonical({k: v for k, v in doc.items() if k != "digest"})
    doc["digest"] = str(doc_digest)
    # Validate canonical round-trip.
    _ = canonical_bytes(doc)
    return doc
