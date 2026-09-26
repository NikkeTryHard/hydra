"""Offline holdout observer: bigger fixed eval shard over landed checkpoints.

Observer-only: reads checkpoints, the stored manifest artifact, and the live
eval.jsonl for a parity gate; writes ONLY the --out dir (default
<artifact-root>/offline-eval/<run-id>/, never inside the run dir). The live
run is untouched: its eval cadence is digest-bound, so this script is how a
larger shard gets measured without restarting training.

Data path mirrors the live holdout exactly (GameStream seed=data_seed,
epoch=0, shuffle 0 + _expand_game_planes with ContractError skip +
assemble_slim_batch + the same privileged gate + _model_forward), with a
larger fixed prefix (default 64 batches x micro rows). The first 10 val
batches must reproduce the live eval.jsonl row within tolerance (parity
gate, else fail-closed: pipeline bug, not a metric). CPU-only: the GPU
stays with the trainer (nice the process, clamp threads).

Reports per checkpoint: pooled NLL/top-k with standard errors, pooled
10-bin ECE (same frozen-bin convention as the training report), per-type
scorecards with SE and low-support flags, legal-uniform gap, games
touched, and a same-size train-split shard so the val-train gap (the
overfit read) is apples-to-apples (both eval mode, no dropout).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

from hydra2.contracts.common import ContractError
from hydra2.data.stream_iter import GameStream
from hydra2.data.stream_manifest import build_manifest
from hydra2.training.loop_batch import (
    _batch_action_kinds,
    _model_forward,
    _validate_batch_no_privileged,
)
from hydra2.training.objectives_metrics import compute_metrics, row_eval_primitives
from hydra2.training.rust_batch import assemble_slim_batch
from hydra2.training.stream_expand import SPLIT_RATIOS
from hydra2.training.stream_expand import _action_kind_for_id as _action_kind_for_id
from hydra2.training.stream_expand import _expand_game_planes as _expand_game_planes

#: Live-eval prefix width reproduced for the parity gate (must stay 10:
#: the live report is mean-of-batch-means over eval.num_batches=10).
_PARITY_BATCHES = 10
#: Loose parity bound: CPU fp32 forwards vs live bf16-autocast forwards.
_PARITY_TOL = 0.01
#: Frozen ECE bins, same convention as the training report.
_ECE_BINS = 10
#: Same low-support threshold as the training per-type scorecards.
_MIN_SUPPORT = 30


def _mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def _pooled_ece(conf: list[float], hit: list[float]) -> float:
    ece = 0.0
    n = len(conf)
    for b in range(_ECE_BINS):
        lo, hi = b / _ECE_BINS, (b + 1) / _ECE_BINS
        idx = [
            i
            for i, c in enumerate(conf)
            if (c >= lo and c < hi) or (b == _ECE_BINS - 1 and c == hi)
        ]
        if idx:
            acc = sum(hit[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            ece += (len(idx) / n) * abs(acc - avg)
    return ece


def _build_model(action_count: int, arch_id: str, params: dict) -> torch.nn.Module:
    from hydra2.models.model import _ARCH_DEFAULTS, Hydra2BaselineModel
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if arch_id not in _ARCH_DEFAULTS:
        raise ContractError(f"unknown architecture_id {arch_id!r}")
    if action_count != BASELINE_ACTION_COUNT:
        raise ContractError(f"action_count {action_count} != baseline {BASELINE_ACTION_COUNT}")
    dims = dict(_ARCH_DEFAULTS[arch_id])
    dims.update(params or {})
    return Hydra2BaselineModel(
        action_count=action_count,
        architecture_id=arch_id,
        d_model=int(dims["d_model"]),
        n_layers=int(dims["n_layers"]),
        n_heads=int(dims["n_heads"]),
        d_ff=int(dims["d_ff"]),
        dropout=float(dims["dropout"]),
    )


def _collect_shard(
    *,
    model: torch.nn.Module,
    manifest,
    seed: int,
    ratios: dict[str, float],
    split: str,
    need_rows: int,
    micro: int,
    action_count: int,
    pack_histories: bool,
    parity_rows: dict[str, list[float]] | None,
) -> dict:
    stream = GameStream(manifest, seed=seed, ratios=ratios, epoch=0, split=split, shuffle_buffer=0)
    rows: list[dict] = []
    games = 0
    game_ids: set[str] = set()
    t0 = time.perf_counter()
    for streamed in stream:
        if len(rows) >= need_rows:
            break
        games += 1
        game_ids.add(f"{streamed.path}")
        try:
            row_dicts, _ = _expand_game_planes(streamed.game, streamed.split, streamed.raw)
        except ContractError:
            continue
        for row_dict in row_dicts:
            rows.append(row_dict)
            if len(rows) >= need_rows:
                break
    rows = rows[:need_rows]
    nll_all: list[float] = []
    hit1_all: list[float] = []
    hit3_all: list[float] = []
    hit5_all: list[float] = []
    conf_all: list[float] = []
    kinds_all: list[str] = []
    log_counts: list[float] = []
    parity_nll: list[float] = []
    parity_top1: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), micro):
            chunk = rows[start : start + micro]
            batch = assemble_slim_batch(
                chunk, action_count=action_count, pack_histories=pack_histories
            )
            _validate_batch_no_privileged(batch)
            out = _model_forward(model, batch)
            logits = out["policy_logits"]
            targets = batch["chosen_action_id"]
            mask = batch["legal_mask"]
            prim = row_eval_primitives(logits, targets, mask)
            nll_all.extend(float(v) for v in prim["row_nll"].tolist())
            hit1_all.extend(float(v) for v in prim["hit1"].tolist())
            hit3_all.extend(float(v) for v in prim["hit3"].tolist())
            hit5_all.extend(float(v) for v in prim["hit5"].tolist())
            conf_all.extend(float(v) for v in prim["conf"].tolist())
            kinds = _batch_action_kinds(batch, targets)
            if kinds is None:
                kinds = [_action_kind_for_id(int(t)) for t in targets.tolist()]
            kinds_all.extend(kinds)
            log_counts.extend(float(v) for v in mask.sum(dim=1).float().log().tolist())
            if parity_rows is not None and start // micro < _PARITY_BATCHES:
                m = compute_metrics(logits, targets, mask)
                parity_nll.append(m["masked_nll"])
                parity_top1.append(m["top1"])
    if parity_rows is not None:
        parity_rows["nll"] = parity_nll
        parity_rows["top1"] = parity_top1
    per_type: dict[str, dict] = {}
    for kind in sorted(set(kinds_all)):
        idx = [i for i, k in enumerate(kinds_all) if k == kind]
        kn = [nll_all[i] for i in idx]
        kh = [hit1_all[i] for i in idx]
        mean, se = _mean_se(kn)
        tmean, _ = _mean_se(kh)
        per_type[kind] = {
            "n": len(idx),
            "nll": mean,
            "nll_se": se,
            "top1": tmean,
            "low_support": len(idx) < _MIN_SUPPORT,
        }
    nll, nll_se = _mean_se(nll_all)
    top1, top1_se = _mean_se(hit1_all)
    top3, _ = _mean_se(hit3_all)
    top5, _ = _mean_se(hit5_all)
    uniform = sum(log_counts) / len(log_counts)
    return {
        "rows": len(nll_all),
        "games": games,
        "distinct_games": len(game_ids),
        "nll": nll,
        "nll_se": nll_se,
        "top1": top1,
        "top1_se": top1_se,
        "top3": top3,
        "top5": top5,
        "ece_pooled": _pooled_ece(conf_all, hit1_all),
        "uniform_gap": uniform - nll,
        "per_type": per_type,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline holdout observer over checkpoints.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--updates", nargs="*", type=int, default=None)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(8)
    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "run.yaml").read_text())
    run_id = cfg["run"]["id"]
    out_dir = Path(args.out or (run_dir.parent.parent / "offline-eval" / run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_seed = int(cfg["seeds"]["data_seed"])
    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]
    ratios = {train_split: SPLIT_RATIOS["train"], val_split: SPLIT_RATIOS["validation"]}
    micro = int(cfg["eval"]["microbatch_size"])
    action_count = int(cfg["model"]["action_count"])
    pack = bool(cfg["loop"].get("pack_histories", False))
    need_rows = args.batches * micro

    live_evals: dict[int, dict] = {}
    eval_path = run_dir / "eval" / "eval.jsonl"
    if eval_path.is_file():
        for line in eval_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                live_evals[int(row["update"])] = row
    print("observer: manifest build (warm artifact expected) ...", flush=True)
    manifest = build_manifest([(r[0], r[1]) for r in cfg["data"]["roots"]])
    print(f"observer: manifest files={len(manifest)}", flush=True)

    model = _build_model(
        action_count, cfg["model"]["architecture_id"], cfg["model"].get("parameters", {})
    )

    ckpt_dir = run_dir / "checkpoints"
    landed = sorted(int(p.stem.rsplit("-", 1)[1]) for p in ckpt_dir.glob("checkpoint-*.pt"))
    updates = args.updates if args.updates else landed
    print(f"observer: updates={updates} batches={args.batches} rows={need_rows}", flush=True)

    out_path = out_dir / "offline-eval.jsonl"
    for update in updates:
        ckpt = ckpt_dir / f"checkpoint-{update:06d}.pt"
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        inner = payload["payload"] if "payload" in payload else payload
        model.load_state_dict(inner["model_state"])
        entry: dict = {"update": update, "shard_batches": args.batches, "micro": micro}
        for split in (val_split, train_split):
            parity: dict[str, list[float]] | None = {} if split == val_split else None
            shard = _collect_shard(
                model=model,
                manifest=manifest,
                seed=data_seed,
                ratios=ratios,
                split=split,
                need_rows=need_rows,
                micro=micro,
                action_count=action_count,
                pack_histories=pack,
                parity_rows=parity,
            )
            entry[split] = shard
            if parity is not None and update in live_evals:
                live = live_evals[update]
                dn = abs(sum(parity["nll"]) / len(parity["nll"]) - live["masked_nll"])
                dt = abs(sum(parity["top1"]) / len(parity["top1"]) - live["top1"])
                entry["parity"] = {"d_nll": dn, "d_top1": dt}
                if dn > _PARITY_TOL or dt > _PARITY_TOL:
                    raise ContractError(
                        f"parity failed at update={update}: d_nll={dn:.4f} d_top1={dt:.4f}"
                    )
        v, t = entry[val_split], entry[train_split]
        gap = v["nll"] - t["nll"]
        gap_se = math.sqrt(v["nll_se"] ** 2 + t["nll_se"] ** 2)
        entry["gap"] = {"val_minus_train_nll": gap, "se": gap_se}
        with out_path.open("a") as h:
            h.write(json.dumps(entry, sort_keys=True) + "\n")
        d = v["per_type"].get("discard", {})
        d_nll = d.get("nll", float("nan"))
        d_se = d.get("nll_se", float("nan"))
        print(
            f"observer: update={update:06d} val_nll={v['nll']:.4f}±{v['nll_se']:.4f} "
            f"discard={d_nll:.4f}±{d_se:.4f}(n={d.get('n', 0)}) "
            f"train_nll={t['nll']:.4f}±{t['nll_se']:.4f} gap={gap:+.4f}±{gap_se:.4f} "
            f"games={v['games']} wall_s={v['wall_s']}",
            flush=True,
        )
    print(f"observer: wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
