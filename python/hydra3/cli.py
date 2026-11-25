"""Run isolated self-play, supervised learning or matched evaluation."""

import json
import math
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns

from .arguments import parser
from .hardware_telemetry import sample_hardware


def _record_path(path: Path, generation: int, generations: int) -> Path:
    if generations == 1:
        return path
    return path.with_name(f"{path.stem}.generation-{generation}{path.suffix}")


def main() -> None:
    args = parser().parse_args()
    import torch

    from .checkpoint import load_checkpoint, save_checkpoint
    from .model import ModelConfig, Policy
    from .runtime import derive_seed

    if args.command == "init":
        torch.manual_seed(derive_seed(args.seed, "model/init", 0))
        config = ModelConfig(
            width=args.width, depth=args.depth, heads=args.heads, ff=args.ff, scorer=args.scorer
        )
        model = Policy(config)
        save_checkpoint(args.output, model)
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "trained": False,
                }
            )
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Pass --device cpu for an explicit CPU run.")
    torch.set_num_threads(args.torch_threads)
    if args.command == "eval":
        from .evaluate import evaluate

        candidate, _ = load_checkpoint(args.candidate, device)
        opponent, _ = load_checkpoint(args.opponent, device)
        result = evaluate(
            candidate,
            opponent,
            blocks=args.blocks,
            seed=args.seed,
            device=device,
            cohort=args.cohort,
        )
        print(json.dumps(result, allow_nan=False))
        return

    from .learn import bc, ppo

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise SystemExit("--lr must be positive and finite")
    model, state = load_checkpoint(args.initial, device)
    fused = getattr(args, "fused_adamw", False)
    if fused and device.type != "cuda":
        raise SystemExit("--fused-adamw requires CUDA")
    if device.type != "cuda" and (
        getattr(args, "compile_learner", False)
        or getattr(args, "graph_actor", False)
        or getattr(args, "compile_actor", False)
    ):
        raise SystemExit("compiled learning and CUDA actor scoring require CUDA")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=fused)
    if state.get("optimizer") is not None:
        saved_optimizer = state["optimizer"]
        for group in saved_optimizer["param_groups"]:
            group["lr"] = args.lr
            group["fused"] = fused
            if fused:
                group["foreach"] = False
        optimizer.load_state_dict(saved_optimizer)
        del saved_optimizer
    start = state["updates"]
    del state
    verbose = args.command == "ppo" and args.verbose
    gpu_uuid = (
        str(torch.cuda.get_device_properties(device).uuid)
        if verbose and device.type == "cuda"
        else None
    )
    for i in range(getattr(args, "generations", 1)):
        generation = start + i
        seed = derive_seed(args.seed, f"{args.command}/generation", generation)
        common = {
            "device": device,
            "epochs": args.epochs,
            "minibatch": args.minibatch,
            "seed": seed,
        }
        telemetry: dict[str, int] | None = (
            {} if verbose or getattr(args, "telemetry", False) else None
        )
        if telemetry is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.command == "bc":
            result = bc(model, optimizer, source=args.source, **common)
        else:
            record = (
                _record_path(args.record, generation, args.generations) if args.record else None
            )
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backends = {
                "math": SDPBackend.MATH,
                "flash": SDPBackend.FLASH_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION,
            }
            attention = (
                nullcontext() if args.attention == "auto" else sdpa_kernel(backends[args.attention])
            )
            hardware = sample_hardware(generation, gpu_uuid) if verbose else nullcontext()
            with hardware, attention:
                result = ppo(
                    model,
                    optimizer,
                    games=args.games,
                    record=record,
                    telemetry=telemetry,
                    compile_learner=args.compile_learner,
                    graph_actor=args.graph_actor,
                    compile_actor=args.compile_actor,
                    **common,
                )
        start_save = perf_counter_ns() if telemetry is not None else 0
        save_checkpoint(args.output, model, optimizer, updates=generation + 1)
        if telemetry is not None:
            telemetry["checkpoint_ns"] = perf_counter_ns() - start_save
        print(
            json.dumps(
                {
                    "generation": generation,
                    "checkpoint": str(args.output),
                    **result,
                    **({"telemetry": telemetry} if telemetry is not None else {}),
                },
                allow_nan=False,
            ),
            flush=True,
        )
