"""User-facing command options. No command reads data outside its explicit inputs."""

import argparse
from pathlib import Path


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def seed_value(value: str) -> int:
    n = int(value)
    if not 0 <= n < 2**64:
        raise argparse.ArgumentTypeError("must be an unsigned 64-bit integer")
    return n


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hydra3", description="Full-match Riichi policy learning")
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write a seeded, untrained challenger or control")
    init.add_argument("output", type=Path)
    init.add_argument("--scorer", choices=("plain", "edit"), default="edit")
    init.add_argument("--width", type=positive, default=128)
    init.add_argument("--depth", type=positive, default=4)
    init.add_argument("--heads", type=positive, default=4)
    init.add_argument("--ff", type=positive, default=384)
    init.add_argument("--seed", type=seed_value, default=1)

    ppo = commands.add_parser(
        "ppo",
        help="one or more full-match PPO generations",
        fromfile_prefix_chars="@",
        epilog=(
            "Pass @ppo.local.args to use a local profile (one argument per line). "
            "Options after the profile override its settings."
        ),
    )
    ppo.add_argument(
        "--games",
        type=positive,
        default=1024,
        help="full hanchan per generation (1024 measured fastest with the archive on the GPU)",
    )
    ppo.add_argument("--generations", type=positive, default=1)
    ppo.add_argument("--telemetry", action="store_true", help="report rollout and learner timings")
    ppo.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="include phase timings and sample CPU and GPU utilization on stderr every 100 ms",
    )
    ppo.add_argument("--attention", choices=("auto", "math", "flash", "cudnn"), default="auto")
    # Compile the learner, but replay the eager actor: those are different arithmetic
    # paths. The compiled actor was faster yet failed the trained-placement gate.
    # Requalify changes end to end and on held-out walls before changing these
    # defaults (python/hydra3/PERFORMANCE.md).
    ppo.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the fused CUDA AdamW kernel (default: on)",
    )
    ppo.add_argument(
        "--compile-learner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile CUDA PPO loss (default: on)",
    )
    ppo.add_argument(
        "--graph-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replay frequent CUDA actor shapes (default: on)",
    )
    ppo.add_argument(
        "--compile-actor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental compiled CUDA move scoring; faster but failed the held-out "
            "placement gate (default: off; overrides --graph-actor)"
        ),
    )
    ppo.add_argument(
        "--record", type=Path, help="write a tensor corpus of these self-play decisions"
    )
    bc = commands.add_parser("bc", help="imitate labeled actions from an explicit Hydra3 corpus")
    bc.add_argument("source", type=Path, help="authorized versioned actor-visible tensor corpus")
    for command in (bc, ppo):
        command.add_argument("initial", type=Path, help="initial or previous Hydra3 checkpoint")
        command.add_argument("output", type=Path)
        command.add_argument("--epochs", type=positive, default=1)
        # A 512-game generation holds about 540k decisions. Measured here, PPO
        # wall time per generation falls from 17.5 s at 128 to 8.2 s at 1024 and
        # rises again at 2048: the larger batch amortises per-update launch cost
        # over the same decisions. Smaller values remain available for the
        # gradient-noise profile a particular run may want.
        command.add_argument("--minibatch", type=positive, default=1024)
        command.add_argument("--lr", type=float, default=3e-4)

    arena = commands.add_parser(
        "eval", help="paired full matches; rotate the candidate through all seats"
    )
    arena.add_argument("candidate", type=Path)
    arena.add_argument("opponent", type=Path)
    arena.add_argument(
        "--blocks", type=positive, default=4, help="each block contains four hanchan"
    )
    arena.add_argument(
        "--cohort", type=positive, default=32, help="maximum simultaneous wall blocks"
    )
    for command in (bc, ppo, arena):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--seed", type=seed_value, default=1)
        command.add_argument(
            "--torch-threads", type=positive, default=1, help="Torch CPU intra-op threads"
        )
    return root
