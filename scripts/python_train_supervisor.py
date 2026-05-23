#!/usr/bin/env python3
"""Run Python learner and TensorBoard as one signal-managed process group."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-pid-file", type=Path, required=True)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    parser.add_argument("--tensorboard-host", required=True)
    parser.add_argument("--tensorboard-port", type=int, required=True)
    parser.add_argument("--tensorboard-log", type=Path, required=True)
    parser.add_argument("--", dest="separator", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("supervisor requires learner command after --")
    return args


def terminate(child: subprocess.Popen[bytes] | None) -> None:
    if child is None or child.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main() -> int:
    args = parse_args()
    args.tensorboard_pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.parent.mkdir(parents=True, exist_ok=True)
    tb_log = args.tensorboard_log.open("ab", buffering=0)
    tb = subprocess.Popen(
        [
            "pixi",
            "run",
            "-e",
            "py-train",
            "tensorboard",
            "--logdir",
            str(args.tensorboard_logdir),
            "--host",
            args.tensorboard_host,
            "--port",
            str(args.tensorboard_port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=tb_log,
        stderr=tb_log,
        start_new_session=True,
    )
    args.tensorboard_pid_file.write_text(f"{tb.pid}\n", encoding="utf-8")
    learner: subprocess.Popen[bytes] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        terminate(learner)
        terminate(tb)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        learner = subprocess.Popen(args.command, start_new_session=True)
        return learner.wait()
    finally:
        terminate(tb)
        tb_log.close()
        with suppress(FileNotFoundError):
            args.tensorboard_pid_file.unlink()


if __name__ == "__main__":
    sys.exit(main())
