"""Environment manifest capture (BUILD WP-01).

Captures the locked environment identity: pixi.lock hash, Python, torch
(version, CUDA, cuDNN, arch list), NVIDIA driver via nvidia-smi, device
name/compute capability, and extension versions. The manifest is canonical
JSON; its sha256 is the environment identity recorded by completion records.

The manifest deliberately contains no wall-clock timestamps: identical
environments produce byte-identical manifests.
"""

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

from hydra2._canon import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_digest_of_json,
    sha256_file,
)
from hydra2.config import MAHJAX_GIT_URL, MAHJAX_PIN_SHA, repo_root

ENV_MANIFEST_ARTIFACT_TYPE = "hydra2.environment"
ENV_MANIFEST_SCHEMA_VERSION = "1.0.0"

IMPORTABLE_RUNTIME_MODULES = (
    "hydra2",
    "torch",
    "lightning_fabric",
    "riichienv",
    "mahjax",
    "jax",
)


def _pixi_lock_hash() -> str:
    """Return pixi.lock sha256 or MISSING sentinel (portable fallback).

    Graceful degradation matches ``dist_version()`` ``MISSING`` pattern at
    capture_environment_manifest (line 93). Strict mode opt-in via
    ``HYDRA2_REQUIRE_PIXI_LOCK=1`` raises instead of degrading.

    Evidence:
    - os.environ.get https://docs.python.org/3/library/os.html#os.environ
    - Path.is_file https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file
    - shutil.which guard pattern https://docs.python.org/3/library/shutil.html#shutil.which
    """
    try:
        lock = repo_root() / "pixi.lock"
    except Exception:
        lock = None  # repo_root marker walk failed (e.g., installed wheel)
    if lock is None or not lock.is_file():
        if os.environ.get("HYDRA2_REQUIRE_PIXI_LOCK") == "1":
            raise FileNotFoundError(
                f"pixi.lock not found at {lock}; environment is unproven "
                "(set HYDRA2_REQUIRE_PIXI_LOCK=1 to require)"
            )
        return "MISSING: pixi.lock not found (set HYDRA2_REQUIRE_PIXI_LOCK=1 to require)"
    return sha256_file(lock)

def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    """Query GPUs via nvidia-smi with graceful degrade (portable).

    Portable pattern mirrors P-009 pixi fallback: return [] when binary
    absent or subprocess fails, never raise. Timeout reduced 30->5s for
    manifest capture responsiveness.

    Evidence:
    - shutil.which https://docs.python.org/3/library/shutil.html#shutil.which
    - subprocess.run timeout https://docs.python.org/3/library/subprocess.html#subprocess.run
    - warnings.warn https://docs.python.org/3/library/warnings.html#warnings.warn
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        proc = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        warnings.warn(f"nvidia-smi query failed (degraded to []): {exc}", stacklevel=2)
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            warnings.warn(f"nvidia-smi unexpected line (skipped): {line!r}", stacklevel=2)
            continue
        index, name, driver_version, compute_cap = parts
        try:
            idx = int(index)
        except ValueError:
            warnings.warn(f"nvidia-smi bad index (skipped): {line!r}", stacklevel=2)
            continue
        gpus.append(
            {
                "index": idx,
                "name": name,
                "driver_version": driver_version,
                "compute_capability": compute_cap,
            }
        )
    return gpus


def capture_environment_manifest() -> tuple[dict[str, Any], str]:
    """Capture the full environment manifest; returns (manifest, sha256)."""
    import importlib.metadata as md

    import torch

    arch_list = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []

    def dist_version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "MISSING"

    manifest = {
        "artifact_type": ENV_MANIFEST_ARTIFACT_TYPE,
        "schema_version": ENV_MANIFEST_SCHEMA_VERSION,
        "pixi_lock_sha256": _pixi_lock_hash(),
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "arch_list": arch_list,
        },
        "driver": {
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
            "gpus": _nvidia_smi_gpus(),
        },
        "extensions": {
            "lightning-fabric": dist_version("lightning-fabric"),
            "riichienv": dist_version("riichienv"),
            "mahjax": dist_version("mahjax"),
            "mahjax_git_url": MAHJAX_GIT_URL,
            "mahjax_pin_sha": MAHJAX_PIN_SHA,
            "jax": dist_version("jax"),
        },
    }
    return manifest, sha256_digest_of_json(manifest)


def write_environment_manifest(destination: Path) -> tuple[Path, str]:
    """Capture and atomically publish the manifest; returns (path, sha256)."""
    manifest, digest = capture_environment_manifest()
    atomic_write_bytes(Path(destination), canonical_json_bytes(manifest))
    return Path(destination), digest


def main() -> int:
    from hydra2.config import artifact_root

    destination_dir = artifact_root() / "environment"
    path, digest = write_environment_manifest(destination_dir / "environment-manifest.json")
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
