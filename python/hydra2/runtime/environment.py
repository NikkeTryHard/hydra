"""Environment manifest capture (locked-environment identity).

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

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical, sha256_file
from hydra2.config import MAHJAX_GIT_URL, MAHJAX_PIN_SHA, repo_root

ENV_MANIFEST_ARTIFACT_TYPE = "hydra2.environment"
ENV_MANIFEST_SCHEMA_VERSION = "1.0.0"

IMPORTABLE_RUNTIME_MODULES = (
    "hydra2",
    "torch",
    "riichienv",
    "mahjax",
    "jax",
)


def _pixi_lock_hash() -> str:
    """Return pixi.lock sha256 or MISSING sentinel (portable fallback).

    Graceful degradation matches the ``dist_version()`` ``MISSING``
    pattern below: degrade to a sentinel instead of raising on
    missing input. Strict mode via ``HYDRA2_REQUIRE_PIXI_LOCK=1``
    raises instead of degrading.
    """
    try:
        lock_path = repo_root() / "pixi.lock"
    except Exception:
        lock_path = None  # repo_root marker walk failed (e.g., installed wheel)
    # Missing lock degrades to a sentinel unless strict mode requires proof.
    if lock_path is None or not lock_path.is_file():
        if os.environ.get("HYDRA2_REQUIRE_PIXI_LOCK") == "1":
            raise FileNotFoundError(
                f"pixi.lock not found at {lock_path}; environment is unproven "
                "(set HYDRA2_REQUIRE_PIXI_LOCK=1 to require)"
            )
        return "MISSING: pixi.lock not found (set HYDRA2_REQUIRE_PIXI_LOCK=1 to require)"
    return sha256_file(lock_path)


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    """Query GPUs via nvidia-smi with graceful degrade (portable).

    Return [] when the binary is absent or the query fails, never
    raise; a 5s timeout keeps capture responsive.
    """
    smi_binary = shutil.which("nvidia-smi")
    # Absent binary means no GPU table; callers record empty, never an error.
    if smi_binary is None:
        return []
    try:
        smi_result = subprocess.run(
            [
                smi_binary,
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
    gpu_entries: list[dict[str, Any]] = []
    for output_line in smi_result.stdout.strip().splitlines():
        if output_line.strip() == "":
            continue
        line_parts = [part.strip() for part in output_line.split(",")]
        # Malformed lines are skipped with a warning; one bad GPU never fails capture.
        if len(line_parts) != 4:
            warnings.warn(f"nvidia-smi unexpected line (skipped): {output_line!r}", stacklevel=2)
            continue
        index_text, gpu_name, driver_version, compute_cap = line_parts
        try:
            gpu_index = int(index_text)
        except ValueError:
            warnings.warn(f"nvidia-smi bad index (skipped): {output_line!r}", stacklevel=2)
            continue
        gpu_entries.append(
            {
                "index": gpu_index,
                "name": gpu_name,
                "driver_version": driver_version,
                "compute_capability": compute_cap,
            }
        )
    return gpu_entries


def capture_environment_manifest() -> tuple[dict[str, Any], str]:
    """Capture the full environment manifest; returns (manifest, sha256)."""
    import importlib.metadata as md

    import torch

    # Empty when CUDA is absent; torch reports the compiled arch list only.
    cuda_arch_list = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []

    def dist_version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            # Absent optional dependency degrades to a sentinel, never an error.
            return "MISSING"

    env_manifest = {
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
            "arch_list": cuda_arch_list,
        },
        "driver": {
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
            "gpus": _nvidia_smi_gpus(),
        },
        "extensions": {
            "riichienv": dist_version("riichienv"),
            "mahjax": dist_version("mahjax"),
            "mahjax_git_url": MAHJAX_GIT_URL,
            "mahjax_pin_sha": MAHJAX_PIN_SHA,
            "jax": dist_version("jax"),
        },
    }
    return env_manifest, of_canonical(env_manifest)


def write_environment_manifest(destination: Path) -> tuple[Path, str]:
    """Capture and atomically publish the manifest; returns (path, sha256)."""
    env_manifest, manifest_digest = capture_environment_manifest()
    atomic_replace_bytes(Path(destination), canonical_bytes(env_manifest))
    return Path(destination), manifest_digest


def main() -> int:
    from hydra2.config import artifact_root

    env_dir = artifact_root() / "environment"
    env_path, env_digest = write_environment_manifest(env_dir / "environment-manifest.json")
    print(json.dumps({"path": str(env_path), "sha256": env_digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
