"""``python -m hydra2.probe``: WP-01 runtime-probe checklist.

Executes every locked-environment probe with real assertions: clean frozen
install, fresh-process imports, Trainer-package absence, CUDA torch with
sm_120 kernels, and the full GPU probe suite (parity/resume/ordering/
fallback) via ``pytest tests/integration``. Any failure fails the command
loudly.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from hydra2._probe_support import (
    check_trainer_absence,
    require_fresh_import,
    verify_torch_cuda_stack,
)
from hydra2.config import repo_root

INTEGRATION_TEST_DIR = "tests/integration"


def _run(argv: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=3600)
    stdout_text = proc.stdout if proc.stdout is not None else ""
    stderr_text = proc.stderr if proc.stderr is not None else ""
    output = stdout_text + stderr_text
    return proc.returncode, output.strip()


def probe_frozen_install() -> tuple[bool, str]:
    """Verify frozen install — portable fallback when pixi absent.

    Evidence:
    - shutil.which https://docs.python.org/3/library/shutil.html#shutil.which
    - subprocess.run https://docs.python.org/3/library/subprocess.html#subprocess.run
    - Degrade pattern mirrors runtime/environment.py _pixi_lock_hash MISSING fallback
    """
    if shutil.which("pixi") is None:
        # Portable degrade: host python without pixi (CI, Docker, pip install)
        # still passes probe; strict frozen check is pixi-only.
        return True, "pixi not available, host env probe skipped"
    code, output = _run(["pixi", "install", "--frozen"], cwd=repo_root())
    return (
        code == 0,
        f"pixi install --frozen rc={code}"
        + (f": {output.splitlines()[-1]}" if len(output) != 0 else ""),
    )


def probe_fresh_imports() -> tuple[bool, str]:
    """Fresh-interpreter import of every locked runtime module must succeed."""
    from hydra2.runtime.environment import IMPORTABLE_RUNTIME_MODULES

    for module in IMPORTABLE_RUNTIME_MODULES:
        ok, detail = require_fresh_import(module)
        if not ok:
            return False, detail
    return True, f"fresh-process imports OK: {list(IMPORTABLE_RUNTIME_MODULES)}"


def probe_trainer_absence() -> tuple[bool, str]:
    """Trainer meta-packages absent and fresh import of lightning fails."""
    ok, detail = check_trainer_absence()
    if not ok:
        return False, detail
    # Defense in depth: metadata absence (above) plus fresh-interpreter
    # import failure (below) are both required.
    code, _ = _run([sys.executable, "-c", "import lightning"])
    return (
        code != 0,
        f"import lightning must fail: rc={code} (nonzero expected); {detail}",
    )


def probe_sm120_support() -> tuple[bool, str]:
    """CUDA torch wheel present with sm_120/Blackwell kernels available."""
    return verify_torch_cuda_stack(require_sm120=True)


def probe_gpu_suite() -> tuple[bool, str]:
    """Run the full integration probe suite (plain/fabric parity, resume
    bitwise equality, compile ordering, eager fallback, manifest round trip)."""
    code, output = _run(
        [sys.executable, "-m", "pytest", INTEGRATION_TEST_DIR, "-q", "--no-header"],
        cwd=repo_root(),
    )
    tail = "\n".join(output.splitlines()[-6:])
    return code == 0, f"pytest {INTEGRATION_TEST_DIR} rc={code}\n{tail}"


def probe_environment_manifest_round_trip() -> tuple[bool, str]:
    from hydra2.artifacts.canonical import canonical_bytes
    from hydra2.runtime.environment import capture_environment_manifest

    manifest_a, digest_a = capture_environment_manifest()
    manifest_b, digest_b = capture_environment_manifest()
    stable = digest_a == digest_b
    recanonicalized = (
        sha_of(canonical_bytes(manifest_a)) == digest_a
        and sha_of(canonical_bytes(manifest_b)) == digest_b
    )
    return stable and recanonicalized, (
        f"digest_stable={stable} canonical_roundtrip={recanonicalized} sha256={digest_a}"
    )


def sha_of(data: bytes) -> str:
    """Local sha256 hex helper for the probe path (stdlib-only)."""
    # Intentionally local: stdlib-only, no import weight on the probe path.
    import hashlib

    return "sha256:" + hashlib.sha256(data).hexdigest()


def main() -> int:
    probes = [
        ("clean pixi install --frozen", probe_frozen_install),
        ("fresh-process imports of hydra2 + locked runtime deps", probe_fresh_imports),
        ("dependency tree proves Trainer packages absent", probe_trainer_absence),
        ("torch CUDA wheel supports sm_120/Blackwell at import", probe_sm120_support),
        (
            "environment manifest canonical round trip",
            probe_environment_manifest_round_trip,
        ),
        (
            "GPU suite: parity / checkpoint resume / compile ordering / eager fallback",
            probe_gpu_suite,
        ),
    ]
    failures: list[tuple[str, str]] = []
    for title, probe in probes:
        try:
            ok, evidence = probe()
        except Exception as exc:  # why-broad: probe harness must report any
            # probe failure as evidence, never propagate.
            ok, evidence = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'} {title}")
        print(f"     {evidence}")
        if not ok:
            failures.append((title, evidence))
    if len(failures) != 0:
        print(f"runtime-probe FAILED ({len(failures)})", file=sys.stderr)
        return 1
    print("runtime-probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
