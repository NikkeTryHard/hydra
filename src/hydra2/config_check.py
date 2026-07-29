"""``python -m hydra2.config_check``: prove the running environment matches
the declared dependency contract.

Checks (all hard failures on mismatch):
- torch == 2.13.0 exactly, with CUDA build and sm_120 kernels present
- lightning-fabric == 2.6.5 standalone; Trainer packages absent
- riichienv == 0.4.8, mahjax at the pinned git SHA, jax importable
- ruff / pyrefly / pytest installed versions match the pyproject pins
- parity tolerance constants are sane
"""

import json
import sys
import tomllib
from typing import Any


def _pypi_pins() -> dict[str, str]:
    # Portable pyproject locate: repo_root() marker walk (not parents[2] depth).
    # Evidence: https://docs.python.org/3/library/pathlib.html
    # Evidence: src/hydra2/config.py:90-108 _find_repo_root walk pattern
    from hydra2.config import repo_root

    pyproject = repo_root() / "pyproject.toml"
    raw = pyproject.read_text(encoding="utf-8")
    data: dict[str, Any] = tomllib.loads(raw)
    tool: Any = data.get("tool")
    assert isinstance(tool, dict)
    pixi: Any = tool.get("pixi")
    assert isinstance(pixi, dict)
    pypi_raw: Any = pixi.get("pypi-dependencies")
    assert isinstance(pypi_raw, dict)
    pypi: dict[Any, Any] = pypi_raw
    pins: dict[str, str] = {}
    for name, value in pypi.items():
        if isinstance(name, str) and isinstance(value, str):
            pins[name] = value.lstrip("=")
    feature: Any = pixi.get("feature")
    assert isinstance(feature, dict)
    dev_feature: Any = feature.get("dev")
    assert isinstance(dev_feature, dict)
    dev_raw: Any = dev_feature.get("pypi-dependencies")
    assert isinstance(dev_raw, dict)
    dev: dict[Any, Any] = dev_raw
    for name, value in dev.items():
        if isinstance(name, str) and isinstance(value, str):
            pins[name] = value.lstrip("=")
    return pins


def main() -> int:
    from hydra2._probe_support import (
        check_trainer_absence,
        require_module_imports,
        verify_torch_cuda_stack,
    )
    from hydra2.config import MAHJAX_PIN_SHA, PARITY_ABS_TOL, PARITY_REL_TOL
    from hydra2.runtime.environment import IMPORTABLE_RUNTIME_MODULES

    findings: list[tuple[str, bool, str]] = []
    pins = _pypi_pins()

    import importlib.metadata as md

    for dist_name in ("torch", "lightning-fabric", "riichienv", "ruff", "pyrefly"):
        declared = pins.get(dist_name)
        installed = md.version(dist_name)
        ok = declared is not None and installed == declared
        findings.append((f"pin:{dist_name}", ok, f"declared={declared} installed={installed}"))
    pytest_declared = pins.get("pytest")
    pytest_installed = md.version("pytest")
    findings.append(
        (
            "pin:pytest",
            pytest_declared is not None and pytest_installed == pytest_declared,
            f"declared={pytest_declared} installed={pytest_installed}",
        )
    )

    mahjax_ok: bool = False
    try:
        raw_direct: str | None = md.distribution("mahjax").read_text("direct_url.json")
        direct_text: str = raw_direct if raw_direct is not None else "{}"
        direct: dict[str, Any] = json.loads(direct_text)
        vcs_info: Any = direct.get("vcs_info", {})
        assert isinstance(vcs_info, dict)
        commit_raw: Any = vcs_info.get("commit_id", "")
        commit: str = commit_raw if isinstance(commit_raw, str) else ""
        mahjax_ok = commit == MAHJAX_PIN_SHA
        detail = f"commit={commit}"
    except Exception as exc:
        detail = f"unreadable: {exc}"
    findings.append(("pin:mahjax_git_sha", mahjax_ok, detail))

    torch_ok, torch_detail = verify_torch_cuda_stack(require_sm120=True)
    findings.append(("torch:cuda_sm120", torch_ok, torch_detail))

    trainer_ok, trainer_detail = check_trainer_absence()
    findings.append(("trainer_packages_absent", trainer_ok, trainer_detail))

    imports_ok, imports_detail = require_module_imports(IMPORTABLE_RUNTIME_MODULES)
    findings.append(("fresh_module_imports", imports_ok, imports_detail))

    tolerance_ok = 0 < PARITY_REL_TOL < 1e-3 and 0 <= PARITY_ABS_TOL <= PARITY_REL_TOL
    findings.append(
        ("config:parity_tolerance", tolerance_ok, f"rtol={PARITY_REL_TOL} atol={PARITY_ABS_TOL}")
    )

    failed = [(name, detail) for name, ok, detail in findings if not ok]
    for name, ok, detail in findings:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if len(failed) != 0:
        print(f"config-check FAILED ({len(failed)} finding(s))", file=sys.stderr)
        return 1
    print("config-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
