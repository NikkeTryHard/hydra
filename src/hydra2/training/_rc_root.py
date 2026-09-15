"""Run-config root assembly: section dispatch, cross-checks, YAML load.

Owns the section-parser table, the top-level strict root parse with
cross-section coherence checks, and the base+override YAML loader that
produces the frozen :class:`RunConfig`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from hydra2.contracts.common import ContractError
from hydra2.training._rc_digest import effective_run_id as effective_run_id
from hydra2.training._rc_parse import _parse_data as _parse_data
from hydra2.training._rc_parse import _parse_loop as _parse_loop
from hydra2.training._rc_parse import _parse_model as _parse_model
from hydra2.training._rc_parse import _parse_optimizer as _parse_optimizer
from hydra2.training._rc_parse import _parse_run as _parse_run
from hydra2.training._rc_parse import _parse_runtime as _parse_runtime
from hydra2.training._rc_parse import _parse_scheduler as _parse_scheduler
from hydra2.training._rc_parse import _parse_weights as _parse_weights
from hydra2.training._rc_parse_aux import _parse_eval as _parse_eval
from hydra2.training._rc_parse_aux import _parse_mirror as _parse_mirror
from hydra2.training._rc_parse_aux import _parse_output as _parse_output
from hydra2.training._rc_parse_aux import _parse_seeds as _parse_seeds
from hydra2.training._rc_parse_aux import _parse_selection as _parse_selection
from hydra2.training._rc_parse_aux import _parse_telemetry as _parse_telemetry
from hydra2.training._rc_require import _interpolate_tree as _interpolate_tree
from hydra2.training._rc_require import _reject_unknown as _reject_unknown
from hydra2.training._rc_require import deep_merge as deep_merge
from hydra2.training._rc_sections import _RUN_ID_RE as _RUN_ID_RE
from hydra2.training._rc_sections import CONFIG_SECTIONS as CONFIG_SECTIONS
from hydra2.training._rc_sections import RunConfig as RunConfig

__all__ = [
    "_SECTION_PARSERS",
    "_cross_validate",
    "_parse_root",
    "_read_yaml_mapping",
    "load_run_config",
]

_SECTION_PARSERS: dict[str, Any] = {
    "run": _parse_run,
    "data": _parse_data,
    "model": _parse_model,
    "weights": _parse_weights,
    "optimizer": _parse_optimizer,
    "scheduler": _parse_scheduler,
    "runtime": _parse_runtime,
    "loop": _parse_loop,
    "seeds": _parse_seeds,
    "selection": _parse_selection,
    "mirror": _parse_mirror,
    "telemetry": _parse_telemetry,
    "eval": _parse_eval,
    "output": _parse_output,
}


def _parse_root(mapping: Any) -> RunConfig:
    if not isinstance(mapping, dict):
        raise ContractError(f"run-config top level must be a mapping, got {type(mapping).__name__}")
    parked_rl = mapping.get("rl")
    if parked_rl is not None:
        raise ContractError(
            "top-level 'rl' is parked in streaming-first v1 (must be null); "
            "RL rollout sources arrive with run.kind=rl in a later wave"
        )
    fields = _reject_unknown(
        {key: value for key, value in mapping.items() if key != "rl"},
        CONFIG_SECTIONS,
        where="run-config",
    )
    parsed: dict[str, Any] = {}
    for section in CONFIG_SECTIONS:
        raw_section: Any = fields.get(section, {})
        if not isinstance(raw_section, dict):
            raise ContractError(f"{section} must be a mapping, got {type(raw_section).__name__}")
        parsed[section] = _SECTION_PARSERS[section](raw_section)
    config = RunConfig(
        run=parsed["run"],
        data=parsed["data"],
        model=parsed["model"],
        weights=parsed["weights"],
        optimizer=parsed["optimizer"],
        scheduler=parsed["scheduler"],
        runtime=parsed["runtime"],
        loop=parsed["loop"],
        seeds=parsed["seeds"],
        selection=parsed["selection"],
        mirror=parsed["mirror"],
        telemetry=parsed["telemetry"],
        eval=parsed["eval"],
        output=parsed["output"],
    )
    _cross_validate(config)
    return config


def _cross_validate(config: RunConfig) -> None:
    """Cross-section coherence checks (fail closed)."""
    runtime_precision = config.runtime.precision
    loop_precision = config.loop.precision
    expected_loop = "fp32" if runtime_precision == "fp32" else "bf16_mixed"
    if loop_precision != expected_loop:
        raise ContractError(
            f"loop.precision {loop_precision!r} disagrees with runtime.precision "
            f"{runtime_precision!r} (expected {expected_loop!r}); autocast scope must match"
        )
    output_id = config.output.run_id
    effective_run_id = (
        output_id if output_id is not None and len(output_id) > 0 else config.run.run_id
    )
    if _RUN_ID_RE.fullmatch(effective_run_id) is None:
        raise ContractError(f"effective run id invalid: {effective_run_id!r}")
    if runtime_precision == "bf16_mixed" and config.runtime.device == "cpu":
        raise ContractError(
            "runtime.precision 'bf16_mixed' requires a CUDA device "
            "(never silent CPU fallback; device availability itself is "
            "checked at bind time via require_device_available)"
        )
    auxiliary_positive = any(
        weight > 0.0
        for weight in [
            config.weights.w_placement,
            config.weights.w_value,
            *((config.weights.w_event if config.weights.w_event is not None else {}).values()),
            *((config.weights.w_belief if config.weights.w_belief is not None else {}).values()),
        ]
    )
    if auxiliary_positive and config.weights.privileged_source_hash is None:
        raise ContractError(
            "positive placement/value/event/belief weight REQUIRES "
            "weights.privileged_source_hash (positive-requires-binding): "
            "privileged labels must come from the pinned manifest"
        )


def _read_yaml_mapping(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"run config unreadable: {path} ({exc})") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"run config YAML malformed: {path} ({exc})") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ContractError(
            f"run config top level must be a mapping, got {type(document).__name__}"
        )
    return document


def load_run_config(
    path: str | Path,
    *,
    override_path: str | Path | None = None,
    environ: Any | None = None,
) -> RunConfig:
    """Load, merge, interpolate, and strictly validate a run config.

    ``override_path`` (optional) is deep-merged over ``path`` (base+override
    composition: operator overrides win key-wise; lists replace wholesale).
    ``${VAR}`` interpolation uses ``environ`` (default ``os.environ``) and
    the :data:`INTERPOLATION_ALLOWLIST`. Unknown keys raise at every level.
    """
    base_path = Path(path)
    document = _read_yaml_mapping(base_path)
    if override_path is not None:
        override = _read_yaml_mapping(Path(override_path))
        if not isinstance(override, dict):
            raise ContractError("override top level must be a mapping")
        if not isinstance(document, dict):
            raise ContractError("base top level must be a mapping")
        document = deep_merge(document, override)
    env: Any = os.environ if environ is None else environ
    resolved = _interpolate_tree(document, environ=env, where="run-config")
    return _parse_root(resolved)
