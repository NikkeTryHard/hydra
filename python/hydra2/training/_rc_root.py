"""Run-config root assembly: section dispatch, cross-checks, YAML load.

Owns the section-parser table, the top-level strict root parse with
cross-section coherence checks, and the base+override YAML loader that
produces the frozen :class:`RunConfig`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import yaml  # pyrefly: ignore[untyped-import] # pyyaml ships no stubs here; documents bound precisely at use

from hydra2.contracts.common import ContractError
from hydra2.training._rc_digest import effective_run_id as effective_run_id
from hydra2.training._rc_parse import _parse_data as _parse_data
from hydra2.training._rc_parse import _parse_loop as _parse_loop
from hydra2.training._rc_parse import _parse_model as _parse_model
from hydra2.training._rc_parse import _parse_optimizer as _parse_optimizer
from hydra2.training._rc_parse import _parse_rl as _parse_rl
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

try:
    from hydra2._native import contracts as _root_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _root_bridge = None


def _require_root_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _root_bridge is None or not hasattr(_root_bridge, "rc_cross_validate"):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _root_bridge


def _cross_validate(config: RunConfig) -> None:
    """Thin bridge delegate: coherence gate per discriminated source kind.

    ``supervised``/``distill`` use ``rc_cross_validate`` (which rejects the
    RL-only replay arm); ``rl`` uses ``rc_cross_validate_rl`` (source union
    plus the frozen RL recipe: rollout source, no dataset roots, AdamW
    1e-5/constant/caps, shared ``w_value`` 0.005; privileged labels bind via
    the artifact, so the supervised manifest-pin gate does not apply).
    """
    weights = config.weights
    try:
        if config.run.kind == "rl":
            data = config.data
            _require_root_bridge().rc_cross_validate_rl(
                config.run.kind,
                data is not None and len(data.roots) > 0,
                config.rl is not None,
                config.optimizer.name,
                config.optimizer.lr,
                config.optimizer.betas[0],
                config.optimizer.betas[1],
                config.optimizer.weight_decay,
                config.optimizer.head_lr_mult,
                config.scheduler.name,
                config.scheduler.warmup_updates,
                config.loop.microbatch_size,
                config.loop.accumulation_steps,
                config.loop.max_updates,
                config.loop.stratified_sampling,
                config.loop.gradient_clip_norm,
                weights.w_value,
                config.runtime.precision,
                config.loop.precision,
                config.runtime.device,
                config.output.run_id,
                config.run.run_id,
                config.runtime.compile_mode,
            )
        else:
            _require_root_bridge().rc_cross_validate(
                config.runtime.precision,
                config.loop.precision,
                config.runtime.device,
                config.output.run_id,
                config.run.run_id,
                weights.w_placement,
                weights.w_value,
                list(weights.w_event.values()) if weights.w_event is not None else [],
                list(weights.w_belief.values()) if weights.w_belief is not None else [],
                weights.privileged_source_hash is not None,
                config.runtime.compile_mode,
            )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


__all__ = [
    "_SECTION_PARSERS",
    "_cross_validate",
    "_parse_root",
    "_parse_section",
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


def _parse_section(fields: dict[str, Any], section: str) -> Any:
    """Parse one closed section (unknown keys reject inside the parser)."""
    raw_section: Any = fields.get(section, {})
    if not isinstance(raw_section, dict):
        raise ContractError(f"{section} must be a mapping, got {type(raw_section).__name__}")
    parser: Callable[[dict[str, Any]], Any] = _SECTION_PARSERS[section]
    return parser(raw_section)


def _parse_root(mapping: Any) -> RunConfig:
    if not isinstance(mapping, dict):
        raise ContractError(f"run-config top level must be a mapping, got {type(mapping).__name__}")
    fields = _reject_unknown(
        {key: value for key, value in mapping.items() if key != "rl"},
        CONFIG_SECTIONS,
        where="run-config",
    )
    # Kind first: the discriminated source union decides which sections parse.
    # `data` is parsed only for supervised/distill (rc_parse_data requires
    # dataset roots); RL skips it after the empty-check and sets data=None.
    run = _parse_section(fields, "run")
    kind: str = run.kind
    raw_rl = mapping.get("rl")
    if kind in ("supervised", "distill"):
        if raw_rl is not None:
            raise ContractError(
                "top-level 'rl' is parked for supervised/distill runs (must be null); "
                "an explicit 'rl' section requires run.kind 'rl'"
            )
        rl = None
        data = _parse_section(fields, "data")
    else:
        if raw_rl is None:
            raise ContractError(
                "run.kind 'rl' requires an explicit 'rl' section (no dummy DataConfig fallback)"
            )
        if not isinstance(raw_rl, dict):
            raise ContractError(f"rl must be a mapping, got {type(raw_rl).__name__}")
        data_raw: Any = fields.get("data", {})
        if data_raw is None:
            data_raw = {}
        if not isinstance(data_raw, dict):
            raise ContractError(f"data must be a mapping, got {type(data_raw).__name__}")
        if "root" in data_raw or "roots" in data_raw:
            raise ContractError(
                "run.kind 'rl' forbids dataset roots/source: omit data.root/data.roots "
                "(rollout source is the immutable artifact ledger)"
            )
        if len(data_raw) > 0:
            raise ContractError(
                "run.kind 'rl' must omit the data section (rollout source replaces it)"
            )
        data = None
        rl = _parse_rl(raw_rl)
    parsed: dict[str, Any] = {}
    for section in CONFIG_SECTIONS:
        if section in ("run", "data"):
            continue
        parsed[section] = _parse_section(fields, section)
    config = RunConfig(
        run=run,
        data=data,
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
        rl=rl,
    )
    _cross_validate(config)
    return config


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
