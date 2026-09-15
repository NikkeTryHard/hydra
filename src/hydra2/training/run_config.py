"""Streaming-first training run configuration: YAML authority, output layout, resume.

Re-export facade over the split modules: :mod:`hydra2.training._rc_sections`
(section records, vocabulary, resume state), :mod:`hydra2.training._rc_require`
(strict-loading primitives), :mod:`hydra2.training._rc_parse` and
:mod:`hydra2.training._rc_parse_aux` (section parsers),
:mod:`hydra2.training._rc_root` (root assembly and YAML load),
:mod:`hydra2.training._rc_digest` (canonical digest and run layout), and
:mod:`hydra2.training._rc_resume` (checkpoint sidecars and resume plans).
Import from this path; it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.training._rc_digest import create_run_layout as create_run_layout
from hydra2.training._rc_digest import read_latest_run as read_latest_run
from hydra2.training._rc_digest import run_config_digest as run_config_digest
from hydra2.training._rc_digest import run_config_to_dict as run_config_to_dict
from hydra2.training._rc_digest import run_dir_for as run_dir_for
from hydra2.training._rc_require import deep_merge as deep_merge
from hydra2.training._rc_resume import find_latest_checkpoint as find_latest_checkpoint
from hydra2.training._rc_resume import format_plan as format_plan
from hydra2.training._rc_resume import resolve_resume_plan as resolve_resume_plan
from hydra2.training._rc_root import load_run_config as load_run_config
from hydra2.training._rc_sections import CONFIG_SECTIONS as CONFIG_SECTIONS
from hydra2.training._rc_sections import INTERPOLATION_ALLOWLIST as INTERPOLATION_ALLOWLIST
from hydra2.training._rc_sections import RUN_KINDS as RUN_KINDS
from hydra2.training._rc_sections import AccumState as AccumState
from hydra2.training._rc_sections import DataConfig as DataConfig
from hydra2.training._rc_sections import EvalConfig as EvalConfig
from hydra2.training._rc_sections import LoopConfig as LoopConfig
from hydra2.training._rc_sections import MirrorConfig as MirrorConfig
from hydra2.training._rc_sections import ModelConfig as ModelConfig
from hydra2.training._rc_sections import OptimizerConfig as OptimizerConfig
from hydra2.training._rc_sections import OutputConfig as OutputConfig
from hydra2.training._rc_sections import ResumePlan as ResumePlan
from hydra2.training._rc_sections import RunConfig as RunConfig
from hydra2.training._rc_sections import RunMeta as RunMeta
from hydra2.training._rc_sections import RuntimeConfig as RuntimeConfig
from hydra2.training._rc_sections import SchedulerConfig as SchedulerConfig
from hydra2.training._rc_sections import SeedsConfig as SeedsConfig
from hydra2.training._rc_sections import SelectionConfig as SelectionConfig
from hydra2.training._rc_sections import ShuffleState as ShuffleState
from hydra2.training._rc_sections import StreamCursor as StreamCursor
from hydra2.training._rc_sections import TelemetryConfig as TelemetryConfig
from hydra2.training._rc_sections import WeightsConfig as WeightsConfig
from hydra2.training._rc_sections import WorkerPlan as WorkerPlan

__all__ = [
    "CONFIG_SECTIONS",
    "INTERPOLATION_ALLOWLIST",
    "RUN_KINDS",
    "AccumState",
    "DataConfig",
    "EvalConfig",
    "LoopConfig",
    "MirrorConfig",
    "ModelConfig",
    "OptimizerConfig",
    "OutputConfig",
    "ResumePlan",
    "RunConfig",
    "RunMeta",
    "RuntimeConfig",
    "SchedulerConfig",
    "SeedsConfig",
    "SelectionConfig",
    "ShuffleState",
    "StreamCursor",
    "TelemetryConfig",
    "WeightsConfig",
    "WorkerPlan",
    "create_run_layout",
    "deep_merge",
    "find_latest_checkpoint",
    "format_plan",
    "load_run_config",
    "read_latest_run",
    "resolve_resume_plan",
    "run_config_digest",
    "run_config_to_dict",
    "run_dir_for",
]
