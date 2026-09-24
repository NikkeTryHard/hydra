"""Verbose sampler construction (split half)."""

from __future__ import annotations

from typing import Any

from hydra2.tracking.verbose_sampler import VerboseSampler as VerboseSampler
from hydra2.tracking.verbose_sampler import is_enabled as is_enabled

__all__ = [
    "NullVerboseSampler",
    "make_verbose_sampler",
]


class NullVerboseSampler(VerboseSampler):
    """Disabled sampler: never spawns a thread, never imports SDKs."""

    def __init__(self) -> None:
        super().__init__(enabled=False, sink_path="verbose-telemetry.jsonl")

    def start(self) -> bool:
        """Always off; cheap no-op."""
        return False


def make_verbose_sampler(**kwargs: Any) -> VerboseSampler:
    """Build an enabled sampler when opted in, else :class:`NullVerboseSampler`.

    Never raises: any misconfiguration falls back to a disabled sampler.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullVerboseSampler()
        return VerboseSampler(enabled=True, **kwargs)
    except Exception:
        return NullVerboseSampler()
