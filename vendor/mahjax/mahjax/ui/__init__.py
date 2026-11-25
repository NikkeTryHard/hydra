"""Web UI for playing against Mahjax agents, replaying saved games and showing states."""

from typing import Any

__all__ = ["create_app", "save_html", "show"]


def __getattr__(name: str) -> Any:
    # Imported lazily: pulling in FastAPI and jitting the env costs seconds, and
    # the other modules here (rules, view, mjai, record) are useful on their own.
    if name == "create_app":
        from .app import create_app

        return create_app
    if name in ("save_html", "show"):
        from . import standalone

        return getattr(standalone, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
