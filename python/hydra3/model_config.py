"""Frozen construction options for the policy network."""

from dataclasses import dataclass

SCORER_EDIT = "edit"
SCORER_PLAIN = "plain"
SCORERS: tuple[str, str] = (SCORER_EDIT, SCORER_PLAIN)


@dataclass(frozen=True)
class ModelConfig:
    """Model geometry and which candidate scorer to build.

    The edit scorer adds the explicit private hand-change channel to the shared
    rich-state control. The plain scorer reads the same state and semantic
    action fields without that before/after comparison.
    """

    width: int = 128
    depth: int = 4
    heads: int = 4
    ff: int = 384
    scorer: str = SCORER_EDIT

    def __post_init__(self) -> None:
        for name, value in (
            ("width", self.width),
            ("depth", self.depth),
            ("heads", self.heads),
            ("ff", self.ff),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.width % self.heads:
            raise ValueError(f"width ({self.width}) must be divisible by heads ({self.heads})")
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {self.scorer!r}")
