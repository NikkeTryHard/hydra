"""Versioned schema for actor-visible decision corpora.

A corpus contains complete games.  Each game stores, per decision, the acting
seat's 39x48 token bundle, every legal descriptor row (no padding rows), the
action id chosen by the behaviour policy, and terminal ranks indexed
by absolute seat.  Corpora recorded by self-play are tagged ``self_play``;
they are behaviour data, not expert demonstrations.

These tensors cannot certify actor visibility or authorization. The producer
must enforce visibility, and the caller must supply authorized data.
"""

from dataclasses import dataclass

import torch

from .model_schema import ACTION_WIDTH, TOKEN_COUNT, TOKEN_WIDTH

DATASET_FORMAT = "hydra3.decisions"
DATASET_VERSION = 1
PROVENANCE_SELF_PLAY = "self_play"

GAME_FIELDS = (
    "seed",
    "tokens",
    "actions",
    "legal",
    "chosen",
    "seats",
    "forced",
    "scores",
    "ranks",
)
PAYLOAD_FIELDS = ("format", "version", "wire", "provenance", "games")

_ACTION_KIND_MAX = 12
_TILE_FACE_MAX = 33
_SEAT_MAX = 3
_DESCRIPTOR_SOURCE_MAX = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_tensor(
    name: str,
    value: object,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value
    _require(tensor.device.type == "cpu", f"{name} must be a CPU tensor")
    _require(tensor.dtype == dtype, f"{name} must have dtype {dtype}")
    if shape is not None:
        _require(tuple(tensor.shape) == shape, f"{name} must have shape {shape}")
    return tensor


@dataclass(frozen=True, slots=True, eq=False)
class GameRecord:
    """One complete game in the behavior-cloning ingest format.

    Validation checks tensor geometry and labels, not actor visibility or data
    authorization. ``seed`` identifies the game's wall stream. ``tokens`` is
    CPU float32 ``[decisions, 39, 48]`` and contains the acting seat's bundle.
    ``actions`` is CPU int32 ``[sum(legal), 16]`` and stores legal descriptor
    rows only, with no padding. ``legal`` gives the candidate count per
    decision and ``chosen`` the selected action id.
    ``seats`` the acting absolute seat, and ``forced`` the no-choice flag.
    ``scores`` and ``ranks`` are the terminal results indexed by absolute seat;
    ranks must be a tie-free permutation of 1..4.
    """

    seed: int
    tokens: torch.Tensor
    actions: torch.Tensor
    legal: torch.Tensor
    chosen: torch.Tensor
    seats: torch.Tensor
    forced: torch.Tensor
    scores: torch.Tensor
    ranks: torch.Tensor

    def __post_init__(self) -> None:
        validate_game(self)


@dataclass(frozen=True, slots=True, eq=False)
class DecisionDataset:
    provenance: str
    games: tuple[GameRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str):
            raise TypeError("provenance must be a string")
        _require(bool(self.provenance.strip()), "provenance must not be empty")
        games = tuple(self.games)
        _require(bool(games), "dataset must contain at least one game")
        if not all(isinstance(game, GameRecord) for game in games):
            raise TypeError("dataset games must be GameRecord instances")
        object.__setattr__(self, "games", games)


def validate_game(game: GameRecord) -> None:
    """Reject malformed decision records; visibility and authorization are external."""
    if type(game.seed) is not int:
        raise TypeError("seed must be an integer")
    _require(0 <= game.seed < 2**64, "seed must be an unsigned 64-bit integer")

    tokens = _require_tensor("tokens", game.tokens, torch.float32)
    _require(tokens.ndim == 3, "tokens must be [decisions, 39, 48]")
    count, token_count, width = tokens.shape
    _require(
        (token_count, width) == (TOKEN_COUNT, TOKEN_WIDTH),
        f"tokens must be [decisions, {TOKEN_COUNT}, {TOKEN_WIDTH}]",
    )
    _require(count > 0, "each game must contain at least one decision")
    _require(bool(torch.isfinite(tokens).all()), "tokens must be finite")

    actions = _require_tensor("actions", game.actions, torch.int32)
    _require(actions.ndim == 2 and actions.shape[1] == ACTION_WIDTH, "actions must be [rows, 16]")
    legal = _require_tensor("legal", game.legal, torch.int64, (count,))
    chosen = _require_tensor("chosen", game.chosen, torch.int64, (count,))
    seats = _require_tensor("seats", game.seats, torch.int64, (count,))
    forced = _require_tensor("forced", game.forced, torch.bool, (count,))
    _require_tensor("scores", game.scores, torch.int32, (4,))
    ranks = _require_tensor("ranks", game.ranks, torch.int32, (4,))

    _require(bool((legal > 0).all()), "every decision requires at least one legal action")
    _require(int(legal.sum()) == actions.shape[0], "legal counts must match descriptor rows")
    _require(bool((chosen >= 0).all()), "chosen action ids must be nonnegative")
    _require(
        bool(((seats >= 0) & (seats <= _SEAT_MAX)).all()),
        "seats must be absolute seats 0..3",
    )
    _require(torch.equal(forced, legal == 1), "forced must mean exactly one legal action")
    _require(
        sorted(int(rank) for rank in ranks.tolist()) == [1, 2, 3, 4],
        "ranks must bind each absolute seat to a tie-free permutation of 1..4",
    )

    ids = actions[:, 0]
    _require(bool((ids >= 0).all()), "legal descriptors cannot contain padding rows")
    _require(
        bool(((actions[:, 1] >= 0) & (actions[:, 1] <= _ACTION_KIND_MAX)).all()),
        "action kind ordinals must be in 0..12",
    )
    _require(
        bool(((actions[:, 2] >= 0) & (actions[:, 2] <= _DESCRIPTOR_SOURCE_MAX)).all()),
        "descriptor source seats must be relative seats 0..4",
    )
    _require(
        bool(((actions[:, 3] >= -1) & (actions[:, 3] <= _TILE_FACE_MAX)).all()),
        "descriptor tile faces must be -1 or 0..33",
    )
    for column in (4, 5, 6, 15):
        _require(
            bool(((actions[:, column] == 0) | (actions[:, column] == 1)).all()),
            f"descriptor column {column} must be a bit",
        )
    removals = actions[:, 7:11]
    _require(
        bool(((removals >= -1) & (removals <= _TILE_FACE_MAX)).all()),
        "descriptor removals must be -1 or tile faces 0..33",
    )
    _require(
        bool(((actions[:, 11:15] == 0) | (actions[:, 11:15] == 1)).all()),
        "descriptor removal red bits must be bits",
    )

    decision_of_row = torch.repeat_interleave(torch.arange(count), legal)
    chosen_per_row = chosen.repeat_interleave(legal)
    matches = ids.eq(chosen_per_row)
    match_counts = torch.zeros(count, dtype=torch.int64)
    match_counts.index_add_(0, decision_of_row, matches.to(torch.int64))
    _require(
        bool((match_counts == 1).all()),
        "chosen action id must name exactly one legal descriptor per decision",
    )
    id_span = int(ids.max()) + 1
    keys = decision_of_row * id_span + ids
    _require(
        int(torch.unique(keys).numel()) == keys.numel(),
        "legal action ids must be unique within a decision",
    )
