"""Atomic, weights-only persistence for versioned decision corpora."""

from collections.abc import Iterable
from pathlib import Path

import torch

from .checkpoint_io import atomic_torch_save
from .learn_schema import (
    DATASET_FORMAT,
    DATASET_VERSION,
    GAME_FIELDS,
    PAYLOAD_FIELDS,
    DecisionDataset,
    GameRecord,
)
from .model_schema import WIRE_VERSION


def save_dataset(path: Path | str, games: Iterable[GameRecord], provenance: str) -> None:
    """Write a validated corpus atomically.  Provenance is never inferred."""
    dataset = DecisionDataset(provenance, tuple(games))
    payload = {
        "format": DATASET_FORMAT,
        "version": DATASET_VERSION,
        "wire": WIRE_VERSION,
        "provenance": dataset.provenance,
        "games": [
            {
                "seed": game.seed,
                "tokens": game.tokens,
                "actions": game.actions,
                "legal": game.legal,
                "chosen": game.chosen,
                "seats": game.seats,
                "forced": game.forced,
                "scores": game.scores,
                "ranks": game.ranks,
            }
            for game in dataset.games
        ],
    }
    atomic_torch_save(Path(path), payload)


def load_dataset(path: Path | str) -> DecisionDataset:
    """Load a corpus without unpickling executable objects."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"decision corpus does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("decision corpus must be a dictionary payload")
    if set(payload) != set(PAYLOAD_FIELDS):
        raise ValueError("decision corpus has an unexpected field layout")
    if payload["format"] != DATASET_FORMAT:
        raise ValueError("not a Hydra3 decision corpus")
    if type(payload["version"]) is not int:
        raise TypeError("decision corpus version must be an integer")
    if payload["version"] != DATASET_VERSION:
        raise ValueError(f"unsupported decision corpus version: {payload['version']!r}")
    if payload["wire"] != WIRE_VERSION:
        raise ValueError("decision corpus was written for a different observation wire")
    provenance = payload["provenance"]
    if not isinstance(provenance, str):
        raise TypeError("decision corpus provenance must be a string")
    if not provenance.strip():
        raise ValueError("decision corpus provenance must not be empty")
    games = payload["games"]
    if not isinstance(games, list):
        raise TypeError("decision corpus games must be a list")
    records: list[GameRecord] = []
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise TypeError(f"decision game {index} must be a dictionary")
        if set(game) != set(GAME_FIELDS):
            raise ValueError(f"decision game {index} has an unexpected field layout")
        records.append(
            GameRecord(
                seed=game["seed"],
                tokens=game["tokens"],
                actions=game["actions"],
                legal=game["legal"],
                chosen=game["chosen"],
                seats=game["seats"],
                forced=game["forced"],
                scores=game["scores"],
                ranks=game["ranks"],
            )
        )
    return DecisionDataset(provenance, tuple(records))
