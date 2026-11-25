"""Cross-game tensor layout and corpus validation tests."""

import pickle
from pathlib import Path

import pytest
import torch
from hydra3.learn_batch import flatten_games, pad_candidates
from hydra3.learn_dataset import load_dataset, save_dataset
from hydra3.learn_schema import (
    DATASET_FORMAT,
    DATASET_VERSION,
    PROVENANCE_SELF_PLAY,
)
from hydra3.model_schema import WIRE_VERSION

from .learn_support import empty_game, make_game


class _UnallowlistedPayload:
    pass


def test_two_game_flatten_and_minibatch_padding() -> None:
    first = make_game(
        [(0, [10, 11]), (3, [20, 21, 22])],
        chosen=[11, 21],
        ranks=[1, 2, 3, 4],
        seed=1,
    )
    second = make_game(
        [(1, [30]), (2, [40, 41, 42]), (0, [50, 51])],
        chosen=[30, 42, 50],
        ranks=[4, 3, 2, 1],
        seed=2,
    )

    flat = flatten_games([first, second])
    assert flat.legal.tolist() == [2, 3, 1, 3, 2]
    assert flat.offsets.tolist() == [0, 2, 5, 6, 9]
    assert flat.seats.tolist() == [0, 3, 1, 2, 0]
    assert flat.target.tolist() == pytest.approx([1.5, -1.5, -0.5, 0.5, -1.5])
    assert flat.forced.tolist() == [False, False, True, False, False]
    assert flat.chosen_index.tolist() == [1, 1, 0, 2, 0]

    padded = pad_candidates(flat, torch.arange(5))
    assert tuple(padded.shape) == (5, 3, 16)
    assert padded[:, :, 0].tolist() == [
        [10, 11, -1],
        [20, 21, 22],
        [30, -1, -1],
        [40, 41, 42],
        [50, 51, -1],
    ]
    assert padded[torch.arange(5), flat.chosen_index, 0].tolist() == [
        11,
        21,
        30,
        42,
        50,
    ]


def test_dataset_rejects_illegal_targets_and_terminal_ranks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chosen"):
        make_game([(0, [5, 6]), (1, [7])], chosen=[9, 7])
    with pytest.raises(ValueError, match="ranks"):
        make_game([(0, [5]), (1, [6])], ranks=[1, 2, 3, 3])

    with pytest.raises(ValueError, match="at least one decision"):
        empty_game()

    with pytest.raises(ValueError, match="at least one game"):
        save_dataset(tmp_path / "empty.pt", [], PROVENANCE_SELF_PLAY)

    bad_version = tmp_path / "version.pt"
    torch.save(
        {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION + 1,
            "wire": WIRE_VERSION,
            "provenance": PROVENANCE_SELF_PLAY,
            "games": [],
        },
        bad_version,
    )
    with pytest.raises(ValueError, match="version"):
        load_dataset(bad_version)


def test_corpus_loader_rejects_unallowlisted_objects(tmp_path: Path) -> None:
    corpus = tmp_path / "hostile.pt"
    torch.save(
        {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION,
            "wire": WIRE_VERSION,
            "provenance": "self_play",
            "games": [_UnallowlistedPayload()],
        },
        corpus,
    )
    with pytest.raises(pickle.UnpicklingError):
        load_dataset(corpus)
