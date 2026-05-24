from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, override

import pytest
import torch

from hydra_learner import arena_eval


class RecordingScalarWriter:
    records: ClassVar[list[tuple[str, float, int]]] = []

    def __init__(self, path: Path | None) -> None:
        self.path = path
        type(self).records = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        type(self).records.append((tag, value, step))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_arena_parse_args_accepts_required_flags() -> None:
    args = arena_eval.parse_args(
        [
            "--baseline",
            "baseline.pt",
            "--candidate",
            "candidate-a.pt",
            "--candidate",
            "candidate-b.pt",
            "--games",
            "4",
            "--seed",
            "17",
            "--temperature",
            "0.75",
            "--output",
            "arena.json",
            "--tensorboard-dir",
            "tb",
        ]
    )

    config = arena_eval.validate_args(args)

    assert config.baseline == Path("baseline.pt")
    assert config.candidates == (Path("candidate-a.pt"), Path("candidate-b.pt"))
    assert config.games == 4
    assert config.seed == 17
    assert config.temperature == 0.75
    assert config.output_path == Path("arena.json")
    assert config.tensorboard_dir == Path("tb")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--baseline", "b.pt", "--candidate", "c.pt", "--games", "0", "--output", "out.json"], "--games"),
        (
            ["--baseline", "b.pt", "--candidate", "c.pt", "--temperature", "0", "--output", "out.json"],
            "--temperature",
        ),
    ],
)
def test_arena_validate_args_rejects_bad_values(argv: list[str], message: str) -> None:
    args = arena_eval.parse_args(argv)

    with pytest.raises(ValueError, match=message):
        arena_eval.validate_args(args)


def test_arena_inference_callback_batches_by_model_and_masks_illegal_actions() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        @override
        def forward(self, obs: torch.Tensor) -> Any:
            batch = obs.shape[0]
            logits = torch.arange(arena_eval.ACTION_SPACE, dtype=torch.float32).repeat(batch, 1) + self.offset
            return SimpleNamespace(policy_logits=logits)

    models = [
        arena_eval.LoadedArenaModel("baseline", Path("b.pt"), TinyModel(0.0), 0, 0, "raw"),
        arena_eval.LoadedArenaModel("candidate", Path("c.pt"), TinyModel(100.0), 0, 0, "raw"),
    ]
    callback = arena_eval.make_inference_callback(models, torch.device("cpu"))
    obs = [[0.0] * (arena_eval.OBS_CHANNELS * arena_eval.TILE_WIDTH) for _ in range(3)]
    legal = [[True] * arena_eval.ACTION_SPACE for _ in range(3)]
    legal[1][45] = False

    logits = callback(obs, legal, [0, 1, 0], [0, 1, 2])

    assert logits[0][45] == 45.0
    assert logits[1][44] == 144.0
    assert logits[1][45] == -float("inf")
    assert logits[2][45] == 45.0


def test_arena_run_writes_summary_per_game_and_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = [
        arena_eval.LoadedArenaModel("baseline", Path("baseline.pt"), torch.nn.Identity(), 10, 100, "raw"),
        arena_eval.LoadedArenaModel("candidate", Path("candidate.pt"), torch.nn.Identity(), 20, 200, "raw"),
    ]

    def fake_load(
        path: Path, *, name: str, config: arena_eval.ArenaEvalConfig, device: torch.device
    ) -> arena_eval.LoadedArenaModel:
        del config, device
        if name == "baseline":
            return loaded[0]
        assert path == Path("candidate.pt")
        return loaded[1]

    class FakeExtension:
        @staticmethod
        def run_paired_arena(
            games: int,
            seed: int,
            temperature: float,
            candidate_seats: list[int],
            candidate_models: int,
            inference: Any,
        ) -> dict[str, Any]:
            assert (games, seed, temperature, candidate_seats, candidate_models) == (4, 7, 1.25, [0], 1)
            assert callable(inference)
            return {
                "compared_games": 4,
                "candidate_mean_placement": 1.25,
                "baseline_mean_placement": 2.75,
                "games": [{"game_index": 0, "candidate_place": 1}],
            }

    def fake_load_extension(config: arena_eval.ArenaEvalConfig) -> type[FakeExtension]:
        del config
        return FakeExtension

    monkeypatch.setattr(arena_eval, "load_arena_model", fake_load)
    monkeypatch.setattr(arena_eval, "load_arena_extension", fake_load_extension)
    output = tmp_path / "arena.json"
    per_game = tmp_path / "arena.jsonl"
    config = arena_eval.ArenaEvalConfig(
        baseline=Path("baseline.pt"),
        candidates=(Path("candidate.pt"),),
        games=4,
        seed=7,
        temperature=1.25,
        output_path=output,
        per_game_path=per_game,
        tensorboard_dir=tmp_path / "tb",
        weight_source="raw",
        device="cpu",
        extension="fake",
        extension_path=None,
        hidden=arena_eval.DEFAULT_HIDDEN,
        blocks=arena_eval.DEFAULT_BLOCKS,
        bottleneck=arena_eval.DEFAULT_SE_BOTTLENECK,
        residual_profile=arena_eval.RESIDUAL_PROFILE_DEFAULT,
        backbone_profile=arena_eval.BACKBONE_PROFILE_DEFAULT,
        conv_memory_format=arena_eval.CONV_MEMORY_FORMAT_DEFAULT,
    )

    monkeypatch.setattr(arena_eval, "ScalarEventWriter", RecordingScalarWriter)

    summary = arena_eval.run_arena_eval(config)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["baseline"]["global_step"] == 10
    assert written["candidates"][0]["result"]["compared_games"] == 4
    assert summary["candidates"][0]["candidate"] == "candidate"
    assert per_game.read_text(encoding="utf-8").strip() == (
        '{"candidate":"candidate","candidate_index":0,"game":{"candidate_place":1,"game_index":0}}'
    )
    assert {tag for tag, _, _ in RecordingScalarWriter.records} >= {
        "arena/candidate/compared_games",
        "arena/candidate/candidate_mean_placement",
        "arena/candidate/baseline_mean_placement",
    }
    assert all(step == 20 for _, _, step in RecordingScalarWriter.records)
