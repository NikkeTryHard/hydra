from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, override

import pytest
import torch

from hydra_learner import arena_eval
from hydra_learner.export_inference import ExportResult


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


def _arena_config(
    tmp_path: Path,
    *,
    baseline: Path,
    candidates: tuple[Path, ...],
    weight_source: arena_eval.WeightSource = "raw",
    rust_native: bool = True,
) -> arena_eval.ArenaEvalConfig:
    return arena_eval.ArenaEvalConfig(
        baseline=baseline,
        candidates=candidates,
        games=4,
        seed=7,
        temperature=1.25,
        output_path=tmp_path / "arena.json",
        per_game_path=None,
        tensorboard_dir=None,
        weight_source=weight_source,
        device="cpu",
        extension="fake",
        extension_path=None,
        arena_batch_decisions=256,
        rust_native=rust_native,
        arena_threads=0,
        hidden=arena_eval.DEFAULT_HIDDEN,
        blocks=arena_eval.DEFAULT_BLOCKS,
        bottleneck=arena_eval.DEFAULT_SE_BOTTLENECK,
        residual_profile=arena_eval.RESIDUAL_PROFILE_DEFAULT,
        backbone_profile=arena_eval.BACKBONE_PROFILE_DEFAULT,
        conv_memory_format=arena_eval.CONV_MEMORY_FORMAT_DEFAULT,
    )


def test_arena_parse_args_defaults_to_native_onnx() -> None:
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
    assert config.rust_native


def test_arena_parse_args_can_use_legacy_python_checkpoints() -> None:
    args = arena_eval.parse_args(
        [
            "--baseline",
            "baseline.pt",
            "--candidate",
            "candidate.pt",
            "--output",
            "arena.json",
            "--python-checkpoints",
        ]
    )

    config = arena_eval.validate_args(args)

    assert not config.rust_native


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


def test_arena_resolves_native_export_dir(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "policy.json").write_text("{}\n", encoding="utf-8")
    (export_dir / "policy.onnx").write_bytes(b"onnx")
    config = _arena_config(tmp_path, baseline=export_dir, candidates=(export_dir,))

    assert arena_eval.resolve_native_arena_path(export_dir, config) == export_dir


def test_arena_auto_exports_pt_checkpoint_for_native_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = _arena_config(tmp_path, baseline=checkpoint, candidates=(checkpoint,), weight_source="ema")
    calls: list[arena_eval.ExportConfig] = []

    def fake_write(export_config: arena_eval.ExportConfig, **kwargs: Any) -> ExportResult:
        calls.append(export_config)
        del kwargs
        export_config.output_dir.mkdir(parents=True)
        metadata_path = export_config.output_dir / "policy.json"
        artifact_path = export_config.output_dir / "policy.onnx"
        fixture_path = export_config.output_dir / "parity_fixture.safetensors"
        metadata_path.write_text("{}\n", encoding="utf-8")
        artifact_path.write_bytes(b"onnx")
        fixture_path.write_bytes(b"fixture")
        return ExportResult(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            fixture_path=fixture_path,
            source_checkpoint_sha256="hash",
            global_step=1,
            samples_seen=2,
            weight_source="ema",
        )

    def fake_load_export_policy(export_config: arena_eval.ExportConfig) -> tuple[None, None, None, None, None]:
        del export_config
        return None, None, None, None, None

    monkeypatch.setattr(arena_eval, "load_export_policy", fake_load_export_policy)
    monkeypatch.setattr(arena_eval, "write_exported_policy", fake_write)

    resolved = arena_eval.resolve_native_arena_path(checkpoint, config)

    assert calls == [
        arena_eval.ExportConfig(
            checkpoint=checkpoint,
            weight_source="ema",
            output_dir=resolved,
            fixture_obs=None,
            num_fixture_rows=8,
            max_batch=4096,
            opset_version=18,
        )
    ]
    assert resolved.name == "ema"
    assert resolved.parent.name.startswith("latest-")


def test_arena_native_path_rejects_invalid_inputs(tmp_path: Path) -> None:
    missing_policy = tmp_path / "missing-policy"
    missing_policy.mkdir()
    (missing_policy / "policy.json").write_text("{}\n", encoding="utf-8")
    config = _arena_config(tmp_path, baseline=missing_policy, candidates=(missing_policy,))

    with pytest.raises(ValueError, match=r"must contain policy\.json and policy\.onnx"):
        arena_eval.resolve_native_arena_path(missing_policy, config)

    with pytest.raises(ValueError, match=r"must be an ONNX export directory or \.pt checkpoint"):
        arena_eval.resolve_native_arena_path(tmp_path / "weights.bin", config)

    with pytest.raises(ValueError, match="does not exist"):
        arena_eval.resolve_native_arena_path(tmp_path / "missing.pt", config)


def test_arena_native_run_uses_resolved_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_input = tmp_path / "baseline.pt"
    candidate_input = tmp_path / "candidate.pt"
    baseline_export = tmp_path / "baseline-export"
    candidate_export = tmp_path / "candidate-export"
    baseline_input.write_bytes(b"baseline")
    candidate_input.write_bytes(b"candidate")
    for export_dir, step in ((baseline_export, 10), (candidate_export, 20)):
        export_dir.mkdir()
        (export_dir / "policy.onnx").write_bytes(b"onnx")
        (export_dir / "policy.json").write_text(
            json.dumps(
                {
                    "checkpoint_global_step": step,
                    "checkpoint_samples_seen": step * 10,
                    "weight_source": "raw",
                }
            ),
            encoding="utf-8",
        )
    calls: list[tuple[list[Path], Path]] = []

    class FakeExtension:
        @staticmethod
        def run_paired_arena_rust_native(
            games: int,
            seed: int,
            temperature: float,
            candidate_paths: list[Path],
            baseline_path: Path,
            batch_decisions: int,
            device: str,
            arena_threads: int,
        ) -> dict[str, Any]:
            del games, seed, temperature, batch_decisions, device, arena_threads
            calls.append((candidate_paths, baseline_path))
            return {
                "games": 4,
                "candidate_winrate": 0.25,
                "baseline_winrate": 0.75,
                "candidate_avg_rank": 2.25,
                "baseline_avg_rank": 2.75,
                "candidate_top2": 0.5,
                "baseline_top2": 0.25,
                "candidate_fourth": 0.0,
                "baseline_fourth": 0.25,
                "candidate_avg_score": 26000.0,
                "baseline_avg_score": 24666.0,
                "score_delta": 1334.0,
                "pt_delta": 1.334,
            }

    def fake_resolve_native_arena_path(path: Path, config: arena_eval.ArenaEvalConfig) -> Path:
        del config
        return baseline_export if path == baseline_input else candidate_export

    monkeypatch.setattr(arena_eval, "resolve_native_arena_path", fake_resolve_native_arena_path)
    monkeypatch.setattr(arena_eval, "ScalarEventWriter", RecordingScalarWriter)
    config = _arena_config(tmp_path, baseline=baseline_input, candidates=(candidate_input,))

    summary = arena_eval.run_arena_eval_rust_native(config, FakeExtension)

    assert calls == [([candidate_export], baseline_export)]
    assert summary["baseline"]["global_step"] == 10
    assert summary["candidates"][0]["candidate_path"] == candidate_export


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

    calls: list[tuple[int, int, float, list[int], int]] = []

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
            calls.append((games, seed, temperature, candidate_seats, candidate_models))
            assert callable(inference)
            return {
                "games": games,
                "candidate_winrate": 0.25,
                "baseline_winrate": 0.75,
                "candidate_avg_rank": 2.25,
                "baseline_avg_rank": 2.75,
                "candidate_top2": 0.5,
                "baseline_top2": 0.25,
                "candidate_fourth": 0.0,
                "baseline_fourth": 0.25,
                "candidate_avg_score": 26000.0,
                "baseline_avg_score": 24666.0,
                "score_delta": 1334.0,
                "pt_delta": 1.334,
            }

    def fake_load_extension(config: arena_eval.ArenaEvalConfig) -> type[FakeExtension]:
        del config
        return FakeExtension

    monkeypatch.setattr(arena_eval, "load_arena_model", fake_load)
    monkeypatch.setattr(arena_eval, "load_arena_extension", fake_load_extension)
    output = tmp_path / "arena.json"
    per_game = tmp_path / "arena.jsonl"
    config = _arena_config(
        tmp_path,
        baseline=Path("baseline.pt"),
        candidates=(Path("candidate.pt"),),
        rust_native=False,
    )
    config = arena_eval.ArenaEvalConfig(
        baseline=config.baseline,
        candidates=config.candidates,
        games=config.games,
        seed=config.seed,
        temperature=config.temperature,
        output_path=config.output_path,
        per_game_path=per_game,
        tensorboard_dir=tmp_path / "tb",
        weight_source=config.weight_source,
        device=config.device,
        extension=config.extension,
        extension_path=config.extension_path,
        arena_batch_decisions=config.arena_batch_decisions,
        rust_native=config.rust_native,
        arena_threads=config.arena_threads,
        hidden=config.hidden,
        blocks=config.blocks,
        bottleneck=config.bottleneck,
        residual_profile=config.residual_profile,
        backbone_profile=config.backbone_profile,
        conv_memory_format=config.conv_memory_format,
    )

    monkeypatch.setattr(arena_eval, "ScalarEventWriter", RecordingScalarWriter)

    summary = arena_eval.run_arena_eval(config)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["baseline"]["global_step"] == 10
    assert written["candidates"][0]["result"]["games"] == 16
    assert summary["candidates"][0]["candidate"] == "candidate"
    assert not per_game.exists()
    assert calls == [
        (4, 7, 1.25, [0], 1),
        (4, 11, 1.25, [1], 1),
        (4, 15, 1.25, [2], 1),
        (4, 19, 1.25, [3], 1),
    ]
    assert {tag for tag, _, _ in RecordingScalarWriter.records} >= {
        "arena/candidate/games",
        "arena/candidate/candidate_avg_rank",
        "arena/candidate/baseline_avg_rank",
    }
    assert all(step == 20 for _, _, step in RecordingScalarWriter.records)


def test_arena_main_prints_summary_json_when_not_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "arena.json"

    def fake_run_arena_eval(config: arena_eval.ArenaEvalConfig) -> dict[str, Any]:
        summary = {
            "config": {"games": config.games, "seed": config.seed},
            "baseline": {"path": config.baseline},
            "candidates": [
                {
                    "candidate": "candidate",
                    "candidate_path": config.candidates[0],
                    "result": {"games": config.games, "candidate_winrate": 0.25},
                }
            ],
        }
        arena_eval._write_json(config.output_path, summary)
        return summary

    monkeypatch.setattr(arena_eval, "run_arena_eval", fake_run_arena_eval)

    exit_code = arena_eval.main(
        [
            "--baseline",
            "baseline.pt",
            "--candidate",
            "candidate.pt",
            "--games",
            "4",
            "--seed",
            "17",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--extension",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    printed = json.loads(captured.out)
    assert printed["config"] == {"games": 4, "seed": 17}
    assert printed["baseline"]["path"] == "baseline.pt"
    assert printed["candidates"][0]["candidate_path"] == "candidate.pt"
    assert captured.err == ""
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == printed


def test_arena_main_quiet_suppresses_summary_stdout_but_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "arena.json"

    def fake_run_arena_eval(config: arena_eval.ArenaEvalConfig) -> dict[str, Any]:
        summary = {
            "config": {"games": config.games, "seed": config.seed},
            "baseline": {"path": config.baseline},
            "candidates": [
                {
                    "candidate": "candidate",
                    "candidate_path": config.candidates[0],
                    "result": {"games": config.games, "candidate_winrate": 0.25},
                }
            ],
        }
        arena_eval._write_json(config.output_path, summary)
        return summary

    monkeypatch.setattr(arena_eval, "run_arena_eval", fake_run_arena_eval)

    exit_code = arena_eval.main(
        [
            "--baseline",
            "baseline.pt",
            "--candidate",
            "candidate.pt",
            "--games",
            "4",
            "--seed",
            "17",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--extension",
            "fake",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["config"] == {"games": 4, "seed": 17}
    assert written["baseline"]["path"] == "baseline.pt"
    assert written["candidates"][0]["candidate_path"] == "candidate.pt"
