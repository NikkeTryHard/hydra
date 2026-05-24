#!/usr/bin/env python3
"""Arena evaluation CLI for PyTorch Hydra checkpoints."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from hydra_learner.checkpoint import ModelConfig, load_checkpoint_init_only
from hydra_learner.hydra_logging import ScalarEventWriter, add_scalars
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_DEFAULT,
    BACKBONE_PROFILES,
    CONV_MEMORY_FORMAT_DEFAULT,
    CONV_MEMORY_FORMATS,
    DEFAULT_BLOCKS,
    DEFAULT_HIDDEN,
    DEFAULT_SE_BOTTLENECK,
    OBS_CHANNELS,
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILES,
    TILE_WIDTH,
    HydraPolicyNet,
)

WeightSource = Literal["raw", "ema"]


@dataclass(frozen=True)
class LoadedArenaModel:
    name: str
    path: Path
    model: torch.nn.Module
    global_step: int
    samples_seen: int
    weight_source: WeightSource


@dataclass(frozen=True)
class ArenaEvalConfig:
    baseline: Path
    candidates: tuple[Path, ...]
    games: int
    seed: int
    temperature: float
    output_path: Path
    per_game_path: Path | None
    tensorboard_dir: Path | None
    weight_source: WeightSource
    device: str
    extension: str | None
    extension_path: Path | None
    hidden: int
    blocks: int
    bottleneck: int
    residual_profile: str
    backbone_profile: str
    conv_memory_format: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline .pt checkpoint")
    parser.add_argument(
        "--candidate",
        dest="candidates",
        type=Path,
        action="append",
        required=True,
        help="candidate .pt checkpoint; repeat for multiple candidates",
    )
    parser.add_argument("--games", type=int, default=128, help="paired games per candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output", "--out", dest="output_path", type=Path, required=True, help="JSON summary output path"
    )
    parser.add_argument("--per-game-output", dest="per_game_path", type=Path, help="JSONL per-game output path")
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--weight-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device", default="cuda", help="torch device for checkpoint inference")
    parser.add_argument("--extension", help="importable PyO3 arena module name")
    parser.add_argument("--extension-path", type=Path, help="direct path to PyO3 arena extension")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--bottleneck", type=int, default=DEFAULT_SE_BOTTLENECK)
    parser.add_argument("--residual-profile", choices=RESIDUAL_PROFILES, default=RESIDUAL_PROFILE_DEFAULT)
    parser.add_argument("--backbone-profile", choices=BACKBONE_PROFILES, default=BACKBONE_PROFILE_DEFAULT)
    parser.add_argument("--conv-memory-format", choices=CONV_MEMORY_FORMATS, default=CONV_MEMORY_FORMAT_DEFAULT)
    parser.add_argument("--quiet", action="store_true", help="write JSON without printing it")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> ArenaEvalConfig:
    if args.games < 1:
        raise ValueError("--games must be >= 1")
    if args.temperature <= 0.0 or not math.isfinite(args.temperature):
        raise ValueError("--temperature must be finite and > 0")
    if not args.candidates:
        raise ValueError("provide at least one --candidate")
    if args.extension is not None and args.extension_path is not None:
        raise ValueError("--extension and --extension-path cannot be combined")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but torch.cuda.is_available() is false")
    return ArenaEvalConfig(
        baseline=args.baseline,
        candidates=tuple(args.candidates),
        games=args.games,
        seed=args.seed,
        temperature=args.temperature,
        output_path=args.output_path,
        per_game_path=args.per_game_path,
        tensorboard_dir=args.tensorboard_dir,
        weight_source=cast(WeightSource, args.weight_source),
        device=args.device,
        extension=args.extension,
        extension_path=args.extension_path,
        hidden=args.hidden,
        blocks=args.blocks,
        bottleneck=args.bottleneck,
        residual_profile=args.residual_profile,
        backbone_profile=args.backbone_profile,
        conv_memory_format=args.conv_memory_format,
    )


def default_arena_pyo3_library_path() -> Path:
    env_path = os.environ.get("HYDRA_ARENA_PYO3_LIB")
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parents[3]
    release_path = repo_root / "target" / "release" / "libhydra_arena_pyo3.so"
    if release_path.exists():
        return release_path
    return repo_root / "target" / "debug" / "libhydra_arena_pyo3.so"


def _load_extension_from_path(path: Path) -> Any:
    if not path.exists():
        raise ImportError(
            f"PyO3 arena extension not found at {path}. Build the Rust extension, set HYDRA_ARENA_PYO3_LIB, "
            "pass --extension-path, or pass --extension for an importable module."
        )
    module_name = path.stem.removeprefix("lib")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arena extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_arena_extension(config: ArenaEvalConfig) -> Any:
    if config.extension is not None:
        return importlib.import_module(config.extension)
    return _load_extension_from_path(config.extension_path or default_arena_pyo3_library_path())


def _model_config(config: ArenaEvalConfig) -> ModelConfig:
    return ModelConfig(
        hidden=config.hidden,
        blocks=config.blocks,
        bottleneck=config.bottleneck,
        actions=ACTION_SPACE,
        residual_profile=config.residual_profile,
        backbone_profile=config.backbone_profile,
        conv_memory_format=config.conv_memory_format,
    )


def _config_int(raw: dict[object, object], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"model_config {key} must be int")
    return value


def _model_config_from_checkpoint(raw: dict[object, object]) -> ModelConfig:
    return ModelConfig(
        hidden=_config_int(raw, "hidden"),
        blocks=_config_int(raw, "blocks"),
        bottleneck=_config_int(raw, "bottleneck"),
        actions=_config_int(raw, "actions", ACTION_SPACE),
        residual_profile=str(raw.get("residual_profile", RESIDUAL_PROFILE_DEFAULT)),
        backbone_profile=str(raw.get("backbone_profile", BACKBONE_PROFILE_DEFAULT)),
        conv_memory_format=str(raw.get("conv_memory_format", CONV_MEMORY_FORMAT_DEFAULT)),
    )


def _checkpoint_name(path: Path) -> str:
    stem = path.stem
    parent = path.parent.name
    return stem if not parent else f"{parent}_{stem}"


def load_arena_model(path: Path, *, name: str, config: ArenaEvalConfig, device: torch.device) -> LoadedArenaModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"checkpoint {path} missing model_config")
    model_config = _model_config_from_checkpoint(raw_config)
    model = HydraPolicyNet(
        hidden=model_config.hidden,
        blocks=model_config.blocks,
        bottleneck=model_config.bottleneck,
        actions=model_config.actions,
        residual_profile=model_config.residual_profile,
        backbone_profile=model_config.backbone_profile,
        conv_memory_format=model_config.conv_memory_format,
    )
    init = load_checkpoint_init_only(
        path, model=model, expected_model_config=model_config, weight_source=config.weight_source
    )
    model.to(device=device)
    model.eval()
    return LoadedArenaModel(
        name=name,
        path=path,
        model=model,
        global_step=init.global_step,
        samples_seen=init.samples_seen,
        weight_source=init.weight_source,
    )


def _tensor_from_nested_f32(rows: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        expected = OBS_CHANNELS * TILE_WIDTH
        if tensor.shape[1] != expected:
            raise ValueError(f"obs batch second dimension must be {expected}, got {tensor.shape[1]}")
        return tensor.reshape(tensor.shape[0], OBS_CHANNELS, TILE_WIDTH)
    if tensor.ndim == 3:
        if tuple(tensor.shape[1:]) != (OBS_CHANNELS, TILE_WIDTH):
            raise ValueError(f"obs batch shape must be (N,{OBS_CHANNELS},{TILE_WIDTH}), got {tuple(tensor.shape)}")
        return tensor
    raise ValueError(f"obs batch must be 2D or 3D, got {tensor.ndim}D")


def make_inference_callback(models: list[LoadedArenaModel], device: torch.device) -> Any:
    @torch.inference_mode()
    def infer(obs_batch: Any, legal_batch: Any, model_ids: Any, seat_ids: Any) -> list[list[float]]:
        del seat_ids
        ids = [int(model_id) for model_id in model_ids]
        obs = _tensor_from_nested_f32(obs_batch, device)
        if obs.shape[0] != len(ids):
            raise ValueError(f"model_ids length {len(ids)} does not match obs batch {obs.shape[0]}")
        legal = torch.as_tensor(legal_batch, dtype=torch.bool, device=device)
        if tuple(legal.shape) != (len(ids), ACTION_SPACE):
            raise ValueError(f"legal batch shape must be ({len(ids)},{ACTION_SPACE}), got {tuple(legal.shape)}")
        out = torch.empty((len(ids), ACTION_SPACE), dtype=torch.float32, device=device)
        unique_ids = sorted(set(ids))
        for model_id in unique_ids:
            if model_id < 0 or model_id >= len(models):
                raise ValueError(f"model id {model_id} out of range 0..{len(models) - 1}")
            indices = [idx for idx, value in enumerate(ids) if value == model_id]
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
            logits = models[model_id].model(obs.index_select(0, index_tensor)).policy_logits
            out.index_copy_(0, index_tensor, logits)
        out = out.masked_fill(~legal, -torch.inf)
        return cast(list[list[float]], out.cpu().tolist())

    return infer


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_per_game_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":")) + "\n")


def _extract_per_game(result: dict[str, Any]) -> list[Any] | None:
    for key in ("games", "per_game", "game_results", "per_game_results"):
        value = result.get(key)
        if isinstance(value, list):
            return value
    return None


def _numeric_scalars(result: dict[str, Any]) -> dict[str, object]:
    return {key: value for key, value in result.items() if isinstance(value, bool | int | float)}


_ARENA_NUMERIC_KEYS = (
    "candidate_winrate",
    "baseline_winrate",
    "candidate_avg_rank",
    "baseline_avg_rank",
    "candidate_top2",
    "baseline_top2",
    "candidate_fourth",
    "baseline_fourth",
    "candidate_avg_score",
    "baseline_avg_score",
    "score_delta",
    "pt_delta",
)


def _aggregate_seat_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("seat-rotation results must not be empty")
    games = sum(int(result["games"]) for result in results)
    aggregate: dict[str, Any] = {"games": games, "seat_rotations": len(results)}
    for key in _ARENA_NUMERIC_KEYS:
        aggregate[key] = sum(float(result[key]) * int(result["games"]) for result in results) / games
    return aggregate


def log_arena_scalars(writer: ScalarEventWriter, candidate: LoadedArenaModel, result: dict[str, Any]) -> None:
    step = candidate.global_step
    add_scalars(writer, f"arena/{candidate.name}", _numeric_scalars(result), step, include_status_scalars=True)


def run_arena_eval(config: ArenaEvalConfig) -> dict[str, Any]:
    device = torch.device(config.device)
    baseline = load_arena_model(config.baseline, name="baseline", config=config, device=device)
    candidates = [
        load_arena_model(path, name=_checkpoint_name(path), config=config, device=device) for path in config.candidates
    ]
    extension = load_arena_extension(config)
    run_paired_arena = extension.run_paired_arena
    writer = ScalarEventWriter(config.tensorboard_dir)
    results: list[dict[str, Any]] = []
    per_game_rows: list[Any] = []
    try:
        for candidate_index, candidate in enumerate(candidates):
            models = [baseline, candidate]
            seat_results: list[dict[str, Any]] = []
            for seat in range(4):
                seat_result = cast(
                    dict[str, Any],
                    run_paired_arena(
                        config.games,
                        config.seed + seat * config.games,
                        config.temperature,
                        [seat],
                        1,
                        make_inference_callback(models, device),
                    ),
                )
                if not isinstance(seat_result, dict):
                    raise TypeError(f"run_paired_arena returned {type(seat_result).__name__}, expected dict")
                seat_results.append(seat_result)
            result = _aggregate_seat_results(seat_results)
            per_game = _extract_per_game(result)
            if per_game is not None:
                per_game_rows.extend(
                    {"candidate": candidate.name, "candidate_index": candidate_index, "game": row} for row in per_game
                )
            log_arena_scalars(writer, candidate, result)
            results.append(
                {
                    "candidate": candidate.name,
                    "candidate_path": candidate.path,
                    "global_step": candidate.global_step,
                    "samples_seen": candidate.samples_seen,
                    "result": result,
                }
            )
        writer.flush()
    finally:
        writer.close()
    summary: dict[str, Any] = {
        "config": asdict(config),
        "baseline": {
            "path": baseline.path,
            "global_step": baseline.global_step,
            "samples_seen": baseline.samples_seen,
            "weight_source": baseline.weight_source,
        },
        "candidates": results,
    }
    _write_json(config.output_path, summary)
    if config.per_game_path is not None and per_game_rows:
        _write_per_game_jsonl(config.per_game_path, per_game_rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_args(args)
    summary = run_arena_eval(config)
    if not args.quiet:
        print(json.dumps(_json_ready(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
