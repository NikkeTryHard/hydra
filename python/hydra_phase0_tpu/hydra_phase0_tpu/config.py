from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export_root: Path
    buffer_games: int = 50_000
    buffer_samples: int = 32_768

    @model_validator(mode="after")
    def validate_positive_buffers(self) -> Self:
        if self.buffer_games < 1:
            raise ValueError("buffer_games must be >= 1")
        if self.buffer_samples < 1:
            raise ValueError("buffer_samples must be >= 1")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    seed: int
    num_epochs: int
    batch_size: int = 2048
    microbatch_size: int | None = None
    validation_microbatch_size: int | None = None
    max_validation_samples: int | None = 8_192
    log_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 500
    validate_every_n_steps: int = 500
    validation_every_n_epochs: int = 1
    resume_from: Path | None = None
    init_weights_from: Path | None = None
    smoke_test: bool = False
    required_backend: Literal["any", "cpu", "gpu", "tpu"] = "any"

    @model_validator(mode="after")
    def apply_batch_defaults(self) -> Self:
        if self.microbatch_size is None:
            self.microbatch_size = self.batch_size
        if self.validation_microbatch_size is None:
            self.validation_microbatch_size = self.microbatch_size
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.microbatch_size < 1:
            raise ValueError("microbatch_size must be >= 1")
        if self.validation_microbatch_size < 1:
            raise ValueError("validation_microbatch_size must be >= 1")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be >= 1")
        if self.checkpoint_every_n_steps < 1:
            raise ValueError("checkpoint_every_n_steps must be >= 1")
        if self.validate_every_n_steps < 1:
            raise ValueError("validate_every_n_steps must be >= 1")
        if self.validation_every_n_epochs < 1:
            raise ValueError("validation_every_n_epochs must be >= 1")
        if self.max_validation_samples is not None and self.max_validation_samples < 1:
            raise ValueError("max_validation_samples must be >= 1 when provided")
        return self


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 2.5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1_000

    @model_validator(mode="after")
    def validate_optimizer(self) -> Self:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be > 0")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_blocks: int = 24
    input_channels: int = 192
    hidden_channels: int = 256
    num_groups: int = 32
    se_bottleneck: int = 64
    action_space: int = 46
    score_bins: int = 64
    num_opponents: int = 3
    grp_classes: int = 24
    num_belief_components: int = 4
    opponent_hand_type_classes: int = 8
    compute_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    param_dtype: Literal["float32"] = "float32"

    @model_validator(mode="after")
    def validate_model_dimensions(self) -> Self:
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        if self.num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if self.se_bottleneck < 1:
            raise ValueError("se_bottleneck must be >= 1")
        if self.action_space < 1:
            raise ValueError("action_space must be >= 1")
        if self.score_bins < 1:
            raise ValueError("score_bins must be >= 1")
        if self.num_opponents < 1:
            raise ValueError("num_opponents must be >= 1")
        if self.grp_classes < 1:
            raise ValueError("grp_classes must be >= 1")
        if self.num_belief_components < 1:
            raise ValueError("num_belief_components must be >= 1")
        if self.opponent_hand_type_classes < 1:
            raise ValueError("opponent_hand_type_classes must be >= 1")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    run: RunConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    return ExperimentConfig.model_validate(raw)
