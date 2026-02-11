"""
VocNet configuration loader.

Reads experiment configuration from a TOML file. The config path can be
overridden via the VOCNET_CFG environment variable, defaulting to
``config.toml`` in the working directory.

Configuration sections:
    [model]       — architecture hyperparameters (channels, depth, etc.)
    [training]    — optimizer, LR schedule, batch size, gradient clipping
    [data]        — dataset paths, spectrogram parameters (n_fft, hop_length)
    [env]         — recording session environment settings for online training
    [distributed] — parameter server addresses and worker topology
"""

import os
import sys
import tomllib
import logging

log = logging.getLogger("vocnet.config")

config_file = os.environ.get("VOCNET_CFG", "config.toml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preferring override values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Default configuration — sensible values for a single-GPU training run
# on 16 kHz mono recordings with 128-bin mel spectrograms.
_defaults = {
    "model": {
        "conv_channels": 128,
        "num_blocks": 12,
        "se_reduction": 8,
        "obs_shape": [1, 128, 64],  # [C, freq_bins, time_frames]
        "num_actions": 47,  # vocalization type classes
        "grp_hidden": 256,
        "grp_layers": 2,
    },
    "training": {
        "batch_size": 256,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "epochs": 120,
        "warmup_steps": 2000,
        "gamma": 0.999,  # reward discount factor
        "lambda_": 0.95,  # GAE lambda
        "ppo_clip": 0.2,
        "entropy_coef": 0.01,
    },
    "data": {
        "archive_dir": "./data/recordings",
        "sample_rate": 16000,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "max_session_length": 300,  # max frames per recording session
    },
    "env": {
        "num_parallel": 64,
        "boltzmann_temp": 1.0,
        "eval_episodes": 1000,
    },
    "distributed": {
        "server_host": "127.0.0.1",
        "server_port": 52710,
        "num_workers": 4,
    },
}


def _load_config(path: str) -> dict:
    """Load and validate configuration from a TOML file."""
    cfg = _defaults.copy()

    if os.path.isfile(path):
        with open(path, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg = _deep_merge(cfg, user_cfg)
        log.info("Loaded config from %s", path)
    else:
        log.warning(
            "Config file %s not found; using defaults. Set VOCNET_CFG to override.",
            path,
        )

    # Validate a few critical invariants early so we get clear error
    # messages instead of cryptic shape mismatches deep in the model.
    obs = cfg["model"]["obs_shape"]
    assert len(obs) == 3, f"obs_shape must be [C, H, W], got {obs}"
    assert cfg["training"]["ppo_clip"] > 0, "ppo_clip must be positive"

    return cfg


config = _load_config(config_file)
