from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from hydra_learner.data.shard_contracts import (
    BC_BASE_RECORD_SIZE,
    BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
    BC_SHARD_HEADER_SIZE,
    BC_SHARD_LAYOUT_VERSION,
    BC_SHARD_MAGIC,
    BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION,
    FLAG_DELTA_Q,
    FLAG_EXIT,
    FLAG_SAFETY_RESIDUAL,
)
from hydra_learner.ppo.compat import _config_digest
from hydra_learner.ppo.config import PpoControlConfig
from hydra_learner.ppo.smoke import RustDecisionRow, RustGameRollout

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TINY_SHARD_DIR = FIXTURES_DIR / "tiny-shard"
TINY_SHARD_MANIFEST = TINY_SHARD_DIR / "manifest.json"
TINY_CHECKPOINT_ROOT = Path("/fixtures/ppo/tiny-source")
TINY_RUN_ROOT = TINY_CHECKPOINT_ROOT / "stages" / "T1_ppo_control" / "runs" / "latest_run"
ACTION_SPACE = 46
NUM_CHANNELS = 192
OBS_SIZE = 192 * 34
TILE_WIDTH = 34


def tiny_ppo_rollout() -> RustGameRollout:
    rows: list[RustDecisionRow] = []
    for index, player_id in enumerate((0, 1, 2, 3)):
        obs = torch.zeros((192, TILE_WIDTH), dtype=torch.float32)
        obs[player_id, index] = 1.0
        obs[32 + player_id, index + 1] = 0.25 * float(index + 1)
        legal_mask = torch.zeros((ACTION_SPACE,), dtype=torch.bool)
        legal_mask[index] = True
        legal_mask[index + 4] = True
        rows.append(
            RustDecisionRow(
                obs=obs,
                legal_mask=legal_mask,
                player_id=player_id,
                seat_id=player_id,
                game_id=17,
                turn=index,
                action=index,
                legal_count=2,
            )
        )
    return RustGameRollout(
        tuple(rows), final_scores=(30000, 26000, 24000, 20000), placements=(0, 1, 2, 3), seed=20260604
    )


def tiny_checkpoint_ppo_control_config() -> PpoControlConfig:
    output_dir = TINY_RUN_ROOT
    return PpoControlConfig(
        init_checkpoint=TINY_CHECKPOINT_ROOT / "logs" / "checkpoints" / "best.pt",
        output_dir=output_dir,
        steps=None,
        games_per_update=1024,
        seed=0,
        device="cuda:0",
        temperature=1.0,
        arena_batch_decisions=3072,
        arena_threads=0,
        extension_path=None,
        hidden=384,
        blocks=16,
        bottleneck=96,
        residual_profile="mish_se",
        backbone_profile="conv2d_local3",
        conv_memory_format="contiguous",
        lr=0.0001,
        min_lr=1.0e-6,
        lr_warmup_samples=0,
        lr_decay_samples=1_000_000_000,
        grad_clip_norm=1.0,
        microbatch_size=768,
        epochs=3,
        target_kl=0.005,
        weight_decay=9.999999747378752e-06,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        adamw_fused="on",
        adamw_foreach="auto",
        bc_kl_reverse_coef=0.0,
        entropy_alpha=1.0e-3,
        entropy_beta=1.0e-2,
        entropy_alpha_max=0.05,
        log_every_steps=1,
        checkpoint_every_steps=250,
        keep_step_checkpoints=False,
        resume=output_dir / "checkpoints" / "latest.pt",
        tensorboard_dir=output_dir / "tensorboard",
        quiet=True,
        rollout_inference="torch-callback",
        ppo_pipeline_depth=0,
        rollout_device=None,
    )


TINY_CHECKPOINT_CONFIG_DIGEST = _config_digest(tiny_checkpoint_ppo_control_config())


def tiny_run_local_paths() -> tuple[Path, Path | None, Path | None]:
    output_dir = TINY_RUN_ROOT / "resume-smoke"
    return output_dir, output_dir / "checkpoints" / "latest.pt", output_dir / "tensorboard"


def write_tiny_shard_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard = root / "train-00000.hybc"
    flags = FLAG_SAFETY_RESIDUAL | FLAG_EXIT | FLAG_DELTA_Q
    record_size = BC_BASE_RECORD_SIZE + 3 * (ACTION_SPACE * 4 + ((ACTION_SPACE + 7) // 8))
    sample_count = 1
    byte_len = BC_SHARD_HEADER_SIZE + sample_count * record_size
    header = struct.pack(
        "<8sIIIIIQIIIQIIQQ",
        BC_SHARD_MAGIC,
        BC_SHARD_VERSION,
        BC_SHARD_HEADER_SIZE,
        record_size,
        0,
        0,
        sample_count,
        NUM_CHANNELS,
        TILE_WIDTH,
        ACTION_SPACE,
        0,
        flags,
        BC_SHARD_LAYOUT_VERSION,
        0,
        0,
    )
    shard.write_bytes(header + _tiny_shard_record())
    manifest = _tiny_shard_manifest(
        shard.name, sample_count=sample_count, byte_len=byte_len, flags=flags, record_size=record_size
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest_path


def _tiny_shard_record() -> bytes:
    record = bytearray(BC_BASE_RECORD_SIZE)
    for start in (51, 197, 343, 489):
        for tile in range(TILE_WIDTH):
            record[start + tile * 4 : start + (tile + 1) * 4] = (0xFFFF_FFFF).to_bytes(4, "little")
    record[757] = 0x01
    record[1675] = 3
    record[1676 : 1676 + ((ACTION_SPACE + 7) // 8)] = _packed_action_mask((3,))
    record[1682:1686] = struct.pack("<i", 12000)
    record[1686] = 7
    record[1687 : 1687 + 16] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype="<f4").tobytes()
    record[1703] = 1
    record[1704] = 0b101
    record[1705:1708] = bytes((3, 8, TILE_WIDTH))
    record[1708 : 1708 + 13] = _packed_spatial_mask((3, 34 + 8))
    record[1721 : 1721 + 13] = _packed_spatial_mask((3, 34 + 8))
    return (
        bytes(record)
        + _optional_pair({5: 0.05}, (4,))
        + _optional_pair({6: 0.75}, (6,))
        + _optional_pair({7: -1.25}, (7,))
    )


def _tiny_shard_manifest(
    file_name: str, *, sample_count: int, byte_len: int, flags: int, record_size: int
) -> dict[str, object]:
    return {
        "manifest_version": BC_SHARD_MANIFEST_VERSION,
        "shard_version": BC_SHARD_VERSION,
        "shard_header_size": BC_SHARD_HEADER_SIZE,
        "base_record_size": BC_BASE_RECORD_SIZE,
        "max_record_size": BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        "obs_size": OBS_SIZE,
        "num_channels": NUM_CHANNELS,
        "action_space": ACTION_SPACE,
        "storage_layout": "compact",
        "split_mode": "train",
        "totals": {"sample_count": sample_count, "shard_count": 1},
        "splits": [
            {
                "split": "train",
                "shard_count": 1,
                "sample_count": sample_count,
                "feature_flags": flags,
                "record_size": record_size,
                "shards": [
                    {
                        "split": "train",
                        "shard_index": 0,
                        "file_name": file_name,
                        "sample_count": sample_count,
                        "first_sample_index": 0,
                        "byte_len": byte_len,
                        "feature_flags": flags,
                        "record_size": record_size,
                    }
                ],
            }
        ],
    }


def _optional_pair(target: dict[int, float], mask_indices: tuple[int, ...]) -> bytes:
    values = np.zeros((ACTION_SPACE,), dtype="<f4")
    for action, value in target.items():
        values[action] = value
    return values.tobytes() + _packed_action_mask(mask_indices)


def _packed_action_mask(indices: tuple[int, ...]) -> bytes:
    mask = bytearray((ACTION_SPACE + 7) // 8)
    for action in indices:
        mask[action // 8] |= 1 << (action % 8)
    return bytes(mask)


def _packed_spatial_mask(indices: tuple[int, ...]) -> bytes:
    mask = bytearray(13)
    for index in indices:
        mask[index // 8] |= 1 << (index % 8)
    return bytes(mask)
