from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


EXPECTED_SCHEMA_VERSION = "hydra_bc_phase0_v1"
EXPECTED_EXPORT_SEMANTICS = "hydra_bc_phase0_v1"
EXPECTED_ENCODER_CONTRACT = "192x34"
EXPECTED_ACTION_SPACE = 46


def _validate_absent_rows_zeroed(
    present: np.ndarray,
    payload: np.ndarray,
    *,
    shard_name: str,
    field_name: str,
) -> None:
    absent_rows = present == 0
    if np.any(payload[absent_rows] != 0):
        raise ValueError(
            f"{field_name} must be zero-filled when absent for {shard_name}"
        )


@dataclass(frozen=True)
class ShardRecord:
    split: str
    shard_name: str
    game_count: int
    sample_count: int
    root: Path
    file_hashes: dict[str, str]


class ExportDataset:
    def __init__(self, export_root: Path) -> None:
        self.export_root = export_root
        self.manifest = json.loads((export_root / "manifest.json").read_text())
        self._validate_manifest()

    @property
    def manifest_fingerprint(self) -> str:
        return self.manifest["manifest_fingerprint"]

    def _validate_manifest(self) -> None:
        if "schema_version" not in self.manifest:
            raise ValueError("manifest missing required field schema_version")
        if "export_semantics" not in self.manifest:
            raise ValueError("manifest missing required field export_semantics")
        if "encoder_contract" not in self.manifest:
            raise ValueError("manifest missing required field encoder_contract")
        if "action_space" not in self.manifest:
            raise ValueError("manifest missing required field action_space")

        schema_version = self.manifest["schema_version"]
        export_semantics = self.manifest["export_semantics"]
        encoder_contract = self.manifest["encoder_contract"]
        action_space = self.manifest["action_space"]
        if schema_version != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version}; expected {EXPECTED_SCHEMA_VERSION}"
            )
        if export_semantics != EXPECTED_EXPORT_SEMANTICS:
            raise ValueError(
                f"unsupported export_semantics {export_semantics}; expected {EXPECTED_EXPORT_SEMANTICS}"
            )
        if encoder_contract != EXPECTED_ENCODER_CONTRACT:
            raise ValueError(
                f"unsupported encoder_contract {encoder_contract}; expected {EXPECTED_ENCODER_CONTRACT}"
            )
        if int(action_space) != EXPECTED_ACTION_SPACE:
            raise ValueError(
                f"unsupported action_space {action_space}; expected {EXPECTED_ACTION_SPACE}"
            )

    def shards(self, split: str) -> list[ShardRecord]:
        out: list[ShardRecord] = []
        for shard in self.manifest["shards"]:
            if shard["split"] != split:
                continue
            file_hashes = dict(shard.get("hashes", {}).get("files", {}))
            if not file_hashes:
                raise ValueError(
                    f"shard {shard['shard_name']} missing required file hashes in manifest"
                )
            root = self.export_root / split / shard["shard_name"]
            out.append(
                ShardRecord(
                    split=split,
                    shard_name=shard["shard_name"],
                    game_count=shard["game_count"],
                    sample_count=shard["sample_count"],
                    root=root,
                    file_hashes=file_hashes,
                )
            )
        return out


def _verify_shard_hashes(shard: ShardRecord, file_names: list[str]) -> None:
    expected = shard.file_hashes
    if not expected:
        return
    # Allow skipping verification when manifest contains a single sentinel entry
    if list(expected.values()) == ["skip"]:
        return
    expected_names = set(expected.keys())
    actual_names = set(file_names)
    if expected_names != actual_names:
        raise ValueError(
            f"shard file set mismatch for {shard.shard_name}: expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    for file_name in file_names:
        digest = hashlib.sha256((shard.root / file_name).read_bytes()).hexdigest()
        expected_digest = expected[file_name]
        if digest != expected_digest:
            raise ValueError(
                f"hash mismatch for {shard.shard_name}/{file_name}: expected {expected_digest}, got {digest}"
            )


def load_shard_arrays(shard: ShardRecord) -> dict[str, np.ndarray]:
    hashed_file_names = [
        "shard.json",
        "game_identities.txt",
        "game_sample_offsets.npy",
        "obs.npy",
        "action.npy",
        "legal_mask.npy",
        "score_delta.npy",
        "value_target.npy",
        "grp_target.npy",
        "tenpai_target.npy",
        "danger_target.npy",
        "danger_mask.npy",
        "opp_next_target.npy",
        "score_pdf_target.npy",
        "score_cdf_target.npy",
        "oracle_target.npy",
        "oracle_target_present.npy",
        "safety_residual_target.npy",
        "safety_residual_present.npy",
        "safety_residual_mask.npy",
        "belief_fields_target.npy",
        "belief_fields_present.npy",
        "mixture_weight_target.npy",
        "mixture_weight_present.npy",
    ]
    npy_file_names = [
        "game_sample_offsets.npy",
        "obs.npy",
        "action.npy",
        "legal_mask.npy",
        "score_delta.npy",
        "value_target.npy",
        "grp_target.npy",
        "tenpai_target.npy",
        "danger_target.npy",
        "danger_mask.npy",
        "opp_next_target.npy",
        "score_pdf_target.npy",
        "score_cdf_target.npy",
        "oracle_target.npy",
        "oracle_target_present.npy",
        "safety_residual_target.npy",
        "safety_residual_present.npy",
        "safety_residual_mask.npy",
        "belief_fields_target.npy",
        "belief_fields_present.npy",
        "mixture_weight_target.npy",
        "mixture_weight_present.npy",
    ]
    _verify_shard_hashes(shard, hashed_file_names)
    arrays = {name[:-4]: np.load(shard.root / name) for name in npy_file_names}
    sample_count = shard.sample_count
    offsets = arrays["game_sample_offsets"]
    if offsets.ndim != 1:
        raise ValueError(
            f"game_sample_offsets rank mismatch for {shard.shard_name}: {offsets.shape}"
        )
    if offsets.size < 2:
        raise ValueError(
            f"game_sample_offsets must contain at least start/end for {shard.shard_name}"
        )
    if int(offsets[0]) != 0:
        raise ValueError(
            f"game_sample_offsets must start at 0 for {shard.shard_name}: {offsets.tolist()}"
        )
    if int(offsets[-1]) != sample_count:
        raise ValueError(
            f"game_sample_offsets must end at sample_count for {shard.shard_name}: expected {sample_count}, got {int(offsets[-1])}"
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"game_sample_offsets must be nondecreasing for {shard.shard_name}: {offsets.tolist()}"
        )
    if arrays["obs"].shape != (sample_count, 192, 34):
        raise ValueError(
            f"obs shape mismatch for {shard.shard_name}: {arrays['obs'].shape}"
        )
    if arrays["action"].shape != (sample_count,):
        raise ValueError(
            f"action shape mismatch for {shard.shard_name}: {arrays['action'].shape}"
        )
    if np.any(arrays["action"] < 0) or np.any(
        arrays["action"] >= EXPECTED_ACTION_SPACE
    ):
        raise ValueError(
            f"action values out of range for {shard.shard_name}: expected [0, {EXPECTED_ACTION_SPACE - 1}]"
        )
    if arrays["legal_mask"].shape != (sample_count, 46):
        raise ValueError(
            f"legal_mask shape mismatch for {shard.shard_name}: {arrays['legal_mask'].shape}"
        )
    if np.any(np.sum(arrays["legal_mask"] > 0.0, axis=1) == 0):
        raise ValueError(f"legal_mask contains empty legal rows for {shard.shard_name}")
    row_indices = np.arange(sample_count)
    if np.any(
        arrays["legal_mask"][row_indices, arrays["action"].astype(np.int64)] <= 0.0
    ):
        raise ValueError(
            f"action must be legal under legal_mask for {shard.shard_name}"
        )
    if arrays["score_delta"].shape != (sample_count,):
        raise ValueError(
            f"score_delta shape mismatch for {shard.shard_name}: {arrays['score_delta'].shape}"
        )
    if arrays["value_target"].shape != (sample_count,):
        raise ValueError(
            f"value_target shape mismatch for {shard.shard_name}: {arrays['value_target'].shape}"
        )
    if arrays["grp_target"].shape != (sample_count, 24):
        raise ValueError(
            f"grp_target shape mismatch for {shard.shard_name}: {arrays['grp_target'].shape}"
        )
    if arrays["tenpai_target"].shape != (sample_count, 3):
        raise ValueError(
            f"tenpai_target shape mismatch for {shard.shard_name}: {arrays['tenpai_target'].shape}"
        )
    if arrays["danger_target"].shape != (sample_count, 3, 34):
        raise ValueError(
            f"danger_target shape mismatch for {shard.shard_name}: {arrays['danger_target'].shape}"
        )
    if arrays["danger_mask"].shape != (sample_count, 3, 34):
        raise ValueError(
            f"danger_mask shape mismatch for {shard.shard_name}: {arrays['danger_mask'].shape}"
        )
    if arrays["opp_next_target"].shape != (sample_count, 3, 34):
        raise ValueError(
            f"opp_next_target shape mismatch for {shard.shard_name}: {arrays['opp_next_target'].shape}"
        )
    if arrays["score_pdf_target"].shape != (sample_count, 64):
        raise ValueError(
            f"score_pdf_target shape mismatch for {shard.shard_name}: {arrays['score_pdf_target'].shape}"
        )
    if arrays["score_cdf_target"].shape != (sample_count, 64):
        raise ValueError(
            f"score_cdf_target shape mismatch for {shard.shard_name}: {arrays['score_cdf_target'].shape}"
        )
    if arrays["oracle_target"].shape != (sample_count, 4):
        raise ValueError(
            f"oracle_target shape mismatch for {shard.shard_name}: {arrays['oracle_target'].shape}"
        )
    if arrays["oracle_target_present"].shape != (sample_count,):
        raise ValueError(
            f"oracle_target_present shape mismatch for {shard.shard_name}: {arrays['oracle_target_present'].shape}"
        )
    _validate_absent_rows_zeroed(
        arrays["oracle_target_present"],
        arrays["oracle_target"],
        shard_name=shard.shard_name,
        field_name="oracle_target",
    )
    if arrays["safety_residual_target"].shape != (sample_count, 46):
        raise ValueError(
            f"safety_residual_target shape mismatch for {shard.shard_name}: {arrays['safety_residual_target'].shape}"
        )
    if arrays["safety_residual_present"].shape != (sample_count,):
        raise ValueError(
            f"safety_residual_present shape mismatch for {shard.shard_name}: {arrays['safety_residual_present'].shape}"
        )
    if arrays["safety_residual_mask"].shape != (sample_count, 46):
        raise ValueError(
            f"safety_residual_mask shape mismatch for {shard.shard_name}: {arrays['safety_residual_mask'].shape}"
        )
    _validate_absent_rows_zeroed(
        arrays["safety_residual_present"],
        arrays["safety_residual_target"],
        shard_name=shard.shard_name,
        field_name="safety_residual_target",
    )
    _validate_absent_rows_zeroed(
        arrays["safety_residual_present"],
        arrays["safety_residual_mask"],
        shard_name=shard.shard_name,
        field_name="safety_residual_mask",
    )
    if arrays["belief_fields_target"].shape != (sample_count, 16, 34):
        raise ValueError(
            f"belief_fields_target shape mismatch for {shard.shard_name}: {arrays['belief_fields_target'].shape}"
        )
    if arrays["belief_fields_present"].shape != (sample_count,):
        raise ValueError(
            f"belief_fields_present shape mismatch for {shard.shard_name}: {arrays['belief_fields_present'].shape}"
        )
    _validate_absent_rows_zeroed(
        arrays["belief_fields_present"],
        arrays["belief_fields_target"],
        shard_name=shard.shard_name,
        field_name="belief_fields_target",
    )
    if arrays["mixture_weight_target"].shape != (sample_count, 4):
        raise ValueError(
            f"mixture_weight_target shape mismatch for {shard.shard_name}: {arrays['mixture_weight_target'].shape}"
        )
    if arrays["mixture_weight_present"].shape != (sample_count,):
        raise ValueError(
            f"mixture_weight_present shape mismatch for {shard.shard_name}: {arrays['mixture_weight_present'].shape}"
        )
    _validate_absent_rows_zeroed(
        arrays["mixture_weight_present"],
        arrays["mixture_weight_target"],
        shard_name=shard.shard_name,
        field_name="mixture_weight_target",
    )
    return arrays


def iter_games(shard: ShardRecord) -> Iterator[dict[str, np.ndarray]]:
    arrays = load_shard_arrays(shard)
    offsets = arrays["game_sample_offsets"]
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        start_i = int(start)
        end_i = int(end)
        yield {
            key: value[start_i:end_i]
            for key, value in arrays.items()
            if key != "game_sample_offsets"
        }


def _next_seed(seed: int) -> int:
    return (seed * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF


def _shuffle_indices(length: int, seed: int) -> list[int]:
    indices = list(range(length))
    state = seed
    for idx in range(length - 1, 0, -1):
        state = _next_seed(state)
        swap_idx = state % (idx + 1)
        indices[idx], indices[int(swap_idx)] = indices[int(swap_idx)], indices[idx]
    return indices


def _stream_shuffle_seed(seed: int, epoch: int, yield_index: int) -> int:
    return ((seed + epoch) * 1_000_003 + yield_index) & 0xFFFFFFFFFFFFFFFF


def _yield_buffer_batches(
    buffer: list[dict[str, np.ndarray]],
    *,
    seed: int,
    epoch: int,
    logical_batch_size: int,
    yield_index: int,
) -> Iterator[dict[str, np.ndarray]]:
    if not buffer:
        return
    samples = []
    for game in buffer:
        game_len = int(game["obs"].shape[0])
        for i in range(game_len):
            samples.append({key: value[i : i + 1] for key, value in game.items()})
    sample_order = _shuffle_indices(
        len(samples), _stream_shuffle_seed(seed, epoch, yield_index)
    )
    shuffled = [samples[idx] for idx in sample_order]
    for start in range(0, len(shuffled), logical_batch_size):
        chunk = shuffled[start : start + logical_batch_size]
        yield {
            key: np.concatenate([sample[key] for sample in chunk], axis=0)
            for key in chunk[0].keys()
        }


def iter_train_epoch_batches(
    dataset: ExportDataset,
    *,
    seed: int,
    epoch: int,
    buffer_games: int,
    buffer_samples: int,
    logical_batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    train_shards = dataset.shards("train")
    if not train_shards:
        return

    shard_order = _shuffle_indices(len(train_shards), seed + epoch)
    buffer: list[dict[str, np.ndarray]] = []
    buffered_samples = 0
    yield_index = 0

    for shard_idx in shard_order:
        shard = train_shards[shard_idx]
        shard_games = list(iter_games(shard))
        game_order = _shuffle_indices(
            len(shard_games), _stream_shuffle_seed(seed, epoch, shard_idx)
        )
        for game_idx in game_order:
            game = shard_games[game_idx]
            game_sample_count = int(game["obs"].shape[0])
            if buffer and (
                len(buffer) >= max(buffer_games, 1)
                or buffered_samples + game_sample_count > max(buffer_samples, 1)
            ):
                yield from _yield_buffer_batches(
                    buffer,
                    seed=seed,
                    epoch=epoch,
                    logical_batch_size=logical_batch_size,
                    yield_index=yield_index,
                )
                yield_index += 1
                buffer = []
                buffered_samples = 0
            buffer.append(game)
            buffered_samples += game_sample_count

    if buffer:
        yield from _yield_buffer_batches(
            buffer,
            seed=seed,
            epoch=epoch,
            logical_batch_size=logical_batch_size,
            yield_index=yield_index,
        )
