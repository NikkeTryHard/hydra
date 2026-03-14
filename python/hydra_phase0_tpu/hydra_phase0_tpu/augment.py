from __future__ import annotations

import numpy as np


ALL_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

AKA_CHANNEL_START = 40
AKA_CHANNELS = 3


def _permute_tile_type(tile: int, perm: tuple[int, int, int]) -> int:
    if tile >= 27:
        return tile
    suit = tile // 9
    rank = tile % 9
    return perm[suit] * 9 + rank


def _permute_tile_extended(action: int, perm: tuple[int, int, int]) -> int:
    if action <= 36:
        if action < 34:
            return _permute_tile_type(action, perm)
        aka_base = {34: 4, 35: 13, 36: 22}[action]
        mapped = _permute_tile_type(aka_base, perm)
        return {4: 34, 13: 35, 22: 36}[mapped]
    return action


def augment_obs_suit(obs: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(obs)
    channels, tiles = obs.shape
    for ch in range(channels):
        if AKA_CHANNEL_START <= ch < AKA_CHANNEL_START + AKA_CHANNELS:
            suit = ch - AKA_CHANNEL_START
            out[AKA_CHANNEL_START + perm[suit], :] = obs[ch, :]
            continue
        for tile in range(tiles):
            out[ch, _permute_tile_type(tile, perm)] = obs[ch, tile]
    return out


def augment_action_suit(action: int, perm: tuple[int, int, int]) -> int:
    return _permute_tile_extended(action, perm)


def augment_mask_suit(mask: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(mask)
    for i in range(37):
        out[_permute_tile_extended(i, perm)] = mask[i]
    out[37:] = mask[37:]
    return out


def augment_action_vector_suit(
    values: np.ndarray, perm: tuple[int, int, int]
) -> np.ndarray:
    out = np.zeros_like(values)
    for i in range(37):
        out[_permute_tile_extended(i, perm)] = values[i]
    out[37:] = values[37:]
    return out


def _augment_tile_axis(values: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(values)
    channels, tiles = values.shape
    for channel in range(channels):
        for tile in range(tiles):
            out[channel, _permute_tile_type(tile, perm)] = values[channel, tile]
    return out


def apply_train_augmentation(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    augmented_batches: list[dict[str, np.ndarray]] = []
    for perm in ALL_PERMUTATIONS:
        augmented = {key: value.copy() for key, value in batch.items()}
        augmented["obs"] = np.stack(
            [augment_obs_suit(sample, perm) for sample in batch["obs"]], axis=0
        )
        augmented["legal_mask"] = np.stack(
            [augment_mask_suit(sample, perm) for sample in batch["legal_mask"]], axis=0
        )
        augmented["action"] = np.asarray(
            [augment_action_suit(int(action), perm) for action in batch["action"]],
            dtype=batch["action"].dtype,
        )
        augmented["opp_next_target"] = np.stack(
            [_augment_tile_axis(sample, perm) for sample in batch["opp_next_target"]],
            axis=0,
        )
        augmented["danger_target"] = np.stack(
            [_augment_tile_axis(sample, perm) for sample in batch["danger_target"]],
            axis=0,
        )
        augmented["danger_mask"] = np.stack(
            [_augment_tile_axis(sample, perm) for sample in batch["danger_mask"]],
            axis=0,
        )
        augmented["safety_residual_target"] = np.stack(
            [
                augment_action_vector_suit(sample, perm)
                for sample in batch["safety_residual_target"]
            ],
            axis=0,
        )
        augmented["safety_residual_mask"] = np.stack(
            [
                augment_mask_suit(sample, perm)
                for sample in batch["safety_residual_mask"]
            ],
            axis=0,
        )
        if "belief_fields_target" in batch:
            augmented["belief_fields_target"] = np.stack(
                [
                    _augment_tile_axis(sample, perm)
                    for sample in batch["belief_fields_target"]
                ],
                axis=0,
            )
        augmented_batches.append(augmented)

    return {
        key: np.concatenate([aug[key] for aug in augmented_batches], axis=0)
        for key in batch.keys()
    }
