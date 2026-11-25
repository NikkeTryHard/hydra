"""Actor batch tensors and deterministic seed streams."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ArenaBatch:
    slot: torch.Tensor
    seat: torch.Tensor
    tokens: torch.Tensor
    actions: torch.Tensor


def derive_seed(root: int, purpose: str, index: int) -> int:
    """Derive independent streams from semantic coordinates, not process scheduling."""
    if not purpose or not (0 <= root < 2**64 and 0 <= index < 2**64):
        raise ValueError("seed coordinates require a purpose and unsigned 64-bit integers")
    # Python's process-randomized hash cannot name repeatable streams. Give
    # each coordinate fixed boundaries in the SeedSequence entropy words.
    entropy = [root & 0xFFFFFFFF, root >> 32, index & 0xFFFFFFFF, index >> 32]
    encoded = purpose.encode("utf-8")
    entropy.append(len(encoded))
    entropy.extend(int.from_bytes(encoded[i : i + 4], "little") for i in range(0, len(encoded), 4))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
