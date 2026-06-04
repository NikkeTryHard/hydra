from __future__ import annotations

import ast
import importlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int32]

HYDRA_TILE_WIDTH = 34
_TERMINALS = np.asarray([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.int32)


def _jax() -> Any:
    return importlib.import_module("jax")


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _strip_line_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def _rust_const_array(name: str) -> list[Any]:
    source = (_repo_root() / "crates" / "hydra-engine" / "src" / "shanten.rs").read_text()
    marker = f"const {name}:"
    start = source.index(marker)
    bracket_start = source.index("=", start)
    bracket_start = source.index("[", bracket_start)
    depth: int = 0
    for idx in range(bracket_start, len(source)):
        char = source[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(_strip_line_comments(source[bracket_start : idx + 1]))
    raise ValueError(f"failed to parse Rust shanten const {name}")


@lru_cache(maxsize=1)
def _tables_np() -> tuple[IntArray, IntArray, IntArray, IntArray, IntArray, IntArray, IntArray]:
    data_dir = _repo_root() / "crates" / "hydra-engine" / "src" / "data"
    return (
        np.asarray(_rust_const_array("SHUPAI_TABLE"), dtype=np.int32),
        np.asarray(_rust_const_array("ZIPAI_TABLE"), dtype=np.int32),
        np.frombuffer((data_dir / "nyanten_shupai_keys.bin").read_bytes(), dtype=np.uint8).astype(np.int32),
        np.frombuffer((data_dir / "nyanten_zipai_keys.bin").read_bytes(), dtype=np.uint8).astype(np.int32),
        np.frombuffer((data_dir / "nyanten_keys1.bin").read_bytes(), dtype=np.uint8).astype(np.int32),
        np.frombuffer((data_dir / "nyanten_keys2.bin").read_bytes(), dtype=np.uint8).astype(np.int32),
        np.frombuffer((data_dir / "nyanten_keys3.bin").read_bytes(), dtype=np.uint8).astype(np.int32),
    )


@lru_cache(maxsize=1)
def _tables_jax() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    jnp = _jnp()

    return tuple(jnp.asarray(table) for table in _tables_np())


def _hash_shupai_jax(counts9: Any) -> Any:
    jnp = _jnp()

    shupai_table, _, _, _, _, _, _ = _tables_jax()
    counts = jnp.asarray(counts9, dtype=jnp.int32)
    total = jnp.asarray(0, dtype=jnp.int32)
    hash_value = jnp.asarray(0, dtype=jnp.int32)
    for idx in range(9):
        count = counts[idx]
        total = total + count
        hash_value = hash_value + shupai_table[idx, total, count]
    return hash_value


def _hash_zipai_jax(counts7: Any) -> Any:
    jnp = _jnp()

    _, zipai_table, _, _, _, _, _ = _tables_jax()
    counts = jnp.asarray(counts7, dtype=jnp.int32)
    total = jnp.asarray(0, dtype=jnp.int32)
    hash_value = jnp.asarray(0, dtype=jnp.int32)
    for idx in range(7):
        count = counts[idx]
        total = total + count
        hash_value = hash_value + zipai_table[idx, total, count]
    return hash_value


def _calc_chitoi_jax(counts34: Any) -> Any:
    jnp = _jnp()

    counts = jnp.asarray(counts34, dtype=jnp.int32)
    pairs = jnp.sum(counts >= 2)
    kinds = jnp.sum(counts > 0)
    redunct = jnp.maximum(7 - kinds, 0)
    return 7 - pairs + redunct - 1


def _calc_kokushi_jax(counts34: Any) -> Any:
    jnp = _jnp()

    terminals = jnp.asarray(_TERMINALS)
    counts = jnp.asarray(counts34, dtype=jnp.int32)
    terminal_counts = counts[terminals]
    kinds = jnp.sum(terminal_counts > 0)
    has_pair = jnp.any(terminal_counts >= 2)
    return 14 - kinds - has_pair.astype(jnp.int32) - 1


def hydra_shanten_number_jax(counts34: Any) -> Any:
    """Exact Hydra/Rust shanten for a 34-wide count vector, device-safe JAX path."""
    jnp = _jnp()

    _, _, shupai_keys, zipai_keys, keys1, keys2, keys3 = _tables_jax()
    counts = jnp.asarray(counts34, dtype=jnp.int32)
    len_div3 = jnp.sum(counts, dtype=jnp.int32) // 3
    k0_m = shupai_keys[_hash_shupai_jax(counts[0:9])]
    k0_p = shupai_keys[_hash_shupai_jax(counts[9:18])]
    k1 = keys1[k0_m * 126 + k0_p]
    k0_s = shupai_keys[_hash_shupai_jax(counts[18:27])]
    k2 = keys2[k1 * 126 + k0_s]
    k0_z = zipai_keys[_hash_zipai_jax(counts[27:34])]
    normal = keys3[(k2 * 55 + k0_z) * 5 + len_div3] - 1
    chitoi = _calc_chitoi_jax(counts)
    kokushi = _calc_kokushi_jax(counts)
    special = jnp.where(normal > 0, jnp.minimum(chitoi, kokushi), normal)
    return jnp.where((normal <= 0) | (len_div3 < 4), normal, jnp.minimum(normal, special)).astype(jnp.int32)


def hydra_discard_shanten_masks_jax(counts34: Any) -> tuple[Any, Any, Any]:
    """Return base, non-increase mask, and decrease mask matching Hydra channels 9/10."""
    jax = _jax()
    jnp = _jnp()

    counts = jnp.asarray(counts34, dtype=jnp.int32)
    base = hydra_shanten_number_jax(counts)

    def discard_shanten(tile: Any) -> Any:
        present = counts[tile] > 0
        after_counts = counts.at[tile].add(jnp.where(present, -1, 0))
        return jnp.where(present, hydra_shanten_number_jax(after_counts), 127)

    after = jax.vmap(discard_shanten)(jnp.arange(HYDRA_TILE_WIDTH, dtype=jnp.int32))
    present = counts > 0
    return base, present & (after <= base), present & (after < base)
