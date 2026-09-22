"""Stubs for the `hydra2._native` Rust extension (single cdylib, maturin `module-name`).

Typecheck covers this file so stub drift fails CI: every name the
`#[pymodule] fn _native` entry exposes in `crates/bridge/src/lib.rs`
appears below (verified against `dir(hydra2._native)` post-`build-ext`).
Member-level types stay `Any` — the submodule boundary is untyped by
design (Step 6 ports pin behavior with parity tests, not stubs).
"""

from typing import Any

canon_rng: Any
columnar: Any
contracts: Any
encoder: Any
eval: Any  # noqa: A001 — submodule name shadows the builtin by contract
mirror: Any
packet: Any
packet_decode: Any
resume: Any
search: Any
tiles: Any

expand_games: Any
replay_game_planes: Any
replay_game_planes_wall: Any

PyHydraStream: Any
PyFill: Any
PyStats: Any
PyQuar: Any
PyDriverSnapshot: Any
PyDriverStats: Any
PyMicrobatchOut: Any
PyPushOut: Any
PyStreamDriver: Any
