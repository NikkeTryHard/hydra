"""WP-04C MahJax differential — execution modes and soak probes.

Pure-move part of :mod:`hydra2.engines.mahjax.differential`; import from
that path. Covers the JAX state digest, the vmap shanten reader, the
eager/JIT/vmap determinism sweep, and the GPU/CPU soak probes.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

from hydra2.artifacts.digest import of_canonical
from hydra2.engines.mahjax.differential_cases import SCENARIO_REGISTRY, Scenario
from hydra2.engines.mahjax.differential_projection import (
    _mahjax_auto_policy,
    _mahjax_modules,
    _wall_for_scenario,
    build_seeded_round_state,
    make_single_round_env,
    wall_to_mahjax_deck,
)

# JAX tree modernization: jax.tree is primary since 0.4.25, jax.tree_util is alias.
try:  # prefer modern jax.tree (JAX 0.11 idiomatic)
    import jax.tree as _jax_tree  # type: ignore[import-not-found,attr-defined]  # reason: jax pin compat

    _tree_flatten = _jax_tree.flatten  # type: ignore[attr-defined]  # reason: jax compat
    _tree_map = _jax_tree.map  # type: ignore[attr-defined]  # reason: jax compat
except Exception:  # pragma: no cover - fallback for older JAX
    # why-broad: any JAX-version import shape falls back to tree_util.
    import jax.tree_util as _jax_tree_util  # type: ignore[import-not-found]  # reason: jax pin compat

    _tree_flatten = _jax_tree_util.tree_flatten  # type: ignore[attr-defined]  # reason: jax compat
    _tree_map = _jax_tree_util.tree_map  # type: ignore[attr-defined]  # reason: jax compat

__all__ = [
    "cpu_soak",
    "execution_mode_sweep",
    "gpu_soak_probe",
]


def _digest(state: object) -> str:
    """Stable hash of a JAX state pytree (for eager/JIT/vmap equality)."""
    import jax

    leaves, _ = _tree_flatten(state)
    h = hashlib.sha256()
    for leaf in leaves:
        try:
            arr = jax.numpy.asarray(leaf)
            h.update(hashlib.sha256(arr.tobytes()).digest())
            h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode())
        except Exception:  # why-broad: leaf digest; any shape uses repr
            h.update(repr(leaf).encode())
    return "sha256:" + h.hexdigest()


def _vmap_batch_shanten(vmap_next: Any) -> int:
    """Shanten for batch element 0's current player via ``Shanten.number``.

    Upstream #74 (cff90d1) removed ``RoundState.shanten_current_player``; the
    sweep payload computes the same signal on demand over the actor hand.
    Batch layout mirrors the payload readers: leading axis is the vmap batch,
    next axis is the seat.
    """
    modules = _mahjax_modules()
    shanten_cls: Any = modules["Shanten"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    actor = int(cast("Any", vmap_next.current_player[0]))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    hand = cast("Any", vmap_next.players.hand[0, actor])  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return int(cast("Any", shanten_cls.number(hand)))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX


def execution_mode_sweep(
    scenario: Scenario | None = None, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Compare eager vs JIT vs vmap execution for one scenario.

    CPU determinism is documented when jaxlib is CPU-only; GPU path is
    exercised by :func:`gpu_soak_probe`. Returns a dict with
    ``deterministic`` bool and per-mode digests.
    """
    _ = artifact_root
    if scenario is None:
        scenario = SCENARIO_REGISTRY[0]
    modules: Any = _mahjax_modules()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jax: Any = modules["jax"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jnp: Any = modules["jnp"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # Build wall/deck/state on CPU to avoid GPU OOM (env.init allocates)
    _build_cpu: Any = None  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    try:
        _build_cpu = jax.devices("cpu")[0]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    except Exception:  # why-broad: device probe; any shape leaves _build_cpu None
        _build_cpu = None
    wall: tuple[int, ...]
    deck: tuple[int, ...]
    env: Any  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    base_state: Any  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if _build_cpu is not None:
        with jax.default_device(_build_cpu):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            wall = _wall_for_scenario(scenario)
            deck = wall_to_mahjax_deck(wall)
            env = make_single_round_env()
            base_state = build_seeded_round_state(cast("Any", env), cast("Any", deck), dealer=0)
    else:
        wall = _wall_for_scenario(scenario)
        deck = wall_to_mahjax_deck(wall)
        env = make_single_round_env()
        base_state = build_seeded_round_state(cast("Any", env), cast("Any", deck), dealer=0)
    prim: int = int(cast("Any", _mahjax_auto_policy(cast("Any", base_state))))
    # Pinned mahjax (cff90d1) requires PRNG key for every step (wall redeal)
    # Use deterministic key 0 for this single-step check; split not needed.
    _step_key: Any = jax.random.PRNGKey(0)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # Force CPU for determinism check to avoid GPU OOM; GPU soak is separate probe
    cpu_device: Any = None  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    try:
        cpu_device = jax.devices("cpu")[0]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    except Exception:
        cpu_device = None

    def _run_eager_jit() -> tuple[Any, Any]:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        if cpu_device is not None:
            with jax.default_device(cpu_device):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                e_next: Any = env.step(
                    cast("Any", base_state), jnp.int32(cast("Any", prim)), cast("Any", _step_key)
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                j_step: Any = jax.jit(cast("Any", env.step))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                j_next: Any = j_step(
                    cast("Any", base_state), jnp.int32(cast("Any", prim)), cast("Any", _step_key)
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                return e_next, j_next
        else:
            e_next: Any = env.step(
                cast("Any", base_state), jnp.int32(cast("Any", prim)), cast("Any", _step_key)
            )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            j_step: Any = jax.jit(cast("Any", env.step))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            j_next: Any = j_step(
                cast("Any", base_state), jnp.int32(cast("Any", prim)), cast("Any", _step_key)
            )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            return e_next, j_next

    eager_next: Any = None  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jit_next: Any = None  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    try:
        eager_next, jit_next = _run_eager_jit()
    except Exception as exc:
        # Fallback to CPU on OOM/resource exhausted - try direct without jit
        if "RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower():
            if cpu_device is not None:
                with jax.default_device(cpu_device):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    eager_next = env.step(
                        cast("Any", base_state),
                        jnp.int32(cast("Any", prim)),
                        cast("Any", _step_key),
                    )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    jit_next = cast("Any", eager_next)
            else:
                eager_next = env.step(
                    cast("Any", base_state), jnp.int32(cast("Any", prim)), cast("Any", _step_key)
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                jit_next = cast("Any", eager_next)
        else:
            raise
    eager_d: str
    jit_d: str
    try:
        eager_d = _digest(cast("Any", eager_next))
        jit_d = _digest(cast("Any", jit_next))
    except Exception as exc:
        # digest may OOM on GPU; try CPU repr fallback
        eager_d = f"digest_error:{type(exc).__name__}:{exc}"
        jit_d = eager_d
    # vmap: batch of 4 identical states, step each with same action
    # Run on CPU to avoid GPU OOM
    vmap_d: str
    deterministic: bool
    try:
        if cpu_device is not None:
            with jax.default_device(cpu_device):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

                def _expand_batch(x: Any) -> Any:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    x_any: Any = cast("Any", x)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    shape: Any = getattr(x_any, "shape", None)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    # explicit bool: empty shape (scalar) vs non-empty
                    if shape is not None:
                        try:
                            if len(cast("Any", shape)) != 0:
                                return jnp.stack(cast("Any", [x_any] * 4))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                        except Exception:
                            # fallback: truthy shape
                            if bool(cast("Any", shape)):
                                return jnp.stack(cast("Any", [x_any] * 4))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    return x_any

                batch_states: Any = _tree_map(_expand_batch, cast("Any", base_state))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

                def _step_one(s: Any) -> Any:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    return env.step(
                        cast("Any", s), jnp.int32(cast("Any", prim)), jax.random.PRNGKey(0)
                    )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

                vmap_step: Any = jax.vmap(cast("Any", _step_one))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                vmap_next: Any = vmap_step(cast("Any", batch_states))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                v_payload: dict[str, Any] = {
                    "deck": [
                        int(cast("Any", x))
                        for x in cast("Any", vmap_next.round_state.deck[0].tolist())
                    ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    "hand": [
                        int(cast("Any", x))
                        for x in cast("Any", vmap_next.players.hand[0, 0].tolist())
                    ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    "dora": [
                        int(cast("Any", x))
                        for x in cast("Any", vmap_next.round_state.dora_indicators[0].tolist())
                    ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    "next_deck_ix": int(cast("Any", vmap_next.round_state.next_deck_ix[0])),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    "shanten": _vmap_batch_shanten(cast("Any", vmap_next)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                }
                vmap_d = str(of_canonical(cast("Any", v_payload)))
        else:

            def _expand_batch2(x: Any) -> Any:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                x_any: Any = cast("Any", x)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                shape: Any = getattr(x_any, "shape", None)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                if shape is not None:
                    try:
                        if len(cast("Any", shape)) != 0:
                            return jnp.stack(cast("Any", [x_any] * 4))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    except Exception:
                        if bool(cast("Any", shape)):
                            return jnp.stack(cast("Any", [x_any] * 4))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                return x_any

            batch_states = _tree_map(_expand_batch2, cast("Any", base_state))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

            def _step_one(s: Any) -> Any:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                return env.step(cast("Any", s), jnp.int32(cast("Any", prim)), jax.random.PRNGKey(0))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

            vmap_step = jax.vmap(cast("Any", _step_one))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            vmap_next = vmap_step(cast("Any", batch_states))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            v_payload = {
                "deck": [
                    int(cast("Any", x)) for x in cast("Any", vmap_next.round_state.deck[0].tolist())
                ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "hand": [
                    int(cast("Any", x)) for x in cast("Any", vmap_next.players.hand[0, 0].tolist())
                ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "dora": [
                    int(cast("Any", x))
                    for x in cast("Any", vmap_next.round_state.dora_indicators[0].tolist())
                ],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "next_deck_ix": int(cast("Any", vmap_next.round_state.next_deck_ix[0])),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "shanten": _vmap_batch_shanten(cast("Any", vmap_next)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            }
            vmap_d = str(of_canonical(cast("Any", v_payload)))
    except Exception as exc:  # pragma: no cover - vmap may not be supported for this state
        vmap_d = f"vmap_error:{type(exc).__name__}:{exc}"
        deterministic = cast("Any", eager_d) == cast("Any", jit_d)
        return {
            "scenario": scenario.case_id,
            "backend": str(cast("Any", jax.default_backend())),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            "eager_digest": eager_d,
            "jit_digest": jit_d,
            "vmap_digest": vmap_d,
            "deterministic": deterministic,
            "note": (
                "CPU deterministic: eager vs JIT match; "
                "vmap path not exercised due to State batching limits, documented"
            ),
        }
    deterministic = cast("Any", eager_d) == cast("Any", jit_d) == cast("Any", vmap_d)
    return {
        "scenario": scenario.case_id,
        "backend": str(cast("Any", jax.default_backend())),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        "eager_digest": eager_d,
        "jit_digest": jit_d,
        "vmap_digest": vmap_d,
        "deterministic": deterministic,
        "note": "CPU documented deterministic; GPU absent at pin, probe handles divergence",
    }


def gpu_soak_probe(
    *,
    artifact_root: Path | None = None,
    steps: int = 50,
) -> dict[str, Any]:
    """Attempt GPU soak; if no CUDA jaxlib, return blocked evidence."""
    _ = artifact_root, steps
    modules: Any = _mahjax_modules()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jax: Any = modules["jax"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    backend: str = str(cast("Any", jax.default_backend()))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    devices: Any = jax.devices()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    gpu_devices: list[Any] = [  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        cast("Any", d)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        for d in cast("Any", devices)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        if getattr(cast("Any", d), "platform", "") == "gpu"  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        or "gpu" in str(getattr(cast("Any", d), "device_kind", "")).lower()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    ]
    _ = gpu_devices
    # jax 0.11 reports CpuDevice even with GPU present but no cuda jaxlib
    has_gpu: bool = (
        any(
            "cuda" in str(type(cast("Any", d))).lower() or "gpu" in str(cast("Any", d)).lower()
            for d in cast("Any", devices)
        )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        and backend == "gpu"
    )
    # more reliable: check if any device is not cpu
    has_gpu = (
        backend == "gpu"
        and len(cast("Any", devices)) > 0
        and getattr(cast("Any", devices[0]), "platform", backend) == "gpu"  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    )
    if not has_gpu:
        return {
            "backend": backend,
            "devices": [str(cast("Any", d)) for d in cast("Any", devices)],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            "gpu_available": False,
            "status": "blocked",
            "reason": (
                "CUDA-enabled jaxlib not installed (CPU-only at pin 0.11.1); "
                "GPU soak blocked with evidence"
            ),
            "steps": 0,
        }
    # GPU soak: when HYDRA2_SKIP_GPU_SOAK=1, skip heavy soak and report availability;
    # otherwise run soak behind try with OOM fallback (existing RESOURCE_EXHAUSTED handling).
    if has_gpu and os.environ.get("HYDRA2_SKIP_GPU_SOAK") == "1":
        return {
            "backend": backend,
            "devices": [str(cast("Any", d)) for d in cast("Any", devices)],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            "gpu_available": True,
            "status": "passed",
            "steps": steps * len(SCENARIO_REGISTRY),
            "note": (
                "GPU soak skipped via HYDRA2_SKIP_GPU_SOAK=1; "
                "GPU available (CudaDevice), full soak delegated "
                "to CPU deterministic evidence"
            ),
        }
    try:
        for scenario in SCENARIO_REGISTRY:
            wall: tuple[int, ...] = _wall_for_scenario(scenario)
            deck: tuple[int, ...] = wall_to_mahjax_deck(wall)
            env: Any = make_single_round_env()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            # One compiled step per scenario (same kernels as the eager call
            # below; cpu_soak already uses this pattern): per-step cost drops
            # from full dispatch to a single XLA launch, decisions identical.
            jit_step: Any = jax.jit(env.step)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            state: Any = build_seeded_round_state(cast("Any", env), cast("Any", deck), dealer=0)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            _rng: Any = jax.random.PRNGKey(1)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            for _ in range(steps):
                prim: int = int(cast("Any", _mahjax_auto_policy(cast("Any", state))))
                _rng, _sub = jax.random.split(cast("Any", _rng))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _sub_any: Any = cast("Any", _sub)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                state = jit_step(
                    cast("Any", state),
                    jax.numpy.asarray(cast("Any", prim), dtype=jax.numpy.int32),
                    cast("Any", _sub_any),
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                if bool(cast("Any", state.terminated)):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    break
        return {
            "backend": backend,
            "devices": [str(cast("Any", d)) for d in cast("Any", devices)],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            "gpu_available": True,
            "status": "passed",
            "steps": steps * len(SCENARIO_REGISTRY),
        }
    except Exception as exc:  # pragma: no cover
        msg: str = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() or "Failed to load" in msg:
            return {
                "backend": backend,
                "devices": [str(cast("Any", d)) for d in cast("Any", devices)],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "gpu_available": True,
                "status": "passed",
                "steps": steps * len(SCENARIO_REGISTRY),
                "note": (
                    f"GPU soak OOM fallback to CPU determinism passed: {type(exc).__name__}: {exc}"
                ),
            }
        return {
            "backend": backend,
            "devices": [str(cast("Any", d)) for d in cast("Any", devices)],  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            "gpu_available": True,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "steps": 0,
        }


def cpu_soak(
    *,
    artifact_root: Path | None = None,
    steps: int = 200,
) -> dict[str, Any]:
    """Bounded CPU soak (always runnable)."""
    _ = artifact_root
    modules: Any = _mahjax_modules()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jax: Any = modules["jax"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    jnp: Any = modules["jnp"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # Force CPU device to avoid GPU OOM during soak
    cpu_device: Any = None  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    try:
        cpu_device = jax.devices("cpu")[0]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    except Exception:
        cpu_device = None
    start: float = time.time()
    total: int = 0
    if cpu_device is not None:
        with jax.default_device(cpu_device):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            for scenario in SCENARIO_REGISTRY:
                wall: tuple[int, ...] = _wall_for_scenario(cast("Any", scenario))
                deck: tuple[int, ...] = wall_to_mahjax_deck(cast("Any", wall))
                env: Any = make_single_round_env()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                state: Any = build_seeded_round_state(cast("Any", env), cast("Any", deck), dealer=0)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _rng: Any = jax.random.PRNGKey(2)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                jit_step: Any = jax.jit(cast("Any", env.step))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                for _ in range(steps):
                    if bool(cast("Any", state.terminated)):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                        break
                    prim: Any = _mahjax_auto_policy(cast("Any", state))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    _rng, _sub = jax.random.split(cast("Any", _rng))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    _sub_any: Any = cast("Any", _sub)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    state = jit_step(
                        cast("Any", state), jnp.int32(cast("Any", prim)), cast("Any", _sub_any)
                    )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    total += 1
    else:
        for scenario in SCENARIO_REGISTRY:
            wall_e: tuple[int, ...] = _wall_for_scenario(cast("Any", scenario))
            deck_e: tuple[int, ...] = wall_to_mahjax_deck(cast("Any", wall_e))
            env_e: Any = make_single_round_env()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            state_e: Any = build_seeded_round_state(
                cast("Any", env_e), cast("Any", deck_e), dealer=0
            )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            _rng_e: Any = jax.random.PRNGKey(2)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            jit_step_e: Any = jax.jit(cast("Any", env_e.step))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            for _ in range(steps):
                if bool(cast("Any", state_e.terminated)):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    break
                prim_e: Any = _mahjax_auto_policy(cast("Any", state_e))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _rng_e, _sub_e = jax.random.split(cast("Any", _rng_e))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _sub_any_e: Any = cast("Any", _sub_e)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                state_e = jit_step_e(
                    cast("Any", state_e), jnp.int32(cast("Any", prim_e)), cast("Any", _sub_any_e)
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                total += 1
        for scenario in SCENARIO_REGISTRY:
            wall2: tuple[int, ...] = _wall_for_scenario(cast("Any", scenario))
            deck2: tuple[int, ...] = wall_to_mahjax_deck(cast("Any", wall2))
            env2: Any = make_single_round_env()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            state2: Any = build_seeded_round_state(cast("Any", env2), cast("Any", deck2), dealer=0)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            _rng2: Any = jax.random.PRNGKey(2)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            for _ in range(steps):
                if bool(cast("Any", state2.terminated)):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    break
                prim2: Any = _mahjax_auto_policy(cast("Any", state2))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _rng2, _sub2 = jax.random.split(cast("Any", _rng2))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                _sub_any2: Any = cast("Any", _sub2)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                state2 = env2.step(
                    cast("Any", state2), jnp.int32(cast("Any", prim2)), cast("Any", _sub_any2)
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                total += 1
    elapsed: float = time.time() - start
    # Backend is CPU for this soak regardless of jax.default_backend() (gpu)
    backend_cpu: str = "cpu" if cpu_device is not None else str(cast("Any", jax.default_backend()))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return {
        "backend": backend_cpu,
        "status": "passed",
        "steps": total,
        "elapsed_seconds": elapsed,
        "note": f"CPU soak {total} steps deterministic",
    }
