"""Closed RL profile: thin typed wrapper over the native ``rl`` parser.

Typing/firewall only: :class:`RlConfig` re-export, section entry point,
and dry-plan rendering. All loops/hashes/validators live in Rust
(``hydra2._native.contracts.rc_parse_rl`` /
``rc_cross_validate_rl``); this module never duplicates them.
"""

from __future__ import annotations

from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_parse import _parse_rl as _parse_rl
from hydra2.training._rc_sections import RlConfig as RlConfig
from hydra2.training._rc_sections import RunConfig as RunConfig

try:
    from hydra2._native import contracts as _rl_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _rl_bridge = None  # type: ignore[assignment]


def _require_rl_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _rl_bridge is None or not hasattr(_rl_bridge, "rc_parse_rl"):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _rl_bridge


__all__ = [
    "RlConfig",
    "RunConfig",
    "describe_rl_plan",
    "parse_rl_section",
]


def parse_rl_section(raw: Any) -> RlConfig:
    """Validate a raw ``rl`` mapping natively; the bridge decides."""
    _require_rl_bridge()
    try:
        return _parse_rl(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def describe_rl_plan(config: RunConfig) -> str:
    """Dry-plan inputs for an RL run: frozen recipe, caps, lineage, guards.

    Pure rendering of the validated :class:`RunConfig` (what
    ``hydra2 train --dry-run`` prints for ``run.kind=rl``); never wall-clock,
    never IO. Raises :class:`ContractError` when the config carries no RL
    profile.
    """
    rl = config.rl
    if rl is None:
        raise ContractError("describe_rl_plan requires run.kind 'rl' (rl section absent)")
    lines = [
        f"rl: profile={rl.schema_version} experiment={rl.experiment_id} "
        f"split={rl.split_scope} replicate={rl.replicate_id} attempt={rl.attempt_id}",
        f"rl: lineage initial_checkpoint={rl.initial_checkpoint_id} "
        f"model_spec={rl.initial_model_spec_hash} (weight-only load; fresh optimizer)",
        f"rl: cohort generations={rl.generations} games_per_generation={rl.games_per_generation} "
        f"epochs={rl.epochs} (every row exactly once, tails included)",
        f"rl: resources workers={rl.native_workers} slots={rl.env_slots} "
        f"lookahead={rl.max_lookahead} inference_cap={rl.inference_cap}",
        f"rl: learner micro_cap={rl.microbatch_cap} accum_cap={rl.accumulation_cap} "
        f"minibatch_cap={rl.optimizer_minibatch_cap} "
        f"max_updates_per_generation={rl.max_updates_per_generation}",
        f"rl: ppo clip_eps={rl.clip_eps} eps_std={rl.eps_std} (INSIDE sqrt) "
        f"w_value={rl.w_value} w_bc={rl.w_bc} alpha={rl.ppo_alpha}",
        f"rl: return gamma={rl.gamma} lam={rl.lam} (terminal MC, no bootstrap/shaping)",
        f"rl: optimizer adamw lr={rl.optimizer_lr} "
        f"betas=[{rl.optimizer_beta1}, {rl.optimizer_beta2}] "
        f"decay={rl.optimizer_weight_decay} head_mult={rl.head_lr_mult} "
        f"constant warmup={rl.warmup_updates} clip={rl.gradient_clip_norm}",
        f"rl: guards row_cap={rl.row_cap} decision_cap={rl.decision_cap_per_game} "
        f"compressed={rl.compressed_byte_cap} min_free_disk={rl.min_free_disk_bytes} "
        f"host={rl.host_byte_budget} gpu_cache={rl.gpu_cache_byte_budget}",
        f"rl: seeds master={rl.master_seed_hex[:16]}... core={rl.core_revision[:12]}...",
        "rl: publication boundary is the first full optimizer update of each generation",
    ]
    return "\n".join(lines) + "\n"
