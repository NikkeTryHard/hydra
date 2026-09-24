"""Shared supervised-loop test builders (WP-05B).

Authoritative synthetic-parquet builders, the session-cached parquet-variant
fixture, the deterministic loop constructor, real-observation helpers, and
the gated-selection fixture shared by the supervised objectives, training,
and loop test modules. TEST-ONLY: real runs supply frozen spec digests and
production datasets; these builders exist so hermetic unit tests can train
small deterministic loops without fabricating production artifacts.

The stub models live in test_supervised_objectives (the objectives
module owns the test doubles); this module imports and re-exports them so
the training and loop modules share a single import source.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch

from hydra2.contracts.observation_actor import make_actor_observation
from hydra2.data.parquet import DecisionRow, write_actor_shards
from hydra2.eval.blocks import WallBlock
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training.dataset_store import AuthoritativeParquetDataset
from hydra2.training.loop_state import TrainingLoopConfig
from hydra2.training.loop_train import SupervisedLoop
from tests.unit._manifest_helpers import make_test_manifest_hashes
from tests.unit.test_supervised_objectives import (
    StubModelPerSeat,
    StubModelWithAux,
    StubPolicyModel,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn

__all__ = [
    "StubModelPerSeat",
    "StubModelWithAux",
    "StubPolicyModel",
    "_build_loop",
    "_gated_selection_fixture",
    "_make_actor_rows",
    "_write_real_parquet",
    "_write_synthetic_parquet",
]


NUM_ACTIONS_SMALL = 16  # small vocab for test speed (real table is 6792)
FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Synthetic parquet helpers (authoritative, actor-only)
# ---------------------------------------------------------------------------


def _make_actor_rows(num_rows: int = 20, num_actions: int = NUM_ACTIONS_SMALL) -> list[DecisionRow]:
    rows: list[DecisionRow] = []
    for i in range(num_rows):
        # Actor observation: privileged-free, dora (5,) sentinel shape
        obs = {
            "dora_indicators": [10 + (i % 5), 11 + (i % 5), -1, -1, -1],
            "hand_counts": [4] * 34,
            "history_mask": [1] * 8 + [0] * 8,
            "legal_mask_bits": [1] * num_actions,
            "phase": "draw_decision",
        }
        rows.append(
            DecisionRow(
                game_id=f"game-{i // 4:03d}",
                round_id=f"round-{i // 4}-0",
                decision_id=f"dec-{i:04d}",
                seat=i % 4,
                source_object_id=f"obj-{i:04d}",
                split="train",
                rules_hash="sha256:" + "a" * 64,
                adapter_hash="sha256:" + "b" * 64,
                observation_hash="sha256:" + hashlib.sha256(f"obs-{i}".encode()).hexdigest(),
                action_table_hash="sha256:" + "c" * 64,
                derivation_hash="sha256:" + "d" * 64,
                actor_observation=obs,  # type: ignore[arg-type]
                chosen_action_id=i % num_actions,
            )
        )
    return rows


def _write_synthetic_parquet(
    tmp_path: Path, num_rows: int = 20, num_actions: int = NUM_ACTIONS_SMALL
) -> Path:
    dest = tmp_path / "actor_parquet"
    rows = _make_actor_rows(num_rows=num_rows, num_actions=num_actions)
    write_actor_shards(
        destination=dest,
        rows=rows,
        dataset_hash="sha256:" + "e" * 64,
        split_manifest_hash="sha256:" + "f" * 64,
    )
    return dest


# ---------------------------------------------------------------------------
# Helpers to build deterministic loop
# ---------------------------------------------------------------------------


def _build_loop(
    tmp_path: Path,
    parquet_dir: Path,
    *,
    seed: int = 123,
    microbatch_size: int = 4,
    accumulation_steps: int = 1,
    num_actions: int = NUM_ACTIONS_SMALL,
    w_policy: float = 1.0,
    w_placement: float = 0.0,
    w_value: float = 0.0,
    w_event: dict[str, float] | None = None,
    checkpoint_subdir: str = "checkpoints",
    gradient_clip_norm: float | None = 1.0,
    max_updates: int = 4,
    precision: str = "fp32",
) -> tuple[SupervisedLoop, AuthoritativeParquetDataset, nn.Module]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=num_actions,
        seed=seed,
        verify=True,
        allow_narrow=True,
    )
    model: nn.Module
    if w_placement != 0.0 or w_value != 0.0 or (w_event and any(v != 0 for v in w_event.values())):
        model = StubModelWithAux(feature_dim=FEATURE_DIM, num_actions=num_actions)
    else:
        model = StubPolicyModel(feature_dim=FEATURE_DIM, num_actions=num_actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        if gradient_clip_norm is not None
        else None
    )
    config = TrainingLoopConfig(
        w_policy=w_policy,
        w_placement=w_placement,
        w_value=w_value,
        w_event=w_event,
        microbatch_size=microbatch_size,
        accumulation_steps=accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        max_updates=max_updates,
        checkpoint_frequency_updates=10,  # avoid auto-checkpoint during small tests; explicit saves
        seed=seed,
        precision=precision,  # type: ignore[arg-type]
    )
    ckpt_dir = tmp_path / checkpoint_subdir
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        config=config,
        checkpoint_dir=ckpt_dir,
        manifest_hashes=make_test_manifest_hashes(),  # test-only digests; src requires real hashes
    )
    return loop, dataset, model


# ---------------------------------------------------------------------------
# Gated-selection fixture (walls plus telemetry plus frozen config)
# ---------------------------------------------------------------------------


def _gated_selection_fixture():  # type: ignore[no-untyped-def]
    """Two valid wall blocks (mean 3.0) + telemetry + frozen fixed_n config."""
    from hydra2.eval.selection import SelectionConfig
    from hydra2.eval.telemetry import make_resource_telemetry

    def _row(wall_id: str):  # type: ignore[no-untyped-def]
        return make_resource_telemetry(
            mode="cuda_eager",
            wall_id=wall_id,
            case_id=None,
            candidate_spec_hash="sha256:" + "11" * 32,
            hardware_hash="sha256:" + "22" * 32,
            environment_hash="sha256:" + "33" * 32,
            cold_start=False,
            synchronized_elapsed_ms=12.5,
            model_calls=3,
            exact_transitions=40,
            particles=0,
            fallback_used=False,
            timeout=False,
            illegal_action=False,
            cuda_peak_allocated_bytes=1024,
            cuda_peak_reserved_bytes=2048,
            host_peak_bytes=None,
            energy_joules=None,
            graph_breaks=None,
            recompiles=None,
            invalid_reason=None,
        )

    blocks = (
        WallBlock(wall_id="w0", game_ids=("w0-g0",), contrasts=(2.0,)),
        WallBlock(wall_id="w1", game_ids=("w1-g0",), contrasts=(4.0,)),
    )
    telemetry = {"w0-g0": _row("w0"), "w1-g0": _row("w1")}
    config = SelectionConfig(
        N=2,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="fixed_n",
        declared_peeks=(2,),
        resamples=200,
        seed=7,
    )
    return blocks, telemetry, config


# ---------------------------------------------------------------------------
# 11 Real-observation tensorization alongside the synthetic stand-in
# ---------------------------------------------------------------------------


def _real_legal_mask(*legal: int) -> tuple[bool, ...]:
    mask = [False] * BASELINE_ACTION_COUNT
    for action in legal:
        mask[action] = True
    return tuple(mask)


def _make_real_observation(
    *,
    decision_id: str,
    concealed_hand: tuple[int, ...],
    sequence: int = 1,
    legal_mask: tuple[bool, ...] | None = None,
) -> object:
    return make_actor_observation(
        game_id="g-wp05b-real",
        decision_id=decision_id,
        sequence=sequence,
        actor=0,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "ab" * 32,
        action_table_hash="sha256:" + "ac" * 32,
        event_schema_hash="sha256:" + "ad" * 32,
        observation_schema_hash="sha256:" + "ae" * 32,
        packet_boundary_hash="sha256:" + "af" * 32,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=concealed_hand,
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=(),
        legal_mask=legal_mask if legal_mask is not None else _real_legal_mask(0, 10),
    )


def _real_row_dict(decision_id: str, obs: object, chosen: int = 0) -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "chosen_action_id": chosen,
        "actor_observation": json.dumps(obs.to_json()),  # type: ignore[union-attr]
    }


def _write_real_parquet(tmp_path: Path, hands: list[tuple[int, ...]]) -> Path:
    rows: list[DecisionRow] = []
    for i, hand in enumerate(hands):
        obs = _make_real_observation(decision_id=f"dec-real-{i:04d}", concealed_hand=hand)
        rows.append(
            DecisionRow(
                game_id=f"game-{i // 4:03d}",
                round_id=f"round-{i // 4}-0",
                decision_id=f"dec-real-{i:04d}",
                seat=i % 4,
                source_object_id=f"obj-{i:04d}",
                split="train",
                rules_hash="sha256:" + "a" * 64,
                adapter_hash="sha256:" + "b" * 64,
                observation_hash="sha256:" + hashlib.sha256(f"obs-{i}".encode()).hexdigest(),
                action_table_hash="sha256:" + "c" * 64,
                derivation_hash="sha256:" + "d" * 64,
                actor_observation=obs.to_json(),  # type: ignore[arg-type]
                chosen_action_id=0,
            )
        )
    dest = tmp_path / "real_actor_parquet"
    write_actor_shards(
        destination=dest,
        rows=rows,
        dataset_hash="sha256:" + "e" * 64,
        split_manifest_hash="sha256:" + "f" * 64,
    )
    return dest
