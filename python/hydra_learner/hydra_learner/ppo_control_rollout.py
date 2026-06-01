from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from hydra_learner.ppo_control_config import RANK_UTILITY, PpoControlConfig
from hydra_learner.ppo_rollout import PpoSnapshotMetadata, snapshot_metadata_from_payload
from hydra_learner.ppo_smoke import RustDecisionRow, RustGameRollout, build_ppo_batch_from_rust_rollout
from hydra_learner.ppo_step import PpoBatch
from hydra_learner.rl import (
    DEFAULT_GAE_GAMMA,
    DEFAULT_GAE_LAMBDA,
    PLACEMENT_UTILITY_DEFAULT,
    PlayerDecisionStep,
    compute_player_local_gae,
    masked_log_prob,
)

if TYPE_CHECKING:
    from hydra_learner.model import HydraPolicyNet


class _PpoInferenceCallback(Protocol):
    _timings: dict[str, float]
    _packed_capacity: Callable[[], int]
    _packed_buffer_ptr: Callable[[], int]

    def __call__(self, obs_f32_le: bytearray, *args: object) -> object: ...


def _collect_native_rollout(
    extension: Any, config: PpoControlConfig, policy_dir: Path, seed: int, snapshot_metadata: PpoSnapshotMetadata
) -> Mapping[str, object]:
    collect = getattr(extension, "collect_ppo_rollouts_rust_native", None)
    if not callable(collect):
        raise ValueError("arena extension missing collect_ppo_rollouts_rust_native")
    payload = collect(
        config.games_per_update,
        seed,
        str(policy_dir),
        config.arena_batch_decisions,
        config.rollout_device or config.device,
        config.arena_threads,
        config.temperature,
        snapshot_metadata.to_payload(),
    )
    if not isinstance(payload, Mapping):
        raise TypeError("native PPO rollout collector must return a mapping")
    return payload


def _collect_callback_rollout(
    extension: Any, config: PpoControlConfig, model: HydraPolicyNet, seed: int, snapshot_metadata: PpoSnapshotMetadata
) -> Mapping[str, object]:
    collect = getattr(extension, "collect_ppo_rollouts_with_callback", None)
    if not callable(collect):
        raise ValueError("arena extension missing collect_ppo_rollouts_with_callback")
    callback = _make_ppo_inference_callback(
        model, torch.device(config.rollout_device or config.device), config.arena_batch_decisions
    )
    payload = collect(
        config.games_per_update,
        seed,
        config.arena_batch_decisions,
        config.arena_threads,
        config.temperature,
        callback,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("native PPO rollout collector must return a mapping")
    payload = dict(payload)
    payload["snapshot_metadata"] = snapshot_metadata.to_payload()
    timings = getattr(callback, "_timings", None)
    if isinstance(timings, dict):
        native_timing = payload.get("timing")
        if isinstance(native_timing, dict):
            native_timing.update(timings)
    return payload


def _make_ppo_inference_callback(
    model: HydraPolicyNet, device: torch.device, initial_capacity: int = 0
) -> Callable[..., object]:
    packed_capacity = max(0, initial_capacity)
    packed_device = torch.empty((packed_capacity, 47), dtype=torch.float32, device=device)
    packed_cpu = torch.empty((packed_capacity, 47), dtype=torch.float32, device="cpu")
    legal_capacity = 0
    legal_cpu = torch.empty((0,), dtype=torch.float32, device="cpu")

    def infer(obs_f32_le: bytearray, *args: object) -> object:
        nonlocal packed_capacity, packed_cpu, packed_device, legal_capacity, legal_cpu
        if len(args) == 1:
            legal_mask_u8 = None
            rows_raw = args[0]
        elif len(args) == 2:
            legal_mask_u8 = args[0]
            rows_raw = args[1]
        else:
            raise TypeError("PPO inference callback expects obs, rows or obs, legal_mask, rows")
        if not isinstance(rows_raw, int):
            raise TypeError("PPO inference callback rows must be an int")
        rows = rows_raw
        if rows < 0:
            raise ValueError(f"PPO inference callback rows must be non-negative, got {rows}")
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        obs = torch.frombuffer(obs_f32_le, dtype=torch.float32).reshape(rows, 192, 34).to(device)
        timings["callback_obs_h2d_ms"] = (time.perf_counter() - t0) * 1000.0
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                t0 = time.perf_counter()
                logits, values = model.policy_value(obs)
                timings["callback_forward_ms"] = (time.perf_counter() - t0) * 1000.0
                if logits.shape != (rows, 46):
                    raise ValueError(f"PPO policy logits must have shape ({rows}, 46), got {tuple(logits.shape)}")
                flat_values = values.reshape(rows)
                if flat_values.dtype != torch.float32:
                    flat_values = flat_values.to(dtype=torch.float32)
                if logits.dtype != torch.float32:
                    logits = logits.to(dtype=torch.float32)
                if legal_mask_u8 is None:
                    t0 = time.perf_counter()
                    if rows > packed_capacity:
                        packed_capacity = rows
                        packed_device = torch.empty((packed_capacity, 47), dtype=torch.float32, device=device)
                        packed_cpu = torch.empty((packed_capacity, 47), dtype=torch.float32, device="cpu")
                    device_buffer = packed_device[:rows]
                    device_buffer[:, :46].copy_(logits.detach())
                    device_buffer[:, 46].copy_(flat_values.detach())
                    row_buffer = packed_cpu[:rows]
                    row_buffer.copy_(device_buffer, non_blocking=False)
                    timings["callback_pack_copy_ms"] = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    packed_view = memoryview(row_buffer.numpy())
                    timings["callback_return_view_ms"] = (time.perf_counter() - t0) * 1000.0
                    timings["callback_d2h_pack_ms"] = (
                        timings["callback_pack_copy_ms"] + timings["callback_return_view_ms"]
                    )
                else:
                    if not isinstance(legal_mask_u8, bytes | bytearray | memoryview):
                        raise TypeError("PPO inference callback legal_mask must be bytes")
                    legal_mask = torch.frombuffer(legal_mask_u8, dtype=torch.uint8).reshape(rows, 46).to(device=device)
                    legal_mask = legal_mask.to(dtype=torch.bool)
                    legal_count = sum(memoryview(legal_mask_u8))
                    t0 = time.perf_counter()
                    legal_logits = logits[legal_mask]
                    timings["callback_legal_gather_ms"] = (time.perf_counter() - t0) * 1000.0
                    total_count = legal_count + rows
                    t0 = time.perf_counter()
                    if total_count > legal_capacity:
                        legal_capacity = total_count
                        legal_cpu = torch.empty((legal_capacity,), dtype=torch.float32, device="cpu")
                    packed_legal = torch.empty((total_count,), dtype=torch.float32, device=device)
                    packed_legal[:legal_count].copy_(legal_logits.detach())
                    packed_legal[legal_count:].copy_(flat_values.detach())
                    row_buffer = legal_cpu[:total_count]
                    row_buffer.copy_(packed_legal, non_blocking=False)
                    timings["callback_legal_d2h_pack_ms"] = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    packed_view = memoryview(row_buffer.numpy())
                    timings["callback_return_view_ms"] = (time.perf_counter() - t0) * 1000.0
                    timings["callback_d2h_pack_ms"] = (
                        timings["callback_legal_d2h_pack_ms"] + timings["callback_return_view_ms"]
                    )
                    full_count = rows * (46 + 1)
                    timings["callback_legal_transport_ratio"] = 0.0 if full_count == 0 else total_count / full_count
        finally:
            model.train(was_training)
        aggregate = getattr(infer, "_timings", None)
        if isinstance(aggregate, dict):
            for key, value in timings.items():
                if key == "callback_legal_transport_ratio":
                    weighted_ratio = value * rows
                    aggregate["callback_legal_transport_ratio_weighted_rows"] = (
                        aggregate.get("callback_legal_transport_ratio_weighted_rows", 0.0) + weighted_ratio
                    )
                    aggregate["callback_legal_transport_ratio_rows"] = (
                        aggregate.get("callback_legal_transport_ratio_rows", 0.0) + rows
                    )
                    total_rows = aggregate["callback_legal_transport_ratio_rows"]
                    aggregate[key] = (
                        0.0
                        if total_rows == 0.0
                        else aggregate["callback_legal_transport_ratio_weighted_rows"] / total_rows
                    )
                else:
                    aggregate[key] = aggregate.get(key, 0.0) + value
        return packed_view

    callback = cast("_PpoInferenceCallback", infer)
    callback._timings = {}
    callback._packed_capacity = lambda: max(packed_capacity, legal_capacity)
    callback._packed_buffer_ptr = (
        lambda: packed_cpu.data_ptr() if packed_capacity >= legal_capacity else legal_cpu.data_ptr()
    )
    return infer


def _snapshot_from_native_payload(payload: Mapping[str, object]) -> PpoSnapshotMetadata | None:
    raw = payload.get("snapshot_metadata")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("native rollout snapshot_metadata must be a mapping")
    return snapshot_metadata_from_payload(raw)


def _batch_from_native_payload_fast(payload: Mapping[str, object], model: HydraPolicyNet) -> PpoBatch:
    snapshot_metadata = _snapshot_from_native_payload(payload)
    binary = _batch_from_native_binary_payload(payload, model, snapshot_metadata)
    if binary is not None:
        return binary
    model_device = next(model.parameters()).device
    rows = payload.get("rows")
    terminals = payload.get("games")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("native rollout rows must be a sequence")
    if not isinstance(terminals, Sequence) or isinstance(terminals, str | bytes):
        raise ValueError("native rollout games must be a sequence")
    rows_by_game: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("native rollout row must be an object")
        rows_by_game.setdefault(int(row["game_id"]), []).append(row)
    ordered_rows: list[Mapping[str, object]] = []
    game_spans: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            raise ValueError("native rollout game terminal must be an object")
        game_id = int(terminal["game_id"])
        game_rows = rows_by_game.get(game_id)
        if not game_rows:
            raise ValueError(f"native rollout game {game_id} has no rows")
        start = len(ordered_rows)
        ordered_rows.extend(game_rows)
        game_spans.append((start, len(ordered_rows), _tuple4(terminal["placements"], "placements")))
    if len(ordered_rows) != len(rows):
        raise ValueError("native rollout rows contain games missing terminal metadata")
    obs = torch.tensor([row["obs"] for row in ordered_rows], dtype=torch.float32, device=model_device).reshape(
        len(ordered_rows), 192, 34
    )
    legal_mask = torch.tensor([row["legal_mask"] for row in ordered_rows], dtype=torch.bool, device=model_device)
    legal_count = torch.tensor(
        [_int_field(row, "legal_count") for row in ordered_rows], dtype=torch.int64, device=model_device
    )
    actions = torch.tensor([_int_field(row, "action") for row in ordered_rows], dtype=torch.int64, device=model_device)
    player_id = torch.tensor(
        [_int_field(row, "player_id") for row in ordered_rows], dtype=torch.int64, device=model_device
    )
    seat_id = torch.tensor([_int_field(row, "seat_id") for row in ordered_rows], dtype=torch.int64, device=model_device)
    game_id = torch.tensor([_int_field(row, "game_id") for row in ordered_rows], dtype=torch.int64, device=model_device)
    turn = torch.tensor([_int_field(row, "turn") for row in ordered_rows], dtype=torch.int64, device=model_device)
    return _finish_batch(
        obs, legal_mask, legal_count, actions, player_id, seat_id, game_id, turn, game_spans, model, snapshot_metadata
    )


def _batch_from_native_binary_payload(
    payload: Mapping[str, object], model: HydraPolicyNet, snapshot_metadata: PpoSnapshotMetadata | None
) -> PpoBatch | None:
    obs_raw = payload.get("obs_f32_le")
    legal_raw = payload.get("legal_mask_u8")
    if obs_raw is None and legal_raw is None:
        return None
    if not isinstance(obs_raw, bytes | bytearray | memoryview):
        raise TypeError("native rollout obs_f32_le must be bytes")
    if not isinstance(legal_raw, bytes | bytearray | memoryview):
        raise TypeError("native rollout legal_mask_u8 must be bytes")
    row_count_raw = payload.get("row_count")
    if not isinstance(row_count_raw, int):
        raise TypeError("native rollout row_count must be an int")
    row_count = row_count_raw
    if row_count < 1:
        raise ValueError("native rollout row_count must be positive")
    pin_memory = torch.cuda.is_available()
    batch_device = torch.device("cpu")
    obs = _clone_cpu(torch.frombuffer(memoryview(obs_raw), dtype=torch.float32).reshape(row_count, 192, 34), pin_memory)
    legal_mask = torch.frombuffer(memoryview(legal_raw), dtype=torch.uint8).reshape(row_count, 46).to(dtype=torch.bool)
    legal_mask = _clone_cpu(legal_mask, pin_memory)
    actions = _u8_column(payload, "actions", row_count, batch_device)
    legal_count = _u8_column(payload, "legal_counts", row_count, batch_device)
    player_ids_raw = _bytes_field(payload, "player_ids")
    player_id = _u8_column(payload, "player_ids", row_count, batch_device)
    seat_id = _u8_column(payload, "seat_ids", row_count, batch_device)
    game_ids_raw = payload.get("game_ids_u64_le")
    if isinstance(game_ids_raw, bytes | bytearray | memoryview):
        game_id = _clone_cpu(torch.frombuffer(memoryview(game_ids_raw), dtype=torch.int64), pin_memory)
    else:
        game_id = torch.tensor(_sequence_field(payload, "game_ids"), dtype=torch.int64, device=batch_device)
    turns_raw = payload.get("turns_u32_le")
    if isinstance(turns_raw, bytes | bytearray | memoryview):
        turn = _clone_cpu(torch.frombuffer(memoryview(turns_raw), dtype=torch.int32).to(dtype=torch.int64), pin_memory)
    else:
        turn = torch.tensor(_sequence_field(payload, "turns"), dtype=torch.int64, device=batch_device)
    old_logits_raw = payload.get("old_logits_f32_le")
    old_legal_logits_raw = payload.get("old_legal_logits_f32_le")
    value_old_raw = payload.get("value_old_f32_le")
    old_logprob_raw = payload.get("old_logprob_f32_le")
    raw_advantages_raw = payload.get("raw_advantages_f32_le")
    returns_raw = payload.get("returns_f32_le")
    has_scalar_payload = (
        isinstance(value_old_raw, bytes | bytearray | memoryview)
        and isinstance(old_logprob_raw, bytes | bytearray | memoryview)
        and isinstance(raw_advantages_raw, bytes | bytearray | memoryview)
        and isinstance(returns_raw, bytes | bytearray | memoryview)
    )
    if isinstance(old_logits_raw, bytes | bytearray | memoryview) and has_scalar_payload:
        old_logits = _clone_cpu(
            torch.frombuffer(memoryview(old_logits_raw), dtype=torch.float32).reshape(row_count, 46), pin_memory
        )
        value_old_cpu = _clone_cpu(torch.frombuffer(memoryview(value_old_raw), dtype=torch.float32), pin_memory)
        old_logprob = _clone_cpu(torch.frombuffer(memoryview(old_logprob_raw), dtype=torch.float32), pin_memory)
        raw_advantages = _clone_cpu(torch.frombuffer(memoryview(raw_advantages_raw), dtype=torch.float32), pin_memory)
        returns = _clone_cpu(torch.frombuffer(memoryview(returns_raw), dtype=torch.float32), pin_memory)
        value_old = value_old_cpu
    elif isinstance(old_legal_logits_raw, bytes | bytearray | memoryview) and has_scalar_payload:
        legal_total = int(legal_count.sum().item())
        old_legal_logits = torch.frombuffer(memoryview(old_legal_logits_raw), dtype=torch.float32)
        if old_legal_logits.shape != (legal_total,):
            actual_legal = old_legal_logits.shape[0]
            raise ValueError(
                "native rollout old_legal_logits_f32_le length must equal "
                f"legal_count sum {legal_total}, got {actual_legal}"
            )
        old_logits = torch.zeros((row_count, 46), dtype=torch.float32, device=batch_device)
        old_logits[legal_mask] = old_legal_logits
        old_logits = _clone_cpu(old_logits, pin_memory)
        value_old_cpu = _clone_cpu(
            torch.frombuffer(memoryview(_bytes_field(payload, "value_old_f32_le")), dtype=torch.float32), pin_memory
        )
        old_logprob = _clone_cpu(
            torch.frombuffer(memoryview(_bytes_field(payload, "old_logprob_f32_le")), dtype=torch.float32), pin_memory
        )
        raw_advantages = _clone_cpu(
            torch.frombuffer(memoryview(_bytes_field(payload, "raw_advantages_f32_le")), dtype=torch.float32),
            pin_memory,
        )
        returns = _clone_cpu(
            torch.frombuffer(memoryview(_bytes_field(payload, "returns_f32_le")), dtype=torch.float32), pin_memory
        )
        value_old = value_old_cpu
    else:
        old_logprob = None
        old_logits = None
        value_old = None
        value_old_cpu = None
        raw_advantages = None
        returns = None
    for name, tensor in (
        ("actions", actions),
        ("legal_counts", legal_count),
        ("player_ids", player_id),
        ("seat_ids", seat_id),
        ("game_ids", game_id),
        ("turns", turn),
    ):
        if tensor.shape != (row_count,):
            raise ValueError(f"native rollout {name} length must equal row_count")
    terminals = payload.get("games")
    if not isinstance(terminals, Sequence) or isinstance(terminals, str | bytes):
        raise ValueError("native rollout games must be a sequence")
    game_spans: list[tuple[int, int, tuple[int, int, int, int]]] = []
    starts_raw = payload.get("game_row_starts_u64_le")
    ends_raw = payload.get("game_row_ends_u64_le")
    placements_raw = payload.get("placements_u8")
    if (
        isinstance(starts_raw, bytes | bytearray | memoryview)
        and isinstance(ends_raw, bytes | bytearray | memoryview)
        and isinstance(placements_raw, bytes | bytearray | memoryview)
    ):
        starts = memoryview(starts_raw).cast("Q")
        ends = memoryview(ends_raw).cast("Q")
        placements = memoryview(placements_raw)
        if len(starts) != len(terminals) or len(ends) != len(terminals) or len(placements) != len(terminals) * 4:
            raise ValueError("native rollout span fields must match games length")
        for idx in range(len(starts)):
            start = starts[idx]
            end = ends[idx]
            if end <= start or end > row_count:
                raise ValueError("native rollout invalid game span")
            placement_offset = idx * 4
            game_spans.append((start, end, _tuple4(placements[placement_offset : placement_offset + 4], "placements")))
        if not game_spans or game_spans[-1][1] != row_count:
            raise ValueError("native rollout spans must cover row_count")
    else:
        if not isinstance(game_ids_raw, bytes | bytearray | memoryview):
            game_ids_cpu = game_id.cpu().tolist()
        else:
            game_ids_cpu = memoryview(game_ids_raw).cast("Q")
        start = 0
        for terminal in terminals:
            if not isinstance(terminal, Mapping):
                raise ValueError("native rollout game terminal must be an object")
            gid = int(terminal["game_id"])
            end = start
            while end < row_count and int(game_ids_cpu[end]) == gid:
                end += 1
            if end == start:
                raise ValueError(f"native rollout game {gid} has no rows")
            game_spans.append((start, end, _tuple4(terminal["placements"], "placements")))
            start = end
        if start != row_count:
            raise ValueError("native rollout rows contain games missing terminal metadata")
    return _finish_batch(
        obs,
        legal_mask,
        legal_count,
        actions,
        player_id,
        seat_id,
        game_id,
        turn,
        game_spans,
        model,
        snapshot_metadata,
        old_logits=old_logits,
        value_old=value_old,
        value_old_cpu=value_old_cpu,
        old_logprob=old_logprob,
        raw_advantages=raw_advantages,
        returns=returns,
        player_ids_cpu=list(player_ids_raw),
        validate=False,
    )


def _finish_batch(
    obs: torch.Tensor,
    legal_mask: torch.Tensor,
    legal_count: torch.Tensor,
    actions: torch.Tensor,
    player_id: torch.Tensor,
    seat_id: torch.Tensor,
    game_id: torch.Tensor,
    turn: torch.Tensor,
    game_spans: list[tuple[int, int, tuple[int, int, int, int]]],
    model: HydraPolicyNet,
    snapshot_metadata: PpoSnapshotMetadata | None = None,
    old_logits: torch.Tensor | None = None,
    value_old: torch.Tensor | None = None,
    value_old_cpu: torch.Tensor | None = None,
    old_logprob: torch.Tensor | None = None,
    raw_advantages: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
    player_ids_cpu: list[int] | None = None,
    validate: bool = True,
) -> PpoBatch:
    batch_device = obs.device
    if old_logits is None or value_old is None:
        model_device = next(model.parameters()).device
        with torch.inference_mode():
            was_training = model.training
            model.eval()
            logits, value = model.policy_value(obs.to(model_device))
            model.train(was_training)
            logits = logits.detach().to(dtype=torch.float32).to(batch_device)
            value_old = value.squeeze(1).detach().to(dtype=torch.float32).to(batch_device)
    else:
        logits = old_logits
    if old_logprob is None:
        old_logprob = masked_log_prob(logits, legal_mask, actions).detach().to(dtype=torch.float32)
    if raw_advantages is not None and returns is not None:
        pass
    elif value_old_cpu is not None and player_ids_cpu is not None:
        raw_advantages, returns = _terminal_gae_from_cpu_values(
            player_ids_cpu=player_ids_cpu,
            value_old_cpu=value_old_cpu,
            game_spans=game_spans,
            device=batch_device,
        )
    else:
        raw_advantages = torch.empty(obs.shape[0], dtype=torch.float32, device=batch_device)
        returns = torch.empty(obs.shape[0], dtype=torch.float32, device=batch_device)
        player_id_cpu = player_id.cpu().tolist()
        value_old_list = value_old.cpu().tolist()
        for start, end, placements in game_spans:
            gae = compute_player_local_gae(
                [
                    PlayerDecisionStep(player_id=int(pid), value_old=float(value))
                    for pid, value in zip(player_id_cpu[start:end], value_old_list[start:end], strict=True)
                ],
                final_placements=placements,
                gamma=DEFAULT_GAE_GAMMA,
                gae_lambda=DEFAULT_GAE_LAMBDA,
            )
            raw_advantages[start:end] = gae.raw_advantages.to(batch_device)
            returns[start:end] = gae.returns.to(batch_device)
    batch = PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=logits,
        legal_count=legal_count,
        player_id=player_id,
        seat_id=seat_id,
        game_id=game_id,
        turn=turn,
        rank_utility_used=RANK_UTILITY,
        snapshot_metadata=None if snapshot_metadata is None else snapshot_metadata.to_payload(),
    )
    if validate:
        batch.validate()
    return batch


def _terminal_gae_from_cpu_values(
    *,
    player_ids_cpu: list[int],
    value_old_cpu: torch.Tensor,
    game_spans: list[tuple[int, int, tuple[int, int, int, int]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_cpu = torch.empty_like(value_old_cpu)
    returns_cpu = torch.empty_like(value_old_cpu)
    discount = DEFAULT_GAE_GAMMA * DEFAULT_GAE_LAMBDA
    for start, end, placements in game_spans:
        players = player_ids_cpu[start:end]
        values = value_old_cpu[start:end]
        for player in range(4):
            running = 0.0
            next_value = 0.0
            has_next = False
            reward = PLACEMENT_UTILITY_DEFAULT[placements[player]]
            for local_index in range(len(players) - 1, -1, -1):
                if players[local_index] != player:
                    continue
                value = float(values[local_index])
                delta = (
                    (reward if not has_next else 0.0) + (DEFAULT_GAE_GAMMA * next_value if has_next else 0.0) - value
                )
                running = delta + (discount * running if has_next else 0.0)
                index = start + local_index
                raw_cpu[index] = running
                returns_cpu[index] = running + value
                next_value = value
                has_next = True
    return raw_cpu.to(device), returns_cpu.to(device)


def _bytes_field(payload: Mapping[str, object], key: str) -> bytes | bytearray | memoryview:
    value = payload.get(key)
    if not isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(f"native rollout {key} must be bytes")
    return value


def _u8_column(payload: Mapping[str, object], key: str, row_count: int, device: torch.device) -> torch.Tensor:
    tensor = torch.frombuffer(memoryview(_bytes_field(payload, key)), dtype=torch.uint8).to(
        dtype=torch.int64, device=device
    )
    if tensor.shape != (row_count,):
        raise ValueError(f"native rollout {key} length must equal row_count")
    return tensor


def _sequence_field(payload: Mapping[str, object], key: str) -> Sequence[object]:
    value = payload.get(key)
    if isinstance(value, str | bytes):
        raise TypeError(f"native rollout {key} must be a sequence")
    return cast("Sequence[object]", value if isinstance(value, Sequence) else list(cast("Any", value)))


def _batch_from_native_payload(payload: Mapping[str, object], model: HydraPolicyNet) -> PpoBatch:
    model_device = next(model.parameters()).device
    rows = payload.get("rows")
    terminals = payload.get("games")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("native rollout rows must be a sequence")
    if not isinstance(terminals, Sequence) or isinstance(terminals, str | bytes):
        raise ValueError("native rollout games must be a sequence")
    rows_by_game: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("native rollout row must be an object")
        game_id = int(row["game_id"])
        rows_by_game.setdefault(game_id, []).append(row)
    batches: list[PpoBatch] = []
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            raise ValueError("native rollout game terminal must be an object")
        game_id = int(terminal["game_id"])
        game_rows = rows_by_game.get(game_id)
        if not game_rows:
            raise ValueError(f"native rollout game {game_id} has no rows")
        rollout = RustGameRollout(
            rows=tuple(_decision_row(row) for row in game_rows),
            final_scores=_tuple4(terminal["final_scores"], "final_scores"),
            placements=_tuple4(terminal["placements"], "placements"),
            seed=int(terminal["seed"]),
        )
        batches.append(
            build_ppo_batch_from_rust_rollout(
                rollout,
                model=model,
                torch_seed=rollout.seed,
                gae_gamma=DEFAULT_GAE_GAMMA,
                gae_lambda=DEFAULT_GAE_LAMBDA,
                output_device=model_device,
            )
        )
    return _concat_batches(batches)


def _decision_row(row: Mapping[str, object]) -> RustDecisionRow:
    obs = torch.tensor(cast(Sequence[float], row["obs"]), dtype=torch.float32).reshape(192, 34)
    legal_mask = torch.tensor(cast(Sequence[bool], row["legal_mask"]), dtype=torch.bool)
    return RustDecisionRow(
        obs=obs,
        legal_mask=legal_mask,
        player_id=_int_field(row, "player_id"),
        seat_id=_int_field(row, "seat_id"),
        game_id=_int_field(row, "game_id"),
        turn=_int_field(row, "turn"),
        action=_int_field(row, "action"),
        legal_count=_int_field(row, "legal_count"),
    )


def _int_field(row: Mapping[str, object], key: str) -> int:
    value = row[key]
    if not isinstance(value, int):
        raise TypeError(f"native rollout row {key} must be int")
    return value


def _tuple4(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{name} must contain four ints")
    return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))


def _concat_batches(batches: list[PpoBatch]) -> PpoBatch:
    if not batches:
        raise ValueError("rollout produced no PPO batches")
    return PpoBatch(
        obs=torch.cat([b.obs for b in batches], dim=0),
        actions=torch.cat([b.actions for b in batches], dim=0),
        legal_mask=torch.cat([b.legal_mask for b in batches], dim=0),
        old_logprob=torch.cat([b.old_logprob for b in batches], dim=0),
        value_old=torch.cat([b.value_old for b in batches], dim=0),
        raw_advantages=torch.cat([b.raw_advantages for b in batches], dim=0),
        returns=torch.cat([b.returns for b in batches], dim=0),
        bc_logits=torch.cat([b.bc_logits for b in batches], dim=0),
        legal_count=torch.cat([b.legal_count for b in batches], dim=0),
        player_id=torch.cat([b.player_id for b in batches if b.player_id is not None], dim=0),
        seat_id=torch.cat([b.seat_id for b in batches if b.seat_id is not None], dim=0),
        game_id=torch.cat([b.game_id for b in batches if b.game_id is not None], dim=0),
        turn=torch.cat([b.turn for b in batches if b.turn is not None], dim=0),
        rank_utility_used=RANK_UTILITY,
    )


def _clone_cpu(tensor: torch.Tensor, pin_memory: bool) -> torch.Tensor:
    if not pin_memory:
        return tensor.clone()
    out = torch.empty_like(tensor, device="cpu", pin_memory=True)
    out.copy_(tensor)
    return out


def _batch_to_device(batch: PpoBatch, device: torch.device) -> PpoBatch:
    non_blocking = device.type == "cuda"
    tensors = (
        batch.obs,
        batch.actions,
        batch.legal_mask,
        batch.old_logprob,
        batch.value_old,
        batch.raw_advantages,
        batch.returns,
        batch.bc_logits,
        batch.legal_count,
        batch.player_id,
        batch.seat_id,
        batch.game_id,
        batch.turn,
    )
    if all(tensor is None or tensor.device == device for tensor in tensors):
        return batch
    return PpoBatch(
        obs=batch.obs.to(device, non_blocking=non_blocking),
        actions=batch.actions.to(device, non_blocking=non_blocking),
        legal_mask=batch.legal_mask.to(device, non_blocking=non_blocking),
        old_logprob=batch.old_logprob.to(device, non_blocking=non_blocking),
        value_old=batch.value_old.to(device, non_blocking=non_blocking),
        raw_advantages=batch.raw_advantages.to(device, non_blocking=non_blocking),
        returns=batch.returns.to(device, non_blocking=non_blocking),
        bc_logits=batch.bc_logits.to(device, non_blocking=non_blocking),
        legal_count=batch.legal_count.to(device, non_blocking=non_blocking),
        player_id=None if batch.player_id is None else batch.player_id.to(device, non_blocking=non_blocking),
        seat_id=None if batch.seat_id is None else batch.seat_id.to(device, non_blocking=non_blocking),
        game_id=None if batch.game_id is None else batch.game_id.to(device, non_blocking=non_blocking),
        turn=None if batch.turn is None else batch.turn.to(device, non_blocking=non_blocking),
        rank_utility_used=batch.rank_utility_used,
        snapshot_metadata=batch.snapshot_metadata,
    )
