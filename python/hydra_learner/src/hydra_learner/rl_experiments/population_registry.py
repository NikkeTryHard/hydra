"""Population registry and promotion evidence ledger."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal, Self, cast

from hydra_learner.checkpointing.eval import (
    PairedCheckpointEvalDecision,
    PairedCheckpointEvalThresholds,
    decide_paired_checkpoint_eval,
    normalize_paired_arena_metrics,
)

SCHEMA_VERSION = 1
REGISTRY_FILENAME = "population_registry.json"
PROMOTIONS_DIRNAME = "promotions"
EVIDENCE_DIRNAME = "population_evidence"

CheckpointRole = Literal["candidate", "champion", "rejected", "seed"]
CheckpointStatus = Literal["registered", "promoted", "rejected", "blocked"]
WeightSource = Literal["raw", "ema"]
PromotionDecision = Literal["promote", "reject", "insufficient_games", "blocked"]

_FORBIDDEN_FIELD_TOKENS = ("psro", "pfsp", "exploiter", "search_teacher", "search_objective", "payoff_matrix")
_CHECKPOINT_EVAL_DECISIONS = {"promote", "reject", "insufficient_games"}
_DELTA_Q_ARENA_REJECTS = {"Reject", "reject", "REJECT"}
_DELTA_Q_REQUIRES_ARENA = {"RequiresArenaConfirmation", "requires_arena_confirmation"}


@dataclass(frozen=True)
class CheckpointEntry:
    checkpoint_id: str
    role: CheckpointRole
    path: str
    path_sha256: str | None
    policy_json_sha256: str | None
    weight_source: WeightSource
    global_step: int | None
    samples_seen: int | None
    model_config: dict[str, object]
    model_config_sha256: str | None
    source: dict[str, object]
    parent_checkpoint_id: str | None
    status: CheckpointStatus
    registered_at_unix_ms: int


@dataclass(frozen=True)
class SeedBank:
    seed_set_id: str
    seeds: tuple[int, ...]
    games_per_seed: int
    temperature: float
    arena_options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OpponentPool:
    pool_id: str
    strategy: Literal["active_baseline_only"]
    baseline_checkpoint_id: str
    opponent_checkpoint_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    max_size: int = 1


@dataclass(frozen=True)
class EvalSchedule:
    enabled: bool
    seed_set_id: str
    opponent_pool_id: str
    min_games: int
    thresholds: dict[str, float | int | None]


@dataclass(frozen=True)
class PromotionDecisionRecord:
    decision: PromotionDecision
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | str | None]
    thresholds: dict[str, float | int | None]


@dataclass(frozen=True)
class PromotionRecord:
    schema_version: int
    promotion_id: str
    candidate_checkpoint_id: str
    baseline_checkpoint_id: str
    opponent_pool_id: str
    seed_set_id: str
    arena_summary_path: str
    paired_eval_summary_path: str
    normalized_metrics: dict[str, float | int | str | None]
    thresholds: dict[str, float | int | None]
    decision: PromotionDecisionRecord
    registry_update: dict[str, object]
    created_at_unix_ms: int
    checkpoint_eval_summary_path: str | None = None
    delta_q_summary_path: str | None = None
    evidence_seed: int | None = None
    seat_coverage_verified: bool = False
    seat_coverage: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PopulationRegistry:
    schema_version: int
    registry_id: str
    run_id: str
    active_baseline_id: str
    latest_candidate_id: str | None
    checkpoints: dict[str, CheckpointEntry]
    seed_banks: dict[str, SeedBank]
    opponent_pools: dict[str, OpponentPool]
    eval_schedule: EvalSchedule
    promotions: dict[str, PromotionRecord]

    @classmethod
    def create(
        cls,
        *,
        registry_id: str,
        run_id: str,
        active_baseline_id: str,
        baseline: CheckpointEntry,
        seed_bank: SeedBank,
        eval_schedule: EvalSchedule,
        latest_candidate_id: str | None = None,
    ) -> Self:
        if baseline.checkpoint_id != active_baseline_id:
            raise ValueError("active baseline id must match baseline checkpoint entry")
        pool = OpponentPool(
            pool_id=eval_schedule.opponent_pool_id,
            strategy="active_baseline_only",
            baseline_checkpoint_id=active_baseline_id,
            opponent_checkpoint_ids=(active_baseline_id,),
            checkpoint_ids=(active_baseline_id,),
            max_size=1,
        )
        registry = cls(
            schema_version=SCHEMA_VERSION,
            registry_id=registry_id,
            run_id=run_id,
            active_baseline_id=active_baseline_id,
            latest_candidate_id=latest_candidate_id,
            checkpoints={baseline.checkpoint_id: baseline},
            seed_banks={seed_bank.seed_set_id: seed_bank},
            opponent_pools={pool.pool_id: pool},
            eval_schedule=eval_schedule,
            promotions={},
        )
        registry.validate()
        return registry

    @classmethod
    def load(cls, path: Path) -> Self:
        payload = _read_json_object(path)
        registry = _registry_from_dict(payload)
        registry.validate()
        return registry

    def save(self, output_dir: Path) -> Path:
        path = output_dir / REGISTRY_FILENAME
        write_json_file(path, registry_to_dict(self))
        return path

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema_version {self.schema_version!r}")
        _require_non_empty_id(self.registry_id, "registry_id")
        _require_non_empty_id(self.run_id, "run_id")
        if self.active_baseline_id not in self.checkpoints:
            raise ValueError("registry missing active baseline checkpoint")
        if self.latest_candidate_id is not None and self.latest_candidate_id not in self.checkpoints:
            raise ValueError("registry latest candidate is not registered")
        if self.eval_schedule.seed_set_id not in self.seed_banks:
            raise ValueError("eval_schedule seed_set_id is not registered")
        if self.eval_schedule.opponent_pool_id not in self.opponent_pools:
            raise ValueError("eval_schedule opponent_pool_id is not registered")
        for checkpoint_id, entry in self.checkpoints.items():
            if checkpoint_id != entry.checkpoint_id:
                raise ValueError("checkpoint key/id mismatch")
            _validate_checkpoint_entry(entry)
        for seed_set_id, seed_bank in self.seed_banks.items():
            if seed_set_id != seed_bank.seed_set_id:
                raise ValueError("seed bank key/id mismatch")
            _validate_seed_bank(seed_bank)
        for pool_id, pool in self.opponent_pools.items():
            if pool_id != pool.pool_id:
                raise ValueError("opponent pool key/id mismatch")
            _validate_opponent_pool(pool)
            _validate_active_baseline_pool(pool, self.active_baseline_id, self.checkpoints)
        _validate_eval_schedule(self.eval_schedule)
        _validate_json_payload(registry_to_dict(self), "registry")

    def with_checkpoint(self, entry: CheckpointEntry) -> Self:
        _validate_checkpoint_entry(entry)
        if entry.checkpoint_id in self.checkpoints:
            raise ValueError(f"checkpoint already registered: {entry.checkpoint_id}")
        checkpoints = dict(self.checkpoints)
        checkpoints[entry.checkpoint_id] = entry
        latest_candidate_id = entry.checkpoint_id if entry.role == "candidate" else self.latest_candidate_id
        registry = _replace_registry(self, checkpoints=checkpoints, latest_candidate_id=latest_candidate_id)
        registry.validate()
        return registry

    def with_promotion(self, record: PromotionRecord) -> Self:
        if record.promotion_id in self.promotions:
            raise ValueError(f"promotion already registered: {record.promotion_id}")
        _validate_promotion_record(record)
        if record.candidate_checkpoint_id not in self.checkpoints:
            raise ValueError("promotion candidate is not registered")
        if record.baseline_checkpoint_id != self.active_baseline_id:
            raise ValueError("promotion baseline does not match active baseline")
        if record.baseline_checkpoint_id not in self.checkpoints:
            raise ValueError("promotion baseline is not registered")
        promotions = dict(self.promotions)
        promotions[record.promotion_id] = record
        checkpoints = dict(self.checkpoints)
        active_baseline_id = self.active_baseline_id
        opponent_pools = self.opponent_pools
        if record.decision.decision == "promote":
            active_baseline_id = record.candidate_checkpoint_id
            checkpoints[record.candidate_checkpoint_id] = _replace_checkpoint(
                checkpoints[record.candidate_checkpoint_id], role="champion", status="promoted"
            )
            checkpoints[record.baseline_checkpoint_id] = _replace_checkpoint(
                checkpoints[record.baseline_checkpoint_id], role="seed"
            )
            opponent_pools = _updated_active_baseline_pools(self.opponent_pools, active_baseline_id)
        elif record.decision.decision in {"reject", "insufficient_games", "blocked"}:
            checkpoints[record.candidate_checkpoint_id] = _replace_checkpoint(
                checkpoints[record.candidate_checkpoint_id],
                status="blocked" if record.decision.decision == "blocked" else "rejected",
            )
        else:
            raise ValueError(f"unsupported promotion decision {record.decision.decision!r}")
        registry = _replace_registry(
            self,
            active_baseline_id=active_baseline_id,
            checkpoints=checkpoints,
            promotions=promotions,
            opponent_pools=opponent_pools,
        )
        registry.validate()
        return registry


def default_registry_path(output_dir: Path) -> Path:
    return output_dir / REGISTRY_FILENAME


def register_immutable_checkpoint(
    *,
    checkpoint_id: str,
    role: CheckpointRole,
    path: Path,
    weight_source: WeightSource,
    global_step: int | None = None,
    samples_seen: int | None = None,
    model_config: Mapping[str, object] | None = None,
    source: Mapping[str, object] | None = None,
    parent_checkpoint_id: str | None = None,
    status: CheckpointStatus = "registered",
    registered_at_unix_ms: int | None = None,
) -> CheckpointEntry:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.name == "latest.pt":
        raise ValueError("mutable latest.pt must be copied to a pinned evidence path before registration")
    path_sha256: str | None = None
    policy_json_sha256: str | None = None
    if resolved.is_file():
        path_sha256 = file_sha256(resolved)
    elif resolved.is_dir():
        policy_json = resolved / "policy.json"
        if not policy_json.is_file():
            raise ValueError("export directory must contain policy.json")
        policy_json_sha256 = file_sha256(policy_json)
    else:
        raise ValueError("checkpoint path must be a file or export directory")
    normalized_model_config = dict(model_config or {})
    normalized_source = dict(source or {})
    entry = CheckpointEntry(
        checkpoint_id=checkpoint_id,
        role=role,
        path=str(resolved),
        path_sha256=path_sha256,
        policy_json_sha256=policy_json_sha256,
        weight_source=weight_source,
        global_step=global_step,
        samples_seen=samples_seen,
        model_config=normalized_model_config,
        model_config_sha256=_mapping_sha256(normalized_model_config) if normalized_model_config else None,
        source=normalized_source,
        parent_checkpoint_id=parent_checkpoint_id,
        status=status,
        registered_at_unix_ms=registered_at_unix_ms if registered_at_unix_ms is not None else unix_ms_now(),
    )
    _validate_checkpoint_entry(entry)
    return entry


def pin_mutable_checkpoint_for_registry(*, checkpoint_id: str, source_path: Path, output_dir: Path) -> Path:
    resolved = source_path.expanduser().resolve(strict=True)
    if resolved.name != "latest.pt":
        raise ValueError("pinning helper is only for mutable latest.pt checkpoints")
    if not resolved.is_file():
        raise ValueError("mutable checkpoint must be a file")
    destination_dir = output_dir / EVIDENCE_DIRNAME / "checkpoints" / checkpoint_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{checkpoint_id}.pt"
    if destination.exists():
        raise ValueError(f"pinned checkpoint already exists: {destination}")
    shutil.copy2(resolved, destination)
    return destination.resolve(strict=True)


def build_seed_bank(
    *,
    seed_set_id: str,
    seeds: tuple[int, ...],
    games_per_seed: int,
    temperature: float,
    arena_options: Mapping[str, object] | None = None,
) -> SeedBank:
    seed_bank = SeedBank(
        seed_set_id=seed_set_id,
        seeds=seeds,
        games_per_seed=games_per_seed,
        temperature=temperature,
        arena_options=dict(arena_options or {}),
    )
    _validate_seed_bank(seed_bank)
    return seed_bank


def build_promotion_record(
    *,
    registry: PopulationRegistry,
    promotion_id: str,
    candidate_checkpoint_id: str,
    baseline_checkpoint_id: str,
    opponent_pool_id: str,
    seed_set_id: str,
    arena_summary_path: Path,
    checkpoint_eval_summary: Mapping[str, object],
    paired_eval_summary_path: Path | None = None,
    thresholds: Mapping[str, float | int | None] | None = None,
    delta_q_summary: Mapping[str, object] | None = None,
    delta_q_summary_path: Path | None = None,
    created_at_unix_ms: int | None = None,
) -> PromotionRecord:
    registry.validate()
    if candidate_checkpoint_id not in registry.checkpoints:
        raise ValueError("candidate checkpoint is not registered")
    if baseline_checkpoint_id != registry.active_baseline_id:
        raise ValueError("baseline checkpoint does not match active registry baseline")
    if opponent_pool_id not in registry.opponent_pools:
        raise ValueError("opponent pool is not registered")
    if seed_set_id not in registry.seed_banks:
        raise ValueError("seed set is not registered")
    arena_path = arena_summary_path.expanduser().resolve(strict=True)
    paired_path = (
        paired_eval_summary_path.expanduser().resolve(strict=True) if paired_eval_summary_path is not None else None
    )
    arena_summary = _read_json_object(arena_path)
    _validate_arena_summary(
        arena_summary, registry.checkpoints[baseline_checkpoint_id], registry.checkpoints[candidate_checkpoint_id]
    )
    _validate_checkpoint_artifact(registry.checkpoints[baseline_checkpoint_id])
    _validate_checkpoint_artifact(registry.checkpoints[candidate_checkpoint_id])
    if delta_q_summary is not None:
        _validate_delta_q_summary(delta_q_summary)
    summary_baseline = _required_str(checkpoint_eval_summary, "baseline")
    summary_candidate = _required_str(checkpoint_eval_summary, "candidate")
    baseline_entry = registry.checkpoints[baseline_checkpoint_id]
    candidate_entry = registry.checkpoints[candidate_checkpoint_id]
    _match_summary_identity(summary_baseline, baseline_entry, "baseline")
    _match_summary_identity(summary_candidate, candidate_entry, "candidate")
    decision_payload = _required_mapping(checkpoint_eval_summary, "decision")
    checkpoint_eval_decision = _required_str(decision_payload, "decision")
    if checkpoint_eval_decision not in _CHECKPOINT_EVAL_DECISIONS:
        raise ValueError("checkpoint_eval decision is not supported")
    metrics = _metric_map(_required_mapping(checkpoint_eval_summary, "metrics"))
    _validate_checkpoint_eval_against_arena(
        arena_summary,
        metrics,
        checkpoint_eval_summary,
        registry.seed_banks[seed_set_id],
        registry.checkpoints[candidate_checkpoint_id],
    )
    games = metrics.get("games")
    if not isinstance(games, int) or games < 1:
        raise ValueError("checkpoint_eval metrics.games must be positive")
    reasons = _string_tuple(decision_payload.get("reasons", ()))
    active_thresholds = dict(thresholds or registry.eval_schedule.thresholds)
    _validate_thresholds(active_thresholds)
    _require_configured_metric_gates(active_thresholds, metrics)
    recomputed_decision = _recompute_checkpoint_eval_decision(metrics, active_thresholds)
    if recomputed_decision.decision != checkpoint_eval_decision or recomputed_decision.reasons != reasons:
        raise ValueError("checkpoint_eval decision mismatch with configured registry thresholds")
    decision = cast("PromotionDecision", checkpoint_eval_decision)
    if (
        decision == "promote"
        and delta_q_summary is not None
        and _delta_q_arena_decision(delta_q_summary) in _DELTA_Q_ARENA_REJECTS
    ):
        decision = "blocked"
        reasons = (*reasons, "delta_q_arena_reject")
    if (
        decision == "promote"
        and delta_q_summary is not None
        and _delta_q_recommendation(delta_q_summary) in _DELTA_Q_REQUIRES_ARENA
    ):
        decision = "blocked"
        reasons = (*reasons, "delta_q_requires_arena_confirmation_is_not_acceptance")
    registry_update: dict[str, object] = {"active_baseline_id_before": registry.active_baseline_id}
    if decision == "promote":
        registry_update["active_baseline_id_after"] = candidate_checkpoint_id
    else:
        registry_update["active_baseline_id_after"] = registry.active_baseline_id
    record = PromotionRecord(
        schema_version=SCHEMA_VERSION,
        promotion_id=promotion_id,
        candidate_checkpoint_id=candidate_checkpoint_id,
        baseline_checkpoint_id=baseline_checkpoint_id,
        opponent_pool_id=opponent_pool_id,
        seed_set_id=seed_set_id,
        arena_summary_path=str(arena_path),
        paired_eval_summary_path=str(paired_path or arena_path),
        normalized_metrics=metrics,
        thresholds=active_thresholds,
        decision=PromotionDecisionRecord(
            decision=decision, reasons=reasons, metrics=metrics, thresholds=active_thresholds
        ),
        registry_update=registry_update,
        created_at_unix_ms=created_at_unix_ms if created_at_unix_ms is not None else unix_ms_now(),
        checkpoint_eval_summary_path=str(paired_path) if paired_path is not None else None,
        delta_q_summary_path=str(delta_q_summary_path.expanduser().resolve(strict=True))
        if delta_q_summary_path is not None
        else None,
        evidence_seed=_extract_evidence_seed(arena_summary, checkpoint_eval_summary),
        seat_coverage_verified=_seat_coverage_verified(arena_summary),
        seat_coverage=_seat_coverage(arena_summary),
    )
    _validate_promotion_record(record)
    return record


def write_promotion_artifact(output_dir: Path, record: PromotionRecord) -> Path:
    _validate_promotion_record(record)
    path = output_dir / PROMOTIONS_DIRNAME / f"{record.promotion_id}.json"
    write_json_file(path, promotion_record_to_dict(record))
    return path


def registry_to_dict(registry: PopulationRegistry) -> dict[str, object]:
    payload = {
        "schema_version": registry.schema_version,
        "registry_id": registry.registry_id,
        "run_id": registry.run_id,
        "active_baseline_id": registry.active_baseline_id,
        "latest_candidate_id": registry.latest_candidate_id,
        "checkpoints": {key: checkpoint_entry_to_dict(value) for key, value in sorted(registry.checkpoints.items())},
        "seed_banks": {key: seed_bank_to_dict(value) for key, value in sorted(registry.seed_banks.items())},
        "opponent_pools": {key: opponent_pool_to_dict(value) for key, value in sorted(registry.opponent_pools.items())},
        "eval_schedule": eval_schedule_to_dict(registry.eval_schedule),
        "promotions": {key: promotion_record_to_dict(value) for key, value in sorted(registry.promotions.items())},
    }
    _reject_forbidden_fields(payload, "registry")
    return cast("dict[str, object]", payload)


def checkpoint_entry_to_dict(entry: CheckpointEntry) -> dict[str, object]:
    return cast("dict[str, object]", _json_ready_dataclass(entry))


def seed_bank_to_dict(seed_bank: SeedBank) -> dict[str, object]:
    return cast("dict[str, object]", _json_ready_dataclass(seed_bank))


def opponent_pool_to_dict(pool: OpponentPool) -> dict[str, object]:
    return cast("dict[str, object]", _json_ready_dataclass(pool))


def eval_schedule_to_dict(schedule: EvalSchedule) -> dict[str, object]:
    return cast("dict[str, object]", _json_ready_dataclass(schedule))


def promotion_record_to_dict(record: PromotionRecord) -> dict[str, object]:
    return cast("dict[str, object]", _json_ready_dataclass(record))


def write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    _reject_forbidden_fields(payload, str(path))
    _validate_json_payload(payload, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unix_ms_now() -> int:
    return time.time_ns() // 1_000_000


def _registry_from_dict(payload: Mapping[str, object]) -> PopulationRegistry:
    _require_exact_keys(
        payload,
        "registry",
        {
            "schema_version",
            "registry_id",
            "run_id",
            "active_baseline_id",
            "latest_candidate_id",
            "checkpoints",
            "seed_banks",
            "opponent_pools",
            "eval_schedule",
            "promotions",
        },
    )
    schema_version = _required_int(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema_version {schema_version!r}")
    checkpoints = {
        key: _dataclass_from_mapping(CheckpointEntry, value, f"checkpoints.{key}")
        for key, value in _required_mapping(payload, "checkpoints").items()
    }
    seed_banks = {
        key: _dataclass_from_mapping(SeedBank, value, f"seed_banks.{key}")
        for key, value in _required_mapping(payload, "seed_banks").items()
    }
    opponent_pools = {
        key: _dataclass_from_mapping(OpponentPool, value, f"opponent_pools.{key}")
        for key, value in _required_mapping(payload, "opponent_pools").items()
    }
    promotions = {
        key: _promotion_record_from_mapping(value, f"promotions.{key}")
        for key, value in _required_mapping(payload, "promotions").items()
    }
    return PopulationRegistry(
        schema_version=schema_version,
        registry_id=_required_str(payload, "registry_id"),
        run_id=_required_str(payload, "run_id"),
        active_baseline_id=_required_str(payload, "active_baseline_id"),
        latest_candidate_id=_optional_str(payload, "latest_candidate_id"),
        checkpoints=checkpoints,
        seed_banks=seed_banks,
        opponent_pools=opponent_pools,
        eval_schedule=_dataclass_from_mapping(
            EvalSchedule, _required_mapping(payload, "eval_schedule"), "eval_schedule"
        ),
        promotions=promotions,
    )


def _promotion_record_from_mapping(value: object, path: str) -> PromotionRecord:
    raw = _ensure_mapping(value, path)
    data = dict(raw)
    data["decision"] = _dataclass_from_mapping(
        PromotionDecisionRecord, _required_mapping(data, "decision"), f"{path}.decision"
    )
    return _dataclass_from_mapping(PromotionRecord, data, path)


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return cast("dict[str, object]", payload)


def _json_ready_dataclass(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _json_ready_dataclass(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_ready_dataclass(item) for item in value]
    if isinstance(value, list):
        return [_json_ready_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready_dataclass(item) for key, item in value.items()}
    return value


def _dataclass_from_mapping[T](cls: type[T], raw_value: object, path: str) -> T:
    raw = _ensure_mapping(raw_value, path)
    dataclass_fields = fields(cls)  # type: ignore[arg-type]
    allowed = {field.name for field in dataclass_fields}
    _require_exact_keys(raw, path, allowed)
    values: dict[str, object] = {}
    for item in dataclass_fields:
        value = raw[item.name]
        if item.name in {"seeds", "checkpoint_ids", "opponent_checkpoint_ids", "reasons"}:
            values[item.name] = tuple(cast("list[object]", value))
        else:
            values[item.name] = value
    return cls(**values)


def _replace_registry(registry: PopulationRegistry, **updates: object) -> PopulationRegistry:
    values = {field.name: getattr(registry, field.name) for field in fields(PopulationRegistry)}
    values.update(updates)
    return PopulationRegistry(**values)


def _replace_checkpoint(entry: CheckpointEntry, **updates: object) -> CheckpointEntry:
    values = {field.name: getattr(entry, field.name) for field in fields(CheckpointEntry)}
    values.update(updates)
    return CheckpointEntry(**values)


def _updated_active_baseline_pools(
    pools: Mapping[str, OpponentPool], active_baseline_id: str
) -> dict[str, OpponentPool]:
    updated: dict[str, OpponentPool] = {}
    for pool_id, pool in pools.items():
        if pool.strategy == "active_baseline_only":
            updated[pool_id] = OpponentPool(
                pool_id=pool.pool_id,
                strategy=pool.strategy,
                baseline_checkpoint_id=active_baseline_id,
                opponent_checkpoint_ids=(active_baseline_id,),
                checkpoint_ids=(active_baseline_id,),
                max_size=pool.max_size,
            )
        else:
            updated[pool_id] = pool
    return updated


def _validate_checkpoint_entry(entry: CheckpointEntry) -> None:
    _require_non_empty_id(entry.checkpoint_id, "checkpoint_id")
    if entry.role not in {"candidate", "champion", "rejected", "seed"}:
        raise ValueError("unsupported checkpoint role")
    if entry.weight_source not in {"raw", "ema"}:
        raise ValueError("unsupported checkpoint weight_source")
    if entry.status not in {"registered", "promoted", "rejected", "blocked"}:
        raise ValueError("unsupported checkpoint status")
    if not entry.path:
        raise ValueError("checkpoint path is required")
    if entry.path.endswith("/latest.pt") or Path(entry.path).name == "latest.pt":
        raise ValueError("mutable latest.pt cannot be registered directly")
    if not entry.path_sha256 and not entry.policy_json_sha256:
        raise ValueError("checkpoint must include path_sha256 or policy_json_sha256")
    if entry.path_sha256 is not None and not _is_sha256(entry.path_sha256):
        raise ValueError("path_sha256 must be a SHA-256 hex digest")
    if entry.policy_json_sha256 is not None and not _is_sha256(entry.policy_json_sha256):
        raise ValueError("policy_json_sha256 must be a SHA-256 hex digest")
    if entry.global_step is not None and entry.global_step < 0:
        raise ValueError("global_step must be non-negative")
    if entry.samples_seen is not None and entry.samples_seen < 0:
        raise ValueError("samples_seen must be non-negative")
    if entry.registered_at_unix_ms < 0:
        raise ValueError("registered_at_unix_ms must be non-negative")
    _validate_json_payload(checkpoint_entry_to_dict(entry), f"checkpoint.{entry.checkpoint_id}")


def _validate_checkpoint_artifact(entry: CheckpointEntry) -> None:
    path = Path(entry.path)
    if entry.path_sha256 is not None:
        if not path.is_file():
            raise ValueError(f"checkpoint artifact missing: {entry.checkpoint_id}")
        if file_sha256(path) != entry.path_sha256:
            raise ValueError(f"checkpoint artifact hash mismatch: {entry.checkpoint_id}")
        return
    if entry.policy_json_sha256 is not None:
        policy_json = path / "policy.json"
        if not policy_json.is_file():
            raise ValueError(f"checkpoint policy.json missing: {entry.checkpoint_id}")
        if file_sha256(policy_json) != entry.policy_json_sha256:
            raise ValueError(f"checkpoint policy_json hash mismatch: {entry.checkpoint_id}")


def _validate_seed_bank(seed_bank: SeedBank) -> None:
    _require_non_empty_id(seed_bank.seed_set_id, "seed_set_id")
    if not seed_bank.seeds:
        raise ValueError("seed bank must contain explicit seeds")
    if any(not isinstance(seed, int) for seed in seed_bank.seeds):
        raise ValueError("seed bank seeds must be integers")
    if seed_bank.games_per_seed < 1:
        raise ValueError("games_per_seed must be positive")
    if not math.isfinite(seed_bank.temperature) or seed_bank.temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    _validate_json_payload(seed_bank_to_dict(seed_bank), f"seed_bank.{seed_bank.seed_set_id}")


def _validate_opponent_pool(pool: OpponentPool) -> None:
    _require_non_empty_id(pool.pool_id, "pool_id")
    if pool.strategy != "active_baseline_only":
        raise ValueError("only active_baseline_only opponent pool is supported")
    if pool.max_size != 1:
        raise ValueError("minimal opponent pool max_size must be 1")
    if len(pool.checkpoint_ids) != 1:
        raise ValueError("active_baseline_only opponent pool must contain exactly one checkpoint")
    if pool.baseline_checkpoint_id != pool.checkpoint_ids[0]:
        raise ValueError("active_baseline_only baseline_checkpoint_id must match checkpoint_ids")
    if pool.opponent_checkpoint_ids != pool.checkpoint_ids:
        raise ValueError("active_baseline_only opponent_checkpoint_ids must match checkpoint_ids")


def _validate_active_baseline_pool(
    pool: OpponentPool, active_baseline_id: str, checkpoints: Mapping[str, CheckpointEntry]
) -> None:
    checkpoint_id = pool.checkpoint_ids[0]
    if pool.baseline_checkpoint_id != active_baseline_id:
        raise ValueError("active_baseline_only pool baseline must match active baseline")
    if pool.opponent_checkpoint_ids != (active_baseline_id,) or checkpoint_id != active_baseline_id:
        raise ValueError("active_baseline_only pool opponents must match active baseline")
    if checkpoint_id not in checkpoints:
        raise ValueError("active_baseline_only pool checkpoint is not registered")


def _validate_eval_schedule(schedule: EvalSchedule) -> None:
    _require_non_empty_id(schedule.seed_set_id, "eval_schedule.seed_set_id")
    _require_non_empty_id(schedule.opponent_pool_id, "eval_schedule.opponent_pool_id")
    if schedule.min_games < 1:
        raise ValueError("eval_schedule.min_games must be positive")
    _validate_thresholds(schedule.thresholds)
    _validate_json_payload(eval_schedule_to_dict(schedule), "eval_schedule")


def _validate_promotion_record(record: PromotionRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported promotion schema_version")
    _require_non_empty_id(record.promotion_id, "promotion_id")
    _require_non_empty_id(record.candidate_checkpoint_id, "candidate_checkpoint_id")
    _require_non_empty_id(record.baseline_checkpoint_id, "baseline_checkpoint_id")
    if record.decision.decision not in {"promote", "reject", "insufficient_games", "blocked"}:
        raise ValueError("unsupported promotion decision")
    if record.decision.decision == "promote" and any(
        reason == "insufficient_games" for reason in record.decision.reasons
    ):
        raise ValueError("insufficient_games cannot promote")
    _validate_thresholds(record.thresholds)
    _validate_metrics(record.normalized_metrics)
    _validate_metrics(record.decision.metrics)
    _validate_materialized_promotion_decision(record)
    _validate_json_payload(promotion_record_to_dict(record), f"promotion.{record.promotion_id}")


def _validate_materialized_promotion_decision(record: PromotionRecord) -> None:
    if record.normalized_metrics != record.decision.metrics:
        raise ValueError("promotion decision metrics mismatch")
    _require_configured_metric_gates(record.thresholds, record.decision.metrics)
    recomputed = _recompute_checkpoint_eval_decision(record.decision.metrics, record.thresholds)
    if _is_delta_q_blocked_record(record, recomputed):
        return
    if recomputed.decision != record.decision.decision or recomputed.reasons != record.decision.reasons:
        raise ValueError("promotion decision mismatch with configured thresholds")


def _is_delta_q_blocked_record(record: PromotionRecord, recomputed: PairedCheckpointEvalDecision) -> bool:
    if record.decision.decision != "blocked" or recomputed.decision != "promote":
        return False
    reasons = record.decision.reasons
    if reasons[: len(recomputed.reasons)] != recomputed.reasons:
        return False
    return any(reason.startswith("delta_q_") for reason in reasons[len(recomputed.reasons) :])


def _validate_delta_q_summary(summary: Mapping[str, object]) -> None:
    recommendation = _delta_q_recommendation(summary)
    arena_decision = _delta_q_arena_decision(summary)
    if recommendation in _DELTA_Q_REQUIRES_ARENA and arena_decision is None:
        return
    if arena_decision in _DELTA_Q_ARENA_REJECTS:
        return
    _validate_json_payload(summary, "delta_q_summary")


def _validate_arena_summary(
    arena_summary: Mapping[str, object], baseline: CheckpointEntry, candidate: CheckpointEntry
) -> None:
    raw_baseline = _ensure_mapping(arena_summary.get("baseline"), "arena_summary.baseline")
    _match_summary_identity(_required_str(raw_baseline, "path"), baseline, "arena baseline")
    weight_source = raw_baseline.get("weight_source")
    if isinstance(weight_source, str) and weight_source != baseline.weight_source:
        raise ValueError("arena baseline weight_source mismatch")
    raw_candidates = arena_summary.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("arena summary must contain candidate evidence")
    matched_candidate = False
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("arena summary candidates must be objects")
        candidate_path = raw_candidate.get("candidate_path") or raw_candidate.get("path")
        if not isinstance(candidate_path, str):
            continue
        try:
            _match_summary_identity(candidate_path, candidate, "arena candidate")
        except ValueError:
            continue
        weight_source = raw_candidate.get("weight_source")
        if isinstance(weight_source, str) and weight_source != candidate.weight_source:
            raise ValueError("arena candidate weight_source mismatch")
        result = raw_candidate.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("arena candidate result must be an object")
        matched_candidate = True
        break
    if not matched_candidate:
        raise ValueError("arena candidate identity mismatch between registry and arena summary")
    _validate_json_payload(arena_summary, "arena_summary")


def _arena_candidate_result(arena_summary: Mapping[str, object], candidate: str) -> Mapping[str, object]:
    raw_candidates = arena_summary.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("arena summary must contain candidate evidence")
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        if raw_candidate.get("candidate_path") == candidate or raw_candidate.get("path") == candidate:
            result = raw_candidate.get("result")
            if not isinstance(result, Mapping):
                raise ValueError("arena candidate result must be an object")
            return result
    raise ValueError("arena candidate evidence not found")


def _validate_checkpoint_eval_against_arena(
    arena_summary: Mapping[str, object],
    metrics: Mapping[str, float | int | str | None],
    checkpoint_eval_summary: Mapping[str, object],
    seed_bank: SeedBank,
    candidate_entry: CheckpointEntry,
) -> None:
    _required_str(checkpoint_eval_summary, "candidate")
    arena_metrics = _arena_candidate_result(arena_summary, candidate_entry.path)
    normalized = normalize_paired_arena_metrics(arena_metrics)
    for key, value in metrics.items():
        if key in normalized and normalized[key] != value:
            raise ValueError(f"checkpoint_eval metric mismatch for {key}")
    if normalized.get("games") != metrics.get("games"):
        raise ValueError("checkpoint_eval games mismatch")
    evidence_seed = _extract_evidence_seed(arena_summary, checkpoint_eval_summary)
    if evidence_seed is not None and evidence_seed not in seed_bank.seeds:
        raise ValueError("promotion evidence seed is not in seed bank")
    arena_seed = _arena_config_value(arena_summary, "seed")
    summary_seed = checkpoint_eval_summary.get("seed")
    if arena_seed is not None and summary_seed is not None and arena_seed != summary_seed:
        raise ValueError("checkpoint_eval seed mismatch")
    for key in ("games", "temperature"):
        arena_value = _arena_config_value(arena_summary, key)
        if arena_value is not None and key in metrics and arena_value != metrics[key]:
            raise ValueError(f"checkpoint_eval {key} mismatch")


def _arena_config_value(arena_summary: Mapping[str, object], key: str) -> object | None:
    config = arena_summary.get("config")
    if isinstance(config, Mapping):
        return config.get(key)
    return None


def _extract_evidence_seed(
    arena_summary: Mapping[str, object], checkpoint_eval_summary: Mapping[str, object]
) -> int | None:
    arena_seed = _arena_config_value(arena_summary, "seed")
    summary_seed = checkpoint_eval_summary.get("seed")
    seed = arena_seed if arena_seed is not None else summary_seed
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("promotion evidence seed must be int when present")
    return seed


def _seat_coverage(arena_summary: Mapping[str, object]) -> dict[str, object]:
    coverage: dict[str, object] = {}
    if "seat_rotations" in arena_summary:
        coverage["seat_rotations"] = arena_summary["seat_rotations"]
    result = _first_arena_result(arena_summary)
    if "seat_rotations" in result:
        coverage["seat_rotations"] = result["seat_rotations"]
    if "seat_coverage" in result:
        coverage["seat_coverage"] = result["seat_coverage"]
    return coverage


def _seat_coverage_verified(arena_summary: Mapping[str, object]) -> bool:
    coverage = _seat_coverage(arena_summary)
    rotations = coverage.get("seat_rotations")
    return isinstance(rotations, int) and not isinstance(rotations, bool) and rotations >= 4


def _first_arena_result(arena_summary: Mapping[str, object]) -> Mapping[str, object]:
    candidates = arena_summary.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        result = candidates[0].get("result")
        if isinstance(result, Mapping):
            return result
    return {}


def _delta_q_recommendation(summary: Mapping[str, object]) -> str | None:
    value = summary.get("recommendation")
    return value if isinstance(value, str) else None


def _delta_q_arena_decision(summary: Mapping[str, object]) -> str | None:
    value = summary.get("arena_decision")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        decision = value.get("decision")
        return decision if isinstance(decision, str) else None
    return None


def _require_configured_metric_gates(
    thresholds: Mapping[str, float | int | None], metrics: Mapping[str, float | int | str | None]
) -> None:
    for threshold_name, threshold_value in thresholds.items():
        if threshold_value is None or threshold_name == "min_games":
            continue
        metric_name = _threshold_metric_name(threshold_name)
        if metric_name not in metrics or metrics[metric_name] is None:
            raise ValueError(f"missing configured metric gate: {metric_name}")
        metric_value = metrics[metric_name]
        if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
            raise TypeError(f"configured metric gate {metric_name} must be numeric")
        if (
            metric_name == "illegal_action_count"
            and threshold_name.startswith("max_")
            and metric_value > threshold_value
        ):
            raise ValueError(f"configured metric gate failed: {metric_name}")


def _recompute_checkpoint_eval_decision(
    metrics: Mapping[str, float | int | str | None], thresholds: Mapping[str, float | int | None]
) -> PairedCheckpointEvalDecision:
    return decide_paired_checkpoint_eval(
        metrics,
        PairedCheckpointEvalThresholds(
            max_fourth_rate_delta=_optional_threshold_float(thresholds, "max_fourth_rate_delta"),
            min_mean_u_a_delta=_optional_threshold_float(thresholds, "min_mean_u_a_delta"),
            min_top2_delta=_optional_threshold_float(thresholds, "min_top2_delta"),
            min_games=_required_threshold_int(thresholds, "min_games"),
        ),
    )


def _optional_threshold_float(thresholds: Mapping[str, float | int | None], key: str) -> float | None:
    value = thresholds.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"thresholds.{key} must be numeric or null")
    return float(value)


def _required_threshold_int(thresholds: Mapping[str, float | int | None], key: str) -> int:
    value = thresholds.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"thresholds.{key} must be an integer")
    return value


def _threshold_metric_name(threshold_name: str) -> str:
    if threshold_name.startswith("max_"):
        return threshold_name[4:]
    if threshold_name.startswith("min_"):
        return threshold_name[4:]
    return threshold_name


def _match_summary_identity(summary_value: str, entry: CheckpointEntry, label: str) -> None:
    if summary_value not in {entry.checkpoint_id, entry.path}:
        raise ValueError(f"{label} identity mismatch between registry and eval summary")


def _metric_map(metrics: Mapping[str, object]) -> dict[str, float | int | str | None]:
    normalized: dict[str, float | int | str | None] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            raise TypeError(f"metrics.{key} must not be bool")
        if isinstance(value, int | float):
            if not math.isfinite(float(value)):
                raise ValueError(f"metrics.{key} must be finite")
            normalized[key] = value
        elif isinstance(value, str) or value is None:
            normalized[key] = value
        else:
            raise TypeError(f"metrics.{key} has unsupported type {type(value).__name__}")
    return normalized


def _validate_metrics(metrics: Mapping[str, object]) -> None:
    _metric_map(metrics)


def _validate_thresholds(thresholds: Mapping[str, object]) -> None:
    for key, value in thresholds.items():
        if isinstance(value, bool):
            raise TypeError(f"thresholds.{key} must not be bool")
        if value is None:
            continue
        if not isinstance(value, int | float):
            raise TypeError(f"thresholds.{key} must be numeric or null")
        if not math.isfinite(float(value)):
            raise ValueError(f"thresholds.{key} must be finite")


def _validate_json_payload(value: object, path: str) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        _reject_forbidden_fields(value, path)
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string key")
            _validate_json_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _validate_json_payload(item, f"{path}[{idx}]")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _reject_forbidden_fields(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lower = str(key).lower()
            if any(token in lower for token in _FORBIDDEN_FIELD_TOKENS):
                raise ValueError(f"{path} contains out-of-scope population field {key!r}")
            _reject_forbidden_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_forbidden_fields(item, f"{path}[{idx}]")


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _ensure_mapping(payload.get(key), key)


def _ensure_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return cast("Mapping[str, object]", value)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("reasons must be a list of strings")
    reasons: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("reasons must be non-empty strings")
        reasons.append(item)
    return tuple(reasons)


def _require_exact_keys(payload: Mapping[str, object], path: str, allowed: set[str]) -> None:
    keys = set(payload)
    unknown = keys - allowed
    missing = allowed - keys
    if unknown:
        raise ValueError(f"{path} contains unsupported fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{path} missing required fields: {sorted(missing)}")


def _require_non_empty_id(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _mapping_sha256(value: Mapping[str, object]) -> str:
    _validate_json_payload(value, "model_config")
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
