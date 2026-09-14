"""WP-10 student distillation — student model, loss, and training.

Owns the frozen :class:`DistillationConfig`, the actor-visible
:class:`StudentModel`, the real encoder feature path, the masked
distillation loss with BC anchors, and deterministic student training. The
gate and registry live in :mod:`hydra2.distillation._teacher_gate`, case
observations in :mod:`hydra2.distillation._teacher_cases`, trajectory
records in :mod:`hydra2.distillation._teacher_records`, and five-arm
evaluation in :mod:`hydra2.distillation._teacher_eval`, so each file stays
inside the review-size ceiling.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from hydra2.contracts.common import ContractError
from hydra2.distillation._teacher_cases import _case_observation as _case_observation
from hydra2.distillation._teacher_gate import _NUM_ACTIONS as _NUM_ACTIONS
from hydra2.distillation._teacher_gate import _REAL_FEATURE_DIM as _REAL_FEATURE_DIM
from hydra2.distillation._teacher_gate import _real_candidate_spec as _real_candidate_spec
from hydra2.distillation._teacher_gate import _spec_digest_of as _spec_digest_of
from hydra2.distillation._teacher_records import TrajectoryRecord

if TYPE_CHECKING:
    from hydra2.contracts.observation import ActorObservation
    from hydra2.distillation._teacher_gate import TeacherJustification
# ---------------------------------------------------------------------------
# Distillation — student model, loss, BC anchors, legal mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Frozen distillation hyperparameters — preserves BC anchors."""

    learning_rate: float = 0.001
    w_policy: float = 1.0  # KL weight
    w_value: float = 0.5  # vector MSE weight
    w_bc: float = 0.1  # behavior-cloning anchor weight
    temperature: float = 1.0
    max_updates: int = 16
    batch_size: int = 8
    l2_reg: float = 0.0

    def __post_init__(self) -> None:
        if not (0 < self.learning_rate < 1):
            raise ContractError(f"learning_rate {self.learning_rate} invalid")
        for name in ("w_policy", "w_value", "w_bc"):
            v = getattr(self, name)
            if v < 0 or not math.isfinite(v):
                raise ContractError(f"{name} {v} invalid")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ContractError(f"temperature {self.temperature} invalid")
        if self.max_updates <= 0:
            raise ContractError("max_updates must be positive")
        if self.batch_size <= 0:
            raise ContractError("batch_size must be positive")


class StudentModel(nn.Module):
    """Tiny actor-visible student — no privileged inputs."""

    def __init__(self, *, num_actions: int, d_model: int = 32) -> None:
        super().__init__()
        self.num_actions = num_actions
        # Actor-visible features: REAL model_input_v1 encoder path — the 48-dim
        # vector is derived from the actor-visible observation tensor
        # (concealed counts, scores, wall state, seats/phase), never from a hash
        # expansion. See `features_for_record`.
        self.encoder = nn.Sequential(
            nn.Linear(_REAL_FEATURE_DIM, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 4)

    def forward(
        self, features: torch.Tensor, *, legal_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Return UNMASKED logits + value; every consumer MUST mask illegal actions."""
        h = self.encoder(features)
        logits = self.policy_head(h)
        values = self.value_head(h)
        # Legal-mask shape check only — masking is enforced by callers (loss at
        # compute_distillation_loss, selection in act); forward keeps raw logits.
        if legal_mask is not None:
            if legal_mask.shape[-1] != self.num_actions:
                raise ContractError(f"legal_mask dim {legal_mask.shape[-1]} != {self.num_actions}")
            pass  # body intentionally empty: shape check only, masking lives in callers
        return {"policy_logits": logits, "value": values}

    def act(self, features: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Argmax over masked logits (illegal forced to -inf)."""
        out = self.forward(features, legal_mask=legal_mask)
        logits = out["policy_logits"]
        # Illegal mask -> -inf
        masked = torch.where(
            legal_mask.bool(), logits, torch.tensor(float("-inf"), device=logits.device)
        )
        return torch.argmax(masked, dim=-1)


def _features_from_actor_observation(obs: ActorObservation) -> torch.Tensor:
    """REAL actor-visible features for one observation (model_input_v1 path).

    Encodes via :func:`hydra2.models.encoder.encode_observations` and reduces
    the batch tensors to a fixed 48-dim student vector: 34 concealed-tile
    counts ([-1,1] normalized), 4 scores (/40000), live-wall remaining (/136),
    4 actor one-hot + 4 turn-actor one-hot + 1 ippatsu-any flag. Deterministic;
    privileged fields never enter (the encoder only sees ActorObservation).
    """
    from hydra2.models.encoder import encode_observations

    try:
        batch = encode_observations([obs])
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"WP-10 blocked: observation encoding failed: {exc}") from exc
    feats = batch.features
    counts = feats["concealed_hand_counts"][0].to(torch.float32) / 4.0
    scores = feats["scores"][0].to(torch.float32) / 40000.0
    wall_left = feats["live_wall_tiles_remaining"][0].to(torch.float32).reshape(1) / 136.0
    actor_oh = (
        F.one_hot(feats["actor"][0].reshape(()).to(torch.int64), num_classes=4)
        .to(torch.float32)
        .reshape(4)
    )
    turn_oh = (
        F.one_hot(feats["turn_actor"][0].reshape(()).to(torch.int64), num_classes=4)
        .to(torch.float32)
        .reshape(4)
    )
    ippatsu = feats["ippatsu_active"][0].to(torch.float32).reshape(-1)
    ippatsu_any = (ippatsu.sum() > 0).to(torch.float32).reshape(1)
    vec = torch.cat([counts, scores, wall_left, actor_oh, turn_oh, ippatsu_any])
    if vec.numel() != _REAL_FEATURE_DIM:
        raise ContractError(f"WP-10 blocked: real feature dim {vec.numel()} != {_REAL_FEATURE_DIM}")
    return vec.to(torch.float32)


def features_for_record(record: TrajectoryRecord) -> torch.Tensor:
    """REAL features for a trajectory record — fail closed without provenance.

    Rebuilds the case observation from the record provenance (case_id, actor,
    teacher id, seed material) against the live CandidateSpec — whose digest
    must match the record's teacher_spec_hash — then encodes via the real
    model_input_v1 path. Raises ContractError when reconstruction is
    impossible (never hash-expands the observation hash).
    """
    if not isinstance(record, TrajectoryRecord):
        raise ContractError(f"expected TrajectoryRecord, got {type(record)}")
    prov = dict(record.provenance)
    try:
        case_id = str(prov["case_id"])
        teacher_id = str(prov["teacher_candidate_id"])
        actor = int(prov["actor"])
        seed_material = bytes.fromhex(str(prov["seed_material_hex"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ContractError(
            f"WP-10 blocked: record provenance lacks case reconstruction fields: {exc}"
        ) from exc
    spec = _real_candidate_spec(teacher_id)
    if _spec_digest_of(spec) != record.teacher_spec_hash:
        raise ContractError("WP-10 blocked: record teacher_spec_hash != live CandidateSpec digest")
    obs = _case_observation(
        case_id=case_id,
        teacher_id=teacher_id,
        actor=actor,
        spec=spec,
        seed_material=seed_material,
    )
    if obs.observation_hash != record.observation_hash:
        raise ContractError(
            "WP-10 blocked: reconstructed observation hash != record observation_hash"
        )
    if tuple(obs.legal_mask) != record.legal_mask:
        raise ContractError("WP-10 blocked: reconstructed legal mask != record mask")
    return _features_from_actor_observation(obs)


def build_student_model(*, num_actions: int | None = None) -> StudentModel:
    """Construct StudentModel, defaulting num_actions to import-time _NUM_ACTIONS."""
    n = num_actions if num_actions is not None else _NUM_ACTIONS
    return StudentModel(num_actions=n)


def compute_distillation_loss(
    *,
    student_logits: torch.Tensor,
    teacher_policy: torch.Tensor,
    legal_mask: torch.Tensor,
    student_value: torch.Tensor | None = None,
    teacher_vector: torch.Tensor | None = None,
    anchor_logits: torch.Tensor | None = None,
    anchor_target: torch.Tensor | None = None,
    config: DistillationConfig,
) -> dict[str, torch.Tensor]:
    """Distillation loss preserves BC anchors and legal mask.

    w_policy * KL(teacher || student)  over legal actions only
    w_value  * MSE(student_value, teacher_vector)
    w_bc     * CE(anchor_logits) if provided (BC anchor)
    Illegal probs exactly zero — enforced by masking.
    """
    if student_logits.shape != teacher_policy.shape:
        raise ContractError(
            f"student {tuple(student_logits.shape)} vs teacher {tuple(teacher_policy.shape)}"
        )
    if legal_mask.shape != teacher_policy.shape:
        raise ContractError("legal_mask shape mismatch")
    # Legal mask must have at least one legal per row
    if torch.all(legal_mask.any(dim=-1)).item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
        raise ContractError("legal_mask all false for some row")
    # Teacher illegal must be 0
    if torch.all((teacher_policy * (~legal_mask.bool()).float()) == 0).item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
        raise ContractError("teacher illegal has non-zero mass")
    # Student masked log_softmax
    masked_logits = torch.where(
        legal_mask.bool(), student_logits, torch.tensor(float("-inf"), device=student_logits.device)
    )
    log_probs = F.log_softmax(masked_logits / config.temperature, dim=-1)
    # Teacher already zero on illegal, safe
    # KL = sum over LEGAL actions only of teacher * (log teacher - log student);
    # illegal log_probs are -inf and 0 * -inf is NaN, so exclude illegal terms
    # explicitly (they contribute exactly 0: teacher mass is 0 there, asserted above).
    # Use teacher * log teacher - teacher * log student
    eps = 1e-12
    teacher_clamped = teacher_policy.clamp_min(eps)
    # Only where legal, teacher may have support
    kl_terms = teacher_policy * (torch.log(teacher_clamped) - log_probs)
    kl_per_row = torch.sum(
        torch.where(legal_mask.bool(), kl_terms, torch.zeros_like(kl_terms)), dim=-1
    )
    # Masked mean over rows
    loss_policy = kl_per_row.mean()

    losses: dict[str, torch.Tensor] = {"policy": loss_policy * config.w_policy}

    if student_value is not None and teacher_vector is not None:
        if student_value.shape != teacher_vector.shape:
            raise ContractError("value shape mismatch")
        mse = F.mse_loss(student_value.float(), teacher_vector.float())
        losses["value"] = mse * config.w_value
    elif (student_value is None) != (teacher_vector is None):
        raise ContractError("value both or neither required")

    if anchor_logits is not None and config.w_bc > 0:
        if anchor_target is None:
            raise ContractError("anchor_target required when w_bc>0 and anchor_logits present")
        # BC anchor: masked CE
        masked_anchor = torch.where(
            legal_mask.bool(),
            anchor_logits,
            torch.tensor(float("-inf"), device=anchor_logits.device),
        )
        ce = F.cross_entropy(masked_anchor, anchor_target.long(), reduction="mean")
        losses["bc"] = ce * config.w_bc
    elif config.w_bc > 0 and anchor_logits is None:
        # No anchor provided but w_bc>0 — allowable if not anchored run
        losses["bc"] = torch.tensor(0.0, device=student_logits.device)
    # sum() with int start would infer Literal[0] | Tensor; use Tensor start for type safety
    total: torch.Tensor = sum(losses.values(), torch.tensor(0.0, device=student_logits.device))
    losses["total"] = total
    # Finite check
    for k, v in losses.items():
        if torch.isfinite(v).all().item() is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            raise ContractError(f"loss {k} non-finite {v}")
    return losses


def train_student_distillation(
    *,
    justification: TeacherJustification,
    records: tuple[TrajectoryRecord, ...],
    config: DistillationConfig | None = None,
    seed: int = 0,
) -> tuple[StudentModel, list[float]]:
    """Deterministic distillation over trajectories — preserves BC anchors and mask.

    Uses deterministic optimizer steps (seeded init, deterministic torch algorithms).
    """
    cfg = config if config is not None else DistillationConfig()
    n_actions = len(records[0].legal_mask) if len(records) > 0 else _NUM_ACTIONS
    _ = torch.manual_seed(seed)  # seeded init pins parameter order (see docstring)
    with contextlib.suppress(Exception):  # why-broad: determinism opt-in is best-effort
        torch.use_deterministic_algorithms(True)
    student = build_student_model(num_actions=n_actions)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2_reg
    )

    # Prepare tensors
    losses_trace: list[float] = []
    # Convert records to tensors once (deterministic order)
    features = torch.stack([features_for_record(r) for r in records])
    teacher_policies = torch.tensor([list(r.teacher_policy) for r in records], dtype=torch.float32)
    legal_masks = torch.tensor([list(r.legal_mask) for r in records], dtype=torch.bool)
    teacher_vectors = torch.tensor([list(r.vector_return) for r in records], dtype=torch.float32)

    # BC anchor target: teacher argmax (deterministic; the argmax is legal because
    # the teacher policy has support only on the exact legal mask).
    anchor_targets = torch.argmax(teacher_policies, dim=-1)

    _ = student.train()
    losses: dict[str, torch.Tensor] = {}
    for _ in range(cfg.max_updates):
        # Mini-batch loop deterministic (shuffle via seeded permutation)
        gen = torch.Generator().manual_seed(seed + _)
        perm = torch.randperm(len(records), generator=gen)
        for start in range(0, len(records), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            b_feat = features[idx]
            b_teacher = teacher_policies[idx]
            b_mask = legal_masks[idx]
            b_vec = teacher_vectors[idx]
            b_anchor_tgt = anchor_targets[idx]
            out: dict[str, torch.Tensor] = student(b_feat, legal_mask=b_mask)
            # Need anchor_logits for BC: use student logits as anchor logits (preserves BC)
            anchor_logits = out["policy_logits"] if cfg.w_bc > 0 else None
            losses = compute_distillation_loss(
                student_logits=out["policy_logits"],
                teacher_policy=b_teacher,
                legal_mask=b_mask,
                student_value=out["value"],
                teacher_vector=b_vec,
                anchor_logits=anchor_logits,
                anchor_target=b_anchor_tgt if cfg.w_bc > 0 else None,
                config=cfg,
            )
            _ = optimizer.zero_grad()
            _ = losses["total"].backward()
            # Gradient clipping per spec
            _ = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            _ = optimizer.step()
        # Trace the last batch total per epoch (empty run traces 0.0).
        if len(losses) > 0:
            losses_trace.append(float(losses["total"].detach().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
        else:
            losses_trace.append(0.0)

    return student, losses_trace
