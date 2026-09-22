"""WP-05B real-model adapter: ``ModelOutput``/rows bridge for supervised training.

Day-one training runs the real :class:`Hydra2BaselineModel` over real
:class:`AuthoritativeParquetDataset` batches.  The model speaks
:class:`ModelOutput` / :class:`ActorTensorBatch` while
:func:`compute_supervised_loss` speaks plain dicts, so this module owns the
two narrow conversions — and nothing else (no loss math, no optimizer, no
dataset edits):

- :func:`encode_actor_rows` reuses :func:`encode_observations` (the existing
  encoder path; no reimplemented features) and guards the returned batch
  shapes.
- :func:`model_output_to_loss_dict` maps :class:`ModelOutput` onto the
  ``policy_logits`` / ``placement_logits`` / ``value_vector`` /
  ``event_logits`` / ``belief_logits`` keys :func:`compute_supervised_loss`
  reads, asserting ``[B,4,4]`` / ``[B,4]`` at return (covers ModelData P2-1).

Target conventions (Wave C1 decisions, documented here at the boundary):

- ``placement_target`` is 0-based ``[B,4]`` per-seat rank indices with a
  range assert at the loss; the ``-1`` bridge to ``utility()`` 1..4 ranks
  lives wherever ranks are reported, not in these logits.
- ``w_value`` defaults to ``0.0`` (explicit enable only); zero-weight heads
  MAY be absent from the loss dict, but this adapter always carries the
  placement/value keys so enabling them later needs no rewiring.
- Belief loss keeps the index-CE contract (per-head ``[B]`` class indices
  into ``belief_logits``); 34-class distribution support is explicitly
  DEFERRED.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import ActorTensorBatch, encode_observations
from hydra2.models.model import ModelOutput
from hydra2.models.schema import BASELINE_ACTION_COUNT

if TYPE_CHECKING:
    from hydra2.contracts.observation_actor import ActorObservation

__all__ = [
    "encode_actor_rows",
    "model_output_to_loss_dict",
]


def _require_batch_tensor(
    value: object, *, name: str, batch_size: int, extra_dims: tuple[int, ...]
) -> torch.Tensor:
    """Narrow ``value`` to a tensor of shape ``(batch_size, *extra_dims)``."""
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} must be a Tensor, got {type(value).__name__}")
    if tuple(value.shape) != (batch_size, *extra_dims):
        raise ContractError(
            f"{name} must be [{batch_size}, {', '.join(str(d) for d in extra_dims)}]"
            f", got {tuple(value.shape)}"
        )
    return value


def model_output_to_loss_dict(output: ModelOutput) -> dict[str, Any]:
    """Map a real :class:`ModelOutput` onto the supervised-loss dict contract.

    Returns ``policy_logits`` ``[B,A]``, ``placement_logits`` ``[B,4,4]``,
    ``value_vector`` ``[B,4]``, plus ``event_logits`` / ``belief_logits``
    head dicts (each entry ``[B,E]`` sharing the batch size).  Every shape is
    asserted here so a head-spec drift fails at the adapter, not deep in the
    loss (ModelData P2-1).
    """
    if not isinstance(output, ModelOutput):
        raise ContractError(f"expected ModelOutput, got {type(output).__name__}")
    policy: torch.Tensor = output.policy_logits
    if not isinstance(policy, torch.Tensor) or policy.dim() != 2:
        raise ContractError(f"policy_logits must be [B,A], got {type(policy).__name__}")
    batch_size: int = policy.shape[0]
    if policy.shape[1] != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"policy_logits A {policy.shape[1]} != baseline {BASELINE_ACTION_COUNT}"
        )
    placement = _require_batch_tensor(
        output.placement_logits,
        name="placement_logits",
        batch_size=batch_size,
        extra_dims=(4, 4),
    )
    value = _require_batch_tensor(
        output.value_vector,
        name="value_vector",
        batch_size=batch_size,
        extra_dims=(4,),
    )
    checked: dict[str, dict[str, torch.Tensor]] = {}
    for field_name in ("event_logits", "belief_logits"):
        mapping: object = getattr(output, field_name)
        if not isinstance(mapping, Mapping):
            raise ContractError(f"{field_name} must be a mapping, got {type(mapping).__name__}")
        heads: dict[str, torch.Tensor] = {}
        for head_id_any, logits_any in mapping.items():
            if not isinstance(logits_any, torch.Tensor) or logits_any.dim() != 2:
                raise ContractError(f"{field_name}[{head_id_any!r}] must be [B,E]")
            if logits_any.shape[0] != batch_size:
                raise ContractError(
                    f"{field_name}[{head_id_any!r}] batch {logits_any.shape[0]}"
                    f" != policy batch {batch_size}"
                )
            head_id: object = head_id_any
            heads[str(head_id)] = logits_any
        checked[field_name] = heads
    return {
        "policy_logits": policy,
        "placement_logits": placement,
        "value_vector": value,
        "event_logits": checked["event_logits"],
        "belief_logits": checked["belief_logits"],
    }


def encode_actor_rows(rows: Sequence[ActorObservation]) -> ActorTensorBatch:
    """Encode actor rows into an :class:`ActorTensorBatch` for the real model.

    Args:
        rows: non-empty sequence of :class:`ActorObservation` (actor-visible
            only; privileged rows have no representation here).

    Delegates to :func:`encode_observations` — the existing encoder path, so
    no feature derivation is reimplemented — then asserts the returned batch
    carries this call's batch size with ``legal_mask`` ``[B,A]`` matching the
    baseline action count (ModelData P2-1).
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        msg = f"rows must be a sequence of ActorObservation, got {type(rows).__name__}"
        raise ContractError(msg)
    observations: list[ActorObservation] = list(rows)
    batch: ActorTensorBatch = encode_observations(observations)
    batch_size = len(observations)
    if batch.history_mask.dim() != 2 or batch.history_mask.shape[0] != batch_size:
        raise ContractError(
            f"history_mask batch {tuple(batch.history_mask.shape)} != [{batch_size}, T]"
        )
    if tuple(batch.legal_mask.shape) != (batch_size, BASELINE_ACTION_COUNT):
        raise ContractError(
            f"legal_mask must be [{batch_size},{BASELINE_ACTION_COUNT}]"
            f", got {tuple(batch.legal_mask.shape)}"
        )
    if tuple(batch.actor_seats.shape) != (batch_size,):
        raise ContractError(
            f"actor_seats must be [{batch_size}], got {tuple(batch.actor_seats.shape)}"
        )
    return batch
