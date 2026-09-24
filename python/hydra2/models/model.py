"""Hydra2 baseline actor model — SDPA, dense heads, masked policy, diagnostics.

Baseline is transformer over bucketed histories with explicit masks
(``True`` = participate). Uses ``torch.nn.functional.scaled_dot_product_attention``
for dense attention; evaluation dropout is exactly ``0.0``. Cache and
full-history encodings agree on valid prefix when masks are applied.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # reason: canonical PyTorch alias; upstream docs use F. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError, DigestText
from hydra2.contracts.event_vocab import EVENT_KINDS
from hydra2.models.schema import (
    BASELINE_ACTION_COUNT,
    HISTORY_BUCKET_LENGTHS,
    _feature_derivation_hash,
    compute_model_spec_digest,
    model_input_schema_digest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.models.encoder import ActorTensorBatch

_NUM_EVENT_KINDS = len(EVENT_KINDS)
_DEFAULT_D_MODEL = 128
_DEFAULT_N_HEADS = 4
_DEFAULT_N_LAYERS = 2
_DEFAULT_D_FF = 256
_DEFAULT_DROPOUT = 0.1

#: Big-trunk defaults (Mortal-class pre-RL imitation base): 8 layers x d256
#: x 8 heads, d_ff 1024 — exact total 9914166 params. The architecture id
#: selects the default row; explicit config parameters always win, so the
#: baseline id with explicit dims is unchanged and the big id without dims
#: can never silently inherit baseline widths.
_BIG_D_MODEL = 256
_BIG_N_HEADS = 8
_BIG_N_LAYERS = 8
_BIG_D_FF = 1024
_ARCH_DEFAULTS: dict[str, dict[str, int | float]] = {
    "hydra2_baseline_transformer_v1": {
        "d_model": _DEFAULT_D_MODEL,
        "n_heads": _DEFAULT_N_HEADS,
        "n_layers": _DEFAULT_N_LAYERS,
        "d_ff": _DEFAULT_D_FF,
        "dropout": _DEFAULT_DROPOUT,
    },
    "hydra2_big_transformer_v1": {
        "d_model": _BIG_D_MODEL,
        "n_heads": _BIG_N_HEADS,
        "n_layers": _BIG_N_LAYERS,
        "d_ff": _BIG_D_FF,
        "dropout": _DEFAULT_DROPOUT,
    },
}

__all__ = [
    "_ARCH_DEFAULTS",
    "Hydra2BaselineModel",
    "ModelOutput",
    "masked_policy",
    "select_actions",
    "validate_actor_batch",
]


def validate_actor_batch(batch: ActorTensorBatch, action_count: int | None = None) -> None:
    """Actor-batch contract check (shapes + nonterminal legal mask).

    Shape checks are host-side Python ints (zero syncs). The legal-mask
    gate is a device-side assert on CUDA (violations trip on the next
    sync and poison the context — abort-is-abort for fail-closed
    training); CPU/eval keeps the exact error. Runs BEFORE a compiled
    forward so the inductor graph holds zero device→host syncs.
    """
    if batch.history_mask.shape[0] != batch.legal_mask.shape[0]:
        raise ContractError("batch size mismatch between history_mask and legal_mask")
    if action_count is not None and batch.legal_mask.shape[1] != action_count:
        raise ContractError(
            f"legal_mask A {batch.legal_mask.shape[1]} != model action_count {action_count}"
        )
    _fail_closed_actor_rows(batch.legal_mask)
    if batch.history_mask.shape != batch.features["history_event_kind"].shape:
        raise ContractError("history_mask vs history_event_kind shape mismatch")


def _fail_closed_actor_rows(legal_mask: torch.Tensor) -> None:
    """Legal-rows gate: device assert on CUDA, exact raise elsewhere.

    Twin of ``hydra2.training.objectives_loss._fail_closed_gate`` (kept separate:
    models must not import training — layer direction).
    """
    pred = legal_mask.any(dim=1).all()
    cuda = False
    with contextlib.suppress(Exception):
        disabled = os.environ.get("HYDRA2_DISABLE_DEVICE_ASSERTS", "").strip().lower()
        cuda = pred.is_cuda and disabled not in ("1", "true", "yes", "on")
    if cuda:
        with contextlib.suppress(Exception):
            torch._assert_async(pred)
            return
    if bool(pred.item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # reason: CPU/eval-only host sync for contract; CUDA path returns above. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("nonterminal batch requires at least one legal per row")


def masked_policy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Masked softmax — illegal probability exactly zero.

    ``legal_mask`` is bool ``[B,A]`` with ``True`` = legal. Requires at least
    one legal per row and ``logits.shape[-1] == legal_mask.shape[-1]``.
    """
    if logits.shape[-1] != legal_mask.shape[-1]:
        raise ContractError(
            f"policy_logits last dim {logits.shape[-1]} != legal_mask {legal_mask.shape[-1]}"
        )
    if legal_mask.dtype != torch.bool:
        raise ContractError("legal_mask must be bool dtype")
    # Eager-only legal-rows check: under torch.compile the guard folds away
    # (short-circuit skips the .item()) so the graph holds no host sync
    # (dynamo-traced _check_tensor_all on the mask tensor graph-breaks every
    # forward). Compiled callers pre-validate via validate_actor_batch;
    # eager callers keep the identical error.
    if not torch.compiler.is_compiling() and not bool(legal_mask.any(dim=-1).all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # reason: eager-only host sync for contract (short-circuit skips it under compile; compiled path pre-validates). Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("masked_policy requires at least one legal per row")
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(masked, dim=-1)
    # Ensure illegal entries are exactly zero (numerical safety).
    return torch.where(legal_mask, probs, torch.zeros_like(probs))


def select_actions(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Deterministic argmax after masking; tie is first legal max (spec: CandidateSpec)."""
    if logits.shape[-1] != legal_mask.shape[-1]:
        raise ContractError("shape mismatch in select_actions")
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.argmax(masked, dim=-1)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Inference output (policy_logits [B,A] with width equal to the legal
    mask width; placement_logits [B,4,4] seat x final-rank; value_vector
    [B,4] retaining all four seats with the root seat selected only at the
    root decision; event/belief head definitions and target support frozen in
    the model spec; diagnostics carry tensor-derived actor-visible values
    only; SDPA boolean mask True means participate with evaluation dropout
    exactly 0.0)."""

    policy_logits: torch.Tensor  # [B,A]
    placement_logits: torch.Tensor  # [B,4,4]
    value_vector: torch.Tensor  # [B,4]
    event_logits: Mapping[str, torch.Tensor]
    belief_logits: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
    utility_id: str
    utility_manifest_hash: DigestText
    model_identity: DigestText


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ContractError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        # Fused QKV (item 1): single Linear[D,3D] reads x once instead of 3x;
        # chunk restores q/k/v views, autograd splits per slice. Kept for
        # exact math + eval latency, not training wall.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout
        #: Scoped mixed precision: run ONLY the SDPA kernel in bf16 (unlocks
        #: the flash / fused attention path, which has no fp32 kernel) while
        #: every projection, norm, and output stays fp32. Off by default;
        #: enable only where measured faster (the cast pair costs more than
        #: it saves below bf16-efficient shapes).
        self.attn_bf16: bool = False

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], key_padding_mask: [B,T] bool True=padding (masked out).
        # Mask polarity: history_mask/legal_mask use True=participate while
        # key_padding_mask uses True=padding (single ~ inversion at :214).
        residual: torch.Tensor = x
        x = self.norm1(x)
        batch: int = int(cast("Any", x.shape[0]))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic shape; int() validates
        seq_len: int = int(cast("Any", x.shape[1]))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic shape; int() validates

        # Chunk restores the three [B,T,D] views; the SAME view/transpose
        # blocks run on the chunks.
        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        queries: torch.Tensor = q.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        keys: torch.Tensor = k.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        values: torch.Tensor = v.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        # Bool mask dispatch avoids O(B·T²) float alloc: bool [B,1,1,T]
        # broadcasts to [B,H,T,T] without materializing [B,1,T,T] (8 MiB +
        # float copy per layer at B=32,T=256).
        # Evidence: SDPA bool attn_mask True=participate (inverse of
        # key_padding_mask); see SDPA tutorial + SDPA docs links below.
        # https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
        # Bucket invariance preserved: padded keys get False identically for
        # 32/64/128 buckets; guarded by
        # tests/unit/test_model_inference.py::
        # test_cache_full_history_encoding_agreement.
        if key_padding_mask.dtype != torch.bool:
            raise ContractError("key_padding_mask must be bool")
        # SDPA bool attn_mask True=attend, so invert padding -> participate mask.
        # Shape [B,1,1,T] broadcasts over H and queries.
        attn_mask: torch.Tensor = ~key_padding_mask[:, None, None, :]
        # For dense attention, query padding also could be masked but we keep it;
        # padded queries ignored in later masked mean.

        dropout_p: float = self.dropout if self.training else 0.0
        if self.attn_bf16 and queries.is_cuda and queries.dtype == torch.float32:
            attended = F.scaled_dot_product_attention(
                queries.to(torch.bfloat16),
                keys.to(torch.bfloat16),
                values.to(torch.bfloat16),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            ).to(queries.dtype)
        else:
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        # Transpose/view fuse: inductor fuses transpose+reshape under
        # max-autotune; avoid contiguous().view copy (4 MiB at B=32,T=256,D=128).
        # Evidence: https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        attended = self.out_proj(attended)
        x = residual + attended
        # FFN
        residual2: torch.Tensor = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual2 + x

    def forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        # Packed twin of forward: x is [total,D] concatenated real prefixes
        # in pull order, cu_seqlens int32 [B+1] boundaries. Same weights,
        # same full (non-causal) attention as the bucketed path — pads are
        # absent instead of masked, so per-query softmax denominators match
        # the bucketed real-key sums exactly (up to kernel numerics, never
        # promised bitwise). Failure mode: a cu entry pointing outside the
        # stream reads a neighbor row — lengths are fail-closed at build.
        residual: torch.Tensor = x
        x = self.norm1(x)
        total: int = int(cast("Any", x.shape[0]))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic shape; int() validates
        heads: int = self.n_heads
        head_dim: int = self.head_dim

        qkv: torch.Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # No transpose needed: packed has no T dim to swap.
        queries: torch.Tensor = q.view(total, heads, head_dim)
        keys: torch.Tensor = k.view(total, heads, head_dim)
        values: torch.Tensor = v.view(total, heads, head_dim)

        dropout_p: float = self.dropout if self.training else 0.0
        from torch.nn.attention.varlen import varlen_attn

        if not queries.is_cuda:
            # varlen dispatches to the flash kernel family (fp16/bf16 only,
            # CUDA only): CPU callers stay on the bucketed path (fail closed
            # here rather than silently computing a different layout).
            raise ContractError("packed attention requires CUDA tensors")
        if queries.dtype == torch.float32:
            attended = cast(
                "torch.Tensor",
                varlen_attn(
                    queries.to(torch.bfloat16),
                    keys.to(torch.bfloat16),
                    values.to(torch.bfloat16),
                    cu_seqlens,
                    cu_seqlens,
                    max_len,
                    max_len,
                ),
            ).to(queries.dtype)
        else:
            attended = cast(
                "torch.Tensor",
                varlen_attn(
                    queries,
                    keys,
                    values,
                    cu_seqlens,
                    cu_seqlens,
                    max_len,
                    max_len,
                ),
            )
        # varlen is dropout-free: preserve the attention-dropout rate on the
        # packed output while training (bucketed drops weights post-softmax;
        # kernel draws differ either way, so parity stays statistical; eval
        # is dropout-free in both paths and matches closely).
        if dropout_p > 0.0:
            attended = F.dropout(attended, p=dropout_p, training=True)
        attended = attended.reshape(total, self.d_model)
        attended = self.out_proj(attended)
        x = residual + attended
        residual2: torch.Tensor = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual2 + x


def _migrate_legacy_fused_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remap legacy unfused QKV/head keys to fused names (fail-closed).

    For each ``<prefix>.q_proj.weight`` with the full q/k/v trio present and no
    ``<prefix>.qkv_proj.weight``, stacks ``cat(dim=0)`` in q,k,v order and drops
    the legacy keys. Root ``placement/value/event/belief_head.{weight,bias}``
    concatenate (weights dim 0, biases flat, same order) when both
    ``small_heads.weight`` and ``small_heads.bias`` are absent. Partial legacy
    sets are left untouched so strict load raises (fail closed, never
    half-migrate). Identity unchanged by fusion (arch params identical); only
    state-dict keys changed.
    """
    migrated: dict[str, Any] = dict(state_dict)
    prefixes: set[str] = set()
    for _key in list(migrated.keys()):
        if _key.endswith(".q_proj.weight"):
            prefixes.add(_key[: -len(".q_proj.weight")])
    for _prefix in sorted(prefixes):
        _qk: str = f"{_prefix}.q_proj.weight"
        _kk: str = f"{_prefix}.k_proj.weight"
        _vk: str = f"{_prefix}.v_proj.weight"
        _fk: str = f"{_prefix}.qkv_proj.weight"
        if _fk in migrated:
            continue
        if _qk in migrated and _kk in migrated and _vk in migrated:
            migrated[_fk] = torch.cat([migrated[_qk], migrated[_kk], migrated[_vk]], dim=0)
            del migrated[_qk]
            del migrated[_kk]
            del migrated[_vk]
        # Partial trio left untouched: strict load then raises (fail closed).
    _w: list[str] = [
        "placement_head.weight",
        "value_head.weight",
        "event_head.weight",
        "belief_head.weight",
    ]
    _b: list[str] = [
        "placement_head.bias",
        "value_head.bias",
        "event_head.bias",
        "belief_head.bias",
    ]
    if (
        "small_heads.weight" not in migrated
        and "small_heads.bias" not in migrated
        and all(_k in migrated for _k in _w)
        and all(_k in migrated for _k in _b)
    ):
        migrated["small_heads.weight"] = torch.cat([migrated[_k] for _k in _w], dim=0)
        migrated["small_heads.bias"] = torch.cat([migrated[_k] for _k in _b], dim=0)
        for _k in _w + _b:
            del migrated[_k]
    # Partial head sets left untouched: strict load then raises (fail closed).
    return migrated


class Hydra2BaselineModel(nn.Module):
    """Baseline transformer — actor-visible only, SDPA, dense heads."""

    # Root-cause type for register_buffer: pyrefly infers Module|Tensor
    # for dynamically registered buffers; explicit annotation narrows
    # to Tensor without runtime cost. Evidence:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
    pos_ids: torch.Tensor

    def __init__(
        self,
        *,
        action_count: int = BASELINE_ACTION_COUNT,
        d_model: int = _DEFAULT_D_MODEL,
        n_layers: int = _DEFAULT_N_LAYERS,
        n_heads: int = _DEFAULT_N_HEADS,
        d_ff: int = _DEFAULT_D_FF,
        dropout: float = _DEFAULT_DROPOUT,
        architecture_id: str = "hydra2_baseline_transformer_v1",
        # hanchan = full game; utility_id names the full-game placement target.
        utility_id: str = "expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash: DigestText | None = None,
        history_buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS,
    ) -> None:
        super().__init__()
        if action_count != BASELINE_ACTION_COUNT:
            raise ContractError(f"action_count {action_count} != baseline {BASELINE_ACTION_COUNT}")
        from hydra2.models.schema import KNOWN_ARCHITECTURES

        if architecture_id not in KNOWN_ARCHITECTURES:
            raise ContractError(f"unknown architecture_id {architecture_id!r}")
        self.action_count = action_count
        self.architecture_id = architecture_id
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.history_buckets = history_buckets
        self.utility_id = utility_id
        if utility_manifest_hash is None:
            # Synthetic utility manifest identical to the Tenhou rules/utility
            # test fixture (expected-final-placement utility, zero-sum only
            # with manifest proof).
            from hydra2.contracts.utility import (
                UTILITY_OBJECTIVE,
                UTILITY_TIE_POLICY,
                make_utility_manifest,
            )

            # Golden rules-envelope digest (tenhou_4p_hanchan_v1 manifest bytes).
            golden_rules_hash = (
                "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
            )
            rules_id: str = _bridge_contracts.RULES_ID
            manifest = make_utility_manifest(
                utility_id=utility_id,
                schema_version="1.0.0",
                rules_id=rules_id,
                rules_hash=golden_rules_hash,
                objective=UTILITY_OBJECTIVE,
                rank_values=(20.0, 10.0, -10.0, -20.0),
                tie_policy=UTILITY_TIE_POLICY,
                value_min=-100.0,
                value_max=100.0,
                zero_sum=True,
            )
            utility_manifest_hash = manifest.digest
        self.utility_manifest_hash: DigestText = _bridge_contracts.make_digest_text(
            utility_manifest_hash
        )

        # Embeddings
        self.history_embedding = nn.Embedding(_NUM_EVENT_KINDS, d_model)
        max_bucket = max(history_buckets)
        self.pos_embedding = nn.Embedding(max_bucket, d_model)
        # Hoist arange pos_ids to a buffer: avoids per-forward [B,T] int64
        # alloc (8 KiB at T=256) and host→device transfer each step. Buffer
        # is device-resident and sliced without alloc; hoisted constants
        # enable fusion.
        self.register_buffer(
            "pos_ids", torch.arange(max_bucket, dtype=torch.long), persistent=False
        )  # reason: single logical buffer registration; splitting harms scan
        # Input dim: actor (one-hot 4 via embedding) + dealer embedding + phase etc.
        # For baseline determinism, we use simple linear over flattened scalars
        # computed in forward; dimension is declared as 64.
        self.scalar_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.actor_emb = nn.Embedding(4, 8)
        self.phase_emb = nn.Embedding(6, 8)
        self.furiten_emb = nn.Embedding(4, 4)
        self.wind_emb = nn.Embedding(4, 4)

        self.layers = nn.ModuleList(
            [_TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        # Heads (policy stays separate: name pinned by run_config LR group +
        # stream_train _POLICY_HEAD_PREFIX + tests; small heads fused item 2).
        # Slice map on small_heads out [B,20+2E]: [0:16) placement / [16:20)
        # value / [20:20+E) event / [20+E:20+2E) belief (E=_NUM_EVENT_KINDS).
        self.policy_head = nn.Linear(d_model * 2, action_count)
        self.small_heads = nn.Linear(d_model * 2, 20 + 2 * _NUM_EVENT_KINDS)

        # Deterministic init
        self._init_weights()

        # Model identity binds architecture + head specs + utility etc.
        self._model_identity = self._compute_model_identity()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("qkv_proj"):
                    # Stacked init (RNG-identical to legacy): legacy drew q,k,v as
                    # three separate same-shape xaviers in module order; three temp
                    # [D,D] draws in q,k,v order reproduce the RNG sequence exactly,
                    # then cat(dim=0) stacks without consuming RNG (a fresh [3D,D]
                    # draw would differ).
                    with torch.no_grad():
                        _q: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(
                                module.weight.shape[0] // 3, module.weight.shape[1]
                            )
                        )
                        _k: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(
                                module.weight.shape[0] // 3, module.weight.shape[1]
                            )
                        )
                        _v: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(
                                module.weight.shape[0] // 3, module.weight.shape[1]
                            )
                        )
                        _ = module.weight.copy_(torch.cat([_q, _k, _v], dim=0))
                elif name.endswith("small_heads"):
                    # Stacked init: 4 draws [16,2D],[4,2D],[E,2D],[E,2D] in
                    # placement/value/event/belief order; bias zeros_ (no RNG).
                    with torch.no_grad():
                        _in: int = module.weight.shape[1]
                        _ne: int = _NUM_EVENT_KINDS
                        _pl: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(16, _in)
                        )
                        _va: torch.Tensor = nn.init.xavier_uniform_(module.weight.new_empty(4, _in))
                        _ev: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(_ne, _in)
                        )
                        _be: torch.Tensor = nn.init.xavier_uniform_(
                            module.weight.new_empty(_ne, _in)
                        )
                        _ = module.weight.copy_(torch.cat([_pl, _va, _ev, _be], dim=0))
                        if module.bias is not None:
                            _ = nn.init.zeros_(module.bias)
                else:
                    _weight: torch.Tensor = nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        _bias: torch.Tensor = nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                _emb: torch.Tensor = nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_model_identity(self) -> DigestText:
        import hashlib

        from hydra2.artifacts.canonical import canonical_bytes

        doc = {
            "architecture_id": self.architecture_id,
            "architecture_parameters": {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout_p,
                "history_buckets": list(self.history_buckets),
            },
            "utility_manifest_hash": self.utility_manifest_hash,
            "action_count": self.action_count,
            "feature_derivation_hash": str(_feature_derivation_hash()),
            "input_schema_hash": str(model_input_schema_digest()),
        }
        return DigestText("sha256:" + hashlib.sha256(canonical_bytes(doc)).hexdigest())

    @property
    def model_identity(self) -> DigestText:
        return self._model_identity

    def load_state_dict(  # type: ignore[override]  # reason: remap legacy keys then delegate; signature matches nn.Module.
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        # Legacy compat: remap unfused keys to fused names (fail-closed partials).
        # model_identity UNCHANGED by fusion (arch params identical); only
        # state-dict keys changed, handled here so all callers migrate free.
        migrated: dict[str, Any] = _migrate_legacy_fused_keys(state_dict)
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    @torch._dynamo.disable
    def _packed_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Decorator form (the context-manager form is rejected inside
        # compiled frames): runs eager even when called from compiled code.
        return self.final_norm(x)

    def _trunk_packed(
        self,
        batch_size: int,
        history_kind: torch.Tensor,
        history_mask: torch.Tensor,
        packed: Any,
    ) -> torch.Tensor:
        # Packed history trunk: same math as the bucketed path with pads
        # absent instead of masked. Positions are within-row (pads trail in
        # the bucketed planes, so real prefix positions match exactly).
        # Per-row pooling is a segment sum over the packed stream divided by
        # the real length (clamped like the bucketed masked mean; empty rows
        # render the zero vector). Failure mode: a cu entry outside the
        # stream or a length disagreeing with the mask fails closed here —
        # the builder already gates both, this re-checks on device.
        cu: torch.Tensor = packed.cu_seqlens.to(torch.int32)
        row_lengths: torch.Tensor = cu[1:] - cu[:-1]
        if int(row_lengths.shape[0]) != batch_size:
            raise ContractError(f"packed bounds {int(row_lengths.shape[0])} != batch {batch_size}")
        device = packed.packed_kind.device
        cu = cu.to(device)
        # Length-vs-mask agreement re-check (eager only): the builder already
        # fail-closed both, and this .item() sync must never enter the
        # compiled graph (same precedent as validate_actor_batch).
        if not torch.compiler.is_compiling():
            expect: torch.Tensor = history_mask.sum(dim=1).to(torch.int32).to(row_lengths.device)
            if bool((row_lengths != expect).any().item()):
                raise ContractError("packed lengths disagree with the batch mask")
        stream: torch.Tensor = packed.packed_kind.to(device)
        # No int() on total: it stays symbolic under dynamo (a value guard
        # here re-specializes every distinct total — the observed recompile
        # drizzle). Eager callers see a plain int, identical semantics.
        total = stream.shape[0]
        # Row of every token + within-row positions WITHOUT repeat_interleave
        # (its data-dependent output shape graph-breaks dynamo out of the
        # compiled region into eager islands). searchsorted output follows
        # the input shape (static), so the whole trunk traces as one graph.
        cu64: torch.Tensor = cu.to(torch.int64)
        order: torch.Tensor = torch.arange(total, device=device)
        row_index: torch.Tensor = torch.searchsorted(cu64, order, right=True) - 1
        positions: torch.Tensor = order - cu64[row_index]
        hist_emb: torch.Tensor = self.history_embedding(
            stream.clamp(min=0, max=_NUM_EVENT_KINDS - 1)
        )
        hist_emb = hist_emb + self.pos_embedding(positions)

        x: torch.Tensor = hist_emb
        if total > 0:
            # varlen max is the frozen bucket cap (256), not the batch max:
            # eager fwd+bwd proven bitwise-identical to actual max, and a
            # constant removes the whole max_len == N recompile dimension
            # (observed guards firing per distinct batch max).
            for layer in self.layers:
                x = cast("Any", layer).forward_packed(x, cu, max(HISTORY_BUCKET_LENGTHS))

        # final_norm runs OUTSIDE the compiled graph (see _packed_final_norm:
        # inductor's fused layernorm-backward miscompiles under dynamic total
        # at production scale — NaN dgamma with finite inputs, 25/25 skipped
        # updates; eager ATEN is proven correct here).
        x = self._packed_final_norm(x)
        if total == 0:
            return torch.zeros((batch_size, self.d_model), dtype=x.dtype, device=device)
        sums: torch.Tensor = torch.zeros((batch_size, self.d_model), dtype=x.dtype, device=device)
        sums.index_add_(0, row_index, x.to(sums.dtype))
        lens_dev: torch.Tensor = row_lengths.to(device)
        denom: torch.Tensor = lens_dev.to(sums.dtype).clamp(min=1.0).unsqueeze(-1)
        return sums / denom

    def forward(self, batch: ActorTensorBatch) -> ModelOutput:  # type: ignore[override]  # reason: nn.Module forward signature narrow; intentional override. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
        return self.evaluate(batch)

    def evaluate(self, batch: ActorTensorBatch) -> ModelOutput:
        # Validate shapes / legal mask (eager only): compiled callers
        # pre-validate via validate_actor_batch (see _model_forward) so the
        # graph holds zero host syncs; direct eager callers keep identical errors.
        if not torch.compiler.is_compiling():
            validate_actor_batch(batch, self.action_count)

        batch_size: int = batch.history_mask.shape[0]
        seq_len: int = batch.history_mask.shape[1]
        history_kind: torch.Tensor = batch.features["history_event_kind"]  # [B,T]
        history_mask: torch.Tensor = batch.history_mask  # [B,T] True=participate
        packed: Any = getattr(batch, "packed", None)

        if packed is not None:
            # Packed branch shares the model/schema identity deliberately
            # (no new architecture id): it alters no frozen field (feature
            # name/order/shape/dtype/range/padding/mask, buckets), only how
            # attention iterates them — same weights, same features, same
            # checkpoint bytes. Precedent: precision flags change numerics
            # more yet share identity; parity-gated equal (single-batch NLL
            # within 1e-4, eval top1 within 0.01pt measured). A new identity
            # would fork interchangeable checkpoints for zero safety gain.
            pooled = self._trunk_packed(batch_size, history_kind, history_mask, packed)
            x_dtype: torch.dtype = pooled.dtype
        else:
            # History embedding with positional addition — padding positions use zero mask.
            hist_emb: torch.Tensor = self.history_embedding(
                history_kind.clamp(min=0, max=_NUM_EVENT_KINDS - 1)
            )  # reason: single logical embedding lookup; splitting harms scan
            # Slice buffer pos_ids instead of torch.arange per forward: avoids
            # [B,T] int64 alloc + H2D each step (buffer is persistent=False,
            # device-resident, sliced via view).
            positions: torch.Tensor = self.pos_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
            hist_emb = hist_emb + self.pos_embedding(positions)

            # SDPA bool mask uses True=attend, so invert participate -> padding.
            key_padding_mask: torch.Tensor = ~history_mask  # [B,T] True where padding

            x: torch.Tensor = hist_emb
            for layer in self.layers:
                x = layer(x, key_padding_mask)

            x = self.final_norm(x)

            # Masked mean pool over history — padded positions excluded. The mask
            # follows x.dtype (bf16 trunk under autocast, fp32 default identical).
            mask_f: torch.Tensor = history_mask.to(x.dtype).unsqueeze(-1)  # [B,T,1]
            # When history empty (all padding), denominator zero; use zero vector.
            denom: torch.Tensor = mask_f.sum(dim=1).clamp(min=1.0)  # [B,1]
            pooled = (x * mask_f).sum(dim=1) / denom  # [B,D]
            x_dtype = x.dtype
        # Scalar branch — build 64-dim vector from actor-visible scalars.
        scalar_vec: torch.Tensor = self._build_scalar_features(batch, dtype=x_dtype)  # [B,64]
        scalar_emb: torch.Tensor = self.scalar_proj(scalar_vec)  # [B,D]
        trunk: torch.Tensor = torch.cat([pooled, scalar_emb], dim=-1)  # [B, 2D]

        policy_logits: torch.Tensor = self.policy_head(trunk)  # [B,A]
        # Fused small heads: slice restores the four outputs exactly. Bounds
        # from _NUM_EVENT_KINDS (trace-safe Python ints; inductor folds).
        # Placement slice needs .contiguous() before .view: slice stride
        # (62,1) breaks .view (131KB @B2048, negligible).
        small_out: torch.Tensor = self.small_heads(trunk)  # [B,20+2E]
        _e: int = _NUM_EVENT_KINDS
        placement_logits: torch.Tensor = small_out[..., 0:16].contiguous().view(batch_size, 4, 4)
        value_vector: torch.Tensor = small_out[..., 16:20]  # [B,4]
        event_logits_single: torch.Tensor = small_out[..., 20 : 20 + _e]  # [B, E]
        belief_logits_single: torch.Tensor = small_out[..., 20 + _e : 20 + 2 * _e]

        # Validate output shapes before returning.
        if policy_logits.shape != (batch_size, self.action_count):
            raise ContractError("policy_logits shape mismatch")

        event_logits: dict[str, torch.Tensor] = {"next_event": event_logits_single}
        belief_logits: dict[str, torch.Tensor] = {"next_event": belief_logits_single}

        # Diagnostics: actor-visible derived tensors only — history length,
        # concealed counts, legal count. Excludes wall order, hidden hands,
        # and privileged labels.
        hist_len = history_mask.sum(dim=1).to(torch.int32)  # [B]
        concealed_sum = batch.features["concealed_hand_counts"].sum(
            dim=1
        )  # [B] should be hand size
        diag: dict[str, torch.Tensor] = {
            "history_length": hist_len,
            "concealed_tiles": concealed_sum,
            "legal_count": batch.legal_mask.sum(dim=1).to(torch.int32),
        }

        return ModelOutput(
            policy_logits=policy_logits,
            placement_logits=placement_logits,
            value_vector=value_vector,
            event_logits=event_logits,
            belief_logits=belief_logits,
            diagnostics=diag,
            utility_id=self.utility_id,
            utility_manifest_hash=self.utility_manifest_hash,
            model_identity=self.model_identity,
        )

    def _build_scalar_features(
        self, batch: ActorTensorBatch, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # Compose 64-dim scalar feature vector from actor-visible fields
        # (no wall order, hidden hands, or privileged labels). Normalizations
        # follow the trunk dtype: evaluate passes x.dtype so bf16 autocast
        # keeps the scalar branch in bf16 while the fp32 default is identical.
        compute_dtype: torch.dtype = dtype if dtype is not None else torch.float32
        feats: list[torch.Tensor] = []

        actor: torch.Tensor = batch.features["actor"]  # [B]
        dealer: torch.Tensor = batch.features["dealer"]
        turn_actor: torch.Tensor = batch.features["turn_actor"]
        phase: torch.Tensor = batch.features["phase"]
        actor_furiten: torch.Tensor = batch.features["actor_furiten"]

        actor_feat: torch.Tensor = self.actor_emb(actor).to(compute_dtype)  # [B,8]
        feats.append(actor_feat)
        dealer_feat: torch.Tensor = self.actor_emb(dealer).to(compute_dtype)
        feats.append(dealer_feat)
        turn_actor_feat: torch.Tensor = self.actor_emb(turn_actor).to(compute_dtype)
        feats.append(turn_actor_feat)
        phase_feat: torch.Tensor = self.phase_emb(phase.clamp(max=5)).to(compute_dtype)
        feats.append(phase_feat)
        furiten_feat: torch.Tensor = self.furiten_emb(actor_furiten.clamp(max=3)).to(compute_dtype)
        feats.append(furiten_feat)

        # Scores normalized / 30000, seat_winds embedding, etc.
        scores: torch.Tensor = batch.features["scores"].to(compute_dtype) / 30000.0  # [B,4]
        feats.append(scores)  # 4

        # Round wind embedding
        round_wind: torch.Tensor = batch.features["round_wind"]
        round_wind_feat: torch.Tensor = self.wind_emb(round_wind.clamp(max=3)).to(compute_dtype)
        feats.append(round_wind_feat)  # [B,4]

        # seat_winds flattened embedding sum
        seat_winds: torch.Tensor = batch.features["seat_winds"]  # [B,4]
        seat_emb: torch.Tensor = (
            self.wind_emb(seat_winds.clamp(max=3)).to(compute_dtype).view(scores.shape[0], -1)
        )  # [B,16]  # reason: single logical embedding reshape; splitting harms scan
        feats.append(seat_emb)

        # Scalar ints normalized
        honba = batch.features["honba"].to(compute_dtype).unsqueeze(-1) / 10.0
        riichi_sticks = batch.features["riichi_sticks"].to(compute_dtype).unsqueeze(-1) / 10.0
        # Live wall = undealt count (public), not wall contents/order.
        live_wall = (
            batch.features["live_wall_tiles_remaining"].to(compute_dtype).unsqueeze(-1) / 70.0
        )
        kan_count = batch.features["kan_count"].to(compute_dtype).unsqueeze(-1) / 4.0
        round_index = batch.features["round_index"].to(compute_dtype).unsqueeze(-1) / 10.0
        hand_number = batch.features["hand_number"].to(compute_dtype).unsqueeze(-1) / 10.0
        feats.extend([honba, riichi_sticks, live_wall, kan_count, round_index, hand_number])

        # Dora + own drawn tile, linearly scaled by tile id/136 (not one-hot).
        dora = batch.features["dora_indicators"].to(compute_dtype) / 136.0  # [B,5]
        feats.append(dora)
        own_drawn = batch.features["own_drawn_tile"].to(compute_dtype).unsqueeze(-1) / 136.0
        feats.append(own_drawn)

        # Concealed counts normalized: 34-dim counts reduced to mean+max per
        # side to bound the 64-dim budget.
        concealed = batch.features["concealed_hand_counts"].to(compute_dtype) / 4.0
        concealed_mean = concealed.mean(dim=1, keepdim=True)  # [B,1]
        concealed_max = concealed.max(dim=1).values.unsqueeze(-1)  # [B,1]
        feats.extend([concealed_mean, concealed_max])

        # Visible discards counts reduced to mean+max, same budget rule.
        vis_disc = batch.features["visible_discards_counts"].to(compute_dtype) / 4.0
        vis_mean = vis_disc.mean(dim=1, keepdim=True)
        vis_max = vis_disc.max(dim=1).values.unsqueeze(-1)
        feats.extend([vis_mean, vis_max])

        # ippatsu_active sum, riichi_states
        ippatsu_sum = (
            batch.features["ippatsu_active"].to(compute_dtype).sum(dim=1, keepdim=True) / 4.0
        )  # [B,1]
        feats.append(ippatsu_sum)
        riichi_sum = (
            batch.features["riichi_states"].to(compute_dtype).sum(dim=1, keepdim=True) / 8.0
        )  # [B,1]
        feats.append(riichi_sum)

        # Bool actor_can
        can_riichi = batch.features["actor_can_riichi"].to(compute_dtype).unsqueeze(-1)
        can_tsumo = batch.features["actor_can_tsumo"].to(compute_dtype).unsqueeze(-1)
        feats.extend([can_riichi, can_tsumo])

        # Segment widths sum to 64; pad/truncate guards the concat boundary.
        concat = torch.cat(feats, dim=-1).to(compute_dtype)
        if concat.shape[-1] < 64:
            pad = torch.zeros(
                (concat.shape[0], 64 - concat.shape[-1]), device=concat.device, dtype=concat.dtype
            )
            concat = torch.cat([concat, pad], dim=-1)
        elif concat.shape[-1] > 64:
            concat = concat[:, :64]
        return concat

    def model_spec(self) -> dict[str, Any]:
        """Build a ModelSpec-like document for this instance (for hashing)."""
        import pathlib

        from hydra2.contracts.action_artifact import load_action_table
        from hydra2.contracts.observation_schema import observation_schema_digest

        action_table_hash = load_action_table(
            pathlib.Path("configs/contracts/action_table_v1.json")
        ).digest
        obs_hash = observation_schema_digest()
        input_hash = model_input_schema_digest()
        deriv_hash = _feature_derivation_hash()

        doc = {
            "schema_version": "1.0.0",
            "input_schema_hash": str(input_hash),
            "feature_derivation_hash": str(deriv_hash),
            "architecture_id": self.architecture_id,
            "architecture_parameters": {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout_p,
                "history_buckets": list(self.history_buckets),
            },
            "head_specs": [
                {
                    "head_id": "belief_next",
                    "output_key": "belief_logits",
                    "target_id": "next_event_kind",
                    "loss_id": "cross_entropy",
                    "parameters": {"num_classes": _NUM_EVENT_KINDS},
                },
                {
                    "head_id": "event_next",
                    "output_key": "event_logits",
                    "target_id": "next_event_kind",
                    "loss_id": "cross_entropy",
                    "parameters": {"num_classes": _NUM_EVENT_KINDS},
                },
                {
                    "head_id": "placement",
                    "output_key": "placement_logits",
                    "target_id": "final_placement",
                    "loss_id": "cross_entropy_4x4",
                    # Per-seat semantics: [B,4,4] logits (dim-1 seat 0..3,
                    # dim-2 rank-logits 1..4) vs [B,4] rank indices; per-seat
                    # CE then mean over seats. Binds Linear(2D->16).view(B,4,4).
                    "parameters": {
                        "logits_shape": [4, 4],
                        "ranks": 4,
                        "seats": 4,
                        "semantics": "per_seat_rank_logits",
                        "target_shape": [4],
                    },
                },
                {
                    "head_id": "policy",
                    "output_key": "policy_logits",
                    "target_id": "selected_action",
                    "loss_id": "masked_cross_entropy",
                    "parameters": {"mask_field": "legal_mask"},
                },
                {
                    "head_id": "value",
                    "output_key": "value_vector",
                    "target_id": "utility_vector",
                    "loss_id": "mse",
                    "parameters": {"seats": 4},
                },
            ],
            "action_table_hash": str(action_table_hash),
            "observation_schema_hash": str(obs_hash),
            "utility_manifest_hash": str(self.utility_manifest_hash),
        }
        doc["digest"] = str(compute_model_spec_digest(doc))
        return doc
