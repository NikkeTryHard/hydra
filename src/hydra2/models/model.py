"""Hydra2 baseline actor model — SDPA, dense heads, masked policy, diagnostics.

Baseline is transformer over bucketed histories with explicit masks
(``True`` = participate). Uses ``torch.nn.functional.scaled_dot_product_attention``
for dense attention; evaluation dropout is exactly ``0.0``. Cache and
full-history encodings agree on valid prefix when masks are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  # reason: canonical PyTorch alias; upstream docs use F. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2.contracts.common import ContractError, DigestText, make_digest_text
from hydra2.contracts.event import EVENT_KINDS
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

__all__ = [
    "Hydra2BaselineModel",
    "ModelOutput",
    "masked_policy",
    "select_actions",
]


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
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            legal_mask.any(dim=-1), lambda: "masked_policy requires at least one legal per row"
        )
    elif not bool(legal_mask.any(dim=-1).all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for contract; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("masked_policy requires at least one legal per row")
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(masked, dim=-1)
    # Ensure illegal entries are exactly zero (numerical safety).
    probs = torch.where(legal_mask, probs, torch.zeros_like(probs))
    return probs


def select_actions(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Deterministic argmax after masking; tie is first legal max (spec: CandidateSpec)."""
    if logits.shape[-1] != legal_mask.shape[-1]:
        raise ContractError("shape mismatch in select_actions")
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return torch.argmax(masked, dim=-1)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Inference output (SPEC 11.3)."""

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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], key_padding_mask: [B,T] bool True=padding (masked out)
        residual: torch.Tensor = x
        x = self.norm1(x)
        batch: int = int(cast("Any", x.shape[0]))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic shape; int() validates
        seq_len: int = int(cast("Any", x.shape[1]))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic shape; int() validates

        queries: torch.Tensor = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E501  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        keys: torch.Tensor = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E501  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        values: torch.Tensor = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E501  # reason: single logical reshape; splitting harms scan. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
        # Perf-A §4.1: bool mask dispatch without O(B·T²) float alloc.
        # Evidence: SDPA tutorial
        #  https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html  # noqa: E501  # reason: URL cannot wrap without breaking link; alternative loses precision
        #  — math backend 3.7x slower vs mem_efficient, Flash rejects any attn_mask per  # noqa: E501  # reason: comment documents perf invariant; wrapping splits sentence
        #  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L check_for_attn_mask.  # noqa: E501  # reason: URL cannot wrap without breaking link; alternative loses precision
        # Docs: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html  # noqa: E501  # reason: URL cannot wrap without breaking link; alternative loses precision
        #  — bool attn_mask True=participate (inverse of key_padding_mask). Bool [B,1,1,T]
        #  broadcasts to [B,H,T,T] without materializing [B,1,T,T] (8 MiB + float copy
        #  per layer at B=32,T=256).
        # Bucket invariance preserved: padded keys get False identically for 32/64/128
        #  buckets; guarded by tests/unit/test_model_inference_wp05a.py::
        #  test_cache_full_history_encoding_agreement.
        if key_padding_mask.dtype != torch.bool:
            raise ContractError("key_padding_mask must be bool")
        # SDPA bool attn_mask True=attend, so invert padding -> participate mask.
        # Shape [B,1,1,T] broadcasts over H and queries.
        attn_mask: torch.Tensor = ~key_padding_mask[:, None, None, :]
        # For dense attention, query padding also could be masked but we keep it;
        # padded queries ignored in later masked mean.

        dropout_p: float = self.dropout if self.training else 0.0
        attended: torch.Tensor = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        # Perf-A §4.1 transpose/view fuse: inductor fuses transpose+reshape under max-autotune;
        #  avoid contiguous().view copy (4 MiB at B=32,T=256,D=128) in eager. Evidence: https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        attended = self.out_proj(attended)
        x = residual + attended
        # FFN
        residual2: torch.Tensor = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual2 + x


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
        utility_id: str = "expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash: DigestText | None = None,
        history_buckets: tuple[int, ...] = HISTORY_BUCKET_LENGTHS,
    ) -> None:
        super().__init__()
        if action_count != BASELINE_ACTION_COUNT:
            raise ContractError(f"action_count {action_count} != baseline {BASELINE_ACTION_COUNT}")
        self.action_count = action_count
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.history_buckets = history_buckets
        self.utility_id = utility_id
        if utility_manifest_hash is None:
            # Synthetic utility manifest identical to WP-02B test fixture (zero-sum).
            from hydra2.contracts.rules import RULES_ID
            from hydra2.contracts.utility import (
                UTILITY_OBJECTIVE,
                UTILITY_TIE_POLICY,
                make_utility_manifest,
            )

            # Golden envelope digest from WP-02B.
            golden_rules_hash = (
                "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
            )
            manifest = make_utility_manifest(
                utility_id=utility_id,
                schema_version="1.0.0",
                rules_id=RULES_ID,
                rules_hash=golden_rules_hash,
                objective=UTILITY_OBJECTIVE,
                rank_values=(20.0, 10.0, -10.0, -20.0),
                tie_policy=UTILITY_TIE_POLICY,
                value_min=-100.0,
                value_max=100.0,
                zero_sum=True,
            )
            utility_manifest_hash = manifest.digest
        self.utility_manifest_hash = make_digest_text(utility_manifest_hash)

        # Embeddings
        self.history_embedding = nn.Embedding(_NUM_EVENT_KINDS, d_model)
        max_bucket = max(history_buckets)
        self.pos_embedding = nn.Embedding(max_bucket, d_model)
        # Perf-A §4.1: hoist arange pos_ids to buffer to avoid per-forward [B,T]
        # int64 alloc (8 KiB at T=256) and host→device transfer each step.
        # Evidence: inductor docs recommend hoisting constants out of forward;
        # buffer is device-resident and sliced without alloc.
        # https://docs.pytorch.org/docs/2.13/generated/torch.compile.html — constants
        # hoisted enable fusion.
        self.register_buffer("pos_ids", torch.arange(max_bucket, dtype=torch.long), persistent=False)  # noqa: E501  # reason: single logical buffer registration; splitting harms scan
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

        # Heads
        self.policy_head = nn.Linear(d_model * 2, action_count)
        self.placement_head = nn.Linear(d_model * 2, 16)  # 4x4
        self.value_head = nn.Linear(d_model * 2, 4)
        self.event_head = nn.Linear(d_model * 2, _NUM_EVENT_KINDS)
        self.belief_head = nn.Linear(d_model * 2, _NUM_EVENT_KINDS)

        # Deterministic init
        self._init_weights()

        # Model identity binds architecture + head specs + utility etc.
        self._model_identity = self._compute_model_identity()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _weight: torch.Tensor = nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    _bias: torch.Tensor = nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                _emb: torch.Tensor = nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def _compute_model_identity(self) -> DigestText:
        import hashlib

        from hydra2.artifacts.canonical import canonical_bytes

        doc = {
            "architecture_id": "hydra2_baseline_transformer_v1",
            "architecture_parameters": {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": _DEFAULT_D_FF,
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

    def forward(self, batch: ActorTensorBatch) -> ModelOutput:  # type: ignore[override]  # reason: nn.Module forward signature narrow; intentional override. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
        return self.evaluate(batch)

    def evaluate(self, batch: ActorTensorBatch) -> ModelOutput:
        # Validate shapes / legal mask.
        if batch.history_mask.shape[0] != batch.legal_mask.shape[0]:
            raise ContractError("batch size mismatch between history_mask and legal_mask")
        if batch.legal_mask.shape[1] != self.action_count:
            raise ContractError(
                f"legal_mask A {batch.legal_mask.shape[1]} != "
                f"model action_count {self.action_count}"
            )
        if torch.compiler.is_compiling():
            torch._check_tensor_all(
                batch.legal_mask.any(dim=1), lambda: "nonterminal batch requires at least one legal per row"  # noqa: E501  # reason: contract string cannot split without harming grep; alternative worse
            )
        elif not bool(batch.legal_mask.any(dim=1).all().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call]  # reason: eager host sync for contract; single scalar must cross host. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            raise ContractError("nonterminal batch requires at least one legal per row")
        if batch.history_mask.shape != batch.features["history_event_kind"].shape:
            raise ContractError("history_mask vs history_event_kind shape mismatch")

        batch_size: int = batch.history_mask.shape[0]
        seq_len: int = batch.history_mask.shape[1]
        history_kind: torch.Tensor = batch.features["history_event_kind"]  # [B,T]
        history_mask: torch.Tensor = batch.history_mask  # [B,T] True=participate

        # History embedding with positional addition — padding positions use zero mask.
        hist_emb: torch.Tensor = self.history_embedding(history_kind.clamp(min=0, max=_NUM_EVENT_KINDS - 1))  # noqa: E501  # reason: single logical embedding lookup; splitting harms scan
        # Perf-A §4.1: use buffer pos_ids sliced instead of torch.arange per forward.
        # Avoids [B,T] int64 alloc + H2D each step; buffer is persistent=False
        # device-resident, sliced via view.
        positions: torch.Tensor = self.pos_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        hist_emb = hist_emb + self.pos_embedding(positions)

        # SDPA expects padding mask True=masked out. So invert.
        key_padding_mask: torch.Tensor = ~history_mask  # [B,T] True where padding

        x: torch.Tensor = hist_emb
        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.final_norm(x)

        # Masked mean pool over history — padded positions excluded.
        mask_f: torch.Tensor = history_mask.float().unsqueeze(-1)  # [B,T,1]
        # When history empty (all padding), denominator zero; use zero vector.
        denom: torch.Tensor = mask_f.sum(dim=1).clamp(min=1.0)  # [B,1]
        pooled: torch.Tensor = (x * mask_f).sum(dim=1) / denom  # [B,D]

        # Scalar branch — build 64-dim vector from actor-visible scalars.
        scalar_vec: torch.Tensor = self._build_scalar_features(batch)  # [B,64]
        scalar_emb: torch.Tensor = self.scalar_proj(scalar_vec)  # [B,D]
        trunk: torch.Tensor = torch.cat([pooled, scalar_emb], dim=-1)  # [B, 2D]

        policy_logits: torch.Tensor = self.policy_head(trunk)  # [B,A]
        placement_logits: torch.Tensor = self.placement_head(trunk).view(batch_size, 4, 4)
        value_vector: torch.Tensor = self.value_head(trunk)  # [B,4]
        event_logits_single: torch.Tensor = self.event_head(trunk)  # [B, E]
        belief_logits_single: torch.Tensor = self.belief_head(trunk)

        # Validate output shapes before returning.
        if policy_logits.shape != (batch_size, self.action_count):
            raise ContractError("policy_logits shape mismatch")

        event_logits: dict[str, torch.Tensor] = {"next_event": event_logits_single}
        belief_logits: dict[str, torch.Tensor] = {"next_event": belief_logits_single}

        # Diagnostics: actor-visible derived tensors only (no hidden info).
        # Provide deterministic, visible quantities: history length, concealed counts sum, etc.
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

    def _build_scalar_features(self, batch: ActorTensorBatch) -> torch.Tensor:
        # Compose 64-dim scalar feature vector from actor-visible fields.
        # All inputs are actor-visible; no hidden state.
        feats: list[torch.Tensor] = []

        actor: torch.Tensor = batch.features["actor"]  # [B]
        dealer: torch.Tensor = batch.features["dealer"]
        turn_actor: torch.Tensor = batch.features["turn_actor"]
        phase: torch.Tensor = batch.features["phase"]
        actor_furiten: torch.Tensor = batch.features["actor_furiten"]

        feats.append(self.actor_emb(actor))  # [B,8]
        feats.append(self.actor_emb(dealer))
        feats.append(self.actor_emb(turn_actor))
        feats.append(self.phase_emb(phase.clamp(max=5)))
        feats.append(self.furiten_emb(actor_furiten.clamp(max=3)))

        # Scores normalized / 30000, seat_winds embedding, etc.
        scores: torch.Tensor = batch.features["scores"].float() / 30000.0  # [B,4]
        feats.append(scores)  # 4

        # Round wind embedding
        round_wind: torch.Tensor = batch.features["round_wind"]
        feats.append(self.wind_emb(round_wind.clamp(max=3)))  # [B,4]

        # seat_winds flattened embedding sum
        seat_winds: torch.Tensor = batch.features["seat_winds"]  # [B,4]
        seat_emb: torch.Tensor = self.wind_emb(seat_winds.clamp(max=3)).view(scores.shape[0], -1)  # [B,16]  # noqa: E501  # reason: single logical embedding reshape; splitting harms scan
        feats.append(seat_emb)

        # Scalar ints normalized
        honba = batch.features["honba"].float().unsqueeze(-1) / 10.0  # [B,1]
        riichi_sticks = batch.features["riichi_sticks"].float().unsqueeze(-1) / 10.0
        live_wall = batch.features["live_wall_tiles_remaining"].float().unsqueeze(-1) / 70.0
        kan_count = batch.features["kan_count"].float().unsqueeze(-1) / 4.0
        round_index = batch.features["round_index"].float().unsqueeze(-1) / 10.0
        hand_number = batch.features["hand_number"].float().unsqueeze(-1) / 10.0
        feats.extend([honba, riichi_sticks, live_wall, kan_count, round_index, hand_number])  # +6

        # Dora + own drawn tile one-hot-ish normalized
        dora = batch.features["dora_indicators"].float() / 136.0  # [B,5]
        feats.append(dora)  # 5
        own_drawn = batch.features["own_drawn_tile"].float().unsqueeze(-1) / 136.0  # [B,1]
        feats.append(own_drawn)  # 1

        # Concealed counts normalized
        concealed = (
            batch.features["concealed_hand_counts"].float() / 4.0
        )  # [B,34] -> compress to sum?
        # Reduce to 4 stats: mean, max, etc. to keep dim 64 bounded
        concealed_mean = concealed.mean(dim=1, keepdim=True)  # [B,1]
        concealed_max = concealed.max(dim=1).values.unsqueeze(-1)  # [B,1]
        feats.extend([concealed_mean, concealed_max])  # +2

        # Visible discards counts similarly
        vis_disc = batch.features["visible_discards_counts"].float() / 4.0
        vis_mean = vis_disc.mean(dim=1, keepdim=True)
        vis_max = vis_disc.max(dim=1).values.unsqueeze(-1)
        feats.extend([vis_mean, vis_max])

        # ippatsu_active sum, riichi_states
        ippatsu_sum = (
            batch.features["ippatsu_active"].float().sum(dim=1, keepdim=True) / 4.0
        )  # [B,1]
        feats.append(ippatsu_sum)
        riichi_sum = batch.features["riichi_states"].float().sum(dim=1, keepdim=True) / 8.0  # [B,1]
        feats.append(riichi_sum)

        # Bool actor_can
        can_riichi = batch.features["actor_can_riichi"].float().unsqueeze(-1)
        can_tsumo = batch.features["actor_can_tsumo"].float().unsqueeze(-1)
        feats.extend([can_riichi, can_tsumo])

        concat = torch.cat(feats, dim=-1)  # should be 64
        # Pad or truncate to exactly 64
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

        from hydra2.contracts.action import load_action_table
        from hydra2.contracts.observation import observation_schema_digest

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
            "architecture_id": "hydra2_baseline_transformer_v1",
            "architecture_parameters": {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": _DEFAULT_D_FF,
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
                    "parameters": {"seats": 4, "ranks": 4},
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
