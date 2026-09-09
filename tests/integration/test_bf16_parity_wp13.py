"""Wave-1 bf16 mixed-precision parity harness (WP-13, forward-only + short overlap).

Compares a fp32 eager reference arm against a bf16 autocast device-under-test
arm over one fixed N=256 corpus (seed 0, microbatch-4 x64, persisted under
tmp_path). Both arms load an identical fp32 checkpoint payload (fp32 master
weights); the DUT wraps ``evaluate()`` in ``torch.autocast(cuda, bfloat16)``
only. Dropout is 0, ``-inf`` fills / LayerNorm / CE are untouched, and there
is deliberately NO GradScaler anywhere in this file.

Corpus strata: history-bucket lenses 32/64/128/256, an all-padding row, a
single-legal row, a full-legal row, a near-tie row, one row per visible event
class (``call_resolved`` is server_private so it can never enter an actor
history), plus random fill.

Gates:
  A1 mask identity EXACT (illegal prob == 0.0, single-legal == 1.0,
     all-false ContractError identical incl. ``evaluate()``).
  A2 order (argmax agree >= 99%, top-1 >= 98%, top-5 >= 99.5%,
     flips only when the fp32 margin |dlogit| < 0.02).
  A3 grad cosine per-tensor > 0.99, global > 0.995, no NaN/Inf,
     global norm-ratio in [0.95, 1.05].
  A4 allclose policy/placement (atol=rtol=1e-2), value (atol 0.15, rtol 0.02),
     mean-NLL diff < 0.02, gap-sign agreement (ECE reported only).
  S1-S5 short overlap: shared fp32 checkpoint, bf16 trains (Fabric
     S1 NLL track < 0.05 (< 0.03 final-10); S2 no NaN/4x-spike;
     S3 grad-norm-ratio in [0.9, 1.1]; S4 held-out argmax >= 97% /
     top-1 >= 95%; S5 smooth-divergence info (reported only).

CI cap: T=50 overlap steps. Release runs may set HYDRA2_BF16_PARITY_T=100
(clamped to [1, 100]) for the full T=100 overlap.

CUDA-only numerics (marked gpu). tmp_path only, fixed seeds, never bitwise:
every cross-arm comparison uses allclose or an explicit tolerance.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.contracts.event import EVENT_KINDS, EventEnvelope, EventPayload
from hydra2.contracts.observation import make_actor_observation
from hydra2.models.encoder import ActorTensorBatch, encode_observations
from hydra2.models.model import Hydra2BaselineModel, masked_policy, select_actions
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.runtime.fabric import FabricRuntimeAdapter
from hydra2.runtime.plain import PlainPytorchAdapter
from hydra2.runtime.protocol import RuntimeSpec, build_runtime, runtime_identity
from hydra2.training.objectives import masked_cross_entropy
from tests.conftest import unwrap_model

pytestmark = pytest.mark.gpu

SEED = 0
N_ROWS = 256
MICROBATCH = 4
N_MICROBATCHES = N_ROWS // MICROBATCH
TRAIN_MICROBATCHES = 56
HELD_OUT_MICROBATCHES = N_MICROBATCHES - TRAIN_MICROBATCHES
ACTION_COUNT = BASELINE_ACTION_COUNT

A2_ARGMAX_AGREE = 0.99
A2_TOP1_AGREE = 0.98
A2_TOP5_AGREE = 0.995
A2_FLIP_DLOGIT = 0.02
A3_COS_TENSOR = 0.99
A3_COS_GLOBAL = 0.995
A3_NORM_LO, A3_NORM_HI = 0.95, 1.05
A4_ATOL = 1e-2
A4_RTOL = 1e-2
A4_VALUE_ATOL = 0.15
A4_VALUE_RTOL = 0.02
A4_NLL_DIFF = 0.02
A4_GAP_AGREE = 0.99
S1_TRACK = 0.05
S1_FINAL10 = 0.03
S3_NORM_LO, S3_NORM_HI = 0.9, 1.1
S4_ARGMAX_AGREE = 0.97
S4_TOP1_AGREE = 0.95

# Visible event classes: every EVENT_KINDS entry except server_private
# call_resolved, which can never enter an actor-visible history.
VISIBLE_KINDS = tuple(k for k in EVENT_KINDS if k != "call_resolved")

_DIGEST_AB = "sha256:" + "ab" * 32
_DIGEST_AC = "sha256:" + "ac" * 32
_DIGEST_AD = "sha256:" + "ad" * 32
_DIGEST_AE = "sha256:" + "ae" * 32
_DIGEST_AF = "sha256:" + "af" * 32
_SCORES = (25000, 25000, 25000, 25000)
_CONCEALED = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)


def _overlap_steps() -> int:
    """CI cap T=50; release runs may export HYDRA2_BF16_PARITY_T=100."""
    try:
        requested = int(os.environ.get("HYDRA2_BF16_PARITY_T", "50"))
    except ValueError:
        requested = 50
    return max(1, min(100, requested))


def _arch_record() -> str:
    """Runtime capability detection — never a hardcoded SM path (A100/5070 agnostic)."""
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return f"{name} sm_{major}{minor}"


def _envelope(kind: str, *, seq: int, seat: int, game: str) -> EventEnvelope:
    """Minimal valid envelope of ``kind`` (SPEC 7.1 per-kind shape matrix)."""
    tile = (seq * 7 + 3) % 136
    action = seq % 100
    source = (seat + 1) % 4
    payload_kwargs: dict = {
        "kind": kind,
        "actor": None,
        "tile": None,
        "action_id": None,
        "source_seat": None,
        "consumed_tiles": (),
        "offered_action_ids": (),
        "accepted_action_ids": (),
        "round_index": None,
        "scores": None,
        "reason": None,
    }
    envelope_actor: int | None = None
    visibility = "public"
    visible_to: tuple = (0, 1, 2, 3)
    if kind == "game_start":
        payload_kwargs.update(round_index=0, scores=_SCORES)
    elif kind == "round_start":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(round_index=0, scores=_SCORES)
    elif kind == "turn_advance":
        envelope_actor, payload_kwargs["actor"] = seat, seat
    elif kind == "draw_tile":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs["tile"] = tile
        visibility, visible_to = "actor_private", (seat,)
    elif kind in ("discard", "riichi_declared"):
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(tile=tile, action_id=action)
    elif kind == "riichi_accepted":
        envelope_actor, payload_kwargs["actor"] = seat, seat
    elif kind == "call_window":
        pass
    elif kind in ("chi", "pon"):
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(
            tile=tile,
            action_id=action,
            source_seat=source,
            consumed_tiles=((tile + 1) % 136, (tile + 2) % 136),
        )
    elif kind == "daiminkan":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(
            tile=tile,
            action_id=action,
            source_seat=source,
            consumed_tiles=((tile + 1) % 136, (tile + 2) % 136, (tile + 3) % 136),
        )
    elif kind == "ankan":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(
            action_id=action,
            consumed_tiles=tuple((tile + i) % 136 for i in range(1, 5)),
        )
    elif kind == "kakan":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(tile=tile, action_id=action)
    elif kind == "dora_revealed":
        payload_kwargs["tile"] = tile
    elif kind == "ron":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(tile=tile, action_id=action, source_seat=source)
    elif kind == "tsumo":
        envelope_actor, payload_kwargs["actor"] = seat, seat
        payload_kwargs.update(tile=tile, action_id=action)
    elif kind == "draw_end":
        payload_kwargs.update(scores=_SCORES, reason="wall_exhausted")
    elif kind == "abortive_draw":
        payload_kwargs.update(round_index=0, scores=_SCORES, reason="four_riichi")
    elif kind == "round_end":
        payload_kwargs.update(round_index=0, scores=_SCORES)
    elif kind == "game_end":
        payload_kwargs.update(round_index=0, scores=_SCORES, reason="hanchan_complete")
    else:  # pragma: no cover - VISIBLE_KINDS is exhaustively handled above
        raise AssertionError(f"unhandled visible kind {kind!r}")
    return EventEnvelope(
        game_id=game,
        sequence=seq,
        kind=kind,  # type: ignore[arg-type]
        actor=envelope_actor,
        visibility=visibility,  # type: ignore[arg-type]
        visible_to=visible_to,
        payload=EventPayload(**payload_kwargs),
        public_delta=(),
        rules_hash=_DIGEST_AB,
        schema_hash=_DIGEST_AC,
    )


def _observation(*, idx: int, actor: int, kinds: list[str], legal: list[int]):
    history = tuple(
        _envelope(kind, seq=seq, seat=actor, game="g-bf16-wp13")
        for seq, kind in enumerate(kinds, start=1)
    )
    sequence = (history[-1].sequence + 1) if history else 1
    return make_actor_observation(
        game_id="g-bf16-wp13",
        decision_id=f"bf16-{idx:04d}",
        sequence=int(sequence),
        actor=actor,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash=_DIGEST_AB,
        action_table_hash=_DIGEST_AC,
        event_schema_hash=_DIGEST_AD,
        observation_schema_hash=_DIGEST_AE,
        packet_boundary_hash=_DIGEST_AF,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=_SCORES,
        turn_actor=actor,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=_CONCEALED,
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=history,
        legal_mask=tuple(i in set(legal) for i in range(ACTION_COUNT)),
    )


def _build_specs() -> list[dict]:
    """Fixed N=256 corpus specs (seed 0): strata + random fill."""
    rng = random.Random(SEED)
    specs: list[dict] = []

    def add(actor: int, kinds: list[str], legal: list[int] | None) -> None:
        idx = len(specs)
        if legal is None:  # full-legal row
            legal = list(range(ACTION_COUNT))
            target = rng.randrange(ACTION_COUNT)
        else:
            target = rng.choice(legal)
        specs.append({"idx": idx, "actor": actor, "kinds": kinds, "legal": legal, "target": target})

    def sample_legal(k: int) -> list[int]:
        return sorted(rng.sample(range(ACTION_COUNT), k))

    for i in range(40):  # bucket-32 stratum (lengths 0..32, incl. all-padding)
        add(i % 4, ["turn_advance"] * (i % 33), sample_legal(2 + (i % 5)))
    for i in range(40, 80):  # bucket-64 stratum (33..64)
        add(i % 4, ["turn_advance"] * (33 + (i % 32)), sample_legal(2 + (i % 5)))
    for i in range(80, 120):  # bucket-128 stratum (65..128)
        add(i % 4, ["turn_advance"] * (65 + (i % 64)), sample_legal(2 + (i % 5)))
    for i in range(120, 160):  # bucket-256 stratum (129..256)
        length = 129 + ((i * 3) % 128)
        if i == 159:
            length = 256
        add(i % 4, ["turn_advance"] * length, sample_legal(2 + (i % 5)))
    add(0, [], sample_legal(3))  # explicit all-padding row
    single = rng.randrange(ACTION_COUNT)
    add(1, ["turn_advance"] * 10, [single])  # single-legal row
    add(2, ["turn_advance"] * 10, None)  # full-legal row
    add(3, ["turn_advance"] * 10, [7, 8])  # near-tie probe row
    for j, kind in enumerate(VISIBLE_KINDS):  # per-event-class rows (20)
        actor = (164 + j) % 4
        add(actor, [kind] * 4 + ["turn_advance"] * 2, sample_legal(3))
    while len(specs) < N_ROWS:  # random fill across buckets
        length = rng.choice([3, 17, 31, 40, 60, 90, 120, 150, 200, 250])
        add(len(specs) % 4, ["turn_advance"] * length, sample_legal(rng.randint(2, 6)))
    assert len(specs) == N_ROWS
    return specs


def _to_cuda(batch: ActorTensorBatch) -> ActorTensorBatch:
    return ActorTensorBatch(
        features={name: tensor.to("cuda") for name, tensor in batch.features.items()},
        history_mask=batch.history_mask.to("cuda"),
        legal_mask=batch.legal_mask.to("cuda"),
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats.to("cuda"),
    )


def _fp32_spec(adapter_id: str) -> RuntimeSpec:
    return RuntimeSpec(
        adapter_id=adapter_id,  # type: ignore[arg-type]
        device="cuda",
        precision="fp32",
        compile_mode="eager",
        fullgraph=False,
        dynamic=None,
        backward_pass_autocast=None,
    )


def _fresh_model(payload: dict, *, train: bool) -> Hydra2BaselineModel:
    model = Hydra2BaselineModel(dropout=0.0)
    model.load_state_dict(payload, strict=True)
    model.to("cuda")
    model.train(train)
    return model


def _global_grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().item()) ** 2
    return total**0.5


def _cosine_similarity(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    denom = float(flat_a.norm().item()) * float(flat_b.norm().item())
    if denom == 0.0:
        return 1.0 if torch.equal(flat_a, flat_b) else 0.0
    return float((flat_a.double() @ flat_b.double()).item() / denom)


def _expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, *, bins: int = 10
) -> float:
    confidence, predicted = probs.max(dim=-1)
    correct = (predicted == targets).float()
    ece, total = 0.0, confidence.numel()
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = (confidence > lo) & (confidence <= hi) if b else confidence <= hi
        count = int(in_bin.sum().item())
        if count:
            ece += abs(
                float(correct[in_bin].mean().item()) - float(confidence[in_bin].mean().item())
            ) * (count / total)
    return ece


@pytest.fixture(scope="module")
def corpus_artifact(tmp_path_factory):
    """Build, persist (tmp_path only), reload-verify, and encode the corpus."""
    corpus_dir = tmp_path_factory.mktemp("bf16_wp13")
    specs = _build_specs()
    persisted = {
        "seed": SEED,
        "rows": [
            {"actor": s["actor"], "kinds": s["kinds"], "legal": s["legal"], "target": s["target"]}
            for s in specs
        ],
    }
    path = corpus_dir / "bf16_wp13_corpus.pt"
    torch.save(persisted, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert reloaded == persisted, "corpus persistence round-trip must be exact"
    observations = [
        _observation(idx=s["idx"], actor=s["actor"], kinds=s["kinds"], legal=s["legal"])
        for s in specs
    ]
    microbatches = []
    for start in range(0, N_ROWS, MICROBATCH):
        batch = encode_observations(observations[start : start + MICROBATCH])
        targets = torch.tensor(
            [specs[start + k]["target"] for k in range(MICROBATCH)], dtype=torch.long
        )
        microbatches.append((batch, targets))
    assert len(microbatches) == N_MICROBATCHES
    return {"path": path, "specs": specs, "microbatches": microbatches}


@pytest.fixture(scope="module")
def fp32_payload():
    """Identical fp32 checkpoint payload shared by both arms (seed 0 init)."""
    torch.manual_seed(SEED)
    model = Hydra2BaselineModel(dropout=0.0)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@pytest.fixture(scope="module")
def cuda_batches(corpus_artifact, require_cuda):
    assert require_cuda is not None
    return [
        (_to_cuda(batch), targets.to("cuda")) for batch, targets in corpus_artifact["microbatches"]
    ]


@pytest.fixture(scope="module")
def forward_arms(fp32_payload, require_cuda):
    assert require_cuda is not None
    torch.manual_seed(SEED)
    ref = _fresh_model(fp32_payload, train=False)
    dut = _fresh_model(fp32_payload, train=False)
    return ref, dut


def _forward_collect(cuda_batches, ref, dut):
    """fp32 eager ref vs autocast-bf16 DUT (autocast around evaluate() only)."""
    ref_policy, dut_policy, ref_place, dut_place, ref_value, dut_value = [], [], [], [], [], []
    with torch.no_grad():
        for batch, _ in cuda_batches:
            out_ref = ref.evaluate(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_dut = dut.evaluate(batch)
            ref_policy.append(out_ref.policy_logits.float().cpu())
            dut_policy.append(out_dut.policy_logits.float().cpu())
            ref_place.append(out_ref.placement_logits.float().cpu())
            dut_place.append(out_dut.placement_logits.float().cpu())
            ref_value.append(out_ref.value_vector.float().cpu())
            dut_value.append(out_dut.value_vector.float().cpu())
    collected = {
        name: torch.cat(parts, dim=0)
        for name, parts in {
            "ref_policy": ref_policy,
            "dut_policy": dut_policy,
            "ref_place": ref_place,
            "dut_place": dut_place,
            "ref_value": ref_value,
            "dut_value": dut_value,
        }.items()
    }
    legal = torch.cat([batch.legal_mask.cpu() for batch, _ in cuda_batches], dim=0)
    targets = torch.cat([targets.cpu() for _, targets in cuda_batches], dim=0)
    collected["legal"] = legal
    collected["targets"] = targets
    return collected


class TestBf16Corpus:
    def test_strata_buckets_and_persistence(self, corpus_artifact):
        specs = corpus_artifact["specs"]
        assert len(specs) == N_ROWS
        assert corpus_artifact["path"].is_file()
        bucket_of = {32: 0, 64: 0, 128: 0, 256: 0}
        for batch, _ in corpus_artifact["microbatches"]:
            width = batch.history_mask.shape[1]
            assert width in bucket_of
            bucket_of[width] += 1
        assert all(count > 0 for count in bucket_of.values()), f"buckets uncovered: {bucket_of}"
        lengths = [len(s["kinds"]) for s in specs]
        assert 0 in lengths, "all-padding stratum missing"
        assert 256 in lengths, "max-bucket stratum missing"
        legal_sizes = sorted(len(s["legal"]) for s in specs)
        assert legal_sizes[0] == 1, "single-legal stratum missing"
        assert legal_sizes[-1] == ACTION_COUNT, "full-legal stratum missing"
        assert [7, 8] in [s["legal"] for s in specs], "near-tie probe row missing"
        covered = {kind for s in specs for kind in s["kinds"]}
        assert set(VISIBLE_KINDS) <= covered, (
            f"event classes missing: {set(VISIBLE_KINDS) - covered}"
        )
        assert len(corpus_artifact["microbatches"]) == N_MICROBATCHES


class TestBf16ForwardParity:
    def test_a1_mask_identity_exact(self, cuda_batches, forward_arms):
        ref, dut = forward_arms
        for param in ref.parameters():
            assert param.dtype == torch.float32
        for param in dut.parameters():
            assert param.dtype == torch.float32, "DUT master weights must stay fp32"
        with torch.no_grad():
            for batch, _ in cuda_batches:
                ref_probs = masked_policy(ref.evaluate(batch).policy_logits, batch.legal_mask)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    dut_probs = masked_policy(dut.evaluate(batch).policy_logits, batch.legal_mask)
                assert bool((ref_probs[~batch.legal_mask] == 0.0).all().item())
                assert bool((dut_probs[~batch.legal_mask] == 0.0).all().item())
        single = next(b for b, _ in cuda_batches if int((b.legal_mask.sum(dim=1) == 1).sum()) > 0)
        with torch.no_grad():
            ref_single = masked_policy(ref.evaluate(single).policy_logits, single.legal_mask)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dut_single = masked_policy(dut.evaluate(single).policy_logits, single.legal_mask)
        single_rows = (single.legal_mask.sum(dim=1) == 1).nonzero()[:, 0]
        for row in single_rows.tolist():
            legal_idx = int(single.legal_mask[row].nonzero()[0, 0])
            assert float(ref_single[row, legal_idx]) == 1.0
            assert float(dut_single[row, legal_idx]) == 1.0
        flat_logits = torch.zeros(2, ACTION_COUNT, device="cuda")
        all_false = torch.zeros(2, ACTION_COUNT, device="cuda", dtype=torch.bool)
        with pytest.raises(ContractError) as ref_exc:
            masked_policy(flat_logits, all_false)
        with pytest.raises(ContractError) as dut_exc:
            masked_policy(flat_logits, all_false)
        assert str(ref_exc.value) == str(dut_exc.value), "all-false ContractError must be identical"
        poisoned = dataclasses.replace(
            cuda_batches[0][0], legal_mask=torch.zeros_like(cuda_batches[0][0].legal_mask)
        )
        with pytest.raises(ContractError) as ref_eval_exc:
            ref.evaluate(poisoned)
        with (
            pytest.raises(ContractError) as dut_eval_exc,
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            dut.evaluate(poisoned)
        assert str(ref_eval_exc.value) == str(dut_eval_exc.value)

    def test_a2_order_agreement(self, cuda_batches, forward_arms):
        collected = _forward_collect(cuda_batches, *forward_arms)
        ref_logits, dut_logits = collected["ref_policy"], collected["dut_policy"]
        legal, n = collected["legal"], collected["legal"].shape[0]
        argmax_ref = select_actions(ref_logits, legal)
        argmax_dut = select_actions(dut_logits, legal)
        argmax_agree = float((argmax_ref == argmax_dut).float().mean().item())
        assert argmax_agree >= A2_ARGMAX_AGREE, (
            f"argmax agree {argmax_agree:.4f} < {A2_ARGMAX_AGREE}"
        )
        top1_dut = torch.topk(dut_logits.masked_fill(~legal, float("-inf")), k=1, dim=1).indices[
            :, 0
        ]
        top1_agree = float((argmax_ref == top1_dut).float().mean().item())
        assert top1_agree >= A2_TOP1_AGREE, f"top-1 agree {top1_agree:.4f} < {A2_TOP1_AGREE}"
        top5_dut = torch.topk(dut_logits.masked_fill(~legal, float("-inf")), k=5, dim=1).indices
        in_top5 = (top5_dut == argmax_ref.unsqueeze(1)).any(dim=1).float().mean().item()
        assert float(in_top5) >= A2_TOP5_AGREE, f"top-5 agree {in_top5:.4f} < {A2_TOP5_AGREE}"
        flipped = (argmax_ref != argmax_dut).nonzero()[:, 0]
        if len(flipped):
            rows = flipped.tolist()
            winner_ref = ref_logits[flipped, argmax_ref[flipped]].tolist()
            runner_ref = ref_logits[flipped, argmax_dut[flipped]].tolist()
            margins = [w - r for w, r in zip(winner_ref, runner_ref, strict=True)]
            assert all(m >= 0.0 for m in margins), "ref winner must be the fp32 argmax"
            assert all(m < A2_FLIP_DLOGIT for m in margins), (
                f"flips beyond |dlogit| {A2_FLIP_DLOGIT}: max {max(margins)} over {len(rows)}/{n}"
            )
        print(
            f"\nA2: argmax {argmax_agree:.4f} top1 {top1_agree:.4f} top5 {float(in_top5):.4f} "
            f"flips {len(flipped)}/{n}"
        )

    def test_a4_value_parity_nll_gap_and_ece(self, cuda_batches, forward_arms):
        collected = _forward_collect(cuda_batches, *forward_arms)
        ref_logits, dut_logits = collected["ref_policy"], collected["dut_policy"]
        assert torch.allclose(ref_logits, dut_logits, atol=A4_ATOL, rtol=A4_RTOL), (
            "policy allclose(1e-2)"
        )
        # Placement atol 0.02 (Main-approved 2026-09-06): the specced 1e-2 sits
        # below the bf16 format floor — placement outputs reach |1.58| where
        # bf16 ulp is 0.0156, and the worst observed element diff is 0.011
        # (mean 0.003 over 4096 elements). 0.02 stays tight; order gates unchanged.
        place_abs = (collected["ref_place"] - collected["dut_place"]).abs()
        print(
            f"\nA4 placement bf16 floor: maxabs {float(place_abs.max().item()):.5f} "
            f"meanabs {float(place_abs.mean().item()):.5f} "
            f"over-1e-2 {int((place_abs > 1e-2).sum().item())}/{place_abs.numel()}"
        )
        assert torch.allclose(
            collected["ref_place"], collected["dut_place"], atol=0.02, rtol=A4_RTOL
        ), "placement allclose(0.02)"
        assert torch.allclose(
            collected["ref_value"], collected["dut_value"], atol=A4_VALUE_ATOL, rtol=A4_VALUE_RTOL
        ), "value allclose(0.15, 0.02)"
        legal, targets = collected["legal"], collected["targets"]
        nll_ref = torch.nn.functional.cross_entropy(
            ref_logits.masked_fill(~legal, float("-inf")), targets, reduction="none"
        )
        nll_dut = torch.nn.functional.cross_entropy(
            dut_logits.masked_fill(~legal, float("-inf")), targets, reduction="none"
        )
        nll_diff = abs(float(nll_ref.mean().item()) - float(nll_dut.mean().item()))
        assert nll_diff < A4_NLL_DIFF, f"mean-NLL diff {nll_diff:.4f} >= {A4_NLL_DIFF}"

        def gap_sign(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.masked_fill(~legal, float("-inf"))
            rows = torch.arange(logits.shape[0])
            masked[rows, targets] = float("-inf")
            return torch.sign(logits[rows, targets] - masked.max(dim=1).values)

        gap_agree = float((gap_sign(ref_logits) == gap_sign(dut_logits)).float().mean().item())
        assert gap_agree >= A4_GAP_AGREE, f"gap-sign agree {gap_agree:.4f} < {A4_GAP_AGREE}"
        ece_ref = _expected_calibration_error(
            torch.softmax(ref_logits.masked_fill(~legal, float("-inf")), dim=-1), targets
        )
        ece_dut = _expected_calibration_error(
            torch.softmax(dut_logits.masked_fill(~legal, float("-inf")), dim=-1), targets
        )
        print(
            f"\nA4: nll-diff {nll_diff:.5f} gap-sign {gap_agree:.4f} "
            f"ECE ref {ece_ref:.4f} dut {ece_dut:.4f} (reported only)"
        )


class TestBf16GradParity:
    def test_a3_gradient_cosine_and_norm(self, cuda_batches, fp32_payload, require_cuda):
        assert require_cuda is not None
        torch.manual_seed(SEED)
        ref_model = _fresh_model(fp32_payload, train=False)
        dut_model = _fresh_model(fp32_payload, train=False)
        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
        dut_opt = torch.optim.AdamW(dut_model.parameters(), lr=1e-3)
        ref_handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=ref_model,
            optimizer=ref_opt,
            spec=_fp32_spec("plain_pytorch"),
        )
        dut_handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=dut_model,
            optimizer=dut_opt,
            spec=_fp32_spec("plain_pytorch"),
        )
        for handle in (ref_handle, dut_handle):
            handle.optimizer.zero_grad(set_to_none=True)
        for batch, targets in cuda_batches:
            ref_loss = masked_cross_entropy(
                ref_handle.model.evaluate(batch).policy_logits, targets, batch.legal_mask
            )
            ref_handle.backward(ref_loss)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dut_logits = dut_handle.model.evaluate(batch).policy_logits
            dut_loss = masked_cross_entropy(dut_logits, targets, batch.legal_mask)
            dut_handle.backward(dut_loss)
        ref_params = dict(unwrap_model(ref_handle.model).named_parameters())
        dut_params = dict(unwrap_model(dut_handle.model).named_parameters())
        assert set(ref_params) == set(dut_params)
        fatal, vacuous = [], 0
        ref_flat, dut_flat = [], []
        for name in sorted(ref_params):
            # Policy-only loss leaves auxiliary heads with grad=None (set_to_none);
            # None is exact-zero on both arms and compares as such below.
            grad_ref = ref_params[name].grad
            grad_dut = dut_params[name].grad
            if grad_ref is None:
                grad_ref = torch.zeros_like(ref_params[name])
            if grad_dut is None:
                grad_dut = torch.zeros_like(dut_params[name])
            assert bool(torch.isfinite(grad_ref).all().item()), f"ref grad non-finite: {name}"
            assert bool(torch.isfinite(grad_dut).all().item()), f"dut grad non-finite: {name}"
            flat_ref = grad_ref.detach().float().flatten()
            flat_dut = grad_dut.detach().float().flatten()
            ref_flat.append(flat_ref)
            dut_flat.append(flat_dut)
            if float(flat_ref.norm().item()) == 0.0 and float(flat_dut.norm().item()) == 0.0:
                vacuous += 1  # auxiliary heads under a policy-only loss: both exactly zero
                continue
            cosine = _cosine_similarity(flat_ref, flat_dut)
            if cosine <= A3_COS_TENSOR:
                fatal.append((name, cosine))
        assert not fatal, f"per-tensor cosine <= {A3_COS_TENSOR}: {fatal[:5]}"
        global_cosine = _cosine_similarity(torch.cat(ref_flat), torch.cat(dut_flat))
        assert global_cosine > A3_COS_GLOBAL, (
            f"global cosine {global_cosine:.5f} <= {A3_COS_GLOBAL}"
        )
        norm_ref = torch.cat(ref_flat).norm().item()
        norm_dut = torch.cat(dut_flat).norm().item()
        ratio = norm_dut / max(norm_ref, 1e-12)
        assert A3_NORM_LO <= ratio <= A3_NORM_HI, (
            f"grad norm-ratio {ratio:.4f} outside [0.95, 1.05]"
        )
        print(
            f"\nA3: global-cosine {global_cosine:.5f} norm-ratio {ratio:.4f} vacuous-zero {vacuous}"
        )


class TestBf16ShortOverlap:
    def test_s1_to_s5_train_track_and_divergence(self, corpus_artifact, fp32_payload, require_cuda):
        assert require_cuda is not None
        steps = _overlap_steps()
        train = corpus_artifact["microbatches"][:TRAIN_MICROBATCHES]
        held_out = corpus_artifact["microbatches"][TRAIN_MICROBATCHES:]
        assert len(held_out) == HELD_OUT_MICROBATCHES
        train_cuda = [(_to_cuda(batch), targets.to("cuda")) for batch, targets in train]
        held_cuda = [(_to_cuda(batch), targets.to("cuda")) for batch, targets in held_out]
        torch.manual_seed(SEED)
        ref_model = _fresh_model(fp32_payload, train=True)
        dut_model = _fresh_model(fp32_payload, train=True)
        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=3e-4, foreach=True)
        dut_opt = torch.optim.AdamW(dut_model.parameters(), lr=3e-4, foreach=True)
        ref_spec = _fp32_spec("plain_pytorch")
        dut_spec = RuntimeSpec(
            adapter_id="fabric_2.6.5",
            device="cuda",
            precision="bf16_mixed",
            compile_mode="eager",
            fullgraph=False,
            dynamic=None,
            backward_pass_autocast="off",
        )
        ref_handle = build_runtime(
            adapter=PlainPytorchAdapter(),
            model=ref_model,
            optimizer=ref_opt,
            spec=ref_spec,
        )
        dut_handle = build_runtime(
            adapter=FabricRuntimeAdapter(),
            model=dut_model,
            optimizer=dut_opt,
            spec=dut_spec,
        )
        for param in unwrap_model(dut_handle.model).parameters():
            assert param.dtype == torch.float32, "Fabric DUT master weights must stay fp32"
        ref_nlls, dut_nlls, norm_ratios = [], [], []
        for step in range(steps):
            batch, targets = train_cuda[step % TRAIN_MICROBATCHES]
            ref_handle.optimizer.zero_grad(set_to_none=True)
            ref_loss = masked_cross_entropy(
                ref_handle.model.evaluate(batch).policy_logits, targets, batch.legal_mask
            )
            ref_handle.backward(ref_loss)
            ref_norm = _global_grad_norm(unwrap_model(ref_handle.model))
            ref_handle.optimizer.step()
            dut_handle.optimizer.zero_grad(set_to_none=True)
            dut_loss = masked_cross_entropy(
                dut_handle.model.evaluate(batch).policy_logits, targets, batch.legal_mask
            )
            dut_handle.backward(dut_loss)
            dut_norm = _global_grad_norm(unwrap_model(dut_handle.model))
            dut_handle.optimizer.step()
            ref_nlls.append(float(ref_loss.detach().float().item()))
            dut_nlls.append(float(dut_loss.detach().float().item()))
            norm_ratios.append(dut_norm / max(ref_norm, 1e-12))
        gaps = [abs(a - b) for a, b in zip(ref_nlls, dut_nlls, strict=True)]
        assert all(math.isfinite(v) for v in ref_nlls), "S2: NaN NLL in ref arm"
        assert all(math.isfinite(v) for v in dut_nlls), "S2: NaN NLL in dut arm"
        assert all(gap < S1_TRACK for gap in gaps), f"S1 track violated: max {max(gaps):.4f}"
        final10 = sum(gaps[-10:]) / len(gaps[-10:])
        assert final10 < S1_FINAL10, f"S1 final-10 {final10:.4f} >= {S1_FINAL10}"
        for arm, nlls in (("ref", ref_nlls), ("dut", dut_nlls)):
            assert all(v != float("inf") and v != float("-inf") for v in nlls), f"S2: Inf in {arm}"
        for t in range(1, steps):
            assert not (dut_nlls[t] > 4.0 * max(dut_nlls[t - 1], 1e-6)), (
                f"S2: 4x loss spike at step {t}: {dut_nlls[t - 1]:.4f} -> {dut_nlls[t]:.4f}"
            )
        bad_ratios = [
            (t, r) for t, r in enumerate(norm_ratios) if not S3_NORM_LO <= r <= S3_NORM_HI
        ]
        assert not bad_ratios, f"S3 norm-ratio outside [0.9, 1.1]: {bad_ratios[:5]}"
        ref_model_eval = unwrap_model(ref_handle.model)
        dut_model_eval = unwrap_model(dut_handle.model)
        ref_model_eval.eval()
        dut_model_eval.eval()
        agree_argmax, agree_top1, total = 0, 0, 0
        with torch.no_grad():
            for batch, _ in held_cuda:
                logits_ref = ref_model_eval.evaluate(batch).policy_logits
                logits_dut = dut_model_eval.evaluate(batch).policy_logits.float()
                argmax_ref = select_actions(logits_ref, batch.legal_mask)
                argmax_dut = select_actions(logits_dut, batch.legal_mask)
                top1_dut = torch.topk(
                    logits_dut.masked_fill(~batch.legal_mask, float("-inf")), k=1, dim=1
                ).indices[:, 0]
                agree_argmax += int((argmax_ref == argmax_dut).sum().item())
                agree_top1 += int((argmax_ref.cpu() == top1_dut.cpu()).sum().item())
                total += batch.legal_mask.shape[0]
        s4_argmax = agree_argmax / total
        s4_top1 = agree_top1 / total
        assert s4_argmax >= S4_ARGMAX_AGREE, (
            f"S4 held-out argmax {s4_argmax:.4f} < {S4_ARGMAX_AGREE}"
        )
        assert s4_top1 >= S4_TOP1_AGREE, f"S4 held-out top-1 {s4_top1:.4f} < {S4_TOP1_AGREE}"
        ref_vec = torch.cat(
            [p.detach().float().cpu().flatten() for p in ref_model_eval.parameters()]
        )
        dut_vec = torch.cat(
            [p.detach().float().cpu().flatten() for p in dut_model_eval.parameters()]
        )
        drift = float((dut_vec - ref_vec).norm().item() / max(ref_vec.norm().item(), 1e-12))
        assert math.isfinite(drift), "S5: non-finite parameter drift"
        print(
            f"\nS(steps={steps}): max|dNLL| {max(gaps):.5f} final10 {final10:.5f} "
            f"norm-ratio[min {min(norm_ratios):.4f}, max {max(norm_ratios):.4f}] "
            f"heldout argmax {s4_argmax:.4f} top1 {s4_top1:.4f} drift {drift:.6f} (S5 info only)"
        )
        print(
            f"arch {_arch_record()} ref {runtime_identity(ref_spec)} "
            f"dut {runtime_identity(dut_spec)}"
        )
