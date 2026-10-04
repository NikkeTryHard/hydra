"""Real-device categorical boundary semantics for the continuous RL actor."""

from __future__ import annotations

import pytest
import torch

from hydra2.training.rl_gpu import strict_categorical

pytestmark = [
    pytest.mark.contract_package("WP-11"),
    pytest.mark.gpu,
    pytest.mark.serial,
]


def test_strict_cdf_skips_illegal_plateaus_and_exact_ties(require_cuda: None) -> None:
    logits = torch.zeros((4, 5), device="cuda")
    mask = torch.tensor(
        [
            [False, True, False, True, False],
            [False, True, False, True, False],
            [False, False, True, False, False],
            [True, False, False, False, True],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    uniform = torch.tensor(
        [0.0, 0.5, 1.0 - 2.0**-53, 1.0 - 2.0**-53], dtype=torch.float64, device="cuda"
    )
    selected, logp = strict_categorical(logits, mask, uniform)
    assert torch.equal(selected.cpu(), torch.tensor([1, 3, 2, 4]))
    expected = torch.tensor(
        [
            -torch.log(torch.tensor(2.0)),
            -torch.log(torch.tensor(2.0)),
            0.0,
            -torch.log(torch.tensor(2.0)),
        ]
    )
    assert torch.equal(logp.cpu(), expected)
    assert torch.equal(
        mask.gather(1, selected.unsqueeze(1)).cpu(), torch.ones((4, 1), dtype=torch.bool)
    )


def test_forced_singleton_uses_exact_zero_log_probability(require_cuda: None) -> None:
    logits = torch.tensor([[1000.0, -1000.0, 999.0]], device="cuda")
    mask = torch.tensor([[False, True, False]], device="cuda")
    for endpoint in (0.0, 1.0 - 2.0**-53):
        selected, logp = strict_categorical(
            logits, mask, torch.tensor([endpoint], dtype=torch.float64, device="cuda")
        )
        assert selected.item() == 1
        assert logp.item() == 0.0


def test_legal_tiny_masses_and_masked_extreme_logits_do_not_repair_selection(
    require_cuda: None,
) -> None:
    logits = torch.tensor([[999.0, -100.0, 0.0, 999.0]], device="cuda")
    mask = torch.tensor([[False, True, True, False]], device="cuda")
    selected, logp = strict_categorical(
        logits, mask, torch.tensor([0.0], dtype=torch.float64, device="cuda")
    )
    assert selected.item() == 1
    assert logp.item() == -100.0
    assert logp.exp().item() > 0.0


def test_replay_preserves_updates_unused_gradients_and_short_tails(require_cuda: None) -> None:
    import copy

    from hydra2.training.rl_learner import LearnerService
    from hydra2.training.rl_objective import PpoParameters

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(60, device="cuda").view(3, 20) / 100)
            self.unused = torch.nn.Parameter(torch.ones(2, device="cuda"))

        def tensor_forward(self, planes, history):
            heads = planes["features"] @ self.weight
            return heads[:, :3], heads

    class Native:
        def slab_geometry(self, t, count):
            return count * 256, count * 48

    base = Model().eval()
    services = []
    for mode in ("eager", "cuda_graph_replay"):
        model = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        service = LearnerService(
            model,
            optimizer,
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            native=Native(),
            compute_stream=torch.cuda.Stream(),
            copy_stream=torch.cuda.Stream(),
            device=torch.device("cuda"),
            parameters=PpoParameters(0.2, 1e-8, 0.005, 0.0, 0.01),
            microbatch_cap=2,
            optimizer_cap=4,
            gradient_clip_norm=1.0,
            cache_byte_budget=65536,
            compile_mode=mode,
        )
        slot = service.slots[0]
        slot.planes = {
            "features": torch.zeros((4, 3), device="cuda"),
            "history_mask": torch.ones((4, 1), device="cuda", dtype=torch.bool),
            "legal_mask": torch.ones((4, 3), device="cuda", dtype=torch.bool),
        }
        slot.selected = torch.zeros(4, device="cuda", dtype=torch.int64)
        slot.old_logp = torch.full((4,), -1.1, device="cuda")
        services.append(service)
    actor_stream = services[1].compute_stream
    actor_input = torch.arange(12, device="cuda", dtype=torch.float32).view(4, 3)
    actor_weight = torch.arange(60, device="cuda", dtype=torch.float32).view(3, 20)
    with torch.cuda.stream(actor_stream):
        for _ in range(3):
            actor_expected = actor_input @ actor_weight
    actor_stream.synchronize()
    actor_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(actor_graph, stream=actor_stream):
        actor_output = actor_input @ actor_weight

    for update, count in enumerate((4, 4, 3, 4)):
        metrics = []
        for service in services:
            slot = service.slots[0]
            slot.count, slot.bucket_t = count, (32 if update < 2 else 64)
            slot.planes["features"].copy_(torch.arange(12, device="cuda").view(4, 3) / (8 + update))
            slot.advantage = torch.arange(count, device="cuda", dtype=torch.float32) - 1.0
            slot.returns = torch.full(
                (count, 4), 0.5 + update / 10, device="cuda", dtype=torch.float64
            )
            slot.ready = torch.cuda.Event()
            slot.ready.record()
            service.begin(0)
            while not service.microstep():
                pass
            rows, terms = service.finish()
            service.boundary()
            assert rows == count
            assert service.model.unused.grad is None
            assert torch.equal(service.model.unused, base.unused)
            metrics.append([term.clone() for term in terms])
        for eager, replay in zip(metrics[0], metrics[1], strict=True):
            torch.testing.assert_close(eager, replay, atol=1e-6, rtol=1e-5)
        for eager, replay in zip(
            services[0].model.parameters(), services[1].model.parameters(), strict=True
        ):
            torch.testing.assert_close(eager, replay, atol=1e-6, rtol=1e-5)
        with torch.cuda.stream(actor_stream):
            actor_graph.replay()
        actor_stream.synchronize()
        assert torch.equal(actor_output, actor_expected)


def test_actor_publish_holds_then_tracks_learner_snapshot(require_cuda: None) -> None:
    """Boundary publish keeps the actor frozen, then tracks the snapshot.

    Actor logits must not move through later learner updates; the
    first-update same-stream publish copies the exact learner snapshot
    in place (stable storage) while the actor stays eval.
    """
    import copy

    from hydra2.training.rl_gpu import GpuService

    torch.manual_seed(41)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    learner = copy.deepcopy(base).to("cuda").eval()
    actor = copy.deepcopy(base).to("cuda").eval()
    service = GpuService(pool=object(), device=torch.device("cuda"), compile_mode="eager")
    fixed = torch.randn(4, 8, device="cuda")
    with torch.inference_mode():
        before = actor(fixed).clone()
    pointers = [parameter.data_ptr() for parameter in actor.parameters()]
    optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        learner(torch.randn(4, 8, device="cuda")).square().mean().backward()
        optimizer.step()
    with torch.inference_mode():
        assert torch.equal(actor(fixed), before)
    service.copy_actor_at_boundary(learner, actor)
    service.compute_stream.synchronize()
    with torch.inference_mode():
        published, expected = actor(fixed), learner(fixed)
    assert torch.equal(published, expected)
    for live, want in zip(actor.state_dict().values(), learner.state_dict().values(), strict=True):
        assert torch.equal(live, want)
    assert [parameter.data_ptr() for parameter in actor.parameters()] == pointers
    assert not actor.training
