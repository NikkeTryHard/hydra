import pytest
import torch
from hydra3.compiled_actor import CompiledActor

from .learn_support import TinyPolicy, make_batch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.no_grad()
def test_replay_keeps_archived_values_and_all_legal_moves():
    device = torch.device("cuda")
    actor = CompiledActor(device)
    policy = TinyPolicy(prefer_first=False).eval().to(device)
    first = make_batch([(0, 1, [2, 4, 8]), (1, 2, [3])], device)
    first.tokens.fill_(2)
    _, archived, mask = actor(policy, first.tokens, first.actions)
    saved = archived.clone()
    torch.testing.assert_close(saved, torch.full_like(saved, 2))
    assert torch.equal(mask, first.actions[:, :, 0] >= 0)

    second = make_batch([(0, 2, [1, 12])], device)
    second.tokens.fill_(7)
    logits, current, _ = actor(policy, second.tokens, second.actions)
    torch.testing.assert_close(current, torch.full_like(current, 7))
    assert torch.equal(archived, saved)
    assert second.actions[0, logits[0].argmax(), 0].item() == 12

    wide = make_batch([(0, 0, list(range(17)))], device)
    logits, _, mask = actor(policy, wide.tokens, wide.actions)
    assert torch.equal(mask, wide.actions[:, :, 0] >= 0)
    assert wide.actions[0, logits[0].argmax(), 0].item() == 16
