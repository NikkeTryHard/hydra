import torch
from hydra3.checkpoint import load_checkpoint, save_checkpoint
from hydra3.model import ModelConfig, Policy
from hydra3.runtime import derive_seed


def test_checkpoint_preserves_architecture_and_predictions(tmp_path):
    torch.manual_seed(derive_seed(17, "test/checkpoint", 0))
    cfg = ModelConfig(width=32, depth=1, heads=2, ff=64, scorer="plain")
    policy = Policy(cfg).eval()
    tokens = torch.rand(1, 39, 48)
    actions = torch.zeros(1, 2, 16, dtype=torch.int32)
    actions[..., 7:11] = -1
    actions[0, 0, :4] = torch.tensor([4, 1, 4, 0])
    actions[0, 1, :4] = torch.tensor([5, 1, 4, 1])
    with torch.no_grad():
        expected = policy(tokens, actions)
        path = tmp_path / "net.pt"
        save_checkpoint(path, policy, updates=3)
        restored, data = load_checkpoint(path, torch.device("cpu"))
        actual = restored(tokens, actions)
    assert restored.config == cfg and data["updates"] == 3
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
