import pytest
import torch
from hydra3.evaluate import evaluate
from hydra3.model import ModelConfig, Policy
from hydra3.runtime import derive_seed


# Whole-hanchan simulation: seconds per game, so it owns the GPU lane alone. The
# device falls back to CPU so the rotation contract still runs on CPU-only hosts.
@pytest.mark.gpu
def test_self_comparison_cancels_exactly_after_wall_rotation():
    torch.manual_seed(derive_seed(21, "test/evaluation-network", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(width=16, depth=1, heads=2, ff=32, scorer="plain")
    model = Policy(cfg).eval().to(device)
    result = evaluate(model, model, blocks=1, seed=78, device=device)
    assert result["completed_hanchan"] == 4
    assert result["placements_first_to_fourth"] == [1, 1, 1, 1]
    assert result["rank_delta_vs_mean_opponent_lower_is_better"] == 0.0
    assert result["wall_block_bootstrap_95pct_descriptive"] is None
