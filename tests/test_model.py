"""Contract tests for the actor-visible policy."""

import hydra3.model_schema as schema
import pytest
import torch
from hydra3.model import ModelConfig, Policy
from hydra3.model_action import action_view
from hydra3.model_config import SCORER_EDIT, SCORER_PLAIN
from hydra3.model_scorer import EditScorer
from torch import Tensor
from torch.nn import functional as F

SCORERS = (SCORER_EDIT, SCORER_PLAIN)


def _model(scorer: str = SCORER_EDIT) -> Policy:
    return Policy(ModelConfig(width=32, depth=2, heads=2, ff=64, scorer=scorer))


def _tokens(batch: int = 2) -> Tensor:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(batch, schema.TOKEN_COUNT, schema.TOKEN_WIDTH, generator=generator)
    tokens[:, : schema.TILE_TOKENS, schema.OWN_COUNT] = 0.0
    tokens[:, 5, schema.OWN_COUNT] = 2.0
    tokens[:, 7, schema.OWN_COUNT] = 1.0
    tokens[:, : schema.TILE_TOKENS, schema.OWN_RED] = 0.0
    tokens[:, 5, schema.OWN_RED] = 1.0
    tokens[:, :, schema.RESERVED_FEATURE] = 0.0
    return tokens


def _candidate(
    kind: int,
    face: int = -1,
    *,
    action_id: int = 1,
    primary_red: int = 0,
    drawn: int = 0,
    riichi: int = 0,
    removals: tuple[int, ...] = (),
    removal_reds: tuple[int, ...] = (),
) -> Tensor:
    candidate = torch.full((schema.ACTION_WIDTH,), -1, dtype=torch.int32)
    candidate[schema.ACTION_ID] = action_id
    candidate[schema.ACTION_KIND] = kind
    candidate[schema.ACTION_SOURCE] = schema.SOURCE_ABSENT
    candidate[schema.ACTION_PRIMARY_FACE] = face
    candidate[schema.ACTION_PRIMARY_RED] = primary_red
    candidate[schema.ACTION_DRAWN_DISCARD] = drawn
    candidate[schema.ACTION_DECLARE_RIICHI] = riichi
    candidate[schema.ACTION_CLAIM_INTENT] = 0
    for index, removal in enumerate(removals):
        candidate[schema.ACTION_REMOVAL_FIRST + index] = removal
        red = removal_reds[index] if index < len(removal_reds) else 0
        candidate[schema.ACTION_REMOVAL_RED_FIRST + index] = red
    return candidate


def _candidate_batch() -> Tensor:
    return torch.stack(
        (
            _candidate(
                schema.KIND_DISCARD,
                5,
                action_id=100,
                primary_red=1,
                removals=(5,),
                removal_reds=(1,),
            ),
            _candidate(
                schema.KIND_TSUMOGIRI,
                5,
                action_id=101,
                primary_red=1,
                drawn=1,
                removals=(5,),
                removal_reds=(1,),
            ),
            _candidate(
                schema.KIND_RIICHI_DISCARD,
                7,
                action_id=102,
                riichi=1,
                removals=(7,),
            ),
            _candidate(schema.KIND_PASS, action_id=103),
            torch.full((schema.ACTION_WIDTH,), -1, dtype=torch.int32),
        )
    )


def _batched(candidates: Tensor, batch: int = 2) -> Tensor:
    return candidates.unsqueeze(0).expand(batch, -1, -1).clone()


def test_config_defaults_and_validation() -> None:
    config = ModelConfig()
    assert (config.width, config.depth, config.heads, config.ff, config.scorer) == (
        128,
        4,
        4,
        384,
        SCORER_EDIT,
    )
    assert Policy(config).config is config
    with pytest.raises(ValueError):
        ModelConfig(width=0)
    with pytest.raises(ValueError):
        ModelConfig(depth=0)
    with pytest.raises(ValueError):
        ModelConfig(heads=0)
    with pytest.raises(ValueError):
        ModelConfig(ff=0)
    with pytest.raises(ValueError):
        ModelConfig(width=30)
    with pytest.raises(ValueError):
        ModelConfig(scorer="unknown")


@pytest.mark.parametrize(
    "config",
    (
        ModelConfig(width=48, depth=1, heads=3, ff=72),
        ModelConfig(width=64, depth=3, heads=4, ff=96, scorer=SCORER_PLAIN),
    ),
)
def test_configurable_geometry_produces_valid_output(config: ModelConfig) -> None:
    torch.manual_seed(8)
    model = Policy(config)
    candidates = _candidate_batch().unsqueeze(0)
    with torch.no_grad():
        logits, value = model(_tokens(batch=1), candidates)
    assert logits.shape == (1, candidates.shape[1])
    assert value.shape == (1,)
    assert torch.isfinite(logits[candidates[..., schema.ACTION_ID] >= 0]).all()


@pytest.mark.parametrize("scorer", SCORERS)
def test_forward_masks_padding_and_masked_loss_is_finite(scorer: str) -> None:
    torch.manual_seed(0)
    model = _model(scorer)
    tokens = _tokens()
    candidates = _batched(_candidate_batch())
    calls: list[int] = []

    def count_call(*_: object) -> None:
        calls.append(1)

    handle = model.encoder.register_forward_hook(count_call)
    try:
        logits, value = model(tokens, candidates)
    finally:
        handle.remove()
    assert len(calls) == 1

    assert logits.shape == (2, 5)
    assert value.shape == (2,)
    legal = candidates[..., schema.ACTION_ID] >= 0
    assert legal.any(dim=-1).all()
    assert torch.isfinite(logits[legal]).all()
    assert torch.isneginf(logits[~legal]).all()
    assert torch.isfinite(value).all()

    targets = legal.to(torch.float32).argmax(dim=-1)
    loss = F.cross_entropy(logits, targets)
    loss = loss + F.mse_loss(value, torch.tensor([1.0, -0.5]))
    loss.backward()
    assert torch.isfinite(loss)
    saw_gradient = False
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            saw_gradient = True
            assert torch.isfinite(gradient).all()
    assert saw_gradient


@pytest.mark.parametrize("scorer", SCORERS)
def test_candidate_order_and_visible_copy_ids_do_not_matter(scorer: str) -> None:
    torch.manual_seed(1)
    model = _model(scorer)
    tokens = _tokens(batch=1)
    candidates = _candidate_batch()
    order = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        logits, _ = model(tokens, candidates.unsqueeze(0))
        permuted, _ = model(tokens, candidates[order].unsqueeze(0))
    assert torch.equal(permuted, logits[:, order])

    first = _candidate(schema.KIND_DISCARD, 5, action_id=20, removals=(5,))
    second = _candidate(schema.KIND_DISCARD, 5, action_id=2000, removals=(5,))
    pair = torch.stack((first, second)).unsqueeze(0)
    with torch.no_grad():
        pair_logits, _ = model(tokens, pair)
    assert torch.equal(pair_logits[:, 0], pair_logits[:, 1])


@pytest.mark.parametrize("scorer", SCORERS)
def test_meaningful_shape_and_drawn_marker_change_scores(scorer: str) -> None:
    torch.manual_seed(2)
    model = _model(scorer)
    tokens = _tokens(batch=1)
    discard = _candidate(schema.KIND_DISCARD, 5, action_id=1, removals=(5,)).reshape(
        1, 1, schema.ACTION_WIDTH
    )
    changed = tokens.clone()
    changed[:, 5, schema.OWN_COUNT] = 3.0
    changed[:, 5, schema.OWN_RED] = 0.0
    with torch.no_grad():
        original, _ = model(tokens, discard)
        shifted, _ = model(changed, discard)
    assert not torch.allclose(original, shifted)

    held = _candidate(schema.KIND_DISCARD, 5, action_id=2, drawn=0, removals=(5,))
    drawn = _candidate(schema.KIND_TSUMOGIRI, 5, action_id=3, drawn=1, removals=(5,))
    pair = torch.stack((held, drawn)).unsqueeze(0)
    with torch.no_grad():
        pair_logits, _ = model(tokens, pair)
    assert not torch.allclose(pair_logits[:, 0], pair_logits[:, 1])


def test_edit_scorer_noop_is_zero_and_ankan_aggregates_copies() -> None:
    torch.manual_seed(3)
    scorer = EditScorer(width=16)
    encoded = torch.randn(1, schema.TOKEN_COUNT, 16)
    counts = torch.zeros(1, schema.TILE_FACES)
    reds = torch.zeros_like(counts)

    noop = _candidate(schema.KIND_PASS, action_id=1).reshape(1, 1, schema.ACTION_WIDTH)
    no_edit = scorer.private_edit(encoded, action_view(noop), counts, reds)
    assert torch.equal(no_edit, torch.zeros_like(no_edit))

    counts[0, 5] = 4.0
    ankan = _candidate(schema.KIND_ANKAN, 5, removals=(5, 5, 5, 5)).reshape(
        1, 1, schema.ACTION_WIDTH
    )
    actual = scorer.private_edit(encoded, action_view(ankan), counts, reds)[0, 0]

    tile = encoded[0, 5]
    before_input = torch.cat((tile, torch.tensor([1.0, 0.0])))
    after_input = torch.cat((tile, torch.tensor([0.0, 0.0])))
    mapped = scorer.edit(torch.stack((before_input, after_input)))
    expected = mapped[1] - mapped[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
