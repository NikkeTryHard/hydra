"""Contract tests for full-match PPO and masked behavior cloning."""

from pathlib import Path

import pytest
import torch
from hydra3.arguments import parser
from hydra3.learn import bc, ppo
from hydra3.learn_batch import flatten_games, minibatch_order
from hydra3.learn_collect import collect_generation
from hydra3.learn_dataset import load_dataset, save_dataset
from hydra3.learn_schema import PROVENANCE_SELF_PLAY
from hydra3.runtime import ArenaBatch

from .learn_support import (
    TinyPolicy,
    assert_finite_metrics,
    float_metric,
    install_scripted_arena,
    make_batch,
    make_game,
)


def _one_game() -> tuple[list[ArenaBatch], list[tuple[int, list[int], list[int]]]]:
    scripts = [
        make_batch([(0, 0, [10, 11])]),
        make_batch([(0, 1, [20, 21, 22])]),
        make_batch([(0, 2, [30, 31])]),
        make_batch([(0, 3, [40])]),
    ]
    return scripts, [(0, [25000] * 4, [1, 2, 3, 4])]


def test_ppo_profile_is_opt_in_and_later_options_override_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["ppo", "initial.pt", "output.pt"]
    plain = parser().parse_args(argv)
    (tmp_path / "ppo.local.args").write_text(
        "--minibatch=32\n--games=48\n--compile-actor\n", encoding="utf-8"
    )
    assert parser().parse_args(argv) == plain
    chosen = parser().parse_args([*argv, "@ppo.local.args", "--games=16", "--no-compile-actor"])
    assert (chosen.minibatch, chosen.games, chosen.compile_actor) == (32, 16, False)


def test_per_seat_terminal_credit_and_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts, outcomes = _one_game()
    arenas = install_scripted_arena(monkeypatch, scripts, outcomes)
    model = TinyPolicy(value_bias=1.0)
    rng_state = torch.random.get_rng_state()

    rollout = collect_generation(model, games=1, seed=123, device=torch.device("cpu"))

    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert len(arenas) == 1
    assert arenas[0].steps == [[10], [20], [30], [40]]
    record = rollout.games[0].record
    assert record.chosen.tolist() == [10, 20, 30, 40]
    assert record.seats.tolist() == [0, 1, 2, 3]
    assert record.legal.tolist() == [2, 3, 2, 1]
    assert torch.allclose(rollout.games[0].old_value, torch.ones(4))
    assert torch.allclose(rollout.games[0].old_logp, torch.zeros(4), atol=1e-6)

    flat = flatten_games([record])
    assert flat.target.tolist() == pytest.approx([1.5, 0.5, -0.5, -1.5])
    assert flat.forced.tolist() == [False, False, False, True]
    assert flat.chosen_index.tolist() == [0, 0, 0, 0]


def test_archive_keeps_games_grouped_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = [
        make_batch([(0, 0, [10]), (1, 1, [20, 21])]),
        make_batch([(0, 2, [30, 31, 32]), (1, 3, [40])]),
        make_batch([(1, 0, [50, 51])]),
    ]
    install_scripted_arena(monkeypatch, scripts, [(i, [25000] * 4, [1, 2, 3, 4]) for i in range(2)])

    left, right = collect_generation(
        TinyPolicy(), games=2, seed=123, device=torch.device("cpu")
    ).games

    assert left.record.seats.tolist() == [0, 2]
    assert left.record.legal.tolist() == [1, 3]
    assert left.record.actions[:, 0].tolist() == [10, 30, 31, 32]
    assert right.record.seats.tolist() == [1, 3, 0]
    assert right.record.legal.tolist() == [2, 1, 2]
    assert right.record.actions[:, 0].tolist() == [20, 21, 40, 50, 51]


def test_ppo_skips_forced_actor_loss_but_optimizes_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = [
        make_batch([(0, 0, [1])]),
        make_batch([(0, 1, [2])]),
        make_batch([(0, 2, [3])]),
        make_batch([(0, 3, [4])]),
    ]
    install_scripted_arena(monkeypatch, scripts, [(0, [25000] * 4, [1, 2, 3, 4])])
    model = TinyPolicy(value_bias=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = ppo(
        model,
        optimizer,
        games=1,
        seed=7,
        device=torch.device("cpu"),
        epochs=1,
        minibatch=2,
    )

    assert_finite_metrics(result)
    assert result["trainable_decisions"] == 0
    assert result["decisions"] == 4
    assert result["updates"] == 2
    assert result["policy_loss"] == 0.0
    assert result["entropy"] == 0.0
    assert float_metric(result, "value_loss") > 0.0
    assert result["mean_rank"] == pytest.approx(2.5)
    assert model.training is False


def test_record_roundtrip_trains_bc_on_the_same_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts, outcomes = _one_game()
    install_scripted_arena(monkeypatch, scripts, outcomes)
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    corpus = tmp_path / "rollout.pt"

    result = ppo(
        model,
        optimizer,
        games=1,
        seed=11,
        device=torch.device("cpu"),
        epochs=1,
        minibatch=2,
        record=corpus,
    )
    assert result["trainable_decisions"] == 3
    assert corpus.is_file()

    dataset = load_dataset(corpus)
    assert dataset.provenance == PROVENANCE_SELF_PLAY
    assert dataset.games[0].legal.tolist() == [2, 3, 2, 1]
    assert dataset.games[0].chosen.tolist() == [10, 20, 30, 40]

    clone = TinyPolicy()
    clone_optimizer = torch.optim.SGD(clone.parameters(), lr=0.1)
    bc_result = bc(
        clone,
        clone_optimizer,
        source=corpus,
        device=torch.device("cpu"),
        epochs=2,
        minibatch=2,
        seed=77,
    )
    assert_finite_metrics(bc_result)
    assert bc_result["source_provenance"] == PROVENANCE_SELF_PLAY
    assert bc_result["trainable_decisions"] == 3
    assert bc_result["accuracy"] == 1.0
    assert clone.training is False


def test_bc_rejects_corpora_without_a_choice(tmp_path: Path) -> None:
    corpus = tmp_path / "forced.pt"
    game = make_game([(0, [1]), (1, [2]), (2, [3]), (3, [4])])
    save_dataset(corpus, [game], "authorized_external")
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="non-forced"):
        bc(
            model,
            optimizer,
            source=corpus,
            device=torch.device("cpu"),
            epochs=1,
            minibatch=2,
            seed=5,
        )


def test_minibatch_streams_are_deterministic() -> None:
    first_generator = torch.Generator().manual_seed(99)
    second_generator = torch.Generator().manual_seed(99)
    state = torch.random.get_rng_state()
    first = list(minibatch_order(11, 3, first_generator))
    second = list(minibatch_order(11, 3, second_generator))
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(first) == len(second) == 4
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    scripts, outcomes = _one_game()
    install_scripted_arena(monkeypatch, scripts, outcomes)
    model = TinyPolicy().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = ppo(
        model,
        optimizer,
        games=1,
        seed=19,
        device=device,
        epochs=1,
        minibatch=2,
    )
    assert_finite_metrics(result)
