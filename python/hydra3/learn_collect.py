"""One frozen-policy collection pass over complete MahJax hanchan."""

from dataclasses import dataclass
from time import perf_counter_ns

import torch

from .actor_graph import ActorGraphs
from .compiled_actor import CompiledActor
from .learn_batch import (
    DecisionTensors,
    gather_action_log_probs,
    masked_log_probs,
    score_policy,
)
from .learn_schema import GameRecord
from .mahjax_runtime import MahJaxArena
from .model_schema import ACTION_ID, ACTION_WIDTH, TOKEN_COUNT, TOKEN_WIDTH
from .runtime import ArenaBatch, derive_seed

GAME_STREAM = "train/game"
SAMPLING_STREAM = "train/action-sampling"


@dataclass(frozen=True, slots=True)
class GameRollout:
    record: GameRecord
    old_logp: torch.Tensor
    old_value: torch.Tensor

    def __post_init__(self) -> None:
        count = self.record.tokens.shape[0]
        for name, tensor in (("old_logp", self.old_logp), ("old_value", self.old_value)):
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be CPU float32")
            if tuple(tensor.shape) != (count,):
                raise ValueError(f"{name} must have one value per decision")
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Rollout:
    """One generation of decisions in slot-major order.

    Training consumes ``decisions``, ``old_logp`` and ``old_value`` directly.
    ``games`` slices that same storage into validated corpus records only when a
    caller asks for them, so a normal PPO generation never pays for a per-game
    copy or a per-game validation scan of the token block.
    """

    decisions: DecisionTensors
    old_logp: torch.Tensor
    old_value: torch.Tensor
    chosen: torch.Tensor
    seeds: tuple[int, ...]
    scores: torch.Tensor
    ranks: torch.Tensor
    counts: torch.Tensor
    order: torch.Tensor

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("a rollout must contain at least one game")
        if len(set(seeds)) != len(seeds):
            raise ValueError("a rollout requires unique game seeds")
        count = self.decisions.tokens.shape[0]
        for name, tensor in (("old_logp", self.old_logp), ("old_value", self.old_value)):
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
            if tuple(tensor.shape) != (count,):
                raise ValueError(f"{name} must have one value per decision")
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"{name} must be finite")
        if tuple(self.counts.shape) != (len(seeds),):
            raise ValueError("counts must hold one decision count per game")
        if int(self.counts.sum()) != count:
            raise ValueError("game decision counts must cover every collected decision")
        if tuple(self.ranks.shape) != (len(seeds), 4) or tuple(self.scores.shape) != (
            len(seeds),
            4,
        ):
            raise ValueError("outcomes must hold one score and rank per seat")
        object.__setattr__(self, "seeds", seeds)

    @property
    def games(self) -> tuple[GameRollout, ...]:
        """Build corpus records on demand; the training path never calls this.

        Decisions are stored in collection order, so each game gathers its rows
        through ``order``. The legal descriptor block is already contiguous per
        decision, which makes the action slice a plain view.
        """
        decisions = self.decisions
        tokens = decisions.tokens.cpu().to(torch.float32)
        actions = decisions.actions.cpu()
        legal_all = decisions.legal.cpu()
        offsets = decisions.offsets.cpu()
        seats_all = decisions.seats.cpu()
        forced_all = decisions.forced.cpu()
        chosen_all = self.chosen.cpu()
        logp_all = self.old_logp.cpu()
        value_all = self.old_value.cpu()
        counts = self.counts.cpu()
        starts = torch.cumsum(counts, dim=0) - counts
        ordered: list[GameRollout] = []
        for slot, game_seed in enumerate(self.seeds):
            count = int(counts[slot])
            if count <= 0:
                raise RuntimeError("a completed game produced no decisions")
            first = int(starts[slot])
            rows = self.order[first : first + count]
            # Descriptor rows follow decision order, but this game's decisions
            # are scattered through that order, so gather the union of their
            # per-decision runs instead of slicing one range.
            legal = legal_all.index_select(0, rows)
            run_start = offsets.index_select(0, rows)
            run_first = torch.repeat_interleave(torch.cumsum(legal, dim=0) - legal, legal)
            inner = torch.arange(int(legal.sum())) - run_first
            picked = torch.repeat_interleave(run_start, legal) + inner
            record = GameRecord(
                seed=game_seed,
                tokens=tokens.index_select(0, rows),
                actions=actions.index_select(0, picked),
                legal=legal,
                chosen=chosen_all.index_select(0, rows),
                seats=seats_all.index_select(0, rows),
                forced=forced_all.index_select(0, rows),
                scores=self.scores[slot],
                ranks=self.ranks[slot],
            )
            ordered.append(
                GameRollout(
                    record=record,
                    old_logp=logp_all.index_select(0, rows),
                    old_value=value_all.index_select(0, rows),
                )
            )
        return tuple(ordered)


class _Archive:
    """Decisions land in growable buffers as each turn completes.

    Retaining whole turns and concatenating them per chunk kept tens of
    gigabytes of float32 tokens alive alongside a second full-size copy, and the
    largest cohorts died allocating it. Writing each turn straight into the
    archive keeps peak memory at the archive itself. Tokens are bfloat16 because
    the policy already runs its forward pass in bf16 autocast, and descriptor
    rows stay int32 because that is what the arena hands out.

    An allocation failure spills the archive to host memory rather than guessing
    a budget from a free-memory reading.
    """

    __slots__ = (
        "_action_capacity",
        "_token_capacity",
        "actions",
        "chosen",
        "count",
        "legal",
        "logp",
        "place",
        "rows",
        "seat",
        "slot",
        "tokens",
        "value",
    )

    def __init__(self, place: torch.device, games: int) -> None:
        # A cohort measures about 1,100 decisions per game and 8.5 candidate
        # rows per decision. Reserving that up front keeps one allocation: a
        # growth copy of a multi-gigabyte archive is what pushed the largest
        # cohorts into host memory, and growth remains the fallback.
        self.place = place
        self._token_capacity = max(games * 1_100, 1 << 16)
        self._action_capacity = max(games * 11_000, 1 << 18)
        self.tokens = torch.empty(
            (self._token_capacity, TOKEN_COUNT, TOKEN_WIDTH), dtype=torch.bfloat16, device=place
        )
        self.actions = torch.empty(
            (self._action_capacity, ACTION_WIDTH), dtype=torch.int32, device=place
        )
        self.count = 0
        self.rows = 0
        self.slot: list[torch.Tensor] = []
        self.seat: list[torch.Tensor] = []
        self.legal: list[torch.Tensor] = []
        self.chosen: list[torch.Tensor] = []
        self.logp: list[torch.Tensor] = []
        self.value: list[torch.Tensor] = []

    def _reserve(self, decisions: int, rows: int) -> None:
        if decisions > self._token_capacity:
            self._token_capacity = max(decisions, self._token_capacity * 5 // 4)
            grown = torch.empty(
                (self._token_capacity, TOKEN_COUNT, TOKEN_WIDTH),
                dtype=torch.bfloat16,
                device=self.place,
            )
            grown[: self.count].copy_(self.tokens[: self.count])
            self.tokens = grown
        if rows > self._action_capacity:
            self._action_capacity = max(rows, self._action_capacity * 5 // 4)
            grown_actions = torch.empty(
                (self._action_capacity, ACTION_WIDTH), dtype=torch.int32, device=self.place
            )
            grown_actions[: self.rows].copy_(self.actions[: self.rows])
            self.actions = grown_actions

    def _spill(self) -> None:
        """Move the archive to host memory after the device refused to grow it."""
        if self.place.type == "cpu":
            raise RuntimeError("the archive cannot spill below host memory")
        torch.cuda.empty_cache()
        self.place = torch.device("cpu")
        self.tokens = self.tokens[: self.count].to("cpu")
        self.actions = self.actions[: self.rows].to("cpu")
        self._token_capacity = self.count
        self._action_capacity = self.rows

    def add(
        self,
        batch: ArenaBatch,
        chosen: torch.Tensor,
        logp: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        decisions = batch.slot.numel()
        rows = batch.actions.reshape(-1, ACTION_WIDTH)
        valid = rows[:, ACTION_ID] >= 0
        kept = int(valid.sum())
        try:
            self._reserve(self.count + decisions, self.rows + kept)
        except torch.cuda.OutOfMemoryError:
            self._spill()
            self._reserve(self.count + decisions, self.rows + kept)
        end = self.count + decisions
        self.tokens[self.count : end].copy_(batch.tokens.to(torch.bfloat16))
        self.actions[self.rows : self.rows + kept].copy_(rows[valid])
        self.count = end
        self.rows += kept
        self.slot.append(batch.slot)
        self.seat.append(batch.seat)
        self.legal.append(valid.view(batch.actions.shape[:2]).sum(dim=1))
        self.chosen.append(chosen)
        self.logp.append(logp)
        self.value.append(value)

    def finish(self) -> _Archive:
        if not self.count or not self.rows:
            raise RuntimeError("a completed generation produced no decisions")
        return self


def _sampling_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    return generator.manual_seed(seed)


def _game_seeds(root: int, count: int) -> list[int]:
    seeds: list[int] = []
    used: set[int] = set()
    index = 0
    while len(seeds) < count:
        candidate = derive_seed(root, GAME_STREAM, index)
        index += 1
        if candidate in used:
            continue
        used.add(candidate)
        seeds.append(candidate)
    return seeds


def _slot_major(slots: torch.Tensor, count: int) -> torch.Tensor:
    """Order decisions by slot, keeping collection order inside each game."""
    order = torch.argsort(slots, stable=True)
    if order.numel() != count:
        raise RuntimeError("slot ordering lost collected decisions")
    return order


def _play_cohort(
    model: torch.nn.Module,
    seeds: list[int],
    sampling_seed: int,
    device: torch.device,
    graph_actor: bool,
    compile_actor: bool,
) -> tuple[_Archive, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Play one cohort of matches and archive its decisions as they arrive."""
    arena = MahJaxArena(seeds, device)
    generator = _sampling_generator(device, sampling_seed)
    archive = _Archive(device, len(seeds))
    if compile_actor:
        graphs = CompiledActor(device)
    else:
        graphs = ActorGraphs(device) if graph_actor else None
    score = score_policy if graphs is None else graphs
    counters = {
        "arena_packet_wait_ns": 0,
        "actor_submit_ns": 0,
        "arena_step_submit_ns": 0,
        "rollout_archive_ns": 0,
        "actor_padded_candidates": 0,
    }

    while True:
        start = perf_counter_ns()
        batch = arena.batch()
        counters["arena_packet_wait_ns"] += perf_counter_ns() - start
        if batch is None:
            break
        counters["actor_padded_candidates"] += batch.actions.shape[0] * batch.actions.shape[1]
        start = perf_counter_ns()
        with torch.no_grad():
            masked_logits, value, _ = score(model, batch.tokens, batch.actions)
            log_probs = masked_log_probs(masked_logits)
            choice = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(1)
            logp = gather_action_log_probs(log_probs, choice)
            selected = batch.actions[:, :, 0].gather(1, choice.unsqueeze(1)).squeeze(1)
        counters["actor_submit_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        arena.step(batch.slot, selected)
        counters["arena_step_submit_ns"] += perf_counter_ns() - start
        start = perf_counter_ns()
        archive.add(batch, selected, logp.detach(), value.detach())
        counters["rollout_archive_ns"] += perf_counter_ns() - start
    if graphs is not None:
        counters["actor_graph_hits"] = graphs.hits
    if isinstance(graphs, CompiledActor):
        counters["actor_compiled_calls"] = graphs.hits
        counters["actor_eager_fallback_calls"] = graphs.fallbacks
    scores, ranks = arena.outcomes()
    return archive.finish(), scores, ranks, counters


def collect_generation(
    model: torch.nn.Module,
    *,
    games: int,
    seed: int,
    device: torch.device,
    telemetry: dict[str, int] | None = None,
    graph_actor: bool = False,
    compile_actor: bool = False,
) -> Rollout:
    """Play the frozen policy through full matches and snapshot its decisions.

    One arena, one serial turn pipeline: the host waits for each turn's packet
    before the actor and the next step run. Two cohorts on separate threads were
    measured twice. With shared executables the second thread tripped a device
    assert on the donated buffers; with a private executable per cohort it ran,
    but the two cohorts only overlapped by 1.11x because a turn is bound by
    device time rather than host time, which is not worth the extra state.
    """
    if type(games) is not int or games < 1:
        raise ValueError("games must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if (graph_actor or compile_actor) and device.type != "cuda":
        raise ValueError("CUDA actor scoring requires CUDA")

    model.eval()
    seeds = _game_seeds(seed, games)
    archive, scores, ranks, counters = _play_cohort(
        model, seeds, derive_seed(seed, SAMPLING_STREAM, 0), device, graph_actor, compile_actor
    )
    outcome_scores = [scores]
    outcome_ranks = [ranks]
    if telemetry is not None:
        telemetry.update(counters)
    finalize_start = perf_counter_ns() if telemetry is not None else 0

    place = archive.place
    tokens = archive.tokens[: archive.count]
    actions = archive.actions[: archive.rows].to(place)
    slots = torch.cat(archive.slot).to(torch.int64)
    legal = torch.cat(archive.legal).to(place)
    count = slots.numel()
    if count != archive.count or bool((legal <= 0).any()) or int(legal.sum()) != actions.shape[0]:
        raise RuntimeError("MahJax candidate rows do not match the decision counts")
    chosen = torch.cat(archive.chosen).to(torch.int64).to(place)
    if bool((chosen < 0).any()):
        raise FloatingPointError("sampling selected a padded MahJax action")
    seats = torch.cat(archive.seat).to(torch.int64).to(place)
    old_logp = torch.cat(archive.logp).to(place)
    old_value = torch.cat(archive.value).to(place)
    if not bool(torch.isfinite(old_logp).all()) or not bool(torch.isfinite(old_value).all()):
        raise FloatingPointError("behaviour log probabilities or values are non-finite")

    counts = torch.zeros(games, dtype=torch.int64)
    counts.index_add_(0, slots, torch.ones_like(slots))
    if bool((counts == 0).any()):
        raise RuntimeError("a completed game produced no decisions")

    offsets = torch.cumsum(legal, dim=0) - legal
    decision_of_row = torch.repeat_interleave(torch.arange(count, device=legal.device), legal)
    matches = actions[:, 0].eq(chosen.repeat_interleave(legal))
    match_counts = torch.zeros(count, dtype=torch.int64, device=legal.device)
    match_counts.index_add_(0, decision_of_row, matches.to(torch.int64))
    if not bool((match_counts == 1).all()):
        raise ValueError("chosen canonical id does not name one legal descriptor")
    match_positions = matches.nonzero(as_tuple=False).squeeze(1)
    decisions_hit = decision_of_row.index_select(0, match_positions)
    if not torch.equal(decisions_hit, torch.arange(count, device=legal.device)):
        raise ValueError("chosen canonical matches are not in decision order")
    chosen_index = match_positions - offsets.index_select(0, decisions_hit)

    scores = torch.cat(outcome_scores, dim=0)
    ranks = torch.cat(outcome_ranks, dim=0)
    if tuple(scores.shape) != (games, 4) or tuple(ranks.shape) != (games, 4):
        raise RuntimeError("cohort outcomes do not cover every game")
    archive_device = tokens.device
    target = (
        2.5
        - ranks.to(archive_device).to(torch.float32)[
            slots.to(archive_device), seats.to(archive_device)
        ]
    )

    decisions = DecisionTensors(
        tokens=tokens,
        actions=actions,
        offsets=offsets,
        legal=legal,
        chosen_index=chosen_index,
        target=target,
        forced=legal == 1,
        seats=seats,
    )
    if telemetry is not None:
        telemetry["rollout_finalize_ns"] = perf_counter_ns() - finalize_start
        telemetry["actor_legal_candidates"] = actions.shape[0]
        telemetry["rollout_archive_on_device"] = int(place.type == "cuda")
    return Rollout(
        decisions=decisions,
        old_logp=old_logp,
        old_value=old_value,
        chosen=chosen,
        seeds=tuple(seeds),
        scores=scores,
        ranks=ranks,
        counts=counts,
        order=_slot_major(slots, count).cpu(),
    )
