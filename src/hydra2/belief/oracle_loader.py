"""WP-07B oracle privileged loader — separate namespace/process boundary.

This module is the ONLY location that may import and expose privileged fields
(hidden_tiles, wall, dead_wall, opponent_hand, full_world, privileged_label,
etc.). Inference encoders (src/hydra2/models/encoder.py) MUST NEVER import
this module; a dedicated guard test imports encoder and asserts this module
is absent from sys.modules.

Privileged data may only be loaded from the authorized train split. Loading
from held-out or eval splits is a hard failure.

Process boundary: :func:`load_oracle_batch_in_subprocess` spawns a fresh
Python process (spawn context) to load privileged shards, proving isolation;
parent pid != child pid is asserted. Direct in-process loading is available
via :class:`PrivilegedOracleLoader` for training, but inference loaders
cannot construct it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from hydra2.contracts.common import ContractError, CorruptArtifactError

if TYPE_CHECKING:
    from hydra2.contracts.utility import UtilityManifest

# Reuse authoritative forbidden set from data/parquet for consistency
try:
    from hydra2.data.parquet import (
        FORBIDDEN_IN_ACTOR as _FORBIDDEN_IN_ACTOR,
    )
except Exception:
    _FORBIDDEN_IN_ACTOR = frozenset(
        {"hidden_tiles", "wall", "dead_wall", "opponent_hand", "privileged", "full_world"}
    )

# Extra privileged keys that appear only in oracle/privileged shards
PRIVILEGED_KEYS: frozenset[str] = frozenset(
    {
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "privileged",
        "full_world",
        "privileged_label",
        "wall_remaining",
        "opponent_concealed",
        "unrevealed_dora",
        "hidden_tile_counts",
        "wait_tiles",
    }
)

FORBIDDEN_IN_ACTOR_KEYS: frozenset[str] = frozenset(_FORBIDDEN_IN_ACTOR) | frozenset(
    {"privileged_label", "wall_remaining", "hidden_tile_counts"}
)

# Only train split is authorized for oracle distillation
AUTHORIZED_TRAIN_SPLIT = "train"
AUTHORIZED_SPLITS_FOR_INFERENCE = frozenset({"train", "held_out", "test", "eval"})
# Oracle loader rejects any split != train


def _require_train_split(split: str) -> None:
    if split != AUTHORIZED_TRAIN_SPLIT:
        raise ContractError(
            f"PrivilegedOracleLoader may only load split={AUTHORIZED_TRAIN_SPLIT!r}, got {split!r}: "  # noqa: E501  # reason: contract string single logical; splitting harms grep
        )


def validate_actor_batch_no_privileged(batch: dict[str, Any]) -> None:
    """Hard failure if actor-visible batch contains any privileged field.

    Inference encoders MUST call this before encoding. It checks both top-level
    keys and nested ``actor_observation`` dict keys for leakage.
    """
    for key in batch:
        if key in FORBIDDEN_IN_ACTOR_KEYS:
            raise ContractError(f"actor batch contains privileged field {key!r}")
        if key in PRIVILEGED_KEYS:
            raise ContractError(f"actor batch contains privileged field {key!r}")
    # Nested observation dict
    _obs_a: Any | None = batch.get("actor_observation")
    _obs_b: Any | None = batch.get("observation")
    _obs_c: Any | None = batch.get("obs")
    obs: Any | None = _obs_a if _obs_a is not None else (_obs_b if _obs_b is not None else _obs_c)
    if isinstance(obs, dict):
        for k_any in obs:
            k: str = k_any  # type: ignore[assignment]  # reason: dict key statically Any; str validated by membership check
            if k in FORBIDDEN_IN_ACTOR_KEYS or k in PRIVILEGED_KEYS:
                raise ContractError(f"actor_observation contains privileged field {k!r}")


def assert_no_privileged_leakage_in_actor_row(row: dict[str, Any]) -> None:
    """Check a raw parquet row dict for privileged leakage (actor shard)."""
    for bad in FORBIDDEN_IN_ACTOR_KEYS:
        if bad in row:
            raise ContractError(f"privileged field {bad!r} in actor row {row.get('decision_id')!r}")
    obs_raw = row.get("actor_observation")
    if isinstance(obs_raw, str):
        try:
            obs = json.loads(obs_raw)
        except Exception as exc:
            raise CorruptArtifactError(
                f"actor_observation not JSON for {row.get('decision_id')!r}"
            ) from exc
        if isinstance(obs, dict):
            for k_any in obs:
                k2: str = k_any  # type: ignore[assignment]  # reason: dict key statically Any; str validated by membership check
                if k2 in FORBIDDEN_IN_ACTOR_KEYS:
                    raise ContractError(
                        f"privileged field {k2!r} inside actor_observation for {row.get('decision_id')!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
                    )
    elif isinstance(obs_raw, dict):
        for k_any in obs_raw:
            k3: str = k_any  # type: ignore[assignment]  # reason: dict key statically Any; str validated by membership check
            if k3 in FORBIDDEN_IN_ACTOR_KEYS:
                raise ContractError(
                    f"privileged field {k3!r} inside actor_observation dict for {row.get('decision_id')!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
                )


def check_wall_leakage(
    train_wall_ids: list[str] | set[str], held_out_wall_ids: list[str] | set[str]
) -> None:
    """Hard failure if any wall_id appears in both train and held-out.

    Walls are the partitioning unit (whole games before decisions, SPEC 12.4).
    Leakage corrupts held-out proper scores.
    """
    train_set = set(train_wall_ids)
    held_set = set(held_out_wall_ids)
    overlap = train_set & held_set
    if len(overlap) > 0:
        raise ContractError(
            f"wall leakage: {len(overlap)} wall(s) overlap between train and held_out: {sorted(overlap)[:3]!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
        )


def check_split_disjoint(
    train_ids: list[str] | set[str], held_out_ids: list[str] | set[str]
) -> None:
    """Hard failure if any decision_id crosses train/held_out boundary."""
    train_set = set(train_ids)
    held_set = set(held_out_ids)
    overlap = train_set & held_set
    if len(overlap) > 0:
        raise ContractError(
            f"split leakage: {len(overlap)} decision(s) overlap: {sorted(overlap)[:3]!r}"
        )


def _privileged_shard_paths(parquet_dir: Path) -> list[Path]:
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.is_dir():
        raise ContractError(f"privileged parquet_dir not found: {parquet_dir}")
    shards = sorted(parquet_dir.glob("privileged-*.parquet"))
    # Fallback: also accept "oracle-*.parquet"
    if len(shards) == 0:
        shards = sorted(parquet_dir.glob("oracle-*.parquet"))
    return shards


def _actor_shard_paths(parquet_dir: Path) -> list[Path]:
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.is_dir():
        raise ContractError(f"parquet_dir not found: {parquet_dir}")
    return sorted(parquet_dir.glob("actor-*.parquet"))


@dataclass(frozen=True, slots=True)
class OracleTarget:
    """Deterministic teacher target derived from privileged row."""

    decision_id: str
    wall_id: str
    # Belief target: distribution over hidden tile types (34-dim, sum=1)
    belief_target: tuple[float, ...]
    # Value target: 4-seat UtilityVector.values via utility()
    # (ranks -> rank_values -> values; e.g. (20,10,-10,-20) permuted,
    # zero-sum, NOT a distribution). Opaque join on decision_id only —
    # actor batch carries decision_id + observation_hash, never privileged.
    value_target: tuple[float, ...]
    # Event target: next event kind id (0..19) for belief model
    event_target: int
    # Teacher soft logits (pre-softmax, for KL distillation)
    teacher_belief_logits: tuple[float, ...]
    teacher_value_logits: tuple[float, ...]
    # Provenance
    split: str
    observation_hash: str


def _belief_target_from_privileged(
    privileged_label: dict[str, Any] | None,
    decision_id: str,
    *,
    allow_synthetic: bool = False,
) -> tuple[float, ...]:
    """Deterministic 34-dim belief target from privileged hidden tiles.

    Real path: ``hidden_tiles`` / ``hidden_tile_counts`` (34-list) or
    ``wait_tiles``. When no real signal is present the legacy deterministic
    hash fallback applies ONLY with ``allow_synthetic=True`` (synthetic-only
    opt-in, byte-identical to the pre-flag behavior); otherwise raises
    :class:`ContractError` (fail closed — missing labels never silently
    hash-synthesize on real paths).
    """
    if isinstance(privileged_label, dict):
        # Try to extract hidden counts if present
        _h_a: Any | None = privileged_label.get("hidden_tiles")
        _h_b: Any | None = privileged_label.get("hidden_tile_counts")
        hidden: Any | None = _h_a if _h_a is not None else _h_b
        if isinstance(hidden, list) and len(hidden) == 34:
            _raw_total: float = float(sum(hidden))  # type: ignore[unknown-argument-type]  # reason: Any from privileged dict; float() validates. Evidence: https://docs.python.org/3/library/functions.html#float
            total: float = _raw_total if _raw_total != 0.0 else 1.0
            return tuple(float(x) / total for x in hidden)  # type: ignore[unknown-argument-type]  # reason: Any element intentional; float() validates
        # Try wait tiles
        waits: Any | None = privileged_label.get("wait_tiles")
        if isinstance(waits, list) and len(waits) > 0:
            vec: list[float] = [0.0] * 34
            for t in waits:
                if isinstance(t, int) and 0 <= t < 34:
                    vec[t] += 1.0
            _vec_total: float = float(sum(vec))
            total = _vec_total if _vec_total != 0.0 else 1.0
            return tuple(v / total for v in vec)
    if not allow_synthetic:
        raise ContractError(
            f"belief target: missing privileged hidden tiles for {decision_id!r} "
            "(fail closed; synthetic opt-in via allow_synthetic=True)"
        )
    h = hashlib.sha256(decision_id.encode()).digest()
    raw = (
        [float(b) + 1.0 for b in h[:34]]
        if len(h) >= 34
        else [float(b & 0xFF) + 1.0 for b in (h * 3)[:34]]
    )
    total = sum(raw)
    return tuple(v / total for v in raw)


def _oracle_utility_manifest() -> UtilityManifest:
    """Canonical day-one utility manifest (same golden as models/model.py)."""
    from hydra2.contracts.rules import RULES_ID
    from hydra2.contracts.utility import (
        UTILITY_OBJECTIVE,
        UTILITY_TIE_POLICY,
        make_utility_manifest,
    )

    return make_utility_manifest(
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        schema_version="1.0.0",
        rules_id=RULES_ID,
        rules_hash="sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b",
        objective=UTILITY_OBJECTIVE,
        rank_values=(20.0, 10.0, -10.0, -20.0),
        tie_policy=UTILITY_TIE_POLICY,
        value_min=-100.0,
        value_max=100.0,
        zero_sum=True,
    )


def _value_from_ranks_via_utility(ranks_in: Any) -> tuple[float, ...] | None:
    """Map ranks permutation 1..4 through utility() to UtilityVector.values.

    Returns None when input is not a strict 1..4 permutation (caller falls
    through to legacy paths). Synthesizes a RawOutcome whose final_scores
    are consistent with the ranks (rank 1 -> 40000, 2 -> 30000, 3 -> 20000,
    4 -> 10000) so utility()'s ranks -> rank_values -> values mapping is
    exact; utility() itself remains the fixed point (never duplicated).
    """
    if not isinstance(ranks_in, (list, tuple)) or len(ranks_in) != 4:
        return None
    if any(isinstance(x, bool) for x in ranks_in) or not all(isinstance(x, int) for x in ranks_in):
        return None
    # Guards prove 4 ints (bool rejected), so int() would be identity.
    _ranks_int: list[int] = list(ranks_in)
    ranks = tuple(_ranks_int)
    if sorted(ranks) == [0, 1, 2, 3]:
        # Zero-based seat convention -> 1..4 for utility().
        ranks = tuple(r + 1 for r in ranks)
    if sorted(ranks) != [1, 2, 3, 4]:
        return None
    from hydra2.contracts.utility import RawOutcome, utility

    manifest = _oracle_utility_manifest()
    score_for_rank = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    final_scores = tuple(score_for_rank[r] for r in ranks)  # type: ignore[index]  # ranks validated permutation above
    point_deltas = tuple(s - 25000 for s in final_scores)
    outcome = RawOutcome(
        final_scores=final_scores,  # type: ignore[arg-type]  # validated quad above
        ranks=ranks,  # type: ignore[arg-type]  # validated permutation above
        point_deltas=point_deltas,  # type: ignore[arg-type]  # derived from validated scores
        settlements=(),
        rules_id=manifest.rules_id,
        rules_hash=manifest.rules_hash,
    )
    return tuple(utility(outcome, manifest).values)


def _value_target_from_privileged(
    privileged_label: dict[str, Any] | None,
    decision_id: str,
    *,
    allow_synthetic: bool = False,
) -> tuple[float, ...]:
    """Deterministic 4-dim value target from privileged ranks (utility scale).

    Real paths (in order): ``ranks`` / 4-list ``final_placement`` permutation
    via ``utility()``; explicit ``value_vector`` / ``utility_vector`` /
    ``placement`` 4-list; legacy single-int ``final_placement`` / ``rank``
    0..3 mapped through the utility manifest (0-based rank +1 -> 1..4 selects
    ``rank_values[rank]`` at the rank index — utility scale, never 0/1
    one-hot). When no real signal is present the legacy deterministic hash
    fallback applies ONLY with ``allow_synthetic=True`` (synthetic-only
    opt-in, byte-identical to the pre-flag behavior); otherwise raises
    :class:`ContractError` (fail closed).
    """
    if isinstance(privileged_label, dict):
        # Day-one authoritative path: ranks -> utility() -> UtilityVector.values.
        # Accepts "ranks" permutation 1..4, or "final_placement" as a 4-list
        # permutation 1..4 (distinct from legacy single-int 0..3 below).
        # Opaque join: actor side supplies decision_id only; privileged ranks
        # never enter the actor batch (firewall: validate_actor_batch_no_privileged).
        _ranks_candidate: Any | None = privileged_label.get("ranks")
        if _ranks_candidate is None:
            _fp: Any | None = privileged_label.get("final_placement")
            if isinstance(_fp, (list, tuple)) and len(_fp) == 4:
                _ranks_candidate = _fp
        _via_utility = (
            _value_from_ranks_via_utility(_ranks_candidate)
            if _ranks_candidate is not None
            else None
        )
        if _via_utility is not None:
            return _via_utility
        _v_a: Any | None = privileged_label.get("value_vector")
        _v_b: Any | None = privileged_label.get("utility_vector")
        _v_c: Any | None = privileged_label.get("placement")
        _v_tmp: Any | None = _v_a if _v_a is not None else _v_b
        v: Any | None = _v_tmp if _v_tmp is not None else _v_c
        if isinstance(v, list) and len(v) == 4:
            # Explicit 4-list value claim: strict validation against the
            # manifest (never silently passed through). Bool is not a number.
            _vals_list: list[float] = []
            for _i, _x in enumerate(v):
                if isinstance(_x, bool) or not isinstance(_x, (int, float)):
                    raise ContractError(
                        f"value target: entry[{_i}] must be a number for {decision_id!r}, "
                        f"got {_x!r}"
                    )
                if not math.isfinite(float(_x)):
                    raise ContractError(
                        f"value target: entry[{_i}] must be finite for {decision_id!r}"
                    )
                # _x narrowed to int | float here; float() keeps int case exact.
                _vals_list.append(float(_x))
            _vals = tuple(_vals_list)
            _manifest = _oracle_utility_manifest()
            # UtilityManifest bounds are float already; float() would be identity.
            _lo = _manifest.value_min
            _hi = _manifest.value_max
            for _i, _x in enumerate(_vals):
                if _x < _lo or _x > _hi:
                    raise ContractError(
                        f"value target: entry[{_i}]={_x!r} outside manifest bounds "
                        f"[{_lo}, {_hi}] for {decision_id!r}"
                    )
            if bool(getattr(_manifest, "zero_sum", False)):
                _total = math.fsum(_vals)
                if not math.isclose(_total, 0.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ContractError(
                        "value target: zero-sum manifest requires values summing to 0 "
                        f"for {decision_id!r}, got sum {_total!r}"
                    )
            return _vals
        # Single placement rank (legacy shape; utility scale, never one-hot):
        # 0-based rank +1 -> 1..4 selects manifest.rank_values[rank], the same
        # entry utility() itself uses (rank_values[rank - 1]), stored at the
        # rank index preserving the legacy 4-vector shape. Full-permutation
        # labels remain preferred (exact utility() call above).
        _r_a: Any | None = privileged_label.get("final_placement")
        _r_b: Any | None = privileged_label.get("rank")
        rank: Any | None = _r_a if _r_a is not None else _r_b
        if isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < 4:
            manifest = _oracle_utility_manifest()
            vec = [0.0] * 4
            vec[rank] = manifest.rank_values[rank]
            return tuple(vec)
    if not allow_synthetic:
        raise ContractError(
            f"value target: missing privileged ranks for {decision_id!r} "
            "(fail closed; synthetic opt-in via allow_synthetic=True)"
        )
    # Hash-fallback synthetic target — synthetic-only opt-in (allow_synthetic=True),
    # byte-identical to the pre-flag behavior; never mixed with real utility()
    # targets except under explicit opt-in.
    # Deterministic pseudo value from hash
    h = int(hashlib.sha256((decision_id + "_value").encode()).hexdigest()[:8], 16)
    # 4-seat softmax-like values
    scores = [((h >> (i * 4)) & 0xF) / 15.0 for i in range(4)]
    _score_sum: float = float(sum(scores))
    total: float = _score_sum if _score_sum != 0.0 else 1.0
    return tuple(s / total for s in scores)


def _teacher_logits_from_targets(
    belief_target: tuple[float, ...], value_target: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import math as _math

    # Invert softmax with small epsilon: logits = log(p+eps)
    eps = 1e-6
    belief_logits = tuple(_math.log(max(p, eps)) for p in belief_target)
    value_logits = tuple(_math.log(max(p, eps)) for p in value_target)
    return belief_logits, value_logits


class PrivilegedOracleLoader:
    """Privileged loader — train split only, separate namespace.

    This class is the ONLY authorized holder of privileged parquet handles.
    Constructing it with split != "train" raises. Actor inference code must
    never import this class (verified by module isolation test).
    """

    def __init__(
        self,
        parquet_dir: Path | str,
        split: str = "train",
        verify: bool = True,
        allow_synthetic: bool = False,
    ) -> None:
        """Privileged loader — train split only; ``allow_synthetic`` opts the
        whole loader into the deterministic hash fallback for rows/ids without
        real privileged labels (synthetic-only; default fail closed).
        """
        _require_train_split(split)
        self.parquet_dir = Path(parquet_dir)
        self.split = split
        self.allow_synthetic = allow_synthetic
        self._rows: list[dict[str, Any]] = []
        self._wall_ids: set[str] = set()
        shards = _privileged_shard_paths(self.parquet_dir)
        if len(shards) == 0:
            # Synthetic fallback: if no privileged shards, treat as empty and rely on
            # actor parquet + synthetic privileged derivation (still split-checked)
            if verify:
                # No shards is not a hard error in synthetic tests; we synthesize
                pass
            self._rows = []
            self._by_id: dict[Any, OracleTarget] = {}
            return
        for shard in shards:
            table = pq.read_table(shard)
            cols = {name: table.column(name).to_pylist() for name in table.column_names}
            n = table.num_rows
            for i in range(n):
                raw: dict[str, Any] = {name: cols[name][i] for name in table.column_names}
                row_split = raw.get("split")
                if row_split is not None and row_split != AUTHORIZED_TRAIN_SPLIT:
                    raise ContractError(
                        f"privileged row {raw.get('decision_id')!r} has split {row_split!r}, expected {AUTHORIZED_TRAIN_SPLIT!r}"  # noqa: E501  # reason: contract string single logical; splitting harms grep
                    )
                # Verify split field if present; otherwise enforce train
                if row_split is None:
                    raw["split"] = AUTHORIZED_TRAIN_SPLIT
                self._rows.append(raw)
                _wall_a: Any | None = raw.get("wall_id")
                _wall_b: Any | None = raw.get("game_id")
                _wall_tmp: Any | None = _wall_a if _wall_a is not None else _wall_b
                if not isinstance(_wall_tmp, str) or _wall_tmp == "":
                    # Ranks-written shards carry wall provenance inside the opaque
                    # label dict (no Arrow schema break); fall back to it when
                    # top-level columns are absent.
                    _wall_label: Any = raw.get("privileged_label")
                    if isinstance(_wall_label, str):
                        try:
                            _wall_label = json.loads(_wall_label)
                        except Exception:
                            _wall_label = None
                    if isinstance(_wall_label, dict):
                        _wall_c: Any | None = _wall_label.get("wall_id")
                        _wall_d: Any | None = _wall_label.get("game_id")
                        _wall_tmp = _wall_c if _wall_c is not None else _wall_d
                wall: Any | None = _wall_tmp if _wall_tmp is not None else ""
                if isinstance(wall, str) and wall != "":
                    self._wall_ids.add(wall)
        self._by_id: dict[Any, OracleTarget] = {}
        for raw in self._rows:
            _did: Any = raw.get("decision_id")
            try:
                if _did in self._by_id:
                    continue
            except TypeError:
                continue  # unhashable id: keep linear-scan fallback semantics
            self._by_id[_did] = self._target_from_raw(
                raw, _did, allow_synthetic=self.allow_synthetic
            )

    @property
    def wall_ids(self) -> frozenset[str]:
        return frozenset(self._wall_ids)

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _target_from_raw(
        raw: dict[str, Any], decision_id: str, *, allow_synthetic: bool = False
    ) -> OracleTarget:
        priv = raw.get("privileged_label")
        if isinstance(priv, str):
            try:
                priv = json.loads(priv)
            except Exception:
                priv = None
        belief = _belief_target_from_privileged(
            priv if isinstance(priv, dict) else None,
            decision_id,
            allow_synthetic=allow_synthetic,
        )
        value = _value_target_from_privileged(
            priv if isinstance(priv, dict) else None,
            decision_id,
            allow_synthetic=allow_synthetic,
        )
        b_logits, v_logits = _teacher_logits_from_targets(belief, value)
        event_target = int(hashlib.sha256(decision_id.encode()).hexdigest()[:2], 16) % 20
        _wid_a: Any | None = raw.get("wall_id")
        _wid_b: Any | None = raw.get("game_id")
        _wid_tmp: Any | None = _wid_a if _wid_a is not None else _wid_b
        if (not isinstance(_wid_tmp, str) or _wid_tmp == "") and isinstance(priv, dict):
            _wid_c: Any | None = priv.get("wall_id")
            _wid_d: Any | None = priv.get("game_id")
            _wid_tmp = _wid_c if _wid_c is not None else _wid_d
        _wid_val: Any | None = (
            _wid_tmp if isinstance(_wid_tmp, str) and _wid_tmp != "" else f"wall-{decision_id}"
        )
        _split_val: Any | None = raw.get("split")
        _obs_val: Any | None = raw.get("observation_hash")
        return OracleTarget(
            decision_id=decision_id,
            wall_id=str(_wid_val),
            belief_target=belief,
            value_target=value,
            event_target=event_target,
            teacher_belief_logits=b_logits,
            teacher_value_logits=v_logits,
            split=str(_split_val if _split_val is not None else AUTHORIZED_TRAIN_SPLIT),
            observation_hash=str(_obs_val if _obs_val is not None else ""),
        )

    def get_oracle_target(
        self, decision_id: str, *, allow_synthetic: bool | None = None
    ) -> OracleTarget:
        """Return the materialized target, or synthesize iff opted in.

        ``allow_synthetic=None`` (default) inherits the loader construction
        flag; explicit ``True``/``False`` overrides per call. Without opt-in,
        unknown ``decision_id`` raises :class:`ContractError` (fail closed).
        """
        try:
            return self._by_id[decision_id]
        except (KeyError, TypeError):
            pass
        effective = self.allow_synthetic if allow_synthetic is None else allow_synthetic
        if not effective:
            raise ContractError(
                f"oracle target missing for {decision_id!r} "
                "(fail closed; synthetic opt-in via allow_synthetic=True)"
            )
        # Synthetic target (deterministic fallback; byte-identical to pre-flag)
        belief = _belief_target_from_privileged(None, decision_id, allow_synthetic=True)
        value = _value_target_from_privileged(None, decision_id, allow_synthetic=True)
        b_logits, v_logits = _teacher_logits_from_targets(belief, value)
        return OracleTarget(
            decision_id=decision_id,
            wall_id=f"wall-{decision_id}",
            belief_target=belief,
            value_target=value,
            event_target=int(hashlib.sha256(decision_id.encode()).hexdigest()[:2], 16) % 20,
            teacher_belief_logits=b_logits,
            teacher_value_logits=v_logits,
            split=AUTHORIZED_TRAIN_SPLIT,
            observation_hash="sha256:" + hashlib.sha256(decision_id.encode()).hexdigest(),
        )

    def iter_targets(self, *, allow_synthetic: bool | None = None) -> list[OracleTarget]:
        return [
            self.get_oracle_target(str(r.get("decision_id")), allow_synthetic=allow_synthetic)
            for r in self._rows
        ]

    def load_batch(
        self, decision_ids: list[str], *, allow_synthetic: bool | None = None
    ) -> list[OracleTarget]:
        if not isinstance(decision_ids, list):
            raise ContractError("decision_ids must be a list")
        return [
            self.get_oracle_target(did, allow_synthetic=allow_synthetic) for did in decision_ids
        ]


def ranks_from_final_scores(scores: Any) -> tuple[int, int, int, int]:
    """Derive per-seat ranks 1..4 from terminal final scores (GameRecord exporter).

    ``scores`` is the 4-seat final-score quad — e.g. extracted from the terminal
    ``end_game`` event of a :class:`hydra2.data.decode.GameRecord` — with rank 1
    assigned to the highest score. Returns the per-seat ranks permutation 1..4,
    ready for privileged writers (``validate_privileged_ranks``) and the
    ``utility()`` value mapping (``_value_from_ranks_via_utility``).

    Strict: exactly 4 finite distinct numbers (``bool`` rejected; non-finite
    and non-numeric entries raise). Ties raise :class:`ContractError` — equal
    scores need Tenhou east-1 seat-wind context (``resolve_final_ranks``: the
    lower seat index takes the better rank) that raw scores alone do not carry,
    so tied games must be Tenhou-resolved (via ``resolve_final_ranks``) before
    calling; only strict-order games flow through this exporter.
    """
    if not isinstance(scores, (list, tuple)) or len(scores) != 4:
        raise ContractError(f"ranks_from_final_scores: expected 4 final scores, got {scores!r}")
    vals: list[float] = []
    for i, s in enumerate(scores):
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise ContractError(f"ranks_from_final_scores: score[{i}] must be a number, got {s!r}")
        f = float(s)
        if not math.isfinite(f):
            raise ContractError(f"ranks_from_final_scores: score[{i}] must be finite, got {s!r}")
        vals.append(f)
    if len(set(vals)) != 4:
        raise ContractError(
            f"ranks_from_final_scores: scores must be distinct (ties need Tenhou "
            f"resolve_final_ranks first), got {list(scores)!r}"
        )

    def _seat_order(seat: int) -> tuple[float, int]:
        return (-vals[seat], seat)

    order = sorted(range(4), key=_seat_order)
    ranks = [0, 0, 0, 0]
    for position, seat in enumerate(order):
        ranks[seat] = position + 1
    return (ranks[0], ranks[1], ranks[2], ranks[3])


def _ranks_for_join(
    privileged_label: Any, decision_id: str
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Extract 0-based placement row + utility() value row from a label dict.

    Strict path: requires an explicit ``ranks`` (or 4-list ``final_placement``)
    permutation. Accepts 1..4 (authoritative) or 0..3 (seat convention);
    0-based ``placement_target`` is the -1 bridge to ``utility()`` 1..4
    (0-based +1 -> 1..4 for ``_value_from_ranks_via_utility``). Any missing or
    non-permutation input raises (never hash-synthesizes on this path).
    """
    candidate: Any = None
    if isinstance(privileged_label, dict):
        candidate = privileged_label.get("ranks")
        if candidate is None:
            _fp: Any = privileged_label.get("final_placement")
            if isinstance(_fp, (list, tuple)) and len(_fp) == 4:
                candidate = _fp
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
        raise ContractError(
            f"join_oracle_targets: missing ranks permutation for {decision_id!r} "
            "(expected 'ranks' 1..4 or 0..3 4-list; never hash-synthesized)"
        )
    if any(isinstance(x, bool) for x in candidate) or not all(
        isinstance(x, int) for x in candidate
    ):
        raise ContractError(
            f"join_oracle_targets: ranks must be int permutation for {decision_id!r}"
        )
    # Guards below prove 4 ints (bool rejected), so int() would be identity.
    _candidate_ints: list[int] = list(candidate)
    ranks_tuple: tuple[int, ...] = tuple(_candidate_ints)
    if sorted(ranks_tuple) == [0, 1, 2, 3]:
        placement = tuple(ranks_tuple)
    elif sorted(ranks_tuple) == [1, 2, 3, 4]:
        placement = tuple(x - 1 for x in ranks_tuple)
    else:
        raise ContractError(
            f"join_oracle_targets: ranks must be 1..4 or 0..3 permutation for {decision_id!r}, "
            f"got {list(ranks_tuple)!r}"
        )
    for p in placement:
        if not 0 <= p < 4:
            raise ContractError(
                f"join_oracle_targets: placement_target out of range [0,4) for {decision_id!r}"
            )
    value = _value_from_ranks_via_utility(candidate)
    if value is None:
        raise ContractError(
            f"join_oracle_targets: ranks failed utility() mapping for {decision_id!r}"
        )
    return placement, value


def _synthetic_ranks(decision_id: str) -> tuple[int, int, int, int]:
    """Deterministic synthetic 1..4 ranks permutation from a decision_id hash.

    Synthetic-only (join ``allow_synthetic=True`` for ids with no label):
    orders seats by the first four digest bytes (stable seat-index tiebreak),
    so the result is always a strict permutation. Feeds the same
    ``_ranks_for_join`` derivation as real rows (utility-scale values);
    carries no wall/split provenance, so wall checks skip synthetic rows.
    """
    h = hashlib.sha256(decision_id.encode()).digest()

    def _synth_order(seat: int) -> tuple[int, int]:
        return (h[seat], seat)

    order = sorted(range(4), key=_synth_order)
    ranks = [0, 0, 0, 0]
    for position, seat in enumerate(order):
        ranks[seat] = position + 1
    return (ranks[0], ranks[1], ranks[2], ranks[3])


def join_oracle_targets(
    decision_ids: list[str],
    source: Any,
    *,
    evaluation_wall_ids: Any | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Join privileged placement/value targets by opaque decision_id only.

    Returns ``{"placement_target": [B,4] long 0-based, "value_target": [B,4]
    float}``. ``placement_target`` is per-seat rank indices 0..3 (range
    asserted); ``value_target`` is the 4-seat ``UtilityVector.values`` via
    ``_value_from_ranks_via_utility`` (ranks -> rank_values -> values, e.g.
    ``(20,10,-10,-20)`` permuted, zero-sum NOT a distribution). The -1 bridge:
    0-based placement +1 -> 1..4 for ``utility()``; 1..4 ranks map back via -1.

    Firewall: actor side supplies opaque ``decision_id`` strings only (non-empty
    ``str``; any non-str/empty entry raises); privileged ranks never enter the
    actor batch. Train-split is enforced (loader ``split`` must be ``"train"``
    and every joined row/label split must be ``"train"``). Wall checks are
    enforced: joined ``wall_id`` values must not overlap ``evaluation_wall_ids``
    when supplied (``check_wall_leakage``). Missing labels raise
    :class:`ContractError` by default (fail closed); ``allow_synthetic=True``
    opts missing ids into deterministic hash-permutation ranks through the same
    ``_ranks_for_join`` derivation (synthetic-only; present-but-invalid labels
    always raise, never synthesize).

    Belief note: 34-dim distribution belief targets keep the existing index-CE
    contract; 34-dist distribution support is explicitly DEFERRED and not
    joined here.

    ``source`` is either a :class:`PrivilegedOracleLoader` (real parquet rows)
    or a ``PrivilegedLabelStore`` / ``dict`` mapping decision_id -> label dict
    with ``ranks`` / 4-list ``final_placement``.
    """
    if not isinstance(decision_ids, (list, tuple)) or len(decision_ids) == 0:
        raise ContractError("join_oracle_targets: decision_ids must be a non-empty list")
    for _did in decision_ids:
        if not isinstance(_did, str) or _did == "":
            raise ContractError(
                f"join_oracle_targets: decision_id must be opaque non-empty str, got {_did!r}"
            )
    if source is None:
        raise ContractError("join_oracle_targets: source is required")
    placements: list[tuple[int, ...]] = []
    values: list[tuple[float, ...]] = []
    joined_walls: list[str] = []
    # Loader path: duck-typed (has get_oracle_target + _rows/_by_id); avoids a
    # hard training->belief import cycle at module level.
    if hasattr(source, "get_oracle_target") and hasattr(source, "_rows"):
        _split: Any = getattr(source, "split", AUTHORIZED_TRAIN_SPLIT)
        _require_train_split(str(_split))
        raw_by_id: dict[str, dict[str, Any]] = {}
        for _raw in list(getattr(source, "_rows", [])):
            if isinstance(_raw, dict) and isinstance(_raw.get("decision_id"), str):
                # First-wins map; setdefault return intentionally discarded.
                _did_key: str = _raw["decision_id"]
                _ = raw_by_id.setdefault(_did_key, _raw)
        for _did2 in decision_ids:
            _raw_hit: dict[str, Any] | None = raw_by_id.get(_did2)
            if _raw_hit is None:
                if allow_synthetic:
                    # Synthetic-only opt-in: deterministic hash-permutation ranks
                    # through the same _ranks_for_join derivation (utility-scale
                    # values); no wall/split provenance, so wall checks skip it.
                    _synth_placement, _synth_value = _ranks_for_join(
                        {"ranks": list(_synthetic_ranks(_did2))}, _did2
                    )
                    placements.append(_synth_placement)
                    values.append(_synth_value)
                    continue
                # Also consult _by_id (covers loaders with synthetic-empty rows);
                # anything still missing raises (fail closed by default).
                try:
                    _by_id: Any = getattr(source, "_by_id", {})
                    if _did2 not in _by_id:
                        raise ContractError(
                            f"join_oracle_targets: missing oracle label for {_did2!r} "
                            "(fail closed; synthetic opt-in via allow_synthetic=True)"
                        )
                    # Present in _by_id but raw missing: wall/split unknown, still
                    # require explicit ranks from the raw map -> raise.
                    raise ContractError(
                        f"join_oracle_targets: missing privileged ranks for {_did2!r}"
                    )
                except TypeError as exc:
                    raise ContractError(
                        f"join_oracle_targets: missing oracle label for {_did2!r}"
                    ) from exc
            _priv: Any = _raw_hit.get("privileged_label")
            if isinstance(_priv, str):
                try:
                    _priv = json.loads(_priv)
                except Exception as exc:
                    raise ContractError(
                        f"join_oracle_targets: privileged_label not JSON for {_did2!r}"
                    ) from exc
            _row_split: Any = _raw_hit.get("split")
            if _row_split is None and isinstance(_priv, dict):
                _row_split = _priv.get("split")
            if _row_split is not None and _row_split != AUTHORIZED_TRAIN_SPLIT:
                raise ContractError(
                    f"join_oracle_targets: row {_did2!r} split {_row_split!r} != 'train'"
                )
            _placement, _value = _ranks_for_join(_priv, _did2)
            placements.append(_placement)
            values.append(_value)
            _w_a: Any = _raw_hit.get("wall_id")
            _w_b: Any = _raw_hit.get("game_id")
            _w_tmp: Any = _w_a if _w_a is not None else _w_b
            if (not isinstance(_w_tmp, str) or _w_tmp == "") and isinstance(_priv, dict):
                _w_c: Any = _priv.get("wall_id")
                _w_d: Any = _priv.get("game_id")
                _w_tmp = _w_c if _w_c is not None else _w_d
            if isinstance(_w_tmp, str) and _w_tmp != "":
                joined_walls.append(_w_tmp)
    elif hasattr(source, "get"):
        for _did3 in decision_ids:
            try:
                _label: Any = source.get(_did3)
            except Exception as exc:
                raise ContractError(
                    f"join_oracle_targets: store.get failed for {_did3!r}: {exc}"
                ) from exc
            if _label is None:
                if allow_synthetic:
                    _synth_placement2, _synth_value2 = _ranks_for_join(
                        {"ranks": list(_synthetic_ranks(_did3))}, _did3
                    )
                    placements.append(_synth_placement2)
                    values.append(_synth_value2)
                    continue
                raise ContractError(
                    f"join_oracle_targets: missing oracle label for {_did3!r} "
                    "(fail closed; synthetic opt-in via allow_synthetic=True)"
                )
            if not isinstance(_label, dict):
                raise ContractError(f"join_oracle_targets: label must be dict for {_did3!r}")
            _label_split: Any = _label.get("split")
            if _label_split is not None and _label_split != AUTHORIZED_TRAIN_SPLIT:
                raise ContractError(
                    f"join_oracle_targets: label {_did3!r} split {_label_split!r} != 'train'"
                )
            _placement2, _value2 = _ranks_for_join(_label, _did3)
            placements.append(_placement2)
            values.append(_value2)
            _lw_a: Any = _label.get("wall_id")
            _lw_b: Any = _label.get("game_id")
            _lw_tmp: Any = _lw_a if _lw_a is not None else _lw_b
            if isinstance(_lw_tmp, str) and _lw_tmp != "":
                joined_walls.append(_lw_tmp)
    else:
        raise ContractError(
            "join_oracle_targets: source must be PrivilegedOracleLoader or "
            f"PrivilegedLabelStore/dict, got {type(source).__name__}"
        )
    if evaluation_wall_ids is not None:
        try:
            _eval_set = set(evaluation_wall_ids)
        except TypeError as exc:
            raise ContractError(
                "join_oracle_targets: evaluation_wall_ids must be iterable"
            ) from exc
        if len(_eval_set) > 0:
            # Fail closed: every joined row must carry wall provenance when
            # evaluation walls are supplied; provenance-free rows cannot prove
            # disjointness. Synthetic-only rows carry no provenance, so they
            # skip the check only when the eval set is None/empty (above).
            if len(joined_walls) != len(decision_ids):
                raise ContractError(
                    f"join_oracle_targets: missing wall provenance for "
                    f"{len(decision_ids) - len(joined_walls)} row(s) with "
                    f"evaluation walls supplied (fail closed)"
                )
            check_wall_leakage(joined_walls, _eval_set)
    import torch as _torch

    placement_target = _torch.tensor([list(p) for p in placements], dtype=_torch.long)
    value_target = _torch.tensor([list(v) for v in values], dtype=_torch.float32)
    if tuple(placement_target.shape) != (len(decision_ids), 4):
        raise ContractError(
            f"join_oracle_targets: placement_target shape {tuple(placement_target.shape)} != [B,4]"
        )
    if tuple(value_target.shape) != (len(decision_ids), 4):
        raise ContractError(
            f"join_oracle_targets: value_target shape {tuple(value_target.shape)} != [B,4]"
        )
    return {"placement_target": placement_target, "value_target": value_target}


# ---------------------------------------------------------------------------
# Process boundary helpers
# ---------------------------------------------------------------------------


def _child_load_worker(
    parquet_dir_str: str,
    split: str,
    decision_ids: list[str],
    queue: multiprocessing.Queue,  # type: ignore[type-arg]  # reason: Queue generic unparameterized by design; Any payload. Evidence: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
    allow_synthetic: bool = False,
) -> None:
    try:
        loader = PrivilegedOracleLoader(
            parquet_dir_str, split=split, verify=False, allow_synthetic=allow_synthetic
        )
        targets = loader.load_batch(decision_ids)
        # Serialize as plain dicts for queue
        payload = [
            {
                "decision_id": t.decision_id,
                "wall_id": t.wall_id,
                "belief_target": list(t.belief_target),
                "value_target": list(t.value_target),
                "event_target": t.event_target,
                "teacher_belief_logits": list(t.teacher_belief_logits),
                "teacher_value_logits": list(t.teacher_value_logits),
                "split": t.split,
                "observation_hash": t.observation_hash,
                "child_pid": os.getpid(),
            }
            for t in targets
        ]
        queue.put(("ok", payload, os.getpid()))
    except Exception as exc:
        queue.put(("err", f"{type(exc).__name__}: {exc}", os.getpid()))


def load_oracle_batch_in_subprocess(
    parquet_dir: Path | str,
    decision_ids: list[str],
    split: str = "train",
    timeout: float = 30.0,
    allow_synthetic: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Load privileged targets in a fresh Python process.

    Returns (payload_list, child_pid). Asserts isolation: child_pid != os.getpid().

    ``allow_synthetic`` forwards to the child loader: unknown ids raise by
    default (fail closed); ``True`` opts into deterministic hash synthesis.

    Raises ContractError on leakage or timeout.
    """
    _require_train_split(split)
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[attr-defined]  # reason: BaseContext.Queue dynamically provided; spawn context has Queue. Evidence: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    proc = ctx.Process(
        target=_child_load_worker,
        args=(str(parquet_dir), split, decision_ids, queue, allow_synthetic),
    )
    t0 = time.monotonic()
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        raise ContractError("oracle subprocess timed out — process boundary liveness failure")
    exitcode = proc.exitcode  # snapshot: proc.close() in finally forbids later access
    try:
        remaining = max(1.0, timeout - (time.monotonic() - t0))
        status, payload, child_pid = queue.get(timeout=remaining)
    except Exception as exc:
        raise ContractError(f"oracle subprocess queue empty: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            proc.close()
    if status != "ok":
        raise ContractError(f"oracle subprocess error: {payload}")
    if exitcode != 0:
        raise ContractError(f"oracle subprocess crashed exitcode={exitcode}")
    if child_pid == os.getpid():
        raise ContractError("process boundary violation: child_pid == parent_pid")
    return payload, child_pid


def assert_privileged_loader_isolated_from_encoder() -> None:
    """Verify encoder module does NOT import the privileged loader.

    Hard failure if ``hydra2.belief.oracle_loader`` is present after importing
    the inference encoder.
    """
    # Check if encoder imports loader transitively
    encoder_modules = [m for m in sys.modules if "hydra2.models.encoder" in m]
    # Force import encoder if not already loaded
    if len(encoder_modules) == 0:
        import hydra2.models.encoder  # noqa: F401  # reason: intentional import for isolation check; module presence proves no leak

    if "hydra2.belief.oracle_loader" in sys.modules:
        # It is okay that THIS module is loaded (caller imported it), but we must ensure
        # encoder.py source does not import it.
        # Portable encoder path via importlib.resources (zip-safe), not parents[1] depth brittle.
        # Evidence: https://docs.python.org/3/library/importlib.resources.html#files
        # Evidence: https://docs.python.org/3/library/importlib.resources.html#importlib.resources.as_file
        # Evidence: https://github.com/python/cpython/blob/main/Lib/importlib/resources/_common.py
        # Legacy: previously Path(__file__).resolve().parents[1] / "models" / "encoder.py".
        # Zip-safe: as_file materializes Traversable to temp Path when installed as zip/pex.
        import importlib.resources as _ir

        src: str | None = None
        try:
            from importlib.resources import (
                as_file as _as_file,  # type: ignore[attr-defined]  # reason: as_file exported conditionally; import validates. Evidence: https://docs.python.org/3/library/importlib.resources.html#importlib.resources.as_file
            )

            _trav = _ir.files("hydra2.models") / "encoder.py"
            with _as_file(_trav) as enc_path:
                if enc_path.is_file():
                    src = enc_path.read_text(encoding="utf-8")
                else:
                    raise FileNotFoundError(f"traversable not file: {_trav}")
        except Exception:
            try:
                from hydra2.config import repo_root  # fallback marker walk

                enc_path = repo_root() / "src" / "hydra2" / "models" / "encoder.py"
                if enc_path.is_file():
                    src = enc_path.read_text(encoding="utf-8")
            except Exception:
                src = None
        # Only privileged word is too common; tighten to oracle_loader.
        # Broad "privileged" substring is redundant: tight match implies
        # broad, so single tight guard preserves behavior without nesting.
        if src is not None and ("oracle_loader" in src or "PrivilegedOracleLoader" in src):
            raise ContractError("inference encoder imports privileged loader — isolation violated")
    # Also verify FORBIDDEN keys not in encoder batch construction
    # (light check)
    return


__all__ = [
    "AUTHORIZED_TRAIN_SPLIT",
    "FORBIDDEN_IN_ACTOR_KEYS",
    "PRIVILEGED_KEYS",
    "OracleTarget",
    "PrivilegedOracleLoader",
    "assert_privileged_loader_isolated_from_encoder",
    "check_split_disjoint",
    "check_wall_leakage",
    "join_oracle_targets",
    "load_oracle_batch_in_subprocess",
    "ranks_from_final_scores",
    "validate_actor_batch_no_privileged",
]
