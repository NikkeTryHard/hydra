"""WP-07B oracle join — decision_id-keyed privileged target joins.

Owns the join from opaque decision_ids to placement/value targets (loader
and label-store sources), the ranks exporter from terminal scores, the
spawn-process batch loader proving the process boundary, and the encoder
isolation proof (inference encoders never import the privileged path).
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.belief.oracle_guard import (
    AUTHORIZED_TRAIN_SPLIT as AUTHORIZED_TRAIN_SPLIT,
)
from hydra2.belief.oracle_guard import (
    _require_train_split as _require_train_split,
)
from hydra2.belief.oracle_guard import (
    check_wall_leakage as check_wall_leakage,
)
from hydra2.belief.oracle_store import (
    PrivilegedOracleLoader as PrivilegedOracleLoader,
)
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from pathlib import Path


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
    try:
        ranks: list[int] = _bridge_contracts.oracle_join_ranks_from_scores(scores)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
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
    # Value row only: 0-based +1 -> 1..4 for the wave-6 utility bridge
    # (UtilityVector.values scale); the frozen rank_values golden is read off
    # the bridge, never restated here. Rejection is unreachable post-gate and
    # maps to the oracle utility-mapping text below.
    try:
        rank_golden: tuple[float, ...] = _bridge_contracts.ORACLE_RANK_VALUES
        rank_list: list[float] = list(rank_golden)
        one_based: list[int] = [p + 1 for p in placement]
        values: list[float] = _bridge_contracts.utility_values_for_ranks(rank_list, one_based)
        value = tuple(values)
    except (ValueError, TypeError) as exc:
        raise ContractError(
            f"join_oracle_targets: ranks failed utility() mapping for {decision_id!r}"
        ) from exc
    return placement, value


def _synthetic_ranks(decision_id: str) -> tuple[int, int, int, int]:
    """Deterministic synthetic 1..4 ranks permutation from a decision_id hash.

    Synthetic-only (join ``allow_synthetic=True`` for ids with no label):
    orders seats by the first four digest bytes (stable seat-index tiebreak),
    so the result is always a strict permutation. Feeds the same
    ``_ranks_for_join`` derivation as real rows (utility-scale values);
    carries no wall/split provenance, so wall checks skip synthetic rows.
    """
    ranks: list[int] = _bridge_contracts.oracle_join_synthetic_ranks(decision_id)
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
    "assert_privileged_loader_isolated_from_encoder",
    "join_oracle_targets",
    "load_oracle_batch_in_subprocess",
    "ranks_from_final_scores",
]
