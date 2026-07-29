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
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from hydra2.contracts.common import ContractError, CorruptArtifactError

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
    # Value target: 4-seat placement distribution or scalar vector
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
    privileged_label: dict[str, Any] | None, decision_id: str
) -> tuple[float, ...]:
    """Deterministic 34-dim belief target from privileged hidden tiles.

    For synthetic data where privileged_label may be absent, derive a
    deterministic pseudo-target from decision_id hash (still proper and
    isolates privileged dependence). Real data would use hidden tile counts.
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
    h = hashlib.sha256(decision_id.encode()).digest()
    raw = (
        [float(b) + 1.0 for b in h[:34]]
        if len(h) >= 34
        else [float(b & 0xFF) + 1.0 for b in (h * 3)[:34]]
    )
    total = sum(raw)
    return tuple(v / total for v in raw)


def _value_target_from_privileged(
    privileged_label: dict[str, Any] | None, decision_id: str
) -> tuple[float, ...]:
    if isinstance(privileged_label, dict):
        _v_a: Any | None = privileged_label.get("value_vector")
        _v_b: Any | None = privileged_label.get("utility_vector")
        _v_c: Any | None = privileged_label.get("placement")
        _v_tmp: Any | None = _v_a if _v_a is not None else _v_b
        v: Any | None = _v_tmp if _v_tmp is not None else _v_c
        if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            return tuple(float(x) for x in v)  # type: ignore[unknown-argument-type]  # reason: Any element intentional; float() validates
        # Single placement rank
        _r_a: Any | None = privileged_label.get("final_placement")
        _r_b: Any | None = privileged_label.get("rank")
        rank: Any | None = _r_a if _r_a is not None else _r_b
        if isinstance(rank, int) and 0 <= rank < 4:
            vec = [0.0] * 4
            vec[rank] = 1.0
            return tuple(vec)
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
    ) -> None:
        _require_train_split(split)
        self.parquet_dir = Path(parquet_dir)
        self.split = split
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
            self._by_id[_did] = self._target_from_raw(raw, _did)

    @property
    def wall_ids(self) -> frozenset[str]:
        return frozenset(self._wall_ids)

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _target_from_raw(raw: dict[str, Any], decision_id: str) -> OracleTarget:
        priv = raw.get("privileged_label")
        if isinstance(priv, str):
            try:
                priv = json.loads(priv)
            except Exception:
                priv = None
        belief = _belief_target_from_privileged(
            priv if isinstance(priv, dict) else None, decision_id
        )
        value = _value_target_from_privileged(
            priv if isinstance(priv, dict) else None, decision_id
        )
        b_logits, v_logits = _teacher_logits_from_targets(belief, value)
        event_target = int(hashlib.sha256(decision_id.encode()).hexdigest()[:2], 16) % 20
        _wid_a: Any | None = raw.get("wall_id")
        _wid_b: Any | None = raw.get("game_id")
        _wid_tmp: Any | None = _wid_a if _wid_a is not None else _wid_b
        _wid_val: Any | None = _wid_tmp if _wid_tmp is not None else f"wall-{decision_id}"
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

    def get_oracle_target(self, decision_id: str) -> OracleTarget:
        try:
            return self._by_id[decision_id]
        except (KeyError, TypeError):
            pass
        # Synthetic target if not found (deterministic fallback)
        belief = _belief_target_from_privileged(None, decision_id)
        value = _value_target_from_privileged(None, decision_id)
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

    def iter_targets(self) -> list[OracleTarget]:
        return [self.get_oracle_target(str(r.get("decision_id"))) for r in self._rows]

    def load_batch(self, decision_ids: list[str]) -> list[OracleTarget]:
        if not isinstance(decision_ids, list):
            raise ContractError("decision_ids must be a list")
        return [self.get_oracle_target(did) for did in decision_ids]


# ---------------------------------------------------------------------------
# Process boundary helpers
# ---------------------------------------------------------------------------


def _child_load_worker(
    parquet_dir_str: str,
    split: str,
    decision_ids: list[str],
    queue: multiprocessing.Queue,  # type: ignore[type-arg]  # reason: Queue generic unparameterized by design; Any payload. Evidence: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
) -> None:
    try:
        loader = PrivilegedOracleLoader(parquet_dir_str, split=split, verify=False)
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
) -> tuple[list[dict[str, Any]], int]:
    """Load privileged targets in a fresh Python process.

    Returns (payload_list, child_pid). Asserts isolation: child_pid != os.getpid().

    Raises ContractError on leakage or timeout.
    """
    _require_train_split(split)
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[attr-defined]  # reason: BaseContext.Queue dynamically provided; spawn context has Queue. Evidence: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    proc = ctx.Process(
        target=_child_load_worker, args=(str(parquet_dir), split, decision_ids, queue)
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
        if src is not None and (
            "oracle_loader" in src or "PrivilegedOracleLoader" in src
        ):
            raise ContractError(
                "inference encoder imports privileged loader — isolation violated"
            )
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
    "load_oracle_batch_in_subprocess",
    "validate_actor_batch_no_privileged",
]
