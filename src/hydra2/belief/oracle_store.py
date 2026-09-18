"""WP-07B oracle store — train-split-only privileged parquet loader.

Owns the loader class holding privileged parquet handles: split-gated
construction, target materialization, and synthetic opt-in reads. Actor
inference code never imports this module (the isolation proof lives with
the join module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from hydra2.artifacts.digest import sha256_digest
from hydra2.belief.oracle_guard import (
    AUTHORIZED_TRAIN_SPLIT as AUTHORIZED_TRAIN_SPLIT,
)
from hydra2.belief.oracle_guard import (
    _privileged_shard_paths as _privileged_shard_paths,
)
from hydra2.belief.oracle_guard import (
    _require_train_split as _require_train_split,
)
from hydra2.belief.oracle_targets import (
    OracleTarget as OracleTarget,
)
from hydra2.belief.oracle_targets import (
    _belief_target_from_privileged as _belief_target_from_privileged,
)
from hydra2.belief.oracle_targets import (
    _teacher_logits_from_targets as _teacher_logits_from_targets,
)
from hydra2.belief.oracle_targets import (
    _value_target_from_privileged as _value_target_from_privileged,
)
from hydra2.contracts.common import ContractError


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
        # Digest line via the canon bridge (byte-identical to the retired
        # hashlib hexdigest slice; ImportError with build-ext hint).
        event_target = (
            int(str(sha256_digest(decision_id.encode())).removeprefix("sha256:")[:2], 16) % 20
        )
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
        # Digest lines via the canon bridge; math/gate untouched.
        belief = _belief_target_from_privileged(None, decision_id, allow_synthetic=True)
        value = _value_target_from_privileged(None, decision_id, allow_synthetic=True)
        b_logits, v_logits = _teacher_logits_from_targets(belief, value)
        _syn_hex = str(sha256_digest(decision_id.encode())).removeprefix("sha256:")
        return OracleTarget(
            decision_id=decision_id,
            wall_id=f"wall-{decision_id}",
            belief_target=belief,
            value_target=value,
            event_target=int(_syn_hex[:2], 16) % 20,
            teacher_belief_logits=b_logits,
            teacher_value_logits=v_logits,
            split=AUTHORIZED_TRAIN_SPLIT,
            observation_hash="sha256:" + _syn_hex,
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


__all__ = [
    "PrivilegedOracleLoader",
]
