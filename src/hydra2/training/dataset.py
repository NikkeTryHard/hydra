"""WP-05B authoritative data integration: synthetic parquet dataset.

Re-export facade over the split modules: :mod:`hydra2.training.dataset_parse`
(actor-observation JSON rebuild, validating parse, live-or-parse resolve,
shard discovery and verification), :mod:`hydra2.training.dataset_encode`
(synthetic tensorize stand-in, real encoder fold, shared
``encode_observation_rows`` path), and
:mod:`hydra2.training.dataset_store` (stratified sampler,
:class:`AuthoritativeParquetDataset`, cursor/epoch resume). Import from
this path; it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.training.dataset_encode import (
    _REAL_COUNT_KEYS as _REAL_COUNT_KEYS,
)
from hydra2.training.dataset_encode import (
    _lexicographic_hash as _lexicographic_hash,
)
from hydra2.training.dataset_encode import (
    _real_features_from_encoder_batch as _real_features_from_encoder_batch,
)
from hydra2.training.dataset_encode import (
    _require_action_width as _require_action_width,
)
from hydra2.training.dataset_encode import (
    encode_observation_rows as encode_observation_rows,
)
from hydra2.training.dataset_encode import (
    tensorize_actor_row as tensorize_actor_row,
)
from hydra2.training.dataset_parse import (
    _actor_observation_from_json_dict as _actor_observation_from_json_dict,
)
from hydra2.training.dataset_parse import (
    _delta_from_json as _delta_from_json,
)
from hydra2.training.dataset_parse import (
    _envelope_from_json as _envelope_from_json,
)
from hydra2.training.dataset_parse import (
    _parse_actor_observation as _parse_actor_observation,
)
from hydra2.training.dataset_parse import (
    _payload_from_json as _payload_from_json,
)
from hydra2.training.dataset_parse import (
    _require_actor_parquet_dir as _require_actor_parquet_dir,
)
from hydra2.training.dataset_parse import (
    _resolve_live_or_parse as _resolve_live_or_parse,
)
from hydra2.training.dataset_parse import (
    _verify_shards as _verify_shards,
)
from hydra2.training.dataset_parse import (
    _visible_meld_from_json as _visible_meld_from_json,
)
from hydra2.training.dataset_store import (
    DEFAULT_STRATIFIED_RATIOS as DEFAULT_STRATIFIED_RATIOS,
)
from hydra2.training.dataset_store import (
    RARE_ACTION_KINDS as RARE_ACTION_KINDS,
)
from hydra2.training.dataset_store import (
    AuthoritativeParquetDataset as AuthoritativeParquetDataset,
)
from hydra2.training.dataset_store import (
    SamplerState as SamplerState,
)
from hydra2.training.dataset_store import (
    _row_action_kind as _row_action_kind,
)
from hydra2.training.dataset_store import (
    _validate_kind_by_id as _validate_kind_by_id,
)
from hydra2.training.dataset_store import (
    _validate_sampling_ratios as _validate_sampling_ratios,
)
from hydra2.training.dataset_store import (
    build_stratified_order as build_stratified_order,
)

__all__ = [
    "DEFAULT_STRATIFIED_RATIOS",
    "RARE_ACTION_KINDS",
    "AuthoritativeParquetDataset",
    "SamplerState",
    "build_stratified_order",
    "encode_observation_rows",
    "tensorize_actor_row",
]
