//! hydra-shard: cold/offline pipeline + parity (NO hot-path import).
//!
//! Legacy homes (module-path remaps `hydra2_row → parity`,
//! `decision_ids → ids`):
//! - `decisions` / `stream` / `engine` / `tile` / `mjai_event`: legacy replay
//!   pipeline (engine.rs promotes to feed::walk; the S1 wall scan is
//!   superseded by feed::ingest `frame_spans`).
//! - `parity`: row + privileged + provenance cold-collate surface
//!   (hydra2_row + privileged + provenance concatenated).
//! - `ids`: positional decision/round id strings (hot path uses numeric keys).
//! - `replay`: frozen `replay_game_text` orchestration entry points.
//!
//! Adds compact/mmap/collate/full-26/parquet-join here (cold only).
//! FORBIDS: pyo3. The root facade re-exports this API.

pub mod collate;
pub mod compact;
pub mod decisions;
pub mod digest;
pub mod engine;
pub mod full26;
pub mod ids;
pub mod mjai_event;
pub mod parity;
pub mod parquet_join;
pub mod reader;
pub mod replay;
pub mod stream;
pub mod tile;
pub mod writer;

pub use decisions::{
    ActionTable, WalkReject, WalkerMode, count_row_decisions, walk_game, walk_game_with_version,
};
pub use engine::{
    DrawOffer, EngineDesync, EngineVersion, KAN_MIN_WALL, LIVE_WALL_BASE, RIICHI_MIN_SCORE,
    RIICHI_MIN_WALL, ResponderView, SeatView, draw_offer, kyushu_first_draw, tenpai_discards,
    window_open, win_shape_14,
};
pub use parity::{
    ACTOR_FIELDS, BAKED_RULES_MANIFEST, PINNED_ACTION_TABLE_HASH, PINNED_ADAPTER_HASH,
    PINNED_RULES_HASH, PrivilegedRow, RowProvenance, RulesInfo, SIM_DERIVATION_MARK,
    WALLED_PROJECTION, WALL_LESS_PROJECTION, ChosenAction, Quarantine, ReplayRow, adapter_hash,
    canonical_json_bytes, check_privileged_label, derivation_hash_for, derivation_hash_for_walled,
    expand_privileged_rows, final_scores, load_baked_rules, load_rules_manifest, observation_hash_for,
    ranks_from_final_scores, schedule_id_for, validate_privileged_ranks, wall_schedule_digest,
};
pub use replay::{replay_game_text, replay_game_text_with_version};
pub use stream::{GameReject, ParsedGame, parse_game};
pub use collate::{CollatedBatch, CollateScratch, G2Mismatch, verify_hot_subset};
pub use compact::{
    BitReader, BitWriter, CompactError, LEGAL_BIT_BYTES, LEGAL_BITS, LEGAL_PACKED_BYTES,
    pack_hist_mask, pack_legal_bits, unpack_hist_mask, unpack_legal_bits,
};
pub use digest::{PayloadHasher, raw_hex_prefixed, sha256_hex};
pub use full26::{
    COMPACT_ACTOR_BLOCK, COMPACT_PREFIX, COMPACT_SIDECAR, COMPACT_T_BUCKETS, ColdSidecar,
    Full26Error, HotRow, HotRowIter, FULL26_FIELDS, compact_mask_bytes, compact_row_bytes,
    decode_row_into_planes, dense_full26_row_bytes, encode_compact_row, full26_plane,
    full26_plane_row_bytes, hot_row_at, off_actor, off_hist_mask, off_legal, off_sidecar,
};
pub use ids::{decision_id, decision_ids_for_game, parse_decision_id, parse_round_id, round_id, round_ids_for_game};
pub use parquet_join::{
    ParquetJoinError, PrivilegedJoin, join_privileged, parse_privileged_label,
    read_privileged_parquet,
};
pub use reader::{HEADER_LEN, SHARD_MAGIC, SHARD_VERSION, ShardHeader, ShardReadError, ShardReader};
pub use writer::{FinishedShard, ShardWriteError, ShardWriter};
