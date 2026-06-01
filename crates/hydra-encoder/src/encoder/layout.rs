use hydra_runtime_types::tile::NUM_TILE_TYPES;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Baseline observation channels (public + safety).
pub const BASELINE_CHANNELS: usize = 85;

/// Group C search/belief context channels.
pub const SEARCH_CONTEXT_CHANNELS: usize = 65;

/// Group D Hand-EV channels.
pub const HAND_EV_CHANNELS: usize = 42;

/// First Group C search/belief channel.
pub const SEARCH_CHANNEL_START: usize = BASELINE_CHANNELS;

/// First Group C belief-field channel.
pub const SEARCH_BELIEF_CHANNEL_START: usize = SEARCH_CHANNEL_START;

/// First Group D Hand-EV channel.
pub const HAND_EV_CHANNEL_START: usize = SEARCH_CHANNEL_START + SEARCH_CONTEXT_CHANNELS;

/// Group C discard-level delta-Q channel.
pub const SEARCH_DELTA_Q_CHANNEL: usize = SEARCH_CHANNEL_START + 22;

/// Group C mixture-entropy scalar channel.
pub const SEARCH_MIXTURE_ENTROPY_CHANNEL: usize = SEARCH_CHANNEL_START + 20;

/// Group C mixture-ESS scalar channel.
pub const SEARCH_MIXTURE_ESS_CHANNEL: usize = SEARCH_CHANNEL_START + 21;

/// First Group C opponent-risk channel.
pub const SEARCH_RISK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 23;

/// First Group C opponent-stress channel.
pub const SEARCH_STRESS_CHANNEL_START: usize = SEARCH_CHANNEL_START + 26;

/// First Group C presence-mask channel.
pub const SEARCH_MASK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 29;

/// Final Group D Hand-EV presence-mask channel.
pub const HAND_EV_MASK_CHANNEL: usize = HAND_EV_CHANNEL_START + HAND_EV_CHANNELS - 1;

/// Total observation channels.
pub const NUM_CHANNELS: usize = BASELINE_CHANNELS + SEARCH_CONTEXT_CHANNELS + HAND_EV_CHANNELS;

/// Tiles per channel (one per tile type).
pub const NUM_TILES: usize = NUM_TILE_TYPES; // 34

/// Total elements in the flat observation buffer.
pub const OBS_SIZE: usize = NUM_CHANNELS * NUM_TILES;

// -- Channel group starts --

pub(super) const CH_HAND: usize = 0; // 0..3   (4 channels)
pub(super) const CH_OPEN_MELD: usize = 4; // 4..7   (4 channels)
pub(super) const CH_DRAWN: usize = 8; // 8      (1 channel)
pub(super) const CH_SHANTEN_MASK: usize = 9; // 9..10  (2 channels)
pub(super) const CH_DISCARDS: usize = 11; // 11..22 (12 channels: 3 per player)
pub(super) const CH_MELDS: usize = 23; // 23..34 (12 channels: 3 per player)
pub(super) const CH_DORA: usize = 35; // 35..39 (5 channels)
pub(super) const CH_AKA: usize = 40; // 40..42 (3 channels)
pub(super) const CH_META: usize = 43; // 43..61 (19 channels)
pub(super) const CH_SAFETY: usize = 62; // 62..84 (23 channels)
pub(super) const CH_SEARCH: usize = SEARCH_CHANNEL_START; // 85..149 (65 channels)
pub(super) const CH_HAND_EV: usize = HAND_EV_CHANNEL_START; // 150..191 (42 channels)

pub(super) const SEARCH_BELIEF_CHANNELS: usize = 16;
pub(super) const SEARCH_MIXTURE_WEIGHT_CHANNELS: usize = 4;
pub(super) const SEARCH_RISK_CHANNELS: usize = 3;
pub(super) const SEARCH_STRESS_CHANNELS: usize = 3;
pub(super) const SEARCH_MASK_CHANNELS: usize = 4;
pub(super) const SEARCH_RESERVED_CHANNELS: usize = 32;

pub(super) const CH_SEARCH_BELIEF: usize = CH_SEARCH; // 85..100
pub(super) const CH_SEARCH_MIXTURE_WEIGHT: usize = CH_SEARCH_BELIEF + SEARCH_BELIEF_CHANNELS; // 101..104
pub(super) const CH_SEARCH_MIXTURE_ENTROPY: usize =
    CH_SEARCH_MIXTURE_WEIGHT + SEARCH_MIXTURE_WEIGHT_CHANNELS; // 105
pub(super) const CH_SEARCH_MIXTURE_ESS: usize = CH_SEARCH_MIXTURE_ENTROPY + 1; // 106
pub(super) const CH_SEARCH_DELTA_Q: usize = CH_SEARCH_MIXTURE_ESS + 1; // 107
pub(super) const CH_SEARCH_RISK: usize = CH_SEARCH_DELTA_Q + 1; // 108..110
pub(super) const CH_SEARCH_STRESS: usize = CH_SEARCH_RISK + SEARCH_RISK_CHANNELS; // 111..113
pub(super) const CH_SEARCH_MASKS: usize = CH_SEARCH_STRESS + SEARCH_STRESS_CHANNELS; // 114..117
pub(super) const CH_SEARCH_RESERVED: usize = CH_SEARCH_MASKS + SEARCH_MASK_CHANNELS; // 118..149

pub(super) const CH_HAND_EV_TENPAI: usize = CH_HAND_EV; // 150..152
pub(super) const CH_HAND_EV_WIN: usize = CH_HAND_EV_TENPAI + 3; // 153..155
pub(super) const CH_HAND_EV_SCORE: usize = CH_HAND_EV_WIN + 3; // 156
pub(super) const CH_HAND_EV_UKEIRE: usize = CH_HAND_EV_SCORE + 1; // 157..190
pub(super) const CH_HAND_EV_MASK: usize = CH_HAND_EV_UKEIRE + NUM_TILES; // 191

/// Number of players at the table.
pub(super) const NUM_PLAYERS: usize = 4;
