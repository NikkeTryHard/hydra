//! Typed MJAI event representation for zero-allocation logging.
//!
//! Replaces `serde_json::Value` construction with stack-allocated enums.
//! JSON serialization only happens on demand (replay export), not during play.

/// Typed MJAI game event. Stack-allocated, no heap.
///
/// Serialize to JSON only when needed (replay export).
#[derive(Debug, Clone)]
pub enum MjaiEvent {
    /// Game start marker.
    StartGame,
    /// Round start with initial deal.
    StartKyoku {
        bakaze: u8,
        dora_marker: u8,
        kyoku: u8,
        honba: u8,
        kyotaku: u32,
        oya: u8,
        scores: [i32; 4],
        tehais: [[u8; 14]; 4],
        tehai_lens: [u8; 4],
    },
    /// Player draws a tile.
    Tsumo { actor: u8, pai: u8 },
    /// Player discards a tile.
    Dahai {
        actor: u8,
        pai: u8,
        tsumogiri: bool,
    },
    /// Player declares riichi.
    Reach { actor: u8 },
    /// Riichi deposit accepted.
    ReachAccepted { actor: u8 },
    /// Chi (sequence) call.
    Chi {
        actor: u8,
        target: u8,
        pai: u8,
        consumed: [u8; 2],
    },
    /// Pon (triplet) call.
    Pon {
        actor: u8,
        target: u8,
        pai: u8,
        consumed: [u8; 2],
    },
    /// Closed kan (ankan).
    Ankan { actor: u8, consumed: [u8; 4] },

    /// Added kan (kakan).
    Kakan {
        actor: u8,
        pai: u8,
        consumed: [u8; 3],
    },
    /// Open kan (daiminkan).
    Daiminkan {
        actor: u8,
        target: u8,
        pai: u8,
        consumed: [u8; 3],
    },
    /// Win declaration.
    Hora { actor: u8, target: u8, pai: u8 },
    /// Exhaustive draw.
    Ryukyoku,
    /// Round end marker.
    EndKyoku,
    /// Game end marker.
    EndGame,
}

/// Zero-cost MJAI event logging macro.
///
/// Arguments are NOT evaluated when logging is disabled at runtime.
/// When the `mjai-logging` feature is off, the macro compiles to nothing.
#[cfg(feature = "mjai-logging")]
macro_rules! mjai_event {
    ($game:expr, $event:expr) => {
        if !$game.skip_mjai_logging {
            $game.mjai_events.push($event);
        }
    };
}

#[cfg(not(feature = "mjai-logging"))]
macro_rules! mjai_event {
    ($game:expr, $event:expr) => {};
}
