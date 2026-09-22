/// Tile types for winds/honors (tile id `/ 4`).
pub const TILE_E: u8 = 27;
/// South wind type.
pub const TILE_S: u8 = 28;
/// West wind type.
pub const TILE_W: u8 = 29;
/// North wind type.
pub const TILE_N: u8 = 30;
/// White dragon type.
pub const TILE_HAKU: u8 = 31;
/// Green dragon type.
pub const TILE_HATSU: u8 = 32;
/// Red dragon type.
pub const TILE_CHUN: u8 = 33;

/// Single yakuman in han units.
pub const HAN_YAKUMAN: u8 = 13;

pub(crate) fn is_terminal_type(t: u8) -> bool {
    matches!(t, 0 | 8 | 9 | 17 | 18 | 26)
}

pub(crate) fn is_honor(t: u8) -> bool {
    t >= 27
}

pub(crate) fn is_simple(t: u8) -> bool {
    t < 27 && !is_terminal_type(t)
}

/// How the winning tile was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinKind {
    /// Self-draw.
    Tsumo,
    /// Discard claim.
    Ron,
}

/// Claimed wait shape (cold caller supplies it; the hot walk never derives it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitKind {
    /// Open-ended (no fu).
    Ryanmen,
    /// Closed (middle) wait (+2 fu).
    Kanchan,
    /// Edge (3 on 12 / 7 on 89) wait (+2 fu).
    Penchan,
    /// Double-pair wait (no fu).
    Shanpon,
    /// Single wait on the pair (+2 fu).
    Tanki,
}

/// Open meld kind (closed quads ride here too: `Ankan` counts closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeldKind {
    /// Open sequence.
    Chi,
    /// Open triplet.
    Pon,
    /// Open quad (claimed).
    Daiminkan,
    /// Closed quad (counts closed for sanankou/menzen).
    Ankan,
    /// Added quad (open).
    Kakan,
}

/// One meld outside the concealed takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenMeld {
    /// Meld kind.
    pub kind: MeldKind,
    /// Tile type (`0..34`) of the meld's base (sequence start / triplet type).
    pub tile_type: u8,
}

impl OpenMeld {
    /// Takes bound in this meld (3 for chi/pon, 4 for quads).
    pub const fn takes(self) -> u8 {
        match self.kind {
            MeldKind::Chi | MeldKind::Pon => 3,
            MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => 4,
        }
    }

    /// True for concealed-equivalent melds (ankan only).
    pub const fn is_closed(self) -> bool {
        matches!(self.kind, MeldKind::Ankan)
    }
}

/// Terminal-hora recompute input (cold-assembled; all takes accounted:
/// `concealed.sum() + open takes == 14`, winning tile INCLUDED in concealed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoraInput {
    /// Closed takes by type, winning tile included.
    pub concealed: [u8; 34],
    /// Winning tile type (`0..34`, present in `concealed`).
    pub win_type: u8,
    /// Tsumo or ron.
    pub win: WinKind,
    /// Ron target seat (required for ron, absent for tsumo).
    pub ron_target: Option<u8>,
    /// Claimed wait shape.
    pub wait: WaitKind,
    /// Melds outside the concealed takes.
    pub open: Vec<OpenMeld>,
    /// Riichi declared.
    pub riichi: bool,
    /// Double riichi declared.
    pub double_riichi: bool,
    /// Ippatsu (one-shot).
    pub ippatsu: bool,
    /// Total dora han (indicators + red fives + ura, cold-summed).
    pub dora: u8,
    /// Win on a rinshan replacement (tsumo only).
    pub rinshan: bool,
    /// Win on a kan rob (ron only).
    pub chankan: bool,
    /// Last-tile win, self-draw (tsumo only).
    pub haitei: bool,
    /// Last-tile win, discard claim (ron only).
    pub houtei: bool,
    /// Winner's seat wind (`27..=30`).
    pub seat_wind: u8,
    /// Round wind (`27..=30`).
    pub round_wind: u8,
    /// Winner is dealer (payment side).
    pub dealer: bool,
    /// Dealer seat (`0..=3`): tsumo distribution side; reconcile checks
    /// `(winner == dealer_seat) == dealer` fail-loud.
    pub dealer_seat: u8,
}

/// Scored han/fu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HanFu {
    /// Han (13/26/39 = single/double/triple yakuman; 13 = counted).
    pub han: u8,
    /// Fu (25 for chiitoi/kokushi; payment ignores fu at mangan+).
    pub fu: u16,
}

/// Score payment in points (caller reads the side it needs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Payment {
    /// Ron receipt (loser pays).
    pub ron: u32,
    /// Tsumo receipt from each non-dealer.
    pub tsumo_ko: u32,
    /// Tsumo receipt from the dealer.
    pub tsumo_oya: u32,
}

/// Recompute outcome: scored, or manual review (fail-loud, never a guess).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreOutcome {
    /// Fully classified hand.
    Known {
        /// Scored han/fu (best interpretation by winner receipt).
        han_fu: HanFu,
        /// Points for the winner's side.
        payment: Payment,
    },
    /// Unclassifiable here (bad input, no yaku found, unimplemented shape):
    /// cold review, never a hot verdict.
    NeedsManual {
        /// Stable machine-readable cause (`bad-input` / `no-yaku` / `exotic`).
        why: &'static str,
    },
}

/// G4 reconciliation outcome (report-only; `Mismatch` goes to cold review,
/// never into hot accept/reject).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reconcile {
    /// Recomputed paymentTransfer equals the logged `deltas`.
    Match {
        /// Scored han.
        han: u8,
        /// Scored fu.
        fu: u16,
    },
    /// Logged `deltas` differ from every interpretation: cold review.
    Mismatch {
        /// Scored han.
        han: u8,
        /// Scored fu.
        fu: u16,
        /// Expected score movement (winner-indexed `[s0, s1, s2, s3]`).
        expected: [i32; 4],
    },
    /// Hand needs manual review (see [`ScoreOutcome::NeedsManual`]).
    NeedsManual {
        /// Stable machine-readable cause.
        why: &'static str,
    },
}

fn ceil100(v: u32) -> u32 {
    v.div_ceil(100) * 100
}

fn base_points(han: u8, fu: u16) -> u32 {
    if han >= 39 {
        24000
    } else if han >= 26 {
        16000
    } else if han >= HAN_YAKUMAN {
        8000
    } else if han >= 11 {
        6000
    } else if han >= 8 {
        4000
    } else if han >= 6 {
        3000
    } else if han >= 5 {
        2000
    } else {
        let b = u32::from(fu) * (1u32 << (u32::from(han) + 2));
        b.min(2000)
    }
}

/// Standard payment table for `(han, fu)` (fu ignored at mangan+).
pub fn hora_payment(han: u8, fu: u16, dealer: bool, tsumo: bool) -> Payment {
    let base = base_points(han, fu);
    if tsumo {
        let ko = ceil100(base);
        let oya = ceil100(base * 2);
        Payment {
            ron: 0,
            tsumo_ko: ko,
            tsumo_oya: oya,
        }
    } else if dealer {
        Payment {
            ron: ceil100(base * 6),
            tsumo_ko: 0,
            tsumo_oya: 0,
        }
    } else {
        Payment {
            ron: ceil100(base * 4),
            tsumo_ko: 0,
            tsumo_oya: 0,
        }
    }
}
