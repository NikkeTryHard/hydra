//! Current-hand feature inputs; no artifact identity or serialized state.

pub(crate) const HISTORY_EVENT_CAP: usize = 256;

#[derive(Clone, Copy)]
pub(crate) enum Visibility {
    Public,
    ActorPrivate,
    #[cfg(test)]
    ServerPrivate,
}

pub(crate) struct TypedEvent {
    pub kind: &'static str,
    pub actor: Option<u8>,
    pub visibility: Visibility,
    pub tile: Option<u8>,
    pub action_id: Option<u32>,
    pub source_seat: Option<u8>,
    pub consumed_tiles: Vec<u8>,
}

impl TypedEvent {
    pub fn public(kind: &'static str, actor: Option<u8>) -> Self {
        Self {
            kind,
            actor,
            visibility: Visibility::Public,
            tile: None,
            action_id: None,
            source_seat: None,
            consumed_tiles: Vec::new(),
        }
    }

    pub fn visible_to(&self, actor: u8) -> bool {
        match self.visibility {
            Visibility::Public => true,
            Visibility::ActorPrivate => self.actor == Some(actor),
            #[cfg(test)]
            Visibility::ServerPrivate => false,
        }
    }
}

pub(crate) type EventArena = Vec<TypedEvent>;

#[derive(Clone)]
pub(crate) struct Observation {
    pub actor: u8,
    pub round_index: u64,
    pub round_wind: i64,
    pub hand_number: u32,
    pub seat_winds: [i64; 4],
    pub honba: u32,
    pub riichi_sticks: u32,
    pub dealer: u8,
    pub scores: [i32; 4],
    pub turn_actor: u8,
    pub phase: &'static str,
    pub live_wall_remaining: u32,
    pub kan_count: u32,
    pub ippatsu_active: [bool; 4],
    pub actor_furiten: &'static str,
    pub actor_can_tsumo: bool,
    pub actor_can_riichi: bool,
    pub concealed_hand: Vec<u8>,
    pub own_drawn_tile: Option<u8>,
    pub meld_counts: [usize; 4],
    pub riichi_states: [&'static str; 4],
    pub dora_indicators: [i32; 5],
    pub history_indices: Vec<usize>,
}

/// One seat's observation together with the exact event log its
/// `history_indices` refer to. Only the history that built the observation
/// constructs this, so encoder code cannot be handed a different log.
pub(crate) struct Snapshot<'a> {
    pub observation: Observation,
    pub events: &'a [TypedEvent],
}
