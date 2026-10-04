//! Immutable event storage retains canonical subbytes and visibility indices;
//! rejected envelopes cannot change any index or leak another actor's history.

use sha2::{Digest, Sha256};

use super::{EventError, TypedEvent, visible_to_actor};
/// Append-only typed arena with stream-grammar admission.
///
/// `Clone` serves the generation-end tape handoff only (the pool clones
/// per-game arenas into `trajectory::register_tape` once; the hot loop
/// never clones arenas).
/// Cache invariant: `canonical_bytes[i]` holds the RFC 8785 canonical bytes of
/// `events[i].to_json_value()`, computed once at admission through
/// `hydra_feed::canon::canonical_bytes_value` (same guards/printer as the
/// identity path, no second printer). Events are immutable after validated
/// admit (`events` stays private, `get()` yields immutable refs only), so the
/// cached subbytes stay exact for the arena lifetime. `Clone` duplicates all
/// fields index-aligned, preserving the tape handoff. No mutation or
/// truncation API exists.
///
/// Derived index invariant: `visible_index[a]` holds the absolute indices of
/// events visible to actor `a`, in admission order (sorted); `hand_starts`
/// holds the sorted hand window starts (`0` plus one `round_end_index + 1`
/// per admitted `round_end`). Both update only after a successful admit,
/// reading the then-immutable event through the existing `visible_to_actor`
/// predicate, so indexed queries observe exactly the old scan results.
#[derive(Debug, Clone)]
pub struct EventArena {
    events: Vec<TypedEvent>,
    canonical_bytes: Vec<Vec<u8>>,
    visible_index: [Vec<usize>; 4],
    hand_starts: Vec<usize>,
}

impl Default for EventArena {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            canonical_bytes: Vec::new(),
            visible_index: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            hand_starts: vec![0],
        }
    }
}

impl EventArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit one envelope only after stream/visibility/shape checks pass.
    /// Validation runs BEFORE any field is appended, so a rejected envelope
    /// leaves `events`, `canonical_bytes`, `visible_index`, and `hand_starts`
    /// index-aligned. Canonical bytes are derived from the validated event
    /// before the push; the derived index lists then read the then-immutable
    /// admitted event, never the pushed value.
    pub fn admit(&mut self, event: TypedEvent) -> Result<(), EventError> {
        if let Some(prev) = self.events.last() {
            if event.game_id != prev.game_id {
                return Err(EventError("stream mixes games".to_string()));
            }
            if event.sequence <= prev.sequence {
                return Err(EventError(format!(
                    "sequence {} does not strictly increase past {}",
                    event.sequence, prev.sequence
                )));
            }
        } else if event.kind != "game_start" {
            return Err(EventError("streams begin with 'game_start'".to_string()));
        }
        event.check_visibility()?;
        event.check_kind_shape()?;
        let bytes =
            hydra_feed::canon::canonical_bytes_value(&event.to_json_value(), "rl:event-envelope")
                .map_err(|e| EventError(format!("envelope canon failed: {e:?}")))?;
        self.events.push(event);
        self.canonical_bytes.push(bytes);
        debug_assert_eq!(self.events.len(), self.canonical_bytes.len());
        let index = self.events.len() - 1;
        // Read the then-immutable event once; the mutable pushes below
        // touch only the derived lists, never the event itself.
        let (v0, v1, v2, v3, is_round_end) = {
            let admitted = &self.events[index];
            (
                visible_to_actor(admitted, 0),
                visible_to_actor(admitted, 1),
                visible_to_actor(admitted, 2),
                visible_to_actor(admitted, 3),
                admitted.kind == "round_end",
            )
        };
        if v0 {
            self.visible_index[0].push(index);
        }
        if v1 {
            self.visible_index[1].push(index);
        }
        if v2 {
            self.visible_index[2].push(index);
        }
        if v3 {
            self.visible_index[3].push(index);
        }
        if is_round_end {
            self.hand_starts.push(index + 1);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&TypedEvent> {
        self.events.get(index)
    }

    /// Immutable canonical event bytes by index (`None` when outside arena).
    /// Reused by the identity hash emitter; never mutated, never truncated.
    /// Stays index-aligned with `events` across `admit`/`Clone`.
    pub fn cached_canonical_bytes(&self, index: usize) -> Option<&[u8]> {
        self.canonical_bytes.get(index).map(Vec::as_slice)
    }

    /// Actor-visible prefix indices (mirrors `filter_events_for_actor`).
    /// NOTE: spans all hands; per-row facts MUST use
    /// `visible_indices_for_hand` instead (the reference builds one fresh
    /// builder per hand, so history never crosses a hand boundary).
    pub fn visible_indices_for(&self, actor: u8, upto: usize) -> Vec<usize> {
        self.events[..upto.min(self.events.len())]
            .iter()
            .enumerate()
            .filter(|(_, e)| visible_to_actor(e, actor))
            .map(|(i, _)| i)
            .collect()
    }

    /// First index of the current hand's window below `upto`: the event
    /// after the latest `round_end`, else 0 (hand 0 keeps `game_start` in
    /// its window, exactly like the reference hand-0 builder which ingests
    /// `game_start` before the first `round_start`). Prior-hand envelopes
    /// stay addressable by absolute prefix (replay/sort identity) but never
    /// enter per-row facts.
    ///
    /// Answered from the sorted `hand_starts` via binary search. `upto` is
    /// clamped to the arena length first, so an arbitrary prefix past the end
    /// observes `len`, `upto == 0` yields `0`, and a `round_end` at index `i`
    /// is excluded for `upto == i` yet starts the next window (`i + 1`) for
    /// `upto == i + 1`.
    pub fn hand_window_start(&self, upto: usize) -> usize {
        let end = upto.min(self.events.len());
        // `hand_starts` is sorted with `hand_starts[0] == 0`: the window
        // is the latest start at or below the clamped prefix end.
        match self.hand_starts.partition_point(|&s| s <= end) {
            0 => 0,
            n => self.hand_starts[n - 1],
        }
    }

    /// Actor-visible indices within the current hand window (mirrors
    /// `filter_events_for_actor` over the hand's builder caches). Indices
    /// stay absolute (addressable into the full arena); only the range is
    /// windowed.
    ///
    /// Valid actors `0..4` slice the sorted per-actor list to
    /// `[hand_start, clamped_upto)` via binary search, returning exactly the
    /// old scan's `Vec<usize>`. Out-of-range actors have no derived list
    /// (public events stay visible to them under the old predicate), so they
    /// run the original scan for an exact fallback.
    pub fn visible_indices_for_hand(&self, actor: u8, upto: usize) -> Vec<usize> {
        if actor < 4 {
            let end = upto.min(self.events.len());
            let start = self.hand_window_start(upto);
            let list = &self.visible_index[actor as usize];
            let lo = list.partition_point(|&i| i < start);
            let hi = list.partition_point(|&i| i < end);
            list[lo..hi].to_vec()
        } else {
            let start = self.hand_window_start(upto);
            let end = upto.min(self.events.len());
            self.events[start..end]
                .iter()
                .enumerate()
                .filter(|(_, e)| visible_to_actor(e, actor))
                .map(|(i, _)| start + i)
                .collect()
        }
    }

    /// Actor-visible event count within the current hand window below `upto`.
    /// Valid actors `0..4` difference the same binary-search bounds as
    /// `visible_indices_for_hand` with zero allocation; out-of-range actors
    /// count the exact same old scan (`hand_window_start` +
    /// `visible_to_actor` filter), so the result always equals
    /// `visible_indices_for_hand(actor, upto).len()`.
    pub fn visible_count_for_hand(&self, actor: u8, upto: usize) -> usize {
        if actor < 4 {
            let end = upto.min(self.events.len());
            let start = self.hand_window_start(upto);
            let list = &self.visible_index[actor as usize];
            let lo = list.partition_point(|&i| i < start);
            let hi = list.partition_point(|&i| i < end);
            hi - lo
        } else {
            let start = self.hand_window_start(upto);
            let end = upto.min(self.events.len());
            self.events[start..end]
                .iter()
                .filter(|e| visible_to_actor(e, actor))
                .count()
        }
    }

    /// sha256 over RFC 8785 canonical bytes of one envelope JSON.
    pub fn envelope_digest(event_json: &serde_json::Value) -> Result<[u8; 32], EventError> {
        let bytes = hydra_feed::canon::canonical_bytes_value(event_json, "rl:event-envelope")
            .map_err(|e| EventError(format!("envelope canon failed: {e:?}")))?;
        Ok(Sha256::digest(&bytes).into())
    }
}
