//! Explicit modes for copy provenance and discard envelopes.

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Drawn {
    Held,
    Drawn,
}

impl From<bool> for Drawn {
    fn from(drawn: bool) -> Self {
        if drawn { Drawn::Drawn } else { Drawn::Held }
    }
}

impl Drawn {
    pub(super) fn is_drawn(self) -> bool {
        matches!(self, Drawn::Drawn)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DiscardEnvelope {
    Discard,
    RiichiDeclared,
}
