#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Advanced,
    Complete,
    StepLimitExceeded,
    NoLegalAction { player: u8 },
}

impl StepOutcome {
    #[inline]
    pub fn advanced(self) -> bool {
        matches!(self, Self::Advanced)
    }
}
