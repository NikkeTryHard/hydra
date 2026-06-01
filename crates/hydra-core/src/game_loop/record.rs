use crate::action::HYDRA_ACTION_SPACE;
use crate::encoder::OBS_SIZE;

#[derive(Debug, Clone)]
pub struct DecisionRecord {
    pub obs: [f32; OBS_SIZE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub action: u8,
    pub legal_count: u8,
    pub player_id: u8,
    pub seat_id: u8,
    pub turn: u32,
}

pub(super) trait DecisionRecorder {
    fn record(&mut self, record: DecisionRecord);
}

pub(super) struct NoopDecisionRecorder;

impl DecisionRecorder for NoopDecisionRecorder {
    #[inline]
    fn record(&mut self, _record: DecisionRecord) {}
}

impl<F> DecisionRecorder for F
where
    F: FnMut(DecisionRecord),
{
    #[inline]
    fn record(&mut self, record: DecisionRecord) {
        self(record);
    }
}
