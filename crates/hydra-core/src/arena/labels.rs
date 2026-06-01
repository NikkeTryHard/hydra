use crate::action::HYDRA_ACTION_SPACE;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryExitLabel {
    pub target: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryDeltaQLabel {
    pub target: [f32; HYDRA_ACTION_SPACE],
    pub mask: [f32; HYDRA_ACTION_SPACE],
}

fn label_from_slices(
    target: &[f32],
    mask: &[f32],
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])> {
    if target.len() != HYDRA_ACTION_SPACE || mask.len() != HYDRA_ACTION_SPACE {
        return None;
    }
    let mut target_arr = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask_arr = [0.0f32; HYDRA_ACTION_SPACE];
    target_arr.copy_from_slice(target);
    mask_arr.copy_from_slice(mask);
    Some((target_arr, mask_arr))
}

fn label_to_vec_pair(
    target: [f32; HYDRA_ACTION_SPACE],
    mask: [f32; HYDRA_ACTION_SPACE],
) -> (Vec<f32>, Vec<f32>) {
    (target.to_vec(), mask.to_vec())
}

impl TrajectoryDeltaQLabel {
    pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
        let (target, mask) = label_from_slices(target, mask)?;
        Some(Self { target, mask })
    }

    pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
        (self.target, self.mask)
    }

    pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
        label_to_vec_pair(self.target, self.mask)
    }
}

impl TrajectoryExitLabel {
    pub fn from_slices(target: &[f32], mask: &[f32]) -> Option<Self> {
        let (target, mask) = label_from_slices(target, mask)?;
        Some(Self { target, mask })
    }

    pub fn to_array_pair(self) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
        (self.target, self.mask)
    }

    pub fn to_vec_pair(self) -> (Vec<f32>, Vec<f32>) {
        label_to_vec_pair(self.target, self.mask)
    }
}
