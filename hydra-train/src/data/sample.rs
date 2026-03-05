//! MjaiSample struct, GRP label construction, and batch collation.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;

pub struct MjaiSample {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    pub placement: u8,
    pub score_delta: i32,
    pub grp_label: u8,
    pub tenpai: [f32; 3],
    pub opp_next: [u8; 3],
    pub danger: [f32; 102],
    pub danger_mask: [f32; 102],
}

const SCORE_BIN_MIN: f32 = -50000.0;
const SCORE_BIN_MAX: f32 = 60000.0;
const SCORE_BINS: usize = 64;

pub fn scores_to_grp_index(scores: [i32; 4]) -> u8 {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ranking = [indexed[0].1, indexed[1].1, indexed[2].1, indexed[3].1];
    GRP_PERM_TABLE
        .iter()
        .position(|p| *p == ranking)
        .expect("valid permutation") as u8
}

pub const GRP_PERM_TABLE: [[u8; 4]; 24] = generate_perm_table();

const fn generate_perm_table() -> [[u8; 4]; 24] {
    let mut table = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        let d = 6 - a - b - c;
                        table[idx] = [a, b, c, d];
                        idx += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    table
}

pub fn score_delta_to_bin(score_delta: i32) -> usize {
    let range = SCORE_BIN_MAX - SCORE_BIN_MIN;
    let normalized = (score_delta as f32 - SCORE_BIN_MIN) / range;
    let bin = (normalized * SCORE_BINS as f32) as usize;
    bin.min(SCORE_BINS - 1)
}

pub fn score_delta_to_value(score_delta: i32) -> f32 {
    (score_delta as f32 / 100_000.0).clamp(-1.0, 1.0)
}

pub fn score_delta_to_pdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let mut pdf = [0.0f32; SCORE_BINS];
    pdf[score_delta_to_bin(score_delta)] = 1.0;
    pdf
}

pub fn score_delta_to_cdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let bin = score_delta_to_bin(score_delta);
    let mut cdf = [0.0f32; SCORE_BINS];
    for v in &mut cdf[bin..] {
        *v = 1.0;
    }
    cdf
}

pub struct MjaiBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub actions: Tensor<B, 1, Int>,
    pub legal_mask: Tensor<B, 2>,
    pub value_target: Tensor<B, 1>,
    pub grp_target: Tensor<B, 2>,
    pub tenpai_target: Tensor<B, 2>,
    pub danger_target: Tensor<B, 3>,
    pub danger_mask: Tensor<B, 3>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
}

pub fn collate_batch<B: Backend>(samples: &[MjaiSample], device: &B::Device) -> MjaiBatch<B> {
    let batch = samples.len();
    let mut obs_flat = vec![0.0f32; batch * OBS_SIZE];
    let mut actions = vec![0i64; batch];
    let mut mask_flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut values = vec![0.0f32; batch];
    let mut grp_flat = vec![0.0f32; batch * 24];
    let mut tenpai_flat = vec![0.0f32; batch * 3];
    let mut danger_flat = vec![0.0f32; batch * 102];
    let mut dmask_flat = vec![0.0f32; batch * 102];
    let mut opp_flat = vec![0.0f32; batch * 102];
    let mut pdf_flat = vec![0.0f32; batch * SCORE_BINS];
    let mut cdf_flat = vec![0.0f32; batch * SCORE_BINS];

    for (i, s) in samples.iter().enumerate() {
        obs_flat[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(&s.obs);
        actions[i] = s.action as i64;
        mask_flat[i * HYDRA_ACTION_SPACE..(i + 1) * HYDRA_ACTION_SPACE]
            .copy_from_slice(&s.legal_mask);
        values[i] = score_delta_to_value(s.score_delta);
        grp_flat[i * 24 + s.grp_label as usize] = 1.0;
        tenpai_flat[i * 3..(i + 1) * 3].copy_from_slice(&s.tenpai);
        danger_flat[i * 102..(i + 1) * 102].copy_from_slice(&s.danger);
        dmask_flat[i * 102..(i + 1) * 102].copy_from_slice(&s.danger_mask);
        for opp in 0..3usize {
            if s.opp_next[opp] < 34 {
                opp_flat[i * 102 + opp * 34 + s.opp_next[opp] as usize] = 1.0;
            }
        }
        let pdf = score_delta_to_pdf(s.score_delta);
        pdf_flat[i * SCORE_BINS..(i + 1) * SCORE_BINS].copy_from_slice(&pdf);
        let cdf = score_delta_to_cdf(s.score_delta);
        cdf_flat[i * SCORE_BINS..(i + 1) * SCORE_BINS].copy_from_slice(&cdf);
    }

    MjaiBatch {
        obs: Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device).reshape([batch, 85, 34]),
        actions: Tensor::<B, 1, Int>::from_ints(actions.as_slice(), device),
        legal_mask: Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
            .reshape([batch, HYDRA_ACTION_SPACE]),
        value_target: Tensor::<B, 1>::from_floats(values.as_slice(), device),
        grp_target: Tensor::<B, 1>::from_floats(grp_flat.as_slice(), device).reshape([batch, 24]),
        tenpai_target: Tensor::<B, 1>::from_floats(tenpai_flat.as_slice(), device)
            .reshape([batch, 3]),
        danger_target: Tensor::<B, 1>::from_floats(danger_flat.as_slice(), device)
            .reshape([batch, 3, 34]),
        danger_mask: Tensor::<B, 1>::from_floats(dmask_flat.as_slice(), device)
            .reshape([batch, 3, 34]),
        opp_next_target: Tensor::<B, 1>::from_floats(opp_flat.as_slice(), device)
            .reshape([batch, 3, 34]),
        score_pdf_target: Tensor::<B, 1>::from_floats(pdf_flat.as_slice(), device)
            .reshape([batch, SCORE_BINS]),
        score_cdf_target: Tensor::<B, 1>::from_floats(cdf_flat.as_slice(), device)
            .reshape([batch, SCORE_BINS]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dummy_sample(action: u8, score_delta: i32) -> MjaiSample {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[45] = 1.0;
        MjaiSample {
            obs: [0.1f32; OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta,
            grp_label: 0,
            tenpai: [0.0; 3],
            opp_next: [0, 1, 255],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
        }
    }

    #[test]
    fn test_grp_index_sorted() {
        assert_eq!(scores_to_grp_index([40000, 30000, 20000, 10000]), 0);
    }

    #[test]
    fn test_grp_index_reversed() {
        let idx = scores_to_grp_index([10000, 20000, 30000, 40000]);
        assert_ne!(idx, 0);
        assert!(idx < 24);
    }

    #[test]
    fn test_score_bin_boundaries() {
        assert_eq!(score_delta_to_bin(-50000), 0);
        assert_eq!(score_delta_to_bin(60000), SCORE_BINS - 1);
        let mid = score_delta_to_bin(5000);
        assert!(mid > 0 && mid < SCORE_BINS - 1);
    }

    #[test]
    fn test_batch_shapes() {
        let device = Default::default();
        let samples: Vec<_> = (0..32)
            .map(|i| dummy_sample(i % 34, 1000 * i as i32))
            .collect();
        let batch = collate_batch::<B>(&samples, &device);
        assert_eq!(batch.obs.dims(), [32, 85, 34]);
        assert_eq!(batch.actions.dims(), [32]);
        assert_eq!(batch.legal_mask.dims(), [32, 46]);
        assert_eq!(batch.value_target.dims(), [32]);
        assert_eq!(batch.grp_target.dims(), [32, 24]);
        assert_eq!(batch.tenpai_target.dims(), [32, 3]);
        assert_eq!(batch.danger_target.dims(), [32, 3, 34]);
        assert_eq!(batch.danger_mask.dims(), [32, 3, 34]);
        assert_eq!(batch.opp_next_target.dims(), [32, 3, 34]);
        assert_eq!(batch.score_pdf_target.dims(), [32, 64]);
        assert_eq!(batch.score_cdf_target.dims(), [32, 64]);
    }

    #[test]
    fn test_score_pdf_is_one_hot() {
        let pdf = score_delta_to_pdf(5000);
        let sum: f32 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "pdf should sum to 1");
        let nonzero = pdf.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 1, "pdf should be one-hot");
    }

    #[test]
    fn test_score_cdf_monotonic() {
        let cdf = score_delta_to_cdf(5000);
        for i in 1..64 {
            assert!(cdf[i] >= cdf[i - 1], "cdf not monotonic at {i}");
        }
        assert!((cdf[63] - 1.0).abs() < 1e-5, "cdf should end at 1");
    }

    #[test]
    fn test_value_target_range() {
        assert!((score_delta_to_value(0) - 0.0).abs() < 1e-5);
        assert!((score_delta_to_value(100_000) - 1.0).abs() < 1e-5);
        assert!((score_delta_to_value(-100_000) - (-1.0)).abs() < 1e-5);
        let mid = score_delta_to_value(50_000);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn test_legal_mask_valid() {
        let device = Default::default();
        let samples: Vec<_> = (0..4).map(|_| dummy_sample(0, 0)).collect();
        let batch = collate_batch::<B>(&samples, &device);
        let mask_data = batch.legal_mask.to_data();
        let mask_slice = mask_data.as_slice::<f32>().expect("f32");
        for row in mask_slice.chunks(46) {
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "all-zero mask found");
        }
    }
}
