use super::*;
use burn::backend::{Autodiff, NdArray};

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn oracle_guidance_mask_values_follow_keep_probability() {
    let mask = oracle_guidance_mask_values(4, 0.5, &[0.1, 0.7, 0.49, 0.9]);
    assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn policy_agreement_counts_match_fraction() {
    let device: <NdArray<f32> as Backend>::Device = Default::default();
    let logits = Tensor::<NdArray<f32>, 2>::from_floats(
        [
            [10.0, 1.0, 0.0],
            [0.0, 9.0, 1.0],
            [0.0, 1.0, 8.0],
            [3.0, 2.0, 1.0],
        ],
        &device,
    );
    let mask = Tensor::<NdArray<f32>, 2>::ones([4, 3], &device);
    let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints([0, 1, 1, 2], &device);

    let acc = policy_agreement(logits.clone(), mask.clone(), targets.clone());
    let (correct, total) = policy_agreement_counts(logits, mask, targets);

    assert_eq!((correct, total), (2, 4));
    assert!((acc - correct as f64 / total as f64).abs() < 1e-12);
}

#[test]
fn policy_target_argmax_matches_batch_actions() {
    let device = Default::default();
    let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[0i32, 7, 45][..], &device);
    let mut policy_target = vec![0.0f32; 3 * 46];
    policy_target[0] = 1.0;
    policy_target[46 + 7] = 1.0;
    policy_target[2 * 46 + 45] = 1.0;
    let recovered = target_actions_from_policy_target(
        Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device).reshape([3, 46]),
    );
    let same = recovered.equal(actions).into_data().convert::<i64>();
    assert_eq!(
        same.as_slice::<i64>().expect("policy action parity"),
        &[1, 1, 1]
    );
}

#[test]
fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
    assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

    let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
    assert!((warmup_lr - 2.5e-4).abs() < 1e-12);

    let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
    let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
    assert!((post_warmup_lr - expected).abs() < 1e-12);
}

#[test]
fn oracle_guidance_mask_tensor_uses_rng_fallback_for_missing_samples() {
    let device = Default::default();
    let mask = oracle_guidance_mask_tensor::<TestBackend>(3, 0.5, &[0.25], &device)
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    assert_eq!(mask, vec![1.0, 1.0, 1.0]);
}

#[test]
fn maybe_add_exit_loss_is_noop_without_targets() {
    let device = Default::default();
    let total = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
    let logits = Tensor::<TestBackend, 2>::zeros([1, 3], &device);
    let output = maybe_add_exit_loss(total, logits, None, None, &BcExitConfig::default())
        .into_scalar()
        .elem::<f32>();
    assert!((output - 2.0).abs() < 1e-6);
}
