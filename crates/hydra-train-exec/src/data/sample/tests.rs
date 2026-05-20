use super::*;
use burn::backend::NdArray;
use hydra_core::tile::permute_tile_type;

use hydra_bc_shards::{BcShardHostBatch, BcShardHostScratch};
type B = NdArray<f32>;
fn expect_collation_err<T>(result: Result<Option<T>, String>, message: &str) -> String {
    match result {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}

fn dummy_sample(action: u8, score_delta: i32) -> MjaiSample {
    let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
    legal_mask[action as usize] = 1.0;
    legal_mask[45] = 1.0;
    MjaiSample {
        obs: [0.1f32; OBS_SIZE],
        compact_facts: None,
        action,
        legal_mask,
        placement: 0,
        score_delta,
        grp_label: 0,
        oracle_target: None,
        tenpai: [0.0; 3],
        opp_next: [0, 1, 255],
        danger: [0.0; 102],
        danger_mask: [1.0; 102],
        safety_residual: None,
        safety_residual_mask: None,
        exit_target: None,
        exit_mask: None,
        delta_q_target: None,
        delta_q_mask: None,
        belief_fields: None,
        mixture_weights: None,
        belief_fields_present: false,
        mixture_weights_present: false,
    }
}

#[test]
fn test_batch_shapes() {
    let device = Default::default();
    let samples: Vec<_> = (0..32)
        .map(|i| dummy_sample(i % 34, 1000 * i as i32))
        .collect();
    let batch = collate_batch::<B>(&samples, &device);
    assert_eq!(batch.obs.dims(), [32, NUM_CHANNELS, 34]);
    assert_eq!(batch.actions.dims(), [32]);
    assert_eq!(batch.legal_mask.dims(), [32, HYDRA_ACTION_SPACE]);
    assert_eq!(batch.value_target.dims(), [32]);
    assert_eq!(batch.grp_target.dims(), [32, GRP_CLASS_COUNT]);
    assert!(batch.oracle_target.is_none());
    assert_eq!(batch.oracle_target_mask.dims(), [32]);
    assert_eq!(batch.tenpai_target.dims(), [32, OPPONENT_COUNT]);
    assert_eq!(batch.danger_target.dims(), [32, OPPONENT_COUNT, TILE_COUNT]);
    assert_eq!(batch.danger_mask.dims(), [32, OPPONENT_COUNT, TILE_COUNT]);
    assert!(batch.safety_residual_target.is_none());
    assert!(batch.safety_residual_mask.is_none());
    assert_eq!(
        batch.opp_next_target.dims(),
        [32, OPPONENT_COUNT, TILE_COUNT]
    );
    assert_eq!(batch.score_pdf_target.dims(), [32, 64]);
    assert_eq!(batch.score_cdf_target.dims(), [32, 64]);
}

#[test]
fn test_legal_mask_valid() {
    let device = Default::default();
    let samples: Vec<_> = (0..4).map(|_| dummy_sample(0, 0)).collect();
    let batch = collate_batch::<B>(&samples, &device);
    let mask_data = batch.legal_mask.to_data();
    let mask_slice = mask_data.as_slice::<f32>().expect("f32");
    for row in mask_slice.chunks(HYDRA_ACTION_SPACE) {
        let sum: f32 = row.iter().sum();
        assert!(sum > 0.0, "all-zero mask found");
    }
}

#[test]
fn test_opp_next_255_is_zero() {
    let device = Default::default();
    let samples = vec![dummy_sample(0, 0)];
    let batch = collate_batch::<B>(&samples, &device);
    let data = batch.opp_next_target.to_data();
    let slice = data.as_slice::<f32>().expect("f32");
    let opp2_start = 2 * TILE_COUNT;
    let opp2_sum: f32 = slice[opp2_start..opp2_start + TILE_COUNT].iter().sum();
    assert!(
        opp2_sum.abs() < 1e-5,
        "opp_next=255 should be all zero, sum={opp2_sum}"
    );
}

#[test]
fn test_single_sample_batch() {
    let device = Default::default();
    let samples = vec![dummy_sample(5, 12000)];
    let batch = collate_batch::<B>(&samples, &device);
    assert_eq!(batch.obs.dims(), [1, NUM_CHANNELS, 34]);
    assert_eq!(batch.actions.dims(), [1]);
    let action_data = batch.actions.to_data();
    assert_eq!(action_data.as_slice::<i64>().expect("i64")[0], 5);
}

#[test]
fn test_extreme_score_deltas() {
    let device = Default::default();
    let samples = vec![
        dummy_sample(0, -100_000),
        dummy_sample(1, 100_000),
        dummy_sample(2, 0),
    ];
    let batch = collate_batch::<B>(&samples, &device);
    let val_data = batch.value_target.to_data();
    let vals = val_data.as_slice::<f32>().expect("f32");
    assert!((vals[0] - (-1.0)).abs() < 1e-5);
    assert!((vals[1] - 1.0).abs() < 1e-5);
    assert!((vals[2] - 0.0).abs() < 1e-5);
}

#[test]
fn augment_samples_6x_permutes_aux_tile_targets() {
    use hydra_core::tile::ALL_PERMUTATIONS;

    let mut sample = dummy_sample(0, 0);
    sample.obs = [0.0; OBS_SIZE];
    sample.opp_next = [0, 9, 27];
    sample.danger[0] = 0.25;
    sample.danger[34 + 9] = 0.5;
    sample.danger_mask[18] = 1.0;
    let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual[0] = -0.75;
    safety_residual[1] = 0.4;
    safety_residual_mask[0] = 1.0;
    safety_residual_mask[1] = 1.0;
    sample.safety_residual = Some(safety_residual);
    sample.safety_residual_mask = Some(safety_residual_mask);
    sample.obs[40 * 34] = 1.0;

    let augmented = augment_samples_6x(&[sample]);
    let swap_mp = &ALL_PERMUTATIONS[2];
    let swapped = augmented
        .iter()
        .find(|s| s.action == 9)
        .expect("swap man-pin permutation sample");

    assert_eq!(permute_tile_type(0, swap_mp), 9);
    assert_eq!(swapped.opp_next, [9, 0, 27]);
    assert!((swapped.danger[9] - 0.25).abs() < 1e-6);
    assert!((swapped.danger[34] - 0.5).abs() < 1e-6);
    assert_eq!(swapped.danger_mask[18], 1.0);
    let sr = swapped.safety_residual.expect("safety residual target");
    let srm = swapped.safety_residual_mask.expect("safety residual mask");
    assert!((sr[9] + 0.75).abs() < 1e-6);
    assert!((sr[10] - 0.4).abs() < 1e-6);
    assert!((srm[9] - 1.0).abs() < 1e-6);
    assert!((srm[10] - 1.0).abs() < 1e-6);
    assert_eq!(swapped.obs[41 * 34], 1.0);
    assert_eq!(swapped.obs[40 * 34], 0.0);
}

#[test]
fn batch_to_hydra_targets_carries_oracle_target() {
    let device = Default::default();
    let mut sample = dummy_sample(5, 12000);
    sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.to_hydra_targets();
    assert_eq!(targets.policy_target.dims(), [1, HYDRA_ACTION_SPACE]);
    let oracle = targets.oracle_target.expect("oracle target present");
    assert_eq!(oracle.dims(), [1, 4]);
    let data = oracle.to_data();
    let slice = data.as_slice::<f32>().expect("f32");
    assert!((slice[0] - 0.1).abs() < 1e-6);
    assert!((slice[1] + 0.1).abs() < 1e-6);
    assert!((slice[2] - 0.2).abs() < 1e-6);
    assert!((slice[3] + 0.2).abs() < 1e-6);
    let mask = targets.oracle_guidance_mask.expect("oracle mask present");
    let mask_slice = mask.to_data().as_slice::<f32>().expect("f32").to_vec();
    assert!((mask_slice[0] - 1.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_policy_matches_actions() {
    let device = Default::default();
    let samples = vec![dummy_sample(2, 0), dummy_sample(7, 0)];
    let batch = collate_batch::<B>(&samples, &device);
    let targets = batch.into_hydra_targets();
    assert_eq!(targets.policy_target.dims(), [2, HYDRA_ACTION_SPACE]);
    let data = targets.policy_target.to_data();
    let slice = data.as_slice::<f32>().expect("f32");
    assert!((slice[2] - 1.0).abs() < 1e-6);
    assert!((slice[HYDRA_ACTION_SPACE + 7] - 1.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_keeps_optional_advanced_targets_narrow() {
    let device = Default::default();
    let mut sample = dummy_sample(1, 0);
    sample.oracle_target = Some([0.25, 0.0, -0.25, 0.0]);
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    assert!(targets.oracle_target.is_some());
    assert!(targets.belief_fields_target.is_none());
    assert!(targets.mixture_weight_target.is_none());
    assert!(targets.opponent_hand_type_target.is_none());
    assert!(targets.delta_q_target.is_none());
    assert!(targets.delta_q_mask.is_none());
    assert!(targets.safety_residual_target.is_none());
}

#[test]
fn batch_to_hydra_targets_keeps_oracle_absent_when_missing() {
    let device = Default::default();
    let batch = collate_batch::<B>(&[dummy_sample(3, 0)], &device);
    assert!(batch.oracle_target.is_none());
    let targets = batch.into_hydra_targets();
    assert!(targets.oracle_target.is_none());
    let mask = targets.oracle_guidance_mask.expect("oracle mask present");
    let mask_slice = mask.to_data().as_slice::<f32>().expect("f32").to_vec();
    assert!((mask_slice[0] - 0.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_carries_safety_residual() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[0] = -0.4;
    target[34] = 0.7;
    mask[0] = 1.0;
    mask[34] = 1.0;
    sample.safety_residual = Some(target);
    sample.safety_residual_mask = Some(mask);
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    let sr = targets
        .safety_residual_target
        .expect("safety residual target");
    let srm = targets.safety_residual_mask.expect("safety residual mask");
    assert_eq!(sr.dims(), [1, HYDRA_ACTION_SPACE]);
    assert_eq!(srm.dims(), [1, HYDRA_ACTION_SPACE]);
    let values = sr.to_data().as_slice::<f32>().expect("f32").to_vec();
    let mask_values = srm.to_data().as_slice::<f32>().expect("f32").to_vec();
    assert!((values[0] + 0.4).abs() < 1e-6);
    assert!((values[34] - 0.7).abs() < 1e-6);
    assert!((mask_values[0] - 1.0).abs() < 1e-6);
    assert!((mask_values[34] - 1.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_carries_delta_q_with_mask() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[0] = 0.6;
    target[9] = -0.25;
    mask[0] = 1.0;
    mask[9] = 1.0;
    sample.delta_q_target = Some(target);
    sample.delta_q_mask = Some(mask);

    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    let dq = targets.delta_q_target.expect("delta_q target");
    let dqm = targets.delta_q_mask.expect("delta_q mask");
    assert_eq!(dq.dims(), [1, HYDRA_ACTION_SPACE]);
    assert_eq!(dqm.dims(), [1, HYDRA_ACTION_SPACE]);
    let values = dq.to_data().as_slice::<f32>().expect("f32").to_vec();
    let mask_values = dqm.to_data().as_slice::<f32>().expect("f32").to_vec();
    assert!((values[0] - 0.6).abs() < 1e-6);
    assert!((values[9] + 0.25).abs() < 1e-6);
    assert!((mask_values[0] - 1.0).abs() < 1e-6);
    assert!((mask_values[9] - 1.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_carries_target_presence_metadata() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
    sample.belief_fields = Some([0.0; 16 * 34]);
    sample.belief_fields_present = true;
    sample.mixture_weights = Some([0.7, 0.3, 0.0, 0.0]);
    sample.mixture_weights_present = true;
    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_target[0] = 0.6;
    delta_q_target[9] = -0.25;
    delta_q_mask[0] = 1.0;
    delta_q_mask[9] = 1.0;
    sample.delta_q_target = Some(delta_q_target);
    sample.delta_q_mask = Some(delta_q_mask);
    let mut safety_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_target[3] = 0.5;
    safety_mask[3] = 1.0;
    sample.safety_residual = Some(safety_target);
    sample.safety_residual_mask = Some(safety_mask);

    let batch = collate_batch::<B>(&[sample], &device);
    let presence = batch.target_presence.expect("batch target presence");
    assert_eq!(presence.batch_size, 1);
    assert_eq!(presence.counts[AdvancedHead::OracleCritic.index()], 1);
    assert_eq!(presence.counts[AdvancedHead::BeliefFields.index()], 1);
    assert_eq!(presence.counts[AdvancedHead::MixtureWeight.index()], 1);
    assert_eq!(presence.counts[AdvancedHead::DeltaQ.index()], 1);
    assert_eq!(presence.counts[AdvancedHead::SafetyResidual.index()], 1);
    assert_eq!(presence.delta_q_actions_present, 2);
    let targets = batch.into_hydra_targets();
    let target_presence = targets
        .target_presence
        .expect("targets should carry cached target presence");
    assert_eq!(target_presence.batch_size, 1);
    assert_eq!(target_presence.delta_q_actions_present, 2);
}

#[test]
fn batch_to_hydra_targets_rejects_delta_q_when_pair_is_incomplete() {
    let device = Default::default();
    let mut target_only = dummy_sample(0, 0);
    let mut mask_only = dummy_sample(1, 0);
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[0] = 0.6;
    mask[1] = 1.0;
    target_only.delta_q_target = Some(target);
    mask_only.delta_q_mask = Some(mask);

    let err = expect_collation_err(
        collate_batch_samples::<B>(&[target_only, mask_only], false, &device),
        "incomplete delta_q pair should fail",
    );
    assert!(err.contains("delta_q target/mask mismatch for sample collation"));
}

#[test]
fn batch_to_hydra_targets_carries_projected_belief_targets() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    let mut belief = [0.0f32; 16 * 34];
    let mut mix = [0.0f32; 4];
    belief[0] = 0.2;
    belief[33] = 0.8;
    mix[0] = 0.7;
    mix[1] = 0.3;
    sample.belief_fields = Some(belief);
    sample.mixture_weights = Some(mix);
    sample.belief_fields_present = true;
    sample.mixture_weights_present = true;
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    let belief_target = targets
        .belief_fields_target
        .expect("belief field target should be present");
    let mix_target = targets
        .mixture_weight_target
        .expect("mixture weights should be present");
    assert_eq!(belief_target.dims(), [1, 16, 34]);
    assert_eq!(mix_target.dims(), [1, 4]);
    let belief_values = belief_target
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let mix_values = mix_target
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let belief_mask = targets.belief_fields_mask.expect("belief mask");
    let mixture_mask = targets.mixture_weight_mask.expect("mixture mask");
    let belief_mask_values = belief_mask
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let mixture_mask_values = mixture_mask
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    assert!((belief_values[0] - 0.2).abs() < 1e-6);
    assert!((belief_values[33] - 0.8).abs() < 1e-6);
    assert!((mix_values[0] - 0.7).abs() < 1e-6);
    assert!((mix_values[1] - 0.3).abs() < 1e-6);
    assert!((belief_mask_values[0] - 1.0).abs() < 1e-6);
    assert!((mixture_mask_values[0] - 1.0).abs() < 1e-6);
}

#[test]
fn batch_to_hydra_targets_rejects_belief_target_without_presence() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.belief_fields = Some([0.0; 16 * 34]);
    let err = expect_collation_err(
        collate_batch_samples::<B>(&[sample], false, &device),
        "belief target without presence should fail",
    );
    assert!(err.contains("belief_fields target/presence mismatch for sample collation"));
}

#[test]
fn batch_to_hydra_targets_rejects_belief_presence_without_target() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.belief_fields_present = true;
    let err = expect_collation_err(
        collate_batch_samples::<B>(&[sample], false, &device),
        "belief presence without target should fail",
    );
    assert!(err.contains("belief_fields target/presence mismatch for sample collation"));
}

#[test]
fn batch_to_hydra_targets_rejects_mixture_target_without_presence() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.mixture_weights = Some([0.0; 4]);
    let err = expect_collation_err(
        collate_batch_samples::<B>(&[sample], false, &device),
        "mixture target without presence should fail",
    );
    assert!(err.contains("mixture_weight target/presence mismatch for sample collation"));
}

#[test]
fn batch_to_hydra_targets_rejects_mixture_presence_without_target() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.mixture_weights_present = true;
    let err = expect_collation_err(
        collate_batch_samples::<B>(&[sample], false, &device),
        "mixture presence without target should fail",
    );
    assert!(err.contains("mixture_weight target/presence mismatch for sample collation"));
}

#[test]
fn batch_to_hydra_targets_keeps_belief_targets_absent_when_missing() {
    let device = Default::default();
    let batch = collate_batch::<B>(&[dummy_sample(0, 0)], &device);
    let targets = batch.into_hydra_targets();
    assert!(targets.belief_fields_target.is_none());
    assert!(targets.mixture_weight_target.is_none());
    assert!(targets.belief_fields_mask.is_none());
    assert!(targets.mixture_weight_mask.is_none());
}

#[test]
fn augment_samples_6x_permutes_belief_fields_and_preserves_mixture_weights() {
    use hydra_core::tile::ALL_PERMUTATIONS;

    let mut sample = dummy_sample(0, 0);
    let mut belief = [0.0f32; 16 * 34];
    belief[0] = 1.0;
    let mut mix = [0.0f32; 4];
    mix[0] = 0.8;
    sample.belief_fields = Some(belief);
    sample.belief_fields_present = true;
    sample.mixture_weights = Some(mix);
    sample.mixture_weights_present = true;

    let augmented = augment_samples_6x(&[sample]);
    let swap_mp = &ALL_PERMUTATIONS[2];
    let swapped = augmented
        .iter()
        .find(|s| s.action == 9)
        .expect("swap man-pin permutation sample");
    let swapped_belief = swapped.belief_fields.expect("belief fields");
    let swapped_mix = swapped.mixture_weights.expect("mixture weights");
    assert_eq!(permute_tile_type(0, swap_mp), 9);
    assert!((swapped_belief[9] - 1.0).abs() < 1e-6);
    assert!((swapped_mix[0] - 0.8).abs() < 1e-6);
}

#[test]
fn collate_sample_refs_matches_owned_collation_without_augmentation() {
    let device = Default::default();
    let samples = vec![dummy_sample(2, 100), dummy_sample(7, -500)];
    let refs: Vec<_> = samples.iter().collect();

    let (obs, targets) = collate_sample_refs::<B>(&refs, false, &device)
        .expect("borrowed collate")
        .expect("borrowed batch present");
    let (owned_obs, owned_targets) = collate_samples::<B>(&samples, false, &device)
        .expect("owned collate")
        .expect("owned batch present");

    assert_eq!(obs.dims(), owned_obs.dims());
    assert_eq!(
        targets.policy_target.dims(),
        owned_targets.policy_target.dims()
    );
    assert_eq!(targets.legal_mask.dims(), owned_targets.legal_mask.dims());
    assert_eq!(
        targets.danger_target.dims(),
        owned_targets.danger_target.dims()
    );
    assert_eq!(obs.to_data(), owned_obs.to_data());
    assert_eq!(
        targets.policy_target.to_data(),
        owned_targets.policy_target.to_data()
    );
    assert_eq!(
        targets.legal_mask.to_data(),
        owned_targets.legal_mask.to_data()
    );
    assert_eq!(
        targets.value_target.to_data(),
        owned_targets.value_target.to_data()
    );
    assert_eq!(
        targets.grp_target.to_data(),
        owned_targets.grp_target.to_data()
    );
    assert_eq!(
        targets.danger_target.to_data(),
        owned_targets.danger_target.to_data()
    );
    assert_eq!(
        targets.danger_mask.to_data(),
        owned_targets.danger_mask.to_data()
    );
    assert_eq!(
        targets.opp_next_target.to_data(),
        owned_targets.opp_next_target.to_data()
    );
    assert_eq!(
        targets.score_pdf_target.to_data(),
        owned_targets.score_pdf_target.to_data()
    );
    assert_eq!(
        targets.score_cdf_target.to_data(),
        owned_targets.score_cdf_target.to_data()
    );
}

#[test]
fn collate_samples_owned_matches_split_batch_and_targets_without_augmentation() {
    let device = Default::default();
    let samples = vec![dummy_sample(2, 100), dummy_sample(7, -500)];

    let (owned_obs, owned_batch, owned_targets) =
        collate_samples_owned::<B>(&samples, false, &device)
            .expect("owned collate")
            .expect("owned batch present");
    let (obs, batch) = collate_batch_samples::<B>(&samples, false, &device)
        .expect("batch collate")
        .expect("batch present");
    let targets = batch.to_hydra_targets();

    assert_eq!(owned_obs.to_data(), obs.to_data());
    assert_eq!(owned_batch.obs.to_data(), batch.obs.to_data());
    assert_eq!(owned_batch.actions.to_data(), batch.actions.to_data());
    assert_eq!(
        owned_targets.policy_target.to_data(),
        targets.policy_target.to_data()
    );
    assert_eq!(
        owned_targets.legal_mask.to_data(),
        targets.legal_mask.to_data()
    );
    assert_eq!(
        owned_targets.value_target.to_data(),
        targets.value_target.to_data()
    );
    assert_eq!(
        owned_targets.grp_target.to_data(),
        targets.grp_target.to_data()
    );
    assert_eq!(
        owned_targets.danger_target.to_data(),
        targets.danger_target.to_data()
    );
    assert_eq!(
        owned_targets.danger_mask.to_data(),
        targets.danger_mask.to_data()
    );
    assert_eq!(
        owned_targets.opp_next_target.to_data(),
        targets.opp_next_target.to_data()
    );
    assert_eq!(
        owned_targets.score_pdf_target.to_data(),
        targets.score_pdf_target.to_data()
    );
    assert_eq!(
        owned_targets.score_cdf_target.to_data(),
        targets.score_cdf_target.to_data()
    );
    let owned_presence = owned_targets
        .target_presence
        .expect("owned target presence");
    let split_presence = targets.target_presence.expect("target presence");
    assert_eq!(owned_presence.batch_size, split_presence.batch_size);
    assert_eq!(owned_presence.counts, split_presence.counts);
    assert_eq!(
        owned_presence.delta_q_actions_present,
        split_presence.delta_q_actions_present
    );
}

#[test]
fn collate_sample_refs_matches_owned_collation_with_augmentation() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.obs = [0.0; OBS_SIZE];
    sample.obs[40 * 34] = 1.0;
    sample.opp_next = [0, 9, 27];
    sample.danger[0] = 0.25;
    sample.danger[34 + 9] = 0.5;
    sample.danger_mask[18] = 1.0;
    let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual[0] = -0.75;
    safety_residual[1] = 0.4;
    safety_residual_mask[0] = 1.0;
    safety_residual_mask[1] = 1.0;
    sample.safety_residual = Some(safety_residual);
    sample.safety_residual_mask = Some(safety_residual_mask);
    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_target[0] = 0.6;
    delta_q_target[1] = -0.25;
    delta_q_mask[0] = 1.0;
    delta_q_mask[1] = 1.0;
    sample.delta_q_target = Some(delta_q_target);
    sample.delta_q_mask = Some(delta_q_mask);

    let refs = vec![&sample];
    let (obs, targets) = collate_sample_refs::<B>(&refs, true, &device)
        .expect("borrowed collate")
        .expect("borrowed batch present");
    let (owned_obs, owned_targets) = collate_samples::<B>(&[sample], true, &device)
        .expect("owned collate")
        .expect("owned batch present");

    assert_eq!(obs.dims(), owned_obs.dims());
    assert_eq!(
        targets.policy_target.to_data(),
        owned_targets.policy_target.to_data()
    );
    assert_eq!(
        targets.legal_mask.to_data(),
        owned_targets.legal_mask.to_data()
    );
    assert_eq!(
        targets.danger_target.to_data(),
        owned_targets.danger_target.to_data()
    );
    assert_eq!(
        targets.danger_mask.to_data(),
        owned_targets.danger_mask.to_data()
    );
    assert_eq!(
        targets.opp_next_target.to_data(),
        owned_targets.opp_next_target.to_data()
    );
    assert_eq!(
        targets.score_pdf_target.to_data(),
        owned_targets.score_pdf_target.to_data()
    );
    assert_eq!(
        targets.score_cdf_target.to_data(),
        owned_targets.score_cdf_target.to_data()
    );
    assert_eq!(
        targets
            .safety_residual_target
            .expect("borrowed safety residual")
            .to_data(),
        owned_targets
            .safety_residual_target
            .expect("owned safety residual")
            .to_data()
    );
    assert_eq!(
        targets
            .safety_residual_mask
            .expect("borrowed safety residual mask")
            .to_data(),
        owned_targets
            .safety_residual_mask
            .expect("owned safety residual mask")
            .to_data()
    );
    assert_eq!(
        targets.delta_q_target.expect("borrowed delta_q").to_data(),
        owned_targets
            .delta_q_target
            .expect("owned delta_q")
            .to_data()
    );
    assert_eq!(
        targets
            .delta_q_mask
            .expect("borrowed delta_q mask")
            .to_data(),
        owned_targets
            .delta_q_mask
            .expect("owned delta_q mask")
            .to_data()
    );
}

#[test]
fn collate_samples_owned_matches_split_batch_and_targets_with_augmentation() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.obs = [0.0; OBS_SIZE];
    sample.obs[40 * 34] = 1.0;
    sample.opp_next = [0, 9, 27];
    sample.danger[0] = 0.25;
    sample.danger[34 + 9] = 0.5;
    sample.danger_mask[18] = 1.0;
    let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual[0] = -0.75;
    safety_residual[1] = 0.4;
    safety_residual_mask[0] = 1.0;
    safety_residual_mask[1] = 1.0;
    sample.safety_residual = Some(safety_residual);
    sample.safety_residual_mask = Some(safety_residual_mask);
    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_target[0] = 0.6;
    delta_q_target[1] = -0.25;
    delta_q_mask[0] = 1.0;
    delta_q_mask[1] = 1.0;
    sample.delta_q_target = Some(delta_q_target);
    sample.delta_q_mask = Some(delta_q_mask);

    let owned_sample = sample;
    let mut split_sample = dummy_sample(0, 0);
    split_sample.obs = [0.0; OBS_SIZE];
    split_sample.obs[40 * 34] = 1.0;
    split_sample.opp_next = [0, 9, 27];
    split_sample.danger[0] = 0.25;
    split_sample.danger[34 + 9] = 0.5;
    split_sample.danger_mask[18] = 1.0;
    split_sample.safety_residual = Some(safety_residual);
    split_sample.safety_residual_mask = Some(safety_residual_mask);
    split_sample.delta_q_target = Some(delta_q_target);
    split_sample.delta_q_mask = Some(delta_q_mask);

    let (owned_obs, owned_batch, owned_targets) =
        collate_samples_owned::<B>(&[owned_sample], true, &device)
            .expect("owned collate")
            .expect("owned batch present");
    let (obs, batch) = collate_batch_samples::<B>(&[split_sample], true, &device)
        .expect("batch collate")
        .expect("batch present");
    let targets = batch.to_hydra_targets();

    assert_eq!(owned_obs.to_data(), obs.to_data());
    assert_eq!(owned_batch.obs.to_data(), batch.obs.to_data());
    assert_eq!(owned_batch.actions.to_data(), batch.actions.to_data());
    assert_eq!(
        owned_targets.policy_target.to_data(),
        targets.policy_target.to_data()
    );
    assert_eq!(
        owned_targets.danger_target.to_data(),
        targets.danger_target.to_data()
    );
    assert_eq!(
        owned_targets.danger_mask.to_data(),
        targets.danger_mask.to_data()
    );
    assert_eq!(
        owned_targets.opp_next_target.to_data(),
        targets.opp_next_target.to_data()
    );
    assert_eq!(
        owned_targets
            .safety_residual_target
            .expect("owned safety residual")
            .to_data(),
        targets
            .safety_residual_target
            .expect("safety residual")
            .to_data()
    );
    assert_eq!(
        owned_targets
            .delta_q_target
            .expect("owned delta_q")
            .to_data(),
        targets.delta_q_target.expect("delta_q").to_data()
    );
    let owned_presence = owned_targets
        .target_presence
        .expect("owned target presence");
    let split_presence = targets.target_presence.expect("target presence");
    assert_eq!(owned_presence.batch_size, split_presence.batch_size);
    assert_eq!(owned_presence.counts, split_presence.counts);
    assert_eq!(
        owned_presence.delta_q_actions_present,
        split_presence.delta_q_actions_present
    );
}

#[test]
fn collate_samples_bc_owned_matches_split_batch_targets_and_exit_surface() {
    let device = Default::default();
    let mut sample = dummy_sample(2, 100);
    sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[2] = 0.75;
    exit_target[45] = -0.25;
    exit_mask[2] = 1.0;
    exit_mask[45] = 1.0;
    sample.exit_target = Some(exit_target);
    sample.exit_mask = Some(exit_mask);
    let samples = [sample];

    let (obs, bc_batch, targets) = collate_samples_bc_owned::<B>(&samples, false, &device)
        .expect("bc owned collate")
        .expect("bc owned batch present");
    let (split_obs, split_batch) = collate_batch_samples::<B>(&samples, false, &device)
        .expect("split batch collate")
        .expect("split batch present");
    let split_targets = split_batch.to_hydra_targets();

    assert_eq!(obs.to_data(), split_obs.to_data());
    assert_eq!(bc_batch.actions.to_data(), split_batch.actions.to_data());
    assert_eq!(
        bc_batch
            .exit_target
            .as_ref()
            .expect("bc exit target")
            .to_data(),
        split_batch
            .exit_target
            .as_ref()
            .expect("split exit target")
            .to_data()
    );
    assert_eq!(
        bc_batch.exit_mask.as_ref().expect("bc exit mask").to_data(),
        split_batch
            .exit_mask
            .as_ref()
            .expect("split exit mask")
            .to_data()
    );
    assert_eq!(
        targets.policy_target.to_data(),
        split_targets.policy_target.to_data()
    );
    assert_eq!(
        targets.legal_mask.to_data(),
        split_targets.legal_mask.to_data()
    );
    assert_eq!(
        targets.value_target.to_data(),
        split_targets.value_target.to_data()
    );
    assert_eq!(
        targets.grp_target.to_data(),
        split_targets.grp_target.to_data()
    );
    assert_eq!(
        targets.oracle_target.expect("bc oracle target").to_data(),
        split_targets
            .oracle_target
            .expect("split oracle target")
            .to_data()
    );
    assert_eq!(
        targets
            .oracle_guidance_mask
            .expect("bc oracle mask")
            .to_data(),
        split_targets
            .oracle_guidance_mask
            .expect("split oracle mask")
            .to_data()
    );
}

#[test]
fn collate_sample_refs_bc_owned_matches_split_batch_targets_and_exit_surface() {
    let device = Default::default();
    let mut sample = dummy_sample(2, 100);
    sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[2] = 0.75;
    exit_target[45] = -0.25;
    exit_mask[2] = 1.0;
    exit_mask[45] = 1.0;
    sample.exit_target = Some(exit_target);
    sample.exit_mask = Some(exit_mask);
    let samples = [sample];
    let sample_refs: Vec<&MjaiSample> = samples.iter().collect();

    let (obs, bc_batch, targets) = collate_sample_refs_bc_owned::<B>(&sample_refs, false, &device)
        .expect("bc owned ref collate")
        .expect("bc owned ref batch present");
    let (split_obs, split_batch) =
        collate_sample_refs_with_batch::<B>(&sample_refs, false, &device)
            .expect("split ref batch collate")
            .expect("split ref batch present");
    let split_targets = split_batch.to_hydra_targets();

    assert_eq!(obs.to_data(), split_obs.to_data());
    assert_eq!(bc_batch.actions.to_data(), split_batch.actions.to_data());
    assert_eq!(
        bc_batch
            .exit_target
            .as_ref()
            .expect("bc ref exit target")
            .to_data(),
        split_batch
            .exit_target
            .as_ref()
            .expect("split ref exit target")
            .to_data()
    );
    assert_eq!(
        bc_batch
            .exit_mask
            .as_ref()
            .expect("bc ref exit mask")
            .to_data(),
        split_batch
            .exit_mask
            .as_ref()
            .expect("split ref exit mask")
            .to_data()
    );
    assert_eq!(
        targets.policy_target.to_data(),
        split_targets.policy_target.to_data()
    );
    assert_eq!(
        targets.legal_mask.to_data(),
        split_targets.legal_mask.to_data()
    );
    assert_eq!(
        targets.value_target.to_data(),
        split_targets.value_target.to_data()
    );
    assert_eq!(
        targets.grp_target.to_data(),
        split_targets.grp_target.to_data()
    );
    assert_eq!(
        targets
            .oracle_target
            .expect("bc ref oracle target")
            .to_data(),
        split_targets
            .oracle_target
            .expect("split ref oracle target")
            .to_data()
    );
    assert_eq!(
        targets
            .oracle_guidance_mask
            .expect("bc ref oracle mask")
            .to_data(),
        split_targets
            .oracle_guidance_mask
            .expect("split ref oracle mask")
            .to_data()
    );
}

#[cfg(feature = "libtorch")]
#[test]
fn collate_samples_into_host_scratch_matches_bc_owned_without_augmentation() {
    let device = Default::default();
    let mut sample = dummy_sample(2, 100);
    sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[2] = 0.75;
    exit_target[45] = -0.25;
    exit_mask[2] = 1.0;
    exit_mask[45] = 1.0;
    sample.exit_target = Some(exit_target);
    sample.exit_mask = Some(exit_mask);
    let samples = [sample];

    let mut scratch = BcShardHostScratch::new(samples.len(), false, true, false);
    let rows = collate_samples_into_host_scratch(&samples, false, &mut scratch)
        .expect("host scratch collation should succeed")
        .expect("non-empty host scratch batch");
    assert_eq!(rows, samples.len());
    let device_batch =
        crate::epoch_runner::materialize_host_batch_owned::<B>(scratch.take_batch(), &device);

    let (obs, bc_batch, targets) = collate_samples_bc_owned::<B>(&samples, false, &device)
        .expect("bc owned collate")
        .expect("bc owned batch present");

    assert_eq!(device_batch.obs.to_data(), obs.to_data());
    assert_eq!(
        device_batch.batch.actions.to_data(),
        bc_batch.actions.to_data()
    );
    assert_eq!(
        device_batch
            .batch
            .exit_target
            .as_ref()
            .expect("host exit target")
            .to_data(),
        bc_batch
            .exit_target
            .as_ref()
            .expect("bc exit target")
            .to_data()
    );
    assert_eq!(
        device_batch
            .batch
            .exit_mask
            .as_ref()
            .expect("host exit mask")
            .to_data(),
        bc_batch.exit_mask.as_ref().expect("bc exit mask").to_data()
    );
    assert_eq!(
        device_batch.targets.policy_target.to_data(),
        targets.policy_target.to_data()
    );
    assert_eq!(
        device_batch.targets.legal_mask.to_data(),
        targets.legal_mask.to_data()
    );
    assert_eq!(
        device_batch.targets.value_target.to_data(),
        targets.value_target.to_data()
    );
    assert_eq!(
        device_batch.targets.grp_target.to_data(),
        targets.grp_target.to_data()
    );
    assert_eq!(
        device_batch
            .targets
            .oracle_target
            .expect("host oracle")
            .to_data(),
        targets.oracle_target.expect("bc oracle").to_data()
    );
    assert_eq!(
        device_batch
            .targets
            .oracle_guidance_mask
            .expect("host oracle mask")
            .to_data(),
        targets
            .oracle_guidance_mask
            .expect("bc oracle mask")
            .to_data()
    );
}

#[cfg(feature = "libtorch")]
#[test]
fn collate_samples_into_host_scratch_matches_bc_owned_with_augmentation() {
    let device = Default::default();
    let mut sample = dummy_sample(0, 0);
    sample.obs = [0.0; OBS_SIZE];
    sample.obs[40 * 34] = 1.0;
    sample.opp_next = [0, 9, 27];
    sample.danger[0] = 0.25;
    sample.danger[34 + 9] = 0.5;
    sample.danger_mask[18] = 1.0;
    let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual[0] = -0.75;
    safety_residual[1] = 0.4;
    safety_residual_mask[0] = 1.0;
    safety_residual_mask[1] = 1.0;
    sample.safety_residual = Some(safety_residual);
    sample.safety_residual_mask = Some(safety_residual_mask);
    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_target[0] = 0.6;
    delta_q_target[1] = -0.25;
    delta_q_mask[0] = 1.0;
    delta_q_mask[1] = 1.0;
    sample.delta_q_target = Some(delta_q_target);
    sample.delta_q_mask = Some(delta_q_mask);
    let samples = [sample];

    let mut scratch = BcShardHostScratch::new(ALL_PERMUTATIONS.len(), true, false, true);
    let rows = collate_samples_into_host_scratch(&samples, true, &mut scratch)
        .expect("host scratch collation should succeed")
        .expect("non-empty host scratch batch");
    assert_eq!(rows, ALL_PERMUTATIONS.len());
    let device_batch =
        crate::epoch_runner::materialize_host_batch_owned::<B>(scratch.take_batch(), &device);

    let (obs, _bc_batch, targets) = collate_samples_bc_owned::<B>(&samples, true, &device)
        .expect("bc owned collate")
        .expect("bc owned batch present");

    assert_eq!(device_batch.obs.to_data(), obs.to_data());
    assert_eq!(
        device_batch.targets.policy_target.to_data(),
        targets.policy_target.to_data()
    );
    assert_eq!(
        device_batch.targets.legal_mask.to_data(),
        targets.legal_mask.to_data()
    );
    assert_eq!(
        device_batch.targets.danger_target.to_data(),
        targets.danger_target.to_data()
    );
    assert_eq!(
        device_batch.targets.danger_mask.to_data(),
        targets.danger_mask.to_data()
    );
    assert_eq!(
        device_batch.targets.opp_next_target.to_data(),
        targets.opp_next_target.to_data()
    );
    assert_eq!(
        device_batch
            .targets
            .safety_residual_target
            .expect("host safety")
            .to_data(),
        targets.safety_residual_target.expect("bc safety").to_data()
    );
    assert_eq!(
        device_batch
            .targets
            .delta_q_target
            .expect("host delta_q")
            .to_data(),
        targets.delta_q_target.expect("bc delta_q").to_data()
    );
}

#[test]
fn collate_samples_into_host_scratch_rejects_optional_target_mask_mismatch() {
    let mut sample = dummy_sample(0, 0);
    sample.exit_target = Some([0.0; HYDRA_ACTION_SPACE]);
    let mut scratch = BcShardHostScratch::new(1, false, true, false);
    let err = collate_samples_into_host_scratch(&[sample], false, &mut scratch)
        .expect_err("incomplete exit pair should fail");
    assert!(err.contains("exit target/mask mismatch for host scratch collation"));
}

#[test]
fn collate_samples_into_recycled_host_batch_drops_optional_buffers_when_absent() {
    let mut sample = dummy_sample(2, 100);
    let mut safety = [0.0f32; HYDRA_ACTION_SPACE];
    let mut safety_mask = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety[2] = 0.25;
    safety_mask[2] = 1.0;
    exit[2] = 0.5;
    exit_mask[2] = 1.0;
    delta_q[2] = -0.75;
    delta_q_mask[2] = 1.0;
    sample.safety_residual = Some(safety);
    sample.safety_residual_mask = Some(safety_mask);
    sample.exit_target = Some(exit);
    sample.exit_mask = Some(exit_mask);
    sample.delta_q_target = Some(delta_q);
    sample.delta_q_mask = Some(delta_q_mask);

    let recycled =
        collate_samples_into_recycled_host_batch(&[sample], false, BcShardHostBatch::empty())
            .expect("first recycled collation should succeed")
            .expect("first recycled collation should produce a batch");
    assert!(recycled.safety_target_flat.is_some());
    assert!(recycled.safety_mask_flat.is_some());
    assert!(recycled.exit_target_flat.is_some());
    assert!(recycled.exit_mask_flat.is_some());
    assert!(recycled.delta_q_target_flat.is_some());
    assert!(recycled.delta_q_mask_flat.is_some());

    let plain = dummy_sample(3, -100);
    let recycled = collate_samples_into_recycled_host_batch(&[plain], false, recycled)
        .expect("second recycled collation should succeed")
        .expect("second recycled collation should produce a batch");

    assert_eq!(recycled.batch_size, 1);
    assert!(recycled.safety_target_flat.is_none());
    assert!(recycled.safety_mask_flat.is_none());
    assert!(recycled.exit_target_flat.is_none());
    assert!(recycled.exit_mask_flat.is_none());
    assert!(recycled.delta_q_target_flat.is_none());
    assert!(recycled.delta_q_mask_flat.is_none());
}

#[test]
fn collate_samples_into_recycled_host_batch_zeroes_optional_rows_and_tail_capacity() {
    let mut with_exit = dummy_sample(2, 100);
    let mut exit = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit[2] = 0.5;
    exit_mask[2] = 1.0;
    with_exit.exit_target = Some(exit);
    with_exit.exit_mask = Some(exit_mask);
    let mut without_exit = dummy_sample(3, -100);
    without_exit.exit_target = None;
    without_exit.exit_mask = None;

    let recycled = collate_samples_into_recycled_host_batch(
        &[with_exit.clone(), without_exit.clone()],
        false,
        BcShardHostBatch::empty(),
    )
    .expect("full recycled collation should succeed")
    .expect("full recycled collation should produce a batch");
    assert_eq!(recycled.batch_size, 2);
    let capacity = recycled
        .exit_target_flat
        .as_ref()
        .expect("exit target buffer")
        .capacity();

    let recycled =
        collate_samples_into_recycled_host_batch(&[without_exit, with_exit], false, recycled)
            .expect("tail recycled collation should succeed")
            .expect("tail recycled collation should produce a batch");

    assert_eq!(recycled.batch_size, 2);
    let exit = recycled
        .exit_target_flat
        .as_ref()
        .expect("exit target buffer");
    let exit_mask = recycled.exit_mask_flat.as_ref().expect("exit mask buffer");
    assert!(exit.capacity() >= capacity);
    assert!(exit[..HYDRA_ACTION_SPACE].iter().all(|&value| value == 0.0));
    assert!(
        exit_mask[..HYDRA_ACTION_SPACE]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_eq!(exit[HYDRA_ACTION_SPACE + 2], 0.5);
    assert_eq!(exit_mask[HYDRA_ACTION_SPACE + 2], 1.0);

    let recycled = collate_samples_into_recycled_host_batch(&[dummy_sample(4, 0)], false, recycled)
        .expect("smaller tail recycled collation should succeed")
        .expect("smaller tail recycled collation should produce a batch");
    assert_eq!(recycled.batch_size, 1);
    assert!(recycled.exit_target_flat.is_none());
    assert!(recycled.exit_mask_flat.is_none());
}

#[test]
fn collate_samples_into_recycled_host_batch_matches_augmented_row_order() {
    let mut sample = dummy_sample(0, 0);
    sample.obs = [0.0; OBS_SIZE];
    sample.obs[40 * 34] = 1.0;
    sample.opp_next = [0, 9, 27];
    sample.danger[0] = 0.25;
    sample.danger[34 + 9] = 0.5;
    sample.danger_mask[18] = 1.0;

    let augmented = augment_samples_6x(&[sample.clone()]);
    let recycled =
        collate_samples_into_recycled_host_batch(&[sample], true, BcShardHostBatch::empty())
            .expect("augmented recycled collation should succeed")
            .expect("augmented recycled collation should produce a batch");

    assert_eq!(recycled.batch_size, ALL_PERMUTATIONS.len());
    for (index, expected) in augmented.iter().enumerate() {
        assert_eq!(recycled.actions[index], expected.action as i64);
        assert_eq!(
            &recycled.obs_flat[index * OBS_SIZE..(index + 1) * OBS_SIZE],
            expected.obs.as_slice()
        );
        assert_eq!(
            &recycled.legal_mask_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE],
            expected.legal_mask.as_slice()
        );
        assert_eq!(
            &recycled.danger_flat[index * SPATIAL_TARGET_SIZE..(index + 1) * SPATIAL_TARGET_SIZE],
            expected.danger.as_slice()
        );
        assert_eq!(
            &recycled.danger_mask_flat
                [index * SPATIAL_TARGET_SIZE..(index + 1) * SPATIAL_TARGET_SIZE],
            expected.danger_mask.as_slice()
        );
    }
}

#[test]
fn collate_samples_into_recycled_host_batch_returns_none_for_empty_input() {
    let recycled = collate_samples_into_recycled_host_batch(&[], false, BcShardHostBatch::empty())
        .expect("empty recycled collation should succeed");
    assert!(recycled.is_none());
}
