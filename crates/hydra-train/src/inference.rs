//! Compatibility re-export for model inference utilities.
//!
//! Inference now lives in `hydra_model::inference`; this module preserves the
//! historical `hydra_train::inference::*` path.

pub use hydra_model::inference::*;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::*;
    use hydra_core::action::HYDRA_ACTION_SPACE;

    #[test]
    fn old_inference_path_reexports_model_inference() {
        let device = Default::default();
        let logits = Tensor::<NdArray<f32>, 2>::zeros([1, HYDRA_ACTION_SPACE], &device);
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[7] = true;

        let (action, policy) = infer_action(logits, &legal);

        assert_eq!(action, 7);
        assert_eq!(policy[7], 1.0);
    }
}
