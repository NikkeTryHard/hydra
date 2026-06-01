use super::tree::NodeIdx;

const OBS_SIZE: usize = 192 * 34;

pub const MIN_BATCH: usize = 32;

pub struct LeafBatch {
    pub obs_buffer: Vec<f32>,
    pub node_indices: Vec<NodeIdx>,
    pub batch_size: usize,
}

impl LeafBatch {
    pub fn new() -> Self {
        Self::with_capacity(MIN_BATCH)
    }

    pub fn with_capacity(batch_capacity: usize) -> Self {
        Self {
            obs_buffer: Vec::with_capacity(batch_capacity * OBS_SIZE),
            node_indices: Vec::with_capacity(batch_capacity),
            batch_size: 0,
        }
    }

    pub fn clear(&mut self) {
        self.obs_buffer.clear();
        self.node_indices.clear();
        self.batch_size = 0;
    }

    pub fn add(&mut self, obs: &[f32], node_idx: NodeIdx) {
        assert_eq!(
            obs.len(),
            OBS_SIZE,
            "leaf observation must have OBS_SIZE elements"
        );
        self.obs_buffer.extend_from_slice(obs);
        self.node_indices.push(node_idx);
        self.batch_size += 1;
    }

    pub fn is_ready(&self) -> bool {
        self.batch_size >= MIN_BATCH
    }

    pub fn len(&self) -> usize {
        self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    pub fn capacity(&self) -> usize {
        self.node_indices.capacity()
    }
}

impl Default for LeafBatch {
    fn default() -> Self {
        Self::new()
    }
}
