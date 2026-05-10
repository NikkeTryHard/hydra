use std::io;

use crate::data::sample::MjaiSample;

/// Groups validation sample buffers into fixed-size microbatches without
/// materializing the full validation split.
pub struct StreamValMicrobatchIterator<I> {
    inner: I,
    microbatch_size: usize,
    current: Vec<MjaiSample>,
    pending: Vec<MjaiSample>,
    pending_start: usize,
    exhausted: bool,
}

impl<I> StreamValMicrobatchIterator<I>
where
    I: Iterator<Item = io::Result<Vec<MjaiSample>>>,
{
    pub fn new(inner: I, microbatch_size: usize) -> Self {
        let microbatch_size = microbatch_size.max(1);
        Self {
            inner,
            microbatch_size,
            current: Vec::with_capacity(microbatch_size),
            pending: Vec::new(),
            pending_start: 0,
            exhausted: false,
        }
    }

    fn take_full_batch(&mut self) -> Option<Vec<MjaiSample>> {
        let pending_len = self.pending.len().saturating_sub(self.pending_start);
        if pending_len < self.microbatch_size {
            return None;
        }
        if self.pending_start > 0
            && (self.pending_start >= self.microbatch_size
                || self.pending_start * 2 >= self.pending.len())
        {
            self.pending.drain(..self.pending_start);
            self.pending_start = 0;
        }
        self.current.clear();
        self.current.extend(
            self.pending
                .drain(self.pending_start..self.pending_start + self.microbatch_size),
        );
        Some(std::mem::take(&mut self.current))
    }
}

impl<I> Iterator for StreamValMicrobatchIterator<I>
where
    I: Iterator<Item = io::Result<Vec<MjaiSample>>>,
{
    type Item = io::Result<Vec<MjaiSample>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = self.take_full_batch() {
                return Some(Ok(batch));
            }

            if self.exhausted {
                if self.pending_start >= self.pending.len() {
                    return None;
                }
                self.current.clear();
                self.current
                    .extend(self.pending.drain(self.pending_start..));
                self.pending.clear();
                self.pending_start = 0;
                return Some(Ok(std::mem::take(&mut self.current)));
            }

            match self.inner.next() {
                Some(Ok(samples)) => {
                    if !samples.is_empty() {
                        self.pending.extend(samples);
                    }
                }
                Some(Err(err)) => return Some(Err(err)),
                None => self.exhausted = true,
            }
        }
    }
}
