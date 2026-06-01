//! Tile representation and suit permutation for data augmentation.
//!
//! Provides the 34-tile type system, aka-dora handling, 136-format conversion,
//! and suit permutation (6 permutations of manzu/pinzu/souzu) used to 6x
//! training data without changing game semantics.

mod constants;
mod format136;
mod kind;
mod permute;

pub use constants::*;
pub use format136::*;
pub use kind::*;
pub use permute::*;

#[cfg(test)]
mod tests;
