//! The pinned Unity.Mathematics RNG and Updater's fixed-point range sampler.
use crate::fixed::Fix;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Random {
    pub state: u32,
}
impl Random {
    pub fn new(seed: u32) -> Self {
        let mut rng = Self { state: seed };
        rng.next_state();
        rng
    }
    fn next_state(&mut self) -> u32 {
        let previous = self.state;
        self.state ^= self.state.wrapping_shl(13);
        self.state ^= self.state >> 17;
        self.state ^= self.state.wrapping_shl(5);
        previous
    }
    pub fn uint_range(&mut self, min: u32, max: u32) -> u32 {
        let width = max.wrapping_sub(min);
        ((u64::from(self.next_state()) * u64::from(width)) >> 32).wrapping_add(u64::from(min))
            as u32
    }
    pub fn int_range(&mut self, min: i32, max: i32) -> i32 {
        let width = max.wrapping_sub(min) as u32;
        (((u64::from(self.next_state()) * u64::from(width)) >> 32) as i32).wrapping_add(min)
    }
    pub fn fix_range(&mut self, min: Fix, max: Fix) -> Fix {
        if min.floor() == max.floor() {
            let fraction = self.uint_range(min.raw() as u32, max.raw() as u32);
            return Fix::from_raw(min.floor().raw().wrapping_add(i64::from(fraction)));
        }
        let integer =
            i64::from(self.int_range(min.floor().to_int() as i32, max.floor().to_int() as i32));
        let fraction = self.uint_range(min.raw() as u32, u32::MAX);
        Fix::from_raw(integer.wrapping_shl(32).wrapping_add(i64::from(fraction)))
    }
}
