//! Unity Color conversion and DetPhysics.BeamColors.Mix, preserving f32 bits.
//!
//! The pinned Mono runtime evaluates float expressions with double intermediates
//! but rounds at float locals, out parameters and fields. Explicit boundaries
//! below match the actual Unity color probe; f32-only arithmetic does not.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Colors {
    pub white: [u32; 4],
    pub yellow: [u32; 4],
    pub orange: [u32; 4],
    pub red: [u32; 4],
}
fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
fn hsv(c: [u32; 4]) -> [f32; 3] {
    let [r, g, b, _] = c.map(f32::from_bits);
    let (offset, v, one, two) = if b > g && b > r {
        (4.0, b, r, g)
    } else if g > r {
        (2.0, g, b, r)
    } else {
        (0.0, r, g, b)
    };
    if v == 0.0 {
        return [0.0, 0.0, v];
    }
    let delta = v - min(one, two);
    let (mut h, s) = if delta != 0.0 {
        (
            (offset + (one as f64 - two as f64) / delta as f64) as f32,
            delta / v,
        )
    } else {
        ((offset + (one as f64 - two as f64)) as f32, 0.0)
    };
    h /= 6.0;
    if h < 0.0 {
        h += 1.0;
    }
    [h, s, v]
}
fn rgb([h, s, v]: [f32; 3]) -> [u32; 4] {
    let color = if s == 0.0 {
        [v, v, v, 1.0]
    } else if v == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        let sector = h * 6.0;
        let index = sector.floor() as i32;
        let part = sector - index as f32;
        let low = (v as f64 * (1.0 - s as f64)) as f32;
        let falling = (v as f64 * (1.0 - s as f64 * part as f64)) as f32;
        let rising = (v as f64 * (1.0 - s as f64 * (1.0 - part as f64))) as f32;
        match index {
            0 | 6 => [v, rising, low, 1.0],
            1 => [falling, v, low, 1.0],
            2 => [low, v, rising, 1.0],
            3 => [low, falling, v, 1.0],
            4 => [rising, low, v, 1.0],
            5 | -1 => [v, low, falling, 1.0],
            _ => [0.0, 0.0, 0.0, 1.0],
        }
    };
    color.map(f32::to_bits)
}
fn mix(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let a = hsv(a);
    let b = hsv(b);
    rgb([(a[0] + b[0]) / 2.0, max(a[1], b[1]), max(a[2], b[2]) + 0.1])
}
impl Colors {
    pub fn mix(self, other: Self) -> Self {
        let a = self.white.map(f32::from_bits);
        let b = other.white.map(f32::from_bits);
        Self {
            white: [
                min(a[0], b[0]),
                min(a[1], b[1]),
                min(a[2], b[2]),
                max(a[3], b[3]),
            ]
            .map(f32::to_bits),
            yellow: mix(self.yellow, other.yellow),
            orange: mix(self.orange, other.orange),
            red: mix(self.red, other.red),
        }
    }
}
