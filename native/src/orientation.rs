//! A FixTransform's rotation, up and right (FixTransform.cs:36-73). They are
//! one orientation: each property's setter rebuilds the other two, so the
//! fields stay private and every write goes through one of those setters.
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orientation {
    rotation: Fix,
    up: Vec2,
}

impl Default for Orientation {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Orientation {
    /// `rotation = Fix.Zero`: Vec2(0) is exactly (1, 0).
    pub const ZERO: Self = Self { rotation: Fix::ZERO, up: Vec2::UP };

    pub fn rotation(self) -> Fix {
        self.rotation
    }
    pub fn up(self) -> Vec2 {
        self.up
    }
    pub fn right(self) -> Vec2 {
        self.up.perp_cw()
    }
    /// The rotation setter: right = new Vec2(value), up = MulByI(right).
    pub fn set_rotation(&mut self, rotation: Fix) {
        self.rotation = rotation;
        self.up = Vec2::angle(rotation).perp();
    }
    /// The up setter: right = MulByNegI(up), rotation its angle.
    pub fn set_up(&mut self, up: Vec2) {
        self.set_up_with_angle(up, up.perp_cw().normalized_angle());
    }
    /// The up setter with `NormalizedVectorAngle(MulByNegI(up))` already
    /// evaluated by the caller.
    pub fn set_up_with_angle(&mut self, up: Vec2, rotation: Fix) {
        self.up = up;
        self.rotation = rotation;
    }
    /// The right setter: up = MulByI(right), rotation its angle.
    pub fn set_right(&mut self, right: Vec2) {
        self.up = right.perp();
        self.rotation = right.normalized_angle();
    }
}
