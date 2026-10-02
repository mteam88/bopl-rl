//! Shapes retain scaled bases and the source's division order when rotating.
use crate::{
    fixed::Fix,
    geometry::{Circle, OrientedBox, RoundedRect},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "shape", rename_all = "snake_case")]
pub enum Collider {
    Circle(Circle),
    Box(OrientedBox),
    RoundedRect(RoundedRect),
}

impl Collider {
    pub fn kind(self) -> usize {
        match self {
            Self::Circle(_) => 0,
            Self::Box(_) => 1,
            Self::RoundedRect(_) => 2,
        }
    }
    pub fn center(self) -> Vec2 {
        match self {
            Self::Circle(c) => c.center,
            Self::Box(b) => b.center,
            Self::RoundedRect(r) => r.inner.center,
        }
    }
    pub fn set_center(&mut self, p: Vec2) {
        match self {
            Self::Circle(c) => c.center = p,
            Self::Box(b) => b.center = p,
            Self::RoundedRect(r) => r.inner.center = p,
        }
    }
    /// IShape.OverlapPoint (PhysTools.cs:1083-1120).
    pub fn overlaps(self, point: Vec2) -> bool {
        match self {
            Self::Circle(c) => c.overlaps(point),
            Self::Box(b) => b.overlaps(point),
            Self::RoundedRect(r) => r.overlaps(point),
        }
    }
    pub fn translate(&mut self, delta: Vec2) {
        self.set_center(self.center() + delta);
    }
    pub fn rotate(&mut self, angle: Fix) {
        let b = match self {
            Self::Circle(_) => return,
            Self::Box(b) => b,
            Self::RoundedRect(r) => &mut r.inner,
        };
        let basis = Vec2::angle(angle);
        b.right = (Vec2::RIGHT / b.inverse_extents.x).complex_mul(basis);
        b.up = (Vec2::UP / b.inverse_extents.y).complex_mul(basis);
    }
    pub fn bounds(self) -> [Vec2; 2] {
        let (center, extent) = match self {
            Self::Circle(c) => (c.center, Vec2::new(c.radius, c.radius)),
            Self::Box(b) => (
                b.center,
                Vec2::new(
                    b.right.x.abs() + b.up.x.abs(),
                    b.right.y.abs() + b.up.y.abs(),
                ),
            ),
            Self::RoundedRect(r) => (
                r.inner.center,
                Vec2::new(
                    r.inner.right.x.abs() + r.inner.up.x.abs() + r.radius,
                    r.inner.right.y.abs() + r.inner.up.y.abs() + r.radius,
                ),
            ),
        };
        [center - extent, center + extent]
    }
    /// Caller orders heterogeneous pairs by Circle, Box, RoundedRect.
    pub fn contact(self, other: Self) -> Option<super::Contact> {
        use super::contact::*;
        match (self, other) {
            (Self::Circle(a), Self::Circle(b)) => circle_circle(a, b),
            (Self::Circle(a), Self::Box(b)) => circle_box(a, b),
            (Self::Circle(a), Self::RoundedRect(b)) => circle_rounded(a, b),
            (Self::Box(a), Self::Box(b)) => box_box(a, b),
            (Self::Box(a), Self::RoundedRect(b)) => rounded_rounded(
                RoundedRect {
                    inner: a,
                    radius: Fix::ZERO,
                },
                b,
            ),
            (Self::RoundedRect(a), Self::RoundedRect(b)) => rounded_rounded(a, b),
            _ => panic!("unordered collider pair"),
        }
    }
}
