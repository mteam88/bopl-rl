//! Collision queries translated from Raycast.cs, RoundedRect.cs and PhysTools.cs.
use crate::{
    fixed::{Fix, MathError},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Circle {
    pub center: Vec2,
    pub radius: Fix,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct OrientedBox {
    pub center: Vec2,
    /// Scaled basis vectors, not unit vectors.
    pub right: Vec2,
    pub up: Vec2,
    pub inverse_extents: Vec2,
}

impl OrientedBox {
    /// A box the way DetPhysics holds one after `Box.SetRotation`
    /// (Box.cs:31-35, RoundedRect.cs:22-26): `inverseExtents = One / extents`
    /// and the scaled axes `ComplexMul(Vec2.right / inverseExtents.x, basis)`,
    /// `ComplexMul(Vec2.up / inverseExtents.y, basis)`. In Q32.32 the axis
    /// length is `One / (One / e)` rather than `e`, and the `up` products are
    /// negated after flooring; `basis * e` differs from both by one raw unit.
    pub fn from_basis(center: Vec2, basis: Vec2, extents: Vec2) -> Self {
        let inverse_extents = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
        Self {
            center,
            right: (Vec2::RIGHT / inverse_extents.x).complex_mul(basis),
            up: (Vec2::UP / inverse_extents.y).complex_mul(basis),
            inverse_extents,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct RoundedRect {
    pub inner: OrientedBox,
    pub radius: Fix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RayHit {
    pub near: Fix,
    pub far: Fix,
}

impl Circle {
    /// DetPhysics' sort-and-sweep pairing of two circles' bounding boxes.
    pub fn bounds_overlap(self, other: Circle) -> bool {
        let reach = self.radius + other.radius;
        (self.center.x - other.center.x).abs() < reach
            && (self.center.y - other.center.y).abs() < reach
    }
    pub fn overlaps(self, point: Vec2) -> bool {
        (self.center - point).length_squared() < self.radius * self.radius
    }
    pub fn raycast(self, origin: Vec2, direction: Vec2) -> Option<RayHit> {
        let center = self.center - origin;
        let projected = center.dot(direction);
        let radius_squared = self.radius * self.radius;
        let inside = radius_squared > center.length_squared();
        // The reference (Raycast.cs circle case) rejects a centre behind the
        // origin after the discriminant; the test does not depend on it, so
        // rejecting first skips the square root for the common miss.
        if projected < Fix::ZERO && !inside {
            return None;
        }
        let discriminant = radius_squared - (direction * projected - center).length_squared();
        if discriminant < Fix::ZERO {
            return None;
        }
        let offset = discriminant.sqrt().unwrap();
        if inside {
            Some(RayHit {
                near: Fix::ZERO,
                far: projected + offset,
            })
        } else {
            Some(RayHit {
                near: projected - offset,
                far: projected + offset,
            })
        }
    }
}

/// Axis-aligned bounds; every point of the bounded shape lies inside.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min: Vec2,
    pub max: Vec2,
}

impl Bounds {
    /// Whether the segment `origin + direction * t`, `0 <= t <= distance`, can
    /// touch the bounds: a conservative separating-axis test on each axis
    /// with a 1e-6 unit margin, so a hit point that the narrow phase computes
    /// with rounding error (~1e-14 units at scene scale) is never rejected.
    /// The segment's endpoints are computed once per query by the caller.
    pub fn may_reach(self, origin: Vec2, end: Vec2) -> bool {
        let margin = Fix::from_f64(1e-6);
        origin.x.min(end.x) <= self.max.x + margin
            && origin.x.max(end.x) >= self.min.x - margin
            && origin.y.min(end.y) <= self.max.y + margin
            && origin.y.max(end.y) >= self.min.y - margin
    }
}

impl OrientedBox {
    pub fn overlaps(self, point: Vec2) -> bool {
        let unit_up = self.up * self.inverse_extents.y;
        let unit_right = self.right * self.inverse_extents.x;
        let x = (self.center - point).dot(unit_right) * self.inverse_extents.x;
        let y = (self.center - point).dot(unit_up) * self.inverse_extents.y;
        x.abs() <= Fix::ONE && y.abs() <= Fix::ONE
    }
    pub fn raycast(self, origin: Vec2, direction: Vec2) -> Option<RayHit> {
        let hits = edge_hits(self, Fix::ZERO, origin, direction);
        match hits.as_slice() {
            [] => None,
            [one] => Some(RayHit {
                near: Fix::ZERO,
                far: *one,
            }),
            [one, two, ..] => Some(RayHit {
                near: (*one).min(*two),
                far: (*one).max(*two),
            }),
        }
    }
    /// Conservative axis-aligned bounds of the box grown by `radius`: the
    /// projected half extents of the (scaled) basis vectors plus the radius.
    pub fn bounds(self, radius: Fix) -> Bounds {
        let extent = Vec2::new(
            self.right.x.abs() + self.up.x.abs() + radius,
            self.right.y.abs() + self.up.y.abs() + radius,
        );
        Bounds {
            min: self.center - extent,
            max: self.center + extent,
        }
    }
    /// `raycast(origin, direction).filter(|hit| hit.near <= distance)` with the
    /// broad phase of `RoundedRect::raycast_within`: two edge hits put `near`
    /// on an edge; a single edge hit means the origin is inside the box and
    /// reports `near == 0` there. Both points are inside `bounds(0)`. The one
    /// exception is a ray from outside whose entry is exactly a corner (the
    /// strict inequalities drop that edge and the reference still reports
    /// `near == 0`); that is an exact-equality tie and is not preserved.
    pub fn raycast_within(self, origin: Vec2, direction: Vec2, distance: Fix) -> Option<RayHit> {
        if !self
            .bounds(Fix::ZERO)
            .may_reach(origin, origin + direction * distance)
        {
            return None;
        }
        self.raycast(origin, direction)
            .filter(|hit| hit.near <= distance)
    }
}

/// Up to two edge distances in side order (top, right, bottom, left), the
/// reference's `Raycast.cs` box loop stops after the second.
#[derive(Clone, Copy, Debug, Default)]
struct EdgeHits {
    hits: [Fix; 2],
    count: usize,
}

impl EdgeHits {
    fn push(&mut self, distance: Fix) {
        self.hits[self.count] = distance;
        self.count += 1;
    }
    fn as_slice(&self) -> &[Fix] {
        &self.hits[..self.count]
    }
}

// Preserve side order, strict inequalities, and the early return after two edges.
// The reference does not use a conventional slab raycast. Each side's endpoints
// are the same expressions as before (corner +- unit * radius), evaluated only
// for the sides that are tested, into a fixed-size result (no allocation).
fn edge_hits(b: OrientedBox, radius: Fix, origin: Vec2, direction: Vec2) -> EdgeHits {
    let c = b.center - origin;
    let unit_right = b.right * b.inverse_extents.x;
    let unit_down = unit_right.perp_cw();
    let horizontal_cross = unit_right.cross(direction);
    let vertical_cross = unit_down.cross(direction);
    let mut hits = EdgeHits::default();
    if horizontal_cross == Fix::ZERO && vertical_cross == Fix::ZERO {
        return hits;
    }
    let tl = c - b.right + b.up;
    let tr = c + b.right + b.up;
    let br = c + b.right - b.up;
    let bl = c - b.right - b.up;
    let down_radius = unit_down * radius;
    let right_radius = unit_right * radius;
    for side in 0..4 {
        // (a, b, tangent, normal, cross, use_end, negate)
        let (a, b, tangent, normal, cross, use_end, negate) = match side {
            0 => (
                tl - down_radius,
                tr - down_radius,
                unit_right,
                unit_down,
                horizontal_cross,
                true,
                true,
            ),
            1 => (
                tr + right_radius,
                br + right_radius,
                unit_down,
                unit_right,
                vertical_cross,
                false,
                false,
            ),
            2 => (
                br + down_radius,
                bl + down_radius,
                -unit_right,
                unit_down,
                horizontal_cross,
                false,
                true,
            ),
            _ => (
                bl - right_radius,
                tl - right_radius,
                -unit_down,
                unit_right,
                vertical_cross,
                true,
                false,
            ),
        };
        if cross == Fix::ZERO {
            continue;
        }
        let numerator = (if use_end { b } else { a }).dot(normal);
        let distance = (if negate { -numerator } else { numerator }) / cross;
        let projected = (direction * distance).dot(tangent);
        if distance > Fix::ZERO && projected > a.dot(tangent) && projected < b.dot(tangent) {
            hits.push(distance);
            if hits.count == 2 {
                break;
            }
        }
    }
    hits
}

impl RoundedRect {
    /// Conservative axis-aligned bounds of the rounded rectangle: the box's
    /// projected half extents plus the corner radius on each axis.
    pub fn bounds(self) -> Bounds {
        self.inner.bounds(self.radius)
    }
    /// `raycast(origin, direction).filter(|hit| hit.near <= distance)` with a
    /// broad phase: a hit with `near <= distance` puts `origin + direction *
    /// near` on the shape's boundary (edge or corner arc) or at the origin
    /// itself (`near == 0` when the origin overlaps), and `near >= 0`, so that
    /// point lies on the segment and inside `bounds()`; when the segment cannot
    /// reach the bounds the narrow phase is skipped.
    pub fn raycast_within(self, origin: Vec2, direction: Vec2, distance: Fix) -> Option<RayHit> {
        if !self
            .bounds()
            .may_reach(origin, origin + direction * distance)
        {
            return None;
        }
        self.raycast(origin, direction)
            .filter(|hit| hit.near <= distance)
    }
    pub fn overlaps(self, point: Vec2) -> bool {
        let b = self.inner;
        let unit_up = b.up * b.inverse_extents.y;
        let unit_right = b.right * b.inverse_extents.x;
        let max_y = Fix::ONE + self.radius * b.inverse_extents.y;
        let max_x = Fix::ONE + self.radius * b.inverse_extents.x;
        let x = ((b.center - point).dot(unit_right) * b.inverse_extents.x).abs();
        let y = ((b.center - point).dot(unit_up) * b.inverse_extents.y).abs();
        if x > max_x || y > max_y {
            return false;
        }
        if (x <= max_x && y <= Fix::ONE) || (x <= Fix::ONE && y <= max_y) {
            return true;
        }
        let radius_squared = self.radius * self.radius;
        self.corners()
            .iter()
            .any(|corner| (*corner - point).length_squared() < radius_squared)
    }
    pub fn corners(self) -> [Vec2; 4] {
        let b = self.inner;
        [
            b.center - b.right + b.up,
            b.center + b.right + b.up,
            b.center - b.right - b.up,
            b.center + b.right - b.up,
        ]
    }
    pub fn raycast(self, origin: Vec2, direction: Vec2) -> Option<RayHit> {
        let edges = edge_hits(self.inner, self.radius, origin, direction);
        let edges = edges.as_slice();
        if edges.len() == 2 {
            return Some(RayHit {
                near: edges[0].min(edges[1]),
                far: edges[0].max(edges[1]),
            });
        }
        let mut near = edges.first().copied().unwrap_or(Fix::MAX);
        let mut far = edges.first().copied().unwrap_or(Fix::from_raw(1));
        let mut found = !edges.is_empty();
        for center in self.corners() {
            if let Some(hit) = (Circle {
                center: center - origin,
                radius: self.radius,
            })
            .raycast(Vec2::ZERO, direction)
            {
                near = near.min(hit.near);
                far = far.max(hit.far);
                found = true;
            }
        }
        if self.overlaps(origin) {
            near = Fix::ZERO;
            found = true;
        }
        found.then_some(RayHit { near, far })
    }
    pub fn closest_point(self, point: Vec2) -> Result<Vec2, MathError> {
        let b = self.inner;
        let unit_up = b.up * b.inverse_extents.y;
        let unit_right = b.right * b.inverse_extents.x;
        let extent_x_squared = b.right.dot(b.right);
        let extent_y_squared = b.up.dot(b.up);
        let x = (point - b.center).dot(unit_right);
        let y = (point - b.center).dot(unit_up);
        let signed_x_squared = Fix::from_int(x.sign2()) * x * x;
        let signed_y_squared = Fix::from_int(y.sign2()) * y * y;
        let corner_point = |corner: Vec2| -> Result<Vec2, MathError> {
            let delta = point - corner;
            let length = delta.length_squared().sqrt()?;
            let normal = Vec2::new(delta.x.checked_div(length)?, delta.y.checked_div(length)?);
            Ok(corner + normal * self.radius)
        };
        if signed_x_squared <= -extent_x_squared {
            if signed_y_squared >= extent_y_squared {
                let corner = b.center - b.right + b.up;
                return corner_point(corner);
            }
            if signed_y_squared <= -extent_y_squared {
                let corner = b.center - b.right - b.up;
                return corner_point(corner);
            }
            return Ok(b.center - b.right - unit_right * self.radius + unit_up * y);
        }
        if signed_x_squared >= extent_x_squared {
            if signed_y_squared >= extent_y_squared {
                let corner = b.center + b.right + b.up;
                return corner_point(corner);
            }
            if signed_y_squared <= -extent_y_squared {
                let corner = b.center + b.right - b.up;
                return corner_point(corner);
            }
            return Ok(b.center + b.right + unit_right * self.radius + unit_up * y);
        }
        Ok(if y >= Fix::ZERO {
            b.center + b.up + unit_up * self.radius + unit_right * x
        } else {
            b.center - b.up - unit_up * self.radius + unit_right * x
        })
    }
}
