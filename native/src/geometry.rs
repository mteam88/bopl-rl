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
    pub fn overlaps(self, point: Vec2) -> bool {
        (self.center - point).length_squared() < self.radius * self.radius
    }
    pub fn raycast(self, origin: Vec2, direction: Vec2) -> Option<RayHit> {
        let center = self.center - origin;
        let projected = center.dot(direction);
        let radius_squared = self.radius * self.radius;
        let discriminant = radius_squared - (direction * projected - center).length_squared();
        if discriminant < Fix::ZERO {
            return None;
        }
        let offset = discriminant.sqrt().unwrap();
        if radius_squared > center.length_squared() {
            Some(RayHit {
                near: Fix::ZERO,
                far: projected + offset,
            })
        } else if projected < Fix::ZERO {
            None
        } else {
            Some(RayHit {
                near: projected - offset,
                far: projected + offset,
            })
        }
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
}

// Preserve side order, strict inequalities, and the early return after two edges.
// The reference does not use a conventional slab raycast.
fn edge_hits(b: OrientedBox, radius: Fix, origin: Vec2, direction: Vec2) -> Vec<Fix> {
    let c = b.center - origin;
    let unit_right = b.right * b.inverse_extents.x;
    let unit_down = unit_right.perp_cw();
    let horizontal_cross = unit_right.cross(direction);
    let vertical_cross = unit_down.cross(direction);
    let tl = c - b.right + b.up;
    let tr = c + b.right + b.up;
    let br = c + b.right - b.up;
    let bl = c - b.right - b.up;
    let mut hits = Vec::with_capacity(2);
    let sides = [
        (
            tl - unit_down * radius,
            tr - unit_down * radius,
            unit_right,
            unit_down,
            horizontal_cross,
            true,
            true,
        ),
        (
            tr + unit_right * radius,
            br + unit_right * radius,
            unit_down,
            unit_right,
            vertical_cross,
            false,
            false,
        ),
        (
            br + unit_down * radius,
            bl + unit_down * radius,
            -unit_right,
            unit_down,
            horizontal_cross,
            false,
            true,
        ),
        (
            bl - unit_right * radius,
            tl - unit_right * radius,
            -unit_down,
            unit_right,
            vertical_cross,
            true,
            false,
        ),
    ];
    for (a, b, tangent, normal, cross, use_end, negate) in sides {
        if cross == Fix::ZERO {
            continue;
        }
        let numerator = (if use_end { b } else { a }).dot(normal);
        let distance = (if negate { -numerator } else { numerator }) / cross;
        let projected = (direction * distance).dot(tangent);
        if distance > Fix::ZERO && projected > a.dot(tangent) && projected < b.dot(tangent) {
            hits.push(distance);
            if hits.len() == 2 {
                break;
            }
        }
    }
    hits
}

impl RoundedRect {
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
        if edges.len() == 2 {
            return Some(RayHit {
                near: edges[0].min(edges[1]),
                far: edges[0].max(edges[1]),
            });
        }
        let mut near = edges.first().copied().unwrap_or(Fix::MAX);
        let mut far = edges.first().copied().unwrap_or(Fix(1));
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
