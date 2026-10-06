//! The perimeter coordinates used when a slime clings to a rounded platform.
//! Surface geometry uses the platform's unit basis, separately from collider axes.
use crate::{
    fixed::{Fix, MathError},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Surface {
    pub center: Vec2,
    pub right: Vec2,
    pub up: Vec2,
    pub extents: Vec2,
    /// The DPhysics shape's live radius.
    pub radius: Fix,
    /// The radius StickyRoundedRectangle.UpdateBounds last read. Its cached
    /// segment lengths use it, like `extents`, until the next UpdateBounds.
    pub bounds_radius: Fix,
}

impl Surface {
    fn segments(&self) -> [Fix; 8] {
        let width = self.extents.x * Fix::from_int(2);
        let height = self.extents.y * Fix::from_int(2);
        let corner = self.bounds_radius * Fix::HALF_PI;
        [width, corner, height, corner, width, corner, height, corner]
    }
    pub fn circumference(&self) -> Fix {
        let segments = self.segments();
        segments[0] * Fix::from_int(2)
            + segments[1] * Fix::from_int(4)
            + segments[2] * Fix::from_int(2)
    }
    pub fn normal_and_zone(&self, point: Vec2) -> (Vec2, usize) {
        let x = (point - self.center).dot(self.right);
        let y = (point - self.center).dot(self.up);
        let up = self.up * self.extents.y;
        let right = self.right * self.extents.x;
        if x <= -self.extents.x {
            if y >= self.extents.y {
                return ((point - (self.center - right + up)).normalized_safe(), 7);
            }
            if y <= -self.extents.y {
                return ((point - (self.center - right - up)).normalized_safe(), 5);
            }
            return (-self.right, 6);
        }
        if x >= self.extents.x {
            if y >= self.extents.y {
                return ((point - (self.center + right + up)).normalized_safe(), 1);
            }
            if y <= -self.extents.y {
                return ((point - (self.center + right - up)).normalized_safe(), 3);
            }
            return (self.right, 2);
        }
        if y >= Fix::ZERO {
            (self.up, 0)
        } else {
            (-self.up, 4)
        }
    }
    pub fn normal(&self, point: Vec2) -> Vec2 {
        self.normal_and_zone(point).0
    }
    pub fn forward(&self, point: Vec2) -> Vec2 {
        self.normal(point).perp_cw()
    }
    pub fn perimeter_zone(&self, local: Fix, player_radius: Fix) -> Option<(usize, Fix)> {
        let distance = local * (self.circumference() + player_radius * Fix::TAU);
        let mut total = Fix::ZERO;
        for (zone, segment) in self.segments().iter().enumerate() {
            let progress = distance - total;
            total += *segment;
            if zone % 2 == 1 {
                total += player_radius * Fix::HALF_PI;
            }
            if distance >= total {
                continue;
            }
            return Some((zone, progress));
        }
        None
    }
    pub fn position(&self, local: Fix, player_radius: Fix) -> Vec2 {
        let Some((zone, progress)) = self.perimeter_zone(local, player_radius) else {
            return Vec2::ZERO;
        };
        let center = self.center;
        let right = self.right;
        let up = self.up;
        let radius = self.radius + player_radius;
        let x = self.extents.x;
        let y = self.extents.y;
        match zone {
            0 => center - right * x + up * (y + radius) + right * progress,
            1 => {
                let angle = Fix::HALF_PI - progress / radius;
                center + right * x + up * y + (right * angle.cos() + up * angle.sin()) * radius
            }
            2 => center + right * (x + radius) + up * (y - progress),
            3 => {
                let angle = -progress / radius;
                center + right * x - up * y + (right * angle.cos() + up * angle.sin()) * radius
            }
            4 => center + right * x - up * (y + radius) - right * progress,
            5 => {
                let angle = Fix::HALF_PI - progress / radius + Fix::PI;
                center - right * x - up * y + (right * angle.cos() + up * angle.sin()) * radius
            }
            6 => center - right * (x + radius) + up * (progress - y),
            7 => {
                let angle = Fix::PI - progress / radius;
                center - right * x + up * y + (right * angle.cos() + up * angle.sin()) * radius
            }
            _ => unreachable!(),
        }
    }
    pub fn closest_point(&self, point: Vec2) -> Result<Vec2, MathError> {
        let x = (point - self.center).dot(self.right);
        let y = (point - self.center).dot(self.up);
        let up = self.up * self.extents.y;
        let right = self.right * self.extents.x;
        let corner_point = |corner: Vec2| -> Result<Vec2, MathError> {
            let delta = point - corner;
            let length = delta.length_squared().sqrt()?;
            let normal = Vec2::new(delta.x.checked_div(length)?, delta.y.checked_div(length)?);
            Ok(corner + normal * self.radius)
        };
        if x <= -self.extents.x {
            if y >= self.extents.y {
                return corner_point(self.center - right + up);
            }
            if y <= -self.extents.y {
                return corner_point(self.center - right - up);
            }
            return Ok(self.center + (-self.right) * (self.extents.x + self.radius) + self.up * y);
        }
        if x >= self.extents.x {
            if y >= self.extents.y {
                return corner_point(self.center + right + up);
            }
            if y <= -self.extents.y {
                return corner_point(self.center + right - up);
            }
            return Ok(self.center + self.right * (self.extents.x + self.radius) + self.up * y);
        }
        Ok(if y >= Fix::ZERO {
            self.center + self.right * x + self.up * (self.extents.y + self.radius)
        } else {
            self.center + self.right * x + (-self.up) * (self.extents.y + self.radius)
        })
    }
    pub fn local_position(&self, point: Vec2, player_radius: Fix) -> Result<Fix, MathError> {
        let (_, zone) = self.normal_and_zone(point);
        let mut total = Fix::ZERO;
        for (index, segment) in self.segments().iter().enumerate().take(zone) {
            total += *segment;
            if index % 2 == 1 {
                total += player_radius * Fix::HALF_PI;
            }
        }
        let c = self.center;
        let up = self.up;
        let right = self.right;
        let radius = self.radius + player_radius;
        let x = self.extents.x;
        let y = self.extents.y;
        let closest = self.closest_point(point)?;
        let distance = match zone {
            0 => total + (c + up * (y + self.radius) - right * x - closest).length(),
            1 => total + (angle2(up, closest - (c + up * y + right * x)) * radius).abs(),
            2 => total + (c + up * y + right * (x + self.radius) - closest).length(),
            3 => total + (angle2(right, closest - (c - up * y + right * x)) * radius).abs(),
            4 => total + (c - up * (y + self.radius) + right * x - closest).length(),
            5 => total + (angle2(-up, closest - (c - up * y - right * x)) * radius).abs(),
            6 => total + (c - up * y - right * (x + self.radius) - closest).length(),
            7 => total + (angle2(-right, closest - (c + up * y - right * x)) * radius).abs(),
            _ => unreachable!(),
        };
        distance.checked_div(self.circumference() + player_radius * Fix::TAU)
    }
    pub fn move_unblocked(&self, local: Fix, radius: Fix, speed: Fix, dt: Fix) -> Fix {
        self.try_move_unblocked(local, radius, speed, dt).unwrap()
    }
    pub fn try_move_unblocked(
        &self,
        local: Fix,
        radius: Fix,
        speed: Fix,
        dt: Fix,
    ) -> Result<Fix, MathError> {
        let movement = speed * dt;
        let circumference = self.circumference() + radius * Fix::TAU;
        (Fix::ONE + local + movement.checked_div(circumference)?).checked_rem(Fix::ONE, true)
    }
}

fn angle2(a: Vec2, b: Vec2) -> Fix {
    let left = a * b.length();
    let right = b * a.length();
    Fix::from_int(2) * (left - right).length().atan2((left + right).length())
}
