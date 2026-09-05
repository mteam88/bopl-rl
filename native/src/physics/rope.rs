//! Fixed rope integration and endpoint operations from RopeBody/DetPhysics.
//! Scene broad-phase registration and grappling ability behavior remain separate.
use crate::{fixed::Fix, geometry::RoundedRect, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rope {
    pub segment: Vec<Vec2>,
    pub previous: Vec<Vec2>,
    pub count: usize,
    pub enabled: bool,
    pub force_start: Vec2,
    pub force_end: Vec2,
    pub mass_start: Fix,
    pub mass_end: Fix,
    pub separation: Fix,
    pub time_since_hook_arrived: Fix,
    pub detached: bool,
    pub colliding_rects: Vec<usize>,
    pub collision_count: usize,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RopeConfig {
    pub dt: Fix,
    pub gravity: Fix,
    pub space: bool,
    pub iterations: usize,
    pub stretchiness: Fix,
}
impl Rope {
    /// Bounds omit the two endpoints, exactly as PhysTools.ComputeBoundingBox.
    pub fn bounds(&self) -> [Vec2; 2] {
        let mut low = self.segment[1];
        let mut high = low;
        for p in &self.segment[2..self.count.saturating_sub(1).max(2)] {
            low.x = low.x.min(p.x);
            low.y = low.y.min(p.y);
            high.x = high.x.max(p.x);
            high.y = high.y.max(p.y);
        }
        [low, high]
    }
    pub fn set_mass(&mut self, value: Fix, top: bool, scale: Fix) {
        let tenth = Fix::from_f64(0.1);
        let value = (value + scale * tenth - tenth).clamp(Fix::ZERO, Fix::ONE);
        if top {
            self.mass_start = value;
        } else {
            self.mass_end = value;
        }
    }
    pub fn detach(&mut self, top: bool) {
        self.detached = true;
        // The original Dettach method assigns the OPPOSITE endpoint's mass.
        if top {
            self.mass_end = Fix::from_f64(0.5);
        } else {
            self.mass_start = Fix::from_f64(0.5);
        }
    }
    pub fn spawn(&mut self, position: Vec2) -> bool {
        if self.count >= self.segment.len() {
            return false;
        }
        self.count += 1;
        self.segment[self.count - 1] = self.segment[self.count - 2];
        self.previous[self.count - 1] = self.previous[self.count - 2];
        self.segment[self.count - 2] = position;
        self.previous[self.count - 2] = position;
        true
    }
    pub fn reel(&mut self, minimum: usize, reverse: bool) -> bool {
        if self.count <= minimum {
            return false;
        }
        if reverse {
            for i in 1..self.count - 1 {
                self.segment[i] = self.segment[i + 1];
                self.previous[i] = self.previous[i + 1];
            }
        } else {
            self.segment[self.count - 2] = self.segment[self.count - 1];
            self.previous[self.count - 2] = self.previous[self.count - 1];
        }
        self.count -= 1;
        true
    }
    pub fn step(&mut self, config: RopeConfig, rectangles: &[RoundedRect]) {
        if !self.enabled {
            return;
        }
        let dt = if config.dt == Fix::ZERO {
            DT * Fix::from_f64(0.1)
        } else {
            config.dt
        };
        let (start_force, end_force, start_mass, end_mass) =
            if self.time_since_hook_arrived > Fix::ZERO {
                (
                    self.force_start,
                    self.force_end,
                    self.mass_start,
                    self.mass_end,
                )
            } else {
                (Vec2::ZERO, Vec2::ZERO, Fix::ONE, Fix::ONE)
            };
        let half = Fix::from_f32(0.5);
        let gravity = (-Vec2::UP) * config.gravity * if config.space { half } else { Fix::ONE };
        let last = self.count - 1;
        // Keep even the source's unused division: a one-segment rope errors.
        let _unused = Fix::ONE / Fix::from_int(last as i64);
        for i in 0..=last {
            let before = self.segment[i];
            let force = if i == 0 {
                start_force
            } else if i == last {
                end_force
            } else {
                gravity
            };
            self.segment[i] = Fix::from_int(2) * before - self.previous[i] + dt * force;
            self.previous[i] = before;
        }
        for _ in 0..config.iterations {
            let delta = self.segment[1] - self.segment[0];
            let distance = delta.length();
            if distance == Fix::ZERO {
                continue;
            }
            let correction = -distance;
            let direction = delta / distance;
            self.segment[1] += correction * direction * start_mass;
            self.segment[0] -= correction * direction * (Fix::ONE - start_mass);
            for i in 1..self.count - 2 {
                let delta = self.segment[i] - self.segment[i + 1];
                let distance = delta.length();
                if distance == Fix::ZERO {
                    continue;
                }
                let correction = self.separation - distance;
                let direction = delta / distance;
                self.segment[i] += correction * direction * half;
                self.segment[i + 1] -= correction * direction * half;
            }
            let delta = self.segment[last - 1] - self.segment[last];
            let distance = delta.length();
            if distance != Fix::ZERO {
                let correction = -distance;
                let direction = delta / distance;
                self.segment[last - 1] += correction * direction * end_mass;
                self.segment[last] -= correction * direction * (Fix::ONE - end_mass);
            }
        }
        let stretch = config.stretchiness / (self.separation * self.separation);
        for &r in &self.colliding_rects[..self.collision_count] {
            let rectangle = rectangles[r];
            for i in 1..last {
                if !rectangle.overlaps(self.segment[i]) {
                    continue;
                }
                self.segment[i] = rectangle.closest_point(self.segment[i]).unwrap();
                if self.mass_end != half && self.mass_start != half {
                    let t = ((self.segment[i] - self.segment[i - 1]).length_squared() * stretch)
                        .clamp(Fix::ZERO, Fix::ONE)
                        * half;
                    // Vec2.Lerp uses (1-t)*a+t*b, not a+t*(b-a).
                    self.segment[i] = (Fix::ONE - t) * self.segment[i] + t * self.segment[i - 1];
                    self.segment[i] = rectangle.closest_point(self.segment[i]).unwrap();
                }
            }
        }
        self.collision_count = 0;
    }
}
