//! StickyRoundedRectangle surface traversal and platform hopping.
use crate::{
    fixed::Fix,
    physics::{scene::Scene, shape::Collider},
    player_physics::Platform,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

/// Metadata for every registered rounded rectangle, including shapes without
/// a StickyRoundedRectangle component. Object IDs also identify platforms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    pub object: u32,
    pub platform: Option<Platform>,
    pub transform_destroyed: bool,
    pub active_self: bool,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Movement {
    pub position: Vec2,
    pub radius: Fix,
    pub speed: Fix,
    pub dt: Fix,
    pub local: Fix,
    pub blocked: bool,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Hop {
    pub local: Fix,
    pub remaining: Fix,
    pub blocking: Option<u32>,
}
pub struct Terrain<'a> {
    pub scene: &'a mut Scene,
    pub bindings: &'a [Binding],
}
impl Terrain<'_> {
    fn platform(&self, object: u32) -> Result<&Platform, &'static str> {
        self.bindings
            .iter()
            .find(|b| b.object == object)
            .and_then(|b| b.platform.as_ref())
            .ok_or("null_reference")
    }
    fn enabled(&mut self, object: u32, enabled: bool) {
        if let Some(o) = self.scene.objects.iter_mut().find(|o| o.id == object) {
            o.enabled = enabled;
        }
    }
    pub fn max_movement(
        &mut self,
        object: u32,
        position: Vec2,
        radius: Fix,
        velocity: Vec2,
        dt: Fix,
    ) -> Result<(Fix, Option<u32>), &'static str> {
        if !self.platform(object)?.has_shape {
            return Err("null_reference");
        }
        self.enabled(object, false);
        let (near, blocking) = movement_limit(
            position,
            radius,
            velocity,
            dt,
            &mut |origin, direction, distance| {
                let hit = self.scene.raycast_rounded(origin, direction, distance);
                hit.object.map(|id| {
                    let ground = self
                        .bindings
                        .iter()
                        .find(|b| b.object == id && b.platform.is_some())
                        .map(|b| b.object);
                    (ground, hit.near_distance)
                })
            },
        );
        self.enabled(object, true);
        Ok((near, blocking.flatten()))
    }
    pub fn move_along(
        &mut self,
        object: u32,
        request: Movement,
        second: Option<(Vec2, Fix)>,
    ) -> Result<Fix, &'static str> {
        let mut speed = request.speed;
        if request.blocked {
            let velocity = self.platform(object)?.surface.forward(request.position) * request.speed;
            speed = self
                .max_movement(
                    object,
                    request.position,
                    request.radius,
                    velocity,
                    request.dt,
                )?
                .0
                .min(request.speed.abs());
            if let Some((position, radius)) = second {
                // The original multiple-body overload replaces the first limit.
                speed = self
                    .max_movement(object, position, radius, velocity, request.dt)?
                    .0
                    .min(request.speed.abs());
            }
            speed *= Fix::from_int(request.speed.sign());
        }
        self.platform(object)?
            .surface
            .try_move_unblocked(request.local, request.radius, speed, request.dt)
            .map_err(|_| "division_by_zero")
    }
    pub fn hop(
        &self,
        object: u32,
        request: Movement,
        disallowed: &[i32],
    ) -> Result<Hop, &'static str> {
        let speed = request.speed * request.dt;
        let mut movement = speed;
        let mut remaining = Fix::ZERO;
        let mut blocking = None;
        if request.blocked {
            let platform = self.platform(object)?;
            let direction = (platform.surface.forward(request.position) * speed).normalized_safe();
            let position = platform.position(request.local, request.radius)?;
            let mut nearest = Fix::from_int(1_000_000_000);
            // This source query does not remove destroyed colliders or check the
            // collider enabled flag. Object validity is inside the exclusion loop.
            for o in &self.scene.objects {
                let Collider::RoundedRect(shape) = o.collider else {
                    continue;
                };
                if let Some(hit) = shape.raycast(position, direction) {
                    let binding = self.bindings.iter().find(|b| b.object == o.id);
                    let allowed = disallowed.iter().all(|id| {
                        *id != o.id as i32
                            && binding.is_some_and(|b| !b.transform_destroyed && b.active_self)
                    });
                    if allowed && hit.near < nearest {
                        nearest = hit.near;
                        blocking = binding.filter(|b| b.platform.is_some()).map(|b| b.object);
                    }
                }
            }
            if speed.abs() < nearest {
                blocking = None;
            }
            movement = speed.abs().min(nearest);
            remaining = (movement - speed.abs()).abs() * Fix::from_int(speed.sign());
            movement *= Fix::from_int(speed.sign());
        }
        let circumference =
            self.platform(object)?.surface.circumference() + request.radius * Fix::TAU;
        let local = (request.local
            + movement
                .checked_div(circumference)
                .map_err(|_| "division_by_zero")?
            + Fix::ONE)
            .checked_rem(Fix::ONE, true)
            .map_err(|_| "division_by_zero")?;
        Ok(Hop {
            local,
            remaining,
            blocking,
        })
    }
}

/// The three-ray limit from maxMovementAllowed. Every query runs before return.
/// The distance `movement_limit` reports when no ray hits anything.
pub const MOVEMENT_UNBLOCKED: Fix = Fix::from_int(10_000_000_000);
pub fn movement_limit<T>(
    position: Vec2,
    radius: Fix,
    velocity: Vec2,
    dt: Fix,
    query: &mut impl FnMut(Vec2, Vec2, Fix) -> Option<(T, Fix)>,
) -> (Fix, Option<T>) {
    // `normalized_safe` and `length` share the one square root.
    let length = velocity.length();
    let direction = if length == Fix::ZERO {
        velocity
    } else {
        velocity / length
    };
    let distance = length * dt;
    let front = direction * radius;
    let mut near = MOVEMENT_UNBLOCKED;
    let mut blocking = None;
    for (ray, origin) in [
        position + front,
        position + front - front.perp_cw(),
        position + front + front.perp_cw(),
    ]
    .into_iter()
    .enumerate()
    {
        if let Some((hit, distance)) = query(origin, direction, distance)
            && (ray == 0 || distance < near)
        {
            near = distance;
            blocking = Some(hit);
        }
    }
    (near, blocking)
}
