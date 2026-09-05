//! PlayerPhysics state transitions used at ability entry/exit and while attached
//! to platforms and ropes. World raycasts and surface hopping are separate.
use crate::{fixed::Fix, player_body::BodyRope, player_ground, surface::Surface, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub surface: Surface,
    pub destroyed: bool,
    pub has_shape: bool,
}
impl Platform {
    pub(crate) fn position(&self, local: Fix, radius: Fix) -> Result<Vec2, &'static str> {
        if !self.has_shape && self.surface.perimeter_zone(local, radius).is_some() {
            return Err("null_reference");
        }
        Ok(self.surface.position(local, radius))
    }
    fn local_position(&self, position: Vec2, radius: Fix) -> Result<Fix, &'static str> {
        if !self.has_shape {
            return Err("null_reference");
        }
        self.surface
            .local_position(position, radius)
            .map_err(|_| "division_by_zero")
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub gravity: Fix,
    pub max_fall_speed: Fix,
    pub gravity_modifier: Fix,
    pub air_accel: Fix,
    pub max_speed: Fix,
    pub inverse_mass: Fix,
    pub jump_strength: Fix,
    pub jump_extra_x: Fix,
    pub jump_momentum: Fix,
    pub jump_normal_scale: Fix,
    pub jump_teleport: Fix,
    pub rope_strength: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub motion: player_ground::State,
    pub scale: Fix,
    pub previous_scale: Fix,
    pub jumped: bool,
    pub sync_rotation: bool,
    pub cannot_unground: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Info {
    pub grounded: bool,
    pub ground: Option<u32>,
    pub local: Fix,
    pub radius: Fix,
    pub grounded_speed: Fix,
    pub position: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
}
impl State {
    pub fn update_scale(&mut self) -> Result<(), &'static str> {
        if self.previous_scale != self.scale {
            self.motion.radius *= self
                .scale
                .checked_div(self.previous_scale)
                .map_err(|_| "division_by_zero")?;
            self.previous_scale = self.scale;
        }
        Ok(())
    }
    fn platform<'a>(&self, platforms: &'a [Platform]) -> Result<&'a Platform, &'static str> {
        self.motion
            .ground
            .and_then(|i| platforms.get(i as usize))
            .ok_or("null_reference")
    }
    pub(crate) fn align_rotation(&mut self, surface: &Surface) {
        self.motion.up = surface.normal(self.motion.position);
        self.motion.right = self.motion.up.perp_cw();
        self.motion.rotation = self.motion.right.normalized_angle();
    }
    pub fn align(&mut self, platforms: &[Platform]) -> Result<(), &'static str> {
        if let Some(id) = self.motion.ground {
            self.align_platform(&platforms[id as usize])?;
        }
        Ok(())
    }
    pub(crate) fn align_platform(&mut self, platform: &Platform) -> Result<(), &'static str> {
        self.motion.position = platform.position(self.motion.local, self.motion.radius)?;
        self.align_rotation(&platform.surface);
        Ok(())
    }
    pub fn gravity(&mut self, config: &Config, space: bool) -> Result<(), &'static str> {
        apply_gravity(
            &mut self.motion.velocity,
            config.gravity,
            config.gravity_modifier,
            config.max_fall_speed,
            space,
        )
    }
    pub fn air_move(&mut self, input: Fix, config: &Config) -> Result<(), &'static str> {
        air_movement(
            &mut self.motion.velocity,
            input,
            config.air_accel,
            config.max_speed,
        )
    }
    pub fn add_force(&mut self, force: Vec2, config: &Config) {
        self.motion.external_velocity += force * config.inverse_mass;
    }
    pub fn jump(&mut self, platforms: &[Platform], config: &Config) -> Result<Fix, &'static str> {
        self.jumped = true;
        let normal = if self.motion.grounded {
            self.platform(platforms)?
                .surface
                .normal(self.motion.position)
        } else {
            Vec2::UP
        };
        let amount = (normal.y + Fix::ONE)
            .checked_div(config.jump_normal_scale)
            .map_err(|_| "division_by_zero")?
            .min(Fix::ONE)
            .max(Fix::ZERO);
        let up = Vec2::UP * amount + Vec2::new(normal.x, Fix::ZERO) * config.jump_extra_x;
        let tangent_velocity = normal.perp_cw() * self.motion.grounded_speed;
        self.motion.velocity = tangent_velocity * config.jump_momentum + up * config.jump_strength;
        self.motion.position += self.motion.velocity * config.jump_teleport;
        let surface = self.motion.ground.map(|id| &platforms[id as usize].surface);
        Ok(self.motion.unground(surface, true))
    }
    /// Dispatch detachment immediately. A listener may change the body position;
    /// Unity uses that changed position when updating the rope endpoint afterward.
    pub fn late_update(
        &mut self,
        platforms: &[Platform],
        rope: Option<(&mut BodyRope, bool)>,
        config: &Config,
        notify: &mut impl FnMut(&mut Self, Fix),
    ) -> Result<(), &'static str> {
        if self.motion.grounded
            && let Some(id) = self.motion.ground
            && !platforms[id as usize].destroyed
        {
            let surface = &platforms[id as usize].surface;
            self.motion.position =
                platforms[id as usize].position(self.motion.local, self.motion.radius)?;
            if self.sync_rotation {
                self.align_rotation(surface);
            }
            if let Some((rope, top)) = rope
                && rope.physics.enabled
                && rope.physics.time_since_hook_arrived > Fix::ZERO
            {
                let end = if top {
                    0
                } else {
                    rope.physics
                        .count
                        .checked_sub(1)
                        .ok_or("index_out_of_range")?
                };
                let endpoint = rope
                    .physics
                    .segment
                    .get_mut(end)
                    .ok_or("index_out_of_range")?;
                let offset = *endpoint - self.motion.position;
                if offset.length_squared() >= Fix::from_f64(0.07) {
                    self.motion.external_velocity += offset * config.rope_strength;
                }
                if surface
                    .normal(self.motion.position)
                    .dot(self.motion.external_velocity)
                    > Fix::from_int(30)
                    && !self.cannot_unground
                {
                    let rotation = self.motion.unground(Some(surface), true);
                    notify(self, rotation);
                }
                *endpoint = self.motion.position;
            }
        }
        self.jumped = false;
        Ok(())
    }
    pub fn offset(&mut self, offset: Vec2, platforms: &[Platform]) -> Result<(), &'static str> {
        if offset != Vec2::ZERO {
            if let Some(id) = self.motion.ground {
                self.motion.position =
                    platforms[id as usize].position(self.motion.local, self.motion.radius)?;
                self.motion.local = platforms[id as usize]
                    .local_position(self.motion.position + offset, self.motion.radius)?;
            } else {
                self.motion.position += offset;
            }
        }
        Ok(())
    }
    pub fn sync(
        &mut self,
        info: &Info,
        exit: bool,
        platforms: &[Platform],
    ) -> Result<(), &'static str> {
        self.motion.grounded = info.grounded;
        self.update_scale()?;
        let ground = info.ground.and_then(|id| platforms.get(id as usize));
        if self.motion.grounded && (!exit || ground.is_some_and(|p| p.has_shape && !p.destroyed)) {
            if exit {
                self.motion.grounded_speed = info.grounded_speed;
            }
            self.motion.ground = info.ground;
            if !exit {
                self.motion.local = info.local;
            }
            let platform = ground.ok_or("null_reference")?;
            let surface = &platform.surface;
            if exit {
                let position = platform.position(info.local, info.radius)?;
                self.motion.local = platform.local_position(position, self.motion.radius)?;
                self.motion.position = position;
                self.motion.external_velocity = info.external_velocity;
                self.motion.velocity = Vec2::ZERO;
            } else {
                self.motion.local = info.local;
                self.motion.position = platform.position(self.motion.local, self.motion.radius)?;
                self.motion.velocity = Vec2::ZERO;
                self.motion.external_velocity = info.external_velocity;
                self.motion.grounded_speed = info.grounded_speed;
            }
            self.align_rotation(surface);
        } else {
            self.motion.external_velocity = info.external_velocity;
            self.motion.velocity = info.velocity;
            self.motion.position = info.position;
            self.motion.rotation = Fix::ZERO;
            self.motion.right = Vec2::angle(Fix::ZERO);
            self.motion.up = -self.motion.right.perp_cw();
            self.motion.ground = None;
            self.motion.grounded_speed = Fix::ZERO;
            // The source leaves isGrounded equal to the incoming flag here,
            // including an invalid grounded ability exit with a null platform.
        }
        Ok(())
    }
}

/// PlayerPhysics.AddGravityFactor; the source applies this per call, without dt.
pub fn apply_gravity(
    velocity: &mut Vec2,
    gravity: Fix,
    modifier: Fix,
    max_fall_speed: Fix,
    space: bool,
) -> Result<(), &'static str> {
    let modifier = if space {
        modifier * Fix::from_f32(0.5)
    } else {
        modifier
    };
    *velocity += -Vec2::UP * gravity * modifier;
    let damping = (gravity * modifier)
        .checked_div(max_fall_speed)
        .map_err(|_| "division_by_zero")?;
    *velocity += -Vec2::UP * damping * velocity.y;
    Ok(())
}

/// PlayerPhysics.AirealMovement likewise ignores its source delta-time argument.
pub fn air_movement(
    velocity: &mut Vec2,
    input: Fix,
    accel: Fix,
    max_speed: Fix,
) -> Result<(), &'static str> {
    *velocity += Vec2::new(input * accel, Fix::ZERO);
    let damping = accel
        .checked_div(max_speed + accel)
        .map_err(|_| "division_by_zero")?;
    *velocity += -Vec2::RIGHT * damping * velocity.x;
    Ok(())
}

/// CollisionRaycasts tests the front ray, then each side ray, stopping at the
/// first successful query rather than comparing hits across the three rays.
pub fn collision_cast<T>(
    position: Vec2,
    velocity: Vec2,
    radius: Fix,
    effective_dt: Fix,
    query: &mut impl FnMut(Vec2, Vec2, Fix) -> Option<(T, Vec2)>,
) -> Option<(T, Vec2)> {
    let magnitude = velocity.length();
    let distance = magnitude * effective_dt;
    let direction = if magnitude == Fix::ZERO {
        Vec2::ZERO
    } else {
        velocity / magnitude
    };
    let side = -direction.perp_cw() * radius;
    for (origin, correction) in [
        (position + direction * radius, Vec2::ZERO),
        (position - side, side),
        (position + side, -side),
    ] {
        if let Some((hit, point)) = query(origin, direction, distance) {
            return Some((hit, point + correction));
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionRaycast {
    pub velocity: Vec2,
    pub effective_dt: Fix,
    pub attach: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundTarget {
    pub object: u32,
    pub ground: Option<u32>,
}
impl State {
    pub fn attach(
        &mut self,
        ground: Option<u32>,
        location: Vec2,
        platforms: &[Platform],
    ) -> Result<(), &'static str> {
        self.attach_platform(
            ground,
            ground.and_then(|id| platforms.get(id as usize)),
            location,
        )
    }
    pub(crate) fn attach_platform(
        &mut self,
        ground: Option<u32>,
        platform: Option<&Platform>,
        location: Vec2,
    ) -> Result<(), &'static str> {
        self.motion.ground = ground;
        self.motion.grounded = true;
        let platform = platform.ok_or("null_reference")?;
        if !platform.has_shape {
            self.motion.grounded_speed = self
                .motion
                .velocity
                .cross(platform.surface.normal(self.motion.position));
            return Err("null_reference");
        }
        self.motion
            .attach(ground.unwrap(), &platform.surface, location)
            .map_err(|_| "division_by_zero")
    }
    pub fn collision_raycasts(
        &mut self,
        request: CollisionRaycast,
        scene: &mut crate::physics::scene::Scene,
        targets: &[GroundTarget],
        platforms: &[Platform],
        notify: &mut impl FnMut(&mut Self, crate::physics::query::SurfaceHit),
    ) -> Result<bool, &'static str> {
        let result = collision_cast(
            self.motion.position,
            request.velocity,
            self.motion.radius,
            request.effective_dt,
            &mut |origin, direction, distance| {
                let hit = scene.raycast_rounded(origin, direction, distance);
                hit.object.map(|_| (hit, hit.near_position))
            },
        );
        let Some((hit, location)) = result else {
            return Ok(false);
        };
        if request.attach {
            let ground = targets
                .iter()
                .find(|t| Some(t.object) == hit.object)
                .and_then(|t| t.ground);
            self.attach(ground, location, platforms)?;
            notify(self, hit);
        }
        Ok(true)
    }
}
