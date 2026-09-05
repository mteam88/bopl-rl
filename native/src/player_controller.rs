//! PlayerPhysics movement/update controller. Platform references are binding indices.
use crate::{
    fixed::Fix,
    game_time,
    physics::{query::SurfaceHit, scene::Scene},
    platform_motion::{Binding, Movement, Terrain},
    player_physics::{self, Info, Platform},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub physics: player_physics::State,
    pub skip_update: bool,
    pub apply_gravity: bool,
    pub blocked: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ground {
    pub ice: bool,
    pub drop_frames: i32,
    pub dont_drop: i32,
    pub drop_force: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub physics: player_physics::Config,
    pub accel: Fix,
    pub friction: Fix,
    pub ice_friction: Fix,
    pub alignment_deadzone: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Event {
    Attached(SurfaceHit),
    Detached(Fix),
    /// Other AbilityExitInfo members have their default values in DropPlatformTest.
    Exit(Info),
    JumpSound(String),
    Sound(String),
}
pub struct Context<'a> {
    pub scene: &'a mut Scene,
    pub bindings: &'a [Binding],
    pub grounds: &'a [Ground],
    pub config: &'a Config,
    pub holder: Option<i32>,
    pub protected: Option<bool>,
    pub time_scale: Fix,
    pub space: bool,
    pub ability: bool,
}
impl Context<'_> {
    pub fn dt(&self, dt: Fix) -> Fix {
        game_time::fixed_delta_time(dt, self.time_scale, self.protected)
    }
    fn id_dt(&self, dt: Fix) -> Result<Fix, &'static str> {
        self.holder.ok_or("null_reference")?;
        Ok(self.dt(dt))
    }
    fn platform(&self, index: Option<u32>) -> Result<&Platform, &'static str> {
        index
            .and_then(|i| self.bindings.get(i as usize))
            .and_then(|b| b.platform.as_ref())
            .ok_or("null_reference")
    }
}
impl State {
    fn ground(&self) -> Option<u32> {
        self.physics.motion.ground
    }
    fn cast(
        &mut self,
        dt: Fix,
        context: &mut Context,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        let m = &self.physics.motion;
        let hit = player_physics::collision_cast(
            m.position,
            m.velocity + m.external_velocity,
            m.radius,
            context.dt(dt),
            &mut |origin, direction, distance| {
                let hit = context.scene.raycast_rounded(origin, direction, distance);
                hit.object.map(|_| (hit, hit.near_position))
            },
        );
        if let Some((hit, position)) = hit {
            let index = context
                .bindings
                .iter()
                .position(|b| Some(b.object) == hit.object && b.platform.is_some())
                .map(|i| i as u32);
            self.physics.attach_platform(
                index,
                index.and_then(|i| context.bindings[i as usize].platform.as_ref()),
                position,
            )?;
            notify(self, Event::Attached(hit))?;
        }
        Ok(())
    }
    pub fn update(
        &mut self,
        dt: Fix,
        context: &mut Context,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        if self.skip_update {
            self.skip_update = false;
            return Ok(());
        }
        self.physics.update_scale()?;
        if !self.physics.motion.grounded && self.apply_gravity {
            self.physics
                .gravity(&context.config.physics, context.space)?;
            self.cast(dt, context, notify)?;
        }
        Ok(())
    }
    pub fn input_alignment(&self, input: Vec2, context: &Context) -> Result<Fix, &'static str> {
        let forward = if self.ground().is_some() {
            context
                .platform(self.ground())?
                .surface
                .forward(self.physics.motion.position)
        } else {
            Vec2::RIGHT
        };
        Ok(input.dot(forward))
    }
    fn unground(
        &mut self,
        context: &Context,
        sound: bool,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        let surface = self
            .ground()
            .map(|i| context.platform(Some(i)).map(|p| &p.surface))
            .transpose()?;
        let angle = self.physics.motion.unground(surface, true);
        notify(self, Event::Detached(angle))?;
        if sound {
            let id = context.holder.ok_or("null_reference")?;
            notify(self, Event::JumpSound(format!("slimeJump{id}")))?;
        }
        Ok(())
    }
    pub fn move_player(
        &mut self,
        input: Vec2,
        dt: Fix,
        multiplier: Option<Fix>,
        move_surface: bool,
        context: &mut Context,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        if !self.physics.motion.grounded {
            self.physics.air_move(input.x, &context.config.physics)?;
            return self.cast(context.dt(dt), context, notify);
        }
        let value = self.input_alignment(input.normalized_safe(), context)?;
        let platform = context.platform(self.ground())?;
        let outward = platform
            .surface
            .normal(self.physics.motion.position)
            .dot(input);
        let sign = if value.abs() >= context.config.alignment_deadzone
            || (outward <= Fix::ZERO && value.abs() >= Fix::from_f32(0.01))
        {
            Fix::from_int(value.sign())
        } else {
            Fix::ZERO
        };
        let index = self.ground().unwrap() as usize;
        if sign == Fix::ZERO {
            self.physics.motion.grounded_speed *= if context.grounds[index].ice {
                context.config.ice_friction
            } else {
                context.config.friction
            };
        } else {
            self.physics.motion.grounded_speed = increment(
                self.physics.motion.grounded_speed,
                sign * context.config.physics.max_speed,
                context.config.accel,
                context.dt(dt),
            );
        }
        if platform.destroyed || !context.bindings[index].active_self {
            return self.unground(context, true, notify);
        }
        if self.physics.sync_rotation {
            self.physics.align_rotation(&platform.surface);
        }
        let external = self
            .physics
            .motion
            .external_velocity
            .dot(platform.surface.forward(self.physics.motion.position));
        if move_surface || multiplier.is_some() {
            let speed =
                (self.physics.motion.grounded_speed + external) * multiplier.unwrap_or(Fix::ONE);
            let request = Movement {
                position: self.physics.motion.position,
                radius: self.physics.motion.radius,
                speed,
                dt: context.id_dt(dt)?,
                local: self.physics.motion.local,
                blocked: self.blocked,
            };
            self.physics.motion.local = Terrain {
                scene: context.scene,
                bindings: context.bindings,
            }
            .move_along(context.bindings[index].object, request, None)?;
        }
        self.drop_platform(context, notify)
    }
    pub fn drop_platform(
        &mut self,
        context: &Context,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        let Some(index) = self.ground() else {
            return Ok(());
        };
        let platform = context.platform(Some(index))?;
        if platform.destroyed || !context.bindings[index as usize].active_self {
            return Ok(());
        }
        let player = context.holder.ok_or("null_reference")?;
        let ground = &context.grounds[index as usize];
        if ground.drop_frames <= 0 || player == ground.dont_drop {
            return Ok(());
        }
        let normal = platform.surface.normal(self.physics.motion.position);
        self.unground(context, false, notify)?;
        self.physics.motion.velocity += normal * ground.drop_force;
        self.physics.motion.position += self.physics.motion.velocity / Fix::from_int(100);
        if context.ability {
            notify(
                self,
                Event::Exit(Info {
                    grounded: false,
                    ground: None,
                    local: Fix::ZERO,
                    radius: Fix::ZERO,
                    grounded_speed: Fix::ZERO,
                    position: self.physics.motion.position,
                    velocity: self.physics.motion.velocity,
                    external_velocity: Vec2::ZERO,
                }),
            )?;
        }
        Ok(())
    }
    pub fn align(&mut self, context: &Context) -> Result<(), &'static str> {
        if self.ground().is_some() {
            self.physics
                .align_platform(context.platform(self.ground())?)?;
        }
        Ok(())
    }
    pub fn move_current(&mut self, dt: Fix, context: &mut Context) -> Result<(), &'static str> {
        let platform = context.platform(self.ground())?;
        let speed = self
            .physics
            .motion
            .external_velocity
            .dot(platform.surface.forward(self.physics.motion.position));
        self.physics.motion.external_velocity = Vec2::ZERO;
        let request = Movement {
            position: self.physics.motion.position,
            radius: self.physics.motion.radius,
            speed: Fix::ZERO + speed,
            dt: context.id_dt(dt)?,
            local: self.physics.motion.local,
            blocked: self.blocked,
        };
        let object = context.bindings[self.ground().unwrap() as usize].object;
        self.physics.motion.local = Terrain {
            scene: context.scene,
            bindings: context.bindings,
        }
        .move_along(object, request, None)?;
        Ok(())
    }
    /// The source recursively reuses dt and passes the signed remaining movement
    /// as the next speed. The loop preserves that recurrence without Rust recursion.
    pub fn hop(
        &mut self,
        radius: Fix,
        mut speed: Fix,
        dt: Fix,
        buffer: &mut [i32],
        context: &mut Context,
    ) -> Result<(), &'static str> {
        loop {
            let index = self.ground().ok_or("null_reference")? as usize;
            let object = context.bindings[index].object;
            if let Some(slot) = buffer.iter_mut().find(|id| **id == -1) {
                *slot = object as i32;
            }
            let request = Movement {
                position: self.physics.motion.position,
                radius,
                speed,
                dt,
                local: self.physics.motion.local,
                blocked: self.blocked,
            };
            let result = Terrain {
                scene: context.scene,
                bindings: context.bindings,
            }
            .hop(object, request, buffer)?;
            self.physics.motion.local = result.local;
            let Some(blocking) = result.blocking else {
                return Ok(());
            };
            let location = context
                .platform(Some(index as u32))?
                .position(result.local, radius)?;
            let next = context
                .bindings
                .iter()
                .position(|b| b.object == blocking)
                .unwrap() as u32;
            self.physics.attach_platform(
                Some(next),
                context.bindings[next as usize].platform.as_ref(),
                location,
            )?;
            speed = result.remaining;
        }
    }
}
pub fn increment(mut current: Fix, target: Fix, accel: Fix, dt: Fix) -> Fix {
    if current * target < Fix::ZERO {
        current = Fix::ZERO;
    }
    let next = current + dt * Fix::from_int(target.sign()) * accel;
    if next.abs() > target.abs() {
        target
    } else {
        next
    }
}
pub fn accelerate(current: Fix, target: Fix, accel: Fix, dt: Fix) -> Fix {
    let sign = Fix::from_int((target - current).sign());
    let mut next = current + sign * dt * accel;
    if target == Fix::ZERO && current * next <= Fix::ZERO {
        return target;
    }
    if (sign >= Fix::ZERO && current <= target && next > target)
        || (sign < Fix::ZERO && current >= target && next < target)
    {
        next = target;
    }
    next
}
