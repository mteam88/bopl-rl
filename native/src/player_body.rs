//! PlayerBody motion and rope transitions. The caller supplies live rope objects
//! and the registered player/input context; scheduling and ability ownership are
//! separate. Endpoint operations reuse the physics rope implementation.
use crate::{fixed::Fix, physics::rope::Rope, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyRope {
    pub physics: Rope,
    pub owner: u8,
    pub time_air_start: Fix,
    pub time_air_end: Fix,
}
impl BodyRope {
    fn set_force(&mut self, force: Vec2, top: bool) {
        if top {
            self.physics.force_start = force;
        } else {
            self.physics.force_end = force;
        }
    }
    fn set_time(&mut self, time: Fix, top: bool) {
        if top {
            self.time_air_start = time;
        } else {
            self.time_air_end = time;
        }
    }
    fn endpoint(&self, top: bool) -> Result<usize, &'static str> {
        if top {
            Ok(0)
        } else {
            self.physics
                .count
                .checked_sub(1)
                .ok_or("index_out_of_range")
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Physics {
    pub grounded: bool,
    pub jumped: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    pub id: i32,
    pub protected: bool,
    pub jump: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Context {
    pub players: Vec<Player>,
    pub time_scale: Fix,
    pub rope_gravity: Fix,
    pub player_gravity: Fix,
    pub max_stretch: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub position: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub previous_position: Vec2,
    pub previous_velocity: Vec2,
    pub previous_imposed: Vec2,
    pub previous_external: Vec2,
    pub extra_swing_force: Vec2,
    pub rope: Option<usize>,
    pub top: bool,
    pub ignore_gravity: bool,
    pub rope_mass: Fix,
    pub holder: Option<i32>,
    pub physics: Option<Physics>,
}
impl State {
    fn total_velocity(&self) -> Vec2 {
        self.external_velocity + self.velocity
    }
    pub fn init(&mut self) {
        self.previous_velocity = self.total_velocity();
        self.previous_imposed = self.velocity;
        self.previous_external = self.external_velocity;
    }
    pub fn add_force(&mut self, force: Vec2) {
        self.external_velocity += force * Fix::from_int(35);
    }
    fn mass(&self) -> Fix {
        if self.rope_mass == Fix::ZERO {
            Fix::from_f64(0.5)
        } else {
            self.rope_mass
        }
    }
    pub fn attach(
        &mut self,
        next: Option<usize>,
        top: bool,
        ropes: &mut [BodyRope],
        context: &Context,
    ) {
        if let Some(index) = self.rope {
            let rope = &mut ropes[index];
            rope.set_force(
                if self.ignore_gravity {
                    Vec2::ZERO
                } else {
                    -Vec2::UP * context.rope_gravity
                },
                top,
            );
            rope.set_time(Fix::from_int(100), top);
            if next.is_none() {
                rope.physics.detach(top);
            } else {
                rope.physics.set_mass(self.mass(), top, Fix::ONE);
            }
        }
        self.init();
        self.rope = next;
        self.top = top;
    }
    pub fn clear_for_ability(&mut self) {
        self.init();
        self.rope = None;
    }
    /// Return true when Unity requests the hookshotLetGo sound.
    pub fn update(
        &mut self,
        dt: Fix,
        ropes: &mut [BodyRope],
        context: &Context,
    ) -> Result<bool, &'static str> {
        let mut sound = false;
        if let Some(index) = self.rope
            && ropes[index].physics.enabled
        {
            let id = self.holder.ok_or("null_reference")?;
            let player = context
                .players
                .iter()
                .find(|p| p.id == id)
                .ok_or("null_reference")?;
            if player.id == i32::from(ropes[index].owner)
                && self
                    .physics
                    .as_ref()
                    .is_none_or(|p| !p.grounded && !p.jumped)
                && player.jump
            {
                let r = &ropes[index];
                let end = r.endpoint(self.top)?;
                let position = *r.physics.segment.get(end).ok_or("index_out_of_range")?;
                let previous = *r.physics.previous.get(end).ok_or("index_out_of_range")?;
                self.velocity = (position - previous) / DT;
                self.attach(None, self.top, ropes, context);
                sound = true;
            }
        }
        if self.rope.is_some_and(|i| !ropes[i].physics.enabled) {
            self.rope = None;
        }
        let protected = self
            .holder
            .and_then(|id| context.players.iter().find(|p| p.id == id))
            .map(|p| p.protected);
        let effective_dt = crate::game_time::fixed_delta_time(dt, context.time_scale, protected);
        self.position += self.total_velocity() * effective_dt;
        if let Some(index) = self.rope {
            let r = &mut ropes[index];
            if r.physics.detached {
                let end = r.endpoint(self.top)?;
                *r.physics.segment.get_mut(end).ok_or("index_out_of_range")? = self.position;
                self.attach(None, self.top, ropes, context);
            } else if r.physics.time_since_hook_arrived <= Fix::ZERO {
                let end = r.endpoint(self.top)?;
                *r.physics.segment.get_mut(end).ok_or("index_out_of_range")? = self.position;
                r.set_time(Fix::ZERO, self.top);
                r.physics.set_mass(Fix::ONE, self.top, Fix::ONE);
                r.set_force(Vec2::ZERO, self.top);
            } else if self.physics.as_ref().is_some_and(|p| p.grounded) {
                let mass = Fix::ONE
                    + (Fix::from_f64(0.9) - Fix::ONE)
                        * r.physics.time_since_hook_arrived.clamp(Fix::ZERO, Fix::ONE);
                r.set_time(Fix::ZERO, self.top);
                r.physics.set_mass(mass, self.top, Fix::ONE);
                r.set_force(Vec2::ZERO, self.top);
            } else {
                let end = r.endpoint(self.top)?;
                let time = if self.top {
                    &mut r.time_air_start
                } else {
                    &mut r.time_air_end
                };
                *time += dt;
                let time = *time;
                r.physics.set_mass(self.mass(), self.top, Fix::ONE);
                let amount = if self.ignore_gravity {
                    Fix::ONE - self.rope_mass
                } else {
                    (time * Fix::from_int(2)).clamp(Fix::ZERO, Fix::ONE)
                };
                let endpoint = r.physics.segment.get_mut(end).ok_or("index_out_of_range")?;
                let offset = *endpoint - self.position;
                let length = offset.length();
                if length != Fix::ZERO {
                    let correction = r.physics.separation - length;
                    let direction = offset / length;
                    *endpoint += direction * correction * (Fix::ONE - amount);
                    self.position -= direction * correction * amount;
                }
            }
            self.init();
        }
        self.external_velocity *= Fix::from_f64(0.95);
        Ok(sound)
    }
    pub fn late_update(
        &mut self,
        ropes: &mut [BodyRope],
        context: &Context,
    ) -> Result<(), &'static str> {
        if let Some(index) = self.rope {
            if !ropes[index].physics.enabled {
                self.rope = None;
            } else {
                let r = &ropes[index].physics;
                for i in 0..r.count.saturating_sub(2) {
                    let a = r.segment.get(i).ok_or("index_out_of_range")?;
                    let b = r.segment.get(i + 1).ok_or("index_out_of_range")?;
                    if (*a - *b).length_squared() > context.max_stretch {
                        self.attach(None, self.top, ropes, context);
                        break;
                    }
                }
            }
        }
        if let Some(index) = self.rope {
            let mut imposed = self.velocity - self.previous_imposed;
            imposed += -Vec2::UP * context.player_gravity;
            let external = self.external_velocity - self.previous_external;
            if self.ignore_gravity {
                imposed.y = Fix::ZERO;
            }
            ropes[index].set_force(imposed + external + self.extra_swing_force, self.top);
            self.extra_swing_force = Vec2::ZERO;
        }
        self.previous_position = self.position;
        Ok(())
    }
}
