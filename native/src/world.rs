//! Independent movement simulation for the exported Level1 reset state.
//! Tick ordering follows PlayerPhysics -> SlimeController -> PlayerBody -> late alignment.
use crate::{
    fixed::Fix,
    geometry::{OrientedBox, RoundedRect},
    surface::Surface,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

mod dash;

pub const DT: Fix = Fix(71_582_788);
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub id: u32,
    pub surface: Surface,
    pub collider: RoundedRect,
    pub ice: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovementConstants {
    pub max_speed: Fix,
    pub accel: Fix,
    pub air_accel: Fix,
    pub gravity_accel: Fix,
    pub gravity_max_fall_speed: Fix,
    pub gravity_modifier: Fix,
    pub jump_strength: Fix,
    pub jump_extra_x: Fix,
    pub jump_momentum: Fix,
    pub jump_normal_scale: Fix,
    pub jump_teleport: Fix,
    pub friction: Fix,
    pub ice_friction: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    pub alive: bool,
    pub id: u32,
    pub position: Vec2,
    pub rotation: Fix,
    pub up: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub radius: Fix,
    pub grounded: bool,
    pub ground: Option<u32>,
    pub local: Fix,
    pub grounded_speed: Fix,
    pub previous_jump: bool,
    pub constants: MovementConstants,
    pub cooldown: Fix,
    pub dash: DashState,
    pub collider: OrientedBox,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DashState {
    pub active: bool,
    pub time: Fix,
    pub updates: u32,
    pub in_dash: bool,
    pub input: Vec2,
    pub hit_ground: Option<u32>,
    pub hit_local: Fix,
    pub saved_local: Fix,
    pub local: Fix,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Action {
    pub angle: u8,
    pub buttons: u8,
}
impl Action {
    pub fn direction(self) -> Vec2 {
        if self.angle == 0 {
            return Vec2::ZERO;
        }
        let angle = Fix::TAU * (Fix::from_int(i64::from(self.angle) - 1) / Fix::from_int(255));
        Vec2::new(angle.cos(), angle.sin())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct World {
    pub tick: u64,
    pub platforms: Vec<Platform>,
    pub players: Vec<Player>,
}

impl World {
    pub fn level1() -> Self {
        serde_json::from_str(include_str!("../data/level1-movement.json"))
            .expect("validated embedded reset")
    }
    pub fn terminal(&self) -> bool {
        self.players.iter().any(|p| !p.alive)
    }
    pub fn step(&mut self, actions: &[Action]) -> Result<(), String> {
        if self.terminal() {
            return Err("episode ended; reset is required".into());
        }
        if actions.len() != self.players.len() {
            return Err("one action per player is required".into());
        }
        if actions.iter().any(|a| a.buttons & !3 != 0) {
            return Err("this slice supports jump and dash only".into());
        }
        for (index, action) in actions.iter().enumerate() {
            self.players[index].step(*action, &self.platforms);
            let caster = &self.players[index];
            if caster.dash.active && caster.dash.time == DT {
                let origin = caster.position;
                let direction = caster.dash.input;
                let wall = closest(&self.platforms, origin, direction, Fix::from_int(13), None)
                    .map(|(_, distance, _)| distance);
                let targets: Vec<_> = self
                    .players
                    .iter()
                    .enumerate()
                    .filter(|(i, p)| *i != index && !p.dash.active)
                    .filter_map(|(i, p)| {
                        p.collider
                            .raycast(origin, direction)
                            .filter(|hit| {
                                hit.near <= Fix::from_int(13)
                                    && wall.is_none_or(|distance| hit.near < distance)
                            })
                            .map(|_| i)
                    })
                    .collect();
                for target in targets {
                    self.players[target].push_from(origin, target < index);
                }
            }
        }
        // For the fixed-size default slime boxes, distance < radius implies box
        // overlap (radius .76, each box inscribed radius .81). Only that subset
        // produces SlimeController.OnCollide's push; non-overlap contacts do nothing.
        for a in 0..self.players.len() {
            for b in a + 1..self.players.len() {
                let a_pos = self.players[a].position;
                let b_pos = self.players[b].position;
                if self.players[a].alive && self.players[b].alive && !self.players[a].dash.active {
                    self.players[a].push_from(b_pos, true);
                }
                if self.players[a].alive && self.players[b].alive && !self.players[b].dash.active {
                    self.players[b].push_from(a_pos, false);
                }
            }
        }
        for player in &mut self.players {
            player.sync_collider();
            player.late_step(&self.platforms);
        }
        self.tick += 1;
        Ok(())
    }
}

fn closest(
    platforms: &[Platform],
    origin: Vec2,
    direction: Vec2,
    distance: Fix,
    exclude: Option<u32>,
) -> Option<(usize, Fix, Vec2)> {
    let mut nearest = Fix::MAX;
    let mut result = None;
    for (index, platform) in platforms.iter().enumerate() {
        if exclude == Some(platform.id) {
            continue;
        }
        if let Some(hit) = platform.collider.raycast(origin, direction)
            && hit.near < nearest
            && hit.near <= distance
        {
            nearest = hit.near;
            result = Some((index, hit.near, origin + direction * hit.near));
        }
    }
    result
}
impl Player {
    fn platform<'a>(&self, platforms: &'a [Platform]) -> Option<&'a Platform> {
        platforms.iter().find(|p| Some(p.id) == self.ground)
    }
    fn align(&mut self, platform: &Platform) {
        self.up = platform.surface.normal(self.position);
        self.rotation = self.up.perp_cw().normalized_angle();
    }
    fn attach(&mut self, platform: &Platform, position: Vec2) {
        self.ground = Some(platform.id);
        self.grounded = true;
        self.grounded_speed = self.velocity.cross(platform.surface.normal(self.position));
        self.local = platform
            .surface
            .local_position(position, self.radius)
            .expect("valid attachment");
        self.position = platform.surface.position(self.local, self.radius);
        self.velocity = Vec2::ZERO;
        self.external_velocity = Vec2::ZERO;
        self.align(platform);
    }
    fn collision_rays(&mut self, platforms: &[Platform]) {
        let velocity = self.velocity + self.external_velocity;
        let magnitude = velocity.length();
        let distance = magnitude * DT;
        let dir = if magnitude == Fix::ZERO {
            Vec2::ZERO
        } else {
            velocity / magnitude
        };
        let side = dir.perp() * self.radius;
        for (origin, correction) in [
            (self.position + dir * self.radius, Vec2::ZERO),
            (self.position - side, side),
            (self.position + side, -side),
        ] {
            if let Some((index, _, position)) = closest(platforms, origin, dir, distance, None) {
                self.attach(&platforms[index], position + correction);
                break;
            }
        }
    }
    fn jump(&mut self, platforms: &[Platform]) {
        let platform = self.platform(platforms).unwrap();
        let normal = platform.surface.normal(self.position);
        let amount = ((normal.y + Fix::ONE) / self.constants.jump_normal_scale)
            .min(Fix::ONE)
            .max(Fix::ZERO);
        let up = Vec2::UP * amount + Vec2::new(normal.x, Fix::ZERO) * self.constants.jump_extra_x;
        self.velocity = normal.perp_cw() * self.grounded_speed * self.constants.jump_momentum
            + up * self.constants.jump_strength;
        self.position += self.velocity * self.constants.jump_teleport;
        self.position += platform.surface.normal(self.position) * Fix::from_f32(0.05);
        self.grounded_speed = Fix::ZERO;
        self.grounded = false;
        self.rotation = Fix::ZERO;
        self.up = Vec2::UP;
    }
    fn move_grounded(&mut self, input: Vec2, platforms: &[Platform]) {
        let platform = self.platform(platforms).unwrap();
        let normal = platform.surface.normal(self.position);
        let forward = normal.perp_cw();
        let alignment = input.normalized_safe().dot(forward);
        let into_ground = normal.dot(input);
        let sign = if alignment.abs() >= Fix::from_f32(0.2)
            || (into_ground <= Fix::ZERO && alignment.abs() >= Fix::from_f32(0.01))
        {
            Fix::from_int(alignment.sign())
        } else {
            Fix::ZERO
        };
        if sign == Fix::ZERO {
            self.grounded_speed *= if platform.ice {
                self.constants.ice_friction
            } else {
                self.constants.friction
            };
        } else {
            let target = sign * self.constants.max_speed;
            if self.grounded_speed * target < Fix::ZERO {
                self.grounded_speed = Fix::ZERO;
            }
            self.grounded_speed += DT * Fix::from_int(target.sign()) * self.constants.accel;
            if self.grounded_speed.abs() > target.abs() {
                self.grounded_speed = target;
            }
        }
        self.align(platform);
        let speed = self.grounded_speed + self.external_velocity.dot(forward);
        let velocity = forward * speed;
        let direction = velocity.normalized_safe();
        let offset = direction * self.radius;
        let distance = velocity.length() * DT;
        let mut allowed = Fix::from_int(10_000_000_000);
        for origin in [
            self.position + offset,
            self.position + offset + offset.perp(),
            self.position + offset + offset.perp_cw(),
        ] {
            if let Some((_, near, _)) =
                closest(platforms, origin, direction, distance, Some(platform.id))
            {
                allowed = allowed.min(near);
            }
        }
        let allowed = allowed.min(speed.abs()) * Fix::from_int(speed.sign());
        self.local = platform
            .surface
            .move_unblocked(self.local, self.radius, allowed, DT);
    }
    fn gravity_step(&mut self, platforms: &[Platform]) {
        if !self.grounded {
            let gravity = self.constants.gravity_accel * self.constants.gravity_modifier;
            self.velocity +=
                (-Vec2::UP) * self.constants.gravity_accel * self.constants.gravity_modifier;
            self.velocity +=
                (-Vec2::UP) * (gravity / self.constants.gravity_max_fall_speed) * self.velocity.y;
            self.collision_rays(platforms);
        }
    }
    fn step(&mut self, action: Action, platforms: &[Platform]) {
        let jump = action.buttons & 1 != 0;
        if self.dash.active {
            self.previous_jump = jump;
            self.dash_step(action, platforms);
            return;
        }
        self.gravity_step(platforms);
        self.cooldown += DT;
        if action.buttons & 2 != 0 && self.cooldown > Fix::from_int(4) {
            self.enter_dash(action, platforms);
            self.previous_jump = jump;
            self.dash_step(action, platforms);
            return;
        }
        if jump && !self.previous_jump && self.grounded {
            self.jump(platforms);
        }
        self.previous_jump = jump;
        let input = action.direction();
        if self.grounded {
            self.move_grounded(input, platforms);
        } else {
            self.velocity += Vec2::new(input.x * self.constants.air_accel, Fix::ZERO);
            self.velocity += (-Vec2::RIGHT)
                * (self.constants.air_accel
                    / (self.constants.max_speed + self.constants.air_accel))
                * self.velocity.x;
            self.collision_rays(platforms);
        }
        if self.outside_bounds() {
            self.alive = false;
            return;
        }
        self.position += (self.velocity + self.external_velocity) * DT;
        self.external_velocity *= Fix::from_f64(0.95);
    }
    fn outside_bounds(&self) -> bool {
        self.position.y < Fix::from_f32(-11.3)
            || self.position.y > Fix::from_f64(72.5)
            || self.position.x < Fix::from_f64(-131.25)
            || self.position.x > Fix::from_f64(131.25)
    }
    fn push_from(&mut self, other: Vec2, lower_id: bool) {
        let delta = self.position - other;
        let distance = delta.length();
        let direction = if distance != Fix::ZERO && distance < self.radius {
            delta / distance
        } else if distance == Fix::ZERO {
            let direction = self.up.perp() / Fix::from_int(10);
            if lower_id { direction } else { -direction }
        } else {
            return;
        };
        self.external_velocity += direction * Fix(21_474_836) * Fix::from_int(35);
    }
    fn sync_collider(&mut self) {
        let extents = if self.dash.active {
            Vec2::new(Fix(5_626_406_912), Fix(4_359_391_744))
        } else {
            Vec2::new(Fix(5_712_306_688), Fix(3_478_923_520))
        };
        let inv = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
        let basis = Vec2::angle(self.rotation);
        self.collider = OrientedBox {
            center: self.position,
            right: (Vec2::RIGHT / inv.x).complex_mul(basis),
            up: (Vec2::UP / inv.y).complex_mul(basis),
            inverse_extents: inv,
        };
    }
    fn late_step(&mut self, platforms: &[Platform]) {
        if self.alive && self.grounded && !self.dash.active {
            let platform = self.platform(platforms).unwrap();
            self.position = platform.surface.position(self.local, self.radius);
            self.align(platform);
        }
    }
}
