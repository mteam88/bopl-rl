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

pub const DT: Fix = Fix::from_raw(71_582_788);
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
    /// PlayerPhysics already ran on the tick where SlimeController notices
    /// that its attached platform was destroyed, so gravity starts one pass
    /// later for the ordinary slime.
    #[serde(default)]
    pub(crate) skip_next_gravity: bool,
    /// PlayerPhysics.jumpedThisFrame: set by Jump, cleared once the tick's
    /// state machine has seen it.
    #[serde(default)]
    pub(crate) jumped_this_frame: bool,
    pub constants: MovementConstants,
    pub cooldown: Fix,
    pub dash: DashState,
    /// Player-layer box as DetPhysics.Simulate last synced it (position and
    /// orientation of the tick's UpdateSim phase; a grounded body moves again
    /// in LateUpdateSim, so this lags the live position by one phase).
    pub collider: OrientedBox,
    #[serde(default)]
    pub collider_rotation: Fix,
    /// `Vec2::angle(collider_rotation)`, kept so an unchanged rotation does
    /// not pay for the sine table again (zero until the first sync).
    #[serde(default)]
    pub collider_basis: Vec2,
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
    /// The Dash body's rotation when Ability.ExitAbility deactivated it:
    /// the dash direction, or the platform alignment of a landing
    /// (PlayerPhysics.AttachToGround, PlayerPhysics.cs:508). The frozen
    /// FixTransform keeps it for DetPhysics' sync of inactive colliders.
    #[serde(default)]
    pub exit_rotation: Fix,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Action {
    pub angle: u8,
    pub buttons: u8,
}
impl Action {
    pub fn direction(self) -> Vec2 {
        static DIRECTIONS: std::sync::OnceLock<[Vec2; 256]> = std::sync::OnceLock::new();
        // The 255 stick directions are fixed; evaluating each once with the
        // same expression gives the same value as evaluating it per call.
        DIRECTIONS.get_or_init(|| std::array::from_fn(|angle| Self::direction_of(angle as u8)))
            [usize::from(self.angle)]
    }
    fn direction_of(angle: u8) -> Vec2 {
        if angle == 0 {
            return Vec2::ZERO;
        }
        let angle = Fix::TAU * (Fix::from_int(i64::from(angle) - 1) / Fix::from_int(255));
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
    pub fn level1_with_players(players: usize) -> Option<Self> {
        let reset = match players {
            1 => include_str!("../data/level1-movement-1.json"),
            2 => return Some(Self::level1()),
            3 => include_str!("../data/level1-movement-3.json"),
            4 => include_str!("../data/level1-movement-4.json"),
            5 => include_str!("../data/level1-movement-5.json"),
            6 => include_str!("../data/level1-movement-6.json"),
            _ => return None,
        };
        Some(serde_json::from_str(reset).expect("validated embedded multiplayer reset"))
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

pub(crate) fn closest(
    platforms: &[Platform],
    origin: Vec2,
    direction: Vec2,
    distance: Fix,
    exclude: Option<u32>,
) -> Option<(usize, Fix, Vec2)> {
    let mut nearest = Fix::MAX;
    let mut result = None;
    // Broad phase: the segment's endpoints once, then each platform's bounds
    // (RoundedRect::raycast_within) before its raycast; the platforms move, so
    // the bounds are recomputed per query rather than cached.
    let end = origin + direction * distance;
    for (index, platform) in platforms.iter().enumerate() {
        if exclude == Some(platform.id) {
            continue;
        }
        if !platform.collider.bounds().may_reach(origin, end) {
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
    pub(crate) fn platform<'a>(&self, platforms: &'a [Platform]) -> Option<&'a Platform> {
        platforms.iter().find(|p| Some(p.id) == self.ground)
    }
    pub(crate) fn align(&mut self, platform: &Platform) {
        self.up = platform.surface.normal(self.position);
        self.rotation = self.up.perp_cw().normalized_angle();
    }
    fn ground_state(&self) -> crate::player_ground::State {
        crate::player_ground::State {
            position: self.position,
            rotation: self.rotation,
            up: self.up,
            right: self.up.perp_cw(),
            velocity: self.velocity,
            external_velocity: self.external_velocity,
            radius: self.radius,
            grounded: self.grounded,
            ground: self.ground,
            local: self.local,
            grounded_speed: self.grounded_speed,
        }
    }
    fn apply_ground_state(&mut self, motion: crate::player_ground::State) {
        self.position = motion.position;
        self.rotation = motion.rotation;
        self.up = motion.up;
        self.velocity = motion.velocity;
        self.external_velocity = motion.external_velocity;
        self.grounded = motion.grounded;
        self.ground = motion.ground;
        self.local = motion.local;
        self.grounded_speed = motion.grounded_speed;
    }
    fn attach(&mut self, platform: &Platform, position: Vec2) {
        let mut motion = self.ground_state();
        motion
            .attach(platform.id, &platform.surface, position)
            .expect("valid attachment");
        self.apply_ground_state(motion);
    }
    pub(crate) fn collision_rays(&mut self, platforms: &[Platform]) {
        let velocity = self.velocity + self.external_velocity;
        if let Some((index, position)) = crate::player_physics::collision_cast(
            self.position,
            velocity,
            self.radius,
            DT,
            &mut |origin, direction, distance| {
                closest(platforms, origin, direction, distance, None)
                    .map(|(index, _, point)| (index, point))
            },
        ) {
            self.attach(&platforms[index], position);
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
        self.jumped_this_frame = true;
        let mut motion = self.ground_state();
        motion.unground(Some(&platform.surface), true);
        self.apply_ground_state(motion);
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
            self.grounded_speed = crate::player_controller::increment(
                self.grounded_speed,
                sign * self.constants.max_speed,
                self.constants.accel,
                DT,
            );
        }
        self.align(platform);
        let speed = self.grounded_speed + self.external_velocity.dot(forward);
        let velocity = forward * speed;
        let (allowed, _) = crate::platform_motion::movement_limit(
            self.position,
            self.radius,
            velocity,
            DT,
            &mut |origin, direction, distance| {
                closest(platforms, origin, direction, distance, Some(platform.id))
                    .map(|(_, near, _)| ((), near))
            },
        );
        let allowed = allowed.min(speed.abs()) * Fix::from_int(speed.sign());
        self.local = platform
            .surface
            .move_unblocked(self.local, self.radius, allowed, DT);
    }
    pub(crate) fn gravity_step(&mut self, platforms: &[Platform]) {
        if std::mem::take(&mut self.skip_next_gravity) {
            return;
        }
        if !self.grounded {
            crate::player_physics::apply_gravity(
                &mut self.velocity,
                self.constants.gravity_accel,
                self.constants.gravity_modifier,
                self.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid movement gravity constants");
            self.collision_rays(platforms);
        }
    }
    pub(crate) fn step(&mut self, action: Action, platforms: &[Platform]) {
        self.step_with_input(action, action.direction(), platforms);
    }

    /// Run the ordinary slime movement with `input` while keeping `action`'s
    /// joystick direction available to abilities such as Dash. Gamepad input
    /// uses the same vector for both; keyboard players use W/A/S/D for motion
    /// and the joystick byte independently for ability aim.
    pub(crate) fn step_with_input(&mut self, action: Action, input: Vec2, platforms: &[Platform]) {
        if !self.dash.active {
            self.gravity_step(platforms);
        }
        self.move_step(action, input, platforms);
    }
    /// SlimeController.UpdateSim after PlayerPhysics.UpdateSim has applied
    /// gravity: dash entry/continuation, jump, ground or air movement, then
    /// PlayerBody's integration.
    pub(crate) fn move_step(&mut self, action: Action, input: Vec2, platforms: &[Platform]) {
        let jump = action.buttons & 1 != 0;
        if self.dash.active {
            self.previous_jump = jump;
            self.dash_step(action, platforms);
            return;
        }
        self.cooldown += DT;
        if action.buttons & 2 != 0 && self.cooldown > Fix::from_int(4) {
            self.enter_dash(action, input, platforms);
            self.previous_jump = jump;
            self.dash_step(action, platforms);
            return;
        }
        if jump && !self.previous_jump && self.grounded {
            self.jump(platforms);
        }
        self.previous_jump = jump;
        self.move_only(input, platforms);
        if self.outside_bounds() {
            self.alive = false;
            return;
        }
        self.position += (self.velocity + self.external_velocity) * DT;
        self.external_velocity *= Fix::from_f64(0.95);
    }
    /// PlayerPhysics.Move (PlayerPhysics.cs:230-260) without PlayerBody's
    /// integration. SlimeController.UpdateSim still calls it on the tick
    /// OldUpdate entered an ability (SlimeController.cs:1067-1084), on the
    /// slime body the ability has already copied; that body is deactivated,
    /// so only its observed velocity (and a landing) change.
    pub(crate) fn move_only(&mut self, input: Vec2, platforms: &[Platform]) {
        if self.grounded {
            self.move_grounded(input, platforms);
        } else {
            crate::player_physics::air_movement(
                &mut self.velocity,
                input.x,
                self.constants.air_accel,
                self.constants.max_speed,
            )
            .expect("valid movement air constants");
            self.collision_rays(platforms);
        }
    }
    pub(crate) fn outside_bounds(&self) -> bool {
        self.position.y < Fix::from_f32(-11.3)
            || self.position.y > Fix::from_f64(72.5)
            || self.position.x < Fix::from_f64(-131.25)
            || self.position.x > Fix::from_f64(131.25)
    }
    pub(crate) fn push_from(&mut self, other: Vec2, lower_id: bool) {
        let Some(direction) = crate::player_collision::overlap_push_direction(
            self.position,
            self.up,
            self.radius,
            other,
            if lower_id { -1 } else { 0 },
        ) else {
            return;
        };
        self.external_velocity += direction * Fix::from_raw(21_474_836) * Fix::from_int(35);
    }
    pub(crate) fn sync_collider(&mut self) {
        let extents = if self.dash.active {
            Vec2::new(Fix::from_raw(5_626_406_912), Fix::from_raw(4_359_391_744))
        } else {
            Vec2::new(Fix::from_raw(5_712_306_688), Fix::from_raw(3_478_923_520))
        };
        let inv = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
        if self.rotation != self.collider_rotation || self.collider_basis == Vec2::ZERO {
            self.collider_basis = Vec2::angle(self.rotation);
            self.collider_rotation = self.rotation;
        }
        let basis = self.collider_basis;
        self.collider = OrientedBox {
            center: self.position,
            right: (Vec2::RIGHT / inv.x).complex_mul(basis),
            up: (Vec2::UP / inv.y).complex_mul(basis),
            inverse_extents: inv,
        };
    }
    /// Detach from the ground without the jump impulse (PlayerPhysics.UnGround).
    pub(crate) fn force_unground(&mut self, platforms: &[Platform]) {
        if let Some(platform) = self.platform(platforms) {
            let surface = platform.surface.clone();
            let mut motion = self.ground_state();
            motion.unground(Some(&surface), false);
            self.apply_ground_state(motion);
        }
    }
    pub(crate) fn late_step(&mut self, platforms: &[Platform]) {
        if self.alive && self.grounded && !self.dash.active {
            let platform = self.platform(platforms).unwrap();
            self.position = platform.surface.position(self.local, self.radius);
            self.align(platform);
        }
    }
}
