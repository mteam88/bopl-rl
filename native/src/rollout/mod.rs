//! Full-match rollout backend for the PufferLib trainer.
//!
//! Implements the bopl_rl.h ABI (see `rollout::abi`) on top of the verified
//! Level1 movement slice (`crate::world`), generalized to the five grass maps
//! with the full episode lifecycle: warmup, deaths, stocks, respawn, the
//! episode tick limit, and sudden death. Bow and Platform are ported
//! approximately from sim/Gameplay for transferable fidelity, not
//! byte-exactness; the .NET sim stays the reference (native/rl/RUST_ROLLOUTS.md).
use crate::{
    fixed::Fix,
    geometry::{Circle, OrientedBox, RoundedRect},
    physics::{Body, Contact, positional_correction, resolve},
    random::Random,
    surface::Surface,
    vector::Vec2,
    world::{Action, DT, DashState, Platform, PlatformBounds, Player, Terrain},
};

/// `Terrain` over this world's platforms, as a field-level borrow so a
/// player can be stepped (`&mut self.players[i]`) against it.
macro_rules! terrain {
    ($world:expr) => {
        Terrain {
            platforms: &$world.platforms,
            bounds: &$world.bounds,
        }
    };
}

pub mod abi;
pub mod data;

pub const ABI_VERSION: u32 = 2;
pub const MAP_COUNT: u32 = 5;
/// The .NET catalog has 39 ability icons; index 0 is the menu placeholder.
pub const ABILITY_COUNT: u32 = 39;
pub const MAX_PLAYERS: usize = 6;
/// Catalog indices with real behaviour in this backend.
pub const ABILITY_RANDOM: u32 = 1;
pub const ABILITY_DASH: u32 = 2;
pub const ABILITY_GRENADE: u32 = 3;
pub const ABILITY_BOW: u32 = 4;
pub const ABILITY_GUST: u32 = 7;
pub const ABILITY_MISSILE: u32 = 10;
pub const ABILITY_SPIKE: u32 = 11;
pub const ABILITY_TIME_STOP: u32 = 12;
pub const ABILITY_SMOKE: u32 = 13;
pub const ABILITY_PLATFORM: u32 = 14;
pub const ABILITY_REVIVAL: u32 = 15;
pub const ABILITY_INVISIBILITY: u32 = 19;
pub const ABILITY_MINE: u32 = 24;
pub const ABILITY_TELEPORT: u32 = 25;
pub const ABILITY_REWIND: u32 = 31;
pub const ABILITY_GOLD_MUSHROOM: u32 = 34;
pub const ABILITY_TACK: u32 = 35;
pub const ABILITY_MJOLNIR: u32 = 37;

// ---- C ABI mirror types (bopl_rl.h, verified against NativeApi.cs) ----

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CAction {
    pub angle: u8,
    pub reserved: u8,
    pub buttons: u16,
}
impl CAction {
    fn aim_action(self) -> Action {
        Action {
            angle: self.angle,
            buttons: (self.buttons & 0xFF) as u8,
        }
    }

    fn input_direction(self, keyboard: bool) -> Vec2 {
        if !keyboard {
            return self.aim_action().direction();
        }
        let x = i64::from(self.buttons & 512 != 0) - i64::from(self.buttons & 128 != 0);
        let y = i64::from(self.buttons & 64 != 0) - i64::from(self.buttons & 256 != 0);
        Vec2::new(Fix::from_int(x), Fix::from_int(y))
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CPlayerConfig {
    pub color: u32,
    pub team: u32,
    pub keyboard: u32,
    pub abilities: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CConfig {
    pub seed: u32,
    pub map: u32,
    pub players: u32,
    pub slots: u32,
    pub stocks: u32,
    pub mode: u32,
    pub warmup_ticks: u32,
    pub max_episode_ticks: u32,
    pub player: [CPlayerConfig; 6],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CPlayer {
    pub id: u32,
    pub team: u32,
    pub flags: u32,
    pub alive_bodies: u32,
    pub stocks: u32,
    pub kills: u32,
    pub deaths: u32,
    pub cause_of_death: u32,
    pub x: i64,
    pub y: i64,
    pub scale: i64,
    pub invisible_time: i64,
    pub ability_time: i64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CEntity {
    pub id: u32,
    pub kind: u32,
    pub layer: u32,
    pub owner: u32,
    pub flags: u32,
    pub hierarchy: u32,
    pub reserved: [u32; 2],
    pub x: i64,
    pub y: i64,
    pub vx: i64,
    pub vy: i64,
    pub right_x: i64,
    pub right_y: i64,
    pub up_x: i64,
    pub up_y: i64,
    pub radius: i64,
    pub rotation: i64,
    pub angular_velocity: i64,
    pub inverse_mass: i64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CSlime {
    pub id: u32,
    pub owner: u32,
    pub flags: u32,
    pub state: u32,
    pub x: i64,
    pub y: i64,
    pub vx: i64,
    pub vy: i64,
    pub rotation: i64,
    pub scale: i64,
    pub abilities: [u32; 3],
    pub ability_flags: [u32; 3],
    pub cooldowns: [i64; 3],
    pub timers: [i64; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CHeader {
    pub version: u32,
    pub tick: u32,
    pub episode_ticks: u32,
    pub flags: u32,
    pub player_count: u32,
    pub entity_count: u32,
    pub warmup_ticks: u32,
    pub slime_count: u32,
    pub simulation_time: i64,
    pub player_time_scale: i64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CTelemetry {
    pub ability_activations: [u32; 3],
    pub arrows_fired: u32,
    pub arrow_hits: u32,
    pub deaths_environment: u32,
    pub deaths_opponent: u32,
    pub deaths_self: u32,
    pub deaths_other: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CState {
    pub header: CHeader,
    pub players: [CPlayer; 6],
    pub entities: *const CEntity,
    pub slimes: *const CSlime,
    pub telemetry: [CTelemetry; 6],
}

pub const FLAG_TERMINAL: u32 = 1;
pub const FLAG_TRUNCATED: u32 = 2;
pub const FLAG_STARTED: u32 = 4;

// Entity flags (RlObservation.Collider).
const E_ACTIVE: u32 = 1;
const E_ENABLED: u32 = 2;
const E_DESTROYED: u32 = 4;
const E_TRIGGER: u32 = 8;
/// Collider belongs to a DetPhysics CompositeBody. This is exported by the
/// oracle for both the rounded-rect root and each attached child.
const E_COMPOSITE: u32 = 16;
/// Internal marker: DetPhysics dropped this collider. Objects destroyed by an
/// updatable are removed before the same pass's contacts and never observed;
/// one destroyed by a collision callback is observed once, flagged
/// `E_DESTROYED`, and removed on the next pass (DetPhysics.Simulate's
/// remove-destroyed step). Such rows are not emitted.
const E_REMOVED: u32 = 1 << 31;

// Slime flags (RlObservation.Slime).
const S_ACTIVE: u32 = 1;
const S_SPAWNED: u32 = 2;
const S_IN_ABILITY: u32 = 4;
const S_SUPPRESS_INPUT: u32 = 8;
const S_SIMULATED: u32 = 16;

// Slime states (SlimeController.State).
const STATE_IDLE: u32 = 0;
const STATE_RUN: u32 = 1;
const STATE_FALL: u32 = 2;
const STATE_LAND: u32 = 4;
const STATE_TURN: u32 = 6;
const STATE_STOP: u32 = 7;
const STATE_NOTHING: u32 = 8;
// SlimeController animation lengths driving the state machine (rollout.json
// player anims: land 6x6, stop 1x6, runTurnAround 4x3 ticks).
const LAND_TICKS: u32 = 36;
const STOP_TICKS: u32 = 6;
const TURN_TICKS: u32 = 12;

// Layers (sim/Export/settings.json).
const LAYER_PLAYER: u32 = 8;
const LAYER_PROJECTILE: u32 = 9;
const LAYER_WALL: u32 = 11;
const LAYER_WATER: u32 = 4;

const CAUSE_OTHER: u32 = 0;
const CAUSE_SQUISHED: u32 = 1;
const CAUSE_ACCIDENT: u32 = 2;
const CAUSE_NOT_DEAD_YET: u32 = 5;
/// Player.cs Constants.ENVIRONMENTAL_PLAYER_ID.
const ENVIRONMENT: u32 = 0xFFFF_FFFF;
/// PlatformTransform.cursorSensitivity.
const CURSOR_SENSITIVITY: Fix = Fix::from_raw(107_374_182_400); // 25.0

fn clamp_revival_position(mut position: Vec2) -> Vec2 {
    // SceneBounds.cs constants are authored through single-precision casts.
    let margin = Fix::from_int(2);
    position.x = position.x.clamp(
        Fix::from_raw(-417_771_454_464) + margin,
        Fix::from_raw(419_188_801_536) - margin,
    );
    position.y = position.y.clamp(
        Fix::from_raw(-111_669_149_696) + margin,
        Fix::from_raw(171_798_691_840) - margin,
    );
    position.y = position.y.max(Fix::from_raw(-48_533_131_264));
    position
}

fn retained_ground(player: &PlayerSlot, platforms: &[Platform]) -> Option<u32> {
    if let Some(ground) = player.base.ground {
        return Some(ground);
    }
    // The ordinary controller can finish a landing animation with its public
    // Idle state one pass before this slice restores `ground`. PlayerInfo in
    // the game still carries the attached platform into the ability prefab.
    if player.obs_state == STATE_FALL || player.obs_state == STATE_NOTHING {
        return None;
    }
    let mut best = None;
    let mut best_error = Fix::from_f32(0.1);
    for platform in platforms {
        let Ok(surface_point) = platform.surface.closest_point(player.base.position) else {
            continue;
        };
        let error = ((player.base.position - surface_point).length() - player.base.radius).abs();
        if error < best_error {
            best_error = error;
            best = Some(platform.id);
        }
    }
    best
}

/// Slime controllers wake this many ticks into the warmup (measured against
/// the .NET reset: ability cooldown timers read -1.5s + (240-62)/60 = 1.4667s
/// after a 240-tick warmup).
const SLIME_AWAKE_TICK: u32 = 62;
const TIMER_START: Fix = Fix::from_raw(-6_442_450_944); // -1.5s

/// A live arrow (Arrow.cs + Projectile.cs): a ballistic projectile that kills
/// players on contact and sticks to supported non-player geometry. Ordinary
/// PlayerBody hurtboxes do not satisfy Arrow's `BoplBody` check, so an arrow
/// keeps flying after killing a slime. A stuck arrow's collider is destroyed
/// in the .NET sim, so it vanishes from the observation; here the slot is
/// simply freed.
#[derive(Clone, Copy, Default)]
pub struct Arrow {
    pub active: bool,
    /// Instantiated this tick, but not added to DetPhysics/updatables until
    /// the next scheduler pass.
    pub registered: bool,
    pub born_tick: u32,
    pub id: u32,
    /// Owning player id (1-based).
    pub owner: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub velocity: Vec2,
    /// Arrow.UpdateSim aligns the sprite before DetPhysics applies gravity.
    pub rotation: Fix,
    pub time_passed: Fix,
    /// Projectile.DelayedEnableHurtOwner: 5 units of travel time.
    pub hurt_owner_after: Fix,
    pub can_hurt_owner: bool,
}

/// Shared `ThrowItem2` payload state. Grenade is the first consumer; Smoke,
/// Poke Ball, Tack, and Mjolnir use the same held-item/throw lifecycle.
#[derive(Clone, Copy, Default)]
pub struct ThrownItem {
    pub active: bool,
    pub id: u32,
    pub owner: usize,
    pub ability: u32,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub body: Body,
    pub fuse: Fix,
    pub hurt_owner_delay: Fix,
    pub detonates_on_owner: bool,
    pub age: Fix,
    pub launched: bool,
    pub settled: bool,
    pub owner_clear: bool,
    pub settle_position: Vec2,
    /// Ability-specific phase (Mjolnir: 0 outbound, 1 returning).
    pub phase: u8,
    /// Mjolnir's allocation-free equivalent of `HashSet<int> alreadyKilled`.
    pub killed_players: u8,
    /// Mine.Init/LateUpdateSim state. Keeping it in the pooled item avoids a
    /// side allocation for every autonomous mine.
    pub random_offset: Fix,
    pub time_since_anticipation: Fix,
    pub prev_chase_dir: Vec2,
    pub prev_rand_vector: Vec2,
    pub prev_velocity_change: Vec2,
}

#[derive(Clone, Copy, Default)]
pub struct GrenadeExplosion {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub owner: usize,
    pub age: u32,
}

/// SparkShot's guided projectile. The projectile remains live after the
/// ability form exits and switches from stabilized flight to its own thrust.
#[derive(Clone, Copy, Default)]
pub struct GuidedMissile {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub owner: usize,
    pub position: Vec2,
    pub body: Body,
    pub up: Vec2,
    pub instability_ticks: u32,
}

#[derive(Clone, Copy, Default)]
pub struct MissileExplosion {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub owner: usize,
    pub credited_owner: bool,
    pub age: u32,
}

#[derive(Clone, Copy, Default)]
pub struct SmokeCloud {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub rotation: Fix,
    pub scale: Fix,
    pub owner: usize,
    pub ability_slot: usize,
    pub age_ticks: u32,
    pub ignition_time: Fix,
    pub ignition_delay: Fix,
}

#[derive(Clone, Copy, Default)]
pub struct SmokeExplosion {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub position: Vec2,
    pub scale: Fix,
    pub owner: usize,
    pub age: u32,
}

#[derive(Clone, Copy, Default)]
pub struct TimeStopEffect {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub caster: usize,
    pub seconds_elapsed: f32,
}

/// SpikeAttack's manually initialized box, attached to a rounded platform as
/// a DetPhysics composite child. Positions are stored in the platform body's
/// frame so platform motion never requires a separate rigid-body integration.
#[derive(Clone, Copy, Default)]
pub struct SpawnedSpike {
    pub active: bool,
    pub born_tick: u32,
    pub hierarchy_base: u32,
    pub entity: Option<usize>,
    pub platform_index: usize,
    pub local: Fix,
    pub relative_center: Vec2,
    pub relative_rotation: Fix,
    pub surface_offset: Fix,
    pub time: Fix,
    pub body: Body,
}

#[derive(Clone, Copy, Default)]
pub struct CastSpellState {
    /// 0: grounded intro, 1: channel, 2: cast animation, 3: ease-out.
    pub phase: u8,
    pub ticks: u32,
    pub time: Fix,
    /// Pose synced by DetPhysics before the updatable callbacks in this pass.
    pub collider_position: Vec2,
    pub collider_right: Vec2,
    pub collider_rotation: Fix,
}

#[derive(Clone, Copy, Default)]
pub struct SpikeCastState {
    /// 0: cast animation, 1: recover animation.
    pub phase: u8,
    pub ticks: u32,
    pub collider_position: Vec2,
    pub collider_up: Vec2,
    pub collider_rotation: Fix,
}

#[derive(Clone, Copy, Default)]
pub struct MissileFormState {
    /// 0: enter, 1: enter-to-idle, 2: idle, 3: exit.
    pub phase: u8,
    pub ticks: u32,
    pub projectile: Option<usize>,
}

#[derive(Clone, Copy, Default)]
pub struct ThrowItemState {
    /// 0: enter animation, 1: exit animation.
    pub phase: u32,
    pub ticks: u32,
    pub grounded: bool,
    pub has_inputted_fire: bool,
    pub has_fired: bool,
    pub time_passed: Fix,
    pub fire_input_time: Fix,
    pub direction: Vec2,
    pub item: Option<usize>,
}

#[derive(Clone, Copy)]
struct ThrowConfig<'a> {
    hurtbox_extents: Vec2,
    air_position: Vec2,
    ground_position: Vec2,
    ground_dummy: &'a [Vec2],
    air_dummy: &'a [Vec2],
    hurt_owner_delay: Fix,
    max_force: Fix,
    min_force: Fix,
    force_gain_speed: Fix,
    max_slow_mo_speed: Fix,
    knockback: Fix,
    constants: &'a crate::world::MovementConstants,
    ground_enter_ticks: u32,
    ground_exit_ticks: u32,
    air_enter_ticks: u32,
    air_exit_ticks: u32,
    item_layer: u32,
    item_updatable_count: u32,
    item_collider_hierarchy: u32,
    item_radius: Fix,
    item_body: Body,
}

fn throw_config(data: &data::RolloutData, ability: u32) -> ThrowConfig<'_> {
    if ability == ABILITY_SMOKE {
        let d = &data.smoke;
        ThrowConfig {
            hurtbox_extents: d.hurtbox_extents,
            air_position: d.air_position,
            ground_position: d.ground_position,
            ground_dummy: &d.ground_dummy,
            air_dummy: &d.air_dummy,
            hurt_owner_delay: d.hurt_owner_delay,
            max_force: d.max_force,
            min_force: d.min_force,
            force_gain_speed: d.force_gain_speed,
            max_slow_mo_speed: d.max_slow_mo_speed,
            knockback: d.knockback,
            constants: &d.constants,
            ground_enter_ticks: d.ground_enter_ticks,
            ground_exit_ticks: d.ground_exit_ticks,
            air_enter_ticks: d.air_enter_ticks,
            air_exit_ticks: d.air_exit_ticks,
            item_layer: d.item_layer,
            item_updatable_count: d.item_updatable_count,
            item_collider_hierarchy: d.item_collider_hierarchy,
            item_radius: d.item_radius,
            item_body: d.item_body,
        }
    } else if ability == ABILITY_MJOLNIR {
        let d = &data.mjolnir;
        ThrowConfig {
            hurtbox_extents: d.hurtbox_extents,
            air_position: d.air_position,
            ground_position: d.ground_position,
            ground_dummy: &d.ground_dummy,
            air_dummy: &d.air_dummy,
            hurt_owner_delay: d.hurt_owner_delay,
            max_force: d.max_force,
            min_force: d.min_force,
            force_gain_speed: d.force_gain_speed,
            max_slow_mo_speed: d.max_slow_mo_speed,
            knockback: d.knockback,
            constants: &d.constants,
            ground_enter_ticks: d.ground_enter_ticks,
            ground_exit_ticks: d.ground_exit_ticks,
            air_enter_ticks: d.air_enter_ticks,
            air_exit_ticks: d.air_exit_ticks,
            item_layer: d.item_layer,
            item_updatable_count: d.item_updatable_count,
            item_collider_hierarchy: d.item_collider_hierarchy,
            item_radius: d.item_radius,
            item_body: d.item_body,
        }
    } else if ability == ABILITY_TACK {
        let d = &data.tack;
        ThrowConfig {
            hurtbox_extents: d.hurtbox_extents,
            air_position: d.air_position,
            ground_position: d.ground_position,
            ground_dummy: &d.ground_dummy,
            air_dummy: &d.air_dummy,
            hurt_owner_delay: d.hurt_owner_delay,
            max_force: d.max_force,
            min_force: d.min_force,
            force_gain_speed: d.force_gain_speed,
            max_slow_mo_speed: d.max_slow_mo_speed,
            knockback: d.knockback,
            constants: &d.constants,
            ground_enter_ticks: d.ground_enter_ticks,
            ground_exit_ticks: d.ground_exit_ticks,
            air_enter_ticks: d.air_enter_ticks,
            air_exit_ticks: d.air_exit_ticks,
            item_layer: d.item_layer,
            item_updatable_count: d.item_updatable_count,
            item_collider_hierarchy: d.item_collider_hierarchy,
            item_radius: d.item_radius,
            item_body: d.item_body,
        }
    } else {
        let d = &data.grenade;
        ThrowConfig {
            hurtbox_extents: d.hurtbox_extents,
            air_position: d.air_position,
            ground_position: d.ground_position,
            ground_dummy: &d.ground_dummy,
            air_dummy: &d.air_dummy,
            hurt_owner_delay: d.hurt_owner_delay,
            max_force: d.max_force,
            min_force: d.min_force,
            force_gain_speed: d.force_gain_speed,
            max_slow_mo_speed: d.max_slow_mo_speed,
            knockback: d.knockback,
            constants: &d.constants,
            ground_enter_ticks: d.ground_enter_ticks,
            ground_exit_ticks: d.ground_exit_ticks,
            air_enter_ticks: d.air_enter_ticks,
            air_exit_ticks: d.air_exit_ticks,
            item_layer: d.item_layer,
            item_updatable_count: d.item_updatable_count,
            item_collider_hierarchy: d.item_collider_hierarchy,
            item_radius: d.item_radius,
            item_body: d.item_body,
        }
    }
}

const REWIND_CAPACITY: usize = 188;
const REWIND_MAX_SEGMENTS: usize = 15;
const REWIND_ARC_SAMPLES: usize = REWIND_MAX_SEGMENTS * 8 + 1;

#[derive(Clone, Copy, Default)]
pub struct RewindSample {
    pub position: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub rotation: Fix,
    pub cooldowns: [Fix; 3],
}

#[derive(Clone)]
pub struct RewindState {
    pub ring: [RewindSample; REWIND_CAPACITY],
    pub head: usize,
    pub count: usize,
    pub rewinding: bool,
    pub playback_ticks: u32,
    pub playback_tick: u32,
    pub ability_slot: usize,
    /// `PlayerPhysics.UnGround` deliberately retains its attached platform;
    /// ReleaseControl calls it again after arrival and repeats the 0.05 push.
    pub attached_ground: Option<u32>,
    pub finish_next_tick: bool,
    pub destination: RewindSample,
    pub path_segments: usize,
    pub path_positions: [Vec2; REWIND_MAX_SEGMENTS + 1],
    pub path_rotations: [Fix; REWIND_MAX_SEGMENTS + 1],
    pub arc_lengths: [Fix; REWIND_ARC_SAMPLES],
    pub arc_count: usize,
    pub total_arc_length: Fix,
}

impl Default for RewindState {
    fn default() -> Self {
        Self {
            ring: [RewindSample::default(); REWIND_CAPACITY],
            head: 0,
            count: 0,
            rewinding: false,
            playback_ticks: 0,
            playback_tick: 0,
            ability_slot: 0,
            attached_ground: None,
            finish_next_tick: false,
            destination: RewindSample::default(),
            path_segments: 0,
            path_positions: [Vec2::ZERO; REWIND_MAX_SEGMENTS + 1],
            path_rotations: [Fix::ZERO; REWIND_MAX_SEGMENTS + 1],
            arc_lengths: [Fix::ZERO; REWIND_ARC_SAMPLES],
            arc_count: 0,
            total_arc_length: Fix::ZERO,
        }
    }
}

impl RewindState {
    fn push(&mut self, sample: RewindSample) {
        self.ring[self.head] = sample;
        self.head = (self.head + 1) % REWIND_CAPACITY;
        self.count = (self.count + 1).min(REWIND_CAPACITY);
    }

    fn sample_back(&self, ticks: usize) -> RewindSample {
        self.ring[(self.head + REWIND_CAPACITY - 1 - ticks) % REWIND_CAPACITY]
    }

    fn evaluate_spline(&self, t: Fix) -> (Vec2, Fix) {
        let s = t * Fix::from_int(self.path_segments as i64);
        let segment = (s.to_int().max(0) as usize).min(self.path_segments - 1);
        let u = s - Fix::from_int(segment as i64);
        let p0 = self.path_positions[segment.saturating_sub(1)];
        let p1 = self.path_positions[segment];
        let p2 = self.path_positions[segment + 1];
        let p3 = self.path_positions[(segment + 2).min(self.path_segments)];
        let u2 = u * u;
        let u3 = u2 * u;
        let half = Fix::from_f32(0.5);
        let position = half
            * (Fix::from_int(2) * p1
                + (p2 - p0) * u
                + (Fix::from_int(2) * p0 - Fix::from_int(5) * p1 + Fix::from_int(4) * p2 - p3)
                    * u2
                + (Fix::from_int(3) * (p1 - p2) + p3 - p0) * u3);
        let rotation =
            self.path_rotations[segment] * (Fix::ONE - u) + self.path_rotations[segment + 1] * u;
        (position, rotation)
    }

    fn evaluate_path(&self, t: Fix) -> (Vec2, Fix) {
        if self.total_arc_length <= Fix::ZERO {
            return self.evaluate_spline(t);
        }
        let target = t * self.total_arc_length;
        let mut lo = 0usize;
        let mut hi = self.arc_count - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.arc_lengths[mid] <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = self.arc_lengths[hi] - self.arc_lengths[lo];
        let fraction = if span > Fix::ZERO {
            (target - self.arc_lengths[lo]) / span
        } else {
            Fix::ZERO
        };
        self.evaluate_spline(
            (Fix::from_int(lo as i64) + fraction) / Fix::from_int((self.arc_count - 1) as i64),
        )
    }

    fn begin(&mut self, position: Vec2, rotation: Fix, ability_slot: usize) -> bool {
        if self.rewinding || self.count < 2 {
            return false;
        }
        let ticks_back = 150usize.min(self.count - 1);
        self.playback_ticks = ((ticks_back * 17) / 25).max(1) as u32;
        self.playback_tick = 0;
        self.ability_slot = ability_slot;
        self.finish_next_tick = false;
        self.destination = self.sample_back(ticks_back);
        self.path_segments = (ticks_back / 10).max(1);
        self.path_positions[0] = position;
        self.path_rotations[0] = rotation;
        for i in 1..=self.path_segments {
            let sample = self.sample_back(ticks_back * i / self.path_segments);
            self.path_positions[i] = sample.position;
            self.path_rotations[i] = sample.rotation;
        }
        self.arc_count = self.path_segments * 8 + 1;
        self.arc_lengths[0] = Fix::ZERO;
        let mut previous = self.path_positions[0];
        for i in 1..self.arc_count {
            let t = Fix::from_int(i as i64) / Fix::from_int((self.arc_count - 1) as i64);
            let current = self.evaluate_spline(t).0;
            self.arc_lengths[i] = self.arc_lengths[i - 1] + (current - previous).length();
            previous = current;
        }
        self.total_arc_length = self.arc_lengths[self.arc_count - 1];
        self.rewinding = true;
        true
    }
}

/// A platform spawned by the Platform ability (ResizablePlatform prefab).
/// Each player keeps at most one (PlatformTransform's one-slot srrBuffer);
/// a re-cast removes the previous platform.
#[derive(Clone, Copy, Default)]
pub struct SpawnedPlatform {
    pub active: bool,
    /// ResizablePlatform.RemovePlatform keeps the body live while shrinking.
    pub being_removed: bool,
    /// Owning player index (not id).
    pub owner: usize,
    /// Index into `RolloutWorld::platforms` (kept in sync on removal).
    pub platform_index: usize,
    pub sudden_death_delay: Fix,
    pub removal_time: Fix,
    pub removal_extents: Vec2,
    pub removal_radius: Fix,
}

/// DetPhysics body plus AnimateVelocity state for one rounded platform.
#[derive(Clone, Copy, Default)]
pub struct PlatformDynamics {
    pub body: Body,
    /// The rounded rectangle's own serialized body. `body` becomes the
    /// effective CompositeBody while SpikeAttack children are attached; the
    /// observation bridge still exports this child body unchanged.
    pub root_body: Body,
    pub attached_spikes: u32,
    /// CompositeBody.position is separate from the rounded-rectangle child
    /// centre. Box/RR positional correction can move the former without
    /// moving the latter until a later operation rebuilds the composite.
    pub composite_position: Vec2,
    /// FixTransform rotation written by a component this tick.
    /// DetPhysics.Simulate copies it into the body only after AnimateVelocity
    /// has computed its torque against the stale orientation
    /// (DetPhysics.cs:2504-2506).
    pub transform_rotation: Option<Fix>,
    /// Tick on which Updater adds a body registered mid-tick and calls
    /// AnimateVelocity.Init (Updater.cs:265-273): the spring homes on the
    /// position physics had already corrected on the spawn tick.
    pub init_tick: Option<u32>,
    pub home_position: Vec2,
    pub home_rotation: Fix,
    pub speed: Fix,
    pub mu: Fix,
    pub rotation_speed: Fix,
    pub rotation_mu: Fix,
    /// `(orientation, Vec2::angle(orientation))` from the last integration:
    /// a platform that translates without rotating (sudden death, a spring
    /// settling) keeps its basis instead of consulting the sine table twice.
    pub basis: Option<(Fix, Vec2)>,
}

/// PlatformTransform.State.
pub const PLATFORM_ANTICIPATION: u32 = 0;
pub const PLATFORM_ROTATION: u32 = 1;
pub const PLATFORM_SIZE: u32 = 2;
pub const PLATFORM_EXIT: u32 = 3;

/// Per-player bow ability state (BowTransform.cs).
#[derive(Clone, Copy, Default)]
pub struct BowState {
    /// Ticks since the ability was entered ("transform" starts at 0).
    pub ticks: u32,
    pub aim: Vec2,
    pub released: bool,
    pub has_fired: bool,
    /// Remaining ticks of the fireGround/fireAir animation before exit.
    pub fire_ticks_left: u32,
    pub loading_frame: u32,
}

/// Per-player platform ability state (PlatformTransform.cs).
#[derive(Clone, Copy, Default)]
pub struct PlatformState {
    pub phase: u32,
    pub ticks: u32,
    /// Seconds since CreatePlatform (the .NET timePassed).
    pub time_passed: Fix,
    pub exit_time: Fix,
    pub released: bool,
    pub cursor: Vec2,
    pub active_corner: Vec2,
    /// PlatformTransform's PlayerBody.right, authored by assigning `body.up`
    /// during State.Rotation and then held constant through State.Size.
    pub sizing_right: Vec2,
    /// Index into `RolloutWorld::platforms` for the platform being shaped.
    pub platform_index: usize,
    pub hurtbox_disabled: bool,
}

/// One SuperStarEffect spawned by a Gold Mushroom slot. The source cannot
/// overlap two effects from the same slot because its cooldown is held at the
/// 10,000-second sentinel until this state ends.
#[derive(Clone, Copy, Default)]
pub struct GoldMushroomState {
    pub active: bool,
    pub pending_init: bool,
    pub time: Fix,
}

#[derive(Clone, Copy, Default)]
pub struct GustState {
    /// Zero while playing castAir, one while playing exit.
    pub phase: u32,
    pub ticks: u32,
    /// PlayerPhysics.UnGround deliberately leaves attachedGround populated.
    /// QuickSpell calls it again on exit and receives a second separation
    /// along the platform's then-current normal.
    pub attached_ground: Option<u32>,
}

/// Revival uses the same QuickSpell animation/body handoff as Gust.
#[derive(Clone, Copy, Default)]
pub struct RevivalState {
    pub phase: u32,
    pub ticks: u32,
    pub attached_ground: Option<u32>,
}

/// A RevivePositionIndicator spawned by one Revival ability instance.
#[derive(Clone, Copy, Default)]
pub struct RevivalMarker {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub owner: usize,
    pub ability_slot: usize,
    pub entity: Option<usize>,
    pub hierarchy_base: u32,
    pub position: Vec2,
    pub velocity: Vec2,
}

/// The portal indicator retained by one Teleport ability instance between
/// its first and second instant casts.
#[derive(Clone, Copy, Default)]
pub struct TeleportMarker {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub owner: usize,
    pub ability_slot: usize,
    pub entity: Option<usize>,
    pub hierarchy_base: u32,
    pub position: Vec2,
    pub velocity: Vec2,
}

/// AirWave_Shockwave has no collider of its own. It captures rigid targets on
/// initialization and keeps applying its curve to that fixed set.
#[derive(Clone, Copy)]
pub struct GustWave {
    pub active: bool,
    pub registered: bool,
    pub born_tick: u32,
    pub position: Vec2,
    pub time: Fix,
    pub target_count: usize,
    pub platform_ids: [u32; 50],
}
impl Default for GustWave {
    fn default() -> Self {
        Self {
            active: false,
            registered: false,
            born_tick: 0,
            position: Vec2::ZERO,
            time: Fix::ZERO,
            target_count: 0,
            platform_ids: [0; 50],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActiveAbility {
    None,
    Dash,
    Bow,
    Grenade,
    Tack,
    Mjolnir,
    Smoke,
    Missile,
    Mine,
    Spike,
    TimeStop,
    Gust,
    Revival,
    Platform,
}

fn active_throw_ability(active: ActiveAbility) -> u32 {
    match active {
        ActiveAbility::Tack => ABILITY_TACK,
        ActiveAbility::Mjolnir => ABILITY_MJOLNIR,
        ActiveAbility::Smoke => ABILITY_SMOKE,
        _ => ABILITY_GRENADE,
    }
}

/// One player's match state beyond the movement body.
#[derive(Clone)]
pub struct PlayerSlot {
    pub base: Player,
    pub team: u32,
    pub keyboard: bool,
    pub stocks: u32,
    pub kills: u32,
    pub deaths: u32,
    pub cause_of_death: u32,
    pub won: bool,
    pub out: bool, // dead with no stock left
    /// Ability catalog ids per slot, and which slots this backend simulates.
    pub abilities: [u32; 3],
    pub timers: [Fix; 3],
    pub cooldowns: [Fix; 3],
    pub active: ActiveAbility,
    pub active_slot: usize,
    /// The original SlimeController stays at the ability entry point while
    /// the active ability's separate body moves.
    pub ability_origin: Vec2,
    pub ability_origin_velocity: Vec2,
    pub ability_origin_rotation: Fix,
    pub ability_origin_state: u32,
    pub ability_entry_action: Action,
    /// Player.isInvisible and its two clocks. `was_invisible` belongs to the
    /// currently active SlimeController and drives cooldown restoration.
    pub invisible: bool,
    pub invisible_time: Fix,
    pub was_invisible: bool,
    pub ability_time: Fix,
    /// SlimeController/Ability publishes Player.Position before the live body
    /// advances later in the same updater pass.
    pub reported_position: Vec2,
    pub reported_tick: u32,
    /// A platform destroyed later in the previous updater pass. Unity keeps
    /// the rider grounded until its own next update, then separates it using
    /// the destroyed StickyRoundedRectangle's last surface normal.
    pub removed_ground_normal: Option<Vec2>,
    pub bow: BowState,
    pub missile: MissileFormState,
    pub throw_item: ThrowItemState,
    pub cast_spell: CastSpellState,
    pub spike_cast: SpikeCastState,
    /// `Spike.currentSpike` belongs to each equipped ability object. The
    /// spawned attack survives form exit, while recasting this same slot
    /// destroys and replaces it.
    pub spike_attacks: [Option<usize>; 3],
    pub platform: PlatformState,
    pub gust: GustState,
    pub revival: RevivalState,
    /// Marker owned by each equipped Revival instance. Recasting the same
    /// instance ends its previous marker before creating another.
    pub revival_markers: [Option<usize>; 3],
    /// A replacement clone is queued on death, then held hidden while the
    /// ReviveEffect wisp and doubly-registered particle reach this marker.
    pub revival_clone_pending: bool,
    /// A stock death instantiates its replacement in the callback, but the
    /// new prefab's colliders and SlimeController register next pass.
    pub stock_clone_pending: bool,
    pub revival_target_marker: Option<usize>,
    pub revival_effect_ticks: u32,
    pub revival_particle_ticks: u32,
    /// Marker owned by each equipped Teleport instance. The first cast fills
    /// it; the second consumes it and queues the two-way area swap.
    pub teleport_markers: [Option<usize>; 3],
    pub teleport_pending_marker: Option<usize>,
    pub teleport_pending_position: Vec2,
    /// TeleportedObjectEffect disables the moved GameObject while its visual
    /// copy shakes. The visual has no observation row, but its RNG and strict
    /// duration boundary are simulation-visible.
    pub teleport_effect_active: bool,
    pub teleport_effect_time: Fix,
    pub gold_mushroom: [GoldMushroomState; 3],
    pub rewind: RewindState,
    /// SlimeController.currentState as the policy observes it. The .NET enum
    /// never takes the `ability` value (nothing assigns it), so the state
    /// freezes at its pre-ability value while an ability runs.
    pub obs_state: u32,
    /// Countdown for the land/stop/turn animation states.
    pub state_ticks: u32,
    pub facing_right: bool,
    /// Ticks until GameSessionHandler.SpawnPlayers instantiates the slime
    /// (only a warmup shorter than SLIME_AWAKE_TICK+1 ends before it).
    pub unspawned: u32,
    /// Respawn countdown while holding a consumed stock (StockRespawn).
    pub respawn_ticks: u32,
    /// ReSpawnFast's accelerated spawn animation after a choice is confirmed.
    pub respawn_spawn_ticks: u32,
    pub respawn_selected: usize,
    pub stock_indicator_entities: [usize; 3],
    /// Player.InputProfile's frame-edge state is updated even without a live
    /// controllable slime and is what the stock chooser reads.
    pub input_jump_was_down: bool,
    /// Last movement input accepted by Player.ForceSetInputProfile. New
    /// packets are ignored for an unprotected player during Time Stop.
    pub last_movement_input: Vec2,
    pub telemetry: CTelemetry,
    // Fixed entity table indices.
    pub circle_entity: usize,
    pub gust_entity: usize,
    pub revival_entity: usize,
    pub body_entity: usize,
    pub dash_entity: usize,
    pub bow_entity: usize,
    pub throw_entity: usize,
    pub mine_entity: usize,
    pub spike_entity: usize,
    pub time_stop_entity: usize,
}

impl PlayerSlot {
    /// SlimeController.ExitAbility enters Fall and raycasts at once
    /// (SlimeController.cs:249, 263-272). A landing on this pass begins the
    /// land animation after the slime's SpriteAnimator (hierarchy 100000007,
    /// before every ability component) has already run, so it ends one tick
    /// later than a landing in PlayerPhysics.UpdateSim; `update_obs_state`
    /// adds this marker to LAND_TICKS.
    fn exit_to_fall(&mut self) {
        self.obs_state = STATE_FALL;
        self.state_ticks = 1;
    }
}

/// Per-tick working storage of `platforms_tick`/`platform_pairs` (contents
/// are meaningless between ticks; never copied by `fork_from`).
#[derive(Default)]
struct PlatformScratch {
    others: Vec<Vec2>,
    bounds: Vec<[Vec2; 2]>,
    active: Vec<usize>,
    window: Vec<usize>,
    pairs: Vec<[usize; 2]>,
    callback_colliders: Vec<RoundedRect>,
    contacts: Vec<(usize, usize, Contact)>,
    spike_wall_contacts: Vec<(usize, usize, Contact)>,
    squished: Vec<usize>,
}

pub struct RolloutWorld {
    pub valid: bool,
    pub failed: bool,
    pub started: bool,
    pub terminal: bool,
    pub map: usize,
    pub stocks: u32,
    pub warmup: u32,
    pub limit: u32,
    pub tick: u32,
    pub episode_ticks: u32,
    pub rng: Random,
    pub players: Vec<PlayerSlot>,
    pub platforms: Vec<Platform>,
    /// Raycast broad phase over `platforms` (`rebuild_bounds`).
    pub bounds: PlatformBounds,
    /// Incremented by every mutation of `platforms`/`platform_dynamics`
    /// (`platforms_changed`). The derived data below is refreshed only when
    /// its recorded epoch differs, so an undisturbed level costs nothing per
    /// tick.
    platform_epoch: u64,
    /// `platform_epoch` the broad phase was built for.
    bounds_epoch: u64,
    /// `platform_epoch` the platform entity rows were written for.
    platform_rows_epoch: u64,
    /// `platform_epoch` at which `platforms_tick` last found every body at
    /// rest at its home, so the check need not be repeated until a change.
    quiet_epoch: u64,
    /// Buffers of the platform physics stage, kept between ticks so a level
    /// with moving platforms does not allocate on every tick.
    scratch: PlatformScratch,
    /// Parallel to `platforms`: rigid body and home spring state.
    pub platform_dynamics: Vec<PlatformDynamics>,
    /// Parallel to `platforms`: index into the map's PlatformDef list, or
    /// usize::MAX for spawned (ability) platforms.
    pub platform_defs: Vec<usize>,
    /// Parallel to `platforms`: index into the entity table.
    pub platform_entities: Vec<usize>,
    pub arrows: Vec<Arrow>,
    pub thrown_items: Vec<ThrownItem>,
    pub guided_missiles: Vec<GuidedMissile>,
    pub missile_explosions: Vec<MissileExplosion>,
    pub grenade_explosions: Vec<GrenadeExplosion>,
    pub smoke_clouds: Vec<SmokeCloud>,
    pub smoke_explosions: Vec<SmokeExplosion>,
    pub time_stop_effects: Vec<TimeStopEffect>,
    pub spike_attacks: Vec<SpawnedSpike>,
    pub player_time_scale: Fix,
    pub time_stop_protected: [bool; MAX_PLAYERS],
    pub spawned: Vec<SpawnedPlatform>,
    pub gust_waves: [GustWave; MAX_PLAYERS * 3],
    pub revival_marker_pool: Vec<RevivalMarker>,
    pub teleport_marker_pool: Vec<TeleportMarker>,
    pub sudden_death: bool,
    pub sudden_death_tick: u32,
    pub sudden_death_delays: Vec<Fix>,
    /// AntiLockPlatform runtime (map 2): platform index and waypoint.
    pub orbits: Vec<(usize, usize)>,
    pub next_platform_id: u32,
    pub next_entity_id: u32,
    /// Base hierarchy for the next prefab instantiated after player spawn.
    pub next_spawn_hierarchy: u32,
    // Observation buffers. `entities`/`slimes` back the pointers in `state`.
    pub entities: Vec<CEntity>,
    pub slimes: Vec<CSlime>,
    pub state: Box<CState>,
    /// Emitted rows (the fixed table minus `E_REMOVED` entries).
    pub state_entities: Vec<CEntity>,
    pub sample_entities: Vec<CEntity>,
    pub sample_slimes: Vec<CSlime>,
    pub sample: Box<CState>,
    pub sample_valid: bool,
    /// Registered actions per simulated tick (warmup included), for the
    /// BOPLREP2 replay export.
    pub recording: Vec<[CAction; MAX_PLAYERS]>,
    /// Config captured at reset for the replay header.
    pub replay_config: CConfig,
    /// Encoded replay archive (bopl_rl_replay hands out this buffer).
    pub replay_buffer: Vec<u8>,
    pub error: [u8; 512],
}

fn empty_state() -> Box<CState> {
    Box::new(CState {
        header: CHeader::default(),
        players: [CPlayer::default(); 6],
        entities: core::ptr::null(),
        slimes: core::ptr::null(),
        telemetry: [CTelemetry::default(); 6],
    })
}

impl RolloutWorld {
    pub fn new() -> Self {
        Self {
            valid: false,
            failed: false,
            started: false,
            terminal: false,
            map: 0,
            stocks: 0,
            warmup: 0,
            limit: 0,
            tick: 0,
            episode_ticks: 0,
            rng: Random::new(1),
            players: Vec::with_capacity(MAX_PLAYERS),
            platforms: Vec::with_capacity(64),
            bounds: PlatformBounds::default(),
            platform_epoch: 1,
            bounds_epoch: 0,
            platform_rows_epoch: 0,
            quiet_epoch: 0,
            scratch: PlatformScratch::default(),
            platform_dynamics: Vec::with_capacity(64),
            platform_defs: Vec::with_capacity(64),
            platform_entities: Vec::with_capacity(64),
            arrows: Vec::with_capacity(64),
            thrown_items: Vec::with_capacity(64),
            guided_missiles: Vec::with_capacity(64),
            missile_explosions: Vec::with_capacity(64),
            grenade_explosions: Vec::with_capacity(64),
            smoke_clouds: Vec::with_capacity(64),
            smoke_explosions: Vec::with_capacity(64),
            time_stop_effects: Vec::with_capacity(8),
            spike_attacks: Vec::with_capacity(32),
            player_time_scale: Fix::ONE,
            time_stop_protected: [false; MAX_PLAYERS],
            spawned: Vec::with_capacity(64),
            gust_waves: [GustWave::default(); MAX_PLAYERS * 3],
            revival_marker_pool: Vec::with_capacity(64),
            teleport_marker_pool: Vec::with_capacity(64),
            sudden_death: false,
            sudden_death_tick: 0,
            sudden_death_delays: Vec::with_capacity(64),
            orbits: Vec::with_capacity(4),
            next_platform_id: 0,
            next_entity_id: 0,
            next_spawn_hierarchy: 0,
            entities: Vec::with_capacity(192),
            slimes: Vec::with_capacity(MAX_PLAYERS),
            state: empty_state(),
            state_entities: Vec::with_capacity(192),
            sample_entities: Vec::with_capacity(192),
            sample_slimes: Vec::with_capacity(MAX_PLAYERS),
            sample: empty_state(),
            sample_valid: false,
            recording: Vec::new(),
            replay_config: CConfig::default(),
            replay_buffer: Vec::new(),
            error: [0; 512],
        }
    }

    /// Refill this world as a private snapshot of `src` for lookahead
    /// (bopl_rl_fork): identical state, including the RNG stream and the
    /// pending-arrow/ability bookkeeping, so the same inputs produce the
    /// same future. Buffers are reused across forks. The replay recording
    /// and export buffer are not copied: a fork is never exported, and the
    /// ~170 KB recording of a long episode would dominate the copy.
    pub fn fork_from(&mut self, src: &RolloutWorld) {
        self.valid = src.valid;
        self.failed = src.failed;
        self.started = src.started;
        self.terminal = src.terminal;
        self.map = src.map;
        self.stocks = src.stocks;
        self.warmup = src.warmup;
        self.limit = src.limit;
        self.tick = src.tick;
        self.episode_ticks = src.episode_ticks;
        self.rng = src.rng;
        self.players.clone_from(&src.players);
        self.platforms.clone_from(&src.platforms);
        self.bounds.clone_from(&src.bounds);
        self.platform_epoch = src.platform_epoch;
        self.bounds_epoch = src.bounds_epoch;
        self.platform_rows_epoch = src.platform_rows_epoch;
        self.quiet_epoch = src.quiet_epoch;
        self.platform_dynamics.clone_from(&src.platform_dynamics);
        self.platform_defs.clone_from(&src.platform_defs);
        self.platform_entities.clone_from(&src.platform_entities);
        self.arrows.clone_from(&src.arrows);
        self.thrown_items.clone_from(&src.thrown_items);
        self.guided_missiles.clone_from(&src.guided_missiles);
        self.missile_explosions.clone_from(&src.missile_explosions);
        self.grenade_explosions.clone_from(&src.grenade_explosions);
        self.smoke_clouds.clone_from(&src.smoke_clouds);
        self.smoke_explosions.clone_from(&src.smoke_explosions);
        self.time_stop_effects.clone_from(&src.time_stop_effects);
        self.spike_attacks.clone_from(&src.spike_attacks);
        self.player_time_scale = src.player_time_scale;
        self.time_stop_protected = src.time_stop_protected;
        self.spawned.clone_from(&src.spawned);
        self.gust_waves = src.gust_waves;
        self.revival_marker_pool
            .clone_from(&src.revival_marker_pool);
        self.teleport_marker_pool
            .clone_from(&src.teleport_marker_pool);
        self.sudden_death = src.sudden_death;
        self.sudden_death_tick = src.sudden_death_tick;
        self.sudden_death_delays
            .clone_from(&src.sudden_death_delays);
        self.orbits.clone_from(&src.orbits);
        self.next_platform_id = src.next_platform_id;
        self.next_entity_id = src.next_entity_id;
        self.next_spawn_hierarchy = src.next_spawn_hierarchy;
        self.entities.clone_from(&src.entities);
        self.state_entities.clone_from(&src.state_entities);
        self.slimes.clone_from(&src.slimes);
        *self.state = *src.state;
        self.state.entities = self.state_entities.as_ptr();
        self.state.slimes = self.slimes.as_ptr();
        self.sample_valid = false;
        self.recording.clear();
        self.replay_config = src.replay_config;
        self.replay_buffer.clear();
        self.error = src.error;
        // Exhaustive: a new field must be added to the copy above (or listed
        // here as deliberately skipped) before this compiles again.
        let RolloutWorld {
            valid: _,
            failed: _,
            started: _,
            terminal: _,
            map: _,
            stocks: _,
            warmup: _,
            limit: _,
            tick: _,
            episode_ticks: _,
            rng: _,
            players: _,
            platforms: _,
            bounds: _,
            platform_epoch: _,
            bounds_epoch: _,
            platform_rows_epoch: _,
            quiet_epoch: _,
            scratch: _,
            platform_dynamics: _,
            platform_defs: _,
            platform_entities: _,
            arrows: _,
            thrown_items: _,
            guided_missiles: _,
            missile_explosions: _,
            grenade_explosions: _,
            smoke_clouds: _,
            smoke_explosions: _,
            time_stop_effects: _,
            spike_attacks: _,
            player_time_scale: _,
            time_stop_protected: _,
            spawned: _,
            gust_waves: _,
            revival_marker_pool: _,
            teleport_marker_pool: _,
            sudden_death: _,
            sudden_death_tick: _,
            sudden_death_delays: _,
            orbits: _,
            next_platform_id: _,
            next_entity_id: _,
            next_spawn_hierarchy: _,
            entities: _,
            slimes: _,
            state: _,
            state_entities: _,
            sample_entities: _,
            sample_slimes: _,
            sample: _,
            sample_valid: _,
            recording: _,
            replay_config: _,
            replay_buffer: _,
            error: _,
        } = src;
    }

    pub fn set_error(&mut self, message: &str) {
        let bytes = message.as_bytes();
        let n = bytes.len().min(self.error.len() - 1);
        self.error[..n].copy_from_slice(&bytes[..n]);
        self.error[n] = 0;
    }

    pub fn reset(&mut self, config: &CConfig) -> Result<(), String> {
        let data = data::get();
        let map = config.map as usize;
        if map >= data.maps.len() {
            return Err("Unknown map".into());
        }
        let players = config.players as usize;
        if !(1..=MAX_PLAYERS).contains(&players) {
            return Err("Invalid player count".into());
        }
        let slots = config.slots as usize;
        if slots > 3 {
            return Err("Invalid ability slot count".into());
        }
        if config.warmup_ticks > 3600 || config.max_episode_ticks == 0 {
            return Err("Invalid warmup or episode limit".into());
        }
        for p in 0..players {
            let c = &config.player[p];
            if c.team >= 6 || c.keyboard > 1 {
                return Err("Invalid player configuration".into());
            }
            for s in 0..slots {
                let ability = c.abilities[s];
                if ability == 0 || ability >= ABILITY_COUNT {
                    return Err("Unknown ability index".into());
                }
            }
        }

        self.map = map;
        self.stocks = config.stocks.saturating_sub(1);
        self.warmup = config.warmup_ticks;
        self.limit = config.max_episode_ticks;
        self.tick = 0;
        self.episode_ticks = 0;
        self.started = false;
        self.terminal = false;
        self.failed = false;
        self.sample_valid = false;
        self.recording.clear();
        self.replay_config = *config;
        self.rng = Random::new(config.seed.max(1));
        self.sudden_death = false;
        self.arrows.clear();
        self.thrown_items.clear();
        self.guided_missiles.clear();
        self.missile_explosions.clear();
        self.grenade_explosions.clear();
        self.smoke_clouds.clear();
        self.smoke_explosions.clear();
        self.time_stop_effects.clear();
        self.spike_attacks.clear();
        self.player_time_scale = Fix::ONE;
        self.time_stop_protected = [false; MAX_PLAYERS];
        self.spawned.clear();
        self.gust_waves = [GustWave::default(); MAX_PLAYERS * 3];
        self.revival_marker_pool.clear();
        self.teleport_marker_pool.clear();
        self.players.clear();
        self.platforms.clear();
        self.platform_dynamics.clear();
        self.platforms_changed();
        self.platform_defs.clear();
        self.platform_entities.clear();
        self.entities.clear();
        self.slimes.clear();

        let def = &data.maps[map];
        // GameSessionHandler.Init samples one randomDelay per authored ground
        // for the level-intro animation before SpawnPlayers resolves Random
        // abilities. Build the resolved loadouts here so entity registration
        // and hierarchy numbering use the concrete prefabs as well.
        for _ in &def.platforms {
            let _ = self.rng.fix_range(Fix::ZERO, Fix::ONE);
        }
        let mut resolved_abilities = [[0u32; 3]; MAX_PLAYERS];
        for (i, abilities) in resolved_abilities.iter_mut().enumerate().take(players) {
            abilities[..slots].copy_from_slice(&config.player[i].abilities[..slots]);
            let all_random = abilities[..slots]
                .iter()
                .all(|&ability| ability == ABILITY_RANDOM);
            let mut has_offensive = false;
            for (s, ability) in abilities.iter_mut().enumerate().take(slots) {
                if *ability != ABILITY_RANDOM {
                    continue;
                }
                let mut resolved = self.rng.uint_range(2, ABILITY_COUNT);
                has_offensive |= data.abilities[resolved as usize].offensive;
                if all_random && !has_offensive && s + 1 == slots {
                    resolved = self.rng.uint_range(2, ABILITY_COUNT);
                }
                *ability = resolved;
            }
        }
        // Entity table, in the .NET DetPhysics shape-group order. Within a
        // group registration follows player then equipped-slot order.
        // Dynamic entities (arrows, spawned platforms) append after.
        self.next_entity_id = 0;
        let mut player_hierarchy = [0u32; MAX_PLAYERS];
        let mut ability_hierarchy = [[0u32; 3]; MAX_PLAYERS];
        let mut hierarchy_cursor = 0u32;
        for i in 0..players {
            player_hierarchy[i] = hierarchy_cursor;
            hierarchy_cursor += 8; // shipped Slime prefab MonoUpdatables
            for (s, hierarchy) in ability_hierarchy[i].iter_mut().enumerate().take(slots) {
                *hierarchy = hierarchy_cursor;
                let ability = resolved_abilities[i][s] as usize;
                hierarchy_cursor += data.abilities[ability].updatable_count;
            }
        }
        self.next_spawn_hierarchy = hierarchy_cursor;
        let mut circle_entity = [usize::MAX; MAX_PLAYERS];
        let mut gust_entity = [usize::MAX; MAX_PLAYERS];
        let mut revival_entity = [usize::MAX; MAX_PLAYERS];
        let mut time_stop_entity = [usize::MAX; MAX_PLAYERS];
        let mut body_entity = [usize::MAX; MAX_PLAYERS];
        let mut dash_entity = [usize::MAX; MAX_PLAYERS];
        let mut bow_entity = [usize::MAX; MAX_PLAYERS];
        let mut throw_entity = [usize::MAX; MAX_PLAYERS];
        let mut mine_entity = [usize::MAX; MAX_PLAYERS];
        let mut spike_entity = [usize::MAX; MAX_PLAYERS];
        for i in 0..players {
            for (s, &hierarchy) in ability_hierarchy[i].iter().enumerate().take(slots) {
                let ability = resolved_abilities[i][s];
                let (radius, offset, relative_hierarchy, target) = match ability {
                    ABILITY_GUST => (
                        data.gust.hurtbox_radius,
                        data.gust.collider_offset,
                        8,
                        &mut gust_entity[i],
                    ),
                    ABILITY_REVIVAL => (
                        data.revival.hurtbox_radius,
                        data.revival.collider_offset,
                        8,
                        &mut revival_entity[i],
                    ),
                    ABILITY_PLATFORM => (
                        data.platform.hurtbox_radius,
                        Vec2::ZERO,
                        7,
                        &mut circle_entity[i],
                    ),
                    ABILITY_TIME_STOP => (
                        data.time_stop.hurtbox_radius,
                        data.time_stop.collider_offset,
                        data.time_stop.collider_hierarchy,
                        &mut time_stop_entity[i],
                    ),
                    _ => continue,
                };
                *target = self.entities.len();
                let id = self.next_id();
                self.entities.push(CEntity {
                    id,
                    kind: 0,
                    layer: LAYER_PLAYER,
                    owner: 0,
                    flags: E_TRIGGER,
                    hierarchy: 100_000_000 + hierarchy + relative_hierarchy,
                    x: offset.x.raw(),
                    y: offset.y.raw(),
                    radius: radius.raw(),
                    inverse_mass: -(1 << 32),
                    ..CEntity::default()
                });
            }
        }
        let water_id = self.next_id();
        // DPhysicsBox's collider axes are `Vec2.right / inverseExtents.x`
        // (Box.SetRotation, Box.cs:33) with `inverseExtents = One / extents`,
        // the same round trip as the platform perimeters.
        let water_extents = data::surface_extents(def.water_extents);
        self.entities.push(CEntity {
            id: water_id,
            kind: 1,
            layer: LAYER_WATER,
            owner: 0,
            flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
            hierarchy: 146,
            x: def.water_position.x.raw(),
            y: def.water_position.y.raw(),
            right_x: water_extents.x.raw(),
            up_y: water_extents.y.raw(),
            inverse_mass: -(1 << 32),
            ..CEntity::default()
        });
        for i in 0..players {
            for s in 0..=slots {
                let (extents, hierarchy, target, body) = if s == 0 {
                    (
                        data.player.hurtbox_extents,
                        100_000_000 + player_hierarchy[i] + 6,
                        &mut body_entity[i],
                        true,
                    )
                } else {
                    let slot_index = s - 1;
                    match resolved_abilities[i][slot_index] {
                        ABILITY_DASH => (
                            data.dash.hurtbox_extents,
                            100_000_000 + ability_hierarchy[i][slot_index] + 7,
                            &mut dash_entity[i],
                            false,
                        ),
                        ABILITY_BOW => (
                            data.bow.hurtbox_extents,
                            100_000_000 + ability_hierarchy[i][slot_index] + 7,
                            &mut bow_entity[i],
                            false,
                        ),
                        ABILITY_MISSILE => (
                            data.missile.hurtbox_extents,
                            100_000_000 + ability_hierarchy[i][slot_index] + 7,
                            &mut throw_entity[i],
                            false,
                        ),
                        ABILITY_MINE => (
                            data::surface_extents(data.mine.hurtbox_extents),
                            100_000_000 + ability_hierarchy[i][slot_index] + 7,
                            &mut mine_entity[i],
                            false,
                        ),
                        ability @ (ABILITY_GRENADE | ABILITY_SMOKE | ABILITY_TACK
                        | ABILITY_MJOLNIR) => (
                            throw_config(data, ability).hurtbox_extents,
                            100_000_000 + ability_hierarchy[i][slot_index] + 8,
                            &mut throw_entity[i],
                            false,
                        ),
                        ABILITY_SPIKE => (
                            data.spike.hurtbox_extents,
                            100_000_000
                                + ability_hierarchy[i][slot_index]
                                + data.spike.collider_hierarchy,
                            &mut spike_entity[i],
                            false,
                        ),
                        _ => continue,
                    }
                };
                *target = self.entities.len();
                let id = self.next_id();
                self.entities.push(CEntity {
                    id,
                    kind: 1,
                    layer: LAYER_PLAYER,
                    owner: if body { i as u32 + 1 } else { 0 },
                    flags: if body {
                        E_ACTIVE | E_ENABLED | E_TRIGGER
                    } else if resolved_abilities[i][s - 1] == ABILITY_SPIKE {
                        E_ENABLED | E_TRIGGER
                    } else {
                        E_TRIGGER
                    },
                    hierarchy,
                    x: if !body && resolved_abilities[i][s - 1] == ABILITY_SPIKE {
                        data.spike.collider_offset.x.raw()
                    } else if !body && resolved_abilities[i][s - 1] == ABILITY_MINE {
                        data.mine.collider_offset.x.raw()
                    } else {
                        0
                    },
                    y: if !body && resolved_abilities[i][s - 1] == ABILITY_SPIKE {
                        data.spike.collider_offset.y.raw()
                    } else if !body && resolved_abilities[i][s - 1] == ABILITY_MINE {
                        data.mine.collider_offset.y.raw()
                    } else {
                        0
                    },
                    right_x: extents.x.raw(),
                    up_y: extents.y.raw(),
                    inverse_mass: -(1 << 32),
                    ..CEntity::default()
                });
            }
        }
        self.next_platform_id = 100;
        self.sudden_death_delays.clear();
        self.orbits.clear();
        for (index, p) in def.platforms.iter().enumerate() {
            let id = self.next_id();
            let entity = self.entities.len();
            self.entities.push(CEntity {
                id,
                kind: 2,
                layer: LAYER_WALL,
                owner: 0,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: p.hierarchy,
                inverse_mass: p.body.inverse_mass.raw(),
                ..CEntity::default()
            });
            self.platforms.push(make_platform(
                self.next_platform_id,
                &p.surface,
                p.collider_extents,
            ));
            self.platform_dynamics.push(PlatformDynamics {
                body: p.body,
                root_body: p.body,
                attached_spikes: 0,
                composite_position: p.surface.center,
                home_position: p.surface.center,
                transform_rotation: None,
                init_tick: None,
                home_rotation: p.body.orientation,
                speed: p.speed,
                mu: p.mu,
                rotation_speed: p.rotation_speed,
                rotation_mu: p.rotation_mu,
                basis: None,
            });
            self.platforms_changed();
            self.platform_defs.push(index);
            self.platform_entities.push(entity);
            self.next_platform_id += 1;
            // AnimateVelocity samples this only when sudden death activates.
            self.sudden_death_delays.push(Fix::ZERO);
            if p.orbit.is_some() {
                self.orbits.push((self.platforms.len() - 1, 0));
            }
        }

        // GameSessionHandler maps sorted distinct team ids onto authored
        // spawn slots, then fans teammates (and wrapped extra teams) out in
        // alternating down-left/down-right offsets.
        let mut distinct_teams = [u32::MAX; MAX_PLAYERS];
        let mut distinct_count = 0usize;
        for player in &config.player[..players] {
            let team = player.team;
            if !distinct_teams[..distinct_count].contains(&team) {
                distinct_teams[distinct_count] = team;
                distinct_count += 1;
            }
        }
        distinct_teams[..distinct_count].sort_unstable();
        let mut spawn_counts = [0u32; MAX_PLAYERS];

        // Players at their resolved team spawns.
        for i in 0..players {
            let team = config.player[i].team;
            let team_rank = distinct_teams[..distinct_count]
                .iter()
                .position(|&candidate| candidate == team)
                .expect("configured team was collected");
            let spawn_index = team_rank % def.team_spawns.len();
            let teammate_index = spawn_counts[spawn_index];
            spawn_counts[spawn_index] += 1;
            let mut spawn = def.team_spawns[spawn_index];
            if teammate_index > 0 {
                let step = teammate_index.div_ceil(2);
                let distance = def.teammate_spacing * Fix::from_int(i64::from(step));
                spawn += if teammate_index % 2 == 1 {
                    Vec2::new(-distance, -distance * Fix::from_f32(0.5))
                } else {
                    Vec2::new(distance, -distance * Fix::from_f32(0.5))
                };
            }
            let abilities = resolved_abilities[i];
            let mut cooldowns = [Fix::ZERO; 3];
            for s in 0..slots {
                cooldowns[s] = data.abilities[abilities[s] as usize].cooldown;
            }
            // -1.5 s on the wake tick, one DT per tick since (every phase
            // below, including the pre-instantiation ticks, advances it).
            let since_awake = i64::from(self.warmup) - i64::from(SLIME_AWAKE_TICK);
            let timer0 = TIMER_START + DT * Fix::from_int(since_awake);
            self.players.push(PlayerSlot {
                base: Player {
                    alive: true,
                    id: i as u32 + 1,
                    position: spawn,
                    rotation: Fix::ZERO,
                    up: Vec2::UP,
                    velocity: Vec2::ZERO,
                    external_velocity: Vec2::ZERO,
                    radius: data.player.radius,
                    grounded: false,
                    ground: None,
                    local: Fix::ZERO,
                    grounded_speed: Fix::ZERO,
                    previous_jump: false,
                    skip_next_gravity: false,
                    jumped_this_frame: false,
                    constants: data.player.constants.clone(),
                    cooldown: Fix::ZERO,
                    dash: DashState::default(),
                    collider: OrientedBox {
                        center: spawn,
                        right: Vec2::RIGHT,
                        up: Vec2::UP,
                        inverse_extents: Vec2::new(Fix::ONE, Fix::ONE),
                    },
                    collider_rotation: Fix::ZERO,
                    collider_basis: Vec2::ZERO,
                    aligned: None,
                    input_normalized: None,
                },
                team,
                keyboard: config.player[i].keyboard != 0,
                stocks: self.stocks,
                kills: 0,
                deaths: 0,
                cause_of_death: CAUSE_NOT_DEAD_YET,
                won: false,
                out: false,
                abilities,
                timers: [timer0; 3],
                cooldowns,
                active: ActiveAbility::None,
                active_slot: 0,
                ability_origin: spawn,
                ability_origin_velocity: Vec2::ZERO,
                ability_origin_rotation: Fix::ZERO,
                ability_origin_state: STATE_FALL,
                ability_entry_action: Action::default(),
                invisible: false,
                invisible_time: Fix::ZERO,
                was_invisible: false,
                ability_time: Fix::ZERO,
                reported_position: spawn,
                reported_tick: u32::MAX,
                removed_ground_normal: None,
                bow: BowState::default(),
                missile: MissileFormState::default(),
                throw_item: ThrowItemState::default(),
                cast_spell: CastSpellState::default(),
                spike_cast: SpikeCastState::default(),
                spike_attacks: [None; 3],
                platform: PlatformState::default(),
                gust: GustState::default(),
                revival: RevivalState::default(),
                revival_markers: [None; 3],
                revival_clone_pending: false,
                stock_clone_pending: false,
                revival_target_marker: None,
                revival_effect_ticks: 0,
                revival_particle_ticks: 0,
                teleport_markers: [None; 3],
                teleport_pending_marker: None,
                teleport_pending_position: Vec2::ZERO,
                teleport_effect_active: false,
                teleport_effect_time: Fix::ZERO,
                gold_mushroom: [GoldMushroomState::default(); 3],
                rewind: RewindState::default(),
                obs_state: STATE_FALL,
                state_ticks: 0,
                facing_right: true,
                unspawned: 0,
                respawn_ticks: 0,
                respawn_spawn_ticks: 0,
                respawn_selected: 1,
                stock_indicator_entities: [usize::MAX; 3],
                input_jump_was_down: false,
                last_movement_input: Vec2::ZERO,
                telemetry: CTelemetry::default(),
                circle_entity: circle_entity[i],
                gust_entity: gust_entity[i],
                revival_entity: revival_entity[i],
                body_entity: body_entity[i],
                dash_entity: dash_entity[i],
                bow_entity: bow_entity[i],
                throw_entity: throw_entity[i],
                mine_entity: mine_entity[i],
                spike_entity: spike_entity[i],
                time_stop_entity: time_stop_entity[i],
            });
        }

        // Warmup, as GameSessionHandler runs it. The level animates in
        // (StartAnimateInLevelRoutine; here platforms start at their homes,
        // where the .NET sim's have settled by tick ~155). SpawnPlayers runs
        // after `timeBeforeTwo` (1 s): the slimes are instantiated at the
        // spawn points on tick SLIME_AWAKE_TICK+1 and play the 105-tick
        // "spawn" animation with physics disabled (BeforeSpawnInit). Its end
        // callback SlimeController.Spawn (SlimeController.cs:567) enables the
        // body and launches it at `jumpStrength * 0.25` (11.25 u/s) straight
        // up; the slime then falls onto its platform with no input, and its
        // landing slide (PlayerPhysics.AttachToGround keeps the impact
        // velocity's tangential part as groundedSpeed, PlayerPhysics.cs:501)
        // depends on that impact velocity. A free fall from rest at tick 0
        // (the previous approximation) landed 0.95 u/s slower on map 1 and
        // came to rest 2.5e-3 u away from the game's spawn position.
        let no_input = vec![Action::default(); players];
        let mut spawn_anim = 0u32;
        for i in 0..self.warmup {
            if i < SLIME_AWAKE_TICK {
                continue;
            }
            if i == SLIME_AWAKE_TICK {
                spawn_anim = data.player.spawn_anim_ticks;
                continue;
            }
            if spawn_anim > 0 {
                spawn_anim -= 1;
                if spawn_anim == 0 {
                    for slot in &mut self.players {
                        slot.base.velocity =
                            Vec2::UP * data.player.constants.jump_strength * Fix::from_f32(0.25);
                    }
                }
                self.record_warmup_rewind(i);
                continue;
            }
            self.movement_tick(&no_input, i);
        }
        // A warmup that ends inside the spawn animation hands the rest of it
        // to the episode (respawn_tick counts it down and launches the same
        // way); one that ends before SpawnPlayers ran leaves the slimes
        // uninstantiated (no slime or hurtbox rows, Player.Position zero)
        // until that tick.
        if spawn_anim > 0 {
            for slot in &mut self.players {
                slot.respawn_spawn_ticks = spawn_anim;
            }
        }
        if self.warmup <= SLIME_AWAKE_TICK {
            for slot in &mut self.players {
                slot.unspawned = SLIME_AWAKE_TICK + 1 - self.warmup;
                slot.base.position = Vec2::ZERO;
            }
        }
        // Replays drive every simulated tick from boot, warmup included.
        for _ in 0..self.warmup {
            self.recording.push([CAction::default(); MAX_PLAYERS]);
        }
        self.tick = self.warmup;
        self.started = true;
        self.observe_main();
        self.valid = true;
        Ok(())
    }

    fn next_id(&mut self) -> u32 {
        self.next_entity_id += 1;
        self.next_entity_id
    }

    fn record_warmup_rewind(&mut self, warmup_tick: u32) {
        let timer = TIMER_START
            + DT * Fix::from_int(i64::from(warmup_tick.saturating_sub(SLIME_AWAKE_TICK)));
        for slot in &mut self.players {
            slot.rewind.push(RewindSample {
                position: slot.base.position,
                velocity: slot.base.velocity,
                external_velocity: slot.base.external_velocity,
                rotation: slot.base.rotation,
                cooldowns: [timer; 3],
            });
        }
    }

    /// One movement-only tick (used for warmup settling).
    fn movement_tick(&mut self, actions: &[Action], warmup_tick: u32) {
        self.rebuild_bounds();
        for (index, action) in actions.iter().enumerate() {
            if !self.players[index].base.alive {
                continue;
            }
            self.players[index].base.step(*action, terrain!(self));
        }
        // RewindRecorder.UpdateSim is after PlayerBody.UpdateSim but before
        // PlayerPhysics.LateUpdateSim snaps grounded riders to their surface.
        self.record_warmup_rewind(warmup_tick);
        for slot in &mut self.players {
            if slot.base.alive {
                slot.base.sync_collider();
                slot.base.late_step(&self.platforms);
            }
        }
        for (index, action) in actions.iter().enumerate() {
            self.update_obs_state(index, action.direction());
        }
    }

    /// SlimeController's observable state machine, evaluated once per tick
    /// after physics. Frozen while an ability (dash included) runs, because
    /// EnterAbility returns before the state switch and nothing ever assigns
    /// State.ability. `land`/`stop`/`turn` last exactly their animation and
    /// then chain like the .NET callbacks (land->idle, stop->idle,
    /// turn->run); reversals while running pass through `stop`.
    fn update_obs_state(&mut self, index: usize, dir: Vec2) {
        let slot = &mut self.players[index];
        if !slot.base.alive
            || slot.respawn_ticks > 0
            || slot.revival_clone_pending
            || slot.revival_effect_ticks > 0
            || slot.active != ActiveAbility::None
            || slot.base.dash.active
        {
            return;
        }
        // PlayerPhysics.Jump ungrounds the body and OnNoLongerGrounded sets
        // State.fall (SlimeController.cs:956-958) before Move's air raycasts
        // can re-attach it on the same pass (PlayerPhysics.cs:236-238), which
        // then lands it again through attachToGround.
        if std::mem::take(&mut slot.base.jumped_this_frame) || !slot.base.grounded {
            slot.obs_state = STATE_FALL;
            slot.state_ticks = 0;
        }
        if !slot.base.grounded {
            return;
        }
        // SlimeController flag5: stick magnitude past DEADZONE 0.4 (the
        // direction table only emits unit vectors) and the tangential ground
        // alignment past GROUNDALIGNMENTDEADZONE 0.2, or input pressed into
        // the ground with any alignment at all.
        let tangent = Vec2::new(slot.base.up.y, -slot.base.up.x); // rightward
        let align = dir.dot(tangent);
        let into_ground = slot.base.up.dot(dir) < Fix::ZERO;
        let moving = dir != Vec2::ZERO
            && (align.abs() >= Fix::from_f32(0.2)
                || (into_ground && align.abs() >= Fix::from_f32(0.01)));
        let input_right = align >= Fix::ZERO;
        match slot.obs_state {
            STATE_FALL => {
                // attachToGround: land animation, then idle (one tick longer
                // after an ability exit, see `PlayerSlot::exit_to_fall`).
                slot.obs_state = STATE_LAND;
                slot.state_ticks += LAND_TICKS;
            }
            STATE_LAND => {
                slot.state_ticks = slot.state_ticks.saturating_sub(1);
                if slot.state_ticks == 0 {
                    slot.obs_state = STATE_IDLE;
                }
            }
            STATE_RUN => {
                if !moving || input_right != slot.facing_right {
                    slot.obs_state = STATE_STOP;
                    slot.state_ticks = STOP_TICKS;
                }
            }
            STATE_STOP => {
                if moving && input_right != slot.facing_right {
                    slot.obs_state = STATE_TURN;
                    slot.state_ticks = TURN_TICKS;
                    slot.facing_right = input_right;
                } else {
                    slot.state_ticks = slot.state_ticks.saturating_sub(1);
                    if slot.state_ticks == 0 {
                        slot.obs_state = STATE_IDLE;
                    }
                }
            }
            STATE_TURN => {
                if !moving {
                    slot.obs_state = STATE_STOP;
                    slot.state_ticks = STOP_TICKS;
                } else {
                    slot.state_ticks = slot.state_ticks.saturating_sub(1);
                    if slot.state_ticks == 0 {
                        slot.obs_state = STATE_RUN;
                    }
                }
            }
            _ => {
                // idle (and anything unexpected).
                if moving {
                    slot.obs_state = STATE_RUN;
                    slot.facing_right = input_right;
                } else {
                    slot.obs_state = STATE_IDLE;
                }
            }
        }
    }

    pub fn step(&mut self, actions: &[CAction], repeat: u32) -> Result<(), String> {
        if !self.valid {
            return Err("No episode in progress".into());
        }
        if self.terminal {
            return Err("Episode has ended".into());
        }
        if actions.len() != self.players.len() {
            return Err("One action per player is required".into());
        }
        for action in actions {
            if action.reserved != 0 || (action.buttons & !1023) != 0 {
                return Err("Malformed action".into());
            }
        }
        self.sample_valid = false;
        for _ in 0..repeat.max(1) {
            let mut frame = [CAction::default(); MAX_PLAYERS];
            frame[..actions.len()].copy_from_slice(actions);
            self.recording.push(frame);
            // RlBridge registers this frame before Updater.TickSimulation.
            // Live SlimeController/PlayerBody state therefore consumes the
            // current input immediately. Only Player.Position is stale; that
            // publication phase is represented explicitly in PlayerSlot.
            self.game_tick(actions);
            if self.terminal {
                break;
            }
        }
        self.observe_main();
        Ok(())
    }

    pub fn step_sampled(
        &mut self,
        actions: &[CAction],
        first: u32,
        second: u32,
    ) -> Result<(), String> {
        if first + second > 16 {
            return Err("Sampled step phases too long".into());
        }
        self.step(actions, first.max(1))?;
        self.observe_sample();
        self.sample_valid = true;
        if !self.terminal {
            for _ in 0..second {
                let mut frame = [CAction::default(); MAX_PLAYERS];
                frame[..actions.len()].copy_from_slice(actions);
                self.recording.push(frame);
                self.game_tick(actions);
                if self.terminal {
                    break;
                }
            }
            self.observe_main();
        }
        Ok(())
    }

    /// One full game tick: abilities, movement, projectiles, platform motion,
    /// deaths, session bookkeeping. Mirrors Updater.TickSimulation order
    /// loosely: inputs -> slime updates (abilities + movement) -> physics ->
    /// session end checks.
    fn game_tick(&mut self, actions: &[CAction]) {
        let data = data::get();
        self.register_pending_arrows();
        self.register_pending_mines(data);
        self.register_pending_guided_missiles(data);
        self.register_pending_missile_explosions(data);
        self.register_pending_revival_markers(data);
        self.register_pending_teleport_markers(data);
        self.register_pending_revival_clones(data);
        self.register_pending_stock_clones(data);
        self.register_pending_grenade_explosions(data);
        self.register_pending_smoke(data);
        self.register_pending_time_stops();
        // Updatables earlier than a TimeStop effect observe the scale from
        // the start of the pass. DetPhysics observes any scale written by the
        // effect later in this same pass.
        let updater_dt = DT * self.player_time_scale;
        // Updater initializes every prefab queued during the prior pass
        // before it invokes any existing MonoUpdatable. A newly spawned Gust
        // wave therefore applies its first time-zero impulse before map home
        // springs, while its same-tick UpdateSim impulse runs after them.
        self.gust_waves_init(data);
        // Platforms moved in the previous tick's physics stage.
        self.rebuild_bounds();
        // GameSessionHandler (hierarchy 100) runs before every instantiated
        // player/ability prefab (100,000,000+). A death produced by a player
        // component or DetPhysics is therefore visible to the match result
        // check on the following tick, not later in the tick that killed it.
        self.session_result_tick();
        // Authored level updaters have small serialized hierarchy numbers
        // (roughly 100), while player prefabs and anything they spawn start at
        // 100,000,000. Compute their home springs before a later ability such
        // as Gust changes a platform body's velocity. DetPhysics integrates
        // both the queued spring force and those later direct impulses below.
        self.platform_updaters_tick(data, updater_dt);
        // Updater runs Player.UpdatePlayer for the full player list before
        // any MonoUpdatable. Invisibility therefore expires before the owning
        // SlimeController gets its cooldown/cast pass.
        for (index, slot) in self.players.iter_mut().enumerate() {
            if slot.invisible {
                // Time-stop protection is added with the Time Stop port. In
                // the ordinary clock state FixedDeltaTime is exactly DT.
                let dt = if self.player_time_scale < Fix::from_f32(0.01)
                    && !self.time_stop_protected[index]
                {
                    Fix::ZERO
                } else {
                    DT
                };
                slot.invisible_time += dt;
                slot.ability_time += dt;
                if slot.invisible_time > Fix::from_int(5) {
                    slot.invisible = false;
                    slot.invisible_time = Fix::ZERO;
                    slot.ability_time = Fix::ZERO;
                }
            } else {
                slot.invisible_time = Fix::ZERO;
                slot.ability_time = Fix::ZERO;
            }
        }
        let mut jump_pressed_by_player = [false; MAX_PLAYERS];
        for (index, packet) in actions.iter().enumerate().take(self.players.len()) {
            if self.player_time_scale < Fix::from_f32(0.01) && !self.time_stop_protected[index] {
                // PlayerPhysics and SlimeController still run. Their motion
                // helpers apply unscaled gravity and air damping, but every
                // position/raycast delta receives GameTime.FixedDeltaTime=0.
                let input = self.players[index].last_movement_input;
                let base = &mut self.players[index].base;
                if !base.grounded {
                    crate::player_physics::apply_gravity(
                        &mut base.velocity,
                        base.constants.gravity_accel,
                        base.constants.gravity_modifier,
                        base.constants.gravity_max_fall_speed,
                        false,
                    )
                    .expect("valid movement gravity constants");
                    crate::player_physics::air_movement(
                        &mut base.velocity,
                        input.x,
                        base.constants.air_accel,
                        base.constants.max_speed,
                    )
                    .expect("valid movement air constants");
                } else if input == Vec2::ZERO {
                    base.grounded_speed *= base.constants.friction;
                }
                base.external_velocity *= Fix::from_f64(0.95);
                self.players[index].reported_position = self.players[index].base.position;
                self.players[index].reported_tick = self.tick;
                continue;
            }
            // Player.UpdatePlayer is on the disabled slime GameObject during
            // TeleportedObjectEffect, so Player.Position remains at its
            // pre-teleport publication until reactivation.
            if !self.players[index].teleport_effect_active {
                self.players[index].reported_position = self.players[index].base.position;
                self.players[index].reported_tick = self.tick;
            } else {
                self.players[index].reported_tick = self.tick;
            }
            let action = packet.aim_action();
            // Updater updates Player's frame-dependent button edges before
            // any MonoUpdatable, including while its slime is parked for a
            // stock choice.
            let movement_input = packet.input_direction(self.players[index].keyboard);
            self.players[index].last_movement_input = movement_input;
            let jump_down = action.buttons & 1 != 0;
            let jump_pressed = jump_down && !self.players[index].input_jump_was_down;
            jump_pressed_by_player[index] = jump_pressed;
            // SlimeController.Jump reads that same edge
            // (jumpButton_PressedThisFrame, SlimeController.cs:796); the
            // movement slice's copy must not go stale while an ability runs.
            self.players[index].base.previous_jump = self.players[index].input_jump_was_down;
            self.players[index].input_jump_was_down = jump_down;
            self.player_tick(index, action, movement_input, jump_pressed, data);
            // The shared movement slice marks a body dead when its previous
            // position is beyond a blast boundary. Convert that marker into
            // the match-level death exactly once; otherwise the later
            // `!alive` guard would skip stocks, telemetry, and `out` forever.
            let boundary_death = {
                let slot = &self.players[index];
                !slot.out
                    && slot.respawn_ticks == 0
                    && slot.respawn_spawn_ticks == 0
                    && !slot.base.alive
                    && slot.base.outside_bounds()
            };
            if boundary_death {
                // The ordinary slime's boundary component kills with
                // Accident (DestroyIfOutsideSceneBounds.cs:148); an ability
                // body's exits through Ability.ExitAbility with a default
                // AbilityExitInfo, whose causeOfDeath is Other
                // (DestroyIfOutsideSceneBounds.cs:135-141, SlimeController.cs:219).
                let cause = if self.players[index].active == ActiveAbility::None {
                    CAUSE_ACCIDENT
                } else {
                    CAUSE_OTHER
                };
                self.kill(index, ENVIRONMENT, cause, false);
            }
        }
        self.rewind_recorders_tick();
        // SuperStarEffect prefabs spawned during a slime update join the
        // updater list on the following tick. Once live they run after player
        // forms and can overwrite the tick's vertical self-imposed velocity.
        for (index, &jump_pressed) in jump_pressed_by_player
            .iter()
            .enumerate()
            .take(self.players.len())
        {
            self.gold_mushroom_tick(index, jump_pressed, data);
        }
        self.gust_waves_update(data);
        self.revival_effects_tick(data);
        self.teleport_effects_tick(data);
        self.teleport_pending_tick(data);
        self.grenade_explosions_tick(data, updater_dt);
        self.missile_explosions_tick(data, updater_dt);
        self.smoke_tick(data, updater_dt);
        self.time_stop_effects_tick(data);
        self.spike_attacks_tick(data, updater_dt);
        // Slime-vs-slime shove (World::step parity).
        for a in 0..self.players.len() {
            for b in a + 1..self.players.len() {
                let (pa, pb) = (a, b);
                if !self.playable(pa) || !self.playable(pb) {
                    continue;
                }
                // Non-instant abilities disable the original slime box, but
                // Bow and Platform anticipation replace it with their own
                // Player-layer trigger. The active ability has no
                // SlimeController callback; the ordinary slime on the other
                // side still receives its one-sided overlap push.
                let collider_enabled = |slot: &PlayerSlot| match slot.active {
                    ActiveAbility::None
                    | ActiveAbility::Dash
                    | ActiveAbility::Bow
                    | ActiveAbility::Grenade
                    | ActiveAbility::Smoke
                    | ActiveAbility::Tack
                    | ActiveAbility::Mjolnir
                    | ActiveAbility::Missile
                    | ActiveAbility::Mine
                    | ActiveAbility::Spike
                    | ActiveAbility::TimeStop
                    | ActiveAbility::Gust
                    | ActiveAbility::Revival => true,
                    ActiveAbility::Platform => slot.platform.phase == PLATFORM_ANTICIPATION,
                };
                if !collider_enabled(&self.players[pa]) || !collider_enabled(&self.players[pb]) {
                    continue;
                }
                let a_pos = self.players[pa].base.position;
                let b_pos = self.players[pb].base.position;
                if self.players[pa].active == ActiveAbility::None {
                    self.players[pa].base.push_from(b_pos, true);
                }
                if self.players[pb].active == ActiveAbility::None {
                    self.players[pb].base.push_from(a_pos, false);
                }
            }
        }
        // DetPhysics.Simulate syncs every collider from the body position the
        // updater pass left (Updater.cs:303-323); a grounded slime's body only
        // moves to its new surface local in PlayerPhysics.LateUpdateSim
        // (PlayerPhysics.cs:138-146), after the contacts below.
        for (index, action) in actions.iter().enumerate() {
            if self.playable(index) {
                self.players[index].base.sync_collider();
            }
            let input = action.input_direction(self.players[index].keyboard);
            self.update_obs_state(index, input);
        }
        let physics_dt = DT * self.player_time_scale;
        self.arrows_integrate(data, updater_dt, physics_dt);
        self.guided_missiles_integrate(data, physics_dt);
        self.thrown_items_integrate(data, updater_dt, physics_dt);
        // DetPhysics.Simulate drops last pass's callback casualties, then
        // syncs every registered collider; the sweep axis below reads them.
        for slot in &self.players {
            if slot.out {
                for e in [
                    slot.body_entity,
                    slot.dash_entity,
                    slot.circle_entity,
                    slot.bow_entity,
                    slot.throw_entity,
                    slot.mine_entity,
                    slot.gust_entity,
                    slot.revival_entity,
                    slot.spike_entity,
                    slot.time_stop_entity,
                ] {
                    if e != usize::MAX {
                        self.entities[e].flags = E_REMOVED;
                    }
                }
            }
        }
        self.refresh_actor_entities();
        self.platforms_tick(data, updater_dt, physics_dt);
        self.arrows_contacts(data);
        self.guided_missile_contacts(data);
        self.thrown_items_contacts(data);
        self.grenade_explosion_contacts(data);
        self.missile_explosion_contacts(data);
        self.smoke_contacts(data);
        self.spike_contacts(data);
        // PlayerPhysics.LateUpdateSim (PlayerPhysics.cs:138-146) positions
        // StickyRoundedRectangle riders from their surface local after
        // physics; their Player-layer box keeps the position DetPhysics
        // synced above until the next Simulate.
        for slot in &mut self.players {
            if slot.base.alive
                && slot.base.grounded
                && slot.removed_ground_normal.is_none()
                && !slot.rewind.rewinding
            {
                slot.base.late_step(&self.platforms);
            }
        }
        self.mines_late_tick(data);
        self.tick += 1;
        self.episode_ticks += 1;
        self.session_tick();
    }

    fn playable(&self, index: usize) -> bool {
        let slot = &self.players[index];
        slot.base.alive
            && slot.unspawned == 0
            && slot.respawn_ticks == 0
            && slot.respawn_spawn_ticks == 0
            && !slot.revival_clone_pending
            && !slot.stock_clone_pending
            && slot.revival_effect_ticks == 0
            && !slot.teleport_effect_active
    }

    /// SlimeController.OldUpdate: cooldowns, ability entry, then movement.
    fn player_tick(
        &mut self,
        index: usize,
        action: Action,
        movement_input: Vec2,
        jump_pressed: bool,
        data: &data::RolloutData,
    ) {
        let in_ability = {
            let slot = &self.players[index];
            slot.active != ActiveAbility::None || slot.base.dash.active
        };
        if !self.playable(index) {
            if self.players[index].teleport_effect_active {
                return;
            }
            self.respawn_tick(index, movement_input, jump_pressed, data);
            return;
        }
        if self.players[index].rewind.rewinding {
            // RewindRecorder disables PlayerPhysics and suppresses the slime
            // controller's input. Cooldowns still advance, SlimeController
            // calls Move(zero), and PlayerBody remains live ahead of the
            // recorder.
            let slot = &mut self.players[index];
            for timer in &mut slot.timers {
                *timer += DT;
            }
            slot.base.move_only(Vec2::ZERO, terrain!(self));
            slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
            slot.base.external_velocity *= Fix::from_f64(0.95);
            return;
        }
        if let Some(normal) = self.players[index].removed_ground_normal.take() {
            let slot = &mut self.players[index];
            // Ordinary SlimeController explicitly zeros rotation before
            // UnGround(false). PlayerPhysics already skipped gravity earlier
            // in this pass while `isGrounded` was still true. BowTransform,
            // by contrast, owns this check and applies gravity after it.
            if slot.active == ActiveAbility::None {
                slot.base.rotation = Fix::ZERO;
                slot.base.up = Vec2::UP;
                slot.base.skip_next_gravity = true;
            }
            slot.base.position += normal * Fix::from_f32(0.05);
            slot.base.grounded = false;
            slot.base.ground = None;
            slot.base.grounded_speed = Fix::ZERO;
        }
        if !in_ability {
            let slot = &mut self.players[index];
            // PlayerPhysics.UpdateSim (PlayerPhysics.cs:116-131) applies
            // gravity and the landing raycasts before SlimeController runs.
            // SlimeController.OldUpdate then publishes body.position
            // (SlimeController.cs:712): on a landing tick that is the
            // AttachToGround position, before Move slides along the surface.
            if !slot.base.grounded {
                slot.base.gravity_step(terrain!(self));
            }
            slot.reported_position = slot.base.position;
            for s in 0..3 {
                // A protected player otherwise receives an unscaled delta,
                // but abilities opting into CooldownStoppedDuringTimeStop
                // still keep their timer fixed (SlimeController.cs:727-731).
                if self.player_time_scale >= Fix::from_f32(0.01)
                    || !data.abilities[slot.abilities[s] as usize].stopped_during_time_stop
                {
                    slot.timers[s] += DT;
                }
                if slot.abilities[s] == ABILITY_REWIND
                    && !slot.rewind.rewinding
                    && slot.rewind.finish_next_tick
                    && action.buttons & (2 << s) as u8 == 0
                {
                    slot.timers[s] = slot.cooldowns[s];
                    slot.rewind.finish_next_tick = false;
                }
                if slot.abilities[s] == ABILITY_TELEPORT
                    && slot.teleport_markers[s]
                        .is_some_and(|marker| self.teleport_marker_pool[marker].active)
                    && action.buttons & (2 << s) as u8 == 0
                {
                    // SlimeController resets an IgnoreNextCooldown slot on
                    // every released frame. Holding the second press skips
                    // this reset, allowing the just-incremented timer to pass
                    // the strict readiness comparison.
                    slot.timers[s] = slot.cooldowns[s];
                }
            }
        }
        // Ability entry (held button, ready, castable, not already in one).
        if !in_ability {
            for s in 0..3 {
                let slot = &self.players[index];
                let button_down = action.buttons & (2 << s) as u8 != 0;
                // SlimeController.isAbilityReady uses a strict comparison.
                if !button_down || slot.timers[s] <= slot.cooldowns[s] {
                    continue;
                }
                let ability = slot.abilities[s];
                let ability_data = &data.abilities[ability as usize];
                if (ability_data.ground_only && !slot.base.grounded)
                    || !ability_data.castable
                    || (ability == ABILITY_INVISIBILITY && slot.invisible)
                {
                    continue;
                }
                {
                    let slot = &mut self.players[index];
                    // A landing in this pass's PlayerPhysics.UpdateSim already
                    // set State.land through attachToGround
                    // (SlimeController.cs:1126-1128) before EnterAbility
                    // freezes the state machine.
                    if slot.base.grounded && slot.obs_state == STATE_FALL {
                        slot.obs_state = STATE_LAND;
                        slot.state_ticks += LAND_TICKS;
                    }
                    // The PlayerInfo package carries the post-gravity state;
                    // PlayerBody has not integrated the position yet. The
                    // deactivated slime still gets this pass's Move
                    // (SlimeController.cs:1084), which is what its record
                    // shows for the rest of the ability.
                    let mut frozen = slot.base.clone();
                    frozen.move_only(movement_input, terrain!(self));
                    slot.ability_origin = frozen.position;
                    slot.ability_origin_velocity = frozen.velocity + frozen.external_velocity;
                    slot.ability_origin_rotation = frozen.rotation;
                    slot.ability_origin_state = slot.obs_state;
                }
                match ability {
                    ABILITY_INVISIBILITY => {
                        let slot = &mut self.players[index];
                        slot.invisible = true;
                        slot.invisible_time = Fix::ZERO;
                        slot.ability_time = Fix::ZERO;
                        // Invisibility.CastInvisibility changes the live
                        // InstantAbility cooldown until SlimeController sees a
                        // visible frame and calls OnInvisibiltyEnded.
                        slot.cooldowns[s] = Fix::from_int(10_000);
                        slot.timers[s] = Fix::ZERO;
                        slot.telemetry.ability_activations[s] += 1;
                        break;
                    }
                    ABILITY_GOLD_MUSHROOM => {
                        let slot = &mut self.players[index];
                        slot.gold_mushroom[s] = GoldMushroomState {
                            active: true,
                            pending_init: true,
                            time: Fix::ZERO,
                        };
                        slot.cooldowns[s] = Fix::from_int(10_000);
                        slot.timers[s] = Fix::ZERO;
                        slot.telemetry.ability_activations[s] += 1;
                        break;
                    }
                    ABILITY_TELEPORT => {
                        let marker = self.players[index].teleport_markers[s]
                            .filter(|&marker| self.teleport_marker_pool[marker].active);
                        {
                            let slot = &mut self.players[index];
                            slot.timers[s] = Fix::ZERO;
                            slot.telemetry.ability_activations[s] += 1;
                        }
                        if let Some(marker) = marker {
                            let slot = &mut self.players[index];
                            slot.teleport_pending_marker = Some(marker);
                            slot.teleport_pending_position = slot.base.position;
                            slot.teleport_markers[s] = None;
                        } else {
                            let hierarchy_base = self.next_spawn_hierarchy;
                            self.next_spawn_hierarchy += data.teleport.marker_updatable_count;
                            let marker_index = self.teleport_marker_pool.len();
                            self.teleport_marker_pool.push(TeleportMarker {
                                active: true,
                                registered: false,
                                born_tick: self.tick,
                                owner: index,
                                ability_slot: s,
                                entity: None,
                                hierarchy_base,
                                position: self.players[index].base.position
                                    + Vec2::UP * data.teleport.up_cast_offset,
                                velocity: Vec2::ZERO,
                            });
                            self.players[index].teleport_markers[s] = Some(marker_index);
                        }
                        break;
                    }
                    ABILITY_REWIND => {
                        let attached_ground = self.players[index].base.ground;
                        let started = {
                            let slot = &mut self.players[index];
                            slot.rewind.begin(slot.base.position, slot.base.rotation, s)
                        };
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.telemetry.ability_activations[s] += 1;
                        if started {
                            slot.rewind.attached_ground = attached_ground;
                            slot.base.force_unground(&self.platforms);
                            slot.base.velocity = Vec2::ZERO;
                            slot.base.external_velocity = Vec2::ZERO;
                            // EnterAbility returns from OldUpdate, but this
                            // already-running SlimeController pass still uses
                            // the input vector captured before Rewind set
                            // suppressInput.
                            slot.base.move_only(movement_input, terrain!(self));
                            slot.base.position +=
                                (slot.base.velocity + slot.base.external_velocity) * DT;
                            slot.base.external_velocity *= Fix::from_f64(0.95);
                            return;
                        }
                        // Rewind sets InstantAbility.ignoreNextCooldown when
                        // the recorder has fewer than two samples.
                        slot.rewind.finish_next_tick = true;
                        break;
                    }
                    ABILITY_DASH => {
                        // The movement slice implements the dash; unlock its
                        // internal cooldown gate and let base.step enter it.
                        self.players[index].base.cooldown = Fix::from_int(100);
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Dash;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = action;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let masked = Action {
                            angle: action.angle,
                            buttons: (action.buttons & 1) | 2,
                        };
                        // Dash is registered after SlimeController in this
                        // prefab, so its freshly enabled UpdateSim runs later
                        // in the same pass. That performs anticipation setup
                        // now; platform and bow begin updating next tick.
                        self.dash_move(index, masked, movement_input);
                        return;
                    }
                    ABILITY_GRENADE | ABILITY_SMOKE | ABILITY_TACK | ABILITY_MJOLNIR => {
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = match ability {
                            ABILITY_TACK => ActiveAbility::Tack,
                            ABILITY_MJOLNIR => ActiveAbility::Mjolnir,
                            ABILITY_SMOKE => ActiveAbility::Smoke,
                            _ => ActiveAbility::Grenade,
                        };
                        slot.active_slot = s;
                        slot.ability_entry_action = action;
                        slot.telemetry.ability_activations[s] += 1;
                        let grounded = slot.base.grounded;
                        slot.throw_item = ThrowItemState {
                            grounded,
                            direction: Vec2::UP,
                            ..ThrowItemState::default()
                        };
                        if grounded {
                            slot.base.velocity = Vec2::ZERO;
                        } else {
                            slot.base.velocity *= Fix::from_f32(0.1);
                        }
                        self.throw_item_tick(index, action, data);
                        return;
                    }
                    ABILITY_MISSILE => {
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::Missile;
                        slot.active_slot = s;
                        slot.ability_entry_action = action;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.missile = MissileFormState::default();
                        if slot.base.grounded {
                            slot.base.velocity = Vec2::ZERO;
                            slot.base.external_velocity = Vec2::ZERO;
                        }
                        return;
                    }
                    ABILITY_MINE => {
                        let grounded = self.players[index].base.grounded;
                        let form_position = if grounded {
                            let base = &self.players[index].base;
                            base.ground
                                .and_then(|ground| {
                                    self.platforms.iter().find(|platform| platform.id == ground)
                                })
                                .map_or(base.position, |platform| {
                                    platform.surface.position(base.local, data.player.radius)
                                })
                        } else {
                            self.players[index].base.position
                        };
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::Mine;
                        slot.active_slot = s;
                        slot.ability_entry_action = action;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.throw_item = ThrowItemState {
                            grounded,
                            direction: Vec2::UP,
                            ..ThrowItemState::default()
                        };
                        slot.base.position = form_position;
                        self.mine_throw_tick(index, action, data);
                        return;
                    }
                    ABILITY_TIME_STOP => {
                        let grounded = self.players[index].base.grounded;
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::TimeStop;
                        slot.active_slot = s;
                        slot.ability_entry_action = action;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.base.velocity = Vec2::ZERO;
                        slot.base.external_velocity = Vec2::ZERO;
                        slot.cast_spell = CastSpellState {
                            phase: if grounded { 0 } else { 1 },
                            ..CastSpellState::default()
                        };
                        if !grounded {
                            slot.base.rotation = Fix::ZERO;
                            slot.base.up = Vec2::UP;
                            slot.base.grounded = false;
                            slot.base.ground = None;
                            slot.base.grounded_speed = Fix::ZERO;
                        }
                        self.time_stop_cast_tick(index, action, data);
                        return;
                    }
                    ABILITY_SPIKE => {
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::Spike;
                        slot.active_slot = s;
                        slot.ability_entry_action = action;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.spike_cast = SpikeCastState {
                            collider_position: slot.base.position,
                            collider_up: slot.base.up,
                            collider_rotation: slot.base.rotation,
                            ..SpikeCastState::default()
                        };
                        return;
                    }
                    ABILITY_BOW => {
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Bow;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = action;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let slot = &mut self.players[index];
                        // BowTransform.OnEnterAbility: aim starts at
                        // PlayerInfo.upVector = Normalized(ground normal) when
                        // grounded, else Vec2.up (SlimeController.cs:666); the
                        // re-normalisation moves the unit normal by an LSB.
                        // Airborne entry slows to a tenth (slow-mo).
                        slot.bow = BowState {
                            aim: if slot.base.grounded {
                                slot.base.up.normalized()
                            } else {
                                Vec2::UP
                            },
                            ..BowState::default()
                        };
                        if slot.base.grounded {
                            // SyncPhysicsTo(PlayerInfo) re-derives the body
                            // from the surface local with no self-imposed
                            // velocity (PlayerPhysics.cs:526-535), and
                            // OnEnterAbility clears the external one
                            // (BowTransform.cs:91): this pass's PlayerBody
                            // integration does not move the Bow.
                            let platforms: &[Platform] = &self.platforms;
                            if let Some(platform) = slot.base.platform(platforms) {
                                slot.base.position =
                                    platform.surface.position(slot.base.local, slot.base.radius);
                            }
                            slot.base.velocity = Vec2::ZERO;
                            slot.base.external_velocity = Vec2::ZERO;
                        } else {
                            slot.base.velocity *= Fix::from_f32(0.1);
                            slot.base.external_velocity *= Fix::from_f32(0.1);
                            // The Bow's own PlayerPhysics starts with no
                            // attached ground (PlayerPhysics.cs:548), so a
                            // later UnGround has nothing to push off from.
                            slot.base.ground = None;
                        }
                        // BowTransform follows Ability in the prefab updater
                        // list, so a newly enabled Bow updates later in this
                        // same pass (like Dash, unlike a newly instantiated
                        // projectile which is deferred).
                        self.bow_tick(index, action, data);
                        return;
                    }
                    ABILITY_GUST => {
                        let attached_ground = self.players[index].base.ground;
                        let detach_normal = self.players[index]
                            .base
                            .ground
                            .and_then(|id| self.platforms.iter().find(|platform| platform.id == id))
                            .map(|platform| {
                                platform.surface.normal(self.players[index].base.position)
                            });
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::Gust;
                        slot.active_slot = s;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.gust = GustState {
                            attached_ground,
                            ..GustState::default()
                        };
                        // QuickSpell synchronizes the copied PlayerInfo,
                        // ungrounds by the standard 0.05-unit separation and
                        // then clears both velocity channels.
                        if let Some(normal) = detach_normal {
                            slot.base.position += normal * Fix::from_f32(0.05);
                        }
                        slot.base.rotation = Fix::ZERO;
                        slot.base.up = Vec2::UP;
                        slot.base.grounded = false;
                        slot.base.ground = None;
                        slot.base.grounded_speed = Fix::ZERO;
                        slot.base.velocity = Vec2::ZERO;
                        slot.base.external_velocity = Vec2::ZERO;
                        self.gust_tick(index, data);
                        return;
                    }
                    ABILITY_REVIVAL => {
                        let attached_ground =
                            retained_ground(&self.players[index], &self.platforms);
                        let detach_normal = attached_ground
                            .and_then(|id| self.platforms.iter().find(|platform| platform.id == id))
                            .map(|platform| {
                                platform.surface.normal(self.players[index].base.position)
                            });
                        let slot = &mut self.players[index];
                        slot.timers[s] = Fix::ZERO;
                        slot.active = ActiveAbility::Revival;
                        slot.active_slot = s;
                        slot.telemetry.ability_activations[s] += 1;
                        slot.revival = RevivalState {
                            attached_ground,
                            ..RevivalState::default()
                        };
                        if let Some(normal) = detach_normal {
                            slot.base.position += normal * Fix::from_f32(0.05);
                        }
                        slot.base.rotation = Fix::ZERO;
                        slot.base.up = Vec2::UP;
                        slot.base.grounded = false;
                        slot.base.ground = None;
                        slot.base.grounded_speed = Fix::ZERO;
                        slot.base.velocity = Vec2::ZERO;
                        slot.base.external_velocity = Vec2::ZERO;
                        self.revival_tick(index, data);
                        return;
                    }
                    ABILITY_PLATFORM => {
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Platform;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = action;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let slot = &mut self.players[index];
                        // PlatformTransform.OnEnterAbility: the body freezes at
                        // its position; the ability object carries no velocity.
                        slot.platform = PlatformState {
                            active_corner: Vec2::new(Fix::ONE, Fix::ONE),
                            ..PlatformState::default()
                        };
                        slot.base.velocity = Vec2::ZERO;
                        slot.base.external_velocity = Vec2::ZERO;
                        // PlatformTransform.OnEnterAbility (PlatformTransform.cs:122).
                        slot.base.rotation = Fix::ZERO;
                        slot.base.up = Vec2::UP;
                        slot.base.grounded = false;
                        slot.base.ground = None;
                        slot.base.grounded_speed = Fix::ZERO;
                        return;
                    }
                    // Other catalog abilities are never simulated here.
                    _ => {}
                }
            }

            // This is after OldUpdate, exactly where SlimeController compares
            // `player.isInvisible` with its previous-frame copy. A new cast on
            // the expiry frame keeps the player invisible and deliberately
            // postpones restoration, permitting chained Invisibility slots.
            let expired = {
                let slot = &self.players[index];
                !slot.invisible && slot.was_invisible
            };
            if expired {
                let slot = &mut self.players[index];
                let mut youngest = None;
                let mut minimum = Fix::from_raw(i64::MAX);
                for s in 0..3 {
                    if slot.abilities[s] != ABILITY_INVISIBILITY {
                        continue;
                    }
                    if slot.timers[s] < minimum {
                        minimum = slot.timers[s];
                        youngest = Some(s);
                    }
                    if slot.cooldowns[s] == Fix::from_int(10_000) {
                        slot.cooldowns[s] = data.abilities[ABILITY_INVISIBILITY as usize]
                            .effect_cooldown
                            .expect("Invisibility effect cooldown");
                    }
                }
                if let Some(s) = youngest {
                    slot.timers[s] = Fix::ZERO;
                }
            }
            self.players[index].was_invisible = self.players[index].invisible;
        }

        if self.players[index].active == ActiveAbility::Bow {
            self.bow_tick(index, action, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Missile {
            self.missile_form_tick(index, action, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Mine {
            self.mine_throw_tick(index, action, data);
            return;
        }
        if matches!(
            self.players[index].active,
            ActiveAbility::Grenade
                | ActiveAbility::Smoke
                | ActiveAbility::Tack
                | ActiveAbility::Mjolnir
        ) {
            self.throw_item_tick(index, action, data);
            return;
        }
        if self.players[index].active == ActiveAbility::TimeStop {
            self.time_stop_cast_tick(index, action, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Spike {
            self.spike_cast_tick(index, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Gust {
            self.gust_tick(index, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Revival {
            self.revival_tick(index, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Platform {
            self.platform_ability_tick(index, action, data);
            // PlatformTransform updates before its own
            // DestroyIfOutsideSceneBounds component. Unlike the ordinary
            // slime, this ability body's position is final for this updater
            // pass (the separate platform moves later in DetPhysics).
            if self.players[index].base.alive && self.players[index].base.outside_bounds() {
                // The slime takes over the ability body's position
                // (DestroyIfOutsideSceneBounds.cs:137, SlimeController.cs:217).
                let slot = &mut self.players[index];
                slot.reported_position = slot.base.position;
                self.kill(index, ENVIRONMENT, CAUSE_OTHER, false);
            }
            return;
        }
        if self.players[index].active == ActiveAbility::Dash {
            let starting = !self.players[index].base.dash.active;
            let source = if starting {
                self.players[index].ability_entry_action
            } else {
                action
            };
            let masked = Action {
                angle: source.angle,
                buttons: (source.buttons & 1) | if starting { 2 } else { 0 },
            };
            self.dash_move(index, masked, movement_input);
            if !self.players[index].base.dash.active {
                // Dash finished this tick.
                self.players[index].active = ActiveAbility::None;
            }
            return;
        }

        let masked = Action {
            angle: action.angle,
            buttons: action.buttons & 1,
        };
        self.players[index]
            .base
            .move_step(masked, movement_input, terrain!(self));
    }

    fn gold_mushroom_tick(&mut self, index: usize, jump_pressed: bool, data: &data::RolloutData) {
        let ability = &data.abilities[ABILITY_GOLD_MUSHROOM as usize];
        let duration = ability.effect_duration.expect("Gold Mushroom duration");
        let air_jump = ability
            .air_jump_strength
            .expect("Gold Mushroom air jump strength");
        let restored_cooldown = ability
            .effect_cooldown
            .expect("Gold Mushroom restored cooldown");
        let slot = &mut self.players[index];
        for effect_slot in 0..slot.gold_mushroom.len() {
            let effect = &mut slot.gold_mushroom[effect_slot];
            if !effect.active {
                continue;
            }
            if effect.pending_init {
                effect.pending_init = false;
                continue;
            }
            // Player.IsAlive remains true while a stock replacement exists;
            // final elimination ends the effect on its next updater pass.
            if slot.out {
                effect.active = false;
                slot.cooldowns[effect_slot] = restored_cooldown;
                continue;
            }
            effect.time += DT;
            if jump_pressed && !slot.base.grounded {
                slot.base.velocity.y = slot.base.constants.jump_strength * air_jump;
            }
            if effect.time >= duration {
                effect.active = false;
                slot.cooldowns[effect_slot] = restored_cooldown;
            }
        }
    }

    fn rewind_recorders_tick(&mut self) {
        for index in 0..self.players.len() {
            if !self.playable(index) {
                continue;
            }
            if !self.players[index].rewind.rewinding {
                if self.players[index].active == ActiveAbility::None
                    && !self.players[index].base.dash.active
                {
                    let slot = &mut self.players[index];
                    slot.rewind.push(RewindSample {
                        position: slot.base.position,
                        velocity: slot.base.velocity,
                        external_velocity: slot.base.external_velocity,
                        rotation: slot.base.rotation,
                        cooldowns: slot.timers,
                    });
                }
                continue;
            }

            if self.players[index].rewind.finish_next_tick {
                let destination = self.players[index].rewind.destination;
                let preserved = self.players[index].timers[self.players[index].rewind.ability_slot];
                let mut position = self.players[index].base.position;
                let radius = self.players[index].base.radius + Fix::from_f32(0.05);
                for _ in 0..6 {
                    let mut changed = false;
                    for platform in &self.platforms {
                        let shape = platform.collider;
                        let Ok(closest) = shape.closest_point(position) else {
                            continue;
                        };
                        let delta = position - closest;
                        let magnitude = delta.length();
                        if shape.overlaps(position) {
                            let direction = if magnitude > Fix::ZERO {
                                (closest - position) / magnitude
                            } else {
                                Vec2::UP
                            };
                            position = closest + direction * radius;
                            changed = true;
                        } else if magnitude < radius {
                            position = closest + delta / magnitude * radius;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                let slot = &mut self.players[index];
                slot.base.position = position;
                slot.base.velocity = destination.velocity;
                slot.base.external_velocity = destination.external_velocity;
                slot.base.rotation = destination.rotation;
                slot.base.up = Vec2::angle(destination.rotation).perp();
                if let Some(ground_id) = slot.rewind.attached_ground
                    && let Some(platform) = self
                        .platforms
                        .iter()
                        .find(|platform| platform.id == ground_id)
                {
                    slot.base.position +=
                        platform.surface.normal(slot.base.position) * Fix::from_f32(0.05);
                }
                slot.base.grounded = false;
                slot.base.ground = None;
                slot.base.grounded_speed = Fix::ZERO;
                slot.timers = destination.cooldowns;
                slot.timers[slot.rewind.ability_slot] = preserved;
                for ability_slot in 0..slot.timers.len() {
                    if ability_slot != slot.rewind.ability_slot {
                        // RewindRecorder (hierarchy 8) restores these before
                        // SlimeController advances its ordinary cooldown pass.
                        slot.timers[ability_slot] += DT;
                    }
                }
                slot.rewind.rewinding = false;
                slot.rewind.finish_next_tick = false;
                slot.rewind.attached_ground = None;
                continue;
            }

            let (from, target, rotation, done) = {
                let slot = &mut self.players[index];
                let rewind = &mut slot.rewind;
                rewind.playback_tick = (rewind.playback_tick + 1).min(rewind.playback_ticks);
                let t = Fix::from_int(i64::from(rewind.playback_tick))
                    / Fix::from_int(i64::from(rewind.playback_ticks));
                let eased = t
                    * t
                    * t
                    * (t * (t * Fix::from_int(6) - Fix::from_int(15)) + Fix::from_int(10));
                let (target, rotation) = rewind.evaluate_path(eased);
                (
                    slot.base.position,
                    target,
                    rotation,
                    rewind.playback_tick >= rewind.playback_ticks,
                )
            };
            let travel = target - from;
            let distance = travel.length();
            if distance > Fix::ZERO {
                let direction = travel / distance;
                let reach = distance + self.players[index].base.radius;
                let mut nearest = Fix::MAX;
                let mut hit_platform = None;
                for (platform, body) in self.platforms.iter().enumerate() {
                    if let Some(hit) = body.collider.raycast_within(from, direction, reach)
                        && hit.near < nearest
                    {
                        nearest = hit.near;
                        hit_platform = Some(platform);
                    }
                }
                if let Some(platform) = hit_platform {
                    self.platform_dynamics[platform].body.force += direction * Fix::from_int(3000);
                    // Rewind can wake an otherwise quiet dynamic platform. The
                    // platform epoch is also the deterministic dirty bit used
                    // by `platforms_tick` to leave its no-work fast path.
                    self.platforms_changed();
                }
            }
            let slot = &mut self.players[index];
            let desired = (target - slot.base.position) / DT;
            slot.base.velocity = Vec2::ZERO;
            slot.base.external_velocity +=
                ((desired - slot.base.external_velocity) / Fix::from_int(35)) * Fix::from_int(35);
            slot.base.rotation = rotation;
            slot.base.up = Vec2::angle(rotation).perp();
            if done {
                slot.rewind.finish_next_tick = true;
            }
        }
    }

    fn gust_tick(&mut self, index: usize, data: &data::RolloutData) {
        let slot = &mut self.players[index];
        slot.gust.ticks += 1;
        if slot.gust.phase == 0 && slot.gust.ticks > data.gust.cast_ticks {
            let wave = index * 3 + slot.active_slot;
            self.gust_waves[wave] = GustWave {
                active: true,
                born_tick: self.tick,
                position: slot.base.position + data.gust.shockwave_offset,
                ..GustWave::default()
            };
            slot.gust.phase = 1;
            slot.gust.ticks = 0;
        } else if slot.gust.phase == 1 && slot.gust.ticks > data.gust.exit_ticks {
            // QuickSpell calls UnGround a second time. PlayerPhysics kept the
            // old attachedGround reference even though isGrounded was false,
            // so a moving platform supplies a fresh normal for this nudge.
            let exit_normal = slot.gust.attached_ground.and_then(|id| {
                self.platforms
                    .iter()
                    .find(|platform| platform.id == id)
                    .map(|platform| platform.surface.normal(slot.base.position))
            });
            if let Some(normal) = exit_normal {
                slot.base.position += normal * Fix::from_f32(0.05);
            }
            // Ability.ExitAbility publishes the QuickSpell collider before
            // SlimeController installs the returned body and immediately
            // raycasts it (Ability.cs:151, SlimeController.cs:217-272).
            let ability_position = slot.base.position;
            slot.active = ActiveAbility::None;
            slot.base.rotation = Fix::ZERO;
            slot.base.up = Vec2::UP;
            slot.exit_to_fall();
            slot.base.collision_rays(terrain!(self));
            slot.reported_position = ability_position;
            slot.reported_tick = self.tick;
            return;
        }
        // Ability.UpdateSim follows QuickSpell in the prefab and publishes
        // the live collider position on entry and every active pass.
        slot.reported_position = slot.base.position;
        slot.reported_tick = self.tick;
    }

    fn time_stop_cast_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let published_position = self.players[index].base.position;
        {
            let slot = &mut self.players[index];
            slot.cast_spell.collider_position = published_position;
            slot.cast_spell.collider_right = slot.base.up.perp_cw();
            slot.cast_spell.collider_rotation = slot.base.rotation;
        }
        let button = 2u8 << self.players[index].active_slot;
        let phase = self.players[index].cast_spell.phase;
        if phase <= 1 {
            self.players[index].cast_spell.time += DT;
        }

        match phase {
            0 => {
                self.players[index].cast_spell.ticks += 1;
                if self.players[index].cast_spell.ticks > data.time_stop.idle_to_fall_ticks {
                    let normal = self.players[index].base.ground.and_then(|ground| {
                        self.platforms
                            .iter()
                            .find(|platform| platform.id == ground)
                            .map(|platform| {
                                platform.surface.normal(self.players[index].base.position)
                            })
                    });
                    let slot = &mut self.players[index];
                    if let Some(normal) = normal {
                        slot.base.position += normal * Fix::from_f32(0.05);
                    }
                    slot.base.grounded = false;
                    slot.base.grounded_speed = Fix::ZERO;
                    slot.base.rotation = Fix::ZERO;
                    slot.base.up = Vec2::UP;
                    slot.cast_spell.phase = 1;
                    slot.cast_spell.ticks = 0;
                }
            }
            1 => {
                let released = action.buttons & button == 0;
                if released && self.players[index].cast_spell.time <= data.time_stop.cast_time {
                    self.exit_time_stop_cast(index, data);
                    return;
                }
                if self.players[index].cast_spell.time > data.time_stop.cast_time {
                    self.players[index].cast_spell.phase = 2;
                    // CastSpell precedes SpriteAnimator in this prefab, so
                    // the newly selected cast animation advances once in the
                    // same updater pass.
                    self.players[index].cast_spell.ticks = 1;
                }
            }
            2 => {
                self.players[index].cast_spell.ticks += 1;
                if self.players[index].cast_spell.ticks > data.time_stop.cast_spell_ticks {
                    self.next_spawn_hierarchy += data.time_stop.effect_updatable_count;
                    let effect = TimeStopEffect {
                        active: true,
                        born_tick: self.tick,
                        caster: index,
                        ..TimeStopEffect::default()
                    };
                    if let Some(free) = self
                        .time_stop_effects
                        .iter()
                        .position(|candidate| !candidate.active)
                    {
                        self.time_stop_effects[free] = effect;
                    } else {
                        self.time_stop_effects.push(effect);
                    }
                    self.players[index].cast_spell.phase = 3;
                    self.players[index].cast_spell.ticks = 0;
                }
            }
            _ => {
                self.players[index].cast_spell.ticks += 1;
                if self.players[index].cast_spell.ticks > data.time_stop.ease_out_ticks {
                    self.exit_time_stop_cast(index, data);
                    return;
                }
            }
        }
        self.players[index].reported_position = published_position;
        self.players[index].reported_tick = self.tick;
    }

    fn spike_cast_tick(&mut self, index: usize, data: &data::RolloutData) {
        let published_position = self.players[index].base.position;
        {
            let slot = &mut self.players[index];
            slot.spike_cast.collider_position = published_position;
            slot.spike_cast.collider_up = slot.base.up;
            slot.spike_cast.collider_rotation = slot.base.rotation;
        }
        let cast_now = {
            let cast = &mut self.players[index].spike_cast;
            cast.ticks += 1;
            cast.phase == 0 && cast.ticks >= data.spike.cast_ticks
        };
        if cast_now {
            if self.spawn_spike(index, data) {
                self.players[index].spike_cast.phase = 1;
                self.players[index].spike_cast.ticks = 0;
            } else {
                self.exit_spike_cast(index);
                return;
            }
        }
        if self.players[index].base.grounded {
            self.players[index]
                .base
                .move_only(Vec2::ZERO, terrain!(self));
        }
        {
            let slot = &mut self.players[index];
            slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
            slot.base.external_velocity *= Fix::from_f64(0.95);
            slot.reported_position = published_position;
            slot.reported_tick = self.tick;
        }
        if self.players[index].spike_cast.phase == 1
            && self.players[index].spike_cast.ticks > data.spike.recover_ticks
        {
            self.exit_spike_cast(index);
        }
    }

    fn exit_spike_cast(&mut self, index: usize) {
        let grounded_pose = self.players[index].base.ground.and_then(|ground| {
            self.platforms
                .iter()
                .find(|platform| platform.id == ground)
                .map(|platform| platform.surface.normal(self.players[index].base.position))
        });
        let slot = &mut self.players[index];
        slot.active = ActiveAbility::None;
        if slot.base.grounded
            && let Some(up) = grounded_pose
        {
            slot.base.up = up;
            slot.base.rotation = up.perp_cw().normalized_angle();
            slot.obs_state = STATE_IDLE;
            slot.state_ticks = 0;
        } else {
            slot.base.rotation = Fix::ZERO;
            slot.base.up = Vec2::UP;
            slot.exit_to_fall();
        }
    }

    fn spawn_spike(&mut self, index: usize, data: &data::RolloutData) -> bool {
        let Some(ground_id) = self.players[index].base.ground else {
            return false;
        };
        let Some(platform_index) = self.platforms.iter().position(|p| p.id == ground_id) else {
            return false;
        };
        let body_position = self.players[index].base.position;
        let normal = self.platforms[platform_index].surface.normal(body_position);
        let Some(hit) = self.platforms[platform_index]
            .collider
            .raycast(body_position, -normal)
        else {
            return false;
        };
        let surface_position = body_position - normal * hit.far;

        let ability_slot = self.players[index].active_slot;
        if let Some(old) = self.players[index].spike_attacks[ability_slot].take() {
            self.remove_spike(old);
        }

        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.spike.attack_updatable_count;
        let local = self.platforms[platform_index]
            .surface
            .local_position(surface_position, Fix::ZERO)
            .expect("spike surface ray hit has a perimeter coordinate");
        let parent_position = self.platforms[platform_index].surface.center;
        let parent_rotation = self.platform_dynamics[platform_index].body.orientation;
        let parent_inverse_basis = Vec2::angle(-parent_rotation);
        let spike_right = normal.perp_cw();
        let spike_rotation = spike_right.normalized_angle();
        let surface_offset = data.spike.down_thrust.evaluate(Fix::ZERO) + data.spike.surface_offset;
        // Initialize calls PositionFromLocalPlayerPos even though CastSpike
        // already supplied the ray-hit point, then DPhysicsBox folds the
        // FixTransform offset into its relative child position.
        let root_position = self.platforms[platform_index]
            .surface
            .position(local, Fix::ZERO)
            + normal * surface_offset;
        let relative_right = spike_right.complex_mul(parent_inverse_basis);
        let relative_center = (root_position - parent_position).complex_mul(parent_inverse_basis)
            + data
                .spike
                .attack_collider_offset
                .complex_mul(relative_right);
        let relative_rotation = spike_rotation - parent_rotation;

        self.attach_spike_body(platform_index, data.spike.attack_body);
        let entity_index = self.entities.len();
        let id = self.next_id();
        self.entities.push(CEntity {
            id,
            kind: 1,
            layer: data.spike.attack_layer,
            owner: 0,
            flags: E_ACTIVE | E_ENABLED | E_COMPOSITE,
            hierarchy: 100_000_000 + hierarchy_base + data.spike.attack_collider_hierarchy,
            inverse_mass: data.spike.attack_body.inverse_mass.raw(),
            ..CEntity::default()
        });
        let attack = SpawnedSpike {
            active: true,
            born_tick: self.tick,
            hierarchy_base,
            entity: Some(entity_index),
            platform_index,
            local,
            relative_center,
            relative_rotation,
            surface_offset,
            time: Fix::ZERO,
            body: data.spike.attack_body,
        };
        let attack_index = if let Some(free) = self.spike_attacks.iter().position(|s| !s.active) {
            // A pooled collider receives a fresh DetPhysics id on ManualInit;
            // the old entity row remains removed like the oracle's shape list.
            self.spike_attacks[free] = attack;
            free
        } else {
            self.spike_attacks.push(attack);
            self.spike_attacks.len() - 1
        };
        self.players[index].spike_attacks[ability_slot] = Some(attack_index);

        // BoplBody.AddForceAtPosition's default Force mode is nevertheless a
        // direct velocity impulse; only ForceMode2D.Impulse multiplies by 60.
        let arm = (body_position - parent_position).normalized_safe();
        self.platform_dynamics[platform_index]
            .body
            .apply_impulse(normal * data.spike.cast_force, arm);
        self.platforms_changed();
        true
    }

    fn attach_spike_body(&mut self, platform_index: usize, child: Body) {
        let dynamics = &mut self.platform_dynamics[platform_index];
        if dynamics.attached_spikes == 0 {
            dynamics.composite_position = self.platforms[platform_index].surface.center;
        }
        let parent_mass = Fix::ONE / dynamics.body.inverse_mass;
        let child_mass = Fix::ONE / child.inverse_mass;
        let parent_moment = Fix::ONE / dynamics.body.inverse_inertia;
        let child_moment = Fix::ONE / child.inverse_inertia;
        dynamics.body.inverse_mass = Fix::ONE / (parent_mass + child_mass);
        dynamics.body.inverse_inertia = Fix::ONE / (parent_moment + child_moment);
        dynamics.attached_spikes += 1;
    }

    fn remove_spike(&mut self, spike_index: usize) {
        if !self.spike_attacks[spike_index].active {
            return;
        }
        let platform_index = self.spike_attacks[spike_index].platform_index;
        let child = self.spike_attacks[spike_index].body;
        self.spike_attacks[spike_index].active = false;
        if let Some(entity) = self.spike_attacks[spike_index].entity {
            self.entities[entity].flags = E_REMOVED;
        }
        let remaining = self
            .spike_attacks
            .iter()
            .filter(|spike| spike.active && spike.platform_index == platform_index)
            .count() as u32;
        // DestroyFix's composite detach retains the moving parent's effective
        // velocity/orientation for the replacement created later in the same
        // callback, while the new root+child mass is recomputed from authored
        // bodies. Keeping the old effective inverse mass here would make the
        // cast impulse behave as if both attacks were still attached.
        let dynamics = &mut self.platform_dynamics[platform_index];
        let old = dynamics.body;
        dynamics.body = dynamics.root_body;
        // CreateMultiComposite's one-root detach restores root mass, but its
        // subsequent same-callback attach observes the prior composite moment
        // through the retained parent object. This shipped asymmetry is
        // visible as root+one-child translation with root+two-child torque.
        if remaining == 0 {
            dynamics.body.inverse_inertia = old.inverse_inertia;
        } else {
            let count = Fix::from_int(i64::from(remaining));
            let root_mass = Fix::ONE / dynamics.root_body.inverse_mass;
            let child_mass = Fix::ONE / child.inverse_mass;
            let root_moment = Fix::ONE / dynamics.root_body.inverse_inertia;
            let child_moment = Fix::ONE / child.inverse_inertia;
            dynamics.body.inverse_mass = Fix::ONE / (root_mass + child_mass * count);
            dynamics.body.inverse_inertia = Fix::ONE / (root_moment + child_moment * count);
        }
        dynamics.body.velocity = old.velocity;
        dynamics.body.angular_velocity = old.angular_velocity;
        dynamics.body.orientation = old.orientation;
        dynamics.body.force = old.force;
        dynamics.attached_spikes = remaining;
        dynamics.composite_position = self.platforms[platform_index].surface.center;
        dynamics.basis = None;
        self.platforms_changed();
    }

    fn exit_time_stop_cast(&mut self, index: usize, data: &data::RolloutData) {
        let normal = self.players[index].base.ground.and_then(|ground| {
            self.platforms
                .iter()
                .find(|platform| platform.id == ground)
                .map(|platform| platform.surface.normal(self.players[index].base.position))
        });
        let slot = &mut self.players[index];
        if let Some(normal) = normal {
            slot.base.position += normal * Fix::from_f32(0.05);
        }
        slot.base.rotation = Fix::ZERO;
        slot.base.up = Vec2::UP;
        let published_position = slot.base.position;
        // Ability.ExitAbility publishes and freezes the ability collider at
        // this pose, while the returned slime is placed at ExitAbilityOffset.
        slot.cast_spell.collider_position = published_position;
        slot.cast_spell.collider_right = Vec2::RIGHT;
        slot.cast_spell.collider_rotation = Fix::ZERO;
        slot.base.position += data.time_stop.exit_offset;
        slot.base.velocity = Vec2::ZERO;
        slot.base.external_velocity = Vec2::ZERO;
        slot.base.grounded = false;
        slot.base.grounded_speed = Fix::ZERO;
        slot.active = ActiveAbility::None;
        slot.exit_to_fall();
        slot.base.collision_rays(terrain!(self));
        slot.reported_position = published_position;
        slot.reported_tick = self.tick;
    }

    fn register_pending_time_stops(&mut self) {
        for effect in &mut self.time_stop_effects {
            if !effect.active || effect.registered || effect.born_tick >= self.tick {
                continue;
            }
            effect.registered = true;
            effect.seconds_elapsed = 0.0;
            let team = self.players[effect.caster].team;
            for index in 0..self.players.len() {
                self.time_stop_protected[index] = self.players[index].team == team;
            }
        }
    }

    fn time_stop_effects_tick(&mut self, data: &data::RolloutData) {
        for effect in &mut self.time_stop_effects {
            if !effect.active || !effect.registered {
                continue;
            }
            effect.seconds_elapsed += 1.0 / 60.0;
            self.player_time_scale = Fix::ZERO;
            if effect.seconds_elapsed
                > data.time_stop.duration_f32 + data.time_stop.exit_seconds_f32
            {
                effect.active = false;
                effect.registered = false;
                self.player_time_scale = Fix::ONE;
                self.time_stop_protected = [false; MAX_PLAYERS];
            } else {
                let team = self.players[effect.caster].team;
                for index in 0..self.players.len() {
                    self.time_stop_protected[index] = self.players[index].team == team;
                }
            }
        }
    }

    fn spike_attacks_tick(&mut self, data: &data::RolloutData, updater_dt: Fix) {
        let curve_end = data
            .spike
            .down_thrust
            .keys
            .last()
            .expect("authored SpikeAttack curve")
            .time;
        for attack in &mut self.spike_attacks {
            if !attack.active || attack.born_tick >= self.tick {
                continue;
            }
            attack.time += updater_dt;
            if attack.time > curve_end || attack.platform_index >= self.platforms.len() {
                continue;
            }
            attack.surface_offset =
                data.spike.down_thrust.evaluate(attack.time) + data.spike.surface_offset;
            let platform = &self.platforms[attack.platform_index];
            let parent_rotation = self.platform_dynamics[attack.platform_index]
                .body
                .orientation;
            let spike_right = Vec2::angle(parent_rotation + attack.relative_rotation);
            let root_position = platform.surface.position(attack.local, Fix::ZERO)
                + spike_right.perp() * attack.surface_offset;
            let inverse_basis = Vec2::angle(-parent_rotation);
            let relative_right = spike_right.complex_mul(inverse_basis);
            attack.relative_center = (root_position - platform.surface.center)
                .complex_mul(inverse_basis)
                + data
                    .spike
                    .attack_collider_offset
                    .complex_mul(relative_right);
        }
    }

    fn spike_contacts(&mut self, data: &data::RolloutData) {
        // Circle/Box and Box/Box callbacks run after all rigid-body groups.
        // Player colliders are triggers, so contact only schedules callbacks:
        // PlayerCollision sees the persistent LethalTerrain Explosion first.
        let mut victims = [false; MAX_PLAYERS];
        for spike in self.spike_attacks.iter().copied() {
            if !spike.active || spike.platform_index >= self.platforms.len() {
                continue;
            }
            let parent = &self.platform_dynamics[spike.platform_index];
            let basis = Vec2::angle(parent.body.orientation + spike.relative_rotation);
            let center = parent.composite_position
                + spike
                    .relative_center
                    .complex_mul(self.platforms[spike.platform_index].surface.right);
            let extents = data::surface_extents(data.spike.attack_extents);
            let attack_box = OrientedBox {
                center,
                right: basis * extents.x,
                up: basis.perp() * extents.y,
                inverse_extents: Vec2::new(
                    Fix::ONE / data.spike.attack_extents.x,
                    Fix::ONE / data.spike.attack_extents.y,
                ),
            };
            for (player_index, victim) in victims.iter_mut().enumerate().take(self.players.len()) {
                if *victim || !self.playable(player_index) {
                    continue;
                }
                let slot = &self.players[player_index];
                let entity = match slot.active {
                    ActiveAbility::None => slot.body_entity,
                    ActiveAbility::Dash => slot.dash_entity,
                    ActiveAbility::Bow => slot.bow_entity,
                    ActiveAbility::Grenade
                    | ActiveAbility::Smoke
                    | ActiveAbility::Tack
                    | ActiveAbility::Mjolnir
                    | ActiveAbility::Missile => slot.throw_entity,
                    ActiveAbility::Mine => slot.mine_entity,
                    ActiveAbility::Spike => slot.spike_entity,
                    ActiveAbility::TimeStop => slot.time_stop_entity,
                    ActiveAbility::Gust => slot.gust_entity,
                    ActiveAbility::Revival => slot.revival_entity,
                    ActiveAbility::Platform => slot.circle_entity,
                };
                if entity == usize::MAX || self.entities[entity].flags & E_ACTIVE == 0 {
                    continue;
                }
                let row = self.entities[entity];
                *victim = if row.kind == 0 {
                    crate::physics::contact::circle_box(
                        Circle {
                            center: Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y)),
                            radius: Fix::from_raw(row.radius),
                        },
                        attack_box,
                    )
                    .is_some()
                } else {
                    let player_box = OrientedBox {
                        center: Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y)),
                        right: Vec2::new(Fix::from_raw(row.right_x), Fix::from_raw(row.right_y)),
                        up: Vec2::new(Fix::from_raw(row.up_x), Fix::from_raw(row.up_y)),
                        inverse_extents: Vec2::new(
                            Fix::ONE
                                / Vec2::new(Fix::from_raw(row.right_x), Fix::from_raw(row.right_y))
                                    .length(),
                            Fix::ONE
                                / Vec2::new(Fix::from_raw(row.up_x), Fix::from_raw(row.up_y))
                                    .length(),
                        ),
                    };
                    crate::physics::contact::box_box(player_box, attack_box).is_some()
                };
            }
        }
        for (player_index, &victim) in victims.iter().enumerate().take(self.players.len()) {
            if victim && self.playable(player_index) {
                // Spike's Explosion.PlayerOwnerId is never assigned and
                // therefore remains zero, which is neither a live player id
                // nor the game's environmental sentinel.
                self.kill(player_index, 0, CAUSE_OTHER, true);
            }
        }
    }

    fn revival_tick(&mut self, index: usize, data: &data::RolloutData) {
        let slot = &mut self.players[index];
        slot.revival.ticks += 1;
        if slot.revival.phase == 0 && slot.revival.ticks > data.revival.cast_ticks {
            let ability_slot = slot.active_slot;
            let spawn = clamp_revival_position(slot.base.position + data.revival.shockwave_offset);
            if let Some(old) = slot.revival_markers[ability_slot].take() {
                self.end_revival_marker(old);
            }
            let marker_index = self.revival_marker_pool.len();
            let hierarchy_base = self.next_spawn_hierarchy;
            self.next_spawn_hierarchy += data.revival.marker_updatable_count;
            self.revival_marker_pool.push(RevivalMarker {
                active: true,
                registered: false,
                born_tick: self.tick,
                owner: index,
                ability_slot,
                entity: None,
                hierarchy_base,
                position: spawn,
                velocity: Vec2::ZERO,
            });
            self.players[index].revival_markers[ability_slot] = Some(marker_index);
            self.players[index].revival.phase = 1;
            self.players[index].revival.ticks = 0;
        } else if slot.revival.phase == 1 && slot.revival.ticks > data.revival.exit_ticks {
            let exit_normal = slot.revival.attached_ground.and_then(|id| {
                self.platforms
                    .iter()
                    .find(|platform| platform.id == id)
                    .map(|platform| platform.surface.normal(slot.base.position))
            });
            if let Some(normal) = exit_normal {
                slot.base.position += normal * Fix::from_f32(0.05);
            }
            slot.reported_position = slot.base.position;
            slot.reported_tick = self.tick;
            slot.exit_to_fall();
            slot.active = ActiveAbility::None;
            slot.base.rotation = Fix::ZERO;
            slot.base.up = Vec2::UP;
            slot.base.collision_rays(terrain!(self));
            return;
        }
        self.players[index].reported_position = self.players[index].base.position;
        self.players[index].reported_tick = self.tick;
    }

    fn end_revival_marker(&mut self, marker_index: usize) {
        let marker = &mut self.revival_marker_pool[marker_index];
        marker.active = false;
        marker.registered = false;
        if let Some(entity) = marker.entity {
            self.entities[entity].flags = E_REMOVED;
        }
    }

    fn register_pending_revival_markers(&mut self, data: &data::RolloutData) {
        for marker_index in 0..self.revival_marker_pool.len() {
            let marker = self.revival_marker_pool[marker_index];
            if !marker.active || marker.registered || marker.born_tick >= self.tick {
                continue;
            }
            let entity = self.entities.len();
            let id = self.next_id();
            self.entities.push(CEntity {
                id,
                kind: 0,
                layer: data.revival.marker_layer,
                owner: marker.owner as u32 + 1,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000
                    + marker.hierarchy_base
                    + data.revival.marker_collider_hierarchy,
                x: (marker.position + data.revival.marker_offset).x.raw(),
                y: (marker.position + data.revival.marker_offset).y.raw(),
                radius: data.revival.marker_radius.raw(),
                inverse_mass: data.revival.marker_body.inverse_mass.raw(),
                ..CEntity::default()
            });
            self.revival_marker_pool[marker_index].registered = true;
            self.revival_marker_pool[marker_index].entity = Some(entity);
        }
    }

    fn register_pending_teleport_markers(&mut self, data: &data::RolloutData) {
        for marker_index in 0..self.teleport_marker_pool.len() {
            let marker = self.teleport_marker_pool[marker_index];
            if !marker.active || marker.registered || marker.born_tick >= self.tick {
                continue;
            }
            let entity = self.entities.len();
            let id = self.next_id();
            self.entities.push(CEntity {
                id,
                kind: 0,
                layer: data.teleport.marker_layer,
                owner: 0,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000
                    + marker.hierarchy_base
                    + data.teleport.marker_collider_hierarchy,
                x: (marker.position + data.teleport.marker_offset).x.raw(),
                y: (marker.position + data.teleport.marker_offset).y.raw(),
                radius: data.teleport.marker_radius.raw(),
                inverse_mass: data.teleport.marker_body.inverse_mass.raw(),
                ..CEntity::default()
            });
            self.teleport_marker_pool[marker_index].registered = true;
            self.teleport_marker_pool[marker_index].entity = Some(entity);
        }
    }

    /// Remove callback-destroyed items, register explosions queued by the
    /// preceding pass, and execute both Shockwave.Init calls before the
    /// updater loop begins.
    fn register_pending_grenade_explosions(&mut self, data: &data::RolloutData) {
        for item in &self.thrown_items {
            if !item.active
                && let Some(entity) = item.entity
                && self.entities[entity].flags & E_DESTROYED != 0
            {
                self.entities[entity].flags = E_REMOVED;
            }
        }
        for explosion_index in 0..self.grenade_explosions.len() {
            let explosion = self.grenade_explosions[explosion_index];
            if !explosion.active || explosion.registered || explosion.born_tick >= self.tick {
                continue;
            }
            let entity = match explosion.entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.grenade_explosions[explosion_index].entity = Some(entity);
                    entity
                }
            };
            let id = self.next_id();
            let collider = OrientedBox::from_basis(
                explosion.position,
                Vec2::RIGHT,
                data.grenade.explosion_extents,
            );
            self.entities[entity] = CEntity {
                id,
                kind: 1,
                layer: data.grenade.explosion_layer,
                flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
                hierarchy: 100_000_000
                    + explosion.hierarchy_base
                    + data.grenade.explosion_collider_hierarchy,
                x: explosion.position.x.raw(),
                y: explosion.position.y.raw(),
                right_x: collider.right.x.raw(),
                right_y: collider.right.y.raw(),
                up_x: collider.up.x.raw(),
                up_y: collider.up.y.raw(),
                inverse_mass: -Fix::ONE.raw(),
                ..CEntity::default()
            };
            self.grenade_explosions[explosion_index].registered = true;
            for wave in data.grenade.explosion_shockwaves {
                self.apply_explosion_shockwave(explosion.position, wave);
            }
        }
    }

    fn register_pending_smoke(&mut self, data: &data::RolloutData) {
        for cloud_index in 0..self.smoke_clouds.len() {
            let cloud = self.smoke_clouds[cloud_index];
            if !cloud.active || cloud.registered || cloud.born_tick >= self.tick {
                continue;
            }
            let entity = match cloud.entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.smoke_clouds[cloud_index].entity = Some(entity);
                    entity
                }
            };
            let id = self.next_id();
            let delay = self
                .rng
                .fix_range(data.smoke.min_ignition_delay, data.smoke.max_ignition_delay);
            self.entities[entity] = CEntity {
                id,
                kind: 0,
                layer: data.smoke.cloud_layer,
                owner: 0,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000 + cloud.hierarchy_base + data.smoke.cloud_collider_hierarchy,
                x: cloud.position.x.raw(),
                y: cloud.position.y.raw(),
                radius: (data.smoke.cloud_radii[0] * cloud.scale).raw(),
                rotation: cloud.rotation.raw(),
                inverse_mass: data.smoke.cloud_body.inverse_mass.raw(),
                ..CEntity::default()
            };
            let cloud = &mut self.smoke_clouds[cloud_index];
            cloud.registered = true;
            cloud.ignition_time = -Fix::ONE;
            cloud.ignition_delay = delay;
        }

        for explosion_index in 0..self.smoke_explosions.len() {
            let explosion = self.smoke_explosions[explosion_index];
            if !explosion.active || explosion.registered || explosion.born_tick >= self.tick {
                continue;
            }
            let entity = match explosion.entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.smoke_explosions[explosion_index].entity = Some(entity);
                    entity
                }
            };
            let id = self.next_id();
            let extents = data.smoke.explosion_extents * explosion.scale;
            let collider = OrientedBox::from_basis(explosion.position, Vec2::RIGHT, extents);
            self.entities[entity] = CEntity {
                id,
                kind: 1,
                layer: data.smoke.explosion_layer,
                owner: explosion.owner as u32 + 1,
                flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
                hierarchy: 100_000_000
                    + explosion.hierarchy_base
                    + data.smoke.explosion_collider_hierarchy,
                x: explosion.position.x.raw(),
                y: explosion.position.y.raw(),
                right_x: collider.right.x.raw(),
                right_y: collider.right.y.raw(),
                up_x: collider.up.x.raw(),
                up_y: collider.up.y.raw(),
                inverse_mass: -Fix::ONE.raw(),
                ..CEntity::default()
            };
            self.smoke_explosions[explosion_index].registered = true;
        }
    }

    fn apply_explosion_shockwave(&mut self, position: Vec2, wave: data::ExplosionShockwaveData) {
        let query = Circle {
            center: position,
            radius: wave.radius,
        };
        if wave.collision_mask & (1 << 11) != 0 {
            let mut applied = false;
            for platform_index in 0..self.platforms.len() {
                if crate::physics::contact::circle_rounded(
                    query,
                    self.platforms[platform_index].collider,
                )
                .is_none()
                {
                    continue;
                }
                let platform = &self.platforms[platform_index];
                let Ok(point) = platform.surface.closest_point(position) else {
                    continue;
                };
                let direction = -platform.surface.normal(point);
                let distance = ((position - point).length_squared() / Fix::from_int(10))
                    .max(Fix::from_f64(1.5));
                let impulse = direction.normalized_safe()
                    * (wave.platform_force / distance)
                    * Fix::from_int(60);
                let arm = (point - platform.surface.center).normalized_safe();
                self.platform_dynamics[platform_index]
                    .body
                    .apply_impulse(impulse, arm);
                applied = true;
            }
            if applied {
                self.platforms_changed();
            }
        }
        if wave.collision_mask & (1 << LAYER_PLAYER) == 0 {
            return;
        }
        for player_index in 0..self.players.len() {
            if !self.playable(player_index)
                || crate::physics::contact::circle_box(
                    query,
                    self.players[player_index].base.collider,
                )
                .is_none()
            {
                continue;
            }
            let target = &mut self.players[player_index];
            let delta = target.base.position - position;
            let distance = (delta.length_squared() / Fix::from_int(10)).max(Fix::from_int(3));
            let mut force = delta * (wave.default_force / distance / Fix::from_int(7));
            if target.base.grounded {
                force *= wave.grounded_player_multiplier;
            }
            target.base.external_velocity += force * Fix::from_int(35);
        }
    }

    fn grenade_explosions_tick(&mut self, data: &data::RolloutData, updater_dt: Fix) {
        for explosion in &mut self.grenade_explosions {
            if !explosion.active || !explosion.registered {
                continue;
            }
            let entity = explosion.entity.expect("registered grenade explosion");
            if explosion.age >= data.grenade.explosion_animation_ticks {
                explosion.active = false;
                explosion.registered = false;
                self.entities[entity].flags = E_REMOVED;
                continue;
            }
            self.entities[entity].flags = if explosion.age == 0 {
                E_ACTIVE | E_ENABLED | E_TRIGGER
            } else {
                E_ACTIVE | E_TRIGGER
            };
            // Explosion.UpdateSim disables the one-frame hitbox even while
            // stopped, while its ownerless SpriteAnimator does not advance.
            if updater_dt > Fix::ZERO || explosion.age == 0 {
                explosion.age += 1;
            }
        }
    }

    fn missile_explosions_tick(&mut self, data: &data::RolloutData, updater_dt: Fix) {
        for explosion in &mut self.missile_explosions {
            if !explosion.active || !explosion.registered {
                continue;
            }
            let entity = explosion.entity.expect("registered Missile explosion");
            if explosion.age >= data.missile.explosion_animation_ticks {
                explosion.active = false;
                explosion.registered = false;
                self.entities[entity].flags = E_REMOVED;
                continue;
            }
            self.entities[entity].flags = if explosion.age == 0 {
                E_ACTIVE | E_ENABLED | E_TRIGGER
            } else {
                E_ACTIVE | E_TRIGGER
            };
            if updater_dt > Fix::ZERO || explosion.age == 0 {
                explosion.age += 1;
            }
        }
    }

    fn smoke_tick(&mut self, data: &data::RolloutData, updater_dt: Fix) {
        let mut detonations = [usize::MAX; 64];
        let mut detonation_count = 0;
        for cloud_index in 0..self.smoke_clouds.len() {
            let cloud = &mut self.smoke_clouds[cloud_index];
            if !cloud.active || !cloud.registered {
                continue;
            }
            let frame = data
                .smoke
                .cloud_enter_frame_ticks
                .partition_point(|&end| end <= cloud.age_ticks)
                .min(data.smoke.cloud_radii.len() - 1);
            if let Some(entity) = cloud.entity {
                self.entities[entity].radius = (data.smoke.cloud_radii[frame] * cloud.scale).raw();
            }
            if updater_dt > Fix::ZERO {
                cloud.age_ticks = cloud.age_ticks.saturating_add(1);
            }
            if cloud.ignition_time >= Fix::ZERO {
                cloud.ignition_time += updater_dt;
                if cloud.ignition_time > cloud.ignition_delay
                    && detonation_count < detonations.len()
                {
                    detonations[detonation_count] = cloud_index;
                    detonation_count += 1;
                }
            }
        }
        for &cloud_index in &detonations[..detonation_count] {
            self.detonate_smoke_cloud(cloud_index, data);
        }

        for explosion in &mut self.smoke_explosions {
            if !explosion.active || !explosion.registered {
                continue;
            }
            let entity = explosion.entity.expect("registered smoke explosion");
            if explosion.age >= data.smoke.explosion_animation_ticks {
                explosion.active = false;
                explosion.registered = false;
                self.entities[entity].flags = E_REMOVED;
                continue;
            }
            self.entities[entity].flags = if explosion.age == 0 {
                E_ACTIVE | E_ENABLED | E_TRIGGER
            } else {
                E_ACTIVE | E_TRIGGER
            };
            if updater_dt > Fix::ZERO || explosion.age == 0 {
                explosion.age += 1;
            }
        }
    }

    fn teleport_effects_tick(&mut self, data: &data::RolloutData) {
        for index in 0..self.players.len() {
            if !self.players[index].teleport_effect_active {
                continue;
            }
            self.players[index].teleport_effect_time += DT;
            if self.players[index].teleport_effect_time > data.teleport.effect_duration {
                self.players[index].teleport_effect_active = false;
                self.players[index].teleport_effect_time = Fix::ZERO;
            }
            // TeleportedObjectEffect still shakes its visual copy on the
            // destruction pass. RandomInsideUnitCircle performs two
            // RandomFix calls; each spans integer floors and consumes two
            // Unity.Mathematics.Random samples.
            let _ = self.rng.fix_range(Fix::ZERO, Fix::TAU);
            let _ = self.rng.fix_range(Fix::ZERO, Fix::ONE);
        }
    }

    fn teleport_pending_tick(&mut self, data: &data::RolloutData) {
        for owner in 0..self.players.len() {
            let Some(marker_index) = self.players[owner].teleport_pending_marker.take() else {
                continue;
            };
            let marker = self.teleport_marker_pool[marker_index];
            if !marker.active {
                continue;
            }
            let from = self.players[owner].teleport_pending_position;
            let diff = marker.position - from;
            let player_query = Circle {
                center: from,
                radius: data.teleport.player_radius,
            };
            let marker_query = Circle {
                center: marker.position,
                radius: data.teleport.marker_teleport_radius,
            };
            let mut at_player = [false; MAX_PLAYERS];
            let mut at_marker = [false; MAX_PLAYERS];
            for target in 0..self.players.len() {
                if !self.players[target].base.alive || self.players[target].teleport_effect_active {
                    continue;
                }
                let collider = self.players[target].base.collider;
                at_player[target] =
                    crate::physics::contact::circle_box_overlaps(player_query, collider);
                at_marker[target] =
                    crate::physics::contact::circle_box_overlaps(marker_query, collider);
            }
            for target in 0..self.players.len() {
                let offset = if at_player[target] {
                    Some(diff)
                } else if at_marker[target] {
                    Some(-diff)
                } else {
                    None
                };
                let Some(offset) = offset else { continue };
                let slot = &mut self.players[target];
                slot.base.force_unground(&self.platforms);
                slot.base.rotation = Fix::ZERO;
                slot.base.up = Vec2::UP;
                slot.base.velocity = Vec2::ZERO;
                slot.base.position += offset;
                slot.base.sync_collider();
                slot.obs_state = STATE_FALL;
                slot.state_ticks = 0;
                slot.teleport_effect_active = true;
                slot.teleport_effect_time = Fix::ZERO;
                self.next_spawn_hierarchy += data.teleport.effect_updatable_count;
            }
            let marker = &mut self.teleport_marker_pool[marker_index];
            marker.active = false;
            marker.registered = false;
            if let Some(entity) = marker.entity {
                self.entities[entity].flags = E_REMOVED;
            }
        }
    }

    fn register_pending_revival_clones(&mut self, data: &data::RolloutData) {
        for index in 0..self.players.len() {
            if !self.players[index].revival_clone_pending {
                continue;
            }
            let marker_index = self.players[index]
                .revival_target_marker
                .expect("pending Revival clone marker");
            let position = self.revival_marker_pool[marker_index].position;
            let abilities = self.players[index].abilities;

            let player_hierarchy = self.next_spawn_hierarchy;
            self.next_spawn_hierarchy += 8;
            let mut ability_hierarchy = [0u32; 3];
            for (slot, hierarchy) in ability_hierarchy.iter_mut().enumerate() {
                *hierarchy = self.next_spawn_hierarchy;
                self.next_spawn_hierarchy +=
                    data.abilities[abilities[slot] as usize].updatable_count;
            }

            let mut platform_entity = usize::MAX;
            let mut gust_entity = usize::MAX;
            let mut revival_entity = usize::MAX;
            let mut time_stop_entity = usize::MAX;
            for (slot, &hierarchy) in ability_hierarchy.iter().enumerate() {
                let (radius, offset, relative_hierarchy, target) = match abilities[slot] {
                    ABILITY_GUST => (
                        data.gust.hurtbox_radius,
                        data.gust.collider_offset,
                        8,
                        &mut gust_entity,
                    ),
                    ABILITY_PLATFORM => (
                        data.platform.hurtbox_radius,
                        Vec2::ZERO,
                        7,
                        &mut platform_entity,
                    ),
                    ABILITY_REVIVAL => (
                        data.revival.hurtbox_radius,
                        data.revival.collider_offset,
                        8,
                        &mut revival_entity,
                    ),
                    ABILITY_TIME_STOP => (
                        data.time_stop.hurtbox_radius,
                        data.time_stop.collider_offset,
                        data.time_stop.collider_hierarchy,
                        &mut time_stop_entity,
                    ),
                    _ => continue,
                };
                let id = self.next_id();
                *target = self.entities.len();
                self.entities.push(CEntity {
                    id,
                    kind: 0,
                    layer: LAYER_PLAYER,
                    flags: E_ENABLED | E_TRIGGER,
                    hierarchy: 100_000_000 + hierarchy + relative_hierarchy,
                    x: offset.x.raw(),
                    y: offset.y.raw(),
                    radius: radius.raw(),
                    inverse_mass: -Fix::ONE.raw(),
                    ..CEntity::default()
                });
            }

            let body_entity = self.entities.len();
            let body_id = self.next_id();
            self.entities.push(CEntity {
                id: body_id,
                kind: 1,
                layer: LAYER_PLAYER,
                owner: index as u32 + 1,
                flags: E_ACTIVE | E_TRIGGER,
                hierarchy: 100_000_000 + player_hierarchy + 6,
                x: position.x.raw(),
                y: position.y.raw(),
                right_x: data.player.hurtbox_extents.x.raw(),
                up_y: data.player.hurtbox_extents.y.raw(),
                inverse_mass: -Fix::ONE.raw(),
                ..CEntity::default()
            });
            let mut dash_entity = usize::MAX;
            let mut bow_entity = usize::MAX;
            let mut throw_entity = usize::MAX;
            let mut mine_entity = usize::MAX;
            let mut spike_entity = usize::MAX;
            for (slot, &hierarchy) in ability_hierarchy.iter().enumerate() {
                let (extents, relative_hierarchy, target) = match abilities[slot] {
                    ABILITY_DASH => (data.dash.hurtbox_extents, 7, &mut dash_entity),
                    ABILITY_BOW => (data.bow.hurtbox_extents, 7, &mut bow_entity),
                    ability
                    @ (ABILITY_GRENADE | ABILITY_SMOKE | ABILITY_TACK | ABILITY_MJOLNIR) => (
                        throw_config(data, ability).hurtbox_extents,
                        8,
                        &mut throw_entity,
                    ),
                    ABILITY_MISSILE => (data.missile.hurtbox_extents, 7, &mut throw_entity),
                    ABILITY_MINE => (
                        data::surface_extents(data.mine.hurtbox_extents),
                        7,
                        &mut mine_entity,
                    ),
                    ABILITY_SPIKE => (
                        data.spike.hurtbox_extents,
                        data.spike.collider_hierarchy,
                        &mut spike_entity,
                    ),
                    _ => continue,
                };
                let id = self.next_id();
                *target = self.entities.len();
                self.entities.push(CEntity {
                    id,
                    kind: 1,
                    layer: LAYER_PLAYER,
                    flags: E_ENABLED | E_TRIGGER,
                    hierarchy: 100_000_000 + hierarchy + relative_hierarchy,
                    x: if abilities[slot] == ABILITY_SPIKE {
                        data.spike.collider_offset.x.raw()
                    } else if abilities[slot] == ABILITY_MINE {
                        data.mine.collider_offset.x.raw()
                    } else {
                        0
                    },
                    y: if abilities[slot] == ABILITY_SPIKE {
                        data.spike.collider_offset.y.raw()
                    } else if abilities[slot] == ABILITY_MINE {
                        data.mine.collider_offset.y.raw()
                    } else {
                        0
                    },
                    right_x: extents.x.raw(),
                    up_y: extents.y.raw(),
                    inverse_mass: -Fix::ONE.raw(),
                    ..CEntity::default()
                });
            }
            // ReviveEffect is instantiated after the clone and its copied
            // ability prefabs. It has no collider but consumes hierarchy.
            self.next_spawn_hierarchy += data.revival.effect_updatable_count;

            let slot = &mut self.players[index];
            slot.base.position = position;
            slot.base.velocity = Vec2::ZERO;
            slot.base.external_velocity = Vec2::ZERO;
            slot.base.rotation = Fix::ZERO;
            slot.base.up = Vec2::UP;
            slot.base.grounded = false;
            slot.base.ground = None;
            slot.base.grounded_speed = Fix::ZERO;
            slot.base.sync_collider();
            slot.reported_position = position;
            slot.reported_tick = self.tick;
            slot.obs_state = STATE_NOTHING;
            slot.timers.fill(Fix::from_f32(-1.5));
            slot.active = ActiveAbility::None;
            slot.missile = MissileFormState::default();
            slot.throw_item = ThrowItemState::default();
            slot.cast_spell = CastSpellState::default();
            slot.spike_cast = SpikeCastState::default();
            slot.spike_attacks = [None; 3];
            slot.rewind = RewindState::default();
            slot.body_entity = body_entity;
            slot.dash_entity = dash_entity;
            slot.bow_entity = bow_entity;
            slot.throw_entity = throw_entity;
            slot.mine_entity = mine_entity;
            slot.spike_entity = spike_entity;
            slot.circle_entity = platform_entity;
            slot.gust_entity = gust_entity;
            slot.revival_entity = revival_entity;
            slot.time_stop_entity = time_stop_entity;
            slot.revival_clone_pending = false;
        }
    }

    fn register_pending_stock_clones(&mut self, data: &data::RolloutData) {
        for index in 0..self.players.len() {
            if !self.players[index].stock_clone_pending {
                continue;
            }
            for entity in [
                self.players[index].body_entity,
                self.players[index].dash_entity,
                self.players[index].circle_entity,
                self.players[index].bow_entity,
                self.players[index].throw_entity,
                self.players[index].mine_entity,
                self.players[index].gust_entity,
                self.players[index].revival_entity,
                self.players[index].spike_entity,
                self.players[index].time_stop_entity,
            ] {
                if entity != usize::MAX {
                    self.entities[entity].flags = E_REMOVED;
                }
            }

            let abilities = self.players[index].abilities;
            let position = self.players[index].base.position;
            let player_hierarchy = self.next_spawn_hierarchy;
            self.next_spawn_hierarchy += 8;
            let mut ability_hierarchy = [0u32; 3];
            for (slot, hierarchy) in ability_hierarchy.iter_mut().enumerate() {
                *hierarchy = self.next_spawn_hierarchy;
                self.next_spawn_hierarchy +=
                    data.abilities[abilities[slot] as usize].updatable_count;
            }

            let mut platform_entity = usize::MAX;
            let mut gust_entity = usize::MAX;
            let mut revival_entity = usize::MAX;
            let mut time_stop_entity = usize::MAX;
            for (slot, &hierarchy) in ability_hierarchy.iter().enumerate() {
                let (radius, offset, relative_hierarchy, target) = match abilities[slot] {
                    ABILITY_GUST => (
                        data.gust.hurtbox_radius,
                        data.gust.collider_offset,
                        8,
                        &mut gust_entity,
                    ),
                    ABILITY_PLATFORM => (
                        data.platform.hurtbox_radius,
                        Vec2::ZERO,
                        7,
                        &mut platform_entity,
                    ),
                    ABILITY_REVIVAL => (
                        data.revival.hurtbox_radius,
                        data.revival.collider_offset,
                        8,
                        &mut revival_entity,
                    ),
                    ABILITY_TIME_STOP => (
                        data.time_stop.hurtbox_radius,
                        data.time_stop.collider_offset,
                        data.time_stop.collider_hierarchy,
                        &mut time_stop_entity,
                    ),
                    _ => continue,
                };
                let id = self.next_id();
                *target = self.entities.len();
                self.entities.push(CEntity {
                    id,
                    kind: 0,
                    layer: LAYER_PLAYER,
                    flags: E_ENABLED | E_TRIGGER,
                    hierarchy: 100_000_000 + hierarchy + relative_hierarchy,
                    x: offset.x.raw(),
                    y: offset.y.raw(),
                    radius: radius.raw(),
                    inverse_mass: -Fix::ONE.raw(),
                    ..CEntity::default()
                });
            }

            let mut indicator_entities = [usize::MAX; 3];
            for (indicator, target) in indicator_entities.iter_mut().enumerate() {
                let hierarchy_base = self.next_spawn_hierarchy;
                self.next_spawn_hierarchy += data.revival.marker_updatable_count;
                let id = self.next_id();
                *target = self.entities.len();
                let indicator_position = data.maps[self.map].respawn_points[indicator];
                self.entities.push(CEntity {
                    id,
                    kind: 0,
                    layer: data.revival.marker_layer,
                    owner: index as u32 + 1,
                    flags: E_ACTIVE | E_ENABLED,
                    hierarchy: 100_000_000
                        + hierarchy_base
                        + data.revival.marker_collider_hierarchy,
                    x: indicator_position.x.raw(),
                    y: indicator_position.y.raw(),
                    radius: data.revival.marker_radius.raw(),
                    inverse_mass: data.revival.marker_body.inverse_mass.raw(),
                    ..CEntity::default()
                });
            }

            let body_entity = self.entities.len();
            let body_id = self.next_id();
            let body_collider =
                OrientedBox::from_basis(position, Vec2::RIGHT, data.player.hurtbox_extents);
            self.entities.push(CEntity {
                id: body_id,
                kind: 1,
                layer: LAYER_PLAYER,
                owner: index as u32 + 1,
                flags: E_ACTIVE | E_TRIGGER,
                hierarchy: 100_000_000 + player_hierarchy + 6,
                x: position.x.raw(),
                y: position.y.raw(),
                right_x: body_collider.right.x.raw(),
                right_y: body_collider.right.y.raw(),
                up_x: body_collider.up.x.raw(),
                up_y: body_collider.up.y.raw(),
                inverse_mass: -Fix::ONE.raw(),
                ..CEntity::default()
            });
            let mut dash_entity = usize::MAX;
            let mut bow_entity = usize::MAX;
            let mut throw_entity = usize::MAX;
            let mut mine_entity = usize::MAX;
            let mut spike_entity = usize::MAX;
            for (slot, &hierarchy) in ability_hierarchy.iter().enumerate() {
                let (extents, relative_hierarchy, target) = match abilities[slot] {
                    ABILITY_DASH => (data.dash.hurtbox_extents, 7, &mut dash_entity),
                    ABILITY_BOW => (data.bow.hurtbox_extents, 7, &mut bow_entity),
                    ability
                    @ (ABILITY_GRENADE | ABILITY_SMOKE | ABILITY_TACK | ABILITY_MJOLNIR) => (
                        throw_config(data, ability).hurtbox_extents,
                        8,
                        &mut throw_entity,
                    ),
                    ABILITY_MISSILE => (data.missile.hurtbox_extents, 7, &mut throw_entity),
                    ABILITY_MINE => (
                        data::surface_extents(data.mine.hurtbox_extents),
                        7,
                        &mut mine_entity,
                    ),
                    ABILITY_SPIKE => (
                        data.spike.hurtbox_extents,
                        data.spike.collider_hierarchy,
                        &mut spike_entity,
                    ),
                    _ => continue,
                };
                let id = self.next_id();
                *target = self.entities.len();
                let collider = OrientedBox::from_basis(Vec2::ZERO, Vec2::RIGHT, extents);
                self.entities.push(CEntity {
                    id,
                    kind: 1,
                    layer: LAYER_PLAYER,
                    flags: E_ENABLED | E_TRIGGER,
                    hierarchy: 100_000_000 + hierarchy + relative_hierarchy,
                    x: if abilities[slot] == ABILITY_SPIKE {
                        data.spike.collider_offset.x.raw()
                    } else if abilities[slot] == ABILITY_MINE {
                        data.mine.collider_offset.x.raw()
                    } else {
                        0
                    },
                    y: if abilities[slot] == ABILITY_SPIKE {
                        data.spike.collider_offset.y.raw()
                    } else if abilities[slot] == ABILITY_MINE {
                        data.mine.collider_offset.y.raw()
                    } else {
                        0
                    },
                    right_x: collider.right.x.raw(),
                    right_y: collider.right.y.raw(),
                    up_x: collider.up.x.raw(),
                    up_y: collider.up.y.raw(),
                    inverse_mass: -Fix::ONE.raw(),
                    ..CEntity::default()
                });
            }

            let slot = &mut self.players[index];
            slot.base.sync_collider();
            slot.active = ActiveAbility::None;
            slot.missile = MissileFormState::default();
            slot.throw_item = ThrowItemState::default();
            slot.cast_spell = CastSpellState::default();
            slot.spike_cast = SpikeCastState::default();
            slot.spike_attacks = [None; 3];
            slot.rewind = RewindState::default();
            slot.body_entity = body_entity;
            slot.dash_entity = dash_entity;
            slot.bow_entity = bow_entity;
            slot.throw_entity = throw_entity;
            slot.mine_entity = mine_entity;
            slot.spike_entity = spike_entity;
            slot.circle_entity = platform_entity;
            slot.gust_entity = gust_entity;
            slot.revival_entity = revival_entity;
            slot.time_stop_entity = time_stop_entity;
            slot.stock_indicator_entities = indicator_entities;
            slot.stock_clone_pending = false;
        }
    }

    fn revival_effects_tick(&mut self, data: &data::RolloutData) {
        for index in 0..self.players.len() {
            if self.players[index].revival_clone_pending
                || self.players[index].revival_effect_ticks == 0
            {
                continue;
            }
            self.players[index].revival_effect_ticks -= 1;
            if self.players[index].revival_effect_ticks
                == self.players[index].revival_particle_ticks
            {
                self.next_spawn_hierarchy += data.revival.particle_updatable_count;
            }
            if self.players[index].revival_effect_ticks != 0 {
                continue;
            }
            let marker_index = self.players[index]
                .revival_target_marker
                .take()
                .expect("live Revival effect marker");
            self.end_revival_marker(marker_index);
            let slot = &mut self.players[index];
            slot.revival_particle_ticks = 0;
            slot.respawn_spawn_ticks = data.player.spawn_anim_ticks + 1;
            slot.obs_state = STATE_FALL;
        }
    }

    /// Updater initializes a prefab registered during the preceding pass
    /// before any MonoUpdatable runs. Shockwave.Init captures targets and
    /// performs the curve's first time-zero impulse immediately.
    fn gust_waves_init(&mut self, data: &data::RolloutData) {
        for wave_index in 0..self.gust_waves.len() {
            let mut wave = self.gust_waves[wave_index];
            if !wave.active || wave.registered || wave.born_tick >= self.tick {
                continue;
            }
            wave.registered = true;
            let query = Circle {
                center: wave.position,
                radius: data.gust.radius,
            };
            for (platform, dynamics) in self.platforms.iter().zip(&self.platform_dynamics) {
                if wave.target_count == wave.platform_ids.len() {
                    break;
                }
                if crate::physics::contact::circle_rounded(query, platform.collider)
                    .is_some_and(|contact| contact.penetration > Fix::ZERO)
                    && dynamics.body.inverse_mass >= Fix::ZERO
                {
                    wave.platform_ids[wave.target_count] = platform.id;
                    wave.target_count += 1;
                }
            }
            // Player-layer colliders are processed during activation and are
            // deliberately excluded from the persistent target set.
            for player_index in 0..self.players.len() {
                if player_index == wave_index / 3 || !self.playable(player_index) {
                    continue;
                }
                let target = &mut self.players[player_index];
                if target.active != ActiveAbility::None {
                    continue;
                }
                // CircleCastAll performs the real circle/box narrow phase.
                // A radius-only approximation incorrectly includes the other
                // map-0 spawn in this golden: its centre is within the sum of
                // broad-phase radii, but the wave misses the oriented slime
                // box's corner.
                if crate::physics::contact::circle_box(query, target.base.collider).is_none() {
                    continue;
                }
                let delta = target.base.position - wave.position;
                let distance = (delta.length_squared() / Fix::from_int(10)).max(Fix::from_int(3));
                let force = data.gust.default_force / distance;
                let force = if target.base.grounded {
                    data.gust.grounded_multiplier * delta * force
                } else {
                    delta * force
                };
                target.base.external_velocity += force * Fix::from_int(35);
            }
            self.gust_waves[wave_index] = wave;
            self.gust_wave_apply(wave_index, Fix::ZERO, data);
        }
    }

    fn gust_waves_update(&mut self, data: &data::RolloutData) {
        for wave_index in 0..self.gust_waves.len() {
            if self.gust_waves[wave_index].active && self.gust_waves[wave_index].registered {
                self.gust_wave_apply(wave_index, DT, data);
            }
        }
    }

    fn gust_wave_apply(&mut self, wave_index: usize, dt: Fix, data: &data::RolloutData) {
        let mut wave = self.gust_waves[wave_index];
        let strength = data.gust.curve.evaluate(wave.time);
        wave.time += dt;
        let duration = data.gust.curve.keys.last().expect("Gust force curve").time;
        if wave.time > duration {
            wave.active = false;
            self.gust_waves[wave_index] = wave;
            return;
        }
        let mut applied = false;
        for target in 0..wave.target_count {
            let Some(platform_index) = self
                .platforms
                .iter()
                .position(|platform| platform.id == wave.platform_ids[target])
            else {
                continue;
            };
            let platform = &self.platforms[platform_index];
            let Ok(point) = platform.surface.closest_point(wave.position) else {
                continue;
            };
            let direction = -platform.surface.normal(point);
            let distance = ((wave.position - point).length_squared() / Fix::from_int(10))
                .max(Fix::from_f64(1.5));
            let force = data.gust.platform_force / distance;
            let impulse = direction.normalized_safe() * force * strength * Fix::from_int(60);
            let arm = (point - platform.surface.center).normalized_safe();
            self.platform_dynamics[platform_index]
                .body
                .apply_impulse(impulse, arm);
            applied = true;
        }
        self.gust_waves[wave_index] = wave;
        if applied {
            self.platforms_changed();
        }
    }

    /// base.step with the dash shove check from World::step.
    fn dash_move(&mut self, index: usize, action: Action, movement_input: Vec2) {
        self.players[index]
            .base
            .move_step(action, movement_input, terrain!(self));
        // Dash precedes Ability in the prefab updater order, so the public
        // player position sees Dash's post-update transform on every active
        // pass, including activation and teleport/exit.
        if self.players[index].active == ActiveAbility::Dash {
            let slot = &mut self.players[index];
            slot.reported_position = slot.base.position;
            slot.reported_tick = self.tick;
            if !slot.base.dash.active {
                // Dash.ExitAbility hands over an ungrounded body (Dash.cs:234),
                // so SlimeController.onExitAbility enters Fall and immediately
                // runs VelocityBasedRaycasts (SlimeController.cs:249, 263-272)
                // with the leave velocity: the slime can land on the ejection
                // tick, after the ejected position was published.
                slot.exit_to_fall();
                slot.base.collision_rays(terrain!(self));
            }
        }
        let dash_cast = {
            let caster = &self.players[index].base;
            (caster.dash.active && caster.dash.time == DT).then_some((
                caster.position,
                caster.dash.input,
                caster.id,
            ))
        };
        if let Some((origin, direction, caster_id)) = dash_cast {
            let wall =
                crate::world::closest(terrain!(self), origin, direction, Fix::from_int(13), None)
                    .map(|(_, distance, _)| distance);

            // Dash.performDash raycasts the Projectile layer too. It first
            // calls Arrow.OnCollide with Dash's Player-layer hitbox. Dash has
            // PlayerBody rather than BoplBody, so Arrow returns without
            // landing or destroying itself. PlayerCollision then kills the
            // dasher unless this is an owner-safe young arrow.
            let mut arrow_hits: Vec<_> = self
                .arrows
                .iter()
                .enumerate()
                .filter(|(_, arrow)| arrow.active && arrow.registered)
                .filter_map(|(arrow_index, arrow)| {
                    (Circle {
                        center: arrow.position,
                        radius: data::get().arrow.radius,
                    })
                    .raycast(origin, direction)
                    .filter(|hit| {
                        hit.near <= Fix::from_int(13)
                            && wall.is_none_or(|distance| hit.near < distance)
                    })
                    .map(|hit| (arrow_index, hit.near.max(Fix::ZERO)))
                })
                .collect();
            arrow_hits.sort_by_key(|&(arrow_index, distance)| (distance, arrow_index));
            for (arrow_index, distance) in arrow_hits {
                let arrow = self.arrows[arrow_index];
                let lethal = arrow.owner != caster_id || arrow.can_hurt_owner;
                if lethal {
                    let cause = if arrow.owner == caster_id {
                        CAUSE_ACCIDENT
                    } else {
                        CAUSE_OTHER
                    };
                    // The dasher dies where the ray met the arrow: Dash moves
                    // its body to nearPos before KillPlayerOnCollision and
                    // never restores it on a kill (Dash.cs:177-183).
                    let slot = &mut self.players[index];
                    slot.base.position = origin + direction * distance;
                    slot.reported_position = slot.base.position;
                    self.kill(index, arrow.owner, cause, false);
                    if arrow.owner != caster_id && arrow.owner >= 1 {
                        self.players[arrow.owner as usize - 1].telemetry.arrow_hits += 1;
                    }
                    return;
                }
            }

            let targets: Vec<usize> = self
                .players
                .iter()
                .enumerate()
                .filter(|(i, p)| *i != index && p.base.alive && !p.base.dash.active)
                .filter_map(|(i, p)| {
                    p.base
                        .collider
                        .raycast_within(origin, direction, Fix::from_int(13))
                        .filter(|hit| wall.is_none_or(|distance| hit.near < distance))
                        .map(|_| i)
                })
                .collect();
            for target in targets {
                self.players[target].base.push_from(origin, target < index);
            }
        }
        if !self.players[index].base.dash.active {
            self.players[index].active = ActiveAbility::None;
        }
    }

    /// Shared `ThrowItem2` player-form lifecycle. The item
    /// collider is manually initialized by `Fire`, so it joins DetPhysics in
    /// the cast pass even though its MonoUpdatables initialize next pass.
    fn spawn_guided_missile(&mut self, index: usize, data: &data::RolloutData) -> usize {
        let rotation = self.players[index].base.rotation;
        let right = Vec2::angle(rotation);
        let up = right.perp();
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.missile.spark_updatable_count;
        let mut body = data.missile.spark_body;
        body.orientation = rotation;
        body.velocity = self.players[index].base.up * data.missile.projectile_start_speed;
        let missile = GuidedMissile {
            active: true,
            born_tick: self.tick,
            hierarchy_base,
            owner: index,
            position: self.players[index].base.position
                + data.missile.projectile_offset.complex_mul(right),
            body,
            up,
            ..GuidedMissile::default()
        };
        if let Some(free) = self
            .guided_missiles
            .iter()
            .position(|missile| !missile.active)
        {
            let entity = self.guided_missiles[free].entity;
            self.guided_missiles[free] = GuidedMissile { entity, ..missile };
            free
        } else {
            self.guided_missiles.push(missile);
            self.guided_missiles.len() - 1
        }
    }

    fn missile_form_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let state = &mut self.players[index].missile;
        state.ticks += 1;
        match state.phase {
            0 if state.ticks >= data.missile.enter_ticks => {
                state.phase = 1;
                state.ticks = 0;
                let projectile = self.spawn_guided_missile(index, data);
                self.players[index].missile.projectile = Some(projectile);
            }
            1 if state.ticks > data.missile.enter_to_idle_ticks => {
                state.phase = 2;
                state.ticks = 0;
            }
            3 if state.ticks > data.missile.idle_ticks => {
                let slot = &mut self.players[index];
                slot.active = ActiveAbility::None;
                slot.reported_position = slot.base.position;
                slot.reported_tick = self.tick;
                return;
            }
            _ => {}
        }

        if self.players[index].missile.phase == 2 && action.buttons & 2 == 0 {
            self.players[index].missile.phase = 3;
            self.players[index].missile.ticks = 0;
        }

        let projectile = self.players[index].missile.projectile;
        if self.players[index].missile.phase != 3
            && let Some(projectile) = projectile
            && self.guided_missiles[projectile].active
            && self.guided_missiles[projectile].registered
        {
            let missile = &mut self.guided_missiles[projectile];
            missile.instability_ticks = 0;
            let aim = action.direction();
            let forward = missile.body.velocity.normalized_safe();
            let signed_turn = if forward.cross(aim) > Fix::ZERO {
                DT * data.missile.rotation_speed
            } else {
                -DT * data.missile.rotation_speed
            };
            let mut next_up = Vec2::angle(signed_turn).complex_mul(missile.up);
            if aim.length_squared() < Fix::from_f32(0.4) * Fix::from_f32(0.4) {
                next_up = missile.up;
            }
            missile.body.velocity = next_up * missile.body.velocity.length();
            missile.up = next_up;
            missile.body.orientation = next_up.perp_cw().normalized_angle();
        }

        if self.players[index].base.grounded {
            self.players[index].base.velocity = Vec2::ZERO;
        } else {
            let slot = &mut self.players[index];
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                data.missile.constants.gravity_accel,
                data.missile.constants.gravity_modifier,
                data.missile.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid Missile gravity constants");
            slot.base.collision_rays(terrain!(self));
            if slot.base.grounded {
                slot.active = ActiveAbility::None;
                slot.obs_state = STATE_LAND;
                slot.state_ticks = LAND_TICKS + 2;
                slot.reported_position = slot.base.position;
            } else {
                slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
                slot.base.external_velocity *= Fix::from_f64(0.95);
            }
        }
    }

    /// Original ThrowItem ability used only by Mine. It differs materially
    /// from ThrowItem2: the payload is instantiated only on the animation
    /// callback, its circle auto-registers on the following scheduler pass,
    /// and grounded throws are forced along the attached surface normal.
    fn mine_throw_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let button = 2u8 << self.players[index].active_slot;
        let released = action.buttons & button == 0;
        let aim = action.direction();
        if released {
            self.players[index].throw_item.has_inputted_fire = true;
        }

        self.players[index].reported_position = self.players[index].base.position;
        self.players[index].reported_tick = self.tick;
        let state = &mut self.players[index].throw_item;
        state.ticks += 1;
        state.time_passed += DT;

        let mut fire = false;
        if state.phase == 0 {
            let enter_ticks = if state.grounded {
                data.mine.ground_enter_ticks
            } else {
                data.mine.air_enter_ticks
            };
            if state.ticks > enter_ticks {
                if state.grounded {
                    state.phase = 1;
                    state.ticks = 0;
                    fire = state.has_inputted_fire;
                } else {
                    fire = true;
                }
            }
        } else if state.phase == 1 {
            fire = state.has_inputted_fire || state.ticks > data.mine.ground_loop_ticks;
        }
        if fire {
            self.fire_mine(index, data);
        }

        if self.players[index].throw_item.phase == 2 {
            let exit_ticks = if self.players[index].throw_item.grounded {
                data.mine.ground_exit_ticks
            } else {
                data.mine.air_exit_ticks
            };
            if self.players[index].throw_item.ticks > exit_ticks {
                let grounded = self.players[index].throw_item.grounded;
                let slot = &mut self.players[index];
                slot.active = ActiveAbility::None;
                if grounded {
                    // PlayerPhysics.SyncPhysicsTo(AbilityExitInfo) discards
                    // selfImposedVelocity on a valid grounded attachment.
                    slot.base.velocity = Vec2::ZERO;
                    slot.obs_state = STATE_LAND;
                    slot.state_ticks = LAND_TICKS + 2;
                } else {
                    slot.exit_to_fall();
                    slot.base.rotation = Fix::ZERO;
                    slot.base.up = Vec2::UP;
                    slot.base.collision_rays(terrain!(self));
                }
                slot.reported_position = slot.base.position;
                slot.reported_tick = self.tick;
                return;
            }
        }

        if self.players[index].throw_item.grounded {
            let slot = &mut self.players[index];
            slot.base.velocity = Vec2::ZERO;
            slot.base.position += slot.base.external_velocity * DT;
            slot.base.external_velocity *= Fix::from_f64(0.95);
        } else {
            let this = &mut *self;
            let slot = &mut this.players[index];
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                data.mine.constants.gravity_accel,
                Fix::ONE,
                data.mine.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid Mine gravity constants");
            crate::player_physics::air_movement(
                &mut slot.base.velocity,
                aim.x,
                data.mine.constants.air_accel,
                data.mine.constants.max_speed,
            )
            .expect("valid Mine air constants");
            slot.base.collision_rays(Terrain {
                platforms: &this.platforms,
                bounds: &this.bounds,
            });
            if slot.base.grounded {
                if !slot.throw_item.has_fired {
                    self.fire_mine(index, data);
                }
                let slot = &mut self.players[index];
                slot.active = ActiveAbility::None;
                slot.obs_state = STATE_LAND;
                slot.state_ticks = LAND_TICKS + 2;
                slot.reported_position = slot.base.position;
            } else {
                slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
                slot.base.external_velocity *= Fix::from_f64(0.95);
            }
        }
    }

    fn fire_mine(&mut self, index: usize, data: &data::RolloutData) {
        if self.players[index].throw_item.has_fired {
            return;
        }
        let grounded = self.players[index].throw_item.grounded;
        let up = if grounded {
            self.players[index]
                .base
                .ground
                .and_then(|ground| self.platforms.iter().find(|platform| platform.id == ground))
                .map_or(self.players[index].base.up, |platform| {
                    platform.surface.normal(self.players[index].base.position)
                })
        } else {
            Vec2::UP
        };
        let side = up.perp_cw()
            * if self.players[index].facing_right {
                -Fix::ONE
            } else {
                Fix::ONE
            };
        let position = self.players[index].base.position
            + side * data.mine.throw_position.x
            + up * data.mine.throw_position.y;
        let velocity = if grounded {
            // onlyThrowUp is authored true. Retain the branch so extraction
            // guards a future source update rather than silently changing it.
            if data.mine.only_throw_up {
                up * data.mine.throw_strength
            } else {
                up * data.mine.min_throw_strength
            }
        } else {
            Vec2::UP * data.mine.aerial_throw_strength
        };
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.mine.item_updatable_count;
        let item = ThrownItem {
            active: true,
            id: 0,
            owner: index,
            ability: ABILITY_MINE,
            born_tick: self.tick,
            hierarchy_base,
            entity: None,
            position,
            body: Body {
                velocity,
                ..data.mine.item_body
            },
            fuse: Fix::MAX,
            hurt_owner_delay: Fix::ZERO,
            detonates_on_owner: false,
            age: Fix::ZERO,
            launched: true,
            settled: false,
            owner_clear: false,
            settle_position: Vec2::ZERO,
            phase: 0,
            killed_players: 0,
            random_offset: Fix::ZERO,
            time_since_anticipation: Fix::ZERO,
            prev_chase_dir: Vec2::RIGHT,
            prev_rand_vector: Vec2::ZERO,
            prev_velocity_change: Vec2::ZERO,
        };
        let item_index = if let Some(free) = self.thrown_items.iter().position(|item| !item.active)
        {
            let entity = self.thrown_items[free].entity;
            self.thrown_items[free] = ThrownItem { entity, ..item };
            free
        } else {
            self.thrown_items.push(item);
            self.thrown_items.len() - 1
        };
        let slot = &mut self.players[index];
        slot.throw_item.has_fired = true;
        slot.throw_item.item = Some(item_index);
        slot.throw_item.phase = 2;
        slot.throw_item.ticks = 0;
        if grounded {
            slot.base.velocity = Vec2::ZERO;
        }
    }

    fn throw_item_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let ability = active_throw_ability(self.players[index].active);
        let gd = throw_config(data, ability);
        let aim = action.direction();
        // Ability.UpdateSim publishes Player.Position before ThrowItem2 and
        // PlayerBody advance the live ability body later in this pass.
        self.players[index].reported_position = self.players[index].base.position;
        self.players[index].reported_tick = self.tick;
        if aim != Vec2::ZERO && !self.players[index].throw_item.has_inputted_fire {
            let slot = &mut self.players[index];
            slot.throw_item.direction = aim;
            if slot.throw_item.grounded {
                let normal = slot
                    .base
                    .ground
                    .and_then(|ground| self.platforms.iter().find(|platform| platform.id == ground))
                    .map_or(slot.base.up, |platform| {
                        platform.surface.normal(slot.base.position)
                    });
                slot.facing_right = normal.cross(aim.normalized_safe()) < Fix::ZERO;
            } else {
                slot.facing_right = aim.x > Fix::ZERO;
            }
        }

        let button = 2u8 << self.players[index].active_slot;
        if !self.players[index].throw_item.has_inputted_fire && action.buttons & button == 0 {
            let state = &mut self.players[index].throw_item;
            state.has_inputted_fire = true;
            state.fire_input_time = state.time_passed;
        }

        let state = &mut self.players[index].throw_item;
        state.ticks += 1;
        state.time_passed += DT;
        if state.phase == 0 {
            let enter_ticks = if state.grounded {
                gd.ground_enter_ticks
            } else {
                gd.air_enter_ticks
            };
            if state.ticks > enter_ticks {
                if state.has_inputted_fire {
                    self.fire_thrown_item(index, data);
                } else if state.item.is_none() {
                    // The held grenade has no registered collider because its
                    // DPhysicsCircle uses manuallyCallInit. Keep only the
                    // payload state until release calls `Fire`.
                    let item = self.spawn_held_item(index, data);
                    self.players[index].throw_item.item = Some(item);
                }
            }
        }

        if self.players[index].throw_item.phase == 0
            && self.players[index].throw_item.item.is_some()
            && self.players[index].throw_item.has_inputted_fire
        {
            self.fire_thrown_item(index, data);
        }

        if self.players[index].throw_item.phase == 0
            && let Some(item_index) = self.players[index].throw_item.item
            && self.thrown_items[item_index].active
        {
            let slot = &self.players[index];
            let local_sign = if slot.facing_right {
                -Fix::ONE
            } else {
                Fix::ONE
            };
            let mut dummy = if slot.throw_item.grounded {
                *gd.ground_dummy.last().expect("Grenade ground dummy")
            } else {
                *gd.air_dummy.last().expect("Grenade air dummy")
            };
            // Unlike Fire's authored vector-wide mirror, dummyPos flips only
            // the child's local x coordinate.
            dummy.x *= local_sign;
            self.thrown_items[item_index].position =
                slot.base.position + dummy.complex_mul(Vec2::angle(slot.base.rotation));
        }

        if self.players[index].throw_item.phase == 1 {
            let exit_ticks = if self.players[index].throw_item.grounded {
                gd.ground_exit_ticks
            } else {
                gd.air_exit_ticks
            };
            if self.players[index].throw_item.ticks > exit_ticks {
                let slot = &mut self.players[index];
                slot.active = ActiveAbility::None;
                if slot.base.grounded {
                    // ThrowItem2 marks a grounded exit as `justlanded`.
                    // SlimeController starts `land` after the slime's own
                    // animator has already updated in this pass.
                    slot.obs_state = STATE_LAND;
                    slot.state_ticks = LAND_TICKS + 2;
                } else {
                    slot.exit_to_fall();
                    slot.base.rotation = Fix::ZERO;
                    slot.base.up = Vec2::UP;
                    slot.base.collision_rays(terrain!(self));
                }
                slot.reported_position = slot.base.position;
                slot.reported_tick = self.tick;
                return;
            }
        }
        if self.players[index].throw_item.grounded {
            // PlayerPhysics.Move(Vec2.zero) consumes the throw recoil while
            // attached to the authored surface. PlayerBody still integrates
            // external velocity afterward, which is observable when a moving
            // platform carried the caster immediately before entry.
            let this = &mut *self;
            let slot = &mut this.players[index];
            slot.base.velocity = Vec2::ZERO;
            slot.base.position += slot.base.external_velocity * DT;
            slot.base.external_velocity *= Fix::from_f64(0.95);
        } else {
            // ThrowItem2 keeps its own PlayerBody live throughout the aerial
            // form. Before release it uses the authored 0.03 gravity modifier
            // and clamps the resulting vector to maxSlowMoSpeed. Fire restores
            // gravity and blends 60% of normal horizontal air movement into
            // the recoil velocity. PlayerBody integrates later in the same
            // updater pass and damps only external velocity.
            let this = &mut *self;
            let slot = &mut this.players[index];
            let terrain = Terrain {
                platforms: &this.platforms,
                bounds: &this.bounds,
            };
            let modifier = if slot.throw_item.has_fired {
                Fix::ONE
            } else {
                Fix::from_f32(0.03)
            };
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                gd.constants.gravity_accel,
                modifier,
                gd.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid Grenade gravity constants");
            if slot.throw_item.has_fired {
                let before = slot.base.velocity;
                crate::player_physics::air_movement(
                    &mut slot.base.velocity,
                    aim.x,
                    gd.constants.air_accel,
                    gd.constants.max_speed,
                )
                .expect("valid Grenade air constants");
                let blend = Fix::from_f32(0.6);
                slot.base.velocity = before * (Fix::ONE - blend) + slot.base.velocity * blend;
            } else if slot.base.velocity.length() > gd.max_slow_mo_speed {
                slot.base.velocity = slot.base.velocity.normalized() * gd.max_slow_mo_speed;
            }
            slot.base.collision_rays(terrain);
            if slot.base.grounded {
                // VelocityBasedRaycasts immediately exits the aerial ability.
                slot.active = ActiveAbility::None;
                slot.obs_state = STATE_LAND;
                slot.state_ticks = LAND_TICKS + 2;
                slot.reported_position = slot.base.position;
            } else {
                slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
                slot.base.external_velocity *= Fix::from_f64(0.95);
            }
        }
    }

    fn spawn_held_item(&mut self, index: usize, data: &data::RolloutData) -> usize {
        let ability = active_throw_ability(self.players[index].active);
        let config = throw_config(data, ability);
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += config.item_updatable_count;
        let item = ThrownItem {
            active: true,
            // Stable collider IDs are assigned when ManualInit joins
            // DetPhysics. A held grenade can explode without consuming one.
            id: 0,
            owner: index,
            ability,
            born_tick: self.tick,
            hierarchy_base,
            entity: None,
            position: self.players[index].base.position,
            body: config.item_body,
            fuse: match ability {
                ABILITY_GRENADE => data.grenade.detonation_time,
                ABILITY_SMOKE => data.smoke.detonation_time,
                _ => Fix::MAX,
            },
            hurt_owner_delay: Fix::ZERO,
            detonates_on_owner: false,
            age: Fix::ZERO,
            launched: false,
            settled: false,
            owner_clear: false,
            settle_position: Vec2::ZERO,
            phase: 0,
            killed_players: 0,
            random_offset: Fix::ZERO,
            time_since_anticipation: Fix::ZERO,
            prev_chase_dir: Vec2::RIGHT,
            prev_rand_vector: Vec2::ZERO,
            prev_velocity_change: Vec2::ZERO,
        };
        if let Some(free) = self.thrown_items.iter().position(|item| !item.active) {
            let entity = self.thrown_items[free].entity;
            self.thrown_items[free] = ThrownItem { entity, ..item };
            free
        } else {
            self.thrown_items.push(item);
            self.thrown_items.len() - 1
        }
    }

    fn fire_thrown_item(&mut self, index: usize, data: &data::RolloutData) {
        let ability = active_throw_ability(self.players[index].active);
        let gd = throw_config(data, ability);
        let item_index = self.players[index]
            .throw_item
            .item
            .unwrap_or_else(|| self.spawn_held_item(index, data));
        let new_collider_id = (self.thrown_items[item_index].active
            && self.thrown_items[item_index].entity.is_none())
        .then(|| self.next_id());
        let slot = &mut self.players[index];
        let state = &mut slot.throw_item;
        let direction = state.direction;
        let local_sign = if slot.facing_right {
            -Fix::ONE
        } else {
            Fix::ONE
        };
        let local = if state.grounded {
            gd.ground_position
        } else {
            gd.air_position
        } * local_sign;
        let basis = Vec2::angle(slot.base.rotation);
        let position = slot.base.position + local.complex_mul(basis);
        let charge = (state.fire_input_time * gd.force_gain_speed)
            .max(Fix::ZERO)
            .min(Fix::ONE);
        let force = gd.min_force + (gd.max_force - gd.min_force) * charge;
        let impulse_force = direction * force * Fix::from_int(60);
        if self.thrown_items[item_index].active {
            let item = &mut self.thrown_items[item_index];
            item.position = position;
            item.body.orientation = slot.base.rotation;
            item.body.force += impulse_force;
            item.hurt_owner_delay = gd.hurt_owner_delay;
            if item.entity.is_none() {
                item.id = new_collider_id.expect("new grenade collider id");
                self.entities.push(CEntity::default());
                item.entity = Some(self.entities.len() - 1);
            }
            let entity = item.entity.expect("grenade collider allocated");
            self.entities[entity] = CEntity {
                id: item.id,
                kind: 0,
                layer: gd.item_layer,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000 + item.hierarchy_base + gd.item_collider_hierarchy,
                radius: gd.item_radius.raw(),
                inverse_mass: item.body.inverse_mass.raw(),
                ..CEntity::default()
            };
        }
        slot.base.velocity -= gd.knockback * direction;
        state.has_fired = true;
        state.item = Some(item_index);
        state.phase = 1;
        state.ticks = 0;
    }

    /// BowTransform.OldUpdate + UpdateSim for one tick: aim tracking, the
    /// loading-frame charge schedule, firing on release, slow-mo aiming
    /// physics, post-fire recoil flight, and the exit conditions.
    fn bow_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let bow = &data.bow;
        // SpriteAnimator (hierarchy 100000016) updates before BowTransform
        // (100000021). The fireGround/fireAir callback runs on the update
        // after the animation's last frame tick (SpriteAnimator.cs:190-231)
        // and exits the ability before this pass's aim and movement.
        let b = &mut self.players[index].bow;
        if b.has_fired {
            if b.fire_ticks_left == 0 {
                bow_exit(&mut self.players[index], terrain!(self));
                return;
            }
            b.fire_ticks_left -= 1;
        }
        let slot_button = 2u8 << self.players[index].active_slot;
        let button_down = action.buttons & slot_button != 0;
        let dir = action.direction();
        let fire_now = {
            let b = &mut self.players[index].bow;
            b.ticks += 1;
            // OldUpdate: the aim follows the stick until release (deadzone:
            // the direction table only emits unit vectors or zero).
            if dir != Vec2::ZERO && !b.released {
                b.aim = dir.normalized();
            }
            if !button_down && !b.has_fired {
                b.released = true;
            }
            // loadingFrame = getNumberOfPassedSprites() of the "load" anim =
            // currentSpriteIndex+1. SpriteAnimator runs after BowTransform, so
            // the Loading callback which ends "transform" is first visible to
            // BowTransform on the following pass. Later frame changes likewise
            // require currentSpritePos to exceed their authored duration.
            // Measured .NET boundaries: frames 1/2/3/4 at ability ticks
            // 31/39/46/53.
            if !b.has_fired && b.ticks > bow.transform_ticks {
                let since = b.ticks - bow.transform_ticks;
                let passed = bow
                    .load_frame_ticks
                    .iter()
                    .filter(|t| since > **t + 1)
                    .count() as u32;
                b.loading_frame = (1 + passed).min(4);
            }
            // "…|| animator.getNumberOfPassedSprites() > 2" during transform:
            // its 6 uniform sprites pass index 2 one tick after the second
            // boundary (tick 11 of the 30-tick animation).
            let early = b.ticks > bow.transform_ticks / 3;
            b.released && !b.has_fired && (b.loading_frame > 0 || early)
        };
        if fire_now {
            self.fire_bow(index, data);
        }

        // Movement (UpdateSim).
        let mut exit = false;
        let this = &mut *self;
        let slot = &mut this.players[index];
        let terrain = Terrain {
            platforms: &this.platforms,
            bounds: &this.bounds,
        };
        let platforms: &[Platform] = terrain.platforms;
        // Bow's boundary component runs after BowTransform but before
        // PlayerBody performs this pass's position integration; its
        // ExitAbility deactivates the Bow, so the body is never integrated.
        // BowTransform.ExitAbility ungrounds the body first
        // (BowTransform.cs:120), so a grounded Bow dies 0.05 off its
        // surface, and that is the position the slime takes over
        // (BowTransform.cs:124, SlimeController.cs:217).
        if slot.base.outside_bounds() {
            slot.base.alive = false; // game_tick attributes it after this updater
            slot.base.force_unground(platforms);
            slot.reported_position = slot.base.position;
            return;
        }
        if slot.base.grounded {
            if slot.bow.has_fired {
                // Landed after firing: exit with the landing (justlanded).
                exit = true;
            } else {
                // Grounded aiming: physics.Move(zero) = friction, stay attached.
                slot.base.step(Action::default(), terrain);
            }
        } else if !slot.bow.has_fired {
            // Airborne aiming: near-frozen fall (gravity_modifier 0.03), the
            // landing raycast on the unclamped velocity (BowTransform.cs:222),
            // then Vec2.Normalized(v) * maxSlowMoSpeed (BowTransform.cs:249).
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                bow.constants.gravity_accel,
                Fix::from_f32(0.03),
                bow.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid bow gravity constants");
            slot.base.collision_rays(terrain);
            if slot.base.grounded {
                // Landed while aiming: physics.Move(zero) in the same pass
                // (BowTransform.cs:240-243); the shot later exits on landing.
                slot.base.move_only(Vec2::ZERO, terrain);
            } else if slot.base.velocity.length() > bow.max_slow_mo_speed {
                slot.base.velocity = slot.base.velocity.normalized() * bow.max_slow_mo_speed;
            }
            slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
            slot.base.external_velocity *= Fix::from_f64(0.95);
        } else {
            // Post-fire flight. BowTransform raycasts immediately after full
            // gravity, before its later aerial-control blend. This matters
            // for grounded shots: the downward recoil reattaches and exits
            // before horizontal aim can skew the landing ray.
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                bow.constants.gravity_accel,
                Fix::ONE,
                bow.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid bow gravity constants");
            slot.base.collision_rays(terrain);
            if slot.base.grounded {
                exit = true;
            } else {
                // The modified velocity is then consumed by PlayerBody's
                // later integration in the same updater pass.
                let before = slot.base.velocity;
                crate::player_physics::air_movement(
                    &mut slot.base.velocity,
                    dir.x,
                    bow.constants.air_accel,
                    bow.constants.max_speed,
                )
                .expect("valid bow air constants");
                // Vec2.Lerp = a*(1-t) + b*t per component (Fix.cs:41608),
                // then Vec2.Normalized(v) * Speed (BowTransform.cs:262).
                let after = slot.base.velocity;
                let blend = bow.aireal_control_blend;
                slot.base.velocity = before * (Fix::ONE - blend) + after * blend;
                if slot.base.velocity.length() > bow.constants.max_speed {
                    slot.base.velocity = slot.base.velocity.normalized() * bow.constants.max_speed;
                }
                slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
                slot.base.external_velocity *= Fix::from_f64(0.95);
            }
        }
        if exit {
            bow_exit(slot, terrain);
        }
    }

    /// BowTransform.Fire + Shoot: spawn the arrow (pre-knockback velocity),
    /// then recoil the shooter.
    fn fire_bow(&mut self, index: usize, data: &data::RolloutData) {
        let bow = &data.bow;
        let (position, aim, loading_frame, velocity_before, grounded, owner) = {
            let slot = &self.players[index];
            (
                slot.base.position,
                slot.bow.aim.normalized(),
                slot.bow.loading_frame,
                slot.base.velocity,
                slot.base.grounded,
                slot.base.id,
            )
        };
        let speed = Fix::from_int(i64::from(loading_frame) + 1) * bow.arrow_speed;
        let start_velocity = aim * speed + velocity_before;
        let magnitude = start_velocity.length();
        let hurt_owner_after = if magnitude > Fix::ZERO {
            bow.time_before_arrows_hurt_owner / magnitude
        } else {
            Fix::ZERO
        };
        self.spawn_arrow(
            owner,
            position,
            start_velocity,
            hurt_owner_after,
            bow.max_arrows,
        );
        self.players[index].telemetry.arrows_fired += 1;
        let this = &mut *self;
        let slot = &mut this.players[index];
        let platforms: &[Platform] = &this.platforms;
        slot.bow.has_fired = true;
        if grounded {
            // selfImposedVelocity = -groundNormal * KnockBack, then UnGround:
            // the shooter slams onto the platform and exits on landing.
            slot.base.velocity = -(slot.base.up * bow.knockback);
            slot.base.force_unground(platforms);
            slot.bow.fire_ticks_left = bow.fire_ground_ticks;
        } else {
            slot.base.velocity = -(aim * bow.knockback);
            slot.bow.fire_ticks_left = bow.fire_air_ticks;
        }
    }

    /// PlatformTransform.UpdateSim: anticipation, rotation while the enter
    /// animation scales the radius in, cursor-driven sizing under the area
    /// cap, and the ejection along the platform's up vector.
    fn platform_ability_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let pd = &data.platform;
        let slot_button = 2u8 << self.players[index].active_slot;
        let button_down = action.buttons & slot_button != 0;
        let aim = action.direction();
        self.players[index].platform.ticks += 1;
        if !button_down && !self.players[index].platform.released {
            self.players[index].platform.released = true;
        }
        let phase = self.players[index].platform.phase;
        match phase {
            PLATFORM_ANTICIPATION => {
                if self.players[index].platform.ticks >= pd.anticipation_ticks {
                    // CreatePlatform: spawn at the frozen player position.
                    let position = self.players[index].base.position;
                    let platform_index = self.spawn_ability_platform(index, position);
                    let st = &mut self.players[index].platform;
                    st.platform_index = platform_index;
                    st.phase = PLATFORM_ROTATION;
                    st.time_passed = Fix::ZERO;
                    st.cursor = position + Vec2::UP * pd.spawn_radius;
                    st.hurtbox_disabled = true;
                    // body.rotation = 0 on entry (PlatformTransform.cs:122).
                    st.sizing_right = Vec2::RIGHT;
                }
            }
            PLATFORM_ROTATION => {
                let st = &mut self.players[index].platform;
                st.time_passed += DT;
                let time_passed = st.time_passed;
                let platform_index = st.platform_index;
                let ticks = st.ticks;
                let radius = pd.enter_scale_curve.evaluate(time_passed);
                let (center, mut up) = {
                    let s = &self.platforms[platform_index].surface;
                    (s.center, s.up)
                };
                // With the stick past the deadzone, PlatformTransform writes
                // the aimed `up` into its own and the spawned platform's
                // FixTransform and the spring's home
                // (PlatformTransform.cs:276-285); DetPhysics.Simulate copies
                // that scalar into the body before integrating
                // (DetPhysics.cs:2504-2506). Otherwise the player's rotation
                // keeps its previous scalar (0 from EnterAbility,
                // PlatformTransform.cs:122) and the platform's FixTransform
                // still holds the orientation SyncGameObject wrote after the
                // last pass, so the body integrates undisturbed. Resizing only
                // changes the collider geometry.
                if aim != Vec2::ZERO {
                    up = slerp(up, aim, DT * pd.rotation_speed);
                    let rotation = up.perp_cw().normalized_angle();
                    self.platform_dynamics[platform_index].transform_rotation = Some(rotation);
                    self.platform_dynamics[platform_index].home_rotation = rotation;
                    self.platforms[platform_index].surface.right = up.perp_cw();
                    self.platforms[platform_index].surface.up = up;
                    self.platforms_changed();
                    self.players[index].base.rotation = rotation;
                    // FixTransform.up stores the exact Slerp axes
                    // (FixTransform.cs:49-59); State.Size sizes along them.
                    self.players[index].platform.sizing_right = up.perp_cw();
                }
                self.reshape_platform(platform_index, center, pd.spawn_extents, radius);
                let slot = &mut self.players[index];
                slot.base.position = center;
                // SpriteAnimator invokes EnterSize after PlatformTransform's
                // UpdateSim pass. The newly selected Size state therefore
                // starts on the following tick, one pass after the authored
                // Enter animation has accumulated all of its frame ticks.
                if ticks > pd.anticipation_ticks + pd.enter_ticks {
                    slot.platform.phase = PLATFORM_SIZE;
                }
            }
            PLATFORM_SIZE => {
                self.players[index].platform.time_passed += DT;
                self.platform_size_tick(index, aim, pd);
                let st = &mut self.players[index].platform;
                if st.released || st.time_passed > pd.max_size_time {
                    st.phase = PLATFORM_EXIT;
                    st.exit_time = st.time_passed;
                }
            }
            _ => {
                let st = &mut self.players[index].platform;
                st.time_passed += DT;
                let (time_passed, exit_time, platform_index) =
                    (st.time_passed, st.exit_time, st.platform_index);
                let (center, up, span) = {
                    let s = &self.platforms[platform_index].surface;
                    (s.center, s.up, s.radius + s.extents.y)
                };
                let eject = pd
                    .ejection_curve
                    .evaluate((time_passed - exit_time) / pd.time_in_exit)
                    * span;
                let slot = &mut self.players[index];
                slot.base.position = center + up * eject;
                if eject + pd.face_height >= span {
                    slot.base.position += up * pd.ejection_offset;
                    slot.base.rotation = Fix::ZERO;
                    slot.base.grounded = false;
                    slot.base.ground = None;
                    slot.base.grounded_speed = Fix::ZERO;
                    slot.active = ActiveAbility::None;
                    slot.exit_to_fall();
                    // Ability.ExitAbility republishes potentialCollider.position
                    // after PlatformTransform has applied the final ejection.
                    // SlimeController's synchronous onExitAbility callback then
                    // enters Fall and immediately performs
                    // VelocityBasedRaycasts. Preserve both phase boundaries:
                    // this tick observes the ejected position, while the live
                    // body may already have reattached for the next tick.
                    slot.reported_position = slot.base.position;
                    slot.reported_tick = self.tick;
                    slot.base.collision_rays(terrain!(self));
                }
            }
        }
    }

    /// The State.Size cursor sizing, ported from PlatformTransform (active
    /// corner selection, per-axis growth, and the maxArea cap with its
    /// quadratic clamp when both axes grow).
    fn platform_size_tick(&mut self, index: usize, aim: Vec2, pd: &data::PlatformAbilityData) {
        let platform_index = self.players[index].platform.platform_index;
        let (center, extents, radius) = {
            let s = &self.platforms[platform_index].surface;
            (s.center, s.extents, s.radius)
        };
        let (x, y) = (extents.x, extents.y);
        // PlatformTransform.State.Size uses the transformed player's stable
        // `body.right` (the exact Slerp vector, FixTransform.cs:49-59), not
        // the potentially wobbling platform collider axes. The cursor is
        // rotated into that frame with `new Vec2(-body.rotation)`
        // (PlatformTransform.cs:294): trig of the scalar, not the conjugate.
        let basis = self.players[index].platform.sizing_right;
        let conj = Vec2::angle(-self.players[index].base.rotation);
        // The cursor is measured from the ability's own fixTrans.position
        // (PlatformTransform.cs:294-300): the platform centre as it was
        // before the previous DetPhysics pass moved it, not the live one.
        let body_position = self.players[index].base.position;
        let st = &mut self.players[index].platform;
        st.cursor += aim * CURSOR_SENSITIVITY * DT;
        let mut local = (st.cursor - body_position).complex_mul(conj);
        local.x = local.x.max(-x - radius).min(x + radius);
        local.y = local.y.max(-y - radius).min(y + radius);
        st.cursor = body_position + local.complex_mul(basis);
        let reach = pd.min_size + radius * Fix::from_int(2);
        if local.x >= x && x <= pd.min_size {
            st.active_corner.x = Fix::ONE;
        } else if local.x < -x && x <= pd.min_size {
            st.active_corner.x = -Fix::ONE;
        }
        if local.y >= y && y <= pd.min_size {
            st.active_corner.y = Fix::ONE;
        } else if local.y < -y && y <= pd.min_size {
            st.active_corner.y = -Fix::ONE;
        }
        let corner = Vec2::new(st.active_corner.x * x, st.active_corner.y * y);
        let diff = local - corner;
        let capped = Vec2::new(diff.x.min(reach), diff.y.min(reach));
        let dir = Vec2::new(
            capped.x * Fix::from_int(corner.x.sign()),
            capped.y * Fix::from_int(corner.y.sign()),
        );
        let mut a = (x + dir.x * DT * pd.size_change_speed).max(pd.min_size);
        let mut a2 = (y + dir.y * DT * pd.size_change_speed).max(pd.min_size);
        let cap = pd.max_area;
        if dir.x >= Fix::ZERO && dir.y >= Fix::ZERO {
            // `vec8.x *= fix * SizeChangeSpeed`: the scalar product first
            // (PlatformTransform.cs:335-336), unlike `a` above.
            let gx = dir.x * (DT * pd.size_change_speed);
            let gy = dir.y * (DT * pd.size_change_speed);
            let grow = gx.min(gy);
            let skew = gx - gy;
            if skew > Fix::ZERO {
                let sized = new_size(x + skew, y, radius, cap, pd.min_size);
                a = sized.0;
                a2 = sized.1;
            } else if skew < Fix::ZERO {
                let sized = new_size(y - skew, x, radius, cap, pd.min_size);
                a2 = sized.0;
                a = sized.1;
            } else {
                a = x;
                a2 = y;
            }
            if platform_area(a + grow, a2 + grow, radius) > cap {
                let quarter = Fix::from_f64(0.25);
                let c0 = (Fix::from_int(4) * a2 * radius
                    + Fix::from_int(4) * a * radius
                    + Fix::from_int(4) * a2 * a
                    + Fix::PI * radius * radius
                    - cap)
                    * quarter;
                let c1 = (Fix::from_int(8) * radius + Fix::from_int(4) * a2 + Fix::from_int(4) * a)
                    * quarter;
                let disc = (c1 * c1 * quarter - c0).sqrt().unwrap_or(Fix::ZERO);
                // `-fix11 * 0.5f`: negate, then scale (PlatformTransform.cs:354).
                let root = (-c1) * Fix::from_f64(0.5) + disc;
                a2 += root;
                a += root;
            } else {
                a += grow;
                a2 += grow;
            }
        } else if dir.x > Fix::ZERO {
            let sized = new_size(a, a2, radius, cap, pd.min_size);
            a = sized.0;
            a2 = sized.1;
        } else if dir.y > Fix::ZERO {
            let sized = new_size(a2, a, radius, cap, pd.min_size);
            a2 = sized.0;
            a = sized.1;
        }
        // Shift the centre so the anchored corner stays put.
        let shift_local = Vec2::new((a - x) * st.active_corner.x, (a2 - y) * st.active_corner.y);
        let shift = shift_local.complex_mul(basis);
        let new_center = center + shift;
        self.reshape_platform(platform_index, new_center, Vec2::new(a, a2), radius);
        self.players[index].base.position = new_center;
    }

    /// Spawn a ResizablePlatform for `owner` (starting removal of the previous one)
    /// and return its index in `platforms`.
    fn spawn_ability_platform(&mut self, owner: usize, position: Vec2) -> usize {
        self.begin_removing_spawned_platform_of(owner);
        let pd = &data::get().platform;
        let entity = self.entities.len();
        let id = self.next_id();
        self.entities.push(CEntity {
            id,
            kind: 2,
            layer: LAYER_WALL,
            owner: 0,
            flags: E_ACTIVE | E_ENABLED,
            hierarchy: 300_000_000 + id,
            inverse_mass: pd.spawn_body.inverse_mass.raw(),
            ..CEntity::default()
        });
        let surface = Surface {
            center: position,
            right: Vec2::RIGHT,
            up: Vec2::UP,
            extents: data::surface_extents(pd.spawn_extents),
            radius: pd.spawn_radius,
        };
        let platform_index = self.platforms.len();
        self.platforms.push(make_platform(
            self.next_platform_id,
            &surface,
            pd.spawn_extents,
        ));
        self.platform_dynamics.push(PlatformDynamics {
            body: pd.spawn_body,
            root_body: pd.spawn_body,
            attached_spikes: 0,
            composite_position: position,
            home_position: position,
            transform_rotation: None,
            init_tick: Some(self.tick + 1),
            home_rotation: Fix::ZERO,
            speed: pd.spawn_speed,
            mu: pd.spawn_mu,
            rotation_speed: pd.spawn_rotation_speed,
            rotation_mu: pd.spawn_rotation_mu,
            basis: None,
        });
        self.next_platform_id += 1;
        self.platform_defs.push(usize::MAX);
        self.platform_entities.push(entity);
        // CreatePlatform immediately calls ResizePlatform with the prefab
        // dimensions after ManualInit. That recomputes mass and inertia even
        // though the geometry itself did not change.
        self.reshape_platform(platform_index, position, pd.spawn_extents, pd.spawn_radius);
        // CreatePlatform explicitly activates a new body's sudden-death mode
        // only when the global transition is already in progress.
        let delay = if self.sudden_death {
            self.rng.fix_range(Fix::ZERO, pd.sudden_death_max_delay)
        } else {
            Fix::ZERO
        };
        self.spawned.push(SpawnedPlatform {
            active: true,
            being_removed: false,
            owner,
            platform_index,
            sudden_death_delay: delay,
            removal_time: Fix::ZERO,
            removal_extents: Vec2::ZERO,
            removal_radius: Fix::ZERO,
        });
        platform_index
    }

    fn begin_removing_spawned_platform_of(&mut self, owner: usize) {
        if let Some(i) = self
            .spawned
            .iter()
            .rposition(|s| s.active && !s.being_removed && s.owner == owner)
        {
            let platform_index = self.spawned[i].platform_index;
            let surface = &self.platforms[platform_index].surface;
            self.spawned[i].being_removed = true;
            self.spawned[i].removal_time = Fix::ZERO;
            self.spawned[i].removal_extents = surface.extents;
            self.spawned[i].removal_radius = surface.radius;
        }
    }

    /// Remove a platform from the world (RemovePlatform / scene-bounds
    /// destruction). Riders retain Unity's destroyed ground reference until
    /// their own next updater pass.
    fn remove_platform(&mut self, index: usize) {
        let id = self.platforms[index].id;
        let surface = self.platforms[index].surface.clone();
        for slot in &mut self.players {
            if slot.base.alive && slot.base.ground == Some(id) {
                slot.removed_ground_normal = Some(surface.normal(slot.base.position));
            }
        }
        let entity = self.platform_entities[index];
        self.entities[entity].flags = E_REMOVED;
        for spike in &mut self.spike_attacks {
            if !spike.active {
                continue;
            }
            if spike.platform_index == index {
                spike.active = false;
                if let Some(entity) = spike.entity {
                    self.entities[entity].flags = E_REMOVED;
                }
            } else if spike.platform_index > index {
                spike.platform_index -= 1;
            }
        }
        self.platforms.remove(index);
        self.platform_dynamics.remove(index);
        self.platform_defs.remove(index);
        self.platform_entities.remove(index);
        self.platforms_changed();
        self.orbits.retain(|orbit| orbit.0 != index);
        for orbit in &mut self.orbits {
            if orbit.0 > index {
                orbit.0 -= 1;
            }
        }
        for s in &mut self.spawned {
            if s.active && s.platform_index > index {
                s.platform_index -= 1;
            }
        }
        for p in &mut self.players {
            if p.active == ActiveAbility::Platform && p.platform.platform_index > index {
                p.platform.platform_index -= 1;
            }
        }
        self.rebuild_bounds();
    }

    /// Record a mutation of `platforms`/`platform_dynamics` (positions,
    /// shapes, bodies, membership) so the derived caches refresh.
    fn platforms_changed(&mut self) {
        self.platform_epoch += 1;
    }

    /// Refresh the raycast broad phase after platforms moved, changed shape
    /// or were added/removed (`world::PlatformBounds`); a no-op while the
    /// platforms are unchanged since the last build.
    fn rebuild_bounds(&mut self) {
        if self.bounds_epoch != self.platform_epoch {
            self.bounds.rebuild(&self.platforms);
            self.bounds_epoch = self.platform_epoch;
        }
    }

    fn reshape_platform(&mut self, index: usize, center: Vec2, extents: Vec2, radius: Fix) {
        let old_center = self.platforms[index].surface.center;
        let platform = &mut self.platforms[index];
        platform.surface.center = center;
        // ResizablePlatform.ResizePlatform ends with StickyRect.UpdateBounds
        // (ResizablePlatform.cs:105): the perimeter reads `One / (One / e)`.
        platform.surface.extents = data::surface_extents(extents);
        platform.surface.radius = radius;
        // Box.UpdateExtents (Box.cs:37-44) rescales the existing axes rather
        // than rebuilding them from the rotation.
        let inner = &mut platform.collider.inner;
        inner.center = center;
        inner.right *= inner.inverse_extents.x;
        inner.up *= inner.inverse_extents.y;
        inner.inverse_extents = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
        inner.right *= extents.x;
        inner.up *= extents.y;
        platform.collider.radius = radius;
        let dynamics = &mut self.platform_dynamics[index];
        // ResizablePlatform.ResizePlatform recomputes these from geometry on
        // every resize; keeping the prefab's mass makes long/thin platforms
        // react far too strongly to contacts.
        // ResizePlatform passes (rHeight, rWidth) into PlatformArea's
        // (rWidth, rHeight) parameters (ResizablePlatform.cs:53).
        let pd = &data::get().platform;
        let mass = Fix::from_int(10)
            + pd.spawn_mass_per_area * platform_area(extents.y, extents.x, radius);
        let w = extents.x + radius;
        let h = extents.y + radius;
        let longest = w.max(h);
        let inertia = w.min(h) * longest * longest * Fix::from_f32(0.2) + Fix::from_int(40);
        dynamics.body.inverse_mass = Fix::ONE / mass;
        dynamics.body.inverse_inertia = Fix::ONE / inertia;
        dynamics.home_position += center - old_center;
        self.platforms_changed();
        self.rebuild_bounds();
    }

    fn spawn_arrow(
        &mut self,
        owner: u32,
        position: Vec2,
        velocity: Vec2,
        hurt_owner_after: Fix,
        max_arrows: u32,
    ) {
        // BowTransform uses a per-owner RingBuffer and destroys its oldest
        // arrow when the authored capacity is reached. Native arrow slots are
        // reused, so age (not vector position) identifies the oldest one.
        let owner_count = self
            .arrows
            .iter()
            .filter(|arrow| arrow.active && arrow.owner == owner)
            .count() as u32;
        if max_arrows > 0
            && owner_count >= max_arrows
            && let Some((oldest, _)) = self
                .arrows
                .iter()
                .enumerate()
                .filter(|(_, arrow)| arrow.active && arrow.owner == owner)
                .max_by_key(|(_, arrow)| arrow.time_passed)
        {
            self.destroy_arrow(oldest);
        }
        let free = self.arrows.iter().position(|a| !a.active);
        let entity = free.and_then(|i| self.arrows[i].entity);
        let id = self.next_id();
        let arrow = Arrow {
            active: true,
            registered: false,
            born_tick: self.tick,
            id,
            owner,
            entity,
            position,
            velocity,
            rotation: Fix::ZERO,
            time_passed: Fix::ZERO,
            hurt_owner_after,
            can_hurt_owner: false,
        };
        match free {
            Some(i) => self.arrows[i] = arrow,
            None => self.arrows.push(arrow),
        }
    }

    /// Updater-phase destruction (bounds, arrow cap): the collider is gone
    /// before this pass's DetPhysics contacts and observation.
    fn destroy_arrow(&mut self, index: usize) {
        let arrow = &mut self.arrows[index];
        arrow.active = false;
        arrow.registered = false;
        if let Some(entity) = arrow.entity {
            self.entities[entity].flags = E_REMOVED;
        }
    }

    fn register_pending_mines(&mut self, data: &data::RolloutData) {
        for index in 0..self.thrown_items.len() {
            let item = self.thrown_items[index];
            if !item.active
                || item.ability != ABILITY_MINE
                || item.entity.is_some()
                || item.born_tick >= self.tick
            {
                continue;
            }
            let id = self.next_id();
            self.entities.push(CEntity::default());
            let entity = self.entities.len() - 1;
            let random_offset =
                Fix::ONE - self.rng.fix_range(Fix::ZERO, Fix::from_int(2)) * data.mine.hover_period;
            let item = &mut self.thrown_items[index];
            item.id = id;
            item.entity = Some(entity);
            item.random_offset = random_offset;
            self.entities[entity] = CEntity {
                id,
                kind: 0,
                layer: data.mine.item_layer,
                // Item does not implement IPlayerIdHolder; the owner is used
                // internally by Mine but its observation owner remains zero.
                owner: 0,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000 + item.hierarchy_base + data.mine.item_collider_hierarchy,
                x: item.position.x.raw(),
                y: item.position.y.raw(),
                vx: item.body.velocity.x.raw(),
                vy: item.body.velocity.y.raw(),
                radius: data.mine.item_radius.raw(),
                rotation: item.body.orientation.raw(),
                angular_velocity: item.body.angular_velocity.raw(),
                inverse_mass: item.body.inverse_mass.raw(),
                ..CEntity::default()
            };
        }
    }

    /// Updater defers objects instantiated during a pass until the next pass.
    /// Register them before player/ability updates so same-tick dash casts can
    /// see the newly live projectile, then let arrows_tick advance them later.
    fn register_pending_arrows(&mut self) {
        let radius = data::get().arrow.radius;
        for i in 0..self.arrows.len() {
            if !self.arrows[i].active
                || self.arrows[i].registered
                || self.arrows[i].born_tick >= self.tick
            {
                continue;
            }
            let entity = match self.arrows[i].entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.arrows[i].entity = Some(entity);
                    entity
                }
            };
            let arrow = &mut self.arrows[i];
            self.entities[entity] = CEntity {
                id: arrow.id,
                kind: 0,
                layer: LAYER_PROJECTILE,
                owner: arrow.owner,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 200_000_000 + arrow.id,
                radius: radius.raw(),
                inverse_mass: Fix::from_f64(0.2).raw(),
                ..CEntity::default()
            };
            arrow.registered = true;
        }
    }

    fn register_pending_guided_missiles(&mut self, data: &data::RolloutData) {
        for missile in &self.guided_missiles {
            if !missile.active
                && let Some(entity) = missile.entity
                && self.entities[entity].flags & E_DESTROYED != 0
            {
                self.entities[entity].flags = E_REMOVED;
            }
        }
        for index in 0..self.guided_missiles.len() {
            let missile = self.guided_missiles[index];
            if !missile.active || missile.registered || missile.born_tick >= self.tick {
                continue;
            }
            let entity = match missile.entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.guided_missiles[index].entity = Some(entity);
                    entity
                }
            };
            let id = self.next_id();
            let collider = OrientedBox::from_basis(
                missile.position,
                Vec2::angle(missile.body.orientation),
                data.missile.spark_extents,
            );
            self.entities[entity] = CEntity {
                id,
                kind: 1,
                layer: data.missile.spark_layer,
                flags: E_ACTIVE | E_ENABLED,
                hierarchy: 100_000_000
                    + missile.hierarchy_base
                    + data.missile.spark_collider_hierarchy,
                x: missile.position.x.raw(),
                y: missile.position.y.raw(),
                vx: missile.body.velocity.x.raw(),
                vy: missile.body.velocity.y.raw(),
                right_x: collider.right.x.raw(),
                right_y: collider.right.y.raw(),
                up_x: collider.up.x.raw(),
                up_y: collider.up.y.raw(),
                rotation: missile.body.orientation.raw(),
                angular_velocity: missile.body.angular_velocity.raw(),
                inverse_mass: missile.body.inverse_mass.raw(),
                ..CEntity::default()
            };
            self.guided_missiles[index].registered = true;
        }
    }

    fn register_pending_missile_explosions(&mut self, data: &data::RolloutData) {
        for index in 0..self.missile_explosions.len() {
            let explosion = self.missile_explosions[index];
            if !explosion.active || explosion.registered || explosion.born_tick >= self.tick {
                continue;
            }
            let entity = match explosion.entity {
                Some(entity) => entity,
                None => {
                    self.entities.push(CEntity::default());
                    let entity = self.entities.len() - 1;
                    self.missile_explosions[index].entity = Some(entity);
                    entity
                }
            };
            let id = self.next_id();
            let collider = OrientedBox::from_basis(
                explosion.position,
                Vec2::RIGHT,
                data.missile.explosion_extents,
            );
            self.entities[entity] = CEntity {
                id,
                kind: 1,
                layer: data.missile.explosion_layer,
                flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
                hierarchy: 100_000_000
                    + explosion.hierarchy_base
                    + data.missile.explosion_collider_hierarchy,
                x: explosion.position.x.raw(),
                y: explosion.position.y.raw(),
                right_x: collider.right.x.raw(),
                right_y: collider.right.y.raw(),
                up_x: collider.up.x.raw(),
                up_y: collider.up.y.raw(),
                inverse_mass: -Fix::ONE.raw(),
                ..CEntity::default()
            };
            self.missile_explosions[index].registered = true;
            self.apply_explosion_shockwave(explosion.position, data.missile.explosion_shockwave);
        }
    }

    fn spawn_missile_explosion(
        &mut self,
        position: Vec2,
        owner: usize,
        credited_owner: bool,
        data: &data::RolloutData,
    ) {
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.missile.explosion_updatable_count;
        let explosion = MissileExplosion {
            active: true,
            born_tick: self.tick,
            hierarchy_base,
            position,
            owner,
            credited_owner,
            ..MissileExplosion::default()
        };
        if let Some(free) = self
            .missile_explosions
            .iter()
            .position(|explosion| !explosion.active)
        {
            let entity = self.missile_explosions[free].entity;
            self.missile_explosions[free] = MissileExplosion {
                entity,
                ..explosion
            };
        } else {
            self.missile_explosions.push(explosion);
        }
    }

    fn destroy_guided_missile(&mut self, index: usize, by_callback: bool) {
        let missile = &mut self.guided_missiles[index];
        missile.active = false;
        missile.registered = false;
        if let Some(entity) = missile.entity {
            self.entities[entity].flags = if by_callback { E_DESTROYED } else { E_REMOVED };
        }
    }

    fn guided_missiles_integrate(&mut self, data: &data::RolloutData, physics_dt: Fix) {
        for index in 0..self.guided_missiles.len() {
            if !self.guided_missiles[index].active || !self.guided_missiles[index].registered {
                continue;
            }
            let position = self.guided_missiles[index].position;
            let mut boundary_explosion = None;
            if position.y < Fix::from_f32(-11.3) {
                self.destroy_guided_missile(index, false);
                continue;
            } else if position.y > Fix::from_f64(72.5) {
                boundary_explosion = Some(Vec2::new(position.x, Fix::from_f64(72.5)));
            } else if position.x < Fix::from_f64(-131.25) {
                boundary_explosion = Some(Vec2::new(Fix::from_f64(-131.25), position.y));
            } else if position.x > Fix::from_f64(131.25) {
                boundary_explosion = Some(Vec2::new(Fix::from_f64(131.25), position.y));
            }
            if let Some(explosion_position) = boundary_explosion {
                let owner = self.guided_missiles[index].owner;
                self.spawn_missile_explosion(explosion_position, owner, false, data);
                self.destroy_guided_missile(index, false);
                continue;
            }

            let missile = &mut self.guided_missiles[index];
            missile.instability_ticks += 1;
            // SparkShot.UpdateSim still advances its instability counter while
            // GameTime is frozen, but AddForce scales by FixedTimeStep. Do not
            // leave thrust queued for the first unfrozen DetPhysics pass.
            if physics_dt > Fix::ZERO {
                if missile.instability_ticks >= 3 {
                    missile.body.force += missile.up * data.missile.instability_accel;
                } else if missile.up.dot(missile.body.velocity) < data.missile.minimum_forward_speed
                {
                    missile.body.force += missile.up * data.missile.accel;
                }
            }
            missile.position += missile.body.integrate(physics_dt, Fix::from_int(-40));
            let entity = missile.entity.expect("registered Missile collider");
            let collider = OrientedBox::from_basis(
                missile.position,
                Vec2::angle(missile.body.orientation),
                data.missile.spark_extents,
            );
            let row = &mut self.entities[entity];
            row.x = missile.position.x.raw();
            row.y = missile.position.y.raw();
            row.vx = missile.body.velocity.x.raw();
            row.vy = missile.body.velocity.y.raw();
            row.right_x = collider.right.x.raw();
            row.right_y = collider.right.y.raw();
            row.up_x = collider.up.x.raw();
            row.up_y = collider.up.y.raw();
            row.rotation = missile.body.orientation.raw();
            row.angular_velocity = missile.body.angular_velocity.raw();
            // DetPhysics writes PhysicsBody.orientation back through the
            // FixTransform rotation setter after simulation, rebuilding the
            // basis that SparkShot reads on the following updater pass.
            missile.up = Vec2::angle(missile.body.orientation).perp();
        }
    }

    /// DetPhysics integrates every BoplBody before constructing contact
    /// manifolds. Keep projectile time/gravity in that body phase; contacts
    /// run after platform integration below.
    fn arrows_integrate(&mut self, data: &data::RolloutData, updater_dt: Fix, physics_dt: Fix) {
        // Scene DetPhysics overrides gravityAccel to -40 on every map (the
        // -9.82 field default never applies).
        let gravity = Fix::from_f64(-40.0) * data.arrow.body.gravity_scale;
        for i in 0..self.arrows.len() {
            if !self.arrows[i].active || !self.arrows[i].registered {
                if let Some(entity) = self.arrows[i].entity {
                    self.entities[entity].flags = E_REMOVED;
                }
                continue;
            }
            // DestroyIfOutsideSceneBounds (Arrow prefab hierarchy 1) runs
            // before Arrow.UpdateSim (hierarchy 2) and DetPhysics. Its
            // NeverDestroyAbove flag leaves top exits alive.
            let position = self.arrows[i].position;
            if position.y < Fix::from_f32(-11.3) || position.x.abs() > Fix::from_f64(131.25) {
                self.destroy_arrow(i);
                continue;
            }
            let arrow = &mut self.arrows[i];
            arrow.time_passed += updater_dt;
            if !arrow.can_hurt_owner
                && arrow.hurt_owner_after > Fix::ZERO
                && arrow.time_passed > arrow.hurt_owner_after
            {
                arrow.can_hurt_owner = true;
            }
            // Arrow.UpdateSim runs in the updater pass, before DetPhysics
            // integrates gravity and position. Preserve that authored phase
            // for the policy-visible collider orientation.
            let speed = arrow.velocity.length();
            let up = if speed == Fix::ZERO {
                Vec2::UP
            } else {
                arrow.velocity / speed
            };
            // DetPhysics.IntegrateBody wraps the orientation into [0, 2pi)
            // (DetPhysics.cs:1645) and SyncGameObject writes it back to the
            // arrow's FixTransform, so a straight shot reads 0, not 2pi.
            arrow.rotation = up
                .perp_cw()
                .normalized_angle()
                .checked_rem(Fix::TAU, true)
                .expect("finite angle");
            arrow.velocity.y += gravity * physics_dt;
            arrow.position += arrow.velocity * physics_dt;
        }
    }

    fn mines_late_tick(&mut self, data: &data::RolloutData) {
        for item_index in 0..self.thrown_items.len() {
            if !self.thrown_items[item_index].active
                || self.thrown_items[item_index].ability != ABILITY_MINE
                || self.thrown_items[item_index].entity.is_none()
            {
                continue;
            }
            let owner = self.thrown_items[item_index].owner;
            if self.player_time_scale < Fix::from_f32(0.01) && !self.time_stop_protected[owner] {
                continue;
            }

            let phase = self.thrown_items[item_index].phase;
            {
                let mine = &mut self.thrown_items[item_index];
                mine.age += DT;
                let hover = (mine.age * data.mine.hover_period + mine.random_offset).cos()
                    * data.mine.hover_strength;
                mine.body.force += Vec2::UP * hover;
                if mine.phase == 0 && mine.age >= data.mine.time_before_armed {
                    mine.phase = 1;
                }
            }

            if phase == 0 {
                continue;
            }
            let scan_target = if matches!(phase, 1 | 2) {
                let position = self.thrown_items[item_index].position;
                let scan = Circle {
                    center: position,
                    radius: data.mine.scan_radius,
                };
                self.players
                    .iter()
                    .enumerate()
                    .find(|(player_index, player)| {
                        *player_index != owner
                            && self.playable_slot(player)
                            && !player.invisible
                            && player.team != self.players[owner].team
                            && crate::physics::contact::circle_box(scan, player.base.collider)
                                .is_some()
                    })
                    .map(|(_, player)| player.base.position)
            } else {
                None
            };

            if phase == 1 {
                if let Some(target) = scan_target {
                    let mine = &mut self.thrown_items[item_index];
                    mine.phase = 2;
                    mine.prev_chase_dir = (target - mine.position).normalized_safe();
                }
                continue;
            }

            if phase == 2 {
                let mut detonate = false;
                {
                    let mine = &mut self.thrown_items[item_index];
                    mine.time_since_anticipation += DT;
                    if mine.time_since_anticipation > data.mine.chase_time {
                        mine.time_since_anticipation = Fix::ZERO;
                        mine.phase = 3;
                    }
                    if let Some(target) = scan_target {
                        let to_target = target - mine.position;
                        if to_target.length_squared() < Fix::from_int(4) {
                            detonate = true;
                        } else {
                            mine.prev_chase_dir = to_target.normalized_safe();
                        }
                    }
                    if !detonate {
                        mine.body.force += mine.prev_chase_dir * data.mine.chase_strength;
                    }
                }
                if detonate {
                    self.detonate_mine(item_index, data);
                }
                continue;
            }

            if phase == 3 {
                let elapsed = self.thrown_items[item_index].time_since_anticipation + DT;
                self.thrown_items[item_index].time_since_anticipation = elapsed;
                if elapsed >= data.mine.anticipation_duration {
                    self.detonate_mine(item_index, data);
                    continue;
                }
                let shake = data.mine.shake_force * (elapsed / data.mine.anticipation_duration);
                let random_angle = self.rng.fix_range(Fix::ZERO, Fix::TAU);
                let random_radius = self.rng.fix_range(Fix::ZERO, Fix::ONE);
                let random_vector =
                    Vec2::angle(random_angle) * random_radius * shake * Fix::from_int(3);
                let mine = &mut self.thrown_items[item_index];
                mine.position -= mine.prev_velocity_change;
                let velocity_change = random_vector - mine.prev_rand_vector;
                mine.position += velocity_change;
                mine.prev_rand_vector = random_vector;
                mine.prev_velocity_change = velocity_change;
            }
        }
    }

    fn detonate_mine(&mut self, item_index: usize, data: &data::RolloutData) {
        if !self.thrown_items[item_index].active {
            return;
        }
        let item = self.thrown_items[item_index];
        // Mine and Missile both reference the same MineExplosion prefab. Mine
        // supplies Explosion.PlayerOwnerId for lethal attribution; the
        // co-located Shockwave/IPlayerIdHolder remains owner zero.
        self.spawn_missile_explosion(item.position, item.owner, true, data);
        // Detonation runs in LateUpdateSim, after DetPhysics has already
        // built this observation's collider arrays. DestroyFix therefore
        // leaves one destroyed row until the next Simulate pass removes it.
        self.remove_thrown_item(item_index, true);
    }

    fn thrown_items_integrate(
        &mut self,
        data: &data::RolloutData,
        updater_dt: Fix,
        physics_dt: Fix,
    ) {
        // BananaPeel.UpdateSim runs before DetPhysics. It starts on the first
        // updater pass after Fire, pins a settled tack before each integration,
        // and performs the armed proximity query without allocating.
        for item_index in 0..self.thrown_items.len() {
            if !self.thrown_items[item_index].active
                || self.thrown_items[item_index].ability != ABILITY_TACK
                || self.thrown_items[item_index].born_tick >= self.tick
            {
                continue;
            }
            {
                let item = &mut self.thrown_items[item_index];
                if !item.launched {
                    item.launched = true;
                    item.body.angular_velocity = Fix::from_int(4);
                    if let Some(entity) = item.entity {
                        self.entities[entity].owner = item.owner as u32 + 1;
                    }
                }
                if item.settled {
                    item.position = item.settle_position;
                    item.body.velocity = Vec2::ZERO;
                    item.body.angular_velocity = Fix::ZERO;
                }
                item.age += updater_dt;
            }
            if self.thrown_items[item_index].age > data.tack.lifetime_seconds {
                self.remove_thrown_item(item_index, false);
                continue;
            }
            if self.thrown_items[item_index].age <= data.tack.arm_seconds {
                continue;
            }
            let position = self.thrown_items[item_index].position;
            let owner = self.thrown_items[item_index].owner;
            let radius = data.tack.item_radius + Fix::from_f32(0.75);
            let mut victim = None;
            for player_index in 0..self.players.len() {
                if !self.playable(player_index) {
                    continue;
                }
                let in_range = (self.players[player_index].base.position - position)
                    .length_squared()
                    < radius * radius;
                if player_index == owner && !self.thrown_items[item_index].owner_clear {
                    if !in_range {
                        self.thrown_items[item_index].owner_clear = true;
                    }
                    continue;
                }
                if in_range {
                    victim = Some(player_index);
                    break;
                }
            }
            if let Some(victim) = victim {
                let killer = owner as u32 + 1;
                let cause = if victim == owner {
                    CAUSE_ACCIDENT
                } else {
                    CAUSE_OTHER
                };
                // BananaPeel invokes PlayerCollision.killPlayer from its
                // updater: it creates the normal death effect, while
                // DetPhysics removes all killed colliders before contacts.
                self.next_spawn_hierarchy += data.revival.death_effect_updatable_count;
                self.kill(victim, killer, cause, false);
                self.remove_thrown_item(item_index, false);
            }
        }

        // Mjolnir.UpdateSim also begins one pass after release. The fixed
        // player bitset replaces the source HashSet without allocating.
        for item_index in 0..self.thrown_items.len() {
            if !self.thrown_items[item_index].active
                || self.thrown_items[item_index].ability != ABILITY_MJOLNIR
                || self.thrown_items[item_index].born_tick >= self.tick
            {
                continue;
            }
            {
                let item = &mut self.thrown_items[item_index];
                if !item.launched {
                    item.launched = true;
                    item.body.angular_velocity = if item.body.velocity.x >= Fix::ZERO {
                        Fix::from_int(-11)
                    } else {
                        Fix::from_int(11)
                    };
                    if let Some(entity) = item.entity {
                        self.entities[entity].owner = item.owner as u32 + 1;
                    }
                }
                item.age += updater_dt;
            }
            if self.thrown_items[item_index].age > data.mjolnir.max_flight_seconds {
                self.remove_thrown_item(item_index, false);
                continue;
            }

            let item_position = self.thrown_items[item_index].position;
            let owner = self.thrown_items[item_index].owner;
            let hit_radius = data.mjolnir.item_radius + Fix::from_f32(0.8);
            let mut victims = 0u8;
            for player_index in 0..self.players.len() {
                if player_index == owner
                    || !self.playable(player_index)
                    || self.thrown_items[item_index].killed_players & (1 << player_index) != 0
                {
                    continue;
                }
                if (self.players[player_index].base.position - item_position).length_squared()
                    < hit_radius * hit_radius
                {
                    victims |= 1 << player_index;
                }
            }
            self.thrown_items[item_index].killed_players |= victims;
            for player_index in 0..self.players.len() {
                if victims & (1 << player_index) == 0 {
                    continue;
                }
                self.next_spawn_hierarchy += data.revival.death_effect_updatable_count;
                self.kill(player_index, owner as u32 + 1, CAUSE_OTHER, false);
            }

            if self.thrown_items[item_index].phase == 0 {
                let damping = (Fix::ONE - data.mjolnir.outbound_drag * updater_dt).max(Fix::ZERO);
                let item = &mut self.thrown_items[item_index];
                item.body.velocity *= damping;
                if item.age > data.mjolnir.clang_grace_seconds
                    && item.body.velocity.length() < data.mjolnir.return_speed_threshold
                {
                    item.phase = 1;
                }
            } else {
                if self.players[owner].out {
                    self.remove_thrown_item(item_index, false);
                    continue;
                }
                if !self.playable(owner) || self.players[owner].teleport_effect_active {
                    continue;
                }
                let to_owner = self.players[owner].base.position + Vec2::UP * Fix::from_f32(1.1)
                    - self.thrown_items[item_index].position;
                {
                    let item = &mut self.thrown_items[item_index];
                    item.body.velocity +=
                        to_owner.normalized_safe() * data.mjolnir.recall_accel * updater_dt;
                    let speed = item.body.velocity.length();
                    if speed > data.mjolnir.recall_max_speed {
                        item.body.velocity *= data.mjolnir.recall_max_speed / speed;
                    }
                }
                if to_owner.length_squared() < Fix::from_int(4) {
                    self.remove_thrown_item(item_index, false);
                }
            }
        }

        for item_index in 0..self.thrown_items.len() {
            if !self.thrown_items[item_index].active {
                continue;
            }
            let ability = self.thrown_items[item_index].ability;
            let item_inverse_mass = if ability == ABILITY_MINE {
                data.mine.item_body.inverse_mass
            } else {
                throw_config(data, ability).item_body.inverse_mass
            };
            let timed_out = {
                let item = &mut self.thrown_items[item_index];
                if matches!(ability, ABILITY_GRENADE | ABILITY_SMOKE) && item.born_tick < self.tick
                {
                    item.fuse -= updater_dt;
                    if item.hurt_owner_delay > Fix::ZERO {
                        item.hurt_owner_delay -= updater_dt;
                        if item.hurt_owner_delay <= Fix::ZERO {
                            item.detonates_on_owner = true;
                        }
                    }
                }
                matches!(ability, ABILITY_GRENADE | ABILITY_SMOKE) && item.fuse <= Fix::ZERO
            };
            if timed_out {
                if ability == ABILITY_SMOKE {
                    self.detonate_smoke(item_index, false, data);
                } else {
                    self.detonate_grenade(item_index, false, data);
                }
                continue;
            }
            if self.thrown_items[item_index].entity.is_none() {
                continue;
            }
            let item = &mut self.thrown_items[item_index];
            let delta = item.body.integrate(physics_dt, Fix::from_int(-40));
            item.position += delta;
            let entity = item.entity.expect("active thrown item collider");
            let row = &mut self.entities[entity];
            row.flags = E_ACTIVE | E_ENABLED;
            row.x = item.position.x.raw();
            row.y = item.position.y.raw();
            row.vx = item.body.velocity.x.raw();
            row.vy = item.body.velocity.y.raw();
            row.rotation = item.body.orientation.raw();
            row.angular_velocity = item.body.angular_velocity.raw();
            row.inverse_mass = item_inverse_mass.raw();
        }
    }

    fn remove_thrown_item(&mut self, item_index: usize, by_callback: bool) {
        let item = &mut self.thrown_items[item_index];
        item.active = false;
        if let Some(entity) = item.entity {
            let row = &mut self.entities[entity];
            row.flags = if by_callback { E_DESTROYED } else { E_REMOVED };
            row.owner = 0;
        }
    }

    fn detonate_grenade(&mut self, item_index: usize, by_callback: bool, data: &data::RolloutData) {
        if !self.thrown_items[item_index].active {
            return;
        }
        let item = self.thrown_items[item_index];
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.grenade.explosion_updatable_count;
        let explosion = GrenadeExplosion {
            active: true,
            born_tick: self.tick,
            hierarchy_base,
            position: item.position,
            owner: item.owner,
            ..GrenadeExplosion::default()
        };
        if let Some(free) = self
            .grenade_explosions
            .iter()
            .position(|explosion| !explosion.active)
        {
            let entity = self.grenade_explosions[free].entity;
            self.grenade_explosions[free] = GrenadeExplosion {
                entity,
                ..explosion
            };
        } else {
            self.grenade_explosions.push(explosion);
        }
        self.remove_thrown_item(item_index, by_callback);
    }

    fn detonate_smoke(&mut self, item_index: usize, by_callback: bool, data: &data::RolloutData) {
        if !self.thrown_items[item_index].active {
            return;
        }
        let item = self.thrown_items[item_index];
        let ability_slot = self.players[item.owner].active_slot;
        for &authored_offset in &data.smoke.cloud_offsets {
            let noise_angle = self.rng.fix_range(Fix::ZERO, Fix::TAU);
            let noise_radius = self.rng.fix_range(Fix::ZERO, Fix::ONE);
            let noise = Vec2::angle(noise_angle) * noise_radius * data.smoke.noise;
            let rotation = self.rng.fix_range(Fix::ZERO, Fix::TAU);
            let offset = (authored_offset + noise) * data.smoke.spread;

            let active_for_form = self
                .smoke_clouds
                .iter()
                .filter(|cloud| {
                    cloud.active && cloud.owner == item.owner && cloud.ability_slot == ability_slot
                })
                .count();
            if active_for_form >= 40
                && let Some((oldest, _)) = self
                    .smoke_clouds
                    .iter()
                    .enumerate()
                    .filter(|(_, cloud)| {
                        cloud.active
                            && cloud.owner == item.owner
                            && cloud.ability_slot == ability_slot
                    })
                    .min_by_key(|(_, cloud)| cloud.hierarchy_base)
            {
                self.remove_smoke_cloud(oldest);
            }

            let hierarchy_base = self.next_spawn_hierarchy;
            self.next_spawn_hierarchy += data.smoke.cloud_updatable_count;
            let cloud = SmokeCloud {
                active: true,
                born_tick: self.tick,
                hierarchy_base,
                position: item.position + offset,
                rotation,
                scale: Fix::ONE,
                owner: item.owner,
                ability_slot,
                ignition_time: -Fix::ONE,
                ..SmokeCloud::default()
            };
            if let Some(free) = self.smoke_clouds.iter().position(|cloud| !cloud.active) {
                let entity = self.smoke_clouds[free].entity;
                self.smoke_clouds[free] = SmokeCloud { entity, ..cloud };
            } else {
                self.smoke_clouds.push(cloud);
            }
            // SmokeGrenadeExplode2 instantiates the presentation burst after
            // each cloud; it has updatables but no observed collider.
            self.next_spawn_hierarchy += data.smoke.burst_updatable_count;
        }
        self.remove_thrown_item(item_index, by_callback);
    }

    fn remove_smoke_cloud(&mut self, cloud_index: usize) {
        let cloud = &mut self.smoke_clouds[cloud_index];
        cloud.active = false;
        cloud.registered = false;
        if let Some(entity) = cloud.entity {
            self.entities[entity].flags = E_REMOVED;
        }
    }

    fn detonate_smoke_cloud(&mut self, cloud_index: usize, data: &data::RolloutData) {
        let cloud = self.smoke_clouds[cloud_index];
        if !cloud.active {
            return;
        }
        let hierarchy_base = self.next_spawn_hierarchy;
        self.next_spawn_hierarchy += data.smoke.explosion_updatable_count;
        let explosion = SmokeExplosion {
            active: true,
            born_tick: self.tick,
            hierarchy_base,
            position: cloud.position,
            scale: cloud.scale,
            owner: cloud.owner,
            ..SmokeExplosion::default()
        };
        if let Some(free) = self
            .smoke_explosions
            .iter()
            .position(|explosion| !explosion.active)
        {
            let entity = self.smoke_explosions[free].entity;
            self.smoke_explosions[free] = SmokeExplosion {
                entity,
                ..explosion
            };
        } else {
            self.smoke_explosions.push(explosion);
        }
        self.remove_smoke_cloud(cloud_index);
    }

    /// DetPhysics contact group order for Arrow's circle is Circle/RR,
    /// Circle/Circle, then Circle/Box. Arrow.OnCollide consumes wall contacts,
    /// but intentionally returns without destroying itself for PlayerBody
    /// hurtboxes; PlayerCollision can therefore kill one or more slimes while
    /// the arrow continues flying.
    fn arrows_contacts(&mut self, data: &data::RolloutData) {
        let radius = data.arrow.radius;
        for i in 0..self.arrows.len() {
            if !self.arrows[i].active || !self.arrows[i].registered {
                continue;
            }
            let arrow = self.arrows[i];
            let circle = Circle {
                center: arrow.position,
                radius,
            };

            // Circle/RR: DetPhysics tests every pair first, then applies the
            // collision impulse and positional correction to the arrow and
            // the platform bodies (DetPhysics.cs:3012-3048); Arrow.OnCollide
            // only runs from the callback queue afterwards
            // (DetPhysics.cs:3195-3199), so a landing arrow kicks its
            // platform before the wall callback destroys it. That callback
            // precedes all player shape groups: an arrow overlapping terrain
            // and a player in the same pass never reaches PlayerCollision.
            let contacts: Vec<_> = self
                .platforms
                .iter()
                .enumerate()
                .filter_map(|(p, platform)| {
                    crate::physics::contact::circle_rounded(circle, platform.collider)
                        .map(|contact| (p, contact))
                })
                .collect();
            if !contacts.is_empty() {
                let mut arrow_body = crate::physics::Body {
                    velocity: arrow.velocity,
                    orientation: arrow.rotation,
                    ..data.arrow.body
                };
                let mut center = circle.center;
                for (p, contact) in contacts {
                    let platform_body = &mut self.platform_dynamics[p].body;
                    resolve(
                        contact,
                        &mut arrow_body,
                        platform_body,
                        center,
                        self.platforms[p].surface.center,
                    );
                    let correction = positional_correction(
                        contact,
                        arrow_body.inverse_mass,
                        platform_body.inverse_mass,
                        false,
                    );
                    center += correction[0];
                    self.platforms[p].surface.center += correction[1];
                    self.platforms[p].collider.inner.center += correction[1];
                }
                self.platforms_changed();
                // Arrow.OnCollide destroys the components from the callback
                // queue: this pass still observes the corrected collider and
                // resolved body, flagged destroyed and ownerless
                // (Arrow.cs:124-128); the next pass removes it.
                self.destroy_arrow(i);
                if let Some(entity) = self.arrows[i].entity {
                    let e = &mut self.entities[entity];
                    e.flags = E_DESTROYED;
                    e.owner = 0;
                    e.x = center.x.raw();
                    e.y = center.y.raw();
                    e.vx = arrow_body.velocity.x.raw();
                    e.vy = arrow_body.velocity.y.raw();
                    e.rotation = arrow.rotation.raw();
                }
                continue;
            }

            let mut victims = Vec::new();
            // Platform anticipation replaces the original slime box with a
            // Player-layer circle and is the only supported player circle.
            for (j, player) in self.players.iter().enumerate() {
                if !self.playable(j)
                    || player.active != ActiveAbility::Platform
                    || player.platform.phase != PLATFORM_ANTICIPATION
                    || (player.base.id == arrow.owner && !arrow.can_hurt_owner)
                {
                    continue;
                }
                let hurtbox = Circle {
                    center: player.base.position,
                    radius: data.platform.hurtbox_radius,
                };
                if crate::physics::contact::circle_circle(circle, hurtbox).is_some() {
                    victims.push(j);
                }
            }

            // Normal, Dash, and Bow hurtboxes are Player-layer boxes. Formed
            // Platform has no player hurtbox until the original slime exits.
            for (j, player) in self.players.iter().enumerate() {
                if !self.playable(j)
                    || player.active == ActiveAbility::Platform
                    || (player.base.id == arrow.owner && !arrow.can_hurt_owner)
                {
                    continue;
                }
                let hurtbox = match player.active {
                    ActiveAbility::Bow => {
                        let extents = data.bow.hurtbox_extents;
                        let up = if player.bow.aim == Vec2::ZERO {
                            Vec2::UP
                        } else {
                            player.bow.aim
                        };
                        OrientedBox {
                            center: player.base.position,
                            right: up.perp_cw() * extents.x,
                            up: up * extents.y,
                            inverse_extents: Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y),
                        }
                    }
                    ActiveAbility::Grenade
                    | ActiveAbility::Smoke
                    | ActiveAbility::Tack
                    | ActiveAbility::Mjolnir => OrientedBox::from_basis(
                        player.base.position,
                        Vec2::angle(player.base.rotation),
                        throw_config(data, active_throw_ability(player.active)).hurtbox_extents,
                    ),
                    _ => player.base.collider,
                };
                if crate::physics::contact::circle_box(circle, hurtbox).is_some() {
                    victims.push(j);
                }
            }

            for victim in victims {
                if !self.players[victim].base.alive {
                    continue;
                }
                let victim_id = self.players[victim].base.id;
                let cause = if arrow.owner == victim_id {
                    CAUSE_ACCIDENT
                } else {
                    CAUSE_OTHER
                };
                self.kill(victim, arrow.owner, cause, true);
                if arrow.owner != victim_id && arrow.owner >= 1 {
                    self.players[arrow.owner as usize - 1].telemetry.arrow_hits += 1;
                }
            }
        }
    }

    fn guided_missile_contacts(&mut self, data: &data::RolloutData) {
        for missile_index in 0..self.guided_missiles.len() {
            if !self.guided_missiles[missile_index].active
                || !self.guided_missiles[missile_index].registered
            {
                continue;
            }
            let missile = self.guided_missiles[missile_index];
            let missile_box = OrientedBox::from_basis(
                missile.position,
                Vec2::angle(missile.body.orientation),
                data.missile.spark_extents,
            );
            let missile_shape = RoundedRect {
                inner: missile_box,
                radius: Fix::ZERO,
            };
            let mut contacts = [(0usize, Contact::default()); 64];
            let mut contact_count = 0;
            for (platform_index, platform) in self.platforms.iter().enumerate() {
                if contact_count == contacts.len() {
                    break;
                }
                if let Some(contact) =
                    crate::physics::contact::rounded_rounded(missile_shape, platform.collider)
                {
                    contacts[contact_count] = (platform_index, contact);
                    contact_count += 1;
                }
            }
            if contact_count > 0 {
                let mut body = missile.body;
                let mut center = missile.position;
                for &(platform_index, contact) in &contacts[..contact_count] {
                    let platform_center = self.platforms[platform_index].surface.center;
                    let platform_body = &mut self.platform_dynamics[platform_index].body;
                    resolve(contact, &mut body, platform_body, center, platform_center);
                    let correction = positional_correction(
                        contact,
                        body.inverse_mass,
                        platform_body.inverse_mass,
                        false,
                    );
                    center += correction[0];
                    self.platforms[platform_index].surface.center += correction[1];
                    self.platforms[platform_index].collider.inner.center += correction[1];
                }
                self.platforms_changed();
                let owner = missile.owner;
                self.spawn_missile_explosion(center, owner, false, data);
                self.destroy_guided_missile(missile_index, true);
                if let Some(entity) = missile.entity {
                    let row = &mut self.entities[entity];
                    row.x = center.x.raw();
                    row.y = center.y.raw();
                    row.vx = body.velocity.x.raw();
                    row.vy = body.velocity.y.raw();
                    row.rotation = body.orientation.raw();
                    row.angular_velocity = body.angular_velocity.raw();
                }
                continue;
            }

            let hit_player = self.players.iter().enumerate().any(|(index, player)| {
                if !self.playable(index) {
                    return false;
                }
                let entity = match player.active {
                    ActiveAbility::None => player.body_entity,
                    ActiveAbility::Dash => player.dash_entity,
                    ActiveAbility::Bow => player.bow_entity,
                    ActiveAbility::Grenade
                    | ActiveAbility::Smoke
                    | ActiveAbility::Tack
                    | ActiveAbility::Mjolnir
                    | ActiveAbility::Missile => player.throw_entity,
                    ActiveAbility::Mine => player.mine_entity,
                    ActiveAbility::Spike => player.spike_entity,
                    ActiveAbility::TimeStop => player.time_stop_entity,
                    ActiveAbility::Gust => player.gust_entity,
                    ActiveAbility::Revival => player.revival_entity,
                    ActiveAbility::Platform => player.circle_entity,
                };
                if entity == usize::MAX || self.entities[entity].flags & E_ACTIVE == 0 {
                    return false;
                }
                let row = self.entities[entity];
                if row.kind == 0 {
                    crate::physics::contact::circle_box(
                        Circle {
                            center: Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y)),
                            radius: Fix::from_raw(row.radius),
                        },
                        missile_box,
                    )
                    .is_some()
                } else {
                    let right = Vec2::new(Fix::from_raw(row.right_x), Fix::from_raw(row.right_y));
                    let up = Vec2::new(Fix::from_raw(row.up_x), Fix::from_raw(row.up_y));
                    crate::physics::contact::box_box(
                        OrientedBox {
                            center: Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y)),
                            right,
                            up,
                            inverse_extents: Vec2::new(
                                Fix::ONE / right.length(),
                                Fix::ONE / up.length(),
                            ),
                        },
                        missile_box,
                    )
                    .is_some()
                }
            });
            if hit_player {
                self.spawn_missile_explosion(missile.position, missile.owner, false, data);
                self.destroy_guided_missile(missile_index, true);
            }
        }
    }

    /// Item circles participate in ordinary rigid-body contacts, then their
    /// queued GrenadeExplode callback observes Player-layer overlaps.
    fn thrown_items_contacts(&mut self, data: &data::RolloutData) {
        for item_index in 0..self.thrown_items.len() {
            if !self.thrown_items[item_index].active
                || self.thrown_items[item_index].entity.is_none()
            {
                continue;
            }
            let item = self.thrown_items[item_index];
            let item_radius = if item.ability == ABILITY_MINE {
                data.mine.item_radius
            } else {
                throw_config(data, item.ability).item_radius
            };
            let circle = Circle {
                center: item.position,
                radius: item_radius,
            };
            let mut contacts = [(0usize, Contact::default()); 64];
            let mut contact_count = 0;
            for (platform_index, platform) in self.platforms.iter().enumerate() {
                if contact_count == contacts.len() {
                    break;
                }
                if let Some(contact) =
                    crate::physics::contact::circle_rounded(circle, platform.collider)
                {
                    contacts[contact_count] = (platform_index, contact);
                    contact_count += 1;
                }
            }
            if contact_count > 0 {
                let mut body = item.body;
                let mut center = item.position;
                for &(platform_index, contact) in &contacts[..contact_count] {
                    let platform_center = self.platforms[platform_index].surface.center;
                    let platform_body = &mut self.platform_dynamics[platform_index].body;
                    resolve(contact, &mut body, platform_body, center, platform_center);
                    let correction = positional_correction(
                        contact,
                        body.inverse_mass,
                        platform_body.inverse_mass,
                        false,
                    );
                    center += correction[0];
                    self.platforms[platform_index].surface.center += correction[1];
                    self.platforms[platform_index].collider.inner.center += correction[1];
                }
                self.platforms_changed();
                let thrown = &mut self.thrown_items[item_index];
                thrown.body = body;
                thrown.position = center;
                let entity = thrown.entity.expect("contacting grenade collider");
                {
                    let row = &mut self.entities[entity];
                    row.x = center.x.raw();
                    row.y = center.y.raw();
                    row.vx = body.velocity.x.raw();
                    row.vy = body.velocity.y.raw();
                    row.rotation = body.orientation.raw();
                    row.angular_velocity = body.angular_velocity.raw();
                }

                if item.ability == ABILITY_MJOLNIR
                    && item.launched
                    && item.age >= data.mjolnir.clang_grace_seconds
                {
                    let force = body.velocity.normalized_safe() * data.mjolnir.platform_push_force;
                    for &(platform_index, _) in &contacts[..contact_count] {
                        self.platform_dynamics[platform_index].body.force += force;
                    }
                    self.platforms_changed();
                    if self.thrown_items[item_index].phase == 0 {
                        self.thrown_items[item_index].phase = 1;
                    }
                }

                if item.ability == ABILITY_TACK
                    && !item.settled
                    && item.launched
                    && item.age > Fix::from_f32(0.1)
                    && item.body.velocity.y <= Fix::from_f32(0.5)
                    && contacts[..contact_count]
                        .iter()
                        .any(|(_, contact)| contact.normal.y < Fix::from_f32(-0.5))
                {
                    let thrown = &mut self.thrown_items[item_index];
                    thrown.settled = true;
                    thrown.settle_position = center;
                    thrown.body.velocity = Vec2::ZERO;
                    thrown.body.angular_velocity = Fix::ZERO;
                    let row = &mut self.entities[entity];
                    row.vx = 0;
                    row.vy = 0;
                    row.angular_velocity = 0;
                }
            }

            if item.ability == ABILITY_MINE && matches!(self.thrown_items[item_index].phase, 1 | 2)
            {
                let mine_circle = Circle {
                    center: self.thrown_items[item_index].position,
                    radius: data.mine.item_radius,
                };
                let hit_player = self.players.iter().any(|player| {
                    self.playable_slot(player)
                        && crate::physics::contact::circle_box(mine_circle, player.base.collider)
                            .is_some()
                });
                let hit_projectile = self.arrows.iter().any(|arrow| {
                    arrow.active
                        && arrow.registered
                        && crate::physics::contact::circle_circle(
                            mine_circle,
                            Circle {
                                center: arrow.position,
                                radius: data.arrow.radius,
                            },
                        )
                        .is_some()
                });
                let hit_grenade_explosion = self.grenade_explosions.iter().any(|explosion| {
                    explosion.active
                        && explosion.registered
                        && explosion.age == 1
                        && crate::physics::contact::circle_box(
                            mine_circle,
                            OrientedBox::from_basis(
                                explosion.position,
                                Vec2::RIGHT,
                                data.grenade.explosion_extents,
                            ),
                        )
                        .is_some()
                });
                let hit_mine_explosion = self.missile_explosions.iter().any(|explosion| {
                    explosion.active
                        && explosion.registered
                        && explosion.age == 1
                        && crate::physics::contact::circle_box(
                            mine_circle,
                            OrientedBox::from_basis(
                                explosion.position,
                                Vec2::RIGHT,
                                data.mine.explosion_extents,
                            ),
                        )
                        .is_some()
                });
                if hit_player || hit_projectile || hit_grenade_explosion || hit_mine_explosion {
                    let mine = &mut self.thrown_items[item_index];
                    mine.time_since_anticipation =
                        data.mine.anticipation_duration * Fix::from_f32(0.8);
                    mine.phase = 3;
                }
            }

            if matches!(item.ability, ABILITY_TACK | ABILITY_MJOLNIR | ABILITY_MINE) {
                continue;
            }

            let owner_id = item.owner as u32 + 1;
            let mut detonate = false;
            for player_index in 0..self.players.len() {
                if !self.playable(player_index) {
                    continue;
                }
                let player = &self.players[player_index];
                if player.base.id == owner_id && !item.detonates_on_owner {
                    continue;
                }
                let hit = if player.active == ActiveAbility::Platform
                    && player.platform.phase == PLATFORM_ANTICIPATION
                {
                    crate::physics::contact::circle_circle(
                        circle,
                        Circle {
                            center: player.base.position,
                            radius: data.platform.hurtbox_radius,
                        },
                    )
                    .is_some()
                } else if player.active == ActiveAbility::Platform {
                    false
                } else {
                    let hurtbox = match player.active {
                        ActiveAbility::Bow => {
                            let extents = data.bow.hurtbox_extents;
                            let up = if player.bow.aim == Vec2::ZERO {
                                Vec2::UP
                            } else {
                                player.bow.aim
                            };
                            OrientedBox {
                                center: player.base.position,
                                right: up.perp_cw() * extents.x,
                                up: up * extents.y,
                                inverse_extents: Vec2::new(
                                    Fix::ONE / extents.x,
                                    Fix::ONE / extents.y,
                                ),
                            }
                        }
                        ActiveAbility::Grenade
                        | ActiveAbility::Smoke
                        | ActiveAbility::Tack
                        | ActiveAbility::Mjolnir => OrientedBox::from_basis(
                            player.base.position,
                            Vec2::angle(player.base.rotation),
                            throw_config(data, active_throw_ability(player.active)).hurtbox_extents,
                        ),
                        _ => player.base.collider,
                    };
                    crate::physics::contact::circle_box(circle, hurtbox).is_some()
                };
                detonate |= hit;
            }
            if detonate {
                if item.ability == ABILITY_SMOKE {
                    self.detonate_smoke(item_index, true, data);
                } else {
                    self.detonate_grenade(item_index, true, data);
                }
            }
        }
    }

    /// Explosion's box is lethal only on its first registered frame. The two
    /// Shockwaves have already run in Init; this is the later DetPhysics
    /// callback phase.
    fn grenade_explosion_contacts(&mut self, data: &data::RolloutData) {
        for explosion_index in 0..self.grenade_explosions.len() {
            let explosion = self.grenade_explosions[explosion_index];
            if !explosion.active || !explosion.registered || explosion.age != 1 {
                continue;
            }
            let hitbox = OrientedBox::from_basis(
                explosion.position,
                Vec2::RIGHT,
                data.grenade.explosion_extents,
            );
            let mut victims = [false; MAX_PLAYERS];
            for (player_index, victim) in victims.iter_mut().enumerate().take(self.players.len()) {
                if !self.playable(player_index) {
                    continue;
                }
                let player = &self.players[player_index];
                *victim = if player.active == ActiveAbility::Platform
                    && player.platform.phase == PLATFORM_ANTICIPATION
                {
                    crate::physics::contact::circle_box(
                        Circle {
                            center: player.base.position,
                            radius: data.platform.hurtbox_radius,
                        },
                        hitbox,
                    )
                    .is_some()
                } else if player.active == ActiveAbility::Platform {
                    false
                } else {
                    crate::physics::contact::box_box(player.base.collider, hitbox).is_some()
                };
            }
            let killer = explosion.owner as u32 + 1;
            for (player_index, &victim) in victims.iter().enumerate().take(self.players.len()) {
                if victim && self.playable(player_index) {
                    let cause = if self.players[player_index].base.id == killer {
                        CAUSE_ACCIDENT
                    } else {
                        CAUSE_OTHER
                    };
                    self.kill(player_index, killer, cause, true);
                }
            }
        }
    }

    fn missile_explosion_contacts(&mut self, data: &data::RolloutData) {
        for explosion_index in 0..self.missile_explosions.len() {
            let explosion = self.missile_explosions[explosion_index];
            if !explosion.active || !explosion.registered || explosion.age != 1 {
                continue;
            }
            let hitbox = OrientedBox::from_basis(
                explosion.position,
                Vec2::RIGHT,
                data.missile.explosion_extents,
            );
            let mut victims = [false; MAX_PLAYERS];
            for (player_index, victim) in victims.iter_mut().enumerate().take(self.players.len()) {
                if !self.playable(player_index) {
                    continue;
                }
                let player = &self.players[player_index];
                *victim = if player.active == ActiveAbility::Platform
                    && player.platform.phase == PLATFORM_ANTICIPATION
                {
                    crate::physics::contact::circle_box(
                        Circle {
                            center: player.base.position,
                            radius: data.platform.hurtbox_radius,
                        },
                        hitbox,
                    )
                    .is_some()
                } else if player.active == ActiveAbility::Platform {
                    false
                } else {
                    crate::physics::contact::box_box(player.base.collider, hitbox).is_some()
                };
            }
            let killer = if explosion.credited_owner {
                explosion.owner as u32 + 1
            } else {
                0
            };
            for (player_index, &victim) in victims.iter().enumerate().take(self.players.len()) {
                if victim && self.playable(player_index) {
                    let cause = if killer == self.players[player_index].base.id {
                        CAUSE_ACCIDENT
                    } else {
                        CAUSE_OTHER
                    };
                    self.kill(player_index, killer, cause, true);
                }
            }
        }
    }

    fn smoke_contacts(&mut self, data: &data::RolloutData) {
        // Both grenade and smoke explosions have `igniteFlammables`; their
        // boxes are enabled only on their first registered physics pass.
        for explosion in self.grenade_explosions.iter().copied() {
            if !explosion.active || !explosion.registered || explosion.age != 1 {
                continue;
            }
            let hitbox = OrientedBox::from_basis(
                explosion.position,
                Vec2::RIGHT,
                data.grenade.explosion_extents,
            );
            for cloud in &mut self.smoke_clouds {
                if !cloud.active || !cloud.registered {
                    continue;
                }
                let entity = cloud.entity.expect("registered smoke cloud");
                let radius = Fix::from_raw(self.entities[entity].radius);
                if crate::physics::contact::circle_box(
                    Circle {
                        center: cloud.position,
                        radius,
                    },
                    hitbox,
                )
                .is_some()
                {
                    cloud.ignition_time = Fix::ZERO;
                    self.entities[entity].flags = E_ACTIVE;
                }
            }
        }

        for explosion_index in 0..self.smoke_explosions.len() {
            let explosion = self.smoke_explosions[explosion_index];
            if !explosion.active || !explosion.registered || explosion.age != 1 {
                continue;
            }
            let extents = data.smoke.explosion_extents * explosion.scale;
            let hitbox = OrientedBox::from_basis(explosion.position, Vec2::RIGHT, extents);
            for cloud in &mut self.smoke_clouds {
                if !cloud.active || !cloud.registered {
                    continue;
                }
                let entity = cloud.entity.expect("registered smoke cloud");
                let radius = Fix::from_raw(self.entities[entity].radius);
                if crate::physics::contact::circle_box(
                    Circle {
                        center: cloud.position,
                        radius,
                    },
                    hitbox,
                )
                .is_some()
                {
                    cloud.ignition_time = Fix::ZERO;
                    self.entities[entity].flags = E_ACTIVE;
                }
            }

            let mut victims = [false; MAX_PLAYERS];
            for (player_index, victim) in victims.iter_mut().enumerate().take(self.players.len()) {
                if !self.playable(player_index) {
                    continue;
                }
                let player = &self.players[player_index];
                *victim = if player.active == ActiveAbility::Platform
                    && player.platform.phase == PLATFORM_ANTICIPATION
                {
                    crate::physics::contact::circle_box(
                        Circle {
                            center: player.base.position,
                            radius: data.platform.hurtbox_radius,
                        },
                        hitbox,
                    )
                    .is_some()
                } else if player.active == ActiveAbility::Platform {
                    false
                } else {
                    crate::physics::contact::box_box(player.base.collider, hitbox).is_some()
                };
            }
            let killer = explosion.owner as u32 + 1;
            for (player_index, &victim) in victims.iter().enumerate().take(self.players.len()) {
                if victim && self.playable(player_index) {
                    let cause = if self.players[player_index].base.id == killer {
                        CAUSE_ACCIDENT
                    } else {
                        CAUSE_OTHER
                    };
                    self.kill(player_index, killer, cause, true);
                }
            }
        }
    }

    /// Stable rounded-rectangle broad phase, matching the RR subset of the
    /// generic DetPhysics sweep (strict touching-edge rejection included).
    fn platform_pairs(&self, scratch: &mut PlatformScratch) {
        scratch.pairs.clear();
        if self.platforms.len() < 2 {
            return;
        }
        // DetPhysics picks the sweep axis from every registered collider:
        // the mean over the active ones, the spread over all of them,
        // including frozen inactive hurtboxes (DetPhysics.cs:2827-2894).
        // Rounded rectangles have just integrated; the other rows were
        // synced by `refresh_actor_entities`.
        let others = &mut scratch.others;
        others.clear();
        others.extend(
            self.entities
                .iter()
                .filter(|e| e.kind != 2 && e.flags & E_REMOVED == 0)
                .map(|e| {
                    (
                        Vec2::new(Fix::from_raw(e.x), Fix::from_raw(e.y)),
                        e.flags & 3 == 3,
                    )
                })
                .filter_map(|(p, active)| active.then_some(p)),
        );
        let mut mean = Vec2::ZERO;
        for platform in &self.platforms {
            mean += platform.surface.center;
        }
        for p in others.iter() {
            mean += *p;
        }
        mean = mean / Fix::from_int((self.platforms.len() + others.len()) as i64);
        let mut variance = Vec2::ZERO;
        for platform in &self.platforms {
            let delta = mean - platform.surface.center;
            variance += Vec2::new(delta.x * delta.x, delta.y * delta.y);
        }
        for e in self
            .entities
            .iter()
            .filter(|e| e.kind != 2 && e.flags & E_REMOVED == 0)
        {
            let delta = mean - Vec2::new(Fix::from_raw(e.x), Fix::from_raw(e.y));
            variance += Vec2::new(delta.x * delta.x, delta.y * delta.y);
        }
        let sweep_x = variance.x > variance.y;
        let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
        let other = |v: Vec2| if sweep_x { v.y } else { v.x };
        let bounds = &mut scratch.bounds;
        bounds.clear();
        bounds.extend(self.platforms.iter().map(|platform| {
            let r = platform.collider;
            let extent = Vec2::new(
                r.inner.right.x.abs() + r.inner.up.x.abs() + r.radius,
                r.inner.right.y.abs() + r.inner.up.y.abs() + r.radius,
            );
            [r.inner.center - extent, r.inner.center + extent]
        }));
        let bounds = &*bounds;
        let active = &mut scratch.active;
        active.clear();
        active.extend(0..self.platforms.len());
        active.sort_by(|&a, &b| {
            (axis(bounds[a][0]) - axis(bounds[b][0]))
                .sign()
                .cmp(&0)
                .then(a.cmp(&b))
        });
        let pairs = &mut scratch.pairs;
        let window = &mut scratch.window;
        window.clear();
        for &b in active.iter() {
            window.retain(|&a: &usize| axis(bounds[a][1]) > axis(bounds[b][0]));
            for &a in window.iter() {
                if other(bounds[a][1]) <= other(bounds[b][0])
                    || other(bounds[a][0]) >= other(bounds[b][1])
                {
                    continue;
                }
                // IdPair(active, current): the body that entered the sweep
                // first is collider A (DetPhysics.cs:2169-2236). Fix
                // multiplication floors, so swapping A and B changes the
                // impulse and correction by an LSB.
                pairs.push([a, b]);
            }
            window.push(b);
        }
    }

    /// Level-authored AnimateVelocity and AntiLockPlatform updater pass. Map
    /// components run before the dynamically registered player prefabs, so
    /// their damping must observe the velocity from the start of the tick,
    /// before Gust or another later ability applies a direct impulse.
    fn platform_updaters_tick(&mut self, data: &data::RolloutData, updater_dt: Fix) {
        // ActivateSuddenDeath moves the spring home; the body follows through
        // the ordinary spring/integration path rather than teleporting.
        if self.sudden_death {
            let map = &data.maps[self.map];
            let elapsed = Fix::from_int(i64::from(self.tick - self.sudden_death_tick)) * DT;
            for i in 0..self.platforms.len() {
                let def_index = self.platform_defs[i];
                let (speed_scale, accel, delay, curve) = if def_index == usize::MAX {
                    let Some(spawn) = self
                        .spawned
                        .iter()
                        .find(|s| s.active && s.platform_index == i)
                    else {
                        continue;
                    };
                    (
                        data.platform.sudden_death_speed,
                        data.platform.sudden_death_acceleration,
                        spawn.sudden_death_delay,
                        &data.platform.sudden_death_y_speed,
                    )
                } else {
                    let def = &map.platforms[def_index];
                    (
                        def.sudden_death_speed,
                        def.sudden_death_acceleration,
                        self.sudden_death_delays[def_index],
                        &def.sudden_death_y_speed,
                    )
                };
                // AnimateVelocity.UpdateSim: home -= down * curve(t - delay)
                //   * suddenDeathSpeed * (2 << min(t * accel, 5)) * dt.
                let steps = (elapsed * accel).to_int().clamp(0, 5);
                let scale = speed_scale * Fix::from_int(2 << steps);
                let speed = curve.evaluate(elapsed - delay);
                let drop = speed * scale * updater_dt;
                self.platform_dynamics[i].home_position.y -= drop;
            }
        }

        // AnimateVelocity.UpdateSim. Keep the apparently cancelling inverse
        // mass and 60 Hz impulse conversions: their intermediate fixed-point
        // rounding is observable over long-running springs (notably map 2's
        // orbiter), so simplifying this to `velocity += spring * DT` drifts.
        for i in 0..self.platforms.len() {
            let position = self.platforms[i].surface.center;
            if self.platform_dynamics[i].init_tick == Some(self.tick) {
                // AnimateVelocity.Init (AnimateVelocity.cs:74-79) homes on
                // the current position and clears inSuddenDeath, so an
                // in-progress sudden death is re-activated with a fresh delay
                // (AnimateVelocity.cs:126-129, 111-118).
                self.platform_dynamics[i].init_tick = None;
                self.platform_dynamics[i].home_position = position;
                if self.sudden_death {
                    let max_delay = data.platform.sudden_death_max_delay;
                    if let Some(spawn) = self
                        .spawned
                        .iter_mut()
                        .find(|s| s.active && s.platform_index == i)
                    {
                        spawn.sudden_death_delay = self.rng.fix_range(Fix::ZERO, max_delay);
                    }
                }
            }
            let dynamics = &mut self.platform_dynamics[i];
            let spring = (dynamics.home_position - position) * dynamics.speed
                - dynamics.body.velocity * dynamics.mu;
            let impulse = spring * updater_dt / dynamics.body.inverse_mass;
            dynamics.body.force += impulse * Fix::from_int(60);
            let mut angle_delta = dynamics.home_rotation - dynamics.body.orientation;
            if angle_delta.abs() > Fix::PI {
                angle_delta = -Fix::from_int(angle_delta.sign()) * (Fix::TAU - angle_delta.abs());
            }
            dynamics.body.angular_velocity = (dynamics.body.angular_velocity
                + angle_delta * dynamics.rotation_speed * updater_dt)
                * (Fix::ONE - dynamics.rotation_mu);
        }

        // AntiLockPlatform runs after AnimateVelocity: add the ordinary force
        // and re-home at the pre-integration position, preserving the existing
        // one-tick-lagged home behavior.
        if !self.sudden_death && updater_dt > Fix::ZERO {
            let sim_time = Fix::from_int(i64::from(self.tick)) * DT;
            let map = &data.maps[self.map];
            for orbit in &mut self.orbits {
                let platform_index = orbit.0;
                let def = &map.platforms[self.platform_defs[platform_index]];
                let Some(o) = def.orbit.as_ref() else {
                    continue;
                };
                if sim_time < o.delay || o.path.is_empty() {
                    continue;
                }
                let position = self.platforms[platform_index].surface.center;
                let direction = (o.path[orbit.1] - position).normalized_safe();
                let dynamics = &mut self.platform_dynamics[platform_index];
                dynamics.body.force += direction * o.force;
                dynamics.home_position = position;
                // AntiLockPlatform computes this tick's force first, then
                // advances the target from the same pre-physics position.
                let current = o.path[orbit.1];
                let previous = o.path[(orbit.1 + o.path.len() - 1) % o.path.len()];
                if (current - previous).dot(position - current) > Fix::ZERO {
                    orbit.1 = (orbit.1 + 1) % o.path.len();
                }
            }
        }
    }

    /// Rounded-platform rigid-body stage. The order matches DetPhysics after
    /// the updater pass: resize, integration, broad phase, narrow phase,
    /// collision impulse, positional correction, and transform sync.
    fn platforms_tick(&mut self, data: &data::RolloutData, updater_dt: Fix, physics_dt: Fix) {
        // ResizablePlatform.UpdateSim runs every tick while an evicted platform
        // remains a normal collidable body. It evaluates before incrementing
        // time and destroys at the final key without applying another resize.
        let removal_end = data
            .platform
            .remove_curve
            .keys
            .last()
            .expect("authored platform removal curve")
            .time;
        let mut removal_resizes = Vec::new();
        let mut removal_destroys = Vec::new();
        for spawn in &mut self.spawned {
            if !spawn.active || !spawn.being_removed {
                continue;
            }
            if spawn.removal_time >= removal_end {
                spawn.active = false;
                removal_destroys.push(spawn.platform_index);
                continue;
            }
            let amount = data.platform.remove_curve.evaluate(spawn.removal_time);
            spawn.removal_time += updater_dt;
            // Fix.Lerp clamps only the weight of `a` (Fix.cs:41606-41609),
            // so the curve's overshoot past 1 scales the original size.
            let minimum = Fix::from_f32(0.1);
            let lerp =
                |o: Fix| minimum * (Fix::ONE - amount.clamp(Fix::ZERO, Fix::ONE)) + o * amount;
            removal_resizes.push((
                spawn.platform_index,
                Vec2::new(lerp(spawn.removal_extents.x), lerp(spawn.removal_extents.y)),
                lerp(spawn.removal_radius),
            ));
        }

        // Most short episodes never create or disturb a platform. Preserve
        // that hot path without changing semantics: a body exactly at its
        // home with zero velocity cannot produce motion or a new contact.
        let sim_time = Fix::from_int(i64::from(self.tick)) * DT;
        let orbit_running = !self.sudden_death
            && self.orbits.iter().any(|orbit| {
                data.maps[self.map].platforms[self.platform_defs[orbit.0]]
                    .orbit
                    .as_ref()
                    .is_some_and(|o| sim_time >= o.delay && !o.path.is_empty())
            });
        // Bodies and homes only change through paths that bump
        // `platform_epoch`, so a level found quiet stays quiet until then.
        let disturbed =
            self.quiet_epoch != self.platform_epoch
                && self.platforms.iter().zip(&self.platform_dynamics).any(
                    |(platform, dynamics)| {
                        platform.surface.center != dynamics.home_position
                            || dynamics.body.force != Vec2::ZERO
                            || dynamics.body.velocity != Vec2::ZERO
                            || dynamics.body.angular_velocity != Fix::ZERO
                            || dynamics.body.orientation != dynamics.home_rotation
                    },
                );
        if !disturbed {
            self.quiet_epoch = self.platform_epoch;
        }
        if !self.sudden_death
            && !orbit_running
            && !disturbed
            && !self.spawned.iter().any(|spawn| spawn.active)
        {
            return;
        }
        // Everything below may move a body.
        self.platforms_changed();
        // ResizablePlatform follows AnimateVelocity in its prefab
        // (ResizablePlatform.prefab component order), so the spring impulse
        // above used the pre-resize mass while DetPhysics integrates it with
        // the new one.
        for (index, extents, radius) in removal_resizes {
            let center = self.platforms[index].surface.center;
            self.reshape_platform(index, center, extents, radius);
        }
        removal_destroys.sort_unstable_by(|a, b| b.cmp(a));
        for index in removal_destroys {
            self.remove_platform(index);
        }

        // DetPhysics integrates every rounded body before it constructs any
        // contacts. All rollout platforms have gravityScale=0, but pass the
        // game's -40 gravity for completeness.
        for i in 0..self.platforms.len() {
            let dynamics = &mut self.platform_dynamics[i];
            if dynamics.attached_spikes != 0 {
                // SyncAllPhysicsObjectsToFixTransformPosition discards any
                // prior Box/RR correction of CompositeBody.position and
                // re-anchors the composite on its rounded-root FixTransform.
                dynamics.composite_position = self.platforms[i].surface.center;
            }
            if let Some(rotation) = dynamics.transform_rotation.take() {
                dynamics.body.orientation = rotation;
            }
            let shift = dynamics.body.integrate(physics_dt, Fix::from_int(-40));
            let orientation = dynamics.body.orientation;
            let basis = match dynamics.basis {
                Some((rotation, basis)) if rotation == orientation => basis,
                _ => {
                    let basis = Vec2::angle(orientation);
                    dynamics.basis = Some((orientation, basis));
                    basis
                }
            };
            let platform = &mut self.platforms[i];
            if dynamics.attached_spikes != 0 {
                dynamics.root_body.orientation = orientation;
                dynamics.composite_position += shift;
            } else {
                dynamics.composite_position = platform.surface.center + shift;
            }
            platform.surface.center += shift;
            platform.surface.right = basis;
            platform.surface.up = platform.surface.right.perp();
            // Box.SetRotation from the retained inverseExtents (RoundedRect.cs:22-26).
            let inner = &mut platform.collider.inner;
            inner.center = platform.surface.center;
            inner.right =
                (Vec2::RIGHT / inner.inverse_extents.x).complex_mul(platform.surface.right);
            inner.up = (Vec2::UP / inner.inverse_extents.y).complex_mul(platform.surface.right);
        }
        // CollisionInformation stores the shape as it was during narrow phase;
        // PlayerCollision's later squish callback tests that copy, before the
        // RR positional corrections below move either platform.
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.callback_colliders.clear();
        scratch
            .callback_colliders
            .extend(self.platforms.iter().map(|p| p.collider));

        // RR-only sweep/narrow-phase. Contacts are all captured before any
        // impulse or correction, just like DetPhysics' collision buffer.
        self.platform_pairs(&mut scratch);
        scratch.contacts.clear();
        for &[a, b] in &scratch.pairs {
            if let Some(contact) = crate::physics::contact::rounded_rounded(
                self.platforms[a].collider,
                self.platforms[b].collider,
            ) {
                scratch.contacts.push((a, b, contact));
            }
        }
        for &(a, b, contact) in &scratch.contacts {
            let pos_a = if self.platform_dynamics[a].attached_spikes == 0 {
                self.platforms[a].surface.center
            } else {
                self.platform_dynamics[a].composite_position
            };
            let pos_b = if self.platform_dynamics[b].attached_spikes == 0 {
                self.platforms[b].surface.center
            } else {
                self.platform_dynamics[b].composite_position
            };
            let (body_a, body_b) = if a < b {
                let (left, right) = self.platform_dynamics.split_at_mut(b);
                (&mut left[a].body, &mut right[0].body)
            } else {
                let (left, right) = self.platform_dynamics.split_at_mut(a);
                (&mut right[0].body, &mut left[b].body)
            };
            resolve(contact, body_a, body_b, pos_a, pos_b);
            let correction =
                positional_correction(contact, body_a.inverse_mass, body_b.inverse_mass, false);
            for (index, delta) in [(a, correction[0]), (b, correction[1])] {
                self.platforms[index].surface.center += delta;
                self.platforms[index].collider.inner.center += delta;
                if self.platform_dynamics[index].attached_spikes != 0 {
                    self.platform_dynamics[index].composite_position += delta;
                }
            }
        }

        // DetPhysics' Box/RoundedRect group follows RoundedRect/RoundedRect.
        // SpikeAttack boxes are composite children, so impulses and
        // positional correction move their attached platform body.
        scratch.spike_wall_contacts.clear();
        let spike_extents = data::surface_extents(data.spike.attack_extents);
        for (spike_index, spike) in self.spike_attacks.iter().enumerate() {
            if !spike.active || spike.platform_index >= self.platforms.len() {
                continue;
            }
            let parent_index = spike.platform_index;
            let parent = &self.platform_dynamics[parent_index].body;
            let basis = Vec2::angle(parent.orientation + spike.relative_rotation);
            let center = self.platform_dynamics[parent_index].composite_position
                + spike
                    .relative_center
                    .complex_mul(self.platforms[parent_index].surface.right);
            let attack_box = OrientedBox {
                center,
                right: basis * spike_extents.x,
                up: basis.perp() * spike_extents.y,
                inverse_extents: Vec2::new(
                    Fix::ONE / data.spike.attack_extents.x,
                    Fix::ONE / data.spike.attack_extents.y,
                ),
            };
            let attack_rr = RoundedRect {
                inner: attack_box,
                radius: Fix::ZERO,
            };
            for (wall_index, platform) in self.platforms.iter().enumerate() {
                if wall_index == parent_index {
                    continue;
                }
                if let Some(contact) =
                    crate::physics::contact::rounded_rounded(attack_rr, platform.collider)
                {
                    scratch
                        .spike_wall_contacts
                        .push((spike_index, wall_index, contact));
                }
            }
        }
        for &(spike_index, wall_index, contact) in &scratch.spike_wall_contacts {
            let parent_index = self.spike_attacks[spike_index].platform_index;
            let spike_body_position = self.platform_dynamics[parent_index].composite_position;
            let wall_center = if self.platform_dynamics[wall_index].attached_spikes == 0 {
                self.platforms[wall_index].surface.center
            } else {
                self.platform_dynamics[wall_index].composite_position
            };
            let wall_child_inverse_mass = self.platform_dynamics[wall_index].root_body.inverse_mass;
            let (spike_body, wall_body) = if parent_index < wall_index {
                let (left, right) = self.platform_dynamics.split_at_mut(wall_index);
                (&mut left[parent_index].body, &mut right[0].body)
            } else {
                let (left, right) = self.platform_dynamics.split_at_mut(parent_index);
                (&mut right[0].body, &mut left[wall_index].body)
            };
            resolve(
                contact,
                spike_body,
                wall_body,
                spike_body_position,
                wall_center,
            );
            let correction = positional_correction(
                contact,
                data.spike.attack_body.inverse_mass,
                wall_child_inverse_mass,
                false,
            );
            // Box correction moves the box and its CompositeBody.position,
            // but does not directly move the rounded-rect child collider.
            self.platform_dynamics[parent_index].composite_position += correction[0];
            self.platforms[wall_index].surface.center += correction[1];
            self.platforms[wall_index].collider.inner.center += correction[1];
            if self.platform_dynamics[wall_index].attached_spikes != 0 {
                self.platform_dynamics[wall_index].composite_position += correction[1];
            }
        }

        if self.player_time_scale >= Fix::from_f32(0.01) {
            let mut pushed_spike = usize::MAX;
            for &(spike_index, wall_index, contact) in &scratch.spike_wall_contacts {
                if self.spike_attacks[spike_index].time >= data.spike.push_time {
                    continue;
                }
                let parent_index = self.spike_attacks[spike_index].platform_index;
                let parent_basis = self.platforms[parent_index].surface.right;
                let spike_rotation = self.platform_dynamics[parent_index].body.orientation
                    + self.spike_attacks[spike_index].relative_rotation;
                let spike_right = Vec2::angle(spike_rotation);
                let spike_center = self.platform_dynamics[parent_index].composite_position
                    + self.spike_attacks[spike_index]
                        .relative_center
                        .complex_mul(parent_basis);
                let spike_root =
                    spike_center - data.spike.attack_collider_offset.complex_mul(spike_right);
                let direction = (contact.point - spike_root).normalized_safe();
                let wall_arm =
                    (contact.point - self.platforms[wall_index].surface.center).normalized_safe();
                self.platform_dynamics[wall_index]
                    .body
                    .apply_impulse(direction * data.spike.knock_wall, wall_arm);
                if pushed_spike != spike_index {
                    let self_arm = (contact.point - spike_root).normalized_safe();
                    self.platform_dynamics[parent_index].body.apply_impulse(
                        -direction * data.spike.knock_wall * data.spike.self_push_multiplier,
                        self_arm,
                    );
                    pushed_spike = spike_index;
                }
            }
        }

        // PlayerCollision kills a grounded slime when a different wall's
        // rounded rectangle contains its squish point (the stock player
        // prefab uses the body centre). Player boxes are triggers, so this is
        // a callback effect rather than a rigid-body response.
        let callback_colliders = &scratch.callback_colliders;
        scratch.squished.clear();
        scratch.squished.extend(
            self.players
                .iter()
                .enumerate()
                .filter(|(_, slot)| self.playable_slot(slot) && slot.base.grounded)
                .filter(|(_, slot)| {
                    callback_colliders.iter().enumerate().any(|(i, &collider)| {
                        Some(self.platforms[i].id) != slot.base.ground
                            && collider.overlaps(slot.base.position)
                    })
                })
                .map(|(index, _)| index),
        );
        for &index in &scratch.squished {
            self.kill(index, ENVIRONMENT, CAUSE_SQUISHED, true);
        }
        self.scratch = scratch;

        // Riders follow in game_tick's late phase. PlatformTransform itself is
        // intentionally not synchronized: its UpdateSim copied the platform's
        // FixTransform before DetPhysics moved/corrected it, matching Unity's
        // phase order.

        // The Platform ability object has DestroyIfOutsideSceneBounds, so its
        // transformed owner is handled by deaths_tick using the position it
        // copied before physics. The separate ResizablePlatform prefab has no
        // boundary component: it remains a collidable body after that death.
    }

    fn playable_slot(&self, slot: &PlayerSlot) -> bool {
        slot.base.alive && slot.respawn_ticks == 0
    }

    /// `by_callback`: the death came from a DetPhysics collision callback
    /// (after this pass's contacts) rather than an updatable.
    pub fn kill(&mut self, index: usize, killer: u32, cause: u32, by_callback: bool) {
        let players = self.players.len();
        let revival_marker = self.players[index]
            .revival_markers
            .iter()
            .flatten()
            .copied()
            .find(|&marker| self.revival_marker_pool[marker].active);
        let slot = &mut self.players[index];
        slot.base.alive = false;
        slot.deaths += 1;
        slot.cause_of_death = cause;
        // Telemetry mirrors RlTelemetry.Death.
        if killer == ENVIRONMENT {
            slot.telemetry.deaths_environment += 1;
        } else if killer == slot.base.id {
            slot.telemetry.deaths_self += 1;
        } else if killer >= 1 && killer <= players as u32 {
            slot.telemetry.deaths_opponent += 1;
        } else {
            slot.telemetry.deaths_other += 1;
        }
        // SlimeController.DropAbilities reads the slime body. A death inside
        // an ability first moves that body to the ability object's position
        // (PlayerCollision.cs:124-131, SlimeController.cs:215-219).
        let drop_position = slot.base.position;
        if slot.active != ActiveAbility::None {
            slot.reported_position = drop_position;
            slot.reported_tick = self.tick;
        }
        slot.active = ActiveAbility::None;
        slot.base.dash.active = false;
        // Teleport's marker reference belongs to the destroyed ability
        // instance. The independent indicator survives, but a replacement
        // slime receives a fresh ability with no route back to that marker.
        slot.teleport_markers = [None; 3];
        slot.spike_attacks = [None; 3];
        if killer >= 1 && killer <= players as u32 {
            self.players[killer as usize - 1].kills += 1;
        }
        let slot = &mut self.players[index];
        if let Some(marker_index) = revival_marker {
            // The ordinary death effect is instantiated before SpawnClone and
            // consumes two hierarchy entries despite having no collider.
            self.next_spawn_hierarchy += data::get().revival.death_effect_updatable_count;
            let rows = [
                slot.body_entity,
                slot.dash_entity,
                slot.circle_entity,
                slot.bow_entity,
                slot.throw_entity,
                slot.mine_entity,
                slot.gust_entity,
                slot.revival_entity,
                slot.spike_entity,
                slot.time_stop_entity,
            ];
            for entity in rows {
                if entity == usize::MAX {
                    continue;
                }
                let row = &mut self.entities[entity];
                if by_callback {
                    row.flags = (row.flags & E_TRIGGER) | E_DESTROYED;
                    row.owner = 0;
                } else {
                    row.flags = E_REMOVED;
                }
            }
            let target = self.revival_marker_pool[marker_index].position;
            let distance = (target - drop_position).length();
            let particle_step = data::get().revival.particle_speed * DT * Fix::from_int(2);
            let particle_ticks =
                ((distance.raw() + particle_step.raw() - 1) / particle_step.raw()) as u32;
            slot.base.alive = true;
            slot.base.velocity = Vec2::ZERO;
            slot.base.external_velocity = Vec2::ZERO;
            slot.base.grounded = false;
            slot.base.ground = None;
            slot.base.grounded_speed = Fix::ZERO;
            slot.cause_of_death = CAUSE_NOT_DEAD_YET;
            slot.revival_markers = [None; 3];
            slot.revival_clone_pending = true;
            slot.revival_target_marker = Some(marker_index);
            slot.revival_effect_ticks = data::get().revival.effect_wisp_ticks + 1 + particle_ticks;
            slot.revival_particle_ticks = particle_ticks;
            slot.respawn_ticks = 0;
            slot.respawn_spawn_ticks = 0;
            slot.obs_state = STATE_NOTHING;
            return;
        }
        // Stocks: spawn the replacement clone at the chooser's safe center
        // point. Since Player.Kill sees that clone, it does not mark the
        // player eliminated and leaves CauseOfDeath at NotDeadYet.
        if slot.stocks > 0 && !self.sudden_death {
            if by_callback {
                self.next_spawn_hierarchy += data::get().revival.death_effect_updatable_count;
            }
            for entity in [
                slot.body_entity,
                slot.dash_entity,
                slot.circle_entity,
                slot.bow_entity,
                slot.throw_entity,
                slot.mine_entity,
                slot.gust_entity,
                slot.revival_entity,
                slot.spike_entity,
                slot.time_stop_entity,
            ] {
                if entity != usize::MAX {
                    let row = &mut self.entities[entity];
                    row.flags = if by_callback {
                        (row.flags & E_TRIGGER) | E_DESTROYED
                    } else {
                        E_REMOVED
                    };
                    row.owner = 0;
                }
            }
            slot.stocks -= 1;
            slot.stock_clone_pending = true;
            slot.respawn_ticks = 360;
            slot.respawn_spawn_ticks = 0;
            slot.respawn_selected = 1;
            slot.base.alive = true;
            slot.base.position = data::get().maps[self.map].respawn_points[1];
            slot.base.velocity = Vec2::ZERO;
            slot.base.external_velocity = Vec2::ZERO;
            slot.base.rotation = Fix::ZERO;
            slot.base.grounded = false;
            slot.base.ground = None;
            slot.base.grounded_speed = Fix::ZERO;
            slot.obs_state = STATE_NOTHING;
            slot.cause_of_death = CAUSE_NOT_DEAD_YET;
            slot.timers.fill(Fix::from_f32(-1.5));
        } else {
            slot.out = true;
            // PlayerCollision destroys the slime and its ability objects
            // (PlayerCollision.cs:201-210). DetPhysics drops updatable-phase
            // casualties before this pass's contacts; callback casualties are
            // observed once as destroyed rows without an owner and dropped at
            // the next pass (DetPhysics.Simulate's remove-destroyed step).
            let rows = [
                slot.body_entity,
                slot.dash_entity,
                slot.circle_entity,
                slot.bow_entity,
                slot.throw_entity,
                slot.mine_entity,
                slot.gust_entity,
                slot.revival_entity,
                slot.spike_entity,
                slot.time_stop_entity,
            ];
            for e in rows {
                if e == usize::MAX {
                    continue;
                }
                let row = &mut self.entities[e];
                if by_callback {
                    row.flags = (row.flags & E_TRIGGER) | E_DESTROYED;
                    row.owner = 0;
                } else {
                    row.flags = E_REMOVED;
                }
            }
            // A final death drops the last ability as a DynamicAbilityPickup
            // at the slime body (SlimeController.cs:274-310). Its circle is
            // registered at once with the collider disabled for
            // `disabledSeconds` (DynamicAbilityPickup.cs:147-155), so
            // DetPhysics neither integrates the launch impulse nor moves it
            // while this match still runs. The pickup's later flight, swaps
            // and the RandomFix launch draw are not simulated: with two
            // players the drop ends the match.
            let pickup = &data::get().pickup;
            let id = self.next_id();
            let position = drop_position + pickup.offset;
            self.entities.push(CEntity {
                id,
                kind: 0,
                layer: pickup.layer,
                owner: 0,
                flags: E_ACTIVE,
                hierarchy: 300_000_000 + id,
                x: position.x.raw(),
                y: position.y.raw(),
                radius: pickup.radius.raw(),
                inverse_mass: pickup.inverse_mass.raw(),
                ..CEntity::default()
            });
        }
    }

    fn respawn_tick(
        &mut self,
        index: usize,
        input_direction: Vec2,
        jump_pressed: bool,
        data: &data::RolloutData,
    ) {
        let slot = &mut self.players[index];
        if !slot.base.alive {
            return;
        }

        if slot.unspawned > 0 {
            // GameSessionHandler.SpawnPlayers instantiates the slime at its
            // team spawn on SLIME_AWAKE_TICK+1 and BeforeSpawnInit starts the
            // spawn animation with physics off (SlimeController.cs:516-533).
            for timer in &mut slot.timers {
                *timer += DT;
            }
            slot.unspawned -= 1;
            if slot.unspawned == 0 {
                let spawns = &data.maps[self.map].team_spawns;
                slot.base.position = spawns[slot.team as usize % spawns.len()];
                slot.reported_position = slot.base.position;
                slot.respawn_spawn_ticks = data.player.spawn_anim_ticks;
            }
            return;
        }

        if slot.respawn_ticks > 0 {
            // StockRespawnChooser.UpdateSim: choose a point from the full
            // directional input, then decrement and allow a jump edge to
            // confirm immediately.
            let threshold = Fix::from_f32(0.3);
            if input_direction.x < -threshold {
                slot.respawn_selected = 0;
            } else if input_direction.x > threshold {
                slot.respawn_selected = 2;
            } else if input_direction.y.abs() > threshold {
                slot.respawn_selected = 1;
            }
            slot.respawn_ticks -= 1;
            if slot.respawn_ticks == 0 || jump_pressed {
                slot.respawn_ticks = 0;
                for entity in &mut slot.stock_indicator_entities {
                    if *entity != usize::MAX {
                        self.entities[*entity].flags = E_REMOVED;
                        *entity = usize::MAX;
                    }
                }
                slot.respawn_spawn_ticks = data.player.fast_spawn_anim_ticks;
                slot.base.position = data.maps[self.map].respawn_points[slot.respawn_selected];
                // The chooser is ordered before its replacement slime. Its
                // ReSpawnFast call moves the body, then that slime's OldUpdate
                // publishes the selected position in this same pass.
                slot.reported_position = slot.base.position;
                slot.reported_tick = self.tick;
                slot.obs_state = STATE_FALL;
                // The replacement slime runs later than its chooser in this
                // Updater pass and advances its fresh cooldowns immediately.
                for timer in &mut slot.timers {
                    *timer += DT;
                }
            }
            return;
        }

        if slot.respawn_spawn_ticks > 0 {
            // OldUpdate continues cooldowns during ReSpawnFast's suppressed
            // animation, but movement and ability input remain disabled.
            for timer in &mut slot.timers {
                *timer += DT;
            }
            slot.respawn_spawn_ticks -= 1;
            if slot.respawn_spawn_ticks == 0 {
                // SlimeController.Spawn pops the clone upward at 1/4 jump
                // strength, then reenables simulation and input.
                slot.base.velocity =
                    Vec2::UP * data.player.constants.jump_strength * Fix::from_f32(0.25);
                slot.base.external_velocity = Vec2::ZERO;
                slot.base.grounded = false;
                slot.base.ground = None;
                slot.base.grounded_speed = Fix::ZERO;
            }
        }
    }

    /// GameSessionHandler.UpdateSim's winner check. Its authored hierarchy is
    /// before player components and physics, so it consumes the previous
    /// tick's deaths. The rest of the current updater pass still runs after it
    /// marks the game ended; RlBridge observes terminal only after the pass.
    fn session_result_tick(&mut self) {
        // Distinct teams still in, on the stack: this runs every tick.
        let mut alive_teams = [0u32; MAX_PLAYERS];
        let mut alive_count = 0;
        for p in self.players.iter().filter(|p| !p.out) {
            if !alive_teams[..alive_count].contains(&p.team) {
                alive_teams[alive_count] = p.team;
                alive_count += 1;
            }
        }
        let alive_teams = &alive_teams[..alive_count];
        if self.players.len() > 1 && alive_teams.len() <= 1 {
            self.terminal = true;
            if alive_teams.len() == 1 {
                let winner = alive_teams[0];
                for p in &mut self.players {
                    if p.team == winner && !p.out {
                        p.won = true;
                    }
                }
            }
        }
    }

    /// Remaining post-physics session work: sudden-death activation and the
    /// RL episode limit. Winner resolution belongs to `session_result_tick`.
    fn session_tick(&mut self) {
        let map = &data::get().maps[self.map];
        let sim_time = Fix::from_int(i64::from(self.tick)) * DT;
        if !self.sudden_death && sim_time > map.sudden_death_time {
            self.sudden_death = true;
            self.sudden_death_tick = self.tick;
            // On the following Updater pass each AnimateVelocity notices the
            // global flag and samples in registration order: authored bodies,
            // then live ability-spawned bodies.
            for (delay, platform) in self.sudden_death_delays.iter_mut().zip(&map.platforms) {
                *delay = self
                    .rng
                    .fix_range(Fix::ZERO, platform.sudden_death_max_delay);
            }
            let spawned_max_delay = data::get().platform.sudden_death_max_delay;
            for spawn in &mut self.spawned {
                if spawn.active {
                    spawn.sudden_death_delay = self.rng.fix_range(Fix::ZERO, spawned_max_delay);
                }
            }
        }
        if self.episode_ticks >= self.limit {
            self.terminal = true;
        }
    }

    // ---- Observation ----

    fn observe_main(&mut self) {
        self.refresh_entities();
        Self::publish_entities(&mut self.state_entities, &self.entities);
        let mut slimes = std::mem::take(&mut self.slimes);
        self.fill_slimes(&mut slimes);
        self.slimes = slimes;
        let flags = self.flags();
        Self::write_state(
            &mut self.state,
            &self.players,
            &self.time_stop_protected,
            &self.state_entities,
            &self.slimes,
            self.tick,
            self.episode_ticks,
            self.warmup,
            flags,
            self.player_time_scale,
        );
        self.state.entities = self.state_entities.as_ptr();
        self.state.slimes = self.slimes.as_ptr();
    }

    fn observe_sample(&mut self) {
        self.refresh_entities();
        Self::publish_entities(&mut self.sample_entities, &self.entities);
        let mut slimes = std::mem::take(&mut self.sample_slimes);
        self.fill_slimes(&mut slimes);
        self.sample_slimes = slimes;
        let flags = self.flags();
        Self::write_state(
            &mut self.sample,
            &self.players,
            &self.time_stop_protected,
            &self.sample_entities,
            &self.sample_slimes,
            self.tick,
            self.episode_ticks,
            self.warmup,
            flags,
            self.player_time_scale,
        );
        self.sample.entities = self.sample_entities.as_ptr();
        self.sample.slimes = self.sample_slimes.as_ptr();
    }

    /// DetPhysics stores one list per shape and observations enumerate
    /// circles, boxes, then rounded rectangles. Reuse the output allocation
    /// while preserving registration order inside each shape group.
    fn publish_entities(out: &mut Vec<CEntity>, entities: &[CEntity]) {
        out.clear();
        for kind in 0..=2 {
            out.extend(
                entities
                    .iter()
                    .filter(|e| e.kind == kind && e.flags & E_REMOVED == 0),
            );
        }
    }

    fn flags(&self) -> u32 {
        let mut flags = 0;
        if self.terminal {
            flags |= FLAG_TERMINAL;
        }
        if self.episode_ticks >= self.limit {
            flags |= FLAG_TRUNCATED;
        }
        if self.started {
            flags |= FLAG_STARTED;
        }
        flags
    }

    /// Update the entity table from live state (positions, flags).
    fn refresh_entities(&mut self) {
        self.refresh_actor_entities();
        self.refresh_spike_entities();
        self.refresh_platform_entities();
    }

    fn refresh_spike_entities(&mut self) {
        let data = data::get();
        for spike in &self.spike_attacks {
            if !spike.active || spike.platform_index >= self.platforms.len() {
                continue;
            }
            let Some(entity) = spike.entity else {
                continue;
            };
            let platform = &self.platforms[spike.platform_index];
            let parent = &self.platform_dynamics[spike.platform_index].body;
            let rotation = parent.orientation + spike.relative_rotation;
            let basis = Vec2::angle(rotation);
            let center = self.platform_dynamics[spike.platform_index].composite_position
                + spike.relative_center.complex_mul(platform.surface.right);
            let extents = data::surface_extents(data.spike.attack_extents);
            self.entities[entity] = CEntity {
                id: self.entities[entity].id,
                kind: 1,
                layer: data.spike.attack_layer,
                owner: 0,
                flags: E_ACTIVE | E_ENABLED | E_COMPOSITE,
                hierarchy: self.entities[entity].hierarchy,
                x: center.x.raw(),
                y: center.y.raw(),
                right_x: (basis * extents.x).x.raw(),
                right_y: (basis * extents.x).y.raw(),
                up_x: (basis.perp() * extents.y).x.raw(),
                up_y: (basis.perp() * extents.y).y.raw(),
                rotation: rotation.raw(),
                inverse_mass: spike.body.inverse_mass.raw(),
                ..CEntity::default()
            };
        }
    }

    /// Player, ability and arrow rows as DetPhysics syncs them. An inactive
    /// GameObject's FixTransform is frozen, and DetPhysics still copies it
    /// into the collider every pass (DetPhysics.cs:2498-2530), so inactive
    /// rows keep the position of their last active pass.
    fn refresh_actor_entities(&mut self) {
        let data = data::get();
        for index in 0..self.players.len() {
            let (
                body_e,
                dash_e,
                circle_e,
                bow_e,
                throw_e,
                mine_e,
                gust_e,
                revival_e,
                spike_e,
                time_stop_e,
            ) = {
                let s = &self.players[index];
                (
                    s.body_entity,
                    s.dash_entity,
                    s.circle_entity,
                    s.bow_entity,
                    s.throw_entity,
                    s.mine_entity,
                    s.gust_entity,
                    s.revival_entity,
                    s.spike_entity,
                    s.time_stop_entity,
                )
            };
            let slot = &self.players[index];
            let alive = slot.base.alive && slot.unspawned == 0;
            let choosing_respawn = slot.respawn_ticks > 0 || slot.revival_effect_ticks > 0;
            let teleporting = slot.teleport_effect_active;
            let dashing = slot.base.dash.active;
            let bowing = slot.active == ActiveAbility::Bow;
            let missile_form = slot.active == ActiveAbility::Missile;
            let mining = slot.active == ActiveAbility::Mine;
            let throwing = matches!(
                slot.active,
                ActiveAbility::Grenade
                    | ActiveAbility::Smoke
                    | ActiveAbility::Tack
                    | ActiveAbility::Mjolnir
            );
            let casting_time_stop = slot.active == ActiveAbility::TimeStop;
            let spiking = slot.active == ActiveAbility::Spike;
            let throw_config = throw_config(data::get(), active_throw_ability(slot.active));
            let in_ability = slot.active != ActiveAbility::None;
            let anticipating = slot.active == ActiveAbility::Platform
                && slot.platform.phase == PLATFORM_ANTICIPATION;
            let bow_aim = slot.bow.aim;
            // Ability.ExitAbility deactivates the ability body where it
            // ended (Ability.cs:151-152) and DetPhysics still syncs that
            // frozen FixTransform every pass (DetPhysics.cs:2498-2530), so
            // the exit pass writes the published exit position.
            let exit_position = slot.reported_position;
            let dash_exit_rotation = slot.base.dash.exit_rotation;
            let data = data::get();

            if slot.out || slot.revival_clone_pending || slot.stock_clone_pending {
                continue; // rows were settled by `kill`
            }
            let body = &mut self.entities[body_e];
            let body_was_active = body.flags & E_ACTIVE != 0;
            if !alive {
                body.flags = E_ENABLED | E_TRIGGER | E_DESTROYED;
            } else if teleporting {
                body.flags = E_ENABLED | E_TRIGGER;
            } else if choosing_respawn {
                body.flags = E_ACTIVE | E_TRIGGER;
            } else if dashing || in_ability {
                body.flags = E_ENABLED | E_TRIGGER;
            } else {
                body.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
            }
            // The observation reads the PhysicsBody DetPhysics synced, not
            // the live body (RlObservation.Collider; PlayerPhysics.cs:138-146
            // moves a grounded body after the sync).
            let synced = slot.base.collider.center;
            let rotation = slot.base.collider_rotation;
            let body_active = body.flags & E_ACTIVE != 0;
            // onEnterAbility deactivates the slime after this pass's Move
            // (SlimeController.cs:1084); its frozen FixTransform is what the
            // hurtbox shows for the rest of the ability.
            let (center, angle) = if body_active || teleporting {
                (synced, rotation)
            } else {
                (slot.ability_origin, slot.ability_origin_rotation)
            };
            if body_active || body_was_active {
                body.x = center.x.raw();
                body.y = center.y.raw();
                // The hurtbox has PlayerBody movement on the same GameObject
                // but no BoplBody. DPhysicsBox therefore registers an
                // infinite-mass trigger PhysicsBody whose observed velocity
                // remains zero; the moving velocity is exposed by the
                // separate slime record.
                body.vx = 0;
                body.vy = 0;
                // Box.SetRotation: axis length `One / inverseExtents`, not
                // `extents`. An active row is the synced Player-layer box
                // itself (the slime is never dashing while its row is active).
                // By reference: a by-value copy of the 64-byte box followed
                // by a vector reload of its axes stalls on store forwarding.
                let rebuilt;
                let hurtbox: &OrientedBox = if body_active {
                    &slot.base.collider
                } else {
                    rebuilt = OrientedBox::from_basis(
                        center,
                        Vec2::angle(angle),
                        data.player.hurtbox_extents,
                    );
                    &rebuilt
                };
                body.right_x = hurtbox.right.x.raw();
                body.right_y = hurtbox.right.y.raw();
                body.up_x = hurtbox.up.x.raw();
                body.up_y = hurtbox.up.y.raw();
                body.rotation = angle.raw();
            }

            if dash_e != usize::MAX {
                let dash_box = &mut self.entities[dash_e];
                let dash_was_active = dash_box.flags & E_ACTIVE != 0;
                if alive && dashing {
                    dash_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                    dash_box.owner = slot.base.id;
                } else {
                    dash_box.flags = E_ENABLED | E_TRIGGER;
                }
                // Ability.ExitAbility deactivates the body where it ended
                // (Ability.cs:151-152); that pass still syncs it.
                if (alive && dashing) || dash_was_active {
                    // The Dash body's FixTransform rotation is
                    // Tools.GetRotationFromDirection (Dash.cs:123-124, zero
                    // before performDash); `sync_collider` already built the
                    // synced box with the dash extents from that rotation.
                    // On the exit pass the slime body has taken over, so the
                    // box is rebuilt at the teleport/landing pose.
                    let (center, angle) = if alive && dashing {
                        (synced, rotation)
                    } else {
                        (exit_position, dash_exit_rotation)
                    };
                    let collider = OrientedBox::from_basis(
                        center,
                        Vec2::angle(angle),
                        data.dash.hurtbox_extents,
                    );
                    dash_box.x = center.x.raw();
                    dash_box.y = center.y.raw();
                    // DPhysicsBox is a trigger body; PlayerBody velocity is not
                    // stored on its PhysicsBody observation record.
                    dash_box.vx = 0;
                    dash_box.vy = 0;
                    dash_box.right_x = collider.right.x.raw();
                    dash_box.right_y = collider.right.y.raw();
                    dash_box.up_x = collider.up.x.raw();
                    dash_box.up_y = collider.up.y.raw();
                    dash_box.rotation = angle.raw();
                }
            }
            // Bow hurtbox: active while aiming/firing. BowTransform writes
            // `body.up = inputVector` each tick until the shot
            // (BowTransform.cs:246; the Bow prefab has SyncRotationToPlatform
            // off), so the FixTransform rotation is the aim's angle
            // (FixTransform.cs:56-59) and DetPhysics rebuilds the box axes
            // from that angle (Box.SetRotation, Box.cs:30-35). Its centre is
            // the DetPhysics-synced position: a grounded Bow body also moves
            // only in PlayerPhysics.LateUpdateSim, after the sync.
            if bow_e != usize::MAX {
                let bow_box = &mut self.entities[bow_e];
                let bow_was_active = bow_box.flags & E_ACTIVE != 0;
                if alive && bowing {
                    bow_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                    bow_box.owner = slot.base.id;
                } else {
                    bow_box.flags = E_ENABLED | E_TRIGGER;
                }
                if (alive && bowing) || bow_was_active {
                    let extents = data.bow.hurtbox_extents;
                    // Both exits zero the Bow body's rotation before
                    // ExitAbility (BowTransform.cs:119, 225).
                    let (center, rotation) = if alive && bowing {
                        (synced, bow_aim.perp_cw().normalized_angle())
                    } else {
                        (exit_position, Fix::ZERO)
                    };
                    let hurtbox = OrientedBox::from_basis(center, Vec2::angle(rotation), extents);
                    bow_box.x = center.x.raw();
                    bow_box.y = center.y.raw();
                    bow_box.vx = 0;
                    bow_box.vy = 0;
                    bow_box.right_x = hurtbox.right.x.raw();
                    bow_box.right_y = hurtbox.right.y.raw();
                    bow_box.up_x = hurtbox.up.x.raw();
                    bow_box.up_y = hurtbox.up.y.raw();
                    bow_box.rotation = rotation.raw();
                }
            }
            if throw_e != usize::MAX {
                let throw_box = &mut self.entities[throw_e];
                let was_active = throw_box.flags & E_ACTIVE != 0;
                let missile_row = missile_form || (was_active && slot.missile.projectile.is_some());
                if alive && (throwing || missile_form) {
                    throw_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                    throw_box.owner = slot.base.id;
                } else {
                    throw_box.flags = E_ENABLED | E_TRIGGER;
                }
                if (alive && (throwing || missile_form)) || was_active {
                    let (center, rotation) = if missile_form {
                        (synced, slot.base.rotation)
                    } else if throwing {
                        // `Fire` applies the recoil before this pass's
                        // grounded `PlayerPhysics.Move`. The ability body
                        // advances once by that velocity while the original
                        // slime body remains frozen at its pre-cast pose.
                        let center = if slot.throw_item.grounded && slot.throw_item.has_fired {
                            synced - slot.throw_item.direction * throw_config.knockback * DT
                        } else {
                            synced
                        };
                        (center, slot.base.rotation)
                    } else {
                        (exit_position, slot.base.rotation)
                    };
                    let hurtbox = OrientedBox::from_basis(
                        center,
                        Vec2::angle(rotation),
                        if missile_row {
                            data.missile.hurtbox_extents
                        } else {
                            throw_config.hurtbox_extents
                        },
                    );
                    throw_box.x = center.x.raw();
                    throw_box.y = center.y.raw();
                    throw_box.vx = 0;
                    throw_box.vy = 0;
                    throw_box.right_x = hurtbox.right.x.raw();
                    throw_box.right_y = hurtbox.right.y.raw();
                    throw_box.up_x = hurtbox.up.x.raw();
                    throw_box.up_y = hurtbox.up.y.raw();
                    throw_box.rotation = rotation.raw();
                }
            }
            if mine_e != usize::MAX {
                let mine_box = &mut self.entities[mine_e];
                let was_active = mine_box.flags & E_ACTIVE != 0;
                if alive && mining {
                    mine_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                    mine_box.owner = slot.base.id;
                } else {
                    mine_box.flags = E_ENABLED | E_TRIGGER;
                }
                if (alive && mining) || was_active {
                    let root = if mining {
                        slot.base.position
                    } else {
                        exit_position
                    };
                    let rotation = slot.base.rotation;
                    let basis = Vec2::angle(rotation);
                    let center = root + data.mine.collider_offset.complex_mul(basis);
                    let hurtbox = OrientedBox::from_basis(
                        center,
                        basis,
                        data::surface_extents(data.mine.hurtbox_extents),
                    );
                    mine_box.x = center.x.raw();
                    mine_box.y = center.y.raw();
                    mine_box.vx = 0;
                    mine_box.vy = 0;
                    mine_box.right_x = hurtbox.right.x.raw();
                    mine_box.right_y = hurtbox.right.y.raw();
                    mine_box.up_x = hurtbox.up.x.raw();
                    mine_box.up_y = hurtbox.up.y.raw();
                    mine_box.rotation = rotation.raw();
                }
            }
            // The Platform ability's hurtbox circle is live during
            // anticipation only (hurtBox.enabled until CreatePlatform).
            if circle_e != usize::MAX {
                let circle = &mut self.entities[circle_e];
                let circle_was_active = circle.flags & E_ACTIVE != 0;
                let transforming = alive && slot.active == ActiveAbility::Platform;
                if alive && anticipating {
                    circle.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                    circle.owner = slot.base.id;
                } else if transforming {
                    circle.flags = E_ACTIVE | E_TRIGGER;
                    circle.owner = slot.base.id;
                } else {
                    circle.flags = if slot.platform.hurtbox_disabled {
                        E_TRIGGER
                    } else {
                        E_ENABLED | E_TRIGGER
                    };
                }
                if transforming || circle_was_active {
                    // DetPhysics copies the ability body's FixTransform rotation
                    // (body.up = Slerp(...), PlatformTransform.cs:281; zero again
                    // in ExitAbility, PlatformTransform.cs:462, at the ejected
                    // position the exit pass published).
                    let (center, rotation) = if transforming {
                        (slot.base.position, slot.base.rotation)
                    } else {
                        (exit_position, Fix::ZERO)
                    };
                    circle.x = center.x.raw();
                    circle.y = center.y.raw();
                    circle.rotation = rotation.raw();
                }
            }
            // Gust's QuickSpell body is a circle with the FixTransform's
            // authored offset. It remains still through both animations.
            if gust_e != usize::MAX {
                let circle = &mut self.entities[gust_e];
                let was_active = circle.flags & E_ACTIVE != 0;
                let gusting = alive && slot.active == ActiveAbility::Gust;
                circle.flags = if gusting {
                    E_ACTIVE | E_ENABLED | E_TRIGGER
                } else {
                    E_ENABLED | E_TRIGGER
                };
                if gusting {
                    circle.owner = slot.base.id;
                }
                if gusting || was_active {
                    let (position, rotation) = if gusting {
                        (slot.base.position, slot.base.rotation)
                    } else {
                        (exit_position, Fix::ZERO)
                    };
                    let offset = Vec2::angle(rotation).complex_mul(data.gust.collider_offset);
                    circle.x = (position + offset).x.raw();
                    circle.y = (position + offset).y.raw();
                    circle.rotation = rotation.raw();
                }
            }
            if revival_e != usize::MAX {
                let circle = &mut self.entities[revival_e];
                let was_active = circle.flags & E_ACTIVE != 0;
                let reviving = alive && slot.active == ActiveAbility::Revival;
                circle.flags = if reviving {
                    E_ACTIVE | E_ENABLED | E_TRIGGER
                } else {
                    E_ENABLED | E_TRIGGER
                };
                if reviving {
                    circle.owner = slot.base.id;
                }
                if reviving || was_active {
                    let (position, rotation) = if reviving {
                        (slot.base.position, slot.base.rotation)
                    } else {
                        (exit_position, Fix::ZERO)
                    };
                    let offset = Vec2::angle(rotation).complex_mul(data.revival.collider_offset);
                    circle.x = (position + offset).x.raw();
                    circle.y = (position + offset).y.raw();
                    circle.rotation = rotation.raw();
                }
            }
            if spike_e != usize::MAX {
                let box_row = &mut self.entities[spike_e];
                let was_active = box_row.flags & E_ACTIVE != 0;
                box_row.flags = if alive && spiking {
                    E_ACTIVE | E_ENABLED | E_TRIGGER
                } else {
                    E_ENABLED | E_TRIGGER
                };
                if spiking {
                    box_row.owner = slot.base.id;
                }
                if spiking || was_active {
                    let (position, rotation, center_basis, collider_basis) = if spiking {
                        (
                            slot.spike_cast.collider_position,
                            slot.spike_cast.collider_rotation,
                            slot.spike_cast.collider_up.perp_cw(),
                            Vec2::angle(slot.spike_cast.collider_rotation),
                        )
                    } else {
                        (exit_position, Fix::ZERO, Vec2::RIGHT, Vec2::RIGHT)
                    };
                    let center = position + center_basis.complex_mul(data.spike.collider_offset);
                    let collider =
                        OrientedBox::from_basis(center, collider_basis, data.spike.hurtbox_extents);
                    box_row.x = center.x.raw();
                    box_row.y = center.y.raw();
                    box_row.right_x = collider.right.x.raw();
                    box_row.right_y = collider.right.y.raw();
                    box_row.up_x = collider.up.x.raw();
                    box_row.up_y = collider.up.y.raw();
                    box_row.rotation = rotation.raw();
                }
            }
            if time_stop_e != usize::MAX {
                let circle = &mut self.entities[time_stop_e];
                let was_active = circle.flags & E_ACTIVE != 0;
                circle.flags = if alive && casting_time_stop {
                    E_ACTIVE | E_ENABLED | E_TRIGGER
                } else {
                    E_ENABLED | E_TRIGGER
                };
                if casting_time_stop {
                    circle.owner = slot.base.id;
                }
                if casting_time_stop || was_active {
                    let (position, rotation, basis) = if casting_time_stop {
                        (
                            slot.base.position,
                            slot.base.rotation,
                            slot.base.up.perp_cw(),
                        )
                    } else {
                        (
                            slot.cast_spell.collider_position,
                            slot.cast_spell.collider_rotation,
                            slot.cast_spell.collider_right,
                        )
                    };
                    // The ability body's FixTransform keeps the exact Slerp
                    // basis copied from PlayerPhysics. Its published angle is
                    // derived from that vector and rebuilding the basis from
                    // the angle loses the source vector's last few Q32 bits.
                    let offset = basis.complex_mul(data.time_stop.collider_offset);
                    circle.x = (position + offset).x.raw();
                    circle.y = (position + offset).y.raw();
                    circle.rotation = rotation.raw();
                }
            }
        }
        for arrow in &self.arrows {
            if !arrow.active || !arrow.registered {
                continue;
            }
            let e = &mut self.entities[arrow.entity.expect("registered arrow entity")];
            e.x = arrow.position.x.raw();
            e.y = arrow.position.y.raw();
            e.vx = arrow.velocity.x.raw();
            e.vy = arrow.velocity.y.raw();
            e.rotation = arrow.rotation.raw();
        }
        for marker in &self.revival_marker_pool {
            if !marker.active || !marker.registered {
                continue;
            }
            let e = &mut self.entities[marker.entity.expect("registered Revival marker")];
            let position = marker.position + data.revival.marker_offset;
            e.x = position.x.raw();
            e.y = position.y.raw();
            e.vx = marker.velocity.x.raw();
            e.vy = marker.velocity.y.raw();
        }
        for marker in &self.teleport_marker_pool {
            if !marker.active || !marker.registered {
                continue;
            }
            let e = &mut self.entities[marker.entity.expect("registered Teleport marker")];
            let position = marker.position + data.teleport.marker_offset;
            e.x = position.x.raw();
            e.y = position.y.raw();
            e.vx = marker.velocity.x.raw();
            e.vy = marker.velocity.y.raw();
        }
    }

    fn refresh_platform_entities(&mut self) {
        // The rows below are pure functions of platform state, which only
        // changes through `platforms_changed`; `remove_platform` marks a
        // removed row itself.
        if self.platform_rows_epoch == self.platform_epoch {
            return;
        }
        self.platform_rows_epoch = self.platform_epoch;
        for (i, platform) in self.platforms.iter().enumerate() {
            let e = &mut self.entities[self.platform_entities[i]];
            let s = &platform.surface;
            let dynamics = &self.platform_dynamics[i];
            e.x = s.center.x.raw();
            e.y = s.center.y.raw();
            let observed_body = if dynamics.attached_spikes == 0 {
                &dynamics.body
            } else {
                &dynamics.root_body
            };
            e.flags = E_ACTIVE
                | E_ENABLED
                | if dynamics.attached_spikes == 0 {
                    0
                } else {
                    E_COMPOSITE
                };
            e.vx = observed_body.velocity.x.raw();
            e.vy = observed_body.velocity.y.raw();
            // RlObservation publishes the DetPhysics collider axes (Box.SetRotation).
            let right = platform.collider.inner.right;
            let up = platform.collider.inner.up;
            e.right_x = right.x.raw();
            e.right_y = right.y.raw();
            e.up_x = up.x.raw();
            e.up_y = up.y.raw();
            e.radius = s.radius.raw();
            // RlObservation publishes PhysicsBody.orientation directly. The
            // collider basis is derived output and must not be converted back
            // into an angle (especially around the Q32 sine-table boundary).
            e.rotation = observed_body.orientation.raw();
            e.angular_velocity = observed_body.angular_velocity.raw();
            e.inverse_mass = observed_body.inverse_mass.raw();
        }
    }

    fn fill_slimes(&self, slimes: &mut Vec<CSlime>) {
        slimes.clear();
        let mut order = [usize::MAX; MAX_PLAYERS];
        let mut count = 0;
        for (index, slot) in self.players.iter().enumerate() {
            if slot.base.alive
                && slot.unspawned == 0
                && !slot.revival_clone_pending
                && !slot.stock_clone_pending
            {
                order[count] = index;
                count += 1;
            }
        }
        // RlObservation walks live SlimeControllers in updater registration
        // order. A Revival clone is registered after the surviving original
        // players, regardless of its lower player id.
        order[..count].sort_unstable_by_key(|&index| {
            self.entities[self.players[index].body_entity].hierarchy
        });
        for &index in &order[..count] {
            let slot = &self.players[index];
            let in_ability = slot.active != ActiveAbility::None;
            let respawning = slot.respawn_ticks > 0
                || slot.respawn_spawn_ticks > 0
                || slot.revival_effect_ticks > 0;
            let mut flags = S_ACTIVE;
            if slot.teleport_effect_active {
                flags = S_SPAWNED | S_SIMULATED;
            } else if respawning {
                flags |= S_SUPPRESS_INPUT;
            } else if slot.rewind.rewinding {
                flags |= S_SPAWNED | S_SUPPRESS_INPUT;
            } else if in_ability {
                // onEnterAbility disables the original slime GameObject but
                // its spawned/simulated state remains visible to observation.
                flags = S_SPAWNED | S_IN_ABILITY | S_SIMULATED;
            } else {
                flags |= S_SPAWNED;
            }
            if !in_ability && !respawning {
                flags |= S_SIMULATED;
            }
            let mut ability_flags = [0u32; 3];
            for (s, flags) in ability_flags.iter_mut().enumerate() {
                if slot.abilities[s] == 0 {
                    continue;
                }
                let mut f = 1u32;
                if slot.timers[s] >= slot.cooldowns[s] {
                    f |= 2;
                }
                if slot.timers[s] > slot.cooldowns[s] {
                    f |= 4;
                }
                if in_ability && slot.active_slot == s {
                    f |= 8;
                    if slot.active == ActiveAbility::Bow {
                        // RlObservation reflects BowTransform.loadingFrame
                        // into bits 4..6 and hasFired into bit 7.
                        f |= (slot.bow.loading_frame & 7) << 4;
                        if slot.bow.has_fired {
                            f |= 0x80;
                        }
                    }
                }
                *flags = f;
            }
            let (slime_position, velocity, rotation, state) = if in_ability {
                (
                    slot.ability_origin,
                    slot.ability_origin_velocity,
                    slot.ability_origin_rotation,
                    slot.ability_origin_state,
                )
            } else {
                (
                    slot.base.position,
                    slot.base.velocity + slot.base.external_velocity,
                    slot.base.rotation,
                    slot.obs_state,
                )
            };
            slimes.push(CSlime {
                id: self.entities[slot.body_entity].id,
                owner: slot.base.id,
                flags,
                state,
                x: slime_position.x.raw(),
                y: slime_position.y.raw(),
                vx: velocity.x.raw(),
                vy: velocity.y.raw(),
                rotation: rotation.raw(),
                scale: 1 << 32,
                abilities: slot.abilities,
                ability_flags,
                cooldowns: [
                    slot.cooldowns[0].raw(),
                    slot.cooldowns[1].raw(),
                    slot.cooldowns[2].raw(),
                ],
                timers: [
                    slot.timers[0].raw(),
                    slot.timers[1].raw(),
                    slot.timers[2].raw(),
                ],
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_state(
        state: &mut CState,
        players: &[PlayerSlot],
        time_stop_protected: &[bool; MAX_PLAYERS],
        entities: &[CEntity],
        slimes: &[CSlime],
        tick: u32,
        episode_ticks: u32,
        warmup: u32,
        flags: u32,
        player_time_scale: Fix,
    ) {
        state.header = CHeader {
            version: ABI_VERSION,
            tick,
            episode_ticks,
            flags,
            player_count: players.len() as u32,
            entity_count: entities.len() as u32,
            warmup_ticks: warmup,
            slime_count: slimes.len() as u32,
            simulation_time: (Fix::from_int(i64::from(tick)) * DT).raw(),
            player_time_scale: player_time_scale.raw(),
        };
        for (i, slot) in players.iter().enumerate() {
            let mut player_flags = 0;
            if slot.base.alive {
                player_flags |= 1;
            }
            if slot.won {
                player_flags |= 2;
            }
            if slot.invisible {
                player_flags |= 4;
            }
            if time_stop_protected[i] {
                player_flags |= 8;
            }
            if slot.active == ActiveAbility::Missile {
                player_flags |= 16;
            }
            if slot.active == ActiveAbility::Platform {
                player_flags |= 32;
            }
            let position = if slot.reported_tick.checked_add(1) == Some(tick) {
                slot.reported_position
            } else {
                slot.base.position
            };
            state.players[i] = CPlayer {
                id: slot.base.id,
                team: slot.team,
                flags: player_flags,
                alive_bodies: u32::from(slot.base.alive),
                stocks: slot.stocks,
                kills: slot.kills,
                deaths: slot.deaths,
                cause_of_death: slot.cause_of_death,
                x: position.x.raw(),
                y: position.y.raw(),
                scale: 1 << 32,
                invisible_time: slot.invisible_time.raw(),
                ability_time: slot.ability_time.raw(),
            };
            state.telemetry[i] = slot.telemetry;
        }
    }
}

impl Default for RolloutWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// `collider_extents` are the authored half-extents DetPhysics holds; the
/// surface's are the perimeter's `One / (One / e)` reading of them.
fn make_platform(id: u32, surface: &Surface, collider_extents: Vec2) -> Platform {
    Platform {
        id,
        surface: surface.clone(),
        collider: RoundedRect {
            inner: OrientedBox::from_basis(surface.center, surface.right, collider_extents),
            radius: surface.radius,
        },
        ice: false,
    }
}

/// The game's sine-weighted `Vec2.Slerp`, including its 0.9999 dot ceiling
/// and zero-sine fallback. PlatformTransform feeds the result back into the
/// next tick without normalizing it, so the vector form is observable near
/// the target and cannot be replaced by scalar-angle interpolation.
fn slerp(from: Vec2, to: Vec2, t: Fix) -> Vec2 {
    let dot = from
        .dot(to)
        .max(-Fix::ONE)
        .min(Fix::ONE)
        .min(Fix::from_f32(0.9999));
    let angle = dot.acos().expect("clamped slerp dot");
    let mut denominator = angle.sin();
    if denominator == Fix::ZERO {
        denominator = Fix::from_f32(0.00001);
    }
    // Vec2.Slerp (Vec2.cs:168-175): each term is scaled by the reciprocal
    // separately, after its own sine product.
    let inverse = Fix::ONE / denominator;
    from * ((Fix::ONE - t) * angle).sin() * inverse + to * (t * angle).sin() * inverse
}

/// DPhysicsRoundedRect.PlatformArea(width, height, radius) with half-extent
/// width/height: 4wr + 4hr + 4wh + pi r^2.
/// DPhysicsRoundedRect.PlatformArea (DPhysicsRoundedRect.cs:621-627), with
/// its factor grouping: each product rounds differently from `4 * w * r`.
fn platform_area(w: Fix, h: Fix, r: Fix) -> Fix {
    let two = Fix::from_int(2);
    let ends = two * h * r;
    let middle = (two * h + two * r) * two * w;
    ends * two + middle + Fix::PI * r * r
}

/// PlatformTransform.GetNewSize: clamp one growing axis to the area cap.
fn new_size(mut w: Fix, mut h: Fix, r: Fix, cap: Fix, min: Fix) -> (Fix, Fix) {
    if platform_area(w, h, r) > cap {
        h = (cap - Fix::from_int(4) * r * w - Fix::PI * r * r)
            / (Fix::from_int(4) * r + Fix::from_int(4) * w);
        if h < min {
            h = min;
            w = (cap - Fix::from_int(4) * h * r - Fix::PI * r * r)
                / (Fix::from_int(4) * h + Fix::from_int(4) * r);
        }
    }
    (w, h)
}

// ---- Replay export (ReplayArchive.Encode: BOPLREP2, little endian) ----

/// 64-hex identity of this backend; the archive format requires SHA256-shaped
/// source and catalog hashes (ReplayArchive.Validate). Replays from the Rust
/// backend are structurally valid but only replayable on builds that skip the
/// hash comparison.
pub const SOURCE_HASH: &[u8; 64] =
    b"c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5c0debab5";
pub const CATALOG_HASH: &[u8; 64] =
    b"ca7a109fca7a109fca7a109fca7a109fca7a109fca7a109fca7a109fca7a109f";

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(u64::try_from(data.len()).unwrap() * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 32];
    for (chunk, word) in digest.chunks_exact_mut(4).zip(h) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

impl RolloutWorld {
    /// Encode the recorded episode as a ReplayArchive (BOPLREP2) into
    /// `replay_buffer`. Mirrors ReplayArchive.Encode byte for byte.
    pub fn encode_replay(&mut self) -> Result<(), String> {
        if !self.valid || self.recording.is_empty() {
            return Err("Replay requires at least one simulated tick".into());
        }
        let config = self.replay_config;
        let players = config.players as usize;
        let slots = (config.slots as usize).max(1);
        let out = &mut self.replay_buffer;
        out.clear();
        out.extend_from_slice(b"BOPLREP2");
        out.extend_from_slice(SOURCE_HASH);
        out.extend_from_slice(CATALOG_HASH);
        out.extend_from_slice(&config.seed.to_le_bytes());
        out.push(config.map as u8);
        out.push(config.mode as u8);
        out.extend_from_slice(&(config.stocks as i32).to_le_bytes());
        out.push(players as u8);
        for (i, p) in config.player[..players].iter().enumerate() {
            out.push(i as u8 + 1);
            out.push(p.color as u8);
            out.push(p.team as u8);
            out.push(p.keyboard as u8);
            out.push(slots as u8);
            for &ability in &p.abilities[..slots] {
                out.extend_from_slice(&(ability as u16).to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.recording.len() as i32).to_le_bytes());
        for (tick, frame) in self.recording.iter().enumerate() {
            for action in &frame[..players] {
                out.extend_from_slice(&(tick as u32 + 1).to_le_bytes());
                out.push(action.angle);
                out.extend_from_slice(&action.buttons.to_le_bytes());
                out.push(0); // targetDelayBufferSize
            }
        }
        let digest = sha256(out);
        self.replay_buffer.extend_from_slice(&digest);
        Ok(())
    }
}

/// Leaves the bow ability: a landing exit (BowTransform.cs:225-239) or the
/// fire animation's end (BowTransform.cs:116-130).
fn bow_exit(slot: &mut PlayerSlot, terrain: Terrain) {
    let platforms = terrain.platforms;
    slot.active = ActiveAbility::None;
    slot.exit_to_fall();
    if slot.base.grounded {
        // Landing exit (BowTransform.cs:225-239): the slime's
        // SyncPhysicsTo(AbilityExitInfo) (PlayerPhysics.cs:556-570)
        // re-derives its surface local from the exit position;
        // LocalPlayerPosFromPos does not invert
        // PositionFromLocalPlayerPos exactly, so the rider shifts.
        let platform = slot
            .base
            .platform(platforms)
            .expect("grounded bow has a platform");
        let radius = slot.base.radius;
        let position = platform.surface.position(slot.base.local, radius);
        slot.base.local = platform
            .surface
            .local_position(position, radius)
            .expect("exit position lies on the platform");
        slot.base.position = position;
        slot.base.velocity = Vec2::ZERO;
        slot.base.align(platform);
        // Ability.ExitAbility publishes the collider position
        // (Ability.cs:151) before the slime's LateUpdateSim slides.
        slot.reported_position = position;
    } else {
        // Animation-end exit (BowTransform.cs:116-130): UnGround
        // nudges the body off any retained ground and zeroes the
        // rotation; the slime then enters fall and immediately
        // raycasts for a landing (SlimeController.cs:251-259), its
        // own attached ground cleared first (PlayerPhysics.cs:577).
        slot.base.force_unground(platforms);
        slot.reported_position = slot.base.position; // Ability.cs:151
        slot.base.ground = None;
        slot.base.collision_rays(terrain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CConfig {
        let player = CPlayerConfig {
            abilities: [ABILITY_PLATFORM, ABILITY_DASH, ABILITY_BOW],
            ..CPlayerConfig::default()
        };
        CConfig {
            seed: 7,
            map: 0,
            players: 2,
            slots: 3,
            stocks: 10,
            warmup_ticks: 240,
            max_episode_ticks: 7200,
            player: [player; MAX_PLAYERS],
            ..CConfig::default()
        }
    }

    #[test]
    fn spawned_platform_uses_authored_rotation_spring() {
        let platform = &data::get().platform;
        assert_eq!(platform.spawn_rotation_speed.raw(), 171_798_691_840);
        assert_eq!(platform.spawn_rotation_mu.raw(), 715_828_032);
    }

    #[test]
    fn resizing_platform_does_not_rederive_body_rotation() {
        let mut world = RolloutWorld::new();
        world.reset(&test_config()).unwrap();
        let index = world.spawn_ability_platform(0, Vec2::new(Fix::from_int(3), Fix::from_int(4)));
        let orientation = Fix::from_f64(3.135_432_668_56);
        let right = Vec2::angle(orientation);
        let up = right.perp();
        world.platform_dynamics[index].body.orientation = orientation;
        world.platform_dynamics[index].home_rotation = orientation;
        world.platforms[index].surface.right = right;
        world.platforms[index].surface.up = up;

        world.reshape_platform(
            index,
            Vec2::new(Fix::from_int(5), Fix::from_int(6)),
            Vec2::new(Fix::from_f64(0.75), Fix::from_f64(3.25)),
            Fix::from_f64(0.25),
        );

        assert_eq!(world.platform_dynamics[index].body.orientation, orientation);
        assert_eq!(world.platform_dynamics[index].home_rotation, orientation);
        assert_eq!(world.platforms[index].surface.right, right);
        assert_eq!(world.platforms[index].surface.up, up);
    }

    #[test]
    fn match_result_consumes_deaths_on_the_following_tick() {
        let mut world = RolloutWorld::new();
        let mut config = test_config();
        config.stocks = 0;
        world.reset(&config).unwrap();

        world.kill(0, ENVIRONMENT, CAUSE_ACCIDENT, true);
        assert!(
            !world.terminal,
            "a collision death must not end its own updater pass"
        );

        world.session_tick();
        assert!(
            !world.terminal,
            "post-physics session work must not observe a same-pass death"
        );

        world.session_result_tick();
        assert!(world.terminal);
        assert!(world.players[1].won);
    }

    #[test]
    fn keyboard_motion_is_independent_of_ability_aim() {
        let action = CAction {
            angle: 65,    // up in the joystick/aim table
            buttons: 512, // D in the keyboard movement bits
            ..CAction::default()
        };
        assert_eq!(action.input_direction(true), Vec2::RIGHT);
        assert_ne!(action.aim_action().direction(), Vec2::RIGHT);
        assert_eq!(
            action.input_direction(false),
            action.aim_action().direction()
        );
    }

    #[test]
    fn destroyed_platform_detaches_rider_on_its_next_update() {
        let mut world = RolloutWorld::new();
        world.reset(&test_config()).unwrap();
        let platform_index =
            world.spawn_ability_platform(0, Vec2::new(Fix::ZERO, Fix::from_int(30)));
        let platform_id = world.platforms[platform_index].id;
        let position = world.platforms[platform_index]
            .surface
            .position(Fix::ZERO, world.players[0].base.radius);
        let normal = world.platforms[platform_index].surface.normal(position);
        {
            let player = &mut world.players[0].base;
            player.position = position;
            player.velocity = Vec2::ZERO;
            player.external_velocity = Vec2::ZERO;
            player.rotation = Fix::from_f32(0.5);
            player.grounded = true;
            player.ground = Some(platform_id);
            player.local = Fix::ZERO;
        }

        world.remove_platform(platform_index);
        assert!(world.players[0].base.grounded);
        assert_eq!(world.players[0].base.ground, Some(platform_id));
        assert_eq!(world.players[0].removed_ground_normal, Some(normal));

        world.player_tick(0, Action::default(), Vec2::ZERO, false, data::get());
        assert!(!world.players[0].base.grounded);
        assert_eq!(world.players[0].base.ground, None);
        assert_eq!(world.players[0].base.rotation, Fix::ZERO);
        assert_eq!(world.players[0].base.velocity, Vec2::ZERO);
        assert_eq!(
            world.players[0].base.position,
            position + normal * Fix::from_f32(0.05)
        );
    }
}
