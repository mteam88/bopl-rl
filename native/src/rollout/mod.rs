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
    physics::{Body, positional_correction, resolve},
    random::Random,
    surface::Surface,
    vector::Vec2,
    world::{Action, DT, DashState, Platform, Player},
};

pub mod abi;
pub mod data;

pub const ABI_VERSION: u32 = 2;
pub const MAP_COUNT: u32 = 5;
/// The .NET catalog has 39 ability icons; index 0 is the menu placeholder.
pub const ABILITY_COUNT: u32 = 39;
pub const MAX_PLAYERS: usize = 6;
/// Catalog indices with real behaviour in this backend.
pub const ABILITY_DASH: u32 = 2;
pub const ABILITY_BOW: u32 = 4;
pub const ABILITY_PLATFORM: u32 = 14;

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
    pub home_position: Vec2,
    pub home_rotation: Fix,
    pub speed: Fix,
    pub mu: Fix,
    pub rotation_speed: Fix,
    pub rotation_mu: Fix,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActiveAbility {
    None,
    Dash,
    Bow,
    Platform,
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
    pub ability_time: Fix,
    /// SlimeController/Ability publishes Player.Position before the live body
    /// advances later in the same updater pass.
    pub reported_position: Vec2,
    pub reported_tick: u32,
    /// First original-slime pass after Dash re-enables it.
    pub dash_exit_followup: bool,
    /// A platform destroyed later in the previous updater pass. Unity keeps
    /// the rider grounded until its own next update, then separates it using
    /// the destroyed StickyRoundedRectangle's last surface normal.
    pub removed_ground_normal: Option<Vec2>,
    pub bow: BowState,
    pub platform: PlatformState,
    /// SlimeController.currentState as the policy observes it. The .NET enum
    /// never takes the `ability` value (nothing assigns it), so the state
    /// freezes at its pre-ability value while an ability runs.
    pub obs_state: u32,
    /// Countdown for the land/stop/turn animation states.
    pub state_ticks: u32,
    pub facing_right: bool,
    /// Respawn countdown while holding a consumed stock (StockRespawn).
    pub respawn_ticks: u32,
    /// ReSpawnFast's accelerated spawn animation after a choice is confirmed.
    pub respawn_spawn_ticks: u32,
    pub respawn_selected: usize,
    /// Player.InputProfile's frame-edge state is updated even without a live
    /// controllable slime and is what the stock chooser reads.
    pub input_jump_was_down: bool,
    pub telemetry: CTelemetry,
    // Fixed entity table indices.
    pub circle_entity: usize,
    pub body_entity: usize,
    pub dash_entity: usize,
    pub bow_entity: usize,
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
    /// Parallel to `platforms`: rigid body and home spring state.
    pub platform_dynamics: Vec<PlatformDynamics>,
    /// Parallel to `platforms`: index into the map's PlatformDef list, or
    /// usize::MAX for spawned (ability) platforms.
    pub platform_defs: Vec<usize>,
    /// Parallel to `platforms`: index into the entity table.
    pub platform_entities: Vec<usize>,
    pub arrows: Vec<Arrow>,
    pub spawned: Vec<SpawnedPlatform>,
    pub sudden_death: bool,
    pub sudden_death_tick: u32,
    pub sudden_death_delays: Vec<Fix>,
    /// AntiLockPlatform runtime (map 2): platform index and waypoint.
    pub orbits: Vec<(usize, usize)>,
    pub next_platform_id: u32,
    pub next_entity_id: u32,
    // Observation buffers. `entities`/`slimes` back the pointers in `state`.
    pub entities: Vec<CEntity>,
    pub slimes: Vec<CSlime>,
    pub state: Box<CState>,
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
            platform_dynamics: Vec::with_capacity(64),
            platform_defs: Vec::with_capacity(64),
            platform_entities: Vec::with_capacity(64),
            arrows: Vec::with_capacity(64),
            spawned: Vec::with_capacity(64),
            sudden_death: false,
            sudden_death_tick: 0,
            sudden_death_delays: Vec::with_capacity(64),
            orbits: Vec::with_capacity(4),
            next_platform_id: 0,
            next_entity_id: 0,
            entities: Vec::with_capacity(192),
            slimes: Vec::with_capacity(MAX_PLAYERS),
            state: empty_state(),
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
        self.platform_dynamics.clone_from(&src.platform_dynamics);
        self.platform_defs.clone_from(&src.platform_defs);
        self.platform_entities.clone_from(&src.platform_entities);
        self.arrows.clone_from(&src.arrows);
        self.spawned.clone_from(&src.spawned);
        self.sudden_death = src.sudden_death;
        self.sudden_death_tick = src.sudden_death_tick;
        self.sudden_death_delays.clone_from(&src.sudden_death_delays);
        self.orbits.clone_from(&src.orbits);
        self.next_platform_id = src.next_platform_id;
        self.next_entity_id = src.next_entity_id;
        self.entities.clone_from(&src.entities);
        self.slimes.clone_from(&src.slimes);
        *self.state = *src.state;
        self.state.entities = self.entities.as_ptr();
        self.state.slimes = self.slimes.as_ptr();
        self.sample_valid = false;
        self.recording.clear();
        self.replay_config = src.replay_config;
        self.replay_buffer.clear();
        self.error = src.error;
        // Exhaustive: a new field must be added to the copy above (or listed
        // here as deliberately skipped) before this compiles again.
        let RolloutWorld {
            valid: _, failed: _, started: _, terminal: _, map: _, stocks: _, warmup: _, limit: _,
            tick: _, episode_ticks: _, rng: _, players: _, platforms: _, platform_dynamics: _,
            platform_defs: _, platform_entities: _, arrows: _, spawned: _, sudden_death: _,
            sudden_death_tick: _, sudden_death_delays: _, orbits: _, next_platform_id: _,
            next_entity_id: _, entities: _, slimes: _, state: _, sample_entities: _,
            sample_slimes: _, sample: _, sample_valid: _, recording: _, replay_config: _,
            replay_buffer: _, error: _,
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
        self.spawned.clear();
        self.players.clear();
        self.platforms.clear();
        self.platform_dynamics.clear();
        self.platform_defs.clear();
        self.platform_entities.clear();
        self.entities.clear();
        self.slimes.clear();

        let def = &data.maps[map];
        // Entity table, in the .NET DetPhysics registration order:
        // circles (one Platform-ability collider per player), boxes (water,
        // then body/dash/bow per player), rounded rects (map platforms).
        // Dynamic entities (arrows, spawned platforms) append after.
        self.next_entity_id = 0;
        for i in 0..players {
            let id = self.next_id();
            self.entities.push(CEntity {
                id,
                kind: 0,
                layer: LAYER_PLAYER,
                owner: 0,
                // The ability GameObject starts inactive. Its collider keeps
                // its trigger body in DetPhysics, but neither activeInHierarchy
                // nor enabled is set; Ability receives its owner on first use.
                flags: E_TRIGGER,
                hierarchy: 100_000_030 + 30 * i as u32,
                radius: data.platform.hurtbox_radius.raw(),
                inverse_mass: -(1 << 32),
                ..CEntity::default()
            });
        }
        let water_id = self.next_id();
        self.entities.push(CEntity {
            id: water_id,
            kind: 1,
            layer: LAYER_WATER,
            owner: 0,
            flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
            hierarchy: 146,
            x: def.water_position.x.raw(),
            y: def.water_position.y.raw(),
            right_x: def.water_extents.x.raw(),
            up_y: def.water_extents.y.raw(),
            inverse_mass: -(1 << 32),
            ..CEntity::default()
        });
        for i in 0..players {
            for (offset, extents, hier) in [
                (0usize, data.player.hurtbox_extents, 100_000_006u32),
                (1, data.dash.hurtbox_extents, 100_000_015),
                (2, data.bow.hurtbox_extents, 100_000_022),
            ] {
                let id = self.next_id();
                let body = offset == 0;
                self.entities.push(CEntity {
                    id,
                    kind: 1,
                    layer: LAYER_PLAYER,
                    owner: if body { i as u32 + 1 } else { 0 },
                    flags: if body {
                        E_ACTIVE | E_ENABLED | E_TRIGGER
                    } else {
                        E_TRIGGER
                    },
                    hierarchy: hier + 30 * i as u32,
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
                hierarchy: 105 + 7 * index as u32,
                inverse_mass: p.body.inverse_mass.raw(),
                ..CEntity::default()
            });
            self.platforms
                .push(make_platform(self.next_platform_id, &p.surface));
            self.platform_dynamics.push(PlatformDynamics {
                body: p.body,
                home_position: p.surface.center,
                home_rotation: p.body.orientation,
                speed: p.speed,
                mu: p.mu,
                rotation_speed: p.rotation_speed,
                rotation_mu: p.rotation_mu,
            });
            self.platform_defs.push(index);
            self.platform_entities.push(entity);
            self.next_platform_id += 1;
            // AnimateVelocity samples this only when sudden death activates.
            self.sudden_death_delays.push(Fix::ZERO);
            if p.orbit.is_some() {
                self.orbits.push((self.platforms.len() - 1, 0));
            }
        }

        // GameSessionHandler.Init samples one randomDelay per authored ground
        // for the level-intro animation. The rollout starts after that cosmetic
        // animation, but must consume the same samples so later gameplay RNG
        // (notably AnimateVelocity's sudden-death delays) stays aligned.
        for _ in &def.platforms {
            let _ = self.rng.fix_range(Fix::ZERO, Fix::ONE);
        }

        // Players at their team spawns (GameSessionHandler.SpawnPlayers: one
        // authored spawn per team slot, wrapped for extra teams).
        for i in 0..players {
            let team = config.player[i].team;
            let spawn = def.team_spawns[team as usize % def.team_spawns.len()];
            let mut abilities = [0u32; 3];
            let mut cooldowns = [Fix::ZERO; 3];
            abilities[..slots].copy_from_slice(&config.player[i].abilities[..slots]);
            for s in 0..slots {
                cooldowns[s] = match abilities[s] {
                    ABILITY_DASH => data.dash.cooldown,
                    ABILITY_BOW => data.bow.cooldown,
                    ABILITY_PLATFORM => data.platform.cooldown,
                    // Unsimulated abilities keep a plausible cooldown so the
                    // ready bit still cycles for the policy.
                    _ => Fix::from_int(4),
                };
            }
            let elapsed_since_awake = self.warmup.saturating_sub(SLIME_AWAKE_TICK);
            let timer0 = TIMER_START + DT * Fix::from_int(i64::from(elapsed_since_awake));
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
                    constants: data.player.constants.clone(),
                    cooldown: Fix::ZERO,
                    dash: DashState::default(),
                    collider: OrientedBox {
                        center: spawn,
                        right: Vec2::RIGHT,
                        up: Vec2::UP,
                        inverse_extents: Vec2::new(Fix::ONE, Fix::ONE),
                    },
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
                ability_time: Fix::ZERO,
                reported_position: spawn,
                reported_tick: u32::MAX,
                dash_exit_followup: false,
                removed_ground_normal: None,
                bow: BowState::default(),
                platform: PlatformState::default(),
                obs_state: STATE_FALL,
                state_ticks: 0,
                facing_right: true,
                respawn_ticks: 0,
                respawn_spawn_ticks: 0,
                respawn_selected: 1,
                input_jump_was_down: false,
                telemetry: CTelemetry::default(),
                circle_entity: i,
                body_entity: players + 1 + 3 * i,
                dash_entity: players + 2 + 3 * i,
                bow_entity: players + 3 + 3 * i,
            });
        }

        // Warmup: the .NET sim animates the level in and plays the spawn
        // animation; players start falling once spawned and are grounded well
        // before a 240-tick warmup ends. Here platforms start at their homes
        // and players simply fall from the spawn points with no input.
        let settle = self.warmup.min(300);
        let no_input = vec![Action::default(); players];
        for _ in 0..settle {
            self.movement_tick(&no_input);
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

    /// One movement-only tick (used for warmup settling).
    fn movement_tick(&mut self, actions: &[Action]) {
        for (index, action) in actions.iter().enumerate() {
            if !self.players[index].base.alive {
                continue;
            }
            self.players[index].base.step(*action, &self.platforms);
        }
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
            || slot.active != ActiveAbility::None
            || slot.base.dash.active
        {
            return;
        }
        if !slot.base.grounded {
            slot.obs_state = STATE_FALL;
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
                // attachToGround: land animation, then idle.
                slot.obs_state = STATE_LAND;
                slot.state_ticks = LAND_TICKS;
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
            self.game_tick(actions, actions);
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
                self.game_tick(actions, actions);
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
    fn game_tick(&mut self, actions: &[CAction], registered_actions: &[CAction]) {
        let data = data::get();
        self.register_pending_arrows();
        // GameSessionHandler (hierarchy 100) runs before every instantiated
        // player/ability prefab (100,000,000+). A death produced by a player
        // component or DetPhysics is therefore visible to the match result
        // check on the following tick, not later in the tick that killed it.
        self.session_result_tick();
        for index in 0..self.players.len() {
            self.players[index].reported_position = self.players[index].base.position;
            self.players[index].reported_tick = self.tick;
            let action = actions[index].aim_action();
            // Updater updates Player's frame-dependent button edges before
            // any MonoUpdatable, including while its slime is parked for a
            // stock choice.
            let registered_action = registered_actions[index].aim_action();
            let movement_input = actions[index].input_direction(self.players[index].keyboard);
            let jump_down = registered_action.buttons & 1 != 0;
            let jump_pressed = jump_down && !self.players[index].input_jump_was_down;
            self.players[index].input_jump_was_down = jump_down;
            self.player_tick(
                index,
                action,
                registered_action,
                movement_input,
                jump_pressed,
                data,
            );
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
                self.kill(index, ENVIRONMENT, CAUSE_ACCIDENT);
            }
        }
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
                    ActiveAbility::None | ActiveAbility::Dash | ActiveAbility::Bow => true,
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
        for index in 0..self.players.len() {
            if self.playable(index) {
                self.players[index].base.sync_collider();
                self.players[index].base.late_step(&self.platforms);
            }
            let input = actions[index].input_direction(self.players[index].keyboard);
            self.update_obs_state(index, input);
        }
        self.arrows_integrate(data);
        self.platforms_tick(data);
        self.arrows_contacts(data);
        self.tick += 1;
        self.episode_ticks += 1;
        self.session_tick();
    }

    fn playable(&self, index: usize) -> bool {
        let slot = &self.players[index];
        slot.base.alive && slot.respawn_ticks == 0 && slot.respawn_spawn_ticks == 0
    }

    /// SlimeController.OldUpdate: cooldowns, ability entry, then movement.
    fn player_tick(
        &mut self,
        index: usize,
        action: Action,
        registered_action: Action,
        movement_input: Vec2,
        jump_pressed: bool,
        data: &data::RolloutData,
    ) {
        let in_ability = {
            let slot = &self.players[index];
            slot.active != ActiveAbility::None || slot.base.dash.active
        };
        if !self.playable(index) {
            self.respawn_tick(index, movement_input, jump_pressed, data);
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
            for s in 0..3 {
                self.players[index].timers[s] += DT;
            }
        } else {
            self.players[index].ability_time += DT;
        }

        // Ability entry (held button, ready, castable, not already in one).
        if !in_ability {
            self.players[index].ability_time = Fix::ZERO;
            for s in 0..3 {
                let slot = &self.players[index];
                let button_down = registered_action.buttons & (2 << s) as u8 != 0;
                if !button_down || slot.timers[s] < slot.cooldowns[s] {
                    continue;
                }
                let ability = slot.abilities[s];
                {
                    let slot = &mut self.players[index];
                    // PlayerPhysics updates before SlimeController. If this
                    // airborne body enters an ability, its PlayerInfo package
                    // has current gravity/collision state but PlayerBody has
                    // not performed another position integration.
                    if !slot.base.grounded {
                        slot.base.gravity_step(&self.platforms);
                    }
                    slot.ability_origin = slot.base.position;
                    slot.ability_origin_velocity = slot.base.velocity + slot.base.external_velocity;
                    slot.ability_origin_rotation = slot.base.rotation;
                    slot.ability_origin_state = slot.obs_state;
                }
                match ability {
                    ABILITY_DASH => {
                        // The movement slice implements the dash; unlock its
                        // internal cooldown gate and let base.step enter it.
                        self.players[index].base.cooldown = Fix::from_int(100);
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Dash;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = registered_action;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let masked = Action {
                            angle: registered_action.angle,
                            buttons: (registered_action.buttons & 1) | 2,
                        };
                        // Dash is registered after SlimeController in this
                        // prefab, so its freshly enabled UpdateSim runs later
                        // in the same pass. That performs anticipation setup
                        // now; platform and bow begin updating next tick.
                        self.dash_move(index, masked, movement_input);
                        return;
                    }
                    ABILITY_BOW => {
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Bow;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = registered_action;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let slot = &mut self.players[index];
                        // BowTransform.OnEnterAbility: aim starts at the up
                        // vector; airborne entry slows to a tenth (slow-mo).
                        slot.bow = BowState {
                            aim: slot.base.up,
                            ..BowState::default()
                        };
                        if !slot.base.grounded {
                            slot.base.velocity *= Fix::from_f32(0.1);
                            slot.base.external_velocity *= Fix::from_f32(0.1);
                        }
                        // BowTransform follows Ability in the prefab updater
                        // list, so a newly enabled Bow updates later in this
                        // same pass (like Dash, unlike a newly instantiated
                        // projectile which is deferred).
                        self.bow_tick(index, registered_action, data);
                        return;
                    }
                    ABILITY_PLATFORM => {
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Platform;
                        self.players[index].active_slot = s;
                        self.players[index].ability_entry_action = registered_action;
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
                        slot.base.grounded = false;
                        slot.base.ground = None;
                        slot.base.grounded_speed = Fix::ZERO;
                        return;
                    }
                    // Other catalog abilities are never simulated here.
                    _ => {}
                }
            }
        }

        if self.players[index].active == ActiveAbility::Bow {
            self.bow_tick(index, registered_action, data);
            return;
        }
        if self.players[index].active == ActiveAbility::Platform {
            self.platform_ability_tick(index, registered_action, data);
            // PlatformTransform updates before its own
            // DestroyIfOutsideSceneBounds component. Unlike the ordinary
            // slime, this ability body's position is final for this updater
            // pass (the separate platform moves later in DetPhysics).
            if self.players[index].base.alive && self.players[index].base.outside_bounds() {
                self.kill(index, ENVIRONMENT, CAUSE_ACCIDENT);
            }
            return;
        }
        if self.players[index].active == ActiveAbility::Dash {
            let starting = !self.players[index].base.dash.active;
            let source = if starting {
                self.players[index].ability_entry_action
            } else {
                registered_action
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
            .step_with_input(masked, movement_input, &self.platforms);
        if self.players[index].dash_exit_followup {
            self.players[index].reported_position = self.players[index].base.position;
            self.players[index].reported_tick = self.tick;
            self.players[index].dash_exit_followup = false;
        }
        let _ = data;
    }

    /// base.step with the dash shove check from World::step.
    fn dash_move(&mut self, index: usize, action: Action, movement_input: Vec2) {
        self.players[index]
            .base
            .step_with_input(action, movement_input, &self.platforms);
        // Dash precedes Ability in the prefab updater order, so the public
        // player position sees Dash's post-update transform on every active
        // pass, including activation and teleport/exit.
        if self.players[index].active == ActiveAbility::Dash {
            self.players[index].reported_position = self.players[index].base.position;
            self.players[index].reported_tick = self.tick;
            if !self.players[index].base.dash.active {
                self.players[index].dash_exit_followup = true;
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
                crate::world::closest(&self.platforms, origin, direction, Fix::from_int(13), None)
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
            for (arrow_index, _) in arrow_hits {
                let arrow = self.arrows[arrow_index];
                let lethal = arrow.owner != caster_id || arrow.can_hurt_owner;
                if lethal {
                    let cause = if arrow.owner == caster_id {
                        CAUSE_ACCIDENT
                    } else {
                        CAUSE_OTHER
                    };
                    self.kill(index, arrow.owner, cause);
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

    /// BowTransform.OldUpdate + UpdateSim for one tick: aim tracking, the
    /// loading-frame charge schedule, firing on release, slow-mo aiming
    /// physics, post-fire recoil flight, and the exit conditions.
    fn bow_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let bow = &data.bow;
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
        let platforms: &[Platform] = &this.platforms;
        // Bow's boundary component runs after BowTransform but before
        // PlayerBody performs this pass's position integration. Retain the
        // pre-integration boundary result instead of testing the new position.
        let boundary_death = slot.base.outside_bounds();
        if slot.base.grounded {
            if slot.bow.has_fired {
                // Landed after firing: exit with the landing (justlanded).
                exit = true;
            } else {
                // Grounded aiming: physics.Move(zero) = friction, stay attached.
                slot.base.step(Action::default(), platforms);
            }
        } else if !slot.bow.has_fired {
            // Airborne aiming: near-frozen fall (gravity_modifier 0.03),
            // speed clamped to maxSlowMoSpeed.
            crate::player_physics::apply_gravity(
                &mut slot.base.velocity,
                bow.constants.gravity_accel,
                Fix::from_f32(0.03),
                bow.constants.gravity_max_fall_speed,
                false,
            )
            .expect("valid bow gravity constants");
            let speed = slot.base.velocity.length();
            if speed > bow.max_slow_mo_speed {
                slot.base.velocity *= bow.max_slow_mo_speed / speed;
            }
            slot.base.collision_rays(platforms);
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
            slot.base.collision_rays(platforms);
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
                slot.base.velocity =
                    before + (slot.base.velocity - before) * bow.aireal_control_blend;
                let speed = slot.base.velocity.length();
                if speed > bow.constants.max_speed {
                    slot.base.velocity *= bow.constants.max_speed / speed;
                }
                slot.base.position += (slot.base.velocity + slot.base.external_velocity) * DT;
                slot.base.external_velocity *= Fix::from_f64(0.95);
            }
        }
        if slot.bow.has_fired && slot.bow.fire_ticks_left > 0 {
            slot.bow.fire_ticks_left -= 1;
            if slot.bow.fire_ticks_left == 0 {
                exit = true;
            }
        }
        if boundary_death {
            slot.base.alive = false; // game_tick attributes it after this updater
        }
        if exit {
            slot.active = ActiveAbility::None;
            // ExitAbility ungrounds the body, so the state machine resumes
            // from fall (and lands again next tick if still grounded).
            slot.obs_state = STATE_FALL;
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
                if aim != Vec2::ZERO {
                    up = slerp(up, aim, DT * pd.rotation_speed);
                }
                // PlatformTransform writes the aimed `up` into the spawned
                // platform's FixTransform. DetPhysics then copies that scalar
                // rotation into the body before integrating it. Resizing only
                // changes the collider geometry; it must not re-derive or pin
                // the body's orientation.
                let rotation = up.perp_cw().normalized_angle();
                self.platform_dynamics[platform_index].body.orientation = rotation;
                self.platform_dynamics[platform_index].home_rotation = rotation;
                self.platforms[platform_index].surface.right = up.perp_cw();
                self.platforms[platform_index].surface.up = up;
                self.reshape_platform(platform_index, center, pd.spawn_extents, radius);
                let slot = &mut self.players[index];
                slot.base.position = center;
                slot.base.rotation = rotation;
                slot.platform.sizing_right = up.perp_cw();
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
                    slot.obs_state = STATE_FALL;
                    // Ability.ExitAbility republishes potentialCollider.position
                    // after PlatformTransform has applied the final ejection.
                    // SlimeController's synchronous onExitAbility callback then
                    // enters Fall and immediately performs
                    // VelocityBasedRaycasts. Preserve both phase boundaries:
                    // this tick observes the ejected position, while the live
                    // body may already have reattached for the next tick.
                    slot.reported_position = slot.base.position;
                    slot.reported_tick = self.tick;
                    slot.base.collision_rays(&self.platforms);
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
        // `body.right`, not the potentially wobbling platform collider axes.
        let basis = self.players[index].platform.sizing_right;
        let conj = Vec2::new(basis.x, -basis.y);
        let st = &mut self.players[index].platform;
        st.cursor += aim * CURSOR_SENSITIVITY * DT;
        let mut local = (st.cursor - center).complex_mul(conj);
        local.x = local.x.max(-x - radius).min(x + radius);
        local.y = local.y.max(-y - radius).min(y + radius);
        st.cursor = center + local.complex_mul(basis);
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
            let gx = dir.x * DT * pd.size_change_speed;
            let gy = dir.y * DT * pd.size_change_speed;
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
                let root = -(c1 * Fix::from_f64(0.5)) + disc;
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
            extents: pd.spawn_extents,
            radius: pd.spawn_radius,
        };
        let platform_index = self.platforms.len();
        self.platforms
            .push(make_platform(self.next_platform_id, &surface));
        self.platform_dynamics.push(PlatformDynamics {
            body: pd.spawn_body,
            home_position: position,
            home_rotation: Fix::ZERO,
            speed: pd.spawn_speed,
            mu: pd.spawn_mu,
            rotation_speed: pd.spawn_rotation_speed,
            rotation_mu: pd.spawn_rotation_mu,
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
        self.entities[entity].flags = E_ENABLED | E_DESTROYED;
        self.platforms.remove(index);
        self.platform_dynamics.remove(index);
        self.platform_defs.remove(index);
        self.platform_entities.remove(index);
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
    }

    fn reshape_platform(&mut self, index: usize, center: Vec2, extents: Vec2, radius: Fix) {
        let old_center = self.platforms[index].surface.center;
        let platform = &mut self.platforms[index];
        platform.surface.center = center;
        platform.surface.extents = extents;
        platform.surface.radius = radius;
        platform.collider.inner.center = center;
        platform.collider.inner.right = platform.surface.right * extents.x;
        platform.collider.inner.up = platform.surface.up * extents.y;
        platform.collider.inner.inverse_extents =
            Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
        platform.collider.radius = radius;
        let dynamics = &mut self.platform_dynamics[index];
        // ResizablePlatform.ResizePlatform recomputes these from geometry on
        // every resize; keeping the prefab's mass makes long/thin platforms
        // react far too strongly to contacts.
        let pd = &data::get().platform;
        let mass = Fix::from_int(10)
            + pd.spawn_mass_per_area * platform_area(extents.x, extents.y, radius);
        let w = extents.x + radius;
        let h = extents.y + radius;
        let longest = w.max(h);
        let inertia = w.min(h) * longest * longest * Fix::from_f32(0.2) + Fix::from_int(40);
        dynamics.body.inverse_mass = Fix::ONE / mass;
        dynamics.body.inverse_inertia = Fix::ONE / inertia;
        dynamics.home_position += center - old_center;
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

    fn destroy_arrow(&mut self, index: usize) {
        let arrow = &mut self.arrows[index];
        arrow.active = false;
        arrow.registered = false;
        if let Some(entity) = arrow.entity {
            let entity = &mut self.entities[entity];
            entity.flags = E_ENABLED | E_DESTROYED;
            entity.vx = 0;
            entity.vy = 0;
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

    /// DetPhysics integrates every BoplBody before constructing contact
    /// manifolds. Keep projectile time/gravity in that body phase; contacts
    /// run after platform integration below.
    fn arrows_integrate(&mut self, data: &data::RolloutData) {
        // Scene DetPhysics overrides gravityAccel to -40 on every map (the
        // -9.82 field default never applies).
        let gravity = Fix::from_f64(-40.0) * data.arrow.gravity_scale;
        for i in 0..self.arrows.len() {
            if !self.arrows[i].active || !self.arrows[i].registered {
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
            arrow.time_passed += DT;
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
                arrow.velocity * (Fix::ONE / speed)
            };
            arrow.rotation = up.perp_cw().normalized_angle();
            arrow.velocity.y += gravity * DT;
            arrow.position += arrow.velocity * DT;
        }
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

            // Circle/RR callbacks precede all player shape groups. If an arrow
            // overlaps terrain and a player in the same physics pass, the wall
            // callback destroys its components before PlayerCollision runs.
            if self.platforms.iter().any(|platform| {
                crate::physics::contact::circle_rounded(circle, platform.collider).is_some()
            }) {
                self.destroy_arrow(i);
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
                self.kill(victim, arrow.owner, cause);
                if arrow.owner != victim_id && arrow.owner >= 1 {
                    self.players[arrow.owner as usize - 1].telemetry.arrow_hits += 1;
                }
            }
        }
    }

    /// Stable rounded-rectangle broad phase, matching the RR subset of the
    /// generic DetPhysics sweep (strict touching-edge rejection included).
    fn platform_pairs(&self) -> Vec<[usize; 2]> {
        if self.platforms.len() < 2 {
            return Vec::new();
        }
        let mut mean = Vec2::ZERO;
        for platform in &self.platforms {
            mean += platform.surface.center;
        }
        mean = mean / Fix::from_int(self.platforms.len() as i64);
        let mut variance = Vec2::ZERO;
        for platform in &self.platforms {
            let delta = mean - platform.surface.center;
            variance += Vec2::new(delta.x * delta.x, delta.y * delta.y);
        }
        let sweep_x = variance.x > variance.y;
        let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
        let other = |v: Vec2| if sweep_x { v.y } else { v.x };
        let bounds: Vec<_> = self
            .platforms
            .iter()
            .map(|platform| {
                let r = platform.collider;
                let extent = Vec2::new(
                    r.inner.right.x.abs() + r.inner.up.x.abs() + r.radius,
                    r.inner.right.y.abs() + r.inner.up.y.abs() + r.radius,
                );
                [r.inner.center - extent, r.inner.center + extent]
            })
            .collect();
        let mut active: Vec<_> = (0..self.platforms.len()).collect();
        active.sort_by(|&a, &b| {
            (axis(bounds[a][0]) - axis(bounds[b][0]))
                .sign()
                .cmp(&0)
                .then(a.cmp(&b))
        });
        let mut pairs = Vec::new();
        let mut window = Vec::new();
        for b in active {
            window.retain(|&a: &usize| axis(bounds[a][1]) > axis(bounds[b][0]));
            for &a in &window {
                if other(bounds[a][1]) <= other(bounds[b][0])
                    || other(bounds[a][0]) >= other(bounds[b][1])
                {
                    continue;
                }
                pairs.push(if a < b { [a, b] } else { [b, a] });
            }
            window.push(b);
        }
        pairs
    }

    /// Rounded-platform rigid-body stage. The order matches DetPhysics:
    /// gameplay forces (AnimateVelocity and AntiLockPlatform), integration,
    /// broad phase, narrow phase, collision impulse, positional correction,
    /// then transform/rider synchronization.
    fn platforms_tick(&mut self, data: &data::RolloutData) {
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
            spawn.removal_time += DT;
            let minimum = Fix::from_f32(0.1);
            removal_resizes.push((
                spawn.platform_index,
                Vec2::new(
                    minimum + (spawn.removal_extents.x - minimum) * amount,
                    minimum + (spawn.removal_extents.y - minimum) * amount,
                ),
                minimum + (spawn.removal_radius - minimum) * amount,
            ));
        }
        for (index, extents, radius) in removal_resizes {
            let center = self.platforms[index].surface.center;
            self.reshape_platform(index, center, extents, radius);
        }
        removal_destroys.sort_unstable_by(|a, b| b.cmp(a));
        for index in removal_destroys {
            self.remove_platform(index);
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
        let disturbed =
            self.platforms
                .iter()
                .zip(&self.platform_dynamics)
                .any(|(platform, dynamics)| {
                    platform.surface.center != dynamics.home_position
                        || dynamics.body.velocity != Vec2::ZERO
                        || dynamics.body.angular_velocity != Fix::ZERO
                        || dynamics.body.orientation != dynamics.home_rotation
                });
        if !self.sudden_death
            && !orbit_running
            && !disturbed
            && !self.spawned.iter().any(|spawn| spawn.active)
        {
            return;
        }
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
                let drop = speed * scale * DT;
                self.platform_dynamics[i].home_position.y -= drop;
            }
        }

        // AnimateVelocity.UpdateSim. Keep the apparently cancelling inverse
        // mass and 60 Hz impulse conversions: their intermediate fixed-point
        // rounding is observable over long-running springs (notably map 2's
        // orbiter), so simplifying this to `velocity += spring * DT` drifts.
        for i in 0..self.platforms.len() {
            let position = self.platforms[i].surface.center;
            let dynamics = &mut self.platform_dynamics[i];
            let spring = (dynamics.home_position - position) * dynamics.speed
                - dynamics.body.velocity * dynamics.mu;
            let impulse = spring * DT / dynamics.body.inverse_mass;
            dynamics.body.force += impulse * Fix::from_int(60);
            let mut angle_delta = dynamics.home_rotation - dynamics.body.orientation;
            if angle_delta.abs() > Fix::PI {
                angle_delta = -Fix::from_int(angle_delta.sign()) * (Fix::TAU - angle_delta.abs());
            }
            dynamics.body.angular_velocity = (dynamics.body.angular_velocity
                + angle_delta * dynamics.rotation_speed * DT)
                * (Fix::ONE - dynamics.rotation_mu);
        }

        // AntiLockPlatform runs after AnimateVelocity: add the ordinary force
        // and re-home at the pre-integration position, preserving the existing
        // one-tick-lagged home behavior.
        if !self.sudden_death {
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

        // DetPhysics integrates every rounded body before it constructs any
        // contacts. All rollout platforms have gravityScale=0, but pass the
        // game's -40 gravity for completeness.
        for i in 0..self.platforms.len() {
            let dynamics = &mut self.platform_dynamics[i];
            let shift = dynamics.body.integrate(DT, Fix::from_int(-40));
            let orientation = dynamics.body.orientation;
            let platform = &mut self.platforms[i];
            platform.surface.center += shift;
            platform.surface.right = Vec2::angle(orientation);
            platform.surface.up = platform.surface.right.perp();
            platform.collider.inner.center = platform.surface.center;
            platform.collider.inner.right = platform.surface.right * platform.surface.extents.x;
            platform.collider.inner.up = platform.surface.up * platform.surface.extents.y;
        }
        // CollisionInformation stores the shape as it was during narrow phase;
        // PlayerCollision's later squish callback tests that copy, before the
        // RR positional corrections below move either platform.
        let callback_colliders: Vec<_> = self.platforms.iter().map(|p| p.collider).collect();

        // RR-only sweep/narrow-phase. Contacts are all captured before any
        // impulse or correction, just like DetPhysics' collision buffer.
        let contacts: Vec<_> = self
            .platform_pairs()
            .into_iter()
            .filter_map(|[a, b]| {
                crate::physics::contact::rounded_rounded(
                    self.platforms[a].collider,
                    self.platforms[b].collider,
                )
                .map(|contact| (a, b, contact))
            })
            .collect();
        for (a, b, contact) in contacts {
            let pos_a = self.platforms[a].surface.center;
            let pos_b = self.platforms[b].surface.center;
            let (left, right) = self.platform_dynamics.split_at_mut(b);
            let body_a = &mut left[a].body;
            let body_b = &mut right[0].body;
            resolve(contact, body_a, body_b, pos_a, pos_b);
            let correction =
                positional_correction(contact, body_a.inverse_mass, body_b.inverse_mass, false);
            for (index, delta) in [(a, correction[0]), (b, correction[1])] {
                self.platforms[index].surface.center += delta;
                self.platforms[index].collider.inner.center += delta;
            }
        }

        // PlayerCollision kills a grounded slime when a different wall's
        // rounded rectangle contains its squish point (the stock player
        // prefab uses the body centre). Player boxes are triggers, so this is
        // a callback effect rather than a rigid-body response.
        let squished: Vec<_> = self
            .players
            .iter()
            .enumerate()
            .filter(|(_, slot)| self.playable_slot(slot) && slot.base.grounded)
            .filter(|(_, slot)| {
                callback_colliders.iter().enumerate().any(|(i, &collider)| {
                    Some(self.platforms[i].id) != slot.base.ground
                        && collider.overlaps(slot.base.position)
                })
            })
            .map(|(index, _)| index)
            .collect();
        for index in squished {
            self.kill(index, ENVIRONMENT, CAUSE_SQUISHED);
        }

        // StickyRoundedRectangle riders are positioned from their surface
        // local after physics. PlatformTransform itself is intentionally not
        // synchronized here: its UpdateSim copied the platform's FixTransform
        // before DetPhysics moved/corrected it, matching Unity's phase order.
        for slot in &mut self.players {
            if slot.base.alive && slot.base.grounded && slot.removed_ground_normal.is_none() {
                slot.base.late_step(&self.platforms);
                slot.base.sync_collider();
            }
        }

        // The Platform ability object has DestroyIfOutsideSceneBounds, so its
        // transformed owner is handled by deaths_tick using the position it
        // copied before physics. The separate ResizablePlatform prefab has no
        // boundary component: it remains a collidable body after that death.
    }

    fn playable_slot(&self, slot: &PlayerSlot) -> bool {
        slot.base.alive && slot.respawn_ticks == 0
    }

    pub fn kill(&mut self, index: usize, killer: u32, cause: u32) {
        let players = self.players.len();
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
        slot.active = ActiveAbility::None;
        slot.base.dash.active = false;
        slot.dash_exit_followup = false;
        if killer >= 1 && killer <= players as u32 && killer != slot.base.id {
            self.players[killer as usize - 1].kills += 1;
        }
        let slot = &mut self.players[index];
        // Stocks: spawn the replacement clone at the chooser's safe center
        // point. Since Player.Kill sees that clone, it does not mark the
        // player eliminated and leaves CauseOfDeath at NotDeadYet.
        if slot.stocks > 0 && !self.sudden_death {
            slot.stocks -= 1;
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
            return;
        }
    }

    /// GameSessionHandler.UpdateSim's winner check. Its authored hierarchy is
    /// before player components and physics, so it consumes the previous
    /// tick's deaths. The rest of the current updater pass still runs after it
    /// marks the game ended; RlBridge observes terminal only after the pass.
    fn session_result_tick(&mut self) {
        let alive_teams: Vec<u32> = {
            let mut teams: Vec<u32> = self
                .players
                .iter()
                .filter(|p| !p.out)
                .map(|p| p.team)
                .collect();
            teams.sort_unstable();
            teams.dedup();
            teams
        };
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
        let mut slimes = std::mem::take(&mut self.slimes);
        self.fill_slimes(&mut slimes);
        self.slimes = slimes;
        let flags = self.flags();
        Self::write_state(
            &mut self.state,
            &self.players,
            &self.entities,
            &self.slimes,
            self.tick,
            self.episode_ticks,
            self.warmup,
            flags,
        );
        self.state.entities = self.entities.as_ptr();
        self.state.slimes = self.slimes.as_ptr();
    }

    fn observe_sample(&mut self) {
        self.refresh_entities();
        self.sample_entities.clear();
        self.sample_entities.extend_from_slice(&self.entities);
        let mut slimes = std::mem::take(&mut self.sample_slimes);
        self.fill_slimes(&mut slimes);
        self.sample_slimes = slimes;
        let flags = self.flags();
        Self::write_state(
            &mut self.sample,
            &self.players,
            &self.sample_entities,
            &self.sample_slimes,
            self.tick,
            self.episode_ticks,
            self.warmup,
            flags,
        );
        self.sample.entities = self.sample_entities.as_ptr();
        self.sample.slimes = self.sample_slimes.as_ptr();
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
        for index in 0..self.players.len() {
            let (body_e, dash_e, circle_e) = {
                let s = &self.players[index];
                (s.body_entity, s.dash_entity, s.circle_entity)
            };
            let slot = &self.players[index];
            let alive = slot.base.alive;
            let choosing_respawn = slot.respawn_ticks > 0;
            let dashing = slot.base.dash.active;
            let bowing = slot.active == ActiveAbility::Bow;
            let in_ability = slot.active != ActiveAbility::None;
            let anticipating = slot.active == ActiveAbility::Platform
                && slot.platform.phase == PLATFORM_ANTICIPATION;
            let bow_aim = slot.bow.aim;
            let position = slot.base.position;
            let rotation = slot.base.rotation;
            let basis = Vec2::angle(rotation);
            let data = data::get();

            let body = &mut self.entities[body_e];
            if !alive {
                body.flags = E_ENABLED | E_TRIGGER | E_DESTROYED;
            } else if choosing_respawn {
                body.flags = E_ACTIVE | E_TRIGGER;
            } else if dashing || in_ability {
                body.flags = E_ENABLED | E_TRIGGER;
            } else {
                body.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
            }
            let extents = data.player.hurtbox_extents;
            body.x = position.x.raw();
            body.y = position.y.raw();
            // The hurtbox has PlayerBody movement on the same GameObject but
            // no BoplBody. DPhysicsBox therefore registers an infinite-mass
            // trigger PhysicsBody whose observed velocity remains zero; the
            // moving velocity is exposed by the separate slime record.
            body.vx = 0;
            body.vy = 0;
            let right = Vec2::new(extents.x, Fix::ZERO).complex_mul(basis);
            let up = Vec2::new(Fix::ZERO, extents.y).complex_mul(basis);
            body.right_x = right.x.raw();
            body.right_y = right.y.raw();
            body.up_x = up.x.raw();
            body.up_y = up.y.raw();
            body.rotation = rotation.raw();

            let dash_box = &mut self.entities[dash_e];
            if alive && dashing {
                let extents = data.dash.hurtbox_extents;
                let angle = slot.base.dash.input.normalized_angle();
                let basis = Vec2::angle(angle);
                dash_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                dash_box.owner = slot.base.id;
                dash_box.x = position.x.raw();
                dash_box.y = position.y.raw();
                // DPhysicsBox is a trigger body; PlayerBody velocity is not
                // stored on its PhysicsBody observation record.
                dash_box.vx = 0;
                dash_box.vy = 0;
                let right = Vec2::new(extents.x, Fix::ZERO).complex_mul(basis);
                let up = Vec2::new(Fix::ZERO, extents.y).complex_mul(basis);
                dash_box.right_x = right.x.raw();
                dash_box.right_y = right.y.raw();
                dash_box.up_x = up.x.raw();
                dash_box.up_y = up.y.raw();
                dash_box.rotation = angle.raw();
            } else {
                dash_box.flags = E_ENABLED | E_TRIGGER;
                dash_box.x = 0;
                dash_box.y = 0;
                dash_box.vx = 0;
                dash_box.vy = 0;
            }
            // Bow hurtbox: active while aiming/firing, up vector = aim.
            let bow_e = self.players[index].bow_entity;
            let bow_box = &mut self.entities[bow_e];
            if alive && bowing {
                let extents = data.bow.hurtbox_extents;
                let up_dir = if bow_aim == Vec2::ZERO {
                    Vec2::UP
                } else {
                    bow_aim
                };
                let right_dir = up_dir.perp_cw();
                bow_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                bow_box.owner = slot.base.id;
                bow_box.x = position.x.raw();
                bow_box.y = position.y.raw();
                bow_box.vx = 0;
                bow_box.vy = 0;
                bow_box.right_x = (right_dir * extents.x).x.raw();
                bow_box.right_y = (right_dir * extents.x).y.raw();
                bow_box.up_x = (up_dir * extents.y).x.raw();
                bow_box.up_y = (up_dir * extents.y).y.raw();
                bow_box.rotation = right_dir.normalized_angle().raw();
            } else {
                bow_box.flags = E_ENABLED | E_TRIGGER;
                bow_box.x = 0;
                bow_box.y = 0;
                bow_box.vx = 0;
                bow_box.vy = 0;
            }
            // The Platform ability's hurtbox circle is live during
            // anticipation only (hurtBox.enabled until CreatePlatform).
            let circle = &mut self.entities[circle_e];
            if alive && anticipating {
                circle.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                circle.owner = slot.base.id;
                circle.x = position.x.raw();
                circle.y = position.y.raw();
            } else if alive && slot.active == ActiveAbility::Platform {
                circle.flags = E_ACTIVE | E_TRIGGER;
                circle.owner = slot.base.id;
                circle.x = position.x.raw();
                circle.y = position.y.raw();
            } else {
                circle.flags = if slot.platform.hurtbox_disabled {
                    E_TRIGGER
                } else {
                    E_ENABLED | E_TRIGGER
                };
                circle.x = 0;
                circle.y = 0;
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
        for (i, platform) in self.platforms.iter().enumerate() {
            let e = &mut self.entities[self.platform_entities[i]];
            let s = &platform.surface;
            let dynamics = &self.platform_dynamics[i];
            e.x = s.center.x.raw();
            e.y = s.center.y.raw();
            e.vx = dynamics.body.velocity.x.raw();
            e.vy = dynamics.body.velocity.y.raw();
            let right = s.right * s.extents.x;
            let up = s.up * s.extents.y;
            e.right_x = right.x.raw();
            e.right_y = right.y.raw();
            e.up_x = up.x.raw();
            e.up_y = up.y.raw();
            e.radius = s.radius.raw();
            // RlObservation publishes PhysicsBody.orientation directly. The
            // collider basis is derived output and must not be converted back
            // into an angle (especially around the Q32 sine-table boundary).
            e.rotation = dynamics.body.orientation.raw();
            e.angular_velocity = dynamics.body.angular_velocity.raw();
            e.inverse_mass = dynamics.body.inverse_mass.raw();
        }
    }

    fn fill_slimes(&self, slimes: &mut Vec<CSlime>) {
        slimes.clear();
        for slot in &self.players {
            if !slot.base.alive {
                continue; // destroyed with the player, like the .NET sim
            }
            let in_ability = slot.active != ActiveAbility::None;
            let respawning = slot.respawn_ticks > 0 || slot.respawn_spawn_ticks > 0;
            let mut flags = S_ACTIVE;
            if respawning {
                flags |= S_SUPPRESS_INPUT;
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
            for s in 0..3 {
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
                ability_flags[s] = f;
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
        entities: &[CEntity],
        slimes: &[CSlime],
        tick: u32,
        episode_ticks: u32,
        warmup: u32,
        flags: u32,
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
            player_time_scale: 1 << 32,
        };
        for (i, slot) in players.iter().enumerate() {
            let mut player_flags = 0;
            if slot.base.alive {
                player_flags |= 1;
            }
            if slot.won {
                player_flags |= 2;
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
                invisible_time: 0,
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

fn make_platform(id: u32, surface: &Surface) -> Platform {
    let extents = surface.extents;
    let inverse = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
    Platform {
        id,
        surface: surface.clone(),
        collider: RoundedRect {
            inner: OrientedBox {
                center: surface.center,
                right: surface.right * extents.x,
                up: surface.up * extents.y,
                inverse_extents: inverse,
            },
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
    (from * ((Fix::ONE - t) * angle).sin() + to * (t * angle).sin()) / denominator
}

/// DPhysicsRoundedRect.PlatformArea(width, height, radius) with half-extent
/// width/height: 4wr + 4hr + 4wh + pi r^2.
fn platform_area(w: Fix, h: Fix, r: Fix) -> Fix {
    Fix::from_int(4) * w * r + Fix::from_int(4) * h * r + Fix::from_int(4) * w * h + Fix::PI * r * r
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

        world.kill(0, ENVIRONMENT, CAUSE_ACCIDENT);
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

        world.player_tick(
            0,
            Action::default(),
            Action::default(),
            Vec2::ZERO,
            false,
            data::get(),
        );
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
