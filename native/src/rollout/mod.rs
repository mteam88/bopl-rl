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
    geometry::{OrientedBox, RoundedRect},
    random::Random,
    surface::Surface,
    vector::Vec2,
    world::{Action, DashState, Platform, Player, DT},
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
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CPlayerConfig {
    pub color: u32,
    pub team: u32,
    pub keyboard: u32,
    pub abilities: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
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
const S_SIMULATED: u32 = 16;

// Slime states (SlimeController.State).
const STATE_IDLE: u32 = 0;
const STATE_RUN: u32 = 1;
const STATE_FALL: u32 = 2;
const STATE_ABILITY: u32 = 5;

// Layers (sim/Export/settings.json).
const LAYER_PLAYER: u32 = 8;
const LAYER_PROJECTILE: u32 = 9;
const LAYER_WALL: u32 = 11;
const LAYER_WATER: u32 = 4;

const CAUSE_ACCIDENT: u32 = 2;
const CAUSE_NOT_DEAD_YET: u32 = 5;
/// Player.cs Constants.ENVIRONMENTAL_PLAYER_ID.
const ENVIRONMENT: u32 = 0xFFFF_FFFF;

/// Slime controllers wake this many ticks into the warmup (measured against
/// the .NET reset: ability cooldown timers read -1.5s + (240-62)/60 = 1.4667s
/// after a 240-tick warmup).
const SLIME_AWAKE_TICK: u32 = 62;
const TIMER_START: Fix = Fix(-6_442_450_944); // -1.5s

/// A live arrow (Arrow.cs + Projectile.cs, simplified: a gravity projectile
/// that sticks to walls/platforms and kills players on contact).
#[derive(Clone, Copy, Default)]
pub struct Arrow {
    pub active: bool,
    pub landed: bool,
    pub destroyed: bool,
    pub owner: u32,
    pub entity: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub ticks_alive: u32,
}

/// A platform spawned by the Platform ability (ResizablePlatform prefab).
#[derive(Clone, Default)]
pub struct SpawnedPlatform {
    pub active: bool,
    pub entity: usize,
    pub platform_index: usize,
}

/// Per-player bow ability state (BowTransform.cs).
#[derive(Clone, Copy, Default)]
pub struct BowState {
    pub ticks: u32,
    pub loading_frame: u32,
    pub has_fired: bool,
    pub fire_ticks_left: u32,
    pub was_grounded: bool,
}

/// Per-player platform ability state (PlatformTransform.cs).
#[derive(Clone, Copy, Default)]
pub struct PlatformState {
    pub ticks: u32,
    pub phase: u32,
    pub rotation: Fix,
    pub extents: Vec2,
    pub exit_ticks: u32,
    pub position: Vec2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ActiveAbility {
    None,
    Dash,
    Bow,
    Platform,
}

/// One player's match state beyond the movement body.
pub struct PlayerSlot {
    pub base: Player,
    pub team: u32,
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
    pub ability_time: Fix,
    pub bow: BowState,
    pub platform: PlatformState,
    /// Respawn countdown while holding a consumed stock (StockRespawn).
    pub respawn_ticks: u32,
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
            platform_defs: Vec::with_capacity(64),
            platform_entities: Vec::with_capacity(64),
            arrows: Vec::with_capacity(64),
            spawned: Vec::with_capacity(64),
            sudden_death: false,
            sudden_death_tick: 0,
            sudden_death_delays: Vec::with_capacity(64),
            next_platform_id: 0,
            next_entity_id: 0,
            entities: Vec::with_capacity(192),
            slimes: Vec::with_capacity(MAX_PLAYERS),
            state: empty_state(),
            sample_entities: Vec::with_capacity(192),
            sample_slimes: Vec::with_capacity(MAX_PLAYERS),
            sample: empty_state(),
            sample_valid: false,
            error: [0; 512],
        }
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
        if players < 1 || players > MAX_PLAYERS {
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
        self.rng = Random::new(config.seed.max(1));
        self.sudden_death = false;
        self.arrows.clear();
        self.spawned.clear();
        self.players.clear();
        self.platforms.clear();
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
                flags: E_ENABLED | E_TRIGGER,
                hierarchy: 100_000_030 + 30 * i as u32,
                radius: data.platform.hurtbox_radius.0,
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
            x: def.water_position.x.0,
            y: def.water_position.y.0,
            right_x: def.water_extents.x.0,
            up_y: def.water_extents.y.0,
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
                        E_ENABLED | E_TRIGGER
                    },
                    hierarchy: hier + 30 * i as u32,
                    right_x: extents.x.0,
                    up_y: extents.y.0,
                    inverse_mass: -(1 << 32),
                    ..CEntity::default()
                });
            }
        }
        self.next_platform_id = 100;
        self.sudden_death_delays.clear();
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
                inverse_mass: Fix::from_f64(1.0 / 15.0).0,
                ..CEntity::default()
            });
            self.platforms.push(make_platform(self.next_platform_id, &p.surface));
            self.platform_defs.push(index);
            self.platform_entities.push(entity);
            self.next_platform_id += 1;
            self.sudden_death_delays
                .push(self.rng.fix_range(Fix::ZERO, p.sudden_death_max_delay));
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
                ability_time: Fix::ZERO,
                bow: BowState::default(),
                platform: PlatformState::default(),
                respawn_ticks: 0,
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
            self.game_tick(actions);
            if self.terminal {
                break;
            }
        }
        self.observe_main();
        Ok(())
    }

    pub fn step_sampled(&mut self, actions: &[CAction], first: u32, second: u32) -> Result<(), String> {
        if first + second > 16 {
            return Err("Sampled step phases too long".into());
        }
        self.step(actions, first.max(1))?;
        self.observe_sample();
        self.sample_valid = true;
        if !self.terminal {
            for _ in 0..second {
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
        for index in 0..self.players.len() {
            let action = Action {
                angle: actions[index].angle,
                buttons: (actions[index].buttons & 0xFF) as u8,
            };
            self.player_tick(index, action, data);
        }
        // Slime-vs-slime shove (World::step parity).
        for a in 0..self.players.len() {
            for b in a + 1..self.players.len() {
                let (pa, pb) = (a, b);
                if !self.playable(pa) || !self.playable(pb) {
                    continue;
                }
                let a_pos = self.players[pa].base.position;
                let b_pos = self.players[pb].base.position;
                let a_dash = self.players[pa].base.dash.active;
                let b_dash = self.players[pb].base.dash.active;
                if !a_dash {
                    self.players[pa].base.push_from(b_pos, true);
                }
                if !b_dash {
                    self.players[pb].base.push_from(a_pos, false);
                }
            }
        }
        for index in 0..self.players.len() {
            if self.playable(index) {
                self.players[index].base.sync_collider();
                self.players[index].base.late_step(&self.platforms);
            }
        }
        self.arrows_tick(data);
        self.platforms_tick(data);
        self.deaths_tick();
        self.tick += 1;
        self.episode_ticks += 1;
        self.session_tick();
    }

    fn playable(&self, index: usize) -> bool {
        let slot = &self.players[index];
        slot.base.alive && slot.respawn_ticks == 0
    }

    /// SlimeController.OldUpdate: cooldowns, ability entry, then movement.
    fn player_tick(&mut self, index: usize, action: Action, data: &data::RolloutData) {
        let in_ability = {
            let slot = &self.players[index];
            slot.active != ActiveAbility::None || slot.base.dash.active
        };
        if !self.playable(index) {
            self.respawn_tick(index);
            return;
        }
        if !in_ability {
            for s in 0..3 {
                self.players[index].timers[s] = self.players[index].timers[s] + DT;
            }
        } else {
            self.players[index].ability_time = self.players[index].ability_time + DT;
        }

        // Ability entry (held button, ready, castable, not already in one).
        if !in_ability {
            self.players[index].ability_time = Fix::ZERO;
            for s in 0..3 {
                let slot = &self.players[index];
                let button_down = action.buttons & (2 << s) as u8 != 0;
                if !button_down || slot.timers[s] < slot.cooldowns[s] {
                    continue;
                }
                match slot.abilities[s] {
                    ABILITY_DASH => {
                        // The movement slice implements the dash; unlock its
                        // internal cooldown gate and let base.step enter it.
                        self.players[index].base.cooldown = Fix::from_int(100);
                        self.players[index].timers[s] = Fix::ZERO;
                        self.players[index].active = ActiveAbility::Dash;
                        self.players[index].active_slot = s;
                        self.players[index].telemetry.ability_activations[s] += 1;
                        let masked = Action {
                            angle: action.angle,
                            buttons: (action.buttons & 1) | 2,
                        };
                        self.dash_move(index, masked);
                        return;
                    }
                    // Bow and Platform activate in later milestones; other
                    // catalog abilities are never simulated here.
                    _ => {}
                }
            }
        }

        if self.players[index].base.dash.active {
            let masked = Action {
                angle: action.angle,
                buttons: action.buttons & 1,
            };
            self.dash_move(index, masked);
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
        self.players[index].base.step(masked, &self.platforms);
        let _ = data;
    }

    /// base.step with the dash shove check from World::step.
    fn dash_move(&mut self, index: usize, action: Action) {
        self.players[index].base.step(action, &self.platforms);
        let caster = &self.players[index].base;
        if caster.dash.active && caster.dash.time == DT {
            let origin = caster.position;
            let direction = caster.dash.input;
            let wall = crate::world::closest(
                &self.platforms,
                origin,
                direction,
                Fix::from_int(13),
                None,
            )
            .map(|(_, distance, _)| distance);
            let targets: Vec<usize> = self
                .players
                .iter()
                .enumerate()
                .filter(|(i, p)| *i != index && p.base.alive && !p.base.dash.active)
                .filter_map(|(i, p)| {
                    p.base
                        .collider
                        .raycast(origin, direction)
                        .filter(|hit| {
                            hit.near <= Fix::from_int(13)
                                && wall.is_none_or(|distance| hit.near < distance)
                        })
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

    fn arrows_tick(&mut self, _data: &data::RolloutData) {
        // Milestone 2 (bow).
    }

    /// Map platform motion: sudden death sinking (AnimateVelocity), and the
    /// map-2 orbiter (AntiLockPlatform) later.
    fn platforms_tick(&mut self, _data: &data::RolloutData) {
        if !self.sudden_death {
            return;
        }
        let map = &data::get().maps[self.map];
        let elapsed = Fix::from_int(i64::from(self.tick - self.sudden_death_tick)) * DT;
        for (i, def_index) in self.platform_defs.iter().enumerate() {
            if *def_index == usize::MAX {
                continue;
            }
            let def = &map.platforms[*def_index];
            let delay = self.sudden_death_delays[*def_index];
            // AnimateVelocity.UpdateSim: home -= down * curve(t - delay)
            //   * suddenDeathSpeed * (2 << min(t * accel, 5)) * dt.
            let steps = (elapsed * def.sudden_death_acceleration).to_int().clamp(0, 5);
            let scale = def.sudden_death_speed * Fix::from_int(2 << steps);
            let speed = curve_eval(&def.sudden_death_y_speed, elapsed - delay);
            let drop = speed * scale * DT;
            if drop == Fix::ZERO {
                continue;
            }
            let platform = &mut self.platforms[i];
            platform.surface.center.y = platform.surface.center.y - drop;
            platform.collider.inner.center.y = platform.collider.inner.center.y - drop;
        }
    }

    fn deaths_tick(&mut self) {
        for index in 0..self.players.len() {
            let slot = &mut self.players[index];
            if !slot.base.alive || slot.respawn_ticks > 0 {
                continue;
            }
            // The movement slice already kills on the blast zones and water
            // (DestroyIfOutsideSceneBounds); death cause is environmental.
            if !slot.base.alive || slot.base.outside_bounds() {
                slot.base.alive = false;
                self.kill(index, ENVIRONMENT, CAUSE_ACCIDENT);
            }
        }
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
        if killer >= 1 && killer <= players as u32 && killer != slot.base.id {
            self.players[killer as usize - 1].kills += 1;
        }
        let slot = &mut self.players[index];
        // Stocks: consume one and schedule the respawn (StockRespawn: a
        // 360-tick chooser; here a fixed 360-tick hold, then the team spawn).
        if slot.stocks > 0 && !self.sudden_death {
            slot.stocks -= 1;
            slot.respawn_ticks = 360;
            slot.base.alive = true; // still in the match, body held offscreen
            slot.base.position = Vec2::new(Fix::ZERO, Fix::from_int(1000));
            slot.base.velocity = Vec2::ZERO;
            slot.base.grounded = false;
            slot.base.ground = None;
        } else {
            slot.out = true;
        }
    }

    fn respawn_tick(&mut self, index: usize) {
        let slot = &mut self.players[index];
        if !slot.base.alive || slot.respawn_ticks == 0 {
            return;
        }
        slot.respawn_ticks -= 1;
        if slot.respawn_ticks == 0 {
            let map = &data::get().maps[self.map];
            let spawn = map.team_spawns[slot.team as usize % map.team_spawns.len()];
            slot.base.position = spawn;
            slot.base.velocity = Vec2::ZERO;
            slot.base.external_velocity = Vec2::ZERO;
            slot.base.grounded = false;
            slot.base.ground = None;
            slot.base.grounded_speed = Fix::ZERO;
            slot.cause_of_death = CAUSE_NOT_DEAD_YET;
        }
    }

    /// GameSessionHandler.UpdateSim: sudden death activation and win checks.
    fn session_tick(&mut self) {
        let map = &data::get().maps[self.map];
        let sim_time = Fix::from_int(i64::from(self.tick)) * DT;
        if !self.sudden_death && sim_time > map.sudden_death_time {
            self.sudden_death = true;
            self.sudden_death_tick = self.tick;
        }
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
            let dashing = slot.base.dash.active;
            let position = slot.base.position;
            let rotation = slot.base.rotation;
            let velocity = slot.base.velocity + slot.base.external_velocity;
            let basis = Vec2::angle(rotation);
            let data = data::get();

            let body = &mut self.entities[body_e];
            if !alive {
                body.flags = E_ENABLED | E_TRIGGER | E_DESTROYED;
            } else if dashing {
                body.flags = E_ENABLED | E_TRIGGER;
            } else {
                body.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
            }
            let extents = data.player.hurtbox_extents;
            body.x = position.x.0;
            body.y = position.y.0;
            body.vx = velocity.x.0;
            body.vy = velocity.y.0;
            let right = Vec2::new(extents.x, Fix::ZERO).complex_mul(basis);
            let up = Vec2::new(Fix::ZERO, extents.y).complex_mul(basis);
            body.right_x = right.x.0;
            body.right_y = right.y.0;
            body.up_x = up.x.0;
            body.up_y = up.y.0;
            body.rotation = rotation.0;

            let dash_box = &mut self.entities[dash_e];
            if alive && dashing {
                let extents = data.dash.hurtbox_extents;
                let angle = slot.base.dash.input.normalized_angle();
                let basis = Vec2::angle(angle);
                dash_box.flags = E_ACTIVE | E_ENABLED | E_TRIGGER;
                dash_box.x = position.x.0;
                dash_box.y = position.y.0;
                dash_box.vx = velocity.x.0;
                dash_box.vy = velocity.y.0;
                let right = Vec2::new(extents.x, Fix::ZERO).complex_mul(basis);
                let up = Vec2::new(Fix::ZERO, extents.y).complex_mul(basis);
                dash_box.right_x = right.x.0;
                dash_box.right_y = right.y.0;
                dash_box.up_x = up.x.0;
                dash_box.up_y = up.y.0;
                dash_box.rotation = angle.0;
            } else {
                dash_box.flags = E_ENABLED | E_TRIGGER;
                dash_box.x = 0;
                dash_box.y = 0;
                dash_box.vx = 0;
                dash_box.vy = 0;
            }
            let circle = &mut self.entities[circle_e];
            circle.flags = E_ENABLED | E_TRIGGER;
        }
        for (i, platform) in self.platforms.iter().enumerate() {
            let e = &mut self.entities[self.platform_entities[i]];
            let s = &platform.surface;
            e.x = s.center.x.0;
            e.y = s.center.y.0;
            let right = s.right * s.extents.x;
            let up = s.up * s.extents.y;
            e.right_x = right.x.0;
            e.right_y = right.y.0;
            e.up_x = up.x.0;
            e.up_y = up.y.0;
            e.radius = s.radius.0;
            e.rotation = s.right.normalized_angle().0;
        }
    }

    fn fill_slimes(&self, slimes: &mut Vec<CSlime>) {
        slimes.clear();
        for slot in &self.players {
            if !slot.base.alive {
                continue; // destroyed with the player, like the .NET sim
            }
            let in_ability = slot.active != ActiveAbility::None;
            let mut flags = S_ACTIVE | S_SPAWNED;
            if in_ability {
                flags |= S_IN_ABILITY;
            } else {
                flags |= S_SIMULATED;
            }
            let state = if in_ability {
                STATE_ABILITY
            } else if !slot.base.grounded {
                STATE_FALL
            } else if slot.base.grounded_speed.abs() > Fix::from_f32(0.5) {
                STATE_RUN
            } else {
                STATE_IDLE
            };
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
                        f |= (slot.bow.loading_frame & 7) << 4;
                        if slot.bow.has_fired {
                            f |= 0x80;
                        }
                    }
                }
                ability_flags[s] = f;
            }
            let velocity = slot.base.velocity + slot.base.external_velocity;
            slimes.push(CSlime {
                id: self.entities[slot.body_entity].id,
                owner: slot.base.id,
                flags,
                state,
                x: slot.base.position.x.0,
                y: slot.base.position.y.0,
                vx: velocity.x.0,
                vy: velocity.y.0,
                rotation: slot.base.rotation.0,
                scale: 1 << 32,
                abilities: slot.abilities,
                ability_flags,
                cooldowns: [
                    slot.cooldowns[0].0,
                    slot.cooldowns[1].0,
                    slot.cooldowns[2].0,
                ],
                timers: [slot.timers[0].0, slot.timers[1].0, slot.timers[2].0],
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
            simulation_time: (Fix::from_int(i64::from(tick)) * DT).0,
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
            state.players[i] = CPlayer {
                id: slot.base.id,
                team: slot.team,
                flags: player_flags,
                alive_bodies: u32::from(slot.base.alive),
                stocks: slot.stocks,
                kills: slot.kills,
                deaths: slot.deaths,
                cause_of_death: slot.cause_of_death,
                x: slot.base.position.x.0,
                y: slot.base.position.y.0,
                scale: 1 << 32,
                invisible_time: 0,
                ability_time: slot.ability_time.0,
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

/// Linear interpolation over serialized AnimationCurveFixed keys (tangents are
/// not exported; the sudden-death sink curve is smooth enough for this).
fn curve_eval(keys: &[(Fix, Fix)], t: Fix) -> Fix {
    if keys.is_empty() {
        return Fix::ZERO;
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for pair in keys.windows(2) {
        let (t0, v0) = pair[0];
        let (t1, v1) = pair[1];
        if t <= t1 {
            if t1 == t0 {
                return v1;
            }
            let alpha = (t - t0) / (t1 - t0);
            return v0 + (v1 - v0) * alpha;
        }
    }
    keys[keys.len() - 1].1
}
