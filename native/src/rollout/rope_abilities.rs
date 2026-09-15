//! Shared fixed-capacity rope runtime for Web Swing and Grappling Hook.

use super::*;
use crate::physics::rope::{Rope, RopeConfig};

const ROPE_POOL: usize = 128;
const MAX_SEGMENTS: usize = 40;
const MAX_CONTACTS: usize = 32;
const SEGMENTS: usize = 12;

const ROPE_GRAVITY: Fix = Fix::from_raw(8_589_934_592);
const ROPE_ITERATIONS: usize = 7;
const PLATFORM_STRETCHINESS: Fix = Fix::from_raw(171_798_688);
const MAX_STRETCH: Fix = Fix::from_raw(429_496_729_600);
const GROUNDED_ROPE_STRENGTH: Fix = Fix::from_raw(42_949_672_960);
const ROPE_SEGMENT_RADIUS: Fix = Fix::from_raw(429_496_736);

const PLATFORM_MASS: Fix = Fix::from_raw(3_865_470_464);
const PLATFORM_HEAVINESS: Fix = Fix::from_raw(42_949_672);
const PLATFORM_PULL: Fix = Fix::from_raw(-42_949_672_960);
const PLATFORM_BREAK_STRETCH: Fix = Fix::from_raw(429_496_729_600);

const WEB_MAX_LENGTH: Fix = Fix::from_raw(85_899_345_920);
const WEB_SHORTEN: Fix = Fix::from_raw(3_693_671_710);
const WEB_COOLDOWN: Fix = Fix::from_raw(5_368_709_120);
const WEB_MISS_COOLDOWN: Fix = Fix::from_raw(1_932_735_283);
const WEB_RELEASE_BOOST: Fix = Fix::from_raw(4_810_363_371);
const WEB_GROUNDED_POP: Fix = Fix::from_raw(15_032_385_536);
const WEB_LINGER: Fix = Fix::from_raw(3_865_470_464);

#[derive(Clone, Copy, Default)]
pub(crate) struct WebSwingState {
    pub rope: Option<usize>,
    pub rope_number: u32,
    pub cooldown_set: bool,
    pub fire_tick: i32,
    pub linger_left: Fix,
}

#[derive(Clone)]
pub(crate) struct RolloutRope {
    pub physics: Rope,
    pub owner: usize,
    pub number: u32,
    pub born_tick: u32,
    pub disabled_this_frame: bool,
    pub time_air_start: Fix,
    pub time_air_end: Fix,
    pub platform: Option<usize>,
    pub platform_local: Fix,
    pub segment_ids: [u32; MAX_SEGMENTS - 1],
}

fn empty_rope() -> RolloutRope {
    RolloutRope {
        physics: Rope {
            segment: vec![Vec2::ZERO; MAX_SEGMENTS],
            previous: vec![Vec2::ZERO; MAX_SEGMENTS],
            count: 0,
            enabled: false,
            force_start: Vec2::ZERO,
            force_end: Vec2::ZERO,
            mass_start: Fix::ZERO,
            mass_end: Fix::ZERO,
            separation: Fix::ZERO,
            time_since_hook_arrived: Fix::ZERO,
            detached: false,
            colliding_rects: vec![0; MAX_CONTACTS],
            collision_count: 0,
        },
        owner: 0,
        number: 0,
        born_tick: 0,
        disabled_this_frame: false,
        time_air_start: Fix::ZERO,
        time_air_end: Fix::ZERO,
        platform: None,
        platform_local: Fix::ZERO,
        segment_ids: [0; MAX_SEGMENTS - 1],
    }
}

pub(super) fn rope_pool() -> Vec<RolloutRope> {
    let source = &data::get().ropes;
    assert_eq!(source.pool_capacity, ROPE_POOL);
    assert_eq!(source.max_segments, MAX_SEGMENTS);
    assert_eq!(source.max_contacts, MAX_CONTACTS);
    assert_eq!(source.segments, SEGMENTS);
    assert_eq!(source.gravity, ROPE_GRAVITY);
    assert_eq!(source.iterations, ROPE_ITERATIONS);
    assert_eq!(source.stretchiness, PLATFORM_STRETCHINESS);
    assert_eq!(source.max_stretch, MAX_STRETCH);
    assert_eq!(source.grounded_strength, GROUNDED_ROPE_STRENGTH);
    assert_eq!(source.segment_radius, ROPE_SEGMENT_RADIUS);
    assert_eq!(source.player_gravity, Fix::from_int(10));
    assert_eq!(source.platform_mass, PLATFORM_MASS);
    assert_eq!(source.platform_heaviness, PLATFORM_HEAVINESS);
    assert_eq!(source.platform_pull, PLATFORM_PULL);
    assert_eq!(source.platform_break_stretch, PLATFORM_BREAK_STRETCH);
    assert_eq!(source.non_platform_mass, Fix::from_f64(0.5));
    assert_eq!(source.web_max_length, WEB_MAX_LENGTH);
    assert_eq!(source.web_shorten, WEB_SHORTEN);
    assert_eq!(source.web_cooldown, WEB_COOLDOWN);
    assert_eq!(source.web_miss_cooldown, WEB_MISS_COOLDOWN);
    assert_eq!(source.web_release_boost, WEB_RELEASE_BOOST);
    assert_eq!(source.web_grounded_pop, WEB_GROUNDED_POP);
    (0..ROPE_POOL).map(|_| empty_rope()).collect()
}

pub(super) fn reset_rope_pool(ropes: &mut [RolloutRope]) {
    for rope in ropes {
        rope.physics.enabled = false;
        rope.physics.count = 0;
        rope.disabled_this_frame = false;
        rope.number = 0;
        rope.segment_ids = [0; MAX_SEGMENTS - 1];
        rope.platform = None;
    }
}

impl RolloutWorld {
    fn allocate_rope(
        &mut self,
        owner: usize,
        anchor: Vec2,
        player: Vec2,
        separation: Fix,
        platform: usize,
        platform_local: Fix,
    ) -> usize {
        let index = self
            .ropes
            .iter()
            .position(|rope| !rope.physics.enabled && !rope.disabled_this_frame)
            .expect("RopeBody pool exhausted");
        self.next_rope_number += 1;
        let mut segment_ids = [0; MAX_SEGMENTS - 1];
        for id in segment_ids.iter_mut().take(SEGMENTS - 1) {
            *id = self.next_id();
        }
        let rope = &mut self.ropes[index];
        rope.owner = owner;
        rope.number = self.next_rope_number;
        rope.born_tick = self.tick;
        rope.disabled_this_frame = false;
        rope.time_air_start = Fix::ZERO;
        rope.time_air_end = if self.players[owner].base.grounded {
            Fix::ZERO
        } else {
            Fix::from_int(100)
        };
        rope.platform = Some(platform);
        rope.platform_local = platform_local;
        rope.segment_ids = segment_ids;
        let body = &mut rope.physics;
        body.count = SEGMENTS;
        body.enabled = true;
        body.force_start = Vec2::ZERO;
        body.force_end = Vec2::ZERO;
        body.mass_start = Fix::ONE;
        body.mass_end = Fix::ZERO;
        body.separation = separation;
        body.time_since_hook_arrived = Fix::ZERO;
        body.detached = false;
        body.collision_count = 0;
        for j in 0..SEGMENTS {
            let point = anchor
                + (player - anchor) * Fix::from_int(j as i64) / Fix::from_int(SEGMENTS as i64);
            body.segment[j] = point;
            body.previous[j] = point;
        }
        index
    }

    fn attach_player_rope(&mut self, player: usize, rope: Option<usize>, top: bool) {
        if let Some(old) = self.players[player].rope {
            let old_top = self.players[player].rope_top;
            let old_rope = &mut self.ropes[old].physics;
            old_rope.force_end = if old_top {
                old_rope.force_end
            } else {
                -Vec2::UP * ROPE_GRAVITY
            };
            old_rope.force_start = if old_top {
                -Vec2::UP * ROPE_GRAVITY
            } else {
                old_rope.force_start
            };
            if rope.is_none() {
                old_rope.detach(old_top);
            } else {
                old_rope.set_mass(Fix::from_f64(0.5), old_top, Fix::ONE);
            }
            if old_top {
                self.ropes[old].time_air_start = Fix::from_int(100);
            } else {
                self.ropes[old].time_air_end = Fix::from_int(100);
            }
        }
        let slot = &mut self.players[player];
        slot.rope_prev_imposed = slot.base.velocity;
        slot.rope_prev_external = slot.base.external_velocity;
        slot.rope = rope;
        slot.rope_top = top;
    }

    pub(super) fn web_cooldowns_tick(&mut self, player: usize, action: Action) {
        for ability_slot in 0..3 {
            if self.players[player].abilities[ability_slot] != ABILITY_WEB_SWING {
                continue;
            }
            let mut valid = {
                let state = self.players[player].web_swing[ability_slot];
                state.rope.is_some_and(|rope| {
                    self.ropes[rope].physics.enabled
                        && self.ropes[rope].number == state.rope_number
                        && self.ropes[rope].owner == player
                        && self.players[player].rope == Some(rope)
                })
            };
            let cooldown_set = self.players[player].web_swing[ability_slot].cooldown_set;
            if !cooldown_set {
                let fire_tick = self.players[player].web_swing[ability_slot].fire_tick;
                if valid
                    && self.tick as i32 - fire_tick > 6
                    && action.buttons & (2 << ability_slot) as u8 == 0
                {
                    self.release_web(player, ability_slot);
                    valid = false;
                }
                if !valid {
                    let state = &mut self.players[player].web_swing[ability_slot];
                    state.cooldown_set = true;
                    state.linger_left = WEB_LINGER;
                    self.players[player].cooldowns[ability_slot] = WEB_COOLDOWN;
                } else {
                    self.players[player].cooldowns[ability_slot] = Fix::from_int(10_000);
                }
            }
            let state = self.players[player].web_swing[ability_slot];
            if state.linger_left > Fix::ZERO {
                let left = state.linger_left - DT;
                self.players[player].web_swing[ability_slot].linger_left = left;
                if left <= Fix::ZERO {
                    if let Some(rope) = state.rope {
                        self.disable_rope(rope);
                    }
                    self.players[player].web_swing[ability_slot].rope = None;
                }
            }
        }
    }

    fn release_web(&mut self, player: usize, ability_slot: usize) {
        let Some(rope_index) = self.players[player].web_swing[ability_slot].rope else {
            return;
        };
        let top = self.players[player].rope_top;
        let rope = &self.ropes[rope_index].physics;
        let endpoint = if top { 0 } else { rope.count - 1 };
        self.players[player].base.velocity =
            (rope.segment[endpoint] - rope.previous[endpoint]) / DT * WEB_RELEASE_BOOST;
        self.attach_player_rope(player, None, top);
    }

    pub(super) fn fire_web(&mut self, player: usize, ability_slot: usize) {
        self.players[player].timers[ability_slot] = Fix::ZERO;
        self.players[player].telemetry.ability_activations[ability_slot] += 1;
        let x = if self.players[player].facing_right {
            Fix::from_raw(2_465_311_232)
        } else {
            Fix::from_raw(-2_465_311_232)
        };
        let direction = Vec2::new(x, Fix::from_raw(3_517_578_240)).normalized_safe();
        let position = self.players[player].base.position;
        let terrain = Terrain {
            platforms: &self.platforms,
            bounds: &self.bounds,
        };
        let hit = crate::world::closest(
            terrain,
            position + direction * Fix::from_raw(4_724_464_128),
            direction,
            WEB_MAX_LENGTH,
            None,
        );
        let Some((platform, _, near)) = hit else {
            self.players[player].cooldowns[ability_slot] = WEB_MISS_COOLDOWN;
            self.players[player].web_swing[ability_slot].cooldown_set = true;
            return;
        };
        let length = (near - position).length();
        let separation = Fix::from_raw(1_503_238_528)
            .max(length * WEB_SHORTEN / Fix::from_int((SEGMENTS - 1) as i64));
        let local = self.platforms[platform]
            .surface
            .local_position(near, Fix::ZERO)
            .expect("web ray endpoint lies on platform");
        let rope = self.allocate_rope(player, near, position, separation, platform, local);
        let number = self.ropes[rope].number;
        self.attach_player_rope(player, None, false);
        self.attach_player_rope(player, Some(rope), false);
        if self.players[player].base.grounded {
            self.players[player].base.force_unground(&self.platforms);
            self.players[player].base.rotation = Fix::ZERO;
            self.players[player].base.up = Vec2::UP;
            self.players[player].base.velocity += Vec2::UP * WEB_GROUNDED_POP;
        }
        self.players[player].web_swing[ability_slot] = WebSwingState {
            rope: Some(rope),
            rope_number: number,
            cooldown_set: false,
            fire_tick: self.tick as i32,
            linger_left: Fix::ZERO,
        };
    }

    pub(super) fn rope_player_move(
        &mut self,
        player: usize,
        action: Action,
        input: Vec2,
        jump_pressed: bool,
    ) {
        if jump_pressed && self.players[player].base.grounded {
            let strength = self.players[player].base.constants.jump_strength;
            self.players[player]
                .base
                .jump_with_strength(&self.platforms, strength);
        }
        self.players[player].base.previous_jump = action.buttons & 1 != 0;
        self.players[player].base.move_only(
            input,
            Terrain {
                platforms: &self.platforms,
                bounds: &self.bounds,
            },
        );
        if self.players[player].base.outside_bounds() {
            self.players[player].base.alive = false;
            return;
        }
        let Some(rope_index) = self.players[player].rope else {
            return;
        };
        if !self.ropes[rope_index].physics.enabled {
            self.players[player].rope = None;
            return;
        }
        let top = self.players[player].rope_top;
        if jump_pressed
            && self.ropes[rope_index].owner == player
            && !self.players[player].base.grounded
            && !self.players[player].base.jumped_this_frame
        {
            let endpoint = if top {
                0
            } else {
                self.ropes[rope_index].physics.count - 1
            };
            let rope = &self.ropes[rope_index].physics;
            self.players[player].base.velocity =
                (rope.segment[endpoint] - rope.previous[endpoint]) / DT;
            self.attach_player_rope(player, None, top);
        }
        let velocity =
            self.players[player].base.velocity + self.players[player].base.external_velocity;
        self.players[player].base.position += velocity * DT;
        if self.players[player].rope == Some(rope_index) {
            let endpoint = if top {
                0
            } else {
                self.ropes[rope_index].physics.count - 1
            };
            if self.ropes[rope_index].physics.detached {
                self.ropes[rope_index].physics.segment[endpoint] =
                    self.players[player].base.position;
                self.attach_player_rope(player, None, top);
            } else if self.ropes[rope_index].physics.time_since_hook_arrived <= Fix::ZERO {
                self.ropes[rope_index].physics.segment[endpoint] =
                    self.players[player].base.position;
                if top {
                    self.ropes[rope_index].time_air_start = Fix::ZERO;
                } else {
                    self.ropes[rope_index].time_air_end = Fix::ZERO;
                }
                self.ropes[rope_index]
                    .physics
                    .set_mass(Fix::ONE, top, Fix::ONE);
                if top {
                    self.ropes[rope_index].physics.force_start = Vec2::ZERO;
                } else {
                    self.ropes[rope_index].physics.force_end = Vec2::ZERO;
                }
            } else if self.players[player].base.grounded {
                let mass = Fix::ONE
                    + (Fix::from_f64(0.9) - Fix::ONE)
                        * self.ropes[rope_index]
                            .physics
                            .time_since_hook_arrived
                            .clamp(Fix::ZERO, Fix::ONE);
                if top {
                    self.ropes[rope_index].time_air_start = Fix::ZERO;
                } else {
                    self.ropes[rope_index].time_air_end = Fix::ZERO;
                }
                self.ropes[rope_index].physics.set_mass(mass, top, Fix::ONE);
            } else {
                let time = if top {
                    self.ropes[rope_index].time_air_start += DT;
                    self.ropes[rope_index].time_air_start
                } else {
                    self.ropes[rope_index].time_air_end += DT;
                    self.ropes[rope_index].time_air_end
                };
                self.ropes[rope_index]
                    .physics
                    .set_mass(Fix::from_f64(0.5), top, Fix::ONE);
                let amount = (time * Fix::from_int(2)).clamp(Fix::ZERO, Fix::ONE);
                let offset = self.ropes[rope_index].physics.segment[endpoint]
                    - self.players[player].base.position;
                let length = offset.length();
                if length != Fix::ZERO {
                    let correction = self.ropes[rope_index].physics.separation - length;
                    let direction = offset / length;
                    self.ropes[rope_index].physics.segment[endpoint] +=
                        direction * correction * (Fix::ONE - amount);
                    self.players[player].base.position -= direction * correction * amount;
                }
            }
            self.players[player].rope_prev_imposed = self.players[player].base.velocity;
            self.players[player].rope_prev_external = self.players[player].base.external_velocity;
        }
        self.players[player].base.external_velocity *= Fix::from_f64(0.95);
    }

    pub(super) fn ropes_update(&mut self) {
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        let mut moved_platform = false;
        for index in 0..self.ropes.len() {
            if !self.ropes[index].physics.enabled || self.ropes[index].born_tick == self.tick {
                continue;
            }
            if self.ropes[index].physics.time_since_hook_arrived >= Fix::ZERO {
                self.ropes[index].physics.time_since_hook_arrived += DT * self.player_time_scale;
            }
            let Some(platform) = self.ropes[index].platform else {
                continue;
            };
            if platform >= self.platforms.len()
                || self.entities[self.platform_entities[platform]].flags & E_ACTIVE == 0
            {
                self.ropes[index].physics.detach(true);
                self.ropes[index].platform = None;
                continue;
            }
            let anchor = self.platforms[platform]
                .surface
                .position(self.ropes[index].platform_local, Fix::ZERO);
            if self.ropes[index].physics.time_since_hook_arrived > Fix::ZERO
                && !self.ropes[index].physics.detached
                && !stopped
            {
                let force = (self.ropes[index].physics.segment[0]
                    - self.ropes[index].physics.segment[1])
                    * PLATFORM_PULL;
                let arm = (anchor - self.platforms[platform].surface.center).normalized_safe();
                self.platform_dynamics[platform]
                    .body
                    .apply_impulse(force, arm);
                moved_platform = true;
            }
            self.ropes[index].time_air_start = Fix::ZERO;
            let scale = PLATFORM_HEAVINESS / self.platform_dynamics[platform].body.inverse_mass;
            self.ropes[index]
                .physics
                .set_mass(PLATFORM_MASS, true, scale);
            self.ropes[index].physics.force_start = Vec2::ZERO;
            self.ropes[index].physics.segment[0] = anchor;
            for segment in 0..self.ropes[index].physics.count.saturating_sub(2) {
                if (self.ropes[index].physics.segment[segment]
                    - self.ropes[index].physics.segment[segment + 1])
                    .length_squared()
                    > PLATFORM_BREAK_STRETCH
                {
                    self.ropes[index].physics.detach(true);
                    self.ropes[index].platform = None;
                    break;
                }
            }
        }
        if moved_platform {
            self.platforms_changed();
        }
    }

    pub(super) fn ropes_physics(&mut self, physics_dt: Fix) {
        self.rope_rectangles.clear();
        self.rope_rectangles
            .extend(self.platforms.iter().map(|platform| platform.collider));
        for rope in &mut self.ropes {
            if !rope.physics.enabled {
                continue;
            }
            rope.physics.collision_count = 0;
            let bounds = rope.physics.bounds();
            for (platform, rectangle) in self.rope_rectangles.iter().enumerate() {
                let rb = rectangle.bounds();
                if bounds[1].x <= rb.min.x
                    || bounds[0].x >= rb.max.x
                    || bounds[1].y <= rb.min.y
                    || bounds[0].y >= rb.max.y
                {
                    continue;
                }
                if rope.physics.collision_count == MAX_CONTACTS {
                    break;
                }
                let at = rope.physics.collision_count;
                rope.physics.colliding_rects[at] = platform;
                rope.physics.collision_count += 1;
            }
            rope.physics.step(
                RopeConfig {
                    dt: physics_dt,
                    gravity: ROPE_GRAVITY,
                    space: false,
                    iterations: ROPE_ITERATIONS,
                    stretchiness: PLATFORM_STRETCHINESS,
                },
                &self.rope_rectangles,
            );
        }
    }

    pub(super) fn ropes_late(&mut self) {
        for player in 0..self.players.len() {
            let Some(rope_index) = self.players[player].rope else {
                continue;
            };
            if !self.ropes[rope_index].physics.enabled {
                self.players[player].rope = None;
                continue;
            }
            let top = self.players[player].rope_top;
            let endpoint = if top {
                0
            } else {
                self.ropes[rope_index].physics.count - 1
            };
            if self.players[player].base.grounded {
                let delta = self.ropes[rope_index].physics.segment[endpoint]
                    - self.players[player].base.position;
                if delta.length_squared() >= Fix::from_raw(300_647_711) {
                    self.players[player].base.external_velocity += delta * GROUNDED_ROPE_STRENGTH;
                }
                if let Some(platform) = self.players[player].base.platform(&self.platforms)
                    && platform
                        .surface
                        .normal(self.players[player].base.position)
                        .dot(self.players[player].base.external_velocity)
                        > Fix::from_int(30)
                {
                    self.players[player].base.force_unground(&self.platforms);
                }
                self.ropes[rope_index].physics.segment[endpoint] =
                    self.players[player].base.position;
            }
            for segment in 0..self.ropes[rope_index].physics.count.saturating_sub(2) {
                if (self.ropes[rope_index].physics.segment[segment]
                    - self.ropes[rope_index].physics.segment[segment + 1])
                    .length_squared()
                    > MAX_STRETCH
                {
                    self.attach_player_rope(player, None, top);
                    break;
                }
            }
            if self.players[player].rope == Some(rope_index) {
                let imposed = self.players[player].base.velocity
                    - self.players[player].rope_prev_imposed
                    - Vec2::UP * Fix::from_int(10);
                let external = self.players[player].base.external_velocity
                    - self.players[player].rope_prev_external;
                if top {
                    self.ropes[rope_index].physics.force_start = imposed + external;
                } else {
                    self.ropes[rope_index].physics.force_end = imposed + external;
                }
            }
        }
        for rope in &mut self.ropes {
            if !rope.physics.enabled {
                continue;
            }
            // A prefab registered from WebSwing.Cast is absent from the
            // current Updater list, so its RopeAttachment has no late pass
            // on the cast tick. PlayerBody is an authored component and does
            // pin the player endpoint immediately.
            if rope.born_tick != self.tick
                && let Some(platform) = rope.platform
            {
                rope.physics.segment[0] = self.platforms[platform]
                    .surface
                    .position(rope.platform_local, Fix::ZERO);
            }
            rope.disabled_this_frame = false;
        }
    }

    pub(super) fn disable_rope(&mut self, rope: usize) {
        if self.ropes[rope].physics.enabled {
            self.ropes[rope].physics.enabled = false;
            self.ropes[rope].disabled_this_frame = true;
        }
    }

    pub(super) fn append_rope_entities(&self, out: &mut Vec<CEntity>) {
        for rope_index in 0..self.ropes.len() {
            if !self.ropes[rope_index].physics.enabled {
                continue;
            }
            let count = self.ropes[rope_index].physics.count.saturating_sub(1);
            for segment in 0..count {
                let rope = &self.ropes[rope_index];
                let start = rope.physics.segment[segment];
                let right = rope.physics.segment[segment + 1] - start;
                out.push(CEntity {
                    id: rope.segment_ids[segment],
                    kind: 3,
                    owner: rope.owner as u32 + 1,
                    flags: E_ACTIVE | E_ENABLED | if rope.physics.detached { 32 } else { 0 },
                    hierarchy: segment as u32,
                    x: start.x.raw(),
                    y: start.y.raw(),
                    right_x: right.x.raw(),
                    right_y: right.y.raw(),
                    radius: ROPE_SEGMENT_RADIUS.raw(),
                    ..CEntity::default()
                });
            }
        }
    }
}
