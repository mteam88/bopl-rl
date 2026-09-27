//! Shared fixed-capacity rope runtime for Web Swing and Grappling Hook.

use super::*;
use crate::physics::rope::{Rope, RopeConfig};

const ROPE_POOL: usize = 128;
const MAX_SEGMENTS: usize = 40;
const MAX_CONTACTS: usize = 32;
const SEGMENTS: usize = 12;

pub(super) const ROPE_GRAVITY: Fix = Fix::from_raw(8_589_934_592);
const ROPE_ITERATIONS: usize = 7;
const PLATFORM_STRETCHINESS: Fix = Fix::from_raw(171_798_688);
const MAX_STRETCH: Fix = Fix::from_raw(429_496_729_600);
const GROUNDED_ROPE_STRENGTH: Fix = Fix::from_raw(42_949_672_960);
const ROPE_SEGMENT_RADIUS: Fix = Fix::from_raw(429_496_736);

const PLATFORM_MASS: Fix = Fix::from_raw(3_865_470_464);
const PLATFORM_HEAVINESS: Fix = Fix::from_raw(42_949_672);
const PLATFORM_PULL: Fix = Fix::from_raw(-42_949_672_960);
const BOULDER_PULL: Fix = Fix::from_raw(-21_474_836_480);
const PLATFORM_BREAK_STRETCH: Fix = Fix::from_raw(429_496_729_600);
/// RopeAttachment.pullStr, the pull on a non-platform parent's body.
const ATTACHMENT_PULL: Fix = Fix::from_raw(-12_884_901_888);

const WEB_MAX_LENGTH: Fix = Fix::from_raw(85_899_345_920);
const WEB_SHORTEN: Fix = Fix::from_raw(3_693_671_710);
const WEB_COOLDOWN: Fix = Fix::from_raw(5_368_709_120);
const WEB_MISS_COOLDOWN: Fix = Fix::from_raw(1_932_735_283);
const WEB_RELEASE_BOOST: Fix = Fix::from_raw(4_810_363_371);
const WEB_GROUNDED_POP: Fix = Fix::from_raw(15_032_385_536);
const WEB_LINGER: Fix = Fix::from_raw(3_865_470_464);

#[derive(Clone, Copy)]
pub(crate) struct WebSwingState {
    pub rope: Option<usize>,
    pub rope_number: u32,
    pub cooldown_set: bool,
    pub fire_tick: i32,
    pub linger_left: Fix,
}

impl Default for WebSwingState {
    /// WebSwing.hasHadCooldownSet starts true: no strand to clean up yet.
    fn default() -> Self {
        Self {
            rope: None,
            rope_number: 0,
            cooldown_set: true,
            fire_tick: -1000,
            linger_left: Fix::ZERO,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GrappleState {
    pub rope: Option<usize>,
    pub rope_number: u32,
    pub cooldown_set: bool,
    pub reel_frames: usize,
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
    /// RopeAttachment's own fixTrans.position: placed on attach and by each
    /// RepositionOnParent, and read by the next pull before it moves again.
    pub platform_anchor: Vec2,
    /// RopeAttachment.effectivePlatformPullStr switches on Boulder parents.
    pub platform_pull: Fix,
    /// RopeAttachment.topAttachment. Web/Grapple platform anchors use the
    /// top endpoint; PlatformTransform can transfer either player endpoint.
    pub platform_top: bool,
    /// A dynamically instantiated RopeAttachment joins Updater next pass.
    pub platform_attach_tick: u32,
    /// The platform was removed; the attachment lets go in its next UpdateSim
    /// and holds the end at `platform_anchor` until then.
    pub platform_gone: bool,
    /// Where the hook met a platform this pass. RopeHook.OnCollide takes the
    /// surface coordinate after DetPhysics synced the platform the contact
    /// passes pushed.
    pub hook_hit: Option<Vec2>,
    /// RopeHook.OnCollide takes the hit collider's centre, so a rope that
    /// meets a black hole hangs off the hole's centre.
    pub black_hole: Option<usize>,
    pub black_hole_reel_frames: usize,
    /// A body the hook stuck to through AttachToNonPlatform: the rope hangs
    /// off its collider centre and pulls on it.
    pub parent: Option<RopeParent>,
    pub segment_ids: [u32; MAX_SEGMENTS - 1],
    pub hook_active: bool,
    pub hook_remove_next: bool,
    pub hook_entity: Option<usize>,
    pub hook_position: Vec2,
    pub hook_velocity: Vec2,
    pub hook_end_player: usize,
    pub hook_time: Fix,
    pub rope_time: Fix,
    pub reel_out_time: Fix,
    pub reel_in: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RopeParent {
    /// A Rock-form player: it has no PlayerBody for the rope to join.
    Rock(usize),
    /// A thrown item by slot and id, so a reused slot reads as destroyed.
    Item(usize, u32),
    /// A Portal Gun shot by slot and birth tick.
    Shot(usize, u32),
}

/// Where a live parent's collider centre is, or None once it is gone.
fn parent_position(
    players: &[PlayerSlot],
    items: &[ThrownItem],
    shots: &[PortalShotState],
    parent: RopeParent,
) -> Option<Vec2> {
    match parent {
        RopeParent::Rock(player) => (players[player].active == ActiveAbility::Rock
            && players[player].base.alive)
            .then_some(players[player].base.position),
        RopeParent::Item(item, id) => {
            (items[item].active && items[item].id == id).then_some(items[item].position)
        }
        RopeParent::Shot(shot, born) => (shots[shot].active && shots[shot].born_tick == born)
            .then_some(shots[shot].position),
    }
}

/// Rope.isOffscreen, with every bound scaled by `scale` for isFarOffscreen.
fn offscreen(point: Vec2, water: Fix, scale: Fix) -> bool {
    point.x > Fix::from_f32(131.25) * scale
        || point.x < Fix::from_f32(-131.25) * scale
        || point.y > Fix::from_f32(72.5) * scale
        || point.y < water * scale
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
        platform_anchor: Vec2::ZERO,
        platform_pull: PLATFORM_PULL,
        platform_top: true,
        platform_attach_tick: 0,
        platform_gone: false,
        hook_hit: None,
        black_hole: None,
        black_hole_reel_frames: 0,
        parent: None,
        segment_ids: [0; MAX_SEGMENTS - 1],
        hook_active: false,
        hook_remove_next: false,
        hook_entity: None,
        hook_position: Vec2::ZERO,
        hook_velocity: Vec2::ZERO,
        hook_end_player: usize::MAX,
        hook_time: Fix::ZERO,
        rope_time: Fix::ZERO,
        reel_out_time: Fix::ZERO,
        reel_in: false,
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
    assert_eq!(source.boulder_pull, BOULDER_PULL);
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
        rope.platform_pull = PLATFORM_PULL;
        rope.platform_top = true;
        rope.platform_attach_tick = 0;
        rope.platform_gone = false;
        rope.hook_hit = None;
        rope.black_hole = None;
        rope.black_hole_reel_frames = 0;
        rope.parent = None;
        rope.hook_active = false;
        rope.hook_remove_next = false;
        rope.hook_entity = None;
        rope.hook_end_player = usize::MAX;
    }
}

impl RolloutWorld {
    fn allocate_rope(
        &mut self,
        owner: usize,
        anchor: Vec2,
        player: Vec2,
        separation: Fix,
        platform: Option<usize>,
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
        rope.platform = platform;
        rope.platform_local = platform_local;
        rope.platform_anchor = platform.map_or(Vec2::ZERO, |platform| {
            self.platforms[platform].surface.position(platform_local, Fix::ZERO)
        });
        rope.platform_gone = false;
        rope.hook_hit = None;
        rope.platform_pull = PLATFORM_PULL;
        rope.platform_top = true;
        rope.platform_attach_tick = self.tick;
        rope.black_hole = None;
        rope.black_hole_reel_frames = 0;
        rope.parent = None;
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
        rope.hook_active = false;
        rope.hook_remove_next = false;
        rope.hook_entity = None;
        rope.hook_position = Vec2::ZERO;
        rope.hook_velocity = Vec2::ZERO;
        rope.hook_end_player = usize::MAX;
        rope.hook_time = Fix::ZERO;
        rope.rope_time = Fix::ZERO;
        rope.reel_out_time = Fix::ZERO;
        rope.reel_in = false;
        for j in 0..SEGMENTS {
            let point = anchor
                + (player - anchor) * Fix::from_int(j as i64) / Fix::from_int(SEGMENTS as i64);
            body.segment[j] = point;
            body.previous[j] = point;
        }
        index
    }

    /// RopePlayerMass01 and ignoreGravityWhenRoped of the PlayerBody that
    /// carries the rope: 0.5 with gravity on the slime and most forms.
    fn rope_body_tuning(&self, player: usize) -> (Fix, bool) {
        match self.players[player].active {
            ActiveAbility::Beam => (Fix::from_raw(3_006_477_056), true),
            ActiveAbility::TimeStop => (Fix::from_raw(3_435_973_888), true),
            ActiveAbility::Revival => (Fix::from_raw(3_865_470_464), true),
            ActiveAbility::Platform => (Fix::ONE, true),
            _ => (Fix::from_f64(0.5), false),
        }
    }

    pub(super) fn attach_player_rope(&mut self, player: usize, rope: Option<usize>, top: bool) {
        if let Some(old) = self.players[player].rope {
            let (mass, ignore_gravity) = self.rope_body_tuning(player);
            let force = if ignore_gravity { Vec2::ZERO } else { -Vec2::UP * ROPE_GRAVITY };
            // PlayerBody.AttachRope releases the old rope at the end the call
            // names, not the end the body held: HookshotInstant and WebSwing
            // pass the bottom end even for a player hooked by the top.
            let old_rope = &mut self.ropes[old].physics;
            if top {
                old_rope.force_start = force;
            } else {
                old_rope.force_end = force;
            }
            if rope.is_none() {
                old_rope.detach(top);
            } else {
                old_rope.set_mass(mass, top, Fix::ONE);
            }
            if top {
                self.ropes[old].time_air_start = Fix::from_int(100);
            } else {
                self.ropes[old].time_air_end = Fix::from_int(100);
            }
        }
        let slot = &mut self.players[player];
        slot.rope_prev[slot.rope_body()] = (slot.base.velocity, slot.base.external_velocity);
        slot.rope = rope;
        slot.rope_top = top;
        if rope.is_some() {
            slot.rope_attach_tick = self.tick;
        }
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
            // The countdown waits while no web is held and a new cast does not
            // reset it, so a strand released just before a recast times out
            // the new web instead (WebSwing.cs:96-109).
            let state = self.players[player].web_swing[ability_slot];
            if state.linger_left > Fix::ZERO
                && let Some(rope) = state.rope
            {
                let left = state.linger_left - DT;
                self.players[player].web_swing[ability_slot].linger_left = left;
                if left <= Fix::ZERO {
                    self.disable_rope(rope);
                    self.players[player].web_swing[ability_slot].rope = None;
                }
            }
        }
    }

    pub(super) fn grapple_cooldowns_tick(&mut self, player: usize) {
        let cooldown = data::get().ropes.grapple_cooldown;
        for ability_slot in 0..3 {
            if self.players[player].abilities[ability_slot] != ABILITY_GRAPPLING_HOOK {
                continue;
            }
            let state = self.players[player].grapple[ability_slot];
            if state.cooldown_set {
                continue;
            }
            let valid = state.rope.is_some_and(|rope| {
                self.ropes[rope].physics.enabled
                    && self.ropes[rope].number == state.rope_number
                    && self.ropes[rope].owner == player
                    && self.players[player].rope == Some(rope)
            });
            if valid {
                self.players[player].cooldowns[ability_slot] = Fix::ZERO;
            } else {
                self.players[player].grapple[ability_slot].cooldown_set = true;
                self.players[player].cooldowns[ability_slot] = cooldown;
            }
        }
    }

    pub(super) fn use_grapple(&mut self, player: usize, ability_slot: usize, action: Action) {
        let source = &data::get().ropes;
        let state = self.players[player].grapple[ability_slot];
        let valid = state.rope.is_some_and(|rope| {
            self.ropes[rope].physics.enabled
                && self.ropes[rope].number == state.rope_number
                && self.ropes[rope].owner == player
                && self.players[player].rope == Some(rope)
        });
        self.players[player].timers[ability_slot] = Fix::ZERO;
        self.players[player].telemetry.ability_activations[ability_slot] += 1;
        if valid {
            let rope_index = state.rope.unwrap();
            let frames = state.reel_frames + 1;
            if frames < source.grapple_reel_frames {
                self.players[player].grapple[ability_slot].reel_frames = frames;
                return;
            }
            let count = self.ropes[rope_index].physics.count;
            let separation = self.ropes[rope_index].physics.separation
                - source.grapple_reel_speed / Fix::from_int(count as i64);
            self.ropes[rope_index].physics.separation =
                separation.max(source.grapple_reeled_separation);
            let mut remains = true;
            if self.ropes[rope_index].physics.separation < source.grapple_delete_separation {
                remains = self.ropes[rope_index].physics.reel(8, false);
            }
            if !remains {
                // HookshotInstant lands the slime on the end it does not hold.
                let endpoint = if self.players[player].rope_top {
                    self.ropes[rope_index].physics.count - 1
                } else {
                    0
                };
                let delta = self.ropes[rope_index].physics.segment[endpoint]
                    - self.players[player].base.position;
                if !self.players[player].base.grounded {
                    self.players[player].base.position = self.ropes[rope_index].physics.segment
                        [endpoint]
                        + delta.normalized_safe() * Fix::from_f64(0.1);
                }
                self.disable_rope(rope_index);
                self.players[player].grapple[ability_slot].rope = None;
            }
            self.players[player].grapple[ability_slot].reel_frames = 0;
            return;
        }
        self.fire_grapple(player, ability_slot, action);
    }

    fn fire_grapple(&mut self, player: usize, ability_slot: usize, action: Action) {
        let source = &data::get().ropes;
        // HookshotInstant.FireHook offsets the spawn by the raw aim vector and
        // only substitutes Vec2.right for the hook's *velocity*
        // (sim/Gameplay/HookshotInstant.cs:152-156). With no aim input the
        // oracle therefore spawns the hook and the rope exactly on the body and
        // still launches it rightwards.
        let aim = action.direction();
        let launch = if aim == Vec2::ZERO {
            Vec2::RIGHT
        } else {
            aim
        };
        let position = self.players[player].base.position;
        let player_query = Circle {
            center: position,
            radius: source.grapple_player_cast_radius,
        };
        let mut nearby_player = None;
        for candidate in 0..self.players.len() {
            if candidate != player
                && self.playable(candidate)
                && self.players[candidate].active != ActiveAbility::Rock
                && crate::physics::contact::circle_box(
                    player_query,
                    self.players[candidate].base.collider,
                )
                .is_some()
            {
                // CircleCastClosestPlayers retains the final matching box.
                nearby_player = Some(candidate);
            }
        }
        // Circles follow the boxes with the cast radius scaled by 1.5, so a
        // Rock ball in reach wins over any slime.
        let circle_query = Circle {
            center: position,
            radius: source.grapple_player_cast_radius * Fix::from_f64(1.5),
        };
        for candidate in 0..self.players.len() {
            if candidate != player
                && self.playable(candidate)
                && self.players[candidate].active == ActiveAbility::Rock
                && crate::physics::contact::circle_circle(
                    circle_query,
                    Circle {
                        center: self.players[candidate].base.position,
                        radius: self.players[candidate].rock.collider_radius,
                    },
                )
                .is_some()
            {
                nearby_player = Some(candidate);
            }
        }
        let rope_end = nearby_player
            .map(|target| self.players[target].base.position)
            .unwrap_or(position);
        let hook_position = rope_end + aim * source.grapple_spawn_offset;
        let hook_id = self.next_id();
        let hook_entity = self.entities.len();
        self.entities.push(CEntity {
            id: hook_id,
            kind: 0,
            layer: source.hook_layer,
            flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
            hierarchy: 100_000_000
                + self.next_spawn_hierarchy
                + 1 // Rope.Awake registers before the RopeHook prefab.
                + source.hook_collider_hierarchy,
            x: hook_position.x.raw(),
            y: hook_position.y.raw(),
            radius: source.hook_radius.raw(),
            inverse_mass: -Fix::ONE.raw(),
            ..CEntity::default()
        });
        // The Rope and RopeHook prefabs both count (HookshotInstant.cs:128-132).
        self.next_spawn_hierarchy += 1 + source.hook_updatable_count;
        let rope_index = self.allocate_rope(
            player,
            hook_position,
            rope_end,
            source.grapple_start_separation,
            None,
            Fix::ZERO,
        );
        {
            let rope = &mut self.ropes[rope_index];
            rope.physics.time_since_hook_arrived = -Fix::ONE;
            rope.physics.mass_start = source.non_platform_mass;
            rope.time_air_start = Fix::ONE;
            rope.hook_active = true;
            rope.hook_entity = Some(hook_entity);
            rope.hook_position = hook_position;
            rope.hook_velocity = launch * source.grapple_speed;
            rope.hook_end_player = nearby_player.unwrap_or(player);
        }
        // FireHook zeroes the shooter's self-imposed velocity on both branches.
        self.players[player].base.velocity = Vec2::ZERO;
        if let Some(target) = nearby_player {
            if self.players[target].active == ActiveAbility::Rock {
                // BounceBall.AttachRope lets the old rope go and leaves the
                // ball's velocity alone. UpdateRopeAttachment then drops the
                // new rope, whose number is not the one the ball entered with.
                if let Some(old) = self.players[target].rope.take() {
                    let rock = &data::get().rock;
                    let mass = if rock.rope_player_mass == Fix::ZERO {
                        Fix::from_f64(0.5)
                    } else {
                        rock.rope_player_mass
                    };
                    let old = &mut self.ropes[old];
                    old.time_air_end = Fix::from_int(100);
                    old.physics.force_end = -Vec2::UP * ROPE_GRAVITY;
                    old.physics.set_mass(mass, false, Fix::ONE);
                }
            } else {
                self.players[target].base.velocity = Vec2::ZERO;
                self.attach_player_rope(target, None, false);
                self.attach_player_rope(target, Some(rope_index), false);
            }
            self.players[player].cooldowns[ability_slot] = source.grapple_cooldown;
            self.players[player].grapple[ability_slot] = GrappleState {
                cooldown_set: true,
                ..GrappleState::default()
            };
        } else {
            self.attach_player_rope(player, None, false);
            self.attach_player_rope(player, Some(rope_index), false);
            self.players[player].grapple[ability_slot] = GrappleState {
                rope: Some(rope_index),
                rope_number: self.ropes[rope_index].number,
                cooldown_set: false,
                reel_frames: 0,
            };
        }
    }

    pub(super) fn grapple_ropes_update(&mut self) {
        let source = &data::get().ropes;
        for index in 0..self.ropes.len() {
            if !self.ropes[index].physics.enabled
                || self.ropes[index].born_tick == self.tick
                || self.ropes[index].hook_entity.is_none()
            {
                continue;
            }
            if self.ropes[index].hook_remove_next {
                if let Some(entity) = self.ropes[index].hook_entity.take() {
                    self.entities[entity].flags = E_REMOVED;
                }
                self.ropes[index].hook_active = false;
                self.ropes[index].hook_remove_next = false;
            }
            self.ropes[index].reel_out_time += DT;
            self.ropes[index].rope_time += DT;
            if self.ropes[index].reel_in {
                // Arrival leaves the current segment count fixed; player
                // input performs the later reel-in and deletion.
            } else if self.ropes[index].rope_time > source.rope_max_travel_time {
                self.disable_rope(index);
                continue;
            } else {
                let count = self.ropes[index].physics.count;
                let direction = (self.ropes[index].physics.segment[count - 1]
                    - self.ropes[index].physics.segment[0])
                    .normalized_safe();
                let perpendicular = direction.perp() * source.rope_sprinkler_speed * DT;
                let mut wave = 0usize;
                let mut positive = true;
                for segment in 0..count - 1 {
                    if wave >= source.rope_segment_wave_length {
                        positive = !positive;
                        wave = 0;
                    }
                    wave += 1;
                    if segment >= count - source.rope_segments_affected {
                        let signed = if positive {
                            perpendicular
                        } else {
                            -perpendicular
                        };
                        self.ropes[index].physics.segment[segment] +=
                            direction * source.rope_reel_out_segment_speed * DT
                                + signed * source.rope_sprinkler_speed * DT;
                    }
                }
                if self.ropes[index].reel_out_time > source.rope_reel_out_interval {
                    self.ropes[index].reel_out_time = Fix::ZERO;
                    let count = self.ropes[index].physics.count;
                    let spawn = self.ropes[index].physics.segment[count - 1]
                        + (self.ropes[index].physics.segment[count - 2]
                            - self.ropes[index].physics.segment[count - 1])
                            * Fix::from_f32(0.5);
                    if self.ropes[index].physics.spawn(spawn) {
                        let segment = self.ropes[index].physics.count - 2;
                        self.ropes[index].segment_ids[segment] = self.next_id();
                    } else {
                        self.ropes[index].reel_in = true;
                    }
                }
            }
            if self.ropes[index].hook_active {
                self.ropes[index].time_air_start = Fix::ZERO;
                let count = self.ropes[index].physics.count;
                let velocity = self.ropes[index].hook_velocity;
                let mut falloff = source.hook_speed_falloff;
                for offset in 2..count.min(source.hook_segments_affected) {
                    self.ropes[index].physics.segment[count - offset] += velocity * DT * falloff;
                    falloff *= source.hook_speed_falloff;
                }
                self.ropes[index].hook_time += DT;
                if self.ropes[index].hook_time > source.hook_fall_time {
                    self.disable_rope(index);
                    continue;
                }
                self.ropes[index].hook_position += velocity * DT;
                if let Some(entity) = self.ropes[index].hook_entity {
                    self.entities[entity].x = self.ropes[index].hook_position.x.raw();
                    self.entities[entity].y = self.ropes[index].hook_position.y.raw();
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
        let rope = self.allocate_rope(player, near, position, separation, Some(platform), local);
        // The Rope and RopeAttachment prefabs both count (WebSwing.cs:149,159).
        self.next_spawn_hierarchy += 2;
        let number = self.ropes[rope].number;
        self.attach_player_rope(player, None, false);
        self.attach_player_rope(player, Some(rope), false);
        if self.players[player].base.grounded {
            // UnGround raises OnNoLongerGrounded: State.fall.
            let slot = &mut self.players[player];
            slot.base.force_unground(&self.platforms);
            slot.base.rotation = Fix::ZERO;
            slot.base.up = Vec2::UP;
            slot.base.velocity += Vec2::UP * WEB_GROUNDED_POP;
            slot.obs_state = STATE_FALL;
            slot.state_ticks = 0;
        }
        self.players[player].web_swing[ability_slot] = WebSwingState {
            rope: Some(rope),
            rope_number: number,
            cooldown_set: false,
            fire_tick: self.tick as i32,
            linger_left: self.players[player].web_swing[ability_slot].linger_left,
        };
    }

    pub(super) fn rope_player_move(
        &mut self,
        player: usize,
        action: Action,
        input: Vec2,
        jump_pressed: bool,
    ) {
        // Same SlimeController jump as `Player::move_step`; the rope's own
        // release below keeps the raw button.
        let cast = self.players[player].base.instant_cast_this_pass;
        if jump_pressed && self.players[player].base.grounded && !cast {
            let strength = self.players[player].base.constants.jump_strength;
            // This is SlimeController.Jump, so it records wasUpsideDown just
            // as `Player::jump` does (SlimeController.cs:897-900).
            self.players[player]
                .base
                .record_upside_down_jump(&self.platforms);
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
            self.players[player].base.out_of_bounds = true;
            return;
        }
        // SlimeController.cs:1086-1089; the rope is live here, or the slime
        // would not be on this path.
        if !self.players[player].base.grounded {
            self.players[player].rope_swing_force =
                Vec2::new(input.x * data::get().ropes.extra_swing_force, Fix::ZERO);
        }
        self.rope_body_move(player, jump_pressed);
    }

    /// PlayerBody.UpdateSim for a body carrying a rope endpoint. Ability
    /// prefabs such as Dash own a PlayerBody too, so this phase remains live
    /// after the ordinary SlimeController GameObject has been disabled.
    pub(super) fn rope_body_move(&mut self, player: usize, jump_pressed: bool) {
        let Some(rope_index) = self.players[player].rope else {
            return;
        };
        if !self.ropes[rope_index].physics.enabled {
            // PlayerBody drops the disabled rope and steps like an unroped body.
            self.players[player].rope = None;
            let body = &mut self.players[player].base;
            body.position += (body.velocity + body.external_velocity) * DT;
            body.external_velocity *= Fix::from_f64(0.95);
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
                if top {
                    self.ropes[rope_index].physics.force_start = Vec2::ZERO;
                } else {
                    self.ropes[rope_index].physics.force_end = Vec2::ZERO;
                }
            } else {
                let time = if top {
                    self.ropes[rope_index].time_air_start += DT;
                    self.ropes[rope_index].time_air_start
                } else {
                    self.ropes[rope_index].time_air_end += DT;
                    self.ropes[rope_index].time_air_end
                };
                let (mass, ignore_gravity) = self.rope_body_tuning(player);
                self.ropes[rope_index].physics.set_mass(mass, top, Fix::ONE);
                let amount = if ignore_gravity {
                    Fix::ONE - mass
                } else {
                    (time * Fix::from_int(2)).clamp(Fix::ZERO, Fix::ONE)
                };
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
            if self.players[player].active != ActiveAbility::Dash {
                // The Dash prefab owns a distinct PlayerBody whose private
                // previous-velocity fields start at zero. Preserve the
                // inactive ordinary slime's fields for the exit handoff.
                let slot = &mut self.players[player];
                slot.rope_prev[slot.rope_body()] =
                    (slot.base.velocity, slot.base.external_velocity);
            }
        }
        self.players[player].base.external_velocity *= Fix::from_f64(0.95);
    }

    /// Dash.ExitAbility transfers the still-attached RopeBody back to the
    /// ordinary slime. The bottom endpoint is rebuilt as a straight strand so
    /// its stored Verlet velocity cannot inject the pre-dash swing on return.
    pub(super) fn dash_rope_exit(&mut self, player: usize, input: Vec2, leave_speed: Fix) {
        let Some(rope_index) = self.players[player].rope else {
            return;
        };
        let position = self.players[player].base.position;
        let top = self.players[player].rope_top;
        let rope = &mut self.ropes[rope_index].physics;
        let force = input * leave_speed * Fix::from_int(10);
        if top {
            rope.force_start = force;
            rope.segment[0] = position;
            return;
        }
        rope.force_end = force;
        let first = rope.segment[0];
        let step = (position - first) / Fix::from_int(rope.count as i64);
        for segment in 1..rope.count {
            rope.segment[segment] = first + step * Fix::from_int(segment as i64);
        }
        rope.previous[..rope.count].copy_from_slice(&rope.segment[..rope.count]);
    }

    /// RopeAttachment.Deattach: the freed endpoint takes an unheld end's
    /// mass, air time and force before RopeBody.Dettach.
    fn deattach_rope(&mut self, index: usize, top: bool) {
        let rope = &mut self.ropes[index];
        rope.physics.set_mass(data::get().ropes.non_platform_mass, top, Fix::ONE);
        if top {
            rope.time_air_start = Fix::from_int(100);
            rope.physics.force_start = Vec2::ZERO;
        } else {
            rope.time_air_end = Fix::from_int(100);
            rope.physics.force_end = Vec2::ZERO;
        }
        rope.physics.detach(top);
    }

    pub(super) fn ropes_update(&mut self) {
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        let mut moved_platform = false;
        let water = data::get().maps[self.map].scene_water_height();
        for index in 0..self.ropes.len() {
            if !self.ropes[index].physics.enabled || self.ropes[index].born_tick == self.tick {
                continue;
            }
            if self.ropes[index].physics.time_since_hook_arrived >= Fix::ZERO {
                self.ropes[index].physics.time_since_hook_arrived += DT * self.player_time_scale;
            }
            // Rope.UpdateSim drops a rope with both ends offscreen or either
            // end twice as far out (below the water counts as offscreen). The
            // hook's own RopeAttachment then destroys the hook with it.
            let physics = &self.ropes[index].physics;
            let (first, last) = (physics.segment[0], physics.segment[physics.count - 1]);
            if (offscreen(first, water, Fix::ONE) && offscreen(last, water, Fix::ONE))
                || offscreen(first, water, Fix::from_int(2))
                || offscreen(last, water, Fix::from_int(2))
            {
                self.disable_rope(index);
                continue;
            }
            if let Some(hole_index) = self.ropes[index].black_hole {
                if hole_index >= self.black_holes.len()
                    || !self.black_holes[hole_index].active
                    || self.blink_object_hidden(self.black_holes[hole_index].entity)
                    || self.black_holes[hole_index].mass <= Fix::ZERO
                {
                    self.deattach_rope(index, true);
                    self.ropes[index].black_hole = None;
                    continue;
                }
                let hole = self.black_holes[hole_index];
                let anchor = hole.position;
                if self.ropes[index].physics.time_since_hook_arrived > Fix::ZERO
                    && !self.ropes[index].physics.detached
                    && !stopped
                {
                    self.ropes[index].black_hole_reel_frames += 1;
                    let frame = self.ropes[index].black_hole_reel_frames;
                    if ((frame > 60 && frame.is_multiple_of(3))
                        || (frame <= 60 && frame.is_multiple_of(8))
                        || self.ropes[index].physics.count < 8)
                        && !self.ropes[index].physics.reel(3, false)
                    {
                        let far = self.ropes[index].physics.count - 1;
                        self.ropes[index].physics.segment[far] = hole.position;
                    }
                }
                self.ropes[index].physics.segment[0] = anchor;
                continue;
            }
            // RopeAttachment on an object runs in rope_parents_update.
            if self.ropes[index].parent.is_some() {
                continue;
            }
            if std::mem::take(&mut self.ropes[index].platform_gone) {
                let top = self.ropes[index].platform_top;
                self.deattach_rope(index, top);
                continue;
            }
            let Some(platform) = self.ropes[index].platform else {
                continue;
            };
            if self.ropes[index].platform_attach_tick == self.tick {
                continue;
            }
            if platform >= self.platforms.len()
                || self.entities[self.platform_entities[platform]].flags & E_ACTIVE == 0
            {
                let top = self.ropes[index].platform_top;
                self.deattach_rope(index, top);
                self.ropes[index].platform = None;
                continue;
            }
            let top = self.ropes[index].platform_top;
            let endpoint = if top {
                0
            } else {
                self.ropes[index].physics.count - 1
            };
            let neighbor = if top {
                1
            } else {
                self.ropes[index].physics.count - 2
            };
            let anchor = self.platforms[platform]
                .surface
                .position(self.ropes[index].platform_local, Fix::ZERO);
            if self.ropes[index].physics.time_since_hook_arrived > Fix::ZERO
                && !self.ropes[index].physics.detached
                && !stopped
            {
                let force = (self.ropes[index].physics.segment[endpoint]
                    - self.ropes[index].physics.segment[neighbor])
                    * self.ropes[index].platform_pull;
                let arm = (self.ropes[index].platform_anchor - self.platforms[platform].surface.center)
                    .normalized_safe();
                self.platform_dynamics[platform]
                    .body
                    .apply_impulse(force, arm);
                moved_platform = true;
            }
            if top {
                self.ropes[index].time_air_start = Fix::ZERO;
            } else {
                self.ropes[index].time_air_end = Fix::ZERO;
            }
            let scale = PLATFORM_HEAVINESS / self.platform_dynamics[platform].body.inverse_mass;
            self.ropes[index]
                .physics
                .set_mass(PLATFORM_MASS, top, scale);
            if top {
                self.ropes[index].physics.force_start = Vec2::ZERO;
            } else {
                self.ropes[index].physics.force_end = Vec2::ZERO;
            }
            self.ropes[index].physics.segment[endpoint] = anchor;
            self.ropes[index].platform_anchor = anchor;
            for segment in 0..self.ropes[index].physics.count.saturating_sub(2) {
                if (self.ropes[index].physics.segment[segment]
                    - self.ropes[index].physics.segment[segment + 1])
                    .length_squared()
                    > PLATFORM_BREAK_STRETCH
                {
                    self.deattach_rope(index, top);
                    self.ropes[index].platform = None;
                    break;
                }
            }
        }
        if moved_platform {
            self.platforms_changed();
        }
    }

    /// RopeAttachment.UpdateSim for a hook caught on an object. The
    /// attachment is created when the hook lands, so it updates after the
    /// object it holds: a Portal Gun shot that lands and goes this pass is
    /// already gone when the attachment looks.
    pub(super) fn rope_parents_update(&mut self) {
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        for index in 0..self.ropes.len() {
            if !self.ropes[index].physics.enabled || self.ropes[index].born_tick == self.tick {
                continue;
            }
            let Some(parent) = self.ropes[index].parent else {
                continue;
            };
            if parent_position(&self.players, &self.thrown_items, &self.portal_shots, parent).is_none() {
                self.deattach_rope(index, true);
                self.ropes[index].parent = None;
                continue;
            }
            let physics = &self.ropes[index].physics;
            if physics.time_since_hook_arrived > Fix::ZERO && !physics.detached && !stopped {
                // AddForceAtPosition at the parent's own centre: no torque.
                let force = (physics.segment[0] - physics.segment[1]) * ATTACHMENT_PULL;
                let body = match parent {
                    RopeParent::Rock(player) => &mut self.players[player].rock.body,
                    RopeParent::Item(item, _) => &mut self.thrown_items[item].body,
                    RopeParent::Shot(shot, _) => &mut self.portal_shots[shot].body,
                };
                body.apply_impulse(force, Vec2::ZERO);
            }
            for segment in 0..self.ropes[index].physics.count.saturating_sub(2) {
                if (self.ropes[index].physics.segment[segment]
                    - self.ropes[index].physics.segment[segment + 1])
                    .length_squared()
                    > PLATFORM_BREAK_STRETCH
                {
                    self.deattach_rope(index, true);
                    self.ropes[index].parent = None;
                    break;
                }
            }
        }
    }

    pub(super) fn ropes_physics(&mut self, physics_dt: Fix) {
        let space = data::get().maps[self.map].is_space();
        // DetPhysics.Simulate steps the ropes after integrating the rounded
        // bodies, before any contact corrects them.
        self.rope_rectangles.clear();
        if self.scratch.callback_tick == Some(self.tick)
            && self.scratch.callback_colliders.len() == self.platforms.len()
        {
            self.rope_rectangles.extend_from_slice(&self.scratch.callback_colliders);
        } else {
            self.rope_rectangles
                .extend(self.platforms.iter().map(|platform| platform.collider));
        }
        // SortAndSweep*_ropes sweeps the axis along which the rounded
        // rectangles alone spread widest about the mean of every active
        // collider, so a rope lists its rectangles by bounding-box minimum on
        // that axis, then by index.
        let mut mean = Vec2::ZERO;
        let mut active_count = 0usize;
        for (platform, rectangle) in self.platforms.iter().zip(&self.rope_rectangles) {
            if platform.active {
                mean += rectangle.inner.center;
                active_count += 1;
            }
        }
        for entity in self.entities.iter().filter(|entity| entity.kind != 2 && entity.flags & 3 == 3) {
            mean += Vec2::new(Fix::from_raw(entity.x), Fix::from_raw(entity.y));
            active_count += 1;
        }
        if active_count != 0 {
            mean = mean / Fix::from_int(active_count as i64);
        }
        let mut variance = Vec2::ZERO;
        for rectangle in &self.rope_rectangles {
            let delta = mean - rectangle.inner.center;
            variance += Vec2::new(delta.x * delta.x, delta.y * delta.y);
        }
        let sweep_x = variance.x > variance.y;
        for rope in &mut self.ropes {
            if !rope.physics.enabled {
                continue;
            }
            rope.physics.collision_count = 0;
            let bounds = rope.physics.bounds();
            for (platform, rectangle) in self.rope_rectangles.iter().enumerate() {
                if !self.platforms[platform].active {
                    continue;
                }
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
            let rectangles = &self.rope_rectangles;
            let minimum = |index: usize| {
                let rb = rectangles[index].bounds();
                if sweep_x { rb.min.x } else { rb.min.y }
            };
            rope.physics.colliding_rects[..rope.physics.collision_count].sort_unstable_by(|&a, &b| {
                minimum(a).partial_cmp(&minimum(b)).unwrap().then(a.cmp(&b))
            });
            rope.physics.step(
                RopeConfig {
                    dt: physics_dt,
                    gravity: ROPE_GRAVITY,
                    space,
                    iterations: ROPE_ITERATIONS,
                    stretchiness: PLATFORM_STRETCHINESS,
                },
                &self.rope_rectangles,
            );
        }
    }

    /// RopeHook.OnCollide's AttachToPlatform, on the platform pose DetPhysics
    /// synced after every contact pass.
    pub(super) fn grapple_hook_callbacks(&mut self) {
        for rope in &mut self.ropes {
            let Some(position) = rope.hook_hit.take() else {
                continue;
            };
            let Some(platform) = rope.platform else {
                continue;
            };
            let surface = &self.platforms[platform].surface;
            if let Ok(local) = surface.local_position(position, Fix::ZERO) {
                rope.platform_local = local;
                rope.platform_anchor = surface.position(local, Fix::ZERO);
            }
        }
    }

    pub(super) fn grapple_hook_contacts(&mut self) {
        let source = &data::get().ropes;
        for rope_index in 0..self.ropes.len() {
            if !self.ropes[rope_index].physics.enabled || !self.ropes[rope_index].hook_active {
                continue;
            }
            let position = self.ropes[rope_index].hook_position;
            let circle = Circle {
                center: position,
                radius: source.hook_radius,
            };
            let mut attached = None;
            for platform in 0..self.platforms.len() {
                if self.entities[self.platform_entities[platform]].flags & E_ACTIVE == 0
                    || crate::physics::contact::circle_rounded(
                        circle,
                        self.platforms[platform].collider,
                    )
                    .is_none()
                    || !self.platforms[platform].collider.overlaps(position)
                {
                    continue;
                }
                attached = Some(platform);
                break;
            }
            if let Some(platform) = attached {
                let local = self.platforms[platform]
                    .surface
                    .local_position(position, Fix::ZERO)
                    .expect("grapple hook centre overlaps platform");
                let anchor = self.platforms[platform].surface.position(local, Fix::ZERO);
                let rope = &mut self.ropes[rope_index];
                rope.physics.mass_start = Fix::ONE;
                rope.physics.force_start = Vec2::ZERO;
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.platform = Some(platform);
                rope.platform_local = local;
                rope.platform_anchor = anchor;
                rope.hook_hit = Some(position);
                rope.platform_pull = source.platform_pull;
                rope.reel_in = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // RopeHook.OnCollide destroys the hitbox and the component
                    // and leaves the GameObject alone (RopeHook.cs:182-189), so
                    // RlObservation's activeInHierarchy bit stays set while the
                    // enabled bit clears (RlObservation.cs:115-117).
                    self.entities[entity].flags =
                        (self.entities[entity].flags & (E_ACTIVE | E_TRIGGER)) | E_DESTROYED;
                }
                continue;
            }
            let mut attached_hole = None;
            for hole_index in 0..self.black_holes.len() {
                let hole = self.black_holes[hole_index];
                if !hole.active
                    || self.blink_object_hidden(hole.entity)
                    || crate::physics::contact::circle_circle(
                        circle,
                        Circle {
                            center: hole.position,
                            radius: hole.radius,
                        },
                    )
                    .is_none()
                {
                    continue;
                }
                attached_hole = Some(hole_index);
                break;
            }
            if let Some(hole_index) = attached_hole {
                let rope = &mut self.ropes[rope_index];
                rope.physics.mass_start = Fix::ONE;
                rope.physics.force_start = Vec2::ZERO;
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.physics
                    .set_mass(data::get().ropes.non_platform_mass, true, Fix::ONE);
                rope.time_air_start = Fix::ONE;
                rope.platform = None;
                rope.black_hole = Some(hole_index);
                rope.black_hole_reel_frames = 0;
                rope.reel_in = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // RopeHook.OnCollide destroys the hitbox and the component
                    // and leaves the GameObject alone (RopeHook.cs:182-189), so
                    // RlObservation's activeInHierarchy bit stays set while the
                    // enabled bit clears (RlObservation.cs:115-117).
                    self.entities[entity].flags =
                        (self.entities[entity].flags & (E_ACTIVE | E_TRIGGER)) | E_DESTROYED;
                }
                continue;
            }
            let owner = self.ropes[rope_index].owner;
            let end_player = self.ropes[rope_index].hook_end_player;
            let mut victim = None;
            for player in 0..self.players.len() {
                if (player == owner || player == end_player) || !self.playable(player) {
                    continue;
                }
                let slot = &self.players[player];
                let hit = if slot.active == ActiveAbility::Rock {
                    let rock = Circle {
                        center: slot.base.position,
                        radius: slot.rock.collider_radius,
                    };
                    crate::physics::contact::circle_circle(circle, rock).is_some()
                } else {
                    crate::physics::contact::circle_box(circle, slot.base.collider).is_some()
                };
                if hit {
                    victim = Some(player);
                    break;
                }
            }
            let mut parent = victim
                .filter(|&v| self.players[v].active == ActiveAbility::Rock)
                .map(RopeParent::Rock);
            if victim.is_none() {
                // Thrown items sit on the item layer, which the hook's Rope
                // layer collides with.
                parent = self.thrown_items.iter().enumerate().find_map(|(index, item)| {
                    let body = Circle {
                        center: item.position,
                        radius: item.collider_radius,
                    };
                    (item.active
                        && item.entity.is_some()
                        && crate::physics::contact::circle_circle(circle, body).is_some())
                    .then_some(RopeParent::Item(index, item.id))
                });
            }
            if victim.is_none() && parent.is_none() {
                // Portal Gun shots share the item layer.
                let radius = data::get().portal_gun.shot_radius;
                parent = self.portal_shots.iter().enumerate().find_map(|(index, shot)| {
                    let body = Circle {
                        center: shot.position,
                        radius,
                    };
                    (shot.active
                        && shot.registered
                        && crate::physics::contact::circle_circle(circle, body).is_some())
                    .then_some(RopeParent::Shot(index, shot.born_tick))
                });
            }
            if let Some(parent) = parent {
                let scale = match parent {
                    RopeParent::Rock(player) => self.players[player].rock.scale,
                    RopeParent::Item(item, _) => self.thrown_items[item].scale,
                    RopeParent::Shot(..) => Fix::ONE,
                };
                let rope = &mut self.ropes[rope_index];
                rope.physics.mass_start = Fix::ONE;
                rope.physics.force_start = Vec2::ZERO;
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.physics.set_mass(source.non_platform_mass, true, scale);
                rope.time_air_start = Fix::ONE;
                rope.platform = None;
                rope.parent = Some(parent);
                rope.reel_in = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // Hitbox-only teardown, as for a wall (RopeHook.cs:182-189).
                    self.entities[entity].flags =
                        (self.entities[entity].flags & (E_ACTIVE | E_TRIGGER)) | E_DESTROYED;
                }
                continue;
            }
            if let Some(victim) = victim {
                if let Some(old) = self.players[victim].rope
                    && old != rope_index
                {
                    self.ropes[old].physics.detach(true);
                    self.ropes[old].physics.detach(false);
                    self.players[victim].rope = None;
                }
                self.attach_player_rope(victim, Some(rope_index), true);
                let rope = &mut self.ropes[rope_index];
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.reel_in = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // Alone among the collide branches, a hit on a body with a
                    // PlayerBody destroys the whole hook GameObject and returns
                    // before the hitbox-only teardown (RopeHook.cs:144-150), so
                    // activeInHierarchy goes with it.
                    self.entities[entity].flags =
                        (self.entities[entity].flags & E_TRIGGER) | E_DESTROYED;
                }
            }
        }
    }

    pub(super) fn ropes_late(&mut self) {
        for player in 0..self.players.len() {
            let Some(rope_index) = self.players[player].rope else {
                continue;
            };
            // A Rock form's PlayerBody is inactive; BounceBall's own late
            // update checks the stretch and its UpdateSim sets the force.
            if self.players[player].active == ActiveAbility::Rock {
                continue;
            }
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
            // HookshotInstant attaches after this pass's PlayerBody updater;
            // a newly spawned grounded rope therefore keeps Initialize's
            // final point until the next pass. Before the hook arrives the end
            // keeps PlayerBody's integrated point (PlayerPhysics.cs:147).
            // RewindRecorder disables PlayerPhysics for the whole rewind, and
            // a destroyed attachedGround skips the whole grounded branch.
            if self.players[player].base.grounded
                && !self.players[player].rewind.rewinding
                && !self.ground_gone(player)
                && self.ropes[rope_index].physics.time_since_hook_arrived > Fix::ZERO
                && self.ropes[rope_index].born_tick != self.tick
                && self.players[player].rope_attach_tick != self.tick
            {
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
                    // PlayerPhysics.LateUpdateSim's UnGround nulls the rotation
                    // and raises the slime's OnNoLongerGrounded: State.fall.
                    let slot = &mut self.players[player];
                    slot.base.force_unground(&self.platforms);
                    slot.base.rotation = Fix::ZERO;
                    slot.base.up = Vec2::UP;
                    if slot.active == ActiveAbility::None {
                        slot.obs_state = STATE_FALL;
                        slot.state_ticks = 0;
                    }
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
                let (previous_imposed, previous_external) =
                    if self.players[player].active == ActiveAbility::Dash {
                        (Vec2::ZERO, Vec2::ZERO)
                    } else {
                        let slot = &self.players[player];
                        slot.rope_prev[slot.rope_body()]
                    };
                let mut imposed = self.players[player].base.velocity
                    - previous_imposed
                    - Vec2::UP * Fix::from_int(10);
                if self.rope_body_tuning(player).1 {
                    imposed.y = Fix::ZERO;
                }
                let mut external =
                    self.players[player].base.external_velocity - previous_external;
                // Only the slime's own PlayerBody carries the steer; an
                // ability form's body has none and leaves it pending.
                if self.players[player].active == ActiveAbility::None {
                    external += std::mem::take(&mut self.players[player].rope_swing_force);
                }
                if top {
                    self.ropes[rope_index].physics.force_start = imposed + external;
                } else {
                    self.ropes[rope_index].physics.force_end = imposed + external;
                }
            }
        }
        let (players, items, shots) = (&self.players, &self.thrown_items, &self.portal_shots);
        for rope in &mut self.ropes {
            // DetPhysics.VeryLateRopeUpdate clears this for all 128 pool
            // entries, including bodies disabled during the current pass.
            rope.disabled_this_frame = false;
            if !rope.physics.enabled {
                continue;
            }
            // A prefab registered from WebSwing.Cast is absent from the
            // current Updater list, so its RopeAttachment has no late pass
            // on the cast tick. PlayerBody is an authored component and does
            // pin the player endpoint immediately.
            if rope.born_tick != self.tick
                && let Some(platform) = rope.platform
                && rope.platform_attach_tick != self.tick
            {
                let endpoint = if rope.platform_top {
                    0
                } else {
                    rope.physics.count - 1
                };
                // RepositionOnParent skips a destroyed platform, and the end
                // stays where the attachment last stood (RopeAttachment.cs).
                if self.entities[self.platform_entities[platform]].flags & E_DESTROYED == 0 {
                    rope.platform_anchor =
                        self.platforms[platform].surface.position(rope.platform_local, Fix::ZERO);
                }
                rope.physics.segment[endpoint] = rope.platform_anchor;
            }
            if rope.platform_gone {
                let endpoint = if rope.platform_top { 0 } else { rope.physics.count - 1 };
                rope.physics.segment[endpoint] = rope.platform_anchor;
            }
            if rope.born_tick != self.tick
                && let Some(hole_index) = rope.black_hole
                && hole_index < self.black_holes.len()
                && self.black_holes[hole_index].active
            {
                let hole = self.black_holes[hole_index];
                rope.physics.segment[0] = hole.position;
            }
            // LateUpdateSim reads parentObject.position without asking whether
            // it went this pass: a shot that lands against a wall in Simulate
            // still holds the end where it landed.
            let position = match rope.parent {
                Some(RopeParent::Shot(shot, born)) if shots[shot].born_tick == born => {
                    Some(shots[shot].position)
                }
                parent => parent.and_then(|parent| parent_position(players, items, shots, parent)),
            };
            if let Some(position) = position {
                rope.physics.segment[0] = position;
            }
            if rope.hook_active {
                rope.physics.segment[0] = rope.hook_position;
            }
        }
    }

    pub(super) fn disable_rope(&mut self, rope: usize) {
        if self.ropes[rope].physics.enabled {
            self.ropes[rope].physics.enabled = false;
            self.ropes[rope].disabled_this_frame = true;
            if let Some(entity) = self.ropes[rope].hook_entity.take() {
                self.entities[entity].flags = E_REMOVED;
            }
            self.ropes[rope].hook_active = false;
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
                    flags: E_ACTIVE
                        | E_ENABLED
                        | if rope.physics.detached { E_DETACHED } else { 0 },
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
