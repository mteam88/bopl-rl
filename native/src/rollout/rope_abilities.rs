//! Shared fixed-capacity rope runtime for Web Swing and Grappling Hook.

use super::*;
use crate::physics::rope::{Rope, RopeConfig};

/// RopeHook.prefab's RopeAttachment is its updatable numbered 1, ahead of
/// the DPhysicsCircle (2) and the RopeHook (3).
const HOOK_ATTACHMENT_LOCAL: u32 = 1;

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

/// One bb_rope_rr entry: a rounded rectangle's or a rope's bounding box and
/// its id, a platform or a rope index.
#[derive(Clone, Copy)]
pub(super) struct RopeSweepBox {
    min: Vec2,
    max: Vec2,
    rope: bool,
    id: usize,
}

const PLATFORM_MASS: Fix = Fix::from_raw(3_865_470_464);
const PLATFORM_HEAVINESS: Fix = Fix::from_raw(42_949_672);
const PLATFORM_PULL: Fix = Fix::from_raw(-42_949_672_960);
const BOULDER_PULL: Fix = Fix::from_raw(-21_474_836_480);
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
    /// Updater hierarchy of the Rope prefab, whose RopeAttachment follows it:
    /// Mjolnir hammers interleave with its UpdateSim by this number.
    pub hierarchy: u32,
    pub appended: u32,
    /// The RopeHook prefab's registered base, for a grappling hook's rope.
    pub hook_hierarchy: u32,
    pub disabled_this_frame: bool,
    pub time_air_start: Fix,
    pub time_air_end: Fix,
    /// The RopeAttachments holding its ends, `[0]` the top (topAttachment)
    /// and `[1]` the bottom: the hook prefab's own, or the one WebSwing
    /// instantiates, on the top end, and one PlatformTransform.CreatePlatform
    /// instantiates on the end its PlayerBody held (PlatformTransform.cs:171-183).
    pub attach: [RopeAttachment; 2],
    /// Where the hook met a platform this pass. RopeHook.OnCollide takes the
    /// surface coordinate after DetPhysics synced the platform the contact
    /// passes pushed.
    pub hook_hit: Option<Vec2>,
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
    /// Rope.hookArrived: the hook met something it collides with.
    pub hook_arrived: bool,
}

/// One RopeAttachment component and the end of its rope it holds
/// (RopeAttachment.cs).
#[derive(Clone, Copy)]
pub(crate) struct RopeAttachment {
    pub platform: Option<usize>,
    /// LocalPlatformPosition on `platform`.
    pub local: Fix,
    /// unscaledParentOffset: where on a non-platform parent the end sits, in
    /// the parent's unrotated, unscaled frame. RopeHook passes the collider
    /// centre, so it is zero except on LethalTerrain, which keeps the hook's
    /// own position.
    pub offset: Vec2,
    /// fixTrans.position: placed on attach and by each RepositionOnParent,
    /// or on an object parent's position, and read by the next pull before
    /// it moves again.
    pub anchor: Vec2,
    /// effectivePlatformPullStr switches on Boulder parents.
    pub pull: Fix,
    /// A dynamically instantiated RopeAttachment joins Updater next pass.
    pub attach_tick: u32,
    /// The platform was removed; the attachment lets go in its next UpdateSim
    /// and holds the end at `anchor` until then.
    pub gone: bool,
    /// The pass and updater hierarchy of an UpdateSim that DestroyFix'd the
    /// platform; an older RopeAttachment still saw it alive that pass.
    pub destroyed_at: (u32, u32),
    /// RopeHook.OnCollide takes the hit collider's centre, so a rope that
    /// meets a black hole hangs off the hole's centre. The hole by slot and
    /// hierarchy, so a later hole in a reused slot reads as this one destroyed.
    pub black_hole: Option<(usize, u32)>,
    pub reel_frames: usize,
    /// A body the hook stuck to through AttachToNonPlatform: the rope hangs
    /// off its collider centre and pulls on it.
    pub parent: Option<RopeParent>,
    /// The Updater place of an instantiated RopeAttachment prefab, appended
    /// and numbered; the hook's own and WebSwing's take their turn with the
    /// rope.
    pub turn: Option<(u32, u32)>,
}

impl RopeAttachment {
    pub(super) const EMPTY: Self = Self {
        platform: None,
        local: Fix::ZERO,
        offset: Vec2::ZERO,
        anchor: Vec2::ZERO,
        pull: PLATFORM_PULL,
        attach_tick: 0,
        gone: false,
        destroyed_at: (0, 0),
        black_hole: None,
        reel_frames: 0,
        parent: None,
        turn: None,
    };
}

/// What a RopeHook's circle met in one DetPhysics pair.
#[derive(Clone, Copy)]
pub(super) enum HookTarget {
    Platform(usize),
    BlackHole(usize),
    Item(usize),
    Shot(usize),
    Arrow(usize),
    Node(usize),
    Pickup,
    Player(usize),
    Clone(usize),
    Spike(usize),
    Engine(usize),
}

/// A pooled parent is named by its slot and its hierarchy, so a later
/// object in a reused slot reads as the parent destroyed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RopeParent {
    /// A Rock-form player: it has no PlayerBody for the rope to join.
    Rock(usize),
    Item(usize, u32),
    Shot(usize, u32),
    /// A SpikeAttack: its box is on LethalTerrain.
    Spike(usize, u32),
    /// A placed Tesla spark node: its circle is on NonLethalTerrain.
    Node(usize, u32),
    /// A Bow arrow: its circle is on Projectile.
    Arrow(usize, u32),
    /// A RocketEngine: its box is on NonLethalTerrain.
    Engine(usize, u32),
}

/// A rope parent's FixTransform: the frame RopeHook measures the end's
/// offset in and RopeAttachment places the end back from (RopeHook.cs:154-157,
/// RopeAttachment.cs:141, 174).
#[derive(Clone, Copy)]
struct ParentPose {
    position: Vec2,
    rotation: Fix,
    right: Vec2,
    scale: Fix,
}

impl ParentPose {
    fn turned(position: Vec2, rotation: Fix, scale: Fix) -> Self {
        Self { position, rotation, right: Vec2::angle(rotation), scale }
    }

    /// `ComplexMul(point - position, new Vec2(-rotation)) / Scale`.
    fn offset(self, point: Vec2) -> Vec2 {
        (point - self.position).complex_mul(Vec2::angle(-self.rotation)) / self.scale
    }

    /// `position + up * offset.y * Scale + right * offset.x * Scale`.
    fn place(self, offset: Vec2) -> Vec2 {
        self.position + self.right.perp() * offset.y * self.scale + self.right * offset.x * self.scale
    }
}

type Attachables<'a> = (&'a [SpawnedSpike], &'a [SpawnedTeslaNode], &'a [Arrow], &'a [SpawnedEngine]);

/// The pose of `parent`'s FixTransform, which a destroyed one keeps.
fn parent_pose(
    players: &[PlayerSlot],
    items: &[ThrownItem],
    shots: &[PortalShotState],
    (spikes, nodes, arrows, engines): Attachables,
    parent: RopeParent,
) -> ParentPose {
    match parent {
        RopeParent::Rock(player) => {
            let slot = &players[player];
            ParentPose {
                position: slot.base.position,
                rotation: slot.base.rotation(),
                right: slot.base.right(),
                scale: slot.rock_circle().scale,
            }
        }
        RopeParent::Item(i, _) => ParentPose::turned(items[i].position, items[i].body.orientation, items[i].scale),
        RopeParent::Shot(i, _) => ParentPose::turned(shots[i].position, shots[i].body.orientation, shots[i].scale),
        RopeParent::Spike(i, _) => {
            ParentPose::turned(spikes[i].transform_position, spikes[i].transform_rotation, spikes[i].scale)
        }
        RopeParent::Node(i, _) => {
            ParentPose::turned(nodes[i].transform_position, nodes[i].transform_rotation, nodes[i].scale)
        }
        RopeParent::Arrow(i, _) => ParentPose::turned(arrows[i].position, arrows[i].rotation, arrows[i].scale),
        RopeParent::Engine(i, _) => {
            ParentPose::turned(engines[i].transform_position, engines[i].transform_rotation, engines[i].scale)
        }
    }
}

/// Where a live parent places the rope end, or None once it is gone.
fn parent_position(
    players: &[PlayerSlot],
    items: &[ThrownItem],
    shots: &[PortalShotState],
    attachables: Attachables,
    parent: RopeParent,
    offset: Vec2,
) -> Option<Vec2> {
    let (spikes, nodes, arrows, engines) = attachables;
    let live = match parent {
        RopeParent::Rock(player) => players[player].active == ActiveAbility::Rock && players[player].base.alive,
        RopeParent::Item(item, base) => items[item].active && items[item].hierarchy_base == base,
        RopeParent::Shot(shot, base) => shots[shot].active && shots[shot].hierarchy_base == base,
        RopeParent::Spike(index, base) => spikes[index].active && spikes[index].hierarchy_base == base,
        RopeParent::Node(index, base) => nodes[index].active && nodes[index].hierarchy_base == base,
        RopeParent::Arrow(index, base) => arrows[index].active && arrows[index].hierarchy_base == base,
        // MachoThrow2 hides a carried engine's GameObject.
        RopeParent::Engine(index, base) => {
            engines[index].active && !engines[index].macho_carried && engines[index].hierarchy_base == base
        }
    };
    live.then(|| parent_pose(players, items, shots, attachables, parent).place(offset))
}

/// Rope.isOffscreen, with every bound scaled by `scale` for isFarOffscreen.
fn offscreen(point: Vec2, water: Fix, scale: Fix) -> bool {
    point.x > Fix::from_f32(131.25) * scale
        || point.x < Fix::from_f32(-131.25) * scale
        || point.y > Fix::from_f32(72.5) * scale
        || point.y < water * scale
}

/// An abilityless clone's PlayerBody hold on a rope: `ropeBody` and
/// `topAttachment`, the velocities UpdateSim last left for LateUpdateSim to
/// measure against, and SlimeController's pending `extraSwingForce`.
#[derive(Clone, Copy, Default)]
pub struct RopeGrip {
    pub rope: Option<usize>,
    pub top: bool,
    pub attach_tick: u32,
    pub previous: (Vec2, Vec2),
    pub swing_force: Vec2,
}

/// The mass and gravity switch PlayerBody gives the rope end of a slime body
/// (RopePlayerMass01 0.5, ignoreGravityWhenRoped off).
const SLIME_ROPE_TUNING: (Fix, bool) = (Fix::from_raw(2_147_483_648), false);

/// PlayerBody.AttachRope's hand-off of the rope it held (PlayerBody.cs:
/// 253-266): the end falls under gravity unless the body ignores it, and
/// the rope lets go of that end unless another rope replaces it.
fn release_rope_end(rope: &mut RolloutRope, top: bool, (mass, ignore_gravity): (Fix, bool), replaced: bool) {
    let force = if ignore_gravity { Vec2::ZERO } else { -Vec2::UP * ROPE_GRAVITY };
    if top {
        rope.physics.force_start = force;
    } else {
        rope.physics.force_end = force;
    }
    if replaced {
        rope.physics.set_mass(mass, top, Fix::ONE);
    } else {
        rope.physics.detach(top);
    }
    if top {
        rope.time_air_start = Fix::from_int(100);
    } else {
        rope.time_air_end = Fix::from_int(100);
    }
}

/// What a rope ability's CircleCastClosestPlayers found.
#[derive(Clone, Copy)]
enum RopeTarget {
    Player(usize),
    Clone(usize),
}

/// PlayerBody.AttachRope for an abilityless clone's body.
pub(super) fn attach_grip(
    body: &Player,
    grip: &mut RopeGrip,
    ropes: &mut [RolloutRope],
    rope: Option<usize>,
    top: bool,
    tick: u32,
) {
    if let Some(old) = grip.rope {
        release_rope_end(&mut ropes[old], top, SLIME_ROPE_TUNING, rope.is_some());
    }
    grip.previous = (body.velocity, body.external_velocity);
    grip.rope = rope;
    grip.top = top;
    if rope.is_some() {
        grip.attach_tick = tick;
    }
}

/// PlayerBody.UpdateSim's hold on its rope end once the body has moved
/// (PlayerBody.cs:187-240). True when the rope was detached, and the body
/// lets go of it.
fn hold_rope_end(
    body: &mut Player,
    rope: &mut RolloutRope,
    top: bool,
    (mass, ignore_gravity): (Fix, bool),
) -> bool {
    let endpoint = if top { 0 } else { rope.physics.count - 1 };
    if rope.physics.detached {
        rope.physics.segment[endpoint] = body.position;
        return true;
    }
    if rope.physics.time_since_hook_arrived <= Fix::ZERO {
        rope.physics.segment[endpoint] = body.position;
        if top {
            rope.time_air_start = Fix::ZERO;
        } else {
            rope.time_air_end = Fix::ZERO;
        }
        rope.physics.set_mass(Fix::ONE, top, Fix::ONE);
        if top {
            rope.physics.force_start = Vec2::ZERO;
        } else {
            rope.physics.force_end = Vec2::ZERO;
        }
    } else if body.grounded {
        let mass = Fix::ONE
            + (Fix::from_f64(0.9) - Fix::ONE)
                * rope.physics.time_since_hook_arrived.clamp(Fix::ZERO, Fix::ONE);
        if top {
            rope.time_air_start = Fix::ZERO;
        } else {
            rope.time_air_end = Fix::ZERO;
        }
        rope.physics.set_mass(mass, top, Fix::ONE);
        if top {
            rope.physics.force_start = Vec2::ZERO;
        } else {
            rope.physics.force_end = Vec2::ZERO;
        }
    } else {
        let time = if top {
            rope.time_air_start += DT;
            rope.time_air_start
        } else {
            rope.time_air_end += DT;
            rope.time_air_end
        };
        rope.physics.set_mass(mass, top, Fix::ONE);
        let amount = if ignore_gravity {
            Fix::ONE - mass
        } else {
            (time * Fix::from_int(2)).clamp(Fix::ZERO, Fix::ONE)
        };
        let offset = rope.physics.segment[endpoint] - body.position;
        let length = offset.length();
        if length != Fix::ZERO {
            let correction = rope.physics.separation - length;
            let direction = offset / length;
            rope.physics.segment[endpoint] += direction * correction * (Fix::ONE - amount);
            body.position -= direction * correction * amount;
        }
    }
    false
}

/// PlayerPhysics.LateUpdateSim's grounded rope pull: a slack end drags the
/// body along, and the end is pinned to it. A pull that throws the body off
/// its ground UnGrounds it with a null rotation (PlayerPhysics.cs:156-159).
fn pin_grounded_rope_end(
    mut body: UngroundBody<'_>,
    rope: &mut RolloutRope,
    top: bool,
    platforms: &[Platform],
    tick: u32,
) {
    let endpoint = if top { 0 } else { rope.physics.count - 1 };
    let base = body.base();
    let delta = rope.physics.segment[endpoint] - base.position;
    if delta.length_squared() >= Fix::from_raw(300_647_711) {
        base.external_velocity += delta * GROUNDED_ROPE_STRENGTH;
    }
    // Gated on a live ground: nothing destroyed to fall back on.
    if !base.constants.cannot_be_ungrounded_by_ropes
        && base.platform(platforms).is_some_and(|platform| {
            platform.surface.normal(base.position).dot(base.external_velocity) > Fix::from_int(30)
        })
    {
        body.unground(platforms, tick, true);
    }
    rope.physics.segment[endpoint] = body.base().position;
}

/// PlayerBody.LateUpdateSim: a rope stretched past maxStretchLength snaps
/// off the body (PlayerBody.cs:130-139).
fn overstretched(rope: &RolloutRope) -> bool {
    (0..rope.physics.count.saturating_sub(2)).any(|segment| {
        (rope.physics.segment[segment] - rope.physics.segment[segment + 1]).length_squared()
            > MAX_STRETCH
    })
}

/// The force PlayerBody.LateUpdateSim sets on its end: the pass's change in
/// both velocities, with the player's gravity unless the body ignores it
/// (PlayerBody.cs:141-152).
fn rope_end_force(body: &Player, previous: (Vec2, Vec2), ignore_gravity: bool, swing: Vec2) -> Vec2 {
    let mut imposed = body.velocity - previous.0 - Vec2::UP * Fix::from_int(10);
    if ignore_gravity {
        imposed.y = Fix::ZERO;
    }
    imposed + (body.external_velocity - previous.1 + swing)
}

/// PlayerBody.UpdateSim for an abilityless clone gripping a rope
/// (PlayerBody.cs:159-247). `owner_jumps`: the clone's Player owns the rope
/// and pressed jump this pass.
pub(super) fn carry_rope(
    body: &mut Player,
    grip: &mut RopeGrip,
    ropes: &mut [RolloutRope],
    owner_jumps: bool,
) {
    if let Some(index) = grip.rope
        && ropes[index].physics.enabled
        && owner_jumps
        && !body.grounded
        && !body.jumped_this_frame
    {
        let rope = &ropes[index].physics;
        let endpoint = if grip.top { 0 } else { rope.count - 1 };
        body.velocity = (rope.segment[endpoint] - rope.previous[endpoint]) / DT;
        attach_grip(body, grip, ropes, None, grip.top, 0);
    }
    if grip.rope.is_some_and(|index| !ropes[index].physics.enabled) {
        grip.rope = None;
    }
    body.position += (body.velocity + body.external_velocity) * DT;
    if let Some(index) = grip.rope {
        if hold_rope_end(body, &mut ropes[index], grip.top, SLIME_ROPE_TUNING) {
            attach_grip(body, grip, ropes, None, grip.top, 0);
        } else {
            grip.previous = (body.velocity, body.external_velocity);
        }
    }
    body.external_velocity *= Fix::from_f64(0.95);
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
        hierarchy: 0,
        appended: 0,
        hook_hierarchy: 0,
        disabled_this_frame: false,
        time_air_start: Fix::ZERO,
        time_air_end: Fix::ZERO,
        attach: [RopeAttachment::EMPTY; 2],
        hook_hit: None,
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
        hook_arrived: false,
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
    assert_eq!(source.non_platform_mass, Fix::from_f64(0.5));
    assert_eq!(source.web_max_length, WEB_MAX_LENGTH);
    assert_eq!(source.web_shorten, WEB_SHORTEN);
    assert_eq!(source.web_cooldown, WEB_COOLDOWN);
    assert_eq!(source.web_miss_cooldown, WEB_MISS_COOLDOWN);
    assert_eq!(source.web_release_boost, WEB_RELEASE_BOOST);
    assert_eq!(source.web_grounded_pop, WEB_GROUNDED_POP);
    Vec::with_capacity(ROPE_POOL)
}

/// RopeBodyList's 128 bodies, held up to the last one AddRope has handed
/// out: the bodies past it are all still fresh, so walks over the pool
/// skip them alike.
pub(super) fn reset_rope_pool(ropes: &mut Vec<RolloutRope>) {
    ropes.clear();
}

impl RolloutWorld {
    /// RopeAttachment.AttachToPlatform pulls a "boulder"-tagged platform with
    /// boulderPullStr (RopeAttachment.cs:185-192).
    fn rope_platform_pull(&self, platform: usize) -> Fix {
        if self
            .macho_boulders
            .iter()
            .any(|boulder| boulder.active && boulder.platform_index == platform)
        {
            BOULDER_PULL
        } else {
            PLATFORM_PULL
        }
    }

    fn allocate_rope(
        &mut self,
        owner: usize,
        anchor: Vec2,
        player: Vec2,
        separation: Fix,
        platform: Option<usize>,
        platform_local: Fix,
        hierarchy: u32,
    ) -> usize {
        // RopeBodyList.AddRope takes the first body neither enabled nor
        // disabled this frame.
        let index = match self.ropes.iter().position(|rope| !rope.physics.enabled && !rope.disabled_this_frame) {
            Some(index) => index,
            None => {
                assert!(self.ropes.len() < ROPE_POOL, "RopeBody pool exhausted");
                self.ropes.push(empty_rope());
                self.ropes.len() - 1
            }
        };
        self.next_rope_number += 1;
        let mut segment_ids = [0; MAX_SEGMENTS - 1];
        for id in segment_ids.iter_mut().take(SEGMENTS - 1) {
            *id = self.next_id();
        }
        let platform_pull =
            platform.map_or(PLATFORM_PULL, |platform| self.rope_platform_pull(platform));
        let rope = &mut self.ropes[index];
        rope.owner = owner;
        rope.number = self.next_rope_number;
        rope.hierarchy = hierarchy;
        rope.appended = self.tick + 1;
        rope.hook_hierarchy = 0;
        rope.disabled_this_frame = false;
        rope.time_air_start = Fix::ZERO;
        rope.time_air_end = if self.players[owner].base.grounded {
            Fix::ZERO
        } else {
            Fix::from_int(100)
        };
        rope.attach[0].platform = platform;
        rope.attach[0].local = platform_local;
        rope.attach[0].anchor = platform.map_or(Vec2::ZERO, |platform| {
            self.platforms[platform].surface.position(platform_local, Fix::ZERO)
        });
        rope.attach[0].gone = false;
        rope.hook_hit = None;
        rope.attach[0].pull = platform_pull;
        rope.attach[0].attach_tick = self.tick;
        rope.attach[0].black_hole = None;
        rope.attach[0].reel_frames = 0;
        rope.attach[0].parent = None;
        rope.attach[0].turn = None;
        rope.attach[1] = RopeAttachment::EMPTY;
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
        rope.hook_arrived = false;
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
            // PlayerBody.AttachRope releases the old rope at the end the call
            // names, not the end the body held: HookshotInstant and WebSwing
            // pass the bottom end even for a player hooked by the top.
            let tuning = self.rope_body_tuning(player);
            release_rope_end(&mut self.ropes[old], top, tuning, rope.is_some());
        }
        let slot = &mut self.players[player];
        slot.rope_prev[slot.rope_body()] = (slot.base.velocity, slot.base.external_velocity);
        slot.rope = rope;
        slot.rope_top = top;
        if rope.is_some() {
            slot.rope_attach_tick = self.tick;
        }
    }

    /// Each rope ability's ready-indicator CircleCastClosestPlayers ahead of
    /// its UpdateCooldowns (SlimeController.cs:737-743).
    pub(super) fn rope_ready_casts(&mut self, player: usize) {
        if self.players[player]
            .abilities
            .iter()
            .any(|&ability| ability == ABILITY_WEB_SWING || ability == ABILITY_GRAPPLING_HOOK)
        {
            self.circle_cast_closest_players(player, data::get().ropes.grapple_player_cast_radius);
        }
    }

    /// DetPhysics.CircleCastClosestPlayers (DetPhysics.cs:1851-1880): purges
    /// destroyed colliders, then keeps the last active Player-layer box but
    /// the caster's the circle touches, then the last such circle within 1.5
    /// times the radius.
    fn circle_cast_closest_players(&mut self, player: usize, radius: Fix) -> Option<RopeTarget> {
        self.remove_destroyed_colliders();
        let position = self.players[player].base.position;
        let player_query = Circle { center: position, radius };
        let mut nearby_player = None;
        // A slime in its spawn animation already has its hurtbox enabled. The
        // cast reads each slime's current collider where DetPhysics holds it:
        // an ability form's own box or circle, not the slime body's.
        for candidate in 0..self.players.len() {
            if candidate != player
                && self.hittable(candidate)
                && let Some(DetCollider::Box(collider)) = self.hurtbox_collider(candidate)
                && crate::physics::contact::circle_box(player_query, collider).is_some()
            {
                // CircleCastClosestPlayers retains the final matching box.
                nearby_player = Some(RopeTarget::Player(candidate));
            }
        }
        // The abilityless clones' boxes registered after every slime's, and
        // the cast skips only the caster's own instance: the caster's own
        // clones are in reach.
        for candidate in self.player_layer_clones() {
            if crate::physics::contact::circle_box(
                player_query,
                self.abilityless_clones[candidate].base.synced,
            )
            .is_some()
            {
                nearby_player = Some(RopeTarget::Clone(candidate));
            }
        }
        // Circles follow the boxes with the cast radius scaled by 1.5, so a
        // circle form in reach (a Rock ball, a spell, a platform transform)
        // wins over any box.
        let circle_query = Circle {
            center: position,
            radius: radius * Fix::from_f64(1.5),
        };
        for candidate in 0..self.players.len() {
            if candidate != player
                && self.hittable(candidate)
                && let Some(DetCollider::Circle(collider)) = self.hurtbox_collider(candidate)
                && crate::physics::contact::circle_circle(circle_query, collider).is_some()
            {
                nearby_player = Some(RopeTarget::Player(candidate));
            }
        }
        nearby_player
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
        let nearby_player = self.circle_cast_closest_players(player, source.grapple_player_cast_radius);
        let rope_end = match nearby_player {
            Some(RopeTarget::Player(target)) => self.players[target].base.position,
            Some(RopeTarget::Clone(clone)) => self.abilityless_clones[clone].base.position,
            None => position,
        };
        let hook_position = rope_end + aim * source.grapple_spawn_offset;
        // The Rope and RopeHook prefabs both count (HookshotInstant.cs:128-132).
        let rope_hierarchy = self.register_prefab(data::get().prefab("Rope"));
        let hook_hierarchy = self.register_prefab(data::get().prefab("RopeHook"));
        let hook_id = self.next_id();
        let hook_entity = self.entities.len();
        self.push_entity(CEntity {
            id: hook_id,
            kind: 0,
            layer: source.hook_layer,
            flags: E_ACTIVE | E_ENABLED | E_TRIGGER,
            hierarchy: 100_000_000 + hook_hierarchy + source.hook_collider_hierarchy,
            x: hook_position.x.raw(),
            y: hook_position.y.raw(),
            radius: source.hook_radius.raw(),
            inverse_mass: -Fix::ONE.raw(),
            ..CEntity::default()
        }, Vec2::ZERO);
        let rope_index = self.allocate_rope(
            player,
            hook_position,
            rope_end,
            source.grapple_start_separation,
            None,
            Fix::ZERO,
            rope_hierarchy,
        );
        {
            let rope = &mut self.ropes[rope_index];
            rope.hook_hierarchy = hook_hierarchy;
            rope.physics.time_since_hook_arrived = -Fix::ONE;
            rope.physics.mass_start = source.non_platform_mass;
            rope.time_air_start = Fix::ONE;
            rope.hook_active = true;
            rope.hook_entity = Some(hook_entity);
            rope.hook_position = hook_position;
            rope.hook_velocity = launch * source.grapple_speed;
            // RopeHook ignores bodies by player id, so the end keeps the
            // target's player, whichever of its bodies the cast found.
            rope.hook_end_player = match nearby_player {
                Some(RopeTarget::Player(target)) => self.players[target].owner,
                Some(RopeTarget::Clone(clone)) => self.abilityless_clones[clone].owner,
                None => self.players[player].owner,
            };
        }
        // FireHook zeroes the shooter's self-imposed velocity on both branches.
        self.players[player].base.velocity = Vec2::ZERO;
        if let Some(target) = nearby_player {
            if let RopeTarget::Clone(clone) = target {
                let tick = self.tick;
                let clone = &mut self.abilityless_clones[clone];
                clone.base.velocity = Vec2::ZERO;
                attach_grip(&clone.base, &mut clone.grip, &mut self.ropes, None, false, tick);
                attach_grip(&clone.base, &mut clone.grip, &mut self.ropes, Some(rope_index), false, tick);
            } else if let RopeTarget::Player(target) = target
                && self.players[target].active == ActiveAbility::Rock
            {
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
            } else if let RopeTarget::Player(target) = target {
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

    /// The Updater key of rope `index`'s Rope.
    pub(super) fn rope_key(&self, index: usize) -> UpdaterKey {
        self.updater_key(self.ropes[index].appended, 100_000_000 + self.ropes[index].hierarchy)
    }

    /// Rope `index`'s turn in Updater's walk: Rope.UpdateSim, with the
    /// hook's RopeAttachment and RopeHook that follow it on a flying hook,
    /// and a platform or Black Hole RopeAttachment. Everything listed before
    /// it has taken its turn: a wall QuantumTunnel (ShootQuantum reuses a
    /// wall's tunnel) has hidden or restored its wall when the RopeAttachment
    /// asks activeInHierarchy, a Throw boulder's ShakablePlatform has moved
    /// the body it reads, an abilityless clone's PlayerBody has moved the
    /// rope end it holds, a ScaleChanger has resized its platform, a hammer
    /// has killed and a TeleportIndicator has detached its holder.
    /// They run in the order Updater lists them: Rope.UpdateSim (the reel, the timer
    /// and the offscreen drop), the hook's RopeAttachment (or the separate
    /// RopeAttachment prefab WebSwing instantiates after the Rope), then the
    /// RopeHook (Rope.cs:171-236, HookshotInstant.cs:128-132,
    /// WebSwing.cs:149,159).
    pub(super) fn rope_turn(&mut self, index: usize) {
        self.grapple_rope_step(index);
        self.rope_step(index);
        self.grapple_hook_step(index);
        self.rope_attachment_step(index);
    }

    fn grapple_rope_step(&mut self, rope: usize) {
        let source = &data::get().ropes;
        for index in [rope] {
            if !self.ropes[index].physics.enabled
                || self.ropes[index].appended > self.tick
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
            // Rope.UpdateSim times out a rope whose hook has not arrived, even
            // one already reeled out to full length.
            if !self.ropes[index].hook_arrived
                && self.ropes[index].rope_time > source.rope_max_travel_time
            {
                self.disable_rope(index);
                continue;
            }
            if self.ropes[index].reel_in {
                // Arrival leaves the current segment count fixed; player
                // input performs the later reel-in and deletion.
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
        }
    }

    /// The flying hook's RopeAttachment, then its RopeHook: the hook prefab
    /// lists them after the Rope (RopeHook.prefab: RopeAttachment 1,
    /// DPhysicsCircle 2, RopeHook 3). A rope stretched past its break length
    /// is let go and the hook destroyed; the Rope keeps reeling out until it
    /// times out.
    fn grapple_hook_step(&mut self, rope: usize) {
        let source = &data::get().ropes;
        for index in [rope] {
            if !self.ropes[index].physics.enabled
                || self.ropes[index].appended > self.tick
                || self.ropes[index].hook_entity.is_none()
            {
                continue;
            }
            if self.ropes[index].hook_active && self.rope_attachment_breaks(index, true, false) {
                self.ropes[index].hook_active = false;
                if let Some(entity) = self.ropes[index].hook_entity {
                    self.entities[entity].flags = E_REMOVED;
                    // RopeAttachment.Deattach DestroyFixes the GameObject it
                    // shares with the hook (RopeAttachment.cs:150-163).
                    self.destroy_fix_hook(index);
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
                    // RopeHook.UpdateSim DestroyFixes the rope's GameObject
                    // and its own (RopeHook.cs:53-56).
                    self.destroy_fix(self.ropes[index].hierarchy, None, None);
                    self.destroy_fix_hook(index);
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
        // RaycastToClosestRoundedRect (WebSwing.cs:138).
        let hit = self.raycast_to_closest_rounded_rect(
            position + direction * Fix::from_raw(4_724_464_128),
            direction,
            WEB_MAX_LENGTH,
            None,
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
        // The Rope and RopeAttachment prefabs both count (WebSwing.cs:149,159).
        let hierarchy = self.register_prefab(data::get().prefab("Rope"));
        self.register_prefab(data::get().prefab("RopeAttachment"));
        let rope = self.allocate_rope(player, near, position, separation, Some(platform), local, hierarchy);
        let number = self.ropes[rope].number;
        self.attach_player_rope(player, None, false);
        self.attach_player_rope(player, Some(rope), false);
        if self.players[player].base.grounded {
            // UnGround raises OnNoLongerGrounded: State.fall.
            let slot = &mut self.players[player];
            UngroundBody::Slime(slot.slime_mut()).unground(&self.platforms, self.tick, true);
            slot.base.velocity += Vec2::UP * WEB_GROUNDED_POP;
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
            // SlimeController.Jump (SlimeController.cs:897-901).
            self.players[player].base.jump(&self.platforms);
        }
        self.players[player].base.previous_jump = action.buttons & 1 != 0;
        self.slot_physics(player, |slot, world| slot.base.move_only(input, world));
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
        if self.players[player].rope.is_none() {
            return;
        }
        let held = self.rope_body_release(player, jump_pressed);
        let body = &mut self.players[player].base;
        body.position += (body.velocity + body.external_velocity) * DT;
        if held {
            self.rope_body_hold(player);
        }
        self.players[player].base.external_velocity *= Fix::from_f64(0.95);
    }

    /// The head of PlayerBody.UpdateSim on a roped body: drop a disabled
    /// rope, and let the owner's airborne jump press release it with the
    /// endpoint's velocity. Whether the body still holds a live rope.
    pub(super) fn rope_body_release(&mut self, player: usize, jump_pressed: bool) -> bool {
        let Some(rope_index) = self.players[player].rope else {
            return false;
        };
        if !self.ropes[rope_index].physics.enabled {
            // PlayerBody drops the disabled rope and steps like an unroped body.
            self.players[player].rope = None;
            return false;
        }
        let top = self.players[player].rope_top;
        // The PlayerPhysics beside this PlayerBody: the slime's, or the form's own.
        let jumped = self.players[player].base.jumped_this_frame;
        if jump_pressed
            && self.ropes[rope_index].owner == player
            && !self.players[player].base.grounded
            && !jumped
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
        self.players[player].rope == Some(rope_index)
    }

    /// The rope branch of PlayerBody.UpdateSim after the body has moved by
    /// its velocity: pin or pull the endpoint and note the velocities the
    /// late pass diffs against.
    pub(super) fn rope_body_hold(&mut self, player: usize) {
        let Some(rope_index) = self.players[player].rope else {
            return;
        };
        let top = self.players[player].rope_top;
        let tuning = self.rope_body_tuning(player);
        let body = &mut self.players[player].base;
        if hold_rope_end(body, &mut self.ropes[rope_index], top, tuning) {
            self.attach_player_rope(player, None, top);
        }
        if self.players[player].active != ActiveAbility::Dash {
            // The Dash prefab owns a distinct PlayerBody whose private
            // previous-velocity fields start at zero. Preserve the
            // inactive ordinary slime's fields for the exit handoff.
            let slot = &mut self.players[player];
            slot.rope_prev[slot.rope_body()] = (slot.base.velocity, slot.base.external_velocity);
        }
    }

    /// Dash.ExitAbility transfers the still-attached RopeBody back to the
    /// ordinary slime. The bottom endpoint is rebuilt as a straight strand so
    /// its stored Verlet velocity cannot inject the pre-dash swing on return.
    /// It reads the Dash body's position before `ability.ExitAbility` hands
    /// over to SlimeController.onExitAbility, whose VelocityBasedRaycasts
    /// move the slime afterwards (Dash.cs:203-221, 240).
    pub(super) fn dash_rope_exit(&mut self, player: usize, input: Vec2, leave_speed: Fix) {
        let Some(rope_index) = self.players[player].rope else {
            return;
        };
        let position = self.players[player].base.dash.exit_position;
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

    /// RopeHook.OnCollide's teardown on attaching: DestroyFix on the hitbox
    /// and on the RopeHook, leaving its RopeAttachment and FixTransform
    /// (RopeHook.cs:182-189).
    fn destroy_fix_hook_parts(&mut self, rope_index: usize) {
        let ropes = &data::get().ropes;
        let owner = self.ropes[rope_index].hook_hierarchy;
        for local in [ropes.hook_collider_hierarchy, ropes.hook_component_hierarchy] {
            self.destroy_fix(owner, Some(100_000_000 + owner + local), None);
        }
    }

    /// Updater.DestroyFix on the hook's GameObject.
    fn destroy_fix_hook(&mut self, rope_index: usize) {
        let rope = &self.ropes[rope_index];
        self.destroy_fix(rope.hook_hierarchy, None, rope.hook_entity);
    }

    /// The tail of every RopeAttachment.UpdateSim: a segment stretched past
    /// the attachment's break length, the platform one when it is a platform
    /// attachment, lets go of its end (RopeAttachment.cs:143-151). Whether it
    /// did; the caller drops the attachment, which Deattach destroys.
    fn rope_attachment_breaks(&mut self, index: usize, top: bool, platform: bool) -> bool {
        let source = &data::get().ropes;
        let limit = if platform { source.platform_break_stretch } else { source.break_stretch };
        let physics = &self.ropes[index].physics;
        let stretched = (0..physics.count.saturating_sub(2))
            .any(|segment| (physics.segment[segment] - physics.segment[segment + 1]).length_squared() > limit);
        if stretched {
            self.deattach_rope(index, top);
        }
        stretched
    }

    /// Rope.UpdateSim's timer and its offscreen drop, after the reel.
    fn rope_step(&mut self, rope: usize) {
        let water = data::get().maps[self.map].scene_water_height();
        for index in [rope] {
            if !self.ropes[index].physics.enabled || self.ropes[index].appended > self.tick {
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
            }
        }
    }

    /// A Black Hole's or platform's RopeAttachment on rope `rope`.
    fn rope_attachment_step(&mut self, rope: usize) {
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        let mut moved_platform = false;
        for index in [rope] {
            if !self.ropes[index].physics.enabled || self.ropes[index].appended > self.tick {
                continue;
            }
            let hierarchy = self.ropes[index].hierarchy;
            if let Some((hole_index, base)) = self.ropes[index].attach[0].black_hole {
                if hole_index >= self.black_holes.len()
                    || !self.black_holes[hole_index].active
                    || self.black_holes[hole_index].hierarchy_base != base
                    || self.blink_object_hidden(self.black_holes[hole_index].entity)
                {
                    self.deattach_rope(index, true);
                    self.ropes[index].attach[0].black_hole = None;
                    continue;
                }
                let hole = self.black_holes[hole_index];
                if self.ropes[index].physics.time_since_hook_arrived > Fix::ZERO
                    && !self.ropes[index].physics.detached
                    && !stopped
                {
                    if hole.mass > Fix::ZERO {
                        self.ropes[index].attach[0].reel_frames += 1;
                        let frame = self.ropes[index].attach[0].reel_frames;
                        if ((frame > 60 && frame.is_multiple_of(3))
                            || (frame <= 60 && frame.is_multiple_of(8))
                            || self.ropes[index].physics.count < 8)
                            && !self.ropes[index].physics.reel(3, false)
                        {
                            let far = self.ropes[index].physics.count - 1;
                            self.ropes[index].physics.segment[far] = hole.position;
                        }
                    } else {
                        // A hole with no mass left is let go of, and the
                        // stretch check below still runs (RopeAttachment.cs:121-125).
                        self.ropes[index].attach[0].black_hole = None;
                        self.deattach_rope(index, true);
                    }
                }
                // The attached end stays where the last LateUpdateSim put it
                // (RopeAttachment.cs:166-178): DetPhysics simulates the rope
                // from there, not from where the older hole moved this pass.
                if self.rope_attachment_breaks(index, true, false) {
                    self.ropes[index].attach[0].black_hole = None;
                }
                continue;
            }
            // RopeAttachment on an object takes its own turn: rope_parent_step.
            if self.ropes[index].attach[0].parent.is_some() {
                continue;
            }
            if self.ropes[index].attach[0].turn.is_none() {
                moved_platform |= self.rope_platform_hold(index, 0, hierarchy, stopped);
            }
        }
        if moved_platform {
            self.platforms_changed();
        }
    }

    /// RopeAttachment.UpdateSim lets go of a parent out of the hierarchy
    /// (RopeAttachment.cs:96-100). A QuantumTunnel or TeleportedObjectEffect
    /// takes any parent out while it holds it. A SpikeAttack, SparkNode or
    /// RocketEngine hangs under its platform's GameObject (Spike.cs:122,
    /// PlaceSparkNode.cs:40, PlaceRocketEngine.cs:45), so it also leaves with
    /// the platform or MachoThrow2's carry.
    fn rope_parent_active(&self, parent: RopeParent) -> bool {
        match parent {
            RopeParent::Spike(spike, _) => {
                let spike = &self.spike_attacks[spike];
                self.attached_live(spike.platform_index, spike.entity)
            }
            RopeParent::Node(node, _) => {
                let node = &self.tesla_nodes[node];
                self.attached_live(node.platform_index, node.entity)
            }
            RopeParent::Engine(engine, _) => {
                let engine = &self.engines[engine];
                self.attached_live(engine.platform_index, engine.entity)
            }
            RopeParent::Rock(player) => !self.blink_object_hidden(Some(self.players[player].rock_entity)),
            RopeParent::Item(item, _) => !self.blink_object_hidden(self.thrown_items[item].entity),
            RopeParent::Shot(shot, _) => !self.blink_object_hidden(self.portal_shots[shot].entity),
            RopeParent::Arrow(arrow, _) => !self.blink_object_hidden(self.arrows[arrow].entity),
        }
    }

    /// A platform RopeAttachment's UpdateSim on rope `index`'s end held by
    /// attachment `slot`, at Updater number `hierarchy`: let go of a platform
    /// gone or out of the hierarchy, pull it by the end's stretch, and pin
    /// the end on it (RopeAttachment.cs:85-151). Whether it pushed the
    /// platform.
    fn rope_platform_hold(&mut self, index: usize, slot: usize, hierarchy: u32, stopped: bool) -> bool {
        let mut moved = false;
        if self.ropes[index].attach[slot].gone
            && self.ropes[index].attach[slot].destroyed_at.0 == self.tick
            && hierarchy < self.ropes[index].attach[slot].destroyed_at.1
        {
            // The platform is destroyed later in this pass, so this
            // attachment held the end on it, where it last stood.
            let rope = &mut self.ropes[index];
            let endpoint = if slot == 0 { 0 } else { rope.physics.count - 1 };
            rope.physics.segment[endpoint] = rope.attach[slot].anchor;
            return false;
        }
        if std::mem::take(&mut self.ropes[index].attach[slot].gone) {
            self.deattach_rope(index, slot == 0);
            return false;
        }
        let Some(platform) = self.ropes[index].attach[slot].platform else {
            return false;
        };
        if self.ropes[index].attach[slot].attach_tick == self.tick {
            return false;
        }
        // A boulder's DestroyIfOutsideSceneBounds, older than this
        // RopeAttachment, drowns it earlier in the pass; an evicted
        // platform's ResizablePlatform may have destroyed it already too.
        if platform >= self.platforms.len()
            || self.entities[self.platform_entities[platform]].flags & E_ACTIVE == 0
            || !self.platforms[platform].active
            || self.boulder_drowns_before(platform, hierarchy)
            || (self.ropes[index].attach[slot].destroyed_at.0 == self.tick
                && self.ropes[index].attach[slot].destroyed_at.1 < hierarchy)
        {
            self.deattach_rope(index, slot == 0);
            self.ropes[index].attach[slot].platform = None;
            return false;
        }
        let top = slot == 0;
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
            .position(self.ropes[index].attach[slot].local, Fix::ZERO);
        if self.ropes[index].physics.time_since_hook_arrived > Fix::ZERO
            && !self.ropes[index].physics.detached
            && !stopped
        {
            let force = (self.ropes[index].physics.segment[endpoint]
                - self.ropes[index].physics.segment[neighbor])
                * self.ropes[index].attach[slot].pull;
            let arm = (self.ropes[index].attach[slot].anchor - self.platforms[platform].surface.center)
                .normalized_safe();
            self.platform_dynamics[platform]
                .body
                .apply_impulse(force, arm);
            moved = true;
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
        self.ropes[index].attach[slot].anchor = anchor;
        if self.rope_attachment_breaks(index, top, true) {
            self.ropes[index].attach[slot].platform = None;
        }
        moved
    }

    /// The Updater key of a RopeAttachment prefab PlatformTransform
    /// instantiated on rope `index`'s end `slot`.
    pub(super) fn rope_end_attachment_key(&self, index: usize, slot: usize) -> Option<UpdaterKey> {
        let (appended, hierarchy) = self.ropes[index].attach[slot].turn?;
        Some(self.updater_member_key(appended, 100_000_000 + hierarchy))
    }

    /// That RopeAttachment's UpdateSim at its own turn in Updater's walk.
    pub(super) fn rope_end_attachment_step(&mut self, index: usize, slot: usize) {
        let Some((appended, hierarchy)) = self.ropes[index].attach[slot].turn else {
            return;
        };
        if !self.ropes[index].physics.enabled || appended > self.tick {
            return;
        }
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        if self.rope_platform_hold(index, slot, hierarchy, stopped) {
            self.platforms_changed();
        }
    }

    /// The Updater key of the RopeAttachment on rope `index`'s hook: the
    /// hook prefab's member numbered HOOK_ATTACHMENT_LOCAL.
    pub(super) fn rope_attachment_key(&self, index: usize) -> UpdaterKey {
        let rope = &self.ropes[index];
        self.updater_member_key(rope.appended, 100_000_000 + rope.hook_hierarchy + HOOK_ATTACHMENT_LOCAL)
    }

    /// RopeAttachment.UpdateSim for a hook caught on an object, at the
    /// attachment's own turn in Updater's walk: it sits on the hook prefab,
    /// so a Portal Gun shot, tack, Poke Ball or hammer listed before it has
    /// taken its turn and one listed after it has not, and a form listed
    /// after it still holds the rope's other end (RopeAttachment.cs:68-151).
    pub(super) fn rope_parent_step(&mut self, index: usize) {
        let stopped = self.player_time_scale < Fix::from_f32(0.01);
        {
            if !self.ropes[index].physics.enabled || self.ropes[index].appended > self.tick {
                return;
            }
            let Some(parent) = self.ropes[index].attach[0].parent else {
                return;
            };
            let Some(position) = parent_position(
                &self.players,
                &self.thrown_items,
                &self.portal_shots,
                (&self.spike_attacks, &self.tesla_nodes, &self.arrows, &self.engines),
                parent,
                self.ropes[index].attach[0].offset,
            )
            .filter(|_| self.rope_parent_active(parent)) else {
                self.deattach_rope(index, true);
                self.ropes[index].attach[0].parent = None;
                return;
            };
            let physics = &self.ropes[index].physics;
            if physics.time_since_hook_arrived > Fix::ZERO && !physics.detached && !stopped {
                // AddForceAtPosition at the attachment's own position
                // (DPhysicsCircle.cs:377-393).
                let force = (physics.segment[0] - physics.segment[1]) * ATTACHMENT_PULL;
                // The arm runs from the collider's FixTransform position, not
                // the collider. A SpikeAttack's box and a SparkNode's circle
                // are composite children, so DPhysicsBox and DPhysicsCircle
                // hand the pull to the platform's combined body
                // (DPhysicsBox.cs:436-451, DPhysicsCircle.cs:377-393).
                let centre = parent_pose(
                    &self.players,
                    &self.thrown_items,
                    &self.portal_shots,
                    (&self.spike_attacks, &self.tesla_nodes, &self.arrows, &self.engines),
                    parent,
                )
                .position;
                let arm = (self.ropes[index].attach[0].anchor - centre).normalized_safe();
                let body = match parent {
                    RopeParent::Rock(player) => Some(&mut self.players[player].rock.body),
                    RopeParent::Item(item, _) => Some(&mut self.thrown_items[item].body),
                    RopeParent::Shot(shot, _) => Some(&mut self.portal_shots[shot].body),
                    RopeParent::Spike(spike, _) => Some(
                        &mut self.platform_dynamics[self.spike_attacks[spike].platform_index].body,
                    ),
                    RopeParent::Node(node, _) => Some(
                        &mut self.platform_dynamics[self.tesla_nodes[node].platform_index].body,
                    ),
                    RopeParent::Engine(engine, _) => Some(
                        &mut self.platform_dynamics[self.engines[engine].platform_index].body,
                    ),
                    RopeParent::Arrow(..) => None,
                };
                if let Some(body) = body {
                    body.apply_impulse(force, arm);
                } else if let RopeParent::Arrow(arrow, _) = parent {
                    // An arrow keeps its BoplBody's velocities on itself.
                    let arrow = &mut self.arrows[arrow];
                    arrow.velocity += force * arrow.inverse_mass;
                    arrow.angular_velocity += arrow.inverse_inertia * arm.cross(force);
                }
            }
            self.ropes[index].attach[0].anchor = position;
            if self.rope_attachment_breaks(index, true, false) {
                self.ropes[index].attach[0].parent = None;
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
        if !self.ropes.iter().any(|rope| rope.physics.enabled) {
            return;
        }
        // DetPhysics.Simulate lists the active rounded rectangles' boxes, then
        // every enabled rope's, in bb_rope_rr, and SortAndSweep*_ropes sweeps
        // them along the axis the rounded rectangles alone spread widest.
        let sweep_x = self.sweep_axes().1;
        let mut boxes = std::mem::take(&mut self.scratch.rope_boxes);
        let mut active = std::mem::take(&mut self.scratch.rope_active);
        boxes.clear();
        for (platform, rectangle) in self.rope_rectangles.iter().enumerate() {
            if self.platforms[platform].active {
                let bounds = rectangle.bounds();
                boxes.push(RopeSweepBox { min: bounds.min, max: bounds.max, rope: false, id: platform });
            }
        }
        for (index, rope) in self.ropes.iter().enumerate() {
            if rope.physics.enabled {
                let [min, max] = rope.physics.bounds();
                boxes.push(RopeSweepBox { min, max, rope: true, id: index });
            }
        }
        let (along, across): (fn(Vec2) -> Fix, fn(Vec2) -> Fix) =
            if sweep_x { (|v| v.x, |v| v.y) } else { (|v| v.y, |v| v.x) };
        // AABBXComparer/AABBYComparer: the minimum, then `a.shape - b.shape`,
        // which Shape's byte wraps positive both ways, so a rounded rectangle
        // and a rope on one minimum each compare greater and Array.Sort's
        // introsort alone decides their order; then the id.
        dotnet_sort::introspective_sort(&mut boxes, |a, b| {
            along(a.min).partial_cmp(&along(b.min)).unwrap().then_with(|| {
                if a.rope != b.rope { std::cmp::Ordering::Greater } else { a.id.cmp(&b.id) }
            })
        });
        active.clear();
        active.push(0);
        for i in 1..boxes.len() {
            let current = boxes[i];
            let mut num = 0;
            while num < active.len() {
                let other = boxes[active[num]];
                if along(other.max) <= along(current.min) {
                    active.remove(num);
                    continue;
                }
                if across(other.max) > across(current.min)
                    && across(other.min) < across(current.max)
                    && other.rope != current.rope
                {
                    let (rope, rectangle) = if other.rope { (other.id, current.id) } else { (current.id, other.id) };
                    let physics = &mut self.ropes[rope].physics;
                    if physics.collision_count < physics.colliding_rects.len() {
                        physics.colliding_rects[physics.collision_count] = rectangle;
                        physics.collision_count += 1;
                    }
                }
                num += 1;
            }
            active.push(i);
        }
        self.scratch.rope_boxes = boxes;
        self.scratch.rope_active = active;
        for rope in &mut self.ropes {
            if !rope.physics.enabled {
                continue;
            }
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
            let Some(platform) = rope.attach[0].platform else {
                continue;
            };
            let surface = &self.platforms[platform].surface;
            if let Ok(local) = surface.local_position(position, Fix::ZERO) {
                rope.attach[0].local = local;
                rope.attach[0].anchor = surface.position(local, Fix::ZERO);
            }
        }
    }

    pub(super) fn hook_live(&self, rope: usize) -> bool {
        self.ropes[rope].physics.enabled && self.ropes[rope].hook_active
    }

    pub(super) fn hook_circle(&self, rope: usize) -> Circle {
        Circle { center: self.ropes[rope].hook_position, radius: data::get().ropes.hook_radius }
    }

    /// Queues RopeHook.OnCollide for one of the hook's pairs at its place in
    /// DetPhysics' callback queue, with the collider the pair met.
    pub(super) fn queue_hook_hit(
        &mut self,
        rope: usize,
        group: u8,
        other: SweepRank,
        target: HookTarget,
        shape: Collider,
    ) {
        let circle = self.hook_circle(rope);
        let min = if self.sweep_x() { circle.center.x } else { circle.center.y } - circle.radius;
        let rank = sweep_rank(&self.entities, min, self.ropes[rope].hook_entity);
        let (later, earlier) = sweep_pair(rank, other);
        self.scratch
            .callback_hits
            .push(((group, later, earlier), CallbackHit::RopeHook { rope, target, shape }));
    }

    /// The hooks' pairs with black holes, which circle_circle_group does not
    /// list yet, and their Circle/Box pairs, on the colliders the
    /// Circle/Circle corrections left.
    pub(super) fn grapple_hook_pairs(&mut self) {
        let sweep_x = self.sweep_x();
        let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
        let circle_circle = GROUP_CC;
        let mut hits = Vec::new();
        for rope in 0..self.ropes.len() {
            if !self.hook_live(rope) {
                continue;
            }
            let circle = self.hook_circle(rope);
            let touches = |other: Circle| crate::physics::contact::circle_circle(circle, other).is_some();
            for (index, hole) in self.black_holes.iter().enumerate() {
                let shape = Circle { center: hole.position, radius: hole.radius };
                if hole.active && !self.blink_object_hidden(hole.entity) && touches(shape) {
                    let rank = sweep_rank(&self.entities, axis(hole.position) - hole.radius, hole.entity);
                    hits.push((rope, circle_circle, rank, HookTarget::BlackHole(index), Collider::Circle(shape)));
                }
            }
            // RopeHook ignores every body of the caster's and the end's
            // player (RopeHook.cs:122-129).
            let owner = self.players[self.ropes[rope].owner].owner;
            let end_player = self.ropes[rope].hook_end_player;
            for player in 0..self.players.len() {
                let body_player = self.players[player].owner;
                if body_player == owner || body_player == end_player {
                    continue;
                }
                // The Player-layer collider DetPhysics holds, a spawning
                // slime's included. A Rock form's circle meets the hook in
                // circle_circle_group. A form's own Player-layer collider
                // carries its PlayerBody, so the hook meets the form's box.
                if let Some(circle_body) = self.body_part_circle(player)
                    && let Some(shape) = self.player_hit_shape(player)
                    && Collider::Circle(circle).contact(shape).is_some()
                {
                    let group = GROUP_CB - circle_body as u8;
                    hits.push((rope, group, self.player_sweep_rank(player), HookTarget::Player(player), shape));
                }
            }
            for index in self.player_layer_clones() {
                let clone = &self.abilityless_clones[index];
                if clone.owner != owner
                    && clone.owner != end_player
                    && crate::physics::contact::circle_box(circle, clone.base.synced).is_some()
                {
                    let collider = clone.base.synced;
                    let min = axis(collider.center) - axis(collider.right).abs() - axis(collider.up).abs();
                    let rank = sweep_rank(&self.entities, min, clone.entity);
                    hits.push((rope, GROUP_CB, rank, HookTarget::Clone(index), Collider::Box(collider)));
                }
            }
            for index in 0..self.spike_attacks.len() {
                let spike = self.spike_attacks[index];
                if !spike.active || !spike.entity.is_some_and(|entity| self.collides(entity)) {
                    continue;
                }
                let shape = self.spike_attack_box(spike);
                if crate::physics::contact::circle_box(circle, shape).is_some() {
                    let rank = self.part_rank(Part::Spike(index));
                    hits.push((rope, GROUP_CB, rank, HookTarget::Spike(index), Collider::Box(shape)));
                }
            }
            for index in 0..self.engines.len() {
                if !self.engines[index].entity.is_some_and(|entity| self.collides(entity)) {
                    continue;
                }
                let shape = self.engines[index].rocket_box(&self.entity_inverse_extents);
                if crate::physics::contact::circle_box(circle, shape).is_some() {
                    let rank = self.part_rank(Part::Engine(index));
                    hits.push((rope, GROUP_CB, rank, HookTarget::Engine(index), Collider::Box(shape)));
                }
            }
        }
        for (rope, group, rank, target, shape) in hits {
            self.queue_hook_hit(rope, group, rank, target, shape);
        }
    }

    /// RopeHook.OnCollide for one pair, with the collider its manifold
    /// copied, in DetPhysics' callback order. The first pair that attaches
    /// destroys the hook, and every later one returns (RopeHook.cs:86-190).
    pub(super) fn rope_hook_collide(&mut self, rope_index: usize, target: HookTarget, shape: Collider) {
        if !self.hook_live(rope_index) {
            return;
        }
        let source = &data::get().ropes;
        let position = self.ropes[rope_index].hook_position;
        // A wall, LethalTerrain or NonLethalTerrain collider takes the hook
        // only when the hook's centre is inside the copy, and the end keeps
        // the hook's place on it; any other holds the end at the copy's
        // centre (RopeHook.cs:92-100).
        let terrain = matches!(
            target,
            HookTarget::Platform(_) | HookTarget::Node(_) | HookTarget::Spike(_) | HookTarget::Engine(_)
        );
        if terrain && !shape.overlaps(position) {
            return;
        }
        let point = if terrain { position } else { shape.center() };
        let parent = match target {
            HookTarget::Platform(platform) => {
                let local = self.platforms[platform]
                    .surface
                    .local_position(position, Fix::ZERO)
                    .expect("grapple hook centre overlaps platform");
                let anchor = self.platforms[platform].surface.position(local, Fix::ZERO);
                let platform_pull = self.rope_platform_pull(platform);
                let rope = &mut self.ropes[rope_index];
                rope.physics.mass_start = Fix::ONE;
                rope.physics.force_start = Vec2::ZERO;
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.attach[0].platform = Some(platform);
                rope.attach[0].local = local;
                rope.attach[0].anchor = anchor;
                rope.hook_hit = Some(position);
                rope.attach[0].pull = platform_pull;
                rope.reel_in = true;
                rope.hook_arrived = true;
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
                self.destroy_fix_hook_parts(rope_index);
                return;
            }
            HookTarget::Pickup => {
                // RopeHook.OnCollide marks the hook arrived before it looks at
                // the layer, so an AbilityPickup on the weapon layer, which it
                // does not attach to, still ends the reel-out (RopeHook.cs:115-120).
                self.ropes[rope_index].reel_in = true;
                self.ropes[rope_index].hook_arrived = true;
                return;
            }
            HookTarget::BlackHole(hole_index) => {
                let rope = &mut self.ropes[rope_index];
                rope.physics.mass_start = Fix::ONE;
                rope.physics.force_start = Vec2::ZERO;
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.physics.set_mass(source.non_platform_mass, true, Fix::ONE);
                rope.time_air_start = Fix::ONE;
                rope.attach[0].platform = None;
                rope.attach[0].black_hole = Some((hole_index, self.black_holes[hole_index].hierarchy_base));
                rope.attach[0].reel_frames = 0;
                rope.reel_in = true;
                rope.hook_arrived = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // Hitbox-only teardown, as for a wall (RopeHook.cs:182-189).
                    self.entities[entity].flags =
                        (self.entities[entity].flags & (E_ACTIVE | E_TRIGGER)) | E_DESTROYED;
                }
                self.destroy_fix_hook_parts(rope_index);
                return;
            }
            HookTarget::Clone(index) => {
                let tick = self.tick;
                let clone = &mut self.abilityless_clones[index];
                if let Some(old) = clone.grip.rope
                    && old != rope_index
                {
                    self.ropes[old].physics.detach(true);
                    self.ropes[old].physics.detach(false);
                    clone.grip.rope = None;
                }
                attach_grip(&clone.base, &mut clone.grip, &mut self.ropes, Some(rope_index), true, tick);
                let rope = &mut self.ropes[rope_index];
                rope.physics.time_since_hook_arrived = Fix::ZERO;
                rope.reel_in = true;
                rope.hook_arrived = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    self.entities[entity].flags = (self.entities[entity].flags & E_TRIGGER) | E_DESTROYED;
                }
                self.destroy_fix_hook(rope_index);
                return;
            }
            HookTarget::Player(victim) if self.players[victim].active != ActiveAbility::Rock => {
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
                rope.hook_arrived = true;
                rope.hook_active = false;
                rope.hook_remove_next = true;
                if let Some(entity) = rope.hook_entity {
                    // Alone among the collide branches, a hit on a body with a
                    // PlayerBody destroys the whole hook GameObject and returns
                    // before the hitbox-only teardown (RopeHook.cs:144-150), so
                    // activeInHierarchy goes with it.
                    self.entities[entity].flags = (self.entities[entity].flags & E_TRIGGER) | E_DESTROYED;
                }
                self.destroy_fix_hook(rope_index);
                return;
            }
            // A BounceBall has no PlayerBody, so the hook attaches to it as to
            // any other body.
            HookTarget::Player(player) => RopeParent::Rock(player),
            HookTarget::Item(index) => RopeParent::Item(index, self.thrown_items[index].hierarchy_base),
            HookTarget::Shot(index) => RopeParent::Shot(index, self.portal_shots[index].hierarchy_base),
            HookTarget::Arrow(index) => RopeParent::Arrow(index, self.arrows[index].hierarchy_base),
            HookTarget::Node(index) => RopeParent::Node(index, self.tesla_nodes[index].hierarchy_base),
            HookTarget::Spike(index) => RopeParent::Spike(index, self.spike_attacks[index].hierarchy_base),
            HookTarget::Engine(index) => RopeParent::Engine(index, self.engines[index].hierarchy_base),
        };
        // The end's offset in the parent's FixTransform as the callback finds
        // it, after SyncGameObject (RopeHook.cs:154-157, 172-175).
        let pose = parent_pose(
            &self.players,
            &self.thrown_items,
            &self.portal_shots,
            (&self.spike_attacks, &self.tesla_nodes, &self.arrows, &self.engines),
            parent,
        );
        let offset = pose.offset(point);
        let scale = pose.scale;
        let rope = &mut self.ropes[rope_index];
        rope.physics.mass_start = Fix::ONE;
        rope.physics.force_start = Vec2::ZERO;
        rope.physics.time_since_hook_arrived = Fix::ZERO;
        rope.physics.set_mass(source.non_platform_mass, true, scale);
        rope.time_air_start = Fix::ONE;
        rope.attach[0].platform = None;
        // The attachment shares the hook's GameObject, so its transform stays
        // on the hook until a pass re-places it.
        rope.attach[0].anchor = rope.hook_position;
        rope.attach[0].parent = Some(parent);
        rope.attach[0].offset = offset;
        rope.reel_in = true;
        rope.hook_arrived = true;
        rope.hook_active = false;
        rope.hook_remove_next = true;
        if let Some(entity) = rope.hook_entity {
            // Hitbox-only teardown, as for a wall (RopeHook.cs:182-189).
            self.entities[entity].flags = (self.entities[entity].flags & (E_ACTIVE | E_TRIGGER)) | E_DESTROYED;
        }
        self.destroy_fix_hook_parts(rope_index);
    }

    /// The abilityless clones whose Player-layer box DetPhysics holds, in
    /// the order their boxes registered.
    pub(super) fn player_layer_clones(&self) -> Vec<usize> {
        let mut clones: Vec<usize> = (0..self.abilityless_clones.len())
            .filter(|&index| {
                let clone = &self.abilityless_clones[index];
                clone.active
                    && clone.base.alive
                    && clone.entity.is_some()
                    && clone.captured_by_ball.is_none()
                    && !self.blink_object_hidden(clone.entity)
            })
            .collect();
        clones.sort_by_key(|&index| self.abilityless_clones[index].entity);
        clones
    }

    /// PlayerPhysics and PlayerBody LateUpdateSim for the abilityless clones
    /// that grip a rope, as `ropes_late` does for a slime.
    fn clone_ropes_late(&mut self) {
        for clone_index in self.player_layer_clones() {
            let clone = &mut self.abilityless_clones[clone_index];
            let Some(rope_index) = clone.grip.rope else {
                continue;
            };
            if !self.ropes[rope_index].physics.enabled {
                clone.grip.rope = None;
                continue;
            }
            let top = clone.grip.top;
            let rope = &mut self.ropes[rope_index];
            if clone.base.grounded
                && clone.ground_destroyed.is_none()
                && rope.physics.time_since_hook_arrived > Fix::ZERO
                && rope.appended <= self.tick
                && clone.grip.attach_tick != self.tick
            {
                let body = UngroundBody::Slime(clone.slime_mut());
                pin_grounded_rope_end(body, rope, top, &self.platforms, self.tick);
            }
            if overstretched(rope) {
                attach_grip(&clone.base, &mut clone.grip, &mut self.ropes, None, top, self.tick);
                continue;
            }
            let swing = std::mem::take(&mut clone.grip.swing_force);
            let force = rope_end_force(&clone.base, clone.grip.previous, false, swing);
            if top {
                rope.physics.force_start = force;
            } else {
                rope.physics.force_end = force;
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
                && self.ropes[rope_index].appended <= self.tick
                && self.players[player].rope_attach_tick != self.tick
            {
                let slot = &mut self.players[player];
                // PlayerPhysics.LateUpdateSim's UnGround nulls the rotation
                // and raises the OnNoLongerGrounded of the slime or the form.
                let body = if slot.active == ActiveAbility::None {
                    UngroundBody::Slime(slot.slime_mut())
                } else {
                    UngroundBody::Form(slot)
                };
                let rope = &mut self.ropes[rope_index];
                pin_grounded_rope_end(body, rope, top, &self.platforms, self.tick);
            }
            // An unsimulated PlayerBody (Push.prefab's, a spawning slime's)
            // skips its late update (stretch check, end force); the
            // PlayerPhysics above still pins a grounded end.
            if !self.players[player].base.simulated {
                continue;
            }
            if overstretched(&self.ropes[rope_index]) {
                self.attach_player_rope(player, None, top);
            }
            if self.players[player].rope == Some(rope_index) {
                let slot = &mut self.players[player];
                let previous = if slot.active == ActiveAbility::Dash {
                    (Vec2::ZERO, Vec2::ZERO)
                } else {
                    slot.rope_prev[slot.rope_body()]
                };
                // Only the slime's own PlayerBody carries the steer; an
                // ability form's body has none and leaves it pending.
                let swing = if slot.active == ActiveAbility::None {
                    std::mem::take(&mut slot.rope_swing_force)
                } else {
                    Vec2::ZERO
                };
                let ignore_gravity = self.rope_body_tuning(player).1;
                let force = rope_end_force(&self.players[player].base, previous, ignore_gravity, swing);
                if top {
                    self.ropes[rope_index].physics.force_start = force;
                } else {
                    self.ropes[rope_index].physics.force_end = force;
                }
            }
        }
        self.clone_ropes_late();
        let (players, items, shots) = (&self.players, &self.thrown_items, &self.portal_shots);
        let spikes = (
            &self.spike_attacks[..],
            &self.tesla_nodes[..],
            &self.arrows[..],
            &self.engines[..],
        );
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
            for (slot, attachment) in rope.attach.iter_mut().enumerate() {
                let endpoint = if slot == 0 { 0 } else { rope.physics.count - 1 };
                if rope.appended <= self.tick
                    && let Some(platform) = attachment.platform
                    && attachment.attach_tick != self.tick
                {
                    // RepositionOnParent skips a destroyed platform, and the end
                    // stays where the attachment last stood (RopeAttachment.cs).
                    if self.entities[self.platform_entities[platform]].flags & E_DESTROYED == 0 {
                        attachment.anchor =
                            self.platforms[platform].surface.position(attachment.local, Fix::ZERO);
                    }
                    rope.physics.segment[endpoint] = attachment.anchor;
                }
                if attachment.gone {
                    rope.physics.segment[endpoint] = attachment.anchor;
                }
            }
            if rope.appended <= self.tick
                && let Some((hole_index, base)) = rope.attach[0].black_hole
                && hole_index < self.black_holes.len()
                && self.black_holes[hole_index].active
                && self.black_holes[hole_index].hierarchy_base == base
            {
                let hole = self.black_holes[hole_index];
                rope.physics.segment[0] = hole.position;
            }
            // LateUpdateSim reads parentObject.position without asking whether
            // it went this pass: a shot that lands against a wall in Simulate
            // still holds the end where it landed. A hook that lands on its
            // cast tick leaves the end on the hook: the attachment has no late
            // pass yet (RopeAttachment.cs:170-178).
            let offset = rope.attach[0].offset;
            let held = |parent| Some(parent_pose(players, items, shots, spikes, parent).place(offset));
            let position = match rope.attach[0].parent {
                _ if rope.appended > self.tick => None,
                Some(parent @ RopeParent::Shot(shot, base)) if shots[shot].hierarchy_base == base => held(parent),
                // An item a DetPhysics callback destroyed after this pass's
                // UpdateSim (DPhysicsCircle.OnCollide) still holds the end
                // where it died; the attachment's UpdateSim saw it alive.
                Some(parent @ RopeParent::Item(item, base))
                    if items[item].hierarchy_base == base
                        && items[item]
                            .entity
                            .is_some_and(|entity| self.entities[entity].flags & E_DESTROYED != 0) =>
                {
                    held(parent)
                }
                // Likewise an arrow that stuck in Simulate (Arrow.OnCollide
                // DestroyFixes its FixTransform, RopeAttachment.cs:174).
                Some(parent @ RopeParent::Arrow(arrow, base)) if spikes.2[arrow].hierarchy_base == base => held(parent),
                parent => parent.and_then(|parent| parent_position(players, items, shots, spikes, parent, offset)),
            };
            if let Some(position) = position {
                rope.attach[0].anchor = position;
                rope.physics.segment[0] = position;
            }
            if rope.hook_active {
                rope.physics.segment[0] = rope.hook_position;
            }
        }
    }

    /// Mine.LateUpdateSim's anticipation Shake moves the mine's transform,
    /// and a RopeAttachment newer than the mine follows it in LateUpdateSim
    /// order, so it pins the end where the shake left the mine.
    pub(super) fn ropes_late_on_older_mines(&mut self) {
        for rope in &mut self.ropes {
            if !rope.physics.enabled || rope.appended > self.tick || rope.hook_active {
                continue;
            }
            let Some(RopeParent::Item(item, base)) = rope.attach[0].parent else {
                continue;
            };
            let mine = &self.thrown_items[item];
            if mine.hierarchy_base != base || mine.ability != ABILITY_MINE || mine.hierarchy_base >= rope.hierarchy {
                continue;
            }
            rope.attach[0].anchor = mine.position;
            rope.physics.segment[0] = mine.position;
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
                    owner: player_id(&self.players, rope.owner),
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
