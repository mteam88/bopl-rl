//! DetPhysics' Circle/Circle and Circle/Box groups (DetPhysics.cs:3052-3129),
//! for the colliders the rollout lists here: every pair of registered
//! colliders the broadphase found on layers the collision matrix pairs, all
//! of a group's manifolds built before any resolves, each pair's callbacks
//! queued first side first (DetPhysics.cs:2405-2437).
use super::*;
use super::rope_abilities::HookTarget;

/// The Circle/Circle group's kinds, in the order it takes their parts.
const CC_ITEMS: usize = 0;
const CC_NODES: usize = 1;
const CC_SHOTS: usize = 2;
const CC_ROCKS: usize = 3;
const CC_PICKUPS: usize = 4;
const CC_DUPLICATED_ROCKS: usize = 5;
const CC_HOOKS: usize = 6;
const CC_ARROWS: usize = 7;
const CC_HOLES: usize = 8;
const CC_CLOUDS: usize = 9;
const CC_BODIES: usize = 10;
const CC_CLONES: usize = 11;
const CC_ORBS: usize = 12;
pub(super) const CC_BLOCKS: usize = 13;

/// A part as a group's pair loop reads it, taken once per part rather than
/// once per pair: nothing the loop reads moves until every manifold is built.
#[derive(Clone, Copy)]
pub(super) struct GroupPart {
    part: Part,
    /// `UNRANKED` for a part the Circle/Box group took (`unranked_part`),
    /// whose rank is read only where it is needed (`group_rank`).
    rank: SweepRank,
    layer: u32,
    composite: Option<usize>,
    bounds: Option<[Vec2; 2]>,
    /// Debug builds: the rank as taken when the part was, which every
    /// rank `group_rank` takes later must equal.
    #[cfg(debug_assertions)]
    eager: SweepRank,
}

/// The rank of a part taken without one (`unranked_part`).
const UNRANKED: SweepRank = (Fix::ZERO, u32::MAX, usize::MAX);

/// Bit-for-bit equality of two ranks, for debug assertions.
#[cfg(debug_assertions)]
fn same_rank(a: SweepRank, b: SweepRank) -> bool {
    a.0.to_f64().to_bits() == b.0.to_f64().to_bits() && (a.1, a.2) == (b.1, b.2)
}

impl GroupPart {
    /// Bit-for-bit equality, for debug assertions: `other` may be unranked,
    /// so its rank is the one it was taken with.
    #[cfg(debug_assertions)]
    fn same(&self, other: &GroupPart) -> bool {
        let bits = |b: Option<[Vec2; 2]>| b.map(|b| [b[0].x, b[0].y, b[1].x, b[1].y].map(|v| v.to_f64().to_bits()));
        self.part == other.part
            && same_rank(self.rank, self.eager)
            && same_rank(self.rank, other.eager)
            && self.layer == other.layer
            && self.composite == other.composite
            && bits(self.bounds) == bits(other.bounds)
    }
    /// Whether the group builds this pair's manifold: not SameCompositeAs,
    /// on layers the collision matrix pairs, and paired by SortAndSweep.
    fn pairs(&self, other: &GroupPart, data: &data::RolloutData) -> bool {
        !self.composite.is_some_and(|c| Some(c) == other.composite)
            && data.layers_collide(self.layer, other.layer)
            && matches!((self.bounds, other.bounds), (Some(a), Some(b)) if super::sweep::swept_meet(a, b))
    }
}

/// The layers of a set of parts, for which of them can pair at all: a pair
/// loop acts only on a pair `GroupPart::pairs` passes, which needs
/// `layers_collide`, so a part with no partner on a layer its own collides
/// with makes no pair, and leaving it out of a rank-sorted group leaves the
/// rest in the same order and every other pair as it was (`sweep_pairs`: a
/// prior's time in `live` reads only its own end). The collision matrix is
/// symmetric, so one row answers both sides.
#[derive(Clone, Copy, Default)]
struct LayerSet {
    present: u32,
    twice: u32,
}

impl LayerSet {
    fn add(&mut self, layer: u32) {
        let bit = 1 << layer;
        self.twice |= self.present & bit;
        self.present |= bit;
    }
    /// Whether a part on `layer` in the set has another there it can pair with.
    fn meets_other(&self, layer: u32, data: &data::RolloutData) -> bool {
        let bit = 1 << layer;
        data.collision_matrix[layer as usize] & ((self.present & !bit) | (self.twice & bit)) != 0
    }
    /// Whether every part in the set has another there it can pair with:
    /// `meets_other` of each layer in it.
    fn each_meets_other(&self, data: &data::RolloutData) -> bool {
        let mut layers = self.present;
        while layers != 0 {
            if !self.meets_other(layers.trailing_zeros(), data) {
                return false;
            }
            layers &= layers - 1;
        }
        true
    }
    /// Whether a part on `layer` outside the set can pair with one in it.
    fn meets(&self, layer: u32, data: &data::RolloutData) -> bool {
        data.collision_matrix[layer as usize] & self.present != 0
    }
}

#[cfg(debug_assertions)]
fn collision_matrix_symmetric(data: &data::RolloutData) -> bool {
    (0..32).all(|a| (0..32).all(|b| data.layers_collide(a, b) == data.layers_collide(b, a)))
}

/// `group.sort_by_key(|part| part.rank)`. A group of a few parts, as most
/// are, takes an insertion sort, which moves a part only past a greater rank;
/// a larger one is sorted without moving the parts about: the (rank, place)
/// pairs are distinct, so an unstable sort of them gives the stable order,
/// and the parts are gathered into it once.
fn sort_by_rank(group: &mut Vec<GroupPart>) {
    if group.len() <= 8 {
        for k in 1..group.len() {
            if group[k - 1].rank <= group[k].rank {
                continue;
            }
            let part = group[k];
            let mut j = k;
            while j > 0 && group[j - 1].rank > part.rank {
                group[j] = group[j - 1];
                j -= 1;
            }
            group[j] = part;
        }
        debug_assert!(group.is_sorted_by_key(|part| part.rank));
        return;
    }
    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<(SweepRank, u32)>, Vec<GroupPart>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    SCRATCH.with_borrow_mut(|(keys, parts)| {
        keys.clear();
        keys.extend(group.iter().enumerate().map(|(place, part)| (part.rank, place as u32)));
        keys.sort_unstable();
        parts.clear();
        parts.extend(keys.iter().map(|&(_, place)| group[place as usize]));
        std::mem::swap(group, parts);
    });
}

/// Whether two parts of a group pass `GroupPart::pairs`, which the pair loops
/// act on only: a group with no such two builds no manifold, sorted or not.
/// Over the test replays one Box/Box group in a hundred has two.
fn any_pairs(group: &[GroupPart], data: &data::RolloutData) -> bool {
    group.iter().enumerate().any(|(k, current)| group[..k].iter().any(|prior| prior.pairs(current, data)))
}

/// A prior still open in `sweep_pairs`: its end on the sweep axis, the
/// layers its own collides with (`layers_collide`'s row), and its place in
/// the group.
#[derive(Clone, Copy)]
pub(super) struct LivePart {
    end: Fix,
    collides: u32,
    place: u32,
}

/// The (prior, current) pairs of a rank-sorted group whose boxes can meet on
/// layers the collision matrix pairs, in the full double loop's order. The
/// ranks rise with the sweep-axis start, so a prior that ends at or before a
/// part's start ends before every later part's too and leaves `live` for
/// good: the pairs skipped are exactly ones `swept_meet` fails on that axis.
/// Each caller acts only on a pair `GroupPart::pairs` passes, which needs
/// `layers_collide`, so the pairs on layers that do not collide are skipped
/// too. Over the test replays the Circle/Circle group's sweep finds 23 pairs
/// a pass, and about one in 20,000 is on layers that collide.
///
/// A group of a few parts, as most are, takes the double loop itself: a
/// prior is in `live` at a part exactly when it ends after that part's
/// start, since the starts before it are no later.
fn sweep_pairs(
    group: &[GroupPart],
    sweep_x: bool,
    data: &data::RolloutData,
    live: &mut Vec<LivePart>,
    mut each: impl FnMut(&GroupPart, &GroupPart),
) {
    let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
    if group.len() <= 8 {
        #[cfg(debug_assertions)]
        let mut met = Vec::new();
        for (k, current) in group.iter().enumerate() {
            let Some(bounds) = current.bounds else { continue };
            let start = axis(bounds[0]);
            for place in 0..k {
                let prior = &group[place];
                if prior.bounds.is_some_and(|prior_bounds| start < axis(prior_bounds[1]))
                    && data.layers_collide(prior.layer, current.layer)
                {
                    #[cfg(debug_assertions)]
                    met.push((place, k));
                    each(prior, current);
                }
            }
        }
        #[cfg(debug_assertions)]
        {
            let mut swept = Vec::new();
            sweep_live(group, axis, data, live, |prior, current| swept.push((prior, current)));
            debug_assert_eq!(met, swept);
        }
        return;
    }
    sweep_live(group, axis, data, live, |prior, current| each(&group[prior], &group[current]));
}

/// `sweep_pairs` over the open priors, by place.
fn sweep_live(
    group: &[GroupPart],
    axis: impl Fn(Vec2) -> Fix,
    data: &data::RolloutData,
    live: &mut Vec<LivePart>,
    mut each: impl FnMut(usize, usize),
) {
    live.clear();
    for (k, current) in group.iter().enumerate() {
        let Some(bounds) = current.bounds else { continue };
        let start = axis(bounds[0]);
        live.retain(|prior| start < prior.end);
        for prior in live.iter() {
            let prior_part = &group[prior.place as usize];
            if prior.collides & (1 << current.layer) != 0 {
                each(prior.place as usize, k);
            } else {
                debug_assert!(!data.layers_collide(prior_part.layer, current.layer));
            }
        }
        live.push(LivePart {
            end: axis(bounds[1]),
            collides: data.collision_matrix[current.layer as usize],
            place: k as u32,
        });
    }
}

/// A DetPhysics collider, by the collection that owns it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Part {
    Item(usize),
    Shot(usize),
    Node(usize),
    Engine(usize),
    Spike(usize),
    Missile(usize),
    /// A Rock form's circle, by player.
    Rock(usize),
    /// A DynamicAbilityPickup's circle.
    Pickup(usize),
    /// A Duplicator's Rock_noPlayer copy.
    DuplicatedRock(usize),
    /// A RopeHook's circle, by rope.
    Hook(usize),
    /// A Bow arrow's circle.
    Arrow(usize),
    /// A BlackHole's circle.
    Hole(usize),
    /// A slime's or a non-Rock form's Player-layer collider, by player.
    Body(usize),
    /// An abilityless clone's box.
    Clone(usize),
    /// A lit Tesla connection's StretchableLightning box, by parent node.
    Line(usize),
    /// A Smoke cloud's FlammableSmoke circle.
    Cloud(usize),
    /// An Explosion-layer collider but the Tesla lightning.
    Explosion(Blast),
    /// The scene's water DPhysicsBox, a layer-4 trigger.
    Water,
}

/// The Explosion-layer colliders, by the collection that owns them; the
/// Tesla lightning is `Part::Line`. The collision matrix pairs them with the
/// Player, Projectile, item, Smoke, waterMask and orb layers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Blast {
    /// GrenadeExplode's XGunExplosion box.
    Grenade(usize),
    /// A MineExplosion box: a Missile's, a Mine's or a bounds UpEffect's.
    Mine(usize),
    /// SmokeGrenadeExplode2's SmokeExplosion box.
    Smoke(usize),
    /// MeteorSmash's MeteorSmear_explo box.
    Smear(usize),
    /// One of a Drill's three invisibleHitbox boxes, by player and box.
    Drill(usize, usize),
    /// A pooled BeamObject box.
    Beam(usize),
    /// The Beam staff's orbAnim circle.
    Orb(usize),
}

/// Arrow.StickTo: Player, item, wall, LethalTerrain and NonLethalTerrain
/// (Arrow.prefab m_Bits 2166016).
const ARROW_STICK_TO: u32 = 2166016;

/// Walls `walls_meeting` takes at a time.
const WALL_GROUP: usize = 32;

/// Bit k is set when wall k of `walls` (at most `WALL_GROUP`) meets `part`
/// as `sweep::swept_meet` does: two walls per vector, each lane one strict
/// ordered compare (false on NaN, like the scalar `<`), and a wall meets
/// when its four lanes all hold. A high edge's `wall > low` is taken as
/// `-wall < -low`: negation is exact and reverses the order, signed zeros
/// and infinities included, and a NaN fails both. Nearly every map's walls
/// fit one call, so the part's edges stay in a register for them all; a
/// second call, after the first's walls' narrow tests, reloaded them from a
/// 512-bit spill whose store had not committed, which does not forward.
#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "bmi2", not(feature = "q32")))]
fn walls_meeting(walls: &[[Vec2; 2]], part: [Vec2; 2]) -> u64 {
    use std::arch::x86_64::*;
    const _: () = assert!(std::mem::size_of::<[Vec2; 2]>() == 4 * std::mem::size_of::<f64>());
    debug_assert!(walls.len() <= WALL_GROUP);
    let [low, high] = part.map(|v| [v.x.to_f64(), v.y.to_f64()]);
    let pairs = walls.len() / 2;
    // A nibble per wall, sixteen walls a word.
    let mut nibbles = [0u64; 2];
    // SAFETY: avx512f and bmi2 are compile-time target features here;
    // `[Vec2; 2]` is four f64 (Vec2 is repr(C) of two repr(transparent)
    // f64), every full load is two walls within `walls` and the last one's
    // mask keeps its lanes within the odd wall.
    unsafe {
        // A wall's lanes are low x, low y, high x, high y: its low edges
        // below the part's high ones, its negated high edges below the
        // part's negated low ones.
        const S: i64 = i64::MIN;
        let signs = _mm512_setr_epi64(0, 0, S, S, 0, 0, S, S);
        let edges = _mm512_setr_pd(high[0], high[1], -low[0], -low[1], high[0], high[1], -low[0], -low[1]);
        let flipped = |wall: __m512d| _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(wall), signs));
        let base = walls.as_ptr().cast::<f64>();
        for pair in 0..pairs.min(8) {
            let wall = flipped(_mm512_loadu_pd(base.add(8 * pair)));
            nibbles[0] |= u64::from(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(wall, edges)) << (8 * pair);
        }
        for pair in 8..pairs {
            let wall = flipped(_mm512_loadu_pd(base.add(8 * pair)));
            nibbles[1] |= u64::from(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(wall, edges)) << (8 * (pair - 8));
        }
        if walls.len() % 2 != 0 {
            let wall = flipped(_mm512_maskz_loadu_pd(0x0f, base.add(8 * pairs)));
            let odd = u64::from(_mm512_mask_cmp_pd_mask::<_CMP_LT_OQ>(0x0f, wall, edges)) << (8 * (pairs % 8));
            nibbles[pairs / 8] |= odd;
        }
        let meets = nibbles.map(|n| _pext_u64(n & (n >> 1) & (n >> 2) & (n >> 3), 0x1111_1111_1111_1111));
        let meets = meets[0] | (meets[1] << 16);
        debug_assert_eq!(meets, walls_meeting_scalar(walls, part));
        meets
    }
}
#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "bmi2", not(feature = "q32"))))]
fn walls_meeting(walls: &[[Vec2; 2]], part: [Vec2; 2]) -> u64 {
    walls_meeting_scalar(walls, part)
}
fn walls_meeting_scalar(walls: &[[Vec2; 2]], part: [Vec2; 2]) -> u64 {
    debug_assert!(walls.len() <= WALL_GROUP);
    walls
        .iter()
        .enumerate()
        .fold(0, |meets, (wall, &bounds)| meets | (u64::from(sweep::swept_meet(part, bounds)) << wall))
}

/// The slime a wall group's body or clone part is.
#[inline(always)]
fn wall_slime(part: Part) -> Slime {
    match part {
        Part::Body(index) => Slime::Player(index),
        Part::Clone(index) => Slime::Clone(index),
        _ => unreachable!("a slime part"),
    }
}

/// Whether a slime's or clone's collider meets a wall: whether its
/// Circle/RoundedRect or Box/RoundedRect manifold is Some.
pub(super) fn slime_wall_meets(shape: Collider, wall: RoundedRect) -> bool {
    match shape {
        Collider::Circle(circle) => crate::physics::contact::circle_rounded(circle, wall).is_some(),
        Collider::Box(inner) => crate::physics::contact::box_rounded_meets(inner, wall),
        Collider::RoundedRect(_) => unreachable!("a slime is a circle or a box"),
    }
}

impl RolloutWorld {
    /// Its collider now, after the corrections of earlier groups.
    pub(super) fn part_collider(&self, part: Part) -> Collider {
        match part {
            Part::Item(i) => Collider::Circle(Circle {
                center: self.thrown_items[i].position,
                radius: self.thrown_items[i].collider_radius,
            }),
            Part::Shot(i) => Collider::Circle(Circle {
                center: self.portal_shots[i].position,
                radius: self.portal_shots[i].collider_radius,
            }),
            Part::Node(i) => Collider::Circle(Circle {
                center: self.tesla_nodes[i].collider_center,
                radius: self.tesla_nodes[i].radius,
            }),
            Part::Engine(i) => Collider::Box(self.engines[i].rocket_box(&self.entity_inverse_extents)),
            Part::Spike(i) => Collider::Box(self.spike_box(i)),
            Part::Missile(i) => Collider::Box(self.guided_missiles[i].collider(&self.entity_inverse_extents)),
            Part::Rock(i) => Collider::Circle(Circle {
                center: self.players[i].base.position,
                radius: self.players[i].rock_circle().circle_radius,
            }),
            Part::Pickup(i) => {
                let pickup = &self.ability_pickups[i];
                Collider::Circle(Circle { center: pickup.synced_center, radius: pickup.radius })
            }
            Part::DuplicatedRock(i) => Collider::Circle(Circle {
                center: self.duplicated_rocks[i].position,
                radius: self.duplicated_rocks[i].collider_radius,
            }),
            Part::Hook(r) => Collider::Circle(self.hook_circle(r)),
            Part::Arrow(i) => Collider::Circle(Circle {
                center: self.arrows[i].position,
                radius: self.arrows[i].collider_radius,
            }),
            Part::Hole(i) => Collider::Circle(Circle {
                center: self.black_holes[i].position,
                radius: self.black_holes[i].radius,
            }),
            Part::Body(_) | Part::Clone(_) => self.body_row_collider(self.part_entity(part).expect("registered body")),
            Part::Line(i) => {
                let entity = self.tesla_nodes[i].line_entity.expect("a registered lightning");
                Collider::Box(Self::row_box(&self.entity_inverse_extents, entity, self.entities[entity]))
            }
            Part::Cloud(i) => {
                let cloud = &self.smoke_clouds[i];
                let entity = cloud.entity.expect("registered smoke cloud");
                Collider::Circle(Circle { center: cloud.position, radius: Fix::from_raw(self.entities[entity].radius) })
            }
            // SetDimensions resized the beam's box after SortAndSweep bounded
            // it, so the narrow phase tests the new box.
            Part::Explosion(Blast::Beam(i)) => Collider::Box(self.beam_pool[i].hitbox.collider.expect("a live beam box")),
            Part::Explosion(blast) => {
                let entity = self.blast_entity(blast);
                let row = self.entities[entity];
                if row.kind == 0 {
                    let center = Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y));
                    return Collider::Circle(Circle { center, radius: Fix::from_raw(row.radius) });
                }
                Collider::Box(Self::row_box(&self.entity_inverse_extents, entity, row))
            }
            Part::Water => Collider::Box(data::water_box(data::get().maps[self.map].water.expect("a map with water"))),
        }
    }

    /// An Explosion-layer collider's row.
    pub(super) fn blast_entity(&self, blast: Blast) -> usize {
        let entity = match blast {
            Blast::Grenade(i) => self.grenade_explosions[i].entity,
            Blast::Mine(i) => self.missile_explosions[i].entity,
            Blast::Smoke(i) => self.smoke_explosions[i].entity,
            Blast::Smear(i) => self.meteor_smears[i].entity,
            Blast::Drill(p, k) => {
                let slot = &self.players[p];
                Some(slot.drill_hitbox_entities[slot.active_slot][k]).filter(|&entity| entity != usize::MAX)
            }
            Blast::Beam(i) => self.beam_pool[i].entity,
            Blast::Orb(i) => self.beam_orbs[i].entity,
        };
        entity.expect("a registered Explosion-layer collider")
    }

    /// Whether an Explosion-layer collider is in this Simulate's pairs:
    /// registered, DetPhysics.IsActive and its hitbox enabled. An Explosion
    /// with hitBoxActiveOnlyFirstFrame enables its box only until its first
    /// UpdateSim (Explosion.cs:37-66); a MeteorSmear's, a BeamObject's and
    /// the orb's stay enabled; a Drill's HitboxCombo enables its boxes once
    /// it drills (Drill.cs:199, 283-289); StartBeam enables the orb's and
    /// StopOrbAnim disables it (Beam.cs:344, 395). A beam box pairs on the
    /// box SortAndSweep bounded before UpdateBeams resized it.
    fn blast_part(&self, blast: Blast) -> bool {
        let live = match blast {
            Blast::Grenade(i) => self.grenade_explosions[i].active && self.grenade_explosions[i].registered,
            Blast::Mine(i) => self.missile_explosions[i].active && self.missile_explosions[i].registered,
            Blast::Smoke(i) => self.smoke_explosions[i].active && self.smoke_explosions[i].registered,
            Blast::Smear(i) => self.meteor_smears[i].active,
            Blast::Drill(p, k) => {
                let slot = &self.players[p];
                slot.active == ActiveAbility::Drill
                    && slot.drill.phase != 0
                    && self.hittable(p)
                    && slot.drill_hitbox_entities[slot.active_slot][k] != usize::MAX
            }
            Blast::Beam(i) => {
                let slot = &self.beam_pool[i];
                i < self.scratch.beam_contacts && slot.swept_collider.is_some() && slot.hitbox.collider.is_some()
            }
            Blast::Orb(i) => self.beam_orbs[i].active && self.beam_orbs[i].enabled,
        };
        live && self.collides(self.blast_entity(blast))
    }

    /// Every Explosion-layer collider in this Simulate's pairs, circles or
    /// boxes.
    fn push_blasts(&self, parts: &mut Vec<(Part, u32)>, circles: bool, data: &data::RolloutData) {
        self.for_blasts(circles, |blast| parts.push(self.layered(Part::Explosion(blast), data)));
    }

    /// `push_blasts`, each blast to `each`.
    #[inline(always)]
    fn for_blasts(&self, circles: bool, mut each: impl FnMut(Blast)) {
        let mut push = |blast: Blast| {
            if self.blast_part(blast) {
                each(blast);
            }
        };
        // The orbs are the circles; the rest, boxes, in this order.
        if circles {
            (0..self.beam_orbs.len()).for_each(|i| push(Blast::Orb(i)));
            return;
        }
        (0..self.grenade_explosions.len()).for_each(|i| push(Blast::Grenade(i)));
        (0..self.missile_explosions.len()).for_each(|i| push(Blast::Mine(i)));
        (0..self.smoke_explosions.len()).for_each(|i| push(Blast::Smoke(i)));
        (0..self.meteor_smears.len()).for_each(|i| push(Blast::Smear(i)));
        for p in 0..self.players.len() {
            let slot = &self.players[p];
            // `blast_part` takes a Drill's boxes only while it drills.
            if slot.active == ActiveAbility::Drill && slot.drill.phase != 0 {
                (0..3).for_each(|k| push(Blast::Drill(p, k)));
            } else {
                debug_assert!((0..3).all(|k| !self.blast_part(Blast::Drill(p, k))));
            }
        }
        (0..self.beam_pool.len()).for_each(|i| push(Blast::Beam(i)));
    }

    /// FlammableSmoke.OnCollide's test: the collider carries an Explosion
    /// with igniteFlammables (XGunExplosion, MineExplosion, SmokeExplosion,
    /// BeamObject, orbAnim and Chain_Lightning prefabs; MeteorSmear_explo's
    /// is off, and a Drill's invisibleHitbox has none).
    fn ignites_flammables(other: Part) -> bool {
        match other {
            Part::Line(_) => true,
            Part::Explosion(blast) => !matches!(blast, Blast::Smear(_) | Blast::Drill(..)),
            _ => false,
        }
    }

    /// PlayerCollision.KillPlayerOnCollision's reading of an Explosion-layer
    /// collider: Explosion.PlayerOwnerId, canHurtOwner and the "beam" tag, or
    /// a Drill box's Hitbox.GetPlayerId with canHurtOwner taken as true
    /// (PlayerCollision.cs:350-375). A grenade's or missile's blast credits
    /// nobody unless its owner was credited; the orb's canHurtOwner is off.
    /// Chain_Lightning's Explosion has canHurtOwner on, and SimpleSparkNode
    /// spawns it without assigning its PlayerOwnerId (Chain_Lightning.prefab,
    /// SimpleSparkNode.cs:208-213).
    fn blast_owner(&self, other: Part) -> (u32, bool, bool) {
        let blast = match other {
            Part::Line(_) => return (0, true, false),
            Part::Explosion(blast) => blast,
            _ => unreachable!("an Explosion-layer part"),
        };
        match blast {
            Blast::Grenade(i) => (self.grenade_explosions[i].killer(&self.players), true, false),
            Blast::Mine(i) => {
                let explosion = &self.missile_explosions[i];
                (if explosion.credited_owner { player_id(&self.players, explosion.owner) } else { 0 }, true, false)
            }
            Blast::Smoke(i) => (player_id(&self.players, self.smoke_explosions[i].owner), true, false),
            Blast::Smear(i) => (player_id(&self.players, self.meteor_smears[i].owner), true, false),
            Blast::Drill(p, _) => (self.players[p].base.id, true, false),
            Blast::Beam(i) => (self.beam_pool[i].hitbox.owner_id as u32, true, true),
            Blast::Orb(i) => (player_id(&self.players, self.beam_orbs[i].owner), false, false),
        }
    }

    /// PlayerCollision.KillPlayerOnCollision past its squish test, for the
    /// victim's PlayerCollision and `other` met through its collider row
    /// `collider`: the (killer, cause) its killPlayer takes, or None where it
    /// returns first (PlayerCollision.cs:305-460), as on a collider an
    /// earlier callback destroyed (PlayerCollision.cs:313). A player's PlayerCollision
    /// skips its ignored colliders, and the Beam form's its own beam's
    /// BeamObjects (Beam.cs:213); an abilityless clone's ignores nothing.
    /// The killer is the Projectile's or Explosion's owner, a Drill box's
    /// Hitbox id or the holder of an Ability that KillPlayersOnContact; a
    /// slime's own body kills nobody. The owner survives what cannot hurt it
    /// and dies of an Accident by what can.
    pub(super) fn collision_kill(&self, victim: Slime, other: Part, collider: Option<usize>) -> Option<(u32, u32)> {
        if self.collider_destroyed(collider) {
            return None;
        }
        if let Slime::Player(index) = victim {
            let slot = &self.players[index];
            let hierarchy = collider.or_else(|| self.part_entity(other)).map_or(0, |entity| self.entities[entity].hierarchy);
            if slot.ignores(hierarchy) {
                return None;
            }
            if let Part::Explosion(Blast::Beam(i)) = other
                && slot.active == ActiveAbility::Beam
                && slot.beam_entity != usize::MAX
            {
                let own = self.entities[slot.beam_entity].hierarchy.wrapping_sub(100_000_000).wrapping_add(2) as i32;
                if self.beam_pool[i].hitbox.beam_id.wrapping_abs() == own.wrapping_abs() {
                    return None;
                }
            }
        }
        let (killer, can_hurt_owner, beam) = match other {
            Part::Arrow(i) if self.arrows[i].active => (self.arrows[i].owner, self.arrows[i].can_hurt_owner, false),
            // The Rock form's BounceBall always KillPlayersOnContact.
            Part::Rock(i) => (self.players[i].base.id, true, false),
            Part::Body(i) if collider.is_some_and(|entity| self.players[i].kills_on_contact(entity)) => {
                (self.players[i].base.id, true, false)
            }
            // A SpikeAttack's and a SimpleSparkNode's Explosion keep
            // PlayerOwnerId's zero default.
            Part::Spike(_) => (0, true, false),
            Part::Node(i) if self.tesla_nodes[i].lethal => (0, true, false),
            // ShootDuplicator gives Rock_noPlayer's Explosion the slime's id.
            Part::DuplicatedRock(i) => (player_id(&self.players, self.duplicated_rocks[i].owner), true, false),
            Part::Explosion(_) | Part::Line(_) => self.blast_owner(other),
            _ => return None,
        };
        let victim_id = self.slime_base(victim).id;
        if killer == victim_id && !can_hurt_owner {
            return None;
        }
        let cause = if killer == victim_id {
            CAUSE_ACCIDENT
        } else if beam {
            CAUSE_BEAMED
        } else {
            CAUSE_OTHER
        };
        Some((killer, cause))
    }

    /// The collider DetPhysics holds for a slime's, form's or clone's row,
    /// where it last synced it.
    pub(super) fn body_row_collider(&self, entity: usize) -> Collider {
        let row = self.entities[entity];
        let center = Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y));
        if row.kind == 0 {
            return Collider::Circle(Circle { center, radius: Fix::from_raw(row.radius) });
        }
        Collider::Box(Self::row_box(&self.entity_inverse_extents, entity, row))
    }

    /// The layer SortAndSweep checks against the collision matrix.
    fn part_layer(&self, part: Part, data: &data::RolloutData) -> u32 {
        match part {
            Part::Item(i) => self.thrown_items[i].entity.map_or(LAYER_ITEM, |entity| self.entities[entity].layer),
            Part::Shot(_) => data.portal_gun.shot_layer,
            // SimpleSparkNode moves a lethal node's circle to LethalTerrain.
            Part::Node(i) if self.tesla_nodes[i].lethal => LAYER_LETHAL_TERRAIN,
            Part::Node(_) => data.tesla.node_layer,
            Part::Engine(_) => data.engine.rocket_layer,
            Part::Spike(_) => data.spike.attack_layer,
            Part::Missile(_) => data.missile.spark_layer,
            Part::Rock(_) => LAYER_PLAYER,
            Part::Pickup(i) => self.ability_pickups[i].entity.map_or(LAYER_WEAPON, |entity| self.entities[entity].layer),
            // Rock_noPlayer is a LethalTerrain circle.
            Part::DuplicatedRock(i) => {
                self.duplicated_rocks[i].entity.map_or(LAYER_LETHAL_TERRAIN, |entity| self.entities[entity].layer)
            }
            Part::Hook(r) => self.ropes[r].hook_entity.map_or(LAYER_ROPE, |entity| self.entities[entity].layer),
            Part::Arrow(i) => self.arrows[i].entity.map_or(LAYER_PROJECTILE, |entity| self.entities[entity].layer),
            Part::Hole(_) => LAYER_RIGID_BODY_AFFECTOR,
            Part::Line(_) => data.tesla.lightning_layer,
            Part::Cloud(i) => self.entities[self.smoke_clouds[i].entity.expect("registered smoke cloud")].layer,
            Part::Explosion(blast) => self.entities[self.blast_entity(blast)].layer,
            Part::Water => LAYER_WATER,
            Part::Body(_) | Part::Clone(_) => self.entities[self.part_entity(part).expect("registered body")].layer,
        }
    }

    /// Its place in SortAndSweep's order.
    pub(super) fn part_rank(&self, part: Part) -> SweepRank {
        sweep_rank(&self.entities, self.swept_min(Swept::Part(part)), self.part_entity(part))
    }

    /// The part as the pair loops read it: `part_rank` on the pass's
    /// `sweep_x`, its `part_layer` `layer`, `part_composite` and its swept
    /// bounds.
    fn group_part(&self, part: Part, layer: u32, sweep_x: bool, data: &data::RolloutData) -> GroupPart {
        self.group_part_bounded(part, layer, self.swept_bounds(Swept::Part(part)), sweep_x, data)
    }

    /// `group_part` with the part's `swept_bounds`.
    #[inline(always)]
    fn group_part_bounded(
        &self,
        part: Part,
        layer: u32,
        bounds: Option<[Vec2; 2]>,
        sweep_x: bool,
        data: &data::RolloutData,
    ) -> GroupPart {
        debug_assert_eq!(layer, self.part_layer(part, data));
        debug_assert_eq!(sweep_x, self.sweep_x());
        #[cfg(debug_assertions)]
        {
            let bits = |b: Option<[Vec2; 2]>| b.map(|b| [b[0].x, b[0].y, b[1].x, b[1].y].map(|v| v.to_f64().to_bits()));
            debug_assert_eq!(bits(bounds), bits(self.swept_bounds(Swept::Part(part))));
        }
        let low = bounds.map_or(Vec2::ZERO, |b| b[0]);
        let min = if sweep_x { low.x } else { low.y };
        let rank = sweep_rank(&self.entities, min, self.part_entity_inlined(part));
        GroupPart {
            part,
            rank,
            layer,
            composite: self.part_composite(part),
            bounds,
            #[cfg(debug_assertions)]
            eager: rank,
        }
    }

    /// `group_part_bounded` without the rank (`UNRANKED`), for the
    /// Circle/Box group's parts: that group reads a rank only for a
    /// manifold, and the Box/Box group, which keeps its boxes, only to sort
    /// a group with two that pair, which over the test replays one pass in a
    /// hundred has; each takes it then (`group_rank`), from the part's
    /// bounds and row as they were here (debug builds check that).
    #[inline(always)]
    fn unranked_part(
        &self,
        part: Part,
        layer: u32,
        bounds: Option<[Vec2; 2]>,
        sweep_x: bool,
        data: &data::RolloutData,
    ) -> GroupPart {
        #[cfg(debug_assertions)]
        let eager = self.group_part_bounded(part, layer, bounds, sweep_x, data).rank;
        let _ = (sweep_x, data);
        GroupPart {
            part,
            rank: UNRANKED,
            layer,
            composite: self.part_composite(part),
            bounds,
            #[cfg(debug_assertions)]
            eager,
        }
    }

    /// `unranked_part` of a part with its `part_layer`, inlined where the
    /// part's kind is known, so its layer, box and composite come straight
    /// from its pool.
    #[inline(always)]
    fn known_unranked(&self, part: Part, sweep_x: bool, data: &data::RolloutData) -> GroupPart {
        let (part, layer) = self.layered(part, data);
        self.unranked_part(part, layer, self.swept_bounds_inlined(Swept::Part(part)), sweep_x, data)
    }

    /// `group_part`, unranked (`unranked_part`).
    fn group_unranked(&self, part: Part, layer: u32, sweep_x: bool, data: &data::RolloutData) -> GroupPart {
        self.unranked_part(part, layer, self.swept_bounds(Swept::Part(part)), sweep_x, data)
    }

    /// A group part's rank, ranked or not: `group_part_bounded`'s, from its
    /// bounds on the pass's `sweep_x` and its row.
    fn group_rank(&self, part: &GroupPart, sweep_x: bool) -> SweepRank {
        let low = part.bounds.map_or(Vec2::ZERO, |b| b[0]);
        let min = if sweep_x { low.x } else { low.y };
        let rank = sweep_rank(&self.entities, min, self.part_entity(part.part));
        #[cfg(debug_assertions)]
        debug_assert!(same_rank(rank, part.eager), "a group part's rank moved since it was taken");
        rank
    }

    /// The part with its `part_layer`, inlined where the part's kind is
    /// known.
    #[inline(always)]
    fn layered(&self, part: Part, data: &data::RolloutData) -> (Part, u32) {
        (part, self.part_layer(part, data))
    }

    /// A slime's or form's `body_part_circle` collider with its layer: that
    /// filter took only a Player-layer row.
    fn body_layered(&self, i: usize, data: &data::RolloutData) -> (Part, u32) {
        debug_assert_eq!(self.part_layer(Part::Body(i), data), LAYER_PLAYER);
        (Part::Body(i), LAYER_PLAYER)
    }

    /// A wall's place in SortAndSweep's order, from the collider the
    /// broadphase bounded.
    pub(super) fn wall_rank(&self, platform: usize) -> SweepRank {
        // `callback_bounds` holds the box of each `callback_colliders` wall.
        let min = match self.scratch.callback_bounds.get(platform).filter(|_| self.scratch.callback_tick == Some(self.tick)) {
            Some(bounds) => bounds[0],
            None => Collider::RoundedRect(self.circle_collider(platform)).bounds()[0],
        };
        let entity = Some(self.platform_entities[platform]);
        sweep_rank(&self.entities, if self.sweep_x() { min.x } else { min.y }, entity)
    }

    fn part_entity(&self, part: Part) -> Option<usize> {
        self.part_entity_inlined(part)
    }

    /// `part_entity`, inlined where the part's kind is known.
    #[inline(always)]
    fn part_entity_inlined(&self, part: Part) -> Option<usize> {
        match part {
            Part::Item(i) => self.thrown_items[i].entity,
            Part::Shot(i) => self.portal_shots[i].entity,
            Part::Node(i) => self.tesla_nodes[i].entity,
            Part::Engine(i) => self.engines[i].entity,
            Part::Spike(i) => self.spike_attacks[i].entity,
            Part::Missile(i) => self.guided_missiles[i].entity,
            Part::Rock(i) => Some(self.players[i].rock_entity),
            Part::Pickup(i) => self.ability_pickups[i].entity,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].entity,
            Part::Hook(r) => self.ropes[r].hook_entity,
            Part::Arrow(i) => self.arrows[i].entity,
            Part::Hole(i) => self.black_holes[i].entity,
            Part::Body(i) => Some(Self::hurtbox_entity(&self.players[i])),
            Part::Clone(i) => self.abilityless_clones[i].entity,
            Part::Line(i) => self.tesla_nodes[i].line_entity,
            Part::Cloud(i) => self.smoke_clouds[i].entity,
            Part::Explosion(blast) => Some(self.blast_entity(blast)),
            Part::Water => self.water_entity,
        }
    }

    /// The platform whose CompositeBody holds a child collider.
    #[inline(always)]
    fn part_composite(&self, part: Part) -> Option<usize> {
        match part {
            Part::Node(i) => Some(self.tesla_nodes[i].platform_index),
            Part::Engine(i) => Some(self.engines[i].platform_index),
            Part::Spike(i) => Some(self.spike_attacks[i].platform_index),
            Part::Item(_)
            | Part::Shot(_)
            | Part::Missile(_)
            | Part::Rock(_)
            | Part::Pickup(_)
            | Part::DuplicatedRock(_)
            | Part::Hook(_)
            | Part::Arrow(_)
            | Part::Hole(_)
            | Part::Body(_)
            | Part::Clone(_)
            | Part::Line(_)
            | Part::Cloud(_)
            | Part::Explosion(_)
            | Part::Water => None,
        }
    }

    /// Its own PhysicsBody, whose inverse mass PositionalCorrection reads
    /// even for a composite child.
    fn part_body(&self, part: Part) -> Body {
        match part {
            Part::Item(i) => self.thrown_items[i].body,
            Part::Shot(i) => self.portal_shots[i].body,
            Part::Node(i) => self.tesla_nodes[i].body,
            Part::Engine(i) => self.engines[i].body,
            Part::Spike(i) => self.spike_attacks[i].body,
            Part::Missile(i) => self.guided_missiles[i].body,
            Part::Rock(i) => self.players[i].rock.body,
            Part::Pickup(i) => self.ability_pickups[i].body,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].body,
            // A DPhysicsCircle with no BoplBody registers inverseMass -1, a
            // trigger that never resolves (DPhysicsCircle.cs:278-283).
            Part::Hook(_) => Body { inverse_mass: -Fix::ONE, ..Body::default() },
            Part::Arrow(i) => self.arrows[i].body(&data::get().arrow.body),
            Part::Hole(i) => Body { inverse_mass: -Fix::ONE, velocity: self.black_holes[i].body_velocity, ..Body::default() },
            // A DPhysicsBox with no BoplBody registers inverseMass -1
            // (DPhysicsBox.cs:339-345), and a form's circle likewise.
            Part::Body(_) | Part::Clone(_) => Body { inverse_mass: -Fix::ONE, ..Body::default() },
            // The lightning's box is a trigger.
            Part::Line(_) => Body { inverse_mass: -Fix::ONE, ..Body::default() },
            Part::Cloud(i) => self.smoke_clouds[i].body,
            // An Explosion's box has no BoplBody.
            Part::Explosion(_) => Body { inverse_mass: -Fix::ONE, ..Body::default() },
            Part::Water => Body { inverse_mass: -Fix::ONE, ..Body::default() },
        }
    }

    pub(super) fn part_destroyed(&self, part: Part) -> bool {
        !match part {
            Part::Item(i) => self.thrown_items[i].active,
            Part::Shot(i) => self.portal_shots[i].active,
            Part::Node(i) => self.tesla_nodes[i].active,
            Part::Engine(i) => self.engines[i].active,
            Part::Spike(i) => self.spike_attacks[i].active,
            Part::Missile(i) => self.guided_missiles[i].active,
            Part::Rock(i) => self.players[i].active == ActiveAbility::Rock,
            Part::Pickup(i) => self.ability_pickups[i].active,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].active,
            Part::Hook(r) => self.hook_live(r),
            Part::Arrow(i) => self.arrows[i].active,
            Part::Hole(i) => self.black_holes[i].active,
            Part::Body(i) => self.hittable(i),
            Part::Clone(i) => self.abilityless_clones[i].active,
            Part::Line(i) => self.tesla_nodes[i].line_entity.is_some(),
            Part::Cloud(i) => self.smoke_clouds[i].active,
            Part::Explosion(blast) => self.blast_part(blast),
            Part::Water => data::get().maps[self.map].water.is_some(),
        }
    }

    /// The body and position ResolveCollision works on: a composite child's
    /// CompositeBody, at CompositeBody.position (PhysTools.cs:1006).
    fn part_resolve_body(&self, part: Part) -> (Body, Vec2) {
        match (part, self.part_composite(part)) {
            (_, Some(p)) => (self.platform_dynamics[p].body, self.platform_dynamics[p].composite_position),
            (Part::Item(i), None) => (self.thrown_items[i].body, self.thrown_items[i].position),
            (Part::Shot(i), None) => (self.portal_shots[i].body, self.portal_shots[i].position),
            (Part::Missile(i), None) => (self.guided_missiles[i].body, self.guided_missiles[i].position),
            (Part::Rock(i), None) => (self.players[i].rock.body, self.players[i].base.position),
            (Part::Pickup(i), None) => {
                let pickup = &self.ability_pickups[i];
                (pickup.body, pickup.synced_center)
            }
            (Part::DuplicatedRock(i), None) => (self.duplicated_rocks[i].body, self.duplicated_rocks[i].position),
            (Part::Hook(r), None) => (self.part_body(part), self.ropes[r].hook_position),
            (Part::Arrow(i), None) => (self.part_body(part), self.arrows[i].position),
            (Part::Hole(i), None) => (self.part_body(part), self.black_holes[i].position),
            (Part::Cloud(i), None) => (self.smoke_clouds[i].body, self.smoke_clouds[i].position),
            _ => unreachable!(),
        }
    }

    /// Writes back a resolved body and moves the collider by its correction,
    /// and a child's CompositeBody.position with it (DetPhysics.cs:3114-3127).
    fn set_part_body(&mut self, part: Part, body: Body, delta: Vec2) {
        if let Some(p) = self.part_composite(part) {
            self.platform_dynamics[p].body = body;
            self.platform_dynamics[p].composite_position += delta;
        }
        match part {
            Part::Item(i) => {
                self.thrown_items[i].body = body;
                self.thrown_items[i].position += delta;
            }
            Part::Shot(i) => {
                self.portal_shots[i].body = body;
                self.portal_shots[i].position += delta;
            }
            Part::Node(i) => self.tesla_nodes[i].collider_center += delta,
            Part::Engine(i) => self.engines[i].collider_center += delta,
            Part::Spike(i) => self.spike_attacks[i].collider_center += delta,
            Part::Missile(i) => {
                self.guided_missiles[i].body = body;
                self.guided_missiles[i].position += delta;
            }
            // The Rock's row is its DetPhysics circle, which the correction
            // moves along with the FixTransform it is synced back to.
            Part::Rock(i) => {
                let slot = &mut self.players[i];
                slot.rock.body = body;
                slot.base.position += delta;
                slot.base.velocity = body.velocity;
                let (entity, position) = (slot.rock_entity, slot.base.position);
                if let Some(row) = self.entities.get_mut(entity) {
                    row.x = position.x.raw();
                    row.y = position.y.raw();
                }
            }
            Part::Pickup(i) => {
                let pickup = &mut self.ability_pickups[i];
                pickup.body = body;
                pickup.synced_center += delta;
                pickup.sync_fix_trans_to_collider();
                let center = pickup.synced_center;
                if let Some(entity) = pickup.entity {
                    let row = &mut self.entities[entity];
                    row.x = center.x.raw();
                    row.y = center.y.raw();
                    row.vx = body.velocity.x.raw();
                    row.vy = body.velocity.y.raw();
                }
            }
            Part::DuplicatedRock(i) => {
                self.duplicated_rocks[i].body = body;
                self.duplicated_rocks[i].position += delta;
                self.sync_duplicated_rock_entity(i);
            }
            Part::Hook(_) => {}
            Part::Arrow(i) => self.set_arrow_body(i, body, delta),
            Part::Hole(i) => self.black_holes[i].body_velocity = body.velocity,
            Part::Cloud(i) => {
                self.smoke_clouds[i].rest.forget();
                self.smoke_clouds[i].body = body;
                self.smoke_clouds[i].position += delta;
            }
            Part::Body(_) | Part::Clone(_) | Part::Line(_) | Part::Explosion(_) | Part::Water => {}
        }
        if let (Part::Item(_) | Part::Shot(_) | Part::Missile(_) | Part::Arrow(_), Some(entity)) =
            (part, self.part_entity(part))
        {
            let (body, position) = self.part_resolve_body(part);
            let row = &mut self.entities[entity];
            row.x = position.x.raw();
            row.y = position.y.raw();
            row.vx = body.velocity.x.raw();
            row.vy = body.velocity.y.raw();
            row.rotation = body.orientation.raw();
            row.angular_velocity = body.angular_velocity.raw();
        }
    }

    /// Its FixTransform.position: a child's collider less its rotated offset.
    fn part_root(&self, part: Part) -> Vec2 {
        match part {
            Part::Item(i) => self.thrown_items[i].position,
            Part::Shot(i) => self.portal_shots[i].position,
            Part::Node(i) => {
                let node = self.tesla_nodes[i];
                node.collider_center - node.collider_offset.complex_mul(Vec2::angle(node.collider_rotation))
            }
            Part::Engine(i) => {
                let engine = self.engines[i];
                engine.collider_center - engine.collider_offset.complex_mul(Vec2::angle(engine.collider_rotation))
            }
            Part::Spike(i) => self.spike_attack_root(i),
            Part::Missile(i) => self.guided_missiles[i].position,
            Part::Rock(i) => self.players[i].base.position,
            Part::Pickup(i) => self.ability_pickups[i].position,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].position,
            Part::Hook(r) => self.ropes[r].hook_position,
            Part::Arrow(i) => self.arrows[i].position,
            Part::Hole(i) => self.black_holes[i].position,
            Part::Body(_) | Part::Clone(_) | Part::Line(_) => self.part_collider(part).center(),
            Part::Cloud(i) => self.smoke_clouds[i].position,
            Part::Explosion(_) | Part::Water => self.part_collider(part).center(),
        }
    }

    fn set_arrow_body(&mut self, i: usize, body: Body, delta: Vec2) {
        let arrow = &mut self.arrows[i];
        arrow.position += delta;
        arrow.velocity = body.velocity;
        arrow.rotation = body.orientation;
        arrow.angular_velocity = body.angular_velocity;
    }

    /// DPhysicsCircle/DPhysicsBox.AddForceAtPosition: the impulse at `point`
    /// with its arm from the FixTransform, on a child's CompositeBody
    /// (DPhysicsCircle.cs:377-392, DPhysicsBox.cs:436-451).
    pub(super) fn part_add_force_at(&mut self, part: Part, force: Vec2, point: Vec2) {
        // A slime's, form's or clone's box and a grapple hook's circle have no
        // BoplBody: the impulse goes to the DetPhysics body its row publishes.
        if let (Part::Body(_) | Part::Clone(_) | Part::Hook(_), Some(entity)) = (part, self.part_entity(part)) {
            let row = &mut self.entities[entity];
            let velocity = Vec2::new(Fix::from_raw(row.vx), Fix::from_raw(row.vy)) + force * Fix::from_raw(row.inverse_mass);
            row.vx = velocity.x.raw();
            row.vy = velocity.y.raw();
            return;
        }
        let arm = (point - self.part_root(part)).normalized_safe();
        if let Part::Arrow(i) = part {
            let mut body = self.part_body(part);
            body.apply_impulse(force, arm);
            self.set_arrow_body(i, body, Vec2::ZERO);
            return;
        }
        if let Part::Hole(i) = part {
            let mut body = self.part_body(part);
            body.apply_impulse(force, arm);
            self.black_holes[i].body_velocity = body.velocity;
            if let Some(entity) = self.black_holes[i].entity {
                self.entities[entity].vx = body.velocity.x.raw();
                self.entities[entity].vy = body.velocity.y.raw();
            }
            return;
        }
        let body = match (part, self.part_composite(part)) {
            (_, Some(p)) => &mut self.platform_dynamics[p].body,
            (Part::Item(i), None) => &mut self.thrown_items[i].body,
            (Part::Shot(i), None) => &mut self.portal_shots[i].body,
            (Part::Missile(i), None) => &mut self.guided_missiles[i].body,
            (Part::Rock(i), None) => &mut self.players[i].rock.body,
            (Part::Cloud(i), None) => {
                self.smoke_clouds[i].rest.forget();
                &mut self.smoke_clouds[i].body
            }
            _ => unreachable!(),
        };
        body.apply_impulse(force, arm);
        let body = *body;
        if let Part::Rock(i) = part {
            self.players[i].base.velocity = body.velocity;
        }
        if let (Part::Item(_) | Part::Shot(_) | Part::Missile(_), Some(entity)) = (part, self.part_entity(part)) {
            let row = &mut self.entities[entity];
            row.vx = body.velocity.x.raw();
            row.vy = body.velocity.y.raw();
            row.angular_velocity = body.angular_velocity.raw();
        }
    }

    /// The live part whose GameObject an entity row is.
    pub(super) fn part_of_entity(&self, entity: usize) -> Option<Part> {
        (0..self.thrown_items.len())
            .map(Part::Item)
            .chain((0..self.portal_shots.len()).map(Part::Shot))
            .chain((0..self.tesla_nodes.len()).map(Part::Node))
            .chain((0..self.engines.len()).map(Part::Engine))
            .chain((0..self.spike_attacks.len()).map(Part::Spike))
            .chain((0..self.guided_missiles.len()).map(Part::Missile))
            .chain((0..self.players.len()).map(Part::Rock))
            .chain((0..self.ability_pickups.len()).map(Part::Pickup))
            .chain((0..self.duplicated_rocks.len()).map(Part::DuplicatedRock))
            .chain((0..self.ropes.len()).map(Part::Hook))
            .chain((0..self.arrows.len()).map(Part::Arrow))
            .find(|&part| !self.part_destroyed(part) && self.part_entity(part) == Some(entity))
    }

    /// DPhysicsCircle/DPhysicsBox.AddForce: a composite child's force goes to
    /// its CompositeBody, any other to its own PhysicsBody
    /// (DPhysicsCircle.cs:405-416, DPhysicsBox.cs:474-485).
    pub(super) fn part_add_force(&mut self, part: Part, force: Vec2) {
        if let Some(p) = self.part_composite(part) {
            self.platform_dynamics[p].body.force += force;
            self.platforms_changed();
            return;
        }
        match part {
            Part::Item(i) => self.thrown_items[i].body.force += force,
            Part::Shot(i) => self.portal_shots[i].body.force += force,
            Part::Missile(i) => self.guided_missiles[i].body.force += force,
            Part::Rock(i) => self.players[i].rock.body.force += force,
            Part::Pickup(i) => self.ability_pickups[i].body.force += force,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].body.force += force,
            Part::Arrow(i) => self.arrows[i].force += force,
            Part::Cloud(i) => {
                self.smoke_clouds[i].rest.forget();
                self.smoke_clouds[i].body.force += force;
            }
            // A RopeHook's circle has no BoplBody, so nothing integrates it.
            Part::Hook(_) => {}
            Part::Node(_)
            | Part::Engine(_)
            | Part::Spike(_)
            | Part::Hole(_)
            | Part::Body(_)
            | Part::Clone(_)
            | Part::Line(_)
            | Part::Explosion(_)
            | Part::Water => unreachable!(),
        }
    }

    /// A part in this Simulate's pairs: DetPhysics.IsActive on the live
    /// hierarchy, so a wall a QuantumTunnel showed again this pass brings its
    /// children back before their rows are republished, and an item a Blink
    /// tunnel holds stays out.
    fn child_registered(&self, entity: Option<usize>) -> bool {
        entity.is_some_and(|entity| self.collides(entity))
    }

    /// Whether a slime's or non-Rock form's Player-layer collider is in this
    /// Simulate's pairs, and whether it is a circle.
    pub(super) fn body_part_circle(&self, i: usize) -> Option<bool> {
        let slot = &self.players[i];
        if slot.active == ActiveAbility::Rock || !self.hittable(i) {
            return None;
        }
        let entity = Self::hurtbox_entity(slot);
        let row = self.entities.get(entity)?;
        if row.layer != LAYER_PLAYER {
            return None;
        }
        // `collides` would find this slot's hurtbox only after scanning every
        // other collider pool; the row is a circle or box, so it is
        // `hurtbox_collider`'s answer, which for a hittable slime that is no
        // Rock and has its row reads only these flags.
        let collides = match row.kind {
            0 | 1 if slot.active == ActiveAbility::None => !slot.base.dash.active,
            0 | 1 => slot.form_hurtbox_enabled(),
            _ => self.collides(entity),
        };
        debug_assert!(row.kind > 1 || collides == self.hurtbox_collider(i).is_some());
        debug_assert_eq!(collides, self.collides(entity));
        collides.then_some(row.kind == 0)
    }

    /// The players whose `body_part_circle` collider is a circle, and those
    /// whose is a box, by bit.
    fn body_part_bits(&self) -> (u32, u32) {
        debug_assert!(self.players.len() <= 32);
        let (mut circles, mut boxes) = (0u32, 0u32);
        for i in 0..self.players.len() {
            match self.body_part_circle(i) {
                Some(true) => circles |= 1 << i,
                Some(false) => boxes |= 1 << i,
                None => {}
            }
        }
        (circles, boxes)
    }

    /// `body_part_bits` for the Box/RoundedRect group, which opens a pass's
    /// groups, kept for the Circle/RoundedRect and Circle/Circle groups:
    /// nothing between them enables, disables or moves a slime's or form's
    /// collider into another pool, and the callbacks that do wait for the
    /// queue (`queue_collide`).
    pub(super) fn fresh_body_parts(&mut self) -> (u32, u32) {
        let bits = self.body_part_bits();
        self.scratch.body_parts = Some(bits);
        bits
    }

    /// The pass's `body_part_bits`, as the Box/RoundedRect group took them
    /// (`fresh_body_parts`), or taken now on a pass without that group.
    pub(super) fn pass_body_parts(&mut self) -> (u32, u32) {
        match self.scratch.body_parts {
            Some(bits) => {
                debug_assert_eq!(bits, self.body_part_bits());
                bits
            }
            None => self.fresh_body_parts(),
        }
    }

    /// Whether a Tesla node's circle is a part of the Circle/Circle group, or
    /// of the Circle/Box group. A node whose platform went inactive earlier
    /// in this pass is skipped by DetPhysics' Circle/Circle group, though its
    /// row still shows the last publish.
    pub(super) fn node_part(&self, i: usize, circle_group: bool) -> bool {
        let node = &self.tesla_nodes[i];
        self.child_registered(node.entity)
            && (!circle_group
                || (node.active
                    && !node.macho_carried
                    && node.platform_index < self.platforms.len()
                    && node.body.inverse_mass >= Fix::ZERO))
    }

    /// Whether a node's lit, enabled lightning is a part of the Circle/Box
    /// and Box/Box groups. RingBuffer.Add DestroyFixes an evicted node
    /// (RingBuffer.cs:60), but its lightning goes only with the deferred
    /// OnDestroyUpdatable (SimpleSparkNode.cs:140-157), so it still meets its
    /// pairs this pass.
    pub(super) fn line_part(&self, i: usize) -> bool {
        let node = &self.tesla_nodes[i];
        let present = if node.active {
            !node.macho_carried && !self.attached_hidden(node.platform_index, node.entity)
        } else {
            fix_destroyed(&self.entities, node.entity)
        };
        present && node.lethal && node.line_entity.is_some_and(|entity| row_is_active(&self.entities[entity]))
    }

    /// A Smoke cloud's circle's layer, if it is in this Simulate's pairs:
    /// Ignite disables it (FlammableSmoke.cs:85-89).
    fn cloud_layer(&self, i: usize, data: &data::RolloutData) -> Option<u32> {
        let cloud = &self.smoke_clouds[i];
        let entity = cloud.entity.filter(|_| cloud.active && cloud.registered)?;
        let layer = self.cloud_collides(entity).then(|| self.entities[entity].layer);
        debug_assert!(layer.is_none_or(|layer| layer == self.part_layer(Part::Cloud(i), data)));
        layer
    }

    /// How many Smoke clouds are parts (`cloud_layer`), counted up to one:
    /// all a `LayerSet` reads of them, since clouds never pair with each
    /// other, so a second cloud's layer meets nothing the first's does not.
    /// Every cloud's row is on the cloud layer, so the clouds themselves are
    /// taken only for a group they can pair in (`insert_cloud_parts`); over
    /// the test replays one pass in eight with a cloud has any partner for
    /// one.
    fn cloud_parts_present(&self, data: &data::RolloutData) -> u32 {
        debug_assert!(!data.layers_collide(data.smoke.cloud_layer, data.smoke.cloud_layer));
        let count = (0..self.smoke_clouds.len()).any(|i| self.cloud_layer(i, data).is_some()) as u32;
        #[cfg(debug_assertions)]
        {
            let every = (0..self.smoke_clouds.len()).filter(|&i| self.cloud_layer(i, data).is_some()).count();
            debug_assert_eq!(count as usize, every.min(1));
        }
        count
    }

    /// The boxes of `partners` a Smoke cloud can pair with, for
    /// `insert_cloud_parts`: up to eight, or a count of `usize::MAX` past
    /// eight.
    fn cloud_partners(partners: &[GroupPart], data: &data::RolloutData) -> ([[Vec2; 2]; 8], usize) {
        let layer = data.smoke.cloud_layer;
        let mut bounds = [[Vec2::ZERO; 2]; 8];
        let mut count = 0;
        for partner in partners {
            if let Some(partner_bounds) = partner.bounds.filter(|_| data.layers_collide(layer, partner.layer)) {
                if count == bounds.len() {
                    count = usize::MAX;
                    break;
                }
                bounds[count] = partner_bounds;
                count += 1;
            }
        }
        (bounds, count)
    }

    /// Every Smoke cloud that is a part and can pair with one of the
    /// partners (`cloud_partners`), in the order the groups take them, as a
    /// pair loop reads it, put in at `place`; how many. Clouds never pair
    /// with each other, so a cloud whose bounds meet no partner's on a layer
    /// it collides with makes no pair, and leaving it out leaves every other
    /// pair as it was (`LayerSet`). Past eight such partners every cloud is
    /// taken.
    fn insert_cloud_parts(
        &self,
        group: &mut Vec<GroupPart>,
        place: usize,
        (bounds, count): ([[Vec2; 2]; 8], usize),
        sweep_x: bool,
        data: &data::RolloutData,
    ) -> usize {
        let layer = data.smoke.cloud_layer;
        debug_assert!(!data.layers_collide(layer, layer));
        let end = group.len();
        for i in 0..self.smoke_clouds.len() {
            if let Some(cloud_layer) = self.cloud_layer(i, data) {
                debug_assert_eq!(cloud_layer, layer);
                let part = Part::Cloud(i);
                let cloud_bounds = self.swept_bounds(Swept::Part(part));
                let meets = |cloud_bounds: [Vec2; 2]| {
                    count == usize::MAX || bounds[..count].iter().any(|&b| sweep::swept_meet(cloud_bounds, b))
                };
                if cloud_bounds.is_some_and(meets) {
                    group.push(self.group_part_bounded(part, layer, cloud_bounds, sweep_x, data));
                }
            }
        }
        let taken = group.len() - end;
        group[place..].rotate_right(taken);
        taken
    }

    /// Every cloud that is a part, with its layer, for debug
    /// assertions against the groups' every part.
    #[cfg(debug_assertions)]
    fn cloud_layered(&self, data: &data::RolloutData) -> Vec<(Part, u32)> {
        (0..self.smoke_clouds.len()).filter_map(|i| self.cloud_layer(i, data).map(|layer| (Part::Cloud(i), layer))).collect()
    }

    /// `collides` for a Smoke cloud's circle row. No node, rocket, spike,
    /// clone, pickup, arrow, item, lightning or hurtbox owns the row, so
    /// `detphysics_collider` would answer with `row_collider` after scanning
    /// every one of those pools; this asks it directly.
    fn cloud_collides(&self, entity: usize) -> bool {
        let collides = match self.entities.get(entity) {
            Some(row) if row.kind == 0 => {
                row_is_active(row) && row.flags & (E_DESTROYED | E_REMOVED) == 0 && !self.blink_object_hidden(Some(entity))
            }
            _ => self.collides(entity),
        };
        debug_assert_eq!(collides, self.collides(entity));
        collides
    }

    /// Whether an abilityless clone's collider is in this Simulate's pairs,
    /// and whether it is a circle.
    pub(super) fn clone_part_circle(&self, i: usize) -> Option<bool> {
        let clone = &self.abilityless_clones[i];
        let entity = clone.entity.filter(|_| clone.active)?;
        self.collides(entity).then_some(self.entities[entity].kind == 0)
    }

    /// The Circle/Box group's circles, in its order, but for the Smoke
    /// clouds: where in it they go, and how many there are up to one
    /// (`cloud_parts_present`).
    fn push_circle_parts(&self, parts: &mut Vec<(Part, u32)>, data: &data::RolloutData) -> (usize, u32) {
        for (i, item) in self.thrown_items.iter().enumerate() {
            if item.active
                && self.child_registered(item.entity)
                && item.callback_registered_tick != self.tick
            {
                parts.push(self.layered(Part::Item(i), data));
            }
        }
        for (i, shot) in self.portal_shots.iter().enumerate() {
            if shot.active && shot.registered && !self.blink_object_hidden(shot.entity) {
                parts.push(self.layered(Part::Shot(i), data));
            }
        }
        for i in 0..self.tesla_nodes.len() {
            if self.node_part(i, false) {
                parts.push(self.layered(Part::Node(i), data));
            }
        }
        for i in 0..self.players.len() {
            if self.players[i].active == ActiveAbility::Rock && self.hittable(i) {
                parts.push(self.layered(Part::Rock(i), data));
            }
        }
        for i in 0..self.arrows.len() {
            if self.arrow_registered(i) {
                parts.push(self.layered(Part::Arrow(i), data));
            }
        }
        for (i, hole) in self.black_holes.iter().enumerate() {
            if hole.active && !self.blink_object_hidden(hole.entity) {
                parts.push(self.layered(Part::Hole(i), data));
            }
        }
        let clouds = (parts.len(), self.cloud_parts_present(data));
        for (i, rock) in self.duplicated_rocks.iter().enumerate() {
            if rock.active && !self.blink_object_hidden(rock.entity) {
                parts.push(self.layered(Part::DuplicatedRock(i), data));
            }
        }
        for (i, pickup) in self.ability_pickups.iter().enumerate() {
            if pickup.active && pickup.enabled && !self.blink_object_hidden(pickup.entity) {
                parts.push(self.layered(Part::Pickup(i), data));
            }
        }
        for rope in 0..self.ropes.len() {
            if self.hook_live(rope) {
                parts.push(self.layered(Part::Hook(rope), data));
            }
        }
        for i in 0..self.players.len() {
            if self.body_part_circle(i) == Some(true) {
                parts.push(self.body_layered(i, data));
            }
        }
        for i in 0..self.abilityless_clones.len() {
            if self.clone_part_circle(i) == Some(true) {
                parts.push(self.layered(Part::Clone(i), data));
            }
        }
        self.push_blasts(parts, true, data);
        clouds
    }

    /// The Circle/Box group's boxes, which are the Box/Box group's parts:
    /// rocket engine and spike boxes, Guided Missiles, lit lightning, slime,
    /// form and clone boxes, the Explosion boxes and the water.
    /// `box_bodies`, the slimes whose box is a part, by bit, when the
    /// Circle/Circle group just took them.
    fn push_box_parts(&self, parts: &mut Vec<(Part, u32)>, box_bodies: Option<u32>, data: &data::RolloutData) {
        for (i, engine) in self.engines.iter().enumerate() {
            if self.child_registered(engine.entity) {
                parts.push(self.layered(Part::Engine(i), data));
            }
        }
        for (i, spike) in self.spike_attacks.iter().enumerate() {
            if self.child_registered(spike.entity) {
                parts.push(self.layered(Part::Spike(i), data));
            }
        }
        for i in 0..self.guided_missiles.len() {
            if self.guided_missile_live(i) {
                parts.push(self.layered(Part::Missile(i), data));
            }
        }
        for i in 0..self.tesla_nodes.len() {
            if self.line_part(i) {
                parts.push(self.layered(Part::Line(i), data));
            }
        }
        for i in 0..self.players.len() {
            let body_box = match box_bodies {
                Some(bodies) => bodies >> i & 1 != 0,
                None => self.body_part_circle(i) == Some(false),
            };
            debug_assert_eq!(body_box, self.body_part_circle(i) == Some(false));
            if body_box {
                parts.push(self.body_layered(i, data));
            }
        }
        for i in 0..self.abilityless_clones.len() {
            if self.clone_part_circle(i) == Some(false) {
                parts.push(self.layered(Part::Clone(i), data));
            }
        }
        self.push_blasts(parts, false, data);
        if !self.part_destroyed(Part::Water) {
            parts.push(self.layered(Part::Water, data));
        }
    }

    /// `push_box_parts` as the pair loops read them (`group_part`), each
    /// built where its kind is known; the water's box from its map's
    /// `water_bounds`. Their layers' `LayerSet`.
    fn push_box_group_parts(
        &self,
        parts: &mut Vec<GroupPart>,
        box_bodies: Option<u32>,
        sweep_x: bool,
        data: &data::RolloutData,
    ) -> LayerSet {
        #[cfg(debug_assertions)]
        let start = parts.len();
        let mut set = LayerSet::default();
        let mut push = |parts: &mut Vec<GroupPart>, part: GroupPart| {
            set.add(part.layer);
            parts.push(part);
        };
        let swept_players = self.swept_players();
        for (i, engine) in self.engines.iter().enumerate() {
            if self.child_registered(engine.entity) {
                push(parts, self.known_unranked(Part::Engine(i), sweep_x, data));
            }
        }
        for (i, spike) in self.spike_attacks.iter().enumerate() {
            if self.child_registered(spike.entity) {
                push(parts, self.known_unranked(Part::Spike(i), sweep_x, data));
            }
        }
        for i in 0..self.guided_missiles.len() {
            if self.guided_missile_live(i) {
                push(parts, self.known_unranked(Part::Missile(i), sweep_x, data));
            }
        }
        for i in 0..self.tesla_nodes.len() {
            if self.line_part(i) {
                push(parts, self.known_unranked(Part::Line(i), sweep_x, data));
            }
        }
        let body_box = |i: usize| {
            let body_box = match box_bodies {
                Some(bodies) => bodies >> i & 1 != 0,
                None => self.body_part_circle(i) == Some(false),
            };
            debug_assert_eq!(body_box, self.body_part_circle(i) == Some(false));
            body_box
        };
        // A loop per source of the bounds: in one loop the part went to
        // `group_part` by memory, and the swept arm's push read the part's
        // tag and index stored apart as one, as it did a bounds both arms
        // wrote; no store forwards either.
        match swept_players {
            Some(players) => {
                for i in 0..self.players.len() {
                    if body_box(i) {
                        let (part, layer) = self.body_layered(i, data);
                        push(parts, self.unranked_part(part, layer, players.get(i).copied(), sweep_x, data));
                    }
                }
            }
            None => {
                for i in 0..self.players.len() {
                    if body_box(i) {
                        let (part, layer) = self.body_layered(i, data);
                        push(parts, self.group_unranked(part, layer, sweep_x, data));
                    }
                }
            }
        }
        for i in 0..self.abilityless_clones.len() {
            if self.clone_part_circle(i) == Some(false) {
                push(parts, self.known_unranked(Part::Clone(i), sweep_x, data));
            }
        }
        self.for_blasts(false, |blast| {
            let part = Part::Explosion(blast);
            push(parts, self.group_unranked(part, self.part_layer(part, data), sweep_x, data));
        });
        // The water is a part on a map with water, and its box the map's.
        let water = data.maps[self.map].water_bounds;
        debug_assert_eq!(water.is_some(), !self.part_destroyed(Part::Water));
        if let Some(bounds) = water {
            push(parts, self.unranked_part(Part::Water, LAYER_WATER, Some(bounds), sweep_x, data));
        }
        #[cfg(debug_assertions)]
        {
            let mut layered = Vec::new();
            self.push_box_parts(&mut layered, None, data);
            debug_assert_eq!(layered.len(), parts.len() - start);
            for (&(part, layer), taken) in layered.iter().zip(&parts[start..]) {
                debug_assert!(self.group_part(part, layer, sweep_x, data).same(taken));
            }
            let mut every = LayerSet::default();
            layered.iter().for_each(|&(_, layer)| every.add(layer));
            debug_assert_eq!((every.present, every.twice), (set.present, set.twice));
        }
        set
    }

    /// DetPhysics' Circle/Box group. Rigid pairs resolve in sweep order with
    /// PositionalCorrection on each side's own inverse mass; every pair
    /// queues both sides' OnCollide, which run after the last group.
    /// Whether the Box/Box group that follows has two parts that pair
    /// (`any_pairs`), or more than a few to sort.
    pub(super) fn circle_box_group(&mut self, data: &data::RolloutData) -> bool {
        let sweep_x = self.sweep_x();
        // The circles, each with its layer.
        let mut layered = std::mem::take(&mut self.scratch.cb_parts);
        layered.clear();
        let cached = self.scratch.circle_parts_tick.take() == Some(self.tick);
        // The clouds' place among the circles, and how many, up to one.
        let clouds;
        if cached {
            // The Circle/Circle group just took each of these kinds' parts,
            // and which slimes' boxes are parts: the same filters over flags
            // nothing since has changed, and rows' layers none of its
            // corrections move. Its nodes are a
            // subset of these (`node_part`), so they, and the holes and
            // hooks, whose boxes are live, are taken afresh.
            let taken = std::mem::take(&mut self.scratch.circle_parts);
            let ends = self.scratch.circle_ends;
            let blocks = |first: usize, last: usize| {
                let start = if first == 0 { 0 } else { ends[first - 1] as usize };
                &taken[start..ends[last] as usize]
            };
            let copy = |layered: &mut Vec<(Part, u32)>, first: usize, last: usize| {
                let parts = blocks(first, last);
                if !parts.is_empty() {
                    layered.extend_from_slice(parts);
                }
            };
            copy(&mut layered, CC_ITEMS, CC_ITEMS);
            copy(&mut layered, CC_SHOTS, CC_SHOTS);
            for i in 0..self.tesla_nodes.len() {
                if self.node_part(i, false) {
                    layered.push(self.layered(Part::Node(i), data));
                }
            }
            copy(&mut layered, CC_ROCKS, CC_ROCKS);
            copy(&mut layered, CC_ARROWS, CC_ARROWS);
            for (i, hole) in self.black_holes.iter().enumerate() {
                if hole.active && !self.blink_object_hidden(hole.entity) {
                    layered.push(self.layered(Part::Hole(i), data));
                }
            }
            debug_assert!(blocks(CC_CLOUDS, CC_CLOUDS).is_empty());
            clouds = (layered.len(), self.scratch.circle_clouds);
            copy(&mut layered, CC_DUPLICATED_ROCKS, CC_DUPLICATED_ROCKS);
            copy(&mut layered, CC_PICKUPS, CC_PICKUPS);
            for rope in 0..self.ropes.len() {
                if self.hook_live(rope) {
                    layered.push(self.layered(Part::Hook(rope), data));
                }
            }
            // The bodies, clones and orbs, in this order in both groups.
            copy(&mut layered, CC_BODIES, CC_ORBS);
            self.scratch.circle_parts = taken;
            #[cfg(debug_assertions)]
            {
                let mut fresh = Vec::new();
                let fresh_clouds = self.push_circle_parts(&mut fresh, data);
                debug_assert!(fresh == layered);
                debug_assert_eq!(fresh_clouds, clouds);
            }
        } else {
            clouds = self.push_circle_parts(&mut layered, data);
        }
        let (cloud_at, cloud_count) = clouds;
        let cloud_layer = data.smoke.cloud_layer;
        // The boxes as the pair loop reads them, which are the Box/Box
        // group's parts too: over the test replays three boxes a pass, mostly
        // the slimes' and the water's, and nearly every one can pair with
        // another box.
        let mut boxes = std::mem::take(&mut self.scratch.box_parts);
        boxes.clear();
        let box_set = self.push_box_group_parts(&mut boxes, cached.then_some(self.scratch.box_bodies), sweep_x, data);
        // Only a circle on a layer some box's collides with can pair
        // (`LayerSet`); over the test replays one circle in five can. Every
        // box goes to the pair loop: a box on a layer no circle's collides
        // with fails each pair's `layers_collide`, so it makes no pair and
        // the rest are as they were. The Box/Box group that follows has the
        // boxes that can pair with another box for its parts, in this order,
        // and its pairs read nothing this group's corrections move: the
        // parts' membership flags, rows' layers and kinds, composites, and
        // the boxes SortAndSweep took (a lightning's and the water's, live,
        // are triggers no group moves).
        let mut group = std::mem::take(&mut self.scratch.group_parts);
        group.clear();
        let mut cloud_place = 0;
        for (place, &(part, layer)) in layered.iter().enumerate() {
            if place == cloud_at {
                cloud_place = group.len();
            }
            if box_set.meets(layer, data) {
                group.push(self.group_unranked(part, layer, sweep_x, data));
            }
        }
        if cloud_at == layered.len() {
            cloud_place = group.len();
        }
        // The clouds among the circles, those that can pair with a box.
        if cloud_count > 0 && box_set.meets(cloud_layer, data) {
            let partners = Self::cloud_partners(&boxes, data);
            self.insert_cloud_parts(&mut group, cloud_place, partners, sweep_x, data);
        }
        // With no circle that can meet a box, over the test replays three
        // passes in five, there is no pair: the contacts stay where they are.
        let paired = !group.is_empty();
        let mut contacts = Vec::new();
        if paired {
            contacts = std::mem::take(&mut self.scratch.cb_contacts);
            contacts.clear();
            self.circle_box_pairs(&group, &boxes, sweep_x, data, &mut contacts);
        }
        #[cfg(debug_assertions)]
        {
            // Every part's pairs are the filtered groups'.
            debug_assert!(collision_matrix_symmetric(data));
            // With every cloud among the circles.
            let mut every_layered = layered.clone();
            let every_clouds = self.cloud_layered(data);
            debug_assert_eq!(every_clouds.len().min(1), cloud_count as usize);
            let every_circles = layered.len() + every_clouds.len();
            every_layered.splice(cloud_at..cloud_at, every_clouds);
            every_layered.extend(boxes.iter().map(|part| (part.part, part.layer)));
            let every_part: Vec<GroupPart> =
                every_layered.iter().map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)).collect();
            let mut every = Vec::new();
            let (circle_parts, box_parts) = every_part.split_at(every_circles);
            self.circle_box_pairs(circle_parts, box_parts, sweep_x, data, &mut every);
            debug_assert!(every.iter().map(|c| (c.0, c.1, c.2)).eq(contacts.iter().map(|c| (c.0, c.1, c.2))));
        }
        // Over the test replays nearly every box can pair with another.
        if !box_set.each_meets_other(data) {
            boxes.retain(|part| box_set.meets_other(part.layer, data));
        }
        debug_assert!(boxes.iter().all(|part| box_set.meets_other(part.layer, data)));
        // The Box/Box group's pairs read nothing this group's corrections
        // move (above), so whether it has two that pair is known now, while
        // its boxes are at hand; one pass in a hundred over the test replays
        // has, and the rest skip that group (`box_box_unpaired`).
        let box_pairs = boxes.len() > 1 && (boxes.len() > 8 || any_pairs(&boxes, data));
        self.scratch.box_pairs = box_pairs;
        self.scratch.box_parts = boxes;
        self.scratch.box_parts_tick = Some(self.tick);
        self.scratch.cb_parts = layered;
        self.scratch.group_parts = group;
        if paired {
            contacts.sort_by_key(|&(key, ..)| key);
            // PrepareCollisionCallBacks: the circle is A, so its OnCollide
            // queues first, then the box's.
            for &(key, circle, part, contact) in &contacts {
                self.queue_collide(key, circle, part, contact);
            }
            for &(_, circle, part, contact) in &contacts {
                self.resolve_parts(circle, part, contact);
            }
            self.scratch.cb_contacts = contacts;
        } else {
            debug_assert!(contacts.is_empty());
        }
        box_pairs
    }

    /// The Circle/Box group's manifolds of `circles` with `boxes`, in the
    /// loop's order, keyed for SortAndSweep's.
    fn circle_box_pairs(
        &self,
        circles: &[GroupPart],
        boxes: &[GroupPart],
        sweep_x: bool,
        data: &data::RolloutData,
        contacts: &mut Vec<(CallbackKey, Part, Part, Contact)>,
    ) {
        for circle in circles {
            for part in boxes {
                // SameCompositeAs: a node never meets its own platform's
                // children; an item, shot or missile has no composite.
                if !circle.pairs(part, data) {
                    continue;
                }
                if let Some(contact) = self.part_collider(circle.part).contact(self.part_collider(part.part)) {
                    let (later, earlier) = sweep_pair(self.group_rank(circle, sweep_x), self.group_rank(part, sweep_x));
                    contacts.push(((GROUP_CB, later, earlier), circle.part, part.part, contact));
                }
            }
        }
    }

    /// DetPhysics' Box/RoundedRect or Circle/RoundedRect group
    /// (DetPhysics.cs:2971-3048): every swept pair of a part with a wall makes
    /// a manifold against `walls`, the colliders as the group's narrow phase
    /// sees them, all built before any resolves. In SortAndSweep pair order
    /// each queues the part's OnCollide (PrepareCollisionCallBacks,
    /// DetPhysics.cs:2405-2437), then each pair that is no trigger resolves
    /// with ResolveCollision and PositionalCorrection on each side's own
    /// inverse mass.
    pub(super) fn wall_group(
        &mut self,
        group: u8,
        parts: &[(Part, [Vec2; 2])],
        walls: &[RoundedRect],
        data: &data::RolloutData,
    ) {
        // No part makes no manifold, and with none nothing below acts (see
        // after the pair loop): over the test replays most Circle/RoundedRect
        // passes have no part.
        if parts.is_empty() {
            return;
        }
        // Left empty by every pass (below), so taken as it is.
        let mut contacts = std::mem::take(&mut self.scratch.wall_contacts);
        debug_assert!(contacts.is_empty());
        // Nothing below moves until every manifold is built. The walls'
        // swept boxes were taken with the callback colliders, once per pass
        // (`callback_bounds`); a wall past them, or every wall when no
        // callback colliders were taken this pass, meets every part's box.
        // The pair tests are pure, so the cheap box test goes first and a
        // part's layer, composite and collider are taken only once a wall
        // passes it; its rank only for a manifold.
        let swept_walls: &[[Vec2; 2]] = if self.scratch.callback_tick == Some(self.tick) {
            &self.scratch.callback_bounds
        } else {
            &[]
        };
        #[cfg(debug_assertions)]
        for wall in 0..self.platforms.len() {
            debug_assert_eq!(
                swept_walls.get(wall).map(|b| [b[0].x, b[0].y, b[1].x, b[1].y]),
                self.wall_swept_bounds(wall).map(|b| [b[0].x, b[0].y, b[1].x, b[1].y])
            );
        }
        // A wall past the swept boxes meets every part's box.
        let boxed_walls = swept_walls.len().min(self.platforms.len());
        // Whether `platforms_changed` below is settled: by a manifold the
        // group found, or by platform rows already stale, which only
        // `refresh_platform_entities` reads and a further bump leaves stale.
        // A slime's or clone's collider is a trigger (`part_body`), so its
        // manifold counts only toward that and for its squish, which reads the
        // wall, not the contact, and acts on few manifolds. Once it is
        // settled, then, a slime's pairs with wall-layer walls queue untested
        // (None) and their squishes test the collider only where they would
        // act; its other pairs leave nothing to queue.
        let stale = self.platform_epoch != self.platform_rows_epoch;
        let mut settled = stale;
        // The pass's sweep axis, for the ranks.
        let sweep_x = self.sweep_x();
        // Debug builds test each untested pair here too, by its key, and
        // whether the group meets anything; and they queue the squishes
        // release leaves out (below), to check each acts on nothing.
        #[cfg(debug_assertions)]
        let (mut eager, mut found): (Vec<(CallbackKey, bool, bool)>, bool) = (Vec::new(), false);
        for &(part, bounds) in parts {
            #[cfg(debug_assertions)]
            {
                let bits = |b: Option<[Vec2; 2]>| b.map(|b| [b[0].x, b[0].y, b[1].x, b[1].y].map(|v| v.to_f64().to_bits()));
                debug_assert_eq!(bits(Some(bounds)), bits(self.swept_bounds(Swept::Part(part))));
            }
            let slime = matches!(part, Part::Body(_) | Part::Clone(_));
            debug_assert!(!slime || self.part_body(part).inverse_mass < Fix::ZERO);
            let mut side: Option<(u32, Option<usize>, Option<usize>)> = None;
            let mut shape: Option<Collider> = None;
            let mut part_rank: Option<SweepRank> = None;
            // The cheap box test first, `WALL_GROUP` walls at a time, then
            // any wall past the swept boxes; in wall order.
            let (mut next, mut first, mut meets, mut unboxed) = (0, 0, 0u64, boxed_walls);
            loop {
                let wall = if meets != 0 {
                    let wall = first + meets.trailing_zeros() as usize;
                    meets &= meets - 1;
                    wall
                } else if next < boxed_walls {
                    first = next;
                    next = boxed_walls.min(next + WALL_GROUP);
                    meets = walls_meeting(&swept_walls[first..next], bounds);
                    continue;
                } else if unboxed < self.platforms.len() {
                    unboxed += 1;
                    unboxed - 1
                } else {
                    break;
                };
                // A settled slime's pair with its position outside the wall
                // queues nothing (below), whatever the tests between: that
                // test first, which nearly every such pair fails. Debug
                // builds take the pair through them all.
                #[cfg(not(debug_assertions))]
                if slime && settled && !self.narrow_wall(walls, wall).overlaps(self.slime_base(wall_slime(part)).position) {
                    continue;
                }
                // SameCompositeAs: a child never meets its own platform. A wall
                // meets parts only while DetPhysics.IsActive, which a wall
                // destroyed before this pass also fails, having left in
                // RemoveDestroyedColliders (DetPhysics.cs:1658-1665, 2731).
                // SortAndSweep pairs only layers the Physics2D matrix lets
                // collide (DetPhysics.cs:2183).
                let (layer, composite, entity) = *side.get_or_insert_with(|| self.wall_part_side(part, data));
                let wall_entity = self.platform_entities[wall];
                let wall_layer = self.entities[wall_entity].layer;
                if composite == Some(wall) || !self.wall_collides(wall) || !data.layers_collide(layer, wall_layer) {
                    continue;
                }
                let rounded = self.narrow_wall(walls, wall);
                let contact = if slime && settled {
                    #[cfg(debug_assertions)]
                    let met = slime_wall_meets(self.wall_part_collider(part), rounded);
                    #[cfg(debug_assertions)]
                    {
                        found |= met;
                    }
                    if wall_layer != LAYER_WALL {
                        continue;
                    }
                    // The squish acts only while the slime's position is
                    // inside the collider the manifold copied (`squish`):
                    // with it outside, nearly every pass's squish, the pair
                    // queues nothing. Nothing moves the slime before its
                    // squish would run; debug builds queue it anyway and
                    // check that it acts on nothing. Nor is the group's
                    // `platforms_changed` lost: `settled` here is a manifold
                    // already queued, or rows already stale, which a further
                    // bump leaves stale. Release builds left it out above.
                    #[cfg(debug_assertions)]
                    {
                        let outside = !rounded.overlaps(self.slime_base(wall_slime(part)).position);
                        eager.push(((0, Default::default(), Default::default()), met, outside));
                    }
                    None
                } else {
                    let collider = *shape.get_or_insert_with(|| self.wall_part_collider_at(part, entity));
                    let contact = if slime {
                        slime_wall_meets(collider, rounded).then(Contact::default)
                    } else {
                        // A Box meets a RoundedRect as a RoundedRect of radius
                        // zero (PhysTools.cs:104-111).
                        match collider {
                            Collider::Circle(circle) => crate::physics::contact::circle_rounded(circle, rounded),
                            Collider::Box(inner) => {
                                crate::physics::contact::rounded_rounded(RoundedRect { inner, radius: Fix::ZERO }, rounded)
                            }
                            Collider::RoundedRect(_) => unreachable!(),
                        }
                    };
                    if contact.is_none() {
                        continue;
                    }
                    settled = true;
                    #[cfg(debug_assertions)]
                    {
                        found = true;
                    }
                    contact
                };
                // `part_rank`, from the box just taken, once a part; `wall_rank`
                // from the wall's swept box, or past them its collider.
                let rank = *part_rank.get_or_insert_with(|| {
                    sweep_rank(&self.entities, if sweep_x { bounds[0].x } else { bounds[0].y }, entity)
                });
                debug_assert!(rank == self.part_rank(part));
                let low = match swept_walls.get(wall) {
                    Some(bounds) => bounds[0],
                    None => Collider::RoundedRect(self.circle_collider(wall)).bounds()[0],
                };
                let rank_wall = sweep_rank(&self.entities, if sweep_x { low.x } else { low.y }, Some(wall_entity));
                debug_assert!(rank_wall == self.wall_rank(wall));
                let (later, earlier) = sweep_pair(rank, rank_wall);
                #[cfg(debug_assertions)]
                if contact.is_none() {
                    eager.last_mut().expect("its eager test").0 = (group, later, earlier);
                }
                contacts.push(((group, later, earlier), part, wall, contact));
            }
        }
        #[cfg(debug_assertions)]
        debug_assert!(stale || contacts.is_empty() != found);
        // With no manifold, over the test replays 49 passes in 50 of either
        // group, nothing below acts: every step reads `contacts` or is taken
        // only with one (the sort, the queued hits, the resolutions, the
        // platforms' bump), and `callback_hits` would be put back as it was.
        if contacts.is_empty() {
            self.scratch.wall_contacts = contacts;
            return;
        }
        contacts.sort_by_key(|&(key, ..)| key);
        // A wall `walls` leaves out is read live up to the resolutions, which
        // are the first to move one.
        #[cfg(debug_assertions)]
        let live_walls: Vec<RoundedRect> = self.platforms.iter().map(|p| p.collider).collect();
        let mut hits = std::mem::take(&mut self.scratch.callback_hits);
        // By reference: the key copied out here was spilled with overlapping
        // stores and reloaded across them for the squish's push (a
        // store-forwarding stall); it is copied into the hit where it is
        // queued.
        for (key, part, wall, contact) in &contacts {
            let (part, wall) = (*part, *wall);
            // collision.layer is the wall's own layer (DetPhysics.cs:2405-2437).
            let layer = self.entities[self.platform_entities[wall]].layer;
            let wall_layer = layer == LAYER_WALL;
            // PlayerCollision.OnCollide's squish acts only on the wall layer
            // (PlayerCollision.cs:326). Queued where it is built: a squish
            // built into `hit` and copied from there reloads the wall's
            // collider across stores of other variants' fields. An untested
            // pair's squish takes the collider its manifold would have
            // tested, which nothing has moved since.
            if let Part::Body(_) | Part::Clone(_) = part {
                if !wall_layer {
                    continue;
                }
                let slime = match part {
                    Part::Body(index) => Slime::Player(index),
                    Part::Clone(index) => Slime::Clone(index),
                    _ => unreachable!(),
                };
                let collider = self.narrow_wall(walls, wall);
                #[cfg(debug_assertions)]
                let (meets, noop) = match contact {
                    Some(_) => (true, false),
                    None => {
                        let &(_, meets, outside) = eager.iter().find(|&&(k, ..)| k == *key).expect("an eager test");
                        (meets, outside)
                    }
                };
                // Written in place, with the room made first and the key
                // copied straight from the contact: `push` built the hit on
                // the stack and copied the key in with loads that straddled
                // the stores that put it there (a store-forwarding stall).
                hits.reserve(1);
                let len = hits.len();
                debug_assert!(len < hits.capacity());
                // SAFETY: `reserve` left room for one more past `len`, both of
                // whose fields are written before the length takes it in.
                unsafe {
                    let slot = hits.as_mut_ptr().add(len);
                    std::ptr::copy_nonoverlapping(key, &raw mut (*slot).0, 1);
                    (&raw mut (*slot).1).write(CallbackHit::Squish {
                        slime,
                        wall,
                        collider,
                        untested: None,
                        #[cfg(debug_assertions)]
                        meets,
                        #[cfg(debug_assertions)]
                        noop,
                    });
                    hits.set_len(len + 1);
                }
                // The untested collider is written into the queued squish, a
                // field at a time: built apart and copied in, its copy reloads
                // it across the stores that built it.
                if contact.is_none() {
                    let Some((_, CallbackHit::Squish { untested, .. })) = hits.last_mut() else { unreachable!() };
                    self.write_wall_part_collider(part, untested);
                }
                continue;
            }
            let contact = contact.as_ref().expect("only a slime's pair is untested");
            let hit = match part {
                // A smoke grenade detonates on any collider but a Projectile
                // that does not ignite explosives, which no rounded rect is
                // (SmokeGrenadeExplode2.cs:63-78); Mjolnir and the Tack act
                // only on the wall layer (Mjolnir.cs:292-318,
                // BananaPeel.cs:113-138).
                Part::Item(item) => match self.thrown_items[item].ability {
                    ABILITY_SMOKE => CallbackHit::ItemDetonate { item, count: 1 },
                    ABILITY_MJOLNIR if wall_layer => CallbackHit::MjolnirWall { item, wall },
                    ABILITY_TACK if wall_layer => CallbackHit::TackWall { item, normal: contact.normal },
                    _ => continue,
                },
                // PortalShot.OnCollide lands on the first wall-layer pair to
                // run, with its normal; the later ones find the shot landed
                // (PortalShot.cs:78-98).
                Part::Shot(shot) if wall_layer => CallbackHit::PortalShotLand { shot, normal: -contact.normal },
                // Arrow.OnCollide lands on a StickTo layer (Arrow.cs:85-131).
                Part::Arrow(arrow) if ARROW_STICK_TO & (1 << layer) != 0 => {
                    CallbackHit::ArrowLand { arrow, wall: Some(wall), part: None }
                }
                // SpikeAttack.OnCollide knocks with knockAwayWallStr only on the
                // wall layer (SpikeAttack.cs:116-141).
                Part::Spike(spike) => CallbackHit::SpikeWallKnock { spike, wall, point: contact.point, wall_layer },
                // Missile.cs:181-192.
                Part::Missile(missile) => CallbackHit::MissileBlast { missile },
                // RopeHook.OnCollide tests the collider the manifold copied
                // (RopeHook.cs:86-131).
                Part::Hook(rope) => {
                    let collider = self.narrow_wall(walls, wall);
                    CallbackHit::RopeHook {
                        rope,
                        target: HookTarget::Platform(wall),
                        shape: Collider::RoundedRect(collider),
                    }
                }
                // BlackHole.OnCollide eats a wall only when the collider the
                // manifold copied contains the hole's centre, even though the
                // two overlap much earlier (BlackHole.cs:207-210, 516-547).
                Part::Hole(hole)
                    if wall_layer
                        && self.narrow_wall(walls, wall).overlaps(self.black_holes[hole].position) =>
                {
                    let inverse_mass = self.platform_dynamics[wall].body.inverse_mass;
                    CallbackHit::BlackHole { hole, target: HoleTarget::Wall(wall), inverse_mass }
                }
                _ => continue,
            };
            hits.push((*key, hit));
        }
        self.scratch.callback_hits = hits;
        #[cfg(debug_assertions)]
        for (wall, before) in live_walls.iter().enumerate() {
            let bits = |r: &RoundedRect| {
                [r.inner.center.x, r.inner.center.y, r.inner.right.x, r.inner.right.y, r.inner.up.x, r.inner.up.y, r.radius]
                    .map(|v| v.to_f64().to_bits())
            };
            debug_assert_eq!(bits(before), bits(&self.platforms[wall].collider));
        }
        for &(_, part, wall, contact) in &contacts {
            // A trigger resolves nothing (`resolve_part_wall`).
            if matches!(part, Part::Body(_) | Part::Clone(_)) {
                debug_assert!(self.part_body(part).inverse_mass < Fix::ZERO);
                continue;
            }
            self.resolve_part_wall(part, wall, contact.expect("only a slime's pair is untested"));
        }
        debug_assert!(!contacts.is_empty());
        self.platforms_changed();
        contacts.clear();
        self.scratch.wall_contacts = contacts;
    }

    /// A wall group part's `part_layer`, `part_composite` and `part_entity`;
    /// a slime's layer from its hurtbox row.
    #[inline(always)]
    fn wall_part_side(&self, part: Part, data: &data::RolloutData) -> (u32, Option<usize>, Option<usize>) {
        let side = match part {
            Part::Body(i) => {
                let entity = Self::hurtbox_entity(&self.players[i]);
                (self.entities[entity].layer, None, Some(entity))
            }
            _ => (self.part_layer(part, data), self.part_composite(part), self.part_entity(part)),
        };
        debug_assert_eq!(side, (self.part_layer(part, data), self.part_composite(part), self.part_entity(part)));
        side
    }

    /// A wall group part's `part_collider`; a slime's from its hurtbox row.
    #[inline(always)]
    #[cfg(debug_assertions)]
    fn wall_part_collider(&self, part: Part) -> Collider {
        let entity = match part {
            Part::Body(i) => Some(Self::hurtbox_entity(&self.players[i])),
            _ => None,
        };
        self.wall_part_collider_at(part, entity)
    }

    /// `*dst = Some(self.wall_part_collider(part))`, each variant written in
    /// place.
    #[inline(always)]
    fn write_wall_part_collider(&self, part: Part, dst: &mut Option<Collider>) {
        match part {
            Part::Body(i) => {
                let entity = Self::hurtbox_entity(&self.players[i]);
                let row = self.entities[entity];
                let center = Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y));
                if row.kind == 0 {
                    *dst = Some(Collider::Circle(Circle { center, radius: Fix::from_raw(row.radius) }));
                } else {
                    *dst = Some(Collider::Box(Self::row_box(&self.entity_inverse_extents, entity, row)));
                }
            }
            _ => *dst = Some(self.part_collider(part)),
        }
        #[cfg(debug_assertions)]
        {
            let bits = |c: Option<Collider>| match c {
                Some(Collider::Circle(c)) => vec![0, c.center.x.to_f64().to_bits(), c.center.y.to_f64().to_bits(), c.radius.to_f64().to_bits()],
                Some(Collider::Box(b)) => [b.center, b.right, b.up, b.inverse_extents]
                    .into_iter()
                    .flat_map(|v| [v.x.to_f64().to_bits(), v.y.to_f64().to_bits()])
                    .chain([1])
                    .collect(),
                _ => unreachable!(),
            };
            debug_assert_eq!(bits(*dst), bits(Some(self.wall_part_collider(part))));
        }
    }

    /// `wall_part_collider` with a slime's `part_entity` (else unread).
    #[inline(always)]
    fn wall_part_collider_at(&self, part: Part, entity: Option<usize>) -> Collider {
        let collider = match (part, entity) {
            (Part::Body(_), Some(entity)) => self.body_row_collider(entity),
            _ => self.part_collider(part),
        };
        debug_assert!(!matches!(part, Part::Body(_)) || entity == self.part_entity(part));
        #[cfg(debug_assertions)]
        {
            let bits = |c: Collider| match c {
                Collider::Circle(c) => vec![c.center.x, c.center.y, c.radius],
                Collider::Box(b) => vec![b.center.x, b.center.y, b.right.x, b.right.y, b.up.x, b.up.y, b.inverse_extents.x, b.inverse_extents.y],
                Collider::RoundedRect(_) => unreachable!(),
            }
            .into_iter()
            .map(|v| v.to_f64().to_bits())
            .collect::<Vec<_>>();
            debug_assert_eq!(bits(collider), bits(self.part_collider(part)));
        }
        collider
    }

    /// Wall `wall` as a wall group's narrow phase sees it: `walls`' copy,
    /// or past them the live collider (read only then).
    #[inline(always)]
    fn narrow_wall(&self, walls: &[RoundedRect], wall: usize) -> RoundedRect {
        match walls.get(wall) {
            Some(&rounded) => rounded,
            None => self.platforms[wall].collider,
        }
    }

    /// The bounds SortAndSweep took of a wall, before this pass's
    /// RoundedRect/RoundedRect corrections; None if it bounded none this pass.
    #[cfg(debug_assertions)]
    fn wall_swept_bounds(&self, wall: usize) -> Option<[Vec2; 2]> {
        self.scratch
            .callback_colliders
            .get(wall)
            .filter(|_| self.scratch.callback_tick == Some(self.tick))
            .map(|&swept| Collider::RoundedRect(swept).bounds())
    }

    /// ResolveCollision on the part's body, a composite child's
    /// CompositeBody at CompositeBody.position, and the wall's, then
    /// PositionalCorrection on each side's own inverse mass, unless either
    /// is a trigger (DetPhysics.cs:2993-3008, 3033-3048).
    fn resolve_part_wall(&mut self, part: Part, wall: usize, contact: Contact) {
        let own = self.part_body(part).inverse_mass;
        let (wall_position, wall_inverse_mass) = self.platform_contact_frame(wall);
        if own < Fix::ZERO || wall_inverse_mass < Fix::ZERO {
            return;
        }
        let (mut body, position) = self.part_resolve_body(part);
        let mut wall_body = self.platform_dynamics[wall].body;
        resolve(contact, &mut body, &mut wall_body, position, wall_position);
        let correction = positional_correction(contact, own, wall_inverse_mass, false);
        self.platform_dynamics[wall].body = wall_body;
        self.set_part_body(part, body, correction[0]);
        self.shift_platform_collider(wall, correction[1]);
    }

    /// An arrow DetPhysics still simulates this pass.
    pub(super) fn arrow_registered(&self, i: usize) -> bool {
        let arrow = &self.arrows[i];
        arrow.active && arrow.registered && !self.blink_object_hidden(arrow.entity)
    }

    /// ResolveCollision on each side's body, a composite child's
    /// CompositeBody at CompositeBody.position, then PositionalCorrection on
    /// each side's own inverse mass.
    fn resolve_parts(&mut self, a: Part, b: Part, contact: Contact) {
        let (a_own, b_own) = (self.part_body(a), self.part_body(b));
        if a_own.inverse_mass < Fix::ZERO || b_own.inverse_mass < Fix::ZERO {
            return;
        }
        let (mut a_body, a_position) = self.part_resolve_body(a);
        let (mut b_body, b_position) = self.part_resolve_body(b);
        resolve(contact, &mut a_body, &mut b_body, a_position, b_position);
        let correction = positional_correction(contact, a_own.inverse_mass, b_own.inverse_mass, false);
        self.set_part_body(a, a_body, correction[0]);
        self.set_part_body(b, b_body, correction[1]);
    }

    /// DetPhysics' Circle/Circle group, after every Circle/RoundedRect pair
    /// and before the Circle/Box pairs' narrow phase: every manifold is built
    /// before the first resolves (DetPhysics.cs:3052-3089). Its callbacks the
    /// rollout reacts to are a hook's, a hole's, a Rock's bounce off another
    /// Rock and a Smoke's, which reads `cc_contacts`.
    #[inline(never)]
    pub(super) fn circle_circle_group(&mut self, data: &data::RolloutData) {
        let sweep_x = self.sweep_x();
        // The parts in the order taken, with their layers, stay for the
        // Circle/Box group (`circle_parts`), with where each kind's end.
        let mut parts = std::mem::take(&mut self.scratch.circle_parts);
        parts.clear();
        // Each kind's end goes straight to `circle_ends`: kept on the stack
        // and copied there at the end, its narrow stores fed a wide reload.
        for (i, item) in self.thrown_items.iter().enumerate() {
            if item.active
                && self.child_registered(item.entity)
                && item.callback_registered_tick != self.tick
            {
                parts.push(self.layered(Part::Item(i), data));
            }
        }
        self.scratch.circle_ends[CC_ITEMS] = parts.len() as u32;
        for i in 0..self.tesla_nodes.len() {
            if self.node_part(i, true) {
                parts.push(self.layered(Part::Node(i), data));
            }
        }
        self.scratch.circle_ends[CC_NODES] = parts.len() as u32;
        for (i, shot) in self.portal_shots.iter().enumerate() {
            if shot.active && shot.registered && !self.blink_object_hidden(shot.entity) {
                parts.push(self.layered(Part::Shot(i), data));
            }
        }
        self.scratch.circle_ends[CC_SHOTS] = parts.len() as u32;
        // The players' parts, by bit, in one pass over them: a Rock form's
        // circle, or a slime's or other form's circle or box (`body_part_circle`,
        // which takes no Rock); the slimes whose box the Circle/Box group
        // takes stay for it.
        // Its last group to read them, it ends the pass's (`pass_body_parts`).
        let (circle_bodies, box_bodies) = self.pass_body_parts();
        self.scratch.body_parts = None;
        let mut rocks = 0u32;
        debug_assert!(self.players.len() <= 32);
        for i in 0..self.players.len() {
            if self.players[i].active == ActiveAbility::Rock {
                debug_assert_eq!(self.body_part_circle(i), None);
                rocks |= (self.hittable(i) as u32) << i;
            }
        }
        let each_bit = |world: &Self, parts: &mut Vec<(Part, u32)>, mut bits: u32, part: fn(usize) -> Part| {
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                parts.push((part(i), world.part_layer(part(i), data)));
                bits &= bits - 1;
            }
        };
        each_bit(self, &mut parts, rocks, Part::Rock);
        #[cfg(debug_assertions)]
        {
            let start = self.scratch.circle_ends[CC_SHOTS] as usize;
            let fresh: Vec<(Part, u32)> = (0..self.players.len())
                .filter(|&i| self.players[i].active == ActiveAbility::Rock && self.hittable(i))
                .map(|i| self.layered(Part::Rock(i), data))
                .collect();
            debug_assert!(fresh[..] == parts[start..]);
        }
        self.scratch.circle_ends[CC_ROCKS] = parts.len() as u32;
        for (i, pickup) in self.ability_pickups.iter().enumerate() {
            if pickup.active && pickup.enabled && !self.blink_object_hidden(pickup.entity) {
                parts.push(self.layered(Part::Pickup(i), data));
            }
        }
        self.scratch.circle_ends[CC_PICKUPS] = parts.len() as u32;
        for (i, rock) in self.duplicated_rocks.iter().enumerate() {
            if rock.active && !self.blink_object_hidden(rock.entity) {
                parts.push(self.layered(Part::DuplicatedRock(i), data));
            }
        }
        self.scratch.circle_ends[CC_DUPLICATED_ROCKS] = parts.len() as u32;
        for rope in 0..self.ropes.len() {
            if self.hook_live(rope) {
                parts.push(self.layered(Part::Hook(rope), data));
            }
        }
        self.scratch.circle_ends[CC_HOOKS] = parts.len() as u32;
        for i in 0..self.arrows.len() {
            if self.arrow_registered(i) {
                parts.push(self.layered(Part::Arrow(i), data));
            }
        }
        self.scratch.circle_ends[CC_ARROWS] = parts.len() as u32;
        for (i, hole) in self.black_holes.iter().enumerate() {
            if hole.active && !self.blink_object_hidden(hole.entity) {
                parts.push(self.layered(Part::Hole(i), data));
            }
        }
        self.scratch.circle_ends[CC_HOLES] = parts.len() as u32;
        // The clouds are taken only if they can pair (`cloud_parts_present`).
        let cloud_count = self.cloud_parts_present(data);
        let cloud_layer = data.smoke.cloud_layer;
        self.scratch.circle_ends[CC_CLOUDS] = parts.len() as u32;
        each_bit(self, &mut parts, circle_bodies, Part::Body);
        #[cfg(debug_assertions)]
        {
            let start = self.scratch.circle_ends[CC_CLOUDS] as usize;
            let fresh: Vec<(Part, u32)> = (0..self.players.len())
                .filter(|&i| self.body_part_circle(i) == Some(true))
                .map(|i| self.body_layered(i, data))
                .collect();
            debug_assert!(fresh[..] == parts[start..]);
            let boxes = (0..self.players.len()).filter(|&i| self.body_part_circle(i) == Some(false));
            debug_assert_eq!(box_bodies, boxes.fold(0, |bits, i| bits | 1 << i));
        }
        self.scratch.circle_ends[CC_BODIES] = parts.len() as u32;
        for i in 0..self.abilityless_clones.len() {
            if self.clone_part_circle(i) == Some(true) {
                parts.push(self.layered(Part::Clone(i), data));
            }
        }
        self.scratch.circle_ends[CC_CLONES] = parts.len() as u32;
        self.push_blasts(&mut parts, true, data);
        self.scratch.circle_ends[CC_ORBS] = parts.len() as u32;
        // The pair loop reads, in rank order, only the parts with a partner
        // on a layer theirs collides with (`LayerSet`): over the test replays
        // four parts in five, most of them Smoke clouds, have none.
        let mut layers = LayerSet::default();
        parts.iter().for_each(|&(_, layer)| layers.add(layer));
        (0..cloud_count).for_each(|_| layers.add(cloud_layer));
        let mut group = std::mem::take(&mut self.scratch.group_parts);
        group.clear();
        let cloud_at = self.scratch.circle_ends[CC_CLOUDS] as usize;
        let mut cloud_place = 0;
        for (place, &(part, layer)) in parts.iter().enumerate() {
            if place == cloud_at {
                cloud_place = group.len();
            }
            if layers.meets_other(layer, data) {
                group.push(self.group_part(part, layer, sweep_x, data));
            }
        }
        if cloud_at == parts.len() {
            cloud_place = group.len();
        }
        if cloud_count > 0 && layers.meets_other(cloud_layer, data) {
            let partners = Self::cloud_partners(&group, data);
            self.insert_cloud_parts(&mut group, cloud_place, partners, sweep_x, data);
        }
        self.scratch.circle_clouds = cloud_count;
        self.scratch.box_bodies = box_bodies;
        self.scratch.circle_parts_tick = Some(self.tick);
        // A group of one part or none makes no pair: its contacts and the
        // sweep's open priors stay where they are.
        if group.len() <= 1 {
            #[cfg(debug_assertions)]
            {
                let mut every_layered = parts.clone();
                let every_clouds = self.cloud_layered(data);
                every_layered.splice(cloud_at..cloud_at, every_clouds);
                let mut every_part: Vec<GroupPart> =
                    every_layered.iter().map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)).collect();
                sort_by_rank(&mut every_part);
                let (mut live, mut every) = (Vec::new(), Vec::new());
                self.circle_circle_pairs(&every_part, data, &mut live, &mut every);
                debug_assert!(every.is_empty(), "a Circle/Circle group of one part made a manifold");
            }
            self.scratch.group_parts = group;
            self.scratch.circle_parts = parts;
            return;
        }
        let mut contacts = std::mem::take(&mut self.scratch.cc_contacts);
        contacts.clear();
        let mut live = std::mem::take(&mut self.scratch.sweep_live);
        sort_by_rank(&mut group);
        self.circle_circle_pairs(&group, data, &mut live, &mut contacts);
        #[cfg(debug_assertions)]
        {
            // Every part's pairs are the filtered group's.
            debug_assert!(collision_matrix_symmetric(data));
            let mut every_layered = parts.clone();
            let every_clouds = self.cloud_layered(data);
            debug_assert_eq!(every_clouds.len().min(1), cloud_count as usize);
            every_layered.splice(cloud_at..cloud_at, every_clouds);
            let mut every_part: Vec<GroupPart> =
                every_layered.iter().map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)).collect();
            sort_by_rank(&mut every_part);
            let mut every = Vec::new();
            self.circle_circle_pairs(&every_part, data, &mut live, &mut every);
            debug_assert!(every.iter().map(|c| (c.0, c.1)).eq(contacts.iter().map(|c| (c.0, c.1))));
        }
        self.scratch.sweep_live = live;
        self.scratch.group_parts = group;
        // PrepareCollisionCallBacks queues every pair's callbacks before the
        // first ResolveCollision (DetPhysics.cs:3068-3071), the
        // earlier-sorted side's first.
        for &(a, b, contact) in &contacts {
            let (later, earlier) = sweep_pair(self.part_rank(a), self.part_rank(b));
            self.queue_collide((GROUP_CC, later, earlier), a, b, contact);
        }
        for &(a, b, contact) in &contacts {
            self.resolve_parts(a, b, contact);
        }
        self.scratch.circle_parts = parts;
        self.scratch.cc_contacts = contacts;
    }

    /// The Circle/Circle group's manifolds of a rank-sorted group, in
    /// SortAndSweep's pair order.
    fn circle_circle_pairs(
        &self,
        group: &[GroupPart],
        data: &data::RolloutData,
        live: &mut Vec<LivePart>,
        contacts: &mut Vec<(Part, Part, Contact)>,
    ) {
        sweep_pairs(group, self.sweep_x(), data, live, |prior, current| {
            // A hook's pair with a hole is grapple_hook_pairs'.
            if matches!((prior.part, current.part), (Part::Hook(_), Part::Hole(_)) | (Part::Hole(_), Part::Hook(_))) {
                return;
            }
            // SameCompositeAs: nodes of one platform share its composite.
            if !prior.pairs(current, data) {
                return;
            }
            if let Some(contact) = self.part_collider(prior.part).contact(self.part_collider(current.part)) {
                contacts.push((prior.part, current.part, contact));
            }
        });
    }

    /// DetPhysics' Box/Box group, after the Circle/Box one: rocket engine and
    /// spike boxes of different platforms, Guided Missiles, and slime, form
    /// and clone boxes. Every manifold
    /// is built before the first resolves, each queues A's OnCollide then
    /// B's, and ResolveCollision works on a child's CompositeBody while
    /// PositionalCorrection takes its own inverse mass (DetPhysics.cs:3131-3168).
    #[inline(never)]
    pub(super) fn box_box_group(&mut self, data: &data::RolloutData) {
        #[cfg(debug_assertions)]
        let sweep_x = self.sweep_x();
        // The Circle/Box group just took these parts (`box_parts`): the
        // boxes that can pair with another (`LayerSet`).
        let cached = self.scratch.box_parts_tick.take() == Some(self.tick);
        let mut group =
            std::mem::take(if cached { &mut self.scratch.box_parts } else { &mut self.scratch.group_parts });
        let mut layered = std::mem::take(&mut self.scratch.cb_parts);
        layered.clear();
        if !cached {
            group.clear();
            self.box_box_parts(data, &mut layered, &mut group);
        }
        #[cfg(debug_assertions)]
        if cached {
            let mut fresh = Vec::new();
            self.box_box_parts(data, &mut layered, &mut fresh);
            debug_assert_eq!(fresh.len(), group.len());
            for (fresh, taken) in fresh.iter().zip(group.iter()) {
                debug_assert!(fresh.same(taken));
            }
        }
        let mut contacts = std::mem::take(&mut self.scratch.cc_contacts);
        contacts.clear();
        let mut live = std::mem::take(&mut self.scratch.sweep_live);
        // A group of a few parts with no two that pair (`any_pairs`) is left
        // unsorted; the Circle/Box group answered that for its boxes.
        let sorted = if cached {
            debug_assert_eq!(
                self.scratch.box_pairs,
                group.len() > 1 && (group.len() > 8 || any_pairs(&group, data))
            );
            self.scratch.box_pairs
        } else {
            group.len() > 1 && (group.len() > 8 || any_pairs(&group, data))
        };
        if sorted {
            // The Circle/Box group's boxes take their ranks now
            // (`unranked_part`).
            if cached {
                let sweep_x = self.sweep_x();
                for part in group.iter_mut() {
                    part.rank = self.group_rank(part, sweep_x);
                }
            }
            #[cfg(debug_assertions)]
            debug_assert!(group.iter().all(|part| same_rank(part.rank, part.eager)));
            sort_by_rank(&mut group);
            self.box_box_pairs(&group, data, &mut live, &mut contacts);
        }
        #[cfg(debug_assertions)]
        {
            // Every part's pairs are the filtered group's.
            debug_assert!(collision_matrix_symmetric(data));
            let mut every_part: Vec<GroupPart> =
                layered.iter().map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)).collect();
            sort_by_rank(&mut every_part);
            let mut every = Vec::new();
            self.box_box_pairs(&every_part, data, &mut live, &mut every);
            debug_assert!(every.iter().map(|c| (c.0, c.1)).eq(contacts.iter().map(|c| (c.0, c.1))));
        }
        self.scratch.cb_parts = layered;
        self.scratch.sweep_live = live;
        *(if cached { &mut self.scratch.box_parts } else { &mut self.scratch.group_parts }) = group;
        for &(a, b, contact) in &contacts {
            let (later, earlier) = sweep_pair(self.part_rank(a), self.part_rank(b));
            self.queue_collide((GROUP_BB, later, earlier), a, b, contact);
        }
        for &(a, b, contact) in &contacts {
            self.resolve_parts(a, b, contact);
        }
        if !contacts.is_empty() {
            self.platforms_changed();
        }
        self.scratch.cc_contacts = contacts;
    }

    /// `box_box_group` on a pass whose boxes the Circle/Box group just took
    /// with no two that pair (`any_pairs`): it would sort none and build no
    /// manifold, so it only drops the boxes' stamp. Debug builds take every
    /// box afresh and check the full group makes no manifold.
    pub(super) fn box_box_unpaired(&mut self, data: &data::RolloutData) {
        let cached = self.scratch.box_parts_tick.take() == Some(self.tick);
        debug_assert!(cached && !self.scratch.box_pairs);
        #[cfg(debug_assertions)]
        {
            let sweep_x = self.sweep_x();
            let mut layered = Vec::new();
            self.push_box_parts(&mut layered, None, data);
            let mut every_part: Vec<GroupPart> =
                layered.iter().map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)).collect();
            sort_by_rank(&mut every_part);
            let (mut live, mut every) = (Vec::new(), Vec::new());
            self.box_box_pairs(&every_part, data, &mut live, &mut every);
            debug_assert!(every.is_empty(), "a Box/Box pass skipped with a manifold");
        }
        let _ = (cached, data);
    }

    /// The Box/Box group's parts that can pair with another (`LayerSet`),
    /// into `group`, with every box and its layer in `layered`.
    fn box_box_parts(&self, data: &data::RolloutData, layered: &mut Vec<(Part, u32)>, group: &mut Vec<GroupPart>) {
        let sweep_x = self.sweep_x();
        self.push_box_parts(layered, None, data);
        let mut layers = LayerSet::default();
        layered.iter().for_each(|&(_, layer)| layers.add(layer));
        group.extend(
            layered
                .iter()
                .filter(|&&(_, layer)| layers.meets_other(layer, data))
                .map(|&(part, layer)| self.group_part(part, layer, sweep_x, data)),
        );
    }

    /// The Box/Box group's manifolds of a rank-sorted group, in
    /// SortAndSweep's pair order.
    fn box_box_pairs(
        &self,
        group: &[GroupPart],
        data: &data::RolloutData,
        live: &mut Vec<LivePart>,
        contacts: &mut Vec<(Part, Part, Contact)>,
    ) {
        sweep_pairs(group, self.sweep_x(), data, live, |prior, current| {
            if prior.pairs(current, data) {
                if let Some(contact) = self.part_collider(prior.part).contact(self.part_collider(current.part)) {
                    contacts.push((prior.part, current.part, contact));
                }
            }
        });
    }

    /// PrepareCollisionCallBacks for one manifold: A's OnCollide with B as the
    /// collider, then B's with A, both with the manifold's normal and point
    /// (DetPhysics.cs:2406-2438). Only colliders whose OnCollide the rollout
    /// models queue.
    pub(super) fn queue_collide(&mut self, key: CallbackKey, a: Part, b: Part, contact: Contact) {
        for (receiver, other) in [(a, b), (b, a)] {
            if matches!(
                receiver,
                Part::Hole(_)
                    | Part::Item(_)
                    | Part::Pickup(_)
                    | Part::Rock(_)
                    | Part::Arrow(_)
                    | Part::Hook(_)
                    | Part::Spike(_)
                    | Part::Missile(_)
                    | Part::Body(_)
                    | Part::Clone(_)
                    | Part::Cloud(_)
            ) {
                let collider = self.part_entity(other);
                let shape = self.part_collider(other);
                self.scratch
                    .callback_hits
                    .push((key, CallbackHit::Collide { receiver, other, collider, shape, contact }));
            }
        }
    }

    /// FixTransform.IsDestroyed for an entry's other collider: Updater.DestroyFix
    /// marked its row earlier in this pass.
    fn collider_destroyed(&self, collider: Option<usize>) -> bool {
        collider.and_then(|entity| self.entities.get(entity)).is_some_and(|row| row.flags & E_DESTROYED != 0)
    }

    /// The receiver's collisionCallback.OnCollide, run from DetPhysics' queue
    /// (DetPhysics.cs:3202-3205), or from a dash's or form's own UpdateSim
    /// for each collider its ray or cast reaches (Dash.cs:150-178,
    /// MeteorSmash.cs:395-408, Roll.cs:314-325).
    pub(super) fn on_collide(
        &mut self,
        receiver: Part,
        other: Part,
        collider: Option<usize>,
        shape: Collider,
        contact: Contact,
        route: KillRoute,
        data: &data::RolloutData,
    ) {
        // A slime's or an abilityless clone's own box calls its
        // SlimeController.OnCollide before its PlayerCollision's: Player.prefab
        // numbers them 2 and 3, and RegisterCollisionCallback keeps the box's
        // callbacks in that order (DPhysicsBox.cs:383-402, SlimeController.cs:471,
        // PlayerCollision.cs:89). It bites and shoves off any Player-layer
        // collider (SlimeController.cs:1206-1244).
        let slime = match receiver {
            Part::Body(player) if self.players[player].active == ActiveAbility::None => Some(Slime::Player(player)),
            Part::Clone(clone) => Some(Slime::Clone(clone)),
            _ => None,
        };
        let prey = match other {
            Part::Body(player) | Part::Rock(player) => Some(Slime::Player(player)),
            Part::Clone(clone) => Some(Slime::Clone(clone)),
            _ => None,
        };
        if let (Some(slime), Some(prey)) = (slime, prey)
            && self.part_layer(other, data) == LAYER_PLAYER
        {
            let hierarchy = |part| self.part_entity(part).and_then(|entity| self.entities.get(entity)).map(|row| row.hierarchy);
            let lower = hierarchy(receiver) < hierarchy(other);
            self.slime_collide(slime, prey, collider, lower);
        }
        match receiver {
            Part::Hole(hole) => self.black_hole_on_collide(hole, other, route, data),
            Part::Item(item) => {
                let layer = self.part_layer(other, data);
                let state = self.thrown_items[item];
                match state.ability {
                    // SmokeGrenadeExplode2 goes off on any collider but a
                    // Projectile that does not ignite explosives, which no
                    // Projectile prefab does (SmokeGrenadeExplode2.cs:63-78;
                    // IgnitesExplosives 0 on Arrow, MeteorSmear, ...).
                    ABILITY_SMOKE if layer != LAYER_PROJECTILE => {
                        self.run_callback_hit(&CallbackHit::ItemDetonate { item, count: 1 }, route, data);
                    }
                    // Mine.OnCollide arms on its Igniters' layers unless the
                    // collider is the holding MagnetGun's hurtbox, and sees an
                    // arrow's contact whether or not it landed first
                    // (Mine.cs:227-235).
                    ABILITY_MINE if MINE_IGNITERS & (1 << layer) != 0 => {
                        if state.hold.holder.and_then(|gun| self.magnet_gun_hurtbox(gun)).is_some_and(|row| Some(row) == collider) {
                            return;
                        }
                        self.run_callback_hit(&CallbackHit::MineIgnite { item }, route, data);
                    }
                    // GrenadeExplode.OnCollide goes off on the Explosion layer
                    // (GrenadeExplode.cs:78-81).
                    ABILITY_GRENADE if layer == LAYER_EXPLOSION => {
                        self.run_callback_hit(&CallbackHit::ItemDetonate { item, count: 1 }, route, data);
                    }
                    // and on a Player-layer collider but its owner's, unless it
                    // detonates on its owner, and the holding MagnetGun's
                    // hurtbox (GrenadeExplode.cs:41-68).
                    ABILITY_GRENADE if layer == LAYER_PLAYER => {
                        let entity = collider.expect("grouped Player-layer part");
                        if state.hold.holder.and_then(|gun| self.magnet_gun_hurtbox(gun)) == Some(entity) {
                            return;
                        }
                        // The collider's IPlayerIdHolder: a slime's or form's
                        // own player, whatever its row last published.
                        let holder_id = match other {
                            Part::Body(player) | Part::Rock(player) => self.players[player].base.id,
                            _ => self.entities[entity].owner,
                        };
                        if holder_id != player_id(&self.players, state.owner) || state.detonates_on_owner {
                            self.run_callback_hit(&CallbackHit::ItemDetonate { item, count: 1 }, route, data);
                        }
                    }
                    // PokeBall.OnCollide captures the SlimeController on a
                    // Player-layer collider's FixTransform, a slime's own box
                    // or an abilityless clone's, unless it is its thrower's;
                    // the first capture in callback order wins (PokeBall.cs:122-141).
                    ABILITY_POKE_BALL if layer == LAYER_PLAYER => {
                        let owner = player_id(&self.players, state.owner);
                        let hit = match (other, collider) {
                            (Part::Body(player), Some(body))
                                if body == self.players[player].body_entity && self.players[player].base.id != owner =>
                            {
                                CallbackHit::PokeCapture { item, victim: player, clone: false }
                            }
                            (Part::Clone(clone), _) if self.abilityless_clones[clone].base.id != owner => {
                                CallbackHit::PokeCapture { item, victim: clone, clone: true }
                            }
                            _ => return,
                        };
                        self.run_callback_hit(&hit, route, data);
                    }
                    _ => {}
                }
            }
            // The form's circle calls BounceBall's OnCollide only: Rock.prefab
            // has no PlayerCollision (DPhysicsCircle.cs:330-346, BounceBall.cs:310-323).
            Part::Rock(rock) => self.rock_on_collide(rock, contact.normal, self.part_layer(other, data)),
            // DynamicAbilityPickup.OnCollide hands its ability to the
            // SlimeController on the collider's FixTransform, a slime's own
            // box or an abilityless clone's, and to nothing else, whatever an
            // earlier callback made of the slime (DynamicAbilityPickup.cs:158-170).
            Part::Pickup(pickup) => {
                let Some(body) = collider else {
                    return;
                };
                let taker = match other {
                    Part::Body(player) if body == self.players[player].body_entity => PickupTaker::Body(player),
                    Part::Clone(clone) => PickupTaker::Clone(clone),
                    _ => return,
                };
                self.run_callback_hit(&CallbackHit::Pickup { pickup, taker, body }, route, data);
            }
            // Arrow.OnCollide on a StickTo layer Drops the arrow out of a
            // Magnet's hold, then lands, but not on a Player-layer collider
            // without a BoplBody (Arrow.cs:87-96): of the Player-layer parts
            // only a Rock form has one.
            Part::Arrow(arrow) => {
                let layer = self.part_layer(other, data);
                if ARROW_STICK_TO & (1 << layer) == 0 || !self.arrows[arrow].active {
                    return;
                }
                if layer != LAYER_PLAYER || matches!(other, Part::Rock(_)) {
                    self.arrow_land(arrow, None, Some(other), data);
                } else {
                    let state = &mut self.arrows[arrow];
                    state.hold.release();
                }
            }
            Part::Hook(rope) => {
                let target = match other {
                    Part::Item(i) => HookTarget::Item(i),
                    Part::Shot(i) => HookTarget::Shot(i),
                    Part::Node(i) => HookTarget::Node(i),
                    Part::Pickup(_) => HookTarget::Pickup,
                    // RopeHook ignores every body of the caster's and the end's
                    // player (RopeHook.cs:108-115); a slime's, form's or
                    // clone's Player-layer collider carries its PlayerBody.
                    Part::Rock(_) | Part::Body(_) | Part::Clone(_) => {
                        let (body_player, target) = match other {
                            Part::Clone(clone) => (self.abilityless_clones[clone].owner, HookTarget::Clone(clone)),
                            Part::Rock(player) | Part::Body(player) => {
                                (self.players[player].owner, HookTarget::Player(player))
                            }
                            _ => unreachable!(),
                        };
                        if body_player == self.players[self.ropes[rope].owner].owner
                            || body_player == self.ropes[rope].hook_end_player
                        {
                            return;
                        }
                        target
                    }
                    Part::Spike(i) => HookTarget::Spike(i),
                    Part::Engine(i) => HookTarget::Engine(i),
                    Part::Arrow(i) => HookTarget::Arrow(i),
                    Part::Missile(i) => HookTarget::Missile(i),
                    // A hook on a Duplicator copy has no rope parent here yet.
                    Part::DuplicatedRock(_) => return,
                    Part::Hook(_)
                    | Part::Hole(_)
                    | Part::Line(_)
                    | Part::Cloud(_)
                    | Part::Explosion(_)
                    | Part::Water => return,
                };
                self.rope_hook_collide(rope, target, shape);
            }
            Part::Spike(spike) => {
                // SpikeAttack.OnCollide returns only on a destroyed collider
                // (SpikeAttack.cs:118): a killPlayer earlier in the queue
                // DestroyFix'd the slime's box, while one through a form only
                // exits it (PlayerCollision.cs:121-132, 211).
                if let (Part::Body(_), Some(entity)) = (other, collider)
                    && self.entities[entity].flags & (E_FIX_DESTROYED | E_DESTROYED | E_REMOVED) != 0
                {
                    return;
                }
                let hit = CallbackHit::SpikeShove { spike, target: other, point: contact.point };
                self.run_callback_hit(&hit, route, data);
            }
            // FlammableSmoke.OnCollide Ignites the cloud on an Explosion with
            // igniteFlammables, as the lightning's (FlammableSmoke.cs:91-98).
            Part::Cloud(cloud) => {
                if let (true, Some(entity)) = (Self::ignites_flammables(other), self.smoke_clouds[cloud].entity) {
                    self.run_callback_hit(&CallbackHit::SmokeIgnite { cloud, entity }, route, data);
                }
            }
            // Missile.OnCollide ignores RigidBodyAffector and Rope (Missile.cs:183).
            Part::Missile(missile) if !matches!(self.part_layer(other, data), LAYER_RIGID_BODY_AFFECTOR | LAYER_ROPE) => {
                self.run_callback_hit(&CallbackHit::MissileBlast { missile }, route, data);
            }
            // PlayerCollision.OnCollide's KillPlayerOnCollision.
            Part::Body(player) => {
                if let Some((killer, cause)) = self.collision_kill(Slime::Player(player), other, collider) {
                    let slime = self.players[player].active == ActiveAbility::None;
                    let arrow = matches!(other, Part::Arrow(_));
                    self.run_callback_hit(&CallbackHit::Kill { player, killer, cause, slime, arrow }, route, data);
                }
            }
            Part::Clone(clone) => {
                if let Some((killer, cause)) = self.collision_kill(Slime::Clone(clone), other, collider) {
                    let arrow = matches!(other, Part::Arrow(_));
                    self.run_callback_hit(&CallbackHit::CloneKill { clone, killer, cause, arrow }, route, data);
                }
            }
            _ => {}
        }
    }

    /// The live part whose collider a ray or cast reached, for its
    /// collisionCallback: any grouped kind, a slime's, form's or clone's box
    /// included.
    pub(super) fn hit_part(&self, entity: usize) -> Option<Part> {
        self.part_of_entity(entity).or_else(|| {
            (0..self.players.len())
                .map(Part::Body)
                .chain((0..self.abilityless_clones.len()).map(Part::Clone))
                .chain((0..self.black_holes.len()).map(Part::Hole))
                .chain((0..self.smoke_clouds.len()).map(Part::Cloud))
                .chain((0..self.tesla_nodes.len()).map(Part::Line))
                .find(|&part| !self.part_destroyed(part) && self.part_entity(part) == Some(entity))
        })
        .or_else(|| self.blast_at(entity).map(Part::Explosion))
    }

    /// The Explosion-layer collider whose row a ray or cast reached, by the
    /// live slot that holds it.
    fn blast_at(&self, entity: usize) -> Option<Blast> {
        let at = |active: bool, row: Option<usize>| active && row == Some(entity);
        if let Some(i) = self.grenade_explosions.iter().position(|e| at(e.active, e.entity)) {
            return Some(Blast::Grenade(i));
        }
        if let Some(i) = self.missile_explosions.iter().position(|e| at(e.active, e.entity)) {
            return Some(Blast::Mine(i));
        }
        if let Some(i) = self.smoke_explosions.iter().position(|e| at(e.active, e.entity)) {
            return Some(Blast::Smoke(i));
        }
        if let Some(i) = self.meteor_smears.iter().position(|smear| at(smear.active, smear.entity)) {
            return Some(Blast::Smear(i));
        }
        if let Some(i) = self.beam_pool.iter().position(|beam| at(beam.active, beam.entity)) {
            return Some(Blast::Beam(i));
        }
        if let Some(i) = self.beam_orbs.iter().position(|orb| at(orb.active, orb.entity)) {
            return Some(Blast::Orb(i));
        }
        self.players.iter().enumerate().find_map(|(p, slot)| {
            let k = slot.drill_hitbox_entities[slot.active_slot].iter().position(|&row| row == entity)?;
            Some(Blast::Drill(p, k))
        })
    }

    /// BlackHole.OnCollide with a collider of a narrow-phase group. It reads
    /// the collider's inverse mass when it runs, a composite child's being
    /// its CompositeBody's, and a Rock form's its BounceBall body's
    /// (BlackHole.cs:445-458).
    pub(super) fn black_hole_on_collide(&mut self, hole: usize, other: Part, route: KillRoute, data: &data::RolloutData) {
        let (target, inverse_mass) = match other {
            Part::Arrow(i) => (HoleTarget::Arrow(i), self.arrows[i].inverse_mass),
            Part::Pickup(i) => (HoleTarget::Pickup(i), self.ability_pickups[i].body.inverse_mass),
            Part::Item(i) => (HoleTarget::Item(i), self.thrown_items[i].body.inverse_mass),
            Part::Shot(i) => (HoleTarget::Shot(i), self.portal_shots[i].body.inverse_mass),
            Part::DuplicatedRock(i) => (HoleTarget::DuplicatedRock(i), self.duplicated_rocks[i].body.inverse_mass),
            Part::Hole(i) => (HoleTarget::Hole(i), -Fix::ONE),
            Part::Rock(player) => (HoleTarget::Player(player, false), self.players[player].rock.body.inverse_mass),
            Part::Node(i) => (HoleTarget::Node(i), self.platform_dynamics[self.tesla_nodes[i].platform_index].body.inverse_mass),
            Part::Engine(_) | Part::Spike(_) | Part::Missile(_) => {
                let target = match other {
                    Part::Engine(i) => HoleTarget::Engine(i),
                    Part::Spike(i) => HoleTarget::Spike(i),
                    Part::Missile(i) => HoleTarget::Missile(i),
                    _ => unreachable!(),
                };
                let inverse_mass = match self.part_composite(other) {
                    Some(platform) => self.platform_dynamics[platform].body.inverse_mass,
                    None => self.part_body(other).inverse_mass,
                };
                (target, inverse_mass)
            }
            // Grow reads -1 from a collider no BoplBody owns.
            Part::Body(player) => (HoleTarget::Player(player, self.players[player].active == ActiveAbility::None), -Fix::ONE),
            Part::Clone(clone) => (HoleTarget::Clone(clone), -Fix::ONE),
            // Smoke (layer 22) pairs with RigidBodyAffector, so an unlit
            // cloud is eaten like any other positive-mass circle.
            Part::Cloud(i) => (HoleTarget::Cloud(i), self.smoke_clouds[i].body.inverse_mass),
            Part::Hook(_) | Part::Line(_) | Part::Explosion(_) | Part::Water => return,
        };
        self.black_hole_collide(hole, target, inverse_mass, route, data);
    }
}
