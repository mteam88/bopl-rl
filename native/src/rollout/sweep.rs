//! DetPhysics.Simulate finds every pair once, in SortAndSweep, on the
//! bounding boxes the colliders had before any contact group corrected
//! anything (DetPhysics.cs:2169-2231); every group then works from that one
//! pair list. The pair set is the colliders whose boxes overlap strictly on
//! both axes, so a predicate over one snapshot of those boxes is that list.

use super::*;

/// A collider in SortAndSweep, by the collection that owns it.
// Release builds bound all but `Part` in `take_sweep` directly; debug builds
// name the rest to check those lists.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(super) enum Swept {
    Part(Part),
    /// The player's registered Player-layer shape: the slime's box, or its
    /// form's collider (a Rock's circle).
    Player(usize),
    Clone(usize),
    Arrow(usize),
    DuplicatedRock(usize),
    Pickup(usize),
}

/// Each collider's bounding box as SortAndSweep took it this pass.
#[derive(Default)]
pub(super) struct SweepBounds {
    tick: Option<u32>,
    items: Vec<[Vec2; 2]>,
    shots: Vec<[Vec2; 2]>,
    nodes: Vec<[Vec2; 2]>,
    engines: Vec<[Vec2; 2]>,
    spikes: Vec<[Vec2; 2]>,
    missiles: Vec<[Vec2; 2]>,
    players: Vec<[Vec2; 2]>,
    clones: Vec<[Vec2; 2]>,
    arrows: Vec<[Vec2; 2]>,
    duplicated_rocks: Vec<[Vec2; 2]>,
    pickups: Vec<[Vec2; 2]>,
    rows: Vec<[Vec2; 2]>,
    /// The spikes and Guided Missiles take_sweep bounded, the active ones;
    /// `spikes` and `missiles` hold NOWHERE for the rest, which nothing
    /// reads.
    #[cfg(debug_assertions)]
    spikes_taken: Vec<bool>,
    #[cfg(debug_assertions)]
    missiles_taken: Vec<bool>,
    /// take_sweep's rows some pool could own.
    #[cfg(debug_assertions)]
    owned: Vec<bool>,
    /// The rows take_sweep bounded; `rows` holds a stale box or NOWHERE for
    /// the rest, which nothing reads.
    #[cfg(debug_assertions)]
    taken: Vec<bool>,
}

/// A box that overlaps nothing, for a collider out of the sweep.
const NOWHERE: [Vec2; 2] = [Vec2::new(Fix::MAX, Fix::MAX), Vec2::new(Fix::MIN, Fix::MIN)];

/// Whether two bounding boxes overlap strictly on both axes.
pub(super) fn swept_meet(a: [Vec2; 2], b: [Vec2; 2]) -> bool {
    a[0].x < b[1].x && b[0].x < a[1].x && a[0].y < b[1].y && b[0].y < a[1].y
}

pub(super) fn bounds(center: Vec2, half: Vec2) -> [Vec2; 2] {
    [center - half, center + half]
}

fn box_half(right: Vec2, up: Vec2) -> Vec2 {
    Vec2::new(right.x.abs() + up.x.abs(), right.y.abs() + up.y.abs())
}

fn collider_bounds(collider: DetCollider) -> [Vec2; 2] {
    match collider {
        DetCollider::Box(collider) => box_bounds(collider),
        DetCollider::Circle(collider) => {
            bounds(collider.center, Vec2::new(collider.radius, collider.radius))
        }
    }
}

fn box_bounds(collider: OrientedBox) -> [Vec2; 2] {
    bounds(collider.center, box_half(collider.right, collider.up))
}

impl RolloutWorld {
    /// Takes every collider's bounding box after integration, before the
    /// first correction moves any of them.
    pub(super) fn take_sweep(&self, sweep: &mut SweepBounds, data: &data::RolloutData) {
        sweep.tick = Some(self.tick);
        let _ = data;
        // Each list from its pool's own arm of `current_bounds`, which the
        // fills call directly rather than through its match; debug builds
        // compare every box with `current_bounds`.
        fn fill(list: &mut Vec<[Vec2; 2]>, count: usize, f: impl Fn(usize) -> [Vec2; 2]) {
            list.clear();
            list.extend((0..count).map(f));
        }
        let circle = |center: Vec2, radius: Fix| bounds(center, Vec2::new(radius, radius));
        let table = &self.entity_inverse_extents;
        fill(&mut sweep.items, self.thrown_items.len(), |i| {
            circle(self.thrown_items[i].position, self.thrown_items[i].collider_radius)
        });
        fill(&mut sweep.shots, self.portal_shots.len(), |i| {
            circle(self.portal_shots[i].position, self.portal_shots[i].collider_radius)
        });
        fill(&mut sweep.nodes, self.tesla_nodes.len(), |i| {
            circle(self.tesla_nodes[i].collider_center, self.tesla_nodes[i].radius)
        });
        fill(&mut sweep.engines, self.engines.len(), |i| box_bounds(self.engines[i].rocket_box(table)));
        // A spike's or a Guided Missile's box is read only for a part of
        // this pass's groups: the Box/RoundedRect group takes only active
        // spikes and live missiles (`guided_missile_live`), and the others a
        // spike only through its collider (`child_registered`), which an
        // inactive spike's row is not (`detphysics_collider`), and a missile
        // only live. Nothing between here and the groups activates either, so
        // an inactive one, about half the spike slots and four missile slots
        // in five over the test replays, is left unbounded and its sin and cos
        // untaken (debug builds check every one read was taken).
        fill(&mut sweep.spikes, self.spike_attacks.len(), |i| {
            if self.spike_attacks[i].active { self.spike_bounds(i) } else { NOWHERE }
        });
        fill(&mut sweep.missiles, self.guided_missiles.len(), |i| {
            let missile = &self.guided_missiles[i];
            if missile.active { box_bounds(missile.collider(table)) } else { NOWHERE }
        });
        #[cfg(debug_assertions)]
        {
            sweep.spikes_taken.clear();
            sweep.spikes_taken.extend(self.spike_attacks.iter().map(|spike| spike.active));
            sweep.missiles_taken.clear();
            sweep.missiles_taken.extend(self.guided_missiles.iter().map(|missile| missile.active));
        }
        fill(&mut sweep.players, self.players.len(), |i| self.player_bounds(i));
        fill(&mut sweep.clones, self.abilityless_clones.len(), |i| self.clone_bounds(i));
        fill(&mut sweep.arrows, self.arrows.len(), |i| self.arrow_bounds(i));
        fill(&mut sweep.duplicated_rocks, self.duplicated_rocks.len(), |i| self.duplicated_rock_bounds(i));
        fill(&mut sweep.pickups, self.ability_pickups.len(), |i| self.pickup_bounds(i));
        #[cfg(debug_assertions)]
        {
            let bits = |b: &[Vec2; 2]| [b[0].x, b[0].y, b[1].x, b[1].y].map(|v| v.to_f64().to_bits());
            let check = |list: &[[Vec2; 2]], f: &dyn Fn(usize) -> Swept| {
                for (i, b) in list.iter().enumerate() {
                    debug_assert_eq!(bits(b), bits(&self.current_bounds(f(i), data)));
                }
            };
            check(&sweep.items, &|i| Swept::Part(Part::Item(i)));
            check(&sweep.shots, &|i| Swept::Part(Part::Shot(i)));
            check(&sweep.nodes, &|i| Swept::Part(Part::Node(i)));
            check(&sweep.engines, &|i| Swept::Part(Part::Engine(i)));
            let check_taken = |list: &[[Vec2; 2]], taken: &[bool], f: &dyn Fn(usize) -> Swept| {
                for (i, b) in list.iter().enumerate() {
                    if taken[i] {
                        debug_assert_eq!(bits(b), bits(&self.current_bounds(f(i), data)));
                    }
                }
            };
            check_taken(&sweep.spikes, &sweep.spikes_taken, &|i| Swept::Part(Part::Spike(i)));
            check_taken(&sweep.missiles, &sweep.missiles_taken, &|i| Swept::Part(Part::Missile(i)));
            check(&sweep.players, &|i| Swept::Player(i));
            check(&sweep.clones, &|i| Swept::Clone(i));
            check(&sweep.arrows, &|i| Swept::Arrow(i));
            check(&sweep.duplicated_rocks, &|i| Swept::DuplicatedRock(i));
            check(&sweep.pickups, &|i| Swept::Pickup(i));
        }
        // `detphysics_collider` looks a row up in each pool that can own one
        // and falls back to `row_collider`; a row no pool names takes the
        // fallback directly. Only a smoke cloud's, a blast's or an orb's row
        // is read back (`swept_bounds`), and no pool that owns rows names one
        // of those, so their rows are bounded from `row_collider` alone; a
        // row taken here that a pool does own is bounded but never read.
        // Debug builds mark the rows a pool owns and
        // the rows taken here, and `swept_bounds` checks it reads only taken,
        // unowned rows; that is why release neither marks the owned rows nor
        // clears the rest to NOWHERE: a row not taken this pass is never
        // read. Rows are only ever appended, so one a cloud or blast takes
        // later in the pass is past `rows` and reads None either way: bound
        // just the rows those pools hold now.
        #[cfg(debug_assertions)]
        {
            let owned = &mut sweep.owned;
            owned.clear();
            owned.resize(self.entities.len(), false);
            let mut mark = |entity: Option<usize>| {
                if let Some(flag) = entity.and_then(|entity| owned.get_mut(entity)) {
                    *flag = true;
                }
            };
            for node in &self.tesla_nodes {
                mark(node.entity);
                mark(node.line_entity);
            }
            self.engines.iter().for_each(|engine| mark(engine.entity));
            self.spike_attacks.iter().for_each(|spike| mark(spike.entity));
            self.abilityless_clones.iter().for_each(|clone| mark(clone.entity));
            self.ability_pickups.iter().for_each(|pickup| mark(pickup.entity));
            self.arrows.iter().for_each(|arrow| mark(arrow.entity));
            self.thrown_items.iter().for_each(|item| mark(item.entity));
            self.players.iter().for_each(|slot| mark(Some(Self::hurtbox_entity(slot))));
            sweep.taken.clear();
            sweep.taken.resize(self.entities.len(), false);
        }
        let rows = &mut sweep.rows;
        let len = self.entities.len();
        rows.truncate(len);
        rows.resize(len, NOWHERE);
        let mut take = |entity: Option<usize>| {
            if let Some(i) = entity.filter(|&i| i < len) {
                #[cfg(debug_assertions)]
                {
                    sweep.taken[i] = true;
                }
                // A row's own collider is its shape, so its box is the
                // shape's whether or not the row is in DetPhysics.
                rows[i] = self.row_shape_bounds(i);
                debug_assert!(
                    rows[i] == self.row_collider(i).map_or_else(|| self.row_shape_bounds(i), collider_bounds)
                );
            }
        };
        // A row is read back only for a part of this pass's groups, and each
        // of those needs its owner active (`cloud_part`, `blast_part`): a
        // Drill's boxes only for the active slot of a player in Drill. Nothing
        // between here and the groups activates a cloud, a blast or an orb or
        // changes a player's ability, so the inactive ones are left untaken
        // (debug builds check every row read was taken).
        let live = |active: bool, entity: Option<usize>| entity.filter(|_| active);
        // Nothing between here and the groups writes a cloud's row (a
        // correction moves only the cloud's own position), so `swept_bounds`
        // bounds a cloud's row when a group reads it, from the row as it is
        // here; debug builds take them here too and check that.
        #[cfg(debug_assertions)]
        self.smoke_clouds.iter().for_each(|cloud| take(live(cloud.active, cloud.entity)));
        self.grenade_explosions.iter().for_each(|blast| take(live(blast.active, blast.entity)));
        self.missile_explosions.iter().for_each(|blast| take(live(blast.active, blast.entity)));
        self.smoke_explosions.iter().for_each(|blast| take(live(blast.active, blast.entity)));
        self.meteor_smears.iter().for_each(|blast| take(live(blast.active, blast.entity)));
        self.beam_orbs.iter().for_each(|orb| take(live(orb.active, orb.entity)));
        for slot in &self.players {
            if slot.active != ActiveAbility::Drill {
                continue;
            }
            for &entity in &slot.drill_hitbox_entities[slot.active_slot] {
                if entity != usize::MAX {
                    take(Some(entity));
                }
            }
        }
    }

    /// The collider's bounding box where it is now.
    fn current_bounds(&self, collider: Swept, data: &data::RolloutData) -> [Vec2; 2] {
        let _ = data;
        match collider {
            Swept::Part(Part::Spike(i)) => self.spike_bounds(i),
            Swept::Part(Part::Rock(i)) => self.player_bounds(i),
            Swept::Part(Part::Pickup(i)) => self.pickup_bounds(i),
            Swept::Part(Part::DuplicatedRock(i)) => self.duplicated_rock_bounds(i),
            Swept::Part(Part::Arrow(i)) => self.arrow_bounds(i),
            Swept::Part(Part::Body(i)) => self.player_bounds(i),
            Swept::Part(Part::Clone(i)) => self.clone_bounds(i),
            Swept::Part(Part::Cloud(i)) => match self.smoke_clouds[i].entity {
                Some(entity) => self.row_bounds(entity),
                None => NOWHERE,
            },
            // DetPhysics swept a beam's box before UpdateBeams resized it.
            Swept::Part(Part::Explosion(Blast::Beam(i))) => {
                self.beam_pool[i].swept_collider.map_or(NOWHERE, box_bounds)
            }
            Swept::Part(Part::Explosion(blast)) => self.row_bounds(self.blast_entity(blast)),
            Swept::Part(part) => self.part_collider(part).bounds(),
            Swept::Player(index) => self.player_bounds(index),
            Swept::Clone(index) => self.clone_bounds(index),
            Swept::Arrow(index) => self.arrow_bounds(index),
            Swept::DuplicatedRock(index) => self.duplicated_rock_bounds(index),
            Swept::Pickup(index) => self.pickup_bounds(index),
        }
    }

    /// A spike's box; one whose wall is gone a point at its collider centre.
    #[inline]
    fn spike_bounds(&self, i: usize) -> [Vec2; 2] {
        let spike = &self.spike_attacks[i];
        if spike.platform_index >= self.platform_dynamics.len() {
            let center = spike.collider_center;
            return [center, center];
        }
        box_bounds(Self::spike_attack_box_in(&self.platform_dynamics, &self.entity_inverse_extents, *spike))
    }

    /// The player's registered Player-layer shape.
    #[inline]
    fn player_bounds(&self, index: usize) -> [Vec2; 2] {
        let slot = &self.players[index];
        if slot.active == ActiveAbility::Rock {
            // A Blink tunnel or teleport disables the form's
            // GameObject, taking its body out of DetPhysics.
            if slot.blink_tunnel_active || slot.teleport_effect_active {
                return NOWHERE;
            }
            // Ability.UpdateSim restores Player.Scale before
            // DetPhysics; BounceBall's pulse is applied only in the
            // later LateUpdate.
            let radius = slot.rock_circle().circle_radius;
            return bounds(slot.base.position, Vec2::new(radius, radius));
        }
        let entity = Self::hurtbox_entity(slot);
        if matches!(slot.active, ActiveAbility::None | ActiveAbility::Dash) || entity >= self.entities.len() {
            return box_bounds(slot.base.synced);
        }
        self.row_bounds(entity)
    }

    #[inline]
    fn clone_bounds(&self, index: usize) -> [Vec2; 2] {
        // A clone's row that is inactive or tunnelled by Blink is out
        // of DetPhysics.
        let clone = &self.abilityless_clones[index];
        match clone.entity {
            Some(entity) => self.detphysics_collider(entity).map_or(NOWHERE, collider_bounds),
            None => box_bounds(clone.base.synced),
        }
    }

    #[inline]
    fn arrow_bounds(&self, index: usize) -> [Vec2; 2] {
        let arrow = &self.arrows[index];
        let radius = arrow.collider_radius;
        bounds(arrow.position, Vec2::new(radius, radius))
    }

    #[inline]
    fn duplicated_rock_bounds(&self, index: usize) -> [Vec2; 2] {
        let rock = &self.duplicated_rocks[index];
        let radius = rock.collider_radius;
        bounds(rock.position, Vec2::new(radius, radius))
    }

    #[inline]
    fn pickup_bounds(&self, index: usize) -> [Vec2; 2] {
        let pickup = &self.ability_pickups[index];
        bounds(pickup.synced_center, Vec2::new(pickup.radius, pickup.radius))
    }

    /// A row's box: a slime's, form's or clone's collider is where
    /// DetPhysics last synced it, which an explicit sync since the export
    /// moves.
    #[inline]
    fn row_bounds(&self, entity: usize) -> [Vec2; 2] {
        match self.detphysics_collider(entity) {
            Some(collider) => collider_bounds(collider),
            None => self.row_shape_bounds(entity),
        }
    }

    /// A row's box from its own shape, for a row with no DetPhysics collider.
    fn row_shape_bounds(&self, entity: usize) -> [Vec2; 2] {
        let row = self.entities[entity];
        let center = Vec2::new(Fix::from_raw(row.x), Fix::from_raw(row.y));
        if row.kind == 0 {
            let radius = Fix::from_raw(row.radius);
            return bounds(center, Vec2::new(radius, radius));
        }
        let right = Vec2::new(Fix::from_raw(row.right_x), Fix::from_raw(row.right_y));
        let up = Vec2::new(Fix::from_raw(row.up_x), Fix::from_raw(row.up_y));
        bounds(center, box_half(right, up))
    }

    /// The collider's bounding box in this pass's SortAndSweep; None for one
    /// registered after it, which meets nothing until the next pass.
    pub(super) fn swept_bounds(&self, collider: Swept) -> Option<[Vec2; 2]> {
        self.swept_bounds_inlined(collider)
    }

    /// A wall group's part with its `swept_bounds`, pushed only if
    /// SortAndSweep bounded it: a part registered after it meets nothing
    /// this pass (DetPhysics.cs:2169-2231). Inlined where the part's kind is
    /// known, so its box comes straight from its pool's list.
    #[inline(always)]
    pub(super) fn push_swept(&self, parts: &mut Vec<(Part, [Vec2; 2])>, part: Part) {
        if let Some(bounds) = self.swept_bounds_inlined(Swept::Part(part)) {
            parts.push((part, bounds));
        }
    }

    /// This pass's SortAndSweep boxes of the players' colliders, by player
    /// (`swept_bounds` of a `Part::Body` or `Part::Rock`); None before the
    /// pass's sweep, where `swept_bounds` takes each where it is.
    #[inline(always)]
    pub(super) fn swept_players(&self) -> Option<&[[Vec2; 2]]> {
        let sweep = self.scratch.sweep.as_deref().filter(|sweep| sweep.tick == Some(self.tick))?;
        Some(&sweep.players)
    }

    #[inline(always)]
    pub(super) fn swept_bounds_inlined(&self, collider: Swept) -> Option<[Vec2; 2]> {
        let Some(sweep) = self.scratch.sweep.as_deref().filter(|sweep| sweep.tick == Some(self.tick)) else {
            return Some(self.current_bounds(collider, data::get()));
        };
        let list = match collider {
            Swept::Part(Part::Item(i)) => sweep.items.get(i),
            Swept::Part(Part::Shot(i)) => sweep.shots.get(i),
            Swept::Part(Part::Node(i)) => sweep.nodes.get(i),
            Swept::Part(Part::Engine(i)) => sweep.engines.get(i),
            Swept::Part(Part::Spike(i)) => {
                #[cfg(debug_assertions)]
                debug_assert!(sweep.spikes_taken.get(i).is_none_or(|&taken| taken), "an untaken spike's sweep box was read");
                sweep.spikes.get(i)
            }
            Swept::Part(Part::Missile(i)) => {
                #[cfg(debug_assertions)]
                debug_assert!(sweep.missiles_taken.get(i).is_none_or(|&taken| taken), "an untaken missile's sweep box was read");
                sweep.missiles.get(i)
            }
            Swept::Part(Part::Rock(i)) => sweep.players.get(i),
            Swept::Part(Part::Pickup(i)) => sweep.pickups.get(i),
            Swept::Part(Part::DuplicatedRock(i)) => sweep.duplicated_rocks.get(i),
            Swept::Part(Part::Arrow(i)) => sweep.arrows.get(i),
            Swept::Part(Part::Body(i)) => sweep.players.get(i),
            Swept::Part(Part::Clone(i)) => sweep.clones.get(i),
            // `take_sweep` leaves a cloud's row to be bounded here, on a row
            // it would have bounded: one there was when it ran.
            Swept::Part(Part::Cloud(i)) => {
                let entity = self.smoke_clouds[i].entity.filter(|&entity| entity < sweep.rows.len())?;
                let bounds = self.row_shape_bounds(entity);
                #[cfg(debug_assertions)]
                {
                    debug_assert!(!sweep.owned.get(entity).is_some_and(|&owned| owned));
                    debug_assert!(sweep.taken[entity]);
                    let bits = |b: [Vec2; 2]| [b[0].x, b[0].y, b[1].x, b[1].y].map(|v| v.to_f64().to_bits());
                    debug_assert_eq!(bits(bounds), bits(sweep.rows[entity]));
                }
                return Some(bounds);
            }
            Swept::Part(Part::Explosion(Blast::Beam(i))) => {
                return Some(self.beam_pool[i].swept_collider.map_or(NOWHERE, box_bounds));
            }
            Swept::Part(Part::Explosion(blast)) => {
                let entity = self.blast_entity(blast);
                #[cfg(debug_assertions)]
                debug_assert!(!sweep.owned.get(entity).is_some_and(|&owned| owned));
                #[cfg(debug_assertions)]
                debug_assert!(sweep.taken.get(entity).is_none_or(|&taken| taken));
                sweep.rows.get(entity)
            }
            // A trigger no group moves: SortAndSweep bounded it where it is.
            Swept::Part(Part::Hook(_) | Part::Hole(_) | Part::Line(_) | Part::Water) => {
                return Some(self.current_bounds(collider, data::get()));
            }
            Swept::Player(i) => sweep.players.get(i),
            Swept::Clone(i) => sweep.clones.get(i),
            Swept::Arrow(i) => sweep.arrows.get(i),
            Swept::DuplicatedRock(i) => sweep.duplicated_rocks.get(i),
            Swept::Pickup(i) => sweep.pickups.get(i),
        };
        list.copied()
    }

    /// The collider's low edge on the sweep axis, where SortAndSweep ranks it.
    pub(super) fn swept_min(&self, collider: Swept) -> Fix {
        let low = self.swept_bounds(collider).map_or(Vec2::ZERO, |b| b[0]);
        if self.sweep_x() { low.x } else { low.y }
    }
}
