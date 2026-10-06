//! DetPhysics.Simulate finds every pair once, in SortAndSweep, on the
//! bounding boxes the colliders had before any contact group corrected
//! anything (DetPhysics.cs:2169-2231); every group then works from that one
//! pair list. The pair set is the colliders whose boxes overlap strictly on
//! both axes, so a predicate over one snapshot of those boxes is that list.

use super::*;

/// A collider in SortAndSweep, by the collection that owns it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Swept {
    Part(Part),
    /// The player's registered Player-layer shape: the slime's box, or its
    /// form's collider (a Rock's circle).
    Player(usize),
    Clone(usize),
    Arrow(usize),
    DuplicatedRock(usize),
    Pickup(usize),
    /// A row with no other owner here, such as an Explosion or an orb.
    Row(usize),
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
    /// take_sweep's rows some pool could own.
    owned: Vec<bool>,
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
        let fill = |list: &mut Vec<[Vec2; 2]>, count: usize, f: &dyn Fn(usize) -> [Vec2; 2]| {
            list.clear();
            list.extend((0..count).map(f));
        };
        fill(&mut sweep.items, self.thrown_items.len(), &|i| self.current_bounds(Swept::Part(Part::Item(i)), data));
        fill(&mut sweep.shots, self.portal_shots.len(), &|i| self.current_bounds(Swept::Part(Part::Shot(i)), data));
        fill(&mut sweep.nodes, self.tesla_nodes.len(), &|i| self.current_bounds(Swept::Part(Part::Node(i)), data));
        fill(&mut sweep.engines, self.engines.len(), &|i| self.current_bounds(Swept::Part(Part::Engine(i)), data));
        fill(&mut sweep.spikes, self.spike_attacks.len(), &|i| self.current_bounds(Swept::Part(Part::Spike(i)), data));
        fill(&mut sweep.missiles, self.guided_missiles.len(), &|i| {
            self.current_bounds(Swept::Part(Part::Missile(i)), data)
        });
        fill(&mut sweep.players, self.players.len(), &|i| self.current_bounds(Swept::Player(i), data));
        fill(&mut sweep.clones, self.abilityless_clones.len(), &|i| self.current_bounds(Swept::Clone(i), data));
        fill(&mut sweep.arrows, self.arrows.len(), &|i| self.current_bounds(Swept::Arrow(i), data));
        fill(&mut sweep.duplicated_rocks, self.duplicated_rocks.len(), &|i| {
            self.current_bounds(Swept::DuplicatedRock(i), data)
        });
        fill(&mut sweep.pickups, self.ability_pickups.len(), &|i| self.current_bounds(Swept::Pickup(i), data));
        // `detphysics_collider` looks a row up in each pool that can own one
        // and falls back to `row_collider`; a row no pool names takes the
        // fallback directly. Nothing moves while the boxes are taken, so one
        // pass over the pools marks every row they could answer for.
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
        let owned = &sweep.owned;
        sweep.rows.clear();
        sweep.rows.extend((0..self.entities.len()).map(|i| {
            if owned[i] {
                self.current_bounds(Swept::Row(i), data)
            } else {
                self.row_collider(i).map_or_else(|| self.row_shape_bounds(i), collider_bounds)
            }
        }));
    }

    /// The collider's bounding box where it is now.
    fn current_bounds(&self, collider: Swept, data: &data::RolloutData) -> [Vec2; 2] {
        match collider {
            Swept::Part(Part::Spike(i))
                if self.spike_attacks[i].platform_index >= self.platform_dynamics.len() =>
            {
                let center = self.spike_attacks[i].collider_center;
                [center, center]
            }
            Swept::Part(Part::Rock(i)) => self.current_bounds(Swept::Player(i), data),
            Swept::Part(Part::Pickup(i)) => self.current_bounds(Swept::Pickup(i), data),
            Swept::Part(Part::DuplicatedRock(i)) => self.current_bounds(Swept::DuplicatedRock(i), data),
            Swept::Part(Part::Arrow(i)) => self.current_bounds(Swept::Arrow(i), data),
            Swept::Part(Part::Body(i)) => self.current_bounds(Swept::Player(i), data),
            Swept::Part(Part::Clone(i)) => self.current_bounds(Swept::Clone(i), data),
            Swept::Part(Part::Cloud(i)) => match self.smoke_clouds[i].entity {
                Some(entity) => self.current_bounds(Swept::Row(entity), data),
                None => NOWHERE,
            },
            // DetPhysics swept a beam's box before UpdateBeams resized it.
            Swept::Part(Part::Explosion(Blast::Beam(i))) => {
                self.beam_pool[i].swept_collider.map_or(NOWHERE, box_bounds)
            }
            Swept::Part(Part::Explosion(blast)) => self.current_bounds(Swept::Row(self.blast_entity(blast)), data),
            Swept::Part(part) => self.part_collider(part).bounds(),
            Swept::Player(index) => {
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
                if matches!(slot.active, ActiveAbility::None | ActiveAbility::Dash)
                    || entity >= self.entities.len()
                {
                    return box_bounds(slot.base.synced);
                }
                self.current_bounds(Swept::Row(entity), data)
            }
            Swept::Clone(index) => {
                // A clone's row that is inactive or tunnelled by Blink is out
                // of DetPhysics.
                let clone = &self.abilityless_clones[index];
                match clone.entity {
                    Some(entity) => self.detphysics_collider(entity).map_or(NOWHERE, collider_bounds),
                    None => box_bounds(clone.base.synced),
                }
            }
            Swept::Arrow(index) => {
                let arrow = &self.arrows[index];
                let radius = arrow.collider_radius;
                bounds(arrow.position, Vec2::new(radius, radius))
            }
            Swept::DuplicatedRock(index) => {
                let rock = &self.duplicated_rocks[index];
                let radius = rock.collider_radius;
                bounds(rock.position, Vec2::new(radius, radius))
            }
            Swept::Pickup(index) => {
                let pickup = &self.ability_pickups[index];
                bounds(pickup.synced_center, Vec2::new(pickup.radius, pickup.radius))
            }
            Swept::Row(entity) => {
                // A slime's, form's or clone's collider is where DetPhysics
                // last synced it, which an explicit sync since the export moves.
                match self.detphysics_collider(entity) {
                    Some(collider) => collider_bounds(collider),
                    None => self.row_shape_bounds(entity),
                }
            }
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
        let sweep = &self.scratch.sweep;
        if sweep.tick != Some(self.tick) {
            return Some(self.current_bounds(collider, data::get()));
        }
        let list = match collider {
            Swept::Part(Part::Item(i)) => sweep.items.get(i),
            Swept::Part(Part::Shot(i)) => sweep.shots.get(i),
            Swept::Part(Part::Node(i)) => sweep.nodes.get(i),
            Swept::Part(Part::Engine(i)) => sweep.engines.get(i),
            Swept::Part(Part::Spike(i)) => sweep.spikes.get(i),
            Swept::Part(Part::Missile(i)) => sweep.missiles.get(i),
            Swept::Part(Part::Rock(i)) => sweep.players.get(i),
            Swept::Part(Part::Pickup(i)) => sweep.pickups.get(i),
            Swept::Part(Part::DuplicatedRock(i)) => sweep.duplicated_rocks.get(i),
            Swept::Part(Part::Arrow(i)) => sweep.arrows.get(i),
            Swept::Part(Part::Body(i)) => sweep.players.get(i),
            Swept::Part(Part::Clone(i)) => sweep.clones.get(i),
            Swept::Part(Part::Cloud(i)) => self.smoke_clouds[i].entity.and_then(|entity| sweep.rows.get(entity)),
            Swept::Part(Part::Explosion(Blast::Beam(i))) => {
                return Some(self.beam_pool[i].swept_collider.map_or(NOWHERE, box_bounds));
            }
            Swept::Part(Part::Explosion(blast)) => sweep.rows.get(self.blast_entity(blast)),
            // A trigger no group moves: SortAndSweep bounded it where it is.
            Swept::Part(Part::Hook(_) | Part::Hole(_) | Part::Line(_) | Part::Water) => {
                return Some(self.current_bounds(collider, data::get()));
            }
            Swept::Player(i) => sweep.players.get(i),
            Swept::Clone(i) => sweep.clones.get(i),
            Swept::Arrow(i) => sweep.arrows.get(i),
            Swept::DuplicatedRock(i) => sweep.duplicated_rocks.get(i),
            Swept::Pickup(i) => sweep.pickups.get(i),
            Swept::Row(i) => sweep.rows.get(i),
        };
        list.copied()
    }

    /// The collider's low edge on the sweep axis, where SortAndSweep ranks it.
    pub(super) fn swept_min(&self, collider: Swept) -> Fix {
        let low = self.swept_bounds(collider).map_or(Vec2::ZERO, |b| b[0]);
        if self.sweep_x() { low.x } else { low.y }
    }
}
