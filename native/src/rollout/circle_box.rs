//! DetPhysics' Circle/Box group (DetPhysics.cs:3090-3129), for the colliders
//! the rollout lists here: every registered circle against every registered
//! box the broadphase paired, all manifolds built before any resolves, each
//! pair's callbacks queued circle side first (DetPhysics.cs:2405-2437).
use super::*;
use crate::physics::shape::Collider;

/// A DetPhysics collider, by the collection that owns it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Part {
    Item(usize),
    Shot(usize),
    Node(usize),
    Engine(usize),
    Spike(usize),
    Missile(usize),
}

impl RolloutWorld {
    /// Its collider now, after the corrections of earlier groups.
    fn part_collider(&self, part: Part, data: &data::RolloutData) -> Collider {
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
            Part::Engine(i) => Collider::Box(self.engines[i].rocket_box()),
            Part::Spike(i) => {
                Collider::Box(Self::spike_attack_box_in(&self.platform_dynamics, self.spike_attacks[i]))
            }
            Part::Missile(i) => Collider::Box(self.guided_missiles[i].collider()),
        }
    }

    /// Its collider where the broadphase took its bounding box.
    pub(super) fn part_swept(&self, part: Part, data: &data::RolloutData) -> Collider {
        let mut collider = self.part_collider(part, data);
        let swept = match part {
            Part::Item(i) => Some(self.thrown_items[i].swept_position),
            Part::Node(i) => Some(self.tesla_nodes[i].swept_center),
            Part::Engine(i) => self.scratch.engine_swept.get(i).copied(),
            Part::Spike(i) => self.scratch.spike_swept.get(i).copied(),
            Part::Missile(i) => Some(self.guided_missiles[i].swept_position),
            Part::Shot(_) => None,
        };
        if let Some(center) = swept {
            collider.set_center(center);
        }
        collider
    }

    /// Its place in SortAndSweep's order.
    pub(super) fn part_rank(&self, part: Part, data: &data::RolloutData) -> SweepRank {
        let min = self.part_swept(part, data).bounds()[0];
        sweep_rank(&self.entities, if self.sweep_x() { min.x } else { min.y }, self.part_entity(part))
    }

    /// A wall's place in SortAndSweep's order, from the collider the
    /// broadphase bounded.
    pub(super) fn wall_rank(&self, platform: usize) -> SweepRank {
        let swept = self
            .scratch
            .callback_colliders
            .get(platform)
            .filter(|_| self.scratch.callback_tick == Some(self.tick))
            .copied()
            .unwrap_or_else(|| self.circle_collider(platform));
        let min = Collider::RoundedRect(swept).bounds()[0];
        let entity = Some(self.platform_entities[platform]);
        sweep_rank(&self.entities, if self.sweep_x() { min.x } else { min.y }, entity)
    }

    fn part_entity(&self, part: Part) -> Option<usize> {
        match part {
            Part::Item(i) => self.thrown_items[i].entity,
            Part::Shot(i) => self.portal_shots[i].entity,
            Part::Node(i) => self.tesla_nodes[i].entity,
            Part::Engine(i) => self.engines[i].entity,
            Part::Spike(i) => self.spike_attacks[i].entity,
            Part::Missile(i) => self.guided_missiles[i].entity,
        }
    }

    /// The platform whose CompositeBody holds a child collider.
    fn part_composite(&self, part: Part) -> Option<usize> {
        match part {
            Part::Node(i) => Some(self.tesla_nodes[i].platform_index),
            Part::Engine(i) => Some(self.engines[i].platform_index),
            Part::Spike(i) => Some(self.spike_attacks[i].platform_index),
            Part::Item(_) | Part::Shot(_) | Part::Missile(_) => None,
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
        }
        if let (Part::Item(_) | Part::Shot(_) | Part::Missile(_), Some(entity)) = (part, self.part_entity(part)) {
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
        }
    }

    /// DPhysicsCircle/DPhysicsBox.AddForceAtPosition: the impulse at `point`
    /// with its arm from the FixTransform, on a child's CompositeBody
    /// (DPhysicsCircle.cs:377-392, DPhysicsBox.cs:436-451).
    pub(super) fn part_add_force_at(&mut self, part: Part, force: Vec2, point: Vec2) {
        let arm = (point - self.part_root(part)).normalized_safe();
        let body = match (part, self.part_composite(part)) {
            (_, Some(p)) => &mut self.platform_dynamics[p].body,
            (Part::Item(i), None) => &mut self.thrown_items[i].body,
            (Part::Shot(i), None) => &mut self.portal_shots[i].body,
            (Part::Missile(i), None) => &mut self.guided_missiles[i].body,
            _ => unreachable!(),
        };
        body.apply_impulse(force, arm);
        let body = *body;
        if let (Part::Item(_) | Part::Shot(_) | Part::Missile(_), Some(entity)) = (part, self.part_entity(part)) {
            let row = &mut self.entities[entity];
            row.vx = body.velocity.x.raw();
            row.vy = body.velocity.y.raw();
            row.angular_velocity = body.angular_velocity.raw();
        }
    }

    /// A platform child DetPhysics still simulates: not carried off by
    /// MachoThrow2, its platform and GameObject active.
    fn child_registered(&self, active: bool, carried: bool, platform: usize, entity: Option<usize>) -> bool {
        active
            && !carried
            && platform < self.platforms.len()
            && !self.attached_hidden(platform, entity)
            && entity.is_none_or(|entity| self.entities[entity].flags & E_ACTIVE != 0)
    }

    /// DetPhysics' Circle/Box group. Rigid pairs resolve in sweep order with
    /// PositionalCorrection on each side's own inverse mass; every pair
    /// queues both sides' OnCollide, which run after the last group.
    pub(super) fn circle_box_group(&mut self, data: &data::RolloutData) {
        let mut parts = std::mem::take(&mut self.scratch.cb_parts);
        parts.clear();
        for (i, item) in self.thrown_items.iter().enumerate() {
            if item.active
                && item.entity.is_some_and(|entity| self.entities[entity].flags & E_ACTIVE != 0)
                && item.callback_registered_tick != self.tick
            {
                parts.push(Part::Item(i));
            }
        }
        for (i, shot) in self.portal_shots.iter().enumerate() {
            if shot.active && shot.registered && !self.blink_object_hidden(shot.entity) {
                parts.push(Part::Shot(i));
            }
        }
        for (i, node) in self.tesla_nodes.iter().enumerate() {
            if self.child_registered(node.active, node.macho_carried, node.platform_index, node.entity) {
                parts.push(Part::Node(i));
            }
        }
        let circles = parts.len();
        for (i, engine) in self.engines.iter().enumerate() {
            if self.child_registered(engine.active, engine.macho_carried, engine.platform_index, engine.entity) {
                parts.push(Part::Engine(i));
            }
        }
        for (i, spike) in self.spike_attacks.iter().enumerate() {
            if self.child_registered(spike.active, spike.macho_carried, spike.platform_index, spike.entity) {
                parts.push(Part::Spike(i));
            }
        }
        let sweep_x = self.sweep_x();
        let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
        let other = |v: Vec2| if sweep_x { v.y } else { v.x };
        let mut contacts = std::mem::take(&mut self.scratch.cb_contacts);
        contacts.clear();
        for &circle in &parts[..circles] {
            let circle_bounds = self.part_swept(circle, data).bounds();
            let circle_rank = sweep_rank(&self.entities, axis(circle_bounds[0]), self.part_entity(circle));
            for &part in &parts[circles..] {
                // SameCompositeAs: a node never meets its own platform's children.
                if self.part_composite(circle) == self.part_composite(part) {
                    continue;
                }
                let bounds = self.part_swept(part, data).bounds();
                let rank = sweep_rank(&self.entities, axis(bounds[0]), self.part_entity(part));
                // SortAndSweep pairs the later collider with each earlier one
                // whose bounds still reach it (DetPhysics.cs:2169-2231).
                let (earlier, later) =
                    if circle_rank < rank { (circle_bounds, bounds) } else { (bounds, circle_bounds) };
                if axis(earlier[1]) <= axis(later[0])
                    || other(earlier[1]) <= other(later[0])
                    || other(earlier[0]) >= other(later[1])
                {
                    continue;
                }
                if let Some(contact) = self.part_collider(circle, data).contact(self.part_collider(part, data)) {
                    let (later, earlier) = sweep_pair(circle_rank, rank);
                    contacts.push(((GROUP_CB, later, earlier), circle, part, contact));
                }
            }
        }
        contacts.sort_by_key(|&(key, ..)| key);
        for &(key, circle, part, contact) in &contacts {
            // PrepareCollisionCallBacks: the circle's OnCollide, then the box's.
            if let Part::Item(item) = circle
                && self.thrown_items[item].ability == ABILITY_SMOKE
            {
                // SmokeGrenadeExplode2 goes off on any layer but a
                // Projectile that does not ignite explosives.
                self.scratch.callback_hits.push((key, CallbackHit::ItemDetonate { item, count: 1 }));
            }
            if let Part::Spike(spike) = part {
                let hit = CallbackHit::SpikeShove { spike, target: circle, point: contact.point };
                self.scratch.callback_hits.push((key, hit));
            }
        }
        for &(_, circle, part, contact) in &contacts {
            let (circle_own, part_own) = (self.part_body(circle), self.part_body(part));
            if circle_own.inverse_mass < Fix::ZERO || part_own.inverse_mass < Fix::ZERO {
                continue;
            }
            let (mut circle_body, circle_position) = self.part_resolve_body(circle);
            let (mut part_body, part_position) = self.part_resolve_body(part);
            resolve(contact, &mut circle_body, &mut part_body, circle_position, part_position);
            let correction =
                positional_correction(contact, circle_own.inverse_mass, part_own.inverse_mass, false);
            self.set_part_body(circle, circle_body, correction[0]);
            self.set_part_body(part, part_body, correction[1]);
        }
        self.scratch.cb_parts = parts;
        self.scratch.cb_contacts = contacts;
    }
}
