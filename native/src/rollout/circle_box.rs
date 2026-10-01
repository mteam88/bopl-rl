//! DetPhysics' Circle/Circle and Circle/Box groups (DetPhysics.cs:3052-3129),
//! for the colliders the rollout lists here: every pair of registered
//! colliders the broadphase found on layers the collision matrix pairs, all
//! of a group's manifolds built before any resolves, each pair's callbacks
//! queued first side first (DetPhysics.cs:2405-2437).
use super::*;
use super::rope_abilities::HookTarget;
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
}

/// Arrow.StickTo: Player, item, wall, LethalTerrain and NonLethalTerrain
/// (Arrow.prefab m_Bits 2166016).
const ARROW_STICK_TO: u32 = 2166016;

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
            Part::Engine(i) => Collider::Box(self.engines[i].rocket_box()),
            Part::Spike(i) => {
                Collider::Box(Self::spike_attack_box_in(&self.platform_dynamics, self.spike_attacks[i]))
            }
            Part::Missile(i) => Collider::Box(self.guided_missiles[i].collider()),
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
        }
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
        }
    }

    /// Its place in SortAndSweep's order.
    pub(super) fn part_rank(&self, part: Part) -> SweepRank {
        sweep_rank(&self.entities, self.swept_min(Swept::Part(part)), self.part_entity(part))
    }

    /// Any collider's place in SortAndSweep's order.
    pub(super) fn swept_rank(&self, collider: Swept) -> SweepRank {
        let entity = match collider {
            Swept::Part(part) => self.part_entity(part),
            Swept::Player(i) => Some(Self::hurtbox_entity(&self.players[i])),
            Swept::Clone(i) => self.abilityless_clones[i].entity,
            Swept::Arrow(i) => self.arrows[i].entity,
            Swept::DuplicatedRock(i) => self.duplicated_rocks[i].entity,
            Swept::Pickup(i) => self.ability_pickups[i].entity,
            Swept::Row(entity) => Some(entity),
            Swept::Water => None,
        };
        sweep_rank(&self.entities, self.swept_min(collider), entity)
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
            Part::Rock(i) => Some(self.players[i].rock_entity),
            Part::Pickup(i) => self.ability_pickups[i].entity,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].entity,
            Part::Hook(r) => self.ropes[r].hook_entity,
            Part::Arrow(i) => self.arrows[i].entity,
            Part::Hole(i) => self.black_holes[i].entity,
        }
    }

    /// The platform whose CompositeBody holds a child collider.
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
            | Part::Hole(_) => None,
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
            Part::Rock(i) => {
                let slot = &mut self.players[i];
                slot.rock.body = body;
                slot.base.position += delta;
                slot.base.velocity = body.velocity;
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
            // A RopeHook's circle has no BoplBody, so nothing integrates it.
            Part::Hook(_) => {}
            Part::Node(_) | Part::Engine(_) | Part::Spike(_) | Part::Hole(_) => unreachable!(),
        }
    }

    /// A part in this Simulate's pairs: DetPhysics.IsActive on the live
    /// hierarchy, so a wall a QuantumTunnel showed again this pass brings its
    /// children back before their rows are republished, and an item a Blink
    /// tunnel holds stays out.
    fn child_registered(&self, entity: Option<usize>) -> bool {
        entity.is_some_and(|entity| self.collides(entity))
    }

    /// DetPhysics' Circle/Box group. Rigid pairs resolve in sweep order with
    /// PositionalCorrection on each side's own inverse mass; every pair
    /// queues both sides' OnCollide, which run after the last group.
    pub(super) fn circle_box_group(&mut self, data: &data::RolloutData) {
        let mut parts = std::mem::take(&mut self.scratch.cb_parts);
        parts.clear();
        for (i, item) in self.thrown_items.iter().enumerate() {
            if item.active
                && self.child_registered(item.entity)
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
            if self.child_registered(node.entity) {
                parts.push(Part::Node(i));
            }
        }
        for i in 0..self.players.len() {
            if self.players[i].active == ActiveAbility::Rock && self.playable(i) {
                parts.push(Part::Rock(i));
            }
        }
        for i in 0..self.arrows.len() {
            if self.arrow_registered(i) {
                parts.push(Part::Arrow(i));
            }
        }
        for (i, hole) in self.black_holes.iter().enumerate() {
            if hole.active && !self.blink_object_hidden(hole.entity) {
                parts.push(Part::Hole(i));
            }
        }
        let circles = parts.len();
        for (i, engine) in self.engines.iter().enumerate() {
            if self.child_registered(engine.entity) {
                parts.push(Part::Engine(i));
            }
        }
        for (i, spike) in self.spike_attacks.iter().enumerate() {
            if self.child_registered(spike.entity) {
                parts.push(Part::Spike(i));
            }
        }
        for i in 0..self.guided_missiles.len() {
            if self.guided_missile_live(i) {
                parts.push(Part::Missile(i));
            }
        }
        let mut contacts = std::mem::take(&mut self.scratch.cb_contacts);
        contacts.clear();
        for &circle in &parts[..circles] {
            let circle_rank = self.part_rank(circle);
            for &part in &parts[circles..] {
                // SameCompositeAs: a node never meets its own platform's
                // children; an item, shot or missile has no composite.
                if self.part_composite(circle).is_some_and(|c| Some(c) == self.part_composite(part))
                    || !data.layers_collide(self.part_layer(circle, data), self.part_layer(part, data))
                    || !self.swept_pair(Swept::Part(circle), Swept::Part(part))
                {
                    continue;
                }
                let rank = self.part_rank(part);
                if let Some(contact) = self.part_collider(circle).contact(self.part_collider(part)) {
                    let (later, earlier) = sweep_pair(circle_rank, rank);
                    contacts.push(((GROUP_CB, later, earlier), circle, part, contact));
                    self.note_arrow_landing(circle, part, (GROUP_CB, later, earlier), data);
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
            // BlackHole.OnCollide reads the collider's inverseMass, a
            // composite child's CompositeBody's.
            if let Part::Hole(hole) = circle {
                let target = match part {
                    Part::Engine(i) => HoleTarget::Engine(i),
                    Part::Spike(i) => HoleTarget::Spike(i),
                    Part::Missile(i) => HoleTarget::Missile(i),
                    _ => unreachable!(),
                };
                let inverse_mass = match self.part_composite(part) {
                    Some(platform) => self.platform_dynamics[platform].body.inverse_mass,
                    None => self.part_body(part).inverse_mass,
                };
                let hit = CallbackHit::BlackHole { hole, target, inverse_mass };
                self.scratch.callback_hits.push((key, hit));
            }
            match part {
                Part::Spike(spike) => {
                    let hit = CallbackHit::SpikeShove { spike, target: circle, point: contact.point };
                    self.scratch.callback_hits.push((key, hit));
                }
                // Missile.OnCollide ignores RigidBodyAffector (Missile.cs:183).
                Part::Missile(missile) if self.part_layer(circle, data) != LAYER_RIGID_BODY_AFFECTOR => {
                    self.scratch.callback_hits.push((key, CallbackHit::MissileBlast { missile }));
                }
                _ => {}
            }
        }
        for &(_, circle, part, contact) in &contacts {
            self.resolve_parts(circle, part, contact);
        }
        self.scratch.cb_parts = parts;
        self.scratch.cb_contacts = contacts;
    }

    /// DetPhysics' Box/RoundedRect or Circle/RoundedRect group
    /// (DetPhysics.cs:2971-3048): every swept pair of a part with a wall makes
    /// a manifold against `walls`, the colliders as the group's narrow phase
    /// sees them, all built before any resolves. In SortAndSweep pair order
    /// each queues the part's OnCollide (PrepareCollisionCallBacks,
    /// DetPhysics.cs:2405-2437), then each pair that is no trigger resolves
    /// with ResolveCollision and PositionalCorrection on each side's own
    /// inverse mass.
    pub(super) fn wall_group(&mut self, group: u8, parts: &[Part], walls: &[RoundedRect], data: &data::RolloutData) {
        let mut contacts = std::mem::take(&mut self.scratch.wall_contacts);
        contacts.clear();
        for &part in parts {
            let collider = self.part_collider(part);
            // SortAndSweep pairs on the bounds it took before any group
            // corrected anything; a part registered after it meets nothing
            // this pass (DetPhysics.cs:2169-2231).
            let Some(bounds) = self.swept_bounds(Swept::Part(part)) else { continue };
            let rank = self.part_rank(part);
            let layer = self.part_layer(part, data);
            for wall in 0..self.platforms.len() {
                // SameCompositeAs: a child never meets its own platform. A wall
                // destroyed before this pass left in RemoveDestroyedColliders.
                // SortAndSweep pairs only layers the Physics2D matrix lets
                // collide (DetPhysics.cs:2183).
                let row = &self.entities[self.platform_entities[wall]];
                if !self.platforms[wall].active
                    || row.flags & E_DESTROYED != 0
                    || !data.layers_collide(layer, row.layer)
                    || self.part_composite(part) == Some(wall)
                    || !self.wall_swept(wall, bounds)
                {
                    continue;
                }
                let rounded = walls.get(wall).copied().unwrap_or(self.platforms[wall].collider);
                // A Box meets a RoundedRect as a RoundedRect of radius zero
                // (PhysTools.cs:104-111).
                let contact = match collider {
                    Collider::Circle(circle) => crate::physics::contact::circle_rounded(circle, rounded),
                    Collider::Box(inner) => {
                        crate::physics::contact::rounded_rounded(RoundedRect { inner, radius: Fix::ZERO }, rounded)
                    }
                    Collider::RoundedRect(_) => unreachable!(),
                };
                if let Some(contact) = contact {
                    let (later, earlier) = sweep_pair(rank, self.wall_rank(wall));
                    contacts.push(((group, later, earlier), part, wall, contact));
                }
            }
        }
        contacts.sort_by_key(|&(key, ..)| key);
        for &(key, part, wall, contact) in &contacts {
            // collision.layer is the wall's own layer (DetPhysics.cs:2405-2437).
            let wall_layer = self.entities[self.platform_entities[wall]].layer == LAYER_WALL;
            let hit = match part {
                // SmokeGrenadeExplode2.cs:63-78, Mjolnir.cs:292-318,
                // BananaPeel.cs:113-138.
                Part::Item(item) => match self.thrown_items[item].ability {
                    ABILITY_SMOKE => CallbackHit::ItemDetonate { item, count: 1 },
                    ABILITY_MJOLNIR => CallbackHit::MjolnirWall { item, wall },
                    ABILITY_TACK => CallbackHit::TackWall { item, normal: contact.normal },
                    _ => continue,
                },
                // PortalShot.OnCollide lands on the first wall-layer pair to
                // run, with its normal; the later ones find the shot landed
                // (PortalShot.cs:78-98).
                Part::Shot(shot) if wall_layer => CallbackHit::PortalShotLand { shot, normal: -contact.normal },
                // Arrow.cs:85-131.
                Part::Arrow(arrow) => CallbackHit::ArrowLand { arrow, wall: Some(wall), part: None },
                // SpikeAttack.OnCollide knocks with knockAwayWallStr only on the
                // wall layer (SpikeAttack.cs:116-141).
                Part::Spike(spike) => CallbackHit::SpikeWallKnock { spike, wall, point: contact.point, wall_layer },
                // Missile.cs:181-192.
                Part::Missile(missile) => CallbackHit::MissileBlast { missile },
                // RopeHook.cs:86-131.
                Part::Hook(rope) => CallbackHit::RopeHook { rope, target: HookTarget::Platform(wall) },
                // BlackHole.OnCollide eats a wall only when the collider the
                // manifold copied contains the hole's centre, even though the
                // two overlap much earlier (BlackHole.cs:207-210, 516-547).
                Part::Hole(hole)
                    if wall_layer
                        && walls
                            .get(wall)
                            .copied()
                            .unwrap_or(self.platforms[wall].collider)
                            .overlaps(self.black_holes[hole].position) =>
                {
                    let inverse_mass = self.platform_dynamics[wall].body.inverse_mass;
                    CallbackHit::BlackHole { hole, target: HoleTarget::Wall(wall), inverse_mass }
                }
                _ => continue,
            };
            self.scratch.callback_hits.push((key, hit));
        }
        for &(_, part, wall, contact) in &contacts {
            self.resolve_part_wall(part, wall, contact);
        }
        if !contacts.is_empty() {
            self.platforms_changed();
        }
        self.scratch.wall_contacts = contacts;
    }

    /// Whether SortAndSweep paired a part's bounds with the wall it bounded
    /// before this pass's RoundedRect/RoundedRect corrections.
    fn wall_swept(&self, wall: usize, bounds: [Vec2; 2]) -> bool {
        self.scratch
            .callback_colliders
            .get(wall)
            .filter(|_| self.scratch.callback_tick == Some(self.tick))
            .is_none_or(|&swept| sweep::swept_meet(bounds, Collider::RoundedRect(swept).bounds()))
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

    /// Arrow.OnCollide lands on a StickTo layer, but not on a Player-layer
    /// collider without a BoplBody (Arrow.cs:87-96): of the Player-layer
    /// parts only a Rock form has one.
    fn note_arrow_landing(&mut self, a: Part, b: Part, key: CallbackKey, data: &data::RolloutData) {
        let (arrow, other) = match (a, b) {
            (Part::Arrow(arrow), other) | (other, Part::Arrow(arrow)) => (arrow, other),
            _ => return,
        };
        let layer = self.part_layer(other, data);
        if ARROW_STICK_TO & (1 << layer) != 0 && (layer != LAYER_PLAYER || matches!(other, Part::Rock(_))) {
            self.scratch.arrow_landings.push((arrow, key, other));
        }
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
    /// rollout reacts to are a hook's, a Rock's bounce off another Rock and a
    /// Smoke's, which reads `cc_contacts`.
    pub(super) fn circle_circle_group(&mut self, data: &data::RolloutData) {
        let mut parts = std::mem::take(&mut self.scratch.cb_parts);
        parts.clear();
        for (i, item) in self.thrown_items.iter().enumerate() {
            if item.active
                && self.child_registered(item.entity)
                && item.callback_registered_tick != self.tick
            {
                parts.push(Part::Item(i));
            }
        }
        // A node whose platform went inactive earlier in this pass is skipped
        // by DetPhysics, though its row still shows the last publish.
        for (i, node) in self.tesla_nodes.iter().enumerate() {
            if node.active
                && !node.macho_carried
                && node.platform_index < self.platforms.len()
                && node.body.inverse_mass >= Fix::ZERO
                && self.child_registered(node.entity)
            {
                parts.push(Part::Node(i));
            }
        }
        for (i, shot) in self.portal_shots.iter().enumerate() {
            if shot.active && shot.registered && !self.blink_object_hidden(shot.entity) {
                parts.push(Part::Shot(i));
            }
        }
        for i in 0..self.players.len() {
            if self.players[i].active == ActiveAbility::Rock && self.playable(i) {
                parts.push(Part::Rock(i));
            }
        }
        for (i, pickup) in self.ability_pickups.iter().enumerate() {
            if pickup.active && pickup.enabled && !self.blink_object_hidden(pickup.entity) {
                parts.push(Part::Pickup(i));
            }
        }
        for (i, rock) in self.duplicated_rocks.iter().enumerate() {
            if rock.active && !self.blink_object_hidden(rock.entity) {
                parts.push(Part::DuplicatedRock(i));
            }
        }
        for rope in 0..self.ropes.len() {
            if self.hook_live(rope) {
                parts.push(Part::Hook(rope));
            }
        }
        for i in 0..self.arrows.len() {
            if self.arrow_registered(i) {
                parts.push(Part::Arrow(i));
            }
        }
        self.scratch.arrow_landings.clear();
        parts.sort_by_key(|&part| self.part_rank(part));
        let mut contacts = std::mem::take(&mut self.scratch.cc_contacts);
        contacts.clear();
        for (k, &current) in parts.iter().enumerate() {
            for &prior in &parts[..k] {
                // SameCompositeAs: nodes of one platform share its composite.
                if self.part_composite(prior).is_some_and(|c| Some(c) == self.part_composite(current))
                    || !data.layers_collide(self.part_layer(prior, data), self.part_layer(current, data))
                    || !self.swept_pair(Swept::Part(prior), Swept::Part(current))
                {
                    continue;
                }
                if let Some(contact) = self.part_collider(prior).contact(self.part_collider(current)) {
                    contacts.push((prior, current, contact));
                    let (later, earlier) = sweep_pair(self.part_rank(prior), self.part_rank(current));
                    self.note_arrow_landing(prior, current, (GROUP_CC, later, earlier), data);
                }
            }
        }
        // PrepareCollisionCallBacks queues every pair's callbacks before the
        // first ResolveCollision (DetPhysics.cs:3068-3071), the
        // earlier-sorted side's first.
        for &(a, b, _) in &contacts {
            for (side, other) in [(a, b), (b, a)] {
                // SmokeGrenadeExplode2 goes off on any collider but a
                // Projectile that does not ignite explosives, as a portal
                // shot's or an arrow's (SmokeGrenadeExplode2.cs:63-78,
                // Arrow.prefab IgnitesExplosives 0). The hook is on the Rope
                // layer, so a Smoke it touches goes off.
                if let Part::Item(item) = side
                    && self.thrown_items[item].ability == ABILITY_SMOKE
                    && !matches!(other, Part::Shot(_) | Part::Arrow(_))
                {
                    let (later, earlier) = sweep_pair(self.part_rank(a), self.part_rank(b));
                    let hit = CallbackHit::ItemDetonate { item, count: 1 };
                    self.scratch.callback_hits.push(((GROUP_CC, later, earlier), hit));
                }
                let Part::Hook(rope) = side else {
                    continue;
                };
                let target = match other {
                    Part::Item(i) => HookTarget::Item(i),
                    Part::Shot(i) => HookTarget::Shot(i),
                    Part::Node(i) => HookTarget::Node(i),
                    Part::Pickup(_) => HookTarget::Pickup,
                    // RopeHook ignores every body of the caster's and the end's
                    // player (RopeHook.cs:108-115).
                    Part::Rock(player) => {
                        let body_player = self.players[player].owner;
                        if body_player == self.players[self.ropes[rope].owner].owner
                            || body_player == self.ropes[rope].hook_end_player
                        {
                            continue;
                        }
                        HookTarget::Player(player)
                    }
                    Part::Arrow(i) => HookTarget::Arrow(i),
                    // A hook on a Duplicator copy has no rope parent here yet.
                    Part::DuplicatedRock(_) => continue,
                    Part::Engine(_) | Part::Spike(_) | Part::Missile(_) | Part::Hook(_) | Part::Hole(_) => unreachable!(),
                };
                self.queue_hook_hit(rope, GROUP_CC, self.part_rank(other), target);
            }
        }
        for &(a, b, contact) in &contacts {
            self.resolve_parts(a, b, contact);
            for (side, other) in [(a, b), (b, a)] {
                // BounceBall.OnCollide: a Rock bounces off another
                // Player-layer collider, and a tenth as hard off a
                // Projectile, unless it hangs on a rope (BounceBall.cs:310-322).
                // Both callbacks get the manifold's normal.
                if let Part::Rock(rock) = side {
                    let scale = match self.part_layer(other, data) {
                        LAYER_PLAYER => Fix::ONE,
                        LAYER_PROJECTILE => Fix::from_f32(0.1),
                        _ => continue,
                    };
                    if self.players[rock].rope.is_none() {
                        self.players[rock].rock.body.force +=
                            contact.normal * data.rock.player_collision_bounce * scale * Fix::from_int(60);
                    }
                }
                // Mine.OnCollide: an arrow is an igniter, and the Mine sees
                // the contact whether or not the arrow landed first.
                if let (Part::Item(item), Part::Arrow(_)) = (side, other) {
                    let mine = &mut self.thrown_items[item];
                    if mine.ability == ABILITY_MINE && matches!(mine.phase, 1 | 2) {
                        mine.time_since_anticipation = data.mine.anticipation_duration * Fix::from_f64(0.8);
                        mine.phase = 3;
                    }
                }
            }
        }
        self.scratch.cb_parts = parts;
        self.scratch.cc_contacts = contacts;
    }
}
