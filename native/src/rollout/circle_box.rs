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
                Collider::Circle(Circle { center: pickup.position + pickup.offset, radius: pickup.radius })
            }
            Part::DuplicatedRock(i) => Collider::Circle(Circle {
                center: self.duplicated_rocks[i].position,
                radius: self.duplicated_rocks[i].collider_radius,
            }),
            Part::Hook(r) => Collider::Circle(self.hook_circle(r)),
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
        }
    }

    /// Its place in SortAndSweep's order.
    pub(super) fn part_rank(&self, part: Part) -> SweepRank {
        sweep_rank(&self.entities, self.swept_min(Swept::Part(part)), self.part_entity(part))
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
            | Part::Hook(_) => None,
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
                (pickup.body, pickup.position + pickup.offset)
            }
            (Part::DuplicatedRock(i), None) => (self.duplicated_rocks[i].body, self.duplicated_rocks[i].position),
            (Part::Hook(r), None) => (self.part_body(part), self.ropes[r].hook_position),
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
                pickup.position += delta;
                let center = pickup.position + pickup.offset;
                pickup.synced_center = center;
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
            Part::Rock(i) => self.players[i].base.position,
            Part::Pickup(i) => self.ability_pickups[i].position,
            Part::DuplicatedRock(i) => self.duplicated_rocks[i].position,
            Part::Hook(r) => self.ropes[r].hook_position,
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
        for i in 0..self.players.len() {
            if self.players[i].active == ActiveAbility::Rock && self.playable(i) {
                parts.push(Part::Rock(i));
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
            match part {
                Part::Spike(spike) => {
                    let hit = CallbackHit::SpikeShove { spike, target: circle, point: contact.point };
                    self.scratch.callback_hits.push((key, hit));
                }
                Part::Missile(missile) => {
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
                && item.entity.is_some_and(|entity| self.entities[entity].flags & E_ACTIVE != 0)
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
                && !self.attached_hidden(node.platform_index, node.entity)
                && node.entity.is_some_and(|entity| {
                    self.entities[entity].flags & (E_ACTIVE | E_ENABLED) == E_ACTIVE | E_ENABLED
                })
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
                }
            }
        }
        // PrepareCollisionCallBacks queues every pair's callbacks before the
        // first ResolveCollision (DetPhysics.cs:3068-3071).
        for &(a, b, _) in &contacts {
            let (rope, other) = match (a, b) {
                (Part::Hook(rope), other) | (other, Part::Hook(rope)) => (rope, other),
                _ => continue,
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
                // A hook on a Duplicator copy has no rope parent here yet.
                Part::DuplicatedRock(_) => continue,
                Part::Engine(_) | Part::Spike(_) | Part::Missile(_) | Part::Hook(_) => unreachable!(),
            };
            self.queue_hook_hit(rope, GROUP_CC, self.part_rank(other), target);
        }
        for &(a, b, contact) in &contacts {
            self.resolve_parts(a, b, contact);
            // BounceBall.OnCollide: a Rock pushed by another Player-layer
            // collider bounces off it, unless it hangs on a rope
            // (BounceBall.cs:310-322). Both callbacks get the manifold's
            // normal.
            if let (Part::Rock(first), Part::Rock(second)) = (a, b) {
                let impulse = contact.normal * data.rock.player_collision_bounce * Fix::from_int(60);
                for rock in [first, second] {
                    if self.players[rock].rope.is_none() {
                        self.players[rock].rock.body.force += impulse;
                    }
                }
            }
        }
        self.scratch.cb_parts = parts;
        self.scratch.cc_contacts = contacts;
    }
}
