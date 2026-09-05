//! Source-order parent/child selection and truncation for crossed beams.
use super::graph::{Beam, Caster, Crossing, Graph, Role, age_time, blend};
use crate::{fixed::Fix, vector::Vec2};
impl Graph {
    pub(super) fn cross(
        &mut self,
        a: usize,
        b: usize,
        age: i32,
        distance: Vec2,
        caster: &mut Caster<'_>,
    ) {
        self.roles[a] = Role::Child;
        self.roles[b] = Role::Child;
        let ba = self.bodies[a];
        let bb = self.bodies[b];
        let base_a = ba.distance - self.reflection_distances[a];
        let base_b = bb.distance - self.reflection_distances[b];
        let after_a = distance.x - base_a;
        let after_b = distance.y - base_b;
        let (mut v, u) = (ba.direction, bb.direction);
        let (position, direction, u) = if distance.x > base_a && distance.y > base_b {
            let rb = Vec2::new(bb.direction.x, -bb.direction.y);
            let ra = Vec2::new(ba.direction.x, -ba.direction.y);
            self.reflection_distances[a] = after_a;
            self.reflection_distances[b] = after_b;
            self.reflected[a] = true;
            self.reflected[b] = true;
            v = ra;
            (
                ba.position + ba.direction * base_a + ra * after_a,
                blend(rb, ra),
                rb,
            )
        } else if distance.x > base_a {
            let reflected = Vec2::new(ba.direction.x, -ba.direction.y);
            self.reflection_distances[a] = after_a;
            self.reflection_distances[b] = Fix::ZERO;
            self.reflected[a] = true;
            self.reflected[b] = false;
            v = reflected;
            (
                bb.position + bb.direction * distance.y,
                blend(bb.direction, reflected),
                u,
            )
        } else if distance.y > base_b {
            let reflected = Vec2::new(bb.direction.x, -bb.direction.y);
            self.reflection_distances[a] = Fix::ZERO;
            self.reflection_distances[b] = after_b;
            self.reflected[a] = false;
            self.reflected[b] = true;
            // Unity assigns v here, retaining the original B direction in u.
            v = reflected;
            (
                ba.position + ba.direction * distance.x,
                blend(ba.direction, reflected),
                u,
            )
        } else {
            self.reflected[a] = false;
            self.reflected[b] = false;
            self.reflection_distances[a] = Fix::ZERO;
            self.reflection_distances[b] = Fix::ZERO;
            (
                bb.position + bb.direction * distance.y,
                blend(bb.direction, ba.direction),
                u,
            )
        };
        self.bodies[a].distance = distance.x;
        self.bodies[b].distance = distance.y;
        assert!(
            self.crossing.len() < 64,
            "Unity beam crossing capacity exceeded"
        );
        let parent = if v.dot(u) < -Fix::from_f64(0.5) {
            self.crossing.push(Crossing { a, b, age });
            self.roles[a] = Role::ChildWithEffect;
            self.roles[b] = Role::ChildWithEffect;
            a
        } else {
            let index = self.count;
            self.bodies[index] = Beam {
                scale: Fix::ZERO,
                colors: bb.colors.mix(ba.colors),
                position: position + direction * self.config.cross_offset,
                time_passed: age_time(age),
                distance: self.config.maximum,
                direction,
                id: i32::MAX - self.crossing.len() as i32 - 1,
                owner_id: self.config.environmental_player,
                ground: None,
            };
            self.count += 1;
            self.map(self.bodies[index].id, index);
            self.crossing.push(Crossing { a, b, age });
            self.roles[index] = Role::Parent;
            self.cast(index, caster);
            self.was_reflected[index] = false;
            index
        };
        self.parents[a] = parent as i32;
        self.parents[b] = parent as i32;
        for i in 0..parent {
            if i == a || i == b || matches!(self.roles[i], Role::Single | Role::MaybeChild) {
                continue;
            }
            let hit = self.intersect(parent, i, true);
            if hit.hit && hit.distances.x < self.bodies[parent].distance {
                self.truncate(parent, hit.distances.x);
                self.roles[parent] = Role::ParentAndChild;
                self.parents[parent] = i as i32;
            }
        }
        for i in 0..self.initial {
            if !matches!(self.roles[i], Role::Single | Role::MaybeChild) || i == a || i == b {
                continue;
            }
            // Each truncation changes the lengths seen by the following test.
            for owner in [parent, a, b] {
                let hit = self.intersect(owner, i, true);
                if hit.hit && hit.distances.y < self.bodies[i].distance {
                    self.truncate(i, hit.distances.y);
                    self.roles[i] = Role::MaybeChild;
                    self.parents[i] = owner as i32;
                }
            }
        }
    }
    fn truncate(&mut self, i: usize, distance: Fix) {
        let remainder = distance - (self.bodies[i].distance - self.reflection_distances[i]);
        self.bodies[i].distance = distance;
        if remainder <= Fix::ZERO {
            self.reflected[i] = false;
            self.reflection_distances[i] = Fix::ZERO;
        } else {
            self.reflected[i] = true;
            self.reflection_distances[i] = remainder;
        }
    }
}
