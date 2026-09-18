//! Persistent beam crossing/merging graph. Hit effects and beam collider creation
//! are applied by gameplay after this graph stage, as in renderBeams.
use super::{
    color::Colors,
    intersection::{self, Intersection, Ray, ReflectedRay},
};
use crate::{fixed::Fix, physics::query::CastHit, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Beam {
    pub id: i32,
    pub owner_id: i32,
    pub position: Vec2,
    pub direction: Vec2,
    pub time_passed: Fix,
    pub scale: Fix,
    pub distance: Fix,
    pub colors: Colors,
    pub ground: Option<u32>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[default]
    Single,
    MaybeChild,
    Child,
    Parent,
    ParentAndChild,
    ChildWithEffect,
    #[serde(rename = "reflected")]
    Reflected,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Crossing {
    pub a: usize,
    pub b: usize,
    pub age: i32,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Record {
    pub a: i32,
    pub b: i32,
    pub age: i32,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub maximum: Fix,
    pub water: Fix,
    pub space: bool,
    pub mask: u32,
    pub cross_offset: Fix,
    pub environmental_player: i32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub bodies: Vec<Beam>,
    pub roles: Vec<Role>,
    pub parents: Vec<i32>,
    pub reflected: Vec<bool>,
    pub was_reflected: Vec<bool>,
    pub reflection_distances: Vec<Fix>,
    pub hits: Vec<Option<u32>>,
    pub crossings: Vec<Record>,
    pub total: usize,
    pub initial: usize,
    pub mapping: Vec<(i32, usize)>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Graph {
    pub(super) config: Config,
    pub(super) bodies: Vec<Beam>,
    pub(super) roles: Vec<Role>,
    pub(super) parents: Vec<i32>,
    pub(super) reflected: Vec<bool>,
    pub(super) was_reflected: Vec<bool>,
    pub(super) reflection_distances: Vec<Fix>,
    pub(super) hits: Vec<Option<u32>>,
    pub(super) crossing: Vec<Crossing>,
    previous: Vec<Record>,
    #[serde(default, skip_serializing)]
    previous_reflected: Vec<i32>,
    mapping: Vec<(i32, usize)>,
    pub(super) count: usize,
    pub(super) initial: usize,
    previous_count: usize,
}
pub type Caster<'a> = dyn FnMut(Vec2, Vec2, Fix, u32) -> CastHit + 'a;
impl Graph {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            bodies: vec![Beam::default(); 128],
            roles: vec![Role::Single; 128],
            parents: vec![0; 128],
            reflected: vec![false; 128],
            was_reflected: vec![false; 128],
            reflection_distances: vec![Fix::ZERO; 128],
            hits: vec![None; 128],
            crossing: Vec::new(),
            previous: Vec::new(),
            previous_reflected: Vec::new(),
            mapping: Vec::new(),
            count: 0,
            initial: 0,
            previous_count: 0,
        }
    }
    pub(super) fn map(&mut self, id: i32, index: usize) {
        if let Some(entry) = self.mapping.iter_mut().find(|p| p.0 == id) {
            entry.1 = index;
        } else {
            assert!(self.mapping.len() < 128);
            self.mapping.push((id, index));
        }
    }
    fn lookup(&self, id: i32) -> Option<usize> {
        self.mapping.iter().find(|p| p.0 == id).map(|p| p.1)
    }
    pub(super) fn ray(&self, i: usize) -> ReflectedRay {
        let b = self.bodies[i];
        ReflectedRay {
            ray: Ray {
                position: b.position,
                direction: b.direction,
                distance: b.distance,
            },
            reflected: self.reflected[i],
            was_reflected: self.was_reflected[i],
            reflection_distance: self.reflection_distances[i],
        }
    }
    pub(super) fn intersect(&self, a: usize, b: usize, min_b: bool) -> Intersection {
        intersection::beams(self.ray(a), self.ray(b), min_b)
    }
    pub(super) fn cast(&mut self, i: usize, caster: &mut Caster<'_>) {
        let b = self.bodies[i];
        let config = self.config;
        let mut hit = caster(b.position, b.direction, config.maximum, config.mask);
        let found = hit.distance != Fix::MAX;
        hit.distance = hit.distance.min(config.maximum);
        self.reflected[i] = false;
        self.reflection_distances[i] = Fix::ZERO;
        self.hits[i] = hit.object;
        if !found || hit.point.y < config.water {
            hit.point = b.position + b.direction * config.maximum;
            if !config.space && hit.point.y < config.water {
                let height = config.water - b.position.y;
                let mut delta = b.direction * config.maximum;
                if delta.y == Fix::ZERO {
                    delta.y = Fix::from_f64(0.001);
                }
                let point = Vec2::new(delta.x * height / delta.y, height) + b.position;
                let direction = Vec2::new(b.direction.x, b.direction.y * -Fix::ONE);
                if b.direction.dot(-Vec2::UP) >= Fix::from_f64(0.997) {
                    self.bodies[i].distance =
                        (point - b.position).length() + self.reflection_distances[i];
                    return;
                }
                let reflected = caster(point, direction, config.maximum, config.mask);
                self.reflected[i] = true;
                self.reflection_distances[i] = reflected.distance.min(config.maximum);
                self.hits[i] = reflected.object;
                hit.distance = (point - b.position).length() + self.reflection_distances[i];
            }
        }
        self.bodies[i].distance = hit.distance;
    }
    pub fn step(&mut self, input: &[Beam], caster: &mut Caster<'_>) -> Snapshot {
        let mut snapshot = Snapshot::default();
        self.step_into(input, caster, &mut snapshot);
        snapshot
    }

    pub fn step_into(&mut self, input: &[Beam], caster: &mut Caster<'_>, snapshot: &mut Snapshot) {
        self.count = input.len();
        self.bodies[..self.count].copy_from_slice(input);
        self.previous_reflected.clear();
        self.previous_reflected.extend(
            self.mapping
                .iter()
                .filter_map(|&(id, i)| self.reflected[i].then_some(id)),
        );
        self.was_reflected.fill(false);
        self.mapping.clear();
        self.roles.fill(Role::Single);
        self.parents[..self.previous_count.max(self.count)].fill(-1);
        for i in 0..self.count {
            self.map(self.bodies[i].id, i);
        }
        self.initial = self.count;
        for i in 0..self.count {
            self.cast(i, caster);
        }
        for &id in &self.previous_reflected {
            if let Some(i) = self.lookup(id) {
                self.was_reflected[i] = true;
            }
        }
        let mut previous = std::mem::take(&mut self.previous);
        for &r in &previous {
            if let (Some(a), Some(b)) = (self.lookup(r.a), self.lookup(r.b)) {
                let hit = self.intersect(a, b, false);
                if hit.hit {
                    self.cross(a, b, r.age, hit.distances, caster);
                }
            }
        }
        for a in 0..self.initial {
            if !matches!(self.roles[a], Role::Single | Role::MaybeChild) {
                continue;
            }
            let mut best = Vec2::new(Fix::MAX, Fix::MAX);
            let mut target = None;
            for b in a + 1..self.initial {
                if matches!(self.roles[b], Role::Single | Role::MaybeChild) {
                    let hit = self.intersect(a, b, false);
                    if hit.hit && hit.distances.x < best.x {
                        best = hit.distances;
                        target = Some(b);
                    }
                }
            }
            if let Some(b) = target {
                self.cross(a, b, 0, best, caster);
            }
        }
        previous.clear();
        previous.extend(self.crossing.iter().map(|c| Record {
            a: self.bodies[c.a].id,
            b: self.bodies[c.b].id,
            age: c.age.wrapping_add(1),
        }));
        self.previous = previous;
        for i in 0..self.count {
            if !matches!(self.roles[i], Role::MaybeChild | Role::Child) {
                continue;
            }
            let mut parent = self.parents[i];
            while parent >= 0 {
                let p = parent as usize;
                let role = self.roles[p];
                let scale = self.bodies[i].scale;
                self.bodies[p].scale += scale;
                if self.roles[i] == Role::MaybeChild {
                    self.bodies[p].colors = self.bodies[p].colors.mix(self.bodies[i].colors);
                }
                if !matches!(role, Role::MaybeChild | Role::Child | Role::ParentAndChild) {
                    break;
                }
                parent = self.parents[p];
            }
        }
        let mut extra = 0;
        for i in 0..self.count {
            if !self.reflected[i] {
                continue;
            }
            let mut b = self.bodies[i];
            b.position += b.direction * (b.distance - self.reflection_distances[i]);
            b.distance = self.reflection_distances[i];
            b.direction.y = -b.direction.y;
            b.id = b.id.wrapping_neg();
            let j = self.count + extra;
            self.bodies[j] = b;
            self.bodies[i].distance -= self.reflection_distances[i];
            self.roles[j] = self.roles[i];
            self.roles[i] = match self.roles[i] {
                Role::Child | Role::ChildWithEffect | Role::MaybeChild => Role::Single,
                Role::ParentAndChild => Role::Parent,
                role => role,
            };
            self.reflected[j] = false;
            self.hits[j] = self.hits[i];
            self.map(b.id, j);
            extra += 1;
        }
        let n = self.count + extra;
        snapshot.bodies.clear();
        snapshot.bodies.extend_from_slice(&self.bodies[..n]);
        snapshot.roles.clear();
        snapshot.roles.extend_from_slice(&self.roles[..n]);
        snapshot.parents.clear();
        snapshot.parents.extend_from_slice(&self.parents[..n]);
        snapshot.reflected.clear();
        snapshot.reflected.extend_from_slice(&self.reflected[..n]);
        snapshot.was_reflected.clear();
        snapshot
            .was_reflected
            .extend_from_slice(&self.was_reflected[..n]);
        snapshot.reflection_distances.clear();
        snapshot
            .reflection_distances
            .extend_from_slice(&self.reflection_distances[..n]);
        snapshot.hits.clear();
        snapshot.hits.extend_from_slice(&self.hits[..n]);
        snapshot.crossings.clone_from(&self.previous);
        snapshot.total = self.count;
        snapshot.initial = self.initial;
        snapshot.mapping.clone_from(&self.mapping);
        self.previous_count = self.count;
        self.crossing.clear();
        self.count = 0;
    }
}
pub(super) fn blend(a: Vec2, b: Vec2) -> Vec2 {
    (a * Fix::from_f64(0.5) + b * Fix::from_f64(0.5)).normalized_safe()
}
pub(super) fn age_time(age: i32) -> Fix {
    Fix::from_int(age as i64) * DT
}
