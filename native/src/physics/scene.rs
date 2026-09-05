//! Ordered rigid/composite-body stage of DetPhysics.Simulate.
//! Beam updates are inserted after ropes; gameplay callbacks are returned to the caller.
use super::{Body, Contact, positional_correction, resolve, shape::Collider};
use crate::{fixed::Fix, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Transform {
    pub position: Vec2,
    pub rotation: Fix,
    pub offset: Vec2,
    // FixTransform can cache a direction assigned directly by gameplay. Its
    // angle roundtrip need not reproduce those exact raw components.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<Vec2>,
}
impl Transform {
    fn basis(self) -> Vec2 {
        self.right.unwrap_or_else(|| Vec2::angle(self.rotation))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    pub id: u32,
    pub collider: Collider,
    pub body: Body,
    pub transform: Transform,
    pub layer: usize,
    /// GameObject activity in the hierarchy, independent of collider enablement.
    pub active: bool,
    #[serde(default = "collider_enabled", skip_serializing_if = "is_enabled")]
    pub enabled: bool,
    pub callback: bool,
    pub callback_destroyed: bool,
    pub destroyed: bool,
    pub composite: Option<usize>,
    pub relative_pos: Vec2,
    pub relative_orientation: Fix,
}
fn collider_enabled() -> bool {
    true
}
fn is_enabled(value: &bool) -> bool {
    *value
}
impl Object {
    pub fn is_active(&self) -> bool {
        self.active && self.enabled
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Composite {
    pub body: Body,
    pub position: Vec2,
    pub destroyed: bool,
    /// Stable object IDs, in registration order, separated by collider type.
    pub circles: Vec<u32>,
    pub boxes: Vec<u32>,
    pub rounded_rects: Vec<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub composites: Vec<Composite>,
    /// Bit j in row i means that layers i and j collide (Unity Physics2D matrix).
    pub layers: [u32; 32],
    pub gravity: Fix,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_system: Option<super::rope_system::RopeSystem>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionEvent {
    pub receiver: u32,
    pub other: u32,
    pub collider: Collider,
    pub layer: usize,
    pub impact_velocity: Vec2,
    pub contact: Contact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepReport {
    pub sweep_x: bool,
    pub rounded_sweep_x: bool,
    /// Pair groups are in the order DetPhysics resolves them.
    pub pairs: [Vec<[u32; 2]>; 6],
    pub callbacks: Vec<CollisionEvent>,
}
const ORDER: [usize; 3] = [2, 0, 1];
const GROUPS: [[usize; 2]; 6] = [[2, 2], [1, 2], [0, 2], [0, 0], [0, 1], [1, 1]];

impl Scene {
    pub(super) fn index(&self, id: u32) -> usize {
        self.objects
            .iter()
            .position(|o| o.id == id)
            .expect("missing composite child")
    }
    pub(super) fn composite(&self, i: usize) -> Option<usize> {
        self.objects[i]
            .composite
            .filter(|&c| !self.composites[c].destroyed)
    }
    fn indices(&self, kind: usize) -> Vec<usize> {
        self.objects
            .iter()
            .enumerate()
            .filter_map(|(i, o)| (o.collider.kind() == kind).then_some(i))
            .collect()
    }
    fn synchronize(&mut self) {
        for kind in ORDER {
            for i in self.indices(kind) {
                if self.composite(i).is_some() {
                    continue;
                }
                let o = &mut self.objects[i];
                o.collider.rotate(o.transform.rotation);
                let center = if kind == 2 {
                    o.transform.position
                } else {
                    o.transform.position + o.transform.basis().complex_mul(o.transform.offset)
                };
                o.collider.set_center(center);
                o.body.orientation = o.transform.rotation;
            }
        }
        for c in 0..self.composites.len() {
            self.synchronize_composite(c);
        }
    }
    pub(super) fn synchronize_composite(&mut self, c: usize) {
        if self.composites[c].destroyed || self.composites[c].rounded_rects.is_empty() {
            return;
        }
        let root = self.index(self.composites[c].rounded_rects[0]);
        let transform = self.objects[root].transform;
        self.composites[c].position = transform.position;
        self.composites[c].body.orientation = transform.rotation;
        self.objects[root].collider.rotate(transform.rotation);
        self.objects[root].collider.set_center(transform.position);
        self.objects[root].body.orientation = transform.rotation;
        let children: Vec<_> = self.composites[c]
            .boxes
            .iter()
            .chain(&self.composites[c].circles)
            .copied()
            .collect();
        for id in children {
            let i = self.index(id);
            let o = &mut self.objects[i];
            o.collider
                .set_center(transform.position + o.relative_pos.complex_mul(transform.basis()));
            o.body.orientation = transform.rotation + o.relative_orientation;
            o.collider.rotate(o.body.orientation);
        }
    }

    fn integrate(&mut self, dt: Fix) {
        for kind in ORDER {
            for i in self.indices(kind) {
                if !self.objects[i].is_active()
                    || self.objects[i].body.inverse_mass <= Fix::ZERO
                    || self.composite(i).is_some()
                {
                    continue;
                }
                let o = &mut self.objects[i];
                let delta = o.body.integrate(dt, self.gravity);
                o.collider.translate(delta);
                if kind != 0 {
                    o.collider.rotate(o.body.orientation);
                }
            }
        }
        for c in 0..self.composites.len() {
            if self.composites[c].destroyed || self.composites[c].body.inverse_mass == Fix::ZERO {
                continue;
            }
            let root = self.index(self.composites[c].rounded_rects[0]);
            if !self.objects[root].is_active() || self.objects[root].body.inverse_mass <= Fix::ZERO
            {
                continue;
            }
            for id in self.composites[c].rounded_rects.clone() {
                let i = self.index(id);
                self.composites[c].body.force += self.objects[i].body.force;
                self.objects[i].body.force = Vec2::ZERO;
            }
            let b = &mut self.composites[c].body;
            b.velocity += dt * Vec2::new(Fix::ZERO, self.gravity * b.gravity_scale)
                + DT * b.force * b.inverse_mass;
            // Composite translation is taken BEFORE drag, unlike ordinary bodies.
            let delta = b.velocity * dt;
            b.force = Vec2::ZERO;
            b.orientation += dt * b.angular_velocity;
            b.orientation = (b.orientation.checked_rem(Fix::TAU, true).unwrap() + Fix::TAU)
                .checked_rem(Fix::TAU, true)
                .unwrap();
            b.velocity *= if dt == Fix::ZERO {
                Fix::ONE
            } else {
                b.air_drag
            };
            b.angular_velocity *= if dt == Fix::ZERO {
                Fix::ONE
            } else {
                b.rotational_drag
            };
            let orientation = b.orientation;
            self.composites[c].position += delta;
            for id in self.composites[c].rounded_rects.clone() {
                let i = self.index(id);
                self.objects[i].collider.translate(delta);
                self.objects[i].body.orientation = orientation;
                self.objects[i].collider.rotate(orientation);
            }
            let children: Vec<_> = self.composites[c]
                .circles
                .iter()
                .chain(&self.composites[c].boxes)
                .copied()
                .collect();
            for id in children {
                let i = self.index(id);
                let o = &mut self.objects[i];
                o.collider.set_center(
                    self.composites[c].position
                        + o.relative_pos.complex_mul(Vec2::angle(orientation)),
                );
                o.body.orientation = orientation + o.relative_orientation;
                if o.collider.kind() == 1 {
                    o.collider.rotate(o.body.orientation);
                }
            }
        }
    }
    fn effective(&self, i: usize) -> (Body, Vec2) {
        if let Some(c) = self.composite(i) {
            (self.composites[c].body, self.composites[c].position)
        } else {
            (self.objects[i].body, self.objects[i].collider.center())
        }
    }
    fn set_effective(&mut self, i: usize, body: Body) {
        if let Some(c) = self.composite(i) {
            self.composites[c].body = body;
        } else {
            self.objects[i].body = body;
        }
    }
    fn queue_callbacks(
        &self,
        a: usize,
        b: usize,
        contact: Contact,
        events: &mut Vec<CollisionEvent>,
    ) {
        for (receiver, other) in [(a, b), (b, a)] {
            // Preserve the source's p1 guard for BOTH callbacks, and same normal.
            if !self.objects[receiver].callback {
                continue;
            }
            assert!(self.objects[a].callback, "source null collision callback");
            if self.objects[a].callback_destroyed {
                continue;
            }
            let o = &self.objects[other];
            events.push(CollisionEvent {
                receiver: self.objects[receiver].id,
                other: o.id,
                collider: o.collider,
                layer: o.layer,
                impact_velocity: o.body.velocity,
                contact,
            });
        }
    }
    pub(super) fn remove_destroyed(&mut self) {
        // Unity removes circles, then rounded rectangles, then boxes. A removed
        // platform can also remove live attachments before their own scan runs.
        for kind in [0, 2, 1] {
            while let Some(id) = self
                .objects
                .iter()
                .find(|o| o.destroyed && o.collider.kind() == kind)
                .map(|o| o.id)
            {
                self.destroy(id);
            }
        }
    }
    /// Rigid/composite/rope step. Worlds with a beam pool use System::simulate.
    pub fn step(&mut self, dt: Fix) -> StepReport {
        self.step_with_beams(dt, |_| {})
    }
    /// Unity updates beams after broad-phase/ropes, before narrow-phase contacts.
    /// New bodies therefore enter the pair search on the following step.
    pub fn step_with_beams(&mut self, dt: Fix, beams: impl FnOnce(&mut Self)) -> StepReport {
        self.remove_destroyed();
        self.synchronize();
        if self.objects.is_empty() {
            return StepReport {
                sweep_x: false,
                rounded_sweep_x: false,
                pairs: std::array::from_fn(|_| Vec::new()),
                callbacks: Vec::new(),
            };
        }
        self.integrate(dt);
        let (sweep_x, rounded_sweep_x, groups) = super::sweep::pairs(self);
        let mut report = StepReport {
            sweep_x,
            rounded_sweep_x,
            pairs: std::array::from_fn(|g| {
                groups[g]
                    .iter()
                    .map(|[a, b]| [self.objects[*a].id, self.objects[*b].id])
                    .collect()
            }),
            callbacks: Vec::new(),
        };
        if let Some(ropes) = &mut self.rope_system {
            ropes.step(&self.objects, rounded_sweep_x, dt);
        }
        beams(self);
        for group in groups {
            let contacts: Vec<_> = group
                .into_iter()
                .filter_map(|[a, b]| {
                    if self.composite(a).is_some() && self.composite(a) == self.composite(b) {
                        return None;
                    }
                    self.objects[a]
                        .collider
                        .contact(self.objects[b].collider)
                        .map(|contact| (a, b, contact))
                })
                .collect();
            for &(a, b, contact) in &contacts {
                self.queue_callbacks(a, b, contact, &mut report.callbacks);
            }
            for (a, b, contact) in contacts {
                if self.objects[a].body.inverse_mass < Fix::ZERO
                    || self.objects[b].body.inverse_mass < Fix::ZERO
                {
                    continue;
                }
                let (mut ba, pa) = self.effective(a);
                let (mut bb, pb) = self.effective(b);
                resolve(contact, &mut ba, &mut bb, pa, pb);
                self.set_effective(a, ba);
                self.set_effective(b, bb);
                // Source correction uses child inverse mass, even for composites.
                let corrections = positional_correction(
                    contact,
                    self.objects[a].body.inverse_mass,
                    self.objects[b].body.inverse_mass,
                    false,
                );
                for (i, delta) in [(a, corrections[0]), (b, corrections[1])] {
                    self.objects[i].collider.translate(delta);
                    if let Some(c) = self.composite(i) {
                        self.composites[c].position += delta;
                    }
                }
            }
        }
        for kind in ORDER {
            for i in self.indices(kind) {
                let o = &mut self.objects[i];
                if o.body.inverse_mass < Fix::ZERO {
                    continue;
                }
                o.transform.rotation = o.body.orientation;
                if o.transform.right.is_some() {
                    o.transform.right = Some(Vec2::angle(o.body.orientation));
                }
                o.transform.position =
                    o.collider.center() - o.transform.basis().complex_mul(o.transform.offset);
            }
        }
        report
    }
}

pub(super) fn group(a: usize, b: usize) -> usize {
    GROUPS.iter().position(|&p| p == [a, b]).unwrap()
}
