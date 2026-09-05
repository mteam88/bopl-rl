//! Beam graph, pooled hitboxes and target effects at the physics-stage boundary.
use super::{
    effects,
    graph::{self, Beam, Graph, Role, Snapshot},
    hitbox,
};
use crate::{
    curve::Curve,
    fixed::Fix,
    physics::{
        Body,
        scene::{Object, Scene, StepReport, Transform},
        shape::Collider,
    },
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub graph: graph::Config,
    pub hitbox: hitbox::Config,
    pub effects: effects::Config,
    pub width: Fix,
    pub parent_offset: Fix,
    pub width_curve: Curve,
    pub layer: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    pub id: u32,
    pub state: effects::Target,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Slot {
    pub id: u32,
    #[serde(default, skip_serializing_if = "is_false")]
    pub destroyed: bool,
    pub hitbox: hitbox::Hitbox,
}
pub struct Step {
    pub graph: Option<Snapshot>,
    pub physics: StepReport,
}
#[derive(Serialize, Deserialize)]
pub struct System {
    pub config: Config,
    pub pool: Vec<Option<Slot>>,
    pub targets: Vec<Target>,
    graph: Graph,
    next_id: u32,
    pending: Vec<Beam>,
}
impl System {
    pub fn new(config: Config, targets: Vec<Target>, next_id: u32) -> Self {
        Self {
            graph: Graph::new(config.graph),
            config,
            pool: Vec::new(),
            targets,
            next_id,
            pending: Vec::new(),
        }
    }
    /// Execute the source-ordered physics step, retaining queued beams if an
    /// empty scene makes Unity return before its beam stage.
    pub fn simulate(&mut self, scene: &mut Scene, dt: Fix, input: &[Beam]) -> Step {
        self.queue(input);
        let dt = if self.config.effects.time_scale < Fix::from_f32(0.01) {
            Fix::ZERO
        } else {
            dt
        };
        let mut graph = None;
        let physics = scene.step_with_beams(dt, |scene| graph = Some(self.step(scene)));
        self.refresh(scene);
        Step { graph, physics }
    }
    fn dimensions(&self, mut beam: Beam, parent: bool) -> hitbox::Dimensions {
        if parent {
            beam.position += beam.direction * self.config.parent_offset;
            beam.distance -= self.config.parent_offset;
        }
        hitbox::Dimensions {
            position: beam.position,
            width: self.config.width * beam.scale,
            direction: beam.direction,
            distance: beam.distance,
            age: beam.time_passed,
            player_id: beam.owner_id,
            beam_id: beam.id,
            curve: self.config.width_curve.clone(),
        }
    }
    pub fn queue(&mut self, input: &[Beam]) {
        self.pending.extend_from_slice(input);
    }
    pub fn step(&mut self, scene: &mut Scene) -> Snapshot {
        let input = std::mem::take(&mut self.pending);
        let graph = self.graph.step(&input, &mut |p, d, length, mask| {
            scene.raycast(p, d, length, mask)
        });
        let count = graph.bodies.len();
        for (i, beam) in graph.bodies.iter().enumerate() {
            let existing = self
                .pool
                .get(i)
                .and_then(Option::as_ref)
                .filter(|slot| !slot.destroyed)
                .and_then(|slot| scene.objects.iter().position(|o| o.id == slot.id));
            if let Some(index) = existing {
                scene.objects[index].active = true;
            } else {
                let mut hitbox = hitbox::Hitbox::default();
                hitbox.set_dimensions(self.config.hitbox, &self.dimensions(*beam, false));
                hitbox.initialize(self.config.hitbox);
                let id = self.next_id;
                self.next_id += 1;
                scene.objects.push(Object {
                    id,
                    collider: Collider::Box(hitbox.collider.unwrap()),
                    body: Body {
                        inverse_mass: -Fix::ONE,
                        orientation: hitbox.rotation,
                        ..Body::default()
                    },
                    transform: transform(&hitbox),
                    layer: self.config.layer,
                    active: true,
                    enabled: true,
                    callback: true,
                    callback_destroyed: false,
                    destroyed: false,
                    composite: None,
                    relative_pos: Vec2::ZERO,
                    relative_orientation: Fix::ZERO,
                });
                let slot = Slot {
                    id,
                    hitbox,
                    destroyed: false,
                };
                if i < self.pool.len() {
                    self.pool[i] = Some(slot);
                } else {
                    self.pool.push(Some(slot));
                }
            }
        }
        // The original deactivation loop stops at its first empty, destroyed or inactive slot.
        for entry in self.pool.iter().skip(count) {
            let Some(slot) = entry else {
                break;
            };
            let Some(object) = scene.objects.iter_mut().find(|o| o.id == slot.id) else {
                break;
            };
            if slot.destroyed || object.destroyed || !object.active {
                break;
            }
            object.active = false;
        }
        for (i, beam) in graph.bodies.iter().enumerate() {
            let dimensions = self.dimensions(
                *beam,
                matches!(graph.roles[i], Role::Parent | Role::ParentAndChild),
            );
            let slot = self.pool[i].as_mut().unwrap();
            let object = scene.objects.iter_mut().find(|o| o.id == slot.id).unwrap();
            if let Collider::Box(collider) = object.collider {
                slot.hitbox.collider = Some(collider);
            }
            slot.hitbox.set_dimensions(self.config.hitbox, &dimensions);
            object.collider = Collider::Box(slot.hitbox.collider.unwrap());
            object.transform = transform(&slot.hitbox);
            object.body.orientation = slot.hitbox.rotation;
        }
        self.apply_effects(scene, &graph);
        graph
    }
    /// Match DestroyFix's flags; collider removal belongs to the next physics step.
    pub fn destroy_slot(&mut self, scene: &mut Scene, index: usize) {
        if let Some(slot) = self.pool.get_mut(index).and_then(Option::as_mut) {
            slot.destroyed = true;
            if let Some(object) = scene.objects.iter_mut().find(|o| o.id == slot.id) {
                object.destroyed = true;
                object.callback_destroyed = true;
            }
        }
    }
    pub fn refresh(&mut self, scene: &Scene) {
        for slot in self.pool.iter_mut().flatten() {
            if let Some(object) = scene.objects.iter().find(|o| o.id == slot.id) {
                if let Collider::Box(collider) = object.collider {
                    slot.hitbox.collider = Some(collider);
                }
                slot.hitbox.orientation = Some(object.body.orientation);
            } else {
                slot.destroyed = true;
                slot.hitbox.collider = None;
                slot.hitbox.orientation = None;
            }
        }
        for target in &mut self.targets {
            if let Some(object) = scene.objects.iter().find(|o| o.id == target.id) {
                target.state.force = object
                    .composite
                    .filter(|&c| !scene.composites[c].destroyed)
                    .map_or(object.body.force, |c| scene.composites[c].body.force);
            }
        }
    }
    fn apply_effects(&mut self, scene: &mut Scene, graph: &Snapshot) {
        for (i, beam) in graph.bodies.iter().enumerate() {
            let Some(target) = self
                .targets
                .iter_mut()
                .find(|t| Some(t.id) == graph.hits[i])
            else {
                continue;
            };
            let Some(object) = scene.objects.iter().find(|o| o.id == target.id) else {
                continue;
            };
            let composite = object.composite.filter(|&c| !scene.composites[c].destroyed);
            let previous_force =
                composite.map_or(object.body.force, |c| scene.composites[c].body.force);
            target.state.force = previous_force;
            let impact = effects::Impact {
                beam: *beam,
                role: graph.roles[i],
                water_reflection: graph.reflected[i] && i < graph.total,
                target: Some(0),
            };
            effects::apply(
                &self.config.effects,
                &[impact],
                std::slice::from_mut(&mut target.state),
            );
            if let Some(c) = composite {
                scene.composites[c].body.force = target.state.force;
            } else {
                scene
                    .objects
                    .iter_mut()
                    .find(|o| o.id == target.id)
                    .unwrap()
                    .body
                    .force = target.state.force;
            }
        }
    }
}
fn transform(hitbox: &hitbox::Hitbox) -> Transform {
    Transform {
        position: hitbox.position,
        rotation: hitbox.rotation,
        offset: hitbox.offset,
        right: Some(hitbox.right),
    }
}

fn is_false(value: &bool) -> bool {
    !value
}
