//! Rope/platform broad-phase and simulation ordering from DetPhysics.Simulate.
use super::{
    rope::{Rope, RopeConfig},
    scene::Object,
    shape::Collider,
};
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeSystem {
    pub gravity: Fix,
    pub space: bool,
    pub iterations: usize,
    pub stretchiness: Fix,
    /// Array position is RopeBody.indexInArray for every enabled rope.
    pub bodies: Vec<Rope>,
}
#[derive(Clone, Copy)]
struct Bounds {
    bounds: [Vec2; 2],
    rope: bool,
    index: usize,
}

impl RopeSystem {
    pub(super) fn step(&mut self, objects: &[Object], sweep_x: bool, dt: Fix) {
        let mut rectangles = Vec::new();
        let mut bounds = Vec::new();
        for object in objects {
            if let Collider::RoundedRect(rectangle) = object.collider {
                if object.is_active() {
                    bounds.push(Bounds {
                        bounds: object.collider.bounds(),
                        rope: false,
                        index: rectangles.len(),
                    });
                }
                rectangles.push(rectangle);
            }
        }
        for (index, rope) in self.bodies.iter().enumerate() {
            if rope.enabled {
                bounds.push(Bounds {
                    bounds: rope.bounds(),
                    rope: true,
                    index,
                });
            }
        }
        let axis = |v: Vec2| if sweep_x { v.x } else { v.y };
        let other = |v: Vec2| if sweep_x { v.y } else { v.x };
        bounds.sort_by(|a, b| {
            (axis(a.bounds[0]) - axis(b.bounds[0]))
                .sign()
                .cmp(&0)
                .then(a.rope.cmp(&b.rope))
                .then(a.index.cmp(&b.index))
        });
        let mut window: Vec<Bounds> = Vec::new();
        for b in bounds {
            window.retain(|a| axis(a.bounds[1]) > axis(b.bounds[0]));
            for a in &window {
                if a.rope == b.rope
                    || other(a.bounds[1]) <= other(b.bounds[0])
                    || other(a.bounds[0]) >= other(b.bounds[1])
                {
                    continue;
                }
                let (rope, rectangle) = if a.rope {
                    (a.index, b.index)
                } else {
                    (b.index, a.index)
                };
                let rope = &mut self.bodies[rope];
                // Source silently stops adding contacts when its 32 slots fill.
                if rope.collision_count < rope.colliding_rects.len() {
                    rope.colliding_rects[rope.collision_count] = rectangle;
                    rope.collision_count += 1;
                }
            }
            window.push(b);
        }
        let config = RopeConfig {
            dt,
            gravity: self.gravity,
            space: self.space,
            iterations: self.iterations,
            stretchiness: self.stretchiness,
        };
        // Unity runs these in parallel; ropes read shared rectangles and write
        // only their own segments, so sequential evaluation has the same result.
        for rope in &mut self.bodies {
            rope.step(config, &rectangles);
        }
    }
}
