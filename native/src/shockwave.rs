//! Shockwave's captured target list, one-time player pushes and timed impulses.
use crate::{
    actors::Actor, curve::Curve, fixed::Fix, physics::scene::Scene, platform_motion::Binding,
    scheduler, vector::Vec2,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub platform_force: Fix,
    pub default_force: Fix,
    pub grounded_multiplier: Fix,
    pub radius: Fix,
    pub mask: u32,
    pub activate_on_start: bool,
    pub every_frame: bool,
    pub curve: Curve,
    pub kill_after_frame: bool,
    pub gust: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub scale: Fix,
    pub time: Fix,
    pub owner: i32,
    pub hit_count: usize,
    pub hits: Vec<Option<u32>>,
    pub hit_objects: BTreeSet<u32>,
    pub ignored: BTreeSet<u32>,
}
pub struct Targets<'a> {
    pub actors: &'a mut [Actor],
    pub scene: &'a mut Scene,
    pub bindings: &'a [Binding],
}
impl State {
    pub fn new(owner: i32, scale: Fix) -> Self {
        Self {
            scale,
            time: Fix::ZERO,
            owner,
            hit_count: 0,
            hits: vec![None; 50],
            hit_objects: BTreeSet::new(),
            ignored: BTreeSet::new(),
        }
    }
    pub fn activate(
        &mut self,
        config: &Config,
        position: Vec2,
        dt: Fix,
        targets: &mut Targets,
    ) -> scheduler::Result<bool> {
        let Targets {
            actors,
            scene,
            bindings,
        } = targets;
        self.hit_objects.clear();
        let hits = scene.circle_cast_all(
            position,
            config.radius * self.scale,
            config.mask,
            self.hits.len(),
        )?;
        self.hit_count = hits.len();
        for (to, id) in self.hits.iter_mut().zip(hits) {
            *to = Some(id);
        }
        for id in self.hits[..self.hit_count].iter().flatten().copied() {
            let object = scene
                .objects
                .iter()
                .find(|o| o.id == id)
                .ok_or("missing_shockwave_target")?;
            if self.hit_objects.contains(&id)
                || object.destroyed
                || bindings
                    .iter()
                    .any(|b| b.object == id && b.transform_destroyed)
            {
                continue;
            }
            if object.layer == 8 {
                let actor = actors
                    .iter_mut()
                    .find(|a| a.object == id)
                    .ok_or("actor_player_collider_not_bound")?;
                if actor
                    .ordinary
                    .as_ref()
                    .is_some_and(|a| !a.is_pushable.unwrap_or(false))
                {
                    continue;
                }
                let divisor = if config.every_frame {
                    Fix::ONE
                } else {
                    Fix::from_int(7)
                };
                let scale = (self.scale + Fix::ONE)
                    .checked_div(Fix::from_int(2))
                    .map_err(|_| "division_by_zero")?;
                let delta = object.transform.position - position;
                let distance = delta
                    .length_squared()
                    .checked_div(Fix::from_int(10))
                    .map_err(|_| "division_by_zero")?
                    .max(Fix::from_int(3));
                let force = (scale * config.default_force)
                    .checked_div(distance * divisor)
                    .map_err(|_| "division_by_zero")?;
                let force = if actor.controller.physics.motion.grounded {
                    config.grounded_multiplier * delta * force
                } else {
                    delta * force
                };
                actor
                    .controller
                    .physics
                    .add_force(force, &actor.config.physics);
                actor.controller_to_body();
                continue;
            }
            if object.layer == 16 {
                // Lethal-terrain bodies and platform-attached weapons require their concrete BoplBody graph.
                return Err("shockwave_lethal_target_not_implemented");
            }
            self.hit_objects.insert(id);
        }
        self.update(config, position, dt, targets)
    }
    pub fn update(
        &mut self,
        config: &Config,
        position: Vec2,
        dt: Fix,
        targets: &mut Targets,
    ) -> scheduler::Result<bool> {
        let Targets {
            scene, bindings, ..
        } = targets;
        let strength = config.curve.evaluate(self.time);
        self.time += dt;
        let duration = config.curve.keys.last().ok_or("index_out_of_range")?.time;
        if self.time > duration {
            return Ok(true);
        }
        for id in self.hits.iter().take(self.hit_count).flatten().copied() {
            let Some(index) = scene.objects.iter().position(|o| o.id == id) else {
                continue;
            };
            let object = &scene.objects[index];
            if object.destroyed
                || !self.hit_objects.contains(&id)
                || self.ignored.contains(&id)
                || bindings
                    .iter()
                    .any(|b| b.object == id && b.transform_destroyed)
            {
                continue;
            }
            let binding = bindings
                .iter()
                .find(|b| b.object == id)
                .and_then(|b| b.platform.as_ref());
            let (point, direction, force) = if let Some(platform) = binding {
                let point = platform
                    .surface
                    .closest_point(position)
                    .map_err(|_| "division_by_zero")?;
                let direction = -platform.surface.normal(point);
                let distance = (position - point)
                    .length_squared()
                    .checked_div(Fix::from_int(10))
                    .map_err(|_| "division_by_zero")?
                    .max(Fix::from_f64(1.5));
                (
                    point,
                    direction,
                    config
                        .platform_force
                        .checked_div(distance)
                        .map_err(|_| "division_by_zero")?,
                )
            } else {
                let point = object.transform.position;
                let direction = point - position;
                let distance = direction
                    .length_squared()
                    .checked_div(Fix::from_int(10))
                    .map_err(|_| "division_by_zero")?
                    .max(Fix::from_f64(1.5));
                (
                    point,
                    direction,
                    config
                        .default_force
                        .checked_div(distance)
                        .map_err(|_| "division_by_zero")?,
                )
            };
            let force = force * self.scale;
            if object.layer == 23 {
                return Err("shockwave_black_hole_not_implemented");
            }
            let impulse = direction.normalized_safe() * force * strength * Fix::from_int(60);
            let arm = (point - object.transform.position).normalized_safe();
            if let Some(composite) = object.composite {
                scene.composites[composite].body.apply_impulse(impulse, arm);
            } else {
                scene.objects[index].body.apply_impulse(impulse, arm);
            }
        }
        Ok(false)
    }
}
