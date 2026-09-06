//! BoplBody component state. Collider ownership and registration remain with the world.
use crate::{
    fixed::Fix,
    physics::{Body, scene::Scene},
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub mass: Fix,
    pub inertia: Fix,
    pub bounciness: Fix,
    pub friction: Fix,
    pub dynamic_friction: Fix,
    pub gravity: Fix,
    pub air_drag: Fix,
    pub rotational_drag: Fix,
    pub velocity: Vec2,
    pub angular_velocity: Fix,
}
impl Config {
    pub fn new_body(&self, orientation: Fix) -> scheduler::Result<Body> {
        let inverse = |value: Fix| {
            if value == Fix::ZERO {
                Ok(Fix::ZERO)
            } else {
                reciprocal(value)
            }
        };
        Ok(Body {
            inverse_mass: inverse(self.mass)?,
            inverse_inertia: inverse(self.inertia)?,
            velocity: self.velocity,
            angular_velocity: self.angular_velocity,
            orientation,
            bounciness: self.bounciness,
            static_friction: self.friction,
            dynamic_friction: self.dynamic_friction,
            gravity_scale: self.gravity,
            air_drag: self.air_drag,
            rotational_drag: self.rotational_drag,
            ..Body::default()
        })
    }
}
fn reciprocal(value: Fix) -> scheduler::Result<Fix> {
    Fix::ONE.checked_div(value).map_err(|_| "division_by_zero")
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sorting {
    pub layer: Option<String>,
    pub order: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub config: Config,
    pub resizable: bool,
    pub prev_scale: Fix,
    pub frames_until_drop: i64,
    pub magnet: Option<usize>,
    pub sorting: Option<Sorting>,
    pub original_sorting: Sorting,
}
pub struct Context<'a> {
    pub scene: &'a mut Scene,
    pub object: u32,
    pub initialized: bool,
    pub transform_scale: Fix,
}
impl Context<'_> {
    fn index(&self) -> scheduler::Result<usize> {
        self.scene
            .objects
            .iter()
            .position(|o| o.id == self.object)
            .ok_or("index_out_of_range")
    }
    fn composite(&self, index: usize) -> Option<usize> {
        self.scene.objects[index]
            .composite
            .filter(|&c| !self.scene.composites[c].destroyed)
    }
    pub fn body(&mut self) -> scheduler::Result<&mut Body> {
        let index = self.index()?;
        Ok(if let Some(c) = self.composite(index) {
            &mut self.scene.composites[c].body
        } else {
            &mut self.scene.objects[index].body
        })
    }
    pub fn add_force(
        &mut self,
        mut force: Vec2,
        position: Option<Vec2>,
        impulse: bool,
    ) -> scheduler::Result<()> {
        if impulse {
            force *= Fix::from_int(60);
        }
        if let Some(position) = position {
            let center = self.scene.objects[self.index()?].transform.position;
            self.body()?
                .apply_impulse(force, (position - center).normalized_safe());
        } else {
            self.body()?.force += force;
        }
        Ok(())
    }
}
impl State {
    pub fn hold(&mut self, frames: i64, magnet: Option<usize>, change_sorting: bool) {
        if self.frames_until_drop <= 0
            && let Some(sorting) = &mut self.sorting
        {
            self.original_sorting = sorting.clone();
            if change_sorting {
                sorting.layer = Some("Player".into());
                sorting.order = 0;
            }
        }
        self.magnet = magnet;
        self.frames_until_drop = frames;
    }
    pub fn drop_held(&mut self) {
        if self.frames_until_drop >= 0 {
            self.magnet = None;
            if let Some(sorting) = &mut self.sorting {
                *sorting = self.original_sorting.clone();
            }
        }
        self.frames_until_drop = -1;
    }
    pub fn update(&mut self, context: &mut Context) -> scheduler::Result<()> {
        self.frames_until_drop = self.frames_until_drop.wrapping_sub(1);
        if self.frames_until_drop == 0 {
            self.drop_held();
        }
        if !self.resizable
            && context.initialized
            && self.config.mass != Fix::ZERO
            && self.config.inertia != Fix::ZERO
            && self.prev_scale != context.transform_scale
        {
            let index = context.index()?;
            if context.composite(index).is_some()
                && context.scene.objects[index].collider.kind() != 2
            {
                self.prev_scale = context.transform_scale;
                return Ok(());
            }
            let factor = self
                .prev_scale
                .checked_div(context.transform_scale)
                .map_err(|_| "division_by_zero")?;
            let body = context.body()?;
            body.inverse_mass *= factor;
            body.inverse_inertia *= factor;
            self.prev_scale = context.transform_scale;
        }
        Ok(())
    }
    pub fn inverse_mass(&self, context: &mut Context) -> scheduler::Result<Fix> {
        if context.initialized {
            Ok(context.body()?.inverse_mass)
        } else {
            reciprocal(self.config.mass)
        }
    }
    pub fn inverse_inertia(&self, context: &mut Context) -> scheduler::Result<Fix> {
        if context.initialized {
            Ok(context.body()?.inverse_inertia)
        } else {
            reciprocal(self.config.inertia)
        }
    }
    pub fn set_inverse_mass(&mut self, value: Fix, context: &mut Context) -> scheduler::Result<()> {
        if context.initialized {
            context.body()?.inverse_mass = value;
        } else {
            self.config.mass = reciprocal(value)?;
        }
        Ok(())
    }
    pub fn set_inverse_inertia(
        &mut self,
        value: Fix,
        context: &mut Context,
    ) -> scheduler::Result<()> {
        if context.initialized {
            context.body()?.inverse_inertia = value;
        } else {
            self.config.inertia = reciprocal(value)?;
        }
        Ok(())
    }
    pub fn set_velocity(&mut self, value: Vec2, context: &mut Context) -> scheduler::Result<()> {
        if context.initialized {
            context.body()?.velocity = value;
        } else {
            self.config.velocity = value;
        }
        Ok(())
    }
    pub fn set_angular_velocity(
        &mut self,
        value: Fix,
        context: &mut Context,
    ) -> scheduler::Result<()> {
        if context.initialized {
            context.body()?.angular_velocity = value;
        } else {
            self.config.angular_velocity = value;
        }
        Ok(())
    }
}
