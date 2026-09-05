//! Collider and GameObject state used by composed actor ability handoffs.
use crate::{
    actors::Actor,
    fixed::Fix,
    physics::{scene::Scene, shape::Collider},
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub node: usize,
    pub active: bool,
    pub body_simulated: bool,
    #[serde(rename = "box")]
    pub collider: BoxComponent,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoxComponent {
    pub scale: Fix,
    pub min_scale: Fix,
    pub max_scale: Fix,
    pub minimum_extents: Fix,
    pub start_extents: Vec2,
    pub initialized: bool,
}
impl BoxComponent {
    pub fn set_scale(
        &mut self,
        value: Fix,
        transform_scale: &mut Fix,
        object_id: u32,
        scene: &mut Scene,
    ) -> scheduler::Result<()> {
        let c = self;
        let inverse = Fix::ONE
            .checked_div(c.scale)
            .map_err(|_| "division_by_zero")?;
        c.scale = value.max(c.min_scale).min(c.max_scale);
        let offset_factor = c
            .scale
            .checked_div(*transform_scale)
            .map_err(|_| "division_by_zero")?;
        scene
            .objects
            .iter_mut()
            .find(|o| o.id == object_id)
            .ok_or("missing_actor_object")?
            .transform
            .offset *= offset_factor;
        *transform_scale = c.scale;
        let factor = inverse * c.scale;
        if c.initialized {
            let object = scene
                .objects
                .iter_mut()
                .find(|o| o.id == object_id)
                .ok_or("missing_actor_object")?;
            let Collider::Box(shape) = &mut object.collider else {
                return Err("actor_box_required");
            };
            shape.right *= factor;
            shape.up *= factor;
            shape.inverse_extents = Vec2::new(
                shape
                    .inverse_extents
                    .x
                    .checked_div(factor)
                    .map_err(|_| "division_by_zero")?,
                shape
                    .inverse_extents
                    .y
                    .checked_div(factor)
                    .map_err(|_| "division_by_zero")?,
            );
        } else {
            c.start_extents *= factor;
        }
        Ok(())
    }
}
impl Actor {
    pub fn set_box_scale(&mut self, value: Fix, scene: &mut Scene) -> scheduler::Result<()> {
        self.entity
            .as_mut()
            .ok_or("missing_actor_entity")?
            .collider
            .set_scale(
                value,
                &mut self.controller.physics.scale,
                self.object,
                scene,
            )
    }
    pub fn sync_box(&self, scene: &mut Scene) -> scheduler::Result<()> {
        self.entity
            .as_ref()
            .ok_or("missing_actor_entity")?
            .collider
            .sync(self.object, scene)
    }
}
impl BoxComponent {
    pub fn sync(&self, object_id: u32, scene: &mut Scene) -> scheduler::Result<()> {
        let c = self;
        if !c.initialized {
            return Ok(());
        }
        let index = scene
            .objects
            .iter()
            .position(|o| o.id == object_id)
            .ok_or("missing_actor_object")?;
        if let Some(composite) = scene.objects[index]
            .composite
            .filter(|&i| !scene.composites[i].destroyed)
        {
            scene.synchronize_composite(composite);
        } else {
            let object = &mut scene.objects[index];
            let Collider::Box(shape) = &mut object.collider else {
                return Err("actor_box_required");
            };
            let basis = Vec2::angle(object.transform.rotation);
            let divide = |v: Vec2, d: Fix| -> scheduler::Result<Vec2> {
                Ok(Vec2::new(
                    v.x.checked_div(d).map_err(|_| "division_by_zero")?,
                    v.y.checked_div(d).map_err(|_| "division_by_zero")?,
                ))
            };
            shape.right = divide(Vec2::RIGHT, shape.inverse_extents.x)?.complex_mul(basis);
            shape.up = divide(Vec2::UP, shape.inverse_extents.y)?.complex_mul(basis);
            object.collider.set_center(
                object.transform.position
                    + object
                        .transform
                        .basis()
                        .complex_mul(object.transform.offset),
            );
            object.body.orientation = object.transform.rotation;
        }
        Ok(())
    }
}
