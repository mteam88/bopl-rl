//! Unity collider component state, separate from registered scene geometry.
use crate::{
    fixed::Fix,
    physics::{
        scene::{Object, Scene},
        shape::Collider,
    },
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Component {
    Box { r#box: BoxComponent },
    Circle { circle: CircleComponent },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircleComponent {
    pub scale: Fix,
    pub min_scale: Fix,
    pub max_scale: Fix,
    pub start_radius: Fix,
    pub initialized: bool,
    pub manually_call_init: bool,
}

impl Component {
    pub fn initialized(&self) -> bool {
        match self {
            Self::Box { r#box: c } => c.initialized,
            Self::Circle { circle: c } => c.initialized,
        }
    }
    pub fn skips_init(&self) -> bool {
        matches!(self, Self::Circle { circle: c } if c.manually_call_init)
    }
    pub fn initialize_geometry(&mut self, object: &mut Object) -> scheduler::Result<()> {
        let basis = object.transform.basis();
        let center = object.transform.position + basis.complex_mul(object.transform.offset);
        match self {
            Self::Box { r#box: c } => {
                if c.start_extents.x == Fix::ZERO {
                    c.start_extents.x = c.minimum_extents;
                }
                if c.start_extents.y == Fix::ZERO {
                    c.start_extents.y = c.minimum_extents;
                }
                object.collider = Collider::Box(crate::geometry::OrientedBox {
                    center,
                    right: basis * c.start_extents.x,
                    up: basis.perp() * c.start_extents.y,
                    inverse_extents: Vec2::new(
                        Fix::ONE
                            .checked_div(c.start_extents.x)
                            .map_err(|_| "division_by_zero")?
                            .abs(),
                        Fix::ONE
                            .checked_div(c.start_extents.y)
                            .map_err(|_| "division_by_zero")?
                            .abs(),
                    ),
                });
                c.initialized = true;
            }
            Self::Circle { circle: c } => {
                object.collider = Collider::Circle(crate::geometry::Circle {
                    center,
                    radius: c.start_radius,
                });
                c.initialized = true;
            }
        }
        Ok(())
    }
    pub fn set_scale(
        &mut self,
        value: Fix,
        transform_scale: &mut Fix,
        object_id: u32,
        scene: &mut Scene,
        pending: Option<&mut Object>,
    ) -> scheduler::Result<()> {
        match self {
            Self::Box { r#box: c } => {
                c.set_scale(value, transform_scale, object_id, scene, pending)
            }
            Self::Circle { circle: c } => {
                let inverse = Fix::ONE
                    .checked_div(c.scale)
                    .map_err(|_| "division_by_zero")?;
                c.scale = value.max(c.min_scale).min(c.max_scale);
                let factor = c
                    .scale
                    .checked_div(*transform_scale)
                    .map_err(|_| "division_by_zero")?;
                let object = scene
                    .objects
                    .iter_mut()
                    .find(|o| o.id == object_id)
                    .or(pending)
                    .ok_or("missing_actor_object")?;
                object.transform.offset *= factor;
                *transform_scale = c.scale;
                let factor = inverse * c.scale;
                if c.initialized {
                    let Collider::Circle(shape) = &mut object.collider else {
                        return Err("actor_circle_required");
                    };
                    shape.radius *= factor;
                } else {
                    c.start_radius *= factor;
                }
                Ok(())
            }
        }
    }
    pub fn set_radius(
        &mut self,
        value: Fix,
        object_id: u32,
        scene: &mut Scene,
    ) -> scheduler::Result<()> {
        let Self::Circle { circle: c } = self else {
            return Err("actor_circle_required");
        };
        if c.initialized {
            let object = scene
                .objects
                .iter_mut()
                .find(|o| o.id == object_id)
                .ok_or("missing_actor_object")?;
            let Collider::Circle(shape) = &mut object.collider else {
                return Err("actor_circle_required");
            };
            shape.radius = value;
        } else {
            c.start_radius = value;
        }
        Ok(())
    }
    pub fn sync(&self, object_id: u32, scene: &mut Scene) -> scheduler::Result<()> {
        match self {
            Self::Box { r#box: c } => c.sync(object_id, scene),
            Self::Circle { circle: c } => {
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
    }
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
        pending: Option<&mut crate::physics::scene::Object>,
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
        let object = scene
            .objects
            .iter_mut()
            .find(|o| o.id == object_id)
            .or(pending)
            .ok_or("missing_actor_object")?;
        object.transform.offset *= offset_factor;
        *transform_scale = c.scale;
        let factor = inverse * c.scale;
        if c.initialized {
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
