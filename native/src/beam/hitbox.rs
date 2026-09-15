//! BeamObject.SetDimensions and its DPhysicsBox initialization/update path.
use crate::{curve::Curve, fixed::Fix, geometry::OrientedBox, vector::Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub base_width: Fix,
    pub extra_length: Fix,
    pub minimum_extents: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dimensions {
    pub position: Vec2,
    pub width: Fix,
    pub direction: Vec2,
    pub distance: Fix,
    pub age: Fix,
    pub player_id: i32,
    pub beam_id: i32,
    pub curve: Curve,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hitbox {
    pub position: Vec2,
    pub right: Vec2,
    pub rotation: Fix,
    pub offset: Vec2,
    pub start_extents: Vec2,
    pub owner_id: i32,
    pub beam_id: i32,
    pub collider: Option<OrientedBox>,
    pub orientation: Option<Fix>,
}
impl Default for Hitbox {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            right: Vec2::RIGHT,
            rotation: Fix::ZERO,
            offset: Vec2::ZERO,
            start_extents: Vec2::ZERO,
            owner_id: 0,
            beam_id: 0,
            collider: None,
            orientation: None,
        }
    }
}
impl Hitbox {
    pub fn set_dimensions(&mut self, config: Config, dimensions: &Dimensions) {
        self.set_dimension_values(
            config,
            dimensions.position,
            dimensions.width,
            dimensions.direction,
            dimensions.distance,
            dimensions.age,
            dimensions.player_id,
            dimensions.beam_id,
            &dimensions.curve,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_dimension_values(
        &mut self,
        config: Config,
        position: Vec2,
        width: Fix,
        direction: Vec2,
        distance: Fix,
        age: Fix,
        player_id: i32,
        beam_id: i32,
        curve: &Curve,
    ) {
        self.owner_id = player_id;
        self.beam_id = beam_id;
        let width = width * (config.base_width * curve.evaluate(age));
        self.position = position;
        self.offset = Vec2::new(
            distance * Fix::from_f64(0.5) + config.extra_length,
            Fix::ZERO,
        );
        let mut extents = Vec2::new(self.offset.x, width);
        if extents.x == Fix::ZERO {
            extents.x = config.minimum_extents;
        }
        if extents.y == Fix::ZERO {
            extents.y = config.minimum_extents;
        }
        if let Some(collider) = &mut self.collider {
            collider.right *= collider.inverse_extents.x;
            collider.up *= collider.inverse_extents.y;
            collider.inverse_extents = Vec2::new(Fix::ONE / extents.x, Fix::ONE / extents.y);
            collider.right *= extents.x;
            collider.up *= extents.y;
        } else {
            self.start_extents = extents;
        }
        self.right = direction.normalized_safe();
        self.rotation = self.right.normalized_angle();
        if let Some(collider) = &mut self.collider {
            let basis = Vec2::angle(self.rotation);
            collider.right = (Vec2::RIGHT / collider.inverse_extents.x).complex_mul(basis);
            collider.up = (Vec2::UP / collider.inverse_extents.y).complex_mul(basis);
            collider.center = self.position + self.right.complex_mul(self.offset);
            self.orientation = Some(self.rotation);
        }
    }
    pub fn initialize(&mut self, config: Config) {
        if self.start_extents.x == Fix::ZERO {
            self.start_extents.x = config.minimum_extents;
        }
        if self.start_extents.y == Fix::ZERO {
            self.start_extents.y = config.minimum_extents;
        }
        self.collider = Some(OrientedBox {
            center: self.position + self.offset.complex_mul(self.right),
            right: self.right * self.start_extents.x,
            up: self.right.perp() * self.start_extents.y,
            inverse_extents: Vec2::new(
                (Fix::ONE / self.start_extents.x).abs(),
                (Fix::ONE / self.start_extents.y).abs(),
            ),
        });
        self.orientation = Some(self.rotation);
    }
}
