//! Shared PlayerPhysics attachment and detachment state. Platform identity is
//! retained after UnGround, as in Unity; callers resolve its current surface.
use crate::{
    fixed::{Fix, MathError},
    surface::Surface,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub position: Vec2,
    pub rotation: Fix,
    pub up: Vec2,
    pub right: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub radius: Fix,
    pub grounded: bool,
    pub ground: Option<u32>,
    pub local: Fix,
    pub grounded_speed: Fix,
}

impl State {
    pub fn attach(&mut self, id: u32, surface: &Surface, location: Vec2) -> Result<(), MathError> {
        self.ground = Some(id);
        self.grounded = true;
        self.grounded_speed = self.velocity.cross(surface.normal(self.position));
        self.local = surface.local_position(location, self.radius)?;
        self.position = surface.position(self.local, self.radius);
        self.velocity = Vec2::ZERO;
        self.external_velocity = Vec2::ZERO;
        self.up = surface.normal(self.position);
        self.right = self.up.perp_cw();
        self.rotation = self.right.normalized_angle();
        Ok(())
    }

    /// Returns the old rotation for OnNoLongerGrounded. Dispatch this event
    /// immediately, then play the optional jump sound. Neither is stored here.
    /// Even repeated calls while airborne displace the body and emit the event.
    /// Resolve any non-null retained platform, including one marked IsDestroyed.
    pub fn unground(&mut self, surface: Option<&Surface>, null_rotation: bool) -> Fix {
        self.grounded_speed = Fix::ZERO;
        self.grounded = false;
        if let Some(surface) = surface {
            self.position += surface.normal(self.position) * Fix::from_f32(0.05);
        }
        let rotation = self.rotation;
        if null_rotation {
            self.rotation = Fix::ZERO;
            self.right = Vec2::new(Fix::ZERO.cos(), Fix::ZERO.sin());
            self.up = -self.right.perp_cw();
        }
        rotation
    }
}
