//! Rigid-body primitives and ordered scenes from DetPhysics and PhysTools.
//! Full gameplay integration, ropes and beams are still in progress.
use crate::{fixed::Fix, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

mod composite;
pub mod contact;
pub mod scene;
pub mod shape;
mod sweep;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Body {
    pub inverse_mass: Fix,
    pub velocity: Vec2,
    pub force: Vec2,
    pub angular_velocity: Fix,
    pub orientation: Fix,
    pub inverse_inertia: Fix,
    pub bounciness: Fix,
    pub static_friction: Fix,
    pub dynamic_friction: Fix,
    pub gravity_scale: Fix,
    pub air_drag: Fix,
    pub rotational_drag: Fix,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Contact {
    pub penetration: Fix,
    pub normal: Vec2,
    pub point: Vec2,
}

impl Body {
    pub fn integrate(&mut self, dt: Fix, gravity: Fix) -> Vec2 {
        self.velocity += dt * Vec2::new(Fix::ZERO, gravity * self.gravity_scale)
            + DT * self.force * self.inverse_mass;
        self.orientation += dt * self.angular_velocity;
        self.orientation = (self.orientation.checked_rem(Fix::TAU, true).unwrap() + Fix::TAU)
            .checked_rem(Fix::TAU, true)
            .unwrap();
        self.velocity *= if dt == Fix::ZERO {
            Fix::ONE
        } else {
            self.air_drag
        };
        self.angular_velocity *= if dt == Fix::ZERO {
            Fix::ONE
        } else {
            self.rotational_drag
        };
        self.force = Vec2::ZERO;
        dt * self.velocity
    }
    pub fn apply_impulse(&mut self, impulse: Vec2, contact: Vec2) {
        self.velocity += self.inverse_mass * impulse;
        self.angular_velocity += self.inverse_inertia * contact.cross(impulse);
    }
}

/// Pass the effective combined body and position for a composite, as Unity does.
pub fn resolve(contact: Contact, a: &mut Body, b: &mut Body, pos_a: Vec2, pos_b: Vec2) {
    let before_a = *a;
    let before_b = *b;
    let arm_a = (contact.point - pos_a).normalized_safe();
    let arm_b = (contact.point - pos_b).normalized_safe();
    let relative = before_b.velocity + before_b.angular_velocity * arm_b.perp()
        - before_a.velocity
        - before_a.angular_velocity * arm_a.perp();
    let normal_speed = relative.dot(contact.normal);
    if normal_speed > Fix::ZERO {
        return;
    }
    let restitution = Fix::ONE + before_a.bounciness.max(before_b.bounciness);
    let angular_a = arm_a.cross(contact.normal);
    let angular_b = arm_b.cross(contact.normal);
    let inverse_denominator = Fix::ONE
        / (before_a.inverse_mass
            + before_b.inverse_mass
            + angular_a * angular_a * before_a.inverse_inertia
            + angular_b * angular_b * before_b.inverse_inertia);
    let impulse_size = -restitution * normal_speed * inverse_denominator;
    let impulse = impulse_size * contact.normal;
    a.apply_impulse(-impulse, arm_a);
    b.apply_impulse(impulse, arm_b);
    // PhysTools computes friction from pre-impulse value copies, not updated bodies.
    let tangent = contact.normal.perp();
    let tangent = Fix::from_int(tangent.dot(relative).sign()) * tangent;
    let friction_size = -tangent.dot(relative) * inverse_denominator;
    let static_friction = before_a.static_friction.max(before_b.static_friction);
    let friction = if friction_size.abs() < impulse_size * static_friction {
        friction_size * tangent
    } else {
        -impulse_size * tangent * before_a.dynamic_friction.max(before_b.dynamic_friction)
    };
    a.apply_impulse(-friction, arm_a);
    b.apply_impulse(friction, arm_b);
}

pub fn positional_correction(
    contact: Contact,
    inverse_a: Fix,
    inverse_b: Fix,
    full: bool,
) -> [Vec2; 2] {
    let penetration = (contact.penetration - Fix::from_f32(0.01)).max(Fix::ZERO);
    let correction = penetration / (inverse_a + inverse_b)
        * if full { Fix::ONE } else { Fix::from_f32(0.3) }
        * contact.normal;
    [-inverse_a * correction, inverse_b * correction]
}
