//! Target effects from DetPhysics.renderBeams, Orb.ApplyBeam and BlackHole.
use super::{
    color::Colors,
    graph::{Beam, Role},
};
use crate::{fixed::Fix, vector::Vec2, world::DT};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Wall,
    Terrain,
    Lethal,
    Player,
    Affector,
    Orb,
    Other,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Orb {
    pub kind: i32,
    pub charge: Fix,
    pub rate: Fix,
    pub firing: bool,
    pub colors: Colors,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    pub kind: Kind,
    pub destroyed: bool,
    pub component: bool,
    pub component_destroyed: bool,
    pub force: Vec2,
    pub velocity: Vec2,
    pub mass: Fix,
    pub orb: Orb,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub water: Fix,
    pub time_scale: Fix,
    pub protected: [bool; 6],
    pub platform_push: Fix,
    pub black_hole_push: Fix,
    pub body_push: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Impact {
    pub beam: Beam,
    pub role: Role,
    pub water_reflection: bool,
    pub target: Option<usize>,
}
impl Config {
    fn stopped(&self, owner: i32) -> bool {
        owner != 1000
            && self.time_scale < Fix::from_f32(0.01)
            && !self.protected[(owner - 1) as usize]
    }
}
pub fn apply(config: &Config, impacts: &[Impact], targets: &mut [Target]) {
    for impact in impacts {
        if matches!(
            impact.role,
            Role::Child | Role::ParentAndChild | Role::MaybeChild
        ) || impact.water_reflection
            || impact.role == Role::ChildWithEffect
        {
            continue;
        }
        let Some(index) = impact.target else {
            continue;
        };
        let target = &mut targets[index];
        if target.destroyed || !target.component || target.component_destroyed {
            continue;
        }
        let beam = impact.beam;
        let scale = beam.scale.min(Fix::from_int(40));
        match target.kind {
            Kind::Wall | Kind::Terrain | Kind::Lethal => {
                let end = beam.position + beam.direction * beam.distance;
                if end.y > config.water + Fix::ONE && !config.stopped(beam.owner_id) {
                    target.force += beam.direction * config.platform_push * scale;
                }
            }
            Kind::Player => target.force += beam.direction * config.body_push * scale,
            Kind::Affector if !config.stopped(beam.owner_id) => {
                let force = beam.direction * config.black_hole_push * scale;
                let mass = target
                    .mass
                    .abs()
                    .sqrt()
                    .expect("negative black hole mass magnitude");
                target.velocity += Fix::from_int(target.mass.sign2()) * force / mass;
            }
            Kind::Orb => {
                if target.orb.kind != 5 {
                    target.orb.charge = Fix::ZERO;
                    target.orb.kind = 5;
                    target.orb.firing = false;
                    target.orb.colors = beam.colors;
                }
                target.orb.charge += target.orb.rate * DT;
                if target.orb.charge > Fix::ONE {
                    target.orb.charge = Fix::ONE;
                    target.orb.firing = true;
                }
            }
            _ => {}
        }
    }
}
