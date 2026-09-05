//! Ordered scene raycasts from DetPhysics.RaycastToClosest.
use super::{scene::Scene, shape::Collider};
use crate::{fixed::Fix, vector::Vec2};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CastHit {
    pub object: Option<u32>,
    pub point: Vec2,
    pub distance: Fix,
}
impl Scene {
    pub fn raycast(&mut self, origin: Vec2, direction: Vec2, distance: Fix, mask: u32) -> CastHit {
        self.remove_destroyed();
        let mut result = CastHit {
            object: None,
            point: Vec2::ZERO,
            distance: Fix::MAX,
        };
        for kind in [2, 0, 1] {
            for o in &self.objects {
                if o.collider.kind() != kind || !o.active || mask & (1 << o.layer) == 0 {
                    continue;
                }
                let hit = match o.collider {
                    Collider::Circle(c) => c.raycast(origin, direction),
                    Collider::Box(b) => b.raycast(origin, direction),
                    Collider::RoundedRect(r) => r.raycast(origin, direction),
                };
                if let Some(hit) = hit
                    && hit.near < result.distance
                    && hit.near <= distance
                {
                    result = CastHit {
                        object: Some(o.id),
                        point: origin + hit.near * direction,
                        distance: hit.near,
                    };
                }
            }
        }
        result
    }
}
