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
                if o.collider.kind() != kind || !o.is_active() || mask & (1 << o.layer) == 0 {
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

/// Complete RaycastInformation for the rounded-rectangle-only query.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SurfaceHit {
    pub object: Option<u32>,
    pub near_position: Vec2,
    pub far_position: Vec2,
    pub near_distance: Fix,
    pub far_distance: Fix,
    pub layer: usize,
}
impl Default for SurfaceHit {
    fn default() -> Self {
        Self {
            object: None,
            near_position: Vec2::ZERO,
            far_position: Vec2::ZERO,
            near_distance: Fix::MAX,
            far_distance: Fix::ZERO,
            layer: 0,
        }
    }
}
impl Scene {
    pub fn raycast_rounded(&mut self, origin: Vec2, direction: Vec2, distance: Fix) -> SurfaceHit {
        self.remove_destroyed();
        let mut result = SurfaceHit::default();
        for object in &self.objects {
            let Collider::RoundedRect(shape) = object.collider else {
                continue;
            };
            if object.is_active()
                && let Some(hit) = shape.raycast(origin, direction)
                && hit.near < result.near_distance
                && hit.near <= distance
            {
                result = SurfaceHit {
                    object: Some(object.id),
                    near_position: origin + hit.near * direction,
                    far_position: origin + hit.far * direction,
                    near_distance: hit.near,
                    far_distance: hit.far,
                    layer: object.layer,
                };
            }
        }
        result
    }
}
