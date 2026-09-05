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

/// RaycastInformation caches PhysicsParent by value, including sort hierarchy.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OrderedHit {
    #[serde(flatten)]
    pub hit: SurfaceHit,
    pub hierarchy: i32,
}
impl Scene {
    /// RaycastAll appends to the caller's list and sorts the whole list. Explicit
    /// hierarchy identities are required because equal-distance hits use them.
    pub fn raycast_all(
        &mut self,
        origin: Vec2,
        direction: Vec2,
        distance: Fix,
        mask: u32,
        caster: u32,
        output: &mut Vec<OrderedHit>,
    ) -> crate::scheduler::Result<bool> {
        self.remove_destroyed();
        let mut found = false;
        for kind in [2, 0, 1] {
            for o in &self.objects {
                if o.collider.kind() != kind
                    || !o.is_active()
                    || mask & (1u32 << o.layer) == 0
                    || o.id == caster
                {
                    continue;
                }
                let hit = match o.collider {
                    Collider::Circle(c) => c.raycast(origin, direction),
                    Collider::Box(b) => b.raycast(origin, direction),
                    Collider::RoundedRect(r) => r.raycast(origin, direction),
                };
                if let Some(hit) = hit
                    && hit.near <= distance
                {
                    output.push(OrderedHit {
                        hierarchy: o.hierarchy.ok_or("missing_raycast_hierarchy")?,
                        hit: SurfaceHit {
                            object: Some(o.id),
                            near_position: origin + hit.near * direction,
                            far_position: origin + hit.far * direction,
                            near_distance: hit.near,
                            far_distance: hit.far,
                            layer: o.layer,
                        },
                    });
                    found = true;
                }
            }
        }
        let length = output.len();
        crate::mono_sort::sort(output, length, |a, b| {
            if a.hit.near_distance != b.hit.near_distance {
                return Ok((a.hit.near_distance - b.hit.near_distance).sign() as i32);
            }
            Ok(a.hierarchy.wrapping_sub(b.hierarchy))
        })?;
        Ok(found)
    }
}
