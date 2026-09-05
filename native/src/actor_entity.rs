//! Collider and GameObject state used by composed actor ability handoffs.
use crate::{actors::Actor, fixed::Fix, physics::scene::Scene, scheduler};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub node: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub present: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destroyed: Option<bool>,
    pub active: bool,
    pub body_simulated: bool,
    #[serde(flatten)]
    pub collider: crate::actor_collider::Component,
}
impl Actor {
    pub(crate) fn initialize_collider(
        &mut self,
        scene: &mut Scene,
        hierarchy: i32,
    ) -> scheduler::Result<()> {
        if self
            .entity
            .as_ref()
            .ok_or("missing_actor_entity")?
            .collider
            .skips_init()
        {
            return Ok(());
        }
        let Some(object) = self.pending_object.as_mut() else {
            return Err("actor_collider_already_initialized");
        };
        let c = &mut self.entity.as_mut().ok_or("missing_actor_entity")?.collider;
        c.initialize_geometry(object)?;
        object.body = crate::physics::Body {
            inverse_mass: -Fix::ONE,
            orientation: object.transform.rotation,
            ..Default::default()
        };
        object.hierarchy = Some(hierarchy);
        self.collision_hierarchy = hierarchy;
        scene.objects.push(self.pending_object.take().unwrap());
        Ok(())
    }
    pub fn is_present(&self) -> bool {
        self.entity
            .as_ref()
            .is_none_or(|e| e.present != Some(false))
    }
    pub fn set_collider_scale(&mut self, value: Fix, scene: &mut Scene) -> scheduler::Result<()> {
        self.entity
            .as_mut()
            .ok_or("missing_actor_entity")?
            .collider
            .set_scale(
                value,
                &mut self.controller.physics.scale,
                self.object,
                scene,
                self.pending_object.as_mut(),
            )
    }
    pub fn sync_collider(&self, scene: &mut Scene) -> scheduler::Result<()> {
        self.entity
            .as_ref()
            .ok_or("missing_actor_entity")?
            .collider
            .sync(self.object, scene)
    }
}

impl crate::actors::Runtime {
    pub fn destroy_actor(
        actors: &mut [Actor],
        context: &mut crate::actors::Context,
        id: usize,
    ) -> scheduler::Result<()> {
        if !actors[id].is_present() {
            return Err("null_reference");
        }
        let root = actors[id]
            .entity
            .as_ref()
            .ok_or("missing_actor_entity")?
            .node;
        context.schedule.destroy_object(Some(root))?;
        for actor in actors {
            let Some(entity) = &mut actor.entity else {
                continue;
            };
            let mut node = Some(entity.node);
            while let Some(id) = node {
                if id == root {
                    entity.destroyed = Some(true);
                    if let Some(slime) = &mut actor.slime {
                        slime.managed_destroyed = Some(true);
                    }
                    if let Some(object) = context
                        .scene
                        .objects
                        .iter_mut()
                        .find(|o| o.id == actor.object)
                        .or(actor.pending_object.as_mut())
                    {
                        object.destroyed = true;
                    }
                    break;
                }
                node = context.schedule.nodes[id].parent;
            }
        }
        Ok(())
    }
}
