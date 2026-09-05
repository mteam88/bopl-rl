//! PlayerCollision's empty-player clone allocation and preparation. Prefab state
//! comes from the pinned player asset; component bindings share scheduler IDs
//! with effects and all other dynamically allocated objects.
use crate::{
    actors::{Actor, Component, Context, Report, Role, Runtime},
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

pub const MAX_CLONES: i32 = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefabComponent {
    pub state: scheduler::Component,
    pub role: Option<Role>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    pub actor: Actor,
    /// Node and component references are local to this prefab.
    pub nodes: Vec<scheduler::Node>,
    /// Unity's component traversal order, including the fixed transform.
    pub components: Vec<PrefabComponent>,
    /// Awake registration can differ from component traversal order.
    pub registration: Vec<usize>,
}

impl Prefab {
    pub fn validate(&self) -> scheduler::Result<()> {
        let entity = self.actor.entity.as_ref().ok_or("missing_clone_entity")?;
        if self.actor.pending_object.is_none()
            || entity.node >= self.nodes.len()
            || entity.collider.initialized()
            || self
                .components
                .iter()
                .any(|c| c.state.owner >= self.nodes.len())
            || self.registration.iter().any(|&i| {
                self.components
                    .get(i)
                    .is_none_or(|c| c.state.kind != scheduler::Kind::Updatable || c.role.is_none())
            })
        {
            return Err("invalid_clone_prefab");
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.parent.is_some_and(|p| p >= i)
                || node
                    .children
                    .iter()
                    .any(|&c| c <= i || self.nodes.get(c).is_none_or(|n| n.parent != Some(i)))
            {
                return Err("invalid_clone_prefab_nodes");
            }
        }
        Ok(())
    }
}

impl Runtime {
    pub fn spawn_abilityless_clone(
        actors: &mut Vec<Actor>,
        context: &mut Context,
        player: usize,
        target: Vec2,
        prefab: Option<&Prefab>,
        report: &mut Report,
    ) -> scheduler::Result<Option<usize>> {
        let owner = context
            .combat
            .as_mut()
            .ok_or("missing_actor_combat")?
            .book
            .players
            .get_mut(player)
            .ok_or("null_reference")?;
        if owner.bodies_alive > MAX_CLONES {
            return Ok(None);
        }
        // The original increments before accessing reviveEffectPrefab.
        owner.bodies_alive = owner.bodies_alive.wrapping_add(1);
        let player_id = owner.id;
        let prefab = prefab.ok_or("null_reference")?;
        prefab.validate()?;
        let actor_id = actors.len();
        let object_id = actors
            .iter()
            .map(|a| a.object)
            .chain(context.scene.objects.iter().map(|o| o.id))
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("actor_object_id_exhausted")?;
        let node_base = context.schedule.nodes.len();
        let component_base = context.schedule.components.len();
        let mut actor = prefab.actor.clone();
        actor.object = object_id;
        actor.body.holder = Some(player_id);
        actor.input = actors
            .iter()
            .find(|a| a.body.holder == Some(player_id))
            .map_or(Vec2::ZERO, |a| a.input);
        actor.entity.as_mut().unwrap().node += node_base;
        actor.pending_object.as_mut().unwrap().id = object_id;
        actor.controller.physics.motion.position = target;
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        for node in &prefab.nodes {
            let mut node = node.clone();
            node.parent = node.parent.map(|i| i + node_base);
            for child in &mut node.children {
                *child += node_base;
            }
            context.schedule.nodes.push(node);
        }
        context.components.resize(component_base, None);
        context.schedule.begin_prefab();
        for component in &prefab.components {
            let mut state = component.state.clone();
            state.owner += node_base;
            context.schedule.components.push(state);
            context
                .components
                .push(component.role.map(|role| Component {
                    actor: actor_id,
                    role,
                }));
        }
        for &local in &prefab.registration {
            context.schedule.register(Some(component_base + local));
        }
        context.schedule.end_prefab()?;
        if let Some(owner) = actors
            .iter()
            .find(|a| a.is_present() && a.body.holder == Some(player_id))
            .and_then(|a| a.slime.as_ref())
            .and_then(|s| s.abilities.as_ref())
            && let Some(routing) = actor.slime.as_mut().and_then(|s| s.abilities.as_mut())
        {
            routing.can_use = owner.can_use;
            routing.buttons = owner.buttons.clone();
        }
        actors.push(actor);
        let slime = actors[actor_id]
            .slime
            .as_mut()
            .ok_or("missing_clone_slime")?;
        let routing = slime.abilities.as_mut().ok_or("missing_ability_routing")?;
        routing.slots.clear();
        routing.indicators.clear();
        slime
            .animator
            .render
            .as_mut()
            .ok_or("null_reference")?
            .sprite = None;
        Self::spawn_actor(
            actors,
            context,
            actor_id,
            crate::actor_spawn::Action::Prepare,
            target,
            report,
        )?;
        Ok(Some(actor_id))
    }
}
