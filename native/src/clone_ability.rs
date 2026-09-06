//! Fresh pooled abilities selected by the source player's indicator sprites.
use crate::{
    ability::Handler,
    ability_routing::{Indicator, IndicatorLifecycle},
    actor_clone::Prefab,
    actors::{Actor, Context, Runtime},
    scheduler,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Registration {
    pub node: usize,
    pub component: usize,
    pub transform: Option<usize>,
}

impl Runtime {
    pub fn copy_clone_abilities(
        actors: &mut [Actor],
        context: &mut Context,
        source: usize,
        target: usize,
        prefab: &Prefab,
    ) -> scheduler::Result<()> {
        let old = actors[source]
            .slime
            .as_ref()
            .ok_or("null_reference")?
            .abilities
            .as_ref()
            .ok_or("missing_ability_routing")?
            .clone();
        let mut slots = vec![];
        for i in 0..old.slots.len() {
            let indicator = old
                .indicators
                .get(i)
                .ok_or("index_out_of_range")?
                .as_ref()
                .ok_or("null_reference")?;
            let icon = indicator.primary_icon.ok_or("null_reference")?;
            let mut ability = prefab
                .abilities
                .as_ref()
                .ok_or("actor_clone_abilities_not_implemented")?
                .get(icon)
                .ok_or("argument_out_of_range")?
                .as_ref()
                .ok_or("actor_clone_ability_prefab_not_implemented")?
                .clone();
            if !matches!(
                ability.handler,
                Some(Handler::Invisibility(_)) | Some(Handler::Instant(_))
            ) {
                return Err("actor_clone_ability_prefab_not_implemented");
            }
            let instant = ability.handler.as_mut().unwrap().instant_mut().unwrap();
            let has_transform = instant.transform_present;
            let active = instant.active;
            let node = context.schedule.nodes.len();
            context.schedule.nodes.push(scheduler::Node {
                parent: None,
                children: vec![],
                active,
                present: true,
            });
            let component = context.schedule.components.len();
            context
                .components
                .resize(component + 1 + usize::from(has_transform), None);
            context.schedule.begin_prefab();
            for i in 0..1 + usize::from(has_transform) {
                context.schedule.components.push(scheduler::Component {
                    owner: node,
                    kind: if i == 0 {
                        scheduler::Kind::Updatable
                    } else {
                        scheduler::Kind::Fix
                    },
                    present: true,
                    destroyed: false,
                    hierarchy: if i == 0 { 1 } else { 0 },
                    enabled: true,
                    ignore_activity: false,
                    execution_gate: None,
                });
            }
            context.schedule.register(Some(component));
            context.schedule.end_prefab()?;
            ability.registration = Some(Registration {
                node,
                component,
                transform: has_transform.then_some(component + 1),
            });
            let routing = actors[target]
                .slime
                .as_mut()
                .unwrap()
                .abilities
                .as_mut()
                .unwrap();
            let index = routing.pool.len();
            routing.pool.push(ability);
            // Fixed instantiation registers the real object before dereferencing
            // its transform. NullAbility's shipped prefab has no fixed transform.
            if !has_transform {
                return Err("null_reference");
            }
            context.schedule.nodes[node].active = false;
            routing.pool[index]
                .handler
                .as_mut()
                .unwrap()
                .instant_mut()
                .unwrap()
                .active = false;
            slots.push(Some(index));
        }
        actors[target]
            .slime
            .as_mut()
            .unwrap()
            .abilities
            .as_mut()
            .unwrap()
            .slots = slots;
        let mut indicators = vec![None; 3];
        for (i, indicator) in old.indicators.iter().enumerate() {
            if let Some(indicator) = indicator {
                let slot = indicators.get_mut(i).ok_or("index_out_of_range")?;
                *slot = Some(Indicator {
                    seconds_since_reset: 11.0,
                    primary_icon: indicator.primary_icon,
                    lifecycle: Some(IndicatorLifecycle {
                        active: false,
                        follow: true,
                    }),
                });
            }
        }
        actors[target]
            .slime
            .as_mut()
            .unwrap()
            .abilities
            .as_mut()
            .unwrap()
            .indicators = indicators;
        Ok(())
    }
    pub fn destroy_pooled_or_actor_abilities(
        actors: &mut [Actor],
        context: &mut Context,
        source: usize,
        include_destroyed: bool,
    ) -> scheduler::Result<()> {
        let routing = actors[source]
            .slime
            .as_ref()
            .unwrap()
            .abilities
            .as_ref()
            .ok_or("missing_ability_routing")?;
        let slots = routing.slots.clone();
        for index in slots.into_iter().flatten() {
            let ability = &actors[source]
                .slime
                .as_ref()
                .unwrap()
                .abilities
                .as_ref()
                .unwrap()
                .pool[index];
            if !ability.present || (!include_destroyed && ability.destroyed) {
                continue;
            }
            if let Some(r) = &ability.registration {
                context.schedule.destroy_object(Some(r.node))?;
                Self::refresh_pooled_abilities(actors, context.schedule);
            } else if let Some(Handler::Entity { actor }) = ability.handler {
                Self::destroy_actor(actors, context, actor)?;
                Self::refresh_entities(context.schedule, context.components, actors, context.scene);
            } else {
                return Err("actor_pooled_ability_lifecycle_not_implemented");
            }
        }
        Ok(())
    }
    pub fn refresh_pooled_abilities(actors: &mut [Actor], schedule: &scheduler::State) {
        for actor in actors {
            if let Some(routing) = actor.slime.as_mut().and_then(|s| s.abilities.as_mut()) {
                for ability in &mut routing.pool {
                    if let Some(registration) = &ability.registration {
                        ability.present = schedule.components[registration.component].present;
                        ability.destroyed = schedule.components[registration.component].destroyed;
                    }
                }
            }
        }
    }
    pub fn pooled_ability_phase(
        actors: &mut [Actor],
        schedule: &scheduler::State,
        phase: scheduler::Event,
    ) -> scheduler::Result<bool> {
        Self::refresh_pooled_abilities(actors, schedule);
        if matches!(phase.kind, scheduler::Phase::DestroyObject) {
            return Ok(false);
        }
        let Some(id) = phase.id else { return Ok(false) };
        Ok(actors
            .iter()
            .filter_map(|a| a.slime.as_ref()?.abilities.as_ref())
            .flat_map(|r| &r.pool)
            .any(|a| {
                a.registration
                    .as_ref()
                    .is_some_and(|r| r.component == id || r.transform == Some(id))
            }))
    }
}
