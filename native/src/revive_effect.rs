//! ReviveEffect's wisp and shoot animations allocate original revival particles.
use crate::{
    animator,
    fixed::Fix,
    scheduler,
    spell_world::{Instance, Prefab, State, World},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub wisp: Vec<animator::Frame>,
    pub shoot: Vec<animator::Frame>,
    pub initial_sprite: Option<u32>,
    pub particle: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    pub slime: Option<usize>,
    pub marker: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    pub player: Option<i32>,
    pub targets: Option<Vec<Target>>,
    pub right: Vec2,
    pub animator: animator::State,
}
impl World {
    pub fn spawn_revive_effect(
        &mut self,
        schedule: &mut scheduler::State,
        prefab: &str,
        actor: usize,
        position: Vec2,
    ) -> scheduler::Result<usize> {
        let Prefab::ReviveEffect(config) =
            self.prefabs.get(prefab).ok_or("spell_prefab_not_bound")?
        else {
            return Err("revive_effect_required");
        };
        let node = schedule.nodes.len();
        schedule.nodes.push(scheduler::Node {
            parent: None,
            children: vec![],
            active: true,
            present: true,
        });
        let first = schedule.components.len();
        let components = vec![first, first + 1, first + 2];
        schedule.begin_prefab();
        for i in 0..3 {
            schedule.components.push(scheduler::Component {
                owner: node,
                kind: if i == 2 {
                    scheduler::Kind::Fix
                } else {
                    scheduler::Kind::Updatable
                },
                present: true,
                destroyed: false,
                hierarchy: if i == 2 { 0 } else { i + 1 },
                enabled: true,
                ignore_activity: false,
                execution_gate: None,
            });
        }
        for i in [1, 0] {
            schedule.register(Some(components[i]));
        }
        schedule.end_prefab()?;
        let index = self.instances.len();
        self.instances.push(Instance {
            prefab: prefab.into(),
            actor,
            node,
            components,
            position,
            present: true,
            destroyed: false,
            state: State::ReviveEffect(Box::new(Effect {
                player: None,
                targets: None,
                right: Vec2::new(Fix::ONE, Fix::ZERO),
                animator: animator::State::idle(config.initial_sprite),
            })),
        });
        Ok(index)
    }
    pub fn init_revive_effect(
        &mut self,
        index: usize,
        player: Option<i32>,
        targets: Option<Vec<Target>>,
    ) -> scheduler::Result<()> {
        let instance = &mut self.instances[index];
        let State::ReviveEffect(e) = &mut instance.state else {
            return Err("revive_effect_required");
        };
        e.player = player;
        e.targets = targets;
        let Prefab::ReviveEffect(config) = &self.prefabs[&instance.prefab] else {
            unreachable!()
        };
        e.animator.start(
            animator::Start::Begin {
                frames: Some(config.wisp.clone()),
                next: None,
                callback: Some(1),
                style: 3,
            },
            &[],
            &mut |_, _| Ok(()),
        )
    }
    pub fn revive_effect_phase(
        &mut self,
        schedule: &mut scheduler::State,
        phase: scheduler::Event,
    ) -> scheduler::Result<bool> {
        let Some(id) = phase.id else { return Ok(false) };
        if matches!(phase.kind, scheduler::Phase::DestroyObject) {
            return Ok(false);
        }
        let Some(index) = self
            .instances
            .iter()
            .position(|e| matches!(e.state, State::ReviveEffect(_)) && e.components.contains(&id))
        else {
            return Ok(false);
        };
        let instance = &self.instances[index];
        if !matches!(phase.kind, scheduler::Phase::Update) || id != instance.components[0] {
            return Ok(true);
        }
        let State::ReviveEffect(e) = &instance.state else {
            unreachable!()
        };
        let protected = if schedule.time_scale < Fix::from_f32(0.01) {
            let player = e.player.ok_or("null_reference")?;
            schedule
                .players
                .iter()
                .find(|p| p.id == player)
                .map(|p| p.protected)
        } else {
            None
        };
        // A callback can grow instances. Keep the current animator separate until
        // the callback returns, then preserve its state even on a partial failure.
        let mut animator = e.animator.clone();
        let result = animator.update(
            schedule.time_scale,
            protected,
            &[],
            &mut |animator, callback| {
                if callback == 1 {
                    let instance = &self.instances[index];
                    let State::ReviveEffect(e) = &instance.state else {
                        unreachable!()
                    };
                    if let Some(targets) = e.targets.clone().filter(|t| !t.is_empty()) {
                        let position = instance.position;
                        let actor = instance.actor;
                        let Prefab::ReviveEffect(config) = &self.prefabs[&instance.prefab] else {
                            unreachable!()
                        };
                        let particle = config.particle.clone();
                        let shoot = config.shoot.clone();
                        for (i, target) in targets.iter().enumerate() {
                            let direction = (self.marker_position(target.marker, schedule)?
                                - position)
                                .normalized_safe();
                            let Some(prefab) = particle.as_deref() else {
                                schedule.begin_prefab();
                                return Err("instantiate_null");
                            };
                            let particle =
                                self.spawn_particle(schedule, prefab, actor, position)?;
                            if direction != Vec2::ZERO {
                                let State::ReviveParticle(p) = &mut self.instances[particle].state
                                else {
                                    unreachable!()
                                };
                                p.right = direction;
                                let State::ReviveEffect(e) = &mut self.instances[index].state
                                else {
                                    unreachable!()
                                };
                                e.right = direction;
                            }
                            self.init_particle(
                                schedule,
                                particle,
                                target.marker,
                                target.slime,
                                i as i32,
                            )?;
                        }
                        return animator.start(
                            animator::Start::Begin {
                                frames: Some(shoot),
                                next: None,
                                callback: Some(2),
                                style: 3,
                            },
                            &[],
                            &mut |_, _| Ok(()),
                        );
                    }
                } else if callback != 2 {
                    return Err("revive_effect_callback_invalid");
                }
                let instance = &mut self.instances[index];
                schedule.destroy_object(Some(instance.node))?;
                instance.destroyed = true;
                Ok(())
            },
        );
        let State::ReviveEffect(e) = &mut self.instances[index].state else {
            unreachable!()
        };
        e.animator = animator;
        result.map(|()| true)
    }
}

impl crate::actors::Runtime {
    pub fn revive_actor(
        actors: &mut Vec<crate::actors::Actor>,
        context: &mut crate::actors::Context,
        id: usize,
        victim: usize,
        request: &crate::actor_combat::Kill,
        report: &mut crate::actors::Report,
    ) -> scheduler::Result<()> {
        Self::spawn_actor(
            actors,
            context,
            id,
            crate::actor_spawn::Action::Unground,
            Vec2::ZERO,
            report,
        )?;
        let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
        combat.revival[victim].status = 2;
        actors[id].slime.as_ref().ok_or("null_reference")?;
        let positions = combat.revival[victim]
            .positions
            .clone()
            .ok_or("missing_revival_positions")?;
        let prefab = combat.clone_prefab.clone();
        let effect = combat.revive_effect.clone();
        let player = combat.book.players[victim].id;
        let mut targets = Vec::with_capacity(positions.len());
        for marker in positions {
            let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
            world.sync_positions(context.scene);
            let position = world.marker_position(marker, context.schedule)?;
            let slime = Self::spawn_clone(
                actors,
                context,
                id,
                victim,
                position,
                prefab.as_deref(),
                report,
            )?;
            targets.push(Target { slime, marker });
        }
        let combat = context.combat.as_mut().unwrap();
        if let Some(event) = combat.book.kill(
            victim,
            request.killer,
            context.schedule.ticks as i64,
            request.cause,
        ) {
            combat.events.push(event);
        }
        combat.revival[victim].positions.as_mut().unwrap().clear();
        combat.revival[victim].respawn_count = 0;
        let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
        let Some(effect) = effect.as_deref() else {
            context.schedule.begin_prefab();
            return Err("instantiate_null");
        };
        let index =
            world.spawn_revive_effect(context.schedule, effect, id, actors[id].body.position)?;
        world.init_revive_effect(index, Some(player), Some(targets))?;
        let routing = actors[id]
            .slime
            .as_mut()
            .unwrap()
            .abilities
            .as_mut()
            .ok_or("missing_ability_routing")?;
        let mut abilities = vec![];
        for slot in routing.slots.iter().flatten() {
            let ability = &routing.pool[*slot];
            if ability.present {
                let Some(crate::ability::Handler::Entity { actor }) = ability.handler else {
                    return Err("actor_pooled_ability_lifecycle_not_implemented");
                };
                abilities.push(actor);
            }
        }
        for ability in abilities {
            Self::destroy_actor(actors, context, ability)?;
        }
        actors[id]
            .slime
            .as_mut()
            .unwrap()
            .abilities
            .as_mut()
            .unwrap()
            .slots
            .clear();
        Self::destroy_actor(actors, context, id)?;
        report.events.push(crate::actors::Event {
            actor: id,
            event: crate::player_controller::Event::Sound("reviveOnDeath".into()),
        });
        Ok(())
    }
}
