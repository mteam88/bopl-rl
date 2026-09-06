//! Original ReviveParticle travel, duplicate registration and arrival callback.
use crate::{
    actors::{Actor, Context, Event, Report, Runtime},
    animator,
    fixed::Fix,
    scheduler,
    spell_world::{Instance, Prefab, State},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub speed: Fix,
    pub spawn: Vec<animator::Frame>,
    pub initial_sprite: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Particle {
    pub target: Option<usize>,
    pub slime: Option<usize>,
    pub clone_id: i32,
    pub distance: Fix,
    pub travelled: Fix,
    pub arrived: bool,
    pub speed: Fix,
    pub right: Vec2,
    pub animator: animator::State,
}
impl crate::spell_world::World {
    pub fn marker_position(
        &self,
        index: Option<usize>,
        _schedule: &scheduler::State,
    ) -> scheduler::Result<Vec2> {
        let marker = index
            .and_then(|i| self.instances.get(i))
            .ok_or("null_reference")?;
        if !matches!(marker.state, State::Marker(_)) {
            return Err("marker_required");
        }
        // GetPos reads a managed FixTransform field even after Unity removes it.
        Ok(marker.position)
    }
    pub fn spawn_particle(
        &mut self,
        schedule: &mut scheduler::State,
        prefab: &str,
        actor: usize,
        position: Vec2,
    ) -> scheduler::Result<usize> {
        let Prefab::ReviveParticle(config) =
            self.prefabs.get(prefab).ok_or("spell_prefab_not_bound")?
        else {
            return Err("particle_required");
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
        for i in [0, 1] {
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
            state: State::ReviveParticle(Box::new(Particle {
                target: None,
                slime: None,
                clone_id: 0,
                distance: Fix::ZERO,
                travelled: Fix::ZERO,
                arrived: false,
                speed: config.speed,
                right: Vec2::new(Fix::ONE, Fix::ZERO),
                animator: animator::State::idle(config.initial_sprite),
            })),
        });
        Ok(index)
    }
    pub fn init_particle(
        &mut self,
        schedule: &mut scheduler::State,
        index: usize,
        target: Option<usize>,
        slime: Option<usize>,
        clone_id: i32,
    ) -> scheduler::Result<()> {
        let instance = &mut self.instances[index];
        let State::ReviveParticle(p) = &mut instance.state else {
            return Err("particle_required");
        };
        p.target = target;
        p.slime = slime;
        p.clone_id = clone_id;
        p.travelled = Fix::ZERO;
        p.arrived = false;
        let position = instance.position;
        let distance = (self.marker_position(target, schedule)? - position).length();
        let instance = &mut self.instances[index];
        let State::ReviveParticle(p) = &mut instance.state else {
            unreachable!()
        };
        p.distance = distance;
        schedule.register(Some(instance.components[0]));
        Ok(())
    }
}
impl Runtime {
    pub fn particle_phase(
        actors: &mut [Actor],
        context: &mut Context,
        phase: scheduler::Event,
        dt: Fix,
        report: &mut Report,
    ) -> scheduler::Result<bool> {
        let Some(id) = phase.id else { return Ok(false) };
        if matches!(phase.kind, scheduler::Phase::DestroyObject) {
            return Ok(false);
        }
        let Some(world) = context.spells.as_mut() else {
            return Ok(false);
        };
        world.sync_positions(context.scene);
        let Some(index) = world.instances.iter().position(|e| {
            matches!(e.state, State::ReviveParticle(_)) && e.components.contains(&id)
        }) else {
            return Ok(false);
        };
        let instance = &mut world.instances[index];
        let State::ReviveParticle(p) = &mut instance.state else {
            unreachable!()
        };
        if !matches!(phase.kind, scheduler::Phase::Update) {
            return Ok(true);
        }
        let slime = p.slime.filter(|&i| actors[i].is_present());
        let owner = slime.and_then(|i| actors[i].body.holder).unwrap_or(1);
        let protected = context
            .schedule
            .players
            .iter()
            .find(|p| p.id == owner)
            .map(|p| p.protected);
        if id == instance.components[1] {
            let mut destroy = false;
            p.animator.update(
                context.schedule.time_scale,
                protected,
                &[],
                &mut |_, callback| {
                    if callback != 1 {
                        return Err("particle_callback_invalid");
                    };
                    destroy = true;
                    Ok(())
                },
            )?;
            if destroy {
                context.schedule.destroy_object(Some(instance.node))?;
                instance.destroyed = true;
            }
            return Ok(true);
        }
        if id != instance.components[0] || p.travelled > p.distance {
            return Ok(true);
        }
        let target = p.target;
        let position = instance.position;
        let step = p.speed
            * if slime.is_some() {
                crate::game_time::fixed_delta_time(dt, context.schedule.time_scale, protected)
            } else {
                dt
            };
        let target_position = world.marker_position(target, context.schedule)?;
        let delta = (target_position - position).normalized_safe() * step;
        let instance = &mut world.instances[index];
        let State::ReviveParticle(p) = &mut instance.state else {
            unreachable!()
        };
        p.travelled += step.abs();
        if p.travelled < p.distance || p.arrived {
            instance.position += delta;
            return Ok(true);
        }
        world.end_marker(
            target.ok_or("null_reference")?,
            context.schedule,
            context.scene,
        )?;
        let instance = &mut world.instances[index];
        let State::ReviveParticle(p) = &mut instance.state else {
            unreachable!()
        };
        instance.position = target_position;
        p.arrived = true;
        p.speed = Fix::ZERO;
        report.events.push(Event {
            actor: instance.actor,
            event: crate::player_controller::Event::Sound("reviveDing".into()),
        });
        let Prefab::ReviveParticle(config) = &world.prefabs[&instance.prefab] else {
            unreachable!()
        };
        p.animator.start(
            animator::Start::Begin {
                frames: Some(config.spawn.clone()),
                next: None,
                callback: Some(1),
                style: 3,
            },
            &[],
            &mut |_, _| Ok(()),
        )?;
        if let Some(slime) = slime {
            Self::spawn_actor(
                actors,
                context,
                slime,
                crate::actor_spawn::Action::Respawn,
                target_position,
                report,
            )?;
        }
        Ok(true)
    }
}
