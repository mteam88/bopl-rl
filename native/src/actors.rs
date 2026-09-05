//! Composed player-controller/body simulation driven by Updater's scheduler.
//! Motors use scripted movement or an installed SlimeController instance.
//! Actor transforms alias scene objects; controller and body projections are
//! reconciled after every call, including a call that fails partway through.
use crate::{
    fixed::Fix,
    physics::scene::{Scene, StepReport},
    platform_motion::Binding,
    player_body, player_controller, scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Actor {
    pub object: u32,
    pub controller: player_controller::State,
    pub body: player_body::State,
    pub config: player_controller::Config,
    pub input: Vec2,
    pub push_strength: Option<Fix>,
    pub collision_hierarchy: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slime: Option<crate::slime::Instance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<crate::actor_entity::Entity>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Physics,
    Motor,
    Animation,
    Body,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Component {
    pub actor: usize,
    pub role: Role,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Runtime {
    pub scheduler: scheduler::State,
    pub components: Vec<Component>,
    pub actors: Vec<Actor>,
    pub scene: Scene,
    pub bindings: Vec<Binding>,
    pub grounds: Vec<player_controller::Ground>,
    pub space: bool,
    pub player_layer: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub actor: usize,
    pub event: player_controller::Event,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Push {
    pub actor: usize,
    pub external_delta: Vec2,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub phases: Vec<scheduler::Event>,
    pub events: Vec<Event>,
    pub physics: Option<StepReport>,
    pub pushes: Vec<Push>,
}
impl Actor {
    fn refresh_callback_gates(&mut self, scene: &Scene) {
        if let Some(slime) = &mut self.slime {
            slime.callback_gates = (
                self.entity.as_ref().is_none_or(|e| e.body_simulated),
                scene
                    .objects
                    .iter()
                    .find(|o| o.id == self.object)
                    .expect("actor scene object")
                    .enabled,
            );
        }
    }
    pub fn apply_ragdoll(
        &mut self,
        request: &crate::slime::RagdollImpulse,
        scene: &mut Scene,
    ) -> scheduler::Result<()> {
        self.refresh_callback_gates(scene);
        let slime = self.slime.as_mut().ok_or("missing_slime_controller")?;
        let result = slime.apply_ragdoll(&mut self.controller, request);
        slime.sync_callback();
        self.controller_to_body();
        self.sync_transform(scene);
        result
    }

    /// The equal-scale shove branch of SlimeController.OnCollide. Bite and
    /// other collision effects must be supplied by the full gameplay controller.
    fn shove(&mut self, other_position: Vec2, other_hierarchy: i32) -> Option<Vec2> {
        let strength = self.push_strength?;
        let m = &self.controller.physics.motion;
        let direction = crate::player_collision::overlap_push_direction(
            m.position,
            m.up,
            m.radius,
            other_position,
            self.collision_hierarchy.wrapping_sub(other_hierarchy),
        )?;
        let before = m.external_velocity;
        self.controller
            .physics
            .add_force(direction * strength, &self.config.physics);
        let delta = self.controller.physics.motion.external_velocity - before;
        self.controller_to_body();
        (delta != Vec2::ZERO).then_some(delta)
    }
    fn controller_to_body(&mut self) {
        let m = &self.controller.physics.motion;
        self.body.position = m.position;
        self.body.velocity = m.velocity;
        self.body.external_velocity = m.external_velocity;
        self.body.physics = Some(player_body::Physics {
            grounded: m.grounded,
            jumped: self.controller.physics.jumped,
        });
    }
    fn body_to_controller(&mut self) {
        let m = &mut self.controller.physics.motion;
        m.position = self.body.position;
        m.velocity = self.body.velocity;
        m.external_velocity = self.body.external_velocity;
    }
    fn sync_transform(&self, scene: &mut Scene) {
        let t = &mut scene
            .objects
            .iter_mut()
            .find(|o| o.id == self.object)
            .expect("actor scene object")
            .transform;
        let m = &self.controller.physics.motion;
        t.position = m.position;
        t.rotation = m.rotation;
        // Preserve optional serialized representation; explicit cached right is
        // required in a general world. Inputs in this runner carry that cache.
        t.right = Some(m.right);
    }
}
impl Runtime {
    pub fn refresh_entities(
        schedule: &mut scheduler::State,
        components: &[Component],
        actors: &[Actor],
        scene: &mut Scene,
    ) {
        for actor in actors {
            if let Some(entity) = &actor.entity {
                schedule.nodes[entity.node].active = entity.active;
                scene
                    .objects
                    .iter_mut()
                    .find(|o| o.id == actor.object)
                    .expect("actor scene object")
                    .active = entity.active;
            }
        }
        for (i, component) in components.iter().enumerate() {
            if let Some(entity) = &actors[component.actor].entity {
                schedule.components[i].execution_gate = Some(
                    entity.active
                        && (!matches!(component.role, Role::Body) || entity.body_simulated),
                );
            }
        }
    }
    pub fn position_priorities(actors: &[Actor]) -> Vec<Option<i32>> {
        actors
            .iter()
            .map(|a| {
                a.slime.as_ref().and_then(|s| {
                    (s.managed_destroyed != Some(true))
                        .then(|| s.abilities.as_ref().map_or(0, |r| r.slots.len() as i32))
                })
            })
            .collect()
    }
    pub fn synchronize(&mut self) {
        for actor in &mut self.actors {
            actor.controller_to_body();
            actor.sync_transform(&mut self.scene);
        }
    }
    pub fn tick(&mut self, dt: Fix, report: &mut Report) -> scheduler::Result<()> {
        self.tick_with(dt, report, &mut |_, _, _, _| Ok(()))
    }
    /// Dispatch external gameplay effects at exact component/physics phases.
    pub fn tick_with(
        &mut self,
        dt: Fix,
        report: &mut Report,
        before: &mut impl FnMut(
            &mut scheduler::State,
            &mut [Actor],
            &mut Scene,
            scheduler::Event,
        ) -> scheduler::Result<()>,
    ) -> scheduler::Result<()> {
        Self::refresh_entities(
            &mut self.scheduler,
            &self.components,
            &self.actors,
            &mut self.scene,
        );
        if self.scene.rope_system.is_some() || self.actors.iter().any(|a| a.body.rope.is_some()) {
            return Err("actor_rope_integration_not_implemented");
        }
        if self.actors.iter().any(|a| a.push_strength.is_some()) {
            let scale = self.actors[0].controller.physics.scale;
            if self
                .actors
                .iter()
                .any(|a| a.controller.physics.scale != scale)
            {
                return Err("actor_bite_integration_not_implemented");
            }
        }
        if self
            .scene
            .objects
            .iter()
            .any(|o| o.layer == self.player_layer && !self.actors.iter().any(|a| a.object == o.id))
        {
            return Err("actor_player_collider_not_bound");
        }
        let Self {
            scheduler,
            components,
            actors,
            scene,
            bindings,
            grounds,
            space,
            player_layer,
        } = self;
        scheduler.tick(dt, &mut |schedule, phase| {
            report.phases.push(phase);
            before(schedule, actors, scene, phase)?;
            Self::refresh_entities(schedule, components, actors, scene);
            match phase.kind {
                scheduler::Phase::Physics => {
                    report.physics =
                        Some(scene.step(if schedule.time_scale < Fix::from_f32(0.01) {
                            Fix::ZERO
                        } else {
                            dt
                        }));
                    for a in actors.iter_mut() {
                        let transform = scene
                            .objects
                            .iter()
                            .find(|o| o.id == a.object)
                            .expect("actor scene object")
                            .transform;
                        a.controller.physics.motion.position = transform.position;
                        a.controller.physics.motion.rotation = transform.rotation;
                        a.controller.physics.motion.right =
                            transform.right.expect("actor cached direction");
                        a.controller.physics.motion.up = a.controller.physics.motion.right.perp();
                        a.controller_to_body();
                    }
                    // DetPhysics delivers queued callbacks only after synchronizing
                    // collider transforms. The equal-size shove branch does not
                    // alter colliders until the following physics step.
                    for hit in &report.physics.as_ref().unwrap().callbacks {
                        if hit.layer != *player_layer {
                            continue;
                        }
                        let Some(index) = actors.iter().position(|a| a.object == hit.receiver)
                        else {
                            continue;
                        };
                        let other = scene
                            .objects
                            .iter()
                            .find(|o| o.id == hit.other)
                            .expect("collision object");
                        // Player-layer colliders in this integration stage all
                        // belong to actors, which retain their collision priority.
                        let other_hierarchy = actors
                            .iter()
                            .find(|a| a.object == hit.other)
                            .expect("player actor collision")
                            .collision_hierarchy;
                        if let Some(external_delta) =
                            actors[index].shove(other.transform.position, other_hierarchy)
                        {
                            report.pushes.push(Push {
                                actor: index,
                                external_delta,
                            });
                        }
                    }
                }
                scheduler::Phase::Init | scheduler::Phase::Update | scheduler::Phase::Late => {
                    let component = components[phase.id.unwrap()];
                    let position_priorities = Self::position_priorities(actors);
                    let actor = &mut actors[component.actor];
                    actor.refresh_callback_gates(scene);
                    let protected = actor
                        .body
                        .holder
                        .and_then(|id| schedule.players.iter().find(|p| p.id == id))
                        .map(|p| p.protected);
                    let mut emit = |event| {
                        report.events.push(Event {
                            actor: component.actor,
                            event,
                        });
                    };
                    let mut context = player_controller::Context {
                        scene,
                        bindings,
                        grounds,
                        config: &actor.config,
                        holder: actor.body.holder,
                        protected,
                        time_scale: schedule.time_scale,
                        space: *space,
                        ability: false,
                    };
                    let platforms: Vec<_> = bindings
                        .iter()
                        .map(|b| b.platform.clone().expect("actor platform binding"))
                        .collect();
                    let body_context = player_body::Context {
                        players: schedule
                            .players
                            .iter()
                            .map(|p| player_body::Player {
                                id: p.id,
                                protected: p.protected,
                                jump: p.input.is_some_and(|i| schedule.inputs[i].pressed),
                            })
                            .collect(),
                        time_scale: schedule.time_scale,
                        rope_gravity: Fix::ZERO,
                        player_gravity: Fix::ZERO,
                        max_stretch: Fix::ZERO,
                    };
                    let result = match (phase.kind, component.role) {
                        (scheduler::Phase::Init, Role::Body) => {
                            actor.body.init();
                            Ok(())
                        }
                        (scheduler::Phase::Update, Role::Physics) => {
                            actor
                                .controller
                                .update(dt, &mut context, &mut |controller, event| {
                                    if let Some(slime) = &mut actor.slime {
                                        slime.physics_event(
                                            controller,
                                            &event,
                                            bindings,
                                            actor.body.holder.ok_or("null_reference")?,
                                            &mut emit,
                                        )?;
                                    }
                                    emit(event);
                                    Ok(())
                                })
                        }
                        (scheduler::Phase::Update, Role::Motor) => {
                            let jump = actor
                                .body
                                .holder
                                .and_then(|id| schedule.players.iter().find(|p| p.id == id))
                                .and_then(|p| p.input)
                                .is_some_and(|i| schedule.inputs[i].pressed);
                            if let Some(slime) = &mut actor.slime {
                                slime.update(
                                    &mut actor.controller,
                                    crate::slime::Update {
                                        dt,
                                        input: actor.input,
                                        jump,
                                        actor: component.actor,
                                        player: schedule
                                            .players
                                            .iter_mut()
                                            .find(|p| Some(p.id) == actor.body.holder)
                                            .ok_or("null_reference")?,
                                        position_priorities: &position_priorities,
                                        entity: actor.entity.as_mut(),
                                        object: actor.object,
                                        body: &mut actor.body,
                                    },
                                    &mut context,
                                    &mut emit,
                                )
                            } else {
                                let mut notify = |_: &mut player_controller::State, event| {
                                    emit(event);
                                    Ok(())
                                };
                                let jump_result =
                                    if jump && actor.controller.physics.motion.grounded {
                                        actor
                                            .controller
                                            .physics
                                            .jump(&platforms, &actor.config.physics)
                                            .and_then(|angle| {
                                                notify(
                                                    &mut actor.controller,
                                                    player_controller::Event::Detached(angle),
                                                )?;
                                                if let Some(id) = actor.body.holder {
                                                    notify(
                                                        &mut actor.controller,
                                                        player_controller::Event::JumpSound(
                                                            format!("slimeJump{id}"),
                                                        ),
                                                    )?;
                                                }
                                                Ok(())
                                            })
                                    } else {
                                        Ok(())
                                    };
                                jump_result.and_then(|()| {
                                    actor.controller.move_player(
                                        actor.input,
                                        dt,
                                        None,
                                        true,
                                        &mut context,
                                        &mut notify,
                                    )
                                })
                            }
                        }
                        (scheduler::Phase::Update, Role::Animation) => actor
                            .slime
                            .as_mut()
                            .ok_or("missing_slime_animator")?
                            .animate(
                                Some(schedule.time_scale),
                                protected,
                                actor.body.holder.ok_or("null_reference")?,
                                &mut emit,
                            ),
                        (scheduler::Phase::Update, Role::Body) => {
                            let result = actor.body.update(dt, &mut [], &body_context).map(|_| ());
                            actor.body_to_controller();
                            result
                        }
                        (scheduler::Phase::Late, Role::Physics) => actor
                            .controller
                            .physics
                            .late_update(&platforms, None, &actor.config.physics, &mut |_, _| {}),
                        (scheduler::Phase::Late, Role::Body) => {
                            actor.body.late_update(&mut [], &body_context)
                        }
                        _ => Ok(()),
                    };
                    if let Some(slime) = &mut actor.slime {
                        slime.sync_callback();
                    }
                    actor.controller_to_body();
                    actor.sync_transform(scene);
                    result?;
                }
                _ => {}
            }
            Ok(())
        })
    }
}
