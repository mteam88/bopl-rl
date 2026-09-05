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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinary: Option<crate::ordinary_ability::Instance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision: Option<crate::actor_combat::Collision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rewind: Option<crate::rewind::State>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Physics,
    Motor,
    Animation,
    Body,
    Ability,
    Dash,
    Collision,
    Rewind,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Component {
    pub actor: usize,
    pub role: Role,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorRopes {
    pub metadata: Vec<player_body::RopeMetadata>,
    pub player_gravity: Fix,
    pub max_stretch: Fix,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Runtime {
    pub scheduler: scheduler::State,
    pub components: Vec<Component>,
    pub actors: Vec<Actor>,
    pub scene: Scene,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_ropes: Option<ActorRopes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combat: Option<crate::actor_combat::Combat>,
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
pub struct Context<'a> {
    pub schedule: &'a mut scheduler::State,
    pub scene: &'a mut Scene,
    pub actor_ropes: &'a mut Option<ActorRopes>,
    pub combat: &'a mut Option<crate::actor_combat::Combat>,
    pub bindings: &'a [Binding],
    pub grounds: &'a [player_controller::Ground],
    pub space: bool,
}
impl Context<'_> {
    pub(crate) fn attach_rope(
        &mut self,
        body: &mut player_body::State,
        next: Option<usize>,
        top: bool,
    ) -> scheduler::Result<()> {
        let system = self
            .scene
            .rope_system
            .as_mut()
            .ok_or("actor_rope_binding_invalid")?;
        let metadata = &mut self
            .actor_ropes
            .as_mut()
            .ok_or("actor_rope_binding_invalid")?
            .metadata;
        if metadata.len() != system.bodies.len()
            || next
                .into_iter()
                .chain(body.rope)
                .any(|i| i >= metadata.len())
        {
            return Err("actor_rope_binding_invalid");
        }
        body.attach_in(
            next,
            top,
            &mut player_body::JoinedRopes {
                physics: &mut system.bodies,
                metadata,
            },
            &player_body::Context {
                players: vec![],
                time_scale: self.schedule.time_scale,
                rope_gravity: system.gravity,
                player_gravity: Fix::ZERO,
                max_stretch: Fix::ZERO,
            },
        );
        Ok(())
    }
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
    pub(crate) fn controller_to_body(&mut self) {
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
    pub(crate) fn sync_transform(&self, scene: &mut Scene) {
        // A ray query can prune a DestroyFix-marked collider before Unity
        // deletes the GameObject. The current component call still finishes
        // against its live body/transform, retained on the actor.
        let Some(object) = scene.objects.iter_mut().find(|o| o.id == self.object) else {
            assert!(
                self.entity
                    .as_ref()
                    .is_some_and(|e| e.destroyed == Some(true)),
                "actor scene object"
            );
            return;
        };
        let t = &mut object.transform;
        let m = &self.controller.physics.motion;
        t.position = m.position;
        t.rotation = m.rotation;
        // Preserve optional serialized representation; explicit cached right is
        // required in a general world. Inputs in this runner carry that cache.
        t.right = Some(m.right);
    }
}
impl Runtime {
    fn finish_slime(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        dt: Fix,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let priorities = Self::position_priorities(actors);
        let actor = &mut actors[id];
        actor.refresh_callback_gates(context.scene);
        let holder = actor.body.holder;
        let player = context
            .schedule
            .players
            .iter_mut()
            .find(|p| Some(p.id) == holder)
            .ok_or("null_reference")?;
        let protected = Some(player.protected);
        let mut update = crate::slime::Update {
            dt,
            input: actor.input,
            jump: false,
            actor: id,
            body: &mut actor.body,
            player,
            position_priorities: &priorities,
            entity: actor.entity.as_mut(),
            object: actor.object,
        };
        let mut controller_context = player_controller::Context {
            scene: context.scene,
            bindings: context.bindings,
            grounds: context.grounds,
            config: &actor.config,
            holder,
            protected,
            time_scale: context.schedule.time_scale,
            space: context.space,
            ability: false,
        };
        let result = actor
            .slime
            .as_mut()
            .ok_or("missing_slime_controller")?
            .finish_update(
                &mut actor.controller,
                &mut update,
                &mut controller_context,
                &mut |event| report.events.push(Event { actor: id, event }),
            );
        if let Some(slime) = &mut actor.slime {
            slime.sync_callback();
        }
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        result
    }
    pub fn exit_actor(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        info: &crate::ability::ExitInfo,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let priorities = Self::position_priorities(actors);
        let actor = &mut actors[id];
        actor.refresh_callback_gates(context.scene);
        let entity = actor.entity.as_mut().ok_or("missing_actor_entity")?;
        entity.active = true;
        context.schedule.nodes[entity.node].active = true;
        context
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == actor.object)
            .ok_or("missing_actor_object")?
            .active = true;
        let holder = actor.body.holder;
        let player = context
            .schedule
            .players
            .iter_mut()
            .find(|p| Some(p.id) == holder)
            .ok_or("null_reference")?;
        let protected = Some(player.protected);
        let mut update = crate::slime::Update {
            dt: Fix::ZERO,
            input: Vec2::ZERO,
            jump: false,
            actor: id,
            body: &mut actor.body,
            player,
            position_priorities: &priorities,
            entity: actor.entity.as_mut(),
            object: actor.object,
        };
        let mut controller_context = player_controller::Context {
            scene: context.scene,
            bindings: context.bindings,
            grounds: context.grounds,
            config: &actor.config,
            holder,
            protected,
            time_scale: context.schedule.time_scale,
            space: context.space,
            ability: false,
        };
        let result = actor
            .slime
            .as_mut()
            .ok_or("missing_slime_controller")?
            .exit_ability(
                info,
                &mut actor.controller,
                &mut update,
                &mut controller_context,
                &mut |event| report.events.push(Event { actor: id, event }),
            );
        if let Some(slime) = &mut actor.slime {
            slime.sync_callback();
        }
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        if result == Err("actor_death_requested") {
            Self::kill_actor(
                actors,
                context,
                id,
                &crate::actor_combat::Kill {
                    killer: info.killer,
                    spawn_effect: false,
                    ignore_invulnerability: false,
                    cause: info.cause,
                },
                report,
            )?;
            actors[id]
                .slime
                .as_mut()
                .unwrap()
                .abilities
                .as_mut()
                .ok_or("missing_ability_routing")?
                .in_ability = false;
            return Ok(());
        }
        result
    }
    pub fn refresh_entities(
        schedule: &mut scheduler::State,
        components: &[Component],
        actors: &mut [Actor],
        scene: &mut Scene,
    ) {
        let lifecycle: Vec<_> = actors
            .iter()
            .map(|a| a.entity.as_ref().map(|e| (e.present, e.destroyed)))
            .collect();
        for actor in actors.iter_mut() {
            if let Some(routing) = actor.slime.as_mut().and_then(|s| s.abilities.as_mut()) {
                for ability in &mut routing.pool {
                    if let Some(crate::ability::Handler::Entity { actor }) = ability.handler
                        && let Some((present, destroyed)) = lifecycle[actor]
                    {
                        if present == Some(false) {
                            ability.present = false;
                        }
                        if destroyed == Some(true) {
                            ability.destroyed = true;
                        }
                    }
                }
            }
        }
        for actor in actors.iter() {
            if let Some(entity) = &actor.entity {
                schedule.nodes[entity.node].active = entity.active;
            }
        }
        for actor in actors.iter() {
            if let Some(entity) = &actor.entity {
                if entity.present == Some(false) {
                    continue;
                }
                if let Some(object) = scene.objects.iter_mut().find(|o| o.id == actor.object) {
                    object.active = schedule.active(entity.node);
                }
            }
        }
        for (i, component) in components.iter().enumerate() {
            if let Some(entity) = &actors[component.actor].entity {
                schedule.components[i].execution_gate = Some(
                    entity.present != Some(false)
                        && entity.active
                        && (!matches!(component.role, Role::Body) || entity.body_simulated)
                        && (!matches!(component.role, Role::Physics)
                            || actors[component.actor]
                                .slime
                                .as_ref()
                                .and_then(|s| s.spawn.as_ref())
                                .is_none_or(|s| s.physics_enabled)),
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
            if !actor.is_present() {
                continue;
            }
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
            &mut [Actor],
            &mut Context,
            &mut Report,
            scheduler::Event,
        ) -> scheduler::Result<()>,
    ) -> scheduler::Result<()> {
        Self::refresh_entities(
            &mut self.scheduler,
            &self.components,
            &mut self.actors,
            &mut self.scene,
        );
        match (&self.actor_ropes, &self.scene.rope_system) {
            (Some(binding), Some(system))
                if binding.metadata.len() == system.bodies.len()
                    && self
                        .actors
                        .iter()
                        .all(|a| a.body.rope.is_none_or(|i| i < system.bodies.len())) => {}
            (None, None) if self.actors.iter().all(|a| a.body.rope.is_none()) => {}
            _ => return Err("actor_rope_binding_invalid"),
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
            actor_ropes,
            combat,
        } = self;
        scheduler.tick(dt, &mut |schedule, phase| {
            report.phases.push(phase);
            before(
                actors,
                &mut Context {
                    schedule,
                    scene,
                    actor_ropes,
                    combat,
                    bindings,
                    grounds,
                    space: *space,
                },
                report,
                phase,
            )?;
            Self::refresh_entities(schedule, components, actors, scene);
            if let Some(effects) = combat.as_mut().and_then(|c| c.death_effects.as_mut())
                && effects.phase(schedule, scene, phase, report)?
            {
                return Ok(());
            }
            match phase.kind {
                scheduler::Phase::Physics => {
                    report.physics =
                        Some(scene.step(if schedule.time_scale < Fix::from_f32(0.01) {
                            Fix::ZERO
                        } else {
                            dt
                        }));
                    for a in actors.iter_mut() {
                        let Some(object) = scene.objects.iter().find(|o| o.id == a.object) else {
                            continue;
                        };
                        if !a.is_present() {
                            continue;
                        }
                        let transform = object.transform;
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
                    let callbacks = std::mem::take(&mut report.physics.as_mut().unwrap().callbacks);
                    for hit in callbacks {
                        report.physics.as_mut().unwrap().callbacks.push(hit.clone());
                        let Some(index) = actors.iter().position(|a| a.object == hit.receiver)
                        else {
                            continue;
                        };
                        if hit.layer == *player_layer {
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
                        if combat.as_ref().is_some_and(|c| c.collisions == Some(true)) {
                            Self::collide_actor(
                                actors,
                                &mut Context {
                                    schedule,
                                    scene,
                                    actor_ropes,
                                    combat,
                                    bindings,
                                    grounds,
                                    space: *space,
                                },
                                index,
                                &hit,
                                report,
                            )?;
                            Self::refresh_entities(schedule, components, actors, scene);
                        }
                    }
                }
                scheduler::Phase::Init | scheduler::Phase::Update | scheduler::Phase::Late => {
                    let component = components[phase.id.unwrap()];
                    if matches!(phase.kind, scheduler::Phase::Update)
                        && matches!(component.role, Role::Rewind)
                    {
                        return actors[component.actor].record_rewind();
                    }
                    if matches!(phase.kind, scheduler::Phase::Init)
                        && matches!(component.role, Role::Motor)
                        && actors[component.actor]
                            .slime
                            .as_ref()
                            .is_some_and(|s| s.spawn.is_some())
                    {
                        let result = Self::spawn_actor(
                            actors,
                            &mut Context {
                                schedule,
                                scene,
                                actor_ropes,
                                combat,
                                bindings,
                                grounds,
                                space: *space,
                            },
                            component.actor,
                            crate::actor_spawn::Action::Init,
                            Vec2::ZERO,
                            report,
                        );
                        Self::refresh_entities(schedule, components, actors, scene);
                        return result;
                    }

                    if matches!(phase.kind, scheduler::Phase::Update)
                        && matches!(component.role, Role::Dash)
                    {
                        let result = Self::update_dash(
                            actors,
                            &mut Context {
                                schedule,
                                scene,
                                actor_ropes,
                                combat,
                                bindings,
                                grounds,
                                space: *space,
                            },
                            component.actor,
                            dt,
                            report,
                        );
                        Self::refresh_entities(schedule, components, actors, scene);
                        return result;
                    }
                    if matches!(phase.kind, scheduler::Phase::Update)
                        && matches!(component.role, Role::Ability)
                    {
                        let result = Self::update_ordinary(
                            actors,
                            &mut Context {
                                schedule,
                                scene,
                                actor_ropes,
                                combat,
                                bindings,
                                grounds,
                                space: *space,
                            },
                            component.actor,
                        );
                        Self::refresh_entities(schedule, components, actors, scene);
                        return result;
                    }
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
                        rope_gravity: scene.rope_system.as_ref().map_or(Fix::ZERO, |s| s.gravity),
                        player_gravity: actor_ropes
                            .as_ref()
                            .map_or(Fix::ZERO, |r| r.player_gravity),
                        max_stretch: actor_ropes.as_ref().map_or(Fix::ZERO, |r| r.max_stretch),
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
                    let mut pending_entry = None;
                    let result = match (phase.kind, component.role) {
                        (scheduler::Phase::Init, Role::Body) => {
                            actor.body.init();
                            Ok(())
                        }
                        (scheduler::Phase::Update, Role::Collision) => {
                            actor.collision.as_mut().ok_or("null_reference")?.was_killed = false;
                            Ok(())
                        }
                        (scheduler::Phase::Update, Role::Physics) => {
                            actor
                                .controller
                                .update(dt, &mut context, &mut |controller, event| {
                                    if let Some(slime) = &mut actor.slime {
                                        slime.physics_event(
                                            &mut controller.physics,
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
                                let mut update = crate::slime::Update {
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
                                };
                                match slime.begin_update(
                                    &mut actor.controller,
                                    &mut update,
                                    &mut context,
                                    &mut emit,
                                ) {
                                    Ok(Some(request)) => {
                                        pending_entry = Some(request);
                                        Ok(())
                                    }
                                    Ok(None) => slime.finish_update(
                                        &mut actor.controller,
                                        &mut update,
                                        &mut context,
                                        &mut emit,
                                    ),
                                    Err(error) => Err(error),
                                }
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
                            let result = if let Some(binding) = actor_ropes {
                                actor.body.update_with(
                                    dt,
                                    &mut player_body::JoinedRopes {
                                        physics: &mut context
                                            .scene
                                            .rope_system
                                            .as_mut()
                                            .unwrap()
                                            .bodies,
                                        metadata: &mut binding.metadata,
                                    },
                                    &body_context,
                                    &mut || {
                                        emit(player_controller::Event::Sound(
                                            "hookshotLetGo".into(),
                                        ))
                                    },
                                )
                            } else {
                                actor.body.update(dt, &mut [], &body_context)
                            };
                            let result = result.map(|_| ());
                            actor.body_to_controller();
                            result
                        }
                        (scheduler::Phase::Late, Role::Physics) => {
                            let rope = actor.body.rope.map(|i| {
                                (
                                    &mut context.scene.rope_system.as_mut().unwrap().bodies[i],
                                    actor.body.top,
                                )
                            });
                            actor.controller.physics.late_update_rope(
                                &platforms,
                                rope,
                                &actor.config.physics,
                                &mut |physics, angle| {
                                    let event = player_controller::Event::Detached(angle);
                                    if let Some(slime) = &mut actor.slime {
                                        slime.physics_event(
                                            physics,
                                            &event,
                                            bindings,
                                            actor.body.holder.ok_or("null_reference")?,
                                            &mut emit,
                                        )?;
                                    }
                                    emit(event);
                                    Ok(())
                                },
                            )
                        }
                        (scheduler::Phase::Late, Role::Body) => {
                            if let Some(binding) = actor_ropes {
                                actor.body.late_update_in(
                                    &mut player_body::JoinedRopes {
                                        physics: &mut context
                                            .scene
                                            .rope_system
                                            .as_mut()
                                            .unwrap()
                                            .bodies,
                                        metadata: &mut binding.metadata,
                                    },
                                    &body_context,
                                )
                            } else {
                                actor.body.late_update(&mut [], &body_context)
                            }
                        }
                        _ => Ok(()),
                    };
                    if let Some(slime) = &mut actor.slime {
                        slime.sync_callback();
                    }
                    actor.controller_to_body();
                    actor.sync_transform(scene);
                    Self::refresh_entities(schedule, components, actors, scene);
                    if result == Err("actor_spawn_requested") {
                        let result = Self::spawn_actor(
                            actors,
                            &mut Context {
                                schedule,
                                scene,
                                actor_ropes,
                                combat,
                                bindings,
                                grounds,
                                space: *space,
                            },
                            component.actor,
                            crate::actor_spawn::Action::Spawn,
                            Vec2::ZERO,
                            report,
                        );
                        Self::refresh_entities(schedule, components, actors, scene);
                        return result;
                    }
                    result?;
                    if let Some(request) = pending_entry {
                        let mut context = Context {
                            schedule,
                            scene,
                            actor_ropes,
                            combat,
                            bindings,
                            grounds,
                            space: *space,
                        };
                        Self::enter_ordinary(
                            actors,
                            &mut context,
                            request.target,
                            request.info,
                            component.actor,
                            report,
                        )?;
                        *actors[component.actor]
                            .slime
                            .as_mut()
                            .unwrap()
                            .abilities
                            .as_mut()
                            .unwrap()
                            .timer(request.slot)? = Fix::ZERO;
                        let result =
                            Self::finish_slime(actors, &mut context, component.actor, dt, report);
                        Self::refresh_entities(schedule, components, actors, scene);
                        result?;
                    }
                }
                scheduler::Phase::DestroyObject => {
                    for actor in actors.iter_mut() {
                        if let Some(entity) = &mut actor.entity {
                            let mut node = Some(entity.node);
                            while let Some(id) = node {
                                if Some(id) == phase.id {
                                    entity.present = Some(false);
                                    break;
                                }
                                node = schedule.nodes[id].parent;
                            }
                        }
                    }
                    for player in &mut schedule.players {
                        if let Some(position) = &mut player.position {
                            position.current = position.current.filter(|&i| actors[i].is_present());
                            position.previous =
                                position.previous.filter(|&i| actors[i].is_present());
                        }
                    }
                }
                scheduler::Phase::RopeMesh => {
                    if let Some(system) = &mut scene.rope_system {
                        system.space = *space;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}
