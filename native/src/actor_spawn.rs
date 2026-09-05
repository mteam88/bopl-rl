//! Bind SlimeController spawn transitions to the live actor, player and collider.
use crate::{
    actors::{Actor, Context, Event, Report, Runtime},
    scheduler, spawn,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub physics_enabled: bool,
    pub renderer_enabled: bool,
    pub fast: bool,
    pub next_ability: i32,
    pub abilityless: bool,
    pub tutorial: bool,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Init,
    Before,
    Spawn,
    Prepare,
    Respawn,
    Fast,
    Unground,
}
impl Runtime {
    pub fn spawn_actor(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        action: Action,
        position: Vec2,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        if !actor.is_present() {
            return Err("missing_reference");
        }
        let slime = actor.slime.as_mut().ok_or("missing_slime_controller")?;
        let metadata = slime.spawn.as_mut().ok_or("missing_actor_spawn")?;
        let routing = slime.abilities.as_mut().ok_or("missing_ability_routing")?;
        let entity = actor.entity.as_mut().ok_or("missing_actor_entity")?;
        let collision = actor.collision.as_mut().ok_or("missing_actor_collision")?;
        let object = context
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == actor.object)
            .or(actor.pending_object.as_mut())
            .ok_or("missing_actor_object")?;
        let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
        let mut players: Vec<_> = context
            .schedule
            .players
            .iter()
            .zip(&combat.revival)
            .map(|(p, r)| {
                let flags = p.flags.clone().unwrap_or_default();
                spawn::Player {
                    id: p.id,
                    protected: p.protected,
                    extra_life: r.status,
                    spark: flags.spark,
                    macho: flags.macho,
                    platform: flags.platform,
                    revive: None,
                }
            })
            .collect();
        let surfaces: Vec<_> = context
            .bindings
            .iter()
            .map(|b| {
                b.platform
                    .as_ref()
                    .expect("actor platform binding")
                    .surface
                    .clone()
            })
            .collect();
        let config = spawn::Config {
            player: actor.body.holder.ok_or("null_reference")?,
            holder: actor.body.holder,
            jump_strength: actor.config.physics.jump_strength,
            tutorial: metadata.tutorial,
            animations: Some(spawn::Animations {
                spawn: slime.animations.get("spawn").cloned(),
                fall_loop: slime.animations.get("fallLeftLoop").cloned(),
                fall_map: Some(slime.fall_map.clone()),
            }),
            animation_speed: 0,
            callback: Some(4),
        };
        let indicator_active: Vec<_> = routing
            .indicators
            .iter()
            .map(|i| {
                i.as_ref()
                    .and_then(|v| v.lifecycle.as_ref())
                    .is_some_and(|v| v.active)
            })
            .collect();
        let mut state = spawn::State {
            motion: actor.controller.physics.motion.clone(),
            physics_enabled: metadata.physics_enabled,
            simulated: entity.body_simulated,
            hurtbox_enabled: object.enabled,
            renderer_enabled: metadata.renderer_enabled,
            mode: slime.state.mode,
            suppress: slime.state.suppress,
            has_spawned: slime.state.has_spawned,
            fast: metadata.fast,
            invulnerable: collision.invulnerable,
            next_ability: metadata.next_ability,
            abilityless: metadata.abilityless,
            abilities: Some(routing.slots.clone()),
            ability_pool: vec![false; routing.pool.len()],
            ability_present: Some(routing.pool.iter().map(|a| a.present).collect()),
            cooldowns: routing.timers.clone(),
            indicator_refs: Some(
                routing
                    .indicators
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v.as_ref().map(|_| i))
                    .collect(),
            ),
            indicators: indicator_active.clone(),
        };
        let mut events = vec![];
        let mut spawn_context = spawn::Context {
            animator: &mut slime.animator,
            speeds: std::slice::from_mut(&mut slime.speed),
            players: &mut players,
            surfaces: &surfaces,
            config: &config,
            events: &mut events,
            on_detach: &mut |_, _| Ok(()),
        };
        let result = match action {
            Action::Init => state.init(&mut spawn_context),
            Action::Before => state.before(&mut spawn_context),
            Action::Spawn => state.spawn(&mut spawn_context),
            Action::Prepare => state.prepare(position, &mut spawn_context),
            Action::Respawn => state.respawn(position, &mut spawn_context),
            Action::Fast => state.fast_respawn(position, &mut spawn_context),
            Action::Unground => state.unground(&mut spawn_context),
        };
        // Apply the prefix even when a later callback/reference fails.
        actor.controller.physics.motion = state.motion;
        entity.body_simulated = state.simulated;
        object.enabled = state.hurtbox_enabled;
        metadata.physics_enabled = state.physics_enabled;
        metadata.renderer_enabled = state.renderer_enabled;
        metadata.fast = state.fast;
        metadata.next_ability = state.next_ability;
        collision.invulnerable = state.invulnerable;
        slime.state.mode = state.mode;
        slime.state.suppress = state.suppress;
        slime.state.has_spawned = state.has_spawned;
        routing.timers = state.cooldowns;
        for (i, indicator) in routing.indicators.iter_mut().enumerate() {
            if let Some(indicator) = indicator {
                if !indicator_active[i]
                    && state.indicators[i]
                    && indicator.seconds_since_reset > 0.0
                {
                    indicator.seconds_since_reset = 11.0;
                }
                if let Some(lifecycle) = &mut indicator.lifecycle {
                    lifecycle.active = state.indicators[i];
                }
            }
        }
        for ((target, revival), source) in context
            .schedule
            .players
            .iter_mut()
            .zip(&mut combat.revival)
            .zip(players)
        {
            revival.status = source.extra_life;
            if let Some(flags) = &mut target.flags {
                flags.spark = source.spark;
                flags.macho = source.macho;
                flags.platform = source.platform;
            }
        }
        for event in events {
            report.events.push(Event {
                actor: id,
                event: match event {
                    spawn::Event::Detached(v) => crate::player_controller::Event::Detached(v),
                    spawn::Event::JumpSound(v) => crate::player_controller::Event::JumpSound(v),
                },
            });
        }
        slime.sync_callback();
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        result
    }
}
