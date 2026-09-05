//! PlayerCollision.killPlayer body lifecycle, sharing scheduled actors and players.
use crate::{
    actors::{Actor, Context, Report, Runtime},
    player_life::{Book, Death},
    scheduler,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Collision {
    pub invulnerable: u32,
    pub was_killed: bool,
    pub death_trauma: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Revival {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_position: Option<crate::vector::Vec2>,
    pub status: i32,
    pub stocks: i32,
    pub respawn_count: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Combat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub death_effects: Option<crate::death_effect::Effects>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collisions: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitboxes: Option<Vec<Hitbox>>,
    pub book: Book,
    pub revival: Vec<Revival>,
    pub events: Vec<Death>,
    pub trauma: f32,
    pub max_trauma: f32,
    pub camera_present: bool,
    pub stocks_enabled: bool,
    pub drop_enabled: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kill {
    pub killer: i32,
    pub spawn_effect: bool,
    pub ignore_invulnerability: bool,
    pub cause: i32,
}
impl Runtime {
    pub fn kill_actor(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        request: &Kill,
        report: &mut Report,
    ) -> scheduler::Result<bool> {
        let actor = &mut actors[id];
        if !actor.is_present() {
            return Err("missing_reference");
        }
        actor.collision.as_ref().ok_or("null_reference")?;
        if actor.body.rope.is_some() {
            let top = actor.body.top;
            context.attach_rope(&mut actor.body, None, top)?;
        }
        let collision = actor.collision.as_ref().ok_or("null_reference")?;
        if (collision.invulnerable != 0 && !request.ignore_invulnerability)
            || !actor.entity.as_ref().ok_or("missing_actor_entity")?.active
        {
            return Ok(false);
        }
        let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
        let victim = combat
            .book
            .players
            .iter()
            .position(|p| Some(p.id) == actor.body.holder);
        if request.spawn_effect && victim.is_none_or(|i| combat.revival[i].status != 2) {
            let effects = combat.death_effects.as_mut().ok_or("instantiate_null")?;
            effects.spawn(
                context.schedule,
                actor.body.position,
                id,
                actor.controller.physics.scale.to_f64() as f32,
                actor
                    .ordinary
                    .as_ref()
                    .map_or(Some(id as u32), |a| a.info.material),
            )?;
        }
        if actor.ordinary.as_ref().is_some_and(|a| a.dash.is_some()) {
            let info = crate::ability::ExitInfo {
                position: actor.body.position,
                dead: true,
                killer: request.killer,
                cause: request.cause,
                ..Default::default()
            };
            Self::exit_dash(actors, context, id, info, report)?;
            return Ok(true);
        }
        let combat = context.combat.as_mut().unwrap();
        if !combat.camera_present {
            return Err("null_reference");
        }
        combat.trauma = (combat.trauma + collision.death_trauma).min(combat.max_trauma);
        let Some(victim) = victim else {
            Self::destroy_actor(actors, context, id)?;
            return Ok(true);
        };
        let revival = &combat.revival[victim];
        if revival.respawn_count > 0 {
            return Err("actor_revival_not_implemented");
        }
        if revival.status == 0
            && combat.book.players[victim].bodies_alive == 1
            && combat.stocks_enabled
            && revival.stocks > 0
        {
            return Err("actor_stock_respawn_not_implemented");
        }
        if revival.status == 0 || revival.status == 2 {
            let slime = actor.slime.as_mut().ok_or("null_reference")?;
            let routing = slime.abilities.as_mut().ok_or("missing_ability_routing")?;
            if combat.book.players[victim].bodies_alive == 1
                && combat.drop_enabled
                && !routing.slots.is_empty()
            {
                return Err("actor_ability_drop_not_implemented");
            }
            if routing.slots.iter().flatten().any(|&i| {
                let a = &routing.pool[i];
                a.present
                    && !a.destroyed
                    && !matches!(a.handler, Some(crate::ability::Handler::Entity { .. }))
            }) {
                return Err("actor_pooled_ability_lifecycle_not_implemented");
            }
            if let Some(event) = combat.book.kill(
                victim,
                request.killer,
                context.schedule.ticks as i64,
                request.cause,
            ) {
                combat.events.push(event);
            }
            let targets: Vec<_> = routing
                .slots
                .iter()
                .flatten()
                .filter_map(|&i| {
                    let a = &mut routing.pool[i];
                    if !a.present || a.destroyed {
                        return None;
                    }
                    a.destroyed = true;
                    match a.handler {
                        Some(crate::ability::Handler::Entity { actor }) => Some(actor),
                        _ => None,
                    }
                })
                .collect();
            routing.slots.clear();
            // Each ability GameObject is destroyed before the owning slime.
            for target in targets {
                Self::destroy_actor(actors, context, target)?;
            }
            Self::destroy_actor(actors, context, id)?;
        } else if revival.status == 3 {
            let position = revival
                .spawn_position
                .ok_or("missing_actor_spawn_position")?;
            Self::spawn_actor(
                actors,
                context,
                id,
                crate::actor_spawn::Action::Unground,
                position,
                report,
            )?;
            context.combat.as_mut().unwrap().revival[victim].status = 2;
            Self::spawn_actor(
                actors,
                context,
                id,
                crate::actor_spawn::Action::Prepare,
                position,
                report,
            )?;
            Self::spawn_actor(
                actors,
                context,
                id,
                crate::actor_spawn::Action::Respawn,
                position,
                report,
            )?;
        }
        Ok(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hitbox {
    pub object: u32,
    pub owner: i32,
}
