//! SlimeController's target selection, delayed eating death and growth.
use crate::{
    ability::Handler,
    actors::{Actor, Context, Event, Report, Runtime},
    animator::Start,
    fixed::Fix,
    player_controller, scheduler,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub target: Option<usize>,
    pub swallowing: bool,
    pub scale_increase: Fix,
}

impl Actor {
    pub(crate) fn should_bite(
        &self,
        other: &Actor,
        schedule: &scheduler::State,
        combat: &Option<crate::actor_combat::Combat>,
    ) -> scheduler::Result<bool> {
        if self.push_strength.is_none()
            || self.slime.as_ref().is_some_and(|s| s.state.mode == 9)
            || !self.controller.physics.motion.grounded
        {
            return Ok(false);
        }
        let holder = self.body.holder.ok_or("null_reference")?;
        if holder != 1000
            && schedule.time_scale < Fix::from_f32(0.01)
            && !schedule
                .players
                .iter()
                .find(|p| p.id == holder)
                .ok_or("null_reference")?
                .protected
        {
            return Ok(false);
        }
        if self.controller.physics.scale - other.controller.physics.scale
            < self.bite_threshold.unwrap_or(Fix::ONE)
            || other.collision.is_none()
        {
            return Ok(false);
        }
        let book = &combat.as_ref().ok_or("actor_bite_context_missing")?.book;
        let mine = book
            .players
            .iter()
            .find(|p| p.id == holder)
            .ok_or("null_reference")?;
        let theirs = book
            .players
            .iter()
            .find(|p| Some(p.id) == other.body.holder)
            .ok_or("null_reference")?;
        Ok(mine.team != theirs.team)
    }
}

impl crate::slime::Instance {
    pub(crate) fn eat_animation(
        &mut self,
        name: &str,
        callback: Option<u32>,
    ) -> scheduler::Result<()> {
        let frames = self
            .animations
            .get(name)
            .cloned()
            .ok_or("animation_missing")?;
        self.animator.start(
            Start::Begin {
                frames: Some(frames),
                next: None,
                callback,
                style: if callback.is_some() { 3 } else { 2 },
            },
            std::slice::from_ref(&self.speed),
            &mut |_, _| Err("unexpected_eating_entry_callback"),
        )
    }
}

impl Runtime {
    /// GetActivePlayerCollision can resolve an active ordinary ability while
    /// the original slime is inactive. The ordered ability list owns priority.
    fn active_eating_collision(
        actors: &[Actor],
        schedule: &scheduler::State,
        id: usize,
    ) -> scheduler::Result<Option<usize>> {
        let actor = &actors[id];
        let slime = actor.slime.as_ref().ok_or("null_reference")?;
        let active = |a: &Actor| {
            a.is_present() && a.entity.as_ref().is_some_and(|e| schedule.active(e.node))
        };
        if slime.managed_destroyed != Some(true) && active(actor) {
            return Ok(actor.collision.as_ref().map(|_| id));
        }
        let routing = slime.abilities.as_ref().ok_or("missing_ability_routing")?;
        for slot in routing.slots.iter().flatten() {
            let ability = &routing.pool[*slot];
            if !ability.present || ability.destroyed {
                continue;
            }
            if let Some(Handler::Entity { actor: target }) = ability.handler {
                let target_actor = &actors[target];
                if active(target_actor)
                    && target_actor
                        .entity
                        .as_ref()
                        .is_none_or(|e| e.destroyed != Some(true))
                    && target_actor.collision.is_some()
                {
                    return Ok(Some(target));
                }
            }
        }
        Ok(None)
    }

    pub fn bite_actor(
        actors: &mut [Actor],
        context: &mut Context,
        eater: usize,
        victim: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let candidate = &actors[victim];
        if !candidate.is_present() {
            return Err("missing_reference");
        }
        candidate.collision.as_ref().ok_or("null_reference")?;
        let target = if candidate
            .slime
            .as_ref()
            .is_some_and(|s| s.managed_destroyed != Some(true))
        {
            Some(Some(victim))
        } else if candidate
            .entity
            .as_ref()
            .is_none_or(|e| e.destroyed != Some(true))
        {
            candidate.ordinary.as_ref().map(|a| a.slime)
        } else {
            None
        };
        let eating = actors[eater]
            .slime
            .as_mut()
            .ok_or("null_reference")?
            .eating
            .as_mut()
            .ok_or("missing_actor_eating")?;
        if let Some(target) = target {
            eating.target = target;
        }
        let Some(target) = eating.target else {
            return Ok(());
        };
        if !actors[target].is_present()
            || actors[target]
                .slime
                .as_ref()
                .is_none_or(|s| s.managed_destroyed == Some(true))
        {
            return Ok(());
        }
        if let Some(active) = Self::active_eating_collision(actors, context.schedule, target)?
            && actors[active]
                .entity
                .as_ref()
                .is_none_or(|e| e.destroyed != Some(true))
            && actors[active].collision.as_ref().unwrap().invulnerable != 0
        {
            return Ok(());
        }
        report.events.push(Event {
            actor: eater,
            event: player_controller::Event::Sound("bite".into()),
        });
        let slime = actors[eater].slime.as_mut().unwrap();
        slime.state.mode = 9;
        slime.eat_animation("bite", Some(5))
    }

    pub fn chew_actor(
        actors: &mut Vec<Actor>,
        context: &mut Context,
        eater: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let target = actors[eater]
            .slime
            .as_ref()
            .ok_or("null_reference")?
            .eating
            .as_ref()
            .ok_or("missing_actor_eating")?
            .target;
        let active = target
            .map(|id| Self::active_eating_collision(actors, context.schedule, id))
            .transpose()?
            .flatten();
        if let Some(active) = active
            && actors[active].is_present()
            && actors[active]
                .entity
                .as_ref()
                .is_none_or(|e| e.destroyed != Some(true))
        {
            let killer = actors[eater].body.holder.ok_or("null_reference")?;
            Self::kill_actor(
                actors,
                context,
                active,
                &crate::actor_combat::Kill {
                    killer,
                    spawn_effect: false,
                    ignore_invulnerability: false,
                    cause: crate::player_life::Cause::Eaten as i32,
                },
                report,
            )?;
            let player = context
                .schedule
                .players
                .iter_mut()
                .find(|p| p.id == killer)
                .ok_or("null_reference")?;
            let count = player.eaten_count.as_mut().ok_or("missing_eaten_count")?;
            *count = count.wrapping_add(1);
        }
        actors[eater]
            .slime
            .as_mut()
            .unwrap()
            .eat_animation("chew", Some(6))
    }
}
