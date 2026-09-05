//! Ability component ownership and synchronous SlimeController callbacks.
use crate::{
    ability::{ExitInfo, PlayerInfo},
    actors::{Actor, Context, Report, Runtime},
    fixed::Fix,
    scheduler,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kill_players_on_contact: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dash: Option<crate::dash_ability::Instance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quick_spell: Option<crate::quick_spell::Instance>,
    pub owner: usize,
    pub info: PlayerInfo,
    pub slime: Option<usize>,
    pub callback: Option<usize>,
    pub previous_scale: Fix,
    pub body_linked: bool,
    pub collider_linked: bool,
}
impl Runtime {
    pub fn enter_ordinary(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        info: PlayerInfo,
        owner: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let priorities = Self::position_priorities(actors);
        let actor = &mut actors[id];
        let ability = actor.ordinary.as_mut().ok_or("missing_ordinary_ability")?;
        ability.info = info;
        ability.callback = Some(owner);
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
        ability.slime = ability.info.slime;
        actor.body.holder = Some(ability.info.player);
        if ability.body_linked && ability.info.rope.is_some() {
            context.attach_rope(&mut actor.body, ability.info.rope, ability.info.top)?;
        }
        let linked = ability.collider_linked;
        let slime = ability.slime;
        let player_id = ability.info.player;
        if linked {
            let scale = context
                .schedule
                .players
                .iter()
                .find(|p| p.id == player_id)
                .ok_or("null_reference")?
                .scale
                .unwrap_or(Fix::ONE);
            actor.set_collider_scale(scale, context.scene)?;
        }
        if let Some(player) = context
            .schedule
            .players
            .iter_mut()
            .find(|p| p.id == player_id)
        {
            if !linked {
                return Err("null_reference");
            }
            if let Some(position) = &mut player.position {
                position.set(actor.controller.physics.motion.position, slime, &priorities);
            }
        }
        Self::enter_dash(actors, context, id, report)?;
        Self::enter_quick_spell(actors, context, id, report)
    }
    pub fn update_ordinary(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
    ) -> scheduler::Result<()> {
        let priorities = Self::position_priorities(actors);
        let actor = &mut actors[id];
        let ability = actor.ordinary.as_mut().ok_or("missing_ordinary_ability")?;
        if !ability.collider_linked {
            return Ok(());
        }
        let player = context
            .schedule
            .players
            .iter_mut()
            .find(|p| p.id == ability.info.player)
            .ok_or("null_reference")?;
        let scale = player.scale.unwrap_or(Fix::ONE);
        if scale != ability.previous_scale {
            let _multiplier = scale
                .checked_div(ability.previous_scale)
                .map_err(|_| "division_by_zero")?;
            ability.previous_scale = scale;
        }
        let slime = ability.slime;
        actor.set_collider_scale(scale, context.scene)?;
        if let Some(position) = &mut player.position {
            position.set(actor.controller.physics.motion.position, slime, &priorities);
        }
        Ok(())
    }
    pub fn exit_ordinary(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        info: &ExitInfo,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let priorities = Self::position_priorities(actors);
        let actor = &mut actors[id];
        let ability = actor.ordinary.as_mut().ok_or("missing_ordinary_ability")?;
        let mut info = info.clone();
        if ability.body_linked && actor.body.rope.is_some() {
            info.rope = actor.body.rope;
            info.top = actor.body.top;
            actor.body.clear_for_ability();
        }
        if let Some(player) = context
            .schedule
            .players
            .iter_mut()
            .find(|p| p.id == ability.info.player)
        {
            if !ability.collider_linked {
                return Err("null_reference");
            }
            if let Some(position) = &mut player.position {
                position.set(
                    actor.controller.physics.motion.position,
                    ability.slime,
                    &priorities,
                );
            }
        }
        let entity = actor.entity.as_mut().ok_or("missing_actor_entity")?;
        entity.active = false;
        context.schedule.nodes[entity.node].active = false;
        context
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == actor.object)
            .ok_or("missing_actor_object")?
            .active = false;
        let callback = ability.callback.ok_or("null_reference")?;
        let first_event = report.events.len();
        let result = Self::exit_actor(actors, context, callback, &info, report);
        // The reference audio hook records the dispatching actor. Physics
        // delegates carry their own actor identity across the nested callback.
        for event in &mut report.events[first_event..] {
            if matches!(
                event.event,
                crate::player_controller::Event::Sound(_)
                    | crate::player_controller::Event::JumpSound(_)
            ) {
                event.actor = id;
            }
        }
        result
    }
}
