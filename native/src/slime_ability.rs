//! SlimeController's instant ability entry and invisibility completion. The
//! stored PlayerInfo is written in source order so failed calls retain its prefix.
use crate::{
    ability::{Handler, PlayerInfo},
    ability_routing::Result,
    fixed::Fix,
    player_controller::{self, Event},
    slime::{Instance, Update},
    vector::Vec2,
};

#[derive(Clone, Debug)]
pub(crate) struct Request {
    pub target: usize,
    pub slot: usize,
    pub info: PlayerInfo,
}
impl Instance {
    pub(crate) fn enter_ability(
        &mut self,
        slot: usize,
        controller: &player_controller::State,
        update: &mut Update,
        context: &player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<Option<Request>> {
        let routing = self.abilities.as_mut().ok_or("missing_ability_routing")?;
        let ability_id = routing.slots[slot].ok_or("null_reference")?;
        if routing.pool[ability_id].handler.is_none() {
            return Err("ability_entry_not_implemented");
        }
        let entry = self.entry.as_mut().ok_or("missing_ability_entry_state")?;
        entry.info = PlayerInfo::default();
        let info = &mut entry.info;
        let motion = &controller.physics.motion;
        info.rope_target = entry.rope_target;
        info.use_rope_on_player = false; // No rope ability is routed here yet.
        info.slot = slot as i32;
        info.grounded_speed = motion.grounded_speed;
        info.grounded = motion.grounded;
        info.player = context.holder.ok_or("null_reference")?;
        let render = self.animator.render.as_ref().ok_or("null_reference")?;
        info.material = Some(update.actor as u32);
        info.position = motion.position;
        info.sprite = render.sprite;
        info.radius = motion.radius;
        let platform = || {
            motion
                .ground
                .and_then(|i| context.bindings.get(i as usize))
                .and_then(|b| b.platform.as_ref())
                .ok_or("null_reference")
        };
        info.forward = if motion.grounded {
            platform()?.surface.forward(motion.position)
        } else {
            Vec2::RIGHT
                * if self.state.facing_right {
                    Fix::ONE
                } else {
                    -Fix::ONE
                }
        };
        info.slime = Some(update.actor);
        info.external_velocity = motion.external_velocity;
        info.velocity = if motion.grounded {
            info.forward * motion.grounded_speed
        } else {
            motion.velocity
        };
        info.rope = update.body.rope;
        info.top = update.body.top;
        if matches!(
            routing.pool[ability_id].handler,
            Some(Handler::Entity { .. })
        ) {
            update.body.velocity = motion.velocity;
            update.body.external_velocity = motion.external_velocity;
            update.body.clear_for_ability();
        }
        info.up = if motion.grounded {
            let normal = platform()?.surface.normal(motion.position);
            let length = normal.length();
            Vec2::new(
                normal
                    .x
                    .checked_div(length)
                    .map_err(|_| "division_by_zero")?,
                normal
                    .y
                    .checked_div(length)
                    .map_err(|_| "division_by_zero")?,
            )
        } else {
            Vec2::UP
        };
        if !self.state.facing_right {
            info.forward *= -Fix::ONE;
        }
        info.rotation = motion.rotation;
        info.facing_right = self.state.facing_right;
        info.ground = motion.ground;
        if motion.grounded {
            info.local = motion.local;
        }
        let ability = &mut routing.pool[ability_id];
        if !ability.present || ability.destroyed {
            *routing.timer(slot)? = Fix::ZERO;
            return Ok(None);
        }
        if let Some(flags) = &mut update.player.flags {
            flags.spark = false;
            flags.macho = false;
            flags.platform = false;
        }
        // InstantAbility does not deactivate the slime or enter its in-ability
        // mode. The callback changes the shared player state synchronously.
        let handler = ability.handler.as_mut().unwrap();
        if let Handler::Entity { actor } = handler {
            routing.in_ability = true;
            update.player.time_ability = Fix::ZERO;
            *entry
                .invisible_on_entry
                .as_mut()
                .ok_or("missing_ability_entry_state")? = update.player.invisible;
            update.entity.as_mut().ok_or("missing_actor_entity")?.active = false;
            emit(Event::AbilityEntry(slot));
            return Ok(Some(Request {
                target: *actor,
                slot,
                info: info.clone(),
            }));
        }
        emit(Event::AbilityEntry(slot));
        let instant = handler.instant_mut().unwrap();
        instant.info = info.clone();
        instant.active = true;
        instant.slime = info.slime;
        if !instant.transform_present {
            return Err("null_reference");
        }
        instant.position = info.position;
        if let Handler::Invisibility(effect) = handler
            && effect.component_present
        {
            if !effect.linked {
                return Err("null_reference");
            }
            update.player.invisible = true;
            update.player.time_invisible = Fix::ZERO;
            update.player.time_ability = Fix::ZERO;
            ability.cooldown = Fix::from_int(10000);
        }
        let instant = handler.instant_mut().unwrap();
        if let Some(audio) = &instant.audio
            && !audio.is_empty()
        {
            emit(Event::Sound(audio.clone()));
        }
        instant.active = false;
        *routing.timer(slot)? = Fix::ZERO;
        Ok(None)
    }
    pub(crate) fn finish_invisibility(&mut self, invisible: bool) -> Result<()> {
        if let Some(entry) = &mut self.entry {
            if !invisible && entry.was_invisible {
                self.abilities
                    .as_mut()
                    .ok_or("missing_ability_routing")?
                    .invisibility_ended()?;
            }
            entry.was_invisible = invisible;
        }
        Ok(())
    }
}
