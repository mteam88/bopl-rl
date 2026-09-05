//! Original SlimeController.onExitAbility ordering, including failure prefixes.
use crate::{
    ability::ExitInfo,
    fixed::Fix,
    player_controller::{self, Event},
    slime::{Instance, Update},
};
type Result<T> = crate::scheduler::Result<T>;
impl Instance {
    pub(crate) fn exit_ability(
        &mut self,
        info: &ExitInfo,
        controller: &mut player_controller::State,
        update: &mut Update,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        if info.rope.is_some() {
            return Err("actor_rope_integration_not_implemented");
        }
        let entity = update.entity.as_mut().ok_or("missing_actor_entity")?;
        entity.active = true;
        update.player.time_ability = Fix::ZERO;
        if self.entry.as_ref().and_then(|e| e.invisible_on_entry) == Some(true)
            && !update.player.invisible
        {
            self.abilities
                .as_mut()
                .ok_or("missing_ability_routing")?
                .invisibility_ended()?;
        }
        entity.collider.set_scale(
            update.player.scale.unwrap_or(Fix::ONE),
            &mut controller.physics.scale,
            update.object,
            context.scene,
        )?;
        if let Some(flags) = &mut update.player.flags {
            flags.spark = false;
            flags.macho = false;
            flags.platform = false;
        }
        if info.dead {
            controller.physics.motion.position = info.position;
            // These composed entities currently have no PlayerCollision. The
            // source dereferences that missing component after writing position.
            return Err("null_reference");
        }
        self.state.facing_right = info.facing_right;
        let platforms: Vec<_> = context
            .bindings
            .iter()
            .map(|b| b.platform.clone().expect("actor platform binding"))
            .collect();
        controller.physics.sync(&info.physics(), true, &platforms)?;
        let id = context.holder.ok_or("null_reference")?;
        if controller.physics.motion.grounded {
            self.enter(if info.enter_turn { 7 } else { 0 }, Fix::ZERO, id, emit)?;
            if info.just_landed {
                self.land(
                    info.ground.and_then(|i| platforms.get(i as usize)),
                    controller.physics.motion.position,
                    id,
                    emit,
                )?;
            }
            if info.enter_jump {
                let ground = controller
                    .physics
                    .motion
                    .ground
                    .and_then(|i| platforms.get(i as usize))
                    .ok_or("null_reference")?;
                self.state.was_upside_down =
                    ground.surface.normal(controller.physics.motion.position).y < Fix::ZERO;
                let angle = controller
                    .physics
                    .jump(&platforms, &context.config.physics)?;
                let event = Event::Detached(angle);
                self.physics_event(controller, &event, context.bindings, id, emit)?;
                emit(event);
                emit(Event::JumpSound(format!("slimeJump{id}")));
            }
        } else {
            self.enter_fall_sprite(controller, info.sprite, context, emit)?;
        }
        let motion = &controller.physics.motion;
        let transform = &mut context
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == update.object)
            .ok_or("missing_actor_object")?
            .transform;
        transform.position = motion.position;
        transform.rotation = motion.rotation;
        transform.right = Some(motion.right);
        entity.collider.sync(update.object, context.scene)?;
        let routing = self.abilities.as_mut().ok_or("missing_ability_routing")?;
        for i in 0..routing.slots.len() {
            if let Some(id) = routing.slots[i]
                && routing.pool[id].present
                && !routing.pool[id].destroyed
            {
                routing.indicator(i)?.reactivate_and_sync()?;
            }
        }
        routing.in_ability = false;
        Ok(())
    }
}
