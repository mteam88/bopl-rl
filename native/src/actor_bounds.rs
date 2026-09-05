//! DestroyIfOutsideSceneBounds for scheduled player and ability actors.
use crate::{
    actors::{Actor, Context, Report, Runtime},
    death_effect::{Kind, Spawn},
    fixed::Fix,
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    pub kind: Kind,
    pub offset: Vec2,
    pub scale: f32,
    pub material: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub x: bool,
    pub y: bool,
    pub no_up: bool,
    pub transform_present: bool,
    pub x_effect: Option<Effect>,
    pub up_effect: Option<Effect>,
    pub down_effect: Option<Effect>,
    pub use_x_material: bool,
    pub use_up_material: bool,
    pub use_down_material: bool,
}
impl Runtime {
    pub(crate) fn update_bounds(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &actors[id];
        let bounds = actor.bounds.as_ref().ok_or("missing_actor_bounds")?;
        // The source first dereferences fixTrans only in an enabled axis test.
        if (bounds.y || bounds.x) && !bounds.transform_present {
            return Err("null_reference");
        }
        let position = actor.body.position;
        let water = Fix::from_f32(-11.3);
        let bottom = Fix::from_int(-26);
        let top = Fix::from_f32(72.5);
        let left = Fix::from_f32(-131.25);
        let right = Fix::from_f32(131.25);
        let (effect, use_material, vertical, water_scale) =
            if !context.space && bounds.y && position.y < water {
                (
                    &bounds.down_effect,
                    bounds.use_down_material,
                    Some(water - Fix::from_f32(1.47) * actor.controller.physics.scale),
                    true,
                )
            } else if context.space && bounds.y && position.y < bottom {
                (
                    &bounds.up_effect,
                    bounds.use_up_material,
                    Some(bottom),
                    false,
                )
            } else if !bounds.no_up && bounds.y && position.y > top {
                (&bounds.up_effect, bounds.use_up_material, Some(top), false)
            } else if bounds.x && (position.x < left || position.x > right) {
                (&bounds.x_effect, bounds.use_x_material, None, false)
            } else {
                return Ok(());
            };
        if let Some(effect) = effect {
            let mut spawn_position = position;
            if vertical.is_none() {
                spawn_position.x = if position.x < left { left } else { right };
            }
            spawn_position += effect.offset;
            if let Some(y) = vertical {
                spawn_position.y = y;
            }
            let scale = effect.scale
                * if water_scale {
                    actor.controller.physics.scale.to_f64() as f32
                } else {
                    1.0
                };
            let effects = context
                .combat
                .as_mut()
                .and_then(|c| c.death_effects.as_mut())
                .ok_or("boundary_effect_not_bound")?;
            effects.spawn_kind(
                context.schedule,
                Spawn {
                    position: spawn_position,
                    actor: id,
                    scale,
                    material: effect.material,
                    kind: effect.kind,
                },
            )?;
            if use_material {
                let material = if let Some(ability) = &actor.ordinary {
                    ability.info.material
                } else {
                    if actor
                        .slime
                        .as_ref()
                        .is_none_or(|s| s.animator.render.is_none())
                    {
                        return Err("null_reference");
                    }
                    Some(id as u32)
                };
                effects.instances.last_mut().unwrap().material = material;
            }
        }
        if actor.ordinary.as_ref().is_some_and(|a| a.dash.is_some()) {
            Self::exit_dash(
                actors,
                context,
                id,
                crate::ability::ExitInfo {
                    dead: true,
                    killer: 1000,
                    position,
                    ..Default::default()
                },
                report,
            )
        } else if actor.collision.is_some() {
            Self::kill_actor(
                actors,
                context,
                id,
                &crate::actor_combat::Kill {
                    killer: 1000,
                    spawn_effect: false,
                    ignore_invulnerability: true,
                    cause: crate::player_life::Cause::Accident as i32,
                },
                report,
            )
            .map(|_| ())
        } else {
            let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
            if let Some(victim) = combat
                .book
                .players
                .iter()
                .position(|p| Some(p.id) == actor.body.holder)
                && let Some(event) = combat.book.kill(
                    victim,
                    1000,
                    context.schedule.ticks as i64,
                    crate::player_life::Cause::Accident as i32,
                )
            {
                combat.events.push(event);
            }
            Self::destroy_actor(actors, context, id)
        }
    }
}
