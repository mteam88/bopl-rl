//! Scene contact delivery into original PlayerCollision lethality and body death.
use crate::{
    actors::{Actor, Context, Report, Runtime},
    physics::scene::CollisionEvent,
    player_collision as policy, scheduler,
    vector::Vec2,
};
impl Runtime {
    pub(crate) fn collide_actor(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        event: &CollisionEvent,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &actors[id];
        let Some(collision) = &actor.collision else {
            return Ok(());
        };
        let combat = context.combat.as_ref().ok_or("missing_actor_combat")?;
        let motion = &actor.controller.physics.motion;
        let ground = motion.ground.map(|i| &context.bindings[i as usize]);
        let mut receiver = policy::Receiver {
            player_id: actor.body.holder.ok_or("null_reference")?,
            player_alive: combat
                .book
                .players
                .iter()
                .find(|p| Some(p.id) == actor.body.holder)
                .map(|p| p.bodies_alive > 0),
            invulnerable: collision.invulnerable,
            was_killed: collision.was_killed,
            position: actor.body.position,
            right: motion.right,
            squish_offset: Vec2::ZERO,
            squishable: true,
            physics: Some(policy::Ground {
                grounded: motion.grounded,
                attached: ground.map(|g| policy::Attached {
                    body: g.object,
                    destroyed: g.platform.as_ref().is_none_or(|p| p.destroyed),
                }),
            }),
            ignored: std::array::from_fn(|_| None),
            ignored_beam: -1,
        };
        let object = context
            .scene
            .objects
            .iter()
            .find(|o| o.id == event.other)
            .ok_or("missing_collision_object")?;
        let other_actor = actors.iter().find(|a| a.object == event.other);
        let ability = other_actor
            .and_then(|a| a.ordinary.as_ref())
            .map(|a| policy::Ability {
                owner: a.info.player,
                kills_on_contact: a.kill_players_on_contact == Some(true),
                rock: None,
            });
        let platform = context
            .bindings
            .iter()
            .find(|b| b.object == event.other)
            .map(|b| policy::Platform {
                body: b.object,
                pickup: false,
                engine: None,
            });
        let hitbox = combat
            .hitboxes
            .as_ref()
            .into_iter()
            .flatten()
            .find(|h| h.object == event.other)
            .map(|h| h.owner);
        let mut hit = policy::Hit {
            id: event.other,
            layer: object.layer as i32,
            collision_layer: event.layer as i32,
            hierarchy: other_actor.map_or(object.hierarchy.unwrap_or(0), |a| a.collision_hierarchy),
            transform_present: true,
            destroyed: object.destroyed,
            collider: event.collider,
            platform,
            orb_kind: None,
            explosion: None,
            hitbox_owner: hitbox,
            projectile: None,
            holder_owner: other_actor.and_then(|a| a.body.holder).or(hitbox),
            ability,
            parent_ability: None,
            chain_lightning: false,
        };
        let mut globals = policy::Globals {
            players: combat.book.players.iter().map(|p| p.id).collect(),
            local_players: vec![],
            replay: false,
            engine_achievement: false,
            arrow_achievement: false,
            rock_achievement: false,
        };
        let mut effects = Vec::new();
        let result = receiver.collide(&mut hit, &mut globals, true, &mut |effect| {
            effects.push(effect);
            true
        });
        actors[id].collision.as_mut().unwrap().was_killed = receiver.was_killed;
        for effect in effects {
            match effect {
                policy::Effect::Kill { killer, cause } => {
                    Self::kill_actor(
                        actors,
                        context,
                        id,
                        &crate::actor_combat::Kill {
                            killer,
                            cause,
                            spawn_effect: true,
                            ignore_invulnerability: false,
                        },
                        report,
                    )?;
                }
                policy::Effect::Attach { target, position } => {
                    let index = context
                        .bindings
                        .iter()
                        .position(|b| b.object == target)
                        .ok_or("null_reference")?;
                    let actor = &mut actors[id];
                    actor.controller.physics.attach_platform(
                        Some(index as u32),
                        context.bindings[index].platform.as_ref(),
                        position,
                    )?;
                    actor.controller_to_body();
                    actor.sync_transform(context.scene);
                }
                policy::Effect::Lightning => report.events.push(crate::actors::Event {
                    actor: id,
                    event: crate::player_controller::Event::Sound("lightning".into()),
                }),
            }
        }
        result.map(|_| ())
    }
}
