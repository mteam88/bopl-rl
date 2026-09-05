//! Concrete Dash component, invoked by Ability entry and the shared scheduler.
use crate::{
    ability::{ExitInfo, PlayerInfo},
    actors::{Actor, Context, Event as ActorEvent, Report, Runtime},
    fixed::Fix,
    game_time,
    physics::query::OrderedHit,
    player_controller::Event,
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub pause: Fix,
    pub length: Fix,
    pub leave_speed: Fix,
    pub fixed_updates: i32,
    pub mask: u32,
    pub anticipation: Option<u32>,
    pub input: Vec2,
    pub info: PlayerInfo,
    pub player: Option<i32>,
    pub time: Fix,
    pub updates: i32,
    pub in_dash: bool,
    pub hits: Vec<OrderedHit>,
    pub hit_ground: Option<u32>,
    pub hit_local: Fix,
    pub sprite_enabled: bool,
    pub smear_enabled: bool,
    pub sprite: Option<u32>,
}
fn aim(context: &Context, player: i32) -> scheduler::Result<Vec2> {
    let input = context
        .schedule
        .players
        .iter()
        .find(|p| p.id == player)
        .and_then(|p| p.input)
        .and_then(|i| context.schedule.inputs.get(i))
        .ok_or("null_reference")?;
    Ok(crate::world::Action {
        angle: input.angle.unwrap_or(0),
        buttons: 0,
    }
    .direction())
}
fn water(space: bool) -> Fix {
    if space {
        Fix::from_int(-50)
    } else {
        Fix::from_f32(-11.3)
    }
}
impl Instance {
    fn enter(
        &mut self,
        actor: &mut Actor,
        context: &mut Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        self.hits.clear();
        self.sprite_enabled = true;
        self.smear_enabled = false;
        self.updates = 0;
        self.info = actor
            .ordinary
            .as_ref()
            .ok_or("missing_ordinary_ability")?
            .info
            .clone();
        self.player = context
            .schedule
            .players
            .iter()
            .find(|p| p.id == self.info.player)
            .map(|p| p.id);
        let platforms: Vec<_> = context
            .bindings
            .iter()
            .map(|b| b.platform.clone().expect("actor platform binding"))
            .collect();
        actor
            .controller
            .physics
            .sync(&self.info.physics(), false, &platforms)?;
        actor.sync_transform(context.scene);
        actor.sync_box(context.scene)?;
        let motion = &mut actor.controller.physics.motion;
        motion.rotation = Fix::ZERO;
        motion.right = Vec2::angle(Fix::ZERO);
        motion.up = motion.right.perp();
        self.sprite = self.anticipation;
        self.in_dash = false;
        self.time = Fix::ZERO;
        actor.body.rope = self.info.rope;
        actor.body.top = self.info.top;
        let ground = motion.ground.and_then(|i| platforms.get(i as usize));
        let rotation = motion.unground(ground.map(|p| &p.surface), true);
        report.events.push(ActorEvent {
            actor: id,
            event: Event::Detached(rotation),
        });
        self.input = aim(context, self.player.ok_or("null_reference")?)?;
        motion.velocity = Vec2::ZERO;
        motion.external_velocity = Vec2::ZERO;
        report.events.push(ActorEvent {
            actor: actor
                .ordinary
                .as_ref()
                .and_then(|a| a.callback)
                .unwrap_or(id),
            event: Event::Sound("dash".into()),
        });
        Ok(())
    }
    fn perform(&mut self, actor: &mut Actor, context: &mut Context) -> scheduler::Result<()> {
        if self.input.length() < Fix::from_f32(0.4) {
            self.input = Vec2::UP;
        }
        self.input = self.input.normalized();
        let angle = self
            .input
            .normalized()
            .dot(Vec2::UP.normalized())
            .acos()
            .map_err(|_| "arithmetic")?;
        let motion = &mut actor.controller.physics.motion;
        motion.rotation = if self.input.x > Fix::ZERO {
            Fix::TAU - angle
        } else {
            angle
        };
        motion.right = Vec2::angle(motion.rotation);
        motion.up = motion.right.perp();
        self.updates = 0;
        motion.velocity = Vec2::ZERO;
        motion.external_velocity = Vec2::ZERO;
        self.smear_enabled = true;
        self.sprite_enabled = false;
        context.scene.raycast_all(
            motion.position,
            self.input,
            self.length,
            self.mask,
            actor.object,
            &mut self.hits,
        )?;
        // With no PlayerCollision bound, the first non-wall hit throws; a wall
        // also returns immediately in the original component.
        if let Some(ordered) = self.hits.first() {
            let hit = &ordered.hit;
            if hit.layer == 11 {
                self.hit_ground = context
                    .bindings
                    .iter()
                    .position(|b| Some(b.object) == hit.object)
                    .map(|i| i as u32);
                let platform = self
                    .hit_ground
                    .and_then(|i| context.bindings[i as usize].platform.as_ref())
                    .ok_or("null_reference")?;
                let closest = platform
                    .surface
                    .closest_point(hit.near_position)
                    .map_err(|_| "division_by_zero")?;
                self.hit_local = platform
                    .surface
                    .local_position(closest, motion.radius)
                    .map_err(|_| "division_by_zero")?;
                return Ok(());
            }
            let target = context
                .scene
                .objects
                .iter()
                .find(|o| Some(o.id) == hit.object)
                .ok_or("null_reference")?;
            if !target.callback {
                return Err("null_reference");
            }
            // The current composed graph binds recording-only target callbacks
            // and no PlayerCollision on the Dash body. Retain the source prefix
            // before its KillPlayerOnCollision dereference.
            motion.position = hit.near_position;
            return Err("null_reference");
        }
        self.hit_ground = None;
        Ok(())
    }
    fn exit(
        &mut self,
        actor: &mut Actor,
        context: &mut Context,
        info: ExitInfo,
    ) -> scheduler::Result<ExitInfo> {
        let physics = &actor.controller.physics;
        if let Some(index) = actor.body.rope {
            let rope = context
                .scene
                .rope_system
                .as_mut()
                .ok_or("actor_rope_binding_invalid")?
                .bodies
                .get_mut(index)
                .ok_or("actor_rope_binding_invalid")?;
            let force = self.input * self.leave_speed * Fix::from_int(10);
            if actor.body.top {
                rope.force_start = force;
                *rope.segment.get_mut(0).ok_or("index_out_of_range")? = physics.motion.position;
            } else {
                rope.force_end = force;
                let first = *rope.segment.first().ok_or("index_out_of_range")?;
                let delta = physics.motion.position - first;
                let count = Fix::from_int(rope.count as i64);
                let step = Vec2::new(
                    delta.x.checked_div(count).map_err(|_| "division_by_zero")?,
                    delta.y.checked_div(count).map_err(|_| "division_by_zero")?,
                );
                for i in 1..rope.count {
                    *rope.segment.get_mut(i).ok_or("index_out_of_range")? =
                        first + step * Fix::from_int(i as i64);
                }
                for i in 0..rope.count {
                    *rope.previous.get_mut(i).ok_or("index_out_of_range")? =
                        *rope.segment.get(i).ok_or("index_out_of_range")?;
                }
            }
        }
        self.smear_enabled = false;
        self.sprite_enabled = true;
        let motion = &physics.motion;
        Ok(ExitInfo {
            grounded_speed: Fix::ZERO,
            facing_right: self.input.dot(Vec2::RIGHT) <= Fix::ZERO,
            velocity: self.input
                * self.leave_speed
                * if context.space {
                    Fix::from_f32(0.5)
                } else {
                    Fix::ONE
                },
            external_velocity: Vec2::ZERO,
            grounded: false,
            ground: motion.ground,
            position: motion.position,
            radius: motion.radius,
            local: motion.local,
            top: actor.body.top,
            rope: actor.body.rope,
            ..info
        })
    }
    fn update(
        &mut self,
        actor: &mut Actor,
        context: &mut Context,
        dt: Fix,
    ) -> scheduler::Result<Option<ExitInfo>> {
        let player = self.player.ok_or("null_reference")?;
        let protected = context
            .schedule
            .players
            .iter()
            .find(|p| p.id == player)
            .map(|p| p.protected);
        self.time += game_time::fixed_delta_time(dt, context.schedule.time_scale, protected);
        if self.time < self.pause {
            let value = aim(context, player)?;
            if value.length() > Fix::from_f32(0.4) {
                self.input = value;
            }
        } else if !self.in_dash {
            self.in_dash = true;
            self.perform(actor, context)?;
        } else {
            if self.updates > self.fixed_updates {
                self.smear_enabled = false;
                self.sprite_enabled = true;
                let physics = &mut actor.controller.physics;
                if let Some(index) = self.hit_ground.filter(|&i| {
                    context.bindings[i as usize]
                        .platform
                        .as_ref()
                        .is_some_and(|p| !p.destroyed)
                }) {
                    let platform = context.bindings[index as usize]
                        .platform
                        .as_ref()
                        .ok_or("null_reference")?;
                    physics.motion.position =
                        platform.position(self.hit_local, physics.motion.radius)?;
                    physics.attach_platform(
                        Some(index),
                        Some(platform),
                        physics.motion.position,
                    )?;
                } else {
                    let before = physics.motion.position;
                    let mut position = before + self.input * self.length;
                    if position.y < water(context.space) {
                        let distance = (water(context.space) - before.y)
                            .checked_div(self.input.y)
                            .map_err(|_| "division_by_zero")?;
                        position.y = water(context.space);
                        position.x = before.x + self.input.x * distance;
                    }
                    physics.motion.position = position;
                }
                return self.exit(actor, context, ExitInfo::default()).map(Some);
            }
            self.updates = self.updates.wrapping_add(1);
        }
        Ok(None)
    }
}
impl Runtime {
    pub(crate) fn exit_dash(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        info: ExitInfo,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let mut dash = actor
            .ordinary
            .as_mut()
            .and_then(|a| a.dash.take())
            .ok_or("missing_dash_component")?;
        let result = dash.exit(actor, context, info);
        actor.ordinary.as_mut().unwrap().dash = Some(dash);
        actor.controller_to_body();
        Self::exit_ordinary(actors, context, id, &result?, report)
    }

    pub(crate) fn enter_dash(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let Some(mut dash) = actor.ordinary.as_mut().and_then(|a| a.dash.take()) else {
            return Ok(());
        };
        let result = dash.enter(actor, context, id, report);
        actor.ordinary.as_mut().unwrap().dash = Some(dash);
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        result
    }
    pub(crate) fn update_dash(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        dt: Fix,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let mut dash = actor
            .ordinary
            .as_mut()
            .and_then(|a| a.dash.take())
            .ok_or("missing_dash_component")?;
        let result = dash.update(actor, context, dt);
        actor.ordinary.as_mut().unwrap().dash = Some(dash);
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        if let Some(info) = result? {
            Self::exit_ordinary(actors, context, id, &info, report)?;
            let dash = actors[id].ordinary.as_mut().unwrap().dash.as_mut().unwrap();
            dash.updates = dash.updates.wrapping_add(1);
        }
        Ok(())
    }
}
