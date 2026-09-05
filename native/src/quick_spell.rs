//! QuickSpell's synchronous cast/exit animation callbacks and Ability handoff.
use crate::{
    ability::{ExitInfo, PlayerInfo},
    actors::{Actor, Context, Event as ActorEvent, Report, Runtime},
    animator::{Frame, Start, State},
    fixed::Fix,
    player_controller::Event,
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub animations: BTreeMap<String, Option<Vec<Frame>>>,
    pub animator: State,
    pub enter_offset: Vec2,
    pub exit_offset: Vec2,
    pub shockwave_offset: Vec2,
    pub audio_enter: Option<String>,
    pub audio_cast: Option<String>,
    pub spell: Option<String>,
    pub particle_prefab: bool,
    pub particle_present: bool,
    pub scale_particle_size: bool,
    pub info: PlayerInfo,
    pub player: Option<i32>,
}
fn begin(
    animator: &mut State,
    animations: &BTreeMap<String, Option<Vec<Frame>>>,
    name: &str,
    callback: u32,
) -> scheduler::Result<()> {
    let frames = animations.get(name).ok_or("animation_missing")?.clone();
    animator.start(
        Start::Begin {
            frames,
            next: None,
            callback: Some(callback),
            style: 3,
        },
        &[],
        &mut |_, _| Err("quick_spell_unexpected_callback"),
    )
}
fn sound(value: &Option<String>, id: usize, report: &mut Report) {
    if let Some(value) = value.as_ref().filter(|v| !v.is_empty()) {
        report.events.push(ActorEvent {
            actor: id,
            event: Event::Sound(value.clone()),
        });
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
        actor.sync_collider(context.scene)?;
        let motion = &mut actor.controller.physics.motion;
        motion.position = self.info.position + self.enter_offset;
        let ground = motion.ground.and_then(|i| platforms.get(i as usize));
        let rotation = motion.unground(ground.map(|p| &p.surface), true);
        report.events.push(ActorEvent {
            actor: id,
            event: Event::Detached(rotation),
        });
        report.events.push(ActorEvent {
            actor: actor
                .ordinary
                .as_ref()
                .and_then(|a| a.callback)
                .unwrap_or(id),
            event: Event::JumpSound(format!("slimeJump{}", self.info.player)),
        });
        motion.velocity = Vec2::ZERO;
        motion.external_velocity = Vec2::ZERO;
        begin(&mut self.animator, &self.animations, "castAir", 1)?;
        sound(
            &self.audio_enter,
            actor
                .ordinary
                .as_ref()
                .and_then(|a| a.callback)
                .unwrap_or(id),
            report,
        );
        Ok(())
    }
    fn cast(
        &mut self,
        animator: &mut State,
        id: usize,
        players: &[scheduler::Player],
        report: &mut Report,
    ) -> scheduler::Result<()> {
        if !players.iter().any(|p| Some(p.id) == self.player) {
            return Err("null_reference");
        }
        if self.spell.is_some() {
            return Err("quick_spell_prefab_not_implemented");
        }
        if !self.particle_prefab {
            return Err("instantiate_null");
        }
        self.particle_present = true;
        sound(&self.audio_cast, id, report);
        begin(animator, &self.animations, "exit", 2)
    }
}
impl Runtime {
    pub(crate) fn enter_quick_spell(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let Some(mut quick) = actor.ordinary.as_mut().and_then(|a| a.quick_spell.take()) else {
            return Ok(());
        };
        let result = quick.enter(actor, context, id, report);
        actor.ordinary.as_mut().unwrap().quick_spell = Some(quick);
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        result
    }
    pub fn exit_quick_spell(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        mut info: ExitInfo,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let quick = actor
            .ordinary
            .as_ref()
            .and_then(|a| a.quick_spell.as_ref())
            .ok_or("missing_quick_spell")?;
        let motion = &mut actor.controller.physics.motion;
        motion.rotation = Fix::ZERO;
        motion.right = Vec2::angle(Fix::ZERO);
        motion.up = motion.right.perp();
        let ground = motion
            .ground
            .and_then(|i| context.bindings[i as usize].platform.as_ref());
        let rotation = motion.unground(ground.map(|p| &p.surface), true);
        report.events.push(ActorEvent {
            actor: id,
            event: Event::Detached(rotation),
        });
        info.grounded_speed = motion.grounded_speed;
        info.grounded = motion.grounded;
        info.position = motion.position + quick.exit_offset;
        info.velocity = motion.velocity;
        info.external_velocity = motion.external_velocity;
        info.ground = motion.ground;
        info.local = motion.local;
        info.radius = motion.radius;
        info.sprite = None;
        actor.controller_to_body();
        actor.sync_transform(context.scene);
        Self::exit_ordinary(actors, context, id, &info, report)
    }
    pub fn cast_quick_spell(
        actors: &mut [Actor],
        context: &Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let quick = actors[id]
            .ordinary
            .as_mut()
            .and_then(|a| a.quick_spell.as_mut())
            .ok_or("missing_quick_spell")?;
        let mut animator = std::mem::replace(&mut quick.animator, State::idle(None));
        let result = quick.cast(&mut animator, id, &context.schedule.players, report);
        quick.animator = animator;
        result
    }
    pub(crate) fn animate_quick_spell(
        actors: &mut [Actor],
        context: &mut Context,
        id: usize,
        report: &mut Report,
    ) -> scheduler::Result<()> {
        let actor = &mut actors[id];
        let quick = actor
            .ordinary
            .as_mut()
            .and_then(|a| a.quick_spell.as_mut())
            .ok_or("missing_quick_spell")?;
        let protected = context
            .schedule
            .players
            .iter()
            .find(|p| Some(p.id) == actor.body.holder)
            .map(|p| p.protected);
        let mut animator = std::mem::replace(&mut quick.animator, State::idle(None));
        let result = animator.update(
            context.schedule.time_scale,
            protected,
            &[],
            &mut |animator, callback| match callback {
                1 => quick.cast(animator, id, &context.schedule.players, report),
                2 => Err("quick_spell_exit_requested"),
                _ => Err("quick_spell_unknown_callback"),
            },
        );
        quick.animator = animator;
        if result == Err("quick_spell_exit_requested") {
            return Self::exit_quick_spell(actors, context, id, ExitInfo::default(), report);
        }
        result
    }
}
