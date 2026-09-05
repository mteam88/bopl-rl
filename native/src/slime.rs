//! SlimeController locomotion, animation and ability routing. Executable handlers
//! are explicit; unsupported selected abilities return an error. Swallowing,
//! general ability handoff and the remaining lifecycle are integration work.
use crate::{
    animator::{self, Frame, Speed, Start, VelocityFrame},
    fixed::Fix,
    platform_motion::Binding,
    player_controller::{self, Event},
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
type Result<T> = animator::Result<T>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub mode: i32,
    pub facing_right: bool,
    pub was_upside_down: bool,
    pub input: Vec2,
    pub previous_input: Fix,
    pub suppress: bool,
    pub has_spawned: bool,
    pub time_moonwalking: Fix,
    /// The captured argument of EnterTurn's animation callback.
    pub turn_input: Fix,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ragdoll {
    pub duration: Fix,
    pub time: Fix,
    pub speed: Vec2,
    pub magnet_holder: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagdollImpulse {
    pub duration: Fix,
    pub speed: Vec2,
    pub fast: bool,
    pub reset_animation: bool,
    /// The pinned ApplyRagDoll accepts this argument but never assigns it.
    pub magnet: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub state: State,
    pub animator: animator::State,
    pub speed: Speed,
    pub animations: BTreeMap<String, Vec<Frame>>,
    pub fall_map: Vec<VelocityFrame>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ragdoll: Option<Ragdoll>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abilities: Option<crate::ability_routing::State>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<crate::ability::SlimeEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_destroyed: Option<bool>,
    #[serde(skip, default = "default_callback_gates")]
    pub(crate) callback_gates: (bool, bool),
}
fn default_callback_gates() -> (bool, bool) {
    (true, true)
}
pub struct Update<'a> {
    pub dt: Fix,
    pub input: Vec2,
    pub jump: bool,
    pub actor: usize,
    pub body: &'a mut crate::player_body::State,
    pub player: &'a mut crate::scheduler::Player,
    pub position_priorities: &'a [Option<i32>],
    pub entity: Option<&'a mut crate::actor_entity::Entity>,
    pub object: u32,
}
fn frames(data: &BTreeMap<String, Vec<Frame>>, name: &str) -> Result<Vec<Frame>> {
    data.get(name).cloned().ok_or("animation_missing")
}
fn start(animator: &mut animator::State, start: Start, speed: &Speed) -> Result<()> {
    animator.start(start, std::slice::from_ref(speed), &mut |_, _| {
        Err("unexpected_slime_entry_callback")
    })
}
fn enter(
    state: &mut State,
    animator: &mut animator::State,
    data: &BTreeMap<String, Vec<Frame>>,
    speed: &Speed,
    transition: (i32, Fix),
    id: i32,
    emit: &mut impl FnMut(Event),
) -> Result<()> {
    let (mode, input) = transition;
    state.mode = mode;
    let (name, callback, style) = match mode {
        0 => ("idle", None, 0),
        1 => {
            state.facing_right = input >= Fix::ZERO;
            ("run", None, 0)
        }
        6 => {
            state.facing_right = input >= Fix::ZERO;
            ("runTurnAround", Some(2), 3)
        }
        7 => ("stop", Some(1), 3),
        _ => return Err("invalid_slime_animation_transition"),
    };
    start(
        animator,
        Start::Begin {
            frames: Some(frames(data, name)?),
            next: None,
            callback,
            style,
        },
        speed,
    )?;
    if mode == 6 {
        state.turn_input = input;
    }
    if mode == 1 {
        emit(Event::Sound(format!("slimeWalk{id}")));
    }
    Ok(())
}
impl Instance {
    pub fn apply_ragdoll(
        &mut self,
        controller: &mut player_controller::State,
        request: &RagdollImpulse,
    ) -> Result<()> {
        if request.reset_animation {
            start(
                &mut self.animator,
                Start::Begin {
                    frames: Some(frames(
                        &self.animations,
                        if request.fast {
                            "ragdollFast"
                        } else {
                            "ragdollSlow"
                        },
                    )?),
                    next: None,
                    callback: None,
                    style: 0,
                },
                &self.speed,
            )?;
        }
        self.state.mode = 10;
        let ragdoll = self.ragdoll.get_or_insert_with(Ragdoll::default);
        ragdoll.speed = request.speed;
        ragdoll.speed.x /= Fix::from_int(3) / Fix::from_int(2);
        ragdoll.duration = request.duration;
        ragdoll.time = Fix::ZERO;
        let motion = &mut controller.physics.motion;
        motion.velocity = ragdoll.speed;
        let right = request.speed.normalized_safe();
        if right != Vec2::ZERO {
            motion.right = right;
            motion.up = right.perp();
            motion.rotation = right.normalized_angle();
        }
        // Applying ragdoll does not unground, clear external velocity, update
        // the animation speed container, or assign the supplied magnet.
        Ok(())
    }
    fn enter_fall(
        &mut self,
        controller: &mut player_controller::State,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        let anticipation = self
            .animator
            .render
            .as_ref()
            .ok_or("null_reference")?
            .sprite;
        self.enter_fall_sprite(controller, anticipation, context, emit)
    }
    pub(crate) fn enter_fall_sprite(
        &mut self,
        controller: &mut player_controller::State,
        anticipation: Option<u32>,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        self.align_air();
        let motion = &mut controller.physics.motion;
        motion.rotation = Fix::ZERO;
        motion.right = Vec2::angle(Fix::ZERO);
        motion.up = motion.right.perp();
        self.state.mode = 2;
        self.speed = Speed {
            velocity: motion.velocity,
            rotation: motion.rotation,
        };
        start(
            &mut self.animator,
            Start::Y {
                map: Some(self.fall_map.clone()),
                looping: Some(frames(&self.animations, "fallLeftLoop")?),
                anticipation,
                body: Some(0),
            },
            &self.speed,
        )?;
        let bindings = context.bindings;
        let id = context.holder.ok_or("null_reference")?;
        // EnterFall uses Updater.standardFixedDeltaTime, not this call's dt.
        controller.cast(crate::world::DT, context, &mut |controller, event| {
            self.physics_event(&mut controller.physics, &event, bindings, id, emit)?;
            emit(event);
            Ok(())
        })
    }
    /// Only an installed turn callback retains its captured argument.
    pub fn sync_callback(&mut self) {
        if self.animator.callback != Some(2) {
            self.state.turn_input = Fix::ZERO;
        }
    }
    pub(crate) fn enter(
        &mut self,
        mode: i32,
        input: Fix,
        id: i32,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        enter(
            &mut self.state,
            &mut self.animator,
            &self.animations,
            &self.speed,
            (mode, input),
            id,
            emit,
        )
    }
    pub fn animate(
        &mut self,
        scale: Option<Fix>,
        protected: Option<bool>,
        id: i32,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        let Self {
            state,
            animator,
            speed,
            animations,
            ..
        } = self;
        let mut callback = |animator: &mut animator::State, callback| match callback {
            1 => enter(state, animator, animations, speed, (0, Fix::ZERO), id, emit),
            2 => enter(
                state,
                animator,
                animations,
                speed,
                (1, state.turn_input),
                id,
                emit,
            ),
            3 => {
                state.mode = 0;
                Ok(())
            }
            _ => Err("unknown_slime_animation_callback"),
        };
        if let Some(scale) = scale {
            animator.update(scale, protected, std::slice::from_ref(speed), &mut callback)
        } else {
            animator.animate(std::slice::from_ref(speed), &mut callback)
        }
    }
    pub fn physics_event(
        &mut self,
        physics: &mut crate::player_physics::State,
        event: &Event,
        bindings: &[Binding],
        id: i32,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        let m = &physics.motion;
        match event {
            Event::Detached(_) => {
                self.state.mode = 2;
                self.animator
                    .render
                    .as_mut()
                    .ok_or("null_reference")?
                    .flip_x = true;
                self.speed = Speed {
                    velocity: m.velocity,
                    rotation: m.rotation,
                };
                if !self.callback_gates.0 {
                    return Ok(());
                }
                let looping = frames(&self.animations, "fallLeftLoop")?;
                let anticipation = self
                    .animator
                    .render
                    .as_ref()
                    .ok_or("null_reference")?
                    .sprite;
                start(
                    &mut self.animator,
                    Start::Y {
                        map: Some(self.fall_map.clone()),
                        looping: Some(looping),
                        anticipation,
                        body: Some(0),
                    },
                    &self.speed,
                )?;
            }
            Event::Attached(_) => {
                if !self.callback_gates.0 || !self.callback_gates.1 {
                    return Ok(());
                }
                self.land(
                    m.ground
                        .and_then(|i| bindings.get(i as usize))
                        .and_then(|b| b.platform.as_ref()),
                    m.position,
                    id,
                    emit,
                )?;
            }
            _ => {}
        }
        Ok(())
    }
    pub(crate) fn land(
        &mut self,
        ground: Option<&crate::player_physics::Platform>,
        position: Vec2,
        id: i32,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        self.state.mode = 4;
        start(
            &mut self.animator,
            Start::Begin {
                frames: Some(frames(&self.animations, "land")?),
                next: Some(frames(&self.animations, "idle")?),
                callback: Some(3),
                style: 1,
            },
            &self.speed,
        )?;
        emit(Event::Sound(format!("slimeLand{id}")));
        if ground.ok_or("null_reference")?.surface.normal(position).y < Fix::ZERO {
            self.state.facing_right = !self.state.facing_right;
        }
        Ok(())
    }
    fn align_air(&mut self) {
        if self.state.was_upside_down {
            self.state.facing_right = !self.state.facing_right;
            self.state.was_upside_down = false;
        }
        if self.state.input.x < -Fix::from_f32(0.02) {
            self.state.facing_right = false;
        } else if self.state.input.x > Fix::from_f32(0.02) {
            self.state.facing_right = true;
        }
    }
    fn old_update(
        &mut self,
        controller: &mut player_controller::State,
        update: &mut Update,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<Option<crate::slime_ability::Request>> {
        let id = context.holder.ok_or("null_reference")?;
        let (dt, input, jump) = (update.dt, update.input, update.jump);
        if let Some(position) = &mut update.player.position {
            position.set(
                controller.physics.motion.position,
                Some(update.actor),
                update.position_priorities,
            );
        }
        if self.state.mode != 8
            && let Some(abilities) = &mut self.abilities
        {
            abilities.cooldowns(context.dt(dt), context.time_scale < Fix::from_f32(0.01))?;
        }
        if self.state.mode == 8 || self.state.mode == 9 || self.state.mode == 10 {
            return Ok(None);
        }
        self.state.input = input;
        let mut alignment = controller.input_alignment(input.normalized_safe(), context)?;
        let m = &controller.physics.motion;
        let normal = if m.grounded {
            context.bindings[m.ground.ok_or("null_reference")? as usize]
                .platform
                .as_ref()
                .ok_or("null_reference")?
                .surface
                .normal(m.position)
        } else {
            Vec2::UP
        };
        let outward = normal.dot(input);
        let mut moving = input.length() >= Fix::from_f32(0.4)
            && (alignment.abs() >= context.config.alignment_deadzone
                || (outward < Fix::ZERO && alignment.abs() >= Fix::from_f32(0.01)));
        let mut jump = jump;
        if self.state.suppress {
            moving = false;
            jump = false;
            self.state.input = Vec2::ZERO;
            alignment = Fix::ZERO;
        }
        let selected = self
            .abilities
            .as_mut()
            .map(|abilities| {
                abilities.select_for_player(
                    controller.physics.motion.grounded,
                    self.state.suppress,
                    update.player.invisible,
                )
            })
            .transpose()?
            .flatten();
        if let Some(slot) = selected {
            return self.enter_ability(slot, controller, update, context, emit);
        }
        match self.state.mode {
            4 => {
                if alignment.abs() >= context.config.alignment_deadzone || outward < Fix::ZERO {
                    self.state.facing_right = alignment >= Fix::ZERO;
                }
            }
            7 if moving && self.state.facing_right != (alignment >= Fix::ZERO) => {
                self.enter(6, alignment, id, emit)?
            }
            6 if !moving => self.enter(7, alignment, id, emit)?,
            0 if moving => self.enter(1, alignment, id, emit)?,
            1 if !moving || (alignment * self.state.previous_input).sign() < 0 => {
                self.enter(7, alignment, id, emit)?
            }
            _ => {}
        }
        if controller.physics.motion.grounded && jump {
            self.state.was_upside_down = normal.y < Fix::ZERO;
            let platforms: Vec<_> = context
                .bindings
                .iter()
                .map(|b| b.platform.clone().expect("slime platform binding"))
                .collect();
            let angle = controller
                .physics
                .jump(&platforms, &context.config.physics)?;
            let event = Event::Detached(angle);
            self.physics_event(&mut controller.physics, &event, context.bindings, id, emit)?;
            emit(event);
            emit(Event::JumpSound(format!("slimeJump{id}")));
            // Jump advances animations immediately, even during time stop.
            self.animate(None, None, id, emit)?;
        }
        if let Some(abilities) = &mut self.abilities {
            abilities.headless_indicator_alpha()?;
        }
        self.state.previous_input = alignment;
        if !controller.physics.motion.grounded {
            self.align_air();
        }
        Ok(None)
    }
    pub(crate) fn begin_update(
        &mut self,
        controller: &mut player_controller::State,
        update: &mut Update,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<Option<crate::slime_ability::Request>> {
        let id = context.holder.ok_or("null_reference")?;
        let dt = update.dt;
        if self.state.mode == 10 {
            let ragdoll = self.ragdoll.as_mut().ok_or("missing_ragdoll_state")?;
            ragdoll.time += context.dt(dt);
            if ragdoll.time > ragdoll.duration {
                ragdoll.magnet_holder = None;
                controller.physics.motion.external_velocity += controller.physics.motion.velocity;
                controller.physics.motion.velocity = Vec2::ZERO;
                self.enter_fall(controller, context, emit)?;
            }
        }
        let motion = &mut controller.physics.motion;
        let ground = motion
            .ground
            .and_then(|i| context.bindings.get(i as usize))
            .and_then(|b| b.platform.as_ref());
        if motion.grounded && ground.is_none_or(|p| p.destroyed) {
            motion.rotation = Fix::ZERO;
            motion.right = Vec2::angle(Fix::ZERO);
            motion.up = motion.right.perp();
            let angle = motion.unground(ground.map(|p| &p.surface), false);
            let event = Event::Detached(angle);
            self.physics_event(&mut controller.physics, &event, context.bindings, id, emit)?;
            emit(event);
            emit(Event::JumpSound(format!("slimeJump{id}")));
        }
        self.old_update(controller, update, context, emit)
    }
    pub(crate) fn finish_update(
        &mut self,
        controller: &mut player_controller::State,
        update: &mut Update,
        context: &mut player_controller::Context,
        emit: &mut impl FnMut(Event),
    ) -> Result<()> {
        let id = context.holder.ok_or("null_reference")?;
        let dt = update.dt;
        self.finish_invisibility(update.player.invisible)?;
        if let Some(entity) = &mut update.entity {
            entity.collider.set_scale(
                update.player.scale.unwrap_or(Fix::ONE),
                &mut controller.physics.scale,
                update.object,
                context.scene,
            )?;
        }
        if matches!(self.state.mode, 8..=10) || !self.state.has_spawned {
            return Ok(());
        }
        let bindings = context.bindings;
        controller.move_player(
            self.state.input,
            dt,
            None,
            true,
            context,
            &mut |controller, event| {
                self.physics_event(&mut controller.physics, &event, bindings, id, emit)?;
                emit(event);
                Ok(())
            },
        )?;
        if self.state.mode == 1 {
            let positive = controller.physics.motion.grounded_speed > Fix::ZERO;
            if positive != self.state.facing_right {
                self.state.time_moonwalking += dt;
            } else {
                self.state.time_moonwalking = Fix::ZERO;
            }
        }
        self.speed = Speed {
            velocity: controller.physics.motion.velocity,
            rotation: controller.physics.motion.rotation,
        };
        Ok(())
    }
}
