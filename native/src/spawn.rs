//! SlimeController spawn/revival transitions, composed with the actual animator
//! and PlayerPhysics detachment order. Object creation and stocks are upstream.
use crate::{
    animator::{self, Frame, Speed, Start, VelocityFrame},
    fixed::Fix,
    player_ground,
    surface::Surface,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
type Result<T> = animator::Result<T>;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    pub id: i32,
    pub protected: bool,
    pub extra_life: i32,
    pub spark: bool,
    pub macho: bool,
    pub platform: bool,
    pub revive: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Animations {
    pub spawn: Option<Vec<Frame>>,
    pub fall_loop: Option<Vec<Frame>>,
    pub fall_map: Option<Vec<VelocityFrame>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub motion: player_ground::State,
    pub physics_enabled: bool,
    pub simulated: bool,
    pub hurtbox_enabled: bool,
    pub renderer_enabled: bool,
    pub mode: i32,
    pub suppress: bool,
    pub has_spawned: bool,
    pub fast: bool,
    pub invulnerable: u32,
    pub next_ability: i32,
    pub abilityless: bool,
    /// References preserve repeated ability/indicator identities.
    pub abilities: Option<Vec<Option<usize>>>,
    pub ability_pool: Vec<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ability_present: Option<Vec<bool>>,
    pub cooldowns: Option<Vec<Fix>>,
    pub indicator_refs: Option<Vec<Option<usize>>>,
    pub indicators: Vec<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callback: Option<u32>,
    pub player: i32,
    pub holder: Option<i32>,
    pub jump_strength: Fix,
    pub tutorial: bool,
    pub animations: Option<Animations>,
    pub animation_speed: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Event {
    Detached(Fix),
    JumpSound(String),
}
pub struct Context<'a> {
    pub animator: &'a mut animator::State,
    pub speeds: &'a mut [Speed],
    pub players: &'a mut [Player],
    pub surfaces: &'a [Surface],
    pub config: &'a Config,
    pub events: &'a mut Vec<Event>,
    pub on_detach: &'a mut dyn FnMut(&mut State, Fix) -> Result<()>,
}
impl Context<'_> {
    fn player(&mut self) -> Result<&mut Player> {
        self.players
            .iter_mut()
            .find(|p| p.id == self.config.player)
            .ok_or("null_reference")
    }
    fn animations(&self) -> Result<&Animations> {
        self.config.animations.as_ref().ok_or("null_reference")
    }
    fn start(&mut self, start: Start) -> Result<()> {
        // These entry methods cannot immediately invoke a registered callback:
        // Begin installs it after SetFirstFrame, Y clears it before selection.
        self.animator.start(start, self.speeds, &mut |_, _| {
            unreachable!("spawn entry callback")
        })
    }
}
impl State {
    fn ability_count(&self) -> Result<usize> {
        Ok(self.abilities.as_ref().ok_or("null_reference")?.len())
    }
    fn indicators(&mut self, active: bool) -> Result<()> {
        for i in 0..self.ability_count()? {
            let index = self
                .indicator_refs
                .as_ref()
                .ok_or("null_reference")?
                .get(i)
                .ok_or("index_out_of_range")?
                .ok_or("null_reference")?;
            *self.indicators.get_mut(index).ok_or("null_reference")? = active;
        }
        Ok(())
    }
    pub(crate) fn unground(&mut self, c: &mut Context<'_>) -> Result<()> {
        let surface = self.motion.ground.and_then(|i| c.surfaces.get(i as usize));
        let angle = self.motion.unground(surface, true);
        // SlimeController.OnNoLongerGrounded runs before other test listeners and sound.
        self.mode = 2;
        c.animator.render.as_mut().ok_or("null_reference")?.flip_x = true;
        *c.speeds
            .get_mut(c.config.animation_speed)
            .ok_or("null_reference")? = Speed {
            velocity: self.motion.velocity,
            rotation: self.motion.rotation,
        };
        if self.simulated {
            let data = c.animations()?;
            let map = data.fall_map.clone().ok_or("velocity_animation_missing")?;
            let looping = data.fall_loop.clone().ok_or("animation_missing")?;
            let anticipation = c.animator.render.as_ref().ok_or("null_reference")?.sprite;
            c.start(Start::Y {
                map: Some(map),
                looping: Some(looping),
                anticipation,
                body: Some(c.config.animation_speed),
            })?;
        }
        c.events.push(Event::Detached(angle));
        (c.on_detach)(self, angle)?;
        let id = c.config.holder.ok_or("null_reference")?;
        c.events.push(Event::JumpSound(format!("slimeJump{id}")));
        Ok(())
    }
    pub fn init(&mut self, c: &mut Context<'_>) -> Result<()> {
        self.next_ability = 0;
        self.cooldowns = Some(vec![Fix::from_f32(-1.5); 3]);
        if !self.abilityless {
            self.before(c)?;
        }
        Ok(())
    }
    pub fn before(&mut self, c: &mut Context<'_>) -> Result<()> {
        self.physics_enabled = false;
        self.suppress = true;
        self.simulated = false;
        self.motion.velocity = Vec2::ZERO;
        self.motion.external_velocity = Vec2::ZERO;
        if self.mode != 8 {
            self.hurtbox_enabled = true;
            let mut frames = c.animations()?.spawn.clone().ok_or("animation_missing")?;
            if self.fast {
                frames = animator::speed_up(&frames, 4)?;
            }
            c.start(Start::Begin {
                frames: Some(frames),
                next: None,
                callback: Some(c.config.callback.unwrap_or(1)),
                style: 3,
            })?;
        }
        if c.config.tutorial {
            self.hurtbox_enabled = false;
        }
        Ok(())
    }
    pub fn spawn(&mut self, c: &mut Context<'_>) -> Result<()> {
        self.physics_enabled = true;
        self.simulated = true;
        self.hurtbox_enabled = true;
        self.unground(c)?;
        self.motion.velocity = Vec2::UP * c.config.jump_strength * Fix::from_f32(0.25);
        self.suppress = false;
        self.has_spawned = true;
        self.invulnerable &= !4;
        let p = c.player()?;
        p.extra_life = 0;
        p.spark = false;
        p.macho = false;
        p.platform = false;
        self.indicators(true)
    }
    pub fn respawn(&mut self, position: Vec2, c: &mut Context<'_>) -> Result<()> {
        self.hurtbox_enabled = false;
        self.simulated = true;
        c.animator.render.as_ref().ok_or("null_reference")?;
        self.renderer_enabled = true;
        self.motion.external_velocity = Vec2::ZERO;
        self.motion.velocity = Vec2::ZERO;
        self.motion.position = position;
        self.mode = 2;
        c.player()?.extra_life = 0;
        self.invulnerable &= !4;
        self.before(c)
    }
    pub fn fast_respawn(&mut self, position: Vec2, c: &mut Context<'_>) -> Result<()> {
        self.fast = true;
        self.respawn(position, c)?;
        self.fast = false;
        Ok(())
    }
    pub fn prepare(&mut self, position: Vec2, c: &mut Context<'_>) -> Result<()> {
        c.player()?.extra_life = 2;
        self.hurtbox_enabled = false;
        self.invulnerable |= 4;
        self.unground(c)?;
        self.mode = 8;
        self.motion.rotation = Fix::ZERO;
        self.motion.right = Vec2::new(Fix::ZERO.cos(), Fix::ZERO.sin());
        self.motion.up = -self.motion.right.perp_cw();
        self.motion.external_velocity = Vec2::ZERO;
        self.motion.velocity = Vec2::ZERO;
        self.motion.position = position;
        self.has_spawned = false;
        self.physics_enabled = false;
        self.simulated = false;
        self.suppress = true;
        c.animator.render.as_ref().ok_or("null_reference")?;
        self.renderer_enabled = false;
        for i in 0..self.ability_count()? {
            let id = self.abilities.as_ref().unwrap()[i].ok_or("null_reference")?;
            if self
                .ability_present
                .as_ref()
                .is_some_and(|p| p.get(id) == Some(&false))
            {
                return Err("null_reference");
            }
            if *self.ability_pool.get(id).ok_or("null_reference")? && c.player()?.revive == Some(id)
            {
                *self
                    .cooldowns
                    .as_mut()
                    .ok_or("null_reference")?
                    .get_mut(i)
                    .ok_or("index_out_of_range")? = Fix::ZERO;
                break;
            }
        }
        self.indicators(false)
    }
    pub fn pop(&mut self, direction: Vec2, jump: Fix) {
        self.motion.velocity = direction * Fix::from_f64(1.3) * jump;
    }
}
