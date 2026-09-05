//! SlimeController's cooldown and cast gates. Actual ability execution is a
//! separate dispatch boundary: selecting an unbound handler must return an error.
use crate::fixed::Fix;
use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ability {
    pub present: bool,
    pub destroyed: bool,
    pub cooldown: Fix,
    pub stopped_during_time_stop: bool,
    pub ground_only: bool,
    pub castable: bool,
    pub ignore_next_cooldown: bool,
}
/// Headless indicators retain the reset clock. Sprite/material presentation is
/// not represented; a missing indicator itself still throws at the source call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Indicator {
    pub seconds_since_reset: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    /// Slots alias the pool, including null and destroyed Unity references.
    pub slots: Vec<Option<usize>>,
    pub pool: Vec<Ability>,
    pub timers: Vec<Fix>,
    pub indicators: Vec<Option<Indicator>>,
    pub in_ability: bool,
    pub can_use: bool,
    pub buttons: Vec<bool>,
}
impl State {
    fn ability(&self, slot: usize) -> Result<&Ability> {
        self.slots
            .get(slot)
            .ok_or("argument_out_of_range")?
            .and_then(|id| self.pool.get(id))
            .ok_or("null_reference")
    }
    fn timer(&mut self, slot: usize) -> Result<&mut Fix> {
        self.timers.get_mut(slot).ok_or("index_out_of_range")
    }
    fn indicator(&mut self, slot: usize) -> Result<&mut Indicator> {
        self.indicators
            .get_mut(slot)
            .ok_or("index_out_of_range")?
            .as_mut()
            .ok_or("null_reference")
    }
    fn button(&self, slot: usize) -> bool {
        slot < 3 && self.buttons.get(slot).copied().unwrap_or(false)
    }
    pub fn ready(&self, slot: usize) -> Result<bool> {
        let timer = self.timers.get(slot).ok_or("index_out_of_range")?;
        Ok(*timer > self.ability(slot)?.cooldown)
    }
    pub fn castable(&self, slot: usize, grounded: bool) -> Result<bool> {
        let ability = self.ability(slot)?;
        Ok((!ability.ground_only || grounded) && self.can_use && ability.castable)
    }
    /// This path handles ordinary Ability components. Rope implementations must
    /// supply their spatial query and dynamic cooldown update before integration.
    pub fn cooldowns(&mut self, dt: Fix, stopped: bool) -> Result<()> {
        if self.in_ability {
            return Ok(());
        }
        for i in 0..self.slots.len() {
            if self.slots[i].is_none() || !self.ability(i)?.present || self.ability(i)?.destroyed {
                continue;
            }
            let cooldown = self.ability(i)?.cooldown;
            let before = *self.timer(i)? >= cooldown;
            if !stopped || !self.ability(i)?.stopped_during_time_stop {
                *self.timer(i)? += dt;
            }
            let after = *self.timer(i)? >= cooldown;
            if !before && after && i < self.indicators.len() {
                self.indicator(i)?.seconds_since_reset = 0.0;
            }
            if i >= self.indicators.len() {
                continue;
            }
            if cooldown == Fix::ZERO {
                *self.timer(i)? = Fix::from_f64(0.01);
            }
            // UpdateOverriden(false) still dereferences a null indicator. The
            // headless adapter has no overridden sprite or renderer to update.
            self.indicator(i)?;
        }
        for i in 0..self.slots.len().min(self.indicators.len()) {
            if self.slots[i].is_some() && self.ability(i)?.present && !self.ability(i)?.destroyed {
                self.timer(i)?;
                self.indicator(i)?;
            }
        }
        Ok(())
    }
    /// Returns the first cast, before EnterAbility and before the movement FSM.
    /// The source selection loop does not skip null/destroyed slots.
    pub fn select(&mut self, grounded: bool, suppress: bool) -> Result<Option<usize>> {
        for i in 0..self.slots.len() {
            if self.ability(i)?.ignore_next_cooldown && !self.button(i) {
                *self.timer(i)? = self.ability(i)?.cooldown;
            }
            if self.button(i) && self.ready(i)? && self.castable(i, grounded)? && !suppress {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }
    /// The normal input path calls GetSpriteRen on every indicator, even when
    /// the corresponding ability is null. All renderers are absent here.
    pub fn headless_indicator_alpha(&mut self) -> Result<()> {
        for i in 0..self.slots.len().min(self.indicators.len()) {
            self.indicator(i)?;
        }
        Ok(())
    }
}
