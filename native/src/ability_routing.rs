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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handler: Option<crate::ability::Handler>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registration: Option<crate::clone_ability::Registration>,
}
/// Headless indicators retain the reset clock. Sprite/material presentation is
/// not represented; a missing indicator itself still throws at the source call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Indicator {
    pub seconds_since_reset: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_icon: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle: Option<IndicatorLifecycle>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndicatorLifecycle {
    pub active: bool,
    pub follow: bool,
}
impl Indicator {
    pub fn reactivate_and_sync(&mut self) -> Result<()> {
        let was_active = self.lifecycle.as_ref().is_some_and(|l| l.active);
        if let Some(l) = &mut self.lifecycle {
            l.active = true;
        }
        if !was_active && self.seconds_since_reset > 0.0 {
            self.seconds_since_reset = 11.0;
        }
        if self.lifecycle.as_ref().is_none_or(|l| !l.follow) {
            return Err("null_reference");
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    /// Slots alias the pool, including null and destroyed Unity references.
    pub slots: Vec<Option<usize>>,
    pub pool: Vec<Ability>,
    pub timers: Option<Vec<Fix>>,
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
    pub(crate) fn timer(&mut self, slot: usize) -> Result<&mut Fix> {
        self.timers
            .as_mut()
            .ok_or("null_reference")?
            .get_mut(slot)
            .ok_or("index_out_of_range")
    }
    pub(crate) fn indicator(&mut self, slot: usize) -> Result<&mut Indicator> {
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
        let timer = self
            .timers
            .as_ref()
            .ok_or("null_reference")?
            .get(slot)
            .ok_or("index_out_of_range")?;
        Ok(*timer > self.ability(slot)?.cooldown)
    }
    pub fn castable(&self, slot: usize, grounded: bool) -> Result<bool> {
        self.castable_for_player(slot, grounded, false)
    }
    pub fn castable_for_player(
        &self,
        slot: usize,
        grounded: bool,
        invisible: bool,
    ) -> Result<bool> {
        let ability = self.ability(slot)?;
        if (ability.ground_only && !grounded) || !self.can_use {
            return Ok(false);
        }
        Ok(match &ability.handler {
            Some(crate::ability::Handler::Invisibility(instant)) => {
                !(instant.gate_invisible && invisible)
            }
            Some(crate::ability::Handler::Instant(_)) => true,
            None | Some(crate::ability::Handler::Entity { .. }) => ability.castable,
        })
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
        self.select_for_player(grounded, suppress, false)
    }
    pub fn select_for_player(
        &mut self,
        grounded: bool,
        suppress: bool,
        invisible: bool,
    ) -> Result<Option<usize>> {
        for i in 0..self.slots.len() {
            if self.ability(i)?.ignore_next_cooldown && !self.button(i) {
                *self.timer(i)? = self.ability(i)?.cooldown;
            }
            if self.button(i)
                && self.ready(i)?
                && self.castable_for_player(i, grounded, invisible)?
                && !suppress
            {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }
    /// Restore every live invisibility cooldown, then reset the timer of the
    /// first minimum-age invisibility slot. Aliases and strict ties matter.
    pub fn invisibility_ended(&mut self) -> Result<()> {
        let mut youngest = None;
        let mut minimum = Fix::from_raw(i64::MAX);
        for i in 0..self.slots.len() {
            let ability = self.ability(i)?;
            let Some(crate::ability::Handler::Invisibility(instant)) = &ability.handler else {
                continue;
            };
            if !ability.present {
                return Err("missing_reference");
            }
            if !instant.component_present {
                continue;
            }
            let timer = *self
                .timers
                .as_ref()
                .ok_or("null_reference")?
                .get(i)
                .ok_or("index_out_of_range")?;
            if timer < minimum {
                minimum = timer;
                youngest = Some(i);
            }
            if ability.cooldown == Fix::from_int(10000) && instant.linked && !ability.destroyed {
                let cooldown = instant.restore_cooldown;
                self.pool[self.slots[i].unwrap()].cooldown = cooldown;
            }
        }
        if let Some(i) = youngest {
            *self.timer(i)? = Fix::ZERO;
        }
        Ok(())
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
