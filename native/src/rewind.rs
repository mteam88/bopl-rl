//! RewindRecorder's ordinary history writes. Playback is not yet integrated.
use crate::{actors::Actor, fixed::Fix, scheduler, vector::Vec2};
use serde::{Deserialize, Serialize};

pub const CAPACITY: usize = 188;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub position: Vec2,
    pub velocity: Vec2,
    pub external_velocity: Vec2,
    pub rotation: Fix,
    pub cooldowns: [Fix; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub ring: Vec<Sample>,
    pub head: usize,
    pub count: usize,
    pub rewinding: bool,
}

impl Actor {
    pub(crate) fn record_rewind(&mut self) -> scheduler::Result<()> {
        let recorder = self.rewind.as_mut().ok_or("missing_rewind_recorder")?;
        if recorder.rewinding {
            return Err("rewind_playback_not_implemented");
        }
        let Some(slime) = self.slime.as_ref() else {
            return Ok(());
        };
        if slime.managed_destroyed == Some(true)
            || self
                .entity
                .as_ref()
                .is_some_and(|e| e.destroyed == Some(true) || e.present == Some(false) || !e.active)
        {
            return Ok(());
        }
        let timers = &slime
            .abilities
            .as_ref()
            .ok_or("missing_ability_routing")?
            .timers
            .as_ref()
            .ok_or("null_reference")?;
        let sample = Sample {
            position: self.body.position,
            velocity: self.body.velocity,
            external_velocity: self.body.external_velocity,
            rotation: self.controller.physics.motion.rotation,
            cooldowns: [
                *timers.first().ok_or("index_out_of_range")?,
                *timers.get(1).ok_or("index_out_of_range")?,
                *timers.get(2).ok_or("index_out_of_range")?,
            ],
        };
        *recorder
            .ring
            .get_mut(recorder.head)
            .ok_or("index_out_of_range")? = sample;
        recorder.head = (recorder.head + 1) % CAPACITY;
        if recorder.count < CAPACITY {
            recorder.count += 1;
        }
        Ok(())
    }
}
